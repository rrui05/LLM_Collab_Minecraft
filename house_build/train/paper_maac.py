from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from comlrl.trainers.actor_critic import MAACConfig as BaseMAACConfig  # type: ignore
from comlrl.trainers.actor_critic import MAACTrainer as BaseMAACTrainer  # type: ignore
from comlrl.trainers.actor_critic.iac import RolloutSample  # type: ignore
from comlrl.trainers.actor_critic.maac import unwrap_model  # type: ignore
from comlrl.utils.reward_utils import call_reward_function, normalize_reward_lengths  # type: ignore


class PaperAlignedMAACConfig(BaseMAACConfig):
    """MAACConfig variant that permits paper-style multi-turn grouped rollouts."""

    def __post_init__(self) -> None:
        # CoLLM-CC Eq. 4 uses the TD error directly; the paper only clips
        # advantages and does not normalize them across a minibatch.
        self.advantage_normalization = False
        original_num_generations = self.num_generations
        if self.num_turns > 1 and self.num_generations != 1:
            self.num_generations = 1
            super().__post_init__()
            self.num_generations = original_num_generations
            return
        super().__post_init__()


class PaperAlignedMAACTrainer(BaseMAACTrainer):
    """HouseBuild MAAC variant closer to the CoLLM-CC paper update rule.

    The upstream MAAC implementation updates the centralized critic inside each
    per-agent actor update. This class keeps rollout collection intact, then
    updates all actors from the same joint-transition minibatch and updates the
    shared critic once.
    """

    algorithm_name = "MAAC-PaperAligned"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.joint_rollout_buffer: List[List[RolloutSample]] = []
        self._detail_log_path = self._resolve_detail_log_path()
        self._best_model_config = self._resolve_best_model_config()
        self._best_metric_value: Optional[float] = None

    def _resolve_detail_log_path(self) -> Optional[str]:
        if not bool(getattr(self.args, "detailed_logging", True)):
            return None
        output_dir = None
        if isinstance(getattr(self, "wandb_config", None), dict):
            sections = self.wandb_config.get("config_sections") or {}
            if isinstance(sections, dict):
                output = sections.get("output") or {}
                if isinstance(output, dict):
                    output_dir = output.get("base_dir")
        output_dir = output_dir or os.getcwd()
        try:
            os.makedirs(str(output_dir), exist_ok=True)
        except Exception:
            return None
        filename = str(
            getattr(
                self.args,
                "maac_detail_log_name",
                "maac_paper_aligned_details.jsonl",
            )
        )
        return os.path.join(str(output_dir), filename)

    def _output_config(self) -> Dict[str, Any]:
        if isinstance(getattr(self, "wandb_config", None), dict):
            sections = self.wandb_config.get("config_sections") or {}
            if isinstance(sections, dict):
                output = sections.get("output") or {}
                if isinstance(output, dict):
                    return output
        return {}

    @staticmethod
    def _as_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return bool(default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            s = value.strip().lower()
            if s in ("true", "1", "yes", "y", "t"):
                return True
            if s in ("false", "0", "no", "n", "f", "none", "null", ""):
                return False
        return bool(value)

    def _resolve_best_model_config(self) -> Dict[str, Any]:
        output = self._output_config()
        enabled = self._as_bool(output.get("save_best_model"), False)
        metric_name = str(output.get("best_metric") or "").strip()
        metric_mode = str(output.get("best_metric_mode") or "max").strip().lower()
        if metric_mode not in {"max", "min"}:
            metric_mode = "max"

        path_cfg = output.get("best_model_path")
        if path_cfg:
            best_model_path = str(path_cfg)
        else:
            save_path = output.get("save_path")
            if save_path:
                best_model_path = os.path.join(os.path.dirname(str(save_path)), "best_model")
            else:
                base_dir = output.get("base_dir") or os.getcwd()
                best_model_path = os.path.join(str(base_dir), "best_model")

        return {
            "enabled": enabled,
            "path": os.path.abspath(best_model_path),
            "metric": metric_name,
            "mode": metric_mode,
        }

    def _select_best_metric(self, metrics: Dict[str, float]) -> Tuple[Optional[str], Optional[float]]:
        if not metrics:
            return None, None

        metric_name = str(self._best_model_config.get("metric") or "").strip()
        if metric_name and metric_name in metrics:
            return metric_name, float(metrics[metric_name])

        suffix = "/iou_mean"
        if metric_name and "/" in metric_name:
            suffix = "/" + metric_name.rsplit("/", 1)[-1]
        candidates = [
            (key, value)
            for key, value in metrics.items()
            if key.startswith("eval/turn_") and key.endswith(suffix)
        ]
        if metric_name and not candidates:
            return None, None
        if not candidates:
            candidates = [
                (key, value)
                for key, value in metrics.items()
                if key.startswith("eval/") and key.endswith("/reward_mean")
            ]
        if not candidates:
            numeric = [(key, value) for key, value in metrics.items()]
            candidates = numeric[:1]
        if not candidates:
            return None, None

        def _turn_index(item: Tuple[str, float]) -> int:
            match = re.search(r"eval/turn_(\d+)/", item[0])
            return int(match.group(1)) if match else -1

        key, value = max(candidates, key=_turn_index)
        return key, float(value)

    def _metric_improved(self, value: float) -> bool:
        if self._best_metric_value is None:
            return True
        mode = str(self._best_model_config.get("mode") or "max")
        if mode == "min":
            return value < self._best_metric_value
        return value > self._best_metric_value

    def _maybe_save_best_model(self, eval_metrics: Dict[str, float]) -> None:
        if not self._as_bool(self._best_model_config.get("enabled"), False):
            return
        dist_env = getattr(self, "dist_env", None)
        if not bool(getattr(dist_env, "is_main", True)):
            return

        metric_name, metric_value = self._select_best_metric(eval_metrics)
        if metric_name is None or metric_value is None:
            expected = str(self._best_model_config.get("metric") or "eval/turn_*/iou_mean")
            self._write_detail(
                {
                    "event": "best_model_metric_missing",
                    "expected_metric": expected,
                    "available_metrics": sorted(eval_metrics.keys()),
                }
            )
            if getattr(self, "verbose", True):
                print(
                    f"Best model not saved: metric {expected!r} not found in eval metrics.",
                    flush=True,
                )
            return
        if not self._metric_improved(metric_value):
            return

        self._best_metric_value = float(metric_value)
        output_dir = str(self._best_model_config["path"])
        self.save_model(output_dir)
        metadata = {
            "event": "best_model_saved",
            "time": time.time(),
            "env_step": int(getattr(self, "env_step", 0)),
            "metric_name": metric_name,
            "metric_value": float(metric_value),
            "metric_mode": str(self._best_model_config.get("mode") or "max"),
            "metrics": eval_metrics,
            "config_sections": (
                self.wandb_config.get("config_sections")
                if isinstance(getattr(self, "wandb_config", None), dict)
                else None
            ),
        }
        try:
            with open(os.path.join(output_dir, "best_model_info.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
        except Exception:
            pass
        self._write_detail(metadata)
        print(
            f"Best model saved to: {output_dir} "
            f"({metric_name}={float(metric_value):.6f}, step={int(getattr(self, 'env_step', 0))})",
            flush=True,
        )

    def evaluate(self) -> Dict[str, float]:
        metrics = super().evaluate()
        self._maybe_save_best_model(metrics)
        return metrics

    def _summarize_rollout_metrics(self, rollouts: List[RolloutSample]) -> Dict[str, float]:
        metrics = super()._summarize_rollout_metrics(rollouts)
        scalar_keys = [
            "iou",
            "coverage_rate",
            "redundancy_rate",
            "score_match",
            "exact_non_air_rate",
            "covered_blocks",
            "extra_blocks",
            "expected_non_air",
            "observed_non_air",
            "build_reward_raw",
            "spider_penalty",
            "reward_raw",
            "level_1",
            "level_2",
            "level_total",
        ]
        by_key: Dict[str, List[float]] = {key: [] for key in scalar_keys}
        for sample in rollouts:
            metadata = getattr(sample, "metadata", {}) or {}
            reward_metrics = metadata.get("reward_metrics") or {}
            if not isinstance(reward_metrics, dict):
                reward_metrics = {}
            for key in scalar_keys:
                value = reward_metrics.get(key, metadata.get(key))
                if value is None:
                    continue
                try:
                    value_f = float(value)
                except Exception:
                    continue
                by_key[key].append(value_f)
        for key, values in by_key.items():
            if values:
                metrics[f"{key}_mean"] = float(sum(values) / len(values))
        return metrics

    def _write_detail(self, payload: Dict[str, Any]) -> None:
        if not self._detail_log_path:
            return
        dist_env = getattr(self, "dist_env", None)
        if not bool(getattr(dist_env, "is_main", True)):
            return
        record = {
            "time": time.time(),
            "env_step": int(getattr(self, "env_step", 0)),
            **payload,
        }
        try:
            with open(self._detail_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception:
            return

    def _agent_device_for(self, agent_idx: int) -> torch.device:
        if hasattr(self, "_agent_device"):
            return self._agent_device(agent_idx)  # type: ignore[attr-defined]
        return getattr(self, "device", torch.device("cpu"))

    def _critic_device_for(self) -> torch.device:
        return getattr(self, "critic_device", getattr(self, "device", torch.device("cpu")))

    def _value_on_prompt_only(
        self,
        model: Any,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_len: int,
    ) -> torch.Tensor:
        """Evaluate V(h) without materializing unused LM logits.

        CoLLM-CC only needs the critic value head for Eq. 4/5. The upstream
        helper calls the CausalLM wrapper and therefore builds full vocabulary
        logits, which is the dominant GPU-2 allocation for long HouseBuild
        prompts.
        """
        prompt_ids = sequences[:, :prompt_len]
        prompt_mask = (
            attention_mask[:, :prompt_len] if attention_mask is not None else None
        )

        value_head = getattr(model, "value_head", None)
        causal_lm = getattr(model, "model", None)
        backbone = None
        if causal_lm is not None:
            backbone = getattr(causal_lm, "model", None)
            if backbone is None:
                backbone = getattr(causal_lm, "transformer", None)

        if value_head is None or backbone is None:
            return super()._value_on_prompt_only(
                model, sequences, attention_mask, prompt_len
            )

        outputs = backbone(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden_states = getattr(outputs, "last_hidden_state", None)
        if hidden_states is None:
            all_hidden = getattr(outputs, "hidden_states", None)
            if not all_hidden:
                return super()._value_on_prompt_only(
                    model, sequences, attention_mask, prompt_len
                )
            hidden_states = all_hidden[-1]

        values = value_head(hidden_states).squeeze(-1)
        return values[:, prompt_len - 1]

    def _generate_one(self, agent_model: Any, prompt: str, agent_idx: int) -> Dict[str, Any]:
        agent_device = self._agent_device(agent_idx)
        encoded_prompt = self._encode_prompt(
            prompt, agent_idx=agent_idx, device=agent_device
        )
        prompt_input_ids = encoded_prompt["input_ids"]
        prompt_attention_mask = encoded_prompt["attention_mask"]
        prompt_len = prompt_input_ids.size(1)

        generation_kwargs: Dict[str, Any] = {
            "input_ids": prompt_input_ids,
            "attention_mask": prompt_attention_mask,
            "max_new_tokens": self.args.max_new_tokens,
            "do_sample": True,
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "num_return_sequences": 1,
            "num_beams": 1,
        }
        if self.args.top_k is not None:
            generation_kwargs["top_k"] = self.args.top_k

        generation_model = unwrap_model(agent_model)
        sequences = generation_model.generate(**generation_kwargs)
        if sequences.size(1) <= prompt_len:
            raise RuntimeError("Model produced an empty completion during rollout.")

        response_tokens = sequences[:, prompt_len:]
        tokenizer = self._get_tokenizer(agent_idx)
        pad_id = tokenizer.pad_token_id
        response_lens: List[int] = []
        completion_texts: List[str] = []
        for seq in response_tokens:
            pad_positions = (seq == pad_id).nonzero(as_tuple=False)
            resp_len = (
                pad_positions[0].item() if pad_positions.numel() > 0 else seq.size(0)
            )
            response_lens.append(resp_len)
            completion_texts.append(
                tokenizer.decode(seq[:resp_len], skip_special_tokens=True)
            )

        return {
            "prompt": prompt,
            "prompt_len": prompt_len,
            "sequences": sequences,
            "attention_mask": torch.ones_like(sequences, device=agent_device),
            "response_lens": response_lens,
            "completions": completion_texts,
        }

    def _group_rollouts_by_turn(
        self, rollouts: Iterable[RolloutSample]
    ) -> List[List[RolloutSample]]:
        groups: Dict[Tuple[int, Tuple[int, ...]], List[RolloutSample]] = defaultdict(list)
        for sample in rollouts:
            metadata = getattr(sample, "metadata", {}) or {}
            turn_idx = int(metadata.get("turn_idx", 0))
            branch_raw = metadata.get("branch")
            if isinstance(branch_raw, (list, tuple)):
                branch = tuple(int(x) for x in branch_raw)
            else:
                branch = (int(metadata.get("generation_idx", 0)),)
            groups[(turn_idx, branch)].append(sample)
        return [groups[idx] for idx in sorted(groups)]

    def _collect_rollouts_multi_turn(
        self, item: Dict[str, Any], num_turns: int
    ) -> List[RolloutSample]:
        num_ret = max(1, int(getattr(self.args, "num_generations", 1)))
        gamma = float(getattr(self.args, "discount", 0.9))
        all_rollouts: List[RolloutSample] = []

        def _mean(values: List[float]) -> float:
            return float(sum(values) / len(values)) if values else 0.0

        for gen_idx in range(num_ret):
            prompt_history = [[] for _ in range(self.args.num_agents)]
            response_history = [[] for _ in range(self.args.num_agents)]
            previous_completions: List[Optional[str]] = [None] * self.args.num_agents
            per_agent_samples: List[List[RolloutSample]] = [
                [] for _ in range(self.args.num_agents)
            ]

            for turn_idx in range(num_turns):
                if turn_idx == 0:
                    turn_prompts = [
                        self._resolve_turn_prompt(item, agent_idx)
                        for agent_idx in range(self.args.num_agents)
                    ]
                else:
                    if self.external_transition is None:
                        raise ValueError("external_transition is required for multi-turn.")
                    transition_result = self.external_transition(
                        prompt=item.get("prompt", ""),
                        agent_completions=previous_completions,
                        num_agents=self.args.num_agents,
                        prompt_history_per_agent=prompt_history,
                        response_history_per_agent=response_history,
                    )
                    if (
                        not isinstance(transition_result, (list, tuple))
                        or len(transition_result) != self.args.num_agents
                    ):
                        raise ValueError(
                            "External transition must return per-agent prompts"
                        )
                    turn_prompts = [
                        self._resolve_turn_prompt(
                            item, agent_idx, external_prompt=transition_result[agent_idx]
                        )
                        for agent_idx in range(self.args.num_agents)
                    ]

                for agent_idx, prompt in enumerate(turn_prompts):
                    prompt_history[agent_idx].append(prompt)

                def _generate_agent_turn(agent_idx: int) -> Dict[str, Any]:
                    agent_model = self.agents[agent_idx]
                    prompt = turn_prompts[agent_idx]
                    gen = self._generate_one(agent_model, prompt, agent_idx)
                    return {
                        "agent_idx": agent_idx,
                        "prompt": prompt,
                        "prompt_len": gen["prompt_len"],
                        "sequences": gen["sequences"],
                        "attention_mask": gen["attention_mask"],
                        "response_lens": gen["response_lens"],
                        "completion_texts": gen["completions"],
                    }

                rollout_data = self._run_agent_tasks(_generate_agent_turn)
                rollout_data = sorted(rollout_data, key=lambda entry: int(entry["agent_idx"]))
                completions_per_agent = [
                    entry["completion_texts"] for entry in rollout_data
                ]

                batch_item = dict(item)
                batch_item["_house_build_turn"] = turn_idx + 1
                rewards = call_reward_function(
                    self.reward_func,
                    turn_prompts,
                    completions_per_agent,
                    num_agents=self.args.num_agents,
                    batch_items=[batch_item],
                    signature=self._reward_signature,
                )
                reward_details_raw = getattr(self.reward_func, "last_details", None)
                reward_detail: Dict[str, Any] = {}
                if isinstance(reward_details_raw, list) and reward_details_raw:
                    maybe_detail = reward_details_raw[0]
                    if isinstance(maybe_detail, dict):
                        reward_detail = maybe_detail
                elif isinstance(reward_details_raw, dict):
                    reward_detail = reward_details_raw
                reward_metrics = reward_detail.get("scalar_metrics") or {}
                if not isinstance(reward_metrics, dict):
                    reward_metrics = {}
                rewards = normalize_reward_lengths(
                    [float(self.reward_processor(r)) for r in rewards],
                    num_agents=self.args.num_agents,
                    num_generations=1,
                    algorithm="MAAC",
                )
                rewards_matrix = self._expand_rewards(rewards, num_ret=1)
                joint_action = [
                    completions_per_agent[agent_idx][0]
                    for agent_idx in range(self.args.num_agents)
                ]
                critic_input = self._build_critic_input(turn_prompts, joint_action)
                with torch.no_grad():
                    critic_pack = self._critic_value_from_text(critic_input)
                joint_ids = critic_pack["input_ids"]
                joint_mask = critic_pack["attention_mask"]
                joint_len = int(critic_pack["prompt_len"])
                joint_value = critic_pack["value"]

                for data in rollout_data:
                    agent_idx = int(data["agent_idx"])
                    seq = data["sequences"][0]
                    attn = data["attention_mask"][0]
                    resp_len = data["response_lens"][0]
                    reward_val = float(rewards_matrix[agent_idx][0])
                    reward_cpu = torch.tensor([reward_val], dtype=torch.float32)

                    logprob, _ = self._policy_eval(
                        self.agents[agent_idx],
                        seq.unsqueeze(0),
                        attn.unsqueeze(0),
                        data["prompt_len"],
                        resp_len,
                        output_values=False,
                    )

                    completion_text = data["completion_texts"][0]
                    sample = RolloutSample(
                        agent_idx=agent_idx,
                        prompt=data["prompt"],
                        completion=completion_text,
                        full_input_ids=seq.detach().cpu(),
                        attention_mask=attn.detach().cpu(),
                        prompt_len=data["prompt_len"],
                        response_len=resp_len,
                        old_logprob=logprob.detach().cpu(),
                        old_value=joint_value.detach().cpu(),
                        reward=reward_cpu,
                        returns=reward_cpu.clone(),
                        advantage=torch.zeros_like(reward_cpu),
                        normalized_advantage=None,
                        metadata={
                            "joint_input_ids": joint_ids.detach().cpu(),
                            "joint_attention_mask": joint_mask.detach().cpu(),
                            "joint_prompt_len": joint_len,
                            "turn_idx": turn_idx,
                            "generation_idx": gen_idx,
                            "branch": (gen_idx,),
                            "trajectory_idx": gen_idx,
                            "adv_target": reward_cpu,
                            "value_target": reward_cpu,
                            "reward_metrics": dict(reward_metrics),
                            "reward_detail": reward_detail,
                        },
                    )
                    all_rollouts.append(sample)
                    per_agent_samples[agent_idx].append(sample)
                    response_history[agent_idx].append(completion_text)
                    previous_completions[agent_idx] = completion_text

                term_threshold = getattr(self.args, "early_termination_threshold", None)
                if term_threshold is not None:
                    mean_reward = _mean([float(r) for r in rewards])
                    if mean_reward > float(term_threshold):
                        break

            for agent_idx in range(self.args.num_agents):
                traj = per_agent_samples[agent_idx]
                for t, sample in enumerate(traj):
                    if t < len(traj) - 1:
                        next_sample = traj[t + 1]
                        sample.metadata["next_joint_input_ids"] = next_sample.metadata[
                            "joint_input_ids"
                        ]
                        sample.metadata["next_joint_attention_mask"] = (
                            next_sample.metadata["joint_attention_mask"]
                        )
                        sample.metadata["next_joint_prompt_len"] = next_sample.metadata[
                            "joint_prompt_len"
                        ]
                    else:
                        sample.metadata["next_joint_input_ids"] = None
                        sample.metadata["next_joint_attention_mask"] = None
                        sample.metadata["next_joint_prompt_len"] = None

                    r = float(sample.reward.view(-1)[0].item())
                    if t < len(traj) - 1:
                        next_v = float(traj[t + 1].old_value.view(-1)[0].item())
                        target = r + gamma * next_v
                    else:
                        target = r
                    target_tensor = torch.tensor([target], dtype=torch.float32)
                    sample.metadata["adv_target"] = target_tensor.cpu()
                    sample.metadata["value_target"] = target_tensor.cpu()

                future = 0.0
                for sample in reversed(traj):
                    immediate = float(sample.reward.view(-1)[0].item())
                    future = immediate + gamma * future
                    sample.returns = torch.tensor([future], dtype=torch.float32)
                    sample.advantage = torch.zeros_like(sample.returns)
                    sample.normalized_advantage = None

        if self.metrics_callback is not None:
            extra = self.metrics_callback(all_rollouts)
            if isinstance(extra, dict):
                self._log_metrics(extra)
        return all_rollouts

    def _run_batch(self, batch: Any, epoch_metrics: Dict[str, List[float]]) -> None:
        for item in batch:
            rollouts = self._collect_rollouts(item)
            for group in self._group_rollouts_by_turn(rollouts):
                self.joint_rollout_buffer.append(group)
                if len(self.joint_rollout_buffer) >= int(self.args.rollout_buffer_size):
                    self._drain_joint_buffer(epoch_metrics)
            if self.args.num_agents > 0:
                self.env_step += len(rollouts) // self.args.num_agents

    def _flush_buffers(self, epoch_metrics: Dict[str, List[float]]) -> None:
        self._drain_joint_buffer(epoch_metrics)

    def _drain_joint_buffer(self, epoch_metrics: Dict[str, List[float]]) -> None:
        if not self.joint_rollout_buffer:
            return
        groups = self.joint_rollout_buffer
        self.joint_rollout_buffer = []

        by_turn: Dict[int, List[List[RolloutSample]]] = defaultdict(list)
        for group in groups:
            if not group:
                continue
            turn_idx = int((group[0].metadata or {}).get("turn_idx", 0))
            by_turn[turn_idx].append(group)

        combined_log: Dict[str, float] = {}
        for turn_idx in sorted(by_turn):
            metrics = self._joint_update(by_turn[turn_idx])
            tagged = self._tag_metrics(metrics, 0, turn_idx=turn_idx)
            combined_log.update(tagged)
            for key, value in tagged.items():
                epoch_metrics[key].append(value)

        if combined_log and self._should_log_train():
            self._log_metrics(combined_log)

    def _clip_advantages(self, samples: List[RolloutSample]) -> None:
        clip = getattr(self.args, "advantage_clip", None)
        if clip is None:
            return
        try:
            clip_val = float(clip)
        except Exception:
            return
        if clip_val <= 0:
            return
        for sample in samples:
            if sample.normalized_advantage is not None:
                sample.normalized_advantage = torch.clamp(
                    sample.normalized_advantage, -clip_val, clip_val
                )

    def _clip_advantage_tensor(self, advantage: torch.Tensor) -> torch.Tensor:
        clip = getattr(self.args, "advantage_clip", None)
        if clip is None:
            return advantage
        try:
            clip_val = float(clip)
        except Exception:
            return advantage
        if clip_val <= 0:
            return advantage
        return torch.clamp(advantage, -clip_val, clip_val)

    def _joint_update(self, groups: List[List[RolloutSample]]) -> Dict[str, float]:
        flat_samples = [sample for group in groups for sample in group]
        if not flat_samples:
            return {}

        metrics = self._summarize_rollout_metrics(flat_samples)

        actor_losses_by_agent: Dict[int, List[torch.Tensor]] = defaultdict(list)
        critic_losses: List[torch.Tensor] = []
        ratio_values: List[float] = []
        unclipped_ratio_values: List[float] = []
        advantage_values: List[float] = []
        td_values: List[float] = []
        td_abs_values: List[float] = []
        current_value_values: List[float] = []
        target_values: List[float] = []

        critic_device = self._critic_device_for()
        gamma = float(getattr(self.args, "discount", 0.9))
        use_ratio = bool(getattr(self.args, "use_importance_ratio", True))
        ratio_clip = getattr(self.args, "policy_ratio_clip", None)
        if ratio_clip is not None:
            try:
                ratio_clip = float(ratio_clip)
            except Exception:
                ratio_clip = None

        for group_idx, group in enumerate(groups):
            if not group:
                continue

            first = group[0]
            joint_ids = first.metadata["joint_input_ids"].to(critic_device)
            joint_mask = first.metadata["joint_attention_mask"].to(critic_device)
            joint_len = int(first.metadata["joint_prompt_len"])
            value = self._value_on_prompt_only(
                self.critics[0], joint_ids, joint_mask, joint_len
            )

            reward_vals = [
                sample.reward.to(value.device, dtype=value.dtype).view(-1)[0]
                for sample in group
            ]
            reward = torch.stack(reward_vals).mean().view_as(value)

            next_ids = first.metadata.get("next_joint_input_ids")
            next_mask = first.metadata.get("next_joint_attention_mask")
            next_len = first.metadata.get("next_joint_prompt_len")
            if next_ids is not None and next_mask is not None and next_len is not None:
                with torch.no_grad():
                    next_value = self._value_on_prompt_only(
                        self.critics[0],
                        next_ids.to(critic_device),
                        next_mask.to(critic_device),
                        int(next_len),
                    )
                value_target = reward + gamma * next_value
            else:
                next_value = None
                value_target = reward

            td_error = value_target - value
            policy_advantage_tensor = self._clip_advantage_tensor(td_error.detach())
            value_error = td_error**2
            critic_losses.append(value_error)

            current_value = float(value.detach().view(-1)[0].float().cpu().item())
            target_value = float(
                value_target.detach().view(-1)[0].float().cpu().item()
            )
            td_error_float = float(
                td_error.detach().view(-1)[0].float().cpu().item()
            )
            current_value_values.append(current_value)
            target_values.append(target_value)
            td_values.append(td_error_float)
            td_abs_values.append(abs(td_error_float))

            for sample in group:
                agent_idx = int(sample.agent_idx)
                agent_device = self._agent_device_for(agent_idx)
                sequences = sample.full_input_ids.to(agent_device).unsqueeze(0)
                attention_mask = sample.attention_mask.to(agent_device).unsqueeze(0)
                logprob, _ = self._policy_eval(
                    self.agents[agent_idx],
                    sequences,
                    attention_mask,
                    sample.prompt_len,
                    sample.response_len,
                    output_values=False,
                )

                policy_advantage = policy_advantage_tensor.to(
                    agent_device, dtype=logprob.dtype
                )
                old_logprob = sample.old_logprob.to(agent_device, dtype=logprob.dtype)
                log_ratio = (logprob - old_logprob).clamp(min=-20.0, max=20.0)
                ratio = torch.exp(log_ratio)
                unclipped_ratio = ratio
                if not use_ratio:
                    ratio = torch.ones_like(logprob)
                elif ratio_clip is not None and ratio_clip > 0:
                    ratio = torch.clamp(ratio, 1.0 - ratio_clip, 1.0 + ratio_clip)

                if not torch.isfinite(logprob).all():
                    raise FloatingPointError("Encountered non-finite logprob.")
                if not torch.isfinite(policy_advantage).all():
                    raise FloatingPointError("Advantage contains non-finite values.")
                if not torch.isfinite(ratio).all():
                    raise FloatingPointError("Importance ratio contains non-finite values.")

                policy_loss = -(ratio * policy_advantage)
                actor_losses_by_agent[agent_idx].append(policy_loss)

                ratio_float = float(ratio.detach().view(-1)[0].float().cpu().item())
                unclip_ratio_float = float(
                    unclipped_ratio.detach().view(-1)[0].float().cpu().item()
                )
                adv_float = float(
                    policy_advantage.detach().view(-1)[0].float().cpu().item()
                )
                ratio_values.append(ratio_float)
                unclipped_ratio_values.append(unclip_ratio_float)
                advantage_values.append(adv_float)

                sample.advantage = td_error.detach().cpu()
                sample.normalized_advantage = policy_advantage.detach().cpu()
                sample.metadata["value_target"] = value_target.detach().cpu()
                self._write_detail(
                    {
                        "event": "maac_update_sample",
                        "group_idx": group_idx,
                        "turn_idx": int((sample.metadata or {}).get("turn_idx", 0)),
                        "agent_idx": agent_idx,
                        "prompt": sample.prompt,
                        "completion": sample.completion,
                        "reward_processed": float(
                            sample.reward.view(-1)[0].float().cpu().item()
                        ),
                        "return": float(sample.returns.view(-1)[0].float().cpu().item()),
                        "value_pred_old": float(
                            sample.old_value.view(-1)[0].float().cpu().item()
                        ),
                        "value_pred_current": current_value,
                        "value_target": target_value,
                        "td_error_current": td_error_float,
                        "next_value_current": (
                            None
                            if next_value is None
                            else float(
                                next_value.detach().view(-1)[0].float().cpu().item()
                            )
                        ),
                        "old_logprob": float(
                            sample.old_logprob.view(-1)[0].float().cpu().item()
                        ),
                        "current_logprob": float(
                            logprob.detach().view(-1)[0].float().cpu().item()
                        ),
                        "importance_ratio": ratio_float,
                        "importance_ratio_unclipped": unclip_ratio_float,
                        "advantage": float(
                            sample.advantage.view(-1)[0].float().cpu().item()
                        ),
                        "policy_advantage": adv_float,
                    }
                )

        value_loss = torch.stack(critic_losses).mean()
        if not torch.isfinite(value_loss):
            raise FloatingPointError("Non-finite value loss detected.")
        self.critic_optimizer.zero_grad(set_to_none=True)
        (float(self.args.value_loss_coef) * value_loss).backward()
        self.critic_optimizer.step()

        actor_loss_values: List[float] = []
        for agent_idx, actor_losses in sorted(actor_losses_by_agent.items()):
            if not actor_losses:
                continue
            actor_loss = torch.stack(actor_losses).mean()
            if not torch.isfinite(actor_loss):
                raise FloatingPointError("Non-finite actor loss detected.")
            optimizer = self.agent_optimizers[agent_idx]
            optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            optimizer.step()
            actor_loss_values.append(float(actor_loss.detach().cpu().item()))

        def _mean(vals: List[float]) -> float:
            return float(sum(vals) / len(vals)) if vals else 0.0

        def _std(vals: List[float]) -> float:
            if len(vals) < 2:
                return 0.0
            mean = _mean(vals)
            return float((sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5)

        metrics.update(
            {
                "policy_loss": _mean(actor_loss_values),
                "value_loss": float(value_loss.detach().cpu().item()),
                "importance_ratio_mean": _mean(ratio_values),
                "importance_ratio_std": _std(ratio_values),
                "importance_ratio_unclipped_mean": _mean(unclipped_ratio_values),
                "advantage_mean": _mean(advantage_values),
                "advantage_std": _std(advantage_values),
                "td_error_mean": _mean(td_values),
                "td_error_abs_mean": _mean(td_abs_values),
                "value_target_mean": _mean(target_values),
                "value_pred_current_mean": _mean(current_value_values),
                "value_target_current_batch_mean": _mean(target_values),
            }
        )
        return metrics
