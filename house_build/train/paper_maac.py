from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

import torch

from comlrl.trainers.actor_critic import MAACTrainer as BaseMAACTrainer  # type: ignore
from comlrl.trainers.actor_critic.iac import RolloutSample  # type: ignore


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

    def _group_rollouts_by_turn(
        self, rollouts: Iterable[RolloutSample]
    ) -> List[List[RolloutSample]]:
        groups: Dict[int, List[RolloutSample]] = defaultdict(list)
        for sample in rollouts:
            turn_idx = int((getattr(sample, "metadata", {}) or {}).get("turn_idx", 0))
            groups[turn_idx].append(sample)
        return [groups[idx] for idx in sorted(groups)]

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

    def _joint_update(self, groups: List[List[RolloutSample]]) -> Dict[str, float]:
        flat_samples = [sample for group in groups for sample in group]
        if not flat_samples:
            return {}

        metrics = self._summarize_rollout_metrics(flat_samples)
        self._prepare_advantages(flat_samples)
        self._clip_advantages(flat_samples)

        actor_losses_by_agent: Dict[int, List[torch.Tensor]] = defaultdict(list)
        critic_losses: List[torch.Tensor] = []
        ratio_values: List[float] = []
        unclipped_ratio_values: List[float] = []
        advantage_values: List[float] = []
        td_abs_values: List[float] = []
        current_value_values: List[float] = []
        target_values: List[float] = []

        critic_device = self._critic_device_for()
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

            group_targets: List[torch.Tensor] = []
            for sample in group:
                value_target = sample.metadata.get("value_target")
                if value_target is None:
                    raise RuntimeError("value_target missing for critic update.")
                group_targets.append(value_target.to(value.device, dtype=value.dtype))
            returns = torch.stack(group_targets).mean(dim=0)
            value_error = (returns - value) ** 2
            critic_losses.append(value_error)

            current_value = float(value.detach().view(-1)[0].float().cpu().item())
            target_value = float(returns.detach().view(-1)[0].float().cpu().item())
            current_value_values.append(current_value)
            target_values.append(target_value)
            td_abs_values.append(abs(target_value - current_value))

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

                policy_advantage = sample.normalized_advantage.to(
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

                sample_target = sample.metadata.get("value_target")
                sample_target_float = (
                    float(sample_target.view(-1)[0].float().cpu().item())
                    if sample_target is not None
                    else None
                )
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
                        "value_target": sample_target_float,
                        "td_error_current": (
                            None
                            if sample_target_float is None
                            else sample_target_float - current_value
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

        actor_loss_values: List[float] = []
        for agent_idx, actor_losses in sorted(actor_losses_by_agent.items()):
            if not actor_losses:
                continue
            actor_loss = torch.stack(actor_losses).mean()
            if not torch.isfinite(actor_loss):
                raise FloatingPointError("Non-finite actor loss detected.")
            optimizer = self.agent_optimizers[agent_idx]
            optimizer.zero_grad()
            actor_loss.backward()
            optimizer.step()
            actor_loss_values.append(float(actor_loss.detach().cpu().item()))

        value_loss = torch.stack(critic_losses).mean()
        if not torch.isfinite(value_loss):
            raise FloatingPointError("Non-finite value loss detected.")
        self.critic_optimizer.zero_grad()
        (float(self.args.value_loss_coef) * value_loss).backward()
        self.critic_optimizer.step()

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
                "td_error_abs_mean": _mean(td_abs_values),
                "value_pred_current_mean": _mean(current_value_values),
                "value_target_current_batch_mean": _mean(target_values),
            }
        )
        return metrics
