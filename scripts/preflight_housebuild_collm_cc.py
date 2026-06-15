from __future__ import annotations

import argparse
import inspect
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT.parent))


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_housebuild_items(config_path: Path) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    from LLM_Collab_Minecraft.house_build.train.train_maac import (
        _prepare_rpg_state,
        _set_seed,
    )
    from LLM_Collab_Minecraft.house_build.utils.config import load_yaml, resolve_path
    from LLM_Collab_Minecraft.house_build.utils.house_builder import load_tasks_from_json
    from LLM_Collab_Minecraft.house_build.utils.prompting import apply_prompt_defaults

    cfg = load_yaml(str(config_path))
    apply_prompt_defaults(cfg)
    seed = int(cfg.get("seed", (cfg.get("maac") or {}).get("seed", 42)))
    _set_seed(seed)
    _prepare_rpg_state(cfg, seed)

    dataset_cfg = cfg.get("dataset") or {}
    tasks = load_tasks_from_json(resolve_path(str(config_path), dataset_cfg.get("json_path")))
    items: List[Dict[str, Any]] = []
    for idx, task in enumerate(tasks, start=1):
        items.append(
            {
                "task_id": task.task_id,
                "dataset_index": idx,
                "local_bbox_from": task.local_bbox_from,
                "local_bbox_to": task.local_bbox_to,
                "inventory": task.inventory,
                "layers_by_y": {
                    str(k): [str(row) for row in rows]
                    for k, rows in task.layers_by_y.items()
                },
                "prompt": f"house_build:{task.task_id}",
            }
        )
    return cfg, items


def _summarize_token_lengths(cfg: Dict[str, Any], items: List[Dict[str, Any]]) -> Dict[str, Any]:
    from transformers import AutoTokenizer

    from LLM_Collab_Minecraft.house_build.external.score_feedback import (
        format_followup_prompts,
    )
    from LLM_Collab_Minecraft.house_build.train.train_maac import _build_formatters

    maac_cfg = cfg.get("maac") or {}
    model_name = str((cfg.get("agent_model") or {}).get("name") or "")
    num_agents = int(maac_cfg.get("num_agents", 2))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def nt(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False).input_ids)

    formatters = _build_formatters(cfg, num_agents=num_agents, tokenizer=tokenizer)
    initial = [[nt(formatter(item)) for formatter in formatters] for item in items]
    flat = [v for row in initial for v in row]

    item = items[0]
    prompts = [formatter(item) for formatter in formatters]
    ctx = {
        "system_prompt": "",
        "user_prompt_single": prompts[0],
        "task_id": item["task_id"],
        "local_bbox_from": item["local_bbox_from"],
        "local_bbox_to": item["local_bbox_to"],
        "inventory": item["inventory"],
        "layers_by_y": item["layers_by_y"],
        "max_commands_total": int((cfg.get("task") or {}).get("max_commands", 600)),
        "limited_resource": bool((cfg.get("task") or {}).get("limited_resource", False)),
        "rpg_state": cfg.get("_rpg_state"),
    }
    for idx, prompt in enumerate(prompts, start=1):
        ctx[f"user_prompt_agent{idx}"] = prompt
        ctx[f"allowed_blocks_agent{idx}"] = [
            "white_concrete",
            "obsidian",
            "stone_stairs",
            "stone_bricks",
            "planks",
            "air",
        ]

    short_completion = "/kill\n/fill 0 0 0 0 0 0 white_concrete"
    turn_prompt_tokens: List[List[int]] = []
    turn_prompts = prompts
    for turn in range(1, int(maac_cfg.get("num_turns", 4)) + 1):
        turn_prompt_tokens.append([nt(prompt) for prompt in turn_prompts])
        if turn < int(maac_cfg.get("num_turns", 4)):
            turn_prompts = format_followup_prompts(
                ctx=ctx,
                agent_completions=[short_completion] * num_agents,
                num_agents=num_agents,
                original_prompt_flag=True,
                previous_response_flag=True,
                prompt_history_per_agent=[["x"] * turn for _ in range(num_agents)],
                response_history_per_agent=[[] for _ in range(num_agents)],
            )

    per_agent_prompt_sum = [
        sum(turn_tokens[agent_idx] for turn_tokens in turn_prompt_tokens)
        for agent_idx in range(num_agents)
    ]
    max_new = int(maac_cfg.get("max_new_tokens", 512))
    trace_upper = [value + int(maac_cfg.get("num_turns", 4)) * max_new for value in per_agent_prompt_sum]
    return {
        "tokenizer": model_name,
        "initial_prompt_tokens_min_mean_p50_max": [
            min(flat),
            round(statistics.mean(flat), 1),
            statistics.median(flat),
            max(flat),
        ],
        "initial_prompt_tokens_agent_mean": [
            round(statistics.mean([row[i] for row in initial]), 1)
            for i in range(num_agents)
        ],
        "short_response_prompt_tokens_by_turn": turn_prompt_tokens,
        "short_response_trace_prompt_tokens_per_agent": per_agent_prompt_sum,
        "trace_upper_tokens_per_agent_prompt_plus_responses": trace_upper,
    }


def run_preflight(config_path: Path, expect_cuda_devices: int | None, token_stats: bool) -> None:
    import torch
    import transformers
    from comlrl.trainers.actor_critic import MAACConfig

    from LLM_Collab_Minecraft.house_build.external import (
        get_external_transition,
        set_context_resolver,
    )
    from LLM_Collab_Minecraft.house_build.rewards.house_builder_reward import (
        get_reward_function,
    )
    from LLM_Collab_Minecraft.house_build.train.train_maac import (
        _build_formatters,
        _slice_items,
    )
    from LLM_Collab_Minecraft.house_build.utils.trainer_args import (
        get_agent_sampling_config,
        get_maac_args,
    )

    try:
        import comlrl

        comlrl_version = getattr(comlrl, "__version__", "unknown")
    except Exception:
        comlrl_version = "unknown"

    cfg, items = _load_housebuild_items(config_path)
    dataset_cfg = cfg.get("dataset") or {}
    train_items = _slice_items(items, dataset_cfg.get("train_split"))
    eval_items = _slice_items(items, dataset_cfg.get("eval_split"))
    maac_cfg = cfg.get("maac") or {}

    maac_sig = inspect.signature(MAACConfig)
    for field in ("parallel_training", "agent_devices", "critic_devices"):
        _check(
            field in maac_sig.parameters,
            f"Installed comlrl MAACConfig has no {field} field. Use comlrl>=1.3.7.",
        )

    if expect_cuda_devices is not None:
        _check(
            torch.cuda.device_count() >= int(expect_cuda_devices),
            f"Expected at least {expect_cuda_devices} CUDA devices, got {torch.cuda.device_count()}.",
        )

    cfg.setdefault("maac", {})
    cfg["maac"]["parallel_training"] = "mp"
    cfg["maac"]["agent_devices"] = ["cuda:0", "cuda:1"]
    cfg["maac"]["critic_devices"] = ["cuda:2"]
    cfg["maac"]["num_generations"] = 1
    sampling_cfg = get_agent_sampling_config(cfg)
    trainer_args = get_maac_args(cfg, sampling_cfg=sampling_cfg)
    _check(getattr(trainer_args, "parallel_training", None) == "mp", "parallel_training override did not stick.")
    _check(
        list(getattr(trainer_args, "agent_devices", []) or [])[:2] == ["cuda:0", "cuda:1"],
        "agent_devices override did not stick.",
    )
    _check(
        str(getattr(trainer_args, "critic_devices", "")) == "cuda:2"
        or list(getattr(trainer_args, "critic_devices", []) or [])[:1] == ["cuda:2"],
        "critic_devices override did not stick.",
    )
    _check(
        int(getattr(trainer_args, "num_turns", 1)) <= 1
        or int(getattr(trainer_args, "num_generations", 1)) == 1,
        "Current CoMLRL multi-turn MAAC requires num_generations == 1.",
    )

    agent_model = str((cfg.get("agent_model") or {}).get("name") or "")
    critic_model = str((cfg.get("critic_model") or {}).get("name") or "")
    _check(agent_model == "Qwen/Qwen3-4B-Instruct-2507", f"Unexpected agent model: {agent_model}")
    _check(critic_model == "Qwen/Qwen3-4B-Instruct-2507", f"Unexpected critic model: {critic_model}")
    _check(len(items) == 10, f"Expected 10 HouseBuild tasks, got {len(items)}.")
    _check(len(train_items) == 8, f"Expected train split size 8, got {len(train_items)}.")
    _check(len(eval_items) == 2, f"Expected eval split size 2, got {len(eval_items)}.")

    num_agents = int(maac_cfg.get("num_agents", 2))
    formatters = _build_formatters(cfg, num_agents=num_agents, tokenizer=None)
    prompts = [formatter(items[0]) for formatter in formatters]
    _check(len(prompts) == num_agents, f"Expected {num_agents} prompts, got {len(prompts)}.")
    _check(all(prompt.strip() for prompt in prompts), "At least one prompt is empty.")
    _check("Resource limits per agent" in prompts[0], "Limited-resource prompt did not include resource limits.")

    reward_fn = get_reward_function(cfg=cfg, num_agents=num_agents)
    reward_val = float(reward_fn(*([""] for _ in range(num_agents)), batch_items=[items[0]])[0])
    _check(-1.0 <= reward_val <= 1.0, f"Unexpected reward range for empty output: {reward_val}.")

    ctx = {
        "task_id": items[0]["task_id"],
        "local_bbox_from": items[0]["local_bbox_from"],
        "local_bbox_to": items[0]["local_bbox_to"],
        "inventory": items[0]["inventory"],
        "layers_by_y": items[0]["layers_by_y"],
        "max_commands_total": int((cfg.get("task") or {}).get("max_commands", 600)),
        "limited_resource": bool((cfg.get("task") or {}).get("limited_resource", False)),
        "rpg_state": cfg.get("_rpg_state"),
        "user_prompt_single": prompts[0],
    }
    for agent_idx, prompt in enumerate(prompts, start=1):
        ctx[f"user_prompt_agent{agent_idx}"] = prompt
        ctx[f"allowed_blocks_agent{agent_idx}"] = [
            "white_concrete",
            "obsidian",
            "stone_stairs",
            "stone_bricks",
            "planks",
            "air",
        ]
    set_context_resolver(lambda prompt: ctx if prompt == items[0]["prompt"] else None)
    next_prompts = get_external_transition(
        prompt=items[0]["prompt"],
        agent_completions=["/kill"] * num_agents,
        num_agents=num_agents,
        mode="score_feedback",
        original_prompt=True,
        previous_response=True,
    )
    _check(len(next_prompts) == num_agents, "External transition did not return one prompt per agent.")
    _check(
        all("Score feedback:" in str(prompt) for prompt in next_prompts),
        "score_feedback transition did not include score feedback.",
    )

    print("HouseBuild CoLLM-CC preflight OK")
    print(f"repo_root: {REPO_ROOT}")
    print(f"config: {config_path}")
    print(f"comlrl: {comlrl_version}")
    print(f"transformers: {transformers.__version__}")
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"cuda_count: {torch.cuda.device_count()}")
    print(f"tasks/train/eval: {len(items)}/{len(train_items)}/{len(eval_items)}")
    print(f"algorithm: CoLLM-CC via CoMLRL MAACTrainer centralized critic")
    print(f"agent_model: {agent_model}")
    print(f"critic_model: {critic_model}")
    print(f"maac.parallel_training: {getattr(trainer_args, 'parallel_training', None)}")
    print(f"maac.agent_devices: {getattr(trainer_args, 'agent_devices', None)}")
    print(f"maac.critic_devices: {getattr(trainer_args, 'critic_devices', None)}")
    print(f"maac.num_turns: {getattr(trainer_args, 'num_turns', None)}")
    print(f"maac.num_generations: {getattr(trainer_args, 'num_generations', None)}")
    print(f"maac.num_train_epochs: {getattr(trainer_args, 'num_train_epochs', None)}")
    print(f"maac.agent_learning_rate: {getattr(trainer_args, 'agent_learning_rate', None)}")
    print(f"maac.critic_learning_rate: {getattr(trainer_args, 'critic_learning_rate', None)}")
    print(f"first_prompt_chars: {[len(prompt) for prompt in prompts]}")
    print(f"empty_raw_reward: {reward_val:.6f}")
    if token_stats:
        stats = _summarize_token_lengths(cfg, items)
        print("token_stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate HouseBuild CoLLM-CC wiring without loading model weights.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "house_build" / "configs" / "house_build_maac_config.yaml"),
        help="HouseBuild MAAC/CoLLM-CC YAML config.",
    )
    parser.add_argument(
        "--expect-cuda-devices",
        type=int,
        default=None,
        help="Fail unless at least this many CUDA devices are visible.",
    )
    parser.add_argument(
        "--token-stats",
        action="store_true",
        help="Load the tokenizer and print approximate prompt/trace token lengths.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_preflight(
        Path(args.config).expanduser().resolve(),
        args.expect_cuda_devices,
        bool(args.token_stats),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
