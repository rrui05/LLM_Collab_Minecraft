from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT.parent))


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_housebuild_items(config_path: Path) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    from LLM_Collab_Minecraft.house_build.train.train_magrpo import (
        _prepare_rpg_state,
        _set_seed,
    )
    from LLM_Collab_Minecraft.house_build.utils.config import load_yaml, resolve_path
    from LLM_Collab_Minecraft.house_build.utils.house_builder import load_tasks_from_json
    from LLM_Collab_Minecraft.house_build.utils.prompting import apply_prompt_defaults

    cfg = load_yaml(str(config_path))
    apply_prompt_defaults(cfg)
    seed = int(cfg.get("seed", (cfg.get("magrpo") or {}).get("seed", 42)))
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


def run_preflight(config_path: Path, expect_cuda_devices: int | None) -> None:
    import torch
    import transformers
    from comlrl.trainers.reinforce import MAGRPOConfig

    from LLM_Collab_Minecraft.house_build.external import (
        get_external_transition,
        set_context_resolver,
    )
    from LLM_Collab_Minecraft.house_build.rewards.house_builder_reward import (
        get_reward_function,
    )
    from LLM_Collab_Minecraft.house_build.train.train_magrpo import (
        _build_formatters,
        _slice_items,
    )
    from LLM_Collab_Minecraft.house_build.utils.trainer_args import (
        get_agent_sampling_config,
        get_trainer_args,
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

    magrpo_sig = inspect.signature(MAGRPOConfig)
    _check(
        "parallel_training" in magrpo_sig.parameters,
        "Installed comlrl MAGRPOConfig has no parallel_training field. "
        "Use comlrl>=1.3.7 for the 2-GPU Slurm workflow.",
    )
    _check(
        "agent_devices" in magrpo_sig.parameters,
        "Installed comlrl MAGRPOConfig has no agent_devices field. "
        "Use comlrl>=1.3.7 for the 2-GPU Slurm workflow.",
    )

    if expect_cuda_devices is not None:
        _check(
            torch.cuda.device_count() >= int(expect_cuda_devices),
            f"Expected at least {expect_cuda_devices} CUDA devices, got {torch.cuda.device_count()}.",
        )

    cfg.setdefault("magrpo", {})
    cfg["magrpo"]["parallel_training"] = "mp"
    cfg["magrpo"]["agent_devices"] = ["cuda:0", "cuda:1"]
    sampling_cfg = get_agent_sampling_config(cfg)
    trainer_args = get_trainer_args(cfg, sampling_cfg=sampling_cfg)
    _check(getattr(trainer_args, "parallel_training", None) == "mp", "parallel_training override did not stick.")
    _check(
        list(getattr(trainer_args, "agent_devices", []) or [])[:2] == ["cuda:0", "cuda:1"],
        "agent_devices override did not stick.",
    )

    num_agents = int((cfg.get("magrpo") or {}).get("num_agents", 2))
    _check(len(items) == 10, f"Expected 10 HouseBuild tasks, got {len(items)}.")
    _check(len(train_items) == 8, f"Expected train split size 8, got {len(train_items)}.")
    _check(len(eval_items) == 2, f"Expected eval split size 2, got {len(eval_items)}.")

    formatters = _build_formatters(cfg, num_agents=num_agents, tokenizer=None)
    prompts = [formatter(items[0]) for formatter in formatters]
    _check(len(prompts) == num_agents, f"Expected {num_agents} prompts, got {len(prompts)}.")
    _check(all(prompt.strip() for prompt in prompts), "At least one prompt is empty.")
    _check(len(set(prompts)) == len(prompts), "Agent prompts should be different.")
    _check(
        "Resource limits per agent" in prompts[0],
        "Limited-resource prompt did not include resource limits.",
    )

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

    print("HouseBuild preflight OK")
    print(f"repo_root: {REPO_ROOT}")
    print(f"config: {config_path}")
    print(f"comlrl: {comlrl_version}")
    print(f"transformers: {transformers.__version__}")
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"cuda_count: {torch.cuda.device_count()}")
    print(f"tasks/train/eval: {len(items)}/{len(train_items)}/{len(eval_items)}")
    print(f"magrpo.parallel_training: {getattr(trainer_args, 'parallel_training', None)}")
    print(f"magrpo.agent_devices: {getattr(trainer_args, 'agent_devices', None)}")
    print(f"first_prompt_chars: {[len(prompt) for prompt in prompts]}")
    print(f"empty_raw_reward: {reward_val:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate HouseBuild MAGRPO baseline wiring without loading models.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "house_build" / "configs" / "house_build_magrpo_config.yaml"),
        help="HouseBuild MAGRPO YAML config.",
    )
    parser.add_argument(
        "--expect-cuda-devices",
        type=int,
        default=None,
        help="Fail unless at least this many CUDA devices are visible.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_preflight(Path(args.config).expanduser().resolve(), args.expect_cuda_devices)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
