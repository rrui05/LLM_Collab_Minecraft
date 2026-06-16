from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Mapping

from LLM_Collab_Minecraft.house_build.utils.agent_utils import (
    parse_agent_overrides,
    split_limits,
)
from LLM_Collab_Minecraft.house_build.utils.house_builder import (
    TaskSpec,
    compute_resource_limits,
    extract_command_lines,
    normalize_block_id,
    score_house_builder,
    simulate_commands_to_scan_blocks,
    unique_block_list,
    validate_and_normalize_mc_commands,
)


def _log_train_metrics(metrics: Mapping[str, float], *, turn_idx: int | None) -> None:
    try:
        import wandb  # type: ignore

        run = getattr(wandb, "run", None)
        if run is None:
            return
        prefix = f"turn_{int(turn_idx)}" if turn_idx else "turn_1"
        payload = {f"{prefix}/{k}": float(v) for k, v in metrics.items()}
        wandb.log(payload, commit=False)
    except Exception:
        return




def _as_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _as_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _as_bool(x: Any, default: bool) -> bool:
    if x is None:
        return bool(default)
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "1", "yes", "y", "t"):
            return True
        if s in ("false", "0", "no", "n", "f"):
            return False
    return bool(x)


def _task_from_batch_item(item: Mapping[str, Any]) -> TaskSpec:
    inventory_raw = item.get("inventory") or {}
    layers_by_y = item.get("layers_by_y") or {}
    if isinstance(layers_by_y, dict):
        layers_by_y = {int(k): list(v) for k, v in layers_by_y.items()}
    return TaskSpec(
        task_id=str(item.get("task_id") or ""),
        local_bbox_from=[_as_int(v, 0) for v in (item.get("local_bbox_from") or [0, 0, 0])],
        local_bbox_to=[_as_int(v, 0) for v in (item.get("local_bbox_to") or [0, 0, 0])],
        inventory={str(k): str(v) for k, v in inventory_raw.items()},
        layers_by_y={int(k): [str(r) for r in v] for k, v in (layers_by_y or {}).items()},
    )


def _get_rpg_state(cfg: Dict[str, Any]) -> Dict[str, Any]:
    state = cfg.get("_rpg_state")
    if isinstance(state, dict):
        return state

    task_cfg = cfg.get("task") or {}
    if not isinstance(task_cfg, dict):
        task_cfg = {}
    player_cfg = task_cfg.get("player") or {}
    if not isinstance(player_cfg, dict):
        player_cfg = {}
    spider_cfg = task_cfg.get("spider") or {}
    if not isinstance(spider_cfg, dict):
        spider_cfg = {}

    player_hp = _as_int(player_cfg.get("hp", 0), 0)
    spider_num = _as_int(spider_cfg.get("num", 0), 0)

    atk_values_raw = spider_cfg.get("atk_values") or spider_cfg.get("atk_list") or spider_cfg.get("atk")
    atk_values: List[float] = []
    if isinstance(atk_values_raw, (list, tuple)):
        for v in atk_values_raw:
            try:
                atk_values.append(float(v))
            except Exception:
                continue
    elif atk_values_raw is not None:
        try:
            atk_val = float(atk_values_raw)
            if spider_num > 0:
                atk_values = [atk_val for _ in range(spider_num)]
            else:
                atk_values = [atk_val]
        except Exception:
            atk_values = []

    total_dmg = float(sum(atk_values))
    return {
        "player_hp": player_hp,
        "spider_num": spider_num,
        "spider_atk_values": atk_values,
        "spider_total_dmg": total_dmg,
    }


def get_reward_function(*, cfg: Dict[str, Any], num_agents: int) -> Callable[..., List[float]]:
    task_cfg = cfg.get("task") or {}
    if not isinstance(task_cfg, dict):
        task_cfg = {}

    max_commands_total = _as_int(task_cfg.get("max_commands", 600), 600)
    limited_resource = bool(task_cfg.get("limited_resource", False))

    def _as_block_list(v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, (list, tuple)):
            out = []
            for x in v:
                s = str(x).strip()
                if s:
                    out.append(s)
            return out
        s = str(v).strip()
        return [s] if s else []

    block_agent_overrides = parse_agent_overrides(
        task_cfg,
        num_agents=num_agents,
        parse_list=_as_block_list,
    )

    output_cfg = cfg.get("output") or {}
    if not isinstance(output_cfg, dict):
        output_cfg = {}
    output_verbose = bool(output_cfg.get("verbose", False))
    output_base_dir = str(output_cfg.get("base_dir") or os.getcwd())
    detailed_log_enabled = _as_bool(output_cfg.get("log_detailed_metrics", True), True)
    log_raw_completions = _as_bool(output_cfg.get("log_raw_completions", True), True)
    reward_detail_path = str(
        output_cfg.get("reward_detail_path")
        or os.path.join(output_base_dir, "house_build_reward_details.jsonl")
    )

    reward_cfg = cfg.get("reward") or {}
    if not isinstance(reward_cfg, dict):
        reward_cfg = {}
    reward_mode = str(reward_cfg.get("mode") or "paper").strip().lower()
    coverage_weight = _as_float(reward_cfg.get("coverage_weight", 2.0), 2.0)
    redundancy_weight = _as_float(reward_cfg.get("redundancy_weight", 1.5), 1.5)
    spider_penalty_weight = _as_float(
        reward_cfg.get("spider_penalty_weight", 0.2), 0.2
    )

    debug_enabled = output_verbose
    debug_empty_char = "."
    debug_raw_output = False
    debug_render_layers = True
    rpg_state = _get_rpg_state(cfg)

    def _allowed_blocks_for_task(task: TaskSpec, overrides: List[str]) -> List[str]:
        if overrides:
            return unique_block_list(overrides)
        return unique_block_list(task.inventory.values())

    def _render_layers(task: TaskSpec, obs_map: Mapping[tuple[int, int, int], str]) -> str:
        inventory_rev: Dict[str, str] = {}
        for key, value in task.inventory.items():
            block_norm = normalize_block_id(value)
            if block_norm and block_norm not in inventory_rev:
                inventory_rev[block_norm] = str(key)
        air_key = inventory_rev.get("air")

        min_x = min(task.local_bbox_from[0], task.local_bbox_to[0])
        max_x = max(task.local_bbox_from[0], task.local_bbox_to[0])
        min_y = min(task.local_bbox_from[1], task.local_bbox_to[1])
        max_y = max(task.local_bbox_from[1], task.local_bbox_to[1])
        min_z = min(task.local_bbox_from[2], task.local_bbox_to[2])
        max_z = max(task.local_bbox_from[2], task.local_bbox_to[2])

        lines: List[str] = []
        for y in range(min_y, max_y + 1):
            lines.append(f"y={y}:")
            for z in range(min_z, max_z + 1):
                row: List[str] = []
                for x in range(min_x, max_x + 1):
                    block = normalize_block_id(obs_map.get((x, y, z), "air"))
                    ch = inventory_rev.get(block)
                    if ch is None:
                        if block in ("air", "cave_air", "void_air"):
                            ch = air_key if air_key is not None else debug_empty_char
                        else:
                            ch = "?"
                    row.append(ch)
                lines.append("".join(row))
            lines.append("")
        return "\n".join(lines).rstrip()

    def _maybe_debug_print(
        *,
        task: TaskSpec,
        reward: float,
        metrics: Mapping[str, Any],
        blocks: List[Mapping[str, Any]],
        turn_idx: int | None,
        raw_outputs: List[str] | None,
    ) -> None:
        if not debug_enabled:
            return
        turn_str = f" turn={int(turn_idx)}" if turn_idx is not None else ""
        print(
            f"[house_build debug] {task.task_id}{turn_str} "
            f"reward={reward:.4f} match={float(metrics.get('score_match', 0.0)):.4f}",
            flush=True,
        )
        if debug_render_layers:
            obs_map = {
                (int(b.get("pos")[0]), int(b.get("pos")[1]), int(b.get("pos")[2])): normalize_block_id(b.get("name") or "air")
                for b in blocks
                if isinstance(b.get("pos"), list) and len(b.get("pos")) == 3
            }
            print(_render_layers(task, obs_map), flush=True)
        if debug_raw_output and raw_outputs is not None:
            for idx, raw in enumerate(raw_outputs):
                print(f"[house_build raw] agent{idx}:", flush=True)
                print((raw or "").rstrip(), flush=True)

    max_commands_by_agent = split_limits(max_commands_total, num_agents)
    player_hp_for_penalty = float(rpg_state.get("player_hp", 0) or 0)
    spider_dmg_for_penalty = float(rpg_state.get("spider_total_dmg", 0) or 0)

    def _write_reward_detail(payload: Mapping[str, Any]) -> None:
        if not detailed_log_enabled:
            return
        try:
            os.makedirs(os.path.dirname(reward_detail_path) or ".", exist_ok=True)
            record = {"time": time.time(), **dict(payload)}
            with open(reward_detail_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception:
            return

    def _has_kill(cmds: List[str]) -> bool:
        for cmd in cmds:
            stripped = (cmd or "").strip()
            if stripped.startswith("/"):
                stripped = stripped[1:].lstrip()
            if stripped.lower().startswith("kill"):
                return True
        return False

    def reward_fn(
        *agent_completions: List[str],
        batch_items: List[Mapping[str, Any]] | None = None,
    ) -> List[float]:
        if len(agent_completions) != int(num_agents):
            raise ValueError(
                f"Expected {int(num_agents)} agent completion lists, got {len(agent_completions)}"
            )
        batch_item = (batch_items or [{}])[0]
        task = _task_from_batch_item(batch_item)
        turn_idx = None
        if isinstance(batch_item, Mapping):
            turn_idx = batch_item.get("_house_build_turn")

        resource_limits = compute_resource_limits(task, num_agents=num_agents) if limited_resource else None
        accepted_by_agent: List[List[str]] = []
        rejected_by_agent: List[List[str]] = []
        extracted_lines_by_agent: List[List[str]] = []
        raw_outputs: List[str] = []

        for agent_idx, completions in enumerate(agent_completions):
            allowed_blocks = _allowed_blocks_for_task(
                task,
                block_agent_overrides[agent_idx]
                if agent_idx < len(block_agent_overrides)
                else [],
            )
            completion = completions[0] if completions else ""
            raw_outputs.append(completion)
            lines = extract_command_lines(completion)
            extracted_lines_by_agent.append(lines)
            accepted, rejected = validate_and_normalize_mc_commands(
                lines=lines,
                allowed_blocks=allowed_blocks,
                world_bbox_from=task.local_bbox_from,
                world_bbox_to=task.local_bbox_to,
                max_commands=max_commands_by_agent[agent_idx],
                resource_limits=resource_limits,
            )
            accepted_by_agent.append(accepted)
            rejected_by_agent.append(rejected)

        merged = [cmd for accepted in accepted_by_agent for cmd in accepted]
        blocks = simulate_commands_to_scan_blocks(
            commands=merged,
            world_bbox_from=task.local_bbox_from,
            world_bbox_to=task.local_bbox_to,
        )
        metrics = score_house_builder(task=task, world_scan_blocks=blocks)
        if reward_mode in ("paper", "paper_aligned", "collm_paper"):
            build_reward = (
                coverage_weight * float(metrics.get("coverage_rate", 0.0))
                - redundancy_weight * float(metrics.get("redundancy_rate", 0.0))
            )
        elif reward_mode in ("match", "exact", "legacy"):
            build_reward = float(metrics.get("score_mean", 0.0))
        else:
            raise ValueError(f"Unsupported house_build reward.mode: {reward_mode}")

        spider_penalty = 0.0
        if spider_dmg_for_penalty > 0 and player_hp_for_penalty > 0:
            if not any(_has_kill(accepted) for accepted in accepted_by_agent):
                spider_penalty = (
                    min(1.0, spider_dmg_for_penalty / player_hp_for_penalty)
                    * spider_penalty_weight
                )
        reward = float(build_reward - spider_penalty)
        scalar_metrics = {
            "iou": float(metrics.get("iou", 0.0)),
            "coverage_rate": float(metrics.get("coverage_rate", 0.0)),
            "redundancy_rate": float(metrics.get("redundancy_rate", 0.0)),
            "score_match": float(metrics.get("score_match", 0.0)),
            "exact_non_air_rate": float(metrics.get("exact_non_air_rate", 0.0)),
            "covered_blocks": float(metrics.get("covered_blocks", 0.0)),
            "extra_blocks": float(metrics.get("extra_blocks", 0.0)),
            "expected_non_air": float(metrics.get("expected_non_air", 0.0)),
            "observed_non_air": float(metrics.get("observed_non_air", 0.0)),
            "build_reward_raw": float(build_reward),
            "spider_penalty": float(spider_penalty),
            "reward_raw": float(reward),
            "level_1": float(build_reward),
            "level_2": -float(spider_penalty),
            "level_total": float(reward),
        }
        _log_train_metrics(
            scalar_metrics,
            turn_idx=turn_idx,
        )
        _write_reward_detail(
            {
                "event": "house_build_reward",
                "task_id": task.task_id,
                "turn_idx": turn_idx,
                "reward_mode": reward_mode,
                "reward_raw": reward,
                "build_reward_raw": build_reward,
                "spider_penalty": spider_penalty,
                "spider_total_dmg": spider_dmg_for_penalty,
                "player_hp": player_hp_for_penalty,
                "metrics": dict(metrics),
                "scalar_metrics": scalar_metrics,
                "accepted_commands_by_agent": accepted_by_agent,
                "rejected_commands_by_agent": rejected_by_agent,
                "extracted_lines_by_agent": extracted_lines_by_agent,
                "raw_outputs_by_agent": raw_outputs if log_raw_completions else None,
                "world_scan_blocks": blocks,
            }
        )

        if debug_enabled:
            _maybe_debug_print(
                task=task,
                reward=reward,
                metrics=metrics,
                blocks=blocks,
                turn_idx=turn_idx,
                raw_outputs=raw_outputs,
            )
        return [reward]

    return reward_fn
