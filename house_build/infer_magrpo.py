from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT.parent))

from LLM_Collab_Minecraft.house_build.external import (  # noqa: E402
    get_external_transition,
    set_context_resolver,
)
from LLM_Collab_Minecraft.house_build.rewards.house_builder_reward import (  # noqa: E402
    get_reward_function,
)
from LLM_Collab_Minecraft.house_build.train.train_magrpo import (  # noqa: E402
    _build_formatters,
    _prepare_rpg_state,
    _render_prompt,
    _rpg_placeholders,
    _set_seed,
    _slice_items,
)
from LLM_Collab_Minecraft.house_build.utils.agent_utils import (  # noqa: E402
    extend_fmt_kwargs_with_agents,
    get_agent_template,
    parse_agent_overrides,
    set_agent_values,
    split_limits,
)
from LLM_Collab_Minecraft.house_build.utils.config import (  # noqa: E402
    apply_overrides,
    load_yaml,
    resolve_path,
)
from LLM_Collab_Minecraft.house_build.utils.house_builder import (  # noqa: E402
    TaskSpec,
    compute_resource_limits,
    extract_command_lines,
    format_layers_text,
    legend_lines,
    load_tasks_from_json,
    normalize_block_id,
    score_house_builder,
    simulate_commands_to_scan_blocks,
    unique_block_list,
    validate_and_normalize_mc_commands,
)
from LLM_Collab_Minecraft.house_build.utils.prompting import (  # noqa: E402
    DEFAULT_USER_TEMPLATE_MULTI_AGENT,
    apply_prompt_defaults,
)


def _normalize_key(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _parse_dtype(value: str | None) -> Any:
    if value is None:
        return None
    key = str(value).strip().lower()
    if key in ("", "none", "null"):
        return None
    if key in ("bf16", "bfloat16"):
        return torch.bfloat16
    if key in ("fp16", "float16"):
        return torch.float16
    if key in ("fp32", "float32"):
        return torch.float32
    if key == "auto":
        return "auto"
    raise ValueError(f"Unsupported dtype: {value}")


def _split_csv(value: str | None) -> List[str]:
    if not value:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _find_latest_final_model(run_root: str | Path) -> Path:
    root = Path(run_root)
    candidates = [p for p in root.glob("*/final_model") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No final_model directory found under {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _task_item(task: TaskSpec, dataset_index: int) -> Dict[str, Any]:
    return {
        "task_id": task.task_id,
        "dataset_index": int(dataset_index),
        "local_bbox_from": list(task.local_bbox_from),
        "local_bbox_to": list(task.local_bbox_to),
        "inventory": dict(task.inventory),
        "layers_by_y": {str(k): [str(r) for r in v] for k, v in task.layers_by_y.items()},
        "prompt": f"house_build:{task.task_id}",
    }


def _task_from_item(item: Mapping[str, Any]) -> TaskSpec:
    return TaskSpec(
        task_id=str(item.get("task_id") or ""),
        local_bbox_from=[int(v) for v in (item.get("local_bbox_from") or [0, 0, 0])],
        local_bbox_to=[int(v) for v in (item.get("local_bbox_to") or [0, 0, 0])],
        inventory={str(k): str(v) for k, v in (item.get("inventory") or {}).items()},
        layers_by_y={
            int(k): [str(r) for r in v]
            for k, v in (item.get("layers_by_y") or {}).items()
        },
    )


def _as_block_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    return [text] if text else []


def _allowed_blocks_for_item(
    item: Mapping[str, Any],
    overrides: Sequence[str],
) -> List[str]:
    if overrides:
        return unique_block_list([str(v) for v in overrides])
    inventory = item.get("inventory") or {}
    if isinstance(inventory, Mapping):
        return unique_block_list([str(v) for v in inventory.values()])
    return []


def _resource_limits_text(task: TaskSpec, *, limited_resource: bool, num_agents: int) -> str:
    if not limited_resource:
        return ""
    limits = compute_resource_limits(task, num_agents=num_agents)
    lines: List[str] = []
    for _key, block in task.inventory.items():
        block_norm = normalize_block_id(block)
        if block_norm in ("air", "cave_air", "void_air"):
            continue
        limit_val = limits.get(block_norm)
        if limit_val is not None:
            lines.append(f"- {block_norm}: {limit_val}")
    if not lines:
        return ""
    return "Resource limits per agent (air unlimited):\n" + "\n".join(lines)


def _build_context(
    cfg: Dict[str, Any],
    item: Mapping[str, Any],
    *,
    num_agents: int,
) -> Dict[str, Any]:
    prompt_cfg = cfg.get("prompt") or {}
    if not isinstance(prompt_cfg, dict):
        prompt_cfg = {}
    task_cfg = cfg.get("task") or {}
    if not isinstance(task_cfg, dict):
        task_cfg = {}

    system_prompt = str(prompt_cfg.get("system") or "").rstrip()
    user_template = str(prompt_cfg.get("user_template") or "").rstrip()
    multi_agent_template = str(
        prompt_cfg.get("user_template_multi_agent") or DEFAULT_USER_TEMPLATE_MULTI_AGENT
    ).rstrip()
    include_air_rects = bool(prompt_cfg.get("include_air_rects", False))
    limited_resource = bool(task_cfg.get("limited_resource", False))
    max_commands_total = int(task_cfg.get("max_commands") or 600)
    rpg_kwargs = _rpg_placeholders(cfg)

    task = _task_from_item(item)
    w_from = task.local_bbox_from
    w_to = task.local_bbox_to
    layers_text = format_layers_text(task, world_from=w_from, include_air=include_air_rects)
    legend = legend_lines(task.inventory)

    block_overrides = parse_agent_overrides(
        task_cfg,
        num_agents=num_agents,
        parse_list=_as_block_list,
    )
    allowed_blocks_by_agent = [
        _allowed_blocks_for_item(item, block_overrides[agent_idx])
        for agent_idx in range(num_agents)
    ]

    base_fmt_kwargs = {
        "task_id": str(item.get("task_id") or ""),
        "world_bbox_from": json.dumps(w_from, separators=(",", ":")),
        "world_bbox_to": json.dumps(w_to, separators=(",", ":")),
        "legend_lines": legend,
        "layers_text": layers_text,
        "spider_num": rpg_kwargs.get("spider_num"),
        "player_hp": rpg_kwargs.get("player_hp"),
        "spider_atk": rpg_kwargs.get("spider_atk"),
        "spider_dmg": rpg_kwargs.get("spider_dmg"),
    }
    base_user_single = user_template.format(
        **extend_fmt_kwargs_with_agents(
            dict(base_fmt_kwargs),
            allowed_blocks_by_agent,
            current_agent_idx=0,
        )
    ).rstrip()

    templates_by_agent = [
        get_agent_template(
            prompt_cfg,
            agent_idx,
            default_template=user_template,
            default_multi_agent_template=multi_agent_template,
        )
        for agent_idx in range(num_agents)
    ]
    base_users_by_agent = [
        tmpl.format(
            **extend_fmt_kwargs_with_agents(
                dict(base_fmt_kwargs),
                allowed_blocks_by_agent,
                current_agent_idx=agent_idx,
            )
        ).rstrip()
        for agent_idx, tmpl in enumerate(templates_by_agent)
    ]

    limits_text = _resource_limits_text(
        task,
        limited_resource=limited_resource,
        num_agents=num_agents,
    )
    if limits_text:
        base_user_single = base_user_single + "\n\n" + limits_text
        base_users_by_agent = [text + "\n\n" + limits_text for text in base_users_by_agent]

    context: Dict[str, Any] = {
        "system_prompt": system_prompt,
        "user_prompt_single": base_user_single,
        "task_id": task.task_id,
        "local_bbox_from": list(task.local_bbox_from),
        "local_bbox_to": list(task.local_bbox_to),
        "inventory": dict(task.inventory),
        "layers_by_y": {str(k): [str(r) for r in v] for k, v in task.layers_by_y.items()},
        "max_commands_total": max_commands_total,
        "limited_resource": limited_resource,
        "resource_limits_text": limits_text,
        "rpg_state": cfg.get("_rpg_state"),
    }
    set_agent_values(context, "user_prompt", base_users_by_agent)
    set_agent_values(context, "allowed_blocks", [list(v) for v in allowed_blocks_by_agent])
    return context


def _register_contexts(
    context_map: Dict[str, Dict[str, Any]],
    prompts: Sequence[str],
    item: Mapping[str, Any],
    context: Dict[str, Any],
) -> None:
    keys = [str(item.get("prompt") or ""), *[str(p or "") for p in prompts]]
    for key in keys:
        norm = _normalize_key(key)
        if norm:
            context_map[norm] = context


def _generate_one(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int | None,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"] = float(top_p)
        if top_k is not None:
            gen_kwargs["top_k"] = int(top_k)
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    input_len = int(inputs["input_ids"].shape[1])
    return tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()


def _score_outputs(
    *,
    cfg: Dict[str, Any],
    item: Mapping[str, Any],
    outputs: Sequence[str],
    num_agents: int,
    reward_fn: Any,
) -> Dict[str, Any]:
    task_cfg = cfg.get("task") or {}
    if not isinstance(task_cfg, dict):
        task_cfg = {}
    max_commands_total = int(task_cfg.get("max_commands") or 600)
    limited_resource = bool(task_cfg.get("limited_resource", False))
    max_commands_by_agent = split_limits(max_commands_total, num_agents)
    block_overrides = parse_agent_overrides(
        task_cfg,
        num_agents=num_agents,
        parse_list=_as_block_list,
    )
    task = _task_from_item(item)
    resource_limits = (
        compute_resource_limits(task, num_agents=num_agents) if limited_resource else None
    )

    accepted_by_agent: List[List[str]] = []
    rejected_by_agent: List[List[str]] = []
    for agent_idx in range(num_agents):
        lines = extract_command_lines(outputs[agent_idx] if agent_idx < len(outputs) else "")
        allowed_blocks = _allowed_blocks_for_item(item, block_overrides[agent_idx])
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

    merged = [cmd for commands in accepted_by_agent for cmd in commands]
    blocks = simulate_commands_to_scan_blocks(
        commands=merged,
        world_bbox_from=task.local_bbox_from,
        world_bbox_to=task.local_bbox_to,
    )
    metrics = score_house_builder(task=task, world_scan_blocks=blocks)
    raw_reward = float(
        reward_fn(*[[outputs[i] if i < len(outputs) else ""] for i in range(num_agents)], batch_items=[item])[0]
    )

    processed_reward = raw_reward
    rp_cfg = cfg.get("reward_processor") or {}
    if isinstance(rp_cfg, dict) and rp_cfg.get("enabled", False):
        if rp_cfg.get("scale_factor") is not None:
            processed_reward *= float(rp_cfg.get("scale_factor"))
        if rp_cfg.get("shift") is not None:
            processed_reward += float(rp_cfg.get("shift"))

    return {
        "raw_reward": raw_reward,
        "processed_reward": processed_reward,
        "metrics": metrics,
        "accepted_by_agent": accepted_by_agent,
        "rejected_by_agent": rejected_by_agent,
    }


def _load_items(config_path: str, dataset_json: str | None, split_expr: str) -> List[Dict[str, Any]]:
    json_path = dataset_json or resolve_path(
        config_path,
        (load_yaml(config_path).get("dataset") or {}).get("json_path"),
    )
    tasks = load_tasks_from_json(json_path)
    items = [_task_item(task, idx) for idx, task in enumerate(tasks, start=1)]
    sliced = _slice_items(items, split_expr)
    return sliced or []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HouseBuild MAGRPO inference only.")
    parser.add_argument("--config", default=str(REPO_ROOT / "house_build" / "configs" / "house_build_magrpo_config.yaml"))
    parser.add_argument("--model-dir", default="latest", help="Path to final_model, or 'latest'.")
    parser.add_argument("--run-root", default=str(REPO_ROOT / "runs" / "house_build"))
    parser.add_argument("--dataset-json", default=str(REPO_ROOT / "house_build" / "dataset" / "unseen_eval.json"))
    parser.add_argument("--split", default="[:]", help="Dataset slice, e.g. '[:]', '[8:]', '[:1]'.")
    parser.add_argument("--output-jsonl", default="", help="Optional path for JSONL results.")
    parser.add_argument("--agent-devices", default="", help="Comma-separated devices. Default: cuda:0,cuda:1 when available.")
    parser.add_argument("--num-agents", type=int, default=2)
    parser.add_argument("--num-turns", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--external-mode", default="score_feedback")
    parser.add_argument("--no-sample", action="store_true")
    parser.add_argument("--print-prompts", action="store_true")
    parser.add_argument("--override", nargs="*", default=None, help="Config overrides key=value.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(int(args.seed))
    _set_seed(int(args.seed))

    config_path = str(Path(args.config).expanduser().resolve())
    cfg = load_yaml(config_path)
    if args.override:
        cfg = apply_overrides(cfg, [str(v) for v in args.override if str(v).strip()])
    apply_prompt_defaults(cfg)
    cfg["seed"] = int(args.seed)
    _prepare_rpg_state(cfg, int(args.seed))

    model_dir = (
        _find_latest_final_model(args.run_root)
        if str(args.model_dir).strip().lower() == "latest"
        else Path(args.model_dir).expanduser().resolve()
    )
    if not model_dir.exists():
        raise FileNotFoundError(f"model_dir not found: {model_dir}")

    items = _load_items(config_path, args.dataset_json, args.split)
    if not items:
        raise ValueError(f"No items selected from dataset_json={args.dataset_json!r}, split={args.split!r}")

    devices = _split_csv(args.agent_devices)
    if not devices:
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            devices = [f"cuda:{min(i, count - 1)}" for i in range(args.num_agents)]
        else:
            devices = ["cpu"] * args.num_agents
    if len(devices) < args.num_agents:
        devices.extend([devices[-1]] * (args.num_agents - len(devices)))

    dtype = _parse_dtype(args.dtype)
    model_kwargs: Dict[str, Any] = {}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    tokenizers = []
    models = []
    for agent_idx in range(args.num_agents):
        agent_dir = model_dir / f"agent_{agent_idx}"
        if not agent_dir.exists():
            raise FileNotFoundError(f"Missing agent model directory: {agent_dir}")
        tokenizer = AutoTokenizer.from_pretrained(agent_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(agent_dir, **model_kwargs)
        device = devices[agent_idx]
        model.to(device)
        model.eval()
        tokenizers.append(tokenizer)
        models.append(model)

    formatters = _build_formatters(cfg, num_agents=args.num_agents, tokenizer=tokenizers[0])
    reward_fn = get_reward_function(cfg=cfg, num_agents=args.num_agents)
    context_map: Dict[str, Dict[str, Any]] = {}

    def resolver(prompt: str) -> Dict[str, Any] | None:
        return context_map.get(_normalize_key(prompt))

    set_context_resolver(resolver)

    output_path = Path(args.output_jsonl).expanduser().resolve() if args.output_jsonl else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    print("MODEL_DIR:", model_dir, flush=True)
    print("DATASET_JSON:", args.dataset_json, flush=True)
    print("SPLIT:", args.split, flush=True)
    print("NUM_ITEMS:", len(items), flush=True)
    print("DEVICES:", ",".join(devices[: args.num_agents]), flush=True)

    jsonl_handle = output_path.open("w", encoding="utf-8") if output_path is not None else None
    try:
        for item_idx, item in enumerate(items, start=1):
            context = _build_context(cfg, item, num_agents=args.num_agents)
            prompts = [formatters[agent_idx](item) for agent_idx in range(args.num_agents)]
            _register_contexts(context_map, prompts, item, context)

            prompt_history = [[] for _ in range(args.num_agents)]
            response_history = [[] for _ in range(args.num_agents)]
            turn_records: List[Dict[str, Any]] = []
            final_outputs: List[str] = [""] * args.num_agents
            final_score: Dict[str, Any] = {}

            print("\n" + "=" * 80, flush=True)
            print(
                f"ITEM {item_idx}/{len(items)} task_id={item.get('task_id')} dataset_index={item.get('dataset_index')}",
                flush=True,
            )

            for turn_idx in range(max(1, int(args.num_turns))):
                outputs: List[str] = []
                for agent_idx in range(args.num_agents):
                    if args.print_prompts:
                        print(f"\n--- turn {turn_idx + 1} agent_{agent_idx} prompt ---", flush=True)
                        print(prompts[agent_idx], flush=True)
                    text = _generate_one(
                        model=models[agent_idx],
                        tokenizer=tokenizers[agent_idx],
                        prompt=prompts[agent_idx],
                        device=devices[agent_idx],
                        max_new_tokens=args.max_new_tokens,
                        do_sample=not args.no_sample,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=None if args.top_k < 0 else int(args.top_k),
                    )
                    outputs.append(text)

                score = _score_outputs(
                    cfg=cfg,
                    item=item,
                    outputs=outputs,
                    num_agents=args.num_agents,
                    reward_fn=reward_fn,
                )
                final_outputs = outputs
                final_score = score

                print(f"\n--- turn {turn_idx + 1} score ---", flush=True)
                print(
                    "raw_reward={raw:.4f} processed_reward={proc:.4f} match={match:.4f} iou={iou:.4f}".format(
                        raw=float(score["raw_reward"]),
                        proc=float(score["processed_reward"]),
                        match=float(score["metrics"].get("score_match", 0.0)),
                        iou=float(score["metrics"].get("iou", 0.0)),
                    ),
                    flush=True,
                )
                for agent_idx, text in enumerate(outputs):
                    print(f"\n--- turn {turn_idx + 1} agent_{agent_idx} output ---", flush=True)
                    print(text, flush=True)
                    accepted = score["accepted_by_agent"][agent_idx]
                    rejected = score["rejected_by_agent"][agent_idx]
                    print(
                        f"[accepted={len(accepted)} rejected={len(rejected)}]",
                        flush=True,
                    )

                turn_records.append(
                    {
                        "turn": turn_idx + 1,
                        "outputs": outputs,
                        "score": score,
                    }
                )
                for agent_idx in range(args.num_agents):
                    prompt_history[agent_idx].append(prompts[agent_idx])
                    response_history[agent_idx].append(outputs[agent_idx])

                if turn_idx < int(args.num_turns) - 1:
                    next_prompts = list(
                        get_external_transition(
                            prompt=prompts[0],
                            agent_completions=outputs,
                            num_agents=args.num_agents,
                            mode=args.external_mode,
                            original_prompt=True,
                            previous_response=True,
                            prompt_history_per_agent=prompt_history,
                            response_history_per_agent=response_history,
                        )
                    )
                    if len(next_prompts) != args.num_agents:
                        raise RuntimeError(
                            f"external transition returned {len(next_prompts)} prompts, expected {args.num_agents}"
                        )
                    prompts = next_prompts
                    _register_contexts(context_map, prompts, item, context)

            record = {
                "model_dir": str(model_dir),
                "task_id": item.get("task_id"),
                "dataset_index": item.get("dataset_index"),
                "final_outputs": final_outputs,
                "final_score": final_score,
                "turns": turn_records,
                "created_at": int(time.time()),
            }
            if jsonl_handle is not None:
                jsonl_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                jsonl_handle.flush()
    finally:
        if jsonl_handle is not None:
            jsonl_handle.close()

    if output_path is not None:
        print("\nJSONL_RESULT:", output_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
