from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT.parent))

from LLM_Collab_Minecraft.house_build.external import (  # noqa: E402
    get_external_transition,
    set_context_resolver,
)
from LLM_Collab_Minecraft.house_build.infer_magrpo import (  # noqa: E402
    _build_context,
    _generate_one,
    _load_items,
    _normalize_key,
    _parse_dtype,
    _register_contexts,
    _score_outputs,
    _split_csv,
)
from LLM_Collab_Minecraft.house_build.rewards.house_builder_reward import (  # noqa: E402
    get_reward_function,
)
from LLM_Collab_Minecraft.house_build.train.train_maac import (  # noqa: E402
    _build_formatters,
    _prepare_rpg_state,
    _set_seed,
)
from LLM_Collab_Minecraft.house_build.utils.config import (  # noqa: E402
    apply_overrides,
    load_yaml,
    resolve_path,
)
from LLM_Collab_Minecraft.house_build.utils.prompting import apply_prompt_defaults  # noqa: E402


def _checkpoint_mtime(path: Path) -> float:
    info_path = path / "best_model_info.json"
    if info_path.exists():
        return info_path.stat().st_mtime
    return path.stat().st_mtime


def _has_agent_dirs(path: Path, num_agents: int) -> bool:
    return all((path / f"agent_{idx}").is_dir() for idx in range(max(1, num_agents)))


def _find_latest_model(run_root: str | Path, *, model_kind: str, num_agents: int) -> Path:
    root = Path(run_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"run_root not found: {root}")

    allowed_names = {
        "any": {"best_model", "final_model"},
        "best": {"best_model"},
        "final": {"final_model"},
    }[model_kind]

    candidates: List[Path] = []
    if root.name in allowed_names and _has_agent_dirs(root, num_agents):
        candidates.append(root)
    for name in sorted(allowed_names):
        direct = root / name
        if direct.is_dir() and _has_agent_dirs(direct, num_agents):
            candidates.append(direct)
    for name in sorted(allowed_names):
        candidates.extend(
            p for p in root.glob(f"*/{name}") if p.is_dir() and _has_agent_dirs(p, num_agents)
        )

    if not candidates:
        names = ", ".join(sorted(allowed_names))
        raise FileNotFoundError(f"No {names} checkpoint found under {root}")
    return max(candidates, key=_checkpoint_mtime)


def _read_model_name(model_dir: Path) -> str:
    info_path = model_dir / "best_model_info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            sections = info.get("config_sections") or {}
            model_cfg = sections.get("model") or {}
            name = model_cfg.get("name")
            if name:
                return str(name)
        except Exception:
            pass

    config_path = model_dir / "agent_0" / "config.json"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            for key in ("_name_or_path", "model_type"):
                val = cfg.get(key)
                if val:
                    return str(val)
            arch = cfg.get("architectures")
            if isinstance(arch, list) and arch:
                return str(arch[0])
        except Exception:
            pass
    return str(model_dir)


def _resolve_dataset_json(config_path: str, cfg: Mapping[str, Any], dataset_json: str) -> str:
    if dataset_json:
        return str(Path(dataset_json).expanduser().resolve())
    dataset_cfg = cfg.get("dataset") or {}
    if not isinstance(dataset_cfg, Mapping):
        dataset_cfg = {}
    return resolve_path(config_path, dataset_cfg.get("json_path"))


def _default_split(cfg: Mapping[str, Any], split_arg: str) -> str:
    if split_arg:
        return split_arg
    dataset_cfg = cfg.get("dataset") or {}
    if isinstance(dataset_cfg, Mapping):
        split = dataset_cfg.get("eval_split") or dataset_cfg.get("test_split")
        if split:
            return str(split)
    return "[:]"


def _fence(text: Any, lang: str = "") -> str:
    body = str(text if text is not None else "")
    body = body.replace("```", "` ` `")
    suffix = lang.strip()
    return f"```{suffix}\n{body.rstrip()}\n```"


def _json_block(value: Any) -> str:
    return _fence(json.dumps(value, ensure_ascii=False, indent=2, default=str), "json")


def _write_markdown(
    *,
    output_path: Path,
    model_dir: Path,
    model_name: str,
    config_path: str,
    dataset_json: str,
    split: str,
    seed: int,
    item: Mapping[str, Any],
    prompts_initial: Sequence[str],
    turn_records: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> None:
    final_score = turn_records[-1].get("score") if turn_records else {}
    lines: List[str] = []
    lines.append("# HouseBuild CoLLM-CC Latest Model Test")
    lines.append("")
    lines.append(f"- created_at: {datetime.now().astimezone().isoformat(timespec='seconds')}")
    lines.append(f"- model_dir: `{model_dir}`")
    lines.append(f"- model_name: `{model_name}`")
    lines.append(f"- checkpoint_kind: `{model_dir.name}`")
    lines.append(f"- config: `{config_path}`")
    lines.append(f"- dataset_json: `{dataset_json}`")
    lines.append(f"- split: `{split}`")
    lines.append(f"- seed: `{seed}`")
    lines.append(f"- num_agents: `{args.num_agents}`")
    lines.append(f"- num_turns: `{args.num_turns}`")
    lines.append(f"- max_new_tokens: `{args.max_new_tokens}`")
    lines.append("")
    lines.append("## Selected Task")
    lines.append("")
    lines.append(_json_block(dict(item)))
    lines.append("")
    lines.append("## Initial Input Prompts")
    for agent_idx, prompt in enumerate(prompts_initial):
        lines.append("")
        lines.append(f"### Agent {agent_idx}")
        lines.append(_fence(prompt, "text"))
    lines.append("")
    lines.append("## Outputs")
    for turn in turn_records:
        turn_idx = int(turn.get("turn", 0))
        score = turn.get("score") or {}
        metrics = score.get("metrics") if isinstance(score, Mapping) else {}
        lines.append("")
        lines.append(f"### Turn {turn_idx}")
        if isinstance(score, Mapping):
            lines.append(
                "- raw_reward: `{:.6f}`, processed_reward: `{:.6f}`, match: `{:.6f}`, iou: `{:.6f}`".format(
                    float(score.get("raw_reward", 0.0)),
                    float(score.get("processed_reward", 0.0)),
                    float((metrics or {}).get("score_match", 0.0)),
                    float((metrics or {}).get("iou", 0.0)),
                )
            )
        for agent_idx, text in enumerate(turn.get("outputs") or []):
            lines.append("")
            lines.append(f"#### Agent {agent_idx} Output")
            lines.append(_fence(text, "text"))
        if isinstance(score, Mapping):
            lines.append("")
            lines.append("#### Score Details")
            lines.append(_json_block(score))
    lines.append("")
    lines.append("## Final Score")
    lines.append("")
    lines.append(_json_block(final_score))
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test the latest HouseBuild CoLLM-CC/MAAC checkpoint on one random eval task."
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "house_build" / "configs" / "house_build_maac_config.yaml"),
    )
    parser.add_argument("--model-dir", default="latest", help="Checkpoint dir, or 'latest'.")
    parser.add_argument(
        "--run-root",
        default=str(REPO_ROOT / "runs" / "house_build_collm_cc"),
        help="Root containing run directories when --model-dir latest.",
    )
    parser.add_argument(
        "--model-kind",
        choices=["best", "final", "any"],
        default="best",
        help="Which checkpoint type to select under --run-root.",
    )
    parser.add_argument("--dataset-json", default="", help="Default: config dataset.json_path.")
    parser.add_argument("--split", default="", help="Default: config dataset.eval_split.")
    parser.add_argument("--output-md", default="", help="Default: runs/house_build_collm_cc_tests/*.md")
    parser.add_argument("--agent-devices", default="", help="Comma-separated devices.")
    parser.add_argument("--num-agents", type=int, default=0)
    parser.add_argument("--num-turns", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=-999)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--external-mode", default="")
    parser.add_argument("--no-sample", action="store_true")
    parser.add_argument("--override", nargs="*", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = str(Path(args.config).expanduser().resolve())
    cfg = load_yaml(config_path)
    if args.override:
        cfg = apply_overrides(cfg, [str(v) for v in args.override if str(v).strip()])
    apply_prompt_defaults(cfg)
    cfg["seed"] = int(args.seed)
    _set_seed(int(args.seed))
    _prepare_rpg_state(cfg, int(args.seed))

    maac_cfg = cfg.get("maac") if isinstance(cfg.get("maac"), dict) else {}
    model_cfg = cfg.get("agent_model") if isinstance(cfg.get("agent_model"), dict) else {}
    external_cfg = cfg.get("external") if isinstance(cfg.get("external"), dict) else {}

    if int(args.num_agents) <= 0:
        args.num_agents = int((maac_cfg or {}).get("num_agents") or 2)
    if int(args.num_turns) <= 0:
        args.num_turns = int((maac_cfg or {}).get("num_turns") or 4)
    if int(args.max_new_tokens) <= 0:
        args.max_new_tokens = int((maac_cfg or {}).get("max_new_tokens") or 512)
    if args.temperature is None:
        args.temperature = float((model_cfg or {}).get("temperature") or 0.6)
    if args.top_p is None:
        args.top_p = float((model_cfg or {}).get("top_p") or 0.6)
    if int(args.top_k) == -999:
        top_k_cfg = (model_cfg or {}).get("top_k")
        args.top_k = -1 if top_k_cfg is None else int(top_k_cfg)
    if not args.external_mode:
        args.external_mode = str((external_cfg or {}).get("mode") or "score_feedback")

    model_dir = (
        _find_latest_model(args.run_root, model_kind=args.model_kind, num_agents=args.num_agents)
        if str(args.model_dir).strip().lower() == "latest"
        else Path(args.model_dir).expanduser().resolve()
    )
    if not model_dir.exists():
        raise FileNotFoundError(f"model_dir not found: {model_dir}")
    if not _has_agent_dirs(model_dir, args.num_agents):
        raise FileNotFoundError(f"model_dir is missing agent dirs: {model_dir}")

    dataset_json = _resolve_dataset_json(config_path, cfg, args.dataset_json)
    split = _default_split(cfg, args.split)
    items = _load_items(config_path, dataset_json, split)
    if not items:
        raise ValueError(f"No test items selected from {dataset_json} split={split!r}")
    rng = random.Random(int(args.seed))
    item = dict(rng.choice(items))

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
        tokenizer = AutoTokenizer.from_pretrained(agent_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(agent_dir, **model_kwargs)
        model.to(devices[agent_idx])
        model.eval()
        tokenizers.append(tokenizer)
        models.append(model)

    formatters = _build_formatters(cfg, num_agents=args.num_agents, tokenizer=tokenizers[0])
    reward_fn = get_reward_function(cfg=cfg, num_agents=args.num_agents)
    context_map: Dict[str, Dict[str, Any]] = {}

    def resolver(prompt: str) -> Dict[str, Any] | None:
        return context_map.get(_normalize_key(prompt))

    set_context_resolver(resolver)
    context = _build_context(cfg, item, num_agents=args.num_agents)
    prompts = [formatters[agent_idx](item) for agent_idx in range(args.num_agents)]
    prompts_initial = list(prompts)
    _register_contexts(context_map, prompts, item, context)

    prompt_history = [[] for _ in range(args.num_agents)]
    response_history = [[] for _ in range(args.num_agents)]
    turn_records: List[Dict[str, Any]] = []

    for turn_idx in range(max(1, int(args.num_turns))):
        outputs: List[str] = []
        for agent_idx in range(args.num_agents):
            text = _generate_one(
                model=models[agent_idx],
                tokenizer=tokenizers[agent_idx],
                prompt=prompts[agent_idx],
                device=devices[agent_idx],
                max_new_tokens=args.max_new_tokens,
                do_sample=not args.no_sample,
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                top_k=None if int(args.top_k) < 0 else int(args.top_k),
            )
            outputs.append(text)

        score = _score_outputs(
            cfg=cfg,
            item=item,
            outputs=outputs,
            num_agents=args.num_agents,
            reward_fn=reward_fn,
        )
        turn_records.append({"turn": turn_idx + 1, "outputs": outputs, "score": score})

        for agent_idx in range(args.num_agents):
            prompt_history[agent_idx].append(prompts[agent_idx])
            response_history[agent_idx].append(outputs[agent_idx])

        if turn_idx < int(args.num_turns) - 1:
            prompts = list(
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
            if len(prompts) != args.num_agents:
                raise RuntimeError(
                    f"external transition returned {len(prompts)} prompts, expected {args.num_agents}"
                )
            _register_contexts(context_map, prompts, item, context)

    if args.output_md:
        output_md = Path(args.output_md).expanduser().resolve()
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output_md = (
            REPO_ROOT
            / "runs"
            / "house_build_collm_cc_tests"
            / f"latest_model_test_{stamp}.md"
        )

    model_name = _read_model_name(model_dir)
    _write_markdown(
        output_path=output_md,
        model_dir=model_dir,
        model_name=model_name,
        config_path=config_path,
        dataset_json=dataset_json,
        split=split,
        seed=int(args.seed),
        item=item,
        prompts_initial=prompts_initial,
        turn_records=turn_records,
        args=args,
    )

    final_score = turn_records[-1]["score"] if turn_records else {}
    final_metrics = final_score.get("metrics", {}) if isinstance(final_score, dict) else {}
    print(f"MODEL_DIR: {model_dir}", flush=True)
    print(f"MODEL_NAME: {model_name}", flush=True)
    print(f"TASK_ID: {item.get('task_id')}", flush=True)
    print(
        "FINAL raw_reward={:.6f} processed_reward={:.6f} match={:.6f} iou={:.6f}".format(
            float(final_score.get("raw_reward", 0.0)) if isinstance(final_score, dict) else 0.0,
            float(final_score.get("processed_reward", 0.0)) if isinstance(final_score, dict) else 0.0,
            float(final_metrics.get("score_match", 0.0)),
            float(final_metrics.get("iou", 0.0)),
        ),
        flush=True,
    )
    print(f"RESULT_MD: {output_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
