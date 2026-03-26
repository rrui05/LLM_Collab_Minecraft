from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Sequence


def agent_number(agent_idx: int) -> int:
    return int(agent_idx) + 1


def agent_key(prefix: str, agent_idx: int) -> str:
    return f"{prefix}_agent{agent_number(agent_idx)}"


def agent_lines_key(prefix: str, agent_idx: int) -> str:
    return f"{prefix}_agent{agent_number(agent_idx)}_lines"


def get_agent_value(
    mapping: Mapping[str, Any], prefix: str, agent_idx: int, default: Any = None
) -> Any:
    return mapping.get(agent_key(prefix, agent_idx), default)


def get_agent_prompt(
    mapping: Mapping[str, Any], agent_idx: int, *, default_single: str = ""
) -> str:
    fallback = str(default_single or "").rstrip()
    return str(get_agent_value(mapping, "user_prompt", agent_idx, fallback) or fallback).rstrip()


def get_agent_template(
    mapping: Mapping[str, Any],
    agent_idx: int,
    *,
    default_template: str = "",
    default_multi_agent_template: str | None = None,
) -> str:
    template = get_agent_value(mapping, "user_template", agent_idx)
    if template is None or not str(template).strip():
        template = (
            default_multi_agent_template
            if default_multi_agent_template is not None
            else default_template
        )
    return str(template or default_template).rstrip()


def set_agent_values(
    target: MutableMapping[str, Any], prefix: str, values: Sequence[Any]
) -> None:
    for agent_idx, value in enumerate(values):
        target[agent_key(prefix, agent_idx)] = value


def build_block_lines(blocks: Sequence[str]) -> str:
    return "\n".join(f"- {b}" for b in blocks)


def extend_fmt_kwargs_with_agents(
    fmt_kwargs: MutableMapping[str, Any],
    allowed_blocks_by_agent: Sequence[Sequence[str]],
    *,
    current_agent_idx: int | None = None,
) -> Dict[str, Any]:
    for agent_idx, blocks in enumerate(allowed_blocks_by_agent):
        fmt_kwargs[agent_lines_key("block", agent_idx)] = build_block_lines(blocks)
    if current_agent_idx is not None and 0 <= int(current_agent_idx) < len(
        allowed_blocks_by_agent
    ):
        fmt_kwargs["block_agent_lines"] = fmt_kwargs.get(
            agent_lines_key("block", int(current_agent_idx)), ""
        )
        fmt_kwargs["agent_index"] = agent_number(int(current_agent_idx))
        fmt_kwargs["num_agents"] = len(allowed_blocks_by_agent)
    return dict(fmt_kwargs)


def parse_agent_overrides(
    task_cfg: Mapping[str, Any],
    *,
    num_agents: int,
    parse_list: Callable[[Any], List[str]],
    prefix: str = "block",
) -> List[List[str]]:
    raw_many = task_cfg.get(f"{prefix}_agents")
    many_values = list(raw_many) if isinstance(raw_many, (list, tuple)) else []
    overrides: List[List[str]] = []
    for agent_idx in range(max(1, int(num_agents))):
        raw = task_cfg.get(agent_key(prefix, agent_idx))
        if raw is None and agent_idx < len(many_values):
            raw = many_values[agent_idx]
        overrides.append(parse_list(raw))
    return overrides


def get_allowed_blocks(
    mapping: Mapping[str, Any],
    agent_idx: int,
    inventory: Mapping[str, str],
    *,
    unique_block_list: Callable[[List[str]], List[str]],
) -> List[str]:
    raw = get_agent_value(mapping, "allowed_blocks", agent_idx, [])
    if isinstance(raw, (list, tuple)):
        blocks = unique_block_list([str(v) for v in raw])
    else:
        blocks = []
    if not blocks:
        blocks = unique_block_list([str(v) for v in inventory.values()])
    return blocks


def split_limits(total: int, num_agents: int) -> List[int]:
    n = max(1, int(num_agents))
    per = max(1, int(total) // n)
    extra = int(total) % n
    limits = [per] * n
    if limits:
        limits[0] += extra
    return limits
