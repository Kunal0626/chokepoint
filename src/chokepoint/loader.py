"""Load and validate a network model from JSON.

Two decisions worth defending in an interview:

1.  **All problems are collected, then reported together.** The obvious
    implementation raises on the first bad field. That produces a
    fix-one-line-and-rerun loop, which is miserable for a 40-edge model. The
    cost is that every check has to be written so it can keep going after a
    failure (hence the ``_Problems`` collector and the ``None`` returns
    threaded through the helpers).

2.  **Floats are rejected at the boundary, not coerced.** The min-cut
    machinery in Phase 4 depends on exact arithmetic, because it compares a
    max-flow value against a sum of capacities and asserts they are equal.
    Floats make that assertion unreliable (``0.1 + 0.2 != 0.3``). Rather than
    accept ``4.0`` and hope no ``4.5`` ever arrives, the loader refuses all
    floats and says so clearly. Everything downstream can then assume ints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .model import Edge, Fix, Network, Node

# The attacker-effort scale. 1 is trivial, 10 is very hard. The range is
# enforced rather than merely documented, because a difficulty of 500 would
# silently dominate every shortest path and the report would look broken
# instead of looking wrong.
DIFFICULTY_MIN = 1
DIFFICULTY_MAX = 10

DEFAULT_FIX_COST = 1
DEFAULT_NODE_TYPE = "host"
DEFAULT_TECHNIQUE = "unspecified"

_TOP_LEVEL_KEYS = frozenset({"name", "nodes", "entry", "targets", "edges"})
_NODE_KEYS = frozenset({"id", "label", "type"})
_EDGE_KEYS = frozenset({"from", "to", "difficulty", "technique", "fix"})
_FIX_KEYS = frozenset({"name", "cost"})


class NetworkValidationError(ValueError):
    """Raised when a network model is structurally invalid.

    Carries every problem found, so the caller can print them all at once.
    """

    def __init__(self, source: str, problems: list[str]) -> None:
        self.source = source
        self.problems = list(problems)
        super().__init__(str(self))

    def __str__(self) -> str:
        count = len(self.problems)
        noun = "problem" if count == 1 else "problems"
        lines = [f"{self.source}: {count} {noun} found"]
        lines.extend(f"  {p}" for p in self.problems)
        return "\n".join(lines)


class _Problems:
    """Accumulates validation messages keyed by their JSON path."""

    def __init__(self) -> None:
        self._items: list[str] = []

    def add(self, path: str, message: str) -> None:
        self._items.append(f"{path}: {message}")

    def __bool__(self) -> bool:
        return bool(self._items)

    @property
    def items(self) -> list[str]:
        return list(self._items)


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------


def load_network(path: str | Path) -> Network:
    """Read and validate a network model from a JSON file.

    Raises FileNotFoundError, NetworkValidationError, or ValueError for
    malformed JSON -- each with a message that names the file.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"network file not found: {path}") from None
    except IsADirectoryError:
        raise IsADirectoryError(f"expected a JSON file but {path} is a directory") from None

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        # Surface line and column; "Expecting ',' delimiter" alone is useless
        # when the file is 200 lines long.
        raise ValueError(
            f"{path}: not valid JSON (line {exc.lineno}, column {exc.colno}): {exc.msg}"
        ) from None

    return parse_network(data, source=str(path), default_name=path.stem)


def parse_network(
    data: Any,
    *,
    source: str = "<input>",
    default_name: str = "unnamed",
) -> Network:
    """Validate an already-parsed JSON object and build a Network."""
    problems = _Problems()

    if not isinstance(data, dict):
        raise NetworkValidationError(
            source,
            [f"<root>: expected a JSON object, got {_type_name(data)}"],
        )

    _check_unknown_keys(data, _TOP_LEVEL_KEYS, "<root>", problems)

    name = _parse_name(data.get("name"), default_name, problems)
    nodes = _parse_nodes(data.get("nodes", _MISSING), problems)

    # Edge and entry/target validation needs the set of legal ids. If the node
    # list itself was unusable we skip reference checking rather than emit a
    # cascade of "unknown node" errors that are really one error.
    known_ids = frozenset(n.id for n in nodes) if nodes is not None else None

    entry = _parse_id_list(data.get("entry", _MISSING), "entry", known_ids, problems)
    targets = _parse_id_list(data.get("targets", _MISSING), "targets", known_ids, problems)
    edges = _parse_edges(data.get("edges", _MISSING), known_ids, problems)

    if entry and targets:
        _check_entry_target_overlap(entry, targets, problems)

    if problems:
        raise NetworkValidationError(source, problems.items)

    # The guards above guarantee these are populated; assert to make that
    # contract explicit for type checkers and for the next reader.
    assert nodes is not None and entry is not None and targets is not None
    assert edges is not None
    return Network(name=name, nodes=nodes, entry=entry, targets=targets, edges=edges)


# --------------------------------------------------------------------------
# Field parsers
# --------------------------------------------------------------------------

# Sentinel distinguishing "key absent" from "key present but null".
_MISSING = object()


def _parse_name(raw: Any, default: str, problems: _Problems) -> str:
    if raw is None:
        return default
    if not isinstance(raw, str) or not raw.strip():
        problems.add("name", f"must be a non-empty string, got {_describe(raw)}")
        return default
    return raw


def _parse_nodes(raw: Any, problems: _Problems) -> tuple[Node, ...] | None:
    entries = _require_list(raw, "nodes", problems, allow_empty=False)
    if entries is None:
        return None

    nodes: list[Node] = []
    first_seen: dict[str, int] = {}

    for index, item in enumerate(entries):
        path = f"nodes[{index}]"
        obj = _require_mapping(item, path, problems)
        if obj is None:
            continue
        _check_unknown_keys(obj, _NODE_KEYS, path, problems)

        node_id = _require_nonempty_str(obj.get("id", _MISSING), f"{path}.id", problems)
        if node_id is None:
            continue

        if node_id in first_seen:
            problems.add(
                f"{path}.id",
                f"duplicate node id {node_id!r} (first seen at nodes[{first_seen[node_id]}])",
            )
            continue
        first_seen[node_id] = index

        # label and type are optional conveniences; a model that only lists
        # ids is still a valid model.
        label = _optional_str(obj.get("label"), f"{path}.label", problems, default=node_id)
        node_type = _optional_str(
            obj.get("type"), f"{path}.type", problems, default=DEFAULT_NODE_TYPE
        )
        nodes.append(Node(id=node_id, label=label, type=node_type))

    return tuple(nodes)


def _parse_id_list(
    raw: Any,
    field_name: str,
    known_ids: frozenset[str] | None,
    problems: _Problems,
) -> tuple[str, ...] | None:
    """Validate 'entry' or 'targets': a non-empty list of known node ids."""
    entries = _require_list(raw, field_name, problems, allow_empty=False)
    if entries is None:
        return None

    ids: list[str] = []
    for index, item in enumerate(entries):
        path = f"{field_name}[{index}]"
        node_id = _require_nonempty_str(item, path, problems)
        if node_id is None:
            continue
        if node_id in ids:
            problems.add(path, f"duplicate {field_name} id {node_id!r}")
            continue
        if known_ids is not None and node_id not in known_ids:
            problems.add(path, f"unknown node id {node_id!r}{_known(known_ids)}")
            continue
        ids.append(node_id)

    return tuple(ids)


def _parse_edges(
    raw: Any,
    known_ids: frozenset[str] | None,
    problems: _Problems,
) -> tuple[Edge, ...] | None:
    entries = _require_list(raw, "edges", problems, allow_empty=False)
    if entries is None:
        return None

    edges: list[Edge] = []
    for index, item in enumerate(entries):
        path = f"edges[{index}]"
        obj = _require_mapping(item, path, problems)
        if obj is None:
            continue
        _check_unknown_keys(obj, _EDGE_KEYS, path, problems)

        source = _require_node_ref(obj.get("from", _MISSING), f"{path}.from", known_ids, problems)
        target = _require_node_ref(obj.get("to", _MISSING), f"{path}.to", known_ids, problems)
        difficulty = _require_int_in_range(
            obj.get("difficulty", _MISSING),
            f"{path}.difficulty",
            DIFFICULTY_MIN,
            DIFFICULTY_MAX,
            problems,
        )

        if source is not None and target is not None and source == target:
            # A self-loop cannot lengthen or shorten any attack path, so it is
            # almost certainly a copy-paste error rather than a model of
            # anything real.
            problems.add(path, f"self-loop on node {source!r}: 'from' and 'to' are the same")
            continue

        technique = _optional_str(
            obj.get("technique"), f"{path}.technique", problems, default=DEFAULT_TECHNIQUE
        )
        fix = _parse_fix(obj.get("fix"), f"{path}.fix", source, target, problems)

        if source is None or target is None or difficulty is None or fix is None:
            continue
        edges.append(
            Edge(
                source=source,
                target=target,
                difficulty=difficulty,
                technique=technique,
                fix=fix,
            )
        )

    return tuple(edges)


def _parse_fix(
    raw: Any,
    path: str,
    source: str | None,
    target: str | None,
    problems: _Problems,
) -> Fix | None:
    """Parse an edge's fix, synthesising a default when it is absent.

    An edge with no stated fix still has to be cuttable, otherwise the
    min-cut would be forced to route around it and would report a more
    expensive answer than the true one.
    """
    if raw is None:
        label = f"{source} -> {target}" if source and target else "this link"
        return Fix(name=f"Block {label}", cost=DEFAULT_FIX_COST)

    obj = _require_mapping(raw, path, problems)
    if obj is None:
        return None
    _check_unknown_keys(obj, _FIX_KEYS, path, problems)

    # A fix needs a name because that name is the handle the user types for
    # `chokepoint whatif --apply "<name>"`. An unnamed fix cannot be applied.
    name = _require_nonempty_str(obj.get("name", _MISSING), f"{path}.name", problems)

    cost_raw = obj.get("cost")
    if cost_raw is None:
        cost: int | None = DEFAULT_FIX_COST
    else:
        # No upper bound: unlike difficulty, fix cost is an open-ended budget
        # figure (engineer-days, money) with no natural ceiling.
        cost = _require_int_in_range(cost_raw, f"{path}.cost", 1, None, problems)

    if name is None or cost is None:
        return None
    return Fix(name=name, cost=cost)


def _check_entry_target_overlap(
    entry: tuple[str, ...], targets: tuple[str, ...], problems: _Problems
) -> None:
    """Reject a node that is both an entry point and a target.

    If the attacker starts on the crown jewel, the cheapest attack path costs
    nothing and no set of edge fixes can disconnect the two -- the max-flow is
    unbounded. That is a modelling mistake, and catching it here gives a
    better message than an infinite min-cut would.
    """
    for node_id in sorted(set(entry) & set(targets)):
        problems.add(
            "entry/targets",
            f"node {node_id!r} is both an entry point and a target; "
            "the attacker would already be on the asset, so no set of fixes can "
            "disconnect them",
        )


# --------------------------------------------------------------------------
# Low-level type checks
# --------------------------------------------------------------------------


def _require_mapping(value: Any, path: str, problems: _Problems) -> dict[str, Any] | None:
    if value is _MISSING or value is None:
        problems.add(path, "is required")
        return None
    if not isinstance(value, dict):
        problems.add(path, f"expected an object, got {_type_name(value)}")
        return None
    return value


def _require_list(
    value: Any, path: str, problems: _Problems, *, allow_empty: bool
) -> list[Any] | None:
    if value is _MISSING or value is None:
        problems.add(path, "is required")
        return None
    if not isinstance(value, list):
        problems.add(path, f"expected a list, got {_type_name(value)}")
        return None
    if not value and not allow_empty:
        problems.add(path, "must not be empty")
        return None
    return value


def _require_nonempty_str(value: Any, path: str, problems: _Problems) -> str | None:
    if value is _MISSING or value is None:
        problems.add(path, "is required")
        return None
    if not isinstance(value, str):
        problems.add(path, f"expected a string, got {_type_name(value)}")
        return None
    if not value.strip():
        problems.add(path, "must not be empty")
        return None
    return value


def _optional_str(value: Any, path: str, problems: _Problems, *, default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        problems.add(path, f"expected a string, got {_type_name(value)}")
        return default
    if not value.strip():
        problems.add(path, "must not be empty (omit the key to accept the default)")
        return default
    return value


def _require_int_in_range(
    value: Any,
    path: str,
    minimum: int,
    maximum: int | None,
    problems: _Problems,
) -> int | None:
    """Require a whole number in [minimum, maximum]; maximum None means no cap."""
    bound = f"from {minimum} to {maximum}" if maximum is not None else f"of at least {minimum}"

    if value is _MISSING or value is None:
        problems.add(path, "is required")
        return None
    # bool is a subclass of int in Python, and JSON `true` parses to True.
    # Checked before the int check so `"difficulty": true` is rejected.
    if isinstance(value, bool):
        problems.add(path, f"must be a whole number {bound}, got boolean {str(value).lower()}")
        return None
    if isinstance(value, float):
        problems.add(
            path,
            f"must be a whole number {bound}, got the decimal {value!r}. "
            "ChokePoint keeps all cost arithmetic exact, so decimals are not accepted.",
        )
        return None
    if not isinstance(value, int):
        problems.add(path, f"must be a whole number {bound}, got {_type_name(value)}")
        return None
    if value < minimum or (maximum is not None and value > maximum):
        problems.add(path, f"must be a whole number {bound}, got {value}")
        return None
    return value


def _require_node_ref(
    value: Any, path: str, known_ids: frozenset[str] | None, problems: _Problems
) -> str | None:
    node_id = _require_nonempty_str(value, path, problems)
    if node_id is None:
        return None
    if known_ids is not None and node_id not in known_ids:
        problems.add(path, f"unknown node id {node_id!r}{_known(known_ids)}")
        return None
    return node_id


def _check_unknown_keys(
    obj: dict[str, Any], allowed: frozenset[str], path: str, problems: _Problems
) -> None:
    """Reject unrecognised keys, to catch typos like "difficlty".

    Keys beginning with an underscore are reserved for the model author's own
    annotations and are always allowed, so the format stays commentable.
    """
    unknown = sorted(k for k in obj if k not in allowed and not k.startswith("_"))
    if unknown:
        problems.add(
            path,
            f"unknown key(s) {', '.join(repr(k) for k in unknown)}; "
            f"allowed here: {', '.join(sorted(allowed))} "
            "(prefix a key with '_' to keep it as a comment)",
        )


def _known(known_ids: frozenset[str]) -> str:
    """Render the legal node ids, abbreviated so the message stays readable."""
    ids = sorted(known_ids)
    shown = ", ".join(ids[:8])
    if len(ids) > 8:
        shown += f", ... ({len(ids)} total)"
    return f"; known ids are: {shown}" if ids else ""


def _type_name(value: Any) -> str:
    return {
        dict: "an object",
        list: "a list",
        str: "a string",
        bool: "a boolean",
        int: "a number",
        float: "a number",
        type(None): "null",
    }.get(type(value), type(value).__name__)


def _describe(value: Any) -> str:
    return "an empty string" if value == "" else _type_name(value)
