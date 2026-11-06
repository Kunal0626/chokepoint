"""Immutable data types for a modelled network.

Design notes that matter later:

*   Everything here is a frozen dataclass holding tuples, not lists. The
    what-if analysis in Phase 5 repeatedly derives modified copies of a
    network ("remove the edges this fix closes, then re-analyse"). If those
    copies shared mutable structure with the original, a bug in one branch of
    the analysis would silently corrupt every later branch. Immutability makes
    that class of bug impossible rather than merely unlikely.

*   Edges live in a tuple and are referred to elsewhere by their *index*.
    Parallel edges are legal and meaningful: two different techniques between
    the same pair of hosts are two separate ways in, each with its own fix. A
    nested-dict adjacency structure (``adj[u][v] = capacity``) would silently
    collapse them into one, which would understate both the number of attack
    paths and the cost of the min-cut. Index-based identity keeps them apart.

*   No field has a default. Defaulting is the loader's job, so that there is
    exactly one place in the codebase that decides what a missing ``technique``
    or a missing ``fix.cost`` means.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Fix:
    """A defensive action that closes one or more edges.

    ``name`` is the handle the user types for ``chokepoint whatif --apply``.
    Names are deliberately *not* required to be unique: one real-world fix
    ("enforce MFA on the VPN") can close several modelled links at once, and
    applying it should close all of them. The flip side is that a min-cut
    treats edges as independent and will therefore count a shared fix's cost
    once per edge it cuts; Phase 5 de-duplicates by name when totalling, and
    the docs record the discrepancy.
    """

    name: str
    cost: int


@dataclass(frozen=True)
class Node:
    """A host, network segment or external zone."""

    id: str
    label: str
    type: str


@dataclass(frozen=True)
class Edge:
    """A directed attacker move from one node to another.

    ``source``/``target`` rather than ``from``/``to`` because ``from`` is a
    Python keyword. The JSON keys stay ``from`` and ``to``; the loader
    translates.

    ``difficulty`` is attacker effort on a 1 (trivial) to 10 (very hard)
    scale and is what the shortest-path algorithms minimise. ``fix.cost`` is
    *defender* effort and is what the min-cut minimises. Two different numbers
    with two different owners, kept apart deliberately: conflating them is the
    easiest way to answer a question nobody asked.
    """

    source: str
    target: str
    difficulty: int
    technique: str
    fix: Fix


@dataclass(frozen=True)
class Network:
    """A validated network model: nodes, directed edges, entries and targets."""

    name: str
    nodes: tuple[Node, ...]
    entry: tuple[str, ...]
    targets: tuple[str, ...]
    edges: tuple[Edge, ...]

    # Built once in __post_init__ so report code can resolve ids to labels
    # without an O(n) scan per lookup. Excluded from equality and repr: it is
    # derived state, not part of the model's identity.
    _by_id: dict[str, Node] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        # object.__setattr__ is the standard escape hatch for populating
        # derived fields on a frozen dataclass.
        object.__setattr__(self, "_by_id", {n.id: n for n in self.nodes})

    def has_node(self, node_id: str) -> bool:
        return node_id in self._by_id

    def node(self, node_id: str) -> Node:
        """Return the node with this id, or raise KeyError with a clear message."""
        try:
            return self._by_id[node_id]
        except KeyError:
            raise KeyError(f"no node with id {node_id!r} in network {self.name!r}") from None

    def label_of(self, node_id: str) -> str:
        """Human-readable name for reports, falling back to the raw id."""
        node = self._by_id.get(node_id)
        return node.label if node is not None else node_id

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(n.id for n in self.nodes)
