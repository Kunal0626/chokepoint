"""The augmented graph that every algorithm in ChokePoint runs on.

A network model can have several entry points and several targets. Every
algorithm here -- shortest path, k-shortest paths, max-flow -- is defined for a
*single* source and a *single* sink. Rather than special-case the multi-entry
and multi-target situation in three different algorithms, we do it once here:
add a virtual super-source with a free arc to every entry node, and a virtual
super-sink with a free arc from every target node.

Why that is the right move rather than running one analysis per
(entry, target) pair and combining the results:

*   For shortest paths the two approaches agree, but the super-source version
    does one Dijkstra run instead of |entry| x |targets| of them.
*   For the min-cut in Phase 4 they do *not* agree. The union of per-pair
    min-cuts is not in general a min-cut of the whole problem, so the per-pair
    approach would recommend more fixes than are actually needed. Since the
    whole point of the tool is "the smallest set of fixes", that would be
    wrong in exactly the way that matters most.

The virtual arcs carry weight 0 and, in Phase 4, effectively infinite
capacity. Weight 0 because the attacker's effort starts at their first real
exploit -- merely "being on the internet" is free. Infinite capacity so that
the min-cut can never be placed on a virtual arc, which would produce the
mathematically valid but operationally useless advice "disconnect the
imaginary source from the internet".
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import Network


@dataclass(frozen=True)
class Arc:
    """One directed arc of the augmented graph.

    ``edge_index`` points into ``Network.edges``, or is None for the virtual
    arcs that attach the super-source and super-sink. Keeping the index rather
    than a copy of the edge is what lets parallel edges stay distinct: two
    arcs can share a (source, target) pair and still be told apart, which a
    nested-dict adjacency could not do.
    """

    source: str
    target: str
    edge_index: int | None

    @property
    def is_virtual(self) -> bool:
        return self.edge_index is None


@dataclass(frozen=True)
class AugmentedGraph:
    """A Network plus its virtual source and sink, in adjacency-list form."""

    network: Network
    arcs: tuple[Arc, ...]
    # node id -> indices of the arcs leaving it. Every node of the augmented
    # graph is a key, so lookups never need a .get() with a default.
    adjacency: dict[str, tuple[int, ...]]
    source: str
    sink: str

    def difficulty_of(self, arc_index: int) -> int:
        """Attacker effort for traversing this arc; 0 for virtual arcs."""
        edge_index = self.arcs[arc_index].edge_index
        if edge_index is None:
            return 0
        return self.network.edges[edge_index].difficulty

    @property
    def nodes(self) -> tuple[str, ...]:
        """Every node including the virtual source and sink."""
        return tuple(self.adjacency)

    def edge_indices_of(self, arc_indices: tuple[int, ...]) -> tuple[int, ...]:
        """Drop the virtual arcs, leaving indices into Network.edges."""
        return tuple(
            arc.edge_index
            for arc in (self.arcs[i] for i in arc_indices)
            if arc.edge_index is not None
        )


def build_augmented_graph(network: Network) -> AugmentedGraph:
    """Wrap a validated Network in its single-source, single-sink form."""
    taken = set(network.node_ids)
    source = _unique_name("__source__", taken)
    taken.add(source)
    sink = _unique_name("__sink__", taken)

    arcs: list[Arc] = []
    # Order matters only for determinism, not correctness: virtual entry arcs,
    # then the real edges in file order, then virtual target arcs. Stable order
    # means stable tie-breaking in Dijkstra, which means reproducible reports.
    for entry in network.entry:
        arcs.append(Arc(source=source, target=entry, edge_index=None))
    for index, edge in enumerate(network.edges):
        arcs.append(Arc(source=edge.source, target=edge.target, edge_index=index))
    for target in network.targets:
        arcs.append(Arc(source=target, target=sink, edge_index=None))

    outgoing: dict[str, list[int]] = {node_id: [] for node_id in network.node_ids}
    outgoing[source] = []
    outgoing[sink] = []
    for arc_index, arc in enumerate(arcs):
        outgoing[arc.source].append(arc_index)

    return AugmentedGraph(
        network=network,
        arcs=tuple(arcs),
        adjacency={node: tuple(indices) for node, indices in outgoing.items()},
        source=source,
        sink=sink,
    )


def _unique_name(base: str, taken: set[str]) -> str:
    """A name not already used by a real node.

    A model is free to contain a node literally called "__source__". Rather
    than reject it, grow the sentinel until it is unique -- the alternative is
    a validation rule that would surprise whoever hit it.
    """
    name = base
    while name in taken:
        name += "_"
    return name
