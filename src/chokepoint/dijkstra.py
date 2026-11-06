"""Dijkstra's shortest-path algorithm, written by hand.

The attacker's cost of a path is the sum of ``difficulty`` along it, so the
cheapest attack path is a shortest path from the super-source to the
super-sink in the augmented graph.

Why Dijkstra needs non-negative weights
---------------------------------------
Dijkstra's correctness rests on one claim: when a node is popped from the
priority queue, its recorded distance is final and no path discovered later
can improve it. That claim holds only if every edge weight is >= 0, because
extending a path can then never make it cheaper.

Break the assumption and the algorithm quietly returns a wrong answer rather
than failing. Take three nodes with A->B costing 2, A->C costing 5, and
C->B costing -4. Dijkstra pops B at distance 2 and settles it. Later it
reaches C at 5 and finds B again at 5 + (-4) = 1 -- cheaper than the value it
already committed to. B has been settled, so the improvement is discarded and
the reported distance to B is 2 when the truth is 1.

This matters for the *model*, not just the code. It is why "this step actually
makes the attacker's life easier" cannot be expressed as a negative
difficulty. The 1-to-10 scale the loader enforces keeps every real weight
positive, and the virtual super-source and super-sink arcs weigh exactly 0 --
which is fine, since the requirement is non-negative, not strictly positive.
If the model ever needed negative weights, the algorithm would have to change
to Bellman-Ford, at O(V*E) instead of O((V+E) log V).

Why heapq and not a decrease-key heap
-------------------------------------
The textbook formulation calls ``decrease-key`` when a shorter route to a
node is found. Python's ``heapq`` has no decrease-key, so this implementation
uses the standard lazy alternative: push a new (distance, node) entry and
ignore any entry that is popped after the node has already been settled. The
heap can hold up to O(E) entries instead of O(V), which makes the bound
O((V+E) log E) -- and since E <= V^2, log E <= 2 log V, so that is the same
O((V+E) log V) up to a constant. Simpler code for no asymptotic cost.

Maintaining an index into the heap and sifting manually would keep it at O(V)
entries, at the price of about fifty lines of error-prone bookkeeping.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable
from dataclasses import dataclass

from .graph import AugmentedGraph, build_augmented_graph
from .model import Network

# Yen's algorithm in Phase 3 needs to run Dijkstra with some arcs and some
# nodes temporarily removed. Those parameters live here rather than in a
# wrapper because removing an arc has to happen inside the scan loop; filtering
# the graph beforehand would mean rebuilding the adjacency lists on every one
# of Yen's O(k*V) spur searches.
_NO_ARCS: frozenset[int] = frozenset()
_NO_NODES: frozenset[str] = frozenset()


@dataclass(frozen=True)
class AttackPath:
    """One attack route, in terms the report can print.

    ``nodes`` and ``edge_indices`` describe the real network only: the virtual
    super-source and super-sink have been stripped. Frozen and tuple-based so
    that Yen's algorithm can put these in a set to de-duplicate candidates.
    """

    nodes: tuple[str, ...]
    edge_indices: tuple[int, ...]
    total_difficulty: int

    @property
    def hop_count(self) -> int:
        return len(self.edge_indices)


def dijkstra(
    graph: AugmentedGraph,
    start: str,
    *,
    banned_arcs: Iterable[int] = _NO_ARCS,
    banned_nodes: Iterable[str] = _NO_NODES,
) -> tuple[dict[str, int], dict[str, int]]:
    """Shortest distances from ``start``, and the arc used to reach each node.

    Returns ``(distance, predecessor_arc)``. A node absent from ``distance``
    is unreachable. ``banned_arcs`` and ``banned_nodes`` are treated as if
    removed from the graph, which is what Yen's algorithm needs in Phase 3.
    """
    banned_arcs = frozenset(banned_arcs)
    banned_nodes = frozenset(banned_nodes)

    if start in banned_nodes:
        return {}, {}

    distance: dict[str, int] = {start: 0}
    predecessor: dict[str, int] = {}
    settled: set[str] = set()

    # Heap entries are (distance, node). Comparing the node id as a tiebreaker
    # is deliberate: it makes the pop order total and therefore deterministic,
    # so the same model always yields the same path. Reports that change
    # between runs on identical input are impossible to trust.
    heap: list[tuple[int, str]] = [(0, start)]

    while heap:
        current_distance, node = heapq.heappop(heap)
        # A stale entry: this node was already settled via a shorter route.
        # This is the lazy stand-in for decrease-key.
        if node in settled:
            continue
        settled.add(node)

        for arc_index in graph.adjacency[node]:
            if arc_index in banned_arcs:
                continue
            neighbour = graph.arcs[arc_index].target
            if neighbour in banned_nodes or neighbour in settled:
                continue

            candidate = current_distance + graph.difficulty_of(arc_index)
            known = distance.get(neighbour)
            # Strictly-less keeps the first route found on a tie, which given
            # the deterministic pop order above is itself deterministic.
            if known is None or candidate < known:
                distance[neighbour] = candidate
                predecessor[neighbour] = arc_index
                heapq.heappush(heap, (candidate, neighbour))

    return distance, predecessor


def shortest_route(
    graph: AugmentedGraph,
    start: str,
    end: str,
    *,
    banned_arcs: Iterable[int] = _NO_ARCS,
    banned_nodes: Iterable[str] = _NO_NODES,
) -> tuple[tuple[int, ...], int] | None:
    """The cheapest arc sequence from ``start`` to ``end``, with its cost.

    Returns None when ``end`` is unreachable. This is the primitive Yen's
    algorithm builds on, which is why it deals in arc indices rather than in
    the prettier AttackPath.
    """
    distance, predecessor = dijkstra(
        graph, start, banned_arcs=banned_arcs, banned_nodes=banned_nodes
    )
    if end not in distance:
        return None
    return _walk_back(graph, predecessor, start, end), distance[end]


def to_attack_path(graph: AugmentedGraph, arc_indices: tuple[int, ...], cost: int) -> AttackPath:
    """Convert an arc sequence over the augmented graph into a reportable path."""
    nodes = [graph.arcs[i].source for i in arc_indices]
    if arc_indices:
        nodes.append(graph.arcs[arc_indices[-1]].target)
    # Strip the virtual endpoints; the user never asked about them.
    real_nodes = tuple(n for n in nodes if n not in (graph.source, graph.sink))
    return AttackPath(
        nodes=real_nodes,
        edge_indices=graph.edge_indices_of(arc_indices),
        total_difficulty=cost,
    )


def cheapest_attack_path(network: Network) -> AttackPath | None:
    """The cheapest route from any entry point to any target.

    Returns None when no target is reachable from any entry point -- a real
    and reportable outcome, not an error.
    """
    graph = build_augmented_graph(network)
    route = shortest_route(graph, graph.source, graph.sink)
    if route is None:
        return None
    arc_indices, cost = route
    return to_attack_path(graph, arc_indices, cost)


def _walk_back(
    graph: AugmentedGraph, predecessor: dict[str, int], start: str, end: str
) -> tuple[int, ...]:
    """Rebuild the arc sequence from the predecessor map, start to end."""
    arcs: list[int] = []
    node = end
    while node != start:
        arc_index = predecessor[node]
        arcs.append(arc_index)
        node = graph.arcs[arc_index].source
    arcs.reverse()
    return tuple(arcs)
