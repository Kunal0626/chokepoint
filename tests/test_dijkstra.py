"""Tests for the hand-written Dijkstra and the augmented-graph construction.

The oracle tests at the bottom are the ones that would actually catch a subtle
bug. The hand-written cases above them exist to pin down specific behaviours
and to fail with a readable message when something breaks.
"""

from __future__ import annotations

import networkx as nx
import pytest

from chokepoint.dijkstra import (
    AttackPath,
    cheapest_attack_path,
    dijkstra,
    shortest_route,
)
from chokepoint.graph import build_augmented_graph
from helpers import (
    assert_path_consistent,
    augmented,
    network,
    networkx_best_pairwise_cost,
    random_network,
    to_networkx,
)

# ---------------------------------------------------------------------------
# Basic shortest-path behaviour
# ---------------------------------------------------------------------------


def test_straight_line():
    net = network(["a", "b", "c"], ["a"], ["c"], [("a", "b", 2), ("b", "c", 3)])
    path = cheapest_attack_path(net)
    assert path is not None
    assert path.nodes == ("a", "b", "c")
    assert path.total_difficulty == 5
    assert path.hop_count == 2
    assert_path_consistent(net, path)


def test_picks_the_cheaper_of_two_routes():
    net = network(
        ["a", "b", "c", "d"],
        ["a"],
        ["d"],
        [("a", "b", 1), ("b", "d", 1), ("a", "c", 1), ("c", "d", 9)],
    )
    path = cheapest_attack_path(net)
    assert path is not None
    assert path.nodes == ("a", "b", "d")
    assert path.total_difficulty == 2


def test_cheap_first_step_is_not_followed_blindly():
    """A greedy walk takes a->b at cost 1 and ends up paying 10."""
    net = network(
        ["a", "b", "c", "d"],
        ["a"],
        ["d"],
        [("a", "b", 1), ("b", "d", 9), ("a", "c", 3), ("c", "d", 2)],
    )
    path = cheapest_attack_path(net)
    assert path is not None
    assert path.nodes == ("a", "c", "d")
    assert path.total_difficulty == 5


def test_parallel_edges_the_cheaper_one_wins():
    """Two techniques between the same hosts; the attacker takes the easy one."""
    net = network(["a", "b"], ["a"], ["b"], [("a", "b", 8), ("a", "b", 2)])
    path = cheapest_attack_path(net)
    assert path is not None
    assert path.total_difficulty == 2
    # Edge index 1 is the difficulty-2 edge, so the report names the right
    # technique and therefore the right fix.
    assert path.edge_indices == (1,)


def test_virtual_arcs_add_nothing_to_the_cost():
    """Reaching the entry point and being on the target are both free."""
    net = network(["a", "b"], ["a"], ["b"], [("a", "b", 4)])
    path = cheapest_attack_path(net)
    assert path is not None
    assert path.total_difficulty == 4
    assert path.nodes == ("a", "b")


# ---------------------------------------------------------------------------
# Unreachable targets
# ---------------------------------------------------------------------------


def test_unreachable_target_returns_none():
    """Not an error: a network with no attack path is a good result."""
    net = network(["a", "b", "c"], ["a"], ["c"], [("a", "b", 1)])
    assert cheapest_attack_path(net) is None


def test_entry_with_no_outgoing_edges():
    net = network(["a", "b", "c"], ["a"], ["c"], [("b", "c", 1)])
    assert cheapest_attack_path(net) is None


def test_edges_pointing_the_wrong_way_do_not_help():
    """Edges are directed; c->b->a is not a route from a to c."""
    net = network(["a", "b", "c"], ["a"], ["c"], [("c", "b", 1), ("b", "a", 1)])
    assert cheapest_attack_path(net) is None


# ---------------------------------------------------------------------------
# Multiple entries and targets
# ---------------------------------------------------------------------------


def test_multiple_entries_cheapest_one_wins():
    net = network(
        ["a", "b", "t"],
        ["a", "b"],
        ["t"],
        [("a", "t", 7), ("b", "t", 2)],
    )
    path = cheapest_attack_path(net)
    assert path is not None
    assert path.nodes == ("b", "t")
    assert path.total_difficulty == 2


def test_multiple_targets_cheapest_one_wins():
    net = network(
        ["a", "x", "y"],
        ["a"],
        ["x", "y"],
        [("a", "x", 6), ("a", "y", 3)],
    )
    path = cheapest_attack_path(net)
    assert path is not None
    assert path.nodes == ("a", "y")
    assert path.total_difficulty == 3


def test_one_reachable_target_among_several():
    net = network(
        ["a", "b", "x", "y"],
        ["a"],
        ["x", "y"],
        [("a", "b", 2), ("b", "y", 2)],
    )
    path = cheapest_attack_path(net)
    assert path is not None
    assert path.nodes == ("a", "b", "y")


# ---------------------------------------------------------------------------
# The augmented graph
# ---------------------------------------------------------------------------


def test_sentinel_names_avoid_colliding_with_real_nodes():
    """A model may legitimately contain a node called __source__."""
    net = network(["__source__", "b"], ["__source__"], ["b"], [("__source__", "b", 2)])
    graph = build_augmented_graph(net)
    assert graph.source != "__source__"
    assert graph.source not in net.node_ids
    assert graph.sink not in net.node_ids

    path = cheapest_attack_path(net)
    assert path is not None
    # The real node must survive the virtual-endpoint stripping.
    assert path.nodes == ("__source__", "b")


def test_every_node_is_a_key_in_the_adjacency_map():
    net = network(["a", "b", "c"], ["a"], ["c"], [("a", "b", 1)])
    graph = build_augmented_graph(net)
    for node in list(net.node_ids) + [graph.source, graph.sink]:
        assert node in graph.adjacency


def test_arc_count_is_edges_plus_virtuals():
    net = network(
        ["a", "b", "x", "y"], ["a", "b"], ["x", "y"], [("a", "x", 1), ("b", "y", 1)]
    )
    graph = build_augmented_graph(net)
    assert len(graph.arcs) == 2 + 2 + 2  # 2 entries + 2 edges + 2 targets


def test_edge_indices_of_strips_virtual_arcs():
    net = network(["a", "b"], ["a"], ["b"], [("a", "b", 3)])
    graph = build_augmented_graph(net)
    all_arcs = tuple(range(len(graph.arcs)))
    assert graph.edge_indices_of(all_arcs) == (0,)


# ---------------------------------------------------------------------------
# The dijkstra() primitive, including the knobs Yen's needs
# ---------------------------------------------------------------------------


def test_distance_map_on_a_known_graph():
    net = network(
        ["a", "b", "c", "d"],
        ["a"],
        ["d"],
        [("a", "b", 2), ("a", "c", 5), ("b", "c", 1), ("c", "d", 4)],
    )
    graph = augmented(net)
    distance, _ = dijkstra(graph, "a")
    assert distance["a"] == 0
    assert distance["b"] == 2
    assert distance["c"] == 3  # via b, not the direct 5
    assert distance["d"] == 7


def test_unreachable_nodes_are_absent_from_the_distance_map():
    net = network(["a", "b", "c"], ["a"], ["c"], [("a", "b", 1)])
    distance, _ = dijkstra(augmented(net), "a")
    assert "c" not in distance


def test_banned_arcs_force_a_detour():
    net = network(
        ["a", "b", "c", "d"],
        ["a"],
        ["d"],
        [("a", "d", 2), ("a", "b", 3), ("b", "d", 3)],
    )
    graph = augmented(net)
    direct = shortest_route(graph, "a", "d")
    assert direct == ((1,), 2)  # arc 1 is the a->d edge

    detoured = shortest_route(graph, "a", "d", banned_arcs={1})
    assert detoured is not None
    assert detoured[1] == 6


def test_banning_every_route_returns_none():
    net = network(["a", "b"], ["a"], ["b"], [("a", "b", 2)])
    graph = augmented(net)
    assert shortest_route(graph, "a", "b", banned_arcs={1}) is None


def test_banned_nodes_are_removed():
    net = network(
        ["a", "b", "c", "d"],
        ["a"],
        ["d"],
        [("a", "b", 1), ("b", "d", 1), ("a", "c", 5), ("c", "d", 5)],
    )
    graph = augmented(net)
    assert shortest_route(graph, "a", "d")[1] == 2
    assert shortest_route(graph, "a", "d", banned_nodes={"b"})[1] == 10


def test_banning_the_start_node_yields_nothing():
    net = network(["a", "b"], ["a"], ["b"], [("a", "b", 2)])
    graph = augmented(net)
    distance, predecessor = dijkstra(graph, "a", banned_nodes={"a"})
    assert distance == {}
    assert predecessor == {}


def test_results_are_deterministic_across_runs():
    """Identical input must give an identical report, ties included."""
    net = network(
        ["a", "b", "c", "d"],
        ["a"],
        ["d"],
        [("a", "b", 2), ("a", "c", 2), ("b", "d", 2), ("c", "d", 2)],
    )
    first = cheapest_attack_path(net)
    for _ in range(5):
        assert cheapest_attack_path(net) == first


def test_attack_path_is_hashable():
    """Yen's algorithm in Phase 3 puts these in a set to de-duplicate."""
    path = AttackPath(nodes=("a", "b"), edge_indices=(0,), total_difficulty=3)
    assert len({path, path}) == 1


# ---------------------------------------------------------------------------
# networkx oracle: hundreds of random graphs, fixed seeds
# ---------------------------------------------------------------------------

SEEDS = range(200)


@pytest.mark.parametrize("seed", SEEDS)
def test_matches_networkx_on_the_augmented_graph(seed):
    """Our Dijkstra against networkx's, on identical topology."""
    net = random_network(seed)
    graph = augmented(net)
    mirror = to_networkx(graph)

    ours = shortest_route(graph, graph.source, graph.sink)
    theirs = nx.single_source_dijkstra_path_length(mirror, graph.source, weight="difficulty")

    if graph.sink not in theirs:
        assert ours is None, f"seed {seed}: we found a path networkx says does not exist"
    else:
        assert ours is not None, f"seed {seed}: we missed a path networkx found"
        assert ours[1] == theirs[graph.sink], f"seed {seed}: cost disagreement"


@pytest.mark.parametrize("seed", SEEDS)
def test_super_source_matches_the_best_entry_target_pair(seed):
    """Independent check on the super-source construction itself.

    networkx runs on the *raw* graph here, with no virtual nodes, so this
    would catch a bug in build_augmented_graph that the test above -- which
    mirrors our own construction -- could not.
    """
    net = random_network(seed, n_entry=2, n_targets=2, n_nodes=14, n_edges=30)
    ours = cheapest_attack_path(net)
    theirs = networkx_best_pairwise_cost(net)

    if theirs is None:
        assert ours is None, f"seed {seed}: no entry-target pair is connected"
    else:
        assert ours is not None, f"seed {seed}: we missed a reachable target"
        assert ours.total_difficulty == theirs, f"seed {seed}: cost disagreement"


@pytest.mark.parametrize("seed", SEEDS)
def test_returned_paths_are_internally_consistent(seed):
    """Catches a right-cost-wrong-route bug, which a cost check alone misses."""
    net = random_network(seed, n_nodes=14, n_edges=30)
    path = cheapest_attack_path(net)
    if path is not None:
        assert_path_consistent(net, path)


@pytest.mark.parametrize("seed", range(40))
def test_dense_and_sparse_graphs_both_agree(seed):
    for n_nodes, n_edges in ((6, 40), (30, 35), (50, 300)):
        net = random_network(seed, n_nodes=n_nodes, n_edges=n_edges)
        graph = augmented(net)
        ours = shortest_route(graph, graph.source, graph.sink)
        theirs = nx.single_source_dijkstra_path_length(
            to_networkx(graph), graph.source, weight="difficulty"
        )
        expected = theirs.get(graph.sink)
        assert (ours[1] if ours else None) == expected, f"seed {seed} at {n_nodes}/{n_edges}"
