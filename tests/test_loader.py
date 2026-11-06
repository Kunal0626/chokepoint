"""Tests for JSON loading and validation.

Most of these assert on the *text* of the error, not just that an error
happened. A validator whose messages are vague is only half built, and
"clear error messages" is a stated requirement, so the messages are part of
the contract being tested.
"""

from __future__ import annotations

import json

import pytest

from chokepoint.loader import (
    NetworkValidationError,
    load_network,
    parse_network,
)
from chokepoint.model import Fix, Network, Node


def base() -> dict:
    """The smallest network that passes validation: a -> b."""
    return {
        "name": "tiny",
        "nodes": [{"id": "a"}, {"id": "b"}],
        "entry": ["a"],
        "targets": ["b"],
        "edges": [{"from": "a", "to": "b", "difficulty": 3}],
    }


def problems(data: dict) -> list[str]:
    """Validate and return the collected problem strings."""
    with pytest.raises(NetworkValidationError) as exc:
        parse_network(data)
    return exc.value.problems


def assert_problem(data: dict, *fragments: str) -> None:
    """Assert that some single problem mentions every given fragment."""
    found = problems(data)
    assert any(all(f in p for f in fragments) for p in found), (
        f"no problem mentioned all of {fragments!r}; got:\n" + "\n".join(found)
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_parses_a_full_edge():
    data = base()
    data["nodes"] = [
        {"id": "internet", "label": "Internet", "type": "external"},
        {"id": "web-dmz", "label": "Web server (DMZ)", "type": "server"},
    ]
    data["entry"] = ["internet"]
    data["targets"] = ["web-dmz"]
    data["edges"] = [
        {
            "from": "internet",
            "to": "web-dmz",
            "difficulty": 4,
            "technique": "exploit unpatched web app",
            "fix": {"name": "Patch web app + WAF", "cost": 3},
        }
    ]

    net = parse_network(data)

    assert isinstance(net, Network)
    assert net.name == "tiny"
    assert net.nodes == (
        Node(id="internet", label="Internet", type="external"),
        Node(id="web-dmz", label="Web server (DMZ)", type="server"),
    )
    assert net.entry == ("internet",)
    assert net.targets == ("web-dmz",)
    assert len(net.edges) == 1

    edge = net.edges[0]
    assert edge.source == "internet"
    assert edge.target == "web-dmz"
    assert edge.difficulty == 4
    assert edge.technique == "exploit unpatched web app"
    assert edge.fix == Fix(name="Patch web app + WAF", cost=3)


def test_optional_fields_get_documented_defaults():
    net = parse_network(base())
    node = net.node("a")
    # label falls back to the id so reports always have something to print.
    assert node.label == "a"
    assert node.type == "host"

    edge = net.edges[0]
    assert edge.technique == "unspecified"
    # An edge with no stated fix must still be cuttable, or the min-cut would
    # be forced to route around it and would overstate the true cost.
    assert edge.fix == Fix(name="Block a -> b", cost=1)


def test_name_defaults_to_the_supplied_default():
    data = base()
    del data["name"]
    assert parse_network(data, default_name="from-filename").name == "from-filename"


@pytest.mark.parametrize("difficulty", [1, 5, 10])
def test_difficulty_bounds_are_inclusive(difficulty):
    data = base()
    data["edges"][0]["difficulty"] = difficulty
    assert parse_network(data).edges[0].difficulty == difficulty


def test_parallel_edges_are_preserved_separately():
    """Two techniques between the same hosts are two distinct ways in.

    Collapsing them would understate both the path count and the min-cut cost,
    so this is load-bearing for Phases 3 and 4.
    """
    data = base()
    data["edges"] = [
        {"from": "a", "to": "b", "difficulty": 2, "technique": "weak SSH password",
         "fix": {"name": "Rotate SSH keys", "cost": 1}},
        {"from": "a", "to": "b", "difficulty": 7, "technique": "SMB exposed",
         "fix": {"name": "Firewall SMB", "cost": 2}},
    ]
    net = parse_network(data)
    assert len(net.edges) == 2
    assert {e.technique for e in net.edges} == {"weak SSH password", "SMB exposed"}


def test_underscore_keys_are_allowed_as_comments():
    data = base()
    data["_note"] = "modelled from the Q3 network diagram"
    data["edges"][0]["_why"] = "difficulty 3 because the host is internet-facing"
    parse_network(data)  # must not raise


def test_network_is_immutable():
    net = parse_network(base())
    with pytest.raises(Exception):
        net.name = "something else"  # type: ignore[misc]


def test_network_lookup_helpers():
    net = parse_network(base())
    assert net.has_node("a")
    assert not net.has_node("nope")
    assert net.node_ids == ("a", "b")
    assert net.label_of("a") == "a"
    # Unknown ids degrade to the raw id rather than raising, so a report can
    # always render something.
    assert net.label_of("ghost") == "ghost"
    with pytest.raises(KeyError, match="no node with id 'ghost'"):
        net.node("ghost")


# ---------------------------------------------------------------------------
# Structural errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [[], "a string", 42, None])
def test_root_must_be_an_object(bad):
    with pytest.raises(NetworkValidationError, match="expected a JSON object"):
        parse_network(bad)


@pytest.mark.parametrize("field", ["nodes", "entry", "targets", "edges"])
def test_required_top_level_fields(field):
    data = base()
    del data[field]
    assert_problem(data, field, "is required")


@pytest.mark.parametrize("field", ["nodes", "entry", "targets", "edges"])
def test_required_lists_must_not_be_empty(field):
    data = base()
    data[field] = []
    assert_problem(data, field, "must not be empty")


@pytest.mark.parametrize("field", ["nodes", "entry", "targets", "edges"])
def test_required_lists_must_be_lists(field):
    data = base()
    data[field] = {"not": "a list"}
    assert_problem(data, field, "expected a list", "got an object")


def test_unknown_top_level_key_is_rejected():
    data = base()
    data["noodes"] = []
    assert_problem(data, "<root>", "'noodes'", "allowed here")


def test_unknown_edge_key_is_rejected():
    """A typo in an optional key would otherwise be silently ignored."""
    data = base()
    data["edges"][0]["dificulty"] = 4
    assert_problem(data, "edges[0]", "'dificulty'")


# ---------------------------------------------------------------------------
# Node errors
# ---------------------------------------------------------------------------


def test_duplicate_node_id_names_both_positions():
    data = base()
    data["nodes"].append({"id": "a"})
    assert_problem(data, "nodes[2].id", "duplicate node id 'a'", "nodes[0]")


def test_node_id_is_required():
    data = base()
    data["nodes"][1] = {"label": "nameless"}
    assert_problem(data, "nodes[1].id", "is required")


def test_node_id_must_not_be_blank():
    data = base()
    data["nodes"][1] = {"id": "   "}
    assert_problem(data, "nodes[1].id", "must not be empty")


def test_node_must_be_an_object():
    data = base()
    data["nodes"][1] = "b"
    assert_problem(data, "nodes[1]", "expected an object", "got a string")


# ---------------------------------------------------------------------------
# Reference errors
# ---------------------------------------------------------------------------


def test_unknown_node_in_edge_lists_the_known_ids():
    data = base()
    data["edges"][0]["to"] = "db-prod"
    assert_problem(data, "edges[0].to", "unknown node id 'db-prod'", "known ids are: a, b")


def test_unknown_node_in_entry():
    data = base()
    data["entry"] = ["ghost"]
    assert_problem(data, "entry[0]", "unknown node id 'ghost'")


def test_unknown_node_in_targets():
    data = base()
    data["targets"] = ["ghost"]
    assert_problem(data, "targets[0]", "unknown node id 'ghost'")


def test_duplicate_entry_id():
    data = base()
    data["entry"] = ["a", "a"]
    assert_problem(data, "entry[1]", "duplicate entry id 'a'")


def test_broken_node_list_does_not_cascade_into_edge_errors():
    """One real error should not produce a dozen downstream ones."""
    data = base()
    data["nodes"] = "oops"
    found = problems(data)
    assert any("nodes" in p and "expected a list" in p for p in found)
    assert not any("unknown node id" in p for p in found)


# ---------------------------------------------------------------------------
# Difficulty and cost errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("difficulty", [0, -1, -10])
def test_zero_or_negative_difficulty_is_rejected(difficulty):
    data = base()
    data["edges"][0]["difficulty"] = difficulty
    assert_problem(data, "edges[0].difficulty", "from 1 to 10", str(difficulty))


def test_difficulty_above_the_scale_is_rejected():
    data = base()
    data["edges"][0]["difficulty"] = 11
    assert_problem(data, "edges[0].difficulty", "from 1 to 10", "11")


def test_difficulty_is_required():
    data = base()
    del data["edges"][0]["difficulty"]
    assert_problem(data, "edges[0].difficulty", "is required")


def test_boolean_difficulty_is_rejected():
    """JSON `true` parses to Python True, and bool is a subclass of int."""
    data = base()
    data["edges"][0]["difficulty"] = True
    assert_problem(data, "edges[0].difficulty", "boolean true")


@pytest.mark.parametrize("difficulty", [4.5, 4.0])
def test_decimal_difficulty_is_rejected_even_when_whole(difficulty):
    """Exact arithmetic is a hard requirement, so floats stop at the door."""
    data = base()
    data["edges"][0]["difficulty"] = difficulty
    assert_problem(data, "edges[0].difficulty", "decimal", "exact")


def test_string_difficulty_is_rejected():
    data = base()
    data["edges"][0]["difficulty"] = "4"
    assert_problem(data, "edges[0].difficulty", "whole number", "got a string")


@pytest.mark.parametrize("cost", [0, -3])
def test_fix_cost_below_one_is_rejected(cost):
    """A zero-cost fix would give a min-cut of 0 and a meaningless answer."""
    data = base()
    data["edges"][0]["fix"] = {"name": "Free lunch", "cost": cost}
    assert_problem(data, "edges[0].fix.cost", "at least 1", str(cost))


def test_fix_cost_has_no_upper_bound():
    data = base()
    data["edges"][0]["fix"] = {"name": "Rebuild the network", "cost": 10_000}
    assert parse_network(data).edges[0].fix.cost == 10_000


def test_fix_cost_defaults_to_one_when_only_a_name_is_given():
    data = base()
    data["edges"][0]["fix"] = {"name": "Patch it"}
    assert parse_network(data).edges[0].fix == Fix(name="Patch it", cost=1)


def test_fix_must_have_a_name():
    data = base()
    data["edges"][0]["fix"] = {"cost": 2}
    assert_problem(data, "edges[0].fix.name", "is required")


def test_fix_must_be_an_object():
    data = base()
    data["edges"][0]["fix"] = "Patch it"
    assert_problem(data, "edges[0].fix", "expected an object", "got a string")


# ---------------------------------------------------------------------------
# Modelling errors
# ---------------------------------------------------------------------------


def test_self_loop_is_rejected():
    data = base()
    data["edges"][0] = {"from": "a", "to": "a", "difficulty": 2}
    assert_problem(data, "edges[0]", "self-loop on node 'a'")


def test_node_that_is_both_entry_and_target_is_rejected():
    """No set of edge fixes can disconnect a node from itself."""
    data = base()
    data["entry"] = ["a"]
    data["targets"] = ["a", "b"]
    assert_problem(data, "entry/targets", "'a'", "no set of fixes can disconnect")


# ---------------------------------------------------------------------------
# Error reporting behaviour
# ---------------------------------------------------------------------------


def test_every_problem_is_reported_at_once():
    """The point of the collector: no fix-one-line-and-rerun loop."""
    data = {
        "nodes": [{"id": "a"}, {"id": "a"}],
        "entry": ["ghost"],
        "targets": ["b"],
        "edges": [
            {"from": "a", "to": "nowhere", "difficulty": 0},
            {"from": "a", "to": "a", "difficulty": 3},
        ],
    }
    found = problems(data)
    assert len(found) >= 5, "\n".join(found)
    joined = "\n".join(found)
    for expected in ["duplicate node id", "entry[0]", "targets[0]", "difficulty"]:
        assert expected in joined


def test_error_message_counts_and_lists_the_problems():
    data = base()
    data["edges"][0]["difficulty"] = 0
    with pytest.raises(NetworkValidationError) as exc:
        parse_network(data, source="net.json")
    text = str(exc.value)
    assert text.startswith("net.json: 1 problem found")
    assert "edges[0].difficulty" in text


def test_error_message_pluralises():
    data = base()
    data["edges"][0]["difficulty"] = 0
    data["entry"] = ["ghost"]
    with pytest.raises(NetworkValidationError) as exc:
        parse_network(data, source="net.json")
    assert "2 problems found" in str(exc.value)


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------


def test_load_network_reads_a_file(tmp_path):
    path = tmp_path / "net.json"
    path.write_text(json.dumps(base()), encoding="utf-8")
    net = load_network(path)
    assert net.name == "tiny"
    assert len(net.edges) == 1


def test_load_network_defaults_name_to_the_file_stem(tmp_path):
    data = base()
    del data["name"]
    path = tmp_path / "small_company.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert load_network(path).name == "small_company"


def test_missing_file_names_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="network file not found"):
        load_network(tmp_path / "nope.json")


def test_directory_instead_of_file(tmp_path):
    with pytest.raises(IsADirectoryError, match="is a directory"):
        load_network(tmp_path)


def test_malformed_json_reports_line_and_column(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{\n  "name": "x",\n  "nodes": [,]\n}', encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_network(path)
    with pytest.raises(ValueError, match=r"line 3"):
        load_network(path)


def test_validation_error_names_the_file(tmp_path):
    data = base()
    data["edges"][0]["difficulty"] = 99
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(NetworkValidationError) as exc:
        load_network(path)
    assert str(path) in str(exc.value)
