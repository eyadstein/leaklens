"""Grouped AST index: walk-order equivalence, immutability and output parity."""

import ast

from leaklens import scanner
from leaklens.cvcheck import scan_cv
from leaklens.flow import scan_flow
from leaklens.scanner import _node_index, scan_source

NESTED = """\
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def outer():
    def inner():
        scaler = StandardScaler()
        return scaler.fit_transform(X)

    return inner

X_scaled = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
"""


def _payload(findings):
    return [finding.to_dict() for finding in findings]


def test_index_covers_every_node_once_in_walk_order():
    nodes = tuple(ast.walk(ast.parse(NESTED)))
    index = _node_index(nodes)

    grouped = [node for group in index.values() for node in group]
    assert len(grouped) == len(nodes)
    assert {id(node) for node in grouped} == {id(node) for node in nodes}
    # Grouping by exact type must equal the filtered walk each pass used before.
    for typ, group in index.items():
        assert group == [node for node in nodes if type(node) is typ]


def test_indexed_passes_match_plain_traversal():
    tree = ast.parse(NESTED)
    nodes = tuple(ast.walk(tree))
    keys = scanner._finding_keys(tree, nodes)
    index = _node_index(nodes)

    for scan in (scan_cv, scan_flow):
        plain = _payload(scan(NESTED, "file.py"))
        indexed = _payload(
            scan(NESTED, "file.py", tree=tree, keys=keys, nodes=nodes, index=index)
        )
        assert indexed == plain


def test_cached_tree_is_not_mutated():
    scanner._parse_cache.clear()
    scanner._parse_cache_chars = 0
    tree, _, nodes, _ = scanner._parse_source(NESTED)
    assert nodes is not None
    before = ast.dump(tree)

    scan_source(NESTED, "file.py")

    cached = scanner._parse_cache[next(iter(scanner._parse_cache))]
    assert cached[0] is tree
    assert ast.dump(tree) == before


def test_output_order_and_fingerprints_are_deterministic():
    first = _payload(scan_source(NESTED, "file.py"))
    second = _payload(scan_source(NESTED, "file.py"))

    assert first
    assert second == first


def test_nested_scopes_keep_distinct_stable_keys():
    keys = [f.stable_key for f in scan_source(NESTED, "file.py") if f.stable_key]

    assert keys
    assert len(set(keys)) == len(keys)


def test_malformed_source_still_reports_parse_error():
    assert [f.kind for f in scan_source("def broken(:", "file.py")] == ["parse_error"]
    assert scan_cv("def broken(:") == []
    assert scan_flow("def broken(:") == []
