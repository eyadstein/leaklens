"""Partition-role inference used by tuning and test-set-reuse detectors."""

import ast

from leaklens.partition import any_test, infer_roles, name_role


def roles(src):
    tree = ast.parse(src)
    assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)]
    return infer_roles(assigns)


def test_explicit_names():
    assert name_role("X_test") == "test"
    assert name_role("X_holdout") == "test"
    assert name_role("X_te") == "test"
    assert name_role("X_train") == "train"
    assert name_role("X_val") == "validation"


def test_unrelated_words_are_not_partitions():
    assert name_role("latest") is None
    assert name_role("contest") is None
    assert name_role("greatest_value") is None


def test_positional_unpack_is_inferred():
    got = roles("a, b, c, d = train_test_split(X, y, random_state=0)\n")
    assert got["a"] == "train" and got["b"] == "test"
    assert got["c"] == "train" and got["d"] == "test"


def test_two_output_unpack_is_inferred():
    got = roles("train, holdout = train_test_split(X, y, random_state=0)\n")
    assert got["train"] == "train" and got["holdout"] == "test"


def test_validation_split_is_not_guessed_as_test():
    got = roles("X_train, X_val, y_train, y_val = train_test_split(X, y)\n")
    assert got["X_val"] == "validation"
    assert not any_test(["X_val"], got)


def test_name_hints_win_over_positional():
    got = roles("X_test, X_train = train_test_split(X, y)\n")
    assert got["X_test"] == "test"
    assert got["X_train"] == "train"


def test_simple_aliases_are_propagated():
    got = roles(
        "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
        "Xt = X_test.copy()\n"
    )
    assert got["Xt"] == "test"
    assert any_test(["Xt"], got)


def test_any_test_without_roles_uses_names():
    assert any_test(["estimator_on_X_holdout"])
    assert not any_test(["X_train"])
