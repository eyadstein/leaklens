"""Notebook execution-order detection."""

import json

from leaklens.scanner import scan_file


def _notebook(cells):
    return {"cells": cells}


def _code(source, count=None):
    return {
        "cell_type": "code",
        "execution_count": count,
        "source": source,
    }


def test_out_of_order_execution_is_reported(tmp_path):
    nb = _notebook(
        [
            _code(["x = load()\n"], 3),
            _code(["y = transform(x)\n"], 1),
            _code(["train_test_split(x, y, random_state=0)\n"], 2),
        ]
    )
    path = tmp_path / "ordered.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")
    findings = scan_file(path)
    order = [f for f in findings if f.kind == "notebook_out_of_order_execution"]
    assert order and order[0].severity == "low"
    assert "3, 1, 2" in order[0].detail


def test_in_order_execution_is_clean(tmp_path):
    nb = _notebook(
        [
            _code(["x = load()\n"], 1),
            _code(["y = transform(x)\n"], 2),
            _code(["train_test_split(x, y, random_state=0)\n"], 3),
        ]
    )
    path = tmp_path / "clean.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")
    assert not [
        f for f in scan_file(path) if f.kind == "notebook_out_of_order_execution"
    ]


def test_missing_execution_counts_are_not_guessed(tmp_path):
    nb = _notebook(
        [
            _code(["x = load()\n"], None),
            _code(["y = transform(x)\n"], None),
        ]
    )
    path = tmp_path / "nulls.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")
    assert not [
        f for f in scan_file(path) if f.kind == "notebook_out_of_order_execution"
    ]


def test_out_of_order_still_reports_other_leaks(tmp_path):
    nb = _notebook(
        [
            _code(
                [
                    "from sklearn.preprocessing import StandardScaler\n",
                    "sc = StandardScaler()\n",
                    "X = sc.fit_transform(X)\n",
                ],
                2,
            ),
            _code(
                [
                    "from sklearn.model_selection import train_test_split\n",
                    "train_test_split(X, y, random_state=0)\n",
                ],
                1,
            ),
        ]
    )
    path = tmp_path / "leak.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")
    kinds = {f.kind for f in scan_file(path)}
    assert "notebook_out_of_order_execution" in kinds
    assert "preprocess_before_split" in kinds
