import json

import numpy as np
import pytest

Image = pytest.importorskip("PIL.Image")

from leaklens.cli import main
from leaklens.images import dhash, image_overlap


def _picture(seed, size=128):
    # smooth random shapes: a coarse random grid upscaled, so the image has structure dHash can see
    rng = np.random.default_rng(seed)
    coarse = Image.fromarray(rng.integers(0, 256, (6, 6, 3), dtype=np.uint8))
    return coarse.resize((size, size), Image.Resampling.BICUBIC)


@pytest.fixture
def splits(tmp_path):
    train, test = tmp_path / "train", tmp_path / "test"
    (train / "cats").mkdir(parents=True)
    test.mkdir()
    for i in range(4):
        _picture(i).save(train / "cats" / f"{i}.png")
    # near duplicate of train image 2: smaller, brighter and re-encoded as JPEG
    near = _picture(2).resize((96, 96)).point(lambda v: min(255, v + 12))
    near.save(test / "copy.jpg", quality=80)
    for i in (10, 11):
        _picture(i).save(test / f"new{i}.png")
    (test / "notes.txt").write_text("not an image", encoding="utf-8")
    return train, test


def test_dhash_is_64_bits_and_stable(splits):
    train, _ = splits
    h = dhash(train / "cats" / "0.png")
    assert 0 <= h < 2**64
    assert h == dhash(train / "cats" / "0.png")


def test_near_duplicate_found_distinct_not(splits):
    train, test = splits
    fs, pairs, stats = image_overlap(train, test)
    assert stats == {"train_images": 4, "test_images": 3, "unreadable": 0, "near_duplicates": 1}
    assert [(p["test"], p["train"]) for p in pairs] == [("copy.jpg", "cats/2.png")]
    assert [(f.severity, f.kind) for f in fs] == [("high", "image_overlap")]


def test_clean_splits(splits):
    train, test = splits
    (test / "copy.jpg").unlink()
    fs, pairs, _ = image_overlap(train, test)
    assert fs == [] and pairs == []


def test_cli_text_and_json(splits, capsys):
    train, test = splits
    assert main(["images", str(train), str(test)]) == 1
    out = capsys.readouterr().out
    assert "[HIGH] image_overlap" in out
    assert "copy.jpg ~ cats/2.png" in out
    assert main(["images", str(train), str(test), "--json"]) == 1
    rep = json.loads(capsys.readouterr().out)
    assert rep["trust_score"] < 100
    assert rep["near_duplicates"] == len(rep["pairs"]) == 1
    assert rep["findings"][0]["kind"] == "image_overlap"
