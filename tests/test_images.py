import json

import numpy as np
import pytest

Image = pytest.importorskip("PIL.Image")

from leaklens import images
from leaklens.cli import main
from leaklens.images import ImageBudget, ImageRefused, dhash, hash_dir, image_overlap


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
    assert stats == {
        "train_images": 4,
        "test_images": 3,
        "unreadable": 0,
        "refused": {},
        "decoded_pixels": 107520,
        "near_duplicates": 1,
    }
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


def _saved(tmp_path, seed, name="shot.png", size=32):
    path = tmp_path / name
    _picture(seed, size).save(path)
    return path


def test_oversized_compressed_file_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(images, "MAX_IMAGE_BYTES", 10)
    path = _saved(tmp_path, 1, size=8)
    with pytest.raises(ImageRefused, match="too_large"):
        dhash(path)


def test_excessive_dimensions_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(images, "MAX_IMAGE_SIDE", 10)
    path = _saved(tmp_path, 2, size=32)
    with pytest.raises(ImageRefused, match="too_large_dimensions"):
        dhash(path)


def test_excessive_pixel_count_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(images, "MAX_IMAGE_PIXELS", 100)
    path = _saved(tmp_path, 3, size=32)
    with pytest.raises(ImageRefused, match="too_many_pixels"):
        dhash(path)


def test_excessive_animation_frames_are_refused(tmp_path, monkeypatch):
    frames = [_picture(i, 16) for i in range(3)]
    path = tmp_path / "anim.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=10, loop=0)
    monkeypatch.setattr(images, "MAX_IMAGE_FRAMES", 2)
    with pytest.raises(ImageRefused, match="too_many_frames"):
        dhash(path)


def test_unsupported_format_is_refused(tmp_path):
    path = tmp_path / "plain.ppm"
    Image.new("RGB", (8, 8)).save(path)
    with pytest.raises(ImageRefused, match="unsupported_format"):
        dhash(path)


def test_shared_budget_refuses_later_images(tmp_path):
    first = _saved(tmp_path, 4)
    second = _saved(tmp_path, 5)
    budget = ImageBudget(max_pixels=2000)
    assert dhash(first, budget=budget) >= 0
    with pytest.raises(ImageRefused, match="budget_exhausted"):
        dhash(second, budget=budget)
    assert budget.refused == {}


def test_hash_dir_records_refusal_reasons(tmp_path):
    _saved(tmp_path, 6, "a.png")
    _saved(tmp_path, 7, "b.png")
    budget = ImageBudget(max_pixels=2000)
    hashes, unreadable = hash_dir(tmp_path, budget=budget)
    assert list(hashes) == ["a.png"]
    assert unreadable == 0
    assert budget.refused == {"budget_exhausted": 1}


def test_truncated_image_is_unreadable_not_refused(tmp_path):
    original = _saved(tmp_path, 8)
    data = original.read_bytes()
    path = tmp_path / "truncated.png"
    path.write_bytes(data[: len(data) // 2])
    original.unlink()
    hashes, unreadable = hash_dir(tmp_path)
    assert hashes == {}
    assert unreadable == 1


def test_malformed_header_is_unreadable_not_refused(tmp_path):
    (tmp_path / "broken.png").write_bytes(b"this is not an image")
    hashes, unreadable = hash_dir(tmp_path)
    assert hashes == {}
    assert unreadable == 1


@pytest.mark.parametrize("limit", [100, 10_000])
def test_pillow_bomb_warning_and_error_become_refusals(tmp_path, monkeypatch, limit):
    path = _saved(tmp_path, 9, size=128)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", limit)
    with pytest.raises(ImageRefused, match="decompression_bomb"):
        dhash(path)


def test_failed_decode_releases_the_file_handle(tmp_path):
    path = _saved(tmp_path, 11, size=64)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])  # header parses, decode fails
    _, unreadable = hash_dir(tmp_path)
    assert unreadable == 1
    path.unlink()  # a leaked handle keeps the file locked on Windows


def test_failed_decode_does_not_charge_the_budget(tmp_path):
    path = _saved(tmp_path, 12, size=64)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    budget = ImageBudget()
    _, unreadable = hash_dir(tmp_path, budget=budget)
    assert unreadable == 1
    assert budget.pixels_used == 0
    assert budget.refused == {}


def test_corrupt_png_chunk_does_not_crash_the_audit(tmp_path):
    path = _saved(tmp_path, 13, size=32)
    data = path.read_bytes()
    index = data.index(b"IDAT")
    path.write_bytes(data[: index - 4] + b"\x00\x00\x00\x03" + data[index:])
    hashes, unreadable = hash_dir(tmp_path)
    assert hashes == {}
    assert unreadable == 1


def test_decoder_exception_never_crashes_the_audit(tmp_path, monkeypatch):
    _saved(tmp_path, 14)

    def boom(*args, **kwargs):
        raise SyntaxError("broken PNG file (chunk)")

    monkeypatch.setattr(images, "dhash", boom)
    hashes, unreadable = hash_dir(tmp_path)
    assert hashes == {}
    assert unreadable == 1


def test_shipped_budgets_match_the_documentation():
    assert images.MAX_IMAGE_BYTES == 64 * 1024 * 1024
    assert images.MAX_IMAGE_SIDE == 20_000
    assert images.MAX_IMAGE_PIXELS == 64_000_000
    assert images.MAX_IMAGE_FRAMES == 64
    assert images.MAX_AUDIT_PIXELS == 2_000_000_000


def test_real_side_limit_boundary(tmp_path):
    ok = tmp_path / "ok.png"
    Image.new("L", (images.MAX_IMAGE_SIDE, 1)).save(ok)
    assert dhash(ok) >= 0

    bad = tmp_path / "bad.png"
    Image.new("L", (images.MAX_IMAGE_SIDE + 1, 1)).save(bad)
    with pytest.raises(ImageRefused, match="too_large_dimensions"):
        dhash(bad)


def test_real_frame_limit_boundary(tmp_path):
    frames = [
        Image.new("RGB", (8, 8), (i % 256, 0, 0))
        for i in range(images.MAX_IMAGE_FRAMES + 1)
    ]
    ok = tmp_path / "ok.gif"
    frames[0].save(ok, save_all=True, append_images=frames[1:-1], duration=10, loop=0)
    assert dhash(ok) >= 0

    bad = tmp_path / "bad.gif"
    frames[0].save(bad, save_all=True, append_images=frames[1:], duration=10, loop=0)
    with pytest.raises(ImageRefused, match="too_many_frames"):
        dhash(bad)


@pytest.mark.parametrize("suffix", [".png", ".jpg", ".bmp", ".gif", ".tif"])
def test_supported_formats_remain_hashable(tmp_path, suffix):
    path = tmp_path / f"valid{suffix}"
    _picture(10, 64).save(path)
    value = dhash(path)
    assert 0 <= value < 2**64


def test_audit_budget_spans_both_splits(splits):
    train, test = splits
    budget = ImageBudget(max_pixels=16384)
    fs, pairs, stats = image_overlap(train, test, budget=budget)
    assert fs == [] and pairs == []
    assert stats["train_images"] == 1
    assert stats["test_images"] == 0
    assert stats["refused"] == {"budget_exhausted": 6}
    assert stats["decoded_pixels"] == 16384


def test_cli_reports_refusal_reasons(splits, capsys, monkeypatch):
    train, test = splits
    monkeypatch.setattr(images, "MAX_IMAGE_PIXELS", 100)
    assert main(["images", str(train), str(test)]) == 0
    out = capsys.readouterr().out
    assert "refused 7 image(s): too_many_pixels: 7" in out
    assert main(["images", str(train), str(test), "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["refused"] == {"too_many_pixels": 7}
    assert rep["decoded_pixels"] == 0
