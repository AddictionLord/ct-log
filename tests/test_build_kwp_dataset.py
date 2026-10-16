import json
import pathlib
from typing import Dict, List

from ann_pipeline.knot.data_prep import decode_bitmap
from ann_pipeline.scripts.build_kwp_dataset import (
    BG,
    KNOT,
    WOOD,
    ClassSource,
    KnotSource,
    auto_annotation,
    build_parser,
    validate,
    wood_mask,
)
from ann_pipeline.scripts.upload_ct_logs_to_phases import _largest_cc_fill
from ann_pipeline.wood.detectors import threshold_peel
import numpy as np
from PIL import Image
import pytest

REQUIRED = ["--version", "v", "--out_root", "o", "--auto_logs", "3", "--human_logs", "10"]
LEGACY = ["--wood_source", "peel", "--knot_rule", "prob", "--pith_source", "obb"]


@pytest.fixture
def log_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """A 64x64 slice: bright disc (wood) with a darker rim pixel ring, plus model outputs for it.

    The class map marks the disc as wood, a 6x6 square as knot and a detached 3x3 speck as wood;
    the knot probability PNG is 200 on the square and 0 elsewhere.
    """
    yy, xx = np.mgrid[:64, :64]
    disc = (yy - 32) ** 2 + (xx - 32) ** 2 <= 20**2
    gray = np.where(disc, 180, 10).astype(np.uint8)
    gray[(yy - 32) ** 2 + (xx - 32) ** 2 == 20**2] = 50
    img = tmp_path / "img"
    img.mkdir()
    Image.fromarray(gray).save(img / "page_000.tiff")
    model = tmp_path / "model" / "3"
    model.mkdir(parents=True)
    cls = np.where(disc, WOOD, BG).astype(np.uint8)
    cls[28:34, 28:34] = KNOT
    cls[2:5, 2:5] = WOOD
    Image.fromarray(cls).save(model / "page_000.cls.png")
    prob = np.zeros((64, 64), np.uint8)
    prob[28:34, 28:34] = 200
    Image.fromarray(prob).save(model / "page_000.png")
    return tmp_path


def masks(ann: dict) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for obj in ann["objects"]:
        if "bitmap" not in obj:
            continue
        m = np.zeros((64, 64), bool)
        x, y = obj["bitmap"]["origin"]
        b = decode_bitmap(obj["bitmap"]["data"]).astype(bool)
        m[y : y + b.shape[0], x : x + b.shape[1]] = b
        out[obj["classTitle"]] = out.get(obj["classTitle"], np.zeros((64, 64), bool)) | m
    return out


def points(ann: dict) -> List[List[int]]:
    return [o["points"]["exterior"][0] for o in ann["objects"] if o["classTitle"] == "Pith"]


def test_defaults_take_all_classes_from_the_model() -> None:
    """Without flags the builder uses model wood, argmax knots and model pith."""
    args = build_parser().parse_args(REQUIRED)
    assert (args.wood_source, args.knot_rule, args.pith_source) == ("model", "argmax", "model")


def test_defaults_require_model_root() -> None:
    """The model defaults refuse to run without --model_root."""
    with pytest.raises(ValueError):
        validate(build_parser().parse_args(REQUIRED))


def test_legacy_flags_need_no_model_outputs() -> None:
    """peel / prob / obb is the old self-contained path and needs no external roots."""
    validate(build_parser().parse_args(REQUIRED + LEGACY))


def test_model_wood_is_filled_largest_component() -> None:
    """Model wood keeps the largest non-background component, fills its holes and drops specks."""
    cls = np.zeros((20, 20), np.uint8)
    cls[2:18, 2:18] = WOOD
    cls[8:10, 8:10] = BG
    cls[0, 19] = WOOD
    wood = wood_mask(np.zeros((20, 20), np.uint8), np.zeros((20, 20), bool), cls, "model")
    assert wood[8, 8] and not wood[0, 19] and wood.sum() == 16 * 16


def test_peel_wood_matches_the_legacy_formula(log_dir: pathlib.Path) -> None:
    """--wood_source peel is exactly largest_cc_fill(threshold_peel | knot)."""
    gray = np.array(Image.open(log_dir / "img" / "page_000.tiff"))
    knot = np.zeros_like(gray, bool)
    knot[30:32, 30:32] = True
    expected = _largest_cc_fill(threshold_peel(gray, 30, 60, 5).astype(bool) | knot).astype(bool)
    assert np.array_equal(wood_mask(gray, knot, None, "peel"), expected)


def test_legacy_path_uses_probability_knots_peel_wood_and_given_pith(log_dir: pathlib.Path) -> None:
    """prob knots from the PNG, peel wood and the external pith point, as in the pre-2026-10-16 builds."""
    knots = KnotSource(log_dir / "model" / "3", 128)
    ann = auto_annotation(log_dir / "img" / "page_000.tiff", None, None, 0.25, 800, {"page_000.tiff": [31, 32]}, knots)
    m = masks(ann)
    gray = np.array(Image.open(log_dir / "img" / "page_000.tiff"))
    expected_wood = wood_mask(gray, m["Knot"], None, "peel")
    assert m["Knot"].sum() == 36 and np.array_equal(m["Wood"], expected_wood) and points(ann) == [[31, 32]]


def test_default_path_uses_argmax_knots_and_model_wood(log_dir: pathlib.Path) -> None:
    """argmax knots and model wood come from the class map; the detached speck is not wood."""
    classes = ClassSource(log_dir / "model" / "3")
    ann = auto_annotation(
        log_dir / "img" / "page_000.tiff",
        None,
        None,
        0.25,
        800,
        {"page_000.tiff": [30, 30]},
        None,
        classes,
        "model",
        "argmax",
    )
    m = masks(ann)
    cls = np.array(Image.open(log_dir / "model" / "3" / "page_000.cls.png"))
    assert np.array_equal(m["Knot"], cls == KNOT)
    assert not m["Wood"][3, 3] and m["Wood"][32, 32] and points(ann) == [[30, 30]]
    assert json.dumps(ann)
