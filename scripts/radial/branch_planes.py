import argparse
import json
import logging
import math
import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import plotly.graph_objects as go
from scipy import ndimage as ndi
from scripts.radial.common import OUT, write_png
from scripts.radial.make_radial import load_volumes

logger = logging.getLogger(__name__)
HALF_WIDTH_PX = 30
PAD_PX = 15
CORE_PX = 1
BG_RANGE_PX = (6, 15)
SCALE = 3


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Branch-aligned (tilted) planes for thin knots: oracle rise from GT, contrast vs axial slices."
    )
    parser.add_argument("--root", type=pathlib.Path, default=OUT)
    parser.add_argument("--n_controls", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    knots = json.loads((args.root / "knots.json").read_text())
    pith = np.array(knots["gt_pith_xy"])
    ct, gt, prob, _ = load_volumes(args.root)
    cases = select_cases(knots["knots"], args.n_controls)
    out_dir = args.root / "branch_planes"
    results: List[Dict[str, object]] = []
    for kind, k in cases:
        partner = partner_slice(k)
        if partner is None:
            logger.info("skip %s: no labelled neighbour slice", k["id"])
            continue
        res = analyse(ct, gt, prob, pith, k, partner, kind)
        render(out_dir / ("%s__%s.png" % (kind, k["id"])), res.pop("_planes"))
        results.append(res)
        logger.info(
            "%-7s %-16s slope %+.3f slice/px  contrast z %.1f  partner %.1f  tilted %.1f  M4 p_max z %.2f",
            kind,
            k["id"],
            res["slope_slices_per_px"],
            res["contrast"]["axial_z"],
            res["contrast"]["axial_partner"],
            res["contrast"]["tilted"],
            res["m4_prob_max"]["axial_z"],
        )
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    figure(results).write_html(out_dir / "contrast.html", include_plotlyjs="cdn")


def select_cases(knots: List[Dict[str, object]], n_controls: int) -> List[Tuple[str, Dict[str, object]]]:
    gt = [k for k in knots if k["source"] == "gt" and k["thin"]]
    missed = [("missed", k) for k in gt if k["iou"] < 0.3]
    found = [k for k in gt if k["iou"] >= 0.5 and partner_slice(k) is not None]
    found.sort(key=lambda k: -(k["r_max_px"] - k["r_min_px"]))
    return missed + [("control", k) for k in found[:n_controls]]


def partner_slice(k: Dict[str, object]) -> Optional[int]:
    n = k["neighbours"]
    if n["gt_z-1"]:
        return int(k["z"]) - 1
    if n["gt_z+1"]:
        return int(k["z"]) + 1
    return None


def analyse(
    ct: np.ndarray,
    gt: np.ndarray,
    prob: np.ndarray,
    pith: np.ndarray,
    k: Dict[str, object],
    partner: int,
    kind: str,
) -> Dict[str, object]:
    z = int(k["z"])
    theta = math.radians(float(k["theta_deg"]))
    r_lo, r_hi, slope, z0 = fit_rise(gt, pith, theta, z, partner)
    s = np.arange(0, int(r_hi) + PAD_PX + 1, dtype=np.float64)
    t = np.arange(-HALF_WIDTH_PX, HALF_WIDTH_PX + 1, dtype=np.float64)
    planes = {
        "axial_partner": plane_coords(pith, theta, s, t, np.full_like(s, float(partner))),
        "axial_z": plane_coords(pith, theta, s, t, np.full_like(s, float(z))),
        "tilted": plane_coords(pith, theta, s, t, np.clip(z0 + slope * s, 0, ct.shape[0] - 1)),
    }
    sampled: Dict[str, Dict[str, np.ndarray]] = {}
    contrast: Dict[str, float] = {}
    p_max: Dict[str, float] = {}
    span = (s >= r_lo) & (s <= r_hi)
    for name, coords in planes.items():
        layers = {
            "ct": ndi.map_coordinates(ct.astype(np.float32), coords, order=1),
            "prob": ndi.map_coordinates(prob.astype(np.float32), coords, order=1) / 255.0,
            "gt": ndi.map_coordinates(gt, coords, order=0),
        }
        sampled[name] = layers
        contrast[name] = line_contrast(layers["ct"], t, span)
        p_max[name] = float(layers["prob"][:, span][np.abs(t) <= 2].max())
    return {
        "id": k["id"],
        "kind": kind,
        "z": z,
        "partner_z": partner,
        "theta_deg": float(k["theta_deg"]),
        "r_span_px": [float(r_lo), float(r_hi)],
        "slope_slices_per_px": float(slope),
        "rise_mm_per_100px": float(slope * 10.0 * 100.0),
        "contrast": contrast,
        "m4_prob_max": p_max,
        "_planes": sampled,
    }


def fit_rise(gt: np.ndarray, pith: np.ndarray, theta: float, z: int, partner: int) -> Tuple[float, float, float, float]:
    rs: List[np.ndarray] = []
    zs: List[np.ndarray] = []
    for zz in (z, partner):
        ys, xs = np.nonzero(gt[zz] == 2)
        dx = xs - pith[zz, 0]
        dy = ys - pith[zz, 1]
        along = dx * math.cos(theta) + dy * math.sin(theta)
        across = np.abs(-dx * math.sin(theta) + dy * math.cos(theta))
        keep = (across <= 6) & (along > 0)
        rs.append(along[keep])
        zs.append(np.full(keep.sum(), float(zz)))
    r = np.concatenate(rs)
    zz = np.concatenate(zs)
    if np.ptp(r) < 1:
        msg = "degenerate radial extent"
        raise ValueError(msg)
    slope, z0 = np.polyfit(r, zz, 1)
    return float(r.min()), float(r.max()), float(slope), float(z0)


def plane_coords(pith: np.ndarray, theta: float, s: np.ndarray, t: np.ndarray, zf: np.ndarray) -> np.ndarray:
    zi = np.clip(zf, 0, pith.shape[0] - 1)
    lo = np.floor(zi).astype(int)
    hi = np.minimum(lo + 1, pith.shape[0] - 1)
    w = zi - lo
    px = (1 - w) * pith[lo, 0] + w * pith[hi, 0]
    py = (1 - w) * pith[lo, 1] + w * pith[hi, 1]
    x = px[None, :] + s[None, :] * math.cos(theta) - t[:, None] * math.sin(theta)
    y = py[None, :] + s[None, :] * math.sin(theta) + t[:, None] * math.cos(theta)
    zz = np.broadcast_to(zi[None, :], x.shape)
    return np.stack([zz, y, x])


def line_contrast(img: np.ndarray, t: np.ndarray, span: np.ndarray) -> float:
    core = np.abs(t) <= CORE_PX
    bg = (np.abs(t) >= BG_RANGE_PX[0]) & (np.abs(t) <= BG_RANGE_PX[1])
    line = img[core][:, span].mean(axis=0)
    back = np.median(img[bg][:, span], axis=0)
    return float((line - back).mean())


def render(path: pathlib.Path, planes: Dict[str, Dict[str, np.ndarray]]) -> None:
    rows: List[np.ndarray] = []
    for name in ("axial_partner", "axial_z", "tilted"):
        layer = planes[name]
        grey = np.clip(layer["ct"], 0, 255).astype(np.uint8)
        ct_rgb = np.repeat(grey[..., None], 3, axis=2)
        over = ct_rgb.copy()
        edge = (layer["gt"] == 2) & ~ndi.binary_erosion(layer["gt"] == 2)
        over[edge] = (0, 255, 0)
        heat = np.zeros_like(ct_rgb)
        heat[..., 0] = np.clip(layer["prob"] * 255, 0, 255).astype(np.uint8)
        heat[..., 1] = (grey * 0.4).astype(np.uint8)
        heat[..., 2] = (grey * 0.4).astype(np.uint8)
        gap = np.full((ct_rgb.shape[0], 4, 3), 255, np.uint8)
        rows.append(np.concatenate([ct_rgb, gap, over, gap, heat], axis=1))
        rows.append(np.full((4, rows[-1].shape[1], 3), 255, np.uint8))
    img = np.concatenate(rows[:-1], axis=0)
    img = np.array(Image.fromarray(img).resize((img.shape[1] * SCALE, img.shape[0] * SCALE), Image.NEAREST))
    write_png(path, img)


def figure(results: List[Dict[str, object]]) -> go.Figure:
    ids = ["%s (%s)" % (r["id"].replace("gt_page_", ""), r["kind"]) for r in results]
    fig = go.Figure()
    for name, label in (
        ("axial_partner", "axial, partner slice"),
        ("axial_z", "axial, knot slice"),
        ("tilted", "tilted along branch"),
    ):
        fig.add_bar(name=label, x=ids, y=[r["contrast"][name] for r in results])
    fig.update_layout(
        barmode="group",
        title="Knot-line contrast (grey levels, core minus local background) per view",
        yaxis_title="contrast",
        template="plotly_white",
    )
    return fig


if __name__ == "__main__":
    main()
