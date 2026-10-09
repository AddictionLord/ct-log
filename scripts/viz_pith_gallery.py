import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HALF = 20
SCALE = 8
COLS = 5


def main() -> None:
    parser = argparse.ArgumentParser(description="Pith gallery: human click vs model, crops around the pith.")
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--image_root", type=Path, required=True, help="Replaces the 'data/kwp_ds_v3' prefix.")
    parser.add_argument("--n_random", type=int, default=24)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    rows = json.load(open(args.points))
    for row in rows:
        row["dist"] = float(np.hypot(row["pred"][0] - row["gt"][0], row["pred"][1] - row["gt"][1]))
    worst = sorted((r for r in rows if r["dist"] > 1), key=lambda r: -r["dist"])
    rest = [r for r in rows if r["dist"] <= 1]
    picks = [rest[i] for i in np.linspace(0, len(rest) - 1, args.n_random).astype(int)]
    chosen = worst + sorted(picks, key=lambda r: r["path"])
    figure = render(chosen, args.image_root, len(worst))
    figure.write_html(args.out + ".html", include_plotlyjs="cdn")
    figure.write_image(args.out + ".png", width=1100, height=figure.layout.height, scale=1.5)
    print("wrote %s.{html,png}: %d slices (%d with error > 1 px)" % (args.out, len(chosen), len(worst)))


def render(chosen: List[dict], image_root: Path, n_worst: int) -> go.Figure:
    n_rows = -(-len(chosen) // COLS)
    titles = []
    for i, row in enumerate(chosen):
        tag = "error" if i < n_worst else "sample"
        titles.append("%s · %s · %.2f px" % (Path(row["path"]).stem, tag, row["dist"]))
    figure = make_subplots(
        rows=n_rows, cols=COLS, subplot_titles=titles, horizontal_spacing=0.01, vertical_spacing=0.035
    )
    for i, row in enumerate(chosen):
        path = image_root / Path(row["path"]).relative_to("data/kwp_ds_v3")
        r, c = i // COLS + 1, i % COLS + 1
        x0, y0 = row["gt"][0] - HALF, row["gt"][1] - HALF
        figure.add_trace(go.Image(z=crop(path, row["gt"])), row=r, col=c)
        figure.add_trace(marker((HALF, HALF), "square-open", "#22c84a", 16, 3), row=r, col=c)
        figure.add_trace(
            marker((row["pred"][0] - x0, row["pred"][1] - y0), "x-thin-open", "#ff2d2d", 14, 3), row=r, col=c
        )
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    figure.update_annotations(font_size=11)
    figure.update_layout(
        height=250 * n_rows,
        width=1100,
        margin=dict(l=5, r=5, t=95, b=5),
        showlegend=False,
        title="Log 10 pith, 41x41 px around the click (x8). Green square = human click, red cross = U-Net ensemble.<br>"
        "First %d: all slices with error > 1 px (1.41 px = diagonal neighbour); rest: evenly spaced sample." % n_worst,
    )
    return figure


def marker(pixel: Tuple[int, int], symbol: str, color: str, size: int, width: int) -> go.Scatter:
    x, y = pixel[0] * SCALE + SCALE / 2, pixel[1] * SCALE + SCALE / 2
    return go.Scatter(
        x=[x],
        y=[y],
        mode="markers",
        marker=dict(symbol=symbol, color=color, size=size, line=dict(color=color, width=width)),
        hoverinfo="skip",
    )


def crop(path: Path, gt: List[int]) -> np.ndarray:
    gray = np.asarray(Image.open(path).convert("L"))
    cx, cy = gt
    window = np.zeros((2 * HALF + 1, 2 * HALF + 1), dtype=np.uint8)
    y0, x0 = cy - HALF, cx - HALF
    ys, xs = slice(max(y0, 0), min(cy + HALF + 1, gray.shape[0])), slice(max(x0, 0), min(cx + HALF + 1, gray.shape[1]))
    window[ys.start - y0 : ys.stop - y0, xs.start - x0 : xs.stop - x0] = gray[ys, xs]
    return np.repeat(np.kron(window, np.ones((SCALE, SCALE), dtype=np.uint8))[..., None], 3, axis=2)


if __name__ == "__main__":
    main()
