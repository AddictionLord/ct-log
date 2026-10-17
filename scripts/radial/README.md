# Radial views of log 10 (phase 1)

Copy of `/mnt/D/datasets/ct_log/radial/log10/README.md` (Vostro). Thin-knot conclusions: [RADIAL.md](../../RADIAL.md).

Built by `scripts/radial/` in the ct-log repo. Run from the repo root, in this order:

```bash
python -m scripts.radial.build_layout    # images/, annotations/, predictions/, meta.json
python -m scripts.radial.knots           # knots.json
python -m scripts.radial.make_radial     # radial/ (+ assertion that GT knots land on the GT label)
python -m scripts.radial.gallery         # gallery/
# optional cross-check against 3D Slicer:
Slicer --no-splash --python-script scripts/radial/slicer_check.py <work>/slicer_check.json
python -m scripts.radial.slicer_compare --work <work>
```

Sources are only read, never modified. Their paths and sha256 hashes are in `meta.json`.

| Path | Content |
|---|---|
| `images/page_NNN.png` | grey uint8 778 × 778, channel 0 of `kwp_ds_v3/10/img/page_NNN.tiff` (R = G = B checked) |
| `annotations/page_NNN.png` | uint8 label map: 0 bg, 1 wood, 2 knot (knot over wood); 293 labelled pages |
| `annotations/pith.json` | human pith clicks `{page: [x, y]}`; `unlabelled` lists pages 000, 001, 002, 100, 291, 298 |
| `predictions/m4/page_NNN_prob.png` | M4 knot probability, uint8 = p × 255 (copy of `knot_probs_ens_v4/10`) |
| `predictions/m4/page_NNN_mask.png` | M4 knot mask 0/255 (mean prob ≥ 0.5 clipped to wood; from `report_20261010/pred_m8`) |
| `predictions/m4/pith.json` | M4 pith `[x, y]`. The npz order is (x, y): median 0 px, mean 0.38 px from the human click |
| `predictions/m5/` | empty; M5 full-log predictions are not on this machine yet |
| `knots.json` | every GT and M4 knot instance (8-connected per slice): centroid, minor axis, thin flag, angle θ and radius from the pith, Hungarian IoU, partner id, presence at z ± 1 |
| `radial/theta_XXX.X/` | one straightened radial plane per angle, see below |
| `radial/planes.json` | geometry, the plane list (uniform or knot-aligned, with knot ids) and the verification result |
| `radial/slicer_check.json` | numeric comparison of three planes against 3D Slicer reslices |
| `gallery/index.html` | thin-knot gallery; `tiles.json` holds the same numbers; `tiles/` holds the PNGs |
| `findings.md` | qualitative observations |

## Radial planes

- **Geometry.** Column j is the signed distance r = j − R from the slice's pith (R = 170 px, width 341). Positive r points in direction θ; negative r points in direction θ + 180°. Row z = slice z.
- **Sample point.** x = pith_x(z) + r cos θ, y = pith_y(z) + r sin θ. θ = 0° is +x (right) and θ = 90° is +y (down in the image).
- **Axis.** The axis is curved, one centre per slice:
  - top-level files use the human pith, linearly interpolated over unlabelled pages and held constant at the ends;
  - `m4axis/` uses M4's pith.
- **Layers.**
  - `ct`, `m4_prob`: bilinear;
  - `gt` (0/1/2), `m4_mask` (0/255): nearest.
  - GT rows of unlabelled pages are 0.
- **Size.** Native files are 341 × 299. The `_x10` files are 341 × 2981, so row 10·z = slice z exactly. They are linear in z for `ct`/`m4_prob` and nearest for `gt`/`m4_mask`.
- **Planes.** 90 uniform planes (0, 2, …, 178°) plus one knot-aligned plane per knot instance. Knot-aligned planes use θ = angle pith → centroid, rounded to 0.1°, and are shared when two knots round to the same angle. The knot then lies at positive r. 523 planes in total.
- **Checks.**
  - For all 272 GT knots, row z of the knot-aligned GT plane equals the axial label along the ray, and it contains knot pixels within the knot's [r_min, r_max].
  - A 3D Slicer reslice of three planes matches the numpy sampler: MAE 0.25 grey levels, i.e. uint8 rounding, and correlation 0.99998. The mirrored control (θ + 180°) gives MAE 25–47.
