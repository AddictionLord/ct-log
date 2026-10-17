# Radial views of log 10: findings (phase 1, qualitative)

Copy of `/mnt/D/datasets/ct_log/radial/log10/findings.md` (Vostro); relative links point into that folder.

**Scope.** These are observations from looking at the 59 tiles in [gallery/index.html](gallery/index.html), plus a few counts from [knots.json](knots.json). Nothing was trained and no fusion was measured. Model: M4 (two v4 U-Nets averaged).

**What the gallery contains.**
- 39 thin GT knots, split by M4's axial per-knot IoU:
  - 11 missed (all at IoU 0);
  - 2 poor;
  - 26 good.
- 38 thin M4 components:
  - 10 have no GT match;
  - 4 are matched to a GT knot that is thick;
  - the rest are matched to thin GT knots and appear in those knots' tiles.
- 6 thick GT controls.

## Observations

1. **A thin knot is already the whole branch in the axial slice.**
   - Thin knots are long rays from near the pith outwards: median radial extent 82 px, median width ≈ 5 px.
   - The branch runs nearly perpendicular to the log axis, so it lies almost in the axial plane.
   - In the knot-aligned radial plane it becomes a horizontal band only 1–2 slices tall.
   - So the radial view does not reveal more of a thin knot than the axial slice already shows. In the raw CT it is not more visible either: faint streaks are equally faint, and are thinner on screen because they are 1–2 rows of 10 mm.
   - Examples: [good 110_00](gallery/tiles/good__gt_page_110_00_radial.png), [extra 066_01](gallery/tiles/m4_extra__m4_page_066_01_radial.png).

2. **Thin GT knots come in pairs of adjacent slices.**
   - 35 of 39 thin GT knots have GT in exactly one neighbouring slice (z − 1 or z + 1), 1 has GT in both, and 3 are isolated.
   - In the radial plane, the GT is a continuous band 2 rows tall; it is not broken.
   - The two rows often differ in length: one slice holds the full ray, the other only a shorter piece. This is what a ~5 mm branch crossing a 10 mm slice boundary looks like.

3. **M4's misses are mostly this second, fainter slice.**
   - 10 of the 11 missed thin knots have GT and M4 knot pixels in the adjacent slice. In the radial plane, M4 covers one row of the 2-row band and drops the other.
   - The missed pieces are short (radial extent 13–56 px, one of 109 px) compared with the found ones (median 93 px).
   - The radial view makes this pattern obvious at a glance, which the axial crop does not. But the missing row looks just as faint in the radial raw image as in the axial one.
   - Example: [missed 026_00](gallery/tiles/missed__gt_page_026_00_radial.png).
   - The radial IoU of these tiles over ±10 rows (0.5–0.8) is inflated by the neighbouring row that M4 did find. Their row-z radial IoU is 0 for all 11.

4. **The radial view helps with context and thick knots.**
   - [missed 203_00](gallery/tiles/missed__gt_page_203_00_radial.png) is an isolated thin label near the bark. The radial plane shows a large knot starting about two slices later on the same ray, so the thin label is plausibly that branch's leading edge.
   - Thick knots ([129_00](gallery/tiles/thick__gt_page_129_00_radial.png)) appear as oblique bands across several slices (129_00 spans about 6). The branch's rise angle and continuity are clear there, with a 10 mm staircase.
   - GT and M4 are continuous along z for thick knots: 134 of 233 thick GT knots have GT in both neighbours.

5. **Several "extra" M4 components look like real knots or are trivial.**
   - [066_01](gallery/tiles/m4_extra__m4_page_066_01_axial.png) is a clear, unlabelled thin ray in the axial CT; it looks like a GT omission.
   - 269_02 and 280_00 are bright streaks next to a dark crack: ambiguous.
   - The rest are specks of 1–20 px in the bark or on the rim of a labelled knot.
   - The radial view doesn't help decide any of these.

## Where the 10 mm spacing makes the radial view too coarse

- **Thin knots.** A thin knot (< 8 mm) spans at most 2 rows. Its rise angle, its taper along z, and whether a faint second slice belongs to it cannot be resolved; the ×10 display only stretches 1–2 rows of data. A radial segmenter would see a thin knot as a 1–2-px-tall horizontal line. That is the same information as two axial rows, with no in-plane context across the branch.
- **Thick knots.** Here the radial view carries real 3D information (oblique band over many slices). But thick knots are not where M4 is weak.

## Read for phase 2

- **Low expected gain on thin knots.** For thin-knot recall, a radial-view segmenter (option 1) or 3D fusion of radial probabilities (option 2) is unlikely to add much at 10 mm. The signal they would add is "a knot exists in the adjacent slice on this ray", and M4 already sees that: its input is the 3-slice stack [z − 1, z, z + 1].
- **What looks more promising.**
  - Explicit z-consistency, e.g. lowering the threshold in slice z where a confident knot sits at the same angle and radius in z ± 1. Check first whether that adds anything beyond the 3-slice input.
  - Reviewing the second-slice labelling convention with the annotators, since 10 of the 11 misses are that case.
- **Worth doing anyway, cheaply.** Run the same gallery for M5 once its log-10 predictions exist, to see whether its 5 extra thin finds are these second-slice pieces.
