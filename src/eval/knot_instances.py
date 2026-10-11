from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage import measure

S8 = np.ones((3, 3), bool)
F1_THRESHOLDS = (0.3, 0.5)
AP_THRESHOLDS = tuple(np.round(np.arange(0.5, 0.96, 0.05), 2))
SIZE_BINS = (("lt150", 0, 150), ("150_1000", 150, 1000), ("gt1000", 1000, 10**9))
THIN_MINOR_AXIS = 8
RECALL_POINTS = np.linspace(0.0, 1.0, 101)


@dataclass
class _Slice:
    iou: np.ndarray
    scores: np.ndarray


@dataclass
class KnotInstanceEvaluator:
    """Instance-level knot metrics over 2D slices.

    Instances are 8-connected components of the GT and predicted knot masks of each slice
    (no size filtering; the prediction is clipped to its own wood mask when one is given).
    Per-slice Hungarian matching on IoU gives instance precision/recall/F1 at IoU >= 0.3 and
    0.5, mean per-GT IoU (unmatched GT = 0), thin recall (GT minor axis < 8 px) and recall by
    GT area. COCO-style single-class AP pools all predicted components over slices, sorts them
    by score and matches each greedily to the unmatched GT of its slice with the highest IoU;
    101-point interpolated AP at IoU 0.50 and averaged over 0.50:0.05:0.95. Without a score map
    every component scores 1.0 and AP degenerates to a single operating point (flagged).
    """

    score_reduce: Literal["mean", "max"] = "mean"
    slices: List[_Slice] = field(default_factory=list)
    hits: Dict[float, List[float]] = field(default_factory=lambda: {t: [] for t in F1_THRESHOLDS})
    n_gt: int = 0
    n_pred: int = 0
    size_hits: Dict[str, List[int]] = field(default_factory=lambda: {name: [0, 0] for name, _, _ in SIZE_BINS})
    thin_hits: List[int] = field(default_factory=lambda: [0, 0])
    has_scores: bool = True
    pixel_inter: int = 0
    pixel_union: int = 0

    def add(
        self,
        gt_knot: np.ndarray,
        pred_knot: np.ndarray,
        wood: Optional[np.ndarray] = None,
        score_map: Optional[np.ndarray] = None,
    ) -> None:
        """Add one slice.

        Args:
            gt_knot: [H, W] bool human knot mask.
            pred_knot: [H, W] bool predicted knot mask.
            wood: [H, W] bool predicted wood mask the prediction is clipped to (optional).
            score_map: [H, W] float per-pixel score (knot probability or detector confidence).
        """
        pred = pred_knot.astype(bool) & wood.astype(bool) if wood is not None else pred_knot.astype(bool)
        self.pixel_inter += int((pred & gt_knot.astype(bool)).sum())
        self.pixel_union += int((pred | gt_knot.astype(bool)).sum())
        gt_labels, n_gt = ndi.label(gt_knot.astype(bool), structure=S8)
        pred_labels, n_pred = ndi.label(pred, structure=S8)
        self.n_gt += n_gt
        self.n_pred += n_pred
        iou = iou_matrix(gt_labels, n_gt, pred_labels, n_pred)
        best = np.zeros(n_gt)
        if n_gt and n_pred:
            rows, cols = linear_sum_assignment(-iou)
            best[rows] = iou[rows, cols]
        for threshold in F1_THRESHOLDS:
            self.hits[threshold] += best.tolist()
        for region in measure.regionprops(gt_labels):
            hit = best[region.label - 1] >= 0.3
            for name, low, high in SIZE_BINS:
                if low <= region.area < high:
                    self.size_hits[name][0] += int(hit)
                    self.size_hits[name][1] += 1
            if region.axis_minor_length < THIN_MINOR_AXIS:
                self.thin_hits[0] += int(hit)
                self.thin_hits[1] += 1
        if score_map is None:
            self.has_scores = False
            scores = np.ones(n_pred)
        else:
            scores = component_scores(pred_labels, n_pred, score_map, self.score_reduce)
        self.slices.append(_Slice(iou=iou, scores=scores))

    def result(self) -> Dict[str, object]:
        """Aggregate metrics.

        Returns:
            Dict[str, object]: Instance F1 blocks per IoU threshold, per-GT IoU, thin and size
                recall, AP50 and AP50-95, and whether AP is degenerate (no scores).
        """
        out: Dict[str, object] = {
            "n_frames": len(self.slices),
            "n_gt": self.n_gt,
            "n_pred": self.n_pred,
            "pixel_iou": round(self.pixel_inter / self.pixel_union, 4) if self.pixel_union else None,
        }
        for threshold in F1_THRESHOLDS:
            best = np.array(self.hits[threshold])
            hit = best >= threshold
            tp = int(hit.sum())
            precision, recall = tp / max(self.n_pred, 1), tp / max(self.n_gt, 1)
            out["iou_%s" % threshold] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(2 * precision * recall / max(precision + recall, 1e-9), 4),
                "tp": tp,
                "mean_matched_iou": round(float(best[hit].mean()), 4) if hit.any() else None,
                "mean_per_gt_iou": round(float(np.where(hit, best, 0.0).mean()), 4) if len(best) else None,
            }
        out["recall_by_size_iou0.3"] = {
            name: {"recall": round(v[0] / max(v[1], 1), 4), "n": v[1]} for name, v in self.size_hits.items()
        }
        out["recall_thin_iou0.3"] = {
            "recall": round(self.thin_hits[0] / max(self.thin_hits[1], 1), 4),
            "n": self.thin_hits[1],
        }
        aps = [average_precision(self.slices, self.n_gt, threshold) for threshold in AP_THRESHOLDS]
        out["map50"] = round(aps[0], 4)
        out["map50_95"] = round(float(np.mean(aps)), 4)
        out["ap_degenerate"] = not self.has_scores
        return out

    def summary(self) -> Dict[str, float]:
        """Flat metrics for experiment logging.

        Returns:
            Dict[str, float]: knot_iou (pixel-pooled), knot_map50, knot_map50_95, knot_inst_f1_30,
                knot_inst_f1_50, knot_per_gt_iou, knot_thin_recall.
        """
        r = self.result()
        return {
            "knot_iou": r["pixel_iou"] or 0.0,
            "knot_map50": r["map50"],
            "knot_map50_95": r["map50_95"],
            "knot_inst_f1_30": r["iou_0.3"]["f1"],
            "knot_inst_f1_50": r["iou_0.5"]["f1"],
            "knot_per_gt_iou": r["iou_0.3"]["mean_per_gt_iou"] or 0.0,
            "knot_thin_recall": r["recall_thin_iou0.3"]["recall"],
        }


def iou_matrix(gt_labels: np.ndarray, n_gt: int, pred_labels: np.ndarray, n_pred: int) -> np.ndarray:
    if n_gt == 0 or n_pred == 0:
        return np.zeros((n_gt, n_pred))
    joint = np.bincount(gt_labels.ravel() * (n_pred + 1) + pred_labels.ravel(), minlength=(n_gt + 1) * (n_pred + 1))
    joint = joint.reshape(n_gt + 1, n_pred + 1)
    gt_area, pred_area = joint.sum(1), joint.sum(0)
    inter = joint[1:, 1:].astype(float)
    return inter / (gt_area[1:, None] + pred_area[None, 1:] - inter)


def component_scores(labels: np.ndarray, n: int, score_map: np.ndarray, reduce: str) -> np.ndarray:
    if n == 0:
        return np.zeros(0)
    index = np.arange(1, n + 1)
    if reduce == "max":
        return np.asarray(ndi.maximum(score_map, labels, index), dtype=float)
    return np.asarray(ndi.mean(score_map, labels, index), dtype=float)


def average_precision(slices: List[_Slice], n_gt: int, threshold: float) -> float:
    if n_gt == 0:
        return 0.0
    order: List[Tuple[float, int, int]] = [
        (-score, s, p) for s, item in enumerate(slices) for p, score in enumerate(item.scores)
    ]
    order.sort()
    taken = [np.zeros(item.iou.shape[0], bool) for item in slices]
    tp = np.zeros(len(order))
    for k, (_, s, p) in enumerate(order):
        ious = np.where(taken[s], -1.0, slices[s].iou[:, p]) if slices[s].iou.shape[0] else np.zeros(0)
        if len(ious) and ious.max() >= threshold:
            taken[s][int(ious.argmax())] = True
            tp[k] = 1
    if not len(order):
        return 0.0
    cum_tp = np.cumsum(tp)
    recall = cum_tp / n_gt
    precision = cum_tp / np.arange(1, len(order) + 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    sampled = [envelope[np.searchsorted(recall, r, side="left")] if r <= recall[-1] else 0.0 for r in RECALL_POINTS]
    return float(np.mean(sampled))
