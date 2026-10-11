import numpy as np
import pytest
from src.eval.knot_instances import KnotInstanceEvaluator, _Slice, average_precision


@pytest.fixture
def two_knots() -> np.ndarray:
    """Slice with one thick (10x10) and one thin (2x30) knot."""
    mask = np.zeros((64, 64), bool)
    mask[5:15, 5:15] = True
    mask[30:32, 10:40] = True
    return mask


def test_perfect_prediction(two_knots: np.ndarray) -> None:
    """A perfect scored prediction gives F1 1, per-GT IoU 1 and AP 1."""
    evaluator = KnotInstanceEvaluator()
    evaluator.add(two_knots, two_knots, score_map=two_knots.astype(float))
    r = evaluator.result()
    assert r["iou_0.5"]["f1"] == 1.0
    assert r["iou_0.3"]["mean_per_gt_iou"] == 1.0
    assert r["map50"] == 1.0
    assert r["map50_95"] == 1.0
    assert r["recall_thin_iou0.3"] == {"recall": 1.0, "n": 1}
    assert r["pixel_iou"] == 1.0
    assert not r["ap_degenerate"]


def test_empty_prediction(two_knots: np.ndarray) -> None:
    """No prediction gives zero recall and zero AP."""
    evaluator = KnotInstanceEvaluator()
    evaluator.add(two_knots, np.zeros_like(two_knots), score_map=np.zeros(two_knots.shape))
    r = evaluator.result()
    assert r["iou_0.3"]["recall"] == 0.0
    assert r["map50"] == 0.0


def test_fragment_counts_as_false_positive(two_knots: np.ndarray) -> None:
    """A GT knot split into two predicted fragments gives one match and one false positive."""
    pred = two_knots.copy()
    pred[5:15, 10] = False
    evaluator = KnotInstanceEvaluator()
    evaluator.add(two_knots, pred)
    r = evaluator.result()["iou_0.3"]
    assert (r["tp"], evaluator.n_pred, evaluator.n_gt) == (2, 3, 2)
    assert r["precision"] == pytest.approx(2 / 3, abs=1e-4)


def test_wood_clipping_removes_prediction(two_knots: np.ndarray) -> None:
    """Predictions outside the predicted wood mask are dropped before matching."""
    wood = np.zeros_like(two_knots)
    wood[:20] = True
    evaluator = KnotInstanceEvaluator()
    evaluator.add(two_knots, two_knots, wood=wood)
    assert evaluator.result()["iou_0.3"]["recall"] == 0.5


def test_score_order_changes_ap(two_knots: np.ndarray) -> None:
    """A false positive scored above the true positives lowers AP; scored below, it does not."""
    pred = two_knots.copy()
    pred[50:55, 50:55] = True
    results = []
    for fp_score in (0.9, 0.1):
        scores = np.where(two_knots, 0.5, fp_score)
        evaluator = KnotInstanceEvaluator()
        evaluator.add(two_knots, pred, score_map=scores)
        results.append(evaluator.result()["map50"])
    assert results[0] < results[1] == 1.0


def test_missing_scores_flag_degenerate_ap(two_knots: np.ndarray) -> None:
    """Without a score map AP is computed at a single operating point and flagged."""
    evaluator = KnotInstanceEvaluator()
    evaluator.add(two_knots, two_knots)
    assert evaluator.result()["ap_degenerate"]


def test_ap_pools_across_slices() -> None:
    """AP matches a hand-computed precision envelope over pooled detections."""
    slices = [
        _Slice(iou=np.array([[0.9, 0.0]]), scores=np.array([0.8, 0.9])),
        _Slice(iou=np.array([[0.7]]), scores=np.array([0.7])),
    ]
    ap = average_precision(slices, n_gt=2, threshold=0.5)
    assert ap == pytest.approx((51 * 2 / 3 + 50 * 2 / 3) / 101, abs=1e-6)
