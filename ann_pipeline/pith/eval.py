"""Evaluation utilities for pith detectors."""

from dataclasses import dataclass
import time
from typing import Callable, List, Tuple

import numpy as np

from ann_pipeline.data import PithSlice

XY = Tuple[int, int]
Detector = Callable[[np.ndarray], XY]


@dataclass
class SliceResult:
    page: int
    gt_xy: XY
    pred_xy: XY
    pixel_error: float
    inference_ms: float


@dataclass
class DetectorReport:
    name: str
    results: List[SliceResult]

    @property
    def errors(self) -> np.ndarray:
        return np.array([r.pixel_error for r in self.results])

    def summary(self) -> dict:
        e = self.errors
        return {
            "name": self.name,
            "n": len(self.results),
            "mean_err_px": float(e.mean()),
            "median_err_px": float(np.median(e)),
            "p90_err_px": float(np.percentile(e, 90)),
            "max_err_px": float(e.max()),
            "mean_ms": float(np.mean([r.inference_ms for r in self.results])),
        }


def evaluate(detector: Detector, slices: List[PithSlice], name: str) -> DetectorReport:
    results: List[SliceResult] = []
    for s in slices:
        t0 = time.perf_counter()
        pred = detector(s.img)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        err = float(np.hypot(pred[0] - s.pith_xy[0], pred[1] - s.pith_xy[1]))
        results.append(
            SliceResult(
                page=s.page,
                gt_xy=s.pith_xy,
                pred_xy=pred,
                pixel_error=err,
                inference_ms=dt_ms,
            )
        )
    return DetectorReport(name=name, results=results)
