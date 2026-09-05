"""``PointSelectionResult`` -- what the point-selection half hands back.

Section 5 of point-plan.md defines this format but nobody had built it, so both
halves would have invented their own dict shape and discovered the mismatch at
integration time. It lives on this side of the fence because the interface is
shared and the combined pipeline (Step 7) is what consumes it.

One field is additive: ``labels``. The plan's ``prompt_sam3`` treats every point
as positive, but the session contract accepts negative points too, and a result
that records only the coordinates of a mixed point set cannot be replayed --
re-running it all-positive gives a different mask. ``labels=None`` means
all-positive, so results written against the plan's literal format stay valid.

Section 3's final deliverable is "a folder of images, each paired with a set of
point locations", which is ``points_by_image`` over every result.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

FORMAT_VERSION = 1


@dataclass
class PointSelectionResult:
    """The final, pruned point set for one region, plus how it scored."""

    image_id: str
    region_id: int
    points: list[tuple[int, int]]      # (x, y) pixel coords, AFTER pruning
    final_coverage: float
    final_iou: float
    num_sam_calls: int
    labels: list[int] | None = None    # 1 positive / 0 negative; None = all positive

    def __post_init__(self) -> None:
        self.points = [(int(x), int(y)) for x, y in self.points]
        if self.labels is not None:
            self.labels = [int(v) for v in self.labels]
            if len(self.labels) != len(self.points):
                raise ValueError(f"{self.key}: {len(self.points)} points but "
                                 f"{len(self.labels)} labels")
            if any(v not in (0, 1) for v in self.labels):
                raise ValueError(f"{self.key}: labels must be 1 or 0")
        for name in ("final_coverage", "final_iou"):
            v = float(getattr(self, name))
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{self.key}: {name}={v} outside 0..1")
            setattr(self, name, v)
        if self.num_sam_calls < 0:
            raise ValueError(f"{self.key}: num_sam_calls={self.num_sam_calls}")

    @property
    def key(self) -> str:
        return f"{self.image_id}#{self.region_id}"

    @property
    def num_points(self) -> int:
        return len(self.points)

    def positive_points(self) -> list[tuple[int, int]]:
        if self.labels is None:
            return list(self.points)
        return [p for p, l in zip(self.points, self.labels) if l == 1]

    def to_dict(self) -> dict:
        d = {
            "image_id": self.image_id,
            "region_id": int(self.region_id),
            "points": [[int(x), int(y)] for x, y in self.points],
            "final_coverage": round(self.final_coverage, 6),
            "final_iou": round(self.final_iou, 6),
            "num_sam_calls": int(self.num_sam_calls),
        }
        if self.labels is not None:
            d["labels"] = list(self.labels)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PointSelectionResult":
        return cls(
            image_id=d["image_id"],
            region_id=int(d["region_id"]),
            points=[tuple(p) for p in d["points"]],
            final_coverage=float(d["final_coverage"]),
            final_iou=float(d["final_iou"]),
            num_sam_calls=int(d["num_sam_calls"]),
            labels=d.get("labels"),
        )

    def __repr__(self) -> str:
        return (f"PointSelectionResult({self.key} n_points={self.num_points} "
                f"cov={self.final_coverage:.3f} iou={self.final_iou:.3f} "
                f"calls={self.num_sam_calls})")


def save_results(results: list[PointSelectionResult], path: str | Path,
                 extra: dict | None = None) -> Path:
    """Write results to JSON, sorted so two runs produce identical files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(results, key=lambda r: (r.image_id, r.region_id))
    payload = {
        "version": FORMAT_VERSION,
        **(extra or {}),
        "results": [r.to_dict() for r in ordered],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_results(path: str | Path) -> list[PointSelectionResult]:
    payload = json.loads(Path(path).read_text())
    if payload.get("version") != FORMAT_VERSION:
        raise ValueError(f"{path}: format version {payload.get('version')}, "
                         f"expected {FORMAT_VERSION}")
    return [PointSelectionResult.from_dict(d) for d in payload["results"]]


def points_by_image(results: list[PointSelectionResult],
                    positive_only: bool = True) -> dict[str, list[tuple[int, int]]]:
    """Collapse per-region results into Section 3's final deliverable shape.

    Points carry no class label there, so regions of one image merge into one
    list. Duplicates are dropped, order is deterministic.
    """
    merged: dict[str, list] = defaultdict(list)
    for r in sorted(results, key=lambda r: (r.image_id, r.region_id)):
        pts = r.positive_points() if positive_only else r.points
        for p in pts:
            if p not in merged[r.image_id]:
                merged[r.image_id].append(p)
    return dict(merged)


def summarise(results: list[PointSelectionResult]) -> dict:
    """Headline numbers for the report: points per region, calls, scores."""
    if not results:
        return {"count": 0}
    n_pts = [r.num_points for r in results]
    return {
        "count": len(results),
        "total_points": int(sum(n_pts)),
        "points_per_region_mean": round(sum(n_pts) / len(results), 3),
        "points_per_region_max": int(max(n_pts)),
        "total_sam_calls": int(sum(r.num_sam_calls for r in results)),
        "mean_final_coverage": round(
            sum(r.final_coverage for r in results) / len(results), 4),
        "mean_final_iou": round(
            sum(r.final_iou for r in results) / len(results), 4),
    }
