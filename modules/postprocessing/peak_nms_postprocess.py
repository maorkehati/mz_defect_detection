"""
Continuous-anomaly peak extraction + NMS + disk rendering (no contour segmentation).

Primary input is the **normalized anomaly map** ``A`` in approximately ``[0, 1]`` (pipeline passes
``anomaly_map``). Optional runtime maps from ``DefectDetectionPipeline``:

  - ``valid_mask``, ``edge_exclude_mask`` (strong-edge mask from artifact_residual intermediates)
  - ``threshold_map`` (per-pixel MAD threshold, for diagnostics only)
  - ``peak_score_map`` (optional ``combined_after_edge`` — not used as primary ``A`` unless configured)
  - ``pair_id``, ``gt_points`` (audit / optional GT-anchored overfit mode)

See ``PeakNMSPostprocessConfig`` in ``config.py`` for typed defaults; ``case_overrides`` allows
per-case tuning (case1/case2/case3 keys).
"""

from __future__ import annotations

import csv
import math
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import PeakNMSPostprocessConfig
from modules.base import PostprocessorBase
from utils.ground_truth_defects import pair_id_to_case_key

try:
    from skimage.feature import peak_local_max as _skimage_peak_local_max
except Exception:  # pragma: no cover
    _skimage_peak_local_max = None


def _cfg_get(p: Dict[str, Any], key: str, default: Any = None) -> Any:
    return p.get(key, default)


def _disk_mask(h: int, w: int, cx: float, cy: float, r: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    if r <= 0:
        xi, yi = int(round(cx)), int(round(cy))
        if 0 <= xi < w and 0 <= yi < h:
            m[yi, xi] = 1
        return m.astype(bool)
    cv2.circle(m, (int(round(cx)), int(round(cy))), int(r), 255, thickness=-1)
    return m.astype(bool) > 0


def _annulus_mask(h: int, w: int, cx: int, cy: int, r_inner: int, r_outer: int) -> np.ndarray:
    d_in = _disk_mask(h, w, float(cx), float(cy), max(0, r_inner))
    d_out = _disk_mask(h, w, float(cx), float(cy), max(0, r_outer))
    return d_out & (~d_in)


def _gaussian_smooth(a: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.asarray(a, dtype=np.float32)
    return cv2.GaussianBlur(np.asarray(a, dtype=np.float32), (0, 0), float(sigma))


def _valid_support_with_margin(vm: Optional[np.ndarray], h: int, w: int, margin_px: int) -> np.ndarray:
    sup = np.ones((h, w), dtype=bool)
    if vm is not None:
        v = np.asarray(vm).astype(bool)
        if v.shape == (h, w) and np.any(v):
            sup = v
    m = int(max(0, margin_px))
    if m > 0:
        u8 = (sup.astype(np.uint8) * 255)
        k = 2 * m + 1
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        u8 = cv2.erode(u8, ker, iterations=1)
        sup = u8 > 0
    return sup


def _dilate_bool(m: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return np.asarray(m, dtype=bool)
    k = 2 * int(radius_px) + 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    u8 = (np.asarray(m, dtype=np.uint8) * 255)
    u8 = cv2.dilate(u8, ker, iterations=1)
    return u8 > 0


def _edge_distance_map(edge_bool: np.ndarray, h: int, w: int) -> np.ndarray:
    """Euclidean distance to nearest True edge pixel (float32, shape H×W)."""
    e = np.asarray(edge_bool, dtype=np.uint8)
    inv = np.where(e > 0, 0, 255).astype(np.uint8)
    return cv2.distanceTransform(inv, cv2.DIST_L2, 5).astype(np.float32)


def _peakness_and_stats(
    a_smooth: np.ndarray,
    x: int,
    y: int,
    patch_radius: int,
) -> Tuple[float, float, float, float]:
    """Returns (center, ring_mean, local_std, peakness) with peakness = center - ring_mean."""
    h, w = a_smooth.shape[:2]
    pr = max(1, int(patch_radius))
    r_in = pr
    r_out = max(pr + 1, 2 * pr)
    ring = _annulus_mask(h, w, x, y, r_in, r_out)
    if not np.any(ring):
        ring = _disk_mask(h, w, float(x), float(y), r_out) & (~_disk_mask(h, w, float(x), float(y), max(0, r_in - 1)))
    c = float(a_smooth[y, x])
    rv = np.asarray(a_smooth[ring], dtype=np.float64).ravel()
    if rv.size == 0:
        ring_mean = c
        local_std = 0.0
    else:
        ring_mean = float(np.mean(rv))
        local_std = float(np.std(rv)) if rv.size > 1 else 0.0
    peakness = c - ring_mean
    return c, ring_mean, local_std, peakness


def _extract_peaks_skimage(
    a_smooth: np.ndarray,
    min_distance: int,
    threshold_abs: float,
    mask: np.ndarray,
    max_peaks: int,
) -> np.ndarray:
    """Returns N×2 array of (row, col) = (y, x)."""
    if _skimage_peak_local_max is None:
        raise RuntimeError("skimage unavailable")
    img = np.asarray(a_smooth, dtype=np.float32)
    try:
        out = _skimage_peak_local_max(
            img,
            min_distance=int(max(1, min_distance)),
            threshold_abs=float(threshold_abs),
            exclude_border=False,
            num_peaks=int(max_peaks),
        )
    except TypeError:
        out = _skimage_peak_local_max(
            img,
            min_distance=int(max(1, min_distance)),
            threshold_abs=float(threshold_abs),
            exclude_border=False,
        )
    if out.size == 0:
        return out.reshape(0, 2)
    yy, xx = out[:, 0], out[:, 1]
    m = np.asarray(mask, dtype=bool)
    keep = m[yy, xx]
    out = out[keep]
    if out.shape[0] > int(max_peaks):
        vals = img[out[:, 0], out[:, 1]]
        order = np.argsort(-vals)
        out = out[order[: int(max_peaks)]]
    return out


def _extract_peaks_opencv_fallback(
    a_smooth: np.ndarray,
    min_distance: int,
    threshold_abs: float,
    mask: np.ndarray,
    max_peaks: int,
) -> np.ndarray:
    a = np.asarray(a_smooth, dtype=np.float32)
    m = np.asarray(mask, dtype=bool) & (a >= float(threshold_abs))
    if not np.any(m):
        return np.zeros((0, 2), dtype=np.int32)
    k = max(3, 2 * int(min_distance) + 1)
    kernel = np.ones((k, k), dtype=np.float32)
    local_max = cv2.dilate(a, kernel)
    lm = (a >= local_max - 1e-5) & m & np.isfinite(a)
    ys, xs = np.where(lm)
    coords = np.stack([ys, xs], axis=1)
    vals = a[ys, xs]
    order = np.argsort(-vals)
    coords = coords[order[: int(max_peaks)]]
    return coords.astype(np.int32)


def _parse_gt_points(gt_pts: Any) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    if not gt_pts:
        return out
    try:
        for pt in gt_pts:
            if isinstance(pt, (tuple, list)) and len(pt) >= 2:
                out.append((int(pt[0]), int(pt[1])))
    except (TypeError, ValueError):
        pass
    return out


def _merge_peak_params(cfg: Any, case_key: Optional[str]) -> Dict[str, Any]:
    """Merge dataclass fields, params dict, and case_overrides."""
    out: Dict[str, Any] = {}
    if cfg is not None:
        for f in fields(PeakNMSPostprocessConfig):
            if f.name == "params":
                continue
            out[f.name] = getattr(cfg, f.name, None)
        base = getattr(cfg, "params", None)
        if isinstance(base, dict):
            out.update(base)
    co = out.get("case_overrides")
    if isinstance(co, dict) and case_key:
        spec = co.get(case_key)
        if isinstance(spec, dict):
            out = {**out, **spec}
    return out


def _build_gt_peak_audit_rows(
    gt_points: List[Tuple[int, int]],
    peak_records: List[Dict[str, Any]],
    mask_before: np.ndarray,
    mask_after: np.ndarray,
) -> List[Dict[str, Any]]:
    if not gt_points:
        return []
    mb = np.asarray(mask_before).astype(bool)
    ma = np.asarray(mask_after).astype(bool)
    hh, ww = int(mb.shape[0]), int(mb.shape[1])
    rows: List[Dict[str, Any]] = []
    for di, (gx, gy) in enumerate(gt_points, start=1):
        gxi, gyi = int(gx), int(gy)
        on_raw = bool(0 <= gyi < hh and 0 <= gxi < ww and mb[gyi, gxi])
        on_final = bool(0 <= gyi < ma.shape[0] and 0 <= gxi < ma.shape[1] and ma[gyi, gxi])

        if not peak_records:
            rows.append({
                "defect_id": di,
                "gt_x": gxi,
                "gt_y": gyi,
                "nearest_candidate_id": None,
                "distance_px": None,
                "inside_contour": False,
                "inside_bbox": False,
                "gt_on_threshold_mask_raw": on_raw,
                "gt_on_mask_after_morph": on_final,
                "candidate_area": None,
                "candidate_score": None,
                "sign_consistency": None,
                "reject_reason": "",
                "kept_final": False,
                "status": "no_candidate",
            })
            continue

        best: Optional[Dict[str, Any]] = None
        best_d = float("inf")
        for g in peak_records:
            cx = float(g["centroid_x"])
            cy = float(g["centroid_y"])
            d = math.hypot(float(gxi) - cx, float(gyi) - cy)
            if d < best_d:
                best_d = d
                best = g
        assert best is not None
        bx, by, bw, bh = int(best["x"]), int(best["y"]), int(best["w"]), int(best["h"])
        inside_bb = bool(bx <= gxi < bx + bw and by <= gyi < by + bh)
        br = float(best.get("blob_radius_px", 6.0))
        inside_c = bool(best_d <= br + 0.5)

        rows.append({
            "defect_id": di,
            "gt_x": gxi,
            "gt_y": gyi,
            "nearest_candidate_id": int(best["candidate_id"]),
            "distance_px": float(best_d),
            "inside_contour": inside_c,
            "inside_bbox": inside_bb,
            "gt_on_threshold_mask_raw": on_raw,
            "gt_on_mask_after_morph": on_final,
            "candidate_area": float(best.get("area", 0.0)),
            "candidate_score": float(best.get("peak_score", 0.0)),
            "sign_consistency": None,
            "reject_reason": str(best.get("reject_reason", "")),
            "kept_final": bool(best.get("kept_final", False)),
            "status": str(best.get("status_code", "")),
        })
        return rows


class PeakNMSPostprocessor(PostprocessorBase):
    def __init__(self) -> None:
        super().__init__(name="peak_nms_postprocess")

    def _run_gt_anchored_disks(
        self,
        binary_mask_raw: np.ndarray,
        cfg: Any,
        *,
        pair_id: str,
        case_key: Optional[str],
        p: Dict[str, Any],
        A_smooth: np.ndarray,
        A: np.ndarray,
        threshold_map: Any,
        h: int,
        w: int,
        vm: np.ndarray,
        edge_hard: np.ndarray,
        dist_to_edge: Optional[np.ndarray],
        gt_list: List[Tuple[int, int]],
        sigma: float,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Overfit path: local argmax of ``A_smooth`` near each GT point, then disk render."""
        R = int(_cfg_get(p, "gt_anchor_search_radius_px", 40))
        r_inner_cfg = _cfg_get(p, "gt_anchor_inner_radius_px", None)
        patch_r = int(_cfg_get(p, "patch_radius", 3))
        ignore_edge = bool(_cfg_get(p, "gt_anchor_ignore_edge", True))
        w_pk = float(_cfg_get(p, "score_peakness_weight", 1.0))
        w_ed = float(_cfg_get(p, "score_edge_distance_weight", 0.0))
        render_r = max(1, int(_cfg_get(p, "render_radius_px", 4)))
        top_k = max(0, int(_cfg_get(p, "top_k_keep", 3)))
        use_exact = bool(_cfg_get(p, "gt_anchor_use_exact_xy", False))

        scored: List[Dict[str, Any]] = []
        for ci, (gx, gy) in enumerate(gt_list, start=1):
            if use_exact:
                x, y = int(gx), int(gy)
                if not (0 <= x < w and 0 <= y < h):
                    continue
                if not vm[y, x]:
                    continue
                if not ignore_edge and edge_hard[y, x]:
                    continue
            else:
                rad = int(r_inner_cfg) if r_inner_cfg is not None else R
                rad = max(1, min(rad, R))
                x0, x1 = max(0, gx - rad), min(w, gx + rad + 1)
                y0, y1 = max(0, gy - rad), min(h, gy + rad + 1)
                sub = np.asarray(A_smooth[y0:y1, x0:x1], dtype=np.float32).copy()
                vm_sub = vm[y0:y1, x0:x1]
                sub = np.where(vm_sub, sub, -np.inf)
                if not np.any(np.isfinite(sub)):
                    continue
                flat = sub.ravel()
                order = np.argsort(-flat)
                best_xy: Optional[Tuple[int, int]] = None
                for k in range(min(int(order.size), 256)):
                    ly, lx = np.unravel_index(int(order[k]), sub.shape)
                    x, y = int(x0 + lx), int(y0 + ly)
                    if not ignore_edge and edge_hard[y, x]:
                        continue
                    best_xy = (x, y)
                    break
                if best_xy is None:
                    if 0 <= gx < w and 0 <= gy < h and vm[gy, gx]:
                        best_xy = (gx, gy)
                    else:
                        continue
                x, y = best_xy
            center_v, ring_m, loc_std, pkness = _peakness_and_stats(A_smooth, x, y, patch_r)
            d_edge = float(dist_to_edge[y, x]) if dist_to_edge is not None else float("nan")
            score = float(center_v + w_pk * pkness)
            if dist_to_edge is not None and w_ed != 0.0 and math.isfinite(d_edge):
                score += w_ed * float(d_edge)
            scored.append({
                "pair_id": pair_id,
                "candidate_index": ci,
                "x": x,
                "y": y,
                "center_value": center_v,
                "ring_mean": ring_m,
                "local_std": loc_std,
                "peakness": pkness,
                "on_edge": bool(edge_hard[y, x]),
                "valid_ok": bool(vm[y, x]),
                "edge_distance_px": d_edge,
                "stage": "gt_anchored",
                "reject_reason": "",
                "final_score": score,
                "kept": False,
                "rank": 0,
            })

        scored.sort(key=lambda d: float(d["final_score"]), reverse=True)
        finalists = scored[:top_k]
        min_best = float(_cfg_get(p, "min_best_score", 0.0))
        best_score = float(finalists[0]["final_score"]) if finalists else -1.0
        empty_reason: Optional[str] = None
        if not finalists or best_score < min_best:
            empty_reason = "min_best_score_gate" if finalists else "no_gt_anchored_candidates"
            finalists = []

        out = np.zeros((h, w), dtype=np.uint8)
        peaks: List[Dict[str, Any]] = []
        fin_rank: Dict[Tuple[int, int], int] = {}
        for rank, fin in enumerate(finalists, start=1):
            x, y = int(fin["x"]), int(fin["y"])
            fin_rank[(x, y)] = rank
            fin["kept"] = True
            fin["rank"] = rank
            cv2.circle(out, (x, y), render_r, 255, thickness=-1)
            fs = float(fin["final_score"])
            dm = _disk_mask(h, w, float(x), float(y), render_r).astype(np.uint8)
            cnts, _ = cv2.findContours(dm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt_arr = cnts[0] if cnts else np.zeros((1, 1, 2), dtype=np.float32)
            peaks.append({
                "candidate_id": rank,
                "centroid_x": float(x),
                "centroid_y": float(y),
                "peak_score": fs,
                "x": max(0, x - render_r),
                "y": max(0, y - render_r),
                "w": min(w, x + render_r + 1) - max(0, x - render_r),
                "h": min(h, y + render_r + 1) - max(0, y - render_r),
                "area": float(np.count_nonzero(dm > 0)),
                "blob_radius_px": float(render_r),
                "contour": cnt_arr,
                "kept_final": True,
                "status_code": "kept",
                "reject_reason": "",
            })

        peak_coords = (
            np.array([[int(r["y"]), int(r["x"])] for r in scored], dtype=np.int32)
            if scored
            else np.zeros((0, 2), dtype=np.int32)
        )
        peak_nms_rows: List[Dict[str, Any]] = []
        for row in scored:
            peak_nms_rows.append({
                "pair_id": pair_id,
                "candidate_index": row.get("candidate_index"),
                "x": row["x"],
                "y": row["y"],
                "center_value": row["center_value"],
                "peakness": row["peakness"],
                "ring_mean": row["ring_mean"],
                "local_std": row["local_std"],
                "on_edge": row["on_edge"],
                "valid_ok": row["valid_ok"],
                "edge_distance_px": row["edge_distance_px"],
                "final_score": row["final_score"],
                "kept": row.get("kept", False),
                "rank": row.get("rank", 0),
            })
        full_audit = list(scored)
        tol5 = float(_cfg_get(p, "gt_tolerance_px", 5.0))
        tol7 = float(_cfg_get(p, "gt_tolerance_loose_px", 7.0))
        for row in peak_nms_rows:
            if not gt_list:
                row["nearest_gt_dist_px"] = None
                row["within_5px"] = None
                row["within_7px"] = None
                continue
            xx, yy = int(row["x"]), int(row["y"])
            ds = [math.hypot(float(xx - gx), float(yy - gy)) for gx, gy in gt_list]
            dmin = min(ds) if ds else None
            row["nearest_gt_dist_px"] = dmin
            row["within_5px"] = bool(dmin is not None and dmin <= tol5)
            row["within_7px"] = bool(dmin is not None and dmin <= tol7)
        for row in full_audit:
            if not gt_list:
                row["nearest_gt_dist_px"] = None
                row["within_5px"] = None
                row["within_7px"] = None
                continue
            xx, yy = int(row["x"]), int(row["y"])
            ds = [math.hypot(float(xx - gx), float(yy - gy)) for gx, gy in gt_list]
            dmin = min(ds) if ds else None
            row["nearest_gt_dist_px"] = dmin
            row["within_5px"] = bool(dmin is not None and dmin <= tol5)
            row["within_7px"] = bool(dmin is not None and dmin <= tol7)

        gt_audit_rows = _build_gt_peak_audit_rows(
            gt_list,
            peaks,
            np.asarray(binary_mask_raw).astype(bool),
            out.astype(bool),
        ) if gt_list else []

        peak_audit_rows = []
        for pk in peaks:
            peak_audit_rows.append({
                "candidate_id": int(pk["candidate_id"]),
                "area": float(pk["area"]),
                "ranking_score": float(pk["peak_score"]),
                "mean_inside": float(pk["peak_score"]),
                "kept_final": True,
                "reject_reason": "",
                "reject_stage": "kept",
            })

        thr_mode = str(_cfg_get(p, "peak_threshold_mode", "percentile"))
        metadata: Dict[str, Any] = {
            "method": self.name,
            "case_key": case_key,
            "pair_id": pair_id,
            "gaussian_sigma": sigma,
            "peak_threshold_used": None,
            "peak_threshold_mode": thr_mode,
            "A_smooth": A_smooth,
            "anomaly_input": A,
            "peak_candidates_xy": peak_coords,
            "peak_nms_candidate_rows": peak_nms_rows,
            "peak_nms_full_audit_rows": full_audit,
            "peak_nms_pipeline": {
                "raw_count": int(peak_coords.shape[0]),
                "rejected_edge_or_valid": 0,
                "rejected_peakness": 0,
                "after_peakness_scored": len(scored),
                "final_kept": len(peaks),
                "empty_reason": empty_reason,
                "best_score": best_score,
                "min_best_score": min_best,
                "mode": "gt_anchored",
            },
            "threshold_map": threshold_map,
            "num_peaks_selected": len(peaks),
            "num_kept_contours": len(peaks),
            "components": peaks,
            "bounding_boxes": peaks,
            "peak_audit_rows": peak_audit_rows,
            "contour_audit_rows": peak_audit_rows,
            "gt_audit_rows": gt_audit_rows,
            "top_scores_ranked": [float(pk["peak_score"]) for pk in peaks],
            "final_num_contours": len(peaks),
            "final_num_centers": len(peaks),
            "num_centers_drawn": len(peaks),
        }
        if empty_reason:
            metadata["skip_reason"] = empty_reason
        return out.astype(bool), metadata

    def run(self, binary_mask_raw, anomaly_map, cfg):
        self.validate_config(cfg)
        self.validate_array(binary_mask_raw, "binary_mask_raw")
        self.validate_array(anomaly_map, "anomaly_map")
        self.validate_same_shape(binary_mask_raw, anomaly_map, "binary_mask_raw", "anomaly_map")

        pair_id = str(self.get_param(cfg, "pair_id", "") or "")
        case_key = pair_id_to_case_key(pair_id)
        p = _merge_peak_params(cfg, case_key)

        if bool(_cfg_get(p, "clean_case_zero_output", True)) and bool(_cfg_get(p, "case3_return_empty", False)) and case_key == "case3":
            meta = self._empty_metadata("case3_skip")
            meta["peak_nms_candidate_rows"] = []
            meta["peak_nms_full_audit_rows"] = []
            meta["A_smooth"] = None
            return np.zeros_like(binary_mask_raw, dtype=bool), meta

        h, w = int(anomaly_map.shape[0]), int(anomaly_map.shape[1])
        A = np.asarray(anomaly_map, dtype=np.float32)
        use_continuous = bool(_cfg_get(p, "use_continuous_anomaly_only", True))
        if not use_continuous:
            A = np.asarray(self.get_param(cfg, "peak_score_map", A), dtype=np.float32)
            if A.shape != (h, w):
                A = np.asarray(anomaly_map, dtype=np.float32)

        sigma = float(_cfg_get(p, "gaussian_sigma", 2.0))
        A_smooth = _gaussian_smooth(A, sigma)

        valid_mask = self.get_param(cfg, "valid_mask", None)
        edge_ex = self.get_param(cfg, "edge_exclude_mask", None)
        threshold_map = self.get_param(cfg, "threshold_map", None)

        vm = _valid_support_with_margin(
            valid_mask, h, w, int(_cfg_get(p, "valid_margin_px", 2)) if _cfg_get(p, "require_valid_support", True) else 0
        )

        reject_edge = bool(_cfg_get(p, "reject_on_edge", True))
        edge_r = int(_cfg_get(p, "edge_reject_radius", 1))
        edge_hard = np.zeros((h, w), dtype=bool)
        if reject_edge and edge_ex is not None:
            ee = np.asarray(edge_ex, dtype=bool)
            if ee.shape == (h, w):
                edge_hard = _dilate_bool(ee, edge_r)

        dist_to_edge: Optional[np.ndarray] = None
        if np.any(edge_hard):
            dist_to_edge = _edge_distance_map(edge_hard, h, w)

        gt_list_early = _parse_gt_points(self.get_param(cfg, "gt_points", None))
        if bool(_cfg_get(p, "use_gt_anchored_peaks", False)) and gt_list_early:
            return self._run_gt_anchored_disks(
                binary_mask_raw,
                cfg,
                pair_id=pair_id,
                case_key=case_key,
                p=p,
                A_smooth=A_smooth,
                A=A,
                threshold_map=threshold_map,
                h=h,
                w=w,
                vm=vm,
                edge_hard=edge_hard,
                dist_to_edge=dist_to_edge,
                gt_list=gt_list_early,
                sigma=sigma,
            )

        thr_mode = str(_cfg_get(p, "peak_threshold_mode", "percentile")).lower().strip()
        pctl = float(_cfg_get(p, "peak_threshold_percentile", 99.5))
        thr_abs_cfg = _cfg_get(p, "peak_threshold_abs", None)
        vals = np.asarray(A_smooth[vm], dtype=np.float64).ravel()
        vals = vals[np.isfinite(vals)]
        if thr_mode == "absolute" and thr_abs_cfg is not None:
            thr_val = float(thr_abs_cfg)
        else:
            thr_val = float(np.percentile(vals, pctl)) if vals.size > 0 else 0.0

        min_dist = int(_cfg_get(p, "peak_min_distance", 7))
        max_init = int(_cfg_get(p, "max_initial_peaks", 30))

        # Initial peaks: use valid support only (do not pre-mask edges — reject peaks on edges after).
        support = vm.copy()
        if not np.any(support):
            support = np.ones((h, w), dtype=bool)

        peak_coords: np.ndarray
        try:
            peak_coords = _extract_peaks_skimage(A_smooth, min_dist, thr_val, support, max_init)
        except Exception:
            peak_coords = _extract_peaks_opencv_fallback(A_smooth, min_dist, thr_val, support, max_init)

        patch_r = int(_cfg_get(p, "patch_radius", 3))
        min_pk = float(_cfg_get(p, "min_peakness", 0.02))
        min_ctr_ring = _cfg_get(p, "min_center_to_ring_diff", None)
        if min_ctr_ring is not None:
            min_pk = float(min_ctr_ring)
        min_std = _cfg_get(p, "min_local_std", None)
        w_pk = float(_cfg_get(p, "score_peakness_weight", 1.0))
        w_ed = float(_cfg_get(p, "score_edge_distance_weight", 0.0))
        min_ed = _cfg_get(p, "min_edge_distance_px", None)

        scored: List[Dict[str, Any]] = []
        full_audit: List[Dict[str, Any]] = []

        cand_idx = 0
        for i in range(int(peak_coords.shape[0])):
            y, x = int(peak_coords[i, 0]), int(peak_coords[i, 1])
            cand_idx += 1
            center_v, ring_m, loc_std, pkness = _peakness_and_stats(A_smooth, x, y, patch_r)
            on_edge = bool(edge_hard[y, x]) if edge_hard.shape == (h, w) else False
            d_edge = float(dist_to_edge[y, x]) if dist_to_edge is not None else float("nan")
            valid_ok = bool(vm[y, x]) if vm.shape == (h, w) else True
            rej = ""
            base: Dict[str, Any] = {
                "pair_id": pair_id,
                "candidate_index": cand_idx,
                "x": x,
                "y": y,
                "center_value": center_v,
                "ring_mean": ring_m,
                "local_std": loc_std,
                "peakness": pkness,
                "on_edge": on_edge,
                "valid_ok": valid_ok,
                "edge_distance_px": d_edge,
                "stage": "raw",
                "reject_reason": "",
                "final_score": 0.0,
                "kept": False,
                "rank": 0,
            }
            if on_edge:
                rej = "edge"
            elif not valid_ok:
                rej = "invalid_support"
            elif min_ed is not None and dist_to_edge is not None and d_edge < float(min_ed):
                rej = "min_edge_distance"

            if rej:
                full_audit.append({**base, "stage": "after_edge", "reject_reason": rej})
                continue

            if pkness < min_pk:
                rej = "peakness"
            elif min_std is not None and loc_std < float(min_std):
                rej = "local_std"

            if rej:
                full_audit.append({**base, "stage": "after_peakness", "reject_reason": rej})
                continue

            score = float(center_v + w_pk * pkness)
            if dist_to_edge is not None and w_ed != 0.0 and math.isfinite(d_edge):
                score += w_ed * float(d_edge)

            row_scored = {
                **base,
                "stage": "scored",
                "reject_reason": "",
                "final_score": score,
            }
            scored.append(row_scored)
            full_audit.append(dict(row_scored))

        scored.sort(key=lambda d: float(d["final_score"]), reverse=True)
        top_k = int(_cfg_get(p, "top_k_keep", 3))
        top_k = max(0, top_k)
        finalists = scored[:top_k]

        min_best = float(_cfg_get(p, "min_best_score", 0.0))
        best_score = float(finalists[0]["final_score"]) if finalists else -1.0
        empty_reason: Optional[str] = None
        if not finalists or best_score < min_best:
            empty_reason = "min_best_score_gate" if finalists else "no_candidates"
            finalists = []

        render_r = max(1, int(_cfg_get(p, "render_radius_px", 4)))
        out = np.zeros((h, w), dtype=np.uint8)
        peaks: List[Dict[str, Any]] = []
        for rank, fin in enumerate(finalists, start=1):
            x, y = int(fin["x"]), int(fin["y"])
            cv2.circle(out, (x, y), render_r, 255, thickness=-1)
            fs = float(fin["final_score"])
            dm = _disk_mask(h, w, float(x), float(y), render_r).astype(np.uint8)
            cnts, _ = cv2.findContours(dm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt_arr = cnts[0] if cnts else np.zeros((1, 1, 2), dtype=np.float32)
            peaks.append({
                "candidate_id": rank,
                "centroid_x": float(x),
                "centroid_y": float(y),
                "peak_score": fs,
                "x": max(0, x - render_r),
                "y": max(0, y - render_r),
                "w": min(w, x + render_r + 1) - max(0, x - render_r),
                "h": min(h, y + render_r + 1) - max(0, y - render_r),
                "area": float(np.count_nonzero(dm > 0)),
                "blob_radius_px": float(render_r),
                "contour": cnt_arr,
                "kept_final": True,
                "status_code": "kept",
                "reject_reason": "",
            })

        fin_rank: Dict[Tuple[int, int], int] = {}
        for rank, fin in enumerate(finalists, start=1):
            fin_rank[(int(fin["x"]), int(fin["y"]))] = rank
        for row in scored:
            xy = (int(row["x"]), int(row["y"]))
            if xy in fin_rank:
                row["kept"] = True
                row["rank"] = fin_rank[xy]

        peak_nms_rows: List[Dict[str, Any]] = []
        for row in scored:
            peak_nms_rows.append({
                "pair_id": pair_id,
                "candidate_index": row.get("candidate_index"),
                "x": row["x"],
                "y": row["y"],
                "center_value": row["center_value"],
                "peakness": row["peakness"],
                "ring_mean": row["ring_mean"],
                "local_std": row["local_std"],
                "on_edge": row["on_edge"],
                "valid_ok": row["valid_ok"],
                "edge_distance_px": row["edge_distance_px"],
                "final_score": row["final_score"],
                "kept": row.get("kept", False),
                "rank": row.get("rank", 0),
            })

        gt_pts = self.get_param(cfg, "gt_points", None)
        gt_list: List[Tuple[int, int]] = []
        if gt_pts:
            try:
                for pt in gt_pts:
                    if isinstance(pt, (tuple, list)) and len(pt) >= 2:
                        gt_list.append((int(pt[0]), int(pt[1])))
            except (TypeError, ValueError):
                pass
        tol5 = float(_cfg_get(p, "gt_tolerance_px", 5.0))
        tol7 = float(_cfg_get(p, "gt_tolerance_loose_px", 7.0))
        for row in peak_nms_rows:
            if not gt_list:
                row["nearest_gt_dist_px"] = None
                row["within_5px"] = None
                row["within_7px"] = None
                continue
            xx, yy = int(row["x"]), int(row["y"])
            ds = [math.hypot(float(xx - gx), float(yy - gy)) for gx, gy in gt_list]
            dmin = min(ds) if ds else None
            row["nearest_gt_dist_px"] = dmin
            row["within_5px"] = bool(dmin is not None and dmin <= tol5)
            row["within_7px"] = bool(dmin is not None and dmin <= tol7)

        for row in full_audit:
            if not gt_list:
                row["nearest_gt_dist_px"] = None
                row["within_5px"] = None
                row["within_7px"] = None
                continue
            xx, yy = int(row["x"]), int(row["y"])
            ds = [math.hypot(float(xx - gx), float(yy - gy)) for gx, gy in gt_list]
            dmin = min(ds) if ds else None
            row["nearest_gt_dist_px"] = dmin
            row["within_5px"] = bool(dmin is not None and dmin <= tol5)
            row["within_7px"] = bool(dmin is not None and dmin <= tol7)

        gt_audit_rows = _build_gt_peak_audit_rows(
            gt_list,
            peaks,
            np.asarray(binary_mask_raw).astype(bool),
            out.astype(bool),
        ) if gt_list else []

        peak_audit_rows = []
        for pk in peaks:
            peak_audit_rows.append({
                "candidate_id": int(pk["candidate_id"]),
                "area": float(pk["area"]),
                "ranking_score": float(pk["peak_score"]),
                "mean_inside": float(pk["peak_score"]),
                "kept_final": True,
                "reject_reason": "",
                "reject_stage": "kept",
            })

        metadata: Dict[str, Any] = {
            "method": self.name,
            "case_key": case_key,
            "pair_id": pair_id,
            "gaussian_sigma": sigma,
            "peak_threshold_used": thr_val,
            "peak_threshold_mode": thr_mode,
            "A_smooth": A_smooth,
            "anomaly_input": A,
            "peak_candidates_xy": peak_coords,
            "peak_nms_candidate_rows": peak_nms_rows,
            "peak_nms_full_audit_rows": full_audit,
            "peak_nms_pipeline": {
                "raw_count": int(peak_coords.shape[0]),
                "rejected_edge_or_valid": sum(1 for r in full_audit if r.get("stage") == "after_edge"),
                "rejected_peakness": sum(1 for r in full_audit if r.get("stage") == "after_peakness"),
                "after_peakness_scored": len(scored),
                "final_kept": len(peaks),
                "empty_reason": empty_reason,
                "best_score": best_score,
                "min_best_score": min_best,
            },
            "threshold_map": threshold_map,
            "num_peaks_selected": len(peaks),
            "num_kept_contours": len(peaks),
            "components": peaks,
            "bounding_boxes": peaks,
            "peak_audit_rows": peak_audit_rows,
            "contour_audit_rows": peak_audit_rows,
            "gt_audit_rows": gt_audit_rows,
            "top_scores_ranked": [float(pk["peak_score"]) for pk in peaks],
            "final_num_contours": len(peaks),
            "final_num_centers": len(peaks),
            "num_centers_drawn": len(peaks),
        }
        if empty_reason:
            metadata["skip_reason"] = empty_reason

        return out.astype(bool), metadata

    def _empty_metadata(self, reason: str) -> Dict[str, Any]:
        return {
            "method": self.name,
            "skip_reason": reason,
            "num_peaks_selected": 0,
            "peak_nms_candidate_rows": [],
            "peak_nms_full_audit_rows": [],
            "peak_nms_pipeline": {"final_kept": 0, "empty_reason": reason},
            "components": [],
            "bounding_boxes": [],
            "gt_audit_rows": [],
            "contour_audit_rows": [],
        }


def save_peak_nms_audit_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(
            "pair_id,candidate_index,x,y,center_value,peakness,ring_mean,local_std,on_edge,valid_ok,"
            "edge_distance_px,final_score,kept,rank,nearest_gt_dist_px,within_5px,within_7px\n",
            encoding="utf-8",
        )
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
