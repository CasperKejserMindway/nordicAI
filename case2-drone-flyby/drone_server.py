"""Drone Flyby endpoint: detect at zoom, remember, dead-reckon, re-emit.

Three parts, each replaceable on its own:

* ``Detector``   -- a fine-tuned YOLO11 on the 960x540 view, boxes lifted to
                    source pixels through ``view.source_region_xyxy``.
* ``Tracker``    -- one track per (object, class). Every frame every track is
                    moved by the shared ground-plane affine (all objects sit on
                    the same plane and the drone flies straight), and fresh
                    detections snap matched tracks back and refine the affine's
                    speed factor. Unobserved tracks are still emitted: the frame
                    is scored against every object in it, seen or not.
* ``Planner``    -- where to point the camera next. Reads the constraints from
                    the request and checks the move locally, so a command is
                    never refused.

Environment:
  DRONE_WEIGHTS   path to .pt (default runs/n_all/weights/best.pt); 'none' = no detector
  DRONE_IMGSZ     inference size, default 960 (the view's long side; rect letterbox)
  DRONE_CONF      detector confidence floor, default 0.05
  DRONE_POLICY    l2band | l1band | mixed  (default l2band)
  DRONE_PORT      default 9053
"""
from __future__ import annotations

import json, logging, math, os, statistics, sys, threading, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'competition' / 'drone-flyby'))
from dtos import (  # noqa: E402
    IMAGE_HEIGHT, IMAGE_WIDTH, OBJECT_CLASSES, SOURCE_REGION_SIZES,
    DroneFlybyPredictRequestDto, DroneFlybyPredictResponseDto, DroneFlybyPredictionDto, RequestedViewDto,
)
from utils import decode_image, describe_camera_rejection  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('drone')

WEIGHTS = os.environ.get('DRONE_WEIGHTS', str(HERE / 'runs' / 'n_all' / 'weights' / 'best.pt'))
IMGSZ = int(os.environ.get('DRONE_IMGSZ', '960'))
CONF = float(os.environ.get('DRONE_CONF', '0.05'))
POLICY = os.environ.get('DRONE_POLICY', 'l2band')
# Per-class linear scale applied to every emitted box, about its centre. The ground truth is
# a projected 3D bounding box, generous around the silhouette; hand labels hugged the pixels
# and the detector learned that on every labelled class. Measured 2026-09-18 04:30: the
# Helsinki GT is 1.2-1.8x the linear size of the hand labels per class, and a helicopter
# oracle enlarged 1.3x scored 55x the tight one. Format: "cls=1.4,cls=1.2"; DRONE_BOX_SCALE_ALL
# applies a uniform factor to every class not listed.
# Class hedging: detections below CONF but above HEDGE_CONF never create tracks; they attach to an
# overlapping track as alternative classes and are emitted as extra boxes ranked below the track's own.
# COCO AP is per class, so a wrong-class box ranked low costs almost nothing, while the right class
# absent costs the whole object. 0 disables.
# Sliced inference. At level 1 an object is 10-23 px in the 960-wide view; cut the view into a
# grid of overlapping tiles and upscale each 2x and the same object is 20-46 px -- exactly the
# size the detector sees in the level-2 native tiles it was trained on. Costs one forward pass
# per tile, and the frame budget is 333 ms against ~30 ms used. 0 or 1 disables.
# Second detector, used only for the classes the first one was never taught. The hand-labelled
# models know 11 classes well and propose nothing at all for condor/spacecraft/medium_plane/jammer;
# the synthetic models know all 16 but are weaker on the hand-labelled ones (small_launcher 2
# detections against 51). Disjoint class sets, so merging costs one extra forward pass (~25 ms of a
# 333 ms budget) and cannot conflict. DRONE_WEIGHTS2_CLASSES lists what to accept from it.
WEIGHTS2 = os.environ.get('DRONE_WEIGHTS2', '')
CLASSES2 = {c for c in os.environ.get('DRONE_WEIGHTS2_CLASSES', '').split(',') if c}
SLICE = int(os.environ.get('DRONE_SLICE', '0'))          # NxN grid over the view
SLICE_OVERLAP = float(os.environ.get('DRONE_SLICE_OVERLAP', '0.25'))
SLICE_LEVELS = {int(v) for v in os.environ.get('DRONE_SLICE_LEVELS', '0,1').split(',') if v != ''}
# Emit every track's box under ALL 16 class names, the extras at a floor confidence far below the
# track's own. A diagnostic first: if the scene contains classes I never name, and one of them sits
# where I am already tracking something, its AP goes from 0 to non-zero and the total moves. Per-class
# AP means a wrong-class box ranked at the bottom is nearly free.
# Diagnostic: emit only this class. The metric is a macro average over the classes present in the
# scene, so serving one class whose AP is known gives score = AP_class / N and hence N, the number
# of scored classes -- which fixes the ceiling for a solution that names only some of them.
# A single spurious detection becomes a track that is then dead-reckoned and emitted for dozens of
# frames: one false positive multiplied by its whole remaining life. Requiring a second sighting
# before a track is ever emitted costs a little recall at entry and removes that entire family of
# false positives. Measured 2026-09-18: 787 of 1512 emitted boxes matched no known object.
MIN_OBS = int(os.environ.get('DRONE_MIN_OBS', '1'))
# Canonical box geometry. The ground truth is a projected 3D box whose size is near-deterministic
# given class and image row (perspective law w = W_c * exp(k*(y - y0)), fitted on the Helsinki
# annotations into box_prior.json). DRONE_BOX_GEOM=1 replaces the detector's regressed width and
# height with that prior, keeping the detector's centre. It fixes both scale and ASPECT (large_tower
# is emitted at w/h 0.50 against a ground truth of 0.97). Overrides DRONE_BOX_SCALE when on.
BOX_GEOM = os.environ.get('DRONE_BOX_GEOM', '0') == '1'
BOX_GEOM_MIX = float(os.environ.get('DRONE_BOX_GEOM_MIX', '1.0'))   # 1 = prior only, 0 = detector only
# Aspect-only correction: keep the emitted box's area but reshape it to the class's Helsinki
# aspect. Object SIZE varies between scenes (the validation hangar is ~0.8x the Helsinki one), so an
# absolute size prior scored worse on the board; the asset's SHAPE does not vary. The towers are the
# case in point: emitted at w/h ~0.5 (the silhouette the detector learned), ground truth 0.97.
# Comma-separated class list, or "all".
BOX_ASPECT = {c for c in os.environ.get('DRONE_BOX_ASPECT', '').split(',') if c}
try:
    _bp = json.load(open(HERE / 'box_prior.json'))
    BOX_PRIOR = {c: (v['w'], v['h']) for c, v in _bp['classes'].items()}; BOX_K = _bp['k']; BOX_Y0 = _bp['y0']
except Exception:
    BOX_PRIOR = {}; BOX_K = 0.0; BOX_Y0 = 1080
ONLY_CLASS = os.environ.get('DRONE_ONLY_CLASS', '')
ALL_CLASSES = float(os.environ.get('DRONE_ALL_CLASSES', '0'))     # confidence for the extra names, 0 = off
HEDGE_CONF = float(os.environ.get('DRONE_HEDGE_CONF', '0'))
HEDGE_RATIO = float(os.environ.get('DRONE_HEDGE_RATIO', '0.5'))   # alt confidence = min(alt, main * ratio)
BOX_SCALE_ALL = float(os.environ.get('DRONE_BOX_SCALE_ALL', '1.0'))
BOX_SCALE = {k: float(v) for k, v in (kv.split('=') for kv in os.environ.get('DRONE_BOX_SCALE', '').split(',') if '=' in kv)}
PORT = int(os.environ.get('DRONE_PORT', '9053'))
CAPTURE = os.environ.get('DRONE_CAPTURE', '')   # directory: save every view + request meta + our response

Box = Tuple[float, float, float, float]  # source pixels, x1 y1 x2 y2


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #

class Detector:
    def __init__(self, weights: str):
        self.model = None
        self.names: Dict[int, str] = {}
        self.last_hedges: List = []
        self.model2 = None; self.names2 = {}        # set before any early return: 'gt'/'none' paths use it too
        self.gt = None
        if weights.lower() == 'gt':
            # Perfect detector on the helsinki scene: measures the tracker + camera policy alone.
            from utils import frame_numbers, load_annotations
            self.gt = {f: load_annotations(f) for f in frame_numbers()}
            log.warning('GT detector: this is an upper bound for the tracker/policy, not a model')
            return
        if weights.lower() == 'none' or not Path(weights).exists():
            log.warning('no detector weights at %s -- serving tracks only', weights)
            return
        from ultralytics import YOLO
        self.model = YOLO(weights)
        if WEIGHTS2 and Path(WEIGHTS2).exists() and CLASSES2:
            self.model2 = YOLO(WEIGHTS2); self.names2 = dict(self.model2.names)
            log.info('second detector %s for %d classes: %s', Path(WEIGHTS2).name, len(CLASSES2), sorted(CLASSES2))
        self.names = dict(self.model.names)
        unknown = [n for n in self.names.values() if n not in OBJECT_CLASSES]
        if unknown:
            log.warning('weights emit classes outside the protocol, they will be dropped: %s', unknown[:5])
        self.warmup()

    def warmup(self):
        if self.model is None:
            return
        # Realistic warm-up: textured images, several passes -- a blank image leaves the NMS and
        # post-processing paths cold, and the first real frame cost 517 ms (two frames lost).
        rng = np.random.default_rng(0)
        for _ in range(6):
            dummy = rng.integers(0, 255, (540, 960, 3), dtype=np.uint8)
            dummy = cv2.GaussianBlur(dummy, (9, 9), 0)
            self.model.predict(dummy, imgsz=IMGSZ, conf=CONF, iou=0.6, max_det=120, verbose=False, device='mps')
        dummy = np.zeros((540, 960, 3), np.uint8)
        t = time.perf_counter(); self.model.predict(dummy, imgsz=IMGSZ, conf=CONF, verbose=False, device='mps')
        log.info('detector warm: %s, imgsz %d, %.0f ms on a blank view', Path(WEIGHTS).name, IMGSZ, (time.perf_counter() - t) * 1000)

    def _raw(self, img: np.ndarray, floor: float):
        """Detections on one image as (name, conf, x1, y1, x2, y2) in that image's pixels."""
        res = self.model.predict(img, imgsz=IMGSZ, conf=floor, iou=0.6, max_det=200, verbose=False, device='mps')[0]
        out = []
        if res.boxes is None or len(res.boxes) == 0:
            return out
        for (x1, y1, x2, y2), c, k in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int)):
            n = self.names.get(int(k))
            if n in OBJECT_CLASSES:
                out.append((n, float(c), float(x1), float(y1), float(x2), float(y2)))
        return out

    def _sliced(self, view: np.ndarray, floor: float, level: int):
        """Full-view pass plus an NxN grid of overlapping tiles, each upscaled, merged by class-wise NMS."""
        h, w = view.shape[:2]
        dets = self._raw(view, floor)
        if SLICE > 1 and level in SLICE_LEVELS:
            step_x, step_y = w / SLICE, h / SLICE
            ox, oy = step_x * SLICE_OVERLAP, step_y * SLICE_OVERLAP
            for iy in range(SLICE):
                for ix in range(SLICE):
                    x0 = int(max(0, ix * step_x - ox)); y0 = int(max(0, iy * step_y - oy))
                    x1 = int(min(w, (ix + 1) * step_x + ox)); y1 = int(min(h, (iy + 1) * step_y + oy))
                    tile = view[y0:y1, x0:x1]
                    if tile.size == 0:
                        continue
                    up = cv2.resize(tile, (w, h), interpolation=cv2.INTER_CUBIC)
                    sx, sy = (x1 - x0) / w, (y1 - y0) / h
                    for n, c, a, b, cc, d in self._raw(up, floor):
                        dets.append((n, c, x0 + a * sx, y0 + b * sy, x0 + cc * sx, y0 + d * sy))
        # class-wise NMS over the merged set
        keep = []
        for n, c, a, b, cc, d in sorted(dets, key=lambda t: -t[1]):
            box = (a, b, cc, d)
            if any(kn == n and iou(kb, box) > 0.55 for kn, _, kb in keep):
                continue
            keep.append((n, c, box))
        return keep

    def __call__(self, view: np.ndarray, region: List[int], frame: int = -1) -> List[Tuple[str, float, Box]]:
        """Detections as (class, conf, source-pixel box)."""
        if self.gt is not None:
            rx1, ry1, rx2, ry2 = region
            out = []
            for a in self.gt.get(frame, []):
                x1, y1, x2, y2 = a['bbox']; cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                if rx1 <= cx <= rx2 and ry1 <= cy <= ry2 and min(x2 - x1, y2 - y1) * (1 if rx2 - rx1 == 960 else 960 / (rx2 - rx1)) >= 6:
                    out.append((a['object_id'], 0.9, (float(x1), float(y1), float(x2), float(y2))))
            return out
        if self.model is None:
            return []
        h, w = view.shape[:2]
        rx1, ry1, rx2, ry2 = region
        sx, sy = (rx2 - rx1) / w, (ry2 - ry1) / h
        floor = min(CONF, HEDGE_CONF) if HEDGE_CONF > 0 else CONF
        level = 0 if (rx2 - rx1) >= IMAGE_WIDTH else (1 if (rx2 - rx1) >= IMAGE_WIDTH // 2 else 2)
        merged = self._sliced(view, floor, level)
        if self.model2 is not None:
            res2 = self.model2.predict(view, imgsz=IMGSZ, conf=floor, iou=0.6, max_det=120, verbose=False, device='mps')[0]
            if res2.boxes is not None and len(res2.boxes):
                for bb, c2, k2 in zip(res2.boxes.xyxy.cpu().numpy(), res2.boxes.conf.cpu().numpy(), res2.boxes.cls.cpu().numpy().astype(int)):
                    n2 = self.names2.get(int(k2))
                    if n2 in CLASSES2:
                        merged.append((n2, float(c2), (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))))
        out = []; self.last_hedges = []
        for name, c, (x1, y1, x2, y2) in merged:
            box = (rx1 + x1 * sx, ry1 + y1 * sy, rx1 + x2 * sx, ry1 + y2 * sy)
            if float(c) < CONF:
                self.last_hedges.append((name, float(c), box)); continue
            if is_partial(box, region):
                # Clipped by the crop, not by the frame: the object continues outside the view.
                # Such a box must never overwrite a track -- it would be dead-reckoned truncated.
                out.append((name, float(c), box, True))
            else:
                out.append((name, float(c), box))
        return out


def is_partial(box: Box, region: List[int], margin: float = 3.0) -> bool:
    """True if the box touches a crop edge that is not also a frame edge."""
    rx1, ry1, rx2, ry2 = region
    return ((box[0] <= rx1 + margin and rx1 > 0) or (box[1] <= ry1 + margin and ry1 > 0) or
            (box[2] >= rx2 - margin and rx2 < IMAGE_WIDTH) or (box[3] >= ry2 - margin and ry2 < IMAGE_HEIGHT))


# --------------------------------------------------------------------------- #
# Motion model: one affine per frame step, shared by every object
# --------------------------------------------------------------------------- #

class Motion:
    """M(lambda) = I + lambda * (M_prior - I): the prior fitted on helsinki, scaled by a
    speed factor that is refined online from re-detections. A different drone speed or
    altitude changes translation and perspective expansion together, and lambda covers
    both to first order. Small residual offsets are absorbed by (tx, ty)."""

    def __init__(self):
        prior = json.load(open(HERE / 'motion_prior.json'))
        self.Mp = np.array(prior['affine_2x3'], float)
        # Initial speed factor. The Helsinki prior is lambda=1 by construction; on the validation
        # city the true scroll speed measured from 112 consecutive detection pairs is 1.0315
        # (95% CI 1.026-1.039) while the online refinement only reached ~1.007-1.015, a residual
        # that accumulates ~27 px of lag over an object's unobserved descent -- enough to take
        # 40-60 px boxes below IoU 0.5. DRONE_LAMBDA sets the start; DRONE_LAMBDA_GAIN scales the
        # refinement so a different evaluation speed is still learned.
        self.lam = float(os.environ.get('DRONE_LAMBDA', '1.0'))
        self.gain_scale = float(os.environ.get('DRONE_LAMBDA_GAIN', '1.0'))
        self.offset = np.zeros(2)
        self.n_updates = 0

    def matrix(self) -> np.ndarray:
        I = np.array([[1, 0, 0], [0, 1, 0]], float)
        M = I + self.lam * (self.Mp - I)
        M[:, 2] += self.offset
        return M

    def step_box(self, box: Box, k: int) -> Box:
        if k <= 0:
            return box
        M = self.matrix()
        x1, y1, x2, y2 = box
        p = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]], float)
        for _ in range(k):
            p = p @ M[:, :2].T + M[:, 2]
        return (float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max()))

    def refine(self, pairs: List[Tuple[Box, Box, int]]):
        """pairs: (predicted box, detected box, frames since last observation)."""
        rel, off = [], []
        M = self.matrix()
        for pred, det, k in pairs:
            if k <= 0:
                continue
            pc = np.array([(pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2])
            dc = np.array([(det[0] + det[2]) / 2, (det[1] + det[3]) / 2])
            r = (dc - pc) / k                       # residual per frame
            prior_dy = (M[1, :2] @ pc + M[1, 2]) - pc[1]
            if abs(prior_dy) > 5:
                rel.append(r[1] / prior_dy)
            off.append(r)
        if not off:
            return
        gain = (0.25 if self.n_updates < 5 else 0.12) * self.gain_scale
        if rel:
            self.lam *= 1 + gain * float(np.clip(statistics.median(rel), -0.5, 0.5))
            self.lam = float(np.clip(self.lam, 0.3, 3.0))
        o = np.median(np.array(off), axis=0)
        self.offset += gain * np.clip(o, -10, 10) * np.array([1.0, 0.35])  # y mostly explained by lambda
        self.offset = np.clip(self.offset, -15, 15)
        self.n_updates += 1


# --------------------------------------------------------------------------- #
# Tracker
# --------------------------------------------------------------------------- #

@dataclass
class Track:
    cls: str
    box: Box
    conf: float
    last_obs: int
    obs: int = 1
    misses: int = 0
    id: int = 0
    alts: Dict[str, float] = field(default_factory=dict)


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def centre_dist(a: Box, b: Box) -> float:
    return math.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2)


class Tracker:
    # Tunable without retraining. The metric is COCO AP with maxDets 100 per class per
    # image, so a track kept alive at low confidence ranks below the good boxes and can
    # only add recall; a track that is dropped emits nothing at all.
    AGE_DECAY = float(os.environ.get('DRONE_AGE_DECAY', '0.985'))   # per frame unobserved
    MISS_DECAY = float(os.environ.get('DRONE_MISS_DECAY', '0.55'))  # per frame in view and not seen
    MIN_EMIT = float(os.environ.get('DRONE_MIN_EMIT', '0.02'))

    def __init__(self):
        self.tracks: List[Track] = []
        self.motion = Motion()
        self.next_id = 1

    def step(self, frame: int, k: int, dets: List[Tuple[str, float, Box]], region: List[int], level: int, hedges: List = ()):
        # 1. move every track k frames forward
        if k > 0:
            for t in self.tracks:
                t.box = self.motion.step_box(t.box, k)
        # 2. match detections to tracks of the same class
        used = set(); pairs = []
        for det in sorted(dets, key=lambda d: -d[1]):
            cls, conf, box = det[0], det[1], det[2]
            partial = len(det) > 3 and det[3]
            best, best_s = None, 0.0
            size = max(box[2] - box[0], box[3] - box[1], 8.0)
            for i, t in enumerate(self.tracks):
                if i in used or t.cls != cls:
                    continue
                s = iou(t.box, box)
                if s < 0.15 and centre_dist(t.box, box) > 0.8 * size:
                    continue
                s = max(s, 0.15 + 0.1 * (1 - centre_dist(t.box, box) / (0.8 * size)))
                if s > best_s:
                    best, best_s = i, s
            if best is None:
                if not partial:
                    self.tracks.append(Track(cls, box, conf, frame, id=self.next_id)); self.next_id += 1
            else:
                t = self.tracks[best]; used.add(best)
                if partial:
                    t.misses = 0            # seen, but the box is truncated: keep dead-reckoning
                    continue
                if t.obs >= 1 and frame - t.last_obs > 0:
                    pairs.append((t.box, box, frame - t.last_obs))
                if t.obs >= 2 and frame - t.last_obs <= 2 and iou(t.box, box) > 0.5:
                    # Established track, fresh prediction: average out the detector's box noise
                    # so the anchor we dead-reckon from is steadier than one detection.
                    w = 0.6
                    box = tuple(w * b + (1 - w) * pb for b, pb in zip(box, t.box))
                t.box = box
                t.conf = max(conf, 0.5 * t.conf + 0.5 * conf)
                t.last_obs = frame; t.obs += 1; t.misses = 0
        # 2b. class hedges: attach low-confidence alternative classes to the track they overlap
        for cls, conf, box in hedges:
            best = max(((iou(t.box, box), t) for t in self.tracks), key=lambda x: x[0], default=(0.0, None))
            if best[1] is not None and best[0] >= 0.5 and cls != best[1].cls:
                t = best[1]; t.alts[cls] = max(t.alts.get(cls, 0.0), conf)
        # 3. refine the shared motion from re-observed tracks (need a few, and only ones that moved)
        good = [p for p in pairs if p[2] <= 12]
        if len(good) >= 2:
            self.motion.refine(good)
        # 4. penalise tracks that were in view and not seen
        rx1, ry1, rx2, ry2 = region
        m = 6.0
        for i, t in enumerate(self.tracks):
            if i in used:
                continue
            inside = t.box[0] >= rx1 + m and t.box[1] >= ry1 + m and t.box[2] <= rx2 - m and t.box[3] <= ry2 - m
            if inside and level >= 1:
                t.misses += 1
        # 5. drop what has left the frame or faded
        alive = []
        for t in self.tracks:
            if t.box[2] <= 0 or t.box[3] <= 0 or t.box[0] >= IMAGE_WIDTH or t.box[1] >= IMAGE_HEIGHT:
                continue
            if self.emit_conf(t, frame) < self.MIN_EMIT:
                continue
            alive.append(t)
        self.tracks = alive
        # 6. same-class duplicate suppression
        self.tracks.sort(key=lambda t: -self.emit_conf(t, frame))
        keep: List[Track] = []
        for t in self.tracks:
            if any(o.cls == t.cls and iou(o.box, t.box) > 0.6 for o in keep):
                continue
            keep.append(t)
        self.tracks = keep

    def emit_conf(self, t: Track, frame: int) -> float:
        c = t.conf * (self.AGE_DECAY ** max(0, frame - t.last_obs)) * (self.MISS_DECAY ** t.misses)
        if t.obs >= 2:
            c = min(1.0, c * 1.15)
        return c

    def annotations(self, frame: int, W: int, H: int) -> List[DroneFlybyPredictionDto]:
        out = []
        def emit(cls, box, conf):
            f = BOX_SCALE.get(cls, BOX_SCALE_ALL)
            bx1, by1, bx2, by2 = box
            if BOX_GEOM and cls in BOX_PRIOR:
                cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
                g = math.exp(BOX_K * (cy - BOX_Y0)); pw, ph = BOX_PRIOR[cls][0] * g, BOX_PRIOR[cls][1] * g
                w = BOX_GEOM_MIX * pw + (1 - BOX_GEOM_MIX) * (bx2 - bx1); h = BOX_GEOM_MIX * ph + (1 - BOX_GEOM_MIX) * (by2 - by1)
                bx1, by1, bx2, by2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
                f = 1.0
            if f != 1.0:
                cx, cy, hw, hh = (bx1 + bx2) / 2, (by1 + by2) / 2, (bx2 - bx1) / 2 * f, (by2 - by1) / 2 * f
                bx1, by1, bx2, by2 = cx - hw, cy - hh, cx + hw, cy + hh
            if BOX_ASPECT and ('all' in BOX_ASPECT or cls in BOX_ASPECT) and cls in BOX_PRIOR:
                cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
                area = max(1.0, (bx2 - bx1) * (by2 - by1)); a = BOX_PRIOR[cls][0] / BOX_PRIOR[cls][1]
                w, h = math.sqrt(area * a), math.sqrt(area / a)
                bx1, by1, bx2, by2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            x1 = max(0.0, min(1.0, bx1 / W)); x2 = max(0.0, min(1.0, bx2 / W))
            y1 = max(0.0, min(1.0, by1 / H)); y2 = max(0.0, min(1.0, by2 / H))
            x1, y1, x2, y2 = (round(v, 6) for v in (x1, y1, x2, y2))
            if 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1:
                out.append(DroneFlybyPredictionDto(object_id=cls, bbox=[x1, y1, x2, y2], confidence=round(max(0.0, min(1.0, conf)), 4)))
        for t in self.tracks:
            if t.obs < MIN_OBS:
                continue
            if ONLY_CLASS and t.cls != ONLY_CLASS:
                continue
            c = self.emit_conf(t, frame)
            emit(t.cls, t.box, c)
            for alt, ac in t.alts.items():
                emit(alt, t.box, min(ac, c * HEDGE_RATIO))
            if ALL_CLASSES > 0:
                for other in OBJECT_CLASSES:
                    if other != t.cls and other not in t.alts:
                        emit(other, t.box, ALL_CLASSES)
        out.sort(key=lambda a: -a.confidence)
        return out[:500]


# --------------------------------------------------------------------------- #
# Camera planner
# --------------------------------------------------------------------------- #

class Planner:
    """Sweeps a band near the top of the frame, where objects enter. The scene
    scrolls downwards ~65 px/frame, so a fixed band sees everything once."""

    L2_XS = [480, 960, 1440, 1920, 2400, 2880, 3360]   # 480-px steps, all <= the 551 L2 cap
    L1_XS = [960, 1920, 2880]                           # 960-px steps, all <= the 1102 L1 cap
    L2_Y = int(os.environ.get('DRONE_L2_Y', '270'))      # survey passes move the level-2 band
    L1_Y = int(os.environ.get('DRONE_L1_Y', '540'))      # 540 = top half, 1620 = bottom half
    # Six L1 positions covering the frame; consecutive moves are 960 or 1080 px, under the 1102 cap.
    L1_SNAKE = [(960, 540), (1920, 540), (2880, 540), (2880, 1620), (1920, 1620), (960, 1620)]

    def __init__(self, policy: str):
        self.policy = policy
        self.direction = 1
        self.n = 0

    def reset(self):
        self.direction = 1; self.n = 0

    def next_view(self, req: DroneFlybyPredictRequestDto) -> Optional[RequestedViewDto]:
        v = req.view; cur = (v.center_x, v.center_y); lvl = v.resolution_level
        self.n += 1
        target = self._target(lvl, cur)
        if target is None:
            return None
        tl, tx, ty = target
        why = describe_camera_rejection(lvl, cur, tl, (tx, ty))
        if why is not None:
            # Fall back to the nearest legal step towards the target at the same level.
            log.warning('planner would be refused (%s); stepping instead', why)
            tl2 = tl if tl in req.camera_constraints.allowed_resolution_levels else lvl
            b = req.camera_constraints.bounds_for_level(tl2)
            if b is None:
                return None
            cap = req.camera_constraints.maximum_center_delta * 0.95
            dx, dy = tx - cur[0], ty - cur[1]; d = math.hypot(dx, dy)
            if d > cap:
                dx, dy = dx * cap / d, dy * cap / d
            tx = int(min(max(cur[0] + dx, b.minimum_center_x), b.maximum_center_x))
            ty = int(min(max(cur[1] + dy, b.minimum_center_y), b.maximum_center_y))
            tl = tl2
            if describe_camera_rejection(lvl, cur, tl, (tx, ty)) is not None:
                return None
        return RequestedViewDto(resolution_level=int(tl), center_x=int(tx), center_y=int(ty))

    def _sweep(self, xs: List[int], cur_x: int) -> int:
        # nearest index to the current x, then one step in the sweep direction, bouncing at the ends
        i = min(range(len(xs)), key=lambda j: abs(xs[j] - cur_x))
        j = i + self.direction
        if j < 0 or j >= len(xs):
            self.direction = -self.direction; j = i + self.direction
        return xs[j]

    def _target(self, lvl: int, cur: Tuple[int, int]):
        if self.policy == 'l1bottom':
            # Survey-only: sweep the level-1 band at whatever DRONE_L1_Y says, three positions,
            # revisiting each every third frame. Used to image ground the serving policy never
            # shows the detector, so that objects there can be found and labelled.
            if lvl == 0:
                return (1, 960, self.L1_Y)
            if lvl == 2:
                return (1, min(max(cur[0], 960), 2880), min(max(cur[1], 540), 1620))
            if cur[1] != self.L1_Y:
                return (1, cur[0], self.L1_Y)
            return (1, self._sweep(self.L1_XS, cur[0]), self.L1_Y)
        if self.policy == 'l1dive':
            # l1band, plus every Nth frame a level-2 look at the top band in the SAME column and
            # straight back. Legal by construction: L1 (x,540) -> L2 (x,270) is 270 px against the
            # 1102 cap, L2 (x,270) -> L1 (x,540) is 270 px against the 551 cap. It exists for the
            # classes only level 2 resolves (medium_launcher), at the cost of one L1 frame in N.
            every = int(os.environ.get('DRONE_DIVE_EVERY', '4'))
            if lvl == 0:
                return (1, 960, self.L1_Y)
            if lvl == 2:
                return (1, min(max(cur[0], 960), 2880), self.L1_Y)
            if self.n % every == 0:
                return (2, min(max(cur[0], 480), 3360), self.L2_Y)
            return (1, self._sweep(self.L1_XS, cur[0]), self.L1_Y)
        if self.policy == 'l1snake':
            if lvl == 0:
                return (1, 960, 540)
            if lvl == 2:
                return (1, min(max(cur[0], 960), 2880), min(max(cur[1], 540), 1620))
            i = min(range(6), key=lambda j: math.hypot(self.L1_SNAKE[j][0] - cur[0], self.L1_SNAKE[j][1] - cur[1]))
            x, y = self.L1_SNAKE[(i + 1) % 6]
            return (1, x, y)
        if self.policy == 'l1band':
            if lvl == 0:
                return (1, 960, self.L1_Y)
            if lvl == 2:
                return (1, min(max(cur[0], 960), 2880), self.L1_Y)
            return (1, self._sweep(self.L1_XS, cur[0]), self.L1_Y)
        # l2band (default) and mixed
        if lvl == 0:
            return (1, 960, self.L1_Y)                    # L0 -> L1 first, L2 is two steps away
        if lvl == 1:
            # DRONE_L2_START_X shifts the phase of the top-band sweep; used to capture the
            # entry band completely across several runs (each sweep period leaves gaps).
            x = int(os.environ.get('DRONE_L2_START_X', cur[0] - 480))
            x = min(max(x, 480), 3360)
            return (2, x, self.L2_Y)
        if self.policy == 'mixed' and self.n % 6 == 0:
            # every sixth frame, one L1 look at the lower half to re-anchor old tracks
            return (1, min(max(cur[0], 960), 2880), min(cur[1] + 500, 1620))
        return (2, self._sweep(self.L2_XS, cur[0]), self.L2_Y)


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #

@dataclass
class Telemetry:
    total_ms: List[float] = field(default_factory=list)
    detect_ms: List[float] = field(default_factory=list)
    frames: int = 0
    errors: int = 0
    refused: int = 0

    def add(self, total, detect):
        self.total_ms.append(total); self.detect_ms.append(detect); self.frames += 1
        if len(self.total_ms) > 2000:
            del self.total_ms[:1000]; del self.detect_ms[:1000]

    def summary(self):
        def pct(xs, q):
            if not xs:
                return None
            s = sorted(xs); return round(s[min(len(s) - 1, int(q * len(s)))], 1)
        return {'frames': self.frames, 'errors': self.errors, 'camera_refused': self.refused,
                'total_ms': {'p50': pct(self.total_ms, .5), 'p95': pct(self.total_ms, .95), 'max': pct(self.total_ms, 1.0)},
                'detect_ms': {'p50': pct(self.detect_ms, .5), 'p95': pct(self.detect_ms, .95)}}


class Service:
    def __init__(self):
        self.lock = threading.Lock()
        self.detector = Detector(WEIGHTS)
        self.tracker = Tracker()
        self.planner = Planner(POLICY)
        self.seq: Optional[str] = None
        self.last_frame: int = -1
        self.tele = Telemetry()
        self.start = time.time()
        self.last_request = 0.0
        self._warm_rng = np.random.default_rng(1)

    def keep_warm_once(self):
        """Run from the event-loop thread only (see the startup task). A second thread
        touching MPS hung the whole process on 2026-09-17 17:47 -- GIL held inside a
        native call forever, 0.0 on a validation. Never call this from another thread."""
        if time.time() - self.last_request < 10 or self.detector.model is None:
            return
        with self.lock:
            dummy = cv2.GaussianBlur(self._warm_rng.integers(0, 255, (540, 960, 3), dtype=np.uint8), (9, 9), 0)
            self.detector.model.predict(dummy, imgsz=IMGSZ, conf=CONF, iou=0.6, max_det=120, verbose=False, device='mps')

    def reset(self, seq: str):
        log.info('new sequence %s -- resetting tracker and planner', seq)
        self.tracker = Tracker(); self.planner = Planner(POLICY); self.seq = seq; self.last_frame = -1

    def predict(self, req: DroneFlybyPredictRequestDto) -> DroneFlybyPredictResponseDto:
        t0 = time.perf_counter()
        self.last_request = time.time()
        with self.lock:
            if req.sequence_id != self.seq or req.frame <= self.last_frame:
                self.reset(req.sequence_id)
            k = req.frame - self.last_frame if self.last_frame >= 0 else 0
            self.last_frame = req.frame
            if req.camera_command_feedback is not None:
                self.tele.refused += 1
                log.warning('camera command from frame %s refused: %s', req.camera_command_feedback.frame, req.camera_command_feedback.reason)
            dets: List = []; det_ms = 0.0
            try:
                view = decode_image(req.view.image)
                t1 = time.perf_counter()
                dets = self.detector(view, list(req.view.source_region_xyxy), req.frame)
                det_ms = (time.perf_counter() - t1) * 1000
            except Exception:
                self.tele.errors += 1
                log.exception('detector failed on frame %s; emitting propagated tracks only', req.frame)
            try:
                self.tracker.step(req.frame, k, dets, list(req.view.source_region_xyxy), req.view.resolution_level, self.detector.last_hedges)
                anns = self.tracker.annotations(req.frame, req.original_width, req.original_height)
            except Exception:
                self.tele.errors += 1
                log.exception('tracker failed on frame %s', req.frame)
                anns = []
            try:
                nxt = self.planner.next_view(req)
            except Exception:
                self.tele.errors += 1
                log.exception('planner failed on frame %s', req.frame)
                nxt = None
        total = (time.perf_counter() - t0) * 1000
        self.tele.add(total, det_ms)
        if CAPTURE:
            try:
                d = Path(CAPTURE) / req.sequence_id[:12]; d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / f'f{req.frame:04d}_L{req.view.resolution_level}_{req.view.center_x}_{req.view.center_y}.png'), view)
                meta = req.model_dump(); meta['view'].pop('image', None)
                meta['_response'] = {'annotations': [a.model_dump() for a in anns], 'requested_view': nxt.model_dump() if nxt else None,
                                     'detections': [(d0[0], round(float(d0[1]), 3), [round(float(v), 1) for v in d0[2]]) for d0 in dets], 'k': k, 'total_ms': round(total, 1)}
                (d / f'f{req.frame:04d}.json').write_text(json.dumps(meta))
            except Exception:
                log.exception('capture failed')
        log.info('f%-4d idx%-4d k%d L%d (%4d,%4d) dets %2d tracks %3d lam %.3f | det %3.0f ms total %3.0f ms',
                 req.frame, req.frame_index, k, req.view.resolution_level, req.view.center_x, req.view.center_y,
                 len(dets), len(anns), self.tracker.motion.lam, det_ms, total)
        return DroneFlybyPredictResponseDto(request_id=req.request_id, frame=req.frame, annotations=anns, requested_view=nxt)


service = Service()
app = FastAPI()


@app.on_event('startup')
async def _start_keep_warm():
    import asyncio

    async def loop():
        while True:
            await asyncio.sleep(20)
            try:
                service.keep_warm_once()      # same thread as the request handlers
            except Exception:
                log.exception('keep-warm failed')
    asyncio.get_event_loop().create_task(loop())


@app.post('/predict')
@app.post('/')
async def predict_endpoint(request: Request):
    body = await request.json()
    try:
        req = DroneFlybyPredictRequestDto.model_validate(body)
    except Exception:
        service.tele.errors += 1
        log.exception('request failed validation')
        return JSONResponse({'request_id': str(body.get('request_id', '')), 'frame': int(body.get('frame', 0)), 'annotations': []})
    try:
        resp = service.predict(req)
        return JSONResponse(resp.model_dump(exclude_none=True))
    except Exception:
        service.tele.errors += 1
        log.exception('predict failed on frame %s -- empty but valid response', req.frame)
        return JSONResponse({'request_id': req.request_id, 'frame': req.frame, 'annotations': []})


@app.get('/api')
def api():
    return {'service': 'drone-flyby', 'uptime_s': round(time.time() - service.start), 'weights': Path(WEIGHTS).name,
            'imgsz': IMGSZ, 'policy': POLICY, 'box_scale_all': BOX_SCALE_ALL, 'box_scale': BOX_SCALE, 'hedge_conf': HEDGE_CONF, 'sequence': service.seq, 'tracks': len(service.tracker.tracks),
            'motion': {'lambda': round(service.tracker.motion.lam, 4), 'offset': [round(float(v), 2) for v in service.tracker.motion.offset],
                       'updates': service.tracker.motion.n_updates},
            **service.tele.summary()}


@app.get('/')
def index():
    return 'Your endpoint is running!'


if __name__ == '__main__':
    import uvicorn
    # Bounded shutdown: on SIGTERM uvicorn otherwise waits for every open connection to close, and the
    # reverse tunnel keeps connections open, so a restart could hold the port for a minute (13:01, 18 Sep).
    uvicorn.run(app, host='0.0.0.0', port=PORT, log_level='warning', timeout_graceful_shutdown=3)
