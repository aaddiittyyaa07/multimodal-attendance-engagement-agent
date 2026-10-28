"""Face tracking + sustained-presence attendance logic (no face recognition => privacy preserving)."""
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import math

Box = Tuple[float, float, float, float]  # x, y, w, h


@dataclass
class Detection:
    box: Box
    frontal: bool = True  # True = facing the front (attentive head pose proxy)


@dataclass
class Observation:
    t: float
    track_id: int
    frontal: bool
    move: float       # centroid displacement / face width since last sample
    inactive: bool    # still for >= inactivity_sec


@dataclass
class Track:
    id: int
    box: Box
    first_seen: float
    last_seen: float
    present_time: float = 0.0
    still_since: float = 0.0
    marked_at: float = None
    frontal_frames: int = 0
    frames: int = 0

    @property
    def center(self):
        x, y, w, h = self.box
        return (x + w / 2, y + h / 2)


class Tracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tracks: Dict[int, Track] = {}
        self.active: List[int] = []
        self._next = 1
        self._last_t = None

    def update(self, detections: List[Detection], t: float) -> List[Observation]:
        cfg = self.cfg
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t
        # drop stale active tracks
        self.active = [i for i in self.active if t - self.tracks[i].last_seen <= cfg.max_gap_sec]

        # greedy nearest-centroid matching
        pairs = []
        for di, d in enumerate(detections):
            cx, cy = d.box[0] + d.box[2] / 2, d.box[1] + d.box[3] / 2
            for tid in self.active:
                tx, ty = self.tracks[tid].center
                dist = math.hypot(cx - tx, cy - ty) / max(d.box[2], 1e-6)
                if dist <= cfg.match_dist:
                    pairs.append((dist, di, tid))
        pairs.sort()
        used_d, used_t, obs = set(), set(), []
        for dist, di, tid in pairs:
            if di in used_d or tid in used_t:
                continue
            used_d.add(di); used_t.add(tid)
            obs.append(self._apply(self.tracks[tid], detections[di], t, dt, dist))
        for di, d in enumerate(detections):
            if di in used_d:
                continue
            tr = Track(self._next, d.box, t, t, still_since=t)
            self._next += 1
            self.tracks[tr.id] = tr
            self.active.append(tr.id)
            obs.append(self._apply(tr, d, t, 0.0, 0.0, new=True))
        return obs

    def _apply(self, tr: Track, d: Detection, t, dt, move, new=False) -> Observation:
        cfg = self.cfg
        gap = t - tr.last_seen
        # gaps (brief occlusion) still count toward presence while the track is alive
        tr.present_time += gap if not new else 0.0
        tr.last_seen = t
        tr.box = d.box
        tr.frames += 1
        tr.frontal_frames += int(d.frontal)
        if move > cfg.move_thresh:
            tr.still_since = t
        inactive = (t - tr.still_since) >= cfg.inactivity_sec
        if tr.marked_at is None and tr.present_time >= cfg.min_presence_sec:
            tr.marked_at = t
        return Observation(t, tr.id, d.frontal, move, inactive)

    def attended(self) -> List[Track]:
        return [t for t in self.tracks.values() if t.marked_at is not None]

    def transient(self) -> List[Track]:
        return [t for t in self.tracks.values() if t.marked_at is None]
