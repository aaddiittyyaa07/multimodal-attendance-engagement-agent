"""Multimodal fusion: visual (head pose, movement, inactivity) + audio => engagement scores."""
from collections import defaultdict
import numpy as np


def label(score, cfg):
    return "High" if score >= cfg.high_threshold else "Medium" if score >= cfg.medium_threshold else "Low"


def _visual_components(obs, cfg):
    attention = float(np.mean([o.frontal for o in obs]))
    inactive = float(np.mean([o.inactive for o in obs]))
    restless = float(np.mean([o.move > cfg.restless_thresh for o in obs]))
    motion = max(0.0, 1.0 - min(1.0, inactive + 0.5 * restless))
    return attention, motion


def fuse(attention, motion, audio, cfg):
    """Weighted fusion; if audio is unavailable the remaining weights are renormalised."""
    parts = [(cfg.w_attention, attention), (cfg.w_motion, motion)]
    if audio is not None:
        parts.append((cfg.w_audio, audio))
    tot = sum(w for w, _ in parts)
    return sum(w * v for w, v in parts) / tot


def compute_windows(observations, audio, duration, cfg, allowed_ids=None):
    """Return per-window engagement trend. `allowed_ids` restricts to participants marked present."""
    n = max(1, int(np.ceil(duration / cfg.window_sec)))
    by_win = defaultdict(list)
    for o in observations:
        if allowed_ids is not None and o.track_id not in allowed_ids:
            continue
        by_win[min(n - 1, int(o.t // cfg.window_sec))].append(o)
    windows = []
    for i in range(n):
        t0, t1 = i * cfg.window_sec, min(duration, (i + 1) * cfg.window_sec)
        obs = by_win.get(i, [])
        a = audio.window(t0, t1) if audio else None
        audio_score = None
        if a:
            audio_score = min(1.0, a["speech_ratio"] / cfg.target_speech_ratio)
        if obs:
            att, mot = _visual_components(obs, cfg)
            score = fuse(att, mot, audio_score, cfg)
        else:
            att = mot = None
            score = 0.0
        windows.append({
            "index": i, "start": t0, "end": t1,
            "participants": len({o.track_id for o in obs}),
            "attention": att, "motion": mot, "audio": audio_score,
            "speech_ratio": a["speech_ratio"] if a else None,
            "turns": a["turns"] if a else None,
            "engagement": float(score), "label": label(score, cfg),
        })
    return windows


def compute_student_stats(observations, tracks, audio_avg, cfg):
    by_id = defaultdict(list)
    for o in observations:
        by_id[o.track_id].append(o)
    stats = {}
    for tr in tracks:
        obs = by_id.get(tr.id, [])
        if not obs:
            continue
        att, mot = _visual_components(obs, cfg)
        score = fuse(att, mot, audio_avg, cfg)
        stats[tr.id] = {"attention": att, "motion": mot, "engagement": float(score),
                        "label": label(score, cfg)}
    return stats
