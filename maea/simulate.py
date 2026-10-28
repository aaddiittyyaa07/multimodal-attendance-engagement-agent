"""Deterministic simulated classroom (detections + audio) for demos, tests and offline evaluation."""
import numpy as np
from .tracker import Detection
from .audio import AudioAnalyzer
from .config import Config
from .pipeline import run_session


def simulated_stream(cfg, duration=300.0, seed=7):
    rng = np.random.default_rng(seed)
    # (x, y, arrive, leave, attention, frozen_in_dip)
    students = [
        (60, 200, 0, duration, 0.92, False), (200, 200, 2, duration, 0.88, True),
        (340, 200, 4, duration, 0.85, True), (480, 200, 5, duration, 0.35, False),   # disengaged
        (130, 320, 75, duration, 0.80, False),                                       # late arrival
        (420, 320, 3, 240, 0.82, True),                                              # leaves early
    ]
    walker = (300, 100, 100, 106)  # walks past for 6s -> transient, must NOT be marked
    dip = (120, 210)
    pos = {i: np.array([s[0], s[1]], float) for i, s in enumerate(students)}
    dt, t = 1.0 / cfg.sample_fps, 0.0
    while t <= duration:
        dets = []
        in_dip = dip[0] <= t < dip[1]
        for i, (x, y, a, l, att, frozen) in enumerate(students):
            if not (a <= t < l):
                continue
            p_front = att * (0.35 if in_dip else 1.0)
            sigma = 0.0 if (in_dip and frozen) else 1.4
            pos[i] = np.array([x, y]) + (0 if sigma == 0 else rng.normal(0, sigma, 2).cumsum()[-1] * np.array([1, 1]))
            dets.append(Detection((float(pos[i][0]), float(pos[i][1]), 60.0, 60.0), bool(rng.random() < p_front)))
        if walker[2] <= t < walker[3]:
            dets.append(Detection((100 + (t - walker[2]) * 60, 40.0, 60.0, 60.0), True))
        yield t, dets
        t += dt


def simulated_audio(duration=300.0, sr=16000, seed=7):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.002, int(duration * sr)).astype(np.float32)
    t = 0.0
    while t < duration:
        seg = rng.uniform(4, 8)
        quiet = 150 <= t < 210   # long silence during the engagement dip
        if not quiet:
            end = min(duration, t + seg)
            n = int((end - t) * sr)
            env = 0.08 * (0.6 + 0.4 * np.sin(np.linspace(0, 40 * np.pi, n)))
            x[int(t * sr): int(t * sr) + n] += (rng.normal(0, 1, n) * env).astype(np.float32)
        t += seg + rng.uniform(1, 3)
    return x, sr


def run_demo(duration=300.0, seed=7, cfg=None):
    cfg = cfg or Config()
    samples, sr = simulated_audio(duration, seed=seed)
    audio = AudioAnalyzer(samples, sr, cfg.frame_ms)
    return run_session(simulated_stream(cfg, duration, seed), duration, audio, cfg,
                       name="Demo: simulated classroom", source="simulation")
