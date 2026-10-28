"""Central, tweakable configuration for the whole pipeline."""
from dataclasses import dataclass, asdict


@dataclass
class Config:
    # --- video ---
    sample_fps: float = 5.0          # frames analysed per second of video
    frame_width: int = 640           # frames are downscaled to this width
    # --- attendance / tracking ---
    min_presence_sec: float = 10.0   # sustained presence needed to be marked present
    max_gap_sec: float = 3.0         # a track survives this long without a detection
    match_dist: float = 1.5          # max centroid distance (in face-widths) to match a track
    # --- engagement ---
    window_sec: float = 30.0         # engagement trend interval
    move_thresh: float = 0.03        # normalised displacement below this == "still"
    restless_thresh: float = 0.35    # normalised displacement above this == restless
    inactivity_sec: float = 20.0     # still this long == inactive
    w_attention: float = 0.50        # head-pose / facing-forward weight
    w_motion: float = 0.20           # liveness (not frozen, not restless) weight
    w_audio: float = 0.30            # speech / interaction weight
    high_threshold: float = 0.70
    medium_threshold: float = 0.45
    # --- audio ---
    frame_ms: int = 30
    target_speech_ratio: float = 0.5  # speech ratio that counts as "fully interactive"

    def to_dict(self):
        return asdict(self)
