"""End-to-end pipeline: video/audio sources -> tracking -> attendance -> engagement -> suggestions."""
from datetime import datetime, timedelta
import cv2

from .config import Config
from .tracker import Tracker
from .detectors import HaarDetector
from .audio import AudioAnalyzer, load_wav, extract_audio_from_video
from .engagement import compute_windows, compute_student_stats, label
from .agent import generate_suggestions


class VideoSource:
    """Iterates (t, detections) from a video file or a live camera index at `sample_fps`."""

    def __init__(self, source, cfg: Config, detector=None, max_seconds=None):
        self.cfg, self.source = cfg, source
        self.detector = detector or HaarDetector()
        self.max_seconds = max_seconds
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise IOError(f"Cannot open video source: {source}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        self.fps = fps if fps > 1 else 25.0
        self.duration = (n / self.fps) if (isinstance(source, str) and n > 0) else (max_seconds or 0)

    def __iter__(self):
        cap = cv2.VideoCapture(self.source)
        step = max(1, round(self.fps / self.cfg.sample_fps))
        i = 0
        last_t = 0.0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                t = i / self.fps
                if self.max_seconds and t > self.max_seconds:
                    break
                if i % step == 0:
                    h, w = frame.shape[:2]
                    if w > self.cfg.frame_width:
                        frame = cv2.resize(frame, (self.cfg.frame_width, int(h * self.cfg.frame_width / w)))
                    last_t = t
                    yield t, self.detector.detect(frame)
                i += 1
        finally:
            cap.release()
            if not self.duration:
                self.duration = last_t


def run_session(stream, duration, audio=None, cfg=None, name="Session", source="video", started_at=None):
    """Run the whole analysis. `stream` yields (t, [Detection])."""
    cfg = cfg or Config()
    started_at = started_at or datetime.now()
    tracker, observations = Tracker(cfg), []
    for t, dets in stream:
        observations.extend(tracker.update(dets, t))
        duration = max(duration, t)

    attended = tracker.attended()
    ids = {t.id for t in attended}
    # stable, human-friendly anonymous numbering in order of arrival
    order = sorted(attended, key=lambda tr: tr.first_seen)
    pname = {tr.id: f"Participant {i + 1}" for i, tr in enumerate(order)}

    windows = compute_windows(observations, audio, duration, cfg, allowed_ids=ids)
    audio_avg = None
    if audio:
        audio_avg = min(1.0, audio.window(0, duration)["speech_ratio"] / cfg.target_speech_ratio)
    students = compute_student_stats(observations, attended, audio_avg, cfg)

    late_after = max(60.0, 0.10 * duration)
    iso = lambda s: (started_at + timedelta(seconds=s)).isoformat(timespec="seconds")
    attendance = []
    for tr in order:
        st = students.get(tr.id, {})
        attendance.append({
            "participant": pname[tr.id], "first_seen": iso(tr.first_seen), "last_seen": iso(tr.last_seen),
            "marked_at": iso(tr.marked_at), "present_sec": round(tr.present_time, 1),
            "status": "Late" if tr.first_seen > late_after else "Present",
            "engagement": round(st.get("engagement", 0.0), 3), "engagement_label": st.get("label", "Low"),
            "attention": round(st.get("attention", 0.0), 3),
        })

    active_w = [w for w in windows if w["participants"]]
    avg = sum(w["engagement"] for w in active_w) / len(active_w) if active_w else 0.0
    summary = {
        "duration_sec": round(duration, 1), "participants_present": len(attended),
        "transient_ignored": len(tracker.transient()), "avg_engagement": round(avg, 3),
        "overall_label": label(avg, cfg), "audio_used": audio is not None,
        "peak_window": max(active_w, key=lambda w: w["engagement"])["index"] if active_w else None,
        "lowest_window": min(active_w, key=lambda w: w["engagement"])["index"] if active_w else None,
    }
    suggestions = generate_suggestions(summary, windows, attendance, students, cfg)
    return {"name": name, "source": source, "started_at": started_at.isoformat(timespec="seconds"),
            "summary": summary, "attendance": attendance, "windows": windows,
            "suggestions": suggestions, "config": cfg.to_dict()}


def analyze_video(video_path, audio_path=None, cfg=None, name=None, detector=None):
    cfg = cfg or Config()
    src = VideoSource(video_path, cfg, detector)
    audio = None
    ap = audio_path or extract_audio_from_video(video_path)
    if ap:
        samples, sr = load_wav(ap)
        audio = AudioAnalyzer(samples, sr, cfg.frame_ms)
    return run_session(src, src.duration, audio, cfg, name or str(video_path), "video")
