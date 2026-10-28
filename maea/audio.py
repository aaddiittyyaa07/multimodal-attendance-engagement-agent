"""Audio signal processing: energy-based voice activity detection + interaction metrics."""
import wave, shutil, subprocess, tempfile, os
import numpy as np
from scipy.ndimage import median_filter


def load_wav(path):
    with wave.open(path, "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[sw]
    x = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if sw == 1:
        x = (x - 128) / 128.0
    else:
        x /= float(np.iinfo(dtype).max)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def save_wav(path, samples, sr):
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def extract_audio_from_video(video_path):
    """Optional: use a locally installed ffmpeg to pull a mono 16k wav. Returns path or None."""
    if not shutil.which("ffmpeg"):
        return None
    out = os.path.join(tempfile.mkdtemp(), "audio.wav")
    r = subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", out],
                       capture_output=True)
    return out if r.returncode == 0 and os.path.exists(out) else None


def frame_rms(samples, sr, frame_ms=30):
    n = int(sr * frame_ms / 1000)
    count = len(samples) // n
    if count == 0:
        return np.zeros(0), n / sr
    f = samples[: count * n].reshape(count, n)
    return np.sqrt((f ** 2).mean(axis=1)), n / sr


def detect_speech(rms):
    """Adaptive threshold above the noise floor, smoothed to remove clicks / short dropouts."""
    if len(rms) == 0:
        return np.zeros(0, dtype=bool)
    floor = np.percentile(rms, 10)
    thr = max(floor * 3.0, 0.01)
    return median_filter((rms > thr).astype(np.uint8), size=9).astype(bool)


class AudioAnalyzer:
    def __init__(self, samples, sr, frame_ms=30):
        rms, self.frame_dur = frame_rms(samples, sr, frame_ms)
        self.rms = rms
        self.speech = detect_speech(rms)
        self.duration = len(samples) / sr

    def window(self, t0, t1):
        i0, i1 = int(t0 / self.frame_dur), int(t1 / self.frame_dur)
        s = self.speech[i0:i1]
        if len(s) == 0:
            return None
        onsets = int(((s[1:] == 1) & (s[:-1] == 0)).sum())
        return {"speech_ratio": float(s.mean()), "turns": onsets,
                "level": float(self.rms[i0:i1].mean())}
