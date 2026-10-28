"""Face detectors. HaarDetector runs fully offline using OpenCV's bundled cascades."""
import cv2
from .tracker import Detection


def _iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


class HaarDetector:
    """Frontal cascade => facing forward. Profile-only hit => head turned away (head-pose proxy)."""

    def __init__(self):
        base = cv2.data.haarcascades
        self.front = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
        self.profile = cv2.CascadeClassifier(base + "haarcascade_profileface.xml")
        if self.front.empty() or self.profile.empty():
            raise RuntimeError("OpenCV Haar cascades not found")

    def detect(self, frame):
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        w = gray.shape[1]
        kw = dict(scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        fronts = [tuple(map(float, b)) for b in self.front.detectMultiScale(gray, **kw)]
        profs = [tuple(map(float, b)) for b in self.profile.detectMultiScale(gray, **kw)]
        for b in self.profile.detectMultiScale(cv2.flip(gray, 1), **kw):  # right-facing profiles
            profs.append((float(w - b[0] - b[2]), float(b[1]), float(b[2]), float(b[3])))
        out = [Detection(b, True) for b in fronts]
        for p in profs:
            if all(_iou(p, d.box) < 0.3 for d in out):
                out.append(Detection(p, False))
        return out
