import os, numpy as np, pytest
from maea.config import Config
from maea.tracker import Tracker, Detection
from maea.audio import AudioAnalyzer, save_wav, load_wav
from maea.engagement import fuse, label
from maea.simulate import run_demo, simulated_audio
from maea.storage import Store
from maea.app import create_app


def feed(tr, box, t0, t1, fps=5, frontal=True):
    t = t0
    while t <= t1:
        tr.update([Detection(box, frontal)], t); t += 1 / fps


def test_sustained_presence_marks_attendance():
    tr = Tracker(Config()); feed(tr, (50, 50, 60, 60), 0, 15)
    assert len(tr.attended()) == 1


def test_brief_appearance_not_marked():
    tr = Tracker(Config()); feed(tr, (50, 50, 60, 60), 0, 4)
    assert tr.attended() == [] and len(tr.transient()) == 1


def test_short_occlusion_keeps_same_track():
    tr = Tracker(Config()); feed(tr, (50, 50, 60, 60), 0, 6); feed(tr, (52, 50, 60, 60), 8, 14)
    assert len(tr.tracks) == 1 and len(tr.attended()) == 1


def test_two_people_two_tracks():
    tr = Tracker(Config())
    for i in range(80):
        tr.update([Detection((50, 50, 60, 60)), Detection((400, 50, 60, 60))], i / 5)
    assert len(tr.attended()) == 2


def test_inactivity_flag():
    cfg = Config(); tr = Tracker(cfg); last = None
    for i in range(150):
        last = tr.update([Detection((50, 50, 60, 60))], i / 5)[0]
    assert last.inactive


def test_vad_detects_speech_vs_silence(tmp_path):
    sr = 16000; rng = np.random.default_rng(0)
    x = rng.normal(0, 0.002, sr * 10).astype(np.float32); x[sr * 2: sr * 6] += rng.normal(0, 0.1, sr * 4)
    p = tmp_path / "a.wav"; save_wav(str(p), x, sr); y, sr2 = load_wav(str(p))
    a = AudioAnalyzer(y, sr2)
    assert a.window(2.5, 5.5)["speech_ratio"] > 0.9 and a.window(7, 10)["speech_ratio"] < 0.1


def test_fusion_weights_and_labels():
    cfg = Config()
    assert fuse(1, 1, 1, cfg) == pytest.approx(1.0) and fuse(0, 0, 0, cfg) == 0
    assert fuse(1, 1, None, cfg) == pytest.approx(1.0)  # renormalised without audio
    assert label(0.9, cfg) == "High" and label(0.5, cfg) == "Medium" and label(0.1, cfg) == "Low"


def test_demo_end_to_end():
    r = run_demo()
    assert r["summary"]["participants_present"] == 6 and r["summary"]["transient_ignored"] == 1
    assert any(a["status"] == "Late" for a in r["attendance"])
    eng = [w["engagement"] for w in r["windows"]]
    assert min(eng) < 0.45 < 0.7 < max(eng)                      # engagement dip is detected
    assert any(s["priority"] == "high" for s in r["suggestions"])  # and the agent reacts to it


def test_storage_and_api(tmp_path):
    st = Store(str(tmp_path / "t.db")); sid = st.save(run_demo())
    assert st.get(sid)["attendance"] and "participant" in st.attendance_csv(sid)
    app = create_app(str(tmp_path), sync=True); c = app.test_client()
    jid = c.post("/api/run", data={"demo": "1"}).get_json()["job_id"]
    j = c.get(f"/api/jobs/{jid}").get_json(); assert j["status"] == "done"
    assert c.get(f"/api/sessions/{j['session_id']}").status_code == 200
    assert c.get(f"/api/sessions/{j['session_id']}/attendance.csv").status_code == 200
    assert c.get("/").status_code == 200
    assert c.post("/api/run", data={}).status_code == 400


def test_haar_detector_loads_and_runs_on_blank_frame():
    from maea.detectors import HaarDetector
    assert HaarDetector().detect(np.zeros((240, 320, 3), np.uint8)) == []
