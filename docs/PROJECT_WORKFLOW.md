# Project workflow: mapped to the Skill Wallet epics

| Epic | Deliverable | Where in the repo |
|---|---|---|
| **1. Requirements specification and environment setup** | Objectives (automated attendance, engagement analysis, AI suggestions); architecture; Python 3.8+, VS Code, Git; `requirements.txt` | `README.md`, `requirements.txt`, this doc |
| **2. Vision module** | Video input (file / camera); face detection and tracking; sustained-presence logic; timestamped attendance logs | `maea/detectors.py`, `maea/tracker.py`, `maea/pipeline.py` (`VideoSource`) |
| **3. Audio module** | Audio input (WAV / ffmpeg extraction); speech detection; interaction levels | `maea/audio.py` |
| **4. AI implementation** | Head pose, movement, inactivity; multimodal fusion; teaching-suggestion agent | `maea/engagement.py`, `maea/agent.py` |
| **5. Analytics and recommendation engine** | Engagement scores, session trends, per-participant stats, recommendations | `maea/engagement.py`, `maea/agent.py`, `maea/storage.py` |
| **6. Interface, testing and deployment** | Dashboard + reports (CSV); privacy via offline/local storage; accuracy tests; GitHub CI | `maea/app.py`, `maea/templates/`, `tests/`, `.github/workflows/ci.yml` |

## Architecture (adapted for offline operation)

The reference diagram uses cloud GenAI services. MAEA replaces the *AI Service Layer* with local components so that nothing leaves the machine:

* **Frontend layer**: HTML5/CSS3/JS dashboard, Flask API gateway, route handlers
* **Local AI service layer**: OpenCV face detection, energy-based VAD, fusion engine, rule-based teaching agent
* **Utility layer**: pipeline orchestration, config, tracker
* **Storage layer**: local SQLite database, uploaded files (`data/uploads`), CSV exports

## Test plan
1. Unit: tracker, VAD, fusion, labels (`tests/test_core.py`)
2. Integration: simulated session with known ground truth (6 participants, 1 transient walker, 1 late arrival, a planned engagement dip)
3. API: run, status, session fetch, CSV export
4. Real-world: record a short class clip, compare attendance counts against a manual roll call
