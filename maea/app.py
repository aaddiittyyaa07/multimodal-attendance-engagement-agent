"""Flask dashboard + JSON API. Runs on localhost only; no external services."""
import os, threading, uuid
from flask import Flask, jsonify, render_template, request, Response, abort
from .config import Config
from .storage import Store
from .pipeline import analyze_video
from .simulate import run_demo
from .audio import AudioAnalyzer, load_wav


def create_app(data_dir="data", sync=False):
    app = Flask(__name__, template_folder="templates")
    app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024
    os.makedirs(os.path.join(data_dir, "uploads"), exist_ok=True)
    store = Store(os.path.join(data_dir, "maea.db"))
    jobs = {}

    def job(jid, fn):
        try:
            jobs[jid] = {"status": "running"}
            sid = store.save(fn())
            jobs[jid] = {"status": "done", "session_id": sid}
        except Exception as e:  # surface errors to the UI
            jobs[jid] = {"status": "error", "error": str(e)}

    @app.get("/")
    def index():
        return render_template("dashboard.html")

    @app.post("/api/run")
    def run():
        jid = uuid.uuid4().hex[:8]
        name = request.form.get("name") or "Session"
        if request.form.get("demo"):
            fn = lambda: run_demo()
        else:
            v = request.files.get("video")
            if not v:
                return jsonify(error="Upload a video or choose demo"), 400
            up = os.path.join(data_dir, "uploads")
            vp = os.path.join(up, f"{jid}_{os.path.basename(v.filename)}")
            v.save(vp)
            ap = None
            a = request.files.get("audio")
            if a and a.filename:
                ap = os.path.join(up, f"{jid}_{os.path.basename(a.filename)}")
                a.save(ap)
            fn = lambda: analyze_video(vp, ap, Config(), name)
        if sync:
            job(jid, fn)
        else:
            threading.Thread(target=job, args=(jid, fn), daemon=True).start()
            jobs[jid] = {"status": "running"}
        return jsonify(job_id=jid)

    @app.get("/api/jobs/<jid>")
    def job_status(jid):
        return jsonify(jobs.get(jid) or {"status": "unknown"})

    @app.get("/api/sessions")
    def sessions():
        return jsonify(store.list_sessions())

    @app.get("/api/sessions/<int:sid>")
    def session(sid):
        s = store.get(sid)
        return jsonify(s) if s else abort(404)

    @app.delete("/api/sessions/<int:sid>")
    def delete(sid):
        store.delete(sid)
        return jsonify(ok=True)

    @app.get("/api/sessions/<int:sid>/attendance.csv")
    def csv_export(sid):
        c = store.attendance_csv(sid)
        if c is None:
            abort(404)
        return Response(c, mimetype="text/csv",
                        headers={"Content-Disposition": f"attachment; filename=attendance_{sid}.csv"})

    return app
