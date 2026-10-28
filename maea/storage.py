"""Local SQLite storage - everything stays on the machine (privacy by design)."""
import sqlite3, json, csv, io

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, source TEXT,
  started_at TEXT, summary TEXT, config TEXT);
CREATE TABLE IF NOT EXISTS attendance(session_id INT, participant TEXT, first_seen TEXT, last_seen TEXT,
  marked_at TEXT, present_sec REAL, status TEXT, engagement REAL, engagement_label TEXT, attention REAL);
CREATE TABLE IF NOT EXISTS windows(session_id INT, idx INT, data TEXT);
CREATE TABLE IF NOT EXISTS suggestions(session_id INT, priority TEXT, category TEXT, title TEXT,
  rationale TEXT, time_range TEXT);
"""
ATT_COLS = ["participant", "first_seen", "last_seen", "marked_at", "present_sec", "status",
            "engagement", "engagement_label", "attention"]


class Store:
    def __init__(self, path):
        self.path = path
        with self._c() as c:
            c.executescript(SCHEMA)

    def _c(self):
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        return c

    def save(self, r):
        c = self._c()
        try:
            cur = c.execute("INSERT INTO sessions(name,source,started_at,summary,config) VALUES(?,?,?,?,?)",
                            (r["name"], r["source"], r["started_at"], json.dumps(r["summary"]), json.dumps(r["config"])))
            sid = cur.lastrowid
            for a in r["attendance"]:
                c.execute(f"INSERT INTO attendance VALUES(?,{','.join('?' * len(ATT_COLS))})",
                          (sid, *[a[k] for k in ATT_COLS]))
            for w in r["windows"]:
                c.execute("INSERT INTO windows VALUES(?,?,?)", (sid, w["index"], json.dumps(w)))
            for s in r["suggestions"]:
                c.execute("INSERT INTO suggestions VALUES(?,?,?,?,?,?)",
                          (sid, s["priority"], s["category"], s["title"], s["rationale"], s["time_range"]))
            c.commit()
            return sid
        finally:
            c.close()

    def list_sessions(self):
        with self._c() as c:
            return [{"id": r["id"], "name": r["name"], "source": r["source"], "started_at": r["started_at"],
                     "summary": json.loads(r["summary"])}
                    for r in c.execute("SELECT * FROM sessions ORDER BY id DESC")]

    def get(self, sid):
        with self._c() as c:
            s = c.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
            if not s:
                return None
            return {
                "id": sid, "name": s["name"], "source": s["source"], "started_at": s["started_at"],
                "summary": json.loads(s["summary"]), "config": json.loads(s["config"]),
                "attendance": [dict(r) for r in c.execute(
                    "SELECT " + ",".join(ATT_COLS) + " FROM attendance WHERE session_id=?", (sid,))],
                "windows": [json.loads(r["data"]) for r in c.execute(
                    "SELECT data FROM windows WHERE session_id=? ORDER BY idx", (sid,))],
                "suggestions": [dict(r) for r in c.execute(
                    "SELECT priority,category,title,rationale,time_range FROM suggestions WHERE session_id=?", (sid,))],
            }

    def delete(self, sid):
        with self._c() as c:
            for t in ("attendance", "windows", "suggestions"):
                c.execute(f"DELETE FROM {t} WHERE session_id=?", (sid,))
            c.execute("DELETE FROM sessions WHERE id=?", (sid,))

    def attendance_csv(self, sid):
        s = self.get(sid)
        if not s:
            return None
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=ATT_COLS)
        w.writeheader(); w.writerows(s["attendance"])
        return buf.getvalue()
