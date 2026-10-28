"""Entry point:  python run.py            -> dashboard on http://127.0.0.1:5000
                python run.py --cli video.mp4 [--audio audio.wav]   -> analyse from terminal
                python run.py --demo         -> simulated classroom, prints a report"""
import argparse, json
from maea.app import create_app
from maea.pipeline import analyze_video
from maea.simulate import run_demo
from maea.storage import Store
from maea.config import Config


def print_report(r):
    s = r["summary"]
    print(f"\n== {r['name']} ==  duration {s['duration_sec']}s | present {s['participants_present']} | "
          f"avg engagement {s['avg_engagement']:.0%} ({s['overall_label']})")
    print("\nAttendance:")
    for a in r["attendance"]:
        print(f"  {a['participant']:<14} {a['status']:<8} marked {a['marked_at']}  present {a['present_sec']}s  "
              f"engagement {a['engagement_label']}")
    print("\nEngagement trend:")
    for w in r["windows"]:
        print(f"  {int(w['start']//60):02d}:{int(w['start']%60):02d}  {'#' * int(w['engagement'] * 30):<30} {w['engagement']:.0%}")
    print("\nTeaching suggestions:")
    for g in r["suggestions"]:
        print(f"  [{g['priority'].upper()}] {g['title']} {g['time_range'] or ''}\n        {g['rationale']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cli", metavar="VIDEO", help="analyse a video file (or camera index like 0)")
    p.add_argument("--audio", help="optional WAV file for audio analysis")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--json", help="write the result to this JSON file")
    p.add_argument("--port", type=int, default=5000)
    a = p.parse_args()
    if a.demo or a.cli:
        src = a.cli
        r = run_demo() if a.demo else analyze_video(int(src) if src.isdigit() else src, a.audio)
        Store("data/maea.db").save(r)
        print_report(r)
        if a.json:
            json.dump(r, open(a.json, "w"), indent=2)
    else:
        create_app("data").run(host="127.0.0.1", port=a.port)
