"""Teaching-suggestion agent.

A transparent, local, rule-based reasoning agent following an Observe -> Reason -> Recommend loop.
It never enforces decisions: every suggestion carries a rationale so the instructor stays in control.
"""
import numpy as np

PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def _fmt(t):
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def _runs(flags):
    """Yield (start, end_exclusive) for consecutive True runs."""
    start = None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            yield start, i
            start = None


def generate_suggestions(summary, windows, attendance, students, cfg):
    sug = []

    def add(priority, category, title, rationale, span=None):
        sug.append({"priority": priority, "category": category, "title": title,
                    "rationale": rationale, "time_range": span})

    active = [w for w in windows if w["participants"] > 0]
    if not active:
        return [{"priority": "low", "category": "data", "title": "Not enough data",
                 "rationale": "No participants were marked present, so no analysis was possible.",
                 "time_range": None}]
    eng = np.array([w["engagement"] for w in windows])

    # 1. sustained low engagement
    for s, e in _runs(eng < cfg.medium_threshold):
        span = f"{_fmt(windows[s]['start'])}-{_fmt(windows[e - 1]['end'])}"
        if e - s >= 2:
            add("high", "engagement", "Introduce an interactive activity or short break",
                f"Engagement stayed low for {e - s} consecutive intervals (avg {eng[s:e].mean():.0%}). "
                "A poll, think-pair-share, or 2-minute stretch break can reset attention.", span)
        else:
            add("medium", "engagement", "Check in with the class",
                f"Engagement dipped to {eng[s]:.0%} in this interval. A quick question or recap may help.", span)

    # 2. sharp drops
    for i in range(1, len(eng)):
        if windows[i]["participants"] and eng[i - 1] - eng[i] >= 0.20 and eng[i] >= cfg.medium_threshold:
            add("medium", "pacing", "Attention dropped sharply - revisit pacing",
                f"Engagement fell from {eng[i - 1]:.0%} to {eng[i]:.0%}. The preceding content may have been "
                "too dense or too fast; consider summarising before moving on.",
                f"{_fmt(windows[i]['start'])}-{_fmt(windows[i]['end'])}")

    # 3. audio-driven insights
    sr = [w["speech_ratio"] for w in windows]
    if all(x is not None for x in sr):
        for s, e in _runs([x < 0.10 for x in sr]):
            if e - s >= 2:
                add("medium", "interaction", "Long silence - prompt discussion",
                    "Very little speech was detected for several intervals. Ask an open question or "
                    "start a discussion to re-activate participation.",
                    f"{_fmt(windows[s]['start'])}-{_fmt(windows[e - 1]['end'])}")
        for s, e in _runs([x > 0.85 and eng[i] < cfg.high_threshold for i, x in enumerate(sr)]):
            if e - s >= 3:
                add("medium", "interaction", "Continuous lecturing - add Q&A",
                    "Near-continuous speech with only moderate engagement suggests one-way delivery. "
                    "Insert a Q&A pause or a worked example.",
                    f"{_fmt(windows[s]['start'])}-{_fmt(windows[e - 1]['end'])}")

    # 4. restlessness / classroom management
    mot = [w["motion"] for w in windows if w["motion"] is not None]
    if mot and min(mot) < 0.5:
        i = int(np.nanargmin([w["motion"] if w["motion"] is not None else 9 for w in windows]))
        add("low", "classroom", "Unusual movement/inactivity pattern",
            "Movement indicators were abnormal (very still or very restless) in this interval; "
            "this can indicate fatigue or distraction.",
            f"{_fmt(windows[i]['start'])}-{_fmt(windows[i]['end'])}")

    # 5. attendance
    late = [a for a in attendance if a["status"] == "Late"]
    if late:
        add("low", "attendance", f"{len(late)} late arrival(s)",
            "Consider a brief recap at the start or posting session objectives so late joiners can catch up.")
    n_trans = summary.get("transient_ignored", 0)
    if n_trans:
        add("low", "attendance", f"{n_trans} brief appearance(s) ignored",
            "These were shorter than the sustained-presence threshold and were not marked present.")

    # 6. disengaged individuals (anonymised)
    low = [s for s in students.values() if s["label"] == "Low"]
    if low:
        add("medium", "students", f"{len(low)} participant(s) showed low overall engagement",
            "Consider a gentle, private check-in. Participants are anonymous (Participant IDs) to protect privacy.")

    # 7. trend & what worked
    half = len(eng) // 2
    if half >= 1 and eng[half:].mean() + 0.10 < eng[:half].mean():
        add("medium", "pacing", "Engagement declined in the second half",
            f"First half avg {eng[:half].mean():.0%} vs second half {eng[half:].mean():.0%}. "
            "Plan shorter segments or a mid-session activity next time.")
    best = int(np.argmax(eng))
    add("low", "positive", "Replicate what worked",
        f"Peak engagement ({eng[best]:.0%}) occurred in this interval - note the activity or topic used.",
        f"{_fmt(windows[best]['start'])}-{_fmt(windows[best]['end'])}")

    if not any(s["priority"] in ("high", "medium") for s in sug):
        add("low", "overall", "Session went well", "No significant engagement issues were detected.")
    sug.sort(key=lambda s: PRIORITY_ORDER[s["priority"]])
    return sug
