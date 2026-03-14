import sys
import os
from datetime import date, datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ml'))

from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
from ml.predictor import predict_stress_from_tracker

app = Flask(__name__)
CORS(app)

LOG_FILE = os.path.join(os.path.dirname(__file__), "activity_log.csv")

# How many days of history to use for graphs/analytics
HISTORY_DAYS = 7  # FIX: increased from 5 to 7 for richer chart history

# Consistent distracting app list used everywhere
DISTRACTING = ["YouTube", "Instagram", "Netflix", "Twitter", "TikTok", "Facebook", "WhatsApp"]


def load_log():
    """Load the activity log and return a DataFrame, or None if empty/missing."""
    if not os.path.exists(LOG_FILE):
        return None
    df = pd.read_csv(LOG_FILE, encoding='utf-8', encoding_errors='ignore')
    if df.empty:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], format='mixed')
    return df


def extract_base_app(title: str) -> str:
    """
    FIX: Extract the base application name from a window title.
    Window titles like 'file.py - Project - Visual Studio Code' should count
    as 'Visual Studio Code', not as a unique app per file/tab.
    Switching between files in VS Code or tabs in Chrome was inflating
    app_switches by ~2x, pushing features outside the model's trained range.
    """
    if not isinstance(title, str):
        return "Unknown"
    parts = [p.strip() for p in title.split(" - ")]
    return parts[-1] if parts else title


def compute_features_for_group(group: pd.DataFrame):
    """Compute all features for a given slice of the log (one day or today)."""
    group = group.sort_values("timestamp")
    total_entries = len(group)
    screen_time = total_entries * 5 / 60  # minutes

    night = group[group["timestamp"].dt.hour >= 22]
    night_usage = len(night) * 5 / 60  # minutes

    group = group.copy()
    group["gap"] = group["timestamp"].diff().dt.total_seconds().fillna(0)

    # Longest continuous usage streak (minutes)
    current_streak = 0
    max_streak = 0
    for gap in group["gap"]:
        if gap <= 300:
            current_streak += 5
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 5
    continuous_usage = max_streak / 60

    # FIX: Use base app name (last segment of window title) instead of full title.
    # Full title changes per browser tab / open file, massively inflating switch count.
    # FIX: fillna with the app itself so the first row doesn't count as a switch
    # (shift(1) produces NaN for row 0, and NaN != any string evaluates to True in pandas).
    group["base_app"] = group["app"].apply(extract_base_app)
    group["prev_base_app"] = group["base_app"].shift(1).fillna(group["base_app"])
    app_switches = int((group["base_app"] != group["prev_base_app"]).sum())

    breaks = int((group["gap"] > 300).sum())

    distracting_count = group[group["app"].str.contains(
        '|'.join(DISTRACTING), case=False, na=False
    )].shape[0]
    productive_ratio = round(1 - (distracting_count / max(total_entries, 1)), 2)

    return {
        "screen_time": round(screen_time, 2),
        "continuous_usage": round(continuous_usage, 2),
        "night_usage": round(night_usage, 2),
        "app_switches": app_switches,
        "breaks": breaks,
        "productive_ratio": productive_ratio,
    }


def compute_features():
    """Compute features for TODAY only — used by the dashboard."""
    df = load_log()
    if df is None:
        return None

    today = date.today()
    today_df = df[df["timestamp"].dt.date == today]

    if today_df.empty:
        return None

    return compute_features_for_group(today_df)


@app.route("/api/stress", methods=["GET"])
def get_stress():
    features = compute_features()

    if features is None:
        return jsonify({"error": "No activity data yet for today. Run tracker.py first."}), 400

    stress_level = predict_stress_from_tracker(
        features["screen_time"],
        features["continuous_usage"],
        features["night_usage"],
        features["app_switches"],
        features["breaks"],
        features["productive_ratio"]
    )

    therapy_map = {
        "High": {
            "message": "High digital fatigue detected. Time to relax.",
            "therapy": "Meditation / Breathing",
            "spotify": "https://open.spotify.com/embed/playlist/37i9dQZF1DX4sWSpwq3LiO"
        },
        "Medium": {
            "message": "Moderate fatigue. Consider a short break.",
            "therapy": "Calm music / Nature sounds",
            "spotify": "https://open.spotify.com/embed/playlist/37i9dQZF1DX3Ogo9pFvBkY"
        },
        "Low": {
            "message": "You are doing well. Stay focused.",
            "therapy": "Focus music",
            "spotify": "https://open.spotify.com/embed/playlist/37i9dQZF1DX8NTLI2TtZa6"
        }
    }

    recommendation = therapy_map.get(stress_level, therapy_map["Medium"])

    return jsonify({
        "stress_level": stress_level,
        "features": features,
        "message": recommendation["message"],
        "therapy": recommendation["therapy"],
        "spotify": recommendation["spotify"]
    })


@app.route("/api/features", methods=["GET"])
def features():
    data = compute_features()
    if data is None:
        return jsonify({"error": "No activity data yet for today."}), 400
    return jsonify(data)


@app.route("/api/history", methods=["GET"])
def history():
    """Return per-day summary for the last HISTORY_DAYS days — used by all charts."""
    df = load_log()
    if df is None:
        return jsonify([])

    cutoff = datetime.now() - timedelta(days=HISTORY_DAYS)
    df = df[df["timestamp"] >= cutoff]

    if df.empty:
        return jsonify([])

    df["date"] = df["timestamp"].dt.date
    result = []

    for day, group in df.groupby("date"):
        f = compute_features_for_group(group)
        stress_level = predict_stress_from_tracker(
            f["screen_time"], f["continuous_usage"], f["night_usage"],
            f["app_switches"], f["breaks"], f["productive_ratio"]
        )
        stress_value = 80 if stress_level == "High" else 50 if stress_level == "Medium" else 25

        result.append({
            "date": str(day),
            "name": day.strftime("%b %d"),
            "screen_time": f["screen_time"],
            "productive": round(f["screen_time"] * f["productive_ratio"], 2),
            "entertainment": round(f["screen_time"] * (1 - f["productive_ratio"]), 2),
            "stress": stress_value,
            "breaks": f["breaks"],
            "app_switches": f["app_switches"],
        })

    result.sort(key=lambda x: x["date"])
    return jsonify(result)


@app.route("/api/heatmap", methods=["GET"])
def heatmap():
    """Return hourly intensity heatmap using last HISTORY_DAYS days."""
    df = load_log()
    if df is None:
        return jsonify([])

    cutoff = datetime.now() - timedelta(days=HISTORY_DAYS)
    df = df[df["timestamp"] >= cutoff]

    if df.empty:
        return jsonify([])

    df["day"] = df["timestamp"].dt.dayofweek
    df["hour"] = df["timestamp"].dt.hour

    counts = df.groupby(["day", "hour"]).size().reset_index(name="count")
    max_count = counts["count"].max() if not counts.empty else 1

    result = []
    for _, row in counts.iterrows():
        result.append({
            "day": int(row["day"]),
            "hour": int(row["hour"]),
            "intensity": round(row["count"] / max_count, 2)
        })

    return jsonify(result)


@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({"status": "Backend running"})


# FIX: Removed duplicate /stress route that shadowed /api/stress and had
# no therapy/message/spotify fields — frontend only uses /api/stress.

if __name__ == "__main__":
    app.run(debug=True, port=5000)
