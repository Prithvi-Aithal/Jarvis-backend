import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ml'))

from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
from ml.predictor import predict_stress_from_tracker

import getpass

from ml.predictor import predict_stress_from_tracker

app = Flask(__name__)
CORS(app)

LOG_FILE = os.path.join(os.path.dirname(__file__), "activity_log.csv")


def compute_features():
    if not os.path.exists(LOG_FILE):
        return None

    df = pd.read_csv(LOG_FILE, encoding='utf-8', encoding_errors='ignore')

    if df.empty:
        return None

    total_entries = len(df)
    screen_time = total_entries * 5 / 60

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    night = df[df["timestamp"].dt.hour >= 22]
    night_usage = len(night) * 5 / 60

    df = df.sort_values("timestamp")
    df["gap"] = df["timestamp"].diff().dt.total_seconds().fillna(0)
    continuous_usage = screen_time

    df["prev_app"] = df["app"].shift(1)
    app_switches = int((df["app"] != df["prev_app"]).sum())

    breaks = int((df["gap"] > 300).sum())

    distracting = ["YouTube", "Instagram", "Netflix", "Twitter", "TikTok", "Facebook"]
    distracting_count = df[df["app"].str.contains('|'.join(distracting), case=False, na=False)].shape[0]
    productive_ratio = round(1 - (distracting_count / max(total_entries, 1)), 2)

    return {
        "screen_time": round(screen_time, 2),
        "continuous_usage": round(continuous_usage, 2),
        "night_usage": round(night_usage, 2),
        "app_switches": app_switches,
        "breaks": breaks,
        "productive_ratio": productive_ratio
    }

@app.route("/api/stress", methods=["GET"])
def get_stress():
    features = compute_features()

    if features is None:
        return jsonify({"error": "No activity data yet. Run tracker.py first."}), 400

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
        return jsonify({"error": "No activity data yet."}), 400
    return jsonify(data)

@app.route("/api/history", methods=["GET"])
def history():
    if not os.path.exists(LOG_FILE):
        return jsonify([])

    df = pd.read_csv(LOG_FILE, encoding='utf-8', encoding_errors='ignore')
    if df.empty:
        return jsonify([])

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date

    result = []
    distracting = ["YouTube", "Instagram", "Netflix", "Twitter", "TikTok", "Facebook", "WhatsApp"]

    for date, group in df.groupby("date"):
        total_entries = len(group)
        screen_time = round(total_entries * 5 / 60, 2)

        group = group.sort_values("timestamp")
        group["gap"] = group["timestamp"].diff().dt.total_seconds().fillna(0)
        breaks = int((group["gap"] > 300).sum())

        group["prev_app"] = group["app"].shift(1)
        app_switches = int((group["app"] != group["prev_app"]).sum())

        night = group[group["timestamp"].dt.hour >= 22]
        night_usage = round(len(night) * 5 / 60, 2)

        distracting_count = group[group["app"].str.contains('|'.join(distracting), case=False, na=False)].shape[0]
        productive_ratio = round(1 - (distracting_count / max(total_entries, 1)), 2)

        stress_level = predict_stress_from_tracker(
            screen_time, screen_time, night_usage,
            app_switches, breaks, productive_ratio
        )
        stress_value = 80 if stress_level == "High" else 50 if stress_level == "Medium" else 25

        result.append({
            "date": str(date),
            "name": date.strftime("%b %d"),
            "screen_time": screen_time,
            "productive": round(screen_time * productive_ratio, 2),
            "entertainment": round(screen_time * (1 - productive_ratio), 2),
            "stress": stress_value,
            "breaks": breaks,
            "app_switches": app_switches,
        })

    return jsonify(result)

@app.route("/api/heatmap", methods=["GET"])
def heatmap():
    if not os.path.exists(LOG_FILE):
        return jsonify([])

    df = pd.read_csv(LOG_FILE, encoding='utf-8', encoding_errors='ignore')
    if df.empty:
        return jsonify([])

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["day"] = df["timestamp"].dt.dayofweek   # 0=Mon, 6=Sun
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
    return jsonify({"status": "Backend running", "user": getpass.getuser()})

@app.route("/stress")
def stress():
    result = predict_stress_from_tracker()
    return jsonify({"stress_level": int(result)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)