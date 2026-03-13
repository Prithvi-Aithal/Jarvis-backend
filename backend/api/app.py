import sys
sys.path.insert(0, r'D:\Prithvi\WEAL Hackathon Project\Jarvis\backend\ml')

from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import os

from predictor import predict_stress

app = Flask(__name__)
CORS(app)

LOG_FILE = os.path.join(os.path.dirname(__file__), "activity_log.csv")


def compute_features():
    if not os.path.exists(LOG_FILE):
        return None

    df = pd.read_csv(LOG_FILE)

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


@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({"status": "Backend running"})


@app.route("/api/stress", methods=["GET"])
def stress():
    features = compute_features()

    if features is None:
        return jsonify({"error": "No activity data yet. Run tracker.py first."}), 400

    stress_level = predict_stress(
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


if __name__ == "__main__":
    app.run(debug=True, port=5000)