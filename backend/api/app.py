import sys
import os
from datetime import date, datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ml'))

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from ml.predictor import predict_stress_from_tracker

app = Flask(__name__)
CORS(app)

LOG_FILE = os.path.join(os.path.dirname(__file__), "activity_log.csv")
HISTORY_DAYS = 7
DISTRACTING = ["YouTube", "Instagram", "Netflix", "Twitter", "TikTok", "Facebook", "WhatsApp"]


def load_log():
    if not os.path.exists(LOG_FILE):
        return None
    df = pd.read_csv(LOG_FILE, encoding='utf-8', encoding_errors='ignore')
    if df.empty:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], format='mixed')
    return df


def extract_base_app(title: str) -> str:
    if not isinstance(title, str):
        return "Unknown"
    parts = [p.strip() for p in title.split(" - ")]
    return parts[-1] if parts else title


def compute_features_for_group(group: pd.DataFrame):
    group = group.sort_values("timestamp")
    total_entries = len(group)
    screen_time = total_entries * 5 / 60

    night = group[group["timestamp"].dt.hour >= 22]
    night_usage = len(night) * 5 / 60

    group = group.copy()
    group["gap"] = group["timestamp"].diff().dt.total_seconds().fillna(0)

    current_streak = 0
    max_streak = 0
    for gap in group["gap"]:
        if gap <= 300:
            current_streak += 5
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 5
    continuous_usage = max_streak / 60

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
    df = load_log()
    if df is None:
        return None
    today = date.today()
    today_df = df[df["timestamp"].dt.date == today]
    if today_df.empty:
        return None
    return compute_features_for_group(today_df)


def compute_wellness_score(features, stress_level):
    score = 100.0
    stress_penalty = {"High": 40, "Medium": 20, "Low": 0}
    score -= stress_penalty.get(stress_level, 20)

    screen_time = features["screen_time"]
    if screen_time > 4:
        score -= min(20, (screen_time - 4) * 5)

    score -= min(15, features["night_usage"] * 10)
    score += features["productive_ratio"] * 10
    score += min(5, features["breaks"] * 1.5)

    return max(0, min(100, round(score)))


@app.route("/api/stress", methods=["GET"])
def get_stress():
    features = compute_features()
    if features is None:
        return jsonify({"error": "No activity data yet for today. Run tracker.py first."}), 400

    stress_level = predict_stress_from_tracker(
        features["screen_time"], features["continuous_usage"],
        features["night_usage"], features["app_switches"],
        features["breaks"], features["productive_ratio"]
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
    wellness_score = compute_wellness_score(features, stress_level)

    return jsonify({
        "stress_level": stress_level,
        "wellness_score": wellness_score,
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


@app.route("/api/wellness", methods=["GET"])
def wellness():
    features = compute_features()
    if features is None:
        return jsonify({
            "wellness_score": 100,
            "stress_level": "Low",
            "screen_time_minutes": 0,
            "screen_time_hours": 0.0,
            "message": "No data yet — start tracker.py"
        })

    stress_level = predict_stress_from_tracker(
        features["screen_time"], features["continuous_usage"],
        features["night_usage"], features["app_switches"],
        features["breaks"], features["productive_ratio"]
    )

    wellness_score = compute_wellness_score(features, stress_level)
    screen_minutes = round(features["screen_time"] * 60)
    screen_hours = round(features["screen_time"], 2)

    return jsonify({
        "wellness_score": wellness_score,
        "stress_level": stress_level,
        "screen_time_minutes": screen_minutes,
        "screen_time_hours": screen_hours,
        "productive_ratio": features["productive_ratio"],
        "breaks": features["breaks"],
        "night_usage": features["night_usage"],
        "message": f"{screen_hours}h {screen_minutes % 60}m of screen time today"
    })


@app.route("/api/history", methods=["GET"])
def history():
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
        wellness = compute_wellness_score(f, stress_level)

        result.append({
            "date": str(day),
            "name": day.strftime("%b %d"),
            "screen_time": f["screen_time"],
            "screen_time_minutes": round(f["screen_time"] * 60),
            "productive": round(f["screen_time"] * f["productive_ratio"], 2),
            "entertainment": round(f["screen_time"] * (1 - f["productive_ratio"]), 2),
            "stress": stress_value,
            "wellness": wellness,
            "breaks": f["breaks"],
            "app_switches": f["app_switches"],
        })

    result.sort(key=lambda x: x["date"])
    return jsonify(result)


@app.route("/api/heatmap", methods=["GET"])
def heatmap():
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


@app.route("/api/alerts", methods=["GET"])
def get_alerts():
    features = compute_features()
    alerts = []
    hour = datetime.now().hour

    if features is None:
        return jsonify([])

    stress_level = predict_stress_from_tracker(
        features["screen_time"], features["continuous_usage"],
        features["night_usage"], features["app_switches"],
        features["breaks"], features["productive_ratio"]
    )
    wellness = compute_wellness_score(features, stress_level)
    screen_mins = int(features["screen_time"] * 60)

    if stress_level == "High":
        alerts.append({
            "id": "stress_high", "type": "danger",
            "title": "Stress Spike Detected 🔴",
            "message": "Digital fatigue is critical. Close some tabs and step away.",
            "action": "Take a 5-min break"
        })
    elif stress_level == "Medium":
        alerts.append({
            "id": "stress_medium", "type": "warning",
            "title": "Fatigue Building Up ⚠️",
            "message": "Stress is climbing steadily. Your brain needs a breather.",
            "action": "Stretch for 2 mins"
        })

    if screen_mins >= 45 and features["breaks"] == 0:
        alerts.append({
            "id": "break_overdue", "type": "warning",
            "title": "Break Overdue 🪑",
            "message": f"{screen_mins} minutes of continuous screen time with zero breaks.",
            "action": "Stand up & stretch now"
        })

    if features["screen_time"] >= 6:
        alerts.append({
            "id": "screentime_6h", "type": "danger",
            "title": "6 Hours on Screen 😵",
            "message": "You have been at this for 6 hours today. Serious eye strain risk.",
            "action": "20-20-20 rule: look 20ft away for 20 sec"
        })
    elif features["screen_time"] >= 4:
        alerts.append({
            "id": "screentime_4h", "type": "warning",
            "title": "4 Hour Mark Crossed ⏱️",
            "message": "You have hit 4 hours of screen time. Go outside for 10 mins.",
            "action": "Touch some grass 🌿"
        })
    elif features["screen_time"] >= 2:
        alerts.append({
            "id": "screentime_2h", "type": "info",
            "title": "Hydration Check 💧",
            "message": "2 hours in. When did you last drink water?",
            "action": "Grab a glass of water"
        })

    if hour >= 22 and features["night_usage"] > 0:
        alerts.append({
            "id": "night_owl", "type": "info",
            "title": "Night Owl Alert 🌙",
            "message": "Late-night screen use crushes sleep quality and next-day focus.",
            "action": "Wind down in 30 mins"
        })

    if features["productive_ratio"] < 0.3:
        alerts.append({
            "id": "distraction_mode", "type": "warning",
            "title": "Distraction Mode: ON 📵",
            "message": "Over 70% of your time is on YouTube / Instagram / Netflix.",
            "action": "25-min Pomodoro sprint"
        })

    if wellness < 40:
        alerts.append({
            "id": "wellness_critical", "type": "danger",
            "title": "Wellness Critical 🚨",
            "message": f"Your wellness score dropped to {wellness}. Jarvis recommends a hard stop.",
            "action": "Close laptop for 15 mins"
        })

    if wellness >= 85 and stress_level == "Low" and screen_mins > 30:
        alerts.append({
            "id": "crushing_it", "type": "success",
            "title": "You're Crushing It 🎯",
            "message": f"Wellness at {wellness}, stress low, great balance. Keep it up.",
            "action": None
        })

    return jsonify(alerts)


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "").lower()

    features = compute_features()
    if features is None:
        return jsonify({"reply": "No tracking data yet — start tracker.py and check back in a few minutes."})

    stress_level = predict_stress_from_tracker(
        features["screen_time"], features["continuous_usage"],
        features["night_usage"], features["app_switches"],
        features["breaks"], features["productive_ratio"]
    )
    wellness = compute_wellness_score(features, stress_level)
    screen_mins = int(features["screen_time"] * 60)
    hours = screen_mins // 60
    mins = screen_mins % 60
    prod_ratio = round(features["productive_ratio"] * 100)
    break_count = features["breaks"]

    if any(w in user_message for w in ["how was", "my day", "summary", "overall"]):
        if stress_level == "Low":
            tail = "Not bad at all — keep it up!"
        elif stress_level == "Medium":
            tail = "Worth taking a proper break soon."
        else:
            tail = "Seriously, close the laptop for 15 mins."
        reply = f"You have clocked {hours}h {mins}m of screen time with a wellness score of {wellness}/100. Stress is sitting at {stress_level.lower()} — {tail}"

    elif any(w in user_message for w in ["stress", "burnout", "tired", "burning"]):
        if stress_level == "Low":
            reply = f"Your stress is low right now. You are in good shape — productive ratio is {prod_ratio}% which is solid."
        else:
            break_msg = "decent." if break_count >= 2 else "not enough. Step away for 5 mins."
            reply = f"Your stress is {stress_level.lower()} right now. You have had {break_count} breaks today — that is {break_msg}"

    elif any(w in user_message for w in ["focus", "productive", "productivity"]):
        if prod_ratio >= 70:
            tail = "Solid focus — keep riding that wave."
        elif prod_ratio >= 40:
            tail = f"About {100 - prod_ratio}% of your time went to distracting apps. A Pomodoro sprint might help."
        else:
            tail = "Majority of screen time has been on distracting apps today. Time to lock in."
        reply = f"Your productive ratio is {prod_ratio}% today. {tail}"

    elif any(w in user_message for w in ["break", "rest", "stop", "should i"]):
        if break_count >= 3:
            tail = "You are pacing yourself well."
        else:
            tail = "That is on the low side — stand up, stretch, grab water. Your brain will thank you."
        reply = f"You have taken {break_count} breaks over {hours}h {mins}m. {tail}"

    elif any(w in user_message for w in ["screen time", "how long", "hours"]):
        if features["screen_time"] < 4:
            tail = "That is within healthy range — nice."
        elif features["screen_time"] < 6:
            tail = "That is getting up there. Try to cap it soon."
        else:
            tail = "That is a long day. Seriously consider wrapping up."
        reply = f"You have been on screen for {hours}h {mins}m today. {tail}"

    elif any(w in user_message for w in ["fix", "improve", "tomorrow", "better"]):
        tips = []
        if break_count < 2:
            tips.append("take more breaks — at least one every 45 mins")
        if features["productive_ratio"] < 0.6:
            tips.append("cut down on YouTube and Instagram during work hours")
        if features["screen_time"] > 5:
            tips.append("set a hard stop time for screens")
        if not tips:
            tips.append("honestly not much — you had a solid day")
        reply = "Tomorrow, focus on: " + ", and ".join(tips) + "."

    elif any(w in user_message for w in ["hi", "hello", "hey"]):
        reply = f"Hey! Wellness at {wellness}/100, stress is {stress_level.lower()}. What do you want to know about your day?"

    else:
        reply = f"Right now: {hours}h {mins}m screen time, wellness score {wellness}/100, stress {stress_level.lower()}. Ask me about your focus, breaks, stress, or what to fix tomorrow."

    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True, port=5000)