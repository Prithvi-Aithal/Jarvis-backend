import csv
from datetime import datetime
import os

LOG_FILE = os.path.join(os.path.dirname(__file__), "../api/activity_log.csv")

PRODUCTIVE_APPS = ["Code", "VS Code", "PyCharm", "Terminal"]

def extract_features():

    screen_time = 0
    night_usage = 0
    app_switches = 0
    breaks = 0
    productive_time = 0

    last_app = None
    last_time = None
    continuous_usage = 0
    current_streak = 0

    with open(LOG_FILE, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            timestamp = datetime.fromisoformat(row["timestamp"])
            app = row["app"]
            duration = int(row["duration_seconds"])

            screen_time += duration

            # night usage
            if timestamp.hour >= 23 or timestamp.hour <= 5:
                night_usage += duration

            # productive apps
            if any(p in app for p in PRODUCTIVE_APPS):
                productive_time += duration

            # app switches
            if last_app and last_app != app:
                app_switches += 1

            # breaks
            if last_time:
                diff = (timestamp - last_time).seconds
                if diff > 300:
                    breaks += 1

            # continuous usage
            current_streak += duration
            if current_streak > continuous_usage:
                continuous_usage = current_streak

            last_app = app
            last_time = timestamp

    productive_ratio = productive_time / screen_time if screen_time else 0

    return [
        screen_time,
        continuous_usage,
        night_usage,
        app_switches,
        breaks,
        productive_ratio
    ]