import time
import csv
import os
from datetime import datetime

LOG_FILE = os.path.join(os.path.dirname(__file__), "activity_log.csv")

def get_active_window():
    try:
        import pygetwindow as gw
        win = gw.getActiveWindow()
        if win and win.title:
            return win.title
        return "Unknown"
    except:
        return "Unknown"

def log_activity():
    # Create file with headers if it doesn't exist
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "app", "duration_seconds"])
        print(f"Created log file: {LOG_FILE}")

    print("Tracker running... Press Ctrl+C to stop.")
    print(f"Logging to: {LOG_FILE}")

    while True:
        app = get_active_window()
        timestamp = datetime.now().isoformat()

        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, app, 5])

        print(f"{timestamp} | {app}")
        time.sleep(5)

if __name__ == "__main__":
    log_activity()