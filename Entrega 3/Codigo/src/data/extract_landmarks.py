"""
Uso:
  py -m src.data.extract_landmarks
"""

import cv2
import csv
from pathlib import Path
import mediapipe as mp

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)

def process_video(video_path: Path):
    activity = video_path.stem.split("_")[0]  # ejemplo: walk_toward_p01_t01 -> walk
    cap = cv2.VideoCapture(str(video_path))
    rows = []
    header = ["frame", "activity"] + [f"l{i}_{c}" for i in range(33) for c in ("x", "y", "z", "v")]
    rows.append(header)
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            flat = [frame_idx, activity] + [val for p in lm for val in (p.x, p.y, p.z, p.visibility)]
            rows.append(flat)
        frame_idx += 1

    cap.release()

    out_path = OUT_DIR / f"{video_path.stem}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"[OK] {video_path.name} → {out_path.name} ({len(rows)-1} frames)")

def main():
    for video in RAW_DIR.glob("*.mp4"):
        process_video(video)

if __name__ == "__main__":
    main()
