import os
from ultralytics import YOLO

model = YOLO("runs/detect/train2/weights/best.pt")

# go one level up
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# path to demo folder
demo_dir = os.path.join(parent_dir, "demo")
# demo video directory
demo_video_path = os.path.join(demo_dir, "demo-trimmed.mp4")


model.predict(
    source=demo_video_path,
    save=True,
    conf=0.25,
    device=0,
    project=demo_dir,
    name="demo_results"
)