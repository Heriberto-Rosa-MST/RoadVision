import os
import kagglehub
import yaml
from ultralytics import YOLO

if __name__ == "__main__":
    # Download and prepare dataset
    dataset_path = kagglehub.dataset_download(
        "barkataliarbab/udacity-self-driving-car-obstacles-dataset", output_dir="./data")

    print(f"Dataset downloaded to: {dataset_path}")

    yaml_path = os.path.join(dataset_path, "data.yaml")
    print(f"Dataset YAML path: {yaml_path}")  # finds data.yaml

    config_path = os.path.join(os.path.dirname(__file__), "custom_yolo.yaml")
    print(f"YOLO config path: {config_path}")  # finds custom_yolo.yaml

    # ref: https://github.com/ultralytics/ultralytics/blob/main/README.md & https://docs.ultralytics.com/usage/python/
    # Load model
    # detect task for object detection
    model = YOLO(config_path, task="detect")

    # Transfer pretrained YOLO neck + head weights
    model.load("yolov8n.pt")

    # save; training checkpoints and final model weights.
    model.train(
        data=yaml_path,
        epochs=30,
        imgsz=512,
        optimizer="AdamW",
        device='cpu',  # auto-detect GPU (set to 'cpu' to force CPU)
        amp=False,   # disable automatic mixed precision - fixes NaN loss issues with custom backbone
        verbose=False  # supresses terminal output
    )