import os
import shutil
import kagglehub
import random
from ultralytics import YOLO

def split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    # output directories
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)

    images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    random.seed(seed)
    random.shuffle(images)

    total_images = len(images)
    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    # copy files to appropriate splits
    for split, file_list in splits.items():
        for img_name in file_list:
            # copy image
            shutil.copy(
                os.path.join(images_dir, img_name),
                os.path.join(output_dir, split, 'images', img_name)
            )
            
            # copy corresponding label
            label_name = os.path.splitext(img_name)[0] + '.txt'
            label_src = os.path.join(labels_dir, label_name)
            if os.path.exists(label_src):
                shutil.copy(
                    label_src,
                    os.path.join(output_dir, split, 'labels', label_name)
                )
    
    print(f"Dataset split complete:")
    print(f"  Train: {len(splits['train'])} images")
    print(f"  Val:   {len(splits['val'])} images")
    print(f"  Test:  {len(splits['test'])} images")


if __name__ == "__main__":
    # Download and prepare dataset
    cache_path = kagglehub.dataset_download("barkataliarbab/udacity-self-driving-car-obstacles-dataset")
    
    images_dir = os.path.join(cache_path, "export", "images")
    labels_dir = os.path.join(cache_path, "export", "labels")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    # split if dataset not already split
    if not os.path.exists(os.path.join(output_dir, "train")):
        print("splitting datset into train/val/test")
        split_dataset(images_dir, labels_dir, output_dir)
    else:
        print("dataset already split)")

    config_path = os.path.join(os.path.dirname(__file__), "custom_yolo.yaml")

    # ref: https://github.com/ultralytics/ultralytics/blob/main/README.md & https://docs.ultralytics.com/usage/python/
    # load model
    model = YOLO(config_path, task="detect") # detect task for object detection

    # transfer pretrained YOLO neck + head weights
    model.load("yolov8n.pt")
   
    model.train(
        data="./data.yaml", # path to data config file
        epochs=5,
        imgsz=512,
        optimizer="AdamW",
        batch=16,
        device=0,  # auto-detect GPU (set to 'cpu' to force CPU)
        amp=True, # automatic mixed precision for faster training on compatible hardware
        verbose=False # supresses terminal output
    )
