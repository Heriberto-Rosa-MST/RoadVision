# RoadVision  
Deep Learning-Based Road Object Detection System  

## Project Scope
**RoadVision** is a computer vision project that uses deep learning to detect and classify road objects in real-time video footage. The goal is to simulate a lightweight perception system similar to those used in autonomous driving and intelligent transportation systems.

This project focuses on:
- Training an object detection model using the **YOLO (You Only Look Once)** framework  
- Detecting road-relevant objects (e.g., vehicles, pedestrians, signage)  
- Running inference on a demo video to visualize real-world performance  
- Evaluating model performance after multiple training epochs  

---

## Tech Stack
- Python  
- PyTorch  
- Ultralytics YOLO  
- OpenCV  
- NumPy  

---

## Project Structure
```bash
RoadVision/
├── src/                 # Training + inference scripts
├── runs/                # Model outputs
├── data/                # Dataset
├── demo/                # Demo video & annotated results
├── data.yaml            # Dataset paths & class defintions
└── requirements.txt     # Dependencies
```
## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Heriberto-Rosa-MST/RoadVision.git
cd RoadVision
```
### 2. Create virtual environment
```bash
python -m venv .roadvision
.roadvision\Scripts\activate
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
## PyTorch Installation (IMPORTANT)
Install PyTorch separately based on your system:

GPU (recommended):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
RTX 50-series (or newer GPUs):
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```
CPU only:
```bash
pip install torch torchvision torchaudio
```
### 4. Install YOLO (Ultralytics)
```bash 
pip install ultralytics
```

## How to Use
### Train the model
```bash
python src/train.py
```
This will:

- Load dataset
- Train YOLO model
- Save weights to:
```bash
runs/detect/trainX/weights/best.pt
```
### Run inference on demo video
```python
from ultralytics import YOLO

model = YOLO("runs/detect/train2/weights/best.pt")

model.predict(
    source="demo-trimmed.mp4",
    save=True,
    conf=0.25,
    device=0
)
```
Output is saved to:
```bash
runs/detect/predict/
```
## Results (50 Epoch Training Run)
### Training Config:
- Epochs: 50
- Batch Size: 32
- Learning Rate (lr0): 0.003
### Observations:
- Model shows clear improvement in object detection accuracy over training
- Bounding boxes become more stable and precise in later epochs
- Confidence scores improve, reducing false positives

## Demo Output
- Input: ```demo-trimmed.mp4```

- Output: Annotated video with bounding boxes and confidence scores

- Saved automatically in ```runs/detect/```