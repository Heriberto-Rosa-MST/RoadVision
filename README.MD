RoadVision
# PyTorch Installation (IMPORTANT)

Install PyTorch separately based on your system:

GPU (recommended):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RTX 50-series (or newer GPUs):
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

CPU only:
pip install torch torchvision torchaudio