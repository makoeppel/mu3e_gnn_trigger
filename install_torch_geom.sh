#!/bin/bash
set -e

# 1. Upgrade pip tooling
python3 -m pip install --upgrade pip setuptools wheel

# 2. Clear any cached partial wheels
pip cache purge || true

# 3. Install PyTorch 2.0.1 (last Pascal sm_61 support) with CUDA 11.8
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# 4. Install PyTorch Geometric + extensions from prebuilt wheels
pip install torch-scatter==2.1.1 torch-sparse==0.6.17 \
    torch-cluster==1.6.1 torch-spline-conv==1.2.2 torch-geometric==2.3.1 \
    -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

# 5. Install the rest of the scientific stack
pip install awkward awkward-pandas matplotlib keras numpy<2.0 pandas scikit-learn \
    tensorflow uproot ipython jupyter seaborn torcheval
