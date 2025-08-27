# 1. Core + ML + PyTorch (no heavy PyG extensions)
pip install numpy pandas matplotlib seaborn scikit-learn ipython jupyter \
    awkward awkward-pandas uproot \
    tensorflow keras \
    torch torchvision torchaudio torcheval \
    torch-geometric

# 2. Detect torch version and CPU/GPU build
TORCH_VER=$(python -c "import torch; print(torch.__version__)")


# 3. Install PyTorch Geometric extensions with the right wheel index
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-$TORCH_VER+cpu.html