# nvidia-smi to verify your GPU is recognized and CUDA is installed
# nvcc --version to check the installed CUDA Toolkit compiler version
# python has_gpu.py

# install PyTorch from here: https://pytorch.org/get-started/locally/  pip3
# old - pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

import torch

print("GPU Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("GPU Name:", torch.cuda.get_device_name(1) if torch.cuda.is_available() else "None")
print("GPU Count:", torch.cuda.device_count())
print("Current Device:", torch.cuda.current_device() if torch.cuda.is_available() else "None")


# Get CUDA version
# nvcc --version

# Cuda compilation tools, release 13.2, V13.2.78
# Build cuda_13.2.r13.2/compiler.37668154_0


# Check CUDA indices with nvidia-smi
# nvidia-smi



# Sun May 24 10:57:03 2026
# +-----------------------------------------------------------------------------------------+
# | NVIDIA-SMI 596.49                 Driver Version: 596.49         CUDA Version: 13.2     |
# +-----------------------------------------+------------------------+----------------------+
# | GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
# |                                         |                        |               MIG M. |
# |=========================================+========================+======================|
# |   0  NVIDIA GeForce RTX 5090      WDDM  |   00000000:01:00.0  On |                  N/A |
# |  0%   39C    P8             21W /  600W |    2796MiB /  32607MiB |      0%      Default |
# |                                         |                        |                  N/A |
# +-----------------------------------------+------------------------+----------------------+
# |   1  NVIDIA GeForce RTX 5090      WDDM  |   00000000:03:00.0 Off |                  N/A |
# |  0%   33C    P8              7W /  600W |       0MiB /  32607MiB |      0%      Default |
# |                                         |                        |                  N/A |
# +-----------------------------------------+------------------------+----------------------+


# GPU1 upgraded to Gigabyte GeForce RTX 5090 OC (32 GB VRAM) on 2026-05-24.
# Run nvidia-smi again to capture updated output after the hardware swap.
