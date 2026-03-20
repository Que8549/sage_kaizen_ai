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


# Check CUDA indices with nvidia-smi
# nvidia-smi
