# ! pip install -U huggingface_hub
# ! pip install huggingface_hub hf_transfer hf_xet
# nvidia-smi

import os
from huggingface_hub import snapshot_download

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# You can then use the `api` object for interactions, but snapshot_download
# still relies on the globally configured token or environment variable.

snapshot_download(
    repo_id = "unsloth/Qwen3-32B-GGUF",  # "unsloth/DeepSeek-V3.2-GGUF",   "leafspark/Llama-3.2-11B-Vision-Instruct-GGUF"
    local_dir ="E:/Qwen3-32B-GGUF",     
    allow_patterns = ["*Q4_K_M*", "*Q5_K_M*", "*Q6_K*", "*Q8_0*", "*UD-IQ1_M*", "*UD-IQ1_S*", "*UD-Q6_K_XL*"],  
    max_workers=16,  # 8 = default
)

# allow_patterns = ["*Q4_K_M*", "*Q5_K_M*", "*Q6_K*", "*Q8_0*",  "*UD-IQ1_M*",  "*UD-IQ1_S*", "*UD-Q6_K_XL*",] 

# NousResearch/Hermes-4.3-36B-GGUF  https://huggingface.co/NousResearch/Hermes-4.3-36B-GGUF
# ACE-Step 1.5 https://huggingface.co/ACE-Step/Ace-Step1.5
# https://huggingface.co/Qwen 
# Qwen/Qwen3-ASR-1.7B https://huggingface.co/Qwen/Qwen3-ASR-1.7B
# Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
# Qwen/Qwen3-Omni-30B-A3B-Instruct


# Q5_K_M (High-quality sweet spot)
# This is where things get good.

# Pros
# Excellent reasoning fidelity
# Very strong coding performance
# Much closer to FP16 behavior
# Still memory-efficient compared to Q6/Q8

# Cons
# Larger VRAM usage
# Slightly slower than Q4
# This is the community “goldilocks” quantization
# → Best balance of quality vs performance

# Q6_K (Near-lossless)

# Pros
# Minimal quality loss
# Very strong math, logic, and code
# Excellent long-context coherence

# Cons
# Big
# Slower
# Heavy VRAM + RAM usage

# Best for
# Research
# Long-form reasoning
# When quality matters more than speed






# If I were configuring your system:
# Default: Q5_K_M
# High-quality mode: Q6_K
# Efficiency mode: Q4_K_M



# UD-IQ2_XXS, Q3_K_M  
#  Use "*UD-IQ1_S*" for Dynamic 1.78bit (151GB) 
# Dynamic 2.7bit (230GB) use "*UD-Q2_K_XL*"
# 3.5bit (320GB) use "*UD-Q3_K_XL*" for Dynamic 3.5bit
# 4.5bit (406GB) use "*UD-Q4_K_XL*" for Dynamic 3.5bit

# ["*UD-Q3_K_XL*", "*UD-Q2_K_XL*"

