# ! pip install -U huggingface_hub
# ! pip install huggingface_hub hf_transfer hf_xet


import os
from huggingface_hub import snapshot_download

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# You can then use the `api` object for interactions, but snapshot_download
# still relies on the globally configured token or environment variable.

snapshot_download(
    repo_id = "unsloth/DeepSeek-V3.2-GGUF",  # "unsloth/DeepSeek-V3-0324-GGUF",   "leafspark/Llama-3.2-11B-Vision-Instruct-GGUF"
    local_dir ="E:/DeepSeek-V3.2-GGUF",     
    allow_patterns = ["*Q4_K_M*", "*Q5_K_M*", "*Q6_K*"],  # allow_patterns = [""*Q8_0*", "*Q6_K*", "*Q4_K_S*", "*Q4_K_M*", "*Q5_K_S*", "*Q5_K_M*", "*Q6_K*", "*Q8_0*"] 
    # , timeout=30.0,  # Increase from default 10s [web:11]
    # max_workers=4,  # Limit parallelism if bandwidth-constrained
    # resume_download=True,  # Resume on retry [web:6]
)

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

