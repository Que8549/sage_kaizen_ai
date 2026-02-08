from llama_cpp import Llama
import time

q5_model_path=r"E:/DeepSeek-V3.2-GGUF/UD-IQ1_S/DeepSeek-V3.2-UD-IQ1_S-00001-of-00004.gguf"
q6_model_path=r"E:/DeepSeek-V3.2-GGUF/UD-IQ1_M/DeepSeek-V3.2-UD-IQ1_M-00001-of-00005.gguf"

use_model_path = q5_model_path

# Initialize model with GPU offloading and tensor splitting
llm = Llama(
    model_path=use_model_path,
    n_ctx=8192,                       # Context window size
    n_batch=2048,                     # Batch size for prompt processing
    n_ubatch=512,                     # Batch size for generation
    n_gpu_layers=61,                  # Offload 48 layers to GPU
    tensor_split=[2.0, 1.0],          # GPU memory distribution
    split_mode=1,                     # Layer-wise splitting
    main_gpu=0,                       # Primary GPU
    # cache_type_k="q8_0",              # K-cache quantization   "q8_0"
    # cache_type_v="q8_0",            # Quantizing the V-vectors  # seems to run slower
    verbose=True                      # Enable verbose output
)

# fp16 (default): This uses full precision (16-bit floating-point) for the KV cache.
# q8_0: This quantizes the KV cache to 8-bit
# q4_0: This quantizes the KV cache to 4-bit

def generate_text(prompt: str) -> str:
    """Generate text using specified sampling parameters"""
    output = llm.create_completion(
        prompt,
        temperature=0.4,              # Q5_K_M (Precision Mode) temperature=0.4, | Q6_K (Depth Mode) temperature=0.6    
        min_p=0.02,                   # Q5_K_M (Precision Mode) min_p=0.02,      | Q6_K (Depth Mode) top_k=50
        top_k=40,                     # Q5_K_M (Precision Mode) top_k=40,        | Q6_K (Depth Mode) top_p=0.95
        top_p=0.92,                   # Q5_K_M (Precision Mode) top_p=0.92,      | Q6_K (Depth Mode) min_p=0.03
        max_tokens=12288,             # Allow longer responses 6144 = 1024 * 6   8192 = 1024 * 8
        stop=None                     # No stop sequence
    )

    with open("output.txt", "a", encoding="utf-8") as f:
        f.write(f"\n\nRaw Output: {output}\n\n")
    
    return output['choices'][0]['text']

           
if __name__ == "__main__":
    prompt = "Provide a detailed, structured explanation with examples. Cover multiple civilizations and explain religious significance in depth. Ans: "

    start_time = time.time()
    response = generate_text(prompt)
    end_time = time.time()
    
    elapsed_time = end_time - start_time

    with open("output.txt", "a", encoding="utf-8") as f:
        f.write(f"\n\nUsing model [{use_model_path}] Total Time: {elapsed_time}\n{prompt}\n{response}\n")

    print(f"\n\n {response}")
