import os, time
import psutil

def _bytes_gb(x): 
    return x / (1024**3)

def snapshot_ram_gb():
    p = psutil.Process()
    return _bytes_gb(p.memory_info().rss)

def snapshot_vram_gb():
    # NVIDIA only; pip install pynvml
    try:
        import pynvml
        pynvml.nvmlInit()
        out = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            out.append((i, name, _bytes_gb(mem.used), _bytes_gb(mem.total)))
        return out
    except Exception as e:
        return [("NVML_ERROR", str(e), None, None)]
    
def snapshot_vram_and_util():
    import pynvml
    pynvml.nvmlInit()
    out = []
    for i in range(pynvml.nvmlDeviceGetCount()):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(h)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        out.append((i, name, _bytes_gb(mem.used), _bytes_gb(mem.total), util.gpu, util.memory))
    return out

def debug_gpu_memory_banner(tag):
    print(f"\n=== {tag} ===")
    print(f"RAM RSS (GB): {snapshot_ram_gb():.2f}")
    for (i, name, used, total, gpu_util, mem_util) in snapshot_vram_and_util():
        print(f"GPU {i}: {name} | VRAM {used:.2f}/{total:.2f} GB | util {gpu_util}% | mem {mem_util}%")

# Example usage:
# debug_gpu_memory_banner("before model load")
# llm = Llama(...)
# debug_gpu_memory_banner("after model load")
# llm.create_completion(...)
# debug_gpu_memory_banner("after first inference")
