# test_using_gpus.py
import psutil

def _bytes_gb(x: int | float) -> float:
    return float(x) / (1024 ** 3)

def snapshot_ram_gb() -> float:
    p = psutil.Process()
    return _bytes_gb(p.memory_info().rss)

def _try_import_pynvml():
    """
    Works with:
      - pip install nvidia-ml-py   (recommended)
      - pip install pynvml         (deprecated, but may exist)
    """
    try:
        import pynvml  # provided by nvidia-ml-py
        return pynvml
    except Exception as e:
        return None

def snapshot_vram_and_util():
    pynvml = _try_import_pynvml()
    if pynvml is None:
        return [("NVML_ERROR", "pynvml import failed (install nvidia-ml-py)", None, None, None, None)]

    try:
        pynvml.nvmlInit()
        out = []
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)

            mem_used_bytes = int(mem.used)
            mem_total_bytes = int(mem.total)

            # name can be bytes on some builds
            if isinstance(name, (bytes, bytearray)):
                name = name.decode("utf-8", errors="replace")

            out.append((i, name, _bytes_gb(mem_used_bytes), _bytes_gb(mem_total_bytes), util.gpu, util.memory))
        return out
    except Exception as e:
        return [("NVML_ERROR", str(e), None, None, None, None)]
    finally:
        # Never let shutdown errors bubble up
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

def debug_gpu_memory_banner(tag: str) -> None:
    """
    MUST NEVER raise. If NVML fails, we print an error and continue.
    """
    try:
        print(f"\n=== {tag} ===")
        print(f"RAM RSS (GB): {snapshot_ram_gb():.2f}")
        rows = snapshot_vram_and_util()
        for (i, name, used, total, gpu_util, mem_util) in rows:
            if i == "NVML_ERROR":
                print(f"GPU: NVML_ERROR: {name}")
            else:
                print(f"GPU {i}: {name} | VRAM {used:.2f}/{total:.2f} GB | util {gpu_util}% | mem {mem_util}%")
    except Exception as e:
        # Absolute last-ditch safety
        print(f"\n=== {tag} ===")
        print(f"[debug_gpu_memory_banner error ignored] {e}")
