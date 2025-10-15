# utils_gpu_mem.py
import os
import time
import threading
from statistics import mean
try:
    import pynvml
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False

def _init_nvml():
    if not _HAS_NVML:
        return False
    try:
        pynvml.nvmlInit()
        return True
    except Exception:
        return False

def _get_handle(device_index: int):
    return pynvml.nvmlDeviceGetHandleByIndex(device_index)

def get_gpu_memory_mb(device_index: int = 0) -> float:
    """VRAM đang dùng (MB) ở thời điểm gọi."""
    if not _init_nvml():
        return -1.0
    h = _get_handle(device_index)
    info = pynvml.nvmlDeviceGetMemoryInfo(h)
    return info.used / (1024 ** 2)

def detect_device_index(default: int = 0) -> int:
    """
    Trả về index GPU để đo. Ưu tiên:
    - ENV GPU_DEVICE_INDEX nếu được set
    - mặc định 0
    Lưu ý: nếu bạn chạy multi-GPU phức tạp, nên set GPU_DEVICE_INDEX cho chắc.
    """
    v = os.getenv("GPU_DEVICE_INDEX")
    if v is not None and v.isdigit():
        return int(v)
    return default

class GPUMemSampler:
    """
    Lấy before/after, average và peak VRAM khi chạy một khối code.
    Dùng:
        with GPUMemSampler(device_index=0, interval=0.1) as m:
            result = run_model(...)
        m.summary() -> dict
    """
    def __init__(self, device_index: int = 0, interval: float = 0.1):
        self.device_index = device_index
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self.samples = []
        self.before = None
        self.after = None
        self._ok = _init_nvml()

    def _loop(self):
        while not self._stop.is_set():
            val = get_gpu_memory_mb(self.device_index)
            if val >= 0:
                self.samples.append(val)
            time.sleep(self.interval)

    def __enter__(self):
        if self._ok:
            self.before = get_gpu_memory_mb(self.device_index)
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._ok:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=1.0)
            self.after = get_gpu_memory_mb(self.device_index)

    def summary(self):
        if not self._ok:
            return {
                "gpu_monitoring": False,
                "gpu_vram_before_mb": None,
                "gpu_vram_after_mb": None,
                "gpu_vram_avg_mb": None,
                "gpu_vram_peak_mb": None,
            }
        avg = mean(self.samples) if self.samples else (None if self.before is None or self.after is None else (self.before + self.after) / 2)
        peak = max(self.samples) if self.samples else None
        return {
            "gpu_monitoring": True,
            "gpu_device_index": self.device_index,
            "gpu_vram_before_mb": None if self.before is None else round(self.before, 2),
            "gpu_vram_after_mb": None if self.after is None else round(self.after, 2),
            "gpu_vram_avg_mb": None if avg is None else round(avg, 2),
            "gpu_vram_peak_mb": None if peak is None else round(peak, 2),
        }
