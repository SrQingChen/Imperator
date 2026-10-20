"""性能监控：2s 后台采样 CPU/内存/磁盘/GPU，内存快照 + deque 历史。"""
from __future__ import annotations

import threading
import time
from collections import deque

import psutil

from . import config


def _query_gpu_smi() -> dict | None:
    import shutil
    import subprocess

    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        out = subprocess.run(
            [smi, "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4)
        parts = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
        return {"util": int(float(parts[0])), "mem_used_mb": int(float(parts[1])),
                "mem_total_mb": int(float(parts[2])), "temp_c": int(float(parts[3]))}
    except Exception:
        return None


def _query_torch_vram() -> dict | None:
    try:
        config.ensure_torch()  # 串行初始化；避免与任务线程并发 import torch
        import torch
        if torch.cuda.is_available():
            return {"allocated_mb": int(torch.cuda.memory_allocated() / 1048576),
                    "reserved_mb": int(torch.cuda.memory_reserved() / 1048576)}
    except Exception:
        pass
    return None


class PerfMonitor:
    """监控自身绝不能影响主流程：所有异常吞掉。"""

    def __init__(self, interval: float = 2.0):
        self.interval = interval
        self._hist_cpu: deque[float] = deque(maxlen=120)
        self._hist_ram: deque[float] = deque(maxlen=120)
        self._hist_gpu: deque[float] = deque(maxlen=120)
        self._hist_vram: deque[float] = deque(maxlen=120)
        self._snap: dict = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="perf-monitor", daemon=True)
        psutil.cpu_percent(interval=None)  # 首次调用初始化

    def start(self) -> None:
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample()
            except Exception:
                pass
            self._stop.wait(self.interval)

    def _sample(self) -> None:
        cpu = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        ram_pct = vm.percent
        disk = psutil.disk_usage(str(config.DATA_DIR))
        gpu = _query_gpu_smi()
        tv = _query_torch_vram()
        self._hist_cpu.append(cpu)
        self._hist_ram.append(ram_pct)
        snap = {
            "ts": time.time(), "cpu_pct": cpu, "ram_pct": ram_pct,
            "ram_used_gb": round(vm.used / 1024 ** 3, 2), "ram_total_gb": round(vm.total / 1024 ** 3, 2),
            "disk_free_gb": round(disk.free / 1024 ** 3, 1),
        }
        if gpu:
            self._hist_gpu.append(float(gpu["util"]))
            self._hist_vram.append(gpu["mem_used_mb"])
            snap.update({"gpu_util_pct": gpu["util"], "gpu_mem_used_mb": gpu["mem_used_mb"],
                         "gpu_mem_total_mb": gpu["mem_total_mb"], "gpu_temp_c": gpu["temp_c"]})
        if tv:
            snap.update({"torch_alloc_mb": tv["allocated_mb"], "torch_reserved_mb": tv["reserved_mb"]})
        self._snap = snap

    def snapshot(self) -> dict:
        snap = dict(self._snap)
        snap["hist_cpu"] = list(self._hist_cpu)
        snap["hist_ram"] = list(self._hist_ram)
        snap["hist_gpu"] = list(self._hist_gpu)
        snap["hist_vram"] = list(self._hist_vram)
        return snap
