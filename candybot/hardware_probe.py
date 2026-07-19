"""Safe GPU/ROCm device probing for candybot.

This dev laptop's iGPU (gfx902, "Picasso") is not on ROCm's officially-supported
list. `rocminfo` detecting it as a KFD compute agent only means the kernel driver
sees it -- it does not guarantee rocBLAS/MIOpen kernels actually work, and since
this iGPU also drives the desktop, a bad kernel launch can hang the session.

So the *entire* probe (import torch, check cuda.is_available(), run a real tensor
op) always happens in an isolated subprocess with a timeout -- never inline in the
calling process. Every other module reads its device via get_device() and never
hardcodes "cuda".
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache

_PROBE_TIMEOUT_S = 15

_SUBPROCESS_CODE = """
import json
try:
    import torch
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print(json.dumps({"cuda_available": False, "ok": False, "gpu_name": None,
                           "error": "torch.cuda.is_available() is False"}))
    else:
        x = torch.randn(256, 256, device="cuda")
        (x @ x).sum().item()
        torch.cuda.synchronize()
        print(json.dumps({"cuda_available": True, "ok": True,
                           "gpu_name": torch.cuda.get_device_name(0), "error": None}))
except Exception as e:
    print(json.dumps({"cuda_available": False, "ok": False, "gpu_name": None, "error": str(e)}))
"""


@dataclass
class DeviceReport:
    device: str  # "cuda" or "cpu" -- what code should actually use
    cuda_available: bool  # what torch.cuda.is_available() claimed (necessary, not sufficient)
    gpu_validated: bool  # whether a real tensor op succeeded in the isolated subprocess
    gpu_name: str | None
    error: str | None


def _run_probe_subprocess() -> dict:
    try:
        result = subprocess.run(
            [sys.executable, "-c", _SUBPROCESS_CODE],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired:
        return {
            "cuda_available": False,
            "ok": False,
            "gpu_name": None,
            "error": f"GPU probe timed out after {_PROBE_TIMEOUT_S}s (kernel launch likely hung)",
        }

    output = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    if not output:
        err = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "probe subprocess crashed with no output"
        return {"cuda_available": False, "ok": False, "gpu_name": None, "error": err}

    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return {"cuda_available": False, "ok": False, "gpu_name": None, "error": "could not parse probe output"}


@lru_cache(maxsize=1)
def probe() -> DeviceReport:
    """Probes GPU availability once per process (isolated + timeout-guarded) and caches it."""
    result = _run_probe_subprocess()
    if result.get("ok"):
        return DeviceReport(
            device="cuda",
            cuda_available=True,
            gpu_validated=True,
            gpu_name=result.get("gpu_name"),
            error=None,
        )
    return DeviceReport(
        device="cpu",
        cuda_available=bool(result.get("cuda_available")),
        gpu_validated=False,
        gpu_name=None,
        error=result.get("error"),
    )


def get_device() -> str:
    """Single source of truth for device selection across candybot. Never hardcode 'cuda'/'cpu' elsewhere."""
    return probe().device


def main() -> None:
    report = probe()
    print(json.dumps(asdict(report), indent=2))
    if report.device == "cuda":
        print(f"\n✓ GPU validated: {report.gpu_name} -- inference will target the iGPU.")
    else:
        print(f"\n✗ Falling back to CPU ({report.error}) -- inference will run on CPU.")


if __name__ == "__main__":
    main()
