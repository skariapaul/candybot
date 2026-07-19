"""Unit tests for candybot.hardware_probe -- exercises the subprocess-isolation
and CPU-fallback logic without touching a real GPU.
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

from candybot.hardware_probe import probe


def _fake_result(stdout: str = "", stderr: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def test_probe_falls_back_to_cpu_on_timeout():
    probe.cache_clear()
    with patch(
        "candybot.hardware_probe.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="x", timeout=15),
    ):
        report = probe()
    assert report.device == "cpu"
    assert report.gpu_validated is False
    assert "timed out" in report.error


def test_probe_falls_back_to_cpu_when_cuda_unavailable():
    probe.cache_clear()
    payload = '{"cuda_available": false, "ok": false, "gpu_name": null, "error": "torch.cuda.is_available() is False"}'
    with patch("candybot.hardware_probe.subprocess.run", return_value=_fake_result(stdout=payload)):
        report = probe()
    assert report.device == "cpu"
    assert report.cuda_available is False


def test_probe_returns_cuda_when_gpu_op_succeeds():
    probe.cache_clear()
    payload = '{"cuda_available": true, "ok": true, "gpu_name": "AMD Radeon Graphics", "error": null}'
    with patch("candybot.hardware_probe.subprocess.run", return_value=_fake_result(stdout=payload)):
        report = probe()
    assert report.device == "cuda"
    assert report.gpu_validated is True
    assert report.gpu_name == "AMD Radeon Graphics"


def test_probe_falls_back_to_cpu_on_unparseable_output():
    probe.cache_clear()
    with patch("candybot.hardware_probe.subprocess.run", return_value=_fake_result(stdout="not json")):
        report = probe()
    assert report.device == "cpu"


def test_probe_falls_back_to_cpu_on_empty_output_with_stderr():
    probe.cache_clear()
    with patch(
        "candybot.hardware_probe.subprocess.run",
        return_value=_fake_result(stdout="", stderr="Traceback...\nRuntimeError: HIP error", returncode=1),
    ):
        report = probe()
    assert report.device == "cpu"
    assert "HIP error" in report.error
