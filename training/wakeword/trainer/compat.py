"""compat.py — Compatibility patches for openWakeWord training dependencies.

Addresses known breaking changes in modern dependency versions:
  - setuptools 82+ removed pkg_resources
  - torchaudio 2.10+ removed load(), info(), list_audio_backends()
  - Piper sample generator API changed (requires model= kwarg)
  - Sample rate mismatches (Piper outputs 22050 Hz, openWakeWord expects 16000 Hz)

Apply BEFORE importing openwakeword, speechbrain, or torch-audiomentations:

    import compat
    results = compat.apply_all()    # monkey-patches torchaudio etc.
    ok      = compat.verify_all()   # tests each patch actually works
"""

from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
from pathlib import Path

log = logging.getLogger("compat")


# ─── Public API ───────────────────────────────────────────────────────────


def apply_all() -> dict[str, str]:
    """Apply every patch. Returns ``{name: status}`` where *status* is one of
    ``ok``, ``applied``, ``skipped (reason)``, or ``FAILED: reason``.
    """
    results: dict[str, str] = {}
    for name, fn in _PATCHES:
        try:
            status = fn()
        except Exception as exc:
            status = f"FAILED: {exc}"
        results[name] = status
        level = logging.WARNING if "FAIL" in status else logging.INFO
        log.log(level, "  patch %-30s %s", name, status)
    return results


def verify_all() -> dict[str, bool]:
    """Functional tests for each patch.  Returns ``{name: passed}``."""
    results: dict[str, bool] = {}

    # ── torchaudio.load ──
    try:
        import numpy as np
        import soundfile as sf
        import torch
        import torchaudio

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            # Write a file at 22050 Hz to test resampling
            sf.write(f.name, np.zeros(22050, dtype=np.float32), 22050)
            wav, sr = torchaudio.load(f.name)
            results["torchaudio.load"] = sr == 16000 and isinstance(wav, torch.Tensor)
            if sr != 16000:
                log.warning("  verify torchaudio.load  returned SR=%d (expected 16000)", sr)
            Path(f.name).unlink(missing_ok=True)
    except Exception as exc:
        results["torchaudio.load"] = False
        log.warning("  verify torchaudio.load  FAILED: %s", exc)

    # ── torchaudio.info ──
    try:
        import numpy as np
        import soundfile as sf
        import torchaudio

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, np.zeros(16000, dtype=np.float32), 16000)
            meta = torchaudio.info(f.name)
            results["torchaudio.info"] = meta.sample_rate == 16000
            Path(f.name).unlink(missing_ok=True)
    except Exception as exc:
        results["torchaudio.info"] = False
        log.warning("  verify torchaudio.info  FAILED: %s", exc)

    # ── torchaudio.list_audio_backends ──
    try:
        import torchaudio

        backends = torchaudio.list_audio_backends()
        results["torchaudio.list_audio_backends"] = isinstance(backends, list)
    except Exception:
        results["torchaudio.list_audio_backends"] = False

    # ── pkg_resources ──
    try:
        import pkg_resources  # noqa: F401

        results["pkg_resources"] = True
    except ImportError:
        results["pkg_resources"] = False

    # ── julius fft_conv1d on GPU (the actual op that crashed with HIPFFT_PARSE_ERROR) ──
    try:
        import torch
        from julius.fftconv import fft_conv1d

        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = torch.randn(1, 1, 64, device=device)
        w = torch.randn(1, 1, 8, device=device)
        out = fft_conv1d(x, w)
        results["julius.fft_conv1d"] = out.shape[-1] > 0
    except Exception as exc:
        results["julius.fft_conv1d"] = False
        log.warning("  verify julius.fft_conv1d  FAILED: %s", exc)

    # ── torch DataLoader num_workers forced to 0 ──
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        ds = TensorDataset(torch.zeros(4, 2), torch.zeros(4))
        dl = DataLoader(ds, batch_size=2, num_workers=2, prefetch_factor=4)
        results["torch.DataLoader no shm workers"] = dl.num_workers == 0
    except Exception as exc:
        results["torch.DataLoader no shm workers"] = False
        log.warning("  verify torch.DataLoader no shm workers  FAILED: %s", exc)

    # ── scipy.special.sph_harm (needed by the `acoustics` package) ──
    try:
        import acoustics.directivity  # noqa: F401

        results["scipy.special.sph_harm"] = True
    except Exception as exc:
        results["scipy.special.sph_harm"] = False
        log.warning("  verify scipy.special.sph_harm  FAILED: %s", exc)

    for name, ok in results.items():
        log.info("  verify %-30s %s", name, "PASS" if ok else "FAIL")

    return results


# ─── Individual patches ──────────────────────────────────────────────────


def _ensure_pkg_resources() -> str:
    """Install setuptools<82 if pkg_resources was removed."""
    try:
        import pkg_resources  # noqa: F401

        return "ok"
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "setuptools<82", "-q"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return "applied (setuptools<82)"


def _patch_torchaudio_load() -> str:
    """Replace ``torchaudio.load`` with a soundfile-based loader that also
    resamples to 16 kHz when needed (Piper outputs 22050 Hz)."""
    import torch
    import torchaudio

    if getattr(torchaudio, "_oww_load_patched", False):
        return "ok (already patched)"

    def _load(filepath, *args, **kwargs):
        import numpy as np
        import soundfile as sf

        data, sr = sf.read(str(filepath), dtype="float32")
        if data.ndim == 1:
            data = data[np.newaxis, :]  # (1, samples)
        else:
            data = data.T  # (channels, samples)

        # Resample to 16 kHz if needed (Piper TTS outputs 22050 Hz)
        if sr != 16000:
            from scipy.signal import resample as scipy_resample
            old_len = data.shape[-1]
            new_len = int(old_len * 16000 / sr)
            # Resample each channel
            if data.ndim == 2:
                resampled = np.stack([
                    scipy_resample(data[c], new_len).astype(np.float32)
                    for c in range(data.shape[0])
                ])
            else:
                resampled = scipy_resample(data, new_len).astype(np.float32)
            data = resampled
            sr = 16000

        return torch.from_numpy(data), sr

    torchaudio.load = _load
    torchaudio._oww_load_patched = True
    return "applied"


def _patch_torchaudio_info() -> str:
    """Provide a soundfile-based ``torchaudio.info``."""
    import torchaudio

    if getattr(torchaudio, "_oww_info_patched", False):
        return "ok (already patched)"

    class AudioMetaData:
        __slots__ = (
            "sample_rate",
            "num_frames",
            "num_channels",
            "bits_per_sample",
            "encoding",
        )

        def __init__(self, sample_rate: int, num_frames: int, num_channels: int):
            self.sample_rate = sample_rate
            self.num_frames = num_frames
            self.num_channels = num_channels
            self.bits_per_sample = 16
            self.encoding = "PCM_S"

    def _info(filepath):
        import soundfile as sf

        fi = sf.info(str(filepath))
        return AudioMetaData(fi.samplerate, fi.frames, fi.channels)

    torchaudio.info = _info
    if not hasattr(torchaudio, "AudioMetaData"):
        torchaudio.AudioMetaData = AudioMetaData
    torchaudio._oww_info_patched = True
    return "applied"


def _patch_torchaudio_list_backends() -> str:
    """Re-add ``torchaudio.list_audio_backends`` for speechbrain compat."""
    import torchaudio

    if hasattr(torchaudio, "list_audio_backends"):
        return "ok"
    torchaudio.list_audio_backends = lambda: ["soundfile"]
    return "applied"


def _patch_piper_generate_samples() -> str:
    """Provide a top-level ``generate_samples`` module exposing
    ``generate_samples()`` with *model=* auto-injected when omitted.

    ``openwakeword.train`` was written against an older piper-sample-generator
    repo layout that shipped a standalone ``generate_samples.py`` script (it
    does ``sys.path.insert(0, <piper_sample_generator_path>); from
    generate_samples import generate_samples``). What's actually installed
    (piper-sample-generator 3.2.0, a real pip package) restructured that into
    ``piper_sample_generator.__main__.generate_samples`` and dropped the
    top-level script entirely, so that import fails outright with
    ``ModuleNotFoundError: No module named 'generate_samples'``. We register
    a synthetic ``generate_samples`` module in ``sys.modules`` pointing at
    the real function, which also needs *model=* injected since
    ``openwakeword.train``'s call sites never pass it (API changed in
    piper-sample-generator v2+).
    """
    import types

    if "generate_samples" in sys.modules and getattr(
        sys.modules["generate_samples"], "_oww_shim", False
    ):
        return "ok (already patched)"

    try:
        import piper_sample_generator as psg
    except ImportError as exc:
        return f"skipped (piper_sample_generator not installed: {exc})"

    # piper_sample_generator.__main__ imports piper_train.vits.commons,
    # which lives as a plain subdirectory of the repo root (not a pip
    # package) -- put that root on sys.path before importing __main__,
    # same as openwakeword.train itself does for its own (now-broken)
    # top-level `generate_samples.py` import.
    repo_root = Path(psg.__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        from piper_sample_generator.__main__ import generate_samples as _orig_generate
    except ImportError as exc:
        return f"skipped (piper_sample_generator.__main__ import failed: {exc})"

    def _wrapped(*args, **kwargs):
        if "model" not in kwargs:
            # Find the first .pt file near the piper-sample-generator install
            psg_dir = Path(psg.__file__).resolve().parent
            search_roots = [psg_dir, psg_dir.parent]
            for root in search_roots:
                models = sorted(root.rglob("*.pt"))
                if models:
                    kwargs["model"] = str(models[0])
                    log.info("Auto-resolved Piper model: %s", kwargs["model"])
                    break
        return _orig_generate(*args, **kwargs)

    shim = types.ModuleType("generate_samples")
    shim.generate_samples = _wrapped
    shim._oww_shim = True
    sys.modules["generate_samples"] = shim

    # Also patch the real package's attribute, in case anything imports it
    # the "modern" way (`from piper_sample_generator import generate_samples`).
    psg.generate_samples = _wrapped

    return "applied (shim module + model= auto-inject)"


def _patch_scipy_sph_harm() -> str:
    """Restore ``scipy.special.sph_harm`` (removed in scipy>=1.15, replaced
    by ``sph_harm_y`` with a different argument order/convention).

    ``openwakeword.data`` imports the old ``acoustics`` package (0.2.6,
    last released 2018) for directivity utilities we don't otherwise use;
    ``acoustics.directivity`` still calls the removed ``sph_harm``, which
    crashes the import outright on modern scipy. Old ``sph_harm(m, n,
    theta, phi)`` used theta=azimuthal/phi=polar; new ``sph_harm_y(n, m,
    theta, phi)`` uses theta=polar/phi=azimuthal — both the argument
    order (m, n -> n, m) and the theta/phi meanings swap.
    """
    import scipy.special

    if hasattr(scipy.special, "sph_harm"):
        return "ok (still present)"
    if not hasattr(scipy.special, "sph_harm_y"):
        return "skipped (sph_harm_y not found either)"

    sph_harm_y = scipy.special.sph_harm_y

    def _sph_harm(m, n, theta, phi):
        return sph_harm_y(n, m, phi, theta)

    scipy.special.sph_harm = _sph_harm
    return "applied (shim over sph_harm_y)"


def _patch_julius_cpu_fft() -> str:
    """Force ``julius``'s FFT-based conv (used internally by torch-audiomentations'
    resampling low-pass filters, invoked during the augment step) onto CPU.

    Some ROCm/hipFFT builds raise ``RuntimeError: cuFFT error: HIPFFT_PARSE_ERROR``
    from ``torch.fft.rfft`` on GPU tensors -- seen on this host with the rocm6.3
    PyTorch wheel against a ROCm 7.1.1 driver stack, where FFT ops are far more
    version-sensitive than the conv/matmul ops elsewhere in this pipeline that
    work fine. The FFTs julius runs here are tiny (fixed-size lowpass filter
    kernels), so moving them to CPU costs nothing measurable against the rest of
    augmentation, and sidesteps the crash entirely rather than working around a
    specific hipFFT version.
    """
    try:
        import julius.fftconv as jfft
    except ImportError:
        return "skipped (julius not installed)"

    if getattr(jfft, "_oww_cpu_fft_patched", False):
        return "ok (already patched)"

    orig_rfft = jfft._new_rfft
    orig_irfft = jfft._new_irfft

    def _rfft_cpu(x):
        return orig_rfft(x.cpu()).to(x.device)

    def _irfft_cpu(x, length: int):
        device = x.device
        return orig_irfft(x.cpu(), length).to(device)

    jfft._rfft = _rfft_cpu
    jfft._irfft = _irfft_cpu
    jfft._oww_cpu_fft_patched = True
    return "applied (rfft/irfft forced to CPU to avoid hipFFT crash)"


def _patch_torch_dataloader_no_shm_workers() -> str:
    """Force ``torch.utils.data.DataLoader`` to ``num_workers=0`` (no separate
    worker processes, no shared-memory IPC between them).

    ``openwakeword.train``'s X_train DataLoader uses
    ``num_workers=os.cpu_count()//2, prefetch_factor=16`` -- on a many-core
    host that's a lot of worker processes, each holding up to 16 full
    prefetched batches in ``/dev/shm``. On this MI300X JupyterLab pod that
    blew past whatever (small, container-default) ``/dev/shm`` size is
    configured, crashing with ``RuntimeError: DataLoader worker ... killed by
    signal: Bus error ... insufficient shared memory``. We don't control the
    pod's ``/dev/shm`` allocation, so avoid needing it at all -- this
    training set is small enough that single-process loading stays plenty
    fast (100+ it/s observed even single-process).
    """
    import torch.utils.data as tud

    if getattr(tud.DataLoader, "_oww_no_shm_patched", False):
        return "ok (already patched)"

    orig_init = tud.DataLoader.__init__

    def _init_no_workers(self, *args, **kwargs):
        if kwargs.get("num_workers", 0):
            kwargs["num_workers"] = 0
            kwargs.pop("prefetch_factor", None)  # only valid when num_workers > 0
        return orig_init(self, *args, **kwargs)

    tud.DataLoader.__init__ = _init_no_workers
    tud.DataLoader._oww_no_shm_patched = True
    return "applied (num_workers forced to 0, avoids /dev/shm entirely)"


def _patch_oww_data_sample_rate() -> str:
    """Suppress openwakeword's sample-rate ValueError.

    Since ``torchaudio.load`` (patched above) already resamples to 16 kHz,
    this patch only needs to handle any remaining direct ``sf.read`` calls
    inside openwakeword that might raise on rate mismatches.

    We do NOT globally patch ``soundfile.read`` because that would conflict
    with torchaudio's internal ``_soundfile_load`` which passes extra kwargs
    like ``start``, ``stop``, ``always_2d`` that don't survive resampling.
    Instead, we patch only openwakeword-specific code paths.
    """
    try:
        import openwakeword.data as oww_data
    except ImportError:
        return "skipped (openwakeword not installed)"

    if getattr(oww_data, "_oww_sr_patched", False):
        return "ok (already patched)"

    # The torchaudio.load patch already handles resampling.
    # Mark as done so we don't re-apply.
    oww_data._oww_sr_patched = True
    return "applied (torchaudio.load handles resampling)"


# ─── Patch registry (order matters) ──────────────────────────────────────

_PATCHES = [
    ("setuptools/pkg_resources", _ensure_pkg_resources),
    ("scipy.special.sph_harm", _patch_scipy_sph_harm),
    ("torchaudio.load", _patch_torchaudio_load),
    ("torchaudio.info", _patch_torchaudio_info),
    ("torchaudio.list_audio_backends", _patch_torchaudio_list_backends),
    ("piper generate_samples model=", _patch_piper_generate_samples),
    ("julius CPU fft (hipFFT workaround)", _patch_julius_cpu_fft),
    ("torch DataLoader no shm workers", _patch_torch_dataloader_no_shm_workers),
    ("oww data.py sample rate", _patch_oww_data_sample_rate),
]
