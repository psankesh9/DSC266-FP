"""Safe GPU acquisition.

The development GPU also drives the desktop, a browser, and games, so this
module deliberately refuses to take the whole card. It caps the fraction of
VRAM this process may allocate, verifies that kernels actually launch for the
device's compute capability, and degrades to CPU rather than crashing.

The capability check matters on Blackwell (sm_120): a CUDA wheel built for an
older architecture installs and reports ``cuda.is_available() == True``, then
fails at the first kernel launch with "no kernel image is available". Catching
that here turns a confusing mid-training crash into a clean startup message.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# Leave the rest of the card for the desktop. Raised from 0.50 when the
# encoder moved to ModernBERT: the context went from 512 to 1,152 tokens, and
# batch 8 at that length peaks around 6.5 GB, which the old cap refused.
MEMORY_FRACTION = 0.75

# Model 2 does not fit under that cap. Qwen3-8B quantised to 4-bit is ~5.5 GB
# of weights before a single activation is allocated, so the default fraction
# leaves nothing to run in. Callers that need the whole card ask for it
# explicitly rather than the cap being raised globally, so the encoder runs
# keep their desktop-friendly headroom.
LARGE_MODEL_FRACTION = 0.92

# Below this much free VRAM, fall back to CPU rather than fight for scraps.
MIN_FREE_GB = 2.0

_state: dict | None = None


def _probe(index: int = 0) -> tuple[bool, str]:
    """Launch a real kernel to confirm this build supports this GPU."""
    try:
        cap = torch.cuda.get_device_capability(index)
        arch = f"sm_{cap[0]}{cap[1]}"
        supported = torch.cuda.get_arch_list()
        if supported and arch not in supported:
            return False, f"{arch} not in wheel arch list {supported}"
        a = torch.randn(64, 64, device=f"cuda:{index}")
        b = a @ a
        torch.cuda.synchronize(index)
        if not torch.isfinite(b).all().item():
            return False, "probe matmul produced non-finite values"
        return True, arch
    except Exception as exc:  # noqa: BLE001 - any failure means "use CPU"
        return False, f"{type(exc).__name__}: {exc}"


def get_device(prefer_gpu: bool = True, index: int = 0,
               memory_fraction: float | None = None) -> torch.device:
    """Return the device to train on, applying VRAM caps on first call.

    ``memory_fraction`` overrides the default cap and must be passed on the
    first call, since the cap is applied once and cached. Model 2 passes
    ``LARGE_MODEL_FRACTION``.
    """
    global _state
    if _state is not None:
        return _state["device"]

    if not prefer_gpu or not torch.cuda.is_available():
        _state = {"device": torch.device("cpu"), "reason": "cuda unavailable"}
        return _state["device"]

    free, total = torch.cuda.mem_get_info(index)
    if free / 1e9 < MIN_FREE_GB:
        logger.warning("only %.1f GB VRAM free; using CPU", free / 1e9)
        _state = {"device": torch.device("cpu"), "reason": "insufficient vram"}
        return _state["device"]

    ok, detail = _probe(index)
    if not ok:
        logger.warning("GPU probe failed (%s); using CPU", detail)
        _state = {"device": torch.device("cpu"), "reason": detail}
        return _state["device"]

    fraction = MEMORY_FRACTION if memory_fraction is None else memory_fraction
    torch.cuda.set_per_process_memory_fraction(fraction, index)
    dev = torch.device(f"cuda:{index}")
    _state = {
        "device": dev,
        "name": torch.cuda.get_device_name(index),
        "arch": detail,
        "cap_gb": fraction * total / 1e9,
        "free_gb": free / 1e9,
    }
    return dev


def describe() -> str:
    """One-line human-readable summary of what we ended up on."""
    if _state is None:
        return "device not yet selected"
    if _state["device"].type == "cpu":
        return f"CPU (reason: {_state.get('reason', 'requested')})"
    return (
        f"{_state['name']} [{_state['arch']}] "
        f"capped at {_state['cap_gb']:.1f} GB of {_state['free_gb']:.1f} GB free"
    )


def empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_with_oom_backoff(fn, batch_size: int, min_batch: int = 1):
    """Call ``fn(batch_size)``, halving the batch on CUDA OOM.

    Returns (result, batch_size_used). Raises if even ``min_batch`` OOMs, since
    at that point the problem is the model, not the batching.
    """
    bs = batch_size
    while True:
        try:
            return fn(bs), bs
        except torch.cuda.OutOfMemoryError:
            empty_cache()
            if bs <= min_batch:
                raise
            bs = max(min_batch, bs // 2)
            logger.warning("CUDA OOM; retrying at batch_size=%d", bs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    d = get_device()
    print("device :", d)
    print("detail :", describe())
