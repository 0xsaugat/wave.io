"""
PINN inference layer — loads pre-trained .pt models when available,
falls back to the analytical engine transparently.
"""
import os
import numpy as np

try:
    import torch
    from pinn_model import WavePINN
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

_MODELS: dict = {}
_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

_SCENARIO_FILES = {
    "a": ["model_a.pt"],
    "b": ["model_b.pt"],
    "c": ["model_c.pt"],
    "d": ["model_d_05.pt", "model_d_10.pt", "model_d_20.pt", "model_d_30.pt", "model_d_50.pt"],
    "e": ["model_e.pt"],
}

_D_GAP_VALS = [0.05, 0.10, 0.20, 0.30, 0.50]


def load_models():
    if not TORCH_AVAILABLE:
        return
    for scenario, files in _SCENARIO_FILES.items():
        loaded = []
        for f in files:
            path = os.path.join(_MODEL_DIR, f)
            if os.path.isfile(path):
                m = WavePINN()
                m.load_state_dict(torch.load(path, map_location="cpu"))
                m.eval()
                loaded.append(m)
        if loaded:
            _MODELS[scenario] = loaded


def _pinn_grid(models, params, t, resolution) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    xt = torch.tensor(X.ravel())
    yt = torch.tensor(Y.ravel())
    tt = torch.full_like(xt, t)
    with torch.no_grad():
        u = models[0](xt, yt, tt).numpy()
    return u.reshape(resolution, resolution).astype(np.float32)


def query_pinn(scenario: str, params: dict, t: float, resolution: int = 64):
    """Return PINN grid or None (caller falls back to analytical)."""
    if not TORCH_AVAILABLE or scenario not in _MODELS:
        return None

    models = _MODELS[scenario]

    if scenario == "d" and len(models) > 1:
        gap = float(params.get("gap_width", 0.2))
        # interpolate between two nearest checkpoints
        idx = np.searchsorted(_D_GAP_VALS, gap)
        idx = np.clip(idx, 1, len(_D_GAP_VALS) - 1)
        g0, g1 = _D_GAP_VALS[idx - 1], _D_GAP_VALS[idx]
        alpha = (gap - g0) / (g1 - g0 + 1e-9)
        u0 = _pinn_grid([models[idx - 1]], params, t, resolution)
        u1 = _pinn_grid([models[idx]], params, t, resolution)
        return ((1 - alpha) * u0 + alpha * u1).astype(np.float32)

    return _pinn_grid(models, params, t, resolution)
