"""
Analytical wave physics engine.
All scenarios computed via vectorised NumPy — no training required.
"""
import numpy as np


def _damp(X, Y, margin=0.13, strength=7.0):
    dx = np.clip((np.abs(X) - (1.0 - margin)) / margin, 0.0, 1.0)
    dy = np.clip((np.abs(Y) - (1.0 - margin)) / margin, 0.0, 1.0)
    return np.exp(-strength * np.maximum(dx, dy) ** 2)


def compute_grid(scenario: str, params: dict, t: float, resolution: int = 64) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    fn = {"a": _single, "b": _two, "c": _reflect, "d": _diffract, "e": _seismic}
    return fn.get(scenario, _single)(X, Y, float(t), params)


# ── Scenario A: single point source ───────────────────────────────────────────
def _single(X, Y, t, p):
    c = float(p.get("speed", 1.0))
    w = float(p.get("omega", 9.0))
    k = w / c
    x0 = float(p.get("source_x", 0.0))
    y0 = float(p.get("source_y", 0.0))
    r = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2) + 1e-6
    u = np.cos(k * r - w * t) / np.sqrt(r)
    return (u * _damp(X, Y)).astype(np.float32)


# ── Scenario B: two-source interference ───────────────────────────────────────
def _two(X, Y, t, p):
    c   = float(p.get("speed", 1.0))
    w   = float(p.get("omega", 11.0))
    k   = w / c
    sep = float(p.get("separation", 0.38))
    r1  = np.sqrt(X ** 2 + (Y - sep) ** 2) + 1e-6
    r2  = np.sqrt(X ** 2 + (Y + sep) ** 2) + 1e-6
    u   = (np.cos(k * r1 - w * t) / np.sqrt(r1) +
           np.cos(k * r2 - w * t) / np.sqrt(r2)) * 0.5
    return (u * _damp(X, Y)).astype(np.float32)


# ── Scenario C: reflections / standing waves (method of images) ────────────────
def _reflect(X, Y, t, p):
    c  = float(p.get("speed", 1.0))
    w  = float(p.get("omega", 8.0))
    k  = w / c
    sx, sy = 0.0, -0.35
    u  = np.zeros_like(X)
    for ix in range(-2, 3):
        for iy in range(-2, 3):
            mx = 2.0 * ix + (sx if ix % 2 == 0 else -sx)
            my = 2.0 * iy + (sy if iy % 2 == 0 else -sy)
            sgn = (-1) ** (abs(ix) + abs(iy))
            r = np.sqrt((X - mx) ** 2 + (Y - my) ** 2) + 1e-6
            u += sgn * np.cos(k * r - w * t) / np.sqrt(r)
    return (u * 0.18 * _damp(X, Y, margin=0.04, strength=22.0)).astype(np.float32)


# ── Scenario D: diffraction through a gap (Huygens–Fresnel) ───────────────────
def _diffract(X, Y, t, p):
    c      = float(p.get("speed", 1.0))
    w      = float(p.get("omega", 11.0))
    k      = w / c
    gap    = float(p.get("gap_width", 0.25))
    n_sec  = max(40, int(gap * 320))
    bx     = 0.0          # barrier x position
    sx, sy = -0.72, 0.0   # incident point source

    gy    = np.linspace(-gap / 2, gap / 2, n_sec, dtype=np.float32)
    gy3   = gy[:, np.newaxis, np.newaxis]

    r_in  = np.sqrt((bx - sx) ** 2 + (gy3 - sy) ** 2)           # (N,1,1)
    r_out = np.sqrt((X[np.newaxis] - bx) ** 2 +
                    (Y[np.newaxis] - gy3) ** 2) + 1e-6            # (N,R,R)

    u_diff = np.sum(
        np.cos(k * (r_in + r_out) - w * t) / np.sqrt(r_out * r_in), axis=0
    ) / n_sec

    r_inc = np.sqrt((X - sx) ** 2 + (Y - sy) ** 2) + 1e-6
    u_inc = np.cos(k * r_inc - w * t) / np.sqrt(r_inc)

    u = np.where(X < bx - 0.02, u_inc,
                 np.where(X > bx + 0.02, u_diff * 5.5, 0.0))
    return (u * _damp(X, Y)).astype(np.float32)


# ── Scenario E: two-layer seismic refraction ──────────────────────────────────
def _seismic(X, Y, t, p):
    c_up  = float(p.get("c_upper", 1.0))
    c_dn  = float(p.get("c_lower", 0.55))
    w     = float(p.get("omega", 8.0))
    k_up  = w / c_up
    k_dn  = w / c_dn
    sy    = -0.5            # source depth
    iface = 0.0             # interface y

    # Direct wave (lower layer)
    r_dir = np.sqrt(X ** 2 + (Y - sy) ** 2) + 1e-6
    u_dn  = np.cos(k_dn * r_dir - w * t) / np.sqrt(r_dir)

    # Reflected wave in lower layer (image source)
    r_ref = np.sqrt(X ** 2 + (Y - (2 * iface - sy)) ** 2) + 1e-6
    u_ref = -0.45 * np.cos(k_dn * r_ref - w * t) / np.sqrt(r_ref)

    # Refracted wave in upper layer (vertical-ray approximation)
    r_if  = np.sqrt(X ** 2 + (iface - sy) ** 2) + 1e-6
    r_up  = np.abs(Y - iface) + 1e-6
    phase = k_dn * r_if + k_up * r_up - w * t
    u_up  = 0.65 * np.cos(phase) / np.sqrt(r_if + r_up)

    u = np.where(Y < iface, u_dn + u_ref, u_up)
    return (u * _damp(X, Y, margin=0.10)).astype(np.float32)
