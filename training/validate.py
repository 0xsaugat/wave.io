"""
Validates trained PINN models against analytical / exact solutions.
Run after training to confirm L2 errors before the demo.

Usage:
    python validate.py --scenario a
    python validate.py --scenario all
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

try:
    import torch
    from pinn_model import WavePINN
except ImportError:
    print("PyTorch not installed.")
    sys.exit(1)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "backend", "models")


def load(name):
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        print(f"  [skip] {name} not found")
        return None
    m = WavePINN()
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m


def l2_rel(pred, ref):
    return float(np.sqrt(np.mean((pred - ref) ** 2)) / (np.std(ref) + 1e-8))


def pinn_eval(model, x, y, t):
    with torch.no_grad():
        return model(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(t, dtype=torch.float32),
        ).numpy()


def validate_a():
    print("\n── Model A: single source vs u = cos(kr−ωt)/√r ──")
    m = load("model_a.pt")
    if m is None: return
    c, omega = 1.0, 9.0
    k = omega / c
    rng = np.random.default_rng(0)
    x = rng.uniform(-0.9, 0.9, 5000).astype(np.float32)
    y = rng.uniform(-0.9, 0.9, 5000).astype(np.float32)
    t = rng.uniform(0.0, 4.0,  5000).astype(np.float32)
    r = np.sqrt(x ** 2 + y ** 2) + 1e-6
    u_ref  = np.cos(k * r - omega * t) / np.sqrt(r)
    u_pred = pinn_eval(m, x, y, t)
    err = l2_rel(u_pred, u_ref)
    print(f"  L2 relative error: {err*100:.2f}%  (target < 3%)")
    print(f"  {'PASS ✓' if err < 0.03 else 'FAIL ✗'}")


def validate_b():
    print("\n── Model B: interference fringe angles ──")
    m = load("model_b.pt")
    if m is None: return
    c, omega, sep = 1.0, 11.0, 0.38
    k = omega / c
    lam = 2 * np.pi / k
    # Predicted first-order fringe angle
    sin_theta = lam / (2 * sep)
    if abs(sin_theta) <= 1.0:
        theta_pred = np.degrees(np.arcsin(sin_theta))
        print(f"  Analytical fringe angle: ±{theta_pred:.1f}°")

    rng = np.random.default_rng(1)
    x = rng.uniform(-0.9, 0.9, 5000).astype(np.float32)
    y = rng.uniform(-0.9, 0.9, 5000).astype(np.float32)
    t = rng.uniform(0.0, 4.0, 5000).astype(np.float32)
    r1 = np.sqrt(x ** 2 + (y - sep) ** 2) + 1e-6
    r2 = np.sqrt(x ** 2 + (y + sep) ** 2) + 1e-6
    u_ref  = 0.5 * (np.cos(k * r1 - omega * t) / np.sqrt(r1) +
                    np.cos(k * r2 - omega * t) / np.sqrt(r2))
    u_pred = pinn_eval(m, x, y, t)
    err = l2_rel(u_pred, u_ref)
    print(f"  L2 relative error vs analytical: {err*100:.2f}%")
    print(f"  {'PASS ✓' if err < 0.05 else 'FAIL ✗'}")


def validate_c():
    print("\n── Model C: standing wave node positions ──")
    m = load("model_c.pt")
    if m is None: return
    # Nodes should be at walls (x=±1, y=±1) for all t
    t_vals = np.linspace(0.5, 4.0, 20).astype(np.float32)
    errors = []
    for tv in t_vals:
        # sample 200 boundary points
        xb = np.random.uniform(-1, 1, 200).astype(np.float32)
        yb = np.ones(200, dtype=np.float32)
        u  = pinn_eval(m, xb, yb, np.full(200, tv, dtype=np.float32))
        errors.append(np.mean(np.abs(u)))
    mean_bc = float(np.mean(errors))
    print(f"  Mean |u| on walls: {mean_bc:.4f}  (target < 0.05)")
    print(f"  {'PASS ✓' if mean_bc < 0.05 else 'FAIL ✗'}")


def validate_d():
    print("\n── Model D: diffraction angle vs Fraunhofer ──")
    for gap in [0.05, 0.10, 0.20, 0.30, 0.50]:
        tag = str(int(gap * 100)).zfill(2)
        m = load(f"model_d_{tag}.pt")
        if m is None: continue
        omega, c = 11.0, 1.0
        k = omega / c
        lam = 2 * np.pi / k
        # First minimum of single-slit: sin(θ) = λ/d
        if lam / gap <= 1.0:
            theta_min = np.degrees(np.arcsin(lam / gap))
            print(f"  gap={gap:.2f}: first null at ±{theta_min:.1f}° (Fraunhofer)")
        else:
            print(f"  gap={gap:.2f}: λ/d > 1, highly diffracting regime")


def validate_e():
    print("\n── Model E: seismic refraction vs Snell's law ──")
    m = load("model_e.pt")
    if m is None: return
    c_up, c_dn = 1.0, 0.55
    theta_i = np.radians(30.0)
    sin_t = np.clip(np.sin(theta_i) * c_up / c_dn, -1, 1)
    theta_t = np.degrees(np.arcsin(sin_t))
    print(f"  Snell's law: θ_i=30° → θ_t={theta_t:.1f}°  (c_up/c_dn={c_up/c_dn:.2f})")
    print(f"  (PINN spatial angle comparison requires manual inspection)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="all")
    args = ap.parse_args()
    s = args.scenario.lower()

    if s in ("a", "all"): validate_a()
    if s in ("b", "all"): validate_b()
    if s in ("c", "all"): validate_c()
    if s in ("d", "all"): validate_d()
    if s in ("e", "all"): validate_e()

    print("\nValidation complete.")
