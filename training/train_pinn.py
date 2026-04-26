"""
Offline PINN training — run BEFORE the hackathon.
Produces .pt files in backend/models/.

Usage:
    python train_pinn.py --scenario a          # single model
    python train_pinn.py --scenario all        # all 9 models (~6-9 hrs on CPU)
    python train_pinn.py --scenario b --epochs 15000
"""
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

try:
    import torch
    import torch.nn as nn
    from pinn_model import WavePINN
except ImportError:
    print("PyTorch not installed. Run: pip install torch")
    sys.exit(1)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "backend", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def gaussian(x, y, x0=0.0, y0=0.0, sigma=0.10):
    return torch.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))


def rand_pts(n, t_max=6.0):
    x = torch.FloatTensor(n).uniform_(-1, 1)
    y = torch.FloatTensor(n).uniform_(-1, 1)
    t = torch.FloatTensor(n).uniform_(0, t_max)
    return x, y, t


def train(model, loss_fn, epochs, lr, save_path, log_every=1000):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=800, min_lr=1e-5
    )
    t0 = time.time()
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        loss = loss_fn(model)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step(loss)
        if ep % log_every == 0:
            elapsed = time.time() - t0
            print(f"  ep {ep:6d} | loss {loss.item():.2e} | {elapsed:.0f}s")
    torch.save(model.state_dict(), save_path)
    print(f"  Saved → {save_path}")


# ── Scenario A: single source ─────────────────────────────────────────────────
def train_a(epochs=15000):
    c, omega = 1.0, 9.0
    k = omega / c

    def loss(m):
        xc, yc, tc = rand_pts(8000)
        res = m.pde_residual(xc, yc, tc, c)
        l_pde = (res ** 2).mean()

        xi, yi = torch.FloatTensor(2000).uniform_(-1, 1), torch.FloatTensor(2000).uniform_(-1, 1)
        ti = torch.zeros(2000)
        ri = torch.sqrt(xi ** 2 + yi ** 2) + 1e-6
        u_ic = torch.cos(k * ri) / torch.sqrt(ri)
        l_ic = ((m(xi, yi, ti) - u_ic) ** 2).mean()

        xb = torch.cat([torch.FloatTensor(250).uniform_(-1, 1), torch.full((250,), -1.0),
                        torch.FloatTensor(250).uniform_(-1, 1), torch.full((250,), 1.0)])
        yb = torch.cat([torch.full((250,), -1.0), torch.FloatTensor(250).uniform_(-1, 1),
                        torch.full((250,), 1.0),  torch.FloatTensor(250).uniform_(-1, 1)])
        tb = torch.FloatTensor(1000).uniform_(0, 6)
        l_bc = (m(xb, yb, tb) ** 2).mean()

        return l_pde + 8.0 * l_ic + 3.0 * l_bc

    model = WavePINN()
    print("Training Model A (single source)…")
    train(model, loss, epochs, 1e-3, os.path.join(MODELS_DIR, "model_a.pt"))


# ── Scenario B: two-source interference ───────────────────────────────────────
def train_b(epochs=18000):
    c, omega, sep = 1.0, 11.0, 0.38
    k = omega / c

    def loss(m):
        xc, yc, tc = rand_pts(8000)
        res = m.pde_residual(xc, yc, tc, c)
        l_pde = (res ** 2).mean()

        xi, yi = torch.FloatTensor(2000).uniform_(-1, 1), torch.FloatTensor(2000).uniform_(-1, 1)
        ti = torch.zeros(2000)
        r1 = torch.sqrt(xi ** 2 + (yi - sep) ** 2) + 1e-6
        r2 = torch.sqrt(xi ** 2 + (yi + sep) ** 2) + 1e-6
        u_ic = 0.5 * (torch.cos(k * r1) / torch.sqrt(r1) + torch.cos(k * r2) / torch.sqrt(r2))
        l_ic = ((m(xi, yi, ti) - u_ic) ** 2).mean()

        return l_pde + 10.0 * l_ic

    model = WavePINN()
    print("Training Model B (interference)…")
    train(model, loss, epochs, 1e-3, os.path.join(MODELS_DIR, "model_b.pt"))


# ── Scenario C: reflection / standing waves ───────────────────────────────────
def train_c(epochs=18000):
    c, omega = 1.0, 8.0

    def loss(m):
        xc, yc, tc = rand_pts(8000)
        res = m.pde_residual(xc, yc, tc, c)
        l_pde = (res ** 2).mean()

        xi, yi = torch.FloatTensor(2000).uniform_(-1, 1), torch.FloatTensor(2000).uniform_(-1, 1)
        ti = torch.zeros(2000)
        l_ic = ((m(xi, yi, ti) - gaussian(xi, yi, 0.0, -0.35)) ** 2).mean()

        n = 500
        walls = [
            (torch.FloatTensor(n).uniform_(-1, 1), torch.full((n,), -1.0)),
            (torch.FloatTensor(n).uniform_(-1, 1), torch.full((n,), 1.0)),
            (torch.full((n,), -1.0), torch.FloatTensor(n).uniform_(-1, 1)),
            (torch.full((n,), 1.0),  torch.FloatTensor(n).uniform_(-1, 1)),
        ]
        l_bc = sum(
            (m(xw, yw, torch.FloatTensor(n).uniform_(0, 6)) ** 2).mean()
            for xw, yw in walls
        ) / 4.0

        return l_pde + 10.0 * l_ic + 8.0 * l_bc

    model = WavePINN()
    print("Training Model C (reflection)…")
    train(model, loss, epochs, 1e-3, os.path.join(MODELS_DIR, "model_c.pt"))


# ── Scenario D: diffraction (one checkpoint per gap width) ────────────────────
def train_d(gap=0.20, epochs=16000):
    c, omega = 1.0, 11.0

    def loss(m):
        xc, yc, tc = rand_pts(8000)
        res = m.pde_residual(xc, yc, tc, c)
        l_pde = (res ** 2).mean()

        xi, yi = torch.FloatTensor(2000).uniform_(-1, 1), torch.FloatTensor(2000).uniform_(-1, 1)
        ti = torch.zeros(2000)
        l_ic = ((m(xi, yi, ti) - gaussian(xi, yi, -0.72, 0.0, sigma=0.08)) ** 2).mean()

        # Barrier BC: u=0 on barrier (x≈0, |y|>gap/2)
        xbar = torch.zeros(600)
        ybar = torch.cat([
            torch.FloatTensor(300).uniform_(-1, -gap / 2),
            torch.FloatTensor(300).uniform_(gap / 2, 1),
        ])
        tbar = torch.FloatTensor(600).uniform_(0, 6)
        l_bar = (m(xbar, ybar, tbar) ** 2).mean()

        return l_pde + 8.0 * l_ic + 6.0 * l_bar

    tag = str(int(gap * 100)).zfill(2)
    model = WavePINN()
    print(f"Training Model D gap={gap} …")
    train(model, loss, epochs, 1e-3, os.path.join(MODELS_DIR, f"model_d_{tag}.pt"))


# ── Scenario E: seismic two-layer ─────────────────────────────────────────────
def train_e(epochs=18000):
    c_up, c_dn, omega = 1.0, 0.55, 8.0

    def loss(m):
        # Lower-layer collocation
        xc, yc, tc = rand_pts(4000, t_max=6)
        yc_dn = -torch.abs(yc)
        res_dn = m.pde_residual(xc, yc_dn, tc, c_dn)
        # Upper-layer collocation
        xcu, ycu, tcu = rand_pts(4000, t_max=6)
        ycu_up = torch.abs(ycu)
        res_up = m.pde_residual(xcu, ycu_up, tcu, c_up)
        l_pde = (res_dn ** 2).mean() + (res_up ** 2).mean()

        xi, yi = torch.FloatTensor(2000).uniform_(-1, 1), torch.FloatTensor(2000).uniform_(-1, 1)
        yi_dn = -torch.abs(yi)
        ti = torch.zeros(2000)
        l_ic = ((m(xi, yi_dn, ti) - gaussian(xi, yi_dn, 0.0, -0.5, sigma=0.1)) ** 2).mean()

        return l_pde + 8.0 * l_ic

    model = WavePINN()
    print("Training Model E (seismic)…")
    train(model, loss, epochs, 1e-3, os.path.join(MODELS_DIR, "model_e.pt"))


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="all",
                    help="a | b | c | d | e | all  (d trains all 5 gap checkpoints)")
    ap.add_argument("--epochs", type=int, default=0)
    args = ap.parse_args()
    s = args.scenario.lower()

    if s in ("a", "all"):
        train_a(args.epochs or 15000)
    if s in ("b", "all"):
        train_b(args.epochs or 18000)
    if s in ("c", "all"):
        train_c(args.epochs or 18000)
    if s in ("d", "all"):
        for g in [0.05, 0.10, 0.20, 0.30, 0.50]:
            train_d(gap=g, epochs=args.epochs or 16000)
    if s in ("e", "all"):
        train_e(args.epochs or 18000)
