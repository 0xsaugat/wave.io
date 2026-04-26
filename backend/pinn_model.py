"""
WavePINN architecture. Requires PyTorch (optional — analytical engine is used by default).
Run training/train_pinn.py to produce .pt files in backend/models/.
"""
try:
    import torch
    import torch.nn as nn

    class WavePINN(nn.Module):
        def __init__(self, layers: int = 8, neurons: int = 128):
            super().__init__()
            net = [nn.Linear(3, neurons), nn.Tanh()]
            for _ in range(layers - 1):
                net += [nn.Linear(neurons, neurons), nn.Tanh()]
            net.append(nn.Linear(neurons, 1))
            self.net = nn.Sequential(*net)
            self._init()

        def _init(self):
            for m in self.net:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.zeros_(m.bias)

        def forward(self, x, y, t):
            return self.net(torch.stack([x, y, t], dim=-1)).squeeze(-1)

        def pde_residual(self, x, y, t, c: float = 1.0):
            x = x.requires_grad_(True)
            y = y.requires_grad_(True)
            t = t.requires_grad_(True)
            u     = self(x, y, t)
            u_t   = torch.autograd.grad(u.sum(), t,  create_graph=True)[0]
            u_tt  = torch.autograd.grad(u_t.sum(), t, create_graph=True)[0]
            u_x   = torch.autograd.grad(u.sum(), x,  create_graph=True)[0]
            u_xx  = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
            u_y   = torch.autograd.grad(u.sum(), y,  create_graph=True)[0]
            u_yy  = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
            return u_tt - c ** 2 * (u_xx + u_yy)

    TORCH_AVAILABLE = True

except ImportError:
    TORCH_AVAILABLE = False
    WavePINN = None
