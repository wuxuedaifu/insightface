import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnAct(nn.Module):
    def __init__(self, in_c, out_c, kernel=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MambaBlock(nn.Module):
    """Mamba S6 selective state space block (pure PyTorch, no mamba_ssm required).

    Follows the Mamba architecture: gated x/z branches, short depthwise conv,
    input-dependent SSM parameters (B, C, dt), sequential selective scan.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dt_rank: int = None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.dt_rank = dt_rank or math.ceil(d_model / 16)

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Fixed log-space A parameter: (d_inner, d_state)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # Skip-connection scalar D
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) → (B, L, d_model)"""
        residual = x
        x = self.norm(x)
        B_sz, L, _ = x.shape

        xz = self.in_proj(x)                            # (B, L, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)                   # (B, L, d_inner) each

        # Causal depthwise conv along sequence
        x_in = x_in.transpose(1, 2)                     # (B, d_inner, L)
        x_in = self.conv1d(x_in)[:, :, :L]              # trim causal padding
        x_in = F.silu(x_in).transpose(1, 2)             # (B, L, d_inner)

        # Input-dependent SSM parameters
        x_dbl = self.x_proj(x_in)                       # (B, L, dt_rank+2*d_state)
        dt, B_ssm, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                # (B, L, d_inner)

        A = -torch.exp(self.A_log.float())               # (d_inner, d_state)
        y = self._selective_scan(x_in, dt, A, B_ssm, C, self.D)

        y = y * F.silu(z)
        return self.out_proj(y) + residual

    def _selective_scan(
        self,
        u: torch.Tensor,    # (B, L, d_inner)
        dt: torch.Tensor,   # (B, L, d_inner)
        A: torch.Tensor,    # (d_inner, d_state)
        B: torch.Tensor,    # (B, L, d_state)
        C: torch.Tensor,    # (B, L, d_state)
        D: torch.Tensor,    # (d_inner,)
    ) -> torch.Tensor:
        B_sz, L, d_in = u.shape
        d_st = A.shape[-1]

        # Discretize: Ā = exp(dt ⊗ A),  B̄u = dt ⊗ B ⊗ u
        # dA: (B, L, d_inner, d_state)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        # dBu: (B, L, d_inner, d_state)
        dBu = dt.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)

        # Sequential scan  h_t = Ā_t h_{t-1} + B̄_t u_t,  y_t = C_t h_t
        # NOTE: for production, replace with a parallel scan or mamba_ssm kernels.
        h = torch.zeros(B_sz, d_in, d_st, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dBu[:, t]               # (B, d_inner, d_state)
            yt = (h * C[:, t].unsqueeze(1)).sum(-1)     # (B, d_inner)
            ys.append(yt)

        y = torch.stack(ys, dim=1)                      # (B, L, d_inner)
        return y + u * D.unsqueeze(0).unsqueeze(0)


class MambaVit(nn.Module):
    """Hybrid CNN + Mamba backbone for face recognition at 112×112.

    Stage 1:   3 → C,  stride=2 twice  → (B, C,  56, 56)
    Stage 2:   C → 2C, stride=2 twice  → (B, 2C, 28, 28)
    Sequence:  flatten → (B, 784, 2C)
    Mamba:     N × MambaBlock(2C)
    Head:      GAP → Linear(2C→512) → BN → Linear(512→num_classes) → BN
    """

    def __init__(
        self,
        stage_dims: tuple = (256, 512),
        num_mamba_blocks: int = 24,
        d_state: int = 16,
        dt_rank: int = None,
        num_classes: int = 512,
    ):
        super().__init__()
        c1, c2 = stage_dims

        self.stage1 = nn.Sequential(
            ConvBnAct(3, c1, 3, stride=2, padding=1),
            ConvBnAct(c1, c1, 3, stride=1, padding=1),
        )
        self.stage2 = nn.Sequential(
            ConvBnAct(c1, c2, 3, stride=2, padding=1),
            ConvBnAct(c2, c2, 3, stride=1, padding=1),
        )

        self.mamba_blocks = nn.ModuleList([
            MambaBlock(d_model=c2, d_state=d_state, dt_rank=dt_rank)
            for _ in range(num_mamba_blocks)
        ])

        self.head = nn.Sequential(
            nn.Linear(c2, 512, bias=False),
            nn.BatchNorm1d(512, eps=2e-5),
            nn.Linear(512, num_classes, bias=False),
            nn.BatchNorm1d(num_classes, eps=2e-5),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)                              # (B, C,  56, 56)
        x = self.stage2(x)                              # (B, 2C, 28, 28)

        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)               # (B, 784, 2C)

        for block in self.mamba_blocks:
            x = block(x)

        x = x.mean(dim=1)                              # (B, 2C) global avg pool
        return self.head(x)                             # (B, num_classes)
