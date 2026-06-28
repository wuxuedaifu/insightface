from typing import Optional
import torch
import torch.nn as nn

from .vit import VisionTransformer


class AttentionWithEntropy(nn.Module):
    """Drop-in replacement for vit.Attention that stores per-patch entropy.

    During training, after the attention softmax, computes row-wise entropy
    over the attention distribution per patch and stores the batch-mean
    in self._last_entropy (shape: (B,)).  At eval time _last_entropy is
    not updated (no overhead).
    """

    def __init__(self, dim: int, num_heads: int = 8,
                 qkv_bias: bool = False,
                 qk_scale: Optional[float] = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale if qk_scale is not None else head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self._last_entropy: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(True):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(
                B, N, 3, self.num_heads, C // self.num_heads
            ).permute(2, 0, 3, 1, 4)
        with torch.cuda.amp.autocast(False):
            q, k, v = qkv[0].float(), qkv[1].float(), qkv[2].float()
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            if self.training:
                # attn: (B, heads, patches, patches)
                # row entropy per patch: (B, heads, patches)
                row_ent = -(attn * attn.clamp(min=1e-8).log()).sum(-1)
                # mean over heads and patch positions -> (B,)
                self._last_entropy = row_ent.mean(dim=(1, 2))
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        with torch.cuda.amp.autocast(True):
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class TransFaceViT(VisionTransformer):
    """VisionTransformer subclass that emits per-patch attention entropy.

    At train time:  forward(x) -> (embedding: Tensor[B,D], entropy: Tensor[B,])
    At eval time:   forward(x) -> embedding: Tensor[B,D]

    The entropy drives the FFT amplitude augmentation in train_transface.py:
    images with below-median entropy get their low-frequency amplitude mixed
    with a randomly-selected image from the same batch.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Swap last block's Attention for AttentionWithEntropy
        last_block = self.blocks[-1]
        old = last_block.attn
        new = AttentionWithEntropy(
            dim=old.qkv.in_features,
            num_heads=old.num_heads,
            qkv_bias=old.qkv.bias is not None,
            qk_scale=old.scale,
            attn_drop=old.attn_drop.p,
            proj_drop=old.proj_drop.p,
        )
        new.load_state_dict(old.state_dict())
        last_block.attn = new

    @property
    def _entropy_head(self) -> AttentionWithEntropy:
        return self.blocks[-1].attn  # type: ignore[return-value]

    def forward(self, x: torch.Tensor):
        emb = super().forward(x)
        if self.training:
            return emb, self._entropy_head._last_entropy
        return emb
