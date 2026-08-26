"""
SphBLA-M: Spherical Biconvex Lens Attention with Magnitude Awareness.

Reference:
  "Addressing the Cone-Widening Problem in Transformer Attention"
  https://arxiv.org/abs/2507.XXXXX
"""
import torch
import numpy as np


def sphbla_m(q, k, theta_max=0.8, lam=3.0, alpha=1.0, beta=1.0):
    """
    SphBLA-M similarity score (Equations 5-9 in the paper).

    Computes the biconvex-lens similarity between query q and key k.
    Drop-in replacement for q·k/sqrt(d) in standard attention.

    Args:
        q, k      : torch.Tensor of shape [N, d] or [d]
        theta_max : float  — angular lens width (rad). Default 0.8.
        lam       : float  — behind-query asymmetry factor. Default 3.0.
        alpha     : float  — sharpness (temperature). Default 1.0.
        beta      : float  — distance signal weight. Default 1.0.
                    beta=0 → pure SphBLA (angle only, broken on this task)
                    beta=1 → recommended default
                    beta→∞ → approaches Euclidean BLA

    Returns:
        score : torch.Tensor of shape [N], values in (0, 1]
    """
    if q.dim() == 1:
        q = q.unsqueeze(0)
    if k.dim() == 1:
        k = k.unsqueeze(0)
    d = q.shape[-1]

    # ── Angular signal (unit-sphere) ─────────────────────────────
    q_hat = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    k_hat = k / (k.norm(dim=-1, keepdim=True) + 1e-8)

    cos_t = (k_hat * q_hat).sum(dim=-1).clamp(-1, 1)   # cos θ
    sin_t = torch.sqrt(torch.clamp(1 - cos_t**2, min=0))  # sin θ

    # Lateral angular penalty  (Eq. 5)
    r_pen = sin_t**2 / np.sin(theta_max)**2

    # Asymmetric axial penalty  (Eq. 6)
    z_pen = torch.where(
        cos_t >= 0,
        (1 - cos_t)**2 / (1 - np.cos(theta_max))**2,   # in front
        lam * (1 + torch.abs(cos_t))**2                 # behind
    )

    # ── Distance signal (raw, un-normalised k) ───────────────────
    # Eq. 7: z̃(k) = k·q̂ / sqrt(d)
    z_raw = (k * q_hat).sum(dim=-1) / float(np.sqrt(d))

    # ── Combined delta and score  (Eq. 8-9) ─────────────────────
    delta = r_pen + z_pen + beta * (z_raw ** 2)
    return torch.exp(-alpha * delta)


def sphbla_m_attention(Q, K, V,
                        theta_max=0.8, lam=3.0,
                        alpha=1.0, beta=1.0,
                        vectorised=False):
    """
    Full SphBLA-M attention module.

    Drop-in replacement for softmax(QK^T/sqrt(d))V in Eq. (1).

    Args:
        Q, K, V    : torch.Tensor of shape [batch, seq_len, d_model]
        theta_max, lam, alpha, beta : SphBLA-M parameters (see sphbla_m)
        vectorised : bool — use faster batched implementation (default False)

    Returns:
        output  : torch.Tensor [batch, seq_len, d_model]
        weights : torch.Tensor [batch, seq_len, seq_len]
    """
    B, T, d = Q.shape

    if vectorised:
        return _sphbla_m_attention_fast(Q, K, V,
                                         theta_max, lam, alpha, beta)

    # Naive loop (clear and correct — matches paper exactly)
    S = torch.zeros(B, T, T, device=Q.device, dtype=Q.dtype)
    for b in range(B):
        for i in range(T):
            q_i = Q[b, i:i+1].expand(T, -1)
            S[b, i] = sphbla_m(q_i, K[b],
                                theta_max, lam, alpha, beta)

    weights = torch.softmax(S, dim=-1)
    output  = weights @ V
    return output, weights


def _sphbla_m_attention_fast(Q, K, V,
                               theta_max=0.8, lam=3.0,
                               alpha=1.0, beta=1.0):
    """Vectorised SphBLA-M attention (~10-50x faster than loop)."""
    B, T, d = Q.shape

    Q_hat = Q / (Q.norm(dim=-1, keepdim=True) + 1e-8)
    K_hat = K / (K.norm(dim=-1, keepdim=True) + 1e-8)

    # Pairwise cos(theta): [B, T_q, T_k]
    cos_t = torch.bmm(Q_hat, K_hat.transpose(1, 2)).clamp(-1, 1)
    sin_t = torch.sqrt(torch.clamp(1 - cos_t**2, min=0))

    r_pen = sin_t**2 / np.sin(theta_max)**2
    z_pen = torch.where(
        cos_t >= 0,
        (1 - cos_t)**2 / (1 - np.cos(theta_max))**2,
        lam * (1 + cos_t.abs())**2
    )

    # z_raw[b, i, j] = Q_hat[b, i] · K[b, j] / sqrt(d)
    z_raw = torch.bmm(Q_hat, K.transpose(1, 2)) / (d ** 0.5)

    delta   = r_pen + z_pen + beta * z_raw**2
    weights = torch.softmax(torch.exp(-alpha * delta), dim=-1)
    return weights @ V, weights
