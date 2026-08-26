"""
Experiment 1: Cone-Widening Distractor Task.
Reproduces Tables 1 and 2 from the paper.

Usage:
    python experiments/exp1_distractor.py

Runtime: ~30 min CPU, ~5 min GPU.
Expected: SphBLA-M ratio ≈ 1.748 at ALL settings.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from sphbla_m import sphbla_m

torch.manual_seed(42)
np.random.seed(42)


def run_trial(d=32, seq_len=20, noise_frac=0.5, beta=1.0, n_trials=500):
    """
    One cell of the distractor experiment.
    Returns (dot_ratio, cosine_ratio, bla_ratio, sphblam_ratio).
    """
    zs = float(np.sqrt(d))
    rs = float(np.sqrt(max(d - 1, 1)))
    r_max = 0.6 * zs
    z_max = 0.8 * zs
    dr, cr, br, mr = [], [], [], []

    for _ in range(n_trials):
        q   = torch.randn(d)
        q_hat = q / q.norm()
        n_rel = max(2, int(seq_len * (1 - noise_frac)))
        n_noi = seq_len - n_rel

        # Relevant tokens: small z, small r
        perp = torch.randn(n_rel, d)
        perp -= (perp @ q_hat.unsqueeze(1)) * q_hat
        perp /= perp.norm(dim=1, keepdim=True) + 1e-8
        k_rel = (torch.rand(n_rel, 1) * 0.3 * zs * q_hat
                 + torch.rand(n_rel, 1) * 0.15 * rs * perp)

        # Noise tokens: aligned with q but axially distant
        perp2 = torch.randn(n_noi, d)
        perp2 -= (perp2 @ q_hat.unsqueeze(1)) * q_hat
        perp2 /= perp2.norm(dim=1, keepdim=True) + 1e-8
        k_noi = ((1.5 + torch.rand(n_noi, 1) * 2.5) * zs * q_hat
                 + torch.rand(n_noi, 1) * 0.08 * rs * perp2)

        tokens = torch.cat([k_rel, k_noi], dim=0)
        q_exp  = q_hat.unsqueeze(0).expand(seq_len, -1)

        # Dot-product attention
        dw = torch.softmax(
            (tokens @ q.unsqueeze(1)).squeeze() / zs, dim=0)

        # Cosine attention
        k_hat = tokens / (tokens.norm(dim=1, keepdim=True) + 1e-8)
        cw = torch.softmax(
            (k_hat @ q_hat.unsqueeze(1)).squeeze(), dim=0)

        # Euclidean BLA
        zt = (tokens * q_hat).sum(1)
        kp = tokens - zt.unsqueeze(1) * q_hat
        rt = kp.norm(1)
        bw = torch.softmax(
            torch.exp(-(rt**2) / r_max**2 - (zt**2) / z_max**2), dim=0)

        # SphBLA-M
        sw = torch.softmax(
            sphbla_m(q_exp, tokens, beta=beta), dim=0)

        eps = 1e-8
        dr.append(float(dw[:n_rel].mean() / (dw[n_rel:].mean() + eps)))
        cr.append(float(cw[:n_rel].mean() / (cw[n_rel:].mean() + eps)))
        br.append(float(bw[:n_rel].mean() / (bw[n_rel:].mean() + eps)))
        mr.append(float(sw[:n_rel].mean() / (sw[n_rel:].mean() + eps)))

    return np.mean(dr), np.mean(cr), np.mean(br), np.mean(mr)


def main():
    print("=" * 60)
    print("  Experiment 1: Cone-Widening Distractor Task")
    print("=" * 60)

    # Table 1: Noise fraction sweep
    print("\n=== Table 1: Noise Fraction Sweep (d=32, beta=1.0) ===")
    print(f"{'Noise%':>7}  {'Dot':>7}  {'Cosine':>7}  "
          f"{'BLA':>7}  {'SphBLA-M':>9}")
    print("-" * 50)
    for nf in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        d, c, b, m = run_trial(d=32, noise_frac=nf, beta=1.0, n_trials=500)
        print(f"  {nf*100:>4.0f}%  {d:>7.3f}  {c:>7.3f}  "
              f"{b:>7.3f}  {m:>9.3f}")

    # Table 2: Dimension sweep
    print("\n=== Table 2: Dimension Sweep (noise=50%, beta=1.0) ===")
    print(f"{'d':>5}  {'Dot':>7}  {'Cosine':>7}  "
          f"{'BLA':>7}  {'SphBLA-M':>9}")
    print("-" * 45)
    for dv in [4, 8, 16, 32, 64, 128, 256, 512]:
        d, c, b, m = run_trial(d=dv, noise_frac=0.5, beta=1.0, n_trials=400)
        print(f"  {dv:>3d}  {d:>7.3f}  {c:>7.3f}  "
              f"{b:>7.3f}  {m:>9.3f}")

    print("\nDone. SphBLA-M should read ~1.748 at every cell.")


if __name__ == "__main__":
    main()
