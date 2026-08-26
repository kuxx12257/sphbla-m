# SphBLA-M: Spherical Biconvex Lens Attention with Magnitude Awareness


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22117420.svg)](https://doi.org/10.5281/zenodo.22117420)

> **Addressing the Cone-Widening Problem in Transformer Attention**

**Status:** Synthetic geometry results are strong and fully verified (10/10 checks
pass — see below). Real-data transfer is actively being investigated; the
honest results are reported below, not hidden. Feedback, ideas, and PRs
testing this on other real tasks are very welcome.

## The Problem

Cosine attention returns **score = 1.000** for aligned tokens at
distances 0.1 **and** 100 — it cannot tell them apart. This
"cone-widening" failure worsens at scale: at d=4096 (GPT-4 scale),
cosine similarity's standard deviation is **0.016** — near-random.

## The Fix

SphBLA-M combines:
- **Angular signal** (unit-sphere direction, asymmetric front/behind)
- **Distance signal** (`β·z̃²`, normalised axial depth)

into a **biconvex lens** that decays with both angle and distance.

## Synthetic Results (Fully Verified)

| Method       | Ratio ρ (noise=50%, d=32) |
|--------------|---------------------------|
| Dot-product  | 0.000                     |
| Cosine       | 0.837                     |
| BLA          | 1.000                     |
| **SphBLA-M** | **1.748**                 |

Identical across noise fractions 20%–80% and all dimensions
d ∈ {4,8,16,32,64,128,256,512}. Run `python experiments/verify_all.py`
to reproduce all 10 verification checks yourself.

## Real-Data Evaluation (Honest Results)

We tested whether the synthetic advantage transfers to real
discrete-token tasks: [ListOps](https://arxiv.org/abs/2011.04006) (Long
Range Arena) and a needle-in-haystack task we built to test relevance
discrimination directly, using real tokens instead of synthetic vectors.

| Task (mean length) | Seeds | Standard attention | SphBLA-M |
|---|---|---|---|
| ListOps (105 tok) | 7 | 0.234 ± 0.022 | 0.231 ± 0.012 |
| ListOps (233 tok) | 3 | 0.198 ± 0.002 | 0.153 ± 0.040 |
| Needle-haystack (79 tok) | 3 | 0.266 ± 0.025 | 0.265 ± 0.010 |

**None of these differences are statistically significant** at
conventional thresholds given the sample sizes (p = 0.73, 0.13, 0.95
respectively). The 233-token ListOps result is directionally
consistent across all 3 seeds (standard ahead) but underpowered.

**Our current interpretation:** ListOps requires exact structural
retrieval — a token far from the query (e.g., the operator
determining the whole expression's value) can be decisively relevant
regardless of distance. That's close to the opposite of what
SphBLA-M's distance penalty assumes. We don't think this contradicts
the geometric claim about the score function itself (which is
task-independent), but it does mean the practical benefit is, at
minimum, not yet demonstrated for tasks like this — and might not
exist for this task class at all. Tasks where relevance genuinely
correlates with distance (retrieval, local coreference) are the
setting we think is most promising to test next. **If you have ideas
on why the synthetic advantage doesn't transfer, or want to try this
on a different real task, please open an issue or PR.**

## Installation

```bash
git clone https://github.com/kuxx12257/sphbla-m
cd sphbla-m
pip install -r requirements.txt
```

Python 3.8+ | PyTorch 1.12+ | NumPy 1.21+

## Reproduce the Synthetic Results

```bash
python experiments/exp1_distractor.py    # Tables 1-2 (noise fraction + dimension sweep)
python experiments/exp2_ablation.py      # Table 3 (beta ablation)
python experiments/verify_all.py         # All 10 verification checks
```

Runtime: ~30 min CPU, ~5 min GPU.

## Reproduce the Real-Data Results

```bash
# Build datasets first (one-time, ~10 seconds)
python benchmark/build_listops_data.py
python benchmark/build_needle_data.py

# Then run any (task, attention-type, seed) combination:
python benchmark/run_listops.py --attn standard --seed 1
python benchmark/run_listops.py --attn sphbla_m --seed 1
python benchmark/run_needle_haystack.py --cond short --attn standard --seed 1
python benchmark/run_needle_haystack.py --cond long --attn sphbla_m --seed 1
```

Each run checkpoints after every epoch and appends its result to
`benchmark/results/*.json` — safe to interrupt and re-run the same
command to resume. The `benchmark/results/` folder already contains
the exact numbers reported above and in the paper.

Runtime per run: SHORT tasks ~1-3 min CPU; LONG tasks ~3-8 min CPU
(SphBLA-M is slower due to its O(T²) angular computation — see
`sphbla_m/attention.py` for a vectorised implementation, ~2-4x
overhead instead of the ~100x of a naive loop).

## Quick Usage

```python
from sphbla_m import sphbla_m, sphbla_m_attention
import torch

# Single pair similarity score
q = torch.randn(1, 64)
k = torch.randn(1, 64)
score = sphbla_m(q, k)   # shape [1], value in (0, 1]

# Drop-in attention module
Q = torch.randn(2, 16, 64)   # [batch, seq_len, d_model]
K = torch.randn(2, 16, 64)
V = torch.randn(2, 16, 64)
output, weights = sphbla_m_attention(Q, K, V)
```

## Repository Structure

```
sphbla-m/
├── sphbla_m/
│   ├── __init__.py
│   └── attention.py               # sphbla_m() + sphbla_m_attention()
├── experiments/
│   ├── exp1_distractor.py         # Synthetic: Tables 1-2
│   ├── exp2_ablation.py           # Synthetic: beta ablation
│   └── verify_all.py              # All 10 synthetic verification checks
├── benchmark/
│   ├── listops_gen.py             # LRA ListOps generator (independently verified)
│   ├── needle_haystack_gen.py     # Needle-in-haystack task generator
│   ├── harness.py                 # Shared model + training code
│   ├── build_listops_data.py      # One-time dataset builder
│   ├── build_needle_data.py       # One-time dataset builder
│   ├── run_listops.py             # Run one (attn, seed) ListOps experiment
│   ├── run_needle_haystack.py     # Run one (cond, attn, seed) needle experiment
│   └── results/                   # Committed: the actual verified numbers
├── requirements.txt
└── README.md
```

## Citation

```bibtex
@article{kushagra yadav 2026sphblam,
  title   = {Spherical Biconvex Lens Attention with Magnitude Awareness:
             Addressing the Cone-Widening Problem in Transformer Attention},
  author  = {Kushagra Yadav},
  year    = {2026},
  url     = {https://github.com/your-username/sphbla-m}

}
```
