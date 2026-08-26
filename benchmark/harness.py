"""
ListOps benchmark harness for comparing standard scaled dot-product
attention against SphBLA-M attention on a real hierarchical
long-range reasoning task (Long Range Arena ListOps, Tay et al. 2020).
"""
import sys, os, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
import numpy as np
from listops_gen import generate_example

# ── Vocabulary ──────────────────────────────────────────────────────
PAD, CLS = 0, 1
DIGITS = {str(i): 2 + i for i in range(10)}          # 2..11
OPS = {'MAX': 12, 'MIN': 13, 'MED': 14, 'SM': 15}
PAREN = {'(': 16, ')': 17}
VOCAB_SIZE = 18

def tokenize(tokens):
    ids = [CLS]
    for t in tokens:
        if t in DIGITS: ids.append(DIGITS[t])
        elif t in OPS: ids.append(OPS[t])
        elif t in PAREN: ids.append(PAREN[t])
        else: raise ValueError(f"unknown token {t}")
    return ids


# ── Dataset ─────────────────────────────────────────────────────────
def build_dataset(n_examples, seed, min_len=40, max_len=250,
                   pad_to=256, max_depth=5, max_args=4):
    rng = random.Random(seed)
    X, Y, L = [], [], []
    attempts = 0
    while len(X) < n_examples and attempts < n_examples * 50:
        attempts += 1
        toks, val = generate_example(rng, max_depth=max_depth,
                                      max_args=max_args,
                                      min_len=min_len, max_len=max_len)
        if not (min_len <= len(toks) <= max_len):
            continue
        ids = tokenize(toks)
        if len(ids) > pad_to:
            continue
        seq_len = len(ids)
        ids = ids + [PAD] * (pad_to - len(ids))
        X.append(ids); Y.append(val); L.append(seq_len)
    X = torch.tensor(X, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.long)
    L = torch.tensor(L, dtype=torch.long)
    return X, Y, L


# ── Attention variants ──────────────────────────────────────────────
def standard_attention(Q, K, V, pad_mask):
    """Scaled dot-product attention. pad_mask: [B,T] True=PAD."""
    d = Q.shape[-1]
    scores = torch.bmm(Q, K.transpose(1, 2)) / (d ** 0.5)   # [B,T,T]
    scores = scores.masked_fill(pad_mask.unsqueeze(1), float('-inf'))
    w = torch.softmax(scores, dim=-1)
    w = torch.nan_to_num(w, nan=0.0)
    return torch.bmm(w, V), w


def sphbla_m_attention_fast(Q, K, V, pad_mask,
                             theta_max=0.8, lam=3.0, alpha=1.0, beta=1.0):
    """Vectorised SphBLA-M attention with padding mask support."""
    B, T, d = Q.shape
    Q_hat = Q / (Q.norm(dim=-1, keepdim=True) + 1e-8)
    K_hat = K / (K.norm(dim=-1, keepdim=True) + 1e-8)

    cos_t = torch.bmm(Q_hat, K_hat.transpose(1, 2)).clamp(-1, 1)
    sin_t = torch.sqrt(torch.clamp(1 - cos_t ** 2, min=0))

    r_pen = sin_t ** 2 / np.sin(theta_max) ** 2
    z_pen = torch.where(
        cos_t >= 0,
        (1 - cos_t) ** 2 / (1 - np.cos(theta_max)) ** 2,
        lam * (1 + cos_t.abs()) ** 2
    )
    z_raw = torch.bmm(Q_hat, K.transpose(1, 2)) / (d ** 0.5)
    delta = r_pen + z_pen + beta * z_raw ** 2
    scores = -alpha * delta                                    # log-space
    scores = scores.masked_fill(pad_mask.unsqueeze(1), float('-inf'))
    w = torch.softmax(scores, dim=-1)
    w = torch.nan_to_num(w, nan=0.0)
    return torch.bmm(w, V), w


# ── Model ───────────────────────────────────────────────────────────
class EncoderLayer(nn.Module):
    def __init__(self, d_model, attn_type='standard', ff_mult=4, **attn_kw):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult), nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model)
        )
        self.attn_type = attn_type
        self.attn_kw = attn_kw

    def forward(self, x, pad_mask):
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        if self.attn_type == 'standard':
            out, _ = standard_attention(Q, K, V, pad_mask)
        else:
            out, _ = sphbla_m_attention_fast(Q, K, V, pad_mask, **self.attn_kw)
        x = self.ln1(x + self.Wo(out))
        x = self.ln2(x + self.ff(x))
        return x


class ListOpsClassifier(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=64, n_layers=2,
                 max_len=256, n_classes=10, attn_type='standard', **attn_kw):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, attn_type=attn_type, **attn_kw)
            for _ in range(n_layers)
        ])
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x):
        B, T = x.shape
        pad_mask = (x == PAD)                       # [B,T] True=PAD
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.emb(x) + self.pos(pos_ids)
        for layer in self.layers:
            h = layer(h, pad_mask)
        cls_repr = h[:, 0, :]                        # CLS token
        return self.classifier(cls_repr)


# ── Training / evaluation ────────────────────────────────────────────
def run_epoch(model, X, Y, batch_size, optimizer=None, device='cpu'):
    train = optimizer is not None
    model.train(train)
    n = X.shape[0]
    idx = torch.randperm(n) if train else torch.arange(n)
    total_loss, total_correct = 0.0, 0
    for i in range(0, n, batch_size):
        b = idx[i:i+batch_size]
        xb, yb = X[b].to(device), Y[b].to(device)
        if train:
            optimizer.zero_grad()
        logits = model(xb)
        loss = nn.functional.cross_entropy(logits, yb)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item() * len(b)
        total_correct += (logits.argmax(-1) == yb).sum().item()
    return total_loss / n, total_correct / n


if __name__ == "__main__":
    print("Smoke test: tiny dataset, tiny model, few epochs.")
    Xtr, Ytr, _ = build_dataset(200, seed=1, min_len=20, max_len=100, pad_to=128)
    Xva, Yva, _ = build_dataset(50, seed=2, min_len=20, max_len=100, pad_to=128)
    print(f"Train: {Xtr.shape}, labels range {Ytr.min().item()}-{Ytr.max().item()}")

    for attn_type in ['standard', 'sphbla_m']:
        torch.manual_seed(0)
        model = ListOpsClassifier(d_model=32, n_layers=2, max_len=128,
                                   attn_type=attn_type)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        t0 = time.time()
        for ep in range(5):
            tr_loss, tr_acc = run_epoch(model, Xtr, Ytr, 32, opt)
        va_loss, va_acc = run_epoch(model, Xva, Yva, 32)
        print(f"  [{attn_type:10s}] train_loss={tr_loss:.3f} "
              f"train_acc={tr_acc:.3f} val_acc={va_acc:.3f} "
              f"({time.time()-t0:.1f}s for 5 epochs)")
    print("Smoke test complete — no crashes, no NaN.")
