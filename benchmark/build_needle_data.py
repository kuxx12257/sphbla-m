"""
Build and cache the needle-in-haystack datasets used by
run_needle_haystack.py. Run this once first:

    python benchmark/build_needle_data.py

Produces two cached files in benchmark/:
  cached_needle_short.pt  -- mean length ~79 tokens
  cached_needle_long.pt   -- mean length ~219 tokens
"""
import sys, os, torch, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from needle_haystack_gen import gen_needle_v2, VOCAB_SIZE

HERE = os.path.dirname(os.path.abspath(__file__))

def build(n, seed, target_len, pad_to):
    rng = random.Random(seed)
    X, Y, L = [], [], []
    for _ in range(n):
        seq, label, actual_len = gen_needle_v2(rng, target_len, pad_to)
        X.append(seq); Y.append(label); L.append(actual_len)
    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long), torch.tensor(L)

CONDITIONS = {
    'short': dict(target_len=90, pad_to=110),
    'long':  dict(target_len=230, pad_to=260),
}

if __name__ == "__main__":
    print("Building needle-in-haystack datasets (short + long range)...")
    for name, cfg in CONDITIONS.items():
        Xtr, Ytr, Ltr = build(3000, seed=6100, target_len=cfg['target_len'], pad_to=cfg['pad_to'])
        Xva, Yva, _ = build(500, seed=6200, target_len=cfg['target_len'], pad_to=cfg['pad_to'])
        Xte, Yte, _ = build(600, seed=6300, target_len=cfg['target_len'], pad_to=cfg['pad_to'])
        torch.save({'Xtr':Xtr,'Ytr':Ytr,'Xva':Xva,'Yva':Yva,'Xte':Xte,'Yte':Yte,
                    'config':cfg, 'vocab_size':VOCAB_SIZE},
                   os.path.join(HERE, f'cached_needle_{name}.pt'))
        print(f"[{name}] Train={Xtr.shape} mean_len={Ltr.float().mean():.1f} "
              f"label_dist={[int((Ytr==c).sum()) for c in range(10)]}")
    print("Done. You can now run: python benchmark/run_needle_haystack.py --cond short --attn standard --seed 1")
