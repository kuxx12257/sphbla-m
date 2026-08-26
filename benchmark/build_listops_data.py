"""
Build and cache the ListOps datasets used by run_listops.py.
Run this once before running any ListOps experiments:

    python benchmark/build_listops_data.py

Produces two cached files in benchmark/:
  cached_dataset.pt       -- short range (mean length ~105 tokens)
  cached_dataset_long.pt  -- long range  (mean length ~233 tokens)
"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import build_dataset

HERE = os.path.dirname(os.path.abspath(__file__))

def build_and_save(name, n_train, n_val, n_test, min_len, max_len, pad_to):
    Xtr, Ytr, Ltr = build_dataset(n_train, seed=100, min_len=min_len, max_len=max_len, pad_to=pad_to)
    Xva, Yva, _ = build_dataset(n_val, seed=200, min_len=min_len, max_len=max_len, pad_to=pad_to)
    Xte, Yte, _ = build_dataset(n_test, seed=300, min_len=min_len, max_len=max_len, pad_to=pad_to)
    cfg = dict(n_train=n_train, n_val=n_val, n_test=n_test,
               min_len=min_len, max_len=max_len, pad_to=pad_to)
    torch.save({'Xtr':Xtr,'Ytr':Ytr,'Xva':Xva,'Yva':Yva,'Xte':Xte,'Yte':Yte,'config':cfg},
               os.path.join(HERE, name))
    print(f"[{name}] Train={Xtr.shape} mean_len={Ltr.float().mean():.1f} "
          f"label_dist={[int((Ytr==c).sum()) for c in range(10)]}")

if __name__ == "__main__":
    print("Building ListOps datasets (short + long range)...")
    build_and_save('cached_dataset.pt', n_train=2500, n_val=600, n_test=1000,
                    min_len=30, max_len=160, pad_to=176)
    build_and_save('cached_dataset_long.pt', n_train=1200, n_val=300, n_test=500,
                    min_len=180, max_len=280, pad_to=288)
    print("Done. You can now run: python benchmark/run_listops.py --attn standard --seed 1")
