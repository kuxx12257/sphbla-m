"""
Needle-in-Haystack v2: SIMPLIFIED to isolate the specific difficulty
we want to test.

v1 problem: compound difficulty (search for rare MARK token + parse
nested expression + ignore decoys) was too hard for a 1000-example,
2-layer model -- it just memorized the training set instead of
learning a generalizable strategy (train_acc 0.42, val_acc 0.10).

v2 fix: needle is ALWAYS the first thing after CLS (no search
required). Decoys (same vocabulary/distribution, structurally
irrelevant) follow it, with variable length. This isolates exactly
one skill: correctly evaluate the relevant expression and do not
let irrelevant-but-similar trailing content corrupt the answer,
at varying distance -- the direct real-data analogue of the paper's
core synthetic claim.
"""
import random
from listops_gen import generate_example

# Vocabulary (self-contained -- no dependency on earlier design iterations)
PAD, CLS = 0, 1
DIGITS = {str(i): 2 + i for i in range(10)}
OPMAP = {'MAX': 12, 'MIN': 13, 'MED': 14, 'SM': 15}
PAREN = {'(': 16, ')': 17}
VOCAB_SIZE = 18

def tok_to_id(t):
    if t in DIGITS: return DIGITS[t]
    if t in OPMAP: return OPMAP[t]
    if t in PAREN: return PAREN[t]
    raise ValueError(t)

def gen_needle_v2(rng, target_total_len, pad_to,
                   needle_depth=3, needle_args=3,
                   needle_min=15, needle_max=45,
                   decoy_depth=3, decoy_args=3):
    needle_toks, label = generate_example(
        rng, max_depth=needle_depth, max_args=needle_args,
        min_len=needle_min, max_len=needle_max)
    needle_ids = [tok_to_id(t) for t in needle_toks]

    decoy_budget = max(0, target_total_len - 1 - len(needle_ids))
    decoys = []
    total = 0
    while total < decoy_budget:
        d_toks, _ = generate_example(rng, max_depth=decoy_depth,
                                      max_args=decoy_args,
                                      min_len=8, max_len=30)
        d_ids = [tok_to_id(t) for t in d_toks]
        if total + len(d_ids) > decoy_budget:
            break
        decoys.extend(d_ids)
        total += len(d_ids)

    seq = [CLS] + needle_ids + decoys
    actual_len = len(seq)
    if len(seq) > pad_to:
        seq = seq[:pad_to]
    seq = seq + [PAD] * (pad_to - len(seq))
    return seq, label, min(actual_len, pad_to)


if __name__ == "__main__":
    rng = random.Random(42)
    for target_len in [60, 150]:
        seq, label, actual_len = gen_needle_v2(rng, target_len, pad_to=target_len+20)
        print(f"target_len={target_len} actual_len={actual_len} label={label}")
        print(f"  first 20 ids: {seq[:20]}")
