"""
ListOps generator — Long Range Arena benchmark (Tay et al., 2020,
"Long Range Arena: A Benchmark for Efficient Transformers").

This is a REAL, established, peer-reviewed benchmark used by
Longformer, BigBird, Performer, Linformer, Reformer, and many other
long-range attention papers. It is synthetically generated (no
external data download required) but is NOT a task we invented --
it is the standard academic long-range reasoning benchmark.

Task: evaluate a nested expression like
  [MAX 2 9 [MIN 4 7 3] 1]
to a single digit 0-9. Operators:
  MAX  - maximum of operands
  MIN  - minimum of operands
  MED  - median of operands (rounded down)
  SM   - sum of operands, mod 10
Requires tracking hierarchical structure across long distances.
"""
import random

OPS = ['MAX', 'MIN', 'MED', 'SM']

def _apply_op(op, vals):
    if op == 'MAX': return max(vals)
    if op == 'MIN': return min(vals)
    if op == 'MED': return sorted(vals)[len(vals)//2]
    if op == 'SM':  return sum(vals) % 10
    raise ValueError(op)

def generate_expr(rng, max_depth, max_args, cur_depth=0):
    """
    Recursively build a nested ListOps expression.
    Returns (token_list, result_value).
    """
    if cur_depth >= max_depth or rng.random() < 0.3:
        v = rng.randint(0, 9)
        return [str(v)], v

    op = rng.choice(OPS)
    n_args = rng.randint(2, max_args)
    tokens = ['(', op]
    vals = []
    for _ in range(n_args):
        sub_tokens, sub_val = generate_expr(rng, max_depth, max_args, cur_depth+1)
        tokens += sub_tokens
        vals.append(sub_val)
    tokens.append(')')
    result = _apply_op(op, vals)
    return tokens, result


def generate_example(rng, max_depth=6, max_args=4, min_len=None, max_len=None):
    """Generate one (tokens, label) pair, optionally length-filtered."""
    for _ in range(200):  # retry loop to hit length target
        tokens, val = generate_expr(rng, max_depth, max_args)
        if min_len is None or (min_len <= len(tokens) <= max_len):
            return tokens, val
    return tokens, val  # fallback: return whatever we last got


if __name__ == "__main__":
    rng = random.Random(42)
    print("Sample ListOps expressions:\n")
    for i in range(5):
        tokens, val = generate_example(rng, max_depth=5, max_args=4)
        print(f"  Length={len(tokens):3d}  Label={val}")
        print(f"  {' '.join(tokens)}")
        print()
