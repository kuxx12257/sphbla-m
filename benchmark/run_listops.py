"""
Run ONE (attn_type, seed) training run with per-epoch checkpointing,
so a tool-call timeout never loses progress -- just re-invoke the
same command and it resumes from the last completed epoch.
"""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import *

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, "results.json")

EPOCHS = 12
BATCH_SIZE = 128
LR = 1e-3
D_MODEL = 64
N_LAYERS = 2

def ckpt_path(attn, seed):
    return os.path.join(HERE, f"ckpt_{attn}_seed{seed}.pt")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", required=True, choices=["standard", "sphbla_m"])
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--max-epochs-this-call", type=int, default=4)
    args = ap.parse_args()

    cache = torch.load(os.path.join(HERE, "cached_dataset.pt"))
    Xtr, Ytr = cache['Xtr'], cache['Ytr']
    Xva, Yva = cache['Xva'], cache['Yva']
    Xte, Yte = cache['Xte'], cache['Yte']
    pad_to = cache['config']['pad_to']

    cpath = ckpt_path(args.attn, args.seed)
    torch.manual_seed(args.seed)
    model = ListOpsClassifier(d_model=D_MODEL, n_layers=N_LAYERS,
                               max_len=pad_to, attn_type=args.attn)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    start_epoch = 0
    history = []

    if os.path.exists(cpath):
        ck = torch.load(cpath)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        start_epoch = ck['epoch']
        history = ck['history']
        print(f"Resuming [{args.attn} seed={args.seed}] from epoch {start_epoch}",
              flush=True)

    if start_epoch < EPOCHS:
        t0 = time.time()
        end_epoch = min(EPOCHS, start_epoch + args.max_epochs_this_call)
        for ep in range(start_epoch, end_epoch):
            tr_loss, tr_acc = run_epoch(model, Xtr, Ytr, BATCH_SIZE, opt)
            va_loss, va_acc = run_epoch(model, Xva, Yva, BATCH_SIZE)
            history.append(dict(epoch=ep, train_loss=round(tr_loss,4),
                                 train_acc=round(tr_acc,4),
                                 val_loss=round(va_loss,4), val_acc=round(va_acc,4)))
            print(f"  [{args.attn:10s} seed={args.seed}] ep{ep+1:02d}/{EPOCHS} "
                  f"train_acc={tr_acc:.3f} val_acc={va_acc:.3f} "
                  f"({time.time()-t0:.0f}s this call)", flush=True)

        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                    'epoch': end_epoch, 'history': history}, cpath)
        print(f"Checkpointed at epoch {end_epoch}/{EPOCHS}", flush=True)

    if len(history) < EPOCHS:
        print("NOT YET COMPLETE -- re-run the same command to continue.", flush=True)
        return

    te_loss, te_acc = run_epoch(model, Xte, Yte, BATCH_SIZE)
    print(f"  [{args.attn:10s} seed={args.seed}] TEST test_acc={te_acc:.4f} "
          f"test_loss={te_loss:.4f}", flush=True)

    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            results = json.load(f)
    else:
        results = {"standard": [], "sphbla_m": []}

    if not any(r['seed'] == args.seed for r in results[args.attn]):
        results[args.attn].append(dict(
            seed=args.seed, test_acc=round(te_acc,4), test_loss=round(te_loss,4),
            final_train_acc=history[-1]['train_acc'],
            final_val_acc=history[-1]['val_acc'],
            history=history
        ))
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Appended to {RESULTS_PATH}")
    else:
        print("Already recorded -- skipping append.")

if __name__ == "__main__":
    main()
