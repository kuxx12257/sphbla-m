"""
Experiment 2: Beta Ablation.
Reproduces Table 3 (beta ablation) from the paper.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np
from sphbla_m import sphbla_m
torch.manual_seed(42); np.random.seed(42)

def run_beta(beta, d=32, noise_frac=0.5, n_trials=300):
    zs=float(np.sqrt(d)); rs=float(np.sqrt(max(d-1,1)))
    mr=[]
    for _ in range(n_trials):
        q=torch.randn(d); qh=q/q.norm()
        nr=max(2,int(20*(1-noise_frac))); nn=20-nr
        pb=torch.randn(nr,d); pb-=(pb@qh.unsqueeze(1))*qh; pb/=pb.norm(dim=1,keepdim=True)+1e-8
        kr=torch.rand(nr,1)*0.3*zs*qh+torch.rand(nr,1)*0.15*rs*pb
        pb2=torch.randn(nn,d); pb2-=(pb2@qh.unsqueeze(1))*qh; pb2/=pb2.norm(dim=1,keepdim=True)+1e-8
        kn=(1.5+torch.rand(nn,1)*2.5)*zs*qh+torch.rand(nn,1)*0.08*rs*pb2
        tok=torch.cat([kr,kn],0); qb=qh.unsqueeze(0).expand(20,-1)
        sw=torch.softmax(sphbla_m(qb,tok,beta=beta),0)
        mr.append(float(sw[:nr].mean()/(sw[nr:].mean()+1e-8)))
    return np.mean(mr)

print("=== Table 3: Beta Ablation (d=32, noise=50%) ===")
print(f"{'beta':>6}  {'ratio':>8}  note")
for b in [0.0, 0.1, 0.3, 0.5, 1.0, 1.5, 3.0, 5.0]:
    m = run_beta(b)
    note = "(broken — angle only)" if b==0.0 else "(recommended)" if b==1.0 else ""
    print(f"  {b:>4.1f}  {m:>8.3f}  {note}")
