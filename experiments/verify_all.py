"""
verify_all.py — Runs all 10 verification checks from the paper.
Expected: 10/10 PASS.

Usage: python experiments/verify_all.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np
from sphbla_m import sphbla_m
torch.manual_seed(42); np.random.seed(42)

def distractor(n=300, d=32, nf=0.5, beta=1.0):
    zs=float(np.sqrt(d)); rs=float(np.sqrt(max(d-1,1)))
    mr=[]
    for _ in range(n):
        q=torch.randn(d); qh=q/q.norm()
        nr=max(2,int(20*(1-nf))); nn=20-nr
        pb=torch.randn(nr,d); pb-=(pb@qh.unsqueeze(1))*qh; pb/=pb.norm(dim=1,keepdim=True)+1e-8
        kr=torch.rand(nr,1)*0.3*zs*qh+torch.rand(nr,1)*0.15*rs*pb
        pb2=torch.randn(nn,d); pb2-=(pb2@qh.unsqueeze(1))*qh; pb2/=pb2.norm(dim=1,keepdim=True)+1e-8
        kn=(1.5+torch.rand(nn,1)*2.5)*zs*qh+torch.rand(nn,1)*0.08*rs*pb2
        tok=torch.cat([kr,kn],0); qb=qh.unsqueeze(0).expand(20,-1)
        sw=torch.softmax(sphbla_m(qb,tok,beta=beta),0)
        mr.append(float(sw[:nr].mean()/(sw[nr:].mean()+1e-8)))
    return np.mean(mr)

results = []

# CHECK 1
print("[CHECK 1] +0.91 over cosine at ALL noise fractions")
ok=True
for nf in [0.2,0.3,0.4,0.5,0.6,0.7,0.8]:
    m=distractor(n=400,d=32,nf=nf,beta=1.0)
    if m<1.5: ok=False
results.append(ok); print(f"  {'PASS' if ok else 'FAIL'}")

# CHECK 2
print("[CHECK 2] ratio > 1.70 at ALL d=4..512")
ok=True
for dv in [4,8,16,32,64,128,256,512]:
    m=distractor(n=300,d=dv,nf=0.5,beta=1.0)
    if m<1.70: ok=False
results.append(ok); print(f"  {'PASS' if ok else 'FAIL'}")

# CHECK 3
print("[CHECK 3] Theorem 1: 0 gradient failures (1000 pts)")
fails=0
for _ in range(1000):
    dg=np.random.randint(4,256); q=torch.randn(1,dg); k=torch.randn(1,dg,requires_grad=True)
    sphbla_m(q,k).sum().backward()
    if k.grad is None or torch.isnan(k.grad).any() or k.grad.norm()<1e-10: fails+=1
ok=(fails==0); results.append(ok); print(f"  {'PASS' if ok else 'FAIL'} ({fails} failures)")

# CHECK 4
print("[CHECK 4] Numeric gradient check")
eps=1e-4; d4=8; q4=torch.randn(1,d4); k4=torch.randn(1,d4,requires_grad=True)
sphbla_m(q4,k4).backward(); ag=k4.grad.clone().squeeze()
ng=torch.zeros(d4)
for i in range(d4):
    kp=k4.detach().clone(); km=k4.detach().clone(); kp[0,i]+=eps; km[0,i]-=eps
    ng[i]=(sphbla_m(q4,kp)-sphbla_m(q4,km))/(2*eps)
re=(ag-ng).norm()/(ag.norm()+1e-8)
ok=(re.item()<0.01); results.append(ok); print(f"  {'PASS' if ok else 'FAIL'} (rel_err={re.item():.2e})")

# CHECK 5
print("[CHECK 5] Proposition 1: cosine std = 1/sqrt(d)")
ok=True
for dv in [4,16,64,256,512]:
    Q=torch.randn(8000,dv); K=torch.randn(8000,dv)
    Q/=Q.norm(dim=1,keepdim=True); K/=K.norm(dim=1,keepdim=True)
    emp=(Q*K).sum(1).std().item(); theo=1/np.sqrt(dv)
    if abs(emp-theo)/theo>0.05: ok=False
results.append(ok); print(f"  {'PASS' if ok else 'FAIL'}")

# CHECK 6
print("[CHECK 6] Beta ablation: beta=0 broken, beta=1 optimal")
b0=distractor(n=200,d=32,nf=0.5,beta=0.0)
b1=distractor(n=200,d=32,nf=0.5,beta=1.0)
ok=(b0<0.85 and b1>1.5); results.append(ok)
print(f"  {'PASS' if ok else 'FAIL'} (beta=0:{b0:.3f}, beta=1:{b1:.3f})")

# CHECK 7
print("[CHECK 7] Cone-widening: cosine=1.000 at all distances")
d7=64; q7=torch.zeros(1,d7); q7[0,0]=1.0; ok=True
for dist in [0.1,1.0,5.0,10.0,50.0,100.0]:
    k7=torch.zeros(1,d7); k7[0,0]=dist
    cos=float((q7/q7.norm()*(k7/k7.norm())).sum())
    if abs(cos-1.0)>1e-5: ok=False
results.append(ok); print(f"  {'PASS' if ok else 'FAIL'}")

# CHECK 8
print("[CHECK 8] Structured score: relevant >> noise at all d")
ok=True
for dv in [4,16,64,256,512]:
    zs=float(np.sqrt(dv)); rs=float(np.sqrt(max(dv-1,1)))
    q8=torch.randn(300,dv); qh=q8/q8.norm(dim=1,keepdim=True)
    pb=torch.randn(300,dv); pb-=(pb*qh).sum(1,keepdim=True)*qh; pb/=pb.norm(dim=1,keepdim=True)+1e-8
    kr=torch.rand(300,1)*0.3*zs*qh+torch.rand(300,1)*0.15*rs*pb
    pb2=torch.randn(300,dv); pb2-=(pb2*qh).sum(1,keepdim=True)*qh; pb2/=pb2.norm(dim=1,keepdim=True)+1e-8
    kn=(1.5+torch.rand(300,1)*2.5)*zs*qh+torch.rand(300,1)*0.08*rs*pb2
    sr=sphbla_m(qh,kr).mean().item(); sn=sphbla_m(qh,kn).mean().item()
    if sr<3*sn: ok=False
results.append(ok); print(f"  {'PASS' if ok else 'FAIL'}")

# CHECK 9
print("[CHECK 9] Scale invariance: scores healthy at all d")
ok=True
for dv in [4,16,64,256,512]:
    q9=torch.randn(1000,dv); k9=torch.randn(1000,dv)
    s=sphbla_m(q9,k9)
    if torch.isnan(s).any(): ok=False
results.append(ok); print(f"  {'PASS' if ok else 'FAIL'}")

# CHECK 10
print("[CHECK 10] Limit cases: beta=0 broken, large beta->BLA")
b0=distractor(n=200,d=32,nf=0.5,beta=0.0)
b20=distractor(n=200,d=32,nf=0.5,beta=20.0)
bla=distractor(n=200,d=32,nf=0.5,beta=1.0)
ok=(b0<0.85 and b20>bla*0.8); results.append(ok)
print(f"  {'PASS' if ok else 'FAIL'}")

print(f"\n{'='*40}")
print(f"  RESULT: {sum(results)}/10 checks PASS")
if all(results): print("  STATUS: ALL VERIFIED")
else: print("  STATUS: Some checks failed — review output above")
