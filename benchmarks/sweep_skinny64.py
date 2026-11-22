"""Wide sweep of the 64-row tile options for one K x N shape (RTX 4050): python benchmarks/sweep_skinny64.py K N"""
import sys, itertools, torch
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
from quantforge.e2e import W4A16Gemm, pack_int4
from quantforge.autotune import device_time
G=128; M=64; K,N=int(sys.argv[1]),int(sys.argv[2])
g=torch.Generator(device="cuda").manual_seed(0)
x=(torch.rand(M,K,device="cuda",generator=g)*2-1).half()
w=pack_int4(torch.randint(0,16,(K,N),device="cuda",dtype=torch.int8,generator=g))
sc=(torch.rand(K//G,N,device="cuda",generator=g)*0.1+0.01).half()
zp=torch.randint(0,16,(K//G,N),device="cuda",dtype=torch.int8,generator=g)
out=torch.empty(M,N,device="cuda",dtype=torch.float32)
flush=torch.zeros(256<<20,dtype=torch.uint8,device="cuda")
res=[]
for bm,wm,bn,wn,bk,st in itertools.product((64,),(32,64),(64,128),(16,32,64),(32,64,128),(1,2,3)):
    if wn>bn: continue
    thr=(bm//wm)*(bn//wn)*32
    if thr>512 or thr<64: continue
    o=f"block-tile-m={bm} warp-tile-m={wm} warp-tile-n={wn} block-tile-n={bn} block-tile-k={bk} smem-stages={st}"+(" prefetch-distance=1" if st==1 else "")
    try:
        k=W4A16Gemm.build(M,K,N,group_size=G,pad=False,pipeline_opts=o,accumulate=False)
        t=device_time(lambda:(flush.max(),k.launch(x,w,sc,zp,out))) 
        ts=[]
        for _ in range(3):
            ts.append(device_time(lambda:k.launch(x,w,sc,zp,out))*1e6)
        res.append((min(ts),o))
    except Exception as e:
        pass
res.sort()
for t,o in res[:10]: print(f"{t:7.1f}  {o}")
