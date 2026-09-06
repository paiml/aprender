import numpy as np, sys
from gguf import GGUFReader
from gguf.quants import dequantize
M='/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf'; R=sys.argv[1]
def aprt(p):
    b=open(p,'rb').read(); return np.frombuffer(b[12:], dtype='<f4')
x=aprt(f"{R}/cpu/pos-0000/layer-26/ffn_norm.bin").astype(np.float64)
y_cpu=aprt(f"{R}/cpu/pos-0000/layer-26/ffn_swigl.bin"); y_gpu=aprt(f"{R}/gpu/pos-0000/layer-26/ffn_swigl.bin")
rd=GGUFReader(M); T={t.name:t for t in rd.tensors}
def W(name):
    t=T[name]; w=dequantize(t.data, t.tensor_type); return np.asarray(w, dtype=np.float64)
Wg=W('blk.26.ffn_gate.weight'); Wu=W('blk.26.ffn_up.weight')
def silu(v): return v/(1+np.exp(-v))
def q8(xv, blk):
    out=np.zeros_like(xv)
    for s in range(0, len(xv), blk):
        b=xv[s:s+blk]; amax=np.max(np.abs(b)); sc=amax/127.0 if amax>0 else 1.0
        out[s:s+blk]=np.round(b/sc)*sc
    return out
def y_of(xq, xo):  # int8 part + exact outlier part
    g=Wg@xq + Wg@xo; u=Wu@xq + Wu@xo; return silu(g)*u
truth=y_of(x, np.zeros_like(x))
rms=np.sqrt(np.mean(x*x))
print(f"rms={rms:.3f} max|x|={np.max(np.abs(x)):.2f}")
for tau in (4.0, 6.0, 8.0, 12.0):
    mask=np.abs(x)>tau*rms; xo=np.where(mask,x,0.0); xm=np.where(mask,0.0,x)
    y=y_of(q8(xm,256), xo)
    rel=lambda n: 100*(y[n]-truth[n])/truth[n]
    cos=float(y@truth/np.linalg.norm(y)/np.linalg.norm(truth))
    print(f"tau={tau:4.1f} outliers={int(mask.sum()):2d} dims={list(np.where(mask)[0][:6])}  n2908={y[2908]:9.2f} ({rel(2908):+.2f}%)  n7035={y[7035]:8.2f} ({rel(7035):+.2f}%)  cos_vs_truth={cos:.6f}")
yb=y_of(q8(x,256), np.zeros_like(x)); print(f"baseline per-256 (current CPU):        n2908={yb[2908]:9.2f} ({100*(yb[2908]-truth[2908])/truth[2908]:+.2f}%) cos={float(yb@truth/np.linalg.norm(yb)/np.linalg.norm(truth)):.6f}")
y32=y_of(q8(x,32), np.zeros_like(x)); print(f"per-32 (GPU-style):                     n2908={y32[2908]:9.2f} ({100*(y32[2908]-truth[2908])/truth[2908]:+.2f}%) cos={float(y32@truth/np.linalg.norm(y32)/np.linalg.norm(truth)):.6f}")
print(f"truth n2908={truth[2908]:.2f} n7035={truth[7035]:.2f}; cpu tap {y_cpu[2908]:.2f}; gpu tap {y_gpu[2908]:.2f}")
