import numpy as np, struct, sys, math
from gguf import GGUFReader
from gguf.quants import dequantize
M='/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf'
R=sys.argv[1]  # per-op dump root (1.5B)
def aprt(p):
    b=open(p,'rb').read(); return np.frombuffer(b[12:], dtype='<f4')
x=aprt(f"{R}/cpu/pos-0000/layer-26/ffn_norm.bin").astype(np.float64)
y_cpu=aprt(f"{R}/cpu/pos-0000/layer-26/ffn_swigl.bin"); y_gpu=aprt(f"{R}/gpu/pos-0000/layer-26/ffn_swigl.bin")
rd=GGUFReader(M)
T={t.name:t for t in rd.tensors}
def W(name):
    t=T[name]; w=dequantize(t.data, t.tensor_type); return np.asarray(w, dtype=np.float64).reshape(t.shape[::-1] if w.ndim==1 else w.shape)
Wg=W('blk.26.ffn_gate.weight'); Wu=W('blk.26.ffn_up.weight')
print("shapes", Wg.shape, Wu.shape, "x", x.shape, "x argmax", int(np.argmax(np.abs(x))), float(x[np.argmax(np.abs(x))]))
def silu(v): return v/(1+np.exp(-v))
def y_of(xv):
    g=Wg@xv; u=Wu@xv; return silu(g)*u, g, u
y_f32, g, u = y_of(x)
for n in (2908, 7035, 974):
    print(f"neuron {n}: cpu_tap={y_cpu[n]:9.3f} gpu_tap={y_gpu[n]:9.3f} f64ref={y_f32[n]:9.3f} (g={g[n]:8.3f} u={u[n]:8.3f})")
# Q8 activation variants: per-block absmax/127, round-to-nearest
def q8(xv, blk):
    out=np.zeros_like(xv)
    for s in range(0, len(xv), blk):
        b=xv[s:s+blk]; amax=np.max(np.abs(b)); sc=amax/127.0 if amax>0 else 1.0
        out[s:s+blk]=np.round(b/sc)*sc
    return out
for blk in (32, 256, 1536):
    yq,_,_=y_of(q8(x,blk))
    print(f"  x quantized Q8 block={blk:4d}: n2908={yq[2908]:9.3f} n7035={yq[7035]:9.3f}")
# how much of neuron 2908's gate/up comes from dim 408?
for n in (2908,7035):
    print(f"  neuron {n}: g contribution of dim408 = {Wg[n,408]*x[408]:8.3f} of g={g[n]:8.3f}; u contribution = {Wu[n,408]*x[408]:8.3f} of u={u[n]:8.3f}; Wg[n,408]={Wg[n,408]:.5f} Wu[n,408]={Wu[n,408]:.5f}")
# per-sub-block contribution profile for neuron 2908 (which 32-blocks of x matter)
c=Wg[2908]*x; blocks=[(s//32, float(c[s:s+32].sum())) for s in range(0,1536,32)]; top=sorted(blocks,key=lambda t:-abs(t[1]))[:5]
print("  neuron 2908 gate: top 32-blocks by contribution", [(b, round(v,2)) for b,v in top])
