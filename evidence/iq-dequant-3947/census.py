import sys, collections, hashlib, os
from gguf import GGUFReader
for p in sys.argv[1:]:
    r = GGUFReader(p)
    c = collections.Counter(); ex = {}
    for t in r.tensors:
        n = t.tensor_type.name; c[n] += 1; ex.setdefault(n, []).append(t.name)
    print(f"== {os.path.basename(p)}  tensors={len(r.tensors)}  size={os.path.getsize(p)}")
    for n,k in sorted(c.items(), key=lambda x:-x[1]):
        print(f"   {n:10s} {k:4d}   e.g. {', '.join(ex[n][:3])}")
