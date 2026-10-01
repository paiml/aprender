#!/usr/bin/env python3
"""Write an F16 copy of a Chronos-Bolt safetensors directory (weights halve; parity cost is measured
by the spike). Usage: uv run --python 3.12 --with safetensors --with numpy tools/to_f16.py SRC DST"""
import os, shutil, sys
import numpy as np
from safetensors.numpy import load_file, save_file

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
t = load_file(os.path.join(src, "model.safetensors"))
save_file({k: v.astype(np.float16) for k, v in t.items()}, os.path.join(dst, "model.safetensors"), metadata={"format": "pt", "converted": "f32->f16 by spike 007 tools/to_f16.py"})
shutil.copy(os.path.join(src, "config.json"), os.path.join(dst, "config.json"))
n = sum(v.size for v in t.values())
print(f"{src} -> {dst}: {len(t)} tensors, {n} params, {os.path.getsize(os.path.join(dst, 'model.safetensors'))/1e6:.1f} MB f16")
