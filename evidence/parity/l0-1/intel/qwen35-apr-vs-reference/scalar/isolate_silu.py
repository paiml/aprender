#!/usr/bin/env python3
"""PMAT-3091 scalar: narrow conv_output_silu (the first non-bit-equal point) to one op. p4 pos 0 (zero conv state), every
DeltaNet layer, each engine's own attn_norm -> apr scalar qkv kernel (bit-exact, isolate_matmul.tsv) -> pos-0 conv -> silu
as apr computes it (libm exp) and as ggml's SSE2 ggml_v_silu computes it. Bit-equal counts vs each engine's dump."""
import subprocess, sys, numpy as np

W = "/tmp/scalar3091"
ISO = f"{W}/iso"
DIRS = {"llama": f"{W}/ref/SC-sub-p4/pos0", "apr": f"{W}/runs/scalar-dump-p4/pos0"}


def f32(path):
    return np.fromfile(path, dtype=np.float32)


def eq(a, b):
    return int((a.view(np.uint32) == b.view(np.uint32)).sum())


def fmt4(v):
    return ",".join(f"{t:.9g}" for t in v[:4])


def jobs(wts, layers):
    out = []
    for il in layers:
        _, _, qt, shape, wf = wts[f"blk.{il}.attn_qkv.weight"]
        i, o = shape.split(",")
        cw = wts[f"blk.{il}.ssm_conv1d.weight"][4]
        for eng, d in DIRS.items():
            out.append(f"convsilu0\t{wf}\t{qt}\t{i}\t{o}\t{d}/attn_norm-{il}.f32\t{cw}\t{ISO}/silu-{eng}in-{il}-aprsilu.f32\t{ISO}/silu-{eng}in-{il}-ggmlsilu.f32")
    return out


def layer_row(il):
    ld, ad = f32(f"{DIRS['llama']}/conv_output_silu-{il}.f32"), f32(f"{DIRS['apr']}/conv_output_silu-{il}.f32")
    lg, la = f32(f"{ISO}/silu-llamain-{il}-ggmlsilu.f32"), f32(f"{ISO}/silu-llamain-{il}-aprsilu.f32")
    ag, aa = f32(f"{ISO}/silu-aprin-{il}-ggmlsilu.f32"), f32(f"{ISO}/silu-aprin-{il}-aprsilu.f32")
    counts = [eq(lg, ld), eq(la, ld), eq(aa, ad), eq(ag, ad)]
    line = f"{il}\t{ld.size}\t" + "\t".join(map(str, counts)) + f"\t{fmt4(ld)}\t{fmt4(la)}\t{fmt4(lg)}"
    return line, counts + [ld.size]


def main():
    wts = {l.split("\t")[0]: l.rstrip("\n").split("\t") for l in open(f"{ISO}/weights.tsv")}
    layers = sorted({int(k.split(".")[1]) for k in wts})
    open(f"{ISO}/jobs_silu.tsv", "w").write("\n".join(jobs(wts, layers)) + "\n")
    r = subprocess.run([sys.argv[1], f"{ISO}/jobs_silu.tsv"], capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=900)
    print(f"# {r.stdout.strip()} rc={r.returncode} {r.stderr.strip()[:300]}", file=sys.stderr)
    print("layer\tn\tllama_in:ggml_silu==llama_dump\tllama_in:apr_silu==llama_dump\tapr_in:apr_silu==apr_dump\tapr_in:ggml_silu==apr_dump\tllama_first4\tapr_silu(llama_in)_first4\tggml_silu(llama_in)_first4")
    tot = np.zeros(5, dtype=np.int64)
    for il in layers:
        line, counts = layer_row(il)
        print(line)
        tot += counts
    print(f"# totals over {tot[4]} elements: ggml_silu(llama_in)==llama {tot[0]}; apr_silu(llama_in)==llama {tot[1]}; apr_silu(apr_in)==apr {tot[2]}; ggml_silu(apr_in)==apr {tot[3]}", file=sys.stderr)


if __name__ == "__main__":
    main()
