# #3953 — IQ2_S GPU GEMV

**Worker:** aprender-f5. **Branch:** `feat/3953-iq2s-gpu-gemv` off
`origin/release/0.69.1-batch-2` @ `8963f91a3`. **Status:** kernel + device A/B,
**deliberately NOT whitelisted** — the flip lands in the combined admission PR with
IQ3_XXS (#3963) and Q2_K (#3960), after #3969 (IQ2_XXS), so three rows do not rebase
over the same whitelist lines three times.

## Verdict

The kernel agrees with the CPU decoder at **every (k, n) the real model uses**, on
synthetic and real bytes, on a card proven to hold only this worktree's binary for the
whole of every run. Four planted faults, each on a different mechanism, go RED.

## 1. What this unblocks, and what it does not

`Qwen3.5-0.8B-UD-IQ2_XXS.gguf` (sha256 `a369165c…6b53e54`) is blocked on FOUR types,
measured against the whitelist at the release head:

| type | ggml | tensors | ticket |
|---|---|---|---|
| IQ2_XXS | 16 | 95 | #3950 / #3969 |
| IQ3_XXS | 18 | 24 | #3963 |
| **IQ2_S** | **22** | **5** | **this** |
| Q2_K | 10 | 3 | #3960 |

**This kernel alone does not turn that row green.** Not claimed.

## 2. Format, and the thread mapping

82 bytes / 256 elements: `d` f16 +0 · `qs[32]` +2 · `signs[32]` +34 · `qh[8]` +66 ·
`scales[8]` +74. Per (sub-block `ib`, group `l`) — one warp lane each, `ib = tid>>2`,
`l = tid&3`, identical to IQ3_S:

```
idx = qs[4ib+l] | ((qh[ib] >> 2l) & 3) << 8        10-bit, into the 1024-entry IQ2S_GRID
nib = scales[ib] low nibble for l<2, high for l>=2  the reference's db[l/2] -> selector l>>1
db  = (d * (0.5 + nib)) * 0.25                      the reference's order
```

Unlike IQ2_XXS (a u32 at a 2-mod-4 offset), every field a lane needs is a **single byte**,
so there is no unaligned load to assemble.

## 3. The grid is generated, not copied

IQ3_S's PTX carries 512 hand-copied numbers in a different base than the Rust table, and
nothing compared them until a later test did. Here the 2048 u32 words are emitted **from
`IQ2S_GRID` at generation time**, so the PTX table cannot disagree with the CPU oracle's.
Each u64 entry is laid out (lo, hi), which is exactly IQ3_S's `(g1, g2)` pair, so the
proven 4-iteration inner loop is reused unchanged. `KMASK_IQ2XS[j] == 1 << j` is pinned by
a test, so reading a sign as "bit j" cannot silently desynchronise from the oracle.

## 3b. The oracle itself, proven against gguf-py (added after the #3963 bar)

The A/B trusts `dequantize_iq2_s`. Its own comment cited 50 random blocks (PMAT-3477),
not the model the kernel's oracle is used on — a gap the IQ3_XXS bar exposed. Closed:
every value of all five real type-22 tensors, decoded by gguf-py (llama.cpp @ `df03399`)
and by `dequantize_iq2_s`, compared ELEMENT-WISE, bytes read by gguf-py's own reader:

| tensor | values | bitwise equal | max ulp | raw sha256 |
|---|---|---|---|---|
| `blk.8.ffn_down.weight` | 3,670,016 | 100% | 0 | `d688fae36905…` |
| `blk.9.ffn_down.weight` | 3,670,016 | 100% | 0 | `39cace39289d…` |
| `blk.10.ffn_down.weight` | 3,670,016 | 100% | 0 | `82fdf867e44d…` |
| `blk.17.ffn_down.weight` | 3,670,016 | 100% | 0 | `207c570c3244…` |
| `blk.21.ffn_down.weight` | 3,670,016 | 100% | 0 | `181e84ae3e4c…` |

**18,350,080 values, every one bitwise identical.** The comparison is the same script that
caught a planted decoder bug on Q2_K (#3960), so it is shown able to fail. Reproducible via
`probe_dump_iq2_s_decodes_for_gguf_py_comparison_3953` (`#[ignore]`, one-shot).

## 4. Oracle strength, before the A/B relies on it

A planted fault proves nothing if the data never exercises the mechanism it breaks. The
generator's bytes are shown to use the `qh` high bits (so dropping them shows), to carry
two different scale nibbles per sub-block (so picking the wrong one shows), and to set
sign bits — before any fault is planted.

## 5. The measurement that will open the whitelist

Admission rule (cop, from aprender-70's IQ4_XS finding): every shape the model uses, real
bytes per shape, per-row `|gpu − cpu| / Σ|w||x| ≤ 1e-5`, output pre-filled with NaN.

The census found **one** shape: all five type-22 tensors are `blk.{8,9,10,17,21}.ffn_down`,
`k=3584, n=1024` — **14 super-blocks per row**, the non-power-of-two case the first
synthetic A/B (`k=512`, 2 blocks) never reached.

| run | worst `err / Σ\|w\|\|x\|` |
|---|---|
| synthetic `k=512 n=48` | 0.000e0 |
| synthetic `k=3584 n=1024` | 0.000e0 |
| `blk.8.ffn_down.weight` | 1.331e-7 (row 202) |
| `blk.9.ffn_down.weight` | 1.543e-7 (row 176) |
| `blk.10.ffn_down.weight` | 1.903e-7 (row 408) |
| `blk.17.ffn_down.weight` | 1.107e-7 (row 126) |
| `blk.21.ffn_down.weight` | 1.303e-7 (row 437) |

Real bytes are non-zero because real scales are not 1.0 and the GPU warp-reduce and the
CPU loop round in a different order; ~50–90× under the bar.

The real-bytes test also proves the bytes **are** the tensor: every block's f16 `d` must
be finite and small. If the offset convention were wrong, GPU and CPU would read the SAME
wrong bytes and agree perfectly. The `dims` order is taken from the census (read from the
file's raw `ne` by an independent parser), not the API — `3584` and `1024` are both
multiples of 256, and `n·(k/256)·82` is symmetric in them, so neither test can recover it.

## 6. Proved able to fail

| planted fault | tensor | row | GPU | CPU | `err/Σ\|w\|\|x\|` |
|---|---|---|---|---|---|
| sign bits ignored | blk.8 | 27 | 0.23065272 | 4.14195 | 6.415e-2 |
| qh high bits dropped | blk.8 | 557 | −6.8798256 | −0.36223584 | 5.991e-2 |
| scale nibble by `l&1`, not `l>>1` | blk.8 | 29 | 2.7876983 | 0.5811428 | 3.357e-2 |
| grid lo/hi words swapped | blk.8 | 197 | −0.18543905 | −3.0357208 | 6.095e-2 |

Each RED is a GPU-vs-CPU disagreement, not a CUDA error. The fourth targets the one design
choice this kernel introduced rather than copied: the generated table's word order. Every
fault's values were **bit-identical** across every run of it, contended or not.

Guard coverage proved, not assumed: a planted `bogus.instr` reds `every_emitted_kernel_assembles`
naming `Iq2SGemv { k: 4096, n: 4096 }` at line 309 — past the ~260-line generated table, so
the table itself assembled.

## 7. Exclusivity, and the tool that proves it (#3964)

Every row above ran through `scripts/lib/gpu_exclusive_run.sh`, which waits for an empty
card **without** holding the fleet lock, takes the lock, re-checks, samples every GPU
process by full path every 0.1 s for the whole run, and refuses (exit 75) if anything
outside this worktree's `target/` appears. RTX 4090, sm_89, driver 580.119.02.

It exists because an earlier run's "card clear" check passed at the start and Ollama's
`llama-server` (1328 MiB, a daemon that never takes the lock) arrived mid-run. Its own case
table is 8/8, with three rules proven load-bearing, including one found on a real run: the
first release **falsely refused two clean runs**, because this test binary reads
`<pid>, [No data]` in nvidia-smi as it exits. A nameless sample is now ours only if that pid
was already seen under our path; an unknown one stays foreign.

The first three rows ran on an earlier revision of the script that was strictly STRICTER
(it refused every nameless sample), so its EXCLUSIVE verdicts hold under the current one.
The kernel was byte-identical across both measurement commits (`c466542e7`, `858f16260`).

## 8. A guard, at an honest scope

`no_generator_leaks_into_another_kernels_ptx_3953`: no emitted PTX may contain Rust-only
tokens, and each module must declare exactly one `.visible .entry` — its own.

Its first doc comment claimed a leak of generator source into a PTX literal "compiles".
**It does not**: every literal here is a plain `r"..."`, where any `"` ends the string, so a
quoted leak is a compile error. The guard covers what the compiler cannot — a quote-free
fragment, a PTX body duplicated inside one literal, and any literal later moved to
`r#"..."#`. Proved on plants of the first two that **compile**.

## 9. Checks

`cargo fmt --all --check` rc=0 · `cargo clippy -p aprender-serve --lib --features cuda --
-D warnings` rc=0 (0 errors). The stricter `--tests` form reports 1295 pre-existing errors
across the workspace's test targets — none in any line this branch adds (checked against
the diff ranges).

## 10. Not claimed

* That `Qwen3.5-0.8B-UD-IQ2_XXS` goes green — three other types still block it.
* That IQ2_S is fast. One warp per output row, scalar grid lookups, no shared-memory staging.
* An end-to-end generation comparison on the real model.
* Coverage of the post-lock re-check in `gpu_exclusive_run.sh` by a dedicated row: it is
  exercised only as a backstop (a mutant bypassing the pre-lock wait was still refused by it).
