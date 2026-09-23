# #3960 — Q2_K GPU GEMV

**Worker:** aprender-f5. **Branch:** `feat/3960-q2k-gpu-gemv` off
`origin/release/0.69.1-batch-2` @ `3abc84855`. **Status:** CPU decoder proven against
gguf-py, kernel + device A/B, **deliberately NOT whitelisted** — the flip lands in the
combined admission PR with IQ2_S (#3953) and IQ3_XXS (#3963).

## Verdict

The Q2_K CPU decoder is **bitwise identical to llama.cpp's gguf-py on every one of the
11,010,048 values** of the model's three Q2_K tensors. The GPU kernel agrees with that
decoder at every (k, n) the model uses, on synthetic and real bytes, on a card proven to
hold only this worktree's binary. Four planted faults, each on a different mechanism, go RED.

## 1. What this unblocks, and what it does not

`Qwen3.5-0.8B-UD-IQ2_XXS.gguf` (sha256 `a369165c…6b53e54`) was blocked on four types.
Q2_K is three tensors of them. **This kernel alone does not turn that row green**; not claimed.

## 2. The oracle had to be proven first — this decoder has been wrong before

Every device A/B trusts its CPU decoder. If the decoder is wrong, the kernel is proven to
match a wrong answer and nothing downstream can tell. `dequantize_q2_k`'s own comment
records that a prior scheme "applied the wrong scale to the wrong 2-bit lanes → corrupt
weights → broken Q2_K inference". So before any kernel:

| tensor | values | bitwise equal to gguf-py | max ulp |
|---|---|---|---|
| `blk.0.ffn_down.weight` | 3,670,016 | 3,670,016 (100%) | 0 |
| `blk.4.ffn_down.weight` | 3,670,016 | 3,670,016 (100%) | 0 |
| `blk.5.ffn_down.weight` | 3,670,016 | 3,670,016 (100%) | 0 |

gguf-py is llama.cpp's own numpy dequantizer (`/mnt/nvme-raid0/llama.cpp-master` @ `df03399`).
Both sides read the bytes from **gguf-py's reader**, so no offset convention could enter.
Compared element-wise, never by summary statistic.

**Proved able to fail**: planting the decoder's own historical bug — the scale for one
16-lane half applied to the other — made the comparison RED on all three tensors, 51% of
values still equal (the untouched half) and the rest wrong, located to the lane (e.g.
super-block 0 lane 129: gguf-py −0.0176 vs planted −0.176, a 10× scale error).

A hermetic golden keeps the proof without Python: 12 real super-blocks (first, second,
middle, last of each tensor) and gguf-py's values for them, in `quantize/fixtures/`
(not gitignored — checked).

## 3. The kernel's index math, proven before the PTX existed

One warp per output row; lane L owns outputs 8L..8L+7 of each super-block. From ggml's
order (o = 128g + 32s + 16h + i; quant at qs[32g+16h+i] >> 2s; scale at scales[8g+2s+h]),
8 consecutive outputs never cross a 16-wide half, so per lane per block:
`g = L>>4, s = (L&15)>>2, h = (L&3)>>1` — **one scale byte, one shift, eight contiguous
quant bytes**. `q2_k_block_by_lanes` reproduces this in Rust and matches the decoder
**bitwise** on all 12 golden blocks. A wrong half selector (`h = L&1`) reds it.

**An equivalent mutant, not a gap.** Replacing `dl*q - ml` with a fused multiply-add
SURVIVES — because it cannot differ: `d` is f16 (11 significant bits), times a 4-bit
nibble and a 2-bit `q`, is at most 17 bits, exact in f32's 24. `dl*q` is exact, so there
is one rounding either way. The kernel's `.rn` qualifiers are therefore defensive only.

## 4. Byte loads, no alignment assumption

The quant run is 4-aligned relative to the tensor base (84 = 4·21), but the tensor's
offset in the real inference path is not something the kernel can see. It loads each
quant byte as u8 rather than relying on it.

## 5. The measurement that will open the whitelist

One shape: all three Q2_K tensors are `blk.{0,4,5}.ffn_down`, `k=3584, n=1024` (14
super-blocks/row). Per-row `|gpu − cpu| / Σ|w||x| ≤ 1e-5`, output pre-filled with NaN, CPU
dot in f64 so any error measured is the GPU's.

| run | worst `err / Σ\|w\|\|x\|` |
|---|---|
| synthetic `k=512 n=48` | 0.000e0 |
| synthetic `k=3584 n=1024` | 3.994e-9 |
| `blk.0.ffn_down.weight` | 4.301e-9 (row 124) |
| `blk.4.ffn_down.weight` | 0.000e0 |
| `blk.5.ffn_down.weight` | 4.342e-9 (row 956) |

~2000× under the bar. The real-bytes test first proves the bytes ARE the tensor (every
block's `d` and `dmin` finite and small); the `dims` order comes from the census, not the
API, since 3584 and 1024 are both multiples of 256.

## 6. Proved able to fail

| planted fault | mechanism | row | GPU | CPU | `err/Σ\|w\|\|x\|` |
|---|---|---|---|---|---|
| scale/min nibbles swapped | the affine packing | 8 | 2.395977 | 0.40855265 | 4.840e-2 |
| half selector `L&1`, not `(L&3)>>1` | **the decoder's own historical bug class** | 8 | −3.4083943 | 0.40855265 | 9.296e-2 |
| affine min dropped | the `− dmin·hi(sc)` term | 14 | −1.2922368 | 0.42875290 | 2.312e-2 |
| shift from `h`, not `s` | the 2-bit lane select | 568 | −11.974408 | 0.81825209 | 1.206e-1 |

All on `blk.0.ffn_down.weight` (real bytes), each a GPU-vs-CPU disagreement rather than a
CUDA error, three on different rows. Guard coverage proved: a planted `bogus.instr` reds
`every_emitted_kernel_assembles` naming `Q2KGemv { k: 4096, n: 4096 }` at line 70.

## 7. Exclusivity

Every row above ran through `scripts/lib/gpu_exclusive_run.sh` and was **EXCLUSIVE** —
every sample of every run under this worktree's `target/` — on RTX 4090, sm_89, driver
580.119.02, all at commit `6f7f47ef2` (one kernel throughout).

Two revisions of that script were used, and why that is equivalent is stated rather than
assumed: the unmutated A/B and faults 1–2 ran on the #3964 copy (sha256 `efb145bce8ec`,
self-test 8/8); faults 3–4 ran on #3983's (folded `66c911359`, `a3218ca50aed`, self-test
12/12), wrapped in `gpu-q --prio 1`. #3983 changes only how an INHERITED lock is handled;
how samples are classified is unchanged, so an EXCLUSIVE verdict means the same thing
under both.

Two runs were REFUSED on the way — correctly. One waited behind a CPU-bound MoE `apr qa`
idle-holding the lock; one had a raw llama.cpp `llama-server` (aprender-19's CRUX cell,
whose server outlived its lock release) appear mid-run, sampled 18 times. Both were
re-run on a clean card rather than reported.

## 8. Checks

`cargo fmt --all --check` rc=0 · `cargo clippy -p aprender-serve --lib --features cuda --
-D warnings` rc=0 (0 errors) · 25 GPU-free tests pass on the final tree (gguf-py golden,
lane mapping, geometry, entry-name + ASCII, ptxas assembly), the one-shot probe ignored.

## 9. Not claimed

* That `Qwen3.5-0.8B-UD-IQ2_XXS` goes green — IQ3_XXS still has to land.
* That Q2_K is fast: one warp per output row, eight scalar byte loads per lane per block.
* An end-to-end generation comparison on the real model.
