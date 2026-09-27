# #3884 — IQ3_S GPU GEMV

**Worker:** aprender worker two. **Branch:** `feat/iq3s-gpu-gemv` off
`origin/release/0.69.1-batch-2` @ `712f6d433`. **Commit:** `ac968613a`.

## Verdict

Done and admitted. The device A/B is **exact**, and the whitelist opened in the
same commit as the guard flip, citing it.

## 1. The pre-flight the cop asked for, answered by demonstration

> *"confirm it fires for 21 rather than assuming the guard covers it"*

`the_dispatch_agrees_with_the_loader_about_every_block_layout` (`fe763c0fc`)
iterates `0u32..=40`, not `IQ_TYPES`, so type 21 is in scope. That is a reading.
The demonstration: planting `GGML_TYPE_IQ3_S => Some(32)` reds it with

```
type 21 (IQ3_S): dispatch says 32 elements per block, loader says 256 —
this is exactly the IQ4_NL defect
```

IQ3_S is a plain 256-element super-block at 110 bytes, so there was no stride
surprise to find. The check cost 40 seconds and the alternative was an
assumption.

## 2. The format, and the two places a port goes wrong

110 bytes per 256 elements: `d` f16 at +0, `qs[64]` at +2, `qh[8]` at +66,
`signs[32]` at +74, `scales[4]` at +106. 9-bit indices into a 512-entry grid —
8 bits in `qs`, the 9th in `qh` — with explicit sign bytes.

**Thread mapping.** 8 sub-blocks x 4 groups is exactly 32, so `ib = tid >> 2`,
`l = tid & 3` gives one warp lane per (sub-block, group) pair owning 8
consecutive outputs. No lane does two groups and none idles.

**The scale pairing is the trap.** ggml walks `ib32` in steps of 2 with an inner
`half`, indexing `scales[ib32/2]` and taking the low nibble when `half == 0`.
Flattened to one `ib` per lane that is `scales[ib >> 1]`, low nibble when
`ib & 1 == 0` — one byte serving two consecutive sub-blocks. Getting it wrong
swaps the two scales within every pair and still produces plausible magnitudes.

Two simplifications over the reference, **verified equivalent in Python against
the CPU algorithm before any Rust was written** (256/256 elements, 0 mismatches):

* `KMASK_IQ2XS` is `[1,2,4,…,128]`, so `sign_byte & KMASK[j]` is just bit `j`
* ggml's `(qh << (8-2l)) & 256` selects bit `2l` of `qh`, so it becomes
  `((qh >> 2l) & 1) << 8`

## 3. Assembled before it was wired

`ptxas` is a compiler and needs no GPU (#3739). The PTX — 11031 characters
including a 512-entry grid table — was assembled at sm_70 from a scratch file
**before insertion**, then re-extracted from the Rust source and assembled again,
because the thing that ships is the string literal and not my scratch file. The
generator asserts the body is ASCII and quote-free.

Guard coverage **proved, not assumed**: a planted `bogus.instr` reds
`every_emitted_kernel_assembles` naming `Iq3SGemv { k: 4096, n: 4096 }`.

## 4. The measurement that opened the whitelist

```
#3884 A/B: 48 rows, worst relative disagreement 0.000e0
```

Exact against `iq_parallel_matvec` on the same bytes, RTX 4090 sm_89,
`in_dim = 512` so the row stride spans two super-blocks, position-dependent
activation so a block read at the wrong offset cannot coincidentally sum the
same. Vacuity floor: at least half the reference rows must be non-zero.

**Proved able to fail first**, on the three mechanisms this format actually has —
a 0.000e0 on a first run is precisely the result that should not be believed
until the comparison has been shown to go red:

| planted fault | worst row | GPU | CPU | relative |
|---|---|---|---|---|
| scale nibble inverted | 13 | 9311 | 1209 | 6.701e0 |
| 9th grid bit from `2l+1` | 13 | 11085 | 1209 | 8.169e0 |
| sign bits ignored | 1 | −17453 | 1523 | 1.246e1 |

The thread-mapping test reds on its own two mutants (scale pairing, 9th bit).

## 5. Unlike IQ4_NL, IQ3_S is safe in `from_size`

110 bytes per 256 elements collides with nothing. **Checked, not assumed:**

```
 110  IQ3S (super)                                   <- unique
 136  IQ4XS (super)
 144  Q4K (super), Q4_0 (32-blk), IQ4NL (32-blk)     <- pre-existing collision
 176  Q5K (super), Q5_0 (32-blk)                     <- pre-existing collision
 210  Q6K (super)
 272  Q8_0 (32-blk)
```

So the admission test asserts `from_size` **does** name IQ3_S — the opposite of
IQ4_NL's guard, and deliberate.

## 6. A row that outlived its premise was converted

`the_types_found_in_the_wild_without_kernels_resolve_to_none` listed IQ3_S as a
census type with no kernel. That stopped being true here. IQ3_S came off the
list and `iq3_s_is_admitted_because_its_kernel_was_measured` carries the other
half — the same treatment IQ4_NL got, and the one #3850's IQ4_XS kernel should
have given the parity refusal guard instead of leaving it red on the batch
branch.

## 7. What this unblocks, censused against the post-#3884 whitelist

```
Qwen2.5-0.5B-Instruct-IQ3_M.gguf     EVERY TYPE NOW WHITELISTED
Qwen3.5-0.8B-IQ4_XS.gguf             EVERY TYPE NOW WHITELISTED
Qwen2.5-0.5B-Instruct-IQ4_XS.gguf    BLOCKED on ONE type: Q5_1 x24
Qwen3.5-0.8B-UD-IQ2_XXS.gguf         BLOCKED on four: IQ2_XXS x95, IQ3_XXS x24,
                                     IQ2_S x5, Q2_K x3
```

**`type 18` was named as the UD-IQ2_XXS blocker; it is one of four and not the
largest.** `blk.0.attn_gate.weight` is genuinely IQ3_XXS, but IQ2_XXS at 95
tensors is the dominant type. Naming one failing tensor is not the blast radius —
the same shape as reporting "IQ4_NL" for the IQ4_XS row when the acceptance run
said `got type 7`.

## 8. Checks

`cargo fmt --all --check` rc=0 · `clippy -p aprender-serve --lib --features cuda
-- -D warnings` rc=0 · `cargo test -p aprender-serve --lib --features cuda
cuda::` **1369 passed, 0 failed**, 4 ignored. All three read before committing.

## 9. Not claimed

* **Not claimed that the two now-unblocked rows go green.** The quantization
  blocker is gone; whether the rows pass needs a ladder run. `Qwen3.5-0.8B-IQ4_XS`
  in particular still has the think-block-unclosed failure, which no kernel
  touches.
* Not claimed IQ3_S is fast. One warp per output row, scalar grid lookups, no
  vectorized loads and no shared-memory staging of the codebook.
* The A/B is synthetic blocks at a single shape plus the mapping test. It is not
  an end-to-end generation comparison on a real IQ3_M tensor.
