# #3869 — IQ4_NL, and the constant that was never a constant

**Worker:** aprender worker two. **Branch:** `kernels`. **Commits:** `97deeceea`
(IQ4_NL + per-type block elems), `3d244d83c` (the parity guard that outlived its
example).

## Verdict

IQ4_NL CPU dequant is done and **measured working**. `apr qa`'s acceptance
(`rc=0`) is **NOT met**, for two reasons that are pre-existing and orthogonal to
IQ4_NL — §4. IQ4_XS.gguf remains blocked on Q5_1, exactly as scoped.

## 1. The ticket was scoped as a match arm. It is not one.

`iq_dispatch.rs` was built on `IQ_BLOCK_ELEMS = 256`, hardcoded into the matvec
arithmetic in three places: `blocks_per_row = in_dim.div_ceil(256)`,
`col0 = b * 256`, `n = 256.min(in_dim - col0)`.

IQ4_NL is **32 elements in 18 bytes** — `ggml-common.h:447` `#define QK4_NL 32`,
and the struct's own `static_assert` gives `2 + QK4_NL/2 = 18`. The tree already
knew: `aprender-quant/src/ggml_type.rs:177`, and the GGUF loader's table at
`ggml_type_table.rs:150` already carried `blck_size: 32, type_size: 18`. **Only
the dispatch assumed 256.**

So `IQ4_NL => Some(18)` alone would not have errored. It would have computed
`blocks_per_row = in_dim/256`, read one 18-byte block where a row has eight, put
every row after the first at the wrong offset, and summed 224 activations
against scratch it never wrote — returning a float. The same shape as #3850's
`resolve_qtype().unwrap_or(Q4K)`: a plausible wrong answer where a refusal
belongs.

`iq_block_elems(qtype)` makes the count a property of the type.
`every_type_with_a_block_size_has_an_element_count` keeps the two accessors from
disagreeing, since a type with bytes but no elems would silently take the 256
default.

## 2. The predicted free RED did not happen, and that is the finding

`rows_are_read_at_their_own_stride` iterates `IQ_TYPES`, so adding IQ4_NL to that
array was expected to turn it red for free. **It stayed green.**

That row builds ONE block per row and calls the matvec with `in_dim = 256`. For
a 32-element type that is 8 blocks — but `blocks_per_row = 256.div_ceil(256) = 1`
makes the length check pass, both rows still land on their own block, the 224
unwritten scratch slots are zero, and an all-ones activation makes them
contribute nothing. **Green, about nothing** — the same vacuous-subject shape as
the Q4_2 catch in the parity guard, twice in one session.

`a_row_spanning_two_blocks_is_read_at_the_types_own_stride` is the row with an
actual subject: `in_dim = 2 * elems_per_block`, four blocks laid out as a real
row-major tensor, and a **position-dependent** activation so a block read at the
wrong offset cannot coincidentally sum the same. RED for type 20 only:

```
type 20 (32 elems/18 B per block), row 1: matvec 44.390625 vs dequantized dot 0
  — the row stride is wrong for this type
```

Green for the other five, so it is specific to the defect rather than to the
change.

## 3. The existing five types are unchanged — measured, not assumed

Captured a `sum / sumsq / matvec` fingerprint per type **before** touching the
arithmetic, re-ran it after. All five byte-identical:

```
qtype=16 IDENTICAL   qtype=18 IDENTICAL   qtype=21 IDENTICAL
qtype=22 IDENTICAL   qtype=23 IDENTICAL
```

The dequantizer is transcribed from llama.cpp's `dequantize_row_iq4_nl`
(`/mnt/nvme-raid0/llama.cpp-master`). Beyond matching reference values it is
cross-checked **structurally** against the independently-transcribed IQ4_XS
module: IQ4_XS is IQ4_NL's codebook with a 6-bit sub-block scale layered on
(`dl = d * (ls - 32)`), so an IQ4_XS super-block with every `ls = 33` has
`dl = d`, and its eight 32-element sub-blocks must equal eight IQ4_NL blocks
carrying the same nibbles. A disagreement means one of the two transcriptions is
wrong — a stronger statement than either checking its own expected values.

Also pinned: the two nibbles of byte `j` land at `j` and `j+16`, **not** `2j` and
`2j+1`. Reading them adjacently transposes every block's halves and still
produces plausible numbers.

## 4. The end-to-end A/B, and why `apr qa` rc=0 is still not met

```
apr=<scratchpad>/apr-3869 sha=0ed8cef6a5599eb9f587d429 version=apr 0.69.1 (97deeceea)
worktree clean · model Qwen2.5-0.5B-Instruct-IQ3_M.gguf sha=0b2e17471ccc784dfd7b042a
```

Same model, same gate, one variable:

| binary | Golden Output |
|---|---|
| pre-IQ4_NL (`apr-3870`) | **could not run** — `owned_fused_matmul not supported` |
| post-IQ4_NL (`apr-3869`) | **3 golden test cases passed** |

`apr run` on it: rc=0, *"The capital of France is Paris."*

**`apr qa` still exits 5**, on two gates that are not about IQ4_NL:

* **Tensor Contract** — 117 violations, *"aprender-core has no dequantizer for
  it"*. `aprender-core/src/format/gguf/shape.rs:134` has **no IQ arms at all**;
  everything 16..=23 hits the honest refusal #3656 installed in place of
  `dequantize_iq_approximate`. Proved pre-existing and type-wide rather than
  assumed: the same gate fails identically on `Qwen3.5-0.8B-IQ4_XS.gguf` with
  **126 violations, ggml type 23** — a type aprender-serve has supported since
  #3850. **No IQ model of any kind can pass Tensor Contract today.**
* **Ollama Parity** — 0.01x. A performance gate, not correctness.

**This is the two-mechanisms rule.** aprender-serve and aprender-core each have
their own dequant dispatch; I extended one, and `apr qa` reads through the other.
Adding IQ dequantizers to aprender-core is an **architectural call I am not
making unilaterally**: aprender-serve already has all six, aprender-core depends
on neither, and the natural shared home is `aprender-quant`, which already owns
the type table. Duplicating six dequantizers to satisfy a gate would be the
wrong answer. Surfaced for a ruling.

## 5. IQ4_XS.gguf — the scope claim, settled by composition

| model | composition |
|---|---|
| `Qwen2.5-0.5B-Instruct-IQ3_M` | F32 x121, **IQ4_NL x96**, Q5_0 x48, IQ3_S x21, Q4_K x3, Q8_0 x1 |
| `Qwen2.5-0.5B-Instruct-IQ4_XS` | F32 x121, **IQ4_NL x120**, IQ4_XS x24, **Q5_1 x24**, Q8_0 x1 |

IQ3_M needed exactly one type and now has it. IQ4_XS needs two, and the negative
control names the remaining one precisely:

```
apr run …IQ4_XS.gguf → rc=8
  Fused matmul only supports F32/BF16/F16/Q4_0/Q4_1/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K and
  the dequant path for IQ*/Q2_K/Q3_K, got type 7
```

**Type 7 = Q5_1, not IQ4_NL.** So the IQ4_NL half is complete and Q5_1 is the
whole of what is left. Q5_1 appears in neither the fused arms (Q4_0, Q8_0, Q4_K,
Q5_K, Q6_K) nor `dequant_fallback_or_refuse`'s admission
(`iq_block_bytes.is_some() || Q2_K || Q3_K`). A Q5_1 dequantizer exists at
`aprender-core/src/format/gguf/dequant.rs:220` but is `pub(crate)` — the same
cross-crate visibility shape as §4.

## 6. Checks

`cargo fmt --all --check` rc=0 · `clippy -p aprender-serve --lib -D warnings`
rc=0 · `cargo test -p aprender-serve --lib` **15993 passed, 0 failed**, 59
ignored · `clippy -p apr-cli --lib -D warnings` rc=0, 4 parity rows pass.

## 7. Not claimed

* Not claimed `apr qa` rc=0. It is 5, on the two gates in §4, neither about
  IQ4_NL.
* Not claimed IQ4_NL is fast. It is the dequantize-then-dot path: correct, not
  optimized. No fused kernel, no SIMD.
* No GPU path yet, and the whitelist is deliberately untouched — the rule is
  that a qtype does not enter `gpu_unsupported_quant_qtype`'s allow-list before
  the kernel behind it exists.
