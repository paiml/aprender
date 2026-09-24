# PMAT-3430 Phases 3-4 — receipt: three ggml enums become one

**Ticket:** #3430 (PP-QUANT-001 M1) · **Epic:** #3421 · **Plan:** v4 (#3452), Phases 3 and 4
**Depends on:** #3581 (the leaf + `TRAITS[43]` + the upstream-extracted fixture)
**Measured on:** this branch, 2026-09-20.
**Verdict:** `PASS`. This is the half of M1 that makes the ticket's title true.

## 1. The falsifier, met

#3430's original acceptance criterion: *"exactly 1 ggml tensor-type enum definition under `crates/*/src`"*.

```
$ grep -rnE '^\s*pub enum (GgmlType|GgmlQuantType)\b' crates/*/src --include='*.rs'
crates/aprender-quant/src/ggml_type.rs:140:pub enum GgmlType {
```

Four declarations of the same fact are gone: serve's `GgmlQuantType` (16 ids),
core's `GgmlType` (12), compute's `GgmlType` (15), and `QTYPE_LABELS` — the private
13-row table #3405 added while three enums of it already existed.

`scripts/check_one_ggml_type_enum.sh` keeps it that way. Green on this tree; 15-row
case table; a zero-match arm that FAILS rather than passing vacuously; and it is
discovered automatically by `guard_tree.sh` via `git ls-files 'scripts/check_*.sh'`,
so committing it is what wires it. **Mutation:** re-adding a second definition to
compute turns it RED naming both files, restoring turns it GREEN.

## 2. Behaviour preservation is a measurement, not a claim

Phase 1 (previous commit) wrote **12 characterization tests against unchanged code**
and they all pass **unchanged** through Phases 3-4. That is the entire argument that
this refactor is safe, and it is the only part of it a reviewer can check without
re-deriving the boundaries by hand.

| crate | boundaries | sweep | result |
|---|---|---|---|
| serve | 7 — `gguf/dtype` ×2, `infer`, `apr/mod`'s `from_binary` dtype arm, `apr/special_tokens`, `apr/dequant`, the loader test that mirrors the predicate | ids 0..=255, every admitted name, its lower-case form, and 8 names that must stay unrecognised | unchanged |
| compute | 1 — the GGUF parse boundary | ids 0..=255, plus all 15 rows of block geometry and the partial-block rounding | unchanged |
| core | 0 — measured: nothing in core turns an integer or a string into a `GgmlType` | 24 `(dtype, elements)` points through `byte_size` | one row moved, §4 |

Two edits to those tests were **named in advance** by the plan and by Phase 1's own
module docs, and they are the only two:

1. compute's `GgmlType::from_u32(x)` → `from_u32(x)`. An inherent method cannot
   follow a type defined in another crate. The admitted **set** is untouched.
2. core's `Q4_1` row, §4.

## 3. Each crate keeps its own admitted set

The leaf knows all 35 live ggml types. No crate may start accepting an id it used to
refuse — a loader that silently widens starts decoding bytes it has no kernel for. So
the sets are stated once per crate and every boundary reads them:

- **compute** — `const ADMITTED: [GgmlType; 15]`, one boundary (`from_u32`).
- **serve** — `const ADMITTED: [GgmlQuantType; 16]`, with `admitted_from_id` and
  `admitted_from_name` at **all seven** boundaries.
- **core** — none, and none is needed.

`apr/dequant` narrows *further*, to the quantized subset it has a dequantiser for
(F32/F16/BF16 → `None`), and that narrowing is characterized too.

## 4. The one value that changes

core sized `Q4_1` at **18** bytes per 32-element block. Upstream ggml says **20**
(`block_q4_1` = 2 × `ggml_half` for scale AND min, plus `QK4_1/2` = 16 nibble bytes),
and core's **own** other size table, `format/gguf/shape.rs`, already said 20. Two
tables in one crate disagreeing.

**It was never reachable.** `export_tensors_to_gguf` sizes tensors with
`tensor.data.len()`, and `byte_size()` has no non-test caller in the workspace, so no
GGUF was ever written wrong by it. A unit test (`test_gguf_tensor_byte_size_q4_1`) had
asserted the 18-byte value since it was written — a test pinning a defect.

Phase 1 anticipated this row, asserted the disagreement *directly* in a companion test
so it could not be fixed by accident, and instructed its own deletion once fixed. Both
happened in one commit: two expectations moved (36→40, 54→60), the companion test was
deleted, and **the other 22 assertions passed untouched** — which is the evidence that
delegation changed exactly one value and not twelve.

## 5. What the collapse was worth, counted honestly

Four tables → one. Removed: **one wrong value** (`Q4_1`) and **one omission**
(`QTYPE_LABELS` had no row for `Q8_1`, so a refusal printed `qtype 9` instead of
`Q8_1(9)`).

That is a smaller number than the collapse deserves credit for, and the reason is
worth stating: **the big find is not in this diff.** Measuring in-tree tables against
the extracted one is what exposed `ggml_dtype_element_size` — a *fifth* id-indexed
table — wrong in **10 of 31 rows**, including BF16 at `0.375` B/elem instead of `2.0`,
which makes the #2569 truncation check fail-open on BF16 GGUFs. That is **#3583**, and
it is deliberately not folded in here.

`QTYPE_LABELS`'s replacement goes through `from_id`, never `TRAITS[id]`: it takes an
arbitrary `u32` from a file and a raw index would panic for any id ≥ 43. It is also
deliberately **not** filtered by the admitted set — it is a diagnostic, and naming id
21 in a refusal beats printing `qtype 21`.

## 6. Verification

```
cargo nextest run --profile ci -p aprender-serve -p aprender-core -p aprender-quant --lib
                                          30,168 passed, 61 skipped
cargo test -p aprender-compute --lib       3,514 passed
cargo check --workspace --all-targets      clean
cargo clippy -p {quant,compute,core,serve} --lib -- -D warnings   clean
cargo fmt --all -- --check                 clean
bash scripts/check_one_ggml_type_enum.sh             exactly 1 definition
bash scripts/check_one_ggml_type_enum.sh --self-test 15 cases, 0 failures
```

## 7. The stop condition, and that it did not fire

I said before starting that if a boundary's behaviour could not be preserved I would
stop and surface it rather than widen an admitted id set to make a test pass — and
that under automatic publishing, widening would be a fabricated green feeding a
release. It did not come up: all 12 characterization tests passed unchanged, and both
edits to them were named before the work began. The sets in §3 were **reproduced from
the measured behaviour**, not chosen.

## 8. Not in this PR

`qtype: u32` field retyping (Table B, 53 rows) — T1/Q2. The 558 bare-integer lines —
#3431. The Table E merges — #3423 Phase 2. `ggml_dtype_element_size` — #3583.
