---
phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
plan: 20
subsystem: format
tags: [apr-format, apr-v2, modernbert, untrusted-input, class-b, mutation-testing]

requires:
  - phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
    provides: "08-12 decide load ladder (rung 3 opens the blob with AprV2ReaderRef::from_bytes); 08-03 ModernBERT encoder"
provides:
  - "APR v2 index must be STRICTLY increasing: both readers refuse a repeated tensor name (InvalidTensorIndex \"duplicate tensor name ...\")"
  - "Index reservation bounded in in-memory entry units (INDEX_RESERVE_UNIT_BYTES), observed through a #[cfg(test)] LAST_INDEX_RESERVE thread-local"
  - "ModernBertLayer::forward / forward_with_rope refuse empty rows (EmptyInput) and mis-sized rows (InputShape) before RopeTable::new"
  - "aprender-core safetensors dev-dependency workspace-pinned (IN-07)"
affects: [08-26, decide-apr-v1 contract table, every APR v2 consumer]

actuals:
  tokens: 4359
  tasks: 3
  commits: 3
plan_head_before: 70125a6b1ded90c1d67d71e23b0babac5e589b5c

tech-stack:
  added: []
  patterns:
    - "Test-only thread_local hook that reads back the REAL reservation (Vec::capacity after allocation) or counts a constructor call, so a mutant of the sizing expression cannot slip past the test (no counting allocator under unsafe_code = forbid)"

key-files:
  created: []
  modified:
    - crates/apr-format/src/v2/reader_impl.rs
    - crates/apr-format/src/v2/tests.rs
    - crates/aprender-core/src/models/modernbert/layer.rs
    - crates/aprender-core/Cargo.toml
    - .planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/deferred-items.md

key-decisions:
  - "The reservation hook records tensor_index.capacity() AFTER Vec::with_capacity, not the computed argument before it, so mutating the with_capacity argument is observed"
  - "The reservation test bounds against the INDEX bytes (file end minus tensor_index_offset), not data.len(): against data.len() the on-disk-units mutant (c) would have stayed green"
  - "A #[cfg(test)] RoPE-table counter makes 'refused BEFORE RopeTable::new' observable, so removing forward's own l == 0 guard is RED even though forward_with_rope carries the same guard"

patterns-established:
  - "Class-B bound + readback hook + mutation row: every size taken from a file or a caller gets a named bound, a test that reads back what was actually sized, and a recorded mutant"

requirements-completed: [D-13, D-17]

coverage:
  - id: D1
    description: "Both APR v2 readers refuse an index that names a tensor twice (WR-01 / AL5)"
    requirement: "D-13"
    verification:
      - kind: unit
        ref: "crates/apr-format/src/v2/tests.rs#duplicate_tensor_names_are_refused_by_both_readers"
        status: pass
      - kind: other
        ref: "mutant (a): equality refusal removed -> RED by assertion"
        status: pass
    human_judgment: false
  - id: D2
    description: "A forged tensor_count reserves no more than the index bytes the file holds, counted in in-memory entries (V2-b / V2-c)"
    requirement: "D-13"
    verification:
      - kind: unit
        ref: "crates/apr-format/src/v2/tests.rs#forged_tensor_count_reserves_nothing_proportional"
        status: pass
      - kind: unit
        ref: "crates/apr-format/src/v2/tests.rs#index_capacity_is_bounded_by_the_index_bytes"
        status: pass
      - kind: other
        ref: "mutants (b) tensor_count as usize and (c) on-disk units -> both RED by assertion"
        status: pass
    human_judgment: false
  - id: D3
    description: "ModernBertLayer::forward refuses empty and mis-sized rows with typed errors before building a RoPE table (WR-04 / V10-b)"
    requirement: "D-17"
    verification:
      - kind: unit
        ref: "crates/aprender-core/src/models/modernbert/layer.rs#layer_forward_empty_row_is_refused"
        status: pass
      - kind: unit
        ref: "crates/aprender-core/src/models/modernbert/layer.rs#layer_forward_with_rope_empty_row_is_refused"
        status: pass
      - kind: unit
        ref: "crates/aprender-core/src/models/modernbert/layer.rs#layer_forward_huge_l_is_refused_before_rope"
        status: pass
      - kind: other
        ref: "mutants (d), (d2), (e) -> all RED"
        status: pass
    human_judgment: false
  - id: D4
    description: "Real-weights Laya parity unchanged (bar 1e-5)"
    requirement: "D-17"
    verification:
      - kind: integration
        ref: "heavy env LAYA_MODEL_DIR=<snapshot 55cf4c4e> LAYA_LADDER_BIN=<spike 025 ladder> cargo test -p aprender-decide --release --test laya_parity -- --nocapture"
        status: pass
    human_judgment: false
  - id: D5
    description: "safetensors dev-dependency workspace-pinned without changing Cargo.lock (IN-07)"
    verification:
      - kind: other
        ref: "grep 'safetensors = { workspace = true }' crates/aprender-core/Cargo.toml; git diff --quiet -- Cargo.lock"
        status: pass
    human_judgment: false

duration: 15min
completed: 2026-09-28
status: complete
---

# Phase 8 Plan 20: Class B Shared-Reader Bounds Summary

**Both APR v2 readers now refuse a repeated tensor name. The index reservation is capped at the index bytes counted in 72-byte in-memory entries. ModernBertLayer returns `EmptyInput` / `InputShape` before it sizes a RoPE table. Each bound has a test that goes RED when the bound is reverted, and real-weights parity is still 3.841e-6.**

## Performance

- **Duration:** about 15 min
- **Started:** 2026-09-28T16:01:06Z
- **Completed:** 2026-09-28T16:16:27Z
- **Tasks:** 3
- **Files modified:** 5 (4 source/manifest, plus deferred-items.md)

## Accomplishments

- **WR-01 / AL5 closed at the shared reader.** `parse_tensor_index_section` requires `name[i] > name[i-1]`. Equal names return `InvalidTensorIndex("duplicate tensor name \"dup\" in tensor index")`. A decrease keeps "not sorted" and now names both entries. Every APR consumer is covered, including the decide ladder's rung 3.
- **V2-b / V2-c closed.** `index_capacity` now counts in `INDEX_RESERVE_UNIT_BYTES = max(20, size_of::<TensorIndexEntry>())`, which is 72 on aarch64. A test-only `LAST_INDEX_RESERVE` reads back `Vec::capacity` after the allocation. At HEAD the forged file reserved 3 x 72 B = 216 B for a 72-byte index; now it reserves 1 entry (72 B). V2-c reproduced: when the reservation is reverted to `tensor_count as usize`, the OLD `forged_tensor_count_is_refused_by_both_readers` still passes, because macOS grants the 309 GB reservation lazily. The new test goes RED.
- **WR-04 / V10-b closed.** `forward` refuses `l == 0` and checks `x == [l, d]` with `check_len`, whose multiply is overflow-checked. Both checks run before `RopeTable::new`. `forward_with_rope` refuses `l == 0` before any slicing. At HEAD, both empty-row calls panicked on `&qkv[d..]` and the huge-`l` call panicked with "attempt to multiply with overflow" inside `RopeTable::new`.
- **IN-07:** `safetensors = { workspace = true }`; Cargo.lock is unchanged.

## Class-B shared-reader surface (for plan 08-26's contract table)

| Size / count (source) | Bound | Refusal | Test |
|---|---|---|---|
| `tensor_count` → initial index `Vec` reservation (file header) | `min(tensor_count, index_bytes / INDEX_RESERVE_UNIT_BYTES)`, unit = max(20 B on-disk, in-memory `TensorIndexEntry`) | none needed: the reservation is capped; the parse loop still fails on the first entry the bytes cannot hold (`InvalidTensorIndex`) | `forged_tensor_count_reserves_nothing_proportional`, `index_capacity_is_bounded_by_the_index_bytes`, `forged_tensor_count_is_refused_by_both_readers` |
| index entry names (file) | strictly increasing | `InvalidTensorIndex("duplicate tensor name ...")` / `("tensor index not sorted: ...")` | `duplicate_tensor_names_are_refused_by_both_readers` |
| `l` for `ModernBertLayer::forward` (caller) | `l >= 1` | `ModernBertError::EmptyInput` | `layer_forward_empty_row_is_refused` |
| `l` vs `x.len()` for `forward` (caller) | `x.len() == l * d`, checked multiply | `ModernBertError::InputShape { what: "layer.x", .. }` | `layer_forward_huge_l_is_refused_before_rope` |
| `l` for `forward_with_rope` (crate-internal) | `l >= 1` | `ModernBertError::EmptyInput` | `layer_forward_with_rope_empty_row_is_refused` |

## Mutation table

Each mutant was applied alone and restored with `git checkout -- <file>`, and `git diff --quiet` was checked before the next one.

| Mutant | Test | RED how | Restored green |
|---|---|---|---|
| (a) sort check back to strict `<` only (equality branch `if false`) | `duplicate_tensor_names_are_refused_by_both_readers` | assertion: `AprV2Reader: expected InvalidTensorIndex(duplicate), got Ok(())` | yes |
| (b) reservation back to `Vec::with_capacity(tensor_count as usize)` | `forged_tensor_count_reserves_nothing_proportional` | **assertion, not abort**: `reserved 4294967295 entries x 72 B = 309237645240 B for a 72-byte index` (macOS granted it lazily); the old refusal test stayed green under the same mutant | yes |
| (c) reservation back to on-disk units (`/ MIN_INDEX_ENTRY_BYTES`) | `forged_tensor_count_reserves_nothing_proportional` | assertion: `reserved 3 entries x 72 B = 216 B for a 72-byte index` | yes |
| (d) `l == 0` guard removed from `forward` | `layer_forward_empty_row_is_refused` | assertion: `the empty row was refused before any RoPE table was built` (`forward_with_rope`'s guard still returned `EmptyInput`, but the RoPE-table counter showed a table had been built) | yes |
| (d2) `l == 0` guard removed from `forward_with_rope` | `layer_forward_with_rope_empty_row_is_refused` | panic: `range start index 32 out of range for slice of length 0` | yes |
| (e) `forward`'s shape check removed, so the only check left is `forward_with_rope`'s, which runs after `RopeTable::new` | `layer_forward_huge_l_is_refused_before_rope` | panic: `attempt to multiply with overflow` at `RopeTable::new` (debug build) | yes |

## Task Commits

1. **Task 1 (tracer): duplicate tensor names refused by both readers**: `fa46bebf0` (fix)
2. **Task 2: in-memory reservation bound, layer guards, IN-07**: `36d3725a2` (fix)
3. **Task 3: mutation proof, real-weights parity, fmt**: `65398e4cb` (test)

## Verification evidence

- `cargo test -p apr-format --lib`: 92 passed, including the 4 named tests.
- `cargo test -p aprender-core --lib format::`: 3955 passed. `models::modernbert`: 24 passed.
- `cargo test -p aprender-decide --lib artifact::`: 52 passed.
- Armed `laya_parity` under `heavy` (lockf): rc=0.
  - `ids 14/14; argmax 14/14; max |dp| 3.841e-6 (bar 1e-5); max |dlogit| 2.146e-5 (bar 1e-4); truncated rows 1; ARCH aarch64`
  - `ladder: 32 blocks within bars`
  - 0 lines carry SKIP
- `cargo clippy -p apr-format --lib --no-deps -- -D warnings`: 0. `cargo clippy -p aprender-core --lib --tests --no-deps` reports nothing in `modernbert/` or `apr-format/`.
- `cargo fmt -p apr-format -p aprender-core -- --check`: 0.
- `git diff --quiet -- Cargo.lock`: 0.
- Tracer gate (interactive, end-of-phase, automated-only verify): re-ran green, then expanded.

## Decisions Made

See `key-decisions`. In short, each test hook observes what was actually sized rather than the value the code intended, because that is the only way a mutant of the sizing expression turns the test RED.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Test theater] The reservation test is bounded against the index bytes, not `data.len()`**
- **Found during:** Task 2
- **Issue:** The plan asserted `last_index_reserve() * size_of::<TensorIndexEntry>() <= data.len()`. The forged file is header (64 B) + padded metadata + 72 index bytes. The on-disk-units mutant (c) reserves 216 B, which fits under `data.len()`, so that assertion would have stayed green.
- **Fix:** The test asserts against `data.len() - tensor_index_offset`, which is what the plan's `<behavior>` line says ("at most the index bytes remaining"). It checks both readers.
- **Verification:** Mutant (c) goes RED (216 > 72).
- **Committed in:** `36d3725a2`

**2. [Rule 1 - Test theater] The hook records `capacity()` after the allocation**
- **Found during:** Task 2
- **Issue:** The plan said to record the computed capacity "just before `Vec::with_capacity`". With that, mutant (b), which rewrites only the `with_capacity` argument, would still have recorded the bounded value.
- **Fix:** The hook records `tensor_index.capacity()` immediately after the allocation.
- **Verification:** Mutant (b) goes RED and records 4294967295.
- **Committed in:** `36d3725a2`

**3. [Rule 2 - Missing proof] RoPE-table counter plus a fourth layer test**
- **Found during:** Task 2
- **Issue:** `forward` delegates to `forward_with_rope`, which carries the same `l == 0` guard. Without an extra observation, removing `forward`'s own guard (mutant d) is masked and still returns `EmptyInput`.
- **Fix:** Added a `#[cfg(test)]` `ROPE_TABLES_STARTED` counter. Both `forward` tests assert that no table was built. Added `layer_forward_with_rope_empty_row_is_refused` so the second guard has its own mutant row (d2).
- **Committed in:** `36d3725a2`

**4. [Scope] The plan's `cargo test -p apr-format` (all targets) is red at HEAD because of `golden_v2_f32_writer_is_byte_identical`**
- **Issue:** The writer produces 516 B; `golden_v2.apr` holds 1092 B.
- **Measured:** With this plan's reader change reverted (HEAD's `reader_impl.rs`), the failure is identical. It is writer/fixture drift, and the three golden tests that READ the fixtures pass.
- **Handling:** Logged to `deferred-items.md` as open and not fixed (out of scope). Task 1's gate was run as `--lib` plus the other targets, and those are green.

**Total deviations:** 3 auto-fixed (2 test-theater fixes, 1 missing proof) and 1 pre-existing out-of-scope failure logged.
**Impact on plan:** The tests are stricter than the plan text, and every mutant went RED. There was no scope creep.

## Issues Encountered

None beyond the pre-existing golden writer failure above.

## Known Stubs

None.

## User Setup Required

None.

## Next Phase Readiness

- Plan 08-26 can fold the five class-B surface rows above into the contract table. The shared reader now refuses duplicates for every consumer, so 08-26's decide-side rung-4 refusal becomes defence in depth rather than the only line of defence.
- The golden_v2 writer drift is open in `deferred-items.md`.

## Self-Check: PASSED

- All four modified source/manifest files exist.
- Commits `fa46bebf0`, `36d3725a2` and `65398e4cb` are present in `git log`.
- `git rev-list --count 70125a6b1..HEAD` = 3 before the docs commit.
