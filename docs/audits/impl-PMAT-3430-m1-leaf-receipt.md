# PMAT-3430 M1 — receipt: the leaf (`GgmlType` + `TRAITS[43]`), oracle-extracted

**Ticket:** #3430 (PP-QUANT-001 M1) · **Epic:** #3421 · **Plan:** #3452 (`docs/audits/impl-PMAT-3430-plan.md`, v4)
**Ruling executed:** operator, 2026-09-20 on #3430, clauses 1–6, with amendment 2'/2a/2b.
**Measured on:** `origin/main` @ `aff11bd0e`, 2026-09-20, this worktree.
**Verdict:** `PASS` for the leaf. Phases 3–4 (re-export + admission) and the consumer
table fixes are NOT in this PR — see §6.

## 1. The oracle, resolved by reference

Clause 2' makes the oracle `scripts/llama_pin.toml` `build_commit`, never a literal sha.

| | value | how |
|---|---|---|
| pin of record | `d1d3c3396` | `llama_pin_get build_commit` — the repo's only reader of that file |
| checkout read | `/home/noah/src/llama.cpp-d1d3c3396` | `git -C … rev-parse HEAD` → `d1d3c3396aa13a5f239109a822666c4870490ad5`, clean tree |
| the ruling's `3173a5647` | **not used** | struck by amendment 2': it was the checkout the 43/35/8 count was first measured at, not a pin |

`scripts/extract_ggml_traits.sh` refuses (exit 3) a checkout whose HEAD is not the pin,
and refuses (exit 4) a dirty one. Neither refusal is theoretical: both were exercised
while writing it — the first `-s` attempt pointed at `/home/noah/src/llama.cpp-master`.

## 2. What was extracted, and why a probe rather than a parse

ggml's `type_traits[]` gives `.type_size = sizeof(block_q4_1)`: a C expression, not a
number. **Every in-tree table that got these numbers wrong got them wrong by hand.** So
the extractor compiles a probe against the pinned checkout's own `libggml-base` and calls
`ggml_type_name` / `ggml_blck_size` / `ggml_type_size` / `ggml_is_quantized`.

Result, independently reproducing the operator's 2b measurement:

| | count |
|---|---|
| `GGML_TYPE_COUNT` | 43 |
| live | 35 |
| removed (commented out in the header) | 8 — ids 4, 5, 31, 32, 33, 36, 37, 38 |

The header is still parsed, as a **cross-check that can fail**: a row is Removed because
its line is COMMENTED, never because the comment says "removed" (`// GGML_TYPE_Q4_0_4_8 = 32,`
says nothing), and the classification must then agree with the compiled table's
`blck_size == 0`. Disagreement is exit 7 and no fixture.

## 3. The falsifier (clause 3): mutation, measured

Four mutations of the leaf, one of the fixture, one of the pin. **All RED; every control GREEN.**

| # | mutation | result |
|---|---|---|
| M1 | `TRAITS[9].type_size` 36 → 40 (gguf-py's Q8_1, refuted by the C `static_assert`) | RED — `traits_table_equals_the_upstream_fixture_row_for_row` *and* `computes_fifteen_sizes_are_unchanged` |
| M2 | `TRAITS[30].name` `BF16` → `Bf16` | RED — `serves_sixteen_strings_still_mean_what_they_meant` |
| M3 | `TRAITS[4].family` `Removed` → `Affine` | RED — the row-for-row test *and* the is_quantized cross-check, independently |
| M4 | fixture gains a fake id 43 | RED — three tests |
| M5 | fixture `pin_build_commit` → `39173bcac` (the superseded comparator) | RED — `falsify_ggml_001`, message naming both shas and the regeneration command |
| M6 | a row deleted from the fixture | RED — `falsify_ggml_002` ("the fixture lists 42 of 43 ids") |
| — | unmutated | GREEN: 8 + 3 tests |

The extractor's own refusal is proved by a 7-case table on synthetic inputs
(`scripts/extract_ggml_traits_selftest.sh`), which needs no llama.cpp checkout and no
compiler, and is run in CI from `falsify_ggml_003` — which **refuses a zero-case run**,
because a case table that runs nothing passes vacuously.

## 4. Clause 2a: a pin bump turns this RED in the bump's own PR

The fixture records `pin_build_commit` and `resolved_sha`.
`crates/aprender-core/tests/monorepo_invariants.rs::falsify_ggml_001` asserts the first
still equals `llama_pin.toml build_commit`. It lives there and not in `aprender-quant`
because trueno_quant is **published**, and a packaged crate has no `scripts/llama_pin.toml`
— a test there could only skip, and a skip is how this becomes decoration. That target is
already in `ci/explicit-test-commands.d/030-*`, so it is not a dark target. The new
`aprender-quant` target is wired by `350-aprender-quant-ggml-traits-fixture.cmd`;
`check_explicit_test_commands.sh` passes (49 fragments).

## 5. Findings — measured, and NOT fixed here

These are clause 4's input (#3431's reconciliation table). Every value below is
`type_size / blck_size` at the pin.

| # | site | in-tree | upstream | verdict |
|---|---|---|---|---|
| F1 | `aprender-compute` `GgmlType::block_bytes` / `block_size`, 15 ids | **all 15 correct** | — | no defect. The plan's premise that compute's `Q8_1 = 36` disagreed with upstream was wrong: 36 IS upstream (C `static_assert`); 40 is gguf-py's pre-f16 number, and it appears in no in-tree code |
| F2 | `aprender-core` `GgufTensor::byte_size`, `format/gguf/types.rs:365` | `Q4_0 \| Q4_1 => …*18` | Q4_1 is **20** | REGRESSION-shaped, but **latent**: `export_tensors_to_gguf` sizes tensors with `tensor.data.len()`, and `byte_size()` has **no non-test caller in the workspace**. So no GGUF was ever written wrong by it. Clause 5's "refusal on load" has no load path to attach to here |
| F3 | `aprender-core` `format/gguf/shape.rs:216-240` | all correct, incl. `Q4_1 = 20` | — | so core contains two tables of this fact that **disagree with each other** |
| F4 | `aprender-core` `ggml_dtype_element_size`, `format/safetensors.rs` | **10 of 31 rows wrong** | — | NEVER-WORKED. From index 16 the table follows the ORDER OF ITS OWN COMMENT (`… I8, I16, BF16, I32, I64, F64, IQ1_M`) instead of ggml id order (24 I8, 25 I16, 26 I32, 27 I64, 28 F64, 29 IQ1_M, 30 BF16), and the IQ rows carry values belonging to no type. Wrong: 16, 17, 18, 19, 21, 22, 23, 26, 27, 29, 30 |
| F5 | same table | ids 34, 35, 39–42 absent (`unwrap_or(4.0)`) | TQ1_0, TQ2_0, MXFP4, NVFP4, Q1_0, Q2_0 | the four "recent" types the ruling predicted would be NEVER-WORKED |
| F6 | `aprender-quant`'s own `Q4_K_BLOCK_BYTES` &c. | correct | — | **fixed structurally here**: six `const` assertions now make a drift a COMPILE error |

**F4 is the one with a user-visible consequence.** `list_tensors_gguf` uses that table for
the #2569 truncation check. BF16 (id 30) read **0.375 B/elem instead of 2.0** — a 5.3×
under-estimate, so a truncated BF16 GGUF passed the extent check (fail-open) and
`apr tensors` printed a size 5.3× too small. The IQ rows err the other way and would
reject an intact file as truncated. PMAT-869 corrected four K-quant rows in this same
table in 2026-06 and stopped there: the audit-lags-main pattern, again.

**Not fixed in this PR on purpose.** The fix is a lookup into `TRAITS`, which is what this
PR creates; doing both at once would make the leaf's "changes no behaviour" claim
unverifiable. The patch is written and measured (11 rows, every value an exact dyadic
fraction) and lands next, before Phases 3–4.

## 6. Deviations from plan v4, each with its reason

| plan said | this PR | why |
|---|---|---|
| Phase 0b (`Q4_1` 18→20) lands FIRST, before the leaf | leaf lands first; the table fixes follow, **derived from `TRAITS`** | the plan's ordering pre-dates the fixture existing. Fixing a table by hand-typing a second set of numbers is the defect, not the remedy. The leaf changes no behaviour, so nothing rides a refactor either way |
| `TRAITS[i].name` is the value `as_str()` returns | `name` (GGUF spelling, `"Q4_K"`, == `as_str()`) **and** `ggml_name` (`"q4_K"`, what `ggml_type_name` returns) | ggml's `type_name` is not serve's string. Carrying one and deriving the other would have meant a case transform in `const fn`; carrying both makes each a fixture-checked reading, and the test asserts they differ in case only |
| falsifier: `TRAITS` == gguf-py `GGML_QUANT_SIZES` with a named exception list | superseded by clause 3 | gguf-py is no longer consulted at all. The C table is the oracle, so Q8_1 is simply 36 and there is no exception list to rot |
| clause 5: wrong-sized types ship with a refusal on load | **no refusal shipped** | measured: the only wrong in-tree size (F2) sits in a function with no production caller, so there is no load to refuse. Stated rather than silently skipped: if #3431's reconciliation finds a wrong size on a real load path, that type gets clause 5's treatment there |

## 6a. What adding ONE contract file actually costs, measured

Not planned for, and worth recording because the plan did not see it and nor did I.
`contracts/ggml-type-v1.yaml` is one new file. It staled **three** tracked derivatives
and tripped a **fourth** guard, and they surfaced **one per CI round**, because each was
masked by the one before it — a failing job ends the run (#3587).

| round | red | derivative | remedy |
|---|---|---|---|
| 1 | guard-cargo | `contracts/census.json` (1799 vs 1800) | `make contracts` (`pv census contracts`) |
| 1 | shard (1/3) | `contracts/contracts.nt` — `the_tracked_repo_graph_is_fresh`, R-18 | `pv extract contracts` |
| 2 | guard-cargo | README's generated `CONTRACT_COUNT` block — **invisible in round 1**, masked by the census red | `make readme-sync` |
| 3 | guard-tree | §11.1 ont-delta: touching README.md at all reclassifies the PR as a *sweep* | an `ont-delta:` line in the PR body |

The fourth is the one to notice: it is not a derivative of the contract, it is a
consequence of the **remedy** for the third. `make readme-sync` is mandatory (guard-cargo
refuses without it) and README.md is a §11.1 prose sink, so fixing guard-cargo is what
made guard-tree fail. Neither guard can see the other, and nothing in either message says
so.

Two further findings that cost a round each, both mine and both the same shape — reasoning
about an environment instead of measuring it:

- **`git rev-parse --show-toplevel` fails in the container** ("dubious ownership in
  repository at `/workspace`": the tree is uid 1000, the container is root). A script
  locating its own siblings never needed git. Now derived from `$0`; generalised as #3586,
  where 7 of 22 such call sites are unguarded and 15 more carry `|| pwd`, which does not
  fail — it silently reads the wrong tree.
- **`python3` is absent from the container.** The extractor's classifier was python and
  its case table drove it with python3, so the one artifact proving the classifier can
  fail could not execute where the code executes. Ported to awk; the port is proved
  faithful by regenerating the fixture **byte-identically**, and proved portable by
  running the case table under `env -i PATH=<coreutils only>` with no python3 on PATH.
  I had inferred python's presence from `python3 scripts/lib/roadmap_merge.py --selftest`
  in ci.yml — a step in a different job, on the host runner.

## 7. Not in this PR

Phases 3–4: the three enums become re-exports, each crate gets its admission function
(compute 15 ids, serve 16, core none), `Bf16`→`BF16`, `QTYPE_LABELS` removed,
`GgufTensor::byte_size` delegates. Until then `scripts/check_one_ggml_type_enum.sh` would
be RED, so it is not shipped yet either — a guard that cannot go green gets disabled
(#2512's lesson), and the honest sequence is guard-with-the-fix.

The 558 bare-integer qtype lines, the 53 raw-integer fields, and the Table E merges remain
#3431 / T1 / #3423 as the plan says.
