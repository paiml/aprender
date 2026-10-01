# 05-11 Task 1 — RETAINED-vs-NARROWED inventory

Prepared 2026-09-08 for the blocking human checkpoint on narrowing
`contracts/setfit-benchmark-claims-v1.yaml` from an 80-cell two-method expectation set to a
40-cell SetFit-only active set. **Nothing in this bundle is committed.** The amended file sits
uncommitted in the working tree; `HEAD` still carries 1.0.0.

## Provenance of the two compared files

```
git show HEAD:contracts/setfit-benchmark-claims-v1.yaml > /tmp/p11/claims-old.yaml
```

| Path | SHA-256 (at capture) |
|---|---|
| `/tmp/p11/claims-old.yaml` (HEAD = `b69498736`) | `22c438a9bd3bba7a3c4b02e68f43d1faf6ad9f35fd7b42968c3b86f25a415b77` |
| `contracts/setfit-benchmark-claims-v1.yaml` (before editing) | `22c438a9bd3bba7a3c4b02e68f43d1faf6ad9f35fd7b42968c3b86f25a415b77` |

Working tree was byte-identical to HEAD before the edit, so the diff below is exactly this
plan's proposal and nothing else.

## Every 1.0.0 clause that names a cell count or a second method

The candidate set was **enumerated mechanically**, not by reading — a script loaded the 1.0.0
YAML and flagged every equation, obligation, falsification test and Kani harness whose
serialized subtree matches `80 | lora | LoRA | two-method | second method | pair`. Ten of ten
equations, four of nine obligations, six of ten falsification tests and the one Kani harness
matched. Every one of those 21 items appears below exactly once. Nothing is unaccounted for.

### Equations (10 of 10 matched)

| # | Equation | Disposition | What actually changed |
|---|---|---|---|
| 1 | `expectation_set` | **NARROWED** | `methods` loses `lora` (now a one-element list); `expected_cells` 80 → 40; formula `2 * 4 * 10 = 80` → `1 * 4 * 10 = 40`; codomain 80 → 40; `scope: active` added. Two invariants restated (FORTY IS A PRODUCT; the duplicate example 80/88 → 40/44), two invariants ADDED (active-method list ≠ representable-method set; an out-of-scope cell is refused by two existing paths). |
| 1b | `expectation_set.deferred_two_method_scope` | **RETAINED-AS-DEFERRED** (new sub-block) | The 80-cell two-method product, moved here intact: `deferred_methods: [setfit, lora]`, `deferred_expected_cells: 80`, `status: deferred`, `deferred_ticket: D-ITEM-05-15`, `deferred_decision: D-19`. Nested inside `expectation_set` rather than promoted to a sibling equation on purpose — see "Why nested" below. |
| 2 | `bench_row_schema` | **UNCHANGED in substance** | One prose clarification on `shared_core.method`: both `setfit` and `lora` remain REPRESENTABLE row tags. The row-validity domain is deliberately NOT narrowed, because the deferred-scope negatives doctor rows a narrowed domain could not construct. |
| 3 | `completeness_rule` | **NARROWED** | "all 80 cells" → "all 40 active cells"; the expectation-set backstop 80 → 40. Two invariants ADDED: the narrowing is itself the adversary this rule names, so it required human approval and no path may reduce the set further; and the active scope publishes no cross-method result, not even an empty section header. Fail-closed behaviour is **unweakened** — no refusal was removed or relaxed. |
| 4 | `pairing_rule` | **RETAINED-AS-DEFERRED** | `status: deferred`, `mechanism_status: active`, ticket + decision pointers. Formula unchanged. Domain re-worded to say the active-scope domain is EMPTY and is reported as *not exercised*, never as *satisfied*. One invariant ADDED: the RULE is deferred, the KEY is not — every active SetFit cell still records `selection_manifest_hash`, so restoration runs the second arm rather than re-running SetFit. |
| 5 | `selection_safety_evidence` | **SPLIT** — SetFit side unchanged/active, `lora_side` **RETAINED-AS-DEFERRED** | `lora_side` gains `status: deferred` + ticket + a note that nothing below has ever been recomputed against real bytes. Its `rule`, `why`, `why_before_evaluation` and the whole `residual_risk` block are byte-unchanged. |
| 6 | `no_selection_attestation` | **RETAINED-AS-DEFERRED** (whole equation) | `status: deferred` + ticket + decision + note. The six conjuncts, the ordering argument and the frozen-defaults invariant are byte-unchanged. |
| 7 | `model_size_comparability` | **SPLIT** — fields active, cross-method claim **RETAINED-AS-DEFERRED** | `field_status: active` / `cross_method_claim_status: deferred` + ticket + note. Both fields stay mandatory on every active row. All three invariants — including "presenting LoRA's adapter-only `artifact_bytes` beside SetFit's standalone APR is a FORBIDDEN COMPARISON" — are byte-unchanged. |
| 8 | `claims_statistics` | **SPLIT** — paired delta **RETAINED-AS-DEFERRED**, seed-dispersion interval **ADDED as the active statistic** | Formula gains an explicit ACTIVE clause (`CI95_seed = mean ± t_crit * s / sqrt(10)`) beside the existing DEFERRED paired clause, which is unchanged. New `seed_dispersion_ci95` sub-block: status, formula, the label it must be presented under, the ONE-definition-of-the-moments rule (OPS-03), the zero-variance typed shape, and the scipy reference case. Two invariants ADDED (the paired half is deferred and emits no empty section; uncertainty survives the descope). All six 1.0.0 invariants retained. `t_crit_975_df9`, `n_seeds`, `degrees_of_freedom`, `std_convention`, `t_crit_source` byte-unchanged. |
| 9 | `resource_protocol` | **SPLIT** — cross-host framing **RETAINED-AS-DEFERRED**, within-row asymmetry **ADDED** | The D-09 no-pooling RULE stands unchanged; its two-host *description* (CPU SetFit vs lambda-vector GPU LoRA, as-deployed method costs) is marked deferred, because that design was not run and a note claiming it would describe something that does not exist. New `not_comparable.within_row_asymmetry`: the train and inference peaks never share a column, each prints its mechanism string, and a `sysinfo_sampled_*` figure carries its lower-bound label everywhere it is rendered. |
| 10 | `pitfall_bindings` | **NARROWED (text only)** | `pf_007.answered_by` 80 → 40 and now names the active seed-dispersion CI95 with paired deltas marked deferred. The PF-007/PF-008 mapping itself is unchanged. |

### Proof obligations (4 of 9 matched; 9 of 9 still present)

| Obligation | Disposition | What changed |
|---|---|---|
| `OBLIG-CLAIMS-EXPECTATION-CLOSED-FORM` | **NARROWED** | 80 → 40 in prose and in `formal`. `formal` gains two conjuncts asserting the deferred scope still yields 80 and that the active method list is a subset of the deferred one. Names `ACTIVE_METHODS` (expectation domain) vs `BENCH_METHODS` (row-validity domain) and states the obligation binds the former. The substring-comparison prohibition is byte-unchanged. |
| `OBLIG-CLAIMS-METHOD-TAGGED` | **UNCHANGED** | Both variants of the tagged enum stay; a `lora` row remains representable and non-Option. |
| `OBLIG-CLAIMS-COMPLETENESS-FAIL-CLOSED` | **NARROWED + pairing clause RETAINED-AS-DEFERRED** | 80 → 40 in prose and `formal`; the pairing conjunct is moved to a `DEFERRED:` line rather than deleted. Adds that the existing second backstop is what refuses a DECLARED out-of-scope cell and the existing slot check is what refuses a mis-slotted one — no new refusal minted. |
| `OBLIG-CLAIMS-SELECTION-RECOMPUTED` | **SPLIT** — SetFit half active, LoRA half **RETAINED-AS-DEFERRED** | Adds a paragraph saying the ledger half has never been recomputed against real bytes and must not be reported as discharged evidence. The residual-risk paragraph is byte-unchanged. |
| `OBLIG-CLAIMS-RESOURCE-BOUNDARIES` | **EXTENDED** (not narrowed) | Gains the within-row presentation obligation. Nothing removed. |
| `DIGEST-BEFORE-READ`, `NO-RNG-FIELD`, `CELL-IDENTITY`, `DETERMINISTIC-ORDER` | **UNTOUCHED** | Did not match the enumeration and are byte-unchanged. |

**Mechanically checked:** the set of obligation IDs is identical before and after — 9 old, 9
new, `removed=[]`, `added=[]`. `pv diff` renders each edited obligation as a `-`/`+` pair
because it has no "modified" verb; all four `-` lines have a matching `+` line with the
identical ID. **No obligation was deleted.**

### Falsification tests (6 of 10 matched; 10 of 10 still present)

| Test | Disposition | What changed |
|---|---|---|
| `FALSIFY-CLAIMS-001` | **NARROWED + STRENGTHENED** | Cardinality 80 → 40. Mutation controls go from two to **three**: extra seed, duplicated seed (both retained), plus a NEW added-method mutation — the axis the narrowing moved, and the only direction the narrowing could be silently undone in. |
| `FALSIFY-CLAIMS-004` | **UNCHANGED** | Still asserts a `lora` row's missing attestation block is refused; the row tag stays representable so this test keeps working. |
| `FALSIFY-CLAIMS-006` | **NARROWED** | The pinned tail moves from `lora/s64/seed{37,41,43,47,53}` to `setfit/s64/seed{37,41,43,47,53}`. Head five unchanged. |
| `FALSIFY-CLAIMS-007` | **NARROWED + SCOPE DISCIPLINE ADDED** | Adds the CLAUDE.md rule-4 paragraph: three of the four shapes are RE-MUTATED at the active scope with the same variant tags; the unpaired-selection shape has no active form and stays deferred-scope-only; two new active-scope out-of-scope negatives reuse existing variants. |
| `FALSIFY-CLAIMS-008` | **NARROWED** | 80 → 40, plus a note that the second backstop is the door refusing a declared out-of-scope cell. |
| `FALSIFY-CLAIMS-009` | **SPLIT** | SetFit lock-record negative re-mutated at the active scope; the two LoRA ledger negatives are the deferred-scope forged-provenance shape and keep their variant tags. |
| `002`, `003`, `005`, `010` | **UNTOUCHED** | Byte-unchanged. |

**Mechanically checked:** 10 old, 10 new, `removed=[]`, `added=[]`.

### Kani harness (1 of 1 matched)

| Harness | Disposition | What changed |
|---|---|---|
| `KANI-CLAIMS-001` | **NARROWED** | `bound: 80` → `bound: 40`. The NOT-EXECUTED honesty paragraph is byte-unchanged. |

### `qa_gate`

**NARROWED (text) + provenance guard ADDED.** The description now says "40-cell ACTIVE
expectation set (with the two-method 80-cell scope retained as deferred)". The three MEASURED
mutation transcripts in `qa_gate.falsification` are **preserved verbatim** as the 1.0.0
historical record, under a new paragraph stating plainly that they were measured at 1.0.0
against the 80-cell set and are NOT being restated as if re-run at 2.0.0 — the 2.0.0
re-measurement is recorded in the SUMMARY with its own rc values. Quoting a 1.0.0 transcript
as evidence about a 2.0.0 gate would be the "labelled by intent rather than by measurement"
error the same paragraph already warns about.

## Closed-form derivation check (how-to-verify #2)

40 is a **product**, not a literal. The narrowing was performed by removing an element from
`methods`, so the arithmetic followed:

```
equations.expectation_set.methods = ['setfit']              -> |methods| = 1
equations.expectation_set.shots   = [8, 16, 32, 64]         -> |shots|   = 4
equations.expectation_set.seeds   = [13,17,23,29,31,37,41,43,47,53] -> |seeds| = 10
                                                            1 * 4 * 10 = 40 = expected_cells
```

Machine-verified against the amended file (not transcribed):

```
ACTIVE   ['setfit'] x [8, 16, 32, 64] x 10 = 40
DEFERRED ['setfit', 'lora'] -> 80  status=deferred  ticket=D-ITEM-05-15
```

The `FORTY IS A PRODUCT` invariant, and the shipped parity test that asserts
`expected_cells == |methods| * |shots| * |seeds|`, are what make a hand-written 40 beside an
unchanged two-element method list unrepresentable.

## Why the deferred scope is NESTED rather than a sibling equation

First attempt promoted it to `equations.deferred_two_method_scope`. **Measured, not assumed:**

```
target/release/pv audit contracts/setfit-benchmark-claims-v1.yaml \
  --binding contracts/aprender/binding.yaml > /tmp/p11/audit-probe.log 2>&1; rc=$?
rc=1
Total equations:    11
[ERROR] BIND-001: Equation 'deferred_two_method_scope' in setfit-benchmark-claims... has no binding entry
```

A top-level key would have needed a binding row in `contracts/aprender/binding.yaml` and would
have read as an **eleventh implemented equation** in `make contract-audit-phase5` — a deferred,
unexercised scope presented in the binding ledger as implemented code. Nesting it inside
`expectation_set`, which is the equation it is the deferred counterpart of, keeps the equation
count at 10 and keeps `binding.yaml` (a file outside this plan's declared scope) untouched.

## Verification bundle (rc captured on its own line, never through a pipe)

```
$ target/release/pv validate contracts/setfit-benchmark-claims-v1.yaml
rc=0
0 error(s), 0 warning(s)
Contract is valid.

$ target/release/pv audit contracts/setfit-benchmark-claims-v1.yaml --binding contracts/aprender/binding.yaml
rc=0
Total equations:    10
BIND- lines: 0

$ target/release/pv diff /tmp/p11/claims-old.yaml contracts/setfit-benchmark-claims-v1.yaml
rc=0
Contract diff: v1.0.0 → v2.0.0
Suggested bump: major
```

**On the suggested bump (how-to-verify #5):** `pv diff` suggests **major**, and the prepared
file already carries `version: 2.0.0`. The tool and the prepared edit agree, so there is no
tension to report — and a narrowing of a declared guarantee should not have landed as a patch
bump had it suggested one. The reasoning is recorded in the file's own metadata header, not
only in a commit message: a consumer who previously read `expectation_set` as a two-method
80-cell guarantee gets a *different answer from the same key* after this edit.

## Byte-level control (how-to-verify #6)

`git diff -U0` hunk headers, grouped by enclosing top-level key:

```
  24  equations:
   8  proof_obligations:
   5  falsification_tests:
   3  metadata:
   2  qa_gate:
   1  kani_harnesses:
```

312 insertions, 68 deletions. The diff touches the expectation/scope/statistics regions, the
obligation and test text that names a count or a second method, the Kani bound, the qa_gate
description, and the metadata header. `references:`, `depends_on:`, `contract:`, `created:`,
`author:` and `kind:` are untouched.

## What this bundle does NOT do

- It authorizes **no compute**. 05-12's 40-cell SetFit run is separately NOT pre-authorized;
  05-03's checkpoint deferred that question to wave 7 by explicit human instruction and 05-12
  asks for it on its own.
- It does not run the D-19(b) falsifier (point aprender's loader at the real Qwen3.5-9B
  `config.json`). That is option-c below and remains un-run; D-19(b) is still an inference
  from structural absence, recorded that way in D-ITEM-05-15.
- It commits nothing. Task 2 commits the contract as its own commit, unmixed with code, only
  after an option is selected.
