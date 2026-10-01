---
phase: "05"
slug: "benchmark-and-claims-gate"
status: verified
# threats_open = count of OPEN threats at or above workflow.security_block_on severity (the blocking gate)
threats_open: 0
asvs_level: 1
created: "2026-09-12"
---

# Phase 05 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

**Register origin:** authored at plan time. All 17 plans (05-01 … 05-17) shipped a
`<threat_model>` block, so `register_authored_at_plan_time: true` — this audit VERIFIED
declared mitigations rather than retroactively constructing a STRIDE register.

**Scope note.** This phase's own subject matter is anti-tampering: it builds a claims gate
whose threat model names *the producer who writes the evidence rows* as the adversary. A
mitigation that existed only as a sentence in a SUMMARY would be precisely the verdict
laundering T-05-10-05 describes. Every CLOSED verdict below therefore cites a code
construct, a named passing test, or a committed artifact — not plan prose.

---

## Trust Boundaries

Consolidated from the 17 per-plan `<threat_model>` blocks.

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| producer-written row → claims gate | Every string in `rows/*.json` is attacker-controlled in this gate's own model. The report asserts a doctored cell "would have REFUSED this report". | benchmark evidence: metrics, digests, mechanism strings, attestations |
| benchmark directory → filesystem | The gate opens files at paths it derives from row content. `bench_dir` is operator-supplied and trusted; anything a ROW names is not. | file paths (lock records, candidate ledgers, selection manifests) |
| committed `selections/` tree → claims gate | The 40 manifests are the evidence; an attacker who can write rows may move manifest files between cell directories. | sealed selection manifests (payload + `semantic_hash`) |
| measurement → contract gate | Numbers measured in 05-01 become the calibrated-regime gate's thresholds in 05-03; a wrong number weakens a Phase 3 safety gate. | epsilon windows, noise floors, exceed factors |
| calibration derivation → frozen contract epsilon | The only evidence any production epsilon is legal. A derivation that reports success on an empty basis makes everything downstream unfounded. | `BasisCoverage`, per-class windows |
| persisted evidence bytes → combined verdict | The 12 committed passes are the entire measured record; a change to their canonical bytes invalidates 05-01's cross-process bit-identity proof. | `UpdateEvidence` canonical bytes |
| regime id → threshold selection | A drifted or attacker-shaped regime id must resolve to NO table, never to a wrong one. | architecture fingerprint string |
| Python fixture env → Rust tests | Fixture bytes cross from the pinned uv env into the test suite. | reference statistics (scipy/sklearn), SHA-256 manifests |
| artifact/credential → evaluation | The evaluator door is reachable only through the reload credential. | `ReloadedSetFitCredential`, corpus identity |
| spawned `apr` binary → verdicts | Every verdict comes from a reaped `ExitStatus` of the cargo-built binary. | exit codes, stdout digests |
| gate verdict → published report | The report attests to what it recomputed. A verdict that overstates the checks performed is a repudiation defect, not a cosmetic one. | `verified:` / `residual:` disclosure lines |
| host cache dir → harness | Pinned MiniLM weights; `MiniLmImport::open` verifies tokenizer SHA-256 against the pin. | model weights, tokenizer |

---

## Threat Register

115 threats, `T-05-01-01` … `T-05-17-08`. All CLOSED. Severities marked `*` were assigned by
the auditor — plans 05-01, 05-02 and 05-04…05-10 shipped the older 5-column table with no
severity column.

### 05-01 — production calibration harness

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-05-01-01 | Tampering | proposed regime entry string | high* | mitigate | `thresholds.rs:82-104` — architecture token byte-copied from a rendered run id; `covers` forbids prefix aliasing/normalization | closed |
| T-05-01-02 | Elevation (gate loosening) | ε derivation | critical* | mitigate | `evidence.rs:2946` `assert!(control_max < real_min)`; `rounding_noise_floor()` `:1634`; single strict `lower < upper` at `:2040-2048` | closed |
| T-05-01-03 | Spoofing (wrong model) | production checkout | high* | mitigate | `evidence.rs:3382` `from_pretrained_dir`; pin enforced at `aprender-core/src/setfit/import.rs:479-483` | closed |
| T-05-01-04 | DoS (compute overrun) | boundary matrix | low* | mitigate | `evidence.rs:3715` timed probe + `:3754` probe-first doc + committed compute projection | closed |
| T-05-01-SC | Tampering | package installs | low* | accept (n/a) | verified truthful — the five 05-01 commits touch only `evidence.rs` | closed |

### 05-02 — regime-keyed thresholds

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-05-02-01 | Tampering (gate weakening) | `table_for` lookup | high* | mitigate | `thresholds.rs:486-488` — `is_calibrated` IS `table_for(..).is_some()`; one source of truth | closed |
| T-05-02-02 | Elevation | `validate_evidence` | high* | mitigate | `thresholds.rs:469-472` single resolution site; `:500-509` regime-less accessors **removed**, not merely gated | closed |
| T-05-02-03 | Repudiation | list vs tables drift | medium* | mitigate | `thresholds.rs:496-498` derived from `self.regimes`; both directions pinned at `:828` | closed |
| T-05-02-SC | Tampering | package installs | low* | accept (n/a) | no manifest touched | closed |

### 05-03 — production epsilon calibration

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-05-03-01 | Tampering (gate loosening) | `calibrated_regimes` list | critical | mitigate | `thresholds.rs:643-646` literal `2` + anti-third-entry rationale; `:667-673` full-list **string equality** (CR-04) | closed |
| T-05-03-02 | Tampering | fixture regime semantics | high | mitigate | `thresholds.rs:690-740` per-regime loop, non-vacuity asserted first, every class compared per regime | closed |
| T-05-03-03 | Spoofing | production regime id | high | mitigate | `thresholds.rs:743-790` all 40 cells resolve the PRODUCTION table via `render_run`; alias prohibition `:95-100` | closed |
| T-05-03-04 | Repudiation | edit provenance | medium | mitigate | `contracts/setfit-train-lifecycle-v1.yaml:3-12` — `pv diff` invocation + verbatim `Suggested bump: major`; one commit `a63bb130b` | closed |
| T-05-03-05 | Tampering (rule chosen to close the table) | replacement lower bound | critical | mitigate | `evidence.rs:1690-1730` `LowerBound`; `:2212-2216` candidates derived by **calling** `epsilon_basis`; `:2277-2308` L3 factors printed; human selection at contract `:277-280`. **Caveat — see Warning 2** | closed |
| T-05-03-06 | Repudiation (weakened claim recorded as unchanged) | lower-edge invariants | high | mitigate | contract `:305-322` — "THIS IS A WEAKENING … WHAT IS GIVEN UP" names the run that would now pass and the three adversaries still refused, with margins | closed |
| T-05-03-07 | Elevation (exemption by omission) | `attention_key_bias` disposition | medium | mitigate | contract `:250,:286,:327` measured numbers + `:1290-1293` obligation; `evidence.rs:2541` proves all classes required with no table | closed |
| T-05-03-SC | Tampering | package installs | low | accept (n/a) | no manifest touched | closed |

### 05-04 — claims numerics substrate

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-05-04-01 | Tampering | `claims_stats` fixtures | medium* | mitigate | `scripts/setfit_fixtures/claims_stats/manifest.sha256` (5 entries); `tests_claims_stats.rs:23-27` `include_str!` pins bytes at compile time | closed |
| T-05-04-02 | Tampering (RNG smuggling) | paired stats path | high* | mitigate | `tests_claims_stats.rs:611-637` `no_rng_enters_the_claims_statistics_path` over non-comment source (`rand::`, `thread_rng`, `bootstrap`, `resample`, `shuffle`) | closed |
| T-05-04-03 | Repudiation | constant provenance | medium* | mitigate | `hypothesis.rs:246` `T_CRIT_975_DF9`; equality vs the scipy-recorded fixture at `tests_claims_stats.rs:88-90`, red on drift | closed |
| T-05-04-04 | Tampering | Phase 1 fixture corpus | medium* | mitigate | new dir + own manifest; pre-existing corpora separate and untouched by 05-04's commits | closed |
| T-05-04-SC | Tampering | package installs | low* | accept (n/a) | no manifest touched | closed |

### 05-05 — claims contract + bench row

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-05-05-01 | Tampering | row files | high* | mitigate | `bench_row.rs:580-605` — `from_bytes` recomputes and compares before returning `Self` | closed |
| T-05-05-02 | Repudiation (selective omission) | expectation set | high* | mitigate | `bench_row.rs:109` derived `EXPECTED_CELLS`; `:162` contract via `include_str!`; parity test with three **inverting** mutations. Scope is the 05-11 ACTIVE 40 | closed |
| T-05-05-03 | Tampering (RNG smuggling) | row schema | high* | mitigate | 14 × `#[serde(deny_unknown_fields)]`; contract textually excludes bootstrap fields | closed |
| T-05-05-04 | Spoofing | method identity | medium* | mitigate | `bench_row.rs:453-465` externally-tagged `MethodEvidence` — a lora row cannot present setfit evidence | closed |
| T-05-05-05 | Tampering | duplicate cell recording | medium* | mitigate | `record()` `:954` + typed `CellDigestCollision` `:1085/:1165`; test `bench_row_tests.rs:680` | closed |
| T-05-05-06 | Repudiation (unprovable selection safety) | LoRA no-selection attestation | medium | mitigate + accept residual | ledger digest + line count recomputed `bench_gate.rs:1567-1600`; residual verbatim in contract `:624` → **AR-01** | closed |
| T-05-05-07 | Spoofing (incomparable size claim) | `artifact_bytes` | medium | mitigate | `bench_row.rs:393` mandatory `deployable_total_bytes`; renderer test `..._never_the_adapter_only_figure` | closed |
| T-05-05-SC | Tampering | package installs | low* | accept (n/a) | no manifest touched | closed |

### 05-06 — LoRA selection manifest + reload tracer

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-05-06-01 | Tampering | selection manifest | high* | mitigate | `manifest.rs:248-268` digest-verifies before return; `finetune.rs:1808-1815` typed refusal before training starts | closed |
| T-05-06-02 | Elevation (unlocked model selection) | early stop / best-epoch | high* | mitigate | `finetune.rs:1739-1752` patience 0 ⇒ `save_every` disabled; `epochs_completed` enforced by the gate | closed |
| T-05-06-03 | Spoofing (wrong seed) | `TrainingConfig.seed` | medium* | mitigate | `finetune.rs:1788-1793` — `--selection-manifest` without `--seed` is a typed refusal | closed |
| T-05-06-04 | Repudiation | which rows trained | medium* | mitigate | `finetune.rs:98` run output carries the manifest hash the row records | closed |
| T-05-06-05 | Spoofing (silently-unapplied adapter) | adapter reload route | high | mitigate | `classify_pipeline/mod.rs:1195-1250` shape-check-before-install, typed `ConfigError`, never partial; two-sided control `classify_reload_tests.rs:341-358` | closed |
| T-05-06-SC | Tampering | package installs | low* | accept (n/a) | no manifest touched | closed |

### 05-07 — F-10 end-to-end proof

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-05-07-01 | Spoofing (wrong binary) | spawned ladder | high* | mitigate | `setfit_cli_lifecycle.rs:115` `env!("CARGO_BIN_EXE_apr")`, one spawn site `:243`, source guard `:1991`, pinned path asserted a file `:477` | closed |
| T-05-07-02 | Tampering (vacuous gate) | test renames | medium* | mitigate | `Makefile:2485-2494` `assert_tests_ran` — refuses a name filter that matched nothing (CR-02) | closed |
| T-05-07-03 | Repudiation | artifact identity | medium* | mitigate | `setfit_cli_lifecycle.rs:782-797` `hex64()` shape-check first, then equality across four surfaces `:988+` | closed |
| T-05-07-04 | Elevation | slice refusal weakening | high* | mitigate | `probe_unicode` refusal suites intact — 21 occurrences; zero assertion changes in the 05-07 commits | closed |
| T-05-07-SC | Tampering | package installs | low* | accept (n/a) | no manifest touched | closed |

### 05-08 — per-row evaluator + metric assembly

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-05-08-01 | Elevation (bypass) | evaluator door | high* | mitigate | **one** construction site `apr_reload.rs:381`; sealed trait `credential.rs:129`; trybuild compile-fail; both doors take `&ReloadedSetFitCredential` | closed |
| T-05-08-02 | Tampering | label attribution | high* | mitigate | `..._changes_attribution_in_both_directions`, `..._is_exact_byte_equality`, `..._is_evidence_from_the_pinned_dataset_revision` | closed |
| T-05-08-03 | Tampering (RNG) | calibration inputs | high* | mitigate | `bench_metrics_does_not_import_the_resampling_evaluator` | closed |
| T-05-08-04 | Information disclosure (split leakage) | calibration split | high* | mitigate | `..._calibration_split_is_validation_and_test_probabilities_cannot_reach_it`; `CALIBRATION_SPLIT = "validation"` structural | closed |
| T-05-08-SC | Tampering | package installs | low* | accept (n/a) | no manifest touched | closed |

### 05-09 — `apr setfit bench run`

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-05-09-01 | Spoofing | backend identity | medium* | mitigate | `setfit_bench.rs:1387` identity read from the encode invocation; `..._cannot_fabricate_a_gpu` | closed |
| T-05-09-02 | Tampering | transported rows | high* | mitigate | `..._refuses_a_doctored_digest`, `..._refuses_a_filename_that_disagrees_with_the_payload` | closed |
| T-05-09-03 | Tampering (uncontracted cell) | seed/shots inputs | medium* | mitigate | `setfit_bench.rs:203-211` refusal names the contract and calls out 42 explicitly | closed |
| T-05-09-04 | Repudiation | resource numbers | medium* | mitigate | `setfit_bench.rs:377-419` four mandatory mechanism strings; `..._names_all_four_mechanism_strings` | closed |
| T-05-09-05 | DoS | oversized reads | medium* | mitigate | `setfit_bench.rs:257` `read_bounded` — declared length, then `take(cap+1)` | closed |
| T-05-09-06 | Repudiation (mislabelled resource numbers) | cold latency / peak RSS | high | mitigate | committed rows carry separate train/inference mechanisms + sample interval; `..._sampled_mechanism_always_carries_a_nonzero_interval` | closed |
| T-05-09-07 | Elevation (undeclared 2nd candidate) | candidate ledger | medium | mitigate | `bench_gate.rs:1573-1600` digest + line count recomputed; `..._a_ledger_carrying_a_second_candidate_the_row_does_not_declare` | closed |
| T-05-09-08 | Tampering (concurrent writers) | run-manifest / bench dir | medium* | mitigate | `run_bench_cells.sh:160-170` `noclobber` pid lock, stale lock **reported not stolen**; `:16-22` no parallel construct; atomic rename `setfit_bench.rs:1171` | closed |
| T-05-09-SC | Tampering | package installs | low* | accept (n/a) | verified in the diff: `159f8ca25` adds `sysinfo = { workspace = true }` + one lock edge, **no new `[[package]]` block** — the only manifest touch in all 136 phase-05 commits | closed |

### 05-10 — claims gate + bench report

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-05-10-01 | Tampering | doctored rows | critical* | mitigate | four default-suite negatives: edited payload bytes, trimmed evidence block, substituted row, wrong slot | closed |
| T-05-10-02 | Repudiation (selective omission) | expectation set | high* | mitigate | `bench_gate.rs:1071-1085` expectation compared **before** any row read; three negatives incl. `..._expectation_set_is_not_the_contracted_forty` | closed |
| T-05-10-03 | Elevation (post-test selection) | lock / attestation rules | high* | mitigate | `bench_gate.rs:1541-1563`; `..._every_conjunct_of_the_lora_attestation_separately` | closed |
| T-05-10-04 | Tampering (RNG) | aggregation | high* | mitigate | `..._verified_run_set_has_no_public_constructor`, `..._no_rng_or_resampling_vocabulary`, `..._two_aggregations_of_one_run_are_bit_identical` | closed |
| T-05-10-05 | Information disclosure (verdict laundering) | report language | medium* | mitigate | `..._prints_no_verdict_word_and_names_the_interval`; p-values `detail`-only; committed `report.md` states "No binary verdict is printed" | closed |
| T-05-10-06 | Repudiation (forged provenance) | lock / ledger evidence | medium | mitigate + accept residual | `bench_gate.rs:1519-1600` digests recomputed from committed bytes; residual in module doc, contract and report → **AR-02** | closed |
| T-05-10-07 | Information disclosure (misleading like-for-like) | resource + size rendering | medium | mitigate | `..._labels_a_mixed_mechanism_comparison_and_leaves_a_matched_one_alone`; `bench_gate_mechanism_class_case_table` | closed |
| T-05-10-SC | Tampering | package installs | low* | accept (n/a) | no manifest touched | closed |

### 05-11 — 40-cell active-scope retarget

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-05-11-01 | Elevation of privilege | the expectation-set edit | critical | mitigate | `05-11-narrowing-inventory.md` — SHA-256 of both files, per-clause RETAINED/NARROWED/SPLIT rows, `pv diff` of two filesystem paths; prepared uncommitted then committed after the human checkpoint | closed |
| T-05-11-02 | Tampering | gate strength during retarget | high | mitigate | `Makefile:2660-2710` enumerates 23 doctored shapes **by the scope each is re-mutated at**, citing Verification Discipline rule 4 | closed |
| T-05-11-03 | Information disclosure | a report implying a comparison never run | high | mitigate | `setfit_bench_tests.rs:1931-1980` must-match/must-not-match table with a **two-way non-vacuity** leg | closed |
| T-05-11-04 | Repudiation | deletions passed off as a scope amendment | medium | mitigate | inventory rows `RETAINED-AS-DEFERRED`, `removed=[]`; contract `:284` keeps the 80-cell product intact | closed |
| T-05-11-05 | Spoofing | a green gate over a shrunken suite | high | mitigate | `Makefile:2656/:2714` floors **raised** to 27/55 from 24/31; `assert_tests_ran` is the induced-red control | closed |
| T-05-11-06 | Information disclosure | single-cell door read as a partial report | medium | mitigate | `..._emits_none_of_the_report_statistic_literals`, `..._scope_note_enumerates_every_recomputation`; clap `conflicts_with_all` | closed |
| T-05-11-07 | Tampering | a future caller widening the set | medium | mitigate | `bench_row.rs:127-128` — the `DeferredTwoMethod` **variant itself** is `#[cfg(test)]`; `verify_run` two-argument with inline rationale | closed |
| T-05-11-08 | Repudiation | stale banner misdescribing a gate | medium | mitigate | `Makefile:2643-2712` banners describe the ACTIVE 40-cell scope | closed |
| T-05-11-SC | Tampering | package installs | low | accept (n/a) | no manifest touched | closed |

### 05-12 — the 40-cell evidence set

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-05-12-01 | DoS | the 40-cell sweep | medium | mitigate | projection committed before any cell (`0b62da726`); hash-based resume; `assert_disk_headroom` | closed |
| T-05-12-02 | Repudiation | contended measurements | high | mitigate | `run_bench_cells.sh:16-22` no dispatch construct; `:160-170` pid lock | closed |
| T-05-12-03 | Information disclosure | dataset text in committed evidence | high | mitigate | **independently confirmed** — the 40 committed rows carry only hashes, numbers and three class names | closed |
| T-05-12-04 | Spoofing | backend identity | medium | mitigate | rows carry `backend_identity: cpu:setfit-core:autograd-trueno-matmul`, read from execution | closed |
| T-05-12-05 | Tampering | late-discovered per-row gate defect | high | mitigate | pilot commit `ea842b6b6` drove both halves of the real gate before the 39 (`86d3d4c34`) | closed |
| T-05-12-06 | Repudiation | mechanism asymmetry averaged away | medium | mitigate | declared per row; `report.md` renders three distinct sample intervals, never one pooled figure | closed |
| T-05-12-07 | Tampering | divergent pilot invocation | low | mitigate | `run_bench_cells.sh:374` SKIP by row digest | closed |
| T-05-12-SC | Tampering | package installs | low | accept (n/a) | no manifest touched | closed |

### 05-13 — report + closing audit

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-05-13-01 | Repudiation | recomputability | high | mitigate | two-run sha equality on both artifacts, `cmp` rc=0, **comparator non-vacuity proven** (rc=1 vs a one-byte-longer copy); all inputs committed | closed |
| T-05-13-02 | Tampering | negative controls | medium | mitigate | destructive controls on `/tmp` copies only, `PATHS_DIFFER: yes`, baseline rc=0 before mutation; real directory re-verified | closed |
| T-05-13-03 | Elevation of privilege | the requirements table | high | mitigate | **independently confirmed** — `git diff 86d3d4c34..65cdade8b -- .planning/REQUIREMENTS.md` is empty | closed |
| T-05-13-04 | Information disclosure | a comparison implied by an artifact | high | mitigate | committed `report.md` carries "SCOPE — ONE METHOD WAS MEASURED … Read the absence as absence"; render-time table + structural refusal | closed |
| T-05-13-05 | Information disclosure | a misleading resource figure | medium | mitigate | committed report renders both peaks separately with mechanism strings and the `[sampled_lower_bound]` label | closed |
| T-05-13-06 | Spoofing | KNOWN-RED misread as regression | low | mitigate | each standing red named, classified and given a measured reason | closed |
| T-05-13-SC | Tampering | package installs | low | accept (n/a) | no manifest touched | closed |

### 05-14 — fail-closed epsilon verdict

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-05-14-01 | Tampering (gate made vacuous) | the fail-closed verdict | critical | mitigate | `evidence.rs:2314-2326` explicit "no `#[ignore]`, no env gate"; two-sided `:2379`/`:2406`; real-data RED control `:2768` red **by construction**, asserting the exact five classes and 05-01's exceed factors | closed |
| T-05-14-02 | Elevation (exemption by omission) | required/gated set | high | mitigate | `evidence.rs:2022-2029,2049-2052` read from `table_for`; `None => true`; no allowlist/constant/env override; declared-ungated rows still print | closed |
| T-05-14-03 | Tampering | persisted canonical bytes | high | mitigate | `evidence.rs:2697-2754` — pair count asserted **first** (`== 12`, anti-vacuity message), then SHA-256 per pair, then round-trip | closed |
| T-05-14-04 | Repudiation (report outruns evidence) | basis coverage | medium | mitigate | `evidence.rs:1798-1812` typed `BasisCoverage::{Complete, Provisional{..}}`, not banner prose | closed |
| T-05-14-05 | Information disclosure (guard tuned until green) | fixture-matrix green control | medium | mitigate | `05-14-controls.md:138-139` — both branches pre-specified; record states "Neither documented refusal branch fired" | closed |
| T-05-14-SC | Tampering | package installs | low | accept (n/a) | no manifest touched | closed |

### 05-15 — evidence-path containment

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-05-15-01 | Tampering | `verify_provenance` lock-path resolution | critical | mitigate | `resolve_committed_evidence_path`, `bench_gate.rs:857-925` — stage 1 refuses empty/absolute/`ParentDir`/`RootDir`/`Prefix` before any syscall; stage 2 canonicalizes both and compares component-wise. Swept by 13 path cases × 2 kinds | closed |
| T-05-15-02 | Information disclosure | arbitrary read via lock / ledger path | high | mitigate | **both arms** through the same helper (`:1519`, `:1567`); 16 MiB two-stage cap | closed |
| T-05-15-03 | Repudiation | the "recomputed from committed bytes" attestation | high | mitigate | **verified by exhaustion** — all four `read_evidence` call sites enumerated; the only `fs::metadata`/`File::open` in the file are inside `read_evidence`. No bypass exists | closed |
| T-05-15-04 | Spoofing | foreign `contract_id` | medium | mitigate | row `:1265-1272` and manifest `:1316-1327`, `RowSchemaRefused` naming both ids | closed |
| T-05-15-05 | Tampering | symlink escaping `bench_dir` | high | mitigate | three case-table rows: last-component, first-component-dir, prefix-sibling | closed |
| T-05-15-06 | DoS | canonicalize on a slow/unavailable mount | low | **accept** | → **AR-03** | closed (accepted) |
| T-05-15-07 | Elevation of privilege | n/a — read-only library call | low | **accept** | → **AR-04**; verified truthful | closed (accepted) |
| (05-15 SC) | Tampering | package installs | low | n/a — absence recorded | the plan records the absence explicitly rather than emitting a vacuous row; confirmed by the manifest diff | closed |

### 05-16 — selection-manifest binding

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-05-16-01 | Tampering | `selection_manifest_hash` never recomputed | high | mitigate | `bench_gate.rs:1683-1733` `verify_selection_binding`; exact byte equality `:1713`; 11 case-table rows asserted | closed |
| T-05-16-02 | Tampering | manifest transplanted from another cell | high | mitigate | `:1721-1731` `shots_per_class`/`root_seed` vs the cell key, typed `SelectionManifestCellMismatch`; three transplant rows + an inverting row | closed |
| T-05-16-03 | DoS (of evidence) | `selections/` deleted | high | mitigate | `read_evidence(.., SelectionManifest)` → `EvidenceFileMissing` `:938-944` | closed |
| T-05-16-04 | Spoofing | manifest edited without resealing | medium | mitigate | delegated to `SelectionManifest::from_bytes`, **not** reimplemented — held by the OPS-03 forbidden-symbol guard `:3175-3183` | closed |
| T-05-16-05 | Repudiation | report implying an unenforced binding | high | mitigate | committed `report.md` `verified:` block names all three recomputations | closed |
| T-05-16-06 | Information disclosure | manifest path escaping `bench_dir` | low | **accept** | → **AR-05**; structurally absent **and guarded** — a test asserts the signature string verbatim | closed (accepted) |
| T-05-16-07 | Elevation of privilege | n/a — read-only library call | low | **accept** | → **AR-06** | closed (accepted) |
| (05-16 SC) | Tampering | package installs | low | n/a — absence recorded | confirmed by the manifest diff | closed |

### 05-17 — closed-form quality cross-check

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-05-17-01 | Tampering | `quality.f_avg` doctored with digests repaired | high | mitigate | `bench_gate.rs:1770-1843` recomputes from the confusion matrix through shipped surfaces, compares by `to_bits()`; 12 case-table rows asserted | closed |
| T-05-17-02 | Tampering | `*_bits` doctored either way | medium | mitigate | `:1815-1842` — **all five** bits siblings, including the two calibration ones | closed |
| T-05-17-03 | Tampering | `n_test_rows` inflated | medium | mitigate | `:1788-1795` compared against the matrix's own total, checked first | closed |
| T-05-17-04 | Repudiation | the `residual:` line over/understating | high | mitigate | all three statements present and matching (module doc, `RESIDUAL_DISCLOSURE`, contract `amended_4_0_0`); gated by the case table + overclaim scan | closed |
| T-05-17-05 | Tampering | the confusion matrix itself doctored | high | **accept** | → **AR-07**; disclosed in **four** places | closed (accepted, disclosed) |
| T-05-17-06 | Tampering | calibration diagnostics doctored | medium | **accept** | → **AR-08**; same four places, clause (2) | closed (accepted, disclosed) |
| T-05-17-07 | DoS | enormous confusion matrix expanded | low | mitigate | plan's stated mitigation did not reach it (deviation 1); a real cap shipped — `bench_metrics.rs:383` `MAX_CROSS_CHECK_ROWS`, `:458-468` saturating + short-circuiting | closed |
| T-05-17-08 | Elevation of privilege | n/a — read-only library call | low | **accept** | → **AR-09** | closed (accepted) |
| (05-17 SC) | Tampering | package installs | low | n/a — absence recorded | confirmed by the manifest diff | closed |

*Status: open · closed · open — below high threshold (non-blocking)*
*Severity: critical > high > medium > low — only open threats at or above `workflow.security_block_on` count toward `threats_open`*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-01 | T-05-05-06 | Residual after mitigation: a producer controlling **both** the candidate ledger and the rows can forge them consistently. Named in-contract (`setfit-benchmark-claims-v1.yaml:624` `residual_risk.statement`) rather than hidden. Closing it needs an evidence producer outside the producer's control. | Phase 05 plan author (05-05) | 2026-09-12 |
| AR-02 | T-05-10-06 | Same residual class at the gate: digests are recomputed from committed bytes, but consistent forgery by a producer controlling both lock and ledger remains reachable. Stated in three places — module doc, contract, and the report's `residual:` line. | Phase 05 plan author (05-10) | 2026-09-12 |
| AR-03 | T-05-15-06 | DoS via canonicalizing a path on a slow or unavailable mount. The gate reads ~123 small committed JSON files under a local benchmark directory; no network path is in scope and the 16 MiB read cap bounds size. Severity low — below the `high` block threshold. | Phase 05 plan author (05-15) | 2026-09-12 |
| AR-04 | T-05-15-07 | No privilege-transition surface: no setuid, no process spawn, no shell interpolation of row content. Verified truthful — zero `Command`/`process::`/`exec`/`Stdio`/`unsafe` in `bench_gate.rs`, `bench_metrics.rs`, `bench_row.rs`; workspace sets `unsafe_code = "forbid"`. | Phase 05 plan author (05-15) | 2026-09-12 |
| AR-05 | T-05-16-06 | The selection-manifest path is DERIVED from the cell key and contains no row-supplied component, so the path-escape attack has no surface. `selection_manifest_path` takes no `BenchRow` — asserted verbatim by a structural guard. | Phase 05 plan author (05-16) | 2026-09-12 |
| AR-06 | T-05-16-07 | Same as AR-04 — read-only library call, no privilege transition. | Phase 05 plan author (05-16) | 2026-09-12 |
| AR-07 | T-05-17-05 | **The material accepted risk of this phase.** If the confusion matrix ITSELF is doctored and the metrics are recomputed from it consistently, nothing committed can detect it — no committed file carries the per-row predictions. Closing it requires a new evidence artifact (committed per-row predictions), out of this round's scope. DISCLOSED in four places rather than smuggled in as a silent gap. | Phase 05 plan author (05-17) | 2026-09-12 |
| AR-08 | T-05-17-06 | `ece_top_label_validation` / `brier_multiclass_validation` are not recomputable — the calibration diagnostics need per-row probability vectors no committed file carries. Disclosed in the corrected residual, recorded as class (iii) in 05-15's enumeration with its reason. | Phase 05 plan author (05-17) | 2026-09-12 |
| AR-09 | T-05-17-08 | Same as AR-04 — read-only library call, no privilege transition, no process spawn from row content. | Phase 05 plan author (05-17) | 2026-09-12 |
| AR-10 | `T-05-NN-SC` ×14 | The fourteen per-plan package-supply-chain rows are `accept (n/a)`: no package-manager install ran. Verified in the diff, not taken on trust — across all 136 phase-05 commits the only Cargo manifest touch is `159f8ca25`, which adds `sysinfo = { workspace = true }` and one Cargo.lock dependency edge, with **no new `[[package]]` block**. Plans 05-15/16/17 record the absence in prose instead of emitting a vacuous row — the honest form. | Auditor (verified) | 2026-09-12 |
| AR-11 | (unregistered — disclosed in source, not a plan register row) | **TOCTOU between containment check and open.** `resolve_committed_evidence_path` checks containment and the file is then opened, so a symlink swapped between the two calls would be read (`bench_gate.rs:843-848`). The source states the residual rather than hiding it. Accepted because the race needs WRITE access to the benchmark directory *during* verification — strictly more than the producer-written-tree adversary this gate is built for. Surfaced by UAT test 4 (2026-09-13), which named it as a required element of this record; it was absent from all 17 plan `<threat_model>` blocks, so the plan-time register could not carry it. | Auditor + UAT test 4 | 2026-09-13 |

---

## Unregistered Threat Flags

| Flag | Source | Disposition |
|------|--------|-------------|
| `CanonicalTestGrant` travelling to a different credential | surfaced in 05-08-SUMMARY as "an additional surface the plan did not enumerate" | Mitigation shipped with it — `TestGrantArtifactMismatch` in `apr_evaluate.rs`. Informational. |

---

## Warnings — 3, none blocking

**1. ~~`make setfit-bench-door-probe` is an orphaned target.~~ — RESOLVED 2026-09-13 in `b5bd11a65`.** `Makefile:2790` defines it; it is
a prerequisite of nothing and absent from `.PHONY` (`Makefile:60`) — the only occurrence of the
string in the file is its own definition. The phase's own `05-VERIFICATION.md:469,484` already
grades it "PRESENT, NOT RUNNABLE AT HEAD / ORPHANED TARGET (WR-07)" and records rc=1 at HEAD on
`apr_bin.sh`'s binary pin. It is the **secondary** leg of T-05-15-01/-03, T-05-16-01/-03 and
T-05-17-01. Each of those threats' **primary** mechanism is a library construct plus a case
table wired into `make setfit-bench-tests`, which `tier3` invokes at `Makefile:408` — so no
threat opens. But the door-level proof those five threats cite does not run in any gate.

> **Closed 2026-09-13.** `/gsd-validate-phase 05` found the same orphan independently and fixed it:
> a new `setfit-bench-door-probe-build` target builds the release `apr` with the non-default
> `setfit` feature and then runs the probe, wired into **tier4** (`Makefile:945`) unprefixed so a
> failure fails the tier. Both targets are now in `.PHONY`. Re-measured at this HEAD: `make
> setfit-bench-door-probe-build` → **rc=0**, with the positive CONTROL passing *before* the four
> attacks (path escape, deleted `selections/`, zeroed pairing key, `f_avg=0.99`), each **rc=5** and
> the last two explicitly refused by the recomputation **and NOT at the row digest**. The secondary
> leg of T-05-15-01/-03, T-05-16-01/-03 and T-05-17-01 now runs in a gate.

**2. T-05-03-05's success-value leg is `#[ignore]`d.** `PRODUCTION_LOWER_BOUND`
(`evidence.rs:1749`) has exactly one caller, `:4221`, inside the `#[ignore]`d production combine.
The clause "every frozen epsilon must make 05-14's fail-closed derivation return the success
value" is proven only by an explicit `--ignored` run, not by the default suite. The mitigation's
other three components (candidates derived by running the shipped combine, L3 factors printed,
human selection recorded) are in-tree and default-suite.

**3. 05-04 and 05-14 SUMMARY files carry no `## Threat Flags` heading.** Their 11 threats are
discharged in body prose under other headings. All 11 were verified against code rather than
prose in this audit, and all 11 hold. This is a workflow-conformance gap, not a security gap —
but a future `/gsd-secure-phase` run that reads only the heading would see nothing there.

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-09-12 | 115 | 115 | 0 | gsd-security-auditor (ASVS L1, block_on: high) |
| 2026-09-13 | 115 | 115 | 0 | UAT test 4 amendment — AR-11 (TOCTOU residual) recorded as an accepted risk; register count unchanged |
| 2026-09-13 | 115 | 115 | 0 | **State A re-audit** (`/gsd-secure-phase 05`) — carried forward on a proven-empty diff; Warning 1 closed |

**Method.** Register parsed from all 17 plan `<threat_model>` blocks and cross-read against the
17 SUMMARY discharge sections. Verification was against the implemented code — `bench_gate.rs`,
`bench_metrics.rs`, `bench_row.rs`, `thresholds.rs`, `evidence.rs`, `apr_evaluate.rs`,
`apr_reload.rs`, `credential.rs`, `setfit_bench.rs`, `finetune.rs`, `classify_pipeline/mod.rs`,
the contracts, the Makefile gates, `run_bench_cells.sh`, and the committed 40-row evidence set —
not against plan prose. Four claims were confirmed independently of the summaries that asserted
them: the dataset-text absence in committed rows (T-05-12-03), the empty REQUIREMENTS.md diff
(T-05-13-03), the exhaustive `read_evidence` call-site enumeration (T-05-15-03), and the
single-manifest-touch finding across 136 commits (the SC rows / AR-10).

---

---

## Security Audit 2026-09-13 (State A re-audit)

| Metric | Count |
|--------|-------|
| Threats found | 115 |
| Closed | 115 |
| Open | **0** |
| Accepted risks logged | 11 (AR-01 … AR-11) |

**Why this re-audit did not re-verify 115 mitigations one by one, and why that is sound
rather than lazy.** Three commits landed since the 2026-09-12 audit — `b5bd11a65`
(Makefile + `binding.yaml`), `824e96941` (VALIDATION.md) and `46bb8b102` (this file). The
question a re-audit has to answer is whether any of them moved code a verdict depends on.
Measured, not assumed:

| Check | Result |
|---|---|
| `git diff --stat 71948eccc..HEAD -- <the 15 files carrying the 115 mitigations>` | **0 bytes** |
| `git diff --stat 71948eccc..HEAD -- benchmarks/tweeteval-stance/` (the committed evidence set) | **0 bytes** |
| `git diff --name-only 71948eccc..HEAD -- 'crates/*/src/*'` | **empty** |
| `git status --porcelain` on those surfaces | **clean** |

Every verdict in the register above therefore points at byte-identical code. Re-running the
auditor over unchanged bytes would produce the same answers at real cost — the short-circuit
the workflow allows at `asvs_level: 1` with `threats_open: 0` is *earned here* by that
measurement, where on 2026-09-12 it was not (nothing had been verified yet, so the auditor ran).

**Warning status carried from the 2026-09-12 audit:**

| # | Warning | Status |
|---|---------|--------|
| 1 | `setfit-bench-door-probe` orphaned — the door-level proof of five threats ran in no gate | ✅ **CLOSED** in `b5bd11a65`; gate re-measured rc=0 at this HEAD |
| 2 | T-05-03-05's frozen-epsilon success leg is `#[ignore]`d | ⚠️ **STANDS** — `PRODUCTION_LOWER_BOUND` still has exactly one caller (`evidence.rs:4221`), inside the ignored combine |
| 3 | 05-04 and 05-14 SUMMARYs carry no `## Threat Flags` heading | ⚠️ **STANDS** — both still 0; their 11 threats were verified against code on 2026-09-12 and hold |

**New this round:** AR-11. UAT test 4 named four items this record had to cover; three were
present and the fourth — the TOCTOU residual at `bench_gate.rs:843-848` — was not, because it
is disclosed in the source module doc but appears in none of the 17 plan `<threat_model>`
blocks. A register derived from plan-time threat models structurally cannot contain it. That
limitation is now stated in this file rather than left for the next reader to rediscover.

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Amendment 2026-09-13 (UAT test 4).** Test 4 required this record to cover four specific
items. Three were present; the fourth — the TOCTOU residual the resolver discloses at
`bench_gate.rs:843-848` — was **not**, because it appears in the source module doc but in none
of the 17 plan `<threat_model>` blocks, so a register built from plan-time threat models could
not contain it. Recorded as **AR-11**, and `resolve_committed_evidence_path` is now named
explicitly in T-05-15-01's mitigation cell. This is a real limitation of plan-time register
derivation worth stating plainly: a residual disclosed only in code is invisible to an audit
that reads only plans.

**Approval:** verified 2026-09-12; amended 2026-09-13
