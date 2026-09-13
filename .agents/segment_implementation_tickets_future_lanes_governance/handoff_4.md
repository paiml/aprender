# Summary

This candidate review evaluates **Segment 2** (`implementation_tickets_future_lanes_governance`) of `docs/specifications/PP-066-release-spec.md` (v1.5, 2026-09-05). The audited segment encompasses:
- **§5 0.66 tickets — paiml-implement units** (lines 167–447): Track I (master rows 0a–0e, 1, 3, 6, 7, 9, 10, 12, 14, 15, 18), Track R (R-0..R-7), Track P (P-0.1..0.6, P-1.1, P-1.2), Track S (S-1..S-3), Track T (T-0h, T-1, T-2, T-0, T-3), Track B (B-A1, B-G1), Track G (G-1..G-4), Track D (D-1).
- **§6 0.67 lane — carried rows** (lines 449–470): W-B kernel, W-D, W-E, W-G, W-H, W-F attach, T-1..T-5, T-7 arm, B-W0, B-W1..W5, B-M1..M4, B-S1..S4, B-A2, PVI 1.3–4.3, §6.7 renames, bar gating.
- **§7 Registered predictions** (lines 472–496): protocol, predictions table, kill-if criteria, hysteresis dead bands.
- **§8 Refusals (0.66)** (lines 498–526): 23 negative requirements and invariants.
- **§9 Toyota Way targets (0.66)** (lines 528–559): jidoka, poka-yoke, andon, genchi genbutsu, kaizen, heijunka, instrument-first, five whys.
- **§10 Verification ledger for this document** (lines 561–586): claims, marks (`[V]`, `[C]`, `[A]`, `[U]`), and authoritative references.
- **§11 Adjudication of `0_66-review.md`** (lines 588–640): 33 review item dispositions and 8 unraised findings.
- **§12 Prior-art register — GPU discovery in three shipped systems** (lines 642–654): llama.cpp/ggml, Ollama, llamafile comparative matrix.
- **Appendix A — Changelog** (lines 655–665): versions 1.0 through 1.5.

The review was conducted across the three mandatory dimensions:
1. **Spec Compliance**: Verification of contract-per-card discipline (kind: pattern, no `registry: true`, executable tests in `check_contract_test_binding.sh`), acceptance command (`A_i`) executability, mutation-to-RED discriminators, quorum classifications, dependency DAG ordering and slack invariants (minimum 6 days), registered prediction protocols, refusal invariants, Toyota Way targets, verification ledger mark discipline (`[A]`, `[C]`, `[V]`, `[U]`), adjudication verdicts (SUSTAINED, NARROWED, REJECTED, INDETERMINATE), and prior-art comparisons.
2. **Grammar and Clarity**: Evaluation of technical readability, precision of architectural descriptions (e.g. `BackendRegistry::discover()`, REG-1..REG-14 requirements, failure-catalogue fixtures, printed discovery blocks), table layouts, ticket handle naming consistency, and changelog completeness.
3. **Code Metrics and Status**: Empirical verification against repository HEAD (`587ad0797` / post-`v0.65.2` `8e1e9ad40`), auditing referenced crates (`crates/apr-cli`, `crates/aprender-serve`, `crates/aprender-compute`, `crates/aprender-train`, `crates/aprender-gpu`, `crates/facades`), script behavior (`scripts/pv_bin.sh`, `scripts/check_no_claim_literals.sh`, `scripts/check_contract_test_binding.sh`, `scripts/check_readme_claims.sh`), and mathematical computations in §5, §6, §7, and §10.

### Overall Assessment
Segment 2 is an extraordinarily thorough, engineering-first specification that replaces aspirational declarations with falsifiable mechanisms. Promoting the runtime backend registry (R-0) to ticket #1, establishing the 14 REG requirements, deriving τ_loss empirically (T-1), and locking down release asset provenance (R-5..R-7) represent exceptional systems engineering.

However, forensic verification against the live codebase revealed **fifteen substantive mistakes and structural defects** that require correction before the specification is executed:
1. **Four Zero-Slack Blocker Pairs Violate the Min-6-Day DAG Invariant**: Cards P-0.3 → P-0.6, P-1.1 → P-1.2, T-1 → T-0, and master 0b → R-4 all share identical expiry dates with their direct blockers (0 days slack), violating §4 C10 and §5 G-4 (`expires(blocker) + 6d <= expires(blockee)`).
2. **Non-Executable Acceptance Command in G-4 (`pmat comply check --rule obligation-dag`)**: `pmat comply check` does not support `--rule`, `--min-slack-days`, or positional file arguments; the command fails with `error: unexpected argument '--rule' found`.
3. **Missing DAG Data Files (`docs/specifications/pp-066-dag.yaml` and `scripts/render_dag.py`)**: G-4 asserts the spec tables are rendered from YAML via `render_dag.py`, but neither file exists in the repository.
4. **`scripts/pv_bin.sh` Sourced-Only Interface Causes Silent-Pass Failure**: In R-0 and C12, `scripts/pv_bin.sh` is invoked as a standalone command. In git, `scripts/pv_bin.sh` is non-executable (`100644`), and when executed with arguments, it ignores all parameters and exits 0 in a subshell without running `pv`, creating a completely vacuous passing gate.
5. **Mismatched Tooling for Output Schema Validation in R-0**: R-0 prescribes `scripts/pv_bin.sh validate contracts/apr-devices-schema-v1.yaml` to validate `apr devices --json` output, but `pv validate` only validates YAML contract syntax, not CLI JSON output streams.
6. **Non-Existent `claims-cite` Tool in D-1**: D-1 relies on an existing `claims-cite` command that does not exist in the repository or PATH.
7. **Short-Circuited Contract Discipline in Track P**: Track P cards omit individual YAML contracts and standard card structures, bypassing the mandatory contract-per-card discipline.
8. **Missing Discrimination Clauses and Incomplete Mutation Specifications**: Cards R-7, T-0, G-3, and D-1 omit discrimination clauses, and T-1 and B-A1 truncate them to "yes".
9. **Internal Fixture Count Contradiction (12 vs 14)**: §4 C11 states "12 fixtures", while §5 R-0, §9, and Appendix A specify 14 fixtures (`FX-1..FX-14`).
10. **Stale Step-0 Premise Count in §9 (9/9 vs 23/23)**: §9 targets 9/9 premises, while §3 defines 23 premises (`S0-1..S0-23`).
11. **Incomplete Prediction Kill Threshold for S-2 in §7**: S-2 predicts both API share < 10% and prefill > 5,700 tok/s, but the kill column only bounds API share > 30%, leaving prefill throughput unmonitored.
12. **REG-10 Orphaned from Prior-Art Register (§12)**: REG-10 is explicitly derived from llama.cpp and Ollama, but omitted from their REG columns in §12.
13. **Factually Inaccurate Claim Regarding `aprender-serve` WASM Support**: The claim that `aprender-serve is #[cfg(not(target_arch = "wasm32"))] entirely` is false; only `safetensors` memory mapping is gated out of wasm32.
14. **Mislocated `MetalBackend` Reference**: Citing `MetalBackend` as residing in `metal_shaders.rs` is incorrect; `metal_shaders.rs` only defines 13 shader strings, while `MetalBackend` is in `crates/aprender-gpu/src/backend/mod.rs`.
15. **Stale Citation for `finetune.rs` in T-2 and §10**: Citing line 317 as currently hardcoding 512 is stale (line 317 is a comment; line 334 is already fixed). The actual hardcoded literal resides at line 717.

---

# Potential Mistakes and Improvements

### 1. Zero-Slack Blocker Pairs Violate the Min-6-Day DAG Invariant (§5 Track P, Track T, Track R)
- **Observation**:
  - Criterion C10 (§4 line 161) mandates: `0 zero-slack blocker pairs (min 6 days)`.
  - Guard G-4 (§5 line 430) specifies: `min slack 6 days between a row and every row it blocks`.
  - The G-4 contract invariant (§5 line 433) formalizes: `expires(blocker) + 6d ≤ expires(blockee)`.
  - In §5, four blocker pairs directly violate this rule with 0 days of slack:
    1. **Track P (lines 301–302)**: Row `P-0.3` expires `2026-09-19`. Row `P-0.6` lists `P-0.3` as a blocker and expires on the exact same date, `2026-09-19` (0 days slack).
    2. **Track P (lines 307–308)**: Row `P-1.1` expires `2026-10-10`. Row `P-1.2` lists `P-1.1` as a blocker and expires on the exact same date, `2026-10-10` (0 days slack).
    3. **Track T (lines 356, 372)**: Row `T-1` expires `2026-09-26`. Row `T-0` explicitly lists `T-1` in its blocker set (`blockers: T-0h, T-1, T-2`) and expires on the exact same date, `2026-09-26` (0 days slack).
    4. **Track R (lines 186, 265)**: Master row `0b` expires `2026-09-19`. Row `R-4` lists `master 0b (sampler pin)` as a blocker and expires on `2026-09-19` (0 days slack).
- **Logic Chain**:
  The author intended to bundle certain activities into the same PR or execution window (e.g. P-1.1 and P-1.2 note "same PR as P-1.1", T-1 notes "bundled with T-0's window"). However, declaring a formal blocker dependency between two tickets scheduled on the same expiry date directly triggers a failure under the `expires(blocker) + 6d ≤ expires(blockee)` rule. When `check_dag_invariants.sh` is wired into CI, these pairs will cause an immediate build failure.
- **Improvement**:
  1. For `P-0.3` / `P-0.6`: Stagger `P-0.3` to `2026-09-12` (leaving 7 days to `P-0.6` at `2026-09-19`).
  2. For `P-1.1` / `P-1.2`: Merge `P-1.2` into `P-1.1` as a single ticket (since they are explicitly defined as landing in the same PR), or stagger `P-1.1` to `2026-10-03`.
  3. For `T-1` / `T-0`: Stagger `T-1` to `2026-09-19` (aligning with `T-0h`), leaving 7 days to `T-0` at `2026-09-26`.
  4. For master `0b` / `R-4`: Shift `R-4` to `2026-09-26` (7 days slack after `0b` lands on `2026-09-19`).

### 2. Non-Executable Acceptance Command in G-4 (`pmat comply check --rule obligation-dag`)
- **Observation**:
  In §5 G-4 (line 432), the acceptance command `A:` is specified as:
  ```bash
  pmat comply check --rule obligation-dag docs/specifications/pp-066-dag.yaml --min-slack-days 6 exit 0
  ```
  Executing this command directly against the repository produces:
  ```
  error: unexpected argument '--rule' found

  Usage: pmat comply check [OPTIONS]

  For more information, try '--help'.
  ```
  `pmat comply check` only accepts `--mode`, `--path`, `--strict`, `--verbose`, `--failures-only`, `--quiet`, `--debug`, `--format`, `--include-project`, `--trace`, and `--color`. It does not accept positional file arguments or flags named `--rule` or `--min-slack-days`.
- **Logic Chain**:
  Card G-4 asserts that "the rule lives in `pmat comply` (it already derives expiries from obligation tables; PV-IMPROVE principle 8: no bash re-implementing a tool the fleet has)". In reality, no such rule or CLI surface exists in the installed `pmat` binary. A card whose acceptance command fails with a CLI parse error cannot be verified.
- **Improvement**:
  Update G-4's acceptance command to execute the dedicated verification script:
  ```bash
  bash scripts/check_dag_invariants.sh docs/specifications/pp-066-dag.yaml --min-slack-days 6
  ```
  If integration into `pmat comply` is planned as a future feature, file a separate ticket under `#pmat` and avoid asserting its current existence.

### 3. Missing DAG Data Files (`docs/specifications/pp-066-dag.yaml` and `scripts/render_dag.py`)
- **Observation**:
  - G-4 states: `scripts/render_dag.py regenerates §5/§6 tables byte-identical to the committed spec (drift = RED)` (line 432).
  - Criterion C10 specifies: `scripts/check_dag_invariants.sh docs/specifications/pp-066-dag.yaml --min-slack-days 6 exit 0` (line 161).
  - Neither `docs/specifications/pp-066-dag.yaml` nor `scripts/render_dag.py` exists in the repository.
- **Logic Chain**:
  Describing the spec's §5 and §6 markdown tables as already being "rendered from the yaml, never hand-edited" is inaccurate when the source YAML and rendering script have not been authored. If a CI gate checks for table drift against a non-existent YAML, it cannot run.
- **Improvement**:
  Clarify in G-4 that `docs/specifications/pp-066-dag.yaml` and `scripts/render_dag.py` are deliverables of ticket G-4 itself, and until G-4 merges, the markdown tables in §5/§6 serve as the initial draft source of truth.

### 4. `scripts/pv_bin.sh` Sourced-Only Behavior Causes Silent-Pass Failure (R-0, C12, PVI)
- **Observation**:
  - In §5 R-0 (line 208) and §4 C12 (line 159), acceptance commands invoke `scripts/pv_bin.sh` directly with subcommands:
    `scripts/pv_bin.sh validate contracts/apr-devices-schema-v1.yaml`
    `scripts/pv_bin.sh lint contracts/ --binding contracts/aprender/binding.yaml --strict-test-binding`
  - In the git tree, `scripts/pv_bin.sh` has mode `100644` (non-executable).
  - Inspecting lines 1–6 and 647–663 of `scripts/pv_bin.sh`:
    ```bash
    # Source it, never execute it:
    #     . scripts/pv_bin.sh || exit 1
    #     "$PV" lint contracts/
    ...
    export PV
    ```
  - `scripts/pv_bin.sh` ends after exporting `PV`. It contains no argument forwarding (`exec "$PV" "$@"`).
  - When executed in bash (`bash scripts/pv_bin.sh validate nonexistent.yaml`), it exits with code 0 without executing `pv` or checking anything.
- **Logic Chain**:
  This is a severe "gate is theater" vulnerability. Any acceptance command or release criterion structured as `scripts/pv_bin.sh <subcommand> <args>` will return code 0 unconditionally, masking invalid or missing contracts.
- **Improvement**:
  1. Add argument execution logic to the end of `scripts/pv_bin.sh`:
     ```bash
     if [ "$#" -gt 0 ]; then
         exec "$PV" "$@"
     fi
     ```
  2. Mark `scripts/pv_bin.sh` as executable (`chmod +x scripts/pv_bin.sh`).
  3. Alternatively, update all cards and criteria to use the sourced idiom:
     ```bash
     . scripts/pv_bin.sh && "$PV" validate contracts/apr-devices-schema-v1.yaml
     ```

### 5. Incompatible Tooling for JSON Output Schema Validation in R-0
- **Observation**:
  In §5 R-0 (line 208), the acceptance criteria state:
  ```bash
  scripts/pv_bin.sh validate contracts/apr-devices-schema-v1.yaml and every apr devices --json output validates against it
  ```
  `pv validate` is dedicated to verifying provable contract YAML syntax against the internal `provable-contracts` schema. It does not validate JSON output emitted by CLI binaries against JSON Schema definitions.
- **Logic Chain**:
  `contracts/apr-devices-schema-v1.yaml` is either a provable contract or a JSON Schema. If it is a JSON schema for `apr devices --json`, `pv validate` cannot validate runtime JSON output against it. Conflating contract validation with CLI payload schema enforcement creates an unverifiable command.
- **Improvement**:
  Clarify that `contracts/apr-devices-schema-v1.yaml` is validated by `pv validate`, while runtime JSON output conformance is verified in integration tests using `jsonschema` validation or Rust deserialization tests:
  ```bash
  cargo test -p apr-cli --test devices_schema_conformance
  ```

### 6. Non-Existent `claims-cite` Tool in D-1 Acceptance Command
- **Observation**:
  In §5 D-1 (line 442), the acceptance command prescribes:
  ```bash
  the existing claims-cite check (it passed on #2868 [A]) extended to API: sentences rather than a new script — scripts/check_doc_citations.sh only if claims-cite cannot express the rule...
  ```
  Searching the entire repository for `claims-cite` reveals that no binary, script, or cargo tool of that name exists in the tree. (The existing scripts are `scripts/check_no_claim_literals.sh`, `scripts/check_readme_claims.sh`, and `scripts/check_verifier_pinning.sh`).
- **Logic Chain**:
  Referring to a non-existent tool as "the existing `claims-cite` check" creates a false dependency. Implementers cannot extend a tool that does not exist.
- **Improvement**:
  Explicitly specify `scripts/check_doc_citations.sh` as the primary deliverable for D-1 rather than an escape hatch:
  ```bash
  bash scripts/check_doc_citations.sh docs/specifications/cuda-backend-architecture.md
  ```

### 7. Contract-Per-Card Discipline Short-Circuit in Track P
- **Observation**:
  The contract discipline block at the top of §5 (line 175) mandates:
  `Until PV-IMPROVE-001 phase 2 lands, pv validate/pv lint cannot reject a hollow contract, so PP-066 holds its own contracts to the phase-2 shape now and the review lane checks it: kind: pattern; no registry: true...`
  Yet in Track P (lines 295–312), the cards (P-0.1 through P-1.2) do not name YAML contract paths, do not follow the standard card structure (`A`, `contract`, `mutation -> RED`, `quorum`), and state: "the doc's deliverable/falsifier columns are the cards' A/mutation and are not restated."
- **Logic Chain**:
  This creates an unacknowledged exception to the contract-per-card rule. If Track P cards do not produce contracts under `contracts/`, they cannot be checked by C12's strict test binding gate.
- **Improvement**:
  Either:
  1. Author contract files for Track P (e.g. `contracts/pvi-proof-evidence-v1.yaml`, `contracts/pvi-lint-gate-v1.yaml`), or
  2. Add an explicit exemption note in the contract discipline block explaining that Track P cards modify the verification engine itself and are governed directly by the `FALSIFY-PVI-nnn` gate suite.

### 8. Missing Discrimination Clauses and Incomplete Mutation Specifications
- **Observation**:
  The ticket template (lines 170–172) requires:
  `mutation → RED and what it must not trip (discrimination)`
  Multiple cards fail to provide discrimination specifications:
  - **R-7** (line 292): `mutation → RED: change the README one-liner's URL → the grep test RED`. No discrimination clause is provided.
  - **T-0** (line 375): `mutation → RED: a receipt with partial=true → validate RED`. No discrimination clause is provided.
  - **T-1** (line 359): `mutation → RED: hand-type τ → validator RED (no sha). Discriminates: yes`. Truncated to "yes" without naming what remains GREEN.
  - **B-A1** (line 393): `mutation → RED: type a date instead of the anchor → RED. Discriminates: yes`. Truncated to "yes".
  - **G-3** (lines 423–427): Completely omits both `mutation → RED` and `discrimination`.
  - **D-1** (line 444): `mutation → RED: strip one citation → RED`. No discrimination clause is provided.
- **Logic Chain**:
  A mutation that causes everything to fail (e.g. a syntax error or broken build) is not a valid discriminator. The purpose of the discrimination clause is to prove that the test specifically fails on the mutated invariant while leaving adjacent witnesses GREEN.
- **Improvement**:
  1. In R-7, add: `Discriminates: check_readme_claims.sh passes on all other assertions (crate count, command count).`
  2. In T-0, add: `Discriminates: receipts with partial=false validate cleanly; canary training execution is unaffected.`
  3. In T-1, replace "yes" with: `Discriminates: valid tau_loss.json receipts with correct sha256 pass validation.`
  4. In B-A1, replace "yes" with: `Discriminates: valid anchored cells in perf-matrix.yaml parse without error.`
  5. In G-3, add: `mutation → RED: introduce a duplicate use-site pattern → count mismatch RED. Discriminates: crate name guard stays GREEN.`
  6. In D-1, add: `Discriminates: valid API citations in unaffected sections remain GREEN.`

### 9. Internal Fixture Count Contradiction (12 vs 14)
- **Observation**:
  - In §4 C11 (line 158): `the failure-catalogue case table (R-0 §REG, 12 fixtures) is green`.
  - In §5 R-0 (line 208): `cargo test -p apr-cli --test registry_failure_catalogue (FX-1..FX-14 below...)`.
  - In §5 R-0 table (lines 215–230): REG-1 through REG-14 are defined.
  - In §9 (line 539): `14/14 REG requirements with a committed fixture`.
  - In Appendix A (line 660): `R-0 gains REG-1..REG-14 with a 14-fixture failure catalogue`.
- **Logic Chain**:
  The failure catalogue originally had 12 fixtures in draft v1.2/v1.3 before REG-13 and REG-14 were added. When the count was updated to 14 across §5, §9, and Appendix A, criterion C11 was overlooked.
- **Improvement**:
  Update §4 C11 (line 158) from `12 fixtures` to `14 fixtures`.

### 10. Stale Step-0 Premise Count in §9 Toyota Way Targets (9/9 vs 23/23)
- **Observation**:
  In §9 (line 544), the target specifies:
  `genchi genbutsu | **9/9** Step-0 premises answered by a pasted command output before ticket #1 | S0 ledger`
  However, §3 (lines 111–135) contains **23 premises** (`S0-1` through `S0-23`).
- **Logic Chain**:
  Draft v1.0 had 9 premises. S0-10..S0-23 were added across subsequent iterations. The target table in §9 was never updated to reflect the expanded scope.
- **Improvement**:
  Update line 544 to:
  `genchi genbutsu | **23/23** Step-0 premises answered by a pasted command output before ticket #1 | S0 ledger`

### 11. Incomplete Prediction Kill Threshold for S-2 in §7
- **Observation**:
  In §7 (line 477), the registered prediction for S-2 (W-C) states:
  `copies+allocs < 10 % of CUDA API time; prefill > 5,700 tok/s`
  The "prediction killed if" column only defines:
  `share > 30 %`
- **Logic Chain**:
  The prediction has two components (API share and prefill throughput). If copies and allocations consume 8% of API time, but prefill throughput achieves only 2,000 tok/s, the kill condition (`share > 30%`) is not triggered, even though the throughput prediction was missed by more than half.
- **Improvement**:
  Update the kill condition in line 477 to:
  `share > 30 % ∨ prefill_tok_s < 4,000`

### 12. REG-10 Orphaned from Prior-Art Register (§12)
- **Observation**:
  In §5 R-0 (line 226), REG-10 states:
  `Never mix vendors in one graph... mixed AMD+NVIDIA in one llama.cpp process segfaults; Ollama binds to the primary driver and drops the other | FX-10`
  In §12 (lines 648–649), the REG column for llama.cpp lists `1, 2, 5, 6, 13, 14`, and for Ollama lists `1, 3, 4, 6, 7, 8, 9, 12`. REG-10 is absent from both rows.
- **Logic Chain**:
  REG-10 was explicitly derived from prior-art failure modes in llama.cpp and Ollama. Omitting REG-10 from the prior-art summary table creates an inconsistency.
- **Improvement**:
  Add `10` to the REG column for both llama.cpp and Ollama in §12:
  - llama.cpp: `1, 2, 5, 6, 10, 13, 14`
  - Ollama: `1, 3, 4, 6, 7, 8, 9, 10, 12`

### 13. Factually Inaccurate Claim: "aprender-serve is `#[cfg(not(target_arch = "wasm32"))]` entirely"
- **Observation**:
  In §6 B-S1..S4 (line 464), §10 (line 575), and §11 MAJ-01 (line 596), the spec asserts:
  `aprender-serve is #[cfg(not(target_arch = "wasm32"))] entirely [A] — B-S1 is a port of the loader/decoder, not a harness`
  Inspection of `crates/aprender-serve/src/lib.rs` shows that only lines 470 and 473 carry `#[cfg(not(target_arch = "wasm32"))]` (for `MappedSafeTensorsModel` and `ShardedSafeTensorsModel`). Furthermore, `src/target.rs` explicitly provides a `cfg!(target_arch = "wasm32")` branch.
- **Logic Chain**:
  The core of `aprender-serve` (GGUF structures, tensor operations, inference execution) is not gated out of wasm32; only memory-mapped filesystem operations are disabled. Claiming that the entire crate is disabled for wasm32 overstates the scope of B-S1 from a targeted loader adaptation to a full-crate rewrite.
- **Improvement**:
  Clarify that only memory-mapped safetensors loading in `aprender-serve` is non-wasm32, whereas GGUF decoding logic compiles for WebAssembly.

### 14. Mislocated `MetalBackend` Reference
- **Observation**:
  In §6 B-M1..M4 (line 463), §10 (line 575), and §3 S0-12 (line 124), the spec claims:
  `13 MSL kernels + MetalBackend via manzana::metal already exist in crates/aprender-gpu/src/backend/metal_shaders.rs [A]`
  Inspecting `crates/aprender-gpu/src/backend/metal_shaders.rs` confirms that it contains only the 13 `pub const` MSL shader strings. `MetalBackend` is defined in `crates/aprender-gpu/src/backend/mod.rs:24`.
- **Logic Chain**:
  Citing the shader string file as containing the backend struct and trait implementation is a factual error.
- **Improvement**:
  Update the citation to: `13 MSL kernels in crates/aprender-gpu/src/backend/metal_shaders.rs and MetalBackend in crates/aprender-gpu/src/backend/mod.rs`.

### 15. Stale Citation for `finetune.rs` in T-2 and §10
- **Observation**:
  In §5 T-2 (lines 363, 367) and §10 (line 575), the spec cites:
  `finetune.rs:317,718 hardcode 512 and drop the CLI value [A]`
  `mutation → RED: restore the literal at :317 → RED`
  Inspecting `crates/apr-cli/src/commands/finetune.rs` at line 317 shows a doc comment describing Defect 2. Line 334 already reads `max_seq_len: max_seq_len.unwrap_or(...)`. However, line 717 (`512, // max_seq_len`) in the wgpu pipeline remains hardcoded.
- **Logic Chain**:
  The instruct pipeline was partially fixed in an earlier commit, leaving line 317 as a comment and line 717 as the active bug. A test attempting to "restore the literal at :317" targets a doc comment line.
- **Improvement**:
  Update T-2 and §10 to cite line 717 as the hardcoded literal in the wgpu pipeline, and specify that the mutation reintroduces hardcoding in `build_instruct_config` at line 334.

---

# Minor Corrections and Typos

### 1. Typo in Track P Description (`repnt`)
- **Location**: §5 line 308 (Row `P-1.2`)
- **Text**: `repnt or author the 10 dangling #[contract] sites`
- **Correction**: Change `repnt` to `repoint`.

### 2. Invalid Master File Name in §10 Verification Ledger
- **Location**: §10 line 567
- **Text**: `Master v3.0 §12 has rows 0a–0e, 1–21; no row 22 | [V] | PP-LLAMA-001-MASTER-v3.md read here`
- **Correction**: Change `PP-LLAMA-001-MASTER-v3.md` to `PP-LLAMA-001-MASTER.md`.

### 3. Discrepancy Between Card Handles and `pmat work add` Titles in Track T
- **Location**: §5 lines 363 and 379
- **Observation**:
  - Ticket `T-2` uses `pmat work add "T-5a: apr finetune --max-seq-len..."`.
  - Ticket `T-3` uses `pmat work add "T-7 (REPORTING): train_tok_per_sec..."`.
- **Correction**: Ensure ticket titles align with card handles:
  - `pmat work add "T-2: apr finetune --max-seq-len..."`
  - `pmat work add "T-3 (REPORTING): train_tok_per_sec..."`

### 4. Markdown Backtick Syntax Glitch in R-5
- **Location**: §5 line 275 (Row `R-5`)
- **Text**: `prerelease=false ⇒ ∃ 4 host receipts with `asset_sha256 == manifest[target]`;`
- **Correction**: Fix the unbalanced backtick formatting:
  `prerelease=false ⇒ ∃ 4 host receipts with \`asset_sha256 == manifest[target]\`;`

### 5. Stale Master Expiry Date Citation in §10
- **Location**: §10 line 568
- **Text**: `Master expiries rows 19/20/21 = 10-16 / 10-23 / 11-06 | [V] | same`
- **Correction**: Note that in master v3.1 §12, rows 19 and 21 were converted to `derived` in v3.0.24, so literal dates only applied in v3.0.

### 6. Outdated Error Line Reference in §10
- **Location**: §10 line 575
- **Text**: `error.rs:86-90,218 FeatureDisabled → 9`
- **Correction**: Update the line reference to `error.rs:63,106`, where `FeatureDisabled` is mapped to exit code `9` in `exit_code_value()`.

### 7. Missing Quorum Classification for G-3
- **Location**: §5 line 427 (Row `G-3`)
- **Text**: `quorum: none`
- **Correction**: Change to `quorum: review-only` (governance audit tickets require review sign-off).

### 8. Numbering Gap in Track T Cards
- **Location**: §5 lines 344–385
- **Observation**: User dispatch notes scope as `Track T T-0..T-4`. In §5, the cards are `T-0h`, `T-1`, `T-2`, `T-0`, and `T-3`. No card `T-4` exists in §5 (the levers T-1..T-5 are in §6).
- **Correction**: Add a brief note in Track T clarifying that ticket handles in §5 span `T-0h, T-0, T-1, T-2, T-3`, while full speed levers `T-1..T-5` are carried to 0.67 (§6).
