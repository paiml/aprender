# Summary

This aggregated review synthesizes the findings of Candidate Reviews 2 (`handoff_2.md`) and 4 (`handoff_4.md`) evaluating **Segment 2: `implementation_tickets_future_lanes_governance`** of `docs/specifications/PP-066-release-spec.md` (v1.5, lines 167–665). The evaluated scope comprises:
- **§5 0.66 Tickets — paiml-implement units**: Track I (master rows 0a–0e, 1, 3, 6, 7, 9, 10, 12, 14, 15, 18), Track R (R-0..R-7), Track P (P-0.1..0.6, P-1.1, P-1.2), Track S (S-1..S-3), Track T (T-0h, T-1, T-2, T-0, T-3), Track B (B-A1, B-G1), Track G (G-1..G-4), Track D (D-1).
- **§6 0.67 Lane — carried rows**: W-B kernel, W-D, W-E, W-G, W-H, W-F resident attach, T-1..T-5 training levers, T-7 arming, B-W0, B-W1..W5, B-M1..M4, B-S1..S4, B-A2, PVI 1.3–4.3, §6.7 renames (51 PRs), comparator bar gating.
- **§7 Registered Predictions**: prediction-kill vs. track-kill protocols, quantitative target register, kill thresholds, hysteresis dead bands.
- **§8 Refusals (0.66)**: 23 negative requirements and invariants.
- **§9 Toyota Way Targets (0.66)**: jidoka, poka-yoke, andon, genchi genbutsu, kaizen, heijunka, instrument-first, five whys.
- **§10 Verification Ledger for this document**: claims, verification marks (`[V]`, `[C]`, `[A]`, `[U]`), and authoritative references.
- **§11 Adjudication of `0_66-review.md`**: 33 review item dispositions and 8 unraised findings.
- **§12 Prior-Art Register — GPU Discovery in Three Shipped Systems**: comparative matrix across llama.cpp/ggml, Ollama, and llamafile.
- **Appendix A — Changelog**: versions 1.0 through 1.5.

The evaluation audits the segment across the three mandatory dimensions:
1. **Spec Compliance**: Contract-per-card discipline (`kind: pattern`, absence of `registry: true`, executable tests in `check_contract_test_binding.sh`), acceptance command ($A_i$) executability, mutation-to-RED discriminators, quorum classifications, dependency DAG ordering and slack invariants (minimum 6 days), registered prediction protocols, refusal invariants, Toyota Way quality targets, verification ledger mark discipline (`[A]`, `[C]`, `[V]`, `[U]`), adjudication verdicts (SUSTAINED, NARROWED, REJECTED, INDETERMINATE), and prior-art comparisons.
2. **Grammar and Clarity**: Technical readability, precision of architectural descriptions (such as `BackendRegistry::discover()`, REG-1..REG-14 requirements, failure-catalogue fixtures, and printed discovery blocks), table layouts, ticket handle naming consistency, and changelog completeness.
3. **Code Metrics and Status**: Empirical verification against repository HEAD (`587ad0797` / post-`v0.65.2` `8e1e9ad40`), auditing referenced crates (`crates/apr-cli`, `crates/aprender-serve`, `crates/aprender-compute`, `crates/aprender-train`, `crates/aprender-gpu`, `crates/facades`), script behavior (`scripts/pv_bin.sh`, `scripts/check_no_claim_literals.sh`, `scripts/check_contract_test_binding.sh`, `scripts/check_readme_claims.sh`, `scripts/check_perf_claims_cite_receipts.sh`), and mathematical computations in §5, §6, §7, and §10.

### Executive Assessment
Segment 2 is an exemplary engineering specification that systematically converts high-level audit observations into concrete, falsifiable `paiml-implement` work packages. By elevating the runtime `BackendRegistry` (R-0) to ticket #1, formulating the 14 REG discovery requirements from prior art (§12), creating an ironclad release asset verification loop (R-5..R-7), and anchoring performance claims in empirical distributions (T-1), the specification eliminates pervasive configuration theater.

However, forensic verification against the active codebase revealed **critical execution defects, internal schedule contradictions, and verification command failures** that must be rectified before execution:
1. **Silent-Pass Bypass via `scripts/pv_bin.sh` Execution**: In R-0 and C12, `scripts/pv_bin.sh` is invoked directly with subcommands. The script is non-executable (`100644`) and designed only to be sourced; when executed directly in bash with arguments, it drops all parameters and exits 0 without running `pv`, creating a completely vacuous passing gate.
2. **DAG Invariant Contradiction (Four Zero-Slack Blocker Pairs)**: §4 C10 and §5 G-4 enforce a strict 6-day minimum slack between blocker and blockee (`expires(blocker) + 6d <= expires(blockee)`). Yet §5 contains four blocker pairs scheduled on identical expiries (0 days slack): `R-4` vs `master 0b` (both 2026-09-19), `P-0.6` vs `P-0.3` (both 2026-09-19), `P-1.2` vs `P-1.1` (both 2026-10-10), and `T-0` vs `T-1` (both 2026-09-26).
3. **Physical Contention and Expiry Inversion on the `gx10` Queue**: §2 and §9 mandate that the single `gx10` queue be strictly ordered by expiry (`master 15 -> T-0 -> S-3`). However, `master 15` is blocked by rows 6 and 12 expiring on **2026-10-02**, whereas `T-0` and `S-3` expire on **2026-09-26**. Placing `master 15` first in the physical queue starves `gx10` throughout late September and violates G-4's rule that earlier queue positions must have earlier or equal expiries (`queue_pos(a) < queue_pos(b) => expires(a) <= expires(b)`).
4. **Non-Executable Acceptance Command in G-4**: G-4 specifies `pmat comply check --rule obligation-dag docs/specifications/pp-066-dag.yaml --min-slack-days 6`. `pmat comply check` does not accept `--rule`, `--min-slack-days`, or positional file arguments; the command fails immediately with `error: unexpected argument '--rule' found`.
5. **Missing DAG Source Files**: G-4 claims that §5/§6 tables are regenerated by `scripts/render_dag.py` from `docs/specifications/pp-066-dag.yaml`. Neither file exists in the repository.
6. **Premise Count Drift (9/9 vs 23/23)**: §9 line 544 targets `9/9 Step-0 premises answered`, but §3 was expanded across revisions to **23 premises** (`S0-1..S0-23`).
7. **Semantic Mismatch for Output Schema Validation in R-0**: R-0 prescribes `scripts/pv_bin.sh validate contracts/apr-devices-schema-v1.yaml and every apr devices --json output validates against it`. `pv validate` only validates YAML contract syntax, not CLI JSON output streams.
8. **Non-Existent `claims-cite` Standalone Tool in D-1**: D-1 relies on an "existing `claims-cite` check" that does not exist on PATH or as a binary; the actual in-tree script is `scripts/check_perf_claims_cite_receipts.sh`.
9. **Internal Fixture Count Contradiction (12 vs 14)**: §4 C11 requires 12 fixtures, whereas §5 R-0, §9, and Appendix A specify 14 fixtures (`FX-1..FX-14`).
10. **Identifier Desynchronization in Track T Cards**: Ticket `T-2` uses `pmat work add "T-5a: ..."`, and `T-3` uses `pmat work add "T-7 (REPORTING): ..."`, which breaks receipt naming (`impl-<id>-receipt.md`) under §4 C9.

### Candidate Review Concordance and Filtering
- **Agreements Verified**: Both candidates independently discovered the `scripts/pv_bin.sh` silent-pass defect (Finding 1.1 / 4), the four zero-slack DAG blocker pairs (Finding 1.2 / 1), the Step-0 premise count drift from 9 to 23 (Finding 1.4 / 10), the Track T handle desynchronization (Finding 1.5 / Minor 3), the orphaned `REG-10` cross-reference in §12 (Finding 2.1 / 12), the unbalanced markdown backtick in R-5 (Minor 3 / Minor 4), the `finetune.rs` line citation staleness (Minor 1 / Finding 15), and the wasm32 scoping nuance in `aprender-serve` (Finding 2.4 / 13).
- **Candidate 2 Unique Validated Strengths**: Uncovered the physical contention and expiry inversion deadlock on the `gx10` queue (Section 1.3), the allow-list line count ambiguity (51 vs 52 in G-1/C6), and duplicate section heading in §11.
- **Candidate 4 Unique Validated Strengths**: Uncovered the CLI failure in G-4's acceptance command (`pmat comply check --rule`), the absence of `pp-066-dag.yaml` and `render_dag.py`, the semantic tooling mismatch in R-0 (`pv validate` vs CLI JSON output), the non-existent `claims-cite` command in D-1, the 12 vs 14 fixture contradiction in C11, the missing prefill kill threshold for S-2 in §7, and missing discrimination clauses in cards R-7, T-0, G-3, D-1, T-1, B-A1.
- **Filtered False Positive**: Candidate 4 claimed a typo `repnt` in row `P-1.2` (line 308) to be replaced by `repoint`. Verification of `docs/specifications/PP-066-release-spec.md` at line 308 confirms the text already reads `repoint or author the 10 dangling #[contract] sites`; `repnt` does not occur anywhere in the document. This finding was rejected as a false positive.

---

# Potential Mistakes and Improvements

## 1. Critical Execution & Verification Defects

### 1.1 Vacuous Execution of `scripts/pv_bin.sh` in Acceptance Commands and Gates (R-0, C12, PVI)
- **Observation**:
  In §5 R-0 acceptance command (line 208), the spec prescribes:
  ```bash
  scripts/pv_bin.sh validate contracts/apr-devices-schema-v1.yaml
  ```
  In §4 C12 (line 159), the spec prescribes:
  ```bash
  scripts/pv_bin.sh lint contracts/ --binding contracts/aprender/binding.yaml --strict-test-binding
  ```
  In git, `scripts/pv_bin.sh` has file mode `100644` (non-executable). Inspecting `scripts/pv_bin.sh` lines 1–6 and 650–663 reveals:
  ```bash
  # pv_bin.sh — resolve THE pv built from THIS TREE at HEAD, and prove it.
  #
  # Source it, never execute it:
  #     . scripts/pv_bin.sh || exit 1
  #     "$PV" lint contracts/
  ...
  PV_BIN_RC=0
  PV=$(pv_bin_resolve) || PV_BIN_RC=$?
  ...
  pv_bin_assert_fresh "$PV" || return 1 2>/dev/null || exit 1
  export PV
  ```
  When executed directly in a subshell (e.g. `bash scripts/pv_bin.sh validate non_existent.yaml`), it resolves `$PV`, asserts freshness, exports `PV`, and exits with return code 0. It drops all command-line arguments and never runs `pv`.
- **Empirical Proof**:
  ```bash
  $ bash scripts/pv_bin.sh validate non_existent.yaml
  $ echo exit=$?
  exit=0
  ```
- **Impact**:
  Every acceptance command and release gate invoking `scripts/pv_bin.sh <subcommand> <args>` will return `PASS` unconditionally without verifying anything. This creates a silent-pass bypass across R-0, C12, and Track P—the exact "contract theater" defect class (F-26) the spec aims to eliminate.
- **Remediation**:
  1. Patch `scripts/pv_bin.sh` to support direct execution by adding an argument dispatcher at the bottom of the script:
     ```bash
     if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
         if [ "$#" -gt 0 ]; then
             exec "$PV" "$@"
         else
             printf '%s\n' "$PV"
         fi
     fi
     ```
  2. Mark `scripts/pv_bin.sh` executable (`chmod +x scripts/pv_bin.sh`).
  3. Update normative spec strings in §4 C12 and §5 R-0 to explicitly use the sourced idiom:
     ```bash
     . scripts/pv_bin.sh && "$PV" validate contracts/apr-devices-schema-v1.yaml
     ```

---

### 1.2 Non-Executable Acceptance Command in G-4 (`pmat comply check --rule`)
- **Observation**:
  In §5 G-4 (line 432), the acceptance command is specified as:
  ```bash
  pmat comply check --rule obligation-dag docs/specifications/pp-066-dag.yaml --min-slack-days 6 exit 0
  ```
  Running this command live against the repo fails with:
  ```
  error: unexpected argument '--rule' found

  Usage: pmat comply check [OPTIONS]

  For more information, try '--help'.
  ```
  `pmat comply check` supports flags like `--mode`, `--path`, `--strict`, `--verbose`, `--failures-only`, `--format`, and `--color`. It does not accept `--rule`, `--min-slack-days`, or positional file paths.
- **Logic Chain**:
  Card G-4 asserts that "the rule lives in `pmat comply` (it already derives expiries from obligation tables; PV-IMPROVE principle 8)". In reality, no such CLI options exist in the installed `pmat` binary. An acceptance command that fails CLI parsing cannot verify ticket completion.
- **Remediation**:
  Update G-4's acceptance command to execute the dedicated verification script:
  ```bash
  bash scripts/check_dag_invariants.sh docs/specifications/pp-066-dag.yaml --min-slack-days 6
  ```
  If native integration into `pmat comply` is planned, file a separate enhancement ticket under `#pmat` rather than asserting its present availability.

---

### 1.3 Missing DAG Source Files (`pp-066-dag.yaml` and `scripts/render_dag.py`)
- **Observation**:
  - G-4 states: `scripts/render_dag.py regenerates §5/§6 tables byte-identical to the committed spec (drift = RED)` (line 432).
  - Criterion C10 specifies: `scripts/check_dag_invariants.sh docs/specifications/pp-066-dag.yaml --min-slack-days 6 exit 0` (line 161).
  - Neither `docs/specifications/pp-066-dag.yaml` nor `scripts/render_dag.py` exists in the repository.
- **Logic Chain**:
  Describing the spec's §5 and §6 markdown tables as already being "rendered from the yaml, never hand-edited" is factually premature when the source YAML and generator script have not yet been authored. If a CI gate checks for table drift against a non-existent YAML, the gate cannot execute.
- **Remediation**:
  Clarify in G-4 that `docs/specifications/pp-066-dag.yaml` and `scripts/render_dag.py` are deliverables of ticket G-4 itself, and that until G-4 lands, the markdown tables in §5/§6 serve as the initial draft source of truth.

---

### 1.4 Tooling and Semantic Mismatch for Output Schema Validation in R-0
- **Observation**:
  In §5 R-0 (line 208), the acceptance criteria state:
  ```bash
  scripts/pv_bin.sh validate contracts/apr-devices-schema-v1.yaml and every apr devices --json output validates against it
  ```
  Running `pv validate --help` shows:
  ```
  Validate a YAML kernel contract
  Usage: pv validate [OPTIONS] <CONTRACT>
  ```
  `pv validate` is dedicated to verifying provable contract YAML syntax against the internal `provable-contracts` schema. It does not validate JSON streams emitted by CLI binaries against JSON Schema definitions.
- **Logic Chain**:
  `contracts/apr-devices-schema-v1.yaml` is either a provable contract or a JSON Schema definition. If it is a schema definition for `apr devices --json`, `pv validate` cannot validate runtime JSON output against it. Conflating contract validation with runtime CLI output schema enforcement creates an unverifiable acceptance command.
- **Remediation**:
  Clarify that `contracts/apr-devices-schema-v1.yaml` is validated by `pv validate`, whereas runtime JSON output conformance is verified in integration tests using Rust deserialization / `jsonschema` validation:
  ```bash
  cargo test -p apr-cli --test devices_schema_conformance
  ```

---

### 1.5 Non-Existent `claims-cite` Standalone Tool in D-1
- **Observation**:
  In §5 D-1 (line 442), the acceptance command prescribes:
  ```bash
  the existing claims-cite check (it passed on #2868 [A]) extended to API: sentences rather than a new script — scripts/check_doc_citations.sh only if claims-cite cannot express the rule...
  ```
  Searching the repository for `claims-cite` confirms that no command, binary, or cargo subtool by that name exists. The actual script run in CI is `scripts/check_perf_claims_cite_receipts.sh` (`.github/workflows/ci.yml:1111,1113`).
- **Logic Chain**:
  Referring to a non-existent executable as "the existing `claims-cite` check" creates a broken dependency. Implementers cannot extend a tool that does not exist.
- **Remediation**:
  Explicitly specify `scripts/check_doc_citations.sh` as the primary deliverable for D-1 rather than an escape hatch:
  ```bash
  bash scripts/check_doc_citations.sh docs/specifications/cuda-backend-architecture.md
  ```

---

## 2. DAG Scheduling, Slack & Queue Contradictions

### 2.1 DAG Invariant Contradiction: Four Zero-Slack Blocker Pairs in §5
- **Observation**:
  - §4 C10 (line 161) mandates: `0 zero-slack blocker pairs (min 6 days)`.
  - §5 G-4 (lines 430, 433) formalizes: `expires(blocker) + 6d <= expires(blockee)`.
  - Auditing ticket expiries and blocker definitions in §5 reveals four explicit zero-slack pairs:
    1. **`R-4` vs `master 0b`**:
       - `R-4` (line 265): `blockers: master 0b · expiry 2026-09-19`.
       - `master 0b` (Track I, line 186): `expiry 2026-09-19`.
       - Slack: $2026\text{-}09\text{-}19 - 2026\text{-}09\text{-}19 = \mathbf{0\text{ days}}$.
    2. **`P-0.6` vs `P-0.3`**:
       - `P-0.3` (line 301): `expiry 2026-09-19`.
       - `P-0.6` (line 302): `blockers: P-0.3 · expiry 2026-09-19`.
       - Slack: $2026\text{-}09\text{-}19 - 2026\text{-}09\text{-}19 = \mathbf{0\text{ days}}$.
    3. **`P-1.2` vs `P-1.1`**:
       - `P-1.1` (line 307): `expiry 2026-10-10`.
       - `P-1.2` (line 308): `blockers: P-1.1 · expiry 2026-10-10`.
       - Slack: $2026\text{-}10\text{-}10 - 2026\text{-}10\text{-}10 = \mathbf{0\text{ days}}$.
    4. **`T-0` vs `T-1`**:
       - `T-1` (line 356): `expiry 2026-09-26`.
       - `T-0` (line 372): `blockers: T-0h, T-1, T-2 · expiry 2026-09-26`.
       - Slack: $2026\text{-}09\text{-}26 - 2026\text{-}09\text{-}26 = \mathbf{0\text{ days}}$.
- **Logic Chain**:
  The author grouped certain activities into the same PR or hardware run window (e.g. P-1.1 and P-1.2 note "same PR as P-1.1", T-1 notes "bundled with T-0's window"). However, declaring formal blocker relationships between nodes with identical expiry dates directly violates G-4's invariant `expires(blocker) + 6d <= expires(blockee)`. `check_dag_invariants.sh` will fail immediately upon running against this DAG.
- **Remediation**:
  1. For `P-0.3` / `P-0.6`: Stagger `P-0.3` to `2026-09-12` (leaving 7 days of slack to `P-0.6` at `2026-09-19`).
  2. For `P-1.1` / `P-1.2`: Merge `P-1.2` into `P-1.1` as a single unit (since they explicitly land in the same PR), or stagger `P-1.1` to `2026-10-03`.
  3. For `T-1` / `T-0`: Stagger `T-1` to `2026-09-19` (aligning with `T-0h`), leaving 7 days to `T-0` at `2026-09-26`.
  4. For master `0b` / `R-4`: Shift `R-4` to `2026-09-26` (7 days of slack after `0b` lands on `2026-09-19`).

---

### 2.2 Physical Contention & Expiry Inversion on the `gx10` Queue
- **Observation**:
  §2 (lines 100–102) specifies the `gx10` physical queue rule:
  ```
  gx10 is one queue, ordered by expiry: master 15 (shakedown) → T-0 → S-3's gx10 leg → master 21 (0.67).
  ```
  §9 line 543 states:
  ```
  heijunka: ≤ 1 speed row in flight per host; gx10 = one queue ordered by expiry
  ```
  §5 G-4 (line 433) specifies the queue invariant:
  ```
  queue_pos(host, a) < queue_pos(host, b) ⇒ expires(a) ≤ expires(b)
  ```
  Auditing expiries and blockers:
  1. `master 15` (Track I, lines 192, 196, 198):
     - Blocked by `master 6` (expiry **2026-10-02**) and `master 12` (expiry **2026-10-02**).
     - Earliest allowable expiry for `master 15` per the 6-day slack rule: **2026-10-08**.
  2. `T-0` (Track T, line 372): Expiry **2026-09-26** (assigned queue slot 2).
  3. `S-3` (Track S, line 336): Expiry **2026-09-26** (assigned queue slot 3).
- **Impact**:
  `master 15` is assigned queue slot 1, yet cannot physically run until after October 2. If `gx10` strictly halts for slot 1 before executing slot 2, the machine will sit idle during late September, causing `T-0` and `S-3` to blow past their September 26 expiry dates. Furthermore, because `expires(master 15) > expires(T-0)`, assigning `master 15` to queue position 1 violates the invariant `queue_pos(a) < queue_pos(b) => expires(a) <= expires(b)`.
- **Remediation**:
  Re-sequence the `gx10` queue:
  - Either execute `T-0` (window slot 1) and `S-3` (window slot 2) in late September ahead of `master 15` (which enters queue slot 3 once rows 6 and 12 land in October);
  - Or decouple `master 15`'s blocker set from rows 6 and 12 if an earlier September shakedown is feasible;
  - Or reschedule `T-0` and `S-3`'s expiries to mid-October.

---

## 3. Specification Scope & Metric Desynchronization

### 3.1 Step-0 Premise Count Drift (9/9 vs 23/23)
- **Observation**:
  In §9 line 544:
  ```
  genchi genbutsu: 9/9 Step-0 premises answered by a pasted command output before ticket #1 | S0 ledger
  ```
  In §3 (lines 109, 111–135):
  - Line 109 states: `Falsify the eight premises the 0.66 plan depends on...`
  - The premise table enumerates **23 premises**: `S0-1` through `S0-23` (`S0-10..S0-13` added in v1.1, `S0-14..S0-15` in v1.2, `S0-16..S0-19` in v1.3, `S0-20..S0-22` in v1.4, and `S0-23` in v1.5).
- **Impact**:
  A gate evaluating §9's target (`9/9`) passes after answering only 9 premises, leaving 14 critical premises (including `S0-14` CUDA toolkit-free build, `S0-20` no cudart, `S0-21` unified memory behavior, and `S0-23` strict check branch protection) unvalidated prior to opening ticket #1.
- **Remediation**:
  Update §9 line 544 to **23/23 Step-0 premises**, and update §3 line 109 to "Falsify the twenty-three premises...".

---

### 3.2 Internal Fixture Count Contradiction (12 vs 14)
- **Observation**:
  - In §4 C11 (line 158): `the failure-catalogue case table (R-0 §REG, 12 fixtures) is green`.
  - In §5 R-0 (line 208): `cargo test -p apr-cli --test registry_failure_catalogue (FX-1..FX-14 below...)`.
  - In §5 R-0 table (lines 215–230): `REG-1` through `REG-14` are defined.
  - In §9 (line 539): `14/14 REG requirements with a committed fixture`.
  - In Appendix A (line 660): `R-0 gains REG-1..REG-14 with a 14-fixture failure catalogue`.
- **Logic Chain**:
  The failure catalogue originally had 12 fixtures in draft v1.2/v1.3 before `REG-13` and `REG-14` were added. When the count was updated to 14 across §5, §9, and Appendix A, criterion C11 was left at 12.
- **Remediation**:
  Update §4 C11 (line 158) from `12 fixtures` to `14 fixtures`.

---

### 3.3 Identifier Desynchronization in Track T Cards (`T-2` vs `T-5a`, `T-3` vs `T-7`)
- **Observation**:
  In §5 Track T:
  - Line 362 defines card: `T-2 · --max-seq-len honoured or refused, never clamped`. Line 363 specifies:
    ```bash
    pmat work add "T-5a: apr finetune --max-seq-len is honoured..."
    ```
  - Line 378 defines card: `T-3 · training gate, REPORTING + self-ratchet`. Line 379 specifies:
    ```bash
    pmat work add "T-7 (REPORTING): train_tok_per_sec, peak_vram..."
    ```
- **Impact**:
  According to §4 C9 and paiml-implement conventions:
  ```bash
  ls docs/audits/impl-*-receipt.md | wc -l = ticket count
  ```
  If card `T-2` mints `T-5a`, the generated receipt will be `impl-t5a-receipt.md`, whereas checkers expecting `impl-t2-receipt.md` will report a missing receipt. Similarly, card `T-3` minting `T-7` produces `impl-t7-receipt.md` instead of `impl-t3-receipt.md`.
- **Remediation**:
  Align the string in `pmat work add` to match the card handle: e.g. `pmat work add "T-2: ..."` and `pmat work add "T-3: ..."`, noting provenance (e.g. "formerly report T-5a / T-7") in the description.

---

### 3.4 Incomplete Prediction Kill Threshold for S-2 in §7
- **Observation**:
  In §7 (line 477), the registered prediction for S-2 (W-C) states:
  `copies+allocs < 10 % of CUDA API time; prefill > 5,700 tok/s`
  The "prediction killed if" column only defines:
  `share > 30 %`
- **Logic Chain**:
  The prediction has two explicit metrics (API share and prefill throughput). If copies and allocations consume 8% of API time, but prefill achieves only 2,000 tok/s, the kill condition (`share > 30%`) does not trigger, even though the throughput prediction failed by more than half.
- **Remediation**:
  Update the kill condition in line 477 to:
  `share > 30 % ∨ prefill_tok_s < 4,000`

---

## 4. Contract Discipline, Mutations & Architecture

### 4.1 Contract-Per-Card Discipline Short-Circuit in Track P
- **Observation**:
  The contract discipline block at the top of §5 (line 175) mandates:
  `Until PV-IMPROVE-001 phase 2 lands, pv validate/pv lint cannot reject a hollow contract, so PP-066 holds its own contracts to the phase-2 shape now and the review lane checks it: kind: pattern; no registry: true...`
  Yet in Track P (lines 295–312), cards `P-0.1` through `P-1.2` omit YAML contract paths, omit standard card fields (`A`, `contract`, `mutation -> RED`, `quorum`), and state: "the doc's deliverable/falsifier columns are the cards' A/mutation and are not restated."
- **Logic Chain**:
  This introduces an unacknowledged exception to the contract-per-card mandate. If Track P cards do not generate contracts under `contracts/`, they cannot be evaluated by C12's strict test binding gate.
- **Remediation**:
  Either author explicit contract files for Track P (e.g. `contracts/pvi-proof-evidence-v1.yaml`, `contracts/pvi-lint-gate-v1.yaml`), or add an explicit exemption note in the §5 contract preamble explaining that Track P cards modify the verification engine itself and are governed directly by the `FALSIFY-PVI-nnn` gate suite.

---

### 4.2 Missing Discrimination Clauses and Truncated Mutation Specifications
- **Observation**:
  The ticket template (lines 170–172) requires:
  `mutation → RED and what it must not trip (discrimination)`
  Multiple cards fail to provide discrimination specifications:
  - **R-7** (line 292): `mutation → RED: change the README one-liner's URL → the grep test RED.` (No discrimination clause).
  - **T-0** (line 375): `mutation → RED: a receipt with partial=true → validate RED.` (No discrimination clause).
  - **T-1** (line 359): `mutation → RED: hand-type τ → validator RED (no sha). Discriminates: yes.` (Truncated to "yes" without naming what remains GREEN).
  - **B-A1** (line 393): `mutation → RED: type a date instead of the anchor → RED. Discriminates: yes.` (Truncated to "yes").
  - **G-3** (lines 423–427): Completely omits both `mutation → RED` and `discrimination`.
  - **D-1** (line 444): `mutation → RED: strip one citation → RED.` (No discrimination clause).
- **Logic Chain**:
  A mutation causing catastrophic build or test suite failure is not a discriminating check. The discrimination clause is required to prove that the test specifically isolates the mutated property while leaving adjacent behaviors intact.
- **Remediation**:
  1. In R-7, add: `Discriminates: check_readme_claims.sh passes on all other assertions (crate count, command count).`
  2. In T-0, add: `Discriminates: receipts with partial=false validate cleanly; canary training execution is unaffected.`
  3. In T-1, replace "yes" with: `Discriminates: valid tau_loss.json receipts with correct sha256 pass validation.`
  4. In B-A1, replace "yes" with: `Discriminates: valid anchored cells in perf-matrix.yaml parse without error.`
  5. In G-3, add: `mutation → RED: introduce a duplicate use-site pattern → count mismatch RED. Discriminates: crate name guard stays GREEN.`
  6. In D-1, add: `Discriminates: valid API citations in unaffected sections remain GREEN.`

---

### 4.3 REG-10 Orphaned from Prior-Art Register (§12)
- **Observation**:
  In §5 R-0 (line 226), `REG-10` is specified:
  `Never mix vendors in one graph... mixed AMD+NVIDIA in one llama.cpp process segfaults; Ollama binds to the primary driver and drops the other | FX-10`
  In §12 (lines 648–649), the REG column lists:
  - llama.cpp: `1, 2, 5, 6, 13, 14`
  - Ollama: `1, 3, 4, 6, 7, 8, 9, 12`
  `REG-10` is absent from both rows.
- **Logic Chain**:
  `REG-10` was derived directly from prior-art failure modes in llama.cpp and Ollama. Omitting `10` from the prior-art summary table creates an internal cross-reference gap.
- **Remediation**:
  Add `10` to the REG column for both llama.cpp and Ollama in §12:
  - llama.cpp: `1, 2, 5, 6, 10, 13, 14`
  - Ollama: `1, 3, 4, 6, 7, 8, 9, 10, 12`

---

### 4.4 Factual Nuances: `aprender-serve` WASM Gating and `MetalBackend` Location
- **Observation**:
  - In §6 B-S1..S4 (line 464), §10 (line 575), and §11 MAJ-01 (line 596), the spec asserts: `aprender-serve is #[cfg(not(target_arch = "wasm32"))] entirely [A]`. In `crates/aprender-serve/src/lib.rs`, only lines 470 and 473 carry this attribute (for `MappedSafeTensorsModel` and `ShardedSafeTensorsModel`). GGUF decoding, tensor math, and inference logic compile for wasm32.
  - In §6 B-M1..M4 (line 463), §10 (line 575), and §3 S0-12 (line 124), the spec cites: `MetalBackend via manzana::metal in crates/aprender-gpu/src/backend/metal_shaders.rs`. `metal_shaders.rs` contains only 13 raw MSL source strings; `MetalBackend` is defined at `crates/aprender-gpu/src/backend/mod.rs:67`.
- **Logic Chain**:
  Inaccurate file and architectural citations mislead implementers during porting and device discovery.
- **Remediation**:
  - Update §6 and §10 to clarify that only memory-mapped safetensors loading in `aprender-serve` is non-wasm32, not the entire crate.
  - Update §3, §6, and §10 to cite `MetalBackend in crates/aprender-gpu/src/backend/mod.rs:67` and MSL shaders in `metal_shaders.rs`.

---

# Minor Corrections and Typos

### 1. Stale Line Citations in §10 Verification Ledger
Forensic line audit against current HEAD reveals four shifted references:
- **`aprender_ml` alias** (line 575): Spec cites root `Cargo.toml:509`. At HEAD, line 509 is `identity_op = "allow"`. The `aprender_ml` alias resides at line **666**.
- **`FeatureDisabled` in `error.rs`** (line 575): Spec cites `error.rs:86-90,218`. In `crates/apr-cli/src/error.rs`, enum variant `FeatureDisabled` is at line 63, mapped to `9` in `exit_code_value()` at line **106**, and tested at line 285.
- **Manifest count under git** (line 575): Spec cites `106 manifests under git ls-files '*/Cargo.toml'`. At HEAD, `git ls-files '*/Cargo.toml'` returns **104 manifests** (105 including root `Cargo.toml`).
- **`finetune.rs` hardcoded clamp** (lines 123, 363, 575): Spec cites `finetune.rs:317,718 hardcode 512`. Line 317 is a doc comment describing Defect 2; line 334 already threads the CLI parameter. The hardcoded literal remains at line **717** in `finetune_wgpu_lora`.

### 2. Duplicate Section Heading in §11 Adjudication
In §11:
- Line 630 reads: `**Five findings no reviewer raised** (standard deliverable) — a sixth, found 2026-09-05...`
- Line 633 repeats: `**Five findings no reviewer raised** (v1.1):` followed by items 1–5.
- *Correction*: Merge into a single heading, e.g. `**Findings no reviewer raised (v1.1–v1.5)**:`.

### 3. Mismatched Markdown Backtick in R-5 Contract Invariants
In §5 line 275:
```markdown
(ii) `prerelease=false ⇒ ∃ 4 host receipts with `asset_sha256 == manifest[target]`;
```
Notice the opening backtick before `prerelease` and unescaped backtick before `asset_sha256`.
- *Correction*: `(ii) prerelease=false ⇒ ∃ 4 host receipts with \`asset_sha256 == manifest[target]\`;`

### 4. Master Specification Filename Typo in §10
In §10 line 567:
```markdown
Master v3.0 §12 has rows 0a–0e, 1–21; no row 22 | [V] | PP-LLAMA-001-MASTER-v3.md read here
```
The committed file in `docs/specifications/` is `PP-LLAMA-001-MASTER.md`, not `PP-LLAMA-001-MASTER-v3.md`.
- *Correction*: Update citation to `PP-LLAMA-001-MASTER.md`.

### 5. Stale Master Expiry Date Status in §10
In §10 line 568:
```markdown
Master expiries rows 19/20/21 = 10-16 / 10-23 / 11-06 | [V] | same
```
In `PP-LLAMA-001-MASTER.md` changelog v3.0.24 (line 432), literal expiries for rows 19 and 21 were replaced by `derived`.
- *Correction*: Note that literal dates applied in v3.0, but were replaced by `derived` in v3.0.24.

### 6. Crate Name Guard Allow-List Count Ambiguity (51 vs 52)
- Line 411 states: `allow-list = 51 rows + the D-5 row (52)`.
- Line 415 states: `remaining=51`.
- Line 417 states: `remaining would read 50 while 51 offend`.
- Line 154 (C6) states: `allow-list = 51, renames = 0`.
- Line 546 (§9) states: `check_crate_names.sh remaining: 51 -> 51 in 0.66`.
- *Correction*: Clarify whether the allow-list file contains 51 lines or 52 lines (inclusive of the special D-5 row for duplicate `[lib] name = "aprender"`).

### 7. Missing Quorum Classification for G-3
In §5 line 427: `quorum: none`.
- *Correction*: Change to `quorum: review-only` (governance metrics and audit tickets require reviewer sign-off).

### 8. Empty Cells in §7 Predictions Table
In §7 lines 483–484, the `note` column contains trailing empty pipes (`| |`).
- *Correction*: Populate with an explicit note or `-` placeholder for clean table rendering.

### 9. Numbering Clarification in Track T Cards
The dispatch refers to Track T scope as `T-0..T-4`. In §5, the cards are `T-0h`, `T-1`, `T-2`, `T-0`, and `T-3`. Levers `T-1..T-5` are carried to 0.67 (§6).
- *Correction*: Add an explanatory note in Track T clarifying that §5 handles comprise `T-0h, T-0, T-1, T-2, T-3`, while full speed levers `T-1..T-5` are carried to §6.

### 10. Audit Record of Filtered False Positive
- Candidate 4 reported an alleged typo `repnt` at line 308 (Row `P-1.2`) to be changed to `repoint`.
- Inspection of line 308 at git HEAD confirms the text reads `repoint or author the 10 dangling #[contract] sites`.
- *Disposition*: Filtered and excluded as an unverified false positive.

---

### Verification and Test Traceability Matrix

| Item Audited | Referenced in Spec | Actual Codebase State | Status |
|---|---|---|---|
| `scripts/pv_bin.sh` direct execution | §5 R-0 A, §4 C12 | Drops `$@` when executed directly; only exports `$PV` and exits 0 | ⚠️ **FAIL (Silent Bypass)** |
| DAG min-slack 6d | §4 C10, §5 G-4 | 4 zero-slack pairs identified (`R-4`, `P-0.6`, `P-1.2`, `T-0`) | ⚠️ **FAIL (Schedule Contradiction)** |
| `gx10` queue order | §2, §9, §5 G-4 | `master 15` blocked by Oct 2 rows, blocking Sept 26 `T-0`/`S-3` | ⚠️ **FAIL (Physical Deadlock)** |
| G-4 `pmat comply check --rule` | §5 G-4 A | `error: unexpected argument '--rule' found` | ⚠️ **FAIL (Invalid CLI)** |
| `pp-066-dag.yaml` & `render_dag.py` | §4 C10, §5 G-4 | Neither file exists in tree | ⚠️ **FAIL (Missing Files)** |
| Step-0 premise count | §9 target | 9 declared vs. 23 actual (`S0-1..S0-23`) | ⚠️ **FAIL (Scope Drift)** |
| Failure catalogue fixtures | §4 C11 | 12 declared vs 14 actual (`FX-1..FX-14`) | ⚠️ **FAIL (Internal Contradiction)** |
| `claims-cite` executable | §5 D-1 A | Tool does not exist; script is `check_perf_claims_cite_receipts.sh` | ⚠️ **FAIL (Non-Existent Command)** |
| `CliError::FeatureDisabled` | §0 D-7, §5 R-0, §10 | Mapped to `9` in `crates/apr-cli/src/error.rs:106` | ✅ **VERIFIED** |
| `accel.rs:28` feature check | §1 F-25, §5 R-0, §10 | `cfg!(any(feature = "cuda", feature = "wgpu"))` in `accel.rs:28` | ✅ **VERIFIED** |
| `accel.rs:114` 3-surface test | §5 R-0 A, §10 | Verifies `run`, `chat`, `serve` surfaces | ✅ **VERIFIED** |
| MSL kernels in `aprender-gpu` | §3 S0-12, §6 B-M1, §10 | Exactly 13 `kernel void` functions in `metal_shaders.rs` | ✅ **VERIFIED** |
| `MetalBackend` struct location | §3 S0-12, §6 B-M1, §10 | Defined at `crates/aprender-gpu/src/backend/mod.rs:67` | ✅ **VERIFIED** |
| `Q4K_GEMV_SHADER` | §3 S0-13, §6 B-W0, §10 | Present at line 555 of `basic_ops.rs` | ✅ **VERIFIED** |
| `OwnedQuantizedModelWgpu` stub | §3 S0-13, §6 B-W0, §10 | Returns `UnsupportedOperation` in `wgpu_backend/mod.rs:197` | ✅ **VERIFIED** |
| Double `[lib] name = "aprender"` | §0 D-5, §5 G-2 | Root `Cargo.toml:693` & `aprender-core/Cargo.toml:72` | ✅ **VERIFIED** |
| Facades separate workspace | §5 G-1, §6 renames | `crates/facades/Cargo.toml` prevents `.rlib` collisions | ✅ **VERIFIED** |
| 7.61B f32 expansion | §6 B-W0, §10 | $7.61 \times 10^9 \times 4\text{ B} = 30.44\text{ GB}$ | ✅ **VERIFIED** |
| 7B QLoRA seq 2048 activation | §5 T-0h, §7 T-0, §10 | $196 \times 117.44\text{ MB} = 23.02\text{ GB}$ | ✅ **VERIFIED** |
| Wasm32 linear memory bound | §6 B-S1..S4, §10 | $2^{32}\text{ B} = 4.00\text{ GiB}$ | ✅ **VERIFIED** |
| Comparator bar ratio | §1 F-14, §6 W-E, §10 | $484.7 / 171.5 = 2.826... \approx 2.83\times$ | ✅ **VERIFIED** |
| Prefill ratio prediction | §5 S-2 note, §7 S-2 | $5,700 / 10,399 = 0.548... \approx 0.55\times$ | ✅ **VERIFIED** |
