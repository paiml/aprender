# Summary

This document presents the rigorous Candidate 3 evaluation of **Segment 2** (`implementation_tickets_future_lanes_governance`) of `PP-066-release-spec.md` (v1.5, dated 2026-09-05). The audit examines §5 (0.66 Tickets: Tracks I, R, P, S, T, B, G, D), §6 (0.67 Lane), §7 (Registered Predictions), §8 (Refusals), §9 (Toyota Way Targets), §10 (Verification Ledger), §11 (Adjudication of `0_66-review.md`), §12 (Prior-Art Register), and Appendix A (Changelog). Every claim, command, arithmetic derivation, and architectural assertion has been empirically verified against the live worktree at `paiml/aprender` (commit `a99236a86`).

### Segment Scope Overview
Segment 2 transitions the high-level strategy established in Segment 1 into concrete, schedulable `paiml-implement` engineering units, multi-release lanes, and verification frameworks:
- **§5 0.66 tickets — paiml-implement units** (lines 167–448): Ticket specifications spanning Track I (Instrument chain rows 0a–19), Track R (Runtime discovery, refusal, release packaging: `R-0`–`R-7`), Track P (Provable contracts: `P-0.1`–`P-0.6`, `P-1.1`, `P-1.2`), Track S (Speed rows: `S-1`–`S-3`), Track T (Training measurement: `T-0h`, `T-1`, `T-2`, `T-0`, `T-3`), Track B (Backends: `B-A1`, `B-G1`), Track G (Guards & DAGs: `G-1`–`G-4`), and Track D (Documentation: `D-1`).
- **§6 0.67 lane — carried rows** (lines 449–471): Deferred performance kernels, hardware lanes (`B-W0`–`B-W5`, `B-M1`–`B-M4`, `B-S1`–`B-S4`, `B-A2`), PVI Phases 1.3–4.3 (#2556), and crate rename sequencing (51 PRs).
- **§7 Registered predictions** (lines 472–497): 16 registered prediction rows establishing pre-experiment falsifiers.
- **§8 Refusals (0.66)** (lines 498–527): 23 non-negotiable negative constraints and invariant bounds.
- **§9 Toyota Way targets (0.66)** (lines 528–560): 26 quantitative quality metrics across Jidoka, Andon, Poka-Yoke, Genchi Genbutsu, Kaizen, and Heijunka.
- **§10 Verification ledger for this document** (lines 561–587): Evidentiary citations and provenance tracking.
- **§11 Adjudication of `0_66-review.md`** (lines 588–641): Formal rulings on 33 review findings and 5 unraised findings.
- **§12 Prior-art register — GPU discovery** (lines 642–654): Architectural comparative synthesis of llama.cpp, Ollama, and llamafile mapped to `REG-1`..`REG-14`.
- **Appendix A — Changelog** (lines 655–665): Version history tracking v1.0 through v1.5.

---

### Cross-Dimension Audit Assessment

1. **Spec Compliance**:
   - *Strengths*: The architectural elevation of runtime discovery (`R-0`) with its 14 explicit failure-catalogue requirements (`REG-1`..`REG-14`) directly addresses the five-whys finding (`F-25`). Refusal invariants in §8 and Toyota Way governance targets in §9 instantiate true industrial discipline. Adjudications in §11 rigorously maintain the distinction between empirical facts and speculative conclusions.
   - *Deficiencies*:
     - **DAG Ordering Invariant Breakdown**: Despite mandating "0 zero-slack blocker pairs (min 6 days)" across §4, §5, §8, and §9, the §5 DAG contains **four zero-slack (0 days) blocker pairs**: `master 0b` $\to$ `R-4`, `P-0.3` $\to$ `P-0.6`, `P-1.1` $\to$ `P-1.2`, and `T-1` $\to$ `T-0`.
     - **Contract Discipline Inconsistencies**: Track P cards completely lack explicit contract file paths. Ticket `G-2` is an unformatted one-liner lacking acceptance commands, contracts, and mutations. Ticket `G-3` lacks a mutation-to-RED discriminator.
     - **Temporal Paradox**: `R-0` lands on 2026-09-19 with an alleged contract, but `P-1.1` (which makes `#[contract]` fail closed) does not land until 2026-10-10. No follow-up ticket exists to decorate `R-0`'s `discover()` once `#[contract]` becomes real.
     - **External Repository Coupling**: `R-2` and `R-5` require `make -C machines/clean-room clean-room-p1`, which does not exist in `paiml/aprender`. `G-4` requires adding CLI flags (`--rule obligation-dag`, `--min-slack-days`) to the external `pmat` binary within an `aprender` PR.

2. **Grammar and Clarity**:
   - *Strengths*: Highly compact, information-dense technical syntax. The normative printed block in R-0 provides clear visual and semantic expectations.
   - *Deficiencies*:
     - **Catalogue Desynchronization**: Criterion `C11` in §4 mandates "12 fixtures", whereas §5 R-0, §9, and Appendix A specify "14 fixtures" (`REG-1`..`REG-14`). In §5, requirement `REG-13` lacks the `FX-13` fixture label found on all other rows.
     - **Identifier & Ticket Minting Collisions**: Ticket `T-2` defines a minting string labeled `T-5a:` (`pmat work add "T-5a: ..."`). Dual namespace collisions exist for `D-1` (Decision 1 vs. Doc Ticket D-1) and `T-1` (0.66 τ_loss instrument vs. 0.67 training speed levers).
     - **Unexecutable Prose in Acceptance Commands**: Commands in `R-0`, `R-5`, and `D-1` embed human prose instructions and non-executable clauses.

3. **Code Metrics and Status**:
   - *Strengths*: Fundamental arithmetic derivations (KV cache slot size of 448 MiB, 23 GB activation footprint before attention, wasm32 4.00 GiB boundary, and nsys copy/alloc rates) are mathematically verified.
   - *Deficiencies*:
     - **Falsification of Master Spec Claims**: In §10 line 567, the spec claims `[V]` that master v3.0 has no row 22. In fact, `PP-LLAMA-001-MASTER.md` is **v3.1**, is already committed on `main`, and carries **row 22** at line 364 (`| **22** |`). Furthermore, §10 line 568 asserts that master rows 19 and 21 have literal dates (10-16, 11-06); in the master spec, both rows have **`derived`** expiries, having had literal dates struck in v3.0.24.
     - **Execution Failure in `scripts/pv_bin.sh`**: `scripts/pv_bin.sh` has non-executable file permissions (`-rw-rw-r--`) and is designed strictly to be sourced. Direct subshell execution ignores CLI arguments and exits 0 unconditionally, rendering `scripts/pv_bin.sh validate ...` in R-0 and C12 completely vacuous.
     - **Crate Architecture Inaccuracies**: §6 asserts `aprender-serve` is `#[cfg(not(wasm32))]` entirely (false; only mmap safetensors models are excluded). §6 also asserts `aprender-gpu` has 13 MSL kernels and `MetalBackend` via `manzana::metal` (false; `metal_shaders.rs` contains inert strings, `aprender-gpu` has no `metal` feature, and `MetalBackend` in `src/backend/mod.rs` has no dispatcher).
     - **Stale Line References**: Multiple line citations in §10 have drifted (`error.rs` line 106, `Cargo.toml:666` instead of 509, `handlers.rs` in `apr-cli` instead of `aprender-serve`).

---

### Consensus Defect Matrix for Segment 2

| ID | Location | Summary of Verified Issue | Severity | Dimension |
|---|---|---|---|---|
| **E2-01** | §5 line 208, §4 C12 | `scripts/pv_bin.sh` is non-executable (`chmod -x`) and ignores CLI arguments in subshells; R-0 validation is a no-op | S1 | Spec Compliance / Metrics |
| **E2-02** | §5 lines 265, 302, 308, 372 | 4 zero-slack (0 days) blocker pairs in §5 DAG violate C10, G-4, and §8 refusal invariants | S1 | Spec Compliance |
| **E2-03** | §10 line 567, §3 S0-1 | Falsified assertion: `PP-LLAMA-001-MASTER.md` v3.1 is committed on `main` and carries Row 22 at line 364 | S1 | Code Metrics |
| **E2-04** | §10 line 568, §0 D-4, §5 S-1 | Stale assertion: Master rows 19 and 21 have `derived` expiries, not literal dates (struck in v3.0.24) | S1 | Code Metrics / Compliance |
| **E2-05** | §5 line 432 | `pmat comply check` lacks `--rule` and `--min-slack-days` flags (exits 2); external repo coupling in G-4 | S1 | Spec Compliance / Metrics |
| **E2-06** | §5 lines 250, 274 | Acceptance command requires `machines/clean-room` which does not exist in `paiml/aprender` | S2 | Code Metrics |
| **E2-07** | §5 line 208 | `pv validate` cannot validate JSON documents (`apr devices --json`) against schemas | S2 | Spec Compliance |
| **E2-08** | §4 C11, §5 line 208, §9 line 539 | Contradiction: C11 specifies "12 fixtures" while R-0, REG table, and §9 mandate 14 fixtures | S2 | Grammar & Clarity |
| **E2-09** | §9 line 544 | Stale metric: Toyota Way targets claim "9/9 Step-0 premises" instead of the actual 23 premises | S2 | Grammar & Clarity |
| **E2-10** | §12 (entire), §5 line 226 | Requirement `REG-10` is completely missing from §12; `REG-9` omits llama.cpp citation | S2 | Spec Compliance |
| **E2-11** | §5 line 420, 426, 297 | Card discipline violations: G-2 lacks A/contract/mutation; G-3 lacks mutation; Track P lacks YAML paths | S2 | Spec Compliance |
| **E2-12** | §5 line 229 | Requirement `REG-13` lacks fixture handle `FX-13` present on all other 13 rows | S3 | Grammar & Clarity |
| **E2-13** | §5 line 363, §2 line 96 | Ticket minting collision: T-2 mints `T-5a: ...`; T-2 is omitted from §2 Scope table | S2 | Spec Compliance |
| **E2-14** | §0 line 33, §5 line 439, §6 line 459 | Dual namespace collisions: `D-1` (Decision vs Ticket); `T-1` (0.66 τ_loss vs 0.67 levers) | S2 | Spec Compliance |
| **E2-15** | §6 line 464, §10 line 575 | Inaccurate claim: `aprender-serve` is not `cfg(not(wasm32))` entirely; core engine supports WASM | S3 | Code Metrics |
| **E2-16** | §6 line 463, §10 line 575 | Inaccurate claim: `aprender-gpu` has no `metal` feature and no `manzana::metal` dependency | S3 | Code Metrics |
| **E2-17** | §7 line 489 | Mutation count mismatch: §7 G-1 claims "both mutations RED" while G-1 defines 3 mutations | S3 | Grammar & Clarity |
| **E2-18** | §10 lines 565, 575 | Stale line citations: `error.rs` (line 106), `Cargo.toml:509` (line 666), `handlers.rs` ambiguity | S3 | Code Metrics |
| **E2-19** | §5 lines 206, 307 | Temporal paradox: R-0 lands 2026-09-19 without fail-closed `#[contract]` (lands 2026-10-10 in P-1.1) | S3 | Spec Compliance |
| **E2-20** | §5 line 442 | Acceptance command for D-1 is written as descriptive prose rather than an executable command | S3 | Spec Compliance |

---

# Potential Mistakes and Improvements

## 1. Release Gate Failures and Inexecutable Verification Commands

### 1.1 Sourcing Architecture and Permission Defect in `scripts/pv_bin.sh` (R-0, C12, §8 Refusal)
- **Location:** §5 line 208 (`R-0`), line 176 (Contract discipline), §4 line 159 (`C12`), §8 line 510.
- **Codebase Evidence:**
  Inspection of `scripts/pv_bin.sh` lines 1–6 confirms:
  ```bash
  # pv_bin.sh — resolve THE pv built from THIS TREE at HEAD, and prove it.
  #
  # Source it, never execute it:
  #     . scripts/pv_bin.sh || exit 1
  #     "$PV" lint contracts/
  ```
  Checking filesystem permissions reveals:
  ```bash
  $ ls -l scripts/pv_bin.sh
  -rw-rw-r-- 1 noah noah 32083 Sep  5 08:57 scripts/pv_bin.sh
  ```
  When executed directly in bash (`./scripts/pv_bin.sh ...`), execution fails with `Permission denied (exit 126)`. When run via `bash scripts/pv_bin.sh validate nonexistent_file.yaml`, the subshell exports `PV` and exits 0 immediately without processing `$@`:
  ```bash
  $ bash scripts/pv_bin.sh validate nonexistent_file.yaml; echo exit=$?
  exit=0
  ```
- **Failure Analysis:**
  In `R-0` (line 208), the acceptance test mandates:
  `scripts/pv_bin.sh validate contracts/apr-devices-schema-v1.yaml`
  In `C12` (line 159), the release gate mandates:
  `scripts/pv_bin.sh lint contracts/ --binding contracts/aprender/binding.yaml --strict-test-binding`
  Because `scripts/pv_bin.sh` has no command dispatcher and is not executable, invoking it directly as a command in CI or scripts either aborts with permission denied or exits 0 as a silent no-op. **This creates the exact verification theater finding `F-26` was formulated to destroy.**
- **Remediation:**
  1. Add an argument dispatcher at the end of `scripts/pv_bin.sh` (lines 662–665):
     ```bash
     export PV
     if [ "${BASH_SOURCE[0]}" = "$0" ] && [ "$#" -gt 0 ]; then
         exec "$PV" "$@"
     fi
     ```
  2. Enforce executable file permissions: `chmod +x scripts/pv_bin.sh`.
  3. Alternatively, update all spec references to use the explicit sourcing pattern:
     `. scripts/pv_bin.sh && "$PV" validate contracts/apr-devices-schema-v1.yaml`

---

### 1.2 Provable Contract Validation Scope Mismatch: `pv validate` vs. JSON Output
- **Location:** §5 line 208 (`R-0`).
- **Spec Text:**
  `scripts/pv_bin.sh validate contracts/apr-devices-schema-v1.yaml and every apr devices --json output validates against it`
- **Codebase Evidence:**
  Running `pv validate --help` shows:
  ```text
  Validate a YAML kernel contract
  Usage: pv validate [OPTIONS] <CONTRACT>
  ```
- **Failure Analysis:**
  `pv validate` is designed solely to validate that a contract YAML matches the schema of provable contracts. It has no capability to ingest an arbitrary JSON instance (such as the payload emitted by `apr devices --json`) and validate it against a JSON Schema. The clause "and every `apr devices --json` output validates against it" is unexecutable prose.
- **Remediation:**
  Split this into two distinct, executable commands:
  ```bash
  . scripts/pv_bin.sh && "$PV" validate contracts/apr-devices-schema-v1.yaml && apr devices --json | jsonschema -i - contracts/apr-devices-schema-v1.yaml
  ```
  (or invoke a dedicated Rust test `cargo test -p apr-cli --test devices_json_schema`).

---

### 1.3 Inexecutable CLI Arguments and Repository Dependency Inversion in Ticket G-4
- **Location:** §5 line 432 (`G-4`).
- **Spec Text:**
  `pmat comply check --rule obligation-dag docs/specifications/pp-066-dag.yaml --min-slack-days 6 exit 0`
- **Codebase Evidence:**
  Running `pmat comply check --help` reveals that `comply check` only accepts `--mode`, `--path`, `--strict`, `--format`, etc. It has no `--rule` or `--min-slack-days` flags and does not take positional arguments. Executing the command in the terminal produces:
  ```text
  $ pmat comply check --rule obligation-dag docs/specifications/pp-066-dag.yaml --min-slack-days 6
  error: unexpected argument '--rule' found
  Usage: pmat comply check [OPTIONS]
  exit=2
  ```
  Furthermore, `which pmat` returns `/home/noah/.cargo/bin/pmat`. The binary belongs to `paiml/pmat`, an entirely separate repository. Ticket `G-4` is assigned to repository `aprender` (`aprender · feat/g4-dag-invariants`).
- **Failure Analysis:**
  An engineer implementing `G-4` within `aprender` cannot modify the CLI parser or rule engine of `pmat`. Attempting to run this command will immediately fail CI with exit code 2.
- **Remediation:**
  Re-anchor the gate to a dedicated Python/shell verifier within `aprender` as cited in Criterion `C10`:
  `python3 scripts/check_dag_invariants.py docs/specifications/pp-066-dag.yaml --min-slack-days 6`
  If a `pmat comply` rule is desired, file an upstream issue on `paiml/pmat` and depend on a released version of `pmat`.

---

### 1.4 Non-Existent External Path in Acceptance Commands for R-2 and R-5
- **Location:** §5 line 250 (`R-2`) and line 274 (`R-5`).
- **Spec Text:**
  `make -C machines/clean-room clean-room-p1 exit 0`
- **Codebase Evidence:**
  `find . -name "clean-room*" -o -name "machines*"` returns zero results in `paiml/aprender`. The clean-room runner definitions reside in the external repository `paiml/infra`.
- **Failure Analysis:**
  Executing `make -C machines/clean-room clean-room-p1` inside the `aprender` worktree fails immediately:
  `make: *** machines/clean-room: No such file or directory. Stop.`
- **Remediation:**
  Qualify the command path relative to the developer environment or CI runner:
  `make -C ../infra/machines/clean-room clean-room-p1`
  or delegate the check to a workflow receipt from the `intel` runner.

---

## 2. Critical Dependency DAG Ordering and Slack Invariant Violations

### 2.1 Four Zero-Slack (0 Days) Blocker Pairs in §5
- **Location:** §5 lines 265, 302, 308, 356, 372.
- **Policy Invariant:**
  - §4 Criterion `C10` (line 161): "0 zero-slack blocker pairs (min 6 days)".
  - §5 `G-4` (line 430): "min slack 6 days between a row and every row it blocks".
  - §8 Refusals (line 522): "No expiry moved without an Appendix D amendment naming who and why (D-4)".
  - §9 Toyota Way (line 551): "**0** DAG invariant violations (cycles, zero-slack pairs, queue-vs-expiry inversions)".
- **Observed Blocker Pairs in §5:**
  1. **Master 0b $\to$ R-4**:
     - `master 0b` (sampler pin) expiry: **2026-09-19** (line 185).
     - `R-4` (W5 CLI wall-clock) blockers: `master 0b`. Expiry: **2026-09-19** (line 265).
     - **Slack: 0 days.**
  2. **P-0.3 $\to$ P-0.6**:
     - `P-0.3` (proof credit from evidence only) expiry: **2026-09-19** (line 301).
     - `P-0.6` (`pv lint` CI step) blockers: `P-0.3`. Expiry: **2026-09-19** (line 302).
     - **Slack: 0 days.**
  3. **P-1.1 $\to$ P-1.2**:
     - `P-1.1` (`#[contract]` fails closed) expiry: **2026-10-10** (line 307).
     - `P-1.2` (repoint 10 dangling contracts) blockers: `P-1.1`. Expiry: **2026-10-10** (line 308).
     - **Slack: 0 days.**
  4. **T-1 $\to$ T-0**:
     - `T-1` (τ_loss derived) expiry: **2026-09-26** (line 356).
     - `T-0` (four WT receipts) blockers: `T-0h`, `T-1`, `T-2`. Expiry: **2026-09-26** (line 372).
     - **Slack: 0 days.**
- **Failure Analysis:**
  If the automated DAG verifier `scripts/check_dag_invariants.sh --min-slack-days 6` (mandated by C10 and G-4) is executed against these §5 tickets, it will fail on day one with 4 distinct invariant violations.
- **Remediation:**
  Re-sequence the ticket expiries to ensure at least 6 days of slack:
  - Move `R-4` expiry from `2026-09-19` to `2026-09-26` (7 days after 0b).
  - Move `P-0.6` expiry from `2026-09-19` to `2026-09-26` (and stagger dependent PVI rows accordingly).
  - Explicitly mark `P-1.2` as merged in the same PR as `P-1.1` (compound ticket `P-1.1+1.2`) rather than an independent blocked card in the DAG.
  - Move `T-1` earlier to `2026-09-19` (bundled with `T-0h` window) or move `T-0` to `2026-10-03` (7 days after `T-1`).

---

## 3. Falsification of Master Spec Status, Expiries, and Row 22

### 3.1 Status and Row 22 Presence in `PP-LLAMA-001-MASTER.md`
- **Location:** §10 lines 567 & 579; §3 line 113 (`S0-1`).
- **Spec Claims:**
  - Line 567: `Master v3.0 §12 has rows 0a–0e, 1–21; no row 22 | [V] | PP-LLAMA-001-MASTER-v3.md read here`
  - Line 579: `PP-LLAMA-001-MASTER.md is committed on main | [U] — review segment 2 asserts it is uncommitted at the explorer's checkout | S0-1`
  - Line 113: `grep -n '^| 22' docs/specifications/PP-LLAMA-001-MASTER.md`
- **Codebase Evidence:**
  Direct inspection of `docs/specifications/PP-LLAMA-001-MASTER.md` in git tree at `origin/main`:
  1. Title line: `# PP-LLAMA-001 v3.1 — MASTER — Inference performance parity with llama.cpp`.
  2. Line 364:
     ```markdown
     | **22** | **instrument**: a top-2 logit margin per generated token on the wire (`logprobs` on the SSE delta, both engines), so the witness can classify an `m=1`↔`m=c` divergence as a near-tie flip (margin below a declared τ at the divergence index) or a defect; PP-26 (c) then becomes a gate | PP-26 (c); the residual that (a)+(b) cannot see — a whole batch that is coherent and identically wrong | serve | — | **OPEN.** Root row. Why it exists: `GET /v1/chat/completions` answers `logprobs: null` (`realize_handlers_completion_request.rs`), so the divergence lambda measured (`evidence/perf041/lambda/m1-vs-m4-three-prompts.txt`) can be read by a person but not classified by the witness. Expires **2026-10-15** |
     ```
  3. Appendix D line 407:
     `3.1 | 2026-09-02 | ... §12 row 22 (margin instrument) added so (c) can become a gate ...`
- **Failure Analysis:**
  The master spec is **already committed on `main`** as v3.1 and **contains row 22**.
  The check in `S0-1` (`grep -n '^| 22'`) failed solely because row 22 is bolded in markdown (`| **22** |`). This led the author of PP-066 to falsely mark row 22 absent (`[V]`), creating fictitious findings `F-9`, `S0-1`, and `W-H instrument [U]`.
- **Remediation:**
  - Update §10 line 567: mark row 22 as `[V] CONFIRMED present at line 364 of PP-LLAMA-001-MASTER.md v3.1`.
  - Update §10 line 579: mark master commit on `main` as `[V] CONFIRMED`.
  - Fix `S0-1` grep command: `grep -nE '^\|[[:space:]]*(\*\*)?22(\*\*)?[[:space:]]*\|' docs/specifications/PP-LLAMA-001-MASTER.md`.

---

### 3.2 Literal Expiries Struck from Master Rows 19 and 21 (v3.0.24)
- **Location:** §10 line 568; §0 line 37 (`D-4`); §1 line 58 (`F-3b`); §5 line 318 (`S-1`).
- **Spec Claims:**
  - Line 568: `Master expiries rows 19/20/21 = 10-16 / 10-23 / 11-06 | [V] | same`
  - Line 37: `The report re-dates master §12 rows 19/20/21 (10-16→09-19, 10-23→09-26, 11-06→10-10) without an Appendix D amendment | Master dates stand until an amendment names who moved them and why`
- **Codebase Evidence:**
  Inspection of `docs/specifications/PP-LLAMA-001-MASTER.md` lines 361–363 and Appendix D line 432 reveals:
  ```markdown
  | 3.0.24 | 2026-09-02 | §12 rows 19 and 21: the literal expiries (2026-10-16, 2026-11-06) are replaced by `derived` — both rows are blocked by live rows (1 and 2), and §12's preamble allows a literal date on root rows only; `scripts/spec_conformance.sh` refuses the typed dates (D3). Their expiry is now the later of their blockers'. |
  ```
  In the master table:
  - Row 19 expiry is **`derived (row 1)`**.
  - Row 20 expiry is **`2026-10-23`**.
  - Row 21 expiry is **`derived (row 2)`**.
- **Failure Analysis:**
  The literal expiries `2026-10-16` and `2026-11-06` were officially struck by the master specification itself in change 3.0.24 because typed dates on non-root rows violate `spec_conformance.sh` rule D3. Decision `D-4` and finding `F-3b` accuse the parity report of "silently moving dates without an amendment", when in reality the master spec explicitly amended them to `derived`.
- **Remediation:**
  Withdraw `F-3b` and update `D-4` and `S-1`: recognize that rows 19 and 21 derive their expiries from their respective blockers (`master 1` and `master 2`).

---

## 4. Architectural Desynchronizations Across Registers and Governance

### 4.1 Missing Requirement REG-10 in Prior-Art Register (§12)
- **Location:** §12 lines 646–651; §5 line 226 (`REG-10`).
- **Spec Text:**
  - §5 line 226:
    `REG-10 | Never mix vendors in one graph. Cross-vendor placement is structurally impossible... | mixed AMD+NVIDIA in one llama.cpp process segfaults; Ollama binds to the primary driver and drops the other | FX-10`
  - §12 table REG columns:
    - llama.cpp: `1, 2, 5, 6, 13, 14`
    - Ollama: `1, 3, 4, 6, 7, 8, 9, 12`
    - llamafile: `3, 5, 6, 8, 11`
    - all three: `5, 7, 11`
- **Failure Analysis:**
  Requirement `REG-10` is completely absent from every entry in §12's `REG` column, despite §5 explicitly deriving `REG-10` from llama.cpp segfaults and Ollama primary-driver bindings. Additionally, `REG-9` explicitly cites llama.cpp's even device-split in §5 line 225, but llama.cpp's entry in §12 omits 9.
- **Remediation:**
  In §12, add `10` to the REG list for both llama.cpp and Ollama (`REG: 1, 2, 5, 6, 9, 10, 13, 14` and `REG: 1, 3, 4, 6, 7, 8, 9, 10, 12`).

---

### 4.2 Fixture Count Contradiction: C11 ("12 fixtures") vs. REG-1..REG-14 ("14 fixtures")
- **Location:** §4 line 158 (`C11`); §5 lines 208, 215–230; §9 line 539; Appendix A line 660.
- **Observed Text:**
  - §4 `C11`: "the failure-catalogue case table (R-0 §REG, **12 fixtures**) is green"
  - §5 `R-0`: "cargo test -p apr-cli --test registry_failure_catalogue (**FX-1..FX-14** below)"
  - §9 Toyota Way: "**14/14** REG requirements with a committed fixture"
  - Appendix A (v1.4): "R-0 gains REG-1..REG-14 with a **14-fixture failure catalogue**"
- **Failure Analysis:**
  Criterion `C11` was written when the failure catalogue had 12 fixtures and was never incremented when `REG-13` and `REG-14` were added in v1.4. This creates a direct contradiction between the release gate in §4 and the implementation card in §5.
- **Remediation:**
  Update §4 `C11` line 158: replace `12 fixtures` with `14 fixtures`.

---

### 4.3 Stale Step-0 Metric in Toyota Way Targets (§9)
- **Location:** §9 line 544.
- **Spec Text:**
  `genchi genbutsu | 9/9 Step-0 premises answered by a pasted command output before ticket #1 | S0 ledger`
- **Failure Analysis:**
  In §3, the Step-0 discovery table contains **23 premises** (`S0-1` through `S0-23`). The metric `9/9` is an un-updated remnant from v1.0 (which had 9 premises). Stating `9/9` in the release quality targets allows 14 premises to go unanswered without violating the target.
- **Remediation:**
  Update §9 line 544 to read: `23/23 Step-0 premises answered by a pasted command output before ticket #1`.

---

### 4.4 Inaccurate Architectural Claims regarding WASM and Metal Support
- **Location:** §6 lines 463–464; §10 line 575.
- **Spec Claims:**
  - §6 line 464: `aprender-serve is #[cfg(not(target_arch = "wasm32"))] entirely [A] — B-S1 is a port of the loader/decoder, not a harness`
  - §6 line 463: `Correction: 13 MSL kernels + MetalBackend via manzana::metal already exist in crates/aprender-gpu/src/backend/metal_shaders.rs [A] (review MIN-05)`
- **Codebase Evidence:**
  1. In `crates/aprender-serve/src/lib.rs`, lines 470 and 473 restrict only `MappedSafeTensorsModel` and `ShardedSafeTensorsModel` to non-wasm32 targets. The core tensor primitives, forward passes, and GGUF inference are designed for WASM execution.
  2. In `crates/aprender-gpu/Cargo.toml`, there is no `metal` feature and no dependency on `manzana`. `crates/aprender-gpu/src/backend/metal_shaders.rs` lines 8–11 explicitly document:
     `These are source strings only. This crate contains no Metal dispatcher, so nothing here compiles or runs them...`
     and `MetalBackend` in `src/backend/mod.rs` is a stub where `is_available()` returns false.
- **Failure Analysis:**
  Both claims misrepresent codebase capabilities to reviewers. `aprender-serve` does not require a full rewrite/port for WASM; and `aprender-gpu` cannot execute Metal shaders natively without implementing a driver dispatcher.
- **Remediation:**
  Clarify in §6 `B-S1`: only memory-mapped safetensors loaders require WASM alternatives (streaming fetch or memory buffer). Clarify in `B-M1`: Metal shaders are raw string constants requiring a host dispatcher or `manzana` integration.

---

### 4.5 Temporal Paradox in Contract Enforcement Between R-0 and P-1.1
- **Location:** §5 line 206 (`R-0`), line 307 (`P-1.1`), line 176 (Contract discipline).
- **Spec Structure:**
  - `R-0` is ticket #1 of 0.66, expiring on **2026-09-19**.
  - `P-1.1` (`#[contract]` fails closed) expires on **2026-10-10**.
  - Line 307 states: "without it `#[contract]` on any 0.66 function is decoration; after it, R-0's `discover()` can carry `#[contract("apr-backend-registry-v1", …)]` and mean it".
- **Failure Analysis:**
  When `R-0` lands on September 19, `#[contract]` remains inert decoration. However, no ticket in §5 or §6 schedules adding `#[contract]` to `R-0::discover()` once `P-1.1` lands on October 10. `P-1.2` only addresses the 10 existing dangling sites. Consequently, `R-0` will ship in 0.66 without fail-closed contract enforcement.
- **Remediation:**
  Add an explicit acceptance task in `P-1.2` or mint a follow-up card `R-0b` to bind `#[contract("apr-backend-registry-v1", ...)]` to `BackendRegistry::discover()` immediately after `P-1.1` merges.

---

# Minor Corrections and Typos

### 1. Missing Fixture Identifier in Requirement REG-13
- **Location:** §5 line 229, table REG-1..REG-14.
- **Issue:** Every requirement in the table assigns an explicit fixture identifier in the last column (`FX-1` through `FX-12`, `FX-14`), except `REG-13`, which begins directly with prose: `MockBackend in tests/ registers, enumerates two fake devices...`.
- **Correction:** Prefix the fixture column of `REG-13` with `FX-13:` for strict structural uniformity with `FX-1..FX-14`.

---

### 2. Ticket Minting Discrepancy and Scope Table Omission in T-2
- **Location:** §5 line 363 (`T-2`); §2 line 96.
- **Issue:** The ticket is headed as `T-2 · --max-seq-len honoured or refused, never clamped`, but the minting command in line 363 reads:
  `pmat work add "T-5a: apr finetune --max-seq-len is honoured..."`
  Additionally, `T-2` is completely missing from the 0.66 scope table in §2 line 96 (`T-0h harness, T-0 four WT receipts, T-1 τ_loss, T-3 gate REPORTING`).
- **Correction:** Update the minting command to `pmat work add "T-2: apr finetune --max-seq-len..."` and add `T-2` to the §2 Scope table.

---

### 3. Dual Namespace Collisions (`D-1` and `T-1`)
- **Location:** §0 line 33 vs. §5 line 439; §5 line 354 vs. §6 line 459.
- **Issue:**
  1. `D-1` designates Decision 1 in §0 (Scope split), but also designates Track D Ticket 1 in §5 (`cuda-backend-architecture.md`).
  2. `T-1` designates Track T Ticket 1 in §5 (τ_loss seed variance instrument), but also designates 0.67 Training Levers in §6 (`T-1..T-5`).
- **Correction:** Rename Documentation Ticket `D-1` to `DOC-1`, and rename 0.66 τ_loss ticket `T-1` to `T-1a` (or rename 0.67 levers to `TL-1..TL-5`).

---

### 4. Stale Mutation Count in Registered Prediction G-1
- **Location:** §7 line 489.
- **Issue:** The prediction row for `G-1` reads:
  `both mutations RED in the guard's PR | either GREEN`
  However, in §5 line 417, `G-1` defines **three** distinct mutations:
  `(i) add an untracked crates/zz-probe/Cargo.toml... (ii) delete one allow-list line... (iii) add zz-probe to workspace.members...`.
- **Correction:** Update §7 line 489 to: `all three mutations RED in the guard's PR | any GREEN`.

---

### 5. Stale Line Citations in §10 Verification Ledger
- **Location:** §10 lines 565 and 575.
- **Issues:**
  1. `error.rs:86-90,218 FeatureDisabled → 9`: Line 218 is in a test helper; the actual enum match mapping `CliError::FeatureDisabled` to exit code 9 is at line 106.
  2. `aprender_ml alias at root Cargo.toml:509`: Line 509 of root `Cargo.toml` is `identity_op = "allow"`. The `aprender_ml` dependency alias is at line 666.
  3. `handlers.rs:912-953`: The file is located in `crates/apr-cli/src/commands/serve/handlers.rs`, not `crates/aprender-serve/src/handlers.rs` (which only contains 482 lines).
  4. `attention.rs:995,1022`: The file is located in `crates/aprender-serve/src/cuda/executor/layers/cublas_prefill/attention.rs`, not `crates/aprender-serve/src/cuda/executor/attention.rs` (which only contains 15 lines).
- **Correction:** Update §10 line references to cite fully qualified crate paths and current line numbers.

---

### 6. Missing Quorum Specification in Ticket G-3
- **Location:** §5 line 427.
- **Issue:** Ticket `G-3` specifies `quorum: none`. In the `paiml-implement` lifecycle, every ticket requires at least a review quorum (`review-only`).
- **Correction:** Update line 427 to `quorum: review-only`.

---

### 7. Internal Review Reference Typo in Changelog
- **Location:** Appendix A line 659 (v1.5 changelog).
- **Issue:** The changelog contains the raw review cross-reference `seg-2 2.4` in the public change list:
  `claim-literal citations on F-11, F-27, S0-9, S-2 note, T-1..T-5, §10, MIN-03, seg-2 2.4, §12 Ollama...`
- **Correction:** Replace `seg-2 2.4` with its formal finding or section name (`§6 T-lane NF4 kernel design`).

---

### 8. Unexecutable Prose and Formatting in Acceptance Commands
- **Location:** §5 line 274 (`R-5`) and line 442 (`D-1`).
- **Issues:**
  1. In `R-5`, line 274 includes human policy instructions inside the `A:` field:
     `gh release edit v0.66.0-rc1 --prerelease=false only after all four receipts are green; the promotion step is a workflow job that reads the receipts, not a hand command`
  2. In `D-1`, line 442 starts with descriptive commentary rather than a runnable command:
     `the existing claims-cite check (it passed on #2868 [A]) extended to API: sentences rather than a new script — scripts/check_doc_citations.sh only if claims-cite cannot express the rule...`
- **Correction:** Formulate strictly executable shell commands:
  - In `R-5`: `scripts/promote_release.sh --tag v0.66.0-rc1 --require-receipts lambda,gx10,intel,mini`
  - In `D-1`: `scripts/check_doc_citations.sh docs/specifications/cuda-backend-architecture.md`
