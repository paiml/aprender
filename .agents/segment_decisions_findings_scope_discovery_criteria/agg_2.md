# Summary

This aggregated review synthesizes evaluations from Candidate 2 (`handoff_2.md`) and Candidate 4 (`handoff_4.md`) for **Segment 1** (`decisions_findings_scope_discovery_criteria`) of `docs/specifications/PP-066-release-spec.md` (v1.5, 2026-09-05). The evaluated text covers the **Preamble** (lines 1–25), **§0 Decisions required before ticket #1** (D-1..D-11, lines 26–46), **§1 Findings register — deltas from the report** (F-1..F-28, lines 47–88), **§2 Scope** (lines 89–105), **§3 Step-0 — discovery** (S0-1..S0-23, lines 106–140), and **§4 0.66 release criterion** (C0..C14, lines 141–166).

### Audit Methodology and Aggregation Logic
1. **Consensus Identification**: Findings independently reported by both Candidate 2 and Candidate 4 were isolated and subjected to direct empirical verification against the working tree (commit `587ad0797` / `b1a6324b8` / `a99236a86`). This yielded 15 verified consensus defects.
2. **False Positive & Weak Assertion Filtering**: Unique findings were cross-checked against actual tool behavior. Where a candidate proposed an imprecise fix—such as Candidate 2 proposing `grep -c CONFORMANT evidence/parity/LEDGER.md -ge 2` for C1—the aggregation adopted Candidate 4's deeper observation proving that loose grepping returns 7 at HEAD despite 0 conformant rows, refining the remediation to enforce structural table-cell matching.
3. **Contradiction Resolution**: Discrepancies regarding ticket collisions and missing deliverables were resolved by auditing full spec cross-references. Both candidates identified distinct, valid namespace collisions: Candidate 4 discovered `D-1` (Decision 1 vs Ticket D-1), while Candidate 2 discovered `T-1` (0.66 τ_loss ticket vs 0.67 training levers) and the omission of `T-2` from §2. Both are reconciled and retained.
4. **Multi-Dimensional Coverage**:
   - **Spec Compliance**: Verified strict adherence to `paiml-implement` units (AUTO-IMPL-SKILL-001), C0 precedence before downstream credit, mark discipline (`[V]`, `[C]`, `[A]`, `[U]`, `[X]`), and scope boundaries.
   - **Grammar & Clarity**: Audited syntactic precision of commands, table sequencing, pseudo-code removal, and elimination of ambiguous prose assertions in exit checks.
   - **Code Metrics & Status**: Verified file paths, crate features, CLI commands (`pmat`, `gh`, `cargo`), file modes, and live branch protection / comply rules (`CB-1700`, `CB-1701`, `CB-2100`).

### Executive Findings Overview
Segment 1 provides an exemplary architectural pivot from uncalibrated speed claims toward instrumented discipline, elevating the runtime backend registry (R-0) and provable contracts (Track P) to blocking prerequisites. However, critical defects undermine its execution:
- **Flawed Step-0 Execution**: The minted ticket specifies falsifying "eight premises" while the table enumerates 23; tasks violate multi-host atomicity by bundling 4 bare-metal hosts into one ticket; and commands S0-1, S0-2, S0-4, and S0-12 fail upon execution.
- **Vacuous and Broken Release Gates**: Direct invocation of `scripts/pv_bin.sh` in C12/S0-18 bypasses contract linting entirely because the script is designed only to be sourced, exiting 0 in a subshell without executing `pv lint` ("gate is theater"). Criterion C1 lacks mandatory CLI arguments and contains invalid shell operators (`≥ 2`). C8 references a directory outside the repository (`machines/clean-room`). C7 literally reads `script exit 0` with no script name.
- **Structural Gaps and Ordering Disorders**: Criterion C14 is missing from §4 despite being mandated by the partition; Findings F-23 and F-24 are missing without explanation; and both Decision D-8 and Criterion C10 are misplaced out of numerical sequence.

---

# Potential Mistakes and Improvements

### 1. Step-0 Premise Count Mismatch and Multi-Host Execution Flaw (§3 line 109, lines 111–135)
- **Observation**:
  - Line 109 mints a single ticket:
    ```bash
    pmat work add "PP-066 S0: discovery ledger at HEAD" --description "Falsify the eight premises the 0.66 plan depends on; emit docs/audits/pp-066-s0-ledger.md with a verdict per premise"
    ```
  - The discovery table immediately follows (lines 111–135) and enumerates **23 premises** (`S0-1` through `S0-23`).
  - Furthermore, premises explicitly dictate commands on distinct physical bare-metal machines: S0-3/S0-4 ("on intel"), S0-12 ("on mini"), S0-14 ("on intel and mini"), S0-21 ("on gx10... on mini"), and S0-22 ("on lambda/gx10... on intel").
- **Logic Chain**:
  1. The spec expanded from 8 premises in early drafts to 23 in v1.5 as new root-cause findings (F-25..F-28) were integrated, but the `pmat work add` description string was never updated.
  2. A single `paiml-implement` worker runs in an isolated runner/worktree. An autonomous worker cannot execute commands natively across four physical architectures (`intel` x86_64, `mini` Apple Silicon, `lambda` x86_64+RTX4090, `gx10` aarch64+GB10) within a single ticket execution without an explicit SSH dogfood dispatch protocol.
- **Remediation**:
  - Update line 109 to specify: `"Falsify the 23 premises the 0.66 plan depends on; emit docs/audits/pp-066-s0-ledger.md with a verdict per premise"`.
  - Establish a multi-host ledger collection protocol: partitioned tickets (`S0-LOCAL`, `S0-INTEL`, `S0-MINI`, `S0-LAMBDA`, `S0-GX10`) or an explicit remote-exec harness that aggregates receipts into `docs/audits/pp-066-s0-ledger.md`.

### 2. Defective, Crashing, and Stale Shell Commands in Step-0 Premises (§3)
Empirical verification against the repository revealed multiple execution failures across the S0 discovery table:

#### a. S0-1 Regex Failure on Markdown Bold (`grep -n '^| 22'`)
- **Failure**: Line 112 runs `grep -n '^| 22' docs/specifications/PP-LLAMA-001-MASTER.md`. In the master spec at line 364, row 22 is formatted with markdown bold: `| **22** | **instrument**: a top-2 logit margin...`. The command exits 1 with no match.
- **Impact**: S0-1 evaluates as FALSIFIED, erroneously triggering the "flips" clause: "I-00 (commit the master) becomes ticket #1", which is false because the master is already committed on `main`.
- **Fix**: Update regex to handle optional whitespace and bold asterisks:
  ```bash
  grep -nE '^\|\s*\*{0,2}22\b' docs/specifications/PP-LLAMA-001-MASTER.md
  ```

#### b. S0-2 Invalid Argument to `pmat work list` (`--status all`)
- **Failure**: Line 113 runs `pmat work list --status all | grep -E 'PP-(6|27|28|21|15|29|26)'`. `pmat` rejects `--status all` with `Error: unknown status 'all' (did you mean 'new'?)`.
- **Impact**: S0-2 crashes immediately upon invocation.
- **Fix**: Remove `--status all` (running `pmat work list` without flags lists all tickets across statuses by default):
  ```bash
  pmat work list | grep -E 'PP-(6|27|28|21|15|29|26)'
  ```

#### c. S0-4 Cargo Multi-Binary Ambiguity (`cargo run -p apr-cli`)
- **Failure**: Line 115 runs `cargo run -p apr-cli --features wgpu -- serve run --backend wgpu --list-adapters`. Cargo aborts with: `error: cargo run could not determine which binary to run. Use the --bin option to specify a binary, or the default-run manifest key. available binaries: apr, apr-corpus-ingest`.
- **Impact**: Command cannot run without manual disambiguation.
- **Fix**: Add `--bin apr` to the command line, or add `default-run = "apr"` to `crates/apr-cli/Cargo.toml`:
  ```bash
  cargo run -p apr-cli --bin apr --features wgpu -- serve run --backend wgpu --list-adapters
  ```

#### d. S0-12 Non-Existent `metal` Cargo Feature in `aprender-gpu`
- **Failure**: Line 123 specifies `cargo check -p aprender-gpu --features metal; cargo test -p aprender-gpu --features metal -- metal_devices`. Cargo fails: `error: the package 'aprender-gpu' does not contain this feature: metal`.
- **Impact**: `crates/aprender-gpu/Cargo.toml` lines 64–80 contains no `metal` feature (removed in PR #2849). `crates/aprender-gpu/src/backend/mod.rs:67-81` defines `MetalBackend` as a stub returning `is_available(&self) -> bool { false }`.
- **Fix**: Clarify that S0-12 verifies `crates/aprender-gpu/src/backend/metal_shaders.rs` contains the 13 MSL shaders, and document that restoring a functional `metal` feature requires re-integrating `manzana` under Ticket B-M1 (0.67).

#### e. S0-18 Unexecutable Script Mode and Stale Match Count
- **Failure**: `scripts/pv_bin.sh` has file mode `100644` (non-executable). Direct execution fails with `Permission denied`. Furthermore, `grep -c 'registry: true' contracts/*.yaml` outputs per-file counts across hundreds of files, and counting total matching files yields **501**, not 481.
- **Fix**: Source the script or execute through bash, update the expected count to 501, and aggregate the match count:
  ```bash
  grep -l 'registry: true' contracts/*.yaml | wc -l # returns 501
  ```

#### f. S0-23 & F-28 Stale Compliance and Branch Protection Assertions
- **Failure**: Line 134 (S0-23) and Line 80 (F-28) claim `CB-1700/1701/2100 fail at HEAD; required_status_checks.strict is false`. Live execution of `pmat comply check` confirms:
  - `✓ CB-1700: Branch Protection: default branch requires ["ci / gate", "workspace-test"], >=1 approving review, and forbids force-push` (PASSES).
  - `✗ CB-1701: Supply Chain: 1 supply-chain violation(s)...` (FAILS).
  - `✗ CB-2100: Comply Gate Effect: 9 severity=error rule(s) unreachable...` (FAILS).
  - GitHub API read (`gh api repos/paiml/aprender/branches/main/protection`) confirms `strict: true`, `force: false`, `del: false`.
- **Impact**: Stating that CB-1700 fails and `strict` is false contradicts empirical reality at HEAD.
- **Fix**: Update text to state: `CB-1700 passes branch protection rules; CB-1701 and CB-2100 fail at HEAD; required_status_checks.strict is already true`.

---

### 3. Critical Sourced Architecture Defect in `scripts/pv_bin.sh` (C12, S0-18, F-26)
- **Observation**:
  - Criterion C12 (line 159) specifies:
    ```bash
    scripts/pv_bin.sh lint contracts/ --binding contracts/aprender/binding.yaml --strict-test-binding exit 0
    ```
  - S0-18 (line 130) specifies: `scripts/pv_bin.sh; pv lint contracts/ ...`.
  - In `scripts/pv_bin.sh` lines 3–5, the author explicitly documented:
    ```bash
    # Source it, never execute it:
    #     . scripts/pv_bin.sh || exit 1
    #     "$PV" lint contracts/
    ```
  - The script terminates at line 662 with `export PV`. It does not contain `exec "$PV" "$@"` or any command-line argument dispatch logic.
  - If invoked directly via bash (`bash scripts/pv_bin.sh lint contracts/...`), the script resolves `PV`, exports it to the subshell environment, ignores all passed arguments (`lint contracts/...`), and exits with code 0!
- **Logic Chain**:
  1. The author of `PP-066-release-spec.md` wrote C12 assuming `scripts/pv_bin.sh` behaves like a CLI executable wrapper (e.g. `pv_bin.sh lint ...`).
  2. Because `scripts/pv_bin.sh` only exports `PV` and ignores arguments, running C12 directly exits 0 unconditionally—even if every contract in `contracts/` is completely invalid!
  3. This reproduces the exact "gate is theater" failure mode that finding F-26 was created to eliminate.
- **Remediation**:
  - **Option A (Fix Script)**: Add CLI argument passthrough to `scripts/pv_bin.sh` before `export PV`:
    ```bash
    if [ "$#" -gt 0 ]; then
        exec "$PV" "$@"
    fi
    ```
    and set executable permissions (`chmod +x scripts/pv_bin.sh`).
  - **Option B (Fix Spec)**: Update C12 and S0-18 to conform to the documented sourced interface:
    ```bash
    . scripts/pv_bin.sh && "$PV" lint contracts/ --binding contracts/aprender/binding.yaml --strict-test-binding
    ```

---

### 4. Non-Executable, Broken, and Semantically Vacuous Commands in §4 Release Criteria
Line 143 mandates:
> *"The tag is cut when **all** hold; each is a command that must exit 0 on the release commit."*

Audit of table §4 revealed widespread violations where checks mix unexecutable prose, missing arguments, invalid shell operators, and vacuous regexes:

| Criterion | Spec Text in "Check" Column | Concrete Flaw / Failure Mode | Correct Executable Replacement |
|---|---|---|---|
| **C0** | `pmat comply check \| grep -E 'CB-(1700\|1701\|2100)' prints no ✗; gh api ... prints true; the #2830 ticket's perf_gate.sh --selftest row ... is GREEN ...` | Compound prose assertions (`prints no ✗`, `is GREEN`). Cannot exit 0 in CI. | `scripts/check_c0_release_gate.sh` wrapping the three programmatic assertions and exiting 0 on clean pass. |
| **C1** | `scripts/perf_gate.sh --phase release receipt validates; grep -c CONFORMANT evidence/parity/LEDGER.md ≥ 2` | 1. `scripts/perf_gate.sh --phase release` fails with exit code 2 (missing `--host`, `--workload`, `--receipt`, `--commit`).<br>2. Raw operator `≥ 2` causes bash syntax error.<br>3. `grep -c CONFORMANT` matches 7 lines at HEAD in prose despite 0 conformant data rows (vacuous pass). | `scripts/perf_gate.sh --host lambda --phase release --workload W1 --receipt evidence/parity/lambda-w1.r1.json --commit $(git rev-parse HEAD) && [ $(grep -E '^\|[[:space:]]*[0-9]+[[:space:]]*\|.*\|[[:space:]]*CONFORMANT[[:space:]]*\|' evidence/parity/LEDGER.md \| wc -l) -ge 2 ]` |
| **C4** | `scripts/check_multiplatform_dogfood.sh --require-resolved-backend cuda exit 0 on both` | Flag `--require-resolved-backend` does not exist in `scripts/check_multiplatform_dogfood.sh`. Contains prose note `exit 0 on both`. | Annotate that `--require-resolved-backend` is delivered under Ticket R-6 / C4, and specify: `scripts/check_multiplatform_dogfood.sh --require-resolved-backend cuda` |
| **C6** | `both scripts exit 0 at HEAD; each has a PR with a mutation commit → RED → revert in its history` | Entirely prose description; no executable test command. | `scripts/check_crate_names.sh && scripts/check_backend_firstclass.sh` |
| **C7** | `script exit 0` | The table cell literally says `script exit 0` without specifying the script name (`scripts/check_no_claim_literals.sh` exists in repo). | `scripts/check_no_claim_literals.sh` |
| **C8** | `make -C machines/clean-room clean-room-p1 exit 0` | Directory `machines/clean-room` does not exist in `paiml/aprender` (resides in sibling repo `../infra/machines/clean-room`). | `make -C ../infra/machines/clean-room clean-room-p1` (with documented sibling repo precondition) or wrap in `scripts/run_clean_room.sh`. |
| **C9** | `ls docs/audits/impl-*-receipt.md \| wc -l = ticket count; grep -L 'partial=false' empty` | Pseudocode (`= ticket count`); `grep -L 'partial=false' empty` looks for a literal file named `empty`. | `[ $(ls docs/audits/impl-*-receipt.md 2>/dev/null \| wc -l) -eq $(pmat work list --status closed \| wc -l) ] && [ -z "$(grep -L 'partial=false' docs/audits/impl-*-receipt.md 2>/dev/null)" ]` |
| **C10** | `scripts/check_dag_invariants.sh docs/specifications/pp-066-dag.yaml --min-slack-days 6 exit 0 (G-4)` | Appends prose note `exit 0 (G-4)` to command line. Placed out of order at the bottom of the table. | `scripts/check_dag_invariants.sh docs/specifications/pp-066-dag.yaml --min-slack-days 6` |
| **C11** | `scripts/check_backend_registry.sh exit 0 on all four hosts; pmat query --regex 'cfg!\(.*feature = "(cuda\|wgpu)"' --path crates/apr-cli/src → 0 hits outside the registry module` | Prose arrow `→ 0 hits...` and multi-host prose annotations. | `scripts/check_backend_registry.sh && [ $(pmat query --regex 'cfg!\(.*feature = "(cuda\|wgpu)"' --path crates/apr-cli/src \| grep -v 'registry.rs' \| wc -l) -eq 0 ]` |
| **C12** | `scripts/pv_bin.sh lint contracts/ ... exit 0 with the PP-066 contracts in the strict set; scripts/check_contract_test_binding.sh baseline unchanged (13 lines, may not grow); per-contract pv score REPORTED...` | Appends prose requirements to the command invocation. Sourced script ignores CLI args. | `. scripts/pv_bin.sh && "$PV" lint contracts/ --binding contracts/aprender/binding.yaml --strict-test-binding && scripts/check_contract_test_binding.sh` |
| **C13** | `gh release view v0.66.0 --json assets --jq '[.assets[].name]' lists 5 apr-* tarballs + checksums + signature; curl -LsSf …/install.sh \| bash on each host exits 0...` | Contains literal Unicode ellipsis `…` instead of valid URL. | `gh release view v0.66.0 --json assets --jq '[.assets[].name]' \| grep -q 'apr-.*\.tar\.gz' && curl -LsSf https://github.com/paiml/aprender/releases/download/v0.66.0/install.sh \| bash` |

---

### 5. Missing Release Criterion C14 and Table Ordering Disorder in §4
- **Observation**:
  - The audit partition (`ANALYSIS_PARTITION.md`) and user dispatch mandate checking `§4 0.66 release criterion (C0..C14)`.
  - Inspection of §4 (lines 145–161) shows exactly 14 rows, with IDs: `C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C11, C12, C13, C10`.
  - **Criterion C14 does not exist anywhere in the document.**
  - **Criterion C10 is placed out of order** at the very bottom of the table below C13.
- **Logic Chain**:
  1. If 14 criteria were intended (zero-indexed C0 through C13), the partition specification miscalculated the range as `C0..C14` by adding 14 to index 0.
  2. If 15 criteria were intended, Criterion `C14` was accidentally dropped during drafting.
  3. Appending C10 after C13 disrupts linear auditability.
- **Remediation**:
  - Re-order table rows numerically: place `C10` between `C9` and `C11`.
  - Harmonize the criteria count: if a 15th criterion was intended (e.g. documentation / changelog gating), add `C14`. If not, update the partition and preamble to declare `C0..C13 (14 criteria total)`.

---

### 6. Dual Identifier Namespace Collisions (D-1 and T-1/T-2)
Forensic cross-referencing across sections revealed two critical namespace collisions and one scope table omission:

#### a. Decision D-1 vs Ticket D-1
- In §0 line 33, Decision 1 is labeled `**D-1**` (`0.66 scope: all five tracks or the split in §2`).
- In §5 line 439 (Track D), Ticket 1 is labeled `**D-1 · cuda-backend-architecture.md**`.
- In §4 line 163, prose references "A RED on the non-blocking track (D-1 CUDA doc, G-3 count)...".
- *Impact*: In ticket databases, commit messages, and automated receipts, `D-1` cannot uniquely resolve between a strategic scope decision and a technical documentation ticket.
- *Fix*: Prefix decisions as `DEC-1`..`DEC-11`, or rename Track D documentation tickets to `DOC-1`.

#### b. Ticket T-1 vs Training Levers T-1..T-5
- In §5 line 354 (Track T), Ticket 1 is defined as `**T-1 · τ_loss derived, not declared** (closes F-8a)`.
- In §0 line 33 (D-1) and §6 line 459, `T-1..T-5` designates the carried 0.67 training optimization levers (`pre-compiled sm_121; fused RMSNorm/RoPE/SwiGLU/chunked CE...`).
- *Impact*: `T-1` refers simultaneously to a statistical loss-derivation ticket in 0.66 and a kernel training lever in 0.67.
- *Fix*: Rename the 0.67 carried training levers in §0, §2, and §6 from `T-1..T-5` to `TL-1..TL-5` (Training Levers).

#### c. Ticket T-2 Omission from §2 Scope Inventory and Ticket Minting Inconsistency
- In §5 line 362, Ticket `T-2` is carded as `**T-2 · --max-seq-len honoured or refused, never clamped** (the T-5 half that is a silent-pass defect; 0.66 regardless of D-2)`.
- In §2 line 95, the 0.66 scope table lists: `T-0h harness, T-0 four WT receipts, T-1 τ_loss, T-3 gate REPORTING`. `T-2` is completely missing from the table!
- Furthermore, in line 363, the minted ticket command states: `pmat work add "T-5a: apr finetune --max-seq-len is honoured..."`, creating a contradiction where the card title is `T-2` but the minted identifier is `T-5a`.
- *Fix*: Add `T-2` to the §2 Scope table under 0.66 contents, and standardize the ticket minting string to `pmat work add "T-2: apr finetune --max-seq-len..."`.

---

### 7. Findings Register Sequence Gaps and Formatting Inconsistencies (§1)
- **Observation**:
  - The findings table lists: `F-1`..`F-22`, followed immediately by `**F-25**`, `**F-26**`, `**F-27**`, `**F-28**`.
  - Findings `F-23` and `F-24` do not exist anywhere in the repository.
  - Finding `F-3b` uses an ad-hoc alphanumeric sub-index rather than an integer sequence.
  - Finding IDs `F-1` through `F-22` are formatted in plain text, whereas `**F-25**` through `**F-28**` are bolded.
- **Logic Chain**:
  - Gaps in a formal findings register impair audit traceability. When numbers are skipped without an explanatory tombstone, reviewers cannot determine whether findings were omitted or unresolved.
- **Fix**:
  - Add an explicit note in §1 documenting that F-23 and F-24 were consolidated during triage, or re-index sequentially. Standardize markdown formatting across all finding IDs.

---

# Minor Corrections and Typos

### 1. Decision Table Ordering Disorder in §0 (lines 33–44)
- **Issue**: Rows are ordered `D-1, D-2, D-3, D-4, D-5, D-6, D-7, D-9, D-10, D-11, D-8`. `D-8` was appended to the bottom below D-11.
- **Correction**: Move `D-8` (Rename sequencing) to its proper numerical place between `D-7` and `D-9`.

### 2. Missing Script Deliverables at HEAD (§4)
- **Issue**: Table §4 cites `scripts/check_backend_firstclass.sh` (C2, C6), `scripts/train_parity.sh` (C5), `scripts/check_crate_names.sh` (C6), `scripts/check_dag_invariants.sh` (C10), and `scripts/check_backend_registry.sh` (C11). None of these scripts exist in `scripts/` at HEAD.
- **Correction**: Add a clear note or column annotation indicating that these scripts are newly minted deliverables created by tickets B-G1, T-1, G-1, G-4, and R-0 respectively.

### 3. Code Citation Line Number Drift
- **§0 Row D-7 (line 39)**:
  - Spec cites: `crates/apr-cli/src/error.rs:86-90,218`.
  - Code state: `CliError::FeatureDisabled(_) => 9` is located at **line 106**. The test asserting exit code 9 (`test_feature_disabled_exit_code`) is at **lines 284–287**. Line 218 is inside `find_model_file()`.
  - *Correction*: Update citation to `crates/apr-cli/src/error.rs:106,284-287`.
- **§3 Row S0-13 (line 124)**:
  - Spec cites: `gguf/wgpu_backend/mod.rs:140-145` returning `UnsupportedOperation`.
  - Code state: `RealizarError::UnsupportedOperation` is returned in `forward_qwen3_moe_wgpu` at **lines 197–208**. Lines 140–145 are doc comments and validation checks.
  - *Correction*: Update citation to `crates/aprender-serve/src/gguf/wgpu_backend/mod.rs:197-208`.

### 4. Preamble Governing Spec Status Stale Marking
- **Location**: Preamble (lines 5–6).
- **Text**: `governing inference spec docs/specifications/PP-LLAMA-001-MASTER.md (v3.0 in hand; v3.1 [U])`.
- **Issue**: `PP-LLAMA-001-MASTER.md` is committed on `main` at version 3.1 (line 406 records the v3.1 release on 2026-09-03).
- **Correction*: Update to `(v3.1 in hand [V])`.

### 5. Preamble Mark Notation Collision (`[X]`)
- **Location**: Preamble (lines 5 and 21–22).
- **Text**: Line 5 cites `([X], §12)`. Line 21 re-defines the report's private `[X]` as `[A]` and states `[X]` keeps its fleet meaning (third-party, excluded from published claims).
- **Issue**: Inline citation `([X], §12)` is ambiguous given the mark re-mapping.
- **Correction**: Change to `(third-party benchmark study [X], §12)`.

### 6. §0 Table Header Promise Discrepancy
- **Location**: §0 Table header (line 31).
- **Text**: Header states `recommended default (applied below, marked *pending*)`.
- **Issue**: None of the rows (D-1 through D-11) use the word `*pending*` or asterisks in their recommended default column.
- **Correction**: Either insert `*pending*` tags into the recommended default cells as promised, or remove the clause from the table header.

### 7. Date Arithmetic Precision in §2 (line 91)
- **Location**: §2 Scope (line 91).
- **Text**: `0.66 tag target: 2026-10-23 [A]. Cycle: 49 days [C].`
- **Issue**: From 2026-09-05 to 2026-10-23: September has 25 days remaining; October adds 23 days (25 + 23 = 48 days). Inclusive duration is 49 days.
- **Correction**: Clarify as `49 days (inclusive) [C]`.

### 8. F-6 Citation Provenance
- **Location**: §1 Finding F-6 (line 60).
- **Text**: Cites `defect class the dogfood skill exists to kill (36/77/103/111)` without an evidentiary mark.
- **Correction**: Mark as `[A]` to reflect provenance from upstream review.
