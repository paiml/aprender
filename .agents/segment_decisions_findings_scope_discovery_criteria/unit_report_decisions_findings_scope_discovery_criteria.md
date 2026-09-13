# Summary

This document constitutes the definitive, Level 2 Root Aggregated unit review for **Segment 1** (`decisions_findings_scope_discovery_criteria`) of `PP-066-release-spec.md` (v1.5, dated 2026-09-05). It synthesizes, reconciles, and elevates the evaluations from Level 1 Aggregator 1 (`agg_1.md`, synthesizing Analysts 1 and 3) and Level 1 Aggregator 2 (`agg_2.md`, synthesizing Analysts 2 and 4). Every finding, logic chain, and proposed remediation has been empirically audited and verified against the live repository tree (`paiml/aprender` at worktree `pp-066-spec`, commit `a99236a86`).

### Scope of Segment 1
The audited segment establishes the foundational governance, discovery baseline, and release criteria for the entire 0.66 release cycle:
- **Preamble** (lines 1–25): Architectural framework, monorepo invariants (APR-MONO), `paiml-implement` execution loop (AUTO-IMPL-SKILL-001), and evidentiary mark definitions (`[V]`, `[C]`, `[A]`, `[U]`, `[X]`).
- **§0 Decisions required before ticket #1** (`D-1`–`D-11`, lines 26–45): Governance escalations, track scope partitioning, refusal exit code mapping, default feature strategies, and PV-IMPROVE-001 phase boundaries.
- **§1 Findings register — deltas from the report** (`F-1`–`F-28`, lines 46–87): 27 registered defects and root-cause analyses, highlighting design vs. configuration in GPU selection (`F-25`), contract verification theater (`F-26`), artifact of record confusion (`F-27`), and unreadable/non-strict branch protection gates (`F-28`).
- **§2 Scope (pending D-1)** (lines 88–105): 0.66 "instrumented and honest" lane vs. 0.67 "speed and lanes" deferrals, WIP caps (heijunka), and single-host queue serialization.
- **§3 Step-0 — discovery** (`S0-1`–`S0-23`, lines 106–139): 23 read-only pre-implementation verification premises, empirical commands, falsification triggers, and fallback branches.
- **§4 0.66 release criterion** (`C0`–`C13`, lines 140–165): 14 exit criteria governing the 0.66 release tag, anchored by `C0` precedence.

---

### Cross-Dimension Audit Assessment

1. **Spec Compliance**:
   - *Strengths*: Segment 1 establishes a rigorous paradigm shift away from uncalibrated speed claims toward verifiable instrumentation. Elevating the runtime hardware backend registry (`R-0`) and contract enforcement (`Track P`) to blocking prerequisites enforces the core philosophy of "correct before fast". The `paiml-implement` ticket execution loop (P0 discovery $\to$ P1 plan $\to$ P2 RED-first execute $\to$ P3 quorum $\to$ PR $\to$ `ci / gate` $\to$ receipt) is clearly articulated.
   - *Deficiencies*: Critical tracking and scope inconsistencies exist. The minted Step-0 discovery ticket specifies falsifying "eight premises" while the table enumerates 23. Release Criterion `C14` is referenced in governing partition documents but omitted from §4. Deliverable card `T-2` (`--max-seq-len` honoured or refused) is an acknowledged 0.66 blocker throughout the text yet is completely omitted from the §2 Scope table. Identifier namespace collisions occur across tracks (`D-1` as Decision vs. Ticket; `T-1` as 0.66 seed variance instrument vs. 0.67 training speed lever).

2. **Grammar and Clarity**:
   - *Strengths*: Technical prose is concise, authoritative, and direct. The five-whys root-cause syntheses in §1 (`F-25` through `F-28`) provide clear architectural context.
   - *Deficiencies*: Multiple check commands in §4 substitute prose instructions, pseudo-code, or unexecutable Unicode symbols for valid shell commands (e.g. `receipt validates`, `exit 0 on both`, `prints no ✗`, `is GREEN`, `grep -c CONFORMANT ... ≥ 2`, `wc -l = ticket count; grep -L 'partial=false' empty`). Table ordering suffers from sequence anomalies: Decision `D-8` is displaced after `D-11` in §0, and Criterion `C10` is displaced after `C13` at the bottom of §4. Inline commands in table cells feature escaped pipes (`\|`) that break upon direct terminal pasting.

3. **Code Metrics and Status**:
   - *Strengths*: Quantitative baselines (KV slot sizes, throughput ratios, memory footprints) were verified and found mathematically sound.
   - *Deficiencies*: Multiple commands fail immediately when executed against the active repository tree. In `S0-2`, `pmat work list --status all` crashes because `--status all` is an invalid flag. In `S0-1`, `grep -n '^| 22'` fails due to markdown bolding (`| **22** |`). In `S0-4`, `cargo run -p apr-cli` aborts due to multi-binary ambiguity (`apr` vs `apr-corpus-ingest`). In `S0-12`, `cargo check -p aprender-gpu --features metal` fails because the `metal` Cargo feature was eliminated in PR #2849. Most critically, `scripts/pv_bin.sh` in `C12` and `S0-18` is designed strictly to be sourced; executing it in a subshell ignores all passed arguments and exits 0 unconditionally, creating the exact verification theater that finding `F-26` sought to eradicate. Live branch compliance checks confirm that `CB-1700` already passes at HEAD, while `CB-1701` and `CB-2100` remain the open blockers.

---

### Unified Consensus Matrix of Verified Findings

| ID | Location | Summary of Verified Issue | Severity | Dimension |
|---|---|---|---|---|
| **E-01** | §4 C12, §3 S0-18 | `scripts/pv_bin.sh` ignores CLI arguments when executed; C12 exits 0 without running `pv lint` (theater) | S1 | Code Metrics / Compliance |
| **E-02** | §4 C1, C7, C9, C13 | Shell syntax errors: `≥ 2` Unicode crash, pseudo-code `wc -l = ...`, missing script in C7, ellipsis in C13 | S1 | Grammar & Clarity |
| **E-03** | §3 line 108 | Ticket description specifies "eight premises" while table defines 23 premises (`S0-1`..`S0-23`) | S2 | Spec Compliance |
| **E-04** | §4 (entire) | Criterion `C14` is missing from the specification despite being mandated in governing partitions | S2 | Spec Compliance |
| **E-05** | §2 line 95, §5 line 363 | Deliverable `T-2` missing from §2 Scope table; ticket minting string refers to `T-5a` instead of `T-2` | S2 | Spec Compliance |
| **E-06** | §3 S0-1 | `grep -n '^| 22'` returns 0 matches due to markdown bold formatting (`| **22** |`), falsely flipping S0-1 | S1 | Code Metrics |
| **E-07** | §3 S0-2 | `pmat work list --status all` crashes with `Error: unknown status 'all'` | S1 | Code Metrics |
| **E-08** | §3 S0-4 | `cargo run -p apr-cli` fails with multi-binary ambiguity (`apr` vs `apr-corpus-ingest`) | S2 | Code Metrics |
| **E-09** | §3 S0-12 | `cargo check -p aprender-gpu --features metal` fails; `metal` feature removed in PR #2849 | S2 | Code Metrics |
| **E-10** | §3 S0-14, S0-19 | `grep -ci 'cuda\|cublas'` and `grep -rn 'minisign\|...'` omit `-E`, treating `\|` as literal characters | S2 | Code Metrics |
| **E-11** | §3 S0-23, §1 F-28, §4 C0 | Stale assertion: `CB-1700` already passes at HEAD (`✓`); `strict=true` is already set | S2 | Code Metrics |
| **E-12** | §4 C8 | `machines/clean-room` does not exist in `paiml/aprender` (resides in sibling repo `../infra`) | S2 | Code Metrics |
| **E-13** | §4 C4 | Dogfood check command asserts only CUDA backend; lacks assertions for CPU/DriverNotFound on intel/mini | S2 | Spec Compliance |
| **E-14** | §4 C0, C6, C11 | Check columns contain compound prose assertions (`prints no ✗`, `is GREEN`, `→ 0 hits`) | S2 | Grammar & Clarity |
| **E-15** | §1 lines 76–77 | Findings register jumps from `F-22` to `F-25`; rows `F-23` and `F-24` are unassigned | S3 | Spec Compliance |
| **E-16** | §0 line 33, §5 line 439 | Dual namespace collision: `D-1` (Decision 1 vs Ticket D-1); `T-1` (0.66 τ_loss vs 0.67 training levers) | S2 | Spec Compliance |
| **E-17** | §0 line 43, §4 line 160 | Table ordering anomalies: `D-8` placed after `D-11`; `C10` placed at bottom after `C13` | S3 | Grammar & Clarity |
| **E-18** | §0 line 39, §3 line 124 | Stale line citations: `error.rs:86-90,218` (now line 106); `handlers.rs:888-953` (only 483 lines) | S3 | Code Metrics |
| **E-19** | Preamble line 6 | Stale status: `PP-LLAMA-001-MASTER.md` v3.1 is already committed on `main` at HEAD (`[V]`) | S3 | Code Metrics |
| **E-20** | §1 F-28, §4 C0 | `arm_a_scaling` in `scripts/perf_gate.sh` was renamed to `arm_a_self_regression` under PP-31 | S3 | Code Metrics |

---

# Potential Mistakes and Improvements

## 1. Release Gate Failures and Verification Theater (§4 Criteria)

### 1.1 Silent No-Op in Release Gate C12 and Subshell Variable Loss in S0-18 (`scripts/pv_bin.sh` Sourced Architecture)
- **Location:** Line 159 (Criterion `C12`) and Line 129 (Premise `S0-18`).
- **Codebase Evidence:**
  Inspection of `scripts/pv_bin.sh` lines 1–6 confirms:
  ```bash
  # pv_bin.sh — resolve THE pv built from THIS TREE at HEAD, and prove it.
  #
  # Source it, never execute it:
  #     . scripts/pv_bin.sh || exit 1
  #     "$PV" lint contracts/
  ```
  At line 662, the script executes `export PV` and terminates. It accepts zero command-line arguments and contains no argument dispatcher (such as `exec "$PV" "$@"`).
- **Execution Failure:**
  1. In `C12`, the check specifies `scripts/pv_bin.sh lint contracts/ --binding ...`. When executed as a command, bash runs the script in a subshell. The script exports `PV` within that subshell, completely ignores all passed arguments (`lint contracts/ ...`), and exits with return code `0`. **The release gate passes unconditionally without validating a single contract file.** This introduces the exact verification theater finding `F-26` was created to eliminate.
  2. In `S0-18`, the command `scripts/pv_bin.sh; pv lint contracts/ | tail -3` runs `scripts/pv_bin.sh` as a separate command. Upon process termination, the exported variable `PV` is discarded. The subsequent `pv` command resolves to whatever unpinned, potentially stale binary is on `PATH`, directly violating the repository's strict binary pinning policy.
- **Remediation:**
  - *Option 1 (Spec Fix - Recommended)*: Update `C12` and `S0-18` to use the documented sourcing pattern:
    - `C12`:
      ```bash
      . scripts/pv_bin.sh && "$PV" lint contracts/ --binding contracts/aprender/binding.yaml --strict-test-binding && scripts/check_contract_test_binding.sh
      ```
    - `S0-18`:
      ```bash
      . scripts/pv_bin.sh && "$PV" lint contracts/ | tail -3
      ```
  - *Option 2 (Script Fix)*: Add an execution fallback dispatcher at lines 662–665 of `scripts/pv_bin.sh`:
    ```bash
    if [ "${BASH_SOURCE[0]}" = "$0" ] && [ "$#" -gt 0 ]; then
      exec "$PV" "$@"
    fi
    ```
    and enforce executable file permissions (`chmod +x scripts/pv_bin.sh`).

---

### 1.2 Syntax Errors, Missing Arguments, and Loose Regex Leak in Parity Gate C1
- **Location:** Line 148, Criterion `C1`.
- **Spec Text:**
  ```bash
  scripts/perf_gate.sh --phase release receipt validates; grep -c CONFORMANT evidence/parity/LEDGER.md ≥ 2
  ```
- **Execution Failures:**
  1. Running `scripts/perf_gate.sh --phase release` without required flags exits with status `2`. The script requires `--host`, `--workload`, `--receipt`, and `--commit`.
  2. `≥ 2` contains the Unicode mathematical glyph `≥` (U+2265). Standard POSIX shells treat `≥` and `2` as positional file arguments to `grep`, emitting `grep: ≥: No such file or directory` and exiting with error code `2`.
  3. `receipt validates` is unquoted human prose embedded in a shell string.
  4. Running `grep -c CONFORMANT evidence/parity/LEDGER.md` matches 7 lines at HEAD in prose comments and column headers, despite **zero** data rows being conformant. A naive comparison `>= 2` would pass vacuously at HEAD today.
- **Remediation:**
  Rewrite the check command as an executable POSIX expression enforcing specific arguments and structural table-cell matching:
  ```bash
  scripts/perf_gate.sh --host lambda --phase release --workload W1 --receipt evidence/parity/lambda-w1.r1.json --commit $(git rev-parse HEAD) && [ $(grep -E '^\|[[:space:]]*[0-9]+[[:space:]]*\|.*\|[[:space:]]*CONFORMANT[[:space:]]*\|' evidence/parity/LEDGER.md | wc -l) -ge 2 ]
  ```

---

### 1.3 Pseudo-code, Missing Script Path, and Unexecutable Expressions in C7, C9, and C13
- **Location:** Lines 154 (`C7`), 156 (`C9`), and 159 (`C13`).
- **Observed Code & Defects:**
  - **C7**: The check column literally reads `script exit 0`. It omits the script name entirely. `scripts/check_no_claim_literals.sh` exists in the repository and is the intended target.
  - **C9**: Specifies `ls docs/audits/impl-*-receipt.md | wc -l = ticket count; grep -L 'partial=false' empty`. This fails with `bash: =: command not found` and attempts to search a non-existent file named `empty`.
  - **C13**: Check contains `curl -LsSf …/install.sh | bash`. The literal Unicode horizontal ellipsis `…` (U+2026) cannot be resolved by DNS or HTTP clients.
- **Remediation:**
  - **C7**: Set check command to `scripts/check_no_claim_literals.sh`.
  - **C9**: Replace with valid shell test assertions:
    ```bash
    [ $(ls docs/audits/impl-*-receipt.md 2>/dev/null | wc -l) -eq $(pmat work list --status closed | wc -l) ] && [ -z "$(grep -L 'partial=false' docs/audits/impl-*-receipt.md 2>/dev/null)" ]
    ```
  - **C13**: Replace ellipsis with the canonical GitHub releases URL:
    ```bash
    gh release view v0.66.0 --json assets --jq '[.assets[].name]' | grep -q 'apr-.*\.tar\.gz' && curl -LsSf https://github.com/paiml/aprender/releases/download/v0.66.0/install.sh | bash
    ```

---

### 1.4 Multi-Platform Asymmetry and Unimplemented CLI Flags in Dogfood Gate C4
- **Location:** Line 151, Criterion `C4`.
- **Observed Text:**
  ```bash
  scripts/check_multiplatform_dogfood.sh --require-resolved-backend cuda exit 0 on both
  ```
- **Defect:**
  1. The prose of `C4` explicitly mandates dogfooding across all four fleet hosts: `cuda` on `lambda` and `gx10`, and `cpu` with `cuda unavailable reason=DriverNotFound` on `intel` and `mini`.
  2. However, the check command specifies only `--require-resolved-backend cuda`. If executed on `intel` or `mini`, this invocation will fail immediately.
  3. Furthermore, `--require-resolved-backend` is a new flag to be delivered by Ticket `R-6` and does not yet exist in `scripts/check_multiplatform_dogfood.sh`.
- **Remediation:**
  Annotate that Ticket `R-6` implements the `--require-resolved-backend` and `--host-set` flags, and define distinct check commands for both host tiers:
  ```bash
  scripts/check_multiplatform_dogfood.sh --host-set cuda --require-resolved-backend cuda && scripts/check_multiplatform_dogfood.sh --host-set cpu --require-resolved-backend cpu
  ```

---

### 1.5 External Repository Sibling Precondition in Clean-Room Gate C8
- **Location:** Line 155, Criterion `C8`.
- **Observed Text:**
  ```bash
  make -C machines/clean-room clean-room-p1 exit 0
  ```
- **Defect:**
  The path `machines/clean-room` does not exist within `paiml/aprender`. It resides in the sibling infrastructure repository (`../infra/machines/clean-room`). Running this command in an isolated CI runner or fresh clone will fail with `make: machines/clean-room: No such file or directory`.
- **Remediation:**
  Explicitly document the sibling repository precondition, or wrap the clean-room invocation in an in-tree runner script (`scripts/run_clean_room.sh`) that verifies directory presence before dispatching:
  ```bash
  test -d ../infra/machines/clean-room && make -C ../infra/machines/clean-room clean-room-p1
  ```

---

### 1.6 Compound Prose Assertions and Automation Infeasibility in C0, C6, and C11
- **Location:** Lines 147 (`C0`), 153 (`C6`), and 157 (`C11`).
- **Defect:**
  Line 143 establishes the inviolable invariant: *"each is a command that must exit 0 on the release commit."* However:
  - `C0` contains human prose verification instructions: `prints no ✗; gh api ... prints true; the #2830 ticket's perf_gate.sh --selftest row ... is GREEN at HEAD and was RED on the mutation in its PR body; CB-2100's output names each of the nine rules under a required context`.
  - `C6` contains: `both scripts exit 0 at HEAD; each has a PR with a mutation commit → RED → revert in its history`.
  - `C11` contains: `pmat query ... → 0 hits outside the registry module`.
- **Remediation:**
  Wrap multi-step compound verifications into dedicated repository verification scripts:
  - Deliver `scripts/check_c0_release_gate.sh` to validate `pmat comply check` rule output, GitHub API branch protection status, and the `#2830` selftest exit code.
  - Deliver `scripts/check_c11_registry.sh` wrapping `scripts/check_backend_registry.sh` and asserting zero occurrences of `cfg!(feature = "cuda")` outside `crates/aprender-serve/src/gpu/backend.rs`:
    ```bash
    scripts/check_backend_registry.sh && [ $(pmat query --regex 'cfg!\(.*feature = "(cuda|wgpu)"' --path crates/apr-cli/src | grep -v 'backend.rs' | wc -l) -eq 0 ]
    ```

---

## 2. Step-0 Discovery Defects and CLI Execution Breakages (§3 Premises)

### 2.1 Premise Count Desynchronization and Multi-Host Execution Atomicity Flaw (§3 line 108)
- **Location:** Line 108, §3.
- **Observed Text:**
  ```bash
  pmat work add "PP-066 S0: discovery ledger at HEAD" --description "Falsify the eight premises the 0.66 plan depends on; emit docs/audits/pp-066-s0-ledger.md with a verdict per premise"
  ```
- **Defect:**
  1. The description explicitly states `"Falsify the eight premises"`, but the table immediately below enumerates **23 premises** (`S0-1` through `S0-23`). The string "eight premises" is a stale artifact of the v1.0 draft.
  2. Premises dictate native command execution across four physically disparate architectures: `intel` (x86_64), `mini` (Apple Silicon M4), `lambda` (x86_64 + RTX 4090), and `gx10` (aarch64 + GB10). An autonomous `paiml-implement` agent executing in a standard runner container cannot execute commands across four physical hosts within a single local ticket execution.
- **Remediation:**
  - Update line 108 description to: `"Falsify the 23 premises (S0-1..S0-23) the 0.66 plan depends on; emit docs/audits/pp-066-s0-ledger.md with a verdict per premise"`.
  - Partition multi-host discovery into explicit per-host sub-tickets or define an SSH remote-dispatch execution harness (`scripts/s0_remote_collect.sh`) to assemble the unified ledger.

---

### 2.2 S0-1 Markdown Bold Regex Failure (`grep -n '^| 22'`)
- **Location:** Line 112, Premise `S0-1`.
- **Command:**
  ```bash
  grep -n '^| 22' docs/specifications/PP-LLAMA-001-MASTER.md
  ```
- **Failure:**
  In `PP-LLAMA-001-MASTER.md` line 364, row 22 is formatted with markdown bold delimiters:
  ```markdown
  | **22** | **instrument**: a top-2 logit margin...
  ```
  The literal regex `^| 22` returns 0 lines. Automated discovery runners will falsely declare `S0-1` FALSIFIED, erroneously triggering the "flips" clause: *"I-00 (commit the master) becomes ticket #1"*, even though the master is already committed on `main`.
- **Remediation:**
  Update the regex to accommodate markdown bolding and whitespace:
  ```bash
  grep -nE '^\|\s*(\*\*)?22(\*\*)?\s*\|' docs/specifications/PP-LLAMA-001-MASTER.md
  ```

---

### 2.3 S0-2 Invalid CLI Argument (`pmat work list --status all`)
- **Location:** Line 113, Premise `S0-2`.
- **Command:**
  ```bash
  pmat work list --status all | grep -E 'PP-(6|27|28|21|15|29|26)'
  ```
- **Failure:**
  `pmat work list` rejects `--status all` with an immediate exit:
  ```
  Error: unknown status 'all' (did you mean 'new'?)
  Valid values: planned, todo, open, pending, new, inprogress, in-progress, wip, active, started, working, blocked, stuck, waiting, on-hold, review, reviewing, pr, pending-review, completed, done, finished, closed, cancelled, canceled, dropped, wontfix
  ```
  Omitting `--status` lists all tickets across all statuses by default.
- **Remediation:**
  Remove the invalid flag:
  ```bash
  pmat work list | grep -E 'PP-(6|27|28|21|15|29|26)'
  ```

---

### 2.4 S0-4 Cargo Multi-Binary Disambiguation Error (`cargo run -p apr-cli`)
- **Location:** Line 115, Premise `S0-4`.
- **Command:**
  ```bash
  cargo run -p apr-cli --features wgpu -- serve run --backend wgpu --list-adapters
  ```
- **Failure:**
  `crates/apr-cli/Cargo.toml` defines two binaries: `apr` (lines 8–10) and `apr-corpus-ingest` (lines 12–14), without declaring `default-run`. Invoking `cargo run -p apr-cli` aborts immediately:
  ```
  error: cargo run could not determine which binary to run. Use the --bin option to specify a binary, or the default-run manifest key.
  available binaries: apr, apr-corpus-ingest
  ```
- **Remediation:**
  Add `--bin apr` to the command:
  ```bash
  cargo run -p apr-cli --bin apr --features wgpu -- serve run --backend wgpu --list-adapters
  ```
  *(Or add `default-run = "apr"` to `crates/apr-cli/Cargo.toml`).*

---

### 2.5 S0-12 Non-Existent `metal` Cargo Feature in `aprender-gpu` (PR #2849 Drift)
- **Location:** Line 123, Premise `S0-12`.
- **Command:**
  ```bash
  cargo check -p aprender-gpu --features metal; cargo test -p aprender-gpu --features metal -- metal_devices
  ```
- **Failure:**
  PR #2849 removed the `metal` feature from `crates/aprender-gpu/Cargo.toml`. Executing the check fails:
  ```
  error: package `aprender-gpu` does not have a feature named `metal`
  ```
  `crates/aprender-gpu/src/backend/metal_shaders.rs` lines 8–11 explicitly document:
  ```rust
  // These are source strings only. This crate contains no Metal dispatcher, so
  // nothing here compiles or runs them...
  ```
- **Remediation:**
  Clarify in `S0-12` that `metal_shaders.rs` contains raw MSL shader string constants only, and that `MetalBackend` was stubbed out in PR #2849. Define ticket `B-M1` (0.67) as a full re-integration and wiring task rather than device enumeration.

---

### 2.6 S0-14 & S0-19 Omission of Extended Regular Expressions (`-E`)
- **Location:** Lines 125 (`S0-14`) and 130 (`S0-19`).
- **Observed Commands:**
  - `S0-14`: `ldd target/release/apr | grep -ci 'cuda|cublas'`
  - `S0-19`: `grep -rn 'minisign|ssh-keygen -Y|signify' scripts/ .github/workflows/`
- **Failure:**
  Standard POSIX `grep` without `-E` treats `|` as a literal vertical pipe character. In `S0-14`, testing `echo "libcuda.so" | grep -ci 'cuda|cublas'` outputs `0`. It fails to match dynamic linkings, yielding a dangerous false negative `0` on binaries linked against CUDA.
- **Remediation:**
  Add `-E` to both invocations:
  - `S0-14`: `ldd target/release/apr | grep -Eci 'cuda|cublas'`
  - `S0-19`: `grep -rnE 'minisign|ssh-keygen -Y|signify' scripts/ .github/workflows/`

---

### 2.7 S0-18 File Mode Permission and Contract Count Drift (481 vs 501)
- **Location:** Line 129, Premise `S0-18`.
- **Observed Code:**
  ```bash
  scripts/pv_bin.sh; pv lint contracts/ | tail -3; grep -c 'registry: true' contracts/*.yaml
  ```
- **Failures:**
  1. `scripts/pv_bin.sh` has file mode `100644` (non-executable). Direct execution returns `Permission denied`.
  2. `grep -c 'registry: true' contracts/*.yaml` outputs 500+ individual line counts instead of a single total.
  3. Auditing the working tree reveals that matching files total **501**, not 481 (481 was measured at historical commit `b1a6324b8`).
- **Remediation:**
  Source the script, aggregate matches via `grep -l ... | wc -l`, and document that the count at current HEAD is 501:
  ```bash
  . scripts/pv_bin.sh && "$PV" lint contracts/ | tail -3; grep -l 'registry: true' contracts/*.yaml | wc -l
  ```

---

### 2.8 S0-23 & F-28 Stale Branch Protection and Comply Gate Assertions (CB-1700 Passes at HEAD)
- **Location:** Line 80 (`F-28`), Line 134 (`S0-23`), and Line 147 (`C0`).
- **Observed Claims:**
  The spec asserts that `CB-1700`, `CB-1701`, and `CB-2100` fail at HEAD, and that `required_status_checks.strict` is `false`.
- **Empirical Codebase Reality:**
  Running `pmat comply check` at HEAD reveals:
  ```
  ✓ CB-1700: Branch Protection: default branch requires ["ci / gate", "workspace-test"], >=1 approving review, and forbids force-push
  ✗ CB-1701: Supply Chain: 1 supply-chain violation(s): no required status check is known to run a blocking cargo deny check...
  ✗ CB-2100: Comply Gate Effect: 9 severity=error rule(s) unreachable from required check(s)...
  ```
  Furthermore, querying branch protection via GitHub API (`gh api repos/paiml/aprender/branches/main/protection`) shows `strict: true`.
- **Impact:**
  Asserting that `CB-1700` fails and `strict` is false contradicts the repository state. Automated executors running `S0-23` will encounter a mismatch.
- **Remediation:**
  Update `F-28`, `S0-23`, and `C0` to reflect that `CB-1700` and branch protection `strict` are already satisfied (`[V]`), isolating `CB-1701` and `CB-2100` as the active gates to resolve.

---

## 3. Structural Scope Gaps, Ticket Tracking Invariants, and Discontinuities

### 3.1 Omission of Release Criterion C14 (Fencepost Error Across Specification Hierarchy)
- **Location:** `ANALYSIS_PARTITION.md` line 18, Dispatch Prompt, and §4 lines 145–161.
- **Defect:**
  The orchestrator dispatch and partition document define the criteria range as `C0..C14` (implying 15 criteria). In `PP-066-release-spec.md` §4, only 14 criteria are present: `C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C11, C12, C13, C10`. Criterion `C14` does not exist in the document or anywhere in the repository.
- **Impact:**
  A fencepost error exists between governing dispatch metadata and the specification text. Reviewers and automated CI harnesses cannot determine whether a 15th criterion (e.g. documentation signoff, binary checksum release verification, or andon compliance) was dropped or intended.
- **Remediation:**
  Harmonize the criteria count: declare explicitly in §4 that the release criteria comprise exactly 14 items indexed `C0` through `C13`, and update `ANALYSIS_PARTITION.md` to reference `C0..C13`. If a 15th criterion was intended (such as `C14: Andon compliance for §5 cards past expiry`), define it explicitly.

---

### 3.2 Omission of Deliverable Card T-2 from §2 Scope Table and Ticket Minting Inconsistency
- **Location:** §2 line 95, §3 line 122, §5 lines 362–363, §6 line 459, Appendix A line 663.
- **Defect:**
  - Card `T-2` (`--max-seq-len honoured or refused, never clamped`) resolves the silent context-length clamp in `finetune.rs:717`.
  - In `S0-11` (line 122), the spec states: `T-2 (new card) precedes T-0`.
  - In §5 Track T (line 372), card `T-0` lists `blockers: T-0h, T-1, T-2`.
  - In Appendix A (line 663), changelog notes: `new T-2 (--max-seq-len honoured-or-refused)`.
  - **Despite being a critical 0.66 deliverable and blocker, `T-2` is completely omitted from the 0.66 lane in §2.**
  - Furthermore, in §5 line 363, the ticket minting command reads `pmat work add "T-5a: apr finetune --max-seq-len..."`, creating an unaligned identifier (`T-5a` vs `T-2`).
- **Remediation:**
  1. Add `T-2 max-seq-len honour/refuse` to the 0.66 Scope table in §2 line 95:
     ```markdown
     | **0.66 — instrumented and honest** | ... T-0h harness, T-0 four WT receipts, T-1 τ_loss, T-2 max-seq-len honour/refuse, T-3 gate REPORTING; ... |
     ```
  2. Standardize the minting command in §5 line 363 to use the card identifier:
     ```bash
     pmat work add "T-2: apr finetune --max-seq-len is honoured or refused, never clamped"
     ```

---

### 3.3 Findings Register Sequence Gaps: Missing Entries F-23 and F-24
- **Location:** Lines 76–77, §1 Table.
- **Defect:**
  The register progresses directly from `F-22` (line 76) to `**F-25**` (line 77). There are no entries for `F-23` or `F-24`. In a formal audit ledger, unexplained numbering gaps impede traceability.
- **Remediation:**
  Add an explicit note in §1 or tombstone rows in the table:
  ```markdown
  | F-23..F-24 | — | Retired / merged into F-25 during root-cause synthesis | — | — |
  ```

---

### 3.4 Dual Identifier Namespace Collisions (D-1 and T-1/T-2)
- **Location:** §0 line 33, §2 lines 95–96, §5 line 439, §6 line 459.
- **Defect:**
  1. `D-1` designates both Decision 1 in §0 ("0.66 scope: all five tracks or the split in §2") and the CUDA documentation card in §2 and §5 Track D ("D-1 · cuda-backend-architecture.md").
  2. `T-1` in 0.66 (§2, §5) designates the `τ_loss` seed variance instrument. In 0.67 (§2 line 96, §6 line 459), `T-1..T-5` designates the carried training speed levers.
- **Impact:**
  In `pmat work` ticket registries, git branch names, and automated verification receipts, reusing identical identifiers creates ambiguity.
- **Remediation:**
  - Rename the documentation card to `DOC-1` or `D-CUDA-1`. Reserve `D-*` strictly for §0 Decisions (`D-1`..`D-11`).
  - Disambiguate the 0.67 carried training levers in §0, §2, and §6 from `T-1..T-5` to `TL-1..TL-5` (Training Levers), preserving `T-1` for the 0.66 statistical instrument.

---

### 3.5 Clarification of Decision D-3 Escalation Scope vs 0.66 Release Hold
- **Location:** Line 35, Decision `D-3`.
- **Observed Code:**
  ```markdown
  | **D-3** | CI home for non-CUDA lanes: may `intel` ... host a wgpu lane ... ; may `mini` ... host nightly Metal? | **Escalate; no default.** APR-QUALITY-001 §9.1 already holds this open ⛔. B-W1, B-S*, B-M* are blocked on D-3 | ...
  ```
- **Defect:**
  `D-3` is marked ⛔ in §0 ("Decisions required before ticket #1"), which implies that 0.66 work cannot begin until `D-3` is resolved. However, in the §2 Scope table, all tickets blocked by `D-3` (`B-W1..B-W5`, `B-M1..B-M4`, `B-S1..B-S4`) are deferred to the **0.67 lane**. 0.66 does not provision non-CUDA CI lanes on `intel` or `mini`.
- **Remediation:**
  Clarify in `D-3` that this escalation blocks 0.67 execution only, and does not block 0.66 ticket #1 or 0.66 release gates.

---

### 3.6 Disambiguation of Hardware `BackendRegistry` from Workspace Crate `crates/aprender-registry` (`pacha`)
- **Location:** Line 77 (`F-25`), Line 95 (§2), Line 157 (`C11`).
- **Codebase Reality:**
  A workspace crate `crates/aprender-registry` already exists in the repository. Its manifest defines `[lib] name = "pacha"`, implementing model, dataset, and recipe registry functionality. In contrast, ticket `R-0` defines a runtime hardware backend registry located at `crates/aprender-serve/src/gpu/backend.rs`.
- **Remediation:**
  Add an explicit disambiguation note in §1 `F-25` and §2:
  *(Note: BackendRegistry for runtime hardware discovery lives in `crates/aprender-serve/src/gpu/backend.rs`, not to be confused with the existing `crates/aprender-registry` / `pacha` data lineage crate).*

---

### 3.7 Forward-Looking Script Deliverables in Release Criteria (§4)
- **Location:** Lines 149–160 (§4).
- **Codebase Reality:**
  Five scripts mandated in §4 do not exist at HEAD:
  - `C2`: `scripts/check_backend_firstclass.sh` (delivered by `B-G1`)
  - `C5`: `scripts/train_parity.sh` (delivered by `T-0h` / `T-1`)
  - `C6`: `scripts/check_crate_names.sh` (delivered by `G-1`)
  - `C10`: `scripts/check_dag_invariants.sh` (delivered by `G-4`)
  - `C11`: `scripts/check_backend_registry.sh` (delivered by `R-0`)
- **Remediation:**
  Annotate each check command in §4 with its delivering ticket handle, making it explicit to CI runners that these gates arm upon ticket landing.

---

# Minor Corrections and Typos

1. **Table Ordering Sequence in §0 (`D-8`) and §4 (`C10`):**
   - In §0 (line 43), `D-8` is placed at the end of the table after `D-11`. Relocate `D-8` between `D-7` and `D-9`.
   - In §4 (line 160), `C10` is placed at the bottom after `C13`. Relocate `C10` between `C9` and `C11`.

2. **Preamble Filename Typo (Line 3):**
   - *Current:* `rewrite of 0_66-performance-parity-report.md (the "report")`
   - *Correction:* Change to `0.66-performance-parity-report.md` (with period `0.66`, matching the committed file).

3. **Stale Line Citations in `D-7` (`error.rs`) and `S0-13` (`handlers.rs` / `wgpu_backend`):**
   - In `crates/apr-cli/src/error.rs`, the citation `86-90,218` is stale. Lines 86–90 are macro calls. The actual numerical mapping `Self::FeatureDisabled(_) => 9` is located at line **106**, and its unit test `test_feature_disabled_exit_code` is at lines **284–287**. Update citation to `crates/apr-cli/src/error.rs:106,284-287`.
   - In `crates/aprender-serve/src/cli/handlers.rs`, total line count is 483. The citation `handlers.rs:888-953` does not exist. Furthermore, in `crates/aprender-serve/src/gguf/wgpu_backend/mod.rs`, `RealizarError::UnsupportedOperation` is returned in `forward_qwen3_moe_wgpu` at lines **197–208** (lines 140–145 are doc comments). Update citation to `crates/aprender-serve/src/gguf/wgpu_backend/mod.rs:197-208`.

4. **Stale Governing Spec Status in Preamble (Lines 5–6):**
   - *Current:* `governing inference spec docs/specifications/PP-LLAMA-001-MASTER.md (v3.0 in hand; v3.1 [U])`
   - *Correction:* `PP-LLAMA-001-MASTER.md` v3.1 is already committed on `main` at HEAD. Update to `(v3.1 committed at HEAD [V])`.

5. **Preamble Mark Notation Collision (`[X]`):**
   - *Location:* Preamble lines 5 and 21–22.
   - *Text:* Line 5 cites `([X], §12)`. Line 21 re-defines the report's private `[X]` as `[A]` and states `[X]` keeps its fleet meaning (third-party, excluded from published claims).
   - *Correction:* Disambiguate line 5 to `(third-party benchmark study [X], §12)`.

6. **§0 Table Header Promise Discrepancy:**
   - *Location:* §0 Table header (line 31).
   - *Text:* Header states `recommended default (applied below, marked *pending*)`.
   - *Issue:* None of the rows (`D-1` through `D-11`) use the string `*pending*`.
   - *Correction:* Either insert `*pending*` tags into the recommended default cells as promised, or remove the clause from the table header.

7. **Date Arithmetic Precision in §2 (Line 91):**
   - *Current:* `0.66 tag target: 2026-10-23 [A]. Cycle: 49 days [C].`
   - *Correction:* From 2026-09-05 to 2026-10-23: September has 25 days remaining; October adds 23 days (25 + 23 = 48 days exclusive). Inclusive duration is 49 days. Clarify as `49 days (inclusive) [C]`.

8. **Mathematical Typography and Backtick Hygiene in `F-11` (Line 65):**
   - *Current:* `max_batch ≥ 22 has no basis=`
   - *Correction:* Enclose code identifier in backticks: `` `max_batch` >= 22 has no `basis=` ``.

9. **Typography and Bolding Inconsistency in §1 Findings Register:**
   - Rows `F-25`, `F-26`, `F-27`, and `F-28` (lines 77–80) are heavily bolded across identifiers, severities, and descriptions, whereas rows `F-1` through `F-22` use unbolded text. Standardize typography across the register.

10. **Provenance Mark Discipline Uniformity in §0:**
    - `D-1` and `D-2` carry `[C]`; `D-5`, `D-7`, `D-9`, `D-10`, `D-11` carry `[A]`; but `D-3`, `D-4`, `D-6`, and `D-8` carry no marks. Add appropriate provenance annotations (`[A]` for `D-3`, `[V]` for `D-4`, `[U]` for `D-6`).

11. **Function Renaming in `scripts/perf_gate.sh` (`arm_a_scaling` to `arm_a_self_regression`):**
    - In `F-28` (line 80), `S0-23` (line 134), and `C0` (line 147), citations reference `perf_gate.sh`'s `arm_a_scaling`. Under PP-31, line 620 of `scripts/perf_gate.sh` renamed this function to `arm_a_self_regression`. Update citations to prevent test symbol confusion.

12. **Escaped Table Cell Pipes (`\|`) Impeding Direct Shell Execution:**
    - Multiple inline code snippets in table cells (e.g. lines 113, 114, 116, 119, 120, 121, 125, 134, 147, 156) escape pipes as `\|` to preserve markdown table structure. Direct copy-pasting into bash executes a syntax error. Document that automated tool extractors must de-escape backslashes before invocation.

13. **WIP Limit Operationalization on `gx10` in §2 (Lines 99–102):**
    - The text states: `Per host, ≤ 1 speed row in flight... gx10 is one queue, ordered by expiry: master 15 (shakedown) → T-0 → S-3's gx10 leg → master 21 (0.67)`.
    - Explicitly document that training runs on `gx10` (`T-0`) consume the single speed-row WIP limit, preventing concurrent VRAM starvation with inference benchmarks.
