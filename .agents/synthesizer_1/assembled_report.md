> **Paper**: PP-066 — 0.66 performance-parity release: implementation spec (v1.5)
> **Category**: Implementation Specification / Performance Engineering
> **Segments reviewed**: 2
> **Date**: 2026-09-06

# Paper Summary

`PP-066 — 0.66 performance-parity release: implementation spec (v1.5)` (dated 2026-09-05) defines the operational implementation plan, quality invariants, and release criteria for the `0.66` release of the `paiml/aprender` monorepo. It rewrites and refines `0.66-performance-parity-report.md` into concrete, trackable `paiml-implement` engineering tickets adhering to the Toyota Way principles of Jidoka, Andon, Poka-Yoke, and Heijunka.

The core architecture operates under the principle of "correct before fast." Rather than pursuing uncalibrated performance claims across divergent hardware targets, the specification enforces a strict two-cycle partition:
1. **0.66 Release Lane ("Instrumented and Honest")**: Prioritizes infrastructure correctness, provable contract validation, and verifiable baseline telemetry. Key deliverables include a machine-derived runtime `BackendRegistry` (`R-0`, designated as ticket #1) that replaces compile-time `cfg!` feature flags with dynamic device discovery and explicit error refusal codes; provable contract enforcement (`Track P` / `PV-IMPROVE-001`); clean-room verified release assets and automated installer scripts (`R-5`–`R-7`); empirical training loss divergence baselines (`T-1`); and 14 blocking release criteria (`C0`–`C13`) anchored by strict CI branch protection gates (`C0`).
2. **0.67 Carried Lane ("Speed and Lanes")**: Defers uncalibrated GPU kernel speedups (`W-B`, `W-D`, `W-E`, `W-G`, `W-H`), speculative training speed levers (`T-1`–`T-5`), secondary platform lanes (wgpu, Metal, ROCm/HIP, wasm), and large-scale repository refactoring (51 crate renames) until stable, conformant measurement cells exist.

Every ticket in the 0.66 program follows the structured `paiml-implement` execution cycle (`AUTO-IMPL-SKILL-001`): Phase 0 (read-only discovery ledger to falsify premises), Phase 1 (acceptance planning with explicit commands $A_i$ and estimation bases), Phase 2 (RED-first test commits, where orchestrators re-verify worker PASS claims), and Phase 3 (quorum review, PR verification, and auditable receipt generation).

# Key Issues Roadmap

* **[decisions_findings_scope_discovery_criteria]:** Release criteria check commands embed unexecutable shell syntax and pseudo-code (`≥ 2`, `wc -l = ...`, missing scripts); direct execution of `scripts/pv_bin.sh` silently bypasses verification in `C12` and `S0-18`; Step-0 discovery commands fail on live code due to invalid flags, markdown bold regex flaws, and multi-binary ambiguity; and blocking ticket `T-2` is omitted from the §2 Scope table while premise counts desynchronize.
  - **Spec Compliance**: Step-0 discovery premise counts desynchronize between ticket minting (claiming 8 premises) and the table enumeration (defining 23 premises); deliverable card `T-2` (`--max-seq-len` honoured or refused) is omitted from the §2 Scope table despite being a recognized 0.66 blocker; release criterion `C14` is referenced in governing partition files but omitted from §4; and dual namespace collisions occur on `D-1` (Decision vs Ticket) and `T-1` (0.66 instrument vs 0.67 levers).
  - **Grammar & Clarity**: Check columns in §4 contain unexecutable POSIX shell commands, including mathematical Unicode glyphs (`grep -c CONFORMANT ... ≥ 2` in `C1`), pseudo-code assertions (`wc -l = ticket count; grep -L 'partial=false' empty` in `C9`), missing script paths (`script exit 0` in `C7`), literal ellipses (`curl -LsSf …/install.sh` in `C13`), sequence anomalies (`D-8` placed after `D-11`, `C10` placed after `C13`), and unexecutable compound prose assertions.
  - **Code Metrics & Status**: `scripts/pv_bin.sh` is designed strictly for sourcing; executing it in a subshell drops CLI arguments and exits 0 unconditionally, creating silent gate theater in `C12` and `S0-18`; Step-0 discovery commands break against the repository due to invalid CLI options (`pmat work list --status all`), regex bold delimiter mismatches (`grep -n '^| 22'`), multi-binary ambiguity (`cargo run -p apr-cli`), and a removed Cargo feature (`cargo check -p aprender-gpu --features metal`); while assertions regarding `CB-1700` and branch protection `strict` are stale as both already pass at HEAD.

* **[implementation_tickets_future_lanes_governance]:** Physical queue ordering on `gx10` induces a machine starvation deadlock under `WIP=1` while §5 contains four zero-slack blocker pairs violating the 6-day slack invariant; G-4 acceptance relies on invalid CLI arguments in `pmat comply check` and non-existent DAG source files; acceptance commands cite non-existent in-tree paths (`machines/clean-room`); and §10 falsifies master spec v3.1 status and row 22 availability.
  - **Spec Compliance**: Physical queue ordering on `gx10` schedules October-expiring `master 15` ahead of September-expiring `T-0` and `S-3`, deadlocking the host under `WIP=1` and inverting G-4's queue-expiry invariant; the §5 DAG contains four zero-slack blocker pairs (`0b → R-4`, `P-0.3 → P-0.6`, `P-1.1 → P-1.2`, `T-1 → T-0`) violating the mandatory 6-day slack rule; fixture counts contradict between `C11` (12 fixtures) and `R-0`/`REG` (14 fixtures); and `pv validate` cannot validate CLI JSON output streams.
  - **Grammar & Clarity**: Acceptance specifications in tickets `R-5` and `D-1` embed conversational descriptive prose and non-executable policy text; ticket minting commands desynchronize from receipt audit naming (`T-5a` for `T-2`, `T-7` for `T-3`); requirement `REG-13` omits the `FX-13:` fixture handle; and unbalanced backticks occur in contract invariants.
  - **Code Metrics & Status**: Ticket `G-4` acceptance relies on unsupported CLI flags in `pmat comply check --rule` and cites non-existent DAG source files (`docs/specifications/pp-066-dag.yaml`, `scripts/render_dag.py`); acceptance commands in `R-2` and `R-5` reference missing external directory `machines/clean-room`; and §10 verification ledger claims that master spec v3.1 is uncommitted and missing row 22 are falsified by commit `027ed889d` on `main`.

---

# Detailed Segment Reports

## decisions_findings_scope_discovery_criteria

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


## implementation_tickets_future_lanes_governance

# Summary

This document establishes the definitive, forensic **Level 2 Root Aggregated Review** for **Segment 2: `implementation_tickets_future_lanes_governance`** of `docs/specifications/PP-066-release-spec.md` (v1.5, lines 167–665). It synthesizes, cross-verifies, and deepens the evaluations produced by Level 1 Aggregators (`agg_1.md` and `agg_2.md`, representing independent candidates 1, 2, 3, and 4), grounded in empirical execution against the active repository tree at `paiml/aprender` (HEAD commit `a99236a86` / `origin/main` commit `027ed889d`).

Segment 2 constitutes the operational and governance core of the PP-066 release program, spanning nine major sections:
- **§5 0.66 Tickets — paiml-implement units**: Implementation ticket specifications across Track I (Instrument chain rows 0a–19), Track R (Runtime discovery, refusal, release packaging: `R-0`–`R-7`), Track P (Provable contracts: `P-0.1`–`P-0.6`, `P-1.1`, `P-1.2`), Track S (Speed rows: `S-1`–`S-3`), Track T (Training measurement: `T-0h`, `T-1`, `T-2`, `T-0`, `T-3`), Track B (Backends: `B-A1`, `B-G1`), Track G (Guards & DAGs: `G-1`–`G-4`), and Track D (Documentation: `D-1`).
- **§6 0.67 Lane — carried rows**: Deferred performance kernels (`W-B`, `W-D`, `W-E`, `W-G`, `W-H`, `W-F` resident attach), 0.67 training levers (`T-1`–`T-5`, `T-7` arming), hardware lanes (`B-W0`–`B-W5`, `B-M1`–`B-M4`, `B-S1`–`B-S4`, `B-A2`), PVI Phases 1.3–4.3 (#2556), and crate rename sequencing (51 PRs with shims).
- **§7 Registered Predictions**: 16 pre-experiment registered prediction rows establishing falsification criteria, kill thresholds, and re-planning triggers.
- **§8 Refusals (0.66)**: 23 non-negotiable negative constraints and invariant bounds governing the release.
- **§9 Toyota Way Targets (0.66)**: 26 quantitative quality metrics across Jidoka, Andon, Poka-Yoke, Genchi Genbutsu, Kaizen, and Heijunka.
- **§10 Verification Ledger for this document**: Evidentiary citations, proof provenance, and verification marks (`[V]`, `[C]`, `[A]`, `[U]`).
- **§11 Adjudication of `0_66-review.md`**: Formal dispositions on 33 review findings and 6 unraised findings.
- **§12 Prior-Art Register — GPU Discovery**: Comparative architectural synthesis of llama.cpp/ggml, Ollama, and llamafile mapped to discovery requirements `REG-1`..`REG-14`.
- **Appendix A — Changelog**: Complete version history tracking revisions from v1.0 through v1.5.

---

### Evaluation Dimensions and Aggregation Methodology

The audit evaluates Segment 2 across three mandatory dimensions:
1. **Spec Compliance**: Contract-per-card discipline (`kind: pattern`, absence of `registry: true`, executable falsification tests, `pv_bin.sh` usage), machine executability of acceptance commands ($A_i$), mutation-to-RED discriminators, quorum classifications, dependency DAG ordering and minimum 6-day slack rules, physical queue sequencing under `WIP=1`, registered prediction protocols and kill thresholds, refusal invariants, Toyota Way quality targets, verification ledger mark discipline (`[A]`, `[C]`, `[V]`, `[U]`), adjudication verdicts (SUSTAINED, NARROWED, REJECTED, INDETERMINATE), and prior-art register completeness.
2. **Grammar and Clarity**: Technical readability, clarity of justifications, terminology precision, architectural fidelity (e.g. `BackendRegistry::discover()`, `REG-1`..`REG-14` requirements, printed discovery blocks), table formatting, ticket minting consistency, namespace collisions, and changelog completeness.
3. **Code Metrics and Status**: Empirical verification against repository HEAD, auditing referenced crates (`crates/apr-cli`, `crates/aprender-serve`, `crates/aprender-compute`, `crates/aprender-train`, `crates/aprender-gpu`, `crates/facades`), script existence, file permissions, and CLI options (`scripts/pv_bin.sh`, `scripts/check_dag_invariants.sh`, `pmat comply check`), presence of external directories (`machines/clean-room`), and mathematical validity of calculations across §5, §6, §7, and §10.

---

### Executive Assessment

Segment 2 is an exceptionally rigorous, battle-hardened specification that establishes a new benchmark for system engineering governance. By elevating `BackendRegistry` (`R-0`) to ticket #1, deriving 14 discovery requirements directly from prior-art failure modes (§12), creating an end-to-end verified release artifact pipeline (`R-5`..`R-7`), and replacing invented performance ratios with empirical distribution anchors (`T-1`, `W-A`), it eradicates systemic configuration theater.

However, forensic verification against the active codebase reveals **critical execution defects, scheduling deadlocks, and unexecutable verification commands** that must be resolved prior to ticket execution:
1. **Critical Gate-Theater in `scripts/pv_bin.sh`**: The script is marked non-executable (`100644`) and designed strictly as a sourcing library. When invoked directly in subshells (e.g. `bash scripts/pv_bin.sh validate <file>`), it drops all CLI arguments, resolves `$PV`, and exits 0 unconditionally without performing any validation, creating a completely vacuous passing gate in `R-0` and Criterion `C12`.
2. **Physical Queue Contention and Expiry Inversion on `gx10`**: §2 orders the single `gx10` queue as `master 15 → T-0 → S-3`. However, `master 15` is blocked by rows 6 and 12 expiring on **2026-10-02** (earliest allowable expiry 2026-10-08), while `T-0` and `S-3` expire on **2026-09-26**. Under strict `WIP=1`, placing `master 15` first starves the machine throughout late September and inverts the G-4 queue-expiry invariant (`queue_pos(a) < queue_pos(b) ⇒ expires(a) ≤ expires(b)`).
3. **Four Zero-Slack Blocker Pairs in §5 DAG**: §4 C10, §5 G-4, §8, and §9 mandate a strict 6-day minimum slack between blocker and blockee (`expires(blocker) + 6d ≤ expires(blockee)`). Four pairs violate this with exactly 0 days of slack: `master 0b → R-4` (both 2026-09-19), `P-0.3 → P-0.6` (both 2026-09-19), `P-1.1 → P-1.2` (both 2026-10-10), and `T-1 → T-0` (both 2026-09-26).
4. **Falsification of Master Spec Status and Row 22 in §10**: §10 lines 567–568 and §3 `S0-1` falsely claim that master spec v3.0 lacks row 22 and that rows 19/21 expiries were moved without amendment. Empirical inspection of `docs/specifications/PP-LLAMA-001-MASTER.md` on `main` confirms that master v3.1 is committed, carries Row 22 at line 364, and that literal dates on rows 19/21 were officially replaced with `derived` in master change 3.0.24. The failure of `S0-1` was caused by a faulty regex (`grep -n '^| 22'`) failing on bolded markdown (`| **22** |`).
5. **Non-Executable Acceptance Command in G-4**: G-4 specifies `pmat comply check --rule obligation-dag docs/specifications/pp-066-dag.yaml --min-slack-days 6`. `pmat comply check` does not accept `--rule`, `--min-slack-days`, or positional file arguments; the command fails immediately with exit code 2.
6. **Missing DAG Source Files**: G-4 asserts that §5/§6 tables are regenerated by `scripts/render_dag.py` from `docs/specifications/pp-066-dag.yaml`. Neither file exists in the repository.
7. **Semantic Tooling Mismatch in R-0**: R-0 prescribes `scripts/pv_bin.sh validate contracts/apr-devices-schema-v1.yaml and every apr devices --json output validates against it`. `pv validate` only validates YAML contract syntax, not CLI JSON output streams.
8. **Non-Existent External Path in Clean-Room Commands**: R-2 and R-5 specify `make -C machines/clean-room clean-room-p1 exit 0`. No `machines` directory exists in `aprender`; clean-room runner definitions reside in `paiml/infra`.

---

### Definitive Defect Matrix for Segment 2

| ID | Location | Summary of Verified Issue | Severity | Primary Dimension |
|---|---|---|---|---|
| **DEF-01** | §5 line 208, §4 C12, §8 line 510 | `scripts/pv_bin.sh` is non-executable and drops all CLI arguments; subshell execution exits 0 unconditionally, creating silent gate theater in R-0 and C12 | **S1** | Spec Compliance / Metrics |
| **DEF-02** | §2 lines 100–102, §5 lines 198, 336, 372 | `gx10` physical queue order (`master 15 → T-0 → S-3`) starves `gx10` under `WIP=1` and inverts G-4 invariant `queue_pos(a) < queue_pos(b) ⇒ expires(a) ≤ expires(b)` | **S1** | Spec Compliance |
| **DEF-03** | §5 lines 265, 302, 308, 372 | Four zero-slack (0 days) blocker pairs in §5 (`0b → R-4`, `P-0.3 → P-0.6`, `P-1.1 → P-1.2`, `T-1 → T-0`) violate C10, G-4, §8, and §9 | **S1** | Spec Compliance |
| **DEF-04** | §10 lines 567–568, §3 S0-1 | Falsified claims in §10: `PP-LLAMA-001-MASTER.md` v3.1 is committed on `main` and carries Row 22 at line 364; rows 19 and 21 have `derived` expiries struck in master v3.0.24 | **S1** | Code Metrics |
| **DEF-05** | §5 line 432 | `pmat comply check` in G-4 exits 2 due to unhandled flags (`--rule`, `--min-slack-days`); cross-repo CLI coupling in `aprender` ticket | **S1** | Spec Compliance / Metrics |
| **DEF-06** | §4 C10, §5 lines 431–433 | Premature automation claim: `docs/specifications/pp-066-dag.yaml` and `scripts/render_dag.py` do not exist in the repository | **S2** | Code Metrics |
| **DEF-07** | §5 lines 250, 274 | Acceptance command requires `make -C machines/clean-room clean-room-p1` which does not exist in `paiml/aprender` | **S2** | Code Metrics |
| **DEF-08** | §5 line 208 | `pv validate` only validates YAML contract schemas; cannot validate CLI JSON output (`apr devices --json`) against schemas | **S2** | Spec Compliance |
| **DEF-09** | §4 C11, §5 line 208, §9 line 539 | Contradiction: C11 specifies "12 fixtures" while R-0, REG table, §9, and Appendix A mandate 14 fixtures (`REG-1`..`REG-14`) | **S2** | Grammar & Clarity |
| **DEF-10** | §9 line 544, §3 line 109 | Stale metrics: §9 claims "9/9 Step-0 premises" and §3 claims "eight premises", but §3 defines 23 premises (`S0-1`..`S0-23`) | **S2** | Grammar & Clarity |
| **DEF-11** | §5 line 226, §12 table | Requirement `REG-10` is omitted from §12's REG column for llama.cpp and Ollama; `REG-9` omits llama.cpp | **S2** | Spec Compliance |
| **DEF-12** | §5 lines 295–313, 420, 427 | Card discipline violations: Track P lacks YAML paths; G-2 is an unformatted one-liner; G-3 lacks mutation-to-RED and sets `quorum: none` | **S2** | Spec Compliance |
| **DEF-13** | §0 D-7, §5 lines 225, 230, §8 line 508 | Refusal exit code ambiguity: `FeatureDisabled` (exit 9) vs `NotImplemented` (exit 12); tests asserting on `--tensor-split` risk asserting wrong code | **S2** | Spec Compliance |
| **DEF-14** | §7 line 477 | Incomplete prediction kill threshold for S-2: only defines `share > 30 %`, omitting prefill throughput threshold (> 5,700 tok/s) | **S2** | Spec Compliance |
| **DEF-15** | §5 lines 292, 359, 375, 393, 427, 444 | Missing discrimination clauses and truncated mutation specifications in cards R-7, T-0, T-1, B-A1, G-3, and D-1 | **S2** | Spec Compliance |
| **DEF-16** | §5 line 363, §2 line 96, §5 line 379 | Ticket handle desynchronization: T-2 mints `"T-5a: ..."`, T-3 mints `"T-7 (REPORTING): ..."`; T-2 omitted from §2 Scope table | **S2** | Spec Compliance |
| **DEF-17** | §0 line 33, §5 line 439, §6 line 459 | Dual namespace collisions: `D-1` (Decision 1 vs Doc Ticket D-1); `T-1` (0.66 τ_loss vs 0.67 levers) | **S2** | Spec Compliance |
| **DEF-18** | §6 lines 463–464, §10 line 575 | Inaccurate architectural claims: `aprender-serve` is not `cfg(not(wasm32))` entirely; `aprender-gpu` has inert Metal shader strings without a driver dispatcher | **S3** | Code Metrics |
| **DEF-19** | §5 line 229 | Requirement `REG-13` lacks fixture handle prefix `FX-13:` present on all other 13 rows | **S3** | Grammar & Clarity |
| **DEF-20** | §7 line 489 | Mutation count mismatch: §7 G-1 claims "both mutations RED" while G-1 defines three mutations | **S3** | Grammar & Clarity |
| **DEF-21** | §5 lines 206, 307 | Temporal paradox: R-0 lands 2026-09-19 without fail-closed contract enforcement; no ticket schedules adding `#[contract]` post-P-1.1 | **S3** | Spec Compliance |
| **DEF-22** | §5 lines 274, 442 | Acceptance fields in R-5 and D-1 embed descriptive prose and non-executable policy clauses | **S3** | Spec Compliance |
| **DEF-23** | §10 lines 565, 575 | Stale line citations in §10: `error.rs:106`, root `Cargo.toml:666`, crate qualification needed for `handlers.rs` and `attention.rs` | **S3** | Code Metrics |
| **DEF-24** | §11 lines 630, 633 | Duplicate section heading in §11 adjudication (`Five findings no reviewer raised`) | **S3** | Grammar & Clarity |
| **DEF-25** | §5 line 275 | Markdown formatting syntax error: unbalanced backtick in R-5 contract invariant | **S3** | Grammar & Clarity |
| **DEF-26** | §5 lines 411–417, §4 C6, §9 line 546 | Crate name guard allow-list count ambiguity: 51 offending crates vs 52 rows (inclusive of D-5 front-door row) | **S3** | Grammar & Clarity |

---

### Verification and Test Traceability Matrix

| Item Audited | Referenced in Spec | Actual Codebase State | Empirical Verdict |
|---|---|---|---|
| `scripts/pv_bin.sh` subshell execution | §5 R-0 A, §4 C12 | Mode `100644`; drops `$@` and exits 0 immediately without executing `$PV` | ⚠️ **FAIL (Silent Bypass)** |
| DAG minimum slack (≥ 6 days) | §4 C10, §5 G-4 | 4 zero-slack blocker pairs confirmed (`R-4`, `P-0.6`, `P-1.2`, `T-0`) | ⚠️ **FAIL (Schedule Contradiction)** |
| `gx10` physical queue order | §2, §9, §5 G-4 | `master 15` blocked by Oct 2 rows, blocking Sept 26 `T-0`/`S-3` | ⚠️ **FAIL (Physical Deadlock)** |
| G-4 `pmat comply check --rule` | §5 G-4 A | Exits 2: `error: unexpected argument '--rule' found` | ⚠️ **FAIL (Invalid CLI)** |
| `pp-066-dag.yaml` & `render_dag.py` | §4 C10, §5 G-4 | Neither file exists in tree | ⚠️ **FAIL (Missing Files)** |
| Clean-room path `machines/clean-room` | §5 R-2 A, R-5 A | Directory does not exist in `aprender` worktree | ⚠️ **FAIL (Non-Existent Path)** |
| Step-0 premise count | §9 line 544 | Declares 9/9; §3 defines 23 premises (`S0-1`..`S0-23`) | ⚠️ **FAIL (Scope Drift)** |
| Failure catalogue fixtures | §4 C11 | Requires 12 fixtures; §5 R-0, §9, and Appendix A mandate 14 fixtures | ⚠️ **FAIL (Internal Contradiction)** |
| Standalone `claims-cite` command | §5 D-1 A | Tool does not exist; actual script is `check_perf_claims_cite_receipts.sh` | ⚠️ **FAIL (Non-Existent Tool)** |
| Master spec status on `main` | §10 line 579, §3 S0-1 | Committed at HEAD; v3.1 with row 22 at line 364 | ⚠️ **FAIL (False Negative in S0-1)** |
| `CliError::FeatureDisabled` | §0 D-7, §5 R-0, §10 | Mapped to `9` in `crates/apr-cli/src/error.rs:106` | ✅ **VERIFIED** |
| `accel.rs:28` feature check | §1 F-25, §5 R-0, §10 | `cfg!(any(feature = "cuda", feature = "wgpu"))` in `accel.rs:28` | ✅ **VERIFIED** |
| `accel.rs:114` 3-surface test | §5 R-0 A, §10 | Verifies `run`, `chat`, `serve` surfaces in `crates/apr-cli` | ✅ **VERIFIED** |
| MSL kernels in `aprender-gpu` | §3 S0-12, §6 B-M1, §10 | 13 `kernel void` source strings in `metal_shaders.rs` (no dispatcher) | ✅ **VERIFIED** |
| `MetalBackend` struct location | §3 S0-12, §6 B-M1, §10 | Defined at `crates/aprender-gpu/src/backend/mod.rs:67` (returns false) | ✅ **VERIFIED** |
| `Q4K_GEMV_SHADER` in compute | §3 S0-13, §6 B-W0, §10 | Present at line 555 of `basic_ops.rs` | ✅ **VERIFIED** |
| `OwnedQuantizedModelWgpu` stub | §3 S0-13, §6 B-W0, §10 | Returns `UnsupportedOperation` in `wgpu_backend/mod.rs:197` | ✅ **VERIFIED** |
| Double `[lib] name = "aprender"` | §0 D-5, §5 G-2 | Root `Cargo.toml:693` & `aprender-core/Cargo.toml:72` | ✅ **VERIFIED** |
| Facades separate workspace | §5 G-1, §6 renames | `crates/facades/Cargo.toml` prevents `.rlib` collisions | ✅ **VERIFIED** |
| 7.61B f32 expansion arithmetic | §6 B-W0, §10 | $7.61 \times 10^9 \times 4\text{ B} = 30.44\text{ GB}$ | ✅ **VERIFIED** |
| 7B QLoRA seq 2048 activation | §5 T-0h, §7 T-0, §10 | $7 \times 28 \times (4 \times 2048 \times 3584 \times 4\text{ B}) \approx 23.02\text{ GB}$ | ✅ **VERIFIED** |
| Wasm32 linear memory bound | §6 B-S1..S4, §10 | $2^{32}\text{ B} = 4.00\text{ GiB}$ ($4{,}294{,}967{,}296\text{ B}$) | ✅ **VERIFIED** |
| Comparator bar scaling ratio | §1 F-14, §6 W-E, §10 | $484.7 / 171.5 = 2.826... \approx 2.83\times$ | ✅ **VERIFIED** |
| Prefill prediction ratio | §5 S-2 note, §7 S-2 | $5{,}700 / 10{,}399 = 0.548... \approx 0.55\times$ | ✅ **VERIFIED** |

---

# Potential Mistakes and Improvements

## 1. Execution and Tooling Vulnerabilities

### 1.1 Critical Gate-Theater: Non-Executable & Argument-Dropping Behavior in `scripts/pv_bin.sh`
- **Observation**:
  - §4 C12 (line 159): `scripts/pv_bin.sh lint contracts/ --binding contracts/aprender/binding.yaml --strict-test-binding exit 0`
  - §5 R-0 Acceptance (line 208): `scripts/pv_bin.sh validate contracts/apr-devices-schema-v1.yaml`
  - §8 Refusals (line 510): `No pv invocation against the binary on PATH; scripts/pv_bin.sh builds it from the tree.`
  - Forensic inspection of `scripts/pv_bin.sh` in the worktree reveals:
    1. File permission mode is `100644` (`-rw-rw-r--`, non-executable). Invoking `./scripts/pv_bin.sh` results in `bash: ./scripts/pv_bin.sh: Permission denied` (exit 126).
    2. The header explicitly mandates sourcing (lines 3–6):
       ```bash
       # Source it, never execute it:
       #     . scripts/pv_bin.sh || exit 1
       #     "$PV" lint contracts/
       ```
    3. The script terminates at line 662 with:
       ```bash
       pv_bin_assert_fresh "$PV" || return 1 2>/dev/null || exit 1
       export PV
       ```
    4. Running `bash scripts/pv_bin.sh validate non_existent.yaml` drops all CLI arguments and exits 0 immediately:
       ```bash
       $ bash scripts/pv_bin.sh validate non_existent.yaml; echo "exit: $?"
       exit: 0
       ```
- **Logic Chain**:
  1. The specification treats `scripts/pv_bin.sh` as a drop-in executable CLI wrapper for `pv`.
  2. Because `scripts/pv_bin.sh` was implemented purely as a sourcing helper, it contains no parameter dispatcher (`"$@"`).
  3. Consequently, any subshell execution (`bash scripts/pv_bin.sh <command> <args>`) silently exports `$PV` in the subshell, ignores all arguments, and returns exit code 0.
  4. This creates a severe silent-pass gate theater condition—the exact failure mode that finding `F-26` and criterion `C12` were explicitly created to eliminate.
- **Remediation**:
  1. Add an argument dispatch block at the end of `scripts/pv_bin.sh` (lines 662–665):
     ```bash
     export PV
     if [ "${BASH_SOURCE[0]}" = "$0" ]; then
         if [ "$#" -gt 0 ]; then
             exec "$PV" "$@"
         else
             printf '%s\n' "$PV"
         fi
     fi
     ```
  2. Mark the script executable: `chmod +x scripts/pv_bin.sh`.
  3. Alternatively, update all normative acceptance commands in §4 C12 and §5 R-0 to use explicit sourcing:
     `. scripts/pv_bin.sh && "$PV" validate contracts/apr-devices-schema-v1.yaml`

---

### 1.2 Invalid CLI Arguments & Cross-Repo Coupling in Ticket G-4 (`pmat comply check --rule`)
- **Observation**:
  - In §5 line 432 (Ticket G-4 acceptance):
    `pmat comply check --rule obligation-dag docs/specifications/pp-066-dag.yaml --min-slack-days 6 exit 0`
  - Executing this command produces:
    ```text
    $ pmat comply check --rule obligation-dag docs/specifications/pp-066-dag.yaml --min-slack-days 6
    error: unexpected argument '--rule' found
    Usage: pmat comply check [OPTIONS]
    exit: 2
    ```
- **Logic Chain**:
  1. `pmat comply check` supports flags like `--mode`, `--path`, `--strict`, `--verbose`, `--format`, and `--color`. It does not accept `--rule`, `--min-slack-days`, or positional file arguments.
  2. Furthermore, `pmat` is an external binary built from `paiml/pmat`. Ticket `G-4` is assigned to repository `aprender` (`aprender · feat/g4-dag-invariants`).
  3. An engineer implementing `G-4` within `aprender` cannot alter `pmat`'s CLI parser. Attempting to run G-4's acceptance command will immediately fail CI with exit code 2.
- **Remediation**:
  Re-anchor the G-4 acceptance command to the dedicated standalone script specified in §4 Criterion C10:
  `bash scripts/check_dag_invariants.sh docs/specifications/pp-066-dag.yaml --min-slack-days 6`
  If a native `pmat comply` check rule is planned, file an upstream issue on `paiml/pmat` and decouple the `aprender` release gate from unreleased `pmat` features.

---

### 1.3 Premature Claim of Generated Tables: Missing DAG Files (`pp-066-dag.yaml` and `scripts/render_dag.py`)
- **Observation**:
  - G-4 states: `scripts/render_dag.py regenerates §5/§6 tables byte-identical to the committed spec (drift = RED)` (line 432).
  - Criterion C10 specifies: `scripts/check_dag_invariants.sh docs/specifications/pp-066-dag.yaml --min-slack-days 6 exit 0` (line 161).
  - Neither `docs/specifications/pp-066-dag.yaml` nor `scripts/render_dag.py` exists in the repository.
- **Logic Chain**:
  Describing the spec's §5 and §6 markdown tables as already being "rendered from the yaml, never hand-edited" is factually premature when the source YAML and generator script have not yet been authored. If a CI gate checks for table drift against a non-existent YAML, the gate cannot execute.
- **Remediation**:
  Clarify in G-4 that `docs/specifications/pp-066-dag.yaml` and `scripts/render_dag.py` are deliverables of ticket G-4 itself, and that until G-4 lands, the markdown tables in §5/§6 serve as the authoritative baseline.

---

### 1.4 Semantic & Tooling Mismatch for Output Schema Validation in R-0 (`pv validate` vs CLI JSON)
- **Observation**:
  - In §5 line 208 (`R-0` Acceptance):
    `scripts/pv_bin.sh validate contracts/apr-devices-schema-v1.yaml and every apr devices --json output validates against it`
  - `pv validate --help` shows:
    ```text
    Validate a YAML kernel contract
    Usage: pv validate [OPTIONS] <CONTRACT>
    ```
- **Logic Chain**:
  1. `pv validate` is dedicated exclusively to verifying provable contract YAML syntax against the internal `provable-contracts` schema.
  2. `pv validate` has no functionality to ingest JSON streams emitted by CLI binaries and validate them against JSON Schema definitions.
  3. The phrase `and every apr devices --json output validates against it` is unexecutable human prose embedded within an acceptance test command.
- **Remediation**:
  Split this into two distinct, machine-executable operations:
  `. scripts/pv_bin.sh && "$PV" validate contracts/apr-devices-schema-v1.yaml && cargo test -p apr-cli --test devices_json_schema`

---

### 1.5 Non-Existent Standalone Tool `claims-cite` in D-1
- **Observation**:
  - In §5 line 443 (`D-1` Acceptance):
    `the existing claims-cite check (it passed on #2868 [A]) extended to API: sentences rather than a new script — scripts/check_doc_citations.sh only if claims-cite cannot express the rule...`
  - Searching git confirms that no tool or command named `claims-cite` exists in the tree or on PATH. The actual in-tree script executed in CI is `scripts/check_perf_claims_cite_receipts.sh` (`.github/workflows/ci.yml:1111`).
- **Logic Chain**:
  Referring to a non-existent binary as "the existing `claims-cite` check" creates a broken dependency. Implementers cannot extend a tool that does not exist.
- **Remediation**:
  Specify `scripts/check_doc_citations.sh` as the primary deliverable for D-1:
  `bash scripts/check_doc_citations.sh docs/specifications/cuda-backend-architecture.md`

---

### 1.6 Non-Existent External Path in Clean-Room Acceptance Commands (R-2, R-5)
- **Observation**:
  - In §5 line 250 (`R-2`) and line 274 (`R-5`):
    `make -C machines/clean-room clean-room-p1 exit 0`
  - In the `aprender` worktree, `machines/clean-room` does not exist:
    ```bash
    $ ls -d machines/clean-room
    ls: cannot access 'machines/clean-room': No such file or directory
    ```
- **Logic Chain**:
  1. The clean-room runner definitions reside in the external repository `paiml/infra`.
  2. Executing `make -C machines/clean-room clean-room-p1` inside the `aprender` root directory aborts immediately with `No such file or directory`.
- **Remediation**:
  Qualify the path relative to the environment (`make -C ../infra/machines/clean-room clean-room-p1`) or delegate verification to a workflow receipt emitted by the `intel` clean-room runner.

---

## 2. Dependency DAG, Scheduling & Queue Deadlocks

### 2.1 Physical Queue Deadlock & Invariant Inversion on `gx10` (`master 15` vs `T-0` / `S-3`)
- **Observation**:
  - §2 lines 100–102 (WIP caps):
    `gx10 is one queue, ordered by expiry: master 15 (shakedown) → T-0 → S-3's gx10 leg → master 21 (0.67)`
  - §5 Track I line 198:
    `15 | gx10 shakedown cell, W1, n ≥ 5 interleaved | derived (blocked by 0b, 0c, 1, 6, 7, 12) | first gx10 queue slot`
  - §5 Track I lines 192 & 196:
    Master row 6 (`GET /v1/effective-config`) expires `2026-10-02`.
    Master row 12 (`perf workflow concurrency`) expires `2026-10-02`.
  - §5 G-4 contract invariant (line 434):
    `queue_pos(host, a) < queue_pos(host, b) ⇒ expires(a) ≤ expires(b)`
  - §5 Track T line 372:
    `T-0 · the four WT receipts ... host: lambda, gx10 (queue slot 2) · expiry 2026-09-26`
  - §5 Track S line 336:
    `S-3 · W-B bench ... host: lambda, then gx10 (queue slot 3) · expiry 2026-09-26`
- **Logic Chain**:
  1. Master row 15 is blocked by rows 6 and 12, both expiring `2026-10-02`.
  2. Under the mandatory 6-day slack rule (`expires(blockee) ≥ expires(blocker) + 6d`), row 15 cannot expire before `2026-10-08`.
  3. Row 15 is assigned queue slot 1 on `gx10`.
  4. Under §2's strict WIP limit of $\le 1$ speed row per host (`WIP=1`), queue slot 2 (`T-0`) cannot execute until queue slot 1 (`master 15`) completes.
  5. Therefore, `T-0` cannot execute until after `2026-10-08`.
  6. However, `T-0`'s mandated expiry is `2026-09-26`—12 days before `master 15` can run.
  7. Furthermore, `queue_pos(15) < queue_pos(T-0)` implies `expires(15) ≤ expires(T-0)`. But `expires(15) ≥ 2026-10-08 > 2026-09-26 = expires(T-0)`. This directly inverts G-4's queue-order invariant.
- **Remediation**:
  Re-sequence the physical queue on `gx10`:
  - `T-0` (receipts) and `S-3` (GEMV bench) must occupy `gx10` queue slots 1 and 2 during late September, as their blockers (`T-0h`, `T-1`, `T-2`, `master 1`) clear between September 19 and September 26.
  - `master 15` (shakedown cell) must occupy queue slot 3 in early October, executing after rows 6 and 12 land on October 02.

---

### 2.2 DAG Invariant Violations: Four Zero-Slack Blocker Pairs in §5
- **Observation**:
  - §4 Criterion C10 (line 161), §5 G-4 (line 434), §8 Refusals (line 522), and §9 Toyota Way (line 551) mandate:
    `0 zero-slack blocker pairs (min 6 days)` and `expires(blocker) + 6d ≤ expires(blockee)`.
  - In §5, four blocker pairs exhibit exactly 0 days of slack:
    1. **Master 0b $\to$ R-4**:
       - `master 0b` (sampler pin) expires `2026-09-19` (line 186).
       - `R-4` (W5 CLI wall-clock) lists `blockers: master 0b` and expires `2026-09-19` (line 266). Slack = 0 days.
    2. **P-0.3 $\to$ P-0.6**:
       - `P-0.3` (proof credit from evidence only) expires `2026-09-19` (line 302).
       - `P-0.6` (`pv lint` CI step) lists `blockers: P-0.3` and expires `2026-09-19` (line 303). Slack = 0 days.
    3. **P-1.1 $\to$ P-1.2**:
       - `P-1.1` (`#[contract]` fails closed) expires `2026-10-10` (line 308).
       - `P-1.2` (repoint dangling contracts) lists `blockers: P-1.1` and expires `2026-10-10` (line 309). Slack = 0 days.
    4. **T-1 $\to$ T-0**:
       - `T-1` (τ_loss derived) expires `2026-09-26` (line 357).
       - `T-0` (four WT receipts) lists `blockers: T-0h, T-1, T-2` and expires `2026-09-26` (line 373). Slack = 0 days.
- **Logic Chain**:
  1. The 6-day slack rule prevents cascading schedule failures when a blocker PR slips by 24 hours.
  2. For `P-1.1` and `P-1.2`, the spec explicitly notes `(same PR as P-1.1: they become compile errors)`. Units landing in the same PR are a single compound ticket, not separate DAG nodes.
  3. For `T-1` and `T-0`, `T-1` derives the loss tolerance threshold $\tau_{\text{loss}}$ on `gx10`, while `T-0` runs the 4 WT receipts. `T-0` cannot evaluate loss validity before `T-1` completes. Co-dating them on September 26 leaves zero margin.
- **Remediation**:
  - Move `R-4` expiry from `2026-09-19` to `2026-09-26` (7 days after 0b).
  - Move `P-0.3` earlier to `2026-09-12`, leaving 7 days of slack to `P-0.6` at `2026-09-19`.
  - Merge `P-1.2` into `P-1.1` as a single atomic compound ticket (`P-1.1+1.2`).
  - Move `T-1` earlier to `2026-09-19` (aligned with `T-0h`), leaving 7 days to `T-0` at `2026-09-26`.

---

### 2.3 Temporal Paradox in Contract Enforcement Between R-0 and P-1.1
- **Observation**:
  - `R-0` is ticket #1 of 0.66, expiring on **2026-09-19** (line 207).
  - `P-1.1` (`#[contract]` fails closed) does not expire until **2026-10-10** (line 308).
  - Line 308 acknowledges: "without it `#[contract]` on any 0.66 function is decoration; after it, R-0's `discover()` can carry `#[contract("apr-backend-registry-v1", …)]` and mean it".
- **Logic Chain**:
  1. When `R-0` merges on September 19, `#[contract]` is decorative.
  2. When `P-1.1` merges on October 10, no ticket in §5 or §6 is scheduled to decorate `R-0::discover()` with `#[contract]`.
  3. Ticket `P-1.2` is scoped exclusively to repointing the 10 existing dangling contract sites.
  4. Consequently, `R-0` will ship in release 0.66 without fail-closed contract enforcement.
- **Remediation**:
  Add an explicit acceptance task in `P-1.2` or mint a follow-up card `R-0b` to decorate `BackendRegistry::discover()` with `#[contract("apr-backend-registry-v1", ...)]` immediately after `P-1.1` merges.

---

## 3. Falsification of Master Spec Status, Row 22, and Expiries in §10

### 3.1 Falsified Claims in §10: Master Spec v3.1 Committed on `main`, Row 22 Present, Expiries `derived`
- **Observation**:
  - In §10 line 568: `Master v3.0 §12 has rows 0a–0e, 1–21; no row 22 | [V] | PP-LLAMA-001-MASTER-v3.md read here`
  - In §10 line 569: `Master expiries rows 19/20/21 = 10-16 / 10-23 / 11-06 | [V] | same`
  - In §10 line 580: `PP-LLAMA-001-MASTER.md is committed on main | [U] — review segment 2 asserts it is uncommitted at the explorer's checkout | S0-1`
- **Empirical Codebase Evidence**:
  Direct inspection of `docs/specifications/PP-LLAMA-001-MASTER.md` in git (at commit `027ed889d` on `origin/main`) demonstrates:
  1. Title line: `# PP-LLAMA-001 v3.1 — MASTER — Inference performance parity with llama.cpp`.
  2. Line 364 contains Row 22:
     ```markdown
     | **22** | **instrument**: a top-2 logit margin per generated token on the wire (`logprobs` on the SSE delta, both engines)... Expires **2026-10-15** |
     ```
  3. Appendix D line 407 (Changelog v3.1, 2026-09-02):
     `3.1 | 2026-09-02 | ... §12 row 22 (margin instrument) added so (c) can become a gate ...`
  4. Lines 361–363 and Appendix D line 432 confirm that literal dates on rows 19 and 21 were replaced with `derived`:
     ```markdown
     | 3.0.24 | 2026-09-02 | §12 rows 19 and 21: the literal expiries (2026-10-16, 2026-11-06) are replaced by `derived` — both rows are blocked by live rows (1 and 2)...
     ```
- **Logic Chain**:
  1. The master specification `PP-LLAMA-001-MASTER.md` is **already committed on `main`**, is at version **v3.1**, and contains **row 22**.
  2. PP-066 falsely claimed row 22 was absent (`[V]`) and accused the parity report of moving expiries on rows 19/21 without an amendment (finding `F-3b`, decision `D-4`). In reality, the master spec itself officially struck those literal dates in v3.0.24.
- **Remediation**:
  - Update §10 line 568: mark row 22 as `[V] CONFIRMED present at line 364 of PP-LLAMA-001-MASTER.md v3.1`.
  - Update §10 line 569: record that rows 19 and 21 have `derived` expiries.
  - Update §10 line 580: mark master commit on `main` as `[V] CONFIRMED`.
  - Withdraw finding `F-3b` and update `D-4` to reflect that rows 19 and 21 derive their dates from blockers.

---

### 3.2 Faulty Regex in Discovery Premise S0-1 Generating False Negatives (`grep -n '^| 22'`)
- **Observation**:
  - In §3 line 113 (`S0-1`):
    `grep -n '^| 22' docs/specifications/PP-LLAMA-001-MASTER.md`
- **Logic Chain**:
  1. In markdown tables, table row numbers are formatted in bold: `| **22** |`.
  2. The regex `^| 22` searches strictly for a pipe followed by a space and literal `22`. It fails on bold asterisks `**22**`.
  3. This regex defect produced an artificial false negative, which was misinterpreted as proof that row 22 was missing.
- **Remediation**:
  Fix the `S0-1` verification command:
  `grep -nE '^\|[[:space:]]*(\*\*)?22(\*\*)?[[:space:]]*\|' docs/specifications/PP-LLAMA-001-MASTER.md`

---

## 4. Contract Discipline, Mutations & Falsification Deficits

### 4.1 Contract-Per-Card Discipline Short-Circuit in Track P
- **Observation**:
  - §5 top (line 176) mandates: "Contract discipline for every card (F-26)... Every PP-066 contract: `kind: pattern`... Every card names its contract".
  - In Track P (lines 298–310), cards `P-0.1` through `P-1.2` omit YAML contract paths and state: "the doc's deliverable/falsifier columns are the cards' A/mutation and are not restated."
- **Logic Chain**:
  This introduces an unacknowledged exception to the universal contract-per-card mandate. If Track P cards do not generate contracts under `contracts/`, they cannot be evaluated by C12's strict test binding gate.
- **Remediation**:
  Add an explicit policy note in Track P:
  `contract: self-verifying via FALSIFY-PVI-nnn gates; exempt from pattern-contract requirement.`

---

### 4.2 Card Formatting & Quorum Violations in Governance Track (G-2 One-Liner, G-3 `quorum: none`)
- **Observation**:
  - In §5 line 421, `G-2` is an unformatted one-liner lacking acceptance commands, contract specifications, and mutation definitions.
  - In §5 lines 426–428, `G-3` specifies `contract: none` and `quorum: none`, and defines no `mutation → RED` clause.
- **Logic Chain**:
  `G-2` and `G-3` bypass standard card schema formatting. In particular, `quorum: none` violates the `paiml-implement` lifecycle requirement that all tickets require at least `review-only` quorum.
- **Remediation**:
  - Format `G-2` into a full card schema with explicit acceptance criteria.
  - Update `G-3` line 428 to specify `quorum: review-only` and add a mutation clause.

---

### 4.3 Missing Discrimination Clauses and Truncated Mutation Specifications
- **Observation**:
  The ticket template (lines 170–172) requires:
  `mutation → RED and what it must not trip (discrimination)`
  Multiple cards fail to provide discrimination specifications:
  - **R-7** (line 293): `mutation → RED: change the README one-liner's URL → the grep test RED.` (No discrimination clause).
  - **T-0** (line 376): `mutation → RED: a receipt with partial=true → validate RED.` (No discrimination clause).
  - **T-1** (line 360): `mutation → RED: hand-type τ → validator RED (no sha). Discriminates: yes.` (Truncated to "yes").
  - **B-A1** (line 394): `mutation → RED: type a date instead of the anchor → RED. Discriminates: yes.` (Truncated to "yes").
  - **G-3** (lines 423–428): Completely omits both `mutation → RED` and `discrimination`.
  - **D-1** (line 445): `mutation → RED: strip one citation → RED.` (No discrimination clause).
- **Logic Chain**:
  A mutation causing catastrophic build or test failure is not a discriminating check. The discrimination clause is required to prove that the test specifically isolates the mutated property while leaving adjacent behaviors intact.
- **Remediation**:
  1. In R-7, add: `Discriminates: check_readme_claims.sh passes on all other assertions (crate count, command count).`
  2. In T-0, add: `Discriminates: receipts with partial=false validate cleanly; canary training execution is unaffected.`
  3. In T-1, replace "yes" with: `Discriminates: valid tau_loss.json receipts with correct sha256 pass validation.`
  4. In B-A1, replace "yes" with: `Discriminates: valid anchored cells in perf-matrix.yaml parse without error.`
  5. In G-3, add: `mutation → RED: introduce a duplicate use-site pattern → count mismatch RED. Discriminates: crate name guard stays GREEN.`
  6. In D-1, add: `Discriminates: valid API citations in unaffected sections remain GREEN.`

---

### 4.4 Incomplete Registered Prediction Kill Threshold for S-2 in §7
- **Observation**:
  - In §7 (line 478), the registered prediction for S-2 (W-C) states:
    `copies+allocs < 10 % of CUDA API time; prefill > 5,700 tok/s`
  - The "prediction killed if" column only defines:
    `share > 30 %`
- **Logic Chain**:
  The prediction has two explicit metrics (API share and prefill throughput). If copies and allocations consume 8% of API time, but prefill achieves only 2,000 tok/s, the kill condition (`share > 30%`) does not trigger, even though the throughput prediction failed by more than half.
- **Remediation**:
  Update the kill condition in line 478 to:
  `share > 30 % ∨ prefill_tok_s < 4,000`

---

## 5. Scope Drift, Internal Contradictions & Architectural Inaccuracies

### 5.1 Fixture Count Contradiction: Release Gate C11 (12 Fixtures) vs REG-1..REG-14 (14 Fixtures)
- **Observation**:
  - §4 C11 (line 158): `the failure-catalogue case table (R-0 §REG, 12 fixtures) is green...`
  - §5 R-0 Acceptance (line 208): `cargo test -p apr-cli --test registry_failure_catalogue (FX-1..FX-14 below...)`
  - §5 Requirements table (lines 215–231): Contains **14 requirements** (`REG-1`..`REG-14`).
  - §9 line 540: `14/14 REG requirements with a committed fixture`.
  - Appendix A line 661: `R-0 gains REG-1..REG-14 with a 14-fixture failure catalogue`.
- **Logic Chain**:
  When `REG-13` and `REG-14` were added in v1.4, §5, §9, and Appendix A were updated to 14 fixtures. Criterion `C11` was overlooked, retaining the stale count `12 fixtures`.
- **Remediation**:
  Update §4 C11 line 158 to replace `(R-0 §REG, 12 fixtures)` with `(R-0 §REG, 14 fixtures)`.

---

### 5.2 Premise Count Drift: §9 Toyota Way Target (9/9) vs §3 Reality (23/23)
- **Observation**:
  - §9 line 545: `genchi genbutsu | 9/9 Step-0 premises answered by a pasted command output before ticket #1 | S0 ledger`
  - §3 line 109: `pmat work add "PP-066 S0: discovery ledger at HEAD" --description "Falsify the eight premises the 0.66 plan depends on..."`
  - §3 table (lines 111–136): Contains **23 premises** (`S0-1` through `S0-23`).
- **Logic Chain**:
  The spec expanded from 9 premises in v1.0 to 23 premises in v1.5 as new edge cases (`S0-10`..`S0-23`) were added. The target metric in §9 and description in §3 were never updated, permitting 14 premises to remain unverified without triggering a target failure.
- **Remediation**:
  Update §9 line 545 to `23/23 Step-0 premises answered...` and §3 line 109 to `"Falsify the 23 premises the 0.66 plan depends on..."`.

---

### 5.3 Prior-Art Register Omissions: REG-10 and REG-9 in §12
- **Observation**:
  - In §5 line 227 (`REG-10`):
    `REG-10 | Never mix vendors in one graph... Lesson (§12): mixed AMD+NVIDIA in one llama.cpp process segfaults; Ollama binds to the primary driver and drops the other`
  - In §12 table (lines 649–652), the `REG` column lists:
    - llama.cpp: `1, 2, 5, 6, 13, 14` (omits 10).
    - Ollama: `1, 3, 4, 6, 7, 8, 9, 12` (omits 10).
    - llamafile: `3, 5, 6, 8, 11` (omits 10).
    - all three: `5, 7, 11` (omits 10).
  - Additionally, `REG-9` cites llama.cpp's even device-split in §5 line 226, but llama.cpp omits 9 in §12.
- **Logic Chain**:
  §5 explicitly derives `REG-10` from the cross-vendor failures of llama.cpp and Ollama documented in §12, but §12's cross-reference index omits `10`.
- **Remediation**:
  Add `10` to the REG column for both llama.cpp and Ollama, and add `9` to llama.cpp in §12.

---

### 5.4 Refusal Exit Code Ambiguity: `FeatureDisabled` (9) vs `NotImplemented` (12)
- **Observation**:
  - §0 D-7 (line 40) and §8 Refusals (line 509) state: `No refusal exit code typed into a test; the constant is read from error.rs (D-7). The tree maps CliError::FeatureDisabled → 9...`
  - In REG-9 (line 226): `--tensor-split is refused with NotImplemented (an owned refusal, not a missing flag)... FX-9: ... --tensor-split 1,1 → refusal code`.
  - In REG-14 (line 231): `--gpu-layers N is accepted only as all ... and otherwise refused with NotImplemented(partial_offload → 0.67 W-D)`.
  - Inspection of `crates/apr-cli/src/error.rs` lines 106 and 112 confirms:
    ```rust
    Self::FeatureDisabled(_) => 9,
    Self::NotImplemented(_) => 12,
    ```
- **Logic Chain**:
  1. `FeatureDisabled` (exit code 9) indicates a backend compiled out or missing from hardware (e.g. `--backend cuda` on CPU).
  2. `NotImplemented` (exit code 12) indicates a parsed CLI flag whose implementation is deferred to 0.67 (e.g. `--tensor-split`).
  3. Citing "refusal code" generically in FX-9 leads test implementers to assert exit code 9, causing tests to fail when the CLI emits exit code 12.
- **Remediation**:
  Explicitly disambiguate the two typed refusal codes in D-7, §8, and REG-9/14:
  - Unavailable backends: `CliError::FeatureDisabled.exit_code_value()` (9).
  - Deferred 0.67 options: `CliError::NotImplemented.exit_code_value()` (12).

---

### 5.5 Inaccurate Architectural Claims: `aprender-serve` WASM Gating and `aprender-gpu` Metal Dispatcher Readiness
- **Observation**:
  - §6 line 465 & §10 line 576 claim: `aprender-serve is #[cfg(not(target_arch = "wasm32"))] entirely [A] — B-S1 is a port of the loader/decoder, not a harness`
  - §6 line 464 claims: `Correction: 13 MSL kernels + MetalBackend via manzana::metal already exist in crates/aprender-gpu/src/backend/metal_shaders.rs [A]`
- **Codebase Verification**:
  1. In `crates/aprender-serve/src/lib.rs` (lines 470, 473), only memory-mapped safetensors loaders (`MappedSafeTensorsModel`, `ShardedSafeTensorsModel`) are excluded from wasm32. In fact, `loading_mmap.rs:67` provides explicit wasm32 handling, and core GGUF decoding and inference logic compile for WASM.
  2. In `crates/aprender-gpu/Cargo.toml`, there is no `metal` feature and no `manzana` dependency. `crates/aprender-gpu/src/backend/metal_shaders.rs` explicitly documents in lines 8–11:
     `These are source strings only. This crate contains no Metal dispatcher, so nothing here compiles or runs them...`
     and `MetalBackend::is_available()` in `crates/aprender-gpu/src/backend/mod.rs:67` unconditionally returns false.
- **Logic Chain**:
  Claiming `aprender-serve` is "entirely" excluded from wasm32 misleads engineers into planning a complete rewrite rather than targeting the mmap loader. Claiming `MetalBackend via manzana::metal already exists` misrepresents static shader strings as an operational driver backend.
- **Remediation**:
  Clarify in §6 and §10:
  - `B-S1`: only memory-mapped safetensors loading requires WASM adaptation; core serving is WASM-compatible.
  - `B-M1`: MSL shaders exist only as source strings; host execution requires authoring a driver dispatcher.

---

# Minor Corrections and Typos

1. **Ticket Handle Desynchronization and Scope Omission in Track T Cards**:
   - In §5 line 363, card `T-2` (`--max-seq-len honoured or refused, never clamped`) mints `pmat work add "T-5a: apr finetune --max-seq-len..."`.
   - In §5 line 379, card `T-3` (`training gate, REPORTING + self-ratchet`) mints `pmat work add "T-7 (REPORTING): train_tok_per_sec..."`.
   - Per §4 C9, receipt auditing checks `docs/audits/impl-*-receipt.md`. Minting `T-5a` and `T-7` produces `impl-t5a-receipt.md` and `impl-t7-receipt.md`, causing receipt checkers expecting `impl-t2-receipt.md` and `impl-t3-receipt.md` to fail.
   - Additionally, `T-2` is omitted from the 0.66 scope table in §2 line 96.
   - *Correction*: Update minting commands to `"T-2: ..."` and `"T-3: ..."` (noting legacy report aliases in the description), and add `T-2` to the §2 Scope table.

2. **Dual Namespace Collisions (`D-1` and `T-1`)**:
   - `D-1` designates Decision 1 in §0 (Scope split), but also designates Track D Ticket 1 in §5 (`cuda-backend-architecture.md`).
   - `T-1` designates Track T Ticket 1 in §5 (τ_loss instrument), but also designates 0.67 Training Levers in §6 (`T-1..T-5`).
   - *Correction*: Rename Documentation Ticket `D-1` to `DOC-1`, and rename 0.66 τ_loss ticket `T-1` to `T-1a` (or 0.67 levers to `TL-1..TL-5`).

3. **Missing Fixture Identifier Prefix `FX-13:` in Requirement REG-13**:
   - In §5 line 230, requirement `REG-13` begins directly with prose: `MockBackend in tests/ registers, enumerates two fake devices...`
   - *Correction*: Prefix the fixture column with `FX-13:` for strict structural uniformity with `FX-1` through `FX-14`.

4. **Stale Mutation Count in Registered Prediction G-1**:
   - In §7 line 490, the prediction reads: `both mutations RED in the guard's PR | either GREEN`.
   - However, §5 line 418 defines **three** mutations: `(i) add an untracked Cargo.toml... (ii) delete one allow-list line... (iii) add to workspace.members...`.
   - *Correction*: Update §7 line 490 to: `all three mutations RED in the guard's PR | any GREEN`.

5. **Track G Scope Truncation in Partition Metadata**:
   - The analysis partition metadata cites `Track G G-1..G-3`, omitting `G-4 · the obligation DAG as data, with invariants in CI` (lines 430–436).
   - *Correction*: Formally record that Track G comprises G-1, G-2, G-3, and G-4.

6. **Stale Line Citations in §10 Verification Ledger**:
   - `error.rs:86-90,218`: Line 218 is in a test; the actual enum match mapping `CliError::FeatureDisabled` to exit code 9 is at line **106**.
   - `root Cargo.toml:509`: Line 509 is `identity_op = "allow"`. The `aprender_ml` alias resides at line **666**.
   - `Manifest count under git` (line 576): Spec cites `106 manifests under git ls-files '*/Cargo.toml'`. At HEAD, `git ls-files '*/Cargo.toml'` returns **104 manifests** (105 including root `Cargo.toml`).
   - *Correction*: Update citations to `error.rs:106`, `Cargo.toml:666`, and `104 manifests`.

7. **Crate Qualification Omissions in §10**:
   - `handlers.rs:912-953`: Qualify crate path as `crates/apr-cli/src/commands/serve/handlers.rs` (there are multiple `handlers.rs` files, and `aprender-serve/src/cli/handlers.rs` has only 482 lines).
   - `attention.rs:995,1022`: Qualify crate path as `crates/aprender-serve/src/cuda/executor/layers/cublas_prefill/attention.rs` (there are 22 `attention.rs` files in the workspace).

8. **Clarification of `finetune.rs` Hardcoded Clamp Citations**:
   - In §5 line 364 (T-2) and §10 line 576: line 317 is a doc comment describing Defect 2; line 717 is the active executable hardcoded 512 clamp on the wgpu path (`512, // max_seq_len`).
   - *Correction*: Clarify that line 317 is documentation and line 717 is the executable clamp site in `crates/apr-cli/src/commands/finetune.rs`.

9. **Duplicate Section Heading in §11 Adjudication**:
   - Line 632 reads: `**Five findings no reviewer raised** (standard deliverable) — a sixth, found 2026-09-05...`
   - Line 634 repeats: `**Five findings no reviewer raised** (v1.1):` followed by items 1–5.
   - *Correction*: Consolidate into a single clear heading: `**Findings no reviewer raised (v1.1–v1.5)**:`.

10. **Markdown Formatting Syntax Error: Unbalanced Backtick in R-5 Contract Invariant**:
    - In §5 line 276:
      ```markdown
      (ii) `prerelease=false ⇒ ∃ 4 host receipts with `asset_sha256 == manifest[target]`;
      ```
    - *Correction*: Fix unbalanced backtick: `(ii) prerelease=false ⇒ ∃ 4 host receipts with \`asset_sha256 == manifest[target]\`;`.

11. **Ambiguous Slack Pair Notation in §10 Line 574**:
    - Line 574 cites: `Report §11 expiry slack ≥ 6 days on the DAG as dated | [C] | pairs 31→35, 14/15→18, 22→26, 20→22`.
    - Issue: `20→22` denotes only 2 days difference (or master rows 20 and 22 where row 20 expires Oct 23 and row 22 expires Oct 15), conflicting with $\ge 6$ days slack.
    - *Correction*: Disambiguate notation and reconcile with the $\ge 6$ days rule.

12. **Disambiguation of `Features::SUBGROUPS` in B-W0**:
    - In §6 line 462: `Features::SUBGROUPS requested and the grant/denial recorded in the receipt`.
    - *Correction*: Qualify as `wgpu::Features::SUBGROUPS`.

13. **Unexecutable Descriptive Prose in Acceptance Fields for R-5 and D-1**:
    - In §5 line 275 (`R-5`): Remove conversational policy prose (`...only after all four receipts are green; the promotion step is a workflow job that reads the receipts, not a hand command`) and provide a runnable command (`scripts/promote_release.sh --tag v0.66.0-rc1 --require-receipts lambda,gx10,intel,mini`).
    - In §5 line 443 (`D-1`): Replace preamble (`the existing claims-cite check ... extended to API: sentences rather than a new script...`) with the exact verification invocation (`scripts/check_doc_citations.sh docs/specifications/cuda-backend-architecture.md`).

14. **Stale Master Spec Filename in §10 Line 568**:
    - Line 568 references `PP-LLAMA-001-MASTER-v3.md`.
    - *Correction*: Update citation to the actual in-tree filename: `PP-LLAMA-001-MASTER.md`.

15. **Script Name Inconsistency in §10 Line 581**:
    - Line 581 cites `check_backend_refusal_surfaces.sh`.
    - *Correction*: Harmonize with §4 C11 and §5 R-0 by renaming to `check_backend_registry.sh`.

16. **Internal Review Reference Leak in Appendix A Line 660**:
    - In the v1.5 changelog: `MIN-03, seg-2 2.4, §12 Ollama`.
    - *Correction*: Replace raw review handle `seg-2 2.4` with its formal section name (`§6 T-lane NF4 kernel design`).

17. **Crate Name Guard Allow-List Count Ambiguity (51 vs 52 lines)**:
    - Line 412 states: `allow-list = 51 rows + the D-5 row (52)`.
    - Line 416 states: `remaining=51`.
    - Line 418 states: `remaining would read 50 while 51 offend`.
    - Line 154 (C6) states: `allow-list = 51, renames = 0`.
    - Line 547 (§9) states: `check_crate_names.sh remaining: 51 -> 51 in 0.66`.
    - *Correction*: Clarify that there are 51 offending crates in the ratchet, and 52 total lines in `scripts/crate_names_allowlist.txt` inclusive of the D-5 root front-door allow-list line.

18. **Trailing Empty Table Cells in §7 Predictions Register**:
    - In §7 lines 484–485, the `note` column contains trailing empty pipes (`| |`).
    - *Correction*: Populate with an explicit note or `-` placeholder for clean markdown table rendering.

19. **Documented Audit Record of Filtered False Positives**:
    - Candidate 4 reported an alleged typo `repnt` at line 308 (Row `P-1.2`) to be changed to `repoint`.
    - Forensic inspection of line 308 at git HEAD confirms the text already reads `repoint or author the 10 dangling #[contract] sites`.
    - *Disposition*: Formally filtered and excluded as an unverified false positive.
