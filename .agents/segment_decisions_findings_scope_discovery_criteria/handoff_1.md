# Summary

This report evaluates Segment 1 of `PP-066-release-spec.md` (v1.5, 2026-09-05), covering:
- **Preamble** (lines 1–25)
- **§0 Decisions required before ticket #1** (`D-1`–`D-11`, lines 26–46)
- **§1 Findings register — deltas from the report** (`F-1`–`F-28`, lines 47–88)
- **§2 Scope (pending D-1)** (lines 89–105)
- **§3 Step-0 — discovery** (`S0-1`–`S0-23`, lines 106–139)
- **§4 0.66 release criterion** (`C0`–`C13`, lines 140–166)

The evaluation was conducted across three mandatory dimensions:
1. **Spec Compliance:** Evaluation of `paiml-implement` units of work, mark discipline (`[V]`, `[C]`, `[A]`, `[U]`, `[X]`), ticket minting rules, strict precedence of release criterion `C0`, and scope partitioning invariants.
2. **Grammar and Clarity:** Audit of technical prose, table formatting, structural ordering, ambiguity, and clear definitions of commands versus prose assertions.
3. **Code Metrics and Status:** Direct verification against the live repository tree (`HEAD` at `a99236a86`), verifying git commits (`d6c6c6f8`, `587ad0797`, `8e1e9ad40`, `b1a6324b8`), file paths, line citations, crate manifests, script execution semantics, and live `pmat` commands.

### Executive Assessment
The segment successfully establishes a rigorous foundation for the 0.66 performance-parity cycle, converting an unwieldy 39-row monolithic document into sequenced, falsifiable tracks with clear kill criteria. The root cause analysis in `F-25` (attributing GPU fallback to build-time `cfg!` branching rather than runtime device discovery) and `F-26` (identifying provable contract enforcement theater) are architecturally sound and supported by the codebase.

However, forensic verification identified **several critical execution defects and inconsistencies**:
1. **Gate Ineffectiveness in `C12` and `S0-18`:** `scripts/pv_bin.sh` is designed solely to be *sourced* (`. scripts/pv_bin.sh`) and exports `$PV`; it ignores positional arguments and does not execute commands. Executing `scripts/pv_bin.sh lint contracts/ ...` exits `0` immediately without executing `pv lint`, creating a silent, vacuous pass in release criterion `C12`. Similarly, running `scripts/pv_bin.sh; pv lint contracts/` in `S0-18` executes in a subshell and leaves the subsequent command running the unpinned system `pv` on `PATH`.
2. **Unexecutable CLI Syntax in Discovery and Acceptance Commands:**
   - In `S0-2`, `pmat work list --status all` fails with `Error: unknown status 'all'`.
   - In `S0-1`, `grep -n '^| 22'` fails to locate row 22 in `PP-LLAMA-001-MASTER.md` because markdown bold syntax renders the line as `| **22** |`.
   - In `C1` and `C9`, commands embed mathematical and prose notation (e.g. `grep -c ... ≥ 2` and `wc -l = ticket count`) which fail when executed in a shell.
3. **Scope Table and Card Omissions:** Card `T-2` (`--max-seq-len` honoured or refused, resolving the silent 512 clamp in `finetune.rs:717`) is a required blocker for `T-0` and is detailed in §5, §6, §7, and Appendix A, but is completely missing from the 0.66 lane in the §2 Scope table.
4. **Premise Count Desynchronization:** In §3, the `pmat work add` ticket command specifies `"Falsify the eight premises the 0.66 plan depends on"`, despite the register expanding to 23 premises (`S0-1` through `S0-23`).
5. **Findings and Criterion Numbering Gaps:**
   - In §1, findings jump from `F-22` to `F-25`, leaving `F-23` and `F-24` unassigned or missing.
   - In §4, criterion `C10` is placed out of order after `C13`. Furthermore, the user dispatch references criteria `C0..C14`, but §4 contains only 14 criteria (`C0`–`C13`), with no `C14` present.
6. **Stale Tree References and Drift:**
   - In `D-7`, the reference to `CliError::FeatureDisabled -> 9` citing `error.rs:86-90,218` is stale; the mapping is located at line 106.
   - In `S0-13`, `handlers.rs:888-953` does not exist in `crates/aprender-serve` (`handlers.rs` has only 483 lines).
   - In `F-28` and `S0-23`, `CB-1700` is reported as failing at `HEAD`, whereas live `pmat comply check` reveals `CB-1700` passes (`✓`), with only `CB-1701` and `CB-2100` failing.

---

# Potential Mistakes and Improvements

## 1. Critical Execution Defects in Release Criteria and Discovery Commands

### 1.1 Silent No-Op in Release Criterion `C12` via `scripts/pv_bin.sh`
- **Location:** Line 159, Criterion `C12` check:
  ```bash
  scripts/pv_bin.sh lint contracts/ --binding contracts/aprender/binding.yaml --strict-test-binding exit 0 ...
  ```
- **Observed Code:** Inspection of `scripts/pv_bin.sh` (lines 1–663) reveals that the script is strictly designed to be sourced in bash (`. scripts/pv_bin.sh || exit 1`). Its terminal logic (lines 647–663) resolves `$PV`, asserts freshness, exports `PV`, and terminates:
  ```bash
  PV_BIN_RC=0
  PV=$(pv_bin_resolve) || PV_BIN_RC=$?
  ...
  pv_bin_assert_fresh "$PV" || return 1 2>/dev/null || exit 1
  export PV
  ```
  The script accepts no command-line arguments and contains no dispatch mechanism (e.g., `exec "$PV" "$@"`).
- **Execution Consequence:** Running `bash scripts/pv_bin.sh lint contracts/ ...` executes the script in a subshell, ignores `lint contracts/ ...`, resolves `$PV`, exports it within the transient subshell, and exits with code `0`. **The release gate passes unconditionally without validating a single contract.**
- **Remedy:** Either:
  1. Update `scripts/pv_bin.sh` to append an execution dispatcher when executed rather than sourced:
     ```bash
     if [ "${BASH_SOURCE[0]}" = "$0" ] && [ "$#" -gt 0 ]; then
       exec "$PV" "$@"
     fi
     ```
  2. Or change the normative check command in `C12` to source the resolver:
     ```bash
     . scripts/pv_bin.sh && "$PV" lint contracts/ --binding contracts/aprender/binding.yaml --strict-test-binding
     ```

### 1.2 Subshell Isolation Bug in Premise `S0-18`
- **Location:** Line 130, `S0-18` command:
  ```bash
  scripts/pv_bin.sh; pv lint contracts/ | tail -3
  ```
- **Observed Behavior:** Because `scripts/pv_bin.sh` is executed as a standalone command rather than sourced, its exported `$PV` variable is discarded when the subshell exits. The subsequent command `pv lint contracts/` resolves to whatever unpinned, stale `pv` binary resides on the user's `PATH`. This violates the explicit architectural invariant documented in `pv_bin.sh:15-17` ("pin the binary, ALWAYS").
- **Remedy:** Update `S0-18` command to:
  ```bash
  . scripts/pv_bin.sh && "$PV" lint contracts/ | tail -3
  ```

### 1.3 Unhandled Option in `pmat work list` (`S0-2`)
- **Location:** Line 114, `S0-2` command:
  ```bash
  pmat work list --status all | grep -E 'PP-(6|27|28|21|15|29|26)'
  ```
- **Observed Behavior:** Running this command produces:
  ```
  Error: unknown status 'all' (did you mean 'new'?)
  Valid values: planned, todo, open, pending, new, inprogress, in-progress, wip, active, started, working, blocked, stuck, waiting, on-hold, review, reviewing, pr, pending-review, completed, done, finished, closed, cancelled, canceled, dropped, wontfix
  ```
  `pmat work list` lists all tickets across all statuses by default when `--status` is omitted. Supplying `--status all` causes an unhandled CLI argument error and non-zero exit.
- **Remedy:** Strip `--status all` from the command:
  ```bash
  pmat work list | grep -E 'PP-(6|27|28|21|15|29|26)'
  ```

### 1.4 Regex Miss in Master Logprob Margin Discovery (`S0-1`)
- **Location:** Line 113, `S0-1` command:
  ```bash
  grep -n '^| 22' docs/specifications/PP-LLAMA-001-MASTER.md
  ```
- **Observed Behavior:** In `docs/specifications/PP-LLAMA-001-MASTER.md` at line 364, row 22 is formatted with markdown bold emphasis around the row ID:
  ```markdown
  | **22** | **instrument**: a top-2 logit margin per generated token on the wire (`logprobs` on the SSE delta, both engines)...
  ```
  Because the literal characters `**` precede `22`, the regex `^| 22` does not match, exiting with code 1. A naive runner would falsely report that row 22 is missing.
- **Remedy:** Adjust regex to tolerate optional markdown bold delimiters:
  ```bash
  grep -nE '^\|\s*(\*\*)?22(\*\*)?\s*\|' docs/specifications/PP-LLAMA-001-MASTER.md
  ```

### 1.5 Multi-File Aggregation Defect in Contract Count (`S0-18`)
- **Location:** Line 130, `S0-18` command:
  ```bash
  grep -c 'registry: true' contracts/*.yaml (expect 481)
  ```
- **Observed Behavior:** `grep -c` on a glob (`contracts/*.yaml`) prints 500+ lines in the format `contracts/<file>.yaml:<count>`. It does not aggregate into a single integer. Furthermore, evaluating this at `HEAD` across all 600+ contract files yields a total of 501 matching files, not 481 (which was the count measured at `b1a6324b8`).
- **Remedy:** Provide an aggregated command:
  ```bash
  grep -l 'registry: true' contracts/*.yaml | wc -l
  ```
  And document that the count at `HEAD` is 501 (or 481 at historical commit `b1a6324b8`).

### 1.6 Non-Executable Shell Assertions in Release Criteria (`C0`, `C1`, `C4`, `C9`)
- **Location:** Lines 148, 149, 151, 157:
  - `C0`: Check column states: `pmat comply check | grep -E 'CB-(1700|1701|2100)' prints no ✗; gh api ... protection --jq .required_status_checks.strict prints true; ...`
  - `C1`: Check column states: `scripts/perf_gate.sh --phase release receipt validates; grep -c CONFORMANT evidence/parity/LEDGER.md ≥ 2`
  - `C4`: Check column states: `scripts/check_multiplatform_dogfood.sh --require-resolved-backend cuda exit 0 on both`
  - `C9`: Check column states: `ls docs/audits/impl-*-receipt.md | wc -l = ticket count; grep -L 'partial=false' empty`
- **Analysis:** Line 144 mandates: *"each is a command that must exit 0 on the release commit."* However:
  - In `C1`, typing `grep -c CONFORMANT evidence/parity/LEDGER.md ≥ 2` fails in bash because `≥` and `2` are treated as non-existent file arguments. It must be written as:
    ```bash
    [ "$(grep -c 'CONFORMANT' evidence/parity/LEDGER.md)" -ge 2 ]
    ```
  - In `C9`, `wc -l = ticket count` is pseudocode. It should be:
    ```bash
    [ $(ls docs/audits/impl-*-receipt.md | wc -l) -eq "$TOTAL_066_TICKETS" ] && [ -z "$(grep -L 'partial=false' docs/audits/impl-*-receipt.md)" ]
    ```
  - In `C4`, the command only covers the CUDA check for `lambda`/`gx10`. For `intel`/`mini`, where the driver is absent, the expected resolved backend is `cpu`. The check command must specify the invocations for all four target architectures:
    ```bash
    scripts/check_multiplatform_dogfood.sh --host-set cuda --require-resolved-backend cuda && scripts/check_multiplatform_dogfood.sh --host-set cpu --require-resolved-backend cpu
    ```
  - In `C0`, to ensure exit code 0 reflects pass/fail:
    ```bash
    ! pmat comply check 2>&1 | grep -E 'CB-(1700|1701|2100)' | grep '✗'
    ```

---

## 2. Structural Scope and Ticket Tracking Discrepancies

### 2.1 Omission of Card `T-2` from the §2 Scope Table
- **Location:** Line 96, §2 Scope table:
  ```markdown
  | **0.66 — instrumented and honest** | ... T-0h harness, T-0 four WT receipts, T-1 τ_loss, T-3 gate REPORTING; ... |
  ```
- **Analysis:**
  - Card `T-2` (`--max-seq-len honoured or refused, never clamped`) addresses the silent-pass bug where `apr finetune` truncates context length to 512 in `finetune.rs:717`.
  - In `S0-11` (line 123), the spec states: `T-2 (new card) precedes T-0`.
  - In §5 Track T (line 372), card `T-0` explicitly lists `blockers: T-0h, T-1, T-2`.
  - In §6 (line 459), the spec states: `packing (T-2 in 0.66 already covers --max-seq-len)`.
  - In Appendix A (line 663), changelog notes: `new T-2 (--max-seq-len honoured-or-refused)`.
  - Despite being an essential 0.66 deliverable, `T-2` is **absent** from the contents column of the 0.66 lane in §2.
- **Remedy:** Update §2 line 96 to include `T-2`:
  ```markdown
  T-0h harness, T-0 four WT receipts, T-1 τ_loss, T-2 max-seq-len honour/refuse, T-3 gate REPORTING;
  ```

### 2.2 Stale Premise Count in Discovery Ticket Creation (`§3`)
- **Location:** Line 108, §3:
  ```markdown
  One ticket: `pmat work add "PP-066 S0: discovery ledger at HEAD" --description "Falsify the eight premises the 0.66 plan depends on; emit docs/audits/pp-066-s0-ledger.md with a verdict per premise"`
  ```
- **Analysis:** The description explicitly specifies `"Falsify the eight premises"`. However, the discovery register in §3 contains **23 premises** (`S0-1` through `S0-23`). The phrase "eight premises" is a relic of specification version 1.0 (which only defined `S0-1`..`S0-8`).
- **Remedy:** Update the ticket description to:
  ```markdown
  --description "Falsify the 23 premises (S0-1..S0-23) the 0.66 plan depends on; emit docs/audits/pp-066-s0-ledger.md with a verdict per premise"
  ```

### 2.3 Findings Register Numbering Gap (`F-23`, `F-24`)
- **Location:** Table in §1 (lines 76–77).
- **Analysis:** The register progresses from `F-22` (line 76) directly to `F-25` (line 77). There are no rows for `F-23` or `F-24` anywhere in the document. While `F-25` through `F-28` are bolded to signify their prominence as terminal Five-Why findings, skipping numeric IDs without an explicit note creates confusion during audits and ticket cross-referencing.
- **Remedy:** Add an explanatory note in the preamble of §1 or re-index the rows with historical aliases:
  ```markdown
  *(Note: IDs F-23 and F-24 were retired/merged into F-25 during root-cause synthesis.)*
  ```

### 2.4 Table Ordering Anomalies (`D-8` and `C10`)
- **Location:** §0 Table (line 44) and §4 Table (line 161).
- **Analysis:**
  - In §0, the decision rows appear in order: `D-1`, `D-2`, `D-3`, `D-4`, `D-5`, `D-6`, `D-7`, `D-9`, `D-10`, `D-11`, followed by `D-8` at the bottom.
  - In §4, the criteria appear as `C0`, `C1`, `C2`, `C3`, `C4`, `C5`, `C6`, `C7`, `C8`, `C9`, `C11`, `C12`, `C13`, followed by `C10` at the bottom.
- **Remedy:** Move `D-8` into its natural numerical sequence between `D-7` and `D-9`. Move `C10` between `C9` and `C11`.

### 2.5 Scope Discrepancy Regarding Criterion `C14`
- **Location:** User request / Segment scope description specifies `C0..C14`.
- **Analysis:** §4 contains only 14 criteria (`C0` through `C13`). A search across the entire document reveals zero occurrences of `C14`.
- **Remedy:** Either explicitly declare that the criteria set comprises 14 criteria indexed `C0` through `C13`, or add the missing criterion if one was intended (e.g. `C14: Andon compliance for overdue §5 cards`).

### 2.6 Decision `D-3` Escalation Scope Clarification
- **Location:** Line 36, Decision `D-3`:
  ```markdown
  | **D-3** | CI home for non-CUDA lanes: may `intel` (clean-room runner, 8 concurrent, memory-bound) host a wgpu lane and a wasm/browser harness; may `mini` (cowork-first) host nightly Metal? | **Escalate; no default.** APR-QUALITY-001 §9.1 already holds this open ⛔. B-W1, B-S*, B-M* are blocked on D-3 | ...
  ```
- **Analysis:** The table notes that `B-W1, B-S*, B-M* are blocked on D-3`. However, in the §2 Scope table, all of `B-W1..B-W5`, `B-M1..B-M4`, and `B-S1..B-S4` are placed in the **0.67 lane**. Readers may assume `D-3` is a release-blocker for 0.66.
- **Remedy:** Clarify in `D-3` that this decision blocks 0.67 execution, and that 0.66 does not provision non-CUDA CI lanes on `intel` or `mini`.

---

## 3. Repository Drift, Stale Line Numbers, and Rule Verification

### 3.1 Live Status of `CB-1700` in `pmat comply check`
- **Location:** Line 80 (`F-28`), Line 134 (`S0-23`), Line 148 (`C0`).
- **Analysis:**
  - `F-28` claims: `CB-1700: required_status_checks.strict=false (a stale branch can merge) and no required_pull_request_reviews block`.
  - `S0-23` claims: `CB-1700/1701/2100 fail at HEAD`.
  - `C0` gates on: `CB-1700, CB-1701 and CB-2100 pass in pmat comply check`.
  - **Live Verification:** Running `pmat comply check` at `HEAD` yields:
    ```
    ✓ CB-1700: Branch Protection: default branch requires ["ci / gate", "workspace-test"], >=1 approving review, and forbids force-push
    ✗ CB-1701: Supply Chain: 1 supply-chain violation(s)...
    ✗ CB-2100: Comply Gate Effect: 9 severity=error rule(s) unreachable from required check(s)...
    ```
  - **Finding:** `CB-1700` already **passes** at `HEAD`. The branch protection rule has already been updated. The open compliance failures are exclusively `CB-1701` (missing cargo deny check) and `CB-2100` (reusable workflow unreadable from repository).
- **Remedy:** Update `S0-23` and `F-28` to note that `CB-1700` was verified green at `HEAD`, leaving `CB-1701` and `CB-2100` as the active defects.

### 3.2 Stale Line Citation in `D-7` (`error.rs`)
- **Location:** Line 40, `D-7`:
  ```markdown
  the tree maps CliError::FeatureDisabled → 9 (crates/apr-cli/src/error.rs:86-90,218 [A] review MIN-07); 2 is clap's usage-error code
  ```
- **Observed Code:** In `crates/apr-cli/src/error.rs`:
  - Lines 86–90 contain contract precondition checks within `exit_code(&self)`.
  - Line 106 defines the mapping:
    ```rust
    Self::FeatureDisabled(_) => 9,
    ```
  - Line 218 is inside `find_model_file()`.
- **Finding:** The citation `86-90,218` is stale. The mapping is at line 106.
- **Remedy:** Correct the citation to `crates/apr-cli/src/error.rs:106`.

### 3.3 Drifted Citation in `S0-13` (`handlers.rs`)
- **Location:** Line 124, `S0-13`:
  ```markdown
  ... while the serve wgpu path dequantizes everything to f32 (handlers.rs:888-953) ...
  ```
- **Observed Code:** In `crates/aprender-serve/src/cli/handlers.rs`, the total line count is only 483 lines.
- **Finding:** The citation `handlers.rs:888-953` was inherited from `0_66-review.md` and no longer corresponds to the file layout in `aprender-serve`.
- **Remedy:** Cite the active dequantization path in `crates/aprender-serve/src/api/` or `crates/aprender-serve/src/quantize/` directly.

### 3.4 Missing Provable Contracts Specification Path
- **Location:** Line 6, Preamble:
  ```markdown
  PV-IMPROVE-001 (docs/specifications/improve-provable-contracts.md [U] path; ...)
  ```
- **Observed Code:** `docs/specifications/improve-provable-contracts.md` does not exist in the repository tree. While the spec properly marks this `[U]`, the file has not been committed.
- **Remedy:** Clarify whether `PV-IMPROVE-001` is tracked in an issue/epic (e.g. `#2556`) or staging crate (`crates/aprender-contracts-staging/docs/specifications/legacy/provable-contracts.md`).

### 3.5 Stale Governing Master Version in Preamble
- **Location:** Line 6, Preamble:
  ```markdown
  governing inference spec docs/specifications/PP-LLAMA-001-MASTER.md (v3.0 in hand; v3.1 [U])
  ```
- **Observed Code:** Inspection of `docs/specifications/PP-LLAMA-001-MASTER.md` demonstrates that it is committed on `main` and is explicitly titled:
  ```markdown
  # PP-LLAMA-001 v3.1 — MASTER — Inference performance parity with llama.cpp
  ```
- **Finding:** Version 3.1 is already committed, verified, and in hand. Labeling it `v3.1 [U]` contradicts the actual state of the tree.
- **Remedy:** Update line 6 to:
  ```markdown
  governing inference spec `docs/specifications/PP-LLAMA-001-MASTER.md` (v3.1 in hand `[V]`)
  ```

---

# Minor Corrections and Typos

1. **Preamble Filename Typo (Line 4):**
   - *Current:* `rewrite of 0_66-performance-parity-report.md (the "report")`
   - *Correction:* Change to `docs/specifications/0.66-performance-parity-report.md` (with period `0.66`, matching the committed file).

2. **Master Expiry Status Clarification in `D-4` (Line 37):**
   - *Current:* `The report re-dates master §12 rows 19/20/21 (10-16→09-19, 10-23→09-26, 11-06→10-10) without an Appendix D amendment | Master dates stand until an amendment names who moved them and why`
   - *Correction:* In `PP-LLAMA-001-MASTER.md` v3.1 (changelog line 432), rows 19 and 21 are defined as `status / expires: OPEN, derived` rather than fixed literal calendar dates. The table should clarify: `Master derived statuses stand (rows 19 and 21 derived from blockers; row 20 expires 2026-10-23)`.

3. **`pmat work add` Decision Ticket Formatting in §0 (Line 29):**
   - *Current:* `Each is a pmat work add "DECISION: …" ticket whose artifact is one line in this section with decided_by and date.`
   - *Improvement:* For consistency with §3 and §5, specify the exact ticket title syntax:
     ```bash
     pmat work add "DECISION: D-X <title>" --description "Record decided_by and resolution date in §0"
     ```

4. **Formatting of Mathematical Comparison in `F-11` (Line 66):**
   - *Current:* `max_batch ≥ 22 has no basis=`
   - *Improvement:* Surround identifier in backticks: `` `max_batch` >= 22 ``.

5. **Clarity of `WIP` Limits in §2 (Lines 100–103):**
   - *Current:* `Per host, ≤ 1 speed row in flight (basis: master §9 ...). gx10 is one queue, ordered by expiry: master 15 (shakedown) → T-0 → S-3's gx10 leg → master 21 (0.67).`
   - *Improvement:* Explicitly state whether `T-0` training runs count against the inference queue on `gx10`. Since `T-0` runs training on `gx10`, sequencing it after `master 15` protects both runs from VRAM starvation.

6. **Missing Markdown Escaping in Table Cells:**
   - In §3 row `S0-2`, line 114: `PP-(6\|27\|28\|21\|15\|29\|26)` uses escaped pipe symbols `\|`. In GitHub-flavored markdown tables, ensure escaping is consistent so table parsers do not treat pipes as cell separators.

7. **Table Column Header Consistency in §1:**
   - The header of §1 table is `| # | sev | finding | root cause (five whys, terminal) | fix → where |`.
   - In rows `F-25`, `F-26`, `F-27`, `F-28`, the text uses dense markdown styling. Standardizing the formatting ensures compatibility across markdown-to-HTML rendering engines.

8. **Backtick Consistency on CLI Flags:**
   - In §4 row `C11` (line 158): `0 cfg!(feature = "cuda"|"wgpu") reads in apr-cli backend decisions`.
   - Wrap `apr-cli` in backticks for typographical consistency with surrounding rows.
