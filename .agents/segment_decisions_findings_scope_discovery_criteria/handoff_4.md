# Summary

This candidate review evaluates **Segment 1** (`decisions_findings_scope_discovery_criteria`) of `docs/specifications/PP-066-release-spec.md` (v1.5, 2026-09-05). The audited segment encompasses the **Preamble** (lines 1–25), **§0 Decisions required before ticket #1** (D-1..D-11, lines 26–46), **§1 Findings register — deltas from the report** (F-1..F-28, lines 47–88), **§2 Scope** (lines 89–105), **§3 Step-0 — discovery** (S0-1..S0-23, lines 106–140), and **§4 0.66 release criterion** (C0..C14, lines 141–166).

The review was conducted across three mandatory dimensions:
1. **Spec Compliance**: Verification of `paiml-implement` lifecycle units, provable contracts and mark discipline (`[V]`, `[C]`, `[A]`, `[U]`, `[X]`), strict C0 precedence enforcement, ticket minting standards, scope partitioning invariants (0.66 instrumented/honest vs 0.67 speed/lanes), and discovery premise falsification semantics.
2. **Grammar and Clarity**: Evaluation of technical prose, table layouts and headers, typographic discipline, syntactic validity of shell pipelines, and unambiguous definition of commands, exits, and preconditions.
3. **Code Metrics and Status**: Empirical verification against repository HEAD (`587ad0797` / post-`v0.65.2` `8e1e9ad40`), auditing referenced crates (`crates/apr-cli`, `crates/aprender-serve`, `crates/aprender-compute`, `crates/aprender-train`, `crates/aprender-core`), gate scripts (`scripts/perf_gate.sh`, `scripts/pv_bin.sh`, `scripts/spec_conformance.sh`, `scripts/check_multiplatform_dogfood.sh`, etc.), cargo manifests, and live toolchain metrics via `pmat comply check` and cargo inspection.

### Overall Assessment
Segment 1 provides an exceptionally rigorous, mathematically grounded architectural framework for the 0.66 release. It successfully shifts the project from unverified speed ambition to instrumented discipline, elevating the backend registry (R-0) and provable contract integrity (F-26 / Track P) to first-class blockers. However, this forensic audit revealed **seven critical structural and executable defects**:
1. **Missing Release Criterion C14**: Although mandated by the partition specification and assignment scope (C0..C14), `C14` is entirely absent from §4 and the entire document.
2. **Missing Findings Register Entries F-23 and F-24**: The findings register jumps directly from F-22 to F-25, leaving an unacknowledged gap of two numbers.
3. **Premise Count Contradiction in §3 (8 vs 23)**: The minted `pmat work add` ticket explicitly specifies "Falsify the eight premises", while the table immediately details **23 premises** (`S0-1` through `S0-23`).
4. **`scripts/pv_bin.sh` Non-Executable Sourced Design vs CLI Execution**: In C12 and S0-18, `scripts/pv_bin.sh` is invoked as a command with subcommands (`scripts/pv_bin.sh lint contracts/...`). In git, `scripts/pv_bin.sh` is checked in as non-executable (`100644`). Furthermore, the script is designed exclusively to be *sourced* (`export PV`), meaning direct invocation ignores CLI arguments and exits 0 in a subshell without executing the linter—creating a vacuous passing gate.
5. **False-Negative Premise Failure in S0-1**: The verification command `grep -n '^| 22' docs/specifications/PP-LLAMA-001-MASTER.md` fails because row 22 in the master spec is rendered in markdown bold (`| **22** |`). This causes S0-1 to falsely evaluate as falsified.
6. **Non-Executable and Broken Commands in Release Criteria (§4 C1, C4, C8, C9, C13)**:
   - `scripts/perf_gate.sh --phase release` fails with exit code 2 because mandatory CLI arguments (`--host`, `--workload`, `--receipt`, `--commit`) are omitted.
   - `grep -c CONFORMANT evidence/parity/LEDGER.md >= 2` is invalid shell syntax and semantically vacuous (matches 7 occurrences in prose at HEAD despite 0 conformant rows).
   - `make -C machines/clean-room clean-room-p1` fails because `machines/clean-room` does not exist inside `paiml/aprender` (it resides in `../infra/machines/clean-room`).
   - `grep -L 'partial=false' empty` in C9 is pseudocode that attempts to read a literal file named `empty`.
7. **Namespace Collision Between Decision D-1 and Ticket D-1**: Decision D-1 (`0.66 scope`) shares the exact handle as Ticket D-1 (`cuda-backend-architecture.md` in §5 Track D).

---

# Potential Mistakes and Improvements

### 1. Missing Criterion C14 and Table Ordering Disorder in §4
- **Observation**:
  `ANALYSIS_PARTITION.md` specifies the segment range as `§4 0.66 release criterion (what the tag means)` covering `C0..C14`. The user prompt reiterates `C0..C14`.
  Inspection of §4 (lines 145–161) reveals exactly 14 rows, but their IDs are: `C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C11, C12, C13, C10`.
  Criterion `C14` is completely missing. Furthermore, Criterion `C10` is placed out of sequence at the bottom of the table after `C13`.
- **Logic Chain**:
  The author omitted `C14` (or miscounted the total number of criteria because C0 was prepended as row 0). Because C0 was added, there are 14 criteria in total (0 through 13), but because C10 was displaced to the end, an indexing confusion occurred.
- **Improvement**:
  1. If a 15th criterion (`C14`) was intended (e.g. for release notes or changelog gating), define `C14` explicitly.
  2. If the intended set is 14 criteria (C0 through C13), re-order the table numerically: move `C10` back to its position between `C9` and `C11`. Clarify in the preamble of §4 that the criteria set spans `C0..C13` (14 total criteria including C0).

### 2. Missing Findings F-23 and F-24 in §1
- **Observation**:
  In §1 (lines 76–77), the findings table contains row `F-22` (line 76), immediately followed by `| **F-25** | **S1** | **Root dysfunction is design, not configuration**` (line 77).
  A search across the entire repository for `F-23` and `F-24` yields zero matches.
- **Logic Chain**:
  Two findings were deleted, merged into F-25/F-26, or skipped during drafting when the four terminal findings F-25..F-28 were appended from the September 5 root-cause analysis. Leaving unacknowledged gaps in a formal findings register damages audit traceability.
- **Improvement**:
  Either document F-23 and F-24 (e.g. if they pertained to specific review tickets or superseded defects), or explicitly state in the §1 preamble: `Note: F-23 and F-24 were consolidated into F-25 during root-cause triage.`

### 3. Step-0 Premise Count Contradiction: "Eight Premises" vs S0-1..S0-23
- **Observation**:
  Line 109 states:
  `One ticket: pmat work add "PP-066 S0: discovery ledger at HEAD" --description "Falsify the eight premises the 0.66 plan depends on; emit docs/audits/pp-066-s0-ledger.md with a verdict per premise".`
  However, the table in lines 111–135 enumerates **23 premises** (`S0-1` through `S0-23`).
- **Logic Chain**:
  Early drafts of the spec had 8 discovery premises (S0-1..S0-8). In v1.2–v1.5, premises S0-9 through S0-23 were added to cover training canaries, rename mechanics, metal kernels, Q4_K shaders, clean-room costs, nightly matrices, commit baselines, pv statistics, signing keys, and comply checks. The description string for `pmat work add` was not updated.
- **Improvement**:
  Update line 109 to:
  `--description "Falsify the twenty-three premises the 0.66 plan depends on; emit docs/audits/pp-066-s0-ledger.md with a verdict per premise"`

### 4. `scripts/pv_bin.sh` Non-Executable File Mode and Sourced Interface Defect (C12, S0-18, F-26)
- **Observation**:
  - In git, `scripts/pv_bin.sh` has file mode `100644` (non-executable). Invoking `scripts/pv_bin.sh` directly returns `bash: scripts/pv_bin.sh: Permission denied`.
  - In `scripts/pv_bin.sh` lines 3–5:
    ```bash
    # Source it, never execute it:
    #     . scripts/pv_bin.sh || exit 1
    #     "$PV" lint contracts/
    ```
  - The script terminates with `export PV` (line 662). It does not execute `exec "$PV" "$@"` or handle subcommands.
  - Criterion C12 (line 159) prescribes:
    `scripts/pv_bin.sh lint contracts/ --binding contracts/aprender/binding.yaml --strict-test-binding exit 0`
  - If executed as a bash script (`bash scripts/pv_bin.sh lint contracts/...`), `pv_bin.sh` executes, ignores the arguments, exports `PV` in the subshell, and exits with code 0 without ever running `pv lint`!
- **Logic Chain**:
  This is a critical "gate is theater" failure. As written, C12 passes unconditionally because `scripts/pv_bin.sh` returns 0 regardless of contract health when executed directly!
- **Improvement**:
  Two options exist:
  1. **Update `scripts/pv_bin.sh`**: Add argument dispatch at the end of `scripts/pv_bin.sh`:
     ```bash
     if [ "$#" -gt 0 ]; then
         exec "$PV" "$@"
     fi
     ```
     And run `chmod +x scripts/pv_bin.sh` (`git update-index --chmod=+x scripts/pv_bin.sh`).
  2. **Update the Spec**: In C12, S0-18, and §5 Track P, write the invocation conforming to its existing sourced API:
     `. scripts/pv_bin.sh && "$PV" lint contracts/ --binding contracts/aprender/binding.yaml --strict-test-binding`

### 5. False Falsifier in S0-1 (`grep -n '^| 22'`)
- **Observation**:
  In S0-1 (line 113), the command to check row 22 in `PP-LLAMA-001-MASTER.md` is:
  `grep -n '^| 22' docs/specifications/PP-LLAMA-001-MASTER.md; grep -n '^| 3\.[01]' docs/specifications/PP-LLAMA-001-MASTER.md`
  Running `grep -n '^| 22'` against `docs/specifications/PP-LLAMA-001-MASTER.md` exits with code 1.
  Inspection of line 364 of `PP-LLAMA-001-MASTER.md` reveals:
  `| **22** | **instrument**: a top-2 logit margin per generated token on the wire...`
- **Logic Chain**:
  Because row 22 is bolded (`| **22** |`), the literal regex `^| 22` does not match. An engineer or automated runner executing S0-1 would conclude that row 22 is missing and trigger the "flips" clause: "I-00 (commit the master) becomes ticket #1", which is completely spurious because master is committed and row 22 is present.
- **Improvement**:
  Change the command in S0-1 to:
  `grep -nE '^\| \**22\**' docs/specifications/PP-LLAMA-001-MASTER.md`

### 6. Invalid CLI Arguments in S0-2 (`pmat work list --status all`)
- **Observation**:
  S0-2 (line 114) specifies:
  `pmat work list --status all | grep -E 'PP-(6|27|28|21|15|29|26)'`
  Running this command produces:
  `Error: unknown status 'all' (did you mean 'new'?)`
  `Valid values: planned, todo, open, pending, new, inprogress, in-progress, wip, active, started, working, blocked, stuck, waiting, on-hold, review, reviewing, pr, pending-review, completed, done, finished, closed, cancelled, canceled, dropped, wontfix`
- **Logic Chain**:
  `pmat work list` does not recognize `all` as a valid status filter. In fact, running `pmat work list` without `--status` lists all tickets by default.
- **Improvement**:
  Change the command to:
  `pmat work list | grep -E 'PP-(6|27|28|21|15|29|26)'`

### 7. Multi-Binary Ambiguity in S0-4 (`cargo run -p apr-cli`)
- **Observation**:
  S0-4 (line 116) specifies:
  `cargo run -p apr-cli --features wgpu -- serve run --backend wgpu --list-adapters`
  Running this command fails immediately with:
  `error: cargo run could not determine which binary to run. Use the --bin option to specify a binary, or the default-run manifest key.`
  `available binaries: apr, apr-corpus-ingest`
- **Logic Chain**:
  `crates/apr-cli/Cargo.toml` defines two binaries (`[[bin]] name = "apr"` and `[[bin]] name = "apr-corpus-ingest"`). Because neither workspace nor crate `Cargo.toml` sets `default-run = "apr"`, `cargo run -p apr-cli` cannot determine the target binary.
- **Improvement**:
  Change the command in S0-4 to include `--bin apr`:
  `cargo run -p apr-cli --bin apr --features wgpu -- serve run --backend wgpu --list-adapters`
  (Alternatively, add `default-run = "apr"` to `crates/apr-cli/Cargo.toml`).

### 8. Release Criterion C1 Has Invalid Invocation Syntax and Vacuous Assertions
- **Observation**:
  C1 (line 149) specifies:
  `scripts/perf_gate.sh --phase release receipt validates; grep -c CONFORMANT evidence/parity/LEDGER.md >= 2`
  - Running `scripts/perf_gate.sh --phase release` fails with exit code 2:
    `perf_gate: usage: perf_gate.sh --host H --phase {merge|release} --workload {W1|W2} --receipt PATH [--commit SHA]`
  - `grep -c CONFORMANT evidence/parity/LEDGER.md >= 2` is invalid bash syntax.
  - Furthermore, running `grep -c CONFORMANT evidence/parity/LEDGER.md` at HEAD returns **7**, even though `evidence/parity/LEDGER.md` contains **0** conformant rows! It matches prose occurrences such as `None is CONFORMANT` and `c=1: NONCONFORMANT-VALID`.
- **Logic Chain**:
  The criterion check is both syntactically unexecutable as a single command and semantically vacuous if executed as written.
- **Improvement**:
  1. Specify the full arguments for `perf_gate.sh`:
     `scripts/perf_gate.sh --host lambda --phase release --workload W1 --receipt <receipt-path> --commit $(git rev-parse HEAD)`
  2. Tighten the ledger check to count table rows where the tier column is specifically `CONFORMANT`:
     `test $(grep -E '^\|[[:space:]]*[0-9]+[[:space:]]*\|.*\|[[:space:]]*CONFORMANT[[:space:]]*\|' evidence/parity/LEDGER.md | wc -l) -ge 2`

### 9. Non-Existent Path in C8 (`machines/clean-room`)
- **Observation**:
  Criterion C8 (line 155) prescribes:
  `make -C machines/clean-room clean-room-p1 exit 0`
  Executing this command in the repository root fails:
  `make: *** machines/clean-room: No such file or directory. Stop.`
  `machines/` does not exist in `paiml/aprender`.
- **Logic Chain**:
  As documented in `crates/aprender-orchestrate/CLAUDE.md:211`, the clean-room harness is located in the sibling repository `../infra/machines/clean-room/`. Specifying a repository-relative path `machines/clean-room` causes immediate build failure.
- **Improvement**:
  Wrap the invocation in a repo-root script (e.g. `scripts/run_clean_room.sh`) that verifies `../infra/machines/clean-room` exists or clones it, or explicitly specify:
  `make -C ../infra/machines/clean-room clean-room-p1` with a precondition noting sibling checkout.

### 10. Pseudocode in C9 (`grep -L 'partial=false' empty`)
- **Observation**:
  Criterion C9 (line 157) states:
  `ls docs/audits/impl-*-receipt.md | wc -l = ticket count; grep -L 'partial=false' empty`
- **Logic Chain**:
  `grep -L 'partial=false' empty` looks for a file literally called `empty`. If no file named `empty` exists, grep exits with an error code. If an empty file named `empty` is created, `grep -L` outputs `empty`, which is non-empty.
- **Improvement**:
  Rewrite as an executable shell assertion:
  `test $(ls docs/audits/impl-*-receipt.md 2>/dev/null | wc -l) -eq <TICKET_COUNT> && test -z "$(grep -L 'partial=false' docs/audits/impl-*-receipt.md 2>/dev/null)"`

### 11. Literal Ellipsis in C13 Install URL
- **Observation**:
  Criterion C13 (line 160) specifies:
  `curl -LsSf …/install.sh | bash on each host exits 0 and its last lines are apr devices output`
- **Logic Chain**:
  The command contains a literal Unicode ellipsis character `…` instead of an actual or parameterized URL.
- **Improvement**:
  Replace `…` with the exact intended URL format:
  `curl -LsSf https://github.com/paiml/aprender/releases/download/v0.66.0/install.sh | bash`

### 12. Namespace Collision: Decision D-1 vs Ticket D-1
- **Observation**:
  - In §0 (line 34), Decision 1 is labeled `**D-1**` (`0.66 scope: all five tracks or the split in §2`).
  - In §5 Track D (line 439), Ticket 1 is labeled `**D-1 · `cuda-backend-architecture.md`**`.
  - In §4 line 163, the text says: "A RED on the non-blocking track (D-1 CUDA doc, G-3 count)...".
- **Logic Chain**:
  Having both a Decision D-1 and a Ticket D-1 introduces ambiguity in audits, tickets, and automated status checking (e.g., when grepping for ticket statuses or decisions in receipts).
- **Improvement**:
  Prefix decisions with `DEC-` (e.g., `DEC-1` through `DEC-11`) or rename Track D tickets to `DOC-1`.

### 13. Ordering of Decision D-8 in §0
- **Observation**:
  In §0 (lines 33–44), the decision table lists:
  `D-1`, `D-2`, `D-3`, `D-4`, `D-5`, `D-6`, `D-7`, `D-9`, `D-10`, `D-11`, `D-8`.
- **Logic Chain**:
  `D-8` was appended to the bottom of the table after `D-11` instead of placed between `D-7` and `D-9`.
- **Improvement**:
  Sort the table rows numerically so `D-8` sits between `D-7` and `D-9`.

---

# Minor Corrections and Typos

### 1. Citation Line Drift in D-7 (`crates/apr-cli/src/error.rs`)
- **Location**: §0, row D-7 (line 40).
- **Text**: `the tree maps CliError::FeatureDisabled → 9 (crates/apr-cli/src/error.rs:86-90,218 [A] review MIN-07)`
- **Issue**: In `crates/apr-cli/src/error.rs`, lines 86–90 are precondition assertion macros in `exit_code()`. The mapping `Self::FeatureDisabled(_) => 9` is located at **line 106**. Line 218 is inside `find_model_file()`.
- **Correction**: Update citation to `crates/apr-cli/src/error.rs:106`.

### 2. Stale Contract Exemption Count in S0-18 (`registry: true`)
- **Location**: §3, row S0-18 (line 130).
- **Text**: `grep -c 'registry: true' contracts/*.yaml (expect 481)`
- **Issue**: Executing `grep -l 'registry: true' contracts/*.yaml | wc -l` at repository HEAD yields **501** matching files. The count has drifted upwards from 481 to 501.
- **Correction**: Update the expected count to 501 (or note `481 at b1a6324b8; 501 at HEAD`).

### 3. Factual Inaccuracy Regarding CB-1700 in S0-23
- **Location**: §3, row S0-23 (line 135).
- **Text**: `the required check is strict and readable (F-28): CB-1700/1701/2100 fail at HEAD; required_status_checks.strict is false`
- **Issue**: Running `pmat comply check` at repository HEAD reveals that `CB-1700` **passes**:
  `✓ CB-1700: Branch Protection: default branch requires ["ci / gate", "workspace-test"], >=1 approving review, and forbids force-push`.
  Only `CB-1701` and `CB-2100` fail.
- **Correction**: Update the text to state: `CB-1701 and CB-2100 fail at HEAD; CB-1700 passes branch protection rules`.

### 4. Preamble Governing Spec Status Stale Marking (`v3.1 [U]`)
- **Location**: Preamble (line 5–6).
- **Text**: `governing inference spec docs/specifications/PP-LLAMA-001-MASTER.md (v3.0 in hand; v3.1 [U])`
- **Issue**: `PP-LLAMA-001-MASTER.md` at HEAD is already v3.1 (`| 3.1 | 2026-09-03 |`). It is committed on `main` in this repository and is fully available.
- **Correction**: Change to `(v3.1 in hand)`.

### 5. Preamble Local Mark Notation Collision (`[X]`)
- **Location**: Preamble (line 5 vs line 21).
- **Text**: Line 5 states `third-party comparative study of GPU discovery in llama.cpp / llamafile / Ollama ([X], §12)`. Line 21–22 states `The report's private [X] ("from a receipt as written") is [A] here; [X] keeps its fleet meaning (third-party, excluded from published claims)`.
- **Issue**: Using `[X]` as an inline citation right next to Section 12 is confusing given the redefinition of report marks vs fleet marks.
- **Correction**: Clarify as `(third-party benchmark study [X], §12)`.

### 6. Unimplemented Flag `--require-resolved-backend` in C4
- **Location**: §4, row C4 (line 152).
- **Text**: `scripts/check_multiplatform_dogfood.sh --require-resolved-backend cuda exit 0 on both`
- **Issue**: `scripts/check_multiplatform_dogfood.sh` does not support any `--require-resolved-backend` flag.
- **Correction**: Note that `--require-resolved-backend` is an extension to `scripts/check_multiplatform_dogfood.sh` delivered under Ticket R-6 / C4.

### 7. Missing Script Deliverables in §4 Commands
- **Location**: §4, rows C2, C5, C6, C10, C11.
- **Text**: Cites `scripts/check_backend_firstclass.sh`, `scripts/train_parity.sh`, `scripts/check_crate_names.sh`, `scripts/check_dag_invariants.sh`, `scripts/check_backend_registry.sh`.
- **Issue**: None of these scripts exist in `scripts/` at HEAD.
- **Correction**: Add a table footnote or column annotation indicating that these scripts are created by tickets B-G1, T-1, G-1, G-4, and R-0 respectively.

### 8. Typographical Inconsistency in Finding IDs (Bold vs Plain)
- **Location**: §1 (lines 55–80).
- **Text**: `F-1` through `F-22` use regular markdown font in the ID column (`F-1`, `F-2`, etc.), whereas `**F-25**`, `**F-26**`, `**F-27**`, and `**F-28**` are bolded.
- **Correction**: Standardize the ID column to bold (`**F-1**` .. `**F-28**`) or plain text throughout.
