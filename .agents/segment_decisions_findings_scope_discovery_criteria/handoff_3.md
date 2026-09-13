# Summary

This report presents a rigorous, independent review of Segment 1 of `PP-066-release-spec.md` (v1.5, dated 2026-09-05) conducted by Analyst 3 (Candidate 3). Segment 1 establishes the foundational strategy, scope boundaries, discovery premises, and release acceptance criteria for the `0.66` release across six core sections:
- **Preamble** (lines 1–25)
- **§0 Decisions required before ticket #1** (`D-1`–`D-11`, lines 26–45)
- **§1 Findings register — deltas from the report** (`F-1`–`F-28`, lines 46–87)
- **§2 Scope (pending D-1)** (lines 88–105)
- **§3 Step-0 — discovery** (`S0-1`–`S0-23`, lines 106–139)
- **§4 0.66 release criterion** (`C0`–`C14`, lines 140–165)

The evaluation was conducted across three mandatory dimensions:
1. **Spec Compliance**: Verification of `paiml-implement` unit discipline, mark notation (`[V]`, `[C]`, `[A]`, `[U]`, `[X]`), release criterion `C0` absolute precedence, ticket minting conventions, and scope-split/heijunka invariants.
2. **Grammar and Clarity**: Linguistic precision, technical prose clarity, markdown table formatting/ordering, unambiguous definitions of executable commands, and absence of semantic contradiction.
3. **Code Metrics and Status**: Empirical verification against the active repository tree (`paiml/aprender` at HEAD, branch worktree `pp-066-spec`), inspecting workspace manifests, crates (`crates/apr-cli`, `crates/aprender-serve`, `crates/aprender-compute`, `crates/aprender-train`, `crates/aprender-gpu`, `crates/aprender-registry`), test fixtures, GitHub Actions workflow definitions, scripts (`scripts/perf_gate.sh`, `scripts/pv_bin.sh`, `scripts/check_contract_test_binding.sh`, etc.), and live PMAT checks (`pmat comply check`).

### Overall Assessment
Segment 1 exhibits extraordinary architectural depth, intellectual honesty, and uncompromising engineering discipline. It correctly identifies and confronts the primary structural failures of the project:
- **F-25**: Exposes that GPU execution failure is a fundamental product design defect (build-time `cfg!` branching rather than runtime dynamic hardware discovery via a `BackendRegistry`).
- **F-26**: Demonstrates that provable contracts at HEAD are largely compliance theater (`pv lint` passes with threshold `0.00`, 481 of 698 root contracts self-exempt via `registry: true`, and `#[contract]` macros emit empty arms).
- **F-27**: Identifies the release artifact crisis (#2869: no pre-built `apr` binaries attached to tagged GitHub releases, forcing ad-hoc client compilation).
- **F-28 / C0**: Discovers that the primary branch gate `ci / gate` is unreadable, non-strict, and susceptible to passing on empty arms (#2830), establishing `C0` as a non-negotiable prerequisite before any other criterion can be credited.

Despite these exceptional strengths, the review identified several critical discrepancies, structural omissions, syntax errors in gate commands, and codebase state drifts that must be resolved prior to cutting the specification to final status.

### Summary of Key Findings
1. **Missing Release Criterion C14**: The dispatch mandate, context instructions, and roadmap designate criteria `C0` through `C14`. However, `PP-066-release-spec.md` contains no criterion `C14` anywhere in the text.
2. **Step-0 Premise Count Desynchronization**: The ticket minting command in §3 specifies `--description "Falsify the eight premises the 0.66 plan depends on..."`, but the table defines **23** distinct premises (`S0-1` through `S0-23`).
3. **Numbering Gaps and Table Dislocations**:
   - In §1, findings `F-23` and `F-24` are completely missing from the register, jumping directly from `F-22` to `F-25`.
   - In §0, Decision `D-8` (0.67 crate rename sequencing) is placed at the end of the table after `D-11`.
   - In §4, Criterion `C10` (Obligation DAG invariants) is displaced to the bottom of the table after `C13`.
4. **Unexecutable Shell Syntax in Release Gates**:
   - In `C1`, the check command uses the Unicode mathematical glyph `≥` (`grep -c CONFORMANT evidence/parity/LEDGER.md ≥ 2`), which produces a fatal shell syntax error when executed in bash.
   - In `C9`, the check string `ls docs/audits/impl-*-receipt.md | wc -l = ticket count; grep -L 'partial=false' empty` contains unparseable shell pseudo-code.
   - In `C7`, the verification command omits the script path entirely, stating only `script exit 0`.
5. **Defective Grep Patterns in Discovery Commands**:
   - In `S0-1`, `grep -n '^| 22'` fails to match the actual row in `PP-LLAMA-001-MASTER.md` because the row header is formatted with markdown bolding (`| **22** |`).
   - In `S0-14` and `S0-19`, grep invocations rely on pipe alternation (`|`) without specifying `-E` / `--extended-regexp`, causing standard POSIX grep to search for literal pipe characters.
6. **Codebase State Drift**:
   - `S0-12` assumes `MetalBackend` exists via `manzana::metal` based on review `MIN-05`. Empirical audit of PR 2849 reveals that `manzana::metal` exports were explicitly removed from `aprender-gpu`.
   - In `F-28` / `S0-23`, `arm_a_scaling` is referenced in connection with defect #2830; codebase inspection reveals that `scripts/perf_gate.sh` at line 620 has already renamed this function to `arm_a_self_regression`.
   - `PP-LLAMA-001-MASTER.md` is currently committed on `main` at version 3.1; the preamble note `(v3.0 in hand; v3.1 [U])` is stale.
7. **Ticket Handle Collisions**:
   - `D-1` designates Decision 1 in §0 ("0.66 scope"), but is also used in Track D of §5 as the ticket handle for CUDA documentation ("D-1 CUDA doc").
   - `T-1` designates the `τ_loss` seed variance instrument in 0.66, while `T-1` in 0.67 designates the training speed lever.

---

# Potential Mistakes and Improvements

### 1. Spec Compliance and Process Invariants

#### 1.1 Complete Omission of Release Criterion C14
- **Observation**: The dispatch assignment, `ANALYSIS_PARTITION.md` (line 18), and user prompt explicitly define the criteria range as `C0..C14`. However, `PP-066-release-spec.md` §4 (lines 145–161) contains only 14 criteria: `C0`, `C1`, `C2`, `C3`, `C4`, `C5`, `C6`, `C7`, `C8`, `C9`, `C11`, `C12`, `C13`, and `C10`. Searching for `C14` across the entire document yields zero results.
- **Logic Chain**: The orchestrator partitioned §4 assuming 15 criteria (`C0` through `C14`). If an intended criterion (such as documentation signoff, clean-room container attestation, or release tag verification) was dropped or re-numbered, the criteria register is incomplete. Alternatively, if 14 criteria were intended (since $0$ to $13$ is 14 items), the labeling `C0..C14` is an off-by-one fencepost error.
- **Improvement**: Determine whether a 15th criterion was drafted and omitted (e.g., C14 covering final changelog/documentation signoff or binary checksum publication), or clarify that the active set terminates at `C13`. Explicitly add `C14` or amend all governing documentation to specify `C0..C13`.

#### 1.2 Step-0 Discovery Ticket Description Desynchronization (S0-1..S0-23 vs "Eight Premises")
- **Observation**: Line 109 of `PP-066-release-spec.md` specifies the discovery ticket creation command:
  ```bash
  pmat work add "PP-066 S0: discovery ledger at HEAD" --description "Falsify the eight premises the 0.66 plan depends on; emit docs/audits/pp-066-s0-ledger.md with a verdict per premise"
  ```
  The table immediately following (lines 111–135) lists **23** numbered premises: `S0-1` through `S0-23`.
- **Logic Chain**: An earlier revision of the specification (v1.0) defined eight initial discovery checks (`S0-1` through `S0-8`). As revisions v1.1 through v1.5 assimilated GPU discovery requirements (`S0-12`–`S0-16`, `S0-20`–`S0-22`), PV-IMPROVE findings (`S0-17`–`S0-18`), and branch protection rules (`S0-23`), the premises table grew to 23 items. The ticket description was never updated to reflect the expanded scope. An engineer executing `pmat work add` verbatim creates a ticket with contradictory metadata.
- **Improvement**: Update the command to match the table:
  ```bash
  pmat work add "PP-066 S0: discovery ledger at HEAD" --description "Falsify the 23 premises the 0.66 plan depends on (S0-1..S0-23); emit docs/audits/pp-066-s0-ledger.md with a verdict per premise"
  ```

#### 1.3 Missing Findings F-23 and F-24 in the Register
- **Observation**: §1 (lines 53–80) contains rows `F-1` through `F-22`, followed immediately by `**F-25**`, `**F-26**`, `**F-27**`, and `**F-28**`. Findings `F-23` and `F-24` are absent from the register and do not appear anywhere in the specification.
- **Logic Chain**: In a formal audit ledger, an unexplained numerical discontinuity violates traceability. A downstream auditor or QA verifier cannot discern whether two findings were retracted, merged into `F-25`, or inadvertently deleted during drafting.
- **Improvement**: Add an explicit note in the register header or within the table stating:
  `| F-23..F-24 | — | Retired / merged into F-25 during five-whys synthesis | — | — |`
  This preserves register integrity and prevents forensic ambiguity.

#### 1.4 Ticket Handle Collisions Across Tracks
- **Observation**:
  1. The identifier `D-1` is assigned to Decision 1 in §0 ("0.66 scope: all five tracks or the split in §2", line 34). In §2 (line 96) and §5 (Track D), `D-1` is also assigned to the documentation ticket ("D-1 CUDA doc (non-blocking)").
  2. The identifier `T-1` is assigned in 0.66 to the statistical derivation of `τ_loss` from Unsloth seed variance (§1 F-8, §2 line 96, §5 Track T). In 0.67 (§2 line 97), `T-1..T-5` designates the training speed levers.
- **Logic Chain**: When `pmat work add` or PMAT queries are run, ticket handles serve as primary keys in progress tracking, git commit tags, and PR review comments. Reusing `D-1` for both a strategic scope decision and a CUDA documentation card risks ticket misattribution. Reusing `T-1` across 0.66 and 0.67 confuses instrument cards with performance optimization cards.
- **Improvement**:
  - Rename the documentation ticket to `DOC-1` or `D-CUDA-1`. Reserve `D-*` strictly for §0 Decisions (`D-1`..`D-11`).
  - Distinguish the training instrument card as `T-0b` or `T-TAU-1`, keeping `T-1` reserved for the 0.67 speed lever.

#### 1.5 C0 Precedence Enforcement and Status-Check API Nuances
- **Observation**: Section 4 establishes that `C0` must be credited first:
  > "C0 is credited first: until it holds, every other criterion is recorded [U] whatever its own command prints, because the check that would credit it cannot be shown to run."
  In `S0-23` and `C0`, the pre-authorized branch protection fix is given as:
  `gh api -X PATCH repos/paiml/aprender/branches/main/protection/required_status_checks -F strict=true`
- **Logic Chain**: The GitHub REST API v3 endpoint `PATCH /repos/{owner}/{repo}/branches/{branch}/protection/required_status_checks` requires a JSON payload with schema `{"strict": boolean, "contexts": string[]}`. Calling `gh api` with `-F strict=true` serializes the parameter as multipart form-data or url-encoded form values. Depending on the `gh` CLI version and API parsing, boolean fields in JSON-only endpoints can be rejected with HTTP 422 (`Invalid request: strict must be a boolean`). Furthermore, modifying `required_status_checks` without supplying the existing `contexts` array can clear the required check contexts list on certain API versions.
- **Improvement**: Ensure the pre-authorized update script uses strict JSON payload formatting:
  ```bash
  gh api -X PATCH repos/paiml/aprender/branches/main/protection/required_status_checks \
    --input - <<< '{"strict": true}'
  ```
  Validate that the existing contexts (`ci / gate`) are explicitly preserved during the patch.

---

### 2. Technical Precision and Shell Invocations

#### 2.1 Unexecutable Shell Syntax in Criterion C1
- **Observation**: In §4 (line 149), the verification command for `C1` reads:
  ```bash
  scripts/perf_gate.sh --phase release receipt validates; grep -c CONFORMANT evidence/parity/LEDGER.md ≥ 2
  ```
- **Logic Chain**: The symbol `≥` (U+2265) is a Unicode character, not a valid POSIX shell operator. When executed in bash, the shell attempts to execute `grep` with arguments `-c`, `CONFORMANT`, `evidence/parity/LEDGER.md`, `≥`, and `2`. `grep` treats `≥` and `2` as additional file paths to search, causing:
  `grep: ≥: No such file or directory`
  `grep: 2: No such file or directory`
  and exiting with error status 2. Furthermore, `receipt validates` is an unquoted human-language phrase, not an executable bash argument.
- **Improvement**: Replace with valid, testable bash commands:
  ```bash
  scripts/perf_gate.sh --phase release && test "$(grep -c 'CONFORMANT' evidence/parity/LEDGER.md)" -ge 2
  ```

#### 2.2 Pseudo-Code in Criterion C9 Check Command
- **Observation**: In §4 (line 157), the verification command for `C9` reads:
  ```bash
  ls docs/audits/impl-*-receipt.md | wc -l = ticket count; grep -L 'partial=false' empty
  ```
- **Logic Chain**: This string cannot be executed by an automated verifier or CI script:
  - `wc -l = ticket count` produces a bash syntax error (`bash: =: command not found`).
  - `grep -L 'partial=false' empty` searches for files containing `partial=false` among a non-existent file named `empty`.
- **Improvement**: Rewrite as a fully executable compound bash expression:
  ```bash
  test "$(ls docs/audits/impl-*-receipt.md 2>/dev/null | wc -l)" -eq "$EXPECTED_TICKET_COUNT" && \
  test -z "$(grep -L 'partial=false' docs/audits/impl-*-receipt.md 2>/dev/null)"
  ```

#### 2.3 Incomplete Executable Path in Criterion C7
- **Observation**: In §4 (line 155), the check column for `C7` states:
  ```bash
  script exit 0
  ```
  The criteria column names `check_no_claim_literals.sh`, but the check column fails to specify the script path.
- **Improvement**: Explicitly write the executable invocation:
  ```bash
  scripts/check_no_claim_literals.sh
  ```

#### 2.4 Defective Grep Regexp in Step-0 Premise S0-1
- **Observation**: In §3 (line 112), `S0-1` instructs:
  ```bash
  git ls-files docs/specifications/PP-LLAMA-001-MASTER.md; grep -n '^| 22' docs/specifications/PP-LLAMA-001-MASTER.md; grep -n '^| 3\.[01]' docs/specifications/PP-LLAMA-001-MASTER.md
  ```
  Running `grep -n '^| 22' docs/specifications/PP-LLAMA-001-MASTER.md` returns zero lines.
- **Logic Chain**: In `docs/specifications/PP-LLAMA-001-MASTER.md` line 364, the obligation table row is formatted with bold markdown emphasis:
  ```markdown
  | **22** | **instrument**: a top-2 logit margin per generated token on the wire...
  ```
  Because the number `22` is preceded by space and two asterisks (`| **22**`), the regular expression `^| 22` does not match. An automated script executing this command will conclude that row 22 is missing and declare `S0-1` FALSIFIED, even though row 22 is present in the document.
- **Improvement**: Update the command to handle markdown formatting:
  ```bash
  grep -nE '^\|\s*\*{0,2}22\*{0,2}\s*\|' docs/specifications/PP-LLAMA-001-MASTER.md
  ```

#### 2.5 Missing Extended Regexp Flag in S0-14 and S0-19
- **Observation**:
  - In `S0-14` (line 125): `ldd target/release/apr | grep -ci 'cuda|cublas'`
  - In `S0-19` (line 130): `grep -rn 'minisign|ssh-keygen -Y|signify' scripts/ .github/workflows/`
- **Logic Chain**: Standard `grep` interprets `|` as a literal character, matching only if the string contains an actual vertical bar. To treat `|` as an alternation operator, `grep -E` (or `egrep`) is required. Without `-E`, both commands silently fail to match their targets.
- **Improvement**: Add the `-E` flag to both commands:
  - In `S0-14`: `ldd target/release/apr | grep -Eci 'cuda|cublas'`
  - In `S0-19`: `grep -rnE 'minisign|ssh-keygen -Y|signify' scripts/ .github/workflows/`

---

### 3. Code Metrics and Status Realities

#### 3.1 Crate State Drift: MetalBackend Removal in PR 2849 vs S0-12
- **Observation**: In §3 (line 123), `S0-12` states:
  > "`crates/aprender-gpu/src/backend/metal_shaders.rs` holds 13 MSL kernels and `MetalBackend` via `manzana::metal` (`[A]` review MIN-05; the report says 'no Metal-specific kernel or backend module') — does it build and enumerate a device on `mini`?"
  Empirical inspection of the repository history reveals that PR 2849 (`evidence/pr-review/2849/9aa58d3671fe084cb9e92f219b8e2590b810b89a/receipt.intoto.jsonl` and `findings.sarif`) explicitly removed the `metal` Cargo feature and the `pub use manzana::metal::*` exports from `crates/aprender-gpu/Cargo.toml` and `src/backend/mod.rs`.
- **Logic Chain**: While `metal_shaders.rs` still contains the 13 raw MSL kernel string constants, there is currently **no** `MetalBackend` struct and **no** dependency on `manzana` in `aprender-gpu`. Running `cargo check -p aprender-gpu --features metal` will fail because the feature does not exist.
- **Improvement**: Update `S0-12` to note that `MetalBackend` was removed in PR 2849 and that `metal_shaders.rs` contains source strings only (as documented in `crates/aprender-gpu/src/backend/metal_shaders.rs:8-11`). Shape `B-M1` as a restoration and wiring ticket rather than a mere device enumeration test.

#### 3.2 Function Renaming in scripts/perf_gate.sh: arm_a_scaling vs arm_a_self_regression
- **Observation**: In §1 `F-28`, §3 `S0-23`, and §4 `C0`, the specification references issue #2830:
  > "#2830: `perf_gate.sh` returns `VERDICT PASS` on a c=1-only receipt whose `arm_a_scaling` emitted nothing — a gate that passes on silence"
  Direct inspection of `scripts/perf_gate.sh` at lines 620 and 1019 reveals that `arm_a_scaling` has been renamed to `arm_a_self_regression`:
  ```bash
  620: arm_a_self_regression() {
  ...
  1019: run_phased A "$phase" arm_a_self_regression "$receipt" "$host" "$workload" || rc=1
  ```
- **Logic Chain**: As documented in `PP-LLAMA-001-MASTER.md` row 0a (line 338), `arm_a_self_regression` replaced `arm_a_scaling`. The polarity bug identified in #2830 applies to the logic inside `arm_a_self_regression` (which skips evaluation when only `c=1` is present).
- **Improvement**: Clarify in `F-28` and `C0` that the affected function in `scripts/perf_gate.sh` at current HEAD is `arm_a_self_regression` (formerly `arm_a_scaling`), ensuring tests targeting the selftest flag verify the correct symbol.

#### 3.3 Status of Non-Existent Verification Scripts
- **Observation**: Multiple criteria in §4 mandate execution of scripts that do not exist at HEAD:
  - `C2`: `scripts/check_backend_firstclass.sh` (No such file or directory)
  - `C5`: `scripts/train_parity.sh` (No such file or directory)
  - `C6`: `scripts/check_crate_names.sh` (No such file or directory)
  - `C10`: `scripts/check_dag_invariants.sh` (No such file or directory)
  - `C11`: `scripts/check_backend_registry.sh` (No such file or directory)
- **Logic Chain**: These scripts are deliverables of tickets defined in §5 (`B-G1`, `T-0h`/`T-1`, `G-1`, `G-4`, and `R-0` respectively). At draft stage (v1.5), referencing these scripts is forward-looking. However, §4 does not explicitly annotate which ticket delivers which script. An automated gate runner executing §4 checks today encounters immediate file-not-found errors.
- **Improvement**: In §4, annotate each check command with its producing ticket:
  - `scripts/check_backend_firstclass.sh` (delivered by `B-G1`)
  - `scripts/train_parity.sh` (delivered by `T-0h` / `T-1`)
  - `scripts/check_crate_names.sh` (delivered by `G-1`)
  - `scripts/check_dag_invariants.sh` (delivered by `G-4`)
  - `scripts/check_backend_registry.sh` (delivered by `R-0`)

#### 3.4 Governing Inference Spec Status at HEAD
- **Observation**: The Preamble (lines 5–6) states:
  > "governing inference spec `docs/specifications/PP-LLAMA-001-MASTER.md` (v3.0 in hand; v3.1 `[U]`)"
  In §1 `F-9` and §3 `S0-1`, the spec treats v3.1 and row 22 as uncertain premises.
- **Logic Chain**: Direct inspection of `docs/specifications/PP-LLAMA-001-MASTER.md` confirmed:
  1. The file is committed and tracked on `main`.
  2. The header states: `# PP-LLAMA-001 v3.1 — MASTER — Inference performance parity with llama.cpp`.
  3. §12 contains row 22 at line 364.
- **Improvement**: In the Preamble, update the status note to:
  `governing inference spec docs/specifications/PP-LLAMA-001-MASTER.md (v3.1 committed at HEAD [V])`.
  Mark premise `S0-1` as `CONFIRMED [V]` in the discovery ledger.

#### 3.5 Disambiguation of Crate `aprender-registry` vs Serve `BackendRegistry`
- **Observation**: In §1 `F-25`, §2, and §4 `C11`, the specification establishes the `BackendRegistry` (ticket `R-0`) as the architectural foundation of runtime discovery. In the repository root, there is an existing workspace crate located at `crates/aprender-registry`.
- **Logic Chain**: Examination of `crates/aprender-registry/Cargo.toml` reveals:
  ```toml
  name = "aprender-registry"
  [lib]
  name = "pacha"
  description = "Model, Data and Recipe Registry with full lineage tracking"
  ```
  `aprender-registry` is the `pacha` data/model lineage registry. In contrast, `R-0`'s `BackendRegistry` is a hardware device discovery module designed for `crates/aprender-serve/src/gpu/backend.rs`. A contributor or LLM agent reading "registry" may mistakenly assume `crates/aprender-registry` is where hardware backend enumeration should be implemented.
- **Improvement**: Explicitly add a disambiguation note in §1 `F-25` and §2:
  *(Note: BackendRegistry for hardware discovery lives in `crates/aprender-serve/src/gpu/backend.rs`, not to be confused with the existing `crates/aprender-registry` (`pacha`) data lineage crate).*

---

# Minor Corrections and Typos

### 1. Table Ordering Anomalies

#### 1.1 Decision D-8 Placed After D-11 in §0
- **Location**: `PP-066-release-spec.md` line 43.
- **Issue**: The decision table rows are ordered: `D-1`, `D-2`, `D-3`, `D-4`, `D-5`, `D-6`, `D-7`, `D-9`, `D-10`, `D-11`, and finally `D-8`.
- **Correction**: Move `D-8` to immediately follow `D-7` and precede `D-9` to restore numerical ordering.

#### 1.2 Criterion C10 Placed After C13 in §4
- **Location**: `PP-066-release-spec.md` line 160.
- **Issue**: The release criteria table rows are ordered: `C0`, `C1`, `C2`, `C3`, `C4`, `C5`, `C6`, `C7`, `C8`, `C9`, `C11`, `C12`, `C13`, and finally `C10`.
- **Correction**: Relocate `C10` to its proper position between `C9` and `C11`.

---

### 2. Typographical and Formatting Defects

#### 2.1 Bolding Inconsistency in §1 Findings Register
- **Location**: `PP-066-release-spec.md` lines 77–80.
- **Issue**: Rows `F-25`, `F-26`, `F-27`, and `F-28` are formatted with bold identifiers, severities, and descriptions:
  `| **F-25** | **S1** | **Root dysfunction is design, not configuration** ... |`
  Rows `F-1` through `F-22` use unbolded text:
  `| F-1 | S1 | Five specs in one document ... |`
- **Correction**: Standardize table typography across all rows. If special emphasis is desired for the four primary architectural root causes, explain the convention in the section preamble.

#### 2.2 Finding F-3b Alphanumeric Numbering
- **Location**: `PP-066-release-spec.md` line 57.
- **Issue**: Finding `F-3b` is the only row using alphanumeric suffix notation.
- **Correction**: If renumbering is permitted, number sequentially. If `F-3b` is preserved to maintain stable external citations to the original review swarm, add an explicit note in the table header explaining the split.

#### 2.3 Stale Line Citations for CliError::FeatureDisabled in D-7
- **Location**: `PP-066-release-spec.md` line 39.
- **Issue**: `D-7` states: `the tree maps CliError::FeatureDisabled → 9 (crates/apr-cli/src/error.rs:86-90,218 [A] review MIN-07)`.
  Inspection of `crates/apr-cli/src/error.rs` at HEAD shows:
  - Lines 86–90 are macro invocations inside `pub fn exit_code(&self)`.
  - The actual numeric mapping `Self::FeatureDisabled(_) => 9` is located at line **106**.
  - Line 218 is inside `find_model_file`.
- **Correction**: Update citation to reflect HEAD:
  `crates/apr-cli/src/error.rs:106 [V] (formerly lines 86-90,218 at 587ad0797 [A])`.

#### 2.4 Escaped Table Pipes in Inline Code Blocks
- **Location**: `PP-066-release-spec.md` lines 113, 114, 116, 119, 120, 121, 125, 134, 148, 157.
- **Issue**: In markdown tables, pipes within inline code blocks are escaped with backslashes (e.g. `pmat work list --status all \| grep -E ...`). While this prevents markdown parser corruption in certain basic renderers, copying the command directly from the markdown source into a terminal causes bash to execute `\|` as a syntax error or unexpected literal.
- **Correction**: Ensure that automated command extractors or readers strip backslashes preceding pipe symbols, or document that code snippets in table cells require backslash de-escaping before execution.

#### 2.5 Inconsistent Mark Annotation on Decision Rows in §0
- **Location**: `PP-066-release-spec.md` lines 33–43.
- **Issue**:
  - `D-1` carries `[C]`.
  - `D-2` carries `[C]`.
  - `D-3`, `D-4`, `D-6`, and `D-8` carry no provenance mark in their evidence column.
  - `D-5`, `D-7`, `D-9`, `D-10`, and `D-11` carry `[A]`.
- **Correction**: Apply mark discipline uniformly across all evidence entries. For instance, `D-3` cites `guide §3 host table` and should be marked `[A]`; `D-4` cites `master §12 preamble` and should be marked `[V]`; `D-6` references `§3 S0-3` and should be marked `[U]`.
