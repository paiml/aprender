# Summary

This candidate evaluation reviews Segment 1 (`Preamble`, `§0 Decisions D-1..D-11`, `§1 Findings register F-1..F-28`, `§2 Scope`, `§3 Step-0 Discovery S0-1..S0-23`, and `§4 Release criteria C0..C14`) of `docs/specifications/PP-066-release-spec.md` (v1.5) at git commit `a99236a86` (HEAD).

The evaluation audited the document across three mandatory dimensions:
1. **Spec Compliance**: Evaluated adherence to the `paiml-implement` lifecycle (AUTO-IMPL-SKILL-001), ticket minting invariants, mark discipline (`[V]`, `[C]`, `[A]`, `[U]`), release criteria precedence (specifically C0 precedence), and scope boundaries between 0.66 and 0.67.
2. **Grammar and Clarity**: Assessed technical prose precision, table formatting, consistency of mathematical notation, unambiguous command phrasing, and adherence to actionable criteria.
3. **Code Metrics and Status**: Verified all file paths, crate names, commit hashes, Cargo feature definitions, shell scripts, and tool CLI invocations (`pmat`, `gh`, `cargo`, `make`, `git`) against the active working tree.

### Key Takeaways
- **Structural Integrity**: The specification establishes a disciplined foundation by decoupling release decisions, converting findings into falsifiable hypotheses, enforcing single-queue WIP limits (heijunka), and prioritizing C0 (readable and strict CI gates) before crediting any downstream performance claims.
- **Critical Flaws in Step-0 Discovery (§3)**:
  - **Ticket Definition Mismatch**: Line 109 mints a single ticket whose description states `"Falsify the eight premises the 0.66 plan depends on"`, yet the accompanying table defines **23 premises** (S0-1 through S0-23).
  - **Multi-Host Atomicity Violation**: Step-0 bundles execution steps requiring physical access to four disparate bare-metal hosts (`intel`, `mini`, `lambda`, `gx10`) into a single `paiml-implement` unit, violating single-host task atomicity.
  - **Broken Commands in S0 Premises**: S0-1 fails due to markdown bolding in table headers (`grep -n '^| 22'`); S0-2 fails because `pmat work list --status all` is an invalid CLI invocation; S0-12 fails because `aprender-gpu` has no `metal` Cargo feature (removed in PR 2849); S0-18 fails because `scripts/pv_bin.sh` lacks execute permissions (`chmod -x`).
- **Non-Executable Release Criteria (§4)**:
  - §4 states that every release criterion is a command that "must exit 0 on the release commit". However, criteria checks mix prose descriptions, non-executable comparison operators (e.g., `≥ 2` in C1), broken shell syntax (`grep -L 'partial=false' empty` in C9), external repository dependencies (`make -C machines/clean-room clean-room-p1` in C8), and unnamed scripts (`script exit 0` in C7).
  - Criteria list ordering is disordered: C10 is placed after C13 at the end of the table. C14, cited in partition scopes, is absent from the spec (the document defines C0 through C13, totaling 14 criteria).
- **Ticket Namespace Collision**:
  - `T-1` is assigned to `τ_loss derived, not declared` in §5 (0.66), while `T-1..T-5` simultaneously designates the carried 0.67 training levers in §0 D-1 and §6.
  - Ticket `T-2` (`--max-seq-len honoured-or-refused`) is fully carded in §5 but omitted from the 0.66 scope inventory table in §2.
- **Code & Repository State Drift**:
  - `CB-1700` (`required_status_checks.strict=true`) already passes at HEAD, contrary to S0-23 and F-28 claims that it fails.
  - Line numbers for `crates/apr-cli/src/error.rs` and `crates/aprender-serve/src/gguf/wgpu_backend/mod.rs` have drifted.
  - `machines/clean-room` does not exist within `paiml/aprender`; it resides in the sibling `paiml/infra` repository.

---

# Potential Mistakes and Improvements

### 1. Step-0 Discovery Ticket Specification and Multi-Host Execution Flaws (§3)
- **Observation**:
  - Line 109 specifies:
    ```bash
    pmat work add "PP-066 S0: discovery ledger at HEAD" --description "Falsify the eight premises the 0.66 plan depends on; emit docs/audits/pp-066-s0-ledger.md with a verdict per premise"
    ```
  - The table that follows (lines 111–135) lists 23 distinct premises (S0-1 through S0-23).
  - Furthermore, premises explicitly demand commands run on specific bare-metal machines: S0-3 and S0-4 ("on intel"), S0-12 ("on mini"), S0-14 ("on intel and mini"), S0-21 ("on gx10... on mini"), and S0-22 ("on lambda/gx10... on intel").
- **Logic Chain**:
  1. In v1.0, S0 contained 8 premises. As v1.1–v1.5 introduced new findings (F-25 through F-28) and added premises up to S0-23, the table expanded, but the `pmat work add` description string was never updated.
  2. A `paiml-implement` worker executing this ticket receives a description specifying 8 premises, conflicting with the 23 rows required by the spec.
  3. A single autonomous worker executes on a single runner/host. It cannot execute physical commands (`lspci`, `vulkaninfo`, `stress-ng`, local GPU builds) simultaneously across `intel`, `mini`, `lambda`, and `gx10`.
- **Improvement / Remediation**:
  - Update line 109 to state: `"Falsify the 23 premises the 0.66 plan depends on"`.
  - Split S0 into host-partitioned discovery tickets or define a coordinated multi-host dispatch protocol:
    - `S0-HOST`: Local runner inspection (`main` commit, GitHub API, git grep, Cargo manifests, local pmat).
    - `S0-INTEL`, `S0-MINI`, `S0-LAMBDA`, `S0-GX10`: Host-specific receipts collected via SSH/dogfood harness into `docs/audits/pp-066-s0-ledger.md`.

### 2. Defective Shell Commands in Step-0 Premises (§3)
Empirical verification of commands specified in §3 revealed multiple syntax, regex, and tool incompatibilities:

#### a. S0-1 Regex Failure on Bold Table Cells
- **Observation**:
  - Line 112 specifies: `grep -n '^| 22' docs/specifications/PP-LLAMA-001-MASTER.md`.
  - In `docs/specifications/PP-LLAMA-001-MASTER.md`, line 364 contains:
    ```markdown
    | **22** | **instrument**: a top-2 logit margin per generated token on the wire...
    ```
  - Running `grep -n '^| 22'` exits with code 1 (no match).
- **Impact**: Any script running S0-1 will report row 22 as missing, triggering an unnecessary fallback ticket (I-00) and incorrectly declaring master specification drift.
- **Fix**: Update command to support optional bold delimiters:
  ```bash
  grep -nE '^\|\s*\*{0,2}22\b' docs/specifications/PP-LLAMA-001-MASTER.md
  ```

#### b. S0-2 Invalid `pmat work list` Argument
- **Observation**:
  - Line 113 specifies: `pmat work list --status all | grep -E 'PP-(6|27|28|21|15|29|26)'`.
  - Executing this command produces:
    ```
    Error: unknown status 'all' (did you mean 'new'?)
    Valid values: planned, todo, open, pending, new, inprogress, in-progress, wip, active, started, working, blocked, stuck, waiting, on-hold, review, reviewing, pr, pending-review, completed, done, finished, closed, cancelled, canceled, dropped, wontfix
    ```
- **Impact**: S0-2 crashes immediately upon execution.
- **Fix**: Omit `--status all` (default `pmat work list` outputs all work items across statuses):
  ```bash
  pmat work list | grep -E 'PP-(6|27|28|21|15|29|26)'
  ```

#### c. S0-12 Non-Existent `metal` Feature in `aprender-gpu`
- **Observation**:
  - Line 124 specifies: `on mini: cargo check -p aprender-gpu --features metal; cargo test -p aprender-gpu --features metal -- metal_devices`.
  - Running `cargo check -p aprender-gpu --features metal` fails:
    ```
    error: the package 'aprender-gpu' does not contain this feature: metal
    ```
  - Inspection of `crates/aprender-gpu/Cargo.toml` lines 64–80 confirms features are: `cuda`, `wasm`, `stress-test`, `tui-monitor`, `gpu-pixels`, `wgpu`.
  - SARIF audit for PR #2849 explicitly documents that the `metal` feature and `manzana::metal` exports were removed.
  - Furthermore, `crates/aprender-gpu/src/backend/metal_shaders.rs` contains only raw MSL shader string constants (`ELEMENTWISE_ADD`, etc.), while `MetalBackend` in `crates/aprender-gpu/src/backend/mod.rs:67-81` is a hardcoded stub where `is_available(&self) -> bool { false }`.
- **Impact**: S0-12 cannot be executed as written.
- **Fix**: Reframe S0-12 as a verification that `crates/aprender-gpu/src/backend/metal_shaders.rs` contains the 13 MSL shaders, and document that restoring a buildable `metal` feature requires re-integrating `manzana` under ticket P-0.5 / B-M1.

#### d. S0-18 Unexecutable `scripts/pv_bin.sh` and Broken Grep Logic
- **Observation**:
  - Line 130 specifies:
    ```bash
    scripts/pv_bin.sh; pv lint contracts/ | tail -3 (expect 0 err / 1,082 warn / PASS); grep -c 'registry: true' contracts/*.yaml (expect 481)
    ```
  - In filesystem: `ls -la scripts/pv_bin.sh` shows `-rw-rw-r--` (permissions 0664, non-executable). Invoking `./scripts/pv_bin.sh` fails with exit code 126 (`Permission denied`).
  - Running `grep -c 'registry: true' contracts/*.yaml` emits a per-file list (`contracts/foo.yaml:1`, `contracts/bar.yaml:0`) over hundreds of files rather than a single aggregated count.
  - Counting matches across files yields **501**, not 481:
    ```bash
    grep -h 'registry: true' contracts/*.yaml | wc -l # returns 501
    ```
- **Fix**: Ensure `scripts/pv_bin.sh` is marked executable (`chmod +x scripts/pv_bin.sh`), update the expected count from 481 to 501 (or mark `[U]`), and pipe grep output:
  ```bash
  grep -rh 'registry: true' contracts/ | wc -l
  ```

#### e. S0-23 / F-28 Stale Compliance Status
- **Observation**:
  - S0-23 (line 134) and F-28 (line 81) state: `CB-1700/1701/2100 fail at HEAD; required_status_checks.strict is false`.
  - Executing `pmat comply check` at HEAD reveals:
    ```
    ✓ CB-1700: Branch Protection: default branch requires ["ci / gate", "workspace-test"], >=1 approving review, and forbids force-push
    ✗ CB-1701: Supply Chain: 1 supply-chain violation(s): no required status check is known to run a blocking cargo deny check...
    ✗ CB-2100: Comply Gate Effect: 9 severity=error rule(s) unreachable from required check(s)...
    ```
  - Executing GitHub API query:
    ```bash
    gh api repos/paiml/aprender/branches/main/protection --jq '{strict:.required_status_checks.strict,force:.allow_force_pushes.enabled,del:.allow_deletions.enabled}'
    ```
    Returns: `{"del": false, "force": false, "strict": true}`.
- **Impact**: `CB-1700` is already green and branch protection is already `strict=true`.
- **Fix**: Update S0-23 and F-28 to acknowledge that CB-1700 and `strict=true` are CONFIRMED/PASSING at HEAD, leaving CB-1701 and CB-2100 as the active blockers under C0.

---

### 3. Non-Executable and Ill-Formed Release Criteria Commands (§4)
Line 144 mandates:
> *"The tag is cut when **all** hold; each is a command that must exit 0 on the release commit."*

However, the entries in table §4 violate this rule:

| ID | Spec Entry in "Check" Column | Failure Mode / Flaw | Concrete Executable Replacement |
|---|---|---|---|
| **C0** | `pmat comply check \| grep -E 'CB-(1700\|1701\|2100)' prints no ✗; gh api ... prints true; the #2830 ticket's perf_gate.sh --selftest row ... is GREEN ...; CB-2100's output names each of the nine rules under a required context` | Compound sentence mixing shell commands with prose assertions (`prints no ✗`, `is GREEN`). Cannot exit 0 directly in CI. | `scripts/check_c0_release_gate.sh` wrapping the three programmatic checks, exiting 0 on clean pass. |
| **C1** | `scripts/perf_gate.sh --phase release receipt validates; grep -c CONFORMANT evidence/parity/LEDGER.md ≥ 2` | Contains raw mathematical comparison symbol `≥`, causing bash parse error (`syntax error near unexpected token 'newline'`). | `scripts/perf_gate.sh --phase release && [ $(grep -c CONFORMANT evidence/parity/LEDGER.md) -ge 2 ]` |
| **C6** | `both scripts exit 0 at HEAD; each has a PR with a mutation commit → RED → revert in its history` | Entirely prose description; no executable test or command. | `scripts/check_crate_names.sh && scripts/check_backend_firstclass.sh` |
| **C7** | `script exit 0` | The check cell literally says `script exit 0` without specifying the script name. | `scripts/check_no_claim_literals.sh` |
| **C8** | `make -C machines/clean-room clean-room-p1 exit 0` | Directory `machines/clean-room` does not exist in `paiml/aprender`. Executing from repo root produces: `make: *** machines/clean-room: No such file or directory. Stop.` | `make -C ../../infra/machines/clean-room clean-room-p1` or wrap in `scripts/check_clean_room.sh`. |
| **C9** | `ls docs/audits/impl-*-receipt.md \| wc -l = ticket count; grep -L 'partial=false' empty` | Non-shell pseudocode (`= ticket count`); `grep -L 'partial=false'` without file arguments hangs waiting for stdin. | `[ $(ls docs/audits/impl-*-receipt.md \| wc -l) -eq $(pmat work list --status closed \| wc -l) ] && ! grep -L 'partial=false' docs/audits/impl-*-receipt.md \| grep -q .` |
| **C11** | `scripts/check_backend_registry.sh exit 0 on all four hosts; pmat query --regex 'cfg!\(.*feature = "(cuda\|wgpu)"' --path crates/apr-cli/src → 0 hits outside the registry module` | Contains prose arrow `→ 0 hits...` and cross-host assertion text. | `scripts/check_backend_registry.sh && [ $(pmat query --regex 'cfg!\(.*feature = "(cuda\|wgpu)"' --path crates/apr-cli/src \| grep -v 'registry.rs' \| wc -l) -eq 0 ]` |
| **C12** | `scripts/pv_bin.sh lint contracts/ ... exit 0 with the PP-066 contracts in the strict set; scripts/check_contract_test_binding.sh baseline unchanged (13 lines, may not grow); per-contract pv score REPORTED in the ticket receipt...` | Prose requirements concatenated after command. | Script invocation with explicit bash assertions: `scripts/pv_bin.sh lint contracts/ --binding contracts/aprender/binding.yaml --strict-test-binding && [ $(wc -l < scripts/check_contract_test_binding.sh.baseline) -le 13 ]` |
| **C13** | `gh release view v0.66.0 --json assets --jq '[.assets[].name]' lists 5 apr-* tarballs + checksums + signature; curl -LsSf …/install.sh \| bash on each host exits 0 and its last lines are apr devices output` | Ellipsis URL (`…/install.sh`) and prose assertions. | Parameterized validation script checking asset schema and test-running installer. |
| **C10** | `scripts/check_dag_invariants.sh docs/specifications/pp-066-dag.yaml --min-slack-days 6 exit 0 (G-4)` | Prose note `exit 0 (G-4)` appended to command line. | `scripts/check_dag_invariants.sh docs/specifications/pp-066-dag.yaml --min-slack-days 6` |

---

### 4. Ticket Namespace Collision (`T-1`, `T-2`) and Scope Table Omission (§0, §2, §5, §6)
- **Observation**:
  - In §5 Track T (line 354), `T-1` is defined as: `**T-1 · τ_loss derived, not declared** (closes F-8a)`.
  - In §5 Track T (line 360), `T-2` is defined as: `**T-2 · --max-seq-len honoured-or-refused** (closes F-8d, MAJ-06)`.
  - However, in §0 D-1 (line 34), the text states: `Backend lanes, kernel work, T-1..T-5, renames → 0.67`.
  - In §6 carried rows (line 459), row `T-1..T-5` defines the 0.67 training levers (`pre-compiled sm_121; fused RMSNorm/RoPE/SwiGLU/chunked CE with their backward kernels...`).
  - In §2 Scope Table (line 96), the 0.66 lane lists: `T-0h harness, T-0 four WT receipts, T-1 τ_loss, T-3 gate REPORTING`. `T-2` is entirely missing from the cell, even though it is carded in §5.
- **Logic Chain**:
  1. The report used `T-1..T-5` for the GPU training optimization levers.
  2. This specification introduced new Track T cards for 0.66 (`T-0h`, `T-0`, `T-1`, `T-2`, `T-3`), creating an identical identifier namespace collision with the 0.67 levers.
  3. Omitting `T-2` from the §2 Scope table creates an internal contradiction between §2 and §5.
- **Fix**:
  - Add `T-2` to the §2 Scope table 0.66 contents list.
  - Rename the 0.67 carried training levers in §0 D-1, §2, and §6 from `T-1..T-5` to `TL-1..TL-5` (Training Levers) or `T-4..T-8` to eliminate ambiguity.

---

### 5. Findings Register Sequence Gaps and Sub-Indices (§1)
- **Observation**:
  - The findings table lists: F-1, F-2, F-3, F-3b, F-4... F-22, F-25, F-26, F-27, F-28.
  - **F-23 and F-24 do not exist anywhere in the document**.
  - **F-3b** uses an ad-hoc alphanumeric sub-index rather than an integer sequence.
- **Impact**: Cross-referencing findings against PRs, commit messages, and audit receipts becomes prone to confusion when numbers are skipped or arbitrarily sub-indexed without an explanatory note.
- **Fix**: Add an explicit note in §1 documenting why F-23 and F-24 were retired or merged (e.g., absorbed into F-25), or re-index sequentially.

---

# Minor Corrections and Typos

### 1. Table Ordering Inconsistencies
- **§0 Decisions Table (lines 33–44)**:
  - Row order is: `D-1, D-2, D-3, D-4, D-5, D-6, D-7, D-9, D-10, D-11, D-8`.
  - `D-8` (Rename sequencing) was added in v1.1 and placed at the bottom of the table below D-11.
  - *Correction*: Move `D-8` between `D-7` and `D-9`.
- **§4 Release Criteria Table (lines 148–161)**:
  - Row order is: `C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C11, C12, C13, C10`.
  - `C10` (DAG invariants) was appended at the end of the table below C13.
  - *Correction*: Move `C10` between `C9` and `C11`.
- **Criteria Range Parity (Prompt & Partition)**:
  - The partition and prompt specify criteria range `C0..C14`.
  - The document defines 14 criteria, but numbers them zero-indexed from `C0` through `C13`. No criterion `C14` exists.
  - *Correction*: Harmonize document text and external audit partition to cite `C0..C13 (14 criteria)`.

### 2. Code Line Number & Reference Drift
- **D-7 / Review MIN-07 (line 40)**:
  - Spec cites `crates/apr-cli/src/error.rs:86-90,218`.
  - In current code:
    - `CliError::exit_code(&self)` is at lines 84–89.
    - `CliError::exit_code_value(&self)` maps `FeatureDisabled(_) => 9` at **line 106**.
    - The corresponding test `test_feature_disabled_exit_code` is at **lines 284–287** (asserting `ExitCode::from(9)`). Line 218 is inside `resolve_model_path`.
  - *Correction*: Update citation to `crates/apr-cli/src/error.rs:106,284-287`.
- **S0-13 (line 124)**:
  - Spec cites `OwnedQuantizedModelWgpu is a stub returning UnsupportedOperation (gguf/wgpu_backend/mod.rs:140-145)`.
  - In current code: `forward_qwen3_moe_wgpu` returns `UnsupportedOperation` at **lines 197–208**. Lines 140–145 are in the middle of validation logic.
  - *Correction*: Update citation to `crates/aprender-serve/src/gguf/wgpu_backend/mod.rs:197-208`.
- **Preamble Formatting (lines 5–6)**:
  - Awkward hard line break inside parenthetical:
    ```markdown
    ... governing inference spec docs/specifications/PP-LLAMA-001-MASTER.md (v3.0 in
    hand; v3.1 [U]) ...
    ```
  - Furthermore, `PP-LLAMA-001-MASTER.md` is committed on `main` at version 3.1 (verified at line 434 of that file).
  - *Correction*: Reflow line and update `(v3.0 in hand; v3.1 [U])` to `(v3.1 in hand [V])`.

### 3. Phrasing and Table Header Typos
- **§0 Table Header vs Recommended Default Cells (lines 32–42)**:
  - Column 3 header states: `recommended default (applied below, marked *pending*)`.
  - In the table rows, neither D-1 nor D-9 uses the literal word `*pending*` or an asterisk in their cells.
  - *Correction*: Either add `*pending*` tags to the recommended default cells as promised, or adjust the column header.
- **F-6 Attribution (line 60)**:
  - Text cites `defect class the dogfood skill exists to kill (36/77/103/111)` without an evidentiary mark.
  - *Correction*: Add provenance or mark as `[A]` (taken from review).
- **Date Arithmetic Precision (§2 line 92)**:
  - Text reads: `0.66 tag target: 2026-10-23 [A]. Cycle: 49 days [C].`
  - From status date 2026-09-05 to 2026-10-23: September has 30 days (30 - 5 = 25 remaining); October has 23 days (25 + 23 = 48 days). Inclusive interval is 49 days.
  - *Correction*: Clarify `49 days (inclusive) [C]`.
