# Summary

This aggregated review synthesizes and evolves the independent forensic evaluations conducted by Analyst 1 (`handoff_1.md`) and Analyst 3 (`handoff_3.md`) on Segment 1 (`decisions_findings_scope_discovery_criteria`) of `PP-066-release-spec.md` (v1.5, dated 2026-09-05).

### Scope of Segment 1
- **Preamble** (lines 1–25): Architectural context, mono-repo topology, workflow contracts, and provenance mark conventions (`[V]`, `[C]`, `[A]`, `[U]`, `[X]`).
- **§0 Decisions required before ticket #1** (`D-1`–`D-11`, lines 26–45): Governance decisions, scope partitioning, non-CUDA host allocations, refusal codes, and PV-IMPROVE-001 phase boundaries.
- **§1 Findings register — deltas from the report** (`F-1`–`F-28`, lines 46–87): 27 registered findings classifying structural defects, five-whys root cause analyses (`F-25` design vs configuration, `F-26` contract theater, `F-27` missing release assets, `F-28` branch gate holes).
- **§2 Scope (pending D-1)** (lines 88–105): 0.66 instrument/honesty lane vs 0.67 speed/expansion lane, WIP limits, and host queue sequencing.
- **§3 Step-0 — discovery** (`S0-1`–`S0-23`, lines 106–139): 23 read-only pre-implementation verification premises, commands, and falsification rules.
- **§4 0.66 release criterion** (`C0`–`C13`, lines 140–165): 14 exit criteria governing the 0.66 release tag, anchored by `C0` precedence.

---

### Aggregation & Verification Methodology
Each finding from candidate reports `handoff_1.md` and `handoff_3.md` was audited against the active repository tree (`paiml/aprender` at HEAD, commit `a99236a86` on worktree `pp-066-spec`). Forensic validation included inspecting crate manifests, source code (`crates/apr-cli`, `crates/aprender-serve`, `crates/aprender-gpu`), shell scripts (`scripts/pv_bin.sh`, `scripts/perf_gate.sh`, `scripts/check_no_claim_literals.sh`), companion specifications (`PP-LLAMA-001-MASTER.md`), and executing live CLI audits (`pmat comply check`, `pmat work list`, bash subshell invocations, and regular expression evaluations).

#### 1. Verified Consensus Agreements
Both candidates converged on the following critical findings, each independently verified:
1. **Step-0 Premise Count Desynchronization:** §3 specifies the ticket description as `"Falsify the eight premises the 0.66 plan depends on"`, while the table contains 23 premises (`S0-1`..`S0-23`).
2. **Missing Criterion C14:** The dispatch specification and partition bounds reference `C0..C14`, yet §4 defines only 14 criteria (`C0`–`C13`), with no `C14` present in the text.
3. **Register Discontinuity (`F-23`, `F-24`):** §1 jumps directly from `F-22` to `F-25`, leaving `F-23` and `F-24` unassigned and unexplained.
4. **Table Ordering Anomalies:** Decision `D-8` is displaced to the bottom of §0 after `D-11`; Criterion `C10` is displaced to the bottom of §4 after `C13`.
5. **Defective Regex in `S0-1`:** In `S0-1`, `grep -n '^| 22'` fails to locate row 22 in `PP-LLAMA-001-MASTER.md` because the row identifier is rendered with markdown bolding (`| **22** |`).
6. **Fatal Shell Syntax in Release Gate `C1`:** `grep -c CONFORMANT evidence/parity/LEDGER.md ≥ 2` treats the Unicode character `≥` and `2` as file paths, exiting with code 2.
7. **Pseudocode in Release Gate `C9`:** `wc -l = ticket count; grep -L 'partial=false' empty` fails to parse in any POSIX shell.
8. **Stale Line Citations in `D-7`:** `crates/apr-cli/src/error.rs:86-90,218` is stale; the numerical mapping `Self::FeatureDisabled(_) => 9` is located at line 106.
9. **Stale Governing Master Status:** Preamble states `(v3.0 in hand; v3.1 [U])`, whereas `docs/specifications/PP-LLAMA-001-MASTER.md` v3.1 is already committed at HEAD on `main` (`[V]`).
10. **Escaped Pipes in Markdown Cells:** Escaped pipes (`\|`) within inline backticks in table cells break direct copy-paste execution in bash.

#### 2. Analyst 1 Unique High-Confidence Findings (Verified & Absorbed)
- **Silent No-Op Gate in `C12` and Subshell Loss in `S0-18`:** `scripts/pv_bin.sh` is designed solely to be *sourced* (`. scripts/pv_bin.sh`). It ignores all positional arguments and exits 0. Running `scripts/pv_bin.sh lint contracts/ ...` exits 0 without running `pv lint`, turning release gate `C12` into a vacuous pass. In `S0-18`, running `scripts/pv_bin.sh; pv lint ...` in a subshell loses `$PV`, falling back to unpinned `pv` on `PATH`.
- **Unhandled Flag in `pmat work list --status all` (`S0-2`):** `pmat work list --status all` crashes with `Error: unknown status 'all'`. Omitting `--status` lists all statuses by default.
- **Omission of Card `T-2` from the §2 Scope Table:** Card `T-2` (`--max-seq-len` honoured or refused) is an essential 0.66 blocker for `T-0` (detailed in §3 `S0-11`, §5, §6, §7, and Appendix A), but is omitted from the 0.66 lane in §2.
- **Drifted Line Citation in `S0-13`:** `handlers.rs:888-953` cited in `S0-13` does not exist; `crates/aprender-serve/src/cli/handlers.rs` has only 483 total lines.
- **Empirical Status of `CB-1700` in `pmat comply check`:** Live execution reveals `CB-1700` already PASSES (`✓`) at HEAD; only `CB-1701` and `CB-2100` remain open defects.

#### 3. Analyst 3 Unique High-Confidence Findings (Verified & Absorbed)
- **Missing Extended Regexp (`-E`) in `S0-14` and `S0-19`:** `grep -ci 'cuda|cublas'` without `-E` treats `|` as a literal character, outputting `0` even on binaries linked against CUDA/cuBLAS, creating a dangerous false negative.
- **Removal of `MetalBackend` in PR 2849 vs `S0-12`:** PR 2849 removed the `metal` Cargo feature and `manzana::metal` exports from `crates/aprender-gpu`. Running `cargo check -p aprender-gpu --features metal` fails immediately because the feature does not exist. `metal_shaders.rs` contains raw MSL string constants only.
- **Function Renaming in `scripts/perf_gate.sh` (`F-28`, `S0-23`, `C0`):** `arm_a_scaling` was renamed to `arm_a_self_regression` at line 620 of `scripts/perf_gate.sh` under PP-31.
- **Missing Executable Path in `C7`:** Check column literally states `script exit 0` instead of naming `scripts/check_no_claim_literals.sh`.
- **Ticket Identifier Collisions:** `D-1` designates both Decision 1 in §0 and the CUDA documentation ticket in §2/§5; `T-1` designates both the 0.66 `τ_loss` seed variance instrument and the 0.67 training speed lever.
- **Disambiguation of `BackendRegistry` from `crates/aprender-registry`:** Clarifies that the hardware discovery registry belongs in `crates/aprender-serve/src/gpu/backend.rs`, preventing confusion with the existing `pacha` model/data lineage crate (`crates/aprender-registry`).

---

# Potential Mistakes and Improvements

## 1. Critical Gate Flaws and Shell Syntax Defects

### 1.1 Silent No-Op in Release Gate `C12` and Subshell Variable Loss in `S0-18`
- **Location:** Line 159 (Criterion `C12`) and Line 129 (Premise `S0-18`).
- **Observed Code:**
  `scripts/pv_bin.sh` lines 1–6 explicitly mandate sourcing:
  ```bash
  # Source it, never execute it:
  #     . scripts/pv_bin.sh || exit 1
  #     "$PV" lint contracts/
  ```
  Lines 650–663 resolve `$PV`, assert freshness, export `PV`, and return/exit. The script accepts zero CLI arguments and has no command execution dispatcher (e.g. `exec "$PV" "$@"`).
- **Execution Failure:**
  1. In `C12`, executing `scripts/pv_bin.sh lint contracts/ --binding ...` runs the script in a subshell. The arguments `lint contracts/ ...` are completely ignored. The script exports `PV` within the subshell and exits with code `0`. **The release gate passes unconditionally without validating a single contract.**
  2. In `S0-18`, running `scripts/pv_bin.sh; pv lint contracts/ | tail -3` executes the script as a standalone command. When the subshell exits, the exported `$PV` is discarded. The subsequent `pv` command resolves to whatever unpinned, potentially stale binary is on `PATH`, violating the repository's strict binary pinning invariant.
- **Remedy:**
  1. Add an execution fallback dispatcher at the bottom of `scripts/pv_bin.sh`:
     ```bash
     if [ "${BASH_SOURCE[0]}" = "$0" ] && [ "$#" -gt 0 ]; then
       exec "$PV" "$@"
     fi
     ```
  2. Update `C12` check command to source the resolver explicitly:
     ```bash
     . scripts/pv_bin.sh && "$PV" lint contracts/ --binding contracts/aprender/binding.yaml --strict-test-binding
     ```
  3. Update `S0-18` command to:
     ```bash
     . scripts/pv_bin.sh && "$PV" lint contracts/ | tail -3
     ```

### 1.2 Unexecutable Shell Expressions in Release Gates `C1`, `C9`, and Incomplete Path in `C7`
- **Location:** Lines 148 (`C1`), 154 (`C7`), and 156 (`C9`).
- **Observed Code:**
  - `C1`: `scripts/perf_gate.sh --phase release receipt validates; grep -c CONFORMANT evidence/parity/LEDGER.md ≥ 2`
  - `C7`: `script exit 0`
  - `C9`: `ls docs/audits/impl-*-receipt.md | wc -l = ticket count; grep -L 'partial=false' empty`
- **Execution Failure:**
  - In `C1`, `≥` (U+2265) is a Unicode glyph. `grep` treats `≥` and `2` as file paths, emitting `grep: ≥: No such file or directory` and exiting with error code 2. Furthermore, `receipt validates` is unquoted human prose.
  - In `C7`, the check column omits the script path entirely, rendering automated execution impossible.
  - In `C9`, `wc -l = ticket count` triggers `bash: =: command not found`. `grep -L 'partial=false' empty` attempts to search a non-existent file named `empty`.
- **Remedy:** Rewrite all three check commands as valid, executable POSIX compound expressions:
  - `C1`:
    ```bash
    scripts/perf_gate.sh --phase release && [ "$(grep -c 'CONFORMANT' evidence/parity/LEDGER.md)" -ge 2 ]
    ```
  - `C7`:
    ```bash
    scripts/check_no_claim_literals.sh
    ```
  - `C9`:
    ```bash
    [ "$(ls docs/audits/impl-*-receipt.md 2>/dev/null | wc -l)" -eq "$TOTAL_066_TICKETS" ] && [ -z "$(grep -L 'partial=false' docs/audits/impl-*-receipt.md 2>/dev/null)" ]
    ```

### 1.3 Defective Regular Expressions in Step-0 Discovery Commands (`S0-1`, `S0-14`, `S0-18`, `S0-19`)
- **Location:** Lines 112 (`S0-1`), 125 (`S0-14`), 129 (`S0-18`), 130 (`S0-19`).
- **Observed Behavior & Failures:**
  - In `S0-1`, `grep -n '^| 22' docs/specifications/PP-LLAMA-001-MASTER.md` returns 0 lines because row 22 is rendered with markdown bolding: `| **22** | **instrument**: ...` at line 364. A naive automated runner reports row 22 missing and falsely declares `S0-1` FALSIFIED.
  - In `S0-14`, `ldd target/release/apr | grep -ci 'cuda|cublas'` omits `-E`. Standard POSIX `grep` treats `|` as a literal vertical bar. Testing live: `echo "cuda" | grep -ci 'cuda|cublas'` outputs `0`, whereas `echo "cuda" | grep -Eci 'cuda|cublas'` outputs `1`. Without `-E`, this check emits a false negative `0` even on binaries dynamically linked to CUDA.
  - In `S0-18`, `grep -c 'registry: true' contracts/*.yaml` outputs 500+ individual file counts instead of an aggregated sum. Furthermore, the count across contracts at HEAD is 501, not 481 (which was measured at commit `b1a6324b8`).
  - In `S0-19`, `grep -rn 'minisign|ssh-keygen -Y|signify' scripts/ .github/workflows/` omits `-E`, matching literal pipe characters rather than alternations.
- **Remedy:**
  - `S0-1`:
    ```bash
    grep -nE '^\|\s*(\*\*)?22(\*\*)?\s*\|' docs/specifications/PP-LLAMA-001-MASTER.md
    ```
  - `S0-14`:
    ```bash
    ldd target/release/apr | grep -Eci 'cuda|cublas'
    ```
  - `S0-18`:
    ```bash
    grep -l 'registry: true' contracts/*.yaml | wc -l
    ```
    *(Document that count at HEAD is 501, or 481 at historical commit `b1a6324b8`)*.
  - `S0-19`:
    ```bash
    grep -rnE 'minisign|ssh-keygen -Y|signify' scripts/ .github/workflows/
    ```

### 1.4 CLI Argument Error in `pmat work list --status all` (`S0-2`)
- **Location:** Line 113, Premise `S0-2`.
- **Observed Code:**
  ```bash
  pmat work list --status all | grep -E 'PP-(6|27|28|21|15|29|26)'
  ```
- **Execution Failure:** Running this command fails immediately with:
  ```
  Error: unknown status 'all' (did you mean 'new'?)
  Valid values: planned, todo, open, pending, new, inprogress, in-progress, wip, active, started, working, blocked, stuck, waiting, on-hold, review, reviewing, pr, pending-review, completed, done, finished, closed, cancelled, canceled, dropped, wontfix
  ```
  `pmat work list` lists all tickets across all statuses by default when `--status` is omitted.
- **Remedy:** Remove `--status all`:
  ```bash
  pmat work list | grep -E 'PP-(6|27|28|21|15|29|26)'
  ```

### 1.5 Multi-Platform Backend Assertion Blindspot in Criterion `C4`
- **Location:** Line 151, Criterion `C4`.
- **Observed Code:**
  ```bash
  scripts/check_multiplatform_dogfood.sh --require-resolved-backend cuda exit 0 on both
  ```
- **Defect:** The text of `C4` mandates dogfooding across all four fleet hosts: `cuda` on `lambda` and `gx10`, and `cpu` with `cuda unavailable reason=DriverNotFound` on `intel` and `mini`. However, the check command only specifies `--require-resolved-backend cuda`. Running that invocation on `intel` or `mini` will fail.
- **Remedy:** Specify the check invocations for both host sets:
  ```bash
  scripts/check_multiplatform_dogfood.sh --host-set cuda --require-resolved-backend cuda && scripts/check_multiplatform_dogfood.sh --host-set cpu --require-resolved-backend cpu
  ```

### 1.6 `C0` GitHub Branch Protection API Nuance (`-F strict=true`)
- **Location:** Line 134 (`S0-23`) and Line 147 (`C0`).
- **Observed Code:**
  ```bash
  gh api -X PATCH repos/paiml/aprender/branches/main/protection/required_status_checks -F strict=true
  ```
- **Defect:** GitHub REST API v3 endpoint `PATCH /repos/{owner}/{repo}/branches/{branch}/protection/required_status_checks` requires a strict JSON payload `{"strict": boolean, "contexts": string[]}`. Passing `-F strict=true` formats parameters as multipart form-data or urlencoded strings. On certain `gh` CLI versions and API endpoints, non-JSON boolean formatting is rejected with HTTP 422 (`strict must be a boolean`). Furthermore, modifying `required_status_checks` without preserving existing contexts can accidentally clear the required status checks list.
- **Remedy:** Format payload as strict JSON:
  ```bash
  gh api -X PATCH repos/paiml/aprender/branches/main/protection/required_status_checks \
    --input - <<< '{"strict": true}'
  ```

---

## 2. Structural Scope Omissions, Ticket Tracking Invariants, and Numbering Gaps

### 2.1 Complete Omission of Release Criterion `C14`
- **Location:** `ANALYSIS_PARTITION.md` line 18, user request, and §4 lines 145–161.
- **Defect:** The orchestrator dispatch and partition document define the criteria range as `C0..C14` (implying 15 criteria). In `PP-066-release-spec.md` §4, only 14 criteria are present: `C0`, `C1`, `C2`, `C3`, `C4`, `C5`, `C6`, `C7`, `C8`, `C9`, `C11`, `C12`, `C13`, and `C10`. Searching the entire repository for `C14` returns zero results.
- **Impact:** An off-by-one fencepost error exists between governing dispatch documents and the specification text, creating uncertainty about whether a 15th criterion (e.g. documentation signoff, binary checksum release verification, or andon compliance) was dropped or intended.
- **Remedy:** Declare explicitly in §4 that the release criteria comprise exactly 14 items indexed `C0` through `C13`, and update `ANALYSIS_PARTITION.md` to reference `C0..C13`. If a 15th criterion was intended (such as `C14: Andon compliance for §5 cards past expiry`), define it explicitly.

### 2.2 Omission of Deliverable Card `T-2` from the §2 Scope Table
- **Location:** Line 95, §2 Scope table.
- **Observed Code:**
  ```markdown
  | **0.66 — instrumented and honest** | ... T-0h harness, T-0 four WT receipts, T-1 τ_loss, T-3 gate REPORTING; ... |
  ```
- **Defect:**
  - Card `T-2` (`--max-seq-len honoured or refused, never clamped`) resolves the silent context-length clamp in `finetune.rs:717`.
  - In `S0-11` (line 122), the spec states: `T-2 (new card) precedes T-0`.
  - In §5 Track T (line 372), card `T-0` explicitly lists `blockers: T-0h, T-1, T-2`.
  - In §6 (line 459), the spec states: `packing (T-2 in 0.66 already covers --max-seq-len)`.
  - In Appendix A (line 663), changelog notes: `new T-2 (--max-seq-len honoured-or-refused)`.
  - Despite being a critical 0.66 deliverable, `T-2` is completely omitted from the 0.66 lane in §2.
- **Remedy:** Update §2 line 95 to include `T-2`:
  ```markdown
  T-0h harness, T-0 four WT receipts, T-1 τ_loss, T-2 max-seq-len honour/refuse, T-3 gate REPORTING;
  ```

### 2.3 Step-0 Discovery Ticket Description Desynchronization (`S0-1..S0-23` vs "Eight Premises")
- **Location:** Line 108, §3.
- **Observed Code:**
  ```bash
  pmat work add "PP-066 S0: discovery ledger at HEAD" --description "Falsify the eight premises the 0.66 plan depends on; emit docs/audits/pp-066-s0-ledger.md with a verdict per premise"
  ```
- **Defect:** The ticket description explicitly states `"Falsify the eight premises"`, but the table immediately following defines **23 premises** (`S0-1` through `S0-23`). The phrase "eight premises" is an un-updated artifact of spec v1.0.
- **Remedy:** Update the ticket description to:
  ```bash
  pmat work add "PP-066 S0: discovery ledger at HEAD" --description "Falsify the 23 premises (S0-1..S0-23) the 0.66 plan depends on; emit docs/audits/pp-066-s0-ledger.md with a verdict per premise"
  ```

### 2.4 Findings Register Discontinuity: Missing Rows `F-23` and `F-24`
- **Location:** Lines 76–77, §1 Table.
- **Defect:** The register progresses from `F-22` (line 76) directly to `**F-25**` (line 77). There are no rows for `F-23` or `F-24`. In a formal audit ledger, unexplained numbering gaps impede traceability.
- **Remedy:** Add an explicit entry or note in the preamble of §1:
  ```markdown
  | F-23..F-24 | — | Retired / merged into F-25 during root-cause synthesis | — | — |
  ```

### 2.5 Ticket Identifier Collisions Across Tracks (`D-1` and `T-1`)
- **Location:** §0 line 33, §2 line 95, §5 line 439, §6 line 459.
- **Defect:**
  1. `D-1` is assigned to Decision 1 in §0 ("0.66 scope: all five tracks or the split in §2"). In §2 and §5 Track D, `D-1` is also assigned to the documentation ticket ("D-1 · cuda-backend-architecture.md").
  2. `T-1` in 0.66 (§2, §5) designates the `τ_loss` seed variance instrument. In 0.67 (§2 line 96, §6 line 459), `T-1..T-5` designates the training speed levers.
- **Impact:** In `pmat work` and git commits, reusing identical ticket identifiers causes misattribution and ambiguity.
- **Remedy:**
  - Rename the documentation card to `DOC-1` or `D-CUDA-1`. Reserve `D-*` strictly for §0 Decisions (`D-1`..`D-11`).
  - Disambiguate the 0.66 training seed instrument as `T-0b` or `T-TAU-1`, keeping `T-1` for the 0.67 speed lever.

### 2.6 Clarification of Decision `D-3` Escalation Scope vs 0.66 Release Hold
- **Location:** Line 35, Decision `D-3`.
- **Observed Code:**
  ```markdown
  | **D-3** | CI home for non-CUDA lanes: may `intel` ... host a wgpu lane ... ; may `mini` ... host nightly Metal? | **Escalate; no default.** APR-QUALITY-001 §9.1 already holds this open ⛔. B-W1, B-S*, B-M* are blocked on D-3 | ...
  ```
- **Defect:** `D-3` is marked ⛔ in §0 ("Decisions required before ticket #1"), implying that 0.66 cannot proceed until `D-3` is resolved. However, in the §2 Scope table, all tickets blocked by `D-3` (`B-W1..B-W5`, `B-M1..B-M4`, `B-S1..B-S4`) are placed in the **0.67 lane**. 0.66 does not provision non-CUDA CI lanes on `intel` or `mini`.
- **Remedy:** Clarify in `D-3` that this escalation blocks 0.67 execution only, and does not block 0.66 ticket #1 or 0.66 release gates.

---

## 3. Empirical Codebase Realities and Specification Drift

### 3.1 Live Status of `CB-1700` in `pmat comply check`
- **Location:** Line 80 (`F-28`), Line 134 (`S0-23`), Line 147 (`C0`).
- **Observed Claims:** The spec asserts that `CB-1700`, `CB-1701`, and `CB-2100` fail at HEAD.
- **Live Empirical Verification:**
  Executing `pmat comply check | grep -E 'CB-(1700|1701|2100)'` at HEAD yields:
  ```
  ✓ CB-1700: Branch Protection: default branch requires ["ci / gate", "workspace-test"], >=1 approving review, and forbids force-push
  ✗ CB-1701: Supply Chain: 1 supply-chain violation(s): no required status check is known to run a blocking cargo deny check...
  ✗ CB-2100: Comply Gate Effect: 9 severity=error rule(s) unreachable from required check(s)...
  ```
- **Finding:** `CB-1700` **already passes** (`✓`) at HEAD. The branch protection configuration has been updated. The active compliance defects are exclusively `CB-1701` and `CB-2100`.
- **Remedy:** Update `F-28`, `S0-23`, and `C0` to reflect that `CB-1700` is verified green at HEAD (`[V]`), leaving `CB-1701` and `CB-2100` as the active gates to satisfy.

### 3.2 MetalBackend Removal in PR 2849 vs `S0-12` Metal Feature Invocation
- **Location:** Line 123, Premise `S0-12`.
- **Observed Code:**
  ```markdown
  crates/aprender-gpu/src/backend/metal_shaders.rs holds 13 MSL kernels and MetalBackend via manzana::metal ... does it build and enumerate a device on mini?
  command: cargo check -p aprender-gpu --features metal; cargo test -p aprender-gpu --features metal -- metal_devices
  ```
- **Codebase Reality:**
  - Inspection of PR 2849 and `crates/aprender-gpu/Cargo.toml` lines 64–80 confirms there is **no `metal` feature** in `aprender-gpu`.
  - `crates/aprender-gpu/src/backend/metal_shaders.rs` lines 8–11 explicitly document:
    ```rust
    // These are source strings only. This crate contains no Metal dispatcher, so
    // nothing here compiles or runs them...
    ```
  - Running `cargo check -p aprender-gpu --features metal` fails immediately with an unknown feature error.
- **Remedy:** Update `S0-12` to state that `metal_shaders.rs` contains raw MSL shader strings only, and that `MetalBackend` was removed in PR 2849. Shape `B-M1` as a restoration and wiring ticket rather than a mere device enumeration test.

### 3.3 Function Renaming in `scripts/perf_gate.sh`: `arm_a_scaling` to `arm_a_self_regression`
- **Location:** Line 80 (`F-28`), Line 134 (`S0-23`), Line 147 (`C0`).
- **Observed Claims:** Cites defect #2830: `perf_gate.sh returns VERDICT PASS on a c=1-only receipt whose arm_a_scaling emitted nothing`.
- **Codebase Reality:**
  Inspection of `scripts/perf_gate.sh` at line 620 reveals:
  ```bash
  620: arm_a_self_regression() {
  621:   # PP-31. SELF-REGRESSION, not scaling efficiency...
  ```
  `arm_a_scaling` was replaced by `arm_a_self_regression` under PP-31.
- **Remedy:** Clarify in `F-28`, `S0-23`, and `C0` that the affected function in `scripts/perf_gate.sh` at current HEAD is `arm_a_self_regression` (formerly `arm_a_scaling`), ensuring tests targeting the selftest flag verify the correct symbol.

### 3.4 Stale Line Citations in `D-7` (`error.rs`) and `S0-13` (`handlers.rs`)
- **Location:** Line 39 (`D-7`) and Line 124 (`S0-13`).
- **Codebase Reality:**
  - In `crates/apr-cli/src/error.rs`, the citation `86-90,218` is stale. Lines 86–90 are macro calls inside `exit_code(&self)`. The actual numerical mapping `Self::FeatureDisabled(_) => 9` is located at line **106**. Line 218 is inside `find_model_file`.
  - In `crates/aprender-serve/src/cli/handlers.rs`, total line count is only 483 lines. The citation `handlers.rs:888-953` does not exist in that file.
- **Remedy:**
  - Update `D-7` to cite `crates/apr-cli/src/error.rs:106`.
  - Update `S0-13` to cite the active dequantization paths in `crates/aprender-serve/src/api/` or `crates/aprender-serve/src/quantize/`.

### 3.5 Governing Inference Spec Status at HEAD
- **Location:** Lines 5–6 (Preamble) and Line 112 (`S0-1`).
- **Codebase Reality:** Line 5-6 states: `governing inference spec docs/specifications/PP-LLAMA-001-MASTER.md (v3.0 in hand; v3.1 [U])`. Direct inspection confirms that `PP-LLAMA-001-MASTER.md` is committed on `main` at HEAD, and line 1 explicitly declares: `# PP-LLAMA-001 v3.1 — MASTER — Inference performance parity with llama.cpp`.
- **Remedy:** Update Preamble line 6 to: `governing inference spec docs/specifications/PP-LLAMA-001-MASTER.md (v3.1 committed at HEAD [V])`.

### 3.6 Disambiguation of Hardware `BackendRegistry` from `crates/aprender-registry` (`pacha`)
- **Location:** Line 77 (`F-25`), Line 95 (§2), Line 157 (`C11`).
- **Codebase Reality:** A workspace crate `crates/aprender-registry` already exists. Its manifest defines `[lib] name = "pacha"`, providing model, data, and recipe registry functionality. In contrast, ticket `R-0` defines a hardware backend registry located at `crates/aprender-serve/src/gpu/backend.rs`.
- **Remedy:** Add an explicit disambiguation note in §1 `F-25` and §2:
  *(Note: BackendRegistry for runtime hardware discovery lives in `crates/aprender-serve/src/gpu/backend.rs`, not to be confused with the existing `crates/aprender-registry` / `pacha` data lineage crate).*

### 3.7 Forward-Looking Script Annotations in Release Criteria (`C2`, `C5`, `C6`, `C10`, `C11`)
- **Location:** Lines 149–160 (§4).
- **Codebase Reality:** Five scripts mandated in §4 do not exist at HEAD:
  - `C2`: `scripts/check_backend_firstclass.sh` (delivered by `B-G1`)
  - `C5`: `scripts/train_parity.sh` (delivered by `T-0h` / `T-1`)
  - `C6`: `scripts/check_crate_names.sh` (delivered by `G-1`)
  - `C10`: `scripts/check_dag_invariants.sh` (delivered by `G-4`)
  - `C11`: `scripts/check_backend_registry.sh` (delivered by `R-0`)
- **Remedy:** Annotate each check command in §4 with its delivering ticket handle, making it explicit to CI runners that these gates arm upon ticket landing.

---

# Minor Corrections and Typos

1. **Table Ordering Sequence in §0 (`D-8`) and §4 (`C10`):**
   - In §0 (line 43), `D-8` is placed at the end of the table after `D-11`. Relocate `D-8` between `D-7` and `D-9`.
   - In §4 (line 160), `C10` is placed at the bottom after `C13`. Relocate `C10` between `C9` and `C11`.

2. **Preamble Filename Typo (Line 3):**
   - *Current:* `rewrite of 0_66-performance-parity-report.md (the "report")`
   - *Correction:* Change to `0.66-performance-parity-report.md` (with period `0.66`, matching the committed file).

3. **Master Expiry Status Precision in `D-4` (Line 36):**
   - *Current:* `The report re-dates master §12 rows 19/20/21 (10-16→09-19, 10-23→09-26, 11-06→10-10) without an Appendix D amendment | Master dates stand until an amendment names who moved them and why`
   - *Correction:* In `PP-LLAMA-001-MASTER.md` v3.1 (lines 361–363), rows 19 and 21 are defined with derived expiries (`OPEN, derived`). Clarify in `D-4`: `Master derived statuses stand (rows 19 and 21 derived from blockers; row 20 expires 2026-10-23)`.

4. **Formatting of Mathematical Comparison in `F-11` (Line 65):**
   - *Current:* `max_batch ≥ 22 has no basis=`
   - *Correction:* Enclose in backticks: `` `max_batch` >= 22 ``.

5. **Clarity of `WIP` Limits in §2 (Lines 99–102):**
   - *Current:* `Per host, ≤ 1 speed row in flight (basis: master §9 ...). gx10 is one queue, ordered by expiry: master 15 (shakedown) → T-0 → S-3's gx10 leg → master 21 (0.67).`
   - *Correction:* Explicitly state that training runs on `gx10` (`T-0`) count against the single speed-row WIP limit, preventing concurrent VRAM starvation with inference runs.

6. **Missing Markdown Escaping in Table Cells:**
   - In table rows containing inline commands with pipes (e.g. lines 113, 114, 116, 119, 120, 121, 125, 134, 147, 156), escaped pipes (`\|`) ensure table layout preservation. Document that command extractors must de-escape backslashes before executing in bash.

7. **Bolding Inconsistency in §1 Findings Register:**
   - Rows `F-25`, `F-26`, `F-27`, and `F-28` (lines 77–80) are heavily bolded across identifiers, severities, and descriptions, whereas rows `F-1` through `F-22` use unbolded text. Standardize typography across the register.

8. **Provenance Mark Discipline Uniformity in §0:**
   - `D-1` and `D-2` carry `[C]`; `D-5`, `D-7`, `D-9`, `D-10`, `D-11` carry `[A]`; but `D-3`, `D-4`, `D-6`, and `D-8` carry no marks. Add appropriate provenance annotations (`[A]` for `D-3`, `[V]` for `D-4`, `[U]` for `D-6`).
