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
