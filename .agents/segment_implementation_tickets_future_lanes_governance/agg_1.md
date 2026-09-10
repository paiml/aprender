# Summary

This document presents the synthesized, independently verified **Aggregated Review (agg_1)** for **Segment 2: `implementation_tickets_future_lanes_governance`** of `docs/specifications/PP-066-release-spec.md` (v1.5, dated 2026-09-05). Segment 2 spans lines 167–664 of the specification, comprising:
- **§5 0.66 tickets — paiml-implement units**: Ticket specifications across Track I (Instrument chain rows 0a–19), Track R (Runtime discovery, refusal, release packaging: `R-0`–`R-7`), Track P (Provable contracts: `P-0.1`–`P-0.6`, `P-1.1`, `P-1.2`), Track S (Speed rows: `S-1`–`S-3`), Track T (Training measurement: `T-0h`, `T-1`, `T-2`, `T-0`, `T-3`), Track B (Backends: `B-A1`, `B-G1`), Track G (Guards & DAGs: `G-1`–`G-4`), and Track D (Documentation: `D-1`).
- **§6 0.67 lane — carried rows**: Deferred performance kernels, hardware lanes (`B-W0`–`B-W5`, `B-M1`–`B-M4`, `B-S1`–`B-S4`, `B-A2`), PVI Phases 1.3–4.3 (#2556), and crate rename sequencing (51 PRs).
- **§7 Registered predictions**: 16 registered prediction rows establishing pre-experiment falsifiers.
- **§8 Refusals (0.66)**: 23 non-negotiable negative constraints and invariant bounds.
- **§9 Toyota Way targets (0.66)**: 26 quantitative quality metrics across Jidoka, Andon, Poka-Yoke, Genchi Genbutsu, Kaizen, and Heijunka.
- **§10 Verification ledger for this document**: Evidentiary citations and provenance tracking.
- **§11 Adjudication of `0_66-review.md`**: Formal rulings on 33 review findings and 5 unraised findings.
- **§12 Prior-art register — GPU discovery**: Comparative architectural synthesis of llama.cpp, Ollama, and llamafile mapped to `REG-1`..`REG-14`.
- **Appendix A — Changelog**: Version history tracking v1.0 through v1.5.

---

### Aggregation Methodology and Verification Baseline

This aggregated report synthesizes the independent evaluations of **Candidate 1** and **Candidate 3**, resolving all discrepancies through empirical re-execution against the active repository tree at `paiml/aprender` (commit `a99236a86`) and against `origin/main` (commit `027ed889d`).

1. **Agreement Identification**: Both candidate reviews exhibited strong consensus on primary architectural and execution defects:
   - The non-executable permissions and argument-dropping behavior of `scripts/pv_bin.sh` creating silent-pass gate theater in `R-0` and `C12`.
   - Four zero-slack (0 days) blocker pairs in the §5 DAG violating C10, G-4, §8, and §9.
   - Release gate `C11` specifying "12 fixtures" while §5 R-0, the REG table, §9, and Appendix A mandate 14 fixtures (`REG-1`..`REG-14`).
   - Stale Step-0 premise counts in §9 line 544 (citing "9/9" instead of the 23 premises in §3).
   - Omission of `REG-10` in the Prior-Art Register table in §12.
   - Missing explicit YAML contract paths for Track P cards.
   - Ticket `T-2` minting string discrepancy (`T-5a`) and omission from §2.
   - Inaccurate architectural claims regarding `aprender-serve` WASM scope and `aprender-gpu` Metal dispatcher readiness.
   - Stale file line citations in §10 (`error.rs:106`, root `Cargo.toml:666`, crate qualification for `handlers.rs` and `attention.rs`).

2. **Integration of Unique High-Confidence Findings**:
   - **From Candidate 1**: Detailed physical queue analysis of `gx10` under `WIP=1`, demonstrating that scheduling `master 15` ahead of `T-0` and `S-3` causes circular starvation and directly inverts the G-4 queue-expiry invariant (`queue_pos(a) < queue_pos(b) ⇒ expires(a) ≤ expires(b)`); analysis of exit code ambiguity between `FeatureDisabled` (9) and `NotImplemented` (12); identification of the Track G scope truncation in the partition metadata; and pinpointing of the unclosed backtick in `R-5`.
   - **From Candidate 3**: Falsification of §10 claims regarding `PP-LLAMA-001-MASTER.md` (proving that master v3.1 is committed on `main`, contains row 22 at line 364, and that rows 19/21 expiries are `derived` per master change 3.0.24); demonstration that `pmat comply check` in `G-4` exits 2 due to unhandled flags (`--rule`, `--min-slack-days`); discovery of the non-existent `machines/clean-room` external path in `R-2` and `R-5`; observation that `pv validate` cannot validate JSON payloads against schemas; discovery that `REG-13` lacks the `FX-13:` fixture prefix; card discipline violations in `G-2` and `G-3`; and mutation count mismatch in §7 `G-1`.

3. **Filtering of False Positives**:
   - Candidate 1's characterization of `PP-LLAMA-001-MASTER-v3.md` as merely a file naming typo was deepened using Candidate 3's empirical proof that row 22 is present and that rows 19 and 21 are `derived`, resolving the root cause of `S0-1` and `F-3b`.

---

### Consensus Defect Matrix for Segment 2

| ID | Location | Summary of Verified Issue | Severity | Dimension |
|---|---|---|---|---|
| **AGG2-01** | §5 line 208, §4 C12, §8 line 510 | `scripts/pv_bin.sh` has non-executable mode (`-rw-rw-r--`) and drops all CLI arguments; subshell execution exits 0 unconditionally, creating silent gate theater in R-0 and C12 | **S1** | Spec Compliance / Metrics |
| **AGG2-02** | §2 lines 100–102, §5 lines 198, 336, 372 | `gx10` queue order (`master 15 → T-0 → S-3`) starves T-0 and S-3 under `WIP=1` and violates G-4 invariant `queue_pos(a) < queue_pos(b) ⇒ expires(a) ≤ expires(b)` because row 15 expires after Oct 08 while T-0/S-3 expire Sep 26 | **S1** | Spec Compliance |
| **AGG2-03** | §5 lines 265, 302, 308, 372 | 4 zero-slack (0 days) blocker pairs in §5 (`0b → R-4`, `P-0.3 → P-0.6`, `P-1.1 → P-1.2`, `T-1 → T-0`) violate C10, G-4, §8, and §9 | **S1** | Spec Compliance |
| **AGG2-04** | §10 lines 567–568, §3 S0-1 | Falsified claims in §10: `PP-LLAMA-001-MASTER.md` v3.1 is committed on `main` and carries Row 22 at line 364; rows 19 and 21 have `derived` expiries struck in master v3.0.24 | **S1** | Code Metrics |
| **AGG2-05** | §5 line 432 | `pmat comply check` in G-4 exits 2 due to unhandled flags (`--rule`, `--min-slack-days`); cross-repo coupling in `aprender` ticket | **S1** | Spec Compliance / Metrics |
| **AGG2-06** | §5 lines 250, 274 | Acceptance command requires `make -C machines/clean-room clean-room-p1` which does not exist in `paiml/aprender` | **S2** | Code Metrics |
| **AGG2-07** | §5 line 208 | `pv validate` only validates YAML contract schemas; cannot validate CLI JSON output (`apr devices --json`) against schemas | **S2** | Spec Compliance |
| **AGG2-08** | §4 C11, §5 line 208, §9 line 539 | Contradiction: C11 specifies "12 fixtures" while R-0, REG table, and §9 mandate 14 fixtures (`REG-1`..`REG-14`) | **S2** | Grammar & Clarity |
| **AGG2-09** | §9 line 544, §3 line 109 | Stale metrics: §9 claims "9/9 Step-0 premises" and §3 claims "eight premises", but §3 defines 23 premises (`S0-1`..`S0-23`) | **S2** | Grammar & Clarity |
| **AGG2-10** | §5 line 226, §12 table | Requirement `REG-10` is omitted from all rows in §12's REG column; `REG-9` omits llama.cpp | **S2** | Spec Compliance |
| **AGG2-11** | §5 lines 295–313, 420, 427 | Card discipline violations: Track P lacks YAML paths; G-2 is an unformatted one-liner; G-3 lacks mutation-to-RED and sets `quorum: none` | **S2** | Spec Compliance |
| **AGG2-12** | §0 D-7, §5 lines 225, 230, §8 line 508 | Refusal exit code ambiguity: `FeatureDisabled` (exit 9) vs `NotImplemented` (exit 12); tests asserting on `--tensor-split` risk asserting wrong code | **S2** | Spec Compliance |
| **AGG2-13** | §5 line 363, §2 line 96 | Ticket minting collision: T-2 mints `"T-5a: ..."`; T-2 is omitted from §2 Scope table | **S2** | Spec Compliance |
| **AGG2-14** | §0 line 33, §5 line 439, §6 line 459 | Dual namespace collisions: `D-1` (Decision 1 vs Doc Ticket D-1); `T-1` (0.66 τ_loss vs 0.67 levers) | **S2** | Spec Compliance |
| **AGG2-15** | §6 lines 463–464, §10 line 575 | Inaccurate architectural claims: `aprender-serve` is not `cfg(not(wasm32))` entirely; `aprender-gpu` has inert Metal shader strings without a driver dispatcher | **S3** | Code Metrics |
| **AGG2-16** | §5 line 229 | Requirement `REG-13` lacks fixture handle prefix `FX-13:` present on all other 13 rows | **S3** | Grammar & Clarity |
| **AGG2-17** | §7 line 489 | Mutation count mismatch: §7 G-1 claims "both mutations RED" while G-1 defines three mutations | **S3** | Grammar & Clarity |
| **AGG2-18** | §5 lines 206, 307 | Temporal paradox: R-0 lands 2026-09-19 without fail-closed contract enforcement; no ticket schedules adding `#[contract]` post-P-1.1 | **S3** | Spec Compliance |
| **AGG2-19** | §5 lines 274, 442 | Acceptance fields in R-5 and D-1 embed descriptive prose and non-executable policy clauses | **S3** | Spec Compliance |
| **AGG2-20** | §10 lines 565, 575 | Stale line citations in §10: `error.rs:106`, root `Cargo.toml:666`, crate qualification needed for `handlers.rs` and `attention.rs` | **S3** | Code Metrics |

---

# Potential Mistakes and Improvements

## 1. Critical Gate-Theater Vulnerability: Non-Executable and Argument-Dropping Behavior in `scripts/pv_bin.sh`

- **Observation**:
  - In §4 C12 (line 159): `scripts/pv_bin.sh lint contracts/ --binding contracts/aprender/binding.yaml --strict-test-binding exit 0`
  - In §5 R-0 Acceptance (line 208): `scripts/pv_bin.sh validate contracts/apr-devices-schema-v1.yaml`
  - In §8 Refusals (line 510): `No pv invocation against the binary on PATH; scripts/pv_bin.sh builds it from the tree.`
  - Direct empirical inspection of `scripts/pv_bin.sh` in the worktree confirms:
    1. File permission mode is `-rw-rw-r--` (non-executable). Invoking `./scripts/pv_bin.sh` fails with `bash: ./scripts/pv_bin.sh: Permission denied` (exit 126).
    2. Lines 3–6 of the script explicitly state:
       ```bash
       # Source it, never execute it:
       #     . scripts/pv_bin.sh || exit 1
       #     "$PV" lint contracts/
       ```
    3. The file ends at line 662 with:
       ```bash
       pv_bin_assert_fresh "$PV" || return 1 2>/dev/null || exit 1
       export PV
       ```
    4. Running `bash scripts/pv_bin.sh validate nonexistent_file.yaml` exits 0 immediately without invoking `$PV` or validating arguments:
       ```bash
       $ bash scripts/pv_bin.sh validate nonexistent_file.yaml; echo "exit: $?"
       exit: 0
       ```
- **Logic Chain**:
  1. The specification treats `scripts/pv_bin.sh` as a drop-in executable wrapper for the `pv` CLI binary.
  2. Because `scripts/pv_bin.sh` was implemented strictly as a sourcing library, it lacks a CLI argument dispatcher (`"$@"`).
  3. Consequently, any subshell invocation such as `bash scripts/pv_bin.sh <command> <args>` silently exports `$PV` in a subshell, ignores all CLI arguments, and exits 0.
  4. This creates a severe silent-pass gate theater condition—the exact failure mode that finding `F-26` and criterion `C12` were explicitly designed to eradicate.
- **Remediation**:
  1. Add an argument dispatch block at the end of `scripts/pv_bin.sh` (lines 662–665):
     ```bash
     export PV
     if [ "${BASH_SOURCE[0]}" = "$0" ] && [ "$#" -gt 0 ]; then
         exec "$PV" "$@"
     fi
     ```
  2. Mark the script executable: `chmod +x scripts/pv_bin.sh`.
  3. Alternatively, update all spec acceptance and gate commands to use explicit sourcing:
     `. scripts/pv_bin.sh && "$PV" validate contracts/apr-devices-schema-v1.yaml`

---

## 2. Contradiction & Deadlock: Physical Queue Contention and Invariant Inversion on `gx10`

- **Observation**:
  - §2 lines 100–102 (WIP caps):
    `gx10 is one queue, ordered by expiry: master 15 (shakedown) → T-0 → S-3's gx10 leg → master 21 (0.67)`
  - §5 Track I line 198:
    `15 | gx10 shakedown cell, W1, n ≥ 5 interleaved | derived (blocked by 0b, 0c, 1, 6, 7, 12) | first gx10 queue slot`
  - §5 Track I lines 191 & 195:
    Master row 6 (`GET /v1/effective-config`) expires `2026-10-02`.
    Master row 12 (`perf workflow concurrency`) expires `2026-10-02`.
  - §5 G-4 contract invariant (line 433):
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
  7. Furthermore, `queue_pos(15) < queue_pos(T-0)` implies `expires(15) ≤ expires(T-0)`. But `expires(15) ≥ 2026-10-08 > 2026-09-26 = expires(T-0)`. This directly violates G-4's queue-order invariant.
- **Remediation**:
  Re-sequence the physical queue order on `gx10` to match dependency expiries:
  - `T-0` (receipts) and `S-3` (GEMV bench) must occupy `gx10` queue slots 1 and 2 during late September, as their blockers (`T-0h`, `T-1`, `T-2`, `master 1`) clear between September 19 and September 26.
  - `master 15` (shakedown cell) must occupy queue slot 3 in early October, executing after rows 6 and 12 land on October 02.

---

## 3. Dependency Invariant Violations: Four Zero-Slack (0 Days) Blocker Pairs in §5 DAG

- **Observation**:
  - §4 Criterion C10 (line 161), §5 G-4 (line 433), §8 Refusals (line 522), and §9 Toyota Way (line 551) mandate:
    `0 zero-slack blocker pairs (min 6 days)` and `expires(blocker) + 6d ≤ expires(blockee)`.
  - In §5, four blocker pairs exhibit exactly 0 days of slack:
    1. **Master 0b $\to$ R-4**:
       - `master 0b` (sampler pin) expires `2026-09-19` (line 185).
       - `R-4` (W5 CLI wall-clock) lists `blockers: master 0b` and expires `2026-09-19` (line 265). Slack = 0 days.
    2. **P-0.3 $\to$ P-0.6**:
       - `P-0.3` (proof credit from evidence only) expires `2026-09-19` (line 301).
       - `P-0.6` (`pv lint` CI step) lists `blockers: P-0.3` and expires `2026-09-19` (line 302). Slack = 0 days.
    3. **P-1.1 $\to$ P-1.2**:
       - `P-1.1` (`#[contract]` fails closed) expires `2026-10-10` (line 307).
       - `P-1.2` (repoint dangling contracts) lists `blockers: P-1.1` and expires `2026-10-10` (line 308). Slack = 0 days.
    4. **T-1 $\to$ T-0**:
       - `T-1` (τ_loss derived) expires `2026-09-26` (line 356).
       - `T-0` (four WT receipts) lists `blockers: T-0h, T-1, T-2` and expires `2026-09-26` (line 372). Slack = 0 days.
- **Logic Chain**:
  1. The 6-day slack rule prevents cascading schedule collapses when a blocker PR slips by 24 hours.
  2. For `P-1.1` and `P-1.2`, the spec explicitly notes `(same PR as P-1.1: they become compile errors)`. Units landing in the same PR are a single compound ticket, not separate DAG nodes.
  3. For `T-1` and `T-0`, `T-1` derives the loss tolerance threshold $\tau_{\text{loss}}$ on `gx10`, while `T-0` runs the 4 WT receipts. `T-0` cannot evaluate loss validity before `T-1` completes. Co-dating them on September 26 leaves zero margin.
- **Remediation**:
  - Move `R-4` expiry from `2026-09-19` to `2026-09-26` (7 days after 0b).
  - Move `P-0.6` expiry from `2026-09-19` to `2026-09-26`.
  - Merge `P-1.2` into `P-1.1` as a single atomic compound ticket (`P-1.1+1.2`).
  - Move `T-1` earlier to `2026-09-20` (bundled with `T-0h` pilot) or move `T-0` to `2026-10-03` (7 days after `T-1`).

---

## 4. Falsification of Master Spec Status, Row 22 Presence, and Expiries in §10 Verification Ledger

- **Observation**:
  - In §10 line 567: `Master v3.0 §12 has rows 0a–0e, 1–21; no row 22 | [V] | PP-LLAMA-001-MASTER-v3.md read here`
  - In §10 line 568: `Master expiries rows 19/20/21 = 10-16 / 10-23 / 11-06 | [V] | same`
  - In §10 line 579: `PP-LLAMA-001-MASTER.md is committed on main | [U] — review segment 2 asserts it is uncommitted at the explorer's checkout | S0-1`
  - In §3 line 113 (`S0-1`): `grep -n '^| 22' docs/specifications/PP-LLAMA-001-MASTER.md`
- **Empirical Codebase Evidence**:
  - Direct inspection of `docs/specifications/PP-LLAMA-001-MASTER.md` in git (at commit `027ed889d` on `origin/main`) demonstrates:
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
  2. The check in `S0-1` (`grep -n '^| 22'`) failed solely because row 22 is bolded in markdown (`| **22** |`), causing an erroneous false-negative report.
  3. Based on this false-negative, PP-066 falsely claimed row 22 was absent (`[V]`) and accused the parity report of moving expiries on rows 19/21 without an amendment (finding `F-3b`, decision `D-4`). In reality, the master spec itself officially struck those literal dates in v3.0.24.
- **Remediation**:
  - Update §10 line 567: mark row 22 as `[V] CONFIRMED present at line 364 of PP-LLAMA-001-MASTER.md v3.1`.
  - Update §10 line 568: record that rows 19 and 21 have `derived` expiries.
  - Update §10 line 579: mark master commit on `main` as `[V] CONFIRMED`.
  - Fix the `S0-1` regex check: `grep -nE '^\|[[:space:]]*(\*\*)?22(\*\*)?[[:space:]]*\|' docs/specifications/PP-LLAMA-001-MASTER.md`.
  - Withdraw finding `F-3b` and update `D-4` to reflect that rows 19 and 21 derive their dates from blockers.

---

## 5. Inexecutable CLI Arguments and Cross-Repository Coupling in Ticket G-4

- **Observation**:
  - In §5 line 432 (Ticket G-4 acceptance):
    `pmat comply check --rule obligation-dag docs/specifications/pp-066-dag.yaml --min-slack-days 6 exit 0`
  - Running this command in bash produces:
    ```text
    $ pmat comply check --rule obligation-dag docs/specifications/pp-066-dag.yaml --min-slack-days 6
    error: unexpected argument '--rule' found
    Usage: pmat comply check [OPTIONS]
    exit: 2
    ```
- **Logic Chain**:
  1. `pmat comply check` accepts options like `--mode`, `--path`, `--strict`, `--format`, but has no `--rule` or `--min-slack-days` flags and does not accept positional file arguments.
  2. Furthermore, `pmat` is an external binary managed in `paiml/pmat`. Ticket `G-4` is assigned to repository `aprender` (`aprender · feat/g4-dag-invariants`).
  3. An engineer implementing `G-4` within `aprender` cannot modify `pmat`'s CLI parser. Attempting to execute G-4's acceptance command will immediately fail CI with exit code 2.
- **Remediation**:
  Re-anchor the G-4 acceptance command to the dedicated shell/Python validator specified in §4 Criterion C10:
  `scripts/check_dag_invariants.sh docs/specifications/pp-066-dag.yaml --min-slack-days 6 exit 0`
  If a native `pmat comply` check rule is desired, file an upstream issue on `paiml/pmat` and decouple the `aprender` release gate from unreleased `pmat` features.

---

## 6. Non-Existent External Path in Acceptance Commands for R-2 and R-5

- **Observation**:
  - In §5 line 250 (`R-2`) and line 274 (`R-5`):
    `make -C machines/clean-room clean-room-p1 exit 0`
  - In the `aprender` worktree:
    ```bash
    $ ls -d machines/clean-room
    ls: cannot access 'machines/clean-room': No such file or directory
    ```
- **Logic Chain**:
  1. The clean-room runner definitions reside in the external repository `paiml/infra`.
  2. Executing `make -C machines/clean-room clean-room-p1` inside the `aprender` root directory aborts immediately with `No such file or directory`.
- **Remediation**:
  Qualify the path relative to the environment (`make -C ../infra/machines/clean-room clean-room-p1`) or delegate verification to a workflow receipt from the `intel` runner.

---

## 7. Provable Contract Validation Scope Mismatch: `pv validate` vs. JSON Output in R-0

- **Observation**:
  - In §5 line 208 (`R-0` Acceptance):
    `scripts/pv_bin.sh validate contracts/apr-devices-schema-v1.yaml and every apr devices --json output validates against it`
  - `pv validate --help` reveals:
    ```text
    Validate a YAML kernel contract
    Usage: pv validate [OPTIONS] <CONTRACT>
    ```
- **Logic Chain**:
  1. `pv validate` is designed solely to validate that a contract YAML conforms to provable contract schemas.
  2. `pv validate` has no functionality to ingest JSON streams emitted by CLI tools and validate them against JSON Schemas.
  3. The clause `and every apr devices --json output validates against it` is unexecutable human prose embedded within an acceptance test command.
- **Remediation**:
  Split this into two distinct, machine-executable operations:
  `. scripts/pv_bin.sh && "$PV" validate contracts/apr-devices-schema-v1.yaml && cargo test -p apr-cli --test devices_json_schema`

---

## 8. Contract Discipline Gaps: Track P, Ticket G-2, and Ticket G-3

- **Observation**:
  - §5 top (line 175) mandates: "Contract discipline for every card (F-26)... Every PP-066 contract: `kind: pattern`... Every card names its contract".
  - In §5 Track P (lines 295–313), cards `P-0.1` through `P-0.6`, `P-1.1`, and `P-1.2` contain no `contract:` field or contract YAML path.
  - In §5 line 420, `G-2` is an unformatted one-liner lacking acceptance commands, contract specifications, and mutation definitions.
  - In §5 lines 426–428, `G-3` specifies `contract: none` and `quorum: none`, and defines no `mutation → RED` clause.
- **Logic Chain**:
  1. Track P implements `pv` verifier enhancements. While tool-internal improvements can legitimately be verified via test suites, omitting explicit contract metadata violates the universal "every card names its contract" rule unless an explicit exemption is stated.
  2. `G-2` and `G-3` bypass standard card schema formatting. In particular, `quorum: none` violates the `paiml-implement` lifecycle requirement that all tickets require at least `review-only` quorum.
- **Remediation**:
  - Add an explicit policy note to Track P: `contract: self-verifying via FALSIFY-PVI-nnn gates; exempt from pattern-contract requirement`.
  - Format `G-2` into a full card schema with explicit acceptance criteria.
  - Update `G-3` line 427 to specify `quorum: review-only` and add a mutation clause.

---

## 9. Temporal Paradox in Contract Enforcement Between R-0 and P-1.1

- **Observation**:
  - `R-0` is ticket #1 of 0.66, expiring on **2026-09-19** (line 206).
  - `P-1.1` (`#[contract]` fails closed) does not expire until **2026-10-10** (line 307).
  - Line 307 acknowledges: "without it `#[contract]` on any 0.66 function is decoration; after it, R-0's `discover()` can carry `#[contract("apr-backend-registry-v1", …)]` and mean it".
- **Logic Chain**:
  1. When `R-0` merges on September 19, `#[contract]` is decorative.
  2. When `P-1.1` merges on October 10, no ticket in §5 or §6 is scheduled to decorate `R-0::discover()` with `#[contract]`.
  3. Ticket `P-1.2` is scoped exclusively to repointing the 10 existing dangling contract sites.
  4. Consequently, `R-0` will ship in release 0.66 without fail-closed contract enforcement.
- **Remediation**:
  Add an explicit acceptance task in `P-1.2` or mint a follow-up card `R-0b` to decorate `BackendRegistry::discover()` with `#[contract("apr-backend-registry-v1", ...)]` immediately after `P-1.1` merges.

---

## 10. Refusal Exit Code Ambiguity: `FeatureDisabled` (9) vs `NotImplemented` (12)

- **Observation**:
  - §0 D-7 (line 40) and §8 Refusals (line 508) state: `No refusal exit code typed into a test; the constant is read from error.rs (D-7). The tree maps CliError::FeatureDisabled → 9...`
  - In REG-9 (line 225): `--tensor-split is refused with NotImplemented (an owned refusal, not a missing flag)... FX-9: ... --tensor-split 1,1 → refusal code`.
  - In REG-14 (line 230): `--gpu-layers N is accepted only as all ... and otherwise refused with NotImplemented(partial_offload → 0.67 W-D)`.
  - Inspection of `crates/apr-cli/src/error.rs` lines 106 and 112 confirms:
    ```rust
    Self::FeatureDisabled(_) => 9,
    Self::NotImplemented(_) => 12,
    ```
- **Logic Chain**:
  1. `FeatureDisabled` (exit code 9) indicates a backend/feature compiled out or missing from hardware (e.g. `--backend cuda` on a CPU machine).
  2. `NotImplemented` (exit code 12) indicates a parsed CLI flag whose implementation is deferred to 0.67 (e.g. `--tensor-split`).
  3. Citing "refusal code" generically in FX-9 leads test implementers to assert exit code 9, causing tests to fail when the CLI emits exit code 12.
- **Remediation**:
  Explicitly disambiguate the two typed refusal codes in D-7, §8, and REG-9/14:
  - Unavailable backends: `CliError::FeatureDisabled.exit_code_value()` (9).
  - Deferred 0.67 options: `CliError::NotImplemented.exit_code_value()` (12).

---

## 11. Fixture Count Contradiction: Release Gate C11 ("12 fixtures") vs. REG-1..REG-14 ("14 fixtures")

- **Observation**:
  - §4 C11 (line 158): `the failure-catalogue case table (R-0 §REG, 12 fixtures) is green...`
  - §5 R-0 Acceptance (line 208): `cargo test -p apr-cli --test registry_failure_catalogue (FX-1..FX-14 below...)`
  - §5 Requirements table (lines 215–231): Contains **14 requirements** (`REG-1`..`REG-14`).
  - §9 line 539: `14/14 REG requirements with a committed fixture`.
  - Appendix A line 660: `R-0 gains REG-1..REG-14 with a 14-fixture failure catalogue`.
- **Logic Chain**:
  1. When `REG-13` and `REG-14` were added in v1.4, §5, §9, and Appendix A were updated to 14 fixtures.
  2. §4 Criterion `C11` was overlooked, retaining the stale count `12 fixtures`.
- **Remediation**:
  Update §4 C11 line 158 to replace `(R-0 §REG, 12 fixtures)` with `(R-0 §REG, 14 fixtures)`.

---

## 12. Stale Step-0 Metrics in Toyota Way Targets (§9) and §3 Ticket Description

- **Observation**:
  - §9 line 544: `genchi genbutsu | 9/9 Step-0 premises answered by a pasted command output before ticket #1 | S0 ledger`
  - §3 line 109: `pmat work add "PP-066 S0: discovery ledger at HEAD" --description "Falsify the eight premises the 0.66 plan depends on..."`
  - §3 table (lines 111–136): Contains **23 premises** (`S0-1` through `S0-23`).
- **Logic Chain**:
  1. The spec expanded from 9 premises in v1.0 to 23 premises in v1.5 as new edge cases (`S0-10`..`S0-23`) were added.
  2. The target metric in §9 and description in §3 were never updated, permitting 14 premises to remain unverified without triggering a target failure.
- **Remediation**:
  Update §9 line 544 to `23/23 Step-0 premises answered...` and §3 line 109 to `"Falsify the 23 premises the 0.66 plan depends on..."`.

---

## 13. Prior-Art Register: Missing REG-10 and REG-9 References in §12

- **Observation**:
  - In §5 line 226 (`REG-10`):
    `REG-10 | Never mix vendors in one graph... Lesson (§12): mixed AMD+NVIDIA in one llama.cpp process segfaults; Ollama binds to the primary driver and drops the other`
  - In §12 table (lines 648–651), the `REG` column lists:
    - llama.cpp: `1, 2, 5, 6, 13, 14` (omits 10).
    - Ollama: `1, 3, 4, 6, 7, 8, 9, 12` (omits 10).
    - llamafile: `3, 5, 6, 8, 11` (omits 10).
    - all three: `5, 7, 11` (omits 10).
  - Additionally, `REG-9` cites llama.cpp's even device-split in §5 line 225, but llama.cpp omits 9 in §12.
- **Logic Chain**:
  §5 explicitly derives `REG-10` from the cross-vendor failures of llama.cpp and Ollama documented in §12, but §12's cross-reference index omits `10`.
- **Remediation**:
  Add `10` to the REG column for both llama.cpp and Ollama, and add `9` to llama.cpp in §12.

---

## 14. Inaccurate Architectural Claims regarding WASM and Metal Support

- **Observation**:
  - §6 line 464 & §10 line 575 claim: `aprender-serve is #[cfg(not(target_arch = "wasm32"))] entirely [A] — B-S1 is a port of the loader/decoder, not a harness`
  - §6 line 463 claims: `Correction: 13 MSL kernels + MetalBackend via manzana::metal already exist in crates/aprender-gpu/src/backend/metal_shaders.rs [A]`
- **Codebase Verification**:
  1. In `crates/aprender-serve/src/lib.rs` (lines 470, 473), only memory-mapped safetensors loaders (`MappedSafeTensorsModel`, `ShardedSafeTensorsModel`) are excluded from wasm32. In fact, `loading_mmap.rs:67` provides explicit wasm32 handling, and the core inference engine compiles for WASM.
  2. In `crates/aprender-gpu/Cargo.toml`, there is no `metal` feature and no `manzana` dependency. `crates/aprender-gpu/src/backend/metal_shaders.rs` explicitly documents in lines 8–11:
     `These are source strings only. This crate contains no Metal dispatcher, so nothing here compiles or runs them...`
     and `MetalBackend::is_available()` in `src/backend/mod.rs` unconditionally returns false.
- **Logic Chain**:
  Claiming `aprender-serve` is "entirely" excluded from wasm32 misleads engineers into planning a complete rewrite rather than targeting the mmap loader. Claiming `MetalBackend via manzana::metal already exists` misrepresents static shader strings as an operational driver backend.
- **Remediation**:
  Clarify in §6 and §10:
  - `B-S1`: only memory-mapped safetensors loading requires WASM adaptation; core serving is WASM-compatible.
  - `B-M1`: MSL shaders exist only as source strings; host execution requires authoring a driver dispatcher.

---

# Minor Corrections and Typos

1. **Missing Fixture Identifier `FX-13:` in Requirement REG-13**:
   - In §5 line 229, requirement `REG-13` begins directly with prose: `MockBackend in tests/ registers, enumerates two fake devices...`
   - Correction: Prefix the fixture column with `FX-13:` for strict structural uniformity with `FX-1` through `FX-14`.

2. **Ticket Minting Discrepancy and Scope Table Omission for T-2**:
   - In §5 line 363, the ticket is headed `T-2 · --max-seq-len honoured or refused, never clamped`, but the minting command reads `pmat work add "T-5a: apr finetune --max-seq-len..."`.
   - Additionally, `T-2` is omitted from the 0.66 scope table in §2 line 96.
   - Correction: Update the minting command to `"T-2: ..."` and add `T-2` to the §2 Scope table.

3. **Dual Namespace Collisions (`D-1` and `T-1`)**:
   - `D-1` designates Decision 1 in §0 (Scope split), but also designates Track D Ticket 1 in §5 (`cuda-backend-architecture.md`).
   - `T-1` designates Track T Ticket 1 in §5 (τ_loss instrument), but also designates 0.67 Training Levers in §6 (`T-1..T-5`).
   - Correction: Rename Documentation Ticket `D-1` to `DOC-1`, and rename 0.66 τ_loss ticket `T-1` to `T-1a` (or 0.67 levers to `TL-1..TL-5`).

4. **Stale Mutation Count in Registered Prediction G-1**:
   - In §7 line 489, the prediction reads: `both mutations RED in the guard's PR | either GREEN`.
   - However, §5 line 417 defines **three** mutations: `(i) add an untracked Cargo.toml... (ii) delete one allow-list line... (iii) add to workspace.members...`.
   - Correction: Update §7 line 489 to: `all three mutations RED in the guard's PR | any GREEN`.

5. **Track G Scope Truncation in Partition Metadata**:
   - The analysis partition metadata cites `Track G G-1..G-3`, omitting `G-4 · the obligation DAG as data, with invariants in CI` (lines 430–436).
   - Correction: Formally record that Track G comprises G-1, G-2, G-3, and G-4.

6. **Stale Line Citations in §10 Verification Ledger**:
   - `error.rs:86-90,218`: Line 218 is in a test; the actual enum match mapping `CliError::FeatureDisabled` to exit code 9 is at line 106.
   - `root Cargo.toml:509`: Line 509 is `identity_op = "allow"`. The `aprender_ml` alias is at line 666.
   - `handlers.rs:912-953`: Qualify crate path as `crates/apr-cli/src/commands/serve/handlers.rs` (not `aprender-serve`).
   - `attention.rs:995,1022`: Qualify crate path as `crates/aprender-serve/src/cuda/executor/layers/cublas_prefill/attention.rs`.

7. **Clarification of `finetune.rs:317, 718` Clamp Citations**:
   - In §5 line 363 (T-2) and §10 line 575: line 317 is inside a doc comment describing the defect; line 717 is the active executable hardcoded 512 clamp on the wgpu path (`512, // max_seq_len`).
   - Correction: Note that line 317 is documentation and line 717 is the executable clamp site.

8. **Markdown Syntax: Unbalanced Backtick in R-5 Contract Invariant**:
   - In §5 line 275: `prerelease=false ⇒ ∃ 4 host receipts with `asset_sha256 == manifest[target]`;`
   - Correction: Fix the mismatched backtick: `prerelease=false ⇒ ∃ 4 host receipts with \`asset_sha256 == manifest[target]\`;`.

9. **Ambiguous Slack Pair Notation in §10 Line 573**:
   - Line 573: `Report §11 expiry slack ≥ 6 days on the DAG as dated | [C] | pairs 31→35, 14/15→18, 22→26, 20→22`.
   - Issue: `20→22` denotes only 2 days difference (or master rows 20 and 22 where row 20 expires Oct 23 and row 22 expires Oct 15), conflicting with $\ge 6$ days slack.
   - Correction: Disambiguate notation and reconcile with the $\ge 6$ days rule.

10. **Disambiguation of `Features::SUBGROUPS` in B-W0**:
    - In §6 line 461: `Features::SUBGROUPS requested and the grant/denial recorded in the receipt`.
    - Correction: Disambiguate by qualifying as `wgpu::Features::SUBGROUPS`.

11. **Unexecutable Prose in Acceptance Fields for R-5 and D-1**:
    - In §5 line 274 (`R-5`): Remove human policy clauses (`...only after all four receipts are green; the promotion step is a workflow job that reads the receipts, not a hand command`) and provide a runnable command (`scripts/promote_release.sh --tag v0.66.0-rc1 --require-receipts lambda,gx10,intel,mini`).
    - In §5 line 442 (`D-1`): Replace conversational preamble (`the existing claims-cite check ... extended to API: sentences rather than a new script...`) with the exact verification invocation (`scripts/check_doc_citations.sh docs/specifications/cuda-backend-architecture.md`).

12. **Stale Master Spec File Name in §10 Line 567**:
    - Line 567 references `PP-LLAMA-001-MASTER-v3.md`.
    - Correction: Update to the actual in-tree file name: `PP-LLAMA-001-MASTER.md`.

13. **Script Name Inconsistency in §10 Line 580**:
    - Line 580 cites `check_backend_refusal_surfaces.sh`.
    - Correction: Harmonize with §4 C11 and §5 R-0 by renaming to `check_backend_registry.sh`.

14. **Internal Review Reference Leak in Appendix A Line 659**:
    - In the v1.5 changelog: `MIN-03, seg-2 2.4, §12 Ollama`.
    - Correction: Replace raw review handle `seg-2 2.4` with its formal section name (`§6 T-lane NF4 kernel design`).
