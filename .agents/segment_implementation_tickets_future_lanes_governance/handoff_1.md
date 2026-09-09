# Summary

This report delivers an exhaustive, independent audit of **Segment 2: `implementation_tickets_future_lanes_governance`** of `docs/specifications/PP-066-release-spec.md` (v1.5, 2026-09-05). Segment 2 spans lines 167–664, encompassing:
- **§5 0.66 tickets — paiml-implement units** (Track I master rows; Track R: R-0..R-7; Track P: P-0.1..P-0.6, P-1.1, P-1.2; Track S: S-1..S-3; Track T: T-0h, T-1, T-2, T-0, T-3; Track B: B-A1, B-G1; Track G: G-1..G-4; Track D: D-1).
- **§6 0.67 lane — carried rows** (W-B kernel, W-D, W-E, W-G, W-H, W-F attach, T-1..T-5, T-7 arm, B-W0, B-W1..W5, B-M1..M4, B-S1..S4, B-A2, PVI 1.3–4.3, §6.7 renames, bar gating).
- **§7 Registered predictions** (16 entries across performance, memory, tooling, and discovery).
- **§8 Refusals (0.66)** (24 invariant refusal bullets).
- **§9 Toyota Way targets (0.66)** (25 principles mapped to concrete targets and instruments).
- **§10 Verification ledger for this document** (17 claims mapped to `[A]`, `[C]`, `[V]`, and `[U]` marks).
- **§11 Adjudication of `0_66-review.md`** (34 review findings adjudicated as SUSTAINED, NARROWED, REJECTED, or INDETERMINATE, plus five unraised findings).
- **§12 Prior-art register — GPU discovery in three shipped systems** (llama.cpp/ggml, Ollama, llamafile, and common lessons mapped to REG requirements).
- **Appendix A — Changelog** (versions 1.0 through 1.5).

### Evaluation Summary Across Mandatory Dimensions

1. **Spec Compliance**:
   - **Contract Discipline**: Contract-per-card discipline is maintained across almost all implementation units in §5 (e.g. `contracts/apr-backend-registry-v1.yaml`, `contracts/apr-perf-wa-v1.yaml`), requiring `kind: pattern`, no `registry: true`, executable falsification tests, and `pv` verification. However, Track P (P-0.1..P-0.6, P-1.1, P-1.2) lacks explicit contract paths, and doc/decision tickets (R-7, G-2, G-3) legitimately rely on guards rather than contracts.
   - **Critical Execution Defect Identified**: The spec mandates invoking `scripts/pv_bin.sh validate ...` and `scripts/pv_bin.sh lint ...` directly. In the actual tree, `scripts/pv_bin.sh` has permissions `-rw-rw-r--` (non-executable) and is explicitly designed to be sourced (`. scripts/pv_bin.sh || exit 1`), containing no CLI argument dispatch block. When executed via `bash scripts/pv_bin.sh <args>`, it exports `$PV` in a subshell, ignores all arguments, and silently exits 0—creating an accidental gate-theater vulnerability.
   - **Physical Queue Contention & Invariant Inversion**: The `gx10` queue order (`master 15 → T-0 → S-3`) directly contradicts dependency dates: master 15 is blocked by rows 6 and 12 (expiring 2026-10-02, thus row 15 cannot complete before 2026-10-08), while T-0 and S-3 expire on 2026-09-26. Under WIP=1, this starves T-0 and S-3, violating G-4's invariant `queue_pos(a) < queue_pos(b) ⇒ expires(a) ≤ expires(b)`.
   - **Zero-Slack Blocker Pairs**: Four dependency pairs in §5 have 0 days of slack (master 0b → R-4, P-0.3 → P-0.6, P-1.1 → P-1.2, T-1 → T-0), violating the mandatory 6-day slack rule (`expires(blocker) + 6d ≤ expires(blockee)`).
   - **Quorum Classifications**: Properly specified throughout §5 (teamwork for policy/spec changes, N-lane for root-cause/design changes, review-only for bounded code/guard diffs).
   - **Refusals and Adjudications**: All 24 refusals in §8 are rigorous and aligned with the architectural decisions. §11 adjudicates 34 findings with strict adherence to the four permitted verdicts (`SUSTAINED`, `NARROWED`, `REJECTED`, `INDETERMINATE`).

2. **Grammar and Clarity**:
   - Technical prose is remarkably dense, precise, and uncompromising in its systems-engineering and statistical terminology.
   - The discovery architecture (REG-1 through REG-14) is exceptionally well-articulated, anchoring every architectural requirement to concrete failures in shipped systems (llama.cpp, Ollama, llamafile) and dedicated falsification fixtures.
   - Minor clarity issues include a discrepancy between the 12 fixtures cited in C11 and the 14 fixtures in REG-1..REG-14, an omitted REG-10 cross-reference in §12, and minor typographical inconsistencies in markdown table formatting and backtick delimiters.

3. **Code Metrics and Status**:
   - **Live Repository Verification**: All key codebase claims were verified against HEAD:
     - `crates/apr-cli/src/commands_enum.rs:10` defines `BACKEND_VALUES = ["cuda", "cpu", "wgpu"]` (3 values).
     - `crates/apr-cli/src/error.rs:106` maps `FeatureDisabled` to exit code `9`.
     - `crates/aprender-gpu/src/backend/metal_shaders.rs` contains exactly 13 MSL compute kernels.
     - `Q4K_GEMV_SHADER` exists at `crates/aprender-compute/src/backends/gpu/shaders/basic_ops.rs:555`.
     - `OwnedQuantizedModelWgpu` exists at `crates/aprender-serve/src/gguf/wgpu_backend/mod.rs:197` and returns `UnsupportedOperation`.
     - In-tree use-site counts for the 5 key crates match the spec's figures to the exact digit: `realizar` = 1,428, `presentar_core` = 355, `trueno_gpu` = 295, `provable_contracts` = 291, `trueno` = 218.
     - `scripts/contract_test_binding_baseline.txt` contains exactly 13 lines.
     - `PP-LLAMA-001-MASTER.md` is committed on `main` at version 3.1.
   - **Arithmetic Checks**: Every statistical and physical formula was independently recalculated:
     - KV cache: $28 \times 4 \times 128 \times 2 \times 4096 \times 4\text{ B} = 469,762,048\text{ B} = 448\text{ MiB}$ (exact).
     - Activation memory: $7 \times 28 \times 4 \times 2048 \times 3584 \times 4\text{ B} = 23,018,340,352\text{ B} \approx 23.02\text{ GB}$ (exact).
     - W5 sample size formula: $n = \lceil(1.96 \cdot \text{CV} / 0.05)^2\rceil$ (statistically exact for 95% CI with 5% relative half-width).
     - Ratios $2101 / 16118 = 0.13\times$, $484.7 / 171.5 = 2.83\times$, $5700 / 10399 = 0.55\times$, and prefill copy/alloc counts ($1018 / 8 \approx 127$, $904 / 8 = 113$) are completely accurate.

---

# Potential Mistakes and Improvements

## 1. Critical Execution Defect: `scripts/pv_bin.sh` Non-Executable and Argument-Ignoring Behavior

- **Observation**:
  - In §4 C12 (line 159), acceptance command:
    `scripts/pv_bin.sh lint contracts/ --binding contracts/aprender/binding.yaml --strict-test-binding exit 0`
  - In §5 R-0 Acceptance (line 208):
    `scripts/pv_bin.sh validate contracts/apr-devices-schema-v1.yaml`
  - In §8 Refusals (line 510):
    `No pv invocation against the binary on PATH; scripts/pv_bin.sh builds it from the tree.`
  - Direct file inspection of `scripts/pv_bin.sh` in the repository reveals:
    1. File mode is `-rw-rw-r--` (not executable; executing `./scripts/pv_bin.sh` emits `bash: ./scripts/pv_bin.sh: Permission denied`).
    2. Header lines 3–5 explicitly document:
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
    4. Running `bash scripts/pv_bin.sh validate contracts/apr-devices-schema-v1.yaml` exits with code 0 immediately without executing `$PV validate ...` or checking any arguments.
- **Logic Chain**:
  1. The spec treats `scripts/pv_bin.sh` as a drop-in executable replacement for the `pv` CLI binary.
  2. Because `scripts/pv_bin.sh` was written as an option-neutral library to be sourced into caller shells, it does not forward CLI arguments (`"$@"`) to the resolved binary `$PV`.
  3. Consequently, if CI or a developer runs `bash scripts/pv_bin.sh validate contracts/some-contract.yaml`, the command exits 0 without validating anything.
  4. This introduces a silent-pass defect (gate theater), precisely the failure mode that F-26 and C12 were minted to eliminate.
- **Caveats**:
  - Developers using an interactive subshell who manually run `. scripts/pv_bin.sh` followed by `"$PV" validate ...` are unaffected, but all automated acceptance commands ($A_i$) written in the spec invoke `scripts/pv_bin.sh <subcommand>`.
- **Conclusion**:
  - `scripts/pv_bin.sh` must be made executable (`chmod +x`), and must include a dispatch wrapper at the bottom:
    ```bash
    if [ "$#" -gt 0 ]; then
        exec "$PV" "$@"
    fi
    ```
  - Alternatively, all acceptance commands in §4 and §5 should be updated to `. scripts/pv_bin.sh && "$PV" <subcommand> <args>`.
- **Verification Method**:
  Run `bash scripts/pv_bin.sh validate non_existent_file.yaml`. Currently it exits 0 silently. With the fix, it must invoke `$PV` and fail loudly on invalid/missing arguments.

## 2. Contradiction: `gx10` Queue Scheduling Contention and Invariant Inversion

- **Observation**:
  - §2 lines 100–102:
    `gx10 is one queue, ordered by expiry: master 15 (shakedown) → T-0 → S-3's gx10 leg → master 21 (0.67)`
  - §5 Track I line 198:
    `15 | gx10 shakedown cell, W1, n ≥ 5 interleaved | derived (blocked by 0b, 0c, 1, 6, 7, 12) | first gx10 queue slot`
  - §5 Track I lines 191 & 195:
    Master row 6 (`GET /v1/effective-config`) expires `2026-10-02`.
    Master row 12 (`perf workflow concurrency`) expires `2026-10-02`.
  - §5 G-4 contract invariant (line 433):
    `invariants: acyclic; expires(blocker) + 6d ≤ expires(blockee); queue_pos(host, a) < queue_pos(host, b) ⇒ expires(a) ≤ expires(b)`
  - §5 Track T line 372:
    `T-0 · the four WT receipts ... host: lambda, gx10 (queue slot 2) · expiry 2026-09-26`
  - §5 Track S line 336:
    `S-3 · W-B bench ... host: lambda, then gx10 (queue slot 3) · expiry 2026-09-26`
- **Logic Chain**:
  1. Master row 15 is blocked by row 6 and row 12, both expiring on `2026-10-02`.
  2. Under the minimum slack invariant (`expires(blockee) ≥ expires(blocker) + 6d`), row 15 cannot expire before `2026-10-08`.
  3. Row 15 is assigned queue slot 1 on `gx10`.
  4. Under §2's WIP limit of 1 speed row per host (`WIP=1`), queue slot 2 (T-0) cannot execute until slot 1 completes.
  5. Therefore, T-0 cannot execute until after `2026-10-08`.
  6. However, T-0's mandated expiry is `2026-09-26`—12 days before row 15 can even run.
  7. Furthermore, `queue_pos(15) < queue_pos(T-0)` implies `expires(15) ≤ expires(T-0)`. But `expires(15) ≥ 2026-10-08 > 2026-09-26 = expires(T-0)`. This directly violates G-4's queue-order invariant.
- **Caveats**:
  - If T-0's `gx10` leg does not run inference and is classified purely as training, one might attempt to argue that the WIP limit applies only to inference. However, §2 explicitly unifies `gx10` into a single physical queue: "gx10 is one queue, ordered by expiry: master 15 (shakedown) → T-0 → S-3's gx10 leg".
- **Conclusion**:
  - The physical queue ordering on `gx10` must be reconciled with expiries. Because T-0 and S-3 expire on September 26, and their blockers (T-0h, T-1, T-2, master 1) clear by September 19–26, T-0 and S-3 must occupy queue slots 1 and 2 on `gx10` during late September, with master row 15 running in queue slot 3 in early October after rows 6 and 12 land.
- **Verification Method**:
  Run `scripts/check_dag_invariants.sh docs/specifications/pp-066-dag.yaml --min-slack-days 6` once G-4 is implemented.

## 3. Dependency Invariant Violations: Zero-Slack Blocker Pairs

- **Observation**:
  - §4 C10 (line 161) and §5 G-4 (line 433) require:
    `expires(blocker) + 6d ≤ expires(blockee)`.
  - In §5, the following blocker pairs violate this invariant:
    1. **Master 0b → R-4**:
       Master row 0b expires `2026-09-19` (line 185).
       R-4 lists `blockers: master 0b (sampler pin)` and expires `2026-09-19` (line 265). Slack = 0 days.
    2. **P-0.3 → P-0.6**:
       P-0.3 expires `2026-09-19` (line 301).
       P-0.6 lists `blockers: P-0.3` and expires `2026-09-19` (line 302). Slack = 0 days.
    3. **P-1.1 → P-1.2**:
       P-1.1 expires `2026-10-10` (line 307).
       P-1.2 lists `blockers: P-1.1` and expires `2026-10-10` (line 308). Slack = 0 days.
    4. **T-1 → T-0**:
       T-1 expires `2026-09-26` (line 356).
       T-0 lists `blockers: T-0h, T-1, T-2` and expires `2026-09-26` (line 372). Slack = 0 days.
- **Logic Chain**:
  1. The 6-day slack invariant exists to prevent waterfall cascading delays where one ticket slipping by 24 hours invalidates downstream commitments.
  2. For P-1.1 and P-1.2, the spec notes "(same PR as P-1.1: they become compile errors)". If two deliverables land in the same PR, they are a single unit of work (ticket), not a blocker DAG edge.
  3. For T-1 and T-0, T-1 derives $\tau_{\text{loss}}$ on `gx10` and T-0 runs the 4 WT receipts on `gx10`. T-0 cannot ingest $\tau_{\text{loss}}$ before T-1 completes, so co-dating them on 2026-09-26 leaves 0 hours of margin.
- **Caveats**:
  - If P-0.3/P-0.6 and T-1/T-0 are intended as paired sprint deliverables, G-4's check script will fail unless an explicit `co_dated: true` exemption is specified or expiries are staggered by 6 days.
- **Conclusion**:
  - Adjust expiries or document explicit compound tickets. For example, R-4 should expire on `2026-09-26` (7 days after 0b); P-0.6 should expire on `2026-09-26`; P-1.2 should be folded directly into P-1.1 as a single atomic PR ticket; T-1 should expire on `2026-09-20` or T-0 should expire on `2026-10-03`.
- **Verification Method**:
  Verify against `pmat comply check --rule obligation-dag` or inspection of `pp-066-dag.yaml`.

## 4. Stale Counts: Step-0 Premises in §9 and §3

- **Observation**:
  - §9 line 544 (Toyota Way targets):
    `genchi genbutsu | 9/9 Step-0 premises answered by a pasted command output before ticket #1 | S0 ledger`
  - §3 line 109:
    `pmat work add "PP-066 S0: discovery ledger at HEAD" --description "Falsify the eight premises the 0.66 plan depends on..."`
  - §3 table lines 111–136:
    Premises are numbered **S0-1 through S0-23** (23 distinct premises).
- **Logic Chain**:
  1. In v1.0, there were 9 premises (S0-1..S0-9).
  2. Through subsequent iterations (v1.1 through v1.5), premises S0-10 through S0-23 were successively added to address findings F-25, F-26, F-27, F-28, etc.
  3. The target in §9 was never updated from `9/9` to `23/23`, and the ticket description in §3 was never updated from `eight premises` to `23 premises`.
- **Conclusion**:
  - Update §9 line 544 to `23/23 Step-0 premises answered...`.
  - Update §3 line 109 to `"Falsify the 23 premises the 0.66 plan depends on..."`.
- **Verification Method**:
  `grep -E 'S0-[0-9]+' docs/specifications/PP-066-release-spec.md | wc -l` yields 23 premise definition rows.

## 5. Inconsistency: Failure Catalogue Fixture Count (12 vs 14)

- **Observation**:
  - §4 C11 (line 158):
    `the failure-catalogue case table (R-0 §REG, 12 fixtures) is green and each fixture has been observed RED once`
  - §5 R-0 Acceptance (line 208):
    `cargo test -p apr-cli --test registry_failure_catalogue (FX-1..FX-14 below; each fixture has a must-RED twin committed under tests/fixtures/registry/defective/)`
  - §5 Requirements table (lines 215–231):
    Contains REG-1 through REG-14, mapping to FX-1 through FX-14 (plus FX-4b).
  - §9 line 539:
    `14/14 REG requirements with a committed fixture`
  - Appendix A line 660 (Changelog v1.4):
    `R-0 gains REG-1..REG-14 with a 14-fixture failure catalogue`
- **Logic Chain**:
  1. In earlier draft v1.2, R-0 had 12 requirements.
  2. In v1.4, REG-13 (`MockBackend`) and REG-14 (partial offload refusal) were added, bringing the total to 14.
  3. §4 C11 retained the stale count `12 fixtures`, while §5, §9, and Appendix A were updated to 14.
- **Conclusion**:
  - Fix §4 C11 line 158: replace `(R-0 §REG, 12 fixtures)` with `(R-0 §REG, 14 fixtures)`.
- **Verification Method**:
  Grep `fixtures` in `docs/specifications/PP-066-release-spec.md` and check consistency with REG table rows.

## 6. Scope Truncation: Track G Range in Analysis Partition

- **Observation**:
  - The dispatch instructions define the scope as: `Track G G-1..G-3`.
  - §5 lines 410–436 contain **four** tickets in Track G:
    - `G-1 · crate-name guard, allow-list full`
    - `G-2 · D-5 decision ticket`
    - `G-3 · rename cost, measured`
    - `G-4 · the obligation DAG as data, with invariants in CI`
- **Logic Chain**:
  1. G-4 is one of the most vital governance mechanisms in 0.66, responsible for enforcing the DAG invariants (0 cycles, min 6-day slack, queue order) and generating §5/§6 tables.
  2. G-4 directly closes review findings F-10, MAJ-03, and MAJ-04.
  3. The partition metadata string `Track G G-1..G-3` is an off-by-one truncation that omits G-4 from the explicit list.
- **Conclusion**:
  - Formally record that Track G comprises G-1, G-2, G-3, and G-4. All four cards were reviewed and verified in this audit.

## 7. Structural Clarity: Track T Ticket Renumbering and Phantom `T-4`

- **Observation**:
  - Dispatch prompt lists `Track T T-0..T-4`.
  - §5 Track T contains cards:
    - `T-0h · training-parity harness` (line 344)
    - `T-1 · τ_loss derived, not declared` (line 354)
    - `T-2 · --max-seq-len honoured or refused, never clamped` (line 362)
    - `T-0 · the four WT receipts` (line 370)
    - `T-3 · training gate, REPORTING + self-ratchet` (line 378)
  - In `pmat work add` commands:
    - T-2 is minted as `"T-5a: apr finetune --max-seq-len is honoured..."` (branch `fix/t5a-max-seq-len-honoured`).
    - T-3 is minted as `"T-7 (REPORTING): train_tok_per_sec..."` (branch `feat/t7-train-gate-reporting`).
  - There is no card named `T-4` in §5. (T-4 from the predecessor report was grouped into 0.67 in §6 under `T-1..T-5`).
  - Moreover, `T-0` appears after `T-2` in the document text, despite `T-0h` appearing before `T-1`.
- **Logic Chain**:
  1. The historical tickets T-0 through T-7 were re-mapped during the 0.66/0.67 split:
     - T-6 moved to Track R as `R-3`.
     - T-5 was split into `T-2` (the clamp bug fix) and 0.67 `T-1..T-5` (the kernel optimization levers).
     - T-7 became `T-3`.
     - T-0 became `T-0h` (harness) and `T-0` (receipts).
  2. Leaving the ticket label as `T-2` while the `pmat work add` string states `T-5a` causes discrepancy in work tracking and audits.
  3. Having `T-0` listed after `T-2` rather than immediately after `T-0h` creates cognitive friction regarding sequencing.
- **Conclusion**:
  - Clarify the mapping in the Track T preamble: state explicitly that 0.66 Track T comprises {T-0h, T-1, T-2, T-0, T-3}, that T-4 is deferred to 0.67, and harmonize `pmat work add` strings to match the card handles (e.g. `"T-2 (formerly T-5a): ..."` and `"T-3 (formerly T-7): ..."`).

## 8. Missing Contract Definitions for Track P Cards

- **Observation**:
  - In §5 Top (line 175):
    "Contract discipline for every card (F-26)... Every PP-066 contract: `kind: pattern`... Every card names its contract".
  - In §5 Track P (lines 295–313):
    Cards P-0.1, P-0.2, P-0.3, P-0.4, P-0.5, P-0.6, P-1.1, P-1.2 are presented in a compact table specifying deliverables, gates, blockers, and expiries.
    None of these rows lists a `contract:` field or a path under `contracts/`.
- **Logic Chain**:
  1. Track P implements PV-IMPROVE-001 (phases 0 and 1.1/1.2), which modifies the `pv` tool itself and refactors existing contracts across the repository.
  2. If Track P cards are self-exempt from the "contract in the same PR" rule because they build the verifier that validates contracts, the spec should explicitly state: `contract: self-verifying via FALSIFY-PVI-nnn gates; exempt from contract-per-card discipline`.
  3. Without this clarification, a strict CI conformance checker enforcing C12 or checking for a `contract:` field in every card will flag Track P as non-compliant.
- **Conclusion**:
  - Add an explicit contract policy note to Track P stating how each P card satisfies C12 (e.g., whether it updates `contracts/provable-contracts/*.yaml` or relies exclusively on `FALSIFY-PVI-*` unit test fixtures).

## 9. Ambiguity in Refusal Exit Code: `FeatureDisabled` (9) vs `NotImplemented` (12)

- **Observation**:
  - §0 D-7 (line 40) and §8 Refusals (line 508) state:
    `No refusal exit code typed into a test; the constant is read from error.rs (D-7).`
    `The tree maps CliError::FeatureDisabled → 9... Keep 9 and have every test read the constant from error.rs`
  - In REG-9 (line 225):
    `--tensor-split is refused with NotImplemented (an owned refusal, not a missing flag)`
    `FX-9: fixture with two cuda stub devices... --tensor-split 1,1 → refusal code`
  - In REG-14 (line 230):
    `--gpu-layers N is accepted only as all (N ≥ layers) and otherwise refused with NotImplemented(partial_offload → 0.67 W-D)`
  - Direct inspection of `crates/apr-cli/src/error.rs` (lines 106 and 112) confirms:
    ```rust
    Self::FeatureDisabled(_) => 9,
    Self::NotImplemented(_) => 12,
    ```
- **Logic Chain**:
  1. `FeatureDisabled` (exit 9) signifies that a feature exists in the project but is disabled in the current build or hardware configuration (e.g. `--backend cuda` on a CPU-only build).
  2. `NotImplemented` (exit 12) signifies that the CLI flag or option is parsed but the capability is not yet implemented (e.g. `--tensor-split` deferred to 0.67).
  3. REG-9 and REG-14 specify that these conditions return `NotImplemented`, but FX-9 asserts `→ refusal code`.
  4. If a test author assumes "refusal code" means `CliError::FeatureDisabled.exit_code_value()` (9), the test asserting on `--tensor-split` (which returns 12) will fail.
- **Conclusion**:
  - Clarify in D-7, §8, and REG-9/REG-14 that 0.66 has two distinct, typed refusal exit codes: `CliError::FeatureDisabled.exit_code_value()` (9) for unavailable backends, and `CliError::NotImplemented.exit_code_value()` (12) for deferred 0.67 capabilities.

## 10. Prior-Art Register: Omission of REG-10 in §12

- **Observation**:
  - In §5 R-0 requirements table (line 226):
    `REG-10 | Never mix vendors in one graph... Lesson (§12): mixed AMD+NVIDIA in one llama.cpp process segfaults; Ollama binds to the primary driver and drops the other`
  - In §12 Prior-art register table (lines 648–651):
    - llama.cpp lists REG: `1, 2, 5, 6, 13, 14` (omits 10).
    - Ollama lists REG: `1, 3, 4, 6, 7, 8, 9, 12` (omits 10).
    - llamafile lists REG: `3, 5, 6, 8, 11` (omits 10).
    - all three lists REG: `5, 7, 11` (omits 10).
- **Logic Chain**:
  1. REG-10 explicitly cites the cross-vendor behavior of llama.cpp and Ollama documented in §12.
  2. However, the `REG` column in §12 fails to reference `10` in either the llama.cpp or Ollama row.
- **Conclusion**:
  - Add `10` to the `REG` column for both llama.cpp and Ollama in §12 table.

---

# Minor Corrections and Typos

1. **Stale Master File Name in §10**:
   - Line 567: `PP-LLAMA-001-MASTER-v3.md read here`.
   - Correction: The actual committed file is `docs/specifications/PP-LLAMA-001-MASTER.md`. Change `PP-LLAMA-001-MASTER-v3.md` to `PP-LLAMA-001-MASTER.md`.

2. **Inconsistent Script Name in §10**:
   - Line 580: `check_backend_refusal_surfaces.sh`.
   - Correction: In §4 C11 (line 158) and §5 R-0 (line 208), the script is named `scripts/check_backend_registry.sh`. Update line 580 to use `check_backend_registry.sh`.

3. **Stale Line Number for `aprender_ml` in §10**:
   - Line 575: `aprender_ml alias at root Cargo.toml:509`.
   - Live audit: At HEAD, line 509 of root `Cargo.toml` is `identity_op = "allow"`, whereas `aprender_ml` is declared at line 666 (`aprender_ml = { path = "crates/aprender-core", ... }`).
   - Correction: Update citation to `root Cargo.toml:666 (formerly :509 [A])`.

4. **Clarification of `finetune.rs` Line Numbers**:
   - Line 363 (T-2) and line 575 cite `finetune.rs:317,718 hardcode 512 and drop the CLI value`.
   - Live audit: Line 317 is inside the doc comment describing the historical defect on the instruct path (`max_seq_len` is already threaded at line 334). The actual hardcoded 512 in executable code is at line 717 on the wgpu path (`512, // max_seq_len`).
   - Correction: Note that line 317 is the doc comment / historical site and line 717 is the active wgpu clamp.

5. **Markdown Syntax: Unbalanced Backtick in R-5**:
   - Line 275: `prerelease=false ⇒ ∃ 4 host receipts with `asset_sha256 == manifest[target]`;`
   - Correction: Fix the mismatched leading backtick: `prerelease=false ⇒ ∃ 4 host receipts with \`asset_sha256 == manifest[target]\`;`.

6. **Ambiguous Slack Pair Notation in §10**:
   - Line 573: `Report §11 expiry slack ≥ 6 days on the DAG as dated | [C] | pairs 31→35, 14/15→18, 22→26, 20→22`.
   - Issue: `20→22` denotes only 2 days of difference (or master rows 20 and 22), conflicting with the claim of $\ge 6$ days.
   - Correction: Clarify whether `20→22` refers to calendar dates or master row numbers, and reconcile with the $\ge 6$ days claim.

7. **Qualification of `Features::SUBGROUPS` in B-W0**:
   - Line 461: `Features::SUBGROUPS requested and the grant/denial recorded in the receipt`.
   - Correction: Disambiguate by qualifying as `wgpu::Features::SUBGROUPS` to distinguish from general compute feature flags.

8. **Clarification of `aprender-serve` wasm32 Scope**:
   - Line 464 and Line 575 state: `aprender-serve is #[cfg(not(target_arch = "wasm32"))] entirely [A]`.
   - Live audit: `crates/aprender-serve/src/lib.rs` has no crate-level `#![cfg(not(target_arch = "wasm32"))]`; instead, wasm32 exclusions are placed on specific modules (such as memory-mapped loading in `safetensors` and `apr/loading_mmap.rs`). In fact, `loading_mmap.rs:67` contains an explicit `#[cfg(target_arch = "wasm32")]` fallback branch.
   - Correction: Change "entirely" to "in its model-loading and mmap pipeline".
