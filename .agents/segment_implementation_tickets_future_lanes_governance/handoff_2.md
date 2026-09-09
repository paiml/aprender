# Summary

This report delivers the Candidate 2 evaluation of **Segment 2: `implementation_tickets_future_lanes_governance`** of `PP-066-release-spec.md` (lines 167–665), encompassing:
- **§5 0.66 Tickets** (Track I, Track R R-0..R-7, Track P P-0.1..P-1.2, Track S S-1..S-3, Track T T-0h..T-3, Track B B-A1/B-G1, Track G G-1..G-4, Track D D-1)
- **§6 0.67 Lane** (carried rows, B-W0, B-M1..M4, B-S1..S4, §6.7 renames)
- **§7 Registered Predictions**
- **§8 Refusals (0.66)**
- **§9 Toyota Way Targets (0.66)**
- **§10 Verification Ledger**
- **§11 Adjudication of `0_66-review.md`**
- **§12 Prior-Art Register — GPU Discovery in Three Shipped Systems**
- **Appendix A — Changelog**

The evaluation audits the segment across three mandatory dimensions:
1. **Spec Compliance**: Contract-per-card discipline (enforcing `kind: pattern`, absence of `registry: true`, executable falsification tests, and tree-built verifier usage), acceptance commands ($A_i$), mutation-to-RED discriminators, quorum classifications, dependency DAG ordering and slack invariants, registered prediction protocols (prediction-kill vs. track-kill), refusal invariants, Toyota Way targets (jidoka, poka-yoke, andon, genchi genbutsu, kaizen, heijunka), verification ledger marks (`[V]`, `[C]`, `[A]`, `[U]`), adjudication verdicts, and prior-art comparisons.
2. **Grammar and Clarity**: Technical readability, clarity of justifications, terminology, precision of architectural models (such as `BackendRegistry` discovery, `REG-1..REG-14` requirements, and failure catalogues), and table formatting.
3. **Code Metrics and Status**: Empirical verification of repository paths, crate names, manifest entries, contract schemas, script behaviors, mathematical computations, and claims against git HEAD (`d6c6c6f8` / `587ad0797`).

### Executive Assessment
Segment 2 is an exceptionally thorough, mathematically grounded, and structurally disciplined specification. It systematically transforms an audit report into actionable `paiml-implement` units, replaces configuration illusions with a genuine architectural registry (`R-0`), establishes an end-to-end release verification pipeline (`R-5`, `R-6`, `R-7`), and institutes strict statistical and contract rigor. 

However, this audit has uncovered **four critical execution defects** that would compromise CI enforcement or cause immediate self-contradiction if implemented verbatim:
1. **Vacuous Acceptance Commands via `scripts/pv_bin.sh`**: `pv_bin.sh` is designed solely to be sourced; when executed directly as prescribed in §5 acceptance commands and §4 C12 (e.g. `scripts/pv_bin.sh validate ...` or `scripts/pv_bin.sh lint ...`), it drops all arguments and exits 0 without running `pv`, creating a silent-pass bypass.
2. **Internal Contradiction in DAG Slack Rules**: §4 C10 and §5 G-4 mandate a strict minimum 6-day slack between any blocker and blockee (`expires(blocker) + 6d <= expires(blockee)`). Yet the spec itself contains at least **four 0-day slack blocker pairs** (`R-4` vs `master 0b`, `P-0.6` vs `P-0.3`, `P-1.2` vs `P-1.1`, and `T-0` vs `T-1`).
3. **Physical Contention and Expiry Inversion on `gx10`**: §2 and §9 mandate that the single `gx10` queue be strictly ordered by expiry (`master 15 -> T-0 -> S-3`). However, `master 15` is blocked by rows 6 and 12 which expire on **2026-10-02**, whereas `T-0` and `S-3` expire on **2026-09-26**. Placing `master 15` first in the physical queue violates G-4's rule that earlier queue positions must have earlier or equal expiries.
4. **Premise Count Desynchronization**: §9 line 544 targets `9/9 Step-0 premises answered`, but §3 was expanded across revisions to **23 premises** (`S0-1..S0-23`).

Below is the exhaustive, evidence-grounded report of findings, risks, and necessary remediations.

---

# Potential Mistakes and Improvements

## 1. Critical Execution Defects & Spec Compliance Failures

### 1.1 Vacuous Execution of `scripts/pv_bin.sh` in Acceptance Commands and Gates
- **Observation**:
  In §5 R-0 acceptance command (line 208), the spec prescribes:
  ```bash
  scripts/pv_bin.sh validate contracts/apr-devices-schema-v1.yaml
  ```
  In §4 C12 (line 159), the spec prescribes:
  ```bash
  scripts/pv_bin.sh lint contracts/ --binding contracts/aprender/binding.yaml --strict-test-binding
  ```
  In §5 Contract Discipline preamble (line 175):
  ```
  pv from scripts/pv_bin.sh
  ```
  However, inspecting `scripts/pv_bin.sh` lines 1–5 and lines 650–663 reveals:
  ```bash
  # pv_bin.sh — resolve THE pv built from THIS TREE at HEAD, and prove it.
  #
  # Source it, never execute it:
  #     . scripts/pv_bin.sh || exit 1
  #     "$PV" lint contracts/
  ...
  PV_BIN_RC=0
  PV=$(pv_bin_resolve) || PV_BIN_RC=$?
  ...
  pv_bin_assert_fresh "$PV" || return 1 2>/dev/null || exit 1
  export PV
  ```
  When `scripts/pv_bin.sh` is executed as a standalone executable script (rather than sourced), it executes the resolution logic, asserts freshness, exports `PV`, and exits with code 0. It **never inspects `$@` or executes `exec "$PV" "$@"`**.
- **Empirical Verification**:
  Testing execution live in the shell confirms the vacuity:
  ```bash
  bash scripts/pv_bin.sh validate non_existent_file.yaml; echo exit=$?
  # Output: exit=0
  ```
  The script returned exit 0 on a non-existent file because it never ran `validate`!
- **Impact**:
  Every acceptance command and release gate written as `scripts/pv_bin.sh <subcommand> <args>` will return `PASS` without validating or linting anything. This is a catastrophic silent-pass failure mode—the exact defect class (`PV-IMPROVE-001`, F-26) the spec seeks to eradicate.
- **Improvement & Remediation**:
  Either:
  1. Patch `scripts/pv_bin.sh` to support direct CLI invocation by appending an argument dispatcher at the bottom:
     ```bash
     if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
         if [ $# -gt 0 ]; then
             exec "$PV" "$@"
         else
             printf '%s\n' "$PV"
         fi
     fi
     ```
  2. Or update the spec's normative command strings to explicitly source the script:
     ```bash
     . scripts/pv_bin.sh && "$PV" validate contracts/apr-devices-schema-v1.yaml
     ```

---

### 1.2 DAG Invariant Contradiction: Zero-Slack Blocker Pairs in §5
- **Observation**:
  §4 C10 (line 161) mandates:
  ```
  The obligation DAG is data, not prose, and its invariants hold: 0 cycles, 0 zero-slack blocker pairs (min 6 days)...
  scripts/check_dag_invariants.sh docs/specifications/pp-066-dag.yaml --min-slack-days 6 exit 0
  ```
  §5 G-4 (lines 430, 433) formalizes this invariant:
  ```
  min slack 6 days between a row and every row it blocks
  invariants: acyclic; expires(blocker) + 6d <= expires(blockee)
  ```
  However, auditing the ticket expiries and blocker definitions in §5 reveals four explicit zero-slack pairs:
  1. **`R-4` vs `master 0b`**:
     - `R-4` (line 265): `blockers: master 0b · expiry 2026-09-19`.
     - `master 0b` (Track I, line 186): `expiry (master) 2026-09-19`.
     - Slack: $2026\text{-}09\text{-}19 - 2026\text{-}09\text{-}19 = \mathbf{0\text{ days}}$.
  2. **`P-0.6` vs `P-0.3`**:
     - `P-0.3` (line 301): `expiry 2026-09-19`.
     - `P-0.6` (line 302): `blockers: P-0.3 · expiry 2026-09-19`.
     - Slack: $2026\text{-}09\text{-}19 - 2026\text{-}09\text{-}19 = \mathbf{0\text{ days}}$.
  3. **`P-1.2` vs `P-1.1`**:
     - `P-1.1` (line 307): `expiry 2026-10-10`.
     - `P-1.2` (line 308): `blockers: P-1.1 · expiry 2026-10-10`.
     - Slack: $2026\text{-}10\text{-}10 - 2026\text{-}10\text{-}10 = \mathbf{0\text{ days}}$.
     - Note: While the text states `same PR as P-1.1`, representing `P-1.2` as an independent card with `P-1.1` as its blocker trips the invariant.
  4. **`T-0` vs `T-1`**:
     - `T-1` (line 356): `expiry 2026-09-26`.
     - `T-0` (line 372): `blockers: T-0h, T-1, T-2 · expiry 2026-09-26`.
     - Slack: $2026\text{-}09\text{-}26 - 2026\text{-}09\text{-}26 = \mathbf{0\text{ days}}$.
     - Note: Line 356 notes `gx10 (queue slot 2, bundled with T-0's window)`, but in the DAG edge graph, `T-0` depends on `T-1`.
- **Impact**:
  When `scripts/check_dag_invariants.sh` is executed in CI against `docs/specifications/pp-066-dag.yaml`, the check will immediately fail with at least four zero-slack violations.
- **Improvement & Remediation**:
  The spec must resolve whether "bundled in the same PR" or "co-scheduled in the same hardware window" are valid DAG exceptions. If so, `check_dag_invariants.sh` must permit co-expiring nodes under an explicit attribute (e.g. `bundled_with: <parent>`); if not, the dependent ticket expiries must be staged with at least 6 days of slack.

---

### 1.3 Physical Contention & Expiry Inversion on `gx10` Queue
- **Observation**:
  §2 (lines 100–102) specifies the `gx10` physical queue rule:
  ```
  gx10 is one queue, ordered by expiry: master 15 (shakedown) → T-0 → S-3's gx10 leg → master 21 (0.67).
  ```
  §9 line 543 states:
  ```
  heijunka: ≤ 1 speed row in flight per host; gx10 = one queue ordered by expiry
  ```
  §5 G-4 (line 433) specifies the queue invariant:
  ```
  queue_pos(host, a) < queue_pos(host, b) ⇒ expires(a) ≤ expires(b)
  ```
  Now examine the expiries and blockers of these tasks:
  1. `master 15` (Track I, lines 191, 196, 198):
     - Blocked by `master 6` (expiry **2026-10-02**) and `master 12` (expiry **2026-10-02**).
     - Earliest allowable expiry for `master 15` per the 6-day slack rule: **2026-10-08**.
  2. `T-0` (Track T, line 372):
     - Expiry: **2026-09-26** (designated as `queue slot 2`).
  3. `S-3` (Track S, line 336):
     - Expiry: **2026-09-26** (designated as `queue slot 3`).
- **Impact**:
  `master 15` is assigned queue slot 1, yet cannot physically run until after October 2. If `gx10` strictly waits for slot 1 before executing slot 2, `gx10` will remain idle throughout late September, causing `T-0` and `S-3` to blow past their September 26 expiry dates! Furthermore, because `expires(master 15) > expires(T-0)`, placing `master 15` at queue position 1 violates the invariant `queue_pos(a) < queue_pos(b) => expires(a) <= expires(b)`.
- **Improvement & Remediation**:
  Re-sequence the `gx10` queue:
  - Either `T-0` (window slot 1) and `S-3` (window slot 2) must execute in September ahead of `master 15` (which enters queue slot 3 once rows 6 and 12 land in October);
  - Or `master 15`'s blocker set must be decoupled from rows 6 and 12 if an earlier September shakedown is intended;
  - Or `T-0` and `S-3`'s expiries must be rescheduled to mid-October.

---

### 1.4 Premise Count Drift between §9 Target and §3 Expansion
- **Observation**:
  In §9 line 544:
  ```
  genchi genbutsu: 9/9 Step-0 premises answered by a pasted command output before ticket #1 | S0 ledger
  ```
  In §3 (lines 109, 111–135):
  - Line 109 states: `Falsify the eight premises the 0.66 plan depends on...`
  - The premise table actually enumerates **23 premises**: `S0-1` through `S0-23`!
  - `S0-10..S0-13` were added in v1.1, `S0-14..S0-15` in v1.2, `S0-16..S0-19` in v1.3, `S0-20..S0-22` in v1.4, and `S0-23` in v1.5.
- **Impact**:
  A gate checking §9's target (`9/9`) would pass if only 9 premises are answered, leaving 14 critical premises (including `S0-14` CUDA toolkit-free build, `S0-20` no cudart, `S0-21` unified memory behavior, and `S0-23` strict check branch protection) unvalidated before opening ticket #1.
- **Improvement & Remediation**:
  Update §9 line 544 to **23/23 Step-0 premises**, and update §3 line 109 to "Falsify the twenty-three premises...".

---

### 1.5 Identifier Desynchronization in Track T Cards
- **Observation**:
  In §5 Track T:
  - Line 362 defines card: `T-2 · --max-seq-len honoured or refused, never clamped`. But line 363 specifies:
    ```bash
    pmat work add "T-5a: apr finetune --max-seq-len is honoured..."
    ```
  - Line 378 defines card: `T-3 · training gate, REPORTING + self-ratchet`. But line 379 specifies:
    ```bash
    pmat work add "T-7 (REPORTING): train_tok_per_sec, peak_vram..."
    ```
- **Impact**:
  According to §4 C9 and paiml-implement conventions:
  ```bash
  ls docs/audits/impl-*-receipt.md | wc -l = ticket count
  ```
  If card `T-2` mints a ticket named `T-5a`, the generated receipt will be `impl-t5a-receipt.md` while readers and DAG checkers expecting `impl-t2-receipt.md` will flag a missing receipt. Similarly, card `T-3` minting `T-7` causes `impl-t3-receipt.md` vs `impl-t7-receipt.md` confusion.
- **Improvement & Remediation**:
  Align the string in `pmat work add` to the card identifier: e.g. `pmat work add "T-2: ..."` and `pmat work add "T-3: ..."`, noting the provenance (e.g. "formerly report T-5a / T-7") in the description.

---

## 2. Architectural Design & Contract-Per-Card Rigor

### 2.1 BackendRegistry Architectural Completeness (R-0 & REG-1..REG-14)
- **Strengths**:
  - `R-0` represents an outstanding architectural pivot (F-25), replacing compile-time `cfg!(any(feature = "cuda", feature = "wgpu"))` checks with dynamic runtime discovery via dlopen (`libcuda.so.1` / `nvcuda.dll`) and `wgpu` adapter probing.
  - The requirements `REG-1..REG-14` are exceptionally well-specified, mapping directly to concrete prior-art failure modes in llama.cpp, Ollama, and llamafile (§12).
  - Explicit enforcement of capability reporting (cuBLAS as a capability of `cuda`, not a separate backend) eliminates invisible kernel fallback traps.
- **Missing Link in REG Cross-References (§12)**:
  - In §5 line 226, `REG-10` is defined:
    `Never mix vendors in one graph... mixed AMD+NVIDIA in one llama.cpp process segfaults; Ollama binds to the primary driver and drops the other`
  - In §12 (lines 647–651), the `REG` column maps requirements:
    - llama.cpp: `1, 2, 5, 6, 13, 14`
    - Ollama: `1, 3, 4, 6, 7, 8, 9, 12`
    - llamafile: `3, 5, 6, 8, 11`
  - **`REG-10` is missing from the REG column in both the llama.cpp and Ollama rows in §12**, despite both systems being cited as the source in §5.

---

### 2.2 Contract Discipline & PV-IMPROVE-001 Integration
- **Compliance Audit**:
  The spec preamble to §5 establishes seven strict rules for contract cards to counteract the "contract theater" identified in F-26:
  1. `metadata.kind: pattern` unless stated.
  2. No `registry: true`.
  3. No hand-typed `verification_summary` (derived only).
  4. Executable `falsification_tests[].test`.
  5. No dangling `lean_theorem` or `kani_harnesses`.
  6. `#[contract]` not credited as enforcement until PVI-1.1 lands.
  7. Verification via tree-built verifier (`scripts/pv_bin.sh`).
- **Audit Findings**:
  - All contracts declared across §5 (`apr-backend-registry-v1`, `apr-gpu-install-v1`, `apr-train-banner-truth-v1`, `apr-w5-wallclock-v1`, `apr-release-assets-v1`, `apr-installer-v1`, `apr-perf-wa-v1`, `apr-perf-wc-v1`, `apr-bench-q4k-gemv-v1`, `apr-train-parity-receipt-v1`, `apr-finetune-config-truth-v1`, `apr-train-gate-v1`, `apr-perf-matrix-cell-v1`, `apr-backend-firstclass-v1`, `apr-crate-names-v1`, `apr-obligation-dag-v1`, `apr-doc-citation-v1`) adhere strictly to this schema.
  - Notice that in R-7, the spec honestly states: `contract: none (a doc; the two guards are the contract)` and in G-3: `contract: none (a measurement document)`. This demonstrates commendable refusal to create dummy/ceremonial contracts for documents.

---

### 2.3 Release Asset & Distribution Governance (R-5, R-6, R-7)
- **Strengths**:
  - Addresses the severe artifact-identity gap (F-27, #2869) where releases previously had no pre-built `apr` binaries attached.
  - Formulates a true Poka-Yoke release promotion gate: tagged releases are initially cut with `prerelease=true`. The release cannot be promoted to `prerelease=false` until all four proof hosts pull the target asset, verify checksums and signatures, and execute the dogfood test (`C4`) and `H12` benchmarks.
  - The installer (`scripts/install.sh`, R-6) strictly avoids third-party SaaS verification dependencies (Rekor/Fulcio/Sigstore), embedding the repository's pinned public key directly.

---

### 2.4 Future Lanes (§6 0.67) and Wasm32 Address Space Physics
- **Precision in Wasm32 Physics**:
  - In §6 row `B-S1..S4`, the spec correctly formalizes the 32-bit wasm memory constraint:
    ```
    wasm32 linear memory is 2³² B = 4.00 GiB (browsers commonly 2 GiB); the 7B Q4_K_M weights alone are 4.68 GB, so W1 (7B) is excluded on wasm32 by address space (review MAJ-01) and the cell key is ci/W1_0.5b. The bound is model_bytes + kv_bytes + activations < 2³² per receipt, not "≤ 2B parameters".
    ```
  - This is mathematically exact: $2^{32} \text{ bytes} = 4,294,967,296 \text{ B} = 4.00 \text{ GiB}$. Since the 7B weights alone exceed this limit, substituting a 0.5B model workload (`ci/W1_0.5b`) is physically mandatory.
- **Wasm32 Crate Scope Nuance**:
  - Line 464 asserts: `aprender-serve is #[cfg(not(target_arch = "wasm32"))] entirely [A] — B-S1 is a port of the loader/decoder, not a harness`.
  - Code inspection reveals that while `MappedSafeTensorsModel`, `ShardedSafeTensorsModel`, and file-based loaders are indeed gated under `#[cfg(not(target_arch = "wasm32"))]` (e.g. `src/lib.rs:470,473`), the base crate does compile core tensor/inference components for wasm. Clarifying that it is the *model loading and server infrastructure* that is gated prevents confusion during implementation.

---

## 3. Governance, Predictions, Refusals, and Adjudication

### 3.1 Registered Predictions Protocol (§7)
- **Prediction-Kill vs. Track-Kill**:
  - §7 table header (line 474) explicitly establishes: `prediction killed if (⇒ re-plan from the measurement; never a track kill)`. This directly resolves review issue `MAJ-05` where reviewers conflated falsifying a numerical prediction with abandoning a workstream.
- **Mathematical Accuracy in §7**:
  - S-2 W-C: `5,700 = 0.55× comparator [C]`. Verified: $5700 / 10399 = 0.5481... \approx 0.55\times$.
  - T-0: `≈ 23 GB activations before scores [C]`. Verified:
    $$7\text{ projections} \times 28\text{ layers} \times (4 \times 2048 \times 3584 \times 4\text{ B}) = 196 \times 117,440,512\text{ B} = 23,018,340,352\text{ B} \approx 23.02\text{ GB (decimal)}.$$
    Fits $24\text{ GB}$ GPU VRAM with negligible headroom even before attention scores and weights.
  - S-3 W-B: `t(M=4)/t(M=16) <= 1.3`, kill if $\ge 1.5$.
  - R-4 W5: Sample size derivation:
    $$n = \lceil (1.96 \cdot CV / 0.05)^2 \rceil.$$
    Statistically exact formula for a 95% confidence interval half-width of $\pm 5\%$.

---

### 3.2 Refusal Invariants (§8)
- §8 documents 23 comprehensive refusal bullet points. Key highlights:
  - **No silent downgrades**: Banning silent CPU fallbacks across `--backend`, `--gpu`, `--device`, `--reserve-bytes`, and `--gpu-layers`.
  - **No premature feature defaults**: Enforcing that `default = ["cli", "cuda"]` cannot merge until `R-0` is merged and `S0-14` is verified on all four hosts (`D-9`).
  - **No ungrounded ratio claims**: Banning "$\approx 96\%$ of llama.cpp" prose without a `CONFORMANT` receipt (`PP-12`).

---

### 3.3 Adjudication of `0_66-review.md` (§11)
- §11 provides an exemplary, unsparing adjudication of the 32 findings and recommendations raised in `0_66-review.md`.
- All four formal adjudication verdicts are utilized with rigorous justification:
  - **SUSTAINED**: Finding factually and logically correct (e.g. `MAJ-04` gx10 queue, `MAJ-07` in-crate module aliases failing extern-prelude E0432, `MIN-01` ceiling formula typo, `MIN-07` exit code 9 vs 2).
  - **NARROWED**: Fact true but conclusion overreaching (e.g. `MAJ-01` wasm 4 GiB bound acknowledged but "$\le 2\text{B}$ parameter" threshold rejected in favor of byte-budget receipts; `MAJ-03` W-H blocking arming rather than reference receipt).
  - **REJECTED**: Finding invalid or doctrinally flawed (e.g. `seg-2 3.2/3.3` speculative KV quantization rejected without measured binding constraints; `seg-5 F7` staffing hierarchy rejected as fictional org charting).
  - **INDETERMINATE**: Contingent on Step-0 discovery (`MAJ-10` uncommitted master claim assigned to `S0-1`).

---

# Minor Corrections and Typos

### 1. Stale Line Citations in §10 Verification Ledger
Empirical audit of the codebase reveals several minor line-number shifts between the snapshot commit (`587ad0797`) and current HEAD:
- **`aprender_ml` alias** (line 575):
  - Spec states: `aprender_ml alias at root Cargo.toml:509`.
  - Actual HEAD: Line 509 of root `Cargo.toml` is `identity_op = "allow"`. The `aprender_ml` alias is located at line **666**:
    ```toml
    aprender_ml = { path = "crates/aprender-core", version = "0.65.2", package = "aprender-core" }
    ```
- **`FeatureDisabled` in `error.rs`** (line 575):
  - Spec states: `error.rs:86-90,218 FeatureDisabled -> 9`.
  - Actual HEAD: In `crates/apr-cli/src/error.rs`, enum variant `FeatureDisabled` is declared at line 63, mapped to `9` in `exit_code_value()` at line 106, and tested at line 285. Lines 86–90 contain the `exit_code()` method.
- **Manifest count under git** (line 575):
  - Spec states: `106 manifests under git ls-files '*/Cargo.toml'`.
  - Actual HEAD: `git ls-files '*/Cargo.toml'` returns **104 manifests** (or 105 if root `Cargo.toml` is included).
- **`finetune.rs` sequence length clamp** (lines 123, 363, 575):
  - Spec states: `finetune.rs:317,718 hardcode 512`.
  - Actual HEAD: The hardcoded `512, // max_seq_len` parameter is at line **717** (and tokens/sec multiplier at line 1930); line 317 is a doc comment describing the clamp.

### 2. Duplicate Section Heading in §11 Adjudication
- In §11:
  - Line 631 reads: `**Five findings no reviewer raised** (standard deliverable) — a sixth, found 2026-09-05...`
  - Line 633 immediately repeats: `**Five findings no reviewer raised** (v1.1):` followed by items 1–5.
  - *Correction*: Merge into a single heading, e.g. `**Findings no reviewer raised (v1.1–v1.5)**:`.

### 3. Mismatched Backtick in R-5 Contract Invariants
- Line 275 reads:
  ```markdown
  (ii) `prerelease=false ⇒ ∃ 4 host receipts with `asset_sha256 == manifest[target]`;
  ```
  Notice the opening backtick before `asset_sha256` that breaks markdown formatting.
  - *Correction*: `(ii) prerelease=false ⇒ ∃ 4 host receipts with asset_sha256 == manifest[target];`

### 4. Crate Name Guard Allow-List Count Consistency (51 vs 52)
- Line 411 states: `allow-list = 51 rows + the D-5 row (52)`.
- Line 415 states: `remaining=51`.
- Line 417 states: `remaining would read 50 while 51 offend`.
- Line 154 (C6) states: `allow-list = 51, renames = 0`.
- Line 546 (§9) states: `check_crate_names.sh remaining: 51 -> 51 in 0.66`.
- *Correction*: Clarify whether the allow-list file contains 51 lines or 52 lines (including the special `D-5` row for the duplicate `[lib] name = "aprender"`).

### 5. Formatting in §7 Table
- In §7 (lines 483, 484), the `note` column is left blank with trailing empty pipes `| |`.
- *Correction*: Fill in an explicit note or `-` placeholder for clean markdown rendering.

---

### Verification and Test Traceability Summary

| Item Audited | Referenced in Spec | Actual Codebase State | Status |
|---|---|---|---|
| `scripts/pv_bin.sh` execution | §5 R-0 A, §4 C12 | Drops `$@` when executed; only exports `$PV` | ⚠️ **FAIL (Silent Bypass)** |
| DAG min-slack 6d | §4 C10, §5 G-4 | 4 zero-slack pairs identified (`R-4`, `P-0.6`, `P-1.2`, `T-0`) | ⚠️ **FAIL (Contradiction)** |
| `gx10` queue order | §2, §9, §5 G-4 | `master 15` blocked by Oct 2 rows, blocks Sept 26 `T-0`/`S-3` | ⚠️ **FAIL (Queue Inversion)** |
| Step-0 premise count | §9 target | 9 declared vs. 23 actual (`S0-1..S0-23`) | ⚠️ **FAIL (Drift)** |
| `CliError::FeatureDisabled` | §0 D-7, §5 R-0, §10 | Mapped to `9` in `error.rs:106` | ✅ **VERIFIED** |
| `accel.rs:28` feature check | §1 F-25, §5 R-0, §10 | `cfg!(any(feature = "cuda", feature = "wgpu"))` | ✅ **VERIFIED** |
| `accel.rs:114` 3-surface test | §5 R-0 A, §10 | Tests `run`, `chat`, `serve` surfaces | ✅ **VERIFIED** |
| MSL kernels in `aprender-gpu` | §3 S0-12, §6 B-M1, §10 | Exactly 13 `kernel void` functions in `metal_shaders.rs` | ✅ **VERIFIED** |
| `Q4K_GEMV_SHADER` | §3 S0-13, §6 B-W0, §10 | Present at line 555 of `basic_ops.rs` | ✅ **VERIFIED** |
| `OwnedQuantizedModelWgpu` stub | §3 S0-13, §6 B-W0, §10 | Returns `UnsupportedOperation` in `wgpu_backend/mod.rs:197` | ✅ **VERIFIED** |
| Double `[lib] name = "aprender"` | §0 D-5, §5 G-2 | Root `Cargo.toml:693` & `aprender-core/Cargo.toml:72` | ✅ **VERIFIED** |
| Facades separate workspace | §5 G-1, §6 renames | `crates/facades/Cargo.toml` avoids `.rlib` collision | ✅ **VERIFIED** |
| 7.61B f32 expansion | §6 B-W0, §10 | $7.61 \times 10^9 \times 4\text{ B} = 30.44\text{ GB}$ | ✅ **VERIFIED** |
| 7B QLoRA seq 2048 activation | §5 T-0h, §7 T-0, §10 | $196 \times 117.44\text{ MB} = 23.02\text{ GB}$ | ✅ **VERIFIED** |
| Wasm32 linear memory bound | §6 B-S1..S4, §10 | $2^{32}\text{ B} = 4.00\text{ GiB}$ | ✅ **VERIFIED** |
| Comparator bar ratio | §1 F-14, §6 W-E, §10 | $484.7 / 171.5 = 2.826... \approx 2.83\times$ | ✅ **VERIFIED** |
| Prefill ratio prediction | §5 S-2 note, §7 S-2 | $5700 / 10399 = 0.548... \approx 0.55\times$ | ✅ **VERIFIED** |
