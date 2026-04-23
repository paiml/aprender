# Audit: SPEC-SHIP-TWO-001 — What's Working, What Isn't

**Document ID:** SPEC-SHIP-TWO-001-AUDIT
**Version:** 1.0.0
**Audit date:** 2026-04-23
**Auditor:** PAIML Engineering (new branch `docs/ship-two-001-spec-audit-what-works` off `main` at `601c0740f`)
**Subject:** `docs/specifications/aprender-train/ship-two-models-spec.md` v2.29.0 (on main) and its pending v2.30.0 amendment (PR #1024 OPEN)

## 0. Why this audit

The ship-two-models spec has accreted **28 amendments in 6 days** (v1.0.0 2026-04-17 → v2.30.0 2026-04-23). The amendment log now consumes roughly 600 lines before readers reach §1 Abstract. Before continuing to extend the spec, verify which claims are backed by code on `main` and which are claims-in-flight on stacked branches or open PRs.

Result: **the spec is running ahead of reality in two places and behind reality in one.**

## 1. Ground truth on `main` (commit `601c0740f`)

### 1.1 MODEL-1 (7B Qwen2.5-Coder teacher + distilled student)

Spec §4.2 defines 10 acceptance criteria. Status on `main`:

| AC               | Spec claim (v2.30.0 wrap text) | On `main` (`601c0740f`)                           | Evidence                                         |
|------------------|--------------------------------|---------------------------------------------------|--------------------------------------------------|
| AC-SHIP1-001     | unwired                        | unwired on main; WIP on stacked branch            | `d4c6b6141` (not merged)                         |
| AC-SHIP1-002     | PARTIAL v2.26.0                | **✓ PARTIAL live**                                | `qa/ship_002.rs`, PR #1017 merged `f61514878`    |
| AC-SHIP1-003     | unwired                        | unwired on main; PARTIAL on stacked branch        | `f9c2d4753` (not merged)                         |
| AC-SHIP1-004     | unwired                        | unwired on main; PARTIAL on stacked branch        | `5f1db6ab7` (not merged)                         |
| AC-SHIP1-005     | PARTIAL v2.27.0                | **✓ PARTIAL live**                                | `metrics/ship_005.rs`, PR #1021 merged `9e3286df9` |
| AC-SHIP1-006     | PARTIAL v2.25.0                | **✓ PARTIAL live**                                | `qa/ship_006.rs`, PR #1013 merged `045785748`    |
| AC-SHIP1-007     | PARTIAL v2.29.0 / v2.30.0      | **✓ PARTIAL live**                                | `bench/ship_007.rs`, PR #1019 merged `651e07b6f` |
| AC-SHIP1-008     | PARTIAL v2.24.0                | **✓ PARTIAL live**                                | `text/chat_template/ship_008.rs`, PR #1012 merged `1263178a3` |
| AC-SHIP1-009     | PARTIAL v2.23.0                | **✗ NOT on main** (PR #1009 OPEN)                 | `90598277a` not ancestor of main; no `GATE-APR-PROV-004` in `contracts/apr-provenance-v1.yaml` |
| AC-SHIP1-010     | PARTIAL v2.28.0                | **✓ PARTIAL live**                                | `format/ship_010.rs`, PR #1022 merged `77296253b` |

**On-main count: 6/10 PARTIAL.**
**Spec v2.30.0 amendment claim: "MODEL-1 coverage 7/10 now live on main (joins SHIP-002/005/006/008/009/010 + this SHIP-007)".**
**→ Drift: the wrap text asserts SHIP-009 is on main; it is not.**

### 1.2 MODEL-2 (albor sovereign 370M)

Spec §5.2 defines 12 acceptance criteria. Status on `main`:

| AC           | Spec claim                         | On `main` (`601c0740f`)                                 | Evidence                         |
|--------------|-------------------------------------|---------------------------------------------------------|----------------------------------|
| AC-SHIP2-001 | DISCHARGED v2.21.0                 | **✓ DISCHARGED**                                        | PR #898 merged `7855dcd37`       |
| AC-SHIP2-002 | PARTIAL v2.21.0                    | **✓ PARTIAL**                                           | PR #898 merged                   |
| AC-SHIP2-003 | compute-blocked by task #132       | **blocked** — no real run                               | —                                |
| AC-SHIP2-004 | compute-blocked by task #132       | **blocked** — no real run                               | —                                |
| AC-SHIP2-005 | PARTIAL v2.21.0                    | **✓ PARTIAL**                                           | PR #898 merged                   |
| AC-SHIP2-006 | PARTIAL v2.23.0                    | **✗ NOT on main** (PR #1008 OPEN, `cbb0542e8`)           | branch `feat/falsify-ship-016-…` |
| AC-SHIP2-007 | PARTIAL v2.23.0                    | **✗ NOT on main** (PR #1004 OPEN, `16ab9caef`)           | branch `feat/falsify-ship-017-…` |
| AC-SHIP2-008 | PARTIAL v2.23.0                    | **✗ NOT on main** (PR #1006 OPEN, `2bd513c05`)           | branch `feat/falsify-ship-018-…` |
| AC-SHIP2-009 | PARTIAL v2.22.0                    | **✓ PARTIAL**                                           | PR #898 merged                   |
| AC-SHIP2-010 | PARTIAL v2.23.0                    | **✗ NOT on main** (PR #1005 OPEN, `3e2c2e4f0`)           | branch `feat/falsify-ship-020-…` |
| AC-SHIP2-011 | DISCHARGED v2.20.0                 | **✓ DISCHARGED**                                        | PR #898 merged                   |
| AC-SHIP2-012 | DISCHARGED v2.20.0                 | **✓ DISCHARGED**                                        | PR #898 merged                   |

**On-main count: 6/12 touched (3 DISCHARGED + 3 PARTIAL).**
**Spec v2.23.x running claim: "10/12 touched".**
**→ Drift: the 10/12 count is branch-state, not main-state. Four MODEL-2 PARTIAL gates (SHIP-016/017/018/020) live only in open PRs.**

### 1.3 Task #132 (CUDA training backend)

Spec §14 (v2.23.0 amendment) describes a 5-phase plan and records only **Phase 0** as shipped. **On `main` today, Phase 1 and Phase 2 have shipped too.**

| Phase | Spec status                     | Reality on `main`                                        |
|-------|---------------------------------|----------------------------------------------------------|
| 0     | "DONE — contract + §14"         | ✓ `contracts/entrenar/gpu-training-backend-v1.yaml` v1.0.0 PROPOSED (861 contracts now total) |
| 1     | "pending — 1 day"               | **✓ SHIPPED** — `crates/aprender-train/src/train/device.rs` exists; `Device` enum + `resolve_device()` wired; `--device` CLI flag accepted (see `crates/apr-cli/src/commands/pretrain.rs`) |
| 2     | "pending — 2 days"              | **✓ SHIPPED** — `drive_real` branches into `drive_real_cpu` + `drive_real_cuda`; cuda path invokes `build_shared_cuda_trainer`; `#[cfg(not(feature = "cuda"))]` stub surfaces `GATE-GPUTRAIN-002` error verbatim from the contract |
| 3     | "pending — 2 days, evidence on RTX 4090" | ✗ **NOT shipped** — `evidence/task-132/` does not exist; no `rtx4090-370m-step-budget.json` |
| 4     | "pending — promote PROPOSED → ACTIVE" | ✗ contract still `status: PROPOSED` |

**Phase 1 + Phase 2 shipped via PR #989 (`a5fee06b6 feat(task-132): Phase 0 — GPU training backend contract + spec v2.23.0`) and follow-ups; the spec §14 text was never revised to catch up.**

**→ Drift: the spec underrepresents task #132 progress.** MODEL-2 compute-dispatch (AC-SHIP2-003/004, plus full discharge of 006/007/008/010) is now blocked **only** on Phase 3 evidence (lambda-labs RTX 4090 `apr pretrain --mode from-scratch --device cuda:0 --num-steps 50`), not on code.

### 1.4 v2.30.0 wrap PR #1024 itself

PR #1024 (`docs/ship-two-001-v2.30.0-wrap`, `68d97935a`) is **OPEN** and not merged. The spec text on `main` is **v2.29.0**. Anyone reading main sees pre-v2.30.0 content; the v2.30.0 narrative (fleet CI fix + SHIP-007 landing) lives only in the open PR.

## 2. What is working

Things the spec describes accurately and which are load-bearing on `main`:

1. **MODEL-1 TEACHER (`paiml/qwen2.5-coder-7b-apache-q4k-v1`) RELEASED** — v2.11.0 milestone. Three-format ship (`.apr` / `.gguf` / `.safetensors`) live on HF Hub; `apr pull` round-trip sha256-clean. EX-01..EX-07 discharge evidence under `evidence/ship-two-001/`.

2. **Pre-flight Poka-Yoke stack** — PM-001 through PM-009 all fire on real 8–15 GiB artifacts before any network socket opens. Exit-2 before upload on every ship-blocker class. FALSIFY-PUB-EXTRA-001..010 round-trip live.

3. **Xet large-file upload** — `hf-xet = "1.5.1"` behind the `xet` feature; `reject_oversized_file()` deleted from production code; NDJSON `lfsFile` schema fix (v2.9.0) discharged the silent-no-op class. Three-format live discharge on real teacher sizes.

4. **MODEL-2 algorithmic scaffold** — `crates/aprender-train/src/models/llama_370m.rs` (byte-equally bound to `contracts/model-families/llama-370m-sovereign-v1.yaml` v1.5.0 ACTIVE), BPE NFC tokenizer, `apr tokenize train` + `apr-corpus-ingest` binaries, `apr pretrain` CLI with synthetic drive, AdamW checkpointing, seed=0 100-step reproducibility (|Δloss|≤1e-6), `apr inspect` provenance block all on main.

5. **Pure-Rust BPE real-scale perf** — Task #118 brought MODEL-2 50,257-vocab BPE training on a real 127 MB corpus from "25h+ and never completes" to **51.68 minutes** via priority-queue + inverted-index + lex-min (contract `tokenizer-bpe-v1.yaml` v1.2.0 PROPOSED with amended 60-min GATE-003).

6. **Contracts schema harmonization** — 760 contracts validate under `pv validate` on main; `aprender-contracts` crate is the dogfooded tool. No bash/python workarounds surviving.

7. **Fleet CI hardening** (v2.30.0 wrap narrative, technically landed on main) — `paiml/.github#31` (`136863e`) ported per-PR `CARGO_TARGET_DIR` isolation into the reusable `sovereign-ci.yml`; closed the 15×-in-a-row disk-guard collision class that wedged PR #1019 and every fleet-test downstream.

8. **PARTIAL_ALGORITHM_LEVEL as a first-class pattern** — 13 PARTIAL gates across MODEL-1/2 each bind a pure Rust `verdict_from_*` fn + 5-8-section mutation survey to a YAML `evidence_discharged_by` slot with `discharge_status: PARTIAL_ALGORITHM_LEVEL` + `full_discharge_blocks_on:`. This is the right shape and should be codified as contract schema v2.

## 3. What is not working

Concrete drift and dead weight.

### 3.1 Spec-vs-main drift (**immediate cleanup**)

| Finding                                                                                                             | Fix                                                                                                 |
|---------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| Spec v2.30.0 wrap text claims MODEL-1 is 7/10 on main; reality is 6/10 (SHIP-009 PR #1009 unmerged)                  | Either merge #1009 before #1024, or change the wrap text to "6/10 on main + SHIP-009 pending #1009" |
| Spec v2.23.x running claim "MODEL-2 10/12 touched" is branch-state, not main-state                                   | Distinguish "touched across branches" from "touched on main" throughout §5 / amendments              |
| Spec §14 says task #132 is at Phase 0; Phase 1 + 2 have shipped                                                      | Amend §14 to reflect device.rs + drive_real_cuda landing; only Phase 3 evidence is the real blocker |
| 4 MODEL-2 open PRs (#1004, #1005, #1006, #1008) implement the spec's own PARTIAL claim but sit unmerged               | Batch-merge the four or close the claim                                                             |
| Stacked SHIP-001/003/004 on `feat/falsify-ship-001-partial-discharge` (3 commits ahead of main) — not yet split into PRs | Split into 3 sibling PRs matching the 6 merged MODEL-1 PARTIAL pattern                              |

### 3.2 Spec readability (**structural debt**)

1. **Amendment log dominates the document.** Lines 1–600 are 28 consecutive amendment paragraphs (v1.0.0 → v2.30.0). By v2.11.0 readers stop reaching §1 Abstract. The amendment log is a `git log` that was inlined into the document.
   **Fix:** move the amendment history to an appendix (`§Appendix A: Amendment log`); keep a single 10-line "Current Status" block at the top with links into §4, §5, §12, §14.

2. **The `**Status:**` field is a run-on sentence.** 1,900 characters, 17 semi-colon clauses. Not readable; not parseable.
   **Fix:** replace with a short YAML-front-matter-style block:
   ```yaml
   status:
     model_1_teacher: RELEASED (2026-04-18, paiml/qwen2.5-coder-7b-apache-q4k-v1)
     model_1_distilled: DEFERRED (task #86 retry plan)
     model_2_sovereign: BLOCKED on task #132 Phase 3 evidence
     spec_version_on_main: 2.29.0
     pending_amendment: PR #1024 → v2.30.0
   ```

3. **Duplication between §4.2 / §7.1 / amendment paragraphs.** Each MODEL-1 AC has its status tracked in three places (table row + falsification table + inlined amendment). These fall out of sync (as §1.1 above shows).
   **Fix:** single source of truth — a generated status table that reads from contract `evidence_discharged_by` + git ancestry of the named test.

### 3.3 Zero-Tolerance conflict (**policy risk**)

Spec §3 row #8: *"No 'pre-existing' carve-outs. No `#[ignore]` as a release valve."*

The v2.30.0 wrap text ships a factual misstatement as "current status" (the "7/10 on main" claim). That is exactly the failure mode §3 row #8 warns against: documenting a green state that is not green. **The policy audit the spec itself mandates should flag the spec.**

**Fix:** add a contract-gated CI step — `cargo xtask audit-ship-two` — that on every push reads `docs/specifications/aprender-train/ship-two-models-spec.md`, extracts any "N/M on main" claim, and fails CI if the matching `evidence_discharged_by` path is not an ancestor of `main` for that many AC-SHIP-*.

## 4. What is blocked, and on what

After all the drift is cleaned up, the actual critical path narrows to **one** compute dispatch:

```
task #132 Phase 3 evidence
         │
         │  lambda-labs RTX 4090
         │  apr pretrain --mode from-scratch --device cuda:0 --num-steps 50
         │  → evidence/task-132/rtx4090-370m-step-budget.json
         │
         ▼
task #132 Phase 4 (contract PROPOSED → ACTIVE)
         │
         ▼
task #126 real-compute MODEL-2 pretraining (full run, not 50 steps)
         │
         ▼
real 370M .apr checkpoint
         │
         ├───► AC-SHIP2-003 target_val_loss (CE ≤ 2.2)
         ├───► AC-SHIP2-004 21-day budget
         ├───► AC-SHIP2-006 full discharge (real apr qa 8 gates)
         ├───► AC-SHIP2-007 full discharge (100 Python held-out prompts)
         ├───► AC-SHIP2-008 full discharge (humaneval ≥ 30%)
         └───► AC-SHIP2-010 full discharge (apr bench ≥ 100 tok/s)
```

Six MODEL-2 ACs all unblock on one shared compute dispatch. Everything else that the spec describes is either shipped on main or trivially mergeable (the 4 open MODEL-2 PARTIAL PRs + SHIP-009 + stacked SHIP-001/003/004).

## 5. Recommended next moves (in priority order)

1. **Clean up main to match v2.30.0's claims before merging #1024.**
   Merge the 6 mergeable PRs: #1009 (SHIP-009) + #1008 (SHIP-016) + #1004 (SHIP-017) + #1006 (SHIP-018) + #1005 (SHIP-020) + split stacked SHIP-001/003/004 into 3 PRs. After this, MODEL-1 is 9/10 or 10/10 on main and MODEL-2 is 10/12 on main — the spec's current claims become true.

2. **Amend spec §14 to reflect task #132 reality.** One-paragraph edit noting Phase 1 + 2 shipped; Phase 3 is the only remaining code gap and it's an evidence-file gate, not a kernel gate.

3. **Dispatch task #132 Phase 3 on lambda-labs.** Compute lanes pre-authorized per `feedback_compute_pre_authorized.md`; run is ≤1 hour at ~500 ms/step × 50 steps; produces `evidence/task-132/rtx4090-370m-step-budget.json`; Phase 4 promotion is a 1-line YAML change on success.

4. **Refactor the spec itself.** Amendment log → appendix. Status → YAML block. Single source of truth for AC status (generate, don't hand-maintain). This buys the next 6 months of spec evolution at constant reader cost.

5. **Add `cargo xtask audit-ship-two` CI gate** matching §3.3 above. Structural enforcement of the spec's own Zero-Tolerance principle.

6. **Close the loop on MODEL-1 v2 distilled.** It's been DEFERRED since v2.10.0 (2026-04-18) under `task #86 retry plan`. Either run the retry or formally cut it from v2.x scope so it stops being in the "planned but not done" shadow.

## 6. What this audit deliberately does not do

- **Does not change the spec.** The spec belongs to PR #1024 (v2.30.0 wrap) and is the subject of the audit. Edits to the spec belong in #1024 or a follow-up.
- **Does not dispatch compute.** Phase 3 evidence is a separate authorized dispatch; this document only maps where it is on the critical path.
- **Does not merge open PRs.** Each of the 6 recommended merges (#1009, #1008, #1004, #1006, #1005, stacked split) has its own review window and is the owner's call.

## 7. Cross-reference

- Spec: `docs/specifications/aprender-train/ship-two-models-spec.md` v2.29.0 on main, v2.30.0 in PR #1024
- Contracts touched on main:
  - `contracts/model-families/llama-370m-sovereign-v1.yaml` v1.5.0 ACTIVE
  - `contracts/apr-provenance-v1.yaml` v1.0.0 ACTIVE (3 gates, no GATE-APR-PROV-004 yet)
  - `contracts/tokenizer-bpe-v1.yaml` v1.2.0 PROPOSED
  - `contracts/chat-template-v1.yaml` v1.1.0
  - `contracts/publish-manifest-v1.yaml` v1.4.0 DRAFT
  - `contracts/apr-model-qa-v1.yaml` v1.2.0
  - `contracts/qwen2-e2e-verification-v1.yaml` v1.3.0
  - `contracts/entrenar/gpu-training-backend-v1.yaml` v1.0.0 PROPOSED
- Open PRs referenced: #1004, #1005, #1006, #1008, #1009, #1024
- Stacked branch: `origin/feat/falsify-ship-001-partial-discharge` @ `d4c6b6141` (3 commits ahead of main)

---

*End of audit. This document is companion to the spec, not a replacement.*
