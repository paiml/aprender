# Roadmap — Next Wave (2026-07-05): Fable Architectural Review Integration

**Supersedes** the beat-batch framing in `roadmap-next-wave-2026-06-21.md` and
`roadmap-v054-beat-batch-2026-06-22.md` as the live continuation of the autonomous campaign
(mandate per `autonomous-continuous-operation-2026-06-15.md`, four-pillar mission per
`project_mission_four_pillars`). **Source of direction:** the 2026-07-05 Fable architectural
review (`docs/specifications/fable-architectural-review.md`, landed on `main` via #2292 — a
14-agent, artifact-grounded audit with an EV-ranked, falsifiable backlog).

---

## 0. The direction shift (why this wave is different)

The v0.42→v0.59 waves manufactured **more** beats (REPLACE then BEAT sklearn/PyTorch/Unsloth/
Ollama). The Fable review's central finding is that **a large fraction of the gates and beats we
already claim are theater** — present in the tree, absent from any workflow, or structurally
blind to the bug class they name. So this wave's ordering inverts the prior emphasis:

> **Enforcement integrity first, new beats second.** A gate that cannot turn RED on a real
> regression is worth negative EV — it manufactures false confidence. Ranks 1–7 of the backlog
> are gate-integrity / anti-theater work; the classic "add another beat" items sink to ranks 8–10;
> the remaining infra-hardening fills ranks 12–18.

This is the same "gates or theater" doctrine the campaign already runs on
(`feedback_workspace_test_lib_only_beats_ungated`), applied reflexively to our **own** gate corpus.

### Grounding (this is not a stale re-quote of the review)

The review is explicitly time-sensitive and was authored against `9f0b89fb1`; this wave is
authored against the current merged tree (`chore-merge-all`). A 5-agent grounding workflow
(`wf_103658cf-4d2`, 2026-07-05) re-verified the load-bearing claims against the live tree:

| item | review verdict | re-grounded 2026-07-05 | evidence |
|---|---|---|---|
| rank-1 CI passgrep bug | STALE/theater | **CONFIRMED still true** | `ci.yml:288` greps `"test result.*0 failed"`; empirically `"…; 10 failed; …"` matches → a 10-failure compute run merges green (cargo's 101 masked by `tee\|grep`). |
| rank-7 drift-gates theater | STALE | **CONFIRMED still true** | `scripts/check_readme_claims.sh` exists + executable but `grep -rn check_readme_claims .github/workflows` = 0 hits; only the `readme_contract` Rust test is wired. |
| rank-5 GPU train step-0-only | STALE (actionable) | **CONFIRMED still true** | `parity_probe.rs:186` is forward-only (no backward / optimizer step); CPU has an enforced 400-step trajectory (`train_to_loss_tests.rs:84`). No GPU trajectory test exists. |
| rank-11 cuda-nightly non-probative | STALE (actionable) | **CONFIRMED still true** | `gqa_attention_parity.rs` horizon = 2 positions (vs PMAT-749's ~64); `gpu_cpu_parity.rs` builds `_cuda_model` and never compares it; neither wired into any workflow. |
| GPU falsifier gap (P3) | STALE (closed / good) | **CONFIRMED still closed** | `cuda-nightly.yml` runs the 5 `FALSIFY-CUDA-*` with `--include-ignored` + provisioned `APR_PARITY_MODEL` on ada-4090 (sm_89) + blackwell-gb10 (sm_121). |
| ci.yml:317 single physical line | (constraint) | **CONFIRMED** | `ci.yml:317` = one `bash -c` chaining 16 `cargo test …` invocations by `&&` → only ONE unmerged PR may edit it (merge-queue conflict otherwise). |

**All 5 clusters completed** (the first pass lost 4 to transient API errors; the resume re-ran them):
**21 of 23 checks CONFIRMED_STILL_TRUE**, so every one of the 18 backlog items is still actionable on
the live tree. The two non-confirmations:

- **rank-5 runtime log — UNVERIFIABLE** (no network to `gh` from this environment; the *code-level*
  step-0-only claim is separately CONFIRMED).
- **roadmap-health — CHANGED, and worse than the review knew:** the "152 inprogress" items are not a
  reconciliation regression, they are **corruption** — 152 entries whose id is a captured ANSI color
  escape (`"\e[36mPMAT-498"`, all-null payload). **Fixed in this same change** (see §1 defect #5).

Two items also grounded *worse* than the review stated: **rank-7** — the README says 1331 provable
contracts while the real tree count has moved to **1766** (was 1460 at review HEAD; drift is growing,
not static); and the **P4 fail-closed 11-class headline is CONFIRMED intact** (7 weight + 4 embedding,
`beat_threshold: 11`, per-PR) — the marquee is real, not theater, and must not be weakened.

---

## 1. The backlog, mapped to the five pillars + infra

All 18 items are now in `docs/roadmaps/roadmap.yaml` (16 net-new appended 2026-07-05; ranks 2 and 8
were already tracked). Each carries a `RED-turning mutation` in its `acceptance_criteria` — the item
is not "done" until that mutation demonstrably turns the gate red.

| rank | id | pillar | type | prio | status | thrust |
|---|---|---|---|---|---|---|
| 1 | PMAT-CI-PASSGREP-001 | infra | gate | critical | planned | anchor the compute-step pass check (kill the `10 failed` bypass) |
| 2 | BEAT-OLLAMA-DECODE-CI-001 | P4 | gate | high | planned* | put the marquee 1.371× decode win on a scheduled workflow + coherence oracle |
| 3 | PMAT-F2-DECODE-PHASE-001 | P4 | gate | high | planned | probe ≥8 decode steps through the production prefill mode |
| 4 | PMAT-QA-FMTPARITY-DECODE-001 | P4 | gate | high | planned | give `apr qa` format_parity a real ≥64-token decode path |
| 5 | FALSIFY-CUDA-NF4-TRAIN-TRAJECTORY-001 | P3 | gate | high | planned | ≥20-step GPU-vs-CPU NF4 training trajectory (kill step-0-only) |
| 6 | PMAT-SERVE-MULTITURN-001 | P4 | gate | high | planned | real 3-turn `/api/chat` + streaming + template-engine gates |
| 7 | PMAT-DRIFT-GATES-001 | infra | contract | high | planned | wire the readme/book/CLI claims checker per-PR; correct the counts |
| 8 | PMAT-739 | P4 | beat | medium | planned* | decode 1.5× marquee (DP4A Q8_1) — rides behind rank 2's re-pin |
| 9 | APR-ANTHROPIC-MSGS-INTEGRITY-001 | P5 | beat | medium | planned | Anthropic `/v1/messages` + tool-call-integrity falsifier |
| 10 | PMAT-GNB-SPEED-DEFENSE-001 | P1 | beat | medium | planned | defend the GaussianNB speed beat (4.9×→2.12× decay) |
| 11 | PMAT-CUDA-NIGHTLY-PROBATIVE-001 | P3 | gate | medium | planned | probative skip-nights + GQA-64/AdamW/NF4 falsifiers |
| 12 | PMAT-CASEFILE-MECH-001 | cross | contract | medium | planned | mechanize #153 (≤8-byte detect) + #1599 (apr-cli optional) |
| 13 | PMAT-MACOS-WGPU-NIGHTLY-001 | infra | gate | medium | planned | execute macOS/Metal tests (currently build-only; T5 gap) |
| 14 | PMAT-MSRV-GATE-001 | infra | gate | medium | planned | MSRV 1.89 PR-blocking check (CI builds on stable/1.93) |
| 15 | PMAT-PUBLISH-POLICY-001 | infra | infra | medium | **blocked** | reconverge publish doctrine (8 versions/14 days behind) — operator-gated |
| 16 | PMAT-MUTANTS-FULLTREE-001 | infra | gate | low | planned | weekly full-tree mutation sweep (currently diff-only) |
| 17 | PMAT-THEATER-TRIAGE-001 | infra | gate | low | planned | triage ~595 unexecuted integration-test files |
| 18 | PMAT-COVERAGE-FLOOR-001 | infra | gate | low | **blocked** | enforce the ≥95% floor (enforced nowhere) — cross-repo |

`*` ranks 2 & 8 pre-existed in roadmap.yaml; this wave promotes/sequences them, does not re-add.

### Per-pillar frontier (review §STEP-2, reconciled to the current tree)

- **P1 (scikit-learn):** 6 accuracy/parity beats per-PR-blocking; 7 speed beats nightly-non-blocking.
  LinReg measured 2.73×. **Live risk:** the GaussianNB speed beat has decayed 4.9×→~2.12× against a
  2.0× gate (rank 10). No new *breadth* — the do-not-do list forbids estimator races.
- **P2 (PyTorch):** autograd-grad beat per-PR (|Δ|≈5e-7, enforced 1e-4); cross-entropy is an honest
  **L3 ceiling** (9 proved + 1 N/A < 12 obligations; the FP/SIMD/oracle obligations are non-analytic).
  Training throughput conceded (specific, sourced). **No backlog item — P2 is the healthiest pillar;**
  the guardrail is "don't race raw training throughput."
- **P3 (Unsloth/QLoRA):** NF4/merge/composed beats enforced; the 5-falsifier CUDA nightly on two
  silicon is **live and confirmed**. Frontier moved from "enforce at all" to "enforce **trajectories**,
  not step-0" (rank 5) and "make skip-nights probative" (rank 11).
- **P4 (Ollama/llama.cpp):** fail-closed **11-class** per-PR headline is **intact** (do not weaken).
  The decode/serve *diagnostics* are the theater surface: ranks 2/3/4/6. The 1.5× marquee (rank 8) is
  real but multi-week and pointless against a dead baseline — it rides behind rank 2's re-pin.
- **P5 (parity / agents):** harness-ir contract at L5. The competitive flank moved: mistral.rs ships
  Anthropic `/v1/messages` + an agent loop **with zero correctness contracts** — rank 9 ships the
  endpoint *with* a tool-call-integrity falsifier (the wedge nobody else claims) and feeds the
  `APR-ANTIGRAVITY-PARITY-001` either-harness epic already atop the planned queue.
- **Infra (the enforcement backbone):** ranks 1, 7, 12, 13, 14, 15, 16, 17, 18 — this is where the
  review found the most leverage, because every pillar's gate credibility transits it.

### New defects the review surfaced (fold into rank 1 / 7 / 17)

1. `ci.yml:288` compute pass-check substring-matches `10 failed` (rank 1). **Grounded true.**
2. sovereign-ci `ci/test` job is vacuous for the root facade; fmt + cargo-deny are `continue-on-error`.
3. Coverage ≥95% enforced **nowhere** (rank 18).
4. ~595 integration-test files compile-checked only, never executed (rank 17).
5. **Roadmap health (FIXED here):** the "152 `inprogress`" items were **corruption** — every one had
   an ANSI color escape captured into its id (`"\e[36mPMAT-498"`) with all-null payload; there were
   **zero** legitimate inprogress items. This change removes all 152 and restores `inprogress: 0`
   (the #2275 target). 595 legitimate entries preserved byte-for-byte; 16 net-new appended. A durable
   guard against re-corruption belongs in rank 7 (a roadmap-lint step that rejects non-ASCII/control
   bytes in ids).

---

## 2. Sprint sequencing (EV-ordered, across the 4-host fleet)

Honors `blocked_by`, the single-physical-line `ci.yml:317` constraint (all per-PR gate additions =
ONE consolidated PR), and the never-idle all-hosts mandate (`feedback_all_silicon_hosts_never_idle`).

- **Wave A — enforcement backbone (days 1–3).** Rank **1** first (one-line, critical, unblocks the
  credibility of every other gate), then rank **7** (drift-gates + roadmap re-reconciliation) and
  rank **4** (`apr qa` real decode). Rank 1 is a standalone small-CI-edit PR; ranks 4/6a/7 that touch
  `ci.yml:317` land as **one** consolidated PR.
- **Wave B — P4 decode/serve + P3 GPU correctness (days 3–6).** Ranks **3** (F2 multi-step probe),
  **6** (serve multi-turn), **2** (decode CI + coherence oracle) on the GPU fleet; ranks **5** (NF4
  trajectory) and **11** (probative nightly) wired into `cuda-nightly.yml`.
- **Wave C — P5 / P1 / portability (days 5–8).** Rank **9** (Anthropic messages integrity), rank
  **10** (GaussianNB defense — five-whys the decay), ranks **13** (macOS/Metal), **14** (MSRV),
  **12** (case-file mechanization).
- **Wave D — infra hardening + marquee (days 7–10).** Ranks **16** (full-tree mutants), **17**
  (theater triage; blocked on 11), **18** (coverage floor; cross-repo), **8** (1.5× decode marquee;
  multi-week), **15** (publish policy; operator GO/NO-GO with clean-room preflight first).

### Host assignment

| host | silicon | wave work |
|---|---|---|
| lambda-vector | RTX 4090 (sm_89 CUDA) | ranks 5, 11 (leg 1), 2, 3 GPU legs; P1 CPU beats |
| gx10 | GB10 Blackwell (sm_121) | ranks 5, 11 (leg 2); cuda-oxide; real GPU training |
| intel | AMD Vulkan / clean-room | CPU infra gates (ranks 1, 7, 14, 16); wgpu backend |
| mini | Apple M4 Metal | rank 13 macOS/wgpu-Metal nightly |

---

## 3. Do-not-do (review §7c) — the EV guardrails

- **P1 estimator-breadth race** — sklearn's pedagogy moat isn't attackable head-on; only
  contract-backed correctness/speed beats enter.
- **LAPACK-bound speed contests** (Ridge/Lasso/KMeans/PCA vs MKL sklearn) — documented narrow losses
  (PCA ~18.6×, Lasso ~19×); no falsifiable beat available.
- **P2 raw training-throughput race** vs PyTorch/MKL — conceded (~11×); burn's distributed push
  doesn't change the calculus.
- **P3 Triton GPU tok/s race** vs Unsloth — conceded; the correctness/trajectory gates are the ground.
- **P4 datacenter-scale concurrent serving** (vLLM class) — conceded segment.
- **Apple-silicon decode-speed war** vs Ollama-MLX/M5 — no winnable beat with the current fleet; ship
  the T5 build/**test** smoke (rank 13), not a speed war.
- **P5 feature-parity chasing** Claude Code / the mistral.rs agent loop — only the through-line +
  tool-call-integrity contract (rank 9).
- **Prose rebuttals to rvLLM's "parity-proven" branding** — answer with L5 proof levels + CI-enforced
  falsifiers, not marketing.
- **APR-BOOK archive epics (PMAT-497..503)** as near-term work — deferred; zero beat value now.
- **Merging status-flip-without-artifact** (the `fix-pmat-737`-as-roadmap-flip class) — a status flip
  with no implementation is the exact theater the doctrine bans. *(Note: #2294 as merged into #2286
  now carries a real Q4_K pre-interleave implementation, `3a8c92f67` — the review's earlier flag on it
  is discharged.)*
- **Re-litigating APR-MONO** (crates.io trueno, standalone-repo refs) — settled; residues are cleanup.

---

## 4. Definition of done for the wave

Every backlog item lands as a PR whose body reproduces the `RED-turning mutation` from its
`acceptance_criteria` and **demonstrates the gate turning red under that mutation** before the fix.
"Green after fix" alone is insufficient — a gate that was never shown to go red is theater by
construction. The one deliberately gated item is rank 15 (publish), blocked on an operator GO/NO-GO
by design, with the clean-room `cargo publish --dry-run` preflight named first.

---

## 5. Provenance

- **Source review:** `docs/specifications/fable-architectural-review.md` (#2292, 2026-07-05).
- **Grounding workflow:** `wf_103658cf-4d2` (5 agents, 2 passes; 21/23 checks CONFIRMED_STILL_TRUE —
  all 18 backlog items actionable on the current tree 2026-07-05; 1 UNVERIFIABLE (no network), 1
  CHANGED (roadmap corruption, fixed here)).
- **roadmap.yaml (net −136 items):** removed 152 ANSI-corrupted `inprogress` entries (restoring
  `inprogress: 0`) and appended 16 net-new Fable-review items (ranks
  1,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18); ranks 2 (`BEAT-OLLAMA-DECODE-CI-001`) and 8 (`PMAT-739`)
  were already tracked. The 595 legitimate entries were preserved byte-for-byte (surgical block
  filtering, NOT a re-serialization — avoids the lossy-reformat class from the wave-5 conflict).
  Final: 611 items, 0 duplicate ids, 0 corrupt ids, re-parsed clean.
