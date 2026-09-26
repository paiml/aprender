# PRM-001 — PROMETHEUS: make local Qwen 3.5 fast enough, and good enough, to break quorum ties

**Spec id:** `PRM-001` v3 (Prometheus). **Supersedes:** PRM-001 v2, REX-001 / PRM-001 v1 (experiment protocol), and PRA-001 (trace-corpus design). Old row ids are kept in the §7 `was` column.
**Target:** `paiml/aprender` → `docs/specifications/PRM-001-prometheus.md`. This is the single source of truth.
- The infra copy becomes a one-line pointer.
- The v1 file `review-experiment-protocol.md` stays byte-frozen in §2–§5, so its v1 lock still verifies. It gains one superseded banner above §0.

**Runner:** the aprender traffic cop, sole minter. Tracking is one sub-epic under E4 (#4381) per APR-EPIC-001 §4d, with its phases replaced by §7 here. Epic #4354 (rows #4355–#4367) is collapsed into its checklist.
**Launch:** from `~/src/aprender`, run `Implement docs/specifications/PRM-001-prometheus.md autonomously.`
**Figure:** `docs/specifications/figures/prometheus-flow.png`. Re-export it: title "PROMETHEUS", remove the silver pool, and add the tie-breaker rung.
**Related:**
- E2 Verbs Are Fast (0.71) and E6 llama.cpp Parity (0.73): Prometheus is their real-workload acceptance test.
- ARB-APR-001, ARB-SELF-001, EXT-001, FLOW-001, APR-PERF-GATE-001 (J3 receipts).

**Provenance marks:** `[V]` verified · `[C]` computed · `[A]` asserted · `[O]` operator ruling · `[U]` unmeasured · `[X]` third-party. No invented thresholds.

---

## §0 Operating assumptions

1. **One goal.** Qwen 3.5 4B on `apr` becomes the **4th-lane tie-breaker**. That needs three things:
   - **fast:** it never lengthens a quorum;
   - **available:** it answers ≥ 95% of rounds;
   - **measured:** its accuracy on sealed gold is at least that of the voters it would overrule.
   Speed is necessary; it is not sufficient.
2. **Speed is the critical path.** Review is prefill-heavy: a diff of 2k–32k tokens goes in, and a verdict plus at most 5 bullets comes out `[U until PRM-S2]`. The binding constraints are prefill and TTFT, not decode.
3. **The shadow has zero weight until promoted** `[O 2026-09-25]`.
   - Every quorum receipt carries an apr row (`Verdict | Refused | NotRun{…}`); a receipt without one is RED.
   - The apr verdict never blocks, breaks a tie or escalates until §5.4 grants a rung.
4. **Hosted-model outputs are never training targets** `[O 2026-09-25]`.
   - Claude and agy/Gemini outputs are captured and stored (you own your outputs). They are used only as **replay inputs, latency baselines and evaluation comparators**.
   - Training uses gold labels plus locally generated rows only: Qwen's own outputs and the local 27B teacher.
   - This makes the program clean under Anthropic's help-center rule on Outputs, Anthropic Consumer §3(2) and Commercial §D.4, and the Google ToS ban on "using AI-generated content … to develop machine learning models" (§1 G19). It replaces the S-6 ruling.
5. **Capture first.** A quorum whose I/O is not captured at dispatch is lost for replay, evaluation and outcome joins. The order is capture → measure → speed → promote.
6. **A scientific experiment.** §2–§5 are frozen by the pre-registration lock `rex-prereg-v2` (PRM-00b), which locks **this v3**, before any measurement. v1 collected 0 data rows `[V]`.
7. **Producer is never the gate.**
   - Released, sha-verified `apr` tags and sha256-pinned weights only.
   - The writer of a trace never labels it.
8. **A refusal is a result.** `Refused{removed_by}` is recorded and filed, never retried under the same cell name.
9. **Genchi genbutsu.** Re-measure §1 at HEAD before relying on it. A false premise is recorded as `premise-falsified`.
10. **Foreign work is transferred.**
    - Capture at dispatch → paiml-implement #436.
    - Executors, the almacen store, the second copy and transcript retention → paiml/infra.
    - Speed defects are aprender issues in E2/E6; Prometheus links them and does not own them.
11. **The operator has no pre-steps.** Pure Rust + bashrs-clean shell, no Python; provisioning by `forjar apply` or make only.

---

## §1 Ground truth (frozen baseline; never quote as current)

| # | Fact | Value | Mark | Source |
|---|---|---|---|---|
| G1 | n=1 pilot | 4B Q4_K_M, apr v0.69.3, lambda CPU at load 33–40: **175.8 s**; Haiku 4.5: **17.8 s** (≈ 10×) | [V] | diff `0c6932fd5` |
| G2 | Recall and precision of every lane, the voters included | unknown | [C] | — |
| G3 | apr v0.69.3 sha256 | `8a67a0103cbb036332908cc184d5a8a8ae422fb9025bd5c78e42d65df86cfbd7` | [V] | REX-04 receipt |
| G4 | Weights sha256 | 4B `00fe7986…69ef11a4` · 9B `03b74727…2b7e8` · 27B `84b5f7f1…d2a806` | [V] | REX-01 receipt |
| G5 | Train themes | 0.70 fast (`--json-schema`, resident serve) · 0.71 wider + training (qwen35 finetune/distill; WGPU/Metal) | [O 2026-09-20] | `06x-release-schedule.md` |
| G6 | `apr parity` v0.69.3 | GPU-vs-CPU only; no released llama.cpp comparison | [V] | REX-04 premise check |
| G7 | rc.1 lane on gx10 | F2 guard (cos 0.8425) → CPU fallback (#4313); `--context-length` ignored, 262144 used (#4443); decode ≈ 0.5× ollama; **32k prefill 0.26×**; temperature-0 divergence at ≈ 55 tokens | [V] cop report | #4313, #4443, `evidence/4252-promotion.md` |
| G8 | TTFT at 0.68.2 on a 4090 | 7.7× slower than llama.cpp; end-to-end 4.9× | [V] | #3596 |
| G9 | Epic speed targets | E2: TTFT ≤ 2× llama.cpp · E6: decode, prefill, TTFT ≥ 1.0× llama.cpp across the certified matrix | [O] | APR-EPIC-001 §2 |
| G10 | Executors | apr pin on gx10, lambda-labs, yoga; no `apr serve` unit, weights or review label anywhere | [V] | infra#1088 |
| G11 | paiml-implement | v1.8.1 has no advisory lane; #420 unreleased | [V] | PRM report |
| G12 | Ledger | findings stored as a count; `weights_sha256: None` hardcoded; free-text `LaneUnavailable` | [V] | `crates/aprender-review-experiment/src/ledger.rs` |
| G13 | Sealed corpus | `review-corpus-v1@787d2026256cc08b`: test = 35 P (size S, mutants) + 35 R + 35 G | [V] | REX-02 receipt |
| G14 | Corpus diffs | single copy on a 94%-full raid (`items-v1.tar`, sha `70dffe56…8bba`) | [V] | REX-02 receipt |
| G15 | Branch | `rex/001` @ `da7b6386f` folded into **closed** #4317; nothing on `main` | [V] | review package README |
| G16 | Volume | 50–200 quorums/day; ≤ ~0.45 TB/yr | [O] / [U] | PRM-C14 |
| G17 | Storage | almacen 3×16 TB RAID 5; re-pull time and scrub policy unmeasured | [V] / [U] | `almacen-storage-integration.md` |
| G18 | Transcripts | Claude Code deletes after `cleanupPeriodDays` (default 30) | [X] | Claude Code docs |
| G19 | Terms | Anthropic: training allowed only for non-competing specialised tools; "using Outputs as training targets" is prohibited. Google ToS: flat ban on developing ML models from AI-generated content. Claude Code under Max = Consumer Terms. **Not legal advice.** | [X] | support.claude.com #12326764; anthropic.com/legal; policies.google.com/terms |

---

## §2 Design

### §2.1 Lane dispatch and states

Order `[O 2026-09-25, R-9 option b]`: `gx10-cuda → lambda-cuda → intel-wgpu → mini-metal → intel-cpu → NotRun{NoExecutor}`.

- **lambda-cuda is rung 2 and shadow only.**
  - Dispatch time only, and only when rung 1 returned NotRun **and** `/tmp/apr-gpu.lock` is free.
  - Never a resident serve, VRAM preload, experiment cell, teacher or finetune.
  - If the lock is taken mid-round, the row is `NotRun{Busy}` and the ladder continues.
  - **Once the lane holds tie-breaker or vote, lambda-cuda leaves the ladder.** A deciding lane runs only on declared, non-contended cells (§4.1).
- **Rows record the backend that actually ran** (`gx10-cpu` on the F2 fallback).
- **train-active falls through; it never blocks.**
- **One typed state enum**, shared by receipt-lint (#436) and the ledger: `Verdict{PASS|FAIL}` · `Refused{removed_by}` · `NotRun{NoExecutor|Busy|Timeout|ContextOverflow|TrainActive}`.
- **Timeout per rung** = 2 × that cell's p95 once it has ≥ 10 rows; until then 300 s `[U]`.

### §2.2 Capture (`agent-trace-v1`): the substrate for replay, evaluation and outcomes

- **What:** a `TraceSpan` in the quorum rail (#436) hashes the exact wire bytes and streams the full reply into a blob, for every lane of every round.
- **On failure:** `trace_status = capture_failed` raises an andon. The quorum never waits on capture.

| Field group | Contents |
|---|---|
| identity | `trace_id` (ULID), `quorum_id`, `round`, `lane`, `counted`, `provider`, exact `model_id` or weights sha, `access_channel`, `apr_tag`, `served_by`, `lane_state` |
| work unit | `repo`, `pr`, `head_sha`, `base_sha`, `diff_sha256`, `input_sha`, `input_parts`, `prompt_version` |
| result | `output_sha` (full, never truncated), `verdict`, `findings[{file,line_start,line_end,severity,category,text}]`, `parse_status` |
| **timing (first-class)** | `tokens{input,output,cached?}`, `latency_ms{queue,ttft,prefill,decode,total}`, `prefill_tps`, `decode_tps`, `load_snapshot` |
| local only | `logits_sha` → `sparse-logits-v1` (k=20 `[A]` + `logsumexp_full` + `topk_mass`); used only for local teacher → student |
| lifecycle | `outcome` ∈ {pending, merged, reverted_le14d, regression_escape, closed_unmerged}; `split_guard{repo#PR, split, dedup_cluster}`; `secret_scan`; `train_eligible` (computed by admission, §4.2) |

- Second stream: Claude Code transcripts (including `subagents/`) and agy logs, harvested hourly `[A]`.
- These are stored for audit and replay only. They are never training data (§0.4).

### §2.3 Storage

```
almacen:/traces/blobs/sha256/ab/cd/<sha>.zst       write-once; address = sha of uncompressed bytes
almacen:/traces/index/agent-trace-v1/YYYY/MM/DD.jsonl  append-only
almacen:/traces/transcripts/{claude-code,agy}/<host>/<sha>.zst
almacen:/traces/replay/review-replay-vN.jsonl       frozen replay sets (§2.4)
almacen:/rex-corpus/v1/items-v1.tar                 the sealed corpus leaves the 94% raid
```

- Keep forever; quarantine only, never delete.
- **Second copy:** nightly, incremental by sha, to separate media.
- **Weekly restore drill:** 100 random blobs `[A]`, 0 mismatches. RAID 5 is not a backup `[C: P(≥1 URE) ≈ 0.92 on a degraded 32 TB read at 10⁻¹⁴]`.
- **Transcript retention:** a forjar-managed Claude Code setting, `cleanupPeriodDays: 36500` `[A]`.

### §2.4 Replay benchmark (`review-replay-v1`): the speed instrument

- **Source:** real quorum diffs from the trace index. Inputs only; no hosted output enters the benchmark.
- **Stratification:** by input tokens, 2k / 8k / 16k / 32k.
- **Set size:** 200 items per version `[A]`. Frozen per version; a new version each train.
- **Exclusions:** sealed-test hashes and clusters.
- **Run on every `apr` tag,** on the primary cell and on llama.cpp `d1d3c3396` with the same GGUF and cell. Output is `review-replay-receipt-v1` rows.
- **Measures:** review wall-clock p50/p95, TTFT, prefill tok/s per stratum, decode tok/s, peak anon memory, parse rate, and the verdict-identity rate vs the prior tag.
- **Queue budget reference:** the p95 of the **slowest counted voter on the same diffs**, taken from captured `latency_ms.total`. The speed target is measured, not asserted.
- **Workload profile (PRM-S2):** the measured distribution of input and output tokens per lane. It confirms or falsifies §0.2 and weights the strata.
- **Consumers:** this receipt is the **acceptance workload for E2 and E6**. Their `done_when` numbers are read on replay-v1, not on synthetic pp512/tg128 alone.

### §2.5 Sealed evaluation corpus (`review-corpus-v1`)

- As built (G13).
- At most 20 promotion decisions per sealed version `[A]`, then 50% rotation.
- Escapes feed the next version, never test v1.
- `review-corpus-contamination-v1` covers exact hashes and near-dup clusters.

### §2.6 Experiment cells and arms

| Cell | Host | Backend | Notes |
|---|---|---|---|
| C1 `intel-wgpu` | intel | wgpu | `Refused` expected before 0.71; must be observed |
| C2 `intel-cpu` | intel | cpu | **reference cell** |
| C3 `lambda-cpu` | lambda-labs | cpu | the GPU is never an experiment cell |
| C4 `gx10-cuda` | gx10 | cuda sm_121 | primary candidate; `gx10-cpu` on F2 fallback |
| C5a/C5b | mini | cpu / metal | C5b `Refused` expected before 0.71 |

- **Arms:** 4B primary; 9B control (test split, reference cell plus top cell).
- **Evaluation comparators on the reference cell:** Haiku 4.5 and agy, exact ids. Evaluation only; no training.
- **Fixed:** prompt sha, greedy decoding, `max_tokens`. Overflow is `NotRun{ContextOverflow}`, never truncated.

### §2.7 Metrics (deterministic; no model judges)

| Metric | Definition |
|---|---|
| **Review wall-clock p50/p95** | request → parsed verdict, per item; primary speed metric |
| **TTFT; prefill tok/s by stratum; decode tok/s** | from `apr` timing and the serve metadata |
| **Speed ratio** | apr ÷ llama.cpp on the same cell and GGUF, per metric |
| Availability | Verdict rows ÷ rounds, rolling 7 d, by cause |
| Parse rate | parsed ÷ attempted |
| Recall, per class | FAIL on P ÷ \|P\|; FAIL on R ÷ \|R\| (reported separately) |
| False-refute rate | FAIL on G ÷ \|G\| (the HRQ cost; does not depend on prevalence) |
| Precision | descriptive |
| Localization | a FAIL finding names the defect's file |
| Cross-cell invariance | verdict identity with the reference cell (bytes are descriptive) |
| Determinism | byte-identical on a 10% same-cell re-run |
| **κ_err** | Cohen's κ on binary error indicators vs gold, for every lane pair (voter–voter and qwen–voter) |
| Interference | slowdown of a reference workload when the serve is co-located vs alone |

---

## §3 Pre-registered hypotheses (locked by `rex-prereg-v2`; Holm family H1–H6, α = 0.05)

| # | Hypothesis | Test | Consequence |
|---|---|---|---|
| H1 | **Verdict invariance** across admitted cells vs reference | count of divergent verdicts > 0 rejects | the cell is inadmissible; one issue per item |
| H2 | **Determinism** on the same cell | byte count on the 10% re-run > 0 rejects | the cell is inadmissible until fixed |
| H3 | **Size:** 9B correctness exceeds 4B | McNemar exact, one-sided | if rejected, §5.4 weighs 9B's gain against its wall-clock |
| H4 | **Precision ≥ min(Haiku, agy)** | paired bootstrap of `p4B − min(pH,pA)`, **min inside each resample**, 10k resamples, seed 4354; Holm-adjusted one-sided p decides; CI reported only | the tripwire gate |
| H5 | **R-class recall ≥ the Claude lane** | same machinery, R items only | the tie-breaker gate (with H4) |
| H6 | **Interference** | 5 runs alone vs 5 co-located; slowdown > 2 × sd(alone) rejects | a rejected cell ranks below every other cell |
| H7 | **Independence** (outside the Holm family; a gate) | κ_err(qwen, each voter) ≤ **max κ_err over voter–voter pairs** on the same items | the tie-breaker gate. It is δ-free: a tie-breaker may not be more correlated with a voter than the voters already are with each other |

- **Sensitivity:** all runs (primary) vs loadavg1 < 0.5 × physical cores (secondary). Decisions read the primary column.
- **Sample size:** after PRM-05, if the Wilson half-width on R-recall > 0.12, grow R; hypotheses unchanged.

## §4 Decision rules

### §4.1 Hardware (lexicographic)

1. **Admissible:** a parity receipt passes against its declared `parity.oracle` (`llama.cpp@d1d3c3396` once it is a declared input; until then `apr-parity-gpu-cpu` plus H1; `ds.yaml` min_cosine 0.98); H1 and H2 hold; the parse rate equals the reference cell's; the executor is forjar-declared; for tie-breaker or vote, the cell is **not** lambda-cuda.
2. Rank by replay p95 under natural load.
3. H6-rejected cells rank last.
4. Ties go to the larger 7-day idle fraction, then to a host outside the clean-room pool.
5. **Queue budget:** replay p95 ≤ the slowest voter's p95 on the same diffs (§2.4). A cell failing it is shadow-only.
6. **Output:** a primary cell and a failover cell. When both are down: `NotRun{NoExecutor}`. The quorum keeps width 3; a tie then goes to HRQ, as today.
7. Committed as `docs/audits/prm-001/hardware-ruling.md`.

### §4.2 Admission (`trace-admit`, a separate binary)

| Gate | Rule | Target |
|---|---|---|
| **G-PROV** | `train_eligible` only if provider = local (qwen, teacher-27b) **or** the row is a gold label (outcome, HRQ ruling, sealed-corpus label). Hosted outputs are never eligible | hosted rows in any training pool = **0** `[O]` |
| G-SEC | Rust scanner plus the gitleaks ruleset ported as data; a hit quarantines the row, with no redaction | 0 hit rows in a pool; 100% canary recall `[A]` |
| G-CON | 0 sealed hashes or near-dup clusters in a pool or replay set | 0 `[A]` |
| G-ID | provider, exact model id, access_channel, input/output sha, served_by, weights sha all present | 100% `[A]` |
| G-DUP | split by `repo#PR`, time-ordered, ≥ 14 d embargo, clusters closed to their earliest member | 0 cross-split clusters `[A]` |
| G-MAT | gold only after the 14 d outcome window; pending is never a negative | 0 immature gold `[A]` |
| G-HASH | blob re-hash equals its address | 0 mismatches `[A]` |

- **Gold** = sealed-corpus labels, matured outcomes, and HRQ rulings (from tripwire onward).
- **Public release** (EXT-001) = base weights plus artifacts trained on G-PROV-eligible rows only.

---

## §5 Improvement loop and promotion

### §5.1 Speed loop (first priority; every aprender tag)

| Lever | Owner | Expected effect |
|---|---|---|
| Keep the GPU path on gx10 (F2 guard) | #4313 | CPU → GPU; the largest single step |
| Honour `--context-length` | #4443 | right-sized KV |
| Prefill kernels and batched prefill | E2 → E6 | the dominant term (§0.2) |
| Resident `apr serve` plus a prefix cache for the fixed system prompt | E2 | TTFT |
| `--json-schema` | 0.70 | bounded output; parse rate → 1.0 |
| Knob sweep (threads; Q4_K_M vs Q8_0; KV) once per train | PRM-10 | per-cell optimum; a new quant goes through §5.4 |

- **Ratchet:** replay p95 on the primary cell must never rise after 3 tags. A +10% planted regression turns RED (paired Harrell–Davis p95, landed).
- Every tag files one aprender issue per stratum where the apr/llama.cpp ratio worsened.

### §5.2 Quality loop (no hosted-output targets)

| Tier | Method | Available |
|---|---|---|
| Q1 | prompt versions | now |
| Q2 | retrieval few-shot from **gold** train-pool specimens (never the test split) | now |
| Q3 | constrained decoding (`--json-schema`) | 0.70 |
| Q4 | **local logit distillation:** Qwen3.5-27B on gx10 writes `sparse-logits-v1` over G-PROV-eligible inputs; the 4B student trains with forward-KL plus tail-mass correction; reverse-KL/GKD as challengers, using Qwen's own shadow outputs as on-policy samples | qwen35 `apr distill`, per D-7 |
| Q5 | QLoRA on gold labels only (`apr finetune`); merge (`apr merge`); re-quantize (`apr quantize`, now) | per D-7 / now |

- Q4/Q5 are aprender acceptance tests: a qwen35 distill/finetune ticket is done only when its challenger passes §5.4.
- Until then the row reads `NotRun{VerbRefused(removed_by)}`.

### §5.3 Champion/challenger (pre-registered)

A challenger replaces the champion only if all of these hold on the current sealed version:
1. McNemar exact, one-sided, α 0.05: correctness improves.
2. The false-refute rate is not higher.
3. H1/H2 hold on the primary cell.
4. H7 holds.
5. G-CON and G-PROV hold on its training data.
6. Replay p95 meets the §4.1.5 queue budget.

- Promotion is a forjar pin bump (weights and adapter sha); rollback is the previous sha. Each decision uses 1 of the 20.
- **Demotion:** 2 consecutive gold escapes where the champion's verdict decided the round → automatic HRQ and re-evaluation.

### §5.4 Promotion ladder (`rex ladder`, contract `rex-promotion-ladder-v1`, extended by PRM-09)

| Rung | Authority | Speed gate | Quality gate (sealed gold) |
|---|---|---|---|
| **shadow** (default; anything unreadable) | none | — | — |
| **tripwire** | a dissent against a 3–0 opens an HRQ, capped at D-8/day | replay p95 ≤ queue budget | H4 |
| **tie-breaker** | decides only a **1–1** split (a counted lane NotRun, or a 2-counted quorum) | queue budget **and** availability ≥ 95% over 7 d `[O cop crit. 4]`, on a non-lambda cell | H4 **and** H5 **and** H7, with H1/H2 on the primary cell |
| **vote** | counted voter | same | same, plus ≥ 20 decided tie-breaks `[A]` whose 14 d outcome agreement ≥ the voters' agreement on the same rounds |

- The ladder reads only `prm-001-report-v1` and §5.3 reports carrying the `rex-prereg-v2` sha.
- It never skips a rung.
- A demotion drops exactly one rung.

---

## §6 Hard rules (each has a §9 falsifier)

- **R-1 Pre-registration is immutable.** `rex-prereg-v2` locks v3 §2–§5, `stats.rs`, analysis plan v2 and prompt v1. Any change is a new version, and earlier data is `exploratory`.
- **R-2 Sealed test.** Only the scorer reads it; it never appears in replay sets or pools.
- **R-3 Released artifacts only.** apr tag and weights by sha256.
- **R-4 Declared executors only.** No ad-hoc SSH or hand-installed apr.
- **R-5 Identity.** 0 `unknown` identity fields in receipts and ledger rows; `Coverage::holds()` requires `identity_gaps == 0`.
- **R-6 NotRun is never a pass,** and never in an accuracy numerator.
- **R-7 No invented thresholds.** `[A]` values are recalibrated after the first test version.
- **R-8 No Python.**
- **R-9 Lambda GPU.**
  - 0 minutes for experiment cells, teacher, finetune and resident serve.
  - Rung-2 shadow dispatch only (§2.1), with its minutes measured.
  - 0 minutes once the lane holds tie-breaker or above.
- **R-10 train-active falls through, never blocks.**
- **R-11 No authority without a rung.** The lane decides nothing beyond its §5.4 rung.
- **R-12 Measurement states its tree.**
- **R-13 Capture is complete.** One `agent-trace-v1` row per lane call, NotRun included, with the full output and findings text.
- **R-14 Producer never labels.**
- **R-15 No secret in any pool or replay set;** no in-place redaction.
- **R-16 Blobs are write-once,** with a second copy within 24 h.
- **R-17 One spelling per lane state.**
- **R-18 Hosted outputs are never training targets.** G-PROV = 0 at every admission, in every pool manifest and in every release artifact.

---

## §7 Tickets (EV-ordered)

**Phases** (these replace APR-EPIC-001 §4d P0–P6):

| Phase | Train | Content |
|---|---|---|
| **P0** | now | capture + shadow |
| **P1** | 0.71 (E2) | speed instrument + ratchet |
| **P2** | 0.71 | experiment + hardware ruling |
| **P3** | 0.72 | tripwire |
| **P4** | 0.73 (E6) | tie-breaker; prefill parity |
| **P5** | D-7 | local distill (Q4/Q5) |
| **P6** | 0.75 (E8) | vote |

| EV | Row | Was | Phase | Work | Contract | Done when | K̂ | Status |
|---|---|---|---|---|---|---|---|---|
| 0 | PRM-00 | REX-00 | — | v1 prereg | `rex-prereg-v1` | — | — | **DONE** `5f0ae10ac` |
| 0 | **PRM-00b** | new | P0 | lock v3: §2–§5, `stats.rs` (min inside resample, R-class H5, H7 max-pair rule), analysis plan v2 | `rex-prereg-v2` | sha recorded; a planted §3 edit fails; 0 rows exist under v1 | 60 | todo, before any measurement |
| 0 | **PRM-L** | new | P0 | land `rex/001` plus this spec on `main` (own PR); banner on v1 file | — | merged; `rex prereg-check` green on `main` | 45 | todo |
| 0 | PRM-01 | REX-01 | P0 | split infra#1088: **gx10 slice** (4B weights by sha, `apr-review-serve` unit, cgroup, health, `exec_sha256`, train-active yield) as P0; corpus tar and replay sets to almacen | — | slice issue filed with checkable criteria | 20 | DONE → split todo |
| 0 | PRM-C1 | PRA T1 | P0 | capture at dispatch, incl. `latency_ms{queue,ttft,prefill,decode,total}` | `agent-trace-v1` | **transferred → pi#436**; F-R13 green there | — | foreign |
| 0 | **PRM-C2** | PRA T2 | P0 | `review-ledger-v2`: findings text, trace shas, weights sha, typed states | `review-ledger-v2` | FALSIFY: no weights sha → `holds()=false`; free-text reason → parse error; v1 fixtures parse | 120 | todo |
| 0 | PRM-C3–C5 | PRA T3–T5 | P0 | almacen store, second copy, transcript retention | `trace-blob-store-v1`, `trace-backup-v1`, `transcript-retention-v1` | **transferred → infra**; drill 0 mismatches; 31-day canary survives | — | foreign |
| 0 | **PRM-07** | REX-07 | P0 | shadow activation | `review-ledger-v2` | on gx10 slice + pi release: 100% of receipts carry an apr row; planted gx10-down → width 3, `NotRun{NoExecutor}` | 90 | active on its 2 deps |
| 1 | **PRM-S1** | new | P1 | replay benchmark `review-replay-v1` + runner (apr vs llama.cpp, same cell and GGUF) | `review-replay-v1` | 200 items, 4 strata, 0 sealed overlap; receipt per tag; queue budget derived from captured voter p95 | 150 | todo |
| 1 | **PRM-S2** | new | P1 | workload profile: input/output token distribution per lane | extends `agent-trace-v1` | §0.2 confirmed or falsified with receipt; strata weights set | 60 | todo |
| 1 | PRM-10 | REX-10 | P1 | perf ratchet wired to replay; per-tag issue per worsened stratum; knob sweep per train | `review-lane-perf-ratchet-v1` | 3 real tags recorded; planted +10% RED | 60 | gate landed `634916cbd` |
| 1 | PRM-02 / 03 | REX-02 / 03 | — | corpus v1; harness + scorer | — | — | — | **DONE** `8185ad1c3` / `0ab6ce731` |
| 2 | PRM-04 | REX-04 | P2 | cell admission re-run under the `parity.oracle` field | `rex-cell-admission-v1` | ≥ 1 cell `Admitted` or a receipted refusal | 30 | DONE (S-7) → re-run |
| 2 | PRM-C6 | PRA T6 | P2 | secret gate plus backfill | `trace-admission-secret-v1` | AWS-key literal plus 20 canaries quarantined | 180 | todo |
| 2 | PRM-C7–C9 | PRA T7–T9 | P2 | contamination on clusters; MinHash dedup; `repo#PR` time split | `trace-dedup-v1`, `trace-split-guard-v1` | perturbed sealed item refused; 0 cross-split clusters | 360 | todo |
| 2 | PRM-C10 | PRA T10 | P2 | outcome join (14 d) plus HRQ rulings as gold | `trace-outcome-join-v1` | 3-day-old gold refused; weekly matured count | 150 | todo |
| 2 | **PRM-C13** | PRA T12–T13 | P2 | G-PROV gate plus κ_err probe (all pairs) | `lane-independence-v1`, `trace-admission-prov-v1` | a planted hosted row in a pool → RED; κ matrix `[V]` | 120 | todo |
| 2 | PRM-C14 | PRA T14 | P2 | datacard (Croissant+RAI) plus weekly yield/bytes receipt | `trace-datacard-v1` | G16 `[U]` → `[V]` | 120 | todo |
| 3 | PRM-05 | REX-05 | P2 | pilot | extends receipt | sample-size rule on R-recall applied | 90 | parked (prep `469bf0c98`) |
| 3 | PRM-06 | REX-06 | P2 | full run plus comparators | extends receipt | 100% receipts or explicit NotRun; 0 unknown identity | 240 | parked |
| 4 | PRM-08 | REX-08 | P2 | analysis plus hardware ruling | `prm-001-report-v1` | H1–H7 verdicts; primary/failover or shadow-only with reason | 150 | parked |
| 5 | **PRM-09** | REX-09 | P3/P4 | ladder: add the **tie-breaker** rung; wire pi (#428) and arbiter (infra#1095) | `rex-promotion-ladder-v2` | planted reports: H4 only → tripwire; H4+H5+H7 → tie-breaker; lambda primary → refused | 60 | gate landed `4a4318f1c` → extend |
| 6 | PRM-11 | REX-11 | P3 | Q1–Q3 loop (prompt, gold few-shot, json-schema) | `review-champion-challenger-v1` | ≥ 1 challenger end-to-end; counter increments | 120 | gate landed `d8696650e` |
| 7 | PRM-C11 | PRA T11 | P5 | `sparse-logits-v1` in `apr serve` rows (local only) | `sparse-logits-v1` | header + residual mass round-trip | 180 | todo |
| 7 | PRM-12 | REX-12 | P5 | Q4/Q5: 27B teacher on gx10 → 4B student; gold QLoRA | `review-b2-loop-v1` | teacher dataset receipted (0 test hashes, 0 hosted rows); `NotRun{VerbRefused}` until D-7's train | 180 | gate landed `da7b6386f` |

**Budget:** remaining aprender K̂ = **2,585 min `[A]`** `[C: sum of the rows above]` · **K = 2,840** (1.1 × K̂) · **andon at 2,275** (0.88 × K̂). Foreign rows are outside K.

---

## §8 STOP conditions (exhaustive)

- **S-1** A measurement before the `rex-prereg-v2` lock.
- **S-2** An undeclared `apr`, or a weights sha mismatch, on an executing host.
- **S-3** Any sealed hash or cluster in a pool or replay set.
- **S-4** Lambda GPU work outside rung-2 shadow, or any lambda work at tie-breaker or above.
- **S-5** A run overlaps train-active on a clean-room/CUDA pool without yielding.
- **S-6** **A hosted-model output appears in any training pool, manifest or release artifact (R-18).**
- **S-7** No admissible cell for the hardware ruling. Capture, shadow and P0/P1 rows continue.
- **S-8** A harness hook refuses writes for a reason other than a missing ticket.
- **S-9** K is reached, or the andon is crossed with more than one row incomplete.
- **S-10** Two consecutive non-passing quorums on the same PR.
- **S-11** A secret must be purged from all copies.
- **S-12** Almacen free < 20%, or the second-copy target is missing from forjar inventory.
- **S-13** Any change grants authority outside §5.4.
- **S-14** Transcripts contain third-party personal data beyond author identities.

### Operator decisions

| ID | Question | Status / default |
|---|---|---|
| R-9 | lambda GPU | **decided: option b**, shadow only (§2.1) |
| Training data | hosted outputs as targets | **decided: never** (§0.4, R-18) |
| D-7 | qwen35 finetune/distill: 0.71 (G5) or 0.75 (E8)? | default 0.71 under E4 as the PRM-12 acceptance test; E8 keeps multi-engine CRUX |
| D-8 | tripwire HRQ cap per day | none; blocks P3 only |
| D-10 | move the counted Claude lanes from Max OAuth to API keys (Commercial Terms; Max limits assume "ordinary, individual usage") | none; does not block Prometheus |

---

## §9 Final report schema and scoreboard

```yaml
spec: PRM-001
version: 3
prereg_sha: <rex-prereg-v2 sha>
session: {date_utc, host, tree: {head, origin_main, worktree}, model}
selected_row: PRM-NN
ticket: PMAT-NNNN
outcome: merged | stopped | already-done | premise-falsified | foreign-linked
stop: {id: S-n, evidence, resume_when}
baseline_remeasured: [{row: G#, value, mark, command}]
lane: {rung: shadow|tripwire|tie-breaker|vote, rows_7d, availability_7d, by_state: {}, by_served_by: {}, lambda_gpu_min}
speed:                                   # PRM-S1 / PRM-10, per tag
  replay_version: review-replay-vN
  apr_tag: vX.Y.Z
  cell: <primary>
  p50_s: ; p95_s: ; p95_ci: []
  queue_budget_p95_s: ; within_budget: bool
  ttft_ms: {apr: , llama_cpp: , ratio: }
  prefill_tps: {"2k": {apr: , llama_cpp: , ratio: }, "8k": {}, "16k": {}, "32k": {}}
  decode_tps: {apr: , llama_cpp: , ratio: }
  parse_rate: ; verdict_identity_vs_prev_tag:
workload: {input_tokens_p50_p95: {}, output_tokens_p50_p95: {}}
corpus: {rows_7d, completeness_7d, capture_failed_open, gold_matured_7d, hosted_rows_in_pools: 0, quarantined_7d, restore_drill_mismatch}
cells: [{cell, status, reason, apr_tag, apr_sha, weights_sha, parity: {oracle, cosine, receipt}}]
results: {hypotheses: [{id: H1..H7, statistic, ci, p_holm, verdict}],
          per_cell: [{cell, n, recall_P, recall_R, precision, false_refute, localization, parse_rate, p50_s, p95_s, ttft_ms, prefill_tps, decode_tps, peak_anon_gb, llama_cpp_ratio, interference}],
          comparators: [{lane, model_id, recall_P, recall_R, precision, false_refute, p95_s}],
          kappa_err: [{pair, kappa, n}]}
hardware_ruling: {primary, failover, rung_eligible, reason}
promotion: {champion: {weights_sha, adapter_sha, prompt_sha}, challengers: [{id, verdict, mcnemar_p, false_refute_delta, kappa_err}], test_version, evaluations_used, tie_breaks_decided, tie_break_outcome_agreement}
issues_filed: [{repo, url, purpose}]
budget: {k_hat_row, k_actual_row, cumulative, andon_crossed}
next_row: PRM-NN
```

| Scoreboard metric | Target | Rendered by |
|---|---|---|
| **Replay p95 on the primary cell** | ≤ slowest-voter p95 (measured); never up after 3 tags | PRM-S1 / PRM-10 |
| **TTFT ratio vs llama.cpp** | ≤ 2× by E2 (0.71) `[O]` | replay receipt |
| **Prefill ratio vs llama.cpp, per stratum** | ≥ 1.0× by E6 (0.73) `[O]` | replay receipt |
| Decode ratio vs llama.cpp | ≥ 1.0× by E6 `[O]` | replay receipt |
| Verdict availability, rolling 7 d | ≥ 95% before tie-breaker `[O]` | ledger |
| Receipts carrying an apr row | **100%** | receipt-lint |
| Closes decided by apr above its rung | **0** | receipt audit |
| Rows whose cell ≠ the backend that ran | **0** | `served_by` audit |
| Capture completeness; `capture_failed` open > 24 h | **100% / 0** | trace index |
| Hosted rows in any training pool or release | **0** | G-PROV |
| Secrets / sealed hashes / cross-split clusters in pools or replay | **0 / 0 / 0** | §4.2 |
| Restore mismatches; blobs without a second copy after 24 h | **0 / 0** | backup |
| Lambda GPU minutes outside rung-2 shadow | **0** | host receipts |
| Champion sealed-test correctness | never down across promotions | §5.3 |
| κ_err(qwen, voter) | ≤ max voter–voter κ_err | H7 |
| Python lines in files this spec touches | **0** | diff scan |

**Five whys (why Qwen cannot break a tie today):**
- A review takes minutes, not seconds (G1: ≈ 10× Haiku).
- The GPU path falls back to CPU (#4313), and prefill runs at 0.26× (G7).
- Prefill was optimised against synthetic pp512, never against the real review workload, and the lane waited on a 4-host executor batch.

Terminal causes, as mechanisms:
- **no real-workload speed gate** → fixed by PRM-S1 as the E2/E6 acceptance test;
- **batch-coupled executor dependency** → fixed by PRM-01, one slice per executor.
