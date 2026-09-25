# PRM-001 (was REX-001): PROMETHEUS — review-lane experiment protocol (Qwen 3.5 4B on apr, across fleet devices)

**Spec id:** `PRM-001` (was `REX-001`; row ids unchanged) · **Rows:** `REX-00..REX-12` (the traffic cop mints one `pmat` ticket per row; single-minter rule applies)
**Target repo:** `paiml/aprender` (`docs/specifications/review-experiment-protocol.md`)
**Runner:** the aprender traffic cop (`aprender-traffic-cop-prompt.md`). It mints the tickets, directs the workers through issues, and owns the pre-registration commit and the final report.
**Launch:** from `~/src/aprender`, run `Implement docs/specifications/review-experiment-protocol.md autonomously.`
**Related:** ARB-APR-001 (arbiter on apr), ARB-SELF-001 (dogfood feedback loop), `stack-30-day-plan.md` §4.4 (trustworthy review), #3558 (Pareto model-size selection rule), APR-PERF-GATE-001 (perf arms, receipt transport J3).
**Status:** spec, not implemented. Dated 2026-09-25.

**Provenance marks:** `[V]` verified at the cited sha/time · `[C]` computed · `[A]` asserted · `[U]` unverified/unmeasured · `[X]` third-party.

---

## §0 Operating assumptions

1. **This is a scientific experiment, not a benchmark run.**
   - Hypotheses, metrics, analysis plan and decision rules are frozen in REX-00 before any measurement.
   - Data gathered under an earlier version of §3–§6 is labelled `exploratory` and never feeds a decision.
2. **The question has two parts:**
   - (A) which device cell should host the Qwen 3.5 4B review lane now, and how its speed improves each release;
   - (B) how the model improves continuously: prompt, decoding, fine-tune, distill, merge.
   - The experiment answers (A) and builds the measuring instrument for (B).
3. **Dogfood starts on day 0, alongside the experiment.**
   - The 4B lane joins the quorum as a **4th, non-voting shadow lane** on a provisional cell (gx10, per the 2026-09-25 ruling).
   - The experiment's hardware ruling (REX-08) supersedes the provisional cell.
4. **Producer is never the gate.**
   - The review lane runs a **released, sha-verified `apr` tag**, never aprender HEAD.
   - The model weights are pinned by sha256.
   - The lane's vote stays asymmetric until it is promoted: an apr dissent sends a 2–0 to human review (HRQ), and apr never breaks a 1–1.
5. **A refusal is a result, not an error.**
   - A cell whose backend refuses the model (for example WGPU or Metal before 0.71) is recorded as `Refused{removed_by}` in the report and filed as an aprender issue.
   - It is never retried under another backend while keeping the same cell name.
6. **Genchi genbutsu.** Every §1 value is re-measured at HEAD in Phase 0 before any row relies on it. A row whose premise is false at HEAD is recorded `premise-falsified` and skipped.
7. **Cross-repo work is filed, not done.**
   - Executor declarations (infra/forjar) and quorum wiring (paiml-implement) are filed as issues with exact acceptance criteria.
   - Cells that depend on them are `NotRun{NoDeclaredExecutor}` until those issues land.
   - Rows that don't depend on them proceed.
8. **The operator has no pre-steps.** Anything only Noah can do is a §8 STOP condition.

---

## §1 Ground truth (baseline, frozen; never quote as current)

| # | Fact | Value | Mark | Source of truth |
|---|---|---|---|---|
| G1 | n=1 pilot, same diff and prompt | Qwen3.5-4B Q4_K_M on `apr` v0.69.3 release binary (checksum `8a67a010`), lambda CPU at load 33–40: **175.8 s**, PASS. Haiku 4.5 via `claude -p`: **17.8 s**, PASS. GPU rerun skipped (GPU lock held > 60 s) | [V] (operator-pasted report, 2026-09-25) | the pasted report; diff `0c6932fd5` |
| G2 | What G1 can establish | A PASS on a correct diff says nothing about defect recall. Precision and recall of the 4B lane, and of every existing voter, are **unknown** | [C] | — |
| G3 | Audited `apr` surface relevant to this spec | `apr run --backend wgpu` q=2, hw UNKNOWN · `apr serve run` q=4 · `apr finetune` q=2 · `apr distill` q=9 · `apr merge` q=6 · `apr quantize` q=4 · `apr parity` q=9 · `apr train apply` q=1 · `mcp:apr.serve` q=1 | [C] at audit snapshot; [U] at HEAD | `docs/audits/surface_audit.csv` |
| G4 | Train themes | 0.69 = verbs that are true; 0.70 = fast (TTFT, resident serve, `--json-schema`); finetune/distill on qwen35 refuse until 0.71; WGPU/Metal refuse until 0.71 | [A] operator ruling 2026-09-20; [U] at HEAD | `docs/specifications/06x-release-schedule.md`, milestones |
| G5 | Parity oracle | llama.cpp pin `d1d3c3396` (`scripts/llama_pin.toml` `build_commit`); `ds.yaml` min_cosine 0.98 | [V] 2026-09-20; [U] at HEAD | `scripts/llama_pin.toml`, `ds.yaml` |
| G6 | TTFT gap | apr 0.68.2 TTFT 7.7×, end-to-end 4.9× slower than llama.cpp on Qwen3.5-4B, 4090 | [V] 2026-09-20 | #3596 |
| G7 | Host roles | lambda-labs = agent host, primary x86 CUDA (scarce); **its GPU is excluded from this experiment by construction** (shadow rung 2 only, R-9). intel = clean-room runner (contended), dual AMD GPU (Vulkan). gx10 = GB10, 120 GB unified, clean-room pool. mini = M4, macOS, `rust-neutral` pool | [A] | `infra/machines/*/forjar.yaml` |
| G8 | Undeclared apr | lambda-labs `~/.local/bin/apr` was a hand-installed 0.64.0 | [V] 2026-09-20; [U] at HEAD | `command -v apr`; `fleet-bins.tsv` |
| G9 | Reviewer identity | 301/301 historical quorum receipts record the reviewer model as `unknown` | [V] 2026-09-10 | PV-LEAN-AUDIT |
| G10 | Decode ceilings (bandwidth ÷ bytes/token) | 4B Q4_K_M ≈ 2.5 GB/token → 4090 ≈ 400, GB10 ≈ 110 tok/s | [C] theoretical, [U] for apr | — |

---

## §2 Design

### §2.1 Factors

**Device cell** (the primary factor). The same model, apr tag, prompt and decoding settings run in every cell.

| Cell | Host | Backend | Notes |
|---|---|---|---|
| C1 `intel-wgpu` | intel | wgpu (Vulkan, dual AMD) | likely `Refused` before 0.71 (G4). The refusal is recorded and filed |
| C2 `intel-cpu` | intel | cpu (AVX-512) | clean-room contention: shares the intel concurrency group (APR-PERF-GATE-001 §4.9.2) |
| C3 `lambda-cpu` | lambda-labs | cpu | GPU excluded. Co-located with arbiter and agent sessions |
| C4 `gx10-cuda` | gx10 | cuda (sm_121) | provisional shadow host. Must yield when `train-active` is set |
| C5a `mini-cpu` | mini | cpu (NEON) | |
| C5b `mini-metal` | mini | metal | likely `Refused` before 0.71 (G4) |

**Reference cell:** C2 `intel-cpu`. It is the cross-cell invariance reference, and its logits are also checked against llama.cpp `d1d3c3396` CPU `[X]`.

**Model arm:**
- Primary: `Qwen3.5-4B-Q4_K_M`, sha pinned.
- Control: `Qwen3.5-9B-Q4_K_M`, **test split, run only on the reference cell and the top-ranked cell.** This control makes the "4B is enough" claim falsifiable (H3).

**External baselines**, reference cell only, once per test item:
- Haiku 4.5 (`claude -p`);
- the agy lane.

Both record their exact model ids. They measure the existing voters on the same corpus. **This is the first recall and precision measurement of the current quorum.**

**Fixed across all runs:**
- a versioned prompt file, sha recorded; initially the G1 prompt ("VERDICT: PASS or FAIL, then at most 5 short bullet findings");
- greedy decoding (temperature 0, fixed seed);
- a fixed `max_tokens`;
- the context policy: diffs are never truncated silently. A diff that exceeds the context is `NotRun{ContextOverflow}` and counted as such.

### §2.2 Corpus (`docs/audits/review-corpus/`)

| Class | Construction | Label source | Target n `[A]` |
|---|---|---|---|
| **P** planted | the fleet's existing mutation RED fixtures, applied as diffs | known defect plus its file:line | 50 |
| **R** real | **revert-the-fix**: take a merged fix PR that has a linked defect issue and reverse-apply its fix hunks, which reintroduces a real, historical defect | the fix hunk's file:line | 50 |
| **G** good | merged PRs with a green clean-room, no revert, and no `regression` label for 14 days `[A]` | clean | 50 |

- **Strata:** diff size S (< 2k tokens), M (2–8k), L (> 8k), balanced across classes where the population allows. The per-stratum n is reported.
- **Splits:**
  - `dev`: 30%, stratified. Prompt iteration is allowed on it.
  - `test`: 70%, **sealed**. Its item hashes are committed in REX-00. Its contents are read only by the scoring harness, never by a prompt author, a training job or a model-selection loop.
- **Sample-size basis `[C]`:** 70 defect items in test give a Wilson 95% CI half-width of ≈ 0.117 on recall at p = 0.5. If the pilot (REX-05) projects a half-width > 0.12, the pre-registered response is to grow the corpus. The hypotheses are not changed.
- **Contamination contract:**
  - The test-item hashes must not appear in any training set (§6).
  - A diff hash overlap is a hard FAIL (`review-corpus-contamination-v1`).
- **Test-set rotation:** a sealed test version supports **at most 20 promotion decisions `[A]`**. Then 50% of its items are rotated out and replaced from new R/G specimens. The rotation is logged, and the old version is kept for trend comparability.

### §2.3 Metrics (all deterministic; no model judges)

| Metric | Definition |
|---|---|
| Verdict | parsed from the first `VERDICT: (PASS\|FAIL)` line. Unparsable → `Unparsed`, which counts as incorrect and is reported separately |
| Parse rate | parsed ÷ attempted |
| Recall | FAIL verdicts on P∪R ÷ \|P∪R\| |
| Precision | FAIL verdicts on P∪R ÷ all FAIL verdicts |
| False-refute rate | FAIL on G ÷ \|G\| |
| Localization | on P∪R FAILs: a finding names the defect's file path (string match). This is the "useful dissent" rate |
| Cross-cell invariance | the fraction of items whose verdict and output bytes equal the reference cell's |
| Determinism | a 10% random re-run on the same cell gives byte-identical output |
| Load, TTFT, prefill tok/s, decode tok/s | taken from `apr` timing output (`--json` timing, or the serve response metadata) |
| Review wall-clock | request sent → verdict parsed, per item |
| Peak memory | cgroup anon, not `memory.peak` (09-18 measurement method) |
| Host load | loadavg1, busy runners (via the `fleet-bin.sh` oracle), co-running jobs; recorded per run |
| Interference cost | the slowdown of a fixed reference workload co-located with the serve, compared with running alone (§3 H6) |
| Energy | RAPL (intel, lambda) / `nvidia-smi` (gx10) / `powermetrics` (mini) where available, otherwise `NotRun{NoSensor}`. Reported only, never used in a decision |

### §2.4 Execution discipline

- **Run order:** randomized per cell with a fixed seed. Cells run concurrently on their own hosts.
- **Cold vs warm:**
  - The first request after the unit starts is recorded as `cold` and reported separately.
  - All other items are measured warm against a resident `apr serve`.
- **Receipts:**
  - One JSONL row per (item, cell, arm), conforming to `review-experiment-receipt-v1`.
  - Required fields: item id+sha, class, stratum, split, cell, host, backend, apr tag+binary sha, model id+weights sha, prompt sha, decoding params, token counts, all timings, verdict, raw-output path+sha, load snapshot, corpus version, prereg sha.
  - A missing field makes the receipt inadmissible, and an inadmissible receipt is never a pass.
- **Transport:** hosts push signed receipts (APR-PERF-GATE-001 §4.9.1, J3). The analysis job runs anywhere and verifies the signature and freshness.

---

## §3 Pre-registered hypotheses and tests

The family for Holm correction is H1–H6, overall α = 0.05. All CIs are 95%.

| # | Hypothesis | Test | Decision consequence |
|---|---|---|---|
| H1 | **Invariance:** on every parity-admissible cell, 4B verdicts and output bytes equal the reference cell's on every test item | a count of divergent items; any count > 0 rejects H1 | each divergence becomes an aprender parity issue with the item attached. **A divergent cell is not admissible for (A)** |
| H2 | **Determinism:** same cell, same item → byte-identical output | a count on the 10% re-run; > 0 rejects | a non-deterministic cell is inadmissible until fixed |
| H3 | **Size:** 9B per-item correctness exceeds 4B's on the test split | McNemar exact, paired, one-sided | if not rejected, 4B stays the lane model. If rejected, the §5 promotion rule decides whether 9B replaces it, weighing its measured wall-clock |
| H4 | **Usefulness vs voters:** 4B precision ≥ the lowest precision among the existing voting lanes (Haiku, agy) on the same items | a paired bootstrap CI of the difference (10k resamples, seeded); H4 holds if the CI lower bound ≥ 0 | this is the gate from shadow to **tripwire** (dissent may trigger HRQ) |
| H5 | **Recall vs voters:** 4B recall ≥ the Claude lane's recall | same test | this is the gate from tripwire to **full vote** |
| H6 | **Interference:** the serve on cell X slows the host's reference workload by more than run-to-run noise | the reference workload runs 5× alone and 5× co-located; noise = 2 × sd(alone) `[C at measurement]` | a cell whose serve slows a gated CI job or the arbiter host beyond noise is ranked below every cell that doesn't |

Descriptive (not tested): the per-cell p50/p95 wall-clock with bootstrap CIs, TTFT, and tok/s, each compared against llama.cpp `d1d3c3396` running the same model on the same cell `[X]`.

**Sensitivity analysis:** the primary analysis uses all runs, because that is the deployment reality. A secondary analysis restricted to loadavg1 < 0.5 × physical cores is reported beside it. G1's "unfair CPU load" objection is answered by that second column, not by excluding data.

---

## §4 Pre-registered decision rule: hardware for the review lane (Goal A)

This rule is lexicographic and fixed in REX-00.

1. **Admissible:**
   - a parity receipt passes (`apr parity` vs llama.cpp `[X]`, `ds.yaml` min_cosine 0.98);
   - H1 and H2 not rejected for that cell;
   - parse rate equal to the reference cell's;
   - the cell's executor is forjar-declared.
2. **Among admissible cells, rank by p95 review wall-clock** under natural load (primary analysis).
3. **Interference override:** a cell whose H6 is rejected ranks below every cell where it isn't.
4. **Ties** (overlapping p95 bootstrap CIs) go to the host with the larger measured 7-day idle fraction. If that also ties: not in the clean-room pool > in it.
5. **Queue budget:** the chosen cell's p95 must be ≤ the slowest existing voting lane's measured p95 on the same items, so the local lane never lengthens the quorum.
   - If no cell meets it, the fastest admissible cell is selected **as shadow only**.
   - Every cell's gap vs llama.cpp is filed as an aprender perf issue.
6. **Output:** a **primary cell and a failover cell.**
   - If both are down, the lane refuses by name (`Unknown{LaneUnavailable}`). A quorum never silently runs below its width.
   - When `train-active` is set on the primary's host, the lane emits `NotRun{TrainActive}` and blocks rather than passing.

The ruling is committed as `docs/audits/rex-001-hardware-ruling.md` with the receipts attached. It supersedes the 2026-09-25 provisional gx10 placement only if it differs, and then through the infra issue that moves the declared unit.

---

## §5 Continuous improvement loop (Goal A and Goal B)

### §5.1 Performance ratchet (A, every aprender tag)

- On each tag, rerun the perf arm on the primary cell (the full `dev` split plus 30 fixed test items for timing only; the verdicts on them are already known):
  - with the new `apr` tag;
  - with llama.cpp `[X]` on the same cell.
- **Ratchet:** review p95 on the primary cell is never allowed to rise after 3 recorded tags.
  - A regression is an andon. It opens an aprender issue carrying both receipts.
- **Runtime knob sweep** on the primary cell only, once per train (0.70, 0.71, …):
  - thread count ∈ {physical cores, physical/2};
  - quant ∈ {Q4_K_M, Q8_0};
  - context/KV setting as exposed.
  - The winner is chosen by the §4 rule restricted to that cell. A quant change also requires the §5.3 quality test, because a new quant is a new model.

### §5.2 The data flywheel (B0; starts on day 0)

- **Every quorum, from every lane, writes a `review-ledger-v1` row:**
  - diff hash, repo, PR, lane verdicts and findings;
  - model ids, apr tag, weights sha;
  - final outcome: merged / reverted ≤ 14 d `[A]` / `regression`-labelled escape.
- **Label tiers:**
  - **Gold:** corpus labels, plus reverts and escapes joined back to the PR that introduced them. Gold is the only tier allowed in evaluation.
  - **Silver:** Claude + agy consensus verdicts and findings. Used for training only, and only if §8 S-6 is resolved. Never used for evaluation.
- The ledger joins ARB-SELF-001's outcome ledger (ASL-09) and escape join (ASL-12). The escape join files regression escapes as new R-class specimens, **so the corpus grows from real misses.**

### §5.3 Candidate improvements, in EV order

| Tier | Method | Available | Gate |
|---|---|---|---|
| B1a | Prompt versions (structure, rubric, output schema) | now | dev split, then sealed test (§5.4) |
| B1b | Retrieval few-shot: the k most similar gold specimens from the train pool (never the test split) | now | same |
| B1c | Constrained decoding once `apr` ships `--json-schema` (0.70) | 0.70 | parse rate → 1.0 is the expected effect; confirmed by the same test |
| B2a | **Logit distillation from a local teacher:** Qwen3.5-27B on gx10 writes top-k logits at m = 1 over the train pool; the 4B student trains with `apr distill` | when `apr distill`/`finetune` accept qwen35 (0.71 per G4) | §5.4 |
| B2b | QLoRA fine-tune on gold-labelled train-pool items (`apr finetune`) | same | §5.4 |
| B2c | Merge: combine successive adapters or checkpoints (`apr merge` / `aprender-train-lora merge`) | same | §5.4 |
| B2d | Re-quantization of a promoted checkpoint (`apr quantize`) | now for existing weights | §5.4 plus §5.1 |

- **B2 rows are aprender acceptance tests.** An `apr finetune`/`distill`/`merge` ticket on qwen35 counts as done only when it produces a challenger that passes §5.4 on the sealed test.
- That makes the review lane a standing consumer of the training verbs, which is the dogfood the 30-day goal I asks for.
- Until those verbs accept qwen35, B2 rows are `NotRun{VerbRefused(removed_by)}`, never skipped silently.

### §5.4 Champion/challenger promotion (pre-registered)

A challenger (any §5.3 candidate) replaces the champion only if all of these hold on the current sealed test version:

1. **Per-item correctness improves:** McNemar exact, paired, one-sided, α = 0.05.
2. **Precision point estimate is not lower** than the champion's, because precision is what spends Noah's HRQ time.
3. **Invariance (H1) and determinism (H2) hold** on the primary cell.
4. **Contamination contract green:** 0 test hashes in its training data.
5. **Wall-clock:** its p95 on the primary cell still meets the §4 queue budget.

- Promotion is a forjar pin bump of the weights sha (and the adapter sha, if any). Rollback is the previous sha.
- Each decision consumes one of the test version's 20 evaluations (§2.2).
- **Demotion:** 2 consecutive gold-labelled escapes in which the champion said PASS and a voter said FAIL trigger an automatic HRQ and re-evaluation.

---

## §6 Hard rules (each has a §9 falsifier)

- **R-1 Pre-registration is immutable.**
  - REX-00 commits §2–§5 and the analysis code; its sha goes into every receipt.
  - Any change creates a new spec version, and data collected under the old version is `exploratory`.
- **R-2 Sealed test.**
  - The test split is read only by the scorer.
  - A diff that reads test items from anywhere else (prompt building, retrieval, training, selection) fails CI (`review-corpus-contamination-v1`).
- **R-3 Released artifacts only.**
  - The lane and every cell run a released `apr` tag verified by sha256, and weights verified by sha256.
  - Never HEAD, never a hand-built binary.
- **R-4 Declared executors only.**
  - Every cell runs through a forjar-declared unit or a declared runner label. No ad-hoc SSH, and no hand-installed `apr` on any host.
  - An undeclared `apr` on an executing host is a STOP (S-2).
- **R-5 Receipts carry identity.**
  - 0 receipts may record `unknown` for model, weights sha, apr tag, host, backend or prompt sha.
- **R-6 NotRun is never a pass.**
  - `Refused`, `NotRun{…}`, `Unparsed` and `ContextOverflow` are each counted and reported.
  - None of them counts toward accuracy numerators.
- **R-7 No invented thresholds.**
  - Every numeric gate cites its measurement command or is marked `[A]`/`[U]`.
  - The `[A]` values in this spec (corpus sizes, 14-day revert window, 20-evaluation budget) are recalibrated after the first test version and logged.
- **R-8 No Python** in any file this spec creates, harness or analysis. Shell passes `bashrs`; everything else is Rust (an aprender workspace crate or xtask).
- **R-9 Lambda GPU is excluded, except as shadow rung 2 at dispatch time** (operator ruling 2026-09-25, option b; `cop-inbox/handoff/ruling-r9-b.md`).
  - No REX experiment cell, teacher or finetune job, resident `apr serve` or VRAM-preloaded weights may place work on the lambda-labs 4090. Every other R-9 exclusion stands.
  - The single exception: the apr SHADOW lane may run on `lambda-cuda` as rung 2, only when rung 1 (`gx10-cuda`) returned `NotRun` and the lambda GPU lock is free at dispatch time.
  - `lambda-cuda` yields to agent sessions and `train-active`. A lock taken before or during the round makes the row `NotRun{Busy}`, never a Verdict, and the ladder continues. The round is not retried on the same cell.
  - Fallback order: `gx10-cuda` → `lambda-cuda` → `intel-wgpu` → `mini-metal` → `intel-cpu` → `NotRun{NoExecutor}`. Each row records the backend that actually ran; the `gx10-cpu` labelling rule applies to lambda too.
  - The rung-2 routing lives in paiml-implement#436. Falsifier: a fixture holding the lambda GPU lock produces a `lambda-cuda` row of `NotRun{Busy}`.
- **R-10 The release train wins.** When `train-active` is set, the lane and the experiment jobs yield on clean-room/CUDA pools.
- **R-11 The asymmetric vote holds** until H4 and H5 pass through the §5.4 machinery. Before that, apr never casts a deciding vote.
- **R-12 Measurement states its tree:** HEAD vs `origin/main`, the host, and a per-session worktree.

---

## §7 Tickets (EV-ordered)

`K̂` is minutes `[A]`, recalibrated after the first three rows.

| EV | Row | Work | Contract | Done when (all must hold) | K̂ |
|---|---|---|---|---|---|
| 0 | **REX-00** pre-register | Commit this spec, the analysis plan (§3 tests as code signatures), the test-item hash manifest (empty placeholder until REX-02), prompt v1 and its sha | `rex-prereg-v1` | prereg sha recorded; a planted edit to §3 after the commit makes the prereg check fail | 45 |
| 0 | **REX-01** executor issues | File the infra issue: forjar declarations per host (intel, lambda-labs, gx10, mini) for the exact `apr` tag, weights (4B, 9B; 27B on gx10 only) by path+sha256, an `apr serve` unit with its own cgroup budget and health endpoint, `exec_sha256`, `train-active` yield, the intel concurrency group shared with clean-room, a non-clean-room label for intel. **gx10's declaration is marked priority 1 (day-0 shadow).** File the paiml-implement issue for the 4th shadow lane (`quorum.lane_models`) and ARB-APR-001's gx10 revision. Phase 0 first checks whether any existing declared label already admits these jobs | — | issue URLs recorded; each issue carries checkable acceptance criteria | 60 |
| 1 | **REX-02** corpus | Build P/R/G per §2.2, strata, splits, the sealed hash manifest, the contamination contract with fixtures | `review-corpus-v1`, `review-corpus-contamination-v1` | counts per class/stratum/split receipted; a planted test-item leak into a train file fails CI | 180 |
| 1 | **REX-03** harness | A Rust bench binary drives a resident `apr serve` over HTTP, applies the fixed prompt, parses verdicts deterministically, writes `review-experiment-receipt-v1` rows, and pushes signed receipts. The scorer implements §2.3 and §3 (Wilson, McNemar exact, seeded bootstrap, Holm) | `review-experiment-receipt-v1` (pv; falsifiers: missing model sha → inadmissible; `NotRun` counted as correct → rejected; `Unparsed` counted as PASS → rejected) | scorer unit-tested against hand-computed fixtures; each falsifier planted RED→GREEN | 240 |
| 2 | **REX-04** parity admission | `apr parity` per cell against llama.cpp `d1d3c3396` `[X]` on the exact weights; refusals recorded with `removed_by` | `rex-cell-admission-v1` | 6/6 cells are `Admitted`, `Refused{removed_by}` or `NotRun{NoDeclaredExecutor}`, each with a receipt; 0 cells silent | 120 |
| 2 | **REX-05** pilot | 10 dev items on every admitted cell, plus cold-start runs and the H6 reference workload alone ×5 | extends REX-03 | variance and runtime projection receipted; the sample-size rule from §2.2 applied and logged | 90 |
| 3 | **REX-06** full run | Test split on all admitted cells; 9B control on the reference cell; Haiku and agy baselines on the reference cell; 10% determinism re-run; H6 co-located runs | extends REX-03 | 100% of (item × cell × arm) have an admissible receipt or an explicit `NotRun`; 0 `unknown` identity fields | 240 |
| 3 | **REX-07** day-0 shadow lane | As soon as gx10's declaration lands: the 4B lane runs as a non-voting 4th lane on every aprender quorum and writes `review-ledger-v1` | `review-ledger-v1` | 100% of quorums after activation carry a shadow row (0 missing); a planted gx10-down leaves the quorum's width at 3, and the shadow row is written as `Unknown{LaneUnavailable}` | 90 |
| 4 | **REX-08** analysis + hardware ruling | Run the §3 tests and apply the §4 rule; commit `rex-001-hardware-ruling.md`; file one aprender issue per H1/H2 divergence and one per cell perf gap vs llama.cpp; file an infra issue if the primary cell ≠ gx10 | `rex-001-report-v1` | every §3 hypothesis has a verdict with its statistic and CI; primary and failover cells named, or "shadow only" with its reason | 150 |
| 5 | **REX-09** promotion ladder | Encode H4/H5 as the shadow→tripwire→vote gates in paiml-implement (filed issue) and arbiter `decide` (filed issue) | extends ARB-APR-001 | the gates read only `rex-001-report-v1` and later §5.4 reports; a planted report with H4 failing keeps the lane in shadow | 60 |
| 6 | **REX-10** perf ratchet | Per-tag §5.1 job on the primary cell; ratchet file; knob sweep per train | `review-lane-perf-ratchet-v1` | 3 consecutive tags recorded; a planted +10% p95 regression turns the andon RED | 90 |
| 7 | **REX-11** B1 loop | Prompt versions and retrieval few-shot evaluated via §5.4 | `review-champion-challenger-v1` | ≥ 1 challenger evaluated end-to-end with its receipt, promoted or rejected by rule; the test-version evaluation counter increments | 120 |
| 8 | **REX-12** B2 loop | Teacher logit generation (27B on gx10, top-k, m = 1) over the train pool; the B2a/B2b/B2c acceptance tests wired to the qwen35 finetune/distill/merge tickets | extends REX-11 | the teacher dataset is receipted (sha, count, 0 test hashes); B2 rows read `NotRun{VerbRefused}` until the verbs land, then run §5.4 automatically | 150 |

**K̂ = 1635 `[A]` · K = 1800 · andon at 1440.**

---

## §8 STOP conditions (exhaustive; stop, write the §9 report, do not work around)

- **S-1** REX-00 cannot be committed before any measurement (pre-registration out of order).
- **S-2** An executing host resolves `apr` to an undeclared binary, or its weights sha mismatches the declaration.
- **S-3** The sealed-test contamination check fails anywhere.
- **S-4** Any change would place work on the lambda-labs GPU outside the R-9 shadow rung-2 exception.
- **S-5** A run overlaps `train-active` on a clean-room/CUDA pool without yielding.
- **S-6** **Training on hosted-model outputs.** Using Claude or agy (Gemini) outputs as training data for B2 may conflict with those providers' terms. It is an operator decision, and until Noah rules, silver labels stay evaluation-irrelevant and unused for training. B2 proceeds with the local 27B teacher and gold labels only.
- **S-7** Every cell is `Refused` or `NotRun`, so there is no admissible cell for (A).
- **S-8** A harness hook refuses the session's writes for a reason other than a missing ticket.
- **S-9** Budget K is reached, or the andon is crossed with more than one row incomplete.
- **S-10** Two consecutive non-passing quorums on the same PR.

---

## §9 Final report schema (one per session)

```yaml
spec: REX-001
session: {date_utc, host, tree: {head, origin_main, worktree}, model}
prereg_sha: <sha>
selected_row: REX-NN
ticket: PMAT-NNNN
outcome: merged | stopped | already-done | premise-falsified
stop: {id: S-n, evidence: "..."}          # only when stopped
baseline_remeasured: [{row: G#, value, mark, command}]
cells:
  - {cell: C1..C5b, status: Admitted|Refused|NotRun, reason, apr_tag, apr_sha, weights_sha, parity: {cosine, receipt}}
results:                                   # REX-08 onward
  hypotheses: [{id: H1..H6, statistic, ci, p_holm, verdict}]
  per_cell: [{cell, n, recall, precision, false_refute, localization, parse_rate,
              p50_s, p95_s, p95_ci, ttft_ms, prefill_tps, decode_tps, peak_anon_gb,
              llama_cpp_ratio, interference: {slowdown, noise, rejected}}]
  baselines: [{lane, model_id, recall, precision, p95_s}]
hardware_ruling: {primary, failover, mode: vote|tripwire|shadow, reason}
promotion: {champion: {weights_sha, prompt_sha}, challengers: [{id, verdict, mcnemar_p, precision_delta}], test_version, evaluations_used}
issues_filed: [{repo, url, purpose}]
scoreboard_moved: [{metric, before, after, command}]
budget: {k_hat_row, k_actual_row, cumulative, andon_crossed: bool}
next_row: REX-NN
```

### §9.1 Scoreboard (targets and probes only; state is rendered, never written here)

| Metric | Target | Rendered by |
|---|---|---|
| Cells with an admissible receipt, a refusal or an explicit NotRun | **6/6**, 0 silent | `rex-001-report-v1` |
| Cross-cell verdict divergence on admitted cells | **0** | H1 |
| Receipts with `unknown` identity | **0** | receipt lint |
| Quorums carrying a shadow row after activation | **100%** | `review-ledger-v1` |
| Sealed-test contamination | **0** hashes | contamination contract |
| Review p95 on the primary cell | never-up after 3 tags | perf ratchet |
| Gap to llama.cpp on the primary cell | measured every tag; filed when it grows | perf ratchet |
| Champion correctness on the sealed test | never-down across promotions | §5.4 reports |
| Lane precision / recall | measured, then ratcheted never-down | §5.4 reports |
| Lambda GPU minutes consumed by this spec outside the R-9 shadow rung 2 | **0** | host receipts |
| Python lines in files this spec touches | **0** | `grep -rl python3` over the diff |
