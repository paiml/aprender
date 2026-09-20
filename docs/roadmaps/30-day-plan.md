# aprender — 30-day plan v2 (2026-09-20 → 2026-10-20)

**Status:** DRAFT v2.2 for team review · prepared 2026-09-20 · owner: Noah Gift
**Architecture:** aprender's place in the stack is modelled in `docs/architecture/stack-architecture.yaml` (infra); see `stack-30-day-plan.md` §1.5. Three consequences for this plan are in §5.12.
**P0 as of 2026-09-20 09:52Z (operator ruling): arbiter runs on apr Qwen 3.5 — dogfood is P0.** See §5.0. It outranks everything except Band 0 (slice M) and the tag-path move, and it does not wait for either.
**Supersedes:** v1, which a 3/3 adversarial quorum returned `do-not-implement-as-written` (`aprender-30-day-plan-quorum-review-2026-09-20.md`). §11 lists every change and every rejected finding with its reason.
**Companion:** `stack-30-day-plan.md` (fleet-wide). This file covers `paiml/aprender` and the infra/pmat work it directly depends on.

**Provenance marks** — `[V]` verified by command · `[C]` cited from a session report, issue or the arbiter status report · `[A]` operator directive or estimate · `[U]` unknown, must be measured before anyone builds on it. Every number is re-verified at HEAD before it enters a gate. If a number here disagrees with the box, believe the box.

---

## 0. Summary

Thirty days, three session slots, a release train every 72 h. The month is a dependency spine, not a list: put the tag path under review first (days 1–3, nothing beside it), make Qwen 3.5 **pass a named verb set with receipts**, put internal consumers on receipted cells immediately, get the ontology checker measured fleet-wide and into users' hands, then ratchet performance from a measured baseline. Qwen 3.5 training is descoped to **ruled plans + a CPU reference** this month.

**The v1 defect, stated once:** v1 wrote the shipping doctrine (fix-or-refuse) into the success criteria, so a release that refused everything would have been green on day 30. **Refusal is a shipping behaviour. It never appears in a success condition.** Every goal below has a floor on the *fix* side and a positive control.

---

## 1. Goals — floors and positive controls

| | Goal | Floor (must be true; no disjunctions) | Positive control (proves the instrument can fail) | Refusals |
|---|---|---|---|---|
| **A** | Internal projects run on aprender Qwen 3.5 | ≥ 3 named consumers (quorum lane, arbiter, RAH or apex) send ≥ N real requests/day `[U — set from week-1 traffic]` to **Qwen 3.5 on `apr serve`**, 7 consecutive days, **receipted cells only** | A planted failing request must appear in the dogfood ledger and open exactly one issue | n/a |
| **B** | Qwen 3.5 works | The blessed model **passes the named verb set V** (§5.3) with receipts on every CUDA host class | **Differential corpus**: known-answer prompts vs pinned llama.cpp; one deliberately corrupted kernel must turn the corpus RED | allowed only outside V; counted and reported |
| **C** | Ontology checker works, improves models/training | ≥ 80 % of a **frozen** capability ledger passes on the **installed, fleet-pinned** `pv`; ≥ 1 model or training artifact rejected by a shape in CI | `pc_shape` plant yields exactly one violation every run; `shapes_n > 0`, `focus_nodes_n > 0` | n/a |
| **D** | cuda-oxide used intelligently | Compute-sanitizer nightly over 100 % of CUDA falsifiers; 1 kernel with an independent second implementation | `initcheck` reproduces the known FP8 stale-read RED | n/a |
| **E** | Fleet utilised, infra fixed | commit→tag ≤ 30 min cold; `ci / gate` p95 ≤ 20 min; under **queue-driven** load (synthetic load excluded) every pool ≥ 0.8 busy/online; 3 consecutive measurements `[A, set 09-18]` | Busy sensor validated against a host known to be running a job | n/a |
| **F** | WGPU, Apple silicon, ARM + x86 GPU | **≥ 1 parity receipt per backend family** (CUDA-x86, CUDA-arm, WGPU-AMD, Metal) for the blessed model, single-stream | Same differential corpus, per backend | remaining matrix cells; counted |
| **G** | Formats and conversions work | **G1 container:** same-quant `gguf ↔ apr ↔ safetensors` is **bit-exact per tensor**. **G2 requant:** cosine ≥ the `ds.yaml` threshold on the live ggml types at the pin | One byte flipped in one tensor turns G1 RED | per type, counted |
| **H** | Performance parity | **One definition:** ratchet from the week-1 measured ratio toward 0.8 × pinned llama.cpp (W1, APR-PERF-GATE-001) per host; never down. No day-30 absolute gate. Fine-tune ratio: Qwen2.5 only this month | Gate refuses a run whose `build_identity` join key mismatches | n/a |
| **I** | bashrs/pmat-grade reliability for stakeholders | **Using the open half of the stack only** (`apr`, `pv`, pmat, bashrs, forjar — no arbiter, no harness): clean host, `install.sh` → serve → fine-tune (Qwen2.5) from the docs alone, in the T-2 dogfood; defect-escape rate recorded per release | A docs step deliberately broken in a fixture fails the dogfood | distill + Qwen 3.5 fine-tune: refuse, counted |

---

## 2. The spine (corrected)

```
tag path in repo (days 1–3) ──► every train
E (fleet + train code) ──► everything              gate latency IS release throughput
fleet pv pin ──► C (ontology) ──► G (formats validated) ──► B (Qwen 3.5) ──► A ──► I
                                   B (receipted cells only) ──► A starts day 1 at m=1
B forward parity (GPU) ──► DeltaNet CPU backward ──► [next month] Qwen 3.5 QLoRA ──► distill
H measures B.   D, F are enablers.
```

A does not wait for all of B: it consumes **only cells that already hold a parity receipt**, at m = 1. That is a constraint, not a circularity.

---

## 3. Doctrine (rulings of 2026-09-20)

Each lands on `main` **with its enforcer in the same PR**. Two rules were cited all day that are not on `main` (`06x-release-schedule.md` §1.1a; APR-RELEASE-001 §1.5).

1. **Fix-or-refuse is how we ship, never how we measure.** A capability ships with a receipt or refuses in one line + exit code. Loud, receipted degradation is allowed; silent downgrade is not. No success condition may contain "or refuses".
2. **Refusals carry `removed_by: v0.N` and never block a cut.** At T-0 an expired refusal auto-bumps one train and increments `slips`; `slips ≥ 2` raises a non-blocking andon naming the ticket. **A refusal is the safe state — it is never auto-reverted.**
3. **Must-carry admission = readiness.** Plan ruled + falsifier proven satisfiable by a RED test on `main`, before the train opens. ≤ 1 per train.
4. **Oracle by reference** — `scripts/llama_pin.toml` `build_commit`; never a literal sha in normative text; evidence records keep `comparator_sha`.
5. **Three-valued verdicts** in release-train code (ONT-6 lattice): crash / missing input / host-never-asked → `Unknown{reason}`, exit 2, excluded from the meet.
6. **Retries are visible** — `{step, attempt, input_sha, gate_sha, rc}`; red→green on unchanged inputs is `Unknown{RerunOnly}`.
7. **Interrupt displacement is mechanical** — `next_departure = max(scheduled, close_of_last_interrupt_release + one takt)`. Otherwise dates do not move; scope does.
8. **Each train pays its own reconcile** — own-train R2/R3/R4 = 0 at the cut; inherited debt never up.
9. **Exclusions are predicates**, not prose (§6).
10. **No threshold without its committed measurement command.**
11. **Status cells are rendered from probes, never typed.**
12. **Currency by reference** — docs include the blessed-model file; a lint refuses literal "current model" names.
13. **The producer is never the reviewer.** The apr quorum lane runs release N-1, blessed model pinned by sha, on a receipted cell — never HEAD.

Standing: `pmat work add` first (driver session only) · feature branch + PR, `ci / gate` · TDD, mutation RED→GREEN in the PR body · five-whys to a mechanism · one provable contract per feature · `pmat query` over grep · `forjar apply` / make targets only · **publishing: clean-room hard gate first, then the cascade from a detached checkout of the promoted tag; no workflow runs `cargo publish`.**

---

## 4. Slots

| Slot | Days 1–3 | Days 4–10 | Days 11–20 | Days 21–30 |
|---|---|---|---|---|
| **1 · Correctness** | slice M (Band 0) | slice M → G1 bit-exact gate → verb set V on CUDA | batched decode parity plan ruled, then work · DeltaNet **CPU backward** | G2 requant gate · plans ruled for Qwen 3.5 QLoRA + offline distill |
| **2 · Train + fleet** | **tag path into the repo — nothing beside it; 0.69 T-0 refused until merged** | four defect PRs · sanitizer nightly · busy-sensor check · queue drain (red / armed-unmergeable aprender PRs) | F: one receipt per backend family · cuda-oxide probe (one kernel) | H ratchets armed · work the measured bottleneck |
| **3 · Dogfood (P0) + ontology** | **P0: §5.0 — constrained decoding ticket + plan; parse-rate measurement; lambda-labs parity receipt.** Ontology rows stay blocked until the shared-contract-file decision (§7.8); fleet `pv` pin bump proceeds | ONT-4d (verify) → ONT-4c × 3 · dogfood ledger with circuit breaker · apr quorum lane in shadow | planted-defect corpus · A at 3 consumers · ONT-11 (cross-repo) | capability ledger ≥ 80 % · I clean-host path |

---

## 5. Workstreams

### 5.0 P0 — arbiter on apr Qwen 3.5 (operator ruling 2026-09-20)
`arbiter decide` now runs one agy, one Claude, one apr (infra PR 787, spec decision D-15.12). Today it **refuses `engine apr unavailable`** on its host — so the verb's availability is 0 until the rows below land. That refusal is the dogfood working; closing it is P0.

**Getting it live needs no new aprender code.** Order:

| # | Row | Repo | Done means |
|---|---|---|---|
| 1 | I2-5c: `apr` as a declared forjar resource on the agent's host — **exact version `0.68.2`, not a floor** — applied with `forjar apply -f machines/lambda-labs/forjar.yaml` | infra | `apr --version` on the agent's PATH resolves to the declared resource; the hand-installed `~/.local/bin/apr` (0.64.0, 2026-08-24) is gone. Identity = crate tarball sha (APR-PERF-GATE-001 §4.2.2); binary sha is host-local anti-substitution only |
| 2 | Weights declared by path + sha256 beside it (`Qwen3.5-9B-Q4_K_M`, **provisional** until the pre-registered selection rule picks) | infra | `arbiter decide` refuses on a sha mismatch, by name, zero dispatches |
| 3 | Parity receipt for that model/quant on the agent's host | aprender | `evidence/…/<host>.json` committed; rule 13 satisfied (receipted cell) |
| 4 | Parse-rate measurement: replay the `decide` fixtures through the apr lane; lenient **deterministic** extraction (strip fences, first JSON object) — not a rerun | infra | unparsable rate recorded per lane; `decide` availability is a number |
| 5 | **Asymmetric vote** (approved): apr's dissent converts 2–0 into an HRQ; apr never breaks a 1–1 split — a strong-lane split is an HRQ | infra | fixture: 1–1 + apr either way → escalate; 2–0 + apr dissent → escalate; 3–0 → decide |
| 6 | Self-tests run with network and GPU denied (sandbox), so a fall-through to a real engine fails loudly | infra | removing any stub turns the self-test RED without dispatching |
| 7 | PATH inventory: every executable on the agent's PATH resolves to a forjar-declared resource | infra | planted undeclared binary is reported |
| 8 | **Schema-constrained decoding in `apr`** (`--json-schema` or grammar) — the fix at the source, and the mechanism tool calling needs anyway | aprender, **P0 label**, slot 3 | the apr lane's unparsable rate → 0 by construction; contract + mutation RED→GREEN |

Expect rows 1–3 to collide with open 0.69 defects — the release installer asset 404, and the `-cuda` asset reporting `NotCompiled` from `apr devices`. **That is the point: each collision is a P0 for the train, found by our own consumer before a stakeholder finds it.**

Priority ≠ must-carry. Row 8 was not ruled before 0.69 opened, so by rule 3 it is not 0.69 must-carry; it ships in the first train where it is green. Rows 1–7 do not wait for it.

### 5.1 Band 0 — PP-QUANT-001 slice M (0.69 must-carry)
Ruling recorded: one `#[repr(u32)] GgmlType`, home `aprender-quant`. Falsifier: `TRAITS[i].{blck_size,type_size}` equals the value extracted from ggml at the pin by a committed command; `fixture.sha == llama_pin.toml build_commit`. Known mismatches `[C]`: Q4_1 18 vs 20 bytes/block; Q8_1 40 vs 36 — each reconciliation row gets NEVER-WORKED / REGRESSION measured on v0.68.1; fix-or-refuse per type on load. Row count 43 (35 + 8) is `[U]` until re-extracted at the pin. Follow-up: two comparator pins in normative text (216 vs 124 refs `[C]`); the CPU-leg parity basis sits on the superseded one.

### 5.2 Release-train integrity — **first in the month**
Four defects in never-reviewed code `[C]`: stale STOP line · missing `mkdir` reporting four hosts FAILED that were never asked · `ledger.py grab()` first-match on an append-only file · publish gate printing `GO` on a failed check.
1. Verbatim move into `scripts/release/`, zero behaviour change, `sha256sum` per file in the PR body.
2. One PR per defect, RED fixture first; every verdict function gets a first-Unknown fixture.
3. Self-identification: script sha + clean tree in the ledger; T-3 refuses otherwise.
4. §1.1a lands here, with `scripts/check_milestone_cut.sh` reading it.
Reconcile at v0.68.2 committed RED: R2 25 · R3 37 · R4 12 · R1 0 · R5 0.7241 `[C]` — **R5 needs a committed definition.**

### 5.3 Qwen 3.5 correctness (B)
**Verb set V (proposed — team to amend):** `apr devices` · `apr run` (CPU and CUDA, prints `Backend:` and tok/s) · `apr serve` single-stream · tool calling · **schema-constrained JSON output** (§5.0 row 8) · gguf load of the blessed quant. Each passes the differential corpus on every CUDA host class.
Open P0s `[C]`: batched decode garbage for m > 1 · cuBLAS route diverges at m ≥ 4 · FP8 prefill GEMM stale read · DP4A GEMV through Gated DeltaNet on sm_89 · `apr devices` says `NotCompiled` on a CUDA build. Order: FP8 stale read → re-measure batched → DeltaNet GEMV → name the blessed model. Receipts per `(model, host, route, m)`.

### 5.4 Ontology (C) — `~/src/infra/docs/specifications/paiml-ontology.md` v4.9
- **Row zero: the fleet `pv` pin is 0.65.2 and predates the shapes gate — every `--shapes` gate fleet-wide reports UNMEASURED `[C]`.** Bump through all four pin layers before any breadth row. "Running on aprender main" is not "measured on the fleet".
- State `[C]`: 25 rows, K̂ 1950; 11 bound (37 %); 1,300 contracts; 6 of 9 entity types.
- Breadth: ONT-4d (check R-19 has not already discharged it) → ONT-4c (README, llm-context, CSV; one PR each; this *is* the docs-validated goal) → ONT-11 (`pmat sync`, cross-repo — start the pmat ticket now) → ONT-10 (CLI rides the cascade; depth gates declared and unarmed, printed by `pv proof-status`; needs ruling).
- ONT-4c3 held on `evidence/kernels/gated_rmsnorm/lambda.json` (0 files today); owner: the CUDA lane / the cuda-oxide probe.
- One baseline (the ledger), one definition of 80 % (capability ledger, frozen N, installed binary). Spec §0.1 needs a dated `[A]` amendment.
- `contracts/census.json` and `contracts/lint-baseline.json` are rewritten by ONT rows **and** every PP-QUANT contract PR × one-PR-at-a-time × 62-min p95 = gridlock. **Slot 3 does not start until this is decided.**

### 5.5 Internal dogfood (A) and the apr quorum lane
- Consumers use **receipted cells only**, m = 1, labelled.
- Ledger: consumer · model sha · host · m · requests · failures · issue ids. **Circuit breaker:** dedupe by failure signature, one rolling issue per signature, daily cap per consumer, single-stream quarantine for any unreceipted path.
- **Consumer #0 — `arbiter decide`** already runs apr as its third engine (§5.0), under the asymmetric vote. It is live the moment I2-5c lands.
- **Consumer #1 — the apr lane in the PR-merge quorum** (paiml-implement; target: one agy, one Claude, one apr). This one stays staged, because a PR quorum gates every merge in every repo:
  - Phase 0 shadow: 4th non-voting lane on every quorum, ledgered. Wired via `quorum.lane_models` in paiml-implement, own ticket, separate session.
  - Phase 1 calibrate: replay a **planted-defect corpus** (existing mutation RED fixtures + known-good PRs) through every lane. No lane's recall is known today.
  - Phase 2 promote: apr replaces one lane when its planted-defect recall ≥ the weakest current voter on the same corpus, and the `Unknown{LaneUnavailable}` fallback is specified. A quorum never silently runs under width.
  - Receipts carry binary version, model sha, host, m (301/301 receipts recorded reviewer `unknown` on 09-10 `[C]`).
- Pareto selection rule pre-registered before measurement; "quorum lane" is one of the task axes.

### 5.6 Fleet (E)
- **Week-1 check:** at 09:32Z every host showed 0 busy runners with 27 PRs pending and intel at load 16.25 `[C]`. Either the busy sensor is wrong (the MainPID-vs-listener class already recorded in APR-RELEASE-001) or the fleet idles under backlog. This is E's most important number.
- forjar 1.31.0 on two hosts, 1.27.0 on three; mini holds all 10 open drift `[C]`.
- gx10 tightest disk (90 GB free `[C]`) — reaper + forjar, no SSH. `rust-cache` registry wipe: finish the remaining call sites. jq 1.6 vs 1.7+ split. `install.sh` asset 404.
- aprender's share of the 17 standing-red `main` workflows (`mdBook CI`, `Book Contract Enforcement`): **fix, or delete the trigger with a ticket.** A red nobody consumes teaches the team to ignore red.
- bashrs 7.4.2: clean-room (`cd machines/clean-room && make clean-room-p1`) first; human-run publish; then the forjar pin PR.

### 5.7 Formats (G)
G1 bit-exact container gate first (cheap, catches real corruption). G2 requant cosine gate over live types at the pin. Replace the 0-byte in-tree fixture model. Shape validation per hop once the fleet `pv` pin moves.

### 5.8 Performance (H)
Week 1: tok/s instrument lands, then the per-host ratio. Then a never-down ratchet. A 48× batched-prefill collapse on gx10 `[C]` says part of the gap is structural. Expose `prefix_cache_hits` before any published prefill figure.

### 5.9 Training — descoped this month
- 0.69: `apr finetune` on a Qwen 3.5 architecture refuses (`removed_by` set, counted); Qwen2.5 QLoRA receipts land. If the four receipts are absent at T-0 the cut script drops "fine-tuning" from the title and moves the rows.
- This month's deliverables: **DeltaNet CPU backward** as the reference (gradient check vs finite differences; threshold set after first measurement) and **ruled plans** for Qwen 3.5 QLoRA and offline distillation (teacher writes top-k logits at m = 1; student uses the ordinary fine-tune path). Implementation is next month.

### 5.10 cuda-oxide (D)
Alpha rustc backend, pinned nightly, git-only, clang-21, Linux — **never in a published dependency graph** (enforced by a `cargo metadata` check). Sanitizer nightly now; one-kernel probe (`gated_rmsnorm`) days 11–20 with a pre-registered exit; toolchain declared in `machines/yoga/forjar.yaml` and `machines/gx10/forjar.yaml`, never on clean-room runners. aarch64 and sm_89 support `[U]`.

### 5.11 Backends (F)
One receipt per backend family this month; the rest of the matrix is counted refusals. No non-CUDA job carries a CUDA label.

### 5.12 What the architecture model changes here
1. **aprender is both controlled by arbiter and depended on by it** (the apr lane). The cycle is broken by exact pin · refuse · never HEAD (rule 13, §5.0). Practical consequence: **every published aprender release is now a control-node dependency** — a bad release degrades the fleet's decision-maker, not only users. T-2 dogfood gains a row: the pinned `decide` fixtures replayed on the release candidate, report-only until 20 records.
2. **Goal I is an open-half claim.** External stakeholders get `apr` + `pv` + pmat + bashrs + forjar — no arbiter, no harness. Anything the internal workflow needs the closed half for is either moved into an open tool or left out of the external claim. Nothing a gate executes lives in a closed repo.
3. **Schema-constrained decoding (§5.0 row 8) now has three customers:** arbiter's quorum lane, tool calling in verb set V, and the human coding language planned on ruchy (`stack-30-day-plan.md` §4.10 — models emit a constrained intermediate, never Rust; ruchy compiles it deterministically). That raises its priority inside aprender above any single-consumer feature, and it should be designed for the general case: a caller-supplied JSON schema **or** grammar, on `apr run` and `apr serve`.
4. **`pv` is the band under every repo, and it ships from this one.** A `pv` release is a fleet event; the fleet pin bump (§5.4 row zero) follows every aprender train that changes `pv`, by rule, not by memory.

Out of scope for aprender this month: any implementation of the ruchy-based language (`label:lang-next` is on every train's standing exclusion list until its spec is ruled).

---

## 6. Train map — exclusions as predicates

Departures assume a 72 h takt `[A]`. Exclusions are `gh` search predicates over labels and paths; the labels must exist before 0.69 cuts.

| Train | Departs | Purpose | Excluded (predicate) | Must-carry |
|---|---|---|---|---|
| **0.69** | 09-23 | Tools stop lying | `label:arch-model-new` · `label:ont-depth` · `label:viz` · `label:perf-claim -label:has-receipt` · PRs adding files under the model-architecture directory | slice M |
| **0.70** | 09-26 | Receipts replace refusals | `label:arch-model-new` · `label:crux-competitor` · any PR adding a refusal without `removed_by` (lint) | candidate: batched parity — **plan ruled before 09-23 or it is not must-carry** |
| **0.71** | 09-29 | One receipt per backend family | `label:gate-framework-new` · `label:release-train-rewrite` · spec-only PRs with no enforcer | ≤ 1 |
| **0.72 – 0.78** | 10-02 … 10-20 | written one train ahead | **standing default:** `label:arch-model-new` · `label:gate-framework-new` · `label:lang-next` · refusal-without-`removed_by` · spec-without-enforcer — until a train overrides it | ≤ 1 each |

Milestones today `[V by the quorum session]`: 0.69 = 39 · 0.70 = 49 · 0.71 = 29 · 0.72 = 27 · 0.73 = 295.
- **0.73 is the overflow bucket, not a train.** Three reviewers read it as a plan — the label is the defect. **Rename to `backlog`, remove the due date.**
- **Strip 0.70 to measured capacity.** Capacity itself is `[U]`: C = 29/train `[C, 09-17]` vs 142 aprender merges in 7 days `[C]` ≈ 61 per 72 h. Re-derive C from the last five tags before anyone argues from it.

---

## 7. Decisions required from Noah

1. ONT-10: ship with depth gates declared-and-unarmed? *Recommend yes.*
2. Capability-ledger definition of 80 % for C? *Recommend yes.*
3. Verb set V for B — accept the §5.3 proposal or amend.
4. A at m = 1 on receipted cells this week? *Recommend yes.*
5. Training descoped to CPU backward + ruled plans this month? *Recommend yes — it frees 0.70's must-carry slot for batched parity.*
6. Ontology non-blocking for tags (permanent slot instead)? *Recommend yes.*
7. bashrs 7.4.2 — run the publish, or authorize in the bashrs session.
8. **Shared contract files** — regenerate in the merge step, or serialise ONT behind slice M? *Blocks slot 3.*
9. ~~apr quorum lane~~ — **RULED 2026-09-20:** apr is a voting engine in `arbiter decide` now, with the asymmetric vote; dogfood is P0 (§5.0). The PR-merge quorum lane stays shadow → calibrate → promote by measured recall.
10. Rename 0.73 → `backlog`.

---

## 8. Risks

| Risk | Signal | Response |
|---|---|---|
| A success condition satisfiable by shipping nothing | any "or" in §1 | review gate on this document: 0 disjunctive falsifiers |
| Must-carry row is a research project | > 2 non-passing quorums | auto-HRQ; rule 3 |
| Expired refusal traps a train | — | rule 2: bump + count, never block, never auto-revert |
| Merge-queue gridlock on shared contract files | ejections; slot 3 idle | decision 8 before slot 3 starts |
| Dogfood telemetry avalanche | > cap issues/day | circuit breaker (§5.5); unreceipted paths quarantined |
| Quorum lanes share a model family | unanimous misreads (happened on 0.73) | ≥ 2 families now; 3 with the apr lane |
| Reviewer depends on the artifact it reviews | apr lane on HEAD | rule 13: N-1 pinned, receipted cell |
| Power: gx10 breaker trip 09-17; lambda-labs 9 unclean losses `[C]` | unclean boot count | hardware/UPS decision (stack plan); no release step depends on a single box |
| Perf gap is structural | week-1 ratio ≪ 0.8 | ratchet; publish the gap; target the measured bottleneck |
| Interrupt releases eat the window | patch train opened | rule 7 |

---

## 9. Scoreboard

**Rule (adopted 2026-09-20 after four state lines went stale in two hours): this plan's prose cites no live numbers.** Targets live here; *state* is rendered by the named probe when the plan is read. The 2026-09-20 baseline is frozen in Appendix B — never edit it, never quote it as current.

| Metric | Target (2026-10-20) | Rendered by |
|---|---|---|
| Disjunctive falsifiers in this plan | **0** | lint over this plan + `scripts/check_milestone_cut.sh` |
| Goals with a positive control | **9/9** | `docs/audits/ONT-001/ledger.jsonl`; capability ledger |
| Verb set V passing with receipts, per CUDA host class | **100 %** | perf gate / parity receipts under `evidence/` |
| Differential corpus detects the planted kernel fault | **yes, every run** | perf gate / parity receipts under `evidence/` |
| Refusals outside V (counted, each with `removed_by`) | **counted; 0 without `removed_by`** | lint over this plan + `scripts/check_milestone_cut.sh` |
| Refusal `slips ≥ 2` | **0** | lint over this plan + `scripts/check_milestone_cut.sh` |
| Internal consumers meeting the request floor on Qwen 3.5, 7 days | **≥ 3** | dogfood ledger; `gh issue view` |
| `arbiter decide` availability on its host | **measured; unparsable rate per lane recorded, → 0 after constrained decoding** | `arbiter status` |
| Decision engines pinned by exact version + weights sha | **1/1** | `arbiter status` |
| Decisions where apr broke a strong-lane split | **0** (asymmetric vote) | `arbiter status` |
| Executables on the agent's PATH undeclared in forjar | **0** | `arbiter status` |
| Schema-constrained decoding in `apr` | **merged, contract + mutation proof** | dogfood ledger; `gh issue view` |
| apr PR-quorum lane runs in shadow | **every quorum** | `receipt-lint.sh` over `docs/audits/quorum-*.json` |
| Quorum lanes with measured planted-defect recall | **all** | `receipt-lint.sh` over `docs/audits/quorum-*.json` |
| Model families per quorum | **≥ 2 (3 on promotion)** | `receipt-lint.sh` over `docs/audits/quorum-*.json` |
| Receipts with reviewer model `unknown` | **0** | `receipt-lint.sh` over `docs/audits/quorum-*.json` |
| Tags cut by code outside the repo | **0 from v0.69.0** | `scripts/release/` self-identification line; `evidence/dogfood/` |
| Verdict functions able to emit GO/FAIL on no evidence | **0** | `scripts/release/` self-identification line; `evidence/dogfood/` |
| Host receipts per release | **4/4 every tag** | `scripts/release/` self-identification line; `evidence/dogfood/` |
| Own-train reconcile debt at cut | **0/0/0 own-train; inherited never up** | `scripts/check_reconcile.sh` |
| TRAITS rows disagreeing with pinned upstream | **0** | tests in `aprender-quant`; round-trip gate |
| ggml enums | **1** | tests in `aprender-quant`; round-trip gate |
| G1 bit-exact container round-trips | **100 % of live types** | tests in `aprender-quant`; round-trip gate |
| `--shapes` gates reporting UNMEASURED | **0** | `pv lint --gate shapes --format json`, per repo |
| ONT entity types | **9/9** | `docs/audits/ONT-001/ledger.jsonl`; capability ledger |
| Capability probes passing on the fleet-pinned `pv` | **≥ 80 % of frozen N** | `docs/audits/ONT-001/ledger.jsonl`; capability ledger |
| Backend families with ≥ 1 parity receipt | **4/4** | `receipt-lint.sh` over `docs/audits/quorum-*.json` |
| `ci / gate` p95 | **≤ 20 min × 3** | ledger command in APR-RELEASE-001 §5 P0 |
| Busy sensor validated | **yes** | `arbiter fleet` (occupancy via `fleet-bin.sh` only) |
| aprender standing-red `main` workflows | **0** | `arbiter red` |
| CUDA falsifiers under sanitizer nightly | **100 %** | perf gate / parity receipts under `evidence/` |
| tok/s ratio vs pinned llama.cpp | **baselined; never down** | perf gate / parity receipts under `evidence/` |
| Qwen 3.5 training | **CPU backward merged; 2 plans ruled** | dogfood ledger; `gh issue view` |
| Milestone moves on judgment | **0** | lint over this plan + `scripts/check_milestone_cut.sh` |


---

## 10. Week-1 measurements

1. TRAITS row count at `llama_pin.toml build_commit`.
2. Busy-sensor truth on intel (0 busy at load 16.25 with 27 pending).
3. C re-derived from the last five tags.
4. Which of the 17 standing-red `main` workflows are aprender's, and which are required checks.
5. tok/s ratio per host, after the instrument lands.
6. Anything depending on the wrong Q4_1 / Q8_1 sizes (`pmat query`, scoped with `--path`).
7. `compute-sanitizer --tool initcheck` reproduces the FP8 stale read.
8. Capability-ledger N, frozen at a sha; request floor N for goal A from observed traffic.
9. Is `aprender-contracts-cli` in the cascade set? Is ONT-4d already discharged by R-19?
10. Count of artifacts naming a model; `examples-nightly.yml` pass rate.
11. cuda-oxide on aarch64 and sm_89.
12. Does `apr distill` exist at HEAD? Definition and direction of R5.
13. Planted-defect corpus assembled; recall per existing lane.

---

## 11. Change log v1 → v2

**Accepted from the quorum:** refusal removed from every falsifier, floors + positive controls added (§1) · rule 2 rewritten — refusals never block a cut (§3) · tag path first, alone, precondition of 0.69 T-0 (§4, §5.2) · exclusions as predicates + a standing default for 0.72 onward (§6) · spine gains fleet-pin → C → G and a training branch (§2) · G split into bit-exact and requant (§1, §5.7) · H has one definition (§1, §5.8) · merge-queue gridlock and telemetry avalanche named, circuit breaker added (§5.4, §5.5, §8) · 0.70 stripped to capacity; Qwen 3.5 training descoped to ruled plans + CPU reference (§5.9).

**Rejected, with reasons:**
- *"Expired refusal … reverts its own PR."* A refusal is the safe state; reverting it re-exposes the wrong output. Bump and count instead.
- *"0.73 = 295 exceeds the whole month."* 0.73 is the overflow bucket from the 09-17 triage, not a scheduled train. The fix is the label (rename to `backlog`), not a purge to ≤ 29.
- *"A before B is circular."* A on receipted cells at m = 1 is deliberate; the missing piece was the constraint, now explicit.

**Added from the arbiter status report:** fleet `pv` 0.65.2 predates the shapes gate (C's row zero) · 0-busy-runners anomaly (§5.6, §10) · forjar skew and mini drift · standing-red `main` workflows · power as a risk class · capacity figure contested (142 merges/7 d).

**Added by operator ruling:** apr Qwen 3.5 as a quorum lane — shadow → calibrate → promote (§5.5, rule 13).

**v2.2 (2026-09-20, architecture model + operator's ruchy direction):** goal I restated as an open-half claim · §5.12 added (control-node dependency; T-2 `decide`-fixture row; constrained decoding has three customers and is designed for schema **or** grammar; `pv` release ⇒ fleet pin bump by rule) · `label:lang-next` added to the standing exclusion list.

**v2.1 (2026-09-20 09:52Z, operator ruling):** arbiter-on-apr dogfood is **P0 now** (§5.0) · asymmetric vote approved for D-15.12 · exact pin, not a floor · model choice provisional pending the selection rule · constrained decoding added to verb set V as a P0 aprender ticket · PR-merge quorum lane remains staged.

---

*Review guidance: look for any success condition that could be met by shipping nothing, and any exclusion a session could not apply without asking a human.*

---

## Appendix B — baseline snapshot, 2026-09-20 (frozen)

Provenance as marked in the body at the time (`[C]` from the arbiter status report and session reports of that day). It exists so targets have a starting point; it is **not** current state and is never updated — re-render from the probes in the scoreboard instead.

| Metric | Value on 2026-09-20 |
|---|---|
| Disjunctive falsifiers in this plan | 3 (v1) |
| Goals with a positive control | 0/9 (v1) |
| Verb set V passing with receipts, per CUDA host class | 0 |
| Differential corpus detects the planted kernel fault | no corpus |
| Refusals outside V (counted, each with `removed_by`) | uncounted |
| Refusal `slips ≥ 2` | — |
| Internal consumers meeting the request floor on Qwen 3.5, 7 days | 0 |
| `arbiter decide` availability on its host | **0 % (refuses: engine apr unavailable)** |
| Decision engines pinned by exact version + weights sha | 0/1 |
| Decisions where apr broke a strong-lane split | possible |
| Executables on the agent's PATH undeclared in forjar | ≥ 1 |
| Schema-constrained decoding in `apr` | absent |
| apr PR-quorum lane runs in shadow | 0 |
| Quorum lanes with measured planted-defect recall | 0 |
| Model families per quorum | 1 |
| Receipts with reviewer model `unknown` | 301/301 (09-10) |
| Tags cut by code outside the repo | every tag |
| Verdict functions able to emit GO/FAIL on no evidence | ≥ 4 |
| Host receipts per release | first at 0.68.2 |
| Own-train reconcile debt at cut | 25/37/12 inherited |
| TRAITS rows disagreeing with pinned upstream | ≥ 2 |
| ggml enums | 3 |
| G1 bit-exact container round-trips | no gate |
| `--shapes` gates reporting UNMEASURED | all |
| ONT entity types | 6/9 |
| Capability probes passing on the fleet-pinned `pv` | `[U]` |
| Backend families with ≥ 1 parity receipt | 1–2 `[U]` |
| `ci / gate` p95 | 62 min (stale) |
| Busy sensor validated | no |
| aprender standing-red `main` workflows | ≥ 2 |
| CUDA falsifiers under sanitizer nightly | 0 |
| tok/s ratio vs pinned llama.cpp | unmeasurable |
| Qwen 3.5 training | nothing filed |
| Milestone moves on judgment | 19 today |
