# APR-LOOKAHEAD-001: Rolling look-ahead workers (one dedicated worker per each of the next three trains)

**Spec id:** `APR-LOOKAHEAD-001` · **Version:** 1.1 (2026-09-27; v1.1 adds the §2a train themes ruled by the operator) · **Rows:** `LA-00..LA-08` (the traffic cop mints one `pmat` ticket per row; single-minter rule applies)
**Target repo:** `paiml/aprender` (`docs/specifications/APR-LOOKAHEAD-001-rolling-epic-workers.md`)
**Runner:** the aprender traffic cop (`aprender-traffic-cop-prompt.md`). The cop owns the slots; the workers own their epics.
**Launch:** from `~/src/aprender`, run `Implement docs/specifications/APR-LOOKAHEAD-001-rolling-epic-workers.md autonomously.`
**Related:** APR-RELEASE-001 (release train), APR-EPIC-001 (themed epics, issue cap, only the cop mints), FLOW-001 (flow rules), FLOW-003 v2 (queue, tiers, fold = move), APR-OBS-001 (measurement), KREG-001 / #4539 (kernel registry), #3598 (0.71 "Verbs Are Fast").
**Status:** operator directive; implementation in progress under epic #4540 (LA-00 #4541 … LA-08 #4549).

**Operator directive (Noah, 2026-09-27):** *"Going forward I ALWAYS want ONE dedicated worker on the next three epics: while we build 0.70, one worker for 0.71, another for 0.72 and another for 0.73."*

**Provenance marks:** `[V]` verified · `[C]` computed · `[A]` asserted · `[U]` unmeasured.

---

## §0 Operating assumptions

1. **Purpose.** Every train starts ready to go. When train $N$ ships, train $N+1$ already has a groomed epic, specs, contracts, falsifiers, baselines and de-risked prototypes. Train $N+2$ and train $N+3$ have a theme, a ranking and specs for their top rows. Planning never lands on the critical path.
2. **The slot invariant.** While train $N$ is the current train, exactly one dedicated look-ahead worker is assigned to each of $N+1$, $N+2$ and $N+3$. There are never zero and never two.
3. **The current train always wins.** Any contention for sessions, runners, merge-queue slots, GPUs, review lanes or budget goes to train $N$ first, then $N+1$, then $N+2$, then $N+3$.
4. **Look-ahead is preparation, not a side channel.** Workers follow the same doctrine as every session:
   - ticket-first, with only the cop minting tickets;
   - branch → PR → `ci / gate`;
   - a contract and a planted falsifier per feature;
   - five-whys to a mechanism;
   - `pmat query` over grep;
   - fold = move, never copy (FLOW-003).
5. **Continuity of ownership.** A worker stays with its epic when the epic moves closer. The 0.71 worker keeps 0.71 when 0.71 becomes the current train; only the slot label changes. Context isn't thrown away at each rotation.
6. **Model routing.** Look-ahead workers run on the Opus class, per `model-routing-strategy.md`. Fable is banned.
7. **The operator has no pre-steps.** Anything only Noah can do is a §8 STOP.

---

## §1 Ground truth (baseline 2026-09-27; never quote as current)

| # | Fact | Value | Mark | Source of truth |
|---|---|---|---|---|
| G1 | Current train | 0.70 (rc.1 expected 17:30–19:00 Madrid; crates.io gated on ONT-10) | [V] cop report | milestone `0.70.0`, #4429 |
| G2 | 0.71 theme | "Verbs Are Fast" | [V] | #3598, `06x-release-schedule.md` |
| G3 | 0.72 / 0.73 themes | **ruled 2026-09-27:** 0.72 = "Train What You Serve"; 0.73 = "Runs Everywhere" (§2a) | [V] operator ruling | `06x-release-schedule.md` (LA-01 records it); milestones `0.72.0`, `0.73.0` (0.73 was renamed `backlog` on 2026-09-20; re-create it as a train) |
| G4 | Existing dedicated worker | `kreg` on #4539 (kernel registry, belongs to 0.71) | [V] | cop-state |
| G5 | aprender epics | 35 / 126 done | [V] cop report 13:12 | epic tracker |
| G6 | Open PRs | aprender 18 (cap 10); infra 22 (cap 5) | [V] | `gh pr list` |
| G7 | Budget at 13:12 | 18% of the 5-hour window, 26% of the week | [V] | cop report |

---

## §2 Roles and slots

| Slot | Epic | Mandate | May merge to `main`? | Open-PR budget |
|---|---|---|---|---|
| **L1** | train $N+1$ | Make the next train ready to start: groom, spec, contract, baseline, de-risk, land enabling work | **Yes**: small PRs under the normal flow, behind a flag if they're user-visible, and never while S-1 holds | ≤ 2 |
| **L2** | train $N+2$ | Shape: theme, ranking, specs and contracts for the top rows, research, spikes | Specs, contracts and docs only; code as draft PRs | ≤ 1 |
| **L3** | train $N+3$ | Scout: theme proposal, hypotheses, risks, measurement plan, external reference study | Specs and docs only | ≤ 1 |

- Look-ahead PRs count toward the repo's open-PR cap. When the repo is at its cap, L3 yields first, then L2. L1 yields only to the current train.
- At most one look-ahead PR per train is in the merge queue at a time, and it always queues behind train-$N$ PRs.

## §2a Train themes and exit criteria (operator ruling 2026-09-27)

A train ships only when every exit criterion holds, each with a receipt. A criterion that can't be met by the cut moves to the next train through an operator ruling. It is never waived.

| Train | Theme | Slot now | Exit criteria (all must hold) |
|---|---|---|---|
| 0.70 | Ontology & Fast Train | current | per `06x-release-schedule.md`; crates.io gated on ONT-10 |
| **0.71** | **Verbs Are Fast** (#3598) | L1 | **V1** TTFT ≤ 2× llama.cpp (pin `d1d3c3396`) on Qwen3.5-4B, APR-OBS identity · **V2** `apr serve` resident, with load + TTFT reported beside pp512/tg128 (#3596) · **V3** batched decode with a serving-shape parity receipt · **V4** `--json-schema` constrained decoding · **V5** kernel registry KREG-001 (#4539) live: unregistered kernels refused, 0 unregistered dispatches · **V6** per-layer serve tracing (APR-OBS OBS-09) landed before the first performance PR · **V7** EXT-001 CRUX competitor gates defined |
| **0.72** | **Train What You Serve** | L2 | **T1** `apr finetune` (QLoRA), `apr distill`, `apr merge`, `apr quantize` complete end to end on Qwen 3.5 with **0 refusals** (the qwen35 refusals are removed) · **T2** fine-tune throughput ≥ 0.8× Unsloth on the same GPU and model, with a committed measurement command · **T3** Prometheus B2: one challenger (local Qwen3.5-27B teacher → 4B student, gold labels only, 0 sealed-test hashes in training data) judged by PRM-001 §5.4 · **T4** first improved dogfood model published to Hugging Face as an rc with receipts (EXT-001) · **T5** trained weights round-trip gguf ↔ safetensors ↔ .apr at cosine ≥ 0.98 |
| **0.73** | **Runs Everywhere** | L3 | **E1** Qwen3.5-4B `apr run` and `apr serve` pass parity (cosine ≥ 0.98 vs llama.cpp) on WGPU (intel, dual AMD), Metal (mini), aarch64 CPU and CUDA (gx10) · **E2** each backend ≥ 0.5× llama.cpp speed on the same host `[A]`, recalibrated by L3's measurement plan · **E3** the MoE model Qwen3-Coder-30B (#3341) loads and runs with parity · **E4** WGPU/Metal refusals removed · **E5** kernel registry covers 100% of kernel keys these backends dispatch · **E6** the Prometheus fallback options (intel-wgpu, mini-metal) become admissible cells |

**Rationale for the order `[C]`:**
- **0.72 first:** it unblocks programmes already running (Prometheus B2, EXT-001) that wait on the training verbs, and it needs only CUDA, which the fleet already has.
- **0.73 second:** it builds on a registry hardened through all of 0.71 and 0.72, which is what keeps new backends from reintroducing the wrong-layout bug class. It also puts idle hardware to work (intel's AMD GPUs, mini).
- **Alternative recorded:** swap the two if external portability demos become the priority.

**Theme ownership:** a theme changes only by operator ruling, recorded in `06x-release-schedule.md` with its date. When the next slot (0.74) opens, its L3 worker proposes the theme.

---

## §3 Invariants, rotation and proofs

### §3.1 State
The cop keeps `cop-state/lookahead.json` (single writer: the cop):

```json
{ "current_train": "0.70",
  "slots": { "L1": {"train": "0.71", "worker": "kreg-71", "since": "…", "heartbeat": "…", "handoff": "docs/lookahead/0.71.md"},
             "L2": {"train": "0.72", "worker": "la-72",   "since": "…", "heartbeat": "…", "handoff": "docs/lookahead/0.72.md"},
             "L3": {"train": "0.73", "worker": "la-73",   "since": "…", "heartbeat": "…", "handoff": "docs/lookahead/0.73.md"} } }
```

### §3.2 Invariants
- **I1 (coverage):** $\{\text{slot trains}\} = \{N+1, N+2, N+3\}$.
- **I2 (uniqueness):** each slot has exactly one worker, and no worker holds two slots.
- **I3 (liveness):** every slot's heartbeat is less than 2 h old.
- **I4 (handoff):** every slot has a current handoff file on `main`, updated within 24 h.

### §3.3 Events and the cop's actions

| Event | Cop action (one atomic cop-state commit) |
|---|---|
| **Tick** (hourly batch) | Check I1–I4. Respawn any slot whose heartbeat is older than 2 h, using the same handoff file and the standing prompt (§7). |
| **Worker death** (session ended or quit) | Respawn at the next tick; the new worker reads the handoff file. |
| **Train $N$ publishes to crates.io** | Rotate: $N+1$ becomes the current train, and its worker stays on it as epic lead in the train pool (§0.5). L2 is relabelled L1 and L3 is relabelled L2. **Spawn a new L3 for $N+4$** before announcing the release. |
| **Train cut slips** | No rotation; slots unchanged. |
| **Operator re-scopes a train** | Update the slot's epic link and handoff file; the worker stays. |

### Proposition 1 (the invariant is restored within one tick)
Assume the cop runs at least one tick every 70 min and a spawn takes at most 10 min `[A]`. Then after any single event, I1 and I2 are restored within 80 min, and I1 is never violated by a rotation.

*Proof.*
- **Rotation.** It is a single atomic commit that relabels L2→L1 and L3→L2 and creates L3 for $N+4$. Before it, the slot trains are $\{N+1, N+2, N+3\}$; after it they are $\{N+2, N+3, N+4\}$, which is exactly $\{N'+1, N'+2, N'+3\}$ for $N' = N+1$. So I1 holds at every committed state. I2 holds because the relabelling is a bijection and the new L3 gets a fresh worker.
- **Death.** Only I3 is affected. It is detected at the first tick with a heartbeat older than 2 h, and the slot is refilled at most 10 min after that tick.

Worst-case vacancy of a slot is therefore under 2 h + 70 min. The 80 min in the statement is measured from detection. ∎

*Why the rotation spawns before it announces.* If the announcement came first, a reader of cop-state could see $\{N+2, N+3\}$ with $N+4$ missing. That violates I1, and a respawn tick could race the rotation and create two L3 workers, which violates I2. The single commit prevents both.

---

## §4 Worker charter (what each worker does, in priority order)

Every worker runs the same idempotent loop. It re-reads live state (milestone, epic, open PRs, handoff file), picks the highest-EV unfinished item for its slot, does one item, updates the handoff file, and writes a heartbeat.

### L1 (train $N+1$)
1. **Epic hygiene** (APR-EPIC-001):
   - one themed epic;
   - every ticket a sub-issue, prioritised and sized ($\hat K$);
   - dependencies linked;
   - ≤ the milestone cap.
   New ticket proposals go to the cop, which mints them. **The worker never mints.**
2. **Definition of Ready**, applied to the top 10 rows by EV:
   - a spec section;
   - a pv contract;
   - a planted falsifier;
   - a measured baseline (APR-OBS identity);
   - owners proposed.
3. **De-risking.** Land enabling work early as small PRs: registries, refactors behind flags, measurement harnesses. Today this means KREG-001 (#4539) for 0.71.
4. **Handoff packet** at the $N$ cut: a ready list, risks, the first 5 PRs to open, and the baselines.

### L2 (train $N+2$)
1. Propose the theme and get an operator ruling, recorded in `06x-release-schedule.md`.
2. Rank the top 20 rows.
3. Write specs and contracts for the top 5.
4. Spikes as draft PRs, each with one question and one measured answer.
5. Update the risk register.

### L3 (train $N+3$)
1. A theme proposal with hypotheses and falsifiers.
2. An external reference study (llama.cpp, vLLM, Unsloth, and so on) as a list of facts, with no code copied.
3. A measurement plan.
4. A risk register.
5. The top 5 ranked rows.

### All slots, never
- Touch the release PR, human contributors' PRs, or `main` directly.
- Copy commits between open PRs (FLOW-003: fold = move).
- Mint tickets.
- Run GPU or clean-room work while `train-active` is set.
- Bypass a gate.

---

## §5 Budget and priority

| Condition | Action |
|---|---|
| 5-hour-window usage below 70% | all three slots active |
| 70–85% | L3 pauses (heartbeat continues, no work) |
| above 85% | L2 and L3 pause; L1 continues |
| account at 100% (rotation per standing rule) | L1 pauses only if the current train needs the budget; resume after rotation |
| S-1 (release cut in progress) | L1 may not arm PRs; all slots continue non-merge work |

**Combined look-ahead spend target:** ≤ 15% of 5-hour-window usage `[A]`. Recalibrate after 2 trains from measured train-start latency (§6).

---

## §6 Readiness gates and metrics

**Definition of Ready for train $N+1$ at the cut of $N$** (checked by the cop, rendered from GitHub and the handoff file):
- the epic exists, is themed, and 100% of its tickets are sub-issues, prioritised and sized;
- the top 10 rows by EV each have a spec section, a contract, a planted falsifier and a baseline;
- the first 5 PRs are identified with owners;
- 0 open questions that need an operator ruling, or each such question is already a ruling request.

**Look-ahead readiness for $N+2$:** theme ruled, top 20 ranked, top 5 specified.
**For $N+3$:** theme proposal, risk register and measurement plan exist.

| Metric | Target | Rendered by |
|---|---|---|
| Slot coverage (I1–I2) | **100%** of hourly ticks | LA-02 |
| Slot vacancy | **< 2 h** worst case; **0** vacancies longer than 3 h | LA-02 |
| **Train-start latency**: tag of $N$ → first merged PR of $N+1$ | **≤ 2 h** | LA-05 |
| $N+1$ Definition of Ready at the cut | **100%** of top-10 rows | LA-05 |
| Rows entering a train without a spec and contract | **0** | LA-05 |
| Look-ahead share of budget | **≤ 15%** | LA-06 |
| Look-ahead PRs over their slot budget | **0** | LA-04 |
| Operator rulings requested at the cut (not before) | **0** | LA-05 |

---

## §7 Standing prompt (one file, parameterised by slot)

`docs/prompts/lookahead-worker.md`, launched as
`Run docs/prompts/lookahead-worker.md with SLOT=L1 TRAIN=0.71 autonomously.`

```
You are the dedicated look-ahead worker for aprender train ${TRAIN} (slot ${SLOT}),
per docs/specifications/APR-LOOKAHEAD-001-rolling-epic-workers.md.
Loop (idempotent; live state wins over memory):
 1. Read: milestone ${TRAIN}, its epic, open PRs you own, docs/lookahead/${TRAIN}.md,
    cop-state/lookahead.json. Write heartbeat.
 2. If S-1 (release cut) or a §5 pause applies to ${SLOT}: do non-merge work only, or idle.
 3. Pick the highest-EV unfinished item from §4 for ${SLOT}. One item per iteration.
 4. Do it under doctrine: ticket (ask the cop to mint) → branch → PR → ci / gate;
    contract + planted falsifier; fold = move, never copy; pmat query over grep.
 5. Respect your open-PR budget (§2). Never touch the release PR or human PRs.
 6. Update docs/lookahead/${TRAIN}.md: done, next, risks, rulings needed, baselines.
 7. Report to the cop in ≤ 5 lines. Stop conditions: §8.
```

---

## §8 STOP conditions (stop, write the §9 report, do not work around)

- **S-1** A release cut is in progress and the action would arm, merge or use a train-active resource.
- **S-2** A slot invariant can't be restored within 2 ticks (for example, no session capacity). Report to the operator.
- **S-3** A worker would need to mint a ticket, touch the release PR or human PRs, or copy commits between PRs.
- **S-4** A theme or scope decision needs the operator. Write the ruling request into the handoff file and continue with other items.
- **S-5** A harness hook refuses writes for a reason other than a missing ticket.
- **S-6** Two consecutive non-passing quorums on the same PR.
- **S-7** Look-ahead spend is above 15% for 2 consecutive windows. Pause L3, then report.

---

## §9 Tickets (EV-ordered)

| EV | Row | Work | Contract | Done when (all must hold) | K̂ |
|---|---|---|---|---|---|
| 0 | **LA-00** spec + state | Commit this spec, the `cop-state/lookahead.json` schema and `docs/lookahead/{0.71,0.72,0.73}.md` skeletons | `lookahead-state-v1` | schema lint; planted two-workers-one-slot state fails; planted missing-$N+3$ state fails | 45 |
| 0 | **LA-01** fill slots + record themes | L1 = the existing `kreg` worker, re-chartered as the 0.71 look-ahead (KREG-001 is its first item). Spawn L2 (0.72, "Train What You Serve") and L3 (0.73, "Runs Everywhere"). Record the 2026-09-27 theme ruling and the §2a exit criteria in `06x-release-schedule.md`. Create one themed epic per train (APR-EPIC-001), with its exit criteria as the epic checklist. Re-create milestone `0.73.0` as a train, separate from `backlog`. Milestone #4539 to 0.71 | — | cop-state shows 3 slots with live heartbeats; the ruling is on `main` with its date; 3 epics exist and each lists its exit criteria (V1–V7, T1–T5, E1–E6) | 45 |
| 1 | **LA-02** tick + respawn | Cop tick checks I1–I4 and respawns from the handoff file | `lookahead-liveness-v1` | a planted dead worker (heartbeat 3 h old) is respawned within 1 tick; planted duplicate is refused | 60 |
| 1 | **LA-03** standing prompt | `docs/prompts/lookahead-worker.md` (§7) | — | launched for all 3 slots; each writes a heartbeat and a handoff update within 1 h | 30 |
| 2 | **LA-04** PR budgets | Cop enforces §2 open-PR budgets and queue ordering behind train $N$ | `lookahead-pr-budget-v1` | a planted third L1 PR is blocked; a planted look-ahead PR ahead of a train PR is re-queued | 60 |
| 2 | **LA-05** readiness gate | Render the Definition of Ready for $N+1$, plus the $N+2$ and $N+3$ checks, into the hourly report; measure train-start latency at each tag | `lookahead-readiness-v1` | a planted top-10 row without a contract shows NOT READY; latency receipt recorded at the 0.70 tag | 90 |
| 3 | **LA-06** budget gating | Apply the §5 pause ladder from the window-usage reading | `lookahead-budget-v1` | a planted 86% reading pauses L2 and L3 and leaves L1 running | 45 |
| 3 | **LA-07** rotation | Atomic rotation at crates.io publish of $N$; spawn $N+4$ before the announcement (Prop. 1) | `lookahead-rotation-v1` | a dry-run rotation 0.70 → 0.71 leaves slots {0.72, 0.73, 0.74} in one commit; a planted announce-before-spawn ordering fails the test | 60 |
| 4 | **LA-08** fleet extension (optional) | Offer the same slot model to Tier-2 repos (pmat, forjar, bashrs, infra, paiml-implement, rmedia) as an operator choice | — | a ruling request filed; no change without it | 20 |

**K̂ = 455 `[A]` · K = 520 · andon at 415.**

---

## §10 Report schema (the cop's hourly batch, look-ahead section)

```yaml
lookahead:
  current_train: "0.70"
  slots:
    - {slot: L1, train: "0.71", worker, heartbeat_age_min, item_in_progress, open_prs, paused: bool}
    - {slot: L2, train: "0.72", ...}
    - {slot: L3, train: "0.73", ...}
  invariants: {I1: ok|fail, I2: ok|fail, I3: ok|fail, I4: ok|fail}
  readiness:
    exit_criteria:
      "0.71": {V1..V7: open|met|moved}
      "0.72": {T1..T5: open|met|moved}
      "0.73": {E1..E6: open|met|moved}
    n_plus_1: {epic_groomed_pct, top10_ready: "x/10", first5_prs_identified: bool, rulings_pending}
    n_plus_2: {theme_ruled: bool, top20_ranked: bool, top5_specified: "x/5"}
    n_plus_3: {theme_proposed: bool, risk_register: bool, measurement_plan: bool}
  budget: {lookahead_share_pct, window_pct, pauses}
  train_start_latency_last_tag_min: null
```
