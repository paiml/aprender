# APR-EPIC-001 — themed epics, ≤100 open issues, cadence-bound trains (aprender)

Status: draft v1.0 (2026-09-25). Parent doctrine: FLOW-001 (`infra/docs/specifications/FLOW-001-fleet-flow-rules.md`) [U: verify at infra HEAD].
Drop at `docs/specifications/APR-EPIC-001-themed-epics-and-issue-cap.md`. Run: `Implement docs/specifications/APR-EPIC-001-themed-epics-and-issue-cap.md autonomously.`
Provenance marks: `[S]` snapshot (triage report 2026-09-25T14:16Z, `~/Desktop/aprender-triage-report.json`), `[M]` measured at HEAD by the cited command, `[O]` operator statement, `[A]` assumption, `[U]` unverified. Every `[S]` is re-derived in A0 before any write.

---

## §0 Operating assumptions

1. The issue tracker is a kanban board, not a backlog store. An open issue is WIP; everything else is parked, and parking is reversible.
2. Every open issue has exactly one parent epic (a GitHub sub-issue link), one priority label P0–P3, and a milestone equal to its epic's train.
3. Epics are **themed per train**. A train epic closes at its train's tag. Standing epics never close.
4. Trains ship on cadence: every 2–3 days `[O 2026-09-12]`. The date holds and scope slips `[O 2026-09-20]`.
5. Epics are planned 5 trains out, with ≤100 open issues and ≤10 open PRs per repo, and a new issue is triaged into an epic within 24 h `[O 2026-09-25, FLOW-001]`.
6. The traffic cop is the only minter of `pmat work add` and the only writer of `docs/roadmaps/epics.yaml`.
7. No Fable in any role `[O 2026-09-25]`. Routing: Opus 5.5 for cross-epic mapping (A4) and the guard (A6); Sonnet 5 for mechanical rows (A2, A3).

## §1 Ground truth

| Fact | Value | Mark | Re-derive with |
|---|---|---|---|
| Open issues | 824 | [S] | `gh issue list -R paiml/aprender --state open --limit 2000 --json number \| jq length` |
| Open epics | 41 | [S] | label `epic` ∪ `is_epic` in the snapshot generator |
| Open issues with no epic link | 509 | [S] | GraphQL `parent == null` and no `#<epic>` body ref |
| No milestone | 4 (#4337 #4350 #4377 #4422) | [S] | `gh issue list --search "no:milestone"` |
| Status | in-PR 67 / branch-only 223 / claimed 33 / unclaimed 253 / stale 248 | [S] | snapshot generator |
| Done-on-main candidates | 6 strong + 35 medium | [S] | `git log origin/main --grep "#N"` |
| Close candidates (stale / old release line) | 25 | [S] | idle > 7 d and no claim |
| 0.70.0 | due 2026-09-26, 81 open, 4.9 closes/day | [S] | milestone API |
| 0.71.0 / 0.72.0 due | 2026-09-29 / 2026-10-02 | [S] | milestone API |
| 0.73–0.76 due | none | [S] | — |
| Throughput per train (merged PRs, median) | 29 | [U, 2026-09-17 figure] | `gh pr list --state merged --search "milestone:X"` per shipped train |
| Project guide `sovereign-ai-stack-claude-project-guide.md` | not available to the author | — | — |

## §2 Epic set (13 epics)

Existing issues are reused where possible (retitle and re-scope) so that links survive. Absorbed epics close as `superseded by EPIC <id>` after their children are re-parented.

| ID | Title | Train | Reuse | Absorbs | Budget (open children) | Successor at cut |
|---|---|---|---|---|---|---|
| **E0** | Ontology (ONT-001) | 0.70 | #3269 | ONT rows 2c/3a/3b/4c/4c4/4c5/4d/4e/4f/5/8/9/10 (#4071 #4072 #4073 #3847 #4069 #3972 #4047 #4070 #4075 #4330 #4074 #4077 #4078 #4079); consumers #3715 #3856 #3559 #3560; resolver #4319 #4379 #4420; shapes honesty #3610 #3611 #3624 #4100 | 13 | S2 |
| **E1** | Fast Train | 0.70 | #3998 | #4033 #4288 #4232 #3081 #3058 | 10 | S1 |
| **E2** | Verbs Are Fast | 0.71 (D-1) | #3598 | #4249 children after 0.69.5 ships | 7 | E6 |
| **E3** | Don't Leave Behind | 0.71 | #3994 | #3062 #3597 | 9 | E7 |
| **E4** | Dogfood Model Lifecycle | 0.71 | #4381 | #4354 (PRM, one-level sub-epic) #4252 | 9 | E8 |
| **E5** | Agent Ready | 0.72 | #4000 | — | 5 | E6 |
| **E6** | llama.cpp Parity | 0.73 | #3999 | #2728 #2880 #3144 | 5 | E7 |
| **E7** | Any Model | 0.74 | #4001 | #3421 #3423 #3428 | 3 | S3 |
| **E8** | CRUX Fine-Tune/Distill | 0.75 | #4002 | — | 3 | E9 |
| **E9** | Classical ML + AutoML | 0.76 | #4164 | #3146 #3370 | 3 | S3 |
| **S1** | Gates That Cannot Lie | standing | #3863 | #2876 #2877 #2878 #2879 #2503 | 4 | — |
| **S2** | Provable Contracts (PVL-001) | standing | #2870 | #2556 (owner aprender-df kept); PVL rows #4080–4084 #4122 #4139 #4166 #4168 #4197–4202 #4238–4241 #4244 #4347 #4351 | 3 | — |
| **S3** | Debt Ratchet | standing | #3997 | #4057 #2814 #2582 #2373 | 3 | — |

Totals: 77 children + 13 epics = **90 steady-state**. Hard cap is 100 and the andon fires at 95. The 10-issue headroom is 24 h intake.

**Feature, done_when per epic.** Each goes into its epic body verbatim. Every threshold is re-measured at A0 or carried `[U]`.

- **E0**: rows bound/total ≥ 0.80 (`pv lint --gate shapes` at the tag sha); entity types 9/9; ghost bindings = 0; shapes over zero focus nodes = RED; ONT-10 releases `aprender-contracts-cli`; #3715's SHACL shape refuses the 0.70 tag on any missing cell; #3559 ruling recorded.
- **E1**: freeze→publish ≤ 4 h `[O]`; `vX.Y.Z-rc.N` on every green merge to `release/*`; rc→final assets sha256-equal; nightly publishes every `[[bin]]` plus a CUDA `apr`; every fleet host runs the newest verified build, or andon.
- **E2**: Qwen3.5 TTFT ≤ 2× llama.cpp d1d3c3396 on the same GGUF `[O 2026-09-20]`; `apr serve` resident with continuous batching; tok/s is generation-only; decode at 32k context ≥ `[U]`× llama.cpp (instrument first, then ratchet).
- **E3**: every (Qwen × quant × verb × backend) cell is Pass or `Refused{removed_by}`; the universe is read from tensor headers; 0 silent CPU fallbacks under forced accelerator.
- **E4**: PRM cells admitted once executors are declared (infra#1088); EXT-18 first public HF release; champion/challenger ratchet live.
- **E5**: `/v1/chat/completions` and `/v1/messages` pass one conformance suite; tool calls; `--json-schema`; `used_gpu` provenance on every response.
- **E6**: decode, prefill, TTFT and memory ≥ llama.cpp across the certified matrix (prefill today 0.19–0.20× [S #4376]).
- **E7**: one TRAITS table is the only quant dispatch; typed `TensorStorage`; a new architecture lands as config plus a small delta.
- **E8**: one YAML recipe for finetune/distill/merge, certified against 3–5 engines.
- **E9**: SVD, neighbour index and rank-typed tensor substrates; AutoML one-call fit/predict.
- **S1**: vacuous PASS = 0, env-death-as-code-failure = 0, hand-enumerated universes = 0 (each is a ratchet that only falls).
- **S2**: Lean/Kani run in required contexts; bindings resolve; L4/L5 have one definition.
- **S3**: clippy member baseline → 0 by 0.74; `.unwrap()` sites → 0; binary debt ledger falls; coverage floor ratchets up.

## §3 Hard rules

1. **Parent-epic invariant.** Every open issue has exactly one parent epic via a GitHub sub-issue link. A body `#N` reference does not count. One nesting level is allowed: a sub-epic (e.g. PRM #4354 under E4) counts toward its parent's budget as 1, and its own children count too.
2. **Priority invariant.** No open issue carries `P?`. P0 means *blocks its train's tag*. An issue created < 24 h ago is exempt.
3. **Milestone invariant.** An open issue's milestone equals its epic's train. Standing-epic children take the nearest open train. Patch milestones (`x.y.z`, z>0) hold only fixes to the shipped train. Everything else moves to its epic's train. 0.69.5 is exempt until it ships.
4. **Budget invariant.** Open children ≤ the epic budget in `docs/roadmaps/epics.yaml`, and open issues ≤ 100 repo-wide.
5. **Parking.** A parked issue is closed as `not planned` with label `parked`, keeps all its other labels, gets milestone `backlog`, is listed in its epic body's `## Parked` checklist, and gets one comment: `Parked by APR-EPIC-001: over <epic> budget (rank <r>/<n>). Reopen = pull.` Parking deletes nothing.
6. **Never park** an issue that:
   - has an open PR linked,
   - is claimed by a session live in `ListAgents`,
   - is a P0 whose train is ≤ current+2, or
   - has `assignee noahgift` with activity in the last 24 h.
7. **Pull (kanban).** An issue reopens only when its epic has budget free. One in, one out: a new issue at cap parks the lowest-EV open issue in the same epic.
8. **EV rank within an epic:**
   - key = (P0 < P1 < P2 < P3; then in-PR < branch ≤ 24 h idle < claimed < branch > 24 h idle < unclaimed < stale; then older first),
   - issues are kept in rank order up to the budget.
9. **Close-as-done** requires all three:
   - (a) a commit on `origin/main` naming `#N` with a closing keyword `[strong]` or a fix/feat line `[medium]`,
   - (b) `git merge-base --is-ancestor <sha> origin/main`,
   - (c) the issue's done_when/falsifier reproduces green at HEAD, or for `[medium]`, a reader agrees the fix line discharges the title.

   If any of the three fails, the issue stays open with a comment naming the missing leg. Rerun-to-pass is banned (jidoka).
10. **Train roll.** At T-24h, open children of the departing train that have no open PR re-parent to the epic's `successor` and take its milestone (#3458 mechanizes this). The date holds and scope slips.
11. **Epic close.** At the tag, a train epic closes only if it has 0 open children. Otherwise the tag cut refuses (poka-yoke via `check_milestone_cut.sh`, #3459).
12. **Write discipline.**
    - Every GitHub write phase first emits a plan receipt (`docs/audits/apr-epic-001/<row>-plan.json`: issue, action, reason, evidence).
    - Apply in batches of ≤ 50 content-creating writes per minute and ≤ 400 per hour (under GitHub's secondary limits `[A]`).
    - The apply step is idempotent: a re-run on the same live state produces 0 writes.
13. **Tree discipline.** No PR writes `docs/roadmaps/roadmap.yaml` or the README census (#4417). `docs/roadmaps/epics.yaml` is new and its single writer is the cop's row.
14. **Code search** uses `pmat query`, not grep. Ticket first (`pmat work add` by the cop), feature branch, PR, `ci / gate`. `main` is protected.

## §4 Rows (EV-ordered; one row = one ticket = one session)

The live-state selector runs the first row whose `done_when` is false.

| Row | Title | Kind | Writes | done_when |
|---|---|---|---|---|
| **A0** | Re-derive live state | read-only | receipt only | `docs/audits/apr-epic-001/a0-state.json` regenerated at HEAD from the §1 commands, with a diff against [S] recorded |
| **A1** | Epic skeleton | tree + GH | `epics.yaml`, 13 epic issues | `epics.yaml` lists 13 epics (id, issue, train, budget, successor, absorbs); each epic issue has the §2 title `EPIC <id> <train>: <name>`, a body with theme, done_when, budget and an empty `## Parked` section, and label `epic`; absorbed epics are listed but not yet closed |
| **A2** | Close verified-done | GH | closes | every [S] done-on-main candidate is either closed with an evidence comment (sha, reproduction) or open with a comment naming the missing leg; 0 closes without rule 9 |
| **A3** | Map every open issue to an epic | GH | sub-issue links | 0 open issues with `parent == null` (epics excluded); mapping receipt records title, chosen epic, and reason; for the [S] 143 "no epic matches" issues, a reader picks the epic, and any issue with no fitting theme goes to S3 |
| **A4** | Close absorbed epics | GH | closes | every absorbed epic in §2 has 0 open children and is closed `superseded by EPIC <id>`; any owner line (e.g. aprender-df on #2556) is copied to the absorbing epic body |
| **A5** | Enforce budgets and milestones | GH | parks, milestone edits | per epic, open children ≤ budget; repo open ≤ 100; every open issue's milestone = epic train (rule 3); 0 `P?` on open issues older than 24 h (the reader assigns priority from the rule-2 definition); the floor check in S-2 passes |
| **A6** | Guard: `check_issue_flow.sh` | tree | scripts, workflow, fixtures | bash + jq only (no python); falsifiers §6 planted RED→GREEN in the PR body; a scheduled workflow runs hourly on a `rust-neutral` runner, not clean-room or CUDA; first-green proof on live aprender is recorded before the workflow may be marked blocking |
| **A7** | Cadence binding | tree + GH | milestone dates, `roll_train.sh` or #3458 | 0.73–0.76 dated (D-5); `epics.yaml` budgets for trains due within 6 days = `floor(C_measured)` with C re-measured by the §1 command; T-24h roll dry-run receipt produced for 0.70.0 |

**Order rationale.**
- A2 runs before A3 because closes shrink the mapping set.
- A5 runs after A3 because budgets are only meaningful per epic.
- A6 runs after A5 so the guard lands green, and its first-green gets proved on real data.
- 0.70.0 is due 2026-09-26, so **A5 applies rule 10 to 0.70 immediately**: only in-PR issues and E0/E1 children in budget stay on 0.70.

## §5 Phases per row (paiml-implement)

1. **Ticket.** The cop mints `pmat work add` → branch `apr-epic-001/<row>`.
2. **Read HEAD (genchi genbutsu).** Run the §1 commands, `ListAgents`, and open PRs; write the row's plan receipt.
3. **Plan quorum.** One agy, one Claude and one apr lane review the plan receipt; the apr lane is advisory per the asymmetric-vote ruling `[O 2026-09-20]`. Any `do-not-implement` verdict is a STOP.
4. **Apply** in throttled batches (rule 12). Resumable: the receipt records the last applied index.
5. **Verify.** Re-fetch live state; the row's `done_when` evaluates true from fresh reads, never from the apply log.
6. **PR** for any tree change (A1, A6, A7), passing `ci / gate`. GitHub-only rows (A2–A5) carry their receipts in a docs PR.

## §6 Falsifiers (A6 guard; each planted RED, then GREEN)

The guard reads a snapshot JSON (produced by `scripts/fetch_issue_snapshot.sh`, a GraphQL fetch through `gh api graphql`) plus `docs/roadmaps/epics.yaml`, and exits 0 only on GREEN.

| ID | Planted input | Expected |
|---|---|---|
| FALSIFY-FLOW-001 | 101 open issues | RED `cap` |
| FALSIFY-FLOW-002 | an open non-epic issue with `parent: null` | RED `orphan` |
| FALSIFY-FLOW-003 | an open issue with `P?`, created 25 h ago / 23 h ago | RED / GREEN (24 h boundary) |
| FALSIFY-FLOW-004 | an empty issue list, or gh exit ≠ 0 | RED `Unknown{NoData}` (anti-vacuity: never PASS on nothing) |
| FALSIFY-FLOW-005 | a closed issue labelled `parked` that is absent from every epic's `## Parked` list | RED `unrecoverable-park` |
| FALSIFY-FLOW-006 | an open child whose milestone ≠ its epic's train | RED `milestone` |
| FALSIFY-FLOW-007 | an epic with open children = budget+1 | RED `budget:<id>` |
| FALSIFY-FLOW-008 | an epic in `epics.yaml` with no GitHub issue, or vice versa | RED `epic-drift` |

Mutation proof: each rule's check is deleted in turn, and the matching fixture must flip to GREEN, proving the check is load-bearing. Recorded in the A6 PR body.

## §7 Budget and routing

- Sessions: K̂ = 5 (A0+A1, A2, A3, A4+A5, A6+A7); K = 7; andon at session 6 if the row is before A5. `[A]`
- GitHub content writes: estimated ≤ 1,300 (≈ 720 parks/closes with comments, ≈ 500 sub-issue links, milestone edits) `[A from S]`. This is ≥ 3.25 h at 400/h, so A3 and A5 are expected to span resumes.
- Context: compact between rows; keep bulk JSON out of session history (write it to files) `[O 2026-09-25]`.
- Routing:
  - A3 and A6: Opus 5.5 at medium effort.
  - A2 and A5: Sonnet 5.
  - A4 and A7: Sonnet 5.
  - Fable: never.

## §8 STOP conditions

- **S-1.** A plan-quorum `do-not-implement` verdict on any row.
- **S-2 (floor exceeds cap).** |never-park set (rule 6)| + 13 epics > 100. The fix is draining PRs, not parking in-flight work. Stop with the set enumerated per epic.
- **S-3.** Any action whose API call would delete an issue, comment, branch or label. Only close, reopen, label, milestone and link are permitted.
- **S-4.** An unresolved operator decision blocks a row. D-1, D-4 and D-5 have defaults (below) and do not block. D-2 blocks A5 only if applying the default would park a P0.
- **S-5.** A secondary rate limit is hit 3 times in one session. Stop, record the resume index, and the next session resumes.
- **S-6.** Live state diverges from the plan receipt by > 5% of planned actions between plan and apply (another session is writing). Re-plan.

### Operator decisions

| ID | Question | Default applied |
|---|---|---|
| D-1 | Placement of E2 Verbs Are Fast | 0.71; 0.70 = E0 + E1 only (heijunka) |
| D-2 | P0 meaning | "blocks its train's tag". P0s in trains ≥ current+3 (e.g. 15 P0 in 0.76 [S]) are parkable |
| D-3 | #3559: 80% ontology target vs the §11 one-row-per-train ratchet | none; E0 records the ruling; does not block this spec |
| D-4 | Steady-state target | 90 open, hard cap 100 |
| D-5 | Dates for 0.73–0.76 | 3-day spacing: 10-05, 10-08, 10-11, 10-14 `[A]` |

## §9 Final report schema

```yaml
spec: APR-EPIC-001
session: {date_utc: , host: , tree: {head: , origin_main: }, model: }
row: A0|A1|A2|A3|A4|A5|A6|A7
ticket: PMAT-
outcome: done|stopped|partial
stop: {id: , reason: , evidence: , resume_when: }
live_state:
  open_total: 
  open_by_epic: {E0: , E1: , E2: , E3: , E4: , E5: , E6: , E7: , E8: , E9: , S1: , S2: , S3: }
  orphans: 
  p_unknown_over_24h: 
  milestone_mismatch: 
  never_park_floor: 
actions: {closed_done: , closed_superseded: , parked: , reopened: , linked: , milestone_moved: , priority_set: }
receipts: [docs/audits/apr-epic-001/...]
guard: {falsifiers_red_then_green: 0/8, first_green_on_live: true|false, workflow_blocking: false}
decisions_applied: {D-1: , D-2: , D-4: , D-5: }
budget: {k_hat: 5, k_actual: , gh_writes: , andon_crossed: false}
next_row: 
```
