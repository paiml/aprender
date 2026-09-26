# APR-EPIC-001 — themed epics, ≤100 open issues, cadence-bound trains (aprender)

Status: draft v1.4 (2026-09-25; v1.1 = D-6 loosen + phase-in, cap unchanged; v1.2 = §4a A3 seed map, rules 15–16; v1.3 = §4b intake control, rules 17–19, row A8; v1.4 = §4c collapse, rule 20, row A4b). Parent doctrine: FLOW-001 (`infra/docs/specifications/FLOW-001-fleet-flow-rules.md`) [U: verify at infra HEAD].
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
6. **Never park** an issue that (v1.1, ruling on the cop's floor>cap blocker):
   - is named by a **closing keyword** in an open PR (a passing mention or fold reference does not protect it),
   - is a P0 whose train is the **current or next** train (today 0.70 and 0.71), or
   - has `assignee noahgift` with activity in the last 24 h.

   Claim-only, branch-only and live-session claims are **not** protected. The park comment records the claim and the branch head sha, so no work is lost; the claimant reopens the issue when its PR opens (rule 7 still applies).
6a. **Phase-in ratchet (the cap is never raised).** If the rule-6 floor + 13 epics > 100 at A5, park down to the floor. From then on, open issues must fall monotonically: `open(d) ≤ open(d-1)` every day, reaching ≤ 100 by the 0.71 tag (2026-09-29). Until then the guard reports `cap` RED but **non-blocking** (andon). From the 0.71 tag on, `cap` blocks (jidoka). A day where `open(d) > open(d-1)` is RED and blocking at any time.
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
| **A5** | Enforce budgets and milestones | GH | parks, milestone edits | per epic, open children ≤ budget; repo open ≤ max(100, rule-6 floor + 13) with the rule-6a ratchet armed; every open issue's milestone = epic train (rule 3); 0 `P?` on open issues older than 24 h (the reader assigns priority from the rule-2 definition); the floor check in S-2 passes |
| **A6** | Guard: `check_issue_flow.sh` | tree | scripts, workflow, fixtures | bash + jq only (no python); falsifiers §6 (001–010) planted RED→GREEN in the PR body; a scheduled workflow runs hourly on a `rust-neutral` runner, not clean-room or CUDA; first-green proof on live aprender is recorded before the workflow may be marked blocking |
| **A7** | Cadence binding | tree + GH | milestone dates, `roll_train.sh` or #3458 | 0.73–0.76 dated (D-5); `epics.yaml` budgets for trains due within 6 days = `floor(C_measured)` with C re-measured by the §1 command; T-24h roll dry-run receipt produced for 0.70.0 |

**Order rationale.**
- A2 runs before A3 because closes shrink the mapping set.
- A5 runs after A3 because budgets are only meaningful per epic.
- A6 runs after A5 so the guard lands green, and its first-green gets proved on real data.
- 0.70.0 is due 2026-09-26, so **A5 applies rule 10 to 0.70 immediately**: only in-PR issues and E0/E1 children in budget stay on 0.70.

### §4a A3 seed map (v1.2, from `unclaimed-clusters.md` 2026-09-25T15:57Z, 240 unclaimed, k=12, silhouette 0.078: themes, not partitions)

The seed is a starting point only. A3's reader confirms every link; the cluster label is never the reason recorded.

| Cluster | Size | Seed epic | Correction to the cluster's own suggestion |
|---|---|---|---|
| c1 EXT rows #4382–#4413 | 32 | E4 (#4381) | Over budget by construction. Open = the next 9 by EXT phase order; the rest are parked in #4381 `## Parked` |
| c10 REX rows #4355–#4367 | 13 | E4, via sub-epic #4354 | REX-00/02/03/04 are done (folded into #4317): #4317 must carry `Closes #4355 #4357 #4358 #4359`. REX-01 is done → close. REX-05..12 are blocked on infra#1088 → park (rule 7 reopens them when it lands). #4422 is not REX → E1 |
| c5 decode perf | 6 | E2 | #3822 (silent corruption) is correctness → E3 |
| c6 thinking/long-ctx | 13 | E6 | Correctness rows #3882 #3919 #3930 #3961 #3977 #3851 → E3; #4376 stays E6 |
| c9 clippy/unwrap | 7 | S3 | #4155/#4156/#4157 are 0.72–0.74 slices → park until their train is current |
| c7 PP-066 rows | 7 | S3 | #2892–#2979 are 20 d idle → close as not planned (rule 9 does not apply; they are not done). #4319 → E0 |
| c8 binaries | 8 | S3 (#4057) | #4288 → E1 (absorbed); #4324/#4337 fleet-bins alarms → E1 nightly |
| c2 contracts/proofs | 23 | S2 / S1 | #3687 (untriaged issues) → E1 (this spec's own guard discharges it) |
| c3 CI/runners | 17 | S1 / E1 | Runner-capacity rows (#3518 #3612 #3986 #4306) → **transfer to paiml/infra** (rule 15). #4433/#4424 → E1 |
| c11 script/guard defects | 17 | S1 | #4100 → E0 |
| c0, c4 grab-bags | 97 | hand triage | Epic-titled issues (#3062 #3081 #3421 #3423 #3428 #3863 #4164) are **not** children; they are §2 reuse/absorb rows |

**Arithmetic.** The 240 unclaimed issues alone are 2.4× the cap. Seeded E4 = 45 against a budget of 9, and S1+S2+S3 = 70+ against a budget of 10. Parking dominates A5. Priority order within each epic follows rule 8.

15. **Transfer, don't park, foreign work.** An issue whose fix lives in another repo (runner capacity, forjar declarations, fleet images) is moved with `gh issue transfer <n> paiml/<repo>`, leaving a cross-link comment. It leaves aprender's count and is not parked. The receiving repo's FLOW-001 cap applies there.
16. **Epics are never counted as unclaimed work** in any triage or cluster input. An epic's "claim" is its owner line. The A0 snapshot excludes `epic`-labelled issues from the unclaimed and clustering sets (anti-vacuity: 7 of 97 c0/c4 members were epics).

### §4b Intake control (v1.3): the root cause is inflow, not stock

**Five whys (from the d1 age report, 2026-09-25 18:06):**
1. 827 issues are open, and 537 (65%) were filed in the last 7 days.
2. About 40 sessions run under "surface defects, don't ask" plus the no-defer rule, so every red check becomes an issue.
3. Filing an issue costs nothing, closing one is not enforced (33 are done on main but still open), and no gate meters intake.
4. The only WIP limit is on PRs (≤10). Issues have none, so the queue absorbs the whole overflow.
5. **Terminal cause:** issue creation is a push system with no kanban signal. Any session can mint WIP.

**Little's law.** Open = arrival rate × time in system. Arrivals are ≈ 77/day [S: 537/7]. A cap of 100 would need every issue resolved in ≤ 1.3 days, so the stock cap alone is infeasible. Arrivals must be metered.

17. **Findings are not issues (poka-yoke).** A session that surfaces a defect appends one line to `docs/findings/<session>.jsonl` (append-only fragment: `{id, title, evidence, repro, suspected_epic, severity, found_at_sha}`), or to the cop inbox. It does not create an issue. **Only the cop mints issues**, and only when the target epic has budget free (rule 7). "Surface defects, don't ask" and no-defer still hold: the finding is recorded, not deferred. It is queued, not made WIP.
    - The project CLAUDE.md line "surface defects" is rewritten to "surface defects to `docs/findings/`; never `gh issue create`".
    - Enforcement lives in the guard, not in trust: any open issue whose author is not the cop account and not on the human allowlist (Noah, Alfredo, external reporters) is RED `unmetered-intake`. It is auto-converted to a findings line and closed as `moved to findings`.
    - External reporters (non-org) are never blocked. Their issues bypass the metering and count against intake.
18. **Every merged PR discharges its issue.** A PR without `Closes #N` or `Fixes #N` for ≥ 1 open issue, or an explicit `no-issue: <reason>` trailer, fails `ci / gate` (`check_pr_closes.sh`). This removes the "done on main, still open" class (33 today).
19. **Flow ratchet.** Daily, per repo: `created(d) ≤ closed(d)` over a 3-day trailing window, measured with `gh issue list --search "created:>=D"` and `closed:>=D`. Any breach raises an andon on arbiter/Slack; a breach two days running is RED and blocking. Findings-ledger depth is reported, not capped. It is the backlog the cop pulls from in EV order.

**Falsifiers (added to the A6 guard):**

| ID | Planted input | Expected |
|---|---|---|
| FALSIFY-FLOW-011 | an open issue authored by a non-cop agent session | RED `unmetered-intake` |
| FALSIFY-FLOW-012 | a PR body with only `Refs #N` and no `no-issue:` trailer | RED `no-close` |
| FALSIFY-FLOW-013 | 3-day window with created = 30, closed = 20 / created = 20, closed = 30 | RED `inflow` / GREEN |

### §4c Collapse (v1.4): one issue per PR, not one issue per row

20. **Issue = unit of PR.** A spec's rows, a sibling family, or ratchet slices that one owner delivers in batches live as `- [ ] <row>: <title> (<evidence>)` lines in **one** parent issue. A line is ticked when its PR merges, and the PR says `Closes #<parent>` only on the last line (otherwise `Refs #<parent> row <id>`). Sub-issues are **not** a collapse, because they still count against the cap.
    - **Collapse** when all of these hold: same owner or same spec, same epic, and delivered in ≤ 1 PR per phase.
    - **Never collapse:** P0s in the current or next train; issues with an open closing-keyword PR; issues from external reporters.
    - **Mechanics:** the cop creates or reuses the parent, copies each child's title, evidence and last comment link into its checklist line, then closes the child as `not planned` with label `collapsed` and a comment `collapsed into #<parent> row <id>`. Reopening a child is allowed (rule 7); its line is then struck.
    - **Seed families [S]:**

      | Family | Now | Parents |
      |---|---|---|
      | EXT rows #4382–#4413 | 32 | 4, one per EXT phase |
      | REX rows #4355–#4367 | 13 | 1 (#4354) |
      | clippy ratchet slices #4154–#4157 | 4 | 1 |
      | CRUX spikes #4220–#4223 | 4 | 1 |
      | CUDA GEMV gaps #3953 #3960 #3963 | 3 | 1 |
      | PVL orphan groups, PP-066 rows, binary-debt #4061–#4065 | ~15 | 3 |

      ≈ 71 issues → 11. Target: ≥ 50 net fewer open issues, measured by A0 before and after.

21. **Epics are never split to fit a platform limit (v1.5).** GitHub caps an epic at 100 sub-issues. An epic over that limit is over its budget by 10–40×, which means the problem is the stock, not the tree. A3 links only issues that survive: an epic gets at most `budget` linked children in rule-8 order. Everything else goes straight to collapse (rule 20) or to the epic's `## Parked` checklist, and is **never linked first**. Linking and then closing is two writes where one would do. The epic count stays at 13, and FLOW-001 governs any change to the epic set.

**Row A4b (runs after A4, before A5):** apply rule 20. **done_when:** every seed family is collapsed or has a written reason why not; the net reduction is measured; and FALSIFY-FLOW-014 (a `collapsed` child with no matching checklist line in its parent is RED `lost-collapse`) is planted RED then GREEN.

**New row A8 (EV ahead of A6, because intake must be metered before the cap can hold):**
- Findings schema and contract `findings-ledger-v1`.
- The CLAUDE.md rewrite.
- `check_pr_closes.sh`.
- The cop's mint-on-budget path.
- **done_when:** FLOW-011..013 planted RED then GREEN, and 24 h of live intake with 0 unmetered agent-filed issues.

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
| FALSIFY-FLOW-009 | 120 open before 2026-09-29 with open(d) ≤ open(d-1) / 121 open after 120 the day before / 101 open after the 0.71 tag | andon, exit 0 / RED `ratchet` / RED `cap` |
| FALSIFY-FLOW-010 | a park plan containing an issue named by `Fixes #N` in an open PR, and one containing an issue only mentioned in a PR body | refused `protected` / allowed |

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
- **S-2 (floor exceeds cap after the phase-in).** At the 0.71 tag, |never-park set (rule 6)| + 13 epics is still > 100. Before then, a floor over the cap is handled by rule 6a, not a STOP. The fix is draining PRs (merge-queue batching), not parking protected work. Stop with the set enumerated per epic.
- **S-3.** Any action whose API call would delete an issue, comment, branch or label. Only close, reopen, label, milestone and link are permitted.
- **S-4.** An unresolved operator decision blocks a row. D-1, D-4 and D-5 have defaults (below) and do not block. D-2 blocks A5 only if applying the default would park a P0.
- **S-5.** A secondary rate limit is hit 3 times in one session. Stop, record the resume index, and the next session resumes.
- **S-6.** Live state diverges from the plan receipt by > 5% of planned actions between plan and apply (another session is writing). Re-plan.

### Operator decisions

| ID | Question | Default applied |
|---|---|---|
| D-1 | Placement of E2 Verbs Are Fast | 0.71; 0.70 = E0 + E1 only (heijunka) |
| D-2 | P0 meaning | "blocks its train's tag". P0s in trains ≥ current+2 (e.g. 18 P0 in 0.72, 15 in 0.76 [S]) are parkable |
| D-6 | Floor > cap (cop blocker 2026-09-25: floor 151 under v1.0 rule 6, 106 without claim-only) | **Decided:** loosen (rule 6) + phase in (rule 6a); cap stays 100 |
| D-3 | #3559: 80% ontology target vs the §11 one-row-per-train ratchet | none; E0 records the ruling; does not block this spec |
| D-4 | Steady-state target | 90 open, hard cap 100 |
| D-5 | Dates for 0.73–0.76 | 3-day spacing: 10-05, 10-08, 10-11, 10-14 `[A]` |

## §9 Final report schema

```yaml
spec: APR-EPIC-001
session: {date_utc: , host: , tree: {head: , origin_main: }, model: }
row: A0|A1|A2|A3|A4|A4b|A5|A8|A6|A7
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
  created_3d: 
  closed_3d: 
  findings_depth: 
  unmetered_intake: 
  cap_mode: andon|blocking
  open_prev_day: 
actions: {closed_done: , closed_superseded: , closed_not_planned: , collapsed: , parked: , transferred: , reopened: , linked: , milestone_moved: , priority_set: }
receipts: [docs/audits/apr-epic-001/...]
guard: {falsifiers_red_then_green: 0/14, first_green_on_live: true|false, workflow_blocking: false}
decisions_applied: {D-1: , D-2: , D-4: , D-5: , D-6: loosen+phase-in}
budget: {k_hat: 5, k_actual: , gh_writes: , andon_crossed: false}
next_row: 
```

## §4d (cop addendum 2026-09-25): phases = PRM-001 v3 §7

Source: handoff/PRM-001-prometheus-v3.md §7, lines 305–315, sha c9b52d08d7eb. That table replaces the §4d P0–P6 phases. Appended by aprender-d1 on the cop's ruling; no other section is edited.

| Phase | Train | Content |
|---|---|---|
| **P0** | now | capture + shadow |
| **P1** | 0.71 (E2) | speed instrument + ratchet |
| **P2** | 0.71 | experiment + hardware ruling |
| **P3** | 0.72 | tripwire |
| **P4** | 0.73 (E6) | tie-breaker; prefill parity |
| **P5** | D-7 | local distill (Q4/Q5) |
| **P6** | 0.75 (E8) | vote |
