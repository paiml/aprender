# Open-epic categorization, 0.73.0 → 0.80.0 (measured 2026-10-30)

Work ticket: #6567 (PMAT-6567, under epic #6532).

**Source.** GraphQL reads of paiml/aprender on 2026-10-30. There are 224 open issues:

| Milestone | Open |
|---|---|
| 0.73.0 | 38 |
| 0.74.0 | 41 |
| 0.75.0 | 41 |
| FT-PARITY (0.76.0; milestone #18) | 37 |
| 0.77.0 | 35 |
| 0.80.0 | 32 |

Every epic-like issue and its sub-issues were read. The raw files are `plan2/open-issues.tsv`, `plan2/epic-scope.tsv` and `plan2/kids.tsv`.

**Dates.** Operator, 08:25Z: 0.73.0 11-06 · 0.74.0 11-13 · 0.75.0 11-20 · 0.76.0 11-27 · later = 0.80.0.
- 0.77.0 (12-04) already exists (milestone #32).
- This plan adds two milestones on the same weekly cadence, each with a themed epic in its own document: 0.78.0 (12-11) and 0.79.0 (12-18).
- 0.80.0 stays "later" and gets no date here. The planner sends a due date for it to a planted quorum (hand-off §6 step 7), so that "every open release milestone has a due date" holds.

**Rules applied, not relitigated:**
- The planning worker (aprender-6f) is the single writer. This document proposes and files nothing.
- Never close a ticket with an open PR or one in progress.
- Closing a *PR* stays the operator's.
- Icebox means close as not planned, with a link to #6521.
- Caps: ≤100 open tickets, ≤5 workers per repo, 1 ticket in progress per worker.

## 1. Rule for every epic

- An epic carries the milestone of the release it is meant to finish in.
- A *standing* epic never finishes, so it carries 0.80.0 ("later").
- Every child carries its own release, never its parent's.
- A child already scheduled in an earlier release stays there. What happens to it when that release closes is set by `scripts/release/carry_milestone_items.sh`, measured on main 2026-10-30:
  - The script picks the next open milestone whose title is exactly X.Y.Z (`semver()`, lines 41-57).
  - An open item moves to that milestone **only** if its number (#n) appears in the body of an issue labeled `epic` whose title starts "EPIC: release train <next>" followed by end or whitespace (lines 58-64).
  - Every other open item moves to the `backlog` milestone. An open issue labeled `must-carry` refuses the whole carry.
- **Two defects this plan must fix before it relies on carrying (hand-off §6):**
  - Milestone #18 is titled "FT-PARITY: Unsloth parity via declarative fine-tuning (epic #6532)", which is not X.Y.Z. At the 0.75.0 cut the script would choose **0.77.0** as "next", skipping 0.76.
  - No open epic is titled "EPIC: release train 0.7x.0 …" today (#6532 is "EPIC 0.76.0: …"). So at every cut, all open items would go to `backlog`, not forward.
- Therefore each release epic lists, in its body, the earlier-release prerequisites it wants carried. Anything it does not list goes to `backlog` at the cut, by design.
- **The script carries only one release ahead.** A prerequisite in 0.73.0 that the 0.79 epic wants would reach `backlog` at the 0.73 cut, since no 0.74 train epic lists it. So at each cut, after the script runs, the planner re-milestones from `backlog` straight to its target every item a later train epic lists, and names those items in the cut report. No extra 0.73–0.75 train epics are filed for this (they would add to the open count).
- The same holds for absorbed items and child epics (#4001, #5317, #5321, …): one still open at its own release's cut goes to `backlog` unless the next train epic lists it. That is the date-driven model working; the cut report names it.
- Open children of a moved or absorbed epic keep their own milestone, and their own cut decides them by the rule above. Named so none is lost: under #4381, #4709 and #3204 (0.73.0) and #5347 (0.74.0); under #5317, #3555, #3558, #3574, #3483 and #4444 (0.74.0) and #4696 (0.75.0); under #5923, #5996 and #5897 (0.74.0) and #6101 and #5761 (0.75.0). #4444 is listed by ANY-MODEL, so it returns there from `backlog`.
- The carry script reads every `#N` in the next train epic's body as a listing (`carry_milestone_items.sh:63-64`). Epic bodies therefore write a reference that is not a listing as `issue N`, without the `#`.

## 2. Categorization

| # | Title (short) | Open children | Now | Proposed | Why |
|---|---|---|---|---|---|
| 5962 | Merge-flow plumbing | 3 | 0.73.0 | 0.73.0 | finishes in its release |
| 4741 | PY-PORT-2 (C301 Rust port) | 2 | 0.73.0 | 0.73.0 | finishes in its release |
| 3715 | pv SHACL release-readiness shape | 1 | 0.73.0 | 0.73.0 | finishes in its release |
| 3269 | E0 Ontology | 4 | 0.74.0 | 0.74.0 | children in 0.73/0.74 |
| 6452 | PVB-001 proofs reach code | 7 | 0.74.0 | 0.74.0 | children in 0.74 |
| 4001 | E7 Any Model | 2 (#3433, #3513, both 0.74.0) | 0.74.0 | **0.79.0** | the epic's exit (one TRAITS table, typed TensorStorage, a new arch as config + small delta) is the 0.79 theme. Its two children stay in 0.74 |
| 4000 | E5 Train What You Serve | 3 (+ #4698, moved to FT-PARITY) | 0.75.0 | 0.75.0 | #4698 has moved to FT-PARITY. The other three open children stay in their own releases: #5348 (0.75.0), #3720 (0.73.0), #5349 (0.73.0) |
| 5348 | Agent Ready (E-1/E-2) | 2 (#5478, #5312) | 0.75.0 | 0.75.0 | E-1/E-2 finish in 0.75. The 0.78 epic builds on it and does not absorb it |
| 5312 | CUDA concurrent batching | 2 (#4971, #4486) | 0.75.0 | 0.75.0 | prerequisite of 0.77 cell S4 |
| 6519 | 20 merged/h, theme A | 2 | 0.75.0 | 0.75.0 | finishes in its release |
| 6522 | speech stack (ASR + diarization) | 5 (0/5 done) | 0.75.0 | 0.75.0 | the operator's ship/slip/close triage of 0.75 decides it; this plan does not move it |
| 3676 | CI diet | 1 | 0.75.0 | 0.75.0 | — |
| 6532 | FT-PARITY | 34 | 0.76.0 | 0.76.0 | filed 10-30 under a 3/3 quorum. Milestone #18 holds 37 open: these 34, #6532 itself, #6392 (parent #5306) and #4698 (parent #4000) |
| 5305 | Qwen3.5 speed parity | 12 | 0.80.0 | **0.77.0**, child of SERVE-PARITY | the 0.77 theme |
| 5317 | Qwen3.5 dogfood lane | 7 | 0.77.0 | 0.77.0, child of SERVE-PARITY | its exit (blessed model + serving-shape receipts) is a 0.77 cell |
| 5321 | 8 GB Orin footprint | 2 | 0.77.0 | 0.77.0, child of SERVE-PARITY | cell S6 |
| 5329 | serve request path | 3 | 0.80.0 | **0.77.0**, child of SERVE-PARITY | cell S1 |
| 5325 | GPU refuse-by-name | 4 | 0.80.0 | **0.77.0**, child of SERVE-PARITY | cell S1 |
| 5323 | CUDA quantized kernels | 7 | 0.80.0 | **0.79.0**, child of ANY-MODEL | its exit, "every (qtype, layout, arch) kernel registered or refused", is cell M3. Its children #3111 and #4378 stay in 0.77.0 and keep #5323 as parent; SERVE-PARITY tracks them as S1 work |
| 3999 | E6 Runs Everywhere | 1 (#3739) | 0.80.0 | 0.80.0 | its only open child, #3739 (0.73.0), is the per-release CRUX dogfood, which is standing; #3739 follows the 0.73 cut like any 0.73 item, and ANY-MODEL lists it |
| 6481 | Colibri CRUX (MoE expert streaming) | 0 | 0.80.0 | **0.79.0**, child of ANY-MODEL | cell M2. Owner aprender-w5032 is kept |
| 4381 | E4 Dogfood Model Lifecycle | 6 | 0.77.0 | **0.80.0** | its exit (PRM cells, first public HF release, champion/challenger ratchet) is in no 0.77–0.79 theme. Child #5317 is reparented to SERVE-PARITY. It is planned under the umbrella test below once #5317 is reparented and #5345 iceboxed; the pre-act check confirms its other open children are in 0.73–0.75, and if one is not, #4381 is iceboxed with a link to #6521 |
| 5510 | 20 merged/h umbrella | 2 (#6519, #6520) | 0.77.0 | **0.80.0** (standing) | an umbrella; its children carry their own releases |
| 6520 | 20 merged/h, theme B | 15 | 0.77.0 | **0.80.0** (standing) | flow work. Each release's flow lane takes ≤10 of its children (§3) |
| 5923 | EMG-001 emergency flow | 16 | 0.77.0 | **0.80.0** (standing) | as #6520. Its 10 parked-PR tickets: §3 |
| 5965 | merge-path checks: GATE or LAB | 7 | 0.80.0 | 0.80.0 (standing) | children carry their own releases |
| 5306 | release path tooling | 12 | 0.80.0 | 0.80.0 (standing) | #6392 stays in 0.76.0 as the release-path blocker |
| 3998 | E1 Fast Train | 14 | 0.80.0 | 0.80.0 | umbrella; 13 open children in 0.73.0 (4) and 0.75.0 (9), plus #5306 (0.80.0) |
| 3598 | E2 Verbs Are Fast | 4 | 0.80.0 | 0.80.0 | umbrella; 4 open children: #5305 and #5321 are reparented to SERVE-PARITY, #3568 (0.74.0) and #4450 stay under #3598. #5305 keeps its own children (#4162, #4487, #4522), which are not serving-parity work and are not moved by this row |
| 2870 | S2 Provable Contracts (standing) | 1 (#6452) | 0.74.0 | **0.80.0** | standing epic (rule §1) |
| 3863 | S1 Gates That Cannot Lie (standing) | 3 | 0.80.0 | 0.80.0 | standing |
| 3997 | S3 Debt Ratchet (standing) | 3 | 0.80.0 | 0.80.0 | standing |
| 2582 | apr-dogfood v3.0 | 1 (#1293, milestone "backlog") | 0.80.0 | 0.80.0 | — |
| 6490 | E-kern CRUX | 0 | 0.80.0 | 0.80.0 | no children yet |
| 6496 | browser-automation harness vs unsloth studio | 0 | 0.80.0 | 0.80.0 | its body says LAB only, UI test tooling; not the 0.78 agent theme. Owner aprender-w5018 is kept |
| 6521 | Icebox | 0 | 0.80.0 | 0.80.0 | stays open by design |
| 4780 | FLAKE-0 | 0 (5/5 closed) | 0.80.0 | **close (completed)** | every sub-issue closed |
| 5875 | Coverage gaps, checklist 1 | 0 (35/35) | 0.80.0 | **close (completed)** | every sub-issue closed |
| 5959 | CI hosts / runner-hours | 0 (3/3) | 0.80.0 | **close (completed)** | every sub-issue closed |
| 5960 | runner capacity 38→46, 4→5 | 0 (4/4) | 0.80.0 | **close (completed)** | every sub-issue closed. The runner targets themselves are one-way doors that stay with #6520 under quorum |
| 5961 | shard split | 0 (1/1) | 0.80.0 | **close (completed)** | every sub-issue closed |
| 5963 | receipt reuse | 0 (1/1) | 0.80.0 | **close (completed)** | every sub-issue closed |
| 5967 | PR supply | 0 (1/1) | 0.80.0 | **close (completed)** | every sub-issue closed |
| 6396 | pv provenance | 0 (7/7) | 0.80.0 | **close (completed)** | its body: done when every sub-issue is closed |
| 5968 | Coverage gaps, checklist 2 | no sub-issues | 0.80.0 | **icebox**, linked to #5875 and #6521 | same shape as #5875; its body lists only closed items |

**Why the 0.80.0 epics that stay are "planned".** The 11:17Z ruling iceboxes what is outside 0.72.x–0.76.0 *unless planned*. Here planned means one of: a standing epic (§1: #2870, #3863, #3997, #5965, #5306, #5510, #5923, #6520, #6521); an umbrella whose open children are all in 0.73–0.75 or are standing epics (#3998, #3598, #3999); or an epic with a named owner (#2582 owner aprender-a7, #6496 owner aprender-w5018). #6490 has no owner or children, but it was filed 10-28 and edited 10-30, so the pre-act check (activity in 48 h) would skip it today. Each still gets triaged at the 0.76 close under ruling B.

**Closed parents that still have open children: reparent them.**

| Closed parent | Open child | New parent |
|---|---|---|
| 3994 E3 Don't Leave Behind | #3714 | ANY-MODEL |
| 3994 | #5323 | ANY-MODEL |
| 3994 | #5325, #5329 | SERVE-PARITY |
| 3994 | #5332 | none: iceboxed (§3b), linked to #5306 and #6521 |
| 4220 CRUX parity spike | #4954 (0.73.0) | #3999 |
| 5307 Gate honesty | #4652 (0.77.0) | #3863 |

**Pre-act check.** Before every close and icebox in this document, the planner reads the issue's open linked PRs, its assignee and its timeline. It skips any issue that has an open PR, an assignee, an `in-progress` label, or *activity* in the last 48 h, and lists the skips instead of acting on them. Activity means a comment, a commit reference or a linked-PR event by anyone other than the planner; label and milestone edits do not count.

## 3. Tickets already in 0.77.0 that sit outside its theme (35 open today)

- **Theme (stay, tracked by SERVE-PARITY), 9:** #2815, #2817, #3111, #3725, #3761, #4374, #4378, #5276, #6389.
  - #3111 and #4378 keep parent #5323 (ANY-MODEL); #6389 keeps parent #5965. SERVE-PARITY lists them in its body; it does not reparent them.
- **Epics, 6:** #4381, #5317, #5321, #5510, #5923, #6520.
  - Four change milestone to 0.80.0 per §2: #4381, #5510, #5923, #6520.
  - #5317 and #5321 stay in 0.77.0 and are reparented to SERVE-PARITY.
- **Flow lane 0.77 (cap 10), 9 from this milestone:** #4526, #4652, #5917, #5951, #6269, #6360, #6391, #6408, #6468. The tenth slot is #5990 (§3b).
- **#5345** (PRM review-replay, under E4) → **icebox**, linked to #4381 and #6521 (it is outside 0.72.x–0.76.0 and in no 0.77–0.79 theme). 1.
- **The 10 parked-PR tickets → icebox, 10:**
  - The tickets: #6363, #6365, #6367, #6369, #6370, #6371, #6374, #6377, #6378, #6379. Each is linked to #5923 and #6521.
  - Every PR they track was read on 10-30 and is CLOSED: #5854, #5878, #6010, #6081, #6011, #5975, #5976, #6016, #6087, #6027.
  - The work stays recoverable from each PR's branch.
  - Re-landing one takes a new ticket, and only when the hourly flow constraint names it.

Sum: 9 + 6 + 9 + 1 + 10 = 35.

### 3b. The seven non-epic tickets in 0.80.0

The 11:17Z icebox ruling applies to these too. Each was read on 10-30:

| # | Parent | Proposed |
|---|---|---|
| #4539 KREG-001 | #5323 | → 0.79.0, cell M3 |
| #5221 decode GEMV bandwidth | #5305 | → 0.77.0, cell S2 |
| #5222 GDN recurrence | #5305 | → 0.77.0, cell S2 |
| #5990 affected_set.sh fail-open | #5965 | → 0.77.0 flow lane (slot 10 of 10) |
| #6196 LAB: guards-nightly red 8 nights | #5965 | stays 0.80.0: it is the one C324 LAB ticket with one owner for that check |
| #6501 release rehearsal measured_cpu | #5306 | stays 0.80.0: it has an open PR (#6507) |
| #5332 release receipts | closed #3994 | **icebox**, linked to #5306 and #6521 |

## 4. Open-count arithmetic (cap ≤100)

| Step | Count |
|---|---|
| Open today | 224 |
| Completed epics closed | −8 |
| #5968 iceboxed | −1 |
| Parked-PR tickets iceboxed | −10 |
| #5345 and #5332 iceboxed | −2 |
| New epic issues: SERVE-PARITY, AGENT-GATEWAY, ANY-MODEL | +3 |
| **Net after this plan** | **206** |

This plan does **not** meet the ≤100 cap, and it says so rather than hiding the gap.
- 157 of the 224 open tickets sit in 0.73.0–0.76.0 (38 + 41 + 41 + 37), inside the window the 11:17Z ruling protects.
- So the remaining ~106 can only come from two places: the planner's ship/slip/close triage of 0.73.0–0.76.0 under the 11:17Z and 15:01Z rulings, and FT-PARITY's own ticket cap. Nothing left outside that window can close the gap.

## 5. Just-in-time filing (keeps the cap from growing)

- **Filed now:** each release's epic issue, with its new ticket rows as a checklist in the body (not as issues).
- **Filed later:** each new row becomes an issue on the day the previous release's milestone closes. So 0.77's rows are filed at 0.76 close (11-27), 0.78's at 0.77 close (12-04), and 0.79's at 0.78 close (12-11).
- **Epic issues are exempt:** the three train epics are filed now regardless of the count (+3 in §4); they hold the rows as a checklist and are what the carry script reads.
- **Earlier filing** of rows is allowed only while the measured open count is below 100. Under §4 that is not expected before 0.76 closes, so the epics plan as if every new row starts at the previous release's close.
- **The cap holds at filing too.** Filing a release's rows must keep the open count ≤100. If it would not, rows are filed in priority order (the never-cut base rows first) up to the cap, and the rest stay as checklist lines in the epic body until room opens. The cut report names them.
- An existing ticket that a theme absorbs keeps its number. It is only re-milestoned and reparented.

## 6. Hand-off

When this document and the three epics have finished review (3/3 PASS, or fix-forward after round 3 under the 15:01Z review cap, with the receipts naming which), the planner (aprender-6f), as the single writer, acts in this order:
1. renames milestone #18 to `0.76.0` and moves its theme text into the milestone description
2. creates milestones 0.78.0 (due 12-11) and 0.79.0 (due 12-18), with each theme in the description. It does **not** retitle 0.77.0; its theme also goes in the description
3. files the three epic issues, label `epic`, titled exactly:
   - "EPIC: release train 0.77.0 SERVE-PARITY …"
   - "EPIC: release train 0.78.0 AGENT-GATEWAY …"
   - "EPIC: release train 0.79.0 ANY-MODEL …"

   Each body lists, by #n, the absorbed issues and the earlier-release prerequisites it wants carried (§1).
4. applies §2, §3 and §3b (they need 0.78.0/0.79.0 and the three epic issues to exist, since reparenting points at them), with the pre-act check
5. retitles #6532 to "EPIC: release train 0.76.0 FT-PARITY …", keeping the rest of the title, so the 0.75.0 cut can carry into 0.76
6. at each later cut, re-milestones from `backlog` the items a later train epic lists (§1)
7. sends a due date for 0.80.0 to a planted quorum and sets it from the receipt. Items carried 0.79 → 0.80 are named in the 0.79 cut report
8. replies with the issue numbers and the measured open count

## Review debt (Haiku, round 3: FAIL; shipped fix-forward under the 15:01Z review cap, not a blocker)

- [ ] (fixed forward 10-30, needs re-review) line 171: Hand-off step 3 applies the reparenting to SERVE-PARITY, AGENT-GATEWAY and ANY-MODEL, but step 4 files those epics. The order is impossible as written.
- [ ] (fixed forward 10-30, needs re-review) line 41: The carry chain cannot implement drop-to-next-release at the 0.73 (11-06) and 0.74 (11-13) cuts. carry_milestone_items.sh:58-64 lists only the next train epic, and no 0.74.0 or 0.75.0 train epic is filed, so unlisted open 0.73/0.74 items go to backlog.
- [ ] (fixed forward 10-30, needs re-review) line 71: Open children of moved or absorbed epics have no disposition. Under 4381 (moved to 0.80): 4709 and 3204 (0.73.0, P0) and 5347 (0.74.0) appear in neither section 2 nor 3. Under 5317: 3555, 3558, 3574, 3483, 4444 (0.74.0) and 4696 (0.75.0) are not dispositioned. Line 16 says every epic's children were read.
- [ ] (fixed forward 10-30, needs re-review) line 48: ANY-MODEL's carried-prerequisite list omits issue 4444 (0.74.0). AM-13 and M1 (lines 90-94, 139) depend on it as the llama.cpp logit oracle, and nothing returns it to 0.79 after the 0.74 cut. The same gap applies to issue 3719 (child of absorbed issue 5325) in 077 line 44 and 078 line 37.
- [ ] (fixed forward 10-30, needs re-review) line 3: All three epics cite plan2/cat-final-307b8bc968a6.md, which does not exist at plan2/ (it is at plan2/old/). The round-2 re-point did not land. The same stale header appears at 078 line 3 and 079 line 3.
- [ ] (fixed forward 10-30, needs re-review) line 181: Internal contradiction on M3. Line 181 says M3 is never cut, but line 205 says that if more than 3 prerequisites carry in, M3 slips first. The cut order (lines 175-179) has no M3 entry.
- [ ] (fixed forward 10-30, needs re-review) line 192: Load does not fit the stated cut order. Load is 13 theme + 10 SP + 6 FT-carry + 10 flow = 39 against 25 ticket-days. Cutting the flow lane (10), SP-12 (1) and S6 (2) still leaves 26 > 25. The line also says '11 theme tickets ... so 13'.
- [ ] (open) line 154: The hand-off proceeds with 206 open tickets against the operator's cap of 100. The remaining ~106 rest on 'ruling B', which is cited at lines 96 and 156 but never defined in this document or in the hand-off (lines 168-181).
- [ ] (fixed forward 10-30, needs re-review) line 96: Planned-basis application is inconsistent. 4381 moves to 0.80 (row 71) with no planned basis under the doc's own test, and its exit condition is 'in no 0.77-0.79 theme'. issue 6490 stays at 0.80 with no owner or children, and its skip rationale relies on edits, which line 109 defines as not activity.
- [ ] (fixed forward 10-30, needs re-review) line 35: The carry script treats every #N in the next train epic body as a listing (carry_milestone_items.sh:63-64, regex #(\d+)\b). Line 35 names issue 6566 as a non-listing reference, so an open issue 6566 in 0.77 would be carried to 0.78 at the 0.77 cut.
- [ ] (fixed forward 10-30, needs re-review) line 127: Cell S6 cannot be measured as written. Its host is 'E1 Qwen3.5 (size per issue 5321)', but issue 5321's body is truncated in the snapshot and its title gives no model size, so the model under test is undefined.
- [ ] (open) line 28: [nit] A5 says the 20 prompts are sent concurrently, but its YELLOW case is served one request at a time. State how the concurrent test is served in YELLOW.
- [ ] (fixed forward 10-30, needs re-review) line 31: [nit] AM-31 'selector gate' uses the word gate, which collides with the C324 GATE/LAB term. Rename it.
- [ ] (fixed forward 10-30, needs re-review) line 156: [nit] The sentence 'ruling B is the operator's ruling, and the planner executes it, and from FT-PARITY's own ticket cap' is garbled. Rewrite it.
- [ ] (open) line 94: [nit] issue 5968 is iceboxed as 'same shape as issue 5875', but issue 5875 is closed as completed. Choose one treatment for both.
- [ ] (fixed forward 10-30, needs re-review) line 69: [nit] 3999 is called 'standing' and kept at 0.80, but its only open child issue 3739 is milestone 0.73.0 (kids.tsv:15) and will be carried at the 0.73 cut. Say so.
- [ ] (fixed forward 10-30, needs re-review) line 63: [nit] EMG-001 (issue 5923) has four open children not in sections 2 or 3: 5996 and 5897 (0.74.0), 6101 and 5761 (0.75.0). Name their disposition.
