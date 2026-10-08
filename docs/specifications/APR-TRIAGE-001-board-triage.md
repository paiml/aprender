# APR-TRIAGE-001: board triage for issues, pull requests and branches

**Spec id:** `APR-TRIAGE-001` · **Version:** 1.0 (2026-10-08) · **Owner:** the release cop; one triage worker per phase
**Drop at:** `docs/specifications/APR-TRIAGE-001-board-triage.md`, as a docs-only PR. It rides `APR-LOOKAHEAD-002` row LA-0.
**Builds on:** `APR-EPIC-001` (on `main`, marked "draft v1.4": issue cap, intake ratchet, collapse), `APR-RELEASE-001` (rule R-3, dead branches), `FLOW-003` (fold means move). `FLOW-001` (PR cap) is not in this repo `[C]`. This spec adds measurements, an order of work and the fixes those specs wait on.
**Related:** `APR-071`, `APR-LOOKAHEAD-002`.
**Review:** one fresh-context lane read the first draft against `main` at `ad8fafc79`, a fresh `ls-remote` and the public pages. Its twelve findings, seven of them unsafe as written, are applied. The final text was not re-read.

**Marks:** `[V]` verified from public git or public GitHub pages at the time named · `[C]` reported by the cop, a worker, a review lane or an earlier document, or computed from verified numbers by the method stated · `[O]` operator statement · `[P]` proposal, not in force · `[A]` assumption · `[U]` unverified.

---

## ELI5

| Question | Answer |
|---|---|
| What is the board? | Open issues, open pull requests and remote branches. |
| How big is it? | 419 issues. 16 PRs against a cap of 10. 1,436 branches. |
| Is it over the issue cap? | Not known. The cap of 100 counts the direct children of epics, and nothing on `main` prints that number. Counting it is the first row. |
| Why does it stay big? | Two loops. A review vote is a commit, so each vote restarts a build that takes about an hour. And since 09-20, nine issues were opened for every PR merged. |
| What is the plan? | Count. Stop the two loops. Give PR places to trains. Drain by rule. Print the board every night. |
| What needs the operator? | Eleven answers in §3. One (#4526) has waited since 09-27. |

**How it works**
- Flow first, inventory second. A drain without the two fixes refills.
- An agent makes no write to a human's PR and never closes a human's issue.
- Nothing is removed that one push cannot restore, and nothing on the never-remove list is touched.

| | Yes / No |
|---|---|
| Is any issue closed without a receipt, a parent, a counted reason or a cited ruling? | No |
| Is any branch deleted before another ref holds its commit and is read back? | No |
| Can `main`, a release branch or a `keep/` branch be deleted? | No (R4a) |
| Does triage run a bulk job during a release pass? | No |
| Does this spec change a gate by itself? | No. TP1, TP2 and TP8 are gate changes and wait for the operator. |

## Purpose and terms posture

- The goal is an open-source Rust ML framework. Models are not sold and are not built to compete with any provider.
- No hosted-model output is ever a training target.
- Claude Code is used only by its paying account owner: no shared credentials, no free accounts.
- Public wording never positions a model as a replacement for a commercial service.

---

## §0 Operating assumptions

1. The cop is the only writer of labels, milestones and epic membership, and the only opener of issues. A triage worker proposes; the cop writes.
2. Every number in §1 is a baseline. Row TR-8 re-measures the board with the GitHub API, which the author could not use.
3. A `[P]` line is not in force until an operator block names it. Each row names what it needs.
4. **Authorship.** Agent items and the operator's own items show one login (B5). An issue is agent-minted only if the cop's mint log holds its number. A PR is an agent's only if the cop's log names the session that opened it. Everything else is human-authored, including what the operator typed himself. Unknown counts as human. `[U]`: how many open items the cop's logs cover.
5. Where this spec and a script disagree, the script wins: quote the line.
6. Rows that read PR state, push times or issue links need the GitHub API. Each says so.

**Terms.** *Release pass*: from a release's freeze until it is live (`APR-071`, Terms). *LAB*: a check that prints and cannot block. *GATE*: a check that can stop a merge or a release. *Lane*: a set of PR places for one purpose; a PR's lane is its milestone plus its epic's `owner:` label `[A]`. *Shepherd*: the one session that answers for a human's PR in the cop's report. *Staged*: listed under `staged:` in a kit file (`APR-LOOKAHEAD-002` K5). *`to_carry`*: the open items of a milestone without `must-carry`; the release script moves them at the cut. *E1*: epic #3998, the build epic of 0.71. *Receipt*: for an issue, a merged PR and its green falsifier; for a review, the reviewer's signed vote.

---

## §1 Ground truth (baseline 2026-10-08, 13:40Z to 14:40Z; never quote as current)

| # | Fact | Mark | Source |
|---|---|---|---|
| B1 | Open issues 419. Open PRs 11 at 13:40Z and 16 at 14:38Z: #4931 to #4935 were opened between 13:45Z and 14:10Z. | [V]; the 16 by PR refs, the public list still showed 11 | repository pages; `git ls-remote origin 'refs/pull/*'` |
| B2 | Open by priority label: P0 145, P1 135, P2 59. About 80 carry none of the three. | [V]; the 80 is [C] | label pages; subtraction |
| B3 | Open items by milestone: 0.71.0 238, `backlog` 105, 0.72.0 38, 0.73.0 16, 0.74.0 to 0.76.0 8. | [V] | milestones page |
| B4 | Open by label: `must-carry` 0, `needs-operator-signoff` 5, `automation` 10, `epic` 21. | [V] | label pages |
| B5 | Authorship. Agent items and operator-typed items show the same login. Told apart on the page: 5 open issues by one outside contributor (#4918, #4472, #4462, #3422, #3418) and one fork PR (#4634). #4472 is P0 and has sat in `backlog` since 09-26. | [V] | author page; PR page |
| B6 | #4634: a draft from a fork, opened 09-29, 1,117 commits, 1,677 files. Its CI runs wait for a maintainer's approval. One human reply, the same day. | [V] | PR page; `git diff --stat` |
| B7 | From #3598 (09-20) to #4927 (10-08): 1,330 numbers, 971 issues and 359 PRs. From #4678 (10-04) on: 159 issues. Peak: 140 issues a day on 09-23 to 09-25 `[C]`. | [V] counts | a number is a PR when `refs/pull/N/head` exists; dates from known items |
| B8 | Open issues were 827 on 09-25 and 432 on 10-07. | [C] | cop and epic report |
| B9 | PRs numbered 4400 to 4927: 178 opened, 74 merged (42%), 93 closed unmerged (52%), 11 open. At least 22 of the 93 are named in a later `main` commit message, so some were folds. | [V]; the 22 by the review lane | a merged PR's number ends a `main` commit subject; all 67 `main` commits since 09-27 have one |
| B10 | By queue-entry time `main` took 106 merges since 09-20 and 50 since 10-04 07:05Z. Its head entered the merge queue at 09:29:30Z. No PR has entered since, and the queue was empty at 14:38Z: 5 h 9 min. | [V] | `git log origin/main --first-parent`; `git ls-remote origin 'refs/heads/gh-readonly-queue/*'` |
| B11 | The 10 agent PRs open at 13:40Z held 112 commits: 31 review receipts, 9 merges of `main`, 5 roadmap regenerations, 13 merges of a stacked part branch (code), 54 other. 21 of the 31 receipts are on PRs that never merged `main`. #4813 merged `main` 3 times and its part-2 branch 13 times in 70 hours. | [V]; the split by the review lane | `git log <base>..<head>`; `merge-base --is-ancestor <c>^2 origin/main` |
| B12 | Generated files written by more than one open PR: `docs/roadmaps/roadmap.yaml` by five (#4813, #4913, #4916, #4917, #4933) and `contracts/contracts.nt` by four (#4634, #4911, #4913, #4916). Guards require both fresh inside the PR. Among the five new PRs, two change `scripts/cascade-publish.sh` and two change `scripts/release/autopilot.sh`. | [V] | `git diff --name-only`; `scripts/check_roadmap_fragment_required.sh:217`; `Makefile:661` |
| B13 | A push to a PR cancels its CI run and starts a new one. A run that reaches its verdict takes 52 to 66 minutes (52m03, 54m06, 65m51 today). The merge queue adds about an hour: `main`'s head entered the queue at 09:29:30Z and reached `main` between 10:31Z and 10:35Z. | [V] durations; the queue time is [C], from run order | `ci.yml:43-45`; Actions page 14:33Z; commit times |
| B14 | A review receipt is a commit under `evidence/pr-review/`, so each vote is a push. The `present` check already selects on the diff's patch id. Today two runs were cut short by a receipt-only push (#4923 at 13m29, #4813 at 18m00). From 13:23Z to 13:26Z four receipt-only pushes restarted CI on four PRs; at 14:33Z two of those runs were still queued. #4923's last code commit dates from 11:52Z; 2 h 41 min later its newest run had not started. | [V] times; that the short runs were cancelled is [C] | #4635; Actions page; `git log` of the PR heads |
| B15 | Branches: 1,431 at 13:40Z, 1,436 at 13:57Z. No PR at the tip: 1,201. By tip commit date 1,065 of them are older than 72 h, 939 older than 7 days, 274 older than 14 days. They include `main`, `fleet-state`, 2 `keep/*` and 4 `release/*`. Tip is a PR head: 230, of which 10 open, 4 merged, 216 neither. 194 of the 216 are older than 3 days. The 216 include `release/0.70.2` and `release/0.69.1-batch-2`. | [V] | `git for-each-ref`; PR head refs; `main` subjects |
| B16 | Every PR's commits stay at `refs/pull/N/head` (2,547 refs). 806 `refs/archive/*` refs already exist; a default clone does not fetch them. `keep/*` tags: 40. | [V] | `git ls-remote` |
| B17 | Nothing on `main` counts the board. The cap is `docs/roadmaps/epics.yaml:5` (100, andon 95). Line 2 names a reader, `scripts/check_issue_flow.sh`, that is not on `main`. An issue-tree lint exists only on branches `build-kaizen/4675-itl-a-check` and `-b-fetch` (no PR, 86 h old). Present: `scripts/check_pr_closes_issue.sh`, `scripts/check_census_derived.sh`, `scripts/nightly_prune_release.sh`, and two workflows that each update one standing issue. | [V] | `git ls-tree -r origin/main`; `coverage-nightly.yml:345`; `qwen-story-daily.yml:267` |
| B18 | Of 230 open train issues listed in the epic report (10-07): 44 are named by at least one merged commit; 80 belong to 8 row families (TR 16, ONT 17, PVL 13, EG 10, version-prefixed 10, QM 6, EXT 4, K 4). | [C] | the report's tables; `git log origin/main` |
| B19 | Build tickets with P0 sit in `backlog`: the nightly producers #4719 to #4721 and #4803. Three have no owner label. | [V] | P0 label page |
| B20 | Rules on `main`: cap 100 and andon 95 (`epics.yaml:5`); a new issue joins an epic within 24 h, at most 10 open PRs (`APR-EPIC-001` rule 5); `created(d) ≤ closed(d)` over 3 days, blocking after two days (rule 19); never collapse a P0 of the current or next train, an issue with an open closing-keyword PR, or an external reporter's issue (rule 20); a branch with no open PR and a tip older than 14 days is archived to `refs/archive/<branch>`, then deleted (`APR-RELEASE-001:577`, R-3). | [V] | those files |
| B21 | Rulings reported, on no file on `main`: the 100 counts direct children of epics; sub-tickets do not count, at most 5 per ticket; no PR open over 24 h; a PR red over 12 h is split; at most 4 rows per PR; an item that cannot land within 5 trains closes as `icebox`; only the cop opens issues. | [O], reported | operator rulings of 09-25 to 09-28 |
| B22 | #4526, "GEN-001: README/doc counts generated by one writer": README counts, the `CLAUDE.md` sample columns and the census. It does not name `roadmap.yaml`, `contracts.nt` or `shapes.ttl`. P0, `needs-operator-signoff`, opened 09-27. `census.json` already has one writer. #4635 (receipts leave the tree) is P2 and carries no sign-off label. | [V] | issue pages; `scripts/check_census_derived.sh` |
| B23 | #4702 asks that release day run no CRUX and read the night's smoke cells, refusing a missing one; it waits on #4690. The standing policy says a release ships on CRUX smoke on the two GPU hosts. It does not say where the cells come from. | [V] | issue page; `contracts/model-capability-ladder-v1.yaml:94-112` |
| B24 | 1,390 of 1,420 distinct branch tips carry one author email. 398 carry an `Agent:` line. Neither a login nor a trailer separates an agent's branch from the operator's own. | [V] | `git log --no-walk` over the tips |
| B25 | `rc-cut.yml:37` starts on a CI run for any `release/**` branch. `pr-review-quorum.yml:85` names merge-queue branches. `carry_milestone_items.sh:79` sends an open item that is not `must-carry` and not listed by the next epic to `backlog`. | [V] | those files |
| B26 | Every PR into `main` runs on the clean-room pool: `ci.yml` has no `paths:` filter. Docs-tier tickets: #3668 (docs-only PRs spend about 45 minutes in two guards), #4529 (tier router), #4472 (the human's P0). | [V] `ci.yml`; tickets [C] | `ci.yml:13-21`; epic report |

**Two loops and three leaks**

| | Mechanism | Evidence |
|---|---|---|
| Loop A, restart | A vote is a commit, and a push cancels the running build. So each vote costs a build of about an hour: 31 of 112 commits are receipts, 21 of them on PRs that never merged `main`. A PR that commits a generated file must also regenerate it after another PR merges, which changes the diff's patch id and voids the vote: 9 merges of `main`, 5 regenerations. | B11 to B14 |
| Loop B, churn | Rows and defects are opened as issues far faster than PRs merge: 971 issues against 106 merges since 09-20, and 159 against 50 since 10-04. The count is then held down by parking and collapsing. | B7, B8, B10 |
| Leak 1 | P0 has no budget: 145 of 419. | B2 |
| Leak 2 | `backlog` swallows build tickets and a human's P0, and the release script feeds it at every cut. | B5, B19, B25 |
| Leak 3 | A branch outlives its PR (216), and 1,201 have none. | B15 |

**What one merge costs today**

| Step | Minutes | Source |
|---|---|---|
| PR run to its verdict | 52 to 66 | B13 |
| A push before the verdict | the run starts again | B13 |
| A receipt push after a green run | 52 to 66 more | B14 |
| Merge queue | about 64 | B13 |
| Floor: code run, one receipt push, queue | 168 to 196 | sum |
| Floor with receipts off the head (TP2) | 116 to 130 | sum |

Why no PR entered the queue from 09:29Z to 14:38Z: two measured contributors (receipt pushes, B14; six new runs from 13:30Z to 14:10Z, B1). The verdict colours of today's runs were not readable. The rest is `[U]`.

---

## §2 Rules

| # | Rule |
|---|---|
| R1 | Flow before inventory: TR-1 and TR-2 are asked and started before any bulk close. |
| R2 | Human items. An agent makes no write of any kind to a human's PR: no comment, label, milestone, assignee, draft toggle, push or close. The one exception is a write an operator block names for that PR. On a human-authored issue an agent may set an owner label, a milestone and an epic link, and post a receipt. It never closes, collapses, parks or lowers it. |
| R3 | An issue closes in one of four ways: on a receipt; as `collapsed` into a named parent's checklist; as `icebox` with the reason "cannot land within 5 trains", trains counted; as `superseded`, naming the operator ruling by file and line, under an answered proposal. |
| R4 | A branch is removed only when its tip equals `refs/pull/N/head` of a closed or merged PR, or equals an archive ref written first and read back. Tags are never removed. No archive ref is ever overwritten. |
| R4a | Never archived or deleted. By name: `main`, `release/**`, `fleet-state`, `keep/**`, `gh-readonly-queue/**`, any branch a ruleset covers, any branch named in a file under `.github/workflows/` on `main`. By relation: the head or base of an open PR; a branch with a commit in `main..tip` whose author is not the fleet's commit identity; a branch a spec row names; a branch claimed as staged. |
| R5 | Bulk writes run one job at a time, at most 30 writes an hour `[A]`, and never during a release pass. One atomic push of at most 50 refs counts as one write `[A]`. |
| R6 | A row that can ride an epic's checklist gets no issue. An automated opener updates one standing issue per lane. A ticket has at most 5 sub-tickets (B21). |
| R7 | Every board number is a ratchet: tonight's value is at most last night's, or the report names, by number, the rows that raised it. |
| R8 | Counters are LAB. No new Python. A planted item lives in a fixture the counter reads, never on the public repo. |

---

## §3 Open proposals

The first option is the author's recommendation.

| # | Question | Options | Why the first |
|---|---|---|---|
| TP1 | Do PRs stop committing generated files? | **ONE-WRITER**: extend #4526 (README counts and census, waiting since 09-27) to `roadmap.yaml`, `contracts.nt` and `shapes.ttl`. CI computes and checks them; one writer regenerates them after merge or in the queue. It changes two gates: rule 2 of `check_roadmap_fragment_required.sh` and the `git diff --exit-code` of `make contracts`. · **COUNTS-ONLY**: #4526 as written. · **KEEP**. | Five open PRs write `roadmap.yaml` (B12). `census.json` already has one writer. |
| TP2 | #4635: do review receipts leave the PR head? | **OFF-HEAD**: a receipt is stored off the head (a side ref or a check run, as #4635 designs), bound to the diff's patch id. A vote needs no push. A comment is not a receipt: an outside account can post one. · **KEEP**. | Each vote restarts a build of 52 to 66 minutes (B13, B14). It takes about an hour off every merge. |
| TP3 | Old branches. | **ARCHIVE-14**: R-3 as written on `main`. First the job prints its two lists and the R4a exclusions; this answer names that listing. Then: delete branches whose tip is the head of a PR closed or merged at least 3 days ago `[A]` (194 at most); archive, then delete, no-PR branches with no push for 14 days (274 at most). Every delete is conditional on the checked sha: `git push --atomic --force-with-lease=<branch>:<sha> origin :<branch>`, or GraphQL `updateRefs` with `beforeOid`. · **ARCHIVE-7**: the same at 7 days (939). · **LIST-ONLY**. | 14 days is already the rule (B20), and 806 archive refs show the route works (B16). Restore is one push. |
| TP4 | #4634: a fork's CI on the self-hosted runners. | **SPLIT-AND-MIRROR**: the cop posts one reply, in words the operator's block gives, asking for a first PR small enough to read. Each later mirror push into this repo needs the operator's yes for that commit range. No agent mirrors fork code on its own. · **APPROVE-ONCE**: approve the waiting runs as they are. · **DECLINE-CI**. | 1,677 files cannot be read before they run on the build hosts. A branch pushed into this repo runs CI with no approval step. #4462, a human's P0, is on this gap `[C]`. |
| TP5 | Two sign-off tickets that later rulings answered. | **AS TABLED** below. | #4687 merged. #4701 is answered by the standing policy. |
| TP6 | Places for the 10 agent PRs. | **LANES** `[A]`: patch release 1; next train 5, with one place for each of its epics that has a PR ready; an epic the operator named 1; kit 1; second-next train 1; one kept free for a `main`-red fix. The third train uses the kit place only. A lane at its limit opens no new PR. · **FLAT**: keep one pool of 10. | Today the next train's build and release tooling holds 12 places and its verbs epic none (B1). |
| TP7 | What P0 means. | **THREE**: a human-reported regression in a shipped verb; `main` red; a `must-carry` row of the patch release or the next train. Everything else is P1 to P3. A human's own P0 keeps its label (R2). · **KEEP**. | 145 P0s rank nothing (B2). |
| TP8 | Docs tier (#4472, #3668, #4529): may a docs-only diff skip the x86 job and take one review? | **DOCS-TIER**: a diff that touches only files under `docs/`, none of them generated, runs the docs checks and one review `[A]`. The workflow on `main` computes the tier from the diff. Any other path runs the full job and the full quorum. · **KEEP**. | Every spec and kit PR costs a full build today (B26). #4472 is a human's P0, 12 days in `backlog`. |
| TP9 | #4696 (its title says "GATE CHANGE"): a thinking-on leg sampled over N seeds with a required closure rate. | **MEASURE**: run the sampled leg nightly and report the rate for `apr` and the reference. Set no required rate until three nights exist `[A]`. · **GREEDY**: keep. · **REFUSE**: the model is refused by name in thinking mode. | A rate set before it is measured is a guess. |
| TP10 | #4702: where do release day's CRUX smoke cells come from? | **READ-AFTER-4690**: release day runs the smoke until #4690 is on `main` (the night tests the binary that ships). After that it reads the night's cells for the same commit and refuses a missing one. · **RUN**: release day always runs the smoke. | Same judge, same commit, no GPU time on release day. It serves the 240-minute rule of `APR-071` B5. Until #4690 the night's cells are for another binary. |
| TP11 | Do human PRs count toward the cap of 10? | **OUTSIDE**: no. Each is listed with a shepherd. · **INSIDE**: yes, as ruled. | A human's PR waits on its author and the operator. Counting it takes a place an agent PR could finish in. |

**TP5's table**

| Ticket | Asks | Recommendation |
|---|---|---|
| #4526 | README counts and the census get one writer | It is TP1. |
| #4687 | one door to `cargo publish` | Merged as #4869 on 10-07 (`1df212813`). Close on its falsifier. The cop cites the sign-off under which it merged `[U]`. |
| #4701 | the full model ladder runs nightly; release day reads it | Answered by the standing policy (`model-capability-ladder-v1.yaml:94-112`): nightly, and it cannot stop a release. Its build rows have no home. Move them to #4721 as checklist lines, give #4721 an owner, then close #4701 as `superseded` with that link. |
| #4702 | release day runs no CRUX and reads the night's cells | Not superseded. It is TP10. It keeps its label. |
| #4696 | a sampled thinking-on leg | It is TP9. |

---

## §4 Rows, in order of expected value

Starts today with no answer: TR-8, TR-9, TR-19, the listings of TR-4 and TR-5, TR-6, TR-11's verdicts, the dry-run lists of TR-16 and TR-17.

**Phase 1: stop the loops**

| Row | Work | Needs | Done when |
|---|---|---|---|
| TR-1 | One writer for generated files. #4526's steps G1 to G4 for the counts. Two more rows, for the roadmap aggregate and for `contracts.nt` with `shapes.ttl`, each with a planted stale file that is red on `main`. | TP1 | No generated file is written by two open PRs. A fixture with a hand-edited generated block is red. |
| TR-2 | Receipts off the PR head, by #4635. Until it lands: all receipts for a reviewed diff go in one push, and nothing else is pushed to that PR until its run ends. | TP2 for the change; nothing for the interim rule | A PR voted on after CI starts reaches a green `present` check with no new commit on its head. |
| TR-3 | Docs tier. The cop moves #4472 out of `backlog` into 0.71.0 under E1, with an owner label, and writes nothing else on it. The tier itself is #4529's work. | TP8 | A docs-only PR's required checks end in at most 10 minutes `[A]`; before: 52 to 66. The tier classifier, given a fixture path list with one `.rs` path among docs paths, answers "full". |

**Phase 2: pull requests**

| Row | Work | Needs | Done when |
|---|---|---|---|
| TR-4 | Lanes bind at open time: a lane at its limit opens no new PR and drains by merging. No PR is parked or closed by a lane rule. | TP6; API | The nightly listing shows each PR's lane. No lane over its limit grew. |
| TR-5 | Age, as ruled (B21), from one nightly listing: PR, lane, hours open, hours red, pushes, non-code pushes. Red 12 h: split. Open 24 h: split into PRs that can land, or close and keep the branch (reopen to resume). A draft is open: it counts toward 10 and toward 24 h. Agent PRs only (§0.4). | API | No agent PR is over a limit at the nightly listing, or the listing names the common cause that holds it (a red `main`, a pool outage) and its ticket. |
| TR-6 | Size: at most 4 rows; no data dump in a code PR. The cop routes #4913's 29,298 lines of audit files to its worker. | nothing | Those files are out of #4913. |
| TR-7 | Human PRs: the cop's report, not the PR, names one shepherd and the age of the last human reply. #4634 waits for TP4. | TP4 for any write | The listing names the shepherd and the last reply. |

**Phase 3: issues**

| Row | Work | Needs | Done when |
|---|---|---|---|
| TR-8 | Count first. Land the issue-tree counter (branches `build-kaizen/4675-itl-a-check`, `-b-fetch`) as a nightly line: raw open; epics; direct children of epics; sub-tickets; by milestone; by priority; human-authored; no epic; no owner. | API | The line prints nightly from `main`. Children are counted from the tree, not by subtraction. A fixture with one orphan issue moves "no epic" by one. |
| TR-9 | Human-authored issues: each has an owner label and a train milestone; none is in `backlog`. | nothing | #4472 is out of `backlog` today. |
| TR-10 | TP5: close #4687 on its falsifier; move #4701's rows to #4721 and close it as `superseded`. | TP5 | `needs-operator-signoff` holds only the numbers the operator's block left open. |
| TR-11 | Done-on-`main` check. For each open issue a merged commit names, run its falsifier on the host and in the mode the ticket names. Green closes it with the receipt. No falsifier, no close. A reference is not a fix. Not candidates: a ticket the standing policy holds open (#4661 to #4666; #3598); an epic (#2373); a human-authored issue. | nothing | Each of the 36 candidates `[C]` has a verdict. |
| TR-12 | Collapse the row families of B18 into their parents' checklists, by `APR-EPIC-001` rule 20 as written. Never collapsed: a P0 of the patch release or the next train; an issue with an open closing-keyword PR; a human-authored issue. Runs after TR-14. | the cop cites the ruling behind rule 20 (the file says "draft v1.4"), or asks | N fewer, N printed by the job. Every collapsed number is a line in its parent. |
| TR-13 | `backlog`, 105 open. Each agent-minted item gets one verdict: ADOPT (an epic owner takes it, within the epic's budget) or ICEBOX with the counted reason (R3). An item no owner adopts and no count condemns stays open and is listed. Build tickets go to E1. | nothing | `backlog` holds no P0 and no build ticket. |
| TR-14 | Priorities, by epic owner. | TP7 | The P0 count equals TP7's three kinds plus the human-authored P0s R2 keeps, listed by number. |
| TR-15 | Milestone 0.71.0 drains before its cut. Each night every epic owner moves what it will not land to its real train or to a TR-13 verdict. | nothing | `to_carry` is at most 40 `[A]` three nights before the cut; 238 today. The 0.71 epics' budgets sum to 35 (`epics.yaml`). |

**Phase 4: branches**

| Row | Work | Needs | Done when |
|---|---|---|---|
| TR-16 | Closed-PR branches. Delete a branch only if, re-read at delete time: (a) the API says its PR is closed or merged; (b) `refs/pull/N/head` equals the branch tip in the same `ls-remote`; (c) R4a holds; (d) the PR closed at least 3 days ago `[A]`; (e) the delete is conditional on the checked sha. A moved tip is skipped and logged. | TP3; API | 194 fewer at most, each logged with its PR number and sha. |
| TR-17 | No-PR branches. Archive, then delete, when: (f) the last push, from the repository activity API, is older than 14 days; (g) the archive name is absent or already holds the same sha; on a clash write `refs/archive/<name>@<sha12>`; (h) the archive ref is read back and compared before the delete; (i) notice went out 24 h `[A]` before, by an explicit owner map, and a branch with no mapped owner was listed in the cop's report for 24 h; plus (c) and (e). | TP3; API | 274 fewer at most. `git ls-remote origin 'refs/archive/*'` lists each. |
| TR-18 | Standing sweep, nightly, with TR-17's conditions and notice. It starts only after TR-17 reports done. It skips branches merged into an open PR's head in the last 7 days `[A]`. | TP3; API | The nightly branch count shrinks or holds, or the report names the rows that raised it. |

**Phase 5: intake**

| Row | Work | Needs | Done when |
|---|---|---|---|
| TR-19 | The cop's report carries one board line each night: issues opened, closed by merge, closed otherwise by reason; open raw and counted; PRs opened, merged, closed unmerged by reason (age rule, fold, split, superseded); pushes per merged PR, code and non-code; minutes from last code push to merge; hours with no queue entry and their cause; branches; archive refs. | API | The line exists for three nights `[A]`. |
| TR-20 | Andon, `APR-EPIC-001` rule 19 as written. While it is on, the cop opens no issue; a new row becomes a checklist line on its epic, and a session works under the epic's number. | API; the cop says whether rule 19 is in force | The andon fires on a fixture burst. The first nightly line says whether it is on today. |

**What the drains can reach.** TR-10, TR-11, TR-12 and TR-13 are sized at 2, 36, 72 and 105 at most: 215 of 419. The 72 is 80 family rows less 8 parents `[A]`. The raw count therefore stays near 200 after every lever here. The cap counts direct children of epics, and that number has never been printed (B17). TR-8 comes before any claim about the cap.

**Targets**

| Measure | Baseline (10-08) | Target | By |
|---|---|---|---|
| Counted open issues | not printed | printed nightly | the first night after TR-8 lands |
| Receipt commits per merged PR | 31 on 10 open PRs | 0 | TR-2 |
| Minutes from last code push to merge, floor | 168 to 196 | 116 to 130 | TR-2 |
| Generated files written by two or more open PRs | 2 gated aggregates, on 5 and 4 PRs | 0 | TR-1 |
| Docs-only PR, minutes to verdict | 52 to 66 | at most 10 `[A]` | TR-3 |
| Open agent PRs | 15 | at most 10, by merging | before any new PR opens |
| Human-authored issues in `backlog` | 1, for 12 days | 0 | today |
| Oldest item in the operator's sign-off queue | 11 days | under 48 h `[A]` | every night |
| P0 share of open issues | 145 of 419 | TP7's three kinds only | TR-14 |
| Branches | 1,436 | under 1,000 | TP3's first job |

---

## §5 Stop conditions

One numbered question, options in words. Other rows keep moving.

1. Any write to a human's PR that no operator block names; closing, collapsing, parking or lowering a human-authored issue.
2. A gate change other than TP1, TP2 and TP8 as answered.
3. Approving a fork's workflow runs, or pushing fork code into this repo (TP4).
4. Deleting or archiving a ref without TP3; touching a ref on R4a's list; removing any tag; overwriting an archive ref; an unconditional delete.
5. A ruleset, secret or credential change.
6. A close on an issue an operator ruling placed, or a bulk close under a rule the cop cannot cite a ruling for.
7. Three failed bulk writes in a row `[A]`: stop the job and report the response.
8. A bulk job that would run during a release pass.

---

## §6 Falsifiers

| Rule or row | Falsifier | Anti-vacuity arm |
|---|---|---|
| R2 | The nightly line lists every human-authored item with its state. An item whose close, collapse, park, priority change or PR write is in the cop's own write log, with no operator block naming it, is red. | The list holds at least the six items of B5, whatever their state. |
| R3 | A closed issue in the cop's write log with no receipt link, no `collapsed` parent, no counted `icebox` reason and no cited ruling is listed. A fixture bare close is listed. | The listing read the last 24 h of closes and found at least one. |
| R4, R4a | Before a delete, one `ls-remote` shows the tip equal to `refs/pull/N/head` or to a `refs/archive/*` ref. A fixture branch with a unique commit and no archive is refused. A fixture named `release/x` is refused. | After the job, one sampled archive ref is fetched and its sha equals the logged tip. |
| R5 | The job log shows one job, at most 30 writes in any hour, and no write inside a pass window. | The log is not empty. |
| R7 | A board number above last night's with no row numbers named is red. | Last night's line exists. |
| TR-1 | A fixture with a hand-edited generated block is red. The nightly count of generated files written by two or more open PRs: target 0. | The count ran over at least two open PRs. |
| TR-2 | Receipt commits per merged PR: target 0. Merges of `main` per merged PR: at most 1 `[A]`. | The PR has at least one code commit. |
| TR-3 | The classifier answers "full" for a fixture path list with one path outside `docs/`. | It answers "docs" for a fixture with only `docs/` paths. |
| TR-4 | A lane over its limit that grew is named. | Every open PR has a lane. |
| TR-5 | A PR over a limit that the listing does not show is red. Each close by age is listed with its hours. | The listing holds at least one PR. |
| TR-8 | Raw minus epics minus children equals the sub-tickets and unparented issues the line lists by number. | A fixture orphan moves "no epic" by one. |
| TR-12 | Every collapsed issue's number appears as a line in its parent. A missing line is red. | The parent's checklist grew by the count collapsed. |
| TR-13 | An `icebox` close without a train count is listed. | At least one verdict exists. |
| TR-16 | After the job each deleted branch's logged sha equals `refs/pull/N/head`. | The log holds at least one delete. |
| TR-17, TR-18 | `refs/archive/<name>` equals the deleted branch's logged tip. No archive ref's sha changed between the listings before and after the job. | One sampled archive ref is fetched and matches. |

**What would prove this spec wrong**

- After TR-2, pushes per merged PR do not fall: votes were not the restart cost.
- After TR-1, merges of `main` per PR do not fall: generated files were not the conflict source.
- Today's five-hour gap has a cause outside both loops, such as an x86 job that is red on `main` itself: the first fix is that job, not this spec.
- Most of the 93 unmerged PRs since #4400 were folds, planned splits or age closes that reopened: B9 overstates waste. At least 22 already look like folds.
- The counted number TR-8 prints is already under 100: the cap holds and only the raw count is noise.
- More than 10 of TR-11's 36 candidates are still broken `[A]`: the done-on-`main` drain is small.

---

## §7 Report

Nightly, in the cop's report: the board line (TR-19); one table of PRs by lane and age; one table of the drains (candidates, done, refused, why); the operator's queue by age; the cost of one merge, as in §1.

---

## Appendix: not checked

- The GitHub API was closed to the author. Counts come from public pages and git refs. Authors, ages and sub-issue links of most issues were not read.
- CI verdict colours. The Actions page gave run durations, not results. That the short runs were cancelled by a push is inferred from commit times.
- The number of runners in the clean-room pool, the merge queue's batch size and its time per entry.
- B7 dates numbers by eight known items.
- B9 reads "merged" from `main` commit subjects.
- B15 matches a branch to a PR by its tip only, and reads age from the tip's commit date. The push date needs the API.
- B18 covers 230 issues listed on 10-07, not all 419.
- Whether the cop keeps a mint log that covers every open item (§0.4).
- Whether GraphQL `updateRefs` deletes a ref when given a zero `afterOid`: the job tests it on a scratch ref it created.
- The final text was not re-read by a review lane. The cop's quorum has not reviewed this file.
