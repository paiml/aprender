# APR-LOOKAHEAD-002: look-ahead workers land kits, not branches

**Spec id:** `APR-LOOKAHEAD-002` · **Version:** 2.0 (2026-10-08) · **Owner:** the release cop owns the slots; each worker owns its train
**Drop at:** `docs/specifications/APR-LOOKAHEAD-002-train-kits.md`, as a docs-only PR (row LA-0).
**Replaces:** `APR-LOOKAHEAD-001`. Its v1.1 is on branch `la/4540-lookahead` only. Its v1.2 was handed to the cop on 09-28 and is on no ref `[C]`. Kept: the directive, three slots, never-stop, the 2-hour heartbeat, the PR budgets, the throttle ladder. Replaced once §3 is answered: the 4-hour push floor, the hand-off file, the freeze on docs.
**Related:** `APR-071` (0.70.3 and 0.71), `APR-TRIAGE-001` (the board).
**Review:** one fresh-context lane read the first draft against `main` at `ad8fafc79` and the public pages. Its ten findings, five of them unsafe or unworkable as written, are applied. The final text was not re-read.

**Marks:** `[V]` verified from public git or public GitHub pages at the time named · `[C]` reported by the cop, a worker, a review lane or an earlier document, not re-checked · `[O]` operator statement · `[P]` proposal, not in force · `[A]` assumption · `[U]` unverified.

---

## ELI5

| Question | Answer |
|---|---|
| What is a look-ahead worker for? | To make each of the next three trains ready before it starts. |
| Who is on which train? | 0.71: `la-0.71`. 0.72: `la-0.72`. 0.73: `la-0.73`. |
| What went wrong? | 142 look-ahead branches and no readiness file on `main`. In 11 days one look-ahead PR merged, and it was release tooling. |
| Why? | The worker prompt said: update the hand-off file, push the branch. It named no PR for it. Every row became a branch. No PR place was kept for readiness, and every PR costs a full build. |
| What changes? | The output is a **kit on `main`**. One working branch per slot. A short list of staged branches. One PR place kept for kits. One public number each night. |
| Does look-ahead ever stop? | No. It throttles. |

**How it works**
- A kit is seven things a train needs before it starts: criteria, baselines, contract drafts, a must-carry list, staged PRs, asked questions, risks.
- A kit is one flat file per train under `docs/lookahead/`. A kit PR changes nothing else.
- A nightly line prints each train's kit score. That score is the only progress measure.

| | Yes / No |
|---|---|
| Does a push count as progress? | No |
| May a look-ahead branch add Python? | No |
| May a slot remove, archive or force-push a ref? | No. It proposes; one job removes (`APR-TRIAGE-001` TP3). |
| May a slot's worker be given another train's rows? | No. It is dedicated `[O 2026-09-27]`. |
| Does a release cut stop kit PRs? | Yes today: every PR runs on the clean-room pool. That ends with the docs tier (`APR-TRIAGE-001` TR-3) and LP2. |
| Does this spec change a gate? | No |

## Purpose and terms posture

- The goal is an open-source Rust ML framework. Models are not sold and are not built to compete with any provider.
- No hosted-model output is ever a training target.
- Claude Code is used only by its paying account owner: no shared credentials, no free accounts.
- Public wording never positions a model as a replacement for a commercial service.
- A number measured against another open-source tool is an engineering parity measurement at a pinned commit.

---

## §0 Operating assumptions

1. **Directive** `[O 2026-09-27]`: "Going forward I ALWAYS want ONE dedicated worker on the next three epics…" Three slots, one worker each. Today: slot 1 is 0.71, slot 2 is 0.72, slot 3 is 0.73 `[O 2026-10-08]`.
2. A patch release (0.70.3, 0.71.1) has its own crew. It is not a slot and does not rotate the slots.
3. Look-ahead never stops `[O 2026-09-28]`. Under pressure a slot throttles (§6).
4. Order for every contended runner, GPU, queue place and review lane: patch release, the next train's build work, slot 1, slot 2, slot 3.
5. A worker never mints a ticket, never writes a label or a milestone, and never touches a release PR, a human's PR or another worker's PR. A row rides its train's epic as a checklist row and uses the epic's number as its ticket (`APR-071` §0.5).
6. A `[P]` line is not in force until an operator block names it. §2's last column says what holds today. Until a rule here is in force, v1.2's text for it holds.
7. Every number in §1 is a baseline. Where this spec and a script disagree, the script wins: quote the line.
8. Model routing follows the fleet's routing spec `[C]`. Fable is not used `[O]`.

**Terms.** *Slot*: one worker and one train. *Kit*: the seven parts of §4, on `main`. *Kit PR*: a PR that changes only files under `docs/lookahead/`. *Working branch*: `la-<NN>/wip`, with NN = 71, 72, 73. *Staged branch*: a branch listed under `staged:` in a kit file; it is one of the slot's next PRs. *Release pass*: from a release's freeze until it is live (`APR-071`, Terms). *Tick*: the cop's hourly pass. *Iteration*: one loop of the worker prompt. *5-hour budget*: the account's rolling usage window, as the cop reports it `[C]`. *LAB*: a check that prints and cannot block. *Criteria ids*: 0.71 uses B1 to B5 and V1 to V7 (`APR-071`); 0.72 uses T1 to T6; 0.73's six are written RE1 to RE6 here, because E1 to E9 are epic ids (v1.2 called them E1 to E6).

---

## §1 Ground truth (baseline 2026-10-08, 13:40Z to 14:40Z; never quote as current)

| # | Fact | Mark | Source |
|---|---|---|---|
| G1 | No path on `main` matches `lookahead`: no `docs/lookahead/`, no look-ahead spec, no worker prompt. | [V] | `git ls-tree -r origin/main` |
| G2 | `la-71/*`: 57 branches. One is a PR head (#4923). 43 are older than 3 days. 19 tips sit inside another `la-71` branch. 19 add Python: 8 distinct files, none on `main`. | [V] | `git for-each-ref --contains`; PR head refs; merge-base diff |
| G3 | `la-72/*`: 77 branches. One is a PR head (#4911). 63 are older than 3 days. 43 tips sit inside another `la-72` branch. 7 are flagged for added Python (32 paths): 23 are already on `main`, all on one branch; 9 are new, on 6 branches. | [V] | same |
| G4 | `la-73/*`: 4 branches, no PR, 2 add Python (11 files). `la/*`: 4 branches from 09-27 and 09-28; one adds a Python file. Total 142. Eight `la73/*` branches (no hyphen, 09-27 and 09-28, no PR) are outside the 142; whether they are look-ahead is `[U]`. | [V] | same |
| G5 | Readiness files exist only on branches. `docs/lookahead/0.71-dor.yaml` is on `la-71/dor-071`. 47 paths under `docs/lookahead/0.73/` are on `la-73/3999-runs-everywhere-drafts`: 27 `.md`, 10 `.py`, 6 `.tsv`, 4 contract drafts, no kit file. `docs/lookahead/0.72.md` is on `la/4540-lookahead` only; no `la-72/*` tip has a path under `docs/lookahead/`. The 0.71 checker is Python. | [V]; counts by the review lane | `git ls-tree` on those branches |
| G6 | Open PRs at 13:40Z: 11. Patch release 1 (#4927). 0.71 build and release tooling 7 (#4813, #4874, #4916, #4917, #4920, #4921, #4923; the last is by `la-0.71`). 0.71 verbs 0. Slot 2 code 1 (#4911, by `la-0.72`). APR-EMBED-001 1 (#4913, by worker `apr-embed`). A human's fork draft 1 (#4634). By 14:38Z five more were open (#4931 to #4935, the D1 to D5 fixes of `APR-071`); #4932 is by `la-0.72`. | [V] trailers and PR refs; trains read from titles `[A]` | pull requests page; `Agent:` trailers; `git ls-remote` |
| G7 | #4913 adds 31,546 lines in 61 files; 29,298 of them are 31 files under `docs/audits/`. | [V] | `git diff --numstat` |
| G8 | v1.1, on the branch. Invariant I4: "every slot has a current handoff file on `main`, updated within 24 h." Worker prompt, step 4: "ticket … → branch → PR"; step 6: "Update docs/lookahead/${TRAIN}.md". Slots 2 and 3 merge "specs, contracts and docs only; code as draft PRs". | [V] | `la/4540-lookahead:docs/specifications/APR-LOOKAHEAD-001-rolling-epic-workers.md` |
| G9 | v1.2 adds: "≥ 1 commit pushed to origin per slot every 4 h"; step 6 gains "Push the branch to origin"; during a cut "no PRs armed, no CI started … all slots continue non-merge work on branches". Its rotation event: when train N is on crates.io, the slots move to N+2, N+3, N+4. | [C] | the advisor's copy of v1.2 |
| G10 | Every PR into `main`, docs-only or draft, runs its required jobs on the clean-room pool: `ci.yml` has no `paths:` filter and no draft condition, and `gate` needs `x86-main`. A branch push with no PR starts no workflow. A run that reaches its verdict took 52 to 66 minutes today. | [V] | `ci.yml:13-21`, `:142`, `:605`; `on: push` of all 32 workflows; Actions page 14:33Z |
| G11 | Tags: `v0.70.0` 10-02 23:42Z, `v0.70.1` 10-04 00:15Z, `v0.70.2` 10-07 16:11Z. 0.70.3 is in flight (`release/0.70.3`, PR #4927). Four cuts in seven days. | [V] | `git tag`; `git ls-remote` |
| G12 | No open issue carries `must-carry`. | [V] | label page |
| G13 | 0.71: its exit criteria are on no file on `main`; 0 of 10 top rows are ready; V1 has no baseline; V2 has no ticket; V3 and V7 have no pass sentence. The fleet's plan puts the 0.71 tag on 10-17 and the 0.72 cut on 10-26. | [C] | `APR-071` §1; epic report |
| G14 | 0.72: T1 to T6. T2 (throughput against Unsloth) waits for a ruling: slip, or an fp32 fallback. T4 and T5 are blocked on #4418, a 0.71 ticket owned by `la-0.71`. T6 is APR-EMBED-001's rows (#4898). 21 `la-72/r15a-*` branches hold its stacked code. | [C]; branches [V] | `docs/reports/0.7x-epic-reports.md` on branch `docs/0.7x-epic-reports` |
| G15 | 0.73: RE1 needs a wgpu forward pass for Qwen3.5 that is not written; whether it moves to 0.74 waits for a ruling. Hosts for the Metal and aarch64 receipts are not budgeted. | [C] | same report |
| G16 | Epic owners by `owner:<worker>` label: #3598 (E2) and #3994 (E3) `la-0.71`; #4000 (E5) `la-0.72`; #3999 (E6) `la-0.73`; #3998 (E1) `build-kaizen`. #4898 (APR-EMBED-001, 0.72.0) and #4381 (E4, 0.71.0) have none. | [V] | `labels/epic` page |
| G17 | `docs/roadmaps/epics.yaml` on `main` disagrees with the epic issues in three places: E5 "Agent Ready" against "Train What You Serve" (ruled 09-27), E6 "llama.cpp Parity" against "Runs Everywhere" (ruled 09-27), E1 in train 0.70 against 0.71. It has no entry for #4898. | [V]; the rulings [C] | `epics.yaml:9`, `:13`, `:14`; `labels/epic` page; v1.2 §2a |
| G18 | No commit on any `la-71`, `la-72` or `la-73` branch from 09-30 02:33Z to 10-03 07:24Z: 76.8 hours. In that gap `main` took 5 merges and `v0.70.0` was tagged. Cause `[U]`. | [V], committer time | `git log --remotes='origin/la-7*/*' --not origin/main` |
| G19 | Slots had a commit in 36, 32 and 22 of the 66 four-hour windows since 09-27 13:40Z. | [V], committer time | same |
| G20 | One PR from a look-ahead branch merged in 11 days: #4910, release tooling, 35 commits, trailer `Agent: la-0.71`. It is the only one of 67 `main` commits since 09-27 with a look-ahead trailer. | [V]; head branch name read once from the PR page | `git log origin/main` |
| G21 | A contract added under `contracts/` rewrites generated files: branch `la/4540-lookahead` (six look-ahead contracts) changes `README.md`, `contracts/census.json` and `contracts/contracts.nt`. | [V] | `Makefile:661`; `git diff --name-only origin/main...` |

**Five whys**

| # | Why | Because |
|---|---|---|
| 1 | No readiness file on `main` | The invariant wanted one within 24 h (G8, I4). The prompt said update the file and push the branch. It named no PR, and nothing public printed I4. |
| 2 | 142 branches | The loop is one item per iteration, each as ticket, branch, PR (G8). With PR budgets of 2, 1 and 1 the branches queued: 62 of 134 tips sit inside another branch of the same slot (G2, G3). |
| 3 | The PR places the slots had went to code | #4910, #4911 and #4923 are code (G6, G20). #4932 is a 0.71 gate-script fix by the 0.72 worker. No place was kept for readiness. |
| 4 | A readiness PR is not cheap | Every PR costs a 52-to-66-minute run on the clean-room pool and a review quorum (G10). A contract rewrites generated files other PRs write (G21). v1.2 also froze slot PRs during a cut, and four cuts fell in seven days (G9, G11). |
| 5 | Nobody saw it, nor a 77-hour stop | The heartbeat, I4 and the readiness count live in the cop's private state. The one floor that was measured counted pushes, and it was met in about half the windows (G18, G19). |

**Mechanism:** readiness had no reserved PR place, no cheap path to `main` and no public number.

---

## §2 Rules

| # | Rule | In force |
|---|---|---|
| R1 | A slot's output is its kit on `main` (§4). A push is not progress. | after LP1 |
| R2 | One working branch per slot: `la-<NN>/wip`. The slot pushes it at the end of every iteration; origin is never more than 2 h behind the worker's local tip `[A]`. | after LP3 |
| R3 | Staged branches, the slot's next PRs in order: slot 1 at most 5 (v1.2's "first 5 PRs"); slot 2 at most 3 `[A]`; slot 3 at most 1 `[A]`. A new branch needs a free place. A slot above its limit creates no branch and proposes verdicts (LA-1). | after LP3 |
| R4 | Open PRs: slot 1 at most 2, slots 2 and 3 at most 1 each. Slots 2 and 3 merge docs only; their code waits as staged branches, not draft PRs, because a draft runs the full CI (G10). A slot opens no code PR until its K1 is on `main`. A kit PR uses the kit place (`APR-TRIAGE-001` TP6); slot 3 uses only that place. | budgets and docs-only now (v1.2); the rest after LP3 and TP6 |
| R5 | No working branch, staged branch or look-ahead PR adds a Python file. A port goes to Rust or to bash with awk. | now |
| R6 | A kit PR changes only files under `docs/lookahead/`. It touches no published path (`crates/`, `src/`, `Cargo.toml`, `Cargo.lock`), nothing under `contracts/` and no generated file. | now |
| R7 | A release pass reserves the GPU hosts, the clean-room pool and queue priority. Today every PR runs on that pool, so during a pass a slot opens and pushes no PR and works on its working branch. After the docs tier (`APR-TRIAGE-001` TR-3) and LP2, a kit PR may open and merge during a pass cut on a release branch. No kit PR merges between the T-2 GO and the tag of a cut made on `main`. | the freeze now (v1.2); the rest after TR-3 and LP2 |
| R8 | Proof of life is a heartbeat every 2 h and the working branch (R2). A slot is never paused. | heartbeat now; the 4-hour push floor ends after LP1 |
| R9 | An operator question is asked when it is found. None is first raised at a cut. | now |
| R10 | A data file, an audit dump or a corpus is not part of a code PR. | now |
| R11 | A slot removes, archives or force-pushes no ref. It proposes verdicts. Removal is one job under `APR-TRIAGE-001` TP3 and its never-remove list (R4a). | now |
| R12 | A slot's worker is dedicated: the cop gives it no row outside its train. An epic the operator named with its own worker is not a slot's: APR-EMBED-001 (#4898, worker `apr-embed`) `[O 2026-10-07]` holds its own PR place. | now |

---

## §3 Open proposals

The first option is the author's recommendation.

| # | Question | Options | Why the first |
|---|---|---|---|
| LP1 | How does a slot show progress and life? | **KIT**: the nightly kit score, a heartbeat every 2 h, and one working branch never more than 2 h behind `[A]`. The 4-hour push floor ends. · **PUSH**: keep v1.2. | The floor counted pushes and was met in about half the windows (G19). In 11 days it put nothing on `main` (G1). Never-stop is unchanged. |
| LP2 | What does a release pass freeze? | **RESOURCES**, after the docs tier, for a pass cut on a release branch: GPU hosts, clean-room pool, queue priority. A kit PR may open and merge. · **ALL**: keep v1.2's freeze on every slot PR. | 0.70.3 runs on a release branch, so `main` need not stand still for it. A cut made on `main` still freezes merges from its T-2 GO to its tag. |
| LP3 | Branch limits per slot. | **5 / 3 / 1** staged, plus one working branch each; code waits staged, not as a draft PR. · other numbers. | 5 is v1.2's own hand-off size. 3 and 1 are assumptions. A draft PR costs a full build per push. |
| LP4 | The 142 existing branches. | **DRAIN**: each gets one proposed verdict: fold, stage or archive. Removal is `APR-TRIAGE-001`'s one archive job, after TP3. An open PR's head and `la/4540-lookahead` are never archived. · **KEEP**. | 62 of 134 tips are already inside another branch. An archive is a ref: restore is one push. |
| LP5 | Which three trains do the slots hold? | **NEXT-UNSHIPPED**: the three lowest trains whose `x.y.0` is not live: 0.71, 0.72, 0.73 today. They rotate when 0.71.0 is live. A patch release never rotates them. · **V1-RULE**: v1.2 as written: 0.70.0 went live on 10-02, so the slots are 0.72, 0.73, 0.74 and a 0.74 worker is spawned now. | It is what runs today, what the operator named on 10-08, and where the fleet's own report puts the rotation. 0.71's kit is the least ready. A fourth planning worker adds issues and branches to a board over its caps. |

---

## §4 The kit

One file per train: `docs/lookahead/<train>.yaml`. It is flat, one item per line, so bash and awk can read it.

```yaml
train: "0.71"
theme: "Verbs Are Fast"
criteria:
  - {id: V1, pass: "…", falsifier: "…", command: "make …", ruling: C332}
baselines:
  - {criterion: V1, receipt: docs/lookahead/0.71/receipts/v1-baseline.json}
contracts:
  - {row: 1, draft: docs/lookahead/0.71/contracts/….yaml, falsifier: "…"}
must_carry:
  - {issue: 4418, size: M, owner: la-0.71, needs: []}
staged:
  - {branch: la-71/v6-tracing, rows: 3}
questions:
  - {id: Q1, text: "…", asked: 2026-10-08, answer: open}
risks:
  - {risk: "…", mechanism: "…", tripwire: "…"}
```

| Part | An item is present when | Slot 1 needs | Slot 2 needs | Slot 3 needs |
|---|---|---|---|---|
| K1 criteria | it has `id`, `pass`, `falsifier`, `command`, and `ruling` is a ruling id or `asked` | every criterion | theme ruled; every criterion listed with `id` and `pass`, or `asked` | theme and criteria proposed |
| K2 baselines | its `receipt` is on `main` under `docs/lookahead/<train>/receipts/` and names host, model sha256, pin and n | one per criterion that has a number | the plan and the first baseline | the plan |
| K3 contract drafts | its `draft` is on `main` under `docs/lookahead/<train>/contracts/` and names a planted falsifier. It moves to `contracts/` in the row's own PR. | top 10 rows `[C]` | top 5 rows `[C]` | none |
| K4 must-carry | the entry has a size, an owner and its dependencies. The slot proposes; the cop labels. | yes | none | none |
| K5 staged PRs | the branch is on origin and `git merge-tree --write-tree origin/main <branch>` exits 0. At most 4 rows, no Python. | first 5 | first 2 `[A]` | none |
| K6 questions | the entry has an `asked` date | every question | every question | every question |
| K7 risks | it has a mechanism and a tripwire | top 3 `[A]` | top 3 `[A]` | top 3 `[A]` and the top 5 rows `[A]` |

- **Kit score** = required items present ÷ required items. `none` means not required. The checker holds this table and prints the score nightly from `main`.
- **A staged branch is never rebased.** One that no longer merges cleanly takes one merge of `main`: a normal push, which starts no CI because it has no PR (G10).
- **A listed staged branch is claimed.** It is outside the branch sweep (`APR-TRIAGE-001` R4a).

**Targets**

| Measure | Baseline | Target | By |
|---|---|---|---|
| Kit files on `main` | 0 of 3 | 3 of 3 | 72 h after LA-0's PR has a place `[A]` |
| Slot 1 kit score | not printed | every required item present | 3 nights before the 0.71 cut `[A]` |
| Look-ahead branches | 142 | at most 12 (3 working, 9 staged) plus open PR heads | after LP3, LP4 and the archive job |
| `.py` files new to `main` on look-ahead branches | 29 | 0 | before a branch is staged |
| Questions first raised at a cut | not counted | 0 | every cut |
| Longest gap with no slot commit | 76.8 h | heartbeat gap at most 2 h; no gap over 3 h | every night |

---

## §5 Rows, in order of expected value

| Row | Work | Needs | Done when |
|---|---|---|---|
| LA-0 | The cop's row. One docs-only PR, 4 files: this spec; `APR-TRIAGE-001`; `docs/prompts/lookahead-worker.md` (appendix A); `docs/lookahead/slots.yaml` naming train, worker and epic for each slot. The branch is pushed at once; workers read it there. The PR takes the kit place when one opens. | nothing | The four files are on `main`; `slots.yaml` lists three slots. |
| LA-0b | One kit PR per slot: `docs/lookahead/<train>.yaml` in §4's shape. 0.71 is converted from `0.71-dor.yaml` and `APR-071`. 0.72 and 0.73 are written new from the epic report. No `.py` and no `.tsv` is carried. | LA-0 | The file is on `main` and the checker reads it. |
| LA-1 | Verdict list. Each slot hands the cop one TSV: branch, tip sha, verdict, target. Verdicts: FOLD (into a named kit PR or staged branch), STAGE, ARCHIVE. Nothing is removed. | nothing for the list; LP3, LP4 and TP3 for removal | The TSV's row count equals `git ls-remote --heads origin 'la-<NN>/*'`. After the archive job each slot is at or under R2 and R3. |
| LA-2 | Python. Each slot lists its `.py` files new to `main` (8, 9, 11, and 1 on `la/*`) with PORT or DROP. | nothing | 0 on working branches, staged branches and PRs. |
| LA-3 | The kit checker, in bash with awk, and one line per slot in the cop's nightly report. LAB. | LA-0b | Removing one required item from a copy of a kit file lowers the printed score by one item. |
| LA-4 | The cop's row. `docs/roadmaps/epics.yaml` is brought to the rulings: the two theme names, E1's train, an entry for #4898 (G17). | the cop cites the rulings, or asks | The file and the epic issue titles agree. |

**Slot 1, train 0.71 (`la-0.71`)**

| Row | Work | Done when |
|---|---|---|
| S1-a | K1: B1 to B5 and V1 to V7 from `APR-071` into `docs/lookahead/0.71.yaml`. | K1 is complete for slot 1. |
| S1-b | K4: propose the `must-carry` set to the cop, sized and with dependencies. #4418 is in it: it blocks 0.72's T4 and T5. The cop labels after `APR-071` P3 and P4 are answered. | The proposal is in the kit file. |
| S1-c | K2: the V1 and V2 baselines on both GPU hosts, through a runner job or a make target, outside a release pass. | Two receipts are named in the kit file. |
| S1-d | K5: list the first five PRs under `staged:`. V6 (tracing) is first; it is cut from `main` after the 0.70.3 merge-back. | Five entries, each merging cleanly. |
| S1-e | K6: one pass sentence and one falsifier each for V3 and V7, asked as questions. | Both are `asked`. |

**Slot 2, train 0.72 (`la-0.72`)**

| Row | Work | Done when |
|---|---|---|
| S2-a | K1: T1 to T6 in `docs/lookahead/0.72.yaml`. T6 lists APR-EMBED-001's rows by reference to #4898; its worker is `apr-embed`. Ask T2 now: slip, or an fp32 fallback. | K1 is complete for slot 2; T2 is `asked`. |
| S2-b | K7: #4418 as a risk with a tripwire: if it leaves 0.71, T4 and T5 cannot pass. | The risk is in the kit file. |
| S2-c | K2: the Unsloth comparison plan with a committed command, same GPU, same model. The command adds no Python to this repo: the reference runs from its upstream command line at a pinned commit. If that needs a driver file, ask (stop 5). | The plan and the command are in the kit file. |
| S2-d | #4911 is a code PR from slot 2, outside the docs-only rule (R4). The operator rules: finish it, or return it to a staged branch. #4932 is a second PR and a 0.71 row (R12): the cop moves its ownership to a build session. No further `r15a` PR opens before K1 is on `main`. | Slot 2 holds at most one open PR, and none outside 0.72. |
| S2-e | #4913 is not slot 2's (R12). The cop routes its audit files (`APR-TRIAGE-001` TR-6) to `apr-embed`. | Slot 2 has pushed nothing to #4913. |

**Slot 3, train 0.73 (`la-0.73`)**

| Row | Work | Done when |
|---|---|---|
| S3-a | K1: RE1 to RE6 in `docs/lookahead/0.73.yaml`, from the drafts on `la-73/3999-runs-everywhere-drafts`. Ask now whether the wgpu forward pass is 0.73 or 0.74. | K1 is complete for slot 3; the question is `asked`. |
| S3-b | K2: a measurement plan that names a host for each backend: wgpu, Metal, aarch64, CUDA. | The plan is in the kit file. |
| S3-c | K6: ask where the four CRUX rows #3739, #3774, #3795, #3797 belong. | `asked`. |
| S3-d | K7 and the top 5 rows, ranked. | Three risks and five rows are in the kit file. |

---

## §6 Rotation, throttle, place

- **Rotation** (after LP5). Slot 1 is the lowest train whose `x.y.0` is not live. When it is live, slot 2 becomes slot 1 and slot 3 becomes slot 2. The cop spawns the new slot 3 first, then moves the `owner:` labels and edits `docs/lookahead/slots.yaml` in one PR, so no train is ever unowned (v1.2, Proposition 1).
- **Continuity** (v1.2 §0.5). The worker of a shipped train stays with its epics until they close. Its kit file stays on `main` and is scored nightly until every criterion in it has closed, including criteria rolled to `x.y.1` (`APR-071` P4). It ranks with the train's build work.
- **Throttle** (v1.2 `[C]`). Under 70% of the 5-hour budget all slots run at full pace. From 70% slot 3 drops to heartbeat and working branch. From 85% slot 2 does too. Slot 1 always runs.
- **Place.** One PR place is kept for a kit PR (`APR-TRIAGE-001` TP6). Slot 1 uses it first. Until TP6 is answered a kit PR counts against the slot's own budget.

---

## §7 Stop conditions

One numbered question, options in words. The slot keeps working on other rows.

1. A theme, a scope or an exit criterion needs the operator.
2. A slot would have to touch a release PR, a human's PR or another worker's PR, or copy commits between PRs.
3. A force-push, or removing or archiving any ref (R11).
4. A run on a GPU host or the clean-room pool during a release pass.
5. A row that cannot be done without new Python in this repo.
6. A slot cannot be refilled within two ticks. This is the cop's stop.
7. A kit PR is red for 12 h and cannot be split, or fails two review quorums in a row.
8. A harness hook refuses writes for a reason other than a missing ticket.

**Not stops.** A missing ticket: the row rides the train epic's number, and the cop settles it. A slot above its limit creates no branch and proposes verdicts. A red kit PR is fixed or split. The repo at its PR cap: slot 3 yields first, then slot 2. The checker and the R2 and R3 reds are LAB.

---

## §8 Falsifiers

Planted items live in a fixture or a local scratch branch that is never pushed.

| Rule | Falsifier | Anti-vacuity arm |
|---|---|---|
| R1 | Removing one required item from a copy of `<train>.yaml` lowers the score by one item. | Adding it back restores it. |
| R2, R3 | `git ls-remote --heads origin 'la-<NN>/*'`, minus open PR heads, above the slot's limit is red in the nightly line. A fixture listing with one extra name is red. | The count is at least 1. |
| R4 | A second open PR for slot 2, or a code PR from a slot whose K1 is not on `main`, is named in the cop's listing. | One open PR is not named. |
| R5 | `git diff --no-renames --diff-filter=A --name-only $(git merge-base origin/main <branch>) <branch> -- '*.py'`, minus the paths in `git ls-tree -r --name-only origin/main`, is non-empty: red. | A planted `x.py` is named. |
| R6 | `git diff --name-only origin/main...<head>` lists a path outside `docs/lookahead/` for a kit PR: red. | A planted `crates/` edit is named. |
| R7 | A slot PR opened or pushed inside a pass window is named. After LP2: a kit PR merged mid-pass leaves the pass's commit and receipts unchanged. | The pass log names its commit and its window. |
| R8 | A heartbeat older than 2 h respawns the slot. A heartbeat whose tip sha differs from `git ls-remote origin la-<NN>/wip` for 2 h is red. | A fresh heartbeat does not respawn. |
| R9 | A question first seen in a cut report is counted. Target 0. | The cut report lists questions. |
| R10 | A code PR that adds a file under `docs/audits/` or a corpus path is named. | #4913 at baseline is named. |
| R11 | A `la-*` head that left origin without a line in the archive job's log is red. | The job's log exists. |
| R12 | A PR whose commits carry a slot's trailer and whose milestone is not the slot's train is named. | A fixture PR with trailer `la-0.72` and milestone 0.71.0 is named. |

**What would prove this spec wrong**

- Kit files land and the next train still starts late: the kit is the wrong measure.
- A docs-only PR already reaches its verdict in minutes: G10 is stale and why 4 falls.
- Each of the 142 branches is a planned PR: the nesting is by design and R3's limits are too tight.
- The 77-hour gap was an outage outside the slots: G18 says nothing about the workers.
- The cop's private state already prints I4 and the readiness count nightly: why 5 falls.

---

## §9 Report

One line per slot, each night: train, worker, kit score as items present over items required, items missing, branches against the limit, open PRs against the budget, `.py` files new to `main`, questions open, heartbeat age, working-branch lag, rows outside the slot's train.

---

## Appendix A: worker prompt (LA-0 writes it to `docs/prompts/lookahead-worker.md`)

```
You are the look-ahead worker for aprender train ${TRAIN} (slot ${SLOT}),
per docs/specifications/APR-LOOKAHEAD-002-train-kits.md.
Loop (live state wins over memory):
 1. Read docs/lookahead/slots.yaml and docs/lookahead/${TRAIN}.yaml on origin/main.
    Write a heartbeat.
 2. During a release pass, or when throttled: work on la-${NN}/wip only.
    No PR, no GPU host, no clean-room job. Never idle, never stop.
 3. Pick the missing kit item with the highest value (spec §4, §5). One item per iteration.
 4. Work on la-${NN}/wip. No new branch unless a staged place is free. No Python.
    No new ticket: the row rides the train epic's number.
 5. A kit item lands through a kit PR that changes only docs/lookahead/**.
    Code waits on a staged branch. Never a draft PR.
 6. Push la-${NN}/wip. Never force-push. Never delete or archive a ref.
 7. Ask an operator question the moment it is found: add it under questions: and tell the cop.
 8. Take no row outside train ${TRAIN}. Report to the cop in five lines or fewer. Stops: spec §7.
```

## Appendix B: not checked

- The cop's private state and each worker's session were not readable.
- G6 reads commit trailers, PR titles and PR refs. The public PR list still showed 11 at 14:39Z.
- Whether sections inside the CI driver skip work for a docs-only diff. The job bodies of the other 31 workflows.
- G13 to G15 come from `APR-071` and from the epic report on a branch.
- The cause of the 77-hour gap. Committer time stands in for push time throughout.
- Whether any guard on `main` scans `docs/lookahead/`.
- Whether the eight `la73/*` branches are look-ahead.
- CI verdict colours: the Actions page gave durations, not results.
- The final text was not re-read by a review lane. The cop's quorum has not reviewed this file.
