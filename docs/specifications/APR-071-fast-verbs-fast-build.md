# APR-071: 0.70.3 patch, then 0.71 "Fast Verbs + Fast Build"

**Spec id:** `APR-071` · **Version:** 1.1 (2026-10-08; v1.1 corrects H9's authorship test: agent items and operator-typed items share one login) · **Owner:** the aprender release cop · **Trains:** 0.70.3, 0.71.0, 0.71.1
**Drop at:** `docs/specifications/APR-071-fast-verbs-fast-build.md`, as a docs-only PR.
**Scope it replaces, once §3 P3 is answered:** `docs/specifications/EPIC-0.71-dont-leave-behind-plan.md` and the 0.71 section of `docs/reports/0.7x-epic-reports.md` (branch `docs/0.7x-epic-reports`). Their ticket tables stay as inventory.
**Review:** three fresh-context lanes (evidence, authority, execution) read the first draft against `main` at `ad8fafc79`; one ran the repo's own guards in a scratch export. Their findings rewrote the 0.70.3 route. A fourth lane read the rewritten route and stop rules. All admitted findings are applied. The cop's own quorum has not reviewed this file.

**Marks:** `[V]` verified from public git or the public issue pages at the sha or time named · `[C]` reported by the cop, a worker, a review lane or an earlier advisor document, not re-checked by the author · `[O]` operator statement · `[P]` proposal, not in force · `[A]` assumption · `[U]` unverified, measured in row A0.

---

## ELI5

| Question | Answer |
|---|---|
| What ships first? | **0.70.3**: one fix. Tool calls work when the client streams (#4918). |
| How does it ship? | The way 0.70.2 shipped: a release branch and a recorded CRUX-smoke scope. No gate changes. |
| What is 0.71? | Two epics. **Fast Verbs** (E2 #3598): `apr` gets measurably fast. **Fast Build** (E1 #3998): the release script runs the release alone. |
| Why is build work in 0.71? | 0.70.2 shipped by hand. The script path has seven defects that stop an unattended release at this sha (§1, D1 to D7). |
| What does 0.71.0 wait for? | Five build criteria (B1 to B5) and the verbs floor. The other verbs criteria roll once, to 0.71.1, if §3 P4 is answered ROLL. |
| What stops the line? | §8. |

**How it works**
- Clean-room green on exactly the tag commit comes before any upload.
- 0.70.3 does not wait for 0.71, and 0.71 does not wait for 0.70.3.
- At a cut, only open issues labelled `must-carry` block. That label marks work that must merge before the bump, and nothing else.
- Speed work merges only after tracing and a baseline exist. A speed row closes on a receipt.

| | Yes / No |
|---|---|
| Is any gate waived by this spec? | No |
| Does this spec change a gate by itself? | No. It proposes four gate changes for 0.71 (P5, P6, P7, B4). Each waits for the operator. |
| Can a nightly model-ladder row stop a release from 0.71 on? | No (standing policy) |
| Can a speed PR merge before tracing (V6) and a baseline? | No |

## Purpose and terms posture

- The goal is an open-source Rust ML framework. Models are not sold and are not built to compete with any provider.
- No hosted-model output is ever a training target.
- Claude Code is used only by its paying account owner: no shared credentials, no free accounts.
- Public wording never positions a model as a replacement for a commercial service.
- llama.cpp numbers in this spec are engineering parity measurements against an open-source reference at a pinned commit.

---

## §0 Operating assumptions

1. The cop runs this spec. The cop is the only minter of tickets and the only writer of milestone, label and epic membership. A worker session holds one ticket.
2. Every fact in §1 is a baseline. Row A0 re-measures it before anything acts on it. Where this spec and a script disagree, the script's gate wins: quote the line.
3. A `[P]` line is not in force. It comes into force only when an operator block names it by number. Handing this file to the cop, merging its docs PR, or silence is not a yes. Rows that depend on an unanswered `[P]` do not start. Every other row starts at once.
4. Flow per row: ticket, branch, PR, `ci / gate`; one contract and one planted falsifier per feature; five-whys end in a mechanism; `pmat query` over grep; fold means move, never copy.
5. A row with no issue of its own rides its epic as a checklist row and uses the epic's number as its ticket. Nothing is minted for it (H8).
6. Model routing follows the fleet's routing spec `[C]`. Fable is not used `[O]`.
7. Unattended, a question the cop may not decide goes to the review quorum (one agy lane, one Claude lane, one apr lane). A §8 stop waits for the operator, and the stop names the work that continues.

**Terms.** *TIP*: the last commit of a release branch. *T-2*, *T-1*: the release script's stages before the bump and before the tag. *Freeze*: the milestone is frozen, before the bump. *Live*: the last crate is on crates.io and the GitHub release is public. *GPU hosts*: `lambda` (x86, RTX 4090) and `gx10` (ARM, GB10). *Rules a, b, c*: the three build rules of C332, quoted in B1 to B3. *OBS-09*: the per-layer serve tracing row of APR-OBS-001. *EXT-001*: the dogfood model lifecycle spec. *X-ids*: the exit-criteria rows of the epic report (G15).

---

## §1 Ground truth (baseline 2026-10-08, 11:35Z to 12:45Z; never quote as current)

| # | Fact | Mark | Source |
|---|---|---|---|
| G1 | `main` is `ad8fafc79`, committed 2026-10-08 09:29Z. Root version `0.70.2`. | [V] | `git rev-parse origin/main`; `Cargo.toml:160` |
| G2 | Tag `v0.70.2` names `89e261cda1`, on `release/0.70.2`. It is not an ancestor of `main`; it was merged back as `87021c996`. | [V] | `git cat-file -p v0.70.2`; `git merge-base --is-ancestor` |
| G3 | The published paths on `main` equal `v0.70.2`: `git diff --stat v0.70.2 origin/main -- crates src Cargo.toml Cargo.lock` prints nothing. | [V] | that command |
| G4 | #4918's root cause is as the issue states. `crates/aprender-serve/src/api/openai_handlers.rs`: parser `build_tool_calling_message` :621, called only from `build_chat_response` :698; SSE builders `pregenerated_sse_response` :842 and `true_streaming_sse_response` :959 end with `FinishReason::from_generation`. `ChatDelta` (`mod_create_demo.rs:801`) has `role` and `content` only. `FinishReason` (`infer/run_report.rs`) has `Stop` and `Length` only. Every chunk stamps `created` from the wall clock (`mod_create_demo.rs:816`). | [V] | `git show origin/main:<path>` |
| G5 | `/v1/chat/completions/stream` sets `request.stream = true` and calls the same handler (`chat_completions_stream.rs:156-157`). It is inside the fix, not beside it. The Ollama `/api/chat` stream (#4182) is separate. | [V] | the file |
| G6 | With no `<function=` in the text, the non-streaming parser scans every `{…}` object for a tool call (`grammar/tool.rs`, `parse_openai`). No tag is needed there. | [V] | the file |
| G7 | The fix touches more than one file: 4 call sites of the live builder and 5 of the replayed one, all inside `crates/aprender-serve`, all with the request in scope; 37 `ChatDelta { … }` literals; `ResponseToolCall` has no `index`. | [C] | evidence lane |
| G8 | #4918: open, P0, bug, size:M (label text: K̂ 3 to 4 h), milestone 0.71.0, opened 2026-10-08 by @alfredodeza. | [V] | issue page |
| G9 | The standing release policy starts at 0.71.0 (`contracts/model-capability-ladder-v1.yaml:104`, `since: "0.71.0"`; `scripts/lib/release_policy.sh`, `rp_ge`). 0.70.3 is outside it. Per-release `emergency_scopes` entries exist for 0.69.1, 0.70.1 and 0.70.2. | [V] | the files |
| G10 | 0.70.2's notes: it shipped under such a scope, with six known failures, #4661 to #4666, on milestone 0.71.0. | [V] | `evidence/release/0.70.2/RELEASE-NOTES-0.70.2.md` |
| G11 | At a cut, any open issue labelled `must-carry` stops the tag. Every other open item of the milestone is moved by the release script: to the next release when that release's epic body cites it, else to `backlog`, one comment each. A failed or partial move stops the cut. The nightly `milestone` lane is red while any `must-carry` issue is open in the train's milestone. | [V] | `scripts/check_milestone_cut.sh`, `scripts/release/carry_milestone_items.sh`, `.github/workflows/release-gates-nightly.yml:66-73` |
| G12 | A train needs one milestone titled `<version>` and one `epic`-labelled item titled `EPIC: release train <version> …`. No open milestone `0.70.3` exists. | [V] | `scripts/release/lib_release_params.sh`; milestones page |
| G13 | Milestone `0.71.0`: 238 open, 122 closed at 11:45Z, due 2026-10-17 (a plan date). Its description names the theme "Don't Leave Behind". | [V] | milestones page |
| G14 | `docs/roadmaps/epics.yaml` on `main`: E1 "Fast Train" has train `"0.70"`, budget 10; E2 "Verbs Are Fast" train `"0.71"`, budget 7; cap 100, andon 95. | [V] | the file |
| G15 | The 0.71 exit criteria X1 to X16 are on no file on `main`. They are on branch `docs/0.7x-epic-reports` (`7c9b7ef46`). | [V] | `git ls-tree`; the branch |
| G16 | The nightly train reads producers' results and runs nothing itself. Of its 15 verdict lanes, `dogfood` and `assets` have no producer; `readiness` has a producer workflow that is not in the lane table and waits on a `dogfood-nightly` workflow that does not exist. The nightly `preflight` lane runs R2 and R6 only. No workflow runs `scripts/release/autopilot.sh`. | [V] | `scripts/release/nightly_train.sh`; `readiness-nightly.yml`; `release-gates-nightly.yml:13-17` |
| G17 | The GATE list (`contracts/release-ready-v1.yaml`) holds 69 gates (13 merge, 20 tag, 36 publish) and 50 child rows that are not counted. Its checker is LAB: no gate calls it. | [V] | the contract |
| G18 | The newest measured release wall time is 0.69.1's: freeze to release 417.4 min, of which the cascade is 41.4. E1's bar is 240. | [V] | `evidence/release/wall-time/0.69.1.json`; issue #3998 |
| G19 | The 0.70.2 ledger records `"DEEP GO": false` and `dogfood_attempts: 0` on a published release. | [V] | `docs/build-ledger/2026-10-08/89e261cda-lambda-vector-train.json` |
| G20 | Merges to `main`: 51 from 2026-10-01T00:00Z to 2026-10-08T00:00Z; 22 in the 7 days before. | [V] | `git log origin/main --first-parent --since … --until … --oneline \| wc -l` |
| G21 | Remote branches: 57 `la-71/*`, 58 `build-kaizen/*`, 1,421 in all. | [V] | `git branch -r` |
| G22 | Operator rulings in force: GATE or LAB (C324). Rules a, b, c are 0.71 exit criteria; d and e are scorecard only (C332). The gate list holds at most 20 entries, 10 merge, 5 tag, 5 publish; seven never move (C333). E1 moves to train 0.71 with done-when = a, b, c, the gate list at 20 or fewer, freeze to publish in 4 h or less; budget 10 (C334). | [C] | the cop's rulings register. A0 lands the four texts verbatim in one file on `main`. |
| G23 | 0.71 readiness: 0 of 10 top rows ready on `main`. V1 has no GPU baseline. V2 has no ticket and its old producer is Python. V3 has no ticket and no threshold. V7's pass condition is unwritten. | [C] | `docs/lookahead/0.71-dor.md` on `la-71/dor-071`; the epic report |

**Seven defects that stop an unattended release from `main` at this sha.** Each was read at the line named; D2 and D5 were also reproduced by running `scripts/check_model_ladder.sh` in a scratch export `[C]`.

| # | Defect | Where |
|---|---|---|
| D1 | The script tags a commit on `main`. Preflight R4 then requires `origin/release/<version>`, which no script creates. The preflight runs after the tag is pushed, and the cascade runs it again before every upload pass. | `autopilot.sh:108,338,436-451`; `check_publish_preflight.sh:20-23,510-517`; `cascade-publish.sh:636` |
| D2 | Under the policy, the T-1 dogfood lane runs the ladder gate with no CRUX receipt directory, in parallel with the models lane that writes those receipts. It reads red. | `autopilot.sh:175-177,215-220`; `Cargo.toml:627`; `check_model_ladder.sh:1344,1384` |
| D3 | Under the policy the bump adds `evidence/crux/<V>/prompt-certification*.json`. The coverage gate's version-only rule does not admit those paths, so the parent's coverage receipt never covers the release commit. | `tag_coverage_gate.sh:52,94-103`; `prepare_bump.sh:86-89` |
| D4 | The cascade's own preflight runs with no CRUX receipt directory. Under the policy R7 reads red after the tag is public. | `cascade-publish.sh:636`; `autopilot.sh:446-448` |
| D5 | `prepare_bump.sh --ship` needs a T-2 GO for `main`'s head. T-2 runs the same ladder gate, which is red there. T-2 has no GO in its history. | `prepare_bump.sh:190-191`; `t2_preflight.sh`; commit `e6f65c0ab` |
| D6 | The freeze is done by hand before the bump script starts. | `prepare_bump.sh:13` |
| D7 | The bump commit's trailer names a model the fleet does not use. | `prepare_bump.sh:247` |

**What these facts change**

| # | Finding | Consequence |
|---|---|---|
| F1 | G3: `main` is 0.70.2 plus release tooling. | 0.70.3 can branch from `main` today and stay a one-fix patch. |
| F2 | G9, D1 to D5: the script path cannot ship 0.70.3 this week. | 0.70.3 uses the route 0.70.2 used. It must not wait for Fast Build. |
| F3 | D1 to D7, G16: rule b cannot pass, and nothing nightly would have shown it. | Fast Build's first rows are these seven and a rehearsal that runs the script itself. |
| F4 | G11: three green nights can only start after the last `must-carry` issue closes. | The 0.71.0 cut is no earlier than that close plus three nights. Keep the `must-carry` set small. |

---

## §2 Hard rules

| # | Rule |
|---|---|
| H1 | No upload without `clean-room (aprender)` green on exactly the tag commit, from the first clean-room run on that commit that reached a test verdict. A re-run attempt of a run never counts. |
| H2 | 0.70.3 changes one thing. Its published-path diff against `v0.70.2` is the #4918 fix's files plus the version bump. |
| H3 | No patching a pass. A red verdict stops it. A restart is a new pass on a new commit that holds a merged fix naming the red's mechanism. A lane's verdict is never re-run on the same commit. Different triggers on one commit (PR head, tag push, rc cut, cascade pass) are different lanes. A lane lost to infrastructure before it reached a verdict (`not_measured`) may run once more on that commit `[A]`, with both runs logged. |
| H4 | `must-carry` marks only work that must merge before a train's bump. A criterion measured by the rehearsal nights or by the release pass itself (B1, B2, B5) is a checklist row of the release epic, never a `must-carry` issue. |
| H5 | Instrument first. A Fast Verbs performance PR stays draft until V6's merge commit is an ancestor of its head and its body names a baseline receipt on `main` for the cell it claims to move. A speed row closes on a before/after receipt from the same host, model file and llama.cpp pin. |
| H6 | A GATE can stop a merge or a release and is a row in the GATE list. A LAB check cannot. No agent moves, merges or retires a gate without the operator's yes for that table (B4). |
| H7 | Never: `--allow-dirty`; a yank; a force-push; a tag delete or re-tag; a registry token in GitHub secrets; a workflow that runs `cargo publish`; a public GitHub release without every asset; a receipt for a review or run that did not happen; new Python in repo files or automation; provisioning outside `forjar apply` or make targets; ad-hoc SSH by an agent (the release script's own host steps are not ad-hoc). |
| H8 | No new issue while the repo is above its andon (G14), except a train's one release epic and the issues the models nightly opens for red rows. Rows are reused, or ride an epic (§0.5). |
| H9 | An agent closes a human-authored issue only as §3 names it. An issue is agent-minted only if the cop's mint log holds its number. Everything else is human-authored, including what the operator typed under the same login; unknown counts as human (`APR-TRIAGE-001` §0.4). Milestone, label and epic membership of any issue are the cop's. |
| H10 | A rehearsal makes no write outside its own state directory: no `git tag`, no `git push` of any ref, no `gh release create` or `edit`, no workflow dispatch on a tag, no `gh issue` or `gh pr` write, no milestone change, no `cargo publish`. `carry_milestone_items.sh` runs with `--dry-run` only. |
| H11 | Look-ahead never stops. Its workers yield runners, queue slots and GPUs to the current train. `[O]` |

---

## §3 Operator statements and open proposals

**Stated by the operator, 2026-10-08 `[O]`:** "One is improving CI/CD as 0.70.2 didn't do it, it only released a few" · "Two is prioritizing Alfredo's tool calling ticket, this will be 0.70.3" (#4918) · "merging the "fast verbs" and "fast build" tickets into 0.71 release".

**Read from that, by the author `[A]`:** 0.70.3 carries #4918 and nothing else (H2). Fast Verbs is E2 #3598. Fast Build is E1 #3998, as C334 placed it.

**Open proposals.** Each is a question with its options in words. The first option is the author's recommendation.

| # | Question | Options | Why the first |
|---|---|---|---|
| P1 | 0.70.3 cannot pass the full model matrix (G10's six rows are red). How does it ship? | **SCOPE**: answering SCOPE is the operator saying "0.70.3 ships on CRUX smoke on lambda and gx10 GPU. Everything bigger is nightly." The cop records it as the **first** item of `emergency_scopes` (the judge takes the first `crux-smoke` entry): name `crux-smoke`, release `"0.70.3"`, the date of the answer, that quote, hosts `[lambda, gx10]`, thinking `["off", "on"]`. #4661 to #4666 are the only known failures. `since` stays 0.71.0. The scope is for 0.70.3 only. · **POLICY**: move `since` to 0.70.3; this switches off the readiness grade for every later 0.70.x and needs D1 to D5 fixed first. · **FIX**: 0.70.3 waits for the six. | No gate changes. The route shipped twice in a week. |
| P2 | Who closes #4918, and when? | **RECEIPT**: the closing keyword of the PR that commits row A8's receipt, measured on 0.70.3 installed from crates.io. · **MERGE**: the keyword of the merge-back PR. · **HUMAN**: a person closes it. | It closes on evidence from the shipped binary. |
| P3 | Which earlier 0.71 exit criteria leave 0.71.0, and where do they go? | **AS TABLED** below. · **KEEP ALL**. | 0.71.0 keeps 12 criteria. Each row that leaves names a train. |
| P4 | Does 0.71.0 wait for all seven verbs criteria? | **ROLL**: 0.71.0 cuts on B1 to B5 plus the verbs floor (V6 on `main`; V1 and V2 measured on both GPU hosts). V1, V3, V4, V5, V7 move to train 0.71.1, once. One not green at the 0.71.1 cut is a stop. · **HOLD**: 0.71.0 waits for all seven. | F4, and V1 has no baseline yet. |
| P5 | D1: how does a cut from `main` satisfy provenance? This touches C333's "provenance" entry. | **RULE**: R4 accepts HEAD as an ancestor of `origin/main` or of `origin/release/<version>`; a commit on neither still refuses. · **REF**: the script pushes `release/<version>` at the release commit; R4 then never refuses on that path. · **BRANCH**: releases stay on release branches; rule b is re-read for them. | It keeps a check that can fail. The repo's own nightly calls a made-up release ref "theater" (`release-gates-nightly.yml:15`). |
| P6 | Rule c needs the release driver to survive a GitHub read that fails and then answers (X15). | **YES**: a read that did not answer (non-zero exit, HTTP 5xx, timeout, empty body) is read at most 3 times in all: rule c's planted read fails twice and then answers. A read that answered is never re-read: `failure`, `cancelled`, `timed_out`, a missing job and an empty list are answers. No write is retried. Each re-read is one status line. · **NO**. | Rule c cannot pass without it. |
| P7 | D2, D3, D4: the standing policy's readers are half wired. | **WIRE**: the T-1 dogfood ladder gate runs after the models lane and reads its receipts; the cascade's preflight reads the same receipts; the coverage version-only rule admits the added `evidence/crux/<V>/prompt-certification*.json`. One PR each, two non-author reviews. · **HOLD**. | It finishes the gate change already ruled for the policy (C332). It changes what three checks read, so it is asked. |

**P3's table.** X-ids are the epic report's (G15).

| Row | What it says | Ruled | Proposed |
|---|---|---|---|
| X1 | Mutation and proof jobs leave the PR path; the GATE list prints in one place | C324 | Delivered by #4912 `[V]`. Its targets (PR CI median 30 min, queue pass 90%) stay scorecard, as C332 says. |
| X2 | The six known failures #4661 to #4666 ship in 0.71 | C322 | Nightly rows with tickets under the standing policy, which is the later ruling. Parent E3 #3994; cited in each next release epic so the carry keeps them on a train. |
| X3 | No end-of-train car; a release is a tag on `main` | 09-28 | Kept, as the shape of B2. |
| X5 | EXT-001 rows #4382, #4386, #4393, #4401 | 09-25 | Train 0.72.0, with the publish criteria they feed. |
| X6 | Local gate in 10 min, pre-push receipt, tiered CI | 09-28 | Train 0.71.1, under E1. |
| X7 | BIN-001: one parent issue with checklist rows | 09-28 | Train 0.72.0. |
| X4, X16 | Census check; every untied P0 proven | pending | Stay pending. Not exit criteria until ruled. |

---

## §4 Rows, in order of expected value

### Train A: 0.70.3 (needs P1 and P2)

The route is the one that shipped 0.70.2, restated. It runs on a release branch, so `main` keeps moving and no merge is held.

| Row | Work | Done when |
|---|---|---|
| A0 | First reads, from commands: re-measure G1, G3, G9, G11, G12, D1 to D5. List open `must-carry` issues in 0.71.0. List what #4869, #4910 and #4912 changed in each lane of A5 and A6 (`git diff --stat 89e261cda origin/main -- scripts .github/workflows`). Confirm a `backlog` milestone exists. Confirm `gh workflow run ci.yml --ref release/0.70.3` would yield `ci / gate` and `workspace-test` on a tip sha. | One report with each delta. |
| A1 | Milestone `0.70.3`. One epic, label `epic`, in that milestone: `EPIC: release train 0.70.3 — streaming tool calls`. Its body cites #4918. #4918 itself stays where it is until it closes; then its milestone is set to 0.70.3. | `release_milestone_number` and `release_epic_number` each resolve exactly one. |
| A2 | Branch `release/0.70.3` from the newest commit of `main` for which G3's diff is empty. Open the release PR `release/0.70.3 → main`. It carries no milestone until it merges. `main` is never merged into the release branch before the tag. | The branch exists; H2's check is green on it. |
| A3 | The fix, on a topic branch cut from `release/0.70.3`, reaching it by fast-forward. Contract `contracts/serve-stream-tool-calls-v1.yaml` (name proposed). Tests T1 to T12. CI and the review quorum judge it on the release PR. No commit message on the branch, and no title or body of the release PR, carries a closing keyword for #4918: write `Refs #4918`. | T1 to T12 green; each named mutant red; T10's receipt in the PR body. |
| A4 | On the branch, after A3: the P1 scope entry; the CRUX prompt certification, as byte copies of `evidence/crux/0.70.2/prompt-certification.json` and `prompt-certification-inventory.json` into `evidence/crux/0.70.3/`, only when the prompt set, the scope judge and the certified GGUF sha256s are unchanged at the TIP (no script carries on this route; say what was compared); `evidence/perf041/lambda/marker.json` re-measured if it would be older than its gate's window when lane 7 runs; `evidence/release/0.70.3/RELEASE-NOTES-0.70.3.md` with the known-failures table; `bump-version.sh 0.70.3` with `--check` green; CHANGELOG. The last commit is the TIP. Nothing lands after it. | Cheap rows (lint, self-tests) green on the TIP before any long lane starts. |
| A5 | Lanes on the TIP, each started when its input exists: (1) `ci / gate` and `workspace-test` on the tip sha, from the release PR, or from a dispatch of `ci.yml` on the branch if the PR shows a conflict; (2) `coverage-nightly` dispatched on the TIP, receipt at or above the floor; (3) `apr` built with CUDA from the TIP on both GPU hosts, CRUX smoke on both, receipts outside the tree, and `check_model_ladder.sh --version 0.70.3` printing the scope satisfied for the TIP; (4) the T-1 deep commands; (5) `rc_publish_gate.sh --verify <clean checkout of the TIP>`; (6) `check_milestone_cut.sh` must-carry, carry, strict, with the milestone holding exactly the release epic; (7) `dogfood.sh --phase pre-publish`, started only after 1, 2 and 3 are green, GO with no deferred row. Every shell that runs `dogfood.sh`, `check_publish_preflight.sh` or `cascade-drain.sh` exports `MODEL_LADDER_CRUX_DIR=<absolute receipt directory>` and `CRUX_CERT=<repo>/evidence/crux/0.70.3/prompt-certification.json`. | Every lane green and logged for the TIP. The preflight has run once with those two set, before the tag is pushed. |
| A6 | Tag and upload, only when A5 is green: annotated tag at the TIP; draft release with notes; asset build; clean-room and GPU clean-room green on the tag commit, that run's id written to the pass log before the first drain; assets check; coverage gate; preflight exits 0 (R1 to R7 hold, R7 under the scope; R8 and the model matrix print as evidence, and their non-zero result for the six known rows is expected); release made public; `cascade-drain.sh --target 0.70.3` from the checkout that holds the TIP's dogfood receipt; `cascade-publish.sh --check` with 0 behind. The tag push also starts a CI run on the TIP: it is its own lane, and a red there is reported, not treated as a red of this pass. | The check reports 0 crates behind; the release is public with every asset. |
| A7 | The release candidate. `rc-cut.yml` fires by itself on each green CI of the release PR's head and dispatches the binary build `[V]`. The cop neither disables nor cancels it. Lane 3's work on the ARM GPU host and A6's clean-room dispatch start after the TIP candidate's binary run has ended; the wait is a line in the step table. Only a candidate cut at the TIP is offered to the reporter on #4918. Do not wait for the answer. | Link posted, or the reason no candidate was cut. |
| A8 | After live: on a CUDA host, through a runner job or make target, install 0.70.3 from crates.io and re-run T10. Merge back to `main` first. Then one PR to `main` commits the receipt under `evidence/`; under P2 RECEIPT that PR alone carries `Closes #4918`. Close the release epic and milestone 0.70.3; set #4918's milestone to 0.70.3. Print the scorecard (§10). Each hand step becomes an E1 checklist row. | Receipt on `main`; #4918 handled as P2 says. |

**Tests for A3.** The detector runs only when the request declares `tools` and `tool_choice` is not `"none"`. Its markers are `<tool_call>` and `<function=`, as the issue's fix shape says. A bare `{` is not a marker.

| Test | Asserts | Mutant that must go red |
|---|---|---|
| T1 | The issue's exact delta sequence, markers split across tokens, through the live builder: no `content` delta carries any byte of the call; exactly one `delta.tool_calls[0]` with `index` 0, an `id`, `type` `function`, name `bash`, arguments `{"command":"ls -la ~/tmp/"}`; terminal `finish_reason` is `tool_calls`. | The detector passes text through. |
| T2 | The same through the replayed builder. | The detector is wired to one builder only. |
| T3 | Split invariance: for any partition of one generation's text into deltas, where every call starts at a marker, the emitted calls and the finish reason equal `build_tool_calling_message` on the captured text. | Parsing runs per delta. |
| T4 | Tools declared, plain-text answer: content still streams delta by delta and ends `stop`. | All content is held to the end. |
| T5 | No `tools`, or `tool_choice: "none"`: with `created` masked, the SSE output for a fixed token sequence equals a golden file generated from a checkout of `v0.70.2` by a named command. | The detector runs unconditionally. |
| T6 | Captured text that parses to no call (an undeclared tool, a mere mention of a marker) is flushed as `content`; the concatenated content equals the whole text. | Captured text is dropped. |
| T7 | The detector's own buffer holds at most `len(longest marker) - 1` bytes while no marker is complete, and releases a tail that stops being a marker prefix in the same delta. | The hold-back never releases. |
| T8 | A call whose last bytes arrive through the end-of-stream flush (`tail_deltas`) is still emitted as a call. | The flush path writes straight to content. |
| T9 | `POST /v1/chat/completions/stream` returns the same `delta.tool_calls` (G5). | The route bypasses the detector. |
| T10 | Live: the issue's request with `"stream": true` against `Qwen3.5-4B-Q4_K_M.gguf` on a CUDA build returns `delta.tool_calls` and `finish_reason: "tool_calls"`. | Not applicable: this is the measurement. |
| T11 | `ChatDelta.tool_calls` is omitted from the JSON when empty. | The field always serializes. |
| T12 | A bare JSON call with no marker still streams as content (D-c). | `{` is treated as a marker. |

Decisions fixed here so the worker does not ask:
- **D-a.** Text before a marker has already streamed as content and stays. The non-streaming path returns empty content beside tool calls (`openai_handlers.rs:678`). The contract states the difference. Parity is asserted on calls and finish reason.
- **D-b.** Arguments arrive in one `delta.tool_calls` chunk at the end of the stream. Incremental arguments are a later E2 row.
- **D-c.** Bare JSON calls with no marker (G6) are a known difference between the two paths. T12 pins it. Closing it is a later E2 row, as is the Ollama stream (#4182).
- **D-d.** The terminal chunk needs a third finish reason. Add the variant where `FinishReason` lives; do not pass a string.

### Train B: 0.71.0 Fast Build (E1 #3998)

| # | Exit criterion | Source | Rows that deliver it |
|---|---|---|---|
| B1 | **Rehearsal (rule a).** "Every night on main's head, the same script and the same policy file as release day run everything except the tag push and the uploads. Green means exit 0, every lane measured on that night's commit, no rerun. The 3 nights before release day are green; a red night resets the count." | C332 | One scheduled job runs the release script's own steps, `deep` to `dryrun`, on `main`'s head in a scratch clone, under H10. `nightly_greens.sh` counts that job. The read-only nightly train stays LAB. |
| B2 | **The script releases (rule b).** "One unattended pass of scripts/release/autopilot.sh from the bump to live: 0 steps typed outside it, 0 emergency scopes, 0 commits on the release branch after the bump, 0 gate-script edits on release day." | C332 | D1 (after P5). D2, D3, D4 (after P7). D6, D7. D5 needs a design first: under the policy T-2 has no CRUX input for `main`'s head. The worker brings one sentence and one falsifier, and the change is asked like P7. #4896 and #4897, the two hand steps 0.70.2 recorded. Every lane except the cascade as a runner job; the upload stays on the train host. An explanation or fix for G19. |
| B3 | **Receipts (rule c).** "Two planted tests pass inside the rehearsal: a GitHub read that fails twice and then answers costs under 2 minutes and no lane rerun; a notes-only commit is ready to tag in under 10 minutes on saved results." | C332 | P6. One code identity over the published paths, with results keyed by it (branches `build-kaizen/4803-b3-bump-reuse`, `build-kaizen/bld002-r2`). |
| B4 | **The gate list holds 20 entries or fewer:** 10 merge, 5 tag, 5 publish. Seven never move: clean-room green on exactly the tag commit; workspace tests green on the commit; CRUX smoke on both GPU hosts under the policy; supply chain; provenance; binaries present before the release is public; nothing secret or unintended in the published crates. | C333 | The cop prepares one table of all 69 entries: keep, nightly, or merge into another. It moves nothing. The table goes to the operator once: ADOPT, AMEND or HOLD. `--gates` prints the count by stage. No gate becomes a child row to lower the count. |
| B5 | **Freeze to publish in 240 minutes or less.** Start: the freeze. End: the later of the GitHub release going public and the last crate's upload. | C334; G18 | The wall-time record for the version, computed without new Python. A missing anchor is `not_measured`. |

B1, B2 and B5 are measured by the nights and by the pass. They are checklist rows of the 0.71.0 release epic (H4). The rehearsal's first night must be **red**, naming every one of D1 to D7 that is still open. A rehearsal that is green while one of them is open is not the release path.

### Train B: 0.71.0 Fast Verbs (E2 #3598)

Reference for every speed number: llama.cpp at pin `d1d3c3396`, same host, same GGUF, Qwen3.5-4B Q4_K_M, on both GPU hosts.

| # | Exit criterion | Source | State `[C]` | Order |
|---|---|---|---|---|
| V6 | Per-layer serve tracing (OBS-09) on `main` before the first performance PR. | C332 | A checklist row of #4487. The tracer is on a branch and conflicts with a file A3 changes. | 1, after 0.70.3 merges back |
| V2 | `apr serve` keeps the model loaded across requests, and its receipt reports `load_ms` and TTFT beside pp512 and tg128. | C332 | No ticket. The producer must not be Python. | 2 |
| V1 | TTFT ratio = reference TTFT ÷ `apr` TTFT. The lower bound of its 95% interval is at least 0.5. The receipt names n and the interval method. | C332 | No baseline. | 3: baseline first, then a Pareto of the gap by phase (load, transfer, prefill, first token), then levers in order of measured milliseconds |
| V5 | Kernel registry live: 0 unregistered dispatches (#4539). | C332 | The first branch carries a Python oracle, which H7 refuses. | parallel |
| V4 | `--json-schema` constrained decoding (#3568). | C332 | Its falsifier ids are on `main`. | parallel |
| V3 | Batched decode with a parity receipt at the serving shape. | C332 | No ticket, no threshold. | after V1's baseline |
| V7 | CRUX gates against the reference engines defined (EXT-001). | C332 | Pass condition unwritten. | last |

For V3 and V7 the worker proposes one sentence and one falsifier. The quorum reviews it and the operator adopts it in one answer; the contract records it before any code. The worker who writes a bar does not close the row against it.

Candidate V1 levers, all existing tickets, ranked only after the baseline: #4450, #4274, #4483, #4313, #4442 (prefill); #4486, #4508, #3725 (decode, which moves throughput more than TTFT).

### Everything else

E3 #3994, E4 #4381 and the carried ontology and proof rows keep their owners and priorities. None carries `must-carry` in 0.71.0. #4418 blocks two 0.72 criteria `[C]`: the 0.72 look-ahead worker owns it now.

---

## §5 Phases

| Phase | What | Needs |
|---|---|---|
| 0 | A0. Land this spec and G22's four ruling texts as docs-only PRs. | nothing |
| 1 | Train A: A1 to A8. | P1, P2 |
| 2 | Fast Build, in parallel with phase 1: D6, D7, D5's design, the rehearsal job (B1), B4's table, B5's record. Then D1 to D5 as their answers arrive. | P5, P6, P7 for the rows that name them |
| 3 | Fast Verbs: the V1 and V2 measurement rows now; V6 as soon as 0.70.3 merges back. | nothing |
| 4 | Scope: `epics.yaml` (E1's train and name, #4790 folded into E1 and closed by that PR's keyword if agent-minted); the milestone description; after A0 prints every issue that would lose `must-carry` (number, title, the ruling that placed it), the label pass. An issue a ruling labelled keeps the label until P3 answers for its row. The cop moves 0.71.0's other open items to their trains or `backlog` in the background, one comment each. The nightly count is `check_milestone_cut.sh 0.71.0 --must-carry --json`, field `to_carry`. | P3, P4 |
| 5 | Three green rehearsal nights. They start after the last `must-carry` issue closes (F4). | phases 2 to 4 |
| 6 | 0.71.0: freeze, bump, one pass, `wait` to `close`. Before the freeze: milestone `0.71.1` is the lowest open version above 0.71.0, and its release epic's body cites every rolling V issue, #3598 and #3998. | P4 |
| 7 | 0.71.1 by the same path. | — |

---

## §6 Definitions of done

- **0.70.3:** A6 and A8 done; `git diff --name-only v0.70.2 v0.70.3 -- crates src Cargo.toml Cargo.lock` lists only the fix's files and the bump's.
- **0.71.0:** a receipt for each of B1 to B5; under ROLL, V6 on `main` and the V1 and V2 measurements from both GPU hosts; under HOLD, a receipt for each of V1 to V7; the milestone holds nothing open but its release epic.
- **The epic pair:** E1 closes when B1 to B5 hold at the 0.71.0 cut and again at the 0.71.1 cut `[A]`. E2 closes at 7 of 7.
- A criterion is never waived. It moves to a later train only by an operator answer. P3 and P4 are those questions.

---

## §7 Budget and routing

| Item | K̂ | K | Andon | Basis |
|---|---|---|---|---|
| A3 | `[U]` | 1.65 × K̂ | 0.8 K | The label says 3 to 4 h (G8). G4, G5 and G7 say more. The cop re-sizes at ticketing. Multipliers are the fleet's convention `[C]`. |
| A5, A6 | none | none | none | 0.70.2 took about 9 h by this route `[C]`; 0.69.1 took 417 min freeze to release (G18). This pass prints its own step table. |
| 0.71 rows | Σ of size labels over the `must-carry` set | 1.65 × K̂ | 0.8 K | Computed in A0 once P3 and P4 are answered. |

- **Capacity check.** The plan cut is the milestone's due date (G13). Demand is the `must-carry` set, in PRs. Supply is G20's merge rate times the days left, less three nights (F4). If demand exceeds supply the cop reports the overflow in §4's order. Under ROLL the answer is already given. Under HOLD it is §8 stop 10.
- **Pools.** From the TIP until the release is live, the 0.70.3 pass has first claim on both GPU hosts and the clean-room pool.
- **Andon.** At a row's andon the worker stops and writes the five-whys. The cop splits or re-plans the row.

---

## §8 Stop conditions

One numbered question, options in words. Every other lane keeps moving.

1. A waiver, a report-only mode, or marking a failure "known" beyond what P1 records.
2. A gate change other than P5, P6, P7 and B4's table as answered.
3. A red CRUX smoke cell whose fix would need stop 1, 2 or 9.
4. A spent version (a red verdict after the tag and before the first upload). The next patch number is a new train below the policy's start, so it needs its own P1 answer.
5. A cascade with one crate or more uploaded that makes no progress in two drains.
6. A third restart of one pass `[A]`: the count follows the fleet's rule that two failed rounds escalate.
7. The tag or a release ref refused by a ruleset; anything touching secrets, rulesets or credentials.
8. Closing a human-authored issue other than as P2 answers, or editing its title or body.
9. Withdrawing or replacing a model a gate names, the CRUX certification list included; putting a dev tool on the gate path.
10. An exit criterion that cannot be met. The question names the later train.
11. Stopping a service.
12. A gate-change PR failing its two non-author reviews twice with a change between the rounds.

**Not stops. The cop decides, acts and reports.**
- A red verdict before the tag: H3.
- Clean-room on the tag commit: H1. A red with a test verdict is final for that version. A run that never reached a test follows H3's one more run.
- A red verdict after the tag and before the first upload: the version is spent. The tag and the draft stay. The fix lands on `main`. Then stop 4.
- A red during the cascade: resume with the same script. Never a yank. Never the next patch number while the check shows a crate behind.
- A red after live: fix forward in the next patch.

Never, with or without a question: everything in H7.

---

## §9 Falsifiers

Each rule has a check that can fail, and an arm that proves it measured something.

| Rule | Falsifier | Anti-vacuity arm |
|---|---|---|
| H1 | A green clean-room run that tested another commit is refused by the cascade (`scripts/check_cascade_clean_room_gate.sh`). | A green run on the tag commit passes. |
| H2 | `git diff --name-only v0.70.2 <TIP> -- crates src Cargo.toml Cargo.lock`, minus the fix's files and the bump's, is empty. A planted one-line change in another crate makes it non-empty. | The fix's file set holds at least one path under `crates/aprender-serve/`; an empty set is `not_measured`. |
| H3 | Two passes logged against one commit is red. A restart whose commit holds no new merged fix is red. | The log names the commit of every pass. |
| H4 | An open `must-carry` issue in the train's milestone that is B1, B2 or B5, or outside the criteria's merge-before-bump work, is named by the cop's nightly listing. A planted one is named. | The listing returned at least the release epic. |
| H5 | A performance PR whose head lacks V6's merge commit, or whose body names no receipt path that exists on `main`, stays draft. Both are planted once. | A PR with both is undrafted. |
| H6, B4 | `--gates` above 20, or a stage above its share, fails B4's receipt. A table line that turns a gate into a child row is refused. | The count is above 0 and the stage counts sum to it. |
| H8, H9 | The open-issue count before and after adoption differs only by release epics and nightly red-row issues. No human-authored issue is closed except as P2 answers. | The read returned a count. |
| H10, B1 | A stub `gh` and `git` record every call of a rehearsal; any write is red. The first night is red naming each open defect of D1 to D7. | A night with every step measured on that night's commit exits 0. |
| A3 | T1 to T12, each with its mutant. | T5's golden file carries the `v0.70.2` sha in its header and is not regenerated from the fix. |
| B2 | The pass log shows exactly one start of the script and every step marker from the release commit to done; `git rev-list <release commit>..v<V>` is empty; no `emergency_scopes` entry names the version. | The log exists for that version. |
| B3 | The two planted tests of rule c. | Each plant is seen to fire in the log. |
| B5 | Above 240 minutes is red. | A missing anchor is `not_measured`, never a pass. |
| V1 | The interval's lower bound is below 0.5: red. A delay planted in one phase moves that phase's field and only that field. | The receipt names the host, the GGUF sha256, the pin and n. |

**What would prove this spec wrong**

- `git diff v0.70.2 origin/main -- crates src Cargo.toml Cargo.lock` is non-empty at A0: F1 is gone. A2's rule still finds the last clean commit.
- A script outside `scripts/`, `.github/` and the Makefile creates `release/<version>`: D1 falls.
- An unattended script pass ships a covered version with D2, D3 and D4 untouched: those three are wrong.
- #4869, #4910 or #4912 changed a lane of A5 or A6 so that it no longer passes under a per-release scope: the 0.70.3 route is not the 0.70.2 route, and A0 must say which lane.
- The fix lands inside the label's 4 hours with all twelve tests: G7's sizing concern was wrong.

---

## §10 Report schema

Every 60 minutes `[A]` and at every state change.

- Line 1: train, mode, messages held or dropped.
- Line 2: the live estimate in UTC, the step it waits on, the open question or "none".
- One table: lane, state, started, minutes left, evidence.
- Odds for live by three named dates.
- At the end of a train, the scorecard, measured: minutes from freeze to live; release commits used; restarts; steps typed by hand; minutes waiting on the operator; entries in the GATE list by stage; green nights in a row; what moved to the next train.

---

## Appendix: not checked

- GitHub Actions runs, required-check rulesets, runner health and the private infra repository were not readable.
- Labels, `must-carry` counts and sub-issue trees: the GitHub API was closed. A0 measures them.
- The 0.70.2 route was restated from the operator's block for that release. It was not re-run at today's `main`.
- T10 needs a CUDA host. The issue's exact delta sequence was read through a page summary.
- `crates/` was read file by file, not searched as a whole.
- C-numbered rulings are quoted from the cop's and the advisor's documents. They are on no file on `main` until row A0 lands them.
