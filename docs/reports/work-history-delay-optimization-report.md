# Work-history delay analysis and optimization report

Repository: paiml/aprender · Window: 2026-07-20 → 2026-09-04 (ISO weeks W30–W36) · Written 2026-09-04 during the 0.65.0 landing, from measurements taken that day.

Every number below was measured from the GitHub API, the git history, `.pmat/jidoka.jsonl`, `docs/audits/impl-PMAT-742-receipt.md` and the paiml-implement estimate ledger. Nothing is estimated from memory. Where a figure is an inference it is labelled as one.

Method. Each delay is treated the way the repository treats a defect: a five-whys chain to the owning mechanism, then a Popperian test — the causal claim is written as a hypothesis with the observation that would refute it, and the current evidence is graded **corroborated**, **refuted**, or **untested**. Two hypotheses held earlier on 2026-09-04 were refuted by their own test and are reported as such, because a delay analysis that only records the surviving story is the "recorded but never compared" failure this repository already knows.

---

## 1. Summary — the delays, ranked by cost

| rank | delay | measured cost | share of a PR's life |
|---|---|---|---|
| 1 | **CI wall-clock per run has grown 2.5× and every PR needs more than one run** | PR `CI` run median 59 min (W36: 55 min; W31: 22 min). Runs per PR branch: median 2, p90 5, max 18. 478 PR runs for 187 branches; 226 of them cancelled by the next push | dominant |
| 2 | **The merge queue serializes and evicts** | 269 merge-group runs: 169 success, 64 failure (24 %), 35 cancelled (13 %). 80 evictions across 170 merged PRs (46 PRs evicted at least once, 24 at least twice). One PR merges per ~1 h of green fleet time | ~1 h per attempt |
| 3 | **A shared, saturated runner fleet** | 16 runners serve five active repos. Since 2026-08-07: aprender 748 run-wall-hours, rmedia 527, forjar 275, paiml-mcp-agent-toolkit 209, whisper.apr 76. PR-run runner wait p90: 388 min (W33), 291 min (W34), 77 min (W35), 0 min (W36) | bursty; up to 6 h |
| 4 | **Defects that exist only on the merged tree** | On 2026-09-04 alone: README contract count, SATD ratchet lower bound, complexity-ratchet stale row, PERF-009 guard — four full CI cycles, each ~1–1.5 h, none reproducible on the branch | 4–6 h on one PR |
| 5 | **Environment deaths read as code failures** | runner-6 failed 14 aprender jobs over 8 h 12 min (00:28–08:40Z) and evicted three merge-queue runs; a registry outage, a rustup download failure, a dep-info race and a container-cache failure on the same day | 8 h + one cycle each |
| 6 | **Gate accretion and gate self-repair** | `scripts/check_*.sh`: 14 → 89 in six weeks; `ci.yml`: 561 → 2,552 lines. Of 160 commits on main since 07-20, 76 are plain `fix`, 26 fix a guard/gate/ratchet, 11 fix dogfood, and 11 are `feat` (7 %) | structural |
| 7 | **Review rounds on guard code** | the PP-9 ledger scanner needed 5 cross-vendor review rounds and six tickets (PMAT-930..935): estimate 15 turns, actual 58 | 4× estimate |
| 8 | **Toolchain defects** | `pmat work add` minted colliding ids twice and rewrote 2,046 roadmap lines; bashrs 7.0.1 counts differ per invocation (13 vs 53 on one tree) and flag six false positives; the pre-commit hook refuses any file with pre-existing complexity debt, forcing 22 unrelated decompositions | days, spread |
| 9 | **Estimation error** | est 7 → actual 152 turns; 60 → 88; 15 → 58; 15 → 34; 13 → 24 | 1.5–20× |
| 10 | **Hardware-window dependencies** | perf-matrix rows 2, 13–21 UNMEASURED with owners; CUDA nightly PP-26 witness FAIL at c=16 every night since 08-30; gx10 lacks the SafeTensors reference the qwen-story beat B2 needs | not release-blocking; standing INVALID band |

Release cadence tells the same story from the outside: v0.55.0 → v0.60.0 shipped in 12 days (06-24 → 07-06); v0.61.0 07-26, v0.62.0 07-31, v0.63.0 08-01, v0.64.0 08-24, and 0.65.0 is landing on 09-04. The interval between releases went from about two days to about three weeks while the volume of work per week stayed flat (21–40 merged PRs per active week).

---

## 2. The last six weeks, week by week

| week | merged PRs | median time-to-merge | merge-group runs (ok/fail/cancel) | CI run median | what happened |
|---|---|---|---|---|---|
| W30 (07-20..26) | 9 | 1.0 h | 9/0/0 | 21 min | Five PRs open since early July finally merge (#2299 "Validation of top 50 HF models", 549 h; #2300, #2303, #2304, #2308 — 480–550 h each). v0.61.0 on 07-26 |
| W31 (07-27..08-02) | 40 | 1.0 h | 40/1/2 | 22 min | The fast week. v0.62.0 (07-31) and v0.63.0 (08-01). Coverage measurement found to have reported 0/0 since APR-MONO and fixed (#2333); `$?`-through-a-pipe fixes (#2336) |
| W32 (08-03..09) | 2 | 0.8 h | 2/0/0 | 20 min | Near-silent on main. Inference: the 0.63.0 dogfood audit (202 defects, 26 P0; 104 CLI / 9 MCP / 45 route rows) was being taken in this window — its fix batches land in W33 |
| W33 (08-10..16) | 38 | 5.2 h | 44/22/13 | 37 min | The dogfood fix batches (#2449: 24 fixes in one PR; #2451). First week with merge-group failures (22) and with runner waits: p90 388 min, 21 % of PR runs waited over 30 min. CI median +70 % |
| W34 (08-17..23) | 26 | 8.6 h | 28/23/12 | 50 min | 32 new guard scripts in one week. The nine-PR consolidation batch #2537 (623 files). Pre-release sweep 08-19 (CLI 165 / MCP 18 / HTTP 45 rows). Merge-group failure rate 37 % |
| W35 (08-24..30) | 34 | 6.2 h | 25/9/1 | 44 min | v0.64.0 tagged 08-24 with the crates.io cascade not run and four host receipts owed. APR-PERF-GATE-001 v2.2 lands and repairs nine gates (#2705). SetFit (#2618, 85 h to merge). PP-26 nightly starts failing at c=16 (08-30) |
| W36 (08-31..09-06) | 21 so far | 9.6 h | 21/9/7 | 55 min | PP-LLAMA-001 v3 spec (#2851, 249 files). Three attempts at the 0.65.0 cut: PMAT-742 run PARTIAL(andon) at K=150; continuation STOPPED(dogfood pre-publish NO-GO); a 43-agent fan-out producing the release-gates batch #2860. 09-04: the landing described in §4 |

Two readings of this table survive the numbers. First, time-to-merge rose from 1 h to 6–10 h at the median as the CI run itself went from 22 to 55 minutes and as each PR started needing two to five runs. Second, the weeks with the most gate work (W33, W34) are the weeks with the highest merge-group failure rates (28 % and 37 %): the gates were catching real things, and the things they caught were often other gates.

---

## 3. The delays, each with five whys and a falsification test

### 3.1 CI wall-clock growth and multiple runs per PR

**Observation.** A `CI` run on a pull request took a median 22 min in W31 and 55 min in W36; the merge-group run went from 21 to 55 min. 187 PR branches produced 478 CI runs (median 2, p90 5). The release batch branch alone produced 18 runs; the ledger-scanner branch 15.

**Five whys.**
1. Why does a PR take 6–10 h to merge? Because it needs a green PR run, then a green merge-group run, and each is now close to an hour when the fleet is free.
2. Why is a run an hour? Because `guard-runner-labels` builds `pv`, a wasm32 target, the profile crate and runs the model-tests suite in a container, and `workspace-test` runs 88,500 lib tests plus integration targets; the two are the critical path.
3. Why do those jobs exist as they are? Because between W33 and W36 the repository added 75 guard scripts and 2,000 lines of workflow in response to the 0.63.0 dogfood audit and to the discovery that its release gates had been theater (0.63.0 provenance: "release gates were themselves broken").
4. Why were guards added as separate serial steps in one host job rather than as a fast tree check? Because each guard was written at the moment its defect was found, by the same process that found it, with the acceptance criterion "goes RED on the mutation", not "runs in under a minute on the merge tree".
5. Why does a PR need more than one run? Because the guards measure the merged tree, the developer measured the branch, and the difference only shows in CI (see 3.4); and because every push cancels the run in progress (226 cancelled PR runs), so a fix pushed at minute 50 restarts the clock.

**Hypothesis H1.** CI wall-clock growth is caused by guard accretion in the host-side jobs, not by test-suite growth. **Prediction:** the `push` runs on main (which run the same tests but not the PR-only guard jobs) should have grown less than PR runs. **Evidence:** push CI median 22 → 52 min, PR CI 22 → 55 min; both grew the same. **Status: refuted as stated** — the growth is in the shared jobs (`workspace-test`, `ci / *`) as well as the guard job, so the cause is broader than the guard job alone. A revised H1′ — "growth is in the shared jobs because per-run target directories cold-build under fleet contention (workspace-test measured 94 min under saturation on 09-04 against a 42-min median)" — is **untested** and is the first thing to measure (§5, O3).

### 3.2 The merge queue serializes and evicts

**Observation.** 269 merge-group runs for 170 merged PRs: 64 failed and 35 were cancelled. 80 evictions. The queue holds one PR at a time; a failed or cancelled merge-group run costs the run's duration plus re-entry.

**Five whys.**
1. Why 1.6 merge-group runs per merged PR? Because 37 % of merge-group runs do not merge.
2. Why do they fail? Two classes: defects that exist only on the merge commit (3.4) and environment deaths (3.5). Of today's five merge-group failures, three were environment (runner-6 ×2, a rustup download) and two were merge-tree defects.
3. Why is a cancelled run so common (35)? Because the queue evicts on a 60-minute timeout that includes runner wait, and because a fresh push to a queued PR removes it.
4. Why is re-entry expensive? Because `gh run rerun` replays the stale merge base and is banned by the brief; the only honest path is close/reopen, which discards the partial run and re-enters at the back of the fleet's queue.
5. Why is the queue depth effectively one? Because every PR that touches `docs/roadmaps/roadmap.yaml` appends at the same line, so the PR behind conflicts the moment the one in front lands and must be re-pushed (measured today: #2861 relocated its six blocks to the head of the list; a real `git merge --no-commit` then showed 0 conflicts against both other branches).

**Hypothesis H2.** Roadmap appends at end-of-file are a first-order cause of re-pushes behind a merge. **Prediction:** a PR that inserts its roadmap blocks away from the append point merges behind another PR without a push. **Evidence:** #2861 (relocated) stayed MERGEABLE when #2809 landed; earlier in the same day, #2860 and #2861 had to be re-pushed with keep-both resolutions after #2857 and #2859 landed. **Status: corroborated** (n=1 relocation, n=2 conflicts; small).

### 3.3 The shared fleet

**Observation.** 16 `intel-clean-room` runners serve aprender, rmedia, forjar, paiml-mcp-agent-toolkit and whisper.apr. Runner wait for PR runs had a p90 of 388 min in W33 and 291 min in W34. On 09-04 the fleet read 16/16 busy from 04:54Z to 13:04Z; at 12:00Z the host showed one paiml-mcp-agent-toolkit `mutation-diff` job holding a runner since 08:03Z and three rmedia `feature-matrix` jobs started at 11:38Z.

**Five whys.**
1. Why did our runs queue for hours? Because the fleet was full.
2. Why full? Because five repositories dispatch to one label with no priority, and a single mutation-testing job can hold a runner for 4+ hours.
3. Why one label? Because the fleet was provisioned as one pool (paiml/infra) and no repository declares a priority or a reservation.
4. Why does aprender feel it most? Because aprender's runs are the longest (748 run-wall-hours in four weeks, the most of any repo) and its merge queue converts any wait into an eviction (3.2).
5. Why has no reservation been made? Because the operator's standing rule is that the fleet's over-subscription is intended headroom (2026-08-22), and until today nothing measured the cost to a release day.

**Hypothesis H3.** Fleet contention, not code, sets the release-day floor. **Prediction:** with other repos paused, our queued runs start within minutes. **Evidence:** the operator paused other projects at ~09:30Z; fleet occupancy fell from 16 to 9 by 13:26Z; #2861's merge-group run started 1 minute after entering the queue at 12:47Z. **Status: corroborated** (one day).

### 3.4 Defects that exist only on the merged tree

**Observation (09-04).** Four failures on #2860 were invisible on the branch and real on the merge commit: the README claimed 1,805 contracts and the merged tree had 1,807 (this PR's own two files); the SATD ratchet's lower bound fired (54 baselined, 37 measured after the sweep); the complexity ratchet found one STALE row because #2809 had reduced `batched_gemv_or_gemm` under the threshold on main; and PERF-009 fired for the first time because the README step before it had always failed first.

**Five whys.**
1. Why did each cost a full cycle? Because the failing step is 30–60 minutes into a job that has no early exit, and the fix re-runs everything.
2. Why were they invisible locally? Because every local check ran on the branch; CI runs on `merge(main, branch)`.
3. Why is the merge commit different enough to matter? Because ratchets and count-claims are properties of the whole tree, and main moves under the PR several times a day (three merges into main on 09-04 before 11:00Z).
4. Why were the guards ordered so that PERF-009 had never run? Because the guard job is a long serial list and stops at the first failure; a guard behind a chronically red one is dark (the same shape as the "beats gated only at ci.yml:317" and "benches are dark targets" findings).
5. Why is there no local "predicted merge" run? Because none was scripted; today it was improvised (`predicted-main-0.65.0`: main + all three branches) and it found the batuta transport race and the seven bashrs findings before CI did, saving two cycles.

**Hypothesis H4.** Running the merge-tree guards on a locally built merge commit before pushing removes the merge-only failure class. **Prediction:** a PR that passed the predicted-merge run does not fail `guard-runner-labels` on a tree-property guard. **Evidence:** two cycles saved today; the complexity-ratchet STALE row was still caught in CI because the prediction was built before #2809 merged. **Status: corroborated with a known gap** — the prediction must be rebuilt after every merge into main.

**A refuted hypothesis, recorded.** Early on 09-04 I attributed the README 1,805 → 1,807 change to #2857 in a commit message. The test — count contract files on `origin/main` — gave 1,805, so the two files were this PR's own. Refuted by measurement; corrected in the PR body, the receipt and the jidoka entry.

### 3.5 Environment deaths read as code failures

**Observation.** intel-clean-room-6 failed every job it received from 00:28Z to 08:40Z (14 aprender jobs, three merge-queue evictions). The visible error, `Value cannot be null. (Parameter 'ContainerId')`, was secondary. The primary line was in the pre-job hook group: `FATAL: host is not provisioned - missing: cargo`. Same day: `localhost:5000` registry unreachable (workspace-test), a rustup channel download failure (lint), and six `could not parse/generate dep info … No such file` errors while a container step and a host step shared one target directory.

**Five whys (runner-6).**
1. Why did every job die at "Set up runner"? The pre-job hook exited 1.
2. Why? Its `cargo --version` probe failed.
3. Why? The probe ran with the previous job's workspace as cwd, and that checkout held a 0-byte `rust-toolchain.toml` (mtime 09-03 16:47, a job killed mid-write); rustup refuses an empty override file.
4. Why did nothing repair it? The checkout step that would rewrite the file runs after the hook; the runner was deadlocked until a person edited the file.
5. Why did diagnosis take two hours? Because the first hypothesis (root-owned residue from pmat's container jobs, paiml-mcp-agent-toolkit#1185) was plausible, was true, and was not the cause.

**Hypothesis H5a (held first).** Root-owned files in `_work` break container setup on runner-6. **Prediction:** after `chown -R`, the next job passes "Set up runner". **Evidence:** ownership restored at 06:44Z; jobs at 07:15Z and 07:16Z failed identically. **Status: refuted.**

**Hypothesis H5b.** The pre-job hook's toolchain probe fails only when cwd is the stale aprender workspace. **Prediction:** the probe passes from the runner root and from the pmat checkout, and fails from `_work/aprender/aprender`. **Evidence:** replicated exactly that way on the host with the runner's own PATH and RUSTUP_HOME; restoring the file from git made `cargo --version` return 1.93.0 there; no further runner-6 failures after 08:40Z. **Status: corroborated.**

The general lesson is the one already in memory: a guard or hook that cannot distinguish "the code is wrong" from "the host is wrong" costs a cycle per event and, when it deadlocks a runner, costs every job that lands there.

### 3.6 Gate accretion and gate self-repair

**Observation.** 89 `scripts/check_*.sh` today against 14 six weeks ago; `ci.yml` 4.5× longer. 37 of 160 commits on main since 07-20 fix a guard, gate, ratchet or dogfood row. The jidoka log for the two 0.65.0 runs has 18 entries; 13 name a guard, a ratchet, a scanner or the release scripts as the owning module.

**Five whys.**
1. Why so many guards? Because the 0.63.0 dogfood audit found 202 defects and 26 P0s, and the repository's answer to every defect class is a falsifiable guard with a case table.
2. Why do the guards themselves need fixing? Because the first version of a guard is usually a regex or a whitelist written against the instance that was found (PP-9's L2 scanner accepted exactly one row shape; PERF-009 matched `date +%s` but not `date -u +%s`).
3. Why does a whitelist survive review? Because the review checks that the named mutation goes RED, and a whitelist does go RED on its own instance; the escapes are in the complement (the "blacklist clauses fail open on their complement" finding).
4. Why are escapes found so late? Because the guard is exercised only when its step is reached in CI, and steps behind a red step are dark.
5. Why does each escape cost a review round? Because the cross-vendor review is the only universe-widening instrument in use; nothing generates the complement cases mechanically.

**Hypothesis H6.** Guards written from a threat model (an enumerated universe plus one normaliser) need fewer review rounds than guards written from the found instance. **Prediction:** PP-9 rounds 1–5 each found an escape in the row-shape whitelist; after the round that replaced the whitelist with "any cell claims a spend tier" plus one normaliser, no further escape was found. **Evidence:** rounds 1–5 found 12 escapes; round 6 (dispatch rejected by the operator, so the test is the CI run and the acceptance script's 16 named cases) found none. **Status: corroborated, weakly** (the last round did not run).

### 3.7 Review rounds on guard code

**Observation.** PMAT-930..935: estimate 15 turns, actual 58; five cross-vendor rounds, 19 dispositioned findings, one refuted (bold-id). PMAT-742's first run: estimate 7, actual 152.

**Five whys.**
1. Why five rounds? Each round found a real escape in the scanner (no leading pipe, extra pipe, dummy first line, foreign header, backticked tier, `__RECORDED__`, zero-width key, CONFORMANT not deduped).
2. Why one escape per round rather than all at once? Because each lane judged the diff, and a diff that fixes one shape invites the next.
3. Why not a case table first? The case table was built incrementally (13 → 40 cases), each case after its escape.
4. Why does a round cost ~8 turns? Dispatch, receipt, re-run of every claim, fix through the pre-commit hook (which refused the file until 11 unrelated functions were decomposed under the cap), push, receipt regeneration.
5. Why the hook cost? Because the hook refuses any file with pre-existing debt, so touching a debt-carrying file for a one-line fix requires paying the file's whole debt.

**Hypothesis H7.** The pre-commit debt rule converts small fixes into large ones. **Prediction:** the diff of a "one-line" guard fix in a debt-carrying file is dominated by decompositions. **Evidence:** the ledger-scanner fix decomposed 11 functions; the SATD sweep decomposed 11 more in seven files; both diffs were mostly decomposition. **Status: corroborated.**

### 3.8 Toolchain defects

Measured instances in the window: `pmat work add` minted PMAT-744 and PMAT-497 while both were live and rewrote 2,046 lines (paiml-mcp-agent-toolkit#1169); `pmat comply` char-boundary panic (#1159, fixed in pmat 3.36.0); bashrs 7.0.1 single-invocation counts of 13 vs 53 on the same tree, four-file cross-contamination, six documented false positives; the fleet measures with pmat 3.31.0 while the workstation has 3.36.0 (today's complexity-ratchet run was compared across the two before the merge-tree explanation was found — the version-skew hypothesis was **refuted** by re-measuring on the CI host with 3.31.0: 692 rows, no diff).

**Five whys.** 1. Why do tool defects reach the release path? Because pmat, bashrs and pv are dogfooded at head. 2. Why at head? Because the stack is the product. 3. Why is a tool defect expensive here? Because the guard that wraps the tool is fail-closed. 4. Why is there no pin? Because CI installs `cargo install pmat --locked` without a version and the fleet image carries whatever it was built with. 5. Why no cross-version check? Because no guard records the tool version in its baseline (the complexity ratchet prints the version but does not compare it).

**Hypothesis H8.** Unpinned tool versions between workstation and fleet produce false ratchet verdicts. **Prediction:** a baseline seeded on 3.36.0 and measured on 3.31.0 differs. **Evidence:** identical (692 = 692) on the bare branch. **Status: refuted for this pair**; the risk is real in general but was not the cause today.

### 3.9 Estimation

| ticket / phase | est | actual | ratio |
|---|---|---|---|
| PMAT-742 (first run) | 7 | 152 | 21.7× |
| PMAT-742 continuation | 60 | 88 | 1.5× |
| PMAT-929 P1 verify | 13 | 24 | 1.8× |
| PMAT-930..935 scanner | 15 | 58 | 3.9× |
| PMAT-936..948 dogfood gates | 15 | 34 | 2.3× |

**Five whys.** 1. Why 2–4× on every phase? The estimator counts phases, not CI cycles. 2. Why not cycles? Because `estimate.sh` has no measurement of cycles per PR (this report supplies one: 2.3 PR runs and 1.6 merge-group runs per merged PR). 3. Why 21× on the first run? First run in the repo, `[U]`, before any of the above was known. 4. Why does the continuation come closest? Because it was estimated after one run had been measured. 5. Why is the fleet never in the estimate? Because turns are counted, not hours, and a fleet-bound hour costs zero turns while polling.

**Hypothesis H9.** An estimate of `phases × (1 + expected extra cycles) × median turns per cycle` with the measured rates predicts within 1.5×. **Status: untested** — the next run's receipt is the test.

### 3.10 Hardware-window dependencies

perf-matrix rows 2 and 13–21 are UNMEASURED with owners and expiries; the CUDA nightly's PP-26 witness has failed at c=16 every night since 08-30 (9 of 16 slots diverge at chunk 31 against a declared minimum of 64) and is identical run to run — a standing INVALID-CORRECTNESS band owned by #2753/#2809, not a regression; the qwen-story beat B2 fails on gx10 because the SafeTensors reference was never provisioned there. None blocks the tag; all three consume diagnosis time on every release day because their failures land in the same feed as the release checks.

**Hypothesis H10.** A standing failure that does not change is not a delay of the release but of the person reading the feed. **Prediction:** the nightly's verdict has been byte-identical for six nights. **Evidence:** 09-03 and 09-04 witness lines identical (`c=16 … intra_agree_to=31`); earlier nights not inspected. **Status: corroborated for two nights.**

---

## 4. 2026-09-04, the landing day, as a case study

| UTC | event | class | cost |
|---|---|---|---|
| 00:28–08:40 | runner-6 fails every job (14 aprender jobs); merge-queue runs for #2859, #2809, #2861 evicted | environment (3.5) | 8 h of one runner; three queue re-entries |
| 02:38 | #2860 run: lint (rustup download), coverage (runner-6), workspace-test and guard (README count 1,805 ≠ 1,807) | 2 env + 1 merge-tree defect | one cycle |
| 05:13 | PERF-009 fires for the first time (the README step ahead of it had never passed); reveals a guard blind spot (`date -u +%s`) that had hidden two harness scripts on main | dark guard (3.4, 3.6) | one cycle; PMAT-949 |
| 06:04 | #2859 merges; #2860 and #2861 must merge main (roadmap and dogfood.sh conflicts) | queue depth (3.2) | two re-pushes |
| 08:16 | SATD ratchet lower bound fires on the merged tree (54 → 37) | merge-tree (3.4) | one cycle |
| 09:27 | dep-info race on runner-9 while a container step and a host step shared one target dir | environment (3.5) | one cycle |
| 09:35 | predicted-main clean-room started locally (main + three branches) | mitigation (O1) | — |
| 10:10 | clean-room finds the batuta stdio transport race (1 of 6,569); fixed with a deterministic test, mutation RED | real defect found pre-merge | 0 cycles |
| 10:29 | dogfood pre-publish preview: NO-GO on one gate, bashrs 7 findings across two branches; fixed on both | found pre-merge | 0 cycles |
| 10:47 | #2809 merges | — | — |
| 11:2x | complexity ratchet STALE row on the merge commit (#2809 reduced a function on main) | merge-tree (3.4) | one cycle |
| 12:47 | #2861 enters the queue 1 min after its checks pass (other repos paused; fleet 16 → 9 busy) | fleet (3.3) | — |

Nine events; four merge-tree defects, four environment deaths, one real code defect. The two pre-merge instruments (predicted clean-room, dogfood preview) found the two most expensive classes before CI did.

---

## 5. Optimizations, each with the measurement that would show it worked

Ordered by expected gain per unit of change. Each has a falsifier: the number that must move, and the reading that would say it did not.

**O1. Predicted-merge run before every push** (addresses 3.4; largest gain, no CI change). Script `scripts/predict_merge.sh`: build `merge(origin/main, HEAD)` in a scratch worktree and run exactly the tree-property guards from `guard-runner-labels` (README claims, all ratchets, SATD bound, bashrs sweep at `--level error`, PERF-009) plus the crate tests of touched crates. Rebuild after every merge into main (the STALE-row miss today). *Measure:* `guard-runner-labels` failure rate on PR runs (today 4 of 4 tree-property failures would have been caught). *Refuted if* the failure rate does not fall below half within two weeks.

**O2. Split `guard-runner-labels` into a fast tree job and a build job** (3.1, 3.4). The tree-property guards need no cargo and run in under five minutes; put them in a job that runs first and gates the hour-long builds. *Measure:* median minutes-to-first-red on PR runs (today: 30–60 min). *Refuted if* the median stays above 15 min.

**O3. Measure, then fix, the shared-job growth** (3.1, H1′). Before any change: a two-week series of `workspace-test` job durations by runner and by concurrent load, from the jobs API. If the 42 → 94 min spread tracks concurrency, warm one target directory per runner with sccache instead of per-run cold dirs (the "two concurrent workspace-tests starve the box" finding). *Measure:* `workspace-test` p90 under load. *Refuted if* duration does not track load, in which case the cause is test growth and the fix is test partitioning.

**O4. Fleet reservation for the merge queue** (3.3). Label two runners `merge-queue` and route `merge_group` jobs to them; leave the other fourteen shared. *Measure:* merge-group runner wait p90 and eviction count per week (today 80 evictions over six weeks). *Refuted if* evictions do not fall.

**O5. One roadmap file per ticket** (3.2, H2). `docs/roadmaps/items/PMAT-nnn.yaml`, assembled by a script into the current `roadmap.yaml` for readers; PRs stop appending to one line. *Measure:* keep-both resolutions and re-pushes behind a merge (three today). *Refuted if* a PR touching the roadmap still conflicts with the PR ahead of it.

**O6. Environment classification in the hook and the guards** (3.5). Pre-job hook probes the toolchain from the runner root, not the stale workspace (filed on paiml/infra alongside paiml-mcp-agent-toolkit#1185); container steps get their own `CARGO_TARGET_DIR`; a job whose failure is classified ENV posts the classification on the check-run so the reader does not spend two hours on the secondary symptom. *Measure:* jobs failed at "Set up runner" per week (14 today) and dep-info errors (6 today). *Refuted if* either recurs after the hook change.

**O7. Guard authoring rule: threat model first** (3.6, 3.7, H6). A new guard ships with (a) the universe it scans stated as a set, (b) one normaliser, (c) a case table with both polarities, (d) a mutation catalogue, (e) a merge-tree note. Review lanes judge the universe, not the diff. *Measure:* review rounds per guard PR (five for PP-9). *Refuted if* the next three guard PRs average more than two rounds.

**O8. Consolidate the 89 check scripts into a single guard runner** with shared universe and classification functions (3.6). Most scripts re-implement `git ls-files ∪ find`, a case table harness and an ENV/CODE classifier. *Measure:* `ci.yml` line count and the count of scripts with their own universe function. *Refuted if* consolidation adds runtime rather than removing it.

**O9. Pin the toolchain per repo and record it in every baseline** (3.8). `pmat`, `bashrs`, `pv` versions in one manifest consumed by CI, the hook and the workstation; every ratchet baseline carries the version it was seeded with and refuses to compare across versions with a clear message. *Measure:* false ratchet verdicts (zero known; the check is cheap insurance). *Refuted if* a pinned run still differs from the fleet.

**O10. Estimate in cycles, not phases** (3.9, H9). `estimate.sh` reads the last four weeks of this report's numbers (2.3 PR runs, 1.6 merge-group runs per PR, 55-min run, fleet wait p90) and the review-round median. *Measure:* the ratio actual/estimate on the next receipt (five ratios today: 1.5–21.7×). *Refuted if* the next ratio exceeds 2×.

**O11. Pre-commit debt rule scoped to the diff** (3.7, H7). The hook refuses growth in the touched functions, not pre-existing debt elsewhere in the file. *Measure:* lines of decomposition per line of intended fix. *Refuted if* debt grows in touched functions after the change.

**O12. Route standing hardware failures out of the release feed** (3.10). The CUDA nightly and qwen-story post to a `hardware-window` summary with a "same as yesterday" flag; the release checklist reads only checks that changed. *Measure:* minutes spent on unchanged nightlies per release day (about 40 today). *Refuted if* a real regression is missed because it was flagged unchanged — so the flag must be a byte comparison of the witness line, never a status colour.

What these have in common: every one of them moves work from the merge queue, where an hour costs an hour of the whole fleet, to the workstation, where the same check costs minutes and blocks nobody.

---

## 6. Data appendix

- PRs: `gh pr list --state merged` filtered to `mergedAt >= 2026-07-20`, 170 PRs. Time-to-merge = mergedAt − createdAt.
- Runs: `repos/paiml/aprender/actions/runs` for `pull_request`, `merge_group`, `push` with `created >= 2026-07-20`, 1,056 runs. Duration = updated_at − run_started_at for completed runs; wait = run_started_at − created_at.
- Merge-queue events: issue timeline per PR; `removed_from_merge_queue` minus one per merged PR = evictions.
- Fleet: `actions/runs` for each org repo with `created >= 2026-08-07`, run wall-hours summed; capped at 800 runs per repo by pagination, so aprender, forjar and paiml-mcp-agent-toolkit are floors, not totals.
- Guards: `git log --diff-filter=A -- 'scripts/check_*.sh'`; `ci.yml` line counts at `origin/main` and at the last commit before 2026-07-24.
- Jidoka: `.pmat/jidoka.jsonl` (18 entries across the PMAT-742 and PMAT-929 runs).
- Estimates: `docs/audits/impl-estimates.jsonl` on `chore/impl-PMAT-742-receipt` and the PMAT-929 receipt draft.
- 09-04 timeline: this session's status log and the GitHub job records cited by run id in the receipt.
