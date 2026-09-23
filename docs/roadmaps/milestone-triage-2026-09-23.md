# Milestone triage: 0.70.0 and unmilestoned issues (2026-09-23, applied)

**APPLIED 2026-09-23** on the operator's approval ("yes apply both", relayed by aprender-cf): 202 issues were moved to the milestone proposed below (97 out of 0.70.0, 105 from no milestone), each with a comment naming this file, #4024 and its reason. Row requested by aprender-cf (cop) under #3998.

**Corrected after the #4024 quorum (Opus 5.5 lane 1, FAIL):**

- 3 of those moves broke this table's own rule 1: #3421 and #3428 (must-carry in #3999 and #4001) and #3997 (named by five epics). They were **reverted** to their pre-move state, each with a comment, and are now decisions. A 4th, #3845 (the same defect as the DECIDE #3978), was reverted in quorum round 6. A 5th, #3714 (qwen3moe), was moved to 0.71 in error. The operator's MoE rulings, read to the end, finish with 11:49Z "so keep it in" (MoE stays in 0.69.1). Round 8 wrongly re-applied the move on the intermediate 11:44Z ruling; round 9 reverted it to its pre-triage state. **Net moves: 197**, plus 9 corrections re-moved (see Corrections).
- 2 of the comments (#3917, #3532) had failed on a GraphQL error and were posted afterwards.
- 3 edits (#3576, #3431, #3646) hit a transient GraphQL error and succeeded on retry.

**NOT applied (left untouched, pending): 41 rows.** These are the 13 `DECIDE` rows (operator: the original 8, #3951, and 4 of the reverted rows), 7 `close?`, 6 `verify-close`, 14 `0.69.1 (in flight)` (including the reverted #3714 and the relabelled #3839) and 1 `none (pinned)`. No issue was closed by the triage itself; 4 duplicates were closed on the operator's separate ruling (see Corrections).

**Open issues per milestone after the moves, reverts, corrections and duplicate closes** (`gh issue list --milestone`, 2026-09-23 ~19:00 CEST; other sessions keep moving issues, so expect drift): 0.70.0 131 · 0.71.0 105 · 0.72.0 52 · 0.73.0 34 · 0.74.0 19 · 0.75.0 5.

0.70.0 also holds the 4 untouched `close?` rows, the 3 reverted DECIDE rows (#3421, #3428, #3845), the reverted #3714, the relabelled #3839, and issues other sessions added after this table's snapshot (#3988's moves, new filings). The milestone API's `open_issues` also counts pull requests.

Each move was checked against the issue's LIVE state first: an issue closed, or moved off its snapshot milestone since the table was made, would have been skipped (none were).
Snapshot: `gh issue list` at 2026-09-23 ~15:40 CEST, 178 open issues in 0.70.0 and 136 open issues with no milestone.

## Themes

| Release | Theme | Epic |
|---|---|---|
| 0.70.0 | Fast Train: freeze to publish in ≤ 4 h; release/CI/gate speed and reliability | #3998 |
| 0.71.0 | Don't Leave Behind: every Qwen × quant × verb × backend, zero REDs | #3994 |
| 0.72.0 | Agent Ready: `apr serve` as the one gateway (OpenAI + Anthropic), honest telemetry | #4000 |
| 0.73.0 | llama.cpp Parity: decode/prefill/TTFT/memory | #3999 |
| 0.74.0 | Any Model: quant/tensor/arch dispatch consolidation | #4001 |
| 0.75.0 | CRUX declarative fine-tune/distill | #4002 |
| 0.70→0.74 | Debt ratchet, 5 equal slices | #3997 |

## How each row was placed

1. **Named must-carry.** An issue an epic names in its body as a must-carry row goes to that epic's release. A dependency mention, such as #3994 citing #3986 as work that 'matters here', does not count. Named by more than one epic → `DECIDE a/b`, which is the operator's call.
2. **Theme.** Otherwise the issue goes to the release whose exit bar it blocks. Correctness of a certified cell → 0.71; the serve/agent surface and telemetry → 0.72; the performance path (including correctness bugs *in* batched/FP8 paths, which must be fixed before parity is claimed) → 0.73; dispatch/tensor/arch consolidation → 0.74; training → 0.75; release/CI/gates/lock/sweep infrastructure → 0.70.
3. **Debt (#3997).** Debt items are PARKED in a release by pillar: A coverage → 0.70 (the floor gates the 0.70 train); B/C pv + ontology → 0.71; D backlog/docs/packaging → 0.73. These are NOT #3997's pillar slices. #3997 and its step-1 plan (#4003) slice every pillar across every release, and pillar D means triaging and purging the backlog. The operator rebalances.
4. **Not a milestone.** `0.69.1 (in flight)` (checked on no-milestone rows at first; the quorum extended it to 0.70.0 rows carrying a current operator 0.69.1 ruling, #3714 and #3839. Operator rulings are read to the END of their issue threads, and the LATEST one is used: stopping at the first superseding ruling put #3714 on a ruling that had itself been reversed five minutes later): worked in the current train, closes at the tag or carries to 0.70. `verify-close`: the release tree already implements it -- cited in code on origin/release/0.69.1-batch-2 @ c619dddd4 (not yet on main). `close?`: a superseded or stale epic/row. `none (pinned)`: a standing coordination thread.
5. **Review.** A keyword pass placed the rows, and a title-level hand review overrode about 120 of them. That was NOT enough: the #4024 quorum (lane 2) found keyword misplacements, and a body-level re-review of 184 rows (the 182 with a template reason at the time of writing, plus the rows the quorum named) found 10 clearly wrong placements (see **Corrections**). Treat a template reason as the weakest evidence in this table.

**Totals, current table** (the proposal after the 4 reverts, which are now DECIDE rows, and before the 9 corrections and the 4 duplicate closes; the post-state counts are at the top).

- 0.70.0 today (178): 0.69.1 (in flight): 2 · close?: 4 · DECIDE 0.71.0/0.72.0: 1 · DECIDE 0.73.0/0.74.0: 2 · 0.70.0: 76 · 0.71.0: 43 · 0.72.0: 14 · 0.73.0: 16 · 0.74.0: 15 · 0.75.0: 5
- no milestone today (136): 0.69.1 (in flight): 12 · verify-close: 6 · close?: 3 · DECIDE (proposed 0.71.0): 1 · DECIDE (recommend 0.70.0): 1 · DECIDE 0.70.0/0.71.0: 4 · DECIDE 0.70.0/0.71.0 (in flight): 1 · DECIDE 0.70.0/0.71.0/0.72.0: 1 · DECIDE 0.71.0/0.72.0: 2 · 0.70.0: 31 · 0.71.0: 46 · 0.72.0: 10 · 0.73.0: 11 · 0.74.0: 5 · 0.75.0: 1 · none (pinned): 1

## Decisions for the operator

- #3845 → **0.71.0/0.72.0, with #3978**: the same apr code backend-selector defect (REVERTED to 0.70.0 after a move made in error)
- #3971, #3848, #3583, #3752, #3991, #3567, #3602 → **consistency re-check**, like #3483. The quorum flagged each as contradicting the reasoning the table applied elsewhere: #3971 is comparator hygiene (0.70?); #3848 is a run-verb defect (0.71?); #3583 is BF16 admission integrity (0.71?); #3752 is release-gate infrastructure whose S1/S2 siblings sit in 0.70 (0.70?); #3991 is a self-declared 0.69.1 blocker found with #3979 (with #3979?). Not re-moved without a ruling
- #3483 → **re-check 0.71 vs 0.73**: applied as 0.71, but rule 2 sends FP8/batched-path correctness to 0.73 (as #2765). Flagged by the quorum; not re-moved without a ruling
- #3421, #3428 → **operator ruled 0.74 (10:43Z on #3999); move on confirmation** (listed as 0.73.0/0.74.0): must-carry in both #3999 and #4001 (REVERTED to 0.70.0 after a move made in error). Note: an operator comment on #3999 (2026-09-23 10:43Z) already assigns PP-QUANT/PP-TENSOR to 0.74 as its core; #3999's body was not updated. Confirm, then move
- #3997 → **recommend 0.70.0 as the first slice**: the ratchet epic is named by five epics (REVERTED to no milestone after a move made in error)
- #3951 → **0.70.0/0.71.0**: must-carry in #3998 and #3994, and likely resolved by #3990 (re-measure)
- #3950 → **0.70.0/0.71.0**: IQ2_XXS CUDA GEMV: same family as #3953/#3960/#3963, which #3998 and #3994 both name -- operator picks one. _IQ2_XXS (ggml 16) has no CUDA GEMV — 0.69.1 release blocker_
- #3953 → **0.70.0/0.71.0**: named must-carry in #3998 (0.70.0) and #3994 (0.71.0) -- operator picks one. _IQ2_S (ggml 22) has no CUDA GEMV — 0.69.1 release blocker_
- #3960 → **0.70.0/0.71.0**: named must-carry in #3998 (0.70.0) and #3994 (0.71.0) -- operator picks one. _Q2_K (ggml 10) has no CUDA GEMV — 0.69.1 release blocker_
- #3963 → **0.70.0/0.71.0**: named must-carry in #3998 (0.70.0) and #3994 (0.71.0) -- operator picks one. _IQ3_XXS (ggml 18) has no CUDA GEMV — 0.69.1 release blocker_
- #3987 → **0.70.0/0.71.0/0.72.0**: qwen3moe verbs: named must-carry in #3998, #3994 and #4000 -- operator picks one. _qwen3moe CUDA reaches run+qa only: chat rc 8, serve 501/500, code rc 1 — serve loads no mapped model_
- #3978 → **0.71.0/0.72.0**: named must-carry in #3994 (0.71.0) and #4000 (0.72.0) -- operator picks one. _apr code hardcodes 'apr serve --gpu' (no CPU lane, no --max-tokens/thinking flag) and picks a collid_
- #3979 → **0.71.0/0.72.0**: named must-carry in #3994 (0.71.0) and #4000 (0.72.0) -- operator picks one. _apr serve: APR-CPU fallback and safetensors routers have no GET / route index and SSE ends without f_
- #4008 → **0.71.0 proposed (not 0.73/0.74)**: goes with #3977. The operator ruled "MOE goes in .71" on #3994 (2026-09-23 10:43Z), and #3977 is in 0.71.0. Left untouched pending confirmation

## Corrections (APPLIED 2026-09-23, operator via cf)

These rows were first applied as the table proposed, and a body-level re-review showed the placement was clearly wrong. On the operator's approval (verbatim "yes", relayed by aprender-cf) **9 were re-moved**, each checked against the live state before the write, read back afterwards, and commented. **#3513 was WITHDRAWN before applying:** the quorum (round 4, lane 2) showed its stated premise is false. The issue body records that the correctness mitigation shipped in 0.68.2 (qwen35 pins the float GEMV), so the remaining work is making DP4A usable, which is performance, and it stays at 0.73. It is not re-moved.

| # | Applied | Correct | Why |
|---|---|---|---|
| #3779 | 0.75.0 | 0.71.0 | placed by a keyword (finetune.rs): the defect is the apr-cli --features wgpu build failing, the same as #3995 (0.71) |
| #3513 | 0.73.0 | ~~0.71.0~~ WITHDRAWN (stays 0.73.0) | a numerical correctness defect in the DEFAULT sm_89 DP4A GEMV through the Gated DeltaNet path (Qwen3.5), not performance; 'profile' means the error profile |
| #3595 | 0.72.0 | 0.71.0 | apr chat hard-skips CUDA for every qwen35 model: a verb missing on a backend (#3994's bar), not the serve surface |
| #3925 | 0.72.0 | 0.71.0 | the ladder's gibberish detector judges apr chat's separator and reddens every rung: the certification instrument, not the serve surface |
| #3850 | 0.74.0 | 0.71.0 | resolve_qtype decodes an UNKNOWN ggml type as Q4_K on the qa GPU path: quant-admission correctness, not type-list consolidation |
| #4018 | 0.70.0 | 0.71.0 | apr run -v panics on a non-ASCII prompt (a str slice at byte 200): a crash in a release verb, not CI/gate speed |
| #4006 | 0.71.0 | 0.72.0 | apr run labels a mixed UD-IQ2_XXS file by its lm_head qtype: a label only, i.e. honest telemetry |
| #3935 | 0.71.0 | 0.73.0 | a stale README claim: docs debt (pillar D), not pv/ontology |
| #3151 | 0.75.0 | 0.73.0 | CNN autodiff ops: general backlog (pillar D); the issue itself says Qwen3.5 fine-tuning does not need them |
| #3170 | 0.72.0 | 0.70.0 | apr-agent is fleet worktree tooling (its --help created a worktree/branch/lock), not the apr serve/code agent surface |

**Duplicates** (operator: "close the later issue of each as a duplicate of the earlier", after re-verifying each pair). **4 closed** as not planned, each with a 'Duplicate of #N' comment, and its unique detail carried to the survivor: #3995→#3779, #3894→#3891, #3870→#3869, #3883→#3882. **2 pairs SKIPPED as not true duplicates:** #3871/#3940 (the producer recording the checkout HEAD vs the judge never binding apr_sha, different surfaces; #3940's judge half is since implemented by #3957 F2's sha binding, so it is a verify-close candidate) and #3636/#3837 (one unused import vs 60 errors: the same ungated cuda-clippy surface, a different scope).

- #3779 / #3995: the same apr-cli --features wgpu build failure, split across 0.75 and 0.71 (the correction above aligns them)
- #3891 / #3894: the same hardcoded GPU quant list (both 0.74)
- #3869 / #3870: IQ4_NL has no dequant path (both 0.71; #3852 shares the root cause)
- #3882 / #3883: the golden ON prompt in neither declared mode (both 0.71)
- #3871 / #3940: the ladder receipt does not bind the binary (both 0.71)
- #3636 / #3837: clippy --features cuda red and ungated (both 0.70; #3996 related)
- #3984 / #3989: found by the quorum after the ruling, NOT closed: the same driver_cuda_gguf.rs 10 E0063 errors; #3984 is must-carry in #3998. Needs its own ruling

## Currently in 0.70.0

| # | Proposed | Why | Title |
|---|---|---|---|
| #2373 | close? | 0.63.0 dogfood epic: superseded by the per-release dogfood gates; review for close, re-home any open child | Epic: dogfood audit of apr 0.63.0 from crates.io — 201 defects, 24 P0 |
| #3062 | close? | stale or superseded epic/fast-track from an earlier train; review for close, carry any open child to its theme | EPIC: nvidia-cuda-rust-library-integration.md — stabilize GPU quality with NVIDIA's CUDA Rust libraries (0.67) |
| #3081 | close? | stale or superseded epic/fast-track from an earlier train; review for close, carry any open child to its theme | EPIC: release train 0.70.0 — receipts replace refusals |
| #3522 | close? | stale or superseded epic/fast-track from an earlier train; review for close, carry any open child to its theme | OXIDE-001 (0.68.3, fast-track): cuda-oxide pure-Rust #[kernel] → PTX — one GDN kernel ported and measured on l |
| #2616 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | test_discover_tools_no_tools_array is flaky 2/3, and ci.yml documents nextest [profile.ci] as retries=0 when t |
| #2701 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | The dogfood release gate is unrunnable for every sibling repo whenever aprender is not on main |
| #3204 | 0.70.0 | the Mac host as a sweep shard: Fast Train parallel hosts P5 (#3998); its ladder role is 0.71 | mini (Apple M4) is a full-time aprender build host: macOS/arm64 CI leg + runner re-registration (operator 2026 |
| #3206 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | APR-RELEASE-001 §3.6: nothing in the repo writes the build ledger — 1091 records committed, no way to produce  |
| #3258 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | Six concurrent full-tier workspace-test jobs starve every short job for 74 minutes — the merge queue head cann |
| #3446 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | docs(release): APR-RELEASE-001 has no must-carry section on `main` — land §1.5 (must-carry rows; T-0 waits; >  |
| #3454 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | release train: the tag step lives in an untracked per-train autopilot copy — the #3445 milestone read is on no |
| #3457 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | ci_queue_steward.sh defaults to the first OPEN milestone by title — today that is 0.66.0 (shipped, 0 items), n |
| #3458 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | release train: automatic milestone triage at T-24h, T-0 and post-tag |
| #3459 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | release train: move the release autopilot's tag path into the repo — tag step calls scripts/check_milestone_cu |
| #3460 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | check_milestone_cut.sh: exempt the release epic by its issue number from 06x-release-schedule.md §5, not by ti |
| #3462 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | cascade-publish.sh TIERS is not a dependency order — 47 non-dev violations at v0.68.1; it only works through t |
| #3464 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | aprender-orchestrate: test_find_apr_binary asserts the HOST has `apr` on PATH — an environment assertion in a  |
| #3466 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | aprender-serve: the realizar --lib test binary peaks at 33 GB RSS under `cargo test` — OOM-killed by a 32 GB c |
| #3478 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | Deferred: sccache cold/warm + P0·Fan-out measurement on the v0.68.1 fixture (operator 2026-09-18: build server |
| #3518 | 0.70.0 | mini-m4 runner idle: Fast Train parallel hosts (#3998) | MINI-001: the mini-m4 runner is idle by construction — no aprender job targets its labels; move the OS-neutral |
| #3531 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | PACK-001 first main run: three shards of one push chose three tiers (reflog-derived base) and partition 2/3 ne |
| #3543 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | The post-publish dogfood CANNOT pass: version-unpublished is not phase-aware and the banner still says pre-rel |
| #3544 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | Make the host-receipt DEFER fail closed — 0.68.2 discharged it by hand, and the gate's first run fabricated 4  |
| #3546 | 0.70.0 | install.sh 404 breaks post-publish verification: Fast Train release path (#3998) | install.sh does not exist: /releases/latest/download/install.sh 404s, nothing in the tree provides it — yet T- |
| #3554 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | jq 1.6 vs 1.7+ disagree on blank input and our fleet straddles the boundary — a 'jq -e' gate passes on lambda/ |
| #3561 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | T-0 preflight returns GO when the dogfood crashes before emitting any row — absence of [FAIL] lines is read as |
| #3562 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | Two consecutive non-passing plan quorums on a must-carry ticket must escalate automatically — #3452 sat throug |
| #3579 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | 12 guards build their universe with find and 8 do not exclude .claude/worktrees — they scan 3 trees at 3 diffe |
| #3584 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | check_apr_bin_pinned.sh prescribes a remedy that is wrong for post-publish verification — apr_bin.sh asserts a |
| #3586 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | 7 scripts resolve the repo root with an unguarded 'git rev-parse --show-toplevel' — git refuses the container' |
| #3587 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | nextest fail-fast discards verdicts: one failure hid 15 unmeasured commands and the shard read as '1 failure', |
| #3591 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | check_hardcoded_paths.sh's default mode scans contracts only and PASSes — the real gate is --full-if-capable,  |
| #3592 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | The hardcoded-path analyser ignores /mnt entirely — a committed file with both /home and /mnt paths scores +1, |
| #3593 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | book-contracts.yml compiles and RUNS the 27 chapter examples on floating `stable`, not the 1.93.0 pin — three  |
| #3594 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | The autonomous-merge arming gate has six assertions that are inert on jq 1.6 (the majority of the pool) — a ze |
| #3603 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | guard-cargo reports a ratchet FAIL when cargo never ran: the per-run target dir becomes root-owned intermitten |
| #3607 | 0.70.0 | gate red-set nondeterminism: fail-fast gates must be deterministic (#3998) | #3540's red set varies across heads that differ by an 11-line quorum JSON: determinism (X64) and a PATH-resolu |
| #3612 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | Sharding works but is eaten by runner scarcity: merge-queue shards wait 26-28 min for a runner vs 13-14 min ru |
| #3615 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | Nothing consumes removed_by: no T-0 expiry check, no slips counter, no two-slip andon — refusals will carry de |
| #3625 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | A shard killed by SIGNAL reports as a bare 'test run failed' — 11,757/40,750 not run on yoga-build3, nextest p |
| #3628 | 0.70.0 | a wide refactor DIRTYs unrelated PRs: merge speed (#3998) | a wide refactor of aprender-contracts made three unrelated PRs DIRTY on the same three files — the class needs |
| #3635 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | apr_bin.sh / pv_bin.sh refuse a FOREIGN worktree's binary but accept an OWN-tree binary OLDER than the tree —  |
| #3636 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | clippy --features cuda is red on main (unused rms_norm_forward import in aprender-train) and no gate runs that |
| #3639 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | Two review receipts, one reader: the §13.11 shadow bot reads evidence/pr-review/ (no producer since #2875) whi |
| #3643 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | Ratchets compare a STALE merge ref against origin/main's LIVE tip ('no merge-base available; stricter') — a ma |
| #3649 | 0.70.0 | python-free automation: runners without python red guards (#3998) | Fleet declares NO python in automation (infra/CLAUDE.md; sovereign-ci image has no python3 by design) — aprend |
| #3650 | 0.70.0 | roadmap aggregate re-creates merge conflicts on every PR: merge speed (#3998) | The committed roadmap.yaml AGGREGATE re-creates the conflict the fragments removed — every fragment-minting PR |
| #3658 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | Every merge-group run is tier=full: ci_test_tier.sh requires HEAD^2 but the queue is SQUASH (one parent) — a t |
| #3667 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | ci.yml:2783 (ci_gpu_touched feed) builds its touched list with rename detection on — a CUDA source moved out o |
| #3668 | 0.70.0 | docs-only PRs spend 45 min in cargo guards: CI speed (#3998) | Docs-only PRs still spend ~45 min in guard-cargo (25) and guard-tree (15): gate them on a single docs_only dec |
| #3676 | 0.70.0 | debt ratchet pillar A (#3997), slice 1: the coverage floor gates the 0.70 train | CI diet: aprender itself burns 99 intel runner-hours/day — coverage on tag/nightly only, path-filter book jobs |
| #3679 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | check_ont_ratchet.sh probes a BARE pv (`pvbin="$(command -v pv)"`) — on a fleet-clobbered runner (pv 0.65.2, n |
| #3680 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | check_sourced_libs_option_neutral is blind to scripts/lib/: it resolves a sourced path by BASENAME in scripts/ |
| #3683 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | lib_baseline_ratchet.sh: a git refusal (container "dubious ownership") is reported as "this branch WROTE it" — |
| #3684 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | release/lib_release_params.sh: release_state_root silently falls back to the tree root when git fails — releas |
| #3690 | 0.70.0 | debt ratchet pillar A (#3997), slice 1: the coverage floor gates the 0.70 train | Release T-4 does not wait for the tag's coverage run — a COV_FLOOR breach on the tag reds a run but does not s |
| #3694 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | parity_receipt_denominator.sh needs a python-free classify() — then drop its UNMEASURED path |
| #3697 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | guard-tree's no-cargo guards depend on python3 + site-packages: 10 guards (16 rows) red without yaml/tomli, 21 |
| #3699 | 0.70.0 | release bump PR fails its own R-2 gate: release path (#3998) | prepare_bump.sh --ship opens a bump PR whose body fails §6 R-2 (check_pr_closes_issue): the CHANGELOG's contex |
| #3701 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | binary-release.yml's CPU apr leg takes yoga-gpu (the only yoga sm_89 GPU runner) and serializes the tag's B2-G |
| #3703 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | aprender-verify-ml test_pipeline_generate_small asserts wall-clock (generation_time_ms > 0) — failed v0.69.0's |
| #3705 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | lib tests assert an UPPER bound on measured elapsed time (elapsed < N) — the slow-host twin of #3703; 59 candi |
| #3708 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | autopilot's INHERITED dogfood GO writes no receipt that check_publish_preflight.sh R5 accepts — v0.69.0 stoppe |
| #3709 | 0.70.0 | stop tracking roadmap.yaml: removes the merge-conflict source (#3998) | aprender: stop tracking roadmap.yaml + Pmat-Ticket trailers — the precondition for sovereign-ci roadmap-fragme |
| #3713 | 0.70.0 | GPU test SIGKILLs on GB10 unified memory: CUDA tests under the lock must be reliable (#3998) | aprender-gpu test_gpu_allocation_under_pressure SIGKILLs on GB10 unified memory (host OOM) — size pressure fro |
| #3806 | 0.70.0 | python-free roadmap guards (#3697 batch 2): runner portability (#3998) | #3697 batch 2: the three roadmap guards are UNMEASURED, not RED, on a runner without python3 / PyYAML |
| #3818 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | Arm 4 (`present`) is RED on every live PR and required by nothing — no PR-review receipt has landed since #349 |
| #3831 | 0.70.0 | llama-cli resolution on every host: the sweep's oracle must resolve (#3998) | A bare `llama-cli` resolves a DEAD shadow on lambda and nothing on gx10, while the pinned build works on both  |
| #3834 | 0.70.0 | comparator installs on every host: the sharded CRUX sweep needs them (#3998) | 0.70.0 comparator-install row (#3739): llamafile absent on both hosts, transformers lambda-only, ollama models |
| #3836 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | check_apr_bin_pinned: an opener INSIDE a quoted string is a latent false positive (sixth wrong pattern in this |
| #3837 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | The --features cuda lint surface has never been gated: clippy -p apr-cli --lib --features cuda is RED with 60  |
| #3839 | 0.69.1 (in flight) | operator ruling 2026-09-23 07:12Z (recorded on #3714), "fold in 88% coverage": a 0.69.1 blocker. The coverage half has no later ruling; #3957 at 07:46Z reaffirms coverage ≥ 88 as blocking 0.69.1. Never moved; relabelled after the #4024 quorum | Coverage owes 0.69.1: 87.82% vs floor 88, and COVERAGE_EXCLUDE_REGEX has described a different tree since Apri |
| #3841 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | Fold PMAT-272 guard + perf-matrix fixes into 0.70.0 (parked off the 0.69.1 batch) |
| #3854 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | Wire check_unwired_capabilities.sh into guard-tree — it ships in tier3 only, so CI never runs it |
| #3855 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | ONE GATE FRAMEWORK: 143 gates, ~25 baselines, 39 ratchets, 38 mutation proofs — one and only one way to build  |
| #3864 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | Guard: `rc=$?` after a pipeline reads the LAST command's status — the documented defect (#2336, #2360) has no  |
| #3865 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | Guard: unbraced "$var:path" in a git-ref position — zsh eats the path's first letter and git says 'Not a valid |
| #3874 | 0.70.0 | pv exit code reds a required gate on fleet state: fail-fast gates (#3998) | pv exits 3 on a schema-name refusal, not 2 — so a schema bump reds a required gate on fleet state the PR canno |
| #3887 | 0.70.0 | case-table harness floor: part of ONE GATE FRAMEWORK #3855 (#3998) | Case-table harness: a red-expecting case with no must_match cannot say WHY it is red — a mutant can survive by |
| #3998 | 0.70.0 | the theme epic itself | EPIC 0.70.0: Fast Train — freeze→publish in ≤4 h; lock scoped to GPU work, sharded sweep, idle same-arch pre-s |
| #4021 | 0.70.0 | CRUX judge thinking-ON calibration: filed for 0.70 at the cop's request; the sweep judge (#3998) | CRUX judge: thinking-ON cells must not be judged on </think> closure alone — use answer-when-both-close, else  |
| #3269 | 0.71.0 | debt ratchet pillars B/C (#3997: pv at the deepest level, ontology merge), slice 2 | ONT-001: aprender owns 16 untriaged ontology rows — one per train, ratcheted (APR-RELEASE-001 §11) |
| #3483 | 0.71.0 | FP8 prefill fails CPU parity (applied as 0.71). NOTE: rule 2 sends FP8/batched-path correctness to 0.73, as it did #2765; the #4024 quorum flagged this as inconsistent. The operator should re-check; it is not re-moved without a ruling | FP8 batched prefill fails CPU parity at the post-prompt decode step — catastrophic on per-head-QK-norm models  |
| #3555 | 0.71.0 | model/backend correctness or the certified matrix: Don't Leave Behind (#3994) | Name the Qwen model 0.69 blesses and commit its parity receipt at the SERVING shape — our gates are all single |
| #3558 | 0.71.0 | first-party Qwen3.5 selection + evidence: part of the certified matrix (#3994) | First-party dogfood: Qwen3.5 at a Pareto-optimal size — publish the selection rule and the evidence a consumer |
| #3559 | 0.71.0 | debt ratchet pillars B/C (#3997: pv at the deepest level, ontology merge), slice 2 | ONT-001 to 80% in 0.69 (≈7 more rows) with SHACL armed and falsifiable — and the §11 one-row-per-train conflic |
| #3560 | 0.71.0 | model/backend correctness or the certified matrix: Don't Leave Behind (#3994) | Docs, all 1037 examples and the cookbook: pv-SHACL validated and current for Qwen3.5 (6 of ~1500 artifacts men |
| #3567 | 0.71.0 | debt ratchet pillars B/C (#3997: pv at the deepest level, ontology merge), slice 2 **[an older operator ruling (09-20) made it 'the enabling row' for the ontology rows; flagged for re-check]** | The fleet-pinned pv is 0.65.2 and has no 'extract' subcommand and no '--gate' flag — every SHACL gate measured |
| #3569 | 0.71.0 | debt ratchet pillars B/C (#3997: pv at the deepest level, ontology merge), slice 2 | Decision 8: census.json becomes derived (CI computes, guard refuses PR diffs); lint-baseline counts become a b |
| #3573 | 0.71.0 | substring-discharged claims: debt ratchet pillar B (#3997), slice 2 | PP-LLAMA-001 §12: a substring test cannot tell a citation from a claim — five live rows discharged themselves  |
| #3576 | 0.71.0 | model/backend correctness or the certified matrix: Don't Leave Behind (#3994) | No committed parity receipt names an external comparator — the 0.98 threshold's own basis is apr-vs-apr self-c |
| #3577 | 0.71.0 | model/backend correctness or the certified matrix: Don't Leave Behind (#3994) | Receipts under contract: parity-receipt-v1 shape + extract:parity-receipt + back-fill the 7 comparator-less re |
| #3602 | 0.71.0 | GPU result fails correctness and falls back: backend correctness (#3994) **[an older operator ruling (09-20) said 'stays P0 on 0.69'; flagged for re-check]** | apr run --gpu is 2-3x SLOWER than --no-gpu on the dense path: the GPU result fails correctness (cosine 0.4153) |
| #3610 | 0.71.0 | debt ratchet pillars B/C (#3997: pv at the deepest level, ontology merge), slice 2 | pv: a shapes gate reports Pass AND armed over ZERO focus nodes — aggregate swallows decline: NoFocus |
| #3611 | 0.71.0 | debt ratchet pillars B/C (#3997: pv at the deepest level, ontology merge), slice 2 | pv: implement sh:lessThan / sh:lessThanOrEquals — SHACL Core property-pair constraints |
| #3624 | 0.71.0 | debt ratchet pillars B/C (#3997: pv at the deepest level, ontology merge), slice 2 | pv: by_entity_type is a hand-written list parallel to Σ — derive it from ontology.yaml, or refuse when an impl |
| #3630 | 0.71.0 | debt ratchet pillars B/C (#3997: pv at the deepest level, ontology merge), slice 2 | 51 contract test files say 'DO NOT EDIT — regenerate with pv probar --binding' and pv probar cannot produce an |
| #3640 | 0.71.0 | debt ratchet pillars B/C (#3997: pv at the deepest level, ontology merge), slice 2 | pv extract emits a PARTIAL graph on an unparseable binding.yaml (one dropped quote → 356 triples silently gone |
| #3641 | 0.71.0 | repo-graph freshness floor, with #3640: debt ratchet pillar C (#3997), slice 2 | The repo-graph freshness gate compares committed == fresh and passes when both are 356 triples short — assert  |
| #3663 | 0.71.0 | model/backend correctness or the certified matrix: Don't Leave Behind (#3994) | apr convert --quantize q4k fails on a plain Q4_K_M GGUF: the Q4K passthrough's detector admits Q5_K, its write |
| #3693 | 0.71.0 | CUDA correctness at long context on a certified Qwen3.5 cell: Don't Leave Behind (#3994) | Qwen3.5-9B Q4_K_M on CUDA at ~12k-token context corrupts identifiers it quotes from the prompt (as_millis→as_m |
| #3714 | 0.69.1 (in flight) | qwen3moe CUDA: a 0.69.1 item (the operator's MoE rulings, read to the end: 07:12Z "fold in MoE"; 11:44Z on #3987 "yes push", out of 0.69.1 into 0.71.0; 11:49Z on #3987 "so keep it in", MoE STAYS in 0.69.1, the latest). The table's 0.71 move was an error; round 7 reverted it, round 8 wrongly re-applied it on the 11:44Z ruling, and round 9 reverted it again to its pre-triage state | qwen3moe (Qwen3-30B-A3B, Qwen3-Coder-30B-A3B, Q4_K_M) has NO working CUDA path: apr run --gpu falls back to CP |
| #3822 | 0.71.0 | transformer_layer_indexed hardcodes the Q4K kernel: every quant must work (#3994) | transformer_layer_indexed hardcodes the Q4K kernel for Q/K/attn_output under a comment claiming exhaustive dis |
| #3823 | 0.71.0 | model/backend correctness or the certified matrix: Don't Leave Behind (#3994) | q8_gemv_tests.rs: 15 tests, 4 asserts — dtype.rs whitelists Q8_0 onto the GPU as a 'verified GPU GEMV kernel'  |
| #3824 | 0.71.0 | model/backend correctness or the certified matrix: Don't Leave Behind (#3994) | gpu_state_isolation: fails on coder-0.5b, and its own message truncates both outputs to an identical prefix so |
| #3827 | 0.71.0 | model/backend correctness or the certified matrix: Don't Leave Behind (#3994) | try_wgpu_generate's doc claims "Proven: cosine=0.999863 on Blackwell sm_121" — #3757 measured 0.955046 on gx10 |
| #3846 | 0.71.0 | model/backend correctness or the certified matrix: Don't Leave Behind (#3994) | THE UNIVERSE IS A GLOB: 10 of 28 held models are never measured, because inventory.patterns only matches q4_k/ |
| #3851 | 0.71.0 | model/backend correctness or the certified matrix: Don't Leave Behind (#3994) | Qwen3.5-4B-Q4_K_M: deterministic GPU/CPU divergence at exactly one interior position (21) on the hybrid forwar |
| #3852 | 0.71.0 | model/backend correctness or the certified matrix: Don't Leave Behind (#3994) | owned_fused_matmul refuses IQ4_NL while its own message claims IQ* is supported — the refusal overclaims its c |
| #3858 | 0.71.0 | debt ratchet pillars B/C (#3997: pv at the deepest level, ontology merge), slice 2 | falsification_tests[].test does not bind at all — a prose test name is decoration, and prose is the majority s |
| #3866 | 0.71.0 | debt ratchet pillars B/C (#3997: pv at the deepest level, ontology merge), slice 2 | check_readme_claims.sh --self-test: a must-RED row returns GREEN, because two counters measure different popul |
| #3867 | 0.71.0 | debt ratchet pillars B/C (#3997: pv at the deepest level, ontology merge), slice 2 (NOTE: a README gate boundary is arguably docs debt, pillar D / 0.73; the reason is flagged, not re-moved) | README.md is not ungated — it is gated in a REGION, and the boundary is invisible in the rendered file: enumer |
| #3868 | 0.71.0 | model/backend correctness or the certified matrix: Don't Leave Behind (#3994) | tensor-layout-v1.yaml asserts a compile-time guarantee that is FALSE — and the copy that ships to crates.io ca |
| #3869 | 0.71.0 | model/backend correctness or the certified matrix: Don't Leave Behind (#3994) | IQ4_NL (ggml type 20) has no dequant path in realizar — two Qwen2.5 inventory models cannot generate on either |
| #3871 | 0.71.0 | model/backend correctness or the certified matrix: Don't Leave Behind (#3994) | Re-apply #3784's binary-identity fix as a design: the ladder receipt names the CHECKOUT HEAD, not the binary t |
| #3872 | 0.71.0 | model/backend correctness or the certified matrix: Don't Leave Behind (#3994) | the ladder's log truncates diagnostics mid-number ([:70]), producing a complete-looking WRONG value — plus one |
| #3873 | 0.71.0 | model/backend correctness or the certified matrix: Don't Leave Behind (#3994) | the golden-output gate reports a GPU failure while asserting a CPU pass it never measured — it turned a model  |
| #3883 | 0.71.0 | model/backend correctness or the certified matrix: Don't Leave Behind (#3994) **[CLOSED as a duplicate of #3882, see Corrections]** | the golden gate's thinking-ON prompt is in NEITHER mode the model declares — Qwen3.5 inherits Qwen3's prefill  |
| #3888 | 0.71.0 | debt ratchet pillars B/C (#3997: pv at the deepest level, ontology merge), slice 2 | apr-cli ships 9 INVALID contracts that nothing reads, and no gate looks at them — pv lint never leaves the roo |
| #3890 | 0.71.0 | debt ratchet pillars B/C (#3997: pv at the deepest level, ontology merge), slice 2 | apr kernel-explain reports ProofLevel::Proven from a SUBSTRING — nothing runs kani, and fixing the shipped con |
| #3923 | 0.71.0 | model/backend correctness or the certified matrix: Don't Leave Behind (#3994) | AprV2ModelCuda is Q4K-only by construction and wrong for Q4K — no configuration does useful GPU work |
| #3924 | 0.71.0 | model/backend correctness or the certified matrix: Don't Leave Behind (#3994) | A GPU/CPU correctness gate exists and only the infer module uses it — enumerate every entry point |
| #3937 | 0.71.0 | a CUDA chat cell that ran on CPU reads rc=0: an honest certified cell (#3994) | A CUDA chat cell can say rc=0 for a chat that ran on CPU: chat cannot refuse a forced accelerator, and the pro |
| #3941 | 0.71.0 | debt ratchet pillars B/C (#3997: pv at the deepest level, ontology merge), slice 2 | PROVABILITY-001 requires every kernel contract to declare kani_harnesses, and nothing in this repo runs kani — |
| #4016 | 0.71.0 | model/backend correctness or the certified matrix: Don't Leave Behind (#3994) | Process rule: llama.cpp fit (llama-fit-params) is the required placement gate for every model certification/te |
| #2507 | 0.72.0 | serve/agent/telemetry surface: Agent Ready (#4000) | R13: `apr serve` has three different HTTP surfaces and which one you get depends on the format of the file you |
| #2853 | 0.72.0 | serve/agent/telemetry surface: Agent Ready (#4000) | SafeTensors serve route drops `ignore_eos` and `seed`: apr-cli's ChatCompletionRequest has no such fields |
| #3170 | 0.72.0 | serve/agent/telemetry surface: Agent Ready (#4000) **[CORRECTED 0.72.0 → 0.70.0, see Corrections]** | apr-agent accepts any flag as a task slug — 'apr-agent --help' created a worktree, a branch and a claim lockfi |
| #3541 | 0.72.0 | serve/agent/telemetry surface: Agent Ready (#4000) | apr run: the --no-gpu arm prints no Backend: line, so the CPU arm cannot be proved (post-publish 0.68.2) |
| #3542 | 0.72.0 | apr run reports no tok/s: honest telemetry (#4000) | apr run emits no tok/s — every throughput claim from it is wall time including model load (post-publish 0.68.2 |
| #3545 | 0.72.0 | serve/agent/telemetry surface: Agent Ready (#4000) | apr devices reports 'cuda unavailable reason=NotCompiled' on a binary that demonstrably runs CUDA — the first  |
| #3553 | 0.72.0 | serve/agent/telemetry surface: Agent Ready (#4000) | prefix_cache_hits is counted and unit-tested but exposed nowhere: /metrics has no prefix/cache counter, so a p |
| #3574 | 0.72.0 | parity receipt for the agent decide lane: Agent Ready (#4000) | ARB-APR-3: parity receipt for the decide lane's cell on the agent's host (Qwen3.5-9B-Q4_K_M @ lambda-labs) |
| #3595 | 0.72.0 | serve/agent/telemetry surface: Agent Ready (#4000) **[CORRECTED 0.72.0 → 0.71.0, see Corrections]** | apr chat hard-skips CUDA for every qwen35 model and prints a banner citing #3090 as unimplemented — #3090 ship |
| #3598 | 0.72.0 | named must-carry in epic #4000 | 0.69 serve + instrument lane: apr run --json timing fields, apr serve resident, apr devices stops reporting No |
| #3776 | 0.72.0 | serve/agent/telemetry surface: Agent Ready (#4000) | apr code --emit-trace never records a tool call (4 records, one text block; code.rs calls it 'M29+') — the tra |
| #3826 | 0.72.0 | serve/agent/telemetry surface: Agent Ready (#4000) | apr run --format json reports "fell_back": false on a run whose own stderr says "attempting fallback" |
| #3838 | 0.72.0 | serve/agent/telemetry surface: Agent Ready (#4000) | 0.70.0: `apr serve` outlives its launcher and keeps the port — a leaked listener on a shared box (found via #3 |
| #3845 | DECIDE 0.71.0/0.72.0 | the same defect as #3978 (apr code has no backend selector / no CPU lane), which #3994 and #4000 both name -- operator picks with #3978. Moved to 0.72 in error and REVERTED to 0.70.0 after the #4024 quorum | 0.70.0: `apr code` has no backend selector — it spawns its own `apr serve`, so "code on cpu" vs "code on cuda" |
| #3848 | 0.72.0 | serve/agent/telemetry surface: Agent Ready (#4000) **[consistency re-check, see Decisions]** | apr run --chat is INERT: the chat template is applied whether or not the flag is passed, and there is no way t |
| #2584 | 0.73.0 | undocumented --task arms: debt ratchet pillar D (#3997), slice 4 | apr eval --task: 8 of 10 dispatch arms are reachable but undocumented in --help |
| #2730 | 0.73.0 | cbtop PASS over its own grade F: the perf instrument must be honest before parity (#3999) | cbtop reports status PASS and ci_result green over its own grade F when the profiler returns zero bricks |
| #2753 | 0.73.0 | batched CUDA decode correctness: must be fixed before 0.73 claims any batched throughput (#3999) | P0: batched CUDA decode emits garbage for every m>1 — PERF-001's 3.32x aggregate is throughput of garbage toke |
| #2765 | 0.73.0 | FP8 prefill GEMM reads stale scratch: prefill path correctness is a precondition of prefill parity (#3999) | cublas_prefill_fp8_gemm reads 16 rows of FP8 scratch where 1 was written: the padded tail is the previous GEMM |
| #2770 | 0.73.0 | cuBLAS decode diverges at m>=4: batched decode correctness before parity (#3999) | cuBLAS decode route diverges at m>=4: coherent but different continuation, so c=4 fails parity while c=2 passe |
| #3076 | 0.73.0 | performance/decode-prefill path: llama.cpp Parity (#3999) | F16/BF16 GGUF CPU matmul has no SIMD path — unquantized models appear to hang |
| #3161 | 0.73.0 | debt ratchet pillar D (#3997: backlog, docs, packaging), slice 4 | P1: three license states across 79 crates — 14 publishable crates offer Apache-2.0 with no LICENSE-APACHE to b |
| #3513 | 0.73.0 | performance/decode-prefill path: llama.cpp Parity (#3999) **[correction WITHDRAWN: stays 0.73.0, see Corrections]** | DP4A Q4K/Q6K GEMV profile is catastrophic through the Gated DeltaNet recurrence — Qwen3.5 on sm_89 needs the f |
| #3547 | 0.73.0 | debt ratchet pillar D (#3997: backlog, docs, packaging), slice 4 | crates/aprender-serve/models/qwen2-0.5b-q4.gguf is 0 bytes — the in-tree fixture model cannot load |
| #3548 | 0.73.0 | debt ratchet pillar D (#3997: backlog, docs, packaging), slice 4 | aprender-viz's lib is trueno_viz and nothing says so — 'use aprender_viz::' does not compile and no tool will  |
| #3549 | 0.73.0 | debt ratchet pillar D (#3997: backlog, docs, packaging), slice 4 | aprender-viz ships 396 KB of fixtures/breaks/ to every consumer in the published .crate |
| #3550 | 0.73.0 | debt ratchet pillar D (#3997: backlog, docs, packaging), slice 4 | No consumer-shaped door into aprender-viz: drawing a chart compiles aprender-compute, aprender-quant and apren |
| #3551 | 0.73.0 | debt ratchet pillar D (#3997: backlog, docs, packaging), slice 4 | aprender-viz is f32, consumers are f64: the narrowing at the DataFrame boundary is undocumented and unguarded |
| #3552 | 0.73.0 | debt ratchet pillar D (#3997: backlog, docs, packaging), slice 4 | Facet/Coord have no runnable example — EV-15/EV-16 have nothing to build against |
| #3859 | 0.73.0 | classical-ML O(n^2) split search: debt ratchet (#3997) pillar D, slice 4 (0.73). It is not LLM parity work | RandomForestRegressor/DecisionTreeRegressor carry the identical O(n²) split search — #3815's fix does not reac |
| #3915 | 0.73.0 | a falsified optimization left as dead env-gated code: debt ratchet (#3997), slice 4 | DIRECT_FP32_GEMV: a FALSIFIED optimization left as an env-gated branch that no gate executes |
| #3075 | 0.74.0 | Phi-2/Phi-3 GPU eligibility: not Qwen, so Any Model (#4001), not 0.71 | Phi-2/Phi-3 GGUF never GPU-eligible: LayerNorm kernel exists but isn't wired into forward_gpu_resident |
| #3077 | 0.74.0 | architecture support table, generated from TRAITS with #3443: Any Model (#4001) | Docs: publish a GPU-vs-CPU model architecture support table (no user-facing doc exists today) |
| #3418 | 0.74.0 | named must-carry in epic #4001 | ARCH: quant-type dispatch is duplicated across ~30 files with no single source of truth — propose a dedicated  |
| #3421 | DECIDE 0.73.0/0.74.0 | named must-carry in #3999 (0.73) and #4001 (0.74) -- operator picks one. Moved to 0.74 in error and REVERTED to 0.70.0 after the #4024 quorum | EPIC: PP-QUANT-001 — quant-type dispatch, Phase 0/1 (0.69 train) |
| #3428 | DECIDE 0.73.0/0.74.0 | named must-carry in #3999 (0.73) and #4001 (0.74) -- operator picks one. Moved to 0.74 in error and REVERTED to 0.70.0 after the #4024 quorum | EPIC: PP-TENSOR-001 — a tensor that has no bytes is a different type from one that does (MoE / tied-embedding  |
| #3429 | 0.74.0 | quant/tensor/arch dispatch consolidation: Any Model core (#4001) | PP-QUANT-001 M3: regression fixtures — #1749 #1789 #2535 #3341 (+ #3091 reopen) on tiny synthetic GGUFs, obser |
| #3431 | 0.74.0 | quant/tensor/arch dispatch consolidation: Any Model core (#4001) | PP-QUANT-001 M2: reconciliation gate — `GgmlType` vs the vendored `ggml.h` id list at a pinned sha; exhaustive |
| #3433 | 0.74.0 | quant/tensor/arch dispatch consolidation: Any Model core (#4001) | PP-TENSOR-001 T1: typed `TensorStorage` — a dense consumer cannot receive an MoE placeholder |
| #3434 | 0.74.0 | quant/tensor/arch dispatch consolidation: Any Model core (#4001) | PP-TENSOR-001 T2: sentinel ban — `qtype: 0` + `byte_size: 0` may not be constructed; 0 is F32 |
| #3435 | 0.74.0 | quant/tensor/arch dispatch consolidation: Any Model core (#4001) | PP-TENSOR-001 T3: diagnostic contract — an absent tensor is reported as absent, never as truncated, corrupt, o |
| #3441 | 0.74.0 | quant/tensor/arch dispatch consolidation: Any Model core (#4001) | PP-QUANT-001 Q3: tier-0 scalar dequant for 35/35 live ggml types — every unsupported-type refusal becomes a sl |
| #3442 | 0.74.0 | quant/tensor/arch dispatch consolidation: Any Model core (#4001) | PP-QUANT-001 Q5: upstream drift job — a new `ggml_type` id upstream opens an issue here the day it lands |
| #3443 | 0.74.0 | quant/tensor/arch dispatch consolidation: Any Model core (#4001) | PP-QUANT-001 P4: #3077 support table GENERATED from `TRAITS` (type × backend × tier) |
| #3583 | 0.74.0 | quant/tensor/arch dispatch consolidation: Any Model core (#4001) **[consistency re-check, see Decisions]** | ggml_dtype_element_size is ordered by its own comment, not by ggml id: BF16 reads 0.375 B/elem instead of 2.0, |
| #3820 | 0.74.0 | name-derived model config: the pattern Any Model replaces with config (#4001) | estimate_model_params_from_name documents 0.0 as "assume large model" but its only caller tests params_b < 2.0 |
| #3850 | 0.74.0 | quant/tensor/arch dispatch consolidation: Any Model core (#4001) **[CORRECTED 0.74.0 → 0.71.0, see Corrections]** | resolve_qtype silently decodes an UNKNOWN ggml quant as Q4_K, and the whitelist that would refuse it is not on |
| #3891 | 0.74.0 | quant/tensor/arch dispatch consolidation: Any Model core (#4001) | A GPU-supported quant list is hardcoded in a user-facing error string — a third source of truth beside gpu_sup |
| #2906 | 0.75.0 | fine-tune/train surface: CRUX declarative fine-tune/distill (#4002) | R-3: T-6 honest training banner: -m lora --gpu-backend cuda refuses or trains on the GPU; the cuBLAS-backward  |
| #2924 | 0.75.0 | fine-tune/train surface: CRUX declarative fine-tune/distill (#4002) | T-2: apr finetune --max-seq-len honoured or refused with one line and the refusal code on every path; the effe |
| #2926 | 0.75.0 | fine-tune/train surface: CRUX declarative fine-tune/distill (#4002) | T-0: the four WT receipts: Unsloth QLoRA and apr finetune -m qlora on Qwen2.5-7B-Instruct, both hosts, n=5 int |
| #3151 | 0.75.0 | fine-tune/train surface: CRUX declarative fine-tune/distill (#4002) **[CORRECTED 0.75.0 → 0.73.0, see Corrections]** | P1: autodiff op coverage — conv/pool/embedding/gather; aprender cannot train a CNN today |
| #3779 | 0.75.0 | fine-tune/train surface: CRUX declarative fine-tune/distill (#4002) **[CORRECTED 0.75.0 → 0.71.0, see Corrections]** | cargo check -p apr-cli --features wgpu fails: finetune.rs imports entrenar's WgpuInstructPipeline / wgpu_train |

## Currently with no milestone

| # | Proposed | Why | Title |
|---|---|---|---|
| #3907 | 0.69.1 (in flight) | PMAT-3907 budget wiring was worked in this train; verify, else 0.71 | THINKING_ON_BUDGET is one 8B model's measurement applied to every thinking model, including a 0.8B — and the t |
| #3930 | 0.69.1 (in flight) | Qwen3NoThink prefill vs the model's template: superseded by #3990 (the model's own template); close with #3990 | Qwen3NoThink prefills <think>\n</think>\n where the model's own template uses <think>\n\n</think>\n\n — the mo |
| #3948 | 0.69.1 (in flight) | folded with #3961 on fix/3957-gate-hardening (aprender-6c); verify and close | golden_output thinking-ON judge passes an EMPTY <think></think> — the leg cannot tell reasoning from skipping  |
| #3951 | DECIDE 0.70.0/0.71.0 (in flight) | named must-carry in #3998 (0.70) and #3994 (0.71) -- operator picks. Also: the F9 oracle (aprender-83, PMAT-3952-crux-greedy2@a78301558) shows the block CLOSES on the official ON template, so it is likely resolved by #3990; re-measure before placing **[also: operator ruling 07:47Z on #3951, "proceed with (a)", kept it as a permanent RED-MODEL cell; the later F9 oracle on the official template refutes that model-defect premise]** | Qwen3.5-0.8B-IQ4_XS never closes its think block while Q4_K_M does: CUDA IQ4 GEMV or real quant damage? |
| #3957 | 0.69.1 (in flight) | the 0.69.1 gate-hardening row (F1-F10); receipted and folded -- close at the tag | 0.69.1 gate hardening (no GPU): sha-bound receipts, missing-field=FAIL, verb output oracles, CRUX judge streng |
| #3961 | 0.69.1 (in flight) | folded on fix/3957-gate-hardening (aprender-6c); verify and close | golden thinking-ON leg discards its backend (_used_gpu) and passes an empty <think></think> as thinking |
| #3962 | 0.69.1 (in flight) | the 0.69.1 CRUX producer row; being folded for the freeze sweep -- close at the tag | 0.69.1 BLOCKER: CRUX producer — per-verb prompt sets (incl. hard prompts) + code verb + hf/vllm quorum rows |
| #3965 | 0.69.1 (in flight) | folded into release/0.69.1-batch-2 (d8/qa-skip-not-pass-3965); verify and close | apr qa --json reports skipped gates as passed:true — a skip reads as a pass to every JSON consumer |
| #3990 | 0.69.1 (in flight) | worked in the current train (aprender-f5, fold/3990-template-chain); closes at the 0.69.1 tag | Qwen2.5 chat template drops the default system prompt (26 vs llama.cpp's 47 tokens) — apr 1/18 vs llama.cpp 10 |
| #3993 | 0.69.1 (in flight) | worked in the current train (aprender-19, the #3990 fold chain); closes at the 0.69.1 tag or carries to 0.70 | SPM encode does not split control tokens: TinyLlama '</s>' in a rendered chat prompt tokenizes as '.</' 's' '> |
| #4004 | 0.69.1 (in flight) | #4004's RED-MODEL claim is judged by the freeze sweep (F9 wrong_answer); resolve at the tag, else 0.71 | Qwen3.5-0.8B-UD-IQ2_XXS fails '2+2' on BOTH apr and llama.cpp (official template) — RED-MODEL candidate pendin |
| #4005 | 0.69.1 (in flight) | folded into release/0.69.1-batch-2 (aprender-6c); verify and close at the tag | 0.69.1: prove non-ASCII tokenization equals llama-tokenize on every tokenizer path apr ships (#3726 re-verify, |
| #4007 | 0.69.1 (in flight) | fixed in fold/3990-template-chain (baef5e961 'fix(4007)'); verify and close at the tag | apr serve picks the chat template from the CLIENT's model string — an unknown name gets a plain prompt and the |
| #3832 | verify-close | superseded by #3957 F6: no UNJUDGED, a split is RED -- verify and close | CRUX judge floor (#3739): a cell judged on fewer than two engines is UNJUDGED, and the receipt states per cell |
| #3886 | verify-close | serve verdict folded into the row's green (#3886) -- verify and close | model_ladder.sh: serve_rc is a dead variable — every serve route can 503/500 and the row is still green |
| #3897 | verify-close | producer/judge asymmetry addressed (#3897, #3902 verb_ok) -- verify and close | model_ladder receipt: per-row `green` is not the gate's verdict — 3 rows marked green that check_model_ladder  |
| #3898 | verify-close | check_model_ladder.sh reads qa_rc/gates_failed (#3898 block in why_of) -- verify and close | RELEASE GATE HOLE: check_model_ladder ignores qa_rc / gates_failed — a Q4_K row with `apr qa` exit 5 is judged |
| #3921 | verify-close | judge_reply / output_bad now judge chat/code/serve output (#3921, #3957 F4a) -- verify and close | P0: chat/code judged by rc and serve by http — a verb emitting 'zombie zombie zombie' is recorded as working,  |
| #3943 | verify-close | teardown 'undetermined' is RED in the judge (#3943) -- verify and close | ladder_serve_teardown reads 'not answering /health' as 'server gone', so a still-loading server is reported te |
| #3770 | close? | a fixture on release/0.69.1-batch-1, a branch that has shipped; review for close | release/0.69.1-batch-1: check_release_bump_pr_body.sh fixture ladder receipts are the pre-#3712 shape — the re |
| #3863 | close? | technical-debt epic: superseded by the debt ratchet #3997; re-home open children | EPIC: technical debt — gates that cannot fail, and the coverage gap |
| #3900 | close? | 0.69.1 deferral decision: answered by the operator's 'no defer' doctrine (2026-09-23) | DECISION NEEDED (published for refutation): two 0.69.1 deferral questions — a gate's inspection limit vs a mod |
| #3950 | DECIDE 0.70.0/0.71.0 | IQ2_XXS CUDA GEMV: same family as #3953/#3960/#3963, which #3998 and #3994 both name -- operator picks one | IQ2_XXS (ggml 16) has no CUDA GEMV — 0.69.1 release blocker |
| #3953 | DECIDE 0.70.0/0.71.0 | named must-carry in #3998 (0.70.0) and #3994 (0.71.0) -- operator picks one | IQ2_S (ggml 22) has no CUDA GEMV — 0.69.1 release blocker |
| #3960 | DECIDE 0.70.0/0.71.0 | named must-carry in #3998 (0.70.0) and #3994 (0.71.0) -- operator picks one | Q2_K (ggml 10) has no CUDA GEMV — 0.69.1 release blocker |
| #3963 | DECIDE 0.70.0/0.71.0 | named must-carry in #3998 (0.70.0) and #3994 (0.71.0) -- operator picks one | IQ3_XXS (ggml 18) has no CUDA GEMV — 0.69.1 release blocker |
| #3987 | DECIDE 0.70.0/0.71.0/0.72.0 | qwen3moe verbs: named must-carry in #3998, #3994 and #4000 -- operator picks one **[operator ruling 11:49Z on #3987: "so keep it in", MoE stays in 0.69.1 and the milestone is restored to none. So a 0.69.1 in-flight candidate; this DECIDE applies only if it carries past the tag]** | qwen3moe CUDA reaches run+qa only: chat rc 8, serve 501/500, code rc 1 — serve loads no mapped model and would |
| #3978 | DECIDE 0.71.0/0.72.0 | named must-carry in #3994 (0.71.0) and #4000 (0.72.0) -- operator picks one | apr code hardcodes 'apr serve --gpu' (no CPU lane, no --max-tokens/thinking flag) and picks a colliding port 1 |
| #3979 | DECIDE 0.71.0/0.72.0 | named must-carry in #3994 (0.71.0) and #4000 (0.72.0) -- operator picks one | apr serve: APR-CPU fallback and safetensors routers have no GET / route index and SSE ends without finish_reas |
| #4008 | DECIDE (proposed 0.71.0) | goes with #3977 (qwen35moe), which is in 0.71.0 per the operator's "MOE goes in .71" on #3994 (2026-09-23 10:43Z); left untouched pending confirmation | arch constraints: qwen35moe (Qwen3.5-35B-A3B) has is_moe=false; replace #3992's substring MoE predicate with a |
| #3493 | 0.70.0 | roadmap aggregate as the one writer: merge-conflict churn (#3998) | roadmap-aggregate: pmat roadmap aggregate is the one writer of roadmap.yaml; roadmap_fragments.py aggregate be |
| #3646 | 0.70.0 | pr-review count gate red on main: review gate (#3998) | check_pr_review_counts.sh is RED on main (6 rows disagree) and check_pr_review_receipt.sh has no caller — both |
| #3652 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | ci_resolve_dirty.sh: `dirty-files=0` reads as "nothing to do" when it means "the driver resolves it" |
| #3722 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | release-readiness waivers need a surface only the operator holds — a gh comment by noahgift is forgeable by ev |
| #3765 | 0.70.0 | flaky test: fail-fast CI (#3998) | flake: detect_ollama_model_file_size_heuristic_tiny fails when NamedTempFile's random name ends in a size mark |
| #3778 | 0.70.0 | llamafile + HF drivers: the sweep's oracle set (#3998) | CRUX engine drivers for #3739: llamafile and HF transformers — probe/gen/tok/tmpl/greedy rows to row contract  |
| #3805 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | check_multiplatform_dogfood demands a cuda parity lane from the crates.io binary, which has no cuda feature —  |
| #3809 | 0.70.0 | 5 dark serve test files: dark test targets into CI (#3998) | 5 test files under apr-cli/serve/ (124 #[test] fns, incl. the apr-serve-v1/http-api-v1 FALSIFY contract tests) |
| #3810 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | CI never runs 1,256 of aprender-serve's cuda-only lib tests: cuda-unit filters `gguf::cuda::` on the premise n |
| #3857 | 0.70.0 | include_str! files missing from the published crate: a publish-gate packaging defect, Fast Train (#3998). The earlier 'pillar A coverage' reason read 'coverage' in the title as line coverage | CB-510 has no coverage for `include_str!`: one guard greps the wrong directive, the other checks git instead o |
| #3879 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | gpu-q does not pin the binary it runs — the GPU has two entrances and only one is guarded |
| #3902 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | No gate runs aprender-serve under --features cuda: 5 tests red there, green in CI, and the tests are the wrong |
| #3904 | 0.70.0 | truncation-slice surface, with #3916: gate hygiene (#3998) | #3872 removed the four slices someone FOUND; nobody enumerated the surface — here it is, and the 5th is in the |
| #3909 | 0.70.0 | gpu-q cannot see a lock-bypassing process: lock hygiene (#3998) | gpu-q cannot see a process that never took the lock: Ollama held 1328 MiB and a cuda suite returned 34 failure |
| #3912 | 0.70.0 | pr-review skill resolves the wrong subject: review gate (#3998) | The pr-review skill resolves its subject from the session cwd, so it can emit a valid signed receipt attesting |
| #3916 | 0.70.0 | truncation-scan surface: gate hygiene (#3998) | #3904 enumerated the surface I could SEE: its scan is Python-shaped, and Rust truncates with .take(N) |
| #3917 | 0.70.0 | a test green only on cache-polluted boxes: hermetic tests (#3998) | test_find_qwen_tokenizer_nonexistent_path passes only on cache-polluted boxes — the green is the wrong answer |
| #3952 | 0.70.0 | vLLM as a comparator: the sharded CRUX sweep's oracle set (#3998) | CRUX: add vLLM as a pre-release comparator engine (plugin driver; lambda x86/4090 + gx10 aarch64/GB10 installa |
| #3956 | 0.70.0 | cuda-feature tests red on the release base: dark test targets (#3998) | cuda-feature tests red on release base: usage_finish_3718 fails 5/5 (ignore_eos 501); gpu_cpu_trace_compare.rs |
| #3964 | 0.70.0 | Ollama bypasses the GPU lock mid-measurement: lock hygiene (#3998) | Ollama daemon loads onto the GPU mid-measurement, bypassing /tmp/apr-gpu.lock — every 'card clear' device rece |
| #3966 | 0.70.0 | gpu-q waiter spins forever: the fleet lock (#3998) | gpu-q: a waiter whose queue ticket disappears spins forever (34h measured on gx10) and ignores SIGTERM |
| #3974 | 0.70.0 | named must-carry in epic #3998 | Jidoka audit: dogfood.sh WARN/SKIP/REPORT/MANUAL verdicts exit 0 on the release surface; qa gpu_speedup swallo |
| #3982 | 0.70.0 | named must-carry in epic #3998 | cuda_combinatorial_coverage: test_tqa023 asserts GGML types 0/1 unmapped (stale since GH-374/#3477); target ne |
| #3984 | 0.70.0 | named must-carry in epic #3998 **[duplicate pair with #3989, pending a ruling; see Corrections]** | tests/driver_cuda_gguf.rs does not compile (GGUFConfig/OwnedQuantizedLayer field drift); target never built in |
| #3985 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | kernel-fusion-v1 call_site guard checks only that 'call_site:' occurs — 4 sites pointed into a dead file, 2 st |
| #3986 | 0.70.0 | GPU lock scoped to GPU work is 0.70's headline (#3998); #3994 lists it only as a dependency | GPU lock efficiency: a 25-min cuda suite held the fleet lock at ~97% idle GPU — scope the lock to GPU work (po |
| #3989 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) **[duplicate pair with #3984, pending a ruling; see Corrections]** | driver_cuda_gguf.rs does not compile under --features cuda (10 missing-field errors); cuda-only integration ta |
| #3996 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | aprender-train: ~60 clippy errors under --features cuda hide every downstream crate's lint (use --no-deps) |
| #3997 | DECIDE (recommend 0.70.0) | the ratchet epic is named by #3998, #4000, #3999, #4001 and #4002; recommend 0.70.0 as its first slice -- operator picks. Moved in error and REVERTED to no milestone after the #4024 quorum | EPIC: debt ratchet 0.70→0.74 — 80% of tech debt in 5 equal slices (coverage→95% w/ yoga CUDA shards, pv deepes |
| #4015 | 0.70.0 | the serve health wait counts lock-queue time: named sweep hygiene in #3998 | ladder serve wait: #3943's stall rule counts time blocked in flock as no progress, a false 'stalled' serve RED |
| #4018 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) **[CORRECTED 0.70.0 → 0.71.0, see Corrections]** | apr run -v PANICS on a non-ASCII prompt: formatted_prompt log slices a str at byte 200 (not a char boundary) |
| #4020 | 0.70.0 | release/CI/gate speed and reliability: Fast Train (#3998) | falsify_2384_run_apr_executes_the_resolved_binary is flaky: ETXTBSY (Text file busy) under parallel lib tests |
| #3532 | 0.71.0 | 23 GB peak VRAM and a golden failure for a 5 GB model: certified-cell correctness (#3994) | qwen3-8b Q4_K_M on CUDA sm_89 (24 GB): 23.1 GB peak VRAM for a 5 GB model, apr qa golden_output Empty output — |
| #3648 | 0.71.0 | debt ratchet pillars B/C (#3997: pv at the deepest level, ontology merge), slice 2 | pv shapes: the LONE-contract venue declines with no report — by_shape and declines exist only on the directory |
| #3675 | 0.71.0 | the declared unknown token is ignored: tokenizer correctness (#3994) | Thread the DECLARED unknown token (tokenizer.ggml.unknown_token_id / tokenizer.json unk_token) through every t |
| #3677 | 0.71.0 | BPE Latin-1 glyph mis-encoding: tokenizer correctness (#3994, with #4005) | BPETokenizer greedy encode matches Latin-1 characters as byte-level GLYPH tokens: "é" encodes as byte 0xE9 and |
| #3746 | 0.71.0 | PTX builder emits an invalid instruction: GPU kernel correctness (#3994) | aprender-gpu PTX builder: shfl_xor_f32 emits 'shflbfly' — ptxas rejects it; no emitter arm for PtxOp::ShflBfly |
| #3752 | 0.71.0 | model/backend correctness or the certified matrix: Don't Leave Behind (#3994) **[consistency re-check, see Decisions]** | #3745 S3: hand-list guard — refuse literal apr verb/flag enumerations in release surfaces (vocabulary from apr |
| #3759 | 0.71.0 | GPU RMSNorm uses the wrong epsilon: GPU correctness (#3994) | GPU RMSNorm runs with ε = 1e-5, not the model's 1e-6: the kernel cache key omits ε and preload bakes 1e-5 — sp |
| #3763 | 0.71.0 | Q4_K universe from the tensor header: the ladder universe (#3994, with #3846) | #3712 row A2: the Q4_K universe is read from each file's tensor header (apr tensors --json), never a *Q4_K* fi |
| #3783 | 0.71.0 | the ladder cells producer (#3712 row B2): the certified matrix (#3994) | #3712 row B2: the cells producer — a resident runner over S2's derived cells (cell_id rows, output_sha256, wal |
| #3799 | 0.71.0 | a hand-typed command list: derive from the surface -- debt ratchet pillar B (#3997), slice 2 | validate.rs extract_model_paths is a hand-typed list of model-loading commands — derive it from the S1 ModelPa |
| #3811 | 0.71.0 | CPU Q8_K activations change greedy answers: CPU backend correctness (#3994) | The CPU default (Q8_K activations) changes the greedy answer on 4 pairs on lambda and 7 on gx10, different set |
| #3847 | 0.71.0 | debt ratchet pillars B/C (#3997: pv at the deepest level, ontology merge), slice 2 | ONT-4c: the first non-code contracts — README.md, CLAUDE.md, one .apr model file, one CSV — with their extract |
| #3856 | 0.71.0 | debt ratchet pillars B/C (#3997: pv at the deepest level, ontology merge), slice 2 | Capability facts as a contract: apr-model-capability-v1 + SHACL + exposure on verb/http/mcp (operator keystone |
| #3860 | 0.71.0 | debt ratchet pillars B/C (#3997: pv at the deepest level, ontology merge), slice 2 | 477 YAMLs live in crate-local `contracts/` dirs that `pv lint` has never read — and a sample says some would f |
| #3862 | 0.71.0 | debt ratchet pillars B/C (#3997: pv at the deepest level, ontology merge), slice 2 | `contracts/apr-cli-commands-v1.yaml`'s top-level command list is compared to nothing — the de-facto registry i |
| #3870 | 0.71.0 | model/backend correctness or the certified matrix: Don't Leave Behind (#3994) **[CLOSED as a duplicate of #3869, see Corrections]** | `IQ4_NL` (ggml type 20) has no CPU dequantizer — two inventory models are majority-IQ4_NL and cannot generate  |
| #3876 | 0.71.0 | model/backend correctness or the certified matrix: Don't Leave Behind (#3994) | ladder receipt: bytes describes the symlink, sha256 describes the model — one row, two objects |
| #3878 | 0.71.0 | model/backend correctness or the certified matrix: Don't Leave Behind (#3994) | ladder receipt: a dangling symlink records "sha256": "" — an identity claim of nothing |
| #3882 | 0.71.0 | model/backend correctness or the certified matrix: Don't Leave Behind (#3994) | golden gate: the thinking-ON leg sends a prompt in NEITHER mode the model declares — it deletes the think bloc |
| #3893 | 0.71.0 | quant label wrong for .apr: it conceals the #3885 dispatch defect, which keeps it in 0.71 (correctness). #4006, the telemetry-only label bug, went to 0.72 in the Corrections | two surfaces report quant=Q4_K for .apr models that contain no Q4_K tensors (f16 and BF16 measured) |
| #3895 | 0.71.0 | GPU L2 diverges 91.92% from CPU: backend correctness (#3994) | gpu_cpu_trace_compare compiles again and fails: GPU L2 diverged 91.92% from CPU on a synthetic fixture (cause  |
| #3901 | 0.71.0 | model/backend correctness or the certified matrix: Don't Leave Behind (#3994) | a ladder row red only on serve prints reason 'unknown' — #3886 moved the verdict and left the explanation behi |
| #3908 | 0.71.0 | model/backend correctness or the certified matrix: Don't Leave Behind (#3994) | Every BF16 SafeTensors import produces a .apr that cannot run on the GPU: wgpu has no type-30 dequant and the  |
| #3910 | 0.71.0 | .apr records nothing about what built it: the .apr->source chain (F8) needs provenance (#3994) | An .apr records nothing about what built it: 'stale artifact or live defect' is unanswerable from the file, an |
| #3911 | 0.71.0 | apr chat ignores the .apr's own tokenizer: .apr correctness (#3994) | apr chat never consults the tokenizer the .apr carries — four filesystem locations, none of them the model |
| #3913 | 0.71.0 | apr diff exits 0 on a 197-tensor corruption: the .apr chain (F8) depends on it failing (#3994) | apr diff measured a 197-of-338-tensor corruption and exited 0: --values cannot fail, --threshold is silently i |
| #3919 | 0.71.0 | model/backend correctness or the certified matrix: Don't Leave Behind (#3994) | The thinking-ON leg exists twice and has now drifted twice: the hybrid copy takes neither #3907's budget resol |
| #3920 | 0.71.0 | embedded .apr tokenizer splits ChatML control tokens: .apr correctness (#3994) | #3911's embedded tokenizer splits every ChatML control token into six pieces — 32 ids where the reference give |
| #3922 | 0.71.0 | model/backend correctness or the certified matrix: Don't Leave Behind (#3994) | RULE A VIOLATION: a Q4_K .apr produces garbage on CUDA on BOTH hosts while answering correctly on CPU — rc=0,  |
| #3928 | 0.71.0 | model/backend correctness or the certified matrix: Don't Leave Behind (#3994) | apr run is the last verb whose generated text nothing judges — and qa's golden leg runs once per rung, not onc |
| #3931 | 0.71.0 | model/backend correctness or the certified matrix: Don't Leave Behind (#3994) | golden_output's GPU leg judges GGUF only, so every .apr rung gets a CPU-only verdict that reports as passed |
| #3935 | 0.71.0 | debt ratchet pillars B/C (#3997: pv at the deepest level, ontology merge), slice 2 **[CORRECTED 0.71.0 → 0.73.0, see Corrections]** | README.md:162 advertises a "100-pt structural audit" that #1870 replaced — the tool scores only the 5 of 26 ch |
| #3936 | 0.71.0 | model/backend correctness or the certified matrix: Don't Leave Behind (#3994) | the ladder's serve /health window is a fixed 90s, so the 27B is red for load time while passing capability_mat |
| #3940 | 0.71.0 | model/backend correctness or the certified matrix: Don't Leave Behind (#3994) | check_model_ladder.sh binds the model file's sha and the release version, but never the binary that produced t |
| #3942 | 0.71.0 | model/backend correctness or the certified matrix: Don't Leave Behind (#3994) | Decide: should golden_output judge .apr on the GPU, or formally delegate that to chat/serve? (the gap #3931 ma |
| #3947 | 0.71.0 | debt ratchet pillars B/C (#3997: pv at the deepest level, ontology merge), slice 2 | aprender-core has no IQ dequantizer, so tensor_contract cannot inspect three release-blocking models — port fr |
| #3968 | 0.71.0 | debt ratchet pillars B/C (#3997: pv at the deepest level, ontology merge), slice 2 | GPU whitelist admits a quant TYPE on one model's shapes — admission must key on per-shape device conformance ( |
| #3971 | 0.71.0 | model/backend correctness or the certified matrix: Don't Leave Behind (#3994) **[REASON WRONG, flagged for operator re-check: the defect is a corrupted comparator HF cache blob plus a judge that scored it GREEN, i.e. the sweep's oracle set (0.70, like #3952/#3778), not apr correctness; not re-moved without a ruling]** | HF cache blob for Qwen2.5-Coder-1.5B-Instruct is a rewritten float32 file under upstream's sha256 name — hf+vL |
| #3972 | 0.71.0 | debt ratchet pillars B/C (#3997: pv at the deepest level, ontology merge), slice 2 | ONT-4c5 (infra#921): aprender capability-cell extractor + arming rule (required Unknown{NotRun} meets to RED) |
| #3973 | 0.71.0 | named must-carry in epic #3994 | F2 GPU validation fails OPEN: returns true ('assume GPU is fine') when the CPU reference or probe is unavailab |
| #3975 | 0.71.0 | named must-carry in epic #3994 | GPU/CPU f32 APR forward diverges at layer-0 QKV (std 4.44 vs 0.30); gpu_cpu_trace_compare fails 91.9% L2 once  |
| #3976 | 0.71.0 | named must-carry in epic #3994 | Q4_K GEMV kernels: FusedKVHwDp4aQ4KGemv generate_ptx returns "" but is launched (q4k_mwv_gemv.rs:602); Dp4aSIM |
| #3992 | 0.71.0 | model/backend correctness or the certified matrix: Don't Leave Behind (#3994) | 17 more files build the dense GGUF CUDA model with no qwen3moe route — audit each (derived from #3987's guard) |
| #3995 | 0.71.0 | the wgpu build does not compile: the wgpu backend row (#3994) **[CLOSED as a duplicate of #3779, see Corrections]** | apr-cli --features wgpu does not compile on the release branch; the wgpu serve router is unbuildable |
| #4006 | 0.71.0 | model/backend correctness or the certified matrix: Don't Leave Behind (#3994) **[CORRECTED 0.71.0 → 0.72.0, see Corrections]** | apr run labels a mixed UD-IQ2_XXS model 'quant=Q5_K' — it prints the tied lm_head's qtype, not the body's |
| #4019 | 0.71.0 | model/backend correctness or the certified matrix: Don't Leave Behind (#3994) | apr thinking-ON closes </think> far less often than llama.cpp on the official template (0.8B Q4_K_M: 1/13 vs 5 |
| #3734 | 0.72.0 | replace the grammar module with TokenConstraint: structured output for agents (#4000) | Replace or remove crates/aprender-serve/src/grammar once #3568's TokenConstraint lands — an unwired, char-leve |
| #3735 | 0.72.0 | schema-constrained decoding with thinking: structured output for agents (#4000) | Schema-constrained decoding with a thinking template: constrain only after </think> (0.70; 0.69.1 refuses Sche |
| #3889 | 0.72.0 | serve/agent/telemetry surface: Agent Ready (#4000) | apr serve reports HTTP 200 for a forced-accelerator request that silently fell back to CPU — it has no after_g |
| #3925 | 0.72.0 | serve/agent/telemetry surface: Agent Ready (#4000) **[CORRECTED 0.72.0 → 0.71.0, see Corrections]** | the ladder judges the terminal, not the model: apr chat's own separator reads as gibberish and reddens every r |
| #3927 | 0.72.0 | serve/agent/telemetry surface: Agent Ready (#4000) | apr chat --json emits the backend, not the reply — the transcript is the only place the model's text exists |
| #3954 | 0.72.0 | aprender-mcp generic server: the MCP protocol row of Agent Ready (#4000, #2794) | aprender-mcp: a generic McpServer over a caller-supplied ToolSet, apr tools behind a default feature (blocks p |
| #3955 | 0.72.0 | chat banner claims an unverified kernel: honest telemetry (#4000) | False provenance: chat banner claims 'fused Q4K, F2-validated' on BF16 with F2 SKIPPED; chat envelope requeste |
| #3980 | 0.72.0 | batch serving F2 probe: serve surface (#4000) | Batch serving F2 GPU validation has never run (empty probe) — needs a representative probe at batch-server ini |
| #3981 | 0.72.0 | named must-carry in epic #4000 | apr run --format json tok_per_sec counts model load + F2 validation as inference (0.2 tok/s reported vs 38.6 m |
| #3991 | 0.72.0 | serve/agent/telemetry surface: Agent Ready (#4000) **[consistency re-check, see Decisions]** | apr serve advertises dead GGUF generation routes: /v1/batch/completions and /stream\|/realize/generate 503; /ge |
| #3589 | 0.73.0 | debt ratchet pillar D (#3997: backlog, docs, packaging), slice 4 | aprender-viz: no vector SVG from the grammar path — BuiltGGPlot renders only to a Framebuffer, and src/output/ |
| #3590 | 0.73.0 | debt ratchet pillar D (#3997: backlog, docs, packaging), slice 4 | aprender-viz: GGPlot::title/xlab/ylab are accepted and silently discarded — never drawn, and #[allow(dead_code |
| #3691 | 0.73.0 | misleading 'Invalid APR format' errors: debt ratchet pillar D (#3997), slice 4 | apr hex / apr trace still say "Invalid APR format" for GGUF and SafeTensors failures — 6 more InvalidFormat si |
| #3727 | 0.73.0 | performance/decode-prefill path: llama.cpp Parity (#3999) | FP8 batched GEMM multiplies a STALE activation on mixed-quant models: the PMAT-084 cache key (ptr, count) alia |
| #3728 | 0.73.0 | performance/decode-prefill path: llama.cpp Parity (#3999) | FP8 prefill GEMM overflows its FP16 output: D = true × 448/act_absmax exceeds 65,504 on qwen2.5-coder-7b (cosi |
| #3729 | 0.73.0 | apr parity --per-op is blind to batched prefill: prefill parity (#3999) | apr parity --per-op is blind to batched prefill (runs the serial path) and scores an undumped GPU stage as cos |
| #3798 | 0.73.0 | performance/decode-prefill path: llama.cpp Parity (#3999) | Internal docs quote 265 unreceipted competitor-throughput figures (docs/specifications, crates/*/CLAUDE.md, do |
| #3800 | 0.73.0 | performance/decode-prefill path: llama.cpp Parity (#3999) | book ch22 example: its 'bootstrap statistics' are 10 literal samples (mean ~273.6) that match neither cited re |
| #3853 | 0.73.0 | simd_bf16_matmul dead code: debt ratchet (#3997), slice 4 | `simd_bf16_matmul` is defined twice and called by nothing — a candidate for the unwired-capability gate (PMAT- |
| #3861 | 0.73.0 | gpu_speedup ratio floor manufactured by a shared-host CPU leg: perf measurement (#3999) | gpu_speedup's floor is a RATIO, so a shared-host CPU leg in the denominator can manufacture or erase a pass |
| #3906 | 0.73.0 | 347 duplicated type names: debt ratchet (#3997), slice 4 | 347 public type names are defined twice or more WITHIN one crate — two were found by hand tonight as real sile |
| #3894 | 0.74.0 | quant/tensor/arch dispatch consolidation: Any Model core (#4001) **[CLOSED as a duplicate of #3891, see Corrections]** | The GPU-supported quant list is hardcoded in prose in a user-facing refusal — a third source of truth, already |
| #3896 | 0.74.0 | two f32_matmul with swapped args: consolidation (#4001) | Two f32_matmul functions with swapped argument orders and identical types — a wrong import is a silent transpo |
| #3918 | 0.74.0 | the name-keyed format table: #3990 routes around it when a template exists; removing it is Any Model (#4001) | detect_format_from_name's ordered substring table has produced three defects — one a correct fix that never ex |
| #3945 | 0.74.0 | qtype x op x backend obligations derived from dispatch: one quant dispatch (#4001) | Enumerated conformance table: derive (qtype × op × backend) proof obligations from the dispatch registry — the |
| #3959 | 0.74.0 | dedupe IQ decoders onto one: dispatch consolidation (#4001) | Dedupe IQ decoders: aprender-serve onto trueno_quant::iq (one decoder, one oracle) |
| #3803 | 0.75.0 | fine-tune/train surface: CRUX declarative fine-tune/distill (#4002) | entrenar's .apr export embeds a tokenizer that drops array-form merges and the added tokens (two hand-rolled t |
| #3585 | none (pinned) | the aprender-queue coordination thread: a standing process issue, not release work | aprender-queue: register here before opening an aprender PR (one PR in CI at a time) |
