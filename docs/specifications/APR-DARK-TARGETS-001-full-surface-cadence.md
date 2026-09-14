# APR-DARK-TARGETS-001 — the full test surface: what it costs, what it catches, when it runs

**Status:** proposed · **Opened:** 2026-09-14 · **Refs:** #3239, #3242, #3248, #3249, #3245
**Supersedes nothing.** BSE-17 (the quick tier) is correct and this specification does not change it.

---

## 1. The measurement

Two runs, one hour apart, on 2026-09-14.

| | tests | binaries | run time | where |
|---|---|---:|---:|---|
| merge-queue full tier (#2838, run `34809681888`) | 82,217 | **73** | 2,226 s (37 min) | `intel-clean-room-14` |
| whole `--workspace --lib --tests` | **98,621** | **727** | 1,141 s (19 min) | lambda-vector, 48 cores |

```bash
cargo nextest run --workspace --lib --tests --no-fail-fast \
  --exclude aprender-gpu --exclude aprender-cuda-edge --exclude aprender-compute
```

**The gap is not "20% more tests" — it is 10× the binaries.** 654 test binaries have never been
*linked* in CI, let alone run. A binary that never links cannot fail, cannot flake, and cannot
appear in a report. That is why this was invisible rather than noisy, and it is the single most
important sentence in this document.

Nothing else covers them either. `make coverage` — the only other whole-workspace run — is also
`--lib` (`Makefile`: `cargo llvm-cov test --workspace --exclude aprender-gpu --lib`), and
`nightly.yml` is a binary-release workflow that runs no tests at all.

### 1.1 Cost, isolated — and the link cost is NOT the problem

Same box, same cache state, cold target dir (`scratchpad/testdelta.sh`):

| phase | wall | crates |
|---|---:|---:|
| `cargo build --workspace --lib` (what CI already pays) | **76 s** | 715 |
| `+ --tests` incrementally (the delta) | **145 s** | 228 |
| total | **221 s** | 808 binaries, 39 GB |

**This falsifies the obvious guess, including the one this document made in its first draft.**
650 extra test binaries were assumed to be link-bound and expensive. They are not: two and a half
minutes. The cost of the full surface is almost entirely the *run*, and the run is embarrassingly
parallel.

### 1.2 What that means on a clean-room runner

The two boxes are not comparable directly — lambda-vector is 48 cores, and an
`intel-clean-room-*` runner is one of sixteen sharing a 32-core box. Calibrate on measured
throughput instead:

| | tests/s |
|---|---:|
| lambda-vector | 86.4 |
| intel-clean-room-14 | 36.9 |
| ratio | **2.34×** |

Extrapolated (and it is an extrapolation, §7.1): the full surface on a clean-room runner is
**≈ 45 min run + ≈ 6 min extra build ≈ 51 min**, against the **37 min** the queue's full tier
costs today. **+14 min, ~+35%.**

---

## 2. What the dark surface actually held

37 failures, 5 binaries:

| binary | n | defect |
|---|---:|---|
| `aprender-present-yaml::examples` | 27 | `../../examples/…` resolves to the repo root; the assets are in the **sibling** crate. Dark since APR-MONO |
| `aprender-present-yaml::prs_examples` | 7 | anchored on `CARGO_MANIFEST_DIR`, then walked to the same wrong root |
| `apr-format::golden_fixtures` | 1 | the golden predates #2254's `skip_serializing_if`; 1092 vs 516 bytes |
| `aprender-mcp::falsify_mcp_008` | 1 | the in-crate contract copy ≠ the workspace-root copy |
| `aprender-cgp::falsify` `doctor_speed_real` | 1 | asserts a runner capability — see §5 |

**Not one is a compute or correctness regression.** Every one is a *global invariant*: a golden
oracle, a path to another crate's assets, a file that must equal another file.

**That is structural, not luck.** BSE-17 selects by **changed crate**. None of these lives in
anybody's diff, so no selector can reach them. The corollary matters as much as the finding:

> A PR touching crate X **does** build and run X's integration targets. Change-proximate
> coverage exists and is proportional to the change. The hole is exactly the class that no diff
> can point at.

### 2.1 The argument is `golden_v2`, not the count

A golden test exists *only* to catch silent format drift. This one missed a real format change
for two months. #2254 happened to be deliberate and correct; had it been wrong, nothing in this
repository would have said so.

The risk is not "tests are red". It is **"the alarm was disconnected, and we found out by
accident."** A dark golden is strictly worse than no golden, because its presence is read as
coverage.

---

## 3. Decision: where it runs

| surface | verdict | why |
|---|---|---|
| **Nightly** | **yes — primary** | ~21 min end-to-end on a 48-core box; no PR or merge latency. |
| **Pre-release (T-1)** | **yes — hard gate** | APR-RELEASE-001 T-1 asserts "green on the cut". With 654 binaries never linked, "green" is a materially weaker claim than it reads. The release is precisely where the whole surface is owed. |
| **Merge queue (full tier)** | **no — but it is close** | ≈ +14 min (+35%) on a 37-min critical path, ×3 parallel groups, against defects that move on a scale of *months*. Queue latency is a measured throughput problem here: main went 8.5 h with no merge on 2026-09-13, and ~2 h on 2026-09-14. 24-hour detection latency is the right trade for this defect class. **Revisit if** queue depth is routinely below `max_entries_to_build` — then the 14 min is spare capacity and the answer flips. |
| **Every PR** | **no** | This is what BSE-17 exists to avoid, and §2 shows the quick tier is not the gap. |
| **PRs touching the SELECTOR** | **yes** | `scripts/ci_test_tier.sh`, `scripts/gate_touched_crates.sh`, `scripts/tree_reader_tests.txt`, and ci.yml's explicit `--test` chain. A change to the selector is exactly when the dark set moves. A `paths:` trigger; fires a few times a month. |

"Nightly and pre-release **only**" is nearly right, and the selector-touch trigger is the one
addition worth making: it is the cheapest possible guard against this specification silently
ceasing to be true.

---

## 4. It must not become `present`

A job that is red for months is a job everyone learns to ignore
(`feedback_present_check_is_a_review_backlog_not_a_broken_gate`). At 37 failures it would be
born red.

**Therefore: start ratcheted, finish hard-failing.**

1. **Phase A — shrink-only ratchet.** Baseline = the measured failing set, keyed by test id.
   A *new* failing id fails the job; a removed one is re-recorded with `--update`. Same shape as
   `unwired_guards_baseline.txt` and `pipe_grep_q_baseline.txt`.
2. **Phase B — hard fail.** When the baseline reaches 0, delete it and the `--update` path.

**Phase B is reachable now, which is why the ratchet is scaffolding rather than a resting place.**
With #3248, #3249 and #3238 landed the count is **1**: `aprender-cgp::falsify
falsify_cgp_061_doctor_speed_real`, another runner-capability assertion (§5). The exit condition
is that one target, not an open-ended backlog.

---

## 5. Precondition: capability gating must be fixed first

Three targets failed in the *quick* tier on 2026-09-14 for a reason that will also make the
nightly permanently red if it is not fixed first: **a test that needs a box capability was gated
by a cargo feature a sibling crate can enable.**

`--tests` builds dev-dependencies, so `aprender-orchestrate`'s
`jugar-probar = { …, features = ["browser"] }` armed `aprender-test-lib`'s browser falsifiers on
a browser-less clean-room runner. Feature unification is a property of the **build graph**;
"default features" is a property of a **package**.

Fixed by giving each a name nothing else asks for — `browser-falsify`, `python-oracle` — with **no
skip introduced**: where the capability feature is on, the test still fails hard without the
capability.

`scripts/check_feature_gated_test_leak.sh` (#3246) measures the remaining surface: **33 edges over
22 (crate, feature) gates**, 46 of the gated targets being `cuda`. It is shrink-only for the same
reason as §4 — most are probably harmless, and failing on all 33 would assert what the tree has
not earned.

`aprender-cgp::falsify doctor_speed_real` is the last known instance and the Phase-B blocker.

---

## 6. Shape of the job

```yaml
name: Full test surface
on:
  schedule: [{ cron: '0 4 * * *' }]          # after the nightly image rebuild
  workflow_dispatch:
  pull_request:
    paths:                                    # §3, the selector-touch trigger
      - 'scripts/ci_test_tier.sh'
      - 'scripts/gate_touched_crates.sh'
      - 'scripts/tree_reader_tests.txt'
      - '.github/workflows/ci.yml'
```

Non-negotiables, each earned by a defect this repository has already paid for:

- **`--no-fail-fast`.** Plain `cargo test` stops at the first failing binary; seven rounds were
  spent on four tests before that was understood (`feedback_nextest_fail_fast_hides_dark_failures`).
- **Classify env-death.** Route the cargo invocation through `scripts/cargo_step.sh` (#3241) so a
  full disk or an un-spawnable `rustc` reads as `ENV`, not as 650 new defects. Both happened on
  2026-09-14.
- **Vacuity floor.** Fewer than ~600 binaries built ⟹ the *selection* is broken, not the tree.
  A sweep that silently runs nothing is the exact failure mode this job exists to end.
- **Not a required check.** It is long and it is not on the merge path.

---

## 7. Open, with owners

### 7.1 The clean-room figure in §1.2 is an EXTRAPOLATION, not a measurement
It scales one box's throughput by a ratio taken from a single pair of runs. The nightly's
`timeout-minutes` must come from real runs of this job (BSE-05: `T = max(15, ceil(1.5·p99),
p99+20)` over its own history), never from §1.2. Until three runs exist, set it generously and
say so in a comment.

The first draft of this document asserted the `--tests` delta was "≫ 20 min" on the strength of a
progress check that was reading a stale PID. It is 145 s. The number was already being measured
when the claim was written; the claim should have waited. Recorded here because the same
temptation will recur every time a build "looks" expensive.

### 7.2 The 329 skipped tests
`nextest` reported 329 skipped in the full sweep vs 131 in the queue's subset. The delta is
unexamined; some are `#[ignore]`, which this repository bans — e.g.
`test_timeout_handling_example` carries `#[ignore] // Timeout handling currently hangs - needs
executor fix`. Each needs a ticket or a removal.

### 7.3 GPU crates stay excluded
`aprender-gpu`, `aprender-cuda-edge` and `aprender-compute` are excluded here exactly as the full
tier excludes them, because feature unification pulls the CUDA driver in under `--workspace`.
Their targets are owed a separate CUDA-runner sweep; `cuda-nightly.yml` is the natural home.

### 7.4 `make coverage` is `--lib` too
The coverage floor (88%) is therefore measured over the same partial surface. Whether the floor
should move to the full surface is a separate decision with its own baseline, and it should not
be bundled with this one.

---

## 8. Falsifiers

Claims here are checkable, and each should be re-run rather than re-read:

| claim | how to falsify |
|---|---|
| 654 binaries never link in CI | `Starting N tests across M binaries` in any merge-queue `workspace-test` log vs the sweep's |
| the quick tier covers change-proximate targets | touch one file in crate X; the tier prints `crates=… X …` and builds X's `--tests` |
| the defects are global-invariant only | read the five: none is reachable from a diff to the crate that owns it |
| Phase B is reachable | after #3248/#3249/#3238, re-run the sweep; expect exactly one failing id |
| the `--tests` delta is cheap | `scratchpad/testdelta.sh`: 76 s for `--lib`, +145 s for `--tests`, 808 binaries |
| the merge-queue cost is ≈ +35% | re-derive §1.2's ratio from any two same-day runs and redo the arithmetic |

---

## 9. What this does not propose

- Changing BSE-17 or the quick tier. §2 is the argument that it is not the gap.
- Adding anything to the merge queue's critical path.
- A permanent ratchet. §4 gives the exit condition and names the one target that blocks it.
