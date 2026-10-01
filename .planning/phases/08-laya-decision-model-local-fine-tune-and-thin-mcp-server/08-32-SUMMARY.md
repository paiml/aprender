---
phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
plan: 32
subsystem: gap-round-regression-and-ci-evidence
tags: [laya-gap-regression, class-A-E, scrub, upstream-merge, draft-pr, ci-evidence, contract-hygiene, gap-closure]
status: complete
outcome: partial
gap_closure: true

# `status: complete` because the plan's work ran to its end. It is NOT a clean pass:
# `outcome: partial` and the `unmet_must_haves` / `gaps` below carry the honest verdict, and any
# reader of this SUMMARY must read them. (`status: halted` was rejected: it would report 08-33,
# which will follow this plan, as blocked, and nothing here was left half-done.)
unmet_must_haves:
  - truth: "CI has executed Phase 8's code at least once: the `workspace-test` job concludes success on the pushed head (VERIFICATION human item 1)"
    state: "NOT MET. workspace-test has not run on any head of this branch. Every workflow run on #4634 is completed/action_required (a fork PR waits for a paiml maintainer to approve it). No job has started."
    blocked_on:
      - "(a) a paiml maintainer approving the workflow runs on the fork PR (the branch owner has pull-only access on paiml/aprender)"
      - "(b) plan 08-33: even once approved, workspace-test would stop at upstream's contract-hygiene gates (fragments 336/338/455) before Phase 8's decide fragments 510/520, so it cannot go green until they pass"
gaps:
  - id: G-08-32-CI
    description: "workspace-test success on the pushed head: unmet; see unmet_must_haves"
    evidence: "08-CI-RUN-EVIDENCE.json (workspace_test.conclusion null)"
    follow_up: "approval upstream, then plan 08-33, then re-observe"
  - id: G-08-32-HYGIENE
    description: "upstream's shrink-only contract-hygiene gates fail on this branch's contracts (5 failures, 389 passed, measured locally)"
    follow_up: "decided ('Repair all (new gap plan)'); planned as 08-33"
  - id: G-08-32-SETFIT-LEGS
    description: "17 Phase 3/4 SetFit CI legs not re-homed into ci/explicit-test-commands.d"
    follow_up: "deferred by owner decision; deferred-items.md D-ITEM-08-32-C"

requires:
  - phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
    provides: "08-19..08-31: the class A-E sweeps, gates, claims ledger and the deployed, verified artifact that laya-gap-regression composes"
provides:
  - "just laya-gap-regression: one recipe re-proving classes A-E and the phase regression (Task 1)"
  - "Branch history scrubbed of account, ECR and instance identifiers (0 hits in a0163010a..HEAD) and pushed"
  - "Upstream main merged in (301 commits, 46 conflicted files resolved and recorded) so draft PR #4634 is MERGEABLE"
  - "08-CI-RUN-EVIDENCE.json: an honest record that CI has NOT executed Phase 8's code, with the measured blocker list"
  - "deferred-items.md entries D-ITEM-08-32-B..H for everything the round left open"
affects: [08-33, phase-8-verification, upstream-pr-split]

actuals:
  tokens: 11158       # chars/4 over the realized diff of the plan's own changes: 68397e1ba..c5b3d89d0 (17199 chars) + 0d24db9c5..30bc2baaa (27435 chars). Excludes the merge's 46 resolutions, upstream's 301 commits and this close-out's docs.
  tasks: 2
  commits: 308        # MEASURED: git rev-list --count 68397e1ba..HEAD at close-out (before this SUMMARY's commit); 301 of them are upstream's, the plan's own are 7 (Task 1, the merge, five follow-ups)
plan_head_before: 68397e1bab8ff4323c2a5bb8b8099c792df344d5   # post-scrub sha of the 08-31 close-out commit (its pre-scrub sha resolves via 08-SCRUB-COMMIT-MAP.tsv)

tech-stack:
  added: []
  patterns:
    - "A gap round ends with one composing recipe over the class sweeps, so a class cannot pass alone while another regressed"
    - "A fork PR is the only way to reach an upstream's self-hosted CI runners; a green claim needs a job that actually ran, and action_required is not a result"

key-files:
  created:
    - .planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-CI-RUN-EVIDENCE.json
    - .planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-SCRUB-COMMIT-MAP.tsv
    - ci/explicit-test-commands.d/510-aprender-decide-ui.cmd
    - ci/explicit-test-commands.d/520-aprender-mcp-decide-e2e-stdio.cmd
  modified:
    - justfile
    - scripts/laya_gates.tsv
    - scripts/laya_claims.tsv
    - Makefile
    - Cargo.lock
    - crates/aprender-core/tests/monorepo_invariants.rs
    - crates/aprender-decide/tests/ui.rs
    - crates/aprender-image/src/lib.rs
    - crates/aprender-mcp-chronos/src/lib.rs
    - .planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/deferred-items.md

key-decisions:
  - "Scrub, then push (owner): history rewritten over a0163010a..HEAD; backup ref backup/pre-scrub-08-32 (local only)"
  - "Yes, draft PR (owner): the plan opened none, but workflow_dispatch is unreachable on a fork; the PR is the only route to the runners"
  - "Merge upstream main in (owner): 301 commits, 46 conflicts; backup ref backup/pre-merge-08-32 (local only)"
  - "Repair all (new gap plan) (owner): the contract-hygiene gates are repaired in 08-33; decided; planned as 08-33"
  - "Defer to deferred-items (owner): the 17 SetFit CI legs are not re-homed; D-ITEM-08-32-C"
  - "Keep one draft for CI (owner): #4634 stays one draft PR purely for CI evidence; splitting into reviewable PRs is planned after phase verification"
  - "pmcp stays at 2.19.3, the version Phase 8's tool boundary is verified on, rather than upstream's 2.21.0"

patterns-established:
  - "Cite SHAs through the commit map: everything written before the scrub is pre-scrub"

requirements-completed: []   # D-11, D-15, D-17 are NOT marked: the plan's CI must-have is unmet

duration: "~12h wall clock across sessions (dominated by the scrub, the merge and gate re-runs; one earlier continuation stalled and was killed by the 600 s watchdog)"
completed: 2026-09-29
---

# Phase 8 Plan 32: Gap-round regression and first CI attempt Summary

**One recipe (`just laya-gap-regression`) proved classes A-E and the phase regression together, and the branch was scrubbed, merged with upstream and opened as draft PR #4634, but CI has NOT executed Phase 8's code: the must-have "workspace-test success on the pushed head" is unmet.**

## Verdict (read this first)

- **Task 1 (tracer): met.** `just laya-gap-regression` printed `PASS CLASS A` .. `PASS CLASS E`, `PASS REGRESSION` and `LAYA GAP REGRESSION OK` on the pre-merge tree (commit `c5b3d89d0`, pre-scrub `7f1cd6c36`).
- **Task 2: NOT met.** The plan required `workspace-test` = success on the pushed head. It has not run. All runs on head `30bc2baaa` (CI 36614147889, mdBook CI 36614148038, Book Contract Enforcement 36614148073) are `action_required`: a fork PR's workflows wait for a paiml maintainer's approval, and the branch owner has pull-only access. `pr-review-quorum` 36614144178 failed with a missing signed review receipt (paiml process, not Phase 8 code).
- **Even after approval it would not go green yet.** Upstream's new shrink-only contract-hygiene gates fail locally on this branch (389 passed, 5 failed). CI's runner stops at the first failing command, so fragments 510 (`aprender-decide --test ui`) and 520 (`aprender-mcp-decide --test e2e_stdio`) would never run. Repair is decided and planned as 08-33.
- The must-have was not weakened, and no claim of CI success is made anywhere.

## Performance

- **Duration:** ~12h wall clock across sessions
- **Completed:** 2026-09-29
- **Tasks:** 2 (Task 1 met; Task 2's outcome unmet)
- **Files modified:** see key-files (the merge itself touched thousands of upstream files)

## Task 1 class table (invariant -> enumeration artifact -> sweep -> result)

Results: pre-merge run `PASS`; on the merged tree classes A-E `PASS` again (apr-format now 104 tests).

| Class | Invariant | Enumeration artifact | Sweep | Result |
|-------|-----------|----------------------|-------|--------|
| A provenance | every manifest leaf, run field and served field is bound to a source or declared report-only | the manifest-leaf bindings table, the run-field bindings table, the served-field source table | `artifact::ladder::every_manifest_leaf_is_bound`, `manifest_bindings_table_matches_manifest_leaves`, `verify::tests::every_run_field_is_bound_or_report_only`, `run_field_bindings_table_matches_fixture_leaves`, `tests::every_served_field_has_a_bound_source`, `probe::` tests | PASS |
| B input bounds | every accepted input dimension has a swept, declared bound | the `untrusted_input_bounds` tables (artifact, request, Lambda) | `artifact::ladder::artifact_bounds_table_is_swept`, `tests::request_bounds_table_is_swept`, `tests::lambda_request_rows_are_swept`, `cargo test -p apr-format`, `models::modernbert` tests | PASS |
| C gate honesty | every gate refuses a vacuous pass and has a must-fail case | `scripts/laya_gates.tsv`, the contract-audit ERE case table | `just laya-gates-selftest`, `make contract-audit-phase8` | PASS |
| D claim honesty | every claim names a test or artifact that exists and runs | `scripts/laya_claims.tsv` (133 rows; 2 rows re-anchored in this plan) | `just laya-claims-check` | PASS |
| E Rust-Python agreement | the gate arithmetic agrees bit for bit in both languages | the shared numeric case table | `verify::tests::gate_numeric_cases_agree_bit_for_bit`, `just laya-train-selftest` (METRICS SELFTEST OK and PYTHON REFUSALS lines) | PASS |

REGRESSION stage: pre-merge PASS. On the merged tree it failed only inside `aprender-contracts-cli` (upstream's contract-hygiene gates and `every_contract_generates_book_page`), which is the 08-33 blocker list below; nothing in Phase 8's crates failed. The real-artifact legs were run separately under the lock on the merged tree:

- `just laya-verify-suite`: `LAYA VERIFY SUITE OK`, 4 legs, ladder 32 blocks, max_abs 3.841e-6
- `just laya-verify`: `deploy_eligible` true, sha `24a44d7e`, shipped_seed 17, argmax 459/459

## Task 2: CI evidence (job table)

`08-CI-RUN-EVIDENCE.json` (schema `laya-ci-run-v1`) holds the full record. The job table is empty by construction: no job started.

| Workflow | Run | Event | Head | Conclusion |
|----------|-----|-------|------|------------|
| CI | 36614147889 | pull_request | 30bc2baaa | action_required |
| mdBook CI | 36614148038 | pull_request | 30bc2baaa | action_required |
| Book Contract Enforcement | 36614148073 | pull_request | 30bc2baaa | action_required |
| pr-review-quorum | 36614144178 | pull_request_target | 30bc2baaa | failure (missing signed review receipt) |
| CI | 36607358541 | pull_request | 05de396d0 (earlier head) | action_required |

`workspace_test: {conclusion: null}`. `event` is `pull_request`, not the plan's `workflow_dispatch`, because the fork has no workflows or runners and ci.yml's jobs need paiml's self-hosted runners, which only a fork PR can reach.

### Predicted blockers (measured locally, not observed in CI)

| # | Gate | Baseline -> measured | Owner of the repair |
|---|------|----------------------|---------------------|
| B1 | `formal:` entries outside the declared vocabulary | 1464 -> 1569 (+105; Phase 8's share 25: decide-apr 6, decide-tool-boundary 6, laya-finetune-gate 9, laya-parity 4; the rest Phases 3-6) | 08-33 |
| B2 | kernel-kind contracts without `metadata.valid_under` | 386 -> 398 (+12) | 08-33 |
| B3 | `pv validate` errors | neon-blis-v1 (`kind:` at top level), spectral-indices-v1 (no `kani_harnesses`) | 08-33 |
| B4 | `the_tracked_repo_graph_is_fresh` | needs regeneration | 08-33 |
| B5 | `every_contract_generates_book_page` | `contracts/binding.yaml` has no `metadata:`; fails on origin/main before any Phase 8 commit | D-ITEM-08-32-A (pre-Phase-8) |

## Owner decisions (verbatim ids)

1. **"Scrub, then push"**: history rewritten with `git filter-repo --replace-text` over `a0163010a..HEAD`. Three categories of identifier (an AWS account id, ECR repository ids, instance ids) removed; 0 hits remain; backup ref `backup/pre-scrub-08-32` = `7f1cd6c36` (local only); `08-SCRUB-COMMIT-MAP.tsv` (213 old->new rows) committed.
2. **"Yes, draft PR"**: #4634 (https://github.com/paiml/aprender/pull/4634), draft, MERGEABLE, BEHIND main. The plan said it would open none; this decision supersedes that line.
3. **"Merge upstream main in"**: merge commit `0d24db9c5` (parents `c5b3d89d0` + upstream/main `2817c6d97`; 301 commits past base `4bbfeb07f`; 46 conflicted files); backup ref `backup/pre-merge-08-32` = `c5b3d89d0` (local only).
4. **"Repair all (new gap plan)"** for the contract-hygiene gates: **decided; planned as 08-33.** Not done here; the 08-33 plan is not written here.
5. **"Defer to deferred-items"** for the 17 SetFit CI legs: D-ITEM-08-32-C, status open (upstream fragments 010/020 cover the setfit lib tests and conformance; the other 17 legs need new `ci/explicit-test-commands.d` fragments; deferring keeps the PR's CI footprint small since upstream CI takes about 90 minutes; owner the Phase 3/4 SetFit owner; re-open on a SetFit regression 010/020 do not catch).
6. **"Keep one draft for CI"**: #4634 stays one draft PR purely to get CI evidence; the user arranges maintainer approval upstream; splitting into reviewable PRs (serving and contracts first, training code for upstream's .71-.75 window per the maintainer's note) is planned after phase verification (D-ITEM-08-32-H).

## Task Commits

1. **Task 1 (tracer): laya-gap-regression** - `c5b3d89d0` (pre-scrub `7f1cd6c36`) (feat)
2. **Task 2: scrub, merge, push, draft PR, evidence**
   - merge `0d24db9c5` (merge upstream main)
   - `5f9191622` style: cargo fmt on aprender-image and aprender-mcp-chronos
   - `519dc4125` fix: keep pmcp at 2.19.3
   - `7749baef0` fix: FALSIFY-INSTALL-001 allowlist for Phase 6's aprender-forecast profile
   - `05de396d0` fix: Makefile PV_BIN -> PV_CARGO_RUN
   - `30bc2baaa` fix: claims ledger anchors Phase 8's CI legs where upstream keeps them

**Plan metadata:** the close-out docs commits (evidence, deferred items, this SUMMARY, STATE/ROADMAP) follow `30bc2baaa`; their shas are in the orchestrator report and in `git log`.

SHAs cited in the earlier SUMMARYs (08-01..08-31) and in evidence files written before the scrub are PRE-SCRUB. Resolve them through `08-SCRUB-COMMIT-MAP.tsv`. The merge commit `0d24db9c5` and everything after it are post-scrub.

## Conflict resolutions (46 files, condensed)

Full text: the merge commit message of `0d24db9c5`.

- **Took upstream (theirs):** `.github/workflows/ci.yml` (whole file: upstream restructured CI into fat jobs driven by `ci/sections.yml`); `contracts/lbfgs-kernel-v1.yaml`; the SetFit/contrastive snapshot files whose blobs are unchanged on this branch since upstream's squashed snapshot PRs (#2618, #2702) (`contrastive-pair-protocol`, `multinomial-head`, `setfit-encoder-conformance` contracts, `aprender-contrastive-data` files, `setfit/classify.rs`, `tolerances_generated.rs`); `crates/aprender-core/src/lib.rs`; `README.md` (counts re-derived afterwards: 92 workspace crates, 1850 contracts, 97 crate dirs); `CLAUDE.md` (Phase 8's Decision-model row and its exception text auto-merged intact); the `.pv` cache deletions.
- **Kept ours:** the SetFit source files superseding upstream's snapshot (`encoder`, `import`, `import_tests`, `mod`, `model_tests`, `tokenizer`); `generated_contracts.rs` tail block; `contracts/aprender/binding.yaml` (ours is a superset with Phase 3-8 bindings); `Makefile` tier3 tail and contract-audit tail (Phase 2-8 targets).
- **Three-way merged, both sides kept:** `contracts/setfit-apr-v1.yaml`, `crates/aprender-core/src/setfit/artifact.rs` (via `git merge-file` against the closest snapshot blob).
- **Combined:** root `Cargo.toml` (upstream's blocks, our now-duplicate keys removed twice; branch-only members kept); `aprender-core`, `aprender-train` and `aprender-mcp` `Cargo.toml` (feature splits honoured); `aprender-mcp/build.rs` and `tools/mod.rs` (upstream's `apr-tools` gating with our predict tool re-applied under it); `apr-cli` `output_verification.rs` (upstream restructure plus our encoder-only QA refusal); `aprender-compute` `blis/parallel.rs` (both sides fixed the same shared-B bug; upstream body, our public helpers kept); `apr-format` `v2/mod.rs` (union of re-exports); `aprender-serve` `router.rs` (our `setfit_routes()` under upstream's `route_table` signature); `.gitignore` (union); `Makefile` `.PHONY` union and contrastive-data recipe blocks (upstream).
- **Re-synced:** 6 conflicted `apr-cli` contract mirrors plus 4 non-conflicted mirrors to byte-identity with the merged root contracts (check_publish_safety.sh check 11: 13 mirrors OK).
- **Compile repairs on auto-merged files:** `aprender-serve` `mod_app_state_new.rs` Default impl (three new upstream `AppState` fields added), `mod_app_state_qwen35.rs` (our `cfg(setfit)` `setfit_model` field added).
- **Path-dep versions:** the branch-only crates' path-dependency requirement moved 0.63.0 -> 0.69.3 to match the workspace; `Cargo.lock` taken from upstream with cargo resolving the branch-only crates (8 non-workspace packages resolve to the same versions as before).

## Deviations from Plan

### Owner-decided departures

**1. [Owner decision] History scrub before the push.** The plan pushed the branch as it was. The owner chose "Scrub, then push" after identifiers of three categories were found in history. Consequence: every SHA cited in pre-scrub SUMMARYs is stale (commit map committed).

**2. [Owner decision] A draft PR was opened; the plan said it opened none.** `workflow_dispatch` is unreachable on a fork (no workflows or runners there; ci.yml's jobs need paiml's self-hosted runners). Consequence: `event` in the evidence is `pull_request`, and CI results depend on a maintainer's approval.

**3. [Owner decision] Upstream main merged in.** The PR was not mergeable and upstream had restructured CI. 301 commits, 46 conflicts.

### Auto-fixed and repair commits

**4. [Rule 3 - Blocking] pmcp pinned back to 2.19.3** (`519dc4125`). Upstream is on 2.21.0. Phase 8's `request_bounds_table_is_swept` tripwire and the tool-boundary contract are verified on 2.19.3. Recorded as D-ITEM-08-32-G; re-verify on 2.21.0 and lift the pin.

**5. [Rule 3 - Blocking] FALSIFY-INSTALL-001 allowlist entry** (`7749baef0`). Phase 6's `profile.dev.package.aprender-forecast` (64 s vs 13 s measured) added to `PROFILE_SPEC_ALLOWLIST`, because the merged install gate rejected a load-bearing profile.

**6. [Rule 3 - Blocking] Makefile `PV_BIN` -> `PV_CARGO_RUN`, 12 uses** (`05de396d0`). Upstream renamed the pv variable; the branch's contract-audit targets called the old name.

**7. [Rule 1 - Bug] `laya_claims.tsv` re-anchor** (`30bc2baaa`). Two rows and the `tests/ui.rs` header named ci.yml lines that no longer exist; re-anchored to `ci/explicit-test-commands.d/510-aprender-decide-ui.cmd` and `520-aprender-mcp-decide-e2e-stdio.cmd`.

**8. [Rule 3 - Blocking] `cargo fmt`** on `aprender-image` and `aprender-mcp-chronos` (`5f9191622`). The merged fmt gate reads them.

### Not done / departures from the plan text

**9. SetFit CI step not re-homed.** The old ci.yml's 19-command step was not carried over; 17 legs are unrun (owner: defer; D-ITEM-08-32-C). `ci.yml` was taken whole from upstream, not edited; the two decide integration lines were re-homed by upstream's documented extension point (fragments 510/520), not by editing a workflow.

**10. Stalled prior continuation.** An earlier continuation of this plan stalled and was killed by the 600 s watchdog; the work was resumed from the committed state and nothing was lost.

**11. Event is `pull_request`, not `workflow_dispatch`** (see decision 2).

**12. The plan's Task 2 verify command cannot pass.** It asserts `workspace_test.conclusion == "success"`; the honest record has `null`. The check was not weakened and the record was not edited to satisfy it.

## Known Stubs

None in code. (`workspace_test.conclusion: null` is a recorded absence, not a stub.)

## Threat Flags

None new. T-08-32-01 (identifiers in pushed files) was mitigated more strongly than planned: the scrub removed three categories from history; the evidence file holds no credential, account id or e-mail. T-08-32-03: no workflow file was edited; the merge took upstream's ci.yml whole.

## Next steps

1. The user arranges a paiml maintainer's approval of the PR's workflow runs (or a later run on the approved head).
2. 08-33 repairs the contract-hygiene gates (B1-B4) so the runner reaches fragments 510/520.
3. Phase 8 verification runs after 08-33. Phase 8 is NOT complete.
4. Pushing this close-out re-triggers the PR workflows (expected; the runs above are for `30bc2baaa`). The close-out commits are docs-only; the exact head the maintainers should approve is the tip named in the orchestrator report.

## Self-Check: PASSED

Verified at close-out: `08-CI-RUN-EVIDENCE.json` parses as JSON; `08-SCRUB-COMMIT-MAP.tsv` exists; `deferred-items.md` carries D-ITEM-08-32-B..H; commits `c5b3d89d0`, `0d24db9c5`, `5f9191622`, `519dc4125`, `7749baef0`, `05de396d0`, `30bc2baaa` exist in `git log`; HEAD == origin/gsd/phase-2-contract-gate == `30bc2baaa` before the close-out commits; PR #4634 isDraft true. This self-check verifies the RECORDS. It does not verify the CI must-have, which is unmet.
