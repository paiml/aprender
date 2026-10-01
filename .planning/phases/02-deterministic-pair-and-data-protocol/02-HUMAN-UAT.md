---
status: partial
phase: 02-deterministic-pair-and-data-protocol
source: [02-VERIFICATION.md]
started: 2026-08-09T08:33:23Z
updated: 2026-08-09T12:15:00Z
---

## Current Test

[awaiting human testing]

## Tests

### 1. Accept or reject the setfit row-byte deviation (FINDING-D1)

Plan 02-06's `must_haves.truths[2]` asserts the relocated loader stays byte-identical to
the D-06 baseline. That is now literally false for the **setfit** profile: all 346 merged
rows carry `source_split: "compatibility_test"` where the baseline wrote `validation` /
`test`. It remains true for the canonical profile.

The change was forced, not incidental: D-19 requires the merged split to be
`Split<CompatibilityTest>`, and plan 02-03's Gate 2 requires every row's `source_split` to
equal the role being built. It was disclosed via `schema_version` 1 -> 2, a
`profiles.setfit.row_source_split` field, and a pinning test.

The verifier classified this UNCERTAIN rather than FAILED, on the grounds that reverting it
would break D-19 and therefore ROADMAP success criterion 5 — so routing it to gap closure
would be actively wrong.

expected: A human accepts the deviation (ready-to-paste override YAML is in
02-VERIFICATION.md), or rejects it and specifies how D-19 should otherwise be satisfied.
result: ACCEPTED (Guy Ernest, 2026-08-09). Override written into 02-VERIFICATION.md
frontmatter and FINDING-D1 status flipped uncertain -> accepted. The must_have is now scoped
to the canonical profile, where byte-identity holds and is proven by independent
re-derivation.

### 2. Approve the crates.io publish cascade

`apr-cli` now depends on `aprender-contrastive-data`, which is not on crates.io. Every form
of `cargo package -p apr-cli` therefore fails, including `--no-verify` (which still performs
the manifest resolution that rewrites the path dep into a registry dep). The crate-only form
is green at 59 files.

This is the only outstanding item behind the known-red pre-release Gate 5. Clearing it needs
`aprender-contrastive-data` published **before** `apr-cli`. CLAUDE.md forbids an agent
self-serving the publish cascade, so no executor attempted it.

expected: A human approves and runs the publish cascade in that order, or defers it and
accepts Gate 5 staying red until they do.
result: APPROVED, BUT NOT AUTOMATED — correction on the record. The approval assumed CI
publishes after the PR merges. It does not: there is no `release.yml`, no
`CARGO_REGISTRY_TOKEN` anywhere in `.github/`, and `binary-release.yml`'s own header says it
is "Decoupled from `cargo publish`". Publishing is the manual `make publish` target
(Makefile:1189), which CLAUDE.md requires asking before. **Merging the PR will not publish
anything and Gate 5 will stay red** until someone runs the cascade by hand, in the order
`aprender-contrastive-data` then `apr-cli`. STILL PENDING that manual run.

### 3. Resolve the untested FALSIFY-CPP-007 prediction (FINDING-W2)

`FALSIFY-CPP-007` in `contracts/contrastive-pair-protocol-v1.yaml` predicts N=512 with
`C(512,2) = 130816`. The maximum K actually exercised anywhere in the suite is 128, and the
literal `130816` appears nowhere in the tree. The prediction is currently unfalsifiable —
the same "a gate that cannot fail for the reason it claims" class this phase repeatedly
surfaced elsewhere in the repo.

Two honest resolutions: add a K=512 case so the prediction is exercised, or amend the
prediction to the layout that is actually tested. Left for a human because choosing between
them is a scope call, not a mechanical fix.

expected: Either a K=512 case exists and passes, or FALSIFY-CPP-007 states a bound the suite
actually tests.
result: RESOLVED. `falsify_cpp_007_pairs_at_n_512_singletons_stays_bounded` asserts every
number the contract names — positive_capacity 0, negative_capacity C(512,2) = 130816,
negatives-only emission — plus linear growth (1536 retained = 4x the K=128 total of 384, not
16x) and a full drain of the fixed 64 budget. Mutation-tested: perturbing negative_capacity
gives "left: 130305, right: 130816".

A second defect surfaced while fixing this and was also fixed: the contract's OWN declared
harness, `cargo test -p aprender-contrastive-data pairs`, selected ZERO tests from
`negative_materializing.rs`. The evidence command named by the contract could not have
reached the new test either. Repointed at
`--test negative_materializing falsify_cpp_007`, which selects 1.

### 4. Triage three code-review blockers (02-REVIEW.md)

The independent code review found 3 blockers and 14 warnings across 34 files. The library core
held up under adversarial reading — zero hash-ordered collections, checked capacity arithmetic,
digest-before-parse, genuinely O(K) retained state, correct union-find coalescing. The defects
cluster at the edges.

- **CR-01 path traversal (security) — FIXED in `3868dc453`.** `role_file` is now an allowlist
  over the four protocol split roles and is fallible; an unknown role fails closed instead of
  becoming a path. Previously the untrusted `attestation.splits` role was joined onto `--data`
  via `format!("{role}.jsonl")`, and `Path::join` replaces the base on an absolute component,
  so `"role": "/etc/passwd"` read `/etc/passwd.jsonl` and `"../../.."` escaped. Mutation-tested:
  restoring the unguarded `format!` turns the new test RED with the traversal quoted verbatim.
  The test carries a mirror, so a `role_file` that refused *everything* could not pass it.
  **Note this survived the `/code-review --fix` pass** — it was reported in the first review and
  not re-reported in the second, which is a reminder that "the review came back clean" is not
  the same as "the earlier finding was addressed".
- **CR-02 `--force` re-prepare destroys the prior benchmark — FIXED in `3868dc453`.**
  `write_outputs`' rollback now removes only files the run actually CREATED, never pre-existing
  ones truncated under `--force`. Previously a failed `verify_prepared_directory` deleted all
  four files, leaving the user with no benchmark directory having started with a complete one.
- **CR-03 the new tier2/tier3 gates cannot fail under GNU Make 4.x.** VERIFIED by the
  orchestrator on this host with both make versions installed:

  ```
  SHELL := /bin/bash ; .ONESHELL: ; recipe = { false ; @echo "done" }
  make  3.81 (macOS default)      -> exit=2   failure caught
  gmake 4.4.1 (Linux dev + CI)    -> exit=0   failure SWALLOWED
  ```

  `.ONESHELL:` (Makefile:20) with no `.SHELLFLAGS` override runs the whole recipe in one shell
  with default `-c` and no `-e`, so only the LAST line's status survives. Both tier recipes end
  in `@echo "Tier N: PASSED"`, which always succeeds. Under Make 4.x that disarms every gate in
  them: tier2's `cargo test --lib`, clippy, and the Phase 1 + Phase 2 suites; tier3's
  `cargo test --all`, clippy, all four `check_*.sh` scripts, `contract-validate`, the new
  BLOCKING `contract-audit-phase2`, `setfit-feature-matrix` and `contrastive-data-boundary`.

  **Scope limit, measured:** CI does NOT currently invoke `make tier2`/`tier3` (it runs cargo
  directly), so no CI status check is vacuously green today. The defect bites Linux developers
  running the tiers locally, and would bite CI the moment the tiers are wired in — which D-26's
  own rationale ("a gate outside the tiers is a target that stops being run") encourages.

  This is why plan 02-08's gate-failure proof passed honestly and still did not transfer: it was
  measured standalone on macOS make 3.81, which ignores `.ONESHELL:`. CLAUDE.md rule 4 —
  extending a guard's scope requires re-mutating in the new scope.

  The reviewer's fix is one line, `.SHELLFLAGS := -e -u -o pipefail -c`. NOT applied: it changes
  failure semantics for every recipe in a 1000+ line Makefile (`-u` in particular will trip on
  unset variables, and several targets rely on `|| true`). It needs its own verification pass
  across all targets on BOTH make versions, then re-mutation in the new scope.

CR-01 and CR-02 are now closed in code with mutation-tested guards, so what remains of this
item is **CR-03 only**. A second `/code-review --fix` pass also closed six further correctness
findings (renamed-class replay, unearned ledger hash, a clamp lost on rebuild, uppercase digest
acceptance, and two constant-returning enum accessors), each with a regression test proven RED
before restore.

expected: A human triages CR-03 as its own Makefile-hardening task — apply
`.SHELLFLAGS := -e -u -o pipefail -c`, then re-verify every target on BOTH make 3.81 and
gmake 4.4.1, and re-mutate the gates in the new scope rather than trusting the earlier
standalone proofs.
result: RESOLVED, with a narrower fix than the reviewer proposed. Applied
`.SHELLFLAGS := -e -c`, NOT `-e -u -o pipefail -c`. Blast radius measured over all 85
targets: 35 recipes reference a shell variable (a `-u` risk) and 5 pipe into `head`/`tail`,
where the reader closing the pipe SIGPIPEs the writer and `pipefail` converts that into a
failure. `-e` alone restores the per-line abort semantics make 3.81 already had, which is
the actual defect; `-u` and `pipefail` are separate hardening that would need their own pass.

Re-mutated in the NEW scope rather than trusting the standalone proof (CLAUDE.md rule 4).
`tier2` already contains a real failure — 19 arm64 clippy errors in `aprender-compute` — so
it is a live mutation needing no synthetic one:

```
gmake 4.4.1, before: rc=0, prints "Tier 2: PASSED"   <- over 19 compile errors
gmake 4.4.1, after:  rc=2
```

make 3.81 is unaffected: `.SHELLFLAGS` arrived in 3.82, and 3.81 also predates `.ONESHELL`
so it never had the bug. Verified it parses, the gate runs rc=0, and tier2 is rc=2 as before.

Scope, for the CLI-compatibility concern: this is BUILD-GATE ONLY. No CLI surface, no
library API, no algorithm behaviour changes — nothing downstream of `apr` or the crates is
affected. The visible change is that Linux developers running `make tier2`/`tier3` now see
real failures instead of a false pass. `bashrs` is not installed here so the Makefile shell
could not be linted; shellcheck was NOT substituted.

## Summary

total: 4
passed: 3
issues: 0
pending: 1
skipped: 0
blocked: 0

## Gaps

**Goal verification: no gaps.** The verifier found no gaps, no debt markers, and no blockers
against the phase goal anywhere in the diff, scoring 5/5 ROADMAP success criteria VERIFIED
(31/32 must-haves). Phase 3 is not blocked by any item in this file.

**Code review: 3 blockers, 14 warnings** — see item 4 and `02-REVIEW.md`. These are a separate
lens from goal verification: the phase does what it set out to do, and CR-01/CR-02 are latent
defects on paths functional verification did not exercise (a hostile attestation, and a
`--force` re-prepare over pre-existing files). CR-03 is a build-gate defect. None of the three
falsifies a ROADMAP success criterion, which is why the two reports differ without conflicting.

FINDING-W1 (a stale module-doc comment claiming `run_pairs` was still a placeholder) was a
plain defect rather than a decision and was fixed directly in `f3ee9c0d2`.
