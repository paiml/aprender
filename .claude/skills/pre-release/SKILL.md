---
# EXPLICIT name (#2332 class, applied here by #2543). Without it the skill takes
# its name from the directory and there is nothing to point at when a user-scope
# skill claims the same triggers — the sibling apr-dogfood skill was edited for
# months while a shadowing copy was what actually ran. This file is amended by
# #2543 and must be reachable for the amendment to mean anything.
name: pre-release
allowed-tools: Bash(cargo:*), Bash(grep:*), Bash(make:*), Bash(bash:*), Bash(batuta:*), Bash(pmat:*), Bash(git:*), Bash(head:*), Bash(tail:*), Bash(wc:*), Bash(cat:*), Bash(awk:*), Bash(sed:*), Bash(diff:*), Bash(rustup:*), Read, Glob, Grep
description: Pre-release QA for apr-cli — runs all gates that prevent crates.io publish breakage
effort: high          # MACS F4: pinned for reproducible cost/behavior - gates a crates.io publish; a wrong verdict here costs a yank

---

## Context

- Current apr-cli version: !`grep '^version' crates/apr-cli/Cargo.toml | head -1`
- Published version: !`cargo search apr-cli 2>/dev/null | head -1`
- Current branch: !`git branch --show-current`
- Uncommitted changes: !`git status --short | wc -l`
- Test count: !`cargo test -p apr-cli --lib 2>&1 | grep 'test result' | tail -1`

## Your Task

Run the apr-cli pre-release QA checklist below. This checklist was derived from 5 historical release failures (CB-510, PMAT-262, GH-342, GH-343, GH-344/345) using Five-Whys root cause analysis on git history.

For each gate, run the check, report PASS/FAIL, and if FAIL explain the root cause and how to fix it. At the end, give a GO/NO-GO verdict.

Run all independent gates in parallel where possible.

### Gate 1: Package Integrity (CB-510)

Verify all `include!()` files are tracked by git and included in the cargo package:

```
bash scripts/check_include_files.sh
bash scripts/check_package_includes.sh
```

If either fails, files will be missing from crates.io publish.

### Gate 2: No External Path Dependencies (GH-344, PMAT-262)

Check that committed Cargo.toml files have NO external `path = "../` references (sibling repos). Only intra-workspace `path = "../.."` is allowed:

```
grep -n 'path = "\.\./\.\.' Cargo.toml crates/apr-cli/Cargo.toml | grep -v '../..'
```

Any external path deps mean `cargo install apr-cli` will fail for users who don't have sibling repos.

### Gate 3: Stale cfg Gate Audit (GH-342)

Search for `#[cfg(` attributes on pub/pub(crate) functions in apr-cli that might hide essential code:

```
grep -rn '#\[cfg(' crates/apr-cli/src/ --include='*.rs' | grep -v test | grep -v '// ' | grep -v '#\[cfg(test' | grep -v '#\[cfg(not(feature'
```

Review each cfg gate. Common failure: `#[cfg(all(feature = "inference", feature = "cuda"))]` applied to utility functions that should always be available. Cross-reference with the feature flags in `crates/apr-cli/Cargo.toml` to verify.

### Gate 4: MSRV Verification (GH-343)

Verify the declared `rust-version` is accurate:

1. Check declared MSRV: `grep rust-version Cargo.toml crates/apr-cli/Cargo.toml`
2. Check actual toolchain: `rustc --version`
3. Verify both Cargo.toml files declare the same MSRV

### Gate 5: Standalone Package Build (GH-344/345) — STAGE-DEPENDENT (#2543)

**Read this before running it.** `cargo package` re-resolves every dependency
against crates.io, so each workspace sibling resolves to its *already published*
copy rather than to this tree. `apr-cli` has 26 workspace-sibling dependencies,
five of which publish in the LAST cascade tier — so Gate 5 for `apr-cli` is not
merely "late", it is meaningful only after the cascade has finished. Run it at
the wrong stage and you get dozens of **symbol**-not-found errors:

```
error[E0432]: unresolved import `aprender::format::q4k_output_size_estimate`
error[E0432]: unresolved import `entrenar_lora::plan_with_rank`
error[E0433]: could not find `CancelToken` in `generate`
```

Those three symbols are genuine post-0.63.0 additions (verified against the
published 0.63.0 tarballs). Nothing is broken. **Do not abort a cut over this.**

STAGE-PRECONDITION: cargo package -p apr-cli requires stage CASCADE_READY
STAGE-PRECONDITION: cargo package -p apr-format requires stage MEANINGFUL

Ask the tree which stage it is at before interpreting any result:

```
bash scripts/check_gate5_stage.sh --explain apr-cli
```

| Verdict | What Gate 5 means right now |
|---|---|
| `MEANINGFUL` | No workspace-sibling deps. A failure is a real defect. |
| `PRE_BUMP` | The workspace version is already on crates.io, so siblings resolve to the stale API at the same version number. **Symbol errors are expected.** Bump, then re-check. |
| `POST_BUMP_PRE_CASCADE` | Version bumped, siblings not published at it yet. Cargo says `failed to select a version … candidate versions found which didn't match` — *not* `no matching package named`, which is the distinct error for a crate that was never published at all. Expected until the cascade reaches them. |
| `CASCADE_READY` | Every sibling is live at this version. A failure here is a real defect. |

**The stage-independent substitute.** `apr-format` is a workspace leaf with zero
sibling dependencies, so it packages identically at every stage and still proves
the tarball/`include!()`/manifest machinery works:

```
cargo package -p apr-format --allow-dirty 2>&1 | tail -5
```

Run that pre-bump. Run the real gate

```
cargo package -p apr-cli --allow-dirty 2>&1 | tail -5
```

only once `check_gate5_stage.sh --explain apr-cli` reports `CASCADE_READY` —
i.e. at the END of the publish cascade, not at apr-cli's own tier. `apr-cli` is
tier 10 of 13 in `scripts/cascade-publish.sh` but depends on five crates that
publish in tier 13, so tier 10 is still too early.

**Do NOT substitute Gate 11 here.** Gate 11 is itself pre-bump-only: after the
version bump `cargo publish -p aprender --dry-run --no-verify` fails with the
very `candidate versions found which didn't match` string Gate 11's own text
declares a FAILURE.

The old wording of this gate said "if it fails, `cargo install apr-cli` will fail
for users". That is false pre-bump — apr-cli 0.63.0 built fine on docs.rs from
published deps alone while this gate was red on the tree.

Enforced by `scripts/check_gate5_stage.sh` (contract
`contracts/publish-workspace-v1.yaml`, FALSIFY-PUB-005/006/007).

### Gate 6: Test Suite

Verify all tests pass:

```
cargo test -p apr-cli --lib 2>&1 | tail -3
```

### Gate 7: Formatting + Clippy

```
cargo fmt -p apr-cli -- --check
```

Report any formatting issues (don't fix them — just report).

### Gate 8: Version Bump Check

Verify the local version is GREATER than the published crates.io version. If not, the publish will fail.

Compare local version from `crates/apr-cli/Cargo.toml` against `cargo search apr-cli`.

### Gate 9: batuta bug-hunter Scan

Run static analysis for high-severity findings:

```
batuta bug-hunter analyze crates/apr-cli/ --format json 2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
findings = data.get('findings', [])
high = [f for f in findings if f.get('severity') == 'High']
categories = {}
for f in high:
    cat = f.get('category', 'Unknown')
    categories[cat] = categories.get(cat, 0) + 1
print(f'High findings: {len(high)}')
for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
    print(f'  {cat}: {count}')
# Flag non-false-positive categories
real = {k: v for k, v in categories.items() if k not in ['SecurityVulnerabilities']}
if any(v > 0 for v in real.values()):
    print(f'WARNING: {sum(real.values())} non-security High findings need triage')
"
```

SecurityVulnerabilities are expected (CLI takes file paths — not a web service). Focus on HiddenDebt, MemorySafety, SilentDegradation, LogicErrors.

### Gate 10: Sibling Repo Versions (GH-345)

If sibling repos are present, verify their versions are compatible:

```
make check-siblings 2>&1
```

### Gate 11: crates.io Cascade Publishability (v0.50.0 dev-dep cycle + missing-version)

The v0.50.0 cascade FAILED MID-PUBLISH (29/68 crates live, then stuck) on two classes that
path-deps mask and `cargo metadata` does NOT catch:
(a) **sibling path-deps with NO `version` field** — `cargo publish` requires a version on every
non-dev dep (locally the path resolves, so it builds fine; publishing errors `dependency X does
not specify a version`);
(b) **version-pinned sibling DEV-dependencies forming publish CYCLES** — cargo tolerates dev-dep
cycles when building locally, but crates.io rejects them (`failed to select a version ... candidate
versions found which didn't match`). Two unused dev-deps (trueno-viz, renacer) closed real cycles.

Only a real publish dry-run of the FLAGSHIP resolves the whole 68-crate tree against the registry:

```
cargo publish -p aprender --dry-run --allow-dirty --no-verify 2>&1 | tail -6
```

PASS if it reaches `Packaged`/`Uploading` with NO `does not specify a version` and NO `candidate
versions found which didn't match`. FAIL on either: a sibling path-dep needs a `version` field, or a
sibling **dev**-dep must be made path-only (no version) so cargo strips it from the published manifest
and the cycle breaks. Also dry-run `apr-cli`, `aprender-core`, `aprender-serve` if `aprender` passes,
to confirm the foundational tier. See memory/feedback_crates_io_devdep_publish_cycles.md.

**Gate 11 is PRE-BUMP ONLY (#2543).** Its pass criterion is stage-dependent in
the mirror image of Gate 5: once the workspace version is bumped, *every* crate
— including the flagship `aprender` — dry-runs to `failed to select a version
for the requirement … candidate versions found which didn't match`, because no
sibling is published at the new version yet. That is the exact string this gate
declares a FAILURE, so post-bump Gate 11 self-reports a defect that does not
exist. Run Gate 11 before `cargo set-version`; after the bump, the equivalent
signal is Gate 5 on a zero-sibling crate plus
`scripts/check_gate5_stage.sh --explain <crate>`.

### Gate 12: Competitive-Parity Ledger (permanent hard requirement)

Competitive parity is a **permanent hard requirement of every release**, not a
nice-to-have, and it is checked here because the RELEASE is a decision surface
CI does not cover: `ci / gate` runs on a PR, a release cut does not have to pass
through one. A guard that skips the surface where the decision is made is
theater.

```
bash scripts/check_competitive_parity.sh --self-test   # case table first
cargo build --bin apr                                   # the universe is enumerated at RUNTIME
bash scripts/check_competitive_parity.sh
```

PASS when it prints `competitive-parity ratchet OK`. What it asserts:

* every in-scope entry point in `scripts/competitive_parity_scope.txt` still
  exists in the SHA-pinned binary's own `apr --help`;
* every row of `contracts/apr-competitive-parity-v1.yaml` has a verdict from the
  closed vocabulary and is INSIDE its `valid_until` **as of today**;
* `__MEASURED__` and `__NON_WINS__` have not fallen below
  `scripts/competitive_parity_baseline.txt`.

**Read the failure correctly.** This gate does NOT require a win. `WORSE`,
`NOT_COMPARABLE` and `UNMEASURED` are first-class verdicts and `WORSE` counts as
MEASURED. A failure means a row was DELETED, EXPIRED, or fell out of scope — never
that apr lost. The two ways this gate goes red at release time and what each means:

| Symptom | Meaning | Fix |
|---|---|---|
| `__MEASURED__ fell` with `__EXPIRED__ > 0` | a measurement aged out during the release window | re-measure, or re-record the row as `UNMEASURED` with a new `valid_until` and an owner |
| `__MEASURED__ fell` with `__EXPIRED__ = 0` | a row was removed from the ledger | put it back. Recording a loss is compliant; deleting one is the PMAT-733 defect (the StandardScaler 0.69x row was deleted the day it was measured) |

Do NOT "fix" this by deleting a row or by editing
`scripts/competitive_parity_baseline.txt` downward. `--update-baseline` refuses
to lower any figure, and a hand-edit is the same act with the safety removed.

## Verdict

After running all gates, provide:

1. A summary table: Gate | Status | Notes
2. **GO** if all gates pass (or only have known-false-positive failures).
   Gate 12 is NOT waivable and has no known false positive: it is a
   permanent hard requirement, and its verdict values are already allowed to be
   losses, so there is nothing left for a waiver to excuse.
3. **NO-GO** with specific blocking issues if any real gate fails
4. If NO-GO, list the exact commands to fix each failure

Do NOT publish or modify any files. This is a read-only audit.
