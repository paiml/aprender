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

## Three documents, three scopes — and only one of them is the fleet protocol

Written down because three skills describe overlapping release work and the
duplication is harder to see in prose than in code: nothing runs it, and no diff
surfaces it (aprender#2640, D6).

| skill | scope | source of truth for |
|---|---|---|
| `dogfood` (`.claude/skills/dogfood/SKILL.md`) | ANY Rust crate in the fleet | the generic pre-release protocol and `scripts/dogfood.sh` |
| `apr-dogfood` | this repo's shipped surface | gate coverage against the surface ledger |
| `pre-release` (this file) | `apr-cli` only | the crates.io publish gates below |

**Do not restate a gate that lives in another of the three.** If a gate here also
belongs to the fleet protocol, it belongs in `scripts/dogfood.sh` and this file
should reference it — that is exactly how the runner came to exist twice.

Run all independent gates in parallel where possible.

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


### Gate 12: Multi-Platform Dogfood (MANDATORY — aprender#2566)

```bash
bash scripts/check_multiplatform_dogfood.sh
```

**Every release is dogfooded on EVERY supported platform, not just the one the release
engineer is sitting at.** This gate does not check that someone ran a sweep — it checks
that a dated **receipt** exists for each host, for the version being cut. A receipt is
evidence; a checklist tick is not. A receipt from a previous release is STALE and fails.

| host | platform | why it is in the matrix |
|---|---|---|
| `lambda` | x86_64 Linux + RTX 4090 (sm_89) | consumer x86, AVX2 path |
| `intel` | x86_64 Linux, Xeon W-3245 | **AVX-512 + VNNI** path |
| `gx10` | aarch64 Linux + GB10 (sm_121) | ARM server, unified memory |
| `mini` | arm64 macOS + Metal | Apple silicon, **no /proc**, APFS case-insensitive |

Each host is in the matrix because it is a distinct combination of **ISA, OS and
accelerator** — not because we happen to own it.

**What one afternoon of this bought (the 0.64.0 cut).** The published crate had never
been verified on either arm64 platform:

- **#2567** — Q4_K GEMV, the hottest kernel in quantized inference, has **zero aarch64
  SIMD**, and `matmul_q4k_f32_parallel` on non-x86 is a direct call to the *serial
  scalar* routine. The numbers are correct and only the speed is wrong, so **no
  correctness gate could ever have caught it.**
- **#2568** — the OOM guard reads `/proc/meminfo` and `.unwrap_or(u64::MAX)`, so on macOS
  the threshold becomes ~12.8 **exabytes** and the guard can never fire. Its only test
  self-skips with `cfg!(target_os = "linux")` — the platform where it is broken.
- **#2572** — `block v0.1.6` faces future-rustc rejection and sits under `wgpu -> metal`,
  the only GPU backend macOS has. Entirely absent from the Linux dependency graph.

Each is invisible from a single host **by construction**. That is the argument for this
gate: not diligence theatre, but the only way to see this class of defect.

**Recording a receipt.** Run the sweep on the host, then write
`evidence/dogfood/<version>/<host>.json` with at least:

```json
{"host":"gx10","arch":"aarch64-unknown-linux-gnu","version_tested":"0.64.0",
 "date":"2026-08-22","install_rc":0}
```

Richer fields (surface counts, findings, notable, verdict) are encouraged — the receipts
already under `evidence/dogfood/` are the worked examples.

**The sweep must `cargo install` the PUBLISHED crate**, not build the local tree.
Building the tree tests what you have; installing tests what a user gets. On a box with a
pre-existing `apr` the install correctly fails closed (rc=101) *before* compiling — use
`--force` and record that in the receipt.

**Watch the CI host.** `intel` runs all 16 self-hosted runners. Build there with
`-j 6`, not the default 32: the merge-queue timeout counts runner wait, so a build that
steals cores is indistinguishable from a flake and can evict queued PRs.

**Non-vacuity:** the gate refuses a matrix of fewer than 4 hosts, because a shrinking
matrix silently narrows what "verified" means. Mutation-verified in all three directions —
a stale receipt version, a non-zero `install_rc`, and a shortened host list each turn it
RED.


## Verdict

After running all gates, provide:

1. A summary table: Gate | Status | Notes
2. **GO** if all gates pass (or only have known-false-positive failures)
3. **NO-GO** with specific blocking issues if any real gate fails
4. If NO-GO, list the exact commands to fix each failure

Do NOT publish or modify any files. This is a read-only audit.
