---
allowed-tools: Bash(cargo:*), Bash(grep:*), Bash(make:*), Bash(bash:*), Bash(batuta:*), Bash(pmat:*), Bash(git:*), Bash(head:*), Bash(tail:*), Bash(wc:*), Bash(cat:*), Bash(awk:*), Bash(sed:*), Bash(diff:*), Bash(rustup:*), Read, Glob, Grep
description: Pre-release QA for apr-cli — runs all gates that prevent crates.io publish breakage
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

### Gate 5: Standalone Package Build (GH-344/345)

Verify the package builds from crates.io deps alone (no sibling path overrides):

```
cargo package -p apr-cli --allow-dirty 2>&1 | tail -5
```

This compiles from the packaged tarball against crates.io dependencies. If it fails, `cargo install apr-cli` will fail for users.

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

## Verdict

After running all gates, provide:

1. A summary table: Gate | Status | Notes
2. **GO** if all gates pass (or only have known-false-positive failures)
3. **NO-GO** with specific blocking issues if any real gate fails
4. If NO-GO, list the exact commands to fix each failure

Do NOT publish or modify any files. This is a read-only audit.
