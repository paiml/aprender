---
allowed-tools: Bash(cargo:*), Bash(apr:*), Bash(pmat:*), Bash(gh:*), Bash(git:*), Bash(find:*), Bash(head:*), Bash(tail:*), Bash(wc:*), Bash(grep:*), Read, Glob, Grep
description: Dogfood apr-cli — rebuild, install, exercise all commands against real models, check quality, find next work
---

## Context

- apr-cli local version: !`grep '^version' crates/apr-cli/Cargo.toml | head -1`
- Current git commit: !`git rev-parse --short HEAD`
- Installed apr version: !`apr --version 2>/dev/null || echo "not installed"`
- Available models: !`find ~/models -maxdepth 2 \( -name "*.apr" -o -name "*.gguf" -o -name "*.safetensors" \) -type f 2>/dev/null | wc -l` files
- Test count: !`cargo test -p apr-cli --lib 2>&1 | grep 'test result' | tail -1`

## Your Task

Run the apr-cli dogfood checklist below. This is the daily development loop: rebuild the binary, exercise it against real model files, check quality gates, and surface next work.

For each gate, run the check, report PASS/FAIL/SKIP, and if FAIL explain the root cause and how to fix it. At the end, give a GO/WARN/FAIL verdict.

Run all independent gates in parallel where possible.

### Gate 1: Build & Install

Rebuild and install apr-cli from source:

```
cargo install --path crates/apr-cli --force 2>&1 | tail -5
```

Then verify the installed binary matches the current commit:

```
apr --version
git rev-parse --short HEAD
```

The version string from `apr --version` should contain the commit hash from `git rev-parse --short HEAD`. FAIL if build errors or version mismatch.

### Gate 2: Multi-Format Exercise

Auto-discover model files:

```
find ~/models -maxdepth 2 \( -name "*.apr" -o -name "*.gguf" -o -name "*.safetensors" \) -type f 2>/dev/null
```

Pick one file per format (APR, GGUF, SafeTensors) from the discovered models. For each discovered model, exercise these commands and report any errors:

- `apr validate <model>`
- `apr inspect <model> | head -30`
- `apr lint <model>`
- `apr tensors <model> | head -10`
- `apr debug <model>`

If any model is available, also run:

```
apr qa <model>
```

SKIP (not FAIL) if no models found in ~/models. FAIL if any command crashes or returns a non-zero exit code on a discovered model.

### Gate 3: Help Text Audit

Verify help text is complete and accurate:

```
apr --help
apr import --help
apr export --help
```

Check that:
1. `apr --help` lists all expected subcommands (run, serve, compile, inspect, debug, validate, diff, tensors, trace, lint, explain, canary, export, import, convert, merge, tui, probar, profile, qa)
2. `apr import --help` lists all architectures in `--arch` (should include llama, gpt2, gpt-neox, opt, starcoder, phi, qwen2, gemma, falcon, mamba, bert, t5, whisper)
3. `apr export --help` lists format options

FAIL if expected subcommands or architecture options are missing.

### Gate 4: Code Quality Gate

Run pmat quality gates:

```
pmat quality-gates 2>&1
```

Report violation count and categories. WARN on soft violations (recommended thresholds like complexity). FAIL only on hard violations (security vulnerabilities, test coverage below 95%).

### Gate 5: Test Suite Smoke

Run the apr-cli test suite:

```
cargo test --lib -p apr-cli 2>&1 | tail -5
```

Report pass/fail/ignored counts from the `test result` line. FAIL if any test fails.

### Gate 6: Next Work Discovery

Surface open issues for next ticket selection:

```
gh issue list --repo paiml/aprender --state open --limit 20 2>&1
```

Report the issue list. Always PASS (informational only).

## Verdict

After running all gates, provide:

1. A summary table: Gate | Status | Notes
2. **GO** if all gates pass — ready for next ticket
3. **WARN** if soft quality violations exist but the binary works correctly
4. **FAIL** if build, test, or binary failures need fixing first

If WARN or FAIL, list the specific issues and commands to fix them.

Do NOT modify any files. This is a read-only audit.
