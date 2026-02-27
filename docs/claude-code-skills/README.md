# Claude Code Skills for aprender

aprender ships with two [Claude Code skills](https://docs.anthropic.com/en/docs/claude-code/skills) that automate common development workflows. If you're contributing to aprender or debugging model issues, these skills replace multi-step manual processes with a single `/slash-command`.

## Prerequisites

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) installed and configured
- Clone of this repository
- Rust toolchain (`cargo`, `rustc`)

## Available Skills

### `/dogfood` — Daily Development Loop

**When to use:** After making changes, before picking up the next ticket.

Rebuilds `apr-cli` from source, exercises it against real model files, runs quality gates, and surfaces open issues for next work.

```
> /dogfood
```

**6 gates:**

| Gate | What it checks | Verdict |
|------|---------------|---------|
| 1. Build & Install | `cargo install --path crates/apr-cli --force`, verifies `apr --version` commit hash matches `git rev-parse --short HEAD` | FAIL on build error or version mismatch |
| 2. Multi-Format Exercise | Auto-discovers `.apr`, `.gguf`, `.safetensors` files in `~/models/`, runs `apr validate`, `inspect`, `lint`, `tensors`, `debug`, `qa` on each | SKIP if no models, FAIL on command crash |
| 3. Help Text Audit | Verifies `apr --help` lists all subcommands, `apr import --help` lists all architectures, `apr export --help` lists format options | FAIL on missing commands/options |
| 4. Code Quality | `pmat quality-gates` — coverage, complexity, clippy | WARN on soft violations, FAIL on hard |
| 5. Test Suite Smoke | `cargo test --lib -p apr-cli` | FAIL on any test failure |
| 6. Next Work Discovery | `gh issue list --repo paiml/aprender --state open` | Always PASS (informational) |

**Verdicts:** GO (all pass) / WARN (soft issues, binary works) / FAIL (blocking issues)

**For debugging contributions:** Run `/dogfood` after your fix. Gate 2 exercises `apr` commands against real models — if your fix breaks model handling, this catches it. Gate 5 runs 3,800+ tests to catch regressions.

---

### `/pre-release` — Publish QA

**When to use:** Before running `cargo publish` or `batuta stack release`.

Runs 10 gates derived from 5 historical release failures (CB-510, PMAT-262, GH-342, GH-343, GH-344/345) via Five-Whys root cause analysis.

```
> /pre-release
```

**10 gates:**

| Gate | What it checks | Root cause |
|------|---------------|------------|
| 1. Package Integrity | All `include!()` files tracked by git and in cargo package | CB-510: missing source files on crates.io |
| 2. No External Path Deps | No `path = "../` references to sibling repos in Cargo.toml | GH-344, PMAT-262: `cargo install` fails for users |
| 3. Stale cfg Audit | `#[cfg(` gates not hiding essential code behind optional features | GH-342: functions gated behind unset features |
| 4. MSRV Verification | `rust-version` matches across workspace Cargo.toml files | GH-343: MSRV mismatch breaks older toolchains |
| 5. Standalone Package Build | `cargo package -p apr-cli` succeeds from tarball | GH-344/345: package builds but install fails |
| 6. Test Suite | All tests pass | — |
| 7. Formatting + Clippy | `cargo fmt --check` | — |
| 8. Version Bump | Local version > published crates.io version | Publish collision |
| 9. batuta bug-hunter | Static analysis for high-severity findings | Hidden debt, memory safety, logic errors |
| 10. Sibling Repo Versions | Compatible versions across aprender/trueno/realizar | GH-345: version drift |

**Verdicts:** GO (safe to publish) / NO-GO (specific blocking issues listed with fix commands)

**For debugging contributions:** If your PR touches `Cargo.toml`, `include!()` files, or feature flags, run `/pre-release` to verify you haven't introduced a publish blocker.

## Debugging Workflow

If you're investigating a bug in aprender:

1. **Reproduce** — Use `apr qa <model>` to get a falsifiable failure
2. **Fix** — Make your code changes
3. **Verify** — Run `/dogfood` to rebuild, exercise the binary, and run tests
4. **Pre-publish** — If touching packaging or deps, run `/pre-release`

### Example: Debugging a Model Import Issue

```
# 1. Reproduce the issue
apr import hf://org/model -o test.apr --verbose
apr validate test.apr

# 2. Make your fix in src/format/converter/...

# 3. Run /dogfood in Claude Code to rebuild and verify
> /dogfood

# 4. If Gate 2 (Multi-Format Exercise) passes with your model,
#    the fix works across all formats
```

### Example: Debugging a Test Failure

```
# 1. Run the failing test directly
cargo test -p apr-cli --lib test_name -- --nocapture

# 2. Make your fix

# 3. Run /dogfood — Gate 5 runs all 3,800+ apr-cli tests
> /dogfood
```

## Adding New Skills

Skills live in `.claude/skills/<name>/SKILL.md`. The format is:

```markdown
---
allowed-tools: Bash(cargo:*), Read, Glob, Grep
description: One-line description shown in /help
---

## Context
- Dynamic values: !`shell command here`

## Your Task
Gate-based checklist with PASS/FAIL criteria.

## Verdict
Summary table and GO/NO-GO decision.
```

See [Claude Code Skills documentation](https://docs.anthropic.com/en/docs/claude-code/skills) for the full spec.
