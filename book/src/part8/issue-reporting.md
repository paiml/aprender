# Issue Reporting

A well-written issue report saves time for everyone. This chapter describes what to include for fast resolution.

## Minimum Reproducible Example

Every issue should include a minimal example that reproduces the problem:

```
**Title:** Transpilation fails on Python generator with yield from

**Steps to reproduce:**
1. Create file `test.py` with `yield from` syntax
2. Run: `batuta transpile --source . --target ./out`
3. Observe: `UnsupportedFeature: yield_from at line 3`

**Expected:** Generator transpiles to Rust Iterator
**Actual:** Pipeline stops with UnsupportedFeature error
```

## Diagnostic Information to Include

```bash
batuta --version && rustc --version && cargo --version
batuta analyze --check-tools
batuta status --verbose

# Attach debug logs
RUST_LOG=debug batuta transpile --source ./minimal_example 2> debug.log
```

## Bug Report Template

```markdown
## Description
[One sentence describing the bug]

## Steps to Reproduce
1. [Step 1]
2. [Step 2]

## Expected vs Actual Behavior
[What should happen vs what happens]

## Environment
- batuta version:
- Rust version:
- OS:

## Minimal Reproduction
[Code or repository link]

## Logs
[Attach RUST_LOG=debug output]
```

## What Happens After Filing

| Stage | Timeline | Action |
|-------|----------|--------|
| Triage | 1-3 days | Issue labeled and prioritized |
| Investigation | 3-7 days | Root cause identified |
| Fix | 1-2 weeks | Patch or documented workaround |
| Release | Next cycle | Fix included in release |

Critical bugs (data loss, security) are prioritized above all other work.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
