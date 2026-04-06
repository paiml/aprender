# Jidoka: Built-in Quality

**Jidoka** (自働化) means "automation with a human touch" - the practice of building quality into the process itself.

## Core Principle

> Stop the line when a defect is detected. Fix the root cause before continuing.

In Batuta, Jidoka manifests as automatic quality gates that halt the pipeline when issues are found.

## Jidoka in Batuta

### Pre-commit Hooks

```bash
# Automatic checks before every commit
cargo fmt --check     # Formatting
cargo clippy          # Linting
cargo test            # Tests
pmat demo-score       # Quality gate
```

If any check fails, the commit is blocked.

### Quality Gates

| Gate | Threshold | Action |
|------|-----------|--------|
| Demo Score | A- (85) | Block release |
| Test Coverage | 85% | Warning |
| Clippy | 0 warnings | Block commit |
| Format | 100% | Block commit |

### Stop-the-Line Examples

```rust
// Jidoka: Fail fast on type errors
fn transpile(source: &str) -> Result<String, Error> {
    let ast = parse(source)?;  // Stop if parse fails
    let typed = typecheck(ast)?;  // Stop if types invalid
    generate(typed)
}
```

## Benefits

1. **Early detection** - Issues caught immediately
2. **Root cause focus** - Fix problems, not symptoms
3. **No defect propagation** - Bad code never reaches production
4. **Team awareness** - Everyone knows quality status

## Implementation

### Andon Board

Batuta's diagnostics module provides Andon-style status:

```
🟢 Green  - All systems healthy
🟡 Yellow - Attention needed
🔴 Red    - Stop the line
```

### Automated Response

When issues are detected:
1. Pipeline stops
2. Team is notified
3. Root cause is investigated
4. Fix is verified
5. Pipeline resumes

---

**Navigate:** [Table of Contents](../SUMMARY.md) | [Next: Kaizen](./kaizen.md)
