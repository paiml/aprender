# Andon: Problem Visualization

**Andon** (行灯) means "lantern" - a signal board that makes quality problems immediately visible to the entire team.

## Core Principle

> Problems must be visible the moment they occur. Hidden failures compound into catastrophes.

In Batuta, Andon manifests as the diagnostics engine that provides colored, at-a-glance status for every stack component and pipeline phase.

## Andon in Batuta

### Stack Health Dashboard

```bash
# Real-time health across all components
batuta stack status

# Component      | Signal | Detail
# ---------------|--------|----------------------------
# trueno         | 🟢     | v0.16.1 — all tests pass
# aprender       | 🟢     | v0.27.2 — coverage 95%
# realizar       | 🟡     | v0.8.0 — 2 clippy warnings
# whisper-apr    | 🔴     | v0.1.0 — build failure
```

### Signal Levels

| Signal | Meaning | Response |
|--------|---------|----------|
| 🟢 Green | All quality gates pass | Continue |
| 🟡 Yellow | Non-blocking warnings detected | Investigate soon |
| 🔴 Red | Blocking failure — stop the line | Fix immediately |

### Diagnostics Engine

The diagnostics module continuously monitors quality signals:

```rust
// Andon: aggregate signals from all quality sources
pub fn diagnose(workspace: &Workspace) -> HealthReport {
    let mut report = HealthReport::new();

    for component in workspace.components() {
        let signal = match (component.tests_pass(), component.clippy_clean()) {
            (true, true)  => Signal::Green,
            (true, false) => Signal::Yellow,
            (false, _)    => Signal::Red,
        };
        report.add(component.name(), signal);
    }

    report
}
```

### Pipeline Andon

Each pipeline phase reports its own Andon signal:

```bash
# Pipeline status with timing and errors
batuta status --verbose

# Phase 1: Analysis    🟢  1.2s
# Phase 2: Transpile   🟢  4.1s (depyler)
# Phase 3: Optimize    🟡  2.3s (SIMD fallback: no AVX-512)
# Phase 4: Validate    🔴  FAILED — output mismatch at line 42
# Phase 5: Build       --  Skipped (Jidoka stop)
```

When Phase 4 signals red, Jidoka halts the pipeline. The Andon board shows exactly where and why.

## Benefits

1. **Instant awareness** - Problems surface immediately, not at release time
2. **Root cause focus** - Signal includes context, not just pass/fail
3. **Team alignment** - Everyone sees the same board, same priorities
4. **Escalation path** - Yellow warns, Red blocks — graduated response

## Andon Cord: Manual Signals

Any team member can pull the Andon cord to flag an issue:

```bash
# Flag a component for investigation
batuta stack flag realizar --reason "output mismatch on Q4K models"

# Clear after resolution
batuta stack clear realizar
```

---

**Navigate:** [Table of Contents](../SUMMARY.md) | [Next: First Principles](./first-principles.md)
