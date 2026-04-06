# Kanban: Visual Workflow

**Kanban** (看板) means "signboard" - the practice of making work visible so teams can manage flow and limit work in progress.

## Core Principle

> Make the invisible visible. Limit work in progress to maximize throughput.

In Batuta, Kanban manifests as real-time dashboards that surface pipeline state, stack health, and quality metrics at a glance.

## Kanban in Batuta

### Pipeline State Visibility

```bash
# Show current pipeline state across all phases
batuta status

# Phase      | Status     | Duration
# -----------|------------|----------
# Analysis   | Complete   | 1.2s
# Transpile  | Running    | 3.4s (depyler)
# Optimize   | Pending    | -
# Validate   | Pending    | -
# Build      | Pending    | -
```

Each phase of the 5-phase pipeline is a Kanban column. Work items flow left to right, and Jidoka stops the line if any phase fails.

### Stack Quality Matrix

```bash
# TUI dashboard showing all stack components
batuta stack status

# Component   | Version | Health | Coverage | TDG
# ------------|---------|--------|----------|-----
# trueno      | 0.16.x  | Green  | 95%      | A
# aprender    | 0.27.x  | Green  | 95%      | A-
# realizar    | 0.8.x   | Yellow | 91%      | B+
# repartir    | 2.0.x   | Green  | 93%      | A
```

### WIP Limits

Batuta enforces WIP limits to prevent overloading any stage:

| Resource | WIP Limit | Rationale |
|----------|-----------|-----------|
| Concurrent transpilations | 4 | CPU-bound, avoid thrashing |
| GPU kernel dispatches | 1 | Single GPU context |
| Validation suites | 2 | Memory-intensive |
| Stack releases | 1 | Sequential dependency graph |

### Pull-Based Execution

```rust
// Kanban: downstream phases pull work when ready
fn run_pipeline(config: &Config) -> Result<Report> {
    let analysis = analyze(config)?;        // Phase 1
    let transpiled = transpile(&analysis)?;  // Phase 2 pulls from 1
    let optimized = optimize(&transpiled)?;  // Phase 3 pulls from 2
    let validated = validate(&optimized)?;   // Phase 4 pulls from 3
    build(&validated)                        // Phase 5 pulls from 4
}
```

## Benefits

1. **Flow visibility** - See bottlenecks before they stall the pipeline
2. **WIP control** - Prevent resource exhaustion from over-parallelism
3. **Pull scheduling** - Each phase processes work only when capacity allows
4. **Stack awareness** - One dashboard for the entire Sovereign AI Stack

## Board Layout

```
| Backlog | Analysis | Transpile | Optimize | Validate | Done |
|---------|----------|-----------|----------|----------|------|
|         | app.py   |           |          |          |      |
|         |          | lib.c     |          |          |      |
|         |          |           |          | util.sh  |      |
| WIP: -  | WIP: 2/4 | WIP: 1/4 | WIP: 0/2 | WIP: 1/2 |      |
```

---

**Navigate:** [Table of Contents](../SUMMARY.md) | [Next: Andon](./andon.md)
