# Workflow State Machine

The Batuta pipeline is a 5-phase state machine with explicit transitions, error states, and recovery paths. Each phase must complete successfully before the next begins (Jidoka principle).

## State Diagram

```
          ┌──────────┐
          │  INIT    │
          └────┬─────┘
               ▼
          ┌──────────┐     ┌─────────┐
          │ ANALYSIS │──X──│ FAILED  │
          └────┬─────┘     └────┬────┘
               ▼                │ reset
          ┌──────────┐         │
          │TRANSPILE │──X──────┤
          └────┬─────┘         │
               ▼               │
          ┌──────────┐         │
          │ OPTIMIZE │──X──────┤
          └────┬─────┘         │
               ▼               │
          ┌──────────┐         │
          │ VALIDATE │──X──────┘
          └────┬─────┘
               ▼
          ┌──────────┐
          │  BUILD   │
          └────┬─────┘
               ▼
          ┌──────────┐
          │ COMPLETE │
          └──────────┘
```

## Phase Transitions

| From | To | Condition |
|------|----|-----------|
| INIT | ANALYSIS | `batuta transpile` invoked |
| ANALYSIS | TRANSPILE | All files analyzed, TDG scored |
| TRANSPILE | OPTIMIZE | All files transpiled successfully |
| OPTIMIZE | VALIDATE | Backend selection complete |
| VALIDATE | BUILD | Traces match, tests pass |
| BUILD | COMPLETE | `cargo build --release` succeeds |
| Any | FAILED | Error in current phase |

## Error Recovery

When a phase fails, state is preserved up to the failure point:

```bash
# Check what failed
batuta status

# Fix the issue, then resume
batuta reset --phase validation
batuta validate --trace
```

## Parallel Sub-Tasks

Some sub-tasks within a phase run in parallel:

```
ANALYSIS:    language detection | dependency analysis | TDG scoring
TRANSPILE:   Python (depyler) | C (decy) | Shell (bashrs)
```

Cross-language dependencies enforce ordering within groups. All sub-tasks in a phase must complete before the next phase begins.

## State Persistence

Pipeline state is persisted as JSON in `.batuta/pipeline_state.json`:

```json
{
  "current_phase": "optimize",
  "status": "in_progress",
  "phases": {
    "analysis": { "status": "completed", "hash": "a1b2c3d4" },
    "transpilation": { "status": "completed", "hash": "e5f6a7b8" },
    "optimization": { "status": "in_progress" }
  }
}
```

The `hash` field enables cache invalidation: if source files change, affected phases are re-run.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
