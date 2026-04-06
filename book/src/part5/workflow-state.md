# Workflow State Management

Batuta tracks progress through its 5-phase pipeline in a JSON state file. This allows you to resume from the last successful phase after an interruption or failure.

## State File

Pipeline state is persisted to `.batuta-state.json` in the current working directory. The file is created automatically when the first pipeline command runs.

```json
{
  "current_phase": "Transpilation",
  "phases": {
    "Analysis": { "status": "Completed", "started_at": "...", "completed_at": "..." },
    "Transpilation": { "status": "InProgress", "started_at": "..." },
    "Optimization": { "status": "NotStarted" },
    "Validation": { "status": "NotStarted" },
    "Deployment": { "status": "NotStarted" }
  }
}
```

## Phase Tracking

Each phase has one of four statuses:

| Status | Meaning |
|--------|---------|
| `NotStarted` | Phase has not been attempted |
| `InProgress` | Phase is currently running |
| `Completed` | Phase finished successfully |
| `Failed` | Phase encountered an error (message stored in `error` field) |

Batuta records `started_at` and `completed_at` timestamps for every transition.

## Viewing Status

Use `batuta status` to display phase statuses, timestamps, durations, and the recommended next step.

```bash
batuta status
```

## Resuming from a Failed Phase

If a phase fails, Batuta records the error and stops (Jidoka principle). Fix the issue, then re-run the same command. Completed phases are not repeated.

```bash
# Phase 2 failed -- fix the source, then re-run
batuta transpile
```

## Reset and Clean

To discard all progress and start from scratch:

```bash
batuta reset         # Interactive confirmation
batuta reset --yes   # Skip confirmation
```

The reset command deletes `.batuta-state.json` but does not remove generated source code. To remove both:

```bash
batuta reset --yes
rm -rf ./rust-output
```

## Progress Percentage

Progress is the fraction of phases with `Completed` status, displayed by `batuta status`.

| Completed Phases | Progress |
|-----------------|----------|
| 0 of 5 | 0% |
| 1 of 5 | 20% |
| 3 of 5 | 60% |
| 5 of 5 | 100% |

---

**Navigate:** [Table of Contents](../SUMMARY.md)
