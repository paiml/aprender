# State Inspection

Batuta persists pipeline state in the `.batuta/` directory. Inspecting this state reveals what happened at each phase when the pipeline behaves unexpectedly.

## The `.batuta/` Directory

```
.batuta/
├── pipeline_state.json     # Current phase and status
├── analysis/
│   ├── languages.json      # Detected languages and line counts
│   ├── dependencies.json   # Dependency graph
│   └── tdg_scores.json     # TDG grades per file
├── transpilation/
│   ├── tool_selection.json # Which transpiler per file
│   ├── errors.json         # Transpilation errors
│   └── mapping.json        # Source-to-output file mapping
├── optimization/
│   └── backend.json        # Backend selection decisions
├── validation/
│   ├── traces/             # renacer trace files
│   └── comparison.json     # Trace diff results
└── cache/
    ├── tool_versions.json  # Cached transpiler versions
    └── dep_mapping.json    # Cached dependency mappings
```

## Inspecting Pipeline State

```bash
cat .batuta/pipeline_state.json
```

```json
{
  "current_phase": "validation",
  "status": "failed",
  "phases": {
    "analysis": { "status": "completed", "duration_ms": 1234 },
    "transpilation": { "status": "completed", "duration_ms": 5678 },
    "validation": { "status": "failed", "error": "trace_mismatch" }
  }
}
```

## Common Inspection Commands

```bash
# Find files that failed transpilation
cat .batuta/transpilation/errors.json | jq '.errors[]'

# Check TDG scores for failing modules
cat .batuta/analysis/tdg_scores.json | jq '.[] | select(.grade == "F")'

# Check backend selection decisions
cat .batuta/optimization/backend.json
```

## Cache Invalidation

| Symptom | Cache to Clear |
|---------|---------------|
| Wrong transpiler version | `rm .batuta/cache/tool_versions.json` |
| Dependency mapping stale | `rm .batuta/cache/dep_mapping.json` |
| Pipeline uses stale data | `rm -rf .batuta/analysis/` |

## Resetting Pipeline State

```bash
# Reset a single phase
batuta reset --phase validation

# Reset the entire pipeline
batuta reset
```

Prefer `batuta reset` over manual deletion -- it handles state transitions correctly.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
