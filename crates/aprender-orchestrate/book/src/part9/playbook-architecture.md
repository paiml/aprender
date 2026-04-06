# Playbook Architecture

The playbook module implements deterministic pipeline orchestration with BLAKE3 content-addressable caching. This chapter covers the internal architecture and data flow.

## Module Structure

```
src/playbook/
  mod.rs          Public API and re-exports
  types.rs        All serde types (Playbook, Stage, LockFile, PipelineEvent, etc.)
  parser.rs       YAML parsing and structural validation
  template.rs     {{params.X}}, {{deps[N].path}}, {{outs[N].path}} resolution
  dag.rs          DAG construction from deps/outs + after edges
  hasher.rs       BLAKE3 hashing for files, directories, params, commands
  cache.rs        Lock file persistence and cache decision logic
  executor.rs     Local sequential executor with Jidoka failure policy
  eventlog.rs     Append-only JSONL event log
```

## Data Flow

```
playbook.yaml
       │
       ▼
   ┌────────┐     ┌──────────┐     ┌─────────┐
   │ parser │────▶│ validate │────▶│ dag.rs  │
   └────────┘     └──────────┘     └─────────┘
                                        │
                                   topo_order
                                        │
                                        ▼
                              ┌──────────────────┐
                              │   executor loop   │
                              │  (per stage)      │
                              └──────┬───────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
        ┌──────────┐          ┌──────────┐          ┌──────────┐
        │ template │          │ hasher   │          │ cache    │
        │ resolve  │          │ hash deps│          │ check    │
        └──────────┘          │ hash cmd │          └──────────┘
                              │ hash parm│               │
                              └──────────┘          Hit / Miss
                                                        │
                                    ┌───────────────────┤
                                    ▼                   ▼
                              ┌──────────┐        ┌──────────┐
                              │  CACHED  │        │ execute  │
                              │  (skip)  │        │ sh -c    │
                              └──────────┘        └──────────┘
                                                       │
                                                       ▼
                                                ┌──────────┐
                                                │ hash outs│
                                                │ update   │
                                                │ lock     │
                                                └──────────┘
```

## Key Components

### types.rs — Type System

All types derive `Serialize` and `Deserialize` for YAML/JSON roundtripping.

- **`Playbook`**: Root type. Uses `IndexMap<String, Stage>` to preserve YAML ordering.
- **`Stage`**: Pipeline stage with `cmd`, `deps`, `outs`, `after`, `params`, `frozen`.
- **`Policy`**: Uses typed enums (`FailurePolicy`, `ValidationPolicy`) instead of strings.
- **`LockFile`**: Per-stage BLAKE3 hashes in `IndexMap<String, StageLock>`.
- **`PipelineEvent`**: Tagged enum for JSONL event log entries.
- **`InvalidationReason`**: Enum with `Display` impl for human-readable cache miss explanations.

Global parameters use `HashMap<String, serde_yaml::Value>` to support strings, numbers, and booleans without type coercion.

### parser.rs — Validation

Structural validation catches errors before execution:

1. Version must be `"1.0"`
2. Stage `cmd` must not be empty
3. `after` references must exist and not self-reference
4. Template references (`{{params.X}}`) must resolve against declared params
5. `{{deps[N].path}}` indices must be in range

Warnings (non-fatal) are emitted for stages with no outputs.

### dag.rs — DAG Construction

Two types of edges build the execution graph:

1. **Implicit data edges**: An output path produced by stage A that appears as a dependency of stage B creates an edge A → B.
2. **Explicit `after` edges**: `after: [A]` on stage B creates A → B.

Kahn's topological sort with deterministic tie-breaking (alphabetical) produces the execution order. Cycles are detected and reported with the participating stage names.

### hasher.rs — BLAKE3 Hashing

All hashes are formatted as `"blake3:{hex}"`.

| Function | Input | Strategy |
|----------|-------|----------|
| `hash_file` | Single file | 64KB streaming I/O |
| `hash_directory` | Directory | Sorted walk, relative paths included in hash |
| `hash_cmd` | Resolved command string | Direct BLAKE3 |
| `hash_params` | Global params + referenced keys | Sorted key=value pairs |
| `compute_cache_key` | cmd_hash + deps_hash + params_hash | Composite BLAKE3 |

**Granular parameter invalidation**: `effective_param_keys()` computes the union of explicitly declared `stage.params` keys and template-extracted references (`{{params.X}}`). Only referenced parameters contribute to the stage's params hash.

Symlinks are skipped during directory walks to prevent circular references and symlink attacks.

### cache.rs — Cache Decisions

The `check_cache()` function returns `CacheDecision::Hit` or `CacheDecision::Miss { reasons }`.

Check order:
1. `--force` flag → immediate Miss (Forced)
2. Upstream stage re-run → Miss (UpstreamRerun)
3. No lock file → Miss (NoLockFile)
4. Stage not in lock → Miss (StageNotInLock)
5. Previous run incomplete → Miss (PreviousRunIncomplete)
6. Cache key mismatch → Miss with detailed component breakdown (CmdChanged, DepChanged, ParamsChanged)
7. Output files missing → Miss (OutputMissing)
8. All checks pass → Hit

Lock files are written atomically via temp file + rename to prevent corruption from interrupted writes.

### executor.rs — Orchestration

The executor implements the full lifecycle:

```
for stage in topo_order:
    1. Check frozen → CACHED
    2. Resolve template variables
    3. Hash command, deps, params
    4. Compute composite cache_key
    5. Check cache → Hit: skip, Miss: execute
    6. Execute via sh -c
    7. Hash outputs
    8. Update lock file entry
    9. Append event log entry
```

**Jidoka** (stop-on-first-failure): When `policy.failure == StopOnFirst`, the executor saves a partial lock file and halts immediately on any stage failure. This prevents cascading failures and preserves the ability to resume from the last good state.

**Localhost targets** are allowed for Phase 1. Remote hosts return an error directing users to Phase 2.

### eventlog.rs — Audit Trail

Events are appended as newline-delimited JSON (JSONL) to a `.events.jsonl` file. Each event is wrapped in a `TimestampedEvent` with ISO 8601 timestamp. Run IDs (`r-{hex}`) correlate events within a single pipeline execution.

## Invariants

| ID | Invariant | Enforced By |
|----|-----------|-------------|
| I1 | Deterministic ordering | IndexMap + sorted toposort |
| I2 | Content-addressable cache | BLAKE3 composite key |
| I3 | Granular param invalidation | effective_param_keys() |
| I4 | Atomic lock writes | temp file + rename |
| I5 | Upstream propagation | rerun_stages tracking |
| I6 | Frozen immutability | frozen flag check before cache |

## Phase 1 Scope

Phase 1 delivers local sequential execution. The following are defined in the type system but not yet executed:

| Feature | Phase | Type |
|---------|-------|------|
| Remote dispatch (repartir) | 2 | `Target.host` |
| Parallel fan-out | 2 | `ParallelConfig` |
| Retry with backoff | 2 | `RetryConfig` |
| Shell purification (bashrs) | 2 | `ShellMode` |
| Resource scheduling | 4 | `ResourceConfig` |
| Compliance gates (pmat) | 5 | `Compliance` |
