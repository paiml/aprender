# Workflow Overview

> **"A conductor doesn't play all instruments at once. Each section performs in sequence, building upon the previous. So too with code migration."**

## The 5-Phase Workflow

Batuta enforces a strict **5-phase Kanban workflow**. You cannot skip phases. You cannot run phases out of order. This is not a limitation - it's a **quality guarantee**.

```
┌──────────────────────────────────────────────────────────────────┐
│                    BATUTA 5-PHASE WORKFLOW                        │
└──────────────────────────────────────────────────────────────────┘

Phase 1: Analysis (20%)
├─ Language detection
├─ Dependency analysis
├─ Technical Debt Grade (TDG)
├─ ML framework identification
└─ Transpiler recommendation
      ↓
Phase 2: Transpilation (40%)
├─ Tool selection (Decy/Depyler/Bashrs)
├─ Code conversion
├─ Type inference
├─ Ownership analysis
└─ Initial Rust generation
      ↓
Phase 3: Optimization (60%)
├─ SIMD vectorization (Trueno)
├─ GPU dispatch (Trueno)
├─ Memory layout optimization
└─ MoE backend selection
      ↓
Phase 4: Validation (80%)
├─ Syscall tracing (Renacer)
├─ Output comparison
├─ Test suite execution
└─ Performance benchmarking
      ↓
Phase 5: Deployment (100%)
├─ Release build
├─ Cross-compilation
├─ WebAssembly target
└─ Distribution packaging
```

## Phase Dependencies

**Why enforce order?**

Consider what happens if you skip Analysis:

```bash
# ❌ Without Analysis
$ batuta transpile
Error: Don't know what language this is!
Error: Don't know which transpiler to use!
Error: Don't know about dependencies!
```

**Each phase builds on the previous:**

| Phase | Consumes | Produces |
|-------|----------|----------|
| Analysis | Source files | Language map, dependency graph, TDG score |
| Transpilation | Language map | Rust code, type signatures, ownership info |
| Optimization | Rust code | Optimized Rust, SIMD/GPU annotations |
| Validation | Original + optimized | Test results, syscall traces, benchmarks |
| Deployment | Validated Rust | Binary artifacts, distribution packages |

## State Persistence

**Every phase updates `.batuta-state.json`:**

```json
{
  "current_phase": "Transpilation",
  "phases": {
    "Analysis": {
      "status": "Completed",
      "started_at": "2025-11-19T14:21:32Z",
      "completed_at": "2025-11-19T14:21:33Z",
      "duration": "0.13s"
    },
    "Transpilation": {
      "status": "InProgress",
      "started_at": "2025-11-19T14:22:15Z"
    },
    "Optimization": {
      "status": "NotStarted"
    },
    ...
  }
}
```

**Benefits:**

1. **Resume after errors:** Fix the problem, run same command
2. **Track progress:** Know exactly where you are
3. **Performance analysis:** See which phases take longest
4. **Audit trail:** Complete history of migration

## Workflow Commands

### Start Fresh

```bash
# Reset everything
$ batuta reset --yes
✅ Workflow state reset successfully!

# Begin migration
$ batuta status
No workflow started yet.

💡 Get started:
  1. Run batuta analyze to analyze your project
```

### Run Full Pipeline

```bash
# Standard workflow (all phases in sequence)
$ batuta analyze --languages --dependencies --tdg
$ batuta init --source ./my-python-app
$ batuta transpile --incremental --cache
$ batuta optimize --enable-gpu --profile aggressive
$ batuta validate --trace-syscalls --benchmark
$ batuta build --release
```

### Check Progress Anytime

```bash
$ batuta status

📊 Workflow Progress
──────────────────────────────────────────────
  ✓ Analysis [Completed]
  ✓ Transpilation [Completed]
  ⏳ Optimization [In Progress]
  ○ Validation [Not Started]
  ○ Deployment [Not Started]

  Overall: 60% complete

Phase Details:
──────────────────────────────────────────────

✓ Analysis
  Started: 2025-11-19 14:21:32 UTC
  Completed: 2025-11-19 14:21:33 UTC
  Duration: 0.13s

✓ Transpilation
  Started: 2025-11-19 14:22:15 UTC
  Completed: 2025-11-19 14:25:48 UTC
  Duration: 213.2s

⏳ Optimization
  Started: 2025-11-19 14:26:02 UTC
```

## Phase Entry Criteria

Each phase has **explicit entry criteria** that must be satisfied:

### Phase 1: Analysis
- **Entry:** Valid source directory
- **Exit:** Language map generated, dependencies resolved, TDG calculated

### Phase 2: Transpilation
- **Entry:** Analysis completed successfully
- **Exit:** All source files transpiled, code compiles, basic tests pass

### Phase 3: Optimization
- **Entry:** Transpilation completed, code compiles
- **Exit:** Optimizations applied, code still compiles, tests pass

### Phase 4: Validation
- **Entry:** Optimization completed
- **Exit:** Equivalence verified, benchmarks complete, acceptance criteria met

### Phase 5: Deployment
- **Entry:** Validation passed
- **Exit:** Binaries built, packaged, ready for distribution

## Error Handling

**Principle:** Fail fast, fail clearly, provide actionable guidance.

### Phase Failure Example

```bash
$ batuta transpile

🔄 Transpiling code...

✓ Loaded configuration
✓ Detected tools: Depyler (Python → Rust)
✓ Primary language: Python

❌ Transpilation failed!

Error: depyler exited with code 1
  File "complex_class.py", line 42
    Unsupported Python feature: metaclass with __prepare__

💡 Troubleshooting:
  • Simplify metaclass usage in complex_class.py
  • Use Ruchy for gradual migration of complex features
  • See: https://github.com/paiml/depyler/issues/23

📊 Workflow Progress
──────────────────────────────────────────────
  ✓ Analysis [Completed]
  ✗ Transpilation [Failed]  ← Fix this!
  ○ Optimization [Not Started]
  ○ Validation [Not Started]
  ○ Deployment [Not Started]

  Overall: 20% complete
```

**Note:** Phase status is "Failed", not "In Progress". This prevents downstream phases from using broken output.

## Workflow Patterns

### Pattern 1: Iterate on Single Phase

```bash
# Fix transpilation errors iteratively
$ batuta transpile
✗ Failed on module auth.py

# Fix auth.py manually or with Ruchy
$ batuta transpile --modules auth
✓ auth.py transpiled successfully

# Continue with full transpilation
$ batuta transpile
✓ All modules transpiled
```

### Pattern 2: Skip Completed Phases

```bash
# Workflow state persists
$ batuta status
Current phase: Optimization

# Running earlier phases does nothing
$ batuta analyze
ℹ️ Analysis already completed

# But you can force re-analysis
$ batuta analyze --force
⚠️  This will reset downstream phases!
Proceed? [y/N] y
```

### Pattern 3: Parallel Development

```bash
# Developer A works on transpilation
$ batuta transpile --modules frontend

# Developer B works on different modules
$ batuta transpile --modules backend

# Merge and complete
$ batuta transpile --modules shared
$ batuta status
✓ Transpilation: 100% complete
```

## Performance Characteristics

Typical phase durations (varies by project size):

| Phase | Small Project (<10K LOC) | Medium (10-100K LOC) | Large (100K+ LOC) |
|-------|--------------------------|----------------------|-------------------|
| Analysis | 0.1-0.5s | 1-5s | 10-30s |
| Transpilation | 5-30s | 1-10min | 10-60min |
| Optimization | 2-10s | 30s-5min | 5-30min |
| Validation | 1-5s | 10-60s | 2-20min |
| Deployment | 0.5-2s | 2-10s | 10-60s |
| **Total** | **~1min** | **~20min** | **~2hr** |

**Note:** Incremental compilation reduces re-transpilation time by 60-80%.

## Workflow Visualization

The workflow is a **state machine**:

```
    [Not Started]
         ↓
    start_phase()
         ↓
    [In Progress] ─── fail_phase() ───→ [Failed]
         ↓                                   ↑
    complete_phase()                         │
         ↓                                   │
    [Completed] ──── retry ─────────────────┘
```

**State transitions:**

| From | To | Trigger |
|------|--| --------|
| NotStarted | InProgress | `start_phase()` |
| InProgress | Completed | `complete_phase()` |
| InProgress | Failed | `fail_phase()` |
| Failed | InProgress | Retry after fixes |
| Completed | (stays) | Cannot regress without reset |

## Key Takeaways

✓ **5 phases, strict order:** No skipping, no reordering
✓ **State persistence:** Resume after errors, track progress
✓ **Quality gates:** Each phase validates previous output
✓ **Visual progress:** Always know where you are
✓ **Fail fast:** Errors stop pipeline, require fixes
✓ **Actionable errors:** Clear guidance on how to proceed

## Next Steps

Now let's dive deep into each phase, starting with Phase 1: Analysis.

---

**Previous:** [Toyota Way Principles](../part1/toyota-way.md)
**Next:** [Phase 1: Analysis](./phase1-analysis.md)
