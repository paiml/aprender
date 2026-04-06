# Trueno-DB Ticket System

This project uses **pmat work** for unified GitHub/YAML ticket management.

## Overview

**pmat** provides a workflow that integrates:
- **GitHub Issues** (e.g., `#8`, `#42`)
- **YAML Tickets** (e.g., `CORE-001`, `PERF-042`) in `docs/roadmaps/roadmap.yaml`

## Quick Start

```bash
# View all tickets
cat docs/roadmaps/roadmap.yaml

# Start work on a ticket
pmat work start CORE-001

# Continue work
pmat work continue CORE-001

# Complete with quality checks
pmat work complete CORE-001

# Check status
pmat work status
```

## Phase 1 MVP Tickets (Current)

### Critical Path (High Priority)

1. **CORE-001**: Arrow storage backend with morsel-based paging
   - **Toyota Way**: Poka-Yoke (prevent VRAM OOM)
   - **Acceptance**: Out-of-core execution for datasets > VRAM

2. **CORE-002**: Cost-based backend dispatcher
   - **Toyota Way**: Genchi Genbutsu (physics-based 5x rule)
   - **Acceptance**: GPU only if `compute_time > 5 * transfer_time`

3. **CORE-003**: JIT WGSL compiler for kernel fusion
   - **Toyota Way**: Muda elimination (no intermediate memory)
   - **Acceptance**: Fused filter+aggregation in single pass

4. **CORE-004**: GPU kernels (sum, avg, count, min, max)
   - **Target**: 50-100x faster than CPU for 100M+ rows
   - **Acceptance**: Parallel reduction benchmarks

5. **CORE-005**: SIMD fallback via Trueno
   - **Toyota Way**: Heijunka (prevent blocking Tokio reactor)
   - **Acceptance**: `spawn_blocking` for CPU-bound operations

6. **CORE-006**: Backend equivalence tests ⚠️ CRITICAL
   - **Toyota Way**: Jidoka (built-in quality)
   - **Acceptance**: GPU == SIMD == Scalar (property-based tests)

### Supporting Tasks

7. **CORE-007**: SQL parser (SELECT, WHERE, GROUP BY, aggregations)
   - **Priority**: Medium
   - **Acceptance**: 100+ parser test cases

8. **CORE-008**: PCIe transfer benchmarks
   - **Toyota Way**: Genchi Genbutsu (measure, don't guess)
   - **Acceptance**: Empirically validate 5x rule

9. **CORE-009**: Competitive benchmarks (vs DuckDB, SQLite, Polars)
   - **Toyota Way**: Kaizen (prove all claims)
   - **Acceptance**: TPC-H style analytics workload

## Workflow Commands

### Start a Ticket
```bash
pmat work start CORE-001

# With specification file
pmat work start CORE-001 --with-spec

# As epic with subtasks
pmat work start CORE-001 --epic
```

This will:
- Mark ticket as "in_progress"
- Create feature branch (e.g., `feature/CORE-001`)
- Optionally create `docs/specifications/CORE-001-arrow-storage.md`

### Continue Work
```bash
pmat work continue CORE-001
```

This will:
- Resume work on existing ticket
- Checkout feature branch
- Show progress status

### Complete Ticket
```bash
pmat work complete CORE-001
```

This will:
- Run quality gates (tests, lint, coverage)
- Mark ticket as "completed"
- Update roadmap.yaml
- Optionally create GitHub PR

### Check Status
```bash
pmat work status
```

Shows:
- All tickets with status (pending, in_progress, completed)
- Current active ticket
- Quality gate results

## Quality Gates (EXTREME TDD)

When completing a ticket with `pmat work complete`, these gates are enforced:

- ✅ `cargo test --all-features` (100% pass)
- ✅ `cargo clippy -- -D warnings` (zero tolerance)
- ✅ `cargo llvm-cov` (>90% coverage)
- ✅ `pmat analyze tdg` (≥B+ / 85)
- ✅ `cargo mutants` (≥80% kill rate)

## Roadmap Structure

`docs/roadmaps/roadmap.yaml`:
```yaml
roadmap:
  - id: CORE-001
    title: "Arrow storage backend with morsel-based paging"
    description: |
      Implement Arrow/Parquet storage with 128MB morsel-based paging.
    status: pending  # or: in_progress, completed
    priority: high   # or: critical, medium, low
    phase: 1
    labels: [storage, poka-yoke, phase-1]
    acceptance_criteria:
      - Parquet reader with Arrow columnar format
      - 128MB morsel chunks
    references:
      - "Funke et al. (2018): GPU paging"
```

## GitHub Integration

If GitHub is enabled (`github_enabled: true`):

```bash
# Sync with GitHub issues
pmat work sync

# Create GitHub issue from YAML ticket
pmat work start CORE-001 --create-github
```

## Toyota Way Alignment

Each ticket includes:
- **Toyota Way Principle**: Which principle it addresses (Muda, Poka-Yoke, etc.)
- **Academic References**: Peer-reviewed research backing the design
- **Acceptance Criteria**: Measurable success metrics
- **Quality Gates**: Enforced on completion

## Next Steps

1. Start with **CORE-001** (Arrow storage) - foundation for all other work
2. Then **CORE-002** (Cost-based dispatcher) - critical for proper backend selection
3. Then **CORE-006** (Backend equivalence) - safety net before GPU work
4. Then **CORE-004** (GPU kernels) - the performance core
5. Finally **CORE-003** (JIT fusion) - optimization layer

## Example Session

```bash
# Start work on storage backend
pmat work start CORE-001 --with-spec

# ... implement Arrow storage with morsel paging ...

# Run quality checks
make quality-gate

# Complete ticket (runs all gates automatically)
pmat work complete CORE-001

# Move to next ticket
pmat work start CORE-002
```

## File Structure

```
trueno-db/
├── docs/
│   ├── roadmaps/
│   │   └── roadmap.yaml          # All tickets (YAML format)
│   └── specifications/
│       ├── db-spec-v1.md         # Main specification
│       └── CORE-001-*.md         # Per-ticket specs (if --with-spec used)
├── .git/
│   └── hooks/
│       └── commit-msg            # Auto-installed by pmat work init
└── ...
```

## Tips

- **Start small**: Begin with CORE-001 to understand the workflow
- **Use --with-spec**: Creates detailed specification for complex tickets
- **Quality gates**: Run `make quality-gate` before completing tickets
- **Sync regularly**: Use `pmat work sync` if using GitHub integration
- **Track progress**: `pmat work status` shows overall roadmap health
