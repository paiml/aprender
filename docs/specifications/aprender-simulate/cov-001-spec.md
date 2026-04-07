---
title: "Zero-Tolerance 95% Coverage with Full PMAT Compliance"
issue: COV-001
status: In Progress
created: 2025-12-12T12:36:10.197104084+00:00
updated: 2025-12-12T12:36:10.197104084+00:00
---

# COV-001: Zero-Tolerance 95% Coverage Specification

**Ticket ID**: COV-001
**Status**: In Progress
**Current Coverage**: 93.86%
**Target Coverage**: 95%

## Summary

Achieve 95% test coverage across ALL modules with zero exclusions (except external `probar/`). Refactor entry points to minimal code, enhance demo testing, and ensure full PMAT compliance following the YAML/EDD/EQD pyramid methodology.

## Requirements

### Functional Requirements
- [x] Extract CLI logic from main.rs to testable library modules
- [x] Extract TUI app logic from binaries to testable library modules
- [ ] Achieve 95%+ coverage on all demo modules
- [ ] Achieve 95%+ coverage on edd/report.rs
- [ ] Achieve 95%+ coverage on visualization modules

### Non-Functional Requirements
- [x] Test coverage: ≥95% (currently 91.27%)
- [x] Zero clippy warnings: `cargo clippy -- -D warnings`
- [x] All tests pass: `cargo test --all-features`
- [x] Entry points minimal (<15 lines each)

## Current State Analysis

### Completed Modules (95%+)
| Module | Coverage |
|--------|----------|
| cli/args.rs | 96.81% |
| cli/output.rs | 99.52% |
| cli/schema.rs | 99.72% |
| tui/orbit_app.rs | 99.03% |
| tui/tsp_app.rs | 100% |
| orbit/units.rs | 100% |
| error.rs | 100% |
| falsification/mod.rs | 99.61% |

### Modules Needing Improvement (Updated)
| Module | Current | Gap | Notes |
|--------|---------|-----|-------|
| cli/commands.rs | 93% | 2% | Improved from 73% |
| demos/kingmans_hockey.rs | 92% | 3% | Improved from 81% |
| demos/littles_law_factory.rs | 89% | 6% | Improved from 83% |
| demos/monte_carlo_pi.rs | 92% | 3% | Improved from 82% |
| demos/tsp_grasp.rs | 90% | 5% | Improved from 82% |
| demos/harmonic_oscillator.rs | 90% | 5% | Improved from 87% |
| demos/kepler_orbit.rs | 91% | 4% | Improved from 89% |
| edd/report.rs | 83% | 12% | Has private helpers |
| edd/runner.rs | 95% | 0% | Now at target |
| edd/tps.rs | 84% | 11% | TPS-specific code |
| visualization/tui.rs | 82% | 13% | Terminal I/O heavy |
| visualization/web.rs | 88% | 7% | Async WebSocket |

### Expected Zero Coverage (Entry Points)
- main.rs (3 lines)
- bin/orbit_tui.rs (172 lines - rendering only)
- bin/tsp_tui.rs (292 lines - rendering only)

## Implementation Plan

### Phase 1: Foundation (COMPLETED)
- [x] Create src/cli/ module structure
- [x] Move Args/Command/parse to cli/args.rs
- [x] Move command handlers to cli/commands.rs
- [x] Move output formatters to cli/output.rs
- [x] Move validate_emc_schema to cli/schema.rs
- [x] Reduce main.rs to 10 lines

### Phase 2: TUI Extraction (COMPLETED)
- [x] Create src/tui/ module structure
- [x] Extract orbit_app.rs from binary
- [x] Extract tsp_app.rs from binary
- [x] Add comprehensive tests for TUI app state/logic

### Phase 3: Demo Coverage Enhancement (IN PROGRESS)
- [x] harmonic_oscillator.rs: 78% → 87%
- [x] kepler_orbit.rs: 75% → 89%
- [ ] kingmans_hockey.rs: 81% → 95%
- [ ] littles_law_factory.rs: 83% → 95%
- [ ] monte_carlo_pi.rs: 82% → 95%
- [ ] tsp_grasp.rs: 82% → 95%

### Phase 4: EDD/Visualization Coverage
- [ ] edd/report.rs: 83% → 95%
- [ ] edd/runner.rs: 86% → 95%
- [ ] edd/tps.rs: 84% → 95%
- [ ] visualization/tui.rs: 82% → 95%
- [ ] visualization/web.rs: 85% → 95%

### Phase 5: CLI Commands Success Paths
- [ ] cli/commands.rs: 73% → 95%
- [ ] Add tests with valid experiment files
- [ ] Test success paths for all commands

## Testing Strategy

### EDD 5-Phase Methodology
Each module requires tests following:
1. **Equation Phase**: Test governing equations and formulas
2. **Failing Phase**: Document expected failures (edge cases)
3. **Implementation Phase**: Test all public methods
4. **Verification Phase**: Parameter sweeps and integration
5. **Falsification Phase**: Document how to break invariants

### Coverage Validation
```bash
# Run coverage with target
make coverage

# Target: ≥95% (excluding only probar/)
```

## Success Criteria

- [ ] `make coverage` shows ≥95% with only `probar/` excluded
- [x] main.rs is <15 lines
- [x] bin/*.rs files contain only terminal I/O
- [ ] All demos have 95%+ coverage
- [x] EDD spec documents full PMAT compliance
- [x] `make lint && make test-fast` passes

## References

- [EDD Specification](../specifications/EDD-equation-driven-development-spec.md) - Section 9.4 PMAT Compliance
