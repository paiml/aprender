# Risk Assessment

Before migrating any module, quantify the risk. Batuta provides automated scoring through TDG analysis and PMAT quality metrics to identify which modules are safe to migrate and which need extra attention.

## Complexity Scoring

Each module receives a composite risk score based on measurable factors:

| Metric | Low Risk (0-3) | Medium Risk (4-6) | High Risk (7-10) |
|--------|----------------|--------------------|--------------------|
| Cyclomatic complexity | < 10 | 10-25 | > 25 |
| Lines of code | < 200 | 200-1000 | > 1000 |
| External dependencies | 0-2 | 3-5 | > 5 |
| Unsafe operations | None | Bounded | Pervasive |
| Test coverage | > 80% | 50-80% | < 50% |

Run the assessment:

```bash
batuta analyze --tdg /path/to/project
```

## Critical Path Identification

Map dependencies between modules to find the critical path -- the chain of modules where a failure would block the entire migration.

```bash
# Visualize module dependency graph
batuta analyze --dependencies --format dot /path/to/project | dot -Tpng -o deps.png
```

Modules on the critical path require:
- Higher test coverage before migration (95%+)
- Dual-stack testing (original and transpiled running simultaneously)
- Explicit rollback plans

## Risk Mitigation Strategies

### For High-Complexity Modules

Break them down before migrating. Extract pure functions first:

```python
# Before: monolithic function (high risk)
def process_data(raw_input):
    parsed = parse(raw_input)       # Pure - migrate first
    validated = validate(parsed)     # Pure - migrate second
    result = save_to_db(validated)   # I/O - migrate last
    return result
```

### For Modules with Low Test Coverage

Write characterization tests in the source language before transpiling:

```bash
# Generate test scaffolding from runtime behavior
batuta analyze --characterize ./src/legacy_module.py
```

### For Modules with Many Dependencies

Use the strangler fig pattern. Create a Rust facade that delegates to the original, then replace internals one at a time.

## Fallback Planning

Every module migration needs a documented fallback:

| Risk Level | Fallback Strategy |
|------------|-------------------|
| Low | Git revert to pre-migration commit |
| Medium | Feature flag toggling old/new implementation |
| High | Parallel deployment with traffic splitting |
| Critical | Full rollback plan with data migration reversal |

## Tracking Risk Over Time

Use `batuta stack quality` to monitor risk scores as the migration progresses. A rising risk score on a module means the migration is introducing complexity rather than reducing it -- a signal to stop and reassess.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
