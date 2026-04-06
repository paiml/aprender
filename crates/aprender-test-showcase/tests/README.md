# Calculator Test Suite

This document describes the comprehensive testing strategy for the showcase calculator.

## Test Categories

### Unit Tests (672 tests)
Located in `src/` modules with inline `#[cfg(test)]` modules.

- **Core Logic** (`src/core/`): Parser, evaluator, operations, history
- **TUI Components** (`src/tui/`): App state, input handling, keypad, UI rendering
- **WASM Bindings** (`src/wasm/`): Browser integration, DOM manipulation, WASM keypad
- **Probar Integration** (`src/probar_tests.rs`): Page objects, accessibility, fixtures

### Integration Tests (`tests/`)
- `gui_coverage_tests.rs`: GUI element coverage verification
- `keypad_proptests.rs`: Property-based testing for keypad invariants

### Playbook Tests (`playbooks/`)
State machine verification using probar playbooks:
- `calculator.yaml`: Main calculator flow (idle -> result)
- `error_handling.yaml`: Error state handling (division by zero)

## Running Tests

```bash
# Run all tests
cargo test -p showcase-calculator --all-features

# Run with coverage
cargo tarpaulin -p showcase-calculator

# Run specific test category
cargo test -p showcase-calculator --test gui_coverage_tests
cargo test -p showcase-calculator --test keypad_proptests

# Run probar playbook validation
probador playbook playbooks/calculator.yaml --validate
probador playbook playbooks/error_handling.yaml --validate
```

## Test Rationale

### Why Property-Based Testing?
The keypad has 20 buttons with specific invariants:
- All digits 0-9 must exist
- All operators +, -, *, /, %, ^ must exist
- Each button has a unique position
- Layout is always 5x4

Property tests verify these invariants hold for all possible inputs.

### Why State Machine Playbooks?
The calculator has distinct states:
1. `idle` - Initial state, display shows 0
2. `entering_first` - User entering first operand
3. `operator_selected` - Operator chosen, awaiting second operand
4. `entering_second` - User entering second operand
5. `showing_result` - Calculation complete
6. `error` - Invalid operation (e.g., division by zero)

Playbooks verify:
- All valid transitions work correctly
- Forbidden transitions are blocked
- Invariants hold in each state

### Why Accessibility Testing?
The calculator must be usable by everyone:
- WCAG 2.1 AA color contrast (4.5:1 minimum)
- Keyboard navigation (Tab, Enter, Escape, digits, operators)
- Screen reader support (ARIA labels, live regions)
- Focus indicators visible

## Coverage Targets

| Metric | Target | Current |
|--------|--------|---------|
| Line Coverage | 95% | 94.2% |
| Branch Coverage | 85% | 82% |
| GUI Element Coverage | 100% | 100% |
| State Coverage | 100% | 100% |
| Mutation Score | 80% | 85.3% |

## Deterministic Replay

Recordings in `recordings/` directory capture:
- `happy_path.json`: Successful calculation
- `error_path.json`: Error handling flow
- `edge_case_*.json`: Boundary conditions

These enable:
- Regression testing
- Bug reproduction
- Performance benchmarking
