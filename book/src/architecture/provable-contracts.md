<!-- PCU: architecture-provable-contracts | contract: contracts/apr-page-architecture-provable-contracts-v1.yaml -->
<!-- Example: cargo run -p aprender-core --example none -->
<!-- Status: enforced -->

# Provable Contracts

Aprender uses **provable contracts** — YAML files that define equations,
preconditions, postconditions, and falsification tests for every kernel
and CLI command.

## 405 Contracts

```bash
ls contracts/*.yaml | wc -l
# 405
```

## Contract Structure

```yaml
metadata:
  version: 1.0.0
  description: Softmax numerical stability
  references:
  - "Goodfellow et al., Deep Learning, MIT Press 2016"

equations:
  softmax_stability:
    formula: |
      softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    invariants:
    - "output sums to 1.0 (within f32 epsilon)"
    - "max-subtraction prevents overflow"

falsification_tests:
- id: FALSIFY-SOFTMAX-001
  prediction: softmax([1000, 1000, 1000]) does not overflow
  test: cargo test softmax_large_values
```

## Enforcement

Contracts are enforced at three levels:

1. **Compile time**: `#[contract]` proc macro from `aprender-contracts-macros`
2. **Test time**: `FALSIFY-*` tests verify predictions
3. **CI time**: `pv lint contracts/` checks schema validity

## CLI Contract

Every `apr` subcommand has a contract entry in `contracts/apr-cli-commands-v1.yaml`:

```bash
apr mono audit  # Runs all contract checks
```

See: [Contract spec](../../contracts/apr-cli-commands-v1.yaml)
