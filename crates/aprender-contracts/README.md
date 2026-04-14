# aprender-contracts

[![Crates.io](https://img.shields.io/crates/v/aprender-contracts.svg)](https://crates.io/crates/aprender-contracts)
[![docs.rs](https://docs.rs/aprender-contracts/badge.svg)](https://docs.rs/aprender-contracts)
[![CI](https://github.com/paiml/aprender/actions/workflows/ci.yml/badge.svg)](https://github.com/paiml/aprender/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Provable contracts for the [Aprender](https://github.com/paiml/aprender) ML framework — YAML contract parsing, validation, lint reporting, and falsification scaffolding. 968 contract files validated, 1,371 tests.

> Previously published as `provable-contracts`.

## Install

```toml
[dependencies]
aprender-contracts = "0.29"
```

Or with cargo-add:

```bash
cargo add aprender-contracts
```

## Quick Start

```rust
use aprender_contracts::{ContractLoader, LintReport};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load and validate all contracts in a directory
    let loader = ContractLoader::from_dir("contracts/")?;
    let report: LintReport = loader.lint()?;

    if report.has_errors() {
        eprintln!("{report}");
        std::process::exit(1);
    }

    println!("All {} contracts valid.", report.contract_count());
    Ok(())
}
```

```rust
use aprender_contracts::FalsificationTest;

// Generate falsification scaffold for a contract
let contract = loader.get("linear-regression-v1")?;
let tests: Vec<FalsificationTest> = contract.falsification_tests();
for test in &tests {
    println!("Precondition: {}", test.precondition);
    println!("Postcondition: {}", test.postcondition);
}
```

## Key Types

| Type | Description |
|---|---|
| `ContractLoader` | Loads and indexes YAML contracts from a directory tree |
| `LintReport` | Aggregated lint results: errors, warnings, contract count |
| `FalsificationTest` | A single (precondition, postcondition) pair for Kani harness generation |
| `ContractSchema` | Parsed representation of one YAML contract file |

## Features

- YAML contract parsing with JSON Schema validation
- Lint rules: missing falsification conditions, placeholder preconditions, schema conformance
- Falsification test scaffolding for [Kani](https://github.com/model-checking/kani) harness generation
- Used by `apr qa` to enforce 405 provable contracts across all 70 crates

## Documentation

- [API docs (docs.rs)](https://docs.rs/aprender-contracts)
- [Full monorepo](https://github.com/paiml/aprender)
- [Contract files](https://github.com/paiml/aprender/tree/main/contracts)

## License

MIT. See [LICENSE](../../LICENSE).
