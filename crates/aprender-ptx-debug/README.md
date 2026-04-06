# trueno-ptx-debug

Pure Rust PTX debugging and static analysis tool.

## Philosophy

Pure Rust PTX Analysis - Zero CUDA SDK Dependency.
Following Popper's falsificationism: we cannot prove PTX correct, but we can systematically attempt to falsify it.

## Features

- **PTX Parser**: Full lexer and AST construction for PTX source.
- **Type Checker**: Validates register types match operations.
- **Control Flow Analyzer**: CFG construction, barrier safety analysis.
- **Data Flow Analyzer**: Detects critical JIT bugs (F081, F082).
- **Address Space Validator**: Detects generic shared access patterns.
- **100-Point Falsification Framework**: 90+ tests across 10 categories.
- **Output Generation**: HTML reports and FKR test generation.
- **CLI Interface**: `analyze` and `gen-fkr` commands.

## Installation

```bash
cargo install --path trueno-ptx-debug
```

## Usage

### Analyze PTX file

```bash
trueno-ptx-debug analyze kernel.ptx --falsify --html report.html
```

### Generate FKR tests

```bash
trueno-ptx-debug gen-fkr kernel.ptx -o tests/kernel_fkr.rs
```

## License

MIT
