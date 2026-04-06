# Supported Languages

Batuta supports transpilation from multiple source languages to Rust.

## Source Languages

| Language | Transpiler | Status | Features |
|----------|------------|--------|----------|
| **Python** | Depyler | ✅ Stable | Type inference, NumPy/sklearn/PyTorch |
| **Shell** | Bashrs | ✅ Stable | POSIX compliance, formal verification |
| **C/C++** | Decy | 🔄 Beta | Memory safety, ownership inference |

## Python Support (Depyler)

### Supported Constructs

- Functions and classes
- Type annotations (PEP 484)
- List/dict/set comprehensions
- Context managers (`with` statements)
- Decorators
- Async/await

### ML Library Mappings

| Python | Rust Equivalent |
|--------|-----------------|
| `numpy` | `trueno` |
| `sklearn` | `aprender` |
| `torch` | `realizar` |
| `pandas` | `polars` (via trueno) |

## Shell Support (Bashrs)

### Supported Features

- Variable assignment and expansion
- Control flow (if/else, for, while, case)
- Functions
- Pipelines and redirections
- Command substitution
- Arrays

### Shell Compatibility

| Shell | Support Level |
|-------|---------------|
| POSIX sh | Full |
| Bash 4.x | Full |
| Bash 5.x | Full |
| Zsh | Partial |

## C/C++ Support (Decy)

### Supported Constructs

- Functions and structs
- Pointers (with ownership inference)
- Arrays and strings
- Memory allocation/deallocation
- Header file parsing

### Safety Analysis

Decy performs automatic safety analysis:
- Buffer overflow detection
- Use-after-free detection
- Memory leak detection
- Null pointer dereference

## Target: Rust

All transpilation targets modern Rust (2021 edition) with:
- Full type safety
- Memory safety guarantees
- Zero-cost abstractions
- No unsafe code (where possible)

---

**Navigate:** [Table of Contents](../SUMMARY.md)
