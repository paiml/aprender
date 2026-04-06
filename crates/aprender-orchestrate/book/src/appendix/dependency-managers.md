# Appendix C: Dependency Managers

Batuta detects dependencies in source projects by analyzing manifest and lock files from multiple package managers, then maps them to Rust crate equivalents.

## Supported Managers

| Manager | Language | Manifest File | Lock File |
|---------|----------|---------------|-----------|
| pip | Python | `requirements.txt`, `pyproject.toml` | `requirements.txt` |
| poetry | Python | `pyproject.toml` | `poetry.lock` |
| npm | JavaScript | `package.json` | `package-lock.json` |
| make | C/C++ | `Makefile` | N/A |
| cmake | C/C++ | `CMakeLists.txt` | N/A |

## Detection and Cargo.toml Generation

```bash
batuta analyze --dependencies /path/to/project
```

Batuta generates a `Cargo.toml` from detected dependencies:

```toml
[dependencies]
trueno = "0.14"           # from: numpy >= 1.24.0
aprender = "0.24"         # from: scikit-learn ~= 1.3
realizar = "0.5"          # from: torch >= 2.0
reqwest = "0.12"          # from: requests >= 2.28
serde = { version = "1", features = ["derive"] }  # from: json (stdlib)
```

## Version Constraint Mapping

| Python Syntax | Meaning | Rust Equivalent |
|--------------|---------|-----------------|
| `== 1.2.3` | Exact | `= "1.2.3"` |
| `>= 1.2.0` | Minimum | `">= 1.2.0"` |
| `~= 1.2` | Compatible (>= 1.2, < 2.0) | `"1.2"` |

## Common Python-to-Rust Mappings

| Python | Rust Crate | Notes |
|--------|------------|-------|
| numpy | trueno | Stack native |
| scikit-learn | aprender | Stack native |
| torch | realizar | Inference only |
| pandas | polars / alimentar | alimentar for Arrow |
| requests | reqwest | Async HTTP |
| flask / fastapi | axum | Async web framework |
| click | clap | CLI argument parsing |
| pydantic | serde | Serialization |
| pytest | (built-in) | `#[test]` + proptest |
| logging | tracing | Structured logging |

## Custom Mappings

Override or extend defaults in `batuta.toml`:

```toml
[dependencies.mapping]
my_internal_lib = { crate = "my-rust-lib", version = "0.5" }
boto3 = { crate = "aws-sdk-s3", version = "1", features = ["behavior-version-latest"] }
setuptools = { ignore = true }
```

---

**Navigate:** [Table of Contents](../SUMMARY.md)
