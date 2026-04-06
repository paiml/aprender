# Error Codes

Batuta error codes follow a hierarchical naming convention for easy identification and resolution.

## Error Code Format

```
BATUTA-[PHASE]-[NUMBER]
```

- **PHASE**: Which phase generated the error (ANALYZE, TRANSPILE, OPTIMIZE, VALIDATE, BUILD)
- **NUMBER**: Specific error within that phase

## Analysis Phase Errors (BATUTA-A-*)

| Code | Description | Resolution |
|------|-------------|------------|
| `BATUTA-A-001` | Language detection failed | Ensure source files have correct extensions |
| `BATUTA-A-002` | Dependency analysis timeout | Increase timeout or reduce project scope |
| `BATUTA-A-003` | TDG calculation error | Check for circular dependencies |
| `BATUTA-A-004` | ML framework not recognized | Update Batuta to latest version |

## Transpilation Phase Errors (BATUTA-T-*)

| Code | Description | Resolution |
|------|-------------|------------|
| `BATUTA-T-001` | Transpiler not found | Install required transpiler (depyler/bashrs/decy) |
| `BATUTA-T-002` | Syntax error in source | Fix source code syntax |
| `BATUTA-T-003` | Type inference failed | Add type annotations |
| `BATUTA-T-004` | Unsupported construct | Check compatibility matrix |

## Optimization Phase Errors (BATUTA-O-*)

| Code | Description | Resolution |
|------|-------------|------------|
| `BATUTA-O-001` | SIMD not available | Use fallback backend |
| `BATUTA-O-002` | GPU memory exhausted | Reduce batch size |
| `BATUTA-O-003` | Backend selection failed | Check hardware compatibility |

## Validation Phase Errors (BATUTA-V-*)

| Code | Description | Resolution |
|------|-------------|------------|
| `BATUTA-V-001` | Output mismatch | Review semantic differences |
| `BATUTA-V-002` | Test suite failed | Fix failing tests |
| `BATUTA-V-003` | Syscall trace divergence | Check I/O operations |

## Build Phase Errors (BATUTA-B-*)

| Code | Description | Resolution |
|------|-------------|------------|
| `BATUTA-B-001` | Compilation failed | Check Rust compiler output |
| `BATUTA-B-002` | Linking error | Verify dependencies |
| `BATUTA-B-003` | Cross-compilation unsupported | Check target architecture |

## Quality Gate Errors (BATUTA-Q-*)

| Code | Description | Resolution |
|------|-------------|------------|
| `BATUTA-Q-001` | Demo score below threshold | Improve code quality to A- (85) |
| `BATUTA-Q-002` | Coverage insufficient | Add more tests |
| `BATUTA-Q-003` | Clippy warnings present | Fix linting issues |

---

**Navigate:** [Table of Contents](../SUMMARY.md)
