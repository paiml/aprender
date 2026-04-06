# Tool Detection System

Batuta discovers external transpilers (depyler, decy, bashrs) and analysis tools (pmat, renacer) at runtime through PATH-based lookup.

## Detection Process

1. Search PATH for the binary name
2. Run `<tool> --version` to get the version
3. Compare against minimum required version
4. Cache the result in `.batuta/cache/tool_versions.json`

## Tool Registry

| Tool | Binary | Version | Purpose |
|------|--------|---------|---------|
| `depyler` | `depyler` | 4.1.1 | Python to Rust |
| `decy` | `decy` | 2.2.0 | C/C++ to Rust |
| `bashrs` | `bashrs` | 6.65.0 | Shell to Rust |
| `pmat` | `pmat` | 3.10.0 | Static analysis, TDG |
| `renacer` | `renacer` | 0.10.1 | Syscall tracing |

## Checking Tools

```bash
batuta analyze --check-tools
```

Output:

```
Tool Detection Report:
  depyler  v3.20   ~/.cargo/bin/depyler   [OK]
  decy     v0.3.1  ~/.cargo/bin/decy      [OK]
  bashrs   v6.65   ~/.cargo/bin/bashrs    [OK]
  pmat     v0.8.3  ~/.cargo/bin/pmat      [OK]
  renacer  v0.10.0 ~/.cargo/bin/renacer   [OK]
```

## Version Mismatch Handling

| Condition | Behavior |
|-----------|----------|
| Tool found, version OK | Proceed normally |
| Tool found, version old | Error with upgrade instructions |
| Tool not found | Error with install instructions |

## Fallback Behavior

Configure in `batuta.toml`:

```toml
[pipeline]
# strict: fail if any tool missing (default)
# lenient: skip unsupported languages, warn only
missing_tool_policy = "strict"
```

## Cache Behavior

Tool detection results are cached to avoid repeated PATH lookups. The cache is invalidated when:

- The PATH environment variable changes
- A tool binary is newer than the cache entry
- The cache is older than 24 hours

Force re-detection:

```bash
rm .batuta/cache/tool_versions.json
batuta analyze --check-tools
```

---

**Navigate:** [Table of Contents](../SUMMARY.md)
