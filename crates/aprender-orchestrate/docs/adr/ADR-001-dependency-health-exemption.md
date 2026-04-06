# ADR-001: CB-081 Dependency Health Exemption

## Status
Accepted

## Context

PMAT CB-081 (Dependency Health) requires:
- Maximum 50 direct dependencies
- Maximum 250 transitive dependencies

Batuta currently has:
- 6 direct dependencies (passes)
- 1083 transitive dependencies (exceeds limit by 4x)

## Decision

We accept non-compliance with CB-081 for batuta because it is an **orchestration framework** that by design integrates the entire Sovereign AI Stack:

### Major transitive dependency contributors:
| Crate | Transitive Deps | Purpose |
|-------|----------------|---------|
| trueno-graph | ~200 | Graph DB (arrow ecosystem) |
| presentar-terminal | ~150 | TUI dashboard |
| reqwest | ~100 | HTTP client (rustls, hyper) |
| pacha | ~80 | Model registry (rusqlite) |
| wgpu (via trueno) | ~150 | GPU compute |

### Why this cannot be reduced:
1. **Arrow ecosystem** (~200 deps): Required for trueno-db and trueno-graph
2. **TLS/HTTP** (~100 deps): Required for stack drift checking, HuggingFace integration
3. **GPU** (~150 deps): Required for trueno GPU acceleration
4. **Terminal UI** (~100 deps): Required for presentar-terminal dashboard

### Alternatives considered:
1. **Split into multiple crates**: batuta-core, batuta-ml, batuta-viz
   - Rejected: Increases maintenance burden, complicates user experience
2. **Make features optional**: Already done, but `native` feature enables most
   - Rejected: Users expect full functionality by default
3. **Use lighter alternatives**: Replace arrow with custom implementation
   - Rejected: Arrow is industry standard, well-tested

## Consequences

### Positive
- Full Sovereign AI Stack integration in single crate
- Simplified user experience (one install)
- Comprehensive functionality

### Negative
- CB-081 will always fail for batuta
- Longer build times due to many deps
- Potential for dep conflicts (126 duplicate crates)

### Mitigation
- Document exemption in .pmat/project.toml
- Track duplicate crates and update when possible
- Consider crate splitting in future major version

## References
- PMAT CB-081: Dependency Health Check
- Sovereign AI Stack: https://github.com/paiml
