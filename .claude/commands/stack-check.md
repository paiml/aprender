Run a comprehensive health check on the PAIML Sovereign AI Stack.

Run: `cargo run --quiet -- stack check --verify-published`

This checks:
- Dependency graph for cycles
- Version alignment across crates
- Path vs crates.io dependency status
- Published version verification

Report any issues found and suggest fixes.
