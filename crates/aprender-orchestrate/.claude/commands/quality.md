Run the full quality gate check for the batuta project.

Execute these commands in sequence:
1. `cargo fmt --check` - Formatting
2. `cargo clippy -- -D warnings -A dead_code` - Linting
3. `cargo test --lib` - Unit tests
4. `make coverage` - Coverage report (if time permits)

Report the results of each check. If any fail, provide specific fixes.
