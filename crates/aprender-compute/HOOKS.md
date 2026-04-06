# Pre-commit Hooks

PMAT pre-commit hooks are automatically installed and managed by the PMAT toolkit.

## Installation

Hooks are automatically installed via:

```bash
pmat hooks install
```

This creates `.git/hooks/pre-commit` with the following checks:

## Pre-commit Quality Gates

The pre-commit hook runs these checks (target: <30 seconds):

1. **Formatting** - `cargo fmt --check`
   - Zero tolerance for formatting violations
   - Auto-fix: `cargo fmt`

2. **Linting** - `cargo clippy -- -D warnings`
   - Zero warnings policy
   - All clippy suggestions must be addressed

3. **Fast Tests** - `cargo test --quiet`
   - Runs all unit tests
   - Property tests included
   - Target: <5 minutes

4. **Dead Code** - PMAT dead code detection
   - Identifies unused code
   - Max 5% dead code allowed

## Hook Configuration

Hooks are configured via `.pmat-gates.toml`:

```toml
[pre-commit]
enabled = true
checks = [
    "formatting",
    "linting",
    "test-fast",
    "dead-code"
]
parallel = true
```

## Environment Variables

The hook sets these environment variables:

- `PMAT_MAX_CYCLOMATIC_COMPLEXITY=30`
- `PMAT_MAX_COGNITIVE_COMPLEXITY=25`
- `PMAT_MIN_TEST_COVERAGE=80`
- `PMAT_MAX_SATD_COMMENTS=5`
- `PMAT_TASK_ID_PATTERN="PMAT-[0-9]{4}"`

## Hook Management

```bash
# Check hook status
pmat hooks status

# Verify hooks work
pmat hooks verify

# Refresh hooks after config changes
pmat hooks refresh

# Uninstall hooks
pmat hooks uninstall

# Run hooks manually (CI/CD)
pmat hooks run
```

## Bypassing Hooks (Emergency Only)

**WARNING**: Bypassing hooks violates EXTREME TDD principles. Only use in genuine emergencies.

```bash
# NOT RECOMMENDED
git commit --no-verify -m "Emergency fix"
```

After bypassing, immediately:
1. Run `make quality-gates` manually
2. Fix all violations
3. Amend the commit with fixes

## Troubleshooting

### Hook fails with "pmat not found"
```bash
# Ensure pmat is in PATH
which pmat

# Reinstall if needed
cargo install pmat
```

### Hook takes too long (>30 seconds)
```bash
# Run with verbose output
PMAT_VERBOSE=1 git commit -m "message"

# Check which check is slow
pmat hooks verify
```

### Hook fails unexpectedly
```bash
# Run manually to debug
pmat hooks run

# Check configuration
cat .pmat-gates.toml
```

## Toyota Way Integration

Pre-commit hooks embody **Jidoka** (built-in quality):

- **Stop the line**: Commits blocked if quality gates fail
- **Immediate feedback**: Developers know about issues before push
- **Built-in quality**: Can't commit bad code

This prevents technical debt accumulation and enforces EXTREME TDD standards at every commit.

## CI/CD Integration

The same hooks run in GitHub Actions CI:

```yaml
- name: Run pre-commit checks
  run: pmat hooks run
```

This ensures local and CI environments enforce identical standards.
