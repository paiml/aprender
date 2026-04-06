# Scripts Directory

Development and maintenance scripts for Repartir.

## Available Scripts

### install-hooks.sh

Installs git pre-commit hooks that enforce quality gates.

**Usage:**
```bash
./scripts/install-hooks.sh
```

**What it does:**
- Copies `.githooks/pre-commit` to `.git/hooks/pre-commit`
- Makes the hook executable
- Enforces quality gates before every commit

**Quality Gates Enforced:**
1. **Clippy**: Zero-tolerance for warnings (`cargo clippy -- -D warnings`)
2. **Formatting**: Code must be formatted (`cargo fmt --check`)
3. **Tests**: All library tests must pass (`cargo test --lib`)
4. **Book**: Documentation must build (`mdbook build book/`)
5. **Shell Scripts**: bashrs linting on staged bash/Makefile files

**Bypass (Emergency Only):**
```bash
git commit --no-verify
```

**Note:** This bypasses all quality gates and should only be used in emergencies.

## Development Workflow

### First Time Setup

After cloning the repository:

```bash
# Install git hooks
./scripts/install-hooks.sh

# Install optional tools
cargo install mdbook    # For book validation
cargo install bashrs    # For shell script linting
```

### Before Committing

The pre-commit hook automatically runs, but you can test manually:

```bash
make quality-gates      # Run all quality gates
make lint              # Clippy check
make fmt-check         # Format check
make test              # Run tests
make build-book        # Build documentation
```

## Continuous Improvement (Kaizen)

These hooks embody the **Iron Lotus Framework** principles:

- **Jidoka (自働化)**: Automated quality gates stop defects at the source
- **Genchi Genbutsu (現地現物)**: Real-time feedback on code quality
- **Kaizen (改善)**: Continuous improvement through enforced standards

Every commit maintains the project's high quality bar: **97.20% coverage, A+ grade, 117 tests passing**.
