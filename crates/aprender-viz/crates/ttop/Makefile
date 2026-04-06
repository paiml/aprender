# ttop - Terminal Top System Monitor
# Fast build/test/coverage workflow
#
# Targets:
#   make lint     - Run clippy (warnings = errors)
#   make test     - Fast tests with nextest
#   make coverage - Fast coverage report (<3 min target)
#   make build    - Release build
#   make install  - Install to ~/.local/bin

.PHONY: all lint test coverage build release install clean help

# Coverage exclusions - binary entry points and external deps
COVERAGE_EXCLUDE := --ignore-filename-regex='main\.rs|bin/.*\.rs|probar/|jugar-probar'

# Default target
all: lint test build

# Lint with strict warnings
lint:
	@echo "Linting Rust code (strict mode)..."
	@cargo clippy -- -D warnings
	@echo "Linting passed!"

# Fast tests with nextest
test:
	@echo "Running fast tests with nextest..."
	@if command -v cargo-nextest >/dev/null 2>&1; then \
		RUST_TEST_THREADS=$$(nproc) cargo nextest run --status-level skip --failure-output immediate; \
	else \
		cargo test; \
	fi

# Fast coverage report (<3 min target, 85%+ achieved)
coverage:
	@echo "Running FAST coverage (target: <3 min, 85%+ achieved)..."
	@which cargo-llvm-cov > /dev/null 2>&1 || cargo install cargo-llvm-cov --locked
	@which cargo-nextest > /dev/null 2>&1 || cargo install cargo-nextest --locked
	@mkdir -p target/coverage
	@echo "Running coverage with nextest ($$(nproc) threads)..."
	@env PROPTEST_CASES=5 \
		cargo llvm-cov nextest \
		--no-tests=warn \
		--html --output-dir target/coverage/html \
		$(COVERAGE_EXCLUDE) \
		-E 'not test(/stress|fuzz|slow|benchmark/)'
	@cargo llvm-cov report --lcov --output-path target/coverage/lcov.info $(COVERAGE_EXCLUDE)
	@echo ""
	@cargo llvm-cov report --summary-only $(COVERAGE_EXCLUDE)
	@echo ""
	@echo "HTML report: target/coverage/html/index.html"

# Coverage summary only (fast)
coverage-summary:
	@echo "Coverage summary..."
	@cargo llvm-cov report --summary-only $(COVERAGE_EXCLUDE) 2>/dev/null || \
		cargo llvm-cov nextest --summary-only $(COVERAGE_EXCLUDE) -E 'not test(/stress|fuzz|slow|benchmark/)'

# Release build
build:
	@echo "Building release..."
	@cargo build --release

# Release with optimizations
release:
	@echo "Building optimized release..."
	@RUSTFLAGS="-C target-cpu=native" cargo build --release

# Install to ~/.local/bin
install: release
	@echo "Installing lltop to ~/.local/bin..."
	@mkdir -p ~/.local/bin
	@cp target/release/ttop ~/.local/bin/lltop
	@echo "Installed: ~/.local/bin/lltop"

# Clean build artifacts
clean:
	@cargo clean
	@rm -rf target/coverage

# Help
help:
	@echo "ttop Makefile targets:"
	@echo "  lint     - Run clippy (warnings = errors)"
	@echo "  test     - Fast tests with nextest"
	@echo "  coverage - Fast coverage report (<3 min)"
	@echo "  build    - Release build"
	@echo "  release  - Optimized release build"
	@echo "  install  - Install to ~/.local/bin/lltop"
	@echo "  clean    - Clean build artifacts"
