# Use bash for shell commands
SHELL := /bin/bash

# Test thread limit - allows parallel test execution
TEST_THREADS ?= $(shell nproc)
export RUST_TEST_THREADS=$(TEST_THREADS)

# PERFORMANCE TARGETS (Toyota Way: Zero Defects, Fast Feedback)
# - make test-fast:  < 30 seconds (minimal property tests)
# - make coverage:   < 2 minutes (core tests, excludes CLI)
# - make test:       comprehensive

.PHONY: build test test-fast lint lint-rust lint-bash lint-bash-verbose fix-bash lint-make lint-docker purify-bash \
        fmt fmt-check coverage coverage-sudo coverage-quick clean \
        docker-build docker-test docker-component docker-io docker-fs docker-stress docker-falsify docker-debug

build:
	cargo build --release

# Fast tests - target: <30 seconds
test-fast:
	@echo "⚡ Running fast tests (target: <30s)..."
	@if command -v cargo-nextest >/dev/null 2>&1; then \
		PROPTEST_CASES=10 cargo nextest run \
			--workspace \
			--lib \
			--status-level skip \
			--failure-output immediate \
			-E 'not test(/benchmark|stress|fuzz/)'; \
	else \
		PROPTEST_CASES=10 cargo test --workspace --lib; \
	fi
	@echo "✅ Fast tests completed!"

# Comprehensive test suite
test: test-fast
	@echo "🧪 Running comprehensive tests..."
	@cargo test --workspace --lib
	@cargo test --doc --workspace
	@echo "✅ All tests completed!"

lint: lint-rust lint-bash lint-make lint-docker
	@echo "✅ All lints passed!"

lint-rust:
	@echo "🦀 Linting Rust code..."
	cargo clippy -- -D warnings

# bashrs enforcement for shell scripts (Batuta Stack)
BASH_SCRIPTS := $(wildcard scripts/*.sh)
DOCKERFILES := $(wildcard docker/Dockerfile.*)

# Lint bash scripts - strict mode (blocking on errors/warnings)
# Fixed: SC2066 warnings in scientific-swap-benchmark.sh (2025-01-13)
lint-bash:
	@echo "🐚 Linting bash scripts with bashrs (strict mode)..."
	@failed=0; \
	for script in $(BASH_SCRIPTS); do \
		echo "  Checking $$script..."; \
		if ! bashrs lint "$$script" --level warning 2>&1 | grep -qE "^✗"; then \
			echo "    ✅ passed"; \
		else \
			bashrs lint "$$script" --level warning 2>&1 | grep -E "^(✗|⚠)" || true; \
			failed=1; \
		fi; \
	done; \
	if [ $$failed -eq 1 ]; then \
		echo "❌ Bash lint failed! Fix warnings above."; \
		exit 1; \
	fi
	@echo "✅ All bash scripts passed lint!"

# Verbose lint - show all warnings
lint-bash-verbose:
	@echo "🐚 Linting bash scripts with bashrs (verbose)..."
	@for script in $(BASH_SCRIPTS); do \
		echo ""; \
		echo "=== $$script ==="; \
		bashrs lint "$$script" --level info || true; \
	done

# Auto-fix safe issues in bash scripts
fix-bash:
	@echo "🔧 Auto-fixing bash scripts with bashrs..."
	@for script in $(BASH_SCRIPTS); do \
		echo "  Fixing $$script..."; \
		bashrs lint "$$script" --fix -o "$$script.fixed" && mv "$$script.fixed" "$$script" || true; \
	done
	@echo "✅ Safe fixes applied!"

lint-make:
	@echo "📋 Linting Makefile with bashrs..."
	@bashrs make lint Makefile 2>/dev/null || echo "⚠️  Makefile lint warnings (non-blocking)"
	@echo "✅ Makefile checked!"

lint-docker:
	@echo "🐳 Linting Dockerfiles with bashrs..."
	@for dockerfile in $(DOCKERFILES); do \
		echo "  Checking $$dockerfile..."; \
		bashrs dockerfile lint "$$dockerfile" 2>&1 | grep -E "^(✗|Summary:)" || true; \
	done
	@echo "✅ Dockerfiles checked (issues logged above if any)"

# Purify scripts (determinism + idempotency + safety)
purify-bash:
	@echo "🧹 Purifying bash scripts..."
	@for script in $(BASH_SCRIPTS); do \
		bashrs purify "$$script" --in-place --backup 2>/dev/null || true; \
	done
	@echo "✅ Scripts purified!"

fmt:
	cargo fmt

fmt-check:
	cargo fmt --check

# Coverage exclusion for modules requiring external/root access or kernel features
# - cli: CLI modules require external binaries
# - zram/device|ops: Require root and zram kernel module
# - perf/tenx: Require io_uring SQPOLL, huge pages, registered buffers (kernel features)
# - perf/numa|affinity: Require libnuma and CPU affinity syscalls
# - backend: Requires kernel ZRAM device
# - stats: Requires /sys filesystem access
# - device.rs: Requires ublk kernel interface
# - daemon.rs: Requires ublk kernel interface (bins/trueno-ublk/src/daemon.rs)
# - cleanup.rs: Requires ublk devices in /dev
# Note: Core library is fully testable and not excluded
# Coverage exclusions for untestable modules (require kernel/root/external deps)
# Added benchmark.rs (0% coverage - only runs via examples)
COVERAGE_EXCLUDE := --ignore-filename-regex='cli|zram/device\.rs|zram/ops\.rs|perf/tenx/|perf/numa|perf/affinity|backend\.rs|stats\.rs|bins/trueno-ublk/src/device\.rs|bins/trueno-ublk/src/daemon\.rs|cleanup\.rs|benchmark\.rs'

# Fast coverage - target: <5 minutes
# CB-127: Use 'cargo llvm-cov test' instead of nextest (profraw explosion)
coverage:
	@echo "📊 Running fast coverage analysis..."
	@which cargo-llvm-cov > /dev/null 2>&1 || (echo "📦 Installing cargo-llvm-cov..." && cargo install cargo-llvm-cov --locked)
	@mkdir -p target/coverage
	@echo "🧪 Running tests with instrumentation..."
	@env PROPTEST_CASES=3 QUICKCHECK_TESTS=3 cargo llvm-cov test \
		--lib \
		-p trueno-zram-core \
		-p trueno-zram-adaptive \
		--html --output-dir target/coverage/html \
		$(COVERAGE_EXCLUDE)
	@echo ""
	@cargo llvm-cov report --summary-only $(COVERAGE_EXCLUDE)
	@echo "💡 HTML report: target/coverage/html/index.html"

# Full coverage - all packages (slower, ~10min)
coverage-full:
	@echo "📊 Running full coverage analysis..."
	@env PROPTEST_CASES=5 QUICKCHECK_TESTS=5 cargo llvm-cov test \
		--workspace \
		--lib \
		--html --output-dir target/coverage/html \
		$(COVERAGE_EXCLUDE)
	@cargo llvm-cov report --summary-only $(COVERAGE_EXCLUDE)
	@echo "💡 HTML report: target/coverage/html/index.html"

# Quick coverage - even faster for iteration (<30s warm)
coverage-quick:
	@echo "⚡ Quick coverage..."
	@env PROPTEST_CASES=1 QUICKCHECK_TESTS=1 cargo llvm-cov \
		--no-cfg-coverage \
		--workspace \
		--lib \
		--html --output-dir target/coverage/html \
		$(COVERAGE_EXCLUDE)
	@cargo llvm-cov report --summary-only $(COVERAGE_EXCLUDE)
	@echo "💡 HTML: target/coverage/html/index.html"

# Coverage with sudo (includes zram integration tests requiring root)
coverage-sudo:
	sudo bash -c 'source $$HOME/.cargo/env && \
		cd $(PWD) && \
		cargo llvm-cov --lib --ignore-filename-regex="cli" --html -- --include-ignored'
	sudo chown -R $(USER):$(USER) target
	@echo "Coverage report: target/llvm-cov/html/index.html"

# CI coverage - LCOV output
coverage-ci:
	@echo "📊 Generating LCOV report for CI..."
	@env PROPTEST_CASES=10 QUICKCHECK_TESTS=10 cargo llvm-cov \
		--no-cfg-coverage \
		--workspace \
		--lib \
		--lcov --output-path lcov.info \
		$(COVERAGE_EXCLUDE)
	@echo "✓ Coverage report: lcov.info"

coverage-open:
	@if [ -f target/coverage/html/index.html ]; then \
		xdg-open target/coverage/html/index.html 2>/dev/null || \
		open target/coverage/html/index.html 2>/dev/null || \
		echo "Open: target/coverage/html/index.html"; \
	else \
		echo "❌ Run 'make coverage' first"; \
	fi

coverage-clean:
	@rm -f lcov.info
	@rm -rf target/llvm-cov target/coverage
	@find . -name "*.profraw" -delete
	@echo "✓ Coverage artifacts cleaned"

clean:
	cargo clean

# ============================================================================
# DOCKER-BASED UBLK TESTING
# Per testing-debugging-troubleshooting.md specification
# Toyota Way: Disposable containers prevent system destabilization
# ============================================================================

DOCKER_COMPOSE := docker-compose -f docker/docker-compose.test.yml
DOCKER_IMAGE := trueno-ublk-test

# Build Docker test image
docker-build:
	@echo "🐳 Building Docker test image..."
	docker build -t $(DOCKER_IMAGE) -f docker/Dockerfile.ublk-test .

# Run full Docker test suite
docker-test: docker-build
	@echo "🧪 Running Docker test suite..."
	$(DOCKER_COMPOSE) up --abort-on-container-exit full-suite

# Component tests only (F001-F015)
docker-component: docker-build
	@echo "🔧 Running component tests in Docker..."
	docker run --privileged \
		-v /lib/modules:/lib/modules:ro \
		-v /dev:/dev \
		-v $(PWD):/workspace:ro \
		$(DOCKER_IMAGE) component

# I/O verification tests (F016-F035)
docker-io: docker-build
	@echo "💾 Running I/O verification in Docker..."
	docker run --privileged \
		-v /lib/modules:/lib/modules:ro \
		-v /dev:/dev \
		-v $(PWD):/workspace:ro \
		$(DOCKER_IMAGE) io-verify

# Filesystem integration tests (F086-F095)
docker-fs: docker-build
	@echo "📂 Running filesystem tests in Docker..."
	docker run --privileged \
		-v /lib/modules:/lib/modules:ro \
		-v /dev:/dev \
		-v $(PWD):/workspace:ro \
		--tmpfs /mnt/test:size=4G \
		$(DOCKER_IMAGE) filesystem

# Stress tests
docker-stress: docker-build
	@echo "🔥 Running stress tests in Docker..."
	docker run --privileged \
		-v /lib/modules:/lib/modules:ro \
		-v /dev:/dev \
		-v $(PWD):/workspace:ro \
		--memory=8g \
		$(DOCKER_IMAGE) stress

# Falsification matrix (F001-F100)
docker-falsify: docker-build
	@echo "🔬 Running falsification matrix in Docker..."
	@mkdir -p test-results
	docker run --privileged \
		-v /lib/modules:/lib/modules:ro \
		-v /dev:/dev \
		-v $(PWD):/workspace:ro \
		-v $(PWD)/test-results:/workspace/test-results \
		$(DOCKER_IMAGE) falsification
	@echo "📊 Results: test-results/falsification-report.json"

# Interactive debug session (Genchi Genbutsu: Go and See)
docker-debug: docker-build
	@echo "🔍 Starting interactive debug session..."
	docker run -it --privileged \
		-v /lib/modules:/lib/modules:ro \
		-v /dev:/dev \
		-v $(PWD):/workspace \
		--tmpfs /mnt/test:size=4G \
		$(DOCKER_IMAGE) debug

# Clean Docker artifacts
docker-clean:
	$(DOCKER_COMPOSE) down -v --remove-orphans
	docker rmi $(DOCKER_IMAGE) 2>/dev/null || true
	@echo "✓ Docker artifacts cleaned"

# Mutation testing
mutants:
	cargo mutants --no-times --timeout 300
