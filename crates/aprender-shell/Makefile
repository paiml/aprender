# aprender-shell Makefile
# Chaos testing, Docker testing, and performance validation (Issue #99)
#
# Toyota Way: Jidoka - Build quality in through automated testing
#
# Usage:
#   make test          - Run unit tests
#   make docker-test   - Run full test suite in Docker
#   make chaos-test    - Run chaos engineering tests
#   make lint          - Lint Makefile and Dockerfile

SHELL := /bin/bash
BINARY := target/release/aprender-shell
MODEL := /tmp/aprender-test-model.apr
HISTORY := /tmp/aprender-test-history
DOCKER_IMAGE := aprender-shell-test
DOCKER_TAG := latest

.PHONY: all build test bench lint docker-build docker-test docker-test-all docker-shell docker-clean chaos-test chaos-gentle chaos-aggressive chaos-memory chaos-cpu clean

all: build test

build:
	cargo build --release -p aprender-shell

test:
	cargo test -p aprender-shell

bench:
	cargo bench -p aprender-shell

# Create test model for chaos testing
$(MODEL): build
	@mkdir -p /tmp
	@echo -e "git status\ngit commit -m 'test'\ngit push\ncargo build\ncargo test\ncargo bench\ndocker run\ndocker ps\nkubectl get pods\naws s3 ls" > $(HISTORY)
	@$(BINARY) train --history $(HISTORY) --output $(MODEL) 2>/dev/null

# Chaos Testing Suite (Issue #99)
# Toyota Way: Jidoka - Build quality in through stress testing

chaos-test: chaos-gentle chaos-memory chaos-cpu
	@echo "✅ All chaos tests passed"

# Gentle chaos - CI-safe, validates P99 < 10ms
chaos-gentle: $(MODEL)
	@echo "🧪 Running gentle chaos test (CI-safe)..."
	@renacer --chaos gentle -c -- $(BINARY) suggest "git " --model $(MODEL) 2>/dev/null
	@echo "✅ Gentle chaos: PASS"

# Memory pressure - simulates containers/embedded (64MB limit)
chaos-memory: $(MODEL)
	@echo "🧪 Running memory pressure test (64MB limit)..."
	@renacer --chaos-memory-limit 64M -c -- $(BINARY) suggest "cargo " --model $(MODEL) 2>/dev/null
	@echo "✅ Memory pressure: PASS"

# CPU throttle - simulates slow CI runners (25% CPU)
chaos-cpu: $(MODEL)
	@echo "🧪 Running CPU throttle test (25% CPU)..."
	@renacer --chaos-cpu-limit 0.25 -c -- $(BINARY) suggest "docker " --model $(MODEL) 2>/dev/null
	@echo "✅ CPU throttle: PASS"

# Aggressive chaos - full stress test (not for CI)
chaos-aggressive: $(MODEL)
	@echo "🧪 Running aggressive chaos test (manual only)..."
	@renacer --chaos aggressive -c -- $(BINARY) suggest "kubectl " --model $(MODEL) 2>/dev/null
	@echo "✅ Aggressive chaos: PASS"

# Signal injection - SIGUSR1/SIGUSR2 handling
chaos-signal: $(MODEL)
	@echo "🧪 Running signal injection test..."
	@renacer --chaos-signal SIGUSR1 -c -- $(BINARY) suggest "aws " --model $(MODEL) 2>/dev/null
	@echo "✅ Signal injection: PASS"

# Performance baseline with syscall profiling
perf: $(MODEL)
	@echo "📊 Running performance baseline..."
	@renacer -c --stats-extended -- $(BINARY) suggest "git " --model $(MODEL)

# Source-correlated trace
perf-trace: $(MODEL)
	@echo "📊 Running source-correlated trace..."
	@renacer --source --function-time -- $(BINARY) suggest "git status" --model $(MODEL)

# Full performance + chaos suite
perf-full: perf chaos-test
	@echo "✅ Full performance + chaos suite complete"

clean:
	@rm -f $(MODEL) $(HISTORY)
	cargo clean -p aprender-shell

# ==============================================================================
# Docker Testing (Reproducible Environment)
# ==============================================================================

# Build Docker image from workspace root
docker-build:
	@echo "🐳 Building Docker image..."
	@cd ../.. && docker build -f crates/aprender-shell/Dockerfile -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

# Run quick smoke test in Docker
docker-test: docker-build
	@echo "🧪 Running Docker smoke test..."
	@docker run --rm $(DOCKER_IMAGE):$(DOCKER_TAG) aprender-shell --version
	@docker run --rm $(DOCKER_IMAGE):$(DOCKER_TAG) aprender-shell --help
	@echo "✅ Docker smoke test: PASS"

# Run full test suite in Docker
docker-test-all: docker-build
	@echo "🧪 Running full Docker test suite..."
	@# Test 1: Version check
	@docker run --rm $(DOCKER_IMAGE):$(DOCKER_TAG) aprender-shell --version
	@# Test 2: Help
	@docker run --rm $(DOCKER_IMAGE):$(DOCKER_TAG) aprender-shell --help
	@# Test 3: Train model
	@docker run --rm $(DOCKER_IMAGE):$(DOCKER_TAG) sh -c '\
		aprender-shell train -f ~/.bash_history -o /tmp/test.model && \
		test -f /tmp/test.model && echo "✅ Train: PASS"'
	@# Test 4: Suggest completions
	@docker run --rm $(DOCKER_IMAGE):$(DOCKER_TAG) sh -c '\
		aprender-shell train -f ~/.bash_history -o /tmp/test.model && \
		aprender-shell suggest "git " -m /tmp/test.model && echo "✅ Suggest: PASS"'
	@# Test 5: Inspect model
	@docker run --rm $(DOCKER_IMAGE):$(DOCKER_TAG) sh -c '\
		aprender-shell train -f ~/.bash_history -o /tmp/test.model && \
		aprender-shell inspect -m /tmp/test.model && echo "✅ Inspect: PASS"'
	@# Test 6: Inspect JSON format
	@docker run --rm $(DOCKER_IMAGE):$(DOCKER_TAG) sh -c '\
		aprender-shell train -f ~/.bash_history -o /tmp/test.model && \
		aprender-shell inspect -m /tmp/test.model --format json | head -5 && echo "✅ Inspect JSON: PASS"'
	@# Test 7: Publish (without token, should show instructions)
	@docker run --rm $(DOCKER_IMAGE):$(DOCKER_TAG) sh -c '\
		aprender-shell train -f ~/.bash_history -o /tmp/test.model && \
		aprender-shell publish -m /tmp/test.model -r test/model 2>&1 | grep -q "HF_TOKEN" && echo "✅ Publish (no token): PASS"'
	@echo ""
	@echo "════════════════════════════════════════════"
	@echo "✅ All Docker tests passed!"
	@echo "════════════════════════════════════════════"

# Interactive Docker shell for debugging
docker-shell: docker-build
	@echo "🐚 Starting interactive Docker shell..."
	@docker run --rm -it $(DOCKER_IMAGE):$(DOCKER_TAG) bash

# Clean Docker artifacts
docker-clean:
	@echo "🧹 Cleaning Docker artifacts..."
	@docker rmi $(DOCKER_IMAGE):$(DOCKER_TAG) 2>/dev/null || true

# ==============================================================================
# Linting (bashrs validation)
# ==============================================================================

# Lint Makefile and Dockerfile
lint:
	@echo "🔍 Linting Makefile and Dockerfile..."
	@if command -v shellcheck >/dev/null 2>&1; then \
		echo "Checking shell commands in Makefile..."; \
		grep -E '^\t@?[^#]' Makefile | sed 's/^\t@\?//' | shellcheck -s bash - 2>/dev/null || true; \
	else \
		echo "⚠️  shellcheck not installed, skipping Makefile lint"; \
	fi
	@if command -v hadolint >/dev/null 2>&1; then \
		echo "Checking Dockerfile..."; \
		hadolint Dockerfile || true; \
	else \
		echo "⚠️  hadolint not installed, skipping Dockerfile lint"; \
	fi
	@echo "✅ Lint complete"

# Lint shell scripts with shellcheck
lint-scripts:
	@echo "🔍 Linting shell scripts with shellcheck..."
	@if command -v shellcheck >/dev/null 2>&1; then \
		shellcheck scripts/*.sh && echo "✅ Shell scripts: PASS"; \
	else \
		echo "⚠️  shellcheck not installed, skipping"; \
	fi
