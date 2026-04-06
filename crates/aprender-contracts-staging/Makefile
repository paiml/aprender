.PHONY: build test test-fast test-full lint clippy deny pv-lint clean coverage \
       coverage-html comply quality-gate book book-serve \
       lean-install lean-build lean-check lean-clean

# Proptest configuration (CB-126-D compliance)
export PROPTEST_CASES ?= 256

# ---------- Build ----------

build:
	cargo build --workspace

# ---------- Test ----------

test:
	PROPTEST_CASES=$(PROPTEST_CASES) cargo test --workspace

test-fast:
	PROPTEST_CASES=32 cargo test --workspace --lib

test-full:
	PROPTEST_CASES=1024 cargo test --workspace

# ---------- Lint ----------

lint: clippy deny pv-lint

clippy:
	cargo clippy --workspace -- -D warnings

deny:
	cargo deny check

pv-lint:
	cargo run --bin pv -- lint contracts/

fmt-check:
	cargo fmt --all -- --check

# ---------- Coverage ----------

coverage:
	cargo llvm-cov --all-features --workspace --lcov --output-path lcov.info

coverage-html:
	cargo llvm-cov --all-features --workspace --html

coverage-summary:
	cargo llvm-cov --all-features --workspace

# ---------- Compliance ----------

comply:
	pmat comply check

quality-gate:
	pmat quality-gate

# ---------- Book ----------

book:
	mdbook build

book-serve:
	mdbook serve --open

# ---------- Lean 4 Proofs (via forjar) ----------

lean-install:
	cd lean && forjar apply -f forjar.yaml -t toolchain --yes
	cd lean && forjar apply -f forjar.yaml -t mathlib --yes

lean-build:
	cd lean && forjar apply -f forjar.yaml --yes

lean-check:
	cd lean && PATH="$$HOME/.elan/bin:$$PATH" lake build

lean-clean:
	cd lean && rm -rf .lake lake-packages lake-manifest.json

# ---------- Clean ----------

clean:
	cargo clean
	rm -rf book/book
