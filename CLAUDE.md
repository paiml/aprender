# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Critical Rules

### ⛔ ABSOLUTE PROHIBITION: NO REBOOTING ⛔

**NEVER, EVER reboot the machine under ANY circumstances.**

This is a HARD BLOCK. Do NOT:
- Run `sudo reboot`
- Run `sudo shutdown`
- Run `systemctl reboot`
- Run `init 6`
- Run ANY command that causes a system restart
- SUGGEST rebooting to the user

If kernel modules are stuck or processes are in D-state:
1. Try a different device ID (e.g., `--id 1` instead of `--id 0`)
2. Wait for kernel timeout (may take minutes)
3. Try `sudo rmmod -f module_name` (force unload)
4. Ask the user for help
5. Continue with other tasks
6. **NEVER REBOOT**

Violation of this rule wastes user time and destroys unsaved work.

## Project Overview

trueno-zram is a SIMD-accelerated memory compression library for Linux systems, part of the PAIML "Batuta Stack" (trueno + bashrs + aprender). It provides userspace Rust implementations of LZ4 and ZSTD compression that leverage modern CPU vector instructions (AVX2, AVX-512, NEON) to replace kernel-level compression.

### Production Status (2026-01-06)

**MILESTONE DT-005 ACHIEVED:** trueno-zram is running as system swap!

| Component | Status | Notes |
|-----------|--------|-------|
| System Swap | ACTIVE | 8GB device, priority 150 |
| CPU SIMD Compress | PRODUCTION | 20-24 GB/s at 10GB scale |
| GPU Decompress | PRODUCTION | 137 GB/s on RTX 4090 |
| GPU Compress | BLOCKED | F082 Computed Address Bug (F081 was FALSIFIED) |
| mlock() Fix | **COMPLETED** | DT-007 via duende-mlock v1.0.0 |

### DT-007 Swap Deadlock Fix (COMPLETED 2026-01-06)

The swap deadlock issue is **FIXED**. Daemon memory is now locked via `mlockall()`:

```bash
# Verify mlock is active
grep VmLck /proc/$(pgrep trueno-ublk)/status
# Expected: VmLck > 200000 kB (daemon memory locked)
```

### Known Issues

1. **Swap Deadlock (DT-007):** ✅ **FIXED** - Daemon memory is now pinned with mlock() via duende-mlock crate. Both foreground and background modes work correctly.

2. **GPU Compression Blocked (KF-002):** NVIDIA PTX bug **F082** (Computed Address Bug) - addresses computed from shared memory values cause crashes. Note: F081 (Loaded Value Bug) was **FALSIFIED** on 2026-01-05. Workaround: hybrid architecture (CPU compress + GPU decompress).

3. **Docker Isolation:** ublk devices are host kernel resources and cannot be isolated in Docker containers. Test on host with controlled swap fill.

## Code Search (pmat query)

**NEVER use grep or rg for code discovery.** Use `pmat query` instead -- it returns quality-annotated, ranked results with TDG scores and fault annotations.

```bash
# Find functions by intent
pmat query "compression kernel" --limit 10

# Find high-quality code
pmat query "lz4 encode" --min-grade A --exclude-tests

# Find with fault annotations (unwrap, panic, unsafe, etc.)
pmat query "decompression" --faults

# Filter by complexity
pmat query "page handler" --max-complexity 10

# Cross-project search
pmat query "simd acceleration" --include-project ../trueno

# Git history search (find code by commit intent via RRF fusion)
pmat query "fix zstd decode" -G
pmat query "fix zstd decode" --git-history

# Enrichment flags (combine freely)
pmat query "compressor" --churn              # git volatility (commit count, churn score)
pmat query "algorithm" --duplicates          # code clone detection (MinHash+LSH)
pmat query "page manager" --entropy             # pattern diversity (repetitive vs unique)
pmat query "compression pipeline" --churn --duplicates --entropy --faults -G  # full audit
```

## ⚠️ Safe Performance Testing (MANDATORY)

**A system crash occurred due to unsafe memory allocation testing.** Follow these rules:

### NEVER DO:
- Allocate more than 10% of available RAM in tests
- Run memory pressure tests without `timeout` command
- Skip health checks between test steps
- Allocate >10GB without explicit user approval

### ALWAYS DO:
```bash
# 1. Check available memory first
free -g | grep Mem | awk '{print $7}'  # Available GB

# 2. Use timeouts for all allocation tests
timeout 10 python3 -c "import mmap; d=mmap.mmap(-1, 1024**3)"

# 3. Prefer Direct I/O tests (SAFE - no memory pressure)
sudo dd if=/dev/ublkb0 of=/dev/null bs=1M count=1024 iflag=direct

# 4. Check daemon health after EVERY test
grep -E "^(State|VmLck)" /proc/$(pgrep trueno-ublk)/status
# Expected: State=S, VmLck > 200000 kB
```

### Performance Results (Safe Testing)
| Test | Throughput | Notes |
|------|------------|-------|
| Sequential Read | **11.7-12.5 GB/s** | Direct I/O, daemon healthy |

See: `docs/specifications/testing-debugging-troubleshooting.md` Section 10

## Build Commands

```bash
# Build all crates
cargo build --all-features

# Run all tests
cargo test --all-features

# Run tests for a specific crate
cargo test -p trueno-zram-core

# Run a single test
cargo test --all-features test_name

# Format code
cargo fmt --all

# Lint
cargo clippy --all-targets --all-features -- -D warnings

# Build documentation
cargo doc --no-deps

# Run benchmarks
cargo bench --all-features

# Run benchmarks with baseline comparison
cargo bench --all-features -- --save-baseline main

# Run mutation testing (requires cargo-mutants)
cargo mutants --package trueno-zram-core

# Security audit (requires cargo-audit)
cargo audit

# Coverage (requires cargo-llvm-cov)
cargo llvm-cov --all-features --lcov --output-path lcov.info
```

## Architecture

### Crate Structure

- **trueno-zram-core**: SIMD-vectorized LZ4/ZSTD compression engines with runtime CPU feature detection and dispatch
- **trueno-zram-adaptive**: ML-driven compression algorithm selection based on page entropy analysis (integrates with aprender)
- **trueno-zram-generator**: systemd integration for zram device configuration at boot
- **trueno-zram-cli**: Rust-native zramctl replacement for device management

### Key Design Patterns

1. **Runtime SIMD dispatch**: CPU features detected at runtime via `simd/detect.rs`, dispatch handled in `simd/dispatch.rs`
2. **Page-based compression**: All compression operates on 4KB pages (`page.rs`)
3. **Builder pattern**: `CompressorBuilder` for configuring algorithm and SIMD backend
4. **Trait-based abstraction**: `PageCompressor` trait for compression implementations

### SIMD Implementation Structure

Each compression algorithm (LZ4, ZSTD) has separate implementations:
- `compress.rs` / `decompress.rs`: Core algorithm logic
- `avx2.rs`: AVX2 (256-bit) vectorized implementation
- `avx512.rs`: AVX-512 (512-bit) vectorized implementation
- `neon.rs`: ARM NEON (128-bit) implementation

## Development Methodology

This project follows Extreme TDD with strict quality gates:

- **Test-first**: Write failing test, verify red, write minimal code, verify green, refactor
- **Coverage target**: >95% line coverage
- **Mutation score target**: >80% (using cargo-mutants)
- **Fuzz testing**: Required for compression/decompression paths

### Quality Gates

**Commit Gate**: `cargo fmt --check`, `cargo clippy`, `cargo test`, `cargo doc`

**PR Gate**: All commit checks + coverage >95%, mutation >80%, no warnings, 2 approvals

**Release Gate**: All PR checks + benchmarks pass, changelog updated, version bumped, tag signed

## Safety Requirements

- All `unsafe` blocks must have safety comments explaining invariants
- Unsafe code should be <5% of codebase
- No panics in library code (`#![deny(clippy::panic)]`)
- Error types must implement `std::error::Error`
- Corrupted/invalid input must return `Error`, never panic

## Performance Targets

- LZ4 compression: >=3 GB/s throughput (AVX2)
- LZ4 decompression: >=5 GB/s throughput (AVX2)
- SIMD implementations must be >=40% faster than scalar baseline
- P99 page compression latency: <100us
- Working memory: <=64KB per compression call

## System Requirements

- Linux Kernel >= 5.10 LTS with zram module
- x86_64 (AVX2/AVX-512) or AArch64 (NEON)
- Rust stable >= 1.82.0 (MSRV)
- CAP_SYS_ADMIN required for device configuration (trueno-cli, trueno-generator)

## Key Source Files

### trueno-zram-core
- `src/lib.rs`: Public API (`CompressorBuilder`, `PageCompressor` trait)
- `src/error.rs`: Error types
- `src/page.rs`: `CompressedPage`, `CompressionStats`, `PAGE_SIZE`
- `src/simd/detect.rs`: Runtime CPU feature detection
- `src/lz4/compress.rs`: Pure Rust LZ4 compression
- `src/lz4/decompress.rs`: Pure Rust LZ4 decompression
- `src/zstd/compress.rs`: Zstandard compression
- `src/zstd/decompress.rs`: Zstandard decompression

### trueno-zram-adaptive
- `src/entropy.rs`: Shannon entropy calculation
- `src/classifier.rs`: Algorithm selection based on entropy

### trueno-zram-cli
- `src/commands/`: CLI subcommands (create, remove, status, benchmark)


## Contract-First Development

All new features and bug fixes must follow provable-contract-first methodology:
1. Write or update the contract YAML in `../provable-contracts/contracts/<crate>/`
2. Run `pmat comply check` to validate compliance
3. Implement the code to satisfy the contract
4. Run `pmat comply check` again to confirm

## Stack Documentation Search (RAG Oracle)

**IMPORTANT: Proactively use the batuta RAG oracle when:**
- Looking up patterns from other stack components
- Finding cross-language equivalents (Python HuggingFace → Rust)
- Understanding how similar problems are solved elsewhere in the stack

```bash
# Search across the entire Sovereign AI Stack
batuta oracle --rag "your question here"

# Reindex after changes (auto-runs via post-commit hook + ora-fresh)
batuta oracle --rag-index

# Check index freshness (runs automatically on shell login)
ora-fresh
```

The RAG index covers 5000+ documents across the Sovereign AI Stack.
Index auto-updates via post-commit hooks and `ora-fresh` on shell login.
To manually check freshness: `ora-fresh`
To force full reindex: `batuta oracle --rag-index --force`
