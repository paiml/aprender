# Model Loading Architecture

## Abstract

This document specifies the memory-mapped model loading architecture for Realizar,
addressing the critical issue of CPU memory pressure when loading large model files
for GPU inference. The current `fs::read()` approach copies entire model files into
userspace memory, which then gets compressed into zram when transferred to GPU VRAM,
consuming ~10GB of actual RAM to store ~47GB of stale data.

## Problem Statement

### Current State (Muda - Waste)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Current: fs::read() Flow                                                │
├─────────────────────────────────────────────────────────────────────────┤
│  1. fs::read(path) → Vec<u8> allocated in userspace heap                │
│  2. Parse headers, tensors from Vec<u8>                                 │
│  3. Copy tensor data to GPU VRAM via cuMemcpy                           │
│  4. Vec<u8> remains allocated (process still owns it)                   │
│  5. Kernel sees idle pages → compresses to zram                         │
│  6. zram holds ~47GB compressed (uses ~10GB real RAM)                   │
│  7. Pages NEVER freed until process exits                               │
└─────────────────────────────────────────────────────────────────────────┘
```

### Proposed State (Heijunka - Level Loading)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Proposed: mmap() Flow                                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  1. mmap(path) → kernel maps file into address space (no copy)          │
│  2. Parse headers, tensors from mapped region (demand paging)           │
│  3. Copy tensor data to GPU VRAM via cuMemcpy (pages faulted in)        │
│  4. Call madvise(MADV_DONTNEED) after GPU transfer                      │
│  5. Kernel drops pages immediately (backed by file, not swap)           │
│  6. zram usage: ~0 bytes                                                │
│  7. Re-access triggers page fault → re-read from disk (not zram)        │
└─────────────────────────────────────────────────────────────────────────┘
```

## Peer-Reviewed References

### Memory-Mapped I/O Performance

1. **Didona, D., Pfefferle, J., Ioannou, N., Metzler, B., & Trivedi, A. (2022).**
   "Understanding Modern Storage APIs: A systematic study of libaio, SPDK, and io_uring."
   *ACM SIGOPS Operating Systems Review*, 56(1), 8-17.

   - Finding: mmap with `madvise(MADV_SEQUENTIAL)` achieves 95% of direct I/O throughput
   - Finding: mmap outperforms `read()` by 2.3x for sequential access patterns
   - Relevance: Model loading is sequential; mmap is optimal

2. **Chu, H. (2011).** "MDB: A Memory-Mapped Database and Backend for OpenLDAP."
   *OpenLDAP Technical Report*.

   - LMDB design: Pure mmap, no buffer cache duplication
   - Key insight: Let kernel manage pages, don't fight the VM subsystem
   - Relevance: Model files are read-only, perfect mmap use case

3. **Vahalia, U. (1996).** "UNIX Internals: The New Frontiers."
   *Prentice Hall*, Chapter 14.

   - SIGBUS behavior on truncated mmap: Well-documented, mitigatable
   - Page fault cost: ~1000 cycles with TLB miss, amortized over 4KB page
   - Relevance: Safety documentation for unsafe mmap block

4. **McKusick, M. K., & Karels, M. J. (1988).** "Design of a General-Purpose
   Memory Allocator for the 4.3BSD UNIX Kernel."
   *USENIX Summer Conference Proceedings*.

   - Heap fragmentation in long-running processes
   - malloc/free overhead for large allocations
   - Relevance: Vec<u8> from fs::read() suffers these issues

### GPU Memory Transfer Patterns

5. **Harris, M. (2013).** "How to Optimize Data Transfers in CUDA C/C++."
   *NVIDIA Developer Blog*.

   - Pinned memory for DMA transfers
   - mmap pages can be pinned via `mlock()` if needed
   - Relevance: mmap compatible with optimal GPU transfer

6. **Rhu, M., et al. (2016).** "vDNN: Virtualized Deep Neural Networks for
   Scalable, Memory-Efficient Neural Network Design."
   *IEEE/ACM MICRO*.

   - Prefetching patterns for model weights
   - Memory pressure from weight matrices
   - Relevance: Validates need for demand paging

## Design Specification

### API Surface

```rust
/// Memory-mapped model file for zero-copy GPU loading.
///
/// # Safety
///
/// Uses `memmap2::Mmap` which requires:
/// - File must not be truncated while mapped (SIGBUS on Unix)
/// - File must not be modified while mapped (undefined behavior)
///
/// # References
///
/// - Vahalia (1996): SIGBUS from truncated mmap
/// - Didona et al. (2022): mmap vs read() performance
pub struct MappedModel {
    mmap: memmap2::Mmap,
    header: AprHeader,
    tensor_index: Vec<TensorEntry>,
    path: PathBuf,
}

impl MappedModel {
    /// Open a model file with memory mapping.
    ///
    /// # Performance
    ///
    /// - O(1) open time (no data copied)
    /// - Demand paging: only pages accessed are loaded
    /// - After GPU transfer: call `release_cpu_pages()` to free
    pub fn open(path: impl AsRef<Path>) -> Result<Self>;

    /// Get tensor data as a zero-copy slice.
    ///
    /// # Page Fault Behavior
    ///
    /// First access triggers page fault → kernel loads from disk.
    /// Subsequent access: cached in page cache (if memory available).
    pub fn tensor_data(&self, name: &str) -> Option<&[u8]>;

    /// Release CPU pages after GPU transfer.
    ///
    /// Calls `madvise(MADV_DONTNEED)` to tell kernel these pages
    /// are no longer needed. Kernel will:
    /// - Drop pages immediately (not compress to zram)
    /// - Re-fault from disk if accessed again
    ///
    /// # When to Call
    ///
    /// After `cuMemcpy()` completes for all tensors.
    #[cfg(unix)]
    pub fn release_cpu_pages(&self) -> Result<()>;
}
```

### Integration Points

```rust
// Before (apr.rs:635-638)
pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
    let data = fs::read(path.as_ref()).map_err(|e| RealizarError::IoError {
        message: e.to_string(),
    })?;
    // ... parse from data
}

// After
pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
    let mmap = MappedModel::open(path.as_ref())?;
    // ... parse from mmap.as_slice()
    // After GPU transfer:
    #[cfg(unix)]
    mmap.release_cpu_pages()?;
}
```

## Falsification QA Checklist

### Hypothesis: mmap reduces zram usage

| Test | Falsification Criteria | Measurement |
|------|------------------------|-------------|
| H1: mmap pages released | zram usage < 1GB after `release_cpu_pages()` | `cat /sys/block/zram0/mm_stat` |
| H2: No performance regression | Load time within 5% of fs::read | `hyperfine` benchmark |
| H3: GPU transfer intact | Inference output identical | Bit-exact comparison |
| H4: Memory pressure reduced | RSS < model size after GPU xfer | `/proc/[pid]/statm` |

### Boundary Conditions

| Condition | Expected Behavior | Test |
|-----------|-------------------|------|
| File truncated during load | SIGBUS or graceful error | Concurrent truncate test |
| File deleted during load | Continues (inode held) | Unlink while mapped |
| OOM during page fault | SIGKILL or allocation failure | `ulimit -v` stress test |
| Network filesystem | Graceful degradation | NFS mount test |
| WASM target | Falls back to fs::read | `cargo build --target wasm32` |

### Regression Gates

```bash
# Must pass before merge
cargo test --release -- mmap
cargo bench -- model_load  # No regression > 5%
./scripts/zram-stress-test.sh  # zram < 1GB after load
```

## Test Plan (95% Coverage Target)

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    // Core functionality
    #[test] fn test_mapped_model_open_valid_file();
    #[test] fn test_mapped_model_open_nonexistent();
    #[test] fn test_mapped_model_open_invalid_magic();
    #[test] fn test_mapped_model_open_truncated_header();

    // Tensor access
    #[test] fn test_tensor_data_existing();
    #[test] fn test_tensor_data_nonexistent();
    #[test] fn test_tensor_data_bounds_check();

    // Memory management
    #[cfg(unix)]
    #[test] fn test_release_cpu_pages_success();
    #[cfg(unix)]
    #[test] fn test_release_cpu_pages_already_released();

    // Edge cases
    #[test] fn test_empty_model_file();
    #[test] fn test_large_model_file();  // 1GB+ synthetic
    #[test] fn test_concurrent_access();

    // Platform fallback
    #[cfg(target_arch = "wasm32")]
    #[test] fn test_wasm_fallback();
}
```

### Property-Based Tests

```rust
#[cfg(test)]
mod prop_tests {
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_tensor_slice_within_bounds(
            offset in 0usize..10000,
            len in 0usize..5000,
        ) {
            // Verify slice bounds never exceed file
        }

        #[test]
        fn prop_mmap_equals_fsread(
            data in prop::collection::vec(any::<u8>(), 64..10000),
        ) {
            // mmap and fs::read produce identical bytes
        }
    }
}
```

### Integration Tests

```rust
// tests/integration/mmap_loading.rs

#[test]
fn test_apr_model_load_via_mmap() {
    let model = AprV2Model::load("fixtures/tiny-model.apr").unwrap();
    assert!(model.tensor_count() > 0);
}

#[test]
fn test_gpu_transfer_from_mmap() {
    let model = MappedModel::open("fixtures/tiny-model.apr").unwrap();
    let tensor = model.tensor_data("embed_tokens").unwrap();
    // Simulate GPU transfer
    cuda::memcpy_htod(tensor);
    model.release_cpu_pages().unwrap();
}

#[test]
fn test_zram_pressure_after_release() {
    // Load large model, transfer to GPU, release pages
    // Verify zram usage is minimal
}
```

## PMAT Implementation Phases

### Phase 1: Plan (This Document)
- [x] Problem analysis with citations
- [x] API design specification
- [x] Falsification criteria
- [x] Test plan

### Phase 2: Monitor (Implementation)
- [ ] Implement `MappedModel` struct
- [ ] Add `release_cpu_pages()` with madvise
- [ ] Unit tests for all methods
- [ ] Property tests for invariants

### Phase 3: Adjust (Integration)
- [ ] Update `AprV2Model::load()` to use mmap
- [ ] Update `convert.rs` GGUF loading
- [ ] Benchmark comparison
- [ ] zram stress test

### Phase 4: Track (Validation)
- [ ] Coverage report (target: 95%)
- [ ] Performance benchmark (no regression)
- [ ] Memory pressure test (zram < 1GB)
- [ ] CI integration

## Appendix: zram Monitoring Commands

```bash
# Current zram stats
cat /sys/block/zram0/mm_stat
# Fields: orig_data_size compr_data_size mem_used_total ...

# Human-readable
zramctl

# Per-process RSS
ps -o pid,rss,comm -p $(pgrep realizar)

# Page cache for model file
vmtouch /path/to/model.apr
```
