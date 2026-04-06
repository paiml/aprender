# Chapter: Memory-Efficient Model Loading

## Introduction

This chapter covers memory-mapped model loading in Realizar, designed to eliminate zram pressure when loading large models for GPU inference. We follow Toyota Production System principles (Muda elimination, Heijunka) and provide peer-reviewed citations for all design decisions.

## Prerequisites

- Understanding of virtual memory and demand paging
- Familiarity with GPU memory transfer patterns
- Basic knowledge of Linux memory management (optional but helpful)

## Learning Objectives

By the end of this chapter, you will understand:
1. Why `fs::read()` causes zram accumulation
2. How memory-mapped I/O solves this problem
3. When to call `release_cpu_pages()` for optimal memory usage
4. How to monitor zram and verify the optimization

## 1. The Problem: zram Accumulation

### Symptom

After loading several large models for GPU inference, you notice:
- High swap usage (e.g., 47GB in zram)
- zram consuming real RAM for compression (e.g., ~10GB)
- Memory pressure even with 125GB+ RAM

### Root Cause

The traditional loading pattern:

```rust
// BAD: Entire file copied to heap
let data = std::fs::read("model.apr")?;  // 8GB allocated
cuda::memcpy_htod(gpu_ptr, &data);        // Copied to GPU
// data still in RAM, kernel compresses to zram
// NEVER freed until process exits
```

The kernel sees idle pages and compresses them to zram. But zram holds them indefinitely because the process still owns the `Vec<u8>`.

### Memory Flow (Before)

```
┌─────────────────────────────────────────────────────────────┐
│ fs::read() Flow                                             │
├─────────────────────────────────────────────────────────────┤
│  1. fs::read(path) → Vec<u8> allocated in userspace heap    │
│  2. Parse headers, tensors from Vec<u8>                     │
│  3. Copy tensor data to GPU VRAM via cuMemcpy               │
│  4. Vec<u8> remains allocated (process still owns it)       │
│  5. Kernel sees idle pages → compresses to zram             │
│  6. zram holds ~47GB compressed (uses ~10GB real RAM)       │
│  7. Pages NEVER freed until process exits                   │
└─────────────────────────────────────────────────────────────┘
```

## 2. The Solution: Memory-Mapped Loading

### How mmap Works

Memory mapping creates a virtual address space backed by a file:
- No data copied on open (O(1) startup)
- Pages loaded on-demand when accessed (page faults)
- Kernel can drop pages anytime (backed by file, not swap)

### Memory Flow (After)

```
┌─────────────────────────────────────────────────────────────┐
│ mmap() Flow                                                 │
├─────────────────────────────────────────────────────────────┤
│  1. mmap(path) → kernel maps file into address space        │
│  2. Parse headers, tensors (pages faulted in on access)     │
│  3. Copy tensor data to GPU VRAM via cuMemcpy               │
│  4. Call madvise(MADV_DONTNEED) after GPU transfer          │
│  5. Kernel drops pages immediately (not zram)               │
│  6. zram usage: ~0 bytes                                    │
│  7. Re-access triggers page fault → re-read from disk       │
└─────────────────────────────────────────────────────────────┘
```

### Implementation

Realizar's `AprV2Model::load()` automatically uses mmap for uncompressed files:

```rust
use realizar::apr::AprV2Model;

// Load with mmap (automatic for uncompressed files)
let model = AprV2Model::load("model.apr")?;

// Verify mmap is being used
assert!(model.is_mmap());

// Transfer tensors to GPU
for name in model.tensor_names() {
    let bytes = model.get_tensor_bytes(&name)?;
    cuda::memcpy_htod(gpu_ptr, bytes);
}

// KEY: Release CPU pages after GPU transfer
#[cfg(unix)]
model.release_cpu_pages()?;

// Pages now backed by file, not zram!
```

## 3. API Reference

### `AprV2Model::load(path)`

Loads an APR model using the optimal strategy:
- **Uncompressed files**: Uses mmap (zero-copy)
- **Compressed files**: Falls back to heap after decompression

```rust
pub fn load<P: AsRef<Path>>(path: P) -> Result<Self>
```

### `AprV2Model::from_bytes(data)`

Loads from a `Vec<u8>` (always uses heap):
- Use for compressed files after decompression
- Use for data received over network
- Use in WASM environments (no mmap support)

```rust
pub fn from_bytes(data: Vec<u8>) -> Result<Self>
```

### `AprV2Model::is_mmap()`

Returns `true` if the model is using memory-mapped I/O:

```rust
pub fn is_mmap(&self) -> bool
```

### `AprV2Model::release_cpu_pages()` (Unix only)

Advises the kernel that mapped pages are no longer needed:

```rust
#[cfg(unix)]
pub fn release_cpu_pages(&self) -> Result<()>
```

**When to call**: After all tensor data has been copied to GPU.

**Effect**: Kernel drops pages immediately. Re-access will fault from disk (not zram).

## 4. Monitoring and Verification

### Check zram Usage

```bash
# Raw stats (bytes)
cat /sys/block/zram0/mm_stat
# Fields: orig_data_size compr_data_size mem_used_total ...

# Human-readable
zramctl

# Example output:
# NAME       ALGORITHM DISKSIZE   DATA  COMPR  TOTAL STREAMS MOUNTPOINT
# /dev/zram0 lz4            4T  47.2G  9.8G  10.1G      24 [SWAP]
```

### Verify Optimization

Before `release_cpu_pages()`:
```bash
$ cat /sys/block/zram0/mm_stat
50238103552 10149679679 10536153088 ...
# ~47GB uncompressed, ~10GB actual RAM
```

After `release_cpu_pages()`:
```bash
$ cat /sys/block/zram0/mm_stat
1048576 524288 1048576 ...
# ~1MB (minimal baseline)
```

### Per-Process Memory

```bash
# Check RSS (Resident Set Size)
ps -o pid,rss,comm -p $(pgrep realizar)

# Check page cache for model file
vmtouch /path/to/model.apr
```

## 5. Example: Complete GPU Loading Pipeline

```rust
use realizar::apr::AprV2Model;
use std::time::Instant;

fn load_model_to_gpu(path: &str) -> Result<()> {
    println!("Loading model: {}", path);

    // 1. Load with mmap
    let start = Instant::now();
    let model = AprV2Model::load(path)?;
    println!("  Load time: {:?}", start.elapsed());
    println!("  Using mmap: {}", model.is_mmap());

    // 2. Transfer to GPU
    let transfer_start = Instant::now();
    for name in model.tensor_names() {
        let bytes = model.get_tensor_bytes(&name)?;
        // cuda::memcpy_htod(gpu_ptr, bytes);
    }
    println!("  Transfer time: {:?}", transfer_start.elapsed());

    // 3. Release CPU pages (critical for memory efficiency!)
    #[cfg(unix)]
    {
        model.release_cpu_pages()?;
        println!("  CPU pages released");
    }

    // Model is now fully on GPU, CPU memory freed
    Ok(())
}
```

Run the example:
```bash
cargo run --example apr_mmap_loading -- /path/to/model.apr
```

## 6. Design Rationale

### Why mmap over read()?

| Aspect | `fs::read()` | `mmap()` |
|--------|-------------|----------|
| Startup time | O(n) - reads entire file | O(1) - just maps |
| Memory copy | Full copy to heap | Zero-copy |
| Page management | Process owns pages | Kernel manages |
| After GPU transfer | Stuck in zram | Dropped immediately |
| Re-access cost | Zero (in RAM/zram) | Page fault (cheap on SSD) |

### When to Use Heap (`from_bytes`)

- **Compressed files**: Must decompress before use
- **Network data**: No file to map
- **WASM**: No mmap support in browsers

### References

1. **Didona et al. (2022)**: "Understanding Modern Storage APIs" - mmap achieves 2.3x throughput vs read() for sequential access
2. **Chu (2011)**: LMDB design - let kernel manage pages, don't fight the VM subsystem
3. **Vahalia (1996)**: "UNIX Internals" - SIGBUS behavior and mmap safety

## 7. Troubleshooting

### "mmap failed: Cannot allocate memory"

Virtual address space exhausted. Check `/proc/sys/vm/max_map_count`:
```bash
sysctl vm.max_map_count
# Increase if needed:
sudo sysctl -w vm.max_map_count=262144
```

### SIGBUS on Access

File was truncated while mapped. Ensure:
- No concurrent writes to model files
- File is on local storage (not network filesystem)

### release_cpu_pages() Has No Effect

- Verify `is_mmap()` returns `true`
- For heap-allocated data, `release_cpu_pages()` is a no-op
- Check that you're on Unix (not Windows/WASM)

## Summary

- Use `AprV2Model::load()` for automatic mmap
- Call `release_cpu_pages()` after GPU transfer
- Monitor with `zramctl` and `/sys/block/zram0/mm_stat`
- For compressed/network data, use `from_bytes()` (heap fallback)
