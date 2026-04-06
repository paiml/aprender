# trueno-ublk Overview

trueno-ublk is a GPU-accelerated ZRAM replacement that uses the Linux ublk interface to provide a high-performance compressed block device in userspace.

## Production Status (2026-01-06)

**MILESTONE DT-005 ACHIEVED:** trueno-ublk is running as system swap!

- 8GB device active as primary swap (priority 150)
- CPU SIMD compression at 20-30 GB/s with 3.87x ratio
- CPU parallel decompression at 50+ GB/s

**DT-007 COMPLETED:** Swap deadlock issue FIXED via mlock() - daemon memory pinned.

**Known Limitations:**
- Docker cannot isolate ublk devices (host kernel resources)

## What is ublk?

ublk (userspace block device) is a Linux kernel interface that allows implementing block devices entirely in userspace. It uses io_uring for efficient I/O handling, avoiding the overhead of kernel context switches.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Applications                          │
├─────────────────────────────────────────────────────────┤
│                    Filesystem                            │
├─────────────────────────────────────────────────────────┤
│                /dev/ublkbN block device                  │
├─────────────────────────────────────────────────────────┤
│                    Linux Kernel                          │
│                   (ublk driver)                          │
├─────────────────────────────────────────────────────────┤
│                    io_uring                              │
├─────────────────────────────────────────────────────────┤
│                  trueno-ublk                             │
│  ┌───────────────────────────────────────────────────┐  │
│  │              Entropy Router                        │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────────────┐   │  │
│  │  │  GPU    │  │  SIMD   │  │     Scalar      │   │  │
│  │  │ (batch) │  │ (AVX2)  │  │ (incompressible)│   │  │
│  │  └─────────┘  └─────────┘  └─────────────────┘   │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────┐  │
│  │          trueno-zram-core (LZ4/ZSTD)              │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Features

### SIMD-Accelerated Compression
Uses trueno-zram-core for vectorized LZ4/ZSTD compression:
- AVX2 (256-bit) on modern x86_64
- AVX-512 (512-bit) on supported CPUs
- NEON (128-bit) on ARM64

### GPU Batch Compression
Offloads compression to CUDA GPUs when beneficial:
- Warp-cooperative LZ4 kernel
- PCIe 5x rule evaluation
- Async DMA for overlap

### Entropy-Based Routing
Analyzes data entropy to choose the optimal compression path:
- Low entropy (< 4.0 bits): GPU batch compression
- Medium entropy (4-7 bits): SIMD compression
- High entropy (> 7.0 bits): Scalar or skip compression

### Zero-Page Deduplication
Automatically detects and deduplicates all-zero pages, achieving 2048:1 compression for sparse data.

### zram-Compatible Statistics
Exports statistics in the same format as kernel zram, enabling drop-in monitoring compatibility.

## CLI Usage

```bash
# Create a 1TB compressed RAM disk
trueno-ublk create -s 1T -a lz4 --gpu

# List devices
trueno-ublk list

# Show statistics
trueno-ublk stat /dev/ublkb0

# Interactive dashboard
trueno-ublk top

# Analyze data entropy
trueno-ublk entropy /dev/ublkb0
```

## Requirements

- Linux kernel 6.0+ with ublk support
- CAP_SYS_ADMIN capability (for device creation)
- Optional: NVIDIA GPU with CUDA support
