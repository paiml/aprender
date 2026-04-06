# trueno-ublk: GPU Block Device

**trueno-ublk** provides a GPU-accelerated ZRAM replacement using Linux's userspace block device (ublk) interface. It achieves 10-50 GB/s throughput by offloading compression to GPU.

## Overview

trueno-ublk delivers:
- **ublk Driver**: Userspace block device via libublk
- **GPU Compression**: CUDA/wgpu accelerated
- **ZRAM Replacement**: Drop-in swap device
- **Adaptive Backend**: Automatic GPU/SIMD/CPU selection
- **High Throughput**: 10-50 GB/s with GPU

```
┌─────────────────────────────────────────────────────────────┐
│                      Linux Kernel                           │
│                    /dev/ublkb0                              │
└───────────────────────┬─────────────────────────────────────┘
                        │ io_uring
┌───────────────────────▼─────────────────────────────────────┐
│                    trueno-ublk                              │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ GPU Backend │  │ SIMD Backend│  │   CPU Backend       │  │
│  │ (CUDA/wgpu) │  │ (AVX/NEON)  │  │   (fallback)        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```toml
[dependencies]
trueno-ublk = "0.1"

# With CUDA support (NVIDIA GPUs)
trueno-ublk = { version = "0.1", features = ["cuda"] }
```

System requirements:
- Linux kernel 6.0+ (ublk support)
- libublk userspace library
- Root privileges for device creation

## Quick Start

```rust
use trueno_ublk::{UblkDevice, DeviceConfig, Backend};

// Create device with 8GB capacity
let config = DeviceConfig {
    capacity_bytes: 8 * 1024 * 1024 * 1024,  // 8 GB
    queue_depth: 128,
    num_queues: 4,
    backend: Backend::Auto,  // Auto-select GPU/SIMD/CPU
};

let device = UblkDevice::create(config).await?;
println!("Created: /dev/{}", device.name());

// Run the device (blocks until shutdown)
device.run().await?;
```

## Backend Selection

| Backend | Throughput | Latency | Condition |
|---------|------------|---------|-----------|
| CUDA | 50+ GB/s | 100 us | NVIDIA GPU |
| wgpu | 20+ GB/s | 200 us | Any GPU |
| AVX-512 | 13 GB/s | 10 us | x86_64 |
| AVX2 | 3 GB/s | 5 us | x86_64 |
| NEON | 2 GB/s | 5 us | ARM64 |
| Scalar | 800 MB/s | 2 us | Fallback |

```rust
use trueno_ublk::Backend;

// Force specific backend
let config = DeviceConfig {
    backend: Backend::Cuda,  // NVIDIA GPU only
    ..Default::default()
};

// Or use adaptive (switches based on load)
let config = DeviceConfig {
    backend: Backend::Adaptive {
        gpu_batch_threshold: 64,  // Use GPU for 64+ pages
    },
    ..Default::default()
};
```

## CLI Usage

```bash
# Create 8GB GPU-accelerated swap
sudo trueno-ublk --capacity 8G --backend auto

# Force CUDA backend with stats
sudo trueno-ublk --capacity 16G --backend cuda --stats

# Use as block device (not swap)
sudo trueno-ublk --capacity 4G --no-swap
sudo mkfs.ext4 /dev/ublkb0
sudo mount /dev/ublkb0 /mnt/fast-storage
```

## systemd Integration

`/etc/systemd/system/trueno-ublk.service`:

```ini
[Unit]
Description=trueno-ublk GPU-accelerated swap
Before=swap.target

[Service]
Type=simple
ExecStart=/usr/local/bin/trueno-ublk \
    --capacity 16G \
    --backend auto
ExecStartPost=/sbin/mkswap /dev/ublkb0
ExecStartPost=/sbin/swapon -p 100 /dev/ublkb0

[Install]
WantedBy=swap.target
```

Enable:
```bash
sudo systemctl enable trueno-ublk
sudo systemctl start trueno-ublk
```

## Performance Monitoring

```rust
use trueno_ublk::Stats;

let stats = device.stats();

println!("Compression ratio: {:.2}x", stats.compression_ratio);
println!("Read throughput:   {:.1} GB/s", stats.read_gbps);
println!("Write throughput:  {:.1} GB/s", stats.write_gbps);
println!("Backend:           {:?}", stats.active_backend);
println!("GPU utilization:   {:.0}%", stats.gpu_utilization * 100.0);
```

Example output:
```
┌─────────────────────────────────────────────────────┐
│ trueno-ublk stats                                   │
├─────────────────────────────────────────────────────┤
│ Device:          /dev/ublkb0                        │
│ Capacity:        16 GB                              │
│ Used:            8.2 GB (51%)                       │
│ Compressed:      2.1 GB (3.9x ratio)                │
│ Backend:         CUDA (RTX 4090)                    │
│ Read:            42.3 GB/s                          │
│ Write:           38.7 GB/s                          │
│ GPU util:        23%                                │
└─────────────────────────────────────────────────────┘
```

## Comparison with zram

| Feature | zram | trueno-ublk |
|---------|------|-------------|
| Compression | CPU only | GPU/SIMD/CPU |
| Throughput | ~1 GB/s | 10-50 GB/s |
| Algorithms | LZ4/ZSTD | LZ4/ZSTD + custom |
| Batch process | No | Yes (GPU) |
| Adaptive | No | Yes |
| Kernel req | Any | 6.0+ (ublk) |

## Running the Example

```bash
cargo run --example trueno_ublk_demo
```

Note: Running the actual ublk driver requires root privileges and Linux 6.0+.

## Related Crates

- **trueno-zram-core**: SIMD compression algorithms used by trueno-ublk
- **trueno-zram-adaptive**: Entropy-based algorithm selection
- **trueno**: SIMD/GPU compute primitives

## References

- [trueno-ublk on crates.io](https://crates.io/crates/trueno-ublk)
- [ublk kernel documentation](https://docs.kernel.org/block/ublk.html)
- [libublk](https://github.com/ming1/libublk-rs)

---

**Navigate:** [Table of Contents](../SUMMARY.md) | [Previous: trueno-zram](./trueno-zram.md) | [Next: Aprender](./aprender.md)
