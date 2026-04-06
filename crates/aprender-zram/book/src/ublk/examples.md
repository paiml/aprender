# trueno-ublk Examples

trueno-ublk includes several examples demonstrating its features.

## Running Examples

```bash
# Basic block device usage
cargo run --example block_device -p trueno-ublk

# Compression statistics comparison
cargo run --example compression_stats -p trueno-ublk

# Entropy-based routing demonstration
cargo run --example entropy_routing -p trueno-ublk

# v3.17.0 Examples
# ----------------

# Visualization demo (VIZ-001/002/003/004)
cargo run --example visualization_demo -p trueno-ublk

# ZSTD vs LZ4 performance comparison (use --release for accurate benchmarks)
cargo run --example zstd_vs_lz4 -p trueno-ublk --release

# Tiered storage architecture demo (KERN-001/002/003)
cargo run --example tiered_storage -p trueno-ublk

# Batched compression benchmark
cargo run --example batched_benchmark -p trueno-ublk --release
```

## Example: Basic Block Device

Demonstrates creating a compressed block device, writing data, and reading it back.

```rust
use trueno_ublk::BlockDevice;
use trueno_zram_core::{Algorithm, CompressorBuilder, PAGE_SIZE};

fn main() -> anyhow::Result<()> {
    // Create an LZ4 compressor
    let compressor = CompressorBuilder::new()
        .algorithm(Algorithm::Lz4)
        .build()?;

    // Create a 64MB block device
    let mut device = BlockDevice::new(64 * 1024 * 1024, compressor);

    // Write different patterns
    let compressible = vec![0xAA; PAGE_SIZE];
    device.write(0, &compressible)?;

    let zeros = vec![0u8; PAGE_SIZE];
    device.write(PAGE_SIZE as u64, &zeros)?;

    // Read back and verify
    let mut buf = vec![0u8; PAGE_SIZE];
    device.read(0, &mut buf)?;
    assert_eq!(buf, compressible);

    // Check statistics
    let stats = device.stats();
    println!("Compression ratio: {:.2}x", stats.compression_ratio());

    Ok(())
}
```

**Output:**
```
trueno-ublk Block Device Example
=================================

Created LZ4 compressor with SIMD backend
Created block device: 64 MB

Writing test patterns...
  Page 0: Highly compressible (all 0xAA)
  Page 1: Zero page
  Page 2: Sequential data
  Page 3: Pseudo-random data

Reading back and verifying...
  Page 0: OK
  Page 1: OK
  Page 2: OK
  Page 3: OK

Device Statistics:
  Pages stored:      4
  Bytes written:     16 KB
  Bytes compressed:  0 KB
  Compression ratio: 27.86x
  Zero pages:        1

All tests passed!
```

## Example: Compression Statistics

Compares LZ4 and ZSTD compression on various data patterns.

```rust
use trueno_ublk::BlockDevice;
use trueno_zram_core::{Algorithm, CompressorBuilder, PAGE_SIZE};

fn main() -> anyhow::Result<()> {
    let test_data = vec![
        ("All zeros", vec![0u8; PAGE_SIZE]),
        ("Repeating pattern", (0..PAGE_SIZE).map(|i| (i % 4) as u8).collect()),
        ("High entropy", (0..PAGE_SIZE).map(|i| ((i * 17 + 31) % 256) as u8).collect()),
    ];

    for algorithm in [Algorithm::Lz4, Algorithm::Zstd { level: 3 }] {
        let compressor = CompressorBuilder::new()
            .algorithm(algorithm)
            .build()?;

        let mut device = BlockDevice::new(64 * 1024 * 1024, compressor);

        for (i, (_, data)) in test_data.iter().enumerate() {
            device.write((i * PAGE_SIZE) as u64, data)?;
        }

        let stats = device.stats();
        println!("{:?}: {:.2}x compression", algorithm, stats.compression_ratio());
    }

    Ok(())
}
```

**Output:**
```
Algorithm: Lz4
------------------------------------------------------------
  Results:
    Compression ratio: 27.25x
    Space savings:     96.3%

Algorithm: Zstd { level: 3 }
------------------------------------------------------------
  Results:
    Compression ratio: 1.50x
    Space savings:     33.3%
```

## Example: Entropy Routing

Shows how data is routed to different backends based on entropy.

```rust
use trueno_ublk::BlockDevice;
use trueno_zram_core::{Algorithm, CompressorBuilder, PAGE_SIZE};

fn main() -> anyhow::Result<()> {
    let compressor = CompressorBuilder::new()
        .algorithm(Algorithm::Lz4)
        .build()?;

    // Set entropy threshold to 7.0 bits/byte
    let mut device = BlockDevice::with_entropy_threshold(
        64 * 1024 * 1024,
        compressor,
        7.0,
    );

    // Low entropy - routed to GPU
    let low_entropy = vec![0u8; PAGE_SIZE];
    device.write(0, &low_entropy)?;

    // Medium entropy - routed to SIMD
    let medium: Vec<u8> = (0..PAGE_SIZE).map(|i| (i % 16) as u8).collect();
    device.write(PAGE_SIZE as u64, &medium)?;

    // High entropy - routed to scalar
    let high: Vec<u8> = (0..PAGE_SIZE).map(|i| ((i * 17 + 31) % 256) as u8).collect();
    device.write(2 * PAGE_SIZE as u64, &high)?;

    let stats = device.stats();
    println!("GPU pages:    {}", stats.gpu_pages);
    println!("SIMD pages:   {}", stats.simd_pages);
    println!("Scalar pages: {}", stats.scalar_pages);
    println!("Zero pages:   {}", stats.zero_pages);

    Ok(())
}
```

**Output:**
```
trueno-ublk Entropy Routing Example
===================================

Routing Statistics:
  GPU pages:    1 (low entropy, highly compressible)
  SIMD pages:   1 (medium entropy, normal data)
  Scalar pages: 3 (high entropy, incompressible)
  Zero pages:   1 (all zeros, deduplicated)

Total compression ratio: 2.88x
```

## CLI Examples

### Create a Device

```bash
# Create 1TB device with LZ4 and GPU acceleration
trueno-ublk create -s 1T -a lz4 --gpu

# Create with memory limit
trueno-ublk create -s 512G -a zstd --mem-limit 8G

# Create with custom entropy threshold
trueno-ublk create -s 256G -a lz4 --entropy-threshold 6.5
```

### Monitor Devices

```bash
# List all devices
trueno-ublk list

# Show detailed stats
trueno-ublk stat /dev/ublkb0

# Interactive dashboard
trueno-ublk top
```

### Manage Devices

```bash
# Reset statistics
trueno-ublk reset /dev/ublkb0

# Compact memory
trueno-ublk compact /dev/ublkb0

# Set runtime parameters
trueno-ublk set /dev/ublkb0 --mem-limit 16G
```
