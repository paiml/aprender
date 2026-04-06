# Block Device API

trueno-ublk provides a `BlockDevice` type for creating compressed block devices in pure Rust, without requiring kernel privileges.

## Basic Usage

```rust
use trueno_ublk::BlockDevice;
use trueno_zram_core::{Algorithm, CompressorBuilder, PAGE_SIZE};

// Create a compressor
let compressor = CompressorBuilder::new()
    .algorithm(Algorithm::Lz4)
    .build()?;

// Create a 1GB block device
let mut device = BlockDevice::new(1 << 30, compressor);

// Write data (must be page-aligned)
let data = vec![0xAB; PAGE_SIZE];
device.write(0, &data)?;

// Read back
let mut buf = vec![0u8; PAGE_SIZE];
device.read(0, &mut buf)?;
assert_eq!(data, buf);
```

## Page Size

All I/O operations must be aligned to `PAGE_SIZE` (4096 bytes):

```rust
use trueno_zram_core::PAGE_SIZE;

// Write at page boundaries
device.write(0, &data)?;                    // Page 0
device.write(PAGE_SIZE as u64, &data)?;     // Page 1
device.write(2 * PAGE_SIZE as u64, &data)?; // Page 2
```

## Statistics

Track compression performance with `BlockDeviceStats`:

```rust
let stats = device.stats();

println!("Pages stored:      {}", stats.pages_stored);
println!("Bytes written:     {}", stats.bytes_written);
println!("Bytes compressed:  {}", stats.bytes_compressed);
println!("Compression ratio: {:.2}x", stats.compression_ratio());
println!("Zero pages:        {}", stats.zero_pages);
println!("GPU pages:         {}", stats.gpu_pages);
println!("SIMD pages:        {}", stats.simd_pages);
println!("Scalar pages:      {}", stats.scalar_pages);
```

## Entropy Threshold

Configure entropy-based routing with `with_entropy_threshold`:

```rust
// Use threshold of 7.0 bits/byte
let device = BlockDevice::with_entropy_threshold(
    1 << 30,         // 1GB
    compressor,
    7.0,             // Entropy threshold
);
```

Data with entropy above this threshold is routed to the scalar path (assumed incompressible).

## Discard Operation

Free pages that are no longer needed:

```rust
// Discard page 0
device.discard(0, PAGE_SIZE as u64)?;

// Reading discarded pages returns zeros
let mut buf = vec![0xFFu8; PAGE_SIZE];
device.read(0, &mut buf)?;
assert!(buf.iter().all(|&b| b == 0));
```

## UblkDevice (Kernel Interface)

For creating actual block devices visible to the system, use `UblkDevice`:

```rust
use trueno_ublk::{UblkDevice, DeviceConfig};
use trueno_zram_core::Algorithm;

let config = DeviceConfig {
    size: 1 << 40,  // 1TB
    algorithm: Algorithm::Lz4,
    streams: 4,
    gpu_enabled: true,
    mem_limit: Some(8 << 30),  // 8GB RAM limit
    backing_dev: None,
    writeback_limit: None,
    entropy_skip_threshold: 7.5,
    gpu_batch_size: 1024,
};

// Requires CAP_SYS_ADMIN
let device = UblkDevice::create(config)?;
println!("Created device: /dev/ublkb{}", device.id());
```

## DeviceStats

The `DeviceStats` struct provides zram-compatible statistics:

```rust
let stats = device.stats();

// mm_stat fields (zram compatible)
println!("Original data:    {} bytes", stats.orig_data_size);
println!("Compressed data:  {} bytes", stats.compr_data_size);
println!("Memory used:      {} bytes", stats.mem_used_total);
println!("Same pages:       {}", stats.same_pages);
println!("Huge pages:       {}", stats.huge_pages);

// io_stat fields
println!("Failed reads:     {}", stats.failed_reads);
println!("Failed writes:    {}", stats.failed_writes);

// trueno-ublk extensions
println!("GPU pages:        {}", stats.gpu_pages);
println!("SIMD pages:       {}", stats.simd_pages);
println!("Throughput:       {:.2} GB/s", stats.throughput_gbps);
println!("Avg entropy:      {:.2} bits", stats.avg_entropy);
println!("SIMD backend:     {}", stats.simd_backend);
```
