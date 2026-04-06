# Kernel Compatibility API

The `compat` module provides sysfs interface compatibility with kernel zram.

## SysfsInterface

Emulates the kernel zram sysfs interface:

```rust
use trueno_zram_core::compat::SysfsInterface;

let mut interface = SysfsInterface::new();

// Configure like kernel zram
interface.write_attr("disksize", "4G")?;
interface.write_attr("comp_algorithm", "lz4")?;
interface.write_attr("mem_limit", "2G")?;

// Read attributes
let disksize = interface.read_attr("disksize")?;
let algorithm = interface.read_attr("comp_algorithm")?;
```

## Supported Attributes

| Attribute | Read | Write | Description |
|-----------|------|-------|-------------|
| `disksize` | Yes | Yes | Disk size in bytes |
| `comp_algorithm` | Yes | Yes | Compression algorithm |
| `mem_limit` | Yes | Yes | Memory limit |
| `mem_used_max` | Yes | No | Peak memory usage |
| `mem_used_total` | Yes | No | Current memory usage |
| `orig_data_size` | Yes | No | Original data size |
| `compr_data_size` | Yes | No | Compressed data size |
| `num_reads` | Yes | No | Read count |
| `num_writes` | Yes | No | Write count |
| `invalid_io` | Yes | No | Invalid I/O count |
| `notify_free` | Yes | No | Free notifications |
| `reset` | No | Yes | Reset device |

## Algorithm Support

```rust
use trueno_zram_core::compat::{ZramAlgorithm, is_algorithm_supported};

// Check algorithm support
assert!(is_algorithm_supported("lz4"));
assert!(is_algorithm_supported("zstd"));
assert!(is_algorithm_supported("lzo"));  // Compatibility alias

// Parse algorithm
let algo = "lz4".parse::<ZramAlgorithm>()?;
```

### Supported Algorithms

| Name | Alias | Description |
|------|-------|-------------|
| `lz4` | - | LZ4 fast compression |
| `lz4hc` | - | LZ4 high compression |
| `zstd` | - | Zstandard |
| `lzo` | `lzo-rle` | LZO (mapped to LZ4) |
| `842` | `deflate` | 842/deflate (mapped to ZSTD) |

## Statistics

### MmStat

Memory statistics (like `/sys/block/zram0/mm_stat`):

```rust
use trueno_zram_core::compat::MmStat;

let stat = interface.mm_stat();

println!("Original size: {} bytes", stat.orig_data_size);
println!("Compressed size: {} bytes", stat.compr_data_size);
println!("Memory used: {} bytes", stat.mem_used_total);
println!("Memory limit: {} bytes", stat.mem_limit);
println!("Memory max: {} bytes", stat.mem_used_max);
println!("Same pages: {}", stat.same_pages);
println!("Pages stored: {}", stat.pages_compacted);
println!("Huge pages: {}", stat.huge_pages);
```

### IoStat

I/O statistics (like `/sys/block/zram0/io_stat`):

```rust
use trueno_zram_core::compat::IoStat;

let stat = interface.io_stat();

println!("Reads: {}", stat.num_reads);
println!("Writes: {}", stat.num_writes);
println!("Invalid I/O: {}", stat.invalid_io);
println!("Notify free: {}", stat.notify_free);
```

## Device Reset

```rust
// Reset all statistics and data
interface.write_attr("reset", "1")?;
```

## Integration Example

```rust
use trueno_zram_core::compat::SysfsInterface;
use trueno_zram_core::PAGE_SIZE;

let mut interface = SysfsInterface::new();

// Configure
interface.write_attr("disksize", "1G")?;
interface.write_attr("comp_algorithm", "lz4")?;

// Simulate writes
let page = [0xAA; PAGE_SIZE];
interface.write_page(0, &page)?;
interface.write_page(1, &page)?;

// Check statistics
let mm = interface.mm_stat();
println!("Compression ratio: {:.2}x",
    mm.orig_data_size as f64 / mm.compr_data_size as f64);
```
