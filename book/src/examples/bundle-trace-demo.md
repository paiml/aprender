# Case Study: Bundle Trace Demo

This example demonstrates model bundling with renacer syscall tracing for performance analysis.

## Running the Demo

```bash
# Build the demo
cargo build --example bundle_trace_demo

# Run normally
./target/debug/examples/bundle_trace_demo

# Trace with renacer
renacer -e trace=file -T -c -- ./target/debug/examples/bundle_trace_demo
```

## What This Example Does

The demo performs three operations to showcase the bundle module:

1. **Creates a bundle** with three models (encoder, decoder, classifier)
2. **Loads the entire bundle** into memory
3. **Loads with memory paging** using a 1KB limit to force evictions

## Example Output

```
=== Model Bundling and Memory Paging Demo ===

1. Creating bundle with 3 models...
   - Encoder: 500 bytes
   - Decoder: 500 bytes
   - Classifier: 300 bytes
   Bundle created with 3 models
   Total size: 1300 bytes

2. Loading bundle into memory...
   Loaded 3 models:
   - encoder: 500 bytes
   - decoder: 500 bytes
   - classifier: 300 bytes

3. Loading with memory paging (limited to 1KB)...
   Memory limit: 1024 bytes
   Initially cached: 0 models

   Accessing encoder...
   - Loaded encoder: 500 bytes
   - Cached: 1, Memory used: 500 bytes

   Accessing decoder...
   - Loaded decoder: 500 bytes
   - Cached: 2, Memory used: 1000 bytes

   Accessing classifier...
   - Loaded classifier: 300 bytes
   - Cached: 2, Memory used: 800 bytes

   Paging Statistics:
   - Hits: 0
   - Misses: 3
   - Evictions: 1
   - Hit rate: 0.0%
   - Total bytes loaded: 1300
```

## Source Code

```rust
use aprender::bundle::{BundleBuilder, BundleConfig, ModelBundle, PagedBundle, PagingConfig};

fn main() {
    let bundle_path = "/tmp/demo_bundle.apbundle";

    // Create a bundle with 3 models
    let bundle = BundleBuilder::new(bundle_path)
        .with_config(BundleConfig::new().with_compression(false))
        .add_model("encoder", vec![1u8; 500])
        .add_model("decoder", vec![2u8; 500])
        .add_model("classifier", vec![3u8; 300])
        .build()
        .expect("Failed to create bundle");

    // Load with memory paging (1KB limit)
    let config = PagingConfig::new()
        .with_max_memory(1024)
        .with_prefetch(false);

    let mut paged = PagedBundle::open(bundle_path, config).unwrap();

    // Each access may trigger loading/eviction
    let _ = paged.get_model("encoder");   // Load
    let _ = paged.get_model("decoder");   // Load (total: 1000 bytes)
    let _ = paged.get_model("classifier"); // Evict encoder, load classifier
}
```

## Tracing with Renacer

Use renacer to see syscall-level I/O patterns:

```bash
$ renacer -e trace=file -T -c -- ./target/debug/examples/bundle_trace_demo

% time     seconds  usecs/call     calls    errors syscall
------ ----------- ----------- --------- --------- ----------------
 36.86    0.000258           8        32           write
 19.71    0.000138           8        17           read
  8.29    0.000058           7         8           close
  7.57    0.000053           6         8           lseek
 17.29    0.000121          15         8           openat
```

Key observations:
- **32 writes**: Bundle creation + stdout output
- **17 reads**: Manifest reads + model data loads
- **8 lseek**: Seeking to different model offsets (indicates paging)

## See Also

- [Tracing Memory Paging with Renacer](./tracing-memory-paging.md) - Comprehensive tracing guide
- [Model Bundling and Memory Paging](./model-bundling-paging.md) - Full bundle API documentation
