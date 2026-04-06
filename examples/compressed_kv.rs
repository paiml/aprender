//! Compressed KV Store Demo (GH-5)
//!
//! Run with: `cargo run --example compressed_kv --features compression`
//!
//! This example demonstrates LZ4/ZSTD transparent compression for KV stores,
//! ideal for reducing memory footprint of LLM KV caches.

#[cfg(feature = "compression")]
use trueno_db::kv::{CompressedKvStore, Compression, KvStore, MemoryKvStore};

#[cfg(feature = "compression")]
#[tokio::main]
async fn main() -> trueno_db::Result<()> {
    println!("=== Trueno-DB Compressed KV Store Demo ===\n");

    // Demo LZ4 compression (fast, real-time)
    demo_lz4_compression().await?;

    // Demo ZSTD compression (better ratio)
    demo_zstd_compression().await?;

    // Demo compression ratio comparison
    demo_compression_comparison().await?;

    // Demo real-world KV cache use case
    demo_kv_cache_simulation().await?;

    println!("All demos completed successfully!");
    Ok(())
}

#[cfg(feature = "compression")]
async fn demo_lz4_compression() -> trueno_db::Result<()> {
    println!("1. LZ4 Compression (Fast, Real-Time)");
    println!("   ---------------------------------");

    let inner = MemoryKvStore::new();
    let store = CompressedKvStore::new(inner, Compression::Lz4);

    // Store some data
    let data = b"Hello, world! ".repeat(100); // Repetitive data compresses well
    store.set("greeting", data.clone()).await?;

    println!("   Algorithm: {}", store.compression().as_str());
    println!("   Original size: {} bytes", data.len());

    // Check compressed size in inner store
    let compressed = store.inner().get("greeting").await?.unwrap();
    println!("   Compressed size: {} bytes", compressed.len());
    println!("   Compression ratio: {:.1}x", data.len() as f64 / compressed.len() as f64);

    // Verify roundtrip
    let retrieved = store.get("greeting").await?.unwrap();
    assert_eq!(retrieved, data);
    println!("   Roundtrip: OK");

    println!();
    Ok(())
}

#[cfg(feature = "compression")]
async fn demo_zstd_compression() -> trueno_db::Result<()> {
    println!("2. ZSTD Compression (Better Ratio)");
    println!("   --------------------------------");

    let inner = MemoryKvStore::new();
    let store = CompressedKvStore::new(inner, Compression::Zstd);

    // Store some data
    let data = b"The quick brown fox jumps over the lazy dog. ".repeat(50);
    store.set("sentence", data.clone()).await?;

    println!("   Algorithm: {}", store.compression().as_str());
    println!("   Original size: {} bytes", data.len());

    let compressed = store.inner().get("sentence").await?.unwrap();
    println!("   Compressed size: {} bytes", compressed.len());
    println!("   Compression ratio: {:.1}x", data.len() as f64 / compressed.len() as f64);

    // Verify roundtrip
    let retrieved = store.get("sentence").await?.unwrap();
    assert_eq!(retrieved, data);
    println!("   Roundtrip: OK");

    println!();
    Ok(())
}

#[cfg(feature = "compression")]
async fn demo_compression_comparison() -> trueno_db::Result<()> {
    println!("3. Compression Algorithm Comparison");
    println!("   ---------------------------------");

    // Test data: simulated embeddings (random-ish floats)
    let embeddings: Vec<u8> = (0..4096).map(|i| ((i * 17 + 31) % 256) as u8).collect();

    // LZ4
    let lz4_store = CompressedKvStore::new(MemoryKvStore::new(), Compression::Lz4);
    lz4_store.set("embed", embeddings.clone()).await?;
    let lz4_size = lz4_store.inner().get("embed").await?.unwrap().len();

    // ZSTD
    let zstd_store = CompressedKvStore::new(MemoryKvStore::new(), Compression::Zstd);
    zstd_store.set("embed", embeddings.clone()).await?;
    let zstd_size = zstd_store.inner().get("embed").await?.unwrap().len();

    println!("   Test data: {} bytes (simulated embeddings)", embeddings.len());
    println!(
        "   LZ4:  {} bytes ({:.1}x ratio)",
        lz4_size,
        embeddings.len() as f64 / lz4_size as f64
    );
    println!(
        "   ZSTD: {} bytes ({:.1}x ratio)",
        zstd_size,
        embeddings.len() as f64 / zstd_size as f64
    );
    println!(
        "   ZSTD advantage: {:.1}% smaller",
        (1.0 - zstd_size as f64 / lz4_size as f64) * 100.0
    );

    println!();
    Ok(())
}

#[cfg(feature = "compression")]
async fn demo_kv_cache_simulation() -> trueno_db::Result<()> {
    println!("4. LLM KV Cache Simulation");
    println!("   -----------------------");

    let store = CompressedKvStore::new(MemoryKvStore::new(), Compression::Lz4);

    // Simulate storing attention KV cache entries
    // Each entry represents key/value tensors for a layer
    let mut total_original = 0usize;
    let mut total_compressed = 0usize;

    for layer in 0..12 {
        // Simulate KV cache: batch_size=1, seq_len=512, head_dim=64, num_heads=8
        // Size per layer: 512 * 64 * 8 * 2 (K and V) * 4 bytes = 2MB
        let cache_data: Vec<u8> =
            (0..2 * 1024 * 1024).map(|i| ((i + layer * 1000) % 256) as u8).collect();

        let key = format!("layer:{layer}:kv_cache");
        store.set(&key, cache_data.clone()).await?;

        let compressed_size = store.inner().get(&key).await?.unwrap().len();
        total_original += cache_data.len();
        total_compressed += compressed_size;
    }

    println!("   Layers: 12");
    println!("   Total original: {:.1} MB", total_original as f64 / 1024.0 / 1024.0);
    println!("   Total compressed: {:.1} MB", total_compressed as f64 / 1024.0 / 1024.0);
    println!(
        "   Memory saved: {:.1} MB ({:.1}x compression)",
        (total_original - total_compressed) as f64 / 1024.0 / 1024.0,
        total_original as f64 / total_compressed as f64
    );

    println!();
    Ok(())
}

#[cfg(not(feature = "compression"))]
fn main() {
    eprintln!("This example requires the 'compression' feature.");
    eprintln!("Run with: cargo run --example compressed_kv --features compression");
    std::process::exit(1);
}
