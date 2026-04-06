//! Content Addressing Example
//!
//! Demonstrates content-addressed storage:
//! - BLAKE3 hashing
//! - Deduplication
//! - Integrity verification
//!
//! Run with: cargo run --example content_addressing

use pacha::prelude::*;

fn main() {
    println!("=== Content Addressing Example ===\n");

    // 1. Basic content addressing
    println!("1. Computing content addresses...");
    let data1 = b"Hello, World!";
    let data2 = b"Hello, World!"; // Same content
    let data3 = b"Hello, World?"; // Different content

    let addr1 = ContentAddress::from_bytes(data1);
    let addr2 = ContentAddress::from_bytes(data2);
    let addr3 = ContentAddress::from_bytes(data3);

    println!("   Data 1: {:?}", String::from_utf8_lossy(data1));
    println!("   Hash:   {}", addr1.hash_hex());
    println!("   Size:   {} bytes", addr1.size());
    println!();
    println!("   Data 2: {:?}", String::from_utf8_lossy(data2));
    println!("   Hash:   {}", addr2.hash_hex());
    println!();
    println!("   Data 3: {:?}", String::from_utf8_lossy(data3));
    println!("   Hash:   {}", addr3.hash_hex());

    // 2. Demonstrate deduplication
    println!("\n2. Deduplication check:");
    println!("   addr1 == addr2: {} (same content)", addr1 == addr2);
    println!("   addr1 == addr3: {} (different content)", addr1 == addr3);

    // 3. Storage path sharding
    println!("\n3. Storage path sharding:");
    println!("   Data 1 prefix: {}", addr1.storage_prefix());
    println!("   Data 1 path:   {}", addr1.storage_path());
    println!("   Data 3 prefix: {}", addr3.storage_prefix());
    println!("   Data 3 path:   {}", addr3.storage_path());

    // 4. Integrity verification
    println!("\n4. Integrity verification:");
    println!("   Verifying data1 against addr1: {}", addr1.verify(data1));
    println!("   Verifying data3 against addr1: {}", addr1.verify(data3));
    println!("   Verifying tampered data: {}", addr1.verify(b"Tampered!"));

    // 5. Display format
    println!("\n5. Content address display format:");
    println!("   Format: blake3:<hash>:<size>:<compression>");
    println!("   Example: {addr1}");

    // 6. Large file hashing
    println!("\n6. Hashing larger data:");
    let large_data: Vec<u8> = (0..1_000_000).map(|i| (i % 256) as u8).collect();
    let large_addr = ContentAddress::from_bytes(&large_data);
    println!("   Data size: {} bytes (1 MB)", large_data.len());
    println!("   Hash:      {}", large_addr.hash_hex());
    println!("   Prefix:    {}", large_addr.storage_prefix());

    // 7. Streaming hash (from reader)
    println!("\n7. Streaming hash computation:");
    let cursor = std::io::Cursor::new(b"Streamed content");
    let stream_addr = ContentAddress::from_reader(cursor).expect("Failed to hash");
    println!("   Streamed hash: {}", stream_addr.hash_hex());

    // 8. Compression support
    println!("\n8. Compression support:");
    println!("   Default compression: {}", addr1.compression());
    #[cfg(feature = "compression")]
    {
        let compressed = addr1.clone().with_compression(Compression::Zstd);
        println!("   With Zstd: {}", compressed.compression());
    }

    println!("\n✅ Content addressing example complete!");
}
