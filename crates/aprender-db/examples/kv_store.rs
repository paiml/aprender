//! Key-Value Store Demo - Phase 6 PAIML Stack Integration
//!
//! Run with: `cargo run --example kv_store`
//!
//! This example demonstrates trueno-db's KV store module designed for
//! pforge state management integration.

use trueno_db::kv::{hash_key, hash_keys_batch, KvStore, MemoryKvStore};

#[tokio::main]
async fn main() -> trueno_db::Result<()> {
    println!("=== Trueno-DB KV Store Demo ===\n");

    // Basic CRUD operations
    demo_basic_crud().await?;

    // Batch operations
    demo_batch_operations().await?;

    // Hash functions
    demo_hash_functions();

    // Concurrent access
    demo_concurrent_access().await?;

    println!("All demos completed successfully!");
    Ok(())
}

async fn demo_basic_crud() -> trueno_db::Result<()> {
    println!("1. Basic CRUD Operations");
    println!("   ---------------------");

    let store = MemoryKvStore::new();

    // Create
    store.set("user:1001", b"Alice".to_vec()).await?;
    store.set("user:1002", b"Bob".to_vec()).await?;
    println!("   SET user:1001 = Alice");
    println!("   SET user:1002 = Bob");

    // Read
    let alice = store.get("user:1001").await?;
    println!("   GET user:1001 = {:?}", alice.map(|v| String::from_utf8_lossy(&v).to_string()));

    // Update
    store.set("user:1001", b"Alice Smith".to_vec()).await?;
    let updated = store.get("user:1001").await?;
    println!(
        "   UPDATE user:1001 = {:?}",
        updated.map(|v| String::from_utf8_lossy(&v).to_string())
    );

    // Exists
    println!("   EXISTS user:1001 = {}", store.exists("user:1001").await?);
    println!("   EXISTS user:9999 = {}", store.exists("user:9999").await?);

    // Delete
    store.delete("user:1002").await?;
    println!("   DELETE user:1002");
    println!("   EXISTS user:1002 = {}", store.exists("user:1002").await?);

    println!();
    Ok(())
}

async fn demo_batch_operations() -> trueno_db::Result<()> {
    println!("2. Batch Operations");
    println!("   -----------------");

    let store = MemoryKvStore::new();

    // Batch set
    let pairs = vec![
        ("config:timeout", b"30000".to_vec()),
        ("config:retries", b"3".to_vec()),
        ("config:debug", b"false".to_vec()),
    ];
    store.batch_set(pairs).await?;
    println!("   BATCH SET 3 config keys");

    // Batch get
    let keys = ["config:timeout", "config:retries", "config:missing"];
    let values = store.batch_get(&keys).await?;
    println!("   BATCH GET results:");
    for (key, value) in keys.iter().zip(values.iter()) {
        let display = value.as_ref().map(|v| String::from_utf8_lossy(v).to_string());
        println!("     {key} = {display:?}");
    }

    println!();
    Ok(())
}

fn demo_hash_functions() {
    println!("3. SIMD Hash Functions (via trueno)");
    println!("   --------------------------------");

    // Single key hash
    let key = "session:abc123";
    let hash = hash_key(key);
    println!("   hash_key({key:?}) = 0x{hash:016x}");

    // Batch hash for partitioning
    let keys = ["shard:0", "shard:1", "shard:2", "shard:3"];
    let hashes = hash_keys_batch(&keys);
    println!("   Batch hashes for sharding:");
    for (key, hash) in keys.iter().zip(hashes.iter()) {
        let partition = hash % 4;
        println!("     {key} -> partition {partition}");
    }

    println!();
}

async fn demo_concurrent_access() -> trueno_db::Result<()> {
    println!("4. Concurrent Access (Thread-Safe)");
    println!("   --------------------------------");

    use std::sync::Arc;

    let store = Arc::new(MemoryKvStore::new());
    let mut handles = vec![];

    // Spawn 10 concurrent writers
    for i in 0..10 {
        let store = Arc::clone(&store);
        handles.push(tokio::spawn(async move {
            let key = format!("counter:{i}");
            let value = format!("{}", i * 100).into_bytes();
            store.set(&key, value).await.unwrap();
        }));
    }

    // Wait for all writes
    for handle in handles {
        handle.await.unwrap();
    }

    // Verify all writes
    println!("   Concurrent writes completed:");
    for i in 0..10 {
        let key = format!("counter:{i}");
        let value = store.get(&key).await?;
        let display = value.map(|v| String::from_utf8_lossy(&v).to_string());
        println!("     {key} = {display:?}");
    }

    println!("\n   Store stats: {} entries", store.len());
    println!();
    Ok(())
}
