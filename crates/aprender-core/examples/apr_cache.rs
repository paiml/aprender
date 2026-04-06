#![allow(clippy::disallowed_methods)]
//! APR Model Cache Example
//!
//! Demonstrates the hierarchical caching system implementing Toyota Way Just-In-Time:
//! - **Right amount**: Cache only what's needed for current inference
//! - **Right time**: Prefetch before access, evict after use
//! - **Right place**: L1 = hot, L2 = warm, L3 = cold storage
//!
//! Run with: `cargo run --example apr_cache`

use aprender::cache::{
    AccessStats, CacheConfig, CacheData, CacheEntry, CacheMetadata, CacheTier, EvictionPolicy,
    MemoryBudget, ModelRegistry, ModelType,
};
use std::time::Duration;

fn main() {
    println!("=== APR Model Cache Demo ===\n");

    // Part 1: Eviction Policies
    eviction_policies_demo();

    // Part 2: Memory Budget
    memory_budget_demo();

    // Part 3: Access Statistics
    access_stats_demo();

    // Part 4: Cache Configuration
    cache_config_demo();

    // Part 5: Model Registry
    model_registry_demo();

    // Part 6: Cache Tiers
    cache_tiers_demo();

    println!("\n=== Cache Demo Complete! ===");
}

fn eviction_policies_demo() {
    println!("--- Part 1: Eviction Policies ---\n");

    let policies = [
        EvictionPolicy::LRU,
        EvictionPolicy::LFU,
        EvictionPolicy::ARC,
        EvictionPolicy::Clock,
        EvictionPolicy::Fixed,
    ];

    println!("{:<10} {:<45} {:>10}", "Policy", "Description", "Eviction");
    println!("{}", "-".repeat(70));

    for policy in &policies {
        println!(
            "{:<10} {:<45} {:>10}",
            format!("{:?}", policy),
            policy.description(),
            if policy.supports_eviction() {
                "Yes"
            } else {
                "No (Fixed)"
            }
        );
    }

    println!("\nRecommended Use Cases:");
    for policy in &policies {
        println!("  {:?}: {}", policy, policy.recommended_use_case());
    }
    println!();
}

fn memory_budget_demo() {
    println!("--- Part 2: Memory Budget ---\n");

    // Default watermarks (90% high, 70% low)
    let budget = MemoryBudget::new(100);
    println!("Memory Budget (100 pages, default watermarks):");
    println!("  Max Pages: {}", budget.max_pages);
    println!(
        "  High Watermark: {} (start eviction)",
        budget.high_watermark
    );
    println!("  Low Watermark: {} (stop eviction)", budget.low_watermark);

    // Eviction decisions
    println!("\nEviction Decisions:");
    for pages in [50, 80, 90, 95, 100] {
        let needs = budget.needs_eviction(pages);
        let can_stop = budget.can_stop_eviction(pages);
        println!(
            "  {} pages: needs_eviction={}, can_stop={}",
            pages, needs, can_stop
        );
    }

    // Custom watermarks
    let custom = MemoryBudget::with_watermarks(1000, 0.95, 0.80);
    println!("\nCustom Budget (1000 pages, 95%/80% watermarks):");
    println!(
        "  High: {}, Low: {}",
        custom.high_watermark, custom.low_watermark
    );

    // Reserved pages
    let mut budget_reserved = MemoryBudget::new(100);
    budget_reserved.reserve_page(1);
    budget_reserved.reserve_page(2);
    budget_reserved.reserve_page(3);

    println!("\nReserved Pages:");
    println!("  Page 1 can_evict: {}", budget_reserved.can_evict(1)); // false
    println!("  Page 4 can_evict: {}", budget_reserved.can_evict(4)); // true

    budget_reserved.release_page(1);
    println!(
        "  After release(1): can_evict(1) = {}",
        budget_reserved.can_evict(1)
    );
    println!();
}

fn access_stats_demo() {
    println!("--- Part 3: Access Statistics ---\n");

    let mut stats = AccessStats::new();

    // Simulate cache accesses
    for i in 0u64..80 {
        stats.record_hit(100 + (i % 50), i); // Varying access times
    }
    for i in 80u64..100 {
        stats.record_miss(i);
    }

    // Some prefetch hits
    for _ in 0..30 {
        stats.record_prefetch_hit();
    }

    println!("Cache Access Statistics:");
    println!("  Hit Count: {}", stats.hit_count);
    println!("  Miss Count: {}", stats.miss_count);
    println!("  Hit Rate: {:.1}%", stats.hit_rate() * 100.0);
    println!("  Avg Access Time: {:.1} ns", stats.avg_access_time_ns());
    println!("  Prefetch Hits: {}", stats.prefetch_hits);
    println!(
        "  Prefetch Effectiveness: {:.1}%",
        stats.prefetch_effectiveness() * 100.0
    );
    println!();
}

fn cache_config_demo() {
    println!("--- Part 4: Cache Configuration ---\n");

    // Default configuration
    let default = CacheConfig::default();
    println!("Default Configuration:");
    println!("  L1 Max: {} MB", default.l1_max_bytes / (1024 * 1024));
    println!("  L2 Max: {} MB", default.l2_max_bytes / (1024 * 1024));
    println!("  Eviction: {:?}", default.eviction_policy);
    println!("  Prefetch: {}", default.prefetch_enabled);

    // Embedded configuration
    let embedded = CacheConfig::embedded(1024 * 1024); // 1MB
    println!("\nEmbedded Configuration (1MB):");
    println!("  L1 Max: {} MB", embedded.l1_max_bytes / (1024 * 1024));
    println!("  L2 Max: {} (disabled)", embedded.l2_max_bytes);
    println!("  Eviction: {:?} (no eviction)", embedded.eviction_policy);
    println!("  Prefetch: {}", embedded.prefetch_enabled);

    // Custom configuration with builder
    let custom = CacheConfig::new()
        .with_l1_size(128 * 1024 * 1024)
        .with_l2_size(2 * 1024 * 1024 * 1024)
        .with_eviction_policy(EvictionPolicy::ARC)
        .with_ttl(Duration::from_secs(3600))
        .with_prefetch(true);

    println!("\nCustom Configuration:");
    println!("  L1 Max: {} MB", custom.l1_max_bytes / (1024 * 1024));
    println!(
        "  L2 Max: {} GB",
        custom.l2_max_bytes / (1024 * 1024 * 1024)
    );
    println!("  Eviction: {:?}", custom.eviction_policy);
    println!("  TTL: {:?}", custom.default_ttl);
    println!();
}

fn model_registry_demo() {
    println!("--- Part 5: Model Registry ---\n");

    let config = CacheConfig::new()
        .with_l1_size(10 * 1024)  // 10KB for demo
        .with_eviction_policy(EvictionPolicy::LRU);

    let mut registry = ModelRegistry::new(config);

    // Create and insert some models
    println!("Inserting models into cache:");
    for i in 0..5 {
        let data = vec![0u8; 2048]; // 2KB each
        let entry = CacheEntry::new(
            [i as u8; 32],
            ModelType::new(1),
            CacheData::Decompressed(data),
        );
        registry.insert_l1(format!("model_{}", i), entry);
        println!("  Inserted model_{} (2KB)", i);
    }

    // Check cache state
    println!("\nCache State:");
    for i in 0..5 {
        let name = format!("model_{}", i);
        let tier = registry.get_tier(&name);
        println!("  {}: {:?}", name, tier);
    }

    // Access some models to update LRU
    println!("\nAccessing models:");
    let _ = registry.get("model_0");
    let _ = registry.get("model_2");
    println!("  Accessed model_0 and model_2");

    // Get statistics
    let stats = registry.stats();
    println!("\nCache Statistics:");
    println!("  L1 Entries: {}", stats.l1_entries);
    println!("  L1 Bytes: {} KB", stats.l1_bytes / 1024);
    println!("  L1 Hits: {}", stats.l1_hits);
    println!("  Total Hit Rate: {:.1}%", stats.hit_rate() * 100.0);
    println!("  Uptime: {:?}", stats.uptime);

    // List models
    let models = registry.list();
    println!("\nCached Models ({}):", models.len());
    for model in &models {
        println!(
            "  {} - {} bytes, tier: {:?}",
            model.name, model.size_bytes, model.cache_tier
        );
    }
    println!();
}

fn cache_tiers_demo() {
    println!("--- Part 6: Cache Tiers ---\n");

    let tiers = [CacheTier::L1Hot, CacheTier::L2Warm, CacheTier::L3Cold];

    println!("{:<15} {:<20} {:>15}", "Tier", "Name", "Typical Latency");
    println!("{}", "-".repeat(55));

    for tier in &tiers {
        println!(
            "{:<15} {:<20} {:>15?}",
            format!("{:?}", tier),
            tier.name(),
            tier.typical_latency()
        );
    }

    // Cache metadata demo
    println!("\nCache Metadata:");

    let meta = CacheMetadata::new(1024 * 1024) // 1MB
        .with_compression_ratio(2.5)
        .with_ttl(Duration::from_secs(3600));

    println!("  Size: {} bytes", meta.size_bytes);
    println!("  Compression Ratio: {:.1}x", meta.compression_ratio);
    println!("  TTL: {:?}", meta.ttl);
    println!("  Age: {:?}", meta.age());
    println!("  Expired: {}", meta.is_expired());

    // Cache data variants
    println!("\nCache Data Variants:");

    let compressed = CacheData::Compressed(vec![0u8; 500]);
    println!(
        "  Compressed: {} bytes, is_compressed={}",
        compressed.size(),
        compressed.is_compressed()
    );

    let decompressed = CacheData::Decompressed(vec![0u8; 1000]);
    println!(
        "  Decompressed: {} bytes, is_compressed={}",
        decompressed.size(),
        decompressed.is_compressed()
    );

    let mapped = CacheData::Mapped {
        path: "/tmp/model.cache".into(),
        offset: 0,
        length: 2000,
    };
    println!(
        "  Mapped: {} bytes, is_mapped={}",
        mapped.size(),
        mapped.is_mapped()
    );
    println!();
}
