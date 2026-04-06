//! Example: Multi-Browser Context Management (Feature 14)
//!
//! Demonstrates: Managing isolated browser contexts for parallel testing
//!
//! Run with: `cargo run --example multi_context`
//!
//! Toyota Way: Heijunka (Level Loading) - Balanced resource allocation

use jugar_probar::prelude::*;

fn main() -> ProbarResult<()> {
    println!("=== Multi-Browser Context Management Example ===\n");

    // 1. Create context pool
    println!("1. Creating context pool...");
    let pool = ContextPool::new(4); // 4 parallel contexts

    println!("   Max contexts: 4");
    println!("   Available: {}", pool.available_count());

    // 2. Context configuration
    println!("\n2. Creating context configuration...");
    let config = ContextConfig::new("test_context")
        .with_user_agent("Probar/1.0")
        .with_locale("en-US")
        .with_timezone("America/New_York");

    println!("   Name: {}", config.name);
    println!("   Locale: {:?}", config.locale);
    println!("   Timezone: {:?}", config.timezone);

    // 3. Context states
    println!("\n3. Context states...");
    let states = [
        ContextState::Creating,
        ContextState::Ready,
        ContextState::InUse,
        ContextState::Cleaning,
        ContextState::Error,
        ContextState::Closed,
    ];

    for state in &states {
        println!("   {:?}", state);
    }

    // 4. Create browser context
    println!("\n4. Creating browser context...");
    let context = BrowserContext::new("context_1", config.clone());

    println!("   ID: {}", context.id);
    println!("   State: {:?}", context.state);

    // 5. Storage state
    println!("\n5. Managing storage state...");
    let storage = StorageState::new()
        .with_local_storage("https://example.com", "key1", "value1")
        .with_session_storage("https://example.com", "session_key", "session_value");

    println!("   Storage state created");
    println!("   Is empty: {}", storage.is_empty());

    // 6. Cookies
    println!("\n6. Creating cookies...");
    let mut cookie = Cookie::new("session_id", "abc123", "example.com").with_path("/");
    cookie.http_only = true;
    cookie.secure = true;
    cookie.same_site = SameSite::Strict;

    println!("   Name: {}", cookie.name);
    println!("   Value: {}", cookie.value);
    println!("   Domain: {}", cookie.domain);
    println!("   Same-site: {:?}", cookie.same_site);

    // 7. Same-site values
    println!("\n7. Same-site cookie options...");
    let same_sites = [SameSite::Strict, SameSite::Lax, SameSite::None];

    for ss in &same_sites {
        println!("   {:?}", ss);
    }

    // 8. Geolocation in context
    println!("\n8. Geolocation configuration...");
    let config_with_geo = ContextConfig::new("geo_context").with_geolocation(40.7128, -74.0060); // New York

    if let Some(geo) = &config_with_geo.geolocation {
        println!("   Latitude: {}", geo.latitude);
        println!("   Longitude: {}", geo.longitude);
    }

    // 9. Create and manage contexts in pool
    println!("\n9. Managing contexts in pool...");
    let ctx_id = pool.create(Some(config))?;
    println!("   Created context: {}", ctx_id);

    let acquired = pool.acquire()?;
    println!("   Acquired context: {}", acquired);

    println!("   Total: {}", pool.count());
    println!("   Available: {}", pool.available_count());
    println!("   In use: {}", pool.in_use_count());

    // Release and cleanup
    pool.release(&acquired)?;
    println!("   Released context: {}", acquired);

    // 10. Pool statistics
    println!("\n10. Pool statistics...");
    let stats = ContextPoolStats {
        total: pool.count(),
        available: pool.available_count(),
        in_use: pool.in_use_count(),
        active_tests: 0,
    };

    println!("   Total: {}", stats.total);
    println!(
        "   Available: {} ({:.0}%)",
        stats.available,
        if stats.total > 0 {
            stats.available as f64 / stats.total as f64 * 100.0
        } else {
            0.0
        }
    );
    println!(
        "   In use: {} ({:.0}%)",
        stats.in_use,
        if stats.total > 0 {
            stats.in_use as f64 / stats.total as f64 * 100.0
        } else {
            0.0
        }
    );

    println!("\n✅ Multi-browser context example completed!");
    Ok(())
}
