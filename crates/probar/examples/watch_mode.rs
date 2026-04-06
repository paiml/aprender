//! Example: Watch Mode (Feature 6)
//!
//! Demonstrates: File watching for hot reload during development
//!
//! Run with: `cargo run --example watch_mode`
//!
//! Toyota Way: Genchi Genbutsu (Go and See) - Real-time feedback on changes

use jugar_probar::prelude::*;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> ProbarResult<()> {
    println!("=== Watch Mode Example ===\n");

    // 1. Create watch config
    println!("1. Creating watch configuration...");
    let config = WatchConfig::new()
        .with_pattern("**/*.rs")
        .with_pattern("**/*.toml")
        .with_ignore("target/**");

    println!("   Patterns: {:?}", config.patterns);
    println!("   Ignore patterns: {:?}", config.ignore_patterns);
    println!("   Debounce: {}ms", config.debounce_ms);

    // 2. Build config with builder pattern
    println!("\n2. Using WatchBuilder...");
    let builder_config = WatchBuilder::new()
        .rust_files()
        .toml_files()
        .ignore_target()
        .build();

    println!("   Patterns: {:?}", builder_config.patterns);

    // 3. FileChange kinds
    println!("\n3. FileChange kinds...");
    let kinds = [
        FileChangeKind::Created,
        FileChangeKind::Modified,
        FileChangeKind::Deleted,
        FileChangeKind::Renamed,
    ];

    for kind in &kinds {
        println!("   {:?}", kind);
    }

    // 4. Create file changes
    println!("\n4. Creating file changes...");
    let changes = vec![
        FileChange {
            path: PathBuf::from("src/lib.rs"),
            kind: FileChangeKind::Modified,
            timestamp: Instant::now(),
        },
        FileChange {
            path: PathBuf::from("tests/test.rs"),
            kind: FileChangeKind::Created,
            timestamp: Instant::now(),
        },
    ];

    for change in &changes {
        println!("   {:?}: {}", change.kind, change.path.display());
    }

    // 5. Watch statistics
    println!("\n5. Watch statistics...");
    let mut stats = WatchStats::new();
    stats.record_trigger(3);
    stats.record_trigger(2);

    println!("   Total triggers: {}", stats.trigger_count);
    println!("   Total changes: {}", stats.change_count);

    // 6. Pattern matching
    println!("\n6. Pattern matching...");
    let test_paths = [
        "src/lib.rs",
        "src/main.rs",
        "target/debug/test",
        "Cargo.toml",
    ];

    for path in &test_paths {
        let matches = config.matches_pattern(&PathBuf::from(path));
        println!("   {} -> matches: {}", path, matches);
    }

    println!("\n   Note: In real usage, start the watcher with:");
    println!("   let mut watcher = FileWatcher::new(config)?;");
    println!("   watcher.start()?;");

    println!("\n✅ Watch mode example completed!");
    Ok(())
}
