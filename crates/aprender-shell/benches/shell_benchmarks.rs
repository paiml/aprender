#![allow(clippy::unwrap_used)] // Benchmarks can use unwrap() for simplicity
//! aprender-shell Performance Benchmarks
//!
//! Modeled after bashrs benchmark patterns for consistency.
//!
//! Performance targets:
//! - Small (~50 commands): <5ms train, <1ms suggest
//! - Medium (~500 commands): <50ms train, <5ms suggest
//! - Large (~5000 commands): <500ms train, <10ms suggest
//!
//! Key metrics:
//! - Training throughput (commands/second)
//! - Suggestion latency (p50, p99)
//! - Memory efficiency

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

// Benchmark fixtures (same pattern as bashrs)
const SMALL_HISTORY: &str = include_str!("fixtures/small_history.txt");
const MEDIUM_HISTORY: &str = include_str!("fixtures/medium_history.txt");
const LARGE_HISTORY: &str = include_str!("fixtures/large_history.txt");

/// Parse history file into commands (filtering comments and empty lines)
fn parse_commands(history: &str) -> Vec<String> {
    history
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.trim().starts_with('#'))
        .map(String::from)
        .collect()
}

/// Benchmark history parsing
fn benchmark_parse_history(c: &mut Criterion) {
    let mut group = c.benchmark_group("parse_history");

    let small_lines = SMALL_HISTORY.lines().count();
    let medium_lines = MEDIUM_HISTORY.lines().count();
    let large_lines = LARGE_HISTORY.lines().count();

    group.throughput(Throughput::Elements(small_lines as u64));
    group.bench_with_input(
        BenchmarkId::new("small", format!("{} lines", small_lines)),
        &SMALL_HISTORY,
        |b, history| b.iter(|| parse_commands(black_box(history))),
    );

    group.throughput(Throughput::Elements(medium_lines as u64));
    group.bench_with_input(
        BenchmarkId::new("medium", format!("{} lines", medium_lines)),
        &MEDIUM_HISTORY,
        |b, history| b.iter(|| parse_commands(black_box(history))),
    );

    group.throughput(Throughput::Elements(large_lines as u64));
    group.bench_with_input(
        BenchmarkId::new("large", format!("{} lines", large_lines)),
        &LARGE_HISTORY,
        |b, history| b.iter(|| parse_commands(black_box(history))),
    );

    group.finish();
}

/// Benchmark n-gram model training
fn benchmark_train(c: &mut Criterion) {
    use aprender_shell::MarkovModel;

    let mut group = c.benchmark_group("train_model");

    let small_cmds = parse_commands(SMALL_HISTORY);
    let medium_cmds = parse_commands(MEDIUM_HISTORY);
    let large_cmds = parse_commands(LARGE_HISTORY);

    // Performance target: <5ms for small
    group.throughput(Throughput::Elements(small_cmds.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("small", format!("{} cmds", small_cmds.len())),
        &small_cmds,
        |b, cmds| {
            b.iter(|| {
                let mut model = MarkovModel::new(3);
                model.train(black_box(cmds));
                black_box(model)
            })
        },
    );

    // Performance target: <50ms for medium
    group.throughput(Throughput::Elements(medium_cmds.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("medium", format!("{} cmds", medium_cmds.len())),
        &medium_cmds,
        |b, cmds| {
            b.iter(|| {
                let mut model = MarkovModel::new(3);
                model.train(black_box(cmds));
                black_box(model)
            })
        },
    );

    // Performance target: <500ms for large
    group.throughput(Throughput::Elements(large_cmds.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("large", format!("{} cmds", large_cmds.len())),
        &large_cmds,
        |b, cmds| {
            b.iter(|| {
                let mut model = MarkovModel::new(3);
                model.train(black_box(cmds));
                black_box(model)
            })
        },
    );

    group.finish();
}

/// Benchmark suggestion latency (critical for UX)
fn benchmark_suggest(c: &mut Criterion) {
    use aprender_shell::MarkovModel;

    let mut group = c.benchmark_group("suggest_latency");

    // Pre-train models for suggestion benchmarks
    let small_cmds = parse_commands(SMALL_HISTORY);
    let medium_cmds = parse_commands(MEDIUM_HISTORY);
    let large_cmds = parse_commands(LARGE_HISTORY);

    let mut small_model = MarkovModel::new(3);
    small_model.train(&small_cmds);

    let mut medium_model = MarkovModel::new(3);
    medium_model.train(&medium_cmds);

    let mut large_model = MarkovModel::new(3);
    large_model.train(&large_cmds);

    // Common prefixes to benchmark
    let prefixes = ["git ", "cargo ", "docker ", "kubectl ", "npm "];

    // Small model suggestions - target <1ms
    for prefix in &prefixes {
        group.bench_with_input(
            BenchmarkId::new("small", format!("prefix={}", prefix.trim())),
            &(&small_model, *prefix),
            |b, (model, prefix)| {
                b.iter(|| {
                    let suggestions = model.suggest(black_box(prefix), 5);
                    black_box(suggestions)
                })
            },
        );
    }

    // Medium model suggestions - target <5ms
    for prefix in &prefixes {
        group.bench_with_input(
            BenchmarkId::new("medium", format!("prefix={}", prefix.trim())),
            &(&medium_model, *prefix),
            |b, (model, prefix)| {
                b.iter(|| {
                    let suggestions = model.suggest(black_box(prefix), 5);
                    black_box(suggestions)
                })
            },
        );
    }

    // Large model suggestions - target <10ms
    for prefix in &prefixes {
        group.bench_with_input(
            BenchmarkId::new("large", format!("prefix={}", prefix.trim())),
            &(&large_model, *prefix),
            |b, (model, prefix)| {
                b.iter(|| {
                    let suggestions = model.suggest(black_box(prefix), 5);
                    black_box(suggestions)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark partial token completion (key UX feature)
fn benchmark_partial_completion(c: &mut Criterion) {
    use aprender_shell::MarkovModel;

    let mut group = c.benchmark_group("partial_completion");

    let medium_cmds = parse_commands(MEDIUM_HISTORY);
    let mut model = MarkovModel::new(3);
    model.train(&medium_cmds);

    // Partial token scenarios
    let partial_prefixes = [
        "git co",    // commit/checkout
        "cargo b",   // build/bench
        "docker-c",  // compose
        "kubectl g", // get
        "npm r",     // run
    ];

    for prefix in &partial_prefixes {
        group.bench_with_input(
            BenchmarkId::new("partial", *prefix),
            &(&model, *prefix),
            |b, (model, prefix)| {
                b.iter(|| {
                    let suggestions = model.suggest(black_box(prefix), 5);
                    black_box(suggestions)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark model serialization (save/load via tempfile)
fn benchmark_serialization(c: &mut Criterion) {
    use aprender_shell::MarkovModel;
    use tempfile::NamedTempFile;

    let mut group = c.benchmark_group("serialization");

    let medium_cmds = parse_commands(MEDIUM_HISTORY);
    let mut model = MarkovModel::new(3);
    model.train(&medium_cmds);

    // Benchmark save (using serde_json for in-memory)
    group.bench_function("serialize_json", |b| {
        b.iter(|| {
            let json = serde_json::to_vec(black_box(&model)).unwrap();
            black_box(json)
        })
    });

    // Benchmark load from JSON
    let json = serde_json::to_vec(&model).unwrap();
    group.bench_function("deserialize_json", |b| {
        b.iter(|| {
            let loaded: MarkovModel = serde_json::from_slice(black_box(&json)).unwrap();
            black_box(loaded)
        })
    });

    // Benchmark file save
    group.bench_function("save_file", |b| {
        b.iter(|| {
            let tmp = NamedTempFile::new().unwrap();
            model.save(tmp.path()).unwrap();
            black_box(tmp)
        })
    });

    // Benchmark file load
    let tmp = NamedTempFile::new().unwrap();
    model.save(tmp.path()).unwrap();
    let path = tmp.path().to_owned();
    group.bench_function("load_file", |b| {
        b.iter(|| {
            let loaded = MarkovModel::load(black_box(&path)).unwrap();
            black_box(loaded)
        })
    });

    group.finish();
}

/// Benchmark end-to-end workflow (parse + train + suggest)
fn benchmark_end_to_end(c: &mut Criterion) {
    use aprender_shell::MarkovModel;

    let mut group = c.benchmark_group("end_to_end");

    // Simulate real user workflow: load history, train, suggest
    group.bench_function("small_workflow", |b| {
        b.iter(|| {
            let cmds = parse_commands(black_box(SMALL_HISTORY));
            let mut model = MarkovModel::new(3);
            model.train(&cmds);
            let suggestions = model.suggest("git ", 5);
            black_box(suggestions)
        })
    });

    group.bench_function("medium_workflow", |b| {
        b.iter(|| {
            let cmds = parse_commands(black_box(MEDIUM_HISTORY));
            let mut model = MarkovModel::new(3);
            model.train(&cmds);
            let suggestions = model.suggest("cargo ", 5);
            black_box(suggestions)
        })
    });

    group.finish();
}

/// Benchmark synthetic data generation (CodeEDA)
fn benchmark_synthetic(c: &mut Criterion) {
    use aprender::synthetic::code_eda::{CodeEda, CodeEdaConfig, CodeLanguage};
    use aprender::synthetic::{SyntheticConfig, SyntheticGenerator};

    let mut group = c.benchmark_group("synthetic_generation");

    let commands = parse_commands(MEDIUM_HISTORY);

    let eda_config = CodeEdaConfig::default()
        .with_rename_prob(0.1)
        .with_comment_prob(0.05)
        .with_reorder_prob(0.1)
        .with_remove_prob(0.05)
        .with_language(CodeLanguage::Generic)
        .with_num_augments(2);

    let code_eda = CodeEda::new(eda_config);

    let synth_config = SyntheticConfig::default()
        .with_augmentation_ratio(0.5)
        .with_quality_threshold(0.7)
        .with_seed(42);

    group.bench_function("code_eda_augment", |b| {
        b.iter(|| {
            let result = code_eda
                .generate(black_box(&commands), black_box(&synth_config))
                .unwrap_or_default();
            black_box(result)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_parse_history,
    benchmark_train,
    benchmark_suggest,
    benchmark_partial_completion,
    benchmark_serialization,
    benchmark_end_to_end,
    benchmark_synthetic,
);
criterion_main!(benches);
