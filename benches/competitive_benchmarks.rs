//! Competitive Benchmarks vs `DuckDB`, `SQLite` (CORE-009)
//!
//! Toyota Way: Kaizen (prove all optimizations with data)
//!
//! This benchmark compares Trueno-DB SIMD performance against:
//! - `DuckDB` CPU (industry-leading analytics engine)
//! - `SQLite` (ubiquitous embedded database)
//!
//! Target: Prove 2-10x speedup for aggregations on 1M+ row datasets
//!
//! Note: GPU comparisons deferred to full integration (need end-to-end query execution)
//! Phase 1 focuses on SIMD backend validation with trueno v0.4.0
//!
//! References:
//! - `DuckDB` (2019): "Push-based execution model for analytics"
//! - CORE-005: SIMD fallback via Trueno
//! - CORE-006: Backend equivalence tests
//!
//! Run with: cargo bench --bench `competitive_benchmarks`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use duckdb::Connection as DuckDB;
use rusqlite::Connection as SQLite;
use trueno::Vector;

// Dataset sizes for competitive analysis
const MEDIUM: usize = 1_000_000; // 1M rows (typical analytics workload)

/// Set up a SQLite connection with integer data for SUM benchmarks
fn setup_sqlite_int() -> Option<SQLite> {
    let conn = SQLite::open_in_memory().ok()?;
    conn.execute_batch("CREATE TABLE data(value INTEGER); BEGIN;").ok()?;
    {
        let mut stmt = conn.prepare("INSERT INTO data VALUES (?)").ok()?;
        for i in 0..MEDIUM {
            stmt.execute([i as i32]).ok()?;
        }
    }
    conn.execute_batch("COMMIT;").ok()?;
    Some(conn)
}

/// Set up a SQLite connection with float data for AVG benchmarks
fn setup_sqlite_float() -> Option<SQLite> {
    let conn = SQLite::open_in_memory().ok()?;
    conn.execute_batch("CREATE TABLE data(value REAL); BEGIN;").ok()?;
    {
        let mut stmt = conn.prepare("INSERT INTO data VALUES (?)").ok()?;
        for i in 0..MEDIUM {
            stmt.execute([i as f32]).ok()?;
        }
    }
    conn.execute_batch("COMMIT;").ok()?;
    Some(conn)
}

/// Set up a DuckDB connection with integer data
fn setup_duckdb_int() -> Option<DuckDB> {
    let conn = DuckDB::open_in_memory().ok()?;
    conn.execute_batch(
        "CREATE TABLE data(value INTEGER);
         INSERT INTO data SELECT * FROM range(0, 1000000);",
    )
    .ok()?;
    Some(conn)
}

/// Set up a DuckDB connection with float data
fn setup_duckdb_float() -> Option<DuckDB> {
    let conn = DuckDB::open_in_memory().ok()?;
    conn.execute_batch(
        "CREATE TABLE data(value DOUBLE);
         INSERT INTO data SELECT CAST(range AS DOUBLE) FROM range(0, 1000000);",
    )
    .ok()?;
    Some(conn)
}

/// Benchmark SUM aggregation across engines
fn bench_sum_competitive(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum_competitive");

    // Medium dataset (1M rows) - primary comparison
    let medium_data: Vec<f32> = (0..MEDIUM).map(|i| i as f32).collect();

    // Trueno-DB SIMD (AVX-512/AVX2/SSE2 auto-detect)
    group.bench_with_input(BenchmarkId::new("trueno_simd", "1M_rows"), &medium_data, |b, data| {
        b.iter(|| {
            let vec = Vector::from_slice(black_box(data));
            vec.sum()
        });
    });

    // Scalar baseline (pure Rust iterator, f32 for fair comparison)
    group.bench_with_input(BenchmarkId::new("rust_scalar", "1M_rows"), &medium_data, |b, data| {
        b.iter(|| {
            let sum: f32 = black_box(data).iter().sum();
            black_box(sum)
        });
    });

    // DuckDB comparison
    if let Some(conn) = setup_duckdb_int() {
        group.bench_function(BenchmarkId::new("duckdb", "1M_rows"), |b| {
            b.iter(|| {
                let result: i64 = conn
                    .query_row("SELECT SUM(value) FROM data", [], |row| row.get(0))
                    .expect("DuckDB SUM query failed");
                black_box(result)
            });
        });
    } else {
        eprintln!("Skipping DuckDB SUM benchmark (setup failed)");
    }

    // SQLite comparison
    if let Some(conn) = setup_sqlite_int() {
        group.bench_function(BenchmarkId::new("sqlite", "1M_rows"), |b| {
            b.iter(|| {
                let result: i64 = conn
                    .query_row("SELECT SUM(value) FROM data", [], |row| row.get(0))
                    .expect("SQLite SUM query failed");
                black_box(result)
            });
        });
    } else {
        eprintln!("Skipping SQLite SUM benchmark (setup failed)");
    }

    group.finish();
}

/// Benchmark AVG aggregation
fn bench_avg_competitive(c: &mut Criterion) {
    let mut group = c.benchmark_group("avg_competitive");

    let medium_data: Vec<f32> = (0..MEDIUM).map(|i| i as f32).collect();

    // Trueno-DB SIMD
    group.bench_with_input(BenchmarkId::new("trueno_simd", "1M_rows"), &medium_data, |b, data| {
        b.iter(|| {
            let vec = Vector::from_slice(black_box(data));
            vec.mean()
        });
    });

    // Scalar baseline
    group.bench_with_input(BenchmarkId::new("rust_scalar", "1M_rows"), &medium_data, |b, data| {
        b.iter(|| {
            let sum: f32 = black_box(data).iter().sum();
            sum / data.len() as f32
        });
    });

    // DuckDB comparison
    if let Some(conn) = setup_duckdb_float() {
        group.bench_function(BenchmarkId::new("duckdb", "1M_rows"), |b| {
            b.iter(|| {
                let result: f64 = conn
                    .query_row("SELECT AVG(value) FROM data", [], |row| row.get(0))
                    .expect("DuckDB AVG query failed");
                black_box(result)
            });
        });
    } else {
        eprintln!("Skipping DuckDB AVG benchmark (setup failed)");
    }

    // SQLite comparison
    if let Some(conn) = setup_sqlite_float() {
        group.bench_function(BenchmarkId::new("sqlite", "1M_rows"), |b| {
            b.iter(|| {
                let result: f64 = conn
                    .query_row("SELECT AVG(value) FROM data", [], |row| row.get(0))
                    .expect("SQLite AVG query failed");
                black_box(result)
            });
        });
    } else {
        eprintln!("Skipping SQLite AVG benchmark (setup failed)");
    }

    group.finish();
}

criterion_group!(benches, bench_sum_competitive, bench_avg_competitive);
criterion_main!(benches);
