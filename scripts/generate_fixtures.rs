//! Fixture Generation Script for QA Specification
//!
//! Generates deterministic test data for the 100-cargo-run-examples-spec.md
//! Run: `cargo run --bin generate_fixtures`
//!
//! Toyota Way: Heijunka (Leveling) - Ensures all prerequisites exist before QA
//! run.

#![allow(
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

use std::{
    fs::{self, File},
    io::Write,
    sync::Arc,
};

use arrow::{
    array::{Float64Array, Int32Array, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};

const FIXTURE_DIR: &str = "test_fixtures";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Alimentar QA Fixture Generator ===");
    println!("Toyota Way: Heijunka (Leveling)\n");

    // Create fixture directory
    fs::create_dir_all(FIXTURE_DIR)?;
    println!("Created directory: {FIXTURE_DIR}/");

    // Generate all fixtures
    generate_basic_parquet()?;
    generate_basic_csv()?;
    generate_basic_json()?;
    generate_empty_parquet()?;
    generate_corrupt_parquet()?;
    generate_messy_dataset()?;
    generate_large_parquet()?;
    generate_drift_datasets()?;

    println!("\n✓ All fixtures generated successfully");
    println!("Ready for: docs/specifications/100-cargo-run-examples-spec.md");
    Ok(())
}

/// Basic parquet file with standard schema (Items 3, 7, 8, 9)
fn generate_basic_parquet() -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{FIXTURE_DIR}/data.parquet");
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("value", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from((1..=1000).collect::<Vec<_>>())),
            Arc::new(StringArray::from(
                (1..=1000).map(|i| format!("item_{i}")).collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                (1..=1000).map(|i| i as f64 * 0.1).collect::<Vec<_>>(),
            )),
        ],
    )?;

    let file = File::create(&path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;

    println!("  ✓ {path} (1000 rows)");
    Ok(())
}

/// Basic CSV file (Item 1)
fn generate_basic_csv() -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{FIXTURE_DIR}/input.csv");
    let mut file = File::create(&path)?;

    writeln!(file, "id,name,value,region")?;
    for i in 1..=500 {
        let region = match i % 4 {
            0 => "US",
            1 => "EU",
            2 => "APAC",
            _ => "LATAM",
        };
        writeln!(file, "{i},item_{i},{:.2},{region}", i as f64 * 0.5)?;
    }

    println!("  ✓ {path} (500 rows)");
    Ok(())
}

/// Basic JSON file (Item 2)
fn generate_basic_json() -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{FIXTURE_DIR}/data.json");
    let mut file = File::create(&path)?;

    for i in 1..=100 {
        let json = format!(
            r#"{{"id":{},"name":"item_{}","value":{:.2},"active":{}}}"#,
            i,
            i,
            i as f64 * 0.3,
            i % 2 == 0
        );
        writeln!(file, "{json}")?;
    }

    println!("  ✓ {path} (100 rows, NDJSON)");
    Ok(())
}

/// Empty parquet file (Item 97)
fn generate_empty_parquet() -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{FIXTURE_DIR}/empty.parquet");
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(Vec::<i32>::new())),
            Arc::new(Float64Array::from(Vec::<f64>::new())),
        ],
    )?;

    let file = File::create(&path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;

    println!("  ✓ {path} (0 rows - empty dataset test)");
    Ok(())
}

/// Corrupt parquet file (Item 98 - Jidoka test)
fn generate_corrupt_parquet() -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{FIXTURE_DIR}/corrupt.parquet");
    let mut file = File::create(&path)?;
    // Write garbage bytes that look like parquet header but are invalid
    file.write_all(b"PAR1\x00\x00\x00\x00INVALID_CONTENT_FOR_JIDOKA_TEST")?;
    println!("  ✓ {path} (corrupt - Jidoka stop-to-fix test)");
    Ok(())
}

/// Messy dataset with nulls and duplicates (Items 46-55)
fn generate_messy_dataset() -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{FIXTURE_DIR}/messy.parquet");
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true), // nullable
        Field::new("score", DataType::Float64, true), // nullable
        Field::new("category", DataType::Utf8, false),
    ]));

    // Create data with nulls, duplicates, and outliers
    let ids: Vec<i32> = (1..=200).collect();
    let names: Vec<Option<&str>> = (1..=200)
        .map(|i| if i % 10 == 0 { None } else { Some("name") })
        .collect();
    let scores: Vec<Option<f64>> = (1..=200)
        .map(|i| {
            if i % 15 == 0 {
                None
            } else if i % 50 == 0 {
                Some(999.0) // outlier
            } else {
                Some(i as f64 * 0.5)
            }
        })
        .collect();
    let categories: Vec<&str> = (1..=200)
        .map(|i| match i % 3 {
            0 => "A",
            1 => "B",
            _ => "C",
        })
        .collect();

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(StringArray::from(names)),
            Arc::new(Float64Array::from(scores)),
            Arc::new(StringArray::from(categories)),
        ],
    )?;

    let file = File::create(&path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;

    println!("  ✓ {path} (200 rows - nulls, outliers for quality tests)");
    Ok(())
}

/// Large parquet file - sparse generation (Item 10)
/// Note: Creates 1M rows efficiently, not 1GB to avoid git bloat
fn generate_large_parquet() -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{FIXTURE_DIR}/large.parquet");
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Float64, false),
        Field::new("category", DataType::Utf8, false),
    ]));

    let file = File::create(&path)?;
    let props = WriterProperties::builder()
        .set_compression(parquet::basic::Compression::SNAPPY)
        .build();
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

    // Write in chunks of 100k rows (10 chunks = 1M rows)
    for chunk in 0..10 {
        let start = chunk * 100_000;
        let ids: Vec<i32> = (start..start + 100_000).collect();
        let values: Vec<f64> = ids.iter().map(|&i| i as f64 * 0.001).collect();
        let cats: Vec<&str> = ids
            .iter()
            .map(|i| match i % 5 {
                0 => "alpha",
                1 => "beta",
                2 => "gamma",
                3 => "delta",
                _ => "epsilon",
            })
            .collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(Float64Array::from(values)),
                Arc::new(StringArray::from(cats)),
            ],
        )?;
        writer.write(&batch)?;
    }

    writer.close()?;
    let size = fs::metadata(&path)?.len();
    println!(
        "  ✓ {path} (1M rows, {:.1}MB compressed - large dataset test)",
        size as f64 / 1_000_000.0
    );
    Ok(())
}

/// Drift detection datasets - baseline and current (Items 56-65)
fn generate_drift_datasets() -> Result<(), Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("age", DataType::Float64, false),
        Field::new("income", DataType::Float64, false),
        Field::new("category", DataType::Utf8, false),
    ]));

    // Baseline dataset - normal distribution centered
    let baseline_path = format!("{FIXTURE_DIR}/baseline.parquet");
    let baseline_ages: Vec<f64> = (0..1000).map(|i| 30.0 + (i % 40) as f64).collect();
    let baseline_incomes: Vec<f64> = (0..1000).map(|i| 50000.0 + (i % 50000) as f64).collect();
    let baseline_cats: Vec<&str> = (0..1000)
        .map(|i| match i % 3 {
            0 => "A",
            1 => "B",
            _ => "C",
        })
        .collect();

    let baseline_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Float64Array::from(baseline_ages)),
            Arc::new(Float64Array::from(baseline_incomes)),
            Arc::new(StringArray::from(baseline_cats)),
        ],
    )?;

    let file = File::create(&baseline_path)?;
    let mut writer = ArrowWriter::try_new(file, schema.clone(), None)?;
    writer.write(&baseline_batch)?;
    writer.close()?;

    // Current dataset - shifted distribution (drift!)
    let current_path = format!("{FIXTURE_DIR}/current.parquet");
    let current_ages: Vec<f64> = (0..1000).map(|i| 45.0 + (i % 30) as f64).collect(); // shifted up
    let current_incomes: Vec<f64> = (0..1000).map(|i| 70000.0 + (i % 60000) as f64).collect(); // shifted
    let current_cats: Vec<&str> = (0..1000)
        .map(|i| match i % 5 {
            // different distribution
            0 | 1 => "A",
            2 => "B",
            _ => "C",
        })
        .collect();

    let current_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Float64Array::from(current_ages)),
            Arc::new(Float64Array::from(current_incomes)),
            Arc::new(StringArray::from(current_cats)),
        ],
    )?;

    let file = File::create(&current_path)?;
    let mut writer = ArrowWriter::try_new(file, schema, None)?;
    writer.write(&current_batch)?;
    writer.close()?;

    println!("  ✓ {baseline_path} (1000 rows - drift baseline)");
    println!("  ✓ {current_path} (1000 rows - drifted distribution)");
    Ok(())
}
