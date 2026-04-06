#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::uninlined_format_args,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::float_cmp,
    clippy::needless_late_init,
    clippy::redundant_clone,
    clippy::doc_markdown,
    clippy::unnecessary_debug_formatting,
    clippy::for_kv_map
)]
//! Drift Detection Example
//!
//! Demonstrates statistical drift detection between datasets:
//! - Column-level drift analysis
//! - Multiple statistical tests (KS, Chi-square, PSI)
//! - Drift severity classification
//! - Drift reports and recommendations
//!
//! Run with: cargo run --example drift_detection

use std::sync::Arc;

use alimentar::{ArrowDataset, Dataset, DriftDetector, DriftSeverity, DriftTest};
use arrow::{
    array::{Float64Array, Int32Array, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};

fn create_baseline_dataset() -> alimentar::Result<ArrowDataset> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("age", DataType::Float64, false),
        Field::new("income", DataType::Float64, false),
        Field::new("category", DataType::Utf8, false),
        Field::new("score", DataType::Int32, false),
    ]));

    // Baseline: Normal distribution for age (~35), income (~50000)
    let ages: Vec<f64> = vec![32.0, 35.0, 28.0, 42.0, 38.0, 31.0, 45.0, 33.0, 37.0, 29.0];
    let incomes: Vec<f64> = vec![
        48000.0, 52000.0, 45000.0, 55000.0, 51000.0, 47000.0, 58000.0, 49000.0, 53000.0, 46000.0,
    ];
    let categories: Vec<&str> = vec!["A", "B", "A", "C", "B", "A", "B", "A", "C", "A"];
    let scores: Vec<i32> = vec![75, 82, 68, 91, 78, 72, 88, 76, 85, 70];

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(ages)),
            Arc::new(Float64Array::from(incomes)),
            Arc::new(StringArray::from(categories)),
            Arc::new(Int32Array::from(scores)),
        ],
    )?;

    ArrowDataset::from_batch(batch)
}

fn create_drifted_dataset() -> alimentar::Result<ArrowDataset> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("age", DataType::Float64, false),
        Field::new("income", DataType::Float64, false),
        Field::new("category", DataType::Utf8, false),
        Field::new("score", DataType::Int32, false),
    ]));

    // Drifted: Age shifted up (~45), income shifted significantly (~70000)
    // Category distribution changed (more C's)
    let ages: Vec<f64> = vec![45.0, 48.0, 42.0, 55.0, 51.0, 44.0, 58.0, 46.0, 52.0, 43.0];
    let incomes: Vec<f64> = vec![
        68000.0, 72000.0, 65000.0, 78000.0, 71000.0, 67000.0, 82000.0, 69000.0, 75000.0, 66000.0,
    ];
    let categories: Vec<&str> = vec!["C", "C", "B", "C", "C", "A", "C", "B", "C", "C"];
    let scores: Vec<i32> = vec![78, 85, 71, 94, 81, 75, 91, 79, 88, 73];

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(ages)),
            Arc::new(Float64Array::from(incomes)),
            Arc::new(StringArray::from(categories)),
            Arc::new(Int32Array::from(scores)),
        ],
    )?;

    ArrowDataset::from_batch(batch)
}

fn create_slightly_drifted_dataset() -> alimentar::Result<ArrowDataset> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("age", DataType::Float64, false),
        Field::new("income", DataType::Float64, false),
        Field::new("category", DataType::Utf8, false),
        Field::new("score", DataType::Int32, false),
    ]));

    // Slight drift: Small changes
    let ages: Vec<f64> = vec![33.0, 36.0, 29.0, 43.0, 39.0, 32.0, 46.0, 34.0, 38.0, 30.0];
    let incomes: Vec<f64> = vec![
        49000.0, 53000.0, 46000.0, 56000.0, 52000.0, 48000.0, 59000.0, 50000.0, 54000.0, 47000.0,
    ];
    let categories: Vec<&str> = vec!["A", "B", "A", "C", "B", "B", "B", "A", "C", "A"];
    let scores: Vec<i32> = vec![76, 83, 69, 92, 79, 73, 89, 77, 86, 71];

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(ages)),
            Arc::new(Float64Array::from(incomes)),
            Arc::new(StringArray::from(categories)),
            Arc::new(Int32Array::from(scores)),
        ],
    )?;

    ArrowDataset::from_batch(batch)
}

fn main() -> alimentar::Result<()> {
    println!("=== Alimentar Drift Detection Example ===\n");

    let baseline = create_baseline_dataset()?;
    let drifted = create_drifted_dataset()?;
    let slight_drift = create_slightly_drifted_dataset()?;

    println!("Created datasets:");
    println!("  Baseline: {} rows", baseline.len());
    println!("  Drifted: {} rows", drifted.len());
    println!("  Slight drift: {} rows", slight_drift.len());

    // 1. Basic drift detection - create detector with baseline (reference)
    println!("\n1. Basic drift detection (baseline vs drifted)");
    let detector = DriftDetector::new(baseline.clone());
    let report = detector.detect(&drifted)?;

    println!("   Overall drift detected: {}", report.drift_detected);
    println!("   Timestamp: {}", report.timestamp);

    // 2. Column-level drift analysis
    println!("\n2. Column-level drift analysis");
    for (col_key, col_drift) in &report.column_scores {
        println!(
            "\n   Column: {} (test: {:?})",
            col_drift.column, col_drift.test
        );
        println!("     Drift detected: {}", col_drift.drift_detected);
        println!("     Severity: {:?}", col_drift.severity);
        println!("     Statistic: {:.4}", col_drift.statistic);
        if let Some(p) = col_drift.p_value {
            println!("     P-value: {:.4}", p);
        }
        let _ = col_key; // Use the key
    }

    // 3. Drift severity levels
    println!("\n3. Drift severity classification");
    for (_, col_drift) in &report.column_scores {
        let severity_desc = match col_drift.severity {
            DriftSeverity::None => "No drift",
            DriftSeverity::Low => "Low drift (monitor)",
            DriftSeverity::Medium => "Medium drift (investigate)",
            DriftSeverity::High => "High drift (action required)",
            DriftSeverity::Critical => "Critical drift (immediate action)",
        };
        println!("   {}: {}", col_drift.column, severity_desc);
    }

    // 4. Compare with slight drift
    println!("\n4. Comparing with slight drift dataset");
    let detector_slight = DriftDetector::new(baseline.clone());
    let report_slight = detector_slight.detect(&slight_drift)?;

    println!("   Overall drift: {}", report_slight.drift_detected);

    // 5. Specific column comparison
    println!("\n5. Detailed comparison: 'income' column");

    let income_drift_heavy = report.column_scores.values().find(|c| c.column == "income");
    let income_drift_slight = report_slight
        .column_scores
        .values()
        .find(|c| c.column == "income");

    if let (Some(heavy), Some(slight)) = (income_drift_heavy, income_drift_slight) {
        println!("   Heavy drift:");
        println!("     Statistic: {:.4}", heavy.statistic);
        println!("     Severity: {:?}", heavy.severity);
        println!("   Slight drift:");
        println!("     Statistic: {:.4}", slight.statistic);
        println!("     Severity: {:?}", slight.severity);
    }

    // 6. Categorical drift (category column)
    println!("\n6. Categorical drift detection");
    let cat_drift = report
        .column_scores
        .values()
        .find(|c| c.column == "category");

    if let Some(drift) = cat_drift {
        println!("   Category column drift:");
        println!("     Has drift: {}", drift.drift_detected);
        println!("     Statistic: {:.4}", drift.statistic);
        println!("     Test used: {:?}", drift.test);
    }

    // 7. Recommendations based on drift
    println!("\n7. Recommendations");
    for (_, col_drift) in &report.column_scores {
        if col_drift.drift_detected {
            match col_drift.severity {
                DriftSeverity::Critical | DriftSeverity::High => {
                    println!(
                        "   [ACTION] {}: Retrain model or investigate data pipeline",
                        col_drift.column
                    );
                }
                DriftSeverity::Medium => {
                    println!("   [MONITOR] {}: Schedule investigation", col_drift.column);
                }
                DriftSeverity::Low => {
                    println!(
                        "   [INFO] {}: Minor drift, continue monitoring",
                        col_drift.column
                    );
                }
                DriftSeverity::None => {}
            }
        }
    }

    // 8. Detector with multiple tests
    println!("\n8. Using multiple statistical tests");
    let multi_detector = DriftDetector::new(baseline.clone())
        .with_test(DriftTest::KolmogorovSmirnov)
        .with_test(DriftTest::PSI);

    let multi_report = multi_detector.detect(&drifted)?;
    println!("   Tests applied: KS + PSI");
    println!("   Overall drift: {}", multi_report.drift_detected);
    println!("   Column results: {}", multi_report.column_scores.len());

    // 9. Custom alpha threshold
    println!("\n9. Custom significance threshold");
    let strict_detector = DriftDetector::new(baseline).with_alpha(0.01); // Stricter threshold

    let strict_report = strict_detector.detect(&drifted)?;
    println!("   With alpha=0.01 (stricter):");
    println!(
        "   Columns with significant drift: {}",
        strict_report
            .column_scores
            .values()
            .filter(|c| c.drift_detected)
            .count()
    );

    // 10. Summary statistics
    println!("\n10. Drift summary");
    let drift_count = report
        .column_scores
        .values()
        .filter(|c| c.drift_detected)
        .count();
    let critical_count = report
        .column_scores
        .values()
        .filter(|c| matches!(c.severity, DriftSeverity::Critical | DriftSeverity::High))
        .count();

    println!("   Total columns analyzed: {}", report.column_scores.len());
    println!("   Columns with drift: {}", drift_count);
    println!("   Critical/High severity: {}", critical_count);

    println!("\n=== Example Complete ===");
    Ok(())
}
