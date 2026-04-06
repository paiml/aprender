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
    clippy::unnecessary_debug_formatting
)]
//! Data Quality Check Example
//!
//! Demonstrates comprehensive data quality analysis:
//! - Missing value detection
//! - Duplicate detection
//! - Type validation
//! - Range/constraint checking
//! - Quality reports and scores
//!
//! Run with: cargo run --example quality_check

use std::sync::Arc;

use alimentar::{ArrowDataset, Dataset, QualityChecker, QualityIssue, QualityReport};
use arrow::{
    array::{Float64Array, Int32Array, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};

fn create_messy_dataset() -> alimentar::Result<ArrowDataset> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true), // nullable - will have missing
        Field::new("age", DataType::Float64, true), // nullable - will have missing
        Field::new("salary", DataType::Float64, true), // nullable - outliers
        Field::new("department", DataType::Utf8, true), // nullable
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            // ID with duplicates
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 3, 9, 10])), // 3 is duplicate
            // Name with missing values
            Arc::new(StringArray::from(vec![
                Some("Alice"),
                Some("Bob"),
                None,
                Some("David"),
                Some("Eve"),
                None,
                Some("Grace"),
                Some("Henry"),
                Some("Ivy"),
                Some("Jack"),
            ])),
            // Age with missing and outliers
            Arc::new(Float64Array::from(vec![
                Some(25.0),
                Some(30.0),
                Some(35.0),
                None,
                Some(28.0),
                Some(150.0), // outlier: 150
                Some(32.0),
                Some(29.0),
                None,
                Some(27.0),
            ])),
            // Salary with negative value (invalid)
            Arc::new(Float64Array::from(vec![
                Some(50000.0),
                Some(60000.0),
                Some(55000.0),
                Some(70000.0),
                Some(-5000.0), // negative salary (invalid)
                Some(65000.0),
                Some(58000.0),
                Some(62000.0),
                Some(54000.0),
                Some(0.0), // zero salary
            ])),
            // Department
            Arc::new(StringArray::from(vec![
                Some("Engineering"),
                Some("Sales"),
                Some("Engineering"),
                Some("HR"),
                Some("Sales"),
                Some("Marketing"),
                Some("Engineering"),
                Some("Sales"),
                Some(""), // empty string
                Some("HR"),
            ])),
        ],
    )?;

    ArrowDataset::from_batch(batch)
}

fn create_clean_dataset() -> alimentar::Result<ArrowDataset> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0, 50.0])),
        ],
    )?;

    ArrowDataset::from_batch(batch)
}

fn print_quality_report(report: &QualityReport) {
    println!("   Total rows: {}", report.row_count);
    println!("   Total columns: {}", report.column_count);

    // Column-level quality analysis
    println!("\n   Column Quality:");
    for (col_name, col_quality) in &report.columns {
        println!(
            "     {}: null_count={}, null_ratio={:.2}%",
            col_name,
            col_quality.null_count,
            col_quality.null_ratio * 100.0
        );
    }
}

fn main() -> alimentar::Result<()> {
    println!("=== Alimentar Data Quality Check Example ===\n");

    let messy_data = create_messy_dataset()?;
    let clean_data = create_clean_dataset()?;

    println!("Created datasets:");
    println!(
        "  Messy data: {} rows, {} columns",
        messy_data.len(),
        messy_data.schema().fields().len()
    );
    println!(
        "  Clean data: {} rows, {} columns",
        clean_data.len(),
        clean_data.schema().fields().len()
    );

    // 1. Basic quality check
    println!("\n1. Basic quality check on messy data");
    let checker = QualityChecker::new();
    let report = checker.check(&messy_data)?;

    print_quality_report(&report);

    // 2. List all issues
    println!("\n2. Detected quality issues");
    for issue in &report.issues {
        match issue {
            QualityIssue::HighNullRatio {
                column,
                null_ratio,
                threshold,
            } => {
                println!(
                    "   [HIGH NULL] Column '{}': {:.1}% nulls (threshold: {:.1}%)",
                    column,
                    null_ratio * 100.0,
                    threshold * 100.0
                );
            }
            QualityIssue::HighDuplicateRatio {
                column,
                duplicate_ratio,
                threshold,
            } => {
                println!(
                    "   [DUPLICATES] Column '{}': {:.1}% duplicates (threshold: {:.1}%)",
                    column,
                    duplicate_ratio * 100.0,
                    threshold * 100.0
                );
            }
            QualityIssue::LowCardinality {
                column,
                unique_count,
                total_count,
            } => {
                println!(
                    "   [LOW CARDINALITY] Column '{}': {} unique of {} total",
                    column, unique_count, total_count
                );
            }
            QualityIssue::OutliersDetected {
                column,
                outlier_count,
                outlier_ratio,
            } => {
                println!(
                    "   [OUTLIERS] Column '{}': {} outliers ({:.1}%)",
                    column,
                    outlier_count,
                    outlier_ratio * 100.0
                );
            }
            QualityIssue::DuplicateRows {
                duplicate_count,
                duplicate_ratio,
            } => {
                println!(
                    "   [DUPLICATE ROWS] {} duplicate rows ({:.1}%)",
                    duplicate_count,
                    duplicate_ratio * 100.0
                );
            }
            QualityIssue::ConstantColumn { column, value } => {
                println!(
                    "   [CONSTANT] Column '{}' has only one unique value: {:?}",
                    column, value
                );
            }
            QualityIssue::EmptySchema => {
                println!("   [EMPTY SCHEMA] Dataset has no columns");
            }
            QualityIssue::EmptyDataset => {
                println!("   [EMPTY DATASET] Dataset has no rows");
            }
        }
    }

    if report.issues.is_empty() {
        println!("   No quality issues detected!");
    }

    // 3. Quality check on clean data
    println!("\n3. Quality check on clean data");
    let clean_checker = QualityChecker::new();
    let clean_report = clean_checker.check(&clean_data)?;

    print_quality_report(&clean_report);
    println!("   Issues found: {}", clean_report.issues.len());
    if clean_report.issues.is_empty() {
        println!("   Data is clean!");
    }

    // 4. Customized quality checker
    println!("\n4. Customized quality checker with thresholds");
    let strict_checker = QualityChecker::new()
        .max_null_ratio(0.05) // 5% max nulls
        .max_duplicate_ratio(0.5); // 50% max duplicates

    let strict_report = strict_checker.check(&messy_data)?;
    println!("   With strict thresholds:");
    println!("   Issues found: {}", strict_report.issues.len());

    for issue in &strict_report.issues {
        println!("   - {:?}", issue);
    }

    // 5. Column analysis
    println!("\n5. Individual column analysis");
    for (col_name, col_quality) in &report.columns {
        println!("\n   Column '{}':", col_name);
        println!("     Null count: {}", col_quality.null_count);
        println!("     Null ratio: {:.2}%", col_quality.null_ratio * 100.0);
        println!("     Unique count: {}", col_quality.unique_count);
        println!(
            "     Unique ratio: {:.2}%",
            col_quality.unique_ratio * 100.0
        );
    }

    // 6. Quality recommendations
    println!("\n6. Quality improvement recommendations");
    for (col_name, col_quality) in &report.columns {
        if col_quality.null_ratio > 0.5 {
            println!("   [CRITICAL] Drop column '{}' (>50% missing)", col_name);
        } else if col_quality.null_ratio > 0.1 {
            println!("   [ACTION] Impute missing values in '{}'", col_name);
        } else if col_quality.null_ratio > 0.0 {
            println!("   [MINOR] Consider imputation for '{}'", col_name);
        }
    }

    // 7. Summary statistics
    println!("\n7. Quality summary");
    let total_nulls: usize = report.columns.values().map(|c| c.null_count).sum();
    let total_cells = report.row_count * report.column_count;
    let overall_completeness = if total_cells > 0 {
        1.0 - (total_nulls as f64 / total_cells as f64)
    } else {
        1.0
    };

    println!(
        "   Overall completeness: {:.1}%",
        overall_completeness * 100.0
    );
    println!("   Total null values: {}", total_nulls);
    println!("   Total cells: {}", total_cells);

    println!("\n=== Example Complete ===");
    Ok(())
}
