#![allow(clippy::disallowed_methods)]
//! DataFrame Basics example
//!
//! Demonstrates core DataFrame operations for ML workflows.

use aprender::prelude::*;

fn main() {
    println!("DataFrame Basics Example");
    println!("========================\n");

    // Create a DataFrame with named columns
    let columns = vec![
        (
            "age".to_string(),
            Vector::from_slice(&[25.0, 30.0, 35.0, 40.0, 45.0]),
        ),
        (
            "income".to_string(),
            Vector::from_slice(&[50000.0, 60000.0, 75000.0, 90000.0, 110000.0]),
        ),
        (
            "score".to_string(),
            Vector::from_slice(&[85.0, 90.0, 88.0, 92.0, 95.0]),
        ),
    ];

    let df = DataFrame::new(columns).expect("Failed to create DataFrame");

    // Basic info
    println!("DataFrame Shape: {:?}", df.shape());
    println!("Columns: {:?}\n", df.column_names());

    // Access individual columns
    let ages = df.column("age").expect("Example data should be valid");
    println!("Ages: {:?}", ages.as_slice());
    println!("Mean age: {:.1}\n", ages.mean());

    // Select specific columns
    let selected = df
        .select(&["age", "income"])
        .expect("Example data should be valid");
    println!("Selected columns shape: {:?}", selected.shape());

    // Access rows
    let row = df.row(2).expect("Example data should be valid");
    println!("Row 2: {:?}", row.as_slice());

    // Convert to Matrix for ML algorithms
    let matrix = df.to_matrix();
    println!("\nMatrix shape: {:?}", matrix.shape());
    println!(
        "First row of matrix: [{:.0}, {:.0}, {:.0}]",
        matrix.get(0, 0),
        matrix.get(0, 1),
        matrix.get(0, 2)
    );

    // Descriptive statistics
    println!("\nDescriptive Statistics:");
    println!(
        "{:>10} {:>10} {:>10} {:>10} {:>10}",
        "Column", "Mean", "Std", "Min", "Max"
    );
    println!("{}", "-".repeat(52));

    for stats in df.describe() {
        println!(
            "{:>10} {:>10.1} {:>10.1} {:>10.1} {:>10.1}",
            stats.name, stats.mean, stats.std, stats.min, stats.max
        );
    }

    // Use DataFrame with Linear Regression
    println!("\n--- Linear Regression with DataFrame ---\n");

    // Use age and income to predict score
    let features = df
        .select(&["age", "income"])
        .expect("Example data should be valid");
    let x = features.to_matrix();
    let y = df
        .column("score")
        .expect("Example data should be valid")
        .clone();

    let mut model = LinearRegression::new();
    model.fit(&x, &y).expect("Failed to fit model");

    let predictions = model.predict(&x);
    let r2 = model.score(&x, &y);

    println!("Coefficients:");
    println!("  age coefficient:    {:.6}", model.coefficients()[0]);
    println!("  income coefficient: {:.10}", model.coefficients()[1]);
    println!("  intercept:          {:.4}", model.intercept());
    println!("\nR² Score: {r2:.4}");

    // Predictions
    println!("\nPredictions vs Actual:");
    println!("{:>8} {:>8}", "Actual", "Pred");
    for i in 0..y.len() {
        println!("{:>8.1} {:>8.1}", y[i], predictions[i]);
    }

    println!("\nDataFrame operations completed successfully!");
}
