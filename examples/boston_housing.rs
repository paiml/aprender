//! Boston Housing example - Linear Regression
//!
//! Demonstrates linear regression using simulated housing data.

use aprender::prelude::*;

fn main() {
    println!("Boston Housing - Linear Regression Example");
    println!("==========================================\n");

    // Simulated housing data
    // Features: [sqft, bedrooms, bathrooms, age]
    // Target: price (in thousands)
    let x = Matrix::from_vec(10, 4, vec![
        1500.0, 3.0, 2.0, 10.0,
        2000.0, 4.0, 2.5, 5.0,
        1200.0, 2.0, 1.0, 30.0,
        1800.0, 3.0, 2.0, 15.0,
        2500.0, 5.0, 3.0, 2.0,
        1000.0, 2.0, 1.0, 50.0,
        2200.0, 4.0, 3.0, 8.0,
        1600.0, 3.0, 2.0, 20.0,
        3000.0, 5.0, 4.0, 1.0,
        1400.0, 3.0, 1.5, 25.0,
    ]).unwrap();

    // Prices (simulated based on features)
    let y = Vector::from_slice(&[
        250.0, 350.0, 180.0, 280.0, 450.0,
        150.0, 380.0, 260.0, 520.0, 220.0,
    ]);

    // Train linear regression model
    println!("Training linear regression model...");
    let mut model = LinearRegression::new();
    model.fit(&x, &y).expect("Failed to fit model");

    // Print coefficients
    let coef = model.coefficients();
    println!("\nModel Coefficients:");
    println!("  sqft:      {:.4}", coef[0]);
    println!("  bedrooms:  {:.4}", coef[1]);
    println!("  bathrooms: {:.4}", coef[2]);
    println!("  age:       {:.4}", coef[3]);
    println!("  intercept: {:.4}", model.intercept());

    // Make predictions
    let predictions = model.predict(&x);

    // Print predictions vs actual
    println!("\nPredictions vs Actual:");
    println!("{:>10} {:>10} {:>10}", "Actual", "Predicted", "Error");
    println!("{}", "-".repeat(32));

    for i in 0..y.len() {
        let actual = y.as_slice()[i];
        let predicted = predictions.as_slice()[i];
        let error = actual - predicted;
        println!("{:>10.1} {:>10.1} {:>10.1}", actual, predicted, error);
    }

    // Calculate metrics
    let r2 = model.score(&x, &y);
    let mse_val = mse(&predictions, &y);
    let mae_val = mae(&predictions, &y);

    println!("\nModel Performance:");
    println!("  R² Score: {:.4}", r2);
    println!("  MSE:      {:.4}", mse_val);
    println!("  MAE:      {:.4}", mae_val);

    // Predict on new house
    let new_house = Matrix::from_vec(1, 4, vec![1900.0, 4.0, 2.0, 12.0]).unwrap();
    let predicted_price = model.predict(&new_house);
    println!("\nNew House Prediction:");
    println!("  Features: 1900 sqft, 4 bed, 2 bath, 12 years old");
    println!("  Predicted Price: ${:.0}k", predicted_price.as_slice()[0]);
}
