//! Example: Model Merging (TIES, DARE, SLERP)
//!
//! This example demonstrates three model merging algorithms for combining
//! multiple fine-tuned models.

use entrenar::merge::{
    dare_merge, slerp_merge, ties_merge, DareConfig, Model, SlerpConfig, TiesConfig,
};
use entrenar::Tensor;
use std::collections::HashMap;

fn create_model(name: &str, values: Vec<f32>) -> Model {
    let mut model = HashMap::new();
    model.insert(name.to_string(), Tensor::from_vec(values, false));
    model
}

fn print_model(name: &str, model: &Model) {
    println!("\n{name}");
    println!("  w: {:?}", model["w"].data().as_slice().expect("operation should succeed"));
}

fn main() {
    println!("=== Model Merging Examples ===\n");

    // Create a base model and three fine-tuned variants
    let base = create_model("w", vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

    let model1 = create_model("w", vec![1.0, 2.0, 3.0, -1.0, -2.0, 0.5]);
    let model2 = create_model("w", vec![2.0, -1.0, 3.0, 1.0, -1.0, 0.3]);
    let model3 = create_model("w", vec![1.5, 1.0, -2.0, 2.0, 1.0, -0.2]);

    print_model("Base Model", &base);
    print_model("Fine-tuned Model 1", &model1);
    print_model("Fine-tuned Model 2", &model2);
    print_model("Fine-tuned Model 3", &model3);

    // 1. TIES Merge (Task Inference via Elimination and Sign)
    println!("\n\n1. TIES MERGE");
    println!("   Algorithm: Trim top-k%, elect sign by majority vote, merge same-sign values");

    let ties_config = TiesConfig::new(0.6).expect("operation should succeed"); // Keep top 60% magnitude
    let models = vec![model1.clone(), model2.clone(), model3.clone()];

    match ties_merge(&models, &base, &ties_config) {
        Ok(merged) => {
            print_model("TIES Merged Model (density=0.6)", &merged);
            println!("   - Trimmed low-magnitude parameters");
            println!("   - Resolved conflicts via sign voting");
        }
        Err(e) => println!("Error: {e}"),
    }

    // 2. DARE Merge (Drop And REscale)
    println!("\n\n2. DARE MERGE");
    println!("   Algorithm: Randomly drop parameters, rescale remaining values");

    let dare_config = DareConfig::new(0.4).expect("operation should succeed").with_seed(42); // Drop 40% of params
    let models = vec![model1.clone(), model2.clone(), model3.clone()];

    match dare_merge(&models, &base, &dare_config) {
        Ok(merged) => {
            print_model("DARE Merged Model (drop_prob=0.4, seed=42)", &merged);
            println!("   - Stochastic parameter dropping");
            println!("   - Rescaled to maintain expected magnitude");
            println!("   - Deterministic (seeded for reproducibility)");
        }
        Err(e) => println!("Error: {e}"),
    }

    // 3. SLERP Merge (Spherical Linear intERPolation)
    println!("\n\n3. SLERP MERGE (Two models only)");
    println!("   Algorithm: Spherical interpolation along the shortest arc");

    let slerp_config = SlerpConfig::new(0.5).expect("operation should succeed"); // 50/50 blend

    match slerp_merge(&model1, &model2, &slerp_config) {
        Ok(merged) => {
            print_model("SLERP Merged Model (t=0.5)", &merged);
            println!("   - Smooth spherical interpolation");
            println!("   - Constant angular velocity");
            println!("   - Better than linear blending for normalized weights");
        }
        Err(e) => println!("Error: {e}"),
    }

    // Demonstrate SLERP at different interpolation points
    println!("\n\n4. SLERP AT DIFFERENT INTERPOLATION POINTS");
    for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
        let config = SlerpConfig::new(t).expect("operation should succeed");
        match slerp_merge(&model1, &model2, &config) {
            Ok(merged) => {
                println!(
                    "   t={:.2}: {:?}",
                    t,
                    merged["w"].data().as_slice().expect("operation should succeed")
                );
            }
            Err(e) => println!("   t={t:.2}: Error - {e}"),
        }
    }

    // Error handling example
    println!("\n\n5. ERROR HANDLING");

    // Invalid config
    println!("   Testing invalid density parameter...");
    match TiesConfig::new(1.5) {
        Ok(_) => println!("   Unexpected success!"),
        Err(e) => println!("   ✓ Caught error: {e}"),
    }

    // Incompatible shapes
    println!("   Testing incompatible model shapes...");
    let bad_model = create_model("w", vec![1.0, 2.0]); // Wrong size!
    let models = vec![model1.clone(), bad_model];
    match ties_merge(&models, &base, &TiesConfig::default()) {
        Ok(_) => println!("   Unexpected success!"),
        Err(e) => println!("   ✓ Caught error: {e}"),
    }

    println!("\n\n=== Example Complete ===\n");
}
