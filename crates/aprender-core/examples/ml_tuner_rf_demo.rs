#![allow(clippy::disallowed_methods)]
//! ML Tuner RandomForest showcase (SHOWCASE-BRICK-001, Section 12).
//!
//! The SIMD `trueno` tuner produces `TunerFeatures` vectors; this feeds them into
//! `aprender::RandomForestRegressor`/`RandomForestClassifier`. Moved here from
//! aprender-compute's `ml_tuner_demo` so aprender-compute needs no dev-dependency on
//! aprender-core (#5833).
//!
//!   cargo run -p aprender-core --example ml_tuner_rf_demo

use aprender::primitives::{Matrix, Vector};
use aprender::tree::{RandomForestClassifier, RandomForestRegressor};
use trueno::tuner::{QuantType, TunerFeatures};

fn main() {
    // Qwen2.5-Coder-1.5B on RTX 4090, as in aprender-compute's ml_tuner_demo.
    let features = TunerFeatures::builder()
        .model_params_b(1.5)
        .hidden_dim(1536)
        .num_layers(28)
        .num_heads(12)
        .batch_size(4)
        .seq_len(512)
        .quant_type(QuantType::Q4K)
        .gpu_mem_bw_gbs(1000.0)
        .gpu_sm_count(128)
        .cuda_graphs(true)
        .build();
    println!("RandomForest Models (ml-tuner showcase)");
    println!("--------------------------------------");
    let features = &features;
    println!("   --------------------------------------");

    let dim = TunerFeatures::DIM;
    let n = 100usize;

    // Regressor: trueno features -> predicted throughput (tok/s).
    let mut xr = Vec::with_capacity(n * dim);
    let mut yr = Vec::with_capacity(n);
    for i in 0..n {
        let batch = 1 + (i % 8) as u32;
        let f = TunerFeatures::builder()
            .model_params_b(1.5)
            .batch_size(batch)
            .quant_type(QuantType::Q4K)
            .gpu_mem_bw_gbs(1000.0)
            .cuda_graphs(batch == 1)
            .build();
        xr.extend(f.to_vector());
        yr.push(200.0 + (batch as f32) * 80.0 + (i as f32 * 0.5));
    }
    println!("   Generated {n} regression samples");
    match Matrix::from_vec(n, dim, xr) {
        Ok(x_mat) => {
            let mut rf = RandomForestRegressor::new(100);
            match rf.fit(&x_mat, &Vector::from_vec(yr)) {
                Ok(()) => {
                    let fx = Matrix::from_vec(1, dim, features.to_vector().to_vec())
                        .expect("single-row feature matrix");
                    let pred = rf.predict(&fx);
                    println!(
                        "   RF regressor: SUCCESS — M=4 prediction {:.1} tok/s",
                        pred.as_slice()[0]
                    );
                }
                Err(e) => println!("   RF regressor: FAILED - {e}"),
            }
        }
        Err(e) => println!("   RF regressor matrix: FAILED - {e}"),
    }

    // Classifier: trueno features -> kernel-type label (3=Batched, 2=Vectorized).
    let mut xc = Vec::with_capacity(n * dim);
    let mut yc: Vec<usize> = Vec::with_capacity(n);
    for i in 0..n {
        let batch = 1 + (i % 8) as u32;
        let f = TunerFeatures::builder()
            .model_params_b(1.5)
            .batch_size(batch)
            .quant_type(QuantType::Q4K)
            .build();
        xc.extend(f.to_vector());
        yc.push(if batch >= 4 { 3 } else { 2 });
    }
    match Matrix::from_vec(n, dim, xc) {
        Ok(xc_mat) => {
            let mut rf = RandomForestClassifier::new(50);
            match rf.fit(&xc_mat, &yc) {
                Ok(()) => {
                    let preds = rf.predict(&xc_mat);
                    let correct = yc
                        .iter()
                        .zip(preds.iter())
                        .filter(|(a, b)| **a == **b)
                        .count();
                    println!(
                        "   RF classifier: SUCCESS — train accuracy {:.1}%",
                        correct as f32 / n as f32 * 100.0
                    );
                }
                Err(e) => println!("   RF classifier: FAILED - {e}"),
            }
        }
        Err(e) => println!("   RF classifier matrix: FAILED - {e}"),
    }
    println!();
}
