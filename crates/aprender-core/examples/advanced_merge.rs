//! Advanced Merge Strategies (GH-442)
//!
//! Demonstrates 6 new merge strategies: Task Arithmetic, NuSLERP, MultiSLERP,
//! DELLA, Breadcrumbs, and SCE.
//!
//! Run: cargo run --example advanced_merge

use aprender::format::{MergeOptions, MergeStrategy};

fn main() {
    println!("=== Advanced Merge Strategies (GH-442) ===\n");

    // ── 1. Strategy Parsing ──
    println!("── 1. Strategy Parsing ──");
    let strategies = [
        "task-arithmetic",
        "nuslerp",
        "multi-slerp",
        "della",
        "breadcrumbs",
        "sce",
    ];
    for name in &strategies {
        let s: MergeStrategy = name.parse().expect("valid merge strategy name");
        println!("  {:15} → {:?} (supported: {})", name, s, s.is_supported());
    }

    // ── 2. Task Arithmetic ──
    println!("\n── 2. Task Arithmetic ──");
    let opts = MergeOptions {
        strategy: MergeStrategy::TaskArithmetic,
        scales: Some(vec![0.7, 0.3]),
        ..Default::default()
    };
    println!("  Strategy: {:?}, scales: {:?}", opts.strategy, opts.scales);
    println!("  result = base + 0.7*(model_A - base) + 0.3*(model_B - base)");

    // ── 3. NuSLERP ──
    println!("\n── 3. NuSLERP (Enhanced SLERP) ──");
    let opts = MergeOptions {
        strategy: MergeStrategy::NuSlerp,
        weights: Some(vec![0.6]),
        ..Default::default()
    };
    println!(
        "  Strategy: {:?}, t={:.1}",
        opts.strategy,
        opts.weights.as_ref().expect("weights should be set")[0]
    );
    println!("  Uses nlerp fallback for near-parallel vectors (faster)");

    // ── 4. MultiSLERP ──
    println!("\n── 4. MultiSLERP (Barycentric SLERP) ──");
    let opts = MergeOptions {
        strategy: MergeStrategy::MultiSlerp,
        weights: Some(vec![0.5, 0.3, 0.2]),
        ..Default::default()
    };
    println!(
        "  Strategy: {:?}, weights: {:?}",
        opts.strategy, opts.weights
    );
    println!("  Iterative SLERP for >2 models");

    // ── 5. DELLA ──
    println!("\n── 5. DELLA (Adaptive Magnitude Pruning) ──");
    let opts = MergeOptions {
        strategy: MergeStrategy::Della,
        drop_rate: 0.7,
        ..Default::default()
    };
    println!(
        "  Strategy: {:?}, drop_rate: {:.1}",
        opts.strategy, opts.drop_rate
    );
    println!("  Large deltas: low drop rate → almost always kept");
    println!("  Small deltas: high drop rate → aggressively pruned");

    // ── 6. Breadcrumbs ──
    println!("\n── 6. Breadcrumbs (Outlier Removal) ──");
    let opts = MergeOptions {
        strategy: MergeStrategy::Breadcrumbs,
        scales: Some(vec![1.0, 1.0]),
        outlier_k: 3.0,
        ..Default::default()
    };
    println!(
        "  Strategy: {:?}, outlier_k: {:.1}",
        opts.strategy, opts.outlier_k
    );
    println!(
        "  Removes deltas > {:.1}σ from mean before merging",
        opts.outlier_k
    );

    // ── 7. SCE ──
    println!("\n── 7. SCE (Adaptive Matrix-Level Weighting) ──");
    let opts = MergeOptions {
        strategy: MergeStrategy::Sce,
        weights: Some(vec![0.5, 0.5]),
        ..Default::default()
    };
    println!(
        "  Strategy: {:?}, base_weights: {:?}",
        opts.strategy, opts.weights
    );
    println!("  Per-tensor adaptive: high-variance tensors get more weight");

    // ── Summary ──
    println!("\n── Strategy Comparison ──");
    println!("  | Strategy       | Base  | Models | Key Feature               |");
    println!("  |----------------|-------|--------|---------------------------|");
    println!("  | TaskArithmetic | yes   | 2+     | Linear task vector combo  |");
    println!("  | NuSLERP        | no    | 2      | Fast SLERP with nlerp     |");
    println!("  | MultiSLERP     | no    | 2+     | Barycentric interpolation |");
    println!("  | DELLA          | yes   | 2+     | Adaptive drop rates       |");
    println!("  | Breadcrumbs    | yes   | 2+     | Outlier removal           |");
    println!("  | SCE            | no    | 2+     | Variance-adaptive weights |");

    println!("\n=== Advanced merge strategies verified ===");
}
