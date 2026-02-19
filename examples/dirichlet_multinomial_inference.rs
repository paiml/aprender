#![allow(clippy::disallowed_methods)]
//! Dirichlet-Multinomial Bayesian Inference Example
//!
//! Demonstrates how to use conjugate priors for categorical data (k > 2 categories).
//! This example shows:
//! 1. Prior specification (uniform, informative) for probability simplex
//! 2. Sequential Bayesian updating for multinomial data
//! 3. Vector-valued posterior statistics (one per category)
//! 4. Real-world scenarios: Customer preference, survey analysis, text classification

use aprender::prelude::*;

fn main() {
    println!("Dirichlet-Multinomial Bayesian Inference");
    println!("=========================================\n");

    // Example 1: Customer product preference
    println!("Example 1: Customer Product Preference");
    println!("--------------------------------------");
    product_preference_example();

    println!("\nExample 2: Survey Response Analysis");
    println!("-----------------------------------");
    survey_analysis_example();

    println!("\nExample 3: Sequential Learning");
    println!("------------------------------");
    sequential_learning_example();

    println!("\nExample 4: Prior Comparison");
    println!("---------------------------");
    prior_comparison_example();
}

fn product_preference_example() {
    // Market research: Customer preference among 4 smartphone brands
    // Brands: A, B, C, D
    // Survey data: 120 customers chose: [35, 45, 25, 15]

    let mut model = DirichletMultinomial::uniform(4);
    println!("Prior: Uniform Dirichlet(1, 1, 1, 1)");
    let prior_probs = model.posterior_mean();
    println!("  Prior probabilities: {prior_probs:?}");
    println!("  (All brands equally likely: 25% each)\n");

    // Update with survey responses
    let brand_counts = vec![35, 45, 25, 15]; // [A, B, C, D]
    model.update(&brand_counts);

    let posterior_probs = model.posterior_mean();
    let posterior_mode = model.posterior_mode().expect("Mode exists");

    println!(
        "After surveying {} customers:",
        brand_counts.iter().sum::<u32>()
    );
    println!("  Brand A: {:.2}% market share", posterior_probs[0] * 100.0);
    println!("  Brand B: {:.2}% market share", posterior_probs[1] * 100.0);
    println!("  Brand C: {:.2}% market share", posterior_probs[2] * 100.0);
    println!("  Brand D: {:.2}% market share", posterior_probs[3] * 100.0);

    println!("\n  Posterior mode (MAP estimates):");
    println!("    Brand A: {:.2}%", posterior_mode[0] * 100.0);
    println!("    Brand B: {:.2}%", posterior_mode[1] * 100.0);
    println!("    Brand C: {:.2}%", posterior_mode[2] * 100.0);
    println!("    Brand D: {:.2}%", posterior_mode[3] * 100.0);

    // 95% credible intervals
    let intervals = model.credible_intervals(0.95).expect("Valid confidence");
    println!("\n  95% Credible Intervals:");
    let brands = ["A", "B", "C", "D"];
    for (i, brand) in brands.iter().enumerate() {
        println!(
            "    Brand {}: [{:.2}%, {:.2}%]",
            brand,
            intervals[i].0 * 100.0,
            intervals[i].1 * 100.0
        );
    }

    println!("\nInterpretation:");
    let max_idx = posterior_probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("Valid f32 comparison"))
        .expect("Non-empty probability vector")
        .0;
    println!(
        "  ✓ Brand {} is the market leader with {:.1}% share",
        brands[max_idx],
        posterior_probs[max_idx] * 100.0
    );

    // Check if leader's lower bound exceeds others' upper bounds
    if intervals[max_idx].0
        > intervals
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != max_idx)
            .map(|(_, iv)| iv.1)
            .fold(0.0, f32::max)
    {
        println!("    Leadership is statistically significant (95% confidence)");
    } else {
        println!("    Consider surveying more customers for conclusive results");
    }
}

fn survey_analysis_example() {
    // Political survey: Voter preference among 5 candidates
    // Compare two regions to detect regional differences

    // Region 1: Urban area (300 voters)
    let region1_votes = vec![85, 70, 65, 50, 30]; // [Candidate 1-5]
    let mut model1 = DirichletMultinomial::uniform(5);
    model1.update(&region1_votes);
    let probs1 = model1.posterior_mean();
    let intervals1 = model1.credible_intervals(0.95).expect("Valid confidence");

    println!(
        "Region 1 (Urban): {} voters",
        region1_votes.iter().sum::<u32>()
    );
    for i in 0..5 {
        println!(
            "  Candidate {}: {:.2}% [{:.2}%, {:.2}%]",
            i + 1,
            probs1[i] * 100.0,
            intervals1[i].0 * 100.0,
            intervals1[i].1 * 100.0
        );
    }

    // Region 2: Rural area (200 voters)
    let region2_votes = vec![30, 45, 60, 40, 25]; // [Candidate 1-5]
    let mut model2 = DirichletMultinomial::uniform(5);
    model2.update(&region2_votes);
    let probs2 = model2.posterior_mean();
    let intervals2 = model2.credible_intervals(0.95).expect("Valid confidence");

    println!(
        "\nRegion 2 (Rural): {} voters",
        region2_votes.iter().sum::<u32>()
    );
    for i in 0..5 {
        println!(
            "  Candidate {}: {:.2}% [{:.2}%, {:.2}%]",
            i + 1,
            probs2[i] * 100.0,
            intervals2[i].0 * 100.0,
            intervals2[i].1 * 100.0
        );
    }

    println!("\nRegional Differences:");
    for i in 0..5 {
        let diff = (probs1[i] - probs2[i]).abs() * 100.0;
        if diff > 5.0 {
            // Check if intervals don't overlap
            if intervals1[i].1 < intervals2[i].0 || intervals2[i].1 < intervals1[i].0 {
                println!(
                    "  ✓ Candidate {} shows significant regional difference: {:.1}% vs {:.1}%",
                    i + 1,
                    probs1[i] * 100.0,
                    probs2[i] * 100.0
                );
            }
        }
    }

    // Identify leaders in each region
    let leader1 = probs1
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("Valid f32 comparison"))
        .expect("Non-empty probability vector")
        .0;
    let leader2 = probs2
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("Valid f32 comparison"))
        .expect("Non-empty probability vector")
        .0;

    println!("\nConclusion:");
    if leader1 == leader2 {
        println!(
            "  ✓ Same leader (Candidate {}) in both regions",
            leader1 + 1
        );
    } else {
        println!(
            "  ⚠ Different regional leaders: Candidate {} (urban) vs Candidate {} (rural)",
            leader1 + 1,
            leader2 + 1
        );
        println!("    Campaign strategy should be region-specific.");
    }
}

fn sequential_learning_example() {
    // Text classification: Document categories in streaming data
    // Categories: Tech, Sports, Politics, Entertainment, Business (5 categories)
    let mut model = DirichletMultinomial::uniform(5);

    let experiments = vec![
        vec![12, 8, 15, 10, 5],   // Batch 1: 50 documents
        vec![18, 12, 20, 15, 10], // Batch 2: 75 documents
        vec![22, 16, 25, 18, 14], // Batch 3: 95 documents
        vec![28, 20, 30, 22, 18], // Batch 4: 118 documents
        vec![35, 25, 38, 28, 22], // Batch 5: 148 documents
    ];

    println!("Sequential document classification (5 categories):\n");
    println!("  Docs | Tech   Sports  Politics  Entmt   Business | Variance(avg)");
    println!("  -----|------------------------------------------------|-------------");

    let mut total_docs = 0;
    for batch in experiments {
        model.update(&batch);
        total_docs += batch.iter().sum::<u32>();

        let probs = model.posterior_mean();
        let variances = model.posterior_variance();
        let avg_variance = variances.iter().sum::<f32>() / variances.len() as f32;

        print!("  {total_docs:4} |");
        for prob in &probs {
            print!(" {prob:.3}");
        }
        println!("  | {avg_variance:.7}");
    }

    println!("\nObservation: Posterior probabilities stabilize as data accumulates.");
    println!("  Average variance decreases, reflecting increased confidence.");

    let final_probs = model.posterior_mean();
    println!("\nFinal probability distribution:");
    let categories = ["Tech", "Sports", "Politics", "Entertainment", "Business"];
    for (i, cat) in categories.iter().enumerate() {
        println!("  {}: {:.2}%", cat, final_probs[i] * 100.0);
    }
}

fn prior_comparison_example() {
    // Compare different priors with same data
    // Data: Website visit counts across 3 pages: [45, 30, 25]

    let page_visits = vec![45, 30, 25];

    println!("Effect of prior choice (data: {page_visits:?}):\n");

    // Uniform prior Dirichlet(1, 1, 1)
    let mut uniform = DirichletMultinomial::uniform(3);
    uniform.update(&page_visits);
    let probs_uniform = uniform.posterior_mean();
    println!("Uniform Prior Dirichlet(1, 1, 1):");
    println!("  Posterior probabilities: {probs_uniform:?}");
    println!(
        "  → Page 1: {:.2}%, Page 2: {:.2}%, Page 3: {:.2}%",
        probs_uniform[0] * 100.0,
        probs_uniform[1] * 100.0,
        probs_uniform[2] * 100.0
    );

    // Weakly informative prior Dirichlet(2, 2, 2)
    let mut weak = DirichletMultinomial::new(vec![2.0, 2.0, 2.0]).expect("Valid parameters");
    weak.update(&page_visits);
    let probs_weak = weak.posterior_mean();
    println!("\nWeakly Informative Prior Dirichlet(2, 2, 2):");
    println!("  Posterior probabilities: {probs_weak:?}");
    println!(
        "  → Page 1: {:.2}%, Page 2: {:.2}%, Page 3: {:.2}%",
        probs_weak[0] * 100.0,
        probs_weak[1] * 100.0,
        probs_weak[2] * 100.0
    );

    // Informative prior Dirichlet(30, 30, 30) [strong equal belief]
    let mut informative =
        DirichletMultinomial::new(vec![30.0, 30.0, 30.0]).expect("Valid parameters");
    informative.update(&page_visits);
    let probs_inf = informative.posterior_mean();
    println!("\nInformative Prior Dirichlet(30, 30, 30) [strong equal belief]:");
    println!("  Posterior probabilities: {probs_inf:?}");
    println!(
        "  → Page 1: {:.2}%, Page 2: {:.2}%, Page 3: {:.2}%",
        probs_inf[0] * 100.0,
        probs_inf[1] * 100.0,
        probs_inf[2] * 100.0
    );

    println!("\nObservation:");
    println!("  Uniform prior: Data-driven (45%, 30%, 25%)");
    println!("  Weak prior: Similar to uniform");
    println!("  Informative prior: Pulled toward equal probabilities (33%, 33%, 33%)");
    println!("    Prior effective sample size = {}", 30 + 30 + 30);
    println!(
        "    Actual sample size = {}",
        page_visits.iter().sum::<u32>()
    );
    println!("  → Strong prior dominates with limited data!");
}
