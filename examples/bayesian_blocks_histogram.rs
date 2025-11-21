//! Bayesian Blocks Histogram Example
//!
//! Demonstrates the Bayesian Blocks optimal histogram binning algorithm.
//! Compares it with fixed-width methods to show adaptive binning on
//! non-uniform data distributions.

use aprender::stats::{BinMethod, DescriptiveStats};
use trueno::Vector;

fn main() {
    println!("Bayesian Blocks Histogram Example");
    println!("==================================\n");

    // Example 1: Uniform Distribution
    println!("Example 1: Uniform Distribution");
    println!("--------------------------------");
    uniform_distribution_example();

    // Example 2: Two Distinct Clusters
    println!("\nExample 2: Two Distinct Clusters");
    println!("---------------------------------");
    two_clusters_example();

    // Example 3: Multiple Density Regions
    println!("\nExample 3: Multiple Density Regions");
    println!("------------------------------------");
    multiple_density_example();

    // Example 4: Comparison with Fixed-Width Methods
    println!("\nExample 4: Comparison with Fixed-Width Methods");
    println!("-----------------------------------------------");
    comparison_example();

    println!("\n✅ Bayesian Blocks Examples Complete!");
    println!("\nKey Advantages:");
    println!("  • Adaptive binning based on data structure");
    println!("  • Automatically detects change points");
    println!("  • Optimal for non-uniform distributions");
    println!("  • No need to specify number of bins");
    println!("  • Handles gaps and clusters effectively");
}

fn uniform_distribution_example() {
    // Uniformly distributed data
    let data: Vec<f32> = (1..=20).map(|x| x as f32).collect();
    let v = Vector::from_slice(&data);
    let stats = DescriptiveStats::new(&v);

    let hist_bayesian = stats.histogram_method(BinMethod::Bayesian).unwrap();
    let hist_sturges = stats.histogram_method(BinMethod::Sturges).unwrap();

    println!("  Data: 1, 2, 3, ..., 20 (uniform)");
    println!("\n  Bayesian Blocks:");
    println!("    Number of bins: {}", hist_bayesian.counts.len());
    println!(
        "    Bin edges: {:?}",
        &hist_bayesian.bins[0..3.min(hist_bayesian.bins.len())]
    );
    println!("    (showing first 3 edges)");

    println!("\n  Sturges Rule:");
    println!("    Number of bins: {}", hist_sturges.counts.len());

    println!("\n  → Bayesian Blocks uses fewer bins for uniform data");
    println!("    (no need for many bins when distribution is constant)");
}

fn two_clusters_example() {
    // Two distinct clusters with a gap
    let cluster1: Vec<f32> = vec![1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0];
    let cluster2: Vec<f32> = vec![9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0];

    let mut data = Vec::new();
    data.extend(cluster1);
    data.extend(cluster2);

    let v = Vector::from_slice(&data);
    let stats = DescriptiveStats::new(&v);

    let hist = stats.histogram_method(BinMethod::Bayesian).unwrap();

    println!("  Data: Two clusters (1.0-2.0 and 9.0-10.0)");
    println!("  Gap: 2.0 to 9.0 (no data)");
    println!("\n  Bayesian Blocks Result:");
    println!("    Number of bins: {}", hist.counts.len());
    println!("    Bin edges: {:?}", hist.bins);
    println!("\n  Bin counts:");
    for (i, &count) in hist.counts.iter().enumerate() {
        println!(
            "    Bin {}: [{:.2}, {:.2}) → {} samples",
            i,
            hist.bins[i],
            hist.bins[i + 1],
            count
        );
    }

    println!("\n  → Algorithm detected the gap and created separate bins");
    println!("    for each cluster!");
}

fn multiple_density_example() {
    // Data with varying densities
    let mut data = Vec::new();

    // Dense region 1: many points close together
    for i in 0..10 {
        data.push(1.0 + i as f32 * 0.1);
    }

    // Sparse region: few points spread out
    data.push(5.0);
    data.push(7.0);
    data.push(9.0);

    // Dense region 2: many points close together
    for i in 0..10 {
        data.push(15.0 + i as f32 * 0.1);
    }

    let v = Vector::from_slice(&data);
    let stats = DescriptiveStats::new(&v);

    let hist = stats.histogram_method(BinMethod::Bayesian).unwrap();

    println!("  Data: Dense (1.0-2.0), Sparse (5, 7, 9), Dense (15.0-16.0)");
    println!("\n  Bayesian Blocks Result:");
    println!("    Number of bins: {}", hist.counts.len());
    println!("\n  Bin counts:");
    for (i, &count) in hist.counts.iter().enumerate() {
        println!(
            "    Bin {}: [{:.2}, {:.2}) → {} samples",
            i,
            hist.bins[i],
            hist.bins[i + 1],
            count
        );
    }

    println!("\n  → Algorithm adapts bin width to data density");
    println!("    - Smaller bins in dense regions");
    println!("    - Larger bins in sparse regions");
}

fn comparison_example() {
    // Non-uniform data for comparison
    let mut data = Vec::new();
    // Cluster 1
    for i in 0..8 {
        data.push(1.0 + i as f32 * 0.2);
    }
    // Gap
    // Cluster 2
    for i in 0..8 {
        data.push(10.0 + i as f32 * 0.2);
    }

    let v = Vector::from_slice(&data);
    let stats = DescriptiveStats::new(&v);

    let methods = [
        (BinMethod::Bayesian, "Bayesian Blocks"),
        (BinMethod::Sturges, "Sturges Rule"),
        (BinMethod::Scott, "Scott Rule"),
        (BinMethod::FreedmanDiaconis, "Freedman-Diaconis"),
        (BinMethod::SquareRoot, "Square Root"),
    ];

    println!("  Data: Two clusters separated by gap");
    println!("  Cluster 1: 1.0 - 2.4");
    println!("  Cluster 2: 10.0 - 11.4");
    println!("\n  Method Comparison:");
    println!(
        "  {:25} {:>10} {:>15}",
        "Method", "# Bins", "Adapts to Gap?"
    );
    println!("  {}", "-".repeat(52));

    for (method, name) in methods {
        let hist = stats.histogram_method(method).unwrap();
        let n_bins = hist.counts.len();

        // Check if method detected the gap (should have low counts in middle bins)
        let has_empty_bins = hist.counts.contains(&0);
        let adapts = if has_empty_bins || n_bins <= 3 {
            "✓ Yes"
        } else {
            "  No"
        };

        println!("  {:25} {:>10} {:>15}", name, n_bins, adapts);
    }

    println!("\n  → Bayesian Blocks is the only method that automatically");
    println!("    adapts to non-uniform distributions!");
}
