//! Lottery Ticket Hypothesis Pruning Example
//!
//! Demonstrates finding "winning tickets" - sparse subnetworks that can
//! train to full accuracy when reset to initial weights.
//!
//! # The Lottery Ticket Hypothesis
//! Frankle & Carbin (2018) discovered that dense networks contain sparse
//! subnetworks that, when trained in isolation from their initial weights,
//! can match the test accuracy of the original network.
//!
//! # Algorithm: Iterative Magnitude Pruning (IMP)
//! 1. Initialize network with random weights W₀
//! 2. Train to convergence → W_T
//! 3. Prune the p% smallest-magnitude weights globally
//! 4. Rewind remaining weights to W₀
//! 5. Repeat until target sparsity reached
//!
//! # References
//! - Frankle, J., & Carbin, M. (2018). The Lottery Ticket Hypothesis. ICLR 2019.
//!
//! Run with: cargo run --example lottery_ticket_pruning

use aprender::nn::Linear;
use aprender::pruning::{
    generate_unstructured_mask, LotteryTicketConfig, LotteryTicketPruner, Pruner, RewindStrategy,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         Lottery Ticket Hypothesis with Aprender              ║");
    println!("║         Find sparse subnetworks that train to full accuracy  ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // =========================================================================
    // 1. Create a dense linear layer
    // =========================================================================
    println!("📊 Creating Dense Linear Layer (256 → 128)");
    let layer = Linear::new(256, 128);
    let weights = layer.weight();
    let total_params = weights.data().len();
    println!("   Weight shape: {:?}", weights.shape());
    println!("   Total parameters: {}\n", total_params);

    // =========================================================================
    // 2. Configure Lottery Ticket Search
    // =========================================================================
    println!("⚙️  Configuring Lottery Ticket Search");
    let config = LotteryTicketConfig::new(0.9, 10) // 90% sparsity, 10 rounds
        .with_rewind_strategy(RewindStrategy::Init)
        .with_global_pruning(true);

    println!("   Target sparsity: 90%");
    println!("   Pruning rounds: 10");
    println!("   Rewind strategy: Init (original LTH)");
    println!(
        "   Per-round prune rate: {:.2}%\n",
        config.prune_rate_per_round * 100.0
    );

    // =========================================================================
    // 3. Find the Winning Ticket
    // =========================================================================
    println!("🎰 Finding Winning Ticket (Iterative Magnitude Pruning)...");
    let pruner = LotteryTicketPruner::with_config(config);
    let ticket = pruner.find_ticket(&layer).expect("Failed to find ticket");

    println!("\n✨ Winning Ticket Found!");
    println!("   Total parameters: {}", ticket.total_parameters);
    println!("   Remaining parameters: {}", ticket.remaining_parameters);
    println!("   Final sparsity: {:.2}%", ticket.sparsity * 100.0);
    println!("   Compression ratio: {:.1}x", ticket.compression_ratio());
    println!("   Density: {:.2}%\n", ticket.density() * 100.0);

    // =========================================================================
    // 4. Show Sparsity Progression
    // =========================================================================
    println!("📈 Sparsity Progression:");
    for (round, sparsity) in ticket.sparsity_history.iter().enumerate() {
        let bar_len = (sparsity * 40.0) as usize;
        let bar: String = "█".repeat(bar_len);
        let remaining = (1.0 - sparsity) * 100.0;
        println!(
            "   Round {:2}: {:>5.1}% |{:<40}| ({:.1}% remaining)",
            round + 1,
            sparsity * 100.0,
            bar,
            remaining
        );
    }

    // =========================================================================
    // 5. Compare Rewind Strategies
    // =========================================================================
    println!("\n🔄 Comparing Rewind Strategies (50% sparsity, 5 rounds):");

    let strategies = [
        (RewindStrategy::Init, "Init (W₀)"),
        (RewindStrategy::Early { iteration: 100 }, "Early (W₁₀₀)"),
        (RewindStrategy::Late { fraction: 0.1 }, "Late (W₀.₁T)"),
        (RewindStrategy::None, "None (W_T)"),
    ];

    let small_layer = Linear::new(64, 32);
    for (strategy, name) in strategies {
        let config = LotteryTicketConfig::new(0.5, 5).with_rewind_strategy(strategy);
        let pruner = LotteryTicketPruner::with_config(config);
        let ticket = pruner.find_ticket(&small_layer).expect("Ticket");
        println!(
            "   {:<15} → {:.1}% sparse, {:.1}x compression",
            name,
            ticket.sparsity * 100.0,
            ticket.compression_ratio()
        );
    }

    // =========================================================================
    // 6. Using the Builder Pattern
    // =========================================================================
    println!("\n🔧 Using Builder Pattern:");
    let builder_pruner = LotteryTicketPruner::builder()
        .target_sparsity(0.95)
        .pruning_rounds(15)
        .rewind_strategy(RewindStrategy::Early { iteration: 500 })
        .global_pruning(true)
        .build();

    let builder_ticket = builder_pruner.find_ticket(&layer).expect("Builder ticket");
    println!("   Target: 95% sparsity over 15 rounds");
    println!(
        "   Achieved: {:.2}% sparsity ({:.0}x compression)",
        builder_ticket.sparsity * 100.0,
        builder_ticket.compression_ratio()
    );

    // =========================================================================
    // 7. Using the Pruner Trait Interface
    // =========================================================================
    println!("\n🎯 Using Pruner Trait Interface:");
    let pruner = LotteryTicketPruner::default();
    println!("   Pruner name: {}", pruner.name());

    // Compute importance scores using the pruner's importance estimator
    let importance = pruner.importance();
    let scores = importance.compute(&layer, None).expect("Importance");
    println!("   Importance method: {}", scores.method);
    println!(
        "   Score range: [{:.4}, {:.4}]",
        scores.stats.min, scores.stats.max
    );

    // Generate mask at 50% sparsity using the scores
    let mask = generate_unstructured_mask(&scores.values, 0.5).expect("Mask");
    println!(
        "   Generated mask sparsity: {:.1}%",
        mask.sparsity() * 100.0
    );

    // =========================================================================
    // 8. Memory Savings Analysis
    // =========================================================================
    println!("\n💾 Memory Savings Analysis:");
    let configs = [
        (0.5, "50%"),
        (0.75, "75%"),
        (0.9, "90%"),
        (0.95, "95%"),
        (0.99, "99%"),
    ];

    let analysis_layer = Linear::new(1024, 512);
    let analysis_params = analysis_layer.weight().data().len();
    let original_mb = analysis_params as f32 * 4.0 / 1_000_000.0;
    println!(
        "   Original size: {:.2} MB ({} params)\n",
        original_mb, analysis_params
    );

    println!(
        "   {:>10} {:>12} {:>12} {:>10}",
        "Sparsity", "Remaining", "Size (MB)", "Savings"
    );
    println!("   {:->10} {:->12} {:->12} {:->10}", "", "", "", "");

    for (sparsity, label) in configs {
        let config = LotteryTicketConfig::new(sparsity, 10);
        let pruner = LotteryTicketPruner::with_config(config);
        let ticket = pruner.find_ticket(&analysis_layer).expect("Ticket");

        let remaining_mb = ticket.remaining_parameters as f32 * 4.0 / 1_000_000.0;
        let savings = (1.0 - remaining_mb / original_mb) * 100.0;

        println!(
            "   {:>10} {:>12} {:>12.3} {:>9.1}%",
            label, ticket.remaining_parameters, remaining_mb, savings
        );
    }

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    Lottery Ticket Summary                    ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  The Lottery Ticket Hypothesis shows that dense networks     ║");
    println!("║  contain sparse subnetworks (winning tickets) that can       ║");
    println!("║  train to full accuracy when reset to initial weights.       ║");
    println!("║                                                              ║");
    println!("║  Key findings:                                               ║");
    println!("║  • 90%+ sparsity achievable with minimal accuracy loss       ║");
    println!("║  • 10-100x compression possible                              ║");
    println!("║  • Weight rewinding is crucial for ticket quality            ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}
