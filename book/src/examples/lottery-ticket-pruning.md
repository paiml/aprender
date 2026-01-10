# Case Study: Lottery Ticket Pruning

This case study demonstrates finding winning tickets using the Lottery Ticket Hypothesis implementation in Aprender.

## Overview

The Lottery Ticket Hypothesis (Frankle & Carbin, 2018) shows that dense networks contain sparse subnetworks that can train to full accuracy. We'll use Aprender's `LotteryTicketPruner` to find these winning tickets.

## Finding a Winning Ticket

### Basic Example

```rust,ignore
use aprender::pruning::{
    LotteryTicketPruner, LotteryTicketConfig, RewindStrategy, Pruner
};
use aprender::nn::Linear;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a dense layer
    let layer = Linear::new(256, 128);

    // Configure lottery ticket search
    // 90% sparsity over 10 iterative pruning rounds
    let config = LotteryTicketConfig::new(0.9, 10)
        .with_rewind_strategy(RewindStrategy::Init);

    let pruner = LotteryTicketPruner::with_config(config);

    // Find the winning ticket
    let ticket = pruner.find_ticket(&layer)?;

    println!("=== Winning Ticket Found ===");
    println!("Total parameters: {}", ticket.total_parameters);
    println!("Remaining parameters: {}", ticket.remaining_parameters);
    println!("Sparsity: {:.2}%", ticket.sparsity * 100.0);
    println!("Compression ratio: {:.1}x", ticket.compression_ratio());
    println!("Density: {:.2}%", ticket.density() * 100.0);

    Ok(())
}
```

**Output:**
```
=== Winning Ticket Found ===
Total parameters: 32768
Remaining parameters: 3277
Sparsity: 90.00%
Compression ratio: 10.0x
Density: 10.00%
```

## Tracking Pruning Progress

### Observing Iterative Pruning

```rust,ignore
use aprender::pruning::{LotteryTicketPruner, LotteryTicketConfig};
use aprender::nn::Linear;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let layer = Linear::new(100, 100);

    let config = LotteryTicketConfig::new(0.95, 15);
    let pruner = LotteryTicketPruner::with_config(config);

    let ticket = pruner.find_ticket(&layer)?;

    println!("Sparsity progression over {} rounds:", ticket.sparsity_history.len());
    for (round, sparsity) in ticket.sparsity_history.iter().enumerate() {
        let remaining = (1.0 - sparsity) * 100.0;
        println!("  Round {:2}: {:.1}% sparse ({:.1}% remaining)",
            round + 1, sparsity * 100.0, remaining);
    }

    Ok(())
}
```

**Output:**
```
Sparsity progression over 15 rounds:
  Round  1: 18.1% sparse (81.9% remaining)
  Round  2: 32.9% sparse (67.1% remaining)
  Round  3: 45.1% sparse (54.9% remaining)
  Round  4: 55.0% sparse (45.0% remaining)
  Round  5: 63.2% sparse (36.8% remaining)
  Round  6: 69.9% sparse (30.1% remaining)
  Round  7: 75.4% sparse (24.6% remaining)
  Round  8: 79.8% sparse (20.2% remaining)
  Round  9: 83.5% sparse (16.5% remaining)
  Round 10: 86.5% sparse (13.5% remaining)
  Round 11: 88.9% sparse (11.1% remaining)
  Round 12: 90.9% sparse (9.1% remaining)
  Round 13: 92.6% sparse (7.4% remaining)
  Round 14: 93.9% sparse (6.1% remaining)
  Round 15: 95.0% sparse (5.0% remaining)
```

## Using the Builder Pattern

### Configuring All Options

```rust,ignore
use aprender::pruning::{
    LotteryTicketPruner, LotteryTicketPrunerBuilder, RewindStrategy
};
use aprender::nn::Linear;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let layer = Linear::new(512, 256);

    // Builder provides fluent configuration
    let pruner = LotteryTicketPruner::builder()
        .target_sparsity(0.8)           // 80% sparsity target
        .pruning_rounds(5)              // 5 iterative rounds
        .rewind_strategy(RewindStrategy::Early { iteration: 100 })
        .global_pruning(true)           // Prune globally across layers
        .build();

    let ticket = pruner.find_ticket(&layer)?;

    println!("Configuration:");
    println!("  Target sparsity: 80%");
    println!("  Pruning rounds: 5");
    println!("  Rewind strategy: Early (iteration 100)");
    println!("\nResult:");
    println!("  Achieved sparsity: {:.2}%", ticket.sparsity * 100.0);

    Ok(())
}
```

## Comparing Rewind Strategies

### Init vs. Early vs. Late Rewinding

```rust,ignore
use aprender::pruning::{
    LotteryTicketPruner, LotteryTicketConfig, RewindStrategy
};
use aprender::nn::Linear;

fn find_ticket_with_strategy(
    layer: &Linear,
    strategy: RewindStrategy,
    name: &str
) -> Result<(), Box<dyn std::error::Error>> {
    let config = LotteryTicketConfig::new(0.9, 10)
        .with_rewind_strategy(strategy);

    let pruner = LotteryTicketPruner::with_config(config);
    let ticket = pruner.find_ticket(layer)?;

    println!("{} Rewinding:", name);
    println!("  Sparsity: {:.2}%", ticket.sparsity * 100.0);
    println!("  Compression: {:.1}x\n", ticket.compression_ratio());

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let layer = Linear::new(256, 256);

    // Original LTH: rewind to initialization
    find_ticket_with_strategy(&layer, RewindStrategy::Init, "Init")?;

    // Early rewinding: rewind to early training checkpoint
    find_ticket_with_strategy(
        &layer,
        RewindStrategy::Early { iteration: 100 },
        "Early"
    )?;

    // Late rewinding: rewind to fraction of training
    find_ticket_with_strategy(
        &layer,
        RewindStrategy::Late { fraction: 0.1 },
        "Late"
    )?;

    // No rewinding: standard pruning
    find_ticket_with_strategy(&layer, RewindStrategy::None, "None")?;

    Ok(())
}
```

## Applying Winning Tickets

### Pruning Weights with Rewinding

```rust,ignore
use aprender::pruning::{LotteryTicketPruner, LotteryTicketConfig};
use aprender::nn::Linear;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let layer = Linear::new(64, 32);

    let config = LotteryTicketConfig::new(0.75, 5);
    let pruner = LotteryTicketPruner::with_config(config);

    // Find winning ticket
    let ticket = pruner.find_ticket(&layer)?;

    // Apply ticket to get pruned weights with rewinding
    let pruned_weights = pruner.apply_ticket(&ticket, &layer)?;

    // Count zeros in pruned weights
    let zeros = pruned_weights.iter().filter(|&&w| w == 0.0).count();
    let total = pruned_weights.len();
    let actual_sparsity = zeros as f32 / total as f32;

    println!("Applied winning ticket:");
    println!("  Total weights: {}", total);
    println!("  Zero weights: {}", zeros);
    println!("  Actual sparsity: {:.2}%", actual_sparsity * 100.0);

    Ok(())
}
```

## Using the Pruner Trait

### Generic Pruning Interface

```rust,ignore
use aprender::pruning::{
    Pruner, LotteryTicketPruner, LotteryTicketConfig, SparsityPattern
};
use aprender::nn::Linear;

fn prune_with_any_pruner<P: Pruner>(
    pruner: &P,
    module: &dyn aprender::nn::Module,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Pruner: {}", pruner.name());

    // Compute importance scores
    let scores = pruner.importance(module, None)?;
    println!("  Importance range: [{:.4}, {:.4}]",
        scores.stats.min, scores.stats.max);

    // Generate mask at 50% sparsity
    let mask = pruner.generate_mask(
        module,
        SparsityPattern::Unstructured,
        0.5,
        None
    )?;
    println!("  Mask sparsity: {:.2}%", mask.sparsity() * 100.0);

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let layer = Linear::new(128, 64);

    let config = LotteryTicketConfig::new(0.5, 3);
    let pruner = LotteryTicketPruner::with_config(config);

    prune_with_any_pruner(&pruner, &layer)?;

    Ok(())
}
```

**Output:**
```
Pruner: LotteryTicket
  Importance range: [0.0001, 0.9823]
  Mask sparsity: 50.00%
```

## High Sparsity Example

### Finding Extremely Sparse Tickets

```rust,ignore
use aprender::pruning::{LotteryTicketPruner, LotteryTicketConfig};
use aprender::nn::Linear;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let layer = Linear::new(1024, 512);

    // Target 99% sparsity (100x compression)
    let config = LotteryTicketConfig::new(0.99, 20);
    let pruner = LotteryTicketPruner::with_config(config);

    let ticket = pruner.find_ticket(&layer)?;

    println!("=== Extreme Sparsity Winning Ticket ===");
    println!("Original parameters: {}", ticket.total_parameters);
    println!("Remaining parameters: {}", ticket.remaining_parameters);
    println!("Sparsity: {:.2}%", ticket.sparsity * 100.0);
    println!("Compression: {:.0}x", ticket.compression_ratio());
    println!("\nMemory savings:");
    let original_mb = ticket.total_parameters as f32 * 4.0 / 1_000_000.0;
    let pruned_mb = ticket.remaining_parameters as f32 * 4.0 / 1_000_000.0;
    println!("  Original: {:.2} MB", original_mb);
    println!("  Pruned: {:.3} MB", pruned_mb);
    println!("  Saved: {:.2} MB ({:.1}%)",
        original_mb - pruned_mb,
        (1.0 - pruned_mb / original_mb) * 100.0);

    Ok(())
}
```

**Output:**
```
=== Extreme Sparsity Winning Ticket ===
Original parameters: 524288
Remaining parameters: 5243
Sparsity: 99.00%
Compression: 100x

Memory savings:
  Original: 2.10 MB
  Pruned: 0.021 MB
  Saved: 2.08 MB (99.0%)
```

## Key Takeaways

1. **Iterative Pruning** - LTH uses multiple prune-rewind cycles to find sparse subnetworks
2. **Rewind Strategies** - Different rewinding points affect ticket quality
3. **Compression Ratios** - 10-100x compression is achievable
4. **Pruner Trait** - `LotteryTicketPruner` implements the standard `Pruner` interface
5. **Builder Pattern** - Fluent API for configuration

## References

- Frankle, J., & Carbin, M. (2018). "The Lottery Ticket Hypothesis." ICLR 2019.
- Aprender Pruning Module: `src/pruning/lottery.rs`
