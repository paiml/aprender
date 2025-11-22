//! Market Basket Analysis with Apriori Algorithm
//!
//! This example demonstrates association rule mining for retail transaction data.
//! We analyze shopping baskets to discover which items are frequently bought together.
//!
//! # Use Case
//!
//! A grocery store wants to understand purchasing patterns to:
//! - Optimize product placement
//! - Create cross-selling promotions
//! - Improve inventory management
//!
//! # Run
//!
//! ```bash
//! cargo run --example market_basket_apriori
//! ```

use aprender::mining::Apriori;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Market Basket Analysis with Apriori ===\n");

    // Item ID mapping for readability
    // 1: Milk, 2: Bread, 3: Butter, 4: Eggs, 5: Cheese
    // 6: Yogurt, 7: Coffee, 8: Tea, 9: Sugar, 10: Flour

    // Example 1: Basic Grocery Transactions
    println!("Example 1: Basic Grocery Store Transactions");
    println!("--------------------------------------------");

    let transactions = vec![
        vec![1, 2, 3],       // Milk, Bread, Butter
        vec![1, 2, 4],       // Milk, Bread, Eggs
        vec![1, 2, 3, 4],    // Milk, Bread, Butter, Eggs
        vec![2, 3],          // Bread, Butter
        vec![1, 2],          // Milk, Bread
        vec![1, 3],          // Milk, Butter
        vec![2, 3, 4],       // Bread, Butter, Eggs
        vec![1, 2, 3, 4, 5], // Milk, Bread, Butter, Eggs, Cheese
        vec![1, 2, 5],       // Milk, Bread, Cheese
        vec![2, 4],          // Bread, Eggs
    ];

    println!("Total transactions: {}", transactions.len());
    println!("Transaction samples:");
    for (i, transaction) in transactions.iter().take(3).enumerate() {
        println!("  Transaction {}: {:?}", i + 1, transaction);
    }
    println!();

    // Find frequent itemsets with minimum support 30%
    let mut apriori = Apriori::new()
        .with_min_support(0.3)
        .with_min_confidence(0.6);

    apriori.fit(&transactions);

    let itemsets = apriori.get_frequent_itemsets();
    println!("Frequent itemsets (support >= 30%):");
    for (itemset, support) in &itemsets {
        println!("  {:?} -> support: {:.2}%", itemset, support * 100.0);
    }
    println!();

    // Generate association rules
    let rules = apriori.get_rules();
    println!("Association rules (confidence >= 60%):");
    for rule in &rules {
        println!("  {:?} => {:?}", rule.antecedent, rule.consequent);
        println!("    Support: {:.2}%", rule.support * 100.0);
        println!("    Confidence: {:.2}%", rule.confidence * 100.0);
        println!("    Lift: {:.2}", rule.lift);
        println!();
    }

    // Example 2: Effect of Support Threshold
    println!("\nExample 2: Effect of Support Threshold");
    println!("---------------------------------------");

    let mut apriori_low = Apriori::new()
        .with_min_support(0.2) // Lower threshold
        .with_min_confidence(0.5);

    apriori_low.fit(&transactions);

    let itemsets_low = apriori_low.get_frequent_itemsets();
    println!(
        "With min_support=20%: {} frequent itemsets",
        itemsets_low.len()
    );

    let mut apriori_high = Apriori::new()
        .with_min_support(0.5) // Higher threshold
        .with_min_confidence(0.5);

    apriori_high.fit(&transactions);

    let itemsets_high = apriori_high.get_frequent_itemsets();
    println!(
        "With min_support=50%: {} frequent itemsets",
        itemsets_high.len()
    );
    println!("Higher support => fewer, more reliable patterns");
    println!();

    // Example 3: Breakfast Category Analysis
    println!("\nExample 3: Breakfast Category Analysis");
    println!("---------------------------------------");

    let breakfast_transactions = vec![
        vec![1, 4, 2],    // Milk, Eggs, Bread
        vec![1, 4, 7],    // Milk, Eggs, Coffee
        vec![1, 2, 3],    // Milk, Bread, Butter
        vec![1, 7],       // Milk, Coffee
        vec![4, 2, 3],    // Eggs, Bread, Butter
        vec![1, 4, 2, 7], // Milk, Eggs, Bread, Coffee
        vec![1, 6],       // Milk, Yogurt
        vec![1, 2, 7],    // Milk, Bread, Coffee
        vec![4, 2],       // Eggs, Bread
        vec![1, 4, 6],    // Milk, Eggs, Yogurt
    ];

    let mut breakfast_apriori = Apriori::new()
        .with_min_support(0.3)
        .with_min_confidence(0.7);

    breakfast_apriori.fit(&breakfast_transactions);

    let breakfast_rules = breakfast_apriori.get_rules();
    println!("Strong breakfast associations (conf >= 70%):");
    for rule in breakfast_rules.iter().take(5) {
        println!(
            "  {:?} => {:?} (conf: {:.1}%, lift: {:.2})",
            rule.antecedent,
            rule.consequent,
            rule.confidence * 100.0,
            rule.lift
        );
    }
    println!();

    // Example 4: Interpreting Lift
    println!("\nExample 4: Interpreting Lift Values");
    println!("------------------------------------");

    println!("Lift interpretation:");
    println!("  Lift > 1.0 => Positive correlation (items bought together)");
    println!("  Lift = 1.0 => No correlation (independent)");
    println!("  Lift < 1.0 => Negative correlation (substitutes)");
    println!();

    if let Some(rule) = rules.first() {
        println!(
            "Example rule: {:?} => {:?}",
            rule.antecedent, rule.consequent
        );
        println!("  Lift = {:.2}", rule.lift);
        if rule.lift > 1.0 {
            println!(
                "  Interpretation: Customers who buy {:?} are {:.1}x more likely to buy {:?}",
                rule.antecedent, rule.lift, rule.consequent
            );
        }
    }
    println!();

    // Example 5: Confidence vs Support Trade-off
    println!("\nExample 5: Confidence vs Support Trade-off");
    println!("-------------------------------------------");

    // High confidence, low support
    let mut high_conf = Apriori::new()
        .with_min_support(0.1) // Low support
        .with_min_confidence(0.9); // High confidence

    high_conf.fit(&transactions);
    let high_conf_rules = high_conf.get_rules();

    println!("High confidence (90%), low support (10%):");
    println!("  {} rules found", high_conf_rules.len());
    println!("  Trade-off: Very reliable but rare patterns");
    println!();

    // Balanced parameters
    let mut balanced = Apriori::new()
        .with_min_support(0.3) // Medium support
        .with_min_confidence(0.6); // Medium confidence

    balanced.fit(&transactions);
    let balanced_rules = balanced.get_rules();

    println!("Balanced (support=30%, confidence=60%):");
    println!("  {} rules found", balanced_rules.len());
    println!("  Trade-off: Reasonably reliable and frequent patterns");
    println!();

    // Example 6: Real-World Recommendations
    println!("\nExample 6: Product Placement Recommendations");
    println!("---------------------------------------------");

    println!("Based on discovered rules:");
    println!();

    let top_rules: Vec<_> = rules.iter().take(3).collect();
    for (i, rule) in top_rules.iter().enumerate() {
        println!("Recommendation #{}:", i + 1);
        println!("  Place {:?} near {:?}", rule.consequent, rule.antecedent);
        println!(
            "  Reason: {:.1}% of customers who buy {:?} also buy {:?}",
            rule.confidence * 100.0,
            rule.antecedent,
            rule.consequent
        );
        println!("  Expected uplift: {:.1}%", (rule.lift - 1.0) * 100.0);
        println!();
    }

    // Example 7: Item Frequency Analysis
    println!("\nExample 7: Item Frequency Analysis");
    println!("-----------------------------------");

    let itemsets_1 = apriori.get_frequent_itemsets();
    let single_items: Vec<_> = itemsets_1
        .iter()
        .filter(|(itemset, _)| itemset.len() == 1)
        .collect();

    println!("Most popular individual items:");
    for (itemset, support) in single_items.iter().take(5) {
        println!(
            "  Item {:?}: {:.1}% of transactions",
            itemset,
            support * 100.0
        );
    }
    println!();

    // Example 8: Finding Cross-Selling Opportunities
    println!("\nExample 8: Cross-Selling Opportunities");
    println!("---------------------------------------");

    println!("Best cross-selling pairs (highest lift):");
    let mut sorted_rules = rules.clone();
    sorted_rules.sort_by(|a, b| {
        b.lift
            .partial_cmp(&a.lift)
            .expect("Example data should be valid")
    });

    for rule in sorted_rules.iter().take(3) {
        println!("  If customer buys {:?}:", rule.antecedent);
        println!("    Suggest: {:?}", rule.consequent);
        println!("    Success rate: {:.1}%", rule.confidence * 100.0);
        println!("    Lift: {:.2}x", rule.lift);
        println!();
    }

    println!("=== Analysis Complete ===");

    Ok(())
}
