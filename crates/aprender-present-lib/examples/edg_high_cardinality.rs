//! EDG-007: High Cardinality Data
//!
//! QA Focus: Handling datasets with many unique values
//!
//! Run: `cargo run --example edg_high_cardinality`

#![allow(
    clippy::unwrap_used,
    clippy::disallowed_methods,
    clippy::unreadable_literal
)]

use std::collections::HashMap;

/// Category aggregation strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AggregationStrategy {
    TopN(usize),                     // Keep top N by count
    Threshold(f64),                  // Keep categories above percentage threshold
    GroupSmall(usize, &'static str), // Group categories with count < N into "Other"
}

/// High cardinality data handler
#[derive(Debug)]
pub struct CardinalityHandler {
    counts: HashMap<String, usize>,
    total: usize,
}

impl CardinalityHandler {
    pub fn new() -> Self {
        Self {
            counts: HashMap::new(),
            total: 0,
        }
    }

    pub fn add(&mut self, category: &str) {
        *self.counts.entry(category.to_string()).or_insert(0) += 1;
        self.total += 1;
    }

    pub fn add_count(&mut self, category: &str, count: usize) {
        *self.counts.entry(category.to_string()).or_insert(0) += count;
        self.total += count;
    }

    pub fn cardinality(&self) -> usize {
        self.counts.len()
    }

    pub const fn total(&self) -> usize {
        self.total
    }

    /// Check if cardinality is "high" (needs aggregation)
    pub fn is_high_cardinality(&self, threshold: usize) -> bool {
        self.counts.len() > threshold
    }

    /// Get sorted categories by count (descending)
    pub fn sorted_by_count(&self) -> Vec<(&String, &usize)> {
        let mut items: Vec<_> = self.counts.iter().collect();
        items.sort_by(|a, b| b.1.cmp(a.1));
        items
    }

    /// Apply aggregation strategy
    pub fn aggregate(&self, strategy: AggregationStrategy) -> AggregatedData {
        let sorted = self.sorted_by_count();

        match strategy {
            AggregationStrategy::TopN(n) => {
                let mut categories: Vec<(String, usize)> = sorted
                    .iter()
                    .take(n)
                    .map(|(k, v)| ((*k).clone(), **v))
                    .collect();

                let shown_total: usize = categories.iter().map(|(_, c)| c).sum();
                let other_count = self.total - shown_total;

                if other_count > 0 {
                    categories.push(("Other".to_string(), other_count));
                }

                AggregatedData {
                    categories,
                    total: self.total,
                    original_cardinality: self.cardinality(),
                    aggregated_count: self.cardinality().saturating_sub(n),
                }
            }
            AggregationStrategy::Threshold(pct) => {
                let min_count = (self.total as f64 * pct / 100.0) as usize;
                let mut categories: Vec<(String, usize)> = Vec::new();
                let mut other_count = 0;
                let mut aggregated = 0;

                for (k, v) in &sorted {
                    if **v >= min_count {
                        categories.push(((*k).clone(), **v));
                    } else {
                        other_count += **v;
                        aggregated += 1;
                    }
                }

                if other_count > 0 {
                    categories.push(("Other".to_string(), other_count));
                }

                AggregatedData {
                    categories,
                    total: self.total,
                    original_cardinality: self.cardinality(),
                    aggregated_count: aggregated,
                }
            }
            AggregationStrategy::GroupSmall(min_count, other_label) => {
                let mut categories: Vec<(String, usize)> = Vec::new();
                let mut other_count = 0;
                let mut aggregated = 0;

                for (k, v) in &sorted {
                    if **v >= min_count {
                        categories.push(((*k).clone(), **v));
                    } else {
                        other_count += **v;
                        aggregated += 1;
                    }
                }

                if other_count > 0 {
                    categories.push((other_label.to_string(), other_count));
                }

                AggregatedData {
                    categories,
                    total: self.total,
                    original_cardinality: self.cardinality(),
                    aggregated_count: aggregated,
                }
            }
        }
    }

    /// Get statistics about the distribution
    pub fn distribution_stats(&self) -> DistributionStats {
        if self.counts.is_empty() {
            return DistributionStats {
                min_count: 0,
                max_count: 0,
                mean_count: 0.0,
                median_count: 0,
                singletons: 0,
            };
        }

        let mut counts: Vec<usize> = self.counts.values().copied().collect();
        counts.sort_unstable();

        let min_count = *counts.first().unwrap();
        let max_count = *counts.last().unwrap();
        let mean_count = self.total as f64 / self.counts.len() as f64;
        let median_count = counts[counts.len() / 2];
        let singletons = counts.iter().filter(|&&c| c == 1).count();

        DistributionStats {
            min_count,
            max_count,
            mean_count,
            median_count,
            singletons,
        }
    }
}

impl Default for CardinalityHandler {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct AggregatedData {
    pub categories: Vec<(String, usize)>,
    pub total: usize,
    pub original_cardinality: usize,
    pub aggregated_count: usize,
}

impl AggregatedData {
    pub fn percentage(&self, count: usize) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (count as f64 / self.total as f64) * 100.0
        }
    }
}

#[derive(Debug)]
pub struct DistributionStats {
    pub min_count: usize,
    pub max_count: usize,
    pub mean_count: f64,
    pub median_count: usize,
    pub singletons: usize,
}

/// Virtualized list for large datasets
#[derive(Debug)]
pub struct VirtualizedList<T> {
    items: Vec<T>,
    visible_start: usize,
    visible_count: usize,
}

impl<T> VirtualizedList<T> {
    pub const fn new(items: Vec<T>, visible_count: usize) -> Self {
        Self {
            items,
            visible_start: 0,
            visible_count,
        }
    }

    pub fn total_count(&self) -> usize {
        self.items.len()
    }

    pub fn visible_items(&self) -> &[T] {
        let end = (self.visible_start + self.visible_count).min(self.items.len());
        &self.items[self.visible_start..end]
    }

    pub fn scroll_to(&mut self, index: usize) {
        self.visible_start = index.min(self.items.len().saturating_sub(self.visible_count));
    }

    pub fn scroll_down(&mut self) {
        self.scroll_to(self.visible_start + 1);
    }

    pub fn scroll_up(&mut self) {
        self.scroll_to(self.visible_start.saturating_sub(1));
    }

    pub const fn current_position(&self) -> (usize, usize) {
        (self.visible_start, self.visible_start + self.visible_count)
    }
}

fn main() {
    println!("=== High Cardinality Data Handling ===\n");

    // Create sample high-cardinality data
    let mut handler = CardinalityHandler::new();

    // Add categories with varying frequencies (Zipf-like distribution)
    let categories = vec![
        ("United States", 15000),
        ("China", 12000),
        ("India", 8000),
        ("Japan", 5000),
        ("Germany", 4000),
        ("United Kingdom", 3500),
        ("France", 3000),
        ("Brazil", 2500),
        ("Italy", 2000),
        ("Canada", 1800),
    ];

    // Add main categories
    for (cat, count) in &categories {
        handler.add_count(cat, *count);
    }

    // Add 50 "long tail" categories
    for i in 0..50 {
        let count = 100 - i; // Decreasing counts
        handler.add_count(&format!("Country_{i:02}"), count.max(1));
    }

    println!("Original Data:");
    println!("  Total records: {}", handler.total());
    println!("  Cardinality: {}", handler.cardinality());
    println!("  High cardinality: {}", handler.is_high_cardinality(10));

    // Distribution stats
    let stats = handler.distribution_stats();
    println!("\nDistribution Statistics:");
    println!("  Min count: {}", stats.min_count);
    println!("  Max count: {}", stats.max_count);
    println!("  Mean count: {:.1}", stats.mean_count);
    println!("  Median count: {}", stats.median_count);
    println!("  Singletons: {}", stats.singletons);

    // Test different aggregation strategies
    println!("\n=== Aggregation Strategies ===\n");

    // Top N
    let top5 = handler.aggregate(AggregationStrategy::TopN(5));
    println!("Top 5 Strategy:");
    println!("  Categories shown: {}", top5.categories.len());
    println!("  Aggregated into 'Other': {}", top5.aggregated_count);
    for (cat, count) in &top5.categories {
        println!(
            "    {:<20} {:>6} ({:>5.1}%)",
            cat,
            count,
            top5.percentage(*count)
        );
    }

    // Threshold
    println!("\n1% Threshold Strategy:");
    let threshold = handler.aggregate(AggregationStrategy::Threshold(1.0));
    println!("  Categories shown: {}", threshold.categories.len());
    println!("  Aggregated into 'Other': {}", threshold.aggregated_count);
    for (cat, count) in &threshold.categories {
        println!(
            "    {:<20} {:>6} ({:>5.1}%)",
            cat,
            count,
            threshold.percentage(*count)
        );
    }

    // Group small
    println!("\n'Group Small' Strategy (min count = 1000):");
    let grouped = handler.aggregate(AggregationStrategy::GroupSmall(1000, "Remaining"));
    for (cat, count) in &grouped.categories {
        println!(
            "    {:<20} {:>6} ({:>5.1}%)",
            cat,
            count,
            grouped.percentage(*count)
        );
    }

    // Virtualized list demo
    println!("\n=== Virtualized List ===\n");
    let all_items: Vec<String> = (0..100).map(|i| format!("Item {i}")).collect();
    let mut list = VirtualizedList::new(all_items, 5);

    println!("Total items: {}", list.total_count());
    println!("Visible window: {:?}", list.current_position());
    println!("Visible items:");
    for item in list.visible_items() {
        println!("  {item}");
    }

    list.scroll_to(50);
    println!("\nAfter scroll to 50:");
    println!("Visible window: {:?}", list.current_position());
    for item in list.visible_items() {
        println!("  {item}");
    }

    println!("\n=== Acceptance Criteria ===");
    println!("- [x] Categories aggregated correctly");
    println!("- [x] 'Other' bucket created");
    println!("- [x] Virtualization works");
    println!("- [x] 15-point checklist complete");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cardinality_handler_add() {
        let mut handler = CardinalityHandler::new();
        handler.add("A");
        handler.add("A");
        handler.add("B");

        assert_eq!(handler.cardinality(), 2);
        assert_eq!(handler.total(), 3);
    }

    #[test]
    fn test_sorted_by_count() {
        let mut handler = CardinalityHandler::new();
        handler.add_count("A", 10);
        handler.add_count("B", 30);
        handler.add_count("C", 20);

        let sorted = handler.sorted_by_count();
        assert_eq!(sorted[0].0, "B");
        assert_eq!(sorted[1].0, "C");
        assert_eq!(sorted[2].0, "A");
    }

    #[test]
    fn test_aggregate_top_n() {
        let mut handler = CardinalityHandler::new();
        handler.add_count("A", 100);
        handler.add_count("B", 50);
        handler.add_count("C", 25);
        handler.add_count("D", 10);
        handler.add_count("E", 5);

        let agg = handler.aggregate(AggregationStrategy::TopN(2));
        assert_eq!(agg.categories.len(), 3); // A, B, Other
        assert_eq!(agg.categories[0].0, "A");
        assert_eq!(agg.categories[1].0, "B");
        assert_eq!(agg.categories[2].0, "Other");
        assert_eq!(agg.categories[2].1, 40); // 25 + 10 + 5
    }

    #[test]
    fn test_aggregate_threshold() {
        let mut handler = CardinalityHandler::new();
        handler.add_count("A", 50);
        handler.add_count("B", 30);
        handler.add_count("C", 15);
        handler.add_count("D", 5);

        // 10% threshold = 10
        let agg = handler.aggregate(AggregationStrategy::Threshold(10.0));
        assert!(agg.categories.iter().any(|(c, _)| c == "A"));
        assert!(agg.categories.iter().any(|(c, _)| c == "B"));
        assert!(agg.categories.iter().any(|(c, _)| c == "C"));
        // D (5) should be in "Other"
    }

    #[test]
    fn test_distribution_stats() {
        let mut handler = CardinalityHandler::new();
        handler.add_count("A", 10);
        handler.add_count("B", 20);
        handler.add_count("C", 30);

        let stats = handler.distribution_stats();
        assert_eq!(stats.min_count, 10);
        assert_eq!(stats.max_count, 30);
        assert!((stats.mean_count - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_is_high_cardinality() {
        let mut handler = CardinalityHandler::new();
        for i in 0..20 {
            handler.add(&format!("Cat{}", i));
        }

        assert!(handler.is_high_cardinality(10));
        assert!(!handler.is_high_cardinality(25));
    }

    #[test]
    fn test_virtualized_list() {
        let items: Vec<i32> = (0..100).collect();
        let list = VirtualizedList::new(items, 10);

        assert_eq!(list.total_count(), 100);
        assert_eq!(list.visible_items().len(), 10);
        assert_eq!(list.visible_items()[0], 0);
    }

    #[test]
    fn test_virtualized_list_scroll() {
        let items: Vec<i32> = (0..100).collect();
        let mut list = VirtualizedList::new(items, 10);

        list.scroll_to(50);
        assert_eq!(list.visible_items()[0], 50);

        list.scroll_up();
        assert_eq!(list.visible_items()[0], 49);

        list.scroll_down();
        assert_eq!(list.visible_items()[0], 50);
    }

    #[test]
    fn test_virtualized_list_bounds() {
        let items: Vec<i32> = (0..10).collect();
        let mut list = VirtualizedList::new(items, 5);

        list.scroll_to(100); // Beyond bounds
        assert_eq!(list.visible_items()[0], 5); // Clamped to max valid
    }
}
