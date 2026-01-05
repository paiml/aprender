//! Pattern mining algorithms for association rule discovery.
//!
//! This module provides algorithms for discovering patterns in transactional data,
//! particularly association rules used in market basket analysis.
//!
//! # Algorithms
//!
//! - [`Apriori`]: Frequent itemset mining and association rule generation
//!
//! # Example
//!
//! ```
//! use aprender::mining::Apriori;
//!
//! // Market basket transactions (each transaction is a set of item IDs)
//! let transactions = vec![
//!     vec![1, 2, 3],    // Transaction 1: items 1, 2, 3
//!     vec![1, 2],       // Transaction 2: items 1, 2
//!     vec![1, 3],       // Transaction 3: items 1, 3
//!     vec![2, 3],       // Transaction 4: items 2, 3
//! ];
//!
//! // Find frequent itemsets with minimum support 0.5 (50%)
//! let mut apriori = Apriori::new()
//!     .with_min_support(0.5)
//!     .with_min_confidence(0.7);
//!
//! apriori.fit(&transactions);
//!
//! // Get association rules
//! let rules = apriori.get_rules();
//! for rule in rules {
//!     println!("{:?} => {:?} (conf={:.2}, lift={:.2})",
//!         rule.antecedent, rule.consequent, rule.confidence, rule.lift);
//! }
//! ```

use std::collections::HashSet;

/// Association rule: antecedent => consequent
#[derive(Debug, Clone, PartialEq)]
pub struct AssociationRule {
    /// Items in the antecedent (left side)
    pub antecedent: Vec<usize>,
    /// Items in the consequent (right side)
    pub consequent: Vec<usize>,
    /// Support: P(antecedent ∪ consequent)
    pub support: f64,
    /// Confidence: P(consequent | antecedent) = support / P(antecedent)
    pub confidence: f64,
    /// Lift: confidence / P(consequent)
    pub lift: f64,
}

/// Apriori algorithm for frequent itemset mining and association rule generation.
///
/// The Apriori algorithm discovers frequent itemsets in transactional data
/// and generates association rules based on support and confidence thresholds.
///
/// # Algorithm
///
/// 1. Find frequent 1-itemsets (support >= `min_support`)
/// 2. Generate candidate k-itemsets from frequent (k-1)-itemsets
/// 3. Prune candidates that don't meet minimum support
/// 4. Repeat until no more frequent itemsets can be generated
/// 5. Generate association rules from frequent itemsets
/// 6. Filter rules by minimum confidence
///
/// # Parameters
///
/// - `min_support`: Minimum support threshold (0.0 to 1.0)
/// - `min_confidence`: Minimum confidence threshold (0.0 to 1.0)
///
/// # Example
///
/// ```
/// use aprender::mining::Apriori;
///
/// let transactions = vec![
///     vec![1, 2, 3],
///     vec![1, 2],
///     vec![1, 3],
///     vec![2, 3],
/// ];
///
/// let mut apriori = Apriori::new()
///     .with_min_support(0.5)
///     .with_min_confidence(0.7);
///
/// apriori.fit(&transactions);
/// let rules = apriori.get_rules();
/// ```
#[derive(Debug, Clone)]
pub struct Apriori {
    min_support: f64,
    min_confidence: f64,
    frequent_itemsets: Vec<(HashSet<usize>, f64)>, // (itemset, support)
    rules: Vec<AssociationRule>,
}

impl Apriori {
    /// Create a new Apriori instance with default parameters.
    ///
    /// # Default Parameters
    ///
    /// - `min_support`: 0.1 (10%)
    /// - `min_confidence`: 0.5 (50%)
    #[must_use] 
    pub fn new() -> Self {
        Self {
            min_support: 0.1,
            min_confidence: 0.5,
            frequent_itemsets: Vec::new(),
            rules: Vec::new(),
        }
    }

    /// Set the minimum support threshold.
    ///
    /// # Arguments
    ///
    /// * `min_support` - Minimum support (0.0 to 1.0)
    #[must_use] 
    pub fn with_min_support(mut self, min_support: f64) -> Self {
        self.min_support = min_support;
        self
    }

    /// Set the minimum confidence threshold.
    ///
    /// # Arguments
    ///
    /// * `min_confidence` - Minimum confidence (0.0 to 1.0)
    #[must_use] 
    pub fn with_min_confidence(mut self, min_confidence: f64) -> Self {
        self.min_confidence = min_confidence;
        self
    }

    /// Find all frequent 1-itemsets.
    fn find_frequent_1_itemsets(&self, transactions: &[Vec<usize>]) -> Vec<(HashSet<usize>, f64)> {
        use std::collections::HashMap;
        let mut item_counts: HashMap<usize, usize> = HashMap::new();

        // Count occurrences of each item
        for transaction in transactions {
            for &item in transaction {
                *item_counts.entry(item).or_insert(0) += 1;
            }
        }

        // Filter by minimum support
        let n_transactions = transactions.len() as f64;
        let mut frequent_1_itemsets = Vec::new();

        for (item, count) in item_counts {
            let support = count as f64 / n_transactions;
            if support >= self.min_support {
                let mut itemset = HashSet::new();
                itemset.insert(item);
                frequent_1_itemsets.push((itemset, support));
            }
        }

        frequent_1_itemsets
    }

    /// Generate candidate k-itemsets from frequent (k-1)-itemsets.
    fn generate_candidates(&self, prev_itemsets: &[(HashSet<usize>, f64)]) -> Vec<HashSet<usize>> {
        let mut candidates = Vec::new();

        // For each pair of (k-1)-itemsets
        for i in 0..prev_itemsets.len() {
            for j in (i + 1)..prev_itemsets.len() {
                let set1 = &prev_itemsets[i].0;
                let set2 = &prev_itemsets[j].0;

                // Join step: combine two (k-1)-itemsets that differ by exactly one item
                let union: HashSet<usize> = set1.union(set2).copied().collect();

                // If union has k items, it's a valid candidate
                if union.len() == set1.len() + 1 {
                    // Prune step: ensure all (k-1)-subsets are frequent
                    if self.has_infrequent_subset(&union, prev_itemsets) {
                        continue;
                    }

                    // Avoid duplicates
                    if !candidates.contains(&union) {
                        candidates.push(union);
                    }
                }
            }
        }

        candidates
    }

    /// Check if an itemset has any infrequent subset.
    #[allow(clippy::unused_self)]
    fn has_infrequent_subset(
        &self,
        itemset: &HashSet<usize>,
        prev_itemsets: &[(HashSet<usize>, f64)],
    ) -> bool {
        // For each (k-1)-subset of itemset
        for &item in itemset {
            let mut subset = itemset.clone();
            subset.remove(&item);

            // Check if this subset is frequent
            let is_frequent = prev_itemsets
                .iter()
                .any(|(freq_set, _)| freq_set == &subset);

            if !is_frequent {
                return true; // Has infrequent subset
            }
        }

        false // All subsets are frequent
    }

    /// Prune candidates by minimum support.
    fn prune_candidates(
        &self,
        candidates: Vec<HashSet<usize>>,
        transactions: &[Vec<usize>],
    ) -> Vec<(HashSet<usize>, f64)> {
        let mut frequent = Vec::new();

        for candidate in candidates {
            let support = Self::calculate_support(&candidate, transactions);
            if support >= self.min_support {
                frequent.push((candidate, support));
            }
        }

        frequent
    }

    /// Generate association rules from frequent itemsets.
    fn generate_rules(&mut self, transactions: &[Vec<usize>]) {
        let mut rules = Vec::new();

        // For each frequent itemset with at least 2 items
        for (itemset, itemset_support) in &self.frequent_itemsets {
            if itemset.len() < 2 {
                continue;
            }

            // Generate all non-empty proper subsets as antecedents
            let items: Vec<usize> = itemset.iter().copied().collect();
            let subsets = self.generate_subsets(&items);

            for antecedent_items in subsets {
                if antecedent_items.is_empty() || antecedent_items.len() == items.len() {
                    continue; // Skip empty and full sets
                }

                // Consequent = itemset \ antecedent
                let antecedent_set: HashSet<usize> = antecedent_items.iter().copied().collect();
                let consequent_set: HashSet<usize> =
                    itemset.difference(&antecedent_set).copied().collect();

                // Calculate confidence = support(itemset) / support(antecedent)
                let antecedent_support = Self::calculate_support(&antecedent_set, transactions);
                let confidence = itemset_support / antecedent_support;

                if confidence >= self.min_confidence {
                    // Calculate lift = confidence / support(consequent)
                    let consequent_support = Self::calculate_support(&consequent_set, transactions);
                    let lift = confidence / consequent_support;

                    let rule = AssociationRule {
                        antecedent: antecedent_items,
                        consequent: consequent_set.into_iter().collect(),
                        support: *itemset_support,
                        confidence,
                        lift,
                    };

                    rules.push(rule);
                }
            }
        }

        self.rules = rules;
    }

    /// Generate all non-empty subsets of items.
    #[allow(clippy::unused_self)]
    fn generate_subsets(&self, items: &[usize]) -> Vec<Vec<usize>> {
        let mut subsets = Vec::new();
        let n = items.len();

        // Generate all 2^n - 1 non-empty subsets (skip 0 and 2^n - 1)
        for mask in 1..(1 << n) {
            let mut subset = Vec::new();
            for (i, &item) in items.iter().enumerate() {
                if (mask & (1 << i)) != 0 {
                    subset.push(item);
                }
            }
            subsets.push(subset);
        }

        subsets
    }

    /// Fit the Apriori algorithm on transaction data.
    ///
    /// # Arguments
    ///
    /// * `transactions` - Vector of transactions, where each transaction is a vector of item IDs
    pub fn fit(&mut self, transactions: &[Vec<usize>]) {
        if transactions.is_empty() {
            self.frequent_itemsets = Vec::new();
            self.rules = Vec::new();
            return;
        }

        self.frequent_itemsets = Vec::new();

        // Step 1: Find frequent 1-itemsets
        let mut current_itemsets = self.find_frequent_1_itemsets(transactions);

        // Step 2: Iteratively generate frequent k-itemsets (k >= 2)
        loop {
            if current_itemsets.is_empty() {
                break;
            }

            // Add current frequent itemsets to results
            self.frequent_itemsets.extend(current_itemsets.clone());

            // Generate candidates for next level
            let candidates = self.generate_candidates(&current_itemsets);
            if candidates.is_empty() {
                break;
            }

            // Prune candidates by support
            current_itemsets = self.prune_candidates(candidates, transactions);
        }

        // Step 3: Generate association rules from frequent itemsets
        self.generate_rules(transactions);

        // Sort frequent itemsets by support descending
        self.frequent_itemsets.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .expect("Support values must be valid f64 (not NaN)")
        });

        // Sort rules by confidence descending
        self.rules.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .expect("Confidence values must be valid f64 (not NaN)")
        });
    }

    /// Get the discovered frequent itemsets.
    ///
    /// Returns a vector of (itemset, support) tuples sorted by support descending.
    #[must_use] 
    pub fn get_frequent_itemsets(&self) -> Vec<(Vec<usize>, f64)> {
        self.frequent_itemsets
            .iter()
            .map(|(itemset, support)| (itemset.iter().copied().collect(), *support))
            .collect()
    }

    /// Get the generated association rules.
    ///
    /// Returns rules sorted by confidence descending.
    #[must_use] 
    pub fn get_rules(&self) -> Vec<AssociationRule> {
        self.rules.clone()
    }

    /// Calculate support for a specific itemset.
    ///
    /// # Arguments
    ///
    /// * `itemset` - The itemset to calculate support for
    /// * `transactions` - Transaction data
    ///
    /// # Returns
    ///
    /// Support value (0.0 to 1.0)
    #[must_use] 
    pub fn calculate_support(itemset: &HashSet<usize>, transactions: &[Vec<usize>]) -> f64 {
        if transactions.is_empty() {
            return 0.0;
        }

        let mut count = 0;

        for transaction in transactions {
            // Check if all items in itemset appear in this transaction
            if itemset.iter().all(|item| transaction.contains(item)) {
                count += 1;
            }
        }

        f64::from(count) / transactions.len() as f64
    }
}

impl Default for Apriori {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apriori_new() {
        let apriori = Apriori::new();
        assert_eq!(apriori.min_support, 0.1);
        assert_eq!(apriori.min_confidence, 0.5);
        assert_eq!(apriori.frequent_itemsets.len(), 0);
        assert_eq!(apriori.rules.len(), 0);
    }

    #[test]
    fn test_apriori_with_min_support() {
        let apriori = Apriori::new().with_min_support(0.3);
        assert_eq!(apriori.min_support, 0.3);
    }

    #[test]
    fn test_apriori_with_min_confidence() {
        let apriori = Apriori::new().with_min_confidence(0.7);
        assert_eq!(apriori.min_confidence, 0.7);
    }

    #[test]
    fn test_apriori_fit_basic() {
        // Simple market basket transactions
        let transactions = vec![
            vec![1, 2, 3],    // Transaction 1: milk, bread, butter
            vec![1, 2],       // Transaction 2: milk, bread
            vec![1, 3],       // Transaction 3: milk, butter
            vec![2, 3],       // Transaction 4: bread, butter
            vec![1, 2, 3, 4], // Transaction 5: milk, bread, butter, eggs
        ];

        let mut apriori = Apriori::new()
            .with_min_support(0.4) // 40% support
            .with_min_confidence(0.6); // 60% confidence

        apriori.fit(&transactions);

        // Should have found frequent itemsets
        assert!(!apriori.frequent_itemsets.is_empty());
    }

    #[test]
    fn test_frequent_itemsets() {
        let transactions = vec![vec![1, 2, 3], vec![1, 2], vec![1, 3], vec![2, 3]];

        let mut apriori = Apriori::new().with_min_support(0.5); // 50% support
        apriori.fit(&transactions);

        let itemsets = apriori.get_frequent_itemsets();

        // With 4 transactions and min_support=0.5, need >= 2 occurrences
        // {1} appears in 3 transactions (75%) - frequent
        // {2} appears in 3 transactions (75%) - frequent
        // {3} appears in 3 transactions (75%) - frequent
        // {1,2} appears in 2 transactions (50%) - frequent
        // {1,3} appears in 2 transactions (50%) - frequent
        // {2,3} appears in 2 transactions (50%) - frequent
        // {1,2,3} appears in 1 transaction (25%) - not frequent

        assert!(itemsets.len() >= 6);

        // Verify itemsets are sorted by support descending
        for i in 1..itemsets.len() {
            assert!(itemsets[i - 1].1 >= itemsets[i].1);
        }
    }

    #[test]
    fn test_association_rules() {
        let transactions = vec![vec![1, 2, 3], vec![1, 2], vec![1, 3], vec![2, 3]];

        let mut apriori = Apriori::new()
            .with_min_support(0.5)
            .with_min_confidence(0.6);

        apriori.fit(&transactions);
        let rules = apriori.get_rules();

        // Should have generated some rules
        assert!(!rules.is_empty());

        // All rules should meet minimum confidence
        for rule in &rules {
            assert!(rule.confidence >= 0.6);
        }

        // Rules should be sorted by confidence descending
        for i in 1..rules.len() {
            assert!(rules[i - 1].confidence >= rules[i].confidence);
        }
    }

    #[test]
    fn test_support_calculation() {
        let transactions = vec![vec![1, 2, 3], vec![1, 2], vec![1, 3], vec![2, 3]];

        // Calculate support for {1, 2}
        let itemset: HashSet<usize> = vec![1, 2].into_iter().collect();
        let support = Apriori::calculate_support(&itemset, &transactions);

        // {1,2} appears in 2 out of 4 transactions = 0.5
        assert!((support - 0.5).abs() < 1e-10);

        // Calculate support for {1}
        let itemset: HashSet<usize> = vec![1].into_iter().collect();
        let support = Apriori::calculate_support(&itemset, &transactions);

        // {1} appears in 3 out of 4 transactions = 0.75
        assert!((support - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_confidence_calculation() {
        let transactions = vec![vec![1, 2, 3], vec![1, 2], vec![1, 3], vec![2, 3]];

        let mut apriori = Apriori::new()
            .with_min_support(0.5)
            .with_min_confidence(0.0); // Accept all rules to verify confidence

        apriori.fit(&transactions);
        let rules = apriori.get_rules();

        // Find rule {1} => {2}
        let rule = rules
            .iter()
            .find(|r| r.antecedent == vec![1] && r.consequent == vec![2])
            .expect("Should have rule {1} => {2}");

        // Confidence({1} => {2}) = P({1,2}) / P({1}) = 0.5 / 0.75 = 0.667
        assert!((rule.confidence - 0.6666666).abs() < 1e-5);
    }

    #[test]
    fn test_lift_calculation() {
        let transactions = vec![vec![1, 2, 3], vec![1, 2], vec![1, 3], vec![2, 3]];

        let mut apriori = Apriori::new()
            .with_min_support(0.5)
            .with_min_confidence(0.0);

        apriori.fit(&transactions);
        let rules = apriori.get_rules();

        // Find rule {1} => {2}
        let rule = rules
            .iter()
            .find(|r| r.antecedent == vec![1] && r.consequent == vec![2])
            .expect("Should have rule {1} => {2}");

        // Lift({1} => {2}) = confidence / P({2}) = 0.667 / 0.75 = 0.889
        assert!((rule.lift - 0.8888888).abs() < 1e-5);

        // Lift > 1.0 means positive correlation
        // Lift < 1.0 means negative correlation
        // Lift = 1.0 means independence
    }

    #[test]
    fn test_min_support_filter() {
        let transactions = vec![
            vec![1, 2],
            vec![1, 2],
            vec![1, 2],
            vec![3, 4], // Infrequent items
        ];

        let mut apriori = Apriori::new().with_min_support(0.5);
        apriori.fit(&transactions);

        let itemsets = apriori.get_frequent_itemsets();

        // Only {1}, {2}, {1,2} should be frequent (75% support each)
        // {3}, {4}, {3,4} are infrequent (25% support)
        for (itemset, support) in itemsets {
            assert!(support >= 0.5, "All itemsets should meet min_support");
            assert!(
                !itemset.contains(&3) && !itemset.contains(&4),
                "Infrequent items should be pruned"
            );
        }
    }

    #[test]
    fn test_min_confidence_filter() {
        let transactions = vec![vec![1, 2, 3], vec![1, 2], vec![1, 3], vec![1]];

        let mut apriori = Apriori::new()
            .with_min_support(0.25)
            .with_min_confidence(0.8); // High confidence threshold

        apriori.fit(&transactions);
        let rules = apriori.get_rules();

        // All rules should meet minimum confidence
        for rule in &rules {
            assert!(
                rule.confidence >= 0.8,
                "Rule {:?} => {:?} has confidence {:.2} < 0.8",
                rule.antecedent,
                rule.consequent,
                rule.confidence
            );
        }
    }

    #[test]
    fn test_empty_transactions() {
        let transactions: Vec<Vec<usize>> = vec![];

        let mut apriori = Apriori::new();
        apriori.fit(&transactions);

        let itemsets = apriori.get_frequent_itemsets();
        assert_eq!(itemsets.len(), 0, "Should have no frequent itemsets");

        let rules = apriori.get_rules();
        assert_eq!(rules.len(), 0, "Should have no rules");
    }

    #[test]
    fn test_single_item_transactions() {
        let transactions = vec![vec![1], vec![2], vec![3], vec![4]];

        let mut apriori = Apriori::new().with_min_support(0.25);
        apriori.fit(&transactions);

        let itemsets = apriori.get_frequent_itemsets();

        // Each item appears once (25% support)
        // Should have 4 frequent 1-itemsets
        assert_eq!(itemsets.len(), 4);

        // No multi-item itemsets possible
        for (itemset, _) in itemsets {
            assert_eq!(itemset.len(), 1);
        }

        let rules = apriori.get_rules();
        // No rules can be generated from single-item itemsets
        assert_eq!(rules.len(), 0);
    }

    #[test]
    fn test_get_rules_before_fit() {
        let apriori = Apriori::new();
        let rules = apriori.get_rules();
        assert_eq!(rules.len(), 0, "Should have no rules before fit");
    }

    #[test]
    fn test_get_itemsets_before_fit() {
        let apriori = Apriori::new();
        let itemsets = apriori.get_frequent_itemsets();
        assert_eq!(itemsets.len(), 0, "Should have no itemsets before fit");
    }
}
