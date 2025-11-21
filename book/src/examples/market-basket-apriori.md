# Case Study: Apriori Implementation

This chapter documents the complete EXTREME TDD implementation of aprender's Apriori algorithm for association rule mining from Issue #21.

## Background

**GitHub Issue #21**: Implement Apriori Algorithm for Association Rule Mining

**Requirements:**
- Frequent itemset mining with Apriori algorithm
- Association rule generation
- Support, confidence, and lift metrics
- Configurable min_support and min_confidence thresholds
- Builder pattern for ergonomic API

**Initial State:**
- Tests: 652 passing (after t-SNE implementation)
- No pattern mining module
- Need new `src/mining/mod.rs` module

## Implementation Summary

### RED Phase

Created 15 comprehensive tests covering:
- Constructor and builder pattern (3 tests)
- Basic fitting and frequent itemset discovery
- Association rule generation
- Support calculation (static method)
- Confidence calculation
- Lift calculation
- Minimum support filtering
- Minimum confidence filtering
- Edge cases: empty transactions, single-item transactions
- Error handling: get_rules/get_itemsets before fit

### GREEN Phase

Implemented complete Apriori algorithm (~400 lines):

**Core Components:**

1. **Apriori**: Public API with builder pattern
   - `new()`, `with_min_support()`, `with_min_confidence()`
   - `fit()`, `get_frequent_itemsets()`, `get_rules()`
   - `calculate_support()` - static method

2. **AssociationRule**: Rule representation
   - `antecedent`: Vec<usize> - items on left side
   - `consequent`: Vec<usize> - items on right side
   - `support`: f64 - P(antecedent ∪ consequent)
   - `confidence`: f64 - P(consequent | antecedent)
   - `lift`: f64 - confidence / P(consequent)

3. **Frequent Itemset Mining**:
   - `find_frequent_1_itemsets()`: Initial scan for individual items
   - `generate_candidates()`: Join step (combine k-1 itemsets)
   - `has_infrequent_subset()`: Prune step (Apriori property)
   - `prune_candidates()`: Filter by minimum support

4. **Association Rule Generation**:
   - `generate_rules()`: Extract rules from frequent itemsets
   - `generate_subsets()`: Power set generation for antecedents
   - Confidence and lift calculation

5. **Helper Methods**:
   - `calculate_support()`: Count transactions containing itemset
   - Sorting: itemsets by support, rules by confidence

**Key Algorithm Steps:**

```text
1. Find frequent 1-itemsets (items with support >= min_support)
2. For k = 2, 3, 4, ...:
   a. Generate candidate k-itemsets from (k-1)-itemsets
   b. Prune candidates with infrequent subsets (Apriori property)
   c. Count support in database
   d. Keep itemsets with support >= min_support
   e. If no frequent k-itemsets, stop
3. Generate association rules:
   a. For each frequent itemset with size >= 2
   b. Generate all non-empty proper subsets as antecedents
   c. Calculate confidence = support(itemset) / support(antecedent)
   d. Keep rules with confidence >= min_confidence
   e. Calculate lift = confidence / support(consequent)
4. Sort itemsets by support (descending)
5. Sort rules by confidence (descending)
```

### REFACTOR Phase

- Added Apriori to prelude
- Zero clippy warnings
- Comprehensive documentation with examples
- Example demonstrating 8 real-world scenarios

**Final State:**
- Tests: 652 → 667 (+15)
- Zero warnings
- All quality gates passing

## Algorithm Details

### Time Complexity

**Theoretical worst case**: O(2^n · |D| · |T|)
- n = number of unique items
- |D| = number of transactions
- |T| = average transaction size

**Practical**: O(n^k · |D|) where k is max frequent itemset size
- k typically < 5 in real data
- Apriori pruning dramatically reduces candidates

### Space Complexity

**O(n + |F|)**
- n = unique items (for counting)
- |F| = number of frequent itemsets (usually small)

### Candidate Generation Strategy

**Join step**: Combine two (k-1)-itemsets that differ by exactly one item
```rust,ignore
fn generate_candidates(&self, prev_itemsets: &[(HashSet<usize>, f64)]) -> Vec<HashSet<usize>> {
    let mut candidates = Vec::new();
    for i in 0..prev_itemsets.len() {
        for j in (i + 1)..prev_itemsets.len() {
            let set1 = &prev_itemsets[i].0;
            let set2 = &prev_itemsets[j].0;
            let union: HashSet<usize> = set1.union(set2).copied().collect();

            if union.len() == set1.len() + 1 {
                // Valid k-itemset candidate
                if !self.has_infrequent_subset(&union, prev_itemsets) {
                    candidates.push(union);
                }
            }
        }
    }
    candidates
}
```

**Prune step**: Remove candidates with infrequent (k-1)-subsets
```rust,ignore
fn has_infrequent_subset(&self, itemset: &HashSet<usize>, prev_itemsets: &[(HashSet<usize>, f64)]) -> bool {
    for &item in itemset {
        let mut subset = itemset.clone();
        subset.remove(&item);

        let is_frequent = prev_itemsets.iter().any(|(freq_set, _)| freq_set == &subset);
        if !is_frequent {
            return true; // Prune this candidate
        }
    }
    false
}
```

## Parameters

### Minimum Support

**Default**: 0.1 (10%)

**Effect**:
- **Higher (50%+)**: Finds common, reliable patterns; faster; fewer results
- **Lower (5-10%)**: Discovers niche patterns; slower; more results

**Example**:
```rust,ignore
use aprender::mining::Apriori;
let apriori = Apriori::new().with_min_support(0.3); // 30%
```

### Minimum Confidence

**Default**: 0.5 (50%)

**Effect**:
- **Higher (80%+)**: High-quality, actionable rules; fewer results
- **Lower (30-50%)**: More exploratory insights; more rules

**Example**:
```rust,ignore
use aprender::mining::Apriori;
let apriori = Apriori::new()
    .with_min_support(0.2)
    .with_min_confidence(0.7); // 70%
```

## Example Highlights

The example (`market_basket_apriori.rs`) demonstrates:

1. **Basic grocery transactions** - 10 transactions, 5 items
2. **Support threshold effects** - 20% vs 50%
3. **Breakfast category analysis** - Domain-specific patterns
4. **Lift interpretation** - Positive/negative correlation
5. **Confidence vs support trade-off** - Parameter tuning
6. **Product placement** - Business recommendations
7. **Item frequency analysis** - Popularity rankings
8. **Cross-selling opportunities** - Sorted by lift

Output excerpt:
```text
Frequent itemsets (support >= 30%):
  [2] -> support: 90.00%  (Bread - most popular)
  [1] -> support: 70.00%  (Milk)
  [3] -> support: 60.00%  (Butter)
  [1, 2] -> support: 60.00%  (Milk + Bread)

Association rules (confidence >= 60%):
  [4] => [2]  (Eggs => Bread)
    Support: 50.00%
    Confidence: 100.00%
    Lift: 1.11  (11% uplift)
```

## Key Takeaways

1. **Apriori Property**: Monotonicity enables efficient pruning
2. **Support vs Confidence**: Trade-off between frequency and reliability
3. **Lift > 1.0**: Actual association, not just popularity
4. **Exponential growth**: Itemset count grows with k (but pruning helps)
5. **Interpretable**: Rules are human-readable business insights

## Comparison: Apriori vs FP-Growth

| Feature | Apriori | FP-Growth |
|---------|---------|-----------|
| Data structure | Horizontal (transactions) | Vertical (FP-tree) |
| Database scans | Multiple (k scans for k-itemsets) | Two (build tree, mine) |
| Candidate generation | Yes (explicit) | No (implicit) |
| Memory | O(n + \|F\|) | O(n + tree size) |
| Speed | Moderate | 2-10x faster |
| Implementation | Simple | Complex |

**When to use Apriori:**
- Moderate-size datasets (< 100K transactions)
- Educational/prototyping
- Need simplicity and interpretability
- Many sparse transactions (few items per transaction)

**When to use FP-Growth:**
- Large datasets (> 100K transactions)
- Production systems requiring speed
- Dense transactions (many items per transaction)

## Use Cases

### 1. Retail Market Basket Analysis
```text
Rule: {diapers} => {beer}
  Support: 8% (common enough to act on)
  Confidence: 75% (reliable pattern)
  Lift: 2.5 (strong positive correlation)

Action: Place beer near diapers, bundle promotions
Result: 10-20% sales increase
```

### 2. E-commerce Recommendations
```text
Rule: {laptop} => {laptop bag}
  Support: 12%
  Confidence: 68%
  Lift: 3.2

Action: "Customers who bought this also bought..."
Result: Higher average order value
```

### 3. Medical Diagnosis Support
```text
Rule: {fever, cough} => {flu}
  Support: 15%
  Confidence: 82%
  Lift: 4.1

Action: Suggest flu test when symptoms present
Result: Earlier diagnosis
```

### 4. Web Analytics
```text
Rule: {homepage, product_page} => {cart}
  Support: 6%
  Confidence: 45%
  Lift: 1.8

Action: Optimize product page conversion flow
Result: Increased checkout rate
```

## Testing Strategy

**Unit Tests** (15 implemented):
- Correctness: Algorithm finds all frequent itemsets
- Parameters: Support/confidence thresholds work correctly
- Metrics: Support, confidence, lift calculated correctly
- Edge cases: Empty data, single items, no rules
- Sorting: Results sorted by support/confidence

**Property-Based Tests** (future work):
- Apriori property: All subsets of frequent itemsets are frequent
- Monotonicity: Higher support => fewer itemsets
- Rule count: More itemsets => more rules
- Confidence bounds: All rules meet min_confidence

**Integration Tests**:
- Full pipeline: fit → get_itemsets → get_rules
- Large datasets: 1000+ transactions
- Many items: 100+ unique items

## Technical Challenges Solved

### Challenge 1: Efficient Candidate Generation
**Problem**: Naively combining all (k-1)-itemsets is O(n^k).
**Solution**: Only join itemsets differing by one item, use HashSet for O(1) checks.

### Challenge 2: Apriori Pruning
**Problem**: Need to verify all (k-1)-subsets are frequent.
**Solution**: Store previous frequent itemsets, check each subset in O(k) time.

### Challenge 3: Rule Generation from Itemsets
**Problem**: Generate all non-empty proper subsets as antecedents.
**Solution**: Bit masking to generate power set in O(2^k) where k is itemset size (usually < 5).

```rust,ignore
fn generate_subsets(&self, items: &[usize]) -> Vec<Vec<usize>> {
    let mut subsets = Vec::new();
    let n = items.len();

    for mask in 1..(1 << n) {  // 2^n - 1 subsets
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
```

### Challenge 4: Sorting Heterogeneous Collections
**Problem**: Need to sort itemsets (HashSet) for display.
**Solution**: Convert to Vec, sort descending by support using partial_cmp.

```rust,ignore
self.frequent_itemsets.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
self.rules.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
```

## Performance Optimizations

1. **HashSet for itemsets**: O(1) membership testing
2. **Early termination**: Stop when no frequent k-itemsets found
3. **Prune before database scan**: Remove candidates with infrequent subsets
4. **Single pass per k**: Count all candidates in one database scan

## Related Topics

- [K-Means Clustering](./kmeans-clustering.md)
- [Decision Tree Classifier](./decision-tree-iris.md)
- [Naive Bayes Classifier](./naive-bayes-iris.md)
- [What is EXTREME TDD?](../methodology/what-is-extreme-tdd.md)

## References

1. Agrawal, R., & Srikant, R. (1994). Fast Algorithms for Mining Association Rules. VLDB.
2. Han, J., et al. (2000). Mining Frequent Patterns without Candidate Generation. SIGMOD.
3. Tan, P., et al. (2006). Introduction to Data Mining. Pearson.
4. Berry, M., & Linoff, G. (2004). Data Mining Techniques. Wiley.
