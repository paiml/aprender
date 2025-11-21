# Apriori Algorithm Theory

The Apriori algorithm is a classic data mining technique for discovering frequent itemsets and association rules in transactional databases. It's widely used in market basket analysis, recommendation systems, and pattern discovery.

## Problem Statement

Given a database of transactions, where each transaction contains a set of items:
- Find **frequent itemsets**: sets of items that appear together frequently
- Generate **association rules**: patterns like "if customers buy {A, B}, they likely buy {C}"

## Key Concepts

### 1. Support

Support measures how frequently an itemset appears in the database:

```text
Support(X) = (Transactions containing X) / (Total transactions)
```

Example: If {milk, bread} appears in 60 out of 100 transactions:
```text
Support({milk, bread}) = 60/100 = 0.6 (60%)
```

### 2. Confidence

Confidence measures the reliability of an association rule:

```text
Confidence(X => Y) = Support(X ∪ Y) / Support(X)
```

Example: For rule {milk} => {bread}:
```text
Confidence = P(bread | milk) = Support({milk, bread}) / Support({milk})
```

If 60 transactions have {milk, bread} and 80 have {milk}:
```text
Confidence = 60/80 = 0.75 (75%)
```

### 3. Lift

Lift measures how much more likely items are bought together than independently:

```text
Lift(X => Y) = Confidence(X => Y) / Support(Y)
```

- **Lift > 1.0**: Positive correlation (bought together)
- **Lift = 1.0**: Independent (no relationship)
- **Lift < 1.0**: Negative correlation (substitutes)

Example: For rule {milk} => {bread}:
```text
Lift = 0.75 / 0.70 = 1.07
```
Customers who buy milk are 7% more likely to buy bread than average.

## The Apriori Algorithm

### Core Principle: Apriori Property

**If an itemset is frequent, all of its subsets must also be frequent.**

Contrapositive: **If an itemset is infrequent, all of its supersets must also be infrequent.**

This enables efficient pruning of the search space.

### Algorithm Steps

```text
1. Find all frequent 1-itemsets (individual items)
   - Scan database, count item occurrences
   - Keep items with support >= min_support

2. For k = 2, 3, 4, ...:
   a. Generate candidate k-itemsets from frequent (k-1)-itemsets
      - Join step: Combine (k-1)-itemsets that differ by one item
      - Prune step: Remove candidates with infrequent (k-1)-subsets

   b. Scan database to count candidate support

   c. Keep candidates with support >= min_support

   d. If no frequent k-itemsets found, stop

3. Generate association rules from frequent itemsets:
   - For each frequent itemset I with |I| >= 2:
     - For each non-empty subset A of I:
       - Generate rule A => (I \ A)
       - Keep rules with confidence >= min_confidence
```

### Example Execution

Transactions:
```text
T1: {milk, bread, butter}
T2: {milk, bread}
T3: {bread, butter}
T4: {milk, butter}
```

**Step 1**: Frequent 1-itemsets (min_support = 50%)
```text
{milk}:   3/4 = 75% ✓
{bread}:  3/4 = 75% ✓
{butter}: 3/4 = 75% ✓
```

**Step 2**: Generate candidate 2-itemsets
```text
Candidates: {milk, bread}, {milk, butter}, {bread, butter}
```

**Step 3**: Count support
```text
{milk, bread}:   2/4 = 50% ✓
{milk, butter}:  2/4 = 50% ✓
{bread, butter}: 2/4 = 50% ✓
```

**Step 4**: Generate candidate 3-itemsets
```text
Candidate: {milk, bread, butter}
Support: 1/4 = 25% ✗ (below threshold)
```

**Frequent itemsets**: {milk}, {bread}, {butter}, {milk, bread}, {milk, butter}, {bread, butter}

**Association rules** (min_confidence = 60%):
```text
{milk} => {bread}    Conf: 2/3 = 67% ✓
{bread} => {milk}    Conf: 2/3 = 67% ✓
{milk} => {butter}   Conf: 2/3 = 67% ✓
{butter} => {milk}   Conf: 2/3 = 67% ✓
{bread} => {butter}  Conf: 2/3 = 67% ✓
{butter} => {bread}  Conf: 2/3 = 67% ✓
```

## Complexity Analysis

### Time Complexity

**Worst case**: O(2^n · |D| · |T|)
- n = number of unique items
- |D| = number of transactions
- |T| = average transaction size

**In practice**: Much better due to pruning
- Typical: O(n^k · |D|) where k is max frequent itemset size (usually < 5)

### Space Complexity

**O(n + |F|)**
- n = unique items
- |F| = number of frequent itemsets (exponential worst case, but usually small)

## Parameters

### Minimum Support

**Higher support (e.g., 50%)**:
- Pros: Find common, reliable patterns
- Cons: Miss rare but important associations

**Lower support (e.g., 10%)**:
- Pros: Discover niche patterns
- Cons: Many spurious associations, slower

**Rule of thumb**: Start with 10-30% for exploratory analysis

### Minimum Confidence

**Higher confidence (e.g., 80%)**:
- Pros: High-quality, actionable rules
- Cons: Miss weaker but still meaningful patterns

**Lower confidence (e.g., 50%)**:
- Pros: More exploratory insights
- Cons: Less reliable rules

**Rule of thumb**: 60-70% for actionable business insights

## Strengths

1. **Simplicity**: Easy to understand and implement
2. **Completeness**: Finds all frequent itemsets (no false negatives)
3. **Pruning**: Apriori property enables efficient search
4. **Interpretability**: Rules are human-readable

## Limitations

1. **Multiple database scans**: One scan per itemset size
2. **Candidate generation**: Exponential in worst case
3. **Low support problem**: Misses rare but important patterns
4. **Binary transactions**: Doesn't handle quantities or sequences

## Improvements and Variants

1. **FP-Growth**: Avoids candidate generation using FP-tree (2x-10x faster)
2. **Eclat**: Vertical data format (item-TID lists)
3. **AprioriTID**: Reduces database scans
4. **Weighted Apriori**: Assigns weights to items
5. **Multi-level Apriori**: Handles concept hierarchies (e.g., "dairy" → "milk")

## Applications

### 1. Market Basket Analysis
- Cross-selling: "Customers who bought X also bought Y"
- Product placement: Put related items near each other
- Promotions: Bundle frequently bought items

### 2. Recommendation Systems
- Collaborative filtering: Users who liked X also liked Y
- Content discovery: Articles often read together

### 3. Medical Diagnosis
- Symptom patterns: Patients with X often have Y
- Drug interactions: Medications prescribed together

### 4. Web Mining
- Clickstream analysis: Pages visited together
- Session patterns: User navigation paths

### 5. Bioinformatics
- Gene co-expression: Genes activated together
- Protein interactions: Proteins that interact

## Best Practices

1. **Data preprocessing**:
   - Remove duplicates
   - Filter noise (very rare items)
   - Group similar items (e.g., "2% milk" and "whole milk" → "milk")

2. **Parameter tuning**:
   - Start with balanced parameters (support=20-30%, confidence=60-70%)
   - Increase support if too many rules
   - Lower confidence to explore weak patterns

3. **Rule filtering**:
   - Focus on high lift rules (> 1.2)
   - Remove obvious rules (e.g., "butter => milk" if everyone buys milk)
   - Check rule support (avoid rare but high-confidence spurious rules)

4. **Validation**:
   - Test rules on holdout data
   - A/B test recommendations
   - Monitor business metrics (sales lift, conversion rate)

## Common Pitfalls

1. **Support too low**: Millions of spurious rules
2. **Support too high**: Miss important niche patterns
3. **Ignoring lift**: High confidence ≠ useful (e.g., everyone buys bread)
4. **Confusing correlation with causation**: Apriori finds associations, not causes

## Example Use Case: Grocery Store

**Goal**: Increase basket size through cross-selling

**Data**: 10,000 transactions, 500 unique items

**Parameters**: support=5%, confidence=60%

**Results**:
```text
Rule: {diapers} => {beer}
  Support: 8% (800 transactions)
  Confidence: 75%
  Lift: 2.5
```

**Interpretation**:
- 8% of all transactions contain both diapers and beer
- 75% of diaper buyers also buy beer
- Diaper buyers are 2.5x more likely to buy beer than average

**Action**:
- Place beer near diapers
- Offer "diaper + beer" bundle discount
- Target diaper buyers with beer promotions

**Expected Result**: 10-20% increase in beer sales among diaper buyers

## Mathematical Foundations

### Set Theory

Frequent itemset mining is fundamentally about:
- **Power set**: All 2^n possible itemsets from n items
- **Subset lattice**: Hierarchical structure of itemsets
- **Anti-monotonicity**: Apriori property (subset frequency ≥ superset frequency)

### Probability

Association rules encode conditional probabilities:
- **Support**: P(X)
- **Confidence**: P(Y|X) = P(X ∩ Y) / P(X)
- **Lift**: P(Y|X) / P(Y)

### Information Theory

- **Mutual information**: Measures dependence between itemsets
- **Entropy**: Quantifies uncertainty in item distributions

## Further Reading

1. **Original Apriori Paper**: Agrawal & Srikant (1994) - "Fast Algorithms for Mining Association Rules"
2. **FP-Growth**: Han et al. (2000) - "Mining Frequent Patterns without Candidate Generation"
3. **Market Basket Analysis**: Berry & Linoff (2004) - "Data Mining Techniques"
4. **Advanced Topics**: Tan et al. (2006) - "Introduction to Data Mining"

## Related Topics

- [K-Means Clustering](./kmeans-clustering.md)
- [Decision Trees](./decision-trees.md)
- [Naive Bayes](./naive-bayes.md)
