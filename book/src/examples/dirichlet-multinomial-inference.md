# Case Study: Dirichlet-Multinomial Bayesian Inference

This case study demonstrates Bayesian inference for categorical data using the Dirichlet-Multinomial conjugate family. We cover four practical scenarios: product preference analysis, survey response comparison, sequential learning, and prior comparison.

## Overview

The Dirichlet-Multinomial conjugate family is fundamental for Bayesian inference on categorical data with k > 2 categories:
- **Prior**: Dirichlet(α₁, ..., αₖ) distribution over probability simplex
- **Likelihood**: Multinomial(θ₁, ..., θₖ) for categorical observations
- **Posterior**: Dirichlet(α₁ + n₁, ..., αₖ + nₖ) with element-wise closed-form update

The probability simplex constraint: Σθᵢ = 1, where each θᵢ ∈ [0, 1] represents the probability of category i.

This enables exact Bayesian inference for multinomial data without numerical integration.

## Running the Example

```bash
cargo run --example dirichlet_multinomial_inference
```

Expected output: Four demonstrations showing prior specification, posterior updating, credible intervals per category, and sequential learning for categorical data.

## Example 1: Customer Product Preference

### Problem

You're conducting market research for smartphones. You survey 120 customers about their brand preference among 4 brands (A, B, C, D). Results: [35, 45, 25, 15].

What is each brand's market share, and which brand is the clear leader?

### Solution

```rust
use aprender::bayesian::DirichletMultinomial;

// Start with uniform prior Dirichlet(1, 1, 1, 1)
// All brands equally likely: 25% each
let mut model = DirichletMultinomial::uniform(4);

// Update with survey responses
let brand_counts = vec![35, 45, 25, 15]; // [A, B, C, D]
model.update(&brand_counts);

// Posterior is Dirichlet(1+35, 1+45, 1+25, 1+15) = Dirichlet(36, 46, 26, 16)
let posterior_probs = model.posterior_mean();
// [0.290, 0.371, 0.210, 0.129] = [29.0%, 37.1%, 21.0%, 12.9%]
```

### Posterior Statistics

```rust
// Point estimates for each category
let means = model.posterior_mean();  // E[θ | D] = (α₁+n₁, ..., αₖ+nₖ) / Σ(αᵢ+nᵢ)
// [0.290, 0.371, 0.210, 0.129]

let modes = model.posterior_mode().unwrap();  // MAP estimates
// [(αᵢ+nᵢ - 1) / (Σαᵢ + Σnᵢ - k)] for all i
// [0.292, 0.375, 0.208, 0.125]

let variances = model.posterior_variance();  // Var[θᵢ | D] for each category
// Individual variances for each brand

// 95% credible intervals (one per category)
let intervals = model.credible_intervals(0.95).unwrap();
// Brand A: [21.1%, 37.0%]
// Brand B: [28.6%, 45.6%]
// Brand C: [13.8%, 28.1%]
// Brand D: [ 7.0%, 18.8%]

// Posterior predictive (next observation probabilities)
let predictive = model.posterior_predictive();  // Same as posterior_mean
```

### Interpretation

**Posterior means**: Brand B leads with 37.1% market share, followed by A (29.0%), C (21.0%), and D (12.9%).

**Credible intervals**: Brand B's interval [28.6%, 45.6%] overlaps with Brand A's [21.1%, 37.0%], so leadership is not statistically conclusive. More data needed.

**Probability simplex constraint**: Note that Σθᵢ = 1.000 exactly (29.0% + 37.1% + 21.0% + 12.9% = 100.0%).

### Practical Application

**Market strategy**:
- Focus advertising budget on Brand B (leader)
- Investigate why Brand D underperforms
- Sample size calculation: Need ~300+ responses for conclusive 95% separation

**Competitive analysis**: If Brand B's lower bound (28.6%) exceeds all other brands' upper bounds, leadership would be statistically significant.

## Example 2: Survey Response Analysis

### Problem

Political survey with 5 candidates. Compare two regions:
- **Region 1 (Urban)**: 300 voters → [85, 70, 65, 50, 30]
- **Region 2 (Rural)**: 200 voters → [30, 45, 60, 40, 25]

Are there significant regional differences in candidate preference?

### Solution

```rust
// Region 1: Urban
let region1_votes = vec![85, 70, 65, 50, 30];
let mut model1 = DirichletMultinomial::uniform(5);
model1.update(&region1_votes);

let probs1 = model1.posterior_mean();
let intervals1 = model1.credible_intervals(0.95).unwrap();
// Candidate 1: 28.2% [23.2%, 33.2%]
// Candidate 2: 23.3% [18.5%, 28.0%]
// Candidate 3: 21.6% [17.0%, 26.3%]
// Candidate 4: 16.7% [12.5%, 20.9%]
// Candidate 5: 10.2% [ 6.8%, 13.6%]

// Region 2: Rural
let region2_votes = vec![30, 45, 60, 40, 25];
let mut model2 = DirichletMultinomial::uniform(5);
model2.update(&region2_votes);

let probs2 = model2.posterior_mean();
let intervals2 = model2.credible_intervals(0.95).unwrap();
// Candidate 1: 15.1% [10.2%, 20.0%]
// Candidate 2: 22.4% [16.7%, 28.1%]
// Candidate 3: 29.8% [23.5%, 36.0%] ← Rural leader
// Candidate 4: 20.0% [14.5%, 25.5%]
// Candidate 5: 12.7% [ 8.1%, 17.2%]
```

### Decision Rules

**Regional difference test**:
```rust
// Check if credible intervals don't overlap
for i in 0..5 {
    if intervals1[i].1 < intervals2[i].0 || intervals2[i].1 < intervals1[i].0 {
        println!("Candidate {} shows significant regional difference", i+1);
    }
}
```

**Leader identification**:
```rust
let leader1 = probs1.iter().enumerate().max_by(...).unwrap().0;  // Candidate 1
let leader2 = probs2.iter().enumerate().max_by(...).unwrap().0;  // Candidate 3
```

### Interpretation

**Regional leaders differ**: Candidate 1 leads urban (28.2%) but Candidate 3 leads rural (29.8%).

**Significant differences**: Candidate 1 shows statistically significant regional difference (28.2% urban vs 15.1% rural), with non-overlapping credible intervals.

**Strategic implications**: Campaign must be region-specific. Candidate 1 should focus on urban centers, while Candidate 3 should campaign in rural areas.

## Example 3: Sequential Learning

### Problem

Text classification system categorizing documents into 5 categories (Tech, Sports, Politics, Entertainment, Business). Demonstrate convergence with streaming data.

### Solution

```rust
let mut model = DirichletMultinomial::uniform(5);

let experiments = vec![
    vec![12, 8, 15, 10, 5],    // Batch 1: 50 documents
    vec![18, 12, 20, 15, 10],  // Batch 2: 75 more documents
    vec![22, 16, 25, 18, 14],  // Batch 3: 95 more documents
    vec![28, 20, 30, 22, 18],  // Batch 4: 118 more documents
    vec![35, 25, 38, 28, 22],  // Batch 5: 148 more documents
];

for batch in experiments {
    model.update(&batch);
    let probs = model.posterior_mean();
    let variances = model.posterior_variance();
    // Print statistics...
}
```

### Results

| Docs | Tech  | Sports | Politics | Entmt | Business | Avg Variance |
|------|-------|--------|----------|-------|----------|--------------|
| 50   | 0.236 | 0.164  | 0.291    | 0.200 | 0.109    | 0.0027887    |
| 125  | 0.238 | 0.162  | 0.277    | 0.200 | 0.123    | 0.0011988    |
| 220  | 0.236 | 0.164  | 0.271    | 0.196 | 0.133    | 0.0006973    |
| 338  | 0.236 | 0.166  | 0.265    | 0.192 | 0.140    | 0.0004591    |
| 486  | 0.236 | 0.167  | 0.263    | 0.191 | 0.143    | 0.0003213    |

### Interpretation

**Convergence**: Probability estimates stabilize after ~200 documents. Changes <1% after n=220.

**Variance reduction**: Average variance decreases from 0.0028 (n=50) to 0.0003 (n=486), reflecting increased confidence.

**Final distribution**: Politics dominates (26.3%), followed by Tech (23.6%), Entertainment (19.1%), Sports (16.7%), and Business (14.3%).

### Practical Application

**Active learning**: Stop collecting labeled data once variance drops below threshold (e.g., 0.001).

**Class imbalance detection**: If true distribution is uniform (20% each), Politics is overrepresented (26.3%) - investigate data source bias.

## Example 4: Prior Comparison

### Problem

Demonstrate how different priors affect posterior inference for website page visit data: [45, 30, 25] visits across 3 pages.

### Solution

```rust
// 1. Uniform Prior Dirichlet(1, 1, 1)
let mut uniform = DirichletMultinomial::uniform(3);
uniform.update(&page_visits);
// Posterior: Dirichlet(46, 31, 26)
// Mean: [0.447, 0.301, 0.252] = [44.7%, 30.1%, 25.2%]

// 2. Weakly Informative Prior Dirichlet(2, 2, 2)
let mut weak = DirichletMultinomial::new(vec![2.0, 2.0, 2.0]).unwrap();
weak.update(&page_visits);
// Posterior: Dirichlet(47, 32, 27)
// Mean: [0.443, 0.302, 0.255] = [44.3%, 30.2%, 25.5%]

// 3. Informative Prior Dirichlet(30, 30, 30) [strong equal belief]
let mut informative = DirichletMultinomial::new(vec![30.0, 30.0, 30.0]).unwrap();
informative.update(&page_visits);
// Posterior: Dirichlet(75, 60, 55)
// Mean: [0.395, 0.316, 0.289] = [39.5%, 31.6%, 28.9%]
```

### Results

| Prior Type      | Prior Dirichlet(α) | Posterior Mean       | Effective N |
|-----------------|-------------------|----------------------|-------------|
| Uniform         | (1, 1, 1)         | (44.7%, 30.1%, 25.2%) | 3           |
| Weak            | (2, 2, 2)         | (44.3%, 30.2%, 25.5%) | 6           |
| Informative     | (30, 30, 30)      | (39.5%, 31.6%, 28.9%) | 90          |

### Interpretation

**Weak priors**: Posterior closely matches data (45%, 30%, 25%).

**Strong prior**: With effective sample size Σαᵢ = 90 vs actual data n = 100, prior significantly influences posterior. Pulls toward equal probabilities (33%, 33%, 33%).

**Prior effective sample size**: Dirichlet(α₁, ..., αₖ) is equivalent to observing αᵢ - 1 counts for category i.

### When to Use Strong Priors

**Use informative priors when**:
- Historical data exists (e.g., long-term website traffic patterns)
- Domain constraints apply (e.g., physics: uniform distribution of particle outcomes)
- Hierarchical models (e.g., learning category distributions across similar classification tasks)
- Regularization needed for sparse categories

**Avoid informative priors when**:
- No reliable prior knowledge
- Exploring new markets/domains
- Prior assumptions may introduce bias
- Data collection is inexpensive (just collect more data instead)

### Prior Sensitivity Analysis

1. Run with uniform prior Dirichlet(1, ..., 1)
2. Run with weak prior Dirichlet(2, ..., 2)
3. Run with domain-informed prior
4. If posteriors diverge, collect more data until convergence

**Convergence criterion**: ||θ̂_uniform - θ̂_informative|| < ε (e.g., ε = 0.05 for 5% tolerance)

## Key Takeaways

**1. k-dimensional conjugate prior for categorical data**
- Operates on probability simplex: Σθᵢ = 1
- Element-wise posterior update: Dirichlet(α + n)
- Generalizes Beta-Binomial to k > 2 categories

**2. Credible intervals for each category**
- Separate interval [θᵢ_lower, θᵢ_upper] for each i
- Can construct joint credible regions (simplexes) for (θ₁, ..., θₖ)
- Useful for detecting statistically significant category differences

**3. Sequential updating is order-independent**
- Batch updates: Dirichlet(α) → Dirichlet(α + Σn_batches)
- Online updates: Update after each observation
- Final posterior is identical regardless of update order

**4. Prior strength affects all categories**
- Effective sample size: Σαᵢ
- Large Σαᵢ = strong prior influence
- With n observations, posterior weight: n/(n + Σαᵢ) on data

**5. Practical applications**
- Market research: product/brand preference
- Natural language: document classification, topic modeling
- User behavior: feature usage, click patterns
- Political polling: multi-candidate elections
- Quality control: defect categorization

**6. Advantages over frequentist methods**
- Direct probability statements for each category
- Natural handling of sparse categories (Bayesian smoothing)
- Coherent framework for sequential testing
- No asymptotic approximations needed (exact inference)

## Related Chapters

- [Bayesian Inference Theory](../ml-fundamentals/bayesian-inference.md)
- [Case Study: Beta-Binomial Bayesian Inference](./beta-binomial-inference.md)
- [Case Study: Gamma-Poisson Bayesian Inference](./gamma-poisson-inference.md)
- [Case Study: Normal-InverseGamma Bayesian Inference](./normal-inverse-gamma-inference.md)

## References

1. **Jaynes, E. T. (2003)**. *Probability Theory: The Logic of Science*. Cambridge University Press. Chapter 18: "The Ap Distribution and Rule of Succession."

2. **Gelman, A., et al. (2013)**. *Bayesian Data Analysis* (3rd ed.). CRC Press. Chapter 5: "Hierarchical Models - Multinomial model."

3. **Murphy, K. P. (2012)**. *Machine Learning: A Probabilistic Perspective*. MIT Press. Chapter 3.5: "The Dirichlet-multinomial model."

4. **Minka, T. (2000)**. "Estimating a Dirichlet distribution." Technical report, MIT. Classic reference for Dirichlet parameter estimation.

5. **Frigyik, B. A., Kapila, A., & Gupta, M. R. (2010)**. "Introduction to the Dirichlet Distribution and Related Processes." UWEE Technical Report. Comprehensive tutorial on Dirichlet mathematics.
