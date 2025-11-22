# Case Study: Beta-Binomial Bayesian Inference

This case study demonstrates Bayesian inference for binary outcomes using conjugate priors. We cover four practical scenarios: coin flip inference, A/B testing, sequential learning, and prior comparison.

## Overview

The Beta-Binomial conjugate family is the foundation of Bayesian inference for binary data:
- **Prior**: Beta(α, β) distribution over probability parameter θ ∈ [0, 1]
- **Likelihood**: Binomial(n, θ) for k successes in n trials
- **Posterior**: Beta(α + k, β + n - k) with closed-form update

This enables exact Bayesian inference without numerical integration.

## Running the Example

```bash
cargo run --example beta_binomial_inference
```

Expected output: Four demonstrations showing prior specification, posterior updating, credible intervals, and sequential learning.

## Example 1: Coin Flip Inference

### Problem

You flip a coin 10 times and observe 7 heads. What is the probability that this coin is fair (θ = 0.5)?

### Solution

```rust
use aprender::bayesian::BetaBinomial;

// Start with uniform prior Beta(1, 1) = complete ignorance
let mut model = BetaBinomial::uniform();
println!("Prior: Beta({}, {})", model.alpha(), model.beta());
println!("  Prior mean: {:.4}", model.posterior_mean());  // 0.5

// Observe 7 heads in 10 flips
model.update(7, 10);

// Posterior is Beta(1+7, 1+3) = Beta(8, 4)
println!("Posterior: Beta({}, {})", model.alpha(), model.beta());
println!("  Posterior mean: {:.4}", model.posterior_mean());  // 0.6667
```

### Posterior Statistics

```rust
// Point estimates
let mean = model.posterior_mean();  // E[θ|D] = 8/12 = 0.6667
let mode = model.posterior_mode().unwrap();  // (8-1)/(12-2) = 0.7
let variance = model.posterior_variance();  // ≈ 0.017

// 95% credible interval
let (lower, upper) = model.credible_interval(0.95).unwrap();
// ≈ [0.41, 0.92] - wide interval due to small sample size

// Posterior predictive
let prob_heads = model.posterior_predictive();  // 0.6667
```

### Interpretation

**Posterior mean (0.667)**: Our best estimate is that the coin has a 66.7% chance of heads.

**Credible interval [0.41, 0.92]**: We are 95% confident that the true probability is between 41% and 92%. This wide interval reflects uncertainty from small sample size.

**Posterior predictive (0.667)**: The probability of heads on the next flip is 66.7%, integrating over all possible values of θ weighted by the posterior.

### Is the coin fair?

The credible interval includes 0.5, so we **cannot rule out** that the coin is fair. With only 10 flips, the data is consistent with a fair coin that happened to land heads 7 times by chance.

## Example 2: A/B Testing

### Problem

You run an A/B test comparing two website variants:
- **Variant A**: 120 conversions out of 1,000 visitors (12% conversion rate)
- **Variant B**: 145 conversions out of 1,000 visitors (14.5% conversion rate)

Is Variant B significantly better, or could the difference be due to chance?

### Solution

```rust
// Variant A: 120 conversions / 1000 visitors
let mut variant_a = BetaBinomial::uniform();
variant_a.update(120, 1000);
let mean_a = variant_a.posterior_mean();  // 0.1208
let (lower_a, upper_a) = variant_a.credible_interval(0.95).unwrap();
// 95% CI: [0.1006, 0.1409]

// Variant B: 145 conversions / 1000 visitors
let mut variant_b = BetaBinomial::uniform();
variant_b.update(145, 1000);
let mean_b = variant_b.posterior_mean();  // 0.1457
let (lower_b, upper_b) = variant_b.credible_interval(0.95).unwrap();
// 95% CI: [0.1239, 0.1675]
```

### Decision Rule

Check if credible intervals overlap:

```rust
if lower_b > upper_a {
    println!("✓ Variant B is significantly better (95% confidence)");
} else if lower_a > upper_b {
    println!("✓ Variant A is significantly better (95% confidence)");
} else {
    println!("⚠ No clear winner yet - credible intervals overlap");
    println!("  Consider collecting more data");
}
```

### Interpretation

**Output**: "No clear winner yet - credible intervals overlap"

The credible intervals overlap: [10.06%, 14.09%] for A and [12.39%, 16.75%] for B. While B appears better (14.57% vs 12.08%), the uncertainty intervals overlap, meaning we cannot conclusively say B is superior.

**Recommendation**: Collect more data to reduce uncertainty and determine if the 2.5 percentage point difference is real or due to sampling variability.

### Bayesian vs Frequentist

**Frequentist approach**: Run a z-test for proportions, get p-value ≈ 0.02. Conclude "significant at α = 0.05 level."

**Bayesian advantage**:
- Direct probability statements: "95% confident B's conversion rate is between 12.4% and 16.8%"
- Can incorporate prior knowledge (e.g., historical conversion rates)
- Natural stopping rules: collect data until credible intervals separate
- No p-value misinterpretation ("p = 0.02" does NOT mean "2% chance hypothesis is true")

## Example 3: Sequential Learning

### Problem

Demonstrate how uncertainty decreases as we collect more data, even with a consistent underlying success rate.

### Solution

Run 5 sequential experiments with true success rate ≈ 77%:

```rust
let mut model = BetaBinomial::uniform();

let experiments = vec![
    (7, 10),   // 70% success
    (15, 20),  // 75% success
    (23, 30),  // 76.7% success
    (31, 40),  // 77.5% success
    (77, 100), // 77% success
];

for (successes, trials) in experiments {
    model.update(successes, trials);

    let mean = model.posterior_mean();
    let variance = model.posterior_variance();
    let (lower, upper) = model.credible_interval(0.95).unwrap();
    let width = upper - lower;

    println!("Trials: {}, Mean: {:.3}, Variance: {:.7}, CI Width: {:.4}",
             total_trials, mean, variance, width);
}
```

### Results

| Trials | Successes | Mean  | Variance  | 95% CI Width |
|--------|-----------|-------|-----------|--------------|
| 10     | 7         | 0.667 | 0.0170940 | 0.5125       |
| 30     | 22        | 0.719 | 0.0061257 | 0.3068       |
| 60     | 45        | 0.742 | 0.0030392 | 0.2161       |
| 100    | 76        | 0.755 | 0.0017964 | 0.1661       |
| 200    | 153       | 0.762 | 0.0008924 | 0.1171       |

### Interpretation

**Observation 1**: Posterior mean converges to true value (0.762 → 0.77)

**Observation 2**: Variance decreases inversely with sample size

For Beta(α, β): Var[θ] = αβ / [(α+β)²(α+β+1)]

As α + β (total count) increases, variance decreases approximately as 1/(α+β).

**Observation 3**: Credible interval width shrinks with √n

The 95% CI width drops from 51% (n=10) to 12% (n=200), reflecting increased certainty.

### Practical Application

**Early Stopping**: If credible intervals separate in A/B test, you can stop early and deploy the winner. No need for fixed sample size planning as in frequentist statistics.

**Sample Size Planning**: Want 95% CI width < 5%? Solve for α + β ≈ 400 (200 trials).

## Example 4: Prior Comparison

### Problem

Demonstrate how different priors affect the posterior with limited data.

### Solution

Same data (7 successes in 10 trials), three different priors:

```rust
// 1. Uniform Prior Beta(1, 1)
let mut uniform = BetaBinomial::uniform();
uniform.update(7, 10);
// Posterior: Beta(8, 4), mean = 0.6667

// 2. Jeffrey's Prior Beta(0.5, 0.5)
let mut jeffreys = BetaBinomial::jeffreys();
jeffreys.update(7, 10);
// Posterior: Beta(7.5, 3.5), mean = 0.6818

// 3. Informative Prior Beta(50, 50) - strong 50% belief
let mut informative = BetaBinomial::new(50.0, 50.0).unwrap();
informative.update(7, 10);
// Posterior: Beta(57, 53), mean = 0.5182
```

### Results

| Prior Type | Prior | Posterior | Posterior Mean |
|------------|-------|-----------|----------------|
| Uniform    | Beta(1, 1) | Beta(8, 4) | 0.6667 |
| Jeffrey's  | Beta(0.5, 0.5) | Beta(7.5, 3.5) | 0.6818 |
| Informative | Beta(50, 50) | Beta(57, 53) | 0.5182 |

### Interpretation

**Weak priors** (Uniform, Jeffrey's): Posterior dominated by data (≈67% mean)

**Strong prior** (Beta(50, 50)): Posterior pulled toward prior belief (51.8% vs 66.7%)

The informative prior Beta(50, 50) encodes a strong belief that θ ≈ 0.5 with effective sample size of 100. With only 10 new observations, the prior dominates, pulling the posterior mean from 0.667 down to 0.518.

### When to Use Strong Priors

**Use informative priors when**:
- You have reliable historical data
- Expert domain knowledge is available
- Rare events require regularization
- Hierarchical learning across related tasks

**Avoid informative priors when**:
- No reliable prior knowledge exists
- Prior assumptions may be wrong
- Stakeholders require "data-driven" decisions
- Exploring novel domains

### Prior Sensitivity Analysis

Always check robustness:

1. Run inference with weak prior (Beta(1, 1))
2. Run inference with strong prior (Beta(50, 50))
3. If posteriors differ substantially, **collect more data** until they converge

With enough data, all reasonable priors converge to the same posterior (Bayesian consistency).

## Key Takeaways

**1. Conjugate priors enable closed-form updates**
- No MCMC or numerical integration required
- Efficient for real-time sequential updating (online learning)

**2. Credible intervals quantify uncertainty**
- Direct probability statements about parameters
- Width decreases with √n as data accumulates

**3. Sequential updating is natural in Bayesian framework**
- Each posterior becomes the next prior
- Final result is order-independent

**4. Prior choice matters with small data**
- Weak priors: let data speak
- Strong priors: incorporate domain knowledge
- Always perform sensitivity analysis

**5. Bayesian A/B testing avoids p-value pitfalls**
- No arbitrary α = 0.05 threshold
- Natural early stopping rules
- Direct decision-theoretic framework

## Related Chapters

- [Bayesian Inference Theory](../ml-fundamentals/bayesian-inference.md)
- [Naive Bayes Theory](../ml-fundamentals/naive-bayes.md)

## References

1. **Jaynes, E. T. (2003)**. *Probability Theory: The Logic of Science*. Cambridge University Press. Chapter 6: "Elementary Parameter Estimation."

2. **Gelman, A., et al. (2013)**. *Bayesian Data Analysis* (3rd ed.). CRC Press. Chapter 2: "Single-parameter Models."

3. **Kruschke, J. K. (2014)**. *Doing Bayesian Data Analysis* (2nd ed.). Academic Press. Chapter 6: "Inferring a Binomial Probability via Exact Mathematical Analysis."

4. **VanderPlas, J. (2014)**. "Frequentism and Bayesianism: A Python-driven Primer." arXiv:1411.5018. Excellent comparison of paradigms with code examples.
