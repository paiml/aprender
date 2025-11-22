# Case Study: Gamma-Poisson Bayesian Inference

This case study demonstrates Bayesian inference for count data using the Gamma-Poisson conjugate family. We cover four practical scenarios: call center analysis, quality control comparison, sequential learning, and prior comparison.

## Overview

The Gamma-Poisson conjugate family is fundamental for Bayesian inference on count data:
- **Prior**: Gamma(α, β) distribution over rate parameter λ > 0
- **Likelihood**: Poisson(λ) for event counts
- **Posterior**: Gamma(α + Σxᵢ, β + n) with closed-form update

This enables exact Bayesian inference for Poisson-distributed data without numerical integration.

## Running the Example

```bash
cargo run --example gamma_poisson_inference
```

Expected output: Four demonstrations showing prior specification, posterior updating, credible intervals, and sequential learning for count data.

## Example 1: Call Center Analysis

### Problem

You manage a call center and want to estimate the hourly call arrival rate. Over a 10-hour period, you observe the following call counts: [3, 5, 4, 6, 2, 4, 5, 3, 4, 4].

What is the expected call rate, and how confident are you in this estimate?

### Solution

```rust
use aprender::bayesian::GammaPoisson;

// Start with noninformative prior Gamma(0.001, 0.001)
let mut model = GammaPoisson::noninformative();
println!("Prior: Gamma({:.3}, {:.3})", model.alpha(), model.beta());
println!("  Prior mean rate: {:.4}", model.posterior_mean());  // ≈ 1.0

// Update with observed hourly call counts
let hourly_calls = vec![3, 5, 4, 6, 2, 4, 5, 3, 4, 4];
model.update(&hourly_calls);

// Posterior is Gamma(0.001 + 40, 0.001 + 10) = Gamma(40.001, 10.001)
println!("Posterior: Gamma({:.3}, {:.3})", model.alpha(), model.beta());
println!("  Posterior mean: {:.4} calls/hour", model.posterior_mean());  // 4.0
```

### Posterior Statistics

```rust
// Point estimates
let mean = model.posterior_mean();  // E[λ|D] = 40.001 / 10.001 ≈ 4.0
let mode = model.posterior_mode().unwrap();  // (40.001 - 1) / 10.001 ≈ 3.9
let variance = model.posterior_variance();  // 40.001 / (10.001)² ≈ 0.40

// 95% credible interval
let (lower, upper) = model.credible_interval(0.95).unwrap();
// ≈ [2.76, 5.24] calls/hour

// Posterior predictive
let predicted_rate = model.posterior_predictive();  // 4.0 calls/hour
```

### Interpretation

**Posterior mean (4.0)**: Our best estimate is that the call center receives 4.0 calls per hour on average.

**Credible interval [2.76, 5.24]**: We are 95% confident that the true call rate is between 2.76 and 5.24 calls per hour. This reflects uncertainty from the limited 10-hour observation period.

**Posterior predictive (4.0)**: The expected number of calls in the next hour is 4.0, integrating over all possible rate values weighted by the posterior.

### Practical Application

**Staffing decisions**: With 95% confidence that the rate is below 5.24 calls/hour, you can plan staffing levels to handle peak loads with high probability.

**Capacity planning**: If each call takes 10 minutes to handle, you need at least one agent available at all times (4 calls/hour × 10 min/call = 40 min/hour).

## Example 2: Quality Control

### Problem

You're evaluating two suppliers for manufacturing components. You need to compare their defect rates:
- **Company A**: 3 defects observed in 20 batches
- **Company B**: 16 defects observed in 20 batches

Which company has a significantly lower defect rate?

### Solution

```rust
// Company A: 3 defects in 20 batches
let company_a_defects = vec![0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0];
let mut model_a = GammaPoisson::noninformative();
model_a.update(&company_a_defects);

let mean_a = model_a.posterior_mean();  // 0.15 defects/batch
let (lower_a, upper_a) = model_a.credible_interval(0.95).unwrap();
// 95% CI: [0.00, 0.32]

// Company B: 16 defects in 20 batches
let company_b_defects = vec![1, 0, 2, 1, 1, 0, 1, 1, 0, 1, 1, 2, 0, 1, 1, 0, 1, 0, 1, 1];
let mut model_b = GammaPoisson::noninformative();
model_b.update(&company_b_defects);

let mean_b = model_b.posterior_mean();  // 0.80 defects/batch
let (lower_b, upper_b) = model_b.credible_interval(0.95).unwrap();
// 95% CI: [0.41, 1.19]
```

### Decision Rule

Check if credible intervals overlap:

```rust
if lower_b > upper_a {
    println!("✓ Company B has significantly higher defect rate (95% confidence)");
    println!("  Company A is the better supplier.");
} else if lower_a > upper_b {
    println!("✓ Company A has significantly higher defect rate (95% confidence)");
    println!("  Company B is the better supplier.");
} else {
    println!("⚠ Credible intervals overlap - no clear difference");
    println!("  Consider testing more batches from each company.");
}
```

### Interpretation

**Output**: "Company B has significantly higher defect rate (95% confidence)"

The credible intervals do NOT overlap: [0.00, 0.32] for A and [0.41, 1.19] for B. Company B's minimum plausible defect rate (0.41) exceeds Company A's maximum plausible rate (0.32), so we can conclusively say Company A is the better supplier.

**Recommendation**: Choose Company A for production. Expected cost savings: If each defect costs $100 to repair, Company A saves approximately (0.80 - 0.15) × $100 = $65 per batch compared to Company B.

### Bayesian vs Frequentist

**Frequentist approach**: Poisson test for rate comparison, get p-value. Interpret significance at α = 0.05 level.

**Bayesian advantage**:
- Direct probability statements: "95% confident A's defect rate is between 0.0 and 0.32 per batch"
- Can incorporate prior knowledge (e.g., historical defect rates from industry)
- Natural stopping rules: test batches until credible intervals separate
- Decision-theoretic framework: minimize expected cost

## Example 3: Sequential Learning

### Problem

Demonstrate how uncertainty decreases as we collect more data from server monitoring (HTTP requests per minute).

### Solution

Run 5 sequential monitoring periods with true rate ≈ 10 requests/min:

```rust
let mut model = GammaPoisson::noninformative();

let experiments = vec![
    vec![8, 12, 10, 11, 9],              // 5 minutes: mean = 10
    vec![9, 11, 10, 12, 8],              // 5 more minutes
    vec![10, 9, 11, 10, 10],             // 5 more minutes
    vec![11, 10, 9, 10, 11, 10, 9],      // 7 more minutes
    vec![10, 11, 10, 9, 10, 11, 10, 10], // 8 more minutes
];

for batch in experiments {
    let batch_u32: Vec<u32> = batch.iter().map(|&x| x).collect();
    model.update(&batch_u32);

    let mean = model.posterior_mean();
    let variance = model.posterior_variance();
    let (lower, upper) = model.credible_interval(0.95).unwrap();
    let width = upper - lower;

    println!("Minutes: {}, Mean: {:.3}, Variance: {:.7}, CI Width: {:.4}",
             total_minutes, mean, variance, width);
}
```

### Results

| Minutes | Total Events | Mean   | Variance   | 95% CI Width |
|---------|--------------|--------|------------|--------------|
| 5       | 50           | 9.998  | 1.9992403  | 5.5427       |
| 10      | 50           | 9.999  | 0.9998102  | 3.9196       |
| 15      | 50           | 9.999  | 0.6665823  | 3.2005       |
| 22      | 70           | 10.000 | 0.4545062  | 2.6427       |
| 30      | 81           | 10.033 | 0.3344233  | 2.2669       |

### Interpretation

**Observation 1**: Posterior mean converges to true value (≈ 10 requests/min)

**Observation 2**: Variance decreases inversely with sample size

For Gamma(α, β): Var[λ] = α / β²

As α increases (from observed events) and β increases (from observation periods), variance decreases approximately as 1/n.

**Observation 3**: Credible interval width shrinks with √n

The 95% CI width drops from 5.54 (n=5) to 2.27 (n=30), reflecting increased certainty about the true rate.

### Practical Application

**Anomaly detection**: If future 5-minute count exceeds upper credible interval (e.g., 15+ requests in 5 min), trigger alert for investigation.

**Capacity planning**: With 95% confidence that rate < 11.5 requests/min (upper bound at n=30), you can provision servers to handle 12 requests/min with high reliability.

## Example 4: Prior Comparison

### Problem

Demonstrate how different priors affect the posterior with limited data.

### Solution

Same data ([3, 5, 4, 6, 2] events over 5 intervals), three different priors:

```rust
// 1. Noninformative Prior Gamma(0.001, 0.001)
let mut noninformative = GammaPoisson::noninformative();
noninformative.update(&counts);
// Posterior: Gamma(20.001, 5.001), mean = 4.00

// 2. Weakly Informative Prior Gamma(1, 1) [mean = 1]
let mut weak = GammaPoisson::new(1.0, 1.0).unwrap();
weak.update(&counts);
// Posterior: Gamma(21, 6), mean = 3.50

// 3. Informative Prior Gamma(50, 10) [mean = 5, strong belief]
let mut informative = GammaPoisson::new(50.0, 10.0).unwrap();
informative.update(&counts);
// Posterior: Gamma(70, 15), mean = 4.67
```

### Results

| Prior Type      | Prior           | Posterior      | Posterior Mean |
|-----------------|-----------------|----------------|----------------|
| Noninformative  | Gamma(0.001, 0.001) | Gamma(20.001, 5.001) | 4.00 |
| Weak            | Gamma(1, 1)     | Gamma(21, 6)   | 3.50           |
| Informative     | Gamma(50, 10)   | Gamma(70, 15)  | 4.67           |

### Interpretation

**Weak priors** (Noninformative, Weak): Posterior dominated by data (mean ≈ 4.0, the empirical mean)

**Strong prior** (Gamma(50, 10)): Posterior pulled toward prior belief (4.67 vs 4.00)

The informative prior Gamma(50, 10) has mean = 50/10 = 5.0 with effective sample size of 10 intervals. With only 5 new observations, the prior still has significant influence, pulling the posterior mean from 4.0 up to 4.67.

### When to Use Strong Priors

**Use informative priors when**:
- You have reliable historical data (e.g., years of defect rate records)
- Expert domain knowledge is available (e.g., typical failure rates for equipment)
- Rare events require regularization (e.g., nuclear accidents, where data is sparse)
- Hierarchical learning across related systems (e.g., defect rates across product lines)

**Avoid informative priors when**:
- No reliable prior knowledge exists
- Prior assumptions may be biased or outdated
- Stakeholders require "data-driven" decisions without prior influence
- Exploring novel systems with no historical analogs

### Prior Sensitivity Analysis

Always check robustness:

1. Run inference with noninformative prior (Gamma(0.001, 0.001))
2. Run inference with weak prior (Gamma(1, 1))
3. Run inference with domain-informed prior (e.g., Gamma(50, 10))
4. If posteriors differ substantially, **collect more data** until they converge

With enough data, all reasonable priors converge to the same posterior (Bayesian consistency).

## Key Takeaways

**1. Conjugate priors enable closed-form updates**
- No MCMC or numerical integration required
- Efficient for real-time sequential updating (e.g., live server monitoring)

**2. Credible intervals quantify uncertainty**
- Direct probability statements about rate parameters
- Width decreases with √n as data accumulates

**3. Sequential updating is natural in Bayesian framework**
- Each posterior becomes the next prior
- Final result is order-independent (commutativity of addition)

**4. Prior choice matters with small data**
- Weak priors: let data speak
- Strong priors: incorporate domain knowledge
- Always perform sensitivity analysis

**5. Bayesian rate comparison avoids p-value pitfalls**
- No arbitrary α = 0.05 threshold
- Natural early stopping rules (wait until credible intervals separate)
- Direct decision-theoretic framework (minimize expected cost)

**6. Gamma-Poisson is ideal for count data**
- Event rates: calls/hour, requests/minute, arrivals/day
- Quality control: defects/batch, failures/unit
- Rare events: accidents, earthquakes, equipment failures

## Related Chapters

- [Bayesian Inference Theory](../ml-fundamentals/bayesian-inference.md)
- [Case Study: Beta-Binomial Bayesian Inference](./beta-binomial-inference.md)

## References

1. **Jaynes, E. T. (2003)**. *Probability Theory: The Logic of Science*. Cambridge University Press. Chapter 6: "Elementary Parameter Estimation."

2. **Gelman, A., et al. (2013)**. *Bayesian Data Analysis* (3rd ed.). CRC Press. Chapter 2: "Single-parameter Models - Poisson Model."

3. **Murphy, K. P. (2012)**. *Machine Learning: A Probabilistic Perspective*. MIT Press. Chapter 3.4: "The Poisson distribution."

4. **Fink, D. (1997)**. "A Compendium of Conjugate Priors." Montana State University. Technical Report. Classic reference for conjugate prior relationships.
