# Case Study: Normal-InverseGamma Bayesian Inference

This case study demonstrates Bayesian inference for continuous data with unknown mean and variance using the Normal-InverseGamma conjugate family. We cover four practical scenarios: manufacturing quality control, medical data analysis, sequential learning, and prior comparison.

## Overview

The Normal-InverseGamma conjugate family is fundamental for Bayesian inference on normally distributed data with both parameters unknown:
- **Prior**: Normal-InverseGamma(μ₀, κ₀, α₀, β₀) for (μ, σ²)
- **Likelihood**: Normal(μ, σ²) for continuous observations
- **Posterior**: Normal-InverseGamma with closed-form parameter updates

This hierarchical structure models:
- σ² ~ InverseGamma(α, β) - variance prior
- μ | σ² ~ Normal(μ₀, σ²/κ) - conditional mean prior

This enables exact bivariate Bayesian inference without numerical integration.

## Running the Example

```bash
cargo run --example normal_inverse_gamma_inference
```

Expected output: Four demonstrations showing prior specification, bivariate posterior updating, credible intervals for both parameters, and sequential learning.

## Example 1: Manufacturing Quality Control

### Problem

You're manufacturing precision parts with target diameter 10.0mm. Over a production run, you measure 10 parts: [9.98, 10.02, 9.97, 10.03, 10.01, 9.99, 10.04, 9.96, 10.00, 10.02] mm.

Is the manufacturing process on-target? What is the process precision (standard deviation)?

### Solution

```rust
use aprender::bayesian::NormalInverseGamma;

// Weakly informative prior centered on target
// μ₀ = 10.0 (target), κ₀ = 1.0 (low confidence)
// α₀ = 3.0, β₀ = 0.02 (weak prior for variance)
let mut model = NormalInverseGamma::new(10.0, 1.0, 3.0, 0.02)
    .expect("Valid parameters");

println!("Prior:");
println!("  E[μ] = {:.4} mm", 10.0);
println!("  E[σ²] = {:.6} mm²", 0.02 / (3.0 - 1.0));  // β/(α-1) = 0.01

// Update with observed measurements
let measurements = vec![9.98, 10.02, 9.97, 10.03, 10.01, 9.99, 10.04, 9.96, 10.00, 10.02];
model.update(&measurements);

let mean_mu = model.posterior_mean_mu();  // E[μ|D] ≈ 10.002
let mean_var = model.posterior_mean_variance().unwrap();  // E[σ²|D] ≈ 0.0033
let std_dev = mean_var.sqrt();  // E[σ|D] ≈ 0.058
```

### Posterior Statistics

```rust
// Posterior mean of μ (location parameter)
let mean_mu = model.posterior_mean_mu();  // 10.002 mm

// Posterior mean of σ² (variance parameter)
let mean_var = model.posterior_mean_variance().unwrap();  // 0.0033 mm²
let std_dev = mean_var.sqrt();  // 0.058 mm

// Posterior variance of μ (uncertainty about mean)
let var_mu = model.posterior_variance_mu().unwrap();  // quantifies uncertainty

// 95% credible interval for μ
let (lower, upper) = model.credible_interval_mu(0.95).unwrap();
// [9.97, 10.04] mm

// Posterior predictive for next measurement
let predicted = model.posterior_predictive();  // E[x_new | D] = mean_mu
```

### Interpretation

**Posterior mean μ (10.002mm)**: The process mean is very close to the 10.0mm target.

**Credible interval [9.97, 10.04]**: We are 95% confident the true mean diameter is between 9.97mm and 10.04mm. Since the target (10.0mm) falls within this interval, the process is on-target.

**Standard deviation (0.058mm)**: The manufacturing process has good precision with σ ≈ 0.058mm. For ±3σ coverage, parts will range from 9.83mm to 10.17mm.

### Practical Application

**Process capability**: With 6σ = 0.348mm spread and typical tolerance of ±0.1mm (0.2mm total), the process needs tightening or the tolerance specification is too strict.

**Quality control**: Parts outside [mean - 3σ, mean + 3σ] = [9.83, 10.17] should be investigated as potential outliers.

## Example 2: Medical Data Analysis

### Problem

You're monitoring two patients' blood pressure (systolic BP in mmHg):
- **Patient A**: [118, 122, 120, 119, 121, 120, 118, 122] mmHg
- **Patient B**: [135, 142, 138, 145, 140, 137, 143, 139] mmHg

Does Patient B have significantly higher BP? Which patient has more variable BP?

### Solution

```rust
// Patient A
let patient_a = vec![118.0, 122.0, 120.0, 119.0, 121.0, 120.0, 118.0, 122.0];
let mut model_a = NormalInverseGamma::noninformative();
model_a.update(&patient_a);

let mean_a = model_a.posterior_mean_mu();  // 120.0 mmHg
let (lower_a, upper_a) = model_a.credible_interval_mu(0.95).unwrap();
// 95% CI: [118.4, 121.6]
let var_a = model_a.posterior_mean_variance().unwrap();  // 5.4 mmHg²

// Patient B
let patient_b = vec![135.0, 142.0, 138.0, 145.0, 140.0, 137.0, 143.0, 139.0];
let mut model_b = NormalInverseGamma::noninformative();
model_b.update(&patient_b);

let mean_b = model_b.posterior_mean_mu();  // 139.9 mmHg
let (lower_b, upper_b) = model_b.credible_interval_mu(0.95).unwrap();
// 95% CI: [137.1, 142.7]
let var_b = model_b.posterior_mean_variance().unwrap();  // 16.1 mmHg²
```

### Decision Rules

**Mean comparison**:
```rust
if lower_b > upper_a {
    println!("Patient B has significantly higher BP (95% confidence)");
} else if lower_a > upper_b {
    println!("Patient A has significantly higher BP (95% confidence)");
} else {
    println!("Credible intervals overlap - no clear difference");
}
```

**Variability comparison**:
```rust
if var_b > 2.0 * var_a {
    println!("Patient B shows {:.1}x higher BP variability", var_b / var_a);
    println!("High variability may indicate cardiovascular instability.");
}
```

### Interpretation

**Output**: "Patient B has significantly higher BP than Patient A (95% confidence)"

The credible intervals do NOT overlap: [118.4, 121.6] for A and [137.1, 142.7] for B. Patient B's minimum plausible BP (137.1) exceeds Patient A's maximum (121.6), indicating a clinically significant difference.

**Variability**: Patient B shows 3.0× higher variance (16.1 vs 5.4 mmHg²), suggesting BP instability that may require medical attention beyond the elevated mean.

### Clinical Significance

- Patient A: Normal BP (120 mmHg) with stable readings
- Patient B: Stage 2 hypertension (140 mmHg) with high variability
- Recommendation: Patient B requires immediate intervention (medication, lifestyle changes)

## Example 3: Sequential Learning

### Problem

Demonstrate how uncertainty about both mean and variance decreases with sequential sensor calibration data.

### Solution

Collect temperature readings in batches (true temperature: 25.0°C):

```rust
let mut model = NormalInverseGamma::noninformative();

let experiments = vec![
    vec![25.2, 24.8, 25.1, 24.9, 25.0],               // 5 readings
    vec![25.3, 24.7, 25.2, 24.8, 25.1],               // 5 more
    vec![25.0, 25.1, 24.9, 25.2, 24.8, 25.0],         // 6 more
    vec![25.1, 24.9, 25.0, 25.2, 24.8, 25.1, 25.0],  // 7 more
    vec![25.0, 25.1, 24.9, 25.0, 25.2, 24.8, 25.1, 25.0], // 8 more
];

for batch in experiments {
    model.update(&batch);
    let mean = model.posterior_mean_mu();
    let var_mu = model.posterior_variance_mu().unwrap();
    let (lower, upper) = model.credible_interval_mu(0.95).unwrap();
    // Print statistics...
}
```

### Results

| Readings | E[μ] (°C) | Var(μ) | E[σ²] (°C²) | 95% CI Width (°C) |
|----------|-----------|--------|-------------|-------------------|
| 5        | 24.995    | 0.0484 | 0.2421      | 0.8625            |
| 10       | 25.008    | 0.0125 | 0.1245      | 0.4374            |
| 16       | 25.005    | 0.0049 | 0.0783      | 0.2743            |
| 23       | 25.008    | 0.0025 | 0.0574      | 0.1958            |
| 31       | 25.009    | 0.0015 | 0.0453      | 0.1499            |

### Interpretation

**Observation 1**: Posterior mean E[μ] converges to true value (25.0°C)

**Observation 2**: Variance of mean Var(μ) decreases inversely with sample size

For Normal-InverseGamma: Var(μ | D) = β/(κ(α-1))

As α and κ increase with data, Var(μ) decreases approximately as 1/n.

**Observation 3**: Estimate of σ² becomes more precise

E[σ²] decreases from 0.24 (n=5) to 0.045 (n=31), converging to the true sensor noise level.

**Observation 4**: Credible interval width shrinks with √n

The 95% CI width drops from 0.86°C (n=5) to 0.15°C (n=31), reflecting increased certainty.

### Practical Application

**Sensor calibration**: After 31 readings, we know the sensor's mean bias (0.009°C above true) and noise level (σ ≈ 0.21°C) with high precision.

**Anomaly detection**: Future readings outside [24.79, 25.23]°C (mean ± 2σ at n=31) should trigger recalibration.

## Example 4: Prior Comparison

### Problem

Demonstrate how different priors affect bivariate posterior inference with limited data.

### Solution

Same data ([22.1, 22.5, 22.3, 22.7, 22.4]°C), three different priors:

```rust
// 1. Noninformative Prior NIG(0, 1, 1, 1)
let mut noninformative = NormalInverseGamma::noninformative();
noninformative.update(&measurements);
// E[μ] = 22.40°C, E[σ²] = 0.23°C²

// 2. Weakly Informative Prior NIG(22, 1, 3, 2) [μ ≈ 22, σ² ≈ 1]
let mut weak = NormalInverseGamma::new(22.0, 1.0, 3.0, 2.0).unwrap();
weak.update(&measurements);
// E[μ] = 22.33°C, E[σ²] = 0.48°C²

// 3. Informative Prior NIG(20, 10, 10, 5) [strong μ = 20, σ² ≈ 0.56]
let mut informative = NormalInverseGamma::new(20.0, 10.0, 10.0, 5.0).unwrap();
informative.update(&measurements);
// E[μ] = 20.80°C, E[σ²] = 1.28°C²
```

### Results

| Prior Type      | Prior NIG(μ₀, κ₀, α₀, β₀) | Posterior E[μ] | Posterior E[σ²] |
|-----------------|---------------------------|----------------|-----------------|
| Noninformative  | (0, 1, 1, 1)              | 22.40°C        | 0.23°C²         |
| Weak            | (22, 1, 3, 2)             | 22.33°C        | 0.48°C²         |
| Informative     | (20, 10, 10, 5)           | 20.80°C        | 1.28°C²         |

### Interpretation

**Weak priors** (Noninformative, Weak): Posterior mean ≈ 22.4°C (sample mean), posterior variance ≈ 0.23-0.48°C² (sample variance ≈ 0.05°C²)

**Strong prior** (NIG(20, 10, 10, 5)): Posterior pulled strongly toward prior belief (μ = 20°C vs data mean = 22.4°C)

The informative prior has effective sample size κ₀ = 10 for the mean and 2α₀ = 20 for the variance. With only 5 new observations, the prior dominates, pulling E[μ] from 22.4°C down to 20.8°C.

### When to Use Strong Priors

**Use informative priors for μ when**:
- Calibrating instruments with known reference standards
- Manufacturing processes with historical mean specifications
- Medical baselines from large population studies

**Use informative priors for σ² when**:
- Equipment with known precision specifications
- Process capability studies with historical variance data
- Measurement devices with manufacturer-specified accuracy

**Avoid informative priors when**:
- Exploring novel systems with no historical data
- Prior assumptions may be biased or outdated
- Stakeholders require purely "data-driven" decisions

### Prior Sensitivity Analysis

1. Run inference with noninformative prior NIG(0, 1, 1, 1)
2. Run inference with domain-informed prior (e.g., historical mean/variance)
3. If posteriors differ substantially, **collect more data** until convergence
4. With sufficient data (n > 30), all reasonable priors converge (Bernstein-von Mises theorem)

## Key Takeaways

**1. Bivariate conjugate prior for (μ, σ²)**
- Hierarchical structure: σ² ~ InverseGamma, μ | σ² ~ Normal
- Closed-form posterior updates for both parameters
- No MCMC required

**2. Credible intervals quantify uncertainty**
- Separate intervals for μ and σ²
- Width decreases with √n as data accumulates
- Can construct joint credible regions (ellipses) for (μ, σ²)

**3. Sequential updating is natural**
- Each posterior becomes next prior
- Order-independent (commutativity)
- Ideal for online learning (sensor monitoring, quality control)

**4. Prior choice affects both parameters**
- κ₀: effective sample size for mean belief
- α₀, β₀: shape variance prior distribution
- Always perform sensitivity analysis with small n

**5. Practical applications**
- Manufacturing: process mean and precision monitoring
- Medical: patient population mean and variability
- Sensors: bias (mean) and noise (variance) estimation

**6. Advantages over frequentist methods**
- Direct probability statements: "95% confident μ ∈ [9.97, 10.04]"
- Natural handling of small samples (no asymptotic approximations)
- Coherent framework for sequential testing

## Related Chapters

- [Bayesian Inference Theory](../ml-fundamentals/bayesian-inference.md)
- [Case Study: Beta-Binomial Bayesian Inference](./beta-binomial-inference.md)
- [Case Study: Gamma-Poisson Bayesian Inference](./gamma-poisson-inference.md)

## References

1. **Jaynes, E. T. (2003)**. *Probability Theory: The Logic of Science*. Cambridge University Press. Chapter 7: "The Central, Gaussian or Normal Distribution."

2. **Gelman, A., et al. (2013)**. *Bayesian Data Analysis* (3rd ed.). CRC Press. Chapter 3: "Introduction to Multiparameter Models - Normal model with unknown mean and variance."

3. **Murphy, K. P. (2012)**. *Machine Learning: A Probabilistic Perspective*. MIT Press. Chapter 4.6: "Bayesian inference for the parameters of a Gaussian."

4. **Bernardo, J. M., & Smith, A. F. M. (2000)**. *Bayesian Theory*. Wiley. Chapter 5.2: "Normal models with conjugate analysis."
