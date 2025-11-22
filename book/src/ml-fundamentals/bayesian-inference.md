# Bayesian Inference Theory

## Overview

Bayesian inference treats probability as an extension of logic under uncertainty, following E.T. Jaynes' "Probability Theory: The Logic of Science." Unlike frequentist statistics, which interprets probability as long-run frequency, Bayesian probability represents degrees of belief updated by evidence.

## Core Principle: Bayes' Theorem

**Bayes' Theorem** is the fundamental equation for updating beliefs:

$$P(\theta | D) = \frac{P(D | \theta) \times P(\theta)}{P(D)}$$

Where:
- **$P(\theta | D)$** = **Posterior**: Updated belief about parameter $\theta$ after observing data $D$
- **$P(D | \theta)$** = **Likelihood**: Probability of observing data $D$ given parameter $\theta$
- **$P(\theta)$** = **Prior**: Initial belief about $\theta$ before seeing data
- **$P(D)$** = **Evidence**: Marginal probability of data (normalization constant)

The posterior is proportional to the likelihood times the prior:

$$P(\theta | D) \propto P(D | \theta) \times P(\theta)$$

## Cox's Theorems: Probability as Logic

E.T. Jaynes showed that **Cox's theorems** prove that any consistent system of reasoning under uncertainty must obey the rules of probability theory. This establishes Bayesian inference as the unique consistent extension of Boolean logic to uncertain propositions.

**Key insights**:
1. Probabilities represent states of knowledge, not physical randomness
2. Prior probabilities encode existing knowledge before observing new data
3. Updating via Bayes' theorem is the only consistent way to learn from evidence

## Conjugate Priors

A **conjugate prior** for a likelihood function is one that produces a posterior distribution in the same family as the prior. This enables closed-form Bayesian updates without numerical integration.

### Beta-Binomial Conjugate Family

For binary outcomes (success/failure):

**Prior**: Beta($\alpha$, $\beta$)

$$p(\theta) = \frac{\theta^{\alpha-1} (1-\theta)^{\beta-1}}{B(\alpha, \beta)}$$

**Likelihood**: Binomial($n$, $\theta$) with $k$ successes

$$p(k | \theta, n) \propto \theta^k (1-\theta)^{n-k}$$

**Posterior**: Beta($\alpha + k$, $\beta + n - k$)

$$p(\theta | k, n) = \text{Beta}(\alpha + k, \beta + n - k)$$

**Interpretation**:
- $\alpha$ = "prior successes + 1"
- $\beta$ = "prior failures + 1"
- $\alpha + \beta$ = "effective sample size" of prior belief (higher = stronger prior)
- After observing data, simply add observed successes to $\alpha$ and failures to $\beta$

### Common Prior Choices

**1. Uniform Prior**: Beta(1, 1)
- Represents complete ignorance
- All probabilities $\theta \in [0, 1]$ are equally likely
- Posterior is dominated by data

**2. Jeffrey's Prior**: Beta(0.5, 0.5)
- Non-informative prior invariant under reparameterization
- Recommended when no prior knowledge exists
- Slightly favors extreme values (0 or 1)

**3. Informative Prior**: Beta($\alpha$, $\beta$) with $\alpha, \beta > 1$
- Encodes domain knowledge from past experience
- Example: Beta(80, 20) = "strong belief in 80% success rate based on 100 trials"
- Requires more data to overcome strong priors

## Posterior Statistics

### Posterior Mean (Expected Value)

For Beta($\alpha$, $\beta$):

$$E[\theta | D] = \frac{\alpha}{\alpha + \beta}$$

This is the **expected value** of the parameter under the posterior distribution.

### Posterior Mode (MAP Estimate)

**Maximum A Posteriori (MAP)** estimate is the most probable value:

For Beta($\alpha$, $\beta$) with $\alpha > 1, \beta > 1$:

$$\text{mode}[\theta | D] = \frac{\alpha - 1}{\alpha + \beta - 2}$$

**Note**: For uniform prior Beta(1, 1), there is no unique mode (flat distribution).

### Posterior Variance (Uncertainty)

For Beta($\alpha$, $\beta$):

$$\text{Var}[\theta | D] = \frac{\alpha \beta}{(\alpha + \beta)^2 (\alpha + \beta + 1)}$$

**Key property**: Variance decreases as $\alpha + \beta$ increases (more data = more certainty).

### Credible Intervals vs Confidence Intervals

**Credible Interval**: Bayesian probability that parameter lies in interval

- 95% credible interval: $P(a \leq \theta \leq b | D) = 0.95$
- **Interpretation**: "There is a 95% probability that $\theta$ is in $[a, b]$ given the data"
- Directly measures uncertainty about parameter

**Confidence Interval** (frequentist): Long-run frequency interpretation

- 95% confidence interval: In repeated sampling, 95% of intervals contain true $\theta$
- **Cannot say**: "95% probability that $\theta$ is in this specific interval"
- Measures sampling variability, not parameter uncertainty

**Why credible intervals are superior**: Bayesian intervals answer the question we actually care about: "What are plausible parameter values given this data?"

## Posterior Predictive Distribution

The **posterior predictive** integrates over all possible parameter values weighted by the posterior:

$$p(\tilde{x} | D) = \int p(\tilde{x} | \theta) \, p(\theta | D) \, d\theta$$

For Beta-Binomial, the posterior predictive probability of success is:

$$p(\text{success} | D) = \frac{\alpha}{\alpha + \beta} = E[\theta | D]$$

This is the expected probability of success on the next trial, accounting for parameter uncertainty.

## Sequential Bayesian Updating

Bayesian inference naturally handles sequential data:

1. Start with prior $P(\theta)$
2. Observe data batch $D_1$, compute posterior $P(\theta | D_1)$
3. Use $P(\theta | D_1)$ as the new prior
4. Observe data batch $D_2$, compute posterior $P(\theta | D_1, D_2)$
5. Repeat indefinitely

**Key insight**: The final posterior is the same regardless of data order (commutativity).

This matches the **PDCA cycle** in the Toyota Production System:
- **Plan**: Specify prior distribution from standardized work
- **Do**: Execute process and collect data (likelihood)
- **Check**: Compute posterior distribution
- **Act**: Update standards (new prior) if needed

## Choosing Priors

### Non-Informative Priors

Use when you have no prior knowledge:
- **Uniform Prior**: Beta(1, 1) for proportions
- **Jeffrey's Prior**: Beta(0.5, 0.5) for invariance
- **Weakly Informative**: Beta(0.1, 0.1) for minimal influence

### Informative Priors

Use when you have domain knowledge:
- **Historical Data**: Estimate $\alpha$, $\beta$ from past experiments
- **Expert Elicitation**: Ask domain experts for mean and certainty
- **Hierarchical Priors**: Learn priors from related tasks

### Prior Sensitivity Analysis

Always check how results change with different priors:
1. Run inference with weak prior (e.g., Beta(1, 1))
2. Run inference with strong prior (e.g., Beta(50, 50))
3. Compare posteriors—if drastically different, collect more data

## Conjugate Families (Summary)

| Likelihood | Prior | Posterior | Use Case |
|------------|-------|-----------|----------|
| Bernoulli/Binomial | Beta | Beta | Binary outcomes (success/fail) |
| Poisson | Gamma | Gamma | Count data (events per interval) |
| Normal (known variance) | Normal | Normal | Continuous data with known noise |
| Normal (unknown variance) | Normal-Inverse-Gamma | Normal-Inverse-Gamma | General continuous data |
| Multinomial | Dirichlet | Dirichlet | Categorical data (k > 2 classes) |

## Bayesian vs Frequentist

| Aspect | Bayesian | Frequentist |
|--------|----------|-------------|
| Probability | Degree of belief | Long-run frequency |
| Parameters | Random variables | Fixed unknowns |
| Inference | Posterior distribution | Point estimate + SE |
| Prior knowledge | Incorporated naturally | Not allowed |
| Uncertainty | Credible intervals | Confidence intervals |
| Sequential learning | Natural | Requires recomputation |
| Small data | Works well | Often unreliable |

## Practical Guidelines

**When to use Bayesian inference**:
- Small datasets where every observation matters
- Sequential decision-making (A/B testing, clinical trials)
- Incorporating prior knowledge or expert opinion
- Need to quantify uncertainty in predictions
- Model comparison via Bayes factors

**Advantages over frequentist**:
- Direct probability statements about parameters
- Natural handling of sequential data
- Automatic regularization through priors
- Principled framework for model selection

**Disadvantages**:
- Computationally intensive for complex models (MCMC required)
- Prior choice can influence results (requires sensitivity analysis)
- Less familiar to many practitioners

## Aprender Implementation

Aprender implements conjugate priors with the following design:

```rust
use aprender::bayesian::BetaBinomial;

// Prior specification
let mut model = BetaBinomial::uniform();  // Beta(1, 1)

// Bayesian update
model.update(successes, trials);

// Posterior statistics
let mean = model.posterior_mean();
let mode = model.posterior_mode().unwrap();
let variance = model.posterior_variance();

// Credible interval
let (lower, upper) = model.credible_interval(0.95).unwrap();

// Predictive distribution
let prob = model.posterior_predictive();
```

See the [Beta-Binomial case study](../examples/beta-binomial-inference.md) for complete examples.

## Further Reading

1. **Jaynes, E. T. (2003)**. *Probability Theory: The Logic of Science*. Cambridge University Press.
   - The foundational text on Bayesian probability as logic

2. **Gelman, A., et al. (2013)**. *Bayesian Data Analysis* (3rd ed.). CRC Press.
   - Comprehensive practical guide to Bayesian methods

3. **McElreath, R. (2020)**. *Statistical Rethinking* (2nd ed.). CRC Press.
   - Intuitive introduction with focus on causal inference

4. **Murphy, K. P. (2022)**. *Probabilistic Machine Learning: An Introduction*. MIT Press.
   - Modern treatment connecting Bayesian methods to ML

## References

1. **Cox, R. T. (1946)**. "Probability, Frequency and Reasonable Expectation." *American Journal of Physics*, 14(1), 1-13.

2. **Jeffreys, H. (1946)**. "An Invariant Form for the Prior Probability in Estimation Problems." *Proceedings of the Royal Society of London A*, 186(1007), 453-461.

3. **Laplace, P.-S. (1814)**. *Essai philosophique sur les probabilités*. Translated as *A Philosophical Essay on Probabilities* (1902).
