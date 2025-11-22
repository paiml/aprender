# Note: Overdispersion and Numerical Stability in Poisson Models

This note covers two distinct issues related to Poisson models: statistical overdispersion and numerical instability in the IRLS algorithm.

---

## Part 1: Statistical Modeling of Overdispersed Count Data

### The Issue: The Poisson Assumption

A common issue in Bayesian models using a Poisson likelihood is the distribution's inherent assumption that the mean of the count data is equal to its variance (E[Y] = Var(Y) = λ) [^1]. This assumption is often violated in real-world datasets, where the observed variance is significantly larger than the mean. This phenomenon is known as **overdispersion** [^2].

### Consequences of Ignoring Overdispersion

Failing to account for overdispersion leads to several problems:
- The model will be a poor fit for the data.
- The model will underestimate the true variability, resulting in posterior distributions for parameters that are artificially narrow [^3].
- This can lead to erroneously high confidence in parameter estimates and an increased rate of false positives (Type I errors) when conducting hypothesis tests or evaluating variable significance [^4].

### The Solution: The Negative Binomial Model

The standard and most effective solution to handle overdispersion is to replace the Poisson likelihood with a **Negative Binomial (NB) likelihood** [^5]. In a Bayesian context, this is often formulated as a **Gamma-Poisson mixture model** [^7]. This approach provides more reliable parameter estimates and more accurate credible intervals for overdispersed count data [^9][^10].

---

## Part 2: Numerical Instability in GLM Poisson IRLS

### The Issue: IRLS Convergence Failure

The Iteratively Reweighted Least Squares (IRLS) algorithm used to fit the Poisson Generalized Linear Model (GLM) can fail to converge. This is a numerical optimization issue, separate from the statistical issue of overdispersion.

The problem arises when the predicted mean value, `mu`, approaches zero during an iteration. For the Poisson GLM with a canonical log-link, `mu = exp(X * beta)`. The calculation of the working response and weights involves division by `mu` (since the variance of a Poisson distribution is `mu`). If `mu` becomes zero or numerically indistinguishable from zero, this leads to division-by-zero errors, resulting in `NaN` or `Inf` values and causing the algorithm to fail.

### Suggested Fix: Epsilon-Smoothing for Numerical Stability

A standard and effective technique to prevent this failure is to add a small constant, or "epsilon" (e.g., `1e-8`), to the predicted mean `mu` within the IRLS loop.

**Proposed Change:**
Modify the calculation of `mu` inside the IRLS algorithm to be:
`let mu = eta.exp() + EPSILON;`

This technique, known as epsilon-smoothing, ensures that `mu` is always positive and non-zero, preventing floating-point errors. The `EPSILON` value should be small enough not to introduce significant bias into the final coefficient estimates but large enough to maintain numerical stability. This simple change can dramatically improve the robustness and convergence rate of the IRLS algorithm for the Poisson family.

---

## Peer-Reviewed Sources

[^1]: Cameron, A. C., & Trivedi, P. K. (2013). *Regression Analysis of Count Data*. Cambridge University Press.
[^2]: Hilbe, J. M. (2011). *Negative Binomial Regression*. Cambridge University Press.
[^3]: Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian Data Analysis, Third Edition*. CRC Press.
[^4]: Gardner, W., Mulvey, E. P., & Shaw, E. C. (1995). Regression analyses of counts and rates: Poisson, overdispersed Poisson, and negative binomial models. *Psychological Bulletin*, 118(3), 392–404.
[^5]: Ver Hoef, J. M., & Boveng, P. L. (2007). Quasi-Poisson vs. negative binomial regression: how should we model overdispersed count data? *Ecology*, 88(11), 2766-2772.
[^6]: Lawless, J. F. (1987). Negative Binomial and Mixed Poisson Regression. *The Canadian Journal of Statistics*, 15(3), 209-225.
[^7]: Greenwood, M., & Yule, G. U. (1920). An Inquiry into the Nature of Frequency Distributions Representative of Multiple Happenings with Particular Reference to the Occurrence of Multiple Attacks of Disease or of Repeated Accidents. *Journal of the Royal Statistical Society*, 83(2), 255-279.
[^8]: Shmueli, G., Minka, T. P., Kadane, J. B., Borle, S., & Boatwright, P. (2005). A useful distribution for fitting discrete data: revival of the Conway-Maxwell-Poisson distribution. *Journal of the Royal Statistical Society: Series C (Applied Statistics)*, 54(1), 127-142.
[^9]: O'Hara, R. B., & Kotze, D. J. (2010). Do not log-transform count data. *Methods in Ecology and Evolution*, 1(2), 118-122.
[^10]: Lindén, A., & Mäntyniemi, S. (2011). Using the negative binomial distribution to model overdispersion in ecological count data. *Ecology*, 92(9), 1414-1421.