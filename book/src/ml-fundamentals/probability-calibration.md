# Probability Calibration Theory

Calibration ensures that predicted probabilities reflect true likelihoods: when a model predicts 70% confidence, it should be correct 70% of the time.

## Why Calibration Matters

### Miscalibrated Models

```
Prediction: 90% confident it's a cat
Reality: Only 60% of 90%-confident predictions are cats
```

**Consequences:**
- Decision-making based on wrong probabilities
- Risk underestimation in safety-critical systems
- Ensemble weighting fails

### Calibrated Models

```
Prediction: 70% confident it's a cat
Reality: 70% of 70%-confident predictions are cats
```

## Measuring Calibration

### Reliability Diagram

Plot predicted probability vs actual frequency:

```
Accuracy │    ·
         │   ·
         │  ·    Perfect calibration (diagonal)
         │ ·
         │·
         └──────────
           Confidence
```

### Expected Calibration Error (ECE)

```
ECE = Σᵦ (nᵦ/N) · |acc(b) - conf(b)|
```

Where:
- B = number of bins
- nᵦ = samples in bin b
- acc(b) = accuracy in bin b
- conf(b) = mean confidence in bin b

### Maximum Calibration Error (MCE)

```
MCE = max_b |acc(b) - conf(b)|
```

Worst-case miscalibration.

### Brier Score

```
BS = (1/N) Σᵢ (pᵢ - yᵢ)²
```

Combines calibration and refinement.

## Calibration Methods

### Temperature Scaling

Simple and effective post-hoc calibration:

```
p_calibrated = softmax(logits / T)
```

Optimize T on validation set:

```
T* = argmin_T NLL(softmax(logits/T), y_val)
```

Typically T > 1 (softens overconfident predictions).

### Platt Scaling

Logistic regression on model outputs:

```
P(y=1|x) = σ(a · f(x) + b)
```

Learn a, b on validation set.

### Isotonic Regression

Non-parametric calibration:

```
Map predicted probability to calibrated probability
using monotonic (isotonic) function
```

No parametric assumptions, but needs more data.

### Histogram Binning

```
For each confidence bin [a, b):
    calibrated_prob = empirical_accuracy_in_bin
```

Simple but discontinuous.

### Beta Calibration

```
P_calibrated = 1 / (1 + 1/(exp(c)·((1-p)/p)^a·p^(b-a)))
```

Three-parameter model, handles asymmetric errors.

## When Models Miscalibrate

### Overconfidence

Modern neural networks are typically overconfident:

| Model | ECE (before) | ECE (after temp scaling) |
|-------|--------------|--------------------------|
| ResNet-110 | 4.5% | 1.2% |
| DenseNet-40 | 3.8% | 0.9% |

**Causes:**
- Cross-entropy loss encourages extreme predictions
- Batch normalization
- Overparameterization

### Underconfidence

Less common, but occurs with:
- Heavy regularization
- Ensemble disagreement
- Out-of-distribution inputs

## Calibration for Multi-Class

### Per-Class Calibration

```
P(y=k|x) = calibrator_k(f_k(x))
```

Separate calibrator per class.

### Focal Calibration

```
L = -Σᵢ (1-pᵢ)^γ log(pᵢ)
```

Focal loss during training improves calibration.

## Calibration Under Distribution Shift

Challenge: Calibration degrades on OOD data.

### Domain-Aware Calibration

```
T_domain = T_base · domain_adjustment
```

### Ensemble Temperature

```
p = Σₖ wₖ · softmax(logits/Tₖ)
```

## Conformal Prediction

Provide prediction sets with coverage guarantee:

```
C(x) = {y : s(x,y) ≤ τ}
```

Where τ chosen so that:

```
P(y* ∈ C(x)) ≥ 1 - α
```

**Properties:**
- Distribution-free
- Finite-sample guarantee
- No model assumptions

## Selective Prediction

Abstain when uncertain:

```
If max(p) < threshold:
    return "I don't know"
```

Trade-off: coverage vs accuracy on non-abstained predictions.

## References

- Guo, C., et al. (2017). "On Calibration of Modern Neural Networks." ICML.
- Platt, J. (1999). "Probabilistic Outputs for Support Vector Machines."
- Niculescu-Mizil, A., & Caruana, R. (2005). "Predicting Good Probabilities with Supervised Learning." ICML.
- Angelopoulos, A., & Bates, S. (2021). "A Gentle Introduction to Conformal Prediction." arXiv.
