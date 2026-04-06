# Phase 1: Extract -- Paper to Canonical Math

## Input

An arXiv paper (or equivalent peer-reviewed source) containing a mathematical
operation used in ML inference or training.

## Process

1. **Identify the governing equation(s).** These are the equations that define
   what the kernel computes. Not the loss function, not the training procedure --
   the forward pass computation.

2. **Identify the domain and codomain.** What goes in, what comes out, what are
   the shapes and types.

3. **Extract proof obligations.** These are the mathematical properties that
   the implementation MUST satisfy. They come in several flavors (see
   [Proof Obligation Taxonomy](./proof-obligation-taxonomy.md)).

4. **Identify numerical stability requirements.** Papers often describe the
   "textbook" formula and then a "numerically stable" variant. Both must be
   documented. The stable variant is what gets implemented; the textbook
   variant is what gets tested against.

5. **Note assumptions and boundary conditions.** Papers assume infinite
   precision. Code doesn't. Document where precision loss occurs and what
   tolerance is acceptable.

## Output

A `MATH.md` file with:
- Paper citation (arXiv URL, authors, year)
- Governing equations in canonical form
- Proof obligation table
- Numerical stability notes
- Boundary conditions

## Example: Softmax

**Paper:** Implicit in many; canonical treatment in Bridle (1990), Goodfellow
et al. (2016) Ch. 6.2.2.

**Governing equation (textbook):**

```
softmax(x)_i = exp(x_i) / sum_j(exp(x_j))
```

**Governing equation (numerically stable):**

```
softmax(x)_i = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))
```

**Proof obligations:**

| ID | Type | Property | Formal |
|----|------|----------|--------|
| SM-INV-001 | Invariant | Output sums to 1 | `|sum(softmax(x)) - 1.0| < e` |
| SM-INV-002 | Invariant | All outputs positive | `softmax(x)_i > 0 for all i` |
| SM-INV-003 | Invariant | Output in (0,1) | `0 < softmax(x)_i < 1 for all i` |
| SM-EQV-001 | Equivalence | Shift invariance | `softmax(x) = softmax(x + c) for all c in R` |
| SM-MON-001 | Monotonicity | Order preservation | `x_i > x_j implies softmax(x)_i > softmax(x)_j` |
| SM-BND-001 | Bound | Argmax dominance | `softmax(x)_{argmax} >= 1/n` |
| SM-LIN-001 | Non-linearity | Not homogeneous | `softmax(ax) != a*softmax(x)` in general |

**Numerical stability:**
- Textbook formula overflows for `x_i > 88.7` (f32) due to `exp(x_i)`.
- Stable variant subtracts `max(x)` first. Largest exponent is `exp(0) = 1`.
- Underflow: for very negative values, `exp(x_i - max(x)) -> 0`. Acceptable --
  these entries are negligible in the softmax output.
