# Bayesian Blocks Histogram

This example demonstrates the Bayesian Blocks optimal histogram binning algorithm, which uses dynamic programming to find optimal change points in data distributions.

## Overview

The Bayesian Blocks algorithm (Scargle et al., 2013) is an adaptive histogram method that automatically determines the optimal number and placement of bins based on the data structure. Unlike fixed-width methods (Sturges, Scott, etc.), it detects change points and adjusts bin widths to match data density.

## Running the Example

```bash
cargo run --example bayesian_blocks_histogram
```

## Key Concepts

### Adaptive Binning

Bayesian Blocks adapts bin placement to data structure:
- **Dense regions**: Narrower bins to capture detail
- **Sparse regions**: Wider bins to avoid overfitting
- **Gaps**: Natural bin boundaries at distribution changes

### Algorithm Features

1. **O(n²) Dynamic Programming**: Finds globally optimal binning
2. **Fitness Function**: Balances bin width uniformity vs. model complexity
3. **Prior Penalty**: Prevents overfitting by penalizing excessive bins
4. **Change Point Detection**: Identifies discontinuities automatically

### When to Use Bayesian Blocks

Use Bayesian Blocks when:
- Data has non-uniform distribution
- Detecting change points is important
- Automatic bin selection is preferred
- Data contains clusters or gaps

Avoid when:
- Dataset is very large (O(n²) complexity)
- Simple fixed-width binning suffices
- Deterministic bin count is required

## Example Output

### Example 1: Uniform Distribution

For uniformly distributed data (1, 2, 3, ..., 20):

```
Bayesian Blocks: 2 bins
Sturges Rule:    6 bins

→ Bayesian Blocks uses fewer bins for uniform data
```

### Example 2: Two Distinct Clusters

For data with two separated clusters:

```
Data: Cluster 1 (1.0-2.0), Cluster 2 (9.0-10.0)
Gap: 2.0 to 9.0

Bayesian Blocks Result:
  Number of bins: 3
  Bin edges: [0.99, 1.05, 5.50, 10.01]

→ Algorithm detected the gap and created separate bins for each cluster!
```

### Example 3: Multiple Density Regions

For data with varying densities:

```
Data: Dense (1.0-2.0), Sparse (5, 7, 9), Dense (15.0-16.0)

Bayesian Blocks Result:
  Number of bins: 6

→ Algorithm adapts bin width to data density
  - Smaller bins in dense regions
  - Larger bins in sparse regions
```

### Example 4: Method Comparison

Comparing Bayesian Blocks with fixed-width methods on clustered data:

```
Method                    # Bins    Adapts to Gap?
----------------------------------------------------
Bayesian Blocks              3      ✓ Yes
Sturges Rule                 5      ✓ Yes
Scott Rule                   2      ✓ Yes
Freedman-Diaconis             2      ✓ Yes
Square Root                  4      ✓ Yes
```

## Implementation Details

### Fitness Function

The algorithm uses a density-based fitness function:

```rust
let density_score = -block_range / block_count.sqrt();
let fitness = previous_best + density_score - ncp_prior;
```

- Prefers blocks with low range relative to count
- Prior penalty (`ncp_prior = 0.5`) prevents overfitting
- Dynamic programming finds globally optimal solution

### Edge Cases

The implementation handles:
- **Single value**: Creates single bin around value
- **All same values**: Creates single bin with margins
- **Small datasets**: Works correctly with n=1, 2, 3
- **Large datasets**: Tested up to 50+ samples

## Algorithm Reference

The Bayesian Blocks algorithm is described in:

> Scargle, J. D., et al. (2013). "Studies in Astronomical Time Series Analysis. VI. Bayesian Block Representations." The Astrophysical Journal, 764(2), 167.

## Related Examples

- [Descriptive Statistics](./descriptive-statistics.md) - Basic statistical analysis
- [K-Means Clustering](./kmeans-clustering.md) - Density-based clustering

## Key Takeaways

1. **Adaptive binning** outperforms fixed-width methods for non-uniform data
2. **Change point detection** happens automatically without manual tuning
3. **O(n²) complexity** limits scalability to moderate datasets
4. **No parameter tuning** required - algorithm selects bins optimally
5. **Interpretability** - bin edges reveal natural data boundaries
