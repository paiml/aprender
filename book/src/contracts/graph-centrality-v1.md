# graph-centrality-v1

**Version:** 1.0.0

Graph centrality measures — node importance in network structures

## References

- Freeman (1978) Centrality in social networks: conceptual clarification
- Brandes (2001) A faster algorithm for betweenness centrality
- Boldi & Vigna (2014) Axioms for centrality

## Equations

### betweenness

$$
C_B(v) = \sum_{s\neq v\neq t} \sigma_{st}(v) / \sigma_{st}
$$

**Domain:** $G = (V, E), \sigma_{st} = shortest path count from s to t$

**Codomain:** $C_B \in [0, (n-1)(n-2)/2]$

**Invariants:**

- $C_B \geq 0 (non-negativity)$
- $C_B(v) = 0 if v is pendant (degree 1) in a tree$
- $Normalized C_B \in [0, 1]$

### closeness

$$
C_C(v) = (n-1) / \sum_{u\neq v} d(v, u)
$$

**Domain:** $G = (V, E), connected, d = shortest path distance$

**Codomain:** $C_C \in (0, 1]$

**Invariants:**

- $C_C > 0 for connected graphs$
- $C_C = 1 for center of star graph$
- $Higher C_C = closer to all other nodes$

### degree

$$
C_D(v) = deg(v) / (n - 1)
$$

**Domain:** $G = (V, E), |V| \geq 2$

**Codomain:** $C_D \in [0, 1]$

**Invariants:**

- $C_D \in [0, 1] (Freeman's normalization)$
- $C_D = 0 for isolated nodes$
- $C_D = 1 for nodes connected to all others$
- $\sum C_D(v) = 2|E| / (n-1) (sum relates to edge count)$

### eigenvector

$$
x_v = (1/\lambda) \sum_{u\in N(v)} x_u, \lambda = largest eigenvalue
$$

**Domain:** $G = (V, E), connected, A = adjacency matrix$

**Codomain:** $x \in \mathbb{R}ⁿ, x_v \geq 0, ||x|| = 1$

**Invariants:**

- $x_v \geq 0 for all v (Perron-Frobenius)$
- $||x||₂ = 1 (unit norm)$
- $Ax = \lambda x (eigenvector equation)$

### harmonic

$$
C_H(v) = (1/(n-1)) \sum_{u\neq v} 1/d(v,u)
$$

**Domain:** $G = (V, E), d = shortest path distance (∞ if disconnected)$

**Codomain:** $C_H \in [0, 1]$

**Invariants:**

- $C_H \in [0, 1]$
- $C_H = 0 for completely isolated node$
- $C_H handles disconnected graphs (1/∞ = 0)$
- $C_H \geq C_C for connected graphs (AM-HM inequality)$

### katz

$$
x_v = \alpha \sum_{u\in N(v)} x_u + \beta
$$

**Domain:** $G = (V, E), \alpha < 1/\lambda_max, \beta > 0$

**Codomain:** $x \in \mathbb{R}ⁿ, x_v > 0$

**Invariants:**

- $x_v > 0 for all v (strictly positive from \beta)$
- $Converges when \alpha < 1/\lambda_max$
- $Reduces to eigenvector centrality as \beta \to 0$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | bound | Degree centrality bounded | $C_D(v) \in [0, 1] for all v$ |
| 2 | bound | Betweenness non-negative | $C_B(v) \geq 0 for all v$ |
| 3 | bound | Closeness positive for connected | $C_C(v) > 0 for all v in connected graph$ |
| 4 | invariant | Eigenvector non-negativity | $x_v \geq 0 for all v (Perron-Frobenius)$ |
| 5 | bound | Katz strictly positive | $x_v > 0 for all v when \beta > 0$ |
| 6 | bound | Harmonic centrality bounded | $C_H(v) \in [0, 1] for all v$ |
| 7 | invariant | Star graph maximum | $degree centrality of center of star K_{1,n-1} = 1$ |
| 8 | invariant | Complete graph symmetry | $All centralities equal for complete graph K_n$ |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-GC-001 | Degree centrality bounded | C_D ∈ [0, 1] for random graphs | Normalization by (n-1) missing |
| FALSIFY-GC-002 | Betweenness non-negative | C_B ≥ 0 for all nodes in random graphs | Path counting error in Brandes algorithm |
| FALSIFY-GC-003 | Eigenvector non-negativity | x_v ≥ 0 for all v after convergence | Power iteration converging to wrong eigenvector |
| FALSIFY-GC-004 | Complete graph symmetry | All nodes have equal centrality in K_n | Algorithm not converging for symmetric inputs |
| FALSIFY-GC-005 | Star graph degree | Center of star has degree centrality = 1 | Freeman normalization by (n-1) incorrect |
| FALSIFY-GC-006 | Harmonic bounded | C_H ∈ [0, 1] for all graphs | Disconnected component handling (1/∞) |
| FALSIFY-GC-007 | Closeness positive for connected | C_C > 0 for all nodes in connected graphs | Distance sum overflow or zero-division in closeness formula |
| FALSIFY-GC-008 | Katz strictly positive | x_v > 0 for all v when β > 0 and α < 1/λ_max | Attenuation factor α too large or β=0 edge case |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-GC-001 | 1 | 4 | exhaustive |
| KANI-GC-002 | 2 | 4 | exhaustive |
| KANI-GC-003 | 6 | 4 | exhaustive |
| KANI-GRAPH_-004 | Degree centrality bounded | 8 | stub_float |
| KANI-GRAPH_-005 | Betweenness non-negative | 8 | exhaustive |
| KANI-GRAPH_-006 | Closeness positive for connected | 8 | exhaustive |
| KANI-GRAPH_-007 | Eigenvector non-negativity | 8 | exhaustive |
| KANI-GRAPH_-008 | Katz strictly positive | 8 | exhaustive |
| KANI-GRAPH_-009 | Harmonic centrality bounded | 8 | stub_float |
| KANI-GRAPH_-010 | Star graph maximum | 8 | exhaustive |
| KANI-GRAPH_-011 | Complete graph symmetry | 8 | exhaustive |

## QA Gate

**Graph Centrality Contract** (F-GC-001)

Graph centrality correctness quality gate

**Checks:** degree_bounded, betweenness_non_negative, eigenvector_non_negative, complete_graph_symmetry, star_graph_degree, harmonic_bounded

**Pass criteria:** All 8 falsification tests pass

