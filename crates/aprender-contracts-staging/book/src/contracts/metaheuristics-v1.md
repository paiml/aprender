# metaheuristics-v1

**Version:** 1.0.0

Metaheuristic optimization algorithms -- SA, GA, PSO

## References

- Kirkpatrick et al. (1983) Optimization by Simulated Annealing
- Deb & Agrawal (1995) Simulated Binary Crossover for Continuous Search Space
- Kennedy & Eberhart (1995) Particle Swarm Optimization

## Equations

### best_monotone

$$
f(x*_{t+1}) <= f(x*_t) for minimization
$$

**Domain:** $optimization history: sequence of best-so-far objective values$

**Codomain:** $boolean invariant (always true)$

**Invariants:**

- $Best-so-far value never increases (monotone non-increasing)$
- $Applies to SA, GA, and PSO independently$
- $Holds regardless of algorithm parameters$

### ga_crossover

$$
child = 0.5 * [(1 + beta) * parent_1 + (1 - beta) * parent_2]
$$

**Domain:** $parent_1, parent_2 within search space bounds, beta >= 0 (spread factor)$

**Codomain:** $child within search space bounds (clamped if necessary)$

**Invariants:**

- $Children are deterministic given parents and beta$
- $beta = 1 produces midpoint of parents$
- $Children are clamped to search space bounds$

### pso_velocity

$$
v_{t+1} = w * v_t + c1 * r1 * (p_best - x_t) + c2 * r2 * (g_best - x_t)
$$

**Domain:** $x_t, v_t, p_best, g_best in R^d, w in [0, 1], c1 > 0, c2 > 0, r1, r2 in [0, 1]$

**Codomain:** $v_{t+1} in R^d (clamped to v_max)$

**Invariants:**

- $Velocity is clamped to [-v_max, v_max] per dimension$
- $Inertia weight w dampens previous velocity$
- $Cognitive (c1) and social (c2) terms attract toward best positions$

### sa_acceptance

$$
P(accept) = 1 if Delta_E < 0, else \exp(-Delta_E / T)
$$

**Domain:** $Delta_E in R, T > 0$

**Codomain:** $P in (0, 1]$

**Invariants:**

- $Improving moves (Delta_E < 0) are always accepted$
- $Acceptance probability decreases as temperature decreases$
- $Acceptance probability is always positive (never exactly 0)$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | Best objective non-increasing across iterations | $forall t: best_val[t+1] <= best_val[t]$ |
| 2 | bound | SA best improves or stays same | $final_best <= initial_best after SA run$ |
| 3 | bound | GA best improves or stays same | $final_best <= initial_best after GA run$ |
| 4 | bound | PSO best improves or stays same | $final_best <= initial_best after PSO run$ |
| 5 | bound | SA acceptance probability in (0, 1] | $forall Delta_E, T > 0: 0 < P(accept) <= 1$ |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-MH-001 | Best objective monotonicity | best_val[t+1] <= best_val[t] for all iterations across SA, GA, PSO | Best-so-far tracking not properly maintained |
| FALSIFY-MH-002 | SA best value improvement | SA final_best <= initial_best | SA acceptance criterion or best tracking is broken |
| FALSIFY-MH-003 | GA best value improvement | GA final_best <= initial_best | GA elitism or selection mechanism loses best solution |
| FALSIFY-MH-004 | PSO best value improvement | PSO final_best <= initial_best | PSO global best update mechanism loses best solution |
| FALSIFY-MH-005 | SA acceptance probability bounds | 0 < P(accept) <= 1 for all Delta_E and T > 0 | Acceptance probability computation overflow or underflow |
| FALSIFY-MH-006 | PSO velocity clamping | \|v_i\| <= v_max for all dimensions after velocity update | Velocity clamping not applied after update |
| FALSIFY-MH-007 | GA crossover bounds | Children within search space bounds after SBX crossover | SBX crossover produces out-of-bounds children without clamping |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-MH-001 | MH-INV-001 | 8 | stub_float |
| KANI-MH-002 | MH-BND-001 | 8 | stub_float |
| KANI-MH-003 | MH-BND-002 | 8 | stub_float |
| KANI-METAHE-004 | Best objective non-increasing across iterations | 8 | exhaustive |
| KANI-METAHE-005 | SA best improves or stays same | 8 | exhaustive |
| KANI-METAHE-006 | GA best improves or stays same | 8 | exhaustive |
| KANI-METAHE-007 | PSO best improves or stays same | 8 | exhaustive |
| KANI-METAHE-008 | SA acceptance probability in (0, 1] | 8 | exhaustive |

## QA Gate

**Metaheuristics Contract** (F-MH-001)

Metaheuristic optimization correctness quality gate

**Checks:** best_monotonicity, sa_improvement, ga_improvement, pso_improvement, sa_acceptance_bounds, pso_velocity_clamping, ga_crossover_bounds

**Pass criteria:** All 7 falsification tests pass + Kani harnesses verify

