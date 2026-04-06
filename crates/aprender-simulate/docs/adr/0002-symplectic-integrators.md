# ADR-002: Symplectic Integrators for Physics Simulations

## Status

Accepted

## Context

Numerical integration of Hamiltonian systems (orbital mechanics, harmonic oscillators) requires special care. Standard methods like Euler or RK4 introduce energy drift that accumulates over time, making long-term simulations unreliable.

### Problem
- Euler method: O(Δt) error, energy drifts monotonically
- RK4: O(Δt⁴) error, but still drifts over many orbits
- For 100-year orbital predictions, small drifts compound catastrophically

### Requirements
- Energy conservation to < 1e-9 relative error per orbit
- Angular momentum conservation to < 1e-12 per orbit
- Stable for millions of timesteps

## Decision

We use **Störmer-Verlet** (leapfrog) integration for all Hamiltonian systems.

### Algorithm

```
v(t + Δt/2) = v(t) + (Δt/2) · a(x(t))
x(t + Δt)   = x(t) + Δt · v(t + Δt/2)
v(t + Δt)   = v(t + Δt/2) + (Δt/2) · a(x(t + Δt))
```

### Properties
- **Symplectic**: Preserves phase space volume (Liouville's theorem)
- **Time-reversible**: v → -v gives exact reversal
- **Energy bounded**: Error oscillates but doesn't drift
- **O(Δt²)**: Second-order accurate

### Higher-Order Option

For high-precision needs, use **Yoshida 4th-order** composition:

```rust
const W0: f64 = -1.702414383919315;
const W1: f64 = 1.351207191959657;
// 3 Verlet steps with weighted timesteps
```

## Consequences

### Positive
- Energy conservation guaranteed to machine precision bounds
- Long-term stability for orbital mechanics
- Efficient: only one force evaluation per step
- Well-understood error characteristics

### Negative
- Cannot handle non-Hamiltonian forces directly
- Requires separable Hamiltonian (H = T(p) + V(q))
- Adaptive timestep is complex (breaks symplecticity)

### Alternatives Considered

| Method | Rejected Because |
|--------|------------------|
| Euler | Energy drift O(Δt) per step |
| RK4 | Energy drift accumulates |
| RK45 | Adaptive step breaks symplecticity |
| Gauss-Legendre | Too computationally expensive |

## References

- Hairer, Lubich, Wanner. "Geometric Numerical Integration" (2006)
- Yoshida. "Construction of higher order symplectic integrators" (1990)
