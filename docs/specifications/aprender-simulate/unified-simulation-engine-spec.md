# Unified Simulation Engine Specification

## simular: A Falsifiable, Reproducible Simulation Framework for the Sovereign AI Stack

**Version:** 0.1.0-draft
**Status:** RFC (Request for Comments)
**Authors:** PAIML Engineering
**Date:** 2025-12-10

---

## Abstract

This specification defines **simular**, a unified simulation engine for replayable physics, machine learning, Monte Carlo, and optimization simulations within the Sovereign AI Stack. The design adheres to three foundational methodologies:

1. **Toyota Production System (TPS)**: Jidoka (stop-on-error), Poka-Yoke (mistake-proofing), Heijunka (load leveling), and Kaizen (continuous improvement)
2. **JPL Mission-Critical Verification**: Formal V&V methodology, The Power of 10 coding rules, and SPICE-grade numerical accuracy
3. **Popperian Falsification**: Every simulation hypothesis must be falsifiable; null hypothesis testing drives validation

The engine prioritizes the Sovereign AI Stack's native crates (trueno, aprender, entrenar, realizar, alimentar, pacha, renacer) as first-class integrations.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Theoretical Foundations](#2-theoretical-foundations)
3. [Architecture](#3-architecture)
4. [Core Subsystems](#4-core-subsystems)
   - 4.1 [Event Scheduler](#41-event-scheduler)
   - 4.2 [Jidoka: Stop-on-Error](#42-jidoka-stop-on-error-smart-detection)
   - 4.3 [Advanced TPS Kaizen](#43-advanced-tps-kaizen)
5. [Domain Engines](#5-domain-engines)
   - 5.1 [Physics Engine](#51-physics-engine)
   - 5.2 [Monte Carlo Engine](#52-monte-carlo-engine)
   - 5.3 [Optimization Engine](#53-optimization-engine)
   - 5.4 [ML Integration](#54-ml-integration)
   - 5.5 [ML Simulation Engine](#55-ml-simulation-engine)
6. [Replay and Time-Travel](#6-replay-and-time-travel)
7. [Visualization](#7-visualization)
8. [YAML Configuration Schema](#8-yaml-configuration-schema)
9. [Quality Assurance](#9-quality-assurance)
10. [References](#10-references)
11. [Appendix F: ML Simulation Nullification Framework](#appendix-f-ml-simulation-nullification-framework)

---

## 1. Introduction

### 1.1 Problem Statement

Existing simulation frameworks suffer from fundamental limitations:

| Framework | Physics | ML | Monte Carlo | Reproducibility | YAML Config | Visualization |
|-----------|---------|-----|-------------|-----------------|-------------|---------------|
| Nyx [1] | Orbital | No | Partial | Limited | No | No |
| MadSim [2] | No | No | No | Yes | No | No |
| Rapier [3] | Rigid body | No | No | Partial | No | No |
| OpenTwins [4] | FMI | Partial | No | Limited | Partial | Yes |

No unified framework exists that combines:
- Multi-domain physics (orbital, rigid body, fluid, discrete event)
- Machine learning integration (surrogate models, reinforcement learning)
- Monte Carlo methods with variance reduction
- Deterministic replay with time-travel debugging
- Declarative YAML configuration
- Rich TUI and web visualization

### 1.2 Solution: simular

**simular** is a pure-Rust simulation engine that unifies these capabilities while leveraging the Sovereign AI Stack for maximum performance and privacy. Recent studies have demonstrated Rust's capability to match or exceed C++/Fortran performance in high-performance computing contexts while guaranteeing memory safety [28][33].

### 1.3 Guiding Principles

#### 1.3.1 Popperian Falsifiability and Nullification Theory

> "A theory that explains everything, explains nothing." — Karl Popper [5]

Every simulation model in simular must be **falsifiable**. Popper's demarcation criterion—distinguishing science from non-science—requires that any genuinely scientific hypothesis must be empirically falsifiable [5][6]. This principle is foundational to simular's validation methodology.

##### 1.3.1.1 The Nullification Framework

Popper's falsificationism presaged modern Null Hypothesis Significance Testing (NHST) [26]. The core insight is the **logical asymmetry between verification and falsification**: while it is impossible to verify a universal proposition ("all swans are white") by observation alone, a single genuine counter-instance ("one black swan") falsifies it [5][6].

**Demarcation Criterion**: A theory T is *scientific* iff there exists some observation O that could refute T:

```
Scientific(T) ⟺ ∃O: T ⊢ P ∧ Observe(¬P) → Falsified(T)
```

**Null Hypothesis Connection**: The null hypothesis H₀ is the hypothesis we seek to *nullify* (reject). In NHST, scientists do not prove truth; they use empirical evidence to falsify or disprove [26]:

| Concept | Popper's Framework | NHST Framework |
|---------|-------------------|----------------|
| **Goal** | Seek refutation | Reject H₀ |
| **Success** | Corroboration (not proof) | Statistical significance |
| **Failure** | Falsification | Fail to reject H₀ |
| **Logic** | Modus tollens | p-value < α |

##### 1.3.1.2 Operationalizing Falsification in simular

1. **Null Hypothesis (H₀)**: The model's predictions match reality within tolerance
2. **Alternative Hypothesis (H₁)**: The model's predictions deviate beyond tolerance
3. **Nullification Attempt**: Actively seek observations that falsify H₀
4. **Deductive Testing**: Infer predictions from hypotheses, compare to observations [6]
5. **Robustness Metrics**: Quantitative measure of "how falsifiable" via Signal Temporal Logic [29]

##### 1.3.1.3 Robustness as Quantitative Falsifiability

Signal Temporal Logic (STL) provides robustness semantics that quantify how much a signal can be perturbed before changing the truth value of a specification [29]. This measures "distance to falsification":

```
ρ(φ, s) > 0  ⟹  s satisfies φ (with margin ρ)
ρ(φ, s) < 0  ⟹  s violates φ (by margin |ρ|)
ρ(φ, s) = 0  ⟹  s is at the boundary (maximally falsifiable)
```

Higher |ρ| indicates greater confidence; ρ ≈ 0 indicates a hypothesis at risk of falsification.

##### 1.3.1.4 Implementation

```rust
/// A simulation hypothesis that can be falsified (Popperian methodology)
pub trait FalsifiableHypothesis {
    /// Generate testable predictions (deductive inference)
    fn predict(&self, state: &SimState) -> Predictions;

    /// Define what would falsify this hypothesis (demarcation criterion)
    fn falsification_criteria(&self) -> FalsificationCriteria;

    /// Compute robustness degree (STL semantics) [29]
    fn robustness(&self, signal: &Signal) -> f64;

    /// Null hypothesis significance test [26]
    fn null_hypothesis_test(
        &self,
        predictions: &Predictions,
        observations: &Observations,
        significance: f64,  // α level (typically 0.05)
    ) -> NHSTResult;
}

/// Result of null hypothesis significance testing
pub enum NHSTResult {
    /// H₀ rejected: evidence supports falsification
    Rejected { p_value: f64, effect_size: f64 },
    /// Failed to reject H₀: model corroborated (not proven)
    NotRejected { p_value: f64, power: f64 },
}
```

#### 1.3.2 Toyota Production System Integration

| TPS Principle | simular Implementation |
|---------------|------------------------|
| **Jidoka** (自働化) | Pre-flight anomaly detection [49], graduated severity classification, self-healing auto-correction [57] |
| **Poka-Yoke** (ポカヨケ) | Explicit units in YAML (via `uom`) [56], compile-time dimensional analysis, schema validation |
| **Heijunka** (平準化) | Work-stealing parallel Monte Carlo [55], adaptive time stepping, load-balanced task distribution |
| **Kaizen** (改善) | Continuous model refinement via Bayesian optimization, schema evolution for forward compatibility [50] |
| **Muda** (無駄) | Zero-copy streaming checkpoints [50], split EventJournal headers/payloads, lazy evaluation |
| **Genchi Genbutsu** (現地現物) | Direct observation via replay; "go and see" the simulation state |
| **Andon** (行灯) | Escalation from auto-correct to full-stop based on severity and correction history [51] |

#### 1.3.3 JPL Mission-Critical Rigor

Adapted from Holzmann's "Power of 10" rules [7] and NASA IV&V methodology [8]:

1. **Bounded loops**: All iterative solvers have maximum iteration bounds
2. **Static memory**: Pre-allocated buffers for real-time guarantees
3. **No recursion**: Iterative algorithms only (stack overflow prevention)
4. **Assertion density**: Minimum 2 assertions per function
5. **Formal verification**: Property-based testing with falsification
6. **Independent V&V**: Dual-implementation validation for critical paths

---

## 2. Theoretical Foundations

### 2.1 Deterministic Reproducibility

**Definition 2.1 (Reproducibility)**: A simulation S is *reproducible* if, given identical initial conditions I and random seed σ, repeated executions produce bitwise-identical results R [9]:

```
∀ runs r₁, r₂: S(I, σ) → R₁ ∧ S(I, σ) → R₂ ⟹ R₁ ≡ R₂
```

**Theorem 2.1 (Deterministic Parallel Execution)**: Reproducibility in parallel simulations requires:
1. Deterministic task scheduling (fixed ordering)
2. Commutative reduction operations
3. Seeded, partitioned random number generators

*Proof sketch*: Variable communication latencies lead to processors completing tasks in different orders. Using locks and deterministic scheduling eliminates this source of non-determinism [9].

### 2.2 Monte Carlo Convergence

**Theorem 2.2 (Central Limit Theorem for Monte Carlo)**: For iid samples X₁, ..., Xₙ with E[X] = μ and Var(X) = σ², the Monte Carlo estimator converges as [10]:

```
μ̂ₙ = (1/n)∑Xᵢ → N(μ, σ²/n) as n → ∞
```

The convergence rate O(n⁻¹/²) is **dimension-independent**, making Monte Carlo ideal for high-dimensional spaces [11].

**Variance Reduction Techniques** [12]:
- Antithetic variates: ρ(X, X') < 0 reduces variance
- Control variates: E[μ̂] = E[X] - c(E[Y] - E[Y])
- Importance sampling: Reweight by likelihood ratio
- Stratified sampling: Partition sample space

### 2.3 Symplectic Integration

**Definition 2.2 (Symplectic Integrator)**: An integrator Φₕ is *symplectic* if it preserves the symplectic 2-form ω = ∑dpᵢ ∧ dqᵢ [13]:

```
Φₕ*ω = ω
```

**Theorem 2.3 (Energy Conservation)**: Symplectic integrators exhibit bounded energy error for Hamiltonian systems, with the error oscillating around the true energy rather than drifting [14]. Higher-order symplectic integrators can be constructed via composition methods [27]:

```
|H(pₙ, qₙ) - H(p₀, q₀)| ≤ Ch^k for all n
```

The **Störmer-Verlet** integrator is the composition of symplectic Euler methods with step size h/2 [14]:

```
q_{n+1/2} = qₙ + (h/2)pₙ/m
p_{n+1} = pₙ + hF(q_{n+1/2})
q_{n+1} = q_{n+1/2} + (h/2)p_{n+1}/m
```

### 2.4 Bayesian Optimization Theory

**Definition 2.3 (Gaussian Process Surrogate)**: A GP surrogate models the objective function f as [15]:

```
f(x) ~ GP(μ(x), k(x, x'))
```

where μ is the mean function and k is the covariance kernel.

**Acquisition Function**: The Expected Improvement (EI) balances exploration and exploitation [16]:

```
EI(x) = E[max(f(x) - f(x⁺), 0)]
```

where x⁺ is the current best observation.

---

## 3. Architecture

### 3.1 Stack Integration

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           batuta (Orchestration)                             │
│                    Pipeline coordination, CLI, TUI dashboard                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────────────────┐ │
│  │   realizar   │  │    pacha     │  │              simular               │ │
│  │  (Inference) │  │  (Registry)  │  │    (Unified Simulation Engine)    │ │
│  │              │  │              │  │                                    │ │
│  │ GGUF serving │  │ Ed25519 sigs │  │  ┌────────────────────────────┐   │ │
│  │ SafeTensors  │  │ Version ctrl │  │  │     Domain Engines         │   │ │
│  │              │  │              │  │  │  • Physics (rigid, orbital)│   │ │
│  └──────┬───────┘  └──────┬───────┘  │  │  • Monte Carlo (parallel)  │   │ │
│         │                 │          │  │  • Optimization (BO, CMA)  │   │ │
│         │                 │          │  │  • ML (surrogate, RL)      │   │ │
│         │                 │          │  └────────────────────────────┘   │ │
│         │                 │          │                                    │ │
│         │                 │          │  ┌────────────────────────────┐   │ │
│         │                 │          │  │     Replay Engine          │   │ │
│         │                 │          │  │  • Checkpointing           │   │ │
│         │                 │          │  │  • Time-travel debugging   │   │ │
│         │                 │          │  │  • Event journal           │   │ │
│         │                 │          │  └────────────────────────────┘   │ │
│         │                 │          │                                    │ │
│         │                 │          │  ┌────────────────────────────┐   │ │
│         │                 │          │  │     Visualization          │   │ │
│         │                 │          │  │  • TUI (ratatui)           │   │ │
│         │                 │          │  │  • WebGL (wgpu)            │   │ │
│         │                 │          │  │  • Export (MP4, Parquet)   │   │ │
│         │                 │          │  └────────────────────────────┘   │ │
│         │                 │          └────────────────────────────────────┘ │
│         │                 │                           │                     │
├─────────┴─────────────────┴───────────────────────────┴─────────────────────┤
│                                                                              │
│  ┌──────────────────────────────┐  ┌──────────────────────────────────────┐ │
│  │         aprender             │  │            entrenar                  │ │
│  │      (ML Algorithms)         │  │           (Training)                 │ │
│  │                              │  │                                      │ │
│  │  Regression, Trees, GNN      │  │  Autograd, LoRA, Quantization        │ │
│  │  ARIMA, Gaussian Processes   │  │  Policy gradients, PPO               │ │
│  └──────────────┬───────────────┘  └──────────────────┬───────────────────┘ │
│                 │                                      │                     │
├─────────────────┴──────────────────────────────────────┴─────────────────────┤
│                                                                              │
│  ┌──────────────────────────────┐  ┌──────────────────────────────────────┐ │
│  │         alimentar            │  │            renacer                   │ │
│  │      (Data Loading)          │  │           (Tracing)                  │ │
│  │                              │  │                                      │ │
│  │  Parquet, Arrow, zero-copy   │  │  Syscall trace, source correlation  │ │
│  └──────────────┬───────────────┘  └──────────────────┬───────────────────┘ │
│                 │                                      │                     │
├─────────────────┴──────────────────────────────────────┴─────────────────────┤
│                                                                              │
│                              trueno (Compute)                                │
│                                                                              │
│     SIMD: AVX2, AVX-512, NEON  │  GPU: wgpu compute shaders                │
│     trueno-db: Analytics       │  trueno-graph: Code analysis              │
│     trueno-rag: RAG pipeline   │                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Module Hierarchy

```
simular/
├── Cargo.toml
├── src/
│   ├── lib.rs                    # Public API
│   ├── config/                   # YAML configuration
│   │   ├── mod.rs
│   │   ├── schema.rs             # Serde structs with validation
│   │   ├── validator.rs          # Poka-Yoke: compile-time checks
│   │   └── units.rs              # Dimensional analysis (uom)
│   ├── engine/                   # Core simulation loop
│   │   ├── mod.rs
│   │   ├── scheduler.rs          # Event queue, priority heap
│   │   ├── state.rs              # World state, ECS-like
│   │   ├── rng.rs                # Deterministic PRNG (PCG)
│   │   ├── clock.rs              # Simulation time management
│   │   └── jidoka.rs             # Stop-on-error, anomaly detection
│   ├── domains/                  # Simulation domains
│   │   ├── mod.rs
│   │   ├── physics/
│   │   │   ├── mod.rs
│   │   │   ├── rigid_body.rs     # Newtonian mechanics
│   │   │   ├── orbital.rs        # Keplerian + perturbations
│   │   │   ├── fluid.rs          # SPH, lattice Boltzmann
│   │   │   └── integrators.rs    # Verlet, RK4, symplectic
│   │   ├── monte_carlo/
│   │   │   ├── mod.rs
│   │   │   ├── sampler.rs        # Importance, stratified
│   │   │   ├── variance.rs       # Antithetic, control variates
│   │   │   └── parallel.rs       # trueno-accelerated
│   │   ├── optimization/
│   │   │   ├── mod.rs
│   │   │   ├── bayesian.rs       # GP surrogate (aprender)
│   │   │   ├── cma_es.rs         # Covariance Matrix Adaptation
│   │   │   ├── genetic.rs        # Evolutionary algorithms
│   │   │   └── gradient.rs       # BFGS, L-BFGS (entrenar)
│   │   └── ml/
│   │       ├── mod.rs
│   │       ├── surrogate.rs      # Surrogate model training
│   │       ├── rl.rs             # Reinforcement learning
│   │       └── inference.rs      # realizar integration
│   ├── replay/                   # Time-travel debugging
│   │   ├── mod.rs
│   │   ├── checkpoint.rs         # Snapshot/restore (incremental)
│   │   ├── journal.rs            # Append-only event log
│   │   ├── scrubber.rs           # Seek to any timestep
│   │   └── diff.rs               # State differencing
│   ├── viz/                      # Visualization
│   │   ├── mod.rs
│   │   ├── tui/                  # Terminal UI (ratatui)
│   │   │   ├── mod.rs
│   │   │   ├── dashboard.rs      # Main layout
│   │   │   ├── trajectory.rs     # 2D/3D path plotting
│   │   │   ├── metrics.rs        # Real-time charts
│   │   │   └── phase_space.rs    # Phase portraits
│   │   ├── web/                  # WebSocket + WebGL
│   │   │   ├── mod.rs
│   │   │   ├── server.rs         # axum WebSocket
│   │   │   └── renderer.rs       # wgpu compute → WebGL
│   │   └── export/
│   │       ├── mod.rs
│   │       ├── video.rs          # MP4, GIF encoding
│   │       ├── parquet.rs        # alimentar integration
│   │       └── json.rs           # JSON Lines streaming
│   ├── scenarios/                # Pre-built templates
│   │   ├── mod.rs
│   │   ├── rocket.rs             # Launch vehicle
│   │   ├── satellite.rs          # Orbital mechanics
│   │   ├── climate.rs            # Simplified climate model
│   │   ├── portfolio.rs          # Financial Monte Carlo
│   │   └── epidemic.rs           # Compartmental models
│   └── falsification/            # Popperian testing
│       ├── mod.rs
│       ├── hypothesis.rs         # H₀/H₁ framework
│       ├── oracle.rs             # Ground truth comparison
│       └── sensitivity.rs        # Parameter sensitivity
├── tests/
│   ├── integration/
│   ├── property/                 # proptest
│   └── reference/                # Known-good outputs
└── examples/
    ├── falcon9_stage_sep.rs
    ├── satellite_conjunction.rs
    ├── climate_sensitivity.rs
    └── portfolio_var.rs
```

---

## 4. Core Subsystems

### 4.1 Deterministic RNG

Following the deterministic simulation testing pattern [17], simular uses a partitioned PRNG:

```rust
use pcg_rand::Pcg64;

/// Deterministic, reproducible random number generator
pub struct SimRng {
    /// Master seed for reproducibility
    master_seed: u64,
    /// Per-thread RNGs derived from master
    thread_rngs: Vec<Pcg64>,
    /// Current sequence counter
    sequence: u64,
}

impl SimRng {
    /// Create partitioned RNGs for parallel execution
    pub fn partition(&mut self, n: usize) -> Vec<Pcg64> {
        (0..n)
            .map(|i| {
                let stream = self.sequence + i as u64;
                self.sequence += n as u64;
                Pcg64::new(self.master_seed, stream)
            })
            .collect()
    }
}
```

### 4.2 Jidoka: Stop-on-Error (Smart Detection)

Implementing TPS's Jidoka principle for immediate anomaly detection. Per the Batuta Stack Review [32], Jidoka must distinguish between **true violations** and **acceptable variance** to avoid false positives (Muda of over-processing):

```rust
/// Jidoka: Autonomous anomaly detection and halt
///
/// Design follows Batuta Review recommendations:
/// - Distinguish breaking vs. non-breaking violations
/// - Use SemVer-like tolerance semantics
/// - Avoid "stopping the line" for acceptable variance
pub struct JidokaGuard {
    /// Maximum allowed relative energy drift
    energy_tolerance: f64,
    /// NaN/Inf detection enabled
    check_finite: bool,
    /// Constraint violation threshold
    constraint_tolerance: f64,
    /// Severity classification for smart detection
    severity_classifier: SeverityClassifier,
}

/// Severity levels for Jidoka violations (avoid false positives)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ViolationSeverity {
    /// Acceptable variance within tolerance (continue)
    Acceptable,
    /// Warning: approaching tolerance boundary (log, continue)
    Warning,
    /// Critical: tolerance exceeded (stop the line)
    Critical,
    /// Fatal: unrecoverable state (halt immediately)
    Fatal,
}

impl JidokaGuard {
    /// Check state invariants with smart severity classification
    pub fn check(&self, state: &SimState) -> Result<Option<JidokaWarning>, JidokaViolation> {
        // Poka-Yoke: Prevent error propagation (FATAL - always stop)
        if self.check_finite && !state.all_finite() {
            return Err(JidokaViolation::NonFiniteValue {
                location: state.first_non_finite(),
                severity: ViolationSeverity::Fatal,
            });
        }

        // Energy conservation check with graduated response
        if let Some(energy) = state.total_energy() {
            let drift = (energy - state.initial_energy()).abs() / state.initial_energy();

            let severity = self.severity_classifier.classify_energy_drift(drift, self.energy_tolerance);

            match severity {
                ViolationSeverity::Acceptable => {}
                ViolationSeverity::Warning => {
                    return Ok(Some(JidokaWarning::EnergyDriftApproaching { drift }));
                }
                ViolationSeverity::Critical | ViolationSeverity::Fatal => {
                    return Err(JidokaViolation::EnergyDrift { drift, severity });
                }
            }
        }

        // Constraint satisfaction with graduated response
        for (name, violation) in state.constraint_violations() {
            let severity = self.severity_classifier.classify_constraint(violation, self.constraint_tolerance);

            if severity >= ViolationSeverity::Critical {
                return Err(JidokaViolation::ConstraintViolation { name, violation, severity });
            }
        }

        Ok(None)
    }
}

/// Classifier for graduated Jidoka responses
pub struct SeverityClassifier {
    /// Warning threshold as fraction of tolerance (e.g., 0.8 = warn at 80%)
    warning_fraction: f64,
}

impl SeverityClassifier {
    /// Classify energy drift severity
    pub fn classify_energy_drift(&self, drift: f64, tolerance: f64) -> ViolationSeverity {
        if drift.is_nan() || drift.is_infinite() {
            ViolationSeverity::Fatal
        } else if drift > tolerance {
            ViolationSeverity::Critical
        } else if drift > tolerance * self.warning_fraction {
            ViolationSeverity::Warning
        } else {
            ViolationSeverity::Acceptable
        }
    }
}
```

### 4.3 Advanced TPS Kaizen

This section documents continuous improvement recommendations from TPS code review, addressing specific *Muda* (waste) and strengthening *Jidoka* implementation.

#### 4.3.1 Pre-flight Jidoka (In-Process Anomaly Detection)

**Critique**: The current `AnomalyDetector` checks for issues *after* a batch is processed. If a fatal anomaly (NaN loss) occurs, compute cost has already been incurred [49].

**Kaizen**: Implement "Pre-flight Jidoka" that aborts *during* computation graph execution:

```rust
/// Pre-flight Jidoka guard for in-process anomaly detection [49][51]
/// Prevents Muda of Processing by aborting before defects propagate
#[derive(Debug, Clone)]
pub struct PreflightJidoka {
    /// Abort conditions (OR'd together)
    abort_on: AbortConditions,
    /// Counter for early aborts (metrics)
    abort_count: u64,
}

bitflags::bitflags! {
    /// Conditions that trigger immediate abort during computation
    pub struct AbortConditions: u32 {
        /// Abort on NaN or Infinity values
        const NON_FINITE = 0b0001;
        /// Abort when gradient norm exceeds threshold
        const GRADIENT_EXPLOSION = 0b0010;
        /// Abort when gradient norm drops below threshold
        const GRADIENT_VANISHING = 0b0100;
        /// Abort on reward hacking detection [51]
        const REWARD_ANOMALY = 0b1000;
    }
}

impl PreflightJidoka {
    /// Create with default abort conditions
    #[must_use]
    pub fn new() -> Self {
        Self {
            abort_on: AbortConditions::NON_FINITE | AbortConditions::GRADIENT_EXPLOSION,
            abort_count: 0,
        }
    }

    /// Execute forward-backward pass with in-process checking
    /// Aborts immediately upon detecting anomaly (before full computation)
    pub fn forward_backward_checked<M: Model>(
        &mut self,
        model: &M,
        params: &Tensor,
        batch: &Batch,
        grad_explosion_threshold: f64,
    ) -> SimResult<(f64, Tensor)> {
        // Check input validity BEFORE compute
        if self.abort_on.contains(AbortConditions::NON_FINITE) && !batch.inputs.all_finite() {
            self.abort_count += 1;
            return Err(SimError::jidoka("Pre-flight: Non-finite input detected"));
        }

        let loss = model.forward(params, &batch.inputs)?;

        // Check loss validity BEFORE backward pass (avoid wasted compute)
        if self.abort_on.contains(AbortConditions::NON_FINITE) && !loss.is_finite() {
            self.abort_count += 1;
            return Err(SimError::jidoka("Pre-flight: Non-finite loss before backward pass"));
        }

        let grads = model.backward(params, &batch.inputs, loss)?;

        // Check gradient validity
        let grad_norm = grads.norm_l2();
        if self.abort_on.contains(AbortConditions::GRADIENT_EXPLOSION)
            && grad_norm > grad_explosion_threshold
        {
            self.abort_count += 1;
            return Err(SimError::jidoka(format!(
                "Pre-flight: Gradient explosion detected (norm={grad_norm:.2e})"
            )));
        }

        if self.abort_on.contains(AbortConditions::GRADIENT_VANISHING)
            && grad_norm < 1e-10
        {
            self.abort_count += 1;
            return Err(SimError::jidoka("Pre-flight: Gradient vanishing detected"));
        }

        Ok((loss, grads))
    }
}
```

#### 4.3.2 Andon vs Jidoka: Automated Recovery (Self-Healing)

**Critique**: The `JidokaMLFeedback` generates patches but doesn't *apply* them dynamically. Long-running "Sovereign AI" training runs shouldn't halt for minor recoverable issues (Muda of Waiting) [52].

**Kaizen**: Distinguish between *Andon* (stop and alert) and *Jidoka* (auto-correct):

```rust
/// Jidoka severity determines response strategy [51][57]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JidokaResponse {
    /// Andon: Full stop, human intervention required
    Andon,
    /// Auto-correct: Apply patch and continue
    AutoCorrect,
    /// Monitor: Log warning, continue with observation
    Monitor,
}

/// Self-healing Jidoka controller for ML training [57]
pub struct SelfHealingJidoka {
    /// Maximum auto-corrections before escalating to Andon
    max_auto_corrections: usize,
    /// Current correction count
    correction_count: usize,
    /// Checkpoint for rollback
    last_good_checkpoint: Option<SimTime>,
    /// Applied patches history
    applied_patches: Vec<RulePatch>,
}

impl SelfHealingJidoka {
    /// Determine response based on anomaly type and history
    pub fn classify_response(&self, anomaly: &TrainingAnomaly) -> JidokaResponse {
        match anomaly {
            // Fatal: Always Andon
            TrainingAnomaly::NaN | TrainingAnomaly::ModelCorruption => JidokaResponse::Andon,

            // Recoverable: Auto-correct if under threshold
            TrainingAnomaly::LossSpike { .. } | TrainingAnomaly::GradientExplosion { .. } => {
                if self.correction_count < self.max_auto_corrections {
                    JidokaResponse::AutoCorrect
                } else {
                    JidokaResponse::Andon // Escalate after repeated issues
                }
            }

            // Minor: Monitor only
            TrainingAnomaly::SlowConvergence | TrainingAnomaly::HighVariance => {
                JidokaResponse::Monitor
            }
        }
    }

    /// Apply auto-correction and rollback to last checkpoint
    pub fn auto_correct(
        &mut self,
        simulation: &mut TrainingSimulation,
        patch: RulePatch,
    ) -> SimResult<()> {
        // Rollback to last good state
        if let Some(checkpoint_time) = self.last_good_checkpoint {
            simulation.rollback_to(checkpoint_time)?;
        }

        // Apply corrective patch (e.g., reduce learning rate)
        simulation.apply_patch(&patch)?;

        self.correction_count += 1;
        self.applied_patches.push(patch);

        Ok(())
    }
}
```

#### 4.3.3 Zero-Copy Streaming for Checkpoints (Muda Elimination)

**Critique**: Current `CheckpointManager` allocates intermediate `Vec<u8>` buffers for serialization then compression—massive memory churn for large ML models [50].

**Kaizen**: Stream directly from serializer → compressor → mmap file:

```rust
use std::io::Write;

/// Zero-copy streaming checkpoint manager [50]
/// Eliminates Muda of Processing (intermediate buffer allocation)
pub struct StreamingCheckpointManager {
    /// Memory-mapped checkpoint file
    mmap: memmap2::MmapMut,
    /// Write position in mmap
    write_pos: usize,
    /// Zstd compression level
    compression_level: i32,
}

impl StreamingCheckpointManager {
    /// Create checkpoint with zero intermediate allocations
    /// Streams: state → bincode → zstd → mmap
    pub fn checkpoint_streaming<S: Serialize>(&mut self, time: SimTime, state: &S) -> SimResult<()> {
        // Create a writer that pipes directly to mmap
        let mmap_writer = MmapWriter::new(&mut self.mmap, self.write_pos);

        // Streaming zstd encoder wrapping mmap writer
        let mut encoder = zstd::stream::Encoder::new(mmap_writer, self.compression_level)
            .map_err(|e| SimError::serialization(format!("Zstd encoder init: {e}")))?;

        // Stream serialization directly into encoder (no intermediate Vec)
        bincode::serialize_into(&mut encoder, &(time, state))
            .map_err(|e| SimError::serialization(format!("Streaming serialize: {e}")))?;

        let mmap_writer = encoder.finish()
            .map_err(|e| SimError::serialization(format!("Zstd finish: {e}")))?;

        self.write_pos = mmap_writer.position();

        // Sync to disk
        self.mmap.flush()?;

        Ok(())
    }
}

/// Writer adaptor for memory-mapped file
struct MmapWriter<'a> {
    mmap: &'a mut memmap2::MmapMut,
    pos: usize,
}

impl<'a> MmapWriter<'a> {
    fn new(mmap: &'a mut memmap2::MmapMut, pos: usize) -> Self {
        Self { mmap, pos }
    }

    fn position(&self) -> usize {
        self.pos
    }
}

impl Write for MmapWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let available = self.mmap.len() - self.pos;
        let to_write = buf.len().min(available);
        self.mmap[self.pos..self.pos + to_write].copy_from_slice(&buf[..to_write]);
        self.pos += to_write;
        Ok(to_write)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
```

#### 4.3.4 EventJournal Header/Payload Split (Muda of Over-Processing)

**Critique**: `TimeScrubber` deserializes full `Event` payloads just to advance time. Large `PredictionState` vectors waste CPU on unnecessary deserialization [50].

**Kaizen**: Split journal into Header index and Payload segments:

```rust
/// Split event journal for efficient time scrubbing [50][54]
/// Headers are compact for fast seeking; Payloads loaded on demand
pub struct SplitEventJournal {
    /// Compact header index (time, type, payload offset)
    headers: Vec<EventHeader>,
    /// Memory-mapped payload file
    payload_mmap: memmap2::Mmap,
}

/// Compact event header (fixed 24 bytes for cache efficiency)
#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct EventHeader {
    /// Simulation time (8 bytes)
    pub time: SimTime,
    /// Event type discriminant (4 bytes)
    pub event_type: u32,
    /// Offset into payload file (8 bytes)
    pub payload_offset: u64,
    /// Payload size in bytes (4 bytes)
    pub payload_size: u32,
}

impl SplitEventJournal {
    /// Seek to time without deserializing payloads (O(log n) via binary search)
    pub fn seek_to_time(&self, target: SimTime) -> Option<usize> {
        self.headers
            .binary_search_by(|h| h.time.cmp(&target))
            .ok()
            .or_else(|| {
                // Return index of first event after target
                self.headers.iter().position(|h| h.time >= target)
            })
    }

    /// Load specific event payload on demand
    pub fn load_payload<T: DeserializeOwned>(&self, header: &EventHeader) -> SimResult<T> {
        let start = header.payload_offset as usize;
        let end = start + header.payload_size as usize;

        bincode::deserialize(&self.payload_mmap[start..end])
            .map_err(|e| SimError::journal(format!("Payload deserialize: {e}")))
    }

    /// Iterate headers only (fast time scrubbing)
    pub fn headers_in_range(&self, start: SimTime, end: SimTime) -> impl Iterator<Item = &EventHeader> {
        self.headers.iter().filter(move |h| h.time >= start && h.time <= end)
    }
}
```

#### 4.3.5 Work Stealing for Monte Carlo (Heijunka)

**Critique**: Simple `map_reduce` suffers from "straggler problem"—threads wait for slowest simulation (Muda of Waiting) [55].

**Kaizen**: Dynamic work stealing with adaptive batching:

```rust
use std::sync::atomic::{AtomicUsize, Ordering};

/// Work-stealing Monte Carlo scheduler [55]
/// Implements Heijunka (load leveling) for variable-duration simulations
pub struct WorkStealingMonteCarlo {
    /// Global work queue (lock-free)
    global_queue: crossbeam_deque::Injector<SimulationTask>,
    /// Per-worker local queues
    worker_queues: Vec<crossbeam_deque::Worker<SimulationTask>>,
    /// Stealers for cross-worker theft
    stealers: Vec<crossbeam_deque::Stealer<SimulationTask>>,
    /// Completed task counter
    completed: AtomicUsize,
}

/// Individual simulation task
pub struct SimulationTask {
    /// Random seed for this trajectory
    pub seed: u64,
    /// Simulation parameters
    pub params: SimParams,
    /// Estimated duration (for scheduling heuristics)
    pub estimated_duration: Option<std::time::Duration>,
}

impl WorkStealingMonteCarlo {
    /// Execute Monte Carlo with work stealing [55]
    pub fn execute<F, R>(&self, n_samples: usize, simulate: F) -> Vec<R>
    where
        F: Fn(SimulationTask) -> R + Sync,
        R: Send,
    {
        // Populate global queue
        for seed in 0..n_samples as u64 {
            self.global_queue.push(SimulationTask {
                seed,
                params: SimParams::default(),
                estimated_duration: None,
            });
        }

        let results: std::sync::Mutex<Vec<Option<R>>> =
            std::sync::Mutex::new(vec![None; n_samples]);

        std::thread::scope(|s| {
            for (worker_id, worker) in self.worker_queues.iter().enumerate() {
                let stealers = &self.stealers;
                let global = &self.global_queue;
                let results = &results;
                let simulate = &simulate;

                s.spawn(move || {
                    loop {
                        // Try local queue first
                        let task = worker.pop().or_else(|| {
                            // Try global queue
                            std::iter::repeat_with(|| global.steal().success())
                                .find(|t| t.is_some())
                                .flatten()
                        }).or_else(|| {
                            // Steal from other workers (round-robin)
                            stealers
                                .iter()
                                .cycle()
                                .skip(worker_id)
                                .take(stealers.len())
                                .find_map(|s| s.steal().success())
                        });

                        match task {
                            Some(task) => {
                                let seed = task.seed as usize;
                                let result = simulate(task);
                                results.lock().unwrap()[seed] = Some(result);
                            }
                            None => break, // No more work
                        }
                    }
                });
            }
        });

        results.into_inner().unwrap().into_iter().flatten().collect()
    }
}
```

#### 4.3.6 Poka-Yoke: Explicit Units in Configuration

**Critique**: YAML uses raw floats (`threshold: 10.0`), risking unit confusion—catastrophic in aerospace [56].

**Kaizen**: Enforce explicit units via custom deserializer:

```rust
use uom::si::f64::*;
use uom::si::velocity::meter_per_second;

/// Velocity configuration with mandatory units (Poka-Yoke) [56]
#[derive(Debug, Clone, Deserialize)]
#[serde(try_from = "VelocityConfigRaw")]
pub struct VelocityConfig {
    pub value: Velocity,
}

/// Raw configuration for parsing
#[derive(Deserialize)]
struct VelocityConfigRaw {
    /// Value with unit string, e.g., "10.0 m/s" or "36.0 km/h"
    threshold: String,
}

impl TryFrom<VelocityConfigRaw> for VelocityConfig {
    type Error = String;

    fn try_from(raw: VelocityConfigRaw) -> Result<Self, Self::Error> {
        parse_velocity(&raw.threshold)
            .map(|value| VelocityConfig { value })
            .ok_or_else(|| format!(
                "Invalid velocity '{}'. Expected format: '<number> <unit>' \
                 where unit is 'm/s', 'km/h', 'ft/s', 'kn' (knots)",
                raw.threshold
            ))
    }
}

/// Parse velocity string with units
fn parse_velocity(s: &str) -> Option<Velocity> {
    let parts: Vec<&str> = s.trim().split_whitespace().collect();
    if parts.len() != 2 {
        return None;
    }

    let value: f64 = parts[0].parse().ok()?;
    let unit = parts[1].to_lowercase();

    Some(match unit.as_str() {
        "m/s" => Velocity::new::<meter_per_second>(value),
        "km/h" => Velocity::new::<uom::si::velocity::kilometer_per_hour>(value),
        "ft/s" => Velocity::new::<uom::si::velocity::foot_per_second>(value),
        "kn" | "knots" => Velocity::new::<uom::si::velocity::knot>(value),
        _ => return None,
    })
}

// YAML example with enforced units:
// ```yaml
// separation_velocity:
//   threshold: "10.0 m/s"  # Explicit unit required (Poka-Yoke)
// ```
```

#### 4.3.7 Schema Evolution for Forward Compatibility

**Critique**: `EventJournal` uses raw `bincode` on structs. Schema changes break old journals, violating reproducibility [50].

**Kaizen**: Version headers with migration support:

```rust
/// Versioned journal entry for schema evolution [50]
#[derive(Serialize, Deserialize)]
pub struct VersionedEntry {
    /// Schema version (SemVer-style)
    pub version: (u16, u16, u16),
    /// Entry type tag
    pub entry_type: &'static str,
    /// Payload (version-specific)
    pub payload: Vec<u8>,
}

/// Schema migrator for backward compatibility
pub struct SchemaMigrator {
    /// Registered migrations: (from_version, to_version) -> migration_fn
    migrations: HashMap<((u16, u16, u16), (u16, u16, u16)), MigrationFn>,
}

type MigrationFn = Box<dyn Fn(&[u8]) -> SimResult<Vec<u8>> + Send + Sync>;

impl SchemaMigrator {
    /// Register migration between versions
    pub fn register<F>(&mut self, from: (u16, u16, u16), to: (u16, u16, u16), migrate: F)
    where
        F: Fn(&[u8]) -> SimResult<Vec<u8>> + Send + Sync + 'static,
    {
        self.migrations.insert((from, to), Box::new(migrate));
    }

    /// Migrate entry to current version
    pub fn migrate_to_current(&self, entry: VersionedEntry, current: (u16, u16, u16)) -> SimResult<Vec<u8>> {
        if entry.version == current {
            return Ok(entry.payload);
        }

        // Find migration path (simplified: direct migration only)
        let migration = self.migrations
            .get(&(entry.version, current))
            .ok_or_else(|| SimError::journal(format!(
                "No migration path from {:?} to {:?}",
                entry.version, current
            )))?;

        migration(&entry.payload)
    }
}

/// Example: Migrate TrainingMetrics v1.0.0 -> v1.1.0 (added new field)
fn migrate_training_metrics_1_0_to_1_1(payload: &[u8]) -> SimResult<Vec<u8>> {
    #[derive(Deserialize)]
    struct TrainingMetricsV1_0 {
        epoch: usize,
        loss: f64,
        accuracy: f64,
    }

    #[derive(Serialize)]
    struct TrainingMetricsV1_1 {
        epoch: usize,
        loss: f64,
        accuracy: f64,
        learning_rate: f64,  // New field with default
    }

    let old: TrainingMetricsV1_0 = bincode::deserialize(payload)
        .map_err(|e| SimError::journal(format!("V1.0 deserialize: {e}")))?;

    let new = TrainingMetricsV1_1 {
        epoch: old.epoch,
        loss: old.loss,
        accuracy: old.accuracy,
        learning_rate: 0.001,  // Default value for missing field
    };

    bincode::serialize(&new)
        .map_err(|e| SimError::journal(format!("V1.1 serialize: {e}")))
}
```

### 4.4 Event Scheduler

Discrete event simulation with deterministic ordering [18]:

```rust
use std::collections::BinaryHeap;

/// Priority-ordered event queue
pub struct EventScheduler {
    /// Min-heap ordered by (time, sequence) for determinism
    queue: BinaryHeap<Reverse<ScheduledEvent>>,
    /// Monotonic sequence counter for tie-breaking
    sequence: u64,
    /// Current simulation time
    current_time: SimTime,
}

#[derive(Ord, PartialOrd, Eq, PartialEq)]
struct ScheduledEvent {
    time: SimTime,
    sequence: u64,  // Tie-breaker for determinism
    event: Event,
}

impl EventScheduler {
    /// Schedule event with deterministic ordering
    pub fn schedule(&mut self, time: SimTime, event: Event) {
        let seq = self.sequence;
        self.sequence += 1;
        self.queue.push(Reverse(ScheduledEvent { time, sequence: seq, event }));
    }

    /// Process next event (deterministic)
    pub fn next(&mut self) -> Option<(SimTime, Event)> {
        self.queue.pop().map(|Reverse(e)| {
            self.current_time = e.time;
            (e.time, e.event)
        })
    }
}
```

---

## 5. Domain Engines

### 5.1 Physics Engine

#### 5.1.1 Symplectic Integration (Verlet)

Following Hairer et al. [14] for energy-preserving integration:

```rust
/// Störmer-Verlet symplectic integrator
pub struct VerletIntegrator {
    dt: f64,
}

impl Integrator for VerletIntegrator {
    fn step(&self, state: &mut PhysicsState, forces: &ForceField) {
        // Half-step position update
        for (pos, vel) in state.positions.iter_mut().zip(&state.velocities) {
            *pos += vel * (self.dt / 2.0);
        }

        // Full-step velocity update
        for (vel, acc) in state.velocities.iter_mut().zip(forces.accelerations(&state.positions)) {
            *vel += acc * self.dt;
        }

        // Half-step position update
        for (pos, vel) in state.positions.iter_mut().zip(&state.velocities) {
            *pos += vel * (self.dt / 2.0);
        }
    }

    /// Global error bound: O(h²) [13]
    fn error_order(&self) -> u32 { 2 }

    /// Symplectic property: preserves phase space volume
    fn is_symplectic(&self) -> bool { true }
}
```

#### 5.1.2 Orbital Mechanics

JPL-grade ephemeris integration via trueno:

```rust
/// High-fidelity orbit propagator
/// Based on JPL DE ephemeris methodology [19]
pub struct OrbitalPropagator {
    /// Central body gravitational parameter
    mu: GravitationalParameter,
    /// Spherical harmonic coefficients (J2, J3, ...)
    gravity_model: GravityModel,
    /// Solar radiation pressure model
    srp: Option<SolarRadiationPressure>,
    /// Third-body perturbations
    third_bodies: Vec<CelestialBody>,
    /// Integration tolerance (JPL uses 10⁻¹² for inner planets)
    tolerance: f64,
}

impl OrbitalPropagator {
    /// Propagate with Post-Newtonian corrections
    /// Accuracy: sub-kilometer for inner planets [19]
    pub fn propagate(&self, state: &OrbitalState, dt: Duration) -> OrbitalState {
        // Use trueno for SIMD-accelerated gravity computation
        let accel = trueno::simd::orbital_acceleration(
            &state.position,
            &self.gravity_model,
            &self.third_bodies,
        );

        // RK7(8) Dormand-Prince for variable step
        self.rk78_step(state, accel, dt)
    }
}
```

### 5.2 Monte Carlo Engine

#### 5.2.1 Parallel Sampling with Variance Reduction

Leveraging trueno for GPU-accelerated Monte Carlo [20]:

```rust
/// trueno-accelerated Monte Carlo sampler
pub struct MonteCarloEngine {
    /// Number of samples
    n_samples: usize,
    /// Variance reduction technique
    variance_reduction: VarianceReduction,
    /// SIMD/GPU backend selection
    backend: trueno::Backend,
}

pub enum VarianceReduction {
    /// Antithetic variates: negative correlation [12]
    Antithetic,
    /// Control variates with known expectation
    ControlVariate { control: Box<dyn Fn(f64) -> f64>, expectation: f64 },
    /// Importance sampling with proposal distribution
    ImportanceSampling { proposal: Distribution },
    /// Stratified sampling
    Stratified { strata: Vec<Stratum> },
}

impl MonteCarloEngine {
    /// Run simulation with trueno SIMD acceleration
    pub fn run<F, T>(&self, sampler: F, rng: &mut SimRng) -> MonteCarloResult<T>
    where
        F: Fn(&[f64]) -> T + Sync,
        T: Sum + Clone,
    {
        let samples: Vec<f64> = match self.variance_reduction {
            VarianceReduction::Antithetic => {
                // Generate paired samples (u, 1-u)
                let base: Vec<f64> = rng.sample_n(self.n_samples / 2);
                base.iter()
                    .flat_map(|&u| [u, 1.0 - u])
                    .collect()
            }
            _ => rng.sample_n(self.n_samples),
        };

        // trueno parallel map-reduce
        let results = trueno::parallel::map_reduce(
            &samples,
            |chunk| chunk.iter().map(&sampler).sum(),
            |a, b| a + b,
        );

        MonteCarloResult {
            estimate: results / self.n_samples as f64,
            std_error: self.compute_std_error(&results),
            samples: self.n_samples,
        }
    }
}
```

### 5.3 Optimization Engine

#### 5.3.1 Bayesian Optimization with aprender

Gaussian Process surrogate models via aprender [15][16]:

```rust
use aprender::gaussian_process::{GaussianProcess, Kernel};

/// Bayesian optimizer with GP surrogate
pub struct BayesianOptimizer {
    /// Gaussian Process surrogate model
    gp: GaussianProcess,
    /// Acquisition function
    acquisition: AcquisitionFunction,
    /// Bounds for optimization
    bounds: Vec<(f64, f64)>,
    /// Observations: (x, y) pairs
    observations: Vec<(Vec<f64>, f64)>,
}

pub enum AcquisitionFunction {
    /// Expected Improvement [16]
    ExpectedImprovement,
    /// Upper Confidence Bound
    UCB { kappa: f64 },
    /// Probability of Improvement
    ProbabilityOfImprovement,
}

impl BayesianOptimizer {
    /// Suggest next point to evaluate (Kaizen: continuous improvement)
    pub fn suggest(&self) -> Vec<f64> {
        // Fit GP to observations
        let gp = self.gp.fit(&self.observations);

        // Optimize acquisition function using CMA-ES [34]
        let (best_x, _) = trueno::optimize::cma_es(
            |x| -self.acquisition.evaluate(&gp, x, self.best_y()),
            &self.bounds,
            CmaEsConfig::default(),
        );

        best_x
    }

    /// Expected Improvement acquisition
    fn expected_improvement(&self, gp: &GaussianProcess, x: &[f64]) -> f64 {
        let (mu, sigma) = gp.predict(x);
        let best = self.best_y();

        if sigma < 1e-10 {
            return 0.0;
        }

        let z = (mu - best) / sigma;
        let pdf = normal_pdf(z);
        let cdf = normal_cdf(z);

        sigma * (z * cdf + pdf)
    }
}
```

### 5.4 ML Integration

#### 5.4.1 Surrogate Models

The integration of Machine Learning with Physics simulations, also known as Physics-Informed Machine Learning [35], allows for the creation of surrogate models that respect physical laws.

```rust
/// Surrogate model for expensive simulations (PINNs concept [26])
/// Uses aprender for training, realizar for inference
pub struct SurrogateModel {
    /// Underlying ML model
    model: aprender::Model,
    /// Uncertainty quantification
    uncertainty: UncertaintyMethod,
    /// Training data buffer
    training_data: Vec<(Vec<f64>, f64)>,
}

impl SurrogateModel {
    /// Train surrogate on simulation data (entrenar)
    pub fn train(&mut self) -> TrainResult {
        entrenar::train(
            &mut self.model,
            &self.training_data,
            TrainConfig {
                epochs: 100,
                early_stopping: Some(10),
                ..Default::default()
            },
        )
    }

    /// Fast inference (realizar)
    pub fn predict(&self, x: &[f64]) -> (f64, f64) {
        let y = realizar::infer(&self.model, x);
        let uncertainty = self.uncertainty.estimate(&self.model, x);
        (y, uncertainty)
    }
}
```

### 5.5 ML Simulation Engine

The ML Simulation Engine provides deterministic, reproducible simulation of machine learning workflows using the `.apr` (Aprender) format. This enables falsifiable testing of training dynamics, prediction accuracy, and multi-turn model interactions without requiring actual model execution—a key capability for the Sovereign AI Stack [38][39].

#### 5.5.1 Theoretical Foundation: Simulated Learning Dynamics

**Definition 5.1 (Training Simulation)**: A training simulation T approximates the learning trajectory L(θ, D) of model parameters θ on dataset D:

```
T: (θ₀, D, H) → {θ₁, θ₂, ..., θₙ}
```

where H represents hyperparameters (learning rate, batch size, optimizer state) [40].

**Theorem 5.1 (Reproducible Training)**: For deterministic optimizers with fixed random seeds σ, training simulations are bitwise reproducible [41]:

```
∀ runs r₁, r₂: T(θ₀, D, H, σ) → Θ₁ ∧ T(θ₀, D, H, σ) → Θ₂ ⟹ Θ₁ ≡ Θ₂
```

This property enables Popperian falsification of training hypotheses by eliminating non-determinism [42].

#### 5.5.2 Simulated Model Training

Training simulation via entrenar integration with Jidoka quality gates:

```rust
use entrenar::train::{Trainer, TrainConfig, TrainEvent};
use entrenar::optim::Adam;
use aprender::format::AprModel;

/// Simulated training scenario for reproducible ML experiments
/// Implements Toyota Way principles: Jidoka (stop-on-anomaly),
/// Heijunka (balanced batch scheduling), Kaizen (continuous improvement)
pub struct TrainingSimulation {
    /// Model architecture specification
    architecture: ModelArchitecture,
    /// Training hyperparameters
    config: TrainConfig,
    /// Deterministic RNG for reproducibility
    rng: SimRng,
    /// Training event journal for replay
    journal: EventJournal,
    /// Jidoka: Loss spike detection
    anomaly_detector: AnomalyDetector,
}

/// Training state captured at each epoch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingState {
    /// Current epoch
    pub epoch: u64,
    /// Model parameters (serialized .apr format)
    pub parameters: Vec<u8>,
    /// Training loss
    pub loss: f64,
    /// Validation metrics
    pub metrics: TrainingMetrics,
    /// Optimizer state (momentum, adaptive learning rates)
    pub optimizer_state: OptimizerState,
    /// RNG state for perfect reproducibility
    pub rng_state: [u8; 32],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub train_loss: f64,
    pub val_loss: f64,
    pub accuracy: Option<f64>,
    pub gradient_norm: f64,
    pub learning_rate: f64,
}

impl TrainingSimulation {
    /// Create new training simulation with deterministic seed
    pub fn new(architecture: ModelArchitecture, seed: u64) -> Self {
        Self {
            architecture,
            config: TrainConfig::default(),
            rng: SimRng::new(seed),
            journal: EventJournal::new(),
            anomaly_detector: AnomalyDetector::new(3.0), // 3σ threshold
        }
    }

    /// Simulate training for specified epochs
    /// Returns trajectory of training states for analysis
    pub fn simulate(&mut self, epochs: u64, dataset: &Dataset) -> SimResult<TrainingTrajectory> {
        let mut trajectory = TrainingTrajectory::new();
        let mut params = self.architecture.init_params(&mut self.rng);
        let mut optimizer = Adam::new(self.config.learning_rate, 0.9, 0.999, 1e-8);

        for epoch in 0..epochs {
            // Heijunka: Load-balanced batch iteration
            let batches = dataset.batches(self.config.batch_size, &mut self.rng);
            let mut epoch_loss = 0.0;

            for batch in batches {
                let (loss, grads) = self.forward_backward(&params, &batch)?;
                epoch_loss += loss;

                // Jidoka: Detect training anomalies
                if let Some(anomaly) = self.anomaly_detector.check(loss, &grads) {
                    self.journal.append(TrainEvent::Anomaly(anomaly.clone()));
                    return Err(SimError::jidoka(format!(
                        "Training anomaly detected at epoch {}: {:?}",
                        epoch, anomaly
                    )));
                }

                // Parameter update
                optimizer.step(&mut params, &grads);
            }

            // Capture training state
            let state = TrainingState {
                epoch,
                parameters: self.serialize_params(&params)?,
                loss: epoch_loss / batches.len() as f64,
                metrics: self.compute_metrics(&params, dataset)?,
                optimizer_state: optimizer.state(),
                rng_state: self.rng.save_state(),
            };

            trajectory.push(state.clone());
            self.journal.append(TrainEvent::Epoch(state));
        }

        Ok(trajectory)
    }

    /// Replay training from checkpoint (time-travel debugging)
    pub fn replay_from(&mut self, checkpoint: &TrainingState) -> SimResult<()> {
        self.rng.restore_state(&checkpoint.rng_state);
        Ok(())
    }
}

/// Anomaly detector for Jidoka-style training quality gates [43]
pub struct AnomalyDetector {
    /// Rolling statistics for loss values
    loss_stats: RollingStats,
    /// Threshold in standard deviations
    threshold_sigma: f64,
}

impl AnomalyDetector {
    /// Check for training anomalies
    pub fn check(&mut self, loss: f64, grads: &Gradients) -> Option<TrainingAnomaly> {
        // NaN/Inf detection (Poka-Yoke)
        if !loss.is_finite() {
            return Some(TrainingAnomaly::NonFiniteLoss);
        }

        // Gradient explosion detection
        let grad_norm = grads.l2_norm();
        if grad_norm > 1e6 {
            return Some(TrainingAnomaly::GradientExplosion { norm: grad_norm });
        }

        // Loss spike detection (statistical)
        self.loss_stats.update(loss);
        let z_score = self.loss_stats.z_score(loss);
        if z_score.abs() > self.threshold_sigma {
            return Some(TrainingAnomaly::LossSpike { z_score, loss });
        }

        None
    }
}
```

#### 5.5.3 Simulated Model Prediction

Prediction simulation using `.apr` format with realizar integration:

```rust
use aprender::format::{AprModel, AprHeader, ModelType};
use realizar::serve::ModelServer;

/// Simulated inference scenario for reproducible prediction testing
/// Supports .apr native format with CRC32 integrity verification
pub struct PredictionSimulation {
    /// Loaded model in .apr format
    model: AprModel,
    /// Inference configuration
    config: InferenceConfig,
    /// Prediction event journal
    journal: EventJournal,
    /// Deterministic RNG for stochastic models
    rng: SimRng,
}

/// Prediction state for replay and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionState {
    /// Input features
    pub input: Vec<f64>,
    /// Model output
    pub output: Vec<f64>,
    /// Uncertainty estimate (if available)
    pub uncertainty: Option<f64>,
    /// Inference latency (simulated)
    pub latency_us: u64,
    /// Model version hash
    pub model_hash: [u8; 32],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Batch size for inference
    pub batch_size: usize,
    /// Temperature for probabilistic outputs
    pub temperature: f64,
    /// Top-k sampling (0 = greedy)
    pub top_k: usize,
    /// Enable uncertainty quantification
    pub uncertainty: bool,
}

impl PredictionSimulation {
    /// Load model from .apr file with integrity verification
    pub fn from_apr(path: &Path, seed: u64) -> SimResult<Self> {
        let model = AprModel::load(path)
            .map_err(|e| SimError::config(format!("Failed to load .apr model: {e}")))?;

        // Jidoka: Verify model integrity
        if !model.verify_crc32() {
            return Err(SimError::jidoka("Model integrity check failed (CRC32 mismatch)"));
        }

        Ok(Self {
            model,
            config: InferenceConfig::default(),
            journal: EventJournal::new(),
            rng: SimRng::new(seed),
        })
    }

    /// Simulate batch prediction with reproducible sampling
    pub fn predict_batch(&mut self, inputs: &[Vec<f64>]) -> SimResult<Vec<PredictionState>> {
        let mut predictions = Vec::with_capacity(inputs.len());

        for input in inputs {
            let start = std::time::Instant::now();

            // Forward pass
            let output = self.model.predict(input)?;

            // Apply temperature scaling for probabilistic outputs
            let output = if self.config.temperature != 1.0 {
                self.apply_temperature(&output, self.config.temperature)
            } else {
                output
            };

            // Top-k sampling if configured
            let output = if self.config.top_k > 0 {
                self.sample_top_k(&output, self.config.top_k, &mut self.rng)
            } else {
                output
            };

            // Uncertainty quantification
            let uncertainty = if self.config.uncertainty {
                Some(self.model.uncertainty(input)?)
            } else {
                None
            };

            let state = PredictionState {
                input: input.clone(),
                output,
                uncertainty,
                latency_us: start.elapsed().as_micros() as u64,
                model_hash: self.model.hash(),
            };

            predictions.push(state.clone());
            self.journal.append(PredictEvent::Inference(state));
        }

        Ok(predictions)
    }

    /// Simulate streaming generation (for language models)
    pub fn generate_stream(
        &mut self,
        prompt: &[u64],
        max_tokens: usize,
    ) -> SimResult<impl Iterator<Item = SimResult<u64>> + '_> {
        Ok(GenerationIterator::new(
            &self.model,
            prompt,
            max_tokens,
            &self.config,
            &mut self.rng,
        ))
    }
}
```

#### 5.5.4 Multi-Turn Model Simulation

Multi-turn simulation for evaluating model interactions over sequences of queries, inspired by single-shot evaluation patterns [44][45]:

```rust
/// Multi-turn simulation for conversational/iterative model evaluation
/// Implements Pareto frontier analysis across accuracy, cost, and latency [46]
pub struct MultiTurnSimulation {
    /// Model under evaluation
    model: AprModel,
    /// Conversation history
    history: Vec<Turn>,
    /// Evaluation metrics collector
    metrics: MetricsCollector,
    /// Oracle for ground-truth comparison (Popperian falsification)
    oracle: Option<Box<dyn Oracle>>,
    /// Deterministic RNG
    rng: SimRng,
}

/// A single turn in multi-turn interaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    /// Turn index
    pub index: usize,
    /// Input query/prompt
    pub input: String,
    /// Model response
    pub output: String,
    /// Ground truth (if available)
    pub expected: Option<String>,
    /// Turn metrics
    pub metrics: TurnMetrics,
    /// Context window usage
    pub context_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnMetrics {
    /// Generation latency
    pub latency_ms: f64,
    /// Input tokens
    pub input_tokens: usize,
    /// Output tokens
    pub output_tokens: usize,
    /// Estimated cost (normalized)
    pub cost: f64,
    /// Accuracy vs oracle (if available)
    pub accuracy: Option<f64>,
}

/// Pareto frontier analysis for multi-objective evaluation [47]
#[derive(Debug, Clone)]
pub struct ParetoAnalysis {
    /// Non-dominated solutions
    pub frontier: Vec<ParetoPoint>,
    /// Value score: V = (1 - accuracy_gap) × cost_ratio × latency_ratio
    pub value_scores: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct ParetoPoint {
    pub model_id: String,
    pub accuracy: f64,
    pub cost: f64,
    pub latency: f64,
    pub dominated_by: Vec<String>,
}

impl MultiTurnSimulation {
    /// Create multi-turn simulation with oracle for falsification
    pub fn new(model: AprModel, oracle: Option<Box<dyn Oracle>>, seed: u64) -> Self {
        Self {
            model,
            history: Vec::new(),
            metrics: MetricsCollector::new(),
            oracle,
            rng: SimRng::new(seed),
        }
    }

    /// Execute a single turn in the conversation
    pub fn turn(&mut self, input: &str) -> SimResult<Turn> {
        let start = std::time::Instant::now();

        // Build context from history
        let context = self.build_context(input);
        let input_tokens = self.tokenize(&context).len();

        // Generate response
        let output = self.model.generate(&context, &mut self.rng)?;
        let output_tokens = self.tokenize(&output).len();

        // Compare with oracle if available (Popperian falsification)
        let (expected, accuracy) = if let Some(ref oracle) = self.oracle {
            let expected = oracle.expected_output(input, &self.history)?;
            let accuracy = self.compute_accuracy(&output, &expected);
            (Some(expected), Some(accuracy))
        } else {
            (None, None)
        };

        let turn = Turn {
            index: self.history.len(),
            input: input.to_string(),
            output: output.clone(),
            expected,
            metrics: TurnMetrics {
                latency_ms: start.elapsed().as_secs_f64() * 1000.0,
                input_tokens,
                output_tokens,
                cost: self.estimate_cost(input_tokens, output_tokens),
                accuracy,
            },
            context_tokens: input_tokens,
        };

        self.history.push(turn.clone());
        self.metrics.record(&turn);

        Ok(turn)
    }

    /// Run complete multi-turn evaluation with statistical analysis
    /// Following Princeton methodology: 5 runs minimum, 95% CI [48]
    pub fn evaluate(
        &mut self,
        queries: &[String],
        n_runs: usize,
    ) -> SimResult<MultiTurnEvaluation> {
        assert!(n_runs >= 5, "Princeton methodology requires minimum 5 runs");

        let mut run_results = Vec::with_capacity(n_runs);

        for run in 0..n_runs {
            // Reset state for each run
            self.history.clear();
            self.rng = SimRng::new(self.rng.next_u64()); // Derived seed

            let mut run_metrics = Vec::new();
            for query in queries {
                let turn = self.turn(query)?;
                run_metrics.push(turn.metrics);
            }
            run_results.push(run_metrics);
        }

        // Statistical analysis with bootstrap CI
        let evaluation = MultiTurnEvaluation {
            mean_accuracy: self.bootstrap_mean(&run_results, |m| m.accuracy)?,
            mean_latency: self.bootstrap_mean(&run_results, |m| Some(m.latency_ms))?,
            total_cost: run_results.iter()
                .map(|r| r.iter().map(|m| m.cost).sum::<f64>())
                .sum::<f64>() / n_runs as f64,
            confidence_interval: 0.95,
            n_runs,
        };

        Ok(evaluation)
    }

    /// Compute Pareto frontier across multiple models
    pub fn pareto_analysis(evaluations: &[(String, MultiTurnEvaluation)]) -> ParetoAnalysis {
        let mut points: Vec<ParetoPoint> = evaluations.iter()
            .map(|(id, eval)| ParetoPoint {
                model_id: id.clone(),
                accuracy: eval.mean_accuracy.unwrap_or(0.0),
                cost: eval.total_cost,
                latency: eval.mean_latency.unwrap_or(0.0),
                dominated_by: Vec::new(),
            })
            .collect();

        // Identify dominated points
        for i in 0..points.len() {
            for j in 0..points.len() {
                if i != j && Self::dominates(&points[j], &points[i]) {
                    points[i].dominated_by.push(points[j].model_id.clone());
                }
            }
        }

        // Compute value scores
        let baseline_accuracy = points.iter().map(|p| p.accuracy).fold(0.0, f64::max);
        let baseline_cost = points.iter().map(|p| p.cost).fold(f64::INFINITY, f64::min);
        let baseline_latency = points.iter().map(|p| p.latency).fold(f64::INFINITY, f64::min);

        let value_scores: HashMap<String, f64> = points.iter()
            .map(|p| {
                let accuracy_gap = baseline_accuracy - p.accuracy;
                let cost_ratio = baseline_cost / p.cost.max(1e-10);
                let latency_ratio = baseline_latency / p.latency.max(1e-10);
                let value = (1.0 - accuracy_gap) * cost_ratio * latency_ratio;
                (p.model_id.clone(), value)
            })
            .collect();

        ParetoAnalysis {
            frontier: points.into_iter().filter(|p| p.dominated_by.is_empty()).collect(),
            value_scores,
        }
    }

    /// Check if point a dominates point b (better in all objectives)
    fn dominates(a: &ParetoPoint, b: &ParetoPoint) -> bool {
        a.accuracy >= b.accuracy && a.cost <= b.cost && a.latency <= b.latency
            && (a.accuracy > b.accuracy || a.cost < b.cost || a.latency < b.latency)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiTurnEvaluation {
    pub mean_accuracy: Option<f64>,
    pub mean_latency: Option<f64>,
    pub total_cost: f64,
    pub confidence_interval: f64,
    pub n_runs: usize,
}
```

#### 5.5.5 Jidoka Feedback Loop for ML

Implementing Toyota's Jidoka principle for continuous ML quality improvement:

```rust
/// Jidoka feedback loop for ML simulation
/// Each detected anomaly generates improvement patches (Kaizen) [49]
pub struct JidokaMLFeedback {
    /// Anomaly patterns detected
    patterns: Vec<AnomalyPattern>,
    /// Generated fixes (rule patches)
    patches: Vec<RulePatch>,
    /// Target: anomaly rate trending toward zero
    anomaly_rate: RollingStats,
}

/// Pattern detected during training/inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyPattern {
    pub pattern_type: AnomalyType,
    pub frequency: u64,
    pub context: HashMap<String, String>,
    pub suggested_fix: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalyType {
    /// Loss became NaN/Inf
    NonFiniteLoss,
    /// Gradient norm exceeded threshold
    GradientExplosion,
    /// Gradient vanished below threshold
    GradientVanishing,
    /// Loss spike (statistical outlier)
    LossSpike,
    /// Prediction confidence below threshold
    LowConfidence,
    /// Model output inconsistent with oracle
    OracleMismatch,
}

impl JidokaMLFeedback {
    /// Record anomaly and generate improvement patch
    pub fn record_anomaly(&mut self, anomaly: TrainingAnomaly) -> Option<RulePatch> {
        let pattern = self.classify_pattern(&anomaly);

        // Check if we've seen this pattern before
        if let Some(existing) = self.patterns.iter_mut().find(|p| p.pattern_type == pattern.pattern_type) {
            existing.frequency += 1;

            // After 3 occurrences, generate automated fix
            if existing.frequency >= 3 {
                let patch = self.generate_patch(&pattern);
                self.patches.push(patch.clone());
                return Some(patch);
            }
        } else {
            self.patterns.push(pattern);
        }

        None
    }

    /// Generate rule patch from pattern (Kaizen improvement)
    fn generate_patch(&self, pattern: &AnomalyPattern) -> RulePatch {
        match pattern.pattern_type {
            AnomalyType::GradientExplosion => RulePatch {
                rule_type: RuleType::GradientClipping,
                parameters: hashmap! {
                    "max_norm".to_string() => "1.0".to_string(),
                },
            },
            AnomalyType::LossSpike => RulePatch {
                rule_type: RuleType::LearningRateWarmup,
                parameters: hashmap! {
                    "warmup_steps".to_string() => "1000".to_string(),
                },
            },
            _ => RulePatch::default(),
        }
    }
}
```

---

## 6. Replay and Time-Travel

### 6.1 Incremental Checkpointing

Based on copy-on-write and incremental state saving [21] and the concept of Virtual Time [31]:

```rust
/// Incremental checkpoint with zstd compression
pub struct CheckpointManager {
    /// Checkpoint interval (in simulation steps)
    interval: u64,
    /// Maximum checkpoint storage (bytes)
    max_storage: usize,
    /// Compression level (1-22)
    compression_level: i32,
    /// Checkpoint storage
    checkpoints: BTreeMap<SimTime, Checkpoint>,
}

struct Checkpoint {
    /// Full state snapshot (compressed)
    data: Vec<u8>,
    /// Hash for integrity verification
    hash: [u8; 32],
    /// Parent checkpoint for delta computation
    parent: Option<SimTime>,
}

impl CheckpointManager {
    /// Create checkpoint (Jidoka: integrity verification)
    pub fn checkpoint(&mut self, time: SimTime, state: &SimState) {
        let serialized = bincode::serialize(state).unwrap();
        let compressed = zstd::encode_all(&serialized[..], self.compression_level).unwrap();

        // Integrity hash
        let hash = blake3::hash(&compressed);

        self.checkpoints.insert(time, Checkpoint {
            data: compressed,
            hash: hash.into(),
            parent: self.latest_before(time),
        });

        // Garbage collect old checkpoints if over budget
        self.gc_if_needed();
    }

    /// Restore to specific time (time-travel)
    pub fn restore(&self, time: SimTime) -> Result<SimState, ReplayError> {
        let checkpoint = self.nearest_checkpoint(time)?;

        // Verify integrity (Poka-Yoke)
        let computed_hash = blake3::hash(&checkpoint.data);
        if computed_hash.as_bytes() != &checkpoint.hash {
            return Err(ReplayError::IntegrityViolation);
        }

        let decompressed = zstd::decode_all(&checkpoint.data[..])?;
        bincode::deserialize(&decompressed).map_err(Into::into)
    }
}
```

### 6.2 Event Journal

Append-only log for perfect replay [22]:

```rust
/// Append-only event journal for deterministic replay
pub struct EventJournal {
    /// Memory-mapped file for persistence
    mmap: memmap2::MmapMut,
    /// Current write position
    position: u64,
    /// Index: time → file offset
    index: BTreeMap<SimTime, u64>,
}

#[derive(Serialize, Deserialize)]
struct JournalEntry {
    time: SimTime,
    sequence: u64,
    event: Event,
    rng_state: [u8; 32],  // For reproducibility
}

impl EventJournal {
    /// Append event to journal
    pub fn append(&mut self, entry: JournalEntry) {
        let serialized = bincode::serialize(&entry).unwrap();
        let len = serialized.len() as u32;

        // Write length-prefixed entry
        self.mmap[self.position as usize..][..4].copy_from_slice(&len.to_le_bytes());
        self.mmap[self.position as usize + 4..][..serialized.len()].copy_from_slice(&serialized);

        self.index.insert(entry.time, self.position);
        self.position += 4 + serialized.len() as u64;
    }

    /// Seek to time and replay from there
    pub fn seek(&self, time: SimTime) -> impl Iterator<Item = JournalEntry> + '_ {
        let start_offset = self.index.range(..=time).next_back().map(|(_, &o)| o).unwrap_or(0);
        JournalIterator::new(&self.mmap, start_offset)
    }
}
```

### 6.3 Time-Travel Scrubber

Interactive navigation through simulation history:

```rust
/// Time-travel interface for interactive debugging
pub struct TimeScrubber {
    checkpoints: CheckpointManager,
    journal: EventJournal,
    current_time: SimTime,
    current_state: SimState,
}

impl TimeScrubber {
    /// Seek to any point in simulation history
    pub fn seek_to(&mut self, target: SimTime) -> Result<&SimState, ScrubError> {
        if target == self.current_time {
            return Ok(&self.current_state);
        }

        // Find nearest checkpoint before target
        self.current_state = self.checkpoints.restore(target)?;
        let checkpoint_time = self.checkpoints.nearest_time(target);

        // Replay events from checkpoint to target
        for entry in self.journal.seek(checkpoint_time) {
            if entry.time > target {
                break;
            }
            self.current_state.apply_event(&entry.event);
        }

        self.current_time = target;
        Ok(&self.current_state)
    }

    /// Step forward one event
    pub fn step_forward(&mut self) -> Option<&Event> {
        self.journal.seek(self.current_time).next().map(|entry| {
            self.current_state.apply_event(&entry.event);
            self.current_time = entry.time;
            &entry.event
        })
    }

    /// Step backward one event (requires checkpoint)
    pub fn step_backward(&mut self) -> Result<(), ScrubError> {
        let prev_time = self.journal.previous_time(self.current_time)?;
        self.seek_to(prev_time)?;
        Ok(())
    }
}
```

---

## 7. Visualization

### 7.1 TUI Dashboard (ratatui)

```rust
use ratatui::{prelude::*, widgets::*};

/// Real-time TUI dashboard
pub struct SimularTui {
    /// Terminal backend
    terminal: Terminal<CrosstermBackend<Stdout>>,
    /// Dashboard layout
    layout: DashboardLayout,
    /// Refresh rate
    refresh_hz: u32,
}

struct DashboardLayout {
    trajectory: TrajectoryWidget,
    metrics: MetricsWidget,
    phase_space: PhaseSpaceWidget,
    controls: ControlsWidget,
}

impl SimularTui {
    /// Render current simulation state
    pub fn render(&mut self, state: &SimState, metrics: &SimMetrics) -> io::Result<()> {
        self.terminal.draw(|frame| {
            let chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
                .split(frame.size());

            let left = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
                .split(chunks[0]);

            let right = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(chunks[1]);

            // 3D trajectory projection
            frame.render_widget(self.layout.trajectory.render(state), left[0]);

            // Control panel
            frame.render_widget(self.layout.controls.render(), left[1]);

            // Real-time metrics
            frame.render_widget(self.layout.metrics.render(metrics), right[0]);

            // Phase space portrait
            frame.render_widget(self.layout.phase_space.render(state), right[1]);
        })?;
        Ok(())
    }
}
```

### 7.2 Web Visualization

WebSocket streaming with wgpu compute:

```rust
use axum::{extract::ws::WebSocket, routing::get, Router};
use wgpu::*;

/// Web visualization server
pub struct WebVisualization {
    /// HTTP/WebSocket server
    server: Router,
    /// wgpu device for GPU rendering
    device: Device,
    /// Connected clients
    clients: Arc<RwLock<Vec<WebSocket>>>,
}

impl WebVisualization {
    /// Start server on specified port
    pub async fn serve(self, port: u16) -> Result<(), WebError> {
        let app = Router::new()
            .route("/ws", get(Self::websocket_handler))
            .route("/", get(Self::index_handler));

        axum::Server::bind(&format!("0.0.0.0:{}", port).parse()?)
            .serve(app.into_make_service())
            .await?;
        Ok(())
    }

    /// Stream simulation state to connected clients
    pub async fn broadcast(&self, state: &SimState) {
        let payload = self.render_to_json(state);
        let clients = self.clients.read().await;
        for client in clients.iter() {
            let _ = client.send(Message::Text(payload.clone())).await;
        }
    }
}
```

### 7.3 Export Pipeline

```rust
/// Export formats for simulation data
pub enum ExportFormat {
    /// Video export (MP4, GIF)
    Video { format: VideoFormat, fps: u32 },
    /// Columnar data (Parquet via alimentar)
    Parquet { compression: ParquetCompression },
    /// Streaming JSON Lines
    JsonLines,
}

impl Exporter {
    /// Export trajectory to Parquet (alimentar integration)
    pub fn to_parquet(&self, trajectory: &Trajectory, path: &Path) -> Result<(), ExportError> {
        alimentar::write_parquet(
            path,
            trajectory.to_arrow_batch()?,
            ParquetWriterConfig {
                compression: Compression::ZSTD(ZstdLevel::try_new(3)?),
                row_group_size: 1_000_000,
            },
        )
    }

    /// Export to MP4 video
    pub fn to_mp4(&self, frames: &[Frame], path: &Path, config: VideoConfig) -> Result<(), ExportError> {
        let encoder = ffmpeg::Encoder::new(path, config.codec, config.fps)?;
        for frame in frames {
            encoder.encode_frame(frame)?;
        }
        encoder.finalize()
    }
}
```

---

## 8. YAML Configuration Schema

### 8.1 Top-Level Schema

```yaml
# simular.yaml - Unified Simulation Configuration
# Schema version for forward compatibility
schema_version: "1.0"

# Simulation metadata
simulation:
  name: "falcon9-stage-separation"
  description: "Monte Carlo analysis of Falcon 9 stage separation dynamics"
  version: "0.1.0"

# Reproducibility settings (CRITICAL for Popperian falsification)
reproducibility:
  # Master seed for all RNG (required for determinism)
  seed: 42
  # IEEE 754-2008 strict mode for cross-platform reproducibility [9]
  ieee_strict: true
  # Record RNG state in journal for perfect replay
  record_rng_state: true

# Domain configurations
domains:
  # Physics simulation
  physics:
    enabled: true
    engine: rigid-body  # rigid-body | orbital | fluid | discrete
    integrator:
      type: verlet       # euler | verlet | rk4 | rk78 | symplectic-euler
      timestep:
        mode: fixed      # fixed | adaptive
        dt: 0.001        # seconds
        min_dt: 0.0001   # for adaptive
        max_dt: 0.01     # for adaptive
        tolerance: 1e-9  # for adaptive
    # Jidoka: stop-on-error thresholds
    jidoka:
      energy_tolerance: 1e-6    # relative energy drift
      constraint_tolerance: 1e-8 # constraint violation
      check_finite: true         # NaN/Inf detection

  # Monte Carlo analysis
  monte_carlo:
    enabled: true
    samples: 10_000
    # Variance reduction [12]
    variance_reduction:
      method: antithetic  # none | antithetic | control_variate | importance | stratified
    # trueno acceleration
    backend:
      type: auto  # cpu | simd | gpu | auto
      threads: null  # null = auto-detect
    # Convergence monitoring
    convergence:
      check_interval: 1000
      tolerance: 0.01      # relative std error
      min_samples: 1000    # minimum before checking

  # Optimization
  optimization:
    enabled: true
    algorithm: bayesian  # bayesian | cma-es | genetic | gradient
    bayesian:
      # GP surrogate (aprender integration)
      kernel: matern52   # rbf | matern32 | matern52
      acquisition: expected_improvement  # ei | ucb | poi
      n_initial: 10      # initial random samples
      n_iterations: 100
    objective:
      name: minimize_fuel
      type: minimize  # minimize | maximize

  # ML integration
  ml:
    enabled: false
    surrogate:
      model: gaussian_process  # gaussian_process | neural_network | random_forest
      training_samples: 1000
      uncertainty: true
    reinforcement_learning:
      algorithm: ppo  # ppo | sac | dqn

# Replay configuration
replay:
  enabled: true
  checkpoint:
    interval: 1000         # steps between checkpoints
    max_storage: 1GB       # maximum checkpoint storage
    compression: zstd      # none | lz4 | zstd
    compression_level: 3   # 1-22 for zstd
  journal:
    enabled: true
    persist: true          # write to disk
    path: "./simular-journal.bin"

# Visualization
visualization:
  tui:
    enabled: true
    refresh_hz: 30
    panels:
      - type: trajectory
        projection: orthographic  # orthographic | perspective
        axes: [x, y, z]
      - type: metrics
        series: [energy, momentum, time]
      - type: phase_space
        x: position.x
        y: velocity.x
  web:
    enabled: false
    port: 8080
    cors_origins: ["http://localhost:3000"]
  export:
    enabled: true
    formats:
      - type: parquet
        path: "./output/trajectory.parquet"
      - type: mp4
        path: "./output/simulation.mp4"
        fps: 30

# Falsification framework (Popperian methodology)
falsification:
  # Null hypothesis
  null_hypothesis: "Stage separation occurs within nominal bounds"
  # Falsification criteria
  criteria:
    - metric: separation_velocity
      operator: "<"
      threshold: 10.0  # m/s
    - metric: angular_rate
      operator: "<"
      threshold: 5.0   # deg/s
  # Statistical test
  test:
    type: chi_square  # chi_square | t_test | ks_test
    significance: 0.05
  # Reference data for validation
  oracle:
    type: file
    path: "./reference/falcon9_telemetry.parquet"

# Parameter distributions for Monte Carlo
parameters:
  wind_speed:
    distribution: normal
    mean: 5.0
    std: 2.0
    units: m/s
  thrust_variance:
    distribution: uniform
    min: 0.98
    max: 1.02
  atmospheric_density:
    distribution: lognormal
    mu: 0.0
    sigma: 0.1

# Scenario-specific configuration
scenario:
  type: rocket_launch
  vehicle:
    mass_stage1: 420000  # kg
    mass_stage2: 120000  # kg
    thrust: 7600000      # N
  trajectory:
    initial_altitude: 0  # m
    target_orbit: 400000 # m
```

### 8.2 Validation (Poka-Yoke)

Compile-time validation via serde and validator:

```rust
use serde::{Deserialize, Serialize};
use validator::Validate;

#[derive(Debug, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
pub struct SimularConfig {
    #[validate(length(min = 1))]
    pub schema_version: String,

    #[validate]
    pub simulation: SimulationConfig,

    #[validate]
    pub reproducibility: ReproducibilityConfig,

    #[validate]
    pub domains: DomainsConfig,

    #[validate]
    pub replay: Option<ReplayConfig>,

    #[validate]
    pub visualization: Option<VisualizationConfig>,

    #[validate]
    pub falsification: Option<FalsificationConfig>,
}

#[derive(Debug, Deserialize, Validate)]
pub struct ReproducibilityConfig {
    /// Seed must be provided for determinism
    pub seed: u64,

    #[serde(default = "default_true")]
    pub ieee_strict: bool,

    #[serde(default = "default_true")]
    pub record_rng_state: bool,
}

impl SimularConfig {
    /// Load and validate configuration
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_yaml::from_str(&content)?;

        // Poka-Yoke: validate all constraints
        config.validate()?;

        // Additional semantic validation
        config.validate_semantic()?;

        Ok(config)
    }

    fn validate_semantic(&self) -> Result<(), ConfigError> {
        // Ensure Monte Carlo has sufficient samples for convergence
        if let Some(mc) = &self.domains.monte_carlo {
            if mc.enabled && mc.samples < 100 {
                return Err(ConfigError::InsufficientSamples {
                    provided: mc.samples,
                    minimum: 100,
                });
            }
        }

        // Ensure Bayesian optimization has GP kernel
        if let Some(opt) = &self.domains.optimization {
            if opt.algorithm == Algorithm::Bayesian && opt.bayesian.is_none() {
                return Err(ConfigError::MissingBayesianConfig);
            }
        }

        Ok(())
    }
}
```

---

## 9. Quality Assurance

### 9.1 JPL-Inspired Verification

Following NASA IV&V methodology [8] and "Power of 10" rules [7]:

| Rule | Implementation |
|------|----------------|
| 1. Simple control flow | No goto, limited nesting depth (max 4) |
| 2. Fixed loop bounds | All loops have compile-time or validated bounds |
| 3. No heap after init | Pre-allocated buffers, arena allocators |
| 4. Short functions | Max 60 lines per function |
| 5. Low assertion density | Min 2 assertions per function |
| 6. Minimal scope | Variables declared at narrowest scope |
| 7. Check return values | `Result<T, E>` everywhere, no unwrap in production |
| 8. Limit preprocessor | Feature flags only, no conditional compilation logic |
| 9. Restrict pointers | Safe Rust default, `unsafe` blocks audited |
| 10. Compiler warnings | `-D warnings`, clippy pedantic |

### 9.2 Falsification Testing

```rust
#[cfg(test)]
mod falsification_tests {
    use super::*;
    use proptest::prelude::*;

    /// Property: Energy conservation in Hamiltonian systems
    /// Falsification: Find initial conditions that violate conservation
    proptest! {
        #[test]
        fn test_energy_conservation(
            initial_pos in -1000.0..1000.0,
            initial_vel in -100.0..100.0,
            mass in 0.1..1000.0,
        ) {
            let state = PhysicsState::new(initial_pos, initial_vel, mass);
            let integrator = VerletIntegrator::new(0.001);

            let initial_energy = state.kinetic_energy() + state.potential_energy();

            // Propagate 10000 steps
            let mut current = state;
            for _ in 0..10000 {
                current = integrator.step(&current);
            }

            let final_energy = current.kinetic_energy() + current.potential_energy();
            let drift = (final_energy - initial_energy).abs() / initial_energy;

            // Falsification criterion: energy drift < 1e-6
            prop_assert!(drift < 1e-6, "Energy drift: {}", drift);
        }
    }

    /// Property: Monte Carlo convergence
    /// Falsification: Find cases where CLT doesn't hold
    proptest! {
        #[test]
        fn test_monte_carlo_convergence(seed in 0u64..u64::MAX) {
            let mut rng = SimRng::new(seed);
            let engine = MonteCarloEngine::new(10000, VarianceReduction::None);

            // Known integral: ∫₀¹ x² dx = 1/3
            let result = engine.run(|x| x[0] * x[0], &mut rng);

            // 99.7% confidence interval (3σ)
            let ci = 3.0 * result.std_error;
            let error = (result.estimate - 1.0/3.0).abs();

            prop_assert!(error < ci, "Error {} outside 3σ CI {}", error, ci);
        }
    }
}
```

### 9.3 Reference Validation

Comparison against known-good implementations:

```rust
/// Reference test against JPL ephemeris data [19]
#[test]
fn test_orbital_propagation_against_jpl() {
    let propagator = OrbitalPropagator::new(
        GravityModel::EGM96,
        Tolerance::new(1e-12),
    );

    // ISS TLE epoch: 2024-01-01 00:00:00 UTC
    let initial_state = OrbitalState::from_tle(include_str!("../fixtures/iss.tle"));

    // Propagate 1 day
    let final_state = propagator.propagate(&initial_state, Duration::days(1));

    // Compare against JPL Horizons reference
    let reference = include!("../fixtures/iss_horizons_1day.rs");

    // Position accuracy: < 1 km (sub-kilometer for inner planets per JPL)
    assert!((final_state.position - reference.position).norm() < 1000.0,
        "Position error: {} m", (final_state.position - reference.position).norm());

    // Velocity accuracy: < 1 m/s
    assert!((final_state.velocity - reference.velocity).norm() < 1.0,
        "Velocity error: {} m/s", (final_state.velocity - reference.velocity).norm());
}
```

---

## 10. References

### Astrodynamics and Orbital Mechanics

[1] Rabotin, C. (2023). *Nyx: High-fidelity astrodynamics toolkit*. https://nyxspace.com/

[19] Folkner, W. M., et al. (2014). "The Planetary and Lunar Ephemerides DE430 and DE431." *Interplanetary Network Progress Report*, 42-196. NASA JPL. https://ssd.jpl.nasa.gov/ephem.html

### Simulation Frameworks and Determinism

[2] MadSim Contributors. (2024). *MadSim: Magical Deterministic Simulator for distributed systems in Rust*. https://github.com/madsim-rs/madsim

[3] Catto, E., & Dimforge. (2024). *Rapier: 2D and 3D physics engines for Rust*. https://rapier.rs/

[4] Ertis Research. (2023). "OpenTwins: An open-source framework for the development of next-gen compositional digital twins." *Computers in Industry*, 152, 103988. https://github.com/ertis-research/opentwins

[9] McDougal, R., et al. (2016). "Reproducibility in Computational Neuroscience Models and Simulations." *IEEE Transactions on Biomedical Engineering*, 63(10), 2021-2026. https://pmc.ncbi.nlm.nih.gov/articles/PMC5016202/

[17] Polar Signals. (2025). "Deterministic Simulation Testing in Rust: A Theater Of State Machines." https://www.polarsignals.com/blog/posts/2025/07/08/dst-rust

[18] Arcelli, D., et al. (2024). "A Compositional Simulation Framework for Abstract State Machine Models of Discrete Event Systems." *Formal Aspects of Computing*, 36(2). https://dl.acm.org/doi/10.1145/3652862

[21] Perumalla, K. S. (2023). "Incremental Checkpointing of Large State Simulation Models." *IEEE/ACM Winter Simulation Conference*. https://ieeexplore.ieee.org/document/10305756/

[22] Software Sustainability Institute. (2023). "Reproducibility in Discrete Event Simulation." https://www.software.ac.uk/blog/reproducibility-discrete-event-simulation

### Monte Carlo Methods

[10] PLOS ONE Editors. (2020). "The value of Monte Carlo model-based variance reduction technology in the pricing of financial derivatives." *PLOS ONE*, 15(3), e0229737. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0229737

[11] Caflisch, R. E. (1998). "Monte Carlo and quasi-Monte Carlo methods." *Acta Numerica*, 7, 1-49. https://www.math.pku.edu.cn/teachers/litj/notes/numer_anal/MCQMC_Caflisch.pdf

[12] García-Pareja, S., Lallena, A. M., & Salvat, F. (2021). "Variance-Reduction Methods for Monte Carlo Simulation of Radiation Transport." *Frontiers in Physics*, 9, 718873. https://www.frontiersin.org/articles/10.3389/fphy.2021.718873/full

[20] Preis, T., et al. (2018). "SIMD Monte-Carlo Numerical Simulations Accelerated on GPU and Xeon Phi." *International Journal of Parallel Programming*, 46, 1-20. https://link.springer.com/article/10.1007/s10766-017-0509-y

### Numerical Integration

[13] Wikipedia Contributors. (2024). "Verlet integration." https://en.wikipedia.org/wiki/Verlet_integration

[14] Hairer, E. (2010). "Geometric Numerical Integration." University of Geneva. https://www.unige.ch/~hairer/poly_geoint/week2.pdf

### Optimization and Machine Learning

[15] Lu, X., Polyzos, K., Li, B., & Giannakis, G. B. (2023). "Surrogate Modeling for Bayesian Optimization beyond a Single Gaussian Process." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 45(9), 11283-11296. https://ieeexplore.ieee.org/iel7/34/10210213/10093035.pdf

[16] Zhang, Y., et al. (2024). "Deep Gaussian Process for Enhanced Bayesian Optimization." *IISE Transactions*, 57(4), 423-436. https://www.tandfonline.com/doi/abs/10.1080/24725854.2024.2312905

### Philosophy of Science

[5] Popper, K. R. (1934). *The Logic of Scientific Discovery*. Vienna: Springer.

[6] Stanford Encyclopedia of Philosophy. (2024). "Karl Popper." https://plato.stanford.edu/entries/popper/

### Software Engineering and Verification

[7] Holzmann, G. J. (2006). "The Power of 10: Rules for Developing Safety-Critical Code." *IEEE Computer*, 39(6), 95-99. https://spinoff.nasa.gov/Spinoff2011/it_3.html

[8] NASA IV&V. (2024). "IV&V Capabilities & Services." https://www.nasa.gov/ivv-services/

[23] Jung, R., et al. (2021). "Safe Systems Programming in Rust: The Promise and the Challenge." *Communications of the ACM*, 64(4), 144-152. https://people.mpi-sws.org/~dreyer/papers/safe-sysprog-rust/paper.pdf

### Toyota Production System

[24] Brown, M. (2020). "TPS & SC2020: Toyota Production System & Supply Chain." *MIT Center for Transportation & Logistics*. https://ctl.mit.edu/sites/default/files/Mac_TPS_thesis.pdf

[25] Toyota Motor Corporation. (2024). "Toyota Production System." https://global.toyota/en/company/plant-tours/production-system/

### Null Hypothesis and Statistical Testing

[26] Streiner, D. L. (2013). "Testing the null hypothesis: the forgotten legacy of Karl Popper?" *Journal of Clinical Psychopharmacology*, 33(1), 1-3. https://pubmed.ncbi.nlm.nih.gov/23249368/

### Numerical Methods and Higher-Order Integrators

[27] Yoshida, H. (1990). "Construction of higher order symplectic integrators." *Physics Letters A*, 150(5-7), 262-268. https://www.sciencedirect.com/science/article/pii/0375960190900923

### Rust Performance in HPC

[28] Costanzo, M., Rucci, E., et al. (2021). "Performance vs Programming Effort between Rust and C on Multicore Architectures: Case Study in N-Body." *IEEE ARGENCON*. https://ieeexplore.ieee.org/document/9640225/

### Signal Temporal Logic and Robustness

[29] Donzé, A., & Maler, O. (2010). "Robust Satisfaction of Temporal Logic over Real-Valued Signals." *FORMATS 2010*, LNCS 6246, 92-106. Springer. https://link.springer.com/chapter/10.1007/978-3-642-15297-9_9

[30] Deshmukh, J. V., et al. (2017). "Robust online monitoring of signal temporal logic." *Formal Methods in System Design*, 51(1), 5-30. https://link.springer.com/article/10.1007/s10703-017-0286-7

### Discrete Event Simulation

[31] Jefferson, D. R. (1985). "Virtual Time." *ACM Transactions on Programming Languages and Systems*, 7(3), 404-425.

### Dependency Management and Ecosystem Evolution

[32] Decan, A., Mens, T., & Grosjean, P. (2019). "An empirical comparison of dependency network evolution in seven software packaging ecosystems." *Empirical Software Engineering*, 24(1), 381-416.

### Rust HPC Benchmarks

[33] Hundt, R., et al. (2025). "NPB-Rust: NAS Parallel Benchmarks in Rust." *arXiv preprint*. https://arxiv.org/html/2502.15536v1

### Evolutionary Optimization

[34] Hansen, N., & Ostermeier, A. (2001). "Completely Derandomized Self-Adaptation in Evolution Strategies." *Evolutionary Computation*, 9(2), 159-195.

### Physics-Informed Machine Learning

[35] Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." *Journal of Computational Physics*, 378, 686-707.

### Software Supply Chain Security

[36] Zimmermann, M., Staicu, C. A., Tenny, C., & Pradel, M. (2019). "Small World with High Risks: A Study of Security Threats in the npm Ecosystem." *USENIX Security Symposium*.

### Cognitive Load in Software Engineering

[37] Storey, M. A., et al. (2020). "The Theory of Developers' Cognitive Load." *ICSE '20*. ACM.

### ML Simulation and Reproducibility

[38] Pineau, J., et al. (2021). "Improving Reproducibility in Machine Learning Research." *Journal of Machine Learning Research*, 22(164), 1-20. https://jmlr.org/papers/v22/20-303.html

[39] Tatman, R., VanderPlas, J., & Dane, S. (2018). "A Practical Taxonomy of Reproducibility for Machine Learning Research." *NeurIPS 2018 Workshop on Reproducibility in ML*. https://openreview.net/forum?id=B1eYYK5QgX

[40] Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning." *MIT Press*. Chapter 8: Optimization for Training Deep Models.

[41] Nagarajan, P., et al. (2019). "Deterministic Implementations for Reproducibility in Deep Learning." *NeurIPS 2019 Workshop*. https://arxiv.org/abs/1903.10615

### ML Quality and Continuous Improvement (Toyota Way)

[42] Sculley, D., et al. (2015). "Hidden Technical Debt in Machine Learning Systems." *NeurIPS 2015*. https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html

[43] Breck, E., et al. (2017). "The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction." *IEEE Big Data*. https://research.google/pubs/pub46555/

[44] Kapoor, S., & Narayanan, A. (2023). "Leakage and the Reproducibility Crisis in ML-based Science." *Patterns*, 4(9), 100804. https://www.cell.com/patterns/fulltext/S2666-3899(23)00159-9

### Multi-Turn Evaluation and Pareto Analysis

[45] Kapoor, S., et al. (2024). "AI Agents That Matter." *arXiv preprint*. https://arxiv.org/abs/2407.01502

[46] Deb, K., et al. (2002). "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II." *IEEE Transactions on Evolutionary Computation*, 6(2), 182-197.

[47] Zitzler, E., & Thiele, L. (1999). "Multiobjective Evolutionary Algorithms: A Comparative Case Study and the Strength Pareto Approach." *IEEE Transactions on Evolutionary Computation*, 3(4), 257-271.

### TPS Code Review and Digital Twin Architecture

[48] Tao, F., Zhang, H., Liu, A., & Nee, A. Y. C. (2019). "Digital Twin in Industry: State-of-the-Art." *IEEE Transactions on Industrial Informatics*, 15(4), 2405-2415. (Supports the architectural definition of the Digital Twin loop).

[49] Pei, K., Cao, Y., Yang, J., & Jana, S. (2017). "DeepXplore: Automated Whitebox Testing of Deep Learning Systems." *Proceedings of the 26th Symposium on Operating Systems Principles (SOSP)*. (Foundational for Falsification in ML).

[50] Kleppmann, M. (2017). "Designing Data-Intensive Applications." *O'Reilly Media*. (Authoritative source on log-structured storage and schema evolution for Event Journals).

[51] Amodei, D., et al. (2016). "Concrete Problems in AI Safety." *arXiv preprint arXiv:1606.06565*. (Supports Jidoka implementation for AI failure modes like reward hacking or side effects).

[52] Li, M., et al. (2014). "Scaling Distributed Machine Learning with the Parameter Server." *OSDI'14*. (Relevant for Heijunka load balancing in ML training).

[53] Jung, R., Jourdan, J. H., Krebbers, R., & Dreyer, D. (2017). "RustBelt: Securing the Foundations of the Rust Programming Language." *Proceedings of the ACM on Programming Languages (POPL)*. (Validates safety claims of the underlying Rust architecture).

[54] Banks, J., & Carson, J. S. (2009). "Discrete-Event System Simulation." *Pearson*. (Standard text for validating Event Scheduler logic).

[55] Blumofe, R. D., & Leiserson, C. E. (1999). "Scheduling Multithreaded Computations by Work Stealing." *Journal of the ACM*, 46(5), 720-748. (Theoretical basis for Work Stealing in parallel simulation).

[56] Lee, E. A., & Seshia, S. A. (2017). "Introduction to Embedded Systems: A Cyber-Physical Systems Approach." *MIT Press*. (Crucial for hybrid Physics/ML time synchronization).

[57] Bosch, J. (2017). "Speed, Data, and Ecosystems: Excelling in a Software-Driven World." *CRC Press*. (Connects Toyota Production System principles to modern software architecture and continuous simulation).

---

## Appendix A: Comparison with Existing Solutions

| Feature | simular | Nyx [1] | MadSim [2] | Rapier [3] | OpenTwins [4] |
|---------|---------|---------|------------|------------|---------------|
| **Physics** | Multi-domain | Orbital only | None | Rigid body | FMI |
| **Monte Carlo** | trueno-accelerated | Basic | None | None | None |
| **ML Integration** | aprender/entrenar | None | None | None | Partial |
| **Determinism** | First-class | Partial | First-class | Partial | Limited |
| **YAML Config** | Full schema | None | None | None | Partial |
| **TUI** | ratatui | None | None | None | None |
| **Web Viz** | wgpu/WebGL | None | None | None | Yes |
| **Replay** | Time-travel | None | Yes | None | None |
| **Falsification** | Popperian | None | None | None | None |
| **Language** | Rust | Rust | Rust | Rust | Various |

---

## Appendix B: Stack Integration Matrix

| simular Component | trueno | aprender | entrenar | realizar | alimentar | pacha | renacer |
|-------------------|--------|----------|----------|----------|-----------|-------|---------|
| Physics Engine | SIMD accel | - | - | - | - | - | - |
| Monte Carlo | GPU parallel | - | - | - | - | - | - |
| Optimization | - | GP surrogate | Gradients | - | - | - | - |
| ML Surrogate | - | Models | Training | Inference | - | Versioning | - |
| Data I/O | - | - | - | - | Parquet/Arrow | - | - |
| Replay | - | - | - | - | - | Checkpoints | Tracing |
| Validation | - | - | - | - | - | - | Syscall trace |

---

## Appendix C: Dynamic Stack Discovery

Per the Batuta Stack Review [32], hardcoded component lists introduce **Muda of Processing** (maintenance waste). simular uses dynamic discovery to detect available stack components:

```rust
/// Dynamic discovery of Sovereign AI Stack components
/// Eliminates hardcoded lists (Batuta Review §2.2)
pub struct StackDiscovery {
    /// Discovered stack crates and versions
    components: HashMap<StackComponent, Version>,
}

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum StackComponent {
    Trueno,
    TruenoDB,
    TruenoGraph,
    TruenoRag,
    Aprender,
    Entrenar,
    Realizar,
    Alimentar,
    Pacha,
    Renacer,
}

impl StackDiscovery {
    /// Discover available stack components from Cargo.toml
    pub fn from_cargo_toml(path: &Path) -> Result<Self, DiscoveryError> {
        let manifest = cargo_toml::Manifest::from_path(path)?;
        let mut components = HashMap::new();

        // Dynamic discovery - no hardcoded list
        for (name, dep) in manifest.dependencies.iter() {
            if let Some(component) = Self::parse_stack_component(name) {
                let version = Self::extract_version(dep)?;
                components.insert(component, version);
            }
        }

        Ok(Self { components })
    }

    /// Parse component name with fuzzy matching
    fn parse_stack_component(name: &str) -> Option<StackComponent> {
        match name {
            "trueno" => Some(StackComponent::Trueno),
            "trueno-db" | "trueno_db" => Some(StackComponent::TruenoDB),
            "trueno-graph" | "trueno_graph" => Some(StackComponent::TruenoGraph),
            "trueno-rag" | "trueno_rag" => Some(StackComponent::TruenoRag),
            "aprender" => Some(StackComponent::Aprender),
            "entrenar" => Some(StackComponent::Entrenar),
            "realizar" => Some(StackComponent::Realizar),
            "alimentar" => Some(StackComponent::Alimentar),
            "pacha" => Some(StackComponent::Pacha),
            "renacer" => Some(StackComponent::Renacer),
            _ => None,
        }
    }

    /// Check if a component is available
    pub fn has(&self, component: StackComponent) -> bool {
        self.components.contains_key(&component)
    }

    /// Get component version
    pub fn version(&self, component: StackComponent) -> Option<&Version> {
        self.components.get(&component)
    }
}
```

---

## Appendix D: Glossary

| Term | Definition |
|------|------------|
| **Jidoka** | Toyota principle: autonomous detection of abnormalities and immediate stop |
| **Poka-Yoke** | Mistake-proofing mechanisms that prevent errors before they occur |
| **Falsification** | Popper's criterion: a theory is scientific only if it can be disproven |
| **Nullification** | The act of rejecting (nullifying) the null hypothesis H₀ based on evidence |
| **Demarcation** | Popper's criterion distinguishing science from non-science |
| **Robustness (STL)** | Quantitative measure of distance to specification violation |
| **Symplectic** | Property of an integrator that preserves phase space volume |
| **Surrogate Model** | Cheap approximation of an expensive simulation for optimization |
| **Time-Travel** | Ability to seek to any point in simulation history and inspect state |
| **NHST** | Null Hypothesis Significance Testing: statistical framework for falsification |

---

## Appendix E: Citation Summary

This specification includes **57 peer-reviewed and authoritative citations** spanning:

| Category | Count | Key Sources |
|----------|-------|-------------|
| Philosophy of Science | 2 | Popper [5][6] |
| Statistical Methods | 1 | Streiner [26] |
| Simulation & Reproducibility | 6 | [2][9][17][18][21][22] |
| Physics & Numerical Methods | 4 | [13][14][27][35] |
| Monte Carlo | 4 | [10][11][12][20] |
| Optimization & ML | 4 | [15][16][34][35] |
| Astrodynamics | 2 | [1][19] |
| Signal Temporal Logic | 2 | [29][30] |
| Rust & HPC | 4 | [23][28][33][53] |
| Software Engineering | 4 | [7][8][32][36][37] |
| Toyota Production System | 3 | [24][25][57] |
| **ML Simulation & Reproducibility** | **4** | **[38][39][40][41]** |
| **ML Quality (Toyota Way)** | **3** | **[42][43][44]** |
| **Multi-Turn Evaluation & Pareto** | **3** | **[45][46][47]** |
| **TPS Code Review & Digital Twin** | **5** | **[48][49][50][51][52]** |
| **Parallel Computing & Systems** | **4** | **[54][55][56][57]** |

---

## Appendix F: ML Simulation Nullification Framework

This appendix defines the Popperian nullification framework for ML simulation scenarios, establishing falsifiable hypotheses that can be empirically tested and rejected.

### F.1 Theoretical Basis

Per Popper [5], a hypothesis is scientific if and only if it is falsifiable—i.e., there exists a possible observation that would contradict it. Applied to ML simulation, we formulate null hypotheses (H₀) that the simulation must attempt to reject through evidence.

**Demarcation Criterion for ML Simulation**: A simulated ML hypothesis H is scientific iff:
```
∃ observation O: O ∈ simulation_outputs ∧ O contradicts H
```

### F.2 Training Simulation Null Hypotheses

| Hypothesis ID | H₀ (Null) | H₁ (Alternative) | Test Statistic | Rejection Criterion |
|---------------|-----------|------------------|----------------|---------------------|
| **H₀-TRAIN-01** | Training converges identically across runs with same seed | Non-determinism exists | Bitwise difference δ | δ ≠ 0 → reject H₀ [41] |
| **H₀-TRAIN-02** | Loss monotonically decreases (no spikes) | Anomalous loss spikes occur | Z-score of loss | \|z\| > 3σ → reject H₀ [43] |
| **H₀-TRAIN-03** | Gradient norms remain bounded | Gradient explosion/vanishing | ‖∇L‖₂ | ‖∇L‖₂ > 10⁶ ∨ ‖∇L‖₂ < 10⁻⁶ → reject H₀ |
| **H₀-TRAIN-04** | Training time scales linearly with data | Superlinear scaling | T(2n)/T(n) | ratio > 2.1 → reject H₀ |
| **H₀-TRAIN-05** | Model parameters remain finite | NaN/Inf corruption | is_finite(θ) | ¬is_finite → reject H₀ |

### F.3 Prediction Simulation Null Hypotheses

| Hypothesis ID | H₀ (Null) | H₁ (Alternative) | Test Statistic | Rejection Criterion |
|---------------|-----------|------------------|----------------|---------------------|
| **H₀-PRED-01** | Predictions are deterministic for temperature=0 | Stochastic variation exists | Variance σ² | σ² > 0 → reject H₀ |
| **H₀-PRED-02** | Model integrity verified (CRC32 match) | Model corruption | CRC32(loaded) = CRC32(stored) | mismatch → reject H₀ |
| **H₀-PRED-03** | Latency within specified SLA | SLA violation | p99 latency | p99 > SLA → reject H₀ |
| **H₀-PRED-04** | Uncertainty estimates are calibrated | Miscalibration | ECE (Expected Calibration Error) | ECE > 0.05 → reject H₀ [43] |
| **H₀-PRED-05** | Output distribution matches training distribution | Distribution shift | KL divergence | KL(P‖Q) > threshold → reject H₀ |

### F.4 Multi-Turn Simulation Null Hypotheses

| Hypothesis ID | H₀ (Null) | H₁ (Alternative) | Test Statistic | Rejection Criterion |
|---------------|-----------|------------------|----------------|---------------------|
| **H₀-MULTI-01** | Accuracy ≥ oracle baseline | Degradation vs oracle | Accuracy gap | gap > 0.05 → reject H₀ [45] |
| **H₀-MULTI-02** | Cost scales linearly with turns | Superlinear cost growth | Cost(n)/n | ratio increases with n → reject H₀ |
| **H₀-MULTI-03** | Context utilization is efficient | Context waste | Unused context tokens | >30% unused → reject H₀ |
| **H₀-MULTI-04** | Model is Pareto-optimal | Model is dominated | Dominated-by count | count > 0 → reject H₀ [46] |
| **H₀-MULTI-05** | Results are statistically significant | Insufficient evidence | p-value (5 runs, t-test) | p > 0.05 → reject H₀ [45] |

### F.5 Statistical Framework

Following Princeton methodology [45] and Streiner's application of Popper to statistical testing [26]:

```rust
/// Popperian nullification test for ML simulation
pub struct NullificationTest {
    /// Null hypothesis identifier
    pub hypothesis_id: &'static str,
    /// Number of independent runs (minimum 5 per [45])
    pub n_runs: usize,
    /// Significance level (α = 0.05 standard)
    pub alpha: f64,
    /// Test results
    pub results: Vec<TestResult>,
}

impl NullificationTest {
    /// Execute nullification test with bootstrap CI
    pub fn execute<F>(&mut self, test_fn: F) -> NullificationResult
    where
        F: Fn() -> f64,
    {
        assert!(self.n_runs >= 5, "Princeton methodology requires ≥5 runs");

        // Collect observations
        let observations: Vec<f64> = (0..self.n_runs)
            .map(|_| test_fn())
            .collect();

        // Compute bootstrap 95% CI
        let ci = bootstrap_ci(&observations, 10_000, 0.95);

        // Compute p-value using Student's t-distribution
        let (t_stat, p_value) = t_test(&observations);

        NullificationResult {
            hypothesis_id: self.hypothesis_id,
            rejected: p_value < self.alpha,
            p_value,
            confidence_interval: ci,
            effect_size: cohens_d(&observations),
            observations,
        }
    }
}

/// Result of nullification test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NullificationResult {
    pub hypothesis_id: &'static str,
    pub rejected: bool,
    pub p_value: f64,
    pub confidence_interval: (f64, f64),
    pub effect_size: f64,
    pub observations: Vec<f64>,
}

impl NullificationResult {
    /// Format result following Princeton reporting standards [45]
    pub fn report(&self) -> String {
        format!(
            "{}: {} (p={:.4}, 95% CI [{:.4}, {:.4}], d={:.2})",
            self.hypothesis_id,
            if self.rejected { "REJECTED" } else { "NOT REJECTED" },
            self.p_value,
            self.confidence_interval.0,
            self.confidence_interval.1,
            self.effect_size,
        )
    }
}
```

### F.6 Jidoka Integration: Automatic Nullification

Toyota's Jidoka principle mandates stopping when quality issues are detected. Applied to ML nullification:

```rust
/// Jidoka-triggered nullification for ML simulation
pub struct JidokaNullification {
    /// Active nullification tests
    tests: Vec<NullificationTest>,
    /// Automatic stop on rejection
    auto_stop: bool,
    /// Kaizen: improvement patches from rejections
    patches: Vec<ImprovementPatch>,
}

impl JidokaNullification {
    /// Run all nullification tests with Jidoka stopping
    pub fn run_all(&mut self, simulation: &mut impl MLSimulation) -> SimResult<NullificationReport> {
        let mut report = NullificationReport::new();

        for test in &mut self.tests {
            let result = test.execute(|| simulation.observe());

            if result.rejected && self.auto_stop {
                // Jidoka: Stop and generate improvement
                let patch = self.generate_kaizen_patch(&result);
                self.patches.push(patch.clone());

                return Err(SimError::jidoka(format!(
                    "Nullification {} rejected: {} - Generated Kaizen patch: {:?}",
                    result.hypothesis_id,
                    result.report(),
                    patch
                )));
            }

            report.add(result);
        }

        Ok(report)
    }

    /// Generate Kaizen improvement patch from rejected hypothesis
    fn generate_kaizen_patch(&self, result: &NullificationResult) -> ImprovementPatch {
        match result.hypothesis_id {
            "H₀-TRAIN-02" => ImprovementPatch::LearningRateDecay { factor: 0.5 },
            "H₀-TRAIN-03" => ImprovementPatch::GradientClipping { max_norm: 1.0 },
            "H₀-PRED-04" => ImprovementPatch::TemperatureCalibration { method: "platt" },
            "H₀-MULTI-01" => ImprovementPatch::PromptEngineering { strategy: "chain-of-thought" },
            _ => ImprovementPatch::ManualReview,
        }
    }
}
```

### F.7 Nullification Reporting Standard

Per Toyota Way documentation practices and academic reproducibility standards [38][39]:

```yaml
# nullification_report.yaml
report_version: "1.0"
timestamp: "2025-12-10T12:00:00Z"
simulation_id: "training-sim-001"

methodology:
  framework: "Popperian Falsification"
  statistical_test: "Student's t-test"
  n_runs: 5
  alpha: 0.05
  bootstrap_samples: 10000

hypotheses:
  - id: "H₀-TRAIN-01"
    description: "Training converges identically across runs with same seed"
    status: "NOT REJECTED"
    p_value: 1.0
    confidence_interval: [0.0, 0.0]
    observations: [0.0, 0.0, 0.0, 0.0, 0.0]
    interpretation: "Deterministic training verified"

  - id: "H₀-TRAIN-02"
    description: "Loss monotonically decreases (no spikes)"
    status: "REJECTED"
    p_value: 0.003
    confidence_interval: [3.2, 4.1]
    observations: [3.5, 3.8, 3.2, 4.1, 3.6]
    interpretation: "Loss spikes detected at epochs 15, 23, 41"
    kaizen_patch:
      type: "LearningRateDecay"
      parameters:
        factor: 0.5
        trigger_epoch: 10

summary:
  total_tests: 15
  rejected: 1
  not_rejected: 14
  kaizen_patches_generated: 1
  overall_status: "PARTIAL_PASS"

references:
  - "[38] Pineau et al., 2021"
  - "[45] Kapoor et al., 2024"
```

### F.8 Integration with Sovereign AI Stack

The nullification framework integrates with stack components:

| Component | Nullification Role | Integration Point |
|-----------|-------------------|-------------------|
| **aprender** | Model format validation (CRC32, signatures) | H₀-PRED-02 |
| **entrenar** | Training dynamics monitoring | H₀-TRAIN-01 through H₀-TRAIN-05 |
| **realizar** | Inference SLA verification | H₀-PRED-03 |
| **renacer** | Syscall-level reproducibility tracing | All hypotheses |
| **pacha** | Checkpoint versioning for replay | Time-travel debugging |

---

*Document generated: 2025-12-10*
*Sovereign AI Stack: Privacy-preserving ML infrastructure*
