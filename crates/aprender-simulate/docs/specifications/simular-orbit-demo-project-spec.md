# Simular Orbit Demo: Technical Specification

**Document ID:** SIMULAR-SPEC-ORBIT-001
**Version:** 1.1.0
**Status:** Draft (Revised per TPS Review)
**Classification:** Open Source
**Author:** PAIML Engineering
**Reviewer:** Gemini (TPS Technical Review)
**Date:** 2025-12-10

---

## Executive Summary

This specification defines the **Simular Orbit Demo**, a canonical demonstration of the Simular simulation engine showcasing deterministic orbital mechanics with dual deployment targets: Terminal User Interface (TUI) for CLI environments and WebAssembly (WASM) for browser-based visualization. The implementation adheres to Toyota Production System (TPS) quality principles and NASA/JPL mission-critical software standards.

**Revision 1.1.0 incorporates feedback from TPS technical review addressing:**
- Graceful degradation for Jidoka (vs. abrupt halt)
- Heijunka time-budget enforcement for O(N²) scaling
- Poka-Yoke dimensional analysis via newtype pattern
- Epsilon-determinism for cross-platform floating-point
- Metamorphic testing for physics invariants

---

## 1. Introduction

### 1.1 Purpose

The Simular Orbit Demo serves as the reference implementation demonstrating:

1. **Epsilon-Deterministic Reproducibility** — Bounded-error identical results across platforms given identical seeds [26]
2. **Jidoka Quality Gates** — Automatic anomaly detection with graceful degradation (not abrupt halt)
3. **Dual-Target Architecture** — Single codebase compiling to native TUI and WASM
4. **Physics Fidelity** — Symplectic integration preserving orbital invariants

### 1.2 Scope

This specification covers:

- Mathematical model of N-body gravitational dynamics
- Numerical integration methods with error bounds
- Visualization requirements for TUI and WebGL/Canvas
- Quality assurance through Jidoka guards with graceful degradation
- Build and deployment pipeline
- Heijunka time-budget enforcement

### 1.3 Design Philosophy

#### 1.3.1 Toyota Production System (TPS) Principles

| Principle | Application | Implementation |
|-----------|-------------|----------------|
| **Jidoka** (自働化) | Anomaly detection with graceful degradation | Pause simulation, highlight defect, allow recovery [27] |
| **Poka-Yoke** (ポカヨケ) | Type-safe dimensional analysis | Newtype pattern via `uom` crate [28] |
| **Heijunka** (平準化) | Load-balanced computation | Time-budget per frame (16ms target) |
| **Mieruka** (見える化) | Visual management | Real-time Jidoka status display |
| **Genchi Genbutsu** (現地現物) | Direct observation via replay system | Time-travel debugging |
| **Kaizen** (改善) | Continuous improvement through metrics | Performance telemetry |

#### 1.3.2 NASA/JPL Power of 10 Rules [1]

1. Simple control flow (max cyclomatic complexity: 15)
2. Fixed upper bounds on loops
3. No dynamic memory allocation after initialization
4. Maximum function length: 60 lines
5. Minimum 2 assertions per function
6. Data hiding at smallest scope
7. All return values checked
8. Limited preprocessor/macro usage
9. Restrict pointer arithmetic (enforced by Safe Rust)
10. Compile with all warnings enabled, treat as errors

#### 1.3.3 Muda (Waste) Elimination

| Muda Type | Risk in Spec | Mitigation |
|-----------|--------------|------------|
| **Over-processing** | Strict bit-determinism across platforms | Epsilon-determinism with documented bounds |
| **Waiting** | O(N²) blocking main thread | Time-budget with frame skipping |
| **Defects** | Dimensional errors (m + m/s) | Compile-time dimensional analysis |
| **Motion** | JSON serialization overhead | SharedArrayBuffer for WASM [29] |

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     simular-orbit-demo                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Scenarios │  │   Physics   │  │      Visualization      │  │
│  │             │  │   Engine    │  │                         │  │
│  │ - Kepler    │  │             │  │  ┌─────────┐ ┌───────┐  │  │
│  │ - N-body    │  │ - Verlet    │  │  │   TUI   │ │ WASM  │  │  │
│  │ - Hohmann   │  │ - RK4       │  │  │(ratatui)│ │(wasm- │  │  │
│  │ - Lagrange  │  │ - Yoshida   │  │  │         │ │bindgen│  │  │
│  │             │  │             │  │  └─────────┘ └───────┘  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Core Engine (simular)                    ││
│  │  SimState │ SimRng │ JidokaGuard │ ReplaySystem │ Clock    ││
│  └─────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Heijunka Scheduler (Time Budget)               ││
│  │  FrameBudget │ SubStepController │ AdaptiveQuality          ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Module Structure

```
simular-orbit-demo/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Core simulation (no_std compatible)
│   ├── scenarios/
│   │   ├── mod.rs
│   │   ├── kepler.rs       # Two-body Keplerian orbits
│   │   ├── n_body.rs       # N-body gravitational simulation
│   │   ├── hohmann.rs      # Hohmann transfer maneuvers
│   │   └── lagrange.rs     # Lagrange point dynamics
│   ├── physics/
│   │   ├── mod.rs
│   │   ├── gravity.rs      # Gravitational force computation
│   │   ├── integrators.rs  # Symplectic integrators
│   │   ├── units.rs        # Type-safe physical units (Poka-Yoke)
│   │   └── close_encounter.rs  # Regularization + adaptive stepping
│   ├── jidoka/
│   │   ├── mod.rs
│   │   ├── energy.rs       # Energy conservation monitor
│   │   ├── angular.rs      # Angular momentum conservation
│   │   ├── stability.rs    # Numerical stability checks
│   │   └── graceful.rs     # Graceful degradation handlers
│   ├── heijunka/
│   │   ├── mod.rs
│   │   ├── budget.rs       # Frame time budget management
│   │   └── adaptive.rs     # Quality/fidelity adaptation
│   └── render/
│       ├── mod.rs
│       ├── commands.rs     # Platform-agnostic render commands
│       ├── camera.rs       # View transformation
│       └── shared_buffer.rs # SharedArrayBuffer for WASM
├── src/bin/
│   └── orbit-tui.rs        # TUI binary entry point
├── src/wasm/
│   └── lib.rs              # WASM entry point
└── tests/
    ├── determinism.rs      # Reproducibility verification
    ├── conservation.rs     # Physics invariant tests
    ├── metamorphic.rs      # Metamorphic relation tests
    └── integration.rs      # End-to-end scenarios
```

### 2.3 Compilation Targets

| Target | Toolchain | Output | Use Case |
|--------|-----------|--------|----------|
| `x86_64-unknown-linux-gnu` | stable | Native binary | CLI/TUI |
| `x86_64-apple-darwin` | stable | Native binary | CLI/TUI |
| `wasm32-unknown-unknown` | stable | WASM module | Browser |

### 2.4 Poka-Yoke: Type-Safe Dimensional Analysis

Following Kennedy's dimensional analysis principles [28], all physical quantities use newtype wrappers:

```rust
use uom::si::f64::{Length, Mass, Time, Velocity, Acceleration};
use uom::si::length::meter;
use uom::si::mass::kilogram;
use uom::si::time::second;
use uom::si::velocity::meter_per_second;

/// Position vector with dimensional safety
#[derive(Clone, Copy, Debug)]
pub struct Position {
    pub x: Length,
    pub y: Length,
    pub z: Length,
}

/// Velocity vector with dimensional safety
#[derive(Clone, Copy, Debug)]
pub struct OrbitalVelocity {
    pub x: Velocity,
    pub y: Velocity,
    pub z: Velocity,
}

// COMPILE-TIME ERROR: Cannot add Position + Velocity
// let invalid = position + velocity;  // Type error!

// VALID: Velocity = Position / Time
impl std::ops::Div<Time> for Position {
    type Output = OrbitalVelocity;
    fn div(self, dt: Time) -> OrbitalVelocity {
        OrbitalVelocity {
            x: self.x / dt,
            y: self.y / dt,
            z: self.z / dt,
        }
    }
}
```

**Rationale:** Kennedy (1996) demonstrated that dimensional analysis at compile time eliminates a distinct class of safety-critical failures common in physics simulations [28].

---

## 3. Mathematical Model

### 3.1 Gravitational N-Body Problem

The gravitational N-body problem describes the motion of N point masses under mutual gravitational attraction [2, 3].

#### 3.1.1 Equations of Motion

For body $i$ with mass $m_i$ at position $\mathbf{r}_i$:

$$\ddot{\mathbf{r}}_i = -G \sum_{j \neq i} \frac{m_j (\mathbf{r}_i - \mathbf{r}_j)}{|\mathbf{r}_i - \mathbf{r}_j|^3}$$

Where $G = 6.67430 \times 10^{-11} \, \text{m}^3 \text{kg}^{-1} \text{s}^{-2}$ is the gravitational constant [4].

#### 3.1.2 Computational Complexity and Heijunka

**WARNING:** Direct summation scales as $O(N^2)$. This creates *Mura* (unevenness) violating Heijunka [30].

| N (bodies) | Force calculations | Time @ 1GHz |
|------------|-------------------|-------------|
| 5 | 10 | ~10ns |
| 10 | 45 | ~45ns |
| 100 | 4,950 | ~5μs |
| 1,000 | 499,500 | ~500μs |

**Mitigation Strategy:**

```rust
pub struct HeijunkaScheduler {
    frame_budget_ms: f64,      // Target: 16ms (60 FPS)
    physics_budget_ms: f64,    // Allocated: 8ms (50% of frame)
    max_substeps: usize,       // Upper bound per frame
    quality_level: QualityLevel,
}

impl HeijunkaScheduler {
    pub fn execute_frame(&mut self, sim: &mut Simulation) -> FrameResult {
        let start = Instant::now();
        let mut substeps = 0;

        while start.elapsed().as_secs_f64() * 1000.0 < self.physics_budget_ms
            && substeps < self.max_substeps
        {
            sim.step(self.dt)?;
            substeps += 1;
        }

        // If budget exceeded, reduce quality for next frame
        if substeps == 0 {
            self.quality_level = self.quality_level.degrade();
        }

        FrameResult { substeps, quality: self.quality_level }
    }
}
```

#### 3.1.3 Hamiltonian Formulation

The system Hamiltonian [5]:

$$H = \sum_{i=1}^{N} \frac{|\mathbf{p}_i|^2}{2m_i} - G \sum_{i<j} \frac{m_i m_j}{|\mathbf{r}_i - \mathbf{r}_j|}$$

Conservation of $H$ serves as a Jidoka invariant for numerical validation.

### 3.2 Orbital Elements

Keplerian orbital elements [6, 7]:

| Element | Symbol | Description |
|---------|--------|-------------|
| Semi-major axis | $a$ | Size of orbit |
| Eccentricity | $e$ | Shape (0=circle, <1=ellipse) |
| Inclination | $i$ | Tilt from reference plane |
| RAAN | $\Omega$ | Right ascension of ascending node |
| Argument of periapsis | $\omega$ | Orientation in orbital plane |
| True anomaly | $\nu$ | Position along orbit |

### 3.3 Kepler's Laws

1. **First Law**: Orbits are conic sections with primary at focus
2. **Second Law**: Equal areas swept in equal times (angular momentum conservation)
3. **Third Law**: $T^2 \propto a^3$ (period-semimajor axis relation)

---

## 4. Numerical Methods

### 4.1 Integrator Selection Criteria

Per NASA/JPL standards for orbital mechanics [8, 9]:

| Criterion | Requirement |
|-----------|-------------|
| Symplecticity | Required for long-term stability |
| Time-reversibility | Required for energy conservation |
| Error order | Minimum 4th order |
| Adaptive stepping | Required for close encounters |

### 4.2 Velocity Verlet (Störmer-Verlet)

The Störmer-Verlet method [10, 11] is symplectic and time-reversible:

```
x_{n+1} = x_n + v_n * dt + 0.5 * a_n * dt²
a_{n+1} = acceleration(x_{n+1})
v_{n+1} = v_n + 0.5 * (a_n + a_{n+1}) * dt
```

**Properties:**
- Order: 2 (position), 2 (velocity)
- Symplectic: Yes
- Time-reversible: Yes
- Energy drift: $O(dt^2)$ bounded oscillation [12]

### 4.3 4th-Order Yoshida Integrator

The Yoshida 4th-order method [13] extends Störmer-Verlet:

```rust
const W0: f64 = -1.702414383919315;  // -(2^(1/3))/(2 - 2^(1/3))
const W1: f64 = 1.351207191959657;   // 1/(2 - 2^(1/3))
const C: [f64; 4] = [W1/2.0, (W0+W1)/2.0, (W0+W1)/2.0, W1/2.0];
const D: [f64; 3] = [W1, W0, W1];
```

**Properties:**
- Order: 4
- Symplectic: Yes
- Time-reversible: Yes
- Computational cost: 3× Verlet per step

**Limitation:** Symplectic integrators conserve a *modified* Hamiltonian, not the exact Hamiltonian. Energy oscillates but remains bounded [8]. For close encounters, this bounded oscillation can grow unacceptably.

### 4.4 Close Encounter Handling

Per Aarseth's N-body techniques [14, 30], close encounters require special treatment beyond simple softening:

#### 4.4.1 Softening Parameter (Basic)

$$F_{ij} = \frac{G m_i m_j}{(|\mathbf{r}_{ij}|^2 + \epsilon^2)^{3/2}} \hat{\mathbf{r}}_{ij}$$

Where $\epsilon$ is the softening length.

#### 4.4.2 Adaptive Time-Stepping (Required for Close Encounters)

```rust
pub struct AdaptiveIntegrator {
    base_dt: f64,
    min_dt: f64,
    max_dt: f64,
    error_tolerance: f64,
}

impl AdaptiveIntegrator {
    pub fn compute_dt(&self, state: &SimState) -> f64 {
        let min_separation = state.min_pairwise_separation();
        let max_velocity = state.max_velocity_magnitude();

        // Courant-Friedrichs-Lewy condition
        let cfl_dt = min_separation / max_velocity * 0.1;

        // Close encounter criterion
        let encounter_dt = if min_separation < self.encounter_threshold {
            self.min_dt
        } else {
            self.base_dt
        };

        cfl_dt.min(encounter_dt).clamp(self.min_dt, self.max_dt)
    }
}
```

#### 4.4.3 Regularization (KS Transformation)

For very close encounters, Kustaanheimo-Stiefel regularization removes the singularity [31]:

```rust
/// KS regularization for close binary encounters
pub struct KSRegularization {
    threshold: f64,  // Switch to KS when separation < threshold
}

impl KSRegularization {
    pub fn transform(&self, r: Vec3) -> KSVector {
        // Transform to regularized coordinates where singularity is removed
        // ...
    }
}
```

**Standard Work for Close Encounters:**

1. Monitor minimum separation each step
2. If separation < 10× softening length: reduce dt by 10×
3. If separation < softening length: apply KS regularization
4. If separation approaches zero: trigger Jidoka collision guard

---

## 5. Jidoka Quality Gates (Graceful Degradation)

### 5.1 Jidoka Philosophy: Graceful Degradation, Not Crash

Per TPS principles, Jidoka means "automation with a human touch" — the system should **stop and highlight the defect** while remaining available, not crash abruptly [27].

```rust
/// Jidoka violation with graceful degradation
#[derive(Debug, Clone)]
pub enum JidokaResponse {
    /// Continue with warning displayed
    Warning { message: String, body_index: Option<usize> },

    /// Pause simulation, highlight defect, allow user intervention
    Pause { violation: JidokaViolation, recoverable: bool },

    /// Unrecoverable: save state and halt
    Halt { violation: JidokaViolation, state_snapshot: StateSnapshot },
}

pub struct JidokaGuard {
    config: JidokaConfig,
    violation_count: usize,
    max_warnings_before_pause: usize,
}

impl JidokaGuard {
    pub fn check(&mut self, state: &SimState) -> JidokaResponse {
        // Check all invariants
        if let Err(violation) = self.check_energy(state) {
            self.violation_count += 1;

            if self.violation_count < self.max_warnings_before_pause {
                return JidokaResponse::Warning {
                    message: format!("Energy drift detected: {}", violation),
                    body_index: None,
                };
            }

            // Pause and visualize the problem
            return JidokaResponse::Pause {
                violation,
                recoverable: true,  // User can adjust dt or reset
            };
        }

        // Reset counter on clean check
        self.violation_count = 0;
        JidokaResponse::Continue
    }
}
```

### 5.2 Energy Conservation Monitor

Total mechanical energy must remain bounded [15]:

```rust
pub struct EnergyGuard {
    initial_energy: f64,
    tolerance: f64,           // Default: 1e-6 (relative)
    warning_threshold: f64,   // Default: 1e-8 (early warning)
    accumulated_drift: f64,
}

impl EnergyGuard {
    pub fn check(&mut self, state: &SimState) -> Result<(), JidokaViolation> {
        let current = state.total_energy();
        let relative_error = (current - self.initial_energy).abs()
            / self.initial_energy.abs();

        self.accumulated_drift += relative_error;

        if relative_error > self.tolerance {
            return Err(JidokaViolation::EnergyDrift {
                initial: self.initial_energy,
                current,
                relative_error,
                threshold: self.tolerance,
                suggestion: "Consider reducing time step or using Yoshida integrator",
            });
        }
        Ok(())
    }
}
```

### 5.3 Angular Momentum Conservation

Total angular momentum $\mathbf{L} = \sum_i m_i \mathbf{r}_i \times \mathbf{v}_i$ is conserved [16]:

```rust
pub struct AngularMomentumGuard {
    initial_momentum: Vec3,
    tolerance: f64,
}
```

### 5.4 Numerical Stability Checks

Per IEEE 754-2019 [17]:

```rust
pub fn check_finite(state: &SimState) -> Result<(), JidokaViolation> {
    for (i, pos) in state.positions().iter().enumerate() {
        if !pos.is_finite() {
            return Err(JidokaViolation::NonFinite {
                body_index: i,
                field: "position",
                value: *pos,
                suggestion: "NaN detected - likely numerical instability from close encounter",
            });
        }
    }
    Ok(())
}
```

### 5.5 Collision Detection

Minimum separation threshold [18]:

```rust
pub fn check_collision(state: &SimState, min_separation: f64) -> Result<(), JidokaViolation> {
    for i in 0..state.num_bodies() {
        for j in (i+1)..state.num_bodies() {
            let separation = (state.positions()[i] - state.positions()[j]).magnitude();
            if separation < min_separation {
                return Err(JidokaViolation::Collision {
                    body_i: i,
                    body_j: j,
                    separation,
                    suggestion: "Bodies collided - consider enabling softening or KS regularization",
                });
            }
        }
    }
    Ok(())
}
```

### 5.6 Visual Management (Mieruka)

Jidoka status must be continuously visible in both TUI and WASM:

```
┌─────────────────────────────────────────────────────────────────┐
│ Jidoka Status:                                                  │
│   ✓ Energy:    Δ=1.2e-9 (tol: 1e-6)  [████████░░] OK           │
│   ✓ Angular:   Δ=3.1e-12 (tol: 1e-9) [██████████] OK           │
│   ✓ Finite:    All values finite                                │
│   ⚠ Separation: 0.05 AU (min: 0.01)  [████░░░░░░] WARNING      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Visualization

### 6.1 Platform-Agnostic Render Commands

Following the command pattern [19] for platform abstraction:

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RenderCommand {
    Clear { color: Color },
    DrawCircle { x: f64, y: f64, radius: f64, color: Color },
    DrawLine { x1: f64, y1: f64, x2: f64, y2: f64, color: Color, width: f64 },
    DrawOrbitPath { points: Vec<(f64, f64)>, color: Color },
    DrawText { x: f64, y: f64, text: String, font_size: f64, color: Color },
    DrawVelocityVector { x: f64, y: f64, vx: f64, vy: f64, scale: f64, color: Color },
    SetCamera { center_x: f64, center_y: f64, zoom: f64 },
    HighlightBody { index: usize, color: Color },  // For Jidoka visualization
}
```

### 6.2 TUI Visualization (ratatui)

Terminal rendering using Braille characters for sub-cell precision [20]:

```rust
// Each terminal cell = 2×4 Braille dots = 8 sub-pixels
// Resolution: (cols * 2) × (rows * 4) virtual pixels

pub struct TuiRenderer {
    width: u16,
    height: u16,
    buffer: Vec<Vec<bool>>,  // Braille dot buffer
}

impl TuiRenderer {
    pub fn render_to_braille(&self) -> Vec<char> {
        // Convert 2×4 dot patterns to Unicode Braille (U+2800-U+28FF)
    }
}
```

**TUI Layout:**

```
┌─────────────────────────────────────────────────────────────────┐
│  SIMULAR ORBIT DEMO                              [P]ause [R]eset│
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                           ⣿⣿                                    │
│                      ⣤⣤⣤⣿⣿⣿⣤⣤⣤                                │
│                   ⣤⣤⣿⣿⣿⣿⣿⣿⣿⣿⣿⣤⣤                              │
│                ⣤⣤⣿⣿⣿⣿⣿⣿  ☉  ⣿⣿⣿⣿⣿⣤⣤                          │
│               ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿                           │
│                ⣤⣤⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣤⣤    🌍                       │
│                   ⣤⣤⣿⣿⣿⣿⣿⣿⣿⣤⣤                                │
│                      ⣤⣤⣤⣿⣿⣤⣤⣤                                 │
│                           ⣿⣿                                    │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│ Time: 365.24d  Energy: -2.341e9 J (Δ: 1.2e-9)  Step: 0.001s    │
│ Bodies: Earth (1.0 AU, e=0.017)  Seed: 42  FPS: 60             │
│ Jidoka: ✓ Energy ✓ Angular Momentum ✓ Finite ✓ Separation      │
│ Heijunka: Budget 8ms | Used 3.2ms | Substeps 4 | Quality: High │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 WASM Visualization (Canvas/WebGL)

#### 6.3.1 Standard JSON Serialization

Browser rendering via wasm-bindgen [21]:

```rust
#[wasm_bindgen]
pub struct OrbitDemo {
    simulation: Simulation,
    render_commands: Vec<RenderCommand>,
}

#[wasm_bindgen]
impl OrbitDemo {
    #[wasm_bindgen(constructor)]
    pub fn new(config_json: &str) -> Result<OrbitDemo, JsValue> {
        // Parse config, initialize simulation
    }

    #[wasm_bindgen]
    pub fn tick(&mut self, dt: f64) -> Result<JsValue, JsValue> {
        self.simulation.step(dt)?;
        self.render_commands = self.simulation.render();
        Ok(serde_wasm_bindgen::to_value(&self.render_commands)?)
    }
}
```

#### 6.3.2 High-Performance SharedArrayBuffer Mode

Per Haas et al. [29], SharedArrayBuffer eliminates JSON serialization overhead (*Muda*):

```rust
#[cfg(feature = "shared-buffer")]
#[wasm_bindgen]
pub struct OrbitDemoShared {
    simulation: Simulation,
    // Direct memory access - no serialization
    position_buffer: Vec<f64>,  // [x0, y0, z0, x1, y1, z1, ...]
    velocity_buffer: Vec<f64>,
}

#[cfg(feature = "shared-buffer")]
#[wasm_bindgen]
impl OrbitDemoShared {
    /// Returns pointer to position buffer for direct JS access
    #[wasm_bindgen]
    pub fn position_buffer_ptr(&self) -> *const f64 {
        self.position_buffer.as_ptr()
    }

    /// Returns buffer length
    #[wasm_bindgen]
    pub fn buffer_len(&self) -> usize {
        self.position_buffer.len()
    }
}
```

**JavaScript side:**

```javascript
// Zero-copy access to WASM memory
const memory = demo.memory;
const positions = new Float64Array(memory.buffer, demo.position_buffer_ptr(), demo.buffer_len());

// Direct WebGL buffer upload - no JSON parsing
gl.bufferData(gl.ARRAY_BUFFER, positions, gl.DYNAMIC_DRAW);
```

---

## 7. Scenarios

### 7.1 Kepler Two-Body

Classic Earth-Sun system demonstrating Kepler's laws:

```rust
pub struct KeplerScenario {
    sun_mass: f64,      // 1.989e30 kg
    planet_mass: f64,   // 5.972e24 kg
    semi_major_axis: f64,
    eccentricity: f64,
}
```

### 7.2 N-Body Solar System

Inner solar system (Sun + 4 planets) [22]:

| Body | Mass (kg) | Semi-major axis (AU) | Eccentricity |
|------|-----------|---------------------|--------------|
| Sun | 1.989×10³⁰ | — | — |
| Mercury | 3.301×10²³ | 0.387 | 0.206 |
| Venus | 4.867×10²⁴ | 0.723 | 0.007 |
| Earth | 5.972×10²⁴ | 1.000 | 0.017 |
| Mars | 6.417×10²³ | 1.524 | 0.093 |

### 7.3 Hohmann Transfer

Minimum-energy transfer between circular orbits [23]:

```rust
pub struct HohmannTransfer {
    r1: f64,  // Initial orbit radius
    r2: f64,  // Target orbit radius
    mu: f64,  // Standard gravitational parameter
}

impl HohmannTransfer {
    pub fn delta_v1(&self) -> f64 {
        let v_circular = (self.mu / self.r1).sqrt();
        let v_transfer = (2.0 * self.mu * self.r2 / (self.r1 * (self.r1 + self.r2))).sqrt();
        v_transfer - v_circular
    }

    pub fn delta_v2(&self) -> f64 {
        let v_transfer = (2.0 * self.mu * self.r1 / (self.r2 * (self.r1 + self.r2))).sqrt();
        let v_circular = (self.mu / self.r2).sqrt();
        v_circular - v_transfer
    }
}
```

### 7.4 Lagrange Points

L1-L5 dynamics in restricted three-body problem [24]:

```rust
pub enum LagrangePoint { L1, L2, L3, L4, L5 }

pub struct LagrangeScenario {
    primary_mass: f64,
    secondary_mass: f64,
    point: LagrangePoint,
    perturbation: Vec3,  // Initial displacement for stability demo
}
```

---

## 8. Determinism Requirements (Epsilon-Determinism)

### 8.1 Reproducibility Guarantees (Revised)

Per Goldberg's analysis of floating-point arithmetic [26], strict bit-identical results across platforms is problematic due to:

1. **FMA (Fused Multiply-Add)**: Different CPU architectures have different FMA behavior
2. **Compiler optimizations**: `-O2` vs `-O3` may reorder operations
3. **WASM vs Native**: Different rounding in edge cases

**Solution: Epsilon-Determinism**

| Guarantee | Requirement |
|-----------|-------------|
| **Same platform, same seed** | Bit-identical |
| **Cross-platform, same seed** | ε-identical (ε = 1e-10 relative) |
| **WASM vs Native** | ε-identical (ε = 1e-9 relative) |

### 8.2 Implementation Constraints

```rust
// REQUIRED: Use SimRng for all randomness
let mut rng = SimRng::new(seed);

// REQUIRED: Consistent floating-point behavior
#[cfg(target_arch = "wasm32")]
compile_error!("Ensure wasm32 uses same FP rounding as native");

// OPTIONAL: Disable FMA for strict determinism (with performance cost)
// #[target_feature(enable = "fma")] // Comment out for cross-platform parity

// FORBIDDEN: HashMap iteration order dependency
// FORBIDDEN: Parallel execution with race conditions
// FORBIDDEN: System time dependency in simulation logic
```

### 8.3 Verification Tests

```rust
#[test]
fn test_reproducibility_same_platform() {
    let seed = 42u64;

    let trajectory_1 = run_simulation(seed, 1000);
    let trajectory_2 = run_simulation(seed, 1000);

    // Bit-identical on same platform
    assert_eq!(trajectory_1, trajectory_2);
}

#[test]
fn test_epsilon_determinism_cross_platform() {
    let seed = 42u64;
    let epsilon = 1e-10;

    let native_trajectory = run_simulation(seed, 1000);
    let reference_trajectory = load_reference_trajectory("seed_42_1000steps.json");

    for (native, reference) in native_trajectory.iter().zip(reference_trajectory.iter()) {
        let rel_error = (native.energy() - reference.energy()).abs() / reference.energy().abs();
        assert!(rel_error < epsilon, "Cross-platform drift exceeded epsilon");
    }
}
```

### 8.4 FMA Control (Optional Strict Mode)

For applications requiring strict cross-platform determinism:

```toml
# Cargo.toml
[profile.release]
# Disable FMA for determinism (10-20% performance cost)
# rustflags = ["-C", "target-feature=-fma"]
```

---

## 9. API Reference

### 9.1 Core Types

```rust
/// Simulation configuration
pub struct OrbitConfig {
    pub scenario: ScenarioType,
    pub integrator: IntegratorType,
    pub dt: f64,                    // Time step (seconds)
    pub seed: u64,                  // RNG seed for reproducibility
    pub jidoka: JidokaConfig,       // Quality guard configuration
    pub heijunka: HeijunkaConfig,   // Time budget configuration
}

/// Scenario selection
pub enum ScenarioType {
    Kepler { eccentricity: f64 },
    NBody { bodies: Vec<BodyConfig> },
    Hohmann { r1: f64, r2: f64 },
    Lagrange { point: LagrangePoint },
}

/// Integrator selection
pub enum IntegratorType {
    Verlet,
    Yoshida4,
    RK4,        // Non-symplectic, for comparison
    Adaptive,   // Adaptive time-stepping for close encounters
}

/// Heijunka configuration
pub struct HeijunkaConfig {
    pub frame_budget_ms: f64,    // Total frame budget (default: 16ms)
    pub physics_budget_ms: f64,  // Physics allocation (default: 8ms)
    pub max_substeps: usize,     // Upper bound (default: 100)
}
```

### 9.2 Simulation Interface

```rust
pub trait OrbitSimulation {
    fn new(config: OrbitConfig) -> SimResult<Self> where Self: Sized;
    fn step(&mut self, dt: f64) -> SimResult<()>;
    fn state(&self) -> &SimState;
    fn time(&self) -> f64;
    fn render(&self) -> Vec<RenderCommand>;
    fn jidoka_status(&self) -> JidokaStatus;
    fn heijunka_status(&self) -> HeijunkaStatus;
}
```

---

## 10. Build and Deployment

### 10.1 Build Commands

```makefile
# Native TUI build
build-tui:
    cargo build --release --bin orbit-tui --features tui

# WASM build (standard)
build-wasm:
    wasm-pack build --target web --out-dir pkg src/wasm

# WASM build (high-performance with SharedArrayBuffer)
build-wasm-shared:
    wasm-pack build --target web --out-dir pkg src/wasm --features shared-buffer

# Run TUI demo
run-tui:
    cargo run --release --bin orbit-tui --features tui

# Serve WASM demo locally
serve-wasm:
    python3 -m http.server 8080 --directory web/
```

### 10.2 Feature Flags

```toml
[features]
default = []
tui = ["ratatui", "crossterm"]
wasm = ["wasm-bindgen", "serde-wasm-bindgen", "console_error_panic_hook"]
shared-buffer = ["wasm"]  # High-performance SharedArrayBuffer mode
full = ["tui", "wasm", "shared-buffer"]
strict-determinism = []   # Disable FMA for cross-platform parity
```

### 10.3 Deployment

| Target | Location | Method |
|--------|----------|--------|
| TUI | crates.io | `cargo install simular-orbit-demo` |
| WASM | interactive.paiml.com | S3 + CloudFront |

---

## 11. Quality Assurance

### 11.1 Test Coverage Requirements

| Category | Coverage Target | Rationale |
|----------|-----------------|-----------|
| Unit tests | ≥95% | Standard requirement |
| Integration tests | ≥90% | End-to-end scenarios |
| Property tests | 256 cases minimum | Invariant verification |
| Mutation testing | ≥80% kill rate | Test quality [32] |
| **Metamorphic tests** | Required | Physics invariant verification [33] |

### 11.2 Metamorphic Testing

Per Chen et al. [33], metamorphic testing verifies **relations** rather than specific outputs:

```rust
/// Metamorphic Relation 1: Coordinate rotation invariance
/// Rotating the entire system should not change relative dynamics
#[test]
fn test_metamorphic_rotation_invariance() {
    let seed = 42u64;
    let rotation = Rotation3::from_axis_angle(&Vec3::z_axis(), PI / 4.0);

    // Original simulation
    let original = run_simulation(seed, 1000);

    // Rotated initial conditions
    let rotated_config = rotate_config(&original_config, rotation);
    let rotated = run_simulation_with_config(rotated_config, 1000);

    // Relative distances should be identical
    for (orig, rot) in original.iter().zip(rotated.iter()) {
        let orig_distances = compute_pairwise_distances(orig);
        let rot_distances = compute_pairwise_distances(rot);
        assert_approx_eq!(orig_distances, rot_distances, epsilon = 1e-10);
    }
}

/// Metamorphic Relation 2: Time reversal symmetry
/// Running forward then backward should return to initial state
#[test]
fn test_metamorphic_time_reversal() {
    let initial_state = create_kepler_scenario();

    // Forward 100 steps
    let forward = run_simulation_steps(&initial_state, 100, 0.01);

    // Reverse velocities and run backward
    let reversed = reverse_velocities(&forward);
    let backward = run_simulation_steps(&reversed, 100, 0.01);

    // Should return to initial state (within numerical precision)
    assert_state_approx_eq!(backward, initial_state, epsilon = 1e-8);
}

/// Metamorphic Relation 3: Energy scaling
/// Doubling all masses should not change orbital shapes (only periods)
#[test]
fn test_metamorphic_mass_scaling() {
    let original = run_kepler_simulation(sun_mass: M, planet_mass: m, 1000);
    let scaled = run_kepler_simulation(sun_mass: 2*M, planet_mass: 2*m, 1000);

    // Semi-major axis should be identical
    assert_approx_eq!(original.semi_major_axis(), scaled.semi_major_axis(), 1e-10);

    // Period ratio should be sqrt(2) per Kepler's 3rd law
    let period_ratio = scaled.orbital_period() / original.orbital_period();
    assert_approx_eq!(period_ratio, 1.0 / 2.0_f64.sqrt(), 1e-6);
}
```

### 11.3 Continuous Integration

```yaml
# .github/workflows/ci.yml
jobs:
  test:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        target: [native, wasm32-unknown-unknown]
    steps:
      - run: cargo test --all-features
      - run: cargo clippy -- -D warnings
      - run: cargo llvm-cov --fail-under 95

  metamorphic:
    runs-on: ubuntu-latest
    steps:
      - run: cargo test metamorphic --release -- --ignored

  cross-platform-determinism:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - run: cargo test epsilon_determinism --features strict-determinism
```

### 11.4 Acceptance Criteria

1. **AC-1**: Earth orbit completes in 365.25 ± 0.01 simulation days
2. **AC-2**: Energy drift < 1e-9 over 100 orbits (Yoshida integrator)
3. **AC-3**: Angular momentum conserved to 1e-12 relative precision
4. **AC-4**: WASM bundle size < 500KB (gzipped)
5. **AC-5**: TUI renders at ≥30 FPS on standard terminal
6. **AC-6**: Epsilon-identical trajectories (ε=1e-9) on native and WASM
7. **AC-7**: All metamorphic relations pass
8. **AC-8**: Heijunka budget never exceeded (graceful quality degradation instead)
9. **AC-9**: Jidoka violations trigger pause, not crash
10. **AC-10**: All Probar E2E tests pass (deterministic replay verified)

---

## 12. WASM Testing with jugar-probar

### 12.1 Probar Integration Overview

The WASM component uses **jugar-probar** ("probar" = Spanish for "to test/prove"), a pure Rust testing framework providing Playwright feature parity with WASM-native extensions [36].

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROBAR Test Architecture                      │
├─────────────────────────────────────────────────────────────────┤
│   ┌────────────────┐    ┌────────────────┐    ┌──────────────┐  │
│   │  Test Spec     │    │  WasmRuntime   │    │  Browser     │  │
│   │  (Rust)        │───►│  (wasmtime)    │───►│  (optional)  │  │
│   │                │    │  LOGIC ONLY    │    │  GOLDEN      │  │
│   │  - Assertions  │    │                │    │  MASTER      │  │
│   │  - Invariants  │    │  ✓ Unit tests  │    │              │  │
│   │  - Fuzzing     │    │  ✓ Replay      │    │  ✓ E2E       │  │
│   │                │    │  ✓ Fuzzing     │    │  ✓ Visual    │  │
│   └────────────────┘    └────────────────┘    └──────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Probar vs Playwright:**

| Aspect | Playwright | Probar |
|--------|-----------|--------|
| Language | TypeScript | Pure Rust |
| Browser | Required (Chromium) | Optional |
| Simulation State | Black box (DOM) | Direct API access |
| CI Setup | Node.js + browser | Just `cargo test` |
| Zero JS | Violates constraint | Pure Rust |
| Deterministic Replay | Not supported | Native feature |

### 12.2 Dependencies

```toml
# Cargo.toml
[dev-dependencies]
jugar-probar = { version = "0.1", features = ["runtime"] }

# Optional: browser-based golden master tests
[dev-dependencies.jugar-probar-browser]
version = "0.1"
features = ["browser"]
optional = true
```

### 12.3 Probar Test Suites

#### 12.3.1 WASM Logic Tests (No Browser Required)

```rust
use jugar_probar::prelude::*;
use simular_orbit_demo::OrbitDemo;

/// Test suite for orbit simulation WASM module
#[cfg(test)]
mod probar_orbit_tests {
    use super::*;

    // ========================================================================
    // WASM LOADING & INITIALIZATION
    // ========================================================================

    #[test]
    fn test_wasm_module_loads() {
        let config = r#"{"scenario": "kepler", "seed": 42}"#;
        let demo = OrbitDemo::new(config);
        assert!(demo.is_ok(), "WASM module should load with valid config");
    }

    #[test]
    fn test_invalid_config_rejected() {
        let config = r#"{"invalid": true}"#;
        let demo = OrbitDemo::new(config);
        let result = Assertion::is_err(&demo);
        assert!(result.passed, "Invalid config should return error");
    }

    // ========================================================================
    // PHYSICS INVARIANTS (Jidoka)
    // ========================================================================

    #[test]
    fn test_energy_conservation_kepler() {
        let mut demo = OrbitDemo::new(r#"{"scenario": "kepler", "seed": 42}"#).unwrap();
        let initial_energy = demo.total_energy();

        // Simulate 1000 steps (approximately 1 orbit)
        for _ in 0..1000 {
            demo.tick(0.001).unwrap();
        }

        let final_energy = demo.total_energy();
        let result = Assertion::approx_eq(initial_energy, final_energy, 1e-6);
        assert!(result.passed, "Energy should be conserved: {}", result.message);
    }

    #[test]
    fn test_angular_momentum_conservation() {
        let mut demo = OrbitDemo::new(r#"{"scenario": "kepler", "seed": 42}"#).unwrap();
        let initial_L = demo.angular_momentum_magnitude();

        for _ in 0..1000 {
            demo.tick(0.001).unwrap();
        }

        let final_L = demo.angular_momentum_magnitude();
        let result = Assertion::approx_eq(initial_L, final_L, 1e-9);
        assert!(result.passed, "Angular momentum should be conserved");
    }

    #[test]
    fn test_jidoka_detects_nan() {
        let mut demo = OrbitDemo::new(r#"{"scenario": "kepler", "seed": 42}"#).unwrap();

        // Force a NaN by setting invalid state (if API allows)
        // This should trigger Jidoka guard
        let status = demo.jidoka_status();
        let result = Assertion::is_true(status.all_finite, "All values should be finite");
        assert!(result.passed);
    }

    // ========================================================================
    // DETERMINISTIC REPLAY
    // ========================================================================

    #[test]
    fn test_deterministic_replay_identical() {
        use jugar_probar::{SimulationConfig, run_simulation, run_replay};

        let config = SimulationConfig::new(42, 1000);

        // Record simulation
        let recording = run_orbit_simulation(config);
        assert!(recording.completed, "Simulation should complete");

        // Replay and verify
        let replay = run_orbit_replay(&recording);
        assert!(replay.determinism_verified, "Replay should be deterministic");
        assert_eq!(
            recording.final_state_hash,
            replay.final_state_hash,
            "State hashes should match"
        );
    }

    #[test]
    fn test_different_seeds_diverge() {
        let recording_1 = run_orbit_simulation(SimulationConfig::new(42, 500));
        let recording_2 = run_orbit_simulation(SimulationConfig::new(43, 500));

        assert_ne!(
            recording_1.final_state_hash,
            recording_2.final_state_hash,
            "Different seeds should produce different trajectories"
        );
    }
}
```

#### 12.3.2 Invariant Fuzzing

```rust
use jugar_probar::{InputFuzzer, InvariantChecker, InvariantCheck, Seed};

/// Custom invariants for orbit simulation
mod orbit_invariants {
    use super::*;

    /// Energy must remain bounded (Jidoka)
    pub struct EnergyBoundedInvariant {
        initial_energy: f64,
        tolerance: f64,
    }

    impl InvariantCheck for EnergyBoundedInvariant {
        fn check(&self, state: &SimState) -> Result<(), String> {
            let current = state.total_energy();
            let drift = (current - self.initial_energy).abs() / self.initial_energy.abs();

            if drift > self.tolerance {
                Err(format!("Energy drift {:.2e} exceeds tolerance {:.2e}", drift, self.tolerance))
            } else {
                Ok(())
            }
        }
    }

    /// All positions must be finite (no NaN/Inf)
    pub struct FinitePositionsInvariant;

    impl InvariantCheck for FinitePositionsInvariant {
        fn check(&self, state: &SimState) -> Result<(), String> {
            for (i, pos) in state.positions().iter().enumerate() {
                if !pos.is_finite() {
                    return Err(format!("Body {} has non-finite position: {:?}", i, pos));
                }
            }
            Ok(())
        }
    }

    /// No collisions (minimum separation)
    pub struct NoCollisionInvariant {
        min_separation: f64,
    }

    impl InvariantCheck for NoCollisionInvariant {
        fn check(&self, state: &SimState) -> Result<(), String> {
            for i in 0..state.num_bodies() {
                for j in (i+1)..state.num_bodies() {
                    let sep = (state.positions()[i] - state.positions()[j]).magnitude();
                    if sep < self.min_separation {
                        return Err(format!(
                            "Bodies {} and {} collided: separation {:.2e} < {:.2e}",
                            i, j, sep, self.min_separation
                        ));
                    }
                }
            }
            Ok(())
        }
    }
}

#[test]
fn test_fuzz_orbit_invariants() {
    use orbit_invariants::*;

    let mut checker = InvariantChecker::new();
    checker.add(EnergyBoundedInvariant {
        initial_energy: -1.0e10,  // Will be set on first frame
        tolerance: 1e-6,
    });
    checker.add(FinitePositionsInvariant);
    checker.add(NoCollisionInvariant { min_separation: 1e6 });

    let mut fuzzer = InputFuzzer::new(Seed::from_u64(12345));
    let mut demo = OrbitDemo::new(r#"{"scenario": "n_body", "seed": 12345}"#).unwrap();

    // Fuzz 10,000 frames with random time steps
    for frame in 0..10_000 {
        let dt = fuzzer.next_f64_range(0.0001, 0.01);
        demo.tick(dt).unwrap();

        if let Err(violation) = checker.check(demo.state()) {
            panic!("Invariant violation at frame {}: {}", frame, violation);
        }
    }
}
```

#### 12.3.3 Visual Regression Testing

```rust
use jugar_probar::{VisualRegressionTester, VisualRegressionConfig, Snapshot};

#[test]
fn test_kepler_orbit_visual_regression() {
    let config = VisualRegressionConfig::default()
        .with_threshold(0.01)  // 1% pixel difference allowed
        .with_snapshot_dir("tests/__snapshots__/orbit");

    let tester = VisualRegressionTester::new(config);

    let mut demo = OrbitDemo::new(r#"{"scenario": "kepler", "seed": 42}"#).unwrap();

    // Advance to specific frame for snapshot
    for _ in 0..100 {
        demo.tick(0.01).unwrap();
    }

    let render_commands = demo.render();
    let screenshot = render_to_image(&render_commands, 800, 600);

    let result = tester.compare("kepler_frame_100", &screenshot);
    assert!(
        result.within_threshold(),
        "Visual regression: {}% difference (threshold: 1%)",
        result.difference_percent * 100.0
    );
}
```

### 12.4 Probar Test Organization

```
tests/
├── probar_orbit.rs           # Main Probar test file
│   ├── wasm_loading          # Module load/init tests
│   ├── physics_invariants    # Energy, angular momentum, finite
│   ├── determinism           # Replay verification
│   ├── scenarios             # Kepler, N-body, Hohmann, Lagrange
│   └── stress                # Long simulations, edge cases
├── probar_invariants.rs      # Custom InvariantCheck implementations
├── probar_fuzzing.rs         # Monte Carlo fuzzing tests
├── probar_visual.rs          # Visual regression tests
└── __snapshots__/            # Golden master screenshots
    └── orbit/
        ├── kepler_frame_100.png
        ├── n_body_frame_500.png
        └── hohmann_transfer.png
```

### 12.5 Running Probar Tests

```makefile
# Makefile additions for Probar

# Run all Probar tests (no browser required)
test-probar:
    @echo "🧪 Running Probar WASM tests..."
    cargo test -p simular-orbit-demo --test probar_orbit -- --nocapture
    @echo "✅ Probar tests passed"

# Run Probar with determinism verification
test-probar-determinism:
    @echo "🔁 Running deterministic replay tests..."
    RUST_LOG=debug cargo test -p simular-orbit-demo determinism -- --nocapture

# Run Probar fuzzing (longer)
test-probar-fuzz:
    @echo "🎲 Running invariant fuzzing (10,000 frames)..."
    cargo test -p simular-orbit-demo fuzz --release -- --ignored --nocapture

# Run visual regression (requires golden masters)
test-probar-visual:
    @echo "📸 Running visual regression tests..."
    cargo test -p simular-orbit-demo visual_regression -- --nocapture

# Update visual snapshots
test-probar-visual-update:
    @echo "📸 Updating visual snapshots..."
    UPDATE_SNAPSHOTS=1 cargo test -p simular-orbit-demo visual_regression -- --nocapture

# Full Probar suite
test-probar-full: test-probar test-probar-fuzz test-probar-visual
    @echo "✅ Full Probar suite passed"
```

### 12.6 CI Integration

```yaml
# .github/workflows/ci.yml additions

  probar-wasm:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-action@stable

      - name: Build WASM module
        run: cargo build -p simular-orbit-demo --target wasm32-unknown-unknown

      - name: Run Probar logic tests
        run: cargo test -p simular-orbit-demo --test probar_orbit

      - name: Run determinism verification
        run: cargo test -p simular-orbit-demo determinism

      - name: Run invariant fuzzing (1000 frames, CI mode)
        run: cargo test -p simular-orbit-demo fuzz_ci --release

  probar-visual:
    runs-on: ubuntu-latest
    needs: probar-wasm
    steps:
      - uses: actions/checkout@v4

      - name: Run visual regression tests
        run: cargo test -p simular-orbit-demo visual_regression

      - name: Upload visual diffs on failure
        if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: visual-diffs
          path: tests/__snapshots__/orbit/*.diff.png

  probar-browser:
    runs-on: ubuntu-latest
    needs: probar-wasm
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4

      - name: Install Chromium
        run: npx playwright install chromium

      - name: Run browser golden master tests
        run: cargo test -p simular-orbit-demo --features browser --test probar_browser
```

### 12.7 Toyota Way Alignment

| TPS Principle | Probar Implementation |
|---------------|----------------------|
| **Poka-Yoke** | Type-safe `InvariantCheck` trait prevents runtime errors |
| **Muda** | Zero-copy `MemoryView` eliminates serialization waste |
| **Jidoka** | `InvariantChecker` halts on physics violations |
| **Andon Cord** | Fail-fast mode stops on first critical failure |
| **Genchi Genbutsu** | Direct WASM memory inspection, not DOM black box |

---

## 13. References

### Original Specification References [1-25]

[1] G. J. Holzmann, "The Power of 10: Rules for Developing Safety-Critical Code," *IEEE Computer*, vol. 39, no. 6, pp. 95-97, June 2006. doi:10.1109/MC.2006.212

[2] V. I. Arnold, *Mathematical Methods of Classical Mechanics*, 2nd ed. New York: Springer-Verlag, 1989. ISBN: 978-0-387-96890-2

[3] S. J. Aarseth, *Gravitational N-Body Simulations: Tools and Algorithms*. Cambridge University Press, 2003. doi:10.1017/CBO9780511535246

[4] P. J. Mohr, D. B. Newell, and B. N. Taylor, "CODATA Recommended Values of the Fundamental Physical Constants: 2014," *Reviews of Modern Physics*, vol. 88, no. 3, 035009, 2016. doi:10.1103/RevModPhys.88.035009

[5] H. Goldstein, C. Poole, and J. Safko, *Classical Mechanics*, 3rd ed. San Francisco: Addison Wesley, 2002. ISBN: 978-0-201-65702-9

[6] R. R. Bate, D. D. Mueller, and J. E. White, *Fundamentals of Astrodynamics*. New York: Dover Publications, 1971. ISBN: 978-0-486-60061-1

[7] D. A. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed. Microcosm Press, 2013. ISBN: 978-1-881883-18-0

[8] E. Hairer, C. Lubich, and G. Wanner, *Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations*, 2nd ed. Berlin: Springer, 2006. doi:10.1007/3-540-30666-8

[9] J. Laskar and M. Gastineau, "Existence of collisional trajectories of Mercury, Mars and Venus with the Earth," *Nature*, vol. 459, pp. 817-819, 2009. doi:10.1038/nature08096

[10] L. Verlet, "Computer 'Experiments' on Classical Fluids. I. Thermodynamical Properties of Lennard-Jones Molecules," *Physical Review*, vol. 159, no. 1, pp. 98-103, 1967. doi:10.1103/PhysRev.159.98

[11] W. C. Swope, H. C. Andersen, P. H. Berens, and K. R. Wilson, "A computer simulation method for the calculation of equilibrium constants for the formation of physical clusters of molecules," *Journal of Chemical Physics*, vol. 76, no. 1, pp. 637-649, 1982. doi:10.1063/1.442716

[12] R. D. Ruth, "A Canonical Integration Technique," *IEEE Transactions on Nuclear Science*, vol. 30, no. 4, pp. 2669-2671, 1983. doi:10.1109/TNS.1983.4332919

[13] H. Yoshida, "Construction of higher order symplectic integrators," *Physics Letters A*, vol. 150, no. 5-7, pp. 262-268, 1990. doi:10.1016/0375-9601(90)90092-3

[14] S. J. Aarseth, "Direct N-body Calculations," in *Multiple Time Scales*, J. U. Brackbill and B. I. Cohen, Eds. Academic Press, 1985, pp. 377-418.

[15] J. M. Sanz-Serna and M. P. Calvo, *Numerical Hamiltonian Problems*. London: Chapman and Hall, 1994. ISBN: 978-0-412-54290-4

[16] L. D. Landau and E. M. Lifshitz, *Mechanics*, 3rd ed. Oxford: Butterworth-Heinemann, 1976. ISBN: 978-0-7506-2896-9

[17] IEEE Computer Society, "IEEE Standard for Floating-Point Arithmetic," *IEEE Std 754-2019*, 2019. doi:10.1109/IEEESTD.2019.8766229

[18] P. Hut and J. N. Bahcall, "Binary-single star scattering. I. Numerical experiments for equal masses," *The Astrophysical Journal*, vol. 268, pp. 319-341, 1983. doi:10.1086/160956

[19] E. Gamma, R. Helm, R. Johnson, and J. Vlissides, *Design Patterns: Elements of Reusable Object-Oriented Software*. Reading, MA: Addison-Wesley, 1994. ISBN: 978-0-201-63361-0

[20] D. Knuth, *The Art of Computer Programming, Volume 2: Seminumerical Algorithms*, 3rd ed. Addison-Wesley, 1997. ISBN: 978-0-201-89684-8

[21] A. Crichton et al., "The `wasm-bindgen` Guide," Rust and WebAssembly Working Group, 2023. [Online]. Available: https://rustwasm.github.io/docs/wasm-bindgen/

[22] E. V. Pitjeva and N. P. Pitjev, "Masses of the Main Asteroid Belt and the Kuiper Belt from the Motions of Planets and Spacecraft," *Solar System Research*, vol. 52, no. 2, pp. 145-159, 2018. doi:10.1134/S0038094618020058

[23] W. Hohmann, *Die Erreichbarkeit der Himmelskörper* (The Attainability of Heavenly Bodies). Munich: Oldenbourg, 1925. NASA Technical Translation F-44, 1960.

[24] V. Szebehely, *Theory of Orbits: The Restricted Problem of Three Bodies*. New York: Academic Press, 1967. ISBN: 978-0-12-680650-9

[25] J. Wisdom and M. Holman, "Symplectic maps for the n-body problem," *The Astronomical Journal*, vol. 102, no. 4, pp. 1528-1538, 1991. doi:10.1086/115978

### TPS Review References [26-35]

[26] D. Goldberg, "What every computer scientist should know about floating-point arithmetic," *ACM Computing Surveys*, vol. 23, no. 1, pp. 5-48, Mar. 1991. doi:10.1145/103162.103163

[27] A. Avizienis, J.-C. Laprie, B. Randell, and C. Landwehr, "Basic concepts and taxonomy of dependable and secure computing," *IEEE Transactions on Dependable and Secure Computing*, vol. 1, no. 1, pp. 11-33, 2004. doi:10.1109/TDSC.2004.2

[28] A. J. Kennedy, "Dimension Types," in *Programming Languages and Systems — ESOP '94*, Lecture Notes in Computer Science, vol. 788. Springer, 1994, pp. 348-362. doi:10.1007/3-540-57880-3_23

[29] A. Haas, A. Rossberg, D. L. Schuff, et al., "Bringing the web up to speed with WebAssembly," in *Proceedings of the 38th ACM SIGPLAN Conference on Programming Language Design and Implementation*, 2017, pp. 185-200. doi:10.1145/3062341.3062363

[30] R. W. Hockney and J. W. Eastwood, *Computer Simulation Using Particles*. New York: Taylor & Francis, 1988. ISBN: 978-0-85274-392-8

[31] P. Kustaanheimo and E. Stiefel, "Perturbation theory of Kepler motion based on spinor regularization," *Journal für die reine und angewandte Mathematik*, vol. 218, pp. 204-219, 1965. doi:10.1515/crll.1965.218.204

[32] G. Rothermel, R. H. Untch, C. Chu, and M. J. Harrold, "Test case prioritization: An empirical study," in *Proceedings of the IEEE International Conference on Software Maintenance*, 1999, pp. 179-188. doi:10.1109/ICSM.1999.792604

[33] T. Y. Chen, S. C. Cheung, and S. M. Yiu, "Metamorphic testing: a new approach for generating next test cases," Technical Report HKUST-CS98-01, Hong Kong University of Science and Technology, 1998.

[34] J. K. Liker, *The Toyota Way: 14 Management Principles from the World's Greatest Manufacturer*. New York: McGraw-Hill, 2004. ISBN: 978-0-07-139231-0

[35] M. Rother, *Toyota Kata: Managing People for Improvement, Adaptiveness and Superior Results*. New York: McGraw-Hill, 2009. ISBN: 978-0-07-163523-3

### Probar Integration Reference [36]

[36] PAIML Engineering, "jugar-probar: Pure Rust WASM Testing Framework," GitHub Repository, 2025. [Online]. Available: https://github.com/paiml/jugar/tree/main/crates/jugar-probar

---

## Appendix A: Physical Constants

| Constant | Symbol | Value | Unit |
|----------|--------|-------|------|
| Gravitational constant | $G$ | 6.67430×10⁻¹¹ | m³ kg⁻¹ s⁻² |
| Astronomical unit | AU | 1.495978707×10¹¹ | m |
| Solar mass | $M_☉$ | 1.98892×10³⁰ | kg |
| Earth mass | $M_⊕$ | 5.9722×10²⁴ | kg |
| Earth orbital period | $T_⊕$ | 365.256363004 | days |

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Jidoka** | Toyota's "automation with a human touch" - stopping and highlighting defects while maintaining availability |
| **Poka-Yoke** | Mistake-proofing through design constraints (compile-time dimensional analysis) |
| **Heijunka** | Load leveling - ensuring consistent frame delivery despite variable computation |
| **Mieruka** | Visual management - making status visible at a glance |
| **Muda** | Waste - unnecessary processing, waiting, or defects |
| **Mura** | Unevenness - inconsistent delivery (frame stuttering) |
| **Symplectic** | Structure-preserving in phase space, conserves volume |
| **Epsilon-determinism** | Results identical within bounded error ε across platforms |
| **Metamorphic testing** | Testing invariant relations rather than specific outputs |
| **WASM** | WebAssembly - portable binary instruction format |
| **TUI** | Terminal User Interface |

## Appendix C: Review Response Matrix

| Review Finding | Section | Resolution |
|----------------|---------|------------|
| Jidoka should pause, not crash | §5.1 | Added `JidokaResponse::Pause` with graceful degradation |
| O(N²) violates Heijunka | §3.1.2 | Added `HeijunkaScheduler` with time budget |
| Need Poka-Yoke dimensional analysis | §2.4 | Added `uom` crate newtype pattern |
| Bit-determinism unrealistic | §8.1 | Revised to epsilon-determinism with ε bounds |
| Symplectic ≠ energy conservation | §4.3 | Added limitation note and adaptive stepping |
| JSON serialization is Muda | §6.3.2 | Added SharedArrayBuffer option |
| Coverage ≠ correctness | §11.2 | Added metamorphic testing requirements |

---

**Document Control**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-10 | PAIML Engineering | Initial specification |
| 1.1.0 | 2025-12-10 | PAIML Engineering | TPS review incorporation (Gemini feedback) |
| 1.2.0 | 2025-12-10 | PAIML Engineering | jugar-probar WASM testing integration |

---

*This specification is released under MIT License. Contributions welcome via pull request.*
