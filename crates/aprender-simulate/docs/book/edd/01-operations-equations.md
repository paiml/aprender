# Operations Science Equations

## The Mathematical Foundation of Lean/TPS

Operations science provides the mathematical foundation for understanding flow, queues, inventory, and variability in any system. These equations are the governing laws that simulations must respect.

## 1. Little's Law: L = λW

**The most fundamental equation in queueing theory.**

```
L = λW

Where:
  L = Average number of items in system (WIP)
  λ = Average arrival rate (Throughput)
  W = Average time in system (Cycle Time)
```

### Citation

Little, J.D.C. (1961). "A Proof for the Queuing Formula: L = λW". *Operations Research*, 9(3), 383-387.

### Key Insight

Little's Law is **distribution-free** - it holds regardless of arrival distribution, service distribution, or queue discipline. This makes it incredibly powerful for validation.

### Usage in simular

```rust
use simular::edd::{LittlesLaw, GoverningEquation};

let law = LittlesLaw::new();

// Evaluate: λ=5 items/hr, W=2 hrs → L=10 items
let wip = law.evaluate(5.0, 2.0);
assert_eq!(wip, 10.0);

// Validate simulation output
let is_valid = law.validate(10.0, 5.0, 2.0, 0.01);
assert!(is_valid.is_ok());
```

### TPS Application

Little's Law explains why **CONWIP (Constant WIP)** works: by fixing L, any increase in λ (throughput) must come from decreased W (cycle time).

---

## 2. Kingman's Formula (VUT Equation)

**The "hockey stick" that explains why high utilization kills flow.**

```
W_q ≈ (ρ / (1-ρ)) × ((c_a² + c_s²) / 2) × t_s

Where:
  W_q = Expected waiting time in queue
  ρ   = Utilization (arrival rate / service rate)
  c_a = Coefficient of variation of arrivals
  c_s = Coefficient of variation of service
  t_s = Mean service time
```

### Citation

Kingman, J.F.C. (1961). "The single server queue in heavy traffic". *Annals of Mathematical Statistics*, 32(4), 1314-1324.

### The Hockey Stick Effect

| Utilization | Wait Time | Ratio to 50% |
|-------------|-----------|--------------|
| 50%         | 1.0       | 1.0x         |
| 70%         | 2.3       | 2.3x         |
| 80%         | 4.0       | 4.0x         |
| 90%         | 9.0       | 9.0x         |
| 95%         | 19.0      | 19.0x        |
| 99%         | 99.0      | 99.0x        |

### Usage in simular

```rust
use simular::edd::{KingmanFormula, GoverningEquation};

let formula = KingmanFormula::new();

// High variability (c_a=1, c_s=1), service time 1 hour
let wait_95 = formula.expected_wait_time(0.95, 1.0, 1.0, 1.0);
assert!(wait_95 > 15.0); // ~19 hours wait at 95% utilization
```

### TPS Application

This is why Toyota targets **~65% utilization**, not 95%+. The exponential wait time at high utilization destroys flow.

---

## 3. Square Root Law (Safety Stock)

**Inventory scales as the square root of demand variability, not linearly.**

```
I_safety = z × σ_D × √L

Where:
  I_safety = Safety stock required
  z        = Service level factor (1.96 for 97.5%)
  σ_D      = Standard deviation of demand
  L        = Lead time (in periods)
```

### Citation

Eppen, G.D. (1979). "Effects of Centralization on Expected Costs in a Multi-Location Newsboy Problem". *Management Science*, 25(5), 498-501.

### Key Insight

If lead time **quadruples**, safety stock only **doubles** (√4 = 2). Linear thinking would predict 4x increase.

### Usage in simular

```rust
use simular::edd::{SquareRootLaw, GoverningEquation};

let law = SquareRootLaw::new();

// σ_D=100, L=1, z=1.96 → 196 units
let stock_l1 = law.safety_stock(100.0, 1.0, 1.96);

// σ_D=100, L=4, z=1.96 → 392 units (2x, not 4x)
let stock_l4 = law.safety_stock(100.0, 4.0, 1.96);

assert!((stock_l4 / stock_l1 - 2.0).abs() < 0.01);
```

### TPS Application

This justifies **supermarket systems** and **kanban sizing** - safety stock doesn't scale linearly with demand uncertainty.

---

## 4. Bullwhip Effect (Variance Amplification)

**Demand variance amplifies upstream through supply chains.**

```
Var(Orders) / Var(Demand) ≥ 1 + (2L/p) + (2L²/p²)

Where:
  L = Lead time
  p = Number of periods in forecast moving average
```

### Citation

Lee, H.L., Padmanabhan, V., & Whang, S. (1997). "The Bullwhip Effect in Supply Chains". *Sloan Management Review*, 38(3), 93-102.

### Multi-Echelon Amplification

With L=2, p=4 (amplification factor ~2.5x):

| Echelon                    | Cumulative Amplification |
|---------------------------|-------------------------|
| Retailer → Wholesaler      | 2.5x                    |
| Wholesaler → Distributor   | 6.25x                   |
| Distributor → Manufacturer | 15.6x                   |

### Usage in simular

```rust
use simular::edd::{BullwhipEffect, GoverningEquation};

let effect = BullwhipEffect::new();

// Lead time 2, forecast period 4
let amp = effect.amplification_factor(2.0, 4.0);
assert!(amp > 2.0); // ~2.5x minimum amplification
```

### TPS Application

**Heijunka** (production leveling) acts as a low-pass filter, dampening demand variance before it amplifies upstream.

---

## Running the Examples

```bash
# Demonstrate all operations equations
cargo run --example edd_operations

# Validate TPS test cases
cargo run --example edd_tps_validation
```

## Next Chapter

Learn about Equation Model Cards - the mandatory documentation that bridges mathematics and code.
