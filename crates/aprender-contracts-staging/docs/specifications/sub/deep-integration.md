# Sub-spec: Deep Stack Integration

**Parent:** [pv-spec.md](../pv-spec.md) Section 24

---

## Motivation

The provable-contracts system currently enforces contract-to-implementation
binding (Sections 9, 23) but does NOT integrate with:

1. **apr-cli inference pipeline** — `apr serve plan` computes roofline
   bounds independently of `roofline-model-v1.yaml`
2. **trueno BrickProfiler** — `ComputeBrick` budget enforcement is
   hardcoded, not derived from YAML contracts
3. **trueno ModelTracer** — 5 trace types (MLT-01..05) observe but
   don't verify contract postconditions
4. **pmat CB-1200+ enforcement** — 10 checks exist but contract trait
   enforcement (CB-1209) is the newest and least exercised

This section closes these gaps to make contracts first-class citizens
in the inference, profiling, and quality pipelines.

---

## Gap Analysis

### Gap 1: apr-cli Roofline Disconnected from Contracts

**Current:** `apr serve plan` uses hardcoded roofline formulas in
`serve_commands.rs`. The `roofline-model-v1.yaml` contract defines
the same equations (`bw_ceiling`, `compute_ceiling`, `throughput_bound`)
but apr-cli doesn't read them.

**Fix:** `apr serve plan` should derive performance ceilings from the
contract YAML, not from inline constants. This makes the roofline
model a single source of truth: change the YAML → the serve plan
updates automatically.

### Gap 2: trueno ComputeBrick Budget Not Contract-Derived

**Current:** `ComputeBrick` has `enforce_budget: bool` and
`TokenBudget`, but thresholds are set manually. The
`roofline-model-v1.yaml` and `kernel-launch-budget-v1.yaml` contracts
define the same bounds but trueno doesn't read them.

**Fix:** At startup, trueno loads the contract YAML and derives
`TokenBudget` from the roofline ceiling for the detected hardware.
When a brick violates its budget, emit a structured tracing event
with the contract ID and violation details.

### Gap 3: Tracing Not Contract-Aware

**Current:** trueno's `ModelTracer` records activation traces,
attention weights, and quantization errors, but doesn't check them
against contract invariants (e.g., "softmax output sums to 1",
"no NaN in attention weights").

**Fix:** A `ContractTracingLayer` (custom `tracing::Layer`) intercepts
spans tagged with `contract.id` and `contract.equation`, then verifies
postconditions against recorded values. Violations emit structured
diagnostics compatible with `pv lint` SARIF output.

### Gap 4: pmat CB-1209 Trait Enforcement Not Exercised

**Current:** CB-1209 exists in pmat but has minimal test coverage.
The 138 trait tests across 13 repos are local — pmat doesn't verify
them in the comply pipeline.

**Fix:** CB-1209 should run `cargo test --test contract_traits` in
each consumer repo during `pmat comply check`, verifying the full
trait enforcement pipeline end-to-end.

---

## Design

### Three-Tier Integration

```
Tier 1: Compile-Time (existing)
  YAML → build.rs → CONTRACT_* env vars → #[contract] debug_assert
  YAML → pv scaffold --trait → trait file → impl → compiler check

Tier 2: CI-Time (existing + enhanced)
  pv lint → 7 gates → SARIF findings
  pv verify-bindings → ghost binding detection
  pmat comply check → CB-1200..1209
  cargo test --test contract_traits → 138 trait tests

Tier 3: Runtime (NEW)
  apr serve plan → load roofline-model-v1.yaml → derive TPS ceiling
  ComputeBrick → load kernel-launch-budget-v1.yaml → enforce budget
  ContractTracingLayer → intercept spans → verify postconditions
  ModelTracer → check activations against contract invariants
```

### Contract-Aware Tracing (`tracing::Layer`)

```rust
// In trueno or aprender:
use tracing::Subscriber;
use tracing_subscriber::Layer;

pub struct ContractTracingLayer {
    contracts: HashMap<String, Contract>,
}

impl<S: Subscriber> Layer<S> for ContractTracingLayer {
    fn on_close(&self, id: span::Id, ctx: Context<'_, S>) {
        let span = ctx.span(&id).unwrap();
        if let Some(contract_id) = span.extensions().get::<ContractId>() {
            let contract = &self.contracts[&contract_id.0];
            // Check postconditions against span fields
            for ob in &contract.proof_obligations {
                if !ob.check_postcondition(span.fields()) {
                    tracing::error!(
                        contract = %contract_id.0,
                        obligation = %ob.property,
                        "Contract postcondition violated at runtime"
                    );
                }
            }
        }
    }
}
```

### BrickProfiler Contract Integration

```rust
// In trueno ComputeBrick:
impl ComputeBrick {
    pub fn from_contract(contract: &Contract, hw: &HardwareProfile) -> Self {
        let roofline = contract.equations.get("throughput_bound").unwrap();
        let ceiling = hw.memory_bandwidth.min(hw.compute_throughput);
        Self {
            budget: TokenBudget::from_throughput(ceiling * 0.9), // 90% of theoretical
            enforce_budget: true,
            contract_id: Some(contract.metadata.description.clone()),
        }
    }
}
```

### apr-cli Roofline from Contract

```rust
// In apr serve plan:
fn compute_roofline(model: &ModelConfig) -> RooflinePlan {
    let contract = provable_contracts::schema::parse_contract(
        Path::new("../provable-contracts/contracts/roofline-model-v1.yaml")
    ).ok();

    if let Some(c) = contract {
        // Derive ceilings from YAML equations
        let bw = c.equations["bw_ceiling"].evaluate(hw_params);
        let compute = c.equations["compute_ceiling"].evaluate(hw_params);
        RooflinePlan { bw_ceiling: bw, compute_ceiling: compute, source: "contract" }
    } else {
        // Fallback to hardcoded
        RooflinePlan::default()
    }
}
```

---

## pmat Enforcement Pipeline (CB-1200 through CB-1209)

| Check | What | Status |
|-------|------|--------|
| CB-1200 | `pv lint` passes | Deployed |
| CB-1201 | PV Lint threshold | Deployed |
| CB-1202 | Contract coverage | Deployed |
| CB-1203 | `#[contract]` annotations | Deployed (49 fns) |
| CB-1204 | build.rs `CONTRACT_*` pipeline | Deployed |
| CB-1205 | Provability invariant (kani+falsify) | Deployed |
| CB-1206 | L1-L5 verification levels | Deployed |
| CB-1207 | Contract drift detection | Deployed |
| CB-1208 | Binding existence (`pv verify-bindings`) | Deployed |
| CB-1209 | Trait enforcement (`tests/contract_traits.rs`) | Deployed |

### Integration with `pmat comply check`

```bash
# Full enforcement pipeline (all 10 checks):
pmat comply check --provable-contracts

# Runs:
# 1. pv lint contracts/ --format json
# 2. pv score contracts/ --format json
# 3. pv verify-bindings contracts/*/binding.yaml
# 4. cargo test --test contract_traits (in each consumer repo)
# 5. Checks CB-1200 through CB-1209
```

---

## References

### Performance Verification

1. Williams, S. et al. (2009). "Roofline: An Insightful Visual
   Performance Model." CACM 52(4).
2. Yuan, Z. et al. (2024). "LLM Inference Unveiled: Survey and
   Roofline Model Insights." arXiv:2402.16363.
3. Yang, D. et al. (2020). "Hierarchical Roofline Performance
   Analysis for Deep Learning." arXiv:2009.05257.

### Contract Verification

4. Denis, X. et al. (2026). "Creusot: Formal Verification of Rust
   Programs." POPL 2026 Tutorial.
5. Wei, T. et al. (2025). "Beyond Postconditions: Can LLMs Infer
   Formal Contracts?" arXiv:2510.12702.
6. Ding, Y. et al. (2026). "ToolGate: Contract-Grounded Tool
   Execution for LLMs." arXiv:2601.04688.

### Rust Ecosystem

7. Rust Compiler Team (2024). "MCP 759: Contracts and Invariants."
8. tokio-rs (2026). "tracing: Application-level tracing for Rust."
9. Rust Contracts RFC Draft (2025). hackmd.io.
