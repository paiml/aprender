# Sub-spec: Contract-Aware Tracing (Tier 3 Runtime Integration)

**Parent:** [trueno-spec.md](../trueno-spec.md) Section 23

**References:**
- [ProofWright](https://arxiv.org/abs/2511.12294) — Agentic formal verification of CUDA kernels
- [Volta](https://arxiv.org/abs/2511.12638) — Equivalence checking of ML GPU kernels
- [NCCLbpf](https://arxiv.org/abs/2603.11438) — Verified eBPF policy hooks for GPU collectives (80-130ns overhead)
- [Rust MCP-759](https://github.com/rust-lang/compiler-team/issues/759) — Native `#[contracts::requires/ensures]`
- [provable-contracts deep-integration](../../../provable-contracts/docs/specifications/sub/deep-integration.md) — Gap analysis

---

## 1. Three-Tier Enforcement Model

```
Tier 1: Compile-Time (existing, enforced)
  YAML → build.rs → CONTRACT_* env vars → #[contract] debug_assert
  YAML → pv scaffold --trait → trait → impl → compiler check

Tier 2: CI-Time (existing, enforced)
  pv lint → 7 gates → SARIF
  pmat comply → CB-1200..1209
  cargo test --test contract_traits → trait tests

Tier 3: Runtime (THIS SPEC)
  ContractRegistry → load YAML at startup
  ContractTracingLayer → verify postconditions on span close
  BrickProfiler → budget from contract YAML
  ModelTracer → check MLT-01..05 against invariants
```

## 2. ContractRegistry

Loads contract YAML at startup into an in-memory registry.
Keyed by `(contract_stem, equation_name)`.

```rust
pub struct ContractRegistry {
    contracts: HashMap<String, ParsedContract>,
}

impl ContractRegistry {
    /// Load all contracts from a directory.
    /// Graceful: missing dir → empty registry (CI/crates.io).
    pub fn load(dir: &Path) -> Self;

    /// Get postconditions for an equation.
    pub fn postconditions(&self, contract: &str, eq: &str)
        -> Option<&[Postcondition]>;

    /// Derive TokenBudget from roofline for detected hardware.
    pub fn derive_budget(&self, hw: &HardwareCapability)
        -> Option<TokenBudget>;
}
```

**Startup cost:** Parse 28 YAML files (~50ms). Cached for process lifetime.

## 3. ContractTracingLayer

A `tracing::Layer` that intercepts spans tagged with contract metadata
and verifies postconditions on span close.

```rust
pub struct ContractTracingLayer {
    registry: Arc<ContractRegistry>,
}

impl<S: Subscriber> Layer<S> for ContractTracingLayer {
    fn on_close(&self, id: span::Id, ctx: Context<'_, S>) {
        let span = ctx.span(&id).unwrap();
        if let Some(cid) = span.extensions().get::<ContractId>() {
            if let Some(posts) = self.registry
                .postconditions(&cid.contract, &cid.equation)
            {
                for post in posts {
                    if !post.check(span.fields()) {
                        tracing::error!(
                            contract = %cid.contract,
                            equation = %cid.equation,
                            obligation = %post.property,
                            "TIER3: contract postcondition violated"
                        );
                    }
                }
            }
        }
    }
}
```

**Usage at call sites:**

```rust
let _span = tracing::info_span!(
    "softmax",
    contract.id = "softmax-kernel-v1",
    contract.equation = "softmax",
).entered();
// ... kernel executes ...
// On span drop → ContractTracingLayer checks postconditions
```

**Performance:** ≤130ns per check. NCCLbpf (arXiv:2603.11438)
demonstrates this overhead for eBPF policy hooks in GPU collective
data paths. trueno's deferred sync mode batches checks with existing
GPU synchronization — zero additional per-kernel overhead on hot path.

## 4. BrickProfiler Contract Integration (Gap 2)

**Current:** `ComputeBrick` budget thresholds are set manually.

**Fix:** Derive from contract YAML at construction.

```rust
impl ComputeBrick {
    /// Construct a brick with budget derived from contract YAML.
    pub fn from_contract(
        registry: &ContractRegistry,
        hw: &HardwareCapability,
    ) -> Self {
        let budget = registry.derive_budget(hw)
            .unwrap_or_else(TokenBudget::default);
        Self {
            budget,
            enforce_budget: true,
            contract_id: Some("roofline-model-v1".into()),
            ..Default::default()
        }
    }
}
```

Budget violations go through `ContractTracingLayer` diagnostic path.

## 5. ModelTracer Contract Hooks (Gap 3)

**Current:** `end_forward()` runs anomaly detection with hardcoded
thresholds.

**Fix:** Check existing MLT-01..05 trace data against contract
postconditions. Zero additional collection overhead — reuses traces.

| Trace | Contract | Postcondition |
|-------|----------|---------------|
| MLT-01 (activations) | activation-kernel-v1 | No NaN/Inf for finite inputs |
| MLT-02 (attention) | softmax-kernel-v1 | Rows sum to 1.0 ± 1e-6 |
| MLT-03 (logits) | model-config-algebra-v1 | Magnitude within bounds |
| MLT-04 (quantization) | quantization-ordering-v1 | Error ≤ threshold |
| MLT-05 (KV cache) | gpu-decode-profiling-v1 | Utilization ≤ capacity |

```rust
impl ModelTracer {
    pub fn end_forward_with_contracts(
        &mut self,
        registry: &ContractRegistry,
    ) {
        // Existing anomaly detection
        self.end_forward();

        // Contract postcondition checks
        if let Some(act) = &self.activation_trace {
            if act.has_nan() {
                if let Some(posts) = registry.postconditions(
                    "activation-kernel-v1", "relu"
                ) {
                    // posts say "no NaN for finite inputs"
                    tracing::error!(
                        contract = "activation-kernel-v1",
                        trace = "MLT-01",
                        "NaN detected in activations"
                    );
                }
            }
        }
        // ... similar for MLT-02..05
    }
}
```

## 6. Rust-Native Path (Future)

When [Rust MCP-759](https://github.com/rust-lang/compiler-team/issues/759)
stabilizes, YAML postcondition checks can migrate to compiler-inserted
`#[contracts::ensures]` assertions. This gives:

- Zero cost in release builds (`-Z contract-checks=off`)
- Compiler-verified postcondition signatures
- Standard Rust tooling (clippy, rustfmt, IDE support)

```rust
// Future: replaces YAML postcondition + ContractTracingLayer
#[contracts::ensures(ret.len() == logits.len())]
#[contracts::ensures(ret.iter().all(|&v| v >= 0.0 && v <= 1.0))]
#[contracts::ensures((ret.iter().sum::<f32>() - 1.0).abs() < 1e-6)]
pub fn softmax_1d(logits: &[f32]) -> Vec<f32> { ... }
```

## 7. FALSIFY Tests for Tier 3

```yaml
falsification_tests:
  - id: FALSIFY-T3-001
    rule: "ContractTracingLayer detects NaN postcondition violation"
    test: "test_tracing_layer_detects_nan_violation"
    prediction: "verified"
    if_fails: "Tracing layer misses NaN — contract not enforced"

  - id: FALSIFY-T3-002
    rule: "BrickProfiler budget derived from roofline YAML"
    test: "test_brick_budget_from_contract"
    prediction: "verified"
    if_fails: "Budget still hardcoded — contract disconnected"

  - id: FALSIFY-T3-003
    rule: "ModelTracer end_forward checks softmax sum-to-1"
    test: "test_model_tracer_softmax_postcondition"
    prediction: "verified"
    if_fails: "Softmax violation undetected at runtime"

  - id: FALSIFY-T3-004
    rule: "ContractTracingLayer overhead ≤ 200ns per check"
    test: "bench_tracing_layer_overhead"
    prediction: "verified"
    if_fails: "Runtime enforcement too expensive for hot path"
```
