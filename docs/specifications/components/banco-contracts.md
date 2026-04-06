# Banco Provable Contracts & Popperian Falsification

> Parent: [banco-spec.md](banco-spec.md) §5
> Depends on: provable-contracts (0.1.x), provable-contracts-macros (0.1.x)

---

## Purpose

Every critical Banco invariant is formalized as a provable contract — a YAML file that generates Rust traits, Kani bounded model checking harnesses, probar property tests, and falsification tests. This is not aspirational: the Sovereign AI Stack already has 165 contracts governing 900K lines of Rust. Banco adds contracts for its own domain: privacy enforcement, budget conservation, routing correctness, and inference pipeline integrity.

---

## Falsification Claims on banco-spec.md

Popperian falsification: for each claim in the spec, we state what would disprove it, and write a test that attempts to.

### Claim F-BANCO-001: Privacy tier enforcement

**Spec says:** "Sovereign mode blocks ALL external API calls."

**Falsification:** Find any code path where a request in Sovereign mode reaches a remote backend.

```yaml
# contracts/banco/privacy-enforcement-v1.yaml
metadata:
  version: "1.0.0"
  description: "Privacy tier enforcement — Poka-Yoke gate"

equations:
  sovereign_gate:
    formula: "∀ req where tier = Sovereign: backend(req) ∈ LocalBackends"
    domain: "req ∈ ChatRequest, tier ∈ {Sovereign, Private, Standard}"
    codomain: "backend ∈ ServingBackend"
    invariants:
      - "Sovereign never routes to remote"
      - "Private allows only enterprise remotes"
      - "Standard allows all backends"

proof_obligations:
  - type: soundness
    property: "Sovereign rejects all remote backends"
    formal: "∀ b ∈ RemoteBackends: ¬PrivacyTier::Sovereign.allows(b)"
    applies_to: all
  - type: completeness
    property: "Standard allows all backends"
    formal: "∀ b ∈ ServingBackend: PrivacyTier::Standard.allows(b)"
    applies_to: all
  - type: invariant
    property: "Middleware injects correct header"
    formal: "∀ resp: resp.headers['x-privacy-tier'] = tier_string(state.privacy_tier)"
    applies_to: all

falsification_tests:
  - id: FALSIFY-PRIV-001
    rule: "Sovereign blocks remote"
    prediction: "BackendSelector with Sovereign returns only local backends"
    test: "Enumerate all ServingBackend variants, assert Sovereign.allows() = false for remotes"
    if_fails: "New remote backend added without privacy gate update"
  - id: FALSIFY-PRIV-002
    rule: "Middleware rejects external hint in Sovereign"
    prediction: "X-Banco-Backend: openai → 403 in Sovereign"
    test: "Send request with external backend header to Sovereign router"
    if_fails: "Middleware bypass or missing check"

kani_harnesses:
  - id: KANI-PRIV-001
    obligation: "Sovereign rejects all remote"
    bound: 24
    strategy: exhaustive
    harness: "verify_sovereign_blocks_all_remote_backends"
```

### Claim F-BANCO-002: Circuit breaker budget conservation

**Spec says:** "CostCircuitBreaker prevents overspending with daily budgets."

**Falsification:** Find a sequence of requests that spends more than the daily budget.

```yaml
# contracts/banco/budget-conservation-v1.yaml
metadata:
  version: "1.0.0"
  description: "Cost circuit breaker budget conservation law"

equations:
  budget_check:
    formula: "check(cost) = Ok iff accumulated + cost ≤ budget"
    invariants:
      - "accumulated monotonically increases within a day"
      - "accumulated resets on new day"
      - "circuit opens when budget exceeded"

proof_obligations:
  - type: conservation
    property: "Total spend never exceeds budget + one max request"
    formal: "accumulated ≤ budget + max_request_cost"
    tolerance: 0.01
    applies_to: all
  - type: monotonicity
    property: "Accumulated cost only increases"
    formal: "record(x) → accumulated' ≥ accumulated"
    applies_to: all
  - type: invariant
    property: "Open state blocks requests"
    formal: "state = Open ∧ ¬cooldown_elapsed → check() = Err"
    applies_to: all

falsification_tests:
  - id: FALSIFY-BUDGET-001
    rule: "Budget conservation"
    prediction: "After N records totaling > budget, check() returns Err"
    test: "proptest: random cost sequence summing to 2× budget"
    if_fails: "Millicent arithmetic overflow or race condition"
  - id: FALSIFY-BUDGET-002
    rule: "Daily reset"
    prediction: "After date change, accumulated = 0"
    test: "Mock clock to next day, verify reset"
    if_fails: "Date comparison bug"

kani_harnesses:
  - id: KANI-BUDGET-001
    obligation: "Conservation"
    bound: 8
    strategy: bounded_int
    harness: "verify_budget_never_exceeded"
```

### Claim F-BANCO-003: Routing determinism

**Spec says:** "SpilloverRouter routes locally when queue depth < threshold."

**Falsification:** Find a state where queue depth < threshold but routing decision is not Local.

```yaml
# contracts/banco/routing-determinism-v1.yaml
metadata:
  version: "1.0.0"
  description: "Spillover router determinism and correctness"

equations:
  route:
    formula: "route() = Local(primary) if depth < threshold, else Spillover(best_available)"
    invariants:
      - "Empty queue always routes local"
      - "Full queue with spillover disabled rejects"
      - "Sovereign spillover targets are all local"

proof_obligations:
  - type: determinism
    property: "Same state → same decision"
    formal: "route(state) = route(state) (no hidden randomness)"
    applies_to: all
  - type: completeness
    property: "All queue depths produce a decision"
    formal: "∀ depth ∈ [0, max]: route() ∈ {Local, Spillover, Reject}"
    applies_to: all
  - type: soundness
    property: "Sovereign spillover stays local"
    formal: "tier = Sovereign → Spillover(b) → b.is_local()"
    applies_to: all

falsification_tests:
  - id: FALSIFY-ROUTE-001
    rule: "Empty queue routes local"
    prediction: "route() = Local(Realizar) when depth = 0"
    test: "Fresh router, immediate route()"
    if_fails: "Default queue depth not zero"
  - id: FALSIFY-ROUTE-002
    rule: "Sovereign spillover is local"
    prediction: "All spillover backends are local when Sovereign"
    test: "Fill queue past threshold with Sovereign config"
    if_fails: "Sovereign config includes remote spillover backends"

kani_harnesses:
  - id: KANI-ROUTE-001
    obligation: "Determinism"
    bound: 4
    strategy: exhaustive
    harness: "verify_routing_deterministic"
```

### Claim F-BANCO-004: Context window enforcement

**Spec says:** "ContextManager prevents prompts from exceeding model context limits."

```yaml
# contracts/banco/context-enforcement-v1.yaml
metadata:
  version: "1.0.0"
  description: "Context window enforcement — never exceed model limits"

equations:
  fits:
    formula: "fits(msgs) = estimate_tokens(msgs) ≤ window.available_input()"
    invariants:
      - "available_input = max_tokens - output_reserve"
      - "estimate_tokens monotonically increases with message count"

proof_obligations:
  - type: bound
    property: "Truncated output fits"
    formal: "estimate_tokens(truncate(msgs)) ≤ available_input"
    applies_to: all
  - type: monotonicity
    property: "More messages → more tokens"
    formal: "msgs₁ ⊂ msgs₂ → estimate(msgs₁) ≤ estimate(msgs₂)"
    applies_to: all
  - type: invariant
    property: "Error strategy refuses truncation"
    formal: "strategy = Error ∧ ¬fits(msgs) → truncate(msgs) = Err"
    applies_to: all

falsification_tests:
  - id: FALSIFY-CTX-001
    rule: "Truncated always fits"
    prediction: "truncate(msgs).map(|m| fits(m)) = Ok(true)"
    test: "proptest: random messages exceeding window"
    if_fails: "Truncation underestimates token count"

kani_harnesses:
  - id: KANI-CTX-001
    obligation: "Truncated fits"
    bound: 16
    strategy: bounded_int
    harness: "verify_truncated_messages_fit_context"
```

### Claim F-BANCO-005: Chat template idempotency

**Spec says:** "ChatTemplateEngine formats prompts according to model template."

```yaml
# contracts/banco/template-correctness-v1.yaml
metadata:
  version: "1.0.0"
  description: "Chat template correctness — format preserves content"

equations:
  apply:
    formula: "apply(msgs) = format-specific wrapping of each msg.content"
    invariants:
      - "All user content appears in output"
      - "Raw format is identity"

proof_obligations:
  - type: invariant
    property: "User content preserved"
    formal: "∀ msg ∈ msgs: apply(msgs).contains(msg.content)"
    applies_to: all
  - type: idempotency
    property: "Raw format is identity"
    formal: "apply_raw([msg]) = msg.content"
    applies_to: [raw]
  - type: completeness
    property: "All formats produce non-empty output"
    formal: "∀ fmt, msgs (non-empty): apply(msgs).len() > 0"
    applies_to: all

falsification_tests:
  - id: FALSIFY-TPL-001
    rule: "Content preservation"
    prediction: "Every message content substring appears in formatted output"
    test: "proptest: random messages through each format"
    if_fails: "Template drops or corrupts content"

kani_harnesses:
  - id: KANI-TPL-001
    obligation: "Raw identity"
    bound: 4
    strategy: exhaustive
    harness: "verify_raw_format_is_identity"
```

---

## Phase 2+ Contract Targets

| Contract | Domain | Key Obligations |
|----------|--------|----------------|
| `inference-pipeline-v1` | Chat completions | Tokenize→forward→decode roundtrip preserves semantics |
| `model-hotswap-v1` | Model management | Arc refcount prevents use-after-free; no in-flight request dropped |
| `arena-fairness-v1` | Arena comparison | Both models receive identical input; latency measured consistently |
| `speculative-equivalence-v1` | Speculative decoding | Speculative output ≡ standard output (mathematically) |
| `prefix-cache-soundness-v1` | Prefix caching | Cached KV = recomputed KV (no stale cache corruption) |
| `structured-output-v1` | JSON grammar | Output always valid JSON matching schema |
| `sse-completeness-v1` | Streaming | Every SSE stream ends with finish_reason + [DONE] |
| `batch-ordering-v1` | Continuous batching | Request order preserved within priority class |

## Phase 3+ Contract Targets

| Contract | Domain | Key Obligations |
|----------|--------|----------------|
| `training-loss-monotonicity-v1` | Training | Loss decreases over sufficient steps (statistical) |
| `lora-rank-bound-v1` | LoRA | Adapter rank ≤ min(in, out) dimensions |
| `export-roundtrip-v1` | Export | load(export(model)) ≈ model (within quant tolerance) |
| `merge-conservation-v1` | Model merge | Weighted merge: Σwᵢ = 1.0 and result = Σwᵢ·modelᵢ |
| `rag-recall-v1` | Built-in RAG | Indexed document always retrievable (no silent drops) |
| `recipe-determinism-v1` | Data recipes | Same input + seed → same output dataset |
| `eval-ppl-bound-v1` | Evaluation | PPL(uniform) = vocab_size (sanity calibration) |

## Phase 4+ Contract Targets

| Contract | Domain | Key Obligations |
|----------|--------|----------------|
| `sandbox-isolation-v1` | Code execution | No filesystem access outside sandbox; timeout enforced |
| `sovereign-no-egress-v1` | Web search | Sovereign tier: zero outbound network connections |
| `agent-termination-v1` | Agent runtime | Agent loop terminates within max_steps |
| `auth-jwt-soundness-v1` | Authentication | Valid token accepted; expired/forged rejected |

---

## Integration: How Contracts Flow Into Banco

### 1. Contract YAML lives in repo

```
contracts/banco/
  privacy-enforcement-v1.yaml
  budget-conservation-v1.yaml
  routing-determinism-v1.yaml
  context-enforcement-v1.yaml
  template-correctness-v1.yaml
```

### 2. `pv generate` produces test artifacts

```bash
pv generate contracts/banco/privacy-enforcement-v1.yaml -o src/serve/banco/contracts/
# Produces:
#   privacy_enforcement_scaffold.rs  (trait)
#   privacy_enforcement_kani.rs      (Kani harnesses)
#   privacy_enforcement_probar.rs    (property tests)
```

### 3. Binding registry maps contracts to implementations

```yaml
# contracts/banco/binding.yaml
bindings:
  - contract: privacy-enforcement-v1.yaml
    equation: sovereign_gate
    module_path: "batuta::serve::backends::PrivacyTier"
    function: "allows"
    signature: "fn allows(&self, backend: ServingBackend) -> bool"
    status: implemented

  - contract: budget-conservation-v1.yaml
    equation: budget_check
    module_path: "batuta::serve::circuit_breaker::CostCircuitBreaker"
    function: "check"
    signature: "fn check(&self, estimated_cost_usd: f64) -> Result<(), CircuitBreakerError>"
    status: implemented

  - contract: routing-determinism-v1.yaml
    equation: route
    module_path: "batuta::serve::router::SpilloverRouter"
    function: "route"
    signature: "fn route(&self) -> RoutingDecision"
    status: implemented

  - contract: context-enforcement-v1.yaml
    equation: fits
    module_path: "batuta::serve::context::ContextManager"
    function: "fits"
    signature: "fn fits(&self, messages: &[ChatMessage]) -> bool"
    status: implemented

  - contract: template-correctness-v1.yaml
    equation: apply
    module_path: "batuta::serve::templates::ChatTemplateEngine"
    function: "apply"
    signature: "fn apply(&self, messages: &[ChatMessage]) -> String"
    status: implemented
```

### 4. CI runs verification

```bash
# Validate contracts
pv lint contracts/banco/ --min-score 0.75

# Audit bindings
pv audit contracts/banco/privacy-enforcement-v1.yaml \
  --binding contracts/banco/binding.yaml

# Run generated property tests
cargo test --features banco banco_contracts

# Run Kani (nightly)
cargo kani --features banco --harness verify_sovereign_blocks_all_remote_backends
```

### 5. Coverage tracked

```bash
pv coverage contracts/banco/ --binding contracts/banco/binding.yaml

# Output:
# 5 contracts, 14 obligations, 14 falsification tests, 5 Kani harnesses
# Binding coverage: 5/5 (100%)
# Obligation coverage: 14/14 (100%)
```

---

## Verification Ladder for Banco

| Level | What | Tool | Banco Status |
|-------|------|------|-------------|
| L0 | Code review | Human | Always |
| L1 | Type system | rustc | Always (privacy tier is enum, not string) |
| L2 | Falsification tests | `#[test]` | Phase 1 done (32 tests) |
| L3 | Property tests | probar + proptest | Phase 2 (generated from contracts) |
| L4 | Bounded model check | Kani | Phase 2 (5 harnesses, expand per phase) |
| L5 | Theorem proving | Lean 4 | Future (privacy + budget proofs) |

---

## Feature Flag

```toml
# Cargo.toml (already present)
provable-contracts = { version = "0.1", optional = true }
provable-contracts-macros = { version = "0.1", optional = true }

# Feature (already present)
agents-contracts = ["agents", "provable-contracts", "provable-contracts-macros"]
```

Contract-generated tests compile without the feature flag — they test public API behavior, not contract internals. Kani harnesses require nightly + kani toolchain.
