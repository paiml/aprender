# Eiffel DbC — Escape-Proof Enforcement (Part 1)

Version: 1.0
Status: proposed
Date: 2026-04-09

*See also: [eiffel-dbc.md](eiffel-dbc.md)*

## 11. Escape-Proof Enforcement by Domain

The escape-proof enforcement pipeline
([escape-proof-enforcement.md](escape-proof-enforcement.md)) defines
six stages where each gates the next. This section traces how each
Eiffel DbC obligation type flows through the pipeline, and how each
domain in the stack concretely enforces escape prevention at compile
time.

### 11.1. The Six Stages, Revisited for DbC Types

```
A. Equation YAML        equations.<name>.preconditions / postconditions
   ↓ must exist              ↓ formal predicates in proof_obligations[]
B. Lean 4 Proof          theorem proves the obligation over ℝ
   ↓ no sorry                ↓ precondition becomes hypothesis
C. YAML Validation       pv lint Gate 5: equations have pre/post/lean_theorem
   ↓ gates pass              ↓ Gate 1-4: obligations ↔ harnesses consistent
D. build.rs Codegen      generates CONTRACT_*_PRE_N / CONTRACT_*_POST_N env vars
   ↓ env vars set            ↓ generates debug_assert!() from Rust expressions
E. #[contract] Macro     reads env vars, injects debug_assert!() at entry/exit
   ↓ compile-time check      ↓ #[requires], #[ensures], #[invariant] macros
F. Test Execution        falsification tests + probar + Kani harnesses
   ↓ zero failures           ↓ cargo test blocks merge on any failure
```

For the five NEW DbC types, the pipeline extends as follows:

| DbC Type | Stage A (YAML) | Stage B (Lean) | Stage D (build.rs) | Stage E (Macro) | Stage F (Test) |
|---|---|---|---|---|---|
| `precondition` | `equations.<eq>.preconditions[]` | Hypothesis `h : P x` | `CONTRACT_*_PRE_N` env vars | `debug_assert!(pre)` at entry | Negative test: violate pre, assert panic |
| `postcondition` | `equations.<eq>.postconditions[]` | Goal `post (f x)` | `CONTRACT_*_POST_N` env vars | `let ret = {...}; debug_assert!(post)` | Conditional test: assume pre, assert post |
| `frame` | `proof_obligations[].type: frame` | `∧ ∀i, old[i] = new[i]` | Clone-and-compare codegen | Snapshot inputs pre-call, assert unchanged post-call | Mutation: corrupt input, assert frame test catches |
| `loop_invariant` | `proof_obligations[].type: loop_invariant` | Induction over iterations | Assert-per-iteration codegen | `debug_assert!(inv)` inside unrolled loop | Proptest: verify invariant at each step |
| `loop_variant` | `proof_obligations[].type: loop_variant` | Well-founded recursion | Assert-decreasing codegen | `debug_assert!(v_after < v_before)` | Proptest: verify strictly decreasing |
| `old_state` | `proof_obligations[].type: old_state` | Universally quantified pre-state | Clone-before codegen | `let old = state.clone(); ...; debug_assert!(Q(old, state))` | Snapshot-compare: assert relationship |
| `subcontract` | `proof_obligations[].type: subcontract` | Implication chain | Cross-contract env var check | Parent contract binding must exist | Load both contracts, verify weakening/strengthening |

### 11.2. Chain of Thought: Kernels (aprender, trueno, entrenar)

**Domain axiom:** `softmax(x)_i = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))`

**Stage A — Equation YAML.** The contract author writes:
```yaml
equations:
  softmax:
    formula: "σ(x)_i = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))"
    preconditions:
      - "!logits.is_empty()"
      - "logits.iter().all(|x| x.is_finite())"
    postconditions:
      - "ret.len() == logits.len()"
      - "(ret.iter().sum::<f32>() - 1.0).abs() < 1e-6"
    lean_theorem: "ProvableContracts.Theorems.Softmax.PartitionOfUnity"
```
**Chain of thought:** The preconditions encode the *caller's* responsibility
(non-empty, finite input). The postconditions encode the *kernel's*
guarantee (output length matches, sums to 1). The formula is the
mathematical ground truth. The `lean_theorem` links to the L5 proof.

**Stage B — Lean 4 Proof.** `build.rs` verifies:
1. `PartitionOfUnity` exists in `lean/ProvableContracts/Theorems/Softmax/`
2. No `sorry` in the theorem or its dependencies
3. If either fails → `compile_error!` — the binary cannot be built

**Chain of thought:** This is Meyer's *seamless development* — the math
proof IS the gate. A developer cannot ship softmax without the
theorem being proved. The Lean proof takes the precondition as a
hypothesis: `theorem partition_of_unity (h : ∀ i, x[i].is_finite) : ...`

**Stage C — YAML Validation.** `pv lint` checks:
- Gate 1: YAML parses against schema (preconditions field exists)
- Gate 4: Falsification test IDs resolve to `fn test_*` in source
- Gate 5: Equation has preconditions, postconditions, lean_theorem

**Chain of thought:** If someone deletes a precondition from the YAML,
Gate 5 warns. If someone deletes a falsification test, Gate 4 errors.
The YAML contract is the single source of truth — tampering is visible.

**Stage D — build.rs Codegen.** `build.rs` reads the YAML and sets:
```
CONTRACT_SOFTMAX_KERNEL_V1_SOFTMAX=bound
CONTRACT_SOFTMAX_KERNEL_V1_SOFTMAX_PRE_COUNT=2
CONTRACT_SOFTMAX_KERNEL_V1_SOFTMAX_PRE_0=!logits.is_empty()
CONTRACT_SOFTMAX_KERNEL_V1_SOFTMAX_PRE_1=logits.iter().all(|x| x.is_finite())
CONTRACT_SOFTMAX_KERNEL_V1_SOFTMAX_POST_COUNT=2
CONTRACT_SOFTMAX_KERNEL_V1_SOFTMAX_POST_0=ret.len() == logits.len()
CONTRACT_SOFTMAX_KERNEL_V1_SOFTMAX_POST_1=(ret.iter().sum::<f32>() - 1.0).abs() < 1e-6
```
**Chain of thought:** The env vars are the *bridge* between YAML and
Rust. They carry the precondition/postcondition strings from the
contract into the compiler's environment. If the YAML is deleted,
these vars are missing, and Stage E fails.

**Stage E — #[contract] Macro.** In aprender's softmax implementation:
```rust
#[contract("softmax-kernel-v1", equation = "softmax")]
pub fn softmax_1d(logits: &[f32]) -> Vec<f32> {
    // Macro expands to:
    // 1. const _: Option<&str> = option_env!("CONTRACT_SOFTMAX_KERNEL_V1_SOFTMAX");
    // 2. debug_assert!(!logits.is_empty(), "Pre-condition violated: ...");
    //    debug_assert!(logits.iter().all(|x| x.is_finite()), "...");
    // 3. let ret = { /* original body */ };
    // 4. debug_assert!(ret.len() == logits.len(), "Post-condition violated: ...");
    //    debug_assert!((ret.iter().sum::<f32>() - 1.0).abs() < 1e-6, "...");
    // 5. ret
}
```
**Chain of thought:** The macro reads the env vars at *compile time*
and generates `debug_assert!` calls. In debug builds, every call to
`softmax_1d` checks the preconditions on entry and postconditions on
exit. In release builds, `debug_assert!` expands to nothing — zero
runtime cost. If someone removes the `#[contract]` attribute, `pv lint`
Gate 4 catches the missing test reference. If someone deletes the
YAML, `option_env!` returns `None` and the binding const fails.

**Stage F — Test Execution.** `cargo test` runs:
- FALSIFY-SM-001: proptest with 10000 random vectors, verify sum ≈ 1.0
- FALSIFY-SM-004: proptest comparing scalar vs AVX2 output within 8 ULP
- Kani harnesses: `verify_softmax_normalization` (bounded ≤16 elements)

**Chain of thought:** The falsification tests are the Popperian gate —
they try to *refute* the postcondition. If any test finds an input
where sum ≠ 1.0, the contract is violated and `cargo test` fails.
CI blocks the merge. The kernel cannot ship with a broken postcondition.

**Escape analysis for kernels:**

| Attempted Escape | Gate That Catches It |
|---|---|
| Delete precondition from YAML | Gate 5: equation without precondition → WARNING |
| Delete `#[contract]` from function | Gate 4: test reference dangling → ERROR |
| Delete YAML file entirely | build.rs: `CONTRACT_*` env var missing → compile error |
| Leave `sorry` in Lean proof | build.rs: sorry detection → compile error |
| Weaken postcondition | Lean proof no longer matches → sorry required → compile error |
| Skip `cargo test` | CI workflow requires test pass |

### 11.3. Chain of Thought: Simulation (simular)

**Domain axiom:** `E(t+dt) = E(t) + O(dt^4)` (symplectic energy conservation)

**Stage A — Equation YAML.** Simular's checkpoint contract:
```yaml
equations:
  checkpoint_roundtrip:
    formula: "deserialize(serialize(state)) = state"
    preconditions:
      - "state.energy.is_finite()"
      - "state.particles.iter().all(|p| p.position.is_finite())"
    postconditions:
      - "(restored.energy - state.energy).abs() < f64::EPSILON"
    lean_theorem: "ProvableContracts.Theorems.Checkpoint.Roundtrip"
```
**Chain of thought:** Simulation preconditions constrain *physical
plausibility* (finite energy, finite positions). The postcondition
asserts *checkpoint fidelity* — serialization must not lose precision.
This is where `old_state` becomes essential: the postcondition
references `state` (the pre-call value) and `restored` (the post-call
value). Without `old_state` as a first-class concept, this relationship
is awkward to express.

**NEW: Frame obligation for integration step:**
```yaml
proof_obligations:
  - type: frame
    property: "Integration step modifies positions and velocities only"
    formal: "modifies(pos, vel) ∧ preserves(mass, G, dt)"
```
**Stage D codegen** for frame obligations generates:
```rust
// build.rs generates:
let old_mass = state.mass.clone();
let old_g = state.gravitational_constant;
// ... run integration step ...
debug_assert!(state.mass == old_mass, "Frame violated: mass changed");
debug_assert!(state.gravitational_constant == old_g, "Frame violated: G changed");
```
**Chain of thought:** The frame obligation is *critical* for simular
because the jidoka module (runtime anomaly detection) already catches
NaN and energy drift — but it cannot catch *silent parameter
corruption* where a bug modifies the gravitational constant during
integration. The frame obligation makes this a compile-time-enforced
invariant: if the integration step touches `G`, `debug_assert!` fires
in debug builds, and Kani can verify it statically.

**NEW: Loop invariant for N-body simulation:**
```yaml
proof_obligations:
  - type: loop_invariant
    property: "Total momentum conserved at every timestep"
    formal: "∀k ≤ i: |Σ m_j·v_j(k) - Σ m_j·v_j(0)| < ε_drift × k"
    applies_to: "nbody.integrate"
```
**Stage D codegen** generates an assertion inside the simulation loop:
```rust
for step in 0..n_steps {
    integrate_step(&mut state);
    debug_assert!(
        (total_momentum(&state) - initial_momentum).norm() < DRIFT_TOL * step as f64,
        "Loop invariant violated: momentum drift at step {step}"
    );
}
```
**Chain of thought:** This replaces simular's runtime jidoka check with
a *declarative contract*. The jidoka module catches drift at runtime
and halts — the loop invariant obligation lets Kani verify *before
deployment* that drift is bounded for all inputs within the natural
bound (max particles × max timesteps).

### 11.4. Chain of Thought: Infrastructure as Code (forjar)

**Domain axiom:** `converge(desired, current) → desired` (idempotent convergence)

**Stage A — Equation YAML.** Forjar's DAG ordering contract:
```yaml
equations:
  topological_sort:
    formula: "∀ (a,b) ∈ E: order(a) < order(b)"
    preconditions:
      - "!graph.has_cycle()"
      - "graph.node_count() > 0"
    postconditions:
      - "result.len() == graph.node_count()"
      - "result.windows(2).all(|w| !graph.has_edge(w[1], w[0]))"
    lean_theorem: "ProvableContracts.Theorems.DAG.TopologicalSort"
```
**Chain of thought:** IaC preconditions enforce *infrastructure
validity* — the resource dependency graph must be acyclic. This is
where precondition-as-obligation-type adds value: the *formal
predicate* `¬∃ cycle in G` can be verified by Kani for all graphs
up to N nodes, while the Rust expression `!graph.has_cycle()` is
a runtime check only.

**NEW: Frame obligation for resource apply:**
```yaml
proof_obligations:
  - type: frame
    property: "Applying resource R modifies only R's lock entry"
    formal: "modifies(lock[R]) ∧ preserves(lock[S] ∀ S ≠ R)"
```
**Stage D codegen** for forjar's apply:
```rust
// build.rs generates for each resource apply:
let old_lock = lock_state.clone();
apply_resource(&mut lock_state, resource_r);
for (name, entry) in &old_lock.entries {
    if name != resource_r.name() {
        debug_assert!(
            lock_state.entries[name] == *entry,
            "Frame violated: resource {} modified by apply of {}",
            name, resource_r.name()
        );
    }
}
```
**Chain of thought:** This is the *most important* DbC type for IaC.
Forjar's jidoka policy (stop on first failure, preserve partial state)
is a runtime enforcement of the frame condition. But without a frame
*contract*, a subtle bug where applying package `nginx` accidentally
modifies the lock entry for service `postgres` would go undetected
until production. The frame obligation makes this a compile-time-
verifiable property: Kani can exhaustively verify that `apply_resource`
only touches the target entry for all resource configurations within
the natural bound.

**NEW: Old-state for drift detection:**
```yaml
proof_obligations:
  - type: old_state
    property: "Drift = hash mismatch between lock and current state"
    formal: "drift(R) ↔ hash(current(R)) ≠ hash(old(lock(R)))"
```
**Stage E macro** on forjar's drift detection:
```rust
#[contract("drift-detection-v1", equation = "drift_hash")]
pub fn detect_drift(lock: &LockState, current: &SystemState) -> Vec<DriftResult> {
    // Macro injects:
    // PRE: debug_assert!(lock.entries.iter().all(|e| e.hash.len() == 32));
    // POST: debug_assert!(ret.iter().all(|d| d.resource_exists_in_lock));
}
```
**Chain of thought:** Forjar's tripwire module already computes
`hash(current) vs hash(lock)` — but the *contract* for what
constitutes drift is implicit in code. Making it an `old_state`
obligation means `pv lint` validates the contract exists, `build.rs`
generates assertions, and Kani can verify that the hash comparison
is correct for all states within bounds. The `old_state` type is
the formal language for "relating pre-state to post-state" — which
is exactly what drift detection does.

**NEW: Subcontract for transport substitutability:**
```yaml
proof_obligations:
  - type: subcontract
    property: "Pepita transport refines SSH transport"
    formal: "pre(SSH) → pre(Pepita) ∧ post(Pepita) → post(SSH)"
    parent_contract: "ssh-transport-v1"
```
**Stage F test** — cross-contract verification:
```rust
#[test]
fn pepita_refines_ssh() {
    // Load both contracts
    let ssh = parse_contract("ssh-transport-v1.yaml");
    let pepita = parse_contract("pepita-transport-v1.yaml");
    // Verify: every script accepted by SSH is also accepted by Pepita
    proptest!(|(script in valid_ssh_scripts())| {
        assert!(pepita.accepts(&script), "Pepita rejects script SSH accepts");
    });
    // Verify: Pepita's postconditions imply SSH's
    proptest!(|(script in valid_ssh_scripts(), state in any_state())| {
        let pepita_result = pepita.execute(script, state);
        assert!(ssh.postcondition_holds(&pepita_result),
            "Pepita result violates SSH postcondition");
    });
}
```
**Chain of thought:** This is Meyer's subcontracting rule made
operational. If a future forjar developer adds a new pepita check
that rejects a script SSH would accept (strengthening the precondition),
the cross-contract test fails immediately. The `subcontract` obligation
type makes the Liskov relationship *machine-checkable* — not just
documented in a comment.


*Continued in [eiffel-dbc-enforcement-2.md](eiffel-dbc-enforcement-2.md)*
