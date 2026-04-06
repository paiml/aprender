# architecture-requirements-v1

**Version:** 1.0.0

Per-architecture tensor weight requirements — source of truth for required/optional roles

## References

- UCBD Spec v1.0.0 Section 7.3 — Architecture Requirements (GH-279)
- realizar/src/arch_requirements.rs — Rust implementation (generated from this contract)
- realizar/src/gguf/config.rs — ArchConstraints::from_architecture()
- Vaswani et al. (2017) Attention Is All You Need
- Touvron et al. (2023) Llama 2: Open Foundation and Fine-Tuned Chat Models
- Yang et al. (2024) Qwen2 Technical Report
- Qwen Team (2025) Qwen3 Technical Report — QK norm
- Jiang et al. (2023) Mistral 7B
- Abdin et al. (2024) Phi-3 Technical Report
- Gemma Team (2024) Gemma: Open Models Based on Gemini Research
- Radford et al. (2023) Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)

## Equations

### constraint_matrix_exhaustiveness

```
∀ (qk: bool, bias: bool): ∃! cell ∈ constraint_matrix such that
  cell.has_qk_norm = qk ∧ cell.has_bias = bias

```

**Domain:** `(has_qk_norm, has_bias) ∈ {true, false}^2`

**Codomain:** $Exactly one constraint cell$

**Invariants:**

- $Four cells cover all four (bool, bool) combinations$
- `No two cells share the same (has_qk_norm, has_bias) pair`
- $Adding a new boolean axis requires 2^(n+1) cells$

### role_mapping

$$
map(role) = field_name in IndexedLayerWeights; \forall role \in required(arch): map(role).ptr \neq 0 ∧ map(role).len > 0

$$

**Domain:** $role \in WeightRole enum, arch \in supported architectures$

**Codomain:** $(ptr: usize, len: usize) — pointer and byte length in mapped model$

**Invariants:**

- $map is injective (no two roles share a field)$
- $map is total on WeightRole (every role has a field name)$
- $field_name matches IndexedLayerWeights struct field exactly$

### weight_completeness

```
required(arch) = base_roles ∪ (qk_norm_roles if has_qk_norm) ∪ (bias_roles if has_bias); complete(model, arch) = ∀ role ∈ required(arch): role.ptr ≠ 0 ∧ role.len > 0

```

**Domain:** $arch \in {llama, qwen2, qwen3, phi, mistral, gemma, whisper, ...}$

**Codomain:** $complete \in {true, false}$

**Invariants:**

- $base_roles ⊆ required(arch) for all arch (base is always required)$
- $|required(arch)| \in {9, 11, 12, 14} (only four possible cardinalities)$
- $complete(model, arch) = true => model produces correct output$
- $complete(model, arch) = false => model MUST NOT run (Jidoka stop)$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | Base roles always required | $\forall arch: base_roles ⊆ required_roles(arch)$ |
| 2 | invariant | Constraint matrix exhaustive | $\forall (qk, bias) \in {true,false}^2: \exists! cell matching (qk, bias)$ |
| 3 | invariant | Role count correctness | $\|base\| = 9 ∧ \|base ∪ qk\| = 11 ∧ \|base ∪ bias\| = 12 ∧ \|base ∪ qk ∪ bias\| = 14$ |
| 4 | completeness | Weight completeness implies correct forward pass | $complete(model, arch) = true => forward(model) produces non-garbage output$ |
| 5 | soundness | Incomplete weights detected before forward pass | $\exists role \in required(arch): role.len = 0 => error raised before any computation$ |
| 6 | equivalence | YAML matches Rust implementation | `∀ arch: yaml.required(arch) = rust.required_roles(ArchConstraints::from_architecture(arch))` |
| 7 | monotonicity | Adding features only adds roles | `required(arch_with_feature) ⊇ required(arch_without_feature)` |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-ARCH-001 | YAML-Rust parity | For every architecture in this YAML, the set of required roles matches the output of required_roles(ArchConstraints::from_architecture(name)) in realizar/src/arch_requirements.rs exactly.
 | YAML and Rust implementation have diverged. One was updated without the other. Fix BOTH — this YAML is source of truth.
 |
| FALSIFY-ARCH-002 | Base roles always present | For every architecture (including unknown/default), required_roles() contains all 9 base roles.
 | An architecture override accidentally removed a base role.
 |
| FALSIFY-ARCH-003 | Constraint matrix exhaustiveness | The four (has_qk_norm, has_bias) combinations {(F,F), (T,F), (F,T), (T,T)} each produce a distinct, non-empty role set.
 | Match arms in required_roles() overlap or a cell is missing.
 |
| FALSIFY-ARCH-004 | Role count invariants | \|base\| = 9, \|base + qk_norm\| = 11, \|base + bias\| = 12, \|base + qk_norm + bias\| = 14.
 | A role was added to or removed from a const slice without updating counts.
 |
| FALSIFY-ARCH-005 | Qwen3 requires QK norm, not bias | required_roles(qwen3) contains attn_q_norm and attn_k_norm, but NOT attn_q_bias, attn_k_bias, attn_v_bias.
 | Qwen3 constraint flags are wrong — this was the root cause of GH-279 (GPU garbage from missing QK norm weights).
 |
| FALSIFY-ARCH-006 | Qwen2 requires bias, not QK norm | required_roles(qwen2) contains attn_q_bias, attn_k_bias, attn_v_bias, but NOT attn_q_norm, attn_k_norm.
 | Qwen2/Qwen3 constraint flags swapped.
 |
| FALSIFY-ARCH-007 | LLaMA/Mistral require base only | required_roles(llama) and required_roles(mistral) each contain exactly 9 roles (base only). No bias, no QK norm.
 | LLaMA or Mistral incorrectly flagged has_bias or has_qk_norm.
 |
| FALSIFY-ARCH-008 | Incomplete model rejected before forward pass | A model missing any single required role for its architecture triggers an error BEFORE any matrix multiplication or attention computation.
 | The completeness gate has a gap — a missing weight silently becomes zero, producing garbage output (the exact GH-279 failure mode).
 |
| FALSIFY-ARCH-009 | Optional roles do not trigger rejection | A model that has all required roles but is missing optional roles passes the completeness check without error.
 | Optional roles incorrectly classified as required.
 |
| FALSIFY-ARCH-010 | Unknown architecture falls back to base | An unrecognized architecture string (e.g., "future_arch_2027") produces ArchConstraints with has_qk_norm=false, has_bias=false, and required_roles returns exactly the 9 base roles.
 | Default match arm in from_architecture or required_roles is wrong.
 |
| FALSIFY-ARCH-011 | Alias equivalence | Architecture aliases produce identical ArchConstraints and identical required role sets. E.g., "llama" == "llama3", "qwen2" == "qwen2.5" == "qwen", "phi" == "phi3", "gemma" == "gemma2".
 | An alias was added to from_architecture() but maps to different constraints.
 |
| FALSIFY-ARCH-012 | Monotonicity — features only add roles | For any architecture A and any superset of its boolean features, required_roles(superset) ⊇ required_roles(A).
 | A feature flag accidentally removes roles instead of adding them.
 |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-ARCH-001 | ARCH-INV-001 | 4 | exhaustive |
| KANI-ARCH-002 | ARCH-INV-002 | 4 | exhaustive |
| KANI-ARCH-003 | ARCH-MONO-001 | 4 | exhaustive |
| KANI-ARCHIT-004 | Base roles always required | 8 | exhaustive |
| KANI-ARCHIT-005 | Constraint matrix exhaustive | 8 | exhaustive |
| KANI-ARCHIT-006 | Role count correctness | 8 | exhaustive |
| KANI-ARCHIT-007 | Weight completeness implies correct forward pass | 8 | exhaustive |
| KANI-ARCHIT-008 | Incomplete weights detected before forward pass | 8 | exhaustive |
| KANI-ARCHIT-009 | YAML matches Rust implementation | 8 | exhaustive |
| KANI-ARCHIT-010 | Adding features only adds roles | 8 | exhaustive |

## QA Gate

**Architecture Requirements Contract** (F-ARCH-001)

Per-architecture tensor weight role completeness quality gate

**Checks:** yaml_rust_parity, base_roles_universal, constraint_matrix_coverage, role_count_invariants, weight_completeness, alias_consistency, monotonicity

**Pass criteria:** All 12 falsification tests pass + 3 Kani harnesses verify

