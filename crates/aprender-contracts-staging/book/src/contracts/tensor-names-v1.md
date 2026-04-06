# tensor-names-v1

**Version:** 1.0.0

Architecture-specific tensor name resolution — source of truth

## References

- GH-311: Tensor name resolution contract
- architecture-requirements-v1.yaml: Weight role definitions
- realizar/src/tensor_names.rs: Generated Rust implementation

## Equations

### architecture_normalization

$$
normalize(raw) = architecture_map[raw] ?? "llama" (default)

$$

**Domain:** $raw: str (from config.json architectures or GGUF metadata)$

**Codomain:** $canonical architecture key$

**Invariants:**

- $Unknown architecture defaults to llama (safest default)$
- $Case-sensitive matching on HF class names$
- $Lowercase matching on GGUF arch strings$

### name_resolution

```
resolve(source, arch, role) =
  first(name ∈ names(arch, role) : source.has_tensor(name))
  ?? first(name ∈ fallback(role) : source.has_tensor(name))
  ?? first(name ∈ names(arch, role) : source.has_tensor(strip_prefix("model.", name)))
  ?? Error("tensor not found")

```

**Domain:** $source: TensorSource, arch: str, role: GlobalTensorRole | LayerTensorRole$

**Codomain:** `Result<Vec<f32>, Error>`

**Invariants:**

- $Architecture-specific names tried before fallbacks$
- $Bare name (without 'model.' prefix) tried as last resort$
- $Error message lists all attempted names for diagnostics$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | Architecture-specific names tried before fallbacks | $Architecture-specific names tried before fallbacks$ |
| 2 | invariant | Bare name (without 'model.' prefix) tried as last resort | $Bare name (without 'model.' prefix) tried as last resort$ |
| 3 | invariant | Unknown architecture defaults to llama (safest default) | $Unknown architecture defaults to llama (safest default)$ |
| 4 | invariant | Case-sensitive matching on HF class names | $Case-sensitive matching on HF class names$ |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-TNAME-001 | YAML-Rust parity | For every (architecture, role) pair in this YAML, the generated Rust lookup functions return identical name lists.
 | Generated code has diverged from YAML. Regenerate via build.rs.
 |
| FALSIFY-TNAME-002 | Architecture map completeness | Every architecture key referenced in global_roles, layer_roles, and fused_roles has a corresponding entry in architecture_map.
 | A new architecture was added to roles but not to the map.
 |
| FALSIFY-TNAME-003 | Fallback non-empty for required roles | Every role marked required=true has at least one fallback name.
 | A required role has no GGUF fallback, breaking models with non-standard naming.
 |
| FALSIFY-TNAME-004 | PhiForCausalLM maps to phi2, Phi3ForCausalLM maps to phi | normalize_architecture("PhiForCausalLM") == "phi2" and normalize_architecture("Phi3ForCausalLM") == "phi".
 | Phi-1.5/Phi-2 models will use wrong tensor names (gate_proj instead of fc1).
 |
| FALSIFY-TNAME-005 | Unknown architecture defaults to llama | normalize_architecture("FutureArch2027") returns "llama".
 | Default match arm is missing or maps to wrong architecture.
 |
| FALSIFY-TNAME-006 | GPT-2 bare names resolved | For GPT-2 architecture, global_names returns ["wte.weight", "transformer.wte.weight"] for embedding role (no "model." prefix).
 | GPT-2 names incorrectly prefixed, resolution will fail on real models.
 |
| FALSIFY-TNAME-007 | Fused QKV resolution for GPT-2/GPT-NeoX | fused_templates("gpt2", FusedQkv) returns non-empty list, while q_proj_weight templates for gpt2 returns empty list.
 | GPT-2 will try to load separate Q/K/V tensors that don't exist.
 |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-TENSOR-001 | Architecture-specific names tried before fallbacks | 8 | exhaustive |
| KANI-TENSOR-002 | Bare name (without 'model.' prefix) tried as last resort | 8 | exhaustive |
| KANI-TENSOR-003 | Unknown architecture defaults to llama (safest default) | 8 | exhaustive |
| KANI-TENSOR-004 | Case-sensitive matching on HF class names | 8 | exhaustive |

## QA Gate

**tensor-names-v1 Contract** (F-TNV-001)

Quality gate for Architecture-specific tensor name resolution — source of tru

**Checks:** validation, falsification

