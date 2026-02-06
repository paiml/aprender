# Compiler-Enforced Model Types & Model Oracle

**Version**: 1.0.0
**Status**: Draft
**Created**: 2026-02-06
**Author**: PAIML Engineering
**Tickets**: PMAT-240 through PMAT-253

> This specification defines a five-layer compiler enforcement architecture for model
> family contracts and introduces the `apr oracle` CLI command for model introspection.
> No code changes are included in this PR; implementation follows in subsequent tickets.

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Motivation](#2-motivation)
3. [Theoretical Foundation](#3-theoretical-foundation)
4. [Model Family Contract YAML Specification](#4-model-family-contract-yaml-specification)
   - [4.1 Directory Structure](#41-directory-structure)
   - [4.2 YAML Schema](#42-yaml-schema)
   - [4.3 Reference: qwen2.yaml](#43-reference-qwen2yaml)
   - [4.4 Reference: llama.yaml](#44-reference-llamayaml)
   - [4.5 Relationship to tensor-layout-v1.yaml](#45-relationship-to-tensor-layout-v1yaml)
5. [Compiler Enforcement Architecture](#5-compiler-enforcement-architecture)
   - [5.1 Layer 1: Clippy disallowed-methods](#51-layer-1-clippy-disallowed-methods)
   - [5.2 Layer 2: ModelFamily trait + ModelFamilyConfig](#52-layer-2-modelfamily-trait--modelfamilyconfig)
   - [5.3 Layer 3: PhantomData\<Layout\> markers on Validated\* types](#53-layer-3-phantomdatalayout-markers-on-validated-types)
   - [5.4 Layer 4: AprTransformer migration to Validated\* fields](#54-layer-4-aprtransformer-migration-to-validated-fields)
   - [5.5 Layer 5: build.rs YAML-to-Rust code generation](#55-layer-5-buildrs-yaml-to-rust-code-generation)
   - [5.6 Enforcement Matrix](#56-enforcement-matrix)
6. [apr oracle CLI Command](#6-apr-oracle-cli-command)
   - [6.1 Mode 1: Local File Analysis](#61-mode-1-local-file-analysis)
   - [6.2 Mode 2: HuggingFace API Query](#62-mode-2-huggingface-api-query)
   - [6.3 Mode 3: Contract Description](#63-mode-3-contract-description)
   - [6.4 Output Format](#64-output-format)
   - [6.5 ModelOracleReport Struct](#65-modeloraclereport-struct)
   - [6.6 Integration with apr-model-qa-playbook](#66-integration-with-apr-model-qa-playbook)
7. [Popperian Falsification Protocol](#7-popperian-falsification-protocol)
   - [7.1 Phase 1: Model Family Contracts](#71-phase-1-model-family-contracts)
   - [7.2 Phase 2: Oracle CLI](#72-phase-2-oracle-cli)
   - [7.3 Phase 3: Compiler Enforcement](#73-phase-3-compiler-enforcement)
   - [7.4 Phase 4: Build-Time Codegen](#74-phase-4-build-time-codegen)
8. [Implementation Roadmap](#8-implementation-roadmap)
   - [8.1 Phase 1: Foundation (PMAT-240..243)](#81-phase-1-foundation)
   - [8.2 Phase 2: Oracle CLI (PMAT-244..247)](#82-phase-2-oracle-cli)
   - [8.3 Phase 3: Type Enforcement (PMAT-248..249)](#83-phase-3-type-enforcement)
   - [8.4 Phase 4: Codegen & Generics (PMAT-250..251)](#84-phase-4-codegen--generics)
   - [8.5 Phase 5: Documentation & Expansion (PMAT-252..253)](#85-phase-5-documentation--expansion)
9. [Architectural Decisions](#9-architectural-decisions)
   - [9.1 YAML as Source of Truth](#91-yaml-as-source-of-truth)
   - [9.2 PhantomData\<Layout\> vs Separate Types](#92-phantomdatalayout-vs-separate-types)
   - [9.3 Generic AprTransformer\<F\> vs Box\<dyn ModelFamily\>](#93-generic-aprtransformerf-vs-boxdyn-modelfamily)
   - [9.4 New oracle Command vs Extending inspect](#94-new-oracle-command-vs-extending-inspect)
10. [References](#10-references)
11. [Appendix A: PMAT Ticket Descriptions](#appendix-a-pmat-ticket-descriptions)

---

## 1. Abstract

Aprender and realizar currently handle model loading through a single code path that
treats all transformer architectures identically. This design has produced three
categories of defects:

1. **Layout errors** (GH-202, GH-208): Column-major/row-major confusion caused garbage
   inference output. The root cause was the absence of compile-time enforcement for
   tensor layout conventions.

2. **Unvalidated model loading** (GH-213): Models from untested families were loaded
   without verifying that their tensor names, shapes, and constraints matched any known
   contract. Failures surfaced at inference time, not load time.

3. **Missing introspection** (GH-190): Users had no way to determine *before* loading
   whether a model file matched a supported architecture, what constraints apply, or
   whether the model's tensors conform to its family's contract.

This specification introduces:

- **Model Family Contract YAML files** that declare per-family architectural constraints,
  tensor templates, size variants, and chat template configurations.

- **A five-layer compiler enforcement stack** that progressively moves validation from
  runtime to compile time, making entire categories of defects unrepresentable.

- **The `apr oracle` CLI command** for model introspection: given a local file, a
  HuggingFace URI, or a family name, oracle reports the model's family, configuration,
  contract compliance, and certification status.

---

## 2. Motivation

### 2.1 Lessons from GH-202, GH-208, and GH-213

**GH-202** (lm_head garbage output): The lm_head weight matrix was stored with
shape `[hidden_dim, vocab_size]` instead of `[vocab_size, hidden_dim]`. The kernel
expected row-major `[out_dim, in_dim]`, but received transposed data. Result: every
inference token was `[PAD151935]`.

**Root cause**: No compile-time mechanism prevented passing a `Vec<f32>` with wrong
shape semantics to a matmul kernel. The `ValidatedWeight` type (PMAT-235) now catches
shape errors at construction, but layout semantics (row-major vs column-major) remain
a runtime convention, not a type-level guarantee.

**GH-208** (embedding transpose regression): A "fix" for GH-202 introduced an
unnecessary data transpose during GGUF import. GGUF data layout is already compatible
with row-major when shape metadata is swapped. The data transpose corrupted embeddings.

**Root cause**: The contract (`contracts/tensor-layout-v1.yaml`) was documentation,
not code. Developers could bypass it without compiler errors.

**GH-213** (sharded model loading): Sharded SafeTensors models from untested families
failed with cryptic "tensor not found" errors because the tensor name mapping
(`Architecture::map_name`) only covered Qwen2 and LLaMA.

**Root cause**: No mechanism existed to declare which model families are supported,
what tensor names they use, or what size variants exist. The `Architecture` enum is
a flat list with no associated contract data.

### 2.2 Current Gaps

| Gap | Description | Risk |
|-----|-------------|------|
| No family contracts | `Architecture` enum has no associated constraints | Runtime failures on unsupported models |
| Layout as convention | Row-major is documented but not enforced at type level | Regression (GH-208 happened despite GH-202 fix) |
| No model introspection | Users cannot query "what is this model?" before loading | Wasted time on incompatible models |
| Certification disconnect | Playbook certifications live in separate repo with no programmatic link | Cannot verify certification status from CLI |
| Hardcoded size configs | Model configs (hidden_dim, num_heads, etc.) are scattered across code | Adding new size variants requires code changes |

### 2.3 Design Goals

1. Make layout errors a **compile-time error**, not a runtime bug.
2. Make unsupported model families a **load-time rejection**, not an inference-time crash.
3. Provide **single-command introspection** for any model file or HuggingFace repo.
4. Maintain **YAML as the single source of truth** for model family contracts,
   consumable by aprender, realizar, and apr-model-qa-playbook.

---

## 3. Theoretical Foundation

This specification draws on seven established works in quality engineering, type
theory, and scientific methodology.

### 3.1 Poka-Yoke / Zero Quality Control

> "The most effective inspection is source inspection, which detects errors at the
> point where they are created and provides immediate feedback to prevent defects."

**Citation**: Shingo, S. (1986). *Zero Quality Control: Source Inspection and the
Poka-Yoke System*. Productivity Press. ISBN 0-915299-07-0.

**Application**: The `ValidatedWeight` newtype pattern (PMAT-235) makes it
impossible to construct a weight matrix without passing shape and data quality
checks. This specification extends Poka-Yoke to the model family level: the
`ModelFamily` trait makes it impossible to load a model without declaring and
validating its family contract.

### 3.2 Toyota Production System

> "We are not simply making automobiles. We are making the process of making
> automobiles."

**Citation**: Ohno, T. (1988). *Toyota Production System: Beyond Large-Scale
Production*. Productivity Press. ISBN 0-915299-14-3.

**Application**: The five-layer enforcement stack is a production line with
progressive quality gates. Each layer catches a class of defects that the
previous layer cannot. Layer 1 (clippy) is the fastest feedback; Layer 5
(build.rs codegen) is the most comprehensive. The goal is *jidoka* (automation
with a human touch): the compiler stops the line when it detects a defect.

### 3.3 Falsificationism

> "The criterion of the scientific status of a theory is its falsifiability."

**Citation**: Popper, K. (1959). *The Logic of Scientific Discovery*. Hutchinson
& Co. ISBN 0-415-27844-9.

**Application**: Every validation rule in this specification has explicit
falsification criteria (Section 7). A rule is meaningful only if it can be
disproven by a concrete test. If a falsification test passes (finds a
counterexample), the implementation is broken. This is the same pattern used
in `contracts/tensor-layout-v1.yaml` (FALSIFY-001 through FALSIFY-006).

### 3.4 Type-Driven Development

> "If it compiles, it works."

**Citation**: Brady, E. (2017). *Type-Driven Development with Idris*. Manning
Publications. ISBN 978-1-61729-302-5.

**Application**: The `PhantomData<RowMajor>` marker on `ValidatedWeight` encodes
layout semantics in the type system. A function that requires `ValidatedWeight<RowMajor>`
cannot accept `ValidatedWeight<ColumnMajor>` (which does not exist in APR, but the
type system prevents future regressions if it were introduced). This is stronger
than runtime assertions: the invalid state is unrepresentable, not merely checked.

### 3.5 Parse, Don't Validate

> "Use the type system to make invalid states unrepresentable. When you validate,
> you check a property but retain the unvalidated type. When you parse, you transform
> to a type that structurally guarantees the property."

**Citation**: Parsons, A. (2019). "Parse, Don't Validate." Blog post.

**Application**: The transition from `Vec<f32>` to `ValidatedEmbedding` is a parse,
not a validation. The raw data is consumed and the validated type is returned. There
is no way to access the data without first proving it satisfies the contract. The
`ModelFamily` trait extends this to the model level: raw tensor data is parsed into
a family-specific validated structure, not merely checked against a list of rules.

### 3.6 Typestate Programming

> "A typestate associates an abstract state with each variable and defines state
> transitions as function signatures."

**Citation**: Strom, R. E. & Yemini, S. (1986). "Typestate: A Programming Language
Concept for Enhancing Software Reliability." *IEEE Transactions on Software
Engineering*, SE-12(1), pp. 157-171.

**Application**: The `AprTransformer<F: ModelFamily>` generic parameter acts as a
typestate. An `AprTransformer<Qwen2>` cannot be passed to a function expecting
`AprTransformer<Llama>`. The model family is part of the type, not a runtime field
that could be wrong.

### 3.7 Deny Capabilities for Safe, Fast Actors

> "Capabilities that are not needed should be denied by default. This principle,
> applied to concurrent systems, prevents entire classes of race conditions."

**Citation**: Clebsch, S., Drossopoulou, S., Blessing, S., & McNeil, A. (2015).
"Deny Capabilities for Safe, Fast Actors." In *Proceedings of the 5th International
Workshop on Programming Based on Actors, Agents, and Decentralized Control (AGERE!)*.
ACM. pp. 1-12.

**Application**: Layer 1 (clippy `disallowed-methods`) denies the capability to call
column-major matmul kernels. This is not a warning or a convention; the denied
function cannot be called. The deny-by-default principle ensures that new code cannot
introduce layout errors without explicitly overriding the deny rule, which would
trigger code review.

---

## 4. Model Family Contract YAML Specification

### 4.1 Directory Structure

```
contracts/
├── tensor-layout-v1.yaml          # Existing: per-tensor layout contract
└── model-families/
    ├── _schema.yaml               # JSON Schema for model family YAMLs
    ├── qwen2.yaml                 # Qwen2 / Qwen2.5 / Qwen2.5-Coder
    ├── llama.yaml                 # LLaMA 2 / LLaMA 3 / LLaMA 3.2
    ├── whisper.yaml               # OpenAI Whisper (encoder-decoder)
    └── bert.yaml                  # Google BERT (encoder-only)
```

Each YAML file is the **single source of truth** for one model family. Consumers:

- **aprender**: `build.rs` reads YAMLs and generates `ModelFamilyConfig` structs
- **realizar**: Reads at runtime to validate loaded models
- **apr-model-qa-playbook**: Generates test matrices from size variants
- **apr oracle**: Renders contract descriptions for users

### 4.2 YAML Schema

```yaml
# Required top-level fields
family: string                    # Canonical family name (e.g., "qwen2")
display_name: string              # Human-readable name (e.g., "Qwen2.5-Coder")
vendor: string                    # Organization (e.g., "Alibaba")
architectures: [string]           # HF config.json model_type values
hf_pattern: string                # Glob for matching HF repo names

# Size variants
size_variants:
  <name>:                         # e.g., "0.5b", "1.5b", "7b"
    parameters: string            # e.g., "0.5B"
    hidden_dim: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    intermediate_dim: int
    vocab_size: int
    max_position_embeddings: int
    head_dim: int                 # Derived: hidden_dim / num_heads
    rope_theta: float             # RoPE base frequency
    rms_norm_eps: float           # RMSNorm epsilon

# Architectural constraints
constraints:
  attention_type: string          # "gqa" | "mha" | "mqa"
  activation: string              # "silu" | "gelu" | "relu"
  norm_type: string               # "rmsnorm" | "layernorm"
  has_bias: bool                  # Whether projections have bias terms
  tied_embeddings: bool           # Whether embed_tokens == lm_head
  positional_encoding: string     # "rope" | "alibi" | "absolute" | "relative"
  mlp_type: string                # "swiglu" | "gelu_mlp" | "gated_mlp"

# Tensor name template (parameterized by layer index)
tensor_template:
  embedding: string               # e.g., "model.embed_tokens.weight"
  lm_head: string                 # e.g., "lm_head.weight"
  final_norm: string              # e.g., "model.norm.weight"
  per_layer:                      # Indexed by {n}
    q_proj: string
    k_proj: string
    v_proj: string
    o_proj: string
    gate_proj: string
    up_proj: string
    down_proj: string
    input_layernorm: string
    post_attention_layernorm: string
    q_proj_bias: string | null     # null = tensor does not exist
    k_proj_bias: string | null
    v_proj_bias: string | null

# Shape template (parameterized by config values)
shape_template:
  embedding: "[vocab_size, hidden_dim]"
  lm_head: "[vocab_size, hidden_dim]"
  q_proj: "[num_heads * head_dim, hidden_dim]"
  k_proj: "[num_kv_heads * head_dim, hidden_dim]"
  v_proj: "[num_kv_heads * head_dim, hidden_dim]"
  o_proj: "[hidden_dim, num_heads * head_dim]"
  gate_proj: "[intermediate_dim, hidden_dim]"
  up_proj: "[intermediate_dim, hidden_dim]"
  down_proj: "[hidden_dim, intermediate_dim]"
  input_layernorm: "[hidden_dim]"
  post_attention_layernorm: "[hidden_dim]"

# Supported quantizations
quantizations:
  - q4_k_m
  - q5_k_m
  - q6_k
  - q8_0
  - f16
  - f32

# Chat template
chat_template:
  format: string                  # e.g., "chatml", "llama", "mistral"
  template: string                # Jinja2 template string
  bos_token: string
  eos_token: string
  special_tokens:
    <token_name>: string          # e.g., im_start: "<|im_start|>"

# Certification cross-reference
certification:
  playbook_path: string           # Relative path to playbook YAML
  csv_family_key: string          # Key in models.csv family column
  size_categories:
    <variant>: string             # Maps variant to size category
```

### 4.3 Reference: qwen2.yaml

```yaml
family: qwen2
display_name: "Qwen2 / Qwen2.5-Coder"
vendor: Alibaba
architectures:
  - Qwen2ForCausalLM
hf_pattern: "Qwen/Qwen2*"

size_variants:
  0.5b:
    parameters: "0.5B"
    hidden_dim: 896
    num_layers: 24
    num_heads: 14
    num_kv_heads: 2
    intermediate_dim: 4864
    vocab_size: 151936
    max_position_embeddings: 32768
    head_dim: 64
    rope_theta: 1000000.0
    rms_norm_eps: 0.000001
  1.5b:
    parameters: "1.5B"
    hidden_dim: 1536
    num_layers: 28
    num_heads: 12
    num_kv_heads: 2
    intermediate_dim: 8960
    vocab_size: 151936
    max_position_embeddings: 32768
    head_dim: 128
    rope_theta: 1000000.0
    rms_norm_eps: 0.000001
  3b:
    parameters: "3B"
    hidden_dim: 2048
    num_layers: 36
    num_heads: 16
    num_kv_heads: 2
    intermediate_dim: 11008
    vocab_size: 151936
    max_position_embeddings: 32768
    head_dim: 128
    rope_theta: 1000000.0
    rms_norm_eps: 0.000001
  7b:
    parameters: "7B"
    hidden_dim: 3584
    num_layers: 28
    num_heads: 28
    num_kv_heads: 4
    intermediate_dim: 18944
    vocab_size: 152064
    max_position_embeddings: 131072
    head_dim: 128
    rope_theta: 1000000.0
    rms_norm_eps: 0.000001
  14b:
    parameters: "14B"
    hidden_dim: 5120
    num_layers: 48
    num_heads: 40
    num_kv_heads: 8
    intermediate_dim: 13824
    vocab_size: 152064
    max_position_embeddings: 131072
    head_dim: 128
    rope_theta: 1000000.0
    rms_norm_eps: 0.000001
  32b:
    parameters: "32B"
    hidden_dim: 5120
    num_layers: 64
    num_heads: 40
    num_kv_heads: 8
    intermediate_dim: 27648
    vocab_size: 152064
    max_position_embeddings: 131072
    head_dim: 128
    rope_theta: 1000000.0
    rms_norm_eps: 0.000001

constraints:
  attention_type: gqa
  activation: silu
  norm_type: rmsnorm
  has_bias: true
  tied_embeddings: false
  positional_encoding: rope
  mlp_type: swiglu

tensor_template:
  embedding: "model.embed_tokens.weight"
  lm_head: "lm_head.weight"
  final_norm: "model.norm.weight"
  per_layer:
    q_proj: "model.layers.{n}.self_attn.q_proj.weight"
    k_proj: "model.layers.{n}.self_attn.k_proj.weight"
    v_proj: "model.layers.{n}.self_attn.v_proj.weight"
    o_proj: "model.layers.{n}.self_attn.o_proj.weight"
    gate_proj: "model.layers.{n}.mlp.gate_proj.weight"
    up_proj: "model.layers.{n}.mlp.up_proj.weight"
    down_proj: "model.layers.{n}.mlp.down_proj.weight"
    input_layernorm: "model.layers.{n}.input_layernorm.weight"
    post_attention_layernorm: "model.layers.{n}.post_attention_layernorm.weight"
    q_proj_bias: "model.layers.{n}.self_attn.q_proj.bias"
    k_proj_bias: "model.layers.{n}.self_attn.k_proj.bias"
    v_proj_bias: "model.layers.{n}.self_attn.v_proj.bias"

shape_template:
  embedding: "[vocab_size, hidden_dim]"
  lm_head: "[vocab_size, hidden_dim]"
  q_proj: "[num_heads * head_dim, hidden_dim]"
  k_proj: "[num_kv_heads * head_dim, hidden_dim]"
  v_proj: "[num_kv_heads * head_dim, hidden_dim]"
  o_proj: "[hidden_dim, num_heads * head_dim]"
  gate_proj: "[intermediate_dim, hidden_dim]"
  up_proj: "[intermediate_dim, hidden_dim]"
  down_proj: "[hidden_dim, intermediate_dim]"
  input_layernorm: "[hidden_dim]"
  post_attention_layernorm: "[hidden_dim]"

quantizations:
  - q4_k_m
  - q5_k_m
  - q6_k
  - q8_0
  - f16
  - f32

chat_template:
  format: chatml
  template: "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
  bos_token: ""
  eos_token: "<|im_end|>"
  special_tokens:
    im_start: "<|im_start|>"
    im_end: "<|im_end|>"
    endoftext: "<|endoftext|>"

certification:
  playbook_path: "../apr-model-qa-playbook/playbooks/models/qwen2.5-coder-{size}.playbook.yaml"
  csv_family_key: "qwen-coder"
  size_categories:
    0.5b: tiny
    1.5b: small
    3b: small
    7b: medium
    14b: large
    32b: xlarge
```

### 4.4 Reference: llama.yaml

```yaml
family: llama
display_name: "LLaMA 3 / LLaMA 3.2"
vendor: Meta
architectures:
  - LlamaForCausalLM
hf_pattern: "meta-llama/Llama-*"

size_variants:
  1b:
    parameters: "1B"
    hidden_dim: 2048
    num_layers: 16
    num_heads: 32
    num_kv_heads: 8
    intermediate_dim: 8192
    vocab_size: 128256
    max_position_embeddings: 131072
    head_dim: 64
    rope_theta: 500000.0
    rms_norm_eps: 0.00001
  3b:
    parameters: "3B"
    hidden_dim: 3072
    num_layers: 28
    num_heads: 24
    num_kv_heads: 8
    intermediate_dim: 8192
    vocab_size: 128256
    max_position_embeddings: 131072
    head_dim: 128
    rope_theta: 500000.0
    rms_norm_eps: 0.00001
  8b:
    parameters: "8B"
    hidden_dim: 4096
    num_layers: 32
    num_heads: 32
    num_kv_heads: 8
    intermediate_dim: 14336
    vocab_size: 128256
    max_position_embeddings: 131072
    head_dim: 128
    rope_theta: 500000.0
    rms_norm_eps: 0.00001
  70b:
    parameters: "70B"
    hidden_dim: 8192
    num_layers: 80
    num_heads: 64
    num_kv_heads: 8
    intermediate_dim: 28672
    vocab_size: 128256
    max_position_embeddings: 131072
    head_dim: 128
    rope_theta: 500000.0
    rms_norm_eps: 0.00001

constraints:
  attention_type: gqa
  activation: silu
  norm_type: rmsnorm
  has_bias: false
  tied_embeddings: false
  positional_encoding: rope
  mlp_type: swiglu

tensor_template:
  embedding: "model.embed_tokens.weight"
  lm_head: "lm_head.weight"
  final_norm: "model.norm.weight"
  per_layer:
    q_proj: "model.layers.{n}.self_attn.q_proj.weight"
    k_proj: "model.layers.{n}.self_attn.k_proj.weight"
    v_proj: "model.layers.{n}.self_attn.v_proj.weight"
    o_proj: "model.layers.{n}.self_attn.o_proj.weight"
    gate_proj: "model.layers.{n}.mlp.gate_proj.weight"
    up_proj: "model.layers.{n}.mlp.up_proj.weight"
    down_proj: "model.layers.{n}.mlp.down_proj.weight"
    input_layernorm: "model.layers.{n}.input_layernorm.weight"
    post_attention_layernorm: "model.layers.{n}.post_attention_layernorm.weight"
    q_proj_bias: null
    k_proj_bias: null
    v_proj_bias: null

shape_template:
  embedding: "[vocab_size, hidden_dim]"
  lm_head: "[vocab_size, hidden_dim]"
  q_proj: "[num_heads * head_dim, hidden_dim]"
  k_proj: "[num_kv_heads * head_dim, hidden_dim]"
  v_proj: "[num_kv_heads * head_dim, hidden_dim]"
  o_proj: "[hidden_dim, num_heads * head_dim]"
  gate_proj: "[intermediate_dim, hidden_dim]"
  up_proj: "[intermediate_dim, hidden_dim]"
  down_proj: "[hidden_dim, intermediate_dim]"
  input_layernorm: "[hidden_dim]"
  post_attention_layernorm: "[hidden_dim]"

quantizations:
  - q4_k_m
  - q5_k_m
  - q6_k
  - q8_0
  - f16
  - f32

chat_template:
  format: llama
  template: "{% for message in messages %}{% if message['role'] == 'system' %}<|start_header_id|>system<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{% elif message['role'] == 'user' %}<|start_header_id|>user<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{% elif message['role'] == 'assistant' %}<|start_header_id|>assistant<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{% endif %}{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
  bos_token: "<|begin_of_text|>"
  eos_token: "<|eot_id|>"
  special_tokens:
    begin_of_text: "<|begin_of_text|>"
    end_of_text: "<|end_of_text|>"
    start_header_id: "<|start_header_id|>"
    end_header_id: "<|end_header_id|>"
    eot_id: "<|eot_id|>"

certification:
  playbook_path: "../apr-model-qa-playbook/playbooks/models/llama-3.2-{size}.playbook.yaml"
  csv_family_key: "llama"
  size_categories:
    1b: small
    3b: small
    8b: medium
    70b: xlarge
```

### 4.5 Relationship to tensor-layout-v1.yaml

`tensor-layout-v1.yaml` defines the **per-tensor** layout contract: which tensors
need transposing, kernel shapes, byte size calculations, and semantic validation
thresholds. It is architecture-agnostic.

Model family YAMLs define the **per-family** architectural contract: which tensors
exist, what their parameterized shapes are, which config values apply, and how
tensor names map between GGUF and APR conventions.

The two contracts are complementary:

| Aspect | tensor-layout-v1.yaml | model-families/*.yaml |
|--------|----------------------|----------------------|
| Scope | Per-tensor rules | Per-family architecture |
| Parametrized by | None (universal rules) | Model config (hidden_dim, etc.) |
| Validates | Shapes, layout, data quality | Tensor names, sizes, constraints |
| Consumed by | `layout_contract.rs`, `validated_tensors.rs` | `model_family.rs`, `build.rs` |
| Changed when | New tensor type discovered | New model family added |

The `shape_template` in a family YAML is evaluated using the family's config values,
then validated against `tensor-layout-v1.yaml` rules. For example, `qwen2.yaml`
declares `lm_head: "[vocab_size, hidden_dim]"` and `tensor-layout-v1.yaml` declares
that `lm_head` must have shape `[vocab, hidden]` and is critical. Both contracts
must be satisfied.

---

## 5. Compiler Enforcement Architecture

### 5.1 Layer 1: Clippy disallowed-methods

**Enforcement**: Compile time
**Catches**: Column-major kernel imports

The `.clippy.toml` file already bans `unwrap()` via `disallowed-methods`. This
layer extends the ban to column-major matmul kernel functions that should never
be called in APR/realizar code.

```toml
# .clippy.toml additions (PMAT-243)
[[disallowed-methods]]
path = "trueno::backends::q4k::matmul_q4k_f32_colmajor"
reason = "LAYOUT-001: APR is exclusively row-major. Use fused_q4k_parallel_matvec."

[[disallowed-methods]]
path = "trueno::backends::q6k::matmul_q6k_f32_colmajor"
reason = "LAYOUT-001: APR is exclusively row-major. Use fused_q6k_parallel_matvec."

[[disallowed-methods]]
path = "trueno::backends::q4k::matmul_q4k_f32_colmajor_dispatch"
reason = "LAYOUT-001: APR is exclusively row-major. Use fused_q4k_parallel_matvec."

[[disallowed-methods]]
path = "trueno::backends::q6k::matmul_q6k_f32_colmajor_dispatch"
reason = "LAYOUT-001: APR is exclusively row-major. Use fused_q6k_parallel_matvec."
```

**Why Layer 1 matters**: This is the fastest feedback loop. `cargo clippy` runs in
seconds and catches the most common layout error (importing a column-major kernel)
before any code executes.

### 5.2 Layer 2: ModelFamily trait + ModelFamilyConfig

**Enforcement**: Load time
**Catches**: Unknown/uncontracted model families

```rust
// src/format/model_family.rs (PMAT-241)

/// Configuration for a specific model size within a family.
///
/// Generated from YAML by build.rs (PMAT-250) or loaded at runtime.
#[derive(Debug, Clone)]
pub struct ModelSizeConfig {
    pub parameters: String,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub intermediate_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub head_dim: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
}

/// Architectural constraints for a model family.
#[derive(Debug, Clone)]
pub struct ModelConstraints {
    pub attention_type: AttentionType,
    pub activation: Activation,
    pub norm_type: NormType,
    pub has_bias: bool,
    pub tied_embeddings: bool,
    pub positional_encoding: PositionalEncoding,
    pub mlp_type: MlpType,
}

/// Complete configuration for a model family.
#[derive(Debug, Clone)]
pub struct ModelFamilyConfig {
    pub family: String,
    pub display_name: String,
    pub vendor: String,
    pub architectures: Vec<String>,
    pub size_variants: std::collections::HashMap<String, ModelSizeConfig>,
    pub constraints: ModelConstraints,
    pub tensor_template: TensorTemplate,
    pub shape_template: ShapeTemplate,
}

/// Trait implemented by each model family.
///
/// This trait is the compile-time bridge between YAML contracts and Rust code.
/// Implementations are generated by build.rs from model family YAMLs (PMAT-250).
pub trait ModelFamily: std::fmt::Debug + Send + Sync {
    /// Canonical family name (e.g., "qwen2")
    fn family_name(&self) -> &str;

    /// Get configuration for a specific size variant
    fn size_config(&self, size: &str) -> Option<&ModelSizeConfig>;

    /// Detect size variant from model config (hidden_dim, num_layers, etc.)
    fn detect_size(&self, hidden_dim: usize, num_layers: usize) -> Option<String>;

    /// Get architectural constraints
    fn constraints(&self) -> &ModelConstraints;

    /// Map GGUF tensor name to APR canonical name
    fn map_tensor_name(&self, gguf_name: &str) -> String;

    /// Expected tensor count for a given size variant
    fn expected_tensor_count(&self, size: &str) -> Option<usize>;

    /// Validate that a set of tensor names matches the contract
    fn validate_tensor_names(&self, names: &[&str], size: &str)
        -> Result<(), ContractError>;
}
```

**Why Layer 2 matters**: This catches the GH-213 class of errors. When a model is
loaded, the `Architecture` detection now maps to a `ModelFamily` implementation. If
no family matches, the error is actionable ("Unknown model family; known families:
qwen2, llama, whisper, bert") rather than a cryptic tensor-not-found crash.

### 5.3 Layer 3: PhantomData\<Layout\> markers on Validated\* types

**Enforcement**: Compile time
**Catches**: Layout mismatch in kernel calls

```rust
// src/format/validated_tensors.rs extension (PMAT-248)

use std::marker::PhantomData;

/// Marker type for row-major layout (APR convention).
#[derive(Debug, Clone, Copy)]
pub struct RowMajor;

/// Validated weight matrix with layout encoded in the type.
///
/// The PhantomData<L> marker ensures that row-major and column-major
/// weights cannot be accidentally mixed. Since APR is exclusively
/// row-major, the ColumnMajor marker does not exist — making the
/// invalid state literally unrepresentable.
#[derive(Debug, Clone)]
pub struct ValidatedWeight<L = RowMajor> {
    data: Vec<f32>,
    out_dim: usize,
    in_dim: usize,
    name: String,
    stats: TensorStats,
    _layout: PhantomData<L>,
}

impl ValidatedWeight<RowMajor> {
    /// Construct a validated row-major weight matrix.
    ///
    /// This is the ONLY constructor. There is no way to create a
    /// ValidatedWeight<ColumnMajor> because ColumnMajor does not exist.
    pub fn new(
        data: Vec<f32>,
        out_dim: usize,
        in_dim: usize,
        name: &str,
    ) -> Result<Self, ContractValidationError> {
        // ... existing validation gates ...
        Ok(Self {
            data,
            out_dim,
            in_dim,
            name: name.to_string(),
            stats,
            _layout: PhantomData,
        })
    }
}
```

**Why Layer 3 matters**: The `PhantomData` marker has zero runtime cost but provides
a compile-time guarantee. If a future developer were to add column-major support
(e.g., for a different backend), they would need to create a `ColumnMajor` marker type
and a separate constructor. Any attempt to pass a `ValidatedWeight<ColumnMajor>` to a
function expecting `ValidatedWeight<RowMajor>` would be a compile error.

### 5.4 Layer 4: AprTransformer migration to Validated\* fields

**Enforcement**: Compile time
**Catches**: Unvalidated tensor data in model structs

The existing `AprTransformer` struct uses `Vec<f32>` for its fields. After migration:

```rust
// realizar/src/apr_transformer/mod.rs (PMAT-249)

pub struct AprTransformer {
    // BEFORE: pub embedding: Vec<f32>,
    // AFTER:
    pub embedding: ValidatedEmbedding,
    pub lm_head: ValidatedWeight<RowMajor>,
    pub final_norm: ValidatedVector,
    pub layers: Vec<AprTransformerLayer>,
}

pub struct AprTransformerLayer {
    pub q_proj: ValidatedWeight<RowMajor>,
    pub k_proj: ValidatedWeight<RowMajor>,
    pub v_proj: ValidatedWeight<RowMajor>,
    pub o_proj: ValidatedWeight<RowMajor>,
    pub gate_proj: ValidatedWeight<RowMajor>,
    pub up_proj: ValidatedWeight<RowMajor>,
    pub down_proj: ValidatedWeight<RowMajor>,
    pub input_layernorm: ValidatedVector,
    pub post_attention_layernorm: ValidatedVector,
    // Bias vectors (optional, family-dependent)
    pub q_proj_bias: Option<ValidatedVector>,
    pub k_proj_bias: Option<ValidatedVector>,
    pub v_proj_bias: Option<ValidatedVector>,
}
```

**Why Layer 4 matters**: This is the Poka-Yoke completion. After migration, it is
**impossible** to construct an `AprTransformer` with unvalidated data. Every field
requires passing through a validation constructor. The PMAT-234 bug (94.5% zeros
loading successfully) is physically unrepresentable.

### 5.5 Layer 5: build.rs YAML-to-Rust code generation

**Enforcement**: Build time
**Catches**: YAML/Rust contract drift

```rust
// build.rs (PMAT-250)

fn main() {
    // Read model family YAMLs
    let families_dir = Path::new("contracts/model-families");

    if families_dir.exists() {
        let mut generated = String::new();

        for entry in std::fs::read_dir(families_dir).expect("read contracts dir") {
            let entry = entry.expect("dir entry");
            let path = entry.path();

            if path.extension().map_or(false, |e| e == "yaml")
                && !path.file_name().map_or(false, |n| n.to_str()
                    .map_or(false, |s| s.starts_with('_')))
            {
                let yaml_content = std::fs::read_to_string(&path)
                    .expect("read YAML");
                generated.push_str(&generate_family_impl(&yaml_content, &path));
            }
        }

        let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR");
        let out_path = Path::new(&out_dir).join("model_families_generated.rs");
        std::fs::write(&out_path, generated).expect("write generated code");
    }

    // Tell Cargo to re-run if any YAML changes
    println!("cargo:rerun-if-changed=contracts/model-families");
}
```

The generated code includes:

1. `const` definitions for each size variant's config values
2. `ModelFamily` trait implementations for each family
3. A `KNOWN_FAMILIES` array for runtime family detection
4. Static assertions that YAML values match Rust expectations

**Why Layer 5 matters**: This closes the loop between the YAML source of truth and the
Rust code. If a developer changes a YAML value (e.g., adjusts `hidden_dim` for a new
size variant), the generated code updates automatically. If the YAML schema changes in
a way that breaks the code generator, the build fails — not inference.

### 5.6 Enforcement Matrix

| Layer | Mechanism | Catches | When | Cost |
|-------|-----------|---------|------|------|
| 1 | Clippy `disallowed-methods` | Column-major kernel imports | `cargo clippy` | 0 runtime |
| 2 | `ModelFamily` trait | Unknown model families, wrong tensor names | Model load | Negligible (name matching) |
| 3 | `PhantomData<RowMajor>` | Layout type mismatch | `cargo build` | 0 runtime |
| 4 | `Validated*` fields on `AprTransformer` | Unvalidated tensor data | `cargo build` | Validation at construction |
| 5 | `build.rs` YAML codegen | YAML/Rust contract drift | `cargo build` | Build-time code generation |

**Cumulative guarantee**: After all five layers are implemented, the following invariant
holds:

> If `cargo build` succeeds and a model loads without error, then:
> 1. No column-major kernel is called anywhere in the codebase.
> 2. The model's family is known and contracted.
> 3. All tensor data is row-major (by type).
> 4. All tensor data has passed density, NaN, Inf, and distribution checks.
> 5. The Rust code matches the YAML contract exactly.

---

## 6. apr oracle CLI Command

### 6.1 Mode 1: Local File Analysis

```bash
apr oracle model.gguf
apr oracle model.safetensors
apr oracle model.apr
```

**Behavior**:

1. Open file using `RosettaStone::inspect()` to get `InspectionReport`
2. Detect model family by matching tensor names against known `tensor_template` patterns
3. Detect size variant by matching `(hidden_dim, num_layers)` against `size_variants`
4. Validate tensor names and shapes against family contract
5. Output `ModelOracleReport`

**Flags**:

- `--compliance`: Run full contract compliance check (tensor names, shapes, counts)
- `--tensors`: List all tensor shapes alongside contract expectations
- `--json`: Output as JSON `ModelOracleReport`
- `--verbose`: Show detailed matching logic

```
$ apr oracle qwen2.5-coder-1.5b-q4_k_m.gguf

Model Oracle Report
  File: qwen2.5-coder-1.5b-q4_k_m.gguf
  Format: GGUF
  Family: qwen2 (Qwen2 / Qwen2.5-Coder)
  Size: 1.5B (hidden=1536, layers=28, heads=12, kv_heads=2)
  Quantization: Q4_K_M
  Tensors: 339 (expected: 339)
  Contract: COMPLIANT
  Certification: BLOCKED (MQS 415, see playbook qwen2.5-coder-1.5b)

Constraints:
  Attention: GQA (12 heads, 2 KV heads)
  Activation: SiLU
  Norm: RMSNorm (eps=1e-6)
  Bias: yes (Q/K/V projections)
  Tied embeddings: no
  MLP: SwiGLU

Chat Template: ChatML
  BOS: (none)
  EOS: <|im_end|>
```

### 6.2 Mode 2: HuggingFace API Query

```bash
apr oracle hf://Qwen/Qwen2.5-Coder-1.5B-Instruct
apr oracle hf://meta-llama/Llama-3.2-1B-Instruct
```

**Behavior**:

1. Query HuggingFace API for `config.json`: `GET https://huggingface.co/{org}/{repo}/raw/main/config.json`
2. Extract `model_type` from config.json (e.g., `"qwen2"`, `"llama"`)
3. Match `model_type` against `architectures` field in known family YAMLs
4. Extract size config from `config.json` fields (`hidden_size`, `num_hidden_layers`, etc.)
5. Match against `size_variants` to identify the exact variant
6. Output `ModelOracleReport` with HuggingFace-specific fields

**Flags**:

- `--json`: Output as JSON
- `--offline`: Fail if network access is needed (Sovereign AI compliance)

```
$ apr oracle hf://Qwen/Qwen2.5-Coder-1.5B-Instruct

Model Oracle Report
  Source: hf://Qwen/Qwen2.5-Coder-1.5B-Instruct
  Family: qwen2 (Qwen2 / Qwen2.5-Coder)
  Size: 1.5B (hidden=1536, layers=28, heads=12, kv_heads=2)
  Formats: safetensors (2 shards), gguf (community)
  Certification: BLOCKED (MQS 415)
  Inference Verified: yes (qwen2 + llama verified)
```

### 6.3 Mode 3: Contract Description

```bash
apr oracle --family qwen2
apr oracle --family llama
apr oracle --family whisper
```

**Behavior**:

1. Load the family YAML from `contracts/model-families/{family}.yaml`
2. Display the complete contract: size variants, constraints, tensor templates
3. Optionally filter to a specific size variant with `--size`

```
$ apr oracle --family qwen2 --size 0.5b

Qwen2 Family Contract
  Vendor: Alibaba
  Architectures: Qwen2ForCausalLM
  HF Pattern: Qwen/Qwen2*

  Size: 0.5B
    hidden_dim: 896
    num_layers: 24
    num_heads: 14
    num_kv_heads: 2
    intermediate_dim: 4864
    vocab_size: 151936
    head_dim: 64
    rope_theta: 1000000.0

  Tensor Template (339 tensors for 24 layers):
    model.embed_tokens.weight           [151936, 896]
    lm_head.weight                      [151936, 896]
    model.norm.weight                   [896]
    model.layers.{n}.self_attn.q_proj.weight   [896, 896]
    model.layers.{n}.self_attn.k_proj.weight   [128, 896]
    ... (14 per layer)

  Constraints:
    Attention: GQA | Activation: SiLU | Norm: RMSNorm
    Bias: yes | Tied: no | MLP: SwiGLU | Position: RoPE
```

### 6.4 Output Format

All modes support `--json` for machine-readable output. The JSON output matches
the `ModelOracleReport` struct.

Text output follows the existing `apr inspect` formatting conventions:
- Section headers in all-caps
- Key-value pairs with fixed-width key columns
- Tensor lists truncated with `... (N more) ...` for large models

### 6.5 ModelOracleReport Struct

```rust
// crates/apr-cli/src/commands/oracle.rs (PMAT-244)

/// Complete oracle report for a model
#[derive(Debug, Clone, Serialize)]
pub struct ModelOracleReport {
    /// Source path, HF URI, or family name
    pub source: String,
    /// Analysis mode (local, hf, family)
    pub mode: OracleMode,
    /// Detected or specified model family
    pub family: Option<FamilyInfo>,
    /// Detected size variant
    pub size_variant: Option<SizeVariantInfo>,
    /// Format information (for local files)
    pub format: Option<FormatInfo>,
    /// Contract compliance result (for local files with --compliance)
    pub compliance: Option<ComplianceResult>,
    /// Certification status from apr-model-qa-playbook
    pub certification: Option<CertificationInfo>,
    /// Tensor list (for --tensors flag)
    pub tensors: Option<Vec<TensorComplianceEntry>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FamilyInfo {
    pub name: String,
    pub display_name: String,
    pub vendor: String,
    pub architectures: Vec<String>,
    pub constraints: ConstraintsSummary,
    pub chat_template_format: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SizeVariantInfo {
    pub name: String,
    pub parameters: String,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub intermediate_dim: usize,
    pub vocab_size: usize,
    pub expected_tensor_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ComplianceResult {
    pub is_compliant: bool,
    pub tensor_count_match: bool,
    pub missing_tensors: Vec<String>,
    pub unexpected_tensors: Vec<String>,
    pub shape_mismatches: Vec<ShapeMismatch>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CertificationInfo {
    pub status: String,           // "CERTIFIED", "BLOCKED", "PENDING"
    pub mqs_score: Option<u32>,
    pub grade: Option<String>,
    pub certified_tier: Option<String>,
    pub playbook_path: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TensorComplianceEntry {
    pub name: String,
    pub expected_shape: Vec<usize>,
    pub actual_shape: Option<Vec<usize>>,
    pub compliant: bool,
    pub note: Option<String>,
}
```

### 6.6 Integration with apr-model-qa-playbook

The oracle command reads certification data from two sources:

1. **models.csv** (`../apr-model-qa-playbook/docs/certifications/models.csv`):
   Contains MQS scores, grades, certification tiers, and gate pass/fail per model.

2. **Playbook YAMLs** (`../apr-model-qa-playbook/playbooks/models/`): Contains
   test matrices, oracle configurations, and gate definitions.

The oracle maps a detected model to its certification entry using the
`certification.csv_family_key` from the family YAML and the detected size variant.

```
model_id → family YAML → csv_family_key → models.csv lookup → CertificationInfo
```

Example: `qwen2.5-coder-1.5b-q4_k_m.gguf` → `qwen2.yaml` → `csv_family_key: "qwen-coder"`
→ match `Qwen/Qwen2.5-Coder-1.5B-Instruct` → `status: BLOCKED, mqs_score: 415`.

If `../apr-model-qa-playbook` is not present on disk, certification fields are omitted
with a note: "Certification data not available (apr-model-qa-playbook not found)."

---

## 7. Popperian Falsification Protocol

Per Popper (1959), each validation rule must make a prediction that could be proven
false. If a falsification test finds a counterexample, the implementation is broken.

### 7.1 Phase 1: Model Family Contracts

**FALSIFY-MFC-001**: Family detection accuracy (best-match scoring)

```
Prediction: detect_family() uses best-match scoring to separate families by
  tensor naming specificity:

  (a) Tensor names WITH bias patterns (q_proj.bias, k_proj.bias, v_proj.bias)
      → detected as a bias-bearing family (Qwen2 or Phi, score 13)
      → NOT detected as a bias-free family (LLaMA, DeepSeek, Mistral, score 10)

  (b) Tensor names WITHOUT bias patterns
      → detected as a bias-free family (LLaMA, DeepSeek, Mistral, score 10)
      → NOT detected as a bias-bearing family (Qwen2, Phi would score 10, not 13)

  (c) model_type detection (from HF config.json) is always unambiguous:
      detect_from_model_type("qwen2") → "qwen2", never "phi"

Mechanism: detect_family() counts matching per-layer patterns per family.
  The highest scorer wins. Ties are broken by alphabetical order (deterministic
  but not semantically meaningful — use detect_from_model_type() when metadata
  is available).

Known limitation: Families with identical tensor naming AND identical
  per-layer pattern counts are indistinguishable from tensor names alone:
  - Bias group: Qwen2, Phi (both 12 per-layer)
  - No-bias group: LLaMA, DeepSeek, Mistral (all 9 per-layer)
  Within each group, model_type from config.json or GGUF metadata is required
  for unambiguous family identification.

Falsification tests:
  1. Tensor names WITH biases → result is in {phi, qwen2} (bias families)
  2. Tensor names WITHOUT biases → result is in {deepseek, llama, mistral}
  3. Whisper/BERT names → NOT detected as any LLaMA-style family
  4. detect_from_model_type("qwen2") → exactly "qwen2"
  5. Random names → None

If tests fail: Best-match scoring or model_type disambiguation is broken.
```

**FALSIFY-MFC-002**: Size variant detection accuracy

```
Prediction: Given hidden_dim=1536 and num_layers=28, detect_size() returns "1.5b".

Falsification test:
  fn falsify_mfc_002() {
      let qwen2 = Qwen2Family::new();
      assert_eq!(qwen2.detect_size(1536, 28), Some("1.5b".to_string()));
      assert_eq!(qwen2.detect_size(896, 24), Some("0.5b".to_string()));
      assert_eq!(qwen2.detect_size(999, 99), None); // Unknown config
  }

If test fails: Size variant detection does not match YAML contract.
```

**FALSIFY-MFC-003**: Tensor name validation rejects wrong names

```
Prediction: validate_tensor_names() rejects a tensor list that includes names
  from a different family (e.g., LLaMA names in a Qwen2 contract check).

Falsification test:
  fn falsify_mfc_003() {
      let qwen2 = Qwen2Family::new();
      let llama_names = ["model.layers.0.mlp.gate_proj.weight"];
      // This should pass (Qwen2 uses the same name)
      assert!(qwen2.validate_tensor_names(&llama_names, "0.5b").is_ok());

      // Whisper names should fail
      let whisper_names = ["encoder.conv1.weight"];
      assert!(qwen2.validate_tensor_names(&whisper_names, "0.5b").is_err());
  }

If test fails: Tensor name validation is not family-specific.
```

### 7.2 Phase 2: Oracle CLI

**FALSIFY-ORC-001**: Local file family detection

```
Prediction: apr oracle correctly identifies a local GGUF file's family.

Falsification test (integration):
  $ apr oracle test-fixtures/qwen2-tiny.gguf --json | jq '.family.name'
  Expected: "qwen2"

If test fails: Oracle file analysis does not match family contracts.
```

**FALSIFY-ORC-002**: HuggingFace API family detection

```
Prediction: apr oracle hf://Qwen/Qwen2.5-Coder-0.5B-Instruct identifies qwen2 family.

Falsification test (integration, requires network):
  $ apr oracle hf://Qwen/Qwen2.5-Coder-0.5B-Instruct --json | jq '.family.name'
  Expected: "qwen2"

If test fails: HuggingFace config.json parsing does not match family contracts.
```

**FALSIFY-ORC-003**: Contract description completeness

```
Prediction: apr oracle --family qwen2 --size 0.5b outputs all config values
  matching the YAML source.

Falsification test:
  $ apr oracle --family qwen2 --size 0.5b --json \
    | jq '.size_variant.hidden_dim'
  Expected: 896

If test fails: Contract description does not reflect YAML content.
```

**FALSIFY-ORC-004**: Compliance detection catches missing tensors

```
Prediction: A model file missing the lm_head tensor fails --compliance.

Falsification test:
  1. Create a GGUF with all Qwen2 tensors EXCEPT lm_head
  2. Run: apr oracle bad-model.gguf --compliance --json
  3. Assert: .compliance.is_compliant == false
  4. Assert: .compliance.missing_tensors contains "lm_head.weight"

If test fails: Compliance check does not detect missing tensors.
```

### 7.3 Phase 3: Compiler Enforcement

**FALSIFY-CMP-001**: PhantomData prevents layout mismatch

```
Prediction: Code that passes ValidatedWeight<RowMajor> to a function expecting
  a different layout type does not compile.

Falsification test (compile-fail test):
  // This code should NOT compile
  fn expects_col_major(_w: ValidatedWeight<ColumnMajor>) {}  // ColumnMajor doesn't exist
  fn falsify_cmp_001() {
      let w = ValidatedWeight::<RowMajor>::new(data, 10, 10, "test").unwrap();
      expects_col_major(w);  // ERROR: type mismatch
  }

If this compiles: PhantomData enforcement is broken.
```

**FALSIFY-CMP-002**: AprTransformer rejects unvalidated data

```
Prediction: Constructing AprTransformer with raw Vec<f32> does not compile.

Falsification test (compile-fail test):
  fn falsify_cmp_002() {
      let raw: Vec<f32> = vec![1.0; 1000];
      let t = AprTransformer {
          embedding: raw,  // ERROR: expected ValidatedEmbedding, found Vec<f32>
          ..
      };
  }

If this compiles: Type enforcement is broken.
```

**FALSIFY-CMP-003**: Clippy catches column-major import

```
Prediction: Code importing matmul_q4k_f32_colmajor fails cargo clippy.

Falsification test:
  // Add to a test file:
  use trueno::backends::q4k::matmul_q4k_f32_colmajor;
  // Run: cargo clippy -- -D warnings
  // Expected: error: use of disallowed method

If clippy passes: disallowed-methods configuration is broken.
```

### 7.4 Phase 4: Build-Time Codegen

**FALSIFY-BGN-001**: Generated code matches YAML

```
Prediction: Modifying a YAML value (e.g., changing qwen2 0.5b hidden_dim from
  896 to 897) causes the generated ModelFamily::size_config() to return 897.

Falsification test:
  1. Modify qwen2.yaml: 0.5b.hidden_dim = 897
  2. Run cargo build
  3. Assert Qwen2Family.size_config("0.5b").hidden_dim == 897
  4. Restore YAML

If test fails: build.rs code generation does not track YAML changes.
```

**FALSIFY-BGN-002**: Invalid YAML causes build failure

```
Prediction: A YAML file with missing required fields (e.g., no size_variants)
  causes cargo build to fail with a descriptive error.

Falsification test:
  1. Create contracts/model-families/bad.yaml with only "family: bad"
  2. Run cargo build
  3. Expected: build error mentioning "missing field: size_variants"
  4. Remove bad.yaml

If build succeeds: build.rs schema validation is missing.
```

### 7.5 Iteration 4: Deep Structural Falsification

**FALSIFY-ITER4-001**: Unique-naming families (BERT, Whisper) detected unambiguously

```
Prediction: BERT and Whisper use unique tensor naming conventions that differ
  from all standard "model.layers.*" families:
  - BERT: embedding = "bert.embeddings.word_embeddings.weight" (unique prefix)
  - Whisper: embedding = "encoder.conv1.weight" (unique prefix)

  detect_family() with BERT-specific tensors MUST return exactly "bert".
  detect_family() with Whisper-specific tensors MUST return exactly "whisper".

If test fails: Unique-naming families are incorrectly matched to another family.
```

**FALSIFY-ITER4-002**: Constraint consistency (activation ↔ MLP type)

```
Prediction: The activation function and MLP type must be consistent:
  - SwiGLU → SiLU (SiLU-gated linear unit)
  - GELU MLP → GELU (standard feedforward)
  - Gated MLP → GELU (GeGLU = GELU-gated linear unit, used by Gemma)

FIXED in iteration 4: gemma.yaml had activation: silu + mlp_type: gelu_mlp.
  Research confirmed Gemma uses GeGLU (GELU-gated), not SwiGLU.
  Fix: activation: gelu, mlp_type: gated_mlp.

If test fails: YAML constraint values are architecturally inconsistent.
```

**FALSIFY-ITER4-003**: head_dim validity

```
Prediction: head_dim >= hidden_dim / num_heads for all families and sizes.
  Standard: head_dim == hidden_dim / num_heads (most models).
  Override: head_dim > hidden_dim / num_heads (e.g., Gemma 7B: 256 > 192).
  head_dim < hidden_dim / num_heads would indicate a contract bug.

Note: Gemma 7B (hidden=3072, heads=16, head_dim=256) intentionally uses
  expanded attention dimensionality (16×256=4096 > 3072) for improved
  attention quality — confirmed by Google's architecture documentation.

If test fails: A YAML contract has a head_dim smaller than the standard,
  which would cause information loss in attention projections.
```

**FALSIFY-ITER4-004**: No-bias families have null bias patterns

```
Prediction: Families declaring has_bias=false should have all bias-related
  per_layer entries set to null (no tensor patterns for q_proj_bias, etc.).
  Conversely, bias families (Qwen2, Phi) must have >= 3 non-null bias patterns.

If test fails: YAML constraint/template mismatch — family declares no bias
  but has bias tensor patterns (or vice versa).
```

**FALSIFY-ITER4-005**: Cross-family tensor validation

```
Prediction: validate_tensor_names() rejects tensor names from any other
  family. Specifically: BERT tensors rejected by Whisper contract, Whisper
  tensors rejected by Qwen2 contract, Qwen2 tensors rejected by BERT contract.

If test fails: Tensor validation is not family-specific enough.
```

---

## 8. Implementation Roadmap

### 8.1 Phase 1: Foundation

| Ticket | Scope | Deliverables |
|--------|-------|-------------|
| PMAT-240 | Create model family YAMLs | `contracts/model-families/qwen2.yaml`, `llama.yaml`, `whisper.yaml`, `bert.yaml` |
| PMAT-241 | `ModelFamily` trait + `ModelFamilyConfig` | `src/format/model_family.rs` with trait, config structs, and `detect_family()` |
| PMAT-242 | YAML contract loader | Runtime YAML parser in `src/format/model_family_loader.rs` |
| PMAT-243 | Clippy column-major bans | `.clippy.toml` additions, CI verification |

**Dependencies**: None. Can begin immediately.
**Estimated effort**: 1-2 days per ticket.

### 8.2 Phase 2: Oracle CLI

| Ticket | Scope | Deliverables |
|--------|-------|-------------|
| PMAT-244 | `apr oracle` local file mode | `crates/apr-cli/src/commands/oracle.rs`, `Commands::Oracle` variant, file analysis logic |
| PMAT-245 | `apr oracle` HuggingFace API mode | HF config.json fetcher, family/size matching from API data |
| PMAT-246 | `apr oracle --family` contract mode | YAML rendering, `--size` filter, text + JSON output |
| PMAT-247 | Playbook certification cross-reference | `models.csv` parser, `CertificationInfo` struct, playbook path resolution |

**Dependencies**: PMAT-240 (YAMLs), PMAT-241 (trait), PMAT-242 (loader).
**Estimated effort**: 1-2 days per ticket.

### 8.3 Phase 3: Type Enforcement

| Ticket | Scope | Deliverables |
|--------|-------|-------------|
| PMAT-248 | `PhantomData<Layout>` on `Validated*` types | `RowMajor` marker, `ValidatedWeight<L>` generic, backward-compatible default |
| PMAT-249 | `AprTransformer` migration to `Validated*` fields | Update `AprTransformer` and `AprTransformerLayer` field types, update all constructors |

**Dependencies**: PMAT-248 must complete before PMAT-249.
**Estimated effort**: PMAT-248: 1 day. PMAT-249: 2-3 days (large refactor with many callsites).

### 8.4 Phase 4: Codegen & Generics

| Ticket | Scope | Deliverables |
|--------|-------|-------------|
| PMAT-250 | `build.rs` YAML-to-Rust code generation | `build.rs` codegen, `include!` in `model_family.rs`, `cargo:rerun-if-changed` |
| PMAT-251 | Generic `AprTransformer<F: ModelFamily>` | Parameterize `AprTransformer` by family, update realizar loader |

**Dependencies**: PMAT-240 (YAMLs), PMAT-241 (trait), PMAT-249 (Validated* fields).
**Estimated effort**: PMAT-250: 2 days. PMAT-251: 3-4 days (significant refactor).

### 8.5 Phase 5: Documentation & Expansion

| Ticket | Scope | Deliverables |
|--------|-------|-------------|
| PMAT-252 | Specification document | This document (already delivered in this PR) |
| PMAT-253 | Additional model family YAMLs | `mistral.yaml`, `phi.yaml`, `gemma.yaml`, `deepseek.yaml` |

**Dependencies**: PMAT-252: None. PMAT-253: PMAT-240 (established YAML schema).
**Estimated effort**: PMAT-253: 1 day per family.

---

## 9. Architectural Decisions

### 9.1 YAML as Source of Truth

**Decision**: Model family contracts are YAML files, not Rust code.

**Rationale**:
- **Cross-project consumption**: apr-model-qa-playbook (Python) needs to read the same
  contracts. YAML is language-agnostic.
- **Non-programmer editing**: Model researchers can update size variants without knowing
  Rust.
- **Separation of concerns**: The *what* (contract data) is separate from the *how*
  (enforcement code).
- **Precedent**: `tensor-layout-v1.yaml` already uses this pattern successfully.

**Alternative considered**: Rust `const` definitions. Rejected because they require
recompilation for every change and cannot be consumed by non-Rust tools.

### 9.2 PhantomData\<Layout\> vs Separate Types

**Decision**: Use `PhantomData<RowMajor>` on `ValidatedWeight<L>` rather than creating
separate `RowMajorWeight` and `ColumnMajorWeight` types.

**Rationale**:
- **Backward compatibility**: `ValidatedWeight` (without explicit type parameter) defaults
  to `ValidatedWeight<RowMajor>` via `L = RowMajor`. Existing code compiles unchanged.
- **Extensibility**: If column-major support is ever needed (e.g., for a GPU backend
  that prefers column-major), a `ColumnMajor` marker can be added without renaming types.
- **Zero cost**: `PhantomData` is a zero-sized type; the generic parameter adds no
  runtime overhead.

**Alternative considered**: Completely separate types (`RowMajorWeight`, `ColumnMajorWeight`).
Rejected because it would break all existing callsites and the naming is less clear about
the relationship between the types.

### 9.3 Generic AprTransformer\<F\> vs Box\<dyn ModelFamily\>

**Decision**: Use `AprTransformer<F: ModelFamily>` (monomorphization) rather than
`AprTransformer { family: Box<dyn ModelFamily> }` (dynamic dispatch).

**Rationale**:
- **Compile-time safety**: A function that expects `AprTransformer<Qwen2>` cannot
  receive `AprTransformer<Llama>`. The model family is a type-level guarantee, not
  a runtime field.
- **Performance**: Monomorphization enables inlining of family-specific code paths
  (e.g., bias handling for Qwen2 vs no-bias for LLaMA).
- **Error messages**: Compile errors reference concrete types ("expected Qwen2, found
  Llama") rather than trait object mismatches.

**Alternative considered**: `Box<dyn ModelFamily>`. This would be simpler but loses
compile-time family safety. A `dyn` approach is still used for the `detect_family()`
path where the family is not known at compile time. `detect_family()` uses
best-match scoring (counting matching per-layer tensor patterns) to disambiguate
families with overlapping naming conventions (e.g., Qwen2's bias tensors
score 13 vs LLaMA/DeepSeek/Mistral's 10). The return value is then downcast
or matched to construct the appropriate generic type.

### 9.4 New oracle Command vs Extending inspect

**Decision**: Add a new `apr oracle` subcommand rather than extending `apr inspect`.

**Rationale**:
- **Different purpose**: `inspect` shows what's *in* a file (tensors, metadata, flags).
  `oracle` answers *what* a model is (family, compliance, certification). These are
  different user questions.
- **Different sources**: `inspect` works on local files only. `oracle` also works on
  HuggingFace URIs and family names (no file required).
- **Namespace cleanliness**: `inspect` already has `--vocab`, `--filters`, `--weights`.
  Adding `--family`, `--compliance`, `--certification` would overload it.
- **Precedent**: `apr qa` (quality assurance) was a new command, not an extension of
  `apr validate`. The oracle/inspect split follows the same pattern.

**Alternative considered**: `apr inspect --oracle`. Rejected because the `hf://` and
`--family` modes have no file to inspect — the command name would be misleading.

---

## 10. References

1. Shingo, S. (1986). *Zero Quality Control: Source Inspection and the Poka-Yoke
   System*. Productivity Press. ISBN 0-915299-07-0.

2. Ohno, T. (1988). *Toyota Production System: Beyond Large-Scale Production*.
   Productivity Press. ISBN 0-915299-14-3.

3. Popper, K. (1959). *The Logic of Scientific Discovery*. Hutchinson & Co.
   ISBN 0-415-27844-9.

4. Brady, E. (2017). *Type-Driven Development with Idris*. Manning Publications.
   ISBN 978-1-61729-302-5.

5. Parsons, A. (2019). "Parse, Don't Validate." Blog post.
   https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/

6. Strom, R. E. & Yemini, S. (1986). "Typestate: A Programming Language Concept for
   Enhancing Software Reliability." *IEEE Transactions on Software Engineering*,
   SE-12(1), pp. 157-171.

7. Clebsch, S., Drossopoulou, S., Blessing, S., & McNeil, A. (2015). "Deny
   Capabilities for Safe, Fast Actors." In *Proceedings of the 5th International
   Workshop on Programming Based on Actors, Agents, and Decentralized Control
   (AGERE!)*. ACM. pp. 1-12.

---

## Appendix A: PMAT Ticket Descriptions

| Ticket | Phase | Title | Description |
|--------|-------|-------|-------------|
| PMAT-240 | 1 | Create model family YAMLs | Write `contracts/model-families/qwen2.yaml`, `llama.yaml`, `whisper.yaml`, and `bert.yaml` following the schema in Section 4.2. Each YAML declares size variants, constraints, tensor templates, shape templates, and chat template configurations. Validate with `python3 -c "import yaml; yaml.safe_load(open('file'))"`. |
| PMAT-241 | 1 | `ModelFamily` trait + `ModelFamilyConfig` | Create `src/format/model_family.rs` with the `ModelFamily` trait, `ModelFamilyConfig`, `ModelSizeConfig`, `ModelConstraints`, and associated enums (`AttentionType`, `Activation`, `NormType`, `PositionalEncoding`, `MlpType`). Include `detect_family()` function that takes a list of tensor names and returns the matching `ModelFamily` implementation. |
| PMAT-242 | 1 | YAML contract loader | Create `src/format/model_family_loader.rs` that parses model family YAML files at runtime. Uses minimal YAML parsing (no serde_yaml dependency — manual parsing or lightweight parser). Returns `ModelFamilyConfig` structs. This is the runtime fallback; `build.rs` codegen (PMAT-250) is the preferred path. |
| PMAT-243 | 1 | Clippy column-major bans | Add `disallowed-methods` entries to `.clippy.toml` for all column-major matmul kernel functions. Verify in CI that `cargo clippy -- -D warnings` catches any column-major imports. Add a test that confirms the ban works by checking clippy output on a known-bad file. |
| PMAT-244 | 2 | `apr oracle` local file mode | Add `Commands::Oracle` to `crates/apr-cli/src/lib.rs`. Implement local file analysis: open with `RosettaStone::inspect()`, match tensor names against family contracts, detect size variant, output `ModelOracleReport` in text and JSON. Include `--compliance` and `--tensors` flags. |
| PMAT-245 | 2 | `apr oracle` HuggingFace API mode | Implement HF API query for `config.json`. Parse `model_type`, `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads` from config. Match against family contracts. Handle gated models (401), rate limits (429), and offline mode. |
| PMAT-246 | 2 | `apr oracle --family` contract mode | Implement contract description rendering. Load YAML, display all size variants or filter with `--size`. Show constraints, tensor templates with evaluated shapes, and chat template. Text and JSON output formats. |
| PMAT-247 | 2 | Playbook certification cross-reference | Parse `../apr-model-qa-playbook/docs/certifications/models.csv`. Map detected model to certification entry using `csv_family_key` and size variant. Populate `CertificationInfo` in `ModelOracleReport`. Handle missing playbook repo gracefully. |
| PMAT-248 | 3 | `PhantomData<Layout>` on `Validated*` types | Add `RowMajor` marker type. Make `ValidatedWeight` generic: `ValidatedWeight<L = RowMajor>`. Existing code uses the default and compiles unchanged. Add `PhantomData<L>` field. Update `ValidatedEmbedding` similarly. |
| PMAT-249 | 3 | `AprTransformer` migration to `Validated*` fields | Change `AprTransformer` and `AprTransformerLayer` fields from `Vec<f32>` to `ValidatedEmbedding`, `ValidatedWeight<RowMajor>`, and `ValidatedVector`. Update all construction sites in aprender and realizar. This is a large refactor — track callsite count before and after. |
| PMAT-250 | 4 | `build.rs` YAML-to-Rust code generation | Write `build.rs` that reads `contracts/model-families/*.yaml`, generates `ModelFamily` trait implementations, and writes to `$OUT_DIR/model_families_generated.rs`. Include `cargo:rerun-if-changed` directives. Validate YAML schema during build. |
| PMAT-251 | 4 | Generic `AprTransformer<F: ModelFamily>` | Parameterize `AprTransformer<F>` by model family. Update realizar's model loading to construct `AprTransformer<Qwen2>` or `AprTransformer<Llama>` based on detected family. Maintain backward compatibility via type alias `AprTransformerDyn = AprTransformer<DynFamily>`. |
| PMAT-252 | 5 | Specification document | This document. |
| PMAT-253 | 5 | Additional model family YAMLs | Create `contracts/model-families/mistral.yaml`, `phi.yaml`, `gemma.yaml`, `deepseek.yaml`. Follow the schema established in PMAT-240. Source config values from HuggingFace model cards and config.json files. |
