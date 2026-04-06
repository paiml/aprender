# pv — Provable Contracts Specification v2.9.9

**Papers to Math to Contracts in Code.**

A Rust library and CLI for converting peer-reviewed research into
mathematically provable kernel implementations via YAML contracts with
Kani bounded model checking and Lean 4 theorem proving.

**Canonical spec.** This is the ONE spec. All other docs are deprecated.
Sub-specs live in `docs/specifications/sub/` and are linked from this TOC.

---

## Table of Contents

| # | Section | Sub-spec |
|---|---------|----------|
| 1 | [Vision and Architecture](#1-vision-and-architecture) | [sub/vision.md](sub/vision.md) |
| 2 | [The Verification Ladder](#2-the-verification-ladder) | [sub/verification-ladder.md](sub/verification-ladder.md) |
| 3 | [Contract Schema](#3-contract-schema) | [sub/schema.md](sub/schema.md), [sub/eiffel-dbc.md](sub/eiffel-dbc.md) |
| 4 | [The Seven-Phase Pipeline](#4-the-seven-phase-pipeline) | [sub/pipeline.md](sub/pipeline.md), [sub/pytorch-extraction.md](sub/pytorch-extraction.md) |
| 5 | [CLI Reference (`pv`)](#5-cli-reference) | [sub/cli.md](sub/cli.md), [sub/lint.md](sub/lint.md) |
| 6 | [Library API](#6-library-api) | [sub/library.md](sub/library.md) |
| 7 | [Scoring System (`pv score`)](#7-scoring-system) | [sub/scoring.md](sub/scoring.md) |
| 8 | [Query Engine (`pv query`)](#8-query-engine) | [sub/query.md](sub/query.md) |
| 9 | [Proc Macro (`#[contract]`)](#9-proc-macro) | [sub/proc-macro.md](sub/proc-macro.md) |
| 10 | [Kernel Contract Registry](#10-kernel-contract-registry) | [sub/registry.md](sub/registry.md) |
| 11 | [Stack Integration](#11-stack-integration) | [sub/integration.md](sub/integration.md) |
| 12 | [References](#12-references) | [sub/references.md](sub/references.md) |
| 13 | [Escape-Proof Enforcement](#13-escape-proof-enforcement) | [sub/escape-proof-enforcement.md](sub/escape-proof-enforcement.md) |
| 14 | [Lean 4 + Kani Composition](#14-lean-kani-composition) | [sub/lean-kani-composition.md](sub/lean-kani-composition.md) |
| 15 | [Verification Extensions](#15-verification-extensions) | [sub/verification-extensions.md](sub/verification-extensions.md) |
| 16 | [Bidirectional Coverage](#16-bidirectional-coverage) | [sub/bidirectional-coverage.md](sub/bidirectional-coverage.md) |
| 17 | [Gradual Enforcement](#17-gradual-enforcement) | [sub/gradual-enforcement.md](sub/gradual-enforcement.md) |
| 18 | [PVScore](#18-pvscore) | [sub/pvscore.md](sub/pvscore.md) |
| 19 | [Sovereign Stack Audit](#19-sovereign-stack-audit) | [sub/sovereign-stack-audit.md](sub/sovereign-stack-audit.md) |
| 20 | [UX, Speech, Probar](#20-ux-speech-probar) | [sub/ux-speech-probar.md](sub/ux-speech-probar.md) |
| 21 | [Contract Gap Analysis](#21-contract-gap-analysis) | [sub/contract-gaps.md](sub/contract-gaps.md) |
| 22 | [Diagnostic Output](#22-diagnostic-output) | [sub/diagnostics.md](sub/diagnostics.md) |
| 23 | [Contract-Trait Enforcement](#23-contract-trait-enforcement) | [sub/contract-trait-enforcement.md](sub/contract-trait-enforcement.md) |
| 24 | [Deep Stack Integration](#24-deep-stack-integration) | [sub/deep-integration.md](sub/deep-integration.md) |
| 25 | [Full Enforcement Mandate](#25-full-enforcement-mandate) | [sub/full-enforcement.md](sub/full-enforcement.md) |
| 26 | [Two-Tier Architecture](#26-two-tier-architecture-and-compositional-contracts) | [sub/two-tier-architecture.md](sub/two-tier-architecture.md) |
| 27 | [The One Way](#27-the-one-way) | [sub/the-one-way.md](sub/the-one-way.md) |
| 28 | [Correctness + Completeness](#28-correctness--completeness) | [sub/correctness-completeness.md](sub/correctness-completeness.md) |
| 29 | [Asset Contracts](#29-asset-contracts) | [sub/asset-contracts.md](sub/asset-contracts.md) |
| 30 | [Real Contract Enforcement](#30-real-contract-enforcement) | [sub/real-contract-enforcement.md](sub/real-contract-enforcement.md) |
| 31 | [Kaizen Fleet Enforcement](#31-kaizen-fleet-enforcement) | [sub/kaizen-fleet-enforcement.md](sub/kaizen-fleet-enforcement.md) |
| 32 | [PMAT Infrastructure Contracts](#32-pmat-infrastructure-contracts) | [sub/pmat-infrastructure-contracts.md](sub/pmat-infrastructure-contracts.md) |
| 33 | [Forjar Heavy Types Contracts](#33-forjar-heavy-types-contracts) | [sub/forjar-heavy-types-contracts.md](sub/forjar-heavy-types-contracts.md) |
| 34 | [Systems Contract Patterns](#34-systems-contract-patterns) | [sub/systems-contract-patterns.md](sub/systems-contract-patterns.md) |
| 35 | [Commit-Level Entity Enforcement](#35-commit-level-entity-enforcement) | [sub/commit-level-entity-enforcement.md](sub/commit-level-entity-enforcement.md) |
| 36 | [Model Layout Provability — P0](#36-model-layout-provability--p0-defect) | [sub/model-layout-provability.md](sub/model-layout-provability.md) |
| 37 | [Aprender Contract Suite](#37-aprender-contract-suite) | inline |

---

## 1. Vision and Architecture

**Sub-spec**: [sub/vision.md](sub/vision.md)

Make the paper→code derivation chain explicit, auditable, and provable.
303 scored YAML contracts (309 total by kind: 273 kernel, 31 registry, 5 pattern),
1430 proof obligations, 1584 falsification tests, 1521 Kani harnesses, 35 Lean
proofs, 1083 bindings, 527 call sites (78.3% penetration) across 40 repos
governing ~6.4M Rust LoC. Fleet enforcement: Grade B (0.44).
Trueno SIMD/PTX: 11 contracts (C, 0.67) covering NEON, AVX2, AVX-512, PTX, Q4K/Q6K/NF4.
Composition gate: 13 edges, 0 broken (blocking since PMAT-487).

Architecture: `crates/provable-contracts/` (library), `crates/provable-contracts-cli/`
(CLI binary `pv`), `contracts/` (YAML registry), `crates/provable-contracts-macros/`
(proc macro). Foundations: Popper falsificationism, Meyer DbC, Kani BMC, Lean 4.

---

## 2. The Verification Ladder

**Sub-spec**: [sub/verification-ladder.md](sub/verification-ladder.md)

Two hierarchies: proof levels (L0 review → L5 Lean theorem) and
enforcement layers (build.rs → trait impl → `#[contract]` → Kani → Lean).
L0–L2 enforce on every build in 7 repos. L3 on 18 annotated functions.
L4/L5 defined in YAML, not yet in CI.

The provability invariant: if a contract has proof obligations, it MUST
have Kani harnesses and falsification tests. Registries (`metadata.registry: true`) exempt.

---

## 3. Contract Schema

**Sub-spec**: [sub/schema.md](sub/schema.md), [sub/eiffel-dbc.md](sub/eiffel-dbc.md)

Every YAML contract follows a fixed schema: `metadata`, `equations`,
`proof_obligations`, `kernel_structure`, `simd_dispatch`, `enforcement`,
`falsification_tests`, `kani_harnesses`, `verification_summary`, `qa_gate`.

26 proof obligation types (19 property + 7 Eiffel DbC). Three expression
languages: Rust expressions (default), regex patterns, refinement types.

**Contract Kinds (`metadata.kind`).** Not every artifact in `contracts/` is
a mathematical kernel. The `kind` field declares which validation rules
apply — kernels get the full provability invariant; other kinds are
first-class but exempt from kernel-specific checks.

| Kind            | Provability | Typical use                                           |
|-----------------|-------------|-------------------------------------------------------|
| `kernel` (default) | required | mathematical kernel contract (softmax, attention, …) |
| `registry`      | exempt      | lookup tables, enum definitions, config bounds       |
| `model-family`  | exempt      | HuggingFace architecture metadata, size variants     |
| `pattern`       | exempt      | cross-cutting verification patterns (threading, async) |
| `schema`        | exempt      | generic reference/schema documents                   |

Non-kernel kinds skip `PROVABILITY-001`, `SCHEMA-003` (empty equations),
`AUDIT-001` (no falsification tests), and the `enforce` / `enforcement-level`
lint gates. They still validate `metadata` (version + description + references)
and any proof/kani/falsification data that IS present. The legacy
`metadata.registry: true` flag is preserved for back-compat and maps to
`kind: registry`.

---

## 4. The Seven-Phase Pipeline

**Sub-spec**: [sub/pipeline.md](sub/pipeline.md), [sub/pytorch-extraction.md](sub/pytorch-extraction.md)

Extract → Specify → Scaffold → Implement → Falsify → Verify → Prove.
Every phase produces a committed artifact in version control.

---

## 5. CLI Reference

**Sub-spec**: [sub/cli.md](sub/cli.md), [sub/lint.md](sub/lint.md)

The `pv` binary provides 34 commands: `validate`, `scaffold`, `kani`,
`probar`, `generate`, `lint`, `score`, `query`, `codegen`, `kaizen`,
`coverage`, `graph`, `lean`, `roofline`, and more.

---

## 6. Library API

**Sub-spec**: [sub/library.md](sub/library.md)

The `provable-contracts` crate exposes 35 public modules: `schema`
(parse/validate), `scaffold`/`kani_gen`/`probar_gen`/`lean_gen`/`codegen` (code generation),
`scoring` (5-dim contract + codebase), `query` (BM25 search), `binding`,
`coverage`, `graph`, `diff`, `audit`, `proof_status`.

---

## 7. Scoring System

**Sub-spec**: [sub/scoring.md](sub/scoring.md)

`pv score` — 5-dim contract scoring (spec_depth 20%, falsification 25%,
kani 25%, lean 10%, binding 20%) and 5-dim codebase scoring (coverage 30%,
critical_path 20%, mean_score 20%, proof_depth 15%, drift 15%).
Grades: A >= 0.90, B >= 0.75, C >= 0.60, D >= 0.40, F < 0.40.

---

## 8. Query Engine

**Sub-spec**: [sub/query.md](sub/query.md)

`pv query` provides O(1) BM25 semantic search across 315+ contracts and
consumer projects. Modes: semantic, regex, literal. Filters: `--obligation`,
`--depends-on`, `--unproven`. Enrichment: `--score`, `--graph`, `--paper`.

---

## 9. Proc Macro

**Sub-spec**: [sub/proc-macro.md](sub/proc-macro.md)

The `#[contract]` attribute from `provable-contracts-macros` provides
compile-time contract enforcement. Four-layer model: L1 build.rs
AllImplemented, L2 trait `impl`, L3 `#[contract]` debug_assert injection,
L4 `pv lint --reverse` for pub fn coverage.

---

## 10. Kernel Contract Registry

**Sub-spec**: [sub/registry.md](sub/registry.md)

271+ scored contracts organized by domain: ML kernels (softmax, matmul,
rmsnorm, attention, rope, swiglu, etc.), quantization, tokenization,
inference pipeline, training, serving, and data registries.

---

## 11. Stack Integration

**Sub-spec**: [sub/integration.md](sub/integration.md)

Sovereign stack integration: binding.yaml per-repo, build.rs enforcement,
`#[contract]` proc macro annotations, `pv codegen` generated macros.
7 repos with build.rs, 18 with codegen macros, 40 in kaizen fleet.

---

## 12. References

**Sub-spec**: [sub/references.md](sub/references.md)

33 references across methodology (Popper, Meyer, Brady), ML kernels
(Vaswani, Dao, Su), formal verification (Kani, Lean, Creusot),
quality gates (SARIF, SonarQube), and gradual typing (Siek, Bader, Lehmann).

---

## 13. Escape-Proof Enforcement

**Sub-spec**: [sub/escape-proof-enforcement.md](sub/escape-proof-enforcement.md)

Six-stage pipeline where each stage gates the next. Skip one → compile error.
Equation (YAML) → Lean 4 proof (no sorry) → YAML validation (pv lint) →
build.rs codegen (debug_assert from preconditions/postconditions/invariants) → #[contract] macro
(compile-time binding check) → test execution (falsification tests pass).

---

## 14. Lean 4 + Kani Composition

**Sub-spec**: [sub/lean-kani-composition.md](sub/lean-kani-composition.md)

Lean and Kani are NOT alternatives — they verify different things about
the SAME obligation. Lean proves the algorithm over R. Kani proves the
Rust code over f32. The `stub_float` strategy bridges them compositionally.

---

## 15. Verification Extensions

**Sub-spec**: [sub/verification-extensions.md](sub/verification-extensions.md)

Six orthogonal verification approaches: Type Invariants, Coq, Coverage-Guided
Fuzzing, Abstract Interpretation (MIRAI), Refinement Types (Flux), TLA+.

---

## 16. Bidirectional Coverage

**Sub-spec**: [sub/bidirectional-coverage.md](sub/bidirectional-coverage.md)

Reverse coverage: `pv coverage --reverse` (static API diff), `#[must_contract]`
(compile-time lint), `pv infer` (semantic matching). `pv lint` Gate 7 enforces
reverse coverage threshold.

---

## 17. Gradual Enforcement

**Sub-spec**: [sub/gradual-enforcement.md](sub/gradual-enforcement.md)

Five enforcement gaps addressed: per-contract enforcement levels, stale
suppression detection, multi-stage pipeline, aggregate coverage metric,
irreversible level lock. All implemented. Pattern: mypy, TypeScript, Rust `#[forbid]`.

---

## 18. PVScore (`pv score .`)

**Sub-spec**: [sub/pvscore.md](sub/pvscore.md)

10-dimension project scoring (0-100 geometric mean): spec depth,
falsification, Kani BMC, Lean 4, binding, reverse coverage, mutation
testing, CI depth, proof freshness, defect patterns. Grade A (90+) for CI merge.

---

## 19. Sovereign Stack Audit

**Sub-spec**: [sub/sovereign-stack-audit.md](sub/sovereign-stack-audit.md)

Full audit of 13 repos. 6.4M LOC under contract enforcement. 16,989 bindings.
Zero unenforced repos. The enforcer (pmat) enforces itself.

---

## 20. UX, Speech, Probar

**Sub-spec**: [sub/ux-speech-probar.md](sub/ux-speech-probar.md)

Four UX contract categories: geometric invariants, perceptual correctness,
pipeline correctness, visual regression. Whisper.apr contracts. probar
integration as PVScore D2 data source. MQS model quality scoring.

---

## 21. Contract Gap Analysis

**Sub-spec**: [sub/contract-gaps.md](sub/contract-gaps.md)

9 ML/systems domains analyzed. Major gaps: training infrastructure,
memory management, tokenization, post-training/alignment. Top 5 additions:
speculative decoding, FP8, DPO loss, BPE tokenization, PagedAttention.

---

## 22. Diagnostic Output

**Sub-spec**: [sub/diagnostics.md](sub/diagnostics.md)

Falsification against 9 reference tools revealed 13 gaps. P0 implemented:
grouped finding display, color terminal output, `pv lint --explain`.

---

## 23. Contract-Trait Enforcement

**Sub-spec**: [sub/contract-trait-enforcement.md](sub/contract-trait-enforcement.md)

Generate Rust traits from YAML contracts. Consumer crates `impl` them.
Missing function = compile error. Wrong signature = compile error. No
build.rs, no scanning. Validated by SPARK/Ada, Eiffel, Kani, Prusti, Creusot.

---

## 24. Deep Stack Integration

**Sub-spec**: [sub/deep-integration.md](sub/deep-integration.md)

Make contracts first-class in inference, profiling, and quality pipelines.
Three-tier: Compile (YAML → build.rs → macro → trait), CI (lint → verify →
comply), Runtime (roofline from YAML → BrickProfiler → postcondition checks).

---

## 25. Full Enforcement Mandate

**Sub-spec**: [sub/full-enforcement.md](sub/full-enforcement.md)

Every repo MUST achieve Grade A (`pv score --min-score 0.90`). Requirements:
real bindings with `module_path`, `pv verify-bindings`, build.rs, trait tests,
zero-warning lint. Scoring: declared/resolved coverage (no ghost inflation).

---

## 26. Two-Tier Architecture and Compositional Contracts

**Sub-spec**: [sub/two-tier-architecture.md](sub/two-tier-architecture.md)

Tier 1: generic kernel contracts (algorithm math). Tier 2: per-library
bindings + library-specific contracts. Three composition levels: intra-contract
(solved), cross-contract pipeline (partial), cross-repo pipeline (not yet).
Pipeline binding files with `stages` + `cross_boundary_obligations` proposed.

---

## 27. The One Way

**Sub-spec**: [sub/the-one-way.md](sub/the-one-way.md)

One mechanism: `pv codegen --binding` generates `debug_assert!()` from YAML
pre/postconditions with real parameter names. Transition: Phase 0 (debug_assert),
Phase 1 (nightly `#[core::contracts]`), Phase 2 (stable contracts, delete build.rs).

---

## 28. Correctness + Completeness

**Sub-spec**: [sub/correctness-completeness.md](sub/correctness-completeness.md)

Correctness (contracts are right) + completeness (everything has a contract).
CD2: developer-declared `critical_path` in binding.yaml. Converged after 4
rounds of falsification that killed 3 alternative designs.

---

## 29. Asset Contracts

**Sub-spec**: [sub/asset-contracts.md](sub/asset-contracts.md)

Contracts for data assets (model weights, tokenizers, configs, media).
Three types: schema (format), shape (dimensions), value (numeric health).
`pv verify-asset` proposed but NOT yet implemented. Closes the last gap
in the load→kernel→output verification chain.

---

## 30. Real Contract Enforcement

**Sub-spec**: [sub/real-contract-enforcement.md](sub/real-contract-enforcement.md)

v2.3.0 falsification: all 530 macros had identical `!is_empty()` body.
Three-layer fix: emit real YAML preconditions, add postcondition codegen,
E0/E1/E2 enforcement quality metric. 9 core kernel contracts prioritized.

---

## 31. Kaizen Fleet Enforcement

**Sub-spec**: [sub/kaizen-fleet-enforcement.md](sub/kaizen-fleet-enforcement.md)

Continuous improvement across 40-repo fleet. Five phases: measure, codegen,
inject, validate, report. Tiered grading: kernel tier (E2 quality) vs tool
tier (penetration). v2.9.9: 725 bindings, **1107 call sites**, 20,110 assertions,
**Grade A fleet**, **Kernel Grade A** (174 postconditions, 315 preconditions),
Tool Grade A (116.3%). 294 contracts, 1025 Lean theorems.
Entrenar: **Grade A** (83 sites, 28 post). Realizar: **Grade A** (128 sites, 40 post).
Trueno: **Grade A** (62 sites, 27 post). Aprender: **Grade A** (216 sites, 79 post).

**PMAT-495 postcondition sweep** (2026-04-06, session 2):
- Realizar: +42 postconditions (39 new + softmax per-row fix + fused_q4k_dot guard)
- Trueno: +21 postconditions (dequant, elementwise, amdahl, attention, matmul)
- Entrenar: +5 postconditions (layernorm, rope, swiglu, rmsnorm batched)
- Aprender: +127 call sites (E0→E1 bulk upgrade + softmax per-row fix)
- Fleet: **489 total call sites** across 4 kernel crates (315 pre + 174 post)
- 31 compilable postcondition macros with `debug_assert!` bodies (up from 20)
- 10 trueno contract YAMLs enriched with domain/codomain/postconditions
- Bug fixes: softmax per-row postcondition (realizar + aprender),
  fused_q4k_dot finite-scale guard, NF4 scalar postcondition

**PMAT-495 sweep results** (2026-04-06, session 1):
- Kernel-tier: **entrenar C→B** (20→58 sites), **realizar D→B** (16→93 sites),
  **aprender D→C** (55→136 sites), **trueno C→B** (21→44 sites, 100% pen)
- Tool-tier: rurl F→D (22 sites), duende 0→20 sites, probar F→D (13 sites)
- Fleet: **+274 call sites**, +52 bindings, penetration 78.6%→**110.8%**
- Kernel tier upgraded **D→B** in a single session
- Falsification: FALSIFY-GPU-008/009 (run/serve GPU parity, rosetta exit code)
- 3 new contracts: apr-cli-mutating-v1, apr-cli-readonly-v1, apr-cli-longrunning-v1

*Tool tier (Grade A, 116% pen — maintenance mode):*
- Remaining F-grades: apr-model-qa-playbook, batuta, pmat, pmcp, faro,
  rclean, zenith, copia, duende

---

## 32. PMAT Infrastructure Contracts

**Sub-spec**: [sub/pmat-infrastructure-contracts.md](sub/pmat-infrastructure-contracts.md)

Fifteen contracts covering CLI/HTTP, MCP protocol, Graph/Index, concurrency,
tracing, memory, state machines, configuration, compression, TDG/composite
scoring, context generation, comply-check, work-DBC lifecycle and claims.
All 51 equations have pre+postconditions (0 lint warnings).

---

## 33. Forjar Heavy Types Contracts

**Sub-spec**: [sub/forjar-heavy-types-contracts.md](sub/forjar-heavy-types-contracts.md)

Eight contracts: content-addressed store, OCI manifests, task/pipeline,
event/rulebook, plugin lifecycle, secret providers, Copia delta sync,
sandbox isolation. Brings forjar from 5 to 13 contracts.

---

## 34. Systems Contract Patterns

**Sub-spec**: [sub/systems-contract-patterns.md](sub/systems-contract-patterns.md)

19 reusable patterns: threading, async, compute dispatch, memory lifecycle,
LLM architecture. Each maps to §3 proof obligation types with ULP tolerances.

---

## 35. Commit-Level Entity Enforcement

**Sub-spec**: [sub/commit-level-entity-enforcement.md](sub/commit-level-entity-enforcement.md)

How PMAT enforces provable-contracts at `git commit` time. 26 CB checks
(CB-1320..1354) across 8 phases validate three entity types — code
entities (function bindings), work entities (active work items), and
asset entities (README, Dockerfile, SVG, CHANGELOG, mdBook, forjar.yaml)
— all from O(1) cached data in < 45ms. Key mechanisms: verification
level monotonicity ratchet (L-levels never decrease), differential
obligation verification (only re-check modified bindings), and
assume-guarantee chains for concurrent work items (Pacti-style A/G
dependency DAGs with DFS cycle detection).

Asset layout enforcement follows the **grid protocol paradigm** from
rmedia: content never exists without a placement contract. Documents
are 1-column slot grids; SVGs are 2D cell grids. Layout is a constraint
satisfaction problem — SMT-Layout (arXiv 2411.12271) encodes it as
MaxSMT; ORCSolver (arXiv 2002.09925) unifies grid/flow under single CSP.
Five planned improvements: AST-based markdown parsing (replace substring
matching), accuracy cross-referencing (stale README metrics), verdict
caching with content hashing (honest O(1)), recursive link validation,
and YAML-driven declarative asset contracts.

---

## 36. Model Layout Provability — P0 DEFECT

**Sub-spec**: [sub/model-layout-provability.md](sub/model-layout-provability.md)

**P0 defect.** 11 contracts cover model architecture (config algebra,
arch constraints, tensor names, shapes, format safety) — each individually
proven (L3 Kani/Lean). But **0 of 11 compose**: no mechanism verifies
that one contract's postconditions satisfy the next's preconditions.
The chain `config.json → shapes → kernels → output` is 11 isolated
proofs with no bridges. Root cause: `Equation` lacks typed `assumes` /
`guarantees` fields; `depends_on` is advisory. Fix: add `ShapeContract`
to `Equation`, new COMPOSITION-001 lint gate that unifies shape types
across `DependencyGraph` edges, `pv verify-pipeline` for end-to-end
compositional proof, `pv verify-structure` for config-to-weight
structural verification, `pv certify` for whole-model proof certificates.
8-step P0 implementation plan with falsification checks. Grounded in
Scalify (verified Llama-405B graph), TensorGuard (527 checker bugs in
TF/PyTorch), ScenicProver (compositional A/G with Lean 4).

**Status (PMAT-487):** Implemented. `ShapeContract` type with `assumes`/
`guarantees` fields on `Equation`. COMPOSITION-001 lint gate blocking
(13 edges, 0 broken). Guarantees on softmax, attention, rmsnorm, silu,
swiglu, layernorm, gelu, rope kernel contracts.

---

## 37. Aprender Contract Suite

27 contracts governing the aprender ML library and CLI, covering:

**CLI Layer** (9 contracts):
- `apr-cli-v1`: command parsing, contract gate, training plan/apply, stdin pipe
- `apr-cli-operations-v1`: side effects, resource management, inference ops
- `apr-cli-sampling-v1`: temperature, top-k/p, seed determinism, repeat penalty
- `cli-dispatch-v1`: exit codes, command routing, error mapping
- `apr-chat-session-v1`: multi-turn chat, context management
- `apr-data-pipeline-v1`: data loading, splitting, audit
- `apr-cli-mutating-v1`: output-path validation, exit-code postconditions,
  atomic write safety, rm confirmation gate (GH-689, 16 commands)
- `apr-cli-readonly-v1`: no-side-effects, idempotent output, exit-code
  postconditions (GH-688, 28 commands)
- `apr-cli-longrunning-v1`: graceful shutdown, resource cleanup, concurrent
  isolation (GH-690, run/serve/chat/tui)

**HTTP/Serve Layer** (2 contracts):
- `apr-serve-v1`: server lifecycle, request routing, CORS, error sanitization,
  GPU token integrity, max_tokens bound, concurrent isolation
- `http-api-v1`: OpenAI-compatible request/response schemas, error envelope

**Model Format Layer** (5 contracts):
- `apr-format-safety-v1`: magic bytes, header integrity, truncation, dtype coercion,
  validate exit codes, flag integrity, metadata completeness
- `model-format-conversion-v1`: roundtrip correctness, quantization bounds
- `apr-model-lifecycle-v1`: load/save/validate lifecycle
- `format-parity-v1`: GGUF/APR/SafeTensors transpose involution, element count,
  tensor name bijection
- `encoder-forward-v1`: BERT encoder layer, CLS pooling (GH-326)

**Architecture/Inference Layer** (6 contracts):
- `apr-architecture-schema-v1`: config invariants, attention/FFN/norm/embedding shapes,
  RoPE, tensor count, oracle detection, layer count, tensor name recognition
- `apr-gpu-backend-v1`: backend selection, GPU detection, temperature-zero,
  GPU/CPU parity, JSON output consistency
- `apr-finetune-v1`: LoRA rank bounds, VRAM safety, merge shape, checkpoint roundtrip
- `kernel-fusion-v1`: fused kernel correctness
- `layer-parity-v1`: CPU/GPU layer output equivalence
- `bidirectional-attention-v1`: full attention matrix for BERT-class models

**MCP/Tool Layer** (1 contract):
- `mcp-tool-schema-v1`: tool registration, schema fidelity, session lifecycle

**Training Layer** (4 contracts):
- `training-loop-v1`, `batch-training-v1`, `tokenizer-loading-v1`,
  `qwen2-weight-loading-v1`

**Other** (3 contracts):
- `apr-model-qa-v1`, `quantized-dot-product-v1`, `tensor-layout-v1`

**Fleet enforcement (aprender):** 125 bindings, **216 call sites**, 172.8% penetration,
**Grade A**. v2.9.9 added 127 call sites (E0→E1 bulk upgrade + postcondition fixes).
79 postcondition call sites, 137 precondition call sites.
Codegen: 294 contracts, 1025 Lean theorems, 31 compilable postcondition macros.

**Addressed contract gap tickets** (PMAT-495):
- GH-688: apr-cli-readonly-v1 — no-side-effects, idempotent output (28 commands)
- GH-689: apr-cli-mutating-v1 — output-path, exit-code, atomic write (16 commands)
- GH-690: apr-cli-longrunning-v1 — graceful shutdown, resource cleanup (4 commands)
- GH-326: encoder-forward-v1, bidirectional-attention-v1 — BERT inference
- Model format: format-parity-v1 — GGUF/APR/SafeTensors tensor parity
- CLI: cli-dispatch-v1, http-api-v1, mcp-tool-schema-v1 bound and injected
- Sampling: apr-cli-sampling-v1 — temperature, top-k/p, seed determinism
- GPU: apr-gpu-backend-v1 — backend selection, GPU/CPU parity
- Finetune: apr-finetune-v1 — LoRA rank bounds, VRAM feasibility
- Tokenizer: tokenizer-loading-v1 — roundtrip encoding, byte encoder
- Architecture: qwen2-weight-loading-v1 — Q/KV projection, SwiGLU expansion

**Remaining work** (maintenance mode):
- GH-686: Per-function `#[contract]` proc macro annotations (Level A)
- GH-687: L5 Lean proofs for 4 work contracts
- GH-691: Per-crate penetration reporting (apr-cli vs aprender lib)
- GH-367: InternLM2.5 architecture — fused QKV tensor naming

---

## 38. Document and Asset Integrity

**Contract**: `document-integrity-v1.yaml`

Mathematical enforcement of document and asset file structure. All
invariants are decidable properties on finite byte sequences — no
approximation, no heuristics, no ML. Pure structural validation.

**Markdown (.md)** — 9 equations:
- `heading_hierarchy`: DAG property — h₁=1, ∀i: hᵢ ≤ hᵢ₋₁+1, |{h=1}|=1
- `link_wellformedness`: ∀ link: url.len()>0 ∧ ¬starts_with("javascript:")
- `code_fence_language`: ∀ fence: lang.len()>0 (no bare ```)
- `table_column_parity`: ∀ rows r in table: |r| = |header|
- `required_sections`: configurable required heading set
- `readme_drift`: byte-level comparison actual vs generate_readme()
- `yaml_frontmatter`: YAML front matter parses as valid YAML
- `badge_format`: alt text non-empty, URL well-formed

**SVG (.svg)** — 1 equation:
- `svg_structural_safety`: valid XML, viewBox present, no `<script>`,
  no `<foreignObject>`, correct namespace, bounded dimensions

**YAML (.yaml/.yml)** — 2 equations:
- `yaml_structural_validity`: parses, no duplicate keys, depth ≤ 20
- `yaml_key_convention`: keys match /^[a-z][a-z0-9_-]*$/

**Media assets** — 3 equations:
- `media_magic_bytes`: file magic matches extension (PNG/JPEG/GIF/MP4/WebM/WAV/MP3)
- `media_metadata_present`: width/height/fps/codec/sample_rate non-zero
- `media_dimension_bounds`: 1≤w≤8192, 1≤h≤8192, fps≤240, size≤100MB

**Animation (GIF/APNG/Lottie)** — 1 equation:
- `animation_bounds`: frame_count≤1000, duration≤60s, no infinite loops

15 falsification tests, 6 Kani harnesses, 10 proof obligations.
Implementation: `pv lint --docs` validates all .md/.svg/.yaml/media files
in a project tree. README drift via `pv lint --readme-drift`.
