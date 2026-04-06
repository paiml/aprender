# Introduction

> **"Batuta orchestrates sovereign AI infrastructure — autonomous agents, ML serving, code analysis, and transpilation pipelines in pure Rust."**

## Welcome to The Batuta Book

This book is your comprehensive guide to **Batuta**, the orchestration framework for the Sovereign AI Stack. Batuta provides autonomous agent runtimes, ML model serving, proactive bug hunting, and transpilation pipelines that convert Python/C/Shell to Rust with semantic preservation.

The Sovereign AI Stack is built on a foundation of **peer-reviewed research**—over 30 academic citations across component specifications—ensuring every design decision is grounded in proven computer science and manufacturing principles.

## What is Batuta?

Batuta (Spanish for "conductor's baton") orchestrates the **22-component Sovereign AI Stack** from Pragmatic AI Labs to convert, optimize, and validate code migrations:

![Sovereign AI Stack](./assets/sovereign-stack.svg)

### Layer 0: Compute Primitives
- **[Trueno](https://github.com/paiml/trueno)** v0.16 - SIMD/GPU compute primitives with zero-copy operations
- **[Trueno-DB](https://github.com/paiml/trueno-db)** v0.3 - Vector database with HNSW indexing ([Malkov 2020])
- **[Trueno-Graph](https://github.com/paiml/trueno-graph)** v0.1 - Graph analytics and lineage DAG tracking
- **[Trueno-Viz](https://github.com/paiml/trueno-viz)** v0.2 - SIMD/GPU/WASM visualization
- **[Trueno-RAG](https://github.com/paiml/trueno-rag)** v0.2 - RAG pipeline: semantic chunking, BM25+dense hybrid retrieval ([Lewis 2020]), cross-encoder reranking

### Layer 1: ML Algorithms
- **[Aprender](https://github.com/paiml/aprender)** v0.27 - First-principles ML in pure Rust

### Layer 2: Training & Inference
- **[Entrenar](https://github.com/paiml/entrenar)** v0.7 - Training with autograd, LoRA, quantization, DP-SGD
- **[Realizar](https://github.com/paiml/realizar)** v0.8 - LLM inference (GGUF, safetensors, transformers)

### Layer 3: Transpilers
- **[Depyler](https://github.com/paiml/depyler)** - Python → Rust with type inference
- **[Decy](https://github.com/paiml/decy)** - C/C++ → Rust with ownership inference
- **[Bashrs](https://github.com/paiml/bashrs)** v6.57 - Rust → Shell (bootstrap scripts)
- **[Ruchy](https://github.com/paiml/ruchy)** v4.1 - Script → Rust (systems scripting)

### Layer 4: Orchestration
- **[Batuta](https://github.com/paiml/batuta)** v0.7 - Orchestration, agents, serving, analysis
- **[Repartir](https://github.com/paiml/repartir)** v2.0 - Distributed computing primitives
- **[pforge](https://github.com/paiml/pforge)** v0.1.4 - MCP server framework (rust-mcp-sdk)

### Layer 5: Quality
- **[Certeza](https://github.com/paiml/certeza)** - Quality validation framework
- **[PMAT](https://github.com/paiml/paiml-mcp-agent-toolkit)** - AI context & code quality
- **[Renacer](https://github.com/paiml/renacer)** v0.10 - Syscall tracing & golden traces
- **[Provable Contracts](https://github.com/paiml/provable-contracts)** - YAML → Kani formal verification for ML kernels
- **[Tiny Model Ground Truth](https://github.com/paiml/tiny-model-ground-truth)** - Popperian model conversion parity tests

### Layer 6: Data & MLOps
- **[Alimentar](https://github.com/paiml/alimentar)** - Data loading with .ald AES-256-GCM encryption
- **[Pacha](https://github.com/paiml/pacha)** - Model/Data/Recipe Registry with BLAKE3 content-addressing, Model Cards ([Mitchell 2019]), Datasheets ([Gebru 2021]), W3C PROV-DM provenance

## The Philosophy

Batuta is built on three core principles, each deeply integrated throughout the stack.

### 1. Toyota Way Manufacturing

We apply Lean Manufacturing principles systematically across all 22 components. This isn't marketing—every specification includes **Toyota Way Review** sections that audit designs against these principles:

#### Muda (Waste Elimination)

The seven wastes, applied to software:

| Waste Type | Traditional Software | Batuta Solution |
|------------|---------------------|-----------------|
| **Transport** | Data copying between services | Zero-copy operations in Trueno |
| **Inventory** | Unused dependencies | Content-addressed deduplication in Pacha |
| **Motion** | Context switching | Single-language stack (pure Rust) |
| **Waiting** | Build times, cold starts | 53,000x faster Lambda cold start |
| **Overproduction** | Features nobody uses | Modular components, use only what you need |
| **Overprocessing** | Redundant transformations | IR-based semantic preservation |
| **Defects** | Bugs, rework | Built-in quality gates at every phase |

> *"By removing dependency hell, we eliminate the waste of waiting and waste of processing associated with complex environments."* — Trueno-RAG Spec

#### Jidoka (Built-in Quality)

Stop the line when defects occur. In Batuta:

- **Chunking**: Semantic chunking stops based on meaning, not arbitrary size—reducing downstream correction waste
- **Validation gates**: Each phase must pass quality checks before proceeding
- **Andon signals**: Immediate visualization of problems via PMAT quality scoring

> *"Fixed-size chunking is prone to defects (cutting semantic context). Semantic chunking stops the chunk based on quality rather than an arbitrary quota."* — Trueno-RAG Spec

#### Kaizen (Continuous Improvement)

Incremental refinement through:

- **Model lineage tracking** in Pacha enables iterative improvement
- **Experiment comparison** identifies what works
- **Golden trace evolution** captures behavioral improvements over time

#### Heijunka (Level Scheduling)

Balance load to avoid overburdening:

- **HNSW parameters** tuned to balance indexing speed with search accuracy
- **Batch processing** in Realizar avoids GPU memory spikes
- **Distributed workloads** via Repartir prevent node overload

#### Genchi Genbutsu (Go and See)

Process data where it resides:

- **Local inference** eliminates waste of transport (sending data to external APIs)
- **Edge deployment** brings computation to the data
- **Sovereign processing** keeps data within your infrastructure

#### Nemawashi (Consensus Decision Making)

Make decisions slowly by consensus, implement rapidly:

- **Hybrid retrieval** uses Reciprocal Rank Fusion (RRF) to integrate diverse "perspectives" (dense and sparse)
- **Multi-query retrieval** pulls more relevant information based on user intent
- **Cross-encoder reranking** ([Nogueira 2019]) refines results through pairwise scoring

> *"Reciprocal Rank Fusion acts as a consensus mechanism, integrating diverse perspectives to make a better decision. This aligns with making decisions slowly by consensus, then implementing rapidly."* — Trueno-RAG Spec

#### One-Piece Flow (Continuous Flow)

Reduce batch sizes to minimize waiting:

- **Streaming retrieval** delivers results the moment they become available
- **Incremental chunking** processes documents as they arrive
- **Async pipelines** eliminate blocking operations

> *"Streaming results implements continuous flow, reducing the batch size to one. This eliminates the waste of waiting for the user, delivering value the moment it is created."* — Trueno-RAG Spec

### 2. Semantic Preservation

**Code migration is NOT a lossy transformation.** Batuta ensures behavioral equivalence through multiple verification layers:

```
Source Code (Python/C/Shell)
        │
        ▼
┌───────────────────┐
│   IR Analysis     │  ← Abstract semantic representation
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   Transpilation   │  ← Idiomatic Rust generation
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   Validation      │  ← Syscall tracing (Renacer)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Golden Trace Diff │  ← Behavioral equivalence proof
└───────────────────┘
```

### 3. First Principles Thinking

Rather than blindly translating code, Batuta rebuilds from fundamental truths:

- **What does this code actually do?** — IR-level semantic analysis
- **What is the minimal correct implementation?** — Eliminate accidental complexity
- **How can we express this idiomatically in Rust?** — Leverage ownership, not fight it

## The 5-Phase Workflow

Batuta follows a strict **5-phase Kanban workflow** with visual control:

```
┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐    ┌────────────┐
│ Analysis │ -> │ Transpilation│ -> │ Optimization │ -> │ Validation│ -> │ Deployment │
└──────────┘    └──────────────┘    └──────────────┘    └───────────┘    └────────────┘
    20%              40%                  60%               80%               100%

 Languages       depyler/decy         SIMD/GPU           Renacer          WASM/Lambda
   Deps          bashrs/ruchy          MoE              Certeza             Edge
   TDG            Caching            Trueno              Tests             Binary
```

Each phase has:
- **Clear entry criteria** — Dependencies on previous phase (Jidoka)
- **Specific deliverables** — Outputs that feed next phase (One-piece flow)
- **Quality gates** — Validation before proceeding (Stop and fix)
- **Automated tracking** — State persistence and progress (Visual control)

## Sovereign AI: Complete Stack

The Sovereign AI Stack is **100% Rust, no Python/C++ dependencies**:

| Capability | Component | Replaces | Key Differentiator |
|------------|-----------|----------|-------------------|
| Tensor ops | Trueno | NumPy | SIMD + GPU, zero-copy operations |
| Vector DB | Trueno-DB | Pinecone, Milvus | Embedded HNSW ([Malkov 2020]) |
| RAG | Trueno-RAG | LangChain | BM25 + dense hybrid, RRF fusion, streaming |
| ML algorithms | Aprender | scikit-learn | .apr format, AES-256-GCM encryption |
| Training | Entrenar | PyTorch | LoRA, quantization, DP-SGD privacy |
| Inference | Realizar | vLLM | GGUF, safetensors, KV-cache, 9.6x faster |
| Data loading | Alimentar | pandas | .ald encryption, Argon2id KDF |
| MLOps | Pacha | MLflow | BLAKE3 deduplication, PROV-DM lineage |

**Why sovereign matters:**
- **No external API calls** — Data never leaves your infrastructure
- **AES-256-GCM encryption** — .apr and .ald formats protect artifacts at rest
- **X25519 + Ed25519** — Key exchange and signatures for secure sharing
- **Pure Rust** — Single audit surface, no C/C++ CVE tracking

## Academic Foundation

Every component specification cites peer-reviewed research. This isn't theory—it's engineering rigor applied to every design decision:

| Specification | References | Key Citations |
|---------------|------------|---------------|
| **Pacha** (MLOps) | 20 papers | Model Cards [Mitchell 2019], Datasheets [Gebru 2021], PROV-DM [W3C 2013], Reproducibility [Pineau 2021] |
| **Trueno-RAG** | 10 papers | RAG [Lewis 2020], DPR [Karpukhin 2020], HNSW [Malkov 2020], BM25 [Robertson 2009], Lost in Middle [Liu 2024] |
| **Oracle Mode** | 20 papers | Stack query interface with academic grounding |

### Selected References

- **[Lewis 2020]** - "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (NeurIPS)
- **[Karpukhin 2020]** - "Dense Passage Retrieval for Open-Domain Question Answering" (EMNLP)
- **[Malkov 2020]** - "Efficient and Robust Approximate Nearest Neighbor Search Using HNSW" (IEEE TPAMI)
- **[Mitchell 2019]** - "Model Cards for Model Reporting" (FAT*)
- **[Gebru 2021]** - "Datasheets for Datasets" (CACM)
- **[Robertson 2009]** - "The Probabilistic Relevance Framework: BM25 and Beyond" (FnTIR)
- **[Liu 2024]** - "Lost in the Middle: How Language Models Use Long Contexts" (TACL)
- **[Nogueira 2019]** - "Passage Re-ranking with BERT" (arXiv)

## Who is This Book For?

This book is for:

- **Legacy codebase maintainers** drowning in Python/C/C++ technical debt
- **Performance engineers** seeking ML inference speedups (10-100x)
- **Systems programmers** modernizing shell-based infrastructure
- **Engineering managers** planning strategic rewrites
- **AI/ML engineers** building sovereign, private AI systems
- **Security teams** requiring single-language audit surfaces

## What You'll Learn

By the end of this book, you will:

1. **Understand the philosophy** — Toyota Way applied to code migration
2. **Master the 5-phase workflow** — Analysis through deployment
3. **Use all stack components** — Hands-on integration patterns
4. **Apply waste elimination** — Identify and remove Muda in your projects
5. **Validate semantic equivalence** — Syscall tracing with Renacer
6. **Optimize performance** — SIMD/GPU acceleration with Trueno
7. **Build RAG pipelines** — Hybrid retrieval with Trueno-RAG
8. **Deploy LLM inference** — GGUF models with Realizar
9. **Track ML experiments** — Model lineage with Pacha
10. **Ensure data privacy** — Encryption and DP-SGD

## Prerequisites

**Required:**
- Basic understanding of Rust (ownership, lifetimes, traits)
- Familiarity with at least one source language (Python, C, C++, Shell)
- Command-line proficiency

**Helpful but not required:**
- Experience with build systems (Cargo, Make, CMake)
- Understanding of ML frameworks (NumPy, PyTorch, scikit-learn)
- Lean manufacturing concepts (helpful for philosophy sections)

## How to Read This Book

**If you're brand new to Batuta:**
Read **Part I (Core Philosophy)** to understand the "why", then work through **Part II (5-Phase Workflow)** hands-on with a small example project.

**If you're experienced with transpilers:**
Start with **Part III (Tool Ecosystem)** to understand Batuta's orchestration capabilities, then dive into **Part IV (Practical Examples)** for real-world patterns.

**If you're migrating a specific project:**
Begin with **Part II (5-Phase Workflow)** for the systematic approach, consult **Part V (Configuration)** for customization, and keep **Part VIII (Troubleshooting)** handy.

**If you're building AI/ML systems:**
Focus on **Part III (Tool Ecosystem)** for Trueno/Aprender/Realizar integration, and **Pacha** for MLOps. Use **Oracle Mode** for intelligent stack queries.

## Running Examples

Batuta includes 30+ runnable examples demonstrating stack capabilities:

```bash
# Core pipeline demo (no features required)
cargo run --example pipeline_demo

# Oracle-mode examples
cargo run --example oracle_local_demo --features oracle-mode

# Stack quality analysis
cargo run --example stack_quality_demo --features native

# PMAT query: function-level code search with quality grades
cargo run --example pmat_query_demo --features native

# Bug-hunter: proactive bug detection with GPU/CUDA patterns
cargo run --example bug_hunter_demo --features native

# ML framework conversion
cargo run --example numpy_conversion
cargo run --example sklearn_conversion
cargo run --example pytorch_conversion
```

See **[Part IV: Example Overview](./part4/example-overview.md)** for the complete list with feature requirements.

## Oracle Mode

Batuta includes **Oracle Mode** — an intelligent query interface backed by a knowledge graph of all 22 components:

```bash
# Natural language queries
batuta oracle "How do I train a model on GPU?"
batuta oracle "What's best for vector similarity search?"
batuta oracle "Which components support WASM?"

# Component discovery
batuta oracle --list-capabilities trueno
batuta oracle --integrations "aprender -> realizar"

# JSON output for automation
batuta oracle --json "RAG pipeline components"
```

Oracle Mode knows capabilities, integration patterns, and recommends optimal component combinations based on your requirements.

## Conventions

Throughout this book:

- **Bold text** emphasizes key concepts
- `Inline code` represents commands, code snippets, or file names
- 💡 **Tips** provide helpful shortcuts
- ⚠️ **Warnings** highlight potential pitfalls
- 🎯 **Best practices** recommend proven approaches
- 🏭 **Toyota Way** callouts show lean manufacturing applications

## Community and Support

- **GitHub**: [paiml/Batuta](https://github.com/paiml/Batuta)
- **Book**: [paiml.github.io/batuta](https://paiml.github.io/batuta/)
- **Issues**: Report bugs and request features
- **Discussions**: Ask questions and share experiences

## Let's Begin

The journey from legacy code to modern Rust is challenging but immensely rewarding. With Batuta orchestrating the 22-component Sovereign AI Stack, you're equipped with:

| Category | Components | Count |
|----------|------------|-------|
| Compute primitives | Trueno, Trueno-DB, Trueno-Graph, Trueno-Viz, Trueno-RAG | 5 |
| ML pipeline | Aprender, Entrenar, Realizar | 3 |
| Transpilers | Depyler, Decy, Bashrs, Ruchy | 4 |
| Orchestration | Batuta, Repartir, pforge | 3 |
| Quality | Certeza, PMAT, Renacer, Provable Contracts, Tiny Model GT | 5 |
| Data & MLOps | Alimentar, Pacha | 2 |
| **Total** | | **22** |

Every component follows Toyota Way principles. Every specification cites peer-reviewed research. Every design decision eliminates waste.

**Welcome to systematic code migration. Let's conduct this orchestra.** 🎵

---

**Next:** [Part I: Core Philosophy](./part1/orchestration-paradigm.md)
