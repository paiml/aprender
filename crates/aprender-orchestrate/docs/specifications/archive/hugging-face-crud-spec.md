# Hugging Face Query & Ecosystem Specification (CRUD)

**Version**: 1.2.0
**Status**: Implementation (Observability Remediation)
**Created**: 2026-01-12
**Last Updated**: 2026-01-12
**Authors**: Pragmatic AI Labs

## Executive Summary

This specification defines the query and ecosystem management capabilities (CRUD) for Hugging Face integration within Batuta. The primary use case is supporting educational content development for the "Next-Gen AI Development with Hugging Face" Coursera specialization (5 courses, 60 hours, 15 weeks), while providing a robust interface for Sovereign AI Stack orchestration.

The first iteration focuses on **Read/Query operations** and **Integration Visibility**, with a roadmap for full CRUD capabilities.

---

## Table of Contents

1. [Purpose and Scope](#1-purpose-and-scope)
2. [Use Case: Coursera Specialization Support](#2-use-case-coursera-specialization-support)
3. [Hugging Face Ecosystem Taxonomy](#3-hugging-face-ecosystem-taxonomy)
4. [Query Requirements](#4-query-requirements)
5. [Technical Architecture](#5-technical-architecture)
6. [TUI & Observability](#6-tui--observability)
7. [Toyota Way & Jidoka Quality Gates](#7-toyota-way--jidoka-quality-gates)
8. [Work Items (pmat Tickets)](#8-work-items-pmat-tickets)
9. [100-Point Falsification Checklist](#9-100-point-falsification-checklist)
10. [Peer-Reviewed Citations](#10-peer-reviewed-citations)
11. [Appendices](#11-appendices)

---

## 1. Purpose and Scope

### 1.1 Problem Statement

Current Batuta HuggingFace integration (`batuta hf tree`) provides a static view of the ecosystem. Course developers and AI engineers need comprehensive, dynamic, and queryable access to:

- **50+ HuggingFace offerings** (libraries, tools, services, formats)
- **Live Hub metadata** (700K+ models, 100K+ datasets, 300K+ spaces)
- **Course-aligned categorization** (module-specific component mapping)
- **Observability** (tracing Hub interactions and model selection)

### 1.2 Scope

**In Scope (v1.1)**:
- **Read (Query)**: Static catalog and live Hub search.
- **Read (Ecosystem)**: PAIML-HF integration mapping (`tree --integration`).
- **Create (Pull)**: Downloading assets from Hub (`hf pull`).
- **Update/Push (Push)**: Publishing assets to Hub (`hf push`).
- **TUI Visualization**: Real-time TUI for catalog browsing.
- **Renacer Tracing**: Golden traces for Hub API calls.

**Out of Scope**:
- Multi-org secret management (handled by `pacha`).
- Direct Hub repo deletion (safety-critical, deferred to v2.0).

### 1.3 Success Criteria

1. Query any of 50+ HuggingFace ecosystem components.
2. Retrieve metadata for Hub assets (models, datasets, spaces).
3. Filter by course module requirements (1-15 weeks).
4. Export structured data for course planning and CI/CD.
5. 100% compliance with the 100-point falsification checklist (Toyota Standard).

---

## 2. Use Case: Coursera Specialization Support

*(Section remains largely the same but updated with TUI references)*

### 2.1 Specialization Overview

| Attribute | Value |
|-----------|-------|
| Title | Next-Gen AI Development with Hugging Face |
| Courses | 5 |
| Duration | 60 hours / 15 weeks |

...

**TUI Interaction Pattern**:
User launches `batuta hf catalog --tui`, selects "Course 3", and sees a filtered list of components (Sentence-Transformers, FAISS, RAG patterns) with live Hub examples.

---

## 3. Hugging Face Ecosystem Taxonomy

*(Section remains largely the same, ensuring alignment with `src/hf/tree.rs` categories: hub, libraries, inference, training, formats, tasks)*

...

---

## 4. Query Requirements

### 4.1 Functional Requirements (Updated)

#### FR-001: Static Ecosystem Query (Enhanced)
- **Description**: Query the complete catalog with integration status.
- **CLI**: `batuta hf catalog`
- **TUI**: Interactive tree with component details.

#### FR-002: Live Hub Search
- **Description**: Search models, datasets, and spaces.
- **Observability**: Every search emits a `renacer` trace.

#### FR-007: Smart Regression Discovery (New)
- **Description**: Automatically find models matching specific tasks and constraints for `batuta falsify` regression suites.
- **Priority**: P1
- **CLI**: `batuta hf search models --task text-generation --min-likes 100 --falsify-suite`

---

## 5. Technical Architecture

### 5.1 Component Diagram (Updated)

```
┌─────────────────────────────────────────────────────────────┐
│                      batuta CLI / TUI                        │
├─────────────────────────────────────────────────────────────┤
│  batuta hf <command>                                         │
│    ├── catalog    (TUI-enabled ecosystem browse)             │
│    ├── search     (Live Hub search + Renacer Traces)         │
│    ├── info       (Asset metadata + Model Card Parsing)      │
│    └── pull/push  (Data movement + Secret Scanning)          │
├─────────────────────────────────────────────────────────────┤
│                    HfQueryEngine                             │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │ EcosystemCatalog│  │  HubApiClient   │                   │
│  │  (Data-driven)  │  │  (Reqwest/Tokio)│                   │
│  └────────┬────────┘  └────────┬────────┘                   │
│           │                    │                             │
│  ┌────────▼────────────────────▼────────┐  ┌───────────────┐│
│  │           ResponseCache              │  │    Renacer    ││
│  │  (Content-Addressable, 15m TTL)      │◄─┤ Observability ││
│  └──────────────────────────────────────┘  └───────────────┘│
└─────────────────────────────────────────────────────────────┘
```

---

## 6. TUI & Observability

### 6.1 Interactive Catalog (TUI)
The `batuta hf catalog --tui` command provides a real-time dashboard:
- **Navigation**: Vim-like keys (j, k, h, l) to navigate the ecosystem tree.
- **Search**: `/` to search within the catalog.
- **Hub Preview**: Side panel showing live Hub metadata for the selected component.

### 6.2 Renacer Integration
All Hugging Face operations are instrumented with Renacer:
- **Trace Context**: Each API call includes the Coursera course/week context if applicable.
- **Smart Regression Detection**: Queries that result in 429 (Rate Limit) or 503 (Hub Down) are flagged for smart retry/isolation forest analysis.
- **Golden Traces**: Successful model info fetches are stored as golden traces for offline development.

---

## 7. Toyota Way & Jidoka Quality Gates

### 7.1 Automated Quality Gates (Jidoka)
The ecosystem catalog is subject to automated verification:
1. **Link Health**: CI/CD task that verifies all `docs_url` and `repo_url` are live.
2. **Version Parity**: Automatically flags when a library (e.g., `peft`) has a new release on PyPI that isn't reflected in the catalog metadata.
3. **Andon Warning**: If the Hub API latency exceeds 5s, Batuta triggers an "Andon" status in the TUI, recommending cached mode.

### 7.2 Genchi Genbutsu (Go and See)
Catalog data must be verified against actual Hub assets. The `batuta falsify` command includes a HuggingFace-specific suite to verify the catalog's accuracy against the live Hub.

---

## 8. Work Items (pmat Tickets)

*(Updated to include TUI and Observability)*

| Ticket | Title | Priority | Points | Description |
|--------|-------|----------|--------|-------------|
| HF-QUERY-001 | Ecosystem Catalog | P0 | 8 | Data-driven registry (JSON) |
| HF-QUERY-002 | Hub Search | P0 | 8 | Live search with Renacer tracing |
| HF-TUI-001 | HF Dashboard | P1 | 10 | Real-time TUI for catalog/search |
| HF-QA-001 | Jidoka Gates | P1 | 5 | Link checker & Version parity |
| **Total** | | | **31** | |

---

## 9. 100-Point Falsification Checklist (v1.1)

### 9.1 Observability & Tracing (10 points - NEW)

| # | Check | Falsification Condition |
|---|-------|------------------------|
| 101 | Search queries emit Renacer traces | Query executed without trace record |
| 102 | Trace context includes course/week | Context missing in `renacer log` |
| 103 | Golden traces exist for catalog items | Missing `golden_traces/hf/*.json` |
| 104 | Rate limit (429) triggers backoff trace | Silent failure on 429 |
| 105 | TUI updates in real-time | Latency > 200ms for UI refresh |
| 106 | Offline mode uses cached traces | Fails when `HF_TOKEN` is unset & offline |
| 107 | Trace ID propagates to Hub headers | Missing `X-Batuta-Trace-Id` in requests |
| 108 | Isolation forest detects Hub anomalies | 503 errors not flagged as anomalies |
| 109 | Trace size is bounded | Traces > 1MB per query |
| 110 | Sensitive tokens never logged | Tokens found in traces |

*(Sections 8.1 - 8.10 from v1.0.0 remain as the core checklist)*

---

## 10. Peer-Reviewed Citations

*(Existing citations remain, plus new ones for Observability)*

26. **Gregg, B. (2020)**. "Systems Performance: Enterprise and the Cloud." *Addison-Wesley*. (Reference for Renacer tracing principles).
27. **Shingo, S. (1986)**. "Zero Quality Control: Source Inspection and the Poka-yoke System." *Productivity Press*. (Reference for Jidoka/Poka-yoke).

...

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-12 | Pragmatic AI Labs | Initial specification |
| 1.1.0 | 2026-01-12 | Batuta Agent | Integrated TUI, Renacer, and Toyota Way principles |
| 1.2.0 | 2026-01-12 | QA-7 Auditor | Falsification audit findings, observability remediation |

---

## Appendix A: Falsification Audit Report (2026-01-12)

### A.1 Audit Summary

| Metric | Required | Actual | Status |
|--------|----------|--------|--------|
| Components | 50+ | 51 | ✅ PASS |
| Tests | 170+ | 177 | ✅ PASS |
| Coverage | 95% | ~97% | ✅ PASS |
| Renacer traces | Required | 0 | ❌ FAIL |
| TUI widgets | Required | 0 | ⚠️ DEFERRED |

**H0 Verdict**: REJECTED (Observability gap)

### A.2 Findings

#### F-001: Missing Renacer Tracing (CRITICAL)
- **Spec Ref**: Section 6.2, Checklist 101-110
- **Evidence**: `grep -r renacer src/hf/` returns 0 matches
- **Impact**: Hub API calls are not traced, violating observability requirements
- **Remediation**: HF-OBS-001, HF-OBS-002

#### F-002: TUI Dashboard Not Implemented (DEFERRED)
- **Spec Ref**: Section 6.1
- **Evidence**: No `--tui` flag in `hf catalog` command
- **Impact**: Interactive browsing not available
- **Remediation**: HF-TUI-001 (deferred to v1.3.0)

#### F-003: Module Structure Deviation (ACCEPTABLE)
- **Spec Ref**: Implied separate `course.rs`, `search.rs`
- **Evidence**: Functionality consolidated in `catalog.rs`, `hub_client.rs`
- **Impact**: None (design choice, functionality complete)
- **Status**: Accepted

### A.3 Remediation Work Items

| Ticket | Title | Priority | Points | Status |
|--------|-------|----------|--------|--------|
| HF-OBS-001 | Add tracing to HubClient::search_* | P0 | 3 | IN PROGRESS |
| HF-OBS-002 | Add tracing to HubClient::get_* | P0 | 2 | PENDING |
| HF-OBS-003 | Add tracing to catalog operations | P1 | 2 | PENDING |
| HF-TUI-001 | HF Dashboard TUI | P2 | 10 | DEFERRED |

---

## 9. Peer-Reviewed Citations

### 9.1 HuggingFace Technical Papers

1. **Wolf, T., et al. (2020)**. "Transformers: State-of-the-Art Natural Language Processing." *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, pp. 38-45. ACL. DOI: 10.18653/v1/2020.emnlp-demos.6

   *Foundational paper on the transformers library architecture and design principles.*

2. **Lhoest, Q., et al. (2021)**. "Datasets: A Community Library for Natural Language Processing." *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, pp. 175-184. ACL. DOI: 10.18653/v1/2021.emnlp-demo.21

   *Describes the datasets library design for efficient data loading and processing.*

3. **Mangrulkar, S., et al. (2022)**. "PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods." *arXiv preprint arXiv:2304.01933*.

   *Technical overview of parameter-efficient fine-tuning methods including LoRA.*

4. **von Werra, L., et al. (2023)**. "TRL: Transformer Reinforcement Learning." *GitHub Repository*. https://github.com/huggingface/trl

   *Reference implementation for RLHF and DPO training.*

5. **Hugging Face Team (2023)**. "Text Generation Inference." *Technical Documentation*. https://huggingface.co/docs/text-generation-inference

   *Production deployment architecture for LLM serving.*

6. **Reimers, N., & Gurevych, I. (2019)**. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing*. DOI: 10.18653/v1/D19-1410

   *Foundational paper for sentence-transformers library.*

7. **Raffel, C., et al. (2020)**. "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." *Journal of Machine Learning Research*, 21(140), 1-67.

   *T5 paper establishing transfer learning paradigms used in HuggingFace ecosystem.*

### 9.2 ML Education Methodology

8. **Ng, A. (2021)**. "Machine Learning Yearning." *deeplearning.ai*.

   *Practical guide for structuring ML education and project-based learning.*

9. **Géron, A. (2022)**. "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow." *O'Reilly Media*, 3rd Edition. ISBN: 978-1098125974.

   *Reference for progressive ML curriculum design from basics to production.*

10. **Chollet, F. (2021)**. "Deep Learning with Python." *Manning Publications*, 2nd Edition. ISBN: 978-1617296864.

    *Pedagogical approach to teaching deep learning concepts progressively.*

11. **Howard, J., & Gugger, S. (2020)**. "Deep Learning for Coders with fastai and PyTorch." *O'Reilly Media*. ISBN: 978-1492045526.

    *Top-down teaching methodology for ML education.*

12. **Bloom, B. S. (1956)**. "Taxonomy of Educational Objectives: The Classification of Educational Goals." *Longmans, Green*.

    *Foundation for learning outcome design in course curriculum.*

### 9.3 Software Engineering & API Design

13. **Fielding, R. T. (2000)**. "Architectural Styles and the Design of Network-based Software Architectures." *Doctoral dissertation*, University of California, Irvine.

    *REST principles for API design used in HuggingFace Hub API.*

14. **Gamma, E., et al. (1994)**. "Design Patterns: Elements of Reusable Object-Oriented Software." *Addison-Wesley*. ISBN: 978-0201633610.

    *Factory and Strategy patterns used in HuggingFace Auto classes.*

15. **Martin, R. C. (2008)**. "Clean Code: A Handbook of Agile Software Craftsmanship." *Prentice Hall*. ISBN: 978-0132350884.

    *Code quality principles applied to query implementation.*

16. **Fowler, M. (2018)**. "Refactoring: Improving the Design of Existing Code." *Addison-Wesley*, 2nd Edition. ISBN: 978-0134757599.

    *Refactoring patterns for evolving query capabilities.*

17. **Newman, S. (2021)**. "Building Microservices." *O'Reilly Media*, 2nd Edition. ISBN: 978-1492034025.

    *Service design principles for Hub API integration.*

### 9.4 Quantization & Optimization

18. **Dettmers, T., et al. (2022)**. "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale." *arXiv preprint arXiv:2208.07339*.

    *Foundation for bitsandbytes INT8 quantization.*

19. **Dettmers, T., et al. (2023)**. "QLoRA: Efficient Finetuning of Quantized LLMs." *arXiv preprint arXiv:2305.14314*.

    *4-bit quantization with LoRA for efficient fine-tuning.*

20. **Frantar, E., et al. (2022)**. "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." *arXiv preprint arXiv:2210.17323*.

    *GPTQ quantization method supported by HuggingFace.*

### 9.5 Alignment & RLHF

21. **Ouyang, L., et al. (2022)**. "Training language models to follow instructions with human feedback." *arXiv preprint arXiv:2203.02155*.

    *InstructGPT paper establishing RLHF methodology.*

22. **Rafailov, R., et al. (2023)**. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." *arXiv preprint arXiv:2305.18290*.

    *DPO paper for preference-based alignment without reward models.*

23. **Schulman, J., et al. (2017)**. "Proximal Policy Optimization Algorithms." *arXiv preprint arXiv:1707.06347*.

    *PPO algorithm used in TRL for RLHF training.*

### 9.6 Production Deployment

24. **Kwon, W., et al. (2023)**. "Efficient Memory Management for Large Language Model Serving with PagedAttention." *arXiv preprint arXiv:2309.06180*.

    *vLLM PagedAttention technique used in TGI.*

25. **Pope, R., et al. (2022)**. "Efficiently Scaling Transformer Inference." *arXiv preprint arXiv:2211.05102*.

    *Scaling techniques for production LLM inference.*

---

## 10. Appendices

### Appendix A: Complete Component Catalog

See `src/hf/catalog.json` for the full component registry.

### Appendix B: Course-Component Mapping Matrix

| Component | C1W1 | C1W2 | C1W3 | C2W1 | C2W2 | C2W3 | C3W1 | C3W2 | C3W3 | C4W1 | C4W2 | C4W3 | C5W1 | C5W2 | C5W3 |
|-----------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| hub | X | | | | | X | | | | | | | | | |
| transformers | | X | X | | X | | X | | | | | | | | |
| datasets | | | | X | | | | | | | X | | | | |
| trainer | | | | | X | | | | | | X | | | | |
| evaluate | | | | | | X | | | | | | | | | |
| whisper | | | X | | | | | | | | | | | | |
| sentence-transformers | | | | | | | | X | | | | | | | |
| peft | | | | | | | | | | X | | | | | |
| trl | | | | | | | | | | | X | X | | | |
| bitsandbytes | | | | | | | | | | X | | | | | |
| tgi | | | | | | | | | | | | | X | | |
| gradio | | | | | | | | | | | | | | X | |
| optimum | | | | | | | | | | | | | | | X |
| transformers.js | | | | | | | | | | | | | | | X |

### Appendix C: API Rate Limits

| Endpoint | Limit | Window | Retry-After |
|----------|-------|--------|-------------|
| /api/models | 1000 | 1 hour | 3600s |
| /api/datasets | 1000 | 1 hour | 3600s |
| /api/spaces | 1000 | 1 hour | 3600s |
| /api/models/{id} | 1000 | 1 hour | 3600s |

### Appendix D: Glossary

| Term | Definition |
|------|------------|
| PEFT | Parameter-Efficient Fine-Tuning |
| LoRA | Low-Rank Adaptation |
| QLoRA | Quantized LoRA |
| DPO | Direct Preference Optimization |
| RLHF | Reinforcement Learning from Human Feedback |
| TGI | Text Generation Inference |
| TEI | Text Embeddings Inference |
| RAG | Retrieval-Augmented Generation |
| SFT | Supervised Fine-Tuning |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-12 | Pragmatic AI Labs | Initial specification |

---

*This specification is part of the Batuta Sovereign AI Stack documentation.*
