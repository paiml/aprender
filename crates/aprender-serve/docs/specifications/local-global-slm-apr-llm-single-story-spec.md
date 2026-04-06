# Local-Global Unified Model Serving Specification

**Version:** 1.0.0
**Status:** Draft
**Authors:** PAIML Team
**Last Updated:** 2024-12-05

---

## Executive Summary

This specification defines a unified model serving architecture that provides seamless local-to-global scaling for Small Language Models (SLMs), Aprender models (.apr), and Large Language Models (LLMs). The system delivers an "ollama-like" developer experience while enabling production-grade deployments across edge devices, containers, serverless functions, and Kubernetes clusters.

**Design Philosophy:** Single binary, single story—from `realizar run model.gguf` on a laptop to auto-scaling Kubernetes deployments serving millions of requests.

---

## 1. Problem Statement

Current model serving suffers from fragmentation:

| Scenario | Current Tools | Pain Points |
|----------|---------------|-------------|
| Local dev | ollama, llamafile | Not integrated with training pipeline |
| Custom models | Manual GGUF export | No registry, no versioning |
| Production | vLLM, TGI, custom | Different APIs, ops burden |
| Edge | Custom builds | No unified deployment story |

**Gap:** No single system handles the full lifecycle from training (Entrenar) → registry (Pacha) → serving (local/global) with consistent APIs and tooling.

<!-- Annotation 1 [Toyota Way - Muda (Waste)]: The elimination of "glue code" and fragmented tooling addresses the "Hidden Technical Debt in Machine Learning Systems" (Sculley et al., NeurIPS 2015). By unifying the lifecycle, we reduce the waste of handover friction and maintenance of disparate systems. -->

---

## 2. Design Principles (Toyota Way)

### 2.1 Genchi Genbutsu (現地現物) — Go and See

> "Go to the source to find the facts to make correct decisions." [^1]

The serving layer must provide deep observability:
- Request/response tracing with Renacer span IDs
- Token-level latency metrics
- Memory pressure indicators
- Model lineage from Pacha

<!-- Annotation 2 [Genchi Genbutsu]: Deep observability is critical for establishing ground truth. As noted in "Operationalizing Machine Learning: An Interview Study" (Shankar et al., 2022), lack of visibility into model behavior in production is a primary barrier to reliable MLOps. This design prioritizes "going and seeing" the actual inference metrics. -->

### 2.2 Jidoka (自働化) — Autonomation with Human Touch

> "Build in quality at the process." [^2]

Automatic quality gates:
- Preflight validation before serving (Entrenar preflight checks)
- Circuit breakers on latency/error thresholds
- Automatic fallback to smaller quantization on OOM
- Human-in-the-loop approval for production promotions

<!-- Annotation 3 [Jidoka]: The implementation of preflight checks aligns with "The ML Test Score: A Rubric for ML Production Readiness" (Breck et al., 2017), which advocates for automated tests of data and models before deployment to prevent downstream defects. -->

### 2.3 Heijunka (平準化) — Level Loading

> "Level out the workload to reduce waste." [^3]

Request distribution:
- Adaptive batching based on queue depth
- Priority queues for interactive vs. batch inference
- Graceful degradation under load (quantization stepping)

<!-- Annotation 4 [Heijunka]: Adaptive batching is supported by "Clipper: A Low-Latency Online Prediction Serving System" (Crankshaw et al., NSDI 2017), which demonstrates that dynamic batching is essential for leveling latency and throughput (smoothing production flow) in inference systems. -->

### 2.4 Kaizen (改善) — Continuous Improvement

> "Change for the better through small, incremental improvements." [^4]

Feedback loops:
- Automatic metric collection to Trueno-DB
- A/B testing support with traffic splitting
- Model comparison dashboards via Presentar

<!-- Annotation 5 [Kaizen]: Continuous feedback loops are foundational to "TFX: A TensorFlow-Based Production-Scale Machine Learning Platform" (Baylor et al., KDD 2019), enabling the iterative improvement of models based on production data (data flywheels). -->

### 2.5 Kanban (看板) — Visual Workflow

> "Signal-based system for managing work in progress." [^5]

Model lifecycle states:
```
[Training] → [Registered] → [Staged] → [Production] → [Deprecated]
     ↑            ↑             ↑            ↑
  Entrenar      Pacha      Realizar     Batuta orchestration
```

<!-- Annotation 6 [Kanban/Standardization]: Explicit lifecycle states enable better visibility and governance, echoing "Model Cards for Model Reporting" (Mitchell et al., FAccT 2019), which emphasizes the need for clear documentation of model status and provenance to manage risk. -->

---

## 3. Architecture

### 3.1 Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Batuta Orchestration                         │
│  batuta serve | batuta deploy | batuta scale                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │   Pacha     │    │  Realizar   │    │   Deployment        │ │
│  │  Registry   │───▶│   Engine    │───▶│   Targets           │ │
│  │             │    │             │    │                     │ │
│  │ • Models    │    │ • GGUF      │    │ • Local (CLI)       │ │
│  │ • Versions  │    │ • .apr      │    │ • Docker            │ │
│  │ • Lineage   │    │ • SafeT     │    │ • Lambda            │ │
│  │ • Cards     │    │ • MoE       │    │ • Kubernetes        │ │
│  └─────────────┘    └─────────────┘    │ • Fly.io/CF Workers │ │
│        ↑                  ↑            └─────────────────────┘ │
│        │                  │                                     │
│  ┌─────────────┐    ┌─────────────┐                            │
│  │  Entrenar   │    │ Trueno-DB   │                            │
│  │  Training   │    │  Metrics    │                            │
│  └─────────────┘    └─────────────┘                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Model Format Support

| Format | Source | Use Case | Quantization |
|--------|--------|----------|--------------|
| GGUF | llama.cpp ecosystem | LLMs, SLMs | Q4_K, Q8_0, F16 |
| .apr | Aprender native | Classical ML, embeddings | Optional |
| SafeTensors | HuggingFace | Fine-tuned transformers | Via Entrenar |

<!-- Annotation 7 [Muda (Waste)]: Supporting quantization natively reduces computational waste. "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., NeurIPS 2023) highlights how maintaining precision only where necessary enables massive efficiency gains without significant quality loss. -->

### 3.3 Registry Integration (Pacha)

Pacha serves as the unified model registry, providing:

```rust
// Local registry (SQLite-backed)
let registry = pacha::Registry::local("~/.realizar/models")?;

// Global registry (S3/GCS-backed)
let registry = pacha::Registry::remote("s3://company-models")?;

// Hybrid (local cache + remote source)
let registry = pacha::Registry::hybrid(local, remote)?;
```

**Model URI Scheme:**
```
pacha://model-name:version          # Explicit version
pacha://model-name:latest           # Latest version
pacha://model-name@sha256:abc123    # Content-addressed
pacha://model-name:production       # Stage alias
file://./model.gguf                 # Local file
hf://meta-llama/Llama-3-8B          # HuggingFace (auto-convert)
```

---

## 4. User Experience

### 4.1 Local Development (ollama-like)

```bash
# Pull and run (cached in ~/.realizar/models)
realizar run pacha://llama3:8b-q4

# Run local file
realizar run ./my-model.gguf

# Run with custom context
realizar run pacha://codellama:13b --ctx 8192 --port 8080

# Interactive chat
realizar chat pacha://llama3:8b

# List cached models
realizar list
```

### 4.2 Custom Model Workflow

```bash
# Train with Entrenar
entrenar train lora-config.yaml --output ./adapter

# Merge and quantize
entrenar merge base.safetensors ./adapter --output merged.safetensors
entrenar quantize merged.safetensors --bits 4 --output model.gguf

# Register in Pacha
realizar push model.gguf --name "my-slm" --version 1.0.0

# Serve immediately
realizar run pacha://my-slm:1.0.0
```

### 4.3 Production Deployment

```bash
# Generate Dockerfile
batuta deploy dockerfile pacha://my-slm:1.0.0 --output Dockerfile

# Generate Lambda package
batuta deploy lambda pacha://my-slm:1.0.0 --output lambda.zip

# Generate Kubernetes manifests
batuta deploy k8s pacha://my-slm:1.0.0 --replicas 3 --output k8s/

# Deploy to Fly.io
batuta deploy fly pacha://my-slm:1.0.0 --region ord,lax
```

---

## 5. API Specification

### 5.1 OpenAI-Compatible REST API

```
POST /v1/chat/completions
POST /v1/completions
POST /v1/embeddings
GET  /v1/models
GET  /health
GET  /metrics (Prometheus format)
```

### 5.2 Native Realizar API

```
POST /realize/generate     # Streaming generation
POST /realize/batch        # Batch inference
POST /realize/embed        # Embeddings
GET  /realize/model        # Model metadata + lineage
POST /realize/reload       # Hot-reload model
```

### 5.3 MCP Integration (via pforge)

```yaml
# pforge.yaml for MCP-enabled serving
forge:
  name: realizar-mcp

tools:
  - type: native
    name: generate
    handler: { path: realizar::mcp::generate }
    params:
      prompt: { type: string, required: true }
      model: { type: string, required: false }
```

---

## 6. Deployment Targets

### 6.1 Local Binary

Single static binary, ~10MB:

```bash
# Cross-compile targets
realizar build --target x86_64-unknown-linux-musl
realizar build --target aarch64-apple-darwin
realizar build --target wasm32-wasi  # Edge/serverless
```

### 6.2 Docker Container

```dockerfile
# Auto-generated by: batuta deploy dockerfile
FROM scratch
COPY realizar /realizar
COPY model.gguf /model.gguf
EXPOSE 8080
ENTRYPOINT ["/realizar", "serve", "/model.gguf"]
```

**Image sizes:**
- Base realizar: ~10MB
- With 7B Q4 model: ~4GB
- With 1B Q4 model: ~600MB

### 6.3 AWS Lambda

```rust
// Lambda handler (auto-generated)
use realizar::lambda::handler;

#[tokio::main]
async fn main() -> Result<(), Error> {
    let engine = realizar::Engine::from_env()?; // MODEL_PATH env
    lambda_runtime::run(handler(engine)).await
}
```

**Constraints:**
- 10GB package limit → Q4 models up to ~7B params
- Cold start: ~3s (model load from EFS recommended)
- Warm invocation: <100ms TTFT [^6]

<!-- Annotation 8 [Flow]: Addressing cold starts is critical for flow. "Serving Deep Learning Models in a Serverless Platform" (Ishakian et al., 2018) identifies cold start latency as the primary bottleneck for interactive AI on serverless, validating the need for specific optimizations like EFS loading or model caching. -->

### 6.4 Kubernetes

```yaml
# Auto-generated by: batuta deploy k8s
apiVersion: apps/v1
kind: Deployment
metadata:
  name: realizar-llm
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: realizar
        image: pacha.io/my-slm:1.0.0
        resources:
          limits:
            memory: "8Gi"
            nvidia.com/gpu: "1"  # Optional GPU
        env:
        - name: REALIZAR_THREADS
          value: "8"
        - name: REALIZAR_CTX
          value: "4096"
---
apiVersion: v1
kind: Service
metadata:
  name: realizar-llm
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8080
```

### 6.5 Edge (Cloudflare Workers / Fly.io)

```toml
# fly.toml (auto-generated)
app = "my-slm-prod"

[build]
  image = "pacha.io/my-slm:1.0.0"

[[services]]
  internal_port = 8080
  protocol = "tcp"

  [[services.ports]]
    port = 443
    handlers = ["tls", "http"]

  [services.concurrency]
    type = "requests"
    hard_limit = 100
    soft_limit = 80
```

<!-- Annotation 9 [Genchi Genbutsu]: Deploying to edge (Cloudflare/Fly.io) moves computation closer to the user, a principle supported by "Edge Intelligence: Paving the Last Mile of Artificial Intelligence With Edge Computing" (Zhou et al., IEEE 2019), reducing latency and improving local decision making. -->

---

## 7. Observability

### 7.1 Metrics (Trueno-DB)

```sql
-- Query inference latency percentiles
SELECT
  model_name,
  percentile_cont(0.50) WITHIN GROUP (ORDER BY latency_ms) as p50,
  percentile_cont(0.99) WITHIN GROUP (ORDER BY latency_ms) as p99
FROM realizar_metrics
WHERE timestamp > now() - interval '1 hour'
GROUP BY model_name;
```

### 7.2 Tracing (Renacer)

```rust
// Automatic span propagation
let span = renacer::span!("inference", model = %model_name);
let _guard = span.enter();

// Traces flow: Batuta → Realizar → Pacha (lineage lookup)
```

### 7.3 Dashboards (Presentar)

```bash
# Launch monitoring dashboard
batuta viz dashboard --source trueno-db://metrics
```

---

## 8. Security

### 8.1 Model Integrity

All models in Pacha are content-addressed (BLAKE3):

```rust
// Verify before loading
let expected = "blake3:a1b2c3...";
let actual = blake3::hash(&model_bytes);
assert_eq!(expected, actual.to_hex());
```

### 8.2 Signing (Ed25519)

```bash
# Sign model for production
realizar sign model.gguf --key ~/.realizar/signing-key

# Verify signature
realizar verify pacha://my-slm:production
```

<!-- Annotation 10 [Respect for People/Safety]: Ensuring model integrity via signing prevents supply chain attacks. "Backdoor Attacks on Neural Networks" (Gu et al., 2017) demonstrates how compromised models can behave normally until triggered; cryptographic verification is a necessary Jidoka (stop at error) control. -->

### 8.3 Encryption at Rest

```bash
# Encrypt model for distribution
realizar encrypt model.gguf --password-env MODEL_KEY

# Decryption happens at load time
REALIZAR_KEY=secret realizar run encrypted-model.gguf.enc
```

---

## 9. Batuta Integration

### 9.1 CLI Commands

```bash
# Serve commands
batuta serve run pacha://model:version    # Run locally
batuta serve list                          # List running servers
batuta serve stop <id>                     # Stop server
batuta serve logs <id>                     # View logs

# Deploy commands
batuta deploy dockerfile <model> [--gpu]   # Generate Dockerfile
batuta deploy lambda <model>               # Generate Lambda package
batuta deploy k8s <model> [--replicas N]   # Generate K8s manifests
batuta deploy fly <model> [--region R]     # Deploy to Fly.io

# Registry commands (delegates to Pacha)
batuta registry list                       # List models
batuta registry push <file> --name <n>     # Push model
batuta registry pull <model>               # Pull model
batuta registry promote <model> --stage X  # Promote stage
```

### 9.2 Configuration (batuta.toml)

```toml
[serve]
default_backend = "realizar"
cache_dir = "~/.realizar/models"
default_ctx = 4096
default_threads = 8

[serve.registry]
local = "~/.realizar/models"
remote = "s3://company-models"
strategy = "hybrid"  # local-first, remote-fallback

[deploy]
default_target = "docker"
gpu_enabled = false
base_image = "gcr.io/distroless/static"
```

---

## 10. Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Cold start (local) | <2s | Model load to first token |
| Cold start (Lambda) | <5s | Including runtime init |
| TTFT (warm) | <100ms | Time to first token |
| Throughput (7B Q4) | >30 tok/s | Single A10G GPU |
| Throughput (1B Q4) | >100 tok/s | M2 MacBook Pro |
| Memory (7B Q4) | <6GB | Peak RSS |
| Binary size | <15MB | Static Linux binary |

---

## 11. Migration Path

### From ollama:

```bash
# Export ollama model
ollama show llama3:8b --modelfile > Modelfile

# Convert to GGUF (already GGUF internally)
cp ~/.ollama/models/llama3-8b/model.gguf ./

# Register in Pacha
realizar push model.gguf --name llama3 --version 8b
```

### From llamafile:

```bash
# llamafile is GGUF + runtime, extract model
unzip model.llamafile model.gguf

# Register
realizar push model.gguf --name my-model --version 1.0.0
```

### From HuggingFace:

```bash
# Auto-convert and register
realizar import hf://meta-llama/Llama-3-8B --quantize q4_k --name llama3
```

---

## 12. Implementation Phases

### Phase 1: Local Experience (4 weeks)
- [ ] `realizar run` CLI with GGUF support
- [ ] Local Pacha registry integration
- [ ] OpenAI-compatible API
- [ ] Basic metrics

### Phase 2: Registry & Versioning (3 weeks)
- [ ] Remote Pacha registry (S3/GCS)
- [ ] Model signing and verification
- [ ] Version promotion workflow
- [ ] Batuta `serve` commands

### Phase 3: Deployment Targets (4 weeks)
- [ ] Dockerfile generation
- [ ] Lambda packaging
- [ ] Kubernetes manifests
- [ ] Fly.io integration

### Phase 4: Observability & Production (3 weeks)
- [ ] Trueno-DB metrics integration
- [ ] Renacer tracing
- [ ] Presentar dashboard
- [ ] A/B testing support

---

## References

### Toyota Way Foundations

[^1]: Liker, J. K. (2004). *The Toyota Way: 14 Management Principles from the World's Greatest Manufacturer*. McGraw-Hill. Chapter 12: Genchi Genbutsu.

[^2]: Ohno, T. (1988). *Toyota Production System: Beyond Large-Scale Production*. Productivity Press. pp. 75-82 on Jidoka.

[^3]: Womack, J. P., & Jones, D. T. (2003). *Lean Thinking: Banish Waste and Create Wealth in Your Corporation*. Free Press. Chapter 4: Heijunka.

[^4]: Imai, M. (1986). *Kaizen: The Key to Japan's Competitive Success*. McGraw-Hill. pp. 23-35.

[^5]: Anderson, D. J. (2010). *Kanban: Successful Evolutionary Change for Your Technology Business*. Blue Hole Press. Chapter 2.

### Inference & Quantization

[^6]: Pope, R., et al. (2023). "Efficiently Scaling Transformer Inference." *Proceedings of MLSys*. https://arxiv.org/abs/2211.05102

[^7]: Dettmers, T., et al. (2022). "GPT3.int8(): 8-bit Matrix Multiplication for Transformers at Scale." *NeurIPS 2022*. https://arxiv.org/abs/2208.07339

[^8]: Frantar, E., et al. (2023). "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." *ICLR 2023*. https://arxiv.org/abs/2210.17323

[^9]: Kwon, W., et al. (2023). "Efficient Memory Management for Large Language Model Serving with PagedAttention." *SOSP 2023*. https://arxiv.org/abs/2309.06180

[^10]: Sheng, Y., et al. (2023). "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU." *ICML 2023*. https://arxiv.org/abs/2303.06865

### MLOps & Production Systems (Annotation Sources)

[^11]: Sculley, D., et al. (2015). "Hidden Technical Debt in Machine Learning Systems." *NeurIPS 2015*. https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems

[^12]: Shankar, S., et al. (2022). "Operationalizing Machine Learning: An Interview Study." *arXiv*. https://arxiv.org/abs/2209.09125

[^13]: Breck, E., et al. (2017). "The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction." *IEEE BigData 2017*. https://research.google/pubs/pub46555/

[^14]: Crankshaw, D., et al. (2017). "Clipper: A Low-Latency Online Prediction Serving System." *NSDI 2017*. https://www.usenix.org/conference/nsdi17/technical-sessions/presentation/crankshaw

[^15]: Baylor, D., et al. (2019). "TFX: A TensorFlow-Based Production-Scale Machine Learning Platform." *KDD 2019*. https://dl.acm.org/doi/10.1145/3292500.3330746

[^16]: Mitchell, M., et al. (2019). "Model Cards for Model Reporting." *FAccT 2019*. https://arxiv.org/abs/1810.03993

[^17]: Dettmers, T., et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs." *NeurIPS 2023*. https://arxiv.org/abs/2305.14314

[^18]: Ishakian, V., et al. (2018). "Serving Deep Learning Models in a Serverless Platform." *IEEE IC2E 2018*. https://arxiv.org/abs/1710.08460

[^19]: Zhou, Z., et al. (2019). "Edge Intelligence: Paving the Last Mile of Artificial Intelligence With Edge Computing." *Proceedings of the IEEE*. https://ieeexplore.ieee.org/document/8736011

[^20]: Gu, T., et al. (2017). "BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain." *arXiv*. https://arxiv.org/abs/1708.06733

---

## Appendix A: Stack Component Versions

| Component | Version | Role |
|-----------|---------|------|
| Realizar | 0.3.0 | Inference engine |
| Pacha | 0.1.0 | Model registry |
| Entrenar | 0.2.4 | Training/quantization |
| Trueno | 0.7.3 | Compute primitives |
| Trueno-DB | 0.3.4 | Metrics storage |
| Renacer | 0.7.0 | Tracing |
| Batuta | 0.1.2 | Orchestration |
| Presentar | 0.1.0 | Dashboards |
| pforge | 0.1.2 | MCP integration |

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| SLM | Small Language Model (<3B parameters) |
| LLM | Large Language Model (>7B parameters) |
| GGUF | GPT-Generated Unified Format (llama.cpp native) |
| .apr | Aprender model format |
| TTFT | Time To First Token |
| MoE | Mixture of Experts |
| Pacha | PAIML Model/Data/Recipe Registry |
| Realizar | PAIML Inference Engine |

