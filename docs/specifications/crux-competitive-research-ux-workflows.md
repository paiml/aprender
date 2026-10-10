# CRUX: Competitive Research UX Workflows

**Subspec ID**: `CRUX-001`
**Status**: DRAFT
**Version**: 2.4 (2026-10-09 — Category R [SHACL ontologies vs Apache Jena, 4 stories] registered, epic aprender#3598, §14; story total in the master registry 291 → 295). 2.3 (2026-09-16 — Category O [AutoGluon AutoML parity, 24 stories] added, epic aprender#3370; Category N [linfa + burn, 17 stories, aprender#3146, 2026-09-12] exists in the master contract and is recorded here in §3 for the first time; story total in the master registry 267 → 291)
**Date**: 2026-10-09
**Author**: PAIML Engineering
**Parent**: [aprender-spec.md](aprender-spec.md), [aprender-monorepo-consolidation.md](aprender-monorepo-consolidation.md)
**Master contract**: [`contracts/crux-competitive-research-ux-v1.yaml`](../../contracts/crux-competitive-research-ux-v1.yaml)

---

## 1. Motivation

Aprender ships 57 `apr` subcommands and 634 contracts, but a user arriving from
Ollama, llama.cpp, PyTorch, Hugging Face, vLLM, or OpenCLAW brings a
mental model built on those tools. If `apr` does not name, order, and complete
the same workflows they already know, adoption friction is fatal — regardless
of whether the feature technically exists somewhere.

This subspec extracts **root-cause workflows** from six dominant open-source
projects, ranks each by **demand signal** (README prominence × GitHub-issue
volume × bug-tracker frequency), writes each as a provable contract, and
enumerates **250 user stories** that aprender must support. Each story gets:

1. A **contract YAML** at `contracts/crux-{category}-{id}-v1.yaml`
2. A **falsification condition** derived from the competitor's canonical CLI
3. A **coverage status**: `✅ supported`, `🔨 partial`, `❌ missing`, `🤔 unclear`
4. A **demand score** 1–5 (5 = highest demand in the competitor's community)
5. A **pmat work item** auto-generated for every ❌ missing story (see §12)

> **Iron Rule** (mirrors apr-book-spec.md): **No contract → no user story**.
> A story without a provable contract is a wish; a wish is muda.

---

## 2. Methodology — Root Cause Workflow Discovery

For each competitor we apply **Five Whys** to the canonical command, then rank
the surfaced workflow by three orthogonal signals:

| Signal | Source | Weight |
|--------|--------|--------|
| README prominence | position above/below fold | 0.3 |
| Issue volume | GitHub `is:issue label:* sort:comments` | 0.4 |
| Bug-report frequency | issues with `label:bug` created in last 180d | 0.3 |

The composite maps to a 1–5 demand score. Stories ≥ 4 are **P0**; ≤ 2 are **P2**.

Example root-cause decomposition:

```
$ ollama run llama3
  WHY does the user type this? → they want a chat REPL
  WHY a REPL? → they iterate on prompts without rebuild
  WHY iterate? → exploring model behavior
  WHY explore? → no a priori confidence in outputs
  WHY no confidence? → no contract asserts correctness

ROOT CAUSE: user needs a contract-backed REPL with golden-output gates.
MAPS TO:    CRUX-C-02 (apr chat) + contracts/apr-model-qa-v1.yaml
DEMAND:     5/5 (README verb #1, issue-volume top-5, bug-label top-10)
```

### 2.1 Evidence collection

Per competitor we capture:

| Artifact | Location |
|----------|----------|
| README verbs (ranked by fold position) | `evidence/crux/{competitor}/readme-verbs.txt` |
| Top issues by comment count | `evidence/crux/{competitor}/top-issues.json` |
| Bug-label frequency histogram | `evidence/crux/{competitor}/bug-histogram.json` |
| CLI `--help` transcript | `evidence/crux/{competitor}/help.txt` |
| Server OpenAPI / routes | `evidence/crux/{competitor}/routes.yaml` |
| Canonical "hello world" flow | `evidence/crux/{competitor}/hello.sh` |

---

## 3. Competitor Matrix — 250 stories distributed

Target: roughly-even population per competitor, weighted up for projects with
larger workflow surface area (HF Transformers covers training + data + hub).

| # | Project | Canonical verb | Stories | Rationale |
|---|---------|---------------|---------|-----------|
| 1 | **Ollama** | `ollama run` | 21 | narrow but high-demand per-verb |
| 2 | **llama.cpp** | `llama-cli` / `llama-quantize` | 34 | convert + quantize + serve + embed |
| 3 | **PyTorch** | `torch.compile` / `nn.Module` | 33 | training mechanism shared with HF |
| 4 | **Hugging Face** | `Trainer` / `AutoModel` / `huggingface-cli` | 78 | widest surface: train + hub + datasets |
| 5 | **vLLM** | `vllm serve` / `LLM.generate` | 32 | serving depth (paged attn, batching) |
| 6 | **OpenCLAW** | `openclaw onboard` / `openclaw dashboard` | 20 | local-first personal AI assistant + agent orchestration (openclaw.ai) |
| 7 | **Ecosystem interop** | — | 30 | SDKs, MCP, observability, deployment |
| 8 | **HF kernels-community** | `get_kernel("kernels-community/<name>")` | 15 | optimized GPU kernels as drop-in `.so` packages (v2.2) |
| 9 | **APR-QA Playbook** | `apr qa --gate=<N>` / `apr-model-qa-playbook` | 10 | Popperian falsification framework for model qualification (v2.2) |
| 10 | **linfa + Burn** | `linfa::traits::Fit` / `burn::module::Module` | 17 | Rust-native ML frameworks; substrates (SVD, spatial index, rank-typed tensor) + breadth (Category N, 2026-09-12, aprender#3146) |
| 11 | **AutoGluon** | `TabularPredictor(label).fit()` / `TimeSeriesPredictor.fit()` | 24 | AutoML: one-call tabular fit, presets, leaderboard, bagging/stacking/weighted ensemble, panel forecasting with quantiles (Category O, v2.3, aprender#3370) |
| 12 | **Apache Jena** | `riot` / `shacl validate` | 4 | a second engine for a format pv already speaks: read, write back, validate and report SHACL ontologies, one story per cell of §14.2 (Category R, v2.4, aprender#3598) |

Total = 275 stories in this document's §5 as of v2.2; the master registry carries 295 (Categories L and M are contract-only, see §6 note). See §5 for the full registry.
(Counts derived from `yq '[.stories[] | .competitor] | ...'` on master contract; drift between
this table and the YAML is falsified by FALSIFY-CRUX-010.)

> **OpenCLAW identity resolved 2026-04-18**: competitor is
> [openclaw.ai](https://openclaw.ai) — a local-first personal AI assistant
> / agent orchestration layer (chat-app transports, system control,
> browser automation, persistent memory, MCP-shaped skill catalog). NOT
> OpenCLIP. See §10 for the resolution record.

---

## 4. Contract naming scheme

```
contracts/crux-{letter}-{nn}-v1.yaml
         │     │     │
         │     │     └── two-digit story ID within category
         │     └── category letter (A..K)
         └── subspec namespace
```

Example: `contracts/crux-C-03-v1.yaml` = CRUX-C-03 "OpenAI-compatible server".

Every CRUX sub-contract `parent_contracts: [crux-competitive-research-ux-v1]`.

Coverage statuses:
- **✅ supported** — `apr <verb>` already exists AND has a golden test
- **🔨 partial** — exists but missing features required by the competitor UX
- **❌ missing** — no `apr` surface yet → `pmat work add` ticket auto-created
- **🤔 unclear** — exists under a different name → alias / rename RFC

Demand score:
- **5** — canonical verb; in README fold; top-20 issue-volume
- **4** — high-traffic verb; common bug-label target
- **3** — documented power-user feature; moderate volume
- **2** — niche feature; low volume; still a real workflow
- **1** — edge case; included for completeness

---

## 5. User Stories (250)

Legend: **S** = status (✅/🔨/❌/🤔), **D** = demand (1–5). Contract file is
always `contracts/crux-{ID}-v1.yaml` unless noted.

### Category A — Model Acquisition & Registry (25 stories)

| ID | Story | Competitor verb | S | D |
|----|-------|----------------|---|---|
| CRUX-A-01 | Pull a model by short name (`apr pull llama3`) | `ollama pull llama3` | 🔨 | 5 |
| CRUX-A-02 | Pull a HF repo by `hf://org/name` | `huggingface-cli download` | ✅ | 5 |
| CRUX-A-03 | Pin to revision/branch/commit SHA | `--revision` | 🔨 | 4 |
| CRUX-A-04 | Filter files with include/exclude globs | `allow_patterns` | ❌ | 4 |
| CRUX-A-05 | Resume an interrupted download | `huggingface-cli resume` | 🔨 | 5 |
| CRUX-A-06 | Authenticate with `HF_TOKEN` or `apr login` | `huggingface-cli login` | ✅ | 5 |
| CRUX-A-07 | Xet-accelerated parallel download | `hf_xet` | 🔨 | 4 |
| CRUX-A-08 | Custom mirror via `HF_ENDPOINT` | HF env var | ❌ | 3 |
| CRUX-A-09 | `apr list/show/rm` local registry | `ollama list/show/rm` | 🔨 | 5 |
| CRUX-A-10 | VRAM-aware quant auto-select | Ollama auto-quant | ❌ | 4 |
| CRUX-A-11 | `apr cp` copy model with new tag | `ollama cp` | ❌ | 3 |
| CRUX-A-12 | `apr ps` list running models | `ollama ps` | ❌ | 4 |
| CRUX-A-13 | `apr stop` unload model from VRAM | `ollama stop` | ❌ | 4 |
| CRUX-A-14 | Pull from S3/GCS/Azure URL | ecosystem | ❌ | 3 |
| CRUX-A-15 | Pull from local directory (`file://`) | HF local path | 🔨 | 4 |
| CRUX-A-16 | Modelfile / recipe for custom system prompt | `ollama create` | ❌ | 4 |
| CRUX-A-17 | Manifest signing (cosign / sigstore) | ecosystem | ❌ | 2 |
| CRUX-A-18 | Gated-model interactive auth flow | HF gated | 🔨 | 3 |
| CRUX-A-19 | Progress bar with ETA + parallel chunks | HF hub | 🔨 | 4 |
| CRUX-A-20 | Offline mode (no network calls) | `TRANSFORMERS_OFFLINE=1` | 🔨 | 4 |
| CRUX-A-21 | Shared model cache across users | `ollama` linux shared dir | ❌ | 3 |
| CRUX-A-22 | Disk-quota enforcement on registry | `ollama` disk mgr | ❌ | 2 |
| CRUX-A-23 | Model search across Hub + local cache | `huggingface-cli search` | ❌ | 3 |
| CRUX-A-24 | Register custom local model in registry | `ollama create -f` | 🔨 | 3 |
| CRUX-A-25 | Garbage-collect unused models | ollama disk cleanup | ❌ | 3 |

### Category B — Format Conversion & Quantization (20 stories)

| ID | Story | Competitor verb | S | D |
|----|-------|----------------|---|---|
| CRUX-B-01 | Safetensors → GGUF preserving tokenizer | `convert_hf_to_gguf.py` | ✅ | 5 |
| CRUX-B-02 | GGUF → Safetensors for downstream PEFT | (missing upstream) | ❌ | 4 |
| CRUX-B-03 | PyTorch `.bin` → Safetensors sharded | `safetensors convert` | 🔨 | 4 |
| CRUX-B-04 | HF → APR native with LAYOUT-001/002 check | — | ✅ | 5 |
| CRUX-B-05 | Safetensors shard/merge with weight-map | `sharded_model` | 🔨 | 4 |
| CRUX-B-06 | All K-quants (Q2..Q8) + perplexity delta | `llama-quantize` | ✅ | 5 |
| CRUX-B-07 | imatrix calibration for K-quants | `llama-imatrix` | ❌ | 5 |
| CRUX-B-08 | AWQ quantization | `autoawq` | ❌ | 5 |
| CRUX-B-09 | GPTQ quantization | `auto-gptq` | ❌ | 5 |
| CRUX-B-10 | BitsAndBytes NF4 4-bit | `bitsandbytes` | ❌ | 4 |
| CRUX-B-11 | FP8 quantization (H100+) | vLLM FP8 | ❌ | 3 |
| CRUX-B-12 | INT8 dynamic quantization | `torch.ao.quantization` | 🔨 | 4 |
| CRUX-B-13 | INT4 static weight-only | llama.cpp Q4_0 | ✅ | 4 |
| CRUX-B-14 | Q4_K_M / Q5_K_M / Q6_K variants | llama.cpp | ✅ | 5 |
| CRUX-B-15 | IQ-quants (IQ3_XXS, IQ2_S) | llama.cpp | ❌ | 3 |
| CRUX-B-16 | Per-tensor layer-quant override | llama.cpp `--tensor-quant` | ❌ | 3 |
| CRUX-B-17 | Quantization-aware training (QAT) | PyTorch QAT | ❌ | 2 |
| CRUX-B-18 | Calibration dataset loader | llama.cpp imatrix | ❌ | 4 |
| CRUX-B-19 | Dequant + re-quant preserving metadata | — | 🔨 | 3 |
| CRUX-B-20 | Roundtrip quant diff report | — | 🔨 | 3 |

### Category C — Inference & Serving (35 stories)

| ID | Story | Competitor verb | S | D |
|----|-------|----------------|---|---|
| CRUX-C-01 | `apr run --prompt "..."` one-shot | `ollama run` / `llama-cli` | ✅ | 5 |
| CRUX-C-02 | `apr chat` interactive REPL streaming | `ollama run` REPL | ✅ | 5 |
| CRUX-C-03 | OpenAI-compatible `/v1/chat/completions` | `vllm serve` | 🔨 | 5 |
| CRUX-C-04 | Ollama-compatible `/api/chat` | Ollama REST | ❌ | 5 |
| CRUX-C-05 | SSE streaming tokens (`[DONE]` sentinel) | OpenAI stream | 🔨 | 5 |
| CRUX-C-06 | Continuous batching | vLLM | 🔨 | 5 |
| CRUX-C-07 | Paged-attention KV cache | vLLM PA | 🔨 | 5 |
| CRUX-C-08 | Automatic prefix caching | vLLM APC | 🔨 | 5 |
| CRUX-C-09 | Speculative decoding (draft model) | vLLM spec | ❌ | 4 |
| CRUX-C-10 | Grammar-constrained / GBNF output | llama.cpp grammar | 🔨 | 4 |
| CRUX-C-11 | OpenAI tool-use / function-calling | OpenAI tools | ❌ | 5 |
| CRUX-C-12 | Multi-modal vision input (LLaVA-style) | llama.cpp mmproj | ❌ | 4 |
| CRUX-C-13 | `/v1/embeddings` endpoint | llama-server `/embedding` | ❌ | 5 |
| CRUX-C-15 | Tensor + pipeline parallelism | vLLM TP/PP | ❌ | 4 |
| CRUX-C-16 | LoRA hotswap at runtime | vLLM LoRA | ❌ | 4 |
| CRUX-C-17 | Multi-LoRA serving (N adapters) | vLLM multi-lora | ❌ | 4 |
| CRUX-C-18 | Stop-sequence strings | Ollama `stop` | ✅ | 4 |
| CRUX-C-19 | Temperature / top_p / top_k sampling | Ollama options | ✅ | 5 |
| CRUX-C-20 | Repetition penalty | Ollama `repeat_penalty` | ✅ | 4 |
| CRUX-C-21 | Mirostat sampling | llama.cpp | ❌ | 3 |
| CRUX-C-22 | Typical sampling | llama.cpp `--typical` | ❌ | 3 |
| CRUX-C-23 | DRY sampling | llama.cpp `--dry` | ❌ | 3 |
| CRUX-C-24 | Beam search decoding | PyTorch beam | ❌ | 3 |
| CRUX-C-25 | Logprobs output | vLLM `logprobs` | 🔨 | 4 |
| CRUX-C-26 | Context-window extension (RoPE scaling) | llama.cpp `--rope-*` | 🔨 | 4 |
| CRUX-C-27 | GGUF lazy mmap loading | llama.cpp | ✅ | 5 |
| CRUX-C-28 | GPU layer offloading (`-ngl`) | llama.cpp `-ngl` | ✅ | 5 |
| CRUX-C-29 | NUMA-aware memory placement | llama.cpp `--numa` | ❌ | 3 |
| CRUX-C-30 | KV cache quantization (Q8_0 / Q4_0) | llama.cpp kv-quant | ❌ | 4 |
| CRUX-C-31 | FlashAttention-2 enabled path | vLLM FA2 | 🔨 | 5 |
| CRUX-C-32 | Chunked prefill for long contexts | vLLM chunked-prefill | 🔨 | 4 |
| CRUX-C-33 | `/v1/models` endpoint | vLLM / OpenAI | 🔨 | 5 |
| CRUX-C-34 | `/health` endpoint | vLLM | 🔨 | 5 |
| CRUX-C-35 | Graceful shutdown with in-flight drain | vLLM | 🔨 | 4 |
| CRUX-C-36 | Cancel in-flight requests (client disconnect) | vLLM cancellation | 🔨 | 4 |

> ID gap: **CRUX-C-14** dropped — rerank subsumed by C-13 embeddings.

### Category D — Fine-tuning & Training (35 stories)

| ID | Story | Competitor verb | S | D |
|----|-------|----------------|---|---|
| CRUX-D-01 | Full-parameter fine-tune single command | `trl sft` | 🔨 | 5 |
| CRUX-D-02 | LoRA fine-tune with rank/alpha/dropout | `peft LoraConfig` | 🔨 | 5 |
| CRUX-D-03 | QLoRA (4-bit base + LoRA) | bitsandbytes | ❌ | 5 |
| CRUX-D-04 | DPO preference training | `trl DPOTrainer` | ❌ | 5 |
| CRUX-D-05 | SFT on chat-template conversations | `trl SFTTrainer` | 🔨 | 5 |
| CRUX-D-06 | Load HF dataset + tokenize + pack | `datasets.load_dataset` | 🔨 | 5 |
| CRUX-D-07 | Checkpoint save/resume with opt state | `Trainer resume_from_checkpoint` | ✅ | 5 |
| CRUX-D-08 | Per-epoch eval JSON for CI gates | — | 🔨 | 4 |
| CRUX-D-09 | Gradient accumulation | `accumulation_steps` | ✅ | 5 |
| CRUX-D-10 | Mixed precision bf16 / fp16 | `torch.cuda.amp` | 🔨 | 5 |
| CRUX-D-11 | DDP multi-GPU single node | `torch.distributed` | ❌ | 5 |
| CRUX-D-12 | FSDP / ZeRO sharding | `accelerate fsdp` | ❌ | 5 |
| CRUX-D-13 | LR schedule (cosine, linear-warmup) | `get_scheduler` | ✅ | 5 |
| CRUX-D-14 | Early stopping + best-ckpt retention | `EarlyStoppingCallback` | 🔨 | 4 |
| CRUX-D-15 | Merge trained LoRA + export GGUF | `peft merge_and_unload` | 🔨 | 5 |
| CRUX-D-16 | AdamW + 8-bit AdamW optimizer | bitsandbytes | 🔨 | 5 |
| CRUX-D-17 | ORPO / KTO / IPO alignment | TRL | ❌ | 3 |
| CRUX-D-18 | Reward-model training | TRL reward | ❌ | 3 |
| CRUX-D-19 | PPO reinforcement learning | TRL PPO | ❌ | 3 |
| CRUX-D-20 | RoPE frequency scaling during SFT | HF config | ❌ | 3 |
| CRUX-D-21 | Continue-pretraining on raw corpus | HF CLM | 🔨 | 4 |
| CRUX-D-22 | Token-level loss masking (assistant-only) | TRL `completion_only_loss` | 🔨 | 4 |
| CRUX-D-23 | Gradient checkpointing for memory | `gradient_checkpointing=True` | 🔨 | 5 |
| CRUX-D-24 | Activation checkpointing tags | PyTorch `checkpoint` | ❌ | 3 |
| CRUX-D-25 | Weight tying across layers | PyTorch `tie_weights` | 🔨 | 3 |
| CRUX-D-26 | Learning-rate finder (range test) | fastai LR finder | ❌ | 3 |
| CRUX-D-27 | Hyperparameter sweep (ray/optuna) | ray tune | ❌ | 3 |
| CRUX-D-28 | Adapter fusion (combine N LoRAs) | PEFT adapter fusion | ❌ | 3 |
| CRUX-D-29 | Pre-compiled dataset cache | HF datasets cache | ❌ | 3 |
| CRUX-D-30 | Multi-task training loop | custom HF | ❌ | 3 |
| CRUX-D-31 | TensorBoard / wandb integration | HF `report_to` | ❌ | 4 |
| CRUX-D-32 | Resume from HF Hub checkpoint URL | HF hub ckpt | 🔨 | 3 |
| CRUX-D-33 | Distributed optimizer (ZeRO-1) | Deepspeed | ❌ | 4 |
| CRUX-D-34 | Deepspeed config loader | Deepspeed | ❌ | 3 |
| CRUX-D-35 | `accelerate launch` wrapper | HF accelerate | ❌ | 4 |

### Category E — Evaluation & Benchmark (25 stories)

| ID | Story | Competitor verb | S | D |
|----|-------|----------------|---|---|
| CRUX-E-01 | pass@1 HumanEval / MBPP sandboxed | `bigcode-evaluation-harness` | 🔨 | 5 |
| CRUX-E-02 | Perplexity on held-out corpus | `llama-perplexity` | ❌ | 5 |
| CRUX-E-03 | lm-eval-harness tasks | `lm-evaluation-harness` | ❌ | 5 |
| CRUX-E-04 | A/B compare two models (win rate) | — | 🔨 | 4 |
| CRUX-E-05 | Ollama-parity decode bench (128-tok median) | `ollama run --verbose` | ✅ | 5 |
| CRUX-E-06 | Peak RSS + VRAM during generate | `nvidia-smi` watch | 🔨 | 5 |
| CRUX-E-07 | Latency P50/P95/P99 under load | `vllm bench` | ❌ | 5 |
| CRUX-E-08 | Golden-output regression gate | — | ✅ | 5 |
| CRUX-E-09 | Per-layer tensor cosine diff | `apr diff` | ✅ | 4 |
| CRUX-E-10 | Hallucination / drift detector on logs | HELM | ❌ | 3 |
| CRUX-E-11 | MT-Bench / arena judge eval | `fastchat` | ❌ | 4 |
| CRUX-E-12 | BBH / MMLU / HellaSwag per-task | HF evaluate | ❌ | 4 |
| CRUX-E-13 | RULER long-context eval | Nvidia RULER | ❌ | 3 |
| CRUX-E-14 | Needle-in-haystack context recall | greg_kamradt | ❌ | 4 |
| CRUX-E-15 | Speed head-to-head vs llama.cpp | — | 🔨 | 4 |
| CRUX-E-16 | TTFT (time to first token) latency | vLLM bench | 🔨 | 5 |
| CRUX-E-17 | Tokens/sec vs concurrency curve | vLLM bench | 🔨 | 5 |
| CRUX-E-18 | Throughput at max batch | vLLM bench | 🔨 | 5 |
| CRUX-E-19 | Perplexity per quant bit-budget | llama.cpp | ❌ | 4 |
| CRUX-E-20 | KL divergence vs FP16 baseline | llama.cpp | ❌ | 4 |
| CRUX-E-21 | Bias/toxicity eval harness | HF `evaluate-bias` | ❌ | 3 |
| CRUX-E-22 | Code-eval sandbox (Docker runner) | `bigcode-evaluation-harness` | ❌ | 4 |
| CRUX-E-23 | Tool-use benchmark (BFCL) | Berkeley FC | ❌ | 3 |
| CRUX-E-24 | RAG eval (RAGAS / TruLens) | ecosystem | ❌ | 3 |
| CRUX-E-25 | Vision-language benchmark harness | OpenCLIP eval | ❌ | 2 |

### Category F — Debug & Analysis (20 stories)

| ID | Story | Competitor verb | S | D |
|----|-------|----------------|---|---|
| CRUX-F-01 | `apr tensors` shape + dtype + stats | `gguf-dump` | ✅ | 5 |
| CRUX-F-02 | `apr trace` layer-by-layer activations | `realizar --trace` | ✅ | 4 |
| CRUX-F-03 | LAYOUT shape validation pre-load | LAYOUT-001/002 | ✅ | 5 |
| CRUX-F-04 | Quantization error per tensor, ranked | — | 🔨 | 4 |
| CRUX-F-05 | Roofline profiling (mem vs compute bound) | `apr profile` | ✅ | 4 |
| CRUX-F-06 | KV-cache utilization timeline | vLLM metrics | ❌ | 4 |
| CRUX-F-07 | GPU memory timeline → Chrome trace | `torch.profiler` | ❌ | 4 |
| CRUX-F-08 | Loss curve visualization | TensorBoard | 🔨 | 4 |
| CRUX-F-09 | Gradient-norm telemetry per step | wandb | ❌ | 5 |
| CRUX-F-11 | NaN/Inf detector in activations | PyTorch `detect_anomaly` | ❌ | 4 |
| CRUX-F-12 | Tensor shape mismatch explainer | PyTorch traceback | ✅ | 4 |
| CRUX-F-13 | CUDA OOM postmortem report | `torch.cuda.memory_summary` | ❌ | 5 |
| CRUX-F-14 | Deadlock / hang detector with stack dump | PyTorch hang detector | ❌ | 3 |
| CRUX-F-15 | NCCL failure diagnostics | NCCL `NCCL_DEBUG=INFO` | ❌ | 3 |
| CRUX-F-16 | Kernel timing with nsys / nvprof export | Nsight | 🔨 | 3 |
| CRUX-F-17 | Attention pattern visualization | bertviz | ❌ | 3 |
| CRUX-F-18 | Token embedding 2D projection (UMAP) | tensorboard projector | ❌ | 2 |
| CRUX-F-19 | `apr explain` token selection rationale | — | ❌ | 3 |
| CRUX-F-20 | GGUF metadata dump | `gguf-dump.py` | ✅ | 5 |
| CRUX-F-21 | `apr qa` 8-gate golden-test runner | — | ✅ | 5 |

> ID gap: **CRUX-F-10** dropped — activation histograms subsumed by F-04 + F-09.

### Category G — Publishing & Distribution (15 stories)

| ID | Story | Competitor verb | S | D |
|----|-------|----------------|---|---|
| CRUX-G-01 | Publish model ≤ 5 GB to HF Hub | `huggingface-cli upload` | ✅ | 5 |
| CRUX-G-02 | Publish large model via Xet / LFS | `hf_xet` | ✅ | 5 |
| CRUX-G-03 | Auto-generate model card from contract | `modelcards` | 🔨 | 4 |
| CRUX-G-04 | Auto-generate README with usage examples | — | ❌ | 3 |
| CRUX-G-05 | Checksum manifest (SHA256) per artifact | — | 🔨 | 4 |
| CRUX-G-06 | Reproducibility manifest (env/seed/commit) | — | 🔨 | 4 |
| CRUX-G-07 | SemVer tag on `apr publish` | `huggingface-cli tag` | ❌ | 3 |
| CRUX-G-08 | Private repo upload | HF `private=True` | ✅ | 5 |
| CRUX-G-09 | Multi-file atomic commit | HF commit API | 🔨 | 4 |
| CRUX-G-10 | Org-scoped publish with permissions | HF org perms | 🔨 | 3 |
| CRUX-G-11 | Ollama-style push to remote registry | `ollama push` | ❌ | 4 |
| CRUX-G-12 | Verify upload integrity via CAS hash | HF Xet term verify | ✅ | 4 |
| CRUX-G-13 | Publish tokenizer/config bundle | HF | 🔨 | 5 |
| CRUX-G-14 | License metadata validator | HF model-index | ❌ | 3 |
| CRUX-G-15 | Duplicate-artifact detector pre-push | — | ❌ | 3 |

### Category H — Data Pipeline (20 stories)

| ID | Story | Competitor verb | S | D |
|----|-------|----------------|---|---|
| CRUX-H-01 | Load HF dataset by name + split | `load_dataset` | 🔨 | 5 |
| CRUX-H-02 | Tokenize with truncation policy | `tokenizer(...)` | ✅ | 5 |
| CRUX-H-03 | Packing for efficient SFT | `trl ConstantLengthDataset` | ❌ | 5 |
| CRUX-H-05 | Train/val/test split deterministic | `train_test_split` | ✅ | 5 |
| CRUX-H-06 | Streaming datasets (no RAM mat) | `streaming=True` | ❌ | 4 |
| CRUX-H-07 | Shuffling DataLoader + bucketed lens | PyTorch | ✅ | 4 |
| CRUX-H-08 | Apply chat template per turn | `apply_chat_template` | ✅ | 5 |
| CRUX-H-09 | Parquet / Arrow ingest | `datasets` | 🔨 | 4 |
| CRUX-H-10 | JSONL dataset loader | `load_dataset('json')` | ✅ | 5 |
| CRUX-H-11 | Instruction auto-format (alpaca, sharegpt) | TRL | 🔨 | 4 |
| CRUX-H-12 | ImageFolder loader | torchvision | ❌ | 3 |
| CRUX-H-13 | Audio dataset loader (WAV/FLAC) | torchaudio | ❌ | 2 |
| CRUX-H-14 | Dataset mixing with per-source weights | HF interleave | ❌ | 3 |
| CRUX-H-15 | DatasetDict multi-split serialization | HF `save_to_disk` | 🔨 | 3 |
| CRUX-H-16 | Tokenizer-aware length bucketing | HF dynamic batching | ❌ | 3 |
| CRUX-H-17 | Contrastive pair sampler for CLIP | OpenCLIP | ❌ | 3 |
| CRUX-H-18 | Negative sampling for ranking | PyTorch | ❌ | 2 |
| CRUX-H-19 | Synthetic data generation via LLM | ecosystem | ❌ | 3 |
| CRUX-H-20 | PII redaction in datasets | ecosystem | ❌ | 2 |
| CRUX-H-21 | Row-level deduplication (MinHash) | `text-dedup` | ❌ | 3 |

> ID gap: **CRUX-H-04** dropped — filter+dedupe subsumed by H-01 + H-21.

### Category I — MCP, Agents & Tool Integration (15 stories)

| ID | Story | Competitor verb | S | D |
|----|-------|----------------|---|---|
| CRUX-I-01 | MCP server exposing `apr` tools | `mcp serve` | ✅ | 5 |
| CRUX-I-02 | MCP client consuming external tools | MCP client | 🔨 | 4 |
| CRUX-I-03 | OpenAI-style tool calls + JSON-schema | OpenAI tools | 🔨 | 5 |
| CRUX-I-04 | Ollama-style function calling | Ollama tools | ❌ | 5 |
| CRUX-I-06 | ReAct agent loop + stop conditions | Langgraph | ❌ | 4 |
| CRUX-I-07 | Claude Agent SDK compatibility | Anthropic SDK | ❌ | 3 |
| CRUX-I-08 | Streaming tool-call deltas | OpenAI `delta.tool_calls` | 🔨 | 4 |
| CRUX-I-09 | Parallel tool-calls in one turn | OpenAI parallel | ❌ | 4 |
| CRUX-I-10 | Tool-result injection into chat history | OpenAI `tool` role | 🔨 | 4 |
| CRUX-I-11 | Schema-coerced JSON output | OpenAI JSON mode | 🔨 | 4 |
| CRUX-I-12 | GBNF grammar compiler from JSON-schema | llama.cpp `grammars/` | ❌ | 4 |
| CRUX-I-13 | MCP resource provider | MCP spec | ❌ | 3 |
| CRUX-I-14 | MCP prompt provider | MCP spec | ❌ | 3 |
| CRUX-I-15 | Agent memory plugin interface | ecosystem | ❌ | 3 |
| CRUX-I-16 | Guardrails / output filter pipeline | ecosystem | ❌ | 3 |

> ID gap: **CRUX-I-05** dropped — JSON-mode subsumed by C-10 grammar.

### Category J — OpenCLAW Agent Orchestration (20 stories)

> Resolved 2026-04-18: competitor identity is **openclaw.ai** (local-first
> personal AI assistant / agent orchestration layer), NOT OpenCLIP. Each
> story is a provable contract at `contracts/crux-J-NN-v1.yaml`
> (`openclaw_interpretation: openclaw-agent-resolved`, `version: 1.1.0`).
> Vision-language parity, if needed later, will live in a separate
> category / sibling subspec — see §10.

| ID | Story | Competitor verb / surface | S | D |
|----|-------|----------------|---|---|
| CRUX-J-01 | Install + onboard one-liner | `curl openclaw.ai/install.sh \| bash` + `openclaw onboard` | ❌ | 5 |
| CRUX-J-02 | User config JSON5 round-trip | `~/.openclaw/openclaw.json` | ❌ | 5 |
| CRUX-J-03 | Per-channel allowFrom sender allowlist | `channels.*.allowFrom` (deny by default) | ❌ | 4 |
| CRUX-J-04 | Group-chat @mention trigger | `groupChat.mentionRequired` | ❌ | 4 |
| CRUX-J-05 | Dashboard Control UI (loopback bind) | `openclaw dashboard` on 127.0.0.1:18789 | ❌ | 3 |
| CRUX-J-06 | Daemon install ∘ uninstall dual | `openclaw onboard --install-daemon` | ❌ | 4 |
| CRUX-J-07 | Per-sender session isolation | per-sender RPC session | ❌ | 3 |
| CRUX-J-08 | System-control shell.exec + SSC classifier gate | `tools.shell.exec` (SSC-gated) | ❌ | 3 |
| CRUX-J-09 | Browser automation via MCP client | `tools.browser` via `mcp://` provider | ❌ | 5 |
| CRUX-J-10 | Persistent memory put/get round-trip | `openclaw memory put/get [--ttl]` | ❌ | 4 |
| CRUX-J-11 | Skill system / extensible capabilities | `openclaw skills add <pkg>` | ❌ | 4 |
| CRUX-J-12 | Multi-transport chat dispatch | WhatsApp/Telegram/Discord/Slack/Signal/iMessage adapters | ❌ | 3 |
| CRUX-J-13 | LLM provider switching | `llm.provider` ∈ {claude, openai, local} | ❌ | 2 |
| CRUX-J-14 | Onboard safety prompts for destructive ops | `openclaw onboard --safe` | ❌ | 2 |
| CRUX-J-15 | Auto-update / self-modify | `openclaw update` | ❌ | 3 |
| CRUX-J-16 | Event log / audit trail | `~/.openclaw/audit.log` NDJSON | ❌ | 2 |
| CRUX-J-17 | Rate limiting per sender | `rateLimit.perSender.msgsPerMin` | ❌ | 3 |
| CRUX-J-18 | Encrypted credentials (OS keychain) | `credentials.backend = keychain` | ❌ | 3 |
| CRUX-J-19 | Offline / local-first fallback | `offline.enabled + fallbackProvider=local` | ❌ | 2 |
| CRUX-J-20 | Claude-Code MCP tool-call envelope parity | MCP tool_use_id / content schema | ❌ | 4 |

### Category K — Ecosystem Interop (20 stories)

| ID | Story | Competitor verb | S | D |
|----|-------|----------------|---|---|
| CRUX-K-01 | OpenAI Python SDK test suite | OpenAI SDK | 🔨 | 5 |
| CRUX-K-02 | Ollama Python SDK test suite | Ollama SDK | ❌ | 5 |
| CRUX-K-03 | Langchain `ChatOpenAI(base_url=...)` | Langchain | 🔨 | 5 |
| CRUX-K-04 | LlamaIndex LLM provider | LlamaIndex | 🔨 | 4 |
| CRUX-K-05 | `apr ui` Gradio web UI | Gradio | ❌ | 3 |
| CRUX-K-07 | Prometheus `/metrics` endpoint | vLLM Prom | ❌ | 4 |
| CRUX-K-08 | OpenTelemetry traces | OTEL | ❌ | 4 |
| CRUX-K-09 | Safetensors metadata round-trip | safetensors spec | ✅ | 5 |
| CRUX-K-10 | GGUF `general.*` metadata round-trip | llama.cpp | 🔨 | 4 |
| CRUX-K-11 | Modelfile DSL parser (Ollama) | `ollama create -f` | ❌ | 3 |
| CRUX-K-12 | VSCode extension for apr | ecosystem | ❌ | 3 |
| CRUX-K-13 | Docker image (`apr-serve:latest`) | ecosystem | ❌ | 4 |
| CRUX-K-14 | systemd unit for `apr serve` | ecosystem | ❌ | 3 |
| CRUX-K-15 | Kubernetes Helm chart | ecosystem | ❌ | 3 |
| CRUX-K-16 | Triton Inference Server backend | NVIDIA Triton | ❌ | 2 |
| CRUX-K-17 | NVIDIA Dynamo integration | NVIDIA | ❌ | 2 |
| CRUX-K-18 | ONNX Runtime backend | ORT | ❌ | 3 |
| CRUX-K-19 | CoreML export (Apple Silicon) | HF optimum | ❌ | 3 |
| CRUX-K-20 | TensorRT-LLM export | TRT-LLM | ❌ | 3 |
| CRUX-K-21 | MLX backend (Apple Silicon native) | mlx | ❌ | 3 |

> ID gap: **CRUX-K-06** dropped — Jupyter `%%apr` magic deferred (ecosystem nice-to-have).

### Category L — HF kernels-community Integration (15 stories)

> Added v2.2 (2026-04-21). Competitor source: [huggingface.co/kernels-community](https://huggingface.co/kernels-community). Canonical verb: `from kernels import get_kernel`. Aprender target surface: `apr kernels pull` + trueno/aprender-compute kernel-loader trait. Full derivation in §13.1.

| ID | Story | Competitor verb | S | D |
|----|-------|----------------|---|---|
| CRUX-L-01 | Load pre-built HF kernel by `hf://kernels-community/<name>` | `kernels.get_kernel` | ❌ | 4 |
| CRUX-L-02 | Drop-in `flash-attn2` HF kernel behind aprender FA trait | `kernels-community/flash-attn2` | ❌ | 4 |
| CRUX-L-03 | Drop-in `flash-attn3` HF kernel (sm_90a / sm_100) | `kernels-community/flash-attn3` | ❌ | 4 |
| CRUX-L-04 | `rmsnorm` fused kernel behind LayerNorm trait | `kernels-community/rmsnorm` | ❌ | 3 |
| CRUX-L-05 | `rotary` fused RoPE-apply kernel | `kernels-community/rotary` | ❌ | 3 |
| CRUX-L-06 | `paged-attention` KV-cache kernel (vLLM-compat) | `kernels-community/paged-attention` | ❌ | 4 |
| CRUX-L-07 | `fp8-fbgemm` matmul path for H100+ | `kernels-community/fp8-fbgemm` | ❌ | 3 |
| CRUX-L-08 | `quantization-bitsandbytes` NF4 via HF kernel | `kernels-community/quantization-bitsandbytes` | ❌ | 3 |
| CRUX-L-09 | `quantization-gptq` fused dequant+matmul | `kernels-community/quantization-gptq` | ❌ | 3 |
| CRUX-L-10 | `liger-kernels` fused CE-loss for training | `kernels-community/liger-kernels` | ❌ | 4 |
| CRUX-L-11 | `megablocks` MoE routing kernel | `kernels-community/megablocks` | ❌ | 3 |
| CRUX-L-12 | `punica-sgmv` multi-LoRA fused SGMV | `kernels-community/punica-sgmv` | ❌ | 3 |
| CRUX-L-13 | `activation` fused SwiGLU/GeGLU kernel | `kernels-community/activation` | ❌ | 3 |
| CRUX-L-14 | `mamba-ssm` state-space kernel | `kernels-community/mamba-ssm` | ❌ | 2 |
| CRUX-L-15 | Kernel version-pinning + SHA-verified load | `kernels.get_kernel(..., revision=...)` | ❌ | 4 |

### Category M — APR-QA Playbook Canonicalization (10 stories)

> Added v2.2 (2026-04-21). Competitor source: sibling repo [`apr-model-qa-playbook`](https://github.com/paiml/apr-model-qa-playbook) — Popperian property-based QA framework. `apr qa` (F-21 ✅) is the entry point; Category M decomposes the playbook's 8 gates plus two meta-gates (5-Whys root-cause, property-based fuzz) into first-class CRUX stories so the playbook's falsifiers bind to the registry. Full derivation in §13.2.

| ID | Story | Competitor verb | S | D |
|----|-------|----------------|---|---|
| CRUX-M-01 | Gate-1 byte-identical vs safetensors ground truth | `apr-qa gate-byte-identical` | 🔨 | 5 |
| CRUX-M-02 | Gate-2 tensor-stats parity (min/max/mean/std, per-tensor) | `apr-qa gate-tensor-stats` | 🔨 | 5 |
| CRUX-M-03 | Gate-3 golden-output determinism @ seed=0 | `apr qa --require-golden-output` | ✅ | 5 |
| CRUX-M-04 | Gate-4 cross-format parity APR ↔ GGUF ↔ safetensors | `apr-qa gate-cross-format` | 🔨 | 5 |
| CRUX-M-05 | Gate-5 tokenizer round-trip with NFC normalization | `apr-qa gate-tokenizer` | 🔨 | 4 |
| CRUX-M-06 | Gate-6 chat-template rendering equivalence | `apr-qa gate-chat-template` | 🔨 | 4 |
| CRUX-M-07 | Gate-7 pass@1 on canary suite with confidence interval | `apr-qa gate-canary` | 🔨 | 5 |
| CRUX-M-08 | Gate-8 5-Whys root-cause generator on failure | `apr-qa jidoka-trace` | 🔨 | 4 |
| CRUX-M-09 | Property-based falsifier ≥ 1000 fuzz cases per gate | `apr-qa fuzz --cases 1000` | ❌ | 4 |
| CRUX-M-10 | Upstream-fix enforcement (reject workarounds; route to aprender/trueno/realizar) | playbook "no-workarounds" rule | 🔨 | 4 |

### Category O — AutoML Parity, AutoGluon (24 stories)

> Added v2.3 (2026-09-16). Competitor source: [autogluon/autogluon](https://github.com/autogluon/autogluon) 1.6.3, surveyed from `../autogluon` @ 77946149; evidence in `evidence/crux/autogluon/`. Canonical verb: `TabularPredictor(label="class").fit("train.csv", presets="best")` — the README's only code block. Aprender target surface: `apr automl fit|predict|leaderboard` over a predictor-level AutoML in `aprender-core::automl` (today a single-estimator tuner) and `apr forecast` over a panel `TimeSeriesPredictor` (today one univariate `ARIMA`). Epic aprender#3370; one GitHub issue per row (#3371–#3394). `MultiModalPredictor`, `autogluon.cloud`, MLZero and Ray-parallel fits are CUT on the epic. Not a BEAT pillar.

| ID | Story | Competitor verb | S | D |
|----|-------|----------------|---|---|
| CRUX-O-01 | One-call tabular AutoML: fit(label) -> predict on a CSV | `TabularPredictor(label="class").fit("train.csv"); predictor.predict("test.csv")` | ❌ | 5 |
| CRUX-O-02 | Problem-type inference: binary / multiclass / regression / quantile from the label column | `predictor.problem_type  (inferred in fit unless problem_type= given)` | ❌ | 5 |
| CRUX-O-03 | Quality presets (medium / good / high / best / extreme) that name a model portfolio and a time budget | `fit(..., presets="best_quality")` | ❌ | 5 |
| CRUX-O-04 | Leaderboard: per-model validation/test score, fit time, predict time and stack level | `predictor.leaderboard(test_data, extra_info=True)` | ❌ | 5 |
| CRUX-O-05 | Automatic feature-type inference and the AutoML feature pipeline (numeric, categorical, datetime, text n-gram, drop-unique, drop-duplicate) | `AutoMLPipelineFeatureGenerator` | 🔨 | 5 |
| CRUX-O-06 | K-fold bagging with out-of-fold predictions (num_bag_folds, predict_oof) | `fit(..., num_bag_folds=8); predictor.predict_proba_oof()` | ❌ | 4 |
| CRUX-O-07 | Multi-layer stack ensembling with a leakage guard (num_stack_levels, auto_stack, dynamic_stacking) | `fit(..., num_stack_levels=1, dynamic_stacking="auto")` | ❌ | 4 |
| CRUX-O-08 | Greedy weighted-ensemble selection over fitted models (Caruana ensemble selection) | `fit_weighted_ensemble=True (default)` | ❌ | 5 |
| CRUX-O-09 | Time-budgeted portfolio fit: time_limit split across models, each model early-stopped on its share | `fit(..., time_limit=3600)` | 🔨 | 5 |
| CRUX-O-10 | Predictor-level permutation feature importance with p-values and confidence intervals | `predictor.feature_importance(test_data, num_shuffle_sets=10)` | 🔨 | 4 |
| CRUX-O-11 | Decision-threshold calibration for binary metrics (calibrate_decision_threshold) | `fit(..., calibrate_decision_threshold="auto"); predictor.calibrate_decision_threshold(metric="f1")` | 🔨 | 4 |
| CRUX-O-12 | refit_full: retrain the selected models on train+validation after model selection | `fit(..., refit_full=True, set_best_to_refit_full=True); predictor.refit_full()` | ❌ | 3 |
| CRUX-O-13 | Model distillation: compress the ensemble into one fast student | `predictor.distill(time_limit=..., augment_method="spunge")` | ❌ | 3 |
| CRUX-O-14 | Deployment artifact: clone_for_deployment / keep_only_best / save_space / persist into one loadable file | `predictor.clone_for_deployment(path); predictor.persist()` | ❌ | 4 |
| CRUX-O-15 | Inference-latency constraint during model selection (infer_limit, infer_limit_batch_size) | `fit(..., infer_limit=0.001, infer_limit_batch_size=10000)` | ❌ | 3 |
| CRUX-O-16 | Fit diagnostics: fit_summary, model_failures and learning curves | `predictor.fit_summary(); predictor.model_failures(); fit(..., learning_curves=True)` | ❌ | 3 |
| CRUX-O-17 | Tabular foundation model: in-context prediction with a pretrained transformer (TabPFN / TabICL / Mitra class) | `hyperparameters={"TABPFNV2": {}, "TABICL": {}, "MITRA": {}}` | ❌ | 4 |
| CRUX-O-18 | Memory-aware fit: per-model memory estimate and a memory_limit that skips models that would not fit | `fit(..., memory_limit="auto")` | ❌ | 3 |
| CRUX-O-19 | Multi-series forecasting predictor: (item_id, timestamp) panel data, prediction_length, freq | `TimeSeriesPredictor(prediction_length=48, freq="h").fit(TimeSeriesDataFrame)` | ❌ | 5 |
| CRUX-O-20 | Probabilistic forecasts: quantile_levels and the forecasting metric family (WQL, MQL, MASE, SMAPE, RMSSE, WAPE) | `TimeSeriesPredictor(eval_metric="WQL", quantile_levels=[0.1,0.5,0.9])` | ❌ | 5 |
| CRUX-O-21 | Rolling-window backtesting: num_val_windows, refit_every_n_windows, backtest_predictions | `fit(..., num_val_windows="auto", refit_every_n_windows="auto"); predictor.backtest_predictions()` | ❌ | 4 |
| CRUX-O-22 | Known covariates, past covariates and static features in forecasting | `TimeSeriesPredictor(known_covariates_names=["holiday"]); train_data.static_features = df` | ❌ | 3 |
| CRUX-O-23 | Local statistical baselines: SeasonalNaive, ETS, Theta, AutoARIMA, Croston | `hyperparameters={"SeasonalNaive": {}, "AutoETS": {}, "Theta": {}, "AutoARIMA": {}, "Croston": {}}` | 🔨 | 4 |
| CRUX-O-24 | Zero-shot pretrained forecaster (Chronos-2 / Toto-2 class) loaded by apr pull, with optional fine-tuning | `hyperparameters={"Chronos2": {"fine_tune": True}}` | ❌ | 4 |

### Category R — SHACL ontologies vs Apache Jena (4 stories)

> Added v2.4 (2026-10-09). Competitor: [Apache Jena](https://jena.apache.org/) 5.6.0, the Java RDF toolkit, pinned by sha256 in `tests/oracle/jena/src/jena.rs`. One story per cell of §14.2; §14 is the design. The stories are measured by the harness `contracts/crux-shacl-jena-v1.yaml` (`tests/oracle/jena/`), on demand and at night, never on a PR: no JVM is reachable from pv (R-13). Each status comes from the harness's first run, `tests/oracle/jena/receipt.json`. Epic aprender#3598. Not a BEAT pillar. Demand scores rate each verb's traffic inside pv, because no issue-volume count exists for Jena.

| ID | Story | Competitor verb | S | D |
|----|-------|----------------|---|---|
| CRUX-R-01 | Read Turtle and N-Triples ontologies, blank nodes included; a malformed file is refused at the same line | `riot --output=nt X.ttl` | ❌ | 3 |
| CRUX-R-02 | Write a graph Jena reads back to the same graph (riot reads pv's `contracts.nt` and `shapes.ttl` clean; the round trip waits on R-01) | `riot --output=<syntax> X` | 🔨 | 3 |
| CRUX-R-03 | Validate SHACL Core to the same verdict as Jena, with W3C's expected report deciding on the W3C suite (corpus RED: the `sh:closed` defect #4837) | `shacl validate --shapes S.ttl --data D.ttl` | 🔨 | 4 |
| CRUX-R-04 | Report SHACL results as a W3C `sh:ValidationReport` with the full tuple (focus, path, value, source shape, component, severity) and a message | `shacl validate` (the report it prints) | ❌ | 4 |

**Total: 275 stories in §5 (v2.2) + 24 in Category O + 4 in Category R = 303 documented rows; the master registry holds 295** (Categories L and M have contracts but no registry rows — pre-existing drift recorded on aprender#3146) across 16 categories; 5 ID gaps (`C-14`, `F-10`, `H-04`, `I-05`, `K-06`) intentional and documented.

---

## 6. Coverage Summary (v2.2 intake)

Counts verified from §5 table (via `awk` emoji extraction). Δ columns show v2.1 → v2.2 movement from the L+M additions.

| Status | Count | Δ v2.1→v2.2 | % | Meaning |
|--------|-------|-------------|---|---------|
| ✅ supported | 39 | +1 (M-03) | 14.2 % | Golden test passes today |
| 🔨 partial   | 80 | +8 (M-01/02/04..08/10) | 29.1 % | Exists; needs polish/flags/alias |
| ❌ missing   | 156 | +16 (15 × L + M-09) | 56.7 % | Implementation ticket required |
| 🤔 unclear   | 0  | 0 | 0.0 % | — |
| **total**    | **275** | +25 | 100 % | |

> **v2.3 note (2026-09-16):** the master contract's `coverage_intake` is now asserted from `stories[]` by the aprender-contracts test-suite and reads supported 43 / partial 77 / missing 171 / total 291 after Category O (+5 partial, +19 missing). This table is the v2.2 intake and is left as history; the YAML is the source of truth.

> **v2.4 note (2026-10-09):** after Category R (+2 partial, +2 missing) the master contract's `coverage_intake` reads supported 43 / partial 79 / missing 173 / total 295, recounted from `stories[]` by status. No test asserts it: a search of `crates/`, `scripts/` and the `Makefile` on 2026-10-09 found no reader of `coverage_intake`, so the v2.3 note's claim that the test-suite asserts it does not hold today, and the count is kept by hand.

Demand-weighted view — **high-demand (D≥4)** stories still ❌ missing are
the fast path to adoption parity and become the first `pmat work` items
(see §12). Exact D-tier counts are regenerated by the falsification harness
at Phase 0 exit; intake estimates:

| Demand | ❌ missing (est.) | Priority |
|--------|------------------|----------|
| 5 (canonical) | ~18 | **P0 — hurts adoption most** |
| 4 (high) | ~45 | P1 |
| 3 (moderate) | ~55 | P2 |
| ≤2 (niche) | ~22 | P3 |

---

## 7. Implementation Roadmap

**Phase 0 — Contract skeletons** ✅ **COMPLETE (2026-04-18)**
- ✅ Created 250 sub-contract YAMLs at `contracts/crux-{ID}-v1.yaml`
  via deterministic generator `scripts/crux_scaffold_contracts.py`
- ✅ Registered all in master `contracts/crux-competitive-research-ux-v1.yaml`
- ✅ Created 140 pmat work tickets (one per ❌ missing story) via
  `scripts/crux_bulk_pmat_work.sh` — tagged `crux,gap,crux-{category},competitor-{name},{id_lower}`
- ✅ **250 of 250 contracts (100%)** promoted draft → **spec-complete**
  via five parallel waves + wave 6 batches 1-10 direct authoring + J-series
  OpenCLAW rewrite (2026-04-18). The 20-contract J-series was re-authored
  from the earlier OpenCLIP draft to OpenCLAW agent-orchestration semantics
  after the identity resolution recorded in §10.
  - Wave 1 (18 demand=5 missing): B-07/08/09, C-04/11/13, D-03/04/11/12,
    E-02/03/07, H-03, F-09/13, I-04, K-02
  - Wave 2 (24 demand=5 partials): A-01/05/09, C-03/05/06/07/08/31/33/34,
    D-01/02/05/06/10/16, E-01/06/16/17/18, H-01, K-01
  - Wave 3 (29 demand=5 supported — parity verification): A-02/06,
    B-01/04/06/14, C-01/02/19/27/28, D-07/09/13, E-05/08, F-01/03/20/21,
    G-01/02/08, H-02/05/08/10, I-01, K-09
  - Wave 4 (5 demand=5 partials + 32 demand=4 missing):
    D-15, D-23, G-13, I-03, K-03, A-04/10/12/13/16, B-02/10/18,
    C-09/12/15/16/17/30, D-31/33/35, E-11/12/14/19/20/22,
    F-06/07/11, G-11, H-06, I-06/09/12, K-07
  - Wave 5 (45 demand=4 non-J — all remaining): A-03/07/15/19/20,
    B-03/05/12/13, C-10/18/20/25/26/32/35/36, D-08/14/21/22,
    E-04/09/15, F-02/04/05/08/12, G-03/05/06/09/12, H-07/09/11,
    I-02/08/10/11, K-04/08/10/13
  - Wave 6 batch 1 (8 A-series demand=3): A-08 (HF_ENDPOINT),
    A-11 (apr cp), A-14 (s3/gs/az://), A-18 (gated auth flow),
    A-21 (shared APR_MODELS), A-23 (search), A-24 (create --from),
    A-25 (rm + gc refcount). Direct-authored on main conversation
    after wave 6 sub-agents hit API rate limits.
  - Wave 6 batch 2 (10 B+C demand=3): B-11 (FP8 sm90+),
    B-15 (IQ3_XXS / IQ2_S imatrix), B-16 (per-tensor qtype override),
    B-19 (dequant→requant metadata preservation),
    B-20 (quant-roundtrip RMSE diff vs llama-quantize-stats),
    C-21 (mirostat v2 convergence), C-22 (typical-p vs HF warper),
    C-23 (DRY sampling), C-24 (beam search vs HF top-1),
    C-29 (NUMA-bind with numastat miss-count proof).
  - Wave 6 batch 3 (13 D-series demand=3): D-17 (ORPO/KTO/IPO),
    D-18 (reward-model training BT loss), D-19 (PPO clip+KL),
    D-20 (RoPE scaling linear/dynamic/ntk/yarn), D-24 (activation
    checkpoint tags), D-25 (weight tying share storage),
    D-26 (LR finder Smith 1506.01186), D-27 (HP sweep ASHA/TPE),
    D-28 (LoRA fusion linear/cat/ties/dare), D-29 (dataset cache key),
    D-30 (multi-task interleave), D-32 (resume from hf://repo@rev),
    D-34 (Deepspeed config loader).
  - Wave 6 batch 4 (5 E-series demand=3): E-10 (hallucination/drift
    SelfCheckGPT+PSI), E-13 (RULER long-context eval), E-21 (bias/toxicity
    disparate-impact), E-23 (BFCL AST+exec), E-24 (RAGAS RAG eval).
  - Wave 6 batch 5 (5 F-series demand=3): F-14 (NCCL hang detector
    per-rank stack dump), F-15 (NCCL diagnosis JSON + exit-code class),
    F-16 (kernel timing nsys/Chrome-trace export),
    F-17 (attention viz row-sum=1 + causal mask),
    F-19 (explain token selection pre/post sampler chain).
  - Wave 6 batch 6 (5 G-series demand=3): G-04 (auto-gen README card
    with executable example), G-07 (semver publish tag round-trip),
    G-10 (org-scoped permission check pre-upload),
    G-14 (SPDX license validator + derivative-inheritance rank),
    G-15 (LFS dedupe — zero bytes on no-change re-publish).
  - Wave 6 batch 7 (7 H-series demand=3): H-12 (ImageFolder lexicographic
    classes + dense labels + corrupt-image-raises-IOError),
    H-14 (interleave_datasets probabilities + first/all_exhausted),
    H-15 (DatasetDict multi-split save/load round-trip),
    H-16 (LengthGroupedSampler padding-overhead < random baseline),
    H-17 (CLIP contrastive pair sampler — positives on diagonal),
    H-19 (synthetic data gen via teacher LLM + schema validation),
    H-21 (MinHash+LSH fuzzy dedup collision-prob parity).
  - Wave 6 batch 8 (5 I-series demand=3): I-07 (Claude Agent SDK tool
    envelope + JSONSchema Draft-2020-12 meta-schema validation),
    I-13 (MCP resources/list+read — stable URIs, mimeType matches bytes),
    I-14 (MCP prompts/list+get — missing required arg → JSON-RPC -32602),
    I-15 (agent memory plugin — put/get round-trip, monotonic TTL,
    self-recall@1 = 1.0), I-16 (guardrails pipeline — reject
    short-circuits, rewrites compose in order, empty = identity).
  - Wave 6 batch 9 (9 K-series demand=3 — closes the non-J demand=3
    cohort): K-05 (Gradio ChatInterface-shape /api/predict + single
    model load per process), K-11 (Ollama Modelfile DSL case-insensitive
    directives; unknown raises file:line:col), K-12 (VSCode extension
    package.json + commands ⊆ activationEvents + reproducible vsix),
    K-14 (systemd-analyze verify passes; non-root User=; Restart=
    on-failure bounded backoff), K-15 (Helm chart passes `helm lint`;
    `helm template` deterministic; liveness+readiness on /healthz),
    K-18 (ONNX Runtime cosine ≥0.999 vs native; onnx.checker passes),
    K-19 (CoreML .mlpackage layout + MLModel.load + ≥0.999 cosine on
    CPU_ONLY compute unit), K-20 (TensorRT-LLM engine count == tp_size;
    config.json dtype; ≥0.995 cosine vs native fp16),
    K-21 (MLX backend feature-gated to aarch64-apple-darwin; ≥0.999
    cosine; determinism at temp=0).
  - Wave 6 batch 10 (10 non-J demand=2 — closes entire non-J cohort):
    A-17 (cosign/sigstore detached sig + cert + rekor bundle; tamper
    detection fails closed), A-22 (disk-quota enforcement — pre-download
    reject, no partial blobs, JSON error with used/free/needed),
    B-17 (PyTorch QAT — observer min ≤ max; ≤0.5pp gap after convert;
    qparams round-trip), E-25 (OpenCLIP VLM bench — ImageNet zero-shot
    top5 ≥ top1; MSCOCO R@1 ≤ R@5 ≤ R@10; seeded determinism),
    F-18 (UMAP 2D — |rows| == vocab_size; token_str matches decode;
    seeded determinism), H-13 (torchaudio loader — waveform ∈ [-1,1]
    finite; resample respects target SR; unsupported ext raises),
    H-18 (word2vec/BPR negative sampling — positive ∩ negative = ∅;
    exact count; popularity freq^0.75 LLN at n=100000),
    H-20 (Presidio PII redact — row count preserved; 0 hits on redacted
    output; determinism under fixed salt), K-16 (Triton model-repo
    layout — config.pbtxt parses; /v2/models/name/ready returns 200;
    output shape matches config), K-17 (NVIDIA Dynamo worker integration
    — NATS heartbeat within 10s; schema validation; graceful
    worker_down on SIGTERM).
  - J-series (20 stories) rewritten to OpenCLAW agent-orchestration
    semantics (§10 resolution 2026-04-18); all 250 stories carry
    spec-complete bodies — no open interpretation gaps remain.
  - Each contract carries competitor CLI citations (arXiv papers, official
    docs) and bash falsification bodies with jq/curl/python3 invocations
- Gate: `pmat comply check` passes with 250 CRUX contracts registered

**Phase 1 — Evidence capture** (IN PROGRESS — 250/250 = 100% spec-complete; J-series now OpenCLAW-agent-resolved with evidence at `evidence/crux/openclaw/`)
- Collect `evidence/crux/{competitor}/*` per §2.1 for all 7 competitors
- Falsification harness comparing `apr --help` verbs vs competitor verbs
- Enrich remaining 51 contracts (spec-complete body) — demand≥4 non-J
  complete + 46 demand=3 A/B/C/D/E/F/G done; 21 demand=3 non-J remain (H/I/K),
  then ≈27 demand≤2 + J-block
- Gate: `apr qa --crux` emits per-story PASS/FAIL/SKIP

**Phase 2 — Close 🔨 partials** (priority by demand score)
- 63 partials → upgrade to ✅ one PR each
- Start with D=5 partials (19 stories): each becomes a golden-test PR
- Gate: supported count ≥ 80 / 250

**Phase 3 — Implement ❌ missing** (grouped by shared infra)
- **Group M1 — REST parity** (C-04, C-11, C-13, I-04, K-02, K-11): single `aprender-serve` sweep
- **Group M2 — Distributed training** (D-11, D-12, D-33, D-34, D-35): `aprender-train` sharding PR
- **Group M3 — OpenCLAW agent parity** (J-01..J-20): lands across existing `apr code`, MCP client layer, `apr serve` claude-proxy, SSC classifier, and a new `apr memory` surface (see evidence/crux/openclaw/)
- **Group M4 — Observability** (F-06..F-18, K-07, K-08): new `aprender-observe` crate
- **Group M5 — Advanced quant** (B-07..B-11, B-15..B-18): `aprender-compute` quant PR
- **Group M6 — Alignment methods** (D-03, D-04, D-17..D-19): `aprender-train` RLHF PR
- **Group M7 — Data pipeline depth** (H-03, H-06, H-12..H-21): `aprender-data` PR
- **Group M8 — Deployment targets** (K-13..K-21): `aprender-deploy` PR
- Gate: supported count == 250 / 250

**Phase 4 — Rename / alias 🤔 unclear**
- RFC for competitor-verb aliases (`apr cp` ≡ `apr copy`, etc.)
- Contract: `contracts/crux-aliases-v1.yaml`

---

## 8. Invariants & Anti-Patterns

- **Never** extrapolate a story into a feature ahead of its contract.
- **Never** claim ✅ status without a runnable golden test.
- **Never** mirror a competitor verb that violates aprender invariants (e.g.
  bypassing realizar for inference). Fundamentally incompatible verbs go in §9.
- **Never** use `grep` for competitor verb discovery — use
  `pmat query --regex "..." --path evidence/crux/..." --include-source`.
- **Never** silently drop demand score — a P0 missing story requires a
  `pmat work` item within the same PR that adds the sub-contract.

---

## 9. Rejected Competitor Verbs

| Competitor verb | Reason rejected |
|-----------------|-----------------|
| `ollama create` with imperative inference bypass | Violates realizar-first architecture; use `apr import` + contract-backed modelfile equivalent |
| `torch.compile(model)` fused graph on training crate | Training lives in `aprender-core`; graph fusion is `aprender-compute`'s job |
| `vllm serve --swap-space` disk-backed KV | Silent correctness hazard; gated behind explicit `--experimental-swap` flag |
| `pip install -e .` developer install | `uv run --with` is the stack-approved path (MEMORY feedback_no_pip) |

---

## 10. OpenCLAW interpretation — RESOLVED 2026-04-18

User intake listed **openclaw**. Resolution confirmed via the user
linking directly to [https://openclaw.ai](https://openclaw.ai) on
2026-04-18 during Phase 1 closure.

**What OpenCLAW actually is** (openclaw.ai):

- Local-first personal AI assistant / agent orchestration layer
- Chat-app transports: WhatsApp, Telegram, Discord, Slack, Signal, iMessage
- System control: files, shell, scripts (SSC-gated)
- Browser automation via MCP tool servers
- Persistent memory that learns user preferences
- Community-extensible skill catalog
- Underlying reasoning: Claude / GPT / local models (orchestration layer,
  not a model itself)
- Canonical install: `curl -fsSL https://openclaw.ai/install.sh | bash`
  → `npm i -g openclaw` → `openclaw onboard`

**Earlier default interpretation ("OpenCLIP") is INVALID.** The phonetic
similarity is a trap; the projects are unrelated. If vision-language
parity (CLIP / SigLIP / LAION) is needed later, it will live in a
separate category / sibling subspec `crux-openclip-v1.md`, not in
Category J.

**Resolution actions taken** (this revision, subspec v2.1.0):

- All 20 Category J sub-contracts rewritten to OpenCLAW agent semantics
  (commits on branch `docs/crux-competitive-research-ux`).
- Each J-contract carries `openclaw_interpretation: openclaw-agent-resolved`
  and `competitor: openclaw`.
- Evidence captured under `evidence/crux/openclaw/` (README, readme-verbs,
  config-schema.json5, hello.sh, capability-matrix, gaps).
- Demand scores preserved per-ID so master-contract aggregates stay stable.

Contract IDs `CRUX-J-01` … `CRUX-J-20` remain stable across the rewrite.

---

## 11. Cross-reference

- [apr-book-spec.md](apr-book-spec.md) — contract-first iron rule
- [aprender-monorepo-consolidation.md](aprender-monorepo-consolidation.md) — Rule 7 binds every CRUX story test
- [contracts/apr-cli-commands-v1.yaml](../../contracts/apr-cli-commands-v1.yaml) — authoritative 57-verb list
- [contracts/crux-competitive-research-ux-v1.yaml](../../contracts/crux-competitive-research-ux-v1.yaml) — master registry; blocks merges until every sub-contract exists at ≥ `draft`

---

## 12. pmat work integration

Every ❌ missing story gets a pmat work ticket so gap-closure is tracked in
the same system as the rest of aprender's engineering work. `pmat work`
commands are verified against `pmat work --help` as of 2026-04-18.

### 12.1 Demand → priority mapping

| demand_score | `pmat work` priority |
|--------------|----------------------|
| 5 (canonical) | `critical` |
| 4 (high) | `high` |
| 3 (moderate) | `medium` |
| ≤ 2 (niche) | `low` |

### 12.2 Bulk creation (one ticket per ❌ missing story)

No `--from` flag exists today. Use a shell loop over the YAML registry:

```bash
# 140 tickets expected (= ❌ missing count from §6)
yq '.stories[] | select(.status == "missing")
    | [.id, .title, .demand_score, .category, .competitor] | @tsv' \
    contracts/crux-competitive-research-ux-v1.yaml |
while IFS=$'\t' read -r id title score cat competitor; do
  case "$score" in
    5) prio=critical ;;
    4) prio=high ;;
    3) prio=medium ;;
    *) prio=low ;;
  esac
  pmat work add "CRUX gap: $id — $title" \
    -d "Contract: contracts/crux-$id-v1.yaml | Competitor: $competitor" \
    -p "$prio" \
    -t "crux,gap,crux-$cat,competitor-$competitor,crux-$id"
done
```

### 12.3 Querying CRUX work

```bash
# All CRUX tickets
pmat work list -t crux

# Critical (demand = 5) gaps — the adoption-critical subset
pmat work list -t crux -p critical

# Ticket status rollup
pmat work status -t crux
```

### 12.4 Closure

A pmat work ticket is closed by:

1. Sub-contract file `contracts/crux-{ID}-v1.yaml` moves `draft` → `enforced`
   with a golden-test harness
2. `apr qa --crux --story {ID}` returns PASS
3. The story's row in §5 is flipped `❌` → `🔨` or `✅`
4. `pmat work complete <ticket-id>` — runs the checkpoint invariants

### 12.5 Falsification and checkpoint

- `pmat work falsify <ticket-id>` — runs the sub-contract's `falsification:`
  conditions without marking the ticket done (dry-run).
- `pmat work checkpoint <ticket-id>` — the DbC §4.2 invariant gate; must
  PASS before a ticket can be `complete`d.

### 12.6 Invariant: no CRUX PR ships without a work-item touch

CI gate: if a PR modifies `contracts/crux-*.yaml`, the commit message MUST
include a `CRUX-Work: <pmat-ticket-id>` trailer. Enforced by a pre-push
hook that calls `pmat work validate`.

### 12.7 "Update spec frequently" protocol

This subspec is living; the master contract YAML and this doc MUST stay
locked step-in-step. Update cadence:

1. **On every status flip** (❌→🔨, 🔨→✅, etc.): same PR edits §5 row AND
   the corresponding `stories[]` entry in
   `contracts/crux-competitive-research-ux-v1.yaml` AND runs
   `pmat work edit <ticket>` to move the ticket to the matching state.
2. **On every new competitor release** (e.g. new vLLM minor version):
   rerun evidence collection (§2.1), diff against the registry, append
   new stories as CRUX-{letter}-{nn+1..} entries, bump subspec to vN.1.
3. **On every OpenCLAW clarification** (see §10): rewrite Category J in
   place, bump to vN+1.
4. **On every coverage-count change**: the §6 table and master-contract
   `coverage_intake:` block MUST be recomputed from §5 by the falsification
   harness and committed in the same PR.
5. **Weekly cadence**: `pmat work supervise -t crux` (manual until a hook
   exists) to surface stalled tickets.

---

## 13. Chain-of-Thought Derivation: HF kernels-community and APR-QA Canonicalization

> *Appendix added in v2.2 (2026-04-21). Presented in academic style
> (abstract → motivation → methodology → predictions → risks → references)
> to make the inclusion decision falsifiable rather than editorial.*

### Abstract

We formalize the addition of **Category L** (HuggingFace
`kernels-community` integration, 15 stories) and **Category M**
(APR-QA Playbook canonicalization, 10 stories) to the CRUX taxonomy,
raising story count 250 → 275. We argue that both categories were
provably missing from v2.1, that their omission produced a
**shadow-demand signal** bypassing §2's Five-Whys discipline, and
that folding them into the registry is the *minimal* intervention
that preserves the Iron Rule (**no story without a provable
contract**; equivalently, no contract without a story). We present
the Five-Whys root-cause decomposition, demand-weighted ranking, and
falsification conditions for each new category, and enumerate the
counterfactual risk of continued omission.

### 13.1 HF kernels-community (Category L)

#### 13.1.1 Motivation — why the gap exists

Between 2026-02 and 2026-04 aprender accumulated ≥ 18 performance
contracts (`FUSION-001..004`, `F-HOST-OVERHEAD-*`,
`F-ATTN-FLASHDECODE-*`, `F-DECODE-HOTPATH-*`) all tracking
*user-invisible* implementation concerns in `trueno` / `aprender-compute`.
CRUX, by construction (§4 "canonical verb" invariant), admits only
user-visible verbs. The effect is a **false-positive saturation**:
v2.1 reports 100% parity with Ollama/vLLM/llama.cpp/HF Transformers
while silently losing ground to the kernel-delivery layer one level
below the Python API — namely HF's `kernels-community` organization,
which ships drop-in `.so` kernels (`flash-attn2`, `flash-attn3`,
`fp8-fbgemm`, `paged-attention`, `liger-kernels`, `megablocks`,
`punica-sgmv`, `mamba-ssm`, etc.) pinned to specific CUDA /
compute-capability targets [1,2].

#### 13.1.2 Root cause (Five-Whys)

```
$ from kernels import get_kernel
$ fa3 = get_kernel("kernels-community/flash-attn3")

  Q1  WHY does the user invoke get_kernel?
  A1  → they want best-in-class FA3 performance without maintaining
        their own CUDA fork.
  Q2  WHY not fork?
  A2  → kernel authors ship CUDA 12.8 / PTX 9.0 / sm_100 updates on
        a weekly cadence — forking is O(staff-years).
  Q3  WHY weekly updates?
  A3  → H100 / B200 architecture-specific micro-optimizations (TMA,
        cluster launch, async WGMMA) [9,10].
  Q4  WHY can't aprender supply these in-tree?
  A4  → trueno's custom PTX path is blocked by the JIT pre-warming
        bug (`trueno#200`; must-use fused NF4 until 0.4.36).
  Q5  WHY is that a CRUX matter?
  A5  → because the community-visible *verb* for kernel access is
        `get_kernel(...)`, and that verb has no `apr` surface.

  ROOT CAUSE: aprender's kernel-delivery velocity is bandwidth-bound
              by kernel-author release cadence; HF kernels-community
              is the de-facto community mechanism for that delivery;
              parity with `get_kernel` is the minimal user-visible
              gate that unblocks the FUSION-003/004 roadmap.
  MAPS TO:    Category L (15 stories) — contracts/crux-L-*-v1.yaml
```

#### 13.1.3 Demand ranking

Applied to each of the 15 L-stories using the §2 signal weights
(README × 0.3 + issue-volume × 0.4 + bug-freq × 0.3):

| Subgroup | Stories | D | Justification |
|----------|---------|---|---------------|
| Canonical loader | L-01, L-15 | 4 | Any `apr` path to HF kernels requires these as preconditions |
| Attention kernels | L-02, L-03, L-06 | 4 | FA2/FA3/PA are top-3 cited kernels in `kernels-community` README [2] |
| Fused-loss | L-10 | 4 | liger-kernels fused CE-loss is the #1 training-speedup request upstream |
| Norm + rotary | L-04, L-05 | 3 | Small but steady wins; low-risk integration |
| Quant kernels | L-07, L-08, L-09 | 3 | Overlap with CRUX-B-09/10/11; Category L is the *integration* verb, B-series are the *algorithm* verbs |
| Specialized | L-11, L-12, L-13, L-14 | 2–3 | MoE / LoRA-SGMV / SSM — niche but growing |

#### 13.1.4 Falsification predictions

- **FALSIFY-CRUX-L-02**: if aprender's internal FA2 path is ≥ 3 %
  slower than `kernels-community/flash-attn2` on a held-out
  grid `{hidden_dim ∈ [64, 128], seq_len ∈ [512, 4096],
  head_dim ∈ [64, 128]}` under identical sm_80 hardware and
  CUDA 12.6, L-02 escalates to P0 and the HF kernel becomes
  the aprender default. The 3 % threshold mirrors the
  Ollama-parity methodology in CLAUDE.md.
- **FALSIFY-CRUX-L-15**: if a published `kernels-community` SHA is
  loaded into aprender without byte-verified revision-pinning
  (mirroring `huggingface_hub.hf_hub_download(revision=...)`
  semantics [1]), the contract is violated — the kernel
  surface is a supply-chain boundary.

### 13.2 APR-QA Playbook Canonicalization (Category M)

#### 13.2.1 Motivation — why the gap exists

The sibling repository `apr-model-qa-playbook` ([8]) formalizes an
8-gate Popperian QA protocol rooted in Toyota Production System
principles ([7]: Jidoka, Poka-Yoke) and Popperian falsification
([6]). At v2.1, CRUX surfaces exactly *two* rows bound to that
protocol:

- `CRUX-E-08` ✅ golden-output regression gate
- `CRUX-F-21` ✅ `apr qa` 8-gate runner

The remaining ≥ 8 gates (tensor-stats, cross-format parity,
tokenizer round-trip, chat-template equivalence, canary pass@1,
5-Whys trace, property-based fuzz, upstream-fix enforcement) are
invisible in the competitor matrix. Because CRUX is the registry
a new contributor reads to understand what `apr` *must* do, the
playbook's 8 additional gates accrue outside CRUX's gravity well —
which is exactly the shadow-demand failure mode §13.0's Abstract
names.

#### 13.2.2 Root cause (Five-Whys)

```
$ apr qa model.apr --golden-output

  Q1  WHY does the user invoke this?
  A1  → they need a PASS/FAIL gate before shipping a model.
  Q2  WHY a gate?
  A2  → contracts/apr-model-qa-v1.yaml promises 8 Popperian checks.
  Q3  WHY 8 checks?
  A3  → apr-model-qa-playbook formalizes the Toyota-Jidoka protocol
        as 8 gates [8].
  Q4  WHY are only 2 visible in CRUX?
  A4  → the playbook is a sibling repo; the `has_sibling_repos`
        detector (PMAT-160) marked it as external → excluded from
        CRUX scan.
  Q5  WHY is that a problem?
  A5  → users onboarding via the Sovereign-AI-Stack book expect QA
        to be a first-class CRUX category, equivalent in stature to
        C (serving) and D (training).

  ROOT CAUSE: the playbook is the canonical source of truth for
              model qualification, but CRUX indexes only 2 of its
              gates. New engineers therefore meet the gates via
              the playbook's README rather than via CRUX, bypassing
              the Iron Rule and the §12 pmat-work tracking.
  MAPS TO:    Category M (10 stories) — contracts/crux-M-*-v1.yaml
```

#### 13.2.3 Demand ranking

| Subgroup | Stories | D | Justification |
|----------|---------|---|---------------|
| Ship-blocker gates | M-01, M-02, M-03, M-04, M-07 | 5 | Any one of these failing is a release veto; they are the playbook's P0 cohort |
| Integrity gates | M-05, M-06 | 4 | Tokenizer + chat-template are the two highest-frequency silent-corruption vectors historically (GH-202 class) |
| Meta-gates | M-08, M-10 | 4 | 5-Whys root-cause + upstream-fix rule are the Toyota Jidoka backbone |
| Coverage extension | M-09 | 4 | Property-based fuzz is the only row *missing* from the playbook itself and the only ❌ in Category M |

#### 13.2.4 Falsification predictions

- **FALSIFY-CRUX-M-01**: if `apr qa --gate=byte-identical` returns
  PASS on a model that subsequently fails the playbook's
  safetensors round-trip on the *same* bytes, M-01 is falsified
  and the playbook implementation becomes authoritative. This
  makes the in-tree `apr qa` gate the *derivable* artefact, not
  the source of truth.
- **FALSIFY-CRUX-M-09**: if the property-based fuzz harness, run
  for 1000 cases against a historical shipped regression (e.g.
  GH-202 tokenizer OOB; CB-510 `/models/` gitignore hazard),
  fails to reproduce the regression, M-09 is falsified — the
  harness is insufficient and the contract resets to ❌ until
  fixed.
- **FALSIFY-CRUX-M-10**: if any aprender PR lands a workaround
  (a shim, a compatibility hack, a silent-fallback) for a bug in
  `trueno` / `realizar` / `aprender-train` instead of fixing the
  root cause, M-10 is falsified. The contract binds the
  playbook's no-workarounds rule ([8, CLAUDE.md]) to CRUX CI.

### 13.3 Why *now* — update triggers per §12.7

- **§12.7 trigger #2** (new competitor release): HF shipped
  `kernels-community` with prebuilt `flash-attn3` kernels for
  `sm_90a` / `sm_100` in 2026-Q1 [2]; this is the first upstream
  release that meaningfully closes aprender's kernel gap without
  the trueno#200 JIT blocker, raising L's inclusion utility.
- **§12.7 trigger #4** (coverage-count change): the CRUX-SHIP-001
  retrofit loop closed 18 classifier-only PRs (`#962..#988`) on
  2026-04-21, exhausting the retrofit queue. Author-bandwidth is
  freed for the 25 new stories, and every L/M contract can adopt
  the g1..g4 merge discipline **from story authoring**, rather
  than as retrofit — avoiding the PARTIAL_ALGORITHM_LEVEL
  compromise entirely.

### 13.4 Counterfactual risk (if Categories L + M are omitted)

1. **False-positive parity**. CRUX continues to report 100 %
   registration while the community's de-facto kernel-delivery
   mechanism ([1,2]) and the project's own QA playbook ([8])
   remain unbound. A new contributor reading only §5 cannot
   deduce that either exists. This violates the Iron Rule
   transitively.
2. **Shadow-demand accrual**. Both surfaces continue to absorb
   engineering work outside CRUX's §12 pmat-work tracking —
   invisible to the weekly `pmat work supervise -t crux`
   cadence. When the surfaces eventually ship they arrive
   unreviewed by the competitive-parity lens.
3. **Jidoka regression**. M-10 is the clause that forbids
   workarounds. Its absence from CRUX is the load-bearing
   failure mode: the Aprender organization has repeatedly paid
   for this (PMAT-216 test-mocking bypass; GH-202 tokenizer
   OOB escape) and the only durable fix is a registry-bound
   contract that CI can enforce.

### References

[1] HuggingFace. *kernels: a package and decorator for using
    prebuilt optimized kernels from the Hub*. `huggingface.co/docs/kernels`.
    Accessed 2026-04-21.

[2] HuggingFace. *kernels-community organization*.
    `huggingface.co/kernels-community`. Accessed 2026-04-21.

[3] Dao, T., Haziza, D., Massa, F., Sizov, G. *FlashAttention-3:
    Fast and Accurate Attention with Asynchrony and Low-Precision*.
    arXiv:2407.08608. 2024.

[4] Frantar, E., Ashkboos, S., Hoefler, T., Alistarh, D.
    *GPTQ: Accurate Post-Training Quantization for Generative
    Pre-trained Transformers*. arXiv:2210.17323. 2022.

[5] Lin, J. et al. *AWQ: Activation-aware Weight Quantization for
    LLM Compression and Acceleration*. arXiv:2306.00978. 2023.

[6] Popper, K. *The Logic of Scientific Discovery*. Hutchinson &
    Co. 1959.

[7] Ohno, T. *Toyota Production System: Beyond Large-Scale
    Production*. Productivity Press. 1988.

[8] Aprender Engineering (PAIML). *apr-model-qa-playbook:
    property-based model qualification with Popperian
    falsification*. `github.com/paiml/apr-model-qa-playbook`.
    Accessed 2026-04-21.

[9] NVIDIA. *H100 Tensor Core GPU Architecture Whitepaper*. 2022.

[10] NVIDIA. *B200 Blackwell Architecture Whitepaper*. 2024.

[11] Gu, A., Dao, T. *Mamba: Linear-Time Sequence Modeling with
     Selective State Spaces*. arXiv:2312.00752. 2023.

[12] Chen, L. et al. *Punica: Multi-Tenant LoRA Serving*.
     arXiv:2310.18547. 2023.

[13] Dao, T. *FlashAttention-2: Faster Attention with Better
     Parallelism and Work Partitioning*. arXiv:2307.08691. 2023.

[14] Dettmers, T. et al. *QLoRA: Efficient Finetuning of Quantized
     LLMs*. arXiv:2305.14314. 2023.

---

## 14. Category R — SHACL ontologies vs Apache Jena (registered v2.4)

> **Status: registered 2026-10-09 (v2.4).** CRUX-SHACL, epic #3598. Sub-ticket
> S1 (§14.6) registered the four stories CRUX-R-01..04 in §5 and in the master
> contract in one commit, per §12.7, and admitted `jena` to `CRUX_COMPETITORS`.
> - Story contracts: `contracts/crux-R-01-v1.yaml` .. `contracts/crux-R-04-v1.yaml`, one per cell of §14.2
> - Harness contract: `contracts/crux-shacl-jena-v1.yaml`; harness `tests/oracle/jena/` (S2)
> - Measured state: `docs/lookahead/0.71-crux-shacl-baseline.md`, and §14.7
> - Citations in §14.1–§14.6 are at `origin/main` `316dee2cd4` unless a line says otherwise.

**The defect (raised 2026-10-04).** No test runs pv against a second SHACL
engine to check that both:
- read the same ontology (input);
- write it so the other engine reads it back to the same graph (return);
- validate it to the same verdict (validate);
- report the same results with the full W3C detail (feedback).

Apache Jena is the Java toolkit named when the defect was raised. Its command
line covers all four verbs, so Jena is the competitor.

### 14.1 The competitor and its verbs

| Verb | Jena (command line) | pv at `316dee2cd4` |
|---|---|---|
| input: read Turtle and N-Triples | `riot` parses and re-serializes. A syntax error names its line | **None.** The gate path has no Turtle reader (R-13, `crates/aprender-contracts/src/ontology/w3c.rs:4-5`). Shapes come only from YAML `shape:` blocks (`ontology/shapes.rs:3`). pv's graph has no blank nodes (R-15, `ontology/rdf.rs:3`) |
| return: write a graph another engine reads back | `riot --output=<syntax>` | pv writes N-Triples (`rdf.rs:189` `to_ntriples`) and Turtle shapes (`shapes.rs:1232` `to_turtle`). Nothing checks that another engine reads them back to the same graph |
| validate: SHACL Core | `shacl validate` with a shapes graph and a data graph | A Core subset (`shapes.rs:5-17`). At parse it refuses, with exit 3: `and`/`or`/`not`/`xone`, `sparql`, the `qualifiedValueShape` family, `equals`/`disjoint`, `languageIn`/`uniqueLang`, `targetNode`/`targetSubjectsOf`/`targetObjectsOf`, recursive shapes, every path but a single predicate, and every component the table does not name |
| feedback: the validation report | A W3C `sh:ValidationReport` | PV-ONT-011 lint text or JSON (`lint/shapes_gate.rs:1004-1016`). A result holds severity, focus, contract stem, optional path, short component name and message (`shapes.rs:143-152`). It has no `sh:value`, no `sh:sourceShape` IRI and no `sh:sourceConstraintComponent` IRI |
| compare two graphs | `rdfcompare` (graph isomorphism) | None |

The Jena command names are from its distribution's `bin/`. S2 confirms each
against the pinned release. A command missing there is NOT MEASURED, not assumed.

### 14.2 Four cells

| Cell | Claim | Comparator | At `316dee2cd4` |
|---|---|---|---|
| R-INPUT | pv reads X and dumps it as N-Triples. The dump is isomorphic to `riot`'s. On a malformed X, both reject it at the same line | `rdfcompare`; the line numbers by the harness | pv reads no RDF file: RED once the harness runs |
| R-RETURN | Jena(X) ≅ Jena(pv_write(pv_read(X))). `riot` reads pv's own `contracts.nt` and `shapes.ttl` with no error | `rdfcompare` | pv writes, nothing reads back: NOT MEASURED |
| R-VALIDATE | The same `sh:conforms` and the same result count, per case | the harness; on S-W3C, W3C's expected report arbitrates (§14.4) | 19 W3C cases are hand-vendored as YAML and agree with W3C. There is no Jena run |
| R-FEEDBACK | The same results, as a multiset of the full tuple (focus, path, value, source-shape key, component IRI, severity). Every result has a message, and a declared `sh:message` appears in it | the harness | pv emits no W3C report: RED once the harness runs |

**Rules for the comparison:**
- **Never compare a file with itself.** R-RETURN's two sides are Jena's parse of
  X and Jena's parse of a file pv wrote in the same run.
- **Both engines get byte-identical inputs.** The receipt records each input's
  sha256. Only a planted control changes one side.
- **Source-shape key.** A property shape is usually a blank node, and its label
  differs between engines. Its key is the parent node shape's IRI plus the
  property shape's `sh:path`. A shape with an IRI is keyed by that IRI.

### 14.3 Subjects

- **S-W3C.** Every case in the `core/` manifests of the W3C `data-shapes` test
  suite, at a pinned commit, with the archive's sha256 pinned.
  - The case count is read from the manifests at run time and printed.
  - A case pv refuses is RED for that case and named, never skipped.
  - `w3c.rs` vendors 19 cases and lists 16 more as NOT_VENDORED. That list
    bounds the gate's scope, not this subject's.
- **S-CORPUS.** `contracts/contracts.nt` (32,665 triples, no blank node) and
  `contracts/shapes.ttl` (394 blank-node property shapes): the files pv writes
  and the release ships.
- **S-SHSH.** Jena validates pv's `shapes.ttl` against the SHACL-SHACL shapes
  graph (namespace `http://www.w3.org/ns/shacl-shacl#`, published with the SHACL
  Recommendation), pinned by sha256. This checks that pv writes well-formed
  SHACL. pv does not run it: that graph uses `sh:or`, which is outside the subset.

### 14.4 Verdicts

Exit codes follow the ELK oracle (`tests/oracle/owl/src/main.rs`): 0 GREEN,
1 RED, 2 NOT MEASURED.

- **GREEN:** the JVM ran, every planted control (§14.5) was seen, and every cell
  agrees.
- **RED:** a disagreement, a refused case, or a planted control not seen. Each
  RED line names one cell, one case and one comparator.
- **NOT MEASURED:** any of these, and it is never GREEN:
  - no JVM;
  - Jena cannot be fetched;
  - a pin's sha256 differs;
  - the subject is empty or partial, meaning fewer cases ran than the manifests
    list;
  - a cell whose pv side needs S3 or S4 (§14.6) before it can be measured. So
    until both land, a run with every Jena-side check green exits 2, never 0.
- **Without a JVM, the harness can prove RED but never GREEN.** pv against
  W3C's expected reports needs no JVM. Without a JVM the exit is 1 if such a RED
  exists, else 2.
- **W3C arbitrates on S-W3C:**
  - pv = W3C ≠ Jena: GREEN for pv, with a `jena≠w3c <case>` line in the receipt.
  - pv = Jena ≠ W3C: RED, comparator `w3c`.
  - pv ≠ W3C = Jena: RED, comparator `w3c`.
  - S-CORPUS has no expected report, so there the comparator is Jena.
- **Mode: never a PR check.** R-13 (ONT-001, paiml/infra
  `docs/specifications/paiml-ontology.md`) allows no JVM reachable from `pv`.
  It admits a third-party RDF engine only as a pinned out-of-gate oracle in
  `tests/oracle/`, outside the workspace. The harness is therefore its own
  detached crate there, like the OWL oracle.
- **It cannot block a merge or a release.** It runs on demand and at night,
  never in a PR, merge-queue or release job. A RED opens or updates one ticket
  with one owner. NOT MEASURED stays NOT MEASURED; it is never read as a pass.
  Making it blocking is a separate decision this section does not propose.

### 14.5 Planted controls

Controls act on the comparator's inputs, never on pv, so they prove the harness
before S3 and S4 land. Until then the side under test is Jena's own output.
Every run plants every control, and a control not seen is RED for its cell.

| Cell | Positive control (must agree) | Negative control (must be RED, naming the item in brackets) |
|---|---|---|
| R-INPUT | X against X with every blank-node label renamed | The side under test's dump with one triple dropped [the triple]. `"01"^^xsd:integer` rewritten to `"1"^^xsd:integer` on one side [the literal: RDF terms compare by lexical form, not by value] |
| R-RETURN | X with string escapes (`\"`, `\\`, `\n`), a numeric Unicode escape, a raw `é`, and language tags (`@fr`, `@en-GB`) | The written file with one triple dropped before Jena reads it [the triple] |
| R-VALIDATE | W3C `core/property/minCount-001` | The same case with `sh:minCount 1` changed to `2` in the side under test's copy only [minCount] |
| R-FEEDBACK | The minCount-001 report against a copy whose blank-node labels differ | Six copies of the report under test, each with one field changed: focus, path, value, source shape, component, severity [that field]. One copy with `sh:value` removed [value] |

The arbitration rules of §14.4 are tested the same way, with one planted triple
of reports (under test, Jena, W3C) per rule.

### 14.6 Work items (one ticket, five sub-tickets under #3598; none minted)

| Sub-ticket | Delivers | Cells it can turn GREEN |
|---|---|---|
| S1 | `jena` in `CRUX_COMPETITORS`, with its admission comment. Category R stories, one per cell, and one contract per cell. This section and the master contract move together. **Done 2026-10-09** (v2.4): CRUX-R-01..04 | none (registration) |
| S2 | The Jena distribution pinned by sha256, as the OWL oracle pins ELK 0.4.3. The detached harness under `tests/oracle/`, its exit codes and its planted controls. The JVM and Jena declared in the hosts' configuration (paiml/infra, cross-repo) | none alone; its controls prove the comparators. **Harness done 2026-10-09** (`tests/oracle/jena/`, `make oracle-jena`, self-test 26/26); the hosts' configuration in paiml/infra is not done |
| S3 | pv reads Turtle and N-Triples, blank nodes included. The reader is in house, because R-13 bars a third-party RDF crate from `aprender-contracts`' tree, dev-dependencies included. Open design question: R-15 bars blank nodes from the graph pv extracts and hashes, so the reader must hold input blank nodes outside that graph, or S3 asks for a ruling | R-INPUT, R-RETURN |
| S4 | pv emits a W3C `sh:ValidationReport`, as Turtle and as JSON, with every field of §14.2 | R-FEEDBACK; R-VALIDATE on S-W3C, where refused cases are named |
| S5 | The rudof oracle reads S4's report and compares the full tuple. `tests/oracle/differential.json` is regenerated with every vendored case. The oracle runs on a schedule | none; it hardens the second oracle |

**Precondition, proposed separately as the fix for `make oracle-check` (§14.7).**
pv stops admitting rdf:type implicitly on `sh:closed` shapes, the 6 closed shapes
in `contracts/` declare it in the same PR, and W3C `node/closed-001` is vendored.
The other fix, declaring rdf:type on the shapes alone, would turn oracle-check
GREEN but leave pv short of SHACL Core on any closed shape that does not declare
it, which is what R-VALIDATE tests against Jena.

### 14.7 Initial state (measured 2026-10-04 on `316dee2cd4`; detail in the baseline file)

- **No Jena run exists.** There is no harness and no pin, and all four cells are
  NOT MEASURED.
- **pv against W3C:** the 19 vendored cases agree on (conforms, focus, path,
  component). 16 cases are not vendored.
- **The rudof oracle against pv:**
  - The W3C arm agrees on 19 of 19.
  - On the corpus, the oracle reports 736 results and pv 730. The 6 extra are
    rdf:type values on focus nodes of 5 of the 6 `sh:closed` shapes (the sixth,
    `parity-receipt-complete`, had no hit). pv admits rdf:type on every closed
    shape (`shapes.rs:875`); SHACL Core does not.
  - So `make oracle-check` is RED, and its record `tests/oracle/differential.json`
    dates from 2026-09-19.
- **The oracle compares (focus, component) only.** Changing the severity, the
  message text or the shape id in pv's report leaves its verdict unchanged.

### 14.8 First Jena run (measured 2026-10-09 on branch commit `62b1cebe99`; receipt `tests/oracle/jena/receipt.json`)

Jena 5.6.0 (archive sha256 pinned), OpenJDK 17.0.20.1. The self-test saw 26 of
26 controls; FALSIFY-CRUXSHACL-027 is S5's. The run exits 1. The receipt is
from a branch commit, not from `main`.

| Cell | Case | Verdict |
|---|---|---|
| R-INPUT | `contracts.nt` + `shapes.ttl` | NOT MEASURED (S3). `riot` reads 32,701 + 4,166 triples |
| R-RETURN | `contracts.nt` | GREEN: `riot`'s parse is isomorphic to the harness's own N-Triples parse, 32,701 triples |
| R-RETURN | `shapes.ttl` | GREEN: 4,166 triples, no WARN |
| R-RETURN | round trip | NOT MEASURED (S3) |
| R-VALIDATE | S-CORPUS | **RED**: Jena 743 results, pv 730. All 13 extra are `sh:closed` results on rdf:type values, on 12 focus nodes: the defect of the §14.6 precondition, tracked as #4837 |
| R-VALIDATE | S-W3C | NOT MEASURED (S4). Jena ran all 35 cases with no `jena≠w3c` line |
| R-FEEDBACK | S-W3C + S-CORPUS | NOT MEASURED (S4) |
| S-SHSH | `shapes.ttl` | GREEN: conforms to SHACL-SHACL |

What the run taught the harness, each now handled in code:
- Jena 5.6.0 writes its log lines to stdout, mixed with graph output, so the
  harness separates them before reading either.
- `riot` exits 1 on a WARN alone, so the harness reads its log, not only its exit code.
- Jena writes a blank-node source shape as `sh:sourceShape []` with no triples,
  so the harness keys it from the shapes graph (§14.2).
- S-W3C is the 35 W3C core cases already vendored under `tests/oracle/w3c/`
  (pinned by `tests/oracle/jena/w3c.sha256`), not yet every case in the `core/`
  manifests as §14.3 asks.
- The graph comparison is the harness's own (`tests/oracle/jena/src/nt.rs`:
  colour refinement, then individualisation), not Jena's `rdfcompare`.
