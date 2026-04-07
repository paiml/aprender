# Multi-Provider LLM API Specification

> Parent: [batuta-spec.md](../batuta-spec.md)
> Inspiration: CCX-RS provider-agnostic architecture (anton-abyzov/ccx-rs)
> See also: [apr-code.md](apr-code.md) (`apr code` consumes this API for model routing)

---

## 1. Overview

Batuta's agent runtime and serving layer currently support local inference (realizar) and a planned `RemoteDriver` for external LLMs. This specification extends the remote driver into a **provider-agnostic LLM API client** that speaks both Anthropic and OpenAI wire protocols, with automatic format translation, streaming SSE, exponential backoff, and provider failover.

### Motivation

The Sovereign AI Stack prioritizes local inference (Sovereign tier), but real-world deployments need hybrid routing:
- **Development**: Use remote LLMs for rapid iteration before local model is ready
- **Fallback**: When local GGUF/APR model OOMs or exceeds latency SLA, spill to remote
- **Capability gap**: Some tasks need frontier models unavailable locally
- **Cost optimization**: Remote inference can be cheaper than GPU idle time for bursty workloads

### Non-Goals

- Not a general-purpose API gateway (batuta serve handles endpoint routing)
- Not a model hosting service (realizar handles local inference)
- Not a chat UI framework (agent runtime handles conversation management)

---

## 2. Architecture

```
batuta/
  +-- serve/
  |     +-- backends.rs          # PrivacyTier (REUSE)
  |     +-- circuit_breaker.rs   # CostCircuitBreaker (REUSE)
  |     +-- router.rs            # SpilloverRouter (REUSE)
  +-- agent/
  |     +-- driver/
  |     |     +-- mod.rs          # LlmDriver trait (REUSE)
  |     |     +-- realizar.rs     # RealizarDriver (REUSE)
  |     |     +-- remote/         # (NEW)
  |     |     |     +-- mod.rs           # RemoteDriver dispatch
  |     |     |     +-- anthropic.rs     # Anthropic Messages API
  |     |     |     +-- openai.rs        # OpenAI Chat Completions API
  |     |     |     +-- translator.rs    # Bidirectional format translation
  |     |     |     +-- sse.rs           # Streaming SSE parser
  |     |     |     +-- backoff.rs       # Retry with exponential backoff + jitter
  |     |     +-- routing.rs      # RoutingDriver (REUSE + EXTEND)
```

### Reuse Matrix

| Existing Module | Reuse in Multi-Provider |
|----------------|------------------------|
| `serve::backends::PrivacyTier` | Block remote calls under Sovereign tier |
| `serve::circuit_breaker::CostCircuitBreaker` | Per-provider cost budget enforcement |
| `serve::router::SpilloverRouter` | Local-first routing with remote fallback |
| `agent::driver::LlmDriver` | Trait implemented by each provider backend |

---

## 3. Provider Protocol Translation

### 3.1 Wire Format Mapping

The translator module converts between Anthropic Messages API and OpenAI Chat Completions API formats bidirectionally. This enables the agent loop to work identically regardless of upstream provider.

```rust
pub struct MessageTranslator;

impl MessageTranslator {
    /// Anthropic CompletionRequest -> OpenAI ChatCompletionRequest
    pub fn to_openai(request: &CompletionRequest) -> OpenAiRequest;

    /// OpenAI ChatCompletionResponse -> Anthropic CompletionResponse
    pub fn from_openai(response: &OpenAiResponse) -> CompletionResponse;

    /// Tool definitions: Anthropic tool_use <-> OpenAI function_call
    pub fn translate_tools(tools: &[ToolDefinition], target: Protocol) -> Value;
}
```

### 3.2 Message Role Mapping

| Anthropic | OpenAI | Notes |
|-----------|--------|-------|
| `user` | `user` | Direct |
| `assistant` | `assistant` | Direct |
| `system` (top-level) | `system` / `developer` | OpenAI uses `developer` for o-series |
| `tool_use` (content block) | `tool_calls` (message field) | Structural difference |
| `tool_result` (role) | `tool` (role) | `tool_call_id` required in OpenAI |

### 3.3 Tool Use Translation

```
Anthropic format:                    OpenAI format:
{                                    {
  "role": "assistant",                 "role": "assistant",
  "content": [{                        "tool_calls": [{
    "type": "tool_use",                  "id": "call_abc",
    "id": "toolu_abc",                   "type": "function",
    "name": "file_read",                 "function": {
    "input": {"path": "src/"}             "name": "file_read",
  }]                                       "arguments": "{\"path\":\"src/\"}"
}                                        }
                                       }]
                                     }
```

Key translation challenges:
- OpenAI `arguments` is a JSON string, Anthropic `input` is a JSON object
- OpenAI requires `tool_call_id` on tool results; generate deterministic IDs
- Anthropic `stop_reason: "tool_use"` maps to OpenAI `finish_reason: "tool_calls"`

### 3.4 Supported Providers

| Provider | Protocol | Endpoint | Privacy Tier |
|----------|----------|----------|-------------|
| **realizar** (local) | Native | N/A | Sovereign |
| **Anthropic** | Anthropic Messages | `api.anthropic.com/v1/messages` | Standard |
| **OpenAI** | OpenAI Chat | `api.openai.com/v1/chat/completions` | Standard |
| **OpenRouter** | OpenAI-compatible | `openrouter.ai/api/v1/chat/completions` | Standard |
| **Ollama** (local) | OpenAI-compatible | `localhost:11434/v1/chat/completions` | Private |
| **vLLM** (self-hosted) | OpenAI-compatible | Configurable | Private |

---

## 4. Streaming SSE Parser

### 4.1 Design

Real-time token streaming for responsive agent output. The SSE parser handles both Anthropic and OpenAI streaming formats, normalizing events into a unified `StreamEvent` enum.

```rust
/// Unified stream events (provider-agnostic)
pub enum StreamEvent {
    /// New text delta
    TextDelta { text: String },
    /// Tool use started
    ToolUseStart { id: String, name: String },
    /// Tool use input delta (partial JSON)
    ToolUseDelta { id: String, partial_json: String },
    /// Content block finished
    BlockStop,
    /// Message complete
    MessageStop { stop_reason: StopReason, usage: TokenUsage },
    /// Stream error
    Error { message: String },
}
```

### 4.2 Protocol-Specific Parsing

| Aspect | Anthropic SSE | OpenAI SSE |
|--------|--------------|------------|
| Event prefix | `event: message_start`, `content_block_delta`, ... | `data: {"choices":[...]}` |
| Delta location | `delta.text` in content block | `choices[0].delta.content` |
| Tool streaming | `input_json_delta` (partial JSON) | `choices[0].delta.tool_calls[0].function.arguments` |
| Done signal | `event: message_stop` | `data: [DONE]` |
| Usage reporting | In `message_delta` event | In final chunk or separate `usage` field |

### 4.3 Implementation

```rust
pub struct SseParser {
    protocol: Protocol,
    buffer: String,
}

impl SseParser {
    /// Parse a raw SSE line into a StreamEvent
    pub fn parse_line(&mut self, line: &str) -> Option<StreamEvent>;

    /// Parse from an async byte stream (reqwest::Response)
    pub fn stream(
        response: reqwest::Response,
        tx: Sender<StreamEvent>,
    ) -> JoinHandle<Result<TokenUsage>>;
}
```

---

## 5. Retry and Failover

### 5.1 Exponential Backoff with Jitter

Based on AWS architecture best practices (full jitter variant):

```rust
pub struct BackoffConfig {
    /// Base delay (default: 200ms)
    pub base: Duration,
    /// Maximum delay cap (default: 30s)
    pub cap: Duration,
    /// Maximum retry attempts (default: 10)
    pub max_retries: u32,
}

impl BackoffConfig {
    /// Full jitter: sleep = random_between(0, min(cap, base * 2^attempt))
    pub fn delay_for(&self, attempt: u32) -> Duration;
}
```

### 5.2 Retry Policy

| HTTP Status | Action | Rationale |
|------------|--------|-----------|
| 429 | Retry with backoff | Rate limited |
| 500, 502, 503 | Retry with backoff | Transient server error |
| 408 | Retry with backoff | Request timeout |
| 401, 403 | Fail immediately | Auth error (no retry helps) |
| 400 | Fail immediately | Bad request (client error) |
| 200 + malformed SSE | Retry once | Intermittent stream corruption |
| 200 + connection drop (no `[DONE]`) | Retry turn from scratch | Mid-stream TCP RST. **FALSIFICATION FIX (F-005):** Discard partial tool use JSON. Emit `Error { message: "stream interrupted" }`. Do NOT attempt to parse incomplete tool calls. |

### 5.3 Provider Failover

When a provider exceeds its retry budget, the `RoutingDriver` cascades to the next provider in priority order:

```
1. realizar (local)          -- Sovereign, zero latency
2. Ollama/vLLM (self-hosted) -- Private, low latency
3. Anthropic                 -- Standard, highest quality
4. OpenRouter                -- Standard, cost fallback
```

Failover triggers:
- **Consecutive failures >= 2**: Switch to next provider
- **Latency SLA breach**: P99 > configured threshold (default 30s)
- **Cost circuit breaker**: Per-provider USD budget exhausted
- **Context window exceeded**: Model's context too small for current conversation — **FALSIFICATION FIX (C-007):** This trigger is BLOCKED under Sovereign tier. Instead, force compaction. Under Private/Standard, failover to larger-context provider is allowed.

```rust
pub struct FailoverConfig {
    /// Max consecutive failures before switching provider
    pub failure_threshold: u32,
    /// P99 latency threshold before switching
    pub latency_sla: Duration,
    /// Per-provider USD budget (resets daily)
    pub daily_budget: HashMap<ProviderId, f64>,
}
```

---

## 6. Cost Tracking

### 6.1 Per-Turn Estimation

```rust
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cache_read_tokens: Option<u32>,
    pub cache_creation_tokens: Option<u32>,
}

pub struct CostEstimate {
    pub provider: ProviderId,
    pub model: String,
    pub usage: TokenUsage,
    pub estimated_usd: f64,
}
```

### 6.2 Budget Enforcement

Integrates with existing `CostCircuitBreaker` from `src/serve/circuit_breaker.rs`:

| Budget Scope | Enforcement |
|-------------|-------------|
| Per-turn | Warn if estimated cost > threshold before sending |
| Per-session | Hard stop when cumulative cost exceeds session budget |
| Per-provider daily | Switch to next provider when daily limit hit |
| Global monthly | Hard stop, require explicit user override |

### 6.3 Pricing Table

Maintained as a TOML configuration file, updatable without recompilation:

```toml
# ~/.config/batuta/pricing.toml
[anthropic."claude-sonnet-4-20250514"]
input_per_mtok = 3.00
output_per_mtok = 15.00
cache_read_per_mtok = 0.30

[openai."gpt-4o"]
input_per_mtok = 2.50
output_per_mtok = 10.00
```

---

## 7. Configuration

### 7.1 Agent Manifest Extension

```toml
# agent.toml
[driver]
type = "routing"  # Uses RoutingDriver with failover

[driver.providers.realizar]
priority = 1
privacy_tier = "sovereign"
model = "llama-3.2-3b.apr"

[driver.providers.ollama]
priority = 2
privacy_tier = "private"
base_url = "http://localhost:11434"
model = "qwen3:1.7b-q4k"

[driver.providers.anthropic]
priority = 3
privacy_tier = "standard"
model = "claude-sonnet-4-20250514"
daily_budget_usd = 5.00

[driver.failover]
failure_threshold = 2
latency_sla_ms = 30000

[driver.backoff]
base_ms = 200
cap_ms = 30000
max_retries = 10
```

### 7.2 Environment Variables

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Anthropic authentication |
| `OPENAI_API_KEY` | OpenAI authentication |
| `OPENROUTER_API_KEY` | OpenRouter authentication |
| `BATUTA_LLM_PROVIDER` | Override default provider |
| `BATUTA_LLM_BUDGET` | Override session budget (USD) |

---

## 8. Privacy Tier Integration

| Tier | Allowed Providers | Network |
|------|-------------------|---------|
| **Sovereign** | realizar only | No egress |
| **Private** | realizar + Ollama/vLLM (self-hosted) | Local network only |
| **Standard** | All providers | Any |

The `RoutingDriver` enforces this at dispatch time. **FALSIFICATION FIX (C-006):** Retry with backoff on the same provider BEFORE cascading to the next:

```rust
impl RoutingDriver {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        for provider in &self.providers_by_priority {
            if provider.privacy_tier() > self.max_privacy_tier {
                continue;  // Skip providers that violate tier policy
            }
            // Retry with backoff on same provider before cascading
            for attempt in 0..self.backoff.max_retries {
                match provider.complete(request.clone()).await {
                    Ok(response) => return Ok(response),
                    Err(e) if e.is_retryable() => {
                        let delay = self.backoff.delay_for(attempt);
                        tokio::time::sleep(delay).await;
                        continue;  // Retry same provider
                    }
                    Err(e) => break,  // Non-retryable: cascade to next provider
                }
            }
            // Retries exhausted on this provider: cascade to next
        }
        Err(AgentError::AllProvidersFailed)
    }
}
```

---

## 9. Design Principles

| Toyota Principle | Application |
|-----------------|-------------|
| **Jidoka** | Provider failover stops bad path, switches to healthy provider |
| **Poka-Yoke** | Privacy tier enforcement prevents accidental data egress |
| **Heijunka** | Load leveling across providers based on cost and latency |
| **Muda** | Cost circuit breakers prevent waste on expensive providers |
| **Kaizen** | Pricing table and failover thresholds tunable without recompilation |

---

## 10. Implementation Phases

| Phase | Scope | Dependencies |
|-------|-------|-------------|
| **1** | Anthropic driver + SSE streaming | `agent::driver::LlmDriver` trait |
| **2** | OpenAI-compatible driver + format translation | Phase 1 |
| **3** | RoutingDriver failover + backoff | Phases 1-2 + `serve::router` |
| **4** | Cost tracking + budget enforcement | Phase 3 + `serve::circuit_breaker` |
| **5** | Ollama/vLLM local-network drivers | Phase 2 (OpenAI-compatible) |

---

## 11. Testing Strategy

| Test Type | Coverage |
|-----------|----------|
| Unit: format translation | Round-trip Anthropic <-> OpenAI for all message types |
| Unit: SSE parser | Both protocols, partial lines, malformed input |
| Unit: backoff | Jitter distribution, cap enforcement, attempt counting |
| Integration: mock server | Full request/response cycle per provider |
| Property: translation fidelity | Arbitrary message trees survive round-trip translation |
| Falsification: cost estimation | Known pricing vs estimated cost within 1% |

---

## 12. CLI Commands

```bash
# List configured providers
batuta agent providers

# Test provider connectivity
batuta agent providers --test

# Show cost summary for current session
batuta agent cost

# Override provider for a single run
batuta agent run --provider anthropic --manifest agent.toml
```

---

## 13. Prior Art and References

### 13.1 GitHub Prior Art

| Project | Stars | Relevance |
|---------|-------|-----------|
| **BerriAI/litellm** | 41.9K | Python SDK + proxy for 100+ LLM APIs in OpenAI format; cost tracking, guardrails, load balancing |
| **Portkey-AI/gateway** | 11.2K | Blazing fast AI gateway with integrated guardrails; routes to 200+ LLMs with <100us overhead |
| **maximhq/bifrost** | 3.4K | Enterprise AI gateway (50x faster than LiteLLM) with adaptive load balancer and cluster mode |
| **0xPlaygrounds/rig** | 6.7K | Build modular, scalable LLM applications in Rust; provider-agnostic agent framework |
| **openai/openai-agents-python** | 20.5K | Lightweight multi-agent workflow framework; tool-use loop reference architecture |
| **microsoft/agent-framework** | 8.4K | Framework for building, orchestrating, and deploying AI agents with Python and .NET |
| **eugene1g/agent-safehouse** | 1.5K | Sandbox local AI agents so they can read/write only what they need; Landlock/Seatbelt |
| **liquidOS-ai/AutoAgents** | 529 | Multi-agent framework in Rust; demonstrates Rust-native agent patterns |
| **vstorm-co/summarization-pydantic-ai** | 17 | Context management for Pydantic AI agents; LLM-powered summarization and sliding window trimming |
| **eventsource-stream** (various) | 467 | Streaming, source-agnostic SSE parser crate |

**Key insight from prior art:** LiteLLM and Portkey prove the multi-provider pattern at scale (400B+ tokens/day through Portkey). Rig proves Rust is viable for this. Our differentiator is privacy tier enforcement (Sovereign/Private/Standard) which none of the above implement.

### 13.2 arXiv References

| Paper | ID | Year | Key Finding |
|-------|-----|------|-------------|
| **Orca: Distributed Serving System for Transformer-Based Models** | Yu et al. | 2022 | Continuous batching and iteration-level scheduling for LLM serving; foundation for our streaming SSE design |
| **ReliabilityBench: Evaluating LLM Agent Reliability** | arXiv:2601.06112 | 2026 | Chaos engineering framework for agents: fault injection via transient timeouts, rate limits, partial responses, schema drift — directly informs our failover testing |
| **Rethinking Multi-agent Reliability: Byzantine Fault Tolerance** | arXiv:2511.10400 | 2025 | CP-WBFT: confidence-probe-based weighted BFT for LLM agents; assigns higher weight to more credible agents — applicable to our provider confidence scoring |
| **Robust LLM Training Infrastructure at ByteDance** | arXiv:2509.16293 | 2025 | 38K explicit + 5.9K implicit failures in 3 months; every-step checkpointing with <0.9% overhead — informs our session persistence design |
| **FAILS: Framework for LLM Service Incident Analysis** | arXiv:2503.12185 | 2025 | Incident analysis across OpenAI, Anthropic, Character.AI, Stability.AI — empirical data on provider failure modes that our failover must handle |
| **Adaptive Fault Tolerance in Cloud LLM Environments** | arXiv:2503.12228 | 2025 | Adaptive fault tolerance mechanisms for LLMs in cloud; GPU-aware routing and connection liveness for long-running requests |
| **Automated Hypothesis Validation with Agentic Sequential Falsification** | arXiv:2502.09858 | 2025 | Popper-inspired LLM agents that design falsification experiments; strict Type-I error control — foundational methodology for our falsification section |
| **LLM Tasks in Software Verification and Falsification** | arXiv:2404.09384 | 2024 | Taxonomy of 100+ papers on LLM-native verification/falsification; generative transformation patterns |
| **RouteLLM: Learning to Route LLMs with Preference Data** | arXiv:2406.18665 | 2024 | Cost-optimal routing between strong/weak models using preference-trained classifiers; 2x cost reduction with minimal quality loss (ICLR 2025) |
| **Unified Routing and Cascading for LLMs** | arXiv:2410.10347 | 2024 | Provably optimal strategies for both routing and cascading; unifies into "cascade routing" framework — directly validates our local→private→standard cascade |
| **Dynamic Model Routing and Cascading: A Survey** | arXiv:2603.04445 | 2026 | Comprehensive survey covering query-adaptive selection, cost-quality Pareto frontiers, production patterns |
| **FrugalGPT: Better LLM Use at Reduced Cost** | arXiv:2305.05176 | 2023 | LLM cascade strategies: prompt adaptation, model approximation, cascade routing; up to 98% cost reduction |
| **Speculative Decoding Side Channels** | arXiv:2411.01076 | 2024 | Streaming token-by-token leaks speculation patterns via packet-size side channels — security consideration for our SSE streaming |

**Key academic insights:**
- RouteLLM and FrugalGPT validate our cascade architecture (local → private → standard)
- ReliabilityBench's chaos engineering approach maps directly to our falsification tests
- FAILS provides empirical failure mode data for realistic testing

### 13.3 Batuta Oracle Context

The Oracle recommends `realizar` (85% confidence) for inference tasks and `entrenar` for training, confirming our driver hierarchy: `RealizarDriver` (Sovereign) as primary, with remote fallback. The `ml-serving` recipe shows realizar's native serving API (`Server::new(model).port(8080)`), and the `rag-pipeline` recipe demonstrates trueno-rag's hybrid BM25+vector retrieval — both reused in the agent's tool chain.

---

## 14. Provable Contracts

### 14.1 Contract: Provider Routing (`provider-routing-v1.yaml`)

```yaml
metadata:
  version: 1.0.0
  created: '2026-04-02'
  author: PAIML Engineering
  description: Multi-provider routing correctness — privacy enforcement, failover, cost
  references:
  - 'RouteLLM (Chen et al., 2024): arXiv:2406.18665'
  - 'FrugalGPT (Chen et al., 2023): arXiv:2305.05176'
  - 'ReliabilityBench: arXiv:2601.06112'
  depends_on:
  - backend-dispatch-v1
  - streaming-tpot-v1

equations:
  privacy_enforcement:
    formula: |
      route(request, tier) ∈ allowed_providers(tier)
      where allowed_providers(Sovereign) = {realizar}
            allowed_providers(Private) = {realizar, ollama, vllm}
            allowed_providers(Standard) = {realizar, ollama, vllm, anthropic, openai, openrouter}
    domain: tier ∈ {Sovereign, Private, Standard}
    invariants:
    - Sovereign tier NEVER routes to external network
    - Privacy tier ordering is total — Sovereign < Private < Standard
    - Downgrading tier always reduces provider set
    preconditions:
    - request.messages.len() > 0
    - request.messages.iter().all(|m| !m.content.is_empty())
    postconditions:
    - result.provider.privacy_tier() <= tier
    lean_theorem: Theorems.Privacy_Enforcement

  failover_cascade:
    formula: |
      route(request) = first(p in providers_by_priority
                             where p.tier <= max_tier
                             AND p.failures < threshold
                             AND p.budget_remaining > 0)
    domain: providers ordered by priority, threshold ∈ ℤ⁺
    invariants:
    - Higher-priority provider always tried first
    - Failed providers skipped (not retried in same turn)
    - All providers exhausted → AllProvidersFailed error (never silent fallback)
    preconditions:
    - providers.len() > 0
    - providers.iter().any(|p| p.tier <= max_tier)
    lean_theorem: Theorems.Failover_Cascade

  cost_budget:
    formula: |
      cost(turn) = (input_tokens × input_rate + output_tokens × output_rate) / 1_000_000
      cumulative(session) = sum(cost(turn) for turn in session)
    domain: input_tokens, output_tokens ∈ ℤ⁺, rates ∈ ℝ⁺
    invariants:
    - cumulative(session) <= session_budget at all times
    - Per-provider daily cost <= provider.daily_budget
    - Cost is non-negative and monotonically increasing
    postconditions:
    - result.estimated_usd >= 0.0
    - result.estimated_usd.is_finite()
    lean_theorem: Theorems.Cost_Budget

  backoff_jitter:
    formula: |
      delay(attempt) = random(0, min(cap, base × 2^attempt))
    domain: attempt ∈ [0, max_retries), base > 0, cap > 0
    invariants:
    - delay(attempt) <= cap for all attempts
    - delay(attempt) >= 0 for all attempts
    - Expected delay grows exponentially but is bounded
    preconditions:
    - attempt < max_retries
    - base > Duration::ZERO
    - cap >= base
    lean_theorem: Theorems.Backoff_Jitter

  format_translation:
    formula: |
      from_openai(to_openai(anthropic_msg)) == anthropic_msg  (per field comparison)
      to_openai(from_openai(openai_msg)) == openai_msg        (per field comparison)
      Equality defined per field:
        - role: exact string match
        - content text: byte-identical
        - tool arguments: JSON semantic equality (parse both, deep compare)
        - tool call IDs: preserved or deterministically regenerated via blake3(provider_id || msg_idx || tool_idx)
        - stop_reason: mapped via lookup table (exact)
    domain: Valid Anthropic/OpenAI message types
    invariants:
    - Role preserved through round-trip
    - Tool call IDs preserved or deterministically regenerated
    - Content text byte-identical after round-trip
    - Tool arguments JSON-semantically equal after round-trip
    lean_theorem: Theorems.Format_Translation

proof_obligations:
- type: invariant
  property: Sovereign tier blocks remote egress
  formal: tier == Sovereign => provider ∈ {realizar}
  applies_to: all
- type: monotonicity
  property: Priority ordering respected in failover
  formal: priority(selected) <= priority(any_available) for first selected
  applies_to: all
- type: bound
  property: Cost never exceeds budget
  formal: cumulative_cost <= session_budget
  applies_to: all
- type: bound
  property: Backoff delay bounded by cap
  formal: delay <= cap for all attempts
  applies_to: all
- type: roundtrip
  property: Format translation preserves semantics
  formal: content(round_trip(msg)) == content(msg)
  applies_to: all
- type: termination
  property: Failover terminates
  formal: route() terminates in O(|providers|) steps
  applies_to: all
- type: frame
  property: Request immutability
  formal: routing does not mutate the original CompletionRequest
  applies_to: all
- type: postcondition
  property: SSE stream completeness
  formal: streaming produces MessageStop event with usage data
  applies_to: all

falsification_tests:
- id: FALSIFY-MPA-001
  rule: Privacy enforcement
  prediction: Sovereign tier never calls external HTTP endpoint
  test: Set tier=Sovereign, configure Anthropic provider, assert zero HTTP calls
  if_fails: Privacy tier leak — Sovereign data sent to external provider
- id: FALSIFY-MPA-002
  rule: Failover cascade
  prediction: After 2 failures on provider A, next request goes to provider B
  test: Mock provider A to return 500, verify provider B receives next request
  if_fails: Failover not triggering — stuck on broken provider
- id: FALSIFY-MPA-003
  rule: Cost budget enforcement
  prediction: Request blocked when cumulative cost exceeds session budget
  test: Set budget=$0.01, send requests until budget exhausted, assert next request returns BudgetExhausted
  if_fails: Cost tracking underestimates or budget not enforced
- id: FALSIFY-MPA-004
  rule: Backoff cap
  prediction: No retry delay exceeds 30s
  test: proptest 10000 attempts with random attempt counts, assert all delays <= cap
  if_fails: Exponential growth not capped — unbounded sleep durations
- id: FALSIFY-MPA-005
  rule: Format translation round-trip
  prediction: Anthropic message survives round-trip through OpenAI format
  test: proptest with arbitrary tool_use messages, assert content equality after round-trip
  if_fails: Lossy translation — tool arguments or content corrupted
- id: FALSIFY-MPA-006
  rule: SSE stream completeness
  prediction: Every stream ends with MessageStop containing usage data
  test: Stream from mock server, inject partial chunks, verify MessageStop always received
  if_fails: Stream parser drops final event — cost tracking breaks
- id: FALSIFY-MPA-007
  rule: Provider failover under chaos
  prediction: Agent completes task despite 50% random provider failures
  test: Chaos injection (ReliabilityBench-style) — random 429/500/timeout on all providers, verify task completion
  if_fails: Failover cascade too brittle for real-world provider instability

kani_harnesses:
- id: KANI-MPA-001
  obligation: Sovereign tier blocks remote egress
  property: Privacy enforcement invariant
  bound: 4
  strategy: exhaustive
  solver: cadical
  harness: verify_sovereign_no_egress
- id: KANI-MPA-002
  obligation: Cost never exceeds budget
  property: Cumulative cost bounded
  bound: 8
  strategy: stub_float
  solver: cadical
  harness: verify_cost_budget_bound
- id: KANI-MPA-003
  obligation: Backoff delay bounded by cap
  property: Delay cap enforcement
  bound: 16
  strategy: bounded_int
  solver: cadical
  harness: verify_backoff_cap
- id: KANI-MPA-004
  obligation: Format translation preserves semantics
  property: Round-trip content equality
  bound: 4
  strategy: bounded_int
  solver: cadical
  harness: verify_translation_roundtrip
- id: KANI-MPA-005
  obligation: Failover terminates
  property: Route function terminates in bounded steps
  bound: 8
  strategy: exhaustive
  solver: cadical
  harness: verify_failover_terminates

qa_gate:
  id: F-MPA-001
  name: Multi-Provider API Contract
  description: Provider routing correctness, privacy enforcement, cost bounds
  checks:
  - sovereign_no_egress
  - failover_cascade_priority
  - cost_budget_bound
  - backoff_cap
  - format_translation_roundtrip
  - sse_stream_completeness
  - chaos_failover_resilience
  pass_criteria: All 7 falsification tests pass + 5 Kani harnesses verify
  falsification: Inject chaos (timeouts, 500s, malformed SSE) at provider boundary
```

### 14.2 Contract Location

Save to `../provable-contracts/contracts/batuta/provider-routing-v1.yaml` and register in `../provable-contracts/contracts/batuta/binding.yaml`.

---

## 15. Popperian Falsification Report

### 15.1 Falsifiable Claims

Every claim in this specification is stated as a testable prediction. If any falsification test fails, the corresponding design assumption is wrong and must be revised.

| Claim | Falsification Test | What Failure Means |
|-------|-------------------|-------------------|
| **Privacy tiers prevent data egress** | FALSIFY-MPA-001 | Architecture-level security flaw; Sovereign meaningless |
| **Failover improves availability** | FALSIFY-MPA-002, 007 | Failover adds complexity without benefit; simpler single-provider better |
| **Cost tracking is accurate** | FALSIFY-MPA-003 | Budget enforcement unreliable; users get surprise bills |
| **Backoff prevents thundering herd** | FALSIFY-MPA-004 | Retry storms degrade provider further; need circuit breaker instead |
| **Format translation is lossless** | FALSIFY-MPA-005 | Multi-provider support introduces data corruption; unsafe for production |
| **SSE streaming is reliable** | FALSIFY-MPA-006 | Non-streaming fallback required; streaming is optimization not baseline |
| **Cascade routing is resilient** | FALSIFY-MPA-007 | Real-world failure modes (correlated outages, cascading failures) defeat cascade |

### 15.2 Pre-Registered Null Hypotheses

Following arXiv:2502.09858 (Automated Hypothesis Validation with Agentic Sequential Falsifications):

| H₀ (Null) | Test to Reject H₀ | Required Evidence |
|-----------|-------------------|-------------------|
| Privacy tier enforcement has at least one bypass path | Fuzz all request paths under Sovereign tier; any external HTTP call rejects H₀ | Zero external calls in 10K fuzzed requests |
| Provider failover does NOT improve p99 latency vs single provider | Benchmark single-provider vs cascade with 10% failure injection | p99 latency improves by ≥20% with cascade |
| Cost estimation error exceeds 5% on known pricing | Compare estimated vs actual cost for 1000 OpenAI/Anthropic calls | Mean absolute error < 2%, max error < 5% |
| Format translation loses information in ≥1% of messages | Property test 100K random message trees through round-trip | Zero content mutations detected |

### 15.3 What Would Disprove This Specification

If ANY of these are true, this specification's architecture is fundamentally wrong:

1. **Correlated provider failures are the norm, not the exception.** If Anthropic and OpenAI outages correlate >50% of the time, cascade routing provides no resilience benefit. (Check: FAILS dataset, arXiv:2503.12185)
2. **Format translation overhead exceeds 10% of request latency.** If serialize→translate→deserialize costs more than 10ms per turn, the translation layer is too expensive. (Check: benchmark Anthropic→OpenAI translation at p99)
3. **Privacy tier enforcement cannot be verified at compile time.** If Landlock/Seatbelt policies require runtime-only verification, the "compile-time safety" claim in section 10 is false. (Check: provable-contracts Kani harness KANI-MPA-001)
4. **Local inference (realizar) cannot meet minimum quality for agentic tasks.** If local 3B models fail >80% of tool-use tasks that Claude succeeds at, the Sovereign tier is unusable for agents. (Check: benchmark tool-use accuracy on SWE-bench subset)
