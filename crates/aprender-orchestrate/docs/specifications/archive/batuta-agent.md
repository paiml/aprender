# Batuta Agent Specification v1.0

**Status:** Draft
**Authors:** PAIML Engineering
**Date:** 2026-02-28
**Refs:** BATUTA-AGENT-001

## Abstract

Batuta Agent extends the orchestration framework with an autonomous agent runtime. Agents execute perceive-reason-act loops using local LLM inference (realizar), retrieval-augmented generation (trueno-rag), and persistent memory (trueno-db) — all sovereign by default, with zero external API dependencies.

This specification defines the agent loop, LLM driver abstraction, tool system, memory substrate, capability model, and integration with existing batuta subsystems.

```
[REVIEW-001] @noah 2026-02-28
Toyota Principle: Jidoka (Autonomation with a Human Touch)
Agents act autonomously within bounded loops, but stop on capability
violations, budget exhaustion, or loop detection — just as a Jidoka
machine stops on defect rather than producing waste.
Status: DRAFT
```

---

## 1. Introduction

### 1.1 Motivation

The Sovereign AI Stack has all infrastructure for autonomous agent execution:

| Capability | Stack Component | External Alternative (Rejected) |
|---|---|---|
| LLM Inference | realizar (GGUF/APR/SafeTensors) | OpenAI API, Anthropic API |
| RAG Retrieval | trueno-rag (BM25+vector, RRF) | Pinecone, Qdrant Cloud |
| State Storage | trueno-db (KvStore, morsels) | Redis, PostgreSQL |
| Embeddings | aprender | OpenAI Embeddings API |
| Distributed Compute | repartir (CPU/GPU/Remote) | AWS Lambda, Ray |
| Model Registry | pacha (Ed25519 signing) | HuggingFace Hub |

What's missing is the **agent orchestration layer** — the perceive-reason-act loop that wires these components into an autonomous reasoning system.

### 1.2 Design Principles

| Principle | Application |
|---|---|
| **Jidoka** | Loop guard stops on ping-pong detection, budget exhaustion, or max iterations |
| **Poka-Yoke** | Capability system prevents unauthorized tool access; PrivacyTier blocks remote egress |
| **Muda** | CostCircuitBreaker prevents runaway spend on hybrid (local+remote) deployments |
| **Heijunka** | SpilloverRouter balances load between local and remote backends |
| **Genchi Genbutsu** | Default sovereign path: go directly to local hardware, no proxy layers [Ref 25, 26] |

### 1.3 Non-Goals

- **Not a chatbot framework** — No channel adapters (Slack, Discord, etc.)
- **Not a workflow engine** — batuta `playbook` already handles multi-step workflows
- **Not a model trainer** — entrenar handles training; agents consume trained models
- **Not an API gateway** — batuta `serve` already handles model serving endpoints

---

## 2. Architecture

### 2.1 Module Position

```
batuta/
  ├── serve/      (routing, privacy, failover, context)      ← REUSE
  ├── oracle/     (RAG pipeline, knowledge graph)             ← REUSE
  ├── agent/      (perceive-reason-act runtime)               ← NEW
  │     ├── driver/   (LlmDriver trait + implementations)
  │     ├── tool/     (Tool trait + builtin tools)
  │     └── memory/   (MemorySubstrate trait + implementations)
  └── cli/agent.rs (CLI subcommand)                           ← NEW
```

### 2.2 Reuse Matrix

The agent module reuses existing batuta subsystems directly, avoiding type duplication:

| Existing Module | Reuse |
|---|---|
| `serve::backends::PrivacyTier` | Driver privacy enforcement — Sovereign blocks all remote |
| `serve::context::ContextManager` | Token counting + truncation before LLM calls |
| `serve::circuit_breaker::CostCircuitBreaker` | Cost budget enforcement per agent invocation |
| `serve::router::SpilloverRouter` | Phase 2: hybrid local→remote routing |
| `serve::failover::FailoverManager` | Phase 2: streaming recovery on backend failure |
| `serve::templates::ChatTemplateEngine` | Prompt formatting for local models (ChatML/Llama/Mistral) |
| `oracle::rag::RagOracle` | RagTool wraps this directly for document retrieval |
| `oracle::knowledge_graph` | Agent can query stack component knowledge |
| `jugar_probar::Browser` / `jugar_probar::Page` | BrowserTool wraps for headless Chromium navigation, screenshots, WASM eval |
| `wos` | Target environment for testing agents and models in WASM browser context |
| `provable_contracts::schema::Contract` | YAML design-by-contract for agent loop invariants |
| `provable_contracts_macros::contract` | `#[contract]` proc macro for compile-time binding audit |
| `presentar::App` | WASM Canvas2D/WebGPU rendering for agent dashboards in wos |
| `presentar_terminal` | Terminal fallback for agent dashboards (already in batuta TUI) |
| `pmcp::Client` | MCP client: agent discovers and calls external MCP tool servers at runtime |
| `pforge_runtime::Handler` | MCP server: agent tools exposed to external LLM clients (Claude Code, other agents) |

```
[REVIEW-002] @noah 2026-02-28
Toyota Principle: Muda Elimination (Waste)
Reusing serve/ and oracle/ eliminates ~2000 lines of duplicated code
for privacy tiers, context management, RAG, and circuit breakers.
Status: DRAFT
```

### 2.3 Data Flow

```
                    ┌──────────────────────────────────────────────┐
                    │              batuta agent run                 │
                    └──────────────────┬───────────────────────────┘
                                       │
                    ┌──────────────────▼───────────────────────────┐
                    │            AgentManifest (TOML)               │
                    │  name, model_path, capabilities, privacy_tier │
                    └──────────────────┬───────────────────────────┘
                                       │
            ┌──────────────────────────▼──────────────────────────┐
            │                   PERCEIVE                           │
            │  MemorySubstrate.recall(query) → inject into prompt  │
            └──────────────────────────┬──────────────────────────┘
                                       │
            ┌──────────────────────────▼──────────────────────────┐
            │                    REASON                            │
            │  ContextManager.truncate(messages)                   │
            │  LlmDriver.complete(request) → CompletionResponse    │
            └──────────────┬───────────────────┬──────────────────┘
                           │                   │
                    ┌──────▼──────┐     ┌──────▼──────┐
                    │  end_turn   │     │  tool_use   │
                    └──────┬──────┘     └──────┬──────┘
                           │                   │
                           │     ┌─────────────▼─────────────┐
                           │     │           ACT              │
                           │     │  capability_check(tool)    │
                           │     │  LoopGuard.check(call)     │
                           │     │  tool.execute(input)       │
                           │     │  feed result → REASON      │
                           │     └────────────────────────────┘
                           │
            ┌──────────────▼──────────────────────────────────┐
            │                   REMEMBER                       │
            │  MemorySubstrate.remember(interaction)           │
            │  → AgentLoopResult { response, usage, cost }     │
            └─────────────────────────────────────────────────┘
```

---

## 3. Core Types

### 3.1 LlmDriver Trait

```rust
/// Abstraction over LLM inference backends.
/// Default implementation: RealizarDriver (sovereign, local).
#[async_trait]
pub trait LlmDriver: Send + Sync {
    /// Non-streaming completion.
    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, AgentError>;

    /// Streaming completion with channel-based events.
    /// Default wraps complete() for drivers that don't support streaming.
    async fn stream(
        &self,
        request: CompletionRequest,
        tx: tokio::sync::mpsc::Sender<StreamEvent>,
    ) -> Result<CompletionResponse, AgentError> {
        let response = self.complete(request).await?;
        let _ = tx.send(StreamEvent::TextDelta {
            text: response.text.clone(),    // [F-009] .text is a field, not a method
        }).await;
        let _ = tx.send(StreamEvent::ContentComplete {
            stop_reason: response.stop_reason.clone(),
            usage: response.usage.clone(),
        }).await;
        Ok(response)
    }

    /// Maximum context window in tokens.
    fn context_window(&self) -> usize;

    /// Privacy tier this driver operates at.
    /// Reuses serve::backends::PrivacyTier.
    fn privacy_tier(&self) -> PrivacyTier;
}
```

**Implementations:**

| Driver | Privacy Tier | Backend | Phase |
|---|---|---|---|
| `RealizarDriver` | Sovereign | Local GGUF/APR via realizar | 1 |
| `MockDriver` | Sovereign | Deterministic responses (testing) | 1 |
| `RemoteDriver` | Standard | HTTP to OpenAI/Anthropic/Groq | 2 |
| `RoutingDriver` | Configurable | SpilloverRouter: local-first, remote fallback | 2 |

### 3.2 Message, CompletionRequest / CompletionResponse

```rust
/// Agent message type extending ChatMessage with tool-use variants.  [F-008]
/// ChatMessage (from serve::templates) only has System/User/Assistant text roles.
/// Agent loop needs tool-use and tool-result message types for multi-turn tool calls.
#[derive(Debug, Clone)]
pub enum Message {
    /// System prompt (injected once at start).
    System(String),
    /// User query or follow-up.
    User(String),
    /// Assistant text response.
    Assistant(String),
    /// Assistant tool use request (maps to ToolCall).
    AssistantToolUse(ToolCall),
    /// Tool execution result.
    ToolResult(ToolResult),
}

impl Message {
    /// Convert to serve::templates::ChatMessage for context truncation.
    /// Tool-use and tool-result messages are serialized as assistant/user JSON.
    pub fn to_chat_message(&self) -> ChatMessage {
        match self {
            Self::System(s) => ChatMessage::system(s),
            Self::User(s) => ChatMessage::user(s),
            Self::Assistant(s) => ChatMessage::assistant(s),
            Self::AssistantToolUse(call) => ChatMessage::assistant(
                format!("[tool_use: {} {}]", call.name, call.input)
            ),
            Self::ToolResult(result) => ChatMessage::user(
                format!("[tool_result: {}]", result.content)
            ),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub system: Option<String>,
}

/// Simplified response — text and tool_calls are separate fields.  [F-009]
/// No ContentBlock enum needed: text is always String, tool calls are Vec<ToolCall>.
#[derive(Debug, Clone)]
pub struct CompletionResponse {
    pub text: String,
    pub stop_reason: StopReason,
    pub tool_calls: Vec<ToolCall>,
    pub usage: TokenUsage,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    StopSequence,
}

#[derive(Debug, Clone, Default)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}
```

### 3.3 StreamEvent

```rust
#[derive(Debug, Clone)]
pub enum StreamEvent {
    PhaseChange { phase: LoopPhase },
    TextDelta { text: String },
    ToolUseStart { id: String, name: String },
    ToolUseEnd { id: String, name: String, result: String },
    ContentComplete { stop_reason: StopReason, usage: TokenUsage },
}

#[derive(Debug, Clone, PartialEq)]
pub enum LoopPhase {
    Perceive,
    Reason,
    Act { tool_name: String },
    Done,
    Error { message: String },
}
```

### 3.4 Tool Trait

```rust
/// Executable tool with capability enforcement.
#[async_trait]
pub trait Tool: Send + Sync {
    /// JSON Schema definition for the LLM.
    fn definition(&self) -> ToolDefinition;

    /// Execute the tool. Returns content string or error.
    async fn execute(
        &self,
        input: serde_json::Value,
    ) -> Result<ToolResult, AgentError>;

    /// Required capability to invoke this tool (Poka-Yoke).
    fn required_capability(&self) -> Capability;

    /// Execution timeout (Jidoka: stop on timeout).
    fn timeout(&self) -> Duration {
        Duration::from_secs(120)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct ToolResult {
    pub tool_use_id: String,
    pub content: String,
    pub is_error: bool,
}
```

**Builtin Tools (Phase 1):**

| Tool | Wraps | Capability | Description |
|---|---|---|---|
| `RagTool` | `oracle::rag::RagOracle` | `Capability::Rag` | Search indexed documents via BM25+vector [Ref 18] |
| `MemoryTool` | `MemorySubstrate` | `Capability::Memory` | Read/write agent persistent state |
| `BrowserTool` | `jugar_probar::Browser` | `Capability::Browser` | Launch headless Chromium, navigate, screenshot, evaluate JS/WASM |

### 3.5 MemorySubstrate Trait

```rust
/// Unique identifier for a stored memory fragment.  [F-010]
pub type MemoryId = String;

/// Filter for memory recall queries.  [F-010]
#[derive(Debug, Clone, Default)]
pub struct MemoryFilter {
    /// Filter by agent ID.
    pub agent_id: Option<String>,
    /// Filter by memory source type.
    pub source: Option<MemorySource>,
    /// Filter memories created after this time.
    pub since: Option<chrono::DateTime<chrono::Utc>>,
}

/// Unified structured + semantic memory store.
#[async_trait]
pub trait MemorySubstrate: Send + Sync {
    /// Store a memory fragment with optional embedding for recall.
    async fn remember(
        &self,
        agent_id: &str,
        content: &str,
        source: MemorySource,
        embedding: Option<&[f32]>,
    ) -> Result<MemoryId, AgentError>;

    /// Recall relevant memories via semantic similarity.
    async fn recall(
        &self,
        query: &str,
        limit: usize,
        filter: Option<MemoryFilter>,
        query_embedding: Option<&[f32]>,
    ) -> Result<Vec<MemoryFragment>, AgentError>;

    /// Store structured key-value data.
    async fn set(
        &self,
        agent_id: &str,
        key: &str,
        value: serde_json::Value,
    ) -> Result<(), AgentError>;

    /// Retrieve structured key-value data.
    async fn get(
        &self,
        agent_id: &str,
        key: &str,
    ) -> Result<Option<serde_json::Value>, AgentError>;

    /// Delete a memory fragment.
    async fn forget(&self, id: MemoryId) -> Result<(), AgentError>;
}

#[derive(Debug, Clone)]
pub struct MemoryFragment {
    pub id: MemoryId,
    pub content: String,
    pub source: MemorySource,
    pub relevance_score: f32,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
pub enum MemorySource {
    Conversation,
    ToolResult,
    System,
    User,
}
```

**Implementations:**

| Implementation | Backend | Recall Strategy | Persistence | Phase |
|---|---|---|---|---|
| `InMemorySubstrate` | HashMap + Vec | Substring matching (no embeddings) [F-011] | None (ephemeral) | 1 |
| `TruenoMemory` | trueno-db KV + trueno-rag vectors | Semantic similarity (vector search) | Durable | 2 |

**Phase 1 limitation [F-011]:** `InMemorySubstrate::recall()` uses case-insensitive substring matching on content, not semantic similarity. The `embedding` parameters are accepted but ignored. True semantic recall requires `TruenoMemory` (Phase 2) which uses trueno-rag vector search. This phased approach is validated by Liu et al. [Ref 19] and Xia et al. [Ref 20] who identify substring vs semantic recall as a known tradeoff in agent memory taxonomies.

### 3.6 AgentManifest

```rust
/// Agent configuration loaded from TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AgentManifest {
    pub name: String,
    pub version: String,
    pub description: String,

    /// LLM model configuration.
    pub model: ModelConfig,

    /// Resource quotas (Muda elimination).
    pub resources: ResourceQuota,

    /// Granted capabilities (Poka-Yoke).
    pub capabilities: Vec<Capability>,

    /// Privacy tier. Reuses serve::backends::PrivacyTier.
    /// Default: Sovereign (local-only).
    pub privacy: PrivacyTier,

    /// External MCP servers to connect to (agents-mcp feature).  [F-022]
    #[cfg(feature = "agents-mcp")]
    #[serde(default)]
    pub mcp_servers: Vec<McpServerConfig>,
}

/// Configuration for an external MCP server connection.  [F-022]
#[cfg(feature = "agents-mcp")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    pub name: String,
    pub transport: McpTransport,
    /// For stdio: command + args to launch the server process.
    #[serde(default)]
    pub command: Vec<String>,
    /// For SSE/WebSocket: URL to connect to.
    pub url: Option<String>,
    /// Capabilities granted to tools from this server.
    pub capabilities: Vec<String>,
}

#[cfg(feature = "agents-mcp")]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum McpTransport {
    Stdio,
    Sse,
    WebSocket,

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Path to local model (GGUF/APR/SafeTensors).
    pub model_path: Option<PathBuf>,
    /// Remote model identifier (Phase 2, for spillover).
    pub remote_model: Option<String>,
    /// Maximum tokens per completion.
    pub max_tokens: u32,
    /// Sampling temperature.
    pub temperature: f32,
    /// System prompt.
    pub system_prompt: String,
    /// Context window override (auto-detected if None).
    pub context_window: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceQuota {
    /// Maximum loop iterations per invocation.
    pub max_iterations: u32,
    /// Maximum tool calls per invocation.
    pub max_tool_calls: u32,
    /// Maximum cost in USD (for hybrid deployments).
    pub max_cost_usd: f64,
}
```

**Example manifest (`agent.toml`):**

```toml
name = "code-analyst"
version = "0.1.0"
description = "Analyzes Rust codebases using local inference"
privacy = "sovereign"

[model]
model_path = "models/llama-3-8b-q4k.gguf"
max_tokens = 4096
temperature = 0.3
system_prompt = """
You are a Rust code analysis agent for the Sovereign AI Stack.
Use the rag tool to search documentation before answering.
Use the memory tool to store findings for later recall.
"""

[resources]
max_iterations = 20
max_tool_calls = 50
max_cost_usd = 0.0  # sovereign = free

[[capabilities]]
type = "rag"

[[capabilities]]
type = "memory"
```

### 3.7 Capability System

```rust
/// Capability grants for tools (Poka-Yoke pattern).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Capability {
    /// Access RAG pipeline for document retrieval.
    Rag,
    /// Read/write agent memory.
    Memory,
    /// Execute shell commands (sandboxed).
    Shell { allowed_commands: Vec<String> },
    /// Launch headless browser via probar (navigate, screenshot, eval WASM).
    Browser,
    /// Invoke sub-inference on a different model.
    Inference,
    /// Submit work to repartir compute pool.
    Compute,
    /// Network egress (blocked in Sovereign tier).
    Network { allowed_hosts: Vec<String> },
    /// MCP tool from external server (agents-mcp feature).  [F-020]
    Mcp { server: String, tool: String },
}

/// Check if granted capabilities satisfy a required capability.
pub fn capability_matches(
    granted: &[Capability],
    required: &Capability,
) -> bool {
    granted.iter().any(|g| match (g, required) {
        (Capability::Rag, Capability::Rag) => true,
        (Capability::Memory, Capability::Memory) => true,
        (Capability::Browser, Capability::Browser) => true,
        (Capability::Shell { allowed_commands: g },
         Capability::Shell { allowed_commands: r }) => {
            r.iter().all(|cmd| g.contains(cmd) || g.iter().any(|p| p == "*"))
        }
        (Capability::Inference, Capability::Inference) => true,
        (Capability::Compute, Capability::Compute) => true,
        (Capability::Network { allowed_hosts: g },
         Capability::Network { allowed_hosts: r }) => {
            r.iter().all(|h| g.contains(h) || g.iter().any(|p| p == "*"))
        }
        (Capability::Mcp { server: gs, tool: gt },     // [F-020]
         Capability::Mcp { server: rs, tool: rt }) => {
            (gs == rs || gs == "*") && (gt == rt || gt == "*")
        }
        _ => false,
    })
}
```

```
[REVIEW-003] @noah 2026-02-28
Toyota Principle: Poka-Yoke (Mistake-Proofing)
Capability matching prevents tool access violations at compile-time
(type system) and runtime (pattern matching). A Sovereign agent
cannot accidentally invoke Network tools. [Ref 21, 22]
Status: DRAFT
```

---

## 4. Agent Loop Algorithm

### 4.1 LoopGuard

```rust
/// Prevents runaway agent loops (Jidoka pattern).
pub struct LoopGuard {
    max_iterations: u32,
    current_iteration: u32,
    tool_call_hashes: Vec<u64>,  // FxHash of (tool_name, input) for ping-pong detection  [F-012]
    max_tool_calls: u32,
    total_tool_calls: u32,
    consecutive_max_tokens: u32, // [F-016] track consecutive MaxTokens responses
    cost_breaker: CostCircuitBreaker,  // reuses serve::circuit_breaker
}

pub enum LoopVerdict {
    /// Proceed with execution.
    Allow,
    /// Proceed but warn (approaching limits).
    Warn(String),
    /// Block this specific tool call (repeated pattern).
    Block(String),
    /// Hard stop the entire loop.
    CircuitBreak(String),
}
```

Detection heuristics:
- **Ping-pong [F-012]**: 64-bit hash of `(tool_name, input)` — if same hash appears 3+ times, Block. Uses FxHash (fast, non-cryptographic) — SHA256 is unnecessary for a small set of ~50 hashes where collision resistance is not a security requirement. Theoretically grounded: Tacheny [Ref 17] formalizes agentic loops as discrete dynamical systems and proves that oscillatory dynamics (cycling among semantic attractors) are a classifiable regime — ping-pong detection is detecting the oscillatory regime.
- **Consecutive MaxTokens [F-016]**: Counter incremented on each consecutive `StopReason::MaxTokens` response, reset on `EndTurn` or `ToolUse`. Circuit-breaks at 5 consecutive (Jidoka: stop on repeated truncation — likely a context overflow loop).
- **Monotonic budget**: iteration count, tool call count, cost accumulator
- **Cost circuit breaker**: reuses `serve::circuit_breaker::CostCircuitBreaker`

### 4.2 Loop Pseudocode

```
fn run_agent_loop(manifest, query, driver, tools, memory, stream_tx):
    guard = LoopGuard::new(manifest.resources)

    // ContextManager takes only messages; system+tool tokens are pre-subtracted  [F-003]
    // Reserve budget: system_prompt + tool_schemas + output_reserve
    system_tokens = estimate_tokens(manifest.model.system_prompt)
    tool_tokens = estimate_tokens(tools.definitions_json())
    effective_window = driver.context_window() - system_tokens - tool_tokens
    context = ContextManager::new(ContextConfig {
        window: ContextWindow::new(effective_window, manifest.model.max_tokens as usize),
        strategy: TruncationStrategy::SlidingWindow,
        preserve_system: false,  // system is separate, not in messages
        min_messages: 2,
    })

    // ═══ PERCEIVE ═══
    emit(PhaseChange::Perceive)
    memories = memory.recall(query, limit=5, agent_id=manifest.name)
    system = manifest.model.system_prompt
    if memories.not_empty():
        system += "\n\n## Recalled Context\n" + format(memories)

    messages = [Message::User(query)]

    consecutive_max_tokens = 0  // [F-016] track consecutive MaxTokens

    for iteration in 0..manifest.resources.max_iterations:
        // ═══ REASON ═══
        emit(PhaseChange::Reason)

        // Context overflow guard — truncate only messages  [F-003]
        chat_messages = messages.to_chat_messages()
        truncated = context.truncate(&chat_messages)?

        request = CompletionRequest {
            messages: messages_from_chat(truncated),
            tools: tools.definitions_for(manifest.capabilities),
            system: Some(system.clone()),
            max_tokens: manifest.model.max_tokens,
            temperature: manifest.model.temperature,
            model: manifest.model.model_name(),
        }

        response = call_with_retry(driver, request, retries=3)
        guard.record_usage(response.usage)

        match response.stop_reason:
            EndTurn:
                consecutive_max_tokens = 0
                // ═══ REMEMBER ═══
                text = response.text
                memory.remember(manifest.name, format(query, text), Conversation)
                emit(PhaseChange::Done)
                return Ok(AgentLoopResult { text, usage: guard.usage() })

            ToolUse:
                consecutive_max_tokens = 0
                for call in response.tool_calls:
                    // Poka-Yoke: capability check
                    tool = tools.get(call.name)?
                    if not capability_matches(manifest.capabilities, tool.required_capability()):
                        result = ToolResult::error("Capability denied: {}", call.name)
                    else:
                        // Jidoka: loop guard check
                        match guard.check(call):
                            Allow | Warn(_):
                                emit(PhaseChange::Act { tool: call.name })
                                result = timeout(tool.timeout(), tool.execute(call.input))
                            Block(msg):
                                result = ToolResult::error(msg)
                            CircuitBreak(msg):
                                return Err(AgentError::CircuitBreak(msg))

                    messages.push(Message::AssistantToolUse(call))
                    messages.push(Message::ToolResult(result))

            MaxTokens:
                consecutive_max_tokens += 1
                if consecutive_max_tokens >= 5:  // [F-016] Jidoka: stop on repeated truncation
                    return Err(AgentError::CircuitBreak("5 consecutive MaxTokens responses"))
                messages.push(Message::Assistant(response.text))

    Err(AgentError::MaxIterationsReached)
```

### 4.3 Error Classification and Retry

```rust
pub enum AgentError {
    /// LLM driver error (may be retryable).
    Driver(DriverError),
    /// Tool execution failed.
    ToolExecution { tool_name: String, message: String },
    /// Capability denied (Poka-Yoke).
    CapabilityDenied { tool_name: String, required: Capability },
    /// Loop guard triggered (Jidoka).
    CircuitBreak(String),
    /// Max iterations reached.
    MaxIterationsReached,
    /// Context overflow after truncation.
    ContextOverflow { required: usize, available: usize },
    /// Manifest parsing error.
    ManifestError(String),
    /// Memory substrate error.
    Memory(String),
}

pub enum DriverError {
    /// Remote API rate limited. Retryable with backoff.
    RateLimited { retry_after_ms: u64 },
    /// Remote API overloaded. Retryable with backoff.
    Overloaded { retry_after_ms: u64 },
    /// Model file not found. Not retryable.
    ModelNotFound(PathBuf),
    /// Inference failed. Not retryable.
    InferenceFailed(String),
    /// Network error (remote driver). Retryable.
    Network(String),
}
```

Retry policy: exponential backoff (1s base, 3 max retries) for RateLimited/Overloaded/Network. Immediate fail for ModelNotFound/InferenceFailed.

---

## 5. RealizarDriver (Sovereign Default)

### 5.1 Integration

```rust
pub struct RealizarDriver {
    model_path: PathBuf,
    template_engine: ChatTemplateEngine,  // reuses serve::templates
    context_window_size: usize,           // from manifest or model metadata
}

impl RealizarDriver {
    /// Construct from model path and optional context window override.
    /// Uses ChatTemplateEngine::from_model() for template detection.  [F-002, F-007]
    pub fn new(
        model_path: PathBuf,
        model_name: &str,
        context_window: Option<usize>,
    ) -> Result<Self, AgentError> {
        let template_engine = ChatTemplateEngine::from_model(model_name);
        let context_window_size = context_window.unwrap_or(4096);
        Ok(Self { model_path, template_engine, context_window_size })
    }
}

#[async_trait]
impl LlmDriver for RealizarDriver {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, AgentError> {
        // 1. Format messages using detected chat template  [F-002]
        let messages: Vec<ChatMessage> = request.messages.iter()
            .map(|m| m.to_chat_message())
            .collect();
        let prompt = self.template_engine.apply(&messages);

        // 2. Build config with struct literal  [F-007]
        let config = realizar::infer::InferenceConfig {
            model_path: self.model_path.clone(),
            prompt: Some(prompt),
            max_tokens: request.max_tokens as usize,
            temperature: request.temperature,
            ..Default::default()
        };

        // 3. Run local inference via spawn_blocking (sync → async)  [F-004]
        let result = tokio::task::spawn_blocking(move || {
            realizar::infer::run_inference(&config)
        }).await
            .map_err(|e| AgentError::Driver(DriverError::InferenceFailed(e.to_string())))?
            .map_err(|e| AgentError::Driver(DriverError::InferenceFailed(e.to_string())))?;

        // 4. Parse tool calls from output (JSON extraction)
        let (text, tool_calls) = parse_tool_calls(&result.text);

        Ok(CompletionResponse {
            text: text.clone(),
            stop_reason: if tool_calls.is_empty() { StopReason::EndTurn } else { StopReason::ToolUse },
            tool_calls,
            usage: TokenUsage {                        // [F-005]
                input_tokens: result.input_token_count as u64,
                output_tokens: result.generated_token_count as u64,
            },
        })
    }

    fn context_window(&self) -> usize {
        self.context_window_size  // [F-001] from manifest, not InferenceConfig
    }

    fn privacy_tier(&self) -> PrivacyTier {
        PrivacyTier::Sovereign  // always local
    }
}
```

### 5.2 Tool Call Parsing

Local models output tool calls as JSON in their text response (unlike API models which return structured tool_use blocks). The driver must parse these:

```
<tool_call>
{"name": "rag", "input": {"query": "how to use trueno SIMD"}}
</tool_call>
```

The parser extracts `<tool_call>` blocks and converts them to `ToolCall` structs. Malformed JSON is treated as text output (no tool call).

---

## 6. CLI Integration

### 6.1 Subcommands

```bash
# Run an agent with a manifest
batuta agent run --manifest agent.toml --prompt "Analyze this codebase"

# Run with explicit sovereign mode and model path
batuta agent run --manifest agent.toml --sovereign --model models/llama3.gguf \
    --prompt "Find security issues"

# Interactive multi-turn chat
batuta agent chat --manifest agent.toml

# Validate a manifest without running
batuta agent validate --manifest agent.toml
```

### 6.2 Example Session

```
$ batuta agent run --manifest agents/code-analyst.toml \
    --prompt "What stack components handle tensor compression?"

[perceive] Recalling 5 relevant memories...
[reason]   Generating response (llama-3-8b-q4k, sovereign)...
[act]      rag: searching "tensor compression stack components"
[reason]   Generating response with RAG context...

Tensor compression in the Sovereign AI Stack is handled by:

1. **trueno-zram-core** (v0.3.x) — SIMD-accelerated LZ4/ZSTD compression
   at 3-13 GB/s throughput. Supports AVX2, AVX-512, NEON, and CUDA.

2. **trueno** (v0.15.x) — Provides the underlying SIMD primitives
   that trueno-zram-core builds on.

3. **aprender** (v0.26.x) — APR v2 model format uses LZ4/ZSTD
   for tensor storage compression.

Usage: 2,847 input + 312 output tokens | 2 iterations | $0.00
```

---

## 7. Feature Gate

```toml
# In batuta Cargo.toml [features]  [F-014]
agents = ["native", "inference", "rag"]

# BrowserTool requires jugar-probar (separate feature, optional for Phase 1)
agents-browser = ["agents", "jugar-probar"]

# Provable design-by-contract (compile-time contract binding)
agents-contracts = ["agents", "provable-contracts", "provable-contracts-macros"]

# WASM visualization for agent dashboards in wos
agents-viz = ["agents", "presentar"]

# MCP integration: agent as client (consume external tools) and server (expose agent tools)
agents-mcp = ["agents", "pmcp", "pforge-runtime"]
```

**Dependencies [F-014, F-015]:**
- `inference` brings in realizar (local GGUF/APR inference)
- `rag` brings in trueno-rag + oracle-mode (trueno-db, trueno-graph) for RagTool
- `jugar-probar` (optional) brings in headless Chromium for BrowserTool
- `provable-contracts` + `provable-contracts-macros` (optional) enable `#[contract]` annotations and Kani harness generation
- `presentar` (optional) enables WASM Canvas2D agent dashboards in wos; `presentar-terminal` is already included via `tui` feature
- `pmcp` (optional) enables MCP client for external tool discovery; `pforge-runtime` enables MCP server exposure of agent tools

BrowserTool adds jugar-probar (~Chromium CDP bindings) as a new dependency. This is gated behind `agents-browser` so the core agent loop can be used without browser overhead. The agent module is compiled only when `agents` is enabled.

```rust
// In src/lib.rs
#[cfg(feature = "agents")]
pub mod agent;
```

---

## 8. Module File Structure

```
src/agent/
  mod.rs                        # pub exports, AgentBuilder
  runtime.rs                    # run_agent_loop()  [contract: agent-loop-v1/loop_termination]
  phase.rs                      # LoopPhase enum
  guard.rs                      # LoopGuard  [contract: agent-loop-v1/guard_budget, pingpong_detection]
  result.rs                     # AgentLoopResult
  driver/
    mod.rs                      # LlmDriver trait, CompletionRequest/Response, StreamEvent
    realizar.rs                 # RealizarDriver (local sovereign inference)
    mock.rs                     # MockDriver (deterministic, for tests)
  tool/
    mod.rs                      # Tool trait, ToolRegistry, ToolDefinition, ToolResult
    rag.rs                      # RagTool (wraps oracle::rag::RagOracle)
    memory.rs                   # MemoryTool (read/write agent state)
    browser.rs                  # BrowserTool (wraps jugar_probar::Browser for headless Chromium)
    mcp_client.rs               # McpClientTool (wraps pmcp::Client for external MCP tool servers)
    mcp_server.rs               # pforge Handler impls (expose agent tools as MCP server)
  memory/
    mod.rs                      # MemorySubstrate trait, MemoryFragment
    in_memory.rs                # InMemorySubstrate (HashMap, ephemeral)
  manifest.rs                   # AgentManifest, ModelConfig, ResourceQuota
  capability.rs                 # Capability enum, capability_matches()  [contract: agent-loop-v1/capability_match]

contracts/
  agent-loop-v1.yaml            # Design-by-contract: loop termination, capability, guard, pingpong

src/cli/agent.rs                # CLI handler for `batuta agent` subcommand
```

---

## 9. Testing Strategy

### 9.1 Unit Tests

| Module | Key Test Cases |
|---|---|
| `guard.rs` | Iteration limit enforced, ping-pong detected after 3 repeats, cost circuit break |
| `capability.rs` | Exact match, wildcard match, Shell command matching, deny on mismatch |
| `runtime.rs` | Single-turn completion, multi-turn tool use, max iteration stop, empty response retry |
| `driver/mock.rs` | Deterministic sequenced responses, tool call responses |
| `driver/realizar.rs` | Tool call parsing from text, chat template formatting |
| `memory/in_memory.rs` | Remember + recall round-trip, relevance ordering, filter by agent_id |
| `manifest.rs` | TOML deserialization, defaults, validation |
| `tool/rag.rs` | Wraps oracle RAG correctly, formats results for LLM |
| `tool/browser.rs` | Navigate, screenshot, eval_wasm actions; Sovereign restricts to localhost; timeout on navigation |
| `tool/mcp_client.rs` | Discovery from MCP server, capability-gated registration, Sovereign blocks SSE/WebSocket |
| `tool/mcp_server.rs` | pforge handler dispatch, RagToolHandler round-trip, MemoryStoreHandler persistence |

### 9.2 Integration Tests

- Full agent loop with MockDriver + InMemorySubstrate + RagTool
- Sovereign privacy enforcement: RealizarDriver only, no remote egress
- Context overflow recovery: large conversation truncated to fit window
- BrowserTool + wos: agent navigates to wos, evaluates WASM, validates output
- BrowserTool Sovereign restriction: navigation blocked for non-localhost URLs
- MCP client: agent discovers tools from mock MCP server, calls them, gets results
- MCP server: pforge serves agent tools, external client calls rag_query and gets results
- MCP Sovereign restriction: SSE/WebSocket MCP servers blocked under Sovereign privacy tier

### 9.3 Contract Verification

All agent module functions annotated with `#[contract]` are verified through:

| Layer | Tool | What It Checks |
|---|---|---|
| Compile-time | `provable-contracts-macros` | Contract binding exists (missing = compile error) |
| Property-based | `provable-contracts probar-gen` | Generated proptest harnesses from `falsification_tests` |
| Bounded model | `cargo kani` | Generated `#[kani::proof]` harnesses from `kani_harnesses` |
| Audit trail | `provable-contracts audit` | Paper → equation → contract → test → proof chain complete |
| Drift detection | `provable-contracts diff` | Contract changes between versions detected |

### 9.4 Quality Targets

| Metric | Target |
|---|---|
| Line coverage | ≥95% |
| Mutation score | ≥80% |
| Clippy warnings | 0 |
| Cyclomatic complexity | ≤30 per function |
| Cognitive complexity | ≤25 per function |
| Contract binding coverage | 100% (all equations bound to implementations) |
| Kani harness pass rate | 100% (all harnesses verify) |
| Falsification test coverage | 100% (all FALSIFY-AL-xxx tests written and passing) |

---

## 10. Phased Delivery

### Phase 1 — Sovereign Agent Loop (MVP)

Core agent loop with local inference, in-memory state, and RAG tool.

| Component | Description |
|---|---|
| `runtime.rs` | Perceive-reason-act loop |
| `guard.rs` | LoopGuard with iteration limits and ping-pong detection |
| `driver/realizar.rs` | RealizarDriver for local GGUF/APR inference |
| `driver/mock.rs` | MockDriver for deterministic testing |
| `tool/rag.rs` | RagTool wrapping oracle RAG pipeline |
| `tool/memory.rs` | MemoryTool for read/write agent state |
| `tool/browser.rs` | BrowserTool wrapping probar headless browser (navigate, screenshot, eval WASM) |
| `memory/in_memory.rs` | InMemorySubstrate (HashMap-based) |
| `manifest.rs` | AgentManifest TOML deserialization |
| `capability.rs` | Capability enum + matching |
| `cli/agent.rs` | `batuta agent run/chat/validate` commands |

### Phase 2 — Hybrid Routing + Persistence

| Component | Description |
|---|---|
| `driver/remote.rs` | RemoteDriver for Anthropic/OpenAI/Groq HTTP APIs |
| `driver/router.rs` | RoutingDriver with SpilloverRouter (local-first, remote fallback) |
| `memory/trueno.rs` | TruenoMemory (trueno-db KV + trueno-rag vectors) |
| `tool/shell.rs` | ShellTool with sandboxed subprocess execution |

### Phase 3 — Distribution + Signing

| Component | Description |
|---|---|
| `tool/compute.rs` | ComputeTool wrapping repartir for parallel tasks |
| `manifest/signing.rs` | Ed25519 manifest signing via pacha |
| Multi-agent | Agent-to-agent message passing, sub-agent spawning |

### Phase 4 — Advanced

| Component | Description |
|---|---|
| TUI dashboard | Agent status in `batuta agent status` |
| MCP client (pmcp) | `McpClientTool` — agent discovers/calls external MCP servers at runtime |
| MCP server (pforge) | Expose agent tools to Claude Code and other MCP clients via pforge YAML config |
| MCP migration | Migrate `src/mcp/McpServer` HF tools from hand-rolled JSON-RPC to pforge handlers |
| WASM core | Browser-compatible agent loop (no filesystem) |
| Tracing audit | renacer integration for tool syscall tracing |

---

## 11. Stack Integration: forjar + apr-cli + apr-qa

### 11.1 Operational Pipeline

The agent runtime sits in the middle of a larger operational pipeline:

```
┌──────────────────────────────────────────────────────────────┐
│  forjar (Infrastructure as Code)                              │
│  Provisions: GPU drivers, CUDA, model downloads, services     │
│  Shell: apr pull 'meta-llama/Llama-3-8B-GGUF' --output ...   │
└──────────────────────────┬───────────────────────────────────┘
                           │ model file on disk
┌──────────────────────────▼───────────────────────────────────┐
│  apr-qa (Model Qualification)                                 │
│  Certifies: MQS 0-1000, 217 falsification gates              │
│  Validates: SafeTensors↔APR↔GGUF parity, layout correctness  │
└──────────────────────────┬───────────────────────────────────┘
                           │ certified model
┌──────────────────────────▼───────────────────────────────────┐
│  batuta agent (Agent Runtime)                                 │
│  Loads: model via RealizarDriver                              │
│  Runs: perceive-reason-act loop with tools + memory           │
└──────────────────────────┬───────────────────────────────────┘
                           │ inference calls
┌──────────────────────────▼───────────────────────────────────┐
│  realizar (Inference Engine)                                  │
│  Executes: GGUF/APR/SafeTensors on CPU/GPU                   │
│  Kernels: Q4K/Q5K/Q6K, FlashAttention, fused ops             │
└──────────────────────────────────────────────────────────────┘
```

### 11.2 apr-cli Integration

The `apr` binary (from `aprender/crates/apr-cli`, v0.4.x) provides model lifecycle management:

| Command | Use in Agent Context |
|---|---|
| `apr pull <repo>` | Download model from HuggingFace to local cache (`~/.cache/pacha/models/`) |
| `apr convert` | Convert between SafeTensors/GGUF/APR formats |
| `apr info <model>` | Inspect model metadata (context window, quantization, architecture) |

**AgentManifest.model.model_path** can reference:
- An absolute path to a pre-downloaded model
- A pacha cache path (downloaded by `apr pull`)
- A HuggingFace repo ID (agent resolves via `apr pull` at startup)

```toml
# Option A: explicit path (forjar-provisioned)
[model]
model_path = "/models/llama-3-8b-q4k.gguf"

# Option B: pacha cache (apr pull'd)
[model]
model_path = "~/.cache/pacha/models/meta-llama/Llama-3-8B-Q4K.gguf"

# Option C: auto-pull at startup (Phase 2)
[model]
model_repo = "meta-llama/Llama-3-8B-GGUF"
model_quantization = "q4_k_m"
```

### 11.3 apr-qa Integration

The apr-model-qa-playbook (`apr-qa`, from `aprender/crates/apr-qa-cli`) validates models before agent use:

| Gate | What It Checks | Agent Relevance |
|---|---|---|
| G0 | File integrity (BLAKE3) | Model not corrupted |
| G1 | Format compliance (header, tensors) | realizar can load it |
| G2 | Inference sanity (perplexity bounds) | Agent gets coherent output |
| G3 | Format parity (SafeTensors↔APR↔GGUF) | Conversion didn't lose data |
| G4 | Garbage detection (layout violations) | No gibberish from layout bugs |

**Integration point (Phase 2):** `batuta agent validate --manifest agent.toml` runs apr-qa gateway checks on the model before allowing agent execution:

```bash
# Validate model before first agent run
batuta agent validate --manifest agent.toml
# → Runs apr-qa G0-G2 on model_path
# → Reports MQS score and any gate failures
# → Blocks agent if G0 or G1 fail (Jidoka: stop on defect)
```

### 11.4 forjar Integration

Forjar provisions the complete agent environment — bare metal, Docker containers, or pepita micro-VMs:

```yaml
# forjar.yaml — provision an agent node
version: "1.0"
name: agent-node

resources:
  nvidia-driver:
    type: gpu
    driver_version: "535"
    cuda_version: "12.3"
    persistence_mode: true

  agent-model:
    type: model
    name: llama-3-8b
    source: "meta-llama/Llama-3-8B-GGUF"
    path: /models/llama-3-8b-q4k.gguf
    format: gguf
    quantization: q4_k_m
    depends_on: [nvidia-driver]

  agent-service:
    type: service
    name: batuta-agent
    command: "batuta agent run --manifest /etc/batuta/agent.toml --daemon"
    enabled: true
    depends_on: [agent-model]
```

**Container-based deployment (Docker):**

```yaml
# forjar.yaml — containerized agent
resources:
  agent-container:
    type: docker
    image: "paiml/batuta-agent:latest"
    gpu: true
    volumes:
      - /models:/models:ro
      - /etc/batuta:/etc/batuta:ro
    command: "batuta agent run --manifest /etc/batuta/agent.toml --daemon"
    restart: always
```

**pepita micro-VM deployment:**

```yaml
# forjar.yaml — pepita kernel-level isolation
resources:
  agent-vm:
    type: pepita
    kernel: "pepita-agent-kernel"
    memory: "8G"
    cpus: 4
    gpu_passthrough: true
    rootfs: "/images/batuta-agent.ext4"
    command: "batuta agent run --manifest /etc/batuta/agent.toml"
```

Forjar handles the full provisioning matrix: GPU drivers → model download (via `apr pull`) → model certification (via `apr-qa`) → agent service start, whether on bare metal, Docker, or pepita.

```
[REVIEW-004] @noah 2026-02-28
Toyota Principle: Genchi Genbutsu (Go and See)
The full pipeline — forjar provisions hardware, apr-qa certifies
the model, batuta agent runs inference locally via realizar — ensures
every layer is verified against real hardware, not proxied through
abstractions. No cloud APIs, no trust assumptions.
Status: DRAFT
```

---

## 12. Stack Integration: probar + wos

### 12.1 Overview

Two additional stack components integrate with the agent runtime for browser-based testing and WASM agent validation:

| Component | Role in Agent Context |
|---|---|
| **probar** (v1.0.x) | Headless Chromium browser automation — agents can navigate, screenshot, evaluate JS/WASM |
| **wos** (v0.1.x) | WASM Operating System — target environment for testing agents and models in-browser |

```
┌──────────────────────────────────────────────────────────────┐
│  batuta agent (Agent Runtime)                                 │
│  Perceive-reason-act loop with tools                          │
└──────────────────────┬──────────────────┬────────────────────┘
                       │                  │
          ┌────────────▼────────┐  ┌──────▼──────────────────┐
          │  BrowserTool        │  │  wos (WASM OS)           │
          │  wraps jugar-probar │  │  Agent test target        │
          │  ┌────────────────┐ │  │  ┌────────────────────┐  │
          │  │ Browser.launch │ │  │  │ Agent WASM runtime  │  │
          │  │ Page.goto      │ │  │  │ Model inference     │  │
          │  │ Page.screenshot│ │  │  │ DOM interaction     │  │
          │  │ Page.eval_wasm │ │  │  │ aprender primitives │  │
          │  └────────────────┘ │  │  └────────────────────┘  │
          └─────────────────────┘  └──────────────────────────┘
```

```
[REVIEW-005] @noah 2026-02-28
Toyota Principle: Genchi Genbutsu (Go and See)
jugar-probar launches a real browser — not a mock DOM, not jsdom.
wos runs in real WASM — not a simulated environment.
The agent tests against actual execution targets.
Status: DRAFT
```

### 12.2 BrowserTool (wraps jugar-probar)

The `BrowserTool` exposes jugar-probar's headless Chromium capabilities as an agent tool. Agents can navigate web pages, interact with WASM applications, take screenshots, and evaluate JavaScript — enabling web testing, scraping, and WASM validation workflows.

**Crate note [F-006]:** On crates.io, `probar` (v0.1.1) is an unrelated progress bar crate. The browser testing framework is published as `jugar-probar` (v1.0.x). The dependency must be `jugar-probar = { version = "1.0", optional = true }` and imports use `jugar_probar::browser::Browser`.

```rust
use jugar_probar::browser::{Browser, BrowserConfig, Page};

pub struct BrowserTool {
    config: BrowserConfig,
    privacy_tier: PrivacyTier,
}

impl BrowserTool {
    pub fn new(privacy_tier: PrivacyTier) -> Self {
        Self {
            config: BrowserConfig::default(),
            privacy_tier,
        }
    }
}
```

**Tool actions** (selected via `input.action`):

| Action | Input | Output | probar API |
|---|---|---|---|
| `navigate` | `{ "url": "..." }` | Page title, status | `Page::goto(url)` |
| `screenshot` | `{ "selector?": "..." }` | Base64 PNG | `Page::screenshot()` |
| `evaluate` | `{ "expression": "..." }` | JSON result | `Page::evaluate(expr)` |
| `eval_wasm` | `{ "expression": "..." }` | JSON result | `Page::eval_wasm::<Value>(expr)` |
| `click` | `{ "selector": "..." }` | Success/failure | `Page::click(selector)` |
| `wait_wasm` | `{}` | Ready status | `Page::wait_for_wasm_ready()` |
| `console` | `{ "clear?": false }` | Console messages | `Page::console_messages()` |

**Capability:** `Capability::Browser` — must be explicitly granted in the agent manifest.

**Async requirement [F-017]:** All jugar-probar browser operations (`Browser::launch`, `Page::goto`, `Page::screenshot`, etc.) are `async fn` and require a tokio runtime. This is naturally satisfied by `batuta agent run` which runs inside `tokio::main`, but means BrowserTool cannot be used in synchronous contexts.

**Privacy enforcement:** Browser navigation is local (headless Chromium on the agent's machine). However, the pages loaded may make network requests. In `Sovereign` privacy tier, the `BrowserTool` restricts navigation to `localhost` and `file://` URLs only. In `Private` tier, an `allowed_origins` list can be specified. `Standard` tier permits any URL.

```toml
# Agent manifest — browser-capable agent
capabilities = [
    { type = "rag" },
    { type = "memory" },
    { type = "browser" },
]
privacy = "sovereign"  # restricts navigation to localhost/file://
```

### 12.3 wos as Agent Test Target

WOS (WASM Operating System) provides a browser-based execution environment where agents and models can be tested end-to-end. The agent uses the `BrowserTool` to interact with a wos instance:

```
Agent ──► BrowserTool ──► jugar-probar ──► headless Chromium ──► wos (WASM)
                                                            ├── kernel syscalls
                                                            ├── aprender ML
                                                            └── DOM rendering
```

**Test workflow:**

1. Agent launches browser via `BrowserTool::navigate("http://localhost:8080/wos")`
2. Agent waits for WASM initialization via `BrowserTool::wait_wasm`
3. Agent evaluates wos APIs via `BrowserTool::eval_wasm` — e.g., `wos.wos_version()`
4. Agent validates model inference output via `eval_wasm` — e.g., runs aprender Vector/Matrix ops through wos
5. Agent captures screenshots for visual regression via `BrowserTool::screenshot`
6. Agent reads console output to verify kernel syscall traces

**Example: Agent validates model in wos:**

```
$ batuta agent run --manifest agents/wos-tester.toml \
    --prompt "Verify that the Qwen2 config loads correctly in wos"

[perceive] Recalling relevant memories...
[reason]   Planning test strategy...
[act]      browser: navigate http://localhost:8080
[act]      browser: wait_wasm
[act]      browser: eval_wasm "wos.wos_version()"
[reason]   WOS v0.1.0 running. Testing Qwen2 config...
[act]      browser: eval_wasm "JSON.stringify(new wos.Qwen2Config())"
[act]      browser: screenshot
[reason]   Qwen2Config loaded: vocab_size=151936, hidden_size=896...

✓ Qwen2Config loads correctly in wos WASM environment.
  - vocab_size: 151936
  - hidden_size: 896
  - num_layers: 24

Usage: 1,203 input + 187 output tokens | 4 iterations | $0.00
```

### 12.4 pmat Quality Enforcement

All agent module code is subject to pmat quality gates. Agent functions must meet the same standards as the rest of the batuta codebase:

| Gate | Threshold | Enforcement Point |
|---|---|---|
| TDG Grade | ≥ A (85+) | `pmat analyze` on every commit |
| Cyclomatic complexity | ≤ 30 per function | Pre-commit hook (`pmat analyze complexity`) |
| Cognitive complexity | ≤ 25 per function | Pre-commit hook (`pmat analyze complexity`) |
| SATD comments | 0 | Pre-commit hook |
| Test coverage | ≥ 95% line coverage | CI gate (`pmat query --coverage-gaps`) |
| Mutation score | ≥ 80% | CI gate (`make mutants`) |

**pmat query for agent development:**

```bash
# Find agent functions by intent
pmat query "agent loop" --limit 10

# Audit agent code quality
pmat query "perceive reason act" --churn --duplicates --entropy --faults

# Find coverage gaps in agent module
pmat query --coverage-gaps --limit 20 --exclude-tests

# Find fault patterns (unwrap, panic) in agent code
pmat query "agent" --faults --exclude-tests

# Verify no high-complexity functions
pmat query "agent" --max-complexity 30 --include-source
```

```
[REVIEW-006] @noah 2026-02-28
Toyota Principle: Jidoka (Built-in Quality)
pmat enforces quality at every stage: pre-commit blocks high-complexity
functions, CI blocks low coverage, mutation testing blocks untested
logic. Quality is not inspected in — it is built in.
Status: DRAFT
```

---

## 13. Provable Design by Contract

### 13.1 Overview

The agent module is the **first batuta module** to ship with YAML design-by-contract specifications via `provable-contracts` (v0.1.x). Contracts define mathematical invariants for agent loop correctness, and `#[contract]` proc macros bind implementations to their contracts at compile time.

```
┌─────────────────────────────────────────────────────────────┐
│   contracts/agent-loop-v1.yaml                               │
│   equations: loop_termination, capability_match, guard_budget │
│   proof_obligations: invariant, bound, idempotency           │
│   falsification_tests: FALSIFY-AL-001..005                   │
│   kani_harnesses: verify_guard_budget, verify_capability     │
└──────────────────────────┬──────────────────────────────────┘
                           │ parsed by provable-contracts
┌──────────────────────────▼──────────────────────────────────┐
│   provable-contracts scaffold                                │
│   → KernelContract trait stubs + failing tests               │
│   → #[kani::proof] harnesses for bounded model checking      │
│   → probar property-based tests from falsification_tests     │
│   → Lean 4 theorem stubs (Phase 7)                           │
└──────────────────────────┬──────────────────────────────────┘
                           │ implemented in src/agent/
┌──────────────────────────▼──────────────────────────────────┐
│   #[contract("agent-loop-v1", equation = "loop_termination")]│
│   pub fn run_agent_loop(...) { ... }                         │
│                                                              │
│   #[contract("agent-loop-v1", equation = "capability_match")]│
│   pub fn capability_matches(...) { ... }                     │
└─────────────────────────────────────────────────────────────┘
```

```
[REVIEW-007] @noah 2026-02-28
Toyota Principle: Poka-Yoke (Mistake-Proofing)
YAML contracts make invariants explicit and machine-verifiable.
#[contract] macro creates a compile-time binding: if the contract
is removed or renamed, the implementation fails to compile.
Status: DRAFT
```

### 13.2 Agent Loop Contract

```yaml
# contracts/agent-loop-v1.yaml
metadata:
  version: "1.0.0"
  created: "2026-02-28"
  author: "PAIML Engineering"
  description: "Perceive-reason-act agent loop invariants"
  references:
    - "Ohno (1988) Toyota Production System — Jidoka, Poka-Yoke"

equations:
  loop_termination:
    formula: "∀ execution: iterations ≤ max_iterations ∧ tool_calls ≤ max_tool_calls"
    domain: "manifest.resources: ResourceQuota"
    codomain: "AgentLoopResult | AgentError"
    invariants:
      - "Loop always terminates (bounded by max_iterations)"
      - "Tool call count monotonically increases"
      - "Cost accumulator is non-negative and monotonically increasing"

  capability_match:
    formula: "capability_matches(granted, required) ↔ ∃g ∈ granted: g ≥ required"
    domain: "granted: [Capability], required: Capability"
    codomain: "bool"
    invariants:
      - "Empty granted set → always false (deny-by-default)"
      - "Reflexive: capability_matches([c], c) = true"
      - "Shell wildcard: Shell{['*']} matches Shell{any}"

  guard_budget:
    formula: "cost_total = Σ(pricing.calculate(usage_i)) ≤ max_cost_usd"
    domain: "usage: [TokenUsage], pricing: TokenPricing"
    codomain: "LoopVerdict"
    invariants:
      - "cost_total ≥ 0.0 (non-negative)"
      - "cost_total monotonically non-decreasing"
      - "CircuitBreak when cost_total > max_cost_usd"

  pingpong_detection:
    formula: "∀h ∈ tool_call_hashes: count(h) < 3 → Allow, count(h) ≥ 3 → Block"
    domain: "tool_call_hashes: [u64]"
    codomain: "LoopVerdict"
    invariants:
      - "First occurrence always Allow"
      - "Second occurrence always Allow"
      - "Third occurrence always Block"

proof_obligations:
  - type: invariant
    property: "Loop termination"
    formal: "iterations ≤ max_iterations"
    applies_to: all
  - type: bound
    property: "Cost non-negative"
    formal: "cost_total ≥ 0.0"
    applies_to: all
  - type: invariant
    property: "Deny-by-default capability"
    formal: "capability_matches([], _) = false"
    applies_to: all
  - type: idempotency
    property: "Capability match is pure"
    formal: "capability_matches(g, r) = capability_matches(g, r) (no side effects)"
    applies_to: all
  - type: monotonicity
    property: "Cost monotonically non-decreasing"
    formal: "cost(t+1) ≥ cost(t)"
    applies_to: all

falsification_tests:
  - id: FALSIFY-AL-001
    rule: "Loop termination"
    prediction: "run_agent_loop returns within max_iterations"
    test: "MockDriver returns ToolUse indefinitely → hits MaxIterationsReached"
    if_fails: "Guard iteration tracking broken"
  - id: FALSIFY-AL-002
    rule: "Capability deny-by-default"
    prediction: "Empty capabilities → all tool calls denied"
    test: "Agent with capabilities=[] tries tool → CapabilityDenied"
    if_fails: "capability_matches allows without grant"
  - id: FALSIFY-AL-003
    rule: "Ping-pong detection"
    prediction: "Same tool call 3x → Block"
    test: "MockDriver repeats same tool_call → 3rd returns Block"
    if_fails: "Hash collision or count threshold wrong"
  - id: FALSIFY-AL-004
    rule: "Cost circuit breaker"
    prediction: "Exceeding max_cost_usd → CircuitBreak"
    test: "MockDriver with high token counts + low budget → CircuitBreak"
    if_fails: "Cost accumulation or threshold check broken"
  - id: FALSIFY-AL-005
    rule: "Sovereign privacy"
    prediction: "Sovereign tier + Network capability → blocked"
    test: "Agent with privacy=sovereign, capabilities=[network] → tool denied"
    if_fails: "Privacy enforcement not checked at tool dispatch"

kani_harnesses:
  - name: verify_guard_budget
    obligation: "Cost non-negative"
    strategy: stub_float
    unwind: 5
    solver: minisat
  - name: verify_capability_deny_default
    obligation: "Deny-by-default capability"
    strategy: exhaustive
    unwind: 3
    solver: cadical
  - name: verify_pingpong_threshold
    obligation: "Ping-pong Block at 3"
    strategy: bounded_int
    unwind: 4
    solver: minisat
```

### 13.3 Compile-Time Contract Binding

Implementations annotate with `#[contract]` to create a traceable paper→equation→code chain:

```rust
use provable_contracts_macros::contract;

#[contract("agent-loop-v1", equation = "loop_termination")]
pub async fn run_agent_loop(
    manifest: &AgentManifest,
    query: &str,
    driver: &dyn LlmDriver,
    tools: &ToolRegistry,
    memory: &dyn MemorySubstrate,
    stream_tx: Option<tokio::sync::mpsc::Sender<StreamEvent>>,
) -> Result<AgentLoopResult, AgentError> {
    // ... implementation ...
}

#[contract("agent-loop-v1", equation = "capability_match")]
pub fn capability_matches(granted: &[Capability], required: &Capability) -> bool {
    // ... implementation ...
}

#[contract("agent-loop-v1", equation = "guard_budget")]
impl LoopGuard {
    pub fn record_usage(&mut self, usage: &TokenUsage) -> LoopVerdict {
        // ... implementation ...
    }
}
```

**Audit chain:** `provable-contracts audit` traces the full chain:
```
agent-loop-v1.yaml
  └── loop_termination
      ├── contract: equations.loop_termination.formula
      ├── binding: src/agent/runtime.rs:run_agent_loop
      ├── kani: verify_guard_budget (PASS)
      ├── falsify: FALSIFY-AL-001 (PASS)
      └── proptest: test_loop_terminates_at_max_iterations (PASS)
```

### 13.4 CI Integration

```bash
# Validate contracts parse correctly
provable-contracts validate contracts/agent-loop-v1.yaml

# Generate scaffold (trait + failing tests) — run once, then implement
provable-contracts scaffold contracts/agent-loop-v1.yaml --output src/agent/

# Generate Kani harnesses
provable-contracts kani contracts/agent-loop-v1.yaml --output src/agent/

# Run Kani verification (requires kani toolchain)
cargo kani --harness verify_guard_budget

# Audit binding coverage
provable-contracts audit contracts/ src/agent/ --format table

# Diff contract versions (detect drift)
provable-contracts diff contracts/agent-loop-v1.yaml contracts/agent-loop-v2.yaml
```

---

## 14. Presentar WASM Visualization

### 14.1 Overview

Presentar (v0.3.x) is the Sovereign AI Stack's WASM-first visualization framework. It provides Canvas2D/WebGPU rendering, widget system, and browser runtime — making it the natural rendering layer for agent dashboards in wos.

| Component | Purpose in Agent Context |
|---|---|
| `presentar-core` | Geometry, Color, DrawCommand, Event, Constraints |
| `presentar-widgets` | Prebuilt widgets (charts, tables, progress bars) |
| `presentar-layout` | Flexbox-style layout engine |
| `presentar` | WASM App runtime (Canvas2D rendering, event handling) |
| `presentar-terminal` | Terminal backend (already used by batuta TUI) |

### 14.2 Agent Dashboard in wos

When an agent targets wos, presentar provides the visualization layer for agent state, tool call history, memory recall, and loop progress:

```
batuta agent ──► wos (WASM OS) ──► presentar (rendering)
                    │                    ├── Canvas2D (browser)
                    │                    └── WebGPU (future)
                    │
                    └── BrowserTool ──► jugar-probar ──► screenshot/validate
```

**Agent dashboard widgets:**

| Widget | Displays | presentar API |
|---|---|---|
| Loop progress | Current iteration / max, phase indicator | `App::render_dashboard(title, value, progress)` |
| Tool call log | Tool name, input summary, result, latency | `App::render_json(tool_calls_json)` |
| Memory timeline | Recalled fragments, relevance scores | `App::render_json(memory_fragments_json)` |
| Token usage | Input/output tokens per iteration, cumulative cost | `App::render_dashboard("Tokens", count, budget_pct)` |
| Loop guard | Ping-pong detection, budget status, verdict | `App::render_json(guard_state_json)` |

### 14.3 Rendering Pipeline

```rust
// In wos — agent dashboard rendered via presentar
use presentar::App;
use presentar_core::DrawCommand;

#[wasm_bindgen]
pub fn render_agent_state(app: &App, state_json: &str) -> Result<(), JsValue> {
    // Parse agent loop state
    app.render_json(state_json)?;
    Ok(())
}

#[wasm_bindgen]
pub fn render_agent_dashboard(
    app: &App,
    phase: &str,
    iteration: i32,
    max_iterations: i32,
    tokens_used: f64,
    token_budget: f64,
) -> Result<(), JsValue> {
    let progress = iteration as f64 / max_iterations as f64;
    app.render_dashboard(phase, tokens_used, progress);
    Ok(())
}
```

**Integration with BrowserTool:** An agent can render its own state into wos via presentar, then use BrowserTool to screenshot the result — enabling visual regression testing of agent dashboards:

```
Agent ──► eval_wasm("render_agent_dashboard(app, 'reason', 3, 20, 1500.0, 4096.0)")
      ──► screenshot()
      ──► compare with baseline (visual regression via jugar-probar)
```

### 14.4 Terminal Fallback

When running without a browser (plain `batuta agent run`), presentar-terminal provides the same dashboard in the terminal. The batuta TUI already uses presentar-terminal for `batuta stack status`:

```rust
// Terminal rendering for non-browser agent runs
#[cfg(feature = "tui")]
use presentar_terminal::TerminalRenderer;

fn render_agent_phase_terminal(phase: &LoopPhase, iteration: u32, max: u32) {
    // Uses existing batuta TUI infrastructure
    // presentar-terminal is already a dependency
}
```

```
[REVIEW-008] @noah 2026-02-28
Toyota Principle: Standardized Work
One rendering framework (presentar) for both WASM and terminal.
Agent dashboard looks the same in wos (Canvas2D) and CLI (terminal).
No divergent rendering code paths.
Status: DRAFT
```

---

## 15. MCP Integration: pforge + pmcp

### 15.1 Current State

Batuta already has a hand-rolled MCP server in `src/mcp/` (`McpServer`) that exposes HuggingFace tools (`hf_search`, `hf_info`, `hf_tree`, `hf_integration`, `stack_status`, `stack_check`) over stdio JSON-RPC 2.0. This works but has limitations:

- Hand-rolled JSON-RPC serialization (no MCP SDK)
- No transport flexibility (stdio only)
- No tool schema derivation (manual JSON Schema)
- No client-side support (batuta can serve tools but not consume external tool servers)

The Sovereign AI Stack includes two crates that solve these problems:

| Crate | Location | crates.io | Purpose |
|---|---|---|---|
| **pmcp** | `../rust-mcp-sdk` | `pmcp` v1.10.x | MCP SDK: Client + Server, typed `ToolHandler`, stdio/SSE/WebSocket transports [Ref 23, 24] |
| **pforge** | `../pforge` | `pforge` v0.1.x | Zero-boilerplate MCP server framework: YAML config → Handler registry, built on pmcp |

### 15.2 Integration Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        batuta agent runtime                          │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  ToolRegistry                                                 │   │
│  │                                                               │   │
│  │  Builtin Tools:        MCP Client Tools (pmcp):               │   │
│  │  ├── RagTool           ├── McpClientTool("filesystem")        │   │
│  │  ├── MemoryTool        ├── McpClientTool("database")          │   │
│  │  ├── BrowserTool       └── McpClientTool("custom-server")     │   │
│  │  └── ShellTool                                                │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  pforge Server (optional)                                     │   │
│  │  Exposes agent tools to external MCP clients (Claude Code,    │   │
│  │  other agents) over stdio/SSE/WebSocket                       │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

Two integration directions:

1. **Agent as MCP Client (pmcp)**: The agent discovers and calls external MCP tool servers at runtime. This lets agents use tools provided by third-party MCP servers without hardcoding them.

2. **Agent Tools as MCP Server (pforge)**: The agent's builtin tools are exposed as an MCP server, so external LLM clients (Claude Code, other agents) can call the agent's RAG, memory, and browser tools over MCP.

### 15.3 MCP Client: Agent Consumes External Tools

The agent manifest declares MCP servers to connect to. At startup, the agent uses `pmcp::Client` to discover available tools and wraps them as `McpClientTool` instances in the `ToolRegistry`.

```toml
# In agent manifest (TOML)
[[mcp_servers]]
name = "filesystem"
transport = "stdio"
command = ["mcp-server-filesystem", "--root", "/data"]
capabilities = ["tool:filesystem:*"]

[[mcp_servers]]
name = "database"
transport = "sse"
url = "http://localhost:8080/mcp"
capabilities = ["tool:database:query"]
```

```rust
// src/agent/tool/mcp_client.rs

use pmcp::client::Client as McpClient;
use pmcp::shared::StdioTransport;

/// Wraps a single tool from an external MCP server
pub struct McpClientTool {
    server_name: String,
    tool_name: String,
    description: String,
    input_schema: serde_json::Value,
    client: Arc<McpClient>,
}

#[async_trait]
impl Tool for McpClientTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {                                  // [F-018] use input_schema, not parameters
            name: format!("mcp_{}_{}", self.server_name, self.tool_name),
            description: self.description.clone(),
            input_schema: self.input_schema.clone(),
        }
    }

    fn required_capability(&self) -> Capability {         // [F-019] separate method, not in ToolDefinition
        Capability::Mcp {                                 // [F-020] new variant
            server: self.server_name.clone(),
            tool: self.tool_name.clone(),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        let response = self.client
            .call_tool(&self.tool_name, input)
            .await
            .map_err(|e| format!("MCP call failed: {e}"))?;

        ToolResult {
            output: response.content_as_text(),
            is_error: response.is_error.unwrap_or(false),
        }
    }
}
```

**Discovery flow:**

```
Agent startup
  → parse manifest.mcp_servers[]
  → for each server:
      → spawn transport (stdio: launch process; SSE: connect)
      → pmcp::Client::initialize()
      → client.list_tools()
      → for each tool:
          → check capability_matches(manifest.capabilities, tool)
          → if allowed: registry.register(McpClientTool::new(...))
          → if denied: log warning, skip
```

**Privacy enforcement (Poka-Yoke):** MCP servers are subject to the same PrivacyTier rules as all other agent actions:

| PrivacyTier | stdio (local process) | SSE/WebSocket (network) |
|---|---|---|
| Sovereign | Allowed | **Blocked** — no network egress |
| Private | Allowed | Allowed (localhost only) |
| Standard | Allowed | Allowed |

### 15.4 MCP Server: Exposing Agent Tools via pforge

pforge allows declarative exposure of agent tools to external MCP clients. A YAML config maps agent tools to MCP-visible endpoints:

```yaml
# forge.yaml — batuta agent MCP server
forge:
  name: batuta-agent
  version: "0.1.0"
  transport: stdio

tools:
  - name: rag_query
    description: "Query the Sovereign AI Stack knowledge base"
    handler: batuta_agent::RagToolHandler
    input:
      query:
        type: string
        description: "Natural language query"
        required: true

  - name: agent_memory_recall
    description: "Recall agent memory fragments by query"
    handler: batuta_agent::MemoryRecallHandler
    input:
      query:
        type: string
        required: true
      limit:
        type: integer
        description: "Max fragments to return"

  - name: agent_memory_store
    description: "Store a memory fragment"
    handler: batuta_agent::MemoryStoreHandler
    input:
      content:
        type: string
        required: true
      metadata:
        type: object
```

```rust
// src/agent/tool/mcp_server.rs

use pforge_runtime::{Handler, HandlerRegistry};

/// Wraps RagTool as a pforge Handler
pub struct RagToolHandler {
    rag_oracle: Arc<RagOracle>,
}

#[async_trait]
impl Handler for RagToolHandler {
    type Input = RagQueryInput;
    type Output = RagQueryOutput;

    async fn handle(&self, input: Self::Input) -> Result<Self::Output, pforge_runtime::Error> {
        let results = self.rag_oracle.query(&input.query);
        Ok(RagQueryOutput {
            results: results.into_iter().map(|r| r.into()).collect(),
        })
    }
}
```

**Note [F-021]:** pforge `Handler` uses associated types (`type Input`/`type Output`), not `dyn` dispatch. pforge-runtime's `HandlerRegistry` handles type erasure internally via JSON serialization boundaries — each handler receives `serde_json::Value`, deserializes to its typed `Input`, and serializes its typed `Output` back.

**Use case:** Claude Code configured with batuta-agent as an MCP server can query the RAG knowledge base and agent memory directly, without needing to run `batuta oracle` as a subprocess.

### 15.5 Migration Path from `src/mcp/`

The existing `src/mcp/McpServer` is a hand-rolled implementation. Migration to pforge/pmcp is phased:

| Phase | Action |
|---|---|
| Phase 1 (MVP) | Agent module ignores MCP. Existing `src/mcp/` unchanged. |
| Phase 2 | Add `McpClientTool` using pmcp Client. Agent can consume external MCP servers. |
| Phase 3 | Add pforge server config. Agent tools exposed via MCP. |
| Phase 4 | Migrate existing `src/mcp/McpServer` HF tools to pforge handlers. Deprecate hand-rolled JSON-RPC. |

**Phase 4 eliminates `src/mcp/types.rs`** (130 lines of manual JSON-RPC types) — pmcp provides these. The `src/mcp/server.rs` handler logic moves to pforge Handler implementations.

### 15.6 Feature Gate

```toml
# In batuta Cargo.toml [features]
agents-mcp = ["agents", "pmcp", "pforge-runtime"]
```

```toml
# In batuta Cargo.toml [dependencies]
pmcp = { version = "1.10", optional = true }
pforge-runtime = { version = "0.1", optional = true }
```

MCP integration is optional — the core agent loop works without it. The `agents-mcp` feature enables both client (consume external MCP servers) and server (expose agent tools via MCP) capabilities.

```
[REVIEW-009] @noah 2026-02-28
Toyota Principle: Genchi Genbutsu (Go and See)
pforge + pmcp replace hand-rolled MCP code with battle-tested SDK.
Agent discovers tools at runtime from external MCP servers —
no need to hardcode every integration.
Status: DRAFT
```

---

## 16. FIRM Quality Requirements (pmat comply)

The agent module MUST meet all quality gates during development. These are not aspirational — they are **build-blocking**. Every CI run, every pre-commit, every PR. pmat comply enforces them.

### 16.1 Mandatory Thresholds

| Requirement | Threshold | Enforcement | Violation Code |
|---|---|---|---|
| **No SATD** | 0 instances | `pmat analyze satd` [F-024] | QA-001 |
| **File length** | ≤500 lines per `.rs` file | `pmat analyze size` [F-024] | QA-002 |
| **Line coverage** | ≥95% | `cargo llvm-cov` | QA-003 |
| **Cyclomatic complexity** | ≤30 per function | `pmat analyze complexity` | QA-004 |
| **Cognitive complexity** | ≤25 per function | `pmat analyze complexity` | QA-005 |
| **Mutation score** | ≥80% | cargo mutants | QA-006 |
| **Clippy warnings** | 0 | cargo clippy -D warnings | QA-007 |
| **Design-by-contract** | 100% equation binding | provable-contracts audit | QA-008 |
| **Kani harnesses** | 100% pass | cargo kani | QA-009 |
| **Zero `unwrap()`** | 0 in non-test code | clippy::unwrap_used = deny | QA-010 |
| **Zero `#[allow(dead_code)]`** | 0 instances | grep (lint) | QA-011 |

**[F-024] Tooling clarification:** SATD, file size, and complexity enforcement uses `pmat analyze` (per-project analysis subcommand), NOT `pmat comply` (cross-project consistency engine). `pmat comply` enforces Makefile, Cargo.toml, CI, and duplication rules across the stack. `pmat analyze` enforces per-file and per-function quality metrics. Both are mandatory — they are complementary, not alternatives.

### 16.2 SATD Policy (Self-Admitted Technical Debt)

**Zero SATD in the agent module.** No TODO, FIXME, HACK, XXX, TEMP, WORKAROUND comments. Every comment must describe intent, not debt.

```bash
# Pre-commit hook check
pmat analyze satd src/agent/
# Output: 0 SATD comments found → PASS
# Output: 3 SATD comments found → BLOCK (exit 1)
```

If a genuine deferral is needed, it MUST be tracked as a GitHub issue with a work item reference (e.g., `// Deferred: BATUTA-AGENT-042 — Phase 2 TruenoMemory`). The `pmat` tool allows `Deferred:` prefixed with issue references; bare `TODO:` is always blocked.

### 16.3 File Size Policy (≤500 Lines)

No source file in `src/agent/` may exceed 500 lines. This forces decomposition into focused, testable modules. The spec file structure (Section 8) already distributes responsibilities across 16 files — this threshold ensures no file accumulates incidental complexity.

**Enforcement:**

```bash
# pmat analyze size checks all .rs files
pmat analyze size src/agent/ --max-lines 500
# Violation: src/agent/runtime.rs (612 lines) > 500 → BLOCK
```

**Decomposition strategy when approaching 500 lines:**

| Pattern | Action |
|---|---|
| Large `impl` block | Extract methods into a separate module, re-export |
| Multiple test cases | Move tests to `module_tests.rs` with `#[path]` pattern |
| Long match arms | Extract each arm to a helper function |
| Type definitions + impls | Split types to `types.rs`, impls to `module.rs` |

### 16.4 Sovereign-by-Design and Airgapped Security

The agent module is **sovereign by default, airgapped by policy**:

| Layer | Policy | Enforcement |
|---|---|---|
| **Network egress** | Blocked at PrivacyTier::Sovereign | Runtime (Poka-Yoke) |
| **External API calls** | None in default feature set | Compile-time (feature gates) |
| **Model loading** | Local filesystem only (Sovereign) | RealizarDriver hardcoded |
| **Data exfiltration** | MemorySubstrate never writes to network | Type system (no network in trait) |
| **MCP servers** | SSE/WebSocket blocked in Sovereign tier | Runtime (Section 15.3) |
| **BrowserTool** | localhost-only in Sovereign tier | Runtime (Section 12.2) |
| **Dependencies** | All Sovereign AI Stack crates, zero cloud SDKs | Cargo.toml audit |
| **Supply chain** | cargo-audit, cargo-vet, BLAKE3 model hashes | CI pipeline |

**Airgapped deployment checklist:**

1. All models pre-downloaded via `apr pull` (no runtime fetches)
2. `privacy = "sovereign"` in manifest (no remote drivers)
3. No `agents-mcp` with SSE/WebSocket servers
4. No `Network` capability granted
5. Model certified via `apr-qa` G0-G4 before deployment
6. Binary built with `--locked` and vendored crates

### 16.5 Pre-Commit Quality Gate

Every commit touching `src/agent/` MUST pass:

```bash
#!/usr/bin/env bash
# .git/hooks/pre-commit (agent quality gate)
set -euo pipefail

AGENT_FILES=$(git diff --cached --name-only -- 'src/agent/*.rs' 'src/cli/agent.rs')
if [ -z "$AGENT_FILES" ]; then exit 0; fi

echo "═══ Agent Quality Gate ═══"

# Tier 1: On-save (<1s)
cargo fmt -- --check
cargo clippy --features agents -- -D warnings

# Tier 2: Pre-commit (<30s)
pmat analyze complexity src/agent/ --max-cyclomatic 30 --max-cognitive 25
pmat analyze satd src/agent/
pmat analyze size src/agent/ --max-lines 500

# Tier 3: Contract verification
provable-contracts audit contracts/ src/agent/ --format table --strict

echo "═══ Quality Gate PASSED ═══"
```

### 16.6 CI Quality Gate (Full)

```yaml
# .github/workflows/agent-quality.yml
name: Agent Quality Gate
on:
  pull_request:
    paths: ['src/agent/**', 'src/cli/agent.rs', 'contracts/**']

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Fmt + Clippy
        run: |
          cargo fmt -- --check
          cargo clippy --features agents -- -D warnings

      - name: Unit Tests
        run: cargo nextest run --features agents

      - name: Coverage (≥95%)
        run: |
          cargo llvm-cov --features agents --lcov --output-path lcov.info
          # Extract agent module coverage
          pmat query --coverage-gaps --limit 50 --exclude-tests
          # Fail if below 95%

      - name: Complexity
        run: pmat analyze complexity src/agent/ --max-cyclomatic 30 --max-cognitive 25

      - name: SATD
        run: pmat analyze satd src/agent/ --strict

      - name: File Size
        run: pmat analyze size src/agent/ --max-lines 500

      - name: Contract Audit
        run: provable-contracts audit contracts/ src/agent/ --format table --strict

      - name: Kani Verification                         # [F-025] requires nightly + kani-verifier
        run: |
          cargo install --locked kani-verifier
          cargo +nightly kani --features agents

      - name: Mutation Testing
        run: cargo mutants --features agents -- --in-diff HEAD~1
```

```
[REVIEW-010] @noah 2026-02-28
Toyota Principle: Jidoka (Stop on Defect)
Quality gates are not advisory. A single SATD comment, a single file
over 500 lines, a single function over complexity 30 — any of these
BLOCK the build. No exceptions, no overrides, no "fix later".
This is Jidoka applied to code quality: the line stops when a defect
is detected, not after it ships.
Status: DRAFT
```

---

## 17. Final Falsification (Round 2)

Popperian falsification pass on the complete v1.0 spec after all additions (Sections 12-16). Attempting to break every claim.

### F-018 MODERATE: `ToolDefinition` field inconsistency — **RESOLVED**

**Claim (Section 3.4):** `ToolDefinition` has `input_schema: serde_json::Value`.
**Claim (Section 15.3):** `McpClientTool::definition()` returns `ToolDefinition` with `parameters: self.input_schema.clone()`.
**Problem:** Field name mismatch — `input_schema` vs `parameters`.
**Fix:** Section 15.3 `McpClientTool::definition()` must use `input_schema`, not `parameters`.
**Status:** Implementation uses `input_schema` correctly (`src/agent/tool/mcp_client.rs:108`).

### F-019 MODERATE: `Tool::required_capability()` vs `ToolDefinition` duplication — **RESOLVED**

**Claim (Section 3.4):** `Tool` trait has `fn required_capability(&self) -> Capability`.
**Claim (Section 15.3):** `McpClientTool::definition()` puts capability into `ToolDefinition { required_capability: ... }`.
**Problem:** `ToolDefinition` (Section 3.4) has 3 fields: `name`, `description`, `input_schema`. It does NOT have `required_capability`.
**Fix:** `McpClientTool` implements `Tool::required_capability()` as a separate method, not inside `ToolDefinition`.
**Status:** `Tool::required_capability()` returns `Capability::Mcp { server, tool }` (`src/agent/tool/mcp_client.rs:129-134`).

### F-020 MINOR: `Capability::Tool(String)` not in enum — **RESOLVED**

**Claim (Section 15.3):** `McpClientTool` uses `Capability::Tool(format!("tool:{}:{}", ...))`.
**Problem:** Section 3.7 defines `Capability` enum with Rag, Memory, Shell, Browser, Inference, Compute, Network. No `Tool(String)` variant.
**Fix:** Add `Mcp { server: String, tool: String }` variant to Capability enum for MCP tool capabilities.
**Status:** `Capability::Mcp { server, tool }` variant added with wildcard matching (`src/agent/capability.rs:35-41`).

### F-021 MINOR: pforge `Handler` is generic, not object-safe

**Claim (Section 15.4):** `RagToolHandler` implements `Handler` with `type Input`/`type Output`.
**Problem:** pforge `Handler` uses associated types, which means it's dispatched via `HandlerRegistry` (type-erased), not via `dyn Handler`. The code snippet is correct but the text should note that pforge handles type erasure internally.
**Fix:** Add note that pforge-runtime's `HandlerRegistry` handles dynamic dispatch over typed handlers.

### F-022 MODERATE: MCP manifest config not in `AgentManifest` — **RESOLVED**

**Claim (Section 15.3):** Agent manifest declares `[[mcp_servers]]` with name, transport, command, capabilities.
**Problem:** Section 3.6 `AgentManifest` struct has no `mcp_servers` field.
**Fix:** Add `pub mcp_servers: Vec<McpServerConfig>` to `AgentManifest` (feature-gated behind `agents-mcp`).
**Status:** `McpServerConfig`, `McpTransport` added to `manifest.rs`. Validation enforces sovereign privacy blocks SSE/WebSocket. Runtime privacy check in `run_agent_loop()`.

### F-023 MINOR: `agents-mcp` feature missing from `sovereign-stack` — **RESOLVED**

**Claim (Section 7):** Feature gates list `agents`, `agents-browser`, `agents-contracts`, `agents-viz`, `agents-mcp`.
**Problem:** Should `agents-mcp` be included in `sovereign-stack`? The sovereign-stack feature (Cargo.toml line 238) includes all components. MCP is a protocol enabler, but SSE/WebSocket are runtime-blocked by PrivacyTier. Answer: yes, include — the feature enables the code, privacy tier enforces the policy.
**Fix:** Document that `sovereign-stack` should include `agents-mcp` (code compiled, SSE/WS blocked at runtime by Sovereign tier).
**Status:** `agents-mcp` added to `sovereign-stack` feature in `Cargo.toml`.

### F-024 CRITICAL: `pmat analyze satd` and `pmat analyze size` commands unverified

**Claim (Section 16):** pmat enforces SATD detection via `pmat analyze satd` and file size via `pmat analyze size`.
**Problem:** The comply module exploration shows pmat comply has 4 rules: makefile-targets, cargo-toml-consistency, ci-workflow-parity, code-duplication. SATD and file size are NOT current comply rules. They exist as `pmat analyze` subcommands (separate from comply).
**Fix:** Clarify that SATD/size/complexity enforcement uses `pmat analyze` (analysis subcommand), not `pmat comply` (cross-project consistency). The pre-commit hook already uses `pmat analyze complexity`. Add note that Phase 2 could add custom comply rules for agent-specific thresholds.

### F-025 MINOR: `cargo kani` requires nightly + kani-verifier — **DOCUMENTED**

**Claim (Section 16.6):** CI runs `cargo kani --features agents`.
**Problem:** Kani requires `cargo +nightly kani` or the `kani-verifier` toolchain to be installed. The CI workflow doesn't show Kani installation step.
**Fix:** Add Kani setup step to CI workflow, or note that Kani verification is a separate CI job with its own toolchain.
**Status:** CI workflow includes Kani setup step (Section 16.6). Kani verification deferred to Phase 3 when provable-contracts integration matures.

---

## References

1. Ohno, T. (1988). Toyota Production System: Beyond Large-Scale Production.
2. OpenFang Agent OS — Agent loop, tool dispatch, memory substrate, capability system.
3. Batuta Model Serving Ecosystem Spec v1.0 — PrivacyTier, SpilloverRouter, CostCircuitBreaker.
4. Batuta Oracle Mode Spec v1.0 — RAG pipeline, knowledge graph, query engine.
5. apr-cli (aprender/crates/apr-cli) — `apr pull`, model lifecycle management.
6. apr-model-qa-playbook — Model qualification, MQS scoring, falsification gates.
7. forjar — Infrastructure as Code for GPU provisioning and model deployment.
8. jugar-probar (v1.0.x) — Rust-native headless browser testing framework, Chromium CDP. Note: `probar` on crates.io is an unrelated progress bar crate.
9. wos (v0.1.x) — WASM Operating System, browser-based agent/model testing target.
10. pmat — Quality analysis, TDG scoring, complexity gates, coverage gap detection.
11. provable-contracts (v0.1.x) — YAML contract → Kani verification, scaffold generation, `#[contract]` proc macro for compile-time binding.
12. provable-contracts-macros (v0.1.x) — `#[contract("name", equation = "eq")]` proc macro for audit-traceable implementation bindings.
13. presentar (v0.3.x) — WASM-first visualization framework, Canvas2D/WebGPU rendering, widget system, browser App runtime.
14. presentar-terminal (v0.3.x) — Terminal backend for presentar, zero-allocation rendering. Already used by batuta TUI.
15. pmcp (v1.10.x) — Rust MCP SDK: Client + Server, typed ToolHandler, stdio/SSE/WebSocket transports. Published as `pmcp` on crates.io.
16. pforge (v0.1.x) — Zero-boilerplate MCP server framework: YAML config → Handler registry, built on pmcp. Includes pforge-runtime (Handler trait, HandlerRegistry, dispatch) and pforge-config (YAML parser).

### arXiv Citations

17. Tacheny, N. (2024). "Geometric Dynamics of Agentic Loops in Large Language Models: Trajectories, Attractors and Dynamical Regimes in Semantic Space." arXiv:2512.10350. — Formalizes agent loops as discrete dynamical systems; classifies loop dynamics as contractive (convergent), oscillatory (ping-pong), or exploratory (divergent). Directly motivates LoopGuard ping-pong detection (Section 4.1).
18. Singh, A. et al. (2025). "Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG." arXiv:2501.09136. — Survey of agentic RAG architectures using reflection, planning, tool use, and multi-agent collaboration. Validates the perceive-reason-act pattern with RAG tool integration (Section 4.2).
19. Liu, S. et al. (2025). "Memory in the Age of AI Agents: A Survey." arXiv:2512.13564. — Taxonomy of agent memory: lightweight semantic, entity-centric, episodic/reflective, structured/hierarchical. Validates MemorySubstrate trait design with recall+remember+set/get (Section 3.5).
20. Xia, Y. et al. (2025). "Anatomy of Agentic Memory: Taxonomy and Empirical Analysis of Evaluation and System Limitations." arXiv:2602.19320. — Empirical analysis of memory system limitations; identifies that substring matching (Phase 1) vs semantic recall (Phase 2) is a known tradeoff. Validates phased memory approach.
21. Li, X. et al. (2024). "GuardAgent: Safeguard LLM Agents via Knowledge-Enabled Reasoning." arXiv:2406.09187. — Proposes guardrail agents for tool-use safety. Validates capability-based access control pattern (Section 3.7) and LoopGuard circuit breaking.
22. Liu, M. et al. (2025). "Secure and Efficient Access Control Framework for Computer-Use Agents via Context Space." arXiv:2509.22256. — Context-space access control for agents with tool-use capabilities. Supports PrivacyTier enforcement model (Sections 3.7, 16.4).
23. Bajaj, A. et al. (2025). "A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, and ANP." arXiv:2505.02279. — Comprehensive survey of MCP, ACP, A2A, ANP protocols. Validates MCP as the standard for tool discovery and typed invocation (Section 15).
24. Rahman, M. et al. (2025). "Model Context Protocol (MCP): Landscape, Security Threats, and Future Research Directions." arXiv:2503.23278. — MCP security analysis: prompt injection, tool poisoning, permission escalation. Motivates capability gating of MCP tools and Sovereign SSE/WS blocking (Section 15.3).
25. Guo, Y. et al. (2025). "Sovereign-by-Design: A Reference Architecture for AI and Blockchain Enabled Systems." arXiv:2602.05486. — Reference architecture for sovereign AI with locality and lifecycle governance. Validates airgapped deployment model and data governance policies (Section 16.4).
26. Singh, A. et al. (2024). "Privacy-Preserving Large Language Models: Mechanisms, Applications, and Future Directions." arXiv:2412.06113. — Survey of privacy-preserving LLM techniques. Validates local-only inference as the strongest privacy guarantee (Section 5).
27. Tacheny, N. (2024). "LLM-Verifier Convergence: The 4/δ Bound for Designing Predictable LLM-Verifier Systems." arXiv:2512.02080. — First formal framework with provable guarantees for termination in multi-stage verification pipelines. Supports provable design-by-contract approach (Section 13).
28. Mehra, D. et al. (2025). "Agentic AI: Architectures, Taxonomies, and Evaluation." arXiv:2601.12560. — Comprehensive taxonomy: Perception, Brain, Planning, Action, Tool Use, Collaboration. Validates the module decomposition into driver/tool/memory/capability (Section 2.1).
29. Brückner, F. et al. (2025). "Fundamentals of Building Autonomous LLM Agents." arXiv:2510.09244. — Practical foundations for agent architectures with perceive-plan-act loops and memory. Confirms tool dispatch + retry + guard design patterns (Section 4).
