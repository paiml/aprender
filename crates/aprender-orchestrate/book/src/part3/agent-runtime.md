# Agent Runtime

The Batuta Agent Runtime provides autonomous agent execution using the
perceive-reason-act pattern. All inference runs locally by default
(sovereign privacy), with optional remote fallback for hybrid deployments.

## Architecture

```
AgentManifest (TOML)
  → PERCEIVE: recall memories (BM25 / substring)
  → REASON:   LlmDriver.complete() with retry+backoff
  → ACT:      Tool.execute() with capability checks
  → GUARD:    LoopGuard checks iteration/cost/ping-pong
  → repeat until Done or circuit-break
```

### Module Structure

```
src/agent/
  mod.rs          # AgentBuilder, pub exports
  runtime.rs      # run_agent_loop() — core perceive-reason-act
  phase.rs        # LoopPhase (Perceive, Reason, Act, Done, Error)
  guard.rs        # LoopGuard (Jidoka: iteration/cost/ping-pong/token budget)
  guard_tests.rs  # Unit + property tests for LoopGuard
  result.rs       # AgentLoopResult, AgentError, StopReason
  manifest.rs     # AgentManifest TOML config
  capability.rs   # Capability enum, capability_matches() (Poka-Yoke)
  pool.rs         # AgentPool, MessageRouter — multi-agent fan-out/fan-in
  signing.rs      # Ed25519 manifest signing via pacha+blake3
  contracts.rs    # Design-by-Contract YAML verification
  tui.rs          # AgentDashboardState (always), event application
  tui_render.rs   # AgentDashboard rendering (feature: presentar-terminal)
  driver/
    mod.rs        # LlmDriver trait, CompletionRequest/Response
    realizar.rs   # RealizarDriver — sovereign local inference
    mock.rs       # MockDriver — deterministic testing
    remote.rs         # RemoteDriver — Anthropic/OpenAI HTTP
    remote_stream.rs  # SSE streaming parsers + response parsers
    router.rs         # RoutingDriver — local-first with fallback
  tool/
    mod.rs        # Tool trait, ToolRegistry
    rag.rs        # RagTool — wraps oracle::rag::RagOracle
    inference.rs  # InferenceTool — sub-model invocation
    memory.rs     # MemoryTool — read/write agent state
    shell.rs      # ShellTool — sandboxed command execution
    compute.rs    # ComputeTool — parallel task execution
    network.rs    # NetworkTool — HTTP with host allowlisting
    browser.rs    # BrowserTool — headless Chromium (agents-browser)
    spawn.rs      # SpawnTool — depth-bounded sub-agent delegation
    mcp_client.rs # McpClientTool, StdioMcpTransport
    mcp_server.rs # HandlerRegistry — expose tools via MCP
  memory/
    mod.rs        # MemorySubstrate trait, MemoryFragment
    in_memory.rs  # InMemorySubstrate (ephemeral)
    trueno.rs     # TruenoMemory (SQLite + FTS5 BM25)
```

## Toyota Production System Principles

| Principle | Application |
|-----------|------------|
| **Jidoka** | `LoopGuard` stops on ping-pong, budget, max iterations |
| **Poka-Yoke** | Capability system prevents unauthorized tool access |
| **Muda** | Cost circuit breaker prevents runaway spend |
| **Heijunka** | `RoutingDriver` balances load between local and remote |
| **Genchi Genbutsu** | Default sovereign — local hardware, no proxies |

## LlmDriver Trait

The driver abstraction separates the agent loop from inference backends:

```rust
#[async_trait]
pub trait LlmDriver: Send + Sync {
    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, AgentError>;

    fn context_window(&self) -> usize;
    fn privacy_tier(&self) -> PrivacyTier;

    /// Estimate cost in USD for a completion's token usage.
    /// Default: 0.0 (sovereign/local inference is free).
    fn estimate_cost(&self, _usage: &TokenUsage) -> f64 { 0.0 }
}
```

### Cost Budget Enforcement (INV-005)

After each LLM completion, the runtime estimates cost via
`driver.estimate_cost(usage)` and feeds it to
`guard.record_cost(cost)`. When accumulated cost exceeds
`max_cost_usd`, the guard triggers a `CircuitBreak` (Muda
elimination — prevent runaway spend).

| Driver | Cost Model |
|--------|-----------|
| `RealizarDriver` | 0.0 (sovereign, free) |
| `MockDriver` | Configurable via `with_cost_per_token(rate)` |
| `RemoteDriver` | $3/$15 per 1M tokens (input/output) |

### Available Drivers

| Driver | Privacy | Feature | Use Case |
|--------|---------|---------|----------|
| `RealizarDriver` | Sovereign | `inference` | Local GGUF/APR inference |
| `MockDriver` | Sovereign | `agents` | Deterministic testing |
| `RemoteDriver` | Standard | `native` | Anthropic/OpenAI APIs |
| `RoutingDriver` | Configurable | `native` | Local-first with remote fallback |

### RemoteDriver

The `RemoteDriver` supports both Anthropic Messages API and OpenAI Chat
Completions API for hybrid deployments:

| Provider | Endpoint | Tool Format |
|----------|----------|-------------|
| Anthropic | `/v1/messages` | `tool_use` content blocks |
| OpenAI | `/v1/chat/completions` | `function` tool_calls |

Error mapping: HTTP 429 → RateLimited, 529/503 → Overloaded, other → Network.

### RoutingDriver

The `RoutingDriver` wraps a primary (typically local/sovereign) and fallback
(typically remote/cloud) driver with three strategies:

| Strategy | Behavior |
|----------|----------|
| `PrimaryWithFallback` | Try primary; on retryable error, spillover to fallback |
| `PrimaryOnly` | Primary only, no fallback |
| `FallbackOnly` | Fallback only, skip primary |

Privacy tier inherits the most permissive of the two drivers — if the
fallback is `Standard`, data *may* leave the machine on spillover.
Metrics track primary attempts, spillovers, and fallback success rate.

The CLI automatically selects the driver based on manifest configuration:
- `model_path` only → `RealizarDriver` (sovereign)
- `remote_model` only → `RemoteDriver` (cloud API)
- Both → `RoutingDriver` (local-first with remote fallback)
- Neither → `MockDriver` (dry-run)

API keys are read from `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` environment
variables based on the model identifier prefix.

### Streaming (SSE)

The `LlmDriver` trait supports optional streaming via `stream()`:

```rust
async fn stream(
    &self,
    request: CompletionRequest,
    tx: mpsc::Sender<StreamEvent>,
) -> Result<CompletionResponse, AgentError>;
```

The default implementation wraps `complete()` in a single `TextDelta` +
`ContentComplete` pair. `RemoteDriver` overrides with native SSE parsing:

| Provider | SSE Format | Tool Call Accumulation |
|----------|-----------|----------------------|
| Anthropic | `content_block_start/delta/stop`, `message_delta` | `partial_json` concatenation |
| OpenAI | `choices[0].delta`, `[DONE]` sentinel | Indexed `tool_calls` array |

Stream events:

| Event | Content |
|-------|---------|
| `TextDelta` | Incremental text token |
| `ToolUseStart` | Tool call ID + name |
| `ToolUseEnd` | Tool result |
| `ContentComplete` | Final stop reason + usage |
| `PhaseChange` | Loop phase transition |

SSE parsers live in `remote_stream.rs` (extracted for QA-002 ≤500 lines).

## Tool System

Tools extend agent capabilities. Each declares a required `Capability`;
the manifest must grant it (Poka-Yoke error-proofing):

```rust
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;
    fn definition(&self) -> ToolDefinition;
    async fn execute(&self, input: serde_json::Value) -> ToolResult;
    fn required_capability(&self) -> Capability;
    fn timeout(&self) -> Duration;
}
```

### Builtin Tools

| Tool | Capability | Description |
|------|-----------|-------------|
| `MemoryTool` | `Memory` | Read/write agent persistent state |
| `RagTool` | `Rag` | Search indexed documentation via BM25+vector |
| `ShellTool` | `Shell` | Sandboxed subprocess execution with allowlisting |
| `ComputeTool` | `Compute` | Parallel task execution via JoinSet |
| `BrowserTool` | `Browser` | Headless Chromium automation |
| `NetworkTool` | `Network` | HTTP GET/POST with host allowlisting |
| `SpawnTool` | `Spawn` | Depth-bounded sub-agent delegation |
| `InferenceTool` | `Inference` | Sub-model invocation for chain-of-thought |
| `McpClientTool` | `Mcp` | Proxy tool calls to external MCP servers |

### ShellTool Security (Poka-Yoke)

The `ShellTool` executes shell commands with multi-layer protection:

1. **Allowlist**: Only commands in the `allowed_commands` list can execute
2. **Injection prevention**: Metacharacters (`;|&&||`$()`) are blocked
3. **Working directory**: Restricted to configured path
4. **Output truncation**: Capped at 8192 bytes
5. **Timeout**: Default 30 seconds, configurable

### ComputeTool

Parallel task execution for compute-intensive workflows:

- Single task execution (`run` action)
- Parallel execution (`parallel` action) via tokio `JoinSet`
- Max concurrent tasks configurable (default: 4)
- Output truncated to 16KB per task
- Configurable timeout (default: 5 minutes)

### MCP Client Tool

The `McpClientTool` wraps external MCP servers as agent tools. Each tool
discovered from an MCP server becomes a separate `McpClientTool` instance:

```rust
use batuta::agent::tool::mcp_client::{McpClientTool, McpTransport};

let tool = McpClientTool::new(
    "code-search",              // server name
    "search",                   // tool name
    "Search codebase",          // description
    serde_json::json!({ ... }), // input schema
    Box::new(transport),        // McpTransport impl
);
```

| Aspect | Detail |
|--------|--------|
| Name format | `mcp_{server}_{tool}` |
| Capability | `Mcp { server, tool }` with wildcard support |
| Privacy | Sovereign tier restricts to stdio transport only |
| Timeout | Default 30 seconds, configurable |

Capability matching supports wildcards: `Mcp { server: "code-search", tool: "*" }`
grants access to all tools on the `code-search` server.

#### StdioMcpTransport

The `StdioMcpTransport` launches a subprocess and communicates via
JSON-RPC 2.0 over stdin/stdout. Allowed in Sovereign tier (no network).

```rust
use batuta::agent::tool::mcp_client::StdioMcpTransport;

let transport = StdioMcpTransport::new(
    "code-search",
    vec!["node".into(), "server.js".into()],
);
```

### Tool Output Sanitization (Poka-Yoke)

All tool results are sanitized before entering the conversation history.
The `ToolResult::sanitized()` method strips known prompt injection patterns:

| Pattern | Example |
|---------|---------|
| ChatML system | `<\|system\|>`, `<\|im_start\|>system` |
| LLaMA instruction | `[INST]`, `<<SYS>>` |
| Override attempts | `IGNORE PREVIOUS INSTRUCTIONS`, `DISREGARD PREVIOUS` |
| System override | `NEW SYSTEM PROMPT:`, `OVERRIDE:` |

Matching is case-insensitive. Detected patterns are replaced with `[SANITIZED]`.
This prevents a malicious tool output from hijacking the LLM's behavior.

## Multi-Agent Pool

The `AgentPool` manages concurrent agent instances with fan-out/fan-in
patterns. Each spawned agent runs its own perceive-reason-act loop in
a separate tokio task.

```rust
use batuta::agent::pool::{AgentPool, SpawnConfig};

let mut pool = AgentPool::new(driver, 4);  // max 4 concurrent

// Fan-out: spawn multiple agents
pool.spawn(SpawnConfig {
    manifest: summarizer_manifest,
    query: "Summarize this doc".into(),
})?;
pool.spawn(SpawnConfig {
    manifest: extractor_manifest,
    query: "Extract entities".into(),
})?;

// Fan-in: collect all results
let results = pool.join_all().await;
```

| Method | Purpose |
|--------|---------|
| `spawn(config)` | Spawn a single agent, returns `AgentId` |
| `fan_out(configs)` | Spawn multiple agents at once |
| `join_all()` | Wait for all agents, return `HashMap<AgentId, Result>` |
| `join_next()` | Wait for next agent to complete |
| `abort_all()` | Cancel all running agents |

Capacity enforcement: `spawn` returns `CircuitBreak` error when the pool
is at `max_concurrent`. This prevents unbounded resource consumption (Muda).

### SpawnTool (Agent-Callable Sub-Agent Delegation)

The `SpawnTool` lets an agent delegate work to a child agent as a tool call.
The child runs its own perceive-reason-act loop and returns its response.

```toml
# Enable in manifest:
[[capabilities]]
type = "spawn"
max_depth = 3
```

Depth tracking prevents unbounded recursive spawning (Jidoka):
- `current_depth` tracks how deep the spawn chain is
- Tool returns error when `current_depth >= max_depth`
- Child agents get reduced `max_iterations` (capped at 10)

### NetworkTool (HTTP Requests with Privacy Enforcement)

The `NetworkTool` allows agents to make HTTP GET/POST requests with
host allowlisting. Sovereign tier blocks all network (Poka-Yoke).

```toml
# Enable in manifest:
[[capabilities]]
type = "network"
allowed_hosts = ["api.example.com", "internal.corp"]
```

Security: requests to hosts not in `allowed_hosts` are rejected.
Wildcard `["*"]` allows all hosts (not recommended for Sovereign tier).

### BrowserTool (Headless Browser Automation)

The `BrowserTool` wraps `jugar-probar` for headless Chromium automation.
Requires `agents-browser` feature and `Capability::Browser`.

```toml
[[capabilities]]
type = "browser"
```

Privacy enforcement: Sovereign tier restricts navigation to
`localhost`, `127.0.0.1`, and `file://` URLs only.

### RagTool (Document Retrieval)

The `RagTool` wraps `oracle::rag::RagOracle` for hybrid document retrieval
(BM25 + dense, RRF fusion). Requires `rag` feature and `Capability::Rag`.

```toml
[[capabilities]]
type = "rag"
```

The oracle indexes Sovereign AI Stack documentation. Query results include
source file, component, line range, and relevance score. Feature-gated
behind `#[cfg(feature = "rag")]`.

### InferenceTool (Sub-Model Invocation)

The `InferenceTool` allows an agent to run a secondary LLM completion
for chain-of-thought delegation or specialized reasoning sub-tasks.
Requires `Capability::Inference`.

```toml
[[capabilities]]
type = "inference"
```

The tool accepts a `prompt` and optional `system_prompt`, runs a single
completion via the agent's driver, and returns the generated text.
Timeout is 300s (longer than standard 120s) for complex reasoning.

## Tracing Instrumentation

The agent runtime emits structured tracing spans for debugging and
observability. Enable with `RUST_LOG=batuta::agent=debug`:

| Span | Fields | When |
|------|--------|------|
| `run_agent_loop` | `agent`, `query_len` | Entire agent session |
| `tool_execute` | `tool`, `id` | Each tool call |
| `call_with_retry` | — | LLM completion with retry |
| `handle_tool_calls` | `num_calls` | Processing tool batch |

Key trace events:
- `agent loop initialized` — tools and capabilities loaded
- `loop iteration start` — iteration count, total tool calls
- `tool execution complete` — tool name, is_error, output_len
- `agent loop complete` — final iterations, tool calls, stop reason
- `retryable driver error` — attempt count, error details

## MCP Server (Handler Registry)

The `HandlerRegistry` exposes agent tools as MCP server endpoints,
allowing external LLM clients to call the agent's tools over MCP:

```rust
use batuta::agent::tool::mcp_server::{HandlerRegistry, MemoryHandler};

let mut registry = HandlerRegistry::new();
registry.register(Box::new(MemoryHandler::new(memory, "agent-id")));

// MCP tools/list
let tools = registry.list_tools();

// MCP tools/call
let result = registry.dispatch("memory", params).await;
```

| Handler | Actions | Feature | Description |
|---------|---------|---------|-------------|
| `MemoryHandler` | `store`, `recall` | `agents` | Store/search agent memory fragments |
| `RagHandler` | `search` | `rag` | Search indexed documentation via BM25+vector |
| `ComputeHandler` | `run`, `parallel` | `agents` | Execute shell commands with output capture |

The handler pattern is forward-compatible with pforge `Handler` trait.
When pforge is added as a dependency, handlers implement the pforge
trait directly for full MCP protocol compliance.

## Memory Substrate

Agents persist state across invocations via the `MemorySubstrate` trait:

| Implementation | Backend | Feature | Recall Strategy |
|---------------|---------|---------|----------------|
| `InMemorySubstrate` | HashMap | `agents` | Case-insensitive substring |
| `TruenoMemory` | SQLite + FTS5 | `rag` | BM25-ranked full-text search |

## Manifest Signing

Agent manifests can be cryptographically signed using Ed25519 via
`pacha` + BLAKE3 hashing:

```bash
# Sign a manifest
batuta agent sign --manifest agent.toml --signer "admin@paiml.com"

# Verify a signature
batuta agent verify-sig --manifest agent.toml --pubkey key.pub
```

The signing system normalizes TOML to canonical form before hashing
to ensure deterministic signatures regardless of formatting.

## Design by Contract

Formal invariants are defined in `contracts/agent-loop-v1.yaml` and
verified at test time. Six functions have compile-time `#[contract]`
bindings (via `provable-contracts-macros`, feature-gated behind
`agents-contracts`):

| Function | Contract | Equation |
|----------|----------|----------|
| `run_agent_loop` | `agent-loop-v1` | `loop_termination` |
| `capability_matches` | `agent-loop-v1` | `capability_match` |
| `LoopGuard::record_cost` | `agent-loop-v1` | `guard_budget` |
| `InferenceTool::execute` | `agent-loop-v1` | `inference_timeout` |
| `NetworkTool::execute` | `agent-loop-v1` | `network_host_allowlist` |
| `SpawnTool::execute` | `agent-loop-v1` | `spawn_depth_bound` |

| ID | Invariant | Verified By |
|----|-----------|-------------|
| INV-001 | Loop terminates within max iterations | `test_iteration_limit` |
| INV-002 | Guard counter monotonically increases | `test_counters` |
| INV-003 | Capability denied returns error | `test_capability_denied_handled` |
| INV-004 | Ping-pong detected and halted | `test_pingpong_detection` |
| INV-005 | Cost budget enforced | `test_cost_budget` |
| INV-006 | Consecutive MaxTokens circuit-breaks | `test_consecutive_max_tokens` |
| INV-007 | Conversation stored in memory | `test_conversation_stored_in_memory` |
| INV-008 | Pool capacity enforcement | `test_pool_capacity_limit` |
| INV-009 | Fan-out count preservation | `test_pool_fan_out_fan_in` |
| INV-010 | Fan-in completeness | `test_pool_join_all` |
| INV-011 | Tool output sanitization | `test_sanitize_output_system_injection` |
| INV-012 | Spawn depth bound (Jidoka) | `test_spawn_tool_depth_limit` |
| INV-013 | Network host allowlist (Poka-Yoke) | `test_blocked_host` |
| INV-014 | Inference timeout bound | `test_inference_tool_timeout` |
| INV-015 | Sovereign blocks network (Poka-Yoke) | `test_sovereign_privacy_blocks_network` |
| INV-016 | Token budget enforcement | `test_token_budget_exhausted` |

### Contract Verification

Run the contract verification example to audit all 16 invariant bindings:

```bash
cargo run --example agent_contracts --features agents
```

The `batuta agent contracts` CLI command performs live verification
against `cargo test --list` output:

```bash
batuta agent contracts --manifest examples/agent.toml
```

**Audit chain** (paper → equation → code → test):

```
contracts/agent-loop-v1.yaml
  └── INV-001 (loop-terminates)
      ├── equation: ∀ n > max_iterations ⟹ CircuitBreak
      ├── #[contract("agent-loop-v1", equation = "loop_termination")]
      │   └── src/agent/runtime.rs:run_agent_loop
      ├── test: agent::guard::tests::test_iteration_limit
      └── falsify: FALSIFY-AL-001 (infinite ToolUse → MaxIterationsReached)
```

## Falsification Tests

Popperian tests that attempt to *break* invariants, per spec §13.2:

| ID | Invariant | Test |
|----|-----------|------|
| FALSIFY-AL-001 | Loop termination | Infinite ToolUse must hit max iterations |
| FALSIFY-AL-002 | Deny-by-default | Empty capabilities deny all tool calls |
| FALSIFY-AL-003 | Ping-pong detection | Same tool call 3x triggers Block |
| FALSIFY-AL-004 | Cost circuit breaker | High tokens + low budget = CircuitBreak |
| FALSIFY-AL-005 | MaxTokens circuit break | 5 consecutive MaxTokens = CircuitBreak |
| FALSIFY-AL-006 | MaxTokens reset | Interleaved ToolUse resets counter |
| FALSIFY-AL-007 | Memory storage | Conversation stored after loop completes |
| FALSIFY-AL-008 | Sovereign privacy | Sovereign tier blocks network egress |

## Property Tests

Mutation-resistant property tests using `proptest` verify boundary
conditions across randomized inputs:

| Module | Property | Invariant |
|--------|----------|-----------|
| `guard.rs` | Loop terminates within max_iterations | INV-001 |
| `guard.rs` | Guard counter monotonically increases | INV-002 |
| `guard.rs` | Ping-pong detected at threshold=3 | INV-004 |
| `guard.rs` | Cost budget enforced for any positive budget | INV-005 |
| `guard.rs` | MaxTokens circuit-breaks at exactly 5 | INV-006 |
| `capability.rs` | Empty grants deny all capabilities | INV-003 |
| `capability.rs` | Capability matches itself (reflexivity) | — |
| `capability.rs` | Network wildcard matches any host | — |
| `capability.rs` | Shell wildcard matches any command | — |
| `capability.rs` | Spawn depth requires sufficient grant | — |
| `guard.rs` | Cost accumulation is non-negative (monotonic) | INV-005 |
| `capability.rs` | `capability_matches` is pure (idempotent) | — |
| `guard.rs` | Token budget enforced when configured | INV-016 |

## Feature Gates

```toml
agents = ["native"]                         # Core agent loop
agents-inference = ["agents", "inference"]  # Local GGUF/APR inference
agents-rag = ["agents", "rag"]              # RAG pipeline
agents-browser = ["agents", "jugar-probar"] # Headless browser tool
agents-mcp = ["agents", "pmcp", "pforge-runtime"]  # MCP client+server
agents-contracts = ["agents", "provable-contracts"] # #[contract] macros
agents-viz = ["agents", "presentar"]        # WASM agent dashboards
agents-full = ["agents-inference", "agents-rag"]    # All agent features
```

### MCP Manifest Configuration

When `agents-mcp` is enabled, `AgentManifest` gains an `mcp_servers` field
for declaring external MCP server connections:

```toml
[[mcp_servers]]
name = "code-search"
transport = "stdio"
command = ["node", "server.js"]
capabilities = ["*"]
```

| Transport | Privacy | Description |
|-----------|---------|-------------|
| `stdio` | Sovereign | Subprocess via stdin/stdout |
| `sse` | Standard only | Server-Sent Events over HTTP |
| `websocket` | Standard only | WebSocket full-duplex |

Sovereign privacy tier blocks `sse` and `websocket` transports at
both validation time and runtime (defense-in-depth Poka-Yoke).

## Model Resolution (Auto-Pull)

The `ModelConfig` supports three model resolution strategies:

```toml
# Option A: explicit local path
[model]
model_path = "/models/llama-3-8b-q4k.gguf"

# Option B: pacha cache path
[model]
model_path = "~/.cache/pacha/models/meta-llama--Llama-3-8B-GGUF-q4_k_m.gguf"

# Option C: auto-pull from HuggingFace repo
[model]
model_repo = "meta-llama/Llama-3-8B-GGUF"
model_quantization = "q4_k_m"
```

Resolution order: `model_path` > `model_repo` > None (dry-run mode).
When `model_repo` is set but the cache file is missing,
`batuta agent validate` reports the download command.

### Auto-Download via `apr pull`

Use the `--auto-pull` flag to automatically download models:

```bash
batuta agent run --manifest agent.toml --prompt "hello" --auto-pull
batuta agent chat --manifest agent.toml --auto-pull
```

This invokes `apr pull <repo>` (or `apr pull <repo>:<quant>`) as a subprocess.
The download timeout is 600 seconds (10 minutes).
Jidoka: agent startup is blocked if the download fails.

Errors are reported clearly:
- `NoRepo` — no `model_repo` in manifest
- `NotInstalled` — `apr` binary not found (install: `cargo install apr-cli`)
- `Subprocess` — download failed (network error, 404, timeout)

## Model Validation (G0-G1)

```bash
batuta agent validate --manifest agent.toml --check-model
```

| Gate | Check | Action on Failure |
|------|-------|-------------------|
| G0 | File exists, BLAKE3 integrity hash | Block agent start |
| G1 | Format detection (GGUF/APR/SafeTensors magic bytes) | Block agent start |
| G2 | Inference sanity (probe prompt, entropy check) | Warn or block |

### G2 Inference Sanity

```bash
batuta agent validate --manifest agent.toml --check-model --check-inference
```

G2 runs a probe prompt through the model and validates:
- Response is non-empty
- Character entropy is within normal bounds (1.0-5.5 bits/char)
- High entropy (> 5.5) indicates garbage output (LAYOUT-002 violation)

Shannon entropy thresholds:
- Normal English: 3.0-4.5 bits/char
- Garbage/layout-corrupted: > 5.5 bits/char
- Single repeated character: < 0.1 bits/char

## Inter-Agent Messaging

`AgentPool` includes a `MessageRouter` for agent-to-agent communication:

```rust
let mut pool = AgentPool::new(driver, 4);

// Spawn agents (auto-registered in router)
pool.spawn(config1)?;
pool.spawn(config2)?;

// Send message from supervisor to agent 1
pool.router().send(AgentMessage {
    from: 0, to: 1,
    content: "priority task".into(),
}).await?;
```

Each agent gets a bounded inbox (mpsc channel, capacity 32).
Agents auto-unregister from the router on completion.

## Quality Gates (QA)

All agent module code enforces strict quality thresholds:

| Gate | Threshold | Code |
|------|-----------|------|
| No SATD | 0 instances | QA-001 |
| File size | ≤500 lines per `.rs` file | QA-002 |
| Line coverage | ≥95% | QA-003 |
| Cyclomatic complexity | ≤30 per function | QA-004 |
| Cognitive complexity | ≤25 per function | QA-005 |
| Clippy warnings | 0 | QA-007 |
| Zero `unwrap()` | 0 in non-test code | QA-010 |
| Zero `#[allow(dead_code)]` | 0 instances | QA-011 |

CI enforced via `.github/workflows/agent-quality.yml`.

## TUI Dashboard

The agent TUI dashboard provides real-time visualization of agent loop
execution using `presentar-terminal`. Feature-gated behind `tui`.

### Module Structure

```
src/agent/
  tui.rs          # AgentDashboardState, ToolLogEntry (always available)
  tui_render.rs   # AgentDashboard rendering (feature: presentar-terminal)
```

### Dashboard State

`AgentDashboardState` tracks agent execution without any feature gates:

```rust
use batuta::agent::tui::AgentDashboardState;

let state = AgentDashboardState::from_manifest(&manifest);
state.apply_event(&stream_event);  // Update from StreamEvent

let pct = state.iteration_pct();       // 0-100
let tok = state.token_budget_pct();    // 0-100
```

| Field | Description |
|-------|-------------|
| `phase` | Current `LoopPhase` |
| `iteration` / `max_iterations` | Loop progress |
| `usage` | Cumulative `TokenUsage` |
| `tool_calls` / `tool_log` | Tool invocation history |
| `recent_text` | Last 20 text fragments |
| `cost_usd` / `max_cost_usd` | Budget tracking |
| `stop_reason` | Final `StopReason` (when done) |

### Interactive Dashboard

When the `tui` feature is enabled, `AgentDashboard` renders a full
terminal interface with progress bars, tool log, and real-time output:

```rust
use batuta::agent::tui::AgentDashboard;

let dashboard = AgentDashboard::new(state);
dashboard.run(&mut rx)?;  // Blocks until q/Esc pressed
```

Dashboard layout: title bar, phase indicator, iteration/tool/token
progress bars, token usage summary, scrolling tool log, recent output
text, and help bar. Press `q` or `Esc` to exit.

### Streaming Output

The `--stream` flag enables real-time token-by-token output during
`batuta agent run` and `batuta agent chat`:

```bash
batuta agent run --manifest agent.toml --prompt "Hello" --stream
batuta agent chat --manifest agent.toml --stream
```

Without `--stream`, events are batch-drained after the loop completes.
With `--stream`, a concurrent tokio task displays events as they arrive.

## CLI Commands

```bash
# Single-turn execution
batuta agent run --manifest agent.toml --prompt "Hello"

# With real-time streaming output
batuta agent run --manifest agent.toml --prompt "Hello" --stream

# With auto-download of model via apr pull
batuta agent run --manifest agent.toml --prompt "Hello" --auto-pull

# Interactive chat (with optional streaming)
batuta agent chat --manifest agent.toml --stream

# Validate manifest
batuta agent validate --manifest agent.toml

# Validate manifest + model file (G0-G1 gates)
batuta agent validate --manifest agent.toml --check-model

# Multi-agent fan-out
batuta agent pool \
  --manifest summarizer.toml \
  --manifest extractor.toml \
  --manifest analyzer.toml \
  --prompt "Analyze this document" \
  --concurrency 2

# Sign and verify manifests
batuta agent sign --manifest agent.toml --signer "admin"
batuta agent verify-sig --manifest agent.toml --pubkey key.pub

# Show contract invariants
batuta agent contracts

# Show manifest status
batuta agent status --manifest agent.toml
```

| Subcommand | Purpose |
|-----------|---------|
| `run` | Single-turn agent execution |
| `chat` | Interactive multi-turn session |
| `validate` | Validate manifest (+ model with `--check-model`) |
| `pool` | Fan-out multiple agents, fan-in results |
| `sign` | Ed25519 manifest signing |
| `verify-sig` | Verify manifest signature |
| `contracts` | Display contract invariant bindings |
| `status` | Show manifest configuration |

See [`batuta agent` CLI Reference](../part6/cli-agent.md) for full details.

## Runnable Examples

The `examples/` directory includes dogfooding demos that exercise the
agent APIs end-to-end. All require `--features agents`.

### Agent Demo (27 scenarios)

```bash
cargo run --example agent_demo --features agents
```

Exercises all core APIs: manifest creation, loop execution, tool
dispatch, capability enforcement, guard invariants, multi-agent pool,
MCP handlers, memory operations, signing, TUI state management,
context truncation, and streaming events.

### Contract Verification

```bash
cargo run --example agent_contracts --features agents
```

Parses `contracts/agent-loop-v1.yaml`, displays all 16 invariants with
formal equations, and verifies every test binding resolves to a real
test in the crate. Reports coverage target (95%), mutation target (80%),
and complexity thresholds.

### Memory Substrate

```bash
cargo run --example agent_memory --features agents
```

Demonstrates `InMemorySubstrate`: storing memories from conversations
and tool results, substring-based recall with filters, key-value
structured storage, and memory deletion (forget).

### Multi-Agent Pool

```bash
cargo run --example agent_pool --features agents
```

Demonstrates `AgentPool` concurrency: individual agent spawning,
capacity enforcement (CircuitBreak at max), message routing between
agents, fan-out (batch spawn), and fan-in (join_all result collection).

### Manifest Signing

```bash
cargo run --example agent_signing --features agents
```

Demonstrates Ed25519 manifest signing: keypair generation, BLAKE3
hashing + Ed25519 signing, tamper detection (modified content caught),
wrong-key detection, and TOML sidecar serialization roundtrip.

## Quality Gate Results

The agent module enforces strict quality gates per the PMAT methodology
(spec §16). Current status:

| Gate | Threshold | Status |
|------|-----------|--------|
| QA-001 SATD | Zero comments | PASS |
| QA-002 File Size | ≤500 lines | PASS |
| QA-003 Coverage | ≥95% line | PASS |
| QA-004 Cyclomatic | ≤30 per fn | PASS |
| QA-005 Cognitive | ≤25 per fn | PASS |
| QA-010 Unwrap | Zero in non-test | PASS |
| QA-011 Dead Code | Zero allow(dead_code) | PASS |

### Design-by-Contract Verification

All 16 invariants from `contracts/agent-loop-v1.yaml` are verified:

```
INV-001  loop-terminates           INV-009  fanout-count
INV-002  guard-monotonic           INV-010  fanin-complete
INV-003  capability-poka-yoke      INV-011  output-sanitization
INV-004  pingpong-halting          INV-012  spawn-depth-bound
INV-005  cost-budget               INV-013  network-host-allowlist
INV-006  truncation-circuit-break  INV-014  inference-timeout
INV-007  memory-store              INV-015  sovereign-blocks-network
INV-008  pool-capacity             INV-016  token-budget-enforcement
```

Run `cargo run --example agent_contracts --features agents` to verify.

## Specification Traceability

This page covers the complete agent specification
(`docs/specifications/batuta-agent.md`). Cross-references to related book pages:

| Spec Section | Topic | Book Location |
|-------------|-------|---------------|
| 2-4 | Core architecture, types, loop algorithm | This page |
| 5-6 | RealizarDriver, ChatTemplate integration | This page |
| 7 | Feature gates | This page: Feature Gates |
| 8-10 | Manifest, tools, memory | This page |
| 11 | Deployment (forjar) | [`batuta agent` CLI](../part6/cli-agent.md) |
| 12 | probar + wos integration | [Probar](./probar.md#agent-integration-browsertool) |
| 13 | Design by contract (provable-contracts) | This page: Design by Contract |
| 14 | Presentar WASM visualization | [Presentar](./presentar.md#agent-dashboard-integration) |
| 15 | MCP integration (pforge + pmcp) | [pmcp](./pmcp.md#agent-integration-mcp-client), [pforge](./pforge.md#agent-integration-mcp-server) |
| 16 | FIRM quality requirements | This page: Quality Gates |
| 17 | Falsification (round 2) | This page: Falsification Tests |

## apr code — Sovereign AI Coding Assistant

`apr code` is the flagship use of the agent runtime. It provides an interactive AI coding assistant that runs entirely on local hardware — no cloud, no API keys, no data egress.

### Quick Start

```bash
# Auto-discovers model from ~/.apr/models/
apr code

# Non-interactive (single prompt)
apr code -p "What is 2+2?"

# With specific model
apr code --model ~/.apr/models/qwen3-1.7b-q4k.apr
```

### Architecture

```
apr code (apr-cli)
  → batuta::agent::code::cmd_code()
    → discover_model() — finds best model (APR preferred over GGUF)
    → scale_prompt() — adapts system prompt to model size
    → AprServeDriver — launches apr serve subprocess (GPU)
    → run_agent_loop() — perceive-reason-act with 9 tools
```

### Key Features (v0.7.4)

- **9 tools**: file_read, file_write, file_edit, glob, grep, shell, memory, pmat_query, rag
- **GPU inference**: CUDA via `apr serve` with serial prefill (27.5 tok/s on RTX 4090)
- **Prompt scaling**: auto-detect model size from filename (<2B compact, 2-7B mid, 7B+ full)
- **APR format preferred**: discovery sorts valid > mtime > APR format
- **13 provable contracts**: 129 FALSIFY enforcement tests
- **Sovereign**: all inference local, `PrivacyTier::Sovereign` hardcoded

### Performance (RTX 4090, Qwen3 1.7B Q4K)

| Metric | Value |
|--------|-------|
| TTFT P99 | 348ms |
| TPOT P99 | 38.7ms |
| Throughput | 27.5 tok/s |
| Startup | 2.5-3.0s |
