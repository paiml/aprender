# Agent and Playbook Specification

> Parent: [batuta-spec.md](../batuta-spec.md)
> Sources: batuta-agent, batuta-playbook
> See also: [multi-provider-api.md](multi-provider-api.md) (provider-agnostic LLM client)
> See also: [presentar-probar-integration.md](presentar-probar-integration.md) (TUI rendering, Brick UX, pixel coverage, state machine testing)
> See also: [apr-code.md](apr-code.md) (`apr code` — user-facing entrypoint; batuta agent is the engine underneath)

---

## Part I: Agent Runtime

### 1. Overview

Batuta Agent extends the orchestration framework with an autonomous agent runtime. Agents execute perceive-reason-act loops using local LLM inference (realizar), retrieval-augmented generation (trueno-rag), and persistent memory (trueno-db) -- all sovereign by default, with zero external API dependencies.

### Non-Goals

- Not a chatbot framework (no Slack/Discord adapters)
- Not a workflow engine (batuta playbook handles that)
- Not a model trainer (entrenar does training)
- Not an API gateway (batuta serve handles endpoints)

### 2. Architecture

```
batuta/
  +-- serve/      # routing, privacy, failover, context     (REUSE)
  +-- oracle/     # RAG pipeline, knowledge graph           (REUSE)
  +-- agent/      # perceive-reason-act runtime             (NEW)
  |     +-- driver/   # LlmDriver trait + implementations
  |     +-- tool/     # Tool trait + builtin tools
  |     +-- memory/   # MemorySubstrate trait + implementations
  +-- cli/agent.rs    # CLI subcommand
```

### Reuse Matrix

| Existing Module | Reuse in Agent |
|----------------|----------------|
| `serve::backends::PrivacyTier` | Driver privacy enforcement |
| `serve::context::ContextManager` | Token counting + truncation |
| `serve::circuit_breaker::CostCircuitBreaker` | Cost budget per invocation |
| `serve::router::SpilloverRouter` | Hybrid local->remote routing |
| `serve::templates::ChatTemplateEngine` | Prompt formatting (ChatML/Llama/Mistral) |
| `oracle::rag::RagOracle` | Document retrieval tool |
| `oracle::knowledge_graph` | Stack component queries |

### 3. Agent Loop

```
                    batuta agent run
                          |
                    AgentManifest (TOML)
                          |
              +-----------v-----------+
              |       PERCEIVE        |
              | memory.recall(query)  |
              | -> inject into prompt |
              +-----------+-----------+
                          |
              +-----------v-----------+
              |        REASON         |
              | context.truncate()    |
              | driver.complete()     |
              +-----+----------+-----+
                    |          |
              end_turn    tool_use
                    |          |
                    |    +-----v---------+
                    |    |     ACT       |
                    |    | capability_check|
                    |    | loop_guard    |
                    |    | tool.execute  |
                    |    +---------------+
                    |
              +-----v-----------------+
              |      REMEMBER         |
              | memory.remember()     |
              | -> AgentLoopResult    |
              +-----------------------+
```

### 4. Core Types

#### LlmDriver Trait

```rust
#[async_trait]
pub trait LlmDriver: Send + Sync {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, AgentError>;
    async fn stream(&self, request: CompletionRequest, tx: Sender<StreamEvent>) -> Result<CompletionResponse, AgentError>;
    fn context_window(&self) -> usize;
    fn privacy_tier(&self) -> PrivacyTier;
}
```

| Driver | Privacy Tier | Backend | Phase |
|--------|-------------|---------|-------|
| `RealizarDriver` | Sovereign | Local GGUF/APR | 1 |
| `MockDriver` | Sovereign | Deterministic (testing) | 1 |
| `RemoteDriver` | Standard | HTTP to OpenAI/Anthropic | 2 |
| `RoutingDriver` | Configurable | SpilloverRouter: local-first | 2 |

#### Tool Trait

```rust
#[async_trait]
pub trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;
    async fn execute(&self, input: Value) -> Result<String, ToolError>;
    fn required_capability(&self) -> Capability;
}
```

#### MemorySubstrate Trait

```rust
#[async_trait]
pub trait MemorySubstrate: Send + Sync {
    async fn remember(&self, key: &str, content: &str, metadata: Value) -> Result<()>;
    async fn recall(&self, query: &str, limit: usize) -> Result<Vec<Memory>>;
    async fn forget(&self, key: &str) -> Result<()>;
}
```

### 5. Safety Guards

| Guard | Mechanism | Toyota Principle |
|-------|-----------|-----------------|
| **LoopGuard** | Detects ping-pong patterns, budget exhaustion, max iterations | Jidoka |
| **Capability System** | Tools require capabilities; manifest declares allowed set | Poka-Yoke |
| **CostCircuitBreaker** | Prevents runaway spend on hybrid deployments | Muda |
| **PrivacyTier** | Sovereign blocks all remote egress | Poka-Yoke |

### 6. Builtin Tools

| Tool | Capability | Description |
|------|-----------|-------------|
| `RagTool` | `rag_search` | Wraps `oracle::rag::RagOracle` for doc retrieval |
| `ShellTool` | `shell_execute` | Execute shell commands (sandboxed) |
| `FileTool` | `file_read`, `file_write` | Read/write local files |
| `BrowserTool` | `browser_navigate` | Headless Chromium via jugar-probar |
| `McpTool` | `mcp_call` | Call external MCP tool servers at runtime |

---

## Part I-B: Agent Runtime Extensions

> Inspired by CCX-RS (anton-abyzov/ccx-rs) provider-agnostic coding assistant architecture.

### 7. Context Compaction

The agent loop operates within a finite context window. As conversations grow, older tool results and message content must be compacted to stay within budget while preserving essential reasoning context.

#### 7.1 Token Estimation

```rust
pub struct TokenEstimator {
    /// Characters-per-token ratio (default: 3.5 for English text, 2.8 for code)
    ratio: f64,
}

impl TokenEstimator {
    /// Fast O(1) estimate without tokenizer
    pub fn estimate(&self, text: &str) -> u32 {
        (text.len() as f64 / self.ratio).ceil() as u32
    }

    /// Exact count via realizar tokenizer (when loaded)
    pub fn count_exact(&self, text: &str, tokenizer: &Tokenizer) -> u32;
}
```

#### 7.2 Compaction Strategy

Two-phase compaction triggered by token thresholds. Strategy adapts to model context size:

| Phase | Trigger | Action | Preserves |
|-------|---------|--------|-----------|
| **Micro-compact** | 60% of context window | Strip tool result bodies, keep summaries | All user messages, assistant reasoning, tool names + status |
| **Auto-compact** | 80% of context window | Summarize or truncate conversation history | System prompt, last N turns (configurable), memory recalls |

**FALSIFICATION FIX (C-001):** Auto-compact has two modes based on model capability:
- **Extractive** (default for context < 16K): Drop oldest tool results and assistant reasoning beyond last N turns. No LLM call required. Inject compaction summary: "Turns 1-15 compacted. Re-run tools if details needed."
- **Abstractive** (for context >= 16K): LLM summarizes history into condensed form. Only used when model has sufficient headroom (context usage < 70% AFTER dropping tool results).

This prevents the failure mode where a small model (3B, 8K context) is asked to summarize at near-capacity.

```rust
pub struct CompactionConfig {
    /// Context window size (from LlmDriver::context_window())
    pub context_window: usize,
    /// Micro-compact threshold (default: 0.6)
    pub micro_threshold: f64,
    /// Auto-compact threshold (default: 0.8)
    pub auto_threshold: f64,
    /// Turns to preserve in full after auto-compact (default: 4)
    pub preserve_recent_turns: usize,
}
```

#### 7.3 Compaction Rules

1. **System prompt**: Never compacted (always in full)
2. **User messages**: Preserved verbatim until auto-compact
3. **Tool results**: First target for micro-compact (large outputs replaced with `[truncated: N bytes]`)
4. **Assistant reasoning**: Preserved in micro-compact, summarized in auto-compact
5. **Memory recalls**: Re-fetched from MemorySubstrate after auto-compact (fresh context)

#### 7.4 Integration with Existing Modules

| Module | Role in Compaction |
|--------|-------------------|
| `serve::context::ContextManager` | Token counting + truncation (REUSE) |
| `agent::memory::MemorySubstrate` | Re-fetch relevant memories post-compact |
| `oracle::rag::RagOracle` | Re-rank context after compaction |

---

### 8. Parallel Tool Execution

When the LLM emits multiple tool_use blocks in a single response, independent tools execute concurrently. This reduces wall-clock time for multi-tool turns (e.g., reading 5 files simultaneously).

#### 8.1 Execution Model

```rust
pub struct ParallelToolExecutor {
    /// Maximum concurrent tool executions (default: 8)
    pub max_concurrency: usize,
}

impl ParallelToolExecutor {
    /// Execute all tool calls concurrently, respecting capability checks
    pub async fn execute_all(
        &self,
        calls: Vec<ToolCall>,
        tools: &ToolRegistry,
        guard: &LoopGuard,
    ) -> Vec<ToolResult> {
        let semaphore = Arc::new(Semaphore::new(self.max_concurrency));
        let futures = calls.into_iter().map(|call| {
            let permit = semaphore.clone().acquire_owned();
            async move {
                let _permit = permit.await;
                guard.check_capability(call.name())?;
                tools.execute(call).await
            }
        });
        futures::future::join_all(futures).await
    }
}
```

#### 8.2 Dependency Detection

Some tool calls have implicit dependencies (e.g., write after read of same file). The executor detects these via resource path overlap:

| Pattern | Detection | Handling |
|---------|-----------|---------|
| Read + Read (same file) | Safe | Parallel |
| Read + Write (same file) | Conflict | Sequential (read first) |
| Write + Write (same file) | Conflict | Sequential (ordered) |
| Shell + Shell | Independent | Parallel (unless piped) |
| Shell + File (overlapping path) | Potential conflict | Sequential |

---

### 9. Tool Hooks

Pre- and post-execution hooks enable extensibility without modifying tool implementations. Hooks follow the Poka-Yoke principle -- intercept mistakes before they cause damage.

#### 9.1 Hook Types

```rust
pub enum HookTiming {
    /// Runs before tool execution; can block or modify input
    Pre,
    /// Runs after tool execution; can modify output or trigger side effects
    Post,
}

#[async_trait]
pub trait ToolHook: Send + Sync {
    fn timing(&self) -> HookTiming;
    fn applies_to(&self) -> Vec<String>;  // Tool names, or ["*"] for all
    async fn execute(&self, context: &HookContext) -> Result<HookAction>;
}

pub enum HookAction {
    /// Continue with (possibly modified) input/output
    Continue(Value),
    /// Block the tool call with a reason
    Block(String),
    /// Log and continue unchanged
    Observe,
}
```

#### 9.2 Builtin Hooks

| Hook | Timing | Purpose | Toyota Principle |
|------|--------|---------|-----------------|
| `ConfirmDestructiveHook` | Pre | Require user confirmation for file writes, shell commands with `rm`, `git push --force` | Poka-Yoke |
| `CostCheckHook` | Pre | Warn before expensive operations (large file reads, long shell commands) | Muda |
| `AuditLogHook` | Post | Append tool call + result to audit trail | Genchi Genbutsu |
| `SandboxValidationHook` | Pre | Verify shell commands against sandbox policy | Jidoka |
| `SecretScanHook` | Pre | Block file writes containing API keys, tokens, credentials | Poka-Yoke |

#### 9.3 Hook Configuration

```toml
# agent.toml
[[hooks]]
name = "confirm-destructive"
timing = "pre"
applies_to = ["shell", "file_write"]
config = { patterns = ["rm ", "git push", "DROP TABLE"] }

[[hooks]]
name = "audit-log"
timing = "post"
applies_to = ["*"]
config = { path = ".batuta/audit.jsonl" }
```

---

### 10. OS-Native Sandboxing

Agent tool execution is sandboxed at the OS level, complementing the capability system (Poka-Yoke layer 1) with kernel-enforced restrictions (Poka-Yoke layer 2). Integrates with renacer for syscall-level audit trails.

#### 10.1 Platform Support

| Platform | Mechanism | Kernel Version | Granularity |
|----------|-----------|----------------|-------------|
| **Linux** | Landlock LSM | 5.13+ | File path + network |
| **macOS** | Seatbelt (sandbox-exec) | All supported | File path + network + IPC |
| **Windows** | AppContainer | Windows 10+ | File path + network |
| **WASM** | Capability-based (no raw I/O) | N/A | By design |

#### 10.2 Sandbox Policy

```rust
pub struct SandboxPolicy {
    /// Directories the agent can read
    pub read_paths: Vec<PathBuf>,
    /// Directories the agent can write
    pub write_paths: Vec<PathBuf>,
    /// Allowed network destinations (empty = no network)
    pub network_allow: Vec<NetworkRule>,
    /// Whether to allow process execution
    pub allow_exec: bool,
}

pub enum NetworkRule {
    /// Allow connections to specific host:port
    HostPort(String, u16),
    /// Allow connections to localhost only
    Localhost,
    /// Allow all (Standard tier only)
    Any,
}
```

#### 10.3 Sandbox per Privacy Tier

| Tier | Read | Write | Network | Exec |
|------|------|-------|---------|------|
| **Sovereign** | Project dir only | Project dir only | None | Allowlisted commands |
| **Private** | Project dir + home config | Project dir | Localhost only | Allowlisted commands |
| **Standard** | Project dir + home config | Project dir | Configured providers | Allowlisted commands |

#### 10.4 Renacer Integration

When `renacer` is available, sandbox violations are captured as syscall events and correlated with the tool call that triggered them:

```
[SANDBOX] Tool "shell" attempted write to /etc/passwd
  -> Blocked by Landlock policy (write_paths = ["/home/user/project"])
  -> Syscall: openat(AT_FDCWD, "/etc/passwd", O_WRONLY) = -EACCES
  -> Renacer event: SandboxViolation { tool: "shell", path: "/etc/passwd", action: "write" }
```

---

### 11. Session Persistence and Resume

Agent conversations persist to disk, enabling resume after interruption, crash recovery, and conversation branching.

#### 11.1 Session Storage

```
.batuta/sessions/
  +-- {session_id}/
  |     +-- manifest.json    # Session metadata (model, provider, start time)
  |     +-- messages.jsonl   # Conversation history (append-only)
  |     +-- memory.json      # Session-local memory snapshot
  |     +-- cost.json        # Cumulative cost tracking
```

#### 11.2 Core Types

```rust
pub struct Session {
    pub id: SessionId,
    pub manifest: SessionManifest,
    pub messages: Vec<Message>,
    pub memory_snapshot: Value,
    pub cumulative_cost: CostEstimate,
}

pub struct SessionManifest {
    pub created_at: DateTime<Utc>,
    pub last_active: DateTime<Utc>,
    pub model: String,
    pub provider: ProviderId,
    pub agent_manifest: PathBuf,
    pub project_root: PathBuf,
}
```

#### 11.3 Resume Behavior

| Scenario | Action |
|----------|--------|
| Normal resume | Load messages.jsonl, restore context, continue |
| Context too large | Auto-compact before resuming (see section 7) |
| Provider changed | Translate message history to new provider format |
| Model changed | Re-estimate token counts, compact if needed |
| Crash recovery | Replay from last complete message in JSONL |

#### 11.4 CLI Commands

```bash
# Resume most recent session
batuta agent resume

# Resume specific session
batuta agent resume --session {id}

# List sessions
batuta agent sessions

# Fork a session (branch conversation)
batuta agent fork --session {id}
```

---

### 12. Updated Safety Guard Matrix

Extends section 5 with new guards from CCX-RS-inspired features:

| Guard | Mechanism | Toyota Principle | New? |
|-------|-----------|-----------------|------|
| **LoopGuard** | Ping-pong detection, budget exhaustion, max iterations | Jidoka | Existing |
| **Capability System** | Tools require capabilities; manifest declares allowed set | Poka-Yoke | Existing |
| **CostCircuitBreaker** | Prevents runaway spend on hybrid deployments | Muda | Existing |
| **PrivacyTier** | Sovereign blocks all remote egress | Poka-Yoke | Existing |
| **ContextCompactor** | Prevents context window overflow, preserves reasoning | Kaizen | **New** |
| **ToolHooks** | Pre/post interception for destructive action prevention | Poka-Yoke | **New** |
| **OsSandbox** | Kernel-enforced file/network/exec restrictions | Jidoka | **New** |
| **SessionPersistence** | Crash recovery, conversation continuity | Andon | **New** |
| **ProviderFailover** | Automatic cascade to healthy provider on failure | Jidoka | **New** |
| **CostEstimator** | Per-turn cost warning before expensive API calls | Muda | **New** |

---

## Part II: Playbook System

### 7. Overview

Batuta Playbook introduces deterministic, YAML-defined workflow execution. Playbooks declare multi-stage data pipelines as directed acyclic graphs (DAGs) with content-addressed caching, hash-based invalidation, and distributed execution.

**Core Invariants:**
- **Determinism**: Identical inputs always produce identical outputs
- **Idempotency**: Re-running skips stages whose input hashes match the lock file
- **Resumability**: Failed pipelines resume from last successful stage
- **Provenance**: Every artifact is content-addressed (BLAKE3) with full lineage in pacha
- **Shell safety**: All `cmd` fields purified through bashrs before execution

### 8. YAML Schema

```yaml
version: "1.0"
name: "pipeline-name"
description: "Pipeline description"

params:
  model: "moonshine-tiny"
  chunk_size: 512

targets:
  workstation:
    host: localhost
  server:
    host: intel
    cores: 32

stages:
  extract:
    description: "Extract data"
    cmd: "ffmpeg -i {{input}} -vn {{output}}"
    deps:
      - path: /data/input/
        type: directory
    outs:
      - path: /data/output/
        type: audio
    target: workstation
    parallel:
      strategy: per_file
      glob: "**/*.mp4"
      max_workers: 8

  process:
    description: "Process extracted data"
    cmd: "tool --input {{deps[0].path}} --output {{outs[0].path}}"
    deps:
      - path: /data/output/
    params:
      - model
    outs:
      - path: /data/processed/
    target: server
    retry:
      limit: 3
      policy: on_failure
      backoff:
        initial: 5s
        factor: 2

compliance:
  pre_flight:
    - tdg:
        path: ../component/src/
        min_grade: C
  post_flight:
    - quality_gate:
        min_grade: B

policy:
  failure: stop_on_first   # Jidoka
  validation: checksum     # BLAKE3
  lock_file: true
```

### 9. DAG Construction

DAG is implicit from deps/outs: if stage B lists path X in deps and stage A lists X in outs, then B depends on A. Explicit ordering via `after` for non-data dependencies.

**Wildcard expansion** (Snakemake pattern): Paths with `{name}` expand into per-sample DAG nodes at runtime, each with independent caching and retry.

### 10. Execution Model

| Feature | Implementation | Prior Art |
|---------|---------------|-----------|
| Content addressing | BLAKE3 hashes for all artifacts | DVC |
| Lock file | Records input/output hashes per stage | DVC |
| Parallel fan-out | per_file strategy with repartir work-stealing | Snakemake |
| Multi-machine | SSH-based remote execution via targets | Nextflow |
| Hash-bucketed work dirs | Isolate parallel stage outputs | Nextflow |
| Event log | Append-only for crash recovery and audit | Temporal |
| Resource scheduling | Per-stage resource declarations | Snakemake |
| Frozen stages | Never re-executed (pinned results) | DVC |

### 11. CLI Commands

```bash
batuta playbook run pipeline.yaml       # Execute pipeline
batuta playbook run --stage extract     # Run single stage + deps
batuta playbook status                  # Show pipeline state
batuta playbook lock pipeline.yaml      # Generate/update lock file
batuta playbook validate pipeline.yaml  # Schema + DAG validation
batuta playbook visualize pipeline.yaml # ASCII DAG visualization
```

### 12. Design Principles

| Principle | Playbook Application |
|-----------|---------------------|
| **Genchi Genbutsu** | Hash-based validation -- trust checksums, not timestamps |
| **Jidoka** | `stop_on_first` failure by default |
| **Muda** | Skip unchanged stages (DVC-style invalidation) |
| **Heijunka** | Load-balanced parallel fan-out via repartir |
| **Poka-Yoke** | JSON Schema validation of YAML before execution |
| **Kaizen** | Lock files enable incremental pipeline improvement |

---

## Part III: Prior Art, Provable Contracts, and Falsification

> Inspired by CCX-RS (anton-abyzov/ccx-rs). References validated against GitHub and arXiv.

### 13. Prior Art

#### 13.1 GitHub Prior Art (Agent Runtimes)

| Project | Stars | Relevance to Agent Runtime |
|---------|-------|---------------------------|
| **openai/openai-agents-python** | 20.5K | Reference architecture for tool-use loops; handoff patterns between agents |
| **microsoft/agent-framework** | 8.4K | Multi-agent orchestration with .NET/Python; plugin system analogous to our Tool trait |
| **0xPlaygrounds/rig** | 6.7K | Rust-native LLM agent framework; proves Rust viable for agentic patterns |
| **liquidOS-ai/AutoAgents** | 529 | Multi-agent framework in Rust; async tool execution, conversation management |
| **anton-abyzov/ccx-rs** | 650 | Rust coding assistant with 19 tools, parallel execution, Landlock sandboxing, SSE streaming |
| **eugene1g/agent-safehouse** | 1.5K | OS-level sandboxing for AI agents; Landlock/Seatbelt policies validated in production |
| **block/goose** | 33.9K | Open-source AI agent with auto-compaction at 80% token limit; validates our threshold design |
| **Martian-Engineering/lossless-claw** | 3.9K | Lossless context management — compacts at 75% window while protecting recent messages |
| **anthropic-experimental/sandbox-runtime** | 3.6K | Official Anthropic sandboxing (Landlock+seccomp Linux, Seatbelt macOS); reference implementation for our OS sandbox |
| **cisco-ai-defense/defenseclaw** | 319 | Security governance for agentic AI — OS isolation + scanning + audit logging |
| **vstorm-co/summarization-pydantic-ai** | 17 | Context management via LLM summarization and sliding window; validates compaction thresholds |

**Key insights:**
- CCX-RS proves that a single Rust binary can implement the full agent pattern (19 tools, parallel execution, SSE) in <5MB
- Agent-safehouse validates that Landlock is production-ready for agent sandboxing
- OpenAI Agents SDK's "handoff" pattern could extend our agent for multi-agent collaboration (future work)

#### 13.2 arXiv References (Agent Architecture)

| Paper | ID | Year | Relevance |
|-------|-----|------|-----------|
| **Toolformer: LMs Can Teach Themselves to Use Tools** | arXiv:2302.04761 | 2023 | Self-supervised tool-use learning; validates that tool dispatch decisions can be learned, not just hard-coded |
| **What Are Tools Anyway? A Survey from the LLM Perspective** | arXiv:2403.15452 | 2024 | Comprehensive taxonomy of tool types and invocation patterns; informs our Tool trait design |
| **ReliabilityBench: LLM Agent Reliability Under Stress** | arXiv:2601.06112 | 2026 | Chaos engineering for agents — fault injection (timeouts, rate limits, partial responses) directly applicable to our LoopGuard and ToolHook testing |
| **Automated Hypothesis Validation with Agentic Sequential Falsifications** | arXiv:2502.09858 | 2025 | Popper-inspired agents designing falsification experiments; methodology adopted for section 16 below |
| **ACON: Optimizing Context Compression for Long-Horizon Agents** | arXiv:2510.00615 | 2025 | Dynamically condenses interaction histories; 26-54% token reduction while maintaining task performance — validates our two-phase compaction |
| **LongLLMLingua: LLMs in Long Context via Prompt Compression** | arXiv:2310.06839 | 2023 | Up to 20x prompt compression with minimal quality loss; prior art for our micro-compact phase |
| **Fault-Tolerant Sandboxing for AI Coding Agents** | arXiv:2512.12806 | 2025 | Policy-based interception + transactional filesystem snapshots; 100% interception of high-risk commands with only 14.5% overhead — validates our hook + sandbox architecture |
| **HAICOSYSTEM: Sandboxing Safety Risks in Human-AI Interactions** | arXiv:2409.16427 | 2024 | State-of-the-art LLMs exhibit safety risks in >50% of tool-equipped scenarios — empirical justification for our sandbox-by-default design |
| **ByteRobust: Robust LLM Training Infrastructure** | arXiv:2509.16293 | 2025 | Every-step checkpointing with <0.9% overhead; validates our session persistence (JSONL append-only) approach |
| **Rethinking Multi-agent Reliability: Byzantine Fault Tolerance** | arXiv:2511.10400 | 2025 | CP-WBFT consensus for LLM agents; future extension for multi-agent batuta playbooks |

---

### 14. Provable Contracts

#### 14.1 Contract: Agent Loop (`agent-loop-v1.yaml`)

```yaml
metadata:
  version: 1.0.0
  created: '2026-04-02'
  author: PAIML Engineering
  description: Agent perceive-reason-act loop correctness — termination, safety, state machine
  references:
  - 'CCX-RS agent loop: anton-abyzov/ccx-rs'
  - 'ReliabilityBench: arXiv:2601.06112'
  - 'Popper Falsification: arXiv:2502.09858'
  depends_on:
  - backend-dispatch-v1
  - streaming-tpot-v1

equations:
  loop_termination:
    formula: |
      iterations(agent_run) <= max_iterations
      AND cost(agent_run) <= cost_budget
      AND NOT ping_pong_detected(last_N_turns)
    domain: max_iterations ∈ ℤ⁺, cost_budget ∈ ℝ⁺, N = 4
    invariants:
    - Agent loop always terminates (no infinite loops)
    - At least one of three guards triggers termination
    - LoopGuard fires BEFORE budget is exceeded, not after
    preconditions:
    - max_iterations > 0
    - cost_budget > 0.0
    postconditions:
    - result.iterations <= max_iterations
    - result.cost <= cost_budget
    lean_theorem: Theorems.Loop_Termination

  state_machine:
    formula: |
      States: {Idle, Perceive, Reason, Act, Remember, Done, Failed}
      Transitions:
        Idle      → Perceive  (on: user_message)
        Perceive  → Reason    (on: memory_recalled)
        Reason    → Act       (on: tool_use response)
        Reason    → Remember  (on: end_turn response)
        Act       → Reason    (on: tool_result)
        Remember  → Done      (on: success)
        *         → Failed    (on: guard_triggered)
    domain: Finite state machine with 7 states
    invariants:
    - No state reached without valid transition
    - Failed is absorbing (no transitions out except reset)
    - Act always returns to Reason (tool results always processed)
    lean_theorem: Theorems.Agent_State_Machine

  context_compaction:
    formula: |
      token_count(messages) <= context_window × auto_threshold
      OR compact(messages) applied before next LLM call
      WHERE compact preserves:
        - system_prompt (always)
        - last_N_turns (configurable)
        - memory_recalls (re-fetched)
    domain: token_count ∈ ℤ⁺, thresholds ∈ (0, 1)
    invariants:
    - LLM never called with token_count > context_window
    - System prompt never truncated
    - Compaction is idempotent (compact(compact(m)) == compact(m))
    preconditions:
    - context_window > 0
    - auto_threshold > micro_threshold
    - micro_threshold > 0.0
    lean_theorem: Theorems.Context_Compaction

  sandbox_enforcement:
    formula: |
      allowed(tool_call) = capability(tool) ∈ manifest.capabilities
        AND path(tool_call) ∈ sandbox.allowed_paths(tier)
        AND network(tool_call) ∈ sandbox.allowed_network(tier)
    domain: Sandbox policies per privacy tier
    invariants:
    - Sovereign sandbox allows NO network egress
    - File writes restricted to project directory
    - Shell commands restricted to allowlisted binaries
    - Sandbox enforced at OS kernel level (Landlock/Seatbelt), not just application level
    preconditions:
    - manifest.capabilities.len() > 0
    - sandbox.write_paths.len() > 0
    lean_theorem: Theorems.Sandbox_Enforcement

  parallel_tool_safety:
    formula: |
      parallel_safe(calls) = ∀(c1, c2) ∈ calls:
        resources(c1) ∩ resources(c2) = ∅
        OR one_of(c1, c2) is read_only
    domain: Set of concurrent tool calls
    invariants:
    - Write-write conflicts always serialized
    - Read-write conflicts serialized (read first)
    - Read-read always parallelized
    lean_theorem: Theorems.Parallel_Tool_Safety

  session_crash_recovery:
    formula: |
      resume(session) = load(messages.jsonl) |> truncate_to_last_complete |> compact_if_needed
    domain: JSONL append-only log
    invariants:
    - Partial writes (crash mid-message) truncated, not corrupted
    - Resumed session produces identical behavior to uninterrupted session (up to LLM nondeterminism)
    - No message appears twice after resume
    lean_theorem: Theorems.Crash_Recovery

  hook_ordering:
    formula: |
      execution_order(tool_call) =
        pre_hooks(tool) → capability_check → tool.execute() → post_hooks(tool)
      WHERE pre_hook returning Block skips all subsequent steps
    domain: Ordered hook chain per tool call
    invariants:
    - Pre-hooks run before capability check (can block before permission eval)
    - Post-hooks run even if tool returns error (for audit logging)
    - Hook ordering is deterministic (registration order)
    lean_theorem: Theorems.Hook_Ordering

proof_obligations:
- type: termination
  property: Agent loop always terminates
  formal: iterations <= max_iterations for all executions
  applies_to: all
- type: state_machine
  property: Valid state transitions only
  formal: No state reached without valid transition edge
  applies_to: all
- type: invariant
  property: Context window never exceeded
  formal: token_count(messages) <= context_window at every LLM call
  applies_to: all
- type: invariant
  property: Sandbox blocks unauthorized access
  formal: Sovereign tier produces zero network syscalls
  applies_to: all
- type: invariant
  property: Parallel tools are conflict-free
  formal: No concurrent write-write on same resource
  applies_to: all
- type: idempotency
  property: Compaction is idempotent
  formal: compact(compact(messages)) == compact(messages)
  applies_to: all
- type: frame
  property: Message history append-only
  formal: messages[0..n] unchanged after appending messages[n+1]
  applies_to: all
- type: ordering
  property: Hook execution order
  formal: pre_hooks before execute before post_hooks for every tool call
  applies_to: all

falsification_tests:
- id: FALSIFY-AL-001
  rule: Loop termination
  prediction: Agent terminates within max_iterations even with adversarial LLM
  test: Mock LLM to always return tool_use (infinite loop attempt); assert LoopGuard fires at max_iterations
  if_fails: LoopGuard bypassed — infinite agent loop possible
- id: FALSIFY-AL-002
  rule: State machine transitions
  prediction: No invalid state transition possible through public API
  test: proptest with random sequences of perceive/reason/act calls; assert all transitions valid
  if_fails: State machine has unreachable or invalid transitions
- id: FALSIFY-AL-003
  rule: Context compaction safety
  prediction: System prompt survives compaction intact
  test: Fill context to 95% capacity, trigger auto-compact, assert system prompt byte-identical
  if_fails: Compaction corrupts system prompt — agent loses its instructions
- id: FALSIFY-AL-004
  rule: Sandbox enforcement
  prediction: Shell tool cannot write outside project directory under Sovereign tier
  test: Execute shell tool with `echo test > /tmp/escape.txt` under Sovereign sandbox; assert EACCES
  if_fails: Sandbox escape — agent can write arbitrary files
- id: FALSIFY-AL-005
  rule: Parallel tool conflict detection
  prediction: Two writes to same file are never executed concurrently
  test: Submit 10 write_file calls to same path simultaneously; assert sequential execution (no interleaving)
  if_fails: Data race — concurrent writes corrupt file
- id: FALSIFY-AL-006
  rule: Session crash recovery
  prediction: Truncated JSONL (simulated crash) resumes without duplicate messages
  test: Write session with 100 messages, truncate mid-write on message 73, resume, assert 72 messages loaded
  if_fails: Crash recovery replays or loses messages
- id: FALSIFY-AL-007
  rule: Hook blocking
  prediction: Pre-hook returning Block prevents tool execution
  test: Register pre-hook that blocks all shell commands containing "rm -rf"; attempt rm -rf; assert tool never executed
  if_fails: Hook bypass — destructive commands execute despite pre-hook
- id: FALSIFY-AL-008
  rule: Ping-pong detection
  prediction: LoopGuard detects repeated identical tool calls within 4 turns
  test: Mock LLM to emit same file_read call 5 times; assert LoopGuard fires on 5th
  if_fails: Agent wastes tokens on futile repeated actions
- id: FALSIFY-AL-009
  rule: Memory re-fetch after compaction
  prediction: Compaction triggers fresh memory recall (not stale cached memories)
  test: Write memory, compact, update memory, observe next recall returns updated version
  if_fails: Stale memories persist through compaction — agent acts on outdated context

kani_harnesses:
- id: KANI-AL-001
  obligation: Agent loop always terminates
  property: Loop terminates within max_iterations
  bound: 16
  strategy: bounded_int
  solver: cadical
  harness: verify_loop_termination
- id: KANI-AL-002
  obligation: Valid state transitions only
  property: State machine transition function total
  bound: 8
  strategy: exhaustive
  solver: cadical
  harness: verify_state_transitions
- id: KANI-AL-003
  obligation: Context window never exceeded
  property: Compaction triggers before overflow
  bound: 8
  strategy: stub_float
  solver: cadical
  harness: verify_context_bound
- id: KANI-AL-004
  obligation: Parallel tools conflict-free
  property: No concurrent writes to same resource
  bound: 4
  strategy: exhaustive
  solver: cadical
  harness: verify_parallel_safety
- id: KANI-AL-005
  obligation: Compaction idempotent
  property: compact(compact(m)) == compact(m)
  bound: 4
  strategy: bounded_int
  solver: cadical
  harness: verify_compaction_idempotent
- id: KANI-AL-006
  obligation: Hook execution order
  property: pre < execute < post for all tool calls
  bound: 8
  strategy: exhaustive
  solver: cadical
  harness: verify_hook_ordering

qa_gate:
  id: F-AL-001
  name: Agent Loop Contract
  description: Agent runtime correctness — termination, state machine, safety, persistence
  checks:
  - loop_termination
  - state_machine_transitions
  - context_compaction_safety
  - sandbox_enforcement
  - parallel_tool_safety
  - crash_recovery
  - hook_blocking
  - ping_pong_detection
  - memory_refetch
  pass_criteria: All 9 falsification tests pass + 6 Kani harnesses verify
  falsification: Adversarial LLM mock that tries to escape loop, sandbox, and context bounds
```

#### 14.2 Contract Location

Save to `../provable-contracts/contracts/batuta/agent-loop-v1.yaml` and register in `../provable-contracts/contracts/batuta/binding.yaml`.

---

### 15. UX Contracts

User-facing behaviors for the agent CLI, expressed as provable contracts and enforced via presentar-terminal Brick architecture + probar falsification. See [presentar-probar-integration.md](presentar-probar-integration.md) for full TUI and testing details.

#### 15.1 UX Invariants

| UX Property | Contract Type | Brick Assertion | Probar Test |
|-------------|--------------|-----------------|-------------|
| **Streaming responsiveness** | `bound` | `MaxLatencyMs(100)` per token | `FalsificationGate` H-STREAM-001 (100 trials, <1% rejection) |
| **Cost transparency** | `postcondition` | `TextVisible` on CostDashboardPanel | Pixel coverage: cost panel region always covered |
| **Session resume** | `roundtrip` | Visual regression: SSIM >0.99 pre/post resume | Probar snapshot diff baseline |
| **Sandbox feedback** | `postcondition` | `ContrastRatio(7.0)` (AAA for warnings) | Pixel coverage: sandbox panel lit on blocked action |
| **Progress visibility** | `invariant` | `Custom("progress_monotonic")` | `FalsificationGate` H-TOOL-001 (per-frame) |
| **Graceful degradation** | `postcondition` | `TextVisible` on provider status | Probar playbook: provider-failover.yaml transition check |
| **History navigation** | `frame` | `TextVisible` on SessionPanel | Probar TUI test: `batuta agent sessions` renders all entries |
| **Frame budget** | `bound` | BrickHouse 16ms total | BrickHouse budget report: utilization <= 100% |
| **Accessibility** | `invariant` | `ContrastRatio(4.5)` all panels | probar `A11yChecker` WCAG AA |
| **Layout correctness** | `invariant` | No pixel overlap at any terminal size | Pixel coverage: 20x10 to 200x60 sweep |

#### 15.2 Brick-Based UX Enforcement

Each agent TUI panel implements the presentar-core `Brick` trait. Jidoka enforcement: if `can_render()` returns false, the panel is not drawn. This prevents rendering invalid states (e.g., negative cost, progress >100%, context usage >window).

```rust
// Every panel is a Brick with verifiable assertions
impl Brick for CostDashboardBrick {
    fn assertions(&self) -> &[BrickAssertion] {
        &[
            BrickAssertion::TextVisible,
            BrickAssertion::ContrastRatio(4.5),
            BrickAssertion::MaxLatencyMs(50),
            BrickAssertion::Custom("cost_non_negative"),
        ]
    }

    fn verify(&self) -> BrickVerification {
        let mut v = BrickVerification::new();
        if self.cost < 0.0 {
            v.fail("Cost is negative — estimation error");
        }
        if self.budget_used > 1.0 {
            v.fail("Budget utilization >100% — display overflow");
        }
        v
    }

    fn can_render(&self) -> bool { self.verify().passed() }
}
```

All 6 agent panels compose into a `BrickHouse` with a 16ms frame budget (60fps). See [presentar-probar-integration.md](presentar-probar-integration.md) section 6 for full BrickHouse composition.

#### 15.3 UX Contract YAML Fragment

```yaml
equations:
  ux_streaming_responsiveness:
    formula: |
      ttft_displayed = t(first_char_on_terminal) - t(user_pressed_enter)
      ttft_displayed <= 2.0s when streaming enabled
    invariants:
    - First character appears within 2s of user input
    - If provider does not support streaming, full response appears within latency SLA
    postconditions:
    - terminal_output.len() > 0 within 2s
    brick_assertion: MaxLatencyMs(100)
    probar_gate: H-STREAM-001

  ux_cost_transparency:
    formula: |
      for each turn t in session:
        display(cost_estimate(t)) before display(next_prompt)
    invariants:
    - Cost is displayed, never hidden
    - Cumulative cost shown alongside per-turn cost
    postconditions:
    - output.contains("$") OR output.contains("cost")
    brick_assertion: TextVisible
    probar_gate: pixel_coverage(cost_panel) >= 0.80

  ux_sandbox_feedback:
    formula: |
      blocked_tool_call => display(reason) AND display(policy)
    invariants:
    - User is never left wondering why a tool call failed silently
    - Blocked reason includes the specific policy rule that triggered
    postconditions:
    - output.contains("blocked") AND output.contains("policy")
    brick_assertion: ContrastRatio(7.0)
    probar_gate: pixel_coverage(sandbox_panel) == 1.0 on block event

  ux_frame_budget:
    formula: |
      frame_time(brick_house) <= 16ms for 60fps
    invariants:
    - All 6 panels render within budget
    - BrickHouse Jidoka fires if any Brick exceeds allocation
    brick_assertion: BrickHouse.budget_ms(16)
    probar_gate: H-BUDGET-001 (10K frames, <1% violation)
```

---

### 16. Popperian Falsification Report

#### 16.1 Falsifiable Claims

| Claim | Falsification Test | Probar Implementation | What Failure Means |
|-------|-------------------|----------------------|-------------------|
| **Agent loop always terminates** | FALSIFY-AL-001 | `probador playbook agent-loop.yaml` (M4 mutation) | Infinite loop risk |
| **State machine has no dead states** | FALSIFY-AL-002 | `probador playbook agent-loop.yaml --validate` | Agent gets stuck |
| **Compaction preserves system prompt** | FALSIFY-AL-003 | Pixel coverage: system prompt panel unchanged after compact | Agent loses instructions |
| **OS sandbox prevents escape** | FALSIFY-AL-004 | `FalsificationGate` with shell escape attempts | Security vulnerability |
| **Parallel tools are race-free** | FALSIFY-AL-005 | Probar deterministic replay with concurrent tool calls | Data corruption |
| **Crash recovery is lossless** | FALSIFY-AL-006 | Probar snapshot + truncate + resume + visual regression | User loses work |
| **Hooks can block destructive actions** | FALSIFY-AL-007 | `FalsificationGate` with destructive command patterns | Safety bypassed |
| **Ping-pong detection works** | FALSIFY-AL-008 | Playbook complexity check: O(n) not O(n^2) | Wasted tokens |
| **Memory refreshes after compaction** | FALSIFY-AL-009 | Probar state capture before/after compact | Stale context |
| **Frame budget met at 60fps** | FALSIFY-AL-010 (new) | BrickHouse budget report over 10K frames | TUI too slow |
| **WCAG AA contrast on all panels** | FALSIFY-AL-011 (new) | probar `A11yChecker` + Brick `ContrastRatio(4.5)` | Accessibility violation |
| **Pixel coverage >= 80%** | FALSIFY-AL-012 (new) | `PixelCoverageTracker` across all TUI states | Untested UI regions |

#### 16.2 Pre-Registered Null Hypotheses

Following arXiv:2502.09858 (Agentic Sequential Falsification methodology):

| H₀ (Null) | Test to Reject H₀ | Required Evidence |
|-----------|-------------------|-------------------|
| Context compaction is lossy (drops non-system content unpredictably) | Compact 1000 random conversation histories; verify system prompt + last N turns byte-identical | 100% preservation rate across 1000 trials |
| Landlock sandbox has bypass on kernel < 6.2 | Test FALSIFY-AL-004 on kernel 5.13, 6.1, 6.2, 6.8 | Zero escapes on all tested kernels |
| Parallel tool execution introduces non-determinism | Run same 5-tool-call batch 100 times; verify identical results (order may vary, content must match) | 100% result parity |
| Session resume produces different behavior from uninterrupted | Fork session at turn 50, resume, compare next 10 turns with uninterrupted (temperature=0) | Byte-identical output for deterministic models |

#### 16.3 What Would Disprove This Specification

If ANY of these are true, the agent runtime architecture is fundamentally wrong:

1. **Compaction loses critical context more than 1% of the time.** If even 1% of compacted conversations lose information that changes the agent's behavior, compaction is too aggressive. (Check: FALSIFY-AL-003 + property test with 10K conversations)

2. **Landlock/Seatbelt cannot enforce file-path granularity for all tools.** If any tool invokes a subprocess that inherits a less-restricted sandbox, the sandbox is theater. (Check: renacer syscall trace of sandboxed agent; verify zero EACCES bypasses)

3. **Parallel tool execution provides <10% wall-clock speedup in practice.** If most LLM turns emit 1-2 tool calls (not 5+), the dependency detection overhead exceeds the parallelism benefit. (Check: measure tool call distribution from 1000 real agent sessions)

4. **Session persistence adds >5% overhead to each turn.** If JSONL append + fsync costs more than 5% of turn latency, the persistence layer is too expensive for interactive use. (Check: benchmark turn latency with and without session persistence)

5. **Hook system creates a false sense of safety.** If pre-hooks can be bypassed by tool calls that construct shell commands dynamically (e.g., `echo rm | sh`), the hook pattern is security theater. (Check: FALSIFY-AL-007 with adversarial command construction)
