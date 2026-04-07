# apr code Feasibility Falsification

> Methodology: Popperian falsification — attempt to BREAK the claim "apr code can work as a Claude Code clone"
> Verdict: **FEASIBLE. Code-verified against actual batuta sources 2026-04-02.**
> Previous draft had 5 "critical gaps" — re-verification found 3 were WRONG (code already exists). Revised to 2 real gaps.

---

## 1. Dependency Chain: CONFIRMED VIABLE

```
apr-cli v0.4.11  (aprender workspace)
  ├─ aprender v0.27.5      (direct, path dep)
  ├─ batuta v0.7.3          (proposed, optional feature "code")
  │   ├─ aprender v0.27     (OPTIONAL, behind "ml" feature — NOT ENABLED)
  │   ├─ presentar-terminal (optional, behind "agent-tui")
  │   └─ ... other optional deps
  └─ batuta-common v0.1     (already present)
```

**No circular dependency.** Batuta's `aprender` dep is optional and gated behind the `ml` feature. `apr-cli` would import `batuta` WITHOUT enabling `ml`. Cargo resolves this correctly.

**Verified in code:**
- `batuta/Cargo.toml:123` — `aprender = { version = "0.27", optional = true }`
- `batuta/Cargo.toml` default features: `["native", "rag"]` — NO `ml`
- `batuta/src/lib.rs:99` — `pub mod agent;` is PUBLIC
- `batuta/src/agent/mod.rs` — exports `AgentBuilder`, `LlmDriver`, `MemorySubstrate`, `ToolRegistry`, `StreamEvent`
- `batuta/src/cli/agent.rs:67` — `Arc<dyn batuta::agent::driver::LlmDriver>` is how drivers are used

**Binary size concern:** Adding batuta adds ~5-15MB to apr binary (estimated from batuta's dep tree). Manageable with feature gating.

---

## 2. Agent Runtime: EXISTS AND IS FUNCTIONAL

**What exists today (1,785 lines across 6 agent CLI files):**

| Component | Status | Lines | Notes |
|-----------|--------|-------|-------|
| `AgentBuilder` | Implemented | agent/mod.rs | Builder pattern with driver, tools, memory, stream |
| `LlmDriver` trait | Implemented | agent/driver/ | `complete()` + `stream()` + `context_window()` + `privacy_tier()` |
| `RealizarDriver` | Implemented | agent/driver/ | Local GGUF/APR inference via realizar |
| `MockDriver` | Implemented | agent/driver/ | Deterministic testing |
| `ToolRegistry` | Implemented | agent/tool/ | Register + execute tools by name |
| `MemorySubstrate` | Implemented | agent/memory/ | `remember()` + `recall()` + `forget()`, in-memory + trueno-db |
| `LoopGuard` | Implemented | agent/guard/ | Max iterations, ping-pong detection |
| `AgentManifest` | Implemented | agent/manifest/ | TOML-based agent config |
| `AgentDashboard` | Implemented | agent/tui_render.rs | Basic TUI rendering |
| Streaming | Implemented | agent_runtime_cmds.rs | `mpsc::channel<StreamEvent>`, token-by-token |
| Ed25519 signing | Implemented | agent/signing/ | Manifest integrity verification |
| MCP tools | Implemented | agents-mcp feature | Model Context Protocol integration |
| Pool (multi-agent) | Implemented | agent/pool/ | Fan-out multiple agents |

**Code-level verification (2026-04-02) corrects the previous draft's gap analysis.**

---

## 3. WHAT ALREADY EXISTS (Previous Draft Was WRONG About These)

### CORRECTED — GAP-2 was FALSE: Shell Tool EXISTS

**Previous claim:** "No File/Shell Tools in Agent ToolRegistry"
**Actual code:** `src/agent/tool/shell.rs` — `ShellTool` with capability-based allowlisting, working directory restriction, output truncation (8KB max), timeout enforcement. **80+ lines, fully implemented.**

Additionally found in `src/agent/tool/`:
- `shell.rs` — `ShellTool` (sandboxed shell commands)
- `rag.rs` — `RagTool` (trueno-rag semantic search)
- `memory.rs` — `MemoryTool` (remember/recall)
- `inference.rs` — `InferenceTool` (call sub-model)
- `spawn.rs` — `SpawnTool` (spawn sub-agent)
- `network.rs` — `NetworkTool` (HTTP requests)
- `browser.rs` — `BrowserTool` (headless browser)
- `compute.rs` — `ComputeTool` (tensor operations)
- `mcp_client.rs` — `McpClientTool` (external MCP tool servers)

**9 tools exist, not "3-4".** Shell is present. Still missing: `file_read`, `file_write`, `file_edit`, `glob`, `grep`. These are ~500 lines total (not 1000).

### CORRECTED — GAP-4 was FALSE: Context Window Management EXISTS

**Previous claim:** "No context window management. No token counting, no compaction, no truncation."
**Actual code:** `src/agent/runtime.rs:199-217` — `build_context()` creates a `ContextManager` with `TokenEstimator`, computes effective window by subtracting system prompt + tool definition tokens, and uses `TruncationStrategy::SlidingWindow`.

`runtime.rs:346-378` — `truncate_messages()` calls `context.fits()` and `context.truncate()` from `crate::serve::context`. This is a working sliding-window truncation system.

**Token estimation, context tracking, and message truncation are all implemented.** What's missing is the two-phase compaction (micro-compact + auto-compact) — current code only does sliding window truncation (drop oldest messages). This is functional but less sophisticated than the spec.

### CORRECTED — GAP-3 PARTIALLY FALSE: RemoteDriver EXISTS (but apr code doesn't use it)

**Previous claim:** "realizar Tool-Use Support Unknown"
**Actual code:** `src/agent/driver/remote.rs` — `RemoteDriver` with `ApiProvider::Anthropic` and `ApiProvider::OpenAi` enum variants. SSE streaming in `remote_stream.rs`. `RoutingDriver` in `router.rs` with failover cascade.

**DESIGN DECISION (PMAT-111):** `apr code` is Sovereign-only and uses ONLY `RealizarDriver` for local GGUF/APR inference. The RemoteDriver and RoutingDriver exist in the batuta agent runtime for other consumers (e.g., `batuta agent run` with custom manifests) but are NOT wired into `apr code`. The remaining question is whether local models can do tool-use — a model capability question, not a code gap.

---

## 4. ACTUAL REMAINING GAPS (2 real, not 5)

### GAP-A: Interactive REPL — PHASE 1 IMPLEMENTED

**UPDATE 2026-04-02:** Phase 1 REPL implemented (PMAT-105). Line-by-line input with streaming output, slash commands (/help, /quit, /cost, /context, /clear), Ctrl+C cancel, session tracking, budget enforcement. Phase 2 will add crossterm raw-mode for true concurrent I/O.

**Previous status: BLOCKING for Phase 1. Now CLOSED for Phase 1 MVP.**

**Dogfooding results (2026-04-03):**
- `batuta code --help` ✓ — outputs usage, options, examples
- `batuta code` ✓ — launches interactive REPL with welcome banner, slash commands
- 7 smoke tests pass (PMAT-107): file_read, file_write, file_edit, shell, glob, no_tools, tool_count
- Full pipeline verified: MockDriver → tool_use → tool executes → result fed back → final response
- **LIVE EXECUTION (2026-04-03):** `batuta code -p "Hello"` runs end-to-end:
  - Detects no model → falls back to MockDriver (dry-run)
  - Prints response and exits cleanly
  - Output: `Hello! I'm running in dry-run mode. Set model_path or remote_model in your agent manifest.`
  - Remaining: test with real GGUF model via RealizarDriver (PMAT-108)

`cmd_agent_chat()` exists as a blocking read→send→print loop. No concurrent input/output. No streaming display while typing. No slash commands. No Ctrl+C interrupt (kills process).

Claude Code's core UX IS the interactive REPL. Without it, `apr code` is just `batuta agent run --prompt "..."` — a single-shot tool, not an interactive assistant.

**What must be built:**
- Raw terminal mode (crossterm) for simultaneous input + streaming output
- Input area at bottom, scrolling output above
- Slash command parser (/help, /model, /cost, /compact, /quit)
- Ctrl+C → cancel generation (not exit process)
- Streaming token display from `mpsc::channel<StreamEvent>` (channel already exists)

**Estimated effort:** ~800 lines. This is the critical path item.

### GAP-B: Missing File Tools ~~(file_read, file_write, file_edit,~~ glob, grep)

**UPDATE 2026-04-02:** GAP-B **CLOSED.** All 5 core tools implemented:
- FileReadTool, FileWriteTool, FileEditTool (19 tests, PMAT-103)
- GlobTool, GrepTool (15 tests, PMAT-104)
- PmatQueryTool deferred — shell tool can invoke `pmat query` directly.

**Status: BLOCKING for Phase 1. ~500 lines of new tool code.**

Shell tool exists (can `cat`, `sed`, `find` via shell). But Claude Code has dedicated `file_read`, `file_write`, `file_edit` tools that are safer and faster than shelling out. Missing dedicated tools means:
- `file_read` via shell: `cat file` — works but no size limit, no line ranges
- `file_edit` via shell: `sed -i` — fragile, no undo, no validation
- `grep` via shell: `grep -r` — works but no pmat quality annotations

**What must be built:**
- `FileReadTool` — read file with line range, size limit (~80 lines)
- `FileWriteTool` — write/create file with backup (~80 lines)
- `FileEditTool` — string replacement with uniqueness check (~120 lines)
- `GlobTool` — file pattern matching (~60 lines)
- `GrepTool` — content search with context lines (~80 lines)
- `PmatQueryTool` — quality-annotated search via pmat (~80 lines)

**Estimated effort:** ~500 lines total.

---

## 5. HONEST REVISED ASSESSMENT

### What actually needs to be built for Phase 1 MVP

| Component | Lines | Status |
|-----------|-------|--------|
| Interactive REPL (crossterm + streaming) | ~800 | **NEW** |
| 6 file/search tools | ~500 | **NEW** |
| `Code` subcommand in apr-cli | ~200 | **NEW** |
| **Total new code** | **~1,500** | |

### What already exists and is reusable

| Component | Lines | Status |
|-----------|-------|--------|
| Agent loop (perceive-reason-act) | runtime.rs (440) | **DONE** |
| LlmDriver trait + 4 impls | driver/ (~1000) | **DONE** |
| RealizarDriver (local inference) | realizar.rs | **DONE** |
| RemoteDriver (Anthropic + OpenAI) | remote.rs + remote_stream.rs | **DONE** |
| RoutingDriver (failover cascade) | router.rs | **DONE** |
| ToolRegistry + 9 tools | tool/ (~1500) | **DONE** |
| MemorySubstrate (in-memory + trueno-db) | memory/ | **DONE** |
| LoopGuard (max iter, ping-pong, cost) | guard/ | **DONE** |
| AgentManifest (TOML config) | manifest/ | **DONE** |
| Context window management | runtime.rs + serve/context | **DONE** |
| Token estimation | serve/context/TokenEstimator | **DONE** |
| Streaming events channel | StreamEvent + mpsc | **DONE** |
| Retry with exponential backoff | runtime.rs:381-407 | **DONE** |
| Capability-based tool access control | capability.rs | **DONE** |
| Privacy tier enforcement (Sovereign blocks network) | runtime.rs:243-249 | **DONE** |
| MCP tool integration | tool/mcp_client.rs | **DONE** |
| Ed25519 manifest signing | signing/ | **DONE** |
| Contract verification | contracts.rs + agent-loop-v1.yaml | **DONE** |
| AgentBuilder (library API) | mod.rs:68-124 | **DONE** |
| Tests (unit + integration) | 6 test files (~1500 lines) | **DONE** |
| **Total existing code** | **~5,000+** | |

### Ratio: 1,500 new lines to leverage 5,000+ existing lines. 77% reuse.

---

## 6. FALSIFIABLE CLAIMS — REVISED AND CODE-VERIFIED

| # | Claim | Verified Against | Verdict |
|---|-------|-----------------|---------|
| 1 | `apr-cli` can depend on `batuta` without circular dependency | `batuta/Cargo.toml:123` — aprender is optional behind `ml` feature | **CONFIRMED** |
| 2 | `batuta::agent` module is public and callable as library | `src/lib.rs:99`, `src/agent/mod.rs:68` — `AgentBuilder::new().run()` | **CONFIRMED** |
| 3 | Agent loop handles tool_use → execute → feed result back | `src/agent/runtime.rs:101-118` — full tool call dispatch with capability check | **CONFIRMED** |
| 4 | RealizarDriver provides local inference | `src/agent/driver/realizar.rs` — implements `LlmDriver` trait | **CONFIRMED** |
| 5 | RemoteDriver supports Anthropic + OpenAI with SSE | `src/agent/driver/remote.rs:24-29` — `ApiProvider::Anthropic` + `ApiProvider::OpenAi` | **CONFIRMED** |
| 6 | RoutingDriver does failover cascade | `src/agent/driver/router.rs` — exists with router tests | **CONFIRMED** |
| 7 | Context window management exists | `src/agent/runtime.rs:199-217` — `ContextManager` + `TokenEstimator` + `SlidingWindow` | **CONFIRMED** |
| 8 | Shell tool is sandboxed | `src/agent/tool/shell.rs:30-37` — allowlist + working_dir + timeout | **CONFIRMED** |
| 9 | Privacy tier blocks network under Sovereign | `src/agent/runtime.rs:243-249` — explicit sovereign network block | **CONFIRMED** |
| 10 | Streaming events via channel | `src/cli/agent_runtime_cmds.rs:80` + `AgentBuilder::stream()` | **CONFIRMED** |
| 11 | Interactive REPL exists | `src/cli/agent_runtime_cmds.rs` — blocking loop only, no concurrent I/O | **FALSE — MUST BUILD** |
| 12 | File read/write/edit tools exist | `src/agent/tool/` — shell exists, no dedicated file tools | **FALSE — MUST BUILD** |

---

## 7. WHAT WOULD DISPROVE "apr code works" — HARD FALSIFICATION

### F-1: Does `AgentBuilder::new(&manifest).driver(&driver).tool(tool).run("query").await` actually execute tools?

**Test:** Already tested in `src/agent/mod.rs:192-232` — `test_builder_with_tool()` registers a `DummyTool`, runs the builder, asserts success. **But:** MockDriver returns `StopReason::EndTurn` immediately — it never returns `StopReason::ToolUse`. The test proves the builder works but NOT that the tool loop actually executes tools.

**Real falsification test needed:** Use `MockDriver::with_tool_calls(vec![ToolCall { name: "dummy", ... }])` — if this constructor exists, test that the tool actually executes. If it doesn't exist, this is a testing gap.

### F-2: Does `RealizarDriver` actually produce parseable `tool_use` responses for Qwen2.5-Coder?

**Cannot verify from code alone.** This requires a runtime test with a real model. The spec claims Sovereign tier works with local models, but if Qwen2.5-Coder 7B cannot generate valid `{"type": "tool_use", "name": "file_read", "input": {"path": "src/main.rs"}}` JSON, the entire Sovereign promise is empty.

**Required:** Integration test: `apr code --offline --model qwen2.5-coder-7b -p "Read src/main.rs and tell me what it does"` — must produce a valid tool call, not hallucinated output.

### F-3: Does the sliding window truncation actually preserve recent messages?

**Verified in code:** `runtime.rs:346-378` iterates from end of original list to reconstruct truncated messages in order. **But:** the matching logic uses `content == chat_msg.content` which could mis-match if two messages have identical content (e.g., two "OK" responses). This is a latent bug — not blocking but should be flagged.

### F-4: Can `apr-cli` binary size stay under 50MB with batuta dependency?

**Cannot verify without building.** Batuta's dep tree includes trueno, trueno-rag, trueno-db, crossterm, serde, tokio. Estimated: 25-40MB with `--release` and LTO. Needs measurement.

**Required:** `cargo build -p apr-cli --features code --release && ls -la target/release/apr`

### F-5: Does the REPL need to be built from scratch or can `batuta agent chat` be adapted?

**Code shows:** `cmd_agent_chat()` uses a simple `loop { read_line → agent_run → print }` pattern. This CANNOT be adapted to concurrent input/output — it must be replaced with an event loop (crossterm + tokio). The streaming channel (`mpsc::Sender<StreamEvent>`) already exists, so the receiver side just needs a TUI renderer.

---

## 8. FINAL VERDICT

**`apr code` is confirmed viable.** The claim was previously "5 critical gaps"; code verification reduces this to **2 gaps totaling ~1,300 lines of new code**, leveraging ~5,000+ lines of existing agent infrastructure.

| Verdict | Evidence |
|---------|----------|
| **Dependency chain** | No circular dep. Cargo feature gates resolve it. |
| **Agent runtime** | Full perceive-reason-act loop with tools, guards, memory, streaming. |
| **Local inference** | RealizarDriver + RemoteDriver + RoutingDriver all implemented. |
| **Context management** | TokenEstimator + SlidingWindow truncation implemented. |
| **Tool execution** | 9 tools including ShellTool with sandbox. Missing file-specific tools (~500 lines). |
| **Interactive REPL** | NOT implemented. This is the main work item (~800 lines). |
| **Privacy enforcement** | Sovereign tier blocks network in runtime.rs. MCP privacy validation implemented. |

The honest framing: **apr code Phase 1 is ~1,500 lines of new code on top of a 5,000+ line foundation that already works.**

---

## 9. PHASE 1 REVISED CRITICAL PATH

```
Week 1: 6 file/search tools (file_read, file_write, file_edit, glob, grep, pmat_query)
         ~500 lines, register in ToolRegistry alongside existing ShellTool

Week 2: Interactive REPL with streaming
         crossterm raw-mode event loop + presentar-terminal rendering
         StreamEvent receiver renders tokens above, input below
         Slash command parser (/help, /model, /cost, /quit)
         Ctrl+C cancels generation via tokio::CancellationToken
         ~800 lines

Week 3: apr-cli integration
         Code subcommand in commands_enum.rs (~50 lines)
         Dispatch to batuta::agent::AgentBuilder (~100 lines)
         Feature flag "code" = ["dep:batuta"] (~10 lines)
         Default AgentManifest for coding assistant (~40 lines)
         ~200 lines
```

**Total Phase 1: ~1,500 lines of new code** leveraging 5,000+ existing lines (77% reuse).

---

## 10. PROBAR-FIRST ENFORCEMENT FOR PHASE 1

| Phase 1 Component | Probar Test Required BEFORE Implementation |
|----|-----|
| `FileReadTool` | Unit: read existing returns content. Falsify: read nonexistent → error, not panic. Falsify: read `/etc/shadow` under Sovereign → blocked |
| `FileWriteTool` | Unit: write creates file. Falsify: write outside project dir → blocked by ShellTool's working_dir restriction |
| `FileEditTool` | Unit: unique string replaced. Falsify: ambiguous match → error with count. Falsify: edit preserves file encoding |
| `GrepTool` | Unit: regex match returns lines. Falsify: invalid regex → error, not panic |
| Interactive REPL | Probar TUI test: input at bottom, streaming above. Pixel coverage: both regions exercised. Falsify: Ctrl+C during streaming → generation stops, process lives |
| `Code` subcommand | Integration: `apr code -p "echo hello" --offline` → runs agent loop → produces output. Falsify: `--offline` with no local model → exit code 5, not crash |
