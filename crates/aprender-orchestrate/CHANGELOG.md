# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.3] - 2026-03-02

### Changed
- Updated benchmark appendix with real GPU measurements: Qwen 1.5B Q4K at 240 tok/s on RTX 4090 (GH-88)

### Stack Fixes (upstream)
- **realizar GH-88**: APR GPU inference for GQA models — `num_key_value_heads` metadata alias fix
- **realizar GH-89**: bench_forward hardcoded model path removed (SATD)
- **aprender GH-375**: GGUF Q4_0/Q8_0 import fallback to dequant-requant path
- **aprender GH-90**: Brick benchmarks now report analytical budgets, not synthetic timing

## [0.7.2] - 2026-03-01

### Changed
- **Drift redesign**: startup drift check now only reports batuta's own stale dependencies, not the full 45-crate ecosystem. `batuta stack drift` retains full ecosystem view for maintainers.
- Updated all stack dependencies: trueno 0.16, aprender 0.27, realizar 0.8, entrenar 0.7, renacer 0.10, bashrs 6.65, trueno-viz 0.2, trueno-graph 0.1.17, trueno-db 0.3.16

### Fixed
- `agent_signing` example now compiles with default features (was gated behind `native` instead of `agents`)

## [0.7.1] - 2026-03-01

### Fixed
- Drift checking no longer blocks commands by default (warn-once per hour)
- `batuta init` works in directories without `.git`
- `batuta analyze --tdg` now counts files correctly (was showing 0)
- CLI help text updated to current description
- README rewritten with current examples and versions

## [0.7.0] - 2026-03-01

### Added

#### Autonomous Agent Runtime (`--features agents`)
- **Perceive-Reason-Act loop** — core agent loop with configurable iteration and tool-call guards
- **TOML manifest format** — declarative agent configuration with capabilities, resources, model config
- **16 formal contract invariants** (INV-001 through INV-016) in `contracts/agent-loop-v1.yaml`
- **LoopGuard** — Jidoka circuit breaker: max iterations, tool calls, cost budget, token budget, max-tokens consecutive hit detection, ping-pong loop detection
- **Privacy tiers** — Sovereign/Private/Standard with Poka-Yoke enforcement at tool, network, and MCP transport levels
- **Context window management** — sliding-window truncation with token estimation and min-message preservation

#### Agent Drivers
- **LocalDriver** — realizar-backed local inference (GGUF/SafeTensors/APR)
- **RemoteDriver** — Anthropic and OpenAI API with SSE streaming
- **RoutingDriver** — failover chain across multiple drivers with circuit breaker

#### Agent Tools (capability-gated)
- **ShellTool** — sandboxed command execution with path/command allowlists
- **FileReadTool** / **FileWriteTool** — filesystem access with path prefix restrictions
- **NetworkTool** — HTTP requests with host allowlisting
- **BrowserTool** — headless browser automation via jugar-probar (`--features agents-browser`)
- **RagTool** — RAG search over indexed documentation (`--features agents-rag`)
- **InferenceTool** — sub-model invocation with configurable timeout
- **SpawnTool** — sub-agent delegation with depth-bounded recursion (max depth 3)
- **McpClientTool** — MCP protocol tool discovery and invocation (`--features agents-mcp`)

#### Agent Infrastructure
- **AgentPool** — fan-out multiple agents with configurable concurrency, join-all collection
- **InMemorySubstrate** — conversation memory with semantic recall
- **MessageRouter** — inter-agent message passing
- **Ed25519 manifest signing** — sign/verify via pacha (tamper detection)
- **Model auto-pull** — `apr pull` subprocess integration for missing models
- **Model validation gates** — G0 (integrity), G1 (format detection), G2 (inference sanity with entropy check)
- **TUI dashboard** — agent loop visualization via presentar-terminal (`--features tui`)
- **`--stream` flag** — real-time token-by-token streaming output for run and chat

#### Agent CLI Commands
- `batuta agent run` — single-prompt non-interactive execution
- `batuta agent chat` — interactive REPL session
- `batuta agent validate` — manifest validation with optional model checks
- `batuta agent sign` / `verify-sig` — Ed25519 manifest signing
- `batuta agent contracts` — display and verify contract invariant bindings
- `batuta agent status` — manifest configuration display
- `batuta agent pool` — multi-agent fan-out with result collection

#### Design-by-Contract
- **provable-contracts** integration — compile-time contract annotations on agent loop and tools
- **agent-loop-v1.yaml** — 16 invariants with 35 unit test bindings
- 7 runnable examples: `agent_demo`, `agent_contracts`, `agent_guard`, `agent_memory`, `agent_pool`, `agent_routing`, `agent_signing`

#### Quality Infrastructure
- **QA-002 compliance** — all source files ≤500 lines (refactored 12 modules)
- **5,641 tests** passing with 95%+ line coverage
- **TDG Score: 98.4 (A+)**
- **0 SATD**, 0 clippy warnings, 0 unwrap() in agent module, 0 dead code

### Changed
- Migrated TUI backend from ratatui to presentar-terminal
- Replaced petgraph with trueno-graph for dependency analysis
- Removed regex-lite and colored dependencies (dep reduction)
- Updated stack: trueno 0.15, aprender 0.26, realizar 0.7, entrenar 0.6, renacer 0.9.8

## [0.6.0] - 2025-02-15

### Added
- apr-qa integration for model quality assurance
- Stack drift detection with `batuta stack drift` and `--fix` command
- Automatic drift blocking on all commands when dependencies are stale
- Bug hunter module with fault pattern analysis
- Playbook executor for multi-stage automation workflows
- `batuta falsify` command for Popperian falsification testing

### Changed
- Updated trueno to 0.14, aprender to 0.24, realizar to 0.5
- Major refactoring for pmat compliance (file health A+)
- Extracted CLI modules: stack.rs, hf.rs, deploy.rs, pipeline_cmds.rs

## [0.5.0] - 2025-01-25

### Added
- **Local workspace intelligence** — multi-project development with oracle
- Oracle RAG mode with SQLite-backed indexed documentation search
- `batuta oracle --rag-index` and `batuta oracle --rag "query"` commands
- Ground truth corpus integration (HuggingFace, TGI, Databricks, Ludwig, tiny-model)
- Private repo support via `.batuta-private.toml`

## [0.4.0] - 2025-01-01

### Added
- Pepita tiny kernel integration
- repartir v2.0.0 distributed computing
- whisper-apr speech recognition integration
- trueno-zram/trueno-ublk compression and block device support
- HuggingFace ecosystem catalog with `batuta hf search`
- 34 cookbook recipes with TDD test companions
- Oracle natural language queries with `--format code` for code + test output

### Changed
- Updated to trueno 0.11, aprender 0.21, realizar 0.4
- Updated pacha to 0.2

## [0.3.0] - 2024-12-20

### Added
- Stack orchestration commands: `batuta stack check|status|versions|quality|gate`
- Release pipeline with preflight verification
- PMAT quality gates integration

## [0.2.0] - 2024-12-16

### Changed
- Updated full PAIML stack dependency versions
- Book documentation refresh

## [0.1.9] - 2025-12-15

### Changed
- Updated trueno from v0.8.4 to v0.8.5 (simulation testing framework)

## [0.1.4] - 2025-12-06

### Changed
- Updated pforge from v0.1.2 to v0.1.4 (pmcp 1.8.6 compatibility fix)
- Updated renacer from v0.6.5 to v0.7.0 in knowledge graph

## [0.1.2] - 2025-12-05

### Added

#### Content Creation Tooling (Spec v1.1.0)
- **Content Module** - LLM prompt generation for educational content
  - 5 content types: HLO, DLO, BCH, BLP, PDM
  - Token budgeting (Heijunka) for Claude/Gemini/GPT-4 context windows
  - Content validation (Jidoka) with quality gates

- **CLI Commands** for content creation
  - `batuta content types` - List all content types
  - `batuta content emit --type <TYPE>` - Generate structured prompts
  - `batuta content validate --type <TYPE> <file>` - Validate content

#### Visualization and Experiment Tracking
- Viz module with framework comparison tree
- Experiment tracking with Entrenar v1.8.0 integration

### Changed
- Refactored experiment module into multi-file structure
- Updated clippy compliance for acronym conventions

## [0.1.1] - 2025-11-28

### Added
- Model Serving Ecosystem integration (native feature flag)
- Data Platforms Integration module

## [0.1.0] - 2025-01-21

### Added

#### Core Features
- **Analysis Phase** - Project analysis with language detection, dependency analysis, TDG scoring
- **NumPy → Trueno Converter** - 12 operation types, automatic backend selection
- **sklearn → Aprender Converter** - 21 algorithm types, all sklearn modules
- **PyTorch → Realizar Converter** - 20+ operation types, GGUF/SafeTensors support
- **Backend Selection** - MoE routing for Scalar/SIMD/GPU
- **Plugin Architecture** - Extensible transpiler plugins
- **PARF Analyzer** - Pattern analysis, dead code detection, dependency analysis
- **Report Generation** - HTML, Markdown, JSON, Text formats
- **5-Phase Workflow** - Kanban-style migration management
- **Tool Registry** - Auto-detection of PAIML tools
- **WASM Interface** - Browser-based code analysis
- **The Batuta Book** - Comprehensive mdBook documentation
- **529 tests**, 90%+ coverage, TDG 92.6 (A)

---

[0.7.0]: https://github.com/paiml/Batuta/compare/v0.6.5...HEAD
[0.6.0]: https://github.com/paiml/Batuta/compare/v0.2.0...v0.6.5
[0.5.0]: https://github.com/paiml/Batuta/compare/v0.2.0...v0.5.0
[0.4.0]: https://github.com/paiml/Batuta/compare/v0.2.0...v0.4.0
[0.3.0]: https://github.com/paiml/Batuta/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/paiml/Batuta/compare/v0.1.9...v0.2.0
[0.1.9]: https://github.com/paiml/Batuta/compare/v0.1.3...v0.1.9
[0.1.0]: https://github.com/paiml/Batuta/releases/tag/v0.1.0
