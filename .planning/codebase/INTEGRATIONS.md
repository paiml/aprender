# External Integrations

**Analysis Date:** 2026-08-07

## APIs & External Services

**Model and Dataset Hubs:**
- Hugging Face Hub - Pull and push `.apr`, GGUF, SafeTensors, tokenizer, configuration, and dataset artifacts.
  - SDK/Client: synchronous `hf-hub` in `crates/aprender-core/src/hf_hub/client_impl.rs`; direct Ureq/Reqwest API and Git LFS flows in `crates/aprender-core/src/hf_hub/upload.rs`, `crates/aprender-core/src/hf_hub/xet.rs`, and `crates/aprender-data/src/hf_hub/`.
  - Auth: `HF_TOKEN`; anonymous public downloads are supported by `crates/aprender-core/src/hf_hub/client_impl.rs`.
  - Large files: artifacts above the Xet threshold route through optional `hf-xet` support in `crates/aprender-core/src/hf_hub/xet.rs`, including token refresh, CAS upload, and LFS pointer commits.
  - CLI: `apr pull`, `hf://` resolution, shard extraction, caching, and dataset tree pagination are implemented under `crates/apr-cli/src/commands/pull.rs`, `crates/apr-cli/src/commands/pull_extract_shard.rs`, and `crates/apr-cli/src/commands/pull_dataset.rs`.
  - Endpoint caveat: `crates/apr-cli/src/commands/hf_endpoint.rs` defines an `HF_ENDPOINT` helper, but the production pull/download paths still construct `https://huggingface.co` URLs directly; do not rely on endpoint override support without wiring it into those callers.

**Remote Model Providers:**
- Anthropic Messages API - Remote agent execution and optional RAG evaluation.
  - SDK/Client: direct Reqwest calls, not an Anthropic SDK, in `crates/aprender-orchestrate/src/agent/driver/remote.rs` and `crates/aprender-rag/src/eval/client.rs`.
  - Auth: `ANTHROPIC_API_KEY`, selected for Claude model identifiers in `crates/aprender-orchestrate/src/cli/agent_helpers.rs`.
- OpenAI-compatible Chat Completions - Remote agent execution against OpenAI-compatible endpoints.
  - SDK/Client: direct Reqwest and SSE parsing in `crates/aprender-orchestrate/src/agent/driver/remote.rs`.
  - Auth: `OPENAI_API_KEY`, selected for non-Claude remote model identifiers in `crates/aprender-orchestrate/src/cli/agent_helpers.rs`.
- Provider scope - `crates/aprender-orchestrate/src/serve/backends.rs` enumerates additional backend names and privacy classifications, but concrete outbound drivers verified in the codebase are Anthropic Messages and OpenAI-compatible Chat Completions. Treat other names as selection metadata until a client implementation is present.
- Privacy policy - Sovereign mode blocks remote APIs, Private mode permits controlled/owned endpoints, and Standard mode permits public providers; enforcement lives in orchestration privacy and backend code under `crates/aprender-orchestrate/src/`.

**Registries and Metadata Services:**
- Pacha registry - Local-first model, dataset, recipe, run, and lineage registry in `crates/aprender-registry/`.
  - SDK/Client: in-tree Rust registry API; optional Reqwest remote client in `crates/aprender-registry/src/remote.rs`.
  - Auth: programmatic `None`, Bearer token, Basic credentials, or custom API-key header in `crates/aprender-registry/src/remote.rs`; no fixed environment-variable contract is defined by the client.
  - Resolution: `pacha://`, `hf://`, `file://`, and local/remote model references are handled by `crates/aprender-registry/src/resolver.rs` and `crates/aprender-registry/src/fetcher.rs`.
- arXiv API - Unauthenticated paper search against `http://export.arxiv.org/api/query` using Reqwest and Quick XML in `crates/aprender-orchestrate/src/oracle/arxiv.rs`; a curated local database is also available.
- crates.io API - Unauthenticated package, version, and dependency metadata with in-memory/persistent caching in `crates/aprender-orchestrate/src/stack/crates_io/client.rs`.

**Object Storage:**
- Amazon S3 and S3-compatible services - Optional data backend supporting AWS S3, MinIO, Ceph, Cloudflare R2, and compatible custom endpoints.
  - SDK/Client: AWS SDK for Rust behind the `s3` feature in `crates/aprender-data/Cargo.toml` and `crates/aprender-data/src/backend/s3.rs`.
  - Auth: AWS default credential chain, explicit static credentials, or anonymous access, selected through `CredentialSource` in `crates/aprender-data/src/backend/s3.rs`.
- URI recognition - `crates/apr-cli/src/commands/pull_scheme.rs` recognizes `s3://`, `gs://`, and `az://` and reports conventional credential names, but its comments identify actual cloud-object download as follow-up work. Do not plan on operational GCS or Azure Blob pulls through this CLI path.

**Model Context Protocol:**
- Aprender MCP server - JSON-RPC 2.0 using MCP protocol version 2024-11-05 over standard I/O, exposing APR validation, tensor, benchmark, QA, trace, run, serve, and fine-tune tools in `crates/aprender-mcp/src/lib.rs` and `crates/aprender-mcp/src/server.rs`.
  - SDK/Client: manual stdio server with optional PMCP dispatcher integration declared in `crates/aprender-mcp/Cargo.toml`.
  - Auth: inherited process/transport security; the stdio server has no separate identity provider.
- External MCP tools - Orchestration loads project `.mcp.json` definitions and spawns stdio subprocesses through `crates/aprender-orchestrate/src/agent/tool/mcp_client.rs` and `crates/aprender-orchestrate/src/agent/tool/mcp_json.rs`.
  - Transport caveat: configuration types include stdio, SSE, and WebSocket, while the verified custom client implementation is stdio-focused. Sovereign privacy validation rejects network transports.
- Banco MCP surface - Banco registers `/api/v1/mcp` in `crates/aprender-orchestrate/src/serve/banco/router.rs`; this HTTP route is distinct from the standalone standard-I/O server in `crates/aprender-mcp/src/server.rs`.

## Data Storage

**Databases:**
- Bundled SQLite - Pacha registry metadata is stored in `registry.db`, with schema and migrations owned by `crates/aprender-registry/src/registry/database.rs`.
  - Connection: local file under the registry root, which defaults to `~/.pacha` in `crates/aprender-registry/src/registry/mod.rs`.
  - Client: `rusqlite` with bundled SQLite, declared in `crates/aprender-registry/Cargo.toml`.
- SQLite with FTS5 - RAG documents, chunks, and retrieval indexes are stored locally by `crates/aprender-rag/`; dependency and feature configuration is in `crates/aprender-rag/Cargo.toml`.
- Arrow/Parquet - Columnar datasets, checkpoints, and analytical storage are handled by `crates/aprender-data/`, `crates/aprender-db/`, and `crates/aprender-distribute/`; these are file formats rather than a managed external database.

**File Storage:**
- Local filesystem - Default model, dataset, registry, checkpoint, conversation, and Hugging Face caches are local-first.
- Content-addressed registry objects - BLAKE3-sharded CAS files with atomic rename are implemented in `crates/aprender-registry/src/storage/object_store.rs` below the Pacha registry root.
- Hugging Face model cache - `crates/aprender-core/src/hf_hub/client_impl.rs` uses the platform cache directory under `huggingface/hub`; CLI token/cache discovery is implemented in `crates/apr-cli/src/commands/pull_extract_shard.rs`.
- Hugging Face dataset cache - `crates/apr-cli/src/commands/pull_dataset.rs` stores data under `~/.cache/aprender/datasets/<repo>` by default.
- Banco state - Configuration defaults to `~/.banco/config.toml` and conversations to JSONL under `~/.banco/conversations/`, as defined by Banco modules under `crates/aprender-orchestrate/src/serve/banco/`.
- Optional S3-compatible storage - Enabled only through the `s3` feature in `crates/aprender-data/Cargo.toml`; local and memory-mapped backends remain the default.

**Caching:**
- Pacha model cache - Uses `XDG_CACHE_HOME/pacha/models`, otherwise platform cache paths such as `~/.cache/pacha/models`, in `crates/aprender-registry/src/fetcher.rs`.
- Pacha cache management - Default maximum size is 50 GB with LRU/age cleanup in cache modules under `crates/aprender-registry/src/`.
- Hugging Face cache selection - `HUGGINGFACE_HUB_CACHE`, then `HF_HOME/hub`, then the default Hugging Face cache is used by `crates/aprender-qa-runner/src/conversion_hf_cache.rs`.
- Service caches - crates.io metadata caching is implemented by `crates/aprender-orchestrate/src/stack/crates_io/client.rs`; no Redis, Memcached, or external managed cache was detected.

## Serving Endpoints

**Aprender Inference Server:**
- Native REST - Health, liveness, readiness, metrics, model management, tokenization, generation, batch, streaming, realization, APR predict/explain/audit, and GPU routes are assembled in `crates/aprender-serve/src/api/router.rs`.
- OpenAI compatibility - `/v1/models`, `/v1/completions`, `/v1/chat/completions`, streaming, and embeddings are registered in `crates/aprender-serve/src/api/router.rs`.
- Ollama compatibility - `/api/chat` and `/api/generate` are registered in `crates/aprender-serve/src/api/router.rs`.
- Streaming - Server-sent event generation is implemented by routes under `crates/aprender-serve/src/api/`; WebSocket is used by Banco rather than the base inference router.
- Network defaults - The APR CLI serve configuration defaults to port 8080 in `crates/apr-cli/src/commands/serve/types.rs`; base server routing uses permissive CORS in `crates/aprender-serve/src/api/router.rs`.

**Banco Workbench:**
- Banco API - `/api/v1` routes cover models, chat, tokenization, embeddings, data, RAG, training, evaluation, experiments, batch jobs, configuration, audit, metrics, audio, tools, and MCP in `crates/aprender-orchestrate/src/serve/banco/router.rs`.
- Compatibility APIs - Banco also exposes OpenAI-compatible `/v1`, Ollama-compatible `/api`, and `/api/v1/ws` WebSocket routes in `crates/aprender-orchestrate/src/serve/banco/router.rs`.
- Network defaults - Banco binds to `127.0.0.1:8090` by default, configured under `crates/aprender-orchestrate/src/serve/banco/`.

## Authentication & Identity

**Auth Provider:**
- Custom API key - The repository does not integrate a third-party identity provider.
  - APR CPU fallback: `crates/apr-cli/src/commands/serve/auth.rs` accepts Bearer credentials using `APR_API_KEY_HASH` (preferred SHA-256 form) or `APR_API_KEY`, with constant-time hash comparison; this gate is wired by `crates/apr-cli/src/commands/serve/handlers.rs`.
  - APR routing caveat: GGUF/GPU/Realizar paths created in `crates/apr-cli/src/commands/serve/server.rs` call the router from `crates/aprender-serve/src/api/router.rs`, where no authentication middleware is registered. Do not assume `APR_API_KEY*` protects every `apr serve` path without adding a common wrapper.
  - Banco local mode: `crates/aprender-orchestrate/src/serve/banco/auth.rs` defines local unauthenticated mode and scoped API-key mode. Production state constructors verified under `crates/aprender-orchestrate/src/serve/banco/` initialize local mode; `AuthStore::api_key_mode()` has no production caller.
- Remote provider authentication - Hugging Face, Anthropic, OpenAI, AWS, OTLP, and Pacha remote credentials are integration-specific and do not establish end-user identity.

## Monitoring & Observability

**Error Tracking:**
- None detected - No Sentry or hosted error-tracking SDK is declared in the workspace manifests. Use structured tracing and metrics already present in `crates/aprender-profile/` and serving crates.

**Logs:**
- `tracing` and `tracing-subscriber` provide structured application and request logs across serving, orchestration, profiling, and CLI crates; shared versions are declared in `Cargo.toml`.
- Prometheus-compatible text metrics are exposed by inference and Banco routers in `crates/aprender-serve/src/api/router.rs` and `crates/aprender-orchestrate/src/serve/banco/router.rs`; Prometheus is a scrape consumer, not a required hosted integration.
- OTLP export is optional/default-enabled for profiling through OpenTelemetry 0.31 in `crates/aprender-profile/Cargo.toml` and `crates/aprender-profile/src/otlp_exporter.rs`.
- OTLP endpoints such as Jaeger or Tempo are supplied with CLI/configuration or `RENACER_OTLP_ENDPOINT`; optional bearer auth uses `RENACER_AUTH_TOKEN` in `crates/aprender-profile/`.

## CI/CD & Deployment

**Hosting:**
- No single managed hosting provider is required. Deployments target local/bare-metal binaries, Docker, optional AWS Lambda detection, and WASM edge environments through `crates/aprender-serve/src/target.rs`.
- CPU and NVIDIA CUDA serving images are defined in `crates/aprender-serve/Dockerfile` and `crates/aprender-serve/Dockerfile.gpu`; orchestration has `crates/aprender-orchestrate/Dockerfile`.
- Docker Compose definitions exist for data, orchestration, presentation, serving, simulation, and training at `crates/aprender-data/docker-compose.yml`, `crates/aprender-orchestrate/docker-compose.yml`, `crates/aprender-present/docker-compose.yml`, `crates/aprender-serve/docker-compose.yml`, `crates/aprender-simulate/docker-compose.yml`, and `crates/aprender-train/docker-compose.yml`.
- Crates are distributed through crates.io according to package manifests and release targets in `Makefile`; Linux binaries are attached to GitHub releases by `.github/workflows/binary-release.yml`.

**CI Pipeline:**
- GitHub Actions - Main CI delegates to the reusable sovereign workflow in `.github/workflows/ci.yml`; contributor authorization is handled by `.github/workflows/pr-gate.yml`.
- Clean-room validation - Self-hosted Docker builds, nextest, sccache, security checks, coverage, CUDA jobs, docs, and model stories are defined across `.github/workflows/`.
- Codecov - Coverage output configured by `codecov.yml` is published from `.github/workflows/coverage.yml`.
- Supply chain - SLSA provenance and dependency-policy checks are part of the GitHub workflow and `deny.toml`/`.cargo/audit.toml` configuration.

## Environment Configuration

**Required env vars:**
- No environment variable is universally required for local CPU library/CLI use; integration variables are conditional.
- `HF_TOKEN` - Private Hugging Face repositories and uploads in `crates/aprender-core/src/hf_hub/client_impl.rs` and `crates/aprender-data/src/hf_hub/upload.rs`.
- `HUGGINGFACE_HUB_CACHE`, `HF_HOME` - Hugging Face cache location discovery in `crates/aprender-qa-runner/src/conversion_hf_cache.rs`.
- `HF_ENDPOINT` - Defined and validated only by `crates/apr-cli/src/commands/hf_endpoint.rs`; active pull paths are not wired to it.
- `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` - Remote orchestration providers selected in `crates/aprender-orchestrate/src/cli/agent_helpers.rs`.
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and the standard AWS SDK environment/default chain - Optional S3 backend in `crates/aprender-data/src/backend/s3.rs`.
- `APR_API_KEY_HASH`, `APR_API_KEY` - Optional Bearer gate for the APR CPU fallback router in `crates/apr-cli/src/commands/serve/auth.rs`; not universal across all serve paths.
- `RENACER_OTLP_ENDPOINT`, `RENACER_AUTH_TOKEN` - Optional OTLP destination and authorization in `crates/aprender-profile/`.
- `AWS_LAMBDA_FUNCTION_NAME` - Runtime target detection in `crates/aprender-serve/src/target.rs`, not application authentication.
- `APR_CONFIG` - Orchestration configuration path under `crates/aprender-orchestrate/`.

**Secrets location:**
- Process environment or external runtime secret management; no root `.env` file is present and secret values must not be committed.
- Hugging Face CLI token discovery also checks user token/cache files in `crates/apr-cli/src/commands/pull_extract_shard.rs`; these are user-local files, not repository assets.
- Pacha remote credentials are supplied programmatically using authentication variants in `crates/aprender-registry/src/remote.rs` rather than a fixed repository environment contract.
- Project MCP configuration can supply subprocess environment settings through `.mcp.json` as parsed by `crates/aprender-orchestrate/src/agent/tool/mcp_json.rs`; keep any credential-bearing local file outside version control.

## Webhooks & Callbacks

**Incoming:**
- None detected - No webhook-specific receiver or signature-verification flow was found. REST, SSE, MCP, and WebSocket surfaces in `crates/aprender-serve/src/api/router.rs`, `crates/aprender-orchestrate/src/serve/banco/router.rs`, and `crates/aprender-mcp/src/server.rs` are interactive APIs rather than webhook handlers.

**Outgoing:**
- None detected - No generic webhook dispatcher or provider-specific outgoing webhook client was found.
- Internal progress callbacks in Hugging Face upload/download code under `crates/aprender-core/src/hf_hub/` are in-process callbacks, not network webhooks.

---

*Integration audit: 2026-08-07*
