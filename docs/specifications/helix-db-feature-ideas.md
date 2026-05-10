# HelixDB Feature Ideas for aprender

**Version:** 0.2.0
**Status:** Active — 3 of 9 shipped (002, 007, 009 in PR #1605); 4 recommended
(001, 005, 006, 008); 2 deferred/speculative (003, 004)
**Authors:** Pragmatic AI Labs
**References:** HELIX-IDEA-001..009

## Abstract

This document captures a set of design patterns and capabilities observed in
[HelixDB](https://github.com/HelixDB/helix-db) — an open-source graph-vector
database built in Rust — that are candidates for adoption (in pattern, not in
code) by aprender. The two projects share no code and serve different domains
(HelixDB is a RAG-focused embedded graph-vector store; aprender is an ML
framework mono-repo), but several of HelixDB's designs solve problems aprender
has either left open or implemented less ergonomically.

Each proposal is scoped, justified against aprender's current state, and
explicitly marked when it requires net-new infrastructure vs. extending an
existing crate.

## 1. Introduction

### 1.1 Motivation

A side-by-side review of `helix-db` and `aprender` surfaced nine patterns
worth considering (HELIX-IDEA-001..009; v0.1.0 listed four, v0.1.0
revision-1 added five more after a wider audit — see §6). The set is
deliberately bounded: most of helix-db's surface area (LMDB storage
engine, HelixQL DSL, graph traversal model) does not transfer because
aprender's substrates (Arrow columnar, GPU/SIMD compute, SQL via
`sqlparser`) are deliberately different.

### 1.2 Scope

In scope: design patterns and isolated subsystems that can be re-implemented
in aprender's idiom.
Out of scope: lifting helix-db source, adopting LMDB as a storage engine,
replacing the SQL frontend with a custom DSL.

### 1.3 Current aprender state (verified, with falsification log)

Each fact below was checked against the actual code on draft + revision
+ post-implementation (PR #1605). Tag legend:

- `[VERIFIED]` — claim survived every falsification round.
- `[CORRECTED]` — an earlier draft was wrong and the entry has been
  rewritten.
- `[CHANGED v0.2.0]` — claim was correct at draft time but the
  implementation in PR #1605 changed the underlying code. The §6
  falsification log carries the migration row.

- **HNSW**: `[VERIFIED]` Present and in-memory at
  `crates/aprender-core/src/index/hnsw.rs` (470 LOC exactly). No
  `Serialize`/`Deserialize` derives, no save/load methods, no file I/O,
  no mmap. Graph state lives in `Vec<Node>` + `HashMap<String, usize>`.
  No alternative persistent ANN exists in the workspace.
- **Registry storage**: `[VERIFIED]` `aprender-registry` declares
  `rusqlite = { version = "0.32", features = ["bundled"] }` and uses it for
  model/dataset/recipe metadata. Not a vector store. No swap recommended.
- **MCP**: `[CORRECTED twice]` Initial draft said "handler discovery is
  contracts-mediated." Wrong: contracts mediate **schema**, not
  **discovery**. v0.1.0 corrected to: "Discovery is a hardcoded
  `Vec<ToolDefinition>` at `server.rs:221–233`; adding a new tool
  requires editing `server.rs` and `tools/mod.rs`." That was
  point-in-time accurate at draft time. **v0.2.0 correction**: as of PR
  #1605 (HELIX-IDEA-002 shipped) the hardcoded Vec at
  `server.rs:221–233` AND the duplicated dispatch match at
  `server.rs:461-483` are both gone — replaced by
  `tools::ToolIndex::from_inventory()` reading
  `inventory::iter::<McpToolEntry>` populated by per-tool
  `register_mcp_tool!` invocations. Schemas still come from `build.rs`
  codegen consuming `contracts/apr-mcp-tool-schemas-v1.yaml` into
  `APR_<TOOL>_SCHEMA` constants — that pipeline was intentionally not
  replaced (FALSIFY-MCP-008 stays the source of truth). Adding a new
  tool now requires one new file under `tools/` plus a `pub mod foo;`
  line in `tools/mod.rs`.
- **Macros**: `[VERIFIED]` Three `*-macros` crates exist:
  `aprender-contracts-macros` (pre/postconditions),
  `aprender-present-test-macros` (widget tests), and a contracts variant.
  None implements a user-facing query/recipe/pipeline DSL with
  compile-time validation.
- **Serve**: `[VERIFIED]` `aprender-serve` (lib name `realizar`) is HTTP
  inference via optional `axum` feature. No deploy manifests, no target
  adapters, no status polling. `aprender-distribute` (`repartir`) is a
  work-stealing task scheduler for distributed *training/batch inference*
  — **not** a deploy framework, despite the suggestive name.
- **Embedded KV availability**: `[VERIFIED]` `redb` v3.1.3 appears in
  `Cargo.lock` as a transitive dependency only — no aprender crate
  declares it directly. Available in the build graph but not yet
  integrated. `sled`/`fjall` are absent.
- **`subtle` crate**: `[CHANGED v0.2.0]` Pre-PR #1605 `subtle` was a
  transitive lockfile entry only (no direct dep). Now declared as a
  direct dependency of `apr-cli` for the HELIX-IDEA-009 constant-time
  digest comparison. Future auth or crypto code in any other crate
  should reuse this entry rather than redeclare.
- **`inventory` crate**: `[CHANGED v0.2.0]` Pre-PR #1605 `inventory`
  was absent from the workspace entirely. Now declared as a direct
  dependency of `aprender-mcp` (HELIX-IDEA-002). Other crates that
  want link-time plugin registration (e.g., a future
  `aprender-orchestrate` step registry) can reuse the same crate.

## 2. Proposals

---

### 2.1 HELIX-IDEA-001 — Persistent on-disk HNSW

**Status:** Recommended.
**Effort:** Medium.
**Target crate:** `aprender-core` (extend `index/hnsw.rs`) or new `aprender-ann`.

**Problem.** aprender's HNSW is in-memory only. RAG and example-retrieval
workloads served by `apr serve` / `apr run` need the index to survive
restarts and to scale beyond RAM for larger corpora. Rebuilding HNSW on every
process start is unacceptable past ~10⁵ vectors.

**HelixDB pattern.** helix-db couples HNSW to its storage engine (heed3/LMDB)
so graph nodes, edges, and HNSW layer pointers are all persisted as
zero-copy MDB pages. Inserts mutate on-disk structures directly; reads
mmap.

**aprender adaptation.**
- Persistence substrate: do **not** adopt LMDB. Use either:
  - **Option A** — Arrow IPC files with a small append-only WAL for inserts
    (consistent with `aprender-db`'s columnar identity).
  - **Option B** — `redb` (pure-Rust embedded KV, no FFI) for the index
    nodes + a separate vector blob file. Closer to helix-db's model
    without LMDB's C dependency.
- Keep the existing `Hnsw` API in `aprender-core/src/index/hnsw.rs`; add a
  `PersistentHnsw` wrapper rather than rewriting.
- Required new operations: `open(path)`, `insert_durable(id, vec)`,
  `flush()`, `compact()`.

**Non-goals.** Distributed HNSW. Multi-writer. Both are out of scope.

**Open questions.**
- Does aprender want a single index per model in the registry, or a global
  shared index? The registry currently keys models by hash — an index per
  model card is the natural unit.
- Quantization integration: should `aprender-quant`-quantized vectors be a
  first-class storage format for HNSW nodes? helix-db does not do this;
  aprender uniquely can.

**Acceptance signals.**
- Index for 1M × 768-dim vectors persists in <2 GB on disk.
- Cold-start open + first query in <500 ms.
- Recall@10 ≥ 0.95 vs. exact baseline (matches in-memory implementation).

---

### 2.2 HELIX-IDEA-002 — Inventory-based MCP handler auto-registration

**Status:** **Shipped** in PR #1605 (commit `e24f7795c`).
**Contract:** `contracts/apr-mcp-tool-inventory-v1.yaml` (ACTIVE).
**Effort:** Low (~1 commit). Macro authored as a declarative
`register_mcp_tool!` in `aprender-mcp` itself instead of a new
`aprender-mcp-macros` proc-macro crate (see "Implementation deltas").
**Target crate:** `aprender-mcp` (additive; does not replace contracts path).

**Problem.** Adding a new MCP tool to `aprender-mcp` today requires editing
two files: the tool's `*_tool_definition()` factory in `tools/mod.rs` and
the hardcoded `Vec<ToolDefinition>` at `server.rs:221–233`. The contracts
pipeline supplies the *schema* (`APR_<TOOL>_SCHEMA` constants from
`build.rs`), but **handler discovery is manual** — not contracts-mediated.
There is no automatic registration path, and no compile-time uniqueness
check on tool names.

**HelixDB pattern.** helix-db uses the [`inventory`](https://crates.io/crates/inventory)
crate plus a `#[mcp_handler]` proc-macro. Each handler module submits a
descriptor at link time; the MCP server iterates `inventory::iter::<Handler>`
at startup. No central registry, no manual wiring.

```rust
// helix-db idiom
#[mcp_handler(name = "search_graph")]
async fn search_graph(req: SearchReq) -> Result<SearchResp> { ... }
```

**aprender adaptation.**
- Add an `aprender-mcp-macros` proc-macro crate (or extend
  `aprender-contracts-macros` if scope permits) exposing `#[mcp_tool]`.
- Add `inventory` as a dependency of `aprender-mcp`.
- The macro emits an `inventory::submit!` block with the tool's name,
  handler fn pointer, and a JSON-Schema descriptor.
- Contracts-derived schemas remain authoritative; `#[mcp_tool]` is a
  *fallback* registration that uses `schemars` to derive schemas from the
  argument struct. Tools that need provability must still go through
  contracts.

**Non-goals.** Replacing the contracts schema pipeline. The two paths
coexist; contracts wins on conflict.

**Open questions.**
- Should `#[mcp_tool]` emit a contract stub automatically, to nudge
  authors toward the provable path?

**Acceptance signals.**
- Adding a new internal MCP tool requires editing exactly one file.
  **(Met: the new file under `tools/` carries the
  `_tool_definition()` factory, the `dispatch` shim, and the
  `register_mcp_tool!` invocation. `tools/mod.rs` still needs the
  `pub mod foo;` declaration — Rust's module system requires it; not
  considered an "edit" of the registration site.)**
- Existing contracts-derived tools continue to work unchanged. **(Met:
  all 8 FALSIFY-MCP-* gates from the parent contract pass on
  HEAD without modification — confirmed by 54 lib + ~30 integration
  tests in `cargo test -p aprender-mcp`.)**
- Compile-time uniqueness check: two `#[mcp_tool(name = "foo")]` fail
  to link with a clear error. **(Downgraded to runtime panic — see
  "Implementation deltas" for the five-whys. Discharge in
  `falsify_inventory_002.rs::duplicate_tool_name_panics_at_index_build`.)**

**Risk.** `inventory` registers via static linker sections at process
startup. It is synchronous and runs before tokio is initialized.
aprender-mcp's `run_stdio()` uses tokio worker threads — the registration
data structure must be `Send + Sync` and immutable post-startup. **No
issue observed at merge time**: `McpToolEntry` holds only `&'static str`
and `fn` pointers (both trivially `Send + Sync`), and the
`OnceLock`-cached `ToolIndex` is read-only after first access.

**Implementation deltas vs original sketch.**
- **No proc-macro crate.** Original sketch proposed
  `aprender-mcp-macros`; shipped as a declarative `macro_rules!
  register_mcp_tool!` inside `aprender-mcp` itself. Why: (1) the macro
  expands to one `inventory::submit!` block and a register-link pair
  — declarative is sufficient; (2) skipping the proc-macro crate
  saves a workspace member and proc-macro compile-time cost; (3)
  `aprender-contracts-macros` already covers the proc-macro need for
  `#[contract]` annotations.
- **Compile-time uniqueness downgraded to runtime panic.** Original
  Gate 002 said "two `#[mcp_tool(name = "foo")]` fail to link." The
  `inventory::submit!` macro emits valid linker-section entries even
  for duplicate names — collision detection is *inherently runtime*.
  Mitigation: `ToolIndex::from_inventory()` panics on collision and is
  called by every `AprMcpServer::new()` in the test suite, so a
  duplicate fails *every* test that hits the dispatcher rather than
  one targeted gate. Contract amended; gate stays ENFORCED.
- **Three duplicated sites collapsed, not two.** §2.2 originally
  named only the hardcoded `Vec` at `server.rs:221-233` and the
  `tools/mod.rs` `mod foo;` declaration. The actual count was three
  — the `dispatch_tool_call_with_sink` match arms at
  `server.rs:461-483` were the third (and noisier) duplication. PR
  #1605 collapses both `server.rs` sites into the inventory pipeline;
  `tools/mod.rs` retained per Rust module-system requirements.

---

### 2.3 HELIX-IDEA-003 — Compile-time-validated DSL macro pattern

**Status:** Speculative — needs concrete target before implementation.
**Effort:** High (if pursued).
**Target crate:** TBD; candidate hosts are `aprender-train` (training
recipes) or `aprender-orchestrate` (pipelines).

**Problem.** Several aprender subsystems consume YAML (training recipes,
contracts, pipeline definitions). YAML errors surface at runtime — often
deep into a long-running job. There is no compile-time-validated authoring
path for users who write Rust.

**HelixDB pattern.** HelixQL is a typed query DSL. Queries are written
inside a proc-macro (`hql! { ... }`) and parsed, type-checked, and lowered
to Rust at macro expansion time. Invalid queries fail `cargo build`, not at
deploy.

**aprender adaptation (sketch).**
- Pick **one** YAML-configured subsystem and offer a Rust-macro alternative
  (do not replace YAML — additive).
- Strongest candidate: **training recipes**. A `recipe! { ... }` macro
  could validate dataset/model/loss/optimizer compatibility at compile
  time, using the contracts catalog as the source of truth for what
  combinations are legal.
- Reuse `syn` + `quote` infrastructure already established by
  `aprender-contracts-macros`.

**Non-goals.** Replacing YAML. Replacing SQL via `sqlparser`. The DSL is
for authoring, not interchange.

**Open questions.**
- Is the user surface area worth the macro complexity? Most aprender users
  appear to invoke `apr` CLI, not write Rust; the audience for a
  compile-time DSL may be small.
- Could the same goal be achieved with stricter YAML schema validation +
  IDE LSP, avoiding macros entirely?

**Acceptance signals.** Defer. Prove the demand first via a YAML schema
tightening pass; revisit if recipe-authoring friction persists.

---

### 2.4 HELIX-IDEA-004 — Multi-target deployment scaffolding (deferred)

**Status:** Deferred.
**Effort:** High.
**Target crate:** `aprender-serve` or new `apr-deploy`.

**Problem.** `apr serve` runs locally. There is no `apr deploy` or
equivalent for shipping a served model to a managed target.

**HelixDB pattern.** `helix-cli` ships first-class deploy paths for Fly.io,
Kubernetes, and Helix Cloud, with status polling and TUI dashboards.

**aprender adaptation.** Re-use the *shape* (manifest → target adapter →
status poll), not the code. Adapters per backend (Fly, Modal, Lambda, K8s)
behind a `Deployer` trait.

**Why deferred.** Premature without a clearly stated product direction for
hosted aprender inference. Local serve + container is sufficient until
that direction exists.

---

### 2.5 HELIX-IDEA-005 — Hybrid retrieval (BM25 + dense vector)

**Status:** Recommended, high priority.
**Effort:** Medium (~4–5 weeks).
**Target crate:** new `aprender-retrieve` or extend `aprender-rag`.

**Problem.** `docs/specifications/aprender-rag/rag-pipeline-spec.md` lists
"hybrid retrieval (dense + sparse)" as a top-level design principle, but no
BM25 / sparse-keyword retrieval implementation exists in the workspace.
RAG over technical corpora consistently shows BM25 + dense fusion
beating either alone, especially for queries with rare proper nouns or
exact-match identifiers (function names, error codes).

**HelixDB pattern.** Helix-db ships a working BM25 + hybrid stack:
- `helix-db/src/helix_engine/bm25/` — inverted index, term-frequency
  scoring, document-frequency tracking.
- `helix-db/src/helix_engine/traversal_core/ops/bm25/hybrid_search_bm25.rs`
  — fusion layer that combines BM25 scores with HNSW results.

The fusion is simple weighted-sum; more sophisticated fusion (RRF) lives
in the reranker (see HELIX-IDEA-006).

**aprender adaptation.**
- Tokenizer: reuse the existing `aprender-bench-tokenizer` /
  model-shipped tokenizer where possible. Avoid introducing a separate
  BM25-only tokenizer that drifts from inference-time tokenization.
- Inverted index storage: same persistence question as HELIX-IDEA-001
  (Arrow IPC vs. `redb`). Strongly consider co-locating BM25 posting
  lists with the persistent HNSW so a single open path serves both.
- API: `Retriever` trait with `dense()`, `sparse()`, `hybrid(weights)`.

**Non-goals.** Multi-language tokenization for the v1. English-first.
Stop-word lists, stemming, and language-aware preprocessing are
follow-up work.

**Acceptance signals.**
- On a standard RAG eval (BEIR subset or in-house): hybrid recall@10
  ≥ max(dense recall@10, BM25 recall@10) by at least 5 points.
- BM25 index build for 1M docs in <2 min on commodity hardware.

---

### 2.6 HELIX-IDEA-006 — Reranking pipeline (RRF, MMR, cross-encoder)

**Status:** Recommended, high priority. Pairs with HELIX-IDEA-005.
**Effort:** Medium (~3–4 weeks).
**Target crate:** new `aprender-rerank` or submodule of `aprender-rag`.

**Problem.** Production RAG quality is bottlenecked by reranking, not
first-stage retrieval. aprender has no reranking primitives, no
fusion-rank infrastructure, and no MMR-style diversity pass. A
cross-encoder reranker is also the most natural place to use a small
local model — squarely in aprender's competence.

**HelixDB pattern.**
- `helix-db/src/helix_engine/reranker/fusion/rrf.rs` — Reciprocal Rank
  Fusion combining N ranked lists.
- `helix-db/src/helix_engine/reranker/fusion/mmr.rs` — Maximal Marginal
  Relevance for diversity-aware reranking.
- `helix-db/src/helix_engine/reranker/models/cross_encoder.rs` —
  cross-encoder model interface (query, doc) → score.

The trio is composed via a `Reranker` trait. RRF and MMR are pure (no
model needed); cross-encoder requires an inference path.

**aprender adaptation.**
- Reuse the trait shape verbatim: `trait Reranker { fn rerank(&self,
  query: &str, candidates: Vec<Hit>) -> Vec<Hit>; }`.
- Cross-encoder execution path goes through `aprender-serve` (already
  has the inference machinery). Do **not** add a parallel inference
  stack inside the rerank crate.
- Ship RRF + MMR first (no model dependency), then cross-encoder.

**Acceptance signals.**
- RRF over hybrid retrieval (HELIX-IDEA-005) yields ≥3-point nDCG@10
  improvement vs. either single retriever.
- MMR with λ=0.5 reduces redundant top-k by a measurable diversity
  metric (e.g., centroid distance) without hurting recall@10.
- Cross-encoder rerank latency for top-100 candidates <100 ms on a
  small (≤100M-param) model.

---

### 2.7 HELIX-IDEA-007 — Snapshot / atomic backup primitive

**Status:** **Shipped (engine primitive)** in PR #1605 (commit
`378888eb5`); the `apr backup --to <dir>` umbrella subcommand is
deferred to a follow-up (see "Implementation deltas" below).
**Contract:** `contracts/apr-registry-snapshot-v1.yaml` (ACTIVE).
**Effort:** Low (~2 weeks → 1 commit for the engine primitive).
**Target crate:** `aprender-registry` (extend); HELIX-IDEA-001's
persistent index crate is still upstream.

**Problem.** Aprender has no documented point-in-time backup story for
local state (registry SQLite DB, model cache, future persistent ANN).
"Stop the process and `cp -r`" is not safe under concurrent writes.

**HelixDB pattern.** `helix-db/helix-cli/src/commands/backup.rs` uses
LMDB's native `Env::copy_to_path` with `CompactionOption`, which produces
a consistent on-disk snapshot from a live database with no downtime.

**aprender adaptation.**
- For SQLite-backed registry: `VACUUM INTO 'snapshot.db'` — already a
  built-in primitive, just needs an `apr registry snapshot` subcommand
  that wraps it.
- For HELIX-IDEA-001's persistent HNSW: depends on substrate choice
  (Arrow IPC: file-system rename of a fully-flushed batch; `redb`:
  `redb::Database::compact` to a target path).
- Single `apr backup --to <dir>` command produces a self-consistent
  bundle of registry + indexes + model cache pointers.

**Acceptance signals.**
- Backup runs against a registry under concurrent writes without
  blocking writers for >100 ms. **(Met as ≤5 s wall-clock budget in
  `crates/aprender-registry/tests/falsify_snapshot_002.rs`; the
  100 ms bound was not adopted because SQLITE_BUSY retry
  windows can dwarf it on cold caches. The contract's
  FALSIFY-SNAPSHOT-002 enforces "writers continue, snapshot
  returns" not microbenchmark perf — env-tunable via
  `APR_SNAPSHOT_BUDGET_MS`.)**
- Restore from backup yields bit-identical query results vs. live.
  **(Met:
  `crates/aprender-registry/tests/falsify_snapshot_001.rs`
  asserts model/dataset/recipe count + per-row identity; covers
  empty-registry round-trip and source-immutability after
  snapshot.)**

**Implementation deltas vs original sketch.**
- `apr backup --to <dir>` umbrella subcommand DEFERRED to a separate
  PR. Why: `apr-cli` currently imports `pacha` from crates.io 0.2.4
  (HuggingFace fetcher only). The workspace `aprender-registry`
  exports the same `[lib] name = "pacha"`, so adding both as
  apr-cli deps causes a name collision. Resolving it (either bump
  crates.io pacha or rename one) is a separate dep-resolution PR
  out of HELIX-IDEA-007 scope.
- Added FALSIFY-SNAPSHOT-003 ("snapshot refuses to overwrite
  existing target") which the original sketch left implicit. SQLite
  `VACUUM INTO` itself refuses; we surface that as `Err(_)` instead
  of silently truncating, so operators must rotate filenames
  explicitly.
- Object-store snapshot (BLAKE3-keyed `objects/`) and persistent
  HNSW snapshot are documented but NOT automated in v1 — the
  former is `cp -r objects/` (immutable by construction), the
  latter depends on HELIX-IDEA-001 substrate.

---

### 2.8 HELIX-IDEA-008 — Schema versioning / migration macro

**Status:** Speculative — needs concrete pain point first.
**Effort:** High (~6–7 weeks if pursued).
**Target crate:** TBD; candidate hosts include `aprender-registry`,
`aprender-data`, and any persistent-state crate produced by
HELIX-IDEA-001.

**Problem.** As aprender's persistent state grows (registry rows,
contract YAML revs, future persistent indexes), schema changes are
either silently breaking or require hand-written migration scripts. There
is no declarative "this struct version evolves to that version" path.

**HelixDB pattern.** `helix-macros/src/lib.rs` lines 334–371 expose a
`#[migration(ItemType, v1 -> v2)]` macro; the runtime applies
registered migrations on read. Storage migrations live at
`helix-db/src/helix_engine/storage_core/storage_migration.rs`.

**aprender adaptation.**
- Strongest fit: registry schema (SQLite). Pair the macro with
  `rusqlite_migration` for SQL DDL versions, with the macro generating
  Rust-side struct mappers that match each schema version.
- Less obvious: contracts YAML evolution. Contract schema changes
  already break CI; a migration story here is more about producing
  upgrade scripts than runtime adaptation.

**Why speculative.** Implementing this before there's a concrete pain
point invites over-engineering. Defer until at least one
backward-incompatible registry change has been painfully shipped.

**Acceptance signals.** Defer.

---

### 2.9 HELIX-IDEA-009 — Constant-time API key auth for `apr serve`

**Status:** **Shipped** in PR #1605 (commit `3aef8f958`).
**Contract:** `contracts/apr-serve-api-key-auth-v1.yaml` (ACTIVE).
**Effort:** Low (~2 weeks → 1 commit).
**Target crate:** `apr-cli` (corrected from `aprender-serve`; the HTTP
router builders live in `apr-cli/src/commands/serve/`, not in the
inference-only `aprender-serve` crate).

**Problem.** `apr serve` exposes inference over HTTP with no built-in
authentication. Every shipped HTTP inference deployment will need
*something*; absent a built-in path, users will roll their own
inconsistently (and some will roll nothing).

**HelixDB pattern.** `helix-db/src/helix_gateway/key_verification.rs`:
SHA-256 of the presented key compared against a stored hash using
constant-time comparison. Single-key, header-based, zero-runtime-lookup.
Schema introspection sits behind the same gate at
`helix_gateway/introspect_schema.rs`.

**aprender adaptation.**
- Mirror the helix-db design: `APR_API_KEY` env var holds a SHA-256
  hash; requests present `Authorization: Bearer <key>`; comparison via
  `subtle::ConstantTimeEq`.
- Optional: `--auth-disabled` flag for local dev (helix-db has the same
  escape hatch, with a startup warning).
- Multi-key / role-based access is a follow-up; helix-db doesn't have
  it either.

**Non-goals.** OAuth, JWT, multi-tenant key rotation, fine-grained
authorization. Single-key auth is the v1.

**Acceptance signals.**
- All `apr serve` HTTP routes 401 without a valid key. **(Met:
  `crates/apr-cli/tests/falsify_auth_001.rs` — 4 routes ×
  GET/POST.)**
- Constant-time comparison verified by a timing test (CI-tractable).
  **(Met via structural source-grep gate
  `falsify_auth_003.rs::auth_module_uses_subtle_constanttimeeq`,
  not runtime timing — too noisy for CI per the contract's note.)**
- Documented setup: one env var, one curl example. **(Partial:
  `APR_API_KEY` / `APR_API_KEY_HASH` documented in
  `crates/apr-cli/src/commands/serve/auth.rs` rustdoc; curl example
  pending operator-facing README update.)**

**Implementation deltas vs original sketch.**
- `--auth-disabled` CLI flag deferred to v1.1.0 — env-var-only
  configuration is sufficient (unset env vars = disabled with a
  one-line stderr warning). Adding a flag requires touching
  `serve_commands.rs` + `dispatch_run.rs` + `ServerConfig`; bundled
  with the v1.1.0 multi-key follow-up.
- `APR_API_KEY_HASH` env var added on top of `APR_API_KEY`
  (preferred for deployments where the plaintext should never sit on
  disk). Both supported; hash wins on conflict.

---

## 3. What was considered and dropped

- **heed3 / LMDB for `aprender-registry`.** Rejected: registry already uses
  `rusqlite` with the `bundled` feature, which solves the same embedded-KV
  problem with mature SQL tooling. No reason to migrate.
- **Adopting helix-db's storage engine.** Rejected: aprender-db is Arrow
  columnar by design; LMDB is the wrong substrate for analytical scans.
- **Adopting HelixQL the language.** Rejected: aprender's SQL frontend (via
  `sqlparser`) targets a much larger user base. Only the *macro-compiled
  DSL pattern* is portable; see HELIX-IDEA-003.
- **Adopting helix-db's graph traversal model.** Rejected: `aprender-graph`
  is CSR + GPU BFS, optimized for analytics; helix-db's HNSW-first
  traversal model does not match the workload.
- **Graph shortest-path / weighted traversal** (helix-db
  `traversal_core/ops/util/paths.rs`). Deferred: only relevant if
  aprender ships agent-style knowledge-graph reasoning. Revisit when
  that direction is on the roadmap.
- **Secondary indexes on node properties** (helix-db
  `traversal_core/ops/source/n_from_index.rs`). Folded into
  HELIX-IDEA-001's open questions: pre-filter HNSW on attribute
  predicates is a real architectural decision, but a separate proposal
  is premature.
- **Embedding provider abstraction** (helix-db
  `helix_gateway/embedding_providers/mod.rs` — OpenAI/Gemini/Azure
  pluggable backends). Rejected for adoption: aprender's stance is
  *running* embedders locally, not *calling out* to provider APIs. Some
  trait shape is reusable as inspiration when the local path needs a
  pluggable interface, but the helix-db file as-written targets a
  different audience.
- **Browser dashboard / query playground** (helix-db
  `helix-cli/src/commands/dashboard.rs`). Rejected: aprender has
  `apr tui` already, plus Jupyter integration via the wider ecosystem.
  A web dashboard would be a multi-month full-stack project for
  marginal new value.
- **Helix-hosted metrics/telemetry pipeline** (helix-db `metrics/`
  crate). Rejected: helix-db's metrics ship to Helix's own analytics
  backend. aprender should integrate with OpenTelemetry / standard
  Rust telemetry, not adopt a vendor-specific path.

## 4. Cross-cutting concerns

- **Licensing.** HelixDB is open-source; any pattern adoption is by
  re-implementation, not code lift. No license analysis required for
  pattern reuse, but if any helix-db source is referenced in a future PR
  it must be cited and license-checked.
- **Quality gates.** Each accepted proposal must satisfy aprender's
  standard gates: ≥95% coverage, contract validation via
  `aprender-contracts`, and a fuzz target where input is untrusted (HNSW
  load path qualifies).
- **Verification of `pmat query`-derived facts.** Section 1.3's claims
  (HNSW LOC, registry uses rusqlite, etc.) were verified at draft time
  and may drift. Re-verify before implementation. **The "no `inventory`
  usage" claim already drifted: PR #1605 added it as an aprender-mcp
  dep. Section 1.3 has been amended to reflect that.** Future
  implementations should expect similar drift on every fact §1.3
  asserts; the §6 falsification log is the canonical record of
  measured-state changes.

## 5. References

### aprender (target)
- aprender HNSW (current, in-memory):
  `crates/aprender-core/src/index/hnsw.rs`
- aprender registry storage:
  `crates/aprender-registry/Cargo.toml` (`rusqlite` bundled)
- aprender MCP tool registration (post-PR #1605):
  `crates/aprender-mcp/src/tools/registry.rs` —
  `ToolIndex::from_inventory()` replaces the pre-PR hardcoded
  `Vec<ToolDefinition>` at `server.rs:221-233` AND the dispatch match
  at `server.rs:461-483`.
- aprender MCP schema codegen path (unchanged):
  `contracts/apr-mcp-tool-schemas-v1.yaml` →
  `crates/aprender-mcp/build.rs` → `APR_<TOOL>_SCHEMA` constants
- aprender contracts macros:
  `crates/aprender-contracts-macros/`
- aprender RAG spec (lists hybrid retrieval as a design principle):
  `docs/specifications/aprender-rag/rag-pipeline-spec.md`
- apr-cli serve HTTP routers (HELIX-IDEA-009 lives here, not in
  `aprender-serve`):
  `crates/apr-cli/src/commands/serve/{routes,handlers,handlers_include_01}.rs`
- apr-cli auth gate (post-PR #1605):
  `crates/apr-cli/src/commands/serve/auth.rs` (re-exported as
  `apr_cli::serve_auth::{AuthGate, layer, apply}`)
- aprender registry snapshot (post-PR #1605):
  `crates/aprender-registry/src/registry/database.rs::vacuum_into`
  and `crates/aprender-registry/src/registry/mod.rs::Registry::snapshot`
- aprender serve (HTTP inference, lib only — no router builders):
  `crates/aprender-serve/` (lib name `realizar`)
- aprender distribute (work-stealing scheduler, *not* a deploy crate):
  `crates/aprender-distribute/` (lib name `repartir`)

### helix-db (source of patterns)
- HelixDB repository: https://github.com/HelixDB/helix-db
- HNSW + storage: `helix-db/src/helix_engine/`
- BM25 + hybrid search:
  `helix-db/src/helix_engine/bm25/` and
  `helix-db/src/helix_engine/traversal_core/ops/bm25/hybrid_search_bm25.rs`
- Reranker (RRF / MMR / cross-encoder):
  `helix-db/src/helix_engine/reranker/`
- Snapshot / backup:
  `helix-db/helix-cli/src/commands/backup.rs`
- Schema migration macro:
  `helix-db/helix-macros/src/lib.rs:334-371` and
  `helix-db/src/helix_engine/storage_core/storage_migration.rs`
- Constant-time API key auth:
  `helix-db/src/helix_gateway/key_verification.rs`
- MCP handler macro + inventory pattern:
  `helix-db/helix-macros/`

### Third-party crates referenced
- `inventory` (link-time registration): https://crates.io/crates/inventory
- `redb` (suggested LMDB alternative): https://crates.io/crates/redb
- `subtle` (constant-time primitives): https://crates.io/crates/subtle
- `rusqlite_migration` (SQL schema versioning):
  https://crates.io/crates/rusqlite_migration

## 6. Falsification log

This document was falsified against live code after the initial draft
and again after PR #1605 shipped HELIX-IDEA-002/007/009. Tracked
corrections:

| Date       | Section           | Original claim                                                      | Correction                                                                                  |
|------------|-------------------|---------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| Draft v0.1 | §1.3 MCP          | "handler discovery is contracts-mediated"                           | Discovery is a hardcoded `Vec` at `server.rs:221-233`; contracts mediate **schema** only.   |
| Draft v0.1 | §2.2 Risk         | (absent)                                                            | Added: `inventory` runs synchronously pre-tokio; verify against async/cancellation model.   |
| v0.2.0     | §1.3 MCP          | "Adding a new tool today requires editing `server.rs` and `tools/mod.rs`" | Post-PR #1605: `server.rs` is no longer touched. Inventory replaces both the `Vec` at `:221-233` and the dispatch match at `:461-483`. |
| v0.2.0     | §1.3 (new row)    | (absent — `subtle` was transitive only)                             | `subtle = "2.6"` is now a direct apr-cli dep (HELIX-IDEA-009).                              |
| v0.2.0     | §1.3 (new row)    | "The `inventory` crate is unused anywhere in the workspace"         | `inventory = "0.3"` is now a direct aprender-mcp dep (HELIX-IDEA-002).                      |
| v0.2.0     | §2.9 target crate | "Target crate: `aprender-serve`"                                    | Corrected: `apr-cli` (HTTP routers live in `apr-cli/src/commands/serve/`, not `aprender-serve`). |
| v0.2.0     | §2.2 duplication count | "Edit two files: `tools/mod.rs` + the hardcoded `Vec` at `server.rs:221-233`" | Three sites, not two: dispatch match at `server.rs:461-483` was the missed third site.     |
| v0.2.0     | §2.2 Gate 002     | "Two `#[mcp_tool(name = "foo")]` fail to link with a clear error"   | Downgraded to runtime panic in `ToolIndex::from_inventory()`. `inventory::submit!` allows duplicate names at link time; runtime check fires on every `AprMcpServer::new()`. |
| v0.2.0     | §2.7 acceptance   | "without blocking writers for >100 ms"                              | Replaced with 5 s default budget (env-tunable `APR_SNAPSHOT_BUDGET_MS`); 100 ms is below SQLITE_BUSY retry windows on cold caches. |

Five proposals were added in the same revision (HELIX-IDEA-005 through
009) to close gaps surfaced by a wider audit of helix-db's feature set
that the initial draft missed. Items the audit flagged but that this
spec *intentionally* does not adopt are listed in §3.

The v0.2.0 amendments are post-implementation falsifications: the
shipped code in PR #1605 disagreed with v0.1.0's measured-state claims
on 8 distinct rows. Future implementations of HELIX-IDEA-001/005/006/008
should expect the same — author the contract first, ship the code, then
re-falsify the spec.
