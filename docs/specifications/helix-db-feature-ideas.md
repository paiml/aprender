# HelixDB Feature Ideas for aprender

**Version:** 0.6.0
**Status:** Active — 3 of 9 fully shipped (002, 007, 009 in PR #1605);
**1 partially shipped** (001 Phases 1-2 of 4 — round-trip + atomic-write
crash safety ENFORCED; recall threshold / cold-open latency pending
Phases 3-4); 2 recommended with **pre-authored gates** (005, 006); 1
recommended without gates (008, speculative pending pain point); 2
deferred/speculative (003, 004)
**Methodology:** Design by Provable Contract (`aprender-contracts` /
`pv` CLI). Every shipped HELIX-IDEA carries an ACTIVE
`contracts/*.yaml`, an ENFORCED set of falsification gates, and an
aprender-contracts integration test that pins the gate→test mapping.
Recommended-but-unshipped ideas now carry pre-authored gate IDs in
their §2.x "Pre-authored falsification gates" tables, so a future
implementation PR can transcribe them directly into the YAML
without inventing gate names under time pressure. See §1.4.
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
existing crate. Authoring follows aprender's **Design by Provable
Contract** discipline: every shipped idea is gated by an ACTIVE
provable-contract YAML whose `falsification_conditions:` entries each
map to a shipped Rust test, with an `aprender-contracts` integration
test asserting the gate→test mapping holds on disk. See §1.4 for the
full chain and the audit table for HELIX-IDEA-002/007/009.

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

### 1.4 Design by Provable Contract

This spec is authored under aprender's **Design by Provable Contract**
discipline. The methodology, instantiated by the in-tree
`aprender-contracts` crate and the `pv` CLI (APR-MONO Phase 2b), is:

1. **Spec proposal** — a `HELIX-IDEA-NNN` entry in §2 names a problem,
   the helix-db pattern that solves it, and an aprender adaptation
   that does not lift code. Each proposal lists explicit
   "Acceptance signals" — the falsifiable assertions that must hold
   for the idea to be considered shipped.
2. **Provable contract YAML** — a `contracts/<idea>-v1.yaml` file
   under [`metadata.kind: registry`](../../crates/aprender-contracts/src/schema/kind.rs)
   declares the contract's `falsification_conditions:` list. Each
   entry binds an ID (e.g., `FALSIFY-AUTH-001`) to a `test_file:` +
   `test_name:` pair plus a `status:` of `ENFORCED`. `pv validate
   contracts/...yaml` parses, validates, and rejects malformed
   contracts; `pv lint contracts/` runs the strict gate
   (validate + audit + score) workspace-wide.
3. **Falsifier discharge** — every `test_file` is a real Rust test
   in `crates/<crate>/tests/` that runs as part of `cargo test`.
   Tests assert *negative* claims ("missing bearer fails to 401"),
   not just positive ones — falsification is the load-bearing
   shape, per Popper.
4. **Integration test** — a sibling `aprender-contracts` test
   (`crates/aprender-contracts/tests/<idea>_contract.rs`) loads the
   YAML, asserts `status: ACTIVE`, asserts the exact set of
   `FALSIFY-*-NNN` IDs is present, and asserts every referenced
   `test_file:` exists on disk. A renamed or deleted falsifier
   breaks this integration test before the crate it points at even
   compiles — drift cannot ship silently.
5. **Spec re-falsification** — after the implementation merges, this
   spec's §1.3 measured-state and §6 falsification log are
   re-walked against HEAD; any drift between v0.1.0 claims and live
   code is recorded as a `[CHANGED vX.Y.Z]` row. v0.2.0's amendments
   are this loop's first execution; v0.3.0's contract-chain audit
   below is the second.

**Why it matters for this spec specifically.** HelixDB itself is
*not* contract-driven — it documents acceptance criteria in prose
and ships tests that resemble them. We deliberately do not lift that
practice. Every helix-db pattern adopted here is reframed in
aprender's contract idiom before merging. The
`aprender-contracts-macros` `#[contract]` annotation is available
but not required for these registry-kind contracts; provability
applies to the dispatch behaviour ("the gate's test fails iff the
property fails"), not to the YAML's mathematical invariants.

#### Contract chain audit (HELIX-IDEA-002/007/009)

Every shipped idea is reachable from this table. A row that doesn't
hold (renamed test, missing YAML, dropped gate) breaks the
corresponding `aprender-contracts` integration test in CI.

| Idea | Contract YAML | Status | Falsifiers (all `ENFORCED`) | Integration test |
|---|---|---|---|---|
| **HELIX-IDEA-001** (Persistent HNSW — Phases 1-2) | `contracts/apr-hnsw-persistence-v1.yaml` v1.1.0 | ACTIVE | `FALSIFY-HNSW-PERSIST-001` → `crates/aprender-core/tests/falsify_hnsw_persist_001.rs::reopen_top_k_matches_in_memory`<br>`FALSIFY-HNSW-PERSIST-002` → `crates/aprender-core/tests/falsify_hnsw_persist_002.rs::partial_write_does_not_silently_corrupt` <br>(Phases 3-4 pending — see §2.1 phase column) | `crates/aprender-contracts/tests/apr_hnsw_persistence_contract.rs` (6 assertions) |
| **HELIX-IDEA-002** (MCP inventory) | `contracts/apr-mcp-tool-inventory-v1.yaml` | ACTIVE | `FALSIFY-INVENTORY-001` → `crates/aprender-mcp/tests/falsify_inventory_001.rs::inventory_yields_same_tool_set_as_hardcoded_list`<br>`FALSIFY-INVENTORY-002` → `crates/aprender-mcp/tests/falsify_inventory_002.rs::duplicate_tool_name_panics_at_index_build`<br>`FALSIFY-INVENTORY-003` → `crates/aprender-mcp/tests/falsify_inventory_003.rs::inventory_dispatch_envelope_matches_hardcoded_path` | `crates/aprender-contracts/tests/apr_mcp_tool_inventory_contract.rs` (6 assertions) |
| **HELIX-IDEA-007** (registry snapshot) | `contracts/apr-registry-snapshot-v1.yaml` | ACTIVE | `FALSIFY-SNAPSHOT-001` → `crates/aprender-registry/tests/falsify_snapshot_001.rs::snapshot_yields_bit_identical_query_results`<br>`FALSIFY-SNAPSHOT-002` → `crates/aprender-registry/tests/falsify_snapshot_002.rs::snapshot_does_not_block_concurrent_writers`<br>`FALSIFY-SNAPSHOT-003` → `crates/aprender-registry/tests/falsify_snapshot_003.rs::snapshot_refuses_to_overwrite_existing_file` | `crates/aprender-contracts/tests/apr_registry_snapshot_contract.rs` (6 assertions) |
| **HELIX-IDEA-009** (API key auth) | `contracts/apr-serve-api-key-auth-v1.yaml` | ACTIVE | `FALSIFY-AUTH-001` → `crates/apr-cli/tests/falsify_auth_001.rs::missing_bearer_returns_401_on_every_route`<br>`FALSIFY-AUTH-002` → `crates/apr-cli/tests/falsify_auth_002.rs::valid_bearer_passes_and_hash_path_is_constant_time`<br>`FALSIFY-AUTH-003` → `crates/apr-cli/tests/falsify_auth_003.rs::auth_module_uses_subtle_constanttimeeq` | `crates/aprender-contracts/tests/apr_serve_api_key_auth_contract.rs` (6 assertions) |

**Audit reproduction:** `pv validate contracts/apr-{mcp-tool-inventory,registry-snapshot,serve-api-key-auth}-v1.yaml`
returns `Contract is valid.` on each. `cargo test -p aprender-contracts
--test apr_mcp_tool_inventory_contract --test apr_registry_snapshot_contract
--test apr_serve_api_key_auth_contract` produces 18 passed; 0 failed.

#### Forward obligations

Every future HELIX-IDEA implementation MUST follow the same chain.
Three of the four recommended-unshipped ideas already have
pre-authored gate IDs in their §2.x bodies — the implementation PR
transcribes them into the YAML's `falsification_conditions:` list
verbatim:

| Idea | Contract YAML | Status | Pre-authored gates |
|---|---|---|---|
| HELIX-IDEA-001 (Persistent HNSW) | `contracts/apr-hnsw-persistence-v1.yaml` | **v1.1.0 ACTIVE — Phases 1-2 (gates 001/002) shipped; Phases 3-4 (gates 003/004) pending amendment** | §2.1: 4 gates (`FALSIFY-HNSW-PERSIST-001..004`) |
| HELIX-IDEA-005 (BM25 + dense) | `contracts/apr-hybrid-retrieval-v1.yaml` | To author | §2.5: 4 gates (`FALSIFY-HYBRID-001..004`) |
| HELIX-IDEA-006 (Reranking) | `contracts/apr-rerank-v1.yaml` | To author | §2.6: 6 gates (`FALSIFY-RERANK-RRF-001/002`, `MMR-001/002`, `XENC-001/002`) |
| HELIX-IDEA-008 (Schema migration) | `contracts/apr-schema-migration-v1.yaml` | To author | Not yet pre-authored — speculative pending concrete pain point (§2.8) |

A PR that merges code without authoring its YAML, or authors a YAML
without the integration test, or alters the live registry without
updating the spec's §6 falsification log, MUST be rejected at review
— the contract chain is the audit trail.

## 2. Proposals

---

### 2.1 HELIX-IDEA-001 — Persistent on-disk HNSW

**Status:** **Shipped (Phases 1-2)**; FALSIFY-HNSW-PERSIST-001
(round-trip identity) and FALSIFY-HNSW-PERSIST-002 (atomic-write
crash safety) ENFORCED. Phases 3 (recall threshold, gate 003) and 4
(cold-open latency, gate 004) ship as separate PRs amending the
contract. See §1.4 forward obligations.
**Contract:** `contracts/apr-hnsw-persistence-v1.yaml` v1.1.0 (ACTIVE).
**Effort:** Medium total; Phases 1-2 fit in two commits each ~150 LOC + tests.
**Target crate:** `aprender-core` (extended `index/`).

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

**Pre-authored falsification gates** (for `contracts/apr-hnsw-persistence-v1.yaml`).

| Gate ID | Property | Test target | Phase |
|---|---|---|---|
| `FALSIFY-HNSW-PERSIST-001` | Insert→close→reopen→query yields exactly the same `Vec<(id, score)>` top-k as the same operations against the in-memory `Hnsw`. Falsifies "persistence loses or reorders neighbours". | `crates/aprender-core/tests/falsify_hnsw_persist_001.rs::reopen_top_k_matches_in_memory` | **Phase 1 (SHIPPED)** |
| `FALSIFY-HNSW-PERSIST-002` | A `flush()` followed by process kill (simulated via `Drop` without `flush`) yields a file that opens cleanly OR errors with a recovery-required diagnostic — never silently returns truncated results. Falsifies "crash mid-write produces a usable-looking but lying index". | `crates/aprender-core/tests/falsify_hnsw_persist_002.rs::partial_write_does_not_silently_corrupt` | **Phase 2 (SHIPPED)** |
| `FALSIFY-HNSW-PERSIST-003` | Recall@10 against a 10⁵-vector golden corpus is ≥ 0.95 vs. exact (brute-force) baseline. Tunable corpus path via `APR_HNSW_BENCH_CORPUS` for the 1M × 768-dim production target; CI uses a smaller fixture so the gate runs in under a minute. Falsifies "persistence layer subtly degraded recall". | `crates/aprender-core/tests/falsify_hnsw_persist_003.rs::recall_at_10_meets_threshold` | Phase 3 |
| `FALSIFY-HNSW-PERSIST-004` | Cold-start open + first query latency is < 500 ms on the CI fixture; production-size budget tunable via `APR_HNSW_OPEN_BUDGET_MS`. Falsifies "open() rebuilds the graph eagerly". | `crates/aprender-core/tests/falsify_hnsw_persist_004.rs::cold_open_first_query_within_budget` | Phase 4 |

**Phase 1 implementation deltas vs original sketch.**
- **Substrate choice:** neither Arrow IPC (Option A) nor `redb` (Option B)
  was needed for Phase 1. The `HNSWIndex` type already has all serializable
  fields; adding `#[derive(Serialize, Deserialize)]` plus `#[serde(skip)]`
  on its `ThreadRng` field gave a complete bincode round-trip. Phase 4
  may revisit the substrate choice if cold-open latency demands mmap.
- **Determinism:** the original sketch's "rebuild on open" semantics
  would have failed under the random layer assignment in
  `HNSWIndex::add()`. Phase 1 sidesteps this by serializing the *whole
  graph* (nodes + connections + entry_point), so reopen is byte-stable
  against the original index. The "rebuild from raw vectors" path is
  not part of the contract — and may never be needed.
- **WAL deferred:** Phase 1 ships overwrite-on-flush. Crash mid-write
  can leave a truncated file; Gate 002 (Phase 2) introduces fsync +
  atomic rename to surface partial writes as a clean error.

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

**Pre-authored falsification gates** (for `contracts/apr-hybrid-retrieval-v1.yaml`).

| Gate ID | Property | Test target |
|---|---|---|
| `FALSIFY-HYBRID-001` | Hybrid `recall@10 ≥ max(dense, sparse) + 0.05` on a frozen BEIR subset (NFCorpus or SciFact — small enough for CI). Tunable corpus path via `APR_BEIR_CORPUS`. Falsifies "hybrid is statistically equivalent to one of the legs". | `crates/aprender-retrieve/tests/falsify_hybrid_001.rs::hybrid_beats_max_of_legs_by_5pts` |
| `FALSIFY-HYBRID-002` | The `Retriever::hybrid(weights)` API is *score-equivalent* to a manual `combine(dense(q), sparse(q), weights)` callsite — i.e., the trait method does not silently change weighting compared to the documented arithmetic. Falsifies "the trait re-normalizes scores in a way callers don't expect". | `crates/aprender-retrieve/tests/falsify_hybrid_002.rs::trait_method_matches_explicit_combine` |
| `FALSIFY-HYBRID-003` | Tokenization for BM25 indexing comes from the **same** tokenizer used by `apr serve` inference (no separate BM25-only tokenizer). Tested via a structural assertion that the BM25 indexer's tokenizer trait object's type-id equals the inference path's. Falsifies "BM25 quietly forks tokenization". | `crates/aprender-retrieve/tests/falsify_hybrid_003.rs::bm25_uses_inference_tokenizer` |
| `FALSIFY-HYBRID-004` | BM25 index build for a 100k-doc fixture completes within 12 s on commodity hardware (extrapolates to <2 min for 1M docs at the same per-doc cost). Tunable via `APR_BM25_BUILD_BUDGET_MS`. Falsifies "indexing is super-linear in corpus size". | `crates/aprender-retrieve/tests/falsify_hybrid_004.rs::index_build_within_budget` |

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

**Pre-authored falsification gates** (for `contracts/apr-rerank-v1.yaml`).

| Gate ID | Property | Test target |
|---|---|---|
| `FALSIFY-RERANK-RRF-001` | RRF over hybrid retrieval (HELIX-IDEA-005) yields ≥3-point nDCG@10 improvement vs. either single retriever on a frozen BEIR subset. Falsifies "RRF is a wash on this corpus". | `crates/aprender-rerank/tests/falsify_rerank_rrf_001.rs::rrf_beats_single_retriever_ndcg10` |
| `FALSIFY-RERANK-RRF-002` | RRF score combination is *order-independent* in input list ordering — `rrf(a, b) == rrf(b, a)` byte-for-byte. Falsifies "RRF accidentally weights one input more than another". | `crates/aprender-rerank/tests/falsify_rerank_rrf_002.rs::rrf_is_input_order_invariant` |
| `FALSIFY-RERANK-MMR-001` | MMR with `λ=0.5` reduces top-k centroid distance (a diversity proxy) by ≥10% vs. unranked top-k while keeping recall@10 within 1 point. Falsifies "MMR trades recall for nothing measurable". | `crates/aprender-rerank/tests/falsify_rerank_mmr_001.rs::mmr_increases_diversity_within_recall_budget` |
| `FALSIFY-RERANK-MMR-002` | MMR with `λ=1.0` (no diversity weight) returns the same top-k as the input scorer. Falsifies "the diversity formula leaks even at λ=1". | `crates/aprender-rerank/tests/falsify_rerank_mmr_002.rs::mmr_lambda_one_is_identity` |
| `FALSIFY-RERANK-XENC-001` | Cross-encoder rerank for 100 candidates completes within 100 ms on a ≤100M-param model. Tunable via `APR_RERANK_BUDGET_MS`. Falsifies "the cross-encoder path is too slow for the production budget". | `crates/aprender-rerank/tests/falsify_rerank_xenc_001.rs::cross_encoder_top_100_within_budget` |
| `FALSIFY-RERANK-XENC-002` | Cross-encoder inference goes through `aprender-serve` (no parallel inference stack inside `aprender-rerank`). Structural source-grep gate, similar to `FALSIFY-AUTH-003`. Falsifies "the rerank crate quietly forked the inference engine". | `crates/aprender-rerank/tests/falsify_rerank_xenc_002.rs::cross_encoder_uses_aprender_serve` |

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
- **Quality gates.** Each accepted proposal MUST satisfy the
  Design-by-Provable-Contract chain in §1.4 (proposal → YAML →
  ENFORCED falsifiers → `aprender-contracts` integration test).
  Standard project-wide gates also apply: ≥95% line coverage on the
  new code, `cargo clippy -- -D warnings`, and a fuzz target where
  input is untrusted (the HNSW load path in HELIX-IDEA-001 qualifies;
  the auth header parser in HELIX-IDEA-009 also qualifies but is
  small enough that proptest in `auth.rs::tests` was deemed
  sufficient — see PR #1605). The §1.4 chain is load-bearing: a
  PR without an authored YAML is rejected at review even if all
  other gates pass.
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
