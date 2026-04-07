# Section 32: PMAT Infrastructure Contracts

> Parent: [pv-spec.md](../pv-spec.md) §32

## 32.1 Motivation

PMAT has 271+ scored provable contracts covering math, ML, GPU, infrastructure, and model-specific
domains across the sovereign stack repos. The **pmat binary itself** now has 15 contracts covering
all 9 infrastructure domains below (TDG scoring, comply check, composite score, context generation,
CLI interface, MCP protocol, graph/index, concurrency, tracing, memory, state machine, config,
compression, work lifecycle, and work DBC).

These are the domains where bugs actually bite users: silent wrong output, MCP protocol violations
that confuse LLM agents, index corruption that requires rebuild, and state machine transitions that
lose work.

## 32.2 Domain Inventory

| ID | Domain | Priority | Contracts | Surface |
|----|--------|----------|-----------|---------|
| PMAT-INF-1 | CLI/HTTP Interface | P0 | cli-interface-v1 | User-facing boundary |
| PMAT-INF-2 | MCP Protocol | P0 | mcp-protocol-v1 | Agent-facing boundary |
| PMAT-INF-3 | Graph/Index | P0 | graph-index-v1 | Core infrastructure |
| PMAT-INF-4 | Concurrency | P1 | concurrency-safety-v1 | Correctness-critical |
| PMAT-INF-5 | Tracing/Observability | P1 | tracing-observability-v1 | Production debugging |
| PMAT-INF-6 | Memory Management | P1 | memory-safety-v1 | WASM/arena safety |
| PMAT-INF-7 | State Machine | P2 | state-machine-v1 | Workflow correctness |
| PMAT-INF-8 | Configuration | P2 | configuration-schema-v1 | Silent misconfiguration |
| PMAT-INF-9 | Compression | P3 | compression-roundtrip-v1 | Index serialization |

## 32.3 CLI/HTTP Interface (cli-interface-v1)

### Equations

**exit_code_semantics**: Every CLI invocation returns a deterministic exit code.

```
exit_code: (Command, Result<AnalysisOutput, PmatError>) -> u8
  0  = success (analysis completed, no violations)
  1  = analysis violation (quality gate failed, threshold exceeded)
  2  = configuration error (invalid args, missing file, bad TOML)
  3  = internal error (panic, OOM, unexpected state)
```

**output_format_fidelity**: OutputFormat enum produces valid, parseable output.

```
render: (AnalysisOutput, OutputFormat) -> String
  Json      => serde_json::from_str(output).is_ok()
  Csv       => csv::Reader::from_reader(output).records().all(|r| r.is_ok())
  Junit     => quick_xml::de::from_str::<TestSuites>(output).is_ok()
  Yaml      => serde_yaml::from_str(output).is_ok()
  Markdown  => output.starts_with('#') || output.starts_with('|')
  Table     => output.lines().count() >= 1
```

**timeout_honoring**: Analysis respects `--timeout` flag.

```
timeout: (Command, Duration) -> Result<PartialOutput, TimeoutError>
  wall_clock(analysis) <= timeout + epsilon
  Where epsilon = 1s (cleanup grace period)
```

**result_cardinality**: `--top-files N` produces at most N entries.

```
top_files: (AnalysisOutput, N: usize) -> Vec<Entry>
  output.len() <= N
  output.len() <= total_available
```

### Proof Obligations

| Type | Property | Formal |
|------|----------|--------|
| completeness | Exit code covers all outcomes | ∀ (cmd, result): exit_code(cmd, result) ∈ {0,1,2,3} |
| determinism | Same input → same exit code | exit_code(cmd, r₁) = exit_code(cmd, r₂) when r₁ = r₂ |
| roundtrip | JSON output is parseable | ∀ output where format=Json: parse(render(output)) = output |
| bound | Result cardinality | ∀ output, N: \|top_files(output, N)\| ≤ N |
| postcondition | Timeout honored | wall_clock ≤ timeout + 1s |

### Falsification Tests

- **FALSIFY-CLI-001**: Zero-result analysis returns exit code 0, not 1
- **FALSIFY-CLI-002**: Invalid path returns exit code 2, not panic
- **FALSIFY-CLI-003**: JSON output with unicode identifiers remains valid JSON
- **FALSIFY-CLI-004**: CSV output with commas in function names uses proper quoting
- **FALSIFY-CLI-005**: `--top-files 0` returns empty result, not all results
- **FALSIFY-CLI-006**: Timeout of 1s on large project returns partial result, not hang

## 32.4 MCP Protocol (mcp-protocol-v1)

### Equations

**tool_schema_fidelity**: Every tool's `inputSchema` matches what the handler accepts.

```
schema_match: (ToolDefinition, HandlerFn) -> bool
  ∀ field in schema.required: handler.accepts(field)
  ∀ field in handler.params: field ∈ schema.properties
```

**session_lifecycle**: MCP session follows protocol state machine.

```
session: State × Method -> State
  Uninitialized × initialize -> Initialized
  Initialized   × tools/list -> Initialized
  Initialized   × tools/call -> Initialized
  Initialized   × shutdown   -> Closed
  Uninitialized × tools/call -> Error (must initialize first)
  Closed        × *          -> Error (session closed)
```

**error_mapping_lossless**: PmatError → McpError preserves diagnostic information.

```
map_error: PmatError -> McpError
  FileNotFound(p) -> McpError { code: -32602, message: contains(p.display()) }
  AnalysisError(e) -> McpError { code: -32603, message: contains(e) }
  No lossy downcast: message.len() >= original_error.len()
```

**idempotency**: Read-only tools are idempotent.

```
idempotent: Tool -> bool
  analyze_*     => true  (read-only analysis)
  quality_gate  => true  (read-only check)
  refactor_*    => false (mutates state)
  tools/call(t, params) = tools/call(t, params) when idempotent(t)
```

### Proof Obligations

| Type | Property | Formal |
|------|----------|--------|
| completeness | Schema covers all handler params | ∀ tool: schema(tool) ⊇ handler_params(tool) |
| state_machine | Session lifecycle valid | No tools/call before initialize |
| conservation | Error info preserved | len(mcp_error.message) ≥ len(pmat_error.to_string()) |
| idempotency | Read tools are pure | f(x) = f(x) for all read-only tools |
| soundness | No phantom tools | ∀ tool in tools/list: handler(tool) exists |

### Falsification Tests

- **FALSIFY-MCP-001**: Tool with required field missing returns -32602, not panic
- **FALSIFY-MCP-002**: tools/call before initialize returns protocol error
- **FALSIFY-MCP-003**: Unknown tool name returns -32601 (method not found)
- **FALSIFY-MCP-004**: Concurrent tools/call on same project produces consistent results
- **FALSIFY-MCP-005**: Tool schema `inputSchema` JSON validates against JSON Schema Draft 7

## 32.5 Graph/Index (graph-index-v1)

### Equations

**csr_construction**: CSR graph node count equals node map size.

```
csr_invariant: CSRGraph -> bool
  num_nodes() = node_map.len()  (NOT graph.num_nodes())
  ∀ edge (u,v): u ∈ node_map ∧ v ∈ node_map
```

**pagerank_convergence**: PageRank produces valid probability distribution.

```
pagerank: CSRGraph -> Vec<f64>
  sum(ranks) = 1.0 ± 1e-6
  ∀ rank: rank >= 0.0
  terminates in ≤ max_iterations
  |ranks[i]_t - ranks[i]_{t-1}| < epsilon for convergence
```

**fts5_consistency**: Insert-then-search returns the inserted document.

```
fts5_roundtrip: (DB, Doc) -> bool
  insert(db, doc)
  results = search(db, doc.content)
  doc ∈ results
```

**sqlite_roundtrip**: save() then load() is identity.

```
roundtrip: AgentContextIndex -> bool
  load(save(index)) ≅ index
  Where ≅ ignores: field ordering, derived indices (rebuilt on load)
  Preserves: function entries, quality metrics, source code, call graph
```

**bm25_scoring**: BM25 scores are non-negative and monotonic in relevance.

```
bm25: (Query, Doc) -> f64
  score >= 0.0
  tf(term, doc₁) > tf(term, doc₂) => bm25(query, doc₁) >= bm25(query, doc₂)
    (when doc lengths are equal and query is single-term)
```

### Proof Obligations

| Type | Property | Formal |
|------|----------|--------|
| invariant | CSR node count | num_nodes() = node_map.len() always |
| conservation | PageRank sums to 1 | \|Σ ranks - 1.0\| < 1e-6 |
| bound | PageRank non-negative | ∀ i: ranks[i] ≥ 0.0 |
| termination | PageRank converges | loop terminates within max_iterations |
| roundtrip | SQLite save/load identity | load(save(idx)).functions = idx.functions |
| monotonicity | BM25 relevance ordering | Higher TF → higher score (ceteris paribus) |

### Falsification Tests

- **FALSIFY-IDX-001**: Empty graph has PageRank sum = 0 (not NaN)
- **FALSIFY-IDX-002**: Single-node graph has PageRank = [1.0]
- **FALSIFY-IDX-003**: FTS5 search for exact function name returns that function
- **FALSIFY-IDX-004**: SQLite roundtrip preserves function count exactly
- **FALSIFY-IDX-005**: SQLite roundtrip preserves TDG scores within f64 epsilon
- **FALSIFY-IDX-006**: BM25 score for absent term is 0.0, not negative
- **FALSIFY-IDX-007**: CSR graph with edges between N nodes reports num_nodes() = N

## 32.6 Concurrency Safety (concurrency-safety-v1)

### Equations

**channel_lossless**: Bounded channels never silently drop messages.

```
channel: (Sender, Receiver, Bound) -> bool
  sent_count = received_count + pending_count
  No message is lost unless sender is explicitly dropped
```

**task_cancellation_cleanup**: Cancelled async tasks release all resources.

```
cancel: Task -> ResourceSet
  ∀ resource in task.acquired: resource.is_released() after cancel
  No leaked file handles, no leaked tempfiles
```

**parallel_determinism**: Parallel analysis produces same results as sequential.

```
parallel: (Files, Analyzer) -> Vec<Result>
  sort(parallel_analyze(files)) = sort(sequential_analyze(files))
```

### Proof Obligations

| Type | Property | Formal |
|------|----------|--------|
| conservation | Channel lossless | sent = received + pending |
| frame | Cancellation cleanup | modifies(task.state), preserves(all resources released) |
| determinism | Parallel = sequential | parallel_result = sequential_result (modulo ordering) |

### Falsification Tests

- **FALSIFY-CONC-001**: 1000 concurrent queries produce consistent function counts
- **FALSIFY-CONC-002**: Cancelled analysis does not leave `.pmat/*.tmp` files
- **FALSIFY-CONC-003**: Rayon parallel map over 10K files = sequential map (sorted)

## 32.7 Tracing/Observability (tracing-observability-v1)

### Equations

**span_parentage**: Child spans always reference valid parent.

```
span_tree: Vec<Span> -> bool
  ∀ span: span.parent_id = None ∨ span.parent_id ∈ active_spans
  root_spans.count() >= 1
```

**metric_monotonicity**: Counters are monotonically non-decreasing.

```
counter: (Metric, t₁, t₂) -> bool
  t₁ < t₂ => counter(t₁) <= counter(t₂)
```

**renacer_backward_compat**: Golden trace format is parseable across versions.

```
trace_compat: (TraceV_old, ParserV_new) -> bool
  parse(serialize(trace)) = trace for all version pairs where major matches
```

### Proof Obligations

| Type | Property | Formal |
|------|----------|--------|
| invariant | Span tree is valid tree | No cycles, no orphan children |
| monotonicity | Counter monotonic | counter(t₁) ≤ counter(t₂) when t₁ < t₂ |
| roundtrip | Trace format backward compat | parse(serialize(trace)) = trace |

### Falsification Tests

- **FALSIFY-TRACE-001**: Nested analysis creates proper span hierarchy
- **FALSIFY-TRACE-002**: Error in child span does not corrupt parent span
- **FALSIFY-TRACE-003**: Renacer trace from v1 parseable by v2 parser

## 32.8 Memory Management (memory-safety-v1)

### Equations

**lru_eviction_correctness**: Evicted entries are fully freed.

```
lru: (Cache, Capacity) -> bool
  cache.len() <= capacity always
  evicted entries have refcount = 0 (no dangling references)
```

**arena_lifecycle**: Allocated objects do not outlive arena.

```
arena: Arena -> bool
  ∀ obj in arena.allocated: obj.lifetime ⊆ arena.lifetime
  drop(arena) => all objects freed
```

**index_memory_budget**: Index loading respects memory bounds.

```
load_index: (Path, Budget) -> Result<Index, OOM>
  peak_memory(load) <= budget
  If exceeds: returns Err(OOM), does not panic
```

### Proof Obligations

| Type | Property | Formal |
|------|----------|--------|
| bound | LRU capacity | cache.len() ≤ capacity |
| frame | Arena lifetime | drop(arena) frees all allocations |
| bound | Memory budget | peak_memory ≤ budget |

### Falsification Tests

- **FALSIFY-MEM-001**: LRU cache at capacity evicts LRU entry on insert
- **FALSIFY-MEM-002**: Loading 50MB index with 40MB budget returns error, not OOM kill
- **FALSIFY-MEM-003**: Arena drop releases all temp strings (no leak under valgrind/dhat)

## 32.9 State Machine (state-machine-v1)

### Equations

**refactor_transitions**: Only valid state transitions are allowed.

```
transition: (State, Event) -> Result<State, InvalidTransition>
  Valid edges:
    Scan → Analyze → Plan → Refactor → Test → Lint → Emit → Complete
  No skip: Scan → Plan is INVALID
  No backward: Refactor → Scan is INVALID
```

**event_store_append_only**: Past events are never mutated.

```
append_only: EventStore -> bool
  ∀ event at index i: event_i is immutable after insert
  replay(events[0..n]) = state_n
```

**snapshot_recovery**: Restore from snapshot + replay = fresh build.

```
recovery: (Snapshot, MissedEvents) -> State
  restore(snapshot) + replay(missed) = build_from_scratch(all_events)
```

### Proof Obligations

| Type | Property | Formal |
|------|----------|--------|
| state_machine | Valid transitions only | No skip, no backward edges |
| invariant | Append-only event store | events[0..n] immutable after write |
| equivalence | Snapshot recovery | restore + replay ≅ fresh build |

### Falsification Tests

- **FALSIFY-SM-001**: Scan→Plan transition returns InvalidTransition error
- **FALSIFY-SM-002**: Refactor→Scan backward transition returns error
- **FALSIFY-SM-003**: Event replay from empty store produces initial state
- **FALSIFY-SM-004**: Snapshot restore + 0 missed events = snapshot state exactly

---

*Continued in [pmat-infrastructure-contracts-2.md](pmat-infrastructure-contracts-2.md) (sections 32.10-32.14).*
