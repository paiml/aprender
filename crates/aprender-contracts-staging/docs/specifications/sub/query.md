# Sub-spec: Query Engine

**Parent:** [pv-spec.md](../pv-spec.md) Section 8

---

## 1. Design Goals

`pv query` provides instant lookup across all contracts AND their
consumer projects across the stack. Design goals:

1. **Sub-second response** for 165+ contracts (O(1) via index)
2. **Semantic search** — find contracts by intent, not exact name
3. **Structured filters** — obligation type, proof level, score
4. **Graph-aware** — dependency traversal
5. **Actionable output** — show gaps, not just matches
6. **Compatible with `pmat query`** — same UX patterns
7. **Automatic cross-project** — discover call sites, bindings, and
   violations across all consumer projects without configuration

---

## 2. Architecture

### Index Design

Modeled on `pmat query`'s multi-index hybrid approach:

```
ContractIndex
+-- entries: Vec<ContractEntry>           All contracts
+-- name_index: HashMap<stem, Vec<idx>>   Contract name -> entries
+-- equation_index: HashMap<eq, Vec<idx>> Equation name -> entries
+-- obligation_index: HashMap<type, Vec<idx>>  By obligation type
+-- formula_corpus: Vec<String>           For BM25 scoring
+-- formula_corpus_lower: Vec<String>     Pre-lowercased
+-- score_cache: HashMap<stem, Score>     Pre-computed scores
+-- dep_graph: DependencyGraph            Contract DAG
```

### Why Not SQLite FTS5?

`pmat query` indexes 42K+ functions and needs FTS5 for scale. Contract
indexes have 165 entries. In-memory BM25 over a pre-computed corpus is
sufficient and avoids the SQLite dependency. If contract count exceeds
~1000, migrate to FTS5.

### Cross-Project Index

`pv query` automatically discovers sibling project directories and
indexes contract usage across the entire stack.

```
CrossProjectIndex
+-- projects: Vec<ProjectEntry>       Discovered sibling projects
+-- call_sites: HashMap<stem, Vec<CallSite>>    #[contract] annotations
+-- binding_refs: HashMap<stem, Vec<BindingRef>> binding.yaml entries
+-- kaizen_refs: HashMap<stem, Vec<KaizenRef>>   KAIZEN ticket references
+-- commit_refs: HashMap<stem, Vec<CommitRef>>   Git commit message refs
```

**Discovery algorithm:**

```
1. Start from provable-contracts repo root
2. Walk parent directory (../) for sibling projects
3. For each sibling with Cargo.toml:
   a. Check for provable-contracts dependency (Cargo dep or YAML refs)
   b. Scan .rs files for #[contract("...")] annotations (ripgrep)
   c. Parse binding.yaml if present (contracts/<project>/binding.yaml)
   d. Scan for KAIZEN-NNN / C-*-NNN patterns in code + commits
4. Cache results in .pv/cross-project.idx
5. Auto-rebuild when any sibling project mtime changes
```

**Known project locations (auto-discovered):**

| Project | Path | Contract Signals |
|---|---|---|
| aprender | `../aprender` | `#[contract]` annotations, `binding.yaml` |
| trueno | `../trueno` | KAIZEN refs in code/commits |
| entrenar | `../entrenar` | KAIZEN refs in code/commits |
| bashrs | `../bashrs` | Contract YAML refs |

**Override via config or CLI:**

```bash
pv query "softmax" --include-project ../custom-project
pv query "softmax" --project aprender   # Filter to one project
pv query --all-projects                 # Force full cross-project scan
```

### Index Persistence [IMPLEMENTED]

```
.pv/contracts.idx              JSON-serialized ContractIndex [IMPLEMENTED]
.pv/contracts.idx.mtime        mtime of contracts/ at index build [IMPLEMENTED]
.pv/cross-project.idx          JSON-serialized CrossProjectIndex
.pv/cross-project.idx.mtime    max mtime of all sibling projects
```

Auto-rebuild when `contracts/` mtime > stored mtime. [IMPLEMENTED]
Cross-project index persistence deferred to Phase 3.

---

## 3. Search Modes

### Semantic (default)

BM25 ranking over a concatenated corpus of: contract description,
equation formulas, obligation properties, enforcement rules, paper
references.

```bash
pv query "numerical stability softmax"
```

**Tokenization:**
```
query_terms = split_on_non_alphanumeric(query)
            |> lowercase
            |> filter(|t| t.len() >= 2)
```

**BM25 scoring:**
```
score(doc, query) = sum(
    IDF(term) * (tf(term, doc) * (k1 + 1))
    / (tf(term, doc) + k1 * (1 - b + b * |doc| / avgdl))
    for term in query_terms
)
where k1 = 1.2, b = 0.75
```

### Regex

Pattern match against all string fields in each contract.

```bash
pv query --regex "SM-INV-\d+"
pv query --regex "softmax|log_softmax"
```

### Literal

Exact substring match (case-insensitive by default).

```bash
pv query --literal "kani::proof"
pv query --literal "Zhang & Sennrich" --case-sensitive
```

---

## 4. Filters

Filters are applied after search, before ranking.

| Filter | Type | Example | Status |
|---|---|---|---|
| `--obligation <type>` | Enum | `--obligation invariant` | [IMPLEMENTED] |
| `--min-score <f64>` | Threshold | `--min-score 0.8` | [IMPLEMENTED] |
| `--min-level <L1-L5>` | Threshold | `--min-level L4` | [IMPLEMENTED] |
| `--depends-on <stem>` | DAG traversal | `--depends-on softmax-kernel-v1` | [IMPLEMENTED] |
| `--depended-by <stem>` | Reverse DAG | `--depended-by attention-kernel-v1` | [IMPLEMENTED] |
| `--unproven` | Boolean | Shows obligations at L2 or below | [IMPLEMENTED] |
| `--binding-gaps` | Boolean | Shows not_implemented bindings | [IMPLEMENTED] |
| `--project <name>` | String | `--project aprender` | [IMPLEMENTED] |
| `--include-project <path>` | Path | `--include-project ../custom` | [IMPLEMENTED] |
| `--all-projects` | Boolean | Force full cross-project scan | [IMPLEMENTED] |
| `--rebuild-index` | Boolean | Force index rebuild | [IMPLEMENTED] |
| `--tier <n>` | Enum | `--tier 1` | [IMPLEMENTED] |
| `--class <A-E>` | Enum | `--class A` | [IMPLEMENTED] |
| `--kind <kind>` | Enum | `--kind registry` | [IMPLEMENTED] |

**`--kind <kernel\|registry\|model-family\|pattern\|schema>`** filters by
`metadata.kind` (see §3 Contract Schema). Use to find all registries
(`--kind registry`), all cross-cutting patterns (`--kind pattern`), or
all architecture metadata schemas (`--kind model-family`). Non-kernel
kinds are tagged in the result output as `[kind]` next to the stem.

---

## 5. Enrichment Flags

Enrichment flags add metadata to search results without changing ranking.

### --score [IMPLEMENTED]

Show contract score inline with results.

```
[1] softmax-kernel-v1.yaml (relevance: 0.95)
    Score: 0.79 (Grade B)
    Spec: 0.92 | Falsify: 0.88 | Kani: 0.75 | Lean: 0.00 | Bind: 1.00
```

### --proof-status [IMPLEMENTED]

Show L1-L5 breakdown per result.

```
[1] softmax-kernel-v1.yaml
    Proof Level: L4
    L1: 7/7 | L2: 5/7 | L3: 5/7 | L4: 3/7 | L5: 0/7
```

### --binding-info [IMPLEMENTED]

Show binding status per equation.

```
[1] softmax-kernel-v1.yaml
    Equations:
      softmax:     implemented (aprender::nn::functional::softmax)
      log_softmax: implemented (aprender::nn::functional::log_softmax)
```

### --graph [IMPLEMENTED]

Show dependency context.

```
[1] attention-kernel-v1.yaml
    Depends on:  softmax-kernel-v1, matmul-kernel-v1
    Depended by: flash-attention-v1
```

### --paper [IMPLEMENTED]

Show paper references.

```
[1] softmax-kernel-v1.yaml
    Papers:
      - Bridle (1990). Probabilistic Interpretation of Feedforward Networks
      - Goodfellow et al. (2016). Deep Learning Ch. 6.2.2
```

### --diff [IMPLEMENTED]

Show recent contract changes (requires git).

```
[1] softmax-kernel-v1.yaml
    Last modified: 2026-02-19 (14 days ago)
    Changes: +2 falsification tests, +1 kani harness
```

### --call-sites [IMPLEMENTED]

Show where the contract is referenced across consumer projects.

```
[1] softmax-kernel-v1.yaml
    Call sites (3 projects):
      aprender/src/nn/functional.rs:42   #[contract("softmax-kernel-v1", eq="softmax")]
      aprender/src/nn/functional.rs:87   #[contract("softmax-kernel-v1", eq="log_softmax")]
      trueno/src/kernels/softmax.rs:15   // KAIZEN-050: fused softmax backward
      entrenar/src/loss.rs:33            // C-XENT-002: refs softmax-kernel-v1
```

### --violations [IMPLEMENTED]

Show contracts whose obligations are violated in consumer code (e.g.,
unchecked invariants, missing tests, binding gaps).

```
[1] matmul-kernel-v1.yaml
    Violations:
      aprender: 3/7 obligations unproven (MM-EQV-002, MM-BND-001, MM-BND-002)
      trueno:   missing SIMD equivalence test for AVX-512 path
```

### --coverage-map [IMPLEMENTED]

Show cross-project contract coverage matrix.

```
Contract                    aprender  trueno  entrenar  bashrs
softmax-kernel-v1           ██████    ████    ██        --
rmsnorm-kernel-v1           ██████    ████    ████      --
attention-kernel-v1         ████      ██      --        --
encoder-forward-v1          --        --      --        ████
```

---

## 6. Output Formats

### Text (default)

```
[1] softmax-kernel-v1.yaml (score: 0.79, grade: B)
    Equations: softmax, log_softmax
    Obligations: 7 (5 proven, 2 L3-only)
    Paper: Bridle 1990; Goodfellow 2016
    ---

[2] cross-entropy-kernel-v1.yaml (score: 0.82, grade: B)
    Equations: cross_entropy
    Obligations: 4 (3 proven, 1 L3-only)
    Paper: Goodfellow 2016
    ---
```

### JSON

```json
{
  "results": [
    {
      "stem": "softmax-kernel-v1",
      "path": "contracts/softmax-kernel-v1.yaml",
      "relevance": 0.95,
      "score": { "composite": 0.79, "grade": "B", ... },
      "equations": ["softmax", "log_softmax"],
      "obligations": { "total": 7, "proven": 5 },
      "papers": ["Bridle 1990", "Goodfellow 2016"]
    }
  ],
  "total_matches": 3,
  "query": "softmax numerical stability"
}
```

### Markdown

```markdown
## Query: "softmax numerical stability"

### 1. softmax-kernel-v1.yaml

- **Score:** 0.79 (Grade B)
- **Equations:** softmax, log_softmax
- **Obligations:** 7 (5 proven, 2 L3-only)
- **Papers:** Bridle 1990; Goodfellow 2016
```

---

## 7. Comparison with `pmat query`

| Feature | `pmat query` | `pv query` |
|---|---|---|
| Index target | Functions (42K+) | Contracts (165+) + consumer projects |
| Index backend | SQLite FTS5 | In-memory BM25 |
| Ranking | BM25 + PageRank | BM25 + dependency DAG |
| Quality metric | TDG grade | Contract score (A-F) |
| Graph metric | Call graph PageRank | Contract dependency PageRank |
| Cross-project | `--include-project` flag | **Automatic** sibling discovery |
| Enrichment: churn | Git volatility | Contract diff/drift |
| Enrichment: faults | Batuta fault patterns | Unproven obligations |
| Enrichment: coverage | Line coverage | Proof level coverage |
| Enrichment: call sites | N/A | `--call-sites` across stack |
| Output formats | text/json/markdown | text/json/markdown |

### Shared UX Patterns

- Same flag style: `--min-score`, `--limit`, `-f json`
- Same search modes: semantic, regex, literal
- Same enrichment model: flags add metadata to results
- Same ranking: relevance is default, graph metrics are opt-in

### Key Differentiator

`pmat query` requires `--include-project` to search across repos.
`pv query` discovers consumer projects **automatically** because it
knows the contract graph — every contract declares its consumers via
binding.yaml and KAIZEN refs. Cross-project is the default, not an
opt-in flag.

---

## 8. Implementation Plan

### Phase 1: Core Index + Semantic Search [DONE]

- Parse all YAML contracts into ContractIndex
- BM25 ranking over corpus
- Name/equation O(1) lookup
- Basic text output

### Phase 2: Filters + Enrichment [DONE]

- Obligation type, score, level, min-score, min-level filters
- Score, proof-status, binding, graph, paper, diff enrichment
- JSON + markdown output formats
- Score cache for O(1) --min-score filtering

### Phase 3: Cross-Project Search [DONE]

- Auto-discover sibling projects via `../` [IMPLEMENTED]
- Scan for `#[contract]` annotations (grep) [IMPLEMENTED]
- Parse consumer binding.yaml files [IMPLEMENTED]
- Scan for KAIZEN/contract ID patterns in code [IMPLEMENTED]
- `--call-sites` enrichment [IMPLEMENTED]
- `--violations` enrichment [IMPLEMENTED]
- `--coverage-map` enrichment [IMPLEMENTED]
- CrossProjectIndex persistence — deferred (in-memory is fast enough)

### Phase 4: Graph-Aware Queries [DONE]

- DAG traversal (--depends-on, --depended-by) [IMPLEMENTED]
- Dependency PageRank [IMPLEMENTED] — pre-computed at index build, O(1) via `cached_pagerank()`
- `--pagerank` enrichment flag [IMPLEMENTED]
- `--graph` shows both depends-on and depended-by [IMPLEMENTED]
- Impact-weighted gap analysis [IMPLEMENTED] — via `score_codebase_with_pagerank()`

### Phase 5: CI Integration [DONE]

- --exit-code for quality gates [IMPLEMENTED]
- Trend tracking via JSON output [IMPLEMENTED via -f json]
- Cross-project violation alerting [IMPLEMENTED via --violations --exit-code]
- Commit refs scanning in CrossProjectIndex [IMPLEMENTED]
- --project, --include-project, --all-projects flags [IMPLEMENTED]
- --rebuild-index flag [IMPLEMENTED]
