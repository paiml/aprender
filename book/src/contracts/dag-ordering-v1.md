# dag-ordering-v1

**Version:** 1.0.0

DAG topological ordering — Kahn's algorithm with deterministic tie-breaking

## References

- Kahn (1962) Topological sorting of large networks
- Cormen et al. (2009) Introduction to Algorithms, Chapter 22

## Equations

### kahn_sort

$$
BFS with priority queue (alphabetical) for zero-indegree nodes
$$

**Domain:** $in_degree: HashMap<String, usize>, adjacency: HashMap<String, Vec<String>>$

**Codomain:** $Vec<String>$

**Invariants:**

- $Output contains only nodes from input$
- $Tie-breaking is alphabetical (deterministic)$

### topological_sort

$$
order = KahnSort(G) where G = (V, E) from resource depends_on edges
$$

**Domain:** $G = DAG (directed acyclic graph) of resource dependencies$

**Codomain:** $Vec<String> in topological order, or Err if cycle$

**Invariants:**

- $\forall edge (u, v) \in E: index(u) < index(v) in output$
- $Cycle detection: returns Err if DAG has cycle$
- $Deterministic: alphabetical tie-breaking for zero-indegree nodes$
- $|output| = |V| when no cycle$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | ordering | Topological ordering respected | $\forall (u, v) \in E: position(u, order) < position(v, order)$ |
| 2 | soundness | Cycle detection is sound | $\exists cycle ⟹ build_execution_order returns Err$ |
| 3 | invariant | Deterministic output | $\forall G: KahnSort(G) = KahnSort(G)$ |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-DAG-001 | Ordering | For any DAG, every dependency appears before its dependent in output | Bug in Kahn's algorithm or adjacency construction |
| FALSIFY-DAG-002 | Cycle detection | Any graph with a cycle returns Err containing 'cycle' | Cycle detection incomplete |
| FALSIFY-DAG-003 | Determinism | build_execution_order(G) = build_execution_order(G) for any G | Non-deterministic iteration order in HashMap |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-DAG-001 | Topological ordering | 6 | exhaustive |

## QA Gate

**DAG Ordering Contract** (F-DAG-001)

Topological sort correctness quality gate

**Checks:** ordering, cycle_detection, determinism

**Pass criteria:** All 3 falsification tests pass

