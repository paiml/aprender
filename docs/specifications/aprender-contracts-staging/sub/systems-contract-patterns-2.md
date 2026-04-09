
Version: 1.0
Status: proposed
Date: 2026-04-09

*See also: [systems-contract-patterns.md](systems-contract-patterns.md) (sections 34.1-34.6)*

## 34.7 Cross-Cutting Contract Patterns

### Pattern: `deterministic-parallel`

Parallel execution must produce identical results to sequential (modulo ordering).

```yaml
equations:
  parallel_determinism:
    formula: |
      sort(parallel_map(f, inputs)) = sort(sequential_map(f, inputs))
      Where f is a pure function (no shared mutable state)
    invariants:
    - Results are set-equal (same elements, possibly different order)
    - No result depends on scheduling order
    - HashMap iteration order does not leak into output
```

**Applies to**: All repos using rayon, tokio::spawn, or parallel iterators.

### Pattern: `timeout-bounded`

All long-running operations must honor a timeout.

```yaml
equations:
  timeout_honored:
    formula: |
      bounded: (Operation, Timeout) -> Result<T, TimeoutError>
        wall_clock(operation) ≤ timeout + epsilon
        epsilon = max(1s, cleanup_time)
    invariants:
    - Timeout cancels in-progress work (not just checked at start)
    - Partial results returned when available
    - Resources released on timeout (no leaks)
```

## 34.8 LLM Architecture Contracts (apr-qa-playbook)

Contracts for the full inference pipeline validated by the Model Quality Score (MQS) framework.
The apr-qa-playbook defines 5 gateways (G0–G4) with Popperian falsification.

### Contract Pattern: `flash-attention-parity`

FlashAttention uses tiled computation with online softmax rescaling. The contract asserts
exact numerical equivalence with naive attention (Dao et al. 2022, arXiv:2205.14135).

```yaml
equations:
  flash_attention_parity:
    formula: |
      FlashAttn(Q, K, V) = NaiveAttn(Q, K, V) within ε
        Where NaiveAttn = softmax(Q·K^T / √d_k) · V
        Online rescaling: tracks running max m(x) and normalizer l(x)
        ε = 0 for f32 (exact), n * ε_bf16 for BF16
    invariants:
    - Tiled and naive produce identical output in f32
    - BF16 deviation bounded by n * 2^{-7} (Golden et al. 2024)
    - Causal mask applied identically in both paths
```

**Proof obligation type**: `equivalence`
**Key insight**: FlashAttention is *exact* in f32 (not approximate) — the online rescaling
is algebraically equivalent. BF16 introduces ~10x more deviation than naive (Golden et al. 2024).

### Contract Pattern: `speculative-decoding-distribution`

Speculative decoding must preserve the exact target model distribution (Leviathan et al. 2023).

```yaml
equations:
  speculative_distribution:
    formula: |
      P_spec(x) = P_target(x) for all tokens x
        Via rejection sampling: accept with prob min(1, P_target/P_draft)
        Plus residual correction distribution for rejected tokens
    invariants:
    - Output distribution identical to autoregressive sampling
    - Batch sync: position IDs, attention masks, KV-cache coherent (Qin et al. 2025)
    postconditions:
    - KL(P_spec || P_target) = 0 (exact, not approximate)
```

**Proof obligation type**: `equivalence`

### Contract Pattern: `quantization-error-bound`

Quantization error is bounded layer-by-layer (Zhang et al. 2022, QEBVerif).

```yaml
equations:
  quant_error_bound:
    formula: |
      ||f(x) - f_q(x)|| ≤ ε_layer × depth
        Where ε_layer = O(sqrt(N) × delta) (SPFQ, Zhang et al. 2023)
        delta = quantization step size (Q4_K: 2^{-4}, Q8: 2^{-8})
        Verified via merged network g(x) = f(x) - f_q(x) reachability
    invariants:
    - Error bounded by arithmetic, not empirical sampling
    - Zero preserved exactly: quant(0) = 0
    - GGUF roundtrip error residual is unstructured (Li et al. 2025)
    postconditions:
    - Relative error ≤ expected_error(quant_config)
```

**Proof obligation type**: `bound`

### Contract Pattern: `tokenizer-transducer`

BPE tokenization is equivalent to a finite-state transducer (Cognetta et al. 2023).

```yaml
equations:
  bpe_transducer:
    formula: |
      BPE(text) = FST(text) for all valid UTF-8
        FST enables incremental left-to-right tokenization
        Memory: O(1) per character (constant, not proportional to text)
        Optimality: greedy merge is 1/e-approximate (Zouhar et al. 2023)
    invariants:
    - Encode-decode roundtrip is identity
    - Incremental tokenization = batch tokenization
    - Deterministic: same text → same tokens always
```

**Proof obligation type**: `roundtrip`, `determinism`, `equivalence`

### Contract Pattern: `model-format-integrity` (APR v2)

```yaml
equations:
  apr_v2_integrity:
    formula: |
      valid: AprV2File -> bool
        magic = 0x41505200 (APR\0)
        data_offset % 64 = 0 (64-byte alignment)
        checksum = CRC32(header[0..40] ++ header[44..64])
        I-1: inference(convert(M)) = inference(M)
        I-2: names(writer(M)) = names(reader(M))
        I-3: unknown_dtype(t) → Error (no silent fallback)
        I-4: |μ(original) - μ(converted)| < atol(dtype)
        I-5: encode(decode(tokens)) = tokens
    invariants:
    - Five format invariants (I-1 through I-5) from apr-format-invariants-v1
    - LAYOUT-002: row-major mandate (column-major triggers Jidoka rejection)
```

**Proof obligation type**: `roundtrip`, `invariant`, `soundness`
**Applies to**: `aprender`, `realizar`, `entrenar`, `apr-model-qa-playbook`

### Contract Pattern: `mqs-gateway-pipeline`

```yaml
equations:
  mqs_gateway:
    formula: |
      MQS(evidence) = Σ(category_scores × weights) + proof_bonus - penalties
        G0: Format invariants (I-1 through I-5)
        G1: Inference parity (cross-backend)
        G2: Performance (throughput, latency)
        G3: Stability (determinism, edge cases)
        G4: Garbage detection (adversarial inputs)
        MAX_TOTAL = 1000, MAX_PROOF_BONUS = 50
    invariants:
    - Gateway ordering: G0 blocks G1-G4 (format must pass first)
    - Score bounded: 0 ≤ MQS ≤ 1050
    - Grade monotonic: higher score → higher or equal grade
    - Zeroing: failed gateway zeros its category score
```

**Proof obligation type**: `bound`, `ordering`, `monotonicity`

## 34.9 Transpiler Contracts (depyler, decy, bashrs, ruchy)

Transpiler correctness is the most demanding contract domain — it requires proving that
source and target programs are semantically equivalent (Leroy 2009, CompCert).

### Contract Pattern: `type-preservation`

The transpiled output must have equivalent type semantics to the source.

```yaml
equations:
  type_preservation:
    formula: |
      types: (Source, Target) -> bool
        ∀ expression e in source:
          type(transpile(e)) is compatible with type(e)
        Where compatible means:
          Python int → Rust i64 (or BigInt for unbounded)
          Python float → Rust f64
          Python str → Rust String
          Python list[T] → Rust Vec<T>
          Python dict[K,V] → Rust HashMap<K,V>
          Python None → Rust Option<T>::None
    invariants:
    - No implicit type narrowing (Python int has arbitrary precision)
    - Optional types preserved (None → Option)
    - Collection types preserve element types recursively
    preconditions:
    - Source program type-checks in source language
    postconditions:
    - Target program type-checks in target language (rustc / gcc)
    lean_theorem: Theorems.TypePreservation
```

**Proof obligation type**: `equivalence`
**Applies to**: `depyler` (Python→Rust), `decy` (C++→Rust)

### Contract Pattern: `semantic-equivalence`

The transpiled program must produce identical observable behavior.

```yaml
equations:
  semantic_equivalence:
    formula: |
      equiv: (Source, Target, Input) -> bool
        ∀ input i in domain(source):
          observe(run(source, i)) = observe(run(target, i))
        Where observe captures:
          - Return value
          - stdout/stderr output
          - Exit code
          - File system mutations
          - Network I/O (if deterministic)
    invariants:
    - Terminating programs produce identical output
    - Non-terminating programs diverge at same inputs
    - Side effects are preserved (file writes, exit codes)
    postconditions:
    - Target output = source output for all tested inputs
    lean_theorem: Theorems.SemanticEquivalence
```

**Proof obligation type**: `equivalence`
**Falsification**: Property-based testing with 10K random inputs, compare source and target output.
**Applies to**: `depyler`, `decy`

### Contract Pattern: `transpile-determinism`

Same source always produces byte-identical target.

```yaml
equations:
  transpile_determinism:
    formula: |
      deterministic: Source -> bool
        transpile(source) = transpile(source) always
        No HashMap iteration order leakage
        No timestamp or PID in generated code
    invariants:
    - Byte-identical output across runs
    - Debug and release builds produce same output
    postconditions:
    - BLAKE3(output_1) = BLAKE3(output_2) for same input
```

**Proof obligation type**: `determinism`
**Key bug class**: HashMap iteration order (same bug found in pmat's CommitEmbedder TF-IDF).

### Contract Pattern: `lint-finding-determinism`

Linter analysis must be deterministic and complete.

```yaml
equations:
  lint_determinism:
    formula: |
      deterministic: (Script, Rules) -> bool
        lint(script, rules) = lint(script, rules) always
        findings sorted by (severity DESC, line ASC, rule_id ASC)
    invariants:
    - Same script → same findings (no non-determinism)
    - Findings are totally ordered
    - No false negatives for enabled rules
    preconditions:
    - Script is syntactically parseable
    - Rules are valid rule IDs
    postconditions:
    - findings_1 = findings_2 for same input
```

**Proof obligation type**: `determinism`, `completeness`
**Applies to**: `bashrs` (shell/Makefile linting), `pmat` (complexity analysis)

### Contract Pattern: `parser-soundness`

Parser must accept all valid programs and reject all invalid ones.

```yaml
equations:
  parser_soundness:
    formula: |
      sound: (Parser, Grammar) -> bool
        ∀ valid program p: parse(p).is_ok()
        ∀ invalid program q: parse(q).is_err()
        parse(p).unwrap().to_string() is semantically equivalent to p
    invariants:
    - No false rejects (soundness): valid programs always parse
    - No false accepts (completeness): invalid programs always fail
    - Roundtrip: AST → source → parse → AST is equivalent
    postconditions:
    - Parsed AST represents the input program faithfully
```

**Proof obligation type**: `soundness`, `completeness`, `roundtrip`
**Applies to**: `ruchy` (parser generator), `bashrs` (bash parser), `depyler` (Python parser)

### Contract Pattern: `include-resolution`

Transpilers that handle `#include`, `import`, or `use` must resolve correctly.

```yaml
equations:
  include_resolution:
    formula: |
      resolve: (Directive, SearchPath) -> Result<Source, NotFound>
        #include "foo.h" → search relative, then system paths
        import foo.bar → search PYTHONPATH, then package
        Resolution order is documented and deterministic
    invariants:
    - Same search path → same resolution
    - Missing include → compile error (not silent skip)
    - Circular includes detected and reported
    postconditions:
    - Resolved path exists and is readable
```

**Proof obligation type**: `determinism`, `completeness`
**Applies to**: `decy` (C++ #include), `depyler` (Python import)

## 34.10 Summary: Contract Pattern Registry

| Pattern | Obligation Type | Domain | Repos |
|---------|----------------|--------|-------|
| lock-ordering | ordering | Threading | pmat, forjar, trueno |
| data-race-freedom | soundness | Threading | all Rust repos |
| cancellation-safety | frame | Async | pmat, realizar, entrenar |
| structured-concurrency | frame | Async | pmat, realizar, entrenar |
| channel-conservation | conservation | Async | pmat, forjar |
| simd-scalar-parity | equivalence | Compute | trueno, aprender, realizar |
| gpu-cpu-parity | equivalence | Compute | trueno, realizar |
| backend-dispatch-completeness | completeness | Compute | trueno, aprender, realizar |
| lazy-initialization | idempotency | Memory | pmat, trueno, aprender |
| lru-cache-correctness | bound+ordering | Memory | pmat, trueno |
| arena-lifetime | frame | Memory | pmat, aprender |
| memory-budget | bound | Memory | pmat, realizar, trueno-zram |
| flash-attention-parity | equivalence | LLM | aprender, realizar |
| speculative-decoding-dist | equivalence | LLM | aprender, realizar |
| quantization-error-bound | bound | LLM | aprender, trueno |
| tokenizer-transducer | roundtrip+determ | LLM | aprender |
| model-format-integrity | roundtrip+invariant | LLM | aprender, realizar, entrenar |
| mqs-gateway-pipeline | bound+ordering | LLM | apr-model-qa-playbook |
| attention-correctness | bound+invariant | LLM | aprender, realizar, entrenar |
| kv-cache-coherence | invariant+equiv | LLM | aprender, realizar |
| sampling-distribution | bound+determinism | LLM | aprender, realizar |
| type-preservation | equivalence | Transpiler | depyler, decy |
| semantic-equivalence | equivalence | Transpiler | depyler, decy |
| transpile-determinism | determinism | Transpiler | depyler, decy |
| lint-finding-determinism | determinism+compl | Transpiler | bashrs, pmat |
| parser-soundness | soundness+compl | Transpiler | ruchy, bashrs, depyler |
| include-resolution | determinism+compl | Transpiler | decy, depyler |
| deterministic-parallel | determinism | Cross | all parallel repos |
| timeout-bounded | bound | Cross | all CLI/server repos |

**Total: 29 contract patterns** across 7 domains, backed by 40+ arXiv papers.

Each pattern maps to one or more of the 26 proof obligation types defined in pv-spec §3.
