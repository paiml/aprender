# Section 34: Systems Contract Patterns

> Parent: [pv-spec.md](../pv-spec.md) §34

## 34.1 Motivation

Sections 1–33 cover **mathematical** contracts (equations, bounds, roundtrips) and **interface**
contracts (CLI, MCP, HTTP). This section specifies contract patterns for **systems** concerns:
threading, concurrency, async, compute backend dispatch, memory lifecycle, and LLM architecture
invariants. These patterns cut across all sovereign stack repos.

### References

**Threading & Concurrency:**
- Lamport (1978). Time, Clocks, and the Ordering of Events. CACM 21(7)
- Flanagan & Freund (2009). FastTrack: Efficient and Precise Dynamic Race Detection. PLDI
- Jung et al. (2020). RustBelt meets Relaxed Memory. POPL
- Zhao & Sanan (2023). Rely-guarantee Concurrent Memory Management. arXiv:2309.09997
- Antonino et al. (2022). Pattern-based Deadlock-Freedom Analysis. arXiv:2207.08854
- Wu et al. (2023). Model Checking Race-Freedom under SC-DRF. arXiv:2305.18198
- Jacobs & Fasse (2025). Modular Verification of Rust Arc. arXiv:2505.00449
- Hahnle et al. (2023). Context-aware Trace Contracts for Async. arXiv:2310.04384
- Lagaillardie et al. (2022). Affine Rust with Multiparty Session Types. arXiv:2204.13464
- Lattuada et al. (2023). Verus: Verifying Rust via Linear Ghost Types. arXiv:2303.05491
- Cutner et al. (2021). Deadlock-free Async Message Reordering in Rust. arXiv:2112.12693
- Barwell et al. (2022). Multiparty Session Types with Crash-Stop. arXiv:2207.02015
- Shi et al. (2025). Complexity of Testing Message-Passing Concurrency. arXiv:2505.05162
- Pearce et al. (2025). RustMC: Stateless Model Checker for Rust. arXiv:2502.06293
- Ayoun et al. (2024). Gillian-Rust: Hybrid Semi-automated Verification. arXiv:2403.15122

**Compute (SIMD, GPU, PTX):**
- Liu et al. (2023). Minotaur: SIMD-Oriented Synthesizing Superoptimizer. arXiv:2306.00229
- Taneja et al. (2024). LLM-Vectorizer: Verified Loop Vectorization via Alive2. arXiv:2406.04693
- Dubey et al. (2025). Volta: Equivalence Checking of ML GPU Kernels. arXiv:2511.12638
- Chatterjee et al. (2025). ProofWright: Agentic Formal Verification of CUDA. arXiv:2511.12294
- Liew et al. (2022). Provable GPU Data-Race Freedom via Memory Access Protocols. arXiv:2203.12878
- Jacobson et al. (2024). HiRace: Accurate Source-Level GPU Race Checking. arXiv:2401.04701
- Abraham & Okoli (2026). Universal GPU ISA: Cross-Vendor Computational Primitives. arXiv:2603.28793
- Chakraborty et al. (2025). GPUMC: Stateless Model Checker for GPU Weak Memory. arXiv:2505.20207
- Khattak & Mikaitis (2025). Accurate Models of NVIDIA Tensor Cores. arXiv:2512.07004
- Xie et al. (2024). FPRev: Revealing FP Accumulation Orders. arXiv:2411.00442
- Shanmugavelu et al. (2024). FP Non-Associativity Impacts on Reproducibility. arXiv:2408.05148

**Memory (Lazy, Arena, Budget):**
- Xia et al. (2024). Bidirectional Demand Semantics for Lazy Programs. arXiv:2406.14787
- He et al. (2025). Arena Type System with Higher-Order Reachability. arXiv:2509.04253
- Hughes et al. (2025). Spegion: Non-Lexical Regions with Sized Allocations. arXiv:2506.02182
- Mannucci & Thuro (2025). Resource-Bounded Type Theory via Graded Modalities. arXiv:2512.06952
- Lian & Wang (2025). RaRust: Automatic Linear Resource Bounds for Rust. arXiv:2502.19810
- Arnold & Marron (2025). Catalpa: GC with Provably Bounded Pauses. arXiv:2509.13429
- Congard et al. (2025). Linear Effects and Resource Safety via Curry-Howard. arXiv:2510.23517
- Tan et al. (2024). Formalising CXL Cache Coherence in Isabelle. arXiv:2410.15908

**LLM Architecture:**
- Dao et al. (2022). FlashAttention: IO-Aware Exact Attention. arXiv:2205.14135
- Golden et al. (2024). Is Flash Attention Stable? arXiv:2405.02803
- Men et al. (2024). Base of RoPE Bounds Context Length. arXiv:2405.14591
- Leviathan et al. (2023). Speculative Decoding. arXiv:2211.17192
- Qin et al. (2025). Batch Speculative Decoding Done Right. arXiv:2510.22876
- Zhang et al. (2022). QEBVerif: Quantization Error Bound Verification. arXiv:2212.02781
- Cooke et al. (2023). Guaranteed Quantization Error via Merged Networks. arXiv:2304.13812
- Rinberg et al. (2025). Verifying LLM Inference for Weight Exfiltration Detection. arXiv:2511.02620
- Zouhar et al. (2023). Formal Perspective on BPE. arXiv:2306.16837
- Cognetta et al. (2023). Formalizing BPE as Finite-State Transducer. arXiv:2309.08715
- Li et al. (2025). Mind the Gap: GGUF Quantization Attack. arXiv:2505.23786
- Dereich & Jentzen (2024). Convergence Rates for Adam Optimizer. arXiv:2407.21078
- Mu & Klabjan (2025). LoRA Gradient Descent Convergence. arXiv:2512.18248

**Transpiler Verification:**
- Lerner et al. (2003). Automated Soundness Proofs for Dataflow Analyses and Transformations. POPL
- Yang et al. (2011). Finding and Understanding Bugs in C Compilers. PLDI (Csmith)
- Leroy (2009). CompCert: Formal Verification of a Realistic Compiler. CACM
- Nandi et al. (2021). Synthesizing Structured CAD Models via Equality Saturation. PLDI
- Sotoudeh & Thakur (2019). Verifying Semantic Equivalence of Translated Programs. arXiv:1911.07671

## 34.2 Threading & Lock Ordering

### Contract Pattern: `lock-ordering`

Prevents deadlocks by enforcing a total order on lock acquisition.

```yaml
equations:
  lock_order_invariant:
    formula: |
      lock_order: (Mutex<A>, Mutex<B>) -> bool
        ∀ thread t: if t.holds(A) ∧ t.acquires(B) then order(A) < order(B)
        Violated: A.lock() then B.lock() where order(A) > order(B)
    invariants:
    - Lock acquisition follows total order (no cycles in wait-for graph)
    - Documented lock levels: L0 (index), L1 (cache), L2 (state), L3 (IO)
    preconditions:
    - All lockable resources have assigned levels
    postconditions:
    - No deadlock reachable from any interleaving
    lean_theorem: Theorems.LockOrderInvariant
```

**Proof obligation type**: `ordering`
**Falsification**: Spawn 100 threads acquiring locks A,B in both orders; detect panic or timeout.
**Applies to**: `pmat` (DashMap in index), `forjar` (state lock + config lock), `trueno` (graph + cache).

### Contract Pattern: `data-race-freedom`

```yaml
equations:
  race_freedom:
    formula: |
      race_free: Program -> bool
        ∀ memory location m, ∀ concurrent accesses (a₁, a₂) to m:
          a₁.is_write ∨ a₂.is_write → synchronized(a₁, a₂)
    invariants:
    - All shared mutable state behind Mutex, RwLock, or atomic
    - No raw pointer aliasing across thread boundaries
    preconditions:
    - Program compiles under Rust borrow checker (Send + Sync)
    postconditions:
    - MIRI detects zero data races under any scheduling
```

**Proof obligation type**: `soundness`
**Falsification**: Run under `cargo +nightly miri test` with `-Zmiri-preemption-rate=0.1`.

## 34.3 Async & Structured Concurrency

### Contract Pattern: `cancellation-safety`

Ensures async tasks release all resources when cancelled.

```yaml
equations:
  cancellation_safe:
    formula: |
      cancel: Task<T> -> ()
        ∀ resource r acquired by task:
          r.is_released() after cancel
        No leaked: file handles, temp files, network connections, semaphore permits
    invariants:
    - Drop impl releases all resources
    - select! branches are cancellation-safe (no partial state)
    preconditions:
    - Task is in Running state
    postconditions:
    - All acquired resources released within 1s of cancel signal
    lean_theorem: Theorems.CancellationSafe
```

**Proof obligation type**: `frame`
**Falsification**: Start analysis, cancel after 100ms via `tokio::time::timeout`, check for `.tmp` files and open fd count.

### Contract Pattern: `structured-concurrency`

```yaml
equations:
  structured_spawn:
    formula: |
      structured: (Parent, Vec<Child>) -> bool
        ∀ child in parent.spawned:
          child.lifetime ⊆ parent.lifetime
        parent.await => all children completed or cancelled
    invariants:
    - No orphan tasks (tasks that outlive their parent scope)
    - JoinSet/TaskSet owns all spawned work
    - Panic in child propagates to parent (not silently lost)
```

**Proof obligation type**: `frame`
**Applies to**: `pmat` (rayon parallel analysis), `realizar` (batched inference), `entrenar` (data loader workers).

### Contract Pattern: `channel-conservation`

```yaml
equations:
  channel_lossless:
    formula: |
      lossless: (Sender<T>, Receiver<T>, Bound) -> bool
        sent_count = received_count + pending_count + dropped_on_close
        ∀ msg: msg is received XOR sender was dropped before delivery
    invariants:
    - No silent message loss in bounded channels
    - Backpressure: send blocks when channel full (not drop)
    postconditions:
    - After close: receiver drains remaining pending messages
```

**Proof obligation type**: `conservation`

## 34.4 Compute Backend Dispatch

### Contract Pattern: `simd-scalar-parity`

The foundational compute contract: SIMD and GPU implementations must match scalar within ULP tolerance.

```yaml
equations:
  simd_scalar_parity:
    formula: |
      parity: (f_scalar, f_simd, x) -> bool
        |f_scalar(x) - f_simd(x)| ≤ ULP_TOLERANCE * ε_format
        Where:
          ε_f32 = 2^{-23} ≈ 1.19e-7
          ε_f16 = 2^{-10} ≈ 9.77e-4
          ULP_TOLERANCE = sqrt(n) for n-element reductions (FMA reassociation)
    domain: x ∈ ℝ^n, n ≤ MAX_VECTOR_LEN
    codomain: bool (parity within tolerance)
    invariants:
    - scalar is the reference implementation (ground truth)
    - SIMD may reassociate FMA operations (different rounding)
    - Tolerance is derived from arithmetic, not guessed
    preconditions:
    - x contains no NaN or Inf (unless testing those paths)
    - f_scalar and f_simd compute the same mathematical function
    postconditions:
    - max element-wise error ≤ ULP_TOLERANCE * ε_format
    lean_theorem: Theorems.SimdScalarParity
```

**Proof obligation type**: `equivalence`
**Tolerance selection** (from Jiang et al. 2024):

| Operation | n | ε_f32 Tolerance | Rationale |
|-----------|---|-----------------|-----------|
| Element-wise (relu, sigmoid) | 1 | 0 ULP | No reassociation |
| Dot product | n | sqrt(n) * ε | FMA reassociation |
| Softmax | n | n * ε | exp + division chain |
| MatMul (M×K×N) | K | sqrt(K) * ε | Reduction dimension |
| LayerNorm/RMSNorm | n | 2*sqrt(n) * ε | sqrt + division |

**Falsification**: Property test with 10K random vectors, compare scalar vs AVX2 vs NEON, assert max ULP error within bound.

### Contract Pattern: `gpu-cpu-parity`

Extends simd-scalar-parity to GPU backends (WGPU, CUDA, PTX).

```yaml
equations:
  gpu_cpu_parity:
    formula: |
      parity: (f_cpu, f_gpu, x) -> bool
        |f_cpu(x) - f_gpu(x)| ≤ GPU_TOLERANCE * ε_format
        Where GPU_TOLERANCE = max(ULP_TOLERANCE, WARP_REASSOCIATION_BOUND)
        WARP_REASSOCIATION_BOUND = log2(WARP_SIZE) * ε for reductions
    invariants:
    - CPU (scalar or SIMD) is reference; GPU must match
    - GPU may have wider reassociation window (warp-level reduce)
    - Different GPU architectures may have different rounding
    - PTX must match WGPU within 1 ULP (same hardware, different frontend)
    postconditions:
    - max element-wise error ≤ GPU_TOLERANCE * ε_format
    lean_theorem: Theorems.GpuCpuParity
```

**Additional GPU invariants**:

```yaml
  gpu_safety:
    formula: |
      safe: Kernel -> bool
        No out-of-bounds shared memory access
        No uninitialized shared memory read
        Barrier sync before shared memory read-after-write
        Workgroup size ≤ device.max_workgroup_size
    invariants:
    - Bounds checks on all shared memory indices
    - __syncthreads() / workgroupBarrier() before cross-lane reads
    - No divergent barriers (all threads in workgroup reach same barrier)
```

**Proof obligation type**: `equivalence` (parity), `soundness` (safety)

### Contract Pattern: `backend-dispatch-completeness`

```yaml
equations:
  dispatch_complete:
    formula: |
      dispatch: (Operation, Backend) -> Impl
        ∀ op in OperationSet, ∀ backend in {Scalar, AVX2, NEON, WGPU, CUDA, PTX}:
          if backend.is_available() then dispatch(op, backend) exists
        Fallback: unavailable backend → scalar (never panic)
    invariants:
    - Every operation has scalar fallback
    - Runtime detection: CPUID for SIMD, device enumeration for GPU
    - No compile-time only dispatch (must handle runtime absence)
    postconditions:
    - dispatch never returns None; always falls back to scalar
```

**Proof obligation type**: `completeness`
**Applies to**: `trueno` (SIMD dispatch), `realizar` (GPU inference), `aprender` (backend selection).

## 34.5 Memory Lifecycle Patterns

### Contract Pattern: `lazy-initialization`

```yaml
equations:
  lazy_init:
    formula: |
      lazy: LazyCell<T> -> T
        First access: T = init()
        Subsequent: T = cached_value (no re-init)
        init() called at most once (even under concurrency)
    invariants:
    - Initialization is idempotent (safe to race)
    - Once initialized, value never changes
    - Thread-safe: OnceCell/OnceLock semantics
    preconditions:
    - init function is pure (no side effects beyond allocation)
    postconditions:
    - All accesses after first return the same value
    lean_theorem: Theorems.LazyInit
```

**Proof obligation type**: `idempotency`
**Applies to**: `pmat` (lazy source loading in SQLite backend), `trueno` (lazy SIMD detection), `aprender` (lazy model loading).

### Contract Pattern: `lru-cache-correctness`

```yaml
equations:
  lru_correctness:
    formula: |
      lru: (Cache<K,V>, Capacity) -> bool
        cache.len() ≤ capacity always
        get(k) exists → k becomes most-recently-used
        put(k,v) when full → evicts least-recently-used
        evicted entries: refcount = 0, memory freed
    invariants:
    - Size invariant: len() ≤ capacity after any operation
    - Recency ordering: access list is permutation of key set
    - No phantom entries: evicted key not findable by get()
    postconditions:
    - After put at capacity: exactly one eviction
    lean_theorem: Theorems.LruCorrectness
```

**Proof obligation type**: `bound` (capacity), `ordering` (recency), `frame` (eviction cleanup)

### Contract Pattern: `arena-lifetime`

```yaml
equations:
  arena_contained:
    formula: |
      arena: Arena<'a> -> bool
        ∀ ref r allocated from arena: lifetime(r) ⊆ lifetime(arena)
        drop(arena) → all allocations freed
        No use-after-free: r is invalid after arena.reset()
    invariants:
    - Allocations cannot escape arena lifetime (enforced by Rust borrow checker)
    - Bulk deallocation on drop (O(1) per arena, not per object)
    postconditions:
    - After drop: heap usage returns to pre-arena level
```

**Proof obligation type**: `frame`

### Contract Pattern: `memory-budget`

```yaml
equations:
  budget_honored:
    formula: |
      budgeted: (Operation, Budget) -> Result<T, OOM>
        peak_memory(operation) ≤ budget
        If would exceed: returns Err(OOM) before allocation
        Never triggers OS OOM killer
    invariants:
    - Budget checked before large allocations (not after)
    - Partial results returned on budget exhaustion (not empty)
    - Budget tracking is conservative (overestimates, never underestimates)
    postconditions:
    - process RSS ≤ budget + baseline_overhead
```

**Proof obligation type**: `bound`
**Applies to**: `pmat` (index loading), `realizar` (model loading), `trueno-zram` (compression buffer).

## 34.6 LLM Architecture Contracts

### Contract Pattern: `attention-correctness`

```yaml
equations:
  attention:
    formula: |
      Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
      Where:
        Q ∈ ℝ^{n×d_k}, K ∈ ℝ^{m×d_k}, V ∈ ℝ^{m×d_v}
        Output ∈ ℝ^{n×d_v}
    invariants:
    - Attention weights sum to 1.0 per query (softmax normalization)
    - Causal mask: attn[i,j] = 0 when j > i (autoregressive)
    - GQA: num_kv_heads divides num_heads evenly
    preconditions:
    - Q, K have matching d_k dimension
    - Sequence length ≤ max_seq_len
    postconditions:
    - Output shape = (batch, n, d_v)
    - attn_weights ∈ [0, 1] and rows sum to 1.0 ± ε
```

**Proof obligation type**: `bound` (weights), `invariant` (causal mask), `conservation` (normalization)

### Contract Pattern: `kv-cache-coherence`

```yaml
equations:
  kv_cache_coherent:
    formula: |
      cache_coherent: (KVCache, NewTokens) -> bool
        append(cache, new_kv) preserves all existing entries
        cache[pos] = kv_at_generation_time(pos) for all pos < current
        Paged: page_table[logical] → physical is bijective for active pages
    invariants:
    - Append-only during generation (no mutation of past KV pairs)
    - Paged attention: no two logical positions map to same physical page
    - Eviction: only evict positions that will never be attended to
    postconditions:
    - cache.len() = total_generated_tokens
    - Recompute from scratch matches cached result within ε
```

**Proof obligation type**: `invariant` (append-only), `equivalence` (recompute parity)

### Contract Pattern: `quantization-roundtrip`

```yaml
equations:
  quant_roundtrip:
    formula: |
      roundtrip: (Tensor, QuantConfig) -> f64
        error = max|dequant(quant(x)) - x| / max|x|
        error ≤ expected_error(config)
        Where:
          Q4_K:  expected ≤ 0.05 (5% relative)
          Q8_0:  expected ≤ 0.004 (0.4% relative)
          FP8:   expected ≤ 0.02 (2% relative)
          INT8:  expected ≤ 0.008 (0.8% relative)
    invariants:
    - Quantize then dequantize preserves value within tolerance
    - Zero is preserved exactly (quant(0) = 0, dequant(0) = 0)
    - Monotonicity: x₁ < x₂ → quant(x₁) ≤ quant(x₂)
    postconditions:
    - Relative error ≤ expected_error(config)
```

**Proof obligation type**: `roundtrip`, `bound`, `monotonicity`

### Contract Pattern: `tokenizer-correctness`

```yaml
equations:
  tokenizer_roundtrip:
    formula: |
      roundtrip: String -> bool
        decode(encode(text)) = text for all valid UTF-8
        encode(text).len() ≤ text.len() * MAX_EXPANSION_FACTOR
    invariants:
    - Encode-decode is identity for valid input
    - Unknown tokens get fallback (byte-level BPE, not panic)
    - Encoding is deterministic (same text → same tokens)
    postconditions:
    - Decoded output = original input
    - Token count is bounded
```

**Proof obligation type**: `roundtrip`, `determinism`, `bound`

### Contract Pattern: `sampling-distribution`

```yaml
equations:
  sampling_valid:
    formula: |
      sample: (Logits, SamplingParams) -> TokenId
        temperature > 0: softmax(logits / temperature) is valid distribution
        top_k > 0: only top k logits considered
        top_p ∈ (0, 1]: nucleus sampling threshold
        Greedy (temperature = 0): argmax(logits) deterministically
    invariants:
    - Sampled token is always a valid vocabulary index
    - Temperature 0 is deterministic (argmax)
    - Temperature > 0 with fixed seed is reproducible
    postconditions:
    - token_id ∈ [0, vocab_size)
    - Greedy sampling: token_id = argmax(logits)
```

**Proof obligation type**: `bound` (valid index), `determinism` (greedy/seeded)

---

*Continued in [systems-contract-patterns-2.md](systems-contract-patterns-2.md) (sections 34.7-34.10).*
