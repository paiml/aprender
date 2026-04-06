# WASM Optimization Specification for Trueno

**Version:** 1.0
**Status:** Draft
**Author:** Claude Code
**Date:** 2025-12-14

## Executive Summary

This specification defines WASM optimization requirements for Trueno to support efficient ML inference in browser environments. Trueno is the SIMD/GPU compute primitive layer that powers the Sovereign AI stack (aprender → realizar → whisper.apr).

## Problem Statement

Current WASM matmul performance is unacceptable:
- **Observed:** 4.5s encode time for 1.5s audio in whisper.apr
- **Target:** <500ms encode time (RTF < 2.0)
- **Root Cause:** Transpose-based matmul algorithm with poor cache locality

## Architecture Context

```
┌─────────────────────────────────────────────────────────────────┐
│                    Sovereign AI Stack                            │
├─────────────────────────────────────────────────────────────────┤
│  whisper.apr    │  demos/UI      │  Browser WASM deployment     │
│  apr-cookbook   │  examples      │  .apr format patterns        │
│  realizar       │  inference     │  GGUF/Safetensors → .apr     │
│  aprender       │  ML library    │  Pure Rust ML primitives     │
│  trueno         │  SIMD/GPU      │  ← THIS SPEC (compute layer) │
└─────────────────────────────────────────────────────────────────┘
```

## WASM Deployment Stories

### Story 1: Browser Inference (Primary)
**User:** Web developer deploying ML model
**Pattern:** Load .apr model → Run inference → Display results
**Example:** whisper.apr realtime transcription demo

**Requirements:**
- First inference latency: <5s
- Sustained RTF: <2.0x
- Memory peak: <200MB for tiny models
- No WebGPU dependency (fallback to SIMD128)

### Story 2: Progressive Model Loading
**User:** App loading large model over slow connection
**Pattern:** Stream .apr blocks → Partial inference → Full inference
**Example:** apr-cookbook/examples/wasm/wasm_progressive_loading.rs

**Requirements:**
- Streaming decompression (LZ4 blocks)
- Incremental weight loading
- Early inference with partial weights

### Story 3: Web Worker Offload
**User:** UI-responsive inference
**Pattern:** Main thread → Worker → postMessage results
**Example:** apr-cookbook/examples/wasm/wasm_web_worker.rs

**Requirements:**
- Zero main-thread blocking
- Efficient audio chunk transfer (SharedArrayBuffer)
- <50ms message latency

### Story 4: WebGPU Acceleration
**User:** High-performance inference with GPU
**Pattern:** WASM + WebGPU compute shaders
**Example:** apr-cookbook/examples/wasm/wasm_webgpu_acceleration.rs

**Requirements:**
- Graceful fallback to SIMD128
- GPU memory management
- Shader compilation caching

### Story 5: Streaming Compilation
**User:** Fast startup with large WASM modules
**Pattern:** Compile WASM while downloading
**Example:** apr-cookbook/examples/wasm/wasm_streaming_compilation.rs

**Requirements:**
- WebAssembly.compileStreaming support
- Module caching (IndexedDB)
- <1s time-to-interactive

## Performance Targets

| Operation | Current | Target | Improvement |
|-----------|---------|--------|-------------|
| matmul(384×74, 74×384) | ~50ms | <10ms | 5x |
| matmul(384×384, 384×384) | ~200ms | <30ms | 6x |
| Whisper tiny encode (1.5s audio) | 4500ms | <500ms | 9x |
| Whisper tiny decode (per token) | ~100ms | <20ms | 5x |
| Full transcription RTF | >3.0x | <2.0x | 1.5x |

## Optimization Strategies

### Strategy 1: Tiled Matrix Multiplication (Priority: HIGH)

**Current Algorithm (matmul_simd_simple):**
```rust
// Problem: Transpose allocates O(n²), poor cache locality
let b_transposed = other.transpose();  // SLOW
for i in 0..rows {
    for j in 0..cols {
        result[i,j] = dot(a_row[i], b_col_transposed[j]);  // Cache miss
    }
}
```

**Proposed Algorithm (matmul_wasm_tiled):**
```rust
// Solution: Tiled blocking, no transpose, register accumulation
const TILE: usize = 8;  // Fits in WASM stack
for ti in (0..rows).step_by(TILE) {
    for tj in (0..cols).step_by(TILE) {
        let mut acc = [[0.0f32; TILE]; TILE];  // Register file
        for tk in (0..inner).step_by(TILE) {
            // Accumulate 8x8 tile
            for i in 0..TILE {
                for k in 0..TILE {
                    let a_val = a[(ti+i)*inner + tk+k];
                    for j in 0..TILE {
                        acc[i][j] += a_val * b[(tk+k)*cols + tj+j];
                    }
                }
            }
        }
        // Write tile to result
        for i in 0..TILE {
            for j in 0..TILE {
                result[(ti+i)*cols + tj+j] = acc[i][j];
            }
        }
    }
}
```

**Benefits:**
- No transpose allocation (O(1) extra memory vs O(n²))
- Cache-friendly access pattern (sequential B reads)
- Register accumulation reduces memory traffic 8x
- SIMD-friendly inner loop (vectorize j dimension)

### Strategy 2: SIMD128 Vectorized Dot Product

**Current:** Loop-based dot with SIMD chunks
**Proposed:** Unrolled 4-wide FMA

```rust
#[target_feature(enable = "simd128")]
unsafe fn dot_simd128_unrolled(a: &[f32], b: &[f32]) -> f32 {
    let mut sum0 = f32x4_splat(0.0);
    let mut sum1 = f32x4_splat(0.0);
    let mut sum2 = f32x4_splat(0.0);
    let mut sum3 = f32x4_splat(0.0);

    let chunks = a.len() / 16;
    for i in 0..chunks {
        let idx = i * 16;
        sum0 = f32x4_add(sum0, f32x4_mul(
            v128_load(a.as_ptr().add(idx) as *const v128),
            v128_load(b.as_ptr().add(idx) as *const v128)));
        sum1 = f32x4_add(sum1, f32x4_mul(
            v128_load(a.as_ptr().add(idx+4) as *const v128),
            v128_load(b.as_ptr().add(idx+4) as *const v128)));
        // ... sum2, sum3
    }

    // Horizontal sum
    let sum = f32x4_add(f32x4_add(sum0, sum1), f32x4_add(sum2, sum3));
    f32x4_extract_lane::<0>(sum) + f32x4_extract_lane::<1>(sum) +
    f32x4_extract_lane::<2>(sum) + f32x4_extract_lane::<3>(sum)
}
```

### Strategy 3: Memory Layout Optimization

**Current:** Row-major for A, transpose B
**Proposed:** Row-major A, column-major B (no runtime transpose)

For models stored in .apr format, pre-transpose weight matrices at serialization time:
- Encoder attention K,V: Store transposed
- Decoder cross-attention K,V: Store transposed
- Linear layers: Store transposed if input is row-major

### Strategy 4: Quantized Operations (INT8/INT4)

For quantized .apr models (whisper-tiny-int8.apr):
```rust
#[target_feature(enable = "simd128")]
unsafe fn dot_i8_simd128(a: &[i8], b: &[i8], scale: f32) -> f32 {
    // Use i8x16 operations for 16 elements per SIMD op
    // 4x throughput vs f32
}
```

### Strategy 5: WebGPU Fallback

When WebGPU is available, offload large matmuls:
```rust
fn matmul_dispatch(a: &Matrix, b: &Matrix) -> Matrix {
    if a.rows * a.cols * b.cols > GPU_THRESHOLD && webgpu_available() {
        matmul_webgpu(a, b)
    } else {
        matmul_wasm_tiled(a, b)
    }
}
```

## Implementation Plan

### Phase 1: Tiled Matmul (Week 1)
1. Add `matmul_wasm_tiled` to `matrix.rs`
2. Add WASM-specific dispatch in `matmul()`
3. Unit tests with property-based testing
4. Benchmark against current implementation

### Phase 2: SIMD128 Optimization (Week 2)
1. Unrolled dot product in `backends/wasm.rs`
2. Vectorized tile accumulation
3. Benchmark isolated operations

### Phase 3: Integration Testing (Week 3)
1. whisper.apr encode benchmark
2. Full transcription RTF measurement
3. Memory profiling
4. Playbook validation

### Phase 4: Advanced Optimizations (Week 4+)
1. INT8 quantized matmul
2. WebGPU compute shader backend
3. Pre-transposed weight format in .apr

## Test Plan

### Unit Tests
```rust
#[test]
fn test_matmul_wasm_tiled_correctness() {
    // Compare against naive implementation
}

#[test]
fn test_matmul_wasm_tiled_non_tile_aligned() {
    // Test 67x89 matrices (not divisible by 8)
}

#[proptest]
fn prop_matmul_wasm_matches_naive(
    #[strategy(1..=256usize)] rows: usize,
    #[strategy(1..=256usize)] inner: usize,
    #[strategy(1..=256usize)] cols: usize,
) {
    // Property: tiled result == naive result (within epsilon)
}
```

### Performance Tests
```rust
#[bench]
fn bench_matmul_wasm_384x384() {
    // Target: <30ms
}

#[bench]
fn bench_whisper_encode_1_5s() {
    // Target: <500ms
}
```

### Integration Tests
```rust
#[test]
fn test_whisper_transcription_rtf() {
    // Target: RTF < 2.0
    let result = whisper.transcribe(&audio_1_5s);
    assert!(result.rtf < 2.0);
}
```

## Success Criteria

| Metric | Threshold | Measurement |
|--------|-----------|-------------|
| matmul 384×384 | <30ms | `cargo bench` |
| Whisper encode | <500ms | Demo console log |
| Transcription RTF | <2.0x | Playbook assertion |
| Memory peak | <200MB | Browser DevTools |
| Test coverage | >95% | `cargo llvm-cov` |
| Mutation score | >85% | `cargo mutants` |

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| SIMD128 not enabled | No speedup | Runtime detection + scalar fallback |
| Tile size suboptimal | Reduced gains | Benchmark multiple sizes (4, 8, 16) |
| Memory pressure | OOM crashes | Streaming/chunked processing |
| Browser variance | Inconsistent perf | Test on Chrome, Firefox, Safari |

## References

- [WebAssembly SIMD Proposal](https://github.com/WebAssembly/simd)
- [WASM Performance Best Practices](https://webassembly.org/docs/high-level-goals/)
- [Matrix Multiplication Optimization](https://en.algorithmica.org/hpc/algorithms/matmul/)
- [trueno SIMD Architecture](../SIMD_PERFORMANCE.md)
- apr-cookbook WASM Examples (external, see aprender repository)

## Appendix A: Benchmark Baseline

```
Platform: Chrome 120, M1 MacBook Pro
WASM: wasm32-unknown-unknown, -O3

Current Performance (2024-12-14):
  matmul(384, 74, 384):    48ms
  matmul(384, 384, 384):   ~180ms (estimated)
  whisper_encode(1.5s):    4534ms
  whisper_decode(1 token): ~100ms (estimated)
  full_transcription_rtf:  >3.0x

Target Performance:
  matmul(384, 74, 384):    <10ms
  matmul(384, 384, 384):   <30ms
  whisper_encode(1.5s):    <500ms
  whisper_decode(1 token): <20ms
  full_transcription_rtf:  <2.0x
```

## Appendix B: .apr Format WASM Considerations

The .apr model format (from aprender) should optimize for WASM delivery:

1. **Weight Layout:** Store weights in compute-optimal layout (pre-transposed for matmul)
2. **Quantization:** INT8/INT4 weights reduce download size and enable faster SIMD ops
3. **Block Compression:** LZ4 blocks enable streaming decompression
4. **Metadata:** Include target platform hints (WASM, Native, GPU)

```
## Review & Feedback (Self-Correction)

Upon review of the draft strategies, the following adjustments are required:

1.  **Cross-Origin Headers for Workers:**
    *   **Observation:** Story 3 mentions `SharedArrayBuffer`.
    *   **Requirement:** Deployment instructions must mandate `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` headers. Without these, `SharedArrayBuffer` will fail in modern browsers.
    *   **Action:** Add header configuration to `apr-cookbook` server examples.

2.  **Memory Fragmentation:**
    *   **Observation:** Long-running sessions (Story 1) with frequent small allocations can fragment WASM linear memory, eventually causing OOM even if total usage is low.
    *   **Action:** Implement a custom arena allocator or slab allocator for temporary inference buffers (tiles) to ensure stable memory footprint.

3.  **SIMD Fallback Mechanism:**
    *   **Observation:** "Graceful fallback" is risky if implemented only as runtime checks inside tight loops.
    *   **Action:** Use `cfg(target_feature)` to build two WASM binaries (`trueno.wasm` and `trueno-simd.wasm`) or use dynamic dispatch at the *module* level (function pointer swapping) rather than per-instruction checking.

4.  **Mobile Performance Targets:**
    *   **Observation:** M1 targets are aggressive but achievable. Mobile targets are missing.
    *   **Action:** Add specific targets for mid-range Android (Snapdragon 7 series) to ensure accessibility. Target: < 2.0s encode (vs 500ms on Desktop).

## Appendix C: Falsification Checklist (100 Points)

To "falsify" the result means to rigorously attempt to prove the implementation *incorrect* or *insufficient*. If the system passes all checks, we accept the optimization as valid.

### I. Correctness & Precision (30 Points)
- [ ] **1. The Identity Test:** `MatMul(I, A) == A` for random matrix A (384x384). (FAIL if diff > 1e-6)
- [ ] **2. The Zero Test:** `MatMul(0, A) == 0`. (FAIL if any non-zero bit)
- [ ] **3. The Transpose Test:** `MatMul(A, B) == MatMul(B^T, A^T)^T`. Verify layout assumptions.
- [ ] **4. The Non-Aligned Test:** Matrix dimensions prime numbers (e.g., 67x89) work with 8x8 tiling. (FAIL if panic or memory corruption)
- [ ] **5. The NaN Propagation:** If input contains `NaN`, output must contain `NaN`. (FAIL if `NaN` is swallowed)
- [ ] **6. The Infinity Check:** Overflowing values behave consistently with IEEE754 (no undefined UB).
- [ ] **7. The Determinism Check:** Running the same inference 100 times produces bit-exact identical logits.
- [ ] **8. The Quantization Error:** `Float - Int8` error < 1% relative difference for standard distribution.
- [ ] **9. The Scale Factor:** Quantization scale factors are applied correctly (output magnitude matches f32).
- [ ] **10. The Empty Set:** Input vectors of size 0 do not crash the WASM instance.

### II. Performance & Resources (30 Points)
- [ ] **11. The 10ms Barrier:** `matmul(384,74,384)` executes in < 10ms on M1/Reference.
- [ ] **12. The Mobile Floor:** `matmul(384,74,384)` executes in < 40ms on Snapdraon 7xx (or equiv).
- [ ] **13. The Zero-Alloc Loop:** No heap allocations occur during the hot `dot` product or `tile` loop.
- [ ] **14. The Memory Ceiling:** Total WASM memory usage never exceeds 256MB for Whisper Tiny.
- [ ] **15. The Leak Check:** Run inference 1000 times. Memory usage is flat (no growth).
- [ ] **16. The Binary Size:** Gzipped WASM binary is < 2.0MB.
- [ ] **17. The Cold Start:** Time from `init()` to first token is < 1000ms.
- [ ] **18. The Cache Hit:** Second run of inference is measurably faster (if JIT warming applies).
- [ ] **19. The UI Blocker:** Main thread is never blocked for > 16ms (60fps) if using async/workers.
- [ ] **20. The Bandwidth:** Model download + compile time on 4G network is < 5s.

### III. Compatibility & Environment (20 Points)
- [ ] **21. The Chrome Test:** Passes all tests in Chrome Stable.
- [ ] **22. The Firefox Test:** Passes all tests in Firefox Stable.
- [ ] **23. The Safari Test:** Passes all tests in Safari (iOS/macOS).
- [ ] **24. The No-SIMD Fallback:** Runs correctly (slowly) when `simd128` is disabled in browser flags.
- [ ] **25. The Thread Check:** Works without `SharedArrayBuffer` (if configured to fallback).
- [ ] **26. The COOP/COEP Check:** Fails gracefully (informative error) if headers are missing for threads.
- [ ] **27. The Incognito Mode:** Works in private browsing (IndexedDB limitations handled).
- [ ] **28. The Offline Mode:** Works if network is cut after loading.
- [ ] **29. The Version Check:** Explicitly rejects unsupported browser versions with clear message.
- [ ] **30. The Headless Check:** Passes `wasm-pack test --headless --firefox`.

### IV. Integration & Resilience (20 Points)
- [ ] **31. The "Real World" Audio:** Transcribes a real microphone recording, not just synthetic zeroes.
- [ ] **32. The Noise Robustness:** Transcribes audio with background noise (verifies model integrity).
- [ ] **33. The Tab Switch:** Background tab throttling does not crash the inference (just slows it).
- [ ] **34. The OOM Recovery:** If allocation fails, returns `Result::Err` instead of hard abort.
- [ ] **35. The Cancelation:** Can abort inference mid-sentence without leaking memory.
- [ ] **36. The Concurrency:** Two tabs running inference simultaneously do not conflict.
- [ ] **37. The Stress Test:** 1 hour continuous transcription loop passes.
- [ ] **38. The API Match:** WASM output logits match Python reference implementation exactly.
- [ ] **39. The Reload:** Page reload cleans up all resources (Workers terminate).
- [ ] **40. The User Metric:** "Time to Reading" (latency) is perceived as "instant" (< 200ms) by human tester.

