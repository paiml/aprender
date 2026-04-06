# TRUENO-SPEC-015: PTX/SIMD/wgpu Visualization and Tracing CLI (trueno-explain)

**Version**: 1.1
**Date**: 2025-12-16
**Status**: APPROVED - Ready for Implementation
**Priority**: P2 - Developer Experience
**Binary**: `trueno-explain`
**Philosophy**: Genchi Genbutsu (Go and See) - Make the invisible visible

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-16 | Batuta Team | Initial specification with 10 peer-reviewed citations |
| 1.1 | 2025-12-16 | Claude Code | Added TUI mode (Presentar), Analyzer trait, and Muda mapping |

---

## Executive Summary

`trueno-explain` is a CLI and TUI tool that visualizes and traces code generation flows across Trueno's three execution targets: CPU SIMD, GPU (PTX/wgpu), and WebAssembly. It embodies the Toyota Way principle of **Genchi Genbutsu** (Go and See) by making invisible compiler transformations visible and interactive.

### Core Thesis

> **Hypothesis**: Developers who can interactively visualize the exact assembly/PTX/WGSL generated from their Rust code will write 2-3x more efficient algorithms and catch performance regressions 10x faster than developers relying on benchmarks alone.

### Toyota Way Principles Applied

1. **Genchi Genbutsu** (Go and See): Visualize actual generated code, not abstractions.
2. **Jidoka** (Built-in Quality): Catch inefficiencies at code-gen time via static analysis.
3. **Kaizen** (Continuous Improvement): Track optimization progress via `diff` mode.
4. **Muda Elimination** (Waste Reduction):
   *   *Muda of Transport*: Register spills (moving data unnecessarily).
   *   *Muda of Waiting*: Uncoalesced memory access (stalls).
   *   *Muda of Overprocessing*: Redundant instructions or excessive precision.
5. **Heijunka** (Level Loading): Visualize warp divergence and lane imbalance to ensure even work distribution.

---

## 1. Problem Statement

### 1.1 The Visibility Gap

Modern high-performance code involves multiple abstraction layers:

```
Rust Source → MIR → LLVM IR → Assembly/PTX → Hardware
     ↓           ↓        ↓           ↓
  Visible    Hidden   Hidden     Invisible
```

Developers face critical questions they cannot easily answer:
- "Did my SIMD hint actually vectorize?"
- "How many registers does my PTX kernel use?"
- "Is my memory access pattern coalesced?"
- "Why is my GPU kernel 10x slower than expected?"

### 1.2 Current Pain Points

| Pain Point | Impact | Toyota Countermeasure |
|------------|--------|-----------------------|
| Cannot see generated PTX | Blind optimization | **Genchi Genbutsu**: Show the PTX |
| No SIMD vectorization feedback | Missed 4-16x speedups | **Visual Control**: Highlight scalar fallbacks |
| Register pressure invisible | Unexpected spills to slow memory | **Muda Elimination**: Flag spill instructions |
| Warp divergence hidden | 32x slowdown undetected | **Heijunka**: Visualize divergence paths |
| Memory coalescing unclear | 32x bandwidth waste | **Jidoka**: Auto-fail on uncoalesced access |

### 1.3 Design Goals

1. **Zero Runtime Overhead**: Analysis happens at compile/generation time.
2. **Actionable Output**: Every warning includes a fix suggestion.
3. **Diff-Friendly**: Track changes across commits.
4. **Interactive Exploration**: TUI mode for deep diving into code generation.
5. **Educational**: Teach optimization through visualization.

---

## 2. Architecture

### 2.1 Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      trueno-explain CLI/TUI                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   PTX        │  │   SIMD       │  │   wgpu       │           │
│  │   Analyzer   │  │   Analyzer   │  │   Analyzer   │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
│         │                 │                 │                    │
│         ▼                 ▼                 ▼                    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  Core Analysis Engine                    │    │
│  │               (implements trait Analyzer)                │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│         ┌────────────────────┼────────────────────┐             │
│         ▼                    ▼                    ▼             │
│  ┌────────────┐       ┌────────────┐       ┌────────────┐       │
│  │   Stdout   │       │   JSON     │       │   TUI      │       │
│  │  Renderer  │       │  Exporter  │       │ (Presentar)│       │
│  └────────────┘       └────────────┘       └────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Extensible Analyzer Trait

To ensure consistency across backends, all analyzers implement a common trait:

```rust
pub trait Analyzer {
    /// The type of IR being analyzed (e.g., "PTX", "x86 ASM", "WGSL")
    fn target_name(&self) -> &str;

    /// Analyze the provided code and return a structured report
    fn analyze(&self, code: &str) -> Result<AnalysisReport>;

    /// Identify specific performance bottlenecks (Muda)
    fn detect_muda(&self, code: &str) -> Vec<MudaWarning>;

    /// Estimate theoretical peak performance
    fn estimate_roofline(&self, analysis: &AnalysisReport) -> RooflineMetric;
}
```

---

## 3. CLI Interface

### 3.1 Command Structure

```bash
trueno-explain <SUBCOMMAND> [OPTIONS]

SUBCOMMANDS:
    ptx      Analyze PTX code generation
    simd     Analyze SIMD vectorization
    wgpu     Analyze wgpu/WGSL shaders
    tui      Launch interactive TUI exploration mode
    compare  Compare backends
    diff     Compare two analyses (git integration)
```

### 3.2 PTX Analysis

```bash
# Analyze a kernel
trueno-explain ptx --kernel q5k_gemm_ggml --size 1024x1024x4096

# Output register pressure
trueno-explain ptx --kernel gemm_tiled --registers

# Show memory access pattern
trueno-explain ptx --kernel softmax --memory-pattern
```

**Example Output:**

```
PTX Analysis: q5k_gemm_ggml
═══════════════════════════════════════════════════════════════

Configuration:
  Dimensions: M=1024, N=1, K=4096 (matvec mode)
  Tile size: 32
  Super-blocks: 16

Register Pressure:                                    [OK]
  ├── .reg .f32: 24 / 255 (9.4%)
  ├── .reg .b32: 18 / 255 (7.1%)
  ├── .reg .b64: 12 / 255 (4.7%)
  ├── .reg .pred: 4 / 7 (57.1%)
  └── Total: 58 registers → 100% occupancy possible

Memory Access Pattern:                                [OK]
  ├── Global loads: 847 (coalesced: 98.2%)
  ├── Global stores: 32 (coalesced: 100%)
  ├── Shared loads: 0
  └── Shared stores: 0

Muda (Waste) Detection:
  ⚠ Muda of Waiting: No shared memory tiling detected (consider for K>1024)
  ✓ No Muda of Transport (spills)
```

### 3.3 SIMD Analysis

```bash
# Analyze vectorization
trueno-explain simd --function vector_add --arch avx2
```

**Example Output:**

```
SIMD Analysis: dot_product (AVX2)
═══════════════════════════════════════════════════════════════

Vectorization Status:                                 [OK]
  ├── Loop vectorized: YES
  ├── Vector width: 8 × f32 (256-bit)
  ├── Unroll factor: 4
  └── Remainder handling: Scalar fallback

Performance Estimate:
  Speedup vs scalar: 7.2x (theoretical: 8x)
  Bottleneck: Memory bandwidth (not compute)

Recommendations:
  ⚠ Consider aligned loads (vmovaps) for 5-10% improvement
```

### 3.4 TUI Mode (Interactive Genchi Genbutsu)

Launch an interactive split-pane interface using `presentar`:

```bash
trueno-explain tui --kernel q5k_gemm_ggml
```

**Layout:**
*   **Left Pane**: Rust Source Code (with syntax highlighting)
*   **Center Pane**: Generated Assembly/PTX/WGSL (synced scrolling)
*   **Right Pane**: Analysis Dashboard (Register pressure, roofline plot, Muda list)
*   **Bottom Pane**: Log/Diagnostics

**Key Features:**
*   **Sync-Scroll**: Moving cursor in Rust source highlights corresponding assembly blocks.
*   **Heatmap**: Color-code assembly lines by instruction cost (latency).
*   **Filtering**: Toggle display of specific instruction types (e.g., "Show only memory ops").

---

## 4. Analysis Algorithms

### 4.1 Register Pressure Analysis [3]

Per Xiao & Feng [3], register pressure directly impacts GPU occupancy.

```rust
/// Register pressure analyzer for PTX
pub struct RegisterPressureAnalyzer;

impl Analyzer for RegisterPressureAnalyzer {
    fn analyze(&self, ptx: &str) -> Result<AnalysisReport> {
        // ... (implementation details)
    }

    fn detect_muda(&self, ptx: &str) -> Vec<MudaWarning> {
         // Detect "Muda of Transport" (Spills)
         if self.spill_count > 0 {
             vec![MudaWarning::Transport {
                 description: format!("{} spills detected", self.spill_count),
                 impact: "High latency local memory access",
             }]
         } else {
             vec![]
         }
    }
}
```

### 4.2 Memory Coalescing Analysis [5]

Per NVIDIA best practices [5], coalesced access is critical to eliminate the **Muda of Waiting**.

```rust
// Logic to detect uncoalesced patterns:
// 1. Parse 'ld.global' / 'st.global'
// 2. Analyze address calculation: base + tid * element_size
// 3. Flag patterns where stride != 1 (vector width)
```

### 4.3 Heijunka (Warp Divergence) [4]

Per Fung et al. [4], divergence causes serialization (imbalanced load).

```rust
// Logic to visualize Heijunka:
// 1. Construct Control Flow Graph (CFG)
// 2. Identify branches dependent on %tid (Thread ID)
// 3. Calculate "Divergence Factor": Max path length difference between branches
```

---

## 5. Output Formats

### 5.1 Text (Default)
Human-readable terminal output with ANSI colors.

### 5.2 JSON (CI/Tooling)
Structured data for regression tracking.

### 5.3 TUI (Interactive)
Rich terminal interface for exploration.

---

## 6. Integration Points

### 6.1 CI/CD Integration

```yaml
# GitHub Actions
- name: Analyze PTX
  run: |
    trueno-explain ptx --all-kernels --json > analysis.json
    trueno-explain diff --baseline main --json > diff.json

- name: Check for regressions
  run: |
    if trueno-explain diff --baseline main --fail-on-regression; then
      echo "No regressions"
    else
      echo "Performance regression detected!"
      exit 1
    fi
```

---

## 7. Academic Foundations

### Peer-Reviewed Citations

| # | Citation | Application |
|---|----------|-------------|
| [1] | T. Hoefler and R. Belli, "Scientific Benchmarking of Parallel Computing Systems," SC '15. | Reproducible methodology |
| [2] | S. Maleki et al., "An Evaluation of Vectorizing Compilers," PACT '11. | SIMD analysis |
| [3] | S. Xiao and W. Feng, "Inter-Block GPU Communication," IEEE IPDPS, 2010. | Register pressure |
| [4] | W. W. L. Fung et al., "Dynamic Warp Formation," MICRO '07. | Warp divergence |
| [5] | NVIDIA, "CUDA C++ Best Practices Guide," 2024. | Memory coalescing |
| [8] | S. Williams et al., "Roofline: An Insightful Visual Performance Model," CACM 52(4). | Roofline model |

---

## 8. Implementation Roadmap

### 8.1 Sprint Planning: TRUENO-EXPLAIN-001

**Sprint Goal**: PTX analyzer MVP with register pressure and memory analysis.

| ID | Task | Effort | Acceptance Criteria |
|----|------|--------|---------------------|
| TE-001 | CLI skeleton (clap) & Analyzer trait | 1 day | `trueno-explain --help` works |
| TE-002 | PTX parser | 3 days | Parse all trueno-gpu kernels |
| TE-003 | Register pressure analyzer | 2 days | Match nvcc output ±5% |
| TE-004 | Memory pattern analyzer | 2 days | Detect coalescing |
| TE-005 | Text output formatter | 1 day | Colored terminal output |
| TE-006 | JSON output | 1 day | Valid JSON schema |
| TE-007 | Integration tests | 2 days | 100% kernel coverage |

### 8.2 Sprint Planning: TRUENO-EXPLAIN-002 (TUI)

**Sprint Goal**: Interactive TUI mode.

| ID | Task | Effort | Acceptance Criteria |
|----|------|--------|---------------------|
| TE-008 | TUI Scaffold (Presentar) | 1 day | Split panes rendering |
| TE-009 | Source-ASM Sync | 3 days | Scrolling linkage |
| TE-010 | Analysis Widgets | 2 days | Charts/Lists rendering |

---

## 9. Falsification Checklist (100 Points)

### 9.1 CLI Foundation (10 points)
| ID | Test | Expected Result | Pass/Fail |
|----|------|-----------------|-----------|
| F001 | `trueno-explain --help` | Shows all subcommands | |
| F002 | `trueno-explain tui --help` | Shows TUI options | |
| F008 | `--json` flag produces valid JSON | Parse with `jq .` succeeds | |

### 9.2 PTX Analysis (15 points)
| ID | Test | Expected Result | Pass/Fail |
|----|------|-----------------|-----------|
| F011 | Analyze `vector_add` | Reports <20 registers | |
| F019 | Calculates occupancy | Matches CUDA calculator | |
| F020 | Warns when registers > 128 | Warning emitted | |

### 9.3 TUI & Interactive (20 points)
| ID | Test | Expected Result | Pass/Fail |
|----|------|-----------------|-----------|
| F026 | Launch TUI mode | Interface renders without panic | |
| F027 | Resize terminal | UI adapts responsive | |
| F028 | Scroll source pane | ASM pane scrolls in sync | |
| F029 | Toggle sidebar | Sidebar hides/shows | |
| F030 | Quit TUI (`q`) | Exits cleanly to shell | |

### 9.4 Memory Patterns (15 points)
| ID | Test | Expected Result | Pass/Fail |
|----|------|-----------------|-----------|
| F030 | Identifies coalesced pattern | tid*4 detected | |
| F034 | Warns on <80% coalescing | Warning emitted | |

### 9.5 SIMD Analysis (15 points)
| ID | Test | Expected Result | Pass/Fail |
|----|------|-----------------|-----------|
| F051 | Detects AVX2 instructions | `vmulps` counted | |
| F055 | Calculates vectorization ratio | > 0% for vectorized code | |

### 9.6 wgpu/WGSL (10 points)
| ID | Test | Expected Result | Pass/Fail |
|----|------|-----------------|-----------|
| F067 | Detects workgroup size | Parsed correctly | |

### 9.7 Diff Mode (15 points)
| ID | Test | Expected Result | Pass/Fail |
|----|------|-----------------|-----------|
| F086 | Diff two analyses | Delta shown | |
| F089 | Exit code 1 on regression | CI fails | |

---

## 10. Quality Gates

- [ ] All 100 falsification tests pass
- [ ] `cargo clippy` clean
- [ ] TUI tested on 80x24 and 4k terminals

---

## 11. PTX Bug Hunting (Probar-Style Static Analysis)

**Inspired by**: bashrs parser bug hunting methodology that found 25 bugs through rigorous edge case testing.

### 11.1 Motivation

PTX assembly is "scary" - invisible bugs cause silent correctness failures or 100x performance regressions. Unlike CPU code where bugs crash, GPU bugs often produce wrong results silently. We apply bashrs's proven bug hunting methodology to catch PTX bugs at generation time.

### 11.2 PTX Bug Classification

| Severity | Category | Examples | Impact |
|----------|----------|----------|--------|
| **P0 Critical** | Correctness | Missing barrier sync, wrong addressing | Silent wrong results |
| **P1 High** | Performance | Uncoalesced access, register spills | 10-100x slowdown |
| **P2 Medium** | Suboptimal | Missed optimizations, redundant ops | 2-10x slowdown |
| **False Positive** | Error handling | Malformed PTX accepted | Security/stability |

### 11.3 Bug Detection Patterns (from probar)

```rust
/// PTX bug classification
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PtxBugClass {
    /// P0: Shared memory accessed with 64-bit register (should be 32-bit)
    SharedMemU64Addressing,
    /// P0: Loop branches to END label instead of START
    LoopBranchToEnd,
    /// P0: Missing barrier sync between shared memory write and read (PARITY-114)
    MissingBarrierSync,
    /// P1: Accumulator not updated in-place in loop
    NonInPlaceLoopAccumulator,
    /// P2: Redundant register moves
    RedundantMoves,
    /// P2: Unoptimized memory access pattern
    UnoptimizedMemoryPattern,
    /// False Positive: Invalid PTX syntax accepted
    InvalidSyntaxAccepted,
    /// False Positive: Missing entry point not detected
    MissingEntryPoint,
}
```

### 11.4 Bug Hunting Test Categories

Following bashrs's `parser_bug_hunting.rs` methodology:

| Category | Test Pattern | Example PTX |
|----------|--------------|-------------|
| **Nested Structures** | Deep control flow nesting | `@%p1 bra; @%p2 bra; ...` |
| **Memory Addressing** | 32-bit vs 64-bit shared mem | `st.shared [%rd0]` vs `[%r0]` |
| **Barrier Sync** | Shared mem without barriers | `st.shared; ld.shared` (missing `bar.sync`) |
| **Loop Patterns** | Branch direction correctness | `bra loop_end` vs `bra loop_start` |
| **Register Allocation** | Spill detection | `.local` usage patterns |
| **Malformed Input** | Error rejection | Unclosed blocks, missing `ret` |

### 11.5 Bug Hunt Test Suite Structure

```rust
/// PTX Bug Hunting - Rigorous Edge Case Testing
/// (mirrors bashrs/rash/tests/parser_bug_hunting.rs)

#[test]
fn test_generate_ptx_bug_report() {
    let mut bugs_found = Vec::new();

    let edge_cases = vec![
        // (ptx, description, should_be_valid, expected_bug)
        (SHARED_MEM_U64_PTX, "Shared mem 64-bit addressing", false, Some(PtxBugClass::SharedMemU64Addressing)),
        (MISSING_BARRIER_PTX, "Missing barrier after st.shared", false, Some(PtxBugClass::MissingBarrierSync)),
        (LOOP_BRANCH_END_PTX, "Loop branches to end", false, Some(PtxBugClass::LoopBranchToEnd)),
        // ... 50+ edge cases
    ];

    for (ptx, desc, should_valid, expected) in edge_cases {
        let result = PtxBugAnalyzer::strict().analyze(ptx);
        // Assert bug detected or valid as expected
    }

    generate_bug_report(&bugs_found);
}
```

### 11.6 Probar-Style Coverage Tracking

```rust
/// Track PTX feature coverage (mirrors bashrs gui_coverage! macro)
#[test]
fn test_ptx_comprehensive_coverage() {
    let mut coverage = PtxCoverageTracker::new()
        .feature("barrier_sync")
        .feature("shared_memory")
        .feature("global_memory")
        .feature("register_allocation")
        .feature("loop_patterns")
        .feature("control_flow")
        .build();

    // Run all PTX test cases
    for kernel in all_kernels() {
        let ptx = kernel.emit_ptx();
        coverage.analyze(&ptx);
    }

    let report = coverage.generate_report();
    assert!(report.coverage >= 0.90, "PTX coverage must be ≥90%");
}
```

### 11.7 Determinism Verification

```rust
/// Verify PTX analysis is deterministic (mirrors bashrs test_parser_determinism)
#[test]
fn test_ptx_analysis_determinism() {
    let kernels = [
        GemmKernel::naive(64, 64, 64),
        GemmKernel::tiled(64, 64, 64, 16),
        SoftmaxKernel::new(1024),
        Q5KKernel::new(64, 64, 256),
    ];

    for kernel in &kernels {
        let ptx = kernel.emit_ptx();
        let result1 = PtxAnalyzer::new().analyze(&ptx);
        let result2 = PtxAnalyzer::new().analyze(&ptx);
        let result3 = PtxAnalyzer::new().analyze(&ptx);

        assert_eq!(result1, result2);
        assert_eq!(result2, result3);
    }
}
```

### 11.8 Implementation Sprint: TRUENO-EXPLAIN-003 (Bug Hunting)

| ID | Task | Effort | Acceptance Criteria |
|----|------|--------|---------------------|
| TE-020 | Add `PtxBugClass` enum | 0.5 day | All 8 bug classes defined |
| TE-021 | Implement `MissingBarrierSync` detection | 1 day | Catches PARITY-114 pattern |
| TE-022 | Implement `SharedMemU64Addressing` detection | 0.5 day | Detects `[%rd*]` in shared ops |
| TE-023 | Implement `LoopBranchToEnd` detection | 0.5 day | Detects wrong branch targets |
| TE-024 | Create `ptx_bug_hunting.rs` test suite | 2 days | 50+ edge cases tested |
| TE-025 | Add coverage tracking | 1 day | Reports feature coverage |
| TE-026 | Add determinism verification | 0.5 day | All kernels verified |
| TE-027 | Integration with TUI | 1 day | Bugs shown in TUI dashboard |

### 11.9 Bug Hunting Falsification Tests

| ID | Test | Expected Result | Pass/Fail |
|----|------|-----------------|-----------|
| F101 | Detect `st.shared [%rd0]` | `SharedMemU64Addressing` bug reported | |
| F102 | Detect missing `bar.sync` | `MissingBarrierSync` bug reported | |
| F103 | Detect `bra loop_end` in loop | `LoopBranchToEnd` bug reported | |
| F104 | Valid PTX passes | No bugs reported | |
| F105 | Unclosed block rejected | `InvalidSyntax` error | |
| F106 | Missing `.entry` detected | `MissingEntryPoint` bug reported | |
| F107 | Coverage ≥90% | Coverage report shows ≥90% | |
| F108 | Determinism verified | 3 runs produce identical results | |

### 11.10 Bug Report Format

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         PTX BUG HUNTING REPORT                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

Kernel: gemm_tiled (M=64, N=64, K=64, tile=16)
PTX Lines Analyzed: 847

P0 CRITICAL BUGS: 1
──────────────────
  BUG-001: Missing barrier synchronization
    Line 234: st.shared.f32 [%r5], %f0
    Line 238: ld.shared.f32 %f1, [%r6]   ← No bar.sync between!
    Impact: Race condition, silent wrong results
    Fix: Add `bar.sync 0;` between st.shared and ld.shared

P1 HIGH BUGS: 0
P2 MEDIUM BUGS: 2
─────────────────
  BUG-002: Suboptimal register usage
    58 registers used, 100% occupancy possible

  BUG-003: Uncoalesced global load pattern
    Line 45: ld.global.f32 %f0, [%rd0 + %r1*512]
    Impact: 32x bandwidth reduction
    Fix: Transpose access pattern

FALSE POSITIVES DETECTED: 0

SUMMARY
═══════
  Total Bugs: 3
  P0 Critical: 1 ← BLOCKS RELEASE
  P1 High: 0
  P2 Medium: 2
```