# PTX Debugger Specification

**Version**: 1.6
**Date**: 2026-01-05
**Status**: DRAFT - Design Phase
**Priority**: P0 - Critical for GPU Kernel Development
**Crate**: `trueno-ptx-debug` (proposed)
**Philosophy**: Pure Rust PTX Analysis - Zero CUDA SDK Dependency

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.6 | 2026-01-05 | Batuta Team | Documented membar.gl failure in complex LZ4 kernel; Kernel Fission mandatory |
| 1.5 | 2026-01-05 | Batuta Team | Added Kernel Fission as robust workaround for F082 where membar.cta fails |
| 1.4 | 2026-01-05 | Claude Opus | Updated F082 test and detection method with LVB-003 workaround notes (membar.cta partial fix) |
| 1.3 | 2026-01-05 | Claude Opus | Added Section 2.5: Workaround attempts (LVB series), documented membar.cta context-dependent behavior |
| 1.2 | 2026-01-05 | Claude Opus | Added ComputedAddrFromLoaded bug (Section 2.4), F082 test, isolation test results |
| 1.1 | 2026-01-05 | Batuta Team | Added safe ring buffer protocol to prevent OOB debug writes |
| 1.0 | 2026-01-05 | Batuta Team | Initial specification based on LZ4 kernel debugging experience |

---

## 1. Executive Summary

This specification defines **trueno-ptx-debug**, a pure Rust PTX debugging and analysis tool built on the batuta stack. The tool addresses critical PTX debugging challenges discovered during GPU LZ4 kernel development (Issue #67, Address 0x1 crashes), providing systematic detection, analysis, and prevention of PTX-level bugs.

### Core Thesis

> **Requirement**: PTX bugs manifest in ways that CUDA error codes cannot diagnose. A purpose-built static analyzer with runtime instrumentation, integrated with renacer's tracing and certeza's falsification methodology, enables systematic PTX bug detection before GPU execution.

### Key Innovations

1. **Static PTX Analysis** - Detect bug patterns without GPU execution
2. **Runtime Debug Instrumentation** - Automatic debug buffer injection
3. **Syscall-Level GPU Tracing** - Integrate with renacer for CUDA driver call analysis
4. **Popperian Falsification Framework** - 100-point systematic bug class elimination
5. **FKR Integration** - Pixel-level regression testing via jugar-probar

---

## 2. Problem Analysis

### 2.1 PTX Bug Taxonomy (Empirical)

Based on extensive debugging of the LZ4 compression kernel (Issue #67, 2026-01-05), the following bug taxonomy has been established:

| Bug Class | Severity | Detection Method | Root Cause | Citation |
|-----------|----------|------------------|------------|----------|
| **GenericAddressCorruption** | CRITICAL | Static + Runtime | cvta.shared creates 64-bit generic addr that SASS clobbers | [Nickolls et al. 2008] |
| **SharedMemU64Addressing** | HIGH | Static Analysis | Using u64 for shared memory addresses (should use 32-bit offset) | [NVIDIA PTX ISA 8.0] |
| **MissingDirectShared** | HIGH | Static Analysis | Using generic ld/st instead of ld.shared/st.shared | [Volkov 2010] |
| **MissingBarrierSync** | HIGH | Control Flow | Missing `bar.sync` between shared memory writes and reads | [Sørensen et al. 2016] |
| **RegisterTypeInvariant** | MEDIUM | Type Analysis | Wrong register type for operation (e.g., f32 vs u32) | [PTX ISA 8.0] |
| **UnalignedMemoryAccess** | HIGH | Offset Analysis | Non-aligned global/shared memory access | [Harris 2013] |
| **DataDependentStore (F081)** | ~~CRITICAL~~ **FALSIFIED** | Popperian Refutation | **Hypothesis refuted**: `ld.shared→st.global` succeeds with 0xBEEFCAFE on sm_89. Original crash was F082/F021. | [Batuta Team 2026-01-05] |
| **ComputedAddrFromLoaded (F082)** | CRITICAL | Data Flow | Address computed from ld.shared value causes store crash | [Discovered 2026-01-05] |
| **SequentialCodeSensitivity** | HIGH | Binary Search | Adding one instruction causes crash | [ptxas JIT bug] |
| **LoopCvtaShared** | HIGH | Loop Analysis | cvta.shared inside loop causes register pressure issues | [Volkov 2010] |
| **IncompatibleAddressSpace** | MEDIUM | Static Analysis | Mismatched address space qualifiers | [NVIDIA CUDA C++ Guide] |

### 2.2 CUDA Error Code Insufficiency

The CUDA driver API returns error codes that are insufficient for root cause analysis:

| CUDA Error | Code | Actual Causes (Observed) | Diagnostic Value |
|------------|------|--------------------------|------------------|
| `CUDA_ERROR_UNKNOWN` | 716 | Data-dependent store, register clobber, SASS scheduling | LOW |
| `CUDA_ERROR_INVALID_PTX` | 218 | Syntax error, invalid atomics, address space mismatch | MEDIUM |
| `CUDA_ERROR_LAUNCH_FAILED` | 719 | Any runtime crash | LOW |
| `CUDA_ERROR_ILLEGAL_ADDRESS` | 700 | Null pointer, OOB access, Address 0x1 | MEDIUM |

**Key Insight**: CUDA errors are symptoms, not diagnoses. A PTX-level analyzer must provide causal analysis [Sørensen & Gopalakrishnan 2016].

### 2.3 The "Loaded Value" Bug Pattern (F081) - FALSIFIED

**Status**: **EMPIRICALLY REFUTED** on sm_89 (RTX 4090) as of 2026-01-05.

**Popperian Falsification Applied**: Following Karl Popper's methodology, we attempted to refute the hypothesis through rigorous experimental testing rather than seeking confirmation.

Original Hypothesis:
```ptx
// HYPOTHESIS: Any store using a value derived from ld.shared crashes
ld.shared.u32 %r_loaded, [%r_src_addr];
st.global.u32 [%r_any_addr], %r_loaded;         // EXPECTED CRASH
```

**Experimental Protocol** (`trueno-gpu/tests/f081_minimal_crash.rs`):

| Test | Pattern | Expected | Actual |
|------|---------|----------|--------|
| `f081_baseline_immediate_to_global` | `mov → st.global` | ✅ Pass | ✅ Pass |
| `f081_global_to_global` | `ld.global → st.global` | ✅ Pass | ✅ Pass |
| `f081_shared_to_global_simple` | `ld.shared → st.global` | ❌ Crash 716 | **✅ Pass** |
| `f081_workaround_shfl_launder` | `shfl → st.global` | ✅ Pass | ✅ Pass |

**Critical PTX That Was Expected to Crash (But Didn't)**:
```ptx
ld.shared.u32 %r5, [%r4];     // Load from shared memory
st.global.u32 [%rd0], %r5;    // Store to global - WAS expected to crash
```

**Result**: Kernel executed successfully, returned correct value **0xBEEFCAFE**.

**Falsification Verdict**:
- **Hypothesis F081 is REFUTED.** The pattern `ld.shared → st.global` does NOT inherently cause CUDA_ERROR_UNKNOWN (716).
- The original crash during LZ4 development had a **different root cause** - likely F082 (Computed Address), Generic Address Corruption (F021), or ptxas JIT scheduling bugs.

**Action**: F081 is **deprecated** from active blocking criteria. Retained as informational warning only (0 points).

### 2.4 The "Computed Address from Loaded Value" Bug (NEW - 2026-01-05)

A more specific variant discovered through systematic isolation testing:

```ptx
// CRASHES: Using loaded value in ADDRESS computation
ld.shared.u32 %r_pos, [%r_state_off];        // Load position from shared
add.u32 %r_addr, %r_base, %r_pos;            // Compute address using loaded value
st.shared.u32 [%r_addr], %r_const;           // CRASH - address is "toxic"

// WORKS: Using CONSTANT in address computation
mov.u32 %r_fixed_off, 100;                   // Constant offset
add.u32 %r_addr, %r_base, %r_fixed_off;      // Compute address from constant
st.shared.u32 [%r_addr], %r_const;           // OK - address is "clean"
```

**Systematic Isolation Results (2026-01-05)**:

| Test | Address Source | Value Source | Result |
|------|----------------|--------------|--------|
| Load from fixed addr, store constant | Fixed | Constant | PASS |
| Load from fixed addr, store loaded | Fixed | Loaded | CRASH |
| Load from computed addr (constant), store constant | Computed (const) | Constant | PASS |
| Load from computed addr (constant), store loaded | Computed (const) | Loaded | CRASH |
| Compute addr from loaded value, store constant | Computed (loaded) | Constant | **CRASH** |
| Compute addr from loaded value, store loaded | Computed (loaded) | Loaded | CRASH |

**Key Insight**: The bug triggers when:
1. A value is loaded from shared memory (`in_pos = ld.shared [state]`)
2. That value is used in address arithmetic (`addr = base + in_pos`)
3. ANY store is performed (even with constant value to the computed address)

This is MORE restrictive than the original pattern - using loaded values in address computation for stores is also forbidden.

**Detection Rule (F083)**:
```
IF ld.shared %r_val, [addr]
AND add/mul/... %r_addr, ..., %r_val  // r_val used in address computation
AND st.XXX [%r_addr], ...             // Any store to that address
THEN BUG: ComputedAddressFromLoadedValue
```

### 2.5 Workaround Attempts (LVB Series) - 2026-01-05

The following workarounds have been systematically tested for the ComputedAddrFromLoaded bug:

| ID | Workaround | Simple Kernel | Complex Kernel | Notes |
|----|------------|---------------|----------------|-------|
| LVB-001 | `bar.sync 0` before store | FAILED | FAILED | Thread barrier doesn't help |
| LVB-002 | `mov` to fresh register | FAILED | FAILED | Breaking dependency chain doesn't help |
| LVB-003 | `membar.cta` before store | **PASS** | **FAIL** | Works in isolation, fails with loops |
| LVB-004 | `.volatile` qualifier | NOT TESTED | NOT TESTED | Likely same limitation as LVB-003 |
| LVB-005 | `membar.gl` before store | **PASS** | **FAIL** | Stronger fence still fails in LZ4 |

**Critical Finding (LVB-005 - membar.gl)**:

Even `membar.gl` (Global Memory Fence), which forces visibility across the entire device, failed to prevent the crash in the full LZ4 kernel.

**Conclusion**: The JIT compiler likely reorders instructions *across* the fence in ways that violate the dependency requirement in complex control flow graphs, or the issue involves a hardware hazard that fences cannot resolve when register pressure is high. **Kernel Fission is the only robust fix.**

**Hypothesis**: The ptxas JIT compiler's SASS code generation differs based on:
1. Total kernel instruction count
2. Register pressure from earlier loops
3. Instruction scheduling optimization decisions

The same PTX pattern that works in isolation fails when embedded in a larger kernel, suggesting the bug is in SASS generation rather than PTX semantics.

**Robust Workaround (LVB-005 - Kernel Fission)**:

Since the ptxas JIT bug is triggered by kernel complexity, the **Kernel Fission** approach splits the monolithic kernel into multiple smaller kernels, each simple enough to avoid the bug:

```
// BEFORE: Monolithic kernel (CRASHES)
┌─────────────────────────────────────────┐
│  LZ4 Compression Kernel                 │
│  ├── Load Loop (4KB page → shared)      │
│  ├── Hash Init Loop (8KB table)         │
│  └── Compress Loop (computed stores) ←CRASH
└─────────────────────────────────────────┘

// AFTER: Kernel Fission (WORKS)
┌─────────────────────────────────────────┐
│  Kernel 1: Load Page to Shared          │
│  ├── Load Loop only                     │
│  └── No computed-addr-from-loaded       │ → PASS
├─────────────────────────────────────────┤
│  Kernel 2: Initialize Hash Table        │
│  ├── Hash Init Loop only                │
│  └── No computed-addr-from-loaded       │ → PASS
├─────────────────────────────────────────┤
│  Kernel 3: Compress with Hash Lookup    │
│  ├── Simple kernel (membar.cta works)   │
│  └── Computed stores with membar.cta    │ → PASS (simple kernel)
└─────────────────────────────────────────┘
```

**Implementation Strategy**:

1. **Identify kernel phases**: Load → Init → Process
2. **Split at phase boundaries**: Each phase becomes a separate kernel
3. **Use persistent shared memory** OR **global memory staging**:
   - Option A: Use CUDA Dynamic Parallelism to share context
   - Option B: Write intermediate state to global memory between kernels
4. **Add membar.cta in simple kernels**: Now that each kernel is simple, membar.cta works

**Trade-offs**:

| Aspect | Monolithic | Kernel Fission |
|--------|------------|----------------|
| Kernel launches | 1 | 3+ |
| Shared memory | Reused | Global staging or CDP |
| Code complexity | Lower | Higher |
| ptxas JIT bug | CRASHES | **WORKS** |
| Performance | Optimal (if it worked) | ~10-20% overhead |

**Recommendation**: Use Kernel Fission for complex kernels where membar.cta fails. The performance overhead is acceptable compared to a non-functional kernel.

**Recommended Action**: File NVIDIA bug report with minimal reproducer (simple vs complex kernel demonstrating context-dependent failure).

---

## 3. Architecture

### 3.1 System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     trueno-ptx-debug                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │  Static Analyzer │  │ Debug Injector  │  │ Runtime Tracer     │  │
│  │  (PTX AST)       │  │ (Marker Gen)    │  │ (renacer bridge)   │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬───────────┘  │
│           │                    │                     │              │
│           ▼                    ▼                     ▼              │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    Analysis Engine                               ││
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────────┐││
│  │  │ Type     │ │ Control  │ │ Data     │ │ Address Space        │││
│  │  │ Checker  │ │ Flow     │ │ Flow     │ │ Validator            │││
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────────────────┘││
│  └─────────────────────────────────────────────────────────────────┘│
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                   Falsification Engine                           ││
│  │           (100-Point Popperian Framework)                        ││
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────────┐││
│  │  │ Bug Class│ │ Test     │ │ Evidence │ │ Confidence           │││
│  │  │ Registry │ │ Generator│ │ Collector│ │ Calculator           │││
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────────────────┘││
│  └─────────────────────────────────────────────────────────────────┘│
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    Output Generators                             ││
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────────┐││
│  │  │ Report   │ │ FKR Test │ │ OTLP     │ │ Patch                │││
│  │  │ (HTML)   │ │ (probar) │ │ (renacer)│ │ Suggestions          │││
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────────────────┘││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│    renacer      │  │    certeza      │  │  jugar-probar   │
│ (syscall trace) │  │ (falsification) │  │ (FKR testing)   │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### 3.2 Integration with Batuta Stack

| Component | Integration Point | Purpose |
|-----------|------------------|---------|
| **renacer** | OTLP spans + syscall tracing | Trace cuModuleLoad, cuLaunchKernel, memory operations |
| **certeza** | Tiered TDD workflow | Tier 1/2/3 verification of PTX analysis correctness |
| **jugar-probar** | FKR test generation | Generate pixel-level regression tests for PTX output |
| **batuta** | Orchestration | Coordinate multi-tool analysis pipelines |
| **trueno** | SIMD statistics | Accelerated analysis of large PTX modules |

### 3.3 Module Structure

```
trueno-ptx-debug/
├── src/
│   ├── lib.rs                 # Public API
│   ├── parser/                # PTX lexer and parser
│   │   ├── mod.rs
│   │   ├── lexer.rs           # PTX tokenization
│   │   ├── ast.rs             # PTX abstract syntax tree
│   │   └── types.rs           # PTX type system
│   ├── analyzer/              # Static analysis passes
│   │   ├── mod.rs
│   │   ├── type_checker.rs    # Register type validation
│   │   ├── control_flow.rs    # CFG construction, barrier analysis
│   │   ├── data_flow.rs       # Def-use chains, liveness
│   │   ├── address_space.rs   # Address space validation
│   │   └── memory_model.rs    # Shared memory access patterns
│   ├── bugs/                  # Bug class definitions
│   │   ├── mod.rs
│   │   ├── generic_address.rs
│   │   ├── barrier_sync.rs
│   │   ├── loaded_value.rs    # The critical pattern
│   │   └── registry.rs        # Bug class registry
│   ├── instrumentation/       # Debug marker injection
│   │   ├── mod.rs
│   │   ├── debug_buffer.rs    # Atomic debug buffer protocol
│   │   ├── marker_gen.rs      # Debug marker generation
│   │   └── ptx_transform.rs   # PTX-to-PTX transformation
│   ├── runtime/               # Runtime analysis
│   │   ├── mod.rs
│   │   ├── renacer_bridge.rs  # renacer integration
│   │   ├── cuda_trace.rs      # CUDA driver call analysis
│   │   └── crash_analysis.rs  # Post-mortem analysis
│   ├── falsification/         # Popperian framework
│   │   ├── mod.rs
│   │   ├── framework.rs       # 100-point falsification
│   │   ├── evidence.rs        # Evidence collection
│   │   └── confidence.rs      # Confidence calculation
│   └── output/                # Report generation
│       ├── mod.rs
│       ├── html_report.rs
│       ├── fkr_generator.rs   # jugar-probar test generation
│       └── patch_suggest.rs   # Suggested PTX fixes
└── tests/
    ├── parser_tests.rs
    ├── analyzer_tests.rs
    └── fkr/                    # Falsification kernel regression
        └── lz4_bugs.rs
```

---

## 4. Static Analysis Engine

### 4.1 PTX Parser

The parser constructs a typed AST from PTX source:

```rust
/// PTX Abstract Syntax Tree
pub struct PtxModule {
    pub version: (u8, u8),           // e.g., (8, 0)
    pub target: SmTarget,            // e.g., sm_70
    pub address_size: u8,            // 32 or 64
    pub globals: Vec<GlobalDecl>,
    pub kernels: Vec<KernelDef>,
    pub functions: Vec<FunctionDef>,
}

pub struct KernelDef {
    pub name: String,
    pub params: Vec<Param>,
    pub registers: Vec<RegisterDecl>,
    pub shared_mem: Vec<SharedMemDecl>,
    pub body: Vec<Statement>,
}

pub enum Statement {
    Label(String),
    Instruction(Instruction),
    Directive(Directive),
}

pub struct Instruction {
    pub opcode: Opcode,
    pub modifiers: Vec<Modifier>,
    pub operands: Vec<Operand>,
    pub predicate: Option<Predicate>,
    pub location: SourceLocation,
}
```

### 4.2 Type Checker

Validates register types match operation requirements:

```rust
/// Type checking pass
pub struct TypeChecker {
    register_types: HashMap<String, PtxType>,
    errors: Vec<TypeError>,
}

impl TypeChecker {
    /// Check instruction operand types match expected types
    pub fn check_instruction(&mut self, instr: &Instruction) -> Result<(), TypeError> {
        match &instr.opcode {
            Opcode::Ld { space, ty } => {
                // Verify destination register type matches load type
                let dest = &instr.operands[0];
                self.verify_type(dest, ty)?;

                // Verify address operand is correct size
                let addr = &instr.operands[1];
                self.verify_address_type(addr, space)?;
            }
            Opcode::St { space, ty } => {
                // Similar checks for store
            }
            // ... other opcodes
        }
        Ok(())
    }
}
```

### 4.3 Control Flow Analyzer

Constructs CFG and validates barrier synchronization:

```rust
/// Control flow graph node
pub struct CfgNode {
    pub id: NodeId,
    pub label: Option<String>,
    pub instructions: Vec<Instruction>,
    pub successors: Vec<NodeId>,
    pub predecessors: Vec<NodeId>,
}

/// Barrier safety analysis (per Sørensen et al. 2016)
pub struct BarrierAnalyzer {
    cfg: ControlFlowGraph,
}

impl BarrierAnalyzer {
    /// Detect missing barriers between shared memory write and read
    ///
    /// Implements the barrier divergence freedom analysis from
    /// [Sørensen & Gopalakrishnan 2016]
    pub fn analyze(&self) -> Vec<BarrierViolation> {
        let mut violations = Vec::new();

        for node in self.cfg.nodes() {
            for instr in &node.instructions {
                if let Some(write) = self.is_shared_write(instr) {
                    // Check all paths from this write to any read
                    let reachable_reads = self.find_reachable_shared_reads(node.id);
                    for (read_node, read_instr) in reachable_reads {
                        if !self.barrier_on_path(node.id, read_node) {
                            violations.push(BarrierViolation {
                                write_loc: instr.location.clone(),
                                read_loc: read_instr.location.clone(),
                                severity: Severity::High,
                            });
                        }
                    }
                }
            }
        }

        violations
    }
}
```

### 4.4 Data Flow Analyzer

Tracks value propagation to detect "loaded value" bug:

```rust
/// Data flow analysis for loaded value tracking
pub struct DataFlowAnalyzer {
    def_use_chains: HashMap<String, Vec<UsePoint>>,
    value_sources: HashMap<String, ValueSource>,
}

#[derive(Clone, Debug)]
pub enum ValueSource {
    /// Value came from a load instruction
    Load { space: AddressSpace, location: SourceLocation },
    /// Value came from a constant/immediate
    Constant(i64),
    /// Value came from computation
    Computed { inputs: Vec<String> },
    /// Value came from parameter
    Parameter(String),
}

impl DataFlowAnalyzer {
    /// Detect the "loaded value" bug pattern
    ///
    /// Pattern: ld.shared -> computation -> st.XXX crashes
    /// This appears to be a ptxas JIT bug affecting certain
    /// instruction sequences.
    pub fn detect_loaded_value_bug(&self) -> Vec<LoadedValueBug> {
        let mut bugs = Vec::new();

        for (reg, source) in &self.value_sources {
            if let ValueSource::Load { space: AddressSpace::Shared, location } = source {
                // Find all stores that use this register
                for use_point in self.def_use_chains.get(reg).unwrap_or(&vec![]) {
                    if use_point.is_store_data_operand() {
                        bugs.push(LoadedValueBug {
                            load_location: location.clone(),
                            store_location: use_point.location.clone(),
                            register: reg.clone(),
                            severity: Severity::Critical,
                            mitigation: "Use constant value or pre-computed address".into(),
                        });
                    }
                }
            }
        }

        bugs
    }

    /// Detect "computed address from loaded value" bug pattern (NEW - 2026-01-05)
    ///
    /// Pattern: ld.shared %r_val -> add %r_addr, base, %r_val -> st.XXX [%r_addr]
    /// Even storing a CONSTANT to an address computed from a loaded value crashes.
    ///
    /// This is MORE restrictive than detect_loaded_value_bug - the loaded value
    /// doesn't need to be the store DATA, just part of the address computation.
    ///
    /// WORKAROUND NOTE (LVB-003): membar.cta before the store is a PARTIAL fix:
    /// - Works in simple kernels (no preceding loops)
    /// - FAILS in complex kernels (with load loop + hash init + compress loop)
    /// The workaround effectiveness is context-dependent on kernel complexity.
    pub fn detect_computed_addr_from_loaded(&self) -> Vec<ComputedAddrFromLoadedBug> {
        let mut bugs = Vec::new();

        // Track which registers come from ld.shared
        let mut shared_loaded_regs: HashSet<String> = HashSet::new();
        for (reg, source) in &self.value_sources {
            if matches!(source, ValueSource::Load { space: AddressSpace::Shared, .. }) {
                shared_loaded_regs.insert(reg.clone());
            }
        }

        // Track registers computed from shared-loaded registers
        let mut tainted_regs: HashSet<String> = shared_loaded_regs.clone();
        let mut changed = true;
        while changed {
            changed = false;
            for (reg, source) in &self.value_sources {
                if let ValueSource::Computed { inputs } = source {
                    if !tainted_regs.contains(reg) {
                        if inputs.iter().any(|i| tainted_regs.contains(i)) {
                            tainted_regs.insert(reg.clone());
                            changed = true;
                        }
                    }
                }
            }
        }

        // Find stores where ADDRESS is computed from tainted register
        for (reg, source) in &self.value_sources {
            if tainted_regs.contains(reg) {
                for use_point in self.def_use_chains.get(reg).unwrap_or(&vec![]) {
                    // Check if this register is used in ADDRESS position of a store
                    if use_point.is_store_address_operand() {
                        let load_loc = self.find_original_load_location(reg, &shared_loaded_regs);
                        bugs.push(ComputedAddrFromLoadedBug {
                            load_location: load_loc.unwrap_or_default(),
                            addr_computation_location: use_point.location.clone(),
                            tainted_register: reg.clone(),
                            severity: Severity::Critical,
                            mitigation: "Use constant-only address computation, try membar.cta (partial), or use Kernel Fission (split kernel)".into(),
                        });
                    }
                }
            }
        }

        bugs
    }
}

/// Bug: Address computed from ld.shared value causes store crash
#[derive(Clone, Debug)]
pub struct ComputedAddrFromLoadedBug {
    pub load_location: SourceLocation,
    pub addr_computation_location: SourceLocation,
    pub tainted_register: String,
    pub severity: Severity,
    pub mitigation: String,
}
```

### 4.5 Address Space Validator

Validates correct address space usage:

```rust
/// Address space validation
pub struct AddressSpaceValidator {
    shared_base_regs: HashSet<String>,  // Registers holding cvta.shared results
}

impl AddressSpaceValidator {
    /// Detect generic addressing of shared memory (bug pattern)
    ///
    /// WRONG: cvta.shared.u64 %rd, smem; ld.u32 [%rd]
    /// RIGHT: ld.shared.u32 [smem_offset]
    pub fn detect_generic_shared_access(&self, module: &PtxModule) -> Vec<GenericSharedBug> {
        let mut bugs = Vec::new();

        // Track cvta.shared results
        for kernel in &module.kernels {
            let mut generic_shared_regs: HashSet<String> = HashSet::new();

            for stmt in &kernel.body {
                if let Statement::Instruction(instr) = stmt {
                    // Track cvta.shared destinations
                    if matches!(instr.opcode, Opcode::Cvta { space: AddressSpace::Shared, .. }) {
                        if let Some(dest) = instr.operands.first() {
                            generic_shared_regs.insert(dest.to_string());
                        }
                    }

                    // Detect generic ld/st using tracked registers
                    if matches!(instr.opcode, Opcode::Ld { space: AddressSpace::Generic, .. }) {
                        if let Some(addr) = instr.operands.get(1) {
                            if self.uses_generic_shared_reg(addr, &generic_shared_regs) {
                                bugs.push(GenericSharedBug {
                                    location: instr.location.clone(),
                                    instruction: instr.clone(),
                                    severity: Severity::Critical,
                                    fix: "Use ld.shared with 32-bit offset instead".into(),
                                });
                            }
                        }
                    }
                }
            }
        }

        bugs
    }
}
```

---

## 5. Debug Instrumentation

### 5.1 Debug Buffer Protocol

Every instrumented kernel follows this protocol:

```rust
/// Debug buffer memory layout
///
/// debug_buf[0]: atomic counter (starts at 0)
/// debug_buf[1..N]: marker values
///
/// Each emit_debug_marker call:
/// 1. Atomically increments counter
/// 2. Writes marker to next slot
pub struct DebugBufferProtocol {
    pub buffer_size: usize,        // Default: 4096 u32 values
    pub counter_offset: usize,     // Always 0
    pub data_offset: usize,        // Always 1
}

impl DebugBufferProtocol {
    /// Generate PTX for debug marker emission
    ///
    /// ENHANCEMENT (v1.1): Implements ring buffer with power-of-2 masking
    /// to prevent OOB writes which would cause secondary Error 700s.
    pub fn emit_marker_ptx(&self, marker_value: u32) -> String {
        let mask = self.buffer_size - 1; // Assumes buffer_size is power of 2
        format!(r#"
    // Debug marker: 0x{:08X}
    atom.global.add.u32 %r_debug_idx, [%rd_debug_buf];
    and.b32 %r_debug_idx, %r_debug_idx, {}; // Ring buffer mask
    shl.b32 %r_debug_off, %r_debug_idx, 2;  // idx * 4
    add.u64 %rd_debug_addr, %rd_debug_buf, 4;  // skip counter
    add.u64 %rd_debug_addr, %rd_debug_addr, %r_debug_off;
    mov.u32 %r_debug_val, {};
    st.global.u32 [%rd_debug_addr], %r_debug_val;
"#, marker_value, mask, marker_value)
    }
}
```

### 5.2 Automatic Instrumentation

```rust
/// PTX transformation for debug instrumentation
pub struct PtxInstrumenter {
    marker_counter: u32,
    instrumentation_points: Vec<InstrumentationPoint>,
}

pub enum InstrumentationPoint {
    /// Before every shared memory store
    BeforeSharedStore,
    /// Before every global memory store
    BeforeGlobalStore,
    /// At loop headers
    LoopHeader,
    /// At branch targets
    BranchTarget,
    /// Before barrier synchronization
    BeforeBarrier,
    /// Custom location
    Custom(SourceLocation),
}

impl PtxInstrumenter {
    /// Transform PTX to add debug instrumentation
    pub fn instrument(&mut self, module: &PtxModule) -> PtxModule {
        let mut instrumented = module.clone();

        for kernel in &mut instrumented.kernels {
            // Add debug buffer parameter
            kernel.params.push(Param {
                name: "debug_buf".into(),
                ty: PtxType::Ptr(AddressSpace::Global, Box::new(PtxType::U32)),
            });

            // Add debug registers
            kernel.registers.extend(vec![
                RegisterDecl { name: "%rd_debug_buf".into(), ty: PtxType::B64 },
                RegisterDecl { name: "%r_debug_idx".into(), ty: PtxType::U32 },
                RegisterDecl { name: "%r_debug_off".into(), ty: PtxType::U32 },
                RegisterDecl { name: "%rd_debug_addr".into(), ty: PtxType::B64 },
                RegisterDecl { name: "%r_debug_val".into(), ty: PtxType::U32 },
            ]);

            // Insert markers at instrumentation points
            let mut new_body = Vec::new();
            for stmt in &kernel.body {
                if self.should_instrument(&stmt) {
                    let marker = self.next_marker();
                    new_body.push(Statement::Comment(format!("DEBUG: marker 0x{:08X}", marker)));
                    new_body.extend(self.generate_marker_instructions(marker));
                }
                new_body.push(stmt.clone());
            }
            kernel.body = new_body;
        }

        instrumented
    }
}
```

---

## 6. Runtime Analysis (renacer Integration)

### 6.1 CUDA Syscall Tracing

Integrate with renacer to trace CUDA driver calls:

```rust
/// renacer bridge for CUDA driver tracing
pub struct RenacerBridge {
    otlp_endpoint: Option<String>,
    trace_filter: TraceFilter,
}

/// CUDA-relevant syscalls to trace
pub enum CudaSyscall {
    /// cuModuleLoad - PTX compilation
    ModuleLoad { ptx_size: usize, result: i32 },
    /// cuModuleGetFunction - Kernel lookup
    GetFunction { name: String, result: i32 },
    /// cuLaunchKernel - Kernel execution
    LaunchKernel {
        function: String,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: usize,
        result: i32,
    },
    /// cuMemcpyHtoD - Host to device
    MemcpyHtoD { size: usize, result: i32 },
    /// cuMemcpyDtoH - Device to host (debug buffer readback)
    MemcpyDtoH { size: usize, result: i32 },
    /// cuCtxSynchronize - Synchronization
    CtxSynchronize { result: i32 },
}

impl RenacerBridge {
    /// Start tracing with renacer
    pub fn start_trace(&self, command: &str) -> Result<TraceSession, Error> {
        // Build renacer command with CUDA-relevant filters
        let renacer_args = vec![
            "--otlp-endpoint", &self.otlp_endpoint.as_ref().unwrap_or(&"".to_string()),
            "--otlp-service-name", "trueno-ptx-debug",
            "-e", "trace=file,mmap,ioctl",  // CUDA uses ioctl for driver calls
            "-T",  // Timing mode
            "--", command,
        ];

        // Execute renacer
        let session = renacer::trace(renacer_args)?;
        Ok(session)
    }

    /// Analyze trace for CUDA errors
    pub fn analyze_trace(&self, session: &TraceSession) -> Vec<CudaError> {
        let mut errors = Vec::new();

        for syscall in session.syscalls() {
            // Detect CUDA driver ioctl failures
            if syscall.name == "ioctl" && syscall.result < 0 {
                // Parse CUDA error from ioctl result
                if let Some(cuda_error) = self.parse_cuda_error(syscall) {
                    errors.push(cuda_error);
                }
            }
        }

        errors
    }
}
```

### 6.2 Crash Analysis

Post-mortem analysis of GPU crashes:

```rust
/// Post-crash analysis using debug buffer
pub struct CrashAnalyzer {
    debug_buffer: Vec<u32>,
    marker_map: HashMap<u32, MarkerInfo>,
}

pub struct MarkerInfo {
    pub value: u32,
    pub location: SourceLocation,
    pub description: String,
}

pub struct CrashReport {
    pub last_marker: Option<MarkerInfo>,
    pub markers_executed: usize,
    pub probable_crash_location: SourceLocation,
    pub bug_class: Option<BugClass>,
    pub confidence: f64,
}

impl CrashAnalyzer {
    /// Analyze debug buffer after crash
    pub fn analyze(&self) -> CrashReport {
        // Read marker count
        let marker_count = self.debug_buffer[0] as usize;

        // Find last marker before crash
        let last_marker = if marker_count > 0 {
            let last_value = self.debug_buffer[marker_count];
            self.marker_map.get(&last_value).cloned()
        } else {
            None
        };

        // Determine probable crash location
        let crash_location = self.infer_crash_location(&last_marker, marker_count);

        // Classify bug
        let bug_class = self.classify_bug(&last_marker, &crash_location);

        CrashReport {
            last_marker,
            markers_executed: marker_count,
            probable_crash_location: crash_location,
            bug_class,
            confidence: self.calculate_confidence(marker_count),
        }
    }
}
```

---

## 7. Popperian Falsification Framework

### 7.1 Philosophical Foundation

Following Karl Popper's falsificationism [Popper 1959], we cannot prove PTX correct, but we can systematically attempt to falsify it. Each bug class represents a falsifiable hypothesis:

> **Hypothesis H_i**: "PTX module M does not contain bug class B_i"

The framework attempts to refute each hypothesis through targeted tests. Survival of refutation attempts increases confidence.

### 7.2 100-Point Falsification Matrix

Each PTX module is scored against 100 falsification tests grouped into 10 categories:

| Category | Points | Tests | Description |
|----------|--------|-------|-------------|
| **Syntax Validity** | 10 | F001-F010 | PTX parses without errors |
| **Type Safety** | 10 | F011-F020 | Register types match operations |
| **Address Space** | 15 | F021-F035 | Correct address space qualifiers |
| **Barrier Safety** | 15 | F036-F050 | Proper synchronization |
| **Memory Model** | 10 | F051-F060 | Coalescing, alignment |
| **Control Flow** | 10 | F061-F070 | CFG well-formed, no dead code |
| **Data Flow** | 10 | F071-F080 | No uninitialized registers |
| **Known Bugs** | 10 | F081-F090 | Absence of known bug patterns |
| **Performance** | 5 | F091-F095 | No obvious performance anti-patterns |
| **Instrumentation** | 5 | F096-F100 | Debug markers present and valid |

### 7.3 Falsification Test Definitions

```rust
/// Falsification test registry
pub struct FalsificationRegistry {
    tests: Vec<FalsificationTest>,
}

pub struct FalsificationTest {
    pub id: String,           // e.g., "F001"
    pub category: Category,
    pub description: String,
    pub points: u8,
    pub test_fn: Box<dyn Fn(&PtxModule) -> TestResult>,
}

pub enum TestResult {
    /// Test passed (hypothesis not refuted)
    Pass,
    /// Test failed (hypothesis refuted)
    Fail { evidence: String, location: Option<SourceLocation> },
    /// Test not applicable
    NotApplicable,
}

impl FalsificationRegistry {
    pub fn new() -> Self {
        let mut registry = Self { tests: Vec::new() };

        // Category 1: Syntax Validity (F001-F010)
        registry.add(FalsificationTest {
            id: "F001".into(),
            category: Category::SyntaxValidity,
            description: "PTX contains .version directive".into(),
            points: 1,
            test_fn: Box::new(|m| {
                if m.version.0 > 0 {
                    TestResult::Pass
                } else {
                    TestResult::Fail {
                        evidence: "Missing .version directive".into(),
                        location: None,
                    }
                }
            }),
        });

        registry.add(FalsificationTest {
            id: "F002".into(),
            category: Category::SyntaxValidity,
            description: "PTX contains .target directive".into(),
            points: 1,
            test_fn: Box::new(|m| {
                if m.target != SmTarget::Unknown {
                    TestResult::Pass
                } else {
                    TestResult::Fail {
                        evidence: "Missing .target directive".into(),
                        location: None,
                    }
                }
            }),
        });

        // ... F003-F010 ...

        // Category 3: Address Space (F021-F035)
        registry.add(FalsificationTest {
            id: "F021".into(),
            category: Category::AddressSpace,
            description: "No cvta.shared followed by generic ld/st".into(),
            points: 2,
            test_fn: Box::new(|m| {
                let validator = AddressSpaceValidator::new();
                let bugs = validator.detect_generic_shared_access(m);
                if bugs.is_empty() {
                    TestResult::Pass
                } else {
                    TestResult::Fail {
                        evidence: format!("{} generic shared access patterns found", bugs.len()),
                        location: bugs.first().map(|b| b.location.clone()),
                    }
                }
            }),
        });

        // Category 8: Known Bugs (F081-F090)
        registry.add(FalsificationTest {
            id: "F081".into(),
            category: Category::KnownBugs,
            description: "No 'loaded value' bug pattern".into(),
            points: 2,
            test_fn: Box::new(|m| {
                let analyzer = DataFlowAnalyzer::new(m);
                let bugs = analyzer.detect_loaded_value_bug();
                if bugs.is_empty() {
                    TestResult::Pass
                } else {
                    TestResult::Fail {
                        evidence: format!("{} loaded value bug patterns found", bugs.len()),
                        location: bugs.first().map(|b| b.load_location.clone()),
                    }
                }
            }),
        });

        // F082: Computed address from loaded value (NEW - 2026-01-05)
        // NOTE: membar.cta is PARTIAL workaround (LVB-003 finding):
        //   - Works in simple kernels (no preceding loops)
        //   - FAILS in complex kernels (with load loop + hash init)
        registry.add(FalsificationTest {
            id: "F082".into(),
            category: Category::KnownBugs,
            description: "No computed-address-from-loaded-value pattern (ptxas JIT bug)".into(),
            points: 2,
            test_fn: Box::new(|m| {
                let analyzer = DataFlowAnalyzer::new(m);
                let bugs = analyzer.detect_computed_addr_from_loaded();
                if bugs.is_empty() {
                    TestResult::Pass
                } else {
                    TestResult::Fail {
                        evidence: format!(
                            "{} computed-addr-from-loaded patterns: address computed from ld.shared used in store. \
                            Workarounds: membar.cta (simple kernels) or Kernel Fission (complex kernels)",
                            bugs.len()
                        ),
                        location: bugs.first().map(|b| b.load_location.clone()),
                    }
                }
            }),
        });

        // ... remaining tests ...

        registry
    }

    /// Run all falsification tests
    pub fn evaluate(&self, module: &PtxModule) -> FalsificationReport {
        let mut results = Vec::new();
        let mut total_points = 0;
        let mut earned_points = 0;

        for test in &self.tests {
            let result = (test.test_fn)(module);
            total_points += test.points as u32;

            match &result {
                TestResult::Pass => earned_points += test.points as u32,
                TestResult::NotApplicable => total_points -= test.points as u32,
                TestResult::Fail { .. } => {}
            }

            results.push((test.clone(), result));
        }

        FalsificationReport {
            results,
            score: (earned_points as f64 / total_points as f64) * 100.0,
            earned_points,
            total_points,
            confidence: self.calculate_confidence(earned_points, total_points),
        }
    }
}
```

### 7.4 Complete Falsification Test Catalog

#### Category 1: Syntax Validity (F001-F010, 10 points)

| ID | Test | Points | Rationale |
|----|------|--------|-----------|
| F001 | `.version` directive present | 1 | Required by PTX ISA |
| F002 | `.target` directive present | 1 | Required for SASS generation |
| F003 | `.address_size` is 32 or 64 | 1 | Must match target architecture |
| F004 | All labels are unique | 1 | Duplicate labels cause undefined behavior |
| F005 | All branch targets exist | 1 | Dangling references cause INVALID_PTX |
| F006 | All registers declared before use | 1 | Undeclared registers fail compilation |
| F007 | Instruction operand counts correct | 1 | Wrong arity causes INVALID_PTX |
| F008 | String literals properly escaped | 1 | Unescaped strings cause parse errors |
| F009 | Comments don't contain null bytes | 1 | Null bytes truncate PTX |
| F010 | UTF-8 encoding valid | 1 | Invalid encoding causes loader errors |

#### Category 2: Type Safety (F011-F020, 10 points)

| ID | Test | Points | Rationale |
|----|------|--------|-----------|
| F011 | Load dest type matches instruction type | 1 | Type mismatch causes undefined behavior |
| F012 | Store source type matches instruction type | 1 | Type mismatch causes undefined behavior |
| F013 | Arithmetic operand types consistent | 1 | Mixed types may produce wrong results |
| F014 | Comparison operand types consistent | 1 | Type mismatch in setp |
| F015 | Conversion types are compatible | 1 | Invalid cvt combinations |
| F016 | Address registers are 64-bit (sm_70+) | 1 | 32-bit addresses invalid on modern GPUs |
| F017 | Predicate registers used correctly | 1 | Non-predicate in predicate position |
| F018 | Vector types match element types | 1 | v4.f32 must use f32 elements |
| F019 | Atomic types are 32/64-bit | 1 | Other sizes not supported |
| F020 | Special registers read-only | 1 | Writing to %tid causes crash |

#### Category 3: Address Space (F021-F035, 15 points)

| ID | Test | Points | Rationale |
|----|------|--------|-----------|
| F021 | No cvta.shared + generic ld/st | 2 | Critical: GenericAddressCorruption |
| F022 | Shared memory uses 32-bit offsets | 2 | Critical: SharedMemU64Addressing |
| F023 | Direct .shared addressing preferred | 2 | MissingDirectShared pattern |
| F024 | Global addresses are 64-bit | 1 | 32-bit truncation causes crash |
| F025 | Local addresses within bounds | 1 | Stack overflow undetectable |
| F026 | Const memory read-only | 1 | Writes to const cause crash |
| F027 | Texture references valid | 1 | Invalid texture causes undefined |
| F028 | Surface references valid | 1 | Invalid surface causes undefined |
| F029 | Parameter space read-only | 1 | Writes to params illegal |
| F030 | Address space casts explicit | 1 | Implicit casts may fail |
| F031 | No mixed generic/specific access | 1 | Inconsistent addressing |

#### Category 4: Barrier Safety (F036-F050, 15 points)

| ID | Test | Points | Rationale |
|----|------|--------|-----------|
| F036 | bar.sync after shared write, before read | 3 | Critical: MissingBarrierSync |
| F037 | bar.sync reaches all threads | 2 | Barrier divergence deadlock |
| F038 | No barrier in divergent code | 2 | Causes deadlock [Sørensen 2016] |
| F039 | Warp-uniform barrier count | 1 | Different counts cause deadlock |
| F040 | bar.arrive/bar.wait paired | 1 | Unpaired causes hang |
| F041 | Named barriers unique per scope | 1 | Reuse causes race |
| F042 | membar.cta for device-scope sync | 1 | Missing causes visibility issue |
| F043 | membar.gl for system-scope sync | 1 | Missing causes visibility issue |
| F044 | atom.acq_rel for synchronization | 1 | Wrong memory order causes race |
| F045 | No barrier inside warp-divergent loop | 1 | Causes deadlock |

#### Category 5: Memory Model (F051-F060, 10 points)

| ID | Test | Points | Rationale |
|----|------|--------|-----------|
| F051 | Global loads coalesced (aligned) | 1 | Performance: 32x difference |
| F052 | Shared memory bank conflict free | 1 | Performance: 32-way conflict |
| F053 | Global stores aligned | 1 | Misaligned stores split |
| F054 | No shared memory overflow | 2 | Overflow corrupts stack |
| F055 | No local memory overflow | 1 | Stack corruption |
| F056 | Texture cache coherent | 1 | Stale data possible |
| F057 | L1 cache hints correct | 1 | Wrong hints degrade perf |
| F058 | Memory fence ordering | 1 | Weak order causes race |

#### Category 6: Control Flow (F061-F070, 10 points)

| ID | Test | Points | Rationale |
|----|------|--------|-----------|
| F061 | All code paths reach ret or exit | 2 | Falling off end undefined |
| F062 | No unreachable code | 1 | Dead code wastes registers |
| F063 | Loop bounds provably finite | 1 | Infinite loop hangs GPU |
| F064 | Recursion depth bounded | 1 | Stack overflow undetectable |
| F065 | Indirect jumps have valid targets | 1 | Invalid jump causes crash |
| F066 | Exception handlers reachable | 1 | Unreachable handlers wasteful |
| F067 | Call stack balanced | 1 | Unbalanced causes corruption |
| F068 | Uniform branches where possible | 1 | Divergence wastes SIMT |

#### Category 7: Data Flow (F071-F080, 10 points)

| ID | Test | Points | Rationale |
|----|------|--------|-----------|
| F071 | No use before def | 2 | Uninitialized value undefined |
| F072 | All parameters read | 1 | Unused params may indicate bug |
| F073 | No dead stores | 1 | Dead store wastes cycles |
| F074 | Loop-carried deps handled | 1 | Race in parallel loop |
| F075 | Phi nodes have all inputs | 1 | Missing input undefined |
| F076 | SSA form valid | 1 | Invalid SSA breaks optimization |
| F077 | No value escapes scope | 1 | Escaping value undefined |
| F078 | Def-use chains complete | 1 | Incomplete chain indicates bug |

#### Category 8: Known Bugs (F081-F090, 12 points)

| ID | Test | Points | Rationale |
|----|------|--------|-----------|
| F081 | No 'loaded value' store pattern | 0 | **FALSIFIED 2026-01-05**. Popperian refutation: `ld.shared → st.global` succeeds with 0xBEEFCAFE. Bug was F082/F021, not F081. |
| F082 | No computed-addr-from-loaded pattern | 2 | Critical: ComputedAddrFromLoaded. Fix: Kernel Fission (split kernel) |
| F083 | No cvta.shared in loop | 1 | LoopCvtaShared pattern |
| F084 | No instruction sequence sensitivity | 1 | SequentialCodeSensitivity |
| F085 | Warp shuffle lane bounds | 1 | OOB shuffle undefined |
| F086 | Vote intrinsics uniform | 1 | Non-uniform vote undefined |
| F087 | No deprecated opcodes | 1 | May be removed in future PTX |
| F088 | Tensor core constraints met | 1 | Wrong shape causes crash |
| F089 | Async copy constraints met | 1 | Wrong size causes crash |
| F090 | No store with computed address from loop counter | 1 | Variant of F082 in loops |

#### Category 9: Performance (F091-F095, 5 points)

| ID | Test | Points | Rationale |
|----|------|--------|-----------|
| F091 | No excessive register spilling | 1 | Spilling causes 100x slowdown |
| F092 | Occupancy >= 25% | 1 | Low occupancy wastes SMs |
| F093 | No redundant barriers | 1 | Unnecessary sync stalls |
| F094 | No redundant memory fences | 1 | Unnecessary fence stalls |
| F095 | Instruction-level parallelism | 1 | Sequential ops waste ILP |

#### Category 10: Instrumentation (F096-F100, 5 points)

| ID | Test | Points | Rationale |
|----|------|--------|-----------|
| F096 | Debug buffer parameter present | 1 | Required for crash analysis |
| F097 | Debug markers at key points | 1 | Enables crash localization |
| F098 | Debug markers unique | 1 | Duplicate markers ambiguous |
| F099 | Debug markers sequential | 1 | Gaps indicate missed points |
| F100 | Marker atomic operations valid | 1 | Race in debug buffer |
| F101 | Debug buffer writes are bounds-checked | 2 | Prevent secondary crashes |

### 7.5 Confidence Calculation

```rust
/// Calculate confidence based on falsification survival
///
/// Based on [Popper 1959] degree of corroboration:
/// C(h,e) = (P(e|h) - P(e)) / (P(e|h) + P(e))
///
/// Simplified for our domain:
/// - More severe tests survived = higher confidence
/// - Category coverage matters (all categories vs. clustered)
pub fn calculate_confidence(report: &FalsificationReport) -> f64 {
    let base_score = report.score / 100.0;

    // Category coverage bonus (Bayesian prior)
    let categories_passed = report.categories_with_all_tests_passed();
    let category_bonus = (categories_passed as f64 / 10.0) * 0.1;

    // Critical bug absence bonus
    let critical_bonus = if report.critical_bugs_absent() { 0.1 } else { 0.0 };

    // Combined confidence (capped at 0.99 - never certain)
    (base_score + category_bonus + critical_bonus).min(0.99)
}
```

---

## 8. Output Generation

### 8.1 HTML Report

```rust
/// Generate HTML analysis report
pub fn generate_html_report(analysis: &AnalysisResult) -> String {
    format!(r#"
<!DOCTYPE html>
<html>
<head>
    <title>PTX Analysis Report - {}</title>
    <style>
        .critical {{ background-color: #ffcccc; }}
        .high {{ background-color: #ffeecc; }}
        .medium {{ background-color: #ffffcc; }}
        .pass {{ background-color: #ccffcc; }}
        .score {{ font-size: 3em; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>PTX Analysis Report</h1>
    <div class="score">{}/100</div>
    <p>Confidence: {:.1}%</p>

    <h2>Falsification Results</h2>
    <table>
        <tr><th>ID</th><th>Test</th><th>Result</th><th>Evidence</th></tr>
        {}
    </table>

    <h2>Bug Patterns Detected</h2>
    <ul>
        {}
    </ul>

    <h2>Suggested Fixes</h2>
    <ul>
        {}
    </ul>
</body>
</html>
"#,
        analysis.module_name,
        analysis.falsification_score,
        analysis.confidence * 100.0,
        format_test_rows(&analysis.falsification_results),
        format_bug_list(&analysis.bugs),
        format_fix_list(&analysis.suggested_fixes),
    )
}
```

### 8.2 FKR Test Generation (jugar-probar)

```rust
/// Generate FKR tests for jugar-probar
pub fn generate_fkr_tests(analysis: &AnalysisResult) -> String {
    let mut tests = String::new();

    tests.push_str(r#"
//! Auto-generated PTX FKR tests
//! Generated by trueno-ptx-debug
//!
//! Run: cargo test -p trueno-gpu --test ptx_fkr --features "cuda"

#![cfg(feature = "cuda")]

use trueno_gpu::kernels::Kernel;

"#);

    // Generate test for each detected bug
    for bug in &analysis.bugs {
        tests.push_str(&format!(r#"
#[test]
fn ptx_fkr_no_{bug_id}() {{
    let kernel = {kernel_name}::new();
    let ptx = kernel.emit_ptx();

    // Verify absence of bug pattern: {description}
    {assertion}
}}
"#,
            bug_id = bug.id.to_lowercase().replace("-", "_"),
            kernel_name = analysis.module_name,
            description = bug.description,
            assertion = generate_assertion(bug),
        ));
    }

    tests
}

fn generate_assertion(bug: &Bug) -> String {
    match &bug.pattern {
        BugPattern::GenericSharedAccess => {
            r#"assert!(!ptx.contains("cvta.shared") ||
                !ptx.contains("ld.u32") && !ptx.contains("st.u32"),
                "Generic shared access pattern detected");"#.into()
        }
        BugPattern::MissingBarrier { write_loc, read_loc } => {
            format!(r#"// Check barrier between line {} and {}
    let analyzer = BarrierAnalyzer::new(&ptx);
    assert!(analyzer.barrier_between({}, {}));"#,
                write_loc.line, read_loc.line, write_loc.line, read_loc.line)
        }
        // ... other patterns
        _ => "// TODO: Generate assertion".into(),
    }
}
```

---

## 9. CLI Interface

### 9.1 Usage

```bash
# Analyze PTX file
trueno-ptx-debug analyze kernel.ptx

# Analyze with full falsification
trueno-ptx-debug analyze kernel.ptx --falsify

# Instrument PTX for debugging
trueno-ptx-debug instrument kernel.ptx -o kernel_debug.ptx

# Analyze crash with debug buffer
trueno-ptx-debug crash-analyze --debug-buffer debug.bin --markers markers.json

# Generate FKR tests
trueno-ptx-debug gen-fkr kernel.ptx -o tests/kernel_fkr.rs

# Trace with renacer
trueno-ptx-debug trace -- ./gpu_program
```

### 9.2 Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Analysis passed, score >= 90 |
| 1 | Analysis passed, score 70-89 (warnings) |
| 2 | Analysis failed, score < 70 |
| 3 | Critical bugs detected |
| 10 | Parse error |
| 11 | I/O error |

---

## 10. Peer-Reviewed Citations

### GPU Architecture and Programming

1. **[Nickolls et al. 2008]** Nickolls, J., Buck, I., Garland, M., & Skadron, K. (2008). Scalable parallel programming with CUDA. *ACM Queue*, 6(2), 40-53. https://doi.org/10.1145/1365490.1365500

2. **[Volkov 2010]** Volkov, V. (2010). Better performance at lower occupancy. *GPU Technology Conference*, GTC 2010.

3. **[Harris 2013]** Harris, M. (2013). How to Access Global Memory Efficiently in CUDA C/C++ Kernels. *NVIDIA Developer Blog*.

### Verification and Correctness

4. **[Sørensen et al. 2016]** Sørensen, T., Donaldson, A. F., Batty, M., Gopalakrishnan, G., & Rakamarić, Z. (2016). Portable inter-workgroup barrier synchronisation for GPUs. *ACM SIGPLAN Notices*, 51(10), 39-58. https://doi.org/10.1145/3022671.2984032

5. **[Betts et al. 2012]** Betts, A., Chong, N., Donaldson, A., Qadeer, S., & Thomson, P. (2012). GPUVerify: a verifier for GPU kernels. *ACM SIGPLAN Notices*, 47(10), 113-132. https://doi.org/10.1145/2398857.2384625

6. **[Li & Gopalakrishnan 2010]** Li, G., & Gopalakrishnan, G. (2010). Scalable SMT-based verification of GPU kernel functions. *Proceedings of the ACM SIGSOFT Symposium on the Foundations of Software Engineering*, 187-196. https://doi.org/10.1145/1882291.1882320

### Falsificationism and Testing Theory

7. **[Popper 1959]** Popper, K. R. (1959). *The Logic of Scientific Discovery*. Hutchinson. ISBN 978-0-415-27844-7

8. **[Jia & Harman 2011]** Jia, Y., & Harman, M. (2011). An analysis and survey of the development of mutation testing. *IEEE Transactions on Software Engineering*, 37(5), 649-678. https://doi.org/10.1109/TSE.2010.62

9. **[Howden 1978]** Howden, W. E. (1978). Theoretical and empirical studies of program testing. *IEEE Transactions on Software Engineering*, SE-4(4), 293-298.

### Toyota Production System

10. **[Liker 2004]** Liker, J. K. (2004). *The Toyota Way: 14 Management Principles from the World's Greatest Manufacturer*. McGraw-Hill. ISBN 978-0-07-139231-0

11. **[Ohno 1988]** Ohno, T. (1988). *Toyota Production System: Beyond Large-Scale Production*. Productivity Press. ISBN 978-0-915299-14-0

### PTX and CUDA Documentation

12. **[NVIDIA PTX ISA 8.0]** NVIDIA Corporation. (2024). *Parallel Thread Execution ISA Version 8.0*. https://docs.nvidia.com/cuda/parallel-thread-execution/

13. **[NVIDIA CUDA C++ Guide]** NVIDIA Corporation. (2024). *CUDA C++ Programming Guide*. https://docs.nvidia.com/cuda/cuda-c-programming-guide/

### Compression Algorithms

14. **[Collet 2011]** Collet, Y. (2011). LZ4 - Extremely fast compression. https://github.com/lz4/lz4

15. **[Ozsoy et al. 2012]** Ozsoy, A., Swany, M., & Chauhan, A. (2012). Pipelined parallel LZSS for streaming data compression on GPGPUs. *IEEE 18th International Conference on Parallel and Distributed Systems*, 37-44. https://doi.org/10.1109/ICPADS.2012.15

### Performance Analysis

16. **[Gregg 2013]** Gregg, B. (2013). *Systems Performance: Enterprise and the Cloud*. Prentice Hall. ISBN 978-0-13-339009-4

17. **[Williams et al. 2009]** Williams, S., Waterman, A., & Patterson, D. (2009). Roofline: an insightful visual performance model for multicore architectures. *Communications of the ACM*, 52(4), 65-76. https://doi.org/10.1145/1498765.1498785

---

## 11. Verification Matrix

| ID | Requirement | Test | Status |
|----|-------------|------|--------|
| REQ-001 | Parse valid PTX | Unit tests for PTX parser | PENDING |
| REQ-002 | Detect GenericAddressCorruption | F021 falsification test | PENDING |
| REQ-003 | Detect MissingBarrierSync | F036-F045 tests | PENDING |
| REQ-004 | Detect LoadedValueBug | F081 falsification test | PENDING |
| REQ-005 | Generate valid instrumented PTX | Round-trip test | PENDING |
| REQ-006 | Integrate with renacer | Integration test | PENDING |
| REQ-007 | Generate FKR tests | Output validation | PENDING |
| REQ-008 | Calculate confidence correctly | Property test | PENDING |
| REQ-009 | HTML report renders | Visual regression | PENDING |
| REQ-010 | 100-point falsification complete | All F001-F100 implemented | PENDING |

---

## 12. Implementation Roadmap

### Phase 1: Core Analysis (Week 1-2)

| Task | Acceptance Criteria | Priority |
|------|---------------------|----------|
| PTX lexer and parser | Parse all trueno-gpu kernels | P0 |
| Type checker | Detect all register type errors | P0 |
| Address space validator | Detect GenericAddressCorruption | P0 |
| Control flow analyzer | Construct CFG, detect dead code | P1 |

### Phase 2: Falsification Framework (Week 3)

| Task | Acceptance Criteria | Priority |
|------|---------------------|----------|
| F001-F020 (Syntax, Type) | All tests implemented | P0 |
| F021-F050 (Address, Barrier) | Critical bugs detected | P0 |
| Confidence calculation | Formula implemented | P1 |
| HTML report generation | Readable output | P1 |

### Phase 3: Integration (Week 4)

| Task | Acceptance Criteria | Priority |
|------|---------------------|----------|
| renacer bridge | Trace CUDA syscalls | P1 |
| Debug instrumentation | Automatic marker injection | P1 |
| FKR test generation | Valid jugar-probar tests | P1 |
| CLI interface | All commands working | P0 |

### Phase 4: Validation (Week 5)

| Task | Acceptance Criteria | Priority |
|------|---------------------|----------|
| Run against LZ4 kernel | All known bugs detected | P0 |
| Coverage >= 90% | make coverage passes | P0 |
| F051-F100 complete | Full 100-point framework | P1 |
| Documentation | Book chapter complete | P2 |

---

## 13. Appendix: Quick Reference

### Bug Pattern Detection Commands

```bash
# Check for generic shared access
grep -n "cvta\.shared" kernel.ptx && grep -n "ld\.u32\|st\.u32" kernel.ptx

# Check for missing barriers
trueno-ptx-debug analyze kernel.ptx --check barriers

# Check for loaded value pattern
trueno-ptx-debug analyze kernel.ptx --check loaded-value

# Full analysis with score
trueno-ptx-debug analyze kernel.ptx --falsify --json
```

### Debug Buffer Analysis

```bash
# Read debug buffer after crash
trueno-ptx-debug crash-analyze \
    --debug-buffer /tmp/debug_buf.bin \
    --markers kernel_markers.json \
    --ptx kernel.ptx
```

### Integration with CI/CD

```yaml
# .github/workflows/ptx-analysis.yml
- name: Analyze PTX
  run: |
    trueno-ptx-debug analyze kernel.ptx --falsify --min-score 90
    if [ $? -ne 0 ]; then
      echo "PTX analysis failed"
      exit 1
    fi
```

---

**End of Specification**

*This specification follows Toyota Way principles of built-in quality (Jidoka), continuous improvement (Kaizen), and systematic problem-solving. The Popperian falsification framework ensures we approach certainty asymptotically through rigorous attempted refutation rather than naive verification.*
