# Eiffel DbC — Domain Applicability (Part 2)

Version: 1.0
Status: proposed
Date: 2026-04-09

*See also: [eiffel-dbc-domains.md](eiffel-dbc-domains.md) (Part 1)*

#### Orchestration / Transpilation (batuta)

Batuta is the stack's orchestration framework — a transpilation
pipeline that converts Python/C/Shell to Rust, coordinates multi-crate
releases, and routes ML workloads to backends (GPU/SIMD/scalar) via
cost-based selection. It applies Toyota Production System principles
(Jidoka, Poka-Yoke, Kaizen) to code transformation.

Transpilation is a domain where DbC provides *correctness guarantees
that are otherwise impossible to test exhaustively*. You cannot run
every possible Python program through a transpiler, but you can prove
structural properties of the transformation.

| Obligation Type | Orchestration / Transpilation Example |
|---|---|
| `precondition` | Input source is syntactically valid Python/C/Shell |
| `postcondition` | Output Rust compiles; semantically equivalent on test suite |
| `equivalence` | `transpile(source).eval(input) = source.eval(input)` for all test inputs |
| `frame` | Transpilation does not modify source files; only output directory written |
| `invariant` | Type safety preserved through all 5 pipeline stages (Analysis → Build) |
| `loop_invariant` | Pipeline context valid at each stage transition |
| `state_machine` | Pipeline stages: `analysis → transpilation → optimization → validation → build` |
| `subcontract` | `PyTorchConverter` refines `TranspilerPlugin` — accepts PyTorch subset, produces Realizar ops |
| `determinism` | Same source + same config → identical transpiled output |
| `completeness` | All NumPy ops in input have Trueno mappings; all sklearn algorithms have Aprender mappings |
| `bound` | Context generation < 5s for 10K LOC; memory < 500MB for 100K LOC |
| `conservation` | Number of functions in output = number of functions in input (no dropping) |
| `old_state` | After adding a converter plugin, `plugin_count = old(plugin_count) + 1` |

**Key insight:** Batuta's 5-phase pipeline with Jidoka validation gates
is a natural DbC structure. Each stage has implicit pre/postconditions
(the `PipelineStage` trait's `validate()` method). Making these
explicit as YAML contracts would let provable-contracts verify the
*transpiler itself* — not just the transpiled code.

The `BackendSelector` (MoE routing for GPU/SIMD/scalar selection) is
a particularly strong candidate for `postcondition` + `bound`
contracts: given an operation's complexity profile, the selected
backend must satisfy both correctness (equivalence to scalar) and
performance bounds (e.g., the 5× PCIe rule from Gregg & Hazelwood).

#### Code Quality / AI Context (pmat)

PMAT (paiml-mcp-agent-toolkit) is the stack's code quality analysis
and AI context generation system — the `pmat` binary used throughout
the stack for TDG scoring, semantic search (`pmat query`), mutation
testing, and MCP server hosting. It supports 17+ languages and
provides a "uniform contracts" architecture pattern for its 19+ MCP
tools.

PMAT's "uniform contracts" pattern is notable: `BaseAnalysisContract`
and its specializations (`AnalyzeComplexityContract`,
`AnalyzeSatdContract`, etc.) enforce identical parameter sets across
CLI, MCP, and HTTP interfaces. This is a *structural* application of
DbC — Meyer's class invariant ensuring interface uniformity — distinct
from our *mathematical* contracts but equally contractable.

| Obligation Type | Code Quality / Analysis Example |
|---|---|
| `precondition` | Project path exists and contains supported language files |
| `postcondition` | Output format matches requested format; all metrics are finite |
| `invariant` | TDG grade is monotonic: improving code never worsens the grade |
| `frame` | Analysis never modifies the analyzed codebase (read-only) |
| `determinism` | Same codebase state → same TDG grade, same complexity scores |
| `idempotency` | Running analysis twice produces identical output |
| `bound` | Analysis completes in < 5s for 10K LOC |
| `completeness` | All files matching language filter are analyzed |
| `conservation` | Total LOC reported = sum of per-file LOC |
| `equivalence` | CLI output matches MCP output matches HTTP output (uniform contracts) |
| `roundtrip` | `deserialize(serialize(analysis_result)) = analysis_result` |
| `subcontract` | `AnalyzeComplexityContract` refines `BaseAnalysisContract` — same base params, adds complexity thresholds |
| `old_state` | After mutation testing, `killed_mutants ≥ old(killed_mutants)` (monotonic improvement tracking) |
| `loop_invariant` | During multi-file analysis: partial results consistent with final aggregate |

**Key insight:** PMAT's uniform contracts pattern is Meyer's class
invariant in disguise. The `BaseAnalysisContract` struct with
`#[serde(flatten)]` inheritance ensures that every interface (CLI, MCP,
HTTP) receives identical parameters — a *structural guarantee* that
could be formalized as a `subcontract` obligation: each specialized
contract (complexity, SATD, TDG) refines the base contract by adding
fields without removing any.

PMAT is also the stack's *quality oracle*. Its TDG scores and mutation
coverage data could *feed into* provable-contracts' scoring system as
an additional dimension — binding provable-contracts' formal
verification metrics with PMAT's empirical quality metrics for a
complete picture.

#### Media Asset Pipeline (rmedia)

Rmedia is the stack's pure Rust headless video editor — an 8-crate
workspace that transforms SVG frame sequences, audio, and SRT
transcripts into rendered MP4 videos with deterministic, scorable
output. It has a 7-dimension render pipeline score (speed, efficiency,
determinism, observability, reliability, quality, pipeline health)
and machine-enforced SVG visual quality floors.

Media asset production is a domain where DbC types provide guarantees
that are otherwise *impossible to test manually*. You cannot watch
every rendered video frame-by-frame, but you can prove codec parity,
duration bounds, determinism, and visual quality minimums.

**Rendering contracts:**

| Obligation Type | Rendering Example |
|---|---|
| `precondition` | Input SVG frames are valid (parseable, 1920x1080 viewBox); SRT file exists and parses |
| `postcondition` | Output MP4 has correct codec (h264), resolution (1920x1080), channels (stereo), sample_rate (48kHz) |
| `equivalence` | `ffprobe(rmedia_output) = ffprobe(melt_output)` for codec, resolution, channels, sample_rate |
| `determinism` | Same inputs → identical file hash (SRT locked via SHA-256) |
| `bound` | RTX (real-time factor) ≥ 1.5x; render time < total_frames / fps / 1.5 |
| `frame` | Rendering modifies output directory only; input SVGs, SRT, and audio unchanged |
| `invariant` | YUV420P color space maintained through entire pipeline (no unnecessary swscale) |
| `loop_invariant` | During frame pipeline: bounded channel depth ≤ 16 at every producer step |
| `loop_variant` | Remaining frames = total_frames - rendered_frames, strictly decreasing |
| `old_state` | SRT lock: `sha256(srt_content) = old(sha256_lock)` — transcript hasn't drifted |
| `conservation` | Output frame count = ceil(transcript_duration × fps) ± 1 frame |
| `roundtrip` | `decode(encode(yuv_frame)) ≈ yuv_frame` within CRF tolerance |

**Animation contracts:**

| Obligation Type | Animation Example |
|---|---|
| `precondition` | Keyframe sequence has ≥ 2 frames; easing function is valid enum variant |
| `postcondition` | Interpolated value at t=0.0 equals start value; at t=1.0 equals end value |
| `invariant` | Easing parameter t ∈ [0, 1] at every interpolation step |
| `bound` | Animation timing aligns to SRT within ±2 frames |
| `monotonicity` | For linear easing: t₁ < t₂ → lerp(t₁) ≤ lerp(t₂) |
| `idempotency` | Rendering same animation plan twice produces identical output |

**SVG quality contracts (machine-enforced visual floors):**

| Obligation Type | SVG Quality Example |
|---|---|
| `bound` | Fill opacity ≥ 0.20 for content, ≥ 0.10 for backgrounds |
| `bound` | Stroke width ≥ 3.0 for icon outlines, ≥ 2.0 for details |
| `bound` | Font size ≥ 36px for hero titles, ≥ 14px for labels |
| `bound` | Icon bounding box ≥ 80×80 for hero banners |
| `completeness` | Full-canvas background rect present as first child |
| `invariant` | No `<text>` elements in no-text images (logo, marketing, nav) |

**Course pipeline contracts:**

| Obligation Type | Course Pipeline Example |
|---|---|
| `completeness` | All lessons rendered with valid video files |
| `conservation` | Number of output videos = number of input lesson directories |
| `postcondition` | Aggregate score computed via harmonic mean; any zero dimension → F grade |
| `old_state` | SRT lock hash matches across multiple renders |
| `state_machine` | Pipeline phases: `discover → validate → render → score → generate_marketing` |
| `frame` | Course pipeline modifies output directory only; source lessons unchanged |

**Key insight:** Rmedia's obsession with reproducibility — SHA-256 SRT
locks, integer-only compositing (`(a * (256-p) + b * p) >> 8`), no
f32 intermediates, deterministic frame output — makes it the most
naturally contractable non-kernel domain in the stack. Every property
is already quantified; the contracts just need to be formalized in
YAML. The 7-dimension render pipeline score is essentially a scoring
contract waiting to be extracted.

The `pv generate` pipeline should produce deterministic README.md
and CI workflow files for rmedia that validate:
- Codec parity with melt (`make bench-parity`)
- Determinism (two renders produce identical output)
- SVG quality floors (fill opacity, stroke width, font size)
- SRT lock integrity (SHA-256 hash match)
- Render pipeline score ≥ threshold (per-dimension and aggregate)

#### Configuration / Infrastructure (General)

Infrastructure contracts beyond IaC — deployment invariants and
resource constraints for orchestration platforms.

| Obligation Type | Infrastructure Example |
|---|---|
| `precondition` | Available memory ≥ model size + KV cache budget before load |
| `postcondition` | After deployment, health check returns 200 within 30s |
| `invariant` | Replica count ≥ `min_replicas` at all times |
| `frame` | Rolling update modifies at most `max_surge` pods; others unchanged |
| `conservation` | Total allocated GPU memory ≤ physical GPU memory |
| `bound` | Container CPU limit ≤ node capacity |
| `loop_variant` | Rolling restart: remaining pods = `total - restarted`, strictly decreasing |

### 8.4. Domain-Specific `references` and `equations`

Meyer's seamless development principle means the `metadata.references`
field and `equations` section should point to the *domain authority*,
not just arXiv papers:

| Domain | `references` Source | `equations` Encode |
|---|---|---|
| ML kernels | arXiv papers | Governing math (softmax formula) |
| Simulation | Physics textbooks, numerical methods papers | Conservation laws, integrator formulas |
| Testing/QA | Playwright docs, WCAG 2.2, W3C specs | Coverage metrics, assertion semantics |
| IaC (forjar) | POSIX spec, SSH RFC 4253, Nix papers | DAG ordering, BLAKE3 hash semantics, convergence |
| Orchestration (batuta) | Language specs (Python, POSIX), PL theory | Transpilation semantics, cost models |
| Code quality (pmat) | McCabe (1976), Halstead (1977), SATD literature | Complexity metrics, TDG scoring formulas |
| Media (rmedia) | FFmpeg docs, H.264/H.265 specs, MLT format, WCAG contrast | Codec parity, compositing math, easing curves, SVG quality floors |
| Presentation | WCAG 2.2, Material Design spec | Layout algorithms (flexbox distribution) |
| Data pipeline | Schema registry, data dictionary | Transform rules (join semantics) |
| API | OpenAPI spec, RFC 7231 | Request/response schemas |
| Infrastructure | Kubernetes API spec, cloud SLAs | Resource models (roofline) |

This is a natural extension: the pipeline's Phase 1 (EXTRACT) already
says "arXiv PDF → canonical math." For non-kernel domains, Phase 1
becomes "domain specification → canonical rules."

### 8.5. When DbC Types Matter Most by Domain

Not every obligation type is equally useful in every domain. The Eiffel
DbC types have different gravity depending on the domain:

| Type | Kernels | Simulation | IaC | Media | Orchestration | Quality | Testing | Presentation | Data | API |
|---|---|---|---|---|---|---|---|---|---|---|
| `precondition` | Medium | **High** | **High** | **High** | **High** | **High** | **High** | **High** | **High** | **High** |
| `postcondition` | Medium | **High** | **High** | **High** | **High** | **High** | **High** | **High** | **High** | **High** |
| `frame` | **High** | **High** | **High** | **High** | **High** | **High** | **High** | **High** | **High** | Medium |
| `loop_invariant` | **High** | **High** | **High** | **High** | **High** | Medium | Medium | Medium | Medium | Low |
| `loop_variant` | Medium | **High** | **High** | **High** | Medium | Low | Medium | Low | Low | Low |
| `old_state` | Medium | **High** | **High** | **High** | Medium | Medium | Medium | **High** | **High** | **High** |
| `subcontract` | Medium | Medium | **High** | Medium | **High** | **High** | **High** | **High** | Medium | **High** |

**Pattern:** The pre/post/frame/old-state cluster is *most* valuable
outside the kernel domain. Kernel contracts can often get away with
algebraic property types (`invariant`, `bound`, `monotonicity`) because
the math is self-contained. But simulation, IaC, orchestration, code
quality, testing, presentation, data, and API contracts inherently
describe *interactions between components* — exactly the caller/callee
relationship that Meyer's framework models.

Simulation and IaC are the two domains where *all seven* DbC types
are high-value. Simulations are stateful (frame, old-state), iterative
(loop_invariant, loop_variant), have strict input requirements
(precondition), must guarantee physical plausibility (postcondition),
and compose via integrator substitution (subcontract). IaC shares
this profile: infrastructure is stateful (frame — don't touch other
resources), convergence is iterative (DAG waves with loop_invariant/
variant), inputs must be validated (precondition — SSH reachable,
bashrs valid), outputs must be verified (postcondition — resource
converged), state comparison is the core operation (old_state — hash
diffing), and transports are substitutable (subcontract — pepita
refines SSH).

### 8.6. Cross-Project Dependency Graph and Contract Flow

The PAIML stack has a layered dependency structure. Contracts flow
*upward* through the dependency graph — a contract on trueno's SIMD
kernel propagates obligations to every consumer (aprender, entrenar,
realizar, presentar, probar, pmat, batuta, renacer).

```
Level 0 (Foundation)     provable-contracts ←──── trueno
                                |                    ↓
Level 1 (Direct)          forjar ◄──┘     ┌── aprender ──┬── entrenar
                                          │              │
                                          ├── presentar  ├── renacer
                                          │      ↓       │
                                          ├── probar     └── pmat
                                          │
Level 2 (Composite)                       ├── realizar ── pacha
                                          │
Level 3 (Orchestration)                   └── batuta ─── simular
```

**Current contract binding coverage:**

| Project | binding.yaml | `#[contract]` macros | Tier |
|---|---|---|---|
| trueno | Yes (22 bindings) | — | 1-2 |
| aprender | Yes (301 bindings) | 38 annotations | 1-5 |
| entrenar | Yes (96 bindings) | — | 4 |
| realizar | Yes (23 bindings) | — | 3 |
| forjar | Yes | 4 annotations | 9 (IaC) |
| simular | Yes (3 contracts) | macros dep | 8 (simulation) |
| presentar | **No** | — | 13 (presentation) |
| probar | **No** | — | 12 (testing) |
| batuta | **No** | — | 10 (orchestration) |
| pmat | **No** | — | 11 (code quality) |
| renacer | **No** | — | — |
| pacha | **No** | — | — |

**Cross-project contract obligations:**

The Eiffel DbC types create new cross-project contract relationships
that don't exist with property-only types:

**1. Subcontract chains across the dependency graph.**

When trueno exposes a `Kernel` trait and aprender implements it, the
implementation is a behavioral subtype. A `subcontract` obligation
makes this explicit:

```yaml
# In aprender's binding
proof_obligations:
  - type: subcontract
    property: "aprender::softmax refines trueno Kernel trait"
    formal: "pre(Kernel::execute) → pre(aprender::softmax)"
    parent_contract: "softmax-kernel-v1"
```

This propagates: if realizar wraps aprender's softmax, it inherits
the subcontract chain. `pv validate` can verify the entire chain
from trueno → aprender → realizar.

**2. Frame conditions at API boundaries.**

When entrenar calls trueno's GPU kernels, the frame condition must
hold across the FFI boundary: trueno's kernel must not corrupt
entrenar's training state. This is a cross-project frame obligation:

```yaml
# In entrenar's contract
proof_obligations:
  - type: frame
    property: "trueno kernel modifies output buffer only"
    formal: "modifies(output) ∧ preserves(weights, gradients, optimizer_state)"
```

**3. Precondition propagation through the stack.**

A precondition on trueno's matmul (input dimensions must match)
propagates to every consumer. Each layer can *weaken* the
precondition (Meyer's `require else`):

```
trueno:    require dimensions_match(A, B)
aprender:  require else dimensions_broadcastable(A, B)  # weaker
realizar:  require else model.expects_shape(input)       # weaker still
```

**4. Performance bound composition (BrickBudget flow).**

Presentar's `BrickBudget` (16ms per frame) decomposes into trueno
GPU kernel bounds. A `bound` obligation on the widget level requires
corresponding bounds on the compute level:

```
presentar Widget:   bound(total_render < 16ms)
  └── trueno GPU:   bound(kernel_dispatch < 2ms)
  └── trueno SIMD:  bound(scalar_fallback < 8ms)
  └── probar:       bound(assertion_check < 1ms)
```

This is a `conservation`-like obligation: the sum of component bounds
must not exceed the parent's budget.

**5. Tracing and profiling contracts (renacer integration).**

Trueno's renacer integration defines golden trace baselines with max
10% deviation. These are `bound` + `old_state` obligations:

```yaml
proof_obligations:
  - type: old_state
    property: "Performance within 10% of golden baseline"
    formal: "|metric(current) - metric(old(golden))| / metric(old(golden)) < 0.10"
  - type: bound
    property: "Matrix operation syscall budget"
    formal: "syscall_count(matrix_ops) ≤ 200"
```

**6. PTX bug detection as contract verification.**

Trueno-explain's `PtxBugClass` classification (SharedMemU64Addressing,
LoopBranchToEnd, MissingBarrierSync) maps directly to `postcondition`
obligations on PTX kernel generation:

```yaml
proof_obligations:
  - type: postcondition
    property: "Generated PTX has no P0 critical bugs"
    formal: "∀ bug ∈ analyze_ptx(output): bug.severity ≠ Critical"
  - type: invariant
    property: "Shared memory accesses use 32-bit addressing"
    formal: "¬∃ instr ∈ ptx: is_shared_mem(instr) ∧ is_64bit_addr(instr)"
```

### 8.7. Implications for the Stack

Extending provable-contracts to non-kernel domains requires:

1. **No schema changes beyond Section 3.** The 7 new obligation types
   and 3 new fields already support all domains above.

2. **New contract tiers.** Add Tier 8+ for non-kernel domains:

   | Tier | Scope | Domain | Primary Consumer |
   |---|---|---|---|
   | Tier 8 | Simulation contracts | Physics conservation, integrators, checkpoints | simular |
   | Tier 9 | IaC contracts | DAG ordering, convergence, drift, transport safety | forjar |
   | Tier 10 | Orchestration contracts | Transpilation semantics, pipeline stages, cost models | batuta |
   | Tier 11 | Code quality contracts | Analysis invariants, TDG scoring, uniform interfaces | pmat |
   | Tier 12 | Media asset contracts | Codec parity, determinism, SVG quality, animation timing | rmedia |
   | Tier 13 | Testing contracts | Coverage, assertions, replay, accessibility | probar |
   | Tier 14 | Presentation contracts | UI layout, accessibility, animation | presentar |
   | Tier 15 | Data pipeline contracts | Schema, transform, quality | — |
   | Tier 16 | API contracts | Protocol, SLA, versioning | — |
   | Tier 17 | Infrastructure contracts | Deployment, resource, scaling | — |

3. **New kernel equivalence classes.** The existing classes (A-E) cover
   ML architectures. Non-kernel domains need their own classification.

4. **Scoring adaptation.** The current 5 scoring dimensions (spec depth,
   falsification, Kani, Lean, binding) apply to non-kernel contracts
   without modification — the *content* of the contracts changes, but
   the quality rubric does not.

5. **Verification ladder adjustment.** L4 (Kani) and L5 (Lean) remain
   applicable but the "natural bound" concept differs:
   - Kernels: natural bound = SIMD width, super-block size
   - Simulation: natural bound = max particles, max timesteps per epoch
   - IaC: natural bound = max resources per config, max DAG depth
   - Media: natural bound = max frame count, max SVG element count, max SRT entries
   - Orchestration: natural bound = max pipeline stages, max converter ops
   - Code quality: natural bound = max files per analysis, max AST depth
   - Testing: natural bound = max DOM depth, max locator chain length
   - Presentation: natural bound = max component tree depth, max children
   - Data: natural bound = max batch size, max column count
   - API: natural bound = max request body size, max concurrent requests

6. **Simular is the nearest adoption target.** It already has 3 YAML
   contracts and a `provable-contracts-macros` dependency. Adding
   `precondition`/`postcondition` pairs to its existing gradient and
   checkpoint contracts, `frame` to its integration step, and
   `loop_invariant`/`loop_variant` to its simulation loops would
   demonstrate the full Eiffel DbC vocabulary on a real, stateful
   system.

7. **Forjar is equally ready.** It already has `#[contract]` annotations
   on DAG ordering, atomic writes, recipe determinism, and codegen
   dispatch. Its planner/executor pipeline is a textbook DbC system
   where desired state = postcondition, current state = old_state, and
   convergence = contract fulfillment. Adding `frame` to its resource
   apply (only the target resource changes), `precondition` to its
   transport safety (bashrs validation), and `old_state` to its drift
   detection (BLAKE3 hash comparison) would complete the picture.

8. **Probar's Brick Architecture is a natural contract surface.** Each
   brick is already a test component with implicit pre/post/frame
   conditions. Extracting these into YAML contracts would make probar
   both a *consumer* of provable-contracts (its own internal
   correctness) and a *tool* for verifying other projects' contracts
   at the property-test level (L3).

---

