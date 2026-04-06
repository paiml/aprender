# Summary

[Introduction](./introduction.md)

---

# Part I: Core Philosophy

- [The Orchestration Paradigm](./part1/orchestration-paradigm.md)
- [Toyota Way Principles](./part1/toyota-way.md)
  - [Muda: Waste Elimination](./part1/muda.md)
  - [Jidoka: Built-in Quality](./part1/jidoka.md)
  - [Kaizen: Continuous Improvement](./part1/kaizen.md)
  - [Heijunka: Level Scheduling](./part1/heijunka.md)
  - [Kanban: Visual Workflow](./part1/kanban.md)
  - [Andon: Problem Visualization](./part1/andon.md)
- [First Principles Thinking](./part1/first-principles.md)
- [Semantic Preservation](./part1/semantic-preservation.md)

---

# Part II: The 5-Phase Workflow

- [Workflow Overview](./part2/workflow-overview.md)
- [Phase 1: Analysis](./part2/phase1-analysis.md)
  - [Language Detection](./part2/language-detection.md)
  - [Dependency Analysis](./part2/dependency-analysis.md)
  - [Technical Debt Grade (TDG)](./part2/tdg-scoring.md)
  - [ML Framework Detection](./part2/ml-detection.md)
- [Phase 2: Transpilation](./part2/phase2-transpilation.md)
  - [Tool Selection](./part2/tool-selection.md)
  - [Incremental Compilation](./part2/incremental.md)
  - [Caching Strategy](./part2/caching.md)
  - [Error Handling](./part2/error-handling.md)
- [Phase 3: Optimization](./part2/phase3-optimization.md)
  - [SIMD Vectorization](./part2/simd.md)
  - [GPU Acceleration](./part2/gpu.md)
  - [Memory Layout](./part2/memory-layout.md)
  - [MoE Backend Selection](./part2/moe.md)
- [Phase 4: Validation](./part2/phase4-validation.md)
  - [Syscall Tracing](./part2/syscall-tracing.md)
  - [Output Comparison](./part2/output-comparison.md)
  - [Test Suite Execution](./part2/test-execution.md)
  - [Benchmarking](./part2/benchmarking.md)
- [Phase 5: Deployment](./part2/phase5-deployment.md)
  - [Release Builds](./part2/release-builds.md)
  - [Cross-compilation](./part2/cross-compilation.md)
  - [WebAssembly](./part2/wasm.md)
  - [Docker Containerization](./part2/docker.md)
  - [Distribution](./part2/distribution.md)

---

# Part III: The Tool Ecosystem

- [Tool Overview](./part3/tool-overview.md)
- [Transpilers](./part3/transpilers.md)
  - [Decy: C/C++ → Rust](./part3/decy.md)
  - [Depyler: Python → Rust](./part3/depyler.md)
  - [Bashrs: Shell → Rust](./part3/bashrs.md)
- [Foundation Libraries](./part3/foundation-libs.md)
  - [Trueno: Multi-target Compute](./part3/trueno.md)
  - [Trueno-zram: SIMD Compression](./part3/trueno-zram.md)
  - [Trueno-ublk: GPU Block Device](./part3/trueno-ublk.md)
  - [Repartir: Distributed Computing](./part3/repartir.md)
  - [Pepita: Kernel Interfaces](./part3/pepita.md)
  - [Aprender: First-Principles ML](./part3/aprender.md)
  - [Realizar: ML Inference Runtime](./part3/realizar.md)
  - [Whisper.apr: Speech Recognition](./part3/whisper-apr.md)
  - [trueno-cuda-edge: GPU Edge Testing](./part3/trueno-cuda-edge.md)
- [Model Serving Ecosystem](./part3/model-serving.md)
  - [Banco: AI Workbench](./part3/banco.md)
- [Support Tools](./part3/support-tools.md)
  - [Ruchy: Rust Scripting](./part3/ruchy.md)
  - [PMAT: Static Analysis](./part3/pmat.md)
  - [OIP: Defect Intelligence](./part3/oip.md)
  - [Probar: Runtime Testing](./part3/probar.md)
  - [Renacer: Syscall Tracing](./part3/renacer.md)
- [MCP Tooling](./part3/mcp-tooling.md)
  - [pmcp: Rust MCP SDK](./part3/pmcp.md)
  - [pforge: Declarative MCP Framework](./part3/pforge.md)
- [Visualization & Apps](./part3/viz-apps.md)
  - [Trueno-Viz: GPU Rendering](./part3/trueno-viz.md)
  - [Presentar: App Framework](./part3/presentar.md)
- [Agent Runtime](./part3/agent-runtime.md)
- [Stack Diagnostics & ML Insights](./part3/stack-diagnostics.md)
- [Oracle Mode: Intelligent Query Interface](./part3/oracle-mode.md)
- [Data Platforms Integration](./part3/data-platforms.md)
- [Visualization Frameworks Integration](./part3/visualization-frameworks.md)

---

# Part IV: Practical Examples

- [Example Overview](./part4/example-overview.md)
- [Example 1: Python ML Project](./part4/python-ml-example.md)
  - [NumPy → Trueno Conversion](./part4/numpy-trueno.md)
  - [sklearn → Aprender Migration](./part4/sklearn-aprender.md)
  - [PyTorch → Realizar Integration](./part4/pytorch-realizar.md)
- [Example 2: C Library Migration](./part4/c-library-example.md)
  - [Memory Management](./part4/c-memory.md)
  - [Ownership Inference](./part4/c-ownership.md)
  - [FFI Boundaries](./part4/c-ffi.md)
- [Example 3: Shell Script Conversion](./part4/shell-script-example.md)
  - [Command Parsing](./part4/shell-commands.md)
  - [Error Handling](./part4/shell-errors.md)
  - [CLI Design](./part4/shell-cli.md)
- [Example 4: Mixed-Language Project](./part4/mixed-language-example.md)
  - [Module Boundaries](./part4/mixed-modules.md)
  - [Gradual Migration](./part4/mixed-gradual.md)
  - [Integration Testing](./part4/mixed-testing.md)

---

# Part V: Configuration & Customization

- [Configuration Overview](./part5/config-overview.md)
- [batuta.toml Reference](./part5/config-reference.md)
  - [Project Settings](./part5/config-project.md)
  - [Transpilation Options](./part5/config-transpilation.md)
  - [Optimization Settings](./part5/config-optimization.md)
  - [Validation Configuration](./part5/config-validation.md)
  - [Build Options](./part5/config-build.md)
- [Workflow State Management](./part5/workflow-state.md)
- [Custom Transpiler Flags](./part5/custom-flags.md)

---

# Part VI: CLI Reference

- [Command Overview](./part6/cli-overview.md)
- [`batuta analyze`](./part6/cli-analyze.md)
- [`batuta init`](./part6/cli-init.md)
- [`batuta transpile`](./part6/cli-transpile.md)
- [`batuta optimize`](./part6/cli-optimize.md)
- [`batuta validate`](./part6/cli-validate.md)
- [`batuta build`](./part6/cli-build.md)
- [`batuta report`](./part6/cli-report.md)
- [`batuta status`](./part6/cli-status.md)
- [`batuta reset`](./part6/cli-reset.md)
- [`batuta oracle`](./part6/cli-oracle.md)
- [`batuta stack`](./part6/cli-stack.md)
- [`batuta hf`](./part6/cli-hf.md)
- [`batuta data`](./part6/cli-data.md)
- [`batuta viz`](./part6/cli-viz.md)
- [`batuta content`](./part6/cli-content.md)
- [`batuta falsify`](./part6/cli-falsify.md)
- [`batuta bug-hunter`](./part6/cli-bug-hunter.md)
- [`batuta mcp`](./part6/cli-mcp.md)
- [`batuta playbook`](./part6/cli-playbook.md)
- [`batuta serve`](./part6/cli-serve.md)
- [`batuta deploy`](./part6/cli-deploy.md)
- [`batuta agent`](./part6/cli-agent.md)

---

# Part VII: Best Practices

- [Migration Strategy](./part7/migration-strategy.md)
  - [Greenfield vs Brownfield](./part7/greenfield-brownfield.md)
  - [Risk Assessment](./part7/risk-assessment.md)
  - [Rollback Planning](./part7/rollback.md)
- [Testing Strategy](./part7/testing-strategy.md)
  - [Test Migration](./part7/test-migration.md)
  - [Property-Based Testing](./part7/property-testing.md)
  - [Regression Prevention](./part7/regression.md)
- [Performance Optimization](./part7/performance.md)
  - [Profiling](./part7/profiling.md)
  - [Bottleneck Identification](./part7/bottlenecks.md)
  - [Optimization Iteration](./part7/optimization-iteration.md)
- [Team Workflow](./part7/team-workflow.md)
  - [Parallel Development](./part7/parallel-dev.md)
  - [Code Review Process](./part7/code-review.md)
  - [Knowledge Transfer](./part7/knowledge-transfer.md)

---

# Part VIII: Troubleshooting

- [Common Issues](./part8/common-issues.md)
  - [Transpilation Failures](./part8/transpilation-failures.md)
  - [Type Inference Problems](./part8/type-inference.md)
  - [Lifetime Errors](./part8/lifetime-errors.md)
  - [Performance Regressions](./part8/performance-regressions.md)
- [Debugging Techniques](./part8/debugging.md)
  - [Log Analysis](./part8/log-analysis.md)
  - [Trace Comparison](./part8/trace-comparison.md)
  - [State Inspection](./part8/state-inspection.md)
- [Getting Help](./part8/getting-help.md)
  - [Issue Reporting](./part8/issue-reporting.md)
  - [Community Resources](./part8/community.md)

---

# Part IX: Architecture & Internals

- [Architecture Overview](./part9/architecture-overview.md)
- [Workflow State Machine](./part9/state-machine.md)
- [Tool Detection System](./part9/tool-detection.md)
- [Configuration System](./part9/config-system.md)
- [Playbook Architecture](./part9/playbook-architecture.md)
- [Plugin Architecture (Future)](./part9/plugin-architecture.md)

---

# Appendices

- [Appendix A: Glossary](./appendix/glossary.md)
- [Appendix B: Supported Languages](./appendix/languages.md)
- [Appendix C: Dependency Managers](./appendix/dependency-managers.md)
- [Appendix D: Optimization Profiles](./appendix/optimization-profiles.md)
- [Appendix E: Error Codes](./appendix/error-codes.md)
- [Appendix F: Performance Benchmarks](./appendix/benchmarks.md)
- [Appendix G: Primitive Comparison (Trueno vs PyTorch vs llama.cpp)](./appendix/primitive-comparison.md)
- [Appendix H: Comparison with Alternatives](./appendix/comparison.md)
- [Appendix I: Roadmap](./appendix/roadmap.md)
- [Appendix J: Contributing Guide](./appendix/contributing.md)
- [Appendix K: License](./appendix/license.md)
