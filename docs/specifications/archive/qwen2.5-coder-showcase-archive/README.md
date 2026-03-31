# Qwen2.5-Coder Showcase: Historical Archive

This directory contains archived content from the Qwen2.5-Coder Showcase specification.
The main spec (`../qwen2.5-coder-showcase-demo.md`) was condensed from ~9,162 lines to ~1,600 lines (v9),
then rewritten to ~950 lines (v10) with single-model focus (7B), then slimmed to ~780 lines (v10.46.0).
All displaced content preserved here.

**No content was lost.** All archived sections are preserved here with original line references.
Duplicate sections (lines 4738-4923 duplicating Round 10 protocols, lines 4924-4992 duplicating Appendix H)
were identified and intentionally omitted.

## Table of Contents

### v10.46.0 Archive (current-state sections moved from main spec)

| File | Description | Lines |
|------|-------------|-------|
| [checklist-300-point.md](checklist-300-point.md) | Section 6: 300-Point Falsification Checklist (passing/incomplete sections, 5 F-CHECKLIST gates) | ~37 |
| [ollama-parity-protocol.md](ollama-parity-protocol.md) | Section 7A: Ollama Parity Protocol (prerequisites, 3 test procedures, 5 F-OLLAMA gates) | ~68 |
| [rosetta-conversion.md](rosetta-conversion.md) | Section 10: Rosetta Format Conversion (canonical import path, 3 primary paths, 6 F-ROSETTA gates) | ~69 |
| [rosetta-diagnostics.md](rosetta-diagnostics.md) | Section 11: Rosetta ML Diagnostics + Hex Forensics + Model Profiling + Performance Sprint (23 gates) | ~195 |
| [performance-protocol.md](performance-protocol.md) | Section 12: Performance Falsification Protocol (KV Cache, 7B targets, 7 F-PERF gates) | ~46 |
| [trueno-compute-layer.md](trueno-compute-layer.md) | Section 13: Trueno Compute Layer (95 kernels, 9 backends, CUDA, WGSL, PTX analysis, 17 gates) | ~299 |
| [realizar-architecture.md](realizar-architecture.md) | Section 14: Realizar Inference Architecture (two-phase gen, KV cache, GQA, sampling, API, 13 gates) | ~184 |
| [cli-surface-verification.md](cli-surface-verification.md) | Section 17: Full CLI Surface Area Verification (49 commands registry, 5 F-SURFACE gates) | ~76 |
| [self-falsification-rounds.md](self-falsification-rounds.md) | Section 18: Spec Self-Falsification Audit (46 rounds, 214 bugs, Five-Whys methodology) | ~569 |
| [mvp-qualification.md](mvp-qualification.md) | Section 21: MVP Qualification (18-cell matrix, G1-G4 gateways, oracle verification, 7 F-MVP gates) | ~219 |
| [appendices-b-through-g.md](appendices-b-through-g.md) | Appendices B-G: PMAT tickets, open issues, Q4_K spec, SafeTensors spec | ~116 |

### Historical archive (pre-v10 content)

| File | Description | Original Lines | Lines |
|------|-------------|----------------|-------|
| [rounds-44-53-recent.md](rounds-44-53-recent.md) | Rounds 44-53 progress (most recent falsification rounds) | 1-427 | ~430 |
| [certification-and-issues.md](certification-and-issues.md) | Round 39 certification results, GitHub issues status | 429-604 | ~180 |
| [quality-philosophy.md](quality-philosophy.md) | Toyota Way quality philosophy, SATD policy, falsification principles | 605-688 | ~88 |
| [pmat-tickets-detailed.md](pmat-tickets-detailed.md) | Detailed PMAT ticket writeups (PMAT-120 through PMAT-205) | 689-1568 | ~884 |
| [critical-failures-resolved.md](critical-failures-resolved.md) | All resolved critical failures and falsifications | 1569-2387 | ~823 |
| [appendix-historical-bugs.md](appendix-historical-bugs.md) | Appendix D: Historical bug fixes (2026-01-21 to 2026-01-28) | 3785-4057 | ~277 |
| [appendix-epistemological.md](appendix-epistemological.md) | Appendix E & I: Epistemological audit (Dr. Popper) + Kabuki Theatre | 4058-4114 | ~61 |
| [t-series-falsification.md](t-series-falsification.md) | Section 13: T-Series and CLI falsification test results | 4115-4359 | ~249 |
| [protocol-evolution-rounds-5-13.md](protocol-evolution-rounds-5-13.md) | Protocol evolution for rounds 5-13 (deduplicated) | 4360-4737 | ~383 |
| [appendix-toyota-tps.md](appendix-toyota-tps.md) | Appendix G: Toyota Production System integration details | 5140-5271 | ~136 |
| [round-12-popperian-audit.md](round-12-popperian-audit.md) | Section 20: Round 12 Popperian audit (100-point checklist) | 5296-5633 | ~342 |
| [rounds-14-24.md](rounds-14-24.md) | Sections 21-31: Rounds 14-24 (tensor import, conversion fixes) | 5634-7928 | ~2,299 |
| [rounds-35-45.md](rounds-35-45.md) | Sections 32-45: Rounds 35-45 (SafeTensors QA, GPU parity) | 7929-9162 | ~1,238 |
| [popperian-advanced-protocols.md](popperian-advanced-protocols.md) | Appendix F: Advanced falsification protocols (Bold Conjectures) | 8065-8136 | ~76 |
| [v9-1.5b-results.md](v9-1.5b-results.md) | v9 1.5B-era results: Section 12.1 diff tracing, QA playbook, UX table, resolved issues | v9.30.0 | ~430 |

**Total archived: ~9,775 lines** (~1,878 new + ~7,897 historical)

## Removed Duplicates

The following sections were identified as exact duplicates and were NOT archived:

- **Lines 4738-4923**: Duplicate of Protocol Evolution Rounds 10-13 (already at lines 4514-4700)
- **Lines 4924-4992**: Duplicate of Appendix H: Cross-Format Invariant Protocol (already at lines 4700-4737)
