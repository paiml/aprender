# APR-CLI-QA: Exhaustive CLI Quality Assurance Specification

**Version**: 2.0
**Date**: 2026-04-08
**Status**: ACTIVE
**Contracts**:
- `contracts/apr-cli-qa-v1.yaml` — baseline (10 equations, 10 falsification tests)
- `contracts/apr-qa-metamorphic-v1.yaml` — quantization equivalence, format roundtrip, multi-arch (5 tests)
- `contracts/apr-qa-silent-fallback-v1.yaml` — bad input injection, truncation, unknown arch (5 tests)
- `contracts/apr-qa-differential-v1.yaml` — ollama parity, tokenizer fidelity, concurrent serve (5 tests)
- `contracts/apr-qa-chaos-v1.yaml` — memory budget, OOM, signals, disk, overwrite protection (5 tests)
- `contracts/apr-qa-coverage-v1.yaml` — category coverage, SATD zero, complexity gate (5 tests)
**Skill**: `.claude/skills/dogfood/SKILL.md`
**Source**: Extended from `apr-cookbook/.claude/skills/qa/SKILL.md`
**arXiv**: 1807.10453, 2207.11976, 2505.03096, 2603.23611, 2103.13630, 2102.05351

---

## Problem

The `apr` CLI has 57 subcommands but the v1.0 dogfood only tested structural
properties (help exits 0, JSON is valid, no NaN). Historical bug analysis
(GH-336 through GH-439) shows the gaps that slipped past:

1. **Silent degradation** — commands swallow errors and emit garbage (GH-336, GH-337, GH-339)
2. **Single-architecture bias** — all tests use Qwen2; hardcoded constants break on LLaMA/Phi/Gemma (GH-317-322)
3. **No conversion fidelity** — GGUF→APR→GGUF roundtrip corruption undetected (GH-172, GH-208)
4. **No resource bounds** — OOM on 32B+ models (GH-434, GH-478), 55 GB RAM on pull (GH-352)
5. **No concurrent/batch testing** — race conditions in batch mode (GH-482, GH-483)

## Goal

A 6-contract dogfood system that covers 5 dimensions beyond structural testing:

| Dimension | Contract | arXiv Basis | Key Gate |
|-----------|----------|-------------|----------|
| Structural | apr-cli-qa-v1 | — | 57 commands respond to --help |
| Metamorphic | apr-qa-metamorphic-v1 | 1807.10453, 2603.23611 | Q6K≈Q4K output similarity |
| Silent fallback | apr-qa-silent-fallback-v1 | 2505.03096 | Truncated/corrupt files fail LOUD |
| Differential | apr-qa-differential-v1 | 2207.11976, 2406.07944 | apr≈ollama top-1 token parity |
| Chaos | apr-qa-chaos-v1 | 2505.03096 | RSS < 3x model + 512 MB |
| Coverage | apr-qa-coverage-v1 | 2102.05351, 1906.10742 | 0 High SATD, CC < 15 tested |

Total: **6 contracts, 35 equations, 35 falsification tests**.

## QA Phases

### Phase 0: Build & Install
- `cargo install --path crates/apr-cli --force`
- Verify `apr --version` matches `git rev-parse --short HEAD`
- Contract: FALSIFY-QA-005

### Phase 1: Command Grid (57 commands x 3 formats)
For each of GGUF, APR, SafeTensors models:

| Category | Commands | Model Required |
|----------|----------|----------------|
| Inspection | inspect, debug, validate, lint, tensors, trace, diff, hex, tree, flow, explain | Yes |
| Inference | run, chat, serve, bench, eval | Yes |
| Transform | convert, export, import, quantize, merge, prune, compile, encrypt, decrypt | Yes |
| Training | finetune, distill, train, tokenize, tune | Yes |
| Registry | pull, list, rm, publish | No |
| Hardware | gpu, profile, parity, ptx, ptx-map, cbtop | Varies |
| QA | check, qa, qualify, canary, compare-hf, rosetta | Yes |
| UI/Monitor | tui, monitor, runs, experiment | No |
| Pipeline | data, pipeline, diagnose | No |
| Misc | showcase, probar, oracle | Varies |

### Phase 2: Protocol Checks (12 protocols from apr-cookbook)
- P1: Silent-Flag Protocol (--json, --quiet, --verbose, --vocab, --drama, --strict)
- P2: Exit-Code Contradiction Protocol
- P3: Flag-Echo Protocol (--rank, --max-tokens, --temperature)
- P4: Cross-Subcommand Consistency (architecture/family agreement)
- P5: Cache Registry Integrity (pull/list/rm consistency)
- P6: GPU/CPU Parity Protocol
- P7: NaN/Inf Sentinel Protocol
- P8: Version Sanity Protocol
- P9: Phantom Subcommand Protocol
- P10: JSON Schema Stability Protocol
- P11: Default-Defamation Protocol
- P12: Hardware Cascade Protocol

### Phase 3: Metamorphic Testing (NEW — v2.0)
Contract: `apr-qa-metamorphic-v1.yaml`

| Gate | Test | Falsification |
|------|------|---------------|
| M1 | Quantization equivalence: Q6K vs Q4K top-5 token overlap ≥ 3/5 | F-META-001 |
| M2 | Format roundtrip: GGUF→APR→GGUF tensor L2 drift < 1% | F-META-002 |
| M3 | Multi-architecture smoke: 3+ arch families produce coherent output | F-META-003 |
| M4 | Prompt invariance: rephrased prompts → same semantic answer | F-META-004 |
| M5 | Temperature determinism: temp=0 → identical output across 3 runs | F-META-005 |

**Academic basis**: METTLE (1807.10453) defines metamorphic relations for ML
without ground truth. LLMORPH (2603.23611) catalogs 191 metamorphic relations
for NLP tasks. Quantization Survey (2103.13630) provides mathematical bounds
for acceptable divergence per quant level.

### Phase 4: Silent-Fallback Injection (NEW — v2.0)
Contract: `apr-qa-silent-fallback-v1.yaml`

| Gate | Test | Falsification |
|------|------|---------------|
| S1 | Truncated file (50%) fails `apr validate` | F-SILENT-001 |
| S2 | 0 tok/s benchmark exits non-zero | F-SILENT-002 |
| S3 | Unknown architecture fails explicitly | F-SILENT-003 |
| S4 | Missing tokenizer warns, never garbles | F-SILENT-004 |
| S5 | Corrupted metadata rejected | F-SILENT-005 |

**Root cause**: GH-336 (silent 0 tok/s), GH-337 (byte-level garble), GH-339
(silent chat template fallback), GH-439 (silent `_ => default` match arms).
The fix is adversarial input injection: if bad input produces "valid-looking"
output, the dogfood FAILS.

### Phase 5: Differential Testing (NEW — v2.0)
Contract: `apr-qa-differential-v1.yaml`

| Gate | Test | Falsification |
|------|------|---------------|
| D1 | Ollama parity: top-1 token agreement at temp=0 | F-DIFF-001 |
| D2 | Tokenizer roundtrip: encode→decode = identity | F-DIFF-002 |
| D3 | Serve concurrent parity: 3 parallel requests → same output | F-DIFF-003 |
| D4 | Perplexity budget: Q4K PPL within 10% of F16 | F-DIFF-004 |
| D5 | Cross-format tensor L2 norm agreement | F-DIFF-005 |

**Academic basis**: Differential testing (2207.11976) catches bugs by comparing
implementations. DLLens (2406.07944) auto-generates counterpart inputs. llama.cpp
tracks delta-PPL per quant level; ollama has 24 integration test files with
per-architecture coverage.

### Phase 6: Chaos Engineering (NEW — v2.0)
Contract: `apr-qa-chaos-v1.yaml`

| Gate | Test | Falsification |
|------|------|---------------|
| C1 | Memory budget: RSS < 3x model + 512 MB | F-CHAOS-001 |
| C2 | Graceful OOM: exits with error, not SIGSEGV | F-CHAOS-002 |
| C3 | SIGINT handling: exits 130, no corrupt state | F-CHAOS-003 |
| C4 | Overwrite protection: no silent file clobber | F-CHAOS-004 |
| C5 | Disk exhaustion: error, not partial corrupt file | F-CHAOS-005 |

**Academic basis**: Chaos framework (2505.03096) defines fault injection
categories for LLM systems: resource starvation, communication failure,
and state corruption.

### Phase 7: Coverage Completeness (NEW — v2.0)
Contract: `apr-qa-coverage-v1.yaml`

| Gate | Test | Falsification |
|------|------|---------------|
| V1 | All 10 command categories ≥ 80% coverage | F-COV-001 |
| V2 | No high-impact uncovered function untracked | F-COV-002 |
| V3 | No CC > 15 function untested | F-COV-003 |
| V4 | Zero High-severity SATD | F-COV-004 |
| V5 | 6 critical modules (hex, profile, cbtop, train, chat, serve) exercised | F-COV-005 |

**PMAT findings**: `sliding_window_entropy.rs` (10+ functions, 0% coverage),
`speedup.rs` (5 functions, 0% coverage), `compute_roofline` (CC=24),
4 High-severity SATD in `cbtop_measure_batch.rs`.

### Phase 8: Report & File Issues
- Summary table: Gate | Status | Notes
- Protocol results: P1-P12 | PASS/FAIL
- Command grid: 57 commands | PASS/FAIL/SKIP count
- GO/WARN/FAIL verdict
- Auto-file issues for FAIL items via `gh issue create`

## Contract Registry

| Contract | Equations | Falsification Tests | Status |
|----------|-----------|---------------------|--------|
| apr-cli-qa-v1 | 10 | 10 | tested |
| apr-qa-metamorphic-v1 | 3 | 5 | proposed |
| apr-qa-silent-fallback-v1 | 5 | 5 | proposed |
| apr-qa-differential-v1 | 5 | 5 | proposed |
| apr-qa-chaos-v1 | 5 | 5 | proposed |
| apr-qa-coverage-v1 | 5 | 5 | proposed |
| **Total** | **33** | **35** | |

## Implementation Priority

| Priority | Contract | Effort | Impact | Rationale |
|----------|----------|--------|--------|-----------|
| P0 | silent-fallback | Low | Critical | Prevents shipping garbage — bad inputs must fail LOUD |
| P1 | metamorphic | Medium | High | Catches quant corruption + multi-arch hardcoding (6+ past bugs) |
| P2 | coverage | Low | High | PMAT already provides data — just wire into dogfood |
| P3 | chaos | Medium | Medium | Memory/OOM bugs rare but catastrophic when they hit |
| P4 | differential | High | High | Requires ollama installed + reference models available |

## Claude Code Skill

`.claude/skills/dogfood/SKILL.md` — invoked via `/dogfood`:
- Runs all 8 phases
- Reports PASS/FAIL/SKIP per gate
- GO/WARN/FAIL verdict
- Files issues for bugs with contract references
