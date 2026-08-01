# APR-CLI-QA: Exhaustive CLI Quality Assurance Specification

**Version**: 2.1
**Date**: 2026-04-09
**Status**: ENFORCED (2 consecutive runs)
**Contracts**:
- `contracts/apr-cli-qa-v1.yaml` — baseline (10 equations, 10 falsification tests)
- `contracts/apr-qa-metamorphic-v1.yaml` — quantization equivalence, format roundtrip, multi-arch (5 tests)
- `contracts/apr-qa-silent-fallback-v1.yaml` — bad input injection, truncation, unknown arch (5 tests)
- `contracts/apr-qa-differential-v1.yaml` — ollama parity, tokenizer fidelity, concurrent serve (5 tests)
- `contracts/apr-qa-chaos-v1.yaml` — memory budget, OOM, signals, disk, overwrite protection (5 tests)
- `contracts/apr-qa-coverage-v1.yaml` — category coverage, SATD zero, complexity gate (5 tests)
**Skill**: `.claude/skills/apr-dogfood/SKILL.md`
**Source**: Extended from `apr-cookbook/.claude/skills/qa/SKILL.md`
**arXiv**: 1807.10453, 2207.11976, 2505.03096, 2603.23611, 2103.13630, 2102.05351

---

## Problem

The `apr` CLI has 58 subcommands (57 original + `mcp` added 2026-04-17 via PR #864) but the v1.0 dogfood only tested structural
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
| Structural | apr-cli-qa-v1 | — | 58 commands respond to --help |
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

### Phase 1: Command Grid (58 commands x 3 formats)
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
| S1 | Truncated file (50%) fails `apr validate` — **GH-707 OPEN** | F-SILENT-001 |
| S2 | Bad file (/dev/null) rejected by `apr bench` | F-SILENT-002 |
| S3 | Unknown/SSM architecture fails explicitly (GH-704 fixed) | F-SILENT-003 |
| S4 | Corrupted metadata (zeroed header) rejected | F-SILENT-004 |
| S5 | Missing model exits non-zero | F-SILENT-005 |

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
- Command grid: 58 commands | PASS/FAIL/SKIP count
- GO/WARN/FAIL verdict
- Auto-file issues for FAIL items via `gh issue create`

## Contract Registry

| Contract | Equations | Falsification Tests | Status | First Run |
|----------|-----------|---------------------|--------|-----------|
| apr-cli-qa-v1 | 10 | 10 | tested | 10/10 PASS |
| apr-qa-silent-fallback-v1 | 5 | 5 | **enforced** | 4/5 PASS, S1 FAIL (GH-707) |
| apr-qa-metamorphic-v1 | 3 | 5 | **enforced** | M2+M3 PASS |
| apr-qa-coverage-v1 | 5 | 5 | **enforced** | V1+V3 PASS |
| apr-qa-chaos-v1 | 5 | 5 | **enforced** | C2+C3 PASS |
| apr-qa-differential-v1 | 5 | 5 | **enforced** | D1+D3 PASS, D2 SKIP |
| **Total** | **33** | **35** | | |

### Falsification Run History

#### Run 2 (2026-04-09) — `apr 0.29.3 (a2165629f)`

| Gate | Result | Evidence |
|------|--------|----------|
| G1: Build | PASS | version matches HEAD |
| G2: Grid | PASS | 60/67 PASS, 5 expected non-zero (lint/validate findings), 2 SKIP |
| G3: Protocols | PASS | P1,P2,P4,P7,P8,P9,P10,P11,P12 all PASS |
| G4: Contracts | PASS | 24/24 integration tests |
| G5: Quality | PASS | 4,515 tests, 0 clippy errors |
| G6: Coverage | **WARN** | 94.53% line (target >= 95%) |
| G7: Issues | PASS | 17 open |
| S1 | **FAIL** | GH-707 still open — truncated GGUF exit 0 |
| S2 | PASS | exit 3 |
| S3 | PASS | SSM error (GH-704 fix confirmed) |
| S4 | PASS | exit 5 |
| S5 | PASS | exit 3 |
| M2 | PASS | 6 models, 3 arch families (qwen2, qwen3, qwen35) |
| M3 | PASS | temp=0 deterministic |
| V1 | PASS | 6/6 contracts valid |
| V3 | WARN | hex timeout on 500 MB GGUF (30s limit) |
| C2 | PASS | overwrite blocked (exit 3) |
| C3 | PASS | SIGINT exit 130 |
| D1 | PASS | GGUF=5, APR=5 |
| D2 | SKIP | ollama not wired |
| D3 | PASS | 4/4 JSON valid |
| **Verdict** | **WARN** | S1 tracked (GH-707), coverage 94.53% |

#### Run 1 (2026-04-08) — `apr 0.29.3 (926d7e060)`

| Gate | Result | Evidence |
|------|--------|----------|
| G1-G5 | PASS | 4,577 tests, 0 clippy errors |
| G6 | WARN | 94.53% |
| G7 | PASS | 17 open |
| S1 | **FAIL** | GH-707 — truncated GGUF exit 0 |
| S2-S5 | PASS | all bad inputs rejected |
| M2+M3 | PASS | 6 archs, temp=0 deterministic |
| V1+V3 | PASS | contracts valid, modules exercised |
| C2+C3 | PASS | overwrite + SIGINT |
| D1+D3 | PASS | cross-format + JSON |
| **Verdict** | **WARN** | S1 tracked (GH-707), coverage 94.53% |

### Findings Across Runs

**Stable PASS (confirmed across 2 runs):**
- S2-S5: Bad inputs rejected (exit 3 or 5)
- S3: SSM architecture detection (GH-704 fix holds)
- P4: Cross-subcommand architecture consistency (GH-705 fix holds)
- M2: Multi-architecture inspection (qwen2, qwen3, qwen35)
- M3: Temperature determinism at temp=0
- C3: SIGINT handling (exit 130, no zombie)
- D3: JSON schema stability (4/4 valid)

**Stable FAIL (needs code fix):**
- S1 / GH-707: `apr validate` accepts 50%-truncated GGUF (exit 0). Validator reads
  only tensors that fit in truncated portion and reports success. Fix: compare actual
  readable tensor count against `tensor_count` from GGUF header.

**Stable WARN (known gaps):**
- Coverage 94.53% (0.47% below 95% target)
- V3: `apr hex` on 500 MB GGUF times out at 30s gate limit (not a bug — large model)

### Lessons Learned

1. **Gate script $? vs pipe**: Original gate scripts used `apr cmd | tail -1; EC=$?`
   which captures `tail`'s exit code (always 0), not `apr`'s. Fixed to
   `OUTPUT=$(apr cmd 2>&1); EC=$?` pattern. Three false FAILs (S2, S4, S5) in the
   first prototype were actually script bugs, not code bugs.

2. **lint/validate non-zero exit is by design**: `apr lint` exits 5 when it finds
   lint issues; `apr validate` exits 5 on validation findings. These are correct
   behavior, not failures. Gate 2 grid must distinguish "command ran but found issues"
   (expected) from "command crashed/panicked" (bug).

3. **Architecture coverage matters**: M2 found 3 distinct architecture families
   (qwen2, qwen3, qwen35) across 6 model files. The GH-704 fix (Qwen3.5 SSM
   detection) is confirmed working by S3 across both runs.

## Implementation Priority

| Priority | Contract | Effort | Impact | Status |
|----------|----------|--------|--------|--------|
| P0 | silent-fallback | Low | Critical | **4/5 enforced**, S1 blocked by GH-707 |
| P1 | metamorphic | Medium | High | **2/5 enforced** (M2+M3), M1/M4/M5 need more models |
| P2 | coverage | Low | High | **2/5 enforced** (V1+V3), V2/V4 need pmat wiring |
| P3 | chaos | Medium | Medium | **2/5 enforced** (C2+C3), C1 needs /usr/bin/time |
| P4 | differential | High | High | **2/5 enforced** (D1+D3), D2 needs ollama |

## Open Work

| Item | Blocking | Priority |
|------|----------|----------|
| GH-707: truncated GGUF validate exit 0 | S1 gate | P0 |
| Coverage 94.53% → 95% | G6 gate | P1 |
| Wire D2 ollama parity | D2 gate | P4 |
| Wire V2 SATD + V4 complexity into dogfood | V2/V4 gates | P2 |
| Wire C1 memory budget (/usr/bin/time) | C1 gate | P3 |
| Add M1 format roundtrip (needs small GGUF) | M1 gate | P1 |

## Claude Code Skill

`.claude/skills/apr-dogfood/SKILL.md` — invoked via `/apr-dogfood`:
- 12 gates (G1-G7 structural, G8-G12 v2.0)
- 18 sub-checks (S1-S5, M2-M3, V1+V3, C2-C3, D1-D3)
- GO/WARN/FAIL verdict
- Auto-files issues for FAIL items
