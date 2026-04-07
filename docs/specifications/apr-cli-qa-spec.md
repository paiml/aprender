# APR-CLI-QA: Exhaustive CLI Quality Assurance Specification

**Version**: 1.0
**Date**: 2026-04-07
**Status**: PROPOSAL
**Contract**: `contracts/apr-cli-qa-v1.yaml`
**Skill**: `.claude/skills/dogfood/SKILL.md`
**Source**: Extended from `apr-cookbook/.claude/skills/qa/SKILL.md`

---

## Problem

The `apr` CLI has 57 subcommands but no exhaustive automated QA process
that runs every command against real models, checks exit codes, validates
JSON output, detects silent flag no-ops, and enforces provable contracts.
The apr-cookbook has a fleet QA skill but it's external to the monorepo.

## Goal

A single Claude Code skill (`/dogfood`) in the monorepo that:

1. Rebuilds `apr` from source and verifies version
2. Exercises ALL 57 commands against real models (GGUF, APR, SafeTensors)
3. Runs 12 protocol checks (silent flags, exit-code lies, NaN sentinel, etc.)
4. Validates every command against its provable contract
5. Reports per-command PASS/FAIL/SKIP with evidence
6. Files GitHub issues for bugs found
7. Drives contract-first fixes

## QA Phases

### Phase 0: Build & Install
- `cargo install --path crates/apr-cli --force`
- Verify `apr --version` matches `git rev-parse --short HEAD`
- Contract: FALSIFY-QA-001

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
| Misc | showcase, probar, oracle, code | Varies |

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

### Phase 3: Contract Validation
- Every command in `contracts/apr-cli-commands-v1.yaml`
- Every command has ≥1 falsification test
- `pv lint contracts/` passes

### Phase 4: Report & File Issues

## Provable Contract

`contracts/apr-cli-qa-v1.yaml` defines:
- 57 command equations (help exits 0, model commands handle all 3 formats)
- 12 protocol invariants
- Coverage threshold: 100% of commands tested

## Claude Code Skill

`.claude/skills/dogfood/SKILL.md` — invoked via `/dogfood`:
- Runs all phases
- Reports PASS/FAIL/SKIP per command
- GO/WARN/FAIL verdict
- Files issues for bugs
