---
allowed-tools: Bash(cargo:*), Bash(apr:*), Bash(pmat:*), Bash(gh:*), Bash(git:*), Bash(find:*), Bash(head:*), Bash(tail:*), Bash(wc:*), Bash(grep:*), Bash(diff:*), Bash(timeout:*), Bash(jq:*), Bash(python3:*), Bash(echo:*), Bash(cat:*), Bash(rm:*), Bash(ssh:*), Read, Glob, Grep, Agent
description: Dogfood apr-cli — rebuild, install, exercise all commands against real models, check quality, find next work
---

# APR CLI Exhaustive QA — Contract-First Dogfood

**Contract**: `contracts/apr-cli-qa-v1.yaml` (10 equations, 10 falsification tests)
**Spec**: `docs/specifications/apr-cli-qa-spec.md`
**Source**: Extended from `apr-cookbook/.claude/skills/qa/SKILL.md` (12 protocols)

## Context

- apr-cli local version: !`grep '^version' crates/apr-cli/Cargo.toml | head -1`
- Current git commit: !`git rev-parse --short HEAD`
- Installed apr version: !`apr --version 2>/dev/null || echo "not installed"`
- Available models: !`find ~/models -maxdepth 2 \( -name "*.apr" -o -name "*.gguf" -o -name "*.safetensors" \) -type f 2>/dev/null | wc -l` files
- Test count: !`cargo test -p apr-cli --lib 2>&1 | grep 'test result' | tail -1`

## Arguments

$ARGUMENTS

If arguments include a model path, use that model. Otherwise auto-discover from `~/models`.

## Your Task

Run ALL gates below. For each gate, run the check, report PASS/FAIL/SKIP with evidence. At the end, give a GO/WARN/FAIL verdict. Run independent gates in parallel.

**Do NOT modify files.** This is a read-only audit. If bugs are found, file GitHub issues.

---

## Gate 1: Build & Install (FALSIFY-QA-005)

```bash
cargo install --path crates/apr-cli --force 2>&1 | tail -5
apr --version
git rev-parse --short HEAD
```

PASS if version string contains the HEAD commit hash. FAIL if build errors or mismatch.

## Gate 2: Full Command Grid (FALSIFY-QA-001, FALSIFY-QA-009)

Auto-discover models:
```bash
find ~/models -maxdepth 2 \( -name "*.apr" -o -name "*.gguf" -o -name "*.safetensors" \) -type f 2>/dev/null
```

Pick one per format. For EACH format, exercise ALL command categories:

### 2a. Inspection (11 commands)
```bash
for cmd in inspect debug validate lint tensors trace diff hex tree flow explain; do
  echo -n "$cmd: " && timeout 30 apr $cmd $MODEL 2>&1 | head -1 && echo "OK" || echo "FAIL"
done
```

### 2b. QA & Evaluation (8 commands)
```bash
for cmd in check qa qualify bench eval canary compare-hf parity; do
  echo -n "$cmd: " && timeout 60 apr $cmd $MODEL 2>&1 | head -1 && echo "OK" || echo "SKIP/FAIL"
done
```

### 2c. Transform (9 commands)
```bash
for cmd in convert export import quantize merge prune compile encrypt decrypt; do
  echo -n "$cmd: " && timeout 30 apr $cmd --help 2>&1 | head -1 && echo "OK" || echo "FAIL"
done
```

### 2d. Inference (4 commands) — timeout 60
```bash
apr run $MODEL "What is 2+2?" --max-tokens 16 2>&1 | tail -3
apr serve plan $MODEL 2>&1 | head -5
```

### 2e. Registry (4 commands)
```bash
apr list 2>&1 | head -5
apr list --json 2>&1 | jq length
apr gpu 2>&1 | head -5
```

### 2f. Training & Data (6 commands)
```bash
for cmd in finetune distill train tokenize tune data; do
  echo -n "$cmd: " && timeout 10 apr $cmd --help 2>&1 | head -1 && echo "OK" || echo "FAIL"
done
```

### 2g. UI & Pipeline (7 commands)
```bash
for cmd in tui monitor runs experiment pipeline diagnose showcase; do
  echo -n "$cmd: " && apr $cmd --help 2>&1 | head -1 && echo "OK" || echo "FAIL"
done
```

### 2h. Remaining (8 commands)
```bash
for cmd in rosetta publish oracle probar ptx ptx-map code cbtop; do
  echo -n "$cmd: " && apr $cmd --help 2>&1 | head -1 && echo "OK" || echo "FAIL"
done
```

SKIP (not FAIL) if no models found. FAIL if any command panics or crashes.

## Gate 3: Protocol Checks (12 protocols from apr-cookbook)

### P1. Silent-Flag Protocol (FALSIFY-QA-007)
```bash
diff <(apr inspect $M 2>&1) <(apr inspect --json $M 2>&1)
diff <(apr inspect $M 2>&1) <(apr inspect --vocab $M 2>&1)
diff <(apr list 2>&1) <(apr list --json 2>&1)
```
FAIL if any flag produces identical output (no-op flag).

### P2. Exit-Code Contradiction (FALSIFY-QA-006)
```bash
for cmd in "apr lint $M" "apr validate /nonexistent" "apr rm nonexistent-id"; do
  out=$(eval "$cmd" 2>&1); ec=$?
  echo "$out" | grep -qiE 'error|fail' && [ "$ec" -eq 0 ] && echo "P1 EXIT-CODE LIE: $cmd"
done
```

### P3. Flag-Echo Protocol
```bash
out=$(apr run $M "test" --max-tokens 8 --temperature 0.5 2>&1)
# Verify temperature is actually 0.5, not default
```

### P4. Cross-Subcommand Consistency (FALSIFY-QA-010)
```bash
F_INSPECT=$(apr inspect --json $M 2>/dev/null | jq -r '.architecture // empty')
F_CHECK=$(apr check $M 2>&1 | grep -i arch | head -1)
echo "inspect=$F_INSPECT check=$F_CHECK"
```

### P5. Cache Integrity
```bash
BEFORE=$(apr list 2>&1 | wc -l)
# pull, list, rm cycle should be consistent
```

### P6. GPU/CPU Parity (if GPU available)
```bash
apr gpu 2>&1 | head -3
# If GPU present: compare apr run --device cpu vs --device gpu
```

### P7. NaN/Inf Sentinel (FALSIFY-QA-004)
```bash
for cmd in "apr run $M 'test' --max-tokens 8" "apr bench $M --iterations 1"; do
  timeout 30 eval "$cmd" 2>&1 | grep -qE '\bNaN\b|\bInf\b' && echo "P0 NaN: $cmd"
done
```

### P8. Version Sanity (FALSIFY-QA-005)
```bash
apr --version | grep -qE '\(unknown\)|0000000' && echo "P3 VERSION SENTINEL"
```

### P9. Phantom Subcommand (FALSIFY-QA-008)
```bash
apr --help | awk '/^  [a-z]/{print $1}' | while read cmd; do
  apr "$cmd" --help 2>&1 | grep -qi "not.*implemented" && echo "PHANTOM: $cmd"
done
```

### P10. JSON Schema Stability (FALSIFY-QA-003)
```bash
for cmd in "apr inspect --json $M" "apr list --json" "apr gpu --json"; do
  eval "$cmd" 2>&1 | jq . > /dev/null 2>&1 || echo "P2 INVALID JSON: $cmd"
done
```

### P11. Default-Defamation Protocol
```bash
apr eval $M 2>&1 | grep -qi 'garbage\|broken\|corrupt' && echo "P3 DEFAMATION"
```

### P12. Hardware Cascade Protocol
```bash
# If GPU fails, does CPU fallback work?
apr gpu 2>&1 | head -3
```

## Gate 4: Contract Validation

```bash
# Verify contract is valid
python3 -c "import yaml; yaml.safe_load(open('contracts/apr-cli-qa-v1.yaml')); print('VALID')"

# Run integration tests that enforce the contract
cargo test -p apr-cli --test cli_commands 2>&1 | tail -3
cargo test -p aprender-core --test readme_contract 2>&1 | tail -3
cargo test -p aprender-core --test monorepo_invariants 2>&1 | tail -3
```

PASS if all 3 test suites pass. FAIL if any test fails.

## Gate 5: Code Quality

```bash
cargo test -p apr-cli --lib 2>&1 | grep "test result:"
cargo clippy -p apr-cli --lib -- -D warnings 2>&1 | grep "^error:" | wc -l
```

PASS if 0 test failures and 0 clippy errors. WARN if clippy warnings.

## Gate 6: Coverage Check

```bash
cargo llvm-cov -p aprender-core --lib --no-report 2>&1 | tail -3
cargo llvm-cov report 2>&1 | tail -1
```

PASS if coverage >= 95%.

## Gate 7: Open Issues

```bash
gh issue list --repo paiml/aprender --state open --limit 20
```

Always PASS (informational).

---

## Verdict

After all gates, provide:

1. **Summary table**: Gate | Status | Notes
2. **Protocol results**: P1-P12 | PASS/FAIL
3. **Command grid**: 57 commands | PASS/FAIL/SKIP count
4. **GO** if all gates pass
5. **WARN** if soft issues only (no panics, no data corruption)
6. **FAIL** if panics, exit-code lies, or contract violations

If bugs found, file with:
```bash
gh issue create --repo paiml/aprender --title "apr <cmd>: <title>" --body "..."
```

## Cleanup

```bash
rm -f /tmp/apr-qa-*.{gguf,apr,enc,jsonl,safetensors}
```
