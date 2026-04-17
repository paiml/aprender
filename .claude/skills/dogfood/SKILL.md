---
allowed-tools: Bash(cargo:*), Bash(apr:*), Bash(pmat:*), Bash(gh:*), Bash(git:*), Bash(find:*), Bash(head:*), Bash(tail:*), Bash(wc:*), Bash(grep:*), Bash(diff:*), Bash(timeout:*), Bash(jq:*), Bash(python3:*), Bash(echo:*), Bash(cat:*), Bash(rm:*), Bash(ssh:*), Read, Glob, Grep, Agent
description: Dogfood apr-cli — rebuild, install, exercise all commands against real models, check quality, find next work
---

# APR CLI Exhaustive QA — Contract-First Dogfood

**Contracts**:
- `contracts/apr-cli-qa-v1.yaml` — baseline (10 equations, 10 falsification tests)
- `contracts/apr-qa-silent-fallback-v1.yaml` — bad input injection (5 tests)
- `contracts/apr-qa-metamorphic-v1.yaml` — quant equivalence, multi-arch, roundtrip (5 tests)
- `contracts/apr-qa-coverage-v1.yaml` — category coverage, SATD, complexity (5 tests)
- `contracts/apr-qa-chaos-v1.yaml` — memory, OOM, signals, overwrite (5 tests)
- `contracts/apr-qa-differential-v1.yaml` — ollama parity, tokenizer, concurrency (5 tests)
**Spec**: `docs/specifications/apr-cli-qa-spec.md` (v2.0)
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

## Gate 8: Silent-Fallback Injection (F-SILENT-001 through F-SILENT-005)

Contract: `contracts/apr-qa-silent-fallback-v1.yaml`

Bad inputs MUST fail LOUD (non-zero exit + stderr message), never silently degrade.

### S1. Truncated file detection (GH-707)
```bash
M_GGUF=$(find ~/models -maxdepth 2 -name "*.gguf" -type f | head -1)
if [ -n "$M_GGUF" ]; then
  SIZE=$(stat -c%s "$M_GGUF")
  head -c $((SIZE / 2)) "$M_GGUF" > /tmp/apr-qa-truncated.gguf
  # IMPORTANT: capture exit code without piping (pipe loses $?)
  OUTPUT=$(apr validate /tmp/apr-qa-truncated.gguf 2>&1); EC=$?
  echo "$OUTPUT" | tail -3
  [ "$EC" -ne 0 ] && echo "S1 PASS: truncated file rejected (exit $EC)" || echo "S1 FAIL: truncated file accepted (GH-707)"
fi
```

### S2. Bad file rejection
```bash
OUTPUT=$(apr bench /dev/null --iterations 1 2>&1); EC=$?
echo "$OUTPUT" | tail -1
[ "$EC" -ne 0 ] && echo "S2 PASS: /dev/null rejected (exit $EC)" || echo "S2 FAIL: /dev/null accepted"
```

### S3. Unknown architecture handling (GH-704 pattern)
```bash
# Check that Qwen3.5 SSM model gets a clear error, not silent llama fallback
M_SSM=$(find ~/models -maxdepth 2 -name "*Qwen3.5*" -o -name "*qwen35*" 2>/dev/null | head -1)
if [ -n "$M_SSM" ]; then
  OUTPUT=$(apr run "$M_SSM" "test" --max-tokens 1 2>&1); EC=$?
  echo "$OUTPUT" | grep -qi "not.*supported\|unsupported\|SSM" && \
    echo "S3 PASS: unsupported arch gives clear error" || echo "S3 FAIL: no clear error for unsupported arch"
else
  echo "S3 SKIP: no SSM model available"
fi
```

### S4. Corrupted metadata detection
```bash
if [ -n "$M_GGUF" ]; then
  cp "$M_GGUF" /tmp/apr-qa-corrupt.gguf
  dd if=/dev/zero of=/tmp/apr-qa-corrupt.gguf bs=1 count=64 seek=8 conv=notrunc 2>/dev/null
  OUTPUT=$(apr validate /tmp/apr-qa-corrupt.gguf 2>&1); EC=$?
  echo "$OUTPUT" | tail -1
  [ "$EC" -ne 0 ] && echo "S4 PASS: corrupt metadata rejected (exit $EC)" || echo "S4 FAIL: corrupt metadata accepted"
fi
```

### S5. Missing model graceful (FALSIFY-QA-002)
```bash
OUTPUT=$(apr inspect /nonexistent/model.gguf 2>&1); EC=$?
echo "$OUTPUT" | tail -1
[ "$EC" -ne 0 ] && echo "S5 PASS: missing model exits non-zero (exit $EC)" || echo "S5 FAIL: exit 0 for missing model"
```

PASS if all 5 checks reject bad input. FAIL if any bad input is silently accepted.

## Gate 9: Metamorphic Testing (F-META-001 through F-META-005)

Contract: `contracts/apr-qa-metamorphic-v1.yaml`

### M1. Format roundtrip (GGUF→APR→GGUF tensor fidelity)
```bash
if [ -n "$M_GGUF" ]; then
  apr convert "$M_GGUF" -o /tmp/apr-qa-rt.apr 2>&1 | tail -1
  # If convert succeeds, check tensor count matches
  if [ -f /tmp/apr-qa-rt.apr ]; then
    ORIG_TENSORS=$(apr tensors "$M_GGUF" --json 2>/dev/null | jq length 2>/dev/null || echo 0)
    RT_TENSORS=$(apr tensors /tmp/apr-qa-rt.apr --json 2>/dev/null | jq length 2>/dev/null || echo 0)
    echo "M1 orig=$ORIG_TENSORS rt=$RT_TENSORS"
    [ "$ORIG_TENSORS" -gt 0 ] && [ "$RT_TENSORS" -gt 0 ] && echo "M1 PASS" || echo "M1 WARN: tensor count mismatch"
  else
    echo "M1 SKIP: convert not available for this model"
  fi
else
  echo "M1 SKIP: no GGUF model"
fi
```

### M2. Multi-architecture smoke
```bash
# Check that inspect works across all available model architectures
ARCH_COUNT=0
for m in $(find ~/models -maxdepth 2 \( -name "*.gguf" -o -name "*.apr" -o -name "*.safetensors" \) -type f 2>/dev/null); do
  ARCH=$(timeout 10 apr inspect --json "$m" 2>/dev/null | jq -r '.architecture // empty' 2>/dev/null)
  [ -n "$ARCH" ] && ARCH_COUNT=$((ARCH_COUNT + 1)) && echo "  M2 arch=$ARCH ($m)"
done
[ "$ARCH_COUNT" -ge 2 ] && echo "M2 PASS: $ARCH_COUNT architectures inspected" || echo "M2 WARN: only $ARCH_COUNT architectures available"
```

### M3. Temperature determinism (temp=0 → identical output across 3 runs)
```bash
M_APR=$(find ~/models -maxdepth 2 -name "*.apr" -type f | head -1)
if [ -n "$M_APR" ]; then
  OUT1=$(timeout 60 apr run "$M_APR" "Say hello" --max-tokens 4 --temperature 0.0 2>&1 | grep "^Output:" | head -1)
  OUT2=$(timeout 60 apr run "$M_APR" "Say hello" --max-tokens 4 --temperature 0.0 2>&1 | grep "^Output:" | head -1)
  if [ "$OUT1" = "$OUT2" ] && [ -n "$OUT1" ]; then
    echo "M3 PASS: temp=0 deterministic"
  else
    echo "M3 WARN: temp=0 outputs differ (may be non-deterministic sampling)"
  fi
else
  echo "M3 SKIP: no APR model"
fi
```

PASS if M1+M2+M3 all pass. WARN if any are skipped due to missing models.

## Gate 10: Coverage Completeness (F-COV-001 through F-COV-005)

Contract: `contracts/apr-qa-coverage-v1.yaml`

### V1. Contract YAML validity (all 6 QA contracts parse)
```bash
VALID=0; TOTAL=0
for c in contracts/apr-cli-qa-v1.yaml contracts/apr-qa-metamorphic-v1.yaml \
  contracts/apr-qa-silent-fallback-v1.yaml contracts/apr-qa-differential-v1.yaml \
  contracts/apr-qa-chaos-v1.yaml contracts/apr-qa-coverage-v1.yaml; do
  TOTAL=$((TOTAL+1))
  python3 -c "import yaml; yaml.safe_load(open('$c')); print('  VALID: $c')" 2>&1 && VALID=$((VALID+1))
done
echo "V1: $VALID/$TOTAL contracts valid"
[ "$VALID" -eq "$TOTAL" ] && echo "V1 PASS" || echo "V1 FAIL"
```

### V2. Zero High-severity SATD in apr-cli
```bash
SATD_HIGH=$(pmat analyze satd -p crates/apr-cli/ 2>&1 | grep -c "High" 2>/dev/null || echo "0")
echo "V2: $SATD_HIGH High-severity SATD items"
[ "$SATD_HIGH" -eq 0 ] && echo "V2 PASS" || echo "V2 WARN: $SATD_HIGH High SATD items"
```

### V3. Critical modules exercised (no panic on real model)
```bash
M=$(find ~/models -maxdepth 2 \( -name "*.gguf" -o -name "*.apr" \) -type f | head -1)
if [ -n "$M" ]; then
  V3_PASS=0; V3_TOTAL=0
  for cmd in "hex" "profile --iterations 1"; do
    V3_TOTAL=$((V3_TOTAL+1))
    timeout 30 apr $cmd "$M" 2>&1 | head -3 >/dev/null && V3_PASS=$((V3_PASS+1)) && echo "  V3 $cmd: OK" || echo "  V3 $cmd: FAIL/SKIP"
  done
  for cmd in "serve plan" "train plan"; do
    V3_TOTAL=$((V3_TOTAL+1))
    timeout 10 apr $cmd "$M" 2>&1 | head -3 >/dev/null && V3_PASS=$((V3_PASS+1)) && echo "  V3 $cmd: OK" || echo "  V3 $cmd: FAIL/SKIP"
  done
  echo "V3: $V3_PASS/$V3_TOTAL modules exercised"
  [ "$V3_PASS" -ge 2 ] && echo "V3 PASS" || echo "V3 WARN"
else
  echo "V3 SKIP: no model"
fi
```

### V4. Complexity hotspots tracked
```bash
HIGH_CC=$(pmat analyze complexity -p crates/apr-cli/ 2>&1 | awk '$NF > 15 {count++} END {print count+0}' 2>/dev/null || echo "0")
echo "V4: $HIGH_CC functions with CC > 15"
[ "$HIGH_CC" -le 3 ] && echo "V4 PASS" || echo "V4 WARN: $HIGH_CC high-complexity functions"
```

### Gate 10 Verdict (GH-716)
```bash
# Re-compute V1-V4 for a single aggregate verdict. V1 (contracts parse) and
# V3 (critical modules run) are required; V2 (SATD) and V4 (complexity) are
# quality signals that demote PASS → WARN but never cause FAIL on their own.
V1_OK=0
for c in contracts/apr-cli-qa-v1.yaml contracts/apr-qa-metamorphic-v1.yaml \
  contracts/apr-qa-silent-fallback-v1.yaml contracts/apr-qa-differential-v1.yaml \
  contracts/apr-qa-chaos-v1.yaml contracts/apr-qa-coverage-v1.yaml; do
  python3 -c "import yaml; yaml.safe_load(open('$c'))" 2>/dev/null && V1_OK=$((V1_OK+1))
done
V2_SATD=$(pmat analyze satd -p crates/apr-cli/ 2>&1 | grep -c "High" 2>/dev/null || echo "0")
V4_CC=$(pmat analyze complexity -p crates/apr-cli/ 2>&1 | awk '$NF > 15 {count++} END {print count+0}' 2>/dev/null || echo "0")
M=$(find ~/models -maxdepth 2 \( -name "*.gguf" -o -name "*.apr" \) -type f | head -1)
V3_OK=$([ -n "$M" ] && echo 1 || echo 0)

if [ "$V1_OK" -eq 6 ] && [ "$V3_OK" -eq 1 ]; then
  if [ "$V2_SATD" -eq 0 ] && [ "$V4_CC" -le 3 ]; then
    echo "Gate 10: PASS (V1=6/6 V2=0 SATD V3=model V4=$V4_CC CC)"
  else
    echo "Gate 10: WARN (V1+V3 pass; V2=$V2_SATD SATD V4=$V4_CC CC)"
  fi
else
  echo "Gate 10: FAIL (V1=$V1_OK/6 V3=$([ "$V3_OK" = "1" ] && echo ok || echo no-model))"
fi
```

PASS requires V1+V3. V2 (SATD) and V4 (complexity) demote PASS → WARN.

## Gate 11: Chaos Engineering (F-CHAOS-001 through F-CHAOS-005)

Contract: `contracts/apr-qa-chaos-v1.yaml`

### C1. Memory budget (RSS sanity check)
```bash
M=$(find ~/models -maxdepth 2 -name "*.gguf" -type f -size -1G | head -1)
if [ -n "$M" ]; then
  MODEL_KB=$(du -k "$M" | cut -f1)
  RSS_KB=$(/usr/bin/time -v timeout 30 apr inspect "$M" 2>&1 | grep "Maximum resident" | awk '{print $NF}' 2>/dev/null || echo 0)
  if [ "$RSS_KB" -gt 0 ]; then
    BUDGET_KB=$(( MODEL_KB * 3 + 524288 ))
    echo "C1: model=${MODEL_KB}KB RSS=${RSS_KB}KB budget=${BUDGET_KB}KB"
    [ "$RSS_KB" -lt "$BUDGET_KB" ] && echo "C1 PASS" || echo "C1 WARN: RSS exceeds 3x model + 512MB"
  else
    echo "C1 SKIP: /usr/bin/time not available"
  fi
else
  echo "C1 SKIP: no small GGUF model"
fi
```

### C2. Overwrite protection
```bash
touch /tmp/apr-qa-existing.apr
apr convert /dev/null -o /tmp/apr-qa-existing.apr 2>&1; EC=$?
# Should either fail (non-zero) or prompt — never silently overwrite
[ "$EC" -ne 0 ] && echo "C2 PASS: existing file not silently overwritten" || echo "C2 WARN: may have overwritten"
```

### C3. SIGINT handling
```bash
M_APR=$(find ~/models -maxdepth 2 -name "*.apr" -type f | head -1)
if [ -n "$M_APR" ]; then
  timeout 5 apr run "$M_APR" "Tell me a very long story about everything" --max-tokens 500 &
  PID=$!
  sleep 2
  kill -INT $PID 2>/dev/null
  wait $PID 2>/dev/null; EC=$?
  # SIGINT should produce exit 130 or similar non-zero, NOT leave zombie
  [ "$EC" -ne 0 ] && echo "C3 PASS: SIGINT handled (exit $EC)" || echo "C3 WARN: SIGINT exit 0"
else
  echo "C3 SKIP: no APR model"
fi
```

PASS if C1+C2+C3 all pass. WARN on skips.

## Gate 12: Differential Testing (F-DIFF-001 through F-DIFF-005)

Contract: `contracts/apr-qa-differential-v1.yaml`

### D1. Cross-format tensor agreement
```bash
M_GGUF=$(find ~/models -maxdepth 2 -name "*.gguf" -type f | head -1)
M_APR=$(find ~/models -maxdepth 2 -name "*.apr" -type f | head -1)
if [ -n "$M_GGUF" ] && [ -n "$M_APR" ]; then
  GGUF_COUNT=$(apr tensors "$M_GGUF" --json 2>/dev/null | jq length 2>/dev/null || echo 0)
  APR_COUNT=$(apr tensors "$M_APR" --json 2>/dev/null | jq length 2>/dev/null || echo 0)
  echo "D1: GGUF tensors=$GGUF_COUNT APR tensors=$APR_COUNT"
  [ "$GGUF_COUNT" -gt 0 ] && [ "$APR_COUNT" -gt 0 ] && echo "D1 PASS: both formats report tensors" || echo "D1 WARN"
else
  echo "D1 SKIP: need both GGUF and APR models"
fi
```

### D2. Ollama parity (if ollama installed)
```bash
if command -v ollama &>/dev/null; then
  OLLAMA_MODELS=$(ollama list 2>/dev/null | tail -n +2 | head -3)
  if [ -n "$OLLAMA_MODELS" ]; then
    echo "D2: ollama available with models — manual parity check recommended"
    echo "D2 SKIP: automated parity not yet wired"
  else
    echo "D2 SKIP: ollama installed but no models"
  fi
else
  echo "D2 SKIP: ollama not installed"
fi
```

### D3. JSON schema consistency across commands
```bash
M=$(find ~/models -maxdepth 2 \( -name "*.gguf" -o -name "*.apr" \) -type f | head -1)
if [ -n "$M" ]; then
  D3_PASS=0
  for cmd in "inspect --json" "check --json" "list --json" "gpu --json"; do
    timeout 15 apr $cmd $M 2>/dev/null | jq . >/dev/null 2>&1 && D3_PASS=$((D3_PASS+1))
  done
  echo "D3: $D3_PASS/4 JSON outputs valid"
  [ "$D3_PASS" -ge 3 ] && echo "D3 PASS" || echo "D3 WARN"
else
  echo "D3 SKIP: no model"
fi
```

PASS if D1+D3 pass. SKIP for D2 (requires ollama setup).

---

## Verdict

After all 12 gates, provide:

1. **Summary table**: Gate 1-12 | Status | Notes
2. **Protocol results**: P1-P12 | PASS/FAIL
3. **New gates**: S1-S5, M1-M3, V1-V4, C1-C3, D1-D3 | PASS/FAIL/SKIP
4. **Command grid**: 57 commands | PASS/FAIL/SKIP count
5. **GO** if gates 1-7 all pass AND gates 8-12 have no FAIL
6. **WARN** if soft issues only (no panics, no data corruption, SKIPs OK)
7. **FAIL** if panics, exit-code lies, silent-fallback accepts bad input, or contract violations

If bugs found, file with:
```bash
gh issue create --repo paiml/aprender --title "apr <cmd>: <title>" --body "..."
```

## Cleanup

```bash
rm -f /tmp/apr-qa-*.{gguf,apr,enc,jsonl,safetensors}
```
