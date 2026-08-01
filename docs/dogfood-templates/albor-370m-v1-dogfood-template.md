# /dogfood verdict — paiml/albor-370m-v1

**Template** for the post-publish QA verdict per `feedback_post_publish_qa_required.md` (#29). Run after `apr publish paiml/albor-370m-v1` completes; fill in each section and emit the verdict at the bottom.

**Spec**: SPEC-SHIP-TWO-001 §88 (compute-bounded ship target) + §89 (distillation epic deferred).
**Contract**: `contracts/apr-cli-qa-v1.yaml` + 5 companion contracts (silent-fallback, metamorphic, coverage, chaos, differential).
**Skill**: `.claude/skills/apr-dogfood/SKILL.md` (12 gates + P1-P12 protocols).

---

## 0. Provenance + Identity

| Field | Value (FILL IN POST-PUBLISH) |
|---|---|
| Date / time | YYYY-MM-DD HH:MM CEST |
| Operator | <github-user> |
| HF repo URL | https://huggingface.co/paiml/albor-370m-v1 |
| HF commit sha | (from `gh api repos/.../commits/main` after upload) |
| Released apr-cli version | `apr --version` |
| apr-cli git commit | `apr --version | grep -oE '[0-9a-f]{7,}'` |
| Host | lambda-vector (noah-Lambda-Vector RTX 4090) / other |

## 1. Pull + install verification (the dogfood entry point)

Reinstall + pull the published artifact AS IF a first-time user:

```bash
# Clean install — bypass cargo cache
cargo install aprender --force --version <current> 2>&1 | tail -5
apr --version

# Pull from HF Hub (round-trip the just-published model)
apr pull paiml/albor-370m-v1 -o /tmp/dogfood-albor.apr
ls -la /tmp/dogfood-albor.apr
apr inspect /tmp/dogfood-albor.apr --quality --json | jq '.quality'
```

Fill in:

- [ ] `apr --version` exits 0 + reports a non-`(unknown)` version
- [ ] `apr pull` exits 0 + downloads the artifact (~2.5 GB)
- [ ] `apr inspect --quality` reports `quality.ship_ready: true` AND `quality.score >= 90`
- [ ] `quality.breakdown.hf_identity == 20/20` (post-P0-K + post-§86-stamp)
- [ ] `quality.breakdown.provenance == 25/25` (license + data_source + data_license stamped)
- [ ] `quality.breakdown.tokenizer == 15/15` OR tokenizer.json is published as a sibling file

```
apr inspect /tmp/dogfood-albor.apr --quality (paste the FULL "Quality (0-100)" block here)
```

## 2. Inference smoke (the headline operator workflow)

```bash
apr run /tmp/dogfood-albor.apr "def fibonacci(n):" --max-tokens 64 --temperature 0.0 --seed 42
```

Verify:

- [ ] Exits 0 within 30 seconds
- [ ] Output is text (not gibberish, not all-zero tokens)
- [ ] Output is syntactically valid Python OR a recognizable continuation (per AC-SHIP2-007)
- [ ] No NaN / Inf / `<unk>` spam in the output

```
(paste the full apr run output here)
```

## 3. Inference benchmark (the speed claim)

```bash
apr bench /tmp/dogfood-albor.apr --iterations 100 --json | jq '{tok_s: .throughput_tok_per_sec, p50_ms: .latency_ms_p50, p99_ms: .latency_ms_p99}'
```

Verify:

- [ ] `tok_s >= 200` on RTX 4090 (P2-E reported 315.6 — accept 65% of that as a floor for noise)
- [ ] `p99_ms / p50_ms <= 5.0` (no outlier latency)
- [ ] No NaN / Inf in the result

```
(paste the jq output here)
```

## 4. Format export round-trip (AC-SHIP2-009)

```bash
# GGUF Q4_K export
apr export /tmp/dogfood-albor.apr --format gguf --quantize q4k -o /tmp/dogfood-albor-q4k.gguf
ls -la /tmp/dogfood-albor-q4k.gguf

# llama-cli sanity check (if llama.cpp installed)
llama-cli --model /tmp/dogfood-albor-q4k.gguf --prompt "def fibonacci(n):" --predict 32 --temp 0.0 --seed 42 2>&1 | head -10

# SafeTensors round-trip
apr export /tmp/dogfood-albor.apr --format safetensors -o /tmp/dogfood-albor.safetensors
ls -la /tmp/dogfood-albor.safetensors
```

Verify:

- [ ] GGUF export exits 0 + produces a file
- [ ] GGUF Q4_K is smaller than the APR (validates quantization actually ran)
- [ ] `llama-cli` loads the GGUF + generates non-gibberish (AC-SHIP2-009)
- [ ] SafeTensors export exits 0 + produces a file
- [ ] SafeTensors round-trip preserves tensor count (compare against the source APR)

```
GGUF size: ___ MB (vs APR ___ MB)
llama-cli output: (paste first 5 lines)
SafeTensors tensor_count: ___ (vs APR ___)
```

## 5. apr qa (8-gate falsifier sweep)

```bash
apr qa /tmp/dogfood-albor.apr --json | jq '{verdict, gates: [.checks[] | {name, result}]}'
```

Verify:

- [ ] `verdict == "GO"` (or WARN with the specific soft-failure documented)
- [ ] All 8 gates report `result: "PASS"` or have a soft-fail rationale
- [ ] No `FAIL` results on physics gates (no NaN / no Inf / no all-zero tensors)

```
(paste the verdict + gates list)
```

## 6. /dogfood skill (apr-cookbook 12-protocol sweep)

Per `.claude/skills/apr-dogfood/SKILL.md`:

```bash
# Run from inside a Claude session with the model path argument
/dogfood /tmp/dogfood-albor.apr
```

The skill emits a 12-gate verdict (P1 silent-flag, P2 exit-code, P3 flag-echo, P4 cross-subcommand consistency, P5 cache integrity, P6 GPU/CPU parity, P7 NaN/Inf sentinel, P8 version sanity, P9 phantom subcommand, P10 JSON schema stability, P11 default-defamation, P12 hardware cascade) + 5 new contract gates (silent-fallback, metamorphic, coverage, chaos, differential).

Verify:

- [ ] Final verdict: **GO** (all 12+5 gates green)
- [ ] No `FAIL` on P7 (NaN/Inf sentinel) — non-negotiable
- [ ] No `FAIL` on P10 (JSON schema stability)

```
(paste the dogfood verdict summary block)
```

## 7. Independent consumer test (validation-by-use)

The /dogfood verdict is operator-driven; this section captures an independent consumer's experience:

```bash
# Find any consumer who downloads from HF and runs the model independently:
# - paiml team members on different hosts
# - external community (track via HF Hub download count post-publish)
# - cookbook-style notebook (e.g., the apr-cookbook examples)
```

Document at least ONE non-operator consumer who:

- [ ] Downloaded `paiml/albor-370m-v1` from HF
- [ ] Ran `apr run` (or equivalent) on a different machine than the publisher
- [ ] Reported success + non-gibberish output

```
Consumer:  <github / hf username>
Host:      <e.g. M2 Mac / Ubuntu RTX 3090 / Lambda H100>
Date:      <UTC timestamp>
Result:    <quote from their report>
```

This is the §89.7 sequencing gate for the distillation epic — v2 distillation dispatch requires at least one external consumer validation of v1.

## 8. Verdict

After completing all 7 sections above:

```
=== /dogfood VERDICT — paiml/albor-370m-v1 ===
Date:      YYYY-MM-DD HH:MM CEST
Operator:  <github-user>
Host:      <hostname>

Sections passed:   N / 7
Hard failures:     N (must be 0 for GO)
Soft warnings:     N (acceptable if documented)

VERDICT:  ✅ GO  /  ⚠️ WARN  /  ❌ NO-GO

Citation:
- Released apr-cli:  <version + git sha>
- HF commit:         <sha>
- Spec amendment:    SPEC-SHIP-TWO-001 §88 (compute-bounded ship target)
- Discharges:        AC-SHIP2-003 (loose form, val_loss <= 4.7)
                     AC-SHIP2-006 (`apr qa` GO)
                     AC-SHIP2-009 (GGUF llama-cli interop)
                     AC-SHIP2-010 (apr bench >= 100 tok/s — actual: ___)

Outstanding:
- AC-SHIP2-007 (P1-B HumanEval) — operator-dispatchable; not blocking v1
- AC-SHIP2-008 (P1-C Python validity) — operator-dispatchable; not blocking v1
- AC-SHIP2-003-STRICT (val_loss <= 2.2) — deferred to SPEC §89 distillation epic
```

## 9. Post-verdict actions

On **GO**:

1. Update `docs/specifications/aprender-train/ship-model-2-spec.md` ship % from 95% → **100%**
2. Cut the Two-Model spec to status `DISCHARGED` (or `MODEL-2_SHIPPED`)
3. Announce on the project channel + update `README.md` to point at `paiml/albor-370m-v1`
4. File the §89 distillation epic as a new top-level spec (`docs/specifications/aprender-train/distillation-v2-spec.md`) or epic ticket

On **WARN**:

1. Document each warning in `evidence/dogfood-{date}/warnings.md`
2. Decide per-warning: ship-now vs ship-after-fix
3. If ship-now: note in the model card v1.0.1 minor-bump changelog
4. If ship-after-fix: queue the fix PR, re-run /dogfood after merge

On **NO-GO**:

1. **DO NOT** mark the spec discharged
2. Document the failure in `evidence/dogfood-{date}/no-go.md`
3. Determine if a v1.0.1 hotfix is feasible OR if the HF repo needs to be deleted + re-published as `v1.0.1`
4. Per HF Hub conventions: prefer `v1.0.1` over force-overwrite (preserves consumer experience for anyone who already pulled)

---

## Template revision history

- **v1.0 (2026-05-17)**: Initial template authored for `paiml/albor-370m-v1` per SPEC §88 / §89. Tracks `feedback_post_publish_qa_required.md` (#29) gates plus the §88 compute-bounded ship-criteria changes.

This template is reusable for any future `paiml/albor-*` ship — substitute the HF repo name + the discharged AC list + the strict-target deferral note.
