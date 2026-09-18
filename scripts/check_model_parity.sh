#!/usr/bin/env bash
# check_model_parity.sh — C14: GPU/CPU parity over the model manifest (PP-066 row L0-1,
# #2971, PMAT-1065; card item 7; wired in apr-dogfood --release, C4 post-publish, R-8 nightly).
#
# For every model in evidence/models/supported.yaml that this host holds (a .gguf whose
# basename starts with the manifest name, under $APR_MODELS_DIR, default ~/models), run
#   "$APR" parity <file> --prompt "<the 78-token corpus prompt>" --json
# and judge: positions >= min_positions (I8) AND min over positions of cosine_similarity
# >= the threshold (evidence/parity/thresholds.yaml, per-model override else default).
#   PASS      model measured, above threshold
#   FAIL      model measured, a position below threshold (named) — or fewer positions than I8 allows
#   UNMEASURED model in the manifest, no file on this host (reported; RED when README.md cites it)
#   UNMEASURED-TOOL  the model IS here and `apr parity` REFUSED its architecture (exit 12 +
#                    `parity: REFUSED architecture=...`): a limit of the TOOL, reported, never
#                    FAIL -- but RED when README.md makes a GPU=CPU/parity claim about it,
#                    because no host can produce that number in this build (PMAT-1098).
# Exit 0 iff no FAIL and no README-cited model is UNMEASURED. Never SKIP: SKIP_PARITY_GATE set
# in the environment is an override — printed, and the run's receipt is INVALID-CORRECTNESS (REG-15).
#
#   bash scripts/check_model_parity.sh --manifest [--models-dir <dir>] [--out <dir>] [--apr <bin>]
#   bash scripts/check_model_parity.sh --judge <apr-parity.json> [--model <name>]   # judge one recorded run
#   bash scripts/check_model_parity.sh --self-test
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROG=check_model_parity
MANIFEST="${PARITY_MANIFEST:-$ROOT/evidence/models/supported.yaml}"
THRESH="${PARITY_THRESHOLDS:-$ROOT/evidence/parity/thresholds.yaml}"
PROMPT='The quick brown fox jumps over the lazy dog while the committee reviewed forty-two proposals about renewable energy storage, distributed consensus, tensor layouts, quantized attention kernels, and the economics of self-hosted inference across four heterogeneous hosts; every paragraph was numbered, every table was cited, and the report ended with a checklist of sixty-four verification steps that had to pass before the release could be tagged.'

judge() { # judge <json> <model name> -> prints "PASS|FAIL <detail>", rc 0/1
    python3 - "$1" "$2" "$THRESH" <<'PY'
import sys, json, yaml
f, model, thr = sys.argv[1:4]
t = yaml.safe_load(open(thr, encoding="utf-8"))
minpos = int(t.get("min_positions", 64))
rule = (t.get("models") or {}).get(model) or t.get("default") or {}
if "min_cosine" not in rule or not str(rule.get("basis") or "").strip():
    print(f"FAIL {model}: the threshold for this model carries no min_cosine/basis in {thr} (I4: a threshold without a basis is not a threshold)"); sys.exit(2)
mc = float(rule["min_cosine"]); basis = str(rule["basis"])
try: d = json.load(open(f, encoding="utf-8"))
except Exception as e: print(f"FAIL {model}: unreadable apr parity output ({e})"); sys.exit(1)
rows = d.get("metrics") if isinstance(d, dict) else None
if not isinstance(rows, list) or not rows: print(f"FAIL {model}: no per-position metrics in the output"); sys.exit(1)
cos = [(r.get("position"), float(r.get("cosine_similarity"))) for r in rows if r.get("cosine_similarity") is not None]
if len(cos) < minpos: print(f"FAIL {model}: {len(cos)} positions < min_positions {minpos} (I8: an autoregressive gate validates over >= 64 positions)"); sys.exit(1)
bad = [(p, c) for p, c in cos if c < mc]
mn = min(cos, key=lambda x: x[1])
if bad:
    print(f"FAIL {model}: {len(bad)} of {len(cos)} positions below cosine {mc} (basis {basis}); min {mn[1]:.4f} at position {mn[0]}; first: " + ", ".join(f"{p}:{c:.4f}" for p, c in bad[:5])); sys.exit(1)
print(f"PASS {model}: {len(cos)} positions, min cosine {mn[1]:.4f} at position {mn[0]} >= {mc} (basis {basis})")
PY
}

# resolve_model_file <models-dir> <manifest-name> -> prints the path, rc:
#   0  exact-case match          (the ordinary path)
#   3  case-INSENSITIVE match     the host HOLDS the model and the exact glob missed it
#   1  no file at all             genuinely absent on this host
#
# The manifest name is a LOGICAL name derived from shipped docs (`qwen3.5-0.8b`);
# the file is whatever the vendor shipped (`Qwen3.5-0.8B-Q4_K_M.gguf`), which is
# not under our control. A case-sensitive glob reported UNMEASURED on a host that
# held the model (#3325) -- a false ABSENT, indistinguishable from "never ran",
# which is the third-state class this check exists to avoid. It survived because
# the same glob MATCHES under zsh, so it resolved by hand and failed only in the
# bash-run workflow. rc=3 is NOT tolerated silently: the caller measures the file
# and still fails, because a registry that disagrees with its artifact is the
# defect, and tolerating it is what let this sit.
resolve_model_file() {
    local dir="$1" name="$2" hit
    hit=$(ls "$dir"/"$name"*.gguf 2>/dev/null | head -1 || true)
    if [ -n "$hit" ]; then printf '%s' "$hit"; return 0; fi
    hit=$(find "$dir" -maxdepth 1 -iname "$name*.gguf" 2>/dev/null | sort | head -1 || true)
    if [ -n "$hit" ]; then printf '%s' "$hit"; return 3; fi
    return 1
}

# PMAT-1098 -------------------------------------------------------------------
# `apr parity` REFUSES an architecture its dense CPU-vs-GPU loop cannot route,
# before loading any weights: one stderr line `parity: REFUSED architecture=...`
# and exit 12 (CliError::NotImplemented; distinct from 3/5/8/9, the codes the
# command already emits). The 0.68.0 T-2 dogfood read three such rows as FAIL,
# i.e. as MODEL defects, when every one of them is a limit of the TOOL.
#
# UNMEASURED-TOOL is therefore a REPORT, like UNMEASURED -- with one difference
# that matters: UNMEASURED says "not on this host", and some other host can
# still measure it; UNMEASURED-TOOL says "no host can, in this build". So the
# moment a shipped claim depends on the number, the report becomes RED.
PARITY_REFUSED_EXIT=12
PARITY_REFUSED_MARK='parity: REFUSED architecture='

# readme_cites_parity <manifest-name> [readme] -> rc 0 iff README makes a
# GPU=CPU / parity claim ABOUT that model.
#
# Naming a model is not claiming parity for it: README.md names
# Qwen3-Coder-30B-A3B under `apr inspect`, which asserts nothing about GPU==CPU.
# The predicate is therefore proximity-based -- the model name within 5 lines of
# a parity keyword -- so a table row under a "parity" heading counts and a bare
# `apr inspect` line does not. Both polarities are in the case table.
readme_cites_parity() {
    python3 - "$1" "${2:-$ROOT/README.md}" <<'RCP'
import re, sys
name, path = sys.argv[1], sys.argv[2]
try:
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
except OSError:
    sys.exit(1)
kw = re.compile(r"parity|GPU\s*=\s*CPU|GPU/CPU|GPU vs CPU", re.I)
nm = re.compile(re.escape(name), re.I)
W = 5
for i, line in enumerate(lines):
    if not kw.search(line):
        continue
    if any(nm.search(lines[j]) for j in range(max(0, i - W), min(len(lines), i + W + 1))):
        sys.exit(0)
sys.exit(1)
RCP
}

# classify_parity_exit <name> <rc> <stderr-file> [readme] -> prints the verdict
# line; rc 0 = reported (not RED), rc 1 = RED.
#
# BOTH signals are required for UNMEASURED-TOOL. Exit 12 alone could come from
# any other NotImplemented path, and the line alone could be echoed by a crash
# that printed it before dying -- either half on its own would let a real
# failure be laundered as "the tool cannot measure this", which is the one
# outcome worse than the FAIL row this replaces.
classify_parity_exit() {
    local name=$1 prc=$2 err=$3 readme=${4:-$ROOT/README.md} line=""
    if [ "$prc" = "$PARITY_REFUSED_EXIT" ] && line=$(grep -m1 -F "$PARITY_REFUSED_MARK" "$err" 2>/dev/null); then
        printf 'UNMEASURED-TOOL %s: %s\n' "$name" "${line#*: REFUSED }"
        if readme_cites_parity "$name" "$readme"; then
            printf 'RED %s: README.md makes a GPU=CPU/parity claim about a model apr parity REFUSES to measure - the claim has no measurement behind it\n' "$name"
            return 1
        fi
        return 0
    fi
    printf 'FAIL %s: apr parity exited non-zero (%s)\n' "$name" "$(tail -1 "$err" 2>/dev/null | cut -c1-100)"
    return 1
}

if [ "${1:-}" = "--self-test" ]; then
    TD=$(mktemp -d "${TMPDIR:-/tmp}/parity.XXXXXX"); trap 'rm -rf "${TD:?}"' EXIT
    L="$ROOT/evidence/parity/l0-1/lambda"; G="$ROOT/evidence/parity/l0-1/gx10"
    printf 'schema: apr-parity-thresholds/v1\nmin_positions: 64\ndefault: {min_cosine: 0.98, basis: "fixture: the gate constant, for the case table only"}\nmodels: {}\n' > "$TD/thr-fixture.yaml"
    export PARITY_THRESHOLDS="$TD/thr-fixture.yaml"; THRESH="$TD/thr-fixture.yaml"
    n=0; red=0
    row() { local want=$1 label=$2; shift 2; local rc=0 out; n=$((n + 1)); out=$("$@" 2>&1) || rc=$?
        if [ "$rc" = "$want" ]; then printf 'ok    row %-2s rc=%s  %s — %s\n' "$n" "$rc" "$label" "$(printf '%s' "$out" | tail -1 | cut -c1-90)"; else printf 'FAIL  row %-2s rc=%s (wanted %s)  %s\n        %s\n' "$n" "$rc" "$want" "$label" "$(printf '%s' "$out" | tail -2)"; red=1; fi; }
    row 1 "the lambda 1.5B record is RED (position 0 at 0.9508 < 0.98)"       judge "$L/qwen2.5-coder-1.5b-instruct-q4_k_m.json" qwen2.5-coder-1.5b-instruct
    row 0 "the lambda 7B record is GREEN (min 0.9986)"                            judge "$L/qwen2.5-coder-7b-instruct-q4_k_m.json" qwen2.5-coder-7b-instruct
    # BOTH polarities on BOTH required hosts: one host agreeing with itself is one host (I6/#2359).
    # gx10 is a different GPU generation, ISA and host architecture, and runs the sm_121 JIT path.
    row 1 "the gx10 1.5B record is RED too (0.9506 — sm_121, aarch64, a different kernel path)" judge "$G/qwen2.5-coder-1.5b-instruct-q4_k_m.json" qwen2.5-coder-1.5b-instruct
    row 0 "the gx10 7B record is GREEN (min 0.9985)"                              judge "$G/qwen2.5-coder-7b-instruct-q4_k_m.json" qwen2.5-coder-7b-instruct
    python3 - "$L/qwen2.5-coder-7b-instruct-q4_k_m.json" "$TD/twin.json" "$TD/short.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); m = d["metrics"]
m[40]["cosine_similarity"] = 0.5; json.dump(d, open(sys.argv[2], "w"))          # must-RED twin: one position at 0.5
d2 = json.load(open(sys.argv[1])); d2["metrics"] = d2["metrics"][:20]; json.dump(d2, open(sys.argv[3], "w"))   # fewer than 64 positions
PY
    [ -f "$ROOT/tests/fixtures/parity/defective/one-position-at-0.5.json" ] || cp "$TD/twin.json" "$ROOT/tests/fixtures/parity/defective/one-position-at-0.5.json"
    row 1 "the must-RED twin (7B with position 40 forced to 0.5) is RED naming the position" judge "$TD/twin.json" qwen2.5-coder-7b-instruct
    row 1 "20 positions is refused (I8: >= 64), never judged"                      judge "$TD/short.json" qwen2.5-coder-7b-instruct
    printf 'schema: apr-parity-thresholds/v1\nmin_positions: 64\ndefault: {min_cosine: 0.90, basis: fixture}\nmodels: {}\n' > "$TD/thr-low.yaml"
    row 0 "under a 0.90 threshold the same 1.5B record PASSES (the threshold is the decision, item 5)" env PARITY_THRESHOLDS="$TD/thr-low.yaml" bash "$0" --judge "$L/qwen2.5-coder-1.5b-instruct-q4_k_m.json" --model qwen2.5-coder-1.5b-instruct
    printf '{}' > "$TD/empty.json"; row 1 "an output with no metrics is RED, not a pass" judge "$TD/empty.json" x
    printf 'schema: apr-parity-thresholds/v1\nmin_positions: 64\ndefault: {min_cosine: 0.98}\nmodels: {}\n' > "$TD/thr-nobasis.yaml"
    row 2 "a threshold without a basis is refused (exit 2), never defaulted (I4)" env PARITY_THRESHOLDS="$TD/thr-nobasis.yaml" bash "$0" --judge "$L/qwen2.5-coder-7b-instruct-q4_k_m.json" --model qwen2.5-coder-7b-instruct
    # #3325: the presence probe has THREE states and the middle one used to be
    # silent. A row written in the registry's own casing cannot see the defect,
    # so the fixture is deliberately mixed-case.
    MD="$TD/models"; mkdir -p "$MD"
    : > "$MD/qwen2-0.5b-instruct-q4_k_m.gguf"          # exact case
    : > "$MD/Qwen3.5-0.8B-Q4_K_M.gguf"                 # vendor casing, registry says qwen3.5-0.8b
    row 0 "exact-case model resolves (rc 0, the ordinary path)"                 resolve_model_file "$MD" qwen2-0.5b-instruct
    row 3 "MIXED-CASE model resolves and is flagged rc=3, never silent UNMEASURED (#3325)" resolve_model_file "$MD" qwen3.5-0.8b
    row 1 "a model genuinely absent on this host is rc=1 (UNMEASURED is correct there)"    resolve_model_file "$MD" no-such-model

    # PMAT-1098: `apr parity` REFUSES architectures its dense CPU-vs-GPU loop cannot
    # route (MoE -> #3367, Qwen3.5 -> #3090) with exit 12 and one stderr line. That is
    # a TOOL limitation; reporting it as FAIL blames the MODEL, which is what the
    # 0.68.0 T-2 dogfood did for three rows. The classifier below must separate the
    # three cases, and a genuine crash must NEVER be laundered into UNMEASURED-TOOL.
    FX="$ROOT/tests/fixtures/parity/refused"
    row 0 "a REFUSED MoE (exit 12 + the refusal line) is UNMEASURED-TOOL, not FAIL"      classify_parity_exit qwen3-coder-30b 12 "$FX/qwen3moe.err" "$FX/README-no-parity-claim.md"
    row 0 "a REFUSED qwen3.5 (no GPU forward, #3090) is UNMEASURED-TOOL too"             classify_parity_exit qwen3.5-0.8b 12 "$FX/qwen35.err" "$FX/README-no-parity-claim.md"
    row 1 "the SAME refusal is RED when the README makes a parity claim about the model" classify_parity_exit qwen3-coder-30b 12 "$FX/qwen3moe.err" "$FX/README-cites-parity.md"
    row 1 "a non-zero exit WITHOUT the refusal line is still FAIL (a crash is a crash)"  classify_parity_exit qwen3-coder-30b 8 "$FX/crash-empty-buffer.err" "$FX/README-no-parity-claim.md"
    row 1 "exit 12 WITHOUT the refusal line is FAIL — the code alone may not launder it" classify_parity_exit qwen3-coder-30b 12 "$FX/crash-empty-buffer.err" "$FX/README-no-parity-claim.md"
    row 1 "the refusal line WITHOUT exit 12 is FAIL — the line alone may not launder it" classify_parity_exit qwen3-coder-30b 8 "$FX/qwen3moe.err" "$FX/README-no-parity-claim.md"
    # The predicate that decides today's C14 verdict on the three refused models. If a
    # future README starts claiming GPU=CPU for one of them, this row goes RED here
    # BEFORE the release reads a green C14 that measured nothing.
    row 0 "the fixture README's parity claim about qwen3-coder-30b is SEEN (must-RED twin)" readme_cites_parity qwen3-coder-30b "$FX/README-cites-parity.md"
    row 1 "README.md makes no parity claim about qwen3-coder-30b today (so the refusal is not RED)" readme_cites_parity qwen3-coder-30b "$ROOT/README.md"
    row 1 "README.md makes no parity claim about qwen3.5-0.8b today"                       readme_cites_parity qwen3.5-0.8b "$ROOT/README.md"
    row 1 "README.md makes no parity claim about qwen3-30b today"                          readme_cites_parity qwen3-30b "$ROOT/README.md"

    printf '%s/%s rows\n' "$((n - red))" "$n"; [ "$red" = 0 ] || exit 1; exit 0
fi

MODE=""; MODELS_DIR="${APR_MODELS_DIR:-$HOME/models}"; OUT="$ROOT/evidence/parity/$(hostname -s)"; JSON=""; MODEL=""; APR_BIN=""
while [ $# -gt 0 ]; do case "$1" in --manifest) MODE=manifest; shift ;; --judge) MODE=judge; JSON=$2; shift 2 ;; --model) MODEL=$2; shift 2 ;; --models-dir) MODELS_DIR=$2; shift 2 ;; --out) OUT=$2; shift 2 ;; --apr) APR_BIN=$2; shift 2 ;; *) printf 'usage: %s --manifest [--models-dir d] [--out d] [--apr bin] | --judge <json> [--model m] | --self-test\n' "$PROG" >&2; exit 2 ;; esac; done
[ -f "$THRESH" ] || { printf '%s: ENV - %s missing\n' "$PROG" "$THRESH" >&2; exit 2; }
if [ "$MODE" = judge ]; then judge "$JSON" "${MODEL:-$(basename "$JSON" .json)}"; exit $?; fi
[ "$MODE" = manifest ] || { printf 'usage: %s --manifest | --judge <json> | --self-test\n' "$PROG" >&2; exit 2; }
[ -f "$MANIFEST" ] || { printf '%s: ENV - %s missing (scripts/derive_model_manifest.sh)\n' "$PROG" "$MANIFEST" >&2; exit 2; }
if [ -n "${SKIP_PARITY_GATE:-}" ]; then printf 'override: SKIP_PARITY_GATE=%s is set — every receipt of this run is INVALID-CORRECTNESS (REG-15); C14 refuses to pass under an override\n' "$SKIP_PARITY_GATE"; OVERRIDE=1; else OVERRIDE=0; fi
if [ -z "$APR_BIN" ]; then . "$ROOT/scripts/apr_bin.sh" >/dev/null 2>&1 || { printf '%s: ENV - scripts/apr_bin.sh could not pin an apr built from HEAD; pass --apr\n' "$PROG" >&2; exit 2; }; APR_BIN="$APR"; fi
mkdir -p "$OUT"; rc=0; measured=0; seen=""
printf '=== C14 model parity on %s (%s; thresholds %s; models %s) ===\n' "$(hostname -s)" "$("$APR_BIN" --version 2>/dev/null | head -1)" "${THRESH#"$ROOT"/}" "$MODELS_DIR"
# names are iterated LONGEST FIRST so an alias (qwen2.5-coder-1.5b) that prefix-globs to the file a
# more specific name (…-1.5b-instruct) already measured is recorded as that measurement, not run twice
while IFS= read -r name; do
    f=$(resolve_model_file "$MODELS_DIR" "$name") || rmrc=$?
    rmrc=${rmrc:-0}
    if [ "$rmrc" = 3 ]; then
        printf 'NAME-MISMATCH %s: the manifest name does not match %s on disk — measuring it, but the registry and the artifact must agree (#3325)\n' \
            "$name" "$(basename "$f")"
        rc=1
    fi
    unset rmrc
    if [ -z "$f" ]; then
        # UNMEASURED is a per-host REPORT, never a per-host RED: no single host holds every model the
        # README names; the fleet-level rule (every README-cited model measured on >= 1 GPU host) is
        # the release's (R-5 promotion: C14 PASS on four receipts, parity != skipped)
        printf 'UNMEASURED %s: no file under %s (reported; the fleet-level rule belongs to the release)\n' "$name" "$MODELS_DIR"; continue
    fi
    case " $seen " in *" $f "*) printf 'ALIAS %s -> %s (already measured under a longer name)\n' "$name" "$(basename "$f")"; continue ;; esac
    seen="$seen $f"
    j="$OUT/$name.json"; measured=$((measured + 1))
    prc=0; "$APR_BIN" parity "$f" --prompt "$PROMPT" --json > "$j" 2> "$j.err" || prc=$?
    if [ "$prc" != 0 ]; then
        # A refusal is NOT a measurement: it must not hold the vacuity floor
        # ("nothing measured is not a pass") green on a host where every model
        # was refused.
        classify_parity_exit "$name" "$prc" "$j.err" || rc=1
        case "$prc" in "$PARITY_REFUSED_EXIT") measured=$((measured - 1)) ;; *) : ;; esac
        continue
    fi
    judge "$j" "$name" || rc=1
done < <(grep -oE '^- name: .*' "$MANIFEST" | sed 's/^- name: //' | awk '{ print length($0) "\t" $0 }' | sort -rn | cut -f2-)
[ "$OVERRIDE" = 0 ] || rc=1
if [ "$measured" -eq 0 ]; then printf 'C14: nothing measured on %s — not a pass (the models dir holds no manifest model)\n' "$(hostname -s)"; rc=1; fi
printf 'C14: measured=%s rc=%s\n' "$measured" "$rc"; exit "$rc"
