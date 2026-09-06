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
mc = float(rule.get("min_cosine", 0.98)); basis = rule.get("basis", "[U]")
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

if [ "${1:-}" = "--self-test" ]; then
    TD=$(mktemp -d "${TMPDIR:-/tmp}/parity.XXXXXX"); trap 'rm -rf "${TD:?}"' EXIT
    L="$ROOT/evidence/parity/l0-1/lambda"
    n=0; red=0
    row() { local want=$1 label=$2; shift 2; local rc=0 out; n=$((n + 1)); out=$("$@" 2>&1) || rc=$?
        if [ "$rc" = "$want" ]; then printf 'ok    row %-2s rc=%s  %s — %s\n' "$n" "$rc" "$label" "$(printf '%s' "$out" | tail -1 | cut -c1-90)"; else printf 'FAIL  row %-2s rc=%s (wanted %s)  %s\n        %s\n' "$n" "$rc" "$want" "$label" "$(printf '%s' "$out" | tail -2)"; red=1; fi; }
    row 1 "the lambda 1.5B record is RED (position 0 at 0.9508 < 0.98 [U])"   judge "$L/qwen2.5-coder-1.5b-instruct-q4_k_m.json" qwen2.5-coder-1.5b-instruct
    row 0 "the lambda 7B record is GREEN (min 0.9986)"                            judge "$L/qwen2.5-coder-7b-instruct-q4_k_m.json" qwen2.5-coder-7b-instruct
    python3 - "$L/qwen2.5-coder-7b-instruct-q4_k_m.json" "$TD/twin.json" "$TD/short.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); m = d["metrics"]
m[40]["cosine_similarity"] = 0.5; json.dump(d, open(sys.argv[2], "w"))          # must-RED twin: one position at 0.5
d2 = json.load(open(sys.argv[1])); d2["metrics"] = d2["metrics"][:20]; json.dump(d2, open(sys.argv[3], "w"))   # fewer than 64 positions
PY
    cp "$TD/twin.json" "$ROOT/tests/fixtures/parity/defective/one-position-at-0.5.json" 2>/dev/null || true
    row 1 "the must-RED twin (7B with position 40 forced to 0.5) is RED naming the position" judge "$TD/twin.json" qwen2.5-coder-7b-instruct
    row 1 "20 positions is refused (I8: >= 64), never judged"                      judge "$TD/short.json" qwen2.5-coder-7b-instruct
    printf 'schema: apr-parity-thresholds/v1\nmin_positions: 64\ndefault: {min_cosine: 0.90, basis: fixture}\nmodels: {}\n' > "$TD/thr-low.yaml"
    row 0 "under a 0.90 threshold the same 1.5B record PASSES (the threshold is the decision, item 5)" env PARITY_THRESHOLDS="$TD/thr-low.yaml" bash "$0" --judge "$L/qwen2.5-coder-1.5b-instruct-q4_k_m.json" --model qwen2.5-coder-1.5b-instruct
    printf '{}' > "$TD/empty.json"; row 1 "an output with no metrics is RED, not a pass" judge "$TD/empty.json" x
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
mkdir -p "$OUT"; rc=0; measured=0
printf '=== C14 model parity on %s (%s; thresholds %s; models %s) ===\n' "$(hostname -s)" "$("$APR_BIN" --version 2>/dev/null | head -1)" "${THRESH#"$ROOT"/}" "$MODELS_DIR"
while IFS= read -r name; do
    f=$(ls "$MODELS_DIR"/"$name"*.gguf 2>/dev/null | head -1 || true)
    if [ -z "$f" ]; then
        if grep -A20 "^- name: $name\$" "$MANIFEST" | grep -q 'README.md:'; then printf 'UNMEASURED %s: no file under %s and README.md cites it — RED\n' "$name" "$MODELS_DIR"; rc=1; else printf 'UNMEASURED %s: no file under %s (not README-cited; reported)\n' "$name" "$MODELS_DIR"; fi; continue
    fi
    j="$OUT/$name.json"; measured=$((measured + 1))
    if ! "$APR_BIN" parity "$f" --prompt "$PROMPT" --json > "$j" 2> "$j.err"; then printf 'FAIL %s: apr parity exited non-zero (%s)\n' "$name" "$(tail -1 "$j.err" | cut -c1-100)"; rc=1; continue; fi
    judge "$j" "$name" || rc=1
done < <(grep -oE '^- name: .*' "$MANIFEST" | sed 's/^- name: //')
[ "$OVERRIDE" = 0 ] || rc=1
printf 'C14: measured=%s rc=%s\n' "$measured" "$rc"; exit "$rc"
