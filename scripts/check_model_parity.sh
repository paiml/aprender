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
#   bash scripts/check_model_parity.sh --serving <perf041 witness.json> --model <name>   # judge a serving-shape receipt (#3555)
#   bash scripts/check_model_parity.sh --blessed    # the release's named model: m=1 + serving receipt per host
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
# A v2 receipt (apr-parity-receipt/v2, #3577) EMBEDS the raw `apr parity --json` document under
# `raw`, so the readings live at `raw.metrics`. A fresh `apr parity` run is the raw document itself
# and carries them at the top level. Those are two different INPUTS — a tool's output and an archived
# receipt that quotes it — not two spellings of one, so the rule is stated once: look inside the
# envelope when there is one. Anything else is still refused by the line below.
raw = d.get("raw") if isinstance(d, dict) and isinstance(d.get("raw"), dict) else d
# #3555: every verdict names the shape it was measured at. `apr parity` is single-stream by
# construction, so a record with no `shape` is {batch: 1, concurrency: 1} and says so. A record
# CLAIMING more is refused: no cosine producer runs the batched path, so the claim has no mechanism
# behind it. The serving shape is judged by --serving, on the path `apr serve` actually runs.
shape = d.get("shape") if isinstance(d, dict) else None
if shape is None: shape = {"batch": 1, "concurrency": 1}
ok = isinstance(shape, dict) and all(type(shape.get(k)) is int and shape[k] >= 1 for k in ("batch", "concurrency"))
if not ok: print(f"FAIL {model}: malformed shape {shape!r} (want positive integers batch, concurrency)"); sys.exit(1)
if (shape["batch"], shape["concurrency"]) != (1, 1):
    print(f"FAIL {model}: a cosine record claims shape batch={shape['batch']} concurrency={shape['concurrency']}, but apr parity is single-stream — judge the serving shape with --serving"); sys.exit(1)
rows = raw.get("metrics") if isinstance(raw, dict) else None
if not isinstance(rows, list) or not rows: print(f"FAIL {model}: no per-position metrics in the output"); sys.exit(1)
cos = [(r.get("position"), float(r.get("cosine_similarity"))) for r in rows if r.get("cosine_similarity") is not None]
if len(cos) < minpos: print(f"FAIL {model}: {len(cos)} positions < min_positions {minpos} (I8: an autoregressive gate validates over >= 64 positions)"); sys.exit(1)
bad = [(p, c) for p, c in cos if c < mc]
mn = min(cos, key=lambda x: x[1])
if bad:
    print(f"FAIL {model}: {len(bad)} of {len(cos)} positions below cosine {mc} (basis {basis}); min {mn[1]:.4f} at position {mn[0]}; first: " + ", ".join(f"{p}:{c:.4f}" for p, c in bad[:5])); sys.exit(1)
print(f"PASS {model}: shape batch=1 concurrency=1, {len(cos)} positions, min cosine {mn[1]:.4f} at position {mn[0]} >= {mc} (basis {basis})")
PY
}

# judge_serving <perf041 witness.json> <model name> -> "PASS|FAIL|UNMEASURED <detail>", rc 0/1 (#3555).
#
# The serving-shape receipt is the PP-26 batch-invariance witness that
# scripts/perf041_batched_parity_probe.sh writes: `apr serve` on CUDA, c concurrent
# greedy requests, every slot of an m=c batch compared token-for-token against the
# others. It is the path #2753/#2770 cover and that C14's m=1 cosine never reaches.
# Its shape is what the SERVER formed (max m_formed), not what the client asked
# for (c): two staggered requests are two m=1 batches, and calling that a serving
# shape is the vacuous pass this mode exists to refuse.
judge_serving() {
    python3 - "$1" "$2" <<'PY'
import sys, json
f, model = sys.argv[1:3]
try: w = json.load(open(f, encoding="utf-8"))
except Exception as e: print(f"FAIL {model}: unreadable serving witness ({e})"); sys.exit(1)
if not isinstance(w, dict) or w.get("probe") != "perf041" or not isinstance(w.get("bands"), list):
    print(f"FAIL {model}: not a perf041 serving witness (probe={w.get('probe') if isinstance(w, dict) else None!r})"); sys.exit(1)
path = str((w.get("model") or {}).get("path") or "")
base = path.rsplit("/", 1)[-1].lower()
if not base.startswith(model.lower()):
    print(f"FAIL {model}: the witness measured {path or '<no model>'!r}, not {model} — a receipt for another model"); sys.exit(1)
if not w.get("commit") or not (w.get("binary_sha256") or ""):
    print(f"FAIL {model}: the witness carries no commit/binary_sha256 — an unattributed run is not a receipt"); sys.exit(1)
bands = [b for b in w["bands"] if isinstance(b, dict)]
formed = [b["m_formed"] for b in bands if b.get("result") in ("PASS", "FAIL") and type(b.get("m_formed")) is int]  # measured bands only
batch = max(formed, default=0)
conc = max((b["c"] for b in bands if b.get("result") in ("PASS", "FAIL") and type(b.get("c")) is int), default=0)
shape = f"shape batch={batch} concurrency={conc}"
bad = [b for b in bands if b.get("result") not in ("PASS", "UNMEASURABLE")]
if bad:
    b = bad[0]
    print(f"FAIL {model}: {shape} — {len(bad)} band(s) diverged; first c={b.get('c')} m_formed={b.get('m_formed')} "
          f"agree_to={b.get('intra_agree_to')} < declared {b.get('declared_min')} ({b.get('result')}; {str(b.get('reason') or '')[:80]})"); sys.exit(1)
if batch < 2:
    asked = max((b["c"] for b in bands if type(b.get("c")) is int), default=0)
    print(f"UNMEASURED {model}: {shape} — no m>1 batch formed (the client asked for up to c={asked}), so the serving path was never exercised (not a pass)"); sys.exit(1)
if w.get("exit") != 0 or any(b.get("result") != "PASS" for b in bands):
    print(f"UNMEASURED {model}: {shape} — witness exit {w.get('exit')}, a band could not decide (not a pass)"); sys.exit(1)
print(f"PASS {model}: {shape}, {len(bands)} bands agree to >= {w.get('declared_min')} tokens (commit {str(w['commit'])[:9]})")
PY
}

# blessed [blessed.yaml] -> rc 0 iff every required host has BOTH receipts for the named model:
# an m=1 cosine record that PASSes `judge`, and a serving-shape witness that PASSes
# `judge_serving` (#3555). A missing receipt is RED, never skipped: the release names
# ONE model as "usable on pure CUDA" and this is the claim's evidence, per host.
# Until #2753 lands the serving row is expected RED; the file says so, the exit does not hide it.
BLESSED="${PARITY_BLESSED:-"$ROOT/evidence/parity/blessed.yaml"}"
blessed() {
    local f="${1:-$BLESSED}" rc=0 model host m1 serving
    [ -f "$f" ] || { printf 'FAIL blessed: %s missing — no model is named\n' "$f"; return 1; }
    while IFS=$'\t' read -r model host m1 serving; do
        [ "$model" = "ERR" ] && { printf 'FAIL blessed: %s\n' "$host"; return 1; }
        printf -- '--- %s @ %s\n' "$model" "$host"
        case "$m1" in /*) ;; *) m1="$ROOT/$m1" ;; esac
        case "$serving" in /*) ;; *) serving="$ROOT/$serving" ;; esac
        if [ -f "$m1" ]; then judge "$m1" "$model" || rc=1; else printf 'FAIL %s: no m=1 record %s\n' "$model" "${m1#"$ROOT"/}"; rc=1; fi
        if [ -f "$serving" ]; then judge_serving "$serving" "$model" || rc=1; else printf 'FAIL %s: no serving-shape receipt %s\n' "$model" "${serving#"$ROOT"/}"; rc=1; fi
    done < <(python3 - "$f" <<'BY'
import sys, yaml
try: b = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
except Exception as e: print(f"ERR\tunreadable {sys.argv[1]} ({e})"); sys.exit(0)
m, hosts, rec = b.get("model"), b.get("hosts") or [], b.get("receipts") or {}
if not m or not hosts: print("ERR\tblessed.yaml names no model or no hosts"); sys.exit(0)
for h in hosts:
    r = rec.get(h) or {}
    print(f"{m}\t{h}\t{r.get('m1', '-')}\t{r.get('serving', '-')}")
BY
)
    return "$rc"
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
#
# Several vendor files can share the logical name (Qwen3.5-0.8B ships as IQ4_XS,
# Q4_K_M and UD-IQ2_XXS side by side). "first in sort order" picked IQ4_XS, whose
# GGML type has no GPU GEMV kernel, so C14 reported UNMEASURED-TOOL for a model
# whose Q4_K_M twin — the ladder's own rung file — it could have measured (#3477).
# The K-quant file is preferred when present; the pick is a preference among the
# files the name matches, never a widening of the match.
prefer_measurable() {
    local k
    k=$(grep -i -m1 'Q4_K_M' || true)
    printf '%s' "$k"
}
resolve_model_file() {
    local dir="$1" name="$2" hits hit
    hits=$(ls "$dir"/"$name"*.gguf 2>/dev/null || true)
    if [ -n "$hits" ]; then
        hit=$(printf '%s\n' "$hits" | prefer_measurable); [ -n "$hit" ] || hit=$(printf '%s\n' "$hits" | head -1)
        printf '%s' "$hit"; return 0
    fi
    hits=$(find "$dir" -maxdepth 1 -iname "$name*.gguf" 2>/dev/null | sort || true)
    if [ -n "$hits" ]; then
        hit=$(printf '%s\n' "$hits" | prefer_measurable); [ -n "$hit" ] || hit=$(printf '%s\n' "$hits" | head -1)
        printf '%s' "$hit"; return 3
    fi
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
# The committed records are apr-parity-receipt/v2 (#3577): the raw `apr parity --json` document is
# embedded under `raw`. Mutate the readings where they actually live, or the twin is built from a
# KeyError and the must-RED control never fires.
def rows(doc): return doc["raw"]["metrics"] if isinstance(doc.get("raw"), dict) else doc["metrics"]
def put(doc, v):
    (doc["raw"] if isinstance(doc.get("raw"), dict) else doc)["metrics"] = v
    return doc
d = json.load(open(sys.argv[1])); rows(d)[40]["cosine_similarity"] = 0.5
json.dump(d, open(sys.argv[2], "w"))                                            # must-RED twin: one position at 0.5
d2 = json.load(open(sys.argv[1])); json.dump(put(d2, rows(d2)[:20]), open(sys.argv[3], "w"))   # fewer than 64 positions
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
    # #3477: three vendor files share the logical name; sort order put IQ4_XS first and
    # C14 reported UNMEASURED-TOOL for a model whose Q4_K_M twin it could measure.
    : > "$MD/Qwen3.5-0.8B-IQ4_XS.gguf"; : > "$MD/Qwen3.5-0.8B-UD-IQ2_XXS.gguf"
    row 3 "among IQ4_XS / Q4_K_M / UD-IQ2_XXS the K-quant file is picked, not the first in sort order" resolve_model_file "$MD" qwen3.5-0.8b
    picked=$(resolve_model_file "$MD" qwen3.5-0.8b || true)
    case "$picked" in *Q4_K_M.gguf) row 0 "the picked file IS the Q4_K_M twin" true ;; *) row 0 "the picked file IS the Q4_K_M twin (got: $picked)" false ;; esac
    # #3477: the CPU-vs-llama.cpp leg row is keyed `<name>@cpu-vs-llama.cpp` and stays
    # fail-closed (no min_cosine); the bare name is the GPU-vs-CPU row `apr parity`
    # produces, judged against the shipped thresholds file, not a fixture.
    row 2 "the CPU-vs-llama.cpp leg row stays fail-closed under its own key (I4)"       env PARITY_THRESHOLDS="$ROOT/evidence/parity/thresholds.yaml" bash "$0" --judge "$L/qwen2.5-coder-7b-instruct-q4_k_m.json" --model qwen3.5-0.8b@cpu-vs-llama.cpp
    row 0 "the shipped GPU-vs-CPU qwen3.5-0.8b row carries a measured basis and judges a good record" env PARITY_THRESHOLDS="$ROOT/evidence/parity/thresholds.yaml" bash "$0" --judge "$L/qwen2.5-coder-7b-instruct-q4_k_m.json" --model qwen3.5-0.8b

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

    # #3555: every receipt names its shape, and the serving shape is judged on the path
    # `apr serve` runs (perf041 witness), never inferred from an m=1 cosine record.
    W="$ROOT/evidence/perf041/lambda/witness.json"; C7="$L/qwen2.5-coder-7b-instruct-q4_k_m.json"
    python3 - "$W" "$C7" "$TD" <<'SV'
import json, sys, copy
w0, c7, td = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]
def put(name, doc): json.dump(doc, open(f"{td}/{name}.json", "w"))
def tw(fn):
    d = copy.deepcopy(w0); fn(d); return d
def band(d, c): return next(b for b in d["bands"] if b["c"] == c)
def fail4(d): band(d, 4).update(result="FAIL", intra_agree_to=3, reason="slot 2 diverged at token 3"); d["exit"] = 1
def stagger(d):
    for b in d["bands"]: b["m_formed"] = 1
def unmeas(d):
    for c in (4, 8, 16): band(d, c).update(result="UNMEASURABLE", reason="no batch formed")
    d["exit"] = 2
def as7b(d): d["model"]["path"] = "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
put("w-fail4", tw(fail4)); put("w-stagger", tw(stagger)); put("w-unmeas", tw(unmeas))
put("w-nocommit", tw(lambda d: d.pop("commit"))); put("w-7b", tw(as7b))
for n, sh in (("c7-b4", {"batch": 4, "concurrency": 4}), ("c7-b1", {"batch": 1, "concurrency": 1}), ("c7-bool", {"batch": True, "concurrency": 1})):
    d = copy.deepcopy(c7); d["shape"] = sh; put(n, d)
SV
    M15=qwen2.5-coder-1.5b-instruct
    # says <text> <cmd...>: rc 0 iff the verdict line CONTAINS text. A crash also exits 1, and a
    # crashing fixture reads as a killed mutant; these rows pin the verdict, not just the code.
    says() { local want=$1 out; shift; out=$("$@" 2>&1) || true; case "$out" in *"$want"*) return 0 ;; *) printf '%s\n' "$out"; return 1 ;; esac; }
    row 0 "the committed lambda perf041 witness PASSes at the serving shape batch=16"          judge_serving "$W" "$M15"
    row 1 "a witness for another model is refused (bound to its model)"                        judge_serving "$W" qwen3.5-9b
    row 1 "the must-RED twin: the c=4 band diverged at token 3 is FAIL naming the band"        judge_serving "$TD/w-fail4.json" "$M15"
    row 1 "staggered requests (every m_formed=1) never exercised batching: UNMEASURED, not PASS" judge_serving "$TD/w-stagger.json" "$M15"
    row 1 "every c>1 band UNMEASURABLE (exit 2) is not a pass"                                  judge_serving "$TD/w-unmeas.json" "$M15"
    row 1 "a witness without its commit is unattributed, not a receipt"                         judge_serving "$TD/w-nocommit.json" "$M15"
    row 0 "...and it says so, rather than crashing on the missing key"                           says "carries no commit" judge_serving "$TD/w-nocommit.json" "$M15"
    row 0 "UNMEASURABLE bands' m_formed never counts toward the reported serving shape"          says "shape batch=1 concurrency=1" judge_serving "$TD/w-unmeas.json" "$M15"
    row 1 "an m=1 cosine record handed to --serving is refused (not a perf041 witness)"         judge_serving "$C7" qwen2.5-coder-7b-instruct
    row 1 "a cosine record CLAIMING batch=4 is refused — apr parity is single-stream"           judge "$TD/c7-b4.json" qwen2.5-coder-7b-instruct
    row 0 "an explicit shape batch=1 concurrency=1 on the 7B record still PASSes"               judge "$TD/c7-b1.json" qwen2.5-coder-7b-instruct
    row 1 "a bool batch is malformed, never read as 1"                                         judge "$TD/c7-bool.json" qwen2.5-coder-7b-instruct
    printf 'model: qwen2.5-coder-7b-instruct\nhosts: [lambda]\nreceipts:\n  lambda: {m1: %s, serving: %s}\n' "$C7" "$TD/w-7b.json" > "$TD/bl-ok.yaml"
    printf 'model: qwen2.5-coder-7b-instruct\nhosts: [lambda, gx10]\nreceipts:\n  lambda: {m1: %s, serving: %s}\n' "$C7" "$TD/w-7b.json" > "$TD/bl-nogx10.yaml"
    printf 'model: qwen2.5-coder-7b-instruct\nhosts: [lambda]\nreceipts:\n  lambda: {m1: %s}\n' "$C7" > "$TD/bl-noserve.yaml"
    row 0 "blessed: m=1 PASS + serving PASS on the one declared host is GREEN"                  blessed "$TD/bl-ok.yaml"
    row 1 "blessed: a declared host with no receipts is RED, never skipped"                     blessed "$TD/bl-nogx10.yaml"
    row 1 "blessed: an m=1 record alone (no serving-shape receipt) is RED — the #3555 gap"       blessed "$TD/bl-noserve.yaml"
    row 0 "the shipped blessed.yaml names a model and both GPU hosts"                           python3 -c "import yaml,sys; b=yaml.safe_load(open('$ROOT/evidence/parity/blessed.yaml')); sys.exit(0 if b.get('model') and {'lambda','gx10'} <= set(b.get('hosts') or []) else 1)"

    printf '%s/%s rows\n' "$((n - red))" "$n"; [ "$red" = 0 ] || exit 1; exit 0
fi

MODE=""; MODELS_DIR="${APR_MODELS_DIR:-$HOME/models}"; OUT="$ROOT/evidence/parity/$(hostname -s)"; JSON=""; MODEL=""; APR_BIN=""
while [ $# -gt 0 ]; do case "$1" in --manifest) MODE=manifest; shift ;; --judge) MODE=judge; JSON=$2; shift 2 ;; --model) MODEL=$2; shift 2 ;; --models-dir) MODELS_DIR=$2; shift 2 ;; --out) OUT=$2; shift 2 ;; --apr) APR_BIN=$2; shift 2 ;; --serving) MODE=serving; JSON=$2; shift 2 ;; --blessed) MODE=blessed; shift ;; *) printf 'usage: %s --manifest [--models-dir d] [--out d] [--apr bin] | --judge <json> [--model m] | --serving <perf041 witness.json> --model m | --blessed | --self-test\n' "$PROG" >&2; exit 2 ;; esac; done
[ -f "$THRESH" ] || { printf '%s: ENV - %s missing\n' "$PROG" "$THRESH" >&2; exit 2; }
if [ "$MODE" = judge ]; then judge "$JSON" "${MODEL:-$(basename "$JSON" .json)}"; exit $?; fi
if [ "$MODE" = serving ]; then [ -n "$MODEL" ] || { printf '%s: --serving needs --model (the witness is bound to one)\n' "$PROG" >&2; exit 2; }; judge_serving "$JSON" "$MODEL"; exit $?; fi
if [ "$MODE" = blessed ]; then blessed; exit $?; fi
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
