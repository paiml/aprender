#!/usr/bin/env bash
# check_crux_plugin_batch.sh: the case table for batching the source-weight engines (#4036 lever 2,
# plugin_batch_cells in scripts/crux_inference_dogfood.sh). Hermetic: stub apr, hf and llamafile drivers inside a
# COPY of scripts/, a private lock file for /tmp/apr-gpu.lock. No GPU, no model.
#
# The stub hf driver logs one LOAD per invocation, i.e. per engine load, and writes one row per item.
# Rows (the REAL dogfood and the REAL judge, --verbs run,chat on the positive control):
#   1. batched   → hf loaded ONCE for the mode's 2 in-process items; the receipt is GREEN
#   2. unbatched (CRUX_NO_BATCH=1) → hf loaded once PER ITEM (2); the receipt is identical cell for cell to row 1,
#                  and hf's rows are the same row for row (every field but the per-load `batch` id and the paths)
#   3. a driver without gen-batch → the per-prompt path, one load per item: batching is a capability, never assumed
#   4. MUST-RED: the batch returns no row for one item → that item is REFUSED by name, the cell RED, never absent
#  4b. MUST-RED: the batch returns TWO rows for one item (one wrong) → refused by name, the cell RED, never a pick
#   5. MUTANT: crux_batched always false → row 1 sees 2 loads, not 1: the table catches a batch that never happens
#
# Exit: 0 every row behaved · 1 a row broke · 2 ENV.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd) || exit 2
PROG=check_crux_plugin_batch
for t in python3 flock; do
  command -v "$t" >/dev/null 2>&1 || { printf '%s: ENV - %s is missing\n' "$PROG" "$t" >&2; exit 2; }
done
for f in scripts/crux_inference_dogfood.sh scripts/lib/crux_inference_judge.py scripts/crux_inference_prompts.v2.json; do
  [ -f "$ROOT/$f" ] || { printf '%s: ENV - %s not found\n' "$PROG" "$f" >&2; exit 2; }
done

TMP=$(mktemp -d) || exit 2
_rm_tmp() {
  local w
  [ -n "${KEEP_TMP:-}" ] && { echo "kept $TMP"; return; }
  for w in $(sed -n 's/^work kept: //p' "$TMP"/*.log 2>/dev/null); do
    [ -n "$w" ] && [ "$w" != / ] || continue
    case "${w:-}" in
      /tmp/?*) rm -rf -- "${w:?}" || : ;;
      *) : ;;
    esac
  done
  case "${TMP:-}" in
    /tmp/?*|/var/folders/?*) rm -rf -- "$TMP" || : ;;
    *) : ;;
  esac
}
trap _rm_tmp EXIT
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
mkdir -p "$TMP/shim"
printf '#!/bin/sh\nwhile [ $# -gt 0 ] && [ "$1" != -- ]; do shift; done\n[ $# -gt 0 ] && shift\nexec "$@"\n' > "$TMP/shim/choom"
chmod +x "$TMP/shim/choom"
export PATH="$TMP/shim:$PATH"

PASS=0
FAIL=0
ok()   { printf '  ok    %s\n' "$1"; PASS=$((PASS + 1)); }
broke(){ printf '  BROKE %s\n' "$1"; FAIL=$((FAIL + 1)); }

# The stub driver, shared by hf (batch-capable unless STUB_NO_BATCH=1) and llamafile (never batch-capable).
cat > "$TMP/stub_driver.py" <<'PY'
import json, os, sys
eng, argv = sys.argv[1], sys.argv[2:]
cmd = argv[0] if argv else ""
batchable = eng == "hf" and not os.environ.get("STUB_NO_BATCH")
if cmd == "probe":
    print("%s=1.0.0 transformers=5.0.0 torch=2.0.0 device=stub" % eng); sys.exit(0)
if cmd == "gen-batch" and "--help" in argv:
    sys.exit(0 if batchable else 2)
if cmd not in ("gen", "gen-batch") or (cmd == "gen-batch" and not batchable):
    sys.exit(2)
opt = {argv[i][2:]: argv[i + 1] for i in range(1, len(argv) - 1) if argv[i].startswith("--")}
if cmd == "gen":
    items = [{"prompt_id": opt["prompt-id"], "verb": opt["verb"], "thinking": opt["thinking"]}]
else:
    items = [json.loads(l) for l in open(opt["batch"]) if l.strip()]
open(os.environ["STUB_CALLS"], "a").write("LOAD %s %d\n" % (eng, len(items)))
sha = opt["model-sha256"]
for it in items:
    if it["verb"] == os.environ.get("STUB_DROP", "-") and eng == "hf":
        continue
    d = os.path.join(os.environ["CRUX_WORK"], sha[:12], it["verb"])
    os.makedirs(d, exist_ok=True)
    stem = "%s-%s-%s" % (eng, it["prompt_id"], it["thinking"])
    out = os.path.join(d, stem + ".json")
    json.dump({"text": "<answer>4</answer>", "raw_text": "<answer>4</answer>\n",
               "reported": {"thinking_requested": it["thinking"], "device": "stub"}}, open(out, "w"))
    open(os.path.join(d, stem + ".err"), "w").close()
    row = {"kind": "gen", "engine": eng, "model_sha256": sha, "host": opt["host"], "verb": it["verb"],
           "thinking": it["thinking"], "backend": opt["backend"], "prompt_id": it["prompt_id"], "rc": 0,
           "stdout": out, "stderr": os.path.join(d, stem + ".err"), "refused": None}
    if eng == "hf":
        row["source"] = {"repo": "stub/src", "revision": "0" * 40, "dtype": "bfloat16"}
        row["batch"] = {"id": str(os.getpid()), "size": len(items)}
    open(os.environ["CRUX_MANIFEST"], "a").write(json.dumps(row) + "\n")
    if it["verb"] == os.environ.get("STUB_DUP", "-") and eng == "hf":
        wrong = os.path.join(d, stem + ".dup.json")
        json.dump({"text": "<answer>5</answer>", "reported": {"device": "stub"}}, open(wrong, "w"))
        open(os.environ["CRUX_MANIFEST"], "a").write(json.dumps(dict(row, stdout=wrong)) + "\n")
PY

mk_tree() { # mk_tree <dir>: the real scripts with the two drivers stubbed
  local t="$1" eng
  mkdir -p "$t"
  cp -r "$ROOT/scripts" "$t/scripts"
  for eng in hf llamafile; do
    printf '#!/usr/bin/env bash\nexec python3 %q %s "$@"\n' "$TMP/stub_driver.py" "$eng" > "$t/scripts/crux_engine_$eng.sh"
    chmod +x "$t/scripts/crux_engine_$eng.sh"
  done
}

BIN="$TMP/bin"; mkdir -p "$BIN"
cat > "$BIN/apr" <<'SH'
#!/usr/bin/env bash
case "${1:-}" in
  --version) echo "apr 0.0.0 (stub)" ;;
  run)
    case " $* " in *" --help "*) echo "  --thinking <MODE>"; exit 0 ;; esac
    printf '{"text": "<answer>4</answer>", "tokens_generated": 1, "tok_per_sec": 1.0, "finish_reason": "stop", "backend": {"requested": "gpu", "ran": "gpu", "fell_back": false}}\n' ;;
  chat)
    while IFS= read -r _turn; do printf 'You: \nAssistant: <answer>4</answer>\n'; done
    printf 'You: \nGoodbye!\n' ;;
  *) echo "stub apr: unhandled '$*'" >&2; exit 1 ;;
esac
SH
chmod +x "$BIN/apr"
MODEL="$TMP/model.gguf"; printf 'GGUF-stub-model' > "$MODEL"
MSHA=$(sha256sum "$MODEL" | cut -d' ' -f1)
printf 'sources:\n  %s:\n    hf: {repo: stub/src, revision: "%s", dtype: bfloat16}\n' "$MSHA" "$(printf '0%.0s' $(seq 40))" > "$TMP/hf-sources.yaml"
python3 - "$ROOT/scripts/crux_inference_prompts.v2.json" "$MSHA" "$TMP/cert.json" <<'PY' || exit 2
import hashlib, json, sys
prompts, sha, out = sys.argv[1:4]
json.dump({"schema": "crux-prompt-certification/v1", "prompts": "scripts/crux_inference_prompts.v2.json",
           "prompts_sha256": hashlib.sha256(open(prompts, "rb").read()).hexdigest(), "admitted": {},
           "admitted_by_sha": {sha: ["ctl-2plus2"]}, "admitted_by_sha_thinking": {sha: {"off": ["ctl-2plus2"]}}},
          open(out, "w"))
PY

run_row() { # run_row <name> <tree> [env...]
  local name="$1" tree="$2"; shift 2
  : > "$TMP/$name.calls"; : > "$TMP/$name.lock"
  ( cd "$tree" && env STUB_CALLS="$TMP/$name.calls" CRUX_GPU_LOCK="$TMP/$name.lock" GPUQ_BIN=/nonexistent/gpu-q \
      CRUX_HF_SOURCES="$TMP/hf-sources.yaml" DOGFOOD_ALLOW_UNPINNED=1 APR="$BIN/apr" "$@" \
      timeout 300 bash scripts/crux_inference_dogfood.sh 0.0.0 --model "$MODEL" --engines apr,hf,llamafile \
        --verbs run,chat --prompts scripts/crux_inference_prompts.v2.json --certification "$TMP/cert.json" \
        --only-prompts ctl-2plus2 --thinking-modes off \
        --host stub --out "$TMP/$name.out" --timeout 60 --keep-work ) > "$TMP/$name.log" 2>&1
  echo "$?" > "$TMP/$name.rc"
}

# summary <name>: "<hf loads> | <cell verdicts>"
summary() {
  python3 - "$TMP/$1.out/stub-gpu.json" "$TMP/$1.calls" <<'PY'
import json, sys
loads = sum(1 for l in open(sys.argv[2]) if l.startswith("LOAD hf "))
try:
    d = json.load(open(sys.argv[1]))
except (OSError, ValueError):
    print("%d | NO-RECEIPT" % loads); sys.exit()
print("%d | %s" % (loads, " ".join(sorted("%s/%s=%s" % (c["key"]["verb"], c["key"]["prompt_id"], c["verdict"])
                                          for c in d.get("cells", []))) or "no-cells"))
PY
}

hf_rows() { # hf_rows <name>: hf's rows, with the per-load batch id and the work-dir paths removed
  python3 - "$(sed -n 's/^work kept: //p' "$TMP/$1.log" | tail -1)" <<'PY'
import json, os, sys
w = sys.argv[1]
rows = []
for l in open(os.path.join(w, "manifest.jsonl")):
    r = json.loads(l)
    if r.get("kind") == "gen" and r.get("engine") == "hf":
        r.pop("batch", None)
        for f in ("stdout", "stderr"):
            if r.get(f):
                r[f] = os.path.relpath(r[f], w)
        rows.append(json.dumps(r, sort_keys=True))
print("\n".join(sorted(rows)))
PY
}

printf '%s: the source-weight engines, one load per batch (#4036)\n' "$PROG"
T="$TMP/tree"; mk_tree "$T"

run_row batched "$T"
b=$(summary batched)
case "$b" in
  "1 | "*=GREEN*) printf '%s' "$b" | grep -q '=RED' && broke "batched: a RED cell ($b)" \
                  || ok "batched: hf loaded ONCE for the mode's in-process items, receipt GREEN ($b)" ;;
  *) broke "batched: $b (rc $(cat "$TMP/batched.rc")); log $TMP/batched.log"; tail -4 "$TMP/batched.log" | sed 's/^/        /' ;;
esac

run_row unbatched "$T" CRUX_NO_BATCH=1
u=$(summary unbatched)
if [ "${u%% |*}" = 2 ] && [ "${u#* | }" = "${b#* | }" ] && [ "$(hf_rows batched)" = "$(hf_rows unbatched)" ] \
   && [ -n "$(hf_rows batched)" ]; then
  ok "CRUX_NO_BATCH: hf loaded per item (2), receipt identical cell for cell, hf rows identical row for row ($u)"
else
  broke "unbatched: $u vs batched $b; rows equal: $([ "$(hf_rows batched)" = "$(hf_rows unbatched)" ] && echo yes || echo no)"
fi

run_row nocap "$T" STUB_NO_BATCH=1
n=$(summary nocap)
[ "${n%% |*}" = 2 ] && [ "${n#* | }" = "${b#* | }" ] \
  && ok "a driver without gen-batch keeps the per-prompt path: 2 loads, same receipt ($n)" \
  || broke "no gen-batch capability: $n"

run_row drop "$T" STUB_DROP=chat
dr=$(summary drop)
if printf '%s' "$dr" | grep -q 'chat/ctl-2plus2=RED' \
   && grep -q 'gen-batch exited 0 without a row for this item' "$(sed -n 's/^work kept: //p' "$TMP/drop.log" | tail -1)/manifest.jsonl"; then
  ok "MUST-RED: an item the batch returned no row for is refused by name, its cell RED ($dr)"
else
  broke "dropped item: $dr"
fi

# Row 5b: MUST-RED — the batch returns TWO rows for one item (one right, one wrong). The count guard only saw a
# MISSING row, so this passed, the judge kept whichever row came last, and the cache replayed both (quorum round 7,
# lane 2, measured). Now the item is refused by name, the refusal is the row the judge keeps, and the cell is RED.
run_row dup "$T" STUB_DUP=chat
du=$(summary dup)
if printf '%s' "$du" | grep -q 'chat/ctl-2plus2=RED' \
   && grep -q 'rows for ONE item is refused' "$(sed -n 's/^work kept: //p' "$TMP/dup.log" | tail -1)/manifest.jsonl"; then
  ok "MUST-RED: an item the batch answered TWICE is refused by name, its cell RED ($du)"
else
  broke "duplicate item: $du"
fi

MT="$TMP/mutant"; mk_tree "$MT"
python3 - "$MT/scripts/crux_inference_dogfood.sh" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
a = 'crux_batched() { # crux_batched <engine>: its rows this mode come from plugin_batch_cells, not the per-prompt cells\n'
assert s.count(a) == 1, "mutation anchor moved: update this check with the dogfood"
open(p, "w").write(s.replace(a, a + "  return 1\n"))
PY
run_row mutant "$MT"
m=$(summary mutant)
[ "${m%% |*}" != 1 ] && ok "MUTANT (crux_batched never true) is caught: $m, not 1 load" \
  || broke "MUTANT not caught: $m"

printf '%s: %d ok, %d broke\n' "$PROG" "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ] || exit 1
exit 0
