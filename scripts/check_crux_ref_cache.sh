#!/usr/bin/env bash
# check_crux_ref_cache.sh: the case table for the CRUX reference cache (#4036, scripts/lib/crux_ref_cache.py and
# its hook in scripts/crux_inference_dogfood.sh). Hermetic: stub apr, hf and llamafile drivers stand in for the
# engines inside a COPY of scripts/, and a private lock file stands in for /tmp/apr-gpu.lock. No GPU, no model.
#
# The rows run the REAL dogfood and the REAL judge, and read the receipt, never the cache's own log:
#   1. cold run (empty cache)   → every reference engine ran, their clean rows were stored, the receipt is GREEN
#   2. warm run (same cache)    → NO reference engine ran, the receipt is the same cell for cell (key, verdict),
#                                 and every reference row says it came from the cache and names its origin
#   3. MUST-RED: one stored artifact edited, its answer still RIGHT → NO reference engine ran, every cell is RED,
#                                 and the receipt's refusal says STALE. Any edit is RED, not only a wrong answer
#                                 (a wrong one the judge would catch anyway, which is why this edit keeps it right).
#   4. oracle key: another host's device on the probe is a HIT (the host is not in the key); a new package
#                                 version is a MISS, recomputed GREEN, never matched against the old truth
#   5. a refused reference row is never stored → the next run is a MISS for that engine, not a hit
#   6. MUTANT: the stale check disabled (why_stale always None) → row 3's receipt is no longer RED: the table
#                                 catches it
#   7. llama.cpp is refused by name: it is apr's reference renderer on the serve routes, so it always runs
#   8. the global cap moved (a bigger budget elsewhere in the prompt set) → a MISS for run/chat, recomputed
#   9. MUST-RED: a row's artifact pointer rewritten to a `../` traversal (key and files{} intact) → STALE, RED
#  10. the lookup's exit contract: hit 0 · miss 10 · stale 11; a keying refusal (1) and a crash (2) are neither
#  11. MUST-DECLINE: a lookup that crashes inside the real dogfood declines the run, never a silent recompute
#  12. an edited pointer that resolves to the SAME hashed file (digest re-sealed): STALE on the real lib, REUSED by a
#      mutant without the row<->file rules (pointer + converse) — they, not a sha256, digest or crash, refuse it
#  13. a field REMOVED from a stored row (stdout; backend, which no per-field rule reads): STALE by the entry's
#      content digest; a mutant without the digest REUSES the backend edit
#  14. an unreferenced file added to files{} (hashed, digest re-sealed) is STALE; a mutant without the converse
#      rule REUSES it
#  15. MUST-RED: an origin-only edit, not re-sealed, is STALE (the seal covers the whole entry)
#  16. a driver answering one item TWICE: the cell is RED, the entry is never stored, the next run is a MISS
#  NOT covered, by design: a coherent re-seal by a cache writer (see the lib's THREAT MODEL)
#
# Exit: 0 every row behaved · 1 a row broke · 2 ENV.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd) || exit 2
PROG=check_crux_ref_cache
for t in python3 flock; do
  command -v "$t" >/dev/null 2>&1 || { printf '%s: ENV - %s is missing\n' "$PROG" "$t" >&2; exit 2; }
done
for f in scripts/crux_inference_dogfood.sh scripts/lib/crux_ref_cache.py scripts/lib/crux_inference_judge.py \
         scripts/crux_inference_prompts.v2.json scripts/lib/crux_prompt_certify.py; do
  [ -f "$ROOT/$f" ] || { printf '%s: ENV - %s not found\n' "$PROG" "$f" >&2; exit 2; }
done

TMP=$(mktemp -d) || exit 2
_rm_tmp() {
  local w
  [ -n "${KEEP_TMP:-}" ] && { echo "kept $TMP"; return; }
  # the dogfood's own work dirs, kept by --keep-work so row 2 can read the manifest
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
# run_cell's OOM-victim marking is shimmed: this table measures the cache, and an unprivileged sandbox denies choom
mkdir -p "$TMP/shim"
printf '#!/bin/sh\nwhile [ $# -gt 0 ] && [ "$1" != -- ]; do shift; done\n[ $# -gt 0 ] && shift\nexec "$@"\n' > "$TMP/shim/choom"
chmod +x "$TMP/shim/choom"
export PATH="$TMP/shim:$PATH"

PASS=0
FAIL=0
ok()   { printf '  ok    %s\n' "$1"; PASS=$((PASS + 1)); }
broke(){ printf '  BROKE %s\n' "$1"; FAIL=$((FAIL + 1)); }

# ---- the tree: a copy of scripts/ with the two reference drivers stubbed ------------------------------------
# mk_tree <dir>: the real scripts, stub drivers. Every stub call appends "CALL <engine> <verb> <prompt>" to $STUB_CALLS.
mk_tree() {
  local t="$1" eng
  mkdir -p "$t"
  cp -r "$ROOT/scripts" "$t/scripts"
  for eng in hf llamafile; do
    cat > "$t/scripts/crux_engine_$eng.sh" <<SH
#!/usr/bin/env bash
# stub $eng driver for check_crux_ref_cache.sh: STUB_PROBE_$eng overrides the probe line; STUB_REFUSE_$eng=1
# makes every gen row a refusal.
eng=$eng
SH
    cat >> "$t/scripts/crux_engine_$eng.sh" <<'SH'
case "${1:-}" in
  probe) pv="STUB_PROBE_$eng"; printf '%s\n' "${!pv:-$eng=1.0.0 transformers=5.0.0 torch=2.0.0 device=stub-$HOSTNAME}"; exit 0 ;;
  gen) shift ;;
  *) exit 2 ;;
esac
while [ $# -gt 0 ]; do
  case "$1" in
    --model-sha256) sha="$2"; shift 2 ;;
    --verb) verb="$2"; shift 2 ;;
    --prompt-id) pid="$2"; shift 2 ;;
    --thinking) think="$2"; shift 2 ;;
    --backend) backend="$2"; shift 2 ;;
    --host) host="$2"; shift 2 ;;
    --interface) shift 2 ;;
    *) if [ "${2:-}" != "" ] && [ "${2#--}" = "$2" ]; then shift 2; else shift; fi ;;
  esac
done
echo "CALL $eng $verb $pid" >> "$STUB_CALLS"
d="$CRUX_WORK/${sha:0:12}/$verb"; mkdir -p "$d"
out="$d/$eng-$pid-$think.json"
printf '{"text": "<answer>4</answer>", "raw_text": "<answer>4</answer>\\n", "reported": {"thinking_requested": "%s", "device": "stub"}}\n' "$think" > "$out"
: > "$d/$eng-$pid-$think.err"
rv="STUB_REFUSE_$eng"
python3 - "$CRUX_MANIFEST" "$eng" "$sha" "$host" "$verb" "$think" "$backend" "$pid" "$out" "$d/$eng-$pid-$think.err" "${!rv:-}" <<'PY'
import json, os, sys
m, eng, sha, host, verb, think, backend, pid, out, err, refuse = sys.argv[1:12]
row = {"kind": "gen", "engine": eng, "model_sha256": sha, "host": host, "verb": verb, "thinking": think,
       "backend": backend, "prompt_id": pid, "rc": 0, "stdout": out, "stderr": err,
       "refused": "stub refusal" if refuse else None}
if eng == "hf":
    row["source"] = {"repo": "stub/src", "revision": "0" * 40, "dtype": "bfloat16"}
open(m, "a").write(json.dumps(row) + "\n")
if os.environ.get("STUB_DUP_" + eng):
    open(m, "a").write(json.dumps(row) + "\n")
PY
SH
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
    printf '{"text": "<answer>4</answer>", "tokens_generated": 1, "tok_per_sec": 1.0, "finish_reason": "stop", "backend": {"requested": "gpu", "ran": "gpu", "fell_back": false}}\n'
    printf '[DEBUG] formatted_prompt="q"\n' >&2 ;;
  *) echo "stub apr: unhandled '$*'" >&2; exit 1 ;;
esac
SH
chmod +x "$BIN/apr"
MODEL="$TMP/model.gguf"; printf 'GGUF-stub-model' > "$MODEL"
MSHA=$(sha256sum "$MODEL" | cut -d' ' -f1)
printf 'sources:\n  %s:\n    hf: {repo: stub/src, revision: "%s", dtype: bfloat16}\n' "$MSHA" "$(printf '0%.0s' $(seq 40))" > "$TMP/hf-sources.yaml"
# A fixture certification over the REAL v2 prompt bytes that admits the positive control for the stub model, OFF.
# The judge checks it through the certifier's own `check`, so a drift in the prompt set is an ENV refusal here.
python3 - "$ROOT/scripts/crux_inference_prompts.v2.json" "$MSHA" "$TMP/cert.json" <<'PY' || exit 2
import hashlib, json, sys
prompts, sha, out = sys.argv[1:4]
json.dump({"schema": "crux-prompt-certification/v1", "prompts": "scripts/crux_inference_prompts.v2.json",
           "prompts_sha256": hashlib.sha256(open(prompts, "rb").read()).hexdigest(), "admitted": {},
           "admitted_by_sha": {sha: ["ctl-2plus2"]}, "admitted_by_sha_thinking": {sha: {"off": ["ctl-2plus2"]}}},
          open(out, "w"))
PY

# run_row <name> <tree> <cache> [env...]: one real dogfood run; receipt at $TMP/<name>.out/stub-gpu.json
run_row() {
  local name="$1" tree="$2" cache="$3"; shift 3
  : > "$TMP/$name.calls"; : > "$TMP/$name.lock"
  ( cd "$tree" && env STUB_CALLS="$TMP/$name.calls" CRUX_GPU_LOCK="$TMP/$name.lock" GPUQ_BIN=/nonexistent/gpu-q \
      CRUX_HF_SOURCES="$TMP/hf-sources.yaml" DOGFOOD_ALLOW_UNPINNED=1 APR="$BIN/apr" "$@" \
      timeout 300 bash scripts/crux_inference_dogfood.sh 0.0.0 --model "$MODEL" --engines apr,hf,llamafile \
        --verbs run --prompts scripts/crux_inference_prompts.v2.json --certification "${RUN_CERT:-$TMP/cert.json}" \
        --only-prompts ctl-2plus2 --thinking-modes off \
        --host stub --out "$TMP/$name.out" --timeout 60 --reference-cache "$cache" --keep-work ) > "$TMP/$name.log" 2>&1
  echo "$?" > "$TMP/$name.rc"
}

# summary <name>: "<engine calls> | <cell verdicts> | <reference rows from cache>/<reference rows>"
summary() {
  python3 - "$TMP/$1.out/stub-gpu.json" "$TMP/$1.calls" "$TMP/$1.out" <<'PY'
import glob, json, os, sys
rec, calls = sys.argv[1], sys.argv[2]
n = sum(1 for _ in open(calls))
try:
    d = json.load(open(rec))
except (OSError, ValueError):
    print("%d | NO-RECEIPT | -" % n); sys.exit()
cells = sorted("%s/%s/%s=%s" % (c["key"]["verb"], c["key"]["thinking"], c["key"]["prompt_id"], c["verdict"])
               for c in d.get("cells", []))
print("%d | %s | %s" % (n, " ".join(cells) or "no-cells", d.get("declined") or d.get("declined_because") or ""))
PY
}

printf '%s: the CRUX reference cache (#4036)\n' "$PROG"
T="$TMP/tree"; mk_tree "$T"
CACHE="$TMP/cache"

run_row cold "$T" "$CACHE"
cold=$(summary cold)
stored=$(find "$CACHE" -name entry.json 2>/dev/null | wc -l)
case "$cold" in
  "2 | "*"=GREEN"*) [ "$stored" -eq 2 ] && ok "cold: both reference engines ran, 2 entries stored, receipt GREEN ($cold)" \
                    || broke "cold: $stored entries stored, want 2 ($cold)" ;;
  *) broke "cold: $cold (rc $(cat "$TMP/cold.rc")); log $TMP/cold.log"; tail -5 "$TMP/cold.log" | sed 's/^/        /' ;;
esac

run_row warm "$T" "$CACHE"
warm=$(summary warm)
# every reference row of the warm manifest: from the cache, host rewritten to this run, origin named, artifact local
prov=$(python3 - "$(sed -n 's/^work kept: //p' "$TMP/warm.log" | tail -1)" <<'PY'
import json, os, sys
bad, n = [], 0
for l in open(os.path.join(sys.argv[1], "manifest.jsonl")):
    r = json.loads(l)
    if r.get("kind") != "gen" or r.get("engine") == "apr":
        continue
    n += 1
    rc = r.get("reference_cache") or {}
    if not (rc.get("digest") and (rc.get("origin") or {}).get("host") == "stub" and r.get("host") == "stub"
            and r["stdout"].startswith(sys.argv[1] + "/refcache/") and os.path.isfile(r["stdout"])):
        bad.append(r["engine"])
print("%d rows, not provenanced: %s" % (n, ",".join(bad) or "none"))
PY
)
if [ "${warm%% |*}" = 0 ] && [ "${warm#* | }" = "${cold#* | }" ] && [ "$prov" = "2 rows, not provenanced: none" ] \
   && grep -q '"result": "hit"' "$TMP/warm.out/stub-gpu.json"; then
  ok "warm: 0 reference engine calls, receipt identical cell for cell, $prov, meta records the hit ($warm)"
else
  broke "warm: $warm vs cold $cold; $prov (rc $(cat "$TMP/warm.rc")); log $TMP/warm.log"; grep -i 'reference cache' "$TMP/warm.log" | sed 's/^/        /'
fi

# Row 3: MUST-RED. Edit one stored artifact (the truth itself).
art=$(find "$CACHE" -path '*/files/*' -name '*stdout*' | head -1)
cp -r "$CACHE" "$TMP/cache-tampered"
art_t="$TMP/cache-tampered/${art#"$CACHE"/}"
# The edit keeps the answer RIGHT: only the stale check can make this RED, not the judge seeing a wrong truth.
printf '{"text": "<answer>4</answer>", "raw_text": "<answer>4</answer>\\n", "reported": {"device": "edited"}}\n' > "$art_t"
run_row stale "$T" "$TMP/cache-tampered"
stale=$(summary stale)
case "$stale" in
  "0 | "*"=RED"*) if grep -q 'is STALE' "$TMP/stale.out/stub-gpu.json" && ! printf '%s' "$stale" | grep -q '=GREEN'; then
                    ok "MUST-RED: an edited entry (answer still right) is refused STALE in the receipt, 0 engine calls, every cell RED ($stale)"
                  else broke "stale: RED but not every cell, or the refusal does not say STALE ($stale)"; fi ;;
  *) broke "stale: $stale — a tampered reference was not RED"; grep -i 'reference cache' "$TMP/stale.log" | sed 's/^/        /' ;;
esac

# Row 4: a new oracle version is a MISS, recomputed.
# The device on the probe line is NOT in the key (a host moved), the package version IS (the oracle moved).
run_row newhost "$T" "$CACHE" "STUB_PROBE_hf=hf=1.0.0 transformers=5.0.0 torch=2.0.0 device=another-host"
nh=$(summary newhost)
run_row newprobe "$T" "$CACHE" "STUB_PROBE_hf=hf=1.0.0 transformers=5.1.0 torch=2.0.0 device=stub"
np=$(summary newprobe)
case "$nh|$np" in
  "0 | "*"=GREEN"*"|2 | "*"=GREEN"*) ok "oracle key: another device is a HIT ($nh); a new transformers is a MISS, recomputed GREEN ($np)" ;;
  *) broke "oracle key: device change $nh; version change $np" ;;
esac

# Row 5: a refused reference row is never stored.
run_row refuse "$T" "$TMP/cache-refuse" STUB_REFUSE_llamafile=1
lf=$(find "$TMP/cache-refuse" -name entry.json -exec grep -l '"engine": "llamafile"' {} + 2>/dev/null | wc -l)
run_row refuse2 "$T" "$TMP/cache-refuse"
r2=$(summary refuse2)
if [ "$lf" -eq 0 ] && [ "${r2%% |*}" = 2 ]; then
  ok "a refused llamafile row was not stored; the next run was a MISS and ran both engines ($r2)"
else
  broke "refused row: $lf llamafile entries stored, next run $r2"
fi

# Row 6: MUTANT — the stale check disabled. Row 3's receipt must stop being RED, or the table is blind.
MT="$TMP/mutant"; mk_tree "$MT"
python3 - "$MT/scripts/lib/crux_ref_cache.py" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
a = '    """None when the entry at edir is intact for key; otherwise why it may not be reused."""\n'
assert s.count(a) == 1, "mutation anchor moved: update this check with the lib"
open(p, "w").write(s.replace(a, a + "    return None\n"))
PY
# The lib is in its own harness digest, so the mutant keys differently from the real lib: it must fill ITS OWN cache
# and have ITS entry tampered. Otherwise it misses, recomputes and is GREEN for the wrong reason (measured: the row
# passed vacuously that way once the lib joined the digest). Required: 0 engine calls AND non-RED — the edited truth
# really reused.
run_row mutant-cold "$MT" "$TMP/cache-mutant"
mart=$(find "$TMP/cache-mutant" -path '*/files/*' -name '*stdout*' | head -1)
if [ -n "$mart" ]; then
  printf '{"text": "<answer>4</answer>", "raw_text": "<answer>4</answer>\\n", "reported": {"device": "edited"}}\n' > "$mart"
fi
run_row mutant "$MT" "$TMP/cache-mutant"
mu=$(summary mutant)
case "$mu" in
  "0 | "*=RED*) broke "MUTANT (stale check disabled) still RED: the table cannot see the stale check ($mu)" ;;
  "0 | "*=GREEN*) ok "MUTANT (stale check disabled) REUSES the edited entry, 0 engine calls, GREEN: row 3 is what catches it ($mu)" ;;
  *) broke "MUTANT row is vacuous: the mutant never hit its own cache ($mu; artifact ${mart:-none})" ;;
esac

# Row 7: llama.cpp is never cached. Its llama-server renders apr's raw-prompt serve routes; a hit that switched it
# off refused 14 of apr's own serve cells on gx10 (2026-09-23). The lib refuses it by name.
why=$(cd "$T" && python3 scripts/lib/crux_ref_cache.py lookup --cache "$TMP/c7" --work "$TMP" --manifest /dev/null \
  --model-sha "$MSHA" --thinking off --backend gpu --host stub --engines llama.cpp --verbs run \
  --oracle "llama.cpp=b1" --temperature 0 --seed 42 --context 4096 --max-tokens 1024 --root "$T" \
  --out-rows "$TMP/c7.rows" 2>&1)
rc=$?
case "$rc:$why" in
  1:*"reference renderer"*) ok "llama.cpp cannot be cached: refused by name (it renders apr's serve routes)" ;;
  *) broke "llama.cpp cacheable? rc $rc: $why" ;;
esac

# Row 8: the run/chat key carries the cap the engine is GIVEN — the mode's global cap. A prompt set whose largest
# budget moved (another prompt edited) changes it; the old truth was generated under a different cap (quorum round 1).
T2="$TMP/tree-cap"; mk_tree "$T2"
python3 - "$T2/scripts/crux_inference_prompts.v2.json" "$MSHA" "$TMP/cert-cap.json" <<'PY' || exit 2
import hashlib, json, sys
p, sha, out = sys.argv[1:4]
d = json.load(open(p))
other = next(x for x in d["prompts"] if x["id"] != "ctl-2plus2" and isinstance(x.get("max_tokens"), dict))
other["max_tokens"]["off"] = max(int(x["max_tokens"]["off"]) for x in d["prompts"] if isinstance(x.get("max_tokens"), dict)) * 2
json.dump(d, open(p, "w"), indent=2)
json.dump({"schema": "crux-prompt-certification/v1", "prompts": "scripts/crux_inference_prompts.v2.json",
           "prompts_sha256": hashlib.sha256(open(p, "rb").read()).hexdigest(), "admitted": {},
           "admitted_by_sha": {sha: ["ctl-2plus2"]}, "admitted_by_sha_thinking": {sha: {"off": ["ctl-2plus2"]}}},
          open(out, "w"))
PY
RUN_CERT="$TMP/cert-cap.json" run_row newcap "$T2" "$CACHE"
nc=$(summary newcap)
case "$nc" in
  "2 | "*"=GREEN"*) ok "a new global cap is a MISS for run/chat, recomputed GREEN ($nc)" ;;
  *) broke "new global cap: $nc (a hit would reuse a truth generated under another cap)" ;;
esac

# Row 9: MUST-RED — a row's artifact POINTER rewritten to a traversal, key and files{} untouched and the content
# digest RE-SEALED, so only the pointer rule can refuse it (quorum round 3, lane 2 measured the traversal passing as
# intact and reading outside the cache). It is STALE: 0 engine calls, every cell RED.
cp -r "$CACHE" "$TMP/cache-ptr"
python3 - "$TMP/cache-ptr" <<'PY' || exit 2
import glob, hashlib, json, sys
# every entry: the cache holds entries of several keys by now (rows 4 and 8), and only this run's must be hit
for p in glob.glob(sys.argv[1] + "/*/*/entry.json"):
    e = json.load(open(p))
    e["rows"][0]["stdout"] = "refcache:" + "../" * 8 + "etc/hostname"
    e["content_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in e.items() if k != "content_sha256"},
                                                    sort_keys=True).encode()).hexdigest()  # RE-SEALED: only the pointer rule may refuse it
    json.dump(e, open(p, "w"), indent=1, sort_keys=True)
PY
run_row ptr "$T" "$TMP/cache-ptr"
pt=$(summary ptr)
case "$pt" in
  "0 | "*"=RED"*) if grep -q "is not one of the entry's own hashed files" "$TMP/ptr.out/stub-gpu.json" && ! printf '%s' "$pt" | grep -q '=GREEN'; then
                    ok "MUST-RED: a traversal pointer (key and files{} intact) is STALE, 0 engine calls, every cell RED ($pt)"
                  else broke "pointer: RED but not for the pointer ($pt)"; fi ;;
  *) broke "pointer: $pt — an edited pointer was reused or recomputed, not refused" ;;
esac

# Row 10: the lookup's exit contract — hit 0, miss 10, stale 11, and anything else is NOT one of them: a keying
# refusal exits 1 and a crash 2 (a crash once shared the miss code, so a corrupted cache was silently recomputed).
lk() { python3 "$T/scripts/lib/crux_ref_cache.py" lookup --cache "$1" --work "$2" --manifest /dev/null \
  --model-sha "$MSHA" --thinking off --backend gpu --host stub --engines hf --verbs "$3" --oracle "hf=transformers=5.0.0" \
  --temperature 0 --seed 42 --context 4096 --max-tokens 1024 --root "$T" --out-rows "$TMP/lk.rows" > /dev/null 2>&1; echo $?; }
W10="$TMP/w10"; mkdir -p "$W10"; printf 'run ctl-2plus2\n' > "$W10/pids-all.txt"; printf '{}' > "$W10/prompt-ctl-2plus2.json"
printf '["ctl-2plus2", ["serve run"]]\n' > "$W10/serve-prompts.jsonl"
r_miss=$(lk "$TMP/c10" "$W10" run); r_key=$(lk "$TMP/c10" "$W10" run,serve); r_crash=$(lk "$TMP/c10" "$TMP/no-such-work" run)
[ "$r_miss:$r_key:$r_crash" = "10:1:2" ] && ok "exit contract: miss 10, keying refusal 1 (no cap file), crash 2 — none of the latter two is a miss" \
  || broke "exit contract: miss $r_miss (want 10), keying $r_key (want 1), crash $r_crash (want 2)"

# Row 11: MUST-DECLINE — a lookup that CRASHES inside the real dogfood declines the run; it is never read as a miss
# and recomputed GREEN.
T3="$TMP/tree-crash"; mk_tree "$T3"
python3 - "$T3/scripts/lib/crux_ref_cache.py" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
a = "def cmd_lookup(a):\n"
assert s.count(a) == 1, "crash anchor moved: update this check with the lib"
open(p, "w").write(s.replace(a, a + "    raise RuntimeError('planted crash')\n"))
PY
run_row crash "$T3" "$TMP/cache-crash"
cr=$(summary crash)
if [ "$(cat "$TMP/crash.rc")" = 2 ] && grep -q 'neither hit, miss nor stale' "$TMP/crash.log" && [ "${cr#* | }" = "NO-RECEIPT | -" ]; then
  ok "a crashing lookup DECLINES the run (rc 2, no receipt), never a silent recompute"
else
  broke "crashing lookup: rc $(cat "$TMP/crash.rc"), $cr"
fi

# Row 12: the row<->file correspondence is what refuses an edited pointer. `refcache:./<same name>` resolves to the
# SAME hashed file, with the digest re-sealed, so no sha256, digest or crash can catch it; the pointer rule (a row's
# pointer is a plain name files{} holds) and its converse (files{} holds only what a row points at) each do. Real lib:
# STALE. MUTANT with both disabled (its own cache, since the lib is in the digest): REUSED, 0 calls, GREEN.
dot_ptr() { # dot_ptr <cache>: every entry's first row points at ./<its own file>
  python3 - "$1" <<'PY' || exit 2
import glob, hashlib, json, sys
for p in glob.glob(sys.argv[1] + "/*/*/entry.json"):
    e = json.load(open(p))
    v = e["rows"][0]["stdout"]
    e["rows"][0]["stdout"] = "refcache:./" + v[len("refcache:"):]
    e["content_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in e.items() if k != "content_sha256"},
                                                    sort_keys=True).encode()).hexdigest()  # RE-SEALED: only the pointer rule may refuse it
    json.dump(e, open(p, "w"), indent=1, sort_keys=True)
PY
}
cp -r "$CACHE" "$TMP/cache-dot"; dot_ptr "$TMP/cache-dot"
run_row dot "$T" "$TMP/cache-dot"
dt=$(summary dot)
MP="$TMP/mutant-ptr"; mk_tree "$MP"
python3 - "$MP/scripts/lib/crux_ref_cache.py" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
a = "            if not rel or rel != os.path.basename(rel) or rel in (\".\", \"..\") or rel not in (ent.get(\"files\") or {}):"
assert s.count(a) == 1, "pointer-rule anchor moved: update this check with the lib"
s = s.replace(a, "            if False:")
b = "    if extra:"
assert s.count(b) == 1, "converse-rule anchor moved: update this check with the lib"
open(p, "w").write(s.replace(b, "    if False:"))
PY
run_row mptr-cold "$MP" "$TMP/cache-mptr"; dot_ptr "$TMP/cache-mptr"
run_row mptr "$MP" "$TMP/cache-mptr"
mp=$(summary mptr)
case "$dt|$mp" in
  "0 | "*"=RED"*"|0 | "*"=GREEN"*) ok "an equivalent-but-edited pointer: STALE on the real lib ($dt); REUSED with the row<->file rules disabled ($mp)" ;;
  *) broke "pointer rule: real lib $dt, mutant $mp" ;;
esac

# Row 13: a FIELD REMOVED from a stored row. Deleting `stdout` passed every per-field rule (quorum round 4, lane 1,
# measured); the entry's content digest now refuses any field added, removed or changed. MUST-RED on `stdout`;
# and `backend` — a field no per-field rule reads, whose removal orphans no file — is STALE on the real lib but
# REUSED by a mutant without the content digest, so the digest (not another rule, not the judge) refuses it.
del_field() { # del_field <cache> <field>: every entry's first row loses <field>; nothing else is touched
  python3 - "$1" "$2" <<'PY' || exit 2
import glob, json, sys
for p in glob.glob(sys.argv[1] + "/*/*/entry.json"):
    e = json.load(open(p))
    e["rows"][0].pop(sys.argv[2], None)
    json.dump(e, open(p, "w"), indent=1, sort_keys=True)
PY
}
cp -r "$CACHE" "$TMP/cache-del"; del_field "$TMP/cache-del" stdout
run_row del "$T" "$TMP/cache-del"
dl=$(summary del)
cp -r "$CACHE" "$TMP/cache-dels"; del_field "$TMP/cache-dels" backend
run_row dels "$T" "$TMP/cache-dels"
ds=$(summary dels)
MC="$TMP/mutant-content"; mk_tree "$MC"
python3 - "$MC/scripts/lib/crux_ref_cache.py" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
a = "    if not ent.get(\"content_sha256\") or content_sha256(ent) != ent[\"content_sha256\"]:"
assert s.count(a) == 1, "content-digest anchor moved: update this check with the lib"
open(p, "w").write(s.replace(a, "    if False:"))
PY
run_row mcon-cold "$MC" "$TMP/cache-mcon"; del_field "$TMP/cache-mcon" backend
run_row mcon "$MC" "$TMP/cache-mcon"
mc=$(summary mcon)
case "$dl|$ds|$mc" in
  "0 | "*"=RED"*"|0 | "*"=RED"*"|0 | "*"=GREEN"*)
    if grep -q 'its content changed since it was stored' "$TMP/del.out/stub-gpu.json" "$TMP/dels.out/stub-gpu.json"; then
      ok "a REMOVED field is STALE (stdout: $dl; backend: $ds); a mutant without the content digest REUSES the backend edit ($mc)"
    else broke "field removal: RED but not for the content digest"; fi ;;
  *) broke "field removal: stdout $dl, stderr $ds, mutant $mc" ;;
esac

# Row 14: an EXTRA file smuggled into files{} (hashed correctly, digest re-sealed) that no row points at is STALE:
# an entry carries only the files its rows need (quorum round 5, lane 2, measured the addition passing).
cp -r "$CACHE" "$TMP/cache-extra"
python3 - "$TMP/cache-extra" <<'PY' || exit 2
import glob, hashlib, json, os, sys
for p in glob.glob(sys.argv[1] + "/*/*/entry.json"):
    e = json.load(open(p))
    extra = os.path.join(os.path.dirname(p), "files", "smuggled.txt")
    open(extra, "w").write("not a row's artifact\n")
    e["files"]["smuggled.txt"] = hashlib.sha256(open(extra, "rb").read()).hexdigest()
    e["content_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in e.items() if k != "content_sha256"},
                                                    sort_keys=True).encode()).hexdigest()
    json.dump(e, open(p, "w"), indent=1, sort_keys=True)
PY
run_row extra "$T" "$TMP/cache-extra"
ex=$(summary extra)
# ...and a MUTANT without the converse rule alone REUSES it: no other rule sees an extra, correctly hashed file.
MX="$TMP/mutant-extra"; mk_tree "$MX"
python3 - "$MX/scripts/lib/crux_ref_cache.py" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
b = "    if extra:"
assert s.count(b) == 1, "converse-rule anchor moved: update this check with the lib"
open(p, "w").write(s.replace(b, "    if False:"))
PY
run_row mext-cold "$MX" "$TMP/cache-mext"
python3 - "$TMP/cache-mext" <<'PY' || exit 2
import glob, hashlib, json, os, sys
for p in glob.glob(sys.argv[1] + "/*/*/entry.json"):
    e = json.load(open(p))
    extra = os.path.join(os.path.dirname(p), "files", "smuggled.txt")
    open(extra, "w").write("not a row's artifact\n")
    e["files"]["smuggled.txt"] = hashlib.sha256(open(extra, "rb").read()).hexdigest()
    e["content_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in e.items() if k != "content_sha256"},
                                                    sort_keys=True).encode()).hexdigest()
    json.dump(e, open(p, "w"), indent=1, sort_keys=True)
PY
run_row mext "$MX" "$TMP/cache-mext"
mx=$(summary mext)
case "$ex|$mx" in
  "0 | "*"=RED"*"|0 | "*"=GREEN"*) grep -q 'which no row points at' "$TMP/extra.out/stub-gpu.json" \
                  && ok "an unreferenced file in files{} (hashed, re-sealed) is STALE ($ex); a mutant without the converse rule REUSES it ($mx)" \
                  || broke "extra file: RED but not for the unreferenced file ($ex)" ;;
  *) broke "extra file: real lib $ex, mutant $mx" ;;
esac

# Row 15: MUST-RED — an ORIGIN-only edit, NOT re-sealed (the provenance a hit reports). The seal once covered only
# {key, rows, files}, so this passed as a HIT and served a false origin (quorum round 6, lane 2, measured).
cp -r "$CACHE" "$TMP/cache-origin"
python3 - "$TMP/cache-origin" <<'PY' || exit 2
import glob, json, sys
for p in glob.glob(sys.argv[1] + "/*/*/entry.json"):
    e = json.load(open(p))
    e["origin"]["host"] = "not-the-host-that-measured-it"
    json.dump(e, open(p, "w"), indent=1, sort_keys=True)
PY
run_row origin "$T" "$TMP/cache-origin"
og=$(summary origin)
case "$og" in
  "0 | "*"=RED"*) grep -q 'its content changed since it was stored' "$TMP/origin.out/stub-gpu.json" \
                  && ok "an origin-only edit (not re-sealed) is STALE ($og)" \
                  || broke "origin edit: RED but not for the seal ($og)" ;;
  *) broke "origin edit: $og — a false provenance was served" ;;
esac

# Row 16: a driver that appends TWO rows for one item: the cell refuses it (RED, never a pick) and the cache never
# stores it — an entry is exactly one row — so the next run is a MISS that measures again (quorum round 7, lane 2).
run_row dup "$T" "$TMP/cache-dup" STUB_DUP_hf=1
dp=$(summary dup)
hfe=$(find "$TMP/cache-dup" -name entry.json -exec grep -l '"engine": "hf"' {} + 2>/dev/null | wc -l)
run_row dup2 "$T" "$TMP/cache-dup"
d2=$(summary dup2)
case "$dp|$hfe|$d2" in
  *"=RED"*"|0|2 | "*"=GREEN"*) ok "an item answered TWICE: its cell RED ($dp), never stored (0 hf entries), next run a MISS measured again ($d2)" ;;
  *) broke "duplicate rows: first $dp, hf entries $hfe, next $d2" ;;
esac

printf '%s: %d ok, %d broke\n' "$PROG" "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ] || exit 1
exit 0
