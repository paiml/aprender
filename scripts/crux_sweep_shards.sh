#!/usr/bin/env bash
# crux_sweep_shards.sh — the 0.69.1 final CRUX sweep for ONE host: every CERTIFIED model on it, each thinking mode,
# exactly that (model, mode)'s admitted prompts; then ONE judge receipt for the host (#3962, final sweep).
#
#   bash scripts/crux_sweep_shards.sh <version> --host <id> --apr <binary> --out <dir>
#        [--backend gpu|cpu] [--scope controls|admitted] [--models-dir <dir>]... [--certification <receipt>]
#        [--greedy-model <gguf>]... [--greedy-only] [--dry-run]
#
# WHY PER (MODEL, MODE). The certification admits prompts per quant sha AND thinking mode
# (admitted_by_sha_thinking); a prompt run outside its admission is RED at the judge by design. So each shard is
# one dogfood run: --model <that gguf> --thinking-modes <mode> --only-prompts <admitted ids> (intersected with the
# set's controls under --scope controls, the cop's default). A (model, mode) with nothing admitted is skipped BY
# NAME in the plan, never silently. A certified sha absent from this host is listed as absent, not run.
#
# MERGE. Every shard keeps its work dir. The host receipt is ONE `crux_inference_judge.py collect` over the
# concatenated shard manifests, with the certification, at <out>/<host>-<backend>.json.
# --greedy-model (#3957 F9) runs greedy-only shards (--greedy --greedy-prompts <first control>, thinking OFF):
# their GREEDY rows join the merge, their gen cells DO NOT — those models are not certified, so their gen cells
# would be RED by design and say nothing about F9. That exclusion is written into the plan file beside the receipt.
# --greedy-only (#4004, the CPU lane): run ONLY the greedy shards — the certified cells' 4096-token thinking-ON
# budgets are impractical on a CPU lane, and F9's CPU reference needs greedy rows alone. The receipt then has no
# judged cell, so the judge DECLINES it (exit 2, "no cell was measured") while its greedy[] carries the rows: a
# greedy-only receipt is F9 evidence, never a CRUX verdict, and the plan file says so.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2
PROG=crux_sweep_shards
die() { printf '%s: %s\n' "$PROG" "$1" >&2; exit 2; }

VERSION=""; HOST=""; APR_BIN=""; OUT=""; BACKEND=gpu; SCOPE=controls; DRY=0; GREEDY_ONLY=0
CERT="evidence/crux/0.69.1/prompt-certification.json"; PROMPTS="scripts/crux_inference_prompts.v2.json"
MODEL_DIRS=(); GREEDY_MODELS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --apr) APR_BIN="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --scope) SCOPE="$2"; shift 2 ;;
    --models-dir) MODEL_DIRS+=("$2"); shift 2 ;;
    --certification) CERT="$2"; shift 2 ;;
    --greedy-model) GREEDY_MODELS+=("$2"); shift 2 ;;
    --greedy-only) GREEDY_ONLY=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -*) die "unknown argument '$1'" ;;
    *) [ -z "$VERSION" ] || die "one version"; VERSION="$1"; shift ;;
  esac
done
[ -n "$VERSION" ] && [ -n "$HOST" ] && [ -n "$APR_BIN" ] && [ -n "$OUT" ] || die "usage: <version> --host <id> --apr <bin> --out <dir>"
case "$SCOPE" in controls|admitted) ;; *) die "--scope is controls or admitted" ;; esac
[ -x "$APR_BIN" ] || die "--apr $APR_BIN is not executable"
[ -f "$CERT" ] || die "certification receipt $CERT not found"
[ "${#MODEL_DIRS[@]}" -gt 0 ] || MODEL_DIRS=("$HOME/models")
mkdir -p "$OUT/shards" || die "cannot create $OUT"
PLAN="$OUT/$HOST-$BACKEND.plan.tsv"

# The plan: one line per (model, mode) — RUN with its ids, or SKIP with the reason.
python3 - "$CERT" "$PROMPTS" "$SCOPE" "$PLAN" "${MODEL_DIRS[@]}" <<'PY' || die "could not build the plan"
import hashlib, json, os, sys
cert, prompts, scope, plan = sys.argv[1:5]
dirs = sys.argv[5:]
c = json.load(open(cert))
p = json.load(open(prompts))
if c.get("prompts_sha256") and c["prompts_sha256"] != hashlib.sha256(open(prompts, "rb").read()).hexdigest():
    sys.exit("the certification binds prompts sha %s, but %s is %s" % (c["prompts_sha256"][:12], prompts,
             hashlib.sha256(open(prompts, "rb").read()).hexdigest()[:12]))
controls = {x["id"] for x in p["prompts"] if x.get("control")}
adm = c["admitted_by_sha_thinking"]
local = {}
for d in dirs:
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if f.endswith(".gguf"):
            path = os.path.join(d, f)
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 24), b""):
                    h.update(chunk)
            local.setdefault(h.hexdigest(), path)
with open(plan, "w") as out:
    for sha, modes in sorted(adm.items()):
        if sha not in local:
            out.write("ABSENT\t%s\t-\t-\tcertified, but no file with this sha256 in %s\n" % (sha, " ".join(dirs)))
            continue
        for mode in ("off", "on"):
            ids = [i for i in modes.get(mode, []) if scope == "admitted" or i in controls]
            if not ids:
                why = "nothing admitted" if not modes.get(mode) else "no CONTROL admitted (scope=controls)"
                out.write("SKIP\t%s\t%s\t%s\t%s\n" % (sha, mode, local[sha], why))
            else:
                out.write("RUN\t%s\t%s\t%s\t%s\n" % (sha, mode, local[sha], ",".join(ids)))
PY
printf '%s: plan for %s (%s lane, scope %s) -> %s\n' "$PROG" "$HOST" "$BACKEND" "$SCOPE" "$PLAN"
sed 's/^/  /' "$PLAN"
[ "$GREEDY_ONLY" = 1 ] && { [ "${#GREEDY_MODELS[@]}" -gt 0 ] || die "--greedy-only needs at least one --greedy-model"; \
  printf '  GREEDY-ONLY\tthe certified RUN lines above are NOT run on this lane; this receipt is F9 greedy evidence, not a CRUX verdict\n' | tee -a "$PLAN"; }
for g in "${GREEDY_MODELS[@]}"; do printf '  GREEDY\t%s\t(off; greedy rows only — its gen cells are NOT merged: uncertified)\n' "$g" | tee -a "$PLAN"; done
[ "$DRY" = 1 ] && exit 0

run_shard() { # run_shard <name> <dogfood args...> — one dogfood run; echoes its kept work dir
  local name="$1"; shift
  local log="$OUT/shards/$name.log"
  DOGFOOD_ALLOW_UNPINNED=1 APR="$APR_BIN" bash scripts/crux_inference_dogfood.sh "$VERSION" "$@" \
    --host "$HOST" --backend "$BACKEND" --certification "$CERT" --out "$OUT/shards/$name" --keep-work > "$log" 2>&1
  printf '%s\t%s\t%s\n' "$name" "$?" "$(sed -n 's/^work kept: //p' "$log" | tail -1)" >> "$OUT/shards.tsv"
}
: > "$OUT/shards.tsv"
while IFS=$'\t' read -r kind sha mode path ids; do
  [ "$kind" = RUN ] && [ "$GREEDY_ONLY" = 0 ] || continue
  run_shard "${sha:0:12}-$mode" --model "$path" --engines apr,llama.cpp,vllm,hf --verbs run,chat,serve,code \
    --thinking-modes "$mode" --only-prompts "$ids"
done < "$PLAN"
CTL=$(python3 -c 'import json,sys; print(next(p["id"] for p in json.load(open(sys.argv[1]))["prompts"] if p.get("control")))' "$PROMPTS")
for g in "${GREEDY_MODELS[@]}"; do
  run_shard "greedy-$(basename "$g" .gguf)" --model "$g" --engines apr,llama.cpp --verbs run --thinking-modes off \
    --only-prompts "$CTL" --greedy --greedy-prompts "$CTL" --greedy-max-tokens 256
done

# Merge: gen/tok rows from the certified shards, GREEDY rows only from the greedy shards; one judge receipt.
MERGED="$OUT/$HOST-$BACKEND.manifest.jsonl"; : > "$MERGED"
META=""
while IFS=$'\t' read -r name rc work; do
  [ -n "$work" ] && [ -f "$work/manifest.jsonl" ] || { printf '%s: shard %s kept no work dir (rc %s)\n' "$PROG" "$name" "$rc" >&2; continue; }
  case "$name" in
    greedy-*) python3 -c 'import json,sys
for l in open(sys.argv[1]):
    if json.loads(l).get("kind") == "greedy": sys.stdout.write(l)' "$work/manifest.jsonl" >> "$MERGED" ;;
    *) cat "$work/manifest.jsonl" >> "$MERGED"; [ -n "$META" ] || META="$work/meta.json" ;;
  esac
done < "$OUT/shards.tsv"
if [ -z "$META" ] && [ "$GREEDY_ONLY" = 1 ]; then
  META=$(sed -n 's/^greedy-[^\t]*\t[^\t]*\t//p' "$OUT/shards.tsv" | head -1)/meta.json
fi
[ -f "$META" ] || die "no shard produced a manifest; nothing to judge"
# A --greedy-only receipt is F9 evidence, never a host verdict (aprender-36's contract, fix/3957-f9-f10@09ea08424):
# it is named <host>-<backend>-greedy.json so it cannot pass for the certified <host>-<backend>.json, and it says
# `"greedy_only": true` with `"cells": []` — the ladder's cell join skips it (no cells claimed), the F9 judge reads its
# greedy[]. Without the marker the ladder globs it, sees DECLINE, and FAILS the cut (measured by 36 in the code).
RECEIPT="$OUT/$HOST-$BACKEND.json"
[ "$GREEDY_ONLY" = 1 ] && RECEIPT="$OUT/$HOST-$BACKEND-greedy.json"
python3 scripts/lib/crux_inference_judge.py collect --manifest "$MERGED" --prompts "$PROMPTS" --meta "$META" \
  --certification "$CERT" --out-json "$RECEIPT" --out-md "${RECEIPT%.json}.md"
rc=$?
if [ "$GREEDY_ONLY" = 1 ]; then
  python3 - "$RECEIPT" <<'PY' || die "the greedy-only receipt could not be marked"
import json, sys
p = sys.argv[1]
r = json.load(open(p))
if r.get("cells"):
    sys.exit("a greedy-only merge produced %d judged cells; it must produce none" % len(r["cells"]))
if not r.get("greedy"):
    sys.exit("a greedy-only receipt with an EMPTY greedy[] proves nothing")
r["greedy_only"] = True
r["cells"] = []
json.dump(r, open(p, "w"), indent=1)
print("greedy_only receipt: %d greedy entries, verdict %s" % (len(r["greedy"]), r["summary"].get("verdict")))
PY
  rc=0  # the judge's DECLINE ("no cell was measured") is the expected verdict of a receipt that claims no cell
fi
printf '%s: %s receipt %s (judge rc %s); shards: %s\n' "$PROG" "$HOST" "$RECEIPT" "$rc" "$OUT/shards.tsv"
exit "$rc"
