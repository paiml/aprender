#!/usr/bin/env bash
# PMAT-3091 variation, fix-up for prompts 2 and 4. llama-tokenize -f keeps the file's trailing "\n";
# the producer's common -f parser pops exactly one. First-attempt ids differed from the producer's in
# the LAST token only (p2 118 vs 117; p4 271 vs 198), so the comparator refused B/C (rc 2).
# Re-tokenize ONCE from the producer's exact text (file minus one trailing "\n", via --stdin; neither
# text contains a backslash, so escape handling is a no-op), prove ids == producer-logged ids,
# re-run apr on p2/p4, re-run B/C, then classify all 5 prompts.
set -uo pipefail
OUT="$HOME/parity-ref/variation-3091"; PDIR="/tmp/flipvar3091"
MODEL="$HOME/models/Qwen3.5-0.8B-Q4_K_M.gguf"; TOK="$HOME/src/llama.cpp-d1d3c3396/build/bin/llama-tokenize"
APR="$OUT/qwen35_raw_logits"; CMPPY="$HOME/parity-ref/apr-3091-subject/compare_raw_logits.py"
cd "$OUT" || exit 2
echo "fix start_utc=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
sha256sum "$TOK" "$APR" "$CMPPY"
for k in 2 4; do
  mv "$PDIR/prompt-$k.ids" "$PDIR/prompt-$k.ids.first-attempt"
  mv "$OUT/p$k-apr.bin" "$OUT/p$k-apr.first-attempt-ids.bin"; mv "$OUT/p$k-apr.log" "$OUT/p$k-apr.first-attempt-ids.log"
  python3 -c "import sys;t=open(sys.argv[1],'rb').read();t=t[:-1] if t.endswith(b'\n') else t;open(sys.argv[2],'wb').write(t)" "$PDIR/prompt-$k.txt" "$PDIR/prompt-$k.tok-input.txt"
  sha256sum "$PDIR/prompt-$k.tok-input.txt"
  timeout 120 "$TOK" -m "$MODEL" --stdin --ids --no-parse-special --log-disable < "$PDIR/prompt-$k.tok-input.txt" > "$PDIR/prompt-$k.ids.raw" 2> "$PDIR/prompt-$k.tok.err"
  echo "tokenize p$k rc=$?"
  python3 -c "import ast,sys;l=ast.literal_eval(open(sys.argv[1]).read().strip());open(sys.argv[2],'w').write(','.join(map(str,l)));print('n',len(l))" "$PDIR/prompt-$k.ids.raw" "$PDIR/prompt-$k.ids"
  sha256sum "$PDIR/prompt-$k.ids"
  for m in batched per-token; do
    if [ "$(sed -n 's/^apr_raw_logits: token_ids=//p' "$OUT/p$k-$m.log")" = "$(cat "$PDIR/prompt-$k.ids")" ]; then echo "p$k $m producer ids == prompt-$k.ids"; else echo "p$k $m producer ids DIFFER"; fi
  done
done
for k in 1 3; do for m in batched per-token; do
  if [ "$(sed -n 's/^apr_raw_logits: token_ids=//p' "$OUT/p$k-$m.log")" = "$(cat "$PDIR/prompt-$k.ids")" ]; then echo "p$k $m producer ids == prompt-$k.ids"; else echo "p$k $m producer ids DIFFER"; fi
done; done

for k in 2 4; do
  name="p$k-apr"; maxth=0
  echo "apr $name start_utc=$(date -u +%H:%M:%SZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
  /usr/bin/time -v timeout 1800 "$APR" "$MODEL" "$PDIR/prompt-$k.ids" "$OUT/$name.bin" < /dev/null > "$OUT/$name.log" 2>&1 &
  tpid=$!
  while kill -0 "$tpid" 2>/dev/null; do
    pid=$(pgrep -x qwen35_raw_logi | head -1)
    if [ -n "$pid" ]; then t=$(awk '/^Threads:/{print $2}' "/proc/$pid/status" 2>/dev/null); [ -n "$t" ] && [ "$t" -gt "$maxth" ] && maxth=$t; fi
    sleep 0.2
  done
  wait "$tpid"; rc=$?
  echo "apr $name rc=$rc end_utc=$(date -u +%H:%M:%SZ) load=$(cut -d' ' -f1-3 /proc/loadavg) max_threads_sampled=$maxth $(grep -E 'Percent of CPU|Elapsed' "$OUT/$name.log" | tr -s ' \t' ' ' | tr '\n' ';')"
  sha256sum "$OUT/$name.bin"
  for pair in "B-per-token-vs-apr per-token" "C-batched-vs-apr batched"; do
    set -- $pair; label="p$k-$1"; ref="$OUT/p$k-$2.bin"
    cmp -s "$ref" "$OUT/$name.bin"; echo "$label cmp rc=$? (0 = byte-identical)"
    timeout 600 python3 "$CMPPY" "$ref" "$OUT/$name.bin" --json "$OUT/$label.json" < /dev/null > "$OUT/$label.tsv"
    echo "$label compare rc=$?"; grep '^summary' "$OUT/$label.tsv"
  done
done
sha256sum "$OUT"/p?-?-*.tsv "$OUT"/p?-?-*.json
timeout 900 python3 "$OUT/classify_flips.py" "$OUT/classify_spec.json" "$OUT/flips.json" < /dev/null > "$OUT/classify.stdout" 2> "$OUT/classify.stderr"
echo "classify rc=$?"; sha256sum "$OUT/flips.json"; cat "$OUT/classify.stderr"
echo "fix done end_utc=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
