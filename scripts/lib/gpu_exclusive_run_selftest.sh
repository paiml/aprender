#!/usr/bin/env bash
# Case table for gpu_exclusive_run.sh. GPU-FREE: a stub nvidia-smi on PATH plays the card,
# scripted per case, so every exit path is exercised without hardware.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
T=$(mktemp -d) || exit 2
trap 'rm -rf "${T:?}"' EXIT
mkdir -p "$T/bin"
cat > "$T/bin/nvidia-smi" <<'STUB'
#!/usr/bin/env bash
# Prints the contents of $CARD_FILE (empty = empty card).
[ -f "${CARD_FILE:-}" ] && cat "$CARD_FILE"
exit 0
STUB
chmod +x "$T/bin/nvidia-smi"
bad=0; n=0
row() { # row <want rc> <label> <must_match> -- env and command follow
  n=$((n + 1)); local want=$1 label=$2 mm=$3; shift 3
  local out rc
  out=$(env PATH="$T/bin:$PATH" GPU_LOCK="$T/lock" GPU_WAIT_SECS=2 "$@" 2>&1); rc=$?
  # Match against the output JOINED: the script names each foreign process on its own
  # line, so a line-based match cannot see "CONTENDED ... llama-server" across the break.
  if [ "$rc" = "$want" ] && grep -qE "$mm" <<< "$(tr '\n' ' ' <<< "$out")"; then printf 'ok    row %s rc=%s  %s\n' "$n" "$rc" "$label"
  else printf 'FAIL  row %s rc=%s (want %s)  %s\n        %s\n' "$n" "$rc" "$want" "$label" "$(echo "$out" | tail -2 | tr '\n' ' ')"; bad=$((bad + 1)); fi
}
MINE=/work/mine/target/
S="$HERE/gpu_exclusive_run.sh"

# 1. exclusive: only our binary appears while the command runs
row 0 "only our binary on the card: EXCLUSIVE, command rc passed through" "EXCLUSIVE -- [1-9]" \
  bash -c "echo '' > $T/card; GPU_OWNED_PREFIX=$MINE CARD_FILE=$T/card $S bash -c 'echo \"111, ${MINE}deps/realizar\" > $T/card; sleep 2.5; : > $T/card; exit 0'"
# 2. the command's own failure is returned, not masked
row 7 "command fails on an exclusive card: its rc (7) is returned" "EXCLUSIVE" \
  bash -c "GPU_OWNED_PREFIX=$MINE CARD_FILE=$T/card $S bash -c 'echo \"111, ${MINE}deps/realizar\" > $T/card; sleep 2.5; : > $T/card; exit 7'"
# 3. THE ROW: a foreigner arrives MID-RUN after a clear start -- the Ollama case
row 75 "foreigner arrives mid-run (the Ollama case): CONTENDED" "CONTENDED -- foreign.*llama-server" \
  bash -c "GPU_OWNED_PREFIX=$MINE CARD_FILE=$T/card $S bash -c 'printf \"111, ${MINE}deps/realizar\n404921, /usr/local/lib/ollama/llama-server\n\" > $T/card; sleep 2.5; : > $T/card; exit 0'"
# 3b. OUR OWN process at teardown reads "<pid>, [No data]". Same pid as our earlier samples
#     -> ours, EXCLUSIVE. This is the false refusal the first release made on a real run.
row 0 "our own pid turns nameless at teardown: still EXCLUSIVE" "EXCLUSIVE -- [1-9]" \
  bash -c "GPU_OWNED_PREFIX=$MINE CARD_FILE=$T/card $S bash -c 'echo \"222, ${MINE}deps/realizar\" > $T/card; sleep 2.5; echo \"222, [No data]\" > $T/card; sleep 0.5; : > $T/card; exit 0'"
# 3c. a nameless pid we NEVER saw with our path is foreign -- it could be one starting up.
row 75 "a nameless pid never seen as ours: CONTENDED (conservative)" "CONTENDED -- foreign.*333, \\[No data\\]" \
  bash -c "GPU_OWNED_PREFIX=$MINE CARD_FILE=$T/card $S bash -c 'printf \"222, ${MINE}deps/realizar\n333, [No data]\n\" > $T/card; sleep 2.5; : > $T/card; exit 0'"
# 4. card never clears within the wait: refuse before taking the lock
row 75 "card occupied throughout: CONTENDED, command never runs" "never cleared" \
  bash -c "echo '404921, /usr/local/lib/ollama/llama-server' > $T/card; GPU_OWNED_PREFIX=$MINE CARD_FILE=$T/card $S bash -c 'echo RAN > $T/ran'"
[ ! -f "$T/ran" ] || { echo "FAIL  row 4 ran the command on an occupied card"; bad=$((bad + 1)); }
# 5. nothing observed: a run no sample caught cannot claim exclusivity
row 75 "run too short to be sampled: UNVERIFIED, not EXCLUSIVE" "UNVERIFIED" \
  bash -c ": > $T/card; GPU_OWNED_PREFIX=$MINE CARD_FILE=$T/card $S true"
# 6. no prefix: refuse -- a default would accept every process as ours
row 2 "GPU_OWNED_PREFIX unset: usage error" "GPU_OWNED_PREFIX is required" \
  bash -c "unset GPU_OWNED_PREFIX; $S true"
# ---- INHERITED LOCK (#3964 follow-up) ---------------------------------------------------
# A caller that already holds the lock -- gpu-q ends in `exec flock $LOCK choom -- <cmd>` --
# used to DEADLOCK: the script re-took the lock on a new file description with no bound, and
# waited forever while its own ancestor held it. Every row below is wrapped in `timeout 20`
# so a regression FAILS the table instead of hanging it; the outer flock uses -w 5.
# 7. THE ROW f5 asked for: caller holds the lock -> proceeds, EXCLUSIVE, no deadlock.
row 0 "caller already holds the lock: proceeds, no deadlock" "held by ancestor pid [0-9]+.*EXCLUSIVE -- [1-9]" \
  bash -c ": > $T/card; timeout 20 flock -w 5 $T/lock env GPU_OWNED_PREFIX=$MINE CARD_FILE=$T/card $S bash -c 'echo \"111, ${MINE}deps/realizar\" > $T/card; sleep 2.5; : > $T/card; exit 0'"
# 8. holding the lock does not excuse a foreigner: one arriving mid-run is still refused.
row 75 "caller holds the lock, a foreigner arrives mid-run: still CONTENDED" "held by ancestor.*CONTENDED -- foreign.*llama-server" \
  bash -c ": > $T/card; timeout 20 flock -w 5 $T/lock env GPU_OWNED_PREFIX=$MINE CARD_FILE=$T/card $S bash -c 'printf \"111, ${MINE}deps/realizar\n404921, /usr/local/lib/ollama/llama-server\n\" > $T/card; sleep 2.5; : > $T/card; exit 0'"
# 9. held lock but the card is ALREADY occupied: refuse, and never run the command.
rm -f "$T/ran9"
row 75 "caller holds the lock but the card is occupied: refused, command never runs" "held by ancestor.*occupied after taking the lock" \
  bash -c "echo '404921, /usr/local/lib/ollama/llama-server' > $T/card; timeout 20 flock -w 5 $T/lock env GPU_OWNED_PREFIX=$MINE CARD_FILE=$T/card $S bash -c 'echo RAN > $T/ran9'"
[ ! -f "$T/ran9" ] || { echo "FAIL  row 9 ran the command on an occupied card"; bad=$((bad + 1)); }
# 10. a lock held by an UNRELATED process is contention, not inheritance: bounded wait,
#     then a named refusal. Proves the check is ANCESTRY, not "someone holds it".
rm -f "$T/ran10"
row 75 "lock held by a non-ancestor: NOT inherited, bounded wait then refused" "not free after [0-9]+s" \
  bash -c ": > $T/card; flock $T/lock sleep 6 & sleep 0.3; timeout 20 env GPU_OWNED_PREFIX=$MINE CARD_FILE=$T/card $S bash -c 'echo RAN > $T/ran10'; rc=\$?; wait; exit \$rc"
[ ! -f "$T/ran10" ] || { echo "FAIL  row 10 ran the command while an unrelated process held the lock"; bad=$((bad + 1)); }
printf '%s/%s rows\n' "$((n - bad))" "$n"
[ "$bad" = 0 ]
