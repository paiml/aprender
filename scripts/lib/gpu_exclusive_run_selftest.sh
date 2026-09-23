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
  else printf 'FAIL  row %s rc=%s (want %s)  %s\n        %s\n' "$n" "$rc" "$want" "$label" "$(echo "$out" | tail -2 | tr '\n' ' ')"; bad=1; fi
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
[ ! -f "$T/ran" ] || { echo "FAIL  row 4 ran the command on an occupied card"; bad=1; }
# 5. nothing observed: a run no sample caught cannot claim exclusivity
row 75 "run too short to be sampled: UNVERIFIED, not EXCLUSIVE" "UNVERIFIED" \
  bash -c ": > $T/card; GPU_OWNED_PREFIX=$MINE CARD_FILE=$T/card $S true"
# 6. no prefix: refuse -- a default would accept every process as ours
row 2 "GPU_OWNED_PREFIX unset: usage error" "GPU_OWNED_PREFIX is required" \
  bash -c "unset GPU_OWNED_PREFIX; $S true"
printf '%s/%s rows\n' "$((n - bad))" "$n"
[ "$bad" = 0 ]
