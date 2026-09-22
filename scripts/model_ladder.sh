#!/usr/bin/env bash
# model_ladder.sh — measure every rung of contracts/model-capability-ladder-v1.yaml AND
# every Q4_K model this host HOLDS, and write the per-host receipt the release reads.
#
# WHY. 0.68.1 shipped with dense Qwen3 producing garbage on CUDA on both fleet
# GPUs; the per-inference parity guard fell back to CPU so `apr run` looked fine,
# `apr qa` was the only surface that said "gibberish", and no release step ran
# `apr qa` on a Qwen3 (EPIC #3477). The unit of evidence was "a model ran". The
# unit here is (architecture, backend, silicon), one model at a time.
#
# THE UNIVERSE IS THE INVENTORY (#3712, operator 2026-09-21: "you must ensure all
# models Q4_K CUDA work; the end", and publishing with "most working" is a "p0 tire
# fire"). A hand-picked ladder let a red model through by leaving it off the list, or
# by marking it `required: false`. This run measures ladder ∪ inventory, where the
# inventory is every file on THIS host matching the contract's `inventory.patterns`
# (case-insensitive, depth 1) under its `inventory.dirs`. An inventory model that is
# not already a rung is measured on the inventory's backends (cuda). The receipt
# (schema v2) carries the measured inventory, so check_model_ladder.sh can name any
# model this host holds that the run did not prove.
#
# Usage:  bash scripts/model_ladder.sh [--host <id>] [--out <dir>] [--dry-run]
#   --host  ladder host id (default: derived from `hostname`, see host_id)
#   --out   receipt dir (default: evidence/dogfood/models/<version>); writes <dir>/<host>.json
#
# Exit: 0 every present required rung AND every inventory model green · 1 any red,
# or a required rung absent · 2 decline (nothing executed, binary unpinned, ladder
# or inventory unreadable). The receipt is written on 0 and 1, never on 2: a
# receipt that measured nothing is not evidence.
#
# Every command's status is captured as `out=$(cmd); rc=$?` — never through a
# pipe (#2336, #2360).
set -uo pipefail

LADDER="contracts/model-capability-ladder-v1.yaml"
HOST_ID=""
OUT_DIR=""
DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --host) [ $# -ge 2 ] || { echo "model_ladder: --host needs a value" >&2; exit 2; }; HOST_ID="$2"; shift 2 ;;
    --out)  [ $# -ge 2 ] || { echo "model_ladder: --out needs a value" >&2; exit 2; };  OUT_DIR="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    # --lock-probe <apr args…>: one apr call through apr_locked, then exit with its rc. For the case
    # table in check_model_ladder.sh, which proves every apr call runs under the lock.
    --lock-probe) shift; LOCK_PROBE=1; break ;;
    -h|--help) awk 'NR == 1 { next } !/^#/ { exit } { sub(/^# ?/, ""); print }' "$0"; exit 0 ;;
    *) echo "model_ladder: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

# The root is derived from this file's path, never from `git rev-parse` (aprender#3581) or the caller's cwd.
# MODEL_LADDER_ROOT tells a mutant copy (in a temp dir) which tree it runs against.
cd "${MODEL_LADDER_ROOT:-$(dirname "$0")/..}" || exit 2
[ -f "$LADDER" ] || { echo "decline: $LADDER not found" >&2; exit 2; }

# Step 0 — pin the binary. A diagnostic against the wrong apr is worse than none.
if [ "${DOGFOOD_ALLOW_UNPINNED:-0}" = "1" ] && [ -n "${APR:-}" ]; then
  : # published-crate mode (apr-dogfood G13): caller pinned $APR deliberately
else
  # shellcheck disable=SC1091
  . scripts/apr_bin.sh || { echo "decline: scripts/apr_bin.sh could not pin a HEAD-built apr" >&2; exit 2; }
fi
[ -x "${APR:-}" ] || { echo "decline: \$APR is not executable: '${APR:-}'" >&2; exit 2; }

# THE GPU LOCK (cop ruling 2026-09-21, #3712). Interactive apr runs on gx10 drove the host into a
# global OOM at 15:56:08Z and the kernel killed CI containers (oom_score_adj 500), not apr (0). So
# every apr call that can touch the GPU goes through apr_locked and nowhere else: serialized on the
# fleet lock, and choom'd to 1000 so a measurement, never the pool, is the OOM victim. The wait is
# bounded: a lock that stays held is an ENV decline naming its holder, never a hang and never a
# model verdict. T-1 (the autopilot) wraps this script in choom only -- a second flock outside
# would deadlock on this one. check_model_ladder.sh audits that no GPU apr call bypasses this.
GPU_LOCK="${MODEL_LADDER_GPU_LOCK:-/tmp/apr-gpu.lock}"
LOCK_WAIT="${MODEL_LADDER_LOCK_WAIT:-1800}"
LOCK_BUSY=75   # flock -E: the lock was not free in LOCK_WAIT seconds (an apr exit 75 also declines -- never a pass)
command -v flock > /dev/null && command -v choom > /dev/null \
  || { echo "decline: flock and choom (util-linux) are required -- every apr call runs under the fleet GPU lock" >&2; exit 2; }
apr_locked() { flock -E "$LOCK_BUSY" -w "$LOCK_WAIT" "$GPU_LOCK" choom -n 1000 -- "$APR" "$@"; }

# ── #3843: ASK THE BINARY whether a verb takes a flag; never assume ───────────
# The verb loop hard-coded one backend flag and passed it to all four verbs. Their
# clap surfaces differ, so `apr code` got `--no-gpu`, exited 2 on a usage error,
# and 48 of 48 cells recorded a harness defect as `ran: false` — a model result.
# That is the operator's standing rule exactly: "tests DERIVED from the interface
# surface (clap tree); hand-coded verb/flag lists are the leak." Deriving it from
# `--help` means the ladder cannot be wrong about a verb again, because the answer
# comes from the binary under test rather than from a table someone maintains.
# Cached: one `--help` per (verb, flag), not one per cell.
declare -A _VERB_FLAG_OK
# <verb-path> may be several words ("serve run"): the flags live on the SUBCOMMAND,
# not on its group. Asking `apr serve --help` reports no `--no-gpu` while
# `apr serve run --help` lists it — querying the group would have re-introduced the
# very defect this helper fixes, one level down. Measured, not assumed.
verb_accepts_flag() { # <verb-path> <flag> -> 0 if that subcommand's --help lists the flag
  local verb="$1" fl="$2" key="$1/$2" out
  [ -z "$fl" ] && return 0
  if [ -z "${_VERB_FLAG_OK[$key]:-}" ]; then
    # shellcheck disable=SC2086
    out=$("$APR" $verb --help 2>&1)
    if grep -qE -- "(^|[[:space:],])${fl}([[:space:],=]|$)" <<< "$out"; then
      _VERB_FLAG_OK[$key]=yes
    else
      _VERB_FLAG_OK[$key]=no
    fi
  fi
  [ "${_VERB_FLAG_OK[$key]}" = yes ]
}
# The flag to pass to <verb>, or empty when that verb does not accept it.
flag_for() { # <verb> <flag>
  if verb_accepts_flag "$1" "$2"; then printf '%s' "$2"; else printf ''; fi
}
lock_timeout() { # lock_timeout <what> -> exit 2, naming the holder from /proc/locks (by inode; lslocks
  local ino pid holder   # leaves PATH empty for a file it cannot resolve, so it cannot be matched by path)
  ino=$(stat -c %i "$GPU_LOCK" 2> /dev/null)
  pid=$(awk -v i=":$ino" '$2 != "->" && substr($6, length($6) - length(i) + 1) == i { print $5; exit }' /proc/locks 2> /dev/null)
  [ -n "$pid" ] && holder="pid $pid: $(ps -o args= -p "$pid" 2> /dev/null | cut -c1-120)"
  echo "decline: ENV the GPU lock $GPU_LOCK was not free after ${LOCK_WAIT}s for $1 -- holder: ${holder:-unknown}. Not a model verdict." >&2
  exit 2
}
if [ "${LOCK_PROBE:-0}" = 1 ]; then
  apr_locked "$@"; rc=$?
  [ "$rc" = "$LOCK_BUSY" ] && lock_timeout "apr $*"
  exit "$rc"
fi

host_id() {
  if [ -n "$HOST_ID" ]; then printf '%s' "$HOST_ID"; return; fi
  local h; h=$(hostname 2>/dev/null || echo unknown)
  case "$h" in
    gx10*) printf 'gx10' ;;
    *[Ll]ambda*) printf 'lambda' ;;
    *) printf '%s' "$h" ;;
  esac
}
HOST=$(host_id)
SHA=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
# apr_sha: the full 40-hex HEAD, which scripts/apr_bin.sh proved the binary was built from. The
# release-readiness shape (#3715) compares it to the release commit by exact equality.
APR_SHA=$(git rev-parse HEAD 2>/dev/null || echo unknown)
VERSION=$(cargo metadata --no-deps --offline --format-version 1 2>/dev/null | python3 -c '
import json, os, sys
m = json.load(sys.stdin)
root = os.path.realpath("Cargo.toml")
for p in m.get("packages", []):
    if os.path.realpath(p["manifest_path"]) == root:
        print(p["version"]); sys.exit(0)
sys.exit(1)' 2>/dev/null) || VERSION=""
[ -n "$VERSION" ] || { echo "decline: root crate version unresolved" >&2; exit 2; }
[ -n "$OUT_DIR" ] || OUT_DIR="evidence/dogfood/models/$VERSION"
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)

# Rungs, one per line: id|file|sha256|backends(csv)|required|hosts(csv)
RUNGS=$(python3 - "$LADDER" <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1]))
for r in d["ladder"]["rungs"]:
    print("|".join([r["id"], r["gguf"], r["sha256"], ",".join(r["backends"]), "1" if r.get("required") else "0", ",".join(r.get("hosts") or [])]))
PY
) || { echo "decline: ladder unreadable" >&2; exit 2; }
[ -n "$RUNGS" ] || { echo "decline: ladder has no rungs" >&2; exit 2; }
LADDER_FILES=$(cut -d'|' -f2 <<< "$RUNGS")   # the files the rungs name; section 2 skips re-measuring them

# The inventory spec (#3712). MODEL_LADDER_INVENTORY_DIRS (colon-separated) is a test seam only.
INV_SPEC=$(python3 - "$LADDER" <<'PY'
import os, sys, yaml
inv = yaml.safe_load(open(sys.argv[1]))["ladder"].get("inventory") or {}
env = os.environ.get("MODEL_LADDER_INVENTORY_DIRS")
dirs = env.split(":") if env else [os.path.expanduser(d) for d in (inv.get("dirs") or [])]
if not dirs or not inv.get("patterns") or not inv.get("backends"):
    sys.exit(1)
print(":".join(dirs)); print(",".join(inv["patterns"])); print(",".join(inv["backends"]))
PY
) || { echo "decline: the ladder declares no inventory (dirs, patterns, backends) -- the universe cannot be measured" >&2; exit 2; }
INV_DIRS=$(sed -n 1p <<< "$INV_SPEC"); INV_PATTERNS=$(sed -n 2p <<< "$INV_SPEC"); INV_BACKENDS=$(sed -n 3p <<< "$INV_SPEC")
# INVENTORY, one per line: file|path. Measured on THIS host, never listed.
INVENTORY=$(python3 - "$INV_DIRS" "$INV_PATTERNS" <<'PY'
import fnmatch, os, sys
dirs, pats = sys.argv[1].split(":"), [p.lower() for p in sys.argv[2].split(",")]
seen = {}
for d in dirs:
    if not os.path.isdir(d):
        continue
    for f in sorted(os.listdir(d)):
        p = os.path.join(d, f)
        if os.path.isfile(p) and f not in seen and any(fnmatch.fnmatch(f.lower(), pat) for pat in pats):
            seen[f] = p
for f, p in seen.items():
    print(f + "|" + p)
PY
) || { echo "decline: the inventory scan failed" >&2; exit 2; }

# ── #3828: ladder_serve_probe — the `serve` verb, over the ROUTER'S OWN ROUTES ──────
# The ladder never started `apr serve`, so no rung ever asked a server to load a model
# (#3571's own words) and /api/chat could not reach the Qwen3.5 hybrid session while
# every gate stayed green (alfredodeza, #3715, 2026-09-22).
#
# The route set is DERIVED from the string literals registered in
# crates/aprender-serve/src/api/router.rs, never hand-listed here: a hand-listed set is
# precisely how /api/chat went unprobed, and a constant in this file would drift from
# the router the moment anyone adds an endpoint. Each generation route is probed
# non-streaming AND streaming, because the ollama-compat wire has its own translation
# layer that has already diverged from the OpenAI-compat one twice independently
# (#3825's tool_calls gap, and this defect).
#
# Prints ONE json object; returns 0 if every probed route answered, 1 otherwise. A route
# that could not be probed is recorded with its reason and is NOT counted as a pass.

# ── teardown: kill the SERVER, not the WRAPPER ───────────────────────────────
# `apr_locked serve run … &` makes $! the flock/choom WRAPPER, not `apr serve`.
# Killing the wrapper does NOT take the server down. Measured 2026-09-22 on both
# hosts, rung 1 of 8: the server survived, kept its port, was re-parented away
# (`ps -o ppid=` showed it adopted by pid 1's subreaper), and the script blocked
# in `wait` FOREVER — after the route loop had already finished, with the server
# idle at 0 jiffies/5s and no curl in flight.
#
# Every bound this function states was holding at the time: the 90s health wait
# and all six `curl --max-time 60`. The unbounded step was the one nobody wrote a
# bound for, because it was assumed instantaneous. That is the same shape as the
# two ollama gates fixed in paiml/infra#911 the same day: the bound, like the
# check, was on the wrong process.
#
# Resolve the listener by PORT. The port is drawn at random per probe, so it can
# never name another session's process — which is what makes escalating to a kill
# safe on a box several agents share.
ladder_serve_teardown() { # <wrapper-pid> <port> -> prints clean|escalated|failed
    local pid="$1" port="$2" i spid
    kill "$pid" 2>/dev/null
    for i in $(seq 1 10); do
        if ! curl -fsS --max-time 1 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
            wait "$pid" 2>/dev/null
            printf 'clean'; return 0
        fi
        sleep 1
    done
    # Still answering: the wrapper died and the server did not. Name it by port,
    # then PROVE it is ours before signalling. The random port makes a collision
    # unlikely; `/proc/PID/cwd` makes it checkable, and on a box several agents
    # share, "unlikely" is an argument while the cwd is evidence (cop, #3828).
    # A pid we cannot attribute to this tree is never signalled — it is reported.
    spid=$(ss -tlnpH "sport = :$port" 2>/dev/null | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
    if [ -n "$spid" ] && [ "$(readlink -f "/proc/$spid/cwd" 2>/dev/null)" != "$PWD" ]; then
        printf 'failed'; return 1
    fi
    if [ -n "$spid" ]; then
        kill "$spid" 2>/dev/null
        for i in $(seq 1 10); do
            if ! kill -0 "$spid" 2>/dev/null; then
                wait "$pid" 2>/dev/null
                printf 'escalated'; return 0
            fi
            sleep 1
        done
        kill -9 "$spid" 2>/dev/null
        wait "$pid" 2>/dev/null
        printf 'escalated'; return 0
    fi
    # Could not resolve a listener and /health still answers: do NOT `wait`, or we
    # reproduce the hang this function exists to end. Report it as cell state.
    printf 'failed'; return 1
}

ladder_serve_probe() { # ladder_serve_probe <model> <backend-flag> <rung-id> <backend>
    local path="$1" flag="$2" rid="$3" bname="$4" td
    # The script cd's to the repo root at startup (line ~53), so this is relative by
    # design — there is no WS_ROOT in this script and inventing one would resolve to
    # "/crates/..." under `set -u`-less expansion and silently find nothing.
    local router="crates/aprender-serve/src/api/router.rs"
    local routes port pid rc=0 out first=1 json="{" waited=0 code body

    if [ ! -f "$router" ]; then
        printf '{"probed":false,"why":"router source not found at %s — the route set is derived from it, and a hand-listed set is what let /api/chat go unprobed","routes":{}}' "$router"
        return 1
    fi
    # Generation routes only: the ones that take a prompt and return a completion.
    routes=$(grep -oE '"/(v1|api)/[a-z0-9/._-]+"' "$router" \
        | tr -d '"' | sort -u \
        | grep -E '^/(v1/(chat/)?completions|api/chat)$' || true)
    if [ -z "$routes" ]; then
        printf '{"probed":false,"why":"no generation route matched in %s — the derivation found nothing, which is a defect in the derivation, not an empty surface","routes":{}}' "$router"
        return 1
    fi

    port=$(( 20000 + (RANDOM % 20000) ))
    apr_locked serve run "$path" --port "$port" $flag > "$WORK/serve-$rid-$bname.log" 2>&1 &
    pid=$!
    # Health, bounded. A server that never comes up is a FAIL naming the wait, never a skip.
    while [ "$waited" -lt 90 ]; do
        curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break
        kill -0 "$pid" 2>/dev/null || break
        sleep 1; waited=$((waited + 1))
    done
    if ! curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
        td=$(ladder_serve_teardown "$pid" "$port")
        printf '{"probed":false,"why":"apr serve did not answer /health within %ss (log: %s)","teardown":"%s","routes":{}}' \
            "$waited" "$WORK/serve-$rid-$bname.log" "$td"
        return 1
    fi

    for r in $routes; do
        for stream in false true; do
            case "$r" in
                /api/chat) body='{"model":"apr","messages":[{"role":"user","content":"What is 2+2?"}],"stream":'"$stream"'}' ;;
                */chat/completions) body='{"model":"apr","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":16,"stream":'"$stream"'}' ;;
                *) body='{"model":"apr","prompt":"What is 2+2?","max_tokens":16,"stream":'"$stream"'}' ;;
            esac
            code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 60 \
                -H 'Content-Type: application/json' -d "$body" \
                "http://127.0.0.1:$port$r" 2>/dev/null) || code=000
            [ "$code" = 200 ] || rc=1
            [ $first = 1 ] || json="$json,"; first=0
            json="$json\"$r|stream=$stream\":{\"http\":$code,\"ok\":$([ "$code" = 200 ] && echo true || echo false)}"
        done
    done
    # A server that will not die is a real property of the `serve` verb, and until
    # now it was invisible: the script simply stopped. Record it in the cell.
    td=$(ladder_serve_teardown "$pid" "$port")
    [ "$td" = failed ] && rc=1
    printf '{"probed":true,"teardown":"%s","routes":{%s}}' "$td" "${json#\{}"
    return $rc
}

find_model() { # find_model <basename> — the inventory's copy first, then the fleet's other model dirs
  local f="$1" d p
  p=$(awk -F'|' -v f="$f" '$1 == f { print $2; exit }' <<< "$INVENTORY")
  [ -n "$p" ] && [ -f "$p" ] && { printf '%s' "$p"; return 0; }
  for d in "${APR_MODELS_DIR:-}" "$HOME/models" "$HOME/.apr/models" "$HOME/.cache/apr/models"; do
    [ -n "$d" ] && [ -f "$d/$f" ] && { printf '%s' "$d/$f"; return 0; }
  done
  return 1
}

WORK=$(mktemp -d)
# The delete is guarded (SEC011): only a path under a temp root is removed.
_rm_work() {
  local v="${WORK:-}"
  case "$v" in
    /tmp/?*|/var/folders/?*|/mnt/?*) if [ -n "$v" ] && [ "$v" != "/" ]; then rm -rf -- "$v" || :; fi ;;
    *) return 0 ;;
  esac
}
trap _rm_work EXIT
ROWS="$WORK/rows.jsonl"; : > "$ROWS"
INV_ROWS="$WORK/inventory.jsonl"; : > "$INV_ROWS"
EXECUTED=0; RED=0
printf -- '--- model capability ladder on %s (%s, cc %s) apr=%s sha=%s version=%s ---\n' \
  "$HOST" "${GPU_NAME:-no-gpu}" "${GPU_CC:-?}" "$APR" "$SHA" "$VERSION"
printf '    inventory: %s model(s) matching %s under %s\n' "$(grep -c . <<< "$INVENTORY")" "$INV_PATTERNS" "$INV_DIRS"

# measure <id> <file> <path> <sha> <backends(csv)> <required 0|1> <inventory_only 0|1>
# The per-model checks, one function so a ladder rung and an inventory model cannot drift:
#   1. apr qa --json: capability_match + golden_output (the two gates that name a wrong model)
#   2. per claimed backend: `apr run` rc 0 and no fallback line (did it stay on that backend?)
measure() {
  local rid=$1 rfile=$2 path=$3 got=$4 rbackends=$5 rreq=$6 rinv=$7
  local qa_json="$WORK/${rid//[^A-Za-z0-9._-]/_}.qa.json" cap_flag="" qa_rc qa_row be_json first b flag run_out run_rc fb ran row why
  # A rung that claims only the CPU is not asked whether the GPU can run it: capability_match is a
  # GPU-capability gate. The judge accepts a SKIPPED capability_match only when cuda is not claimed,
  # and check_model_ladder.sh refuses any Q4_K rung that does not claim cuda (#3712).
  case ",$rbackends," in *,cuda,*|*,gpu,*) ;; *) cap_flag="--skip-capability" ;; esac
  # shellcheck disable=SC2086
  apr_locked qa "$path" --json --offline --skip-throughput --skip-ollama --skip-gpu-speedup \
      --skip-ptx-parity --skip-gpu-state --skip-format-parity $cap_flag > "$qa_json" 2> "$qa_json.err"; qa_rc=$?
  [ "$qa_rc" = "$LOCK_BUSY" ] && lock_timeout "apr qa $rid"
  qa_row=$(python3 - "$qa_json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print(json.dumps({"parse_error": str(e)})); sys.exit(0)
g = {x["name"]: x for x in d.get("gates", [])}
def gate(n):
    x = g.get(n)
    if x is None: return {"passed": False, "skipped": True, "message": "gate absent from report"}
    # `apr qa --json` marks a skipped gate passed:true. Skipped is not passed.
    return {"passed": bool(x.get("passed")) and not x.get("skipped") and not str(x.get("message","")).startswith("Skipped"),
            "skipped": bool(x.get("skipped")) or str(x.get("message","")).startswith("Skipped"),
            "message": str(x.get("message",""))[:200]}
print(json.dumps({"capability_match": gate("capability_match"), "golden_output": gate("golden_output")}))
PY
)
  be_json="{"; first=1
  IFS=',' read -r -a bes <<< "$rbackends"
  # ── #3843: `code` is measured ONCE per rung, and NOT per backend ─────────────
  # `apr code` spawns its OWN `apr serve` ("Launched apr serve on port N (pid N)"),
  # so the backend is chosen by the server it starts, not by its caller. There is
  # no backend selector on `apr code` at all. Passing this loop's `--no-gpu`/`--gpu`
  # to it produced `error: unexpected argument '--no-gpu' found`, clap exit 2, on
  # 48 of 48 cells across both hosts — a HARNESS defect recorded as `ran: false`,
  # i.e. as a model result. Running it per backend would also record one
  # measurement twice under two labels.
  #
  # The cell still appears under each backend, because the judge requires all four
  # verbs there, but it carries `backend: "inherited-from-spawned-serve"` so the
  # receipt states what was actually measured instead of implying a backend.
  code_out=$(apr_locked code -p "Reply with the single word: ok" --model "$path" \
      --output-format json 2>&1); code_rc=$?
  [ "$code_rc" = "$LOCK_BUSY" ] && lock_timeout "apr code $rid"
  code_ran=true; [ $code_rc -eq 0 ] || code_ran=false

  for b in "${bes[@]}"; do
    case "$b" in cpu) flag="--no-gpu" ;; cuda|gpu) flag="--gpu" ;; *) flag="" ;; esac
    # --verbose prints realizar's `[DEBUG] formatted_prompt=…`, which the #3743 check reads.
    # apr_locked / LOCK_BUSY are KEPT from main: the GPU lock is what serialises the
    # ladder against every other session on the box. #3743's side dropped it.
    run_flag=$(flag_for run "$flag")
    # shellcheck disable=SC2086
    run_out=$(apr_locked run "$path" --prompt "What is the capital of France? Answer briefly." --max-tokens 16 --verbose $run_flag 2>&1); run_rc=$?
    [ "$run_rc" = "$LOCK_BUSY" ] && lock_timeout "apr run $rid ($b)"
    fb=false; ran=true; esc=false
    if grep -qE 'falling back to CPU|path rejected, attempting fallback|runs on the CPU; the GPU backend' <<< "$run_out"; then fb=true; fi
    # #3743: realizar zero-width-escapes special tokens it finds INSIDE the user text,
    # so an escaped special in the formatted prompt means the prompt reached the model
    # templated twice (0.69.0: `--prompt` on an instruct-named model). The debug line
    # prints it Debug-escaped (`\u{200b}`); the raw character is checked too.
    if grep -F 'formatted_prompt=' <<< "$run_out" | grep -qF -e '\u{200b}' -e $'\u200b'; then esc=true; fi
    if [ "$b" != cpu ] && [ -z "$GPU_NAME" ]; then ran=false; fi
    [ $run_rc -eq 0 ] || ran=false

    # ── #3828: the OTHER THREE VERBS ────────────────────────────────────────────
    # This loop ran `apr run` and nothing else, so `serve`, `chat` and `code` cells in
    # release-readiness-v1 (#3715) were computed over a verb set of ONE. A live defect
    # walked straight through: /api/chat could not reach the Qwen3.5 hybrid session at
    # all (alfredodeza, #3715 comment 2026-09-22) while every gate we own stayed green,
    # because no rung had ever asked `apr serve` to load a model (#3571's own words).
    # A verb that is not probed must never read as a verb that passed, so each of these
    # records its own rc and the judge refuses a receipt missing any of them.

    # chat: stdin-driven, one turn, machine envelope. `/exit` closes the session.
    chat_flag=$(flag_for chat "$flag")
    # shellcheck disable=SC2086
    chat_out=$(printf 'What is the capital of France? Answer briefly.\n/exit\n' \
        | apr_locked chat "$path" --json --max-tokens 16 $chat_flag 2>&1); chat_rc=$?
    [ "$chat_rc" = "$LOCK_BUSY" ] && lock_timeout "apr chat $rid ($b)"
    chat_ran=true; [ $chat_rc -eq 0 ] || chat_ran=false

    # code: measured once per rung, above this loop — see #3843.

    # serve: the route set is DERIVED from the router's registered paths, never a
    # hand-listed constant (alfredodeza's amendment, adopted on #3715) — a route that
    # exists but is not probed is exactly how /api/chat stayed broken. Each route is
    # probed non-streaming and streaming, because the ollama-compat wire has its own
    # translation layer that has already diverged from the OpenAI-compat one twice
    # independently (#3825's tool_calls gap, and this defect).
    serve_json=$(ladder_serve_probe "$path" "$(flag_for "serve run" "$flag")" "$rid" "$b")
    serve_rc=$?

    [ $first = 1 ] || be_json="$be_json,"; first=0
    be_json="$be_json\"$b\":{\"ran\":$ran,\"fallback\":$fb,\"escaped_special\":$esc,\"rc\":$run_rc"
    be_json="$be_json,\"verbs\":{\"run\":{\"ran\":$ran,\"rc\":$run_rc}"
    be_json="$be_json,\"chat\":{\"ran\":$chat_ran,\"rc\":$chat_rc}"
    be_json="$be_json,\"code\":{\"ran\":$code_ran,\"rc\":$code_rc,\"backend\":\"inherited-from-spawned-serve\"}"
    be_json="$be_json,\"serve\":$serve_json}}"
  done
  be_json="$be_json}"
  # The receipt carries the MEASURED file hash (ONT-4c1): a resolver joining the ladder contract to
  # this receipt compares two measurements instead of trusting the receipt's own claim that it checked.
  row=$(python3 - "$rid" "$qa_row" "$be_json" "$qa_rc" "$rreq" "$got" "$rfile" "$rinv" <<'PY'
import json, sys
rid, qa, be, qa_rc = sys.argv[1], json.loads(sys.argv[2]), json.loads(sys.argv[3]), int(sys.argv[4])
req, sha, rfile, inv_only = sys.argv[5] == "1", sys.argv[6], sys.argv[7], sys.argv[8] == "1"
cap = qa.get("capability_match", {})
# `passed` is already normalised (skipped => passed=False) by the gate() reader above, but the
# judge must not depend on that: a skipped gate counts only when no GPU backend is claimed.
cap_ok = (cap.get("passed", False) and not cap.get("skipped", False)) \
         or (cap.get("skipped", False) and not ({"cuda", "gpu"} & set(be)))
green = cap_ok and qa.get("golden_output", {}).get("passed", False) \
        and all(v["ran"] and not v["fallback"] and not v.get("escaped_special") for v in be.values())
print(json.dumps({"id": rid, "file": rfile, "inventory_only": inv_only, "present": True, "sha_ok": True,
                  "sha256": sha, "required": req, "qa_rc": qa_rc, "capability_match": qa.get("capability_match"),
                  "golden_output": qa.get("golden_output"), "backends": be, "green": green}))
PY
)
  # ── #3842: NEVER append an empty line ────────────────────────────────────────
  # If the row builder above produced nothing — because `apr qa --json` wrote zero
  # bytes, or the builder itself threw — this used to append an EMPTY LINE. The
  # receipt assembler then dropped the empty line, the row vanished, and RED went
  # on being counted separately: gx10's 0.69.1 receipt said `red: 3` while its rows
  # held two, with the failed REQUIRED rung `qwen35-27b-q4km` absent from its own
  # receipt. A red that hides is worse than a red that fails, because no per-row
  # judging can find a row that was never written.
  #
  # So an unbuildable row becomes an explicit RED row that NAMES why, and carries
  # the qa byte count so "apr qa wrote nothing" is distinguishable from "the row
  # builder crashed".
  if [ -z "$row" ] || ! python3 -c 'import json,sys; json.load(sys.stdin)' <<< "$row" 2>/dev/null; then
    qa_bytes=$(stat -c %s "$qa_json" 2>/dev/null || echo 0)
    row=$(python3 - "$rid" "$rfile" "$got" "$rreq" "$rinv" "$qa_rc" "$qa_bytes" <<'PY'
import json, sys
rid, rfile, sha, req, inv, qa_rc, qa_bytes = sys.argv[1:8]
why = (f"apr qa --json wrote 0 bytes (exit {qa_rc}) — no document to judge"
       if qa_bytes == "0" else
       f"the receipt row could not be built from a {qa_bytes}-byte qa document, exit {qa_rc} from apr qa")
print(json.dumps({
    "id": rid, "file": rfile, "inventory_only": inv == "1", "present": True,
    "sha_ok": True, "sha256": sha, "required": req == "1", "qa_rc": int(qa_rc),
    "capability_match": {"passed": False, "skipped": False, "message": why},
    "golden_output": {"passed": False, "skipped": False, "message": why},
    "backends": {}, "green": False, "row_synthesized": True, "why": why}))
PY
)
  fi
  printf '%s\n' "$row" >> "$ROWS"
  EXECUTED=$((EXECUTED + 1))
  if grep -q '"green": true' <<< "$row"; then
    printf '  [ OK   ] %-30s qa cap+golden pass, backends %s honoured\n' "$rid" "$rbackends"
  else
    RED=$((RED + 1))
    why=$(printf '%s' "$row" | python3 -c '
import json,sys; r=json.load(sys.stdin); w=[]
cap=r["capability_match"] or {}; claims_gpu=bool({"cuda","gpu"} & set(r["backends"]))
if not (cap.get("passed") and not cap.get("skipped")) and not (cap.get("skipped") and not claims_gpu): w.append("capability_match: "+("SKIPPED on a GPU-claiming model: " if cap.get("skipped") else "")+str(cap.get("message",""))[:70])
gold=r["golden_output"] or {}
if not gold.get("passed"): w.append("golden_output: "+("SKIPPED: " if gold.get("skipped") else "")+str(gold.get("message",""))[:70])
for b,v in r["backends"].items():
    if v["fallback"]: w.append(b+": FELL BACK (claimed backend did not run)")
    elif v.get("escaped_special"): w.append(b+": escaped special token in the formatted prompt: templated twice (#3743)")
    elif not v["ran"]: w.append(b+": did not run (rc=%s)"%v["rc"])
print("; ".join(w) or "unknown")')
    printf '  [FAIL  ] %-30s %s\n' "$rid" "$why"
  fi
}

# ---- 1. the ladder's rungs
while IFS='|' read -r -t 5 rid rfile rsha rbackends rreq rhosts; do
  [ -n "$rid" ] || continue
  # A rung that lists hosts: is a claim only on those hosts (a 122B file fits gx10's unified memory and no
  # 24 GB card). Elsewhere it is neither absent nor passed: recorded as not listed, never counted RED.
  # An inventory model is never "not listed": if this host holds it, section 2 measures it.
  if [ -n "$rhosts" ] && ! grep -qE "(^|,)${HOST}(,|$)" <<< "$rhosts"; then
    printf '  [N/A   ] %-30s not listed for %s (hosts: %s)\n' "$rid" "$HOST" "$rhosts"
    printf '{"id":"%s","file":"%s","present":false,"required":false,"not_listed_host":true}\n' "$rid" "$rfile" >> "$ROWS"
    continue
  fi
  path=$(find_model "$rfile") || {
    printf '  [ABSENT] %-30s %s not in any model dir\n' "$rid" "$rfile"
    printf '{"id":"%s","file":"%s","present":false,"required":%s}\n' "$rid" "$rfile" "$([ "$rreq" = 1 ] && echo true || echo false)" >> "$ROWS"
    [ "$rreq" = 1 ] && RED=$((RED + 1))
    continue
  }
  got=$(sha256sum "$path" | cut -d' ' -f1)
  if [ "$got" != "$rsha" ]; then
    printf '  [FAIL  ] %-30s sha256 mismatch (%s… vs ladder %s…) — a different file is a different measurement\n' "$rid" "${got:0:12}" "${rsha:0:12}"
    printf '{"id":"%s","file":"%s","present":true,"sha_ok":false,"sha256":"%s","required":%s}\n' "$rid" "$rfile" "$got" "$([ "$rreq" = 1 ] && echo true || echo false)" >> "$ROWS"
    RED=$((RED + 1)); continue
  fi
  if [ "$DRY" = 1 ]; then printf '  [DRY   ] %-30s %s\n' "$rid" "$path"; continue; fi
  measure "$rid" "$rfile" "$path" "$got" "$rbackends" "$rreq" 0
done <<EOF2
$RUNGS
EOF2

# ---- 2. every inventory model: recorded in the receipt, and measured unless a rung already did
while IFS='|' read -r -t 5 ifile ipath; do
  [ -n "$ifile" ] || continue
  isha=$(sha256sum "$ipath" | cut -d' ' -f1); ibytes=$(stat -c %s "$ipath" 2>/dev/null || echo 0)
  printf '{"file":"%s","sha256":"%s","bytes":%s}\n' "$ifile" "$isha" "$ibytes" >> "$INV_ROWS"
  if grep -qxF -- "$ifile" <<< "$LADDER_FILES"; then continue; fi   # a rung measured it above
  if [ "$DRY" = 1 ]; then printf '  [DRY   ] %-30s %s (inventory, not a rung)\n' "inv:$ifile" "$ipath"; continue; fi
  measure "inv:$ifile" "$ifile" "$ipath" "$isha" "$INV_BACKENDS" 1 1
done <<EOF3
$INVENTORY
EOF3

[ "$DRY" = 1 ] && exit 0
if [ "$EXECUTED" -eq 0 ]; then
  echo "decline: nothing executed on $HOST — nothing was measured, no receipt written" >&2
  exit 2
fi
mkdir -p "$OUT_DIR"
# The receipt names the binary by what it SAYS it is (`apr --version`, which
# carries the built-from sha), never by its path: a path is machine-specific
# (check_no_shipped_machine_paths) and says nothing about what was run.
APR_VERSION=$("$APR" --version 2>/dev/null | head -1)
python3 - "$ROWS" "$OUT_DIR/$HOST.json" "$HOST" "$VERSION" "$SHA" "${GPU_NAME:-}" "${GPU_CC:-}" "$EXECUTED" "$RED" "$APR_VERSION" "$INV_ROWS" "$INV_DIRS" "$INV_PATTERNS" "$APR_SHA" <<'PY'
import json, sys, datetime, platform
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
inv = [json.loads(l) for l in open(sys.argv[11]) if l.strip()]
out = {"schema": "apr-model-ladder-receipt/v2", "host": sys.argv[3], "version": sys.argv[4], "sha": sys.argv[5], "apr_sha": sys.argv[14],
       "isa": platform.machine(), "gpu": sys.argv[6] or None, "cc": sys.argv[7] or None,
       "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
       "apr_version": sys.argv[10], "executed": int(sys.argv[8]), "red": int(sys.argv[9]),
       "inventory": inv, "inventory_dirs": sys.argv[12].split(":"), "inventory_patterns": sys.argv[13].split(","),
       "rungs": rows}
json.dump(out, open(sys.argv[2], "w"), indent=2); open(sys.argv[2], "a").write("\n")
PY
printf 'receipt: %s/%s.json (executed=%s red=%s inventory=%s)\n' "$OUT_DIR" "$HOST" "$EXECUTED" "$RED" "$(grep -c . "$INV_ROWS")"
[ "$RED" -eq 0 ] && exit 0
exit 1
