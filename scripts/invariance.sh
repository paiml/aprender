#!/usr/bin/env bash
# invariance.sh — invoke ONE verb across EVERY declared transport AT THE SAME TIME, compare.
# (#4377: ported from invariance.py; no step of the dogfood gate path may need Python.)
#
# Why this is a separate gate from interface-parity
# -------------------------------------------------
# interface-parity proves each transport has an e2e that spawns the binary and
# passes. That is reachability, and it is necessary. It is not sufficient: three
# transports can each be reachable, each be green in its own test file, and still
# disagree about what a verb RETURNS. Nothing compares them, because each e2e only
# ever sees its own surface.
#
# Why SIMULTANEOUS rather than sequential
# ---------------------------------------
# Running them one after another cannot distinguish "the transports agree" from
# "the transports share a process-global that only one of them may hold at a
# time". Standing every transport up at once and invoking through all of them
# while all are live is the configuration a real client fleet produces, and it is
# the one that surfaces a shared listener, a shared lock, or a runtime that only
# tolerates a single owner.
#
# Why DERIVED rather than hand-written
# ------------------------------------
# The verb list comes from the BINARY, not from a list in this file. A hand-written
# probe tests the verbs someone remembered; a derived one tests the surface that
# shipped, and grows automatically when the surface does.
#
# Usage:  invariance.sh <binpath> <declaration-json>
#   declaration: {list, probe:{verb, params}, cli, http:{serve ("{port}"), path ("{verb}")}}
# First output line, and exit status:
#   INVARIANCE_PASS …   0     INVARIANCE_SKIP …   0  (fewer than 2 transports invocable)
#   INVARIANCE_FAIL …   1     INVARIANCE_ERROR …  2  (the check itself could not run)
# Needs: jq, curl, timeout.
set -euo pipefail

PIDS=()
cleanup() {
  local p
  for p in "${PIDS[@]+"${PIDS[@]}"}"; do
    kill -9 "$p" 2>/dev/null || true
    wait "$p" 2>/dev/null || true
  done
}
trap cleanup EXIT

die() {
  printf 'INVARIANCE_ERROR %s\n' "$1"
  exit "${2:-2}"
}

# strip VAR-VALUE -> the value without leading and trailing whitespace (Python's str.strip()).
strip() {
  local s=$1
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

# A short, quoted rendering of an arbitrary string for a one-line message.
quoted() {
  printf '%s' "${1:0:$2}" | jq -Rs .
}

for tool in jq curl timeout; do
  command -v "$tool" >/dev/null 2>&1 || die "\`$tool\` is not installed; the check cannot run"
done
[ "$#" -eq 2 ] || die "usage: invariance.sh <binpath> <declaration-json>"
BINPATH=$1
DECL=$2

# ── the declaration ─────────────────────────────────────────────────────────
printf '%s' "$DECL" | jq -e 'type == "object"' >/dev/null 2>&1 \
  || die "the declaration is not a JSON object"
field() { printf '%s' "$DECL" | jq -r "$1"; }
LIST_CMD=$(field 'if (.list | type) == "string" then .list else "" end')
VERB=$(field 'if (.probe | type) == "object" and (.probe.verb | type) == "string" then .probe.verb else "" end')
[ -n "$(strip "$LIST_CMD")" ] && [ -n "$VERB" ] \
  || die "declaration needs \`list\` and \`probe = { verb, params }\`"
PARAMS=$(field 'if .probe.params == null then "{}" elif (.probe.params | type) == "string" then .probe.params else error("params must be a string") end' 2>/dev/null) \
  || die "probe.params must be a JSON string"
CLI_TMPL=$(field 'if (.cli | type) == "string" then .cli else "" end')
HTTP_SERVE=$(field 'if (.http | type) == "object" then (.http.serve // "" | tostring) else "" end')
HTTP_PATH=$(field 'if (.http | type) == "object" then (.http.path // "" | tostring) else "" end')
HAS_HTTP=$(field 'if (.http | type) == "object" and (.http | length) > 0 then "yes" else "" end')

# ── the surface, DERIVED from the binary; an empty list dies ────────────────
read -ra LIST_ARGV <<< "$LIST_CMD"
WORK=$(mktemp -d)
PIDS_WORK=$WORK
trap 'cleanup; rm -rf "${PIDS_WORK:?}"' EXIT
rc=0
timeout 60 "$BINPATH" "${LIST_ARGV[@]}" > "$WORK/list.out" 2> "$WORK/list.err" || rc=$?
if [ "$rc" -ne 0 ]; then
  die "\`$LIST_CMD\` exited $rc: $(strip "$(cat "$WORK/list.err")" | head -c 200)"
fi
VERBS=()
while IFS= read -r line || [ -n "$line" ]; do
  line=$(strip "$line")
  if [ -n "$line" ]; then VERBS+=("$line"); fi
done < "$WORK/list.out"
[ "${#VERBS[@]}" -gt 0 ] \
  || die "the binary lists NO verbs — a parity check over an empty surface is vacuous"
found=""
for v in "${VERBS[@]}"; do
  if [ "$v" = "$VERB" ]; then found=yes; fi
done
if [ -z "$found" ]; then
  first6=$(printf '%s\n' "${VERBS[@]:0:6}" | paste -sd, - | sed 's/,/, /g')
  die "probe verb \`$VERB\` is not in the binary's own list ($first6…)"
fi

# ── stand every transport up FIRST, so all are live together ────────────────
# A port is free when a connect to it is REFUSED (curl exit 7). Probe by
# connecting, never by binding: a probe that binds competes with the server it
# is waiting for and can starve it.
connect_rc() {
  local r=0
  curl -s -o /dev/null --connect-timeout 0.5 --max-time 1 "http://127.0.0.1:$1/" || r=$?
  printf '%s' "$r"
}
free_port() {
  local i p
  for i in $(seq 1 50); do
    p=$(( 20000 + (RANDOM * 32768 + RANDOM) % 40000 ))
    if [ "$(connect_rc "$p")" = 7 ]; then printf "%s" "$p"; return 0; fi
  done
  return 1
}
wait_ready() {
  local end=$((SECONDS + 25)) r
  while [ "$SECONDS" -lt "$end" ]; do
    r=$(connect_rc "$1")
    # 7 refused, 28 no connect within the timeout: not ready. Anything else connected.
    if [ "$r" != 7 ] && [ "$r" != 28 ]; then return 0; fi
    sleep 0.05
  done
  return 1
}

NAMES=()
RESULTS=()
PORT=""
if [ -n "$HAS_HTTP" ]; then
  PORT=$(free_port) || die "no free local port found for the http transport"
  read -ra SERVE_ARGV <<< "${HTTP_SERVE//\{port\}/$PORT}"
  "$BINPATH" "${SERVE_ARGV[@]}" >/dev/null 2>&1 &
  PIDS+=("$!")
  wait_ready "$PORT" || die "http transport did not accept a connection on $PORT"
fi

# ── invoke through each while all are still up ──────────────────────────────
if [ -n "$CLI_TMPL" ]; then
  # Split the TEMPLATE, then substitute per token. Splitting after substitution
  # would tear a JSON params value containing spaces into several argv entries,
  # and the verb would be invoked with garbage.
  read -ra TMPL <<< "$CLI_TMPL"
  CLI_ARGV=()
  for t in "${TMPL[@]+"${TMPL[@]}"}"; do
    if [ "$t" = "{params}" ]; then CLI_ARGV+=("$PARAMS"); else CLI_ARGV+=("${t//\{verb\}/$VERB}"); fi
  done
  rc=0
  timeout 120 "$BINPATH" "${CLI_ARGV[@]+"${CLI_ARGV[@]}"}" > "$WORK/cli.out" 2> "$WORK/cli.err" || rc=$?
  [ "$rc" -eq 0 ] || die "cli invoke failed ($rc): $(strip "$(cat "$WORK/cli.err")" | head -c 200)"
  NAMES+=(cli)
  RESULTS+=("$(strip "$(cat "$WORK/cli.out")")")
fi
if [ -n "$HAS_HTTP" ]; then
  rc=0
  code=$(curl -sS -o "$WORK/http.out" -w '%{http_code}' --max-time 120 \
    -H 'Content-Type: application/json' --data-binary "$PARAMS" \
    "http://127.0.0.1:$PORT${HTTP_PATH//\{verb\}/$VERB}" 2> "$WORK/http.err") || rc=$?
  [ "$rc" -eq 0 ] || die "http invoke could not complete (curl exit $rc): $(head -c 200 "$WORK/http.err")"
  [ "$code" -lt 400 ] || die "http invoke returned $code: $(quoted "$(cat "$WORK/http.out")" 200)"
  NAMES+=(http)
  RESULTS+=("$(strip "$(cat "$WORK/http.out")")")
fi

# ── identical AND valid: two identically-wrong strings must not pass ────────
if [ "${#NAMES[@]}" -lt 2 ]; then
  printf 'INVARIANCE_SKIP only %s transport(s) invocable: [%s]\n' "${#NAMES[@]}" "${NAMES[*]+"${NAMES[*]}"}"
  exit 0
fi
FIRST=${RESULTS[0]}
for i in $(seq 1 $((${#NAMES[@]} - 1))); do
  if [ "${RESULTS[$i]}" != "$FIRST" ]; then
    printf 'INVARIANCE_FAIL `%s` differs between %s and %s\n' "$VERB" "${NAMES[0]}" "${NAMES[$i]}"
    printf '  %s: %s\n' "${NAMES[0]}" "$(quoted "$FIRST" 300)"
    printf '  %s: %s\n' "${NAMES[$i]}" "$(quoted "${RESULTS[$i]}" 300)"
    exit 1
  fi
done
# Exactly one JSON value: jq alone would accept `1 2` or an empty payload.
if [ "$(printf '%s' "$FIRST" | jq -s 'length' 2>/dev/null)" != 1 ]; then
  printf 'INVARIANCE_FAIL all transports agree but the payload is not JSON: %s\n' "$(quoted "$FIRST" 200)"
  exit 1
fi
printf 'INVARIANCE_PASS %s verb(s) derived from the binary; `%s` byte-identical across %s with all transports live\n' \
  "${#VERBS[@]}" "$VERB" "$(printf '%s\n' "${NAMES[@]}" | paste -sd, - | sed 's/,/, /g')"
