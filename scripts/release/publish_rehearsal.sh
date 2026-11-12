#!/usr/bin/env bash
# publish_rehearsal.sh -- push ONE throwaway crate through the TRUE publish path (aprender PMAT-6914). Called by a nightly workflow.
#   publish_rehearsal.sh <run-number> [dry_run 0|1]
# 0.72.5 went red three times on release day at steps no nightly had ever run (the crates.io auth exchange, a gh read of another
# repo, a job log). release-rehearsal-nightly.yml runs everything except the upload and holds no credential. This is the missing leg:
#   a  argument      run-number ^[1-9][0-9]*$, the version is 0.0.<run-number>
#   b  clean-room    the SAME clean_room_gate ROOT TAG release day asks (scripts/cascade-publish.sh), for the newest vX.Y.Z tag
#   c  upload        ci/publish-rehearsal copied to scratch, version set THERE, cargo publish with CARGO_REGISTRY_TOKEN from the environment
#   d  read-back     the version shows on the crates.io sparse index (bounded retry)
# Dry run does a and b for real, prints WOULD publish, uploads nothing. The tree is never written: the version change and the lock
# file live in a scratch copy under ${RUNNER_TEMP:-${TMPDIR:-/tmp}}. The token is read by cargo from its environment and nowhere else:
# never printed, never an argument, never handed to the clean-room gate, and scrubbed from every line of cargo output shown here.
# Fail-closed: every step is a STOP naming the step; "already uploaded" for exactly this version is success (a rerun resumes).
# Seams for the case table: PUB_REH_{GATE,PUBLISH,INDEX,CASCADE,TRIES,WAIT}.
set -uo pipefail
say() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }  # bashrs disable-line=DET002
die() { say "STOP $*"; exit 1; }
CRATE=aprender-publish-rehearsal
TMP=""   # global: the EXIT trap reads it after main returns

# run_gate ROOT TAG: loads the five clean-room functions from cascade-publish.sh (without running the file: its top level
# publishes) into a subshell and asks clean_room_gate. One copy of the question; the guard check_cascade_clean_room_gate.sh loads it the same way.
run_gate() {
  local cascade="${PUB_REH_CASCADE:-$1/scripts/cascade-publish.sh}" fns="$TMP/gate-fns.sh" fn
  if [ -n "${PUB_REH_GATE:-}" ]; then $PUB_REH_GATE "$1" "$2"; return; fi
  [ -r "$cascade" ] || { echo "cannot read $cascade"; return 1; }
  : > "$fns"
  for fn in clean_room_parse_runs clean_room_parse_job clean_room_tested_abbrevs clean_room_tested_shas clean_room_gate; do
    sed -n "/^${fn}() {/,/^}/p" "$cascade" >> "$fns"
    grep -q "^${fn}() {" "$fns" || { echo "cascade-publish.sh does not define $fn"; return 1; }
  done
  (
    set +u   # the cascade's functions were written without -u
    . "$fns" || exit 1
    clean_room_gate "$1" "$2"
  )
}

# publish_cmd: run in the scratch copy. A crate with no dependencies needs no network to lock; cargo publish --locked then proves the
# lock it publishes with is the one just generated (nothing resolved behind our back). cargo reads CARGO_REGISTRY_TOKEN itself.
publish_cmd() {
  if [ -n "${PUB_REH_PUBLISH:-}" ]; then $PUB_REH_PUBLISH; return; fi
  cargo generate-lockfile --offline && cargo publish --locked
}

# already_uploaded OUTFILE: cargo's refusal for a version that exists, naming exactly this version ($N is digits, so the dots are escaped)
already_uploaded() {
  grep -q -E -e 'already exists on crates\.io index|is already uploaded' "$1" \
    && grep -q -E -e "(^|[^0-9.])0\\.0\\.${N}([^0-9]|\$)" "$1"
}

# scrub FILE [N]: print the last N lines of cargo's output with the token value replaced (no regex: the token is data)
scrub() {
  local tok="${CARGO_REGISTRY_TOKEN:-}" line
  while IFS= read -r line; do
    [ -z "$tok" ] || line=${line//"$tok"/***}
    printf '%s\n' "${line//$'\r'/}"
  done < <(tail -n "${2:-12}" "$1")
}

# sparse_has CRATE VERSION: the sparse index (a static CDN, what cargo resolves against): a line for this version that is not yanked
sparse_has() {
  local c=$1 v=$2 p idx
  case ${#c} in 1) p="1/$c" ;; 2) p="2/$c" ;; 3) p="3/${c:0:1}/$c" ;; *) p="${c:0:2}/${c:2:2}/$c" ;; esac
  idx=$(curl -sS --max-time 20 --retry 2 -H "User-Agent: aprender-publish-rehearsal (https://github.com/paiml/aprender)" "https://index.crates.io/$p" 2> /dev/null) || return 1
  printf '%s\n' "$idx" | grep -F "\"vers\":\"$v\"" | grep -q -F '"yanked":false'
}

# One function, called on the last line: bash has read the whole file before anything below can change it.
main() {
  local N="${1:-}" DRY="${2:-0}" V ROOT HERE CRATE_DIR TAG grc prc i seen TRIES WAIT INDEX
  [[ $N =~ ^[1-9][0-9]*$ ]] || { echo "STOP bad run-number '$N' (usage: publish_rehearsal.sh <run-number> [dry_run 0|1])" >&2; exit 2; }
  case "$DRY" in 0|1) ;; *) echo "STOP bad dry_run '$DRY' (want 0 or 1)" >&2; exit 2 ;; esac
  V="0.0.$N"
  say "step a argument ok: run-number $N, version $V, dry_run $DRY"
  HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || die "argument: cannot locate the script"
  ROOT=$(cd "$HERE/../.." && pwd) || die "argument: cannot locate the repo root"
  CRATE_DIR="$ROOT/ci/publish-rehearsal"
  [ -f "$CRATE_DIR/Cargo.toml" ] || die "argument: $CRATE_DIR/Cargo.toml is missing"
  TMP=$(mktemp -d -p "${RUNNER_TEMP:-${TMPDIR:-/tmp}}" pubreh.XXXXXX) || die "argument: no scratch dir"
  trap 'rm -rf -- "${TMP:?}"' EXIT

  # b. clean-room read, for the newest release tag (a vX.Y.Z tag; rc, nightly and other tags are not releases)
  TAG=$(git -C "$ROOT" tag --sort=-creatordate 2> /dev/null | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sed -n 1p) || TAG=""
  [ -n "$TAG" ] || die "clean-room read: no release tag vX.Y.Z in $ROOT (fetch the tags)"
  ( unset CARGO_REGISTRY_TOKEN; run_gate "$ROOT" "$TAG" ) > "$TMP/gate.out" 2>&1; grc=$?   # the gate has no business with the registry token
  sed 's/^/gate: /' "$TMP/gate.out"
  [ "$grc" = 0 ] || die "clean-room read: gate refused $TAG: $(tail -n 1 "$TMP/gate.out")"
  say "step b clean-room read ok: $TAG"

  if [ "$DRY" = 1 ]; then
    say "WOULD publish $CRATE $V (clean-room read ok for $TAG); nothing uploaded"; exit 0
  fi

  # c. upload, from a scratch copy; the version is set there
  [ -n "${CARGO_REGISTRY_TOKEN:-}" ] || die "upload: CARGO_REGISTRY_TOKEN is empty; nothing uploaded"
  mkdir "$TMP/crate" && cp -R "$CRATE_DIR/Cargo.toml" "$CRATE_DIR/README.md" "$CRATE_DIR/src" "$TMP/crate/" || die "upload: cannot copy $CRATE_DIR"
  sed -i "s/^version = \"0\\.0\\.0\"\$/version = \"$V\"/" "$TMP/crate/Cargo.toml"
  grep -q -F -x -e "version = \"$V\"" "$TMP/crate/Cargo.toml" || die "upload: the committed Cargo.toml has no version = \"0.0.0\" placeholder to set"
  ( cd "$TMP/crate" && publish_cmd ) > "$TMP/publish.out" 2>&1; prc=$?
  if [ "$prc" != 0 ] && already_uploaded "$TMP/publish.out"; then
    say "step c upload ok: $CRATE $V was already uploaded (a rerun resumes)"
  elif [ "$prc" != 0 ]; then
    scrub "$TMP/publish.out" 12 | sed 's/^/upload output: /'
    die "upload: refused or failed for $CRATE $V (exit $prc); crates.io is the truth"
  else
    say "step c upload ok: $CRATE $V"
  fi

  # d. read-back on the sparse index; PUB_REH_TRIES x PUB_REH_WAIT seconds (default 30 x 10 s = at most about 5 min), then STOP
  TRIES="${PUB_REH_TRIES:-30}"; WAIT="${PUB_REH_WAIT:-10}"
  [[ $TRIES =~ ^[1-9][0-9]*$ && $WAIT =~ ^[0-9]+$ ]] || die "index read-back: bad PUB_REH_TRIES/PUB_REH_WAIT"
  INDEX="${PUB_REH_INDEX:-sparse_has}"; seen=0; i=0
  while [ "$i" -lt "$TRIES" ]; do
    i=$((i+1))
    $INDEX "$CRATE" "$V" > /dev/null 2>&1 && { seen=1; break; }
    [ "$i" -lt "$TRIES" ] && sleep "$WAIT"
  done
  [ "$seen" = 1 ] || die "index read-back: $CRATE $V is not on the sparse index after $TRIES reads over about $((TRIES*WAIT)) s"
  say "step d index read-back ok: $CRATE $V on the sparse index (read $i of at most $TRIES)"
  say "REHEARSED $CRATE $V: clean-room read ok, uploaded, on the index"
}
main "$@"
