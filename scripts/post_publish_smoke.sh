#!/usr/bin/env bash
# post_publish_smoke.sh — run `cargo install aprender@<v>` in a CLEAN container
# after a publish, and yank that one version when the smoke fails (#7000).
#
# Operator ruling 2026-10-25 16:23Z (verbatim): "Yank leaves the never list for
# ONE case only: the version the release train (R1/R2) published in the last
# 24 h, when the clean-container smoke (`cargo install aprender@<v>` +
# `apr --version` + verb smoke) fails. Then: yank immediately, no quorum, andon
# line, five whys. Any other yank (older versions, other crates) stays on the
# never list."
#
# Usage:
#   scripts/post_publish_smoke.sh --version <v> --train-published <v> \
#       [--container-cmd docker|podman] [--timeout-min N] [--dry-run]
#   scripts/post_publish_smoke.sh --version <v> --train-published <v> --not-measured <reason>
#   scripts/post_publish_smoke.sh --published-at <RFC3339> ...   # ONLY with --dry-run
#   scripts/post_publish_smoke.sh --self-test      # cargo-free, no container, no network
#   scripts/post_publish_smoke.sh --mutants        # plants mutants; each must turn --self-test red
#
# Smoke (inside a fresh official rust container: no mounts, no credentials,
# empty cargo home), in this order, each step with its own marker:
#   0. probe: fetch the sparse-index entry of the crate and require <v> in it.
#      Failure is NOT a release failure: SMOKE_NOTMEASURED (index-probe,
#      version-not-in-index).
#   1. cargo install aprender --version <v> --locked.  Failure is classified from
#      cargo's own output: network / environment (missing system library, no
#      space, OOM) -> SMOKE_NOTMEASURED; compile or resolve error of the crate
#      -> SMOKE_FAIL step=cargo-install; anything else -> SMOKE_NOTMEASURED.
#   2. $CARGO_HOME/bin/apr --version  must carry <v> as an EXACT word (0.72.6 does
#      not match 0.72.60).               SMOKE_FAIL step=apr-version[-mismatch]
#   3. $CARGO_HOME/bin/apr explain E001  must exit 0 and print "Invalid Magic Bytes".
#      WHY THIS VERB: `explain` is requires_model:false, side_effects:[] in
#      contracts/apr-cli-commands-v1.yaml, so it needs no model and no network,
#      yet `explain E001` executes real code (clap dispatch + the error-code
#      table in crates/apr-cli/src/commands/explain.rs). `--help` of a verb only
#      proves clap parsed, not that the shipped code runs.
#                                         SMOKE_FAIL step=verb-smoke[-output]
#
# WHAT MAY YANK. Only a CLASSIFIED failure: the container exited 10 AND its log
# holds `SMOKE_FAIL step=<one of the five names above>`. Everything else is
# not_measured (rc 3, an andon line, NO yank), retried once first: no runtime,
# daemon down, image pull failure, runtime rc 124/125/126/127/137, the timeout
# (default 45 min), no network or index unreachable inside the container,
# version not yet in the index, rc 0 without SMOKE_PASS, any failure without a
# step marker.
#
# THE 24 h CHECK takes the publish time from crates.io `created_at` only.
# --published-at is refused (rc 2) unless --dry-run, which never yanks. A
# publish time more than 5 min in the future, or unknown, never yanks.
#
# Image: SMOKE_IMAGE, default docker.io/library/rust:<channel>-bookworm where
# <channel> is read from rust-toolchain.toml (the toolchain this repo is built
# and gated with). Pinned by exact tag, NOT by digest: no registry was reachable
# when this was written, so a digest could not be resolved; set SMOKE_IMAGE to a
# tag@sha256:... reference to pin harder.
#
# Last stdout line of a smoke run: `SMOKE_RESULT <pass|yanked|failed_no_yank|
# failed_dry_run|yank_failed|not_measured> version=<v> ...` (cascade-drain.sh reads it).
# Exit codes: 0 smoke passed; 1 smoke failed (yanked, refused, dry-run);
# 2 usage; 3 not_measured — never a pass.
#
# The ONLY yank in this file is do_yank(), called from one place, with the
# literal crate `aprender` and the version that passed the guards. The cargo
# credential is used only through cargo: nothing here reads, prints or logs it.

set -euo pipefail

CRATE="aprender"
MAX_AGE_SECS=86400
SKEW_SECS=300
SCRATCH_ROOT="${TMPDIR:-/tmp}"
SELF="${BASH_SOURCE[0]}"
ROOT_DIR=$(cd "$(dirname "$SELF")/.." && pwd)
DRY_RUN=0
PUBLISHED_AT=""
CONTAINER_CMD=""
TIMEOUT_SECS=2700
TICKET="aprender#7000"

default_image() {
  local chan=""
  if [ -f "$ROOT_DIR/rust-toolchain.toml" ]; then
    chan=$(sed -n 's/^channel[[:space:]]*=[[:space:]]*"\([0-9][0-9.]*\)".*/\1/p' "$ROOT_DIR/rust-toolchain.toml" | head -1)
  fi
  [ -n "$chan" ] || chan="1.93.0"
  printf 'docker.io/library/rust:%s-bookworm\n' "$chan"
}
SMOKE_IMAGE="${SMOKE_IMAGE:-$(default_image)}"

# ---------------------------------------------------------------- seams
# Everything with a side effect or a clock is a function so --self-test can
# replace it with a stub.

now_epoch() { date -u +%s; }

crates_io_json() { # <version> -> the crates.io version document, or nothing
  curl -fsS -H "User-Agent: aprender-post-publish-smoke (release automation)" \
    "https://crates.io/api/v1/crates/$CRATE/$1" 2>/dev/null || true
}

# prints epoch seconds of the publish time of $CRATE@$1, or nothing.
# Authoritative source: crates.io. --published-at is honoured only in a dry run
# (smoke_main refuses it otherwise; the DRY_RUN test here is the second lock).
published_epoch() {
  local v=$1 body ts
  if [ -n "$PUBLISHED_AT" ] && [ "$DRY_RUN" = 1 ]; then
    date -u -d "$PUBLISHED_AT" +%s 2>/dev/null || true
    return 0
  fi
  body=$(crates_io_json "$v")
  [ -n "$body" ] || return 0
  # the document must be about THIS version, not some other one
  grep -qE "\"num\"[[:space:]]*:[[:space:]]*\"$v\"" <<<"$body" || return 0
  ts=$(grep -o '"created_at"[[:space:]]*:[[:space:]]*"[^"]*"' <<<"$body" | head -1 | cut -d'"' -f4) || true
  [ -n "$ts" ] || return 0
  date -u -d "$ts" +%s 2>/dev/null || true
}

# safe_rm_rf <path> <prefix> — the ONLY recursive delete in this file. It removes
# <path> only when <path> is strictly under <prefix>, <prefix> is a path this script
# made with mktemp (it starts with "$SCRATCH_ROOT/pps-"), and <path> has no ".."
# component. Anything else is refused: rc 1, nothing removed, a line on stderr.
safe_rm_rf() {
  local p=${1:-} pre=${2:-}
  if [ -z "$p" ] || [ -z "$pre" ]; then
    echo "safe_rm_rf: refused (empty path or prefix)" >&2; return 1
  fi
  case "$pre" in
    "${SCRATCH_ROOT:?}"/pps-*) : ;;
    *) echo "safe_rm_rf: refused ($pre is not one of this script's own mktemp roots)" >&2; return 1 ;;
  esac
  case "/$p/$pre/" in
    */../*) echo "safe_rm_rf: refused ($p has a .. component)" >&2; return 1 ;;
  esac
  case "$p" in
    "$pre"|"$pre"/*) : ;;
    *) echo "safe_rm_rf: refused ($p is outside $pre)" >&2; return 1 ;;
  esac
  rm -rf "${p:?}"
}

detect_container() {
  if [ -n "${CONTAINER_CMD:-}" ]; then
    command -v "$CONTAINER_CMD" 2>/dev/null || true
    return 0
  fi
  command -v docker 2>/dev/null || command -v podman 2>/dev/null || true
}

# The script the container runs. Single-quoted heredoc: nothing is expanded here.
inner_script() {
  local s idx
  idx="${CRATE:0:2}/${CRATE:2:2}/${CRATE}"
  s=$(cat <<'INNER'
set +e
set -f
CARGO_HOME=$(mktemp -d); export CARGO_HOME
notmeasured() { echo "SMOKE_NOTMEASURED step=$1 $2"; exit 11; }
failed() { # a killed or timed-out step is the machine's fault, not the release's
  case "$2" in 124|137) notmeasured "$1" "rc=$2 (killed)" ;; esac
  echo "SMOKE_FAIL step=$1 rc=$2"; exit 10
}
idx=$(curl -fsS --retry 2 --max-time 60 -A "aprender-post-publish-smoke (release automation)" "https://index.crates.io/@IDX@" 2>&1)
rc=$?
[ "$rc" -eq 0 ] || notmeasured index-probe "rc=$rc"
case "$idx" in *"\"vers\":\"$SMOKE_V\""*) : ;; *) notmeasured version-not-in-index "v=$SMOKE_V" ;; esac
ilog=$(mktemp)
cargo install @CRATE@ --version "$SMOKE_V" --locked >"$ilog" 2>&1
rc=$?
cat "$ilog"
if [ "$rc" -ne 0 ]; then
  if grep -q -i -E 'Could not resolve|resolve host|failed to download|download of .* failed|spurious network|network failure|Connection (refused|reset|timed out)|timed out|SSL (connect|error)|TLS (handshake|error)|certificate verif|Temporary failure|unable to update registry|failed to query replaced source|No route to host' "$ilog"; then
    notmeasured cargo-install-network "rc=$rc"
  fi
  if grep -q -i -E 'cannot find -l|pkg-config|could not find system library|OpenSSL|No space left|Killed|out of memory|Cannot allocate memory|signal: 9|SIGKILL|linker .cc. not found|cc: command not found|requires rustc|is not supported by' "$ilog"; then
    notmeasured cargo-install-environment "rc=$rc"
  fi
  if grep -q -E 'could not compile|failed to compile|failed to select a version|no matching package|failed to parse manifest|needs to be updated but --locked|error\[E[0-9]+\]' "$ilog"; then
    failed cargo-install "$rc"
  fi
  notmeasured cargo-install-unclassified "rc=$rc"
fi
out=$("$CARGO_HOME/bin/apr" --version 2>&1)
rc=$?
echo "apr --version: $out"
[ "$rc" -eq 0 ] || failed apr-version "$rc"
ok=0
for w in $out; do [ "$w" = "$SMOKE_V" ] && ok=1; done
[ "$ok" -eq 1 ] || failed apr-version-mismatch 1
out=$("$CARGO_HOME/bin/apr" explain E001 2>&1)
rc=$?
echo "apr explain E001: $out"
[ "$rc" -eq 0 ] || failed verb-smoke "$rc"
case "$out" in *"Invalid Magic Bytes"*) : ;; *) failed verb-smoke-output 1 ;; esac
echo "SMOKE_PASS"
exit 0
INNER
)
  s=${s//@IDX@/$idx}
  s=${s//@CRATE@/$CRATE}
  printf '%s\n' "$s"
}

# container_run <runtime> <version> <logfile> -> rc of the runtime (timeout: 124)
container_run() {
  local rt=$1 v=$2 log=$3 rc name inner
  name="pps-$$-$RANDOM"
  inner=$(inner_script)
  timeout "$TIMEOUT_SECS" "$rt" run --rm --name "$name" -e SMOKE_V="$v" "$SMOKE_IMAGE" bash -c "$inner" >"$log" 2>&1
  rc=$?
  if [ "$rc" -eq 124 ]; then "$rt" rm -f "$name" >/dev/null 2>&1 || true; fi
  return "$rc"
}

# classify_run <rc> <log> -> "pass" | "fail <step>" | "nm <reason>"
classify_run() {
  local rc=$1 log=$2 step
  case "$rc" in
    0) if grep -q '^SMOKE_PASS$' "$log" 2>/dev/null; then echo "pass"
       else echo "nm exit 0 without a SMOKE_PASS marker"; fi ;;
    10) step=$(grep -o '^SMOKE_FAIL step=[a-z-]*' "$log" 2>/dev/null | tail -1 | sed 's/^SMOKE_FAIL step=//') || true
        case "$step" in
          cargo-install|apr-version|apr-version-mismatch|verb-smoke|verb-smoke-output) echo "fail $step" ;;
          *) echo "nm exit 10 without a known SMOKE_FAIL step marker (got '${step:-none}')" ;;
        esac ;;
    11) step=$(grep -o '^SMOKE_NOTMEASURED step=[a-z-]*' "$log" 2>/dev/null | tail -1 | sed 's/^SMOKE_NOTMEASURED step=//') || true
        echo "nm ${step:-container} (reported by the container)" ;;
    124) echo "nm timeout (${TIMEOUT_SECS}s)" ;;
    125) echo "nm container runtime error rc=125 (daemon down or image pull failed)" ;;
    126|127) echo "nm container runtime cannot exec rc=$rc" ;;
    137) echo "nm container killed rc=137 (OOM or stop)" ;;
    *) echo "nm no step marker, runtime rc=$rc" ;;
  esac
}

# THE ONLY YANK. Literal crate, version already guarded by the caller.
do_yank() {
  local v=$1
  cargo yank --version "$v" "$CRATE"
}

# inbox line: `time | session | ticket | status`, 300 characters at most
andon_line() { # <version> <what> <detail>
  local l
  l=$(printf '%s | aprender-release | %s %s@%s | ANDON %s' "$(date -u +%H:%MZ)" "$TICKET" "$CRATE" "$1" "$2: $3")
  printf '%s' "${l:0:300}"
}

write_andon() { # one line on $1
  local inbox="${ANDON_INBOX:-}"
  if [ -z "$inbox" ] && [ -e /mnt/nvme-raid0/cop-inbox/inbox.md ]; then
    inbox=/mnt/nvme-raid0/cop-inbox/inbox.md
  fi
  if [ -n "$inbox" ]; then
    printf '%s\n' "$1" >>"$inbox"
  else
    printf '%s\n' "$1" >&2
  fi
}

rerun_cmd() { # <version>
  printf 'bash scripts/post_publish_smoke.sh --version %s --train-published %s' "$1" "$1"
}

report_not_measured() { # <version> <reason>; rc 3, no yank, never a pass
  local v=$1 why=$2 line
  line=$(andon_line "$v" "not_measured" "$why; no yank; rerun: $(rerun_cmd "$v")")
  if [ "$DRY_RUN" = 1 ]; then echo "WOULD_ANDON $line"; else write_andon "$line"; fi
  echo "post-publish smoke: not_measured ($why). No yank. Run later: $(rerun_cmd "$v")" >&2
  echo "SMOKE_RESULT not_measured version=$v reason=$why"
  return 3
}

# ---------------------------------------------------------------- logic

# yank_allowed <version> <train_v> -> rc 0 allowed; rc 1 with reason on stdout
yank_allowed() {
  local v=$1 train_v=$2 pub now age
  if [ -z "$train_v" ]; then echo "no --train-published version given"; return 1; fi
  if [ "$train_v" != "$v" ]; then
    echo "version $v is not the version this release run published ($train_v)"; return 1
  fi
  pub=$(published_epoch "$v")
  if [ -z "$pub" ]; then echo "publish time of $CRATE@$v unknown"; return 1; fi
  now=$(now_epoch)
  age=$((now - pub))
  if [ "$age" -lt "-$SKEW_SECS" ]; then echo "publish time is in the future (clock skew over ${SKEW_SECS}s)"; return 1; fi
  if [ "$age" -lt "$MAX_AGE_SECS" ]; then return 0; fi
  echo "published ${age}s ago, not within 24 h"
  return 1
}

smoke_main() {
  local version="" train_v="" nm_reason="" rt log smoke_rc verdict attempt why step
  DRY_RUN=0; PUBLISHED_AT=""; CONTAINER_CMD=""
  TIMEOUT_SECS="${SMOKE_TIMEOUT_SECS:-2700}"
  while [ $# -gt 0 ]; do
    case "$1" in
      --dry-run) DRY_RUN=1; shift; continue ;;
      --version|--train-published|--published-at|--container-cmd|--timeout-min|--not-measured)
        [ $# -ge 2 ] || { echo "usage: $1 needs a value" >&2; return 2; } ;;
      *) echo "usage: unknown argument '$1'" >&2; return 2 ;;
    esac
    case "$1" in
      --version) version="$2" ;;
      --train-published) train_v="$2" ;;
      --published-at) PUBLISHED_AT="$2" ;;
      --container-cmd) CONTAINER_CMD="$2" ;;
      --not-measured) nm_reason="$2" ;;
      --timeout-min)
        [[ "$2" =~ ^[1-9][0-9]*$ ]] || { echo "usage: --timeout-min needs a positive integer (got '$2')" >&2; return 2; }
        TIMEOUT_SECS=$(($2 * 60)) ;;
    esac
    shift 2
  done
  if ! [[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([-+][0-9A-Za-z.+-]+)?$ ]]; then
    echo "usage: --version <semver> is required (got '$version')" >&2; return 2
  fi
  if [ -n "$PUBLISHED_AT" ] && [ "$DRY_RUN" != 1 ]; then
    echo "usage: --published-at is accepted only with --dry-run; the 24 h check reads crates.io" >&2; return 2
  fi
  if [ -n "$nm_reason" ]; then report_not_measured "$version" "$nm_reason"; return $?; fi
  rt=$(detect_container)
  if [ -z "$rt" ]; then
    report_not_measured "$version" "no container runtime (docker|podman) found; the smoke did not run" || true
    return 3
  fi
  mkdir -p "$SCRATCH_ROOT"
  log=$(mktemp "$SCRATCH_ROOT/post-publish-smoke.XXXXXX")
  echo "post-publish smoke: $CRATE@$version in a clean container ($rt, $SMOKE_IMAGE, timeout ${TIMEOUT_SECS}s)"
  verdict=""
  for attempt in 1 2; do
    smoke_rc=0
    container_run "$rt" "$version" "$log.$attempt" || smoke_rc=$?
    verdict=$(classify_run "$smoke_rc" "$log.$attempt")
    case "$verdict" in
      nm*) echo "post-publish smoke: attempt $attempt not_measured (${verdict#nm }) (log: $log.$attempt)" >&2
           [ "$attempt" = 2 ] || sleep "${SMOKE_RETRY_SLEEP:-20}" ;;
      *) break ;;
    esac
  done
  case "$verdict" in
    pass)
      echo "post-publish smoke: PASS ($CRATE@$version)"
      echo "SMOKE_RESULT pass version=$version"
      rm -f "${log:?}" "${log:?}".[12]
      return 0 ;;
    nm*)
      report_not_measured "$version" "${verdict#nm } after 2 attempts (log: $log.2)" || true
      return 3 ;;
  esac
  step=${verdict#fail }
  echo "post-publish smoke: FAIL step=$step rc=$smoke_rc (log: $log.$attempt)" >&2
  if why=$(yank_allowed "$version" "$train_v"); then
    if [ "$DRY_RUN" = 1 ]; then
      echo "WOULD_YANK $CRATE@$version"
      echo "WOULD_ANDON $(andon_line "$version" "yank" "$step rc=$smoke_rc; log $log.$attempt; five whys owed")"
      echo "SMOKE_RESULT failed_dry_run version=$version step=$step"
      return 1
    fi
    if do_yank "$version"; then
      write_andon "$(andon_line "$version" "yank" "$step rc=$smoke_rc; log $log.$attempt; five whys owed")"
      echo "post-publish smoke: yanked $CRATE@$version (undo: cargo yank --undo --version $version $CRATE)" >&2
      echo "NOTIFY-COP: SendMessage the cop now, a yank is destructive (inbox README): $CRATE@$version yanked, step=$step" >&2
      echo "SMOKE_RESULT yanked version=$version step=$step"
    else
      write_andon "$(andon_line "$version" "yank-FAILED" "$step rc=$smoke_rc; broken release still live; log $log.$attempt")"
      echo "post-publish smoke: cargo yank FAILED for $CRATE@$version" >&2
      echo "NOTIFY-COP: SendMessage the cop now, broken release $CRATE@$version is live and the yank failed" >&2
      echo "SMOKE_RESULT yank_failed version=$version step=$step"
    fi
    return 1
  fi
  echo "smoke failed; yank not permitted by rule ($why)" >&2
  echo "SMOKE_RESULT failed_no_yank version=$version step=$step reason=$why"
  return 1
}

# ---------------------------------------------------------------- self-test

ST_FAIL=0
ST_CALLS=""
# st_case name want_rc want_yanks want_andon want_out setup_fn args...
# ST_CALLS (global, "" = unchecked) is the expected number of container runs.
st_case() {
  local name=$1 want_rc=$2 want_yanks=$3 want_andon=$4 want_out=$5 setup=$6; shift 6
  local d rc out yanks andon calls ok=1
  d=$(mktemp -d "$SCRATCH_ROOT/pps-selftest.XXXXXX")
  : >"$d/yanks"; : >"$d/andon"; : >"$d/calls"
  out=$(
    set +e
    ST_DIR="$d"; export ST_DIR
    ANDON_INBOX="$d/andon"
    SCRATCH_ROOT="$d"
    SMOKE_RETRY_SLEEP=0
    do_yank() { echo "$1" >>"$d/yanks"; return 0; }
    now_epoch() { echo 1000000; }
    detect_container() { echo /stub/docker; }
    "$setup"
    smoke_main "$@" 2>&1
    echo "RC=$?"
  )
  rc=${out##*RC=}
  yanks=$(tr '\n' ' ' <"$d/yanks" | sed 's/ $//')
  andon=$(cat "$d/andon")
  calls=$(wc -l <"$d/calls" | tr -d ' ')
  [ "$rc" = "$want_rc" ] || ok=0
  [ "$yanks" = "$want_yanks" ] || ok=0
  if [ -n "$want_andon" ]; then case "$andon" in *"$want_andon"*) : ;; *) ok=0 ;; esac
  else [ -z "$andon" ] || ok=0; fi
  if [ -n "$want_out" ]; then case "$out" in *"$want_out"*) : ;; *) ok=0 ;; esac; fi
  if [ -n "$ST_CALLS" ] && [ "$calls" != "$ST_CALLS" ]; then ok=0; fi
  if [ "$ok" = 1 ]; then
    printf 'PASS  %-46s rc=%s yanks=[%s] runs=%s\n' "$name" "$rc" "$yanks" "$calls"
  else
    printf 'FAIL  %-46s rc=%s (want %s) yanks=[%s] (want [%s]) runs=%s (want %s) andon=[%s]\n' \
      "$name" "$rc" "$want_rc" "$yanks" "$want_yanks" "$calls" "${ST_CALLS:-any}" "$andon"
    printf '%s\n' "$out" | sed 's/^/      | /'
    ST_FAIL=$((ST_FAIL + 1))
  fi
  safe_rm_rf "$d" "$d" || ST_FAIL=$((ST_FAIL + 1))
  ST_CALLS=""
}

st_unit() { # name want got
  if [ "$2" = "$3" ]; then printf 'PASS  %-46s [%s]\n' "$1" "$3"
  else printf 'FAIL  %-46s want [%s] got [%s]\n' "$1" "$2" "$3"; ST_FAIL=$((ST_FAIL + 1)); fi
}

# the stub runtime and the stub tools it puts on PATH: they run the REAL
# in-container script text (inner_script) with `bash -c`, no container involved.
st_make_fakes() {
  mkdir -p "$ST_DIR/bin"
  cat >"$ST_DIR/rt" <<'RT'
#!/usr/bin/env bash
[ "$1" = rm ] && exit 0
case "${STUB_RT:-}" in
  daemon) echo "Cannot connect to the Docker daemon at unix:///var/run/docker.sock" >&2; exit 125 ;;
  pull)   echo "Error response from daemon: pull access denied, repository does not exist" >&2; exit 125 ;;
  hang)   sleep 30; exit 0 ;;
esac
shift  # run
while [ $# -gt 0 ]; do
  case "$1" in
    --rm) shift ;;
    --name) shift 2 ;;
    -e) export "${2?}"; shift 2 ;;
    *) break ;;
  esac
done
shift  # image
[ "$1" = bash ] && [ "$2" = -c ] || exit 99
exec env -i PATH="$ST_DIR/bin:/usr/bin:/bin" TMPDIR="$ST_DIR" SMOKE_V="$SMOKE_V" \
  STUB_CARGO="${STUB_CARGO:-ok}" STUB_APR="${STUB_APR:-ok}" STUB_CURL="${STUB_CURL:-ok}" ST_DIR="$ST_DIR" \
  bash -c "$3"
RT
  cat >"$ST_DIR/bin/curl" <<'CURL'
#!/usr/bin/env bash
case "${STUB_CURL:-ok}" in
  down) echo "curl: (6) Could not resolve host: index.crates.io"; exit 6 ;;
  missing) echo '{"name":"aprender","vers":"0.0.1","yanked":false}'; exit 0 ;;
  *) printf '{"name":"aprender","vers":"%s","yanked":false}\n' "$SMOKE_V"; exit 0 ;;
esac
CURL
  cat >"$ST_DIR/bin/cargo" <<'CARGO'
#!/usr/bin/env bash
[ "$1" = install ] || { echo "stub cargo: only install is stubbed" >&2; exit 99; }
shift
args=" $* "
case "$args" in *" aprender "*) : ;; *) echo "stub cargo: wrong crate: $args"; exit 99 ;; esac
case "$args" in *" --locked "*) : ;; *) echo "stub cargo: --locked missing: $args"; exit 99 ;; esac
case "$args" in *" --version $SMOKE_V "*) : ;; *) echo "stub cargo: wrong version: $args"; exit 99 ;; esac
case "${STUB_CARGO:-ok}" in
  compile) echo "error[E0432]: unresolved import"; echo "error: could not compile \`aprender\` (bin \"apr\") due to 1 previous error"; exit 101 ;;
  resolve) echo "error: failed to select a version for the requirement \`aprender-core = \"=$SMOKE_V\"\`"; exit 101 ;;
  network) echo "error: failed to download from \`https://static.crates.io/crates/serde/serde-1.0.crate\`"; echo "Caused by: [6] Could not resolve host: static.crates.io"; exit 101 ;;
  syslib)  echo "error: failed to run custom build command for \`openssl-sys\`"; echo "Could not find openssl via pkg-config"; echo "error: could not compile \`openssl-sys\`"; exit 101 ;;
  disk)    echo "error: could not compile \`aprender\`"; echo "No space left on device (os error 28)"; exit 101 ;;
  boom)    echo "boom"; exit 101 ;;
esac
mkdir -p "$CARGO_HOME/bin"
cat >"$CARGO_HOME/bin/apr" <<APR
#!/usr/bin/env bash
case "\$1" in
  --version)
    case "$STUB_APR" in
      ver60) echo "apr ${SMOKE_V}0" ;;
      verfail) echo "apr: segfault"; exit 139 ;;
      *) echo "apr $SMOKE_V" ;;
    esac ;;
  explain)
    case "$STUB_APR" in
      explainfail) echo "panic"; exit 101 ;;
      explainwrong) echo "E001: something else"; exit 0 ;;
      *) echo "E001: Invalid Magic Bytes"; exit 0 ;;
    esac ;;
esac
APR
chmod +x "$CARGO_HOME/bin/apr"
echo "Installed package \`aprender v$SMOKE_V\`"
exit 0
CARGO
  chmod +x "$ST_DIR/rt" "$ST_DIR/bin/curl" "$ST_DIR/bin/cargo"
}

self_test() {
  [ -n "${BASH_VERSION:-}" ] || { echo "needs bash" >&2; return 2; }

  # 1000000 is the stubbed clock; publish times are stubbed as epochs.
  st_cr_calls() { echo x >>"$ST_DIR/calls"; }
  st_fresh()  { published_epoch() { echo $((1000000 - 3600)); }; container_run() { st_cr_calls; echo "SMOKE_FAIL step=verb-smoke rc=7" >"$3"; return 10; }; }
  st_stale()  { published_epoch() { echo $((1000000 - 90000)); }; container_run() { st_cr_calls; echo "SMOKE_FAIL step=verb-smoke rc=7" >"$3"; return 10; }; }
  st_edge()   { published_epoch() { echo $((1000000 - 86400)); }; container_run() { st_cr_calls; echo "SMOKE_FAIL step=cargo-install rc=101" >"$3"; return 10; }; }
  st_nopub()  { published_epoch() { echo ""; }; container_run() { st_cr_calls; echo "SMOKE_FAIL step=cargo-install rc=101" >"$3"; return 10; }; }
  st_future() { published_epoch() { echo $((1000000 + 3600)); }; container_run() { st_cr_calls; echo "SMOKE_FAIL step=verb-smoke rc=7" >"$3"; return 10; }; }
  st_skew()   { published_epoch() { echo $((1000000 + 100)); }; container_run() { st_cr_calls; echo "SMOKE_FAIL step=verb-smoke rc=7" >"$3"; return 10; }; }
  st_ok()     { published_epoch() { echo $((1000000 - 3600)); }; container_run() { st_cr_calls; echo SMOKE_PASS >"$3"; return 0; }; }
  st_none()   { detect_container() { echo ""; }; published_epoch() { echo $((1000000 - 3600)); }; }
  # not_measured family: a FRESH publish time and a yank stub that would record a yank
  st_rc()     { # <rc> <log text>
    ST_RC=$1; ST_TXT=$2
    published_epoch() { echo $((1000000 - 3600)); }
    container_run() { st_cr_calls; printf '%s\n' "$ST_TXT" >"$3"; return "$ST_RC"; }
  }
  st_daemon() { st_rc 125 "Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?"; }
  st_pull()   { st_rc 125 "Unable to find image 'rust:1.93.0-bookworm' locally; pull access denied"; }
  st_r126()   { st_rc 126 "docker: Error response from daemon: permission denied"; }
  st_r127()   { st_rc 127 "docker: command not found"; }
  st_r137()   { st_rc 137 "Killed"; }
  st_r124()   { st_rc 124 ""; }
  st_nomark() { st_rc 9 "something broke with no marker"; }
  st_r10nomark() { st_rc 10 "exit 10 but no SMOKE_FAIL line"; }
  st_r10unk() { st_rc 10 "SMOKE_FAIL step=container rc=1"; }
  st_r0nopass() { st_rc 0 "exited zero but never said SMOKE_PASS"; }
  st_netin()  { st_rc 11 "SMOKE_NOTMEASURED step=cargo-install-network rc=101"; }
  st_idxdown() { st_rc 11 "SMOKE_NOTMEASURED step=index-probe rc=6"; }
  st_flaky()  { # first run daemon-down, second run passes
    published_epoch() { echo $((1000000 - 3600)); }
    container_run() { st_cr_calls; if [ "$(wc -l <"$ST_DIR/calls")" -ge 2 ]; then echo SMOKE_PASS >"$3"; return 0; fi; echo "daemon" >"$3"; return 125; }
  }
  # the in-container script text, run by the real container_run through the stub runtime
  st_fake() { # <STUB_RT> <STUB_CURL> <STUB_CARGO> <STUB_APR>
    st_make_fakes
    published_epoch() { echo $((1000000 - 3600)); }
    detect_container() { echo "$ST_DIR/rt"; }
    STUB_RT=$1; STUB_CURL=$2; STUB_CARGO=$3; STUB_APR=$4
    export STUB_RT STUB_CURL STUB_CARGO STUB_APR
    st_cr_real=$(declare -f container_run)
    eval "${st_cr_real/container_run/real_container_run}"
    container_run() { st_cr_calls; real_container_run "$@"; }
  }
  st_f_ok()      { st_fake none ok ok ok; }
  st_f_compile() { st_fake none ok compile ok; }
  st_f_resolve() { st_fake none ok resolve ok; }
  st_f_network() { st_fake none ok network ok; }
  st_f_syslib()  { st_fake none ok syslib ok; }
  st_f_disk()    { st_fake none ok disk ok; }
  st_f_boom()    { st_fake none ok boom ok; }
  st_f_curl()    { st_fake none down ok ok; }
  st_f_noidx()   { st_fake none missing ok ok; }
  st_f_ver60()   { st_fake none ok ok ver60; }
  st_f_verfail() { st_fake none ok ok verfail; }
  st_f_expfail() { st_fake none ok ok explainfail; }
  st_f_expwrong() { st_fake none ok ok explainwrong; }
  st_f_daemon()  { st_fake daemon ok ok ok; }
  st_f_pull()    { st_fake pull ok ok ok; }
  st_f_hang()    { st_fake hang ok ok ok; TIMEOUT_SECS=1; SMOKE_TIMEOUT_SECS=1; export SMOKE_TIMEOUT_SECS; }
  # a run that must refuse before it does anything
  st_refuse() { published_epoch() { echo $((1000000 - 3600)); }; container_run() { st_cr_calls; echo SMOKE_PASS >"$3"; return 0; }; }

  local A="--version 0.75.0 --train-published 0.75.0"
  local FAILIN="ANDON yank: verb-smoke rc=10; log "
  ST_CALLS=1; st_case "smoke pass"                        0 ""      ""                  "SMOKE_RESULT pass"     st_ok      $A
  ST_CALLS=1; st_case "classified fail, fresh, train version" 1 "0.75.0" "| aprender#7000 aprender@0.75.0 | ANDON yank: verb-smoke rc=10; log " "SMOKE_RESULT yanked" st_fresh $A
  st_case "fail at 25 h"                1 ""      ""                  "not permitted by rule" st_stale   $A
  st_case "fail at exactly 24 h"        1 ""      ""                  "not permitted by rule" st_edge    $A
  st_case "version mismatch"            1 ""      ""                  "not the version this release run published" st_fresh --version 0.75.0 --train-published 0.74.9
  st_case "no --train-published"        1 ""      ""                  "no --train-published"  st_fresh    --version 0.75.0
  st_case "publish time unknown"        1 ""      ""                  "unknown"               st_nopub   $A
  st_case "publish time 1 h in the future"  1 ""  ""                  "in the future"         st_future  $A
  st_case "publish time 100 s ahead (skew ok)" 1 "0.75.0" "ANDON yank" ""                     st_skew    $A
  st_case "dry-run fail prints WOULD_YANK"  1 ""  ""                  "WOULD_YANK aprender@0.75.0" st_fresh $A --dry-run
  st_case "--published-at without --dry-run"  2 "" ""                 "only with --dry-run"   st_fresh   $A --published-at 2026-11-13T09:00:00Z
  st_case "--published-at with --dry-run"   1 ""  ""                  "WOULD_YANK"            st_fresh   $A --published-at 2026-01-01T00:00:00Z --dry-run
  st_case "bad version arg"             2 ""      ""                  "usage"                 st_ok      --version "0.75.0; rm -rf /" --train-published 0.75.0
  st_case "unknown argument"            2 ""      ""                  "unknown argument"      st_ok      --bogus
  local f
  for f in --version --train-published --published-at --container-cmd --timeout-min --not-measured; do
    st_case "trailing $f with no value"  2 ""     ""                  "needs a value"         st_ok      $A "$f"
  done
  st_case "--timeout-min not a number"  2 ""      ""                  "positive integer"      st_ok      $A --timeout-min abc
  # not_measured: no yank, rc 3, andon line says not_measured, retried once (2 runs)
  st_case "no container runtime"        3 ""      "not_measured"      "no container runtime"  st_none    $A
  ST_CALLS=2; st_case "daemon down rc125 -> not_measured"   3 "" "not_measured" "SMOKE_RESULT not_measured" st_daemon $A
  ST_CALLS=2; st_case "image pull failure rc125"            3 "" "not_measured" ""   st_pull    $A
  ST_CALLS=2; st_case "runtime rc126"                       3 "" "not_measured" ""   st_r126    $A
  ST_CALLS=2; st_case "runtime rc127"                       3 "" "not_measured" ""   st_r127    $A
  ST_CALLS=2; st_case "container killed rc137 (OOM)"        3 "" "not_measured" ""   st_r137    $A
  ST_CALLS=2; st_case "timeout rc124"                       3 "" "not_measured" "timeout" st_r124 $A
  ST_CALLS=2; st_case "no marker, rc 9 (was: yank)"         3 "" "not_measured" ""   st_nomark  $A
  ST_CALLS=2; st_case "rc 10 without a marker"              3 "" "not_measured" ""   st_r10nomark $A
  ST_CALLS=2; st_case "rc 10, unknown step 'container'"     3 "" "not_measured" ""   st_r10unk  $A
  ST_CALLS=2; st_case "rc 0 without SMOKE_PASS"             3 "" "not_measured" ""   st_r0nopass $A
  ST_CALLS=2; st_case "no network inside container"         3 "" "not_measured" ""   st_netin   $A
  ST_CALLS=2; st_case "index unreachable inside container"  3 "" "not_measured" ""   st_idxdown $A
  ST_CALLS=2; st_case "retry: daemon down then pass"        0 "" ""             "SMOKE_RESULT pass" st_flaky $A
  st_case "--not-measured reason (index wait timeout)" 3 "" "not_measured" "SMOKE_RESULT not_measured" st_refuse $A --not-measured "index wait timed out"
  # the REAL in-container script, under stub cargo/curl/apr (no container, no network)
  ST_CALLS=1; st_case "in-container: all steps pass"        0 "" ""             "SMOKE_RESULT pass" st_f_ok $A
  ST_CALLS=1; st_case "in-container: compile error -> yank" 1 "0.75.0" "ANDON yank: cargo-install" "SMOKE_RESULT yanked" st_f_compile $A
  ST_CALLS=1; st_case "in-container: resolve error -> yank" 1 "0.75.0" "ANDON yank" "step=cargo-install" st_f_resolve $A
  ST_CALLS=2; st_case "in-container: cargo network error"   3 "" "not_measured" "cargo-install-network" st_f_network $A
  ST_CALLS=2; st_case "in-container: missing system library" 3 "" "not_measured" "cargo-install-environment" st_f_syslib $A
  ST_CALLS=2; st_case "in-container: disk full"             3 "" "not_measured" "cargo-install-environment" st_f_disk $A
  ST_CALLS=2; st_case "in-container: unclassified install failure" 3 "" "not_measured" "cargo-install-unclassified" st_f_boom $A
  ST_CALLS=2; st_case "in-container: index probe fails"     3 "" "not_measured" "index-probe" st_f_curl $A
  ST_CALLS=2; st_case "in-container: version not in index"  3 "" "not_measured" "version-not-in-index" st_f_noidx $A
  ST_CALLS=1; st_case "in-container: apr 0.72.60 vs 0.72.6" 1 "0.72.6" "ANDON yank" "apr-version-mismatch" st_f_ver60 --version 0.72.6 --train-published 0.72.6
  ST_CALLS=1; st_case "in-container: apr --version rc 139" 1 "0.75.0" "ANDON yank" "step=apr-version" st_f_verfail $A
  ST_CALLS=1; st_case "in-container: verb smoke rc 101"    1 "0.75.0" "ANDON yank" "step=verb-smoke" st_f_expfail $A
  ST_CALLS=1; st_case "in-container: verb smoke wrong output" 1 "0.75.0" "ANDON yank" "verb-smoke-output" st_f_expwrong $A
  ST_CALLS=2; st_case "runtime: daemon down (stub runtime)" 3 "" "not_measured" "" st_f_daemon $A
  ST_CALLS=2; st_case "runtime: pull failure (stub runtime)" 3 "" "not_measured" "" st_f_pull $A
  ST_CALLS=2; st_case "runtime: hang -> real timeout, no yank" 3 "" "not_measured" "timeout (1s)" st_f_hang $A

  # published_epoch against canned crates.io documents
  local doc
  doc='{"version":{"num":"0.75.0","created_at":"2026-11-13T08:00:00.123456+00:00","yanked":false}}'
  (
    crates_io_json() { printf '%s' "$doc"; }
    DRY_RUN=0; PUBLISHED_AT=""
    st_unit "crates.io created_at -> epoch"        1794556800 "$(published_epoch 0.75.0)"
    st_unit "crates.io document of another version" "" "$(published_epoch 0.75.1)"
    DRY_RUN=0; PUBLISHED_AT="2020-01-01T00:00:00Z"
    st_unit "PUBLISHED_AT ignored when not dry-run"  1794556800 "$(published_epoch 0.75.0)"
    DRY_RUN=1
    st_unit "PUBLISHED_AT honoured in dry-run"       1577836800 "$(published_epoch 0.75.0)"
    crates_io_json() { printf ''; }
    DRY_RUN=0; PUBLISHED_AT=""
    st_unit "crates.io unreachable -> unknown"       "" "$(published_epoch 0.75.0)"
  ) || ST_FAIL=$((ST_FAIL + 1))
  # st_unit ran in a subshell: its FAIL count is lost, so re-run and count from the output
  local ub
  ub=$(
    crates_io_json() { printf '%s' "$doc"; }
    DRY_RUN=0; PUBLISHED_AT=""
    [ "$(published_epoch 0.75.0)" = 1794556800 ] || echo bad1
    [ -z "$(published_epoch 0.75.1)" ] || echo bad2
    PUBLISHED_AT="2020-01-01T00:00:00Z"
    [ "$(published_epoch 0.75.0)" = 1794556800 ] || echo bad3
    DRY_RUN=1
    [ "$(published_epoch 0.75.0)" = 1577836800 ] || echo bad4
    crates_io_json() { printf ''; }
    DRY_RUN=0; PUBLISHED_AT=""
    [ -z "$(published_epoch 0.75.0)" ] || echo bad5
  )
  if [ -n "$ub" ]; then echo "FAIL  published_epoch unit rows: $ub"; ST_FAIL=$((ST_FAIL + 1)); fi

  # safe_rm_rf refuses everything outside this script's own mktemp roots. The
  # decoy is a directory made by mktemp but NOT under a pps- root: a delete that
  # reaches it is a delete outside the script's own scratch tree.
  local rr decoy rrc rm_ok=1
  rr=$(mktemp -d "$SCRATCH_ROOT/pps-rmtest.XXXXXX")
  decoy=$(mktemp -d "$SCRATCH_ROOT/decoy-rmtest.XXXXXX")
  mkdir "$rr/sub"; : >"$decoy/keep"
  rrc=0; safe_rm_rf "$decoy" "$rr" 2>/dev/null || rrc=$?
  [ "$rrc" = 1 ] && [ -e "$decoy/keep" ] || { rm_ok=0; echo "FAIL  safe_rm_rf: path outside the prefix (rc=$rrc)"; }
  rrc=0; safe_rm_rf "$decoy" "$decoy" 2>/dev/null || rrc=$?
  [ "$rrc" = 1 ] && [ -e "$decoy/keep" ] || { rm_ok=0; echo "FAIL  safe_rm_rf: prefix that is not a pps- mktemp root (rc=$rrc)"; }
  rrc=0; safe_rm_rf "" "$rr" 2>/dev/null || rrc=$?
  [ "$rrc" = 1 ] || { rm_ok=0; echo "FAIL  safe_rm_rf: empty path (rc=$rrc)"; }
  rrc=0; safe_rm_rf "$rr" "" 2>/dev/null || rrc=$?
  [ "$rrc" = 1 ] && [ -d "$rr" ] || { rm_ok=0; echo "FAIL  safe_rm_rf: empty prefix (rc=$rrc)"; }
  rrc=0; safe_rm_rf "$rr/sub/../../${decoy##*/}" "$rr" 2>/dev/null || rrc=$?
  [ "$rrc" = 1 ] && [ -e "$decoy/keep" ] || { rm_ok=0; echo "FAIL  safe_rm_rf: .. component (rc=$rrc)"; }
  rrc=0; safe_rm_rf "$rr/../${decoy##*/}" "$rr/../${decoy##*/}" 2>/dev/null || rrc=$?
  [ "$rrc" = 1 ] && [ -e "$decoy/keep" ] || { rm_ok=0; echo "FAIL  safe_rm_rf: .. in the prefix (rc=$rrc)"; }
  rrc=0; safe_rm_rf "/" "$rr" 2>/dev/null || rrc=$?
  [ "$rrc" = 1 ] || { rm_ok=0; echo "FAIL  safe_rm_rf: / (rc=$rrc)"; }
  rrc=0; safe_rm_rf "$rr/sub" "$rr" || rrc=$?
  [ "$rrc" = 0 ] && [ ! -e "$rr/sub" ] && [ -d "$rr" ] || { rm_ok=0; echo "FAIL  safe_rm_rf: a path inside the root is removed (rc=$rrc)"; }
  rrc=0; safe_rm_rf "$rr" "$rr" || rrc=$?
  [ "$rrc" = 0 ] && [ ! -e "$rr" ] || { rm_ok=0; echo "FAIL  safe_rm_rf: the root itself is removed (rc=$rrc)"; }
  rm -f "$decoy/keep"; rmdir "$decoy" 2>/dev/null || true
  if [ "$rm_ok" = 1 ]; then echo "PASS  safe_rm_rf refuses paths outside its own mktemp roots (9 rows)"
  else ST_FAIL=$((ST_FAIL + 1)); fi

  # structural
  local root y chan
  root="$ROOT_DIR"
  y=$(grep -c '^  cargo yank --version "\$v" "\$CRATE"$' "$SELF") || y=0
  if [ "$y" = 1 ]; then echo "PASS  exactly one yank command in do_yank"
  else echo "FAIL  yank command count=$y"; ST_FAIL=$((ST_FAIL + 1)); fi
  if grep -q '^CRATE="aprender"$' "$SELF"; then echo "PASS  CRATE is the literal aprender"
  else echo "FAIL  CRATE is not the literal aprender"; ST_FAIL=$((ST_FAIL + 1)); fi
  chan=$(sed -n 's/^channel[[:space:]]*=[[:space:]]*"\([0-9][0-9.]*\)".*/\1/p' "$root/rust-toolchain.toml" 2>/dev/null | head -1)
  case "$(SMOKE_IMAGE='' default_image)" in
    *":${chan:-NOCHANNEL}-bookworm") echo "PASS  default image is pinned to the rust-toolchain.toml channel ($chan)" ;;
    *) echo "FAIL  default image does not carry the toolchain channel '$chan'"; ST_FAIL=$((ST_FAIL + 1)) ;;
  esac
  if grep -q 'post_publish_smoke.sh' "$root/scripts/cascade-publish.sh" 2>/dev/null \
     && grep -q -- '--train-published' "$root/scripts/cascade-publish.sh" \
     && grep -q -- '--not-measured' "$root/scripts/cascade-publish.sh"; then
    echo "PASS  cascade-publish.sh calls the smoke and reports an index-wait timeout as not_measured"
  else echo "FAIL  cascade-publish.sh does not call the smoke / does not report an index-wait timeout"; ST_FAIL=$((ST_FAIL + 1)); fi
  if grep -qF 'crc=${PIPESTATUS[0]}' "$root/scripts/cascade-drain.sh" \
     && grep -qF "grep '^SMOKE_RESULT '" "$root/scripts/cascade-drain.sh" \
     && grep -qF 'yanked\":true' "$root/scripts/cascade-drain.sh" \
     && ! grep -qE 'cascade-publish\.sh \) 2>&1 *\| *grep' "$root/scripts/cascade-drain.sh"; then
    echo "PASS  cascade-drain.sh keeps the cascade rc, reads SMOKE_RESULT, and does not count a yanked version"
  else echo "FAIL  cascade-drain.sh loses the cascade rc / ignores the smoke / counts a yanked version"; ST_FAIL=$((ST_FAIL + 1)); fi
  if [ "$ST_FAIL" -eq 0 ]; then echo "self-test: ALL PASS"; return 0; fi
  echo "self-test: $ST_FAIL FAILED" >&2
  return 1
}

# mutants: each edited copy must turn --self-test red.
mutants() {
  local d n=0 bad=0 rc target
  d=$(mktemp -d "$SCRATCH_ROOT/pps-mutants.XXXXXX")
  mut() { # name file sed-expression
    local name=$1 file=$2 expr=$3
    safe_rm_rf "$d/scripts" "$d" || { bad=$((bad + 1)); return; }
    mkdir -p "$d/scripts"
    cp "$SELF" "$d/scripts/post_publish_smoke.sh"
    cp "$(dirname "$SELF")/cascade-publish.sh" "$d/scripts/cascade-publish.sh"
    cp "$(dirname "$SELF")/cascade-drain.sh" "$d/scripts/cascade-drain.sh"
    cp "$ROOT_DIR/rust-toolchain.toml" "$d/rust-toolchain.toml"
    target="$d/scripts/$file"
    cp "$target" "$target.orig"
    sed -i "$expr" "$target"
    if cmp -s "$target.orig" "$target"; then
      echo "MUTANT $name: NOT APPLIED (pattern drifted)"; bad=$((bad + 1)); return
    fi
    rm -f "$target.orig"
    n=$((n + 1))
    rc=0
    bash "$d/scripts/post_publish_smoke.sh" --self-test >/dev/null 2>&1 || rc=$?
    if [ "$rc" -ne 0 ]; then echo "MUTANT $name: killed (self-test rc=$rc)"
    else echo "MUTANT $name: SURVIVED"; bad=$((bad + 1)); fi
  }
  local P=post_publish_smoke.sh
  mut "flip-24h-comparison"      $P 's/if \[ "\$age" -lt "\$MAX_AGE_SECS" \]/if [ "$age" -gt "$MAX_AGE_SECS" ]/'
  mut "drop-version-equality"    $P 's/if \[ "\$train_v" != "\$v" \]/if false/'
  mut "swallow-smoke-rc"         $P 's/container_run "\$rt" "\$version" "\$log.\$attempt" || smoke_rc=\$?/container_run "$rt" "$version" "$log.$attempt" || smoke_rc=0/'
  mut "yank-in-dry-run"          $P 's/if \[ "\$DRY_RUN" = 1 \]; then$/if false; then/'
  mut "no-runtime-is-pass"       $P 's/^    return 3$/    return 0/'
  mut "skip-train-check"         $P 's/if \[ -z "\$train_v" \]; then echo "no --train-published version given"; return 1; fi//'
  mut "R3-crate-from-env"        $P 's/^CRATE="aprender"$/CRATE="${SMOKE_CRATE:-aprender}"/'
  mut "R4-version-check-true"    $P 's/\[ "\$ok" -eq 1 \] || failed apr-version-mismatch 1/:/'
  mut "R4b-substring-version"    $P 's/for w in \$out; do \[ "\$w" = "\$SMOKE_V" \] \&\& ok=1; done/case "$out" in *"$SMOKE_V"*) ok=1 ;; esac/'
  mut "R6-future-time-allowed"   $P 's/if \[ "\$age" -lt "-\$SKEW_SECS" \]; then/if false; then/'
  mut "R7-drop-locked"           $P 's/ --version "\$SMOKE_V" --locked >"\$ilog"/ --version "$SMOKE_V" >"$ilog"/'
  mut "B1-unmarked-failure-yanks" $P 's/^    \*) echo "nm no step marker, runtime rc=\$rc" ;;/    *) echo "fail container" ;;/'
  mut "B1-daemon-down-yanks"     $P 's/^    125) echo "nm container runtime error.*$/    125) echo "fail container" ;;/'
  mut "B1-rc0-without-pass-ok"   $P 's/else echo "nm exit 0 without a SMOKE_PASS marker"; fi ;;/else echo "pass"; fi ;;/'
  mut "B1-network-is-release-fail" $P 's/notmeasured cargo-install-network "rc=\$rc"/failed cargo-install "$rc"/'
  mut "B1-unclassified-is-fail"  $P 's/notmeasured cargo-install-unclassified "rc=\$rc"/failed cargo-install "$rc"/'
  mut "B1-no-retry"              $P 's/for attempt in 1 2; do/for attempt in 1; do/'
  mut "B1-no-timeout"            $P 's/^  timeout "\$TIMEOUT_SECS" "\$rt" run/  "$rt" run/'
  mut "B2-published-at-anywhere" $P 's/if \[ -n "\$PUBLISHED_AT" \] \&\& \[ "\$DRY_RUN" != 1 \]; then/if false; then/'
  mut "B2-published-at-in-lookup" $P 's/if \[ -n "\$PUBLISHED_AT" \] \&\& \[ "\$DRY_RUN" = 1 \]; then/if [ -n "$PUBLISHED_AT" ]; then/'
  mut "B2-skip-num-check"        $P 's/^  grep -qE "\\"num.*|| return 0$//'
  mut "S5-flag-without-value-loops" $P 's/\[ \$# -ge 2 \] || { echo "usage: \$1 needs a value" >\&2; return 2; } ;;/: ;;/'
  mut "R8-rm-outside-prefix-allowed" $P 's/^    "\$pre"|"\$pre"\/\*) : ;;$/    *) : ;;/'
  mut "R9-rm-dotdot-allowed"     $P 's/^    \*\/\.\.\/\*) echo "safe_rm_rf/    NOMATCH) echo "safe_rm_rf/'
  mut "R10-rm-root-not-pps"      $P 's/^    "\${SCRATCH_ROOT:?}"\/pps-\*) : ;;$/    *) : ;;/'
  mut "B3-index-timeout-falls-through" cascade-publish.sh 's/--not-measured "sparse index/--version "$TARGET_VERSION" "sparse index/'
  mut "B4-drain-loses-rc"        cascade-drain.sh 's/^    crc=\${PIPESTATUS\[0\]}$/    crc=0/'
  mut "B4-drain-ignores-smoke"   cascade-drain.sh 's/grep '"'"'^SMOKE_RESULT /grep '"'"'^NOSUCH_RESULT /'
  mut "B4-drain-counts-yanked"   cascade-drain.sh 's/\*\\"yanked\\":true\*) exit 0/*\\"yanked\\":nope*) exit 0/'
  safe_rm_rf "$d" "$d" || bad=$((bad + 1))
  if [ "$bad" -eq 0 ]; then echo "mutants: all $n killed"; return 0; fi
  echo "mutants: $bad problem(s)" >&2
  return 1
}

# ---------------------------------------------------------------- entry

case "${1:-}" in
  --self-test) self_test; exit $? ;;
  --mutants) mutants; exit $? ;;
  -h|--help|"") awk 'NR>1 && !/^#/ {exit} NR>1 {sub(/^# ?/, ""); print}' "$SELF"; [ -n "${1:-}" ] && exit 0 || exit 2 ;;
esac
rc=0
smoke_main "$@" || rc=$?
exit "$rc"
