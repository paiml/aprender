#!/usr/bin/env bash
# post_publish_smoke.sh — run `cargo install aprender@<v>` in a CLEAN container
# after a publish, and yank that one version only when the failure is proven to
# be the release's (#7000).
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
#       [--container-cmd docker|podman] [--timeout-min N] [--repeat-wait SECS] [--dry-run]
#   scripts/post_publish_smoke.sh --version <v> --train-published <v> --not-measured <reason>
#   scripts/post_publish_smoke.sh --published-at <RFC3339> ...   # ONLY with --dry-run
#   scripts/post_publish_smoke.sh --self-test    # stubbed cargo/runtime/clock; no container, no network
#   scripts/post_publish_smoke.sh --case-table   # the failure classifier against its case table + fixtures
#   scripts/post_publish_smoke.sh --mutants      # plants mutants; each must turn --case-table or --self-test red
#
# THREE VERDICTS, one final stdout line, one exit code each:
#   SMOKE_RESULT PASS version=<v>                                   rc 0
#   SMOKE_RESULT YANK version=<v> step=<s> action=<a>               rc 1
#       a = yanked | dry-run | refused (24 h / train rule) | yank-failed
#   SMOKE_RESULT NOT_MEASURED version=<v> reason=<why>              rc 3
#   (rc 2 is a usage error and prints no verdict.) NOT_MEASURED is never a pass
#   and never a yank; it writes an andon line with the exact rerun command.
#
# One ATTEMPT = one fresh container (official rust image: no mounts, no
# credentials, empty cargo home, CARGO_TERM_COLOR=never) running, in order:
#   0. probe the crates.io sparse index for <v>; missing/unreachable ->
#      SMOKE_NOTMEASURED (the version is not resolvable yet; says nothing).
#   1. cargo install aprender --version <v> --locked          step cargo-install
#   2. $CARGO_HOME/bin/apr --version carries <v> as an EXACT word
#                                         step apr-version / apr-version-mismatch
#   3. $CARGO_HOME/bin/apr explain E001 exits 0 and prints "Invalid Magic Bytes"
#      (`explain` is requires_model:false, side_effects:[] in
#      contracts/apr-cli-commands-v1.yaml: no model, no network, yet it runs real
#      code, crates/apr-cli/src/commands/explain.rs)
#                                         step verb-smoke / verb-smoke-output
#   On ANY failed step the container then runs THE CONTROL: `cargo install
#   aprender --version <prev> --locked` in the same image, same run, where <prev>
#   is the newest non-yanked, non-prerelease version below <v> in the sparse
#   index. It prints `SMOKE_CONTROL result=pass|fail|undeterminable`.
#
# WHAT MAY YANK, exactly. ALL of:
#   (1) the control PASSED in the attempt (and in the repeat). Control fail,
#       undeterminable or missing -> NOT_MEASURED (control-*), never a yank.
#   (2) the SAME step failed again in a REPEAT attempt started at least
#       --repeat-wait seconds (env SMOKE_REPEAT_WAIT, default 120; 0 allowed for
#       tests) after the first failing attempt started, measured on the clock.
#       Fail then pass, fail then another step, fail then not_measured, or a
#       repeat that started too soon -> NOT_MEASURED.
#   (3) NEITHER attempt's output matches a network, HTTP 5xx/429, resolver/index
#       or environment pattern (classify_log, below; one pattern per line in the
#       *_PATTERNS arrays, each with a must-match row in --case-table). Any match
#       -> NOT_MEASURED (<step>-<class>).
#   (4) for step cargo-install: the output has a BUILD-failure line (`could not
#       compile`, `failed to run custom build command`, rustc `error[E...]`,
#       `failed to parse manifest`, a `--locked` lockfile mismatch). The bare
#       `error: failed to compile \`aprender vX\`` wrapper cargo prints around
#       EVERY install failure is not evidence. No build line ->
#       NOT_MEASURED (cargo-install-unclassified).
#   (5) the 24 h rule: <v> is --train-published and crates.io `created_at` is
#       under 24 h ago (unknown, or more than 5 min in the future, refuses).
#   (1)-(4) hold but (5) does not -> YANK action=refused (andon, no yank).
#   Everything else is NOT_MEASURED: no runtime, daemon down, pull failure,
#   runtime rc 124/125/126/127/137, the timeout (default 45 min), rc 0 without
#   SMOKE_PASS, a failure without a known step marker. A not_measured first
#   attempt is retried once (SMOKE_RETRY_SLEEP, default 20 s).
#
# classify_log strips ANSI colour codes and drops cargo progress lines
# (`Compiling pkg-config v0.3.31` names a crate; a crate name is not evidence),
# then tries the classes in order: network, http, resolver, environment, build.
#
# THE 24 h CHECK takes the publish time from crates.io `created_at` only.
# --published-at is refused (rc 2) unless --dry-run, which never yanks.
#
# Image: SMOKE_IMAGE, default docker.io/library/rust:<channel>-bookworm where
# <channel> is read from rust-toolchain.toml. Pinned by tag, not digest; set
# SMOKE_IMAGE to tag@sha256:... to pin harder.
#
# Scratch: every file this script writes lives in a `mktemp -d` root it made
# (pps-*) and registered; safe_rm_rf is the ONLY delete and refuses anything
# else. The ONLY yank is do_yank(), called from one place, with the literal
# crate `aprender`. The cargo credential is used only through cargo: nothing
# here reads, prints or logs it.

set -euo pipefail

CRATE="aprender"
MAX_AGE_SECS=86400
SKEW_SECS=300
REPEAT_WAIT_DEFAULT=120
SCRATCH_ROOT="${TMPDIR:-/tmp}"
SELF="${BASH_SOURCE[0]}"
ROOT_DIR=$(cd "$(dirname "$SELF")/.." && pwd)
FIXTURES="$(cd "$(dirname "$SELF")" && pwd)/fixtures/post-publish-smoke"
DRY_RUN=0
PUBLISHED_AT=""
CONTAINER_CMD=""
TIMEOUT_SECS=2700
TICKET="aprender#7000"
OWN_ROOTS="|"
OWN_ROOT_LAST=""

default_image() {
  local chan=""
  if [ -f "$ROOT_DIR/rust-toolchain.toml" ]; then
    chan=$(sed -n 's/^channel[[:space:]]*=[[:space:]]*"\([0-9][0-9.]*\)".*/\1/p' "$ROOT_DIR/rust-toolchain.toml" | head -1)
  fi
  [ -n "$chan" ] || chan="1.93.0"
  printf 'docker.io/library/rust:%s-bookworm\n' "$chan"
}
SMOKE_IMAGE="${SMOKE_IMAGE:-$(default_image)}"

# ---------------------------------------------------------------- classifier
# One pattern per line (the mutant check deletes each line in turn; every one
# must be load-bearing in --case-table). grep -i -E, after ANSI strip and the
# progress-line filter.

NET_PATTERNS=(
  'could not resolve'
  "couldn't resolve"
  'failed to download'
  'download of .* failed'
  'spurious network error'
  'network failure'
  'connection (refused|reset|timed out)'
  'timed out'
  'timeout was reached'
  '\[(6|7|28|35|52|55|56)\] '
  'ssl (connect|error)'
  'tls (handshake|error)'
  'certificate verif'
  'temporary failure in name resolution'
  'no route to host'
)
HTTP_PATTERNS=(
  'got (429|5[0-9][0-9])'
  'http/[0-9.]+ (429|5[0-9][0-9])'
  'status( code)?:? (429|5[0-9][0-9])'
  'too many requests'
  'service unavailable'
  'bad gateway'
  'gateway time-?out'
  'internal server error'
)
RESOLVER_PATTERNS=(
  'failed to select a version'
  'no matching package named'
  'candidate versions found'
  'location searched'
  'could not find .* in registry'
  'failed to get '
  'unable to get packages'
  'failed to load source'
  'failed to update'
  'failed to fetch'
  'failed to query'
  'unable to update registry'
  'failed to read .*index'
  'failed to parse registry'
  'index .*(lag|stale|out of date|not (yet )?(updated|synced))'
)
ENV_PATTERNS=(
  'cannot find -l'
  'pkg-config'
  'could not find system library'
  'openssl'
  'no space left'
  'killed'
  'out of memory'
  'cannot allocate memory'
  'signal: 9'
  'sigkill'
  'linker .cc. not found'
  'cc: command not found'
  'requires rustc'
  'is not supported by'
)
BUILD_PATTERNS=(
  'could not compile'
  'failed to run custom build command'
  'failed to parse manifest'
  'needs to be updated but --locked'
  'error\[e[0-9]+\]'
)
PROGRESS_RE='^[[:space:]]*(Compiling|Downloaded|Downloading|Updating|Installing|Installed|Fresh|Checking|Locking|Adding|Building|Finished|Replacing|Replaced|Ignored|Packaging|Blocking)[[:space:]]'

pat_any() { # <text> <pattern>... -> rc 0 when any pattern matches
  local text=$1; shift
  local IFS='|'
  grep -q -i -E -- "$*" <<<"$text"
}

# classify_stream < log -> network | http | resolver | environment | build | none
classify_stream() {
  local diag esc
  esc=$(printf '\033')
  diag=$(tr -d '\r' | sed "s/${esc}\[[0-9;]*[A-Za-z]//g" | grep -v -E "$PROGRESS_RE" || true)
  if pat_any "$diag" "${NET_PATTERNS[@]}"; then echo network; return 0; fi
  if pat_any "$diag" "${HTTP_PATTERNS[@]}"; then echo http; return 0; fi
  if pat_any "$diag" "${RESOLVER_PATTERNS[@]}"; then echo resolver; return 0; fi
  if pat_any "$diag" "${ENV_PATTERNS[@]}"; then echo environment; return 0; fi
  if pat_any "$diag" "${BUILD_PATTERNS[@]}"; then echo build; return 0; fi
  echo none
}
classify_log() { classify_stream <"$1"; }

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

owned_by_me() { [ -O "$1" ]; } # seam: --self-test overrides it

# previous_version <v> -> newest non-yanked, non-prerelease version below <v> in the
# crates.io sparse index (the data cargo itself resolves against), or nothing.
index_doc() { # the sparse-index document of $CRATE, or nothing
  curl -fsS --max-time 60 -A "aprender-post-publish-smoke (release automation)" \
    "https://index.crates.io/${CRATE:0:2}/${CRATE:2:2}/${CRATE}" 2>/dev/null || true
}
previous_version() {
  local v=$1 doc line vers cands prev
  doc=$(index_doc)
  [ -n "$doc" ] || return 0
  cands=""
  while IFS= read -r line; do
    vers=$(sed -n 's/.*"vers":"\([^"]*\)".*/\1/p' <<<"$line")
    [ -n "$vers" ] || continue
    case "$vers" in *-*|*+*) continue ;; esac
    grep -q '"yanked":false' <<<"$line" || continue
    cands="$cands$vers"$'\n'
  done <<<"$doc"
  [ -n "$cands" ] || return 0
  prev=$(printf '%s%s\n' "$cands" "$v" | sort -V -u | grep -B1 -x -F -- "$v" | head -1) || true
  [ "$prev" != "$v" ] || prev=""
  printf '%s' "$prev"
}

# own_root <tag> — make a `mktemp -d` root "$SCRATCH_ROOT/pps-<tag>.XXXXXX" and
# register it; the path lands in OWN_ROOT_LAST. Called directly (never inside
# $(...)), so the registration survives.
own_root() {
  local r
  r=$(mktemp -d "${SCRATCH_ROOT:?}/pps-$1.XXXXXX") || return 1
  OWN_ROOTS="$OWN_ROOTS$r|"
  OWN_ROOT_LAST=$r
}

srr_refuse() { echo "safe_rm_rf: refused ($1)" >&2; return 1; }

# safe_rm_rf <path> <root> — the ONLY delete in this file. Removes <path> only
# when ALL hold: <root> was made and registered by own_root in this process and
# is "$SCRATCH_ROOT/pps-*"; <root> is a real directory (not a symlink) owned by
# the current uid; <path> is <root> or under it, has no "." or ".." component,
# is not a symlink, is owned by the current uid, and its physical parent
# resolves under the physical <root>. Anything else: rc 1, nothing removed.
safe_rm_rf() {
  local p=${1:-} r=${2:-} rp pp
  if [ -z "$p" ] || [ -z "$r" ]; then srr_refuse "empty path or root"; return 1; fi
  case "$OWN_ROOTS" in
    *"|$r|"*) : ;;
    *) srr_refuse "$r is not a root this run made with mktemp -d"; return 1 ;;
  esac
  case "$r" in
    "${SCRATCH_ROOT:?}"/pps-*) : ;;
    *) srr_refuse "$r is not under $SCRATCH_ROOT/pps-"; return 1 ;;
  esac
  if [ -L "$r" ]; then srr_refuse "root $r is a symlink"; return 1; fi
  if [ ! -d "$r" ]; then srr_refuse "root $r is not a directory"; return 1; fi
  if ! owned_by_me "$r"; then srr_refuse "root $r is not owned by the current uid"; return 1; fi
  case "/$p/" in
    */../*|*/./*) srr_refuse "$p has a . or .. component"; return 1 ;;
  esac
  case "$p" in
    "$r"|"$r"/*) : ;;
    *) srr_refuse "$p is outside $r"; return 1 ;;
  esac
  if [ -L "$p" ]; then srr_refuse "$p is a symlink"; return 1; fi
  [ -e "$p" ] || return 0
  if ! owned_by_me "$p"; then srr_refuse "$p is not owned by the current uid"; return 1; fi
  rp=$(cd -P -- "$r" && pwd) || { srr_refuse "cannot resolve $r"; return 1; }
  pp=$(cd -P -- "$(dirname -- "$p")" && pwd) || { srr_refuse "cannot resolve the parent of $p"; return 1; }
  case "$pp" in
    "$rp"|"$rp"/*|"${rp%/*}") : ;;
    *) srr_refuse "$p resolves outside $rp"; return 1 ;;
  esac
  if [ "$p" != "$r" ] && [ "$pp" = "${rp%/*}" ]; then srr_refuse "$p resolves outside $rp"; return 1; fi
  rm -rf -- "${p:?}"
}

detect_container() {
  if [ -n "${CONTAINER_CMD:-}" ]; then
    command -v "$CONTAINER_CMD" 2>/dev/null || true
    return 0
  fi
  command -v docker 2>/dev/null || command -v podman 2>/dev/null || true
}

# The script the container runs. Single-quoted heredoc: nothing is expanded here.
# It only REPORTS (markers + cargo output); the host classifies.
inner_script() {
  local s idx
  idx="${CRATE:0:2}/${CRATE:2:2}/${CRATE}"
  s=$(cat <<'INNER'
set +e
set -f
CARGO_TERM_COLOR=never; export CARGO_TERM_COLOR
CARGO_HOME=$(mktemp -d); export CARGO_HOME
notmeasured() { echo "SMOKE_NOTMEASURED step=$1 $2"; exit 11; }
control() { # the previous release, same image, same run
  if [ -z "$SMOKE_PREV" ]; then echo "SMOKE_CONTROL result=undeterminable rc=none prev=none"; return 0; fi
  clog=$(mktemp)
  cargo install @CRATE@ --version "$SMOKE_PREV" --locked --root "$(mktemp -d)" >"$clog" 2>&1
  crc=$?
  tail -n 40 "$clog"
  if [ "$crc" -eq 0 ]; then cres=pass; else cres=fail; fi
  echo "SMOKE_CONTROL result=$cres rc=$crc prev=$SMOKE_PREV"
}
failed() { # a killed or timed-out step is the machine's fault, not the release's
  case "$2" in 124|137) notmeasured "$1" "rc=$2 (killed)" ;; esac
  control
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
[ "$rc" -eq 0 ] || failed cargo-install "$rc"
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
container_run() { real_container_run "$@"; } # the seam --self-test overrides
real_container_run() {
  local rt=$1 v=$2 log=$3 rc name inner
  name="pps-${log##*/}-$$" # unique per run and attempt
  inner=$(inner_script)
  timeout "$TIMEOUT_SECS" "$rt" run --rm --name "$name" -e SMOKE_V="$v" -e SMOKE_PREV="${SMOKE_PREV:-}" "$SMOKE_IMAGE" bash -c "$inner" >"$log" 2>&1
  rc=$?
  if [ "$rc" -eq 124 ]; then "$rt" rm -f "${name:?}" >/dev/null 2>&1 || true; fi
  return "$rc"
}

# classify_run <rc> <log> -> "pass" | "fail <step>" | "nm <reason>"
classify_run() {
  local rc=$1 log=$2 step cls ctl
  case "$rc" in
    0) if grep -q '^SMOKE_PASS$' "$log" 2>/dev/null; then echo "pass"
       else echo "nm exit 0 without a SMOKE_PASS marker"; fi
       return 0 ;;
    10) : ;;
    11) step=$(grep -o '^SMOKE_NOTMEASURED step=[a-z-]*' "$log" 2>/dev/null | tail -1 | sed 's/^SMOKE_NOTMEASURED step=//') || true
        echo "nm ${step:-container} (reported by the container)"; return 0 ;;
    124) echo "nm timeout (${TIMEOUT_SECS}s)"; return 0 ;;
    125) echo "nm container runtime error rc=125 (daemon down or image pull failed)"; return 0 ;;
    126|127) echo "nm container runtime cannot exec rc=$rc"; return 0 ;;
    137) echo "nm container killed rc=137 (OOM or stop)"; return 0 ;;
    *) echo "nm no step marker, runtime rc=$rc"; return 0 ;;
  esac
  step=$(grep -o '^SMOKE_FAIL step=[a-z-]*' "$log" 2>/dev/null | tail -1 | sed 's/^SMOKE_FAIL step=//') || true
  case "$step" in
    cargo-install|apr-version|apr-version-mismatch|verb-smoke|verb-smoke-output) : ;;
    *) echo "nm exit 10 without a known SMOKE_FAIL step marker (got '${step:-none}')"; return 0 ;;
  esac
  cls=$(classify_log "$log")
  case "$cls" in
    network|http|resolver|environment) echo "nm $step-$cls"; return 0 ;;
  esac
  ctl=$(grep -o '^SMOKE_CONTROL result=[a-z]*' "$log" 2>/dev/null | tail -1 | sed 's/^SMOKE_CONTROL result=//') || true
  case "$ctl" in
    pass) : ;;
    fail) echo "nm control-failed"; return 0 ;;
    undeterminable) echo "nm control-undeterminable"; return 0 ;;
    *) echo "nm control-missing"; return 0 ;;
  esac
  if [ "$step" = cargo-install ] && [ "$cls" != build ]; then echo "nm cargo-install-unclassified"; return 0; fi
  echo "fail $step"
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

andon() { # dry-run prints it, a live run writes it
  if [ "$DRY_RUN" = 1 ]; then echo "WOULD_ANDON $1"; else write_andon "$1"; fi
}

rerun_cmd() { # <version>
  printf 'bash scripts/post_publish_smoke.sh --version %s --train-published %s' "$1" "$1"
}

report_not_measured() { # <version> <reason>; rc 3, no yank, never a pass
  local v=$1 why=$2
  andon "$(andon_line "$v" "not_measured" "$why; no yank; rerun: $(rerun_cmd "$v")")"
  echo "post-publish smoke: NOT_MEASURED ($why). No yank. Run later: $(rerun_cmd "$v")" >&2
  echo "SMOKE_RESULT NOT_MEASURED version=$v reason=$why"
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

# one fresh-container attempt; globals in: SM_RT SM_VERSION SM_LOG
# globals out: SM_ATTEMPT (incremented) SM_RC SM_VERDICT
smoke_attempt() {
  SM_ATTEMPT=$((SM_ATTEMPT + 1))
  SM_RC=0
  container_run "$SM_RT" "$SM_VERSION" "$SM_LOG.$SM_ATTEMPT" || SM_RC=$?
  SM_VERDICT=$(classify_run "$SM_RC" "$SM_LOG.$SM_ATTEMPT")
}

smoke_main() {
  local version="" train_v="" nm_reason="" repeat_wait first step why t1 t2 gap alog
  DRY_RUN=0; PUBLISHED_AT=""; CONTAINER_CMD=""
  TIMEOUT_SECS="${SMOKE_TIMEOUT_SECS:-2700}"
  repeat_wait=${SMOKE_REPEAT_WAIT:-$REPEAT_WAIT_DEFAULT}
  while [ $# -gt 0 ]; do
    case "$1" in
      --dry-run) DRY_RUN=1; shift; continue ;;
      --version|--train-published|--published-at|--container-cmd|--timeout-min|--not-measured|--repeat-wait)
        [ $# -ge 2 ] || { echo "usage: $1 needs a value" >&2; return 2; } ;;
      *) echo "usage: unknown argument '$1'" >&2; return 2 ;;
    esac
    case "$1" in
      --version) version="$2" ;;
      --train-published) train_v="$2" ;;
      --published-at) PUBLISHED_AT="$2" ;;
      --container-cmd) CONTAINER_CMD="$2" ;;
      --not-measured) nm_reason="$2" ;;
      --repeat-wait) repeat_wait="$2" ;;
      --timeout-min)
        [[ "$2" =~ ^[1-9][0-9]*$ ]] || { echo "usage: --timeout-min needs a positive integer (got '$2')" >&2; return 2; }
        TIMEOUT_SECS=$(($2 * 60)) ;;
    esac
    shift 2
  done
  if ! [[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([-+][0-9A-Za-z.+-]+)?$ ]]; then
    echo "usage: --version <semver> is required (got '$version')" >&2; return 2
  fi
  if ! [[ "$repeat_wait" =~ ^[0-9]+$ ]]; then
    echo "usage: --repeat-wait / SMOKE_REPEAT_WAIT needs a whole number of seconds (got '$repeat_wait')" >&2; return 2
  fi
  if [ -n "$PUBLISHED_AT" ] && [ "$DRY_RUN" != 1 ]; then
    echo "usage: --published-at is accepted only with --dry-run; the 24 h check reads crates.io" >&2; return 2
  fi
  if [ -n "$nm_reason" ]; then report_not_measured "$version" "$nm_reason"; return $?; fi
  SM_RT=$(detect_container)
  if [ -z "$SM_RT" ]; then
    report_not_measured "$version" "no container runtime (docker|podman) found; the smoke did not run" || true
    return 3
  fi
  mkdir -p "$SCRATCH_ROOT"
  if ! own_root run; then
    report_not_measured "$version" "mktemp -d under $SCRATCH_ROOT failed; the smoke did not run" || true
    return 3
  fi
  SM_LOG="$OWN_ROOT_LAST/attempt"
  SM_VERSION=$version
  SM_ATTEMPT=0
  SMOKE_PREV=$(previous_version "$version") || SMOKE_PREV=""
  export SMOKE_PREV
  echo "post-publish smoke: $CRATE@$version in a clean container ($SM_RT, $SMOKE_IMAGE, timeout ${TIMEOUT_SECS}s, control ${SMOKE_PREV:-none}, repeat wait ${repeat_wait}s)"

  # first measurement: a not_measured attempt is retried once
  t1=$(now_epoch); smoke_attempt
  case "$SM_VERDICT" in
    pass|fail*) : ;;
    *)
      echo "post-publish smoke: attempt $SM_ATTEMPT not_measured (${SM_VERDICT#nm }) (log: $SM_LOG.$SM_ATTEMPT); retrying once" >&2
      sleep "${SMOKE_RETRY_SLEEP:-20}"
      t1=$(now_epoch); smoke_attempt ;;
  esac
  alog="$SM_LOG.$SM_ATTEMPT"
  case "$SM_VERDICT" in
    pass)
      echo "post-publish smoke: PASS ($CRATE@$version)"
      safe_rm_rf "${SM_LOG%/*}" "${SM_LOG%/*}" || true
      echo "SMOKE_RESULT PASS version=$version"
      return 0 ;;
    nm*)
      report_not_measured "$version" "${SM_VERDICT#nm } (log: $alog)" || true
      return 3 ;;
  esac

  # a failure: it must repeat, identically, at least repeat_wait seconds later
  first=$SM_VERDICT
  step=${first#fail }
  echo "post-publish smoke: attempt $SM_ATTEMPT FAIL step=$step (log: $alog); repeating after ${repeat_wait}s to confirm" >&2
  sleep "$repeat_wait"
  t2=$(now_epoch); smoke_attempt
  gap=$((t2 - t1))
  if [ "$gap" -lt "$repeat_wait" ]; then
    report_not_measured "$version" "repeat started ${gap}s after the first attempt, under the ${repeat_wait}s wait; unconfirmed (log: $alog)" || true
    return 3
  fi
  case "$SM_VERDICT" in
    "$first") : ;;
    pass)
      report_not_measured "$version" "unconfirmed: $step failed then passed on the repeat (log: $alog)" || true
      return 3 ;;
    *)
      report_not_measured "$version" "unconfirmed: $step failed, the repeat gave '${SM_VERDICT}' (log: $SM_LOG.$SM_ATTEMPT)" || true
      return 3 ;;
  esac
  alog="$SM_LOG.$SM_ATTEMPT"

  echo "post-publish smoke: FAIL step=$step confirmed ${gap}s apart, control passed both times (log: $alog)" >&2
  if why=$(yank_allowed "$version" "$train_v"); then
    if [ "$DRY_RUN" = 1 ]; then
      echo "WOULD_YANK $CRATE@$version"
      andon "$(andon_line "$version" "yank" "$step; log $alog; five whys owed")"
      echo "SMOKE_RESULT YANK version=$version step=$step action=dry-run"
      return 1
    fi
    if do_yank "$version"; then
      write_andon "$(andon_line "$version" "yank" "$step; log $alog; five whys owed")"
      echo "post-publish smoke: yanked $CRATE@$version (undo: cargo yank --undo --version $version $CRATE)" >&2
      echo "NOTIFY-COP: SendMessage the cop now, a yank is destructive (inbox README): $CRATE@$version yanked, step=$step" >&2
      echo "SMOKE_RESULT YANK version=$version step=$step action=yanked"
    else
      write_andon "$(andon_line "$version" "yank-FAILED" "$step; broken release still live; log $alog")"
      echo "post-publish smoke: cargo yank FAILED for $CRATE@$version" >&2
      echo "NOTIFY-COP: SendMessage the cop now, broken release $CRATE@$version is live and the yank failed" >&2
      echo "SMOKE_RESULT YANK version=$version step=$step action=yank-failed"
    fi
    return 1
  fi
  andon "$(andon_line "$version" "yank-refused" "$step confirmed failing but $why; no yank; log $alog")"
  echo "post-publish smoke: confirmed failure; yank not permitted by rule ($why)" >&2
  echo "SMOKE_RESULT YANK version=$version step=$step action=refused reason=$why"
  return 1
}

# ---------------------------------------------------------------- case table
# class<TAB>text. One must-match row per pattern above (a row that only THAT
# pattern of its class matches, so deleting the pattern changes the class), and
# must-not-match rows (class none). `\e` is ESC. Then every fixture file in
# scripts/fixtures/post-publish-smoke/ (class = the file-name prefix before `--`).
case_rows() {
  cat <<'ROWS'
network	error: Could not resolve host: index.crates.io
network	  Couldn't resolve proxy name
network	  failed to download from `https://static.crates.io/crates/aprender-core/0.75.0/download`
network	error: download of ap/re/aprender failed
network	warning: spurious network error (2 tries remaining)
network	  network failure seems to have happened
network	  Connection refused (os error 111)
network	  operation timed out
network	  Timeout was reached
network	  [56] Failure when receiving data from the peer
network	  SSL connect error
network	  TLS handshake failed
network	  server certificate verification failed
network	  Temporary failure in name resolution
network	  No route to host (os error 113)
http	error: failed to get successful HTTP response from `https://index.crates.io/ap/re/aprender`, got 503
http	  < HTTP/2 502
http	  status code: 500
http	  Too Many Requests
http	  Service Unavailable
http	  Bad Gateway
http	  Gateway Timeout
http	  Internal Server Error
resolver	  failed to select a version for the requirement `aprender-core = "=0.75.0"`
resolver	error: no matching package named `aprender-core` found
resolver	  candidate versions found which didn't match: 0.74.9, 0.74.8
resolver	  location searched: crates.io index
resolver	error: could not find `aprender` in registry `crates-io` with version `=0.75.0`
resolver	  failed to get `aprender-core` as a dependency of package `aprender v0.75.0`
resolver	  unable to get packages from source
resolver	  failed to load source for dependency `aprender-core`
resolver	  failed to update registry `crates-io`
resolver	  failed to fetch `https://github.com/rust-lang/crates.io-index`
resolver	  failed to query replaced source registry `crates-io`
resolver	error: unable to update registry `crates-io`
resolver	  failed to read `/tmp/h/registry/index/index.crates.io-1949cf8c6b5b557f/config.json`
resolver	  failed to parse registry's information for: aprender
resolver	  the crates.io index is stale
environment	  = note: /usr/bin/ld: cannot find -lclang
environment	  The pkg-config command could not be found.
environment	  could not find system library 'libzstd' required by the 'zstd-sys' crate
environment	  Could not find directory of OpenSSL installation
environment	  No space left on device (os error 28)
environment	  Killed
environment	rustc: out of memory
environment	  Cannot allocate memory (os error 12)
environment	  process didn't exit successfully: `rustc --crate-name aprender_core` (signal: 9)
environment	  terminated by SIGKILL
environment	error: linker `cc` not found
environment	  sh: 1: cc: command not found
environment	error: package `half v2.7.0` cannot be built because it requires rustc 1.96 or newer
environment	  feature `edition2027` is not supported by this version of Cargo
build	error: could not compile `aprender-core` (lib) due to 1 previous error
build	error: failed to run custom build command for `clang-sys v1.8.1`
build	error: failed to parse manifest at `/tmp/cargo-installAbC123/Cargo.toml`
build	error: the lock file /tmp/cargo-installAbC123/Cargo.lock needs to be updated but --locked was passed to prevent this
build	error[E0432]: unresolved import `crate::format::gone`
none	   Compiling pkg-config v0.3.31
none	   Compiling openssl-sys v0.9.103
none	   Compiling openssl-probe v0.1.5
none	  Downloaded tokio v1.40.0
none	    Updating crates.io index
none	\e[1m\e[92m   Compiling\e[0m pkg-config v0.3.31
none	error: failed to compile `aprender v0.75.0`, intermediate artifacts can be found at `/tmp/cargo-installAbC123`.
none	To reuse those artifacts with a future compilation, set the environment variable `CARGO_TARGET_DIR` to that path.
none	  Installed package `aprender v0.75.0` (executable `apr`)
none	    Finished `release` profile [optimized] target(s) in 4m 02s
none	warning: unused import: `std::fmt`
none	SMOKE_CONTROL result=pass rc=0 prev=0.74.9
none	apr explain E001: E001: Invalid Magic Bytes
none	apr --version: apr 0.75.0
ROWS
}

case_table() {
  local esc cls text got n=0 bad=0 nfx=0 f base want
  esc=$(printf '\033')
  while IFS=$'\t' read -r cls text; do
    [ -n "$cls" ] || continue
    text=${text//\\e/$esc}
    got=$(printf '%s\n' "$text" | classify_stream)
    n=$((n + 1))
    if [ "$got" != "$cls" ]; then
      bad=$((bad + 1))
      printf 'FAIL  row %d: want %s got %s: %s\n' "$n" "$cls" "$got" "$text"
    fi
  done < <(case_rows)
  if [ -d "$FIXTURES" ]; then
    for f in "$FIXTURES"/*--*.txt; do
      [ -f "$f" ] || continue
      base=${f##*/}; want=${base%%--*}
      got=$(classify_log "$f")
      nfx=$((nfx + 1))
      if [ "$got" != "$want" ]; then
        bad=$((bad + 1)); printf 'FAIL  fixture %s: want %s got %s\n' "$base" "$want" "$got"
      fi
    done
  fi
  local npat=$(( ${#NET_PATTERNS[@]} + ${#HTTP_PATTERNS[@]} + ${#RESOLVER_PATTERNS[@]} + ${#ENV_PATTERNS[@]} + ${#BUILD_PATTERNS[@]} ))
  if [ "$n" -lt "$npat" ]; then bad=$((bad + 1)); echo "FAIL  vacuity: $n rows for $npat patterns"; fi
  if [ "$nfx" -lt 14 ]; then bad=$((bad + 1)); echo "FAIL  vacuity: $nfx fixtures in $FIXTURES (want 14 or more)"; fi
  if [ "$bad" -eq 0 ]; then
    echo "case-table: ALL PASS ($n rows for $npat patterns, $nfx fixtures)"; return 0
  fi
  echo "case-table: $bad FAILED ($n rows, $nfx fixtures)" >&2
  return 1
}

# ---------------------------------------------------------------- self-test

ST_FAIL=0
ST_CALLS=""
ST_SLEEPS=""
ST_PREV=0.74.9
# st_case name want_rc want_yanks want_andon want_out setup_fn args...
# ST_CALLS (global, "" = unchecked) is the expected number of container runs.
# ST_SLEEPS (global, "" = unchecked) is the expected list of host-side sleeps.
# `sleep` is stubbed: it records its argument and ADVANCES the stubbed clock, so
# no row really waits and the repeat-gap check sees the wait.
st_case() {
  local name=$1 want_rc=$2 want_yanks=$3 want_andon=$4 want_out=$5 setup=$6; shift 6
  local d rc out yanks andon calls sleeps ok=1
  own_root selftest || { ST_FAIL=$((ST_FAIL + 1)); return 0; }
  d=$OWN_ROOT_LAST
  : >"$d/yanks"; : >"$d/andon"; : >"$d/calls"; : >"$d/sleeps"
  out=$(
    set +e
    ST_DIR="$d"; export ST_DIR
    ANDON_INBOX="$d/andon"
    SCRATCH_ROOT="$d"
    unset SMOKE_RETRY_SLEEP SMOKE_REPEAT_WAIT
    ST_CLOCK=1000000
    now_epoch() { echo "$ST_CLOCK"; }
    sleep() { echo "$1" >>"$d/sleeps"; ST_CLOCK=$((ST_CLOCK + $1)); }
    previous_version() { echo "${ST_PREV-0.74.9}"; }
    do_yank() { echo "$1" >>"$d/yanks"; return 0; }
    detect_container() { echo /stub/docker; }
    "$setup"
    smoke_main "$@" 2>&1
    echo "RC=$?"
  )
  rc=${out##*RC=}
  yanks=$(tr '\n' ' ' <"$d/yanks" | sed 's/ $//')
  andon=$(cat "$d/andon")
  calls=$(wc -l <"$d/calls" | tr -d ' ')
  sleeps=$(tr '\n' ' ' <"$d/sleeps" | sed 's/ $//')
  if [ -n "$ST_SLEEPS" ] && [ "$sleeps" != "$ST_SLEEPS" ]; then ok=0; fi
  [ "$rc" = "$want_rc" ] || ok=0
  [ "$yanks" = "$want_yanks" ] || ok=0
  if [ -n "$want_andon" ]; then case "$andon" in *"$want_andon"*) : ;; *) ok=0 ;; esac
  else [ -z "$andon" ] || ok=0; fi
  if [ -n "$want_out" ]; then case "$out" in *"$want_out"*) : ;; *) ok=0 ;; esac; fi
  if [ -n "$ST_CALLS" ] && [ "$calls" != "$ST_CALLS" ]; then ok=0; fi
  if [ "$ok" = 1 ]; then
    printf 'PASS  %-56s rc=%s yanks=[%s] runs=%s waits=[%s]\n' "$name" "$rc" "$yanks" "$calls" "$sleeps"
  else
    printf 'FAIL  %-56s rc=%s (want %s) yanks=[%s] (want [%s]) runs=%s (want %s) waits=[%s] (want [%s]) andon=[%s]\n' \
      "$name" "$rc" "$want_rc" "$yanks" "$want_yanks" "$calls" "${ST_CALLS:-any}" "$sleeps" "${ST_SLEEPS:-any}" "$andon"
    printf '%s\n' "$out" | sed 's/^/      | /'
    ST_FAIL=$((ST_FAIL + 1))
  fi
  safe_rm_rf "$d" "$d" || ST_FAIL=$((ST_FAIL + 1))
  ST_CALLS=""; ST_SLEEPS=""; ST_PREV=0.74.9
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
exec env -i PATH="$ST_DIR/bin:/usr/bin:/bin" TMPDIR="$ST_DIR" SMOKE_V="$SMOKE_V" SMOKE_PREV="${SMOKE_PREV:-}" \
  STUB_CARGO="${STUB_CARGO:-ok}" STUB_APR="${STUB_APR:-ok}" STUB_CURL="${STUB_CURL:-ok}" STUB_CONTROL="${STUB_CONTROL:-ok}" ST_DIR="$ST_DIR" \
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
# Realistic cargo output: every install failure ends in the wrapper cargo prints.
wrapper() { # <version>
  echo "error: failed to compile \`aprender v$1\`, intermediate artifacts can be found at \`/tmp/cargo-installAbC123\`."
  echo "To reuse those artifacts with a future compilation, set the environment variable \`CARGO_TARGET_DIR\` to that path."
}
progress() {
  echo "    Updating crates.io index"
  echo "  Downloaded clang-sys v1.8.1"
  echo "   Compiling pkg-config v0.3.31"
  echo "   Compiling openssl-probe v0.1.5"
  echo "   Compiling clang-sys v1.8.1"
}
libclang() { # <version>: bindgen's build script in an image with no libclang
  progress
  echo "error: failed to run custom build command for \`clang-sys v1.8.1\`"
  echo ""
  echo "Caused by:"
  echo "  process didn't exit successfully: \`/tmp/cargo-installAbC123/release/build/clang-sys-0f3a/build-script-build\` (exit status: 101)"
  echo "  --- stderr"
  echo "  thread 'main' panicked at build.rs:56:39:"
  echo "  Unable to find libclang: \"couldn't find any valid shared libraries matching: ['libclang.so', 'libclang-*.so'], set the \`LIBCLANG_PATH\` environment variable to a path where one of these files can be found (invalid: [])\""
  echo "warning: build failed, waiting for other jobs to finish..."
  wrapper "$1"
}
case "$args" in
  *" --version $SMOKE_V "*) : ;;
  *" --version ${SMOKE_PREV:-NOPREV} "*) # the CONTROL install
    echo "control" >>"$ST_DIR/control"
    case "${STUB_CONTROL:-ok}" in
      fail) libclang "$SMOKE_PREV"; exit 101 ;;
      *) echo "  Installed package \`aprender v$SMOKE_PREV\`"; exit 0 ;;
    esac ;;
  *) echo "stub cargo: wrong version: $args"; exit 99 ;;
esac
mode=${STUB_CARGO:-ok}
if [ "$mode" = flaky ]; then # first install of SMOKE_V fails, later ones pass
  echo x >>"$ST_DIR/vcount"
  if [ "$(wc -l <"$ST_DIR/vcount")" -le 1 ]; then mode=libclang; else mode=ok; fi
fi
case "$mode" in
  compile)
    echo "   Compiling aprender-core v$SMOKE_V"
    echo "error[E0432]: unresolved import \`crate::format::gone\`"
    echo "error: could not compile \`aprender-core\` (lib) due to 1 previous error"
    wrapper "$SMOKE_V"; exit 101 ;;
  libclang) libclang "$SMOKE_V"; exit 101 ;;
  depsel) # a dependency whose sparse-index entry lags the root crate
    echo "    Updating crates.io index"
    echo "  Downloaded aprender v$SMOKE_V"
    wrapper "$SMOKE_V"
    echo ""
    echo "Caused by:"
    echo "  failed to select a version for the requirement \`aprender-core = \"=$SMOKE_V\"\`"
    echo "  candidate versions found which didn't match: 0.74.9, 0.74.8"
    echo "  location searched: crates.io index"
    echo "  required by package \`aprender v$SMOKE_V\`"
    exit 101 ;;
  curl28)
    echo "    Updating crates.io index"
    echo "  Downloading crates ..."
    wrapper "$SMOKE_V"
    echo ""
    echo "Caused by:"
    echo "  failed to download from \`https://static.crates.io/crates/aprender-core/$SMOKE_V/download\`"
    echo ""
    echo "Caused by:"
    echo "  [28] Timeout was reached (Operation timed out after 30001 milliseconds with 1048576 out of 9437184 bytes received)"
    exit 101 ;;
  http503)
    echo "    Updating crates.io index"
    echo "error: failed to get successful HTTP response from \`https://index.crates.io/ap/re/aprender-core\`, got 503"
    exit 101 ;;
  bsnet) # a build script that downloads, offline: a build-failure line AND a network line
    echo "   Compiling onnx-sys v0.4.0"
    echo "error: failed to run custom build command for \`onnx-sys v0.4.0\`"
    echo "  --- stderr"
    echo "  Error: Could not resolve host: github.com"
    wrapper "$SMOKE_V"; exit 101 ;;
  syslib)
    echo "   Compiling openssl-sys v0.9.103"
    echo "error: failed to run custom build command for \`openssl-sys v0.9.103\`"
    echo "  Could not find openssl via pkg-config"
    wrapper "$SMOKE_V"; exit 101 ;;
  disk)
    echo "error: could not compile \`aprender-core\` (lib)"
    echo "  No space left on device (os error 28)"
    wrapper "$SMOKE_V"; exit 101 ;;
  sig137) # cargo itself killed (OOM killer): the step is the machine's fault
    wrapper "$SMOKE_V"; exit 137 ;;
  boom) # the wrapper and an unknown cause: no build-failure line
    wrapper "$SMOKE_V"
    echo ""
    echo "Caused by:"
    echo "  boom"
    exit 101 ;;
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

  # 1000000 is the stubbed clock at the start of a row; publish times are epochs.
  st_cr_calls() { echo x >>"$ST_DIR/calls"; }
  st_failog() { # <step> <file> [control result]: what a container prints for a failed step
    printf 'SMOKE_CONTROL result=%s rc=0 prev=0.74.9\nSMOKE_FAIL step=%s rc=7\n' "${3:-pass}" "$1" >"$2"
  }
  st_fresh()  { published_epoch() { echo $((1000000 - 3600)); }; container_run() { st_cr_calls; st_failog verb-smoke "$3"; return 10; }; }
  st_stale()  { published_epoch() { echo $((1000000 - 90000)); }; container_run() { st_cr_calls; st_failog verb-smoke "$3"; return 10; }; }
  st_edge()   { published_epoch() { echo $((1000000 - 86400)); }; container_run() { st_cr_calls; st_failog verb-smoke "$3"; return 10; }; }
  st_nopub()  { published_epoch() { echo ""; }; container_run() { st_cr_calls; st_failog verb-smoke "$3"; return 10; }; }
  st_future() { published_epoch() { echo $((1000000 + 3600)); }; container_run() { st_cr_calls; st_failog verb-smoke "$3"; return 10; }; }
  st_skew()   { published_epoch() { echo $((1000000 + 100)); }; container_run() { st_cr_calls; st_failog verb-smoke "$3"; return 10; }; }
  st_ok()     { published_epoch() { echo $((1000000 - 3600)); }; container_run() { st_cr_calls; echo SMOKE_PASS >"$3"; return 0; }; }
  st_none()   { detect_container() { echo ""; }; published_epoch() { echo $((1000000 - 3600)); }; }
  st_w0env()  { st_fresh; SMOKE_REPEAT_WAIT=0; export SMOKE_REPEAT_WAIT; }
  st_wbadenv() { st_fresh; SMOKE_REPEAT_WAIT=abc; export SMOKE_REPEAT_WAIT; }
  st_noclock() { st_fresh; sleep() { echo "$1" >>"$ST_DIR/sleeps"; }; } # the wait did not happen
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
  st_r10unk() { st_rc 10 "SMOKE_CONTROL result=pass rc=0 prev=0.74.9
SMOKE_FAIL step=container rc=1"; }
  st_r0nopass() { st_rc 0 "exited zero but never said SMOKE_PASS"; }
  st_netin()  { st_rc 11 "SMOKE_NOTMEASURED step=index-probe rc=6"; }
  st_flaky()  { # first run daemon-down, second run passes
    published_epoch() { echo $((1000000 - 3600)); }
    container_run() { st_cr_calls; if [ "$(wc -l <"$ST_DIR/calls")" -ge 2 ]; then echo SMOKE_PASS >"$3"; return 0; fi; echo "daemon" >"$3"; return 125; }
  }
  # the in-container script text, run by the real container_run through the stub runtime
  st_fake() { # <STUB_RT> <STUB_CURL> <STUB_CARGO> <STUB_APR> [<STUB_CONTROL>]
    st_make_fakes
    published_epoch() { echo $((1000000 - 3600)); }
    detect_container() { echo "$ST_DIR/rt"; }
    STUB_RT=$1; STUB_CURL=$2; STUB_CARGO=$3; STUB_APR=$4; STUB_CONTROL=${5:-ok}
    export STUB_RT STUB_CURL STUB_CARGO STUB_APR STUB_CONTROL
    container_run() { st_cr_calls; real_container_run "$@"; }
  }
  st_f_ok()      { st_fake none ok ok ok; }
  st_f_compile() { st_fake none ok compile ok; }
  st_f_compile_ctlfail() { st_fake none ok compile ok fail; }
  st_f_libclang_ctlfail() { st_fake none ok libclang ok fail; }
  st_f_libclang_ctlok()   { st_fake none ok libclang ok ok; }
  st_f_expfail_ctlfail()  { st_fake none ok ok explainfail fail; }
  st_f_depsel()  { st_fake none ok depsel ok ok; }
  st_f_curl28()  { st_fake none ok curl28 ok ok; }
  st_f_http503() { st_fake none ok http503 ok ok; }
  st_f_bsnet()   { st_fake none ok bsnet ok ok; }
  st_f_flaky()   { st_fake none ok flaky ok ok; }
  st_f_syslib()  { st_fake none ok syslib ok; }
  st_f_disk()    { st_fake none ok disk ok; }
  st_f_boom()    { st_fake none ok boom ok; }
  st_f_sig137()  { st_fake none ok sig137 ok; }
  # a scripted sequence of container outcomes, one word of ST_SEQ per run:
  # pass | nm | f:<step> (control pass) | cf:<step> (control fail) |
  # cu:<step> (control undeterminable) | cm:<step> (no control line)
  st_seq() {
    published_epoch() { echo $((1000000 - 3600)); }
    container_run() {
      local n tok
      st_cr_calls; n=$(wc -l <"$ST_DIR/calls" | tr -d ' ')
      tok=$(printf '%s\n' "$ST_SEQ" | tr ' ' '\n' | sed -n "${n}p")
      case "$tok" in
        pass) echo SMOKE_PASS >"$3"; return 0 ;;
        f:*) st_failog "${tok#f:}" "$3" pass; return 10 ;;
        cf:*) st_failog "${tok#cf:}" "$3" fail; return 10 ;;
        cu:*) st_failog "${tok#cu:}" "$3" undeterminable; return 10 ;;
        cm:*) echo "SMOKE_FAIL step=${tok#cm:} rc=7" >"$3"; return 10 ;;
        *) echo "SMOKE_NOTMEASURED step=index-probe rc=6" >"$3"; return 11 ;;
      esac
    }
  }
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
  local YK="SMOKE_RESULT YANK version=0.75.0 step=verb-smoke action=yanked"
  ST_CALLS=1; st_case "smoke pass"                          0 ""       ""   "SMOKE_RESULT PASS version=0.75.0" st_ok $A
  ST_CALLS=2; ST_SLEEPS=120; st_case "same fail twice, 120 s apart, control ok -> YANK" 1 "0.75.0" "| aprender#7000 aprender@0.75.0 | ANDON yank: verb-smoke; log " "$YK" st_fresh $A
  # (2) the repeat: one failure is never enough
  ST_SEQ="f:verb-smoke pass"
  ST_CALLS=2; ST_SLEEPS=120; st_case "repeat: fail then pass -> NOT_MEASURED" 3 "" "not_measured" "failed then passed" st_seq $A
  ST_SEQ="f:verb-smoke nm"
  ST_CALLS=2; ST_SLEEPS=120; st_case "repeat: fail then not_measured -> NOT_MEASURED" 3 "" "not_measured" "unconfirmed" st_seq $A
  ST_SEQ="f:apr-version f:verb-smoke"
  ST_CALLS=2; ST_SLEEPS=120; st_case "repeat: fail then a DIFFERENT fail -> NOT_MEASURED" 3 "" "not_measured" "unconfirmed" st_seq $A
  ST_SEQ="nm f:verb-smoke f:verb-smoke"
  ST_CALLS=3; ST_SLEEPS="20 120"; st_case "repeat: nm, fail, same fail -> YANK" 1 "0.75.0" "ANDON yank: verb-smoke" "$YK" st_seq $A
  ST_SEQ="nm nm"
  ST_CALLS=2; ST_SLEEPS=20; st_case "nm twice -> NOT_MEASURED"   3 "" "not_measured" "SMOKE_RESULT NOT_MEASURED" st_seq $A
  ST_SEQ="f:apr-version f:apr-version"
  ST_CALLS=2; st_case "repeat: apr --version fails twice -> YANK" 1 "0.75.0" "ANDON yank: apr-version" "step=apr-version action=yanked" st_seq $A
  # (1) the control, on every failed step, in both attempts
  ST_SEQ="cf:verb-smoke cf:verb-smoke"
  ST_CALLS=2; st_case "control fails -> NOT_MEASURED"         3 "" "not_measured" "reason=control-failed" st_seq $A
  ST_SEQ="f:verb-smoke cf:verb-smoke"
  ST_CALLS=2; st_case "control fails on the repeat -> NOT_MEASURED" 3 "" "not_measured" "unconfirmed" st_seq $A
  ST_SEQ="cu:verb-smoke cu:verb-smoke"
  ST_CALLS=2; st_case "control undeterminable -> NOT_MEASURED" 3 "" "not_measured" "reason=control-undeterminable" st_seq $A
  ST_SEQ="cm:verb-smoke cm:verb-smoke"
  ST_CALLS=2; st_case "no control line -> NOT_MEASURED"       3 "" "not_measured" "reason=control-missing" st_seq $A
  # the wait: default 120, a parameter (0 allowed), and measured on the clock
  ST_CALLS=2; ST_SLEEPS=0;   st_case "SMOKE_REPEAT_WAIT=0 -> YANK, no wait" 1 "0.75.0" "ANDON yank" "$YK" st_w0env $A
  ST_CALLS=2; ST_SLEEPS=0;   st_case "--repeat-wait 0 -> YANK, no wait"   1 "0.75.0" "ANDON yank" "$YK" st_fresh $A --repeat-wait 0
  ST_CALLS=2; ST_SLEEPS=300; st_case "--repeat-wait 300"                  1 "0.75.0" "ANDON yank" "$YK" st_fresh $A --repeat-wait 300
  ST_CALLS=2; ST_SLEEPS=120; st_case "repeat came too soon (clock) -> NOT_MEASURED" 3 "" "not_measured" "under the 120s wait" st_noclock $A
  st_case "--repeat-wait abc"                 2 "" "" "whole number of seconds" st_fresh $A --repeat-wait abc
  st_case "--repeat-wait -5"                  2 "" "" "whole number of seconds" st_fresh $A --repeat-wait -5
  st_case "SMOKE_REPEAT_WAIT=abc"             2 "" "" "whole number of seconds" st_wbadenv $A
  # (5) the 24 h / train rule: a confirmed failure that may not be yanked
  st_case "fail at 25 h -> refused"           1 "" "ANDON yank-refused" "action=refused" st_stale $A
  st_case "fail at exactly 24 h -> refused"   1 "" "ANDON yank-refused" "action=refused" st_edge $A
  st_case "version mismatch -> refused"       1 "" "ANDON yank-refused" "not the version this release run published" st_fresh --version 0.75.0 --train-published 0.74.9
  st_case "no --train-published -> refused"   1 "" "ANDON yank-refused" "no --train-published" st_fresh --version 0.75.0
  st_case "publish time unknown -> refused"   1 "" "ANDON yank-refused" "unknown" st_nopub $A
  st_case "publish time 1 h in the future"    1 "" "ANDON yank-refused" "in the future" st_future $A
  st_case "publish time 100 s ahead (skew ok)" 1 "0.75.0" "ANDON yank" "$YK" st_skew $A
  st_case "dry-run prints WOULD_YANK"         1 "" "" "WOULD_YANK aprender@0.75.0" st_fresh $A --dry-run
  st_case "dry-run verdict line"              1 "" "" "action=dry-run" st_fresh $A --dry-run
  st_case "--published-at without --dry-run"  2 "" "" "only with --dry-run" st_fresh $A --published-at 2026-11-13T09:00:00Z
  st_case "--published-at with --dry-run"     1 "" "" "WOULD_YANK" st_fresh $A --published-at 2026-01-01T00:00:00Z --dry-run
  st_case "bad version arg"                   2 "" "" "usage" st_ok --version '0.75.0;$(id)' --train-published 0.75.0
  st_case "unknown argument"                  2 "" "" "unknown argument" st_ok --bogus
  local f
  for f in --version --train-published --published-at --container-cmd --timeout-min --not-measured --repeat-wait; do
    st_case "trailing $f with no value"       2 "" "" "needs a value" st_ok $A "$f"
  done
  st_case "--timeout-min not a number"        2 "" "" "positive integer" st_ok $A --timeout-min abc
  # NOT_MEASURED: no yank, rc 3, an andon line, retried once (2 runs)
  st_case "no container runtime"              3 "" "not_measured" "no container runtime" st_none $A
  ST_CALLS=2; st_case "daemon down rc125"     3 "" "not_measured" "SMOKE_RESULT NOT_MEASURED" st_daemon $A
  ST_CALLS=2; st_case "image pull failure rc125" 3 "" "not_measured" "" st_pull $A
  ST_CALLS=2; st_case "runtime rc126"         3 "" "not_measured" "" st_r126 $A
  ST_CALLS=2; st_case "runtime rc127"         3 "" "not_measured" "" st_r127 $A
  ST_CALLS=2; st_case "container killed rc137 (OOM)" 3 "" "not_measured" "" st_r137 $A
  ST_CALLS=2; st_case "timeout rc124"         3 "" "not_measured" "timeout" st_r124 $A
  ST_CALLS=2; st_case "no marker, rc 9"       3 "" "not_measured" "" st_nomark $A
  ST_CALLS=2; st_case "rc 10 without a marker" 3 "" "not_measured" "" st_r10nomark $A
  ST_CALLS=2; st_case "rc 10, unknown step 'container'" 3 "" "not_measured" "" st_r10unk $A
  ST_CALLS=2; st_case "rc 0 without SMOKE_PASS" 3 "" "not_measured" "" st_r0nopass $A
  ST_CALLS=2; st_case "index unreachable inside container" 3 "" "not_measured" "index-probe" st_netin $A
  ST_CALLS=2; st_case "retry: daemon down then pass" 0 "" "" "SMOKE_RESULT PASS" st_flaky $A
  st_case "--not-measured reason (index wait timeout)" 3 "" "not_measured" "SMOKE_RESULT NOT_MEASURED" st_refuse $A --not-measured "index wait timed out"
  # the REAL in-container script, under stub cargo/curl/apr (no container, no network)
  ST_CALLS=1; st_case "in-container: all steps pass"  0 "" "" "SMOKE_RESULT PASS" st_f_ok $A
  ST_CALLS=2; ST_SLEEPS=120; st_case "in-container: rustc error, control ok, repeats -> YANK" 1 "0.75.0" "ANDON yank: cargo-install" "step=cargo-install action=yanked" st_f_compile $A
  ST_CALLS=2; st_case "in-container: rustc error, control FAILS"   3 "" "not_measured" "reason=control-failed" st_f_compile_ctlfail $A
  ST_CALLS=2; st_case "in-container: libclang, control FAILS"      3 "" "not_measured" "reason=control-failed" st_f_libclang_ctlfail $A
  ST_CALLS=2; ST_SLEEPS=120; st_case "in-container: libclang, control ok, repeats -> YANK" 1 "0.75.0" "ANDON yank: cargo-install" "step=cargo-install action=yanked" st_f_libclang_ctlok $A
  ST_CALLS=2; st_case "in-container: verb smoke fails, control FAILS" 3 "" "not_measured" "reason=control-failed" st_f_expfail_ctlfail $A
  ST_PREV=""
  ST_CALLS=2; st_case "in-container: libclang, control undeterminable" 3 "" "not_measured" "reason=control-undeterminable" st_f_libclang_ctlok $A
  ST_CALLS=2; ST_SLEEPS=120; st_case "in-container: install fails once, then passes" 3 "" "not_measured" "failed then passed" st_f_flaky $A
  ST_CALLS=2; st_case "in-container: dep 'failed to select a version'" 3 "" "not_measured" "cargo-install-resolver" st_f_depsel $A
  ST_CALLS=2; st_case "in-container: curl [28] Timeout was reached" 3 "" "not_measured" "cargo-install-network" st_f_curl28 $A
  ST_CALLS=2; st_case "in-container: index HTTP 503"   3 "" "not_measured" "cargo-install-http" st_f_http503 $A
  ST_CALLS=2; st_case "in-container: build script offline" 3 "" "not_measured" "cargo-install-network" st_f_bsnet $A
  ST_CALLS=2; st_case "in-container: missing system library" 3 "" "not_measured" "cargo-install-environment" st_f_syslib $A
  ST_CALLS=2; st_case "in-container: disk full"        3 "" "not_measured" "cargo-install-environment" st_f_disk $A
  ST_CALLS=2; st_case "in-container: only the failed-to-compile wrapper" 3 "" "not_measured" "cargo-install-unclassified" st_f_boom $A
  ST_CALLS=2; st_case "in-container: cargo killed rc 137"  3 "" "not_measured" "reason=cargo-install (reported by the container)" st_f_sig137 $A
  ST_CALLS=2; st_case "in-container: index probe fails" 3 "" "not_measured" "index-probe" st_f_curl $A
  ST_CALLS=2; st_case "in-container: version not in index" 3 "" "not_measured" "version-not-in-index" st_f_noidx $A
  ST_CALLS=2; st_case "in-container: apr 0.72.60 vs 0.72.6" 1 "0.72.6" "ANDON yank" "apr-version-mismatch" st_f_ver60 --version 0.72.6 --train-published 0.72.6
  ST_CALLS=2; st_case "in-container: apr --version rc 139" 1 "0.75.0" "ANDON yank" "step=apr-version " st_f_verfail $A
  ST_CALLS=2; st_case "in-container: verb smoke rc 101"    1 "0.75.0" "ANDON yank" "step=verb-smoke " st_f_expfail $A
  ST_CALLS=2; st_case "in-container: verb smoke wrong output" 1 "0.75.0" "ANDON yank" "verb-smoke-output" st_f_expwrong $A
  ST_CALLS=2; st_case "runtime: daemon down (stub runtime)" 3 "" "not_measured" "" st_f_daemon $A
  ST_CALLS=2; st_case "runtime: pull failure (stub runtime)" 3 "" "not_measured" "" st_f_pull $A
  ST_CALLS=2; st_case "runtime: hang -> real timeout"  3 "" "not_measured" "timeout (1s)" st_f_hang $A

  # published_epoch against canned crates.io documents (subshell: count from output)
  local doc ub
  doc='{"version":{"num":"0.75.0","created_at":"2026-11-13T08:00:00.123456+00:00","yanked":false}}'
  ub=$(
    crates_io_json() { printf '%s' "$doc"; }
    DRY_RUN=0; PUBLISHED_AT=""
    [ "$(published_epoch 0.75.0)" = 1794556800 ] || echo "created_at->epoch"
    [ -z "$(published_epoch 0.75.1)" ] || echo "another-version-doc"
    PUBLISHED_AT="2020-01-01T00:00:00Z"
    [ "$(published_epoch 0.75.0)" = 1794556800 ] || echo "published-at-honoured-live"
    DRY_RUN=1
    [ "$(published_epoch 0.75.0)" = 1577836800 ] || echo "published-at-ignored-dry-run"
    crates_io_json() { printf ''; }
    DRY_RUN=0; PUBLISHED_AT=""
    [ -z "$(published_epoch 0.75.0)" ] || echo "unreachable-not-unknown"
  )
  if [ -z "$ub" ]; then echo "PASS  published_epoch unit rows (5)"
  else echo "FAIL  published_epoch unit rows: $ub"; ST_FAIL=$((ST_FAIL + 1)); fi

  # previous_version (the CONTROL version) against a canned sparse-index document
  ST_IDX='{"name":"aprender","vers":"0.74.8","deps":[],"cksum":"a","features":{},"yanked":false}
{"name":"aprender","vers":"0.74.9","deps":[],"cksum":"b","features":{},"yanked":true}
{"name":"aprender","vers":"0.74.10-rc.1","deps":[],"cksum":"c","features":{},"yanked":false}
{"name":"aprender","vers":"0.75.0","deps":[],"cksum":"d","features":{},"yanked":false}
{"name":"aprender","vers":"0.75.1","deps":[],"cksum":"e","features":{},"yanked":false}'
  index_doc() { printf '%s\n' "$ST_IDX"; }
  st_unit "control: skips yanked and prerelease"      0.74.8 "$(previous_version 0.75.0)"
  st_unit "control: newest below, not the newest"     0.75.0 "$(previous_version 0.75.1)"
  st_unit "control: none below the first version"     ""     "$(previous_version 0.74.8)"
  st_unit "control: version absent from the index"    0.75.1 "$(previous_version 0.75.2)"
  ST_IDX=""
  st_unit "control: index unreachable -> none"        ""     "$(previous_version 0.75.0)"

  # no delete outside safe_rm_rf
  local nrm
  nrm=$(grep -cE '(^[[:space:]]*|[;&|(][[:space:]]*)rm +-' "$SELF") || nrm=0
  if [ "$nrm" = 1 ]; then echo "PASS  only safe_rm_rf calls rm"
  else echo "FAIL  $nrm rm calls (want 1: safe_rm_rf)"; ST_FAIL=$((ST_FAIL + 1)); fi

  # safe_rm_rf: every refusal leaves its target in place. Two registered roots
  # (rr, dc) and one unregistered (ur), all made by mktemp -d in SCRATCH_ROOT.
  local rr dc ur rrc rm_ok=1 rows=0
  own_root rmtest; rr=$OWN_ROOT_LAST
  own_root decoy; dc=$OWN_ROOT_LAST
  ur=$(mktemp -d "$SCRATCH_ROOT/pps-unreg.XXXXXX")
  mkdir "$rr/sub" "$rr/plain"; : >"$dc/keep"; : >"$ur/keep"
  ln -s "$dc" "$rr/link"
  ln -s "$dc" "$rr/pps-lnk"
  ln -s "$rr" "$dc/into"
  srr() { # <want rc> <must-exist> <label> -- <safe_rm_rf args>  (runs in a subshell)
    local want=$1 keep=$2 label=$3 got=0; shift 4
    rows=$((rows + 1))
    ( safe_rm_rf "$@" ) 2>/dev/null || got=$?
    if [ "$got" != "$want" ] || { [ -n "$keep" ] && [ ! -e "$keep" ]; }; then
      rm_ok=0; echo "FAIL  safe_rm_rf: $label (rc=$got, want $want)"
    fi
  }
  srr 1 "$dc/keep"  "path outside the root"            -- "$dc/keep" "$rr"
  srr 1 "$rr/sub"   "path outside the root, resolving into it" -- "$dc/into/sub" "$rr"
  srr 1 "$ur/keep"  "root not registered by this run"  -- "$ur/keep" "$ur"
  srr 1 ""          "empty path"                       -- "" "$rr"
  srr 1 "$rr"       "empty root"                       -- "$rr" ""
  srr 1 "$dc/keep"  ".. component"                     -- "$rr/sub/../../${dc##*/}/keep" "$rr"
  srr 1 "$rr/sub"   ". component"                      -- "$rr/./sub" "$rr"
  srr 1 ""          "/"                                -- "/" "$rr"
  srr 1 "$dc/keep"  "path is a symlink"                -- "$rr/link" "$rr"
  srr 1 "$dc/keep"  "path through a symlinked dir"     -- "$rr/link/keep" "$rr"
  rows=$((rows + 1)); rrc=0
  ( OWN_ROOTS="|$rr/pps-lnk|"; SCRATCH_ROOT=$rr; safe_rm_rf "$rr/pps-lnk/keep" "$rr/pps-lnk" ) 2>/dev/null || rrc=$?
  { [ "$rrc" = 1 ] && [ -e "$dc/keep" ]; } || { rm_ok=0; echo "FAIL  safe_rm_rf: root is a symlink (rc=$rrc)"; }
  rows=$((rows + 1)); rrc=0
  ( OWN_ROOTS="|$rr/plain|"; SCRATCH_ROOT=$rr; safe_rm_rf "$rr/plain" "$rr/plain" ) 2>/dev/null || rrc=$?
  { [ "$rrc" = 1 ] && [ -d "$rr/plain" ]; } || { rm_ok=0; echo "FAIL  safe_rm_rf: registered root not named pps-* (rc=$rrc)"; }
  rows=$((rows + 1)); rrc=0
  ( owned_by_me() { [ "$1" != "$rr" ]; }; safe_rm_rf "$rr/sub" "$rr" ) 2>/dev/null || rrc=$?
  { [ "$rrc" = 1 ] && [ -d "$rr/sub" ]; } || { rm_ok=0; echo "FAIL  safe_rm_rf: root not owned by the current uid (rc=$rrc)"; }
  rows=$((rows + 1)); rrc=0
  ( owned_by_me() { [ "$1" != "$rr/sub" ]; }; safe_rm_rf "$rr/sub" "$rr" ) 2>/dev/null || rrc=$?
  { [ "$rrc" = 1 ] && [ -d "$rr/sub" ]; } || { rm_ok=0; echo "FAIL  safe_rm_rf: path not owned by the current uid (rc=$rrc)"; }
  # what it must remove
  rows=$((rows + 1)); rrc=0; safe_rm_rf "$rr/sub" "$rr" || rrc=$?
  { [ "$rrc" = 0 ] && [ ! -e "$rr/sub" ] && [ -d "$rr" ]; } || { rm_ok=0; echo "FAIL  safe_rm_rf: a path inside the root is removed (rc=$rrc)"; }
  rows=$((rows + 1)); rrc=0; safe_rm_rf "$rr" "$rr" || rrc=$?
  { [ "$rrc" = 0 ] && [ ! -e "$rr" ] && [ -e "$dc/keep" ]; } || { rm_ok=0; echo "FAIL  safe_rm_rf: the root is removed, symlink targets kept (rc=$rrc)"; }
  OWN_ROOTS="$OWN_ROOTS$ur|"
  safe_rm_rf "$ur" "$ur" || rm_ok=0
  safe_rm_rf "$dc" "$dc" || rm_ok=0
  if [ "$rm_ok" = 1 ]; then echo "PASS  safe_rm_rf: $rows rows (refuses symlinks, foreign uid, unregistered or outside roots; removes inside)"
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
  local dr="$root/scripts/cascade-drain.sh"
  if grep -qF 'crc=${PIPESTATUS[0]}' "$dr" \
     && grep -qF "grep '^SMOKE_RESULT '" "$dr" \
     && grep -qF 'yanked\":true' "$dr" \
     && grep -A2 -F '"SMOKE_RESULT YANK "*)' "$dr" | grep -q 'exit 4' \
     && grep -A2 -F '"SMOKE_RESULT NOT_MEASURED "*)' "$dr" | grep -q 'exit 5' \
     && ! grep -qE 'cascade-publish\.sh \) 2>&1 *\| *grep' "$dr"; then
    echo "PASS  cascade-drain.sh keeps the cascade rc, maps YANK->4 and NOT_MEASURED->5, does not count a yanked version"
  else echo "FAIL  cascade-drain.sh loses the rc / mis-maps a smoke verdict / counts a yanked version"; ST_FAIL=$((ST_FAIL + 1)); fi
  local nmk
  nmk=$(grep -cE '^[^#]*mktemp' "$dr") || nmk=0
  if [ "$nmk" = 1 ] && grep -qF 'drain_root=$(mktemp -d ' "$dr" \
     && grep -qF '| tail -c "$DRAIN_LOG_CAP" >"$cascade_log"' "$dr" \
     && grep -qF 'mv -f -- "$cascade_log" "$cascade_log.prev"' "$dr" \
     && ! grep -qE '(^|[;&|(])[[:space:]]*rm +-' "$dr"; then
    echo "PASS  cascade-drain.sh: one mktemp -d root, log capped and rotated inside it, no rm"
  else echo "FAIL  cascade-drain.sh: $nmk mktemp calls / log uncapped or not rotated / an rm"; ST_FAIL=$((ST_FAIL + 1)); fi
  if [ "$ST_FAIL" -eq 0 ]; then echo "self-test: ALL PASS"; return 0; fi
  echo "self-test: $ST_FAIL FAILED" >&2
  return 1
}

# ---------------------------------------------------------------- mutants
# Each edited copy must turn --case-table or --self-test red.
mutants() {
  local d n=0 bad=0 rc target arr ln lines pat
  own_root mutants || return 1
  d=$OWN_ROOT_LAST
  mut() { # name file sed-expression
    local name=$1 file=$2 expr=$3
    safe_rm_rf "$d/scripts" "$d" || { bad=$((bad + 1)); return 0; }
    mkdir -p "$d/scripts"
    cp "$SELF" "$d/scripts/post_publish_smoke.sh"
    cp "$(dirname "$SELF")/cascade-publish.sh" "$d/scripts/cascade-publish.sh"
    cp "$(dirname "$SELF")/cascade-drain.sh" "$d/scripts/cascade-drain.sh"
    cp -R "$(dirname "$SELF")/fixtures" "$d/scripts/fixtures"
    cp "$ROOT_DIR/rust-toolchain.toml" "$d/rust-toolchain.toml"
    target="$d/scripts/$file"
    cp "$target" "$d/orig"
    sed -i "$expr" "$target"
    if cmp -s "$d/orig" "$target"; then
      echo "MUTANT $name: NOT APPLIED (pattern drifted)"; bad=$((bad + 1)); return 0
    fi
    n=$((n + 1))
    rc=0
    bash "$d/scripts/post_publish_smoke.sh" --case-table >/dev/null 2>&1 || rc=$?
    if [ "$rc" -ne 0 ]; then echo "MUTANT $name: killed (case-table rc=$rc)"; return 0; fi
    bash "$d/scripts/post_publish_smoke.sh" --self-test >/dev/null 2>&1 || rc=$?
    if [ "$rc" -ne 0 ]; then echo "MUTANT $name: killed (self-test rc=$rc)"
    else echo "MUTANT $name: SURVIVED"; bad=$((bad + 1)); fi
  }
  local P=post_publish_smoke.sh
  # B1: the control, the repeat, the classifier
  mut "B1-control-fail-ok"           $P 's/^    fail) echo "nm control-failed"; return 0 ;;$/    fail) : ;;/'
  mut "B1-control-undeterminable-ok" $P 's/^    undeterminable) echo "nm control-undeterminable"; return 0 ;;$/    undeterminable) : ;;/'
  mut "B1-control-missing-ok"        $P 's/^    \*) echo "nm control-missing"; return 0 ;;$/    *) : ;;/'
  mut "B1-control-not-run"           $P 's/^  control$/  :/'
  mut "B1-control-always-pass"       $P 's/if \[ "\$crc" -eq 0 \]; then cres=pass; else cres=fail; fi/cres=pass/'
  mut "B1-no-repeat"                 $P 's/^  t2=\$(now_epoch); smoke_attempt$/  t2=$(now_epoch)/'
  mut "B1-no-repeat-sleep"           $P 's/^  sleep "\$repeat_wait"$/  :/'
  mut "B1-gap-check-removed"         $P 's/^  if \[ "\$gap" -lt "\$repeat_wait" \]; then$/  if false; then/'
  mut "B1-default-wait-0"            $P 's/^REPEAT_WAIT_DEFAULT=120$/REPEAT_WAIT_DEFAULT=0/'
  mut "B1-wait-0-refused"            $P 's/ \^\[0-9\]+\$ \]\]; then$/ ^[1-9][0-9]*$ ]]; then/'
  mut "B1-flaky-is-pass"             $P 's/^      report_not_measured "\$version" "unconfirmed: \$step failed then passed.*$/      echo "SMOKE_RESULT PASS version=$version"; return 0/'
  mut "B1-different-step-yanks"      $P 's/^    "\$first") : ;;$/    fail*) : ;;/'
  mut "B1-nm-class-yanks"            $P 's/^    network|http|resolver|environment) echo "nm \$step-\$cls"; return 0 ;;$/    NOCLASS) : ;;/'
  mut "B1-drop-network-class"        $P 's/^  if pat_any "\$diag" "\${NET_PATTERNS\[@\]}"; then echo network; return 0; fi$//'
  mut "B1-drop-http-class"           $P 's/^  if pat_any "\$diag" "\${HTTP_PATTERNS\[@\]}"; then echo http; return 0; fi$//'
  mut "B1-drop-resolver-class"       $P 's/^  if pat_any "\$diag" "\${RESOLVER_PATTERNS\[@\]}"; then echo resolver; return 0; fi$//'
  mut "B1-drop-environment-class"    $P 's/^  if pat_any "\$diag" "\${ENV_PATTERNS\[@\]}"; then echo environment; return 0; fi$//'
  mut "B1-unclassified-install-yanks" $P 's/if \[ "\$step" = cargo-install \] && \[ "\$cls" != build \]; then/if false; then/'
  mut "B1-no-ansi-strip"             $P "s/^  esc=\$(printf '\\\\033')\$/  esc=NOESC/"
  mut "B1-no-progress-filter"        $P 's/| grep -v -E "\$PROGRESS_RE" || true)/|| true)/'
  mut "B1-wrapper-is-build"          $P "s/^  'could not compile'\$/  'could not compile'\\n  'failed to compile'/"
  mut "B1-unmarked-failure-yanks"    $P 's/^    \*) echo "nm no step marker, runtime rc=\$rc"; return 0 ;;$/    *) echo "fail cargo-install"; return 0 ;;/'
  mut "B1-daemon-down-yanks"         $P 's/^    125) echo "nm container runtime error.*$/    125) echo "fail verb-smoke"; return 0 ;;/'
  mut "B1-rc0-without-pass-ok"       $P 's/else echo "nm exit 0 without a SMOKE_PASS marker"; fi$/else echo "pass"; fi/'
  mut "B1-unknown-step-yanks"        $P 's/^    \*) echo "nm exit 10 without a known SMOKE_FAIL step marker.*$/    *) : ;;/'
  mut "B1-no-retry-of-nm"            $P 's/^      t1=\$(now_epoch); smoke_attempt ;;$/      : ;;/'
  mut "B1-killed-step-is-fail"       $P 's/^  case "\$2" in 124|137) notmeasured "\$1" "rc=\$2 (killed)" ;; esac$/  :/'
  mut "B1-no-timeout"                $P 's/^  timeout "\$TIMEOUT_SECS" "\$rt" run/  "$rt" run/'
  # the yank rule
  mut "flip-24h-comparison"          $P 's/if \[ "\$age" -lt "\$MAX_AGE_SECS" \]/if [ "$age" -gt "$MAX_AGE_SECS" ]/'
  mut "drop-version-equality"        $P 's/if \[ "\$train_v" != "\$v" \]/if false/'
  mut "skip-train-check"             $P 's/if \[ -z "\$train_v" \]; then echo "no --train-published version given"; return 1; fi//'
  mut "R6-future-time-allowed"       $P 's/if \[ "\$age" -lt "-\$SKEW_SECS" \]; then/if false; then/'
  mut "swallow-smoke-rc"             $P 's/|| SM_RC=\$?$/|| SM_RC=0/'
  mut "yank-in-dry-run"              $P 's/^    if \[ "\$DRY_RUN" = 1 \]; then$/    if false; then/'
  mut "refused-is-silent"            $P 's/^  andon "\$(andon_line "\$version" "yank-refused".*$/  :/'
  mut "no-runtime-is-pass"           $P 's/^    return 3$/    return 0/'
  mut "R3-crate-from-env"            $P 's/^CRATE="aprender"$/CRATE="${SMOKE_CRATE:-aprender}"/'
  mut "R4-version-check-true"        $P 's/\[ "\$ok" -eq 1 \] || failed apr-version-mismatch 1/:/'
  mut "R4b-substring-version"        $P 's/for w in \$out; do \[ "\$w" = "\$SMOKE_V" \] \&\& ok=1; done/case "$out" in *"$SMOKE_V"*) ok=1 ;; esac/'
  mut "R7-drop-locked"               $P 's/ --version "\$SMOKE_V" --locked >"\$ilog"/ --version "$SMOKE_V" >"$ilog"/'
  mut "C5-control-keeps-yanked"      $P 's/grep -q .*"yanked":false.* <<<"\$line" || continue/:/'
  mut "B2-published-at-anywhere"     $P 's/if \[ -n "\$PUBLISHED_AT" \] \&\& \[ "\$DRY_RUN" != 1 \]; then/if false; then/'
  mut "B2-published-at-in-lookup"    $P 's/if \[ -n "\$PUBLISHED_AT" \] \&\& \[ "\$DRY_RUN" = 1 \]; then/if [ -n "$PUBLISHED_AT" ]; then/'
  mut "B2-skip-num-check"            $P 's/^  grep -qE "\\"num.*|| return 0$//'
  mut "S5-flag-without-value-loops"  $P 's/\[ \$# -ge 2 \] || { echo "usage: \$1 needs a value" >\&2; return 2; } ;;/: ;;/'
  # safe_rm_rf
  mut "R8-rm-outside-root"           $P 's/^    "\$r"|"\$r"\/\*) : ;;$/    *) : ;;/'
  mut "R9-rm-dotdot-allowed"         $P 's/^    \*\/\.\.\/\*|\*\/\.\/\*) srr_refuse/    NOMATCH) srr_refuse/'
  mut "R10-rm-root-not-pps"          $P 's/^    "\${SCRATCH_ROOT:?}"\/pps-\*) : ;;$/    *) : ;;/'
  mut "R11-rm-symlink-root"          $P 's/^  if \[ -L "\$r" \]; then srr_refuse/  if false; then srr_refuse/'
  mut "R12-rm-foreign-root"          $P 's/^  if ! owned_by_me "\$r"; then srr_refuse/  if false; then srr_refuse/'
  mut "R13-rm-unregistered-root"     $P 's/^    \*"|\$r|"\*) : ;;$/    *) : ;;/'
  mut "R14-rm-symlink-path"          $P 's/^  if \[ -L "\$p" \]; then srr_refuse/  if false; then srr_refuse/'
  mut "R15-rm-foreign-path"          $P 's/^  if ! owned_by_me "\$p"; then srr_refuse/  if false; then srr_refuse/'
  mut "R16-rm-no-physical-check"     $P 's/^    \*) srr_refuse "\$p resolves outside \$rp"; return 1 ;;$/    *) : ;;/'
  # the callers
  mut "B3-index-timeout-falls-through" cascade-publish.sh 's/--not-measured "sparse index/--version "$TARGET_VERSION" "sparse index/'
  mut "B4-drain-loses-rc"            cascade-drain.sh 's/^    crc=\${PIPESTATUS\[0\]}$/    crc=0/'
  mut "B4-drain-ignores-smoke"       cascade-drain.sh 's/grep '"'"'^SMOKE_RESULT /grep '"'"'^NOSUCH_RESULT /'
  mut "B4-drain-counts-yanked"       cascade-drain.sh 's/\*\\"yanked\\":true\*) exit 0/*\\"yanked\\":nope*) exit 0/'
  mut "B4-drain-yank-not-failure"    cascade-drain.sh '/"SMOKE_RESULT YANK "\*)/,+2 s/exit 4/:/'
  mut "B4-drain-nm-not-failure"      cascade-drain.sh '/"SMOKE_RESULT NOT_MEASURED "\*)/,+2 s/exit 5/:/'
  mut "B5-drain-log-uncapped"        cascade-drain.sh 's/| tail -c "\$DRAIN_LOG_CAP" >"\$cascade_log"/>"$cascade_log"/'
  mut "B5-drain-log-not-rotated"     cascade-drain.sh 's/mv -f -- "\$cascade_log" "\$cascade_log.prev"/:/'
  mut "B5-drain-log-per-pass"        cascade-drain.sh 's/^cascade_log="\$drain_root\/cascade.log"$/cascade_log=$(mktemp "${TMPDIR:-\/tmp}\/cascade-drain.XXXXXX")/'
  mut "B5-drain-rm-config"           cascade-drain.sh 's/^    mv -- "\${REPO_ROOT:?}\/.cargo\/config.toml".*$/    rm -f "${REPO_ROOT:?}\/.cargo\/config.toml"/'
  # every classifier pattern is load-bearing: delete each line in turn
  for arr in NET_PATTERNS HTTP_PATTERNS RESOLVER_PATTERNS ENV_PATTERNS BUILD_PATTERNS; do
    lines=$(awk -v a="$arr" '$0 == a"=(" {on=1; next} on && $0 == ")" {on=0} on {print NR}' "$SELF")
    for ln in $lines; do
      pat=$(sed -n "${ln}p" "$SELF" | sed 's/^ *//')
      mut "P-$arr:$ln $pat" $P "${ln}d"
    done
  done
  safe_rm_rf "$d" "$d" || bad=$((bad + 1))
  if [ "$bad" -eq 0 ]; then echo "mutants: all $n killed"; return 0; fi
  echo "mutants: $bad problem(s) of $n applied" >&2
  return 1
}

# ---------------------------------------------------------------- entry

case "${1:-}" in
  --self-test) self_test; exit $? ;;
  --case-table) case_table; exit $? ;;
  --mutants) mutants; exit $? ;;
  -h|--help|"") awk 'NR>1 && !/^#/ {exit} NR>1 {sub(/^# ?/, ""); print}' "$SELF"; [ -n "${1:-}" ] && exit 0 || exit 2 ;;
esac
rc=0
smoke_main "$@" || rc=$?
exit "$rc"
