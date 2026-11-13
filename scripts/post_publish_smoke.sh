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
#       [--published-at <RFC3339>] [--container-cmd docker|podman] [--dry-run]
#   scripts/post_publish_smoke.sh --self-test      # cargo-free, no container, no network
#   scripts/post_publish_smoke.sh --mutants        # plants mutants; each must turn --self-test red
#
# Smoke (inside a fresh official rust container: no mounts, no env, no
# credentials, empty cargo home):
#   1. cargo install aprender --version <v> --locked
#   2. $CARGO_HOME/bin/apr --version  must contain <v>
#   3. $CARGO_HOME/bin/apr explain E001  must exit 0 and print "Invalid Magic Bytes".
#      WHY THIS VERB: `explain` is requires_model:false, side_effects:[] in
#      contracts/apr-cli-commands-v1.yaml, so it needs no model and no network,
#      yet `explain E001` executes real code (clap dispatch + the error-code
#      table in crates/apr-cli/src/commands/explain.rs). `--help` of a verb only
#      proves clap parsed, not that the shipped code runs.
#
# Exit codes: 0 smoke passed; 1 smoke failed (yank done, or refused/dry-run);
# 2 usage; 3 not_measured (no container runtime) — never a pass.
#
# The ONLY yank in this file is do_yank(), called from one place, with the
# literal crate `aprender` and the version that passed the guards. The cargo
# credential is used only through cargo: nothing here reads, prints or logs it.

set -euo pipefail

CRATE="aprender"
MAX_AGE_SECS=86400
SMOKE_IMAGE="${SMOKE_IMAGE:-docker.io/library/rust:1}"
SCRATCH_ROOT="${TMPDIR:-/tmp}"
SELF="${BASH_SOURCE[0]}"

# ---------------------------------------------------------------- seams
# Everything with a side effect or a clock is a function so --self-test can
# replace it with a stub.

now_epoch() { date -u +%s; }

# prints epoch seconds of the publish time of $CRATE@$1, or nothing
published_epoch() {
  local v=$1 ts
  if [ -n "${PUBLISHED_AT:-}" ]; then
    date -u -d "$PUBLISHED_AT" +%s 2>/dev/null || true
    return 0
  fi
  ts=$(curl -fsS -H "User-Agent: aprender-post-publish-smoke (release automation)" \
        "https://crates.io/api/v1/crates/$CRATE/$v" 2>/dev/null \
        | grep -o '"created_at":"[^"]*"' | head -1 | cut -d'"' -f4) || true
  [ -n "$ts" ] && date -u -d "$ts" +%s 2>/dev/null || true
}

detect_container() {
  if [ -n "${CONTAINER_CMD:-}" ]; then
    command -v "$CONTAINER_CMD" 2>/dev/null || true
    return 0
  fi
  command -v docker 2>/dev/null || command -v podman 2>/dev/null || true
}

# container_run <runtime> <version> <logfile> -> rc of the smoke.
# The log carries `SMOKE_FAIL step=<name> rc=<n>` on failure.
container_run() {
  local rt=$1 v=$2 log=$3 rc
  local inner
  inner='set +e
CARGO_HOME=$(mktemp -d); export CARGO_HOME
cargo install '"$CRATE"' --version "$SMOKE_V" --locked
rc=$?
if [ "$rc" -ne 0 ]; then echo "SMOKE_FAIL step=cargo-install rc=$rc"; exit "$rc"; fi
out=$("$CARGO_HOME/bin/apr" --version 2>&1)
rc=$?
echo "apr --version: $out"
if [ "$rc" -ne 0 ]; then echo "SMOKE_FAIL step=apr-version rc=$rc"; exit "$rc"; fi
case "$out" in *"$SMOKE_V"*) : ;; *) echo "SMOKE_FAIL step=apr-version-mismatch rc=1"; exit 1 ;; esac
out=$("$CARGO_HOME/bin/apr" explain E001 2>&1)
rc=$?
echo "apr explain E001: $out"
if [ "$rc" -ne 0 ]; then echo "SMOKE_FAIL step=verb-smoke rc=$rc"; exit "$rc"; fi
case "$out" in *"Invalid Magic Bytes"*) : ;; *) echo "SMOKE_FAIL step=verb-smoke-output rc=1"; exit 1 ;; esac
echo "SMOKE_PASS"
exit 0'
  "$rt" run --rm -e SMOKE_V="$v" "$SMOKE_IMAGE" bash -c "$inner" >"$log" 2>&1
  rc=$?
  return "$rc"
}

# THE ONLY YANK. Literal crate, version already guarded by the caller.
do_yank() {
  local v=$1
  cargo yank --version "$v" "$CRATE"
}

andon_line() { # <version> <step> <rc>
  printf '%s | aprender-release | ANDON yank %s@%s: %s rc=%s; five whys owed\n' \
    "$(date -u +%H:%MZ)" "$CRATE" "$1" "$2" "$3"
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
  if [ "$age" -lt 0 ]; then echo "publish time is in the future"; return 1; fi
  if [ "$age" -lt "$MAX_AGE_SECS" ]; then return 0; fi
  echo "published ${age}s ago, not within 24 h"
  return 1
}

smoke_main() {
  local version="" train_v="" dry_run=0 rt log smoke_rc step why
  PUBLISHED_AT=""; CONTAINER_CMD=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --version) version="${2:-}"; shift 2 ;;
      --train-published) train_v="${2:-}"; shift 2 ;;
      --published-at) PUBLISHED_AT="${2:-}"; shift 2 ;;
      --container-cmd) CONTAINER_CMD="${2:-}"; shift 2 ;;
      --dry-run) dry_run=1; shift ;;
      *) echo "usage: unknown argument '$1'" >&2; return 2 ;;
    esac
  done
  if ! [[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([-+][0-9A-Za-z.+-]+)?$ ]]; then
    echo "usage: --version <semver> is required (got '$version')" >&2; return 2
  fi
  rt=$(detect_container)
  if [ -z "$rt" ]; then
    echo "not_measured: no container runtime (docker|podman) found; the smoke did not run" >&2
    return 3
  fi
  mkdir -p "$SCRATCH_ROOT"
  log=$(mktemp "$SCRATCH_ROOT/post-publish-smoke.XXXXXX")
  echo "post-publish smoke: $CRATE@$version in a clean container ($rt, $SMOKE_IMAGE)"
  smoke_rc=0
  container_run "$rt" "$version" "$log" || smoke_rc=$?
  if [ "$smoke_rc" -eq 0 ]; then
    echo "post-publish smoke: PASS ($CRATE@$version)"
    rm -f "${log:?}"
    return 0
  fi
  step=$(grep -o 'SMOKE_FAIL step=[^ ]*' "$log" | tail -1 | sed 's/SMOKE_FAIL step=//') || true
  [ -n "$step" ] || step="container"
  echo "post-publish smoke: FAIL step=$step rc=$smoke_rc (log: $log)" >&2
  if why=$(yank_allowed "$version" "$train_v"); then
    if [ "$dry_run" = 1 ]; then
      echo "WOULD_YANK $CRATE@$version"
      echo "WOULD_ANDON $(andon_line "$version" "$step" "$smoke_rc")"
      return 1
    fi
    if do_yank "$version"; then
      write_andon "$(andon_line "$version" "$step" "$smoke_rc")"
      echo "post-publish smoke: yanked $CRATE@$version (undo: cargo yank --undo --version $version $CRATE)" >&2
    else
      write_andon "$(date -u +%H:%MZ) | aprender-release | ANDON yank-FAILED $CRATE@$version: $step rc=$smoke_rc; broken release is still live, five whys owed"
      echo "post-publish smoke: cargo yank FAILED for $CRATE@$version" >&2
    fi
    return 1
  fi
  echo "smoke failed; yank not permitted by rule ($why)" >&2
  return 1
}

# ---------------------------------------------------------------- self-test

# one case: name, expected rc, expected yank calls ("" none, else the version),
# expected stdout/andon substring ("" none), then a setup function body.
ST_FAIL=0
st_case() { # name want_rc want_yanks want_andon want_out setup_fn args...
  local name=$1 want_rc=$2 want_yanks=$3 want_andon=$4 want_out=$5 setup=$6; shift 6
  local d rc out yanks andon
  d=$(mktemp -d "$SCRATCH_ROOT/pps-selftest.XXXXXX")
  : >"$d/yanks"; : >"$d/andon"
  out=$(
    set +e
    ANDON_INBOX="$d/andon"
    do_yank() { echo "$1" >>"$d/yanks"; return 0; }
    now_epoch() { echo 1000000; }
    detect_container() { echo /stub/docker; }
    container_run() { return 0; }
    "$setup"
    smoke_main "$@" 2>&1
    echo "RC=$?"
  )
  rc=${out##*RC=}
  yanks=$(tr '\n' ' ' <"$d/yanks" | sed 's/ $//')
  andon=$(cat "$d/andon")
  local ok=1
  [ "$rc" = "$want_rc" ] || ok=0
  [ "$yanks" = "$want_yanks" ] || ok=0
  if [ -n "$want_andon" ]; then case "$andon" in *"$want_andon"*) : ;; *) ok=0 ;; esac
  else [ -z "$andon" ] || ok=0; fi
  if [ -n "$want_out" ]; then case "$out" in *"$want_out"*) : ;; *) ok=0 ;; esac; fi
  if [ "$ok" = 1 ]; then
    printf 'PASS  %-34s rc=%s yanks=[%s]\n' "$name" "$rc" "$yanks"
  else
    printf 'FAIL  %-34s rc=%s (want %s) yanks=[%s] (want [%s]) andon=[%s]\n' \
      "$name" "$rc" "$want_rc" "$yanks" "$want_yanks" "$andon"
    printf '%s\n' "$out" | sed 's/^/      | /'
    ST_FAIL=$((ST_FAIL + 1))
  fi
  rm -rf "${d:?}"
}


self_test() {
  [ -n "${BASH_VERSION:-}" ] || { echo "needs bash" >&2; return 2; }

  # 1000000 is the stubbed clock; publish times are RFC3339 only for the real
  # path, so the stubs below return epochs directly.
  st_fresh()  { published_epoch() { echo $((1000000 - 3600)); }; container_run() { echo "SMOKE_FAIL step=verb-smoke rc=7" >"$3"; return 7; }; }
  st_stale()  { published_epoch() { echo $((1000000 - 90000)); }; container_run() { echo "SMOKE_FAIL step=verb-smoke rc=7" >"$3"; return 7; }; }
  st_edge()   { published_epoch() { echo $((1000000 - 86400)); }; container_run() { echo "SMOKE_FAIL step=cargo-install rc=101" >"$3"; return 101; }; }
  st_nopub()  { published_epoch() { echo ""; }; container_run() { echo "SMOKE_FAIL step=cargo-install rc=101" >"$3"; return 101; }; }
  st_silent() { published_epoch() { echo $((1000000 - 3600)); }; container_run() { return 9; }; }
  st_ok()     { published_epoch() { echo $((1000000 - 3600)); }; container_run() { echo SMOKE_PASS >"$3"; return 0; }; }
  st_none()   { detect_container() { echo ""; }; published_epoch() { echo $((1000000 - 3600)); }; }

  st_case "smoke pass"                  0 ""      ""            "PASS"        st_ok      --version 0.75.0 --train-published 0.75.0
  st_case "fail fresh + train version"  1 "0.75.0" "ANDON yank aprender@0.75.0: verb-smoke rc=7; five whys owed" "" st_fresh --version 0.75.0 --train-published 0.75.0
  st_case "fail at 25 h"                1 ""      ""            "not permitted by rule" st_stale --version 0.75.0 --train-published 0.75.0
  st_case "fail at exactly 24 h"        1 ""      ""            "not permitted by rule" st_edge  --version 0.75.0 --train-published 0.75.0
  st_case "version mismatch"            1 ""      ""            "not the version this release run published" st_fresh --version 0.75.0 --train-published 0.74.9
  st_case "no --train-published"        1 ""      ""            "no --train-published" st_fresh --version 0.75.0
  st_case "publish time unknown"        1 ""      ""            "unknown"     st_nopub   --version 0.75.0 --train-published 0.75.0
  st_case "no container runtime"        3 ""      ""            "not_measured" st_none   --version 0.75.0 --train-published 0.75.0
  st_case "dry-run fail prints WOULD_YANK" 1 ""   ""            "WOULD_YANK aprender@0.75.0" st_fresh --version 0.75.0 --train-published 0.75.0 --dry-run
  st_case "swallowed-rc guard (rc 9, no marker)" 1 "0.75.0" "container rc=9" "" st_silent --version 0.75.0 --train-published 0.75.0
  st_case "bad version arg"             2 ""      ""            "usage"       st_ok      --version "0.75.0; rm -rf /" --train-published 0.75.0

  # structural: exactly one cargo yank in this file, and the cascade calls us.
  local root y
  root=$(cd "$(dirname "$SELF")/.." && pwd)
  y=$(grep -c '^  cargo yank --version "\$v" "\$CRATE"$' "$SELF") || y=0
  if [ "$y" = 1 ]; then echo "PASS  exactly one yank command in do_yank"
  else echo "FAIL  yank command count=$y"; ST_FAIL=$((ST_FAIL + 1)); fi
  if grep -q 'post_publish_smoke.sh' "$root/scripts/cascade-publish.sh" 2>/dev/null \
     && grep -q -- '--train-published' "$root/scripts/cascade-publish.sh"; then
    echo "PASS  cascade-publish.sh calls post_publish_smoke.sh"
  else echo "FAIL  cascade-publish.sh does not call post_publish_smoke.sh"; ST_FAIL=$((ST_FAIL + 1)); fi
  if [ "$ST_FAIL" -eq 0 ]; then echo "self-test: ALL PASS"; return 0; fi
  echo "self-test: $ST_FAIL FAILED" >&2
  return 1
}

# mutants: each edited copy must turn --self-test red.
mutants() {
  local d n=0 bad=0 rc
  d=$(mktemp -d "$SCRATCH_ROOT/pps-mutants.XXXXXX")
  mut() { # name sed-expression
    local name=$1 expr=$2
    mkdir -p "$d/scripts"
    cp "$SELF" "$d/scripts/post_publish_smoke.sh"
    cp "$(dirname "$SELF")/cascade-publish.sh" "$d/scripts/cascade-publish.sh"
    sed -i "$expr" "$d/scripts/post_publish_smoke.sh"
    if cmp -s "$SELF" "$d/scripts/post_publish_smoke.sh"; then
      echo "MUTANT $name: NOT APPLIED (pattern drifted)"; bad=$((bad + 1)); return
    fi
    n=$((n + 1))
    rc=0
    bash "$d/scripts/post_publish_smoke.sh" --self-test >/dev/null 2>&1 || rc=$?
    if [ "$rc" -ne 0 ]; then echo "MUTANT $name: killed (self-test rc=$rc)"
    else echo "MUTANT $name: SURVIVED"; bad=$((bad + 1)); fi
  }
  mut "flip-24h-comparison"   's/if \[ "\$age" -lt "\$MAX_AGE_SECS" \]/if [ "$age" -gt "$MAX_AGE_SECS" ]/'
  mut "drop-version-equality" 's/if \[ "\$train_v" != "\$v" \]/if false/'
  mut "swallow-smoke-rc"      's/container_run "\$rt" "\$version" "\$log" || smoke_rc=\$?/container_run "$rt" "$version" "$log" || smoke_rc=0/'
  mut "yank-in-dry-run"       's/if \[ "\$dry_run" = 1 \]; then/if false; then/'
  mut "no-runtime-is-pass"    's/    return 3$/    return 0/'
  mut "skip-train-check"      's/if \[ -z "\$train_v" \]; then echo "no --train-published version given"; return 1; fi//'
  rm -rf "${d:?}"
  if [ "$bad" -eq 0 ]; then echo "mutants: all $n killed"; return 0; fi
  echo "mutants: $bad problem(s)" >&2
  return 1
}

# ---------------------------------------------------------------- entry

case "${1:-}" in
  --self-test) self_test; exit $? ;;
  --mutants) mutants; exit $? ;;
  -h|--help|"") sed -n '2,32p' "$SELF"; [ -n "${1:-}" ] && exit 0 || exit 2 ;;
esac
rc=0
smoke_main "$@" || rc=$?
exit "$rc"
