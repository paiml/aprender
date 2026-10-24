#!/usr/bin/env bash
# ci_run_ctl.sh cancel|rerun RUN_ID...   |   ci_run_ctl.sh --self-test   (PMAT-5851, #5851)
#
# Cancel or re-run workflow runs and PROVE it: each id is acted on, then the run's final
# state is read back, and any id whose state is not the wanted one makes the exit non-zero.
#   cancel   wanted: status completed with conclusion cancelled (or already completed)
#   rerun    wanted: the run is no longer completed (queued / in_progress / waiting / requested)
# Exit: 0 every id verified; 1 at least one miss (named); 2 usage (nothing was acted on).
#
# Why: at 11:29Z a zsh variable holding seven run ids went to `gh run cancel $ids` as ONE word
# (zsh does not word-split), gh refused it, the loop printed nothing, and 0 of 7 were
# cancelled with no error anyone read. So an id must be ONE argument that is only digits:
# "123 456" or a newline-joined list is refused by name before any call is made.
#
# Reads are spaced (CI_RUN_CTL_WAIT seconds, default 10, at most 2 reads per id): GH-1.
# Tests swap `gh` through GH_BIN and the repo through CI_RUN_CTL_REPO. No python.
set -u

GH_BIN=${GH_BIN:-gh}
REPO=${CI_RUN_CTL_REPO:-paiml/aprender}
WAIT=${CI_RUN_CTL_WAIT:-10}

usage() { printf 'usage: %s cancel|rerun RUN_ID [RUN_ID...] | --self-test\n' "$0" >&2; exit 2; }

# state ID -> "status/conclusion" on stdout, empty when the read failed.
state() { "$GH_BIN" run view "$1" -R "$REPO" --json status,conclusion -q '.status + "/" + .conclusion' 2> /dev/null; }

wanted() { # MODE STATE
  case "$1:$2" in
    cancel:completed/*) return 0 ;;
    rerun:completed/*) return 1 ;;
    rerun:queued/* | rerun:in_progress/* | rerun:waiting/* | rerun:requested/* | rerun:pending/*) return 0 ;;
    *) return 1 ;;
  esac
}

run_ctl() {
  local mode=${1:-}; shift || true
  case "$mode" in cancel | rerun) ;; *) usage ;; esac
  [ $# -ge 1 ] || { printf 'ci_run_ctl: no run id given\n' >&2; exit 2; }
  local id
  for id in "$@"; do
    case "$id" in
      '' | *[!0-9]*) printf 'ci_run_ctl: refused %q: a run id is ONE argument of digits (an unsplit id list?); nothing was acted on\n' "$id" >&2; exit 2 ;;
    esac
  done
  local miss=0 ok=0 st tries
  for id in "$@"; do
    case "$mode" in
      cancel) "$GH_BIN" run cancel "$id" -R "$REPO" > /dev/null 2>&1 ;;
      rerun) "$GH_BIN" run rerun "$id" -R "$REPO" > /dev/null 2>&1 ;;
    esac
    st=""
    for tries in 1 2; do
      sleep "$WAIT"
      st=$(state "$id")
      wanted "$mode" "$st" && break
    done
    if wanted "$mode" "$st"; then
      ok=$((ok + 1)); printf 'ok    %s %s -> %s\n' "$mode" "$id" "$st"
    else
      miss=$((miss + 1)); printf 'MISS  %s %s -> %s\n' "$mode" "$id" "${st:-unreadable}"
    fi
  done
  printf 'ci_run_ctl: %s: %s/%s verified, %s missed\n' "$mode" "$ok" "$#" "$miss"
  [ "$miss" -eq 0 ]
}

self_test() {
  local fail=0 rc out T
  TMPD=$(mktemp -d) || exit 2; T=$TMPD
  trap 'rm -rf "${TMPD:?}"' EXIT
  # A fake gh over a state dir: <id>.state holds "status/conclusion"; <id>.stuck makes
  # cancel/rerun a silent no-op; every call is logged to calls.log.
  cat > "$T/gh" << 'EOF'
#!/usr/bin/env bash
d=$FAKE_GH_DIR; echo "$*" >> "$d/calls.log"
[ "$1" = run ] || exit 1
act=$2; id=$3
case "$act" in
  cancel) [ -e "$d/$id.stuck" ] || echo "completed/cancelled" > "$d/$id.state" ;;
  rerun) [ -e "$d/$id.stuck" ] || echo "queued/" > "$d/$id.state" ;;
  view) [ -f "$d/$id.state" ] && cat "$d/$id.state" ;;
esac
exit 0
EOF
  chmod +x "$T/gh"
  export FAKE_GH_DIR="$T/d" GH_BIN="$T/gh" CI_RUN_CTL_WAIT=0
  reset() { rm -rf "${T:?}/d"; mkdir -p "$T/d"; for i in "$@"; do echo "in_progress/" > "$T/d/$i.state"; done; }
  check() { # LABEL WANT_RC GOT_RC
    if [ "$2" = "$3" ]; then printf 'ok    %s (rc=%s)\n' "$1" "$3"; else printf 'FAIL  %s: want rc %s, got %s\n' "$1" "$2" "$3"; fail=$((fail + 1)); fi
  }
  me=$(readlink -f "$0")

  reset 11 12 13; out=$(bash "$me" cancel 11 12 13 2>&1); rc=$?
  check "C1 three ids cancelled and read back" 0 $rc
  reset 11 12 13; out=$(bash "$me" cancel "11 12 13" 2>&1); rc=$?
  check "C2 FALSIFIER: an id list passed unsplit is refused, rc 2" 2 $rc
  [ ! -s "$T/d/calls.log" ] && printf 'ok    C2b the unsplit list made no gh call at all\n' || { printf 'FAIL  C2b gh was called on an unsplit list\n'; fail=$((fail + 1)); }
  reset 11 12 13; out=$(bash "$me" cancel "$(printf '11\n12\n13')" 2>&1); rc=$?
  check "C3 a newline-joined list is refused, rc 2" 2 $rc
  reset 11 12 13; touch "$T/d/12.stuck"; out=$(bash "$me" cancel 11 12 13 2>&1); rc=$?
  check "C4 one id that stays in_progress is a MISS, rc 1" 1 $rc
  grep -q '^MISS  cancel 12 ' <<< "$out" && printf 'ok    C4b the miss names run 12\n' || { printf 'FAIL  C4b run 12 not named\n'; fail=$((fail + 1)); }
  reset 11; rm -f "$T/d/11.state"; touch "$T/d/11.stuck"; out=$(bash "$me" cancel 11 2>&1); rc=$?
  check "C5 an unreadable final state is a MISS, rc 1" 1 $rc
  reset 21 22; for i in 21 22; do echo "completed/failure" > "$T/d/$i.state"; done; out=$(bash "$me" rerun 21 22 2>&1); rc=$?
  check "C6 rerun of two finished runs verifies they restarted" 0 $rc
  reset 21 22; for i in 21 22; do echo "completed/failure" > "$T/d/$i.state"; done; touch "$T/d/22.stuck"; out=$(bash "$me" rerun 21 22 2>&1); rc=$?
  check "C7 a rerun that never restarted is a MISS, rc 1" 1 $rc
  reset; out=$(bash "$me" cancel 2>&1); rc=$?
  check "C8 no id is usage, rc 2" 2 $rc
  reset; out=$(bash "$me" cancel abc 2>&1); rc=$?
  check "C9 a non-numeric id is usage, rc 2" 2 $rc
  reset; out=$(bash "$me" frob 1 2>&1); rc=$?
  check "C10 an unknown verb is usage, rc 2" 2 $rc

  # MUTANT: with the digits check removed, the C2 input must reach gh (so C2 would go red).
  sed '/\*\[!0-9\]\*) printf/d' "$me" > "$T/mut.sh"
  reset 11 12 13; out=$(bash "$T/mut.sh" cancel "11 12 13" 2>&1); rc=$?
  if [ -s "$T/d/calls.log" ] || [ "$rc" != 2 ]; then printf 'ok    M1 MUTANT without the digits check acts on the unsplit list (C2 goes red)\n'; else printf 'FAIL  M1 mutant did not change behaviour\n'; fail=$((fail + 1)); fi
  # MUTANT: with the read-back verdict forced true, the stuck run must pass (C4 would go red).
  sed 's/^  \[ "\$miss" -eq 0 \]$/  true/' "$me" > "$T/mut2.sh"
  reset 11 12; touch "$T/d/12.stuck"; out=$(bash "$T/mut2.sh" cancel 11 12 2>&1); rc=$?
  [ "$rc" = 0 ] && printf 'ok    M2 MUTANT ignoring the read-back exits 0 on a miss (C4 goes red)\n' || { printf 'FAIL  M2 mutant still exited %s\n' "$rc"; fail=$((fail + 1)); }

  printf 'ci_run_ctl self-test: %s failed\n' "$fail"
  [ "$fail" -eq 0 ]
}

case "${1:-}" in
  --self-test) self_test ;;
  *) run_ctl "$@" ;;
esac
