#!/usr/bin/env bash
# check_release_timer_guard.sh - the case table for R10 (#4670): no tool installs on a release host
# while a release is open.
#
# During the 0.70.1 release a user timer on the train host installed a new llama.cpp build while the
# release dogfood was running there. scripts/release/release_timer_guard.sh is the pre-tag check that
# refuses that state. This table drives the real guard over planted host answers:
#
#   release open + tool-installing timer enabled   -> fail
#   release open + the same timer disabled          -> pass
#   no release-open marker                          -> pass
#   + every other refusal the guard makes, each with its own row, and the real probe run against a
#     stub systemctl.
#
#   bash scripts/check_release_timer_guard.sh             # the table
#   bash scripts/check_release_timer_guard.sh --mutants   # delete each `# R-*` refusal in a copy;
#                                                         # each must turn the table red
#
# No host, no ssh, no systemd: the probe is a stub, or the real probe with a stub systemctl.

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
GUARD="${TIMER_GUARD_SUBJECT:-$HERE/release/release_timer_guard.sh}"

if [ "${1:-}" = "--mutants" ]; then
  src="$HERE/release/release_timer_guard.sh"
  mapfile -t markers < <(grep -o '# R-[A-Z]*$' "$src" | sed 's/^# //')
  [ "${#markers[@]}" -gt 0 ] || { echo "check_release_timer_guard: no R-* markers in $src (vacuous)"; exit 1; }
  # A red table kills every mutant for free: the unmutated table must be green first.
  bash "$0" >/dev/null 2>&1 || { echo "check_release_timer_guard: the unmutated table is red - mutants not_measured"; exit 1; }
  work="$(mktemp -d)"
  cp "$HERE/release/tool-install-timers.txt" "$work/"
  killed=0
  for m in "${markers[@]}"; do
    grep -v "# $m\$" "$src" >"$work/guard.sh"
    if cmp -s "$src" "$work/guard.sh"; then
      echo "  SURVIVED $m (the mutation changed nothing)"
      continue
    fi
    if TIMER_GUARD_SUBJECT="$work/guard.sh" bash "$0" >/dev/null 2>&1; then
      echo "  SURVIVED $m (the table stayed green without it)"
    else
      killed=$((killed+1)); echo "  killed   $m"
    fi
  done
  rm -rf "${work:?}"
  echo "check_release_timer_guard: mutants $killed/${#markers[@]} killed"
  [ "$killed" -eq "${#markers[@]}" ]
  exit $?
fi

[ -f "$GUARD" ] || { echo "check_release_timer_guard: missing $GUARD"; exit 1; }

T="$(mktemp -d)"
trap 'rm -rf "${T:?}"' EXIT

fails=0
ok()   { printf '  ok    %s\n' "$1"; }
bad()  { fails=$((fails+1)); printf '  FAIL  %s\n     expected: %s\n     actual:   %s\n' "$1" "$2" "$3"; }
want() { if [ "$2" = "$3" ]; then ok "$1"; else bad "$1" "$2" "$3"; fi; }
# want_out <name> <substring>: the verdict names its own cause, not a neighbour's.
want_out() { case "$OUT" in *"$2"*) ok "$1";; *) bad "$1" "output containing '$2'" "$OUT";; esac; }

# The list the table judges against, independent of the committed one.
printf '# planted\ncrux-latest.timer   # installs tools\nfleet-bins.timer\n' >"$T/list.txt"
export TIMER_GUARD_LIST="$T/list.txt"

# The stub probe: prints $T/ans/<host>.out and exits with $T/ans/<host>.rc (default 0).
mkdir -p "$T/ans"
cat >"$T/probe" <<'EOF'
#!/usr/bin/env bash
d="$(dirname "$0")/ans"; h=$2
cat "$d/$h.out" 2>/dev/null
exit "$(cat "$d/$h.rc" 2>/dev/null || echo 0)"
EOF
chmod +x "$T/probe"
answer() { printf '%b' "$2" >"$T/ans/$1.out"; rm -f "$T/ans/$1.rc"; }  # answer <host> <text>
answer_rc() { echo "$2" >"$T/ans/$1.rc"; }

guard() { OUT="$(TIMER_GUARD_PROBE="$T/probe" bash "$GUARD" "$@" 2>&1)"; RC=$?; }

OPEN='MARKER present 0.70.2\n'
SHUT='MARKER absent\n'
ON='TIMER crux-latest.timer enabled active\n'
OFF='TIMER crux-latest.timer disabled inactive\n'
BINS_OFF='TIMER fleet-bins.timer disabled inactive\n'

echo "check_release_timer_guard: R10 pre-tag check, no tool installs while a release is open"

# -- the three signed rows ----------------------------------------------------
answer h1 "${OPEN}${ON}${BINS_OFF}END\n"
guard --ssh h1
want "S1 release open + tool timer enabled -> fail, exit 1" 1 "$RC"
want_out "S1 names the armed unit" "crux-latest.timer(enabled/active)"

answer h1 "${OPEN}${OFF}${BINS_OFF}END\n"
guard --ssh h1
want "S2 release open + the same timer disabled -> pass, exit 0" 0 "$RC"
want_out "S2 says every timer is disarmed" "all 2 tool-installing timer(s) disarmed"

answer h1 "${SHUT}${ON}${BINS_OFF}END\n"
guard --ssh h1
want "S3 no release-open marker -> pass, exit 0" 0 "$RC"
want_out "S3 still lists the armed timer" "armed: crux-latest.timer(enabled/active)"

# -- the rest of the refusals -------------------------------------------------
answer h1 "${OPEN}TIMER crux-latest.timer disabled active\n${BINS_OFF}END\n"
guard --ssh h1
want "R1 disabled but running now, release open -> fail" 1 "$RC"

answer h1 "${OPEN}TIMER crux-latest.timer enabled-runtime inactive\n${BINS_OFF}END\n"
guard --ssh h1
want "R2 enabled-runtime, release open -> fail" 1 "$RC"

answer h1 "${OPEN}TIMER crux-latest.timer not-found unknown\n${BINS_OFF}END\n"
guard --ssh h1
want "R3 timer not installed on the host, release open -> pass" 0 "$RC"

answer h1 "ssh: connect to host h1 port 22: No route to host\n"; answer_rc h1 255
guard --ssh h1
want "R4 host unreachable -> could not judge, exit 2" 2 "$RC"
want_out "R4 names the probe exit" "the probe exited 255"

answer h1 "${ON}${BINS_OFF}END\n"
guard --ssh h1
want "R5 no MARKER line -> could not judge, exit 2" 2 "$RC"
want_out "R5 names the missing marker line" "no MARKER line"

answer h1 "${OPEN}NOBUS systemctl --user cannot reach the user manager\n"
guard --ssh h1
want "R6 user manager unreachable -> could not judge, exit 2" 2 "$RC"
want_out "R6 names the user manager" "cannot reach the user manager"

answer h1 "${OPEN}${OFF}"
guard --ssh h1
want "R7 probe cut off before END -> could not judge, exit 2" 2 "$RC"
want_out "R7 names the partial list" "stopped before END"

answer h1 "${OPEN}${OFF}END\n"
guard --ssh h1
want "R8 one of two timers unanswered -> could not judge, exit 2" 2 "$RC"
want_out "R8 names the count" "asked about 2 timer(s), the probe answered 1"

guard
want "R9 no host named -> caller error, exit 3" 3 "$RC"

printf '# only comments\n\n' >"$T/empty.txt"
answer h1 "${OPEN}END\n"
OUT="$(TIMER_GUARD_PROBE="$T/probe" TIMER_GUARD_LIST="$T/empty.txt" bash "$GUARD" --ssh h1 2>&1)"; RC=$?
want "R10 empty timer list -> caller error, exit 3" 3 "$RC"

printf 'crux-latest.timer; touch pwned\n' >"$T/evil.txt"
OUT="$(TIMER_GUARD_PROBE="$T/probe" TIMER_GUARD_LIST="$T/evil.txt" bash "$GUARD" --ssh h1 2>&1)"; RC=$?
want "R11 a list entry that is not a unit name -> caller error, exit 3" 3 "$RC"

# -- several hosts: the worst verdict wins, armed over could-not-judge --------
answer h1 "${OPEN}${OFF}${BINS_OFF}END\n"
answer h2 "${OPEN}${ON}${BINS_OFF}END\n"
answer h3 "x\n"; answer_rc h3 255
guard --ssh h1 --ssh h2 --ssh h3
want "M1 one host armed, one unreachable -> fail, exit 1" 1 "$RC"
guard --ssh h1 --ssh h3
want "M2 one host clean, one unreachable -> could not judge, exit 2" 2 "$RC"
guard --ssh h1 --local h1
want "M3 every host clean -> pass, exit 0" 0 "$RC"
want_out "M3 prints the PASS line" "R10 TIMERS PASS hosts=2 timers=2"

# -- the real probe, against a stub systemctl --------------------------------
mkdir -p "$T/bin" "$T/state/apr"
cat >"$T/bin/systemctl" <<'EOF'
#!/usr/bin/env bash
# stub: --user show-environment ok; is-enabled/is-active read STUB_<state>_<unit-sans-dots>
[ "$1" = --user ] || exit 1
case "$2" in
  show-environment) [ -z "${STUB_NOBUS:-}" ] ;;
  is-enabled) v="STUB_EN_${3//[.-]/_}"; [ -n "${!v:-}" ] || { echo "Failed to get unit file state" >&2; exit 1; }; echo "${!v}" ;;
  is-active)  v="STUB_AC_${3//[.-]/_}"; echo "${!v:-inactive}"; [ "${!v:-inactive}" = active ] ;;
  *) exit 1 ;;
esac
EOF
chmod +x "$T/bin/systemctl"
real() { OUT="$(env -u TIMER_GUARD_PROBE PATH="$T/bin:$PATH" XDG_STATE_HOME="$T/state" "$@" bash "$GUARD" --local h0 2>&1)"; RC=$?; }

echo 0.70.2 >"$T/state/apr/release-open"
real STUB_EN_crux_latest_timer=enabled STUB_AC_crux_latest_timer=active
want "P1 real probe: marker + enabled timer -> fail, exit 1" 1 "$RC"
want_out "P1 reads the marker's version" "release open (present 0.70.2)"
real STUB_EN_crux_latest_timer=disabled
want "P2 real probe: marker + disabled timer -> pass, exit 0" 0 "$RC"
rm -f "${T:?}/state/apr/release-open"
real STUB_EN_crux_latest_timer=enabled STUB_AC_crux_latest_timer=active
want "P3 real probe: no marker -> pass, exit 0" 0 "$RC"
echo 0.70.2 >"$T/state/apr/release-open"
real STUB_NOBUS=1
want "P4 real probe: no user manager -> could not judge, exit 2" 2 "$RC"

if [ "$fails" -eq 0 ]; then
  echo "check_release_timer_guard: OK - release open + armed=fail, disabled=pass, no marker=pass, unjudged=stop"
  exit 0
fi
echo "check_release_timer_guard: $fails assertion(s) FAILED"
exit 1
