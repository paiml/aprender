#!/usr/bin/env bash
# mutate_story_pmat_hunt_guard.sh - prove check_story_pmat_hunt.sh turns RED when
# pmat_hunt's not_measured / zero-row rules in scripts/lib_story_pmat.sh break.
#
# The guard is a table its own author wrote. Until each rule it states has been
# broken on purpose and watched to turn a row RED, it proves only that it agrees
# with itself. The headline mutant is #4834's own defect: restore the combined
# `PMAT_HUNT != 1 || pmat absent -> return 0` condition, and the absent-binary
# rows must go RED.
#
# Every mutant is applied to a COPY of the library in a mktemp tree, never in
# place, and judged by running the guard in that tree. The tree carries no
# qwen-story.sh, so the guard's static path row (section 8) is skipped there; it
# reads the real repo, not the library, and no mutant below touches it.
#
# THE PATCH STEP FAILS CLOSED. The old text must occur EXACTLY ONCE in the
# library and the patched copy must differ from the original. Either false makes
# the mutant INVALID - never a kill, never a survivor - and an INVALID mutant
# fails the run. Pure bash: no Python in the build (C301).
#
# M0 is the discrimination case: the UNMUTATED copy must pass the guard. Without
# it, "the guard fails unconditionally" scores a perfect kill rate.
#
# Exit 0 = baseline GREEN and every mutant KILLED.
# Exit 1 = a mutant survived, a patch did not apply, or the baseline was not GREEN.
# Exit 2 = the environment could not run the guard at all.

set -uo pipefail

cd "$(dirname "$0")/.." || exit 2
REPO="$PWD"

LIB_REL=scripts/lib_story_pmat.sh
GUARD_REL=scripts/check_story_pmat_hunt.sh

command -v jq >/dev/null 2>&1 || { printf 'jq not on PATH: UNMEASURED\n' >&2; exit 2; }
[ -f "$REPO/$LIB_REL" ] || { printf 'no library at %s\n' "$LIB_REL" >&2; exit 2; }
[ -f "$REPO/$GUARD_REL" ] || { printf 'no guard at %s\n' "$GUARD_REL" >&2; exit 2; }

T="$(mktemp -d "${TMPDIR:-/tmp}/mutate-story-pmat.XXXXXX")" || exit 2
trap 'rm -rf "${T:?}"' EXIT

killed=0
survived=0
invalid=0

# replace_once <file> <old> <new> - rc 0 applied; 3 old not present exactly
# once; 4 file unchanged. The file is read whole with its trailing newline kept.
replace_once() {
  local f="$1" old="$2" new="$3" s rest t n
  s="$(cat "$f"; printf x)"; s="${s%x}"
  rest="${s//"$old"/}"
  n=$(( (${#s} - ${#rest}) / ${#old} ))
  [ "$n" -eq 1 ] || return 3
  t="${s/"$old"/"$new"}"
  [ "$t" != "$s" ] || return 4
  printf '%s' "$t" > "$f"
}

# judge <lib-file> - prints the first FAIL row; rc 0 = the guard passed.
judge() {
  local lib="$1" d rc
  d="$(mktemp -d "$T/tree.XXXXXX")"
  mkdir -p "$d/scripts"
  cp "$REPO/$GUARD_REL" "$d/$GUARD_REL"
  cp "$lib" "$d/$LIB_REL"
  rc=0
  bash "$d/$GUARD_REL" > "$d/out" 2>&1 || rc=$?
  if [ "$rc" -eq 0 ]; then return 0; fi
  grep -m1 '  FAIL  ' "$d/out" | sed 's/^  FAIL  //' || printf 'guard rc=%s' "$rc"
  return 1
}

# mutant <id> <old> <new>
mutant() {
  local id="$1" old="$2" new="$3" m prc=0 why
  m="$T/$id.sh"
  cp "$REPO/$LIB_REL" "$m"
  replace_once "$m" "$old" "$new" || prc=$?
  if [ "$prc" -eq 0 ] && cmp -s "$REPO/$LIB_REL" "$m"; then prc=4; fi
  if [ "$prc" -ne 0 ]; then
    printf '  %-34s INVALID   patch did not apply (rc=%s) - verdict not read\n' "$id" "$prc"
    invalid=$((invalid + 1))
    return 0
  fi
  if why="$(judge "$m")"; then
    printf '  %-34s SURVIVED  <-- no guard row noticed\n' "$id"
    survived=$((survived + 1))
  else
    printf '  %-34s KILLED    by %s\n' "$id" "$why"
    killed=$((killed + 1))
  fi
}

printf 'mutate_story_pmat_hunt_guard.sh - mutation set for %s, judged by %s\n\n' "$LIB_REL" "$GUARD_REL"

# --- M0: discrimination case --------------------------------------------------
if why="$(judge "$REPO/$LIB_REL")"; then
  printf '  %-34s GREEN     as required\n' 'M0-baseline-unmutated'
else
  printf '  %-34s RED at %s - every kill below would be a kill of the harness\n' 'M0-baseline-unmutated' "$why"
  exit 1
fi

# --- #4834: the absent-binary branch -------------------------------------------
# The pre-fix shape, verbatim: the opt-out and an absent binary share return 0.
mutant restore-combined-condition \
  '[ "${PMAT_HUNT:-1}" = "1" ] || return 0' \
  'if [ "${PMAT_HUNT:-1}" != "1" ] || ! command -v "${PMAT_BIN:-pmat}" >/dev/null 2>&1; then return 0; fi'
mutant absent-returns-zero \
  "0 of \$# path(s) hunted (#4834)\"
    return 1" \
  "0 of \$# path(s) hunted (#4834)\"
    return 0"
mutant absent-drops-emit-fail \
  "    emit_fail \"pmat-hunt \$beat\" \"not_measured:" \
  "    : \"pmat-hunt \$beat\" \"not_measured:"
mutant absent-drops-not-measured-line \
  "    printf '    pmat-hunt (%s): not_measured - no pmat binary %s\\n'" \
  "    : '    pmat-hunt (%s): not_measured - no pmat binary %s\\n'"
mutant absent-check-never-fires \
  'if ! command -v "${PMAT_BIN:-pmat}" >/dev/null 2>&1; then' \
  'if false; then'
mutant absent-check-always-fires \
  'if ! command -v "${PMAT_BIN:-pmat}" >/dev/null 2>&1; then' \
  'if true; then'

# --- the opt-out ----------------------------------------------------------------
mutant opt-out-never-fires \
  '[ "${PMAT_HUNT:-1}" = "1" ] || return 0' \
  'true || return 0'
mutant opt-out-after-absent-check \
  '[ "${PMAT_HUNT:-1}" = "1" ] || return 0' \
  '{ ! command -v "${PMAT_BIN:-pmat}" >/dev/null 2>&1 || [ "${PMAT_HUNT:-1}" = "1" ]; } || return 0'

# --- the zero-row andon (#2356) --------------------------------------------------
mutant zero-rows-returns-zero \
  '    fi
    return 1
  fi
  return 0' \
  '    fi
    return 0
  fi
  return 0'
mutant zero-rows-drops-emit-fail \
  'emit_fail "pmat-hunt $beat" "manifest header printed with 0 rows over $paths existing' \
  ': "pmat-hunt $beat" "manifest header printed with 0 rows over $paths existing'

total=$((killed + survived + invalid))
printf '\nmutants: %s killed, %s survived, %s invalid (of %s)\n' "$killed" "$survived" "$invalid" "$total"
if [ "$total" -eq 0 ] || [ "$survived" -ne 0 ] || [ "$invalid" -ne 0 ]; then
  printf 'FAIL: %s is not proved to turn RED for every rule it states.\n' "$GUARD_REL"
  exit 1
fi
printf 'guard_mutation_score = 100%% (%s/%s)\n' "$killed" "$total"
printf 'PASS: baseline GREEN, every mutant KILLED, every patch observed to apply.\n'
exit 0
