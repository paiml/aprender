#!/usr/bin/env bash
# include_fmt_ratchet.sh — include!d .rs files are inside the formatting gate, as a shrink-only ratchet (#4151).
#
# WHY. `cargo fmt --all -- --check` (CI's fmt gate) never formats a file pulled in with `include!()`:
# rustfmt follows `mod` declarations, not `include!`. Measured (#4140): two 141/142-column lines in
# apr-cli's golden_output.rs, include!d from qa.rs, passed it with rc 0, twice. On main 49fe19c28,
# 1584 of the 1778 include!d .rs files are misformatted. A full reformat is ~264k lines of churn
# across 1783 files (165 do not converge). The cop deferred that to a quiet window after the 0.70
# fold and ruled this shape instead (2026-09-24):
#
#   * scripts/lib/include_fmt.py enumerates the include!d .rs files FROM SOURCE and runs the pinned
#     rustfmt on them (--edition of the owning crate).
#   * NEW DEBT IS RED, BY NAME: an include!d file that rustfmt would change and that is not in
#     scripts/include_fmt_baseline.txt fails.
#   * THE BASELINE MAY ONLY SHRINK: scripts/lib_baseline_ratchet.sh compares it against
#     merge-base(HEAD, origin/main) (set mode). An entry may leave; one may not arrive.
#   * A baseline entry that is now clean, or no longer include!d, is reported. `--update` removes
#     it (shrink only: it never adds). It is not a failure, so fixing a file never blocks a PR.
#
# Called by scripts/check_package_includes.sh (its main run and its --self-test), which ci.yml's
# guard-cargo job runs, so there is no workflow edit.
#
#   bash scripts/include_fmt_ratchet.sh              # the gate
#   bash scripts/include_fmt_ratchet.sh --update     # drop entries that are now clean (never adds)
#   bash scripts/include_fmt_ratchet.sh --self-test  # case table (a scratch repository)
#
# Exit: 0 no new debt and the baseline did not grow · 1 either one · 2 could not check (not a pass).
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${INCLUDE_FMT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
BASELINE_REL="scripts/include_fmt_baseline.txt"

entries() { # <file> -> its entries, sorted, comments and blanks dropped
  grep -vE '^[[:space:]]*(#|$)' "$1" 2>/dev/null | LC_ALL=C sort -u
}

scan() { # <root> <out-file> -> the misformatted list; rc 2 when the scan could not answer
  local out rc
  out="$(python3 "$SCRIPT_DIR/lib/include_fmt.py" "$1" 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ]; then printf '%s\n' "$out" >&2; return 2; fi
  sed -n 's/^MISFORMATTED //p' <<< "$out" | LC_ALL=C sort -u > "$2"
  SCAN_SUMMARY="$(grep '^SCANNED ' <<< "$out")"
  return 0
}

# The instrument. rustfmt's verdict IS the finding, so a baseline measured by one rustfmt and a tree
# checked by another compare two instruments. The baseline pins the exact `rustfmt --version` line
# (cwd = the repo, so rust-toolchain.toml applies). lib_baseline_ratchet.sh's generic tool_version
# check compares "<tool> <word>" and cannot hold rustfmt's "(hash date)" suffix, hence it lives here.
live_rustfmt() { (cd "$1" && rustfmt --version 2>/dev/null) | head -n 1; }
pinned_rustfmt() { sed -n 's/^#[[:space:]]*rustfmt_version=//p' "$1" | head -n 1; }

gate() { # <root> -> 0 / 1 / 2
  local root="$1" base="$1/$BASELINE_REL" t rc=0 new stale live pinned
  [ -f "$base" ] || { echo "  cannot check: $BASELINE_REL is missing (the ratchet has nothing to hold)"; return 2; }
  live="$(live_rustfmt "$root")"; pinned="$(pinned_rustfmt "$base")"
  [ -n "$live" ] || { echo "  cannot check: no rustfmt on PATH in $root"; return 2; }
  if [ "$live" != "$pinned" ]; then
    echo "  cannot check: $BASELINE_REL was measured by '${pinned:-<no rustfmt_version header>}', this runner has '$live' - two instruments, no verdict"
    return 2
  fi
  t=$(mktemp -d) || return 2
  scan "$root" "$t/cur" || { rm -rf -- "${t:?}"; echo "  cannot check: the include!d scan did not answer"; return 2; }
  entries "$base" > "$t/base"
  new="$(LC_ALL=C comm -23 "$t/cur" "$t/base")"
  stale="$(LC_ALL=C comm -13 "$t/cur" "$t/base")"
  rm -rf -- "${t:?}"
  echo "$SCAN_SUMMARY; $(grep -c . <<< "$(entries "$base")" || true) grandfathered in $BASELINE_REL"
  if [ -n "$new" ]; then
    printf 'FAIL  include!d file(s) rustfmt would change, NOT in the baseline (new debt; cargo fmt cannot see them):\n'
    sed 's/^/        /' <<< "$new"
    printf '      fix: rustfmt --edition <the crate edition> <file>   (a baselined file may stay; a new one may not)\n'
    rc=1
  fi
  if [ -n "$stale" ]; then
    printf 'REPORT %s baseline entr(ies) are now clean or no longer include!d: run --update to lock the gain in\n' \
      "$(grep -c . <<< "$stale")"
  fi
  # the baseline itself may only shrink, against a ref this branch cannot rewrite
  # shellcheck source=scripts/lib_baseline_ratchet.sh
  . "$root/scripts/lib_baseline_ratchet.sh" || { echo "  cannot check: no lib_baseline_ratchet.sh"; return 2; }
  baseline_ratchet_check "$root" "$BASELINE_REL" set || rc=1
  return "$rc"
}

if [ "${1:-}" = "--update" ]; then
  t=$(mktemp -d) || exit 2
  scan "$REPO_ROOT" "$t/cur" || { rm -rf -- "${t:?}"; exit 2; }
  base="$REPO_ROOT/$BASELINE_REL"
  if [ -f "$base" ]; then
    # shrink only: keep an entry only while it is still misformatted
    entries "$base" > "$t/base"
    LC_ALL=C comm -12 "$t/cur" "$t/base" > "$t/keep"
  else
    cp -- "$t/cur" "$t/keep"   # the introducing commit: the measured set, reviewed in that PR
  fi
  live="$(live_rustfmt "$REPO_ROOT")"
  [ -n "$live" ] || { rm -rf -- "${t:?}"; echo "  cannot update: no rustfmt"; exit 2; }
  { echo "# include_fmt_baseline.txt — include!d .rs files rustfmt would change, grandfathered (#4151)."
    echo "# SHRINK-ONLY (scripts/include_fmt_ratchet.sh, lib_baseline_ratchet.sh set mode): an entry may leave, never arrive."
    echo "# Regenerate with: bash scripts/include_fmt_ratchet.sh --update"
    echo "# tool_version=none -- the instrument is pinned EXACTLY on the next line and enforced by include_fmt_ratchet.sh;"
    echo "#   baseline_require_tool_version compares two words and cannot hold rustfmt's '(hash date)' suffix"
    echo "# rustfmt_version=$live"
    cat "$t/keep"; } > "$base"
  printf 'baseline: %s entr(ies)\n' "$(grep -c . "$t/keep" || true)"
  rm -rf -- "${t:?}"; exit 0
fi

if [ "${1:-}" = "--self-test" ]; then
  TD=$(mktemp -d) || exit 2
  case "$TD" in /tmp/?*) ;; *) echo "self-test: unexpected temp dir $TD"; exit 2 ;; esac
  cleanup() { case "${TD:-}" in /tmp/?*) [ -d "$TD" ] && rm -rf -- "$TD" ;; esac; }
  trap cleanup EXIT
  fails=0; R="$TD/repo"
  commit() { git -C "$R" add -A && git -C "$R" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -qm "$1"; }
  row() { # <name> <want-rc> <needle> [env...]
    local name="$1" want="$2" needle="$3" out rc; shift 3
    out="$(env "$@" INCLUDE_FMT_ROOT="$R" BASELINE_RATCHET_BASE_REF=fixture-main bash "$SCRIPT_DIR/include_fmt_ratchet.sh" 2>&1)"; rc=$?
    if [ "$rc" = "$want" ] && grep -qF -- "$needle" <<< "$out"; then printf 'ok    %s\n' "$name"
    else printf 'FAIL  %s: rc=%s (want %s), no "%s" in:\n%s\n' "$name" "$rc" "$want" "$needle" "$out"; fails=1; fi
  }
  # The fallback TOML reader (py3.10 runners have no tomllib, #4315): its case table, with and without
  # the real parser, and its parity with the real parser on every tracked manifest when one exists.
  for fb in 0 1; do
    if INCLUDE_FMT_FORCE_FALLBACK="$fb" python3 "$SCRIPT_DIR/lib/include_fmt.py" --self-test "$SCRIPT_DIR/.."; then
      printf 'ok    include_fmt.py reader case table (forced fallback=%s)\n' "$fb"
    else printf 'FAIL  include_fmt.py reader case table (forced fallback=%s)\n' "$fb"; fails=1; fi
  done
  mkdir -p "$R/src" "$R/scripts/lib"
  cp -- "$SCRIPT_DIR/include_fmt_ratchet.sh" "$SCRIPT_DIR/lib_baseline_ratchet.sh" "$R/scripts/"
  cp -- "$SCRIPT_DIR/lib/include_fmt.py" "$R/scripts/lib/"
  printf '[package]\nname = "fx"\nversion = "0.1.0"\nedition = "2021"\n' > "$R/Cargo.toml"
  printf 'include!("part.rs");\ninclude!("old.rs");\n' > "$R/src/lib.rs"
  printf 'pub fn part() -> u8 {\n    1\n}\n' > "$R/src/part.rs"                  # clean
  printf 'pub fn old()->u8{2}\n' > "$R/src/old.rs"                               # grandfathered debt
  fxv="$(live_rustfmt "$R")"
  printf '# fixture\n# rustfmt_version=%s\nsrc/old.rs\n' "$fxv" > "$R/$BASELINE_REL"
  git -C "$R" init -q -b fixture-main && commit fixture
  row "grandfathered debt is not new debt" 0 "SCANNED 2 include!d file(s), 1 misformatted"
  # must-RED 1: a NEW misformatted include!d file fails BY NAME (golden_output.rs's shape: over-long line)
  git -C "$R" checkout -q -b topic
  printf 'pub fn part() -> u8 { let a_very_long_binding_name_that_rustfmt_will_split_across_lines = 1; a_very_long_binding_name_that_rustfmt_will_split_across_lines }\n' > "$R/src/part.rs"
  commit newdebt
  row "a NEW misformatted include!d file is RED by name" 1 "        src/part.rs"
  git -C "$R" checkout -q fixture-main -- src/part.rs && commit revert
  # must-RED 2: the baseline GROWS (grandfathering the new debt instead of fixing it) -> RED
  printf 'pub fn part()->u8{1}\n' > "$R/src/part.rs"
  printf '# fixture\n# rustfmt_version=%s\nsrc/old.rs\nsrc/part.rs\n' "$fxv" > "$R/$BASELINE_REL"
  commit grow
  row "the baseline growing is RED (shrink-only vs the comparand)" 1 "+ src/part.rs"
  git -C "$R" checkout -q fixture-main -- src/part.rs "$BASELINE_REL" && commit revert2
  # a fixed file leaves the baseline: reported, --update removes it, the shrink is green
  printf 'pub fn old() -> u8 {\n    2\n}\n' > "$R/src/old.rs"
  commit fixed
  row "a now-clean baseline entry is reported, not failed" 0 "now clean or no longer include!d"
  INCLUDE_FMT_ROOT="$R" bash "$SCRIPT_DIR/include_fmt_ratchet.sh" --update > /dev/null 2>&1
  left="$(entries "$R/$BASELINE_REL")"
  if [ -n "$left" ]; then printf 'FAIL  --update did not drop the clean entry\n'; fails=1
  else commit shrink; row "--update drops it and the shrink is green" 0 "did not grow (1 removed)"; fi
  # another instrument: a baseline pinned to a different rustfmt gives no verdict (2), never a pass
  sed -i 's/^# rustfmt_version=.*/# rustfmt_version=rustfmt 0.0.0-other (0000000000 1970-01-01)/' "$R/$BASELINE_REL"; commit otherfmt
  row "a baseline measured by another rustfmt is 'cannot check' (2)" 2 "two instruments, no verdict"
  sed -i "s/^# rustfmt_version=.*/# rustfmt_version=$fxv/" "$R/$BASELINE_REL"; commit samefmt
  # vacuity: a tree with no include!d file cannot be clean, it is unscanned
  printf 'pub fn only() {}\n' > "$R/src/lib.rs"; commit noinclude
  row "no include!d file at all is 'cannot check' (2), never a pass" 2 "cannot check: the include!d scan did not answer"
  [ "$fails" -eq 0 ] || { echo "SELF-TEST FAILED"; exit 1; }
  echo "SELF-TEST PASSED"; exit 0
fi

echo "=== include!d .rs files are formatted: new debt RED, baseline shrink-only (include_fmt_ratchet.sh, #4151) ==="
gate "$REPO_ROOT"; rc=$?
[ "$rc" = 0 ] && echo "PASS"
[ "$rc" = 1 ] && echo "FAIL"
exit "$rc"
