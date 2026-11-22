#!/usr/bin/env bash
# nightly_preflight_scope.sh -- does this PR touch what the nightly depends on? (APR-GREEN-001 item 3)
#
# The nightly reds that reach main are runner/tool drift found at night: #7375 (train landed on the wrong
# pool), a pmat pin, a reachable registry token. The `nightly-preflight` workflow runs the nightly's PREFLIGHT
# (no train) on any PR that changes one of those inputs. This script decides "does it apply": the workflow
# itself is NOT path-filtered (a path-filtered required check reports "missing", not a pass), so scope=no
# lets the aggregator job report success without spending the intel runner.
#
# APPLICABLE when the PR diff has ANY of:
#   - a nightly/release workflow: .github/workflows/{*nightly*,release*,rc-cut,binary-release,toolchain-ceiling}.yml
#   - a changed `runs-on` line (added or removed) in ANY workflow under .github/workflows/
#   - a tool pin: rust-toolchain.toml, tools.toml, scripts/*_pin.sh, scripts/*pin*.toml, scripts/check_tool_versions.sh
#   - the nightly driver and its guards: scripts/release/**, scripts/check_nightly_train_pool.sh, scripts/nightly_*.sh,
#     scripts/ci/nightly_preflight*.sh
# Anything it cannot judge (git failure, empty path list from git) is applicable=yes: fail closed.
#
#   scope BASE HEAD            diff merge-base(BASE,HEAD)..HEAD in the current repo; prints applicable=yes|no + reason= lines
#   scope --classify PATHS [DIFF]   same decision from a path list file and an optional `git diff -U0` file
#   scope --self-test | --mutants
# Exit 0 decided (read the applicable= line), 2 environment.
set -uo pipefail
PROG=nightly_preflight_scope.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

path_reason() { # path -> reason on stdout when the path alone makes the PR applicable
  case "$1" in
    .github/workflows/*nightly*.yml | .github/workflows/release*.yml | .github/workflows/rc-cut.yml \
      | .github/workflows/binary-release.yml | .github/workflows/toolchain-ceiling.yml) echo "nightly/release workflow $1" ;;
    rust-toolchain.toml | tools.toml | scripts/check_tool_versions.sh) echo "tool pin $1" ;;
    scripts/*_pin.sh | scripts/*pin*.toml) echo "tool pin $1" ;;
    scripts/release/*) echo "nightly/release script $1" ;;
    scripts/check_nightly_train_pool.sh | scripts/nightly_*.sh | scripts/ci/nightly_preflight*.sh) echo "nightly preflight input $1" ;;
  esac
}

classify() { # classify PATHSFILE [DIFFFILE]
  local paths="$1" diff="${2:-}" n=0 hit=0 p r f=""
  while IFS= read -r p || [ -n "$p" ]; do
    [ -n "$p" ] || continue
    n=$((n + 1))
    case "$p" in '"'*) echo "reason=quoted path $p, cannot judge"; hit=1; continue ;; esac
    r="$(path_reason "$p")"
    if [ -n "$r" ]; then echo "reason=$r"; hit=1; fi
  done < "$paths"
  if [ $# -ge 2 ]; then
    if [ -z "$diff" ] || [ ! -r "$diff" ] || [ -d "$diff" ]; then
      echo "reason=unreadable diff ${diff:-<empty>}, cannot judge"; echo "applicable=yes"; return 0
    fi
    # a +/- line naming runs-on, in a hunk of a workflow file
    while IFS= read -r p || [ -n "$p" ]; do
      case "$p" in
        "+++ /dev/null") ;;
        "+++ b/.github/workflows/"* | "--- a/.github/workflows/"*) f="${p#??? ?/}" ;;
        "+++ "* | "--- "*) f="" ;;
        [+-]*)
          if [ -n "$f" ] && grep -Eq '^[+-][[:space:]]*(-[[:space:]]+)?runs-on:' <<< "$p"; then
            echo "reason=runs-on changed in $f"; hit=1
          fi ;;
      esac
    done < "$diff"
  fi
  if [ "$n" -eq 0 ]; then echo "reason=empty path list, cannot judge"; echo "applicable=yes"; return 0; fi
  if [ "$hit" = 1 ]; then echo "applicable=yes"; else echo "applicable=no"; fi
}

from_git() { # BASE HEAD
  local base="$1" head="$2" mb d
  d="$(mktemp -d "${TMPDIR:-/tmp}/npf-scope.XXXXXX")" || return 2
  mb="$(git merge-base "$base" "$head" 2>/dev/null)" || { echo "reason=no merge-base for $base $head"; echo "applicable=yes"; rm -rf -- "${d:?}"; return 0; }
  if ! git -c core.quotepath=off diff --no-renames --name-only "$mb" "$head" > "$d/paths" 2>/dev/null \
    || ! git -c core.quotepath=off diff --no-renames -U0 "$mb" "$head" -- .github/workflows > "$d/diff" 2>/dev/null; then
    echo "reason=git diff failed"; echo "applicable=yes"; rm -rf -- "${d:?}"; return 0
  fi
  classify "$d/paths" "$d/diff"
  rm -rf -- "${d:?}"
}

self_test() {
  local bad=0 got
  d="$(mktemp -d "${TMPDIR:-/tmp}/npf-scope-t.XXXXXX")" || exit 2
  trap 'rm -rf -- "${d:?}"' EXIT
  expect() { # NAME WANT PATHS(printf %b) [DIFF(printf %b)]
    printf '%b' "$3" > "$d/p"; : > "$d/d"; [ -n "${4:-}" ] && printf '%b' "$4" > "$d/d"
    got="$(classify "$d/p" "$d/d" | grep '^applicable=')"
    if [ "$got" = "applicable=$2" ]; then echo "  ok    $1"; else echo "  BAD   $1: got '$got' want '$2'"; bad=$((bad + 1)); fi
  }
  expect nightly_train_workflow yes '.github/workflows/nightly-train.yml\n'
  expect nightly_yml yes '.github/workflows/nightly.yml\n'
  expect other_nightly_workflow yes '.github/workflows/mutants-nightly.yml\n'
  expect release_workflow yes '.github/workflows/release.yml\n'
  expect release_rehearsal yes '.github/workflows/release-gates-nightly.yml\n'
  expect rc_cut yes '.github/workflows/rc-cut.yml\n'
  expect rust_toolchain yes 'rust-toolchain.toml\n'
  expect tools_toml yes 'tools.toml\n'
  expect pmat_pin yes 'scripts/pmat_pin.sh\n'
  expect bashrs_pin yes 'scripts/bashrs_pin.sh\n'
  expect llama_pin_toml yes 'scripts/llama_pin.toml\n'
  expect nightly_train_sh yes 'scripts/release/nightly_train.sh\n'
  expect release_dir_any_file yes 'scripts/release/ledger.py\n'
  expect pool_checker yes 'scripts/check_nightly_train_pool.sh\n'
  expect preflight_itself yes 'scripts/ci/nightly_preflight.sh\n'
  expect mixed_with_unrelated yes 'src/lib.rs\ncrates/a/src/x.rs\nscripts/pmat_pin.sh\n'
  expect runs_on_in_plain_workflow yes '.github/workflows/ci.yml\n' '--- a/.github/workflows/ci.yml\n+++ b/.github/workflows/ci.yml\n@@ -1 +1 @@\n-    runs-on: [self-hosted, Linux, X64, clean-room]\n+    runs-on: [self-hosted, Linux, X64, clean-room, intel]\n'
  expect runs_on_list_item_form yes '.github/workflows/ci.yml\n' '--- a/.github/workflows/ci.yml\n+++ b/.github/workflows/ci.yml\n@@ -1 +1 @@\n+  - runs-on: [x]\n'
  expect empty_path_list_fail_closed yes ''
  expect plain_source_change no 'crates/aprender-core/src/lib.rs\ndocs/x.md\n'
  expect plain_workflow_no_runs_on no '.github/workflows/ci.yml\n' '--- a/.github/workflows/ci.yml\n+++ b/.github/workflows/ci.yml\n@@ -1 +1 @@\n-      - run: echo a\n+      - run: echo b\n'
  expect runs_on_in_non_workflow_diff_ignored no 'docs/x.md\n' '--- a/docs/x.md\n+++ b/docs/x.md\n@@ -1 +1 @@\n+runs-on: x\n'
  expect runs_on_after_workflow_in_other_file_ignored no '.github/workflows/ci.yml\ndocs/x.md\n' '--- a/.github/workflows/ci.yml\n+++ b/.github/workflows/ci.yml\n@@ -1 +1 @@\n-a\n+b\n--- a/docs/x.md\n+++ b/docs/x.md\n@@ -1 +1 @@\n+runs-on: x\n'
  expect quoted_path_fail_closed yes '"scripts/release/a\\303\\251.sh"\n'
  expect path_with_spaces_hit yes 'scripts/release/my script.sh\n'
  expect path_with_spaces_miss no 'docs/my notes.md\n'
  expect deleted_pin_file_is_a_path yes 'scripts/pmat_pin.sh\n'
  expect lookalike_dirs no 'scripts/releases/x.sh\nxscripts/release/y.sh\nscripts/nightly/x.rs\n'
  expect lookalike_names no '.github/workflows/ci-release-notes.yml.bak\nscripts/mypin.sh\n'
  # reason lines are printed for a hit
  printf 'tools.toml\n' > "$d/p"
  got="$(classify "$d/p")"
  if grep -q '^reason=tool pin tools.toml$' <<< "$got"; then echo "  ok    reason_line"; else echo "  BAD   reason_line"; bad=$((bad + 1)); fi
  # Finding 1: deleted workflow file containing runs-on
  expect deleted_workflow_runs_on yes '.github/workflows/ci.yml\n' '--- a/.github/workflows/ci.yml\n+++ /dev/null\n@@ -1,3 +0,0 @@\n-jobs:\n-  build:\n-    runs-on: [self-hosted, intel]\n'
  # Finding 2: non-ASCII workflow path in git diff
  mkdir -p "$d/g/.github/workflows"
  git -C "$d/g" init -q 2>/dev/null
  git -C "$d/g" config user.email t@example.com
  git -C "$d/g" config user.name t
  printf 'jobs:\n  b:\n    runs-on: ubuntu-latest\n' > "$d/g/.github/workflows/$(printf 'caf\303\251.yml')"
  git -C "$d/g" add . && git -C "$d/g" commit -q -m init
  local base_sha head_sha
  base_sha="$(git -C "$d/g" rev-parse HEAD)"
  printf 'jobs:\n  b:\n    runs-on: [self-hosted, intel]\n' > "$d/g/.github/workflows/$(printf 'caf\303\251.yml')"
  git -C "$d/g" add . && git -C "$d/g" commit -q -m change
  head_sha="$(git -C "$d/g" rev-parse HEAD)"
  got="$(GIT_DIR="$d/g/.git" from_git "$base_sha" "$head_sha" | grep '^applicable=')"
  if [ "$got" = "applicable=yes" ]; then echo "  ok    non_ascii_workflow_git_diff"; else echo "  BAD   non_ascii_workflow_git_diff: got '$got' want 'applicable=yes'"; bad=$((bad + 1)); fi
  # Finding 3: last path without trailing newline
  expect last_path_no_trailing_nl yes 'src/lib.rs\ntools.toml'
  # Finding 4: unreadable or missing DIFF arg fails closed
  printf 'src/lib.rs\n' > "$d/p"
  got="$(bash "$HERE/$(basename "${BASH_SOURCE[0]}")" --classify "$d/p" "$d/does_not_exist.diff" | grep '^applicable=')"
  if [ "$got" = "applicable=yes" ]; then echo "  ok    missing_diff_fail_closed"; else echo "  BAD   missing_diff_fail_closed: got '$got' want 'applicable=yes'"; bad=$((bad + 1)); fi
  got="$(bash "$HERE/$(basename "${BASH_SOURCE[0]}")" --classify "$d/p" "" | grep '^applicable=')"
  if [ "$got" = "applicable=yes" ]; then echo "  ok    empty_diff_arg_fail_closed"; else echo "  BAD   empty_diff_arg_fail_closed: got '$got' want 'applicable=yes'"; bad=$((bad + 1)); fi
  echo "$PROG self-test: $bad broke"
  [ "$bad" = 0 ]
}

mutants() { # each planted defect in a copy of this script must turn --self-test red (literal replace)
  local n=0 bad=0 pat rep m src out
  d="$(mktemp -d "${TMPDIR:-/tmp}/npf-scope-m.XXXXXX")" || exit 2
  trap 'rm -rf -- "${d:?}"' EXIT
  src="$(cat "$HERE/$PROG")"
  while IFS=$'\t' read -r pat rep; do
    [ -n "$pat" ] || continue
    n=$((n + 1)); m="$d/m$n.sh"
    out="${src/"$pat"/"$rep"}"
    if [ "$out" = "$src" ]; then echo "  BAD   mutant $n did not change the script: $pat"; bad=$((bad + 1)); continue; fi
    printf '%s\n' "$out" > "$m"
    if bash "$m" --self-test > /dev/null 2>&1; then echo "  BAD   mutant $n SURVIVED: $pat -> $rep"; bad=$((bad + 1)); else echo "  ok    mutant $n killed: $pat -> $rep"; fi
  done <<'M'
.github/workflows/*nightly*.yml |	.github/workflows/NOPE.yml |
rust-toolchain.toml | tools.toml |	rust-toolchain.toml |
scripts/*_pin.sh | scripts/*pin*.toml)	scripts/*pin*.toml)
scripts/release/*) echo	scripts/releasex/*) echo
scripts/check_nightly_train_pool.sh | scripts/nightly_*.sh |	scripts/nightly_*.sh |
.github/workflows/release*.yml |	.github/workflows/releasex.yml |
.github/workflows/rc-cut.yml \	.github/workflows/rc-cutx.yml \
scripts/ci/nightly_preflight*.sh)	scripts/ci/xpreflight*.sh)
runs-on:'	runs-onX:'
if [ "$n" -eq 0 ]; then echo "reason=empty path list, cannot judge"; echo "applicable=yes"; return 0; fi	:
if [ "$hit" = 1 ]; then echo "applicable=yes"; else echo "applicable=no"; fi	echo "applicable=no"
"+++ "* | "--- "*) f="" ;;	"+++ "* | "--- "*) f="$f" ;;
"+++ b/.github/workflows/"* | "--- a/.github/workflows/"*) f="${p#??? ?/}" ;;	"--- a/.github/workflows/"*) f="${p#??? ?/}" ;;
case "$p" in '"'*) echo "reason=quoted path $p, cannot judge"; hit=1; continue ;; esac	case "$p" in '"'*) continue ;; esac
"+++ /dev/null") ;;	"+++ /dev/null") f="" ;;
git -c core.quotepath=off diff --no-renames -U0	git diff --no-renames -U0
while IFS= read -r p || [ -n "$p" ]; do	while IFS= read -r p; do
if [ -z "$diff" ] || [ ! -r "$diff" ] || [ -d "$diff" ]; then	if false; then
M
  echo "$PROG mutants: $n planted, $bad not killed"
  [ "$bad" = 0 ]
}

# two args that are not an option: BASE HEAD, even when BASE is empty (a merge_group event can carry no base_sha): from_git fails closed
if [ $# -eq 2 ] && [ "${1#-}" = "$1" ]; then from_git "$1" "$2"; exit $?; fi
case "${1:-}" in
  --self-test) self_test; exit $? ;;
  --mutants) mutants; exit $? ;;
  --classify)
    [ -r "${2:-}" ] && [ ! -d "${2:-}" ] || { echo "$PROG: unreadable paths file" >&2; exit 2; }
    if [ $# -ge 3 ]; then classify "$2" "$3"; else classify "$2"; fi
    exit 0 ;;
  -h | --help | "") echo "usage: $PROG BASE HEAD | --classify PATHS [DIFF] | --self-test | --mutants"; [ -n "${1:-}" ]; exit $? ;;
esac
[ $# -eq 2 ] || { echo "usage: $PROG BASE HEAD" >&2; exit 2; }
from_git "$1" "$2"
