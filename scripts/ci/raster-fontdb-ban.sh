#!/usr/bin/env bash
# APEX-001 EV-2d: prove that the resvg `default-features = false` pin in
# crates/aprender-viz/Cargo.toml is LIVE, not merely written down.
#
# WHY THIS EXISTS
#
# crates/aprender-viz/Cargo.toml pins:
#
#     resvg = { version = "0.45", optional = true, default-features = false }
#
# Turning resvg's default features back on compiles a font stack (fontdb, rustybuzz,
# ttf-parser, ...) into aprender-viz. That decision is currently protected by NOTHING that
# can fail: a Rust test cannot assert the absence of a method, and an empty font database
# behaves exactly like a missing one, so every behavioural test stays green whether or not
# the font stack is linked in. This script is the fixture that CAN fail.
#
# Measured on this checkout:
#
#     as shipped:               `cargo tree -p aprender-viz --features raster -e normal`
#                                matched fontdb/rustybuzz/ttf-parser 0 times
#     default-features removed: the same command matched them 4 times, including
#                                `fontdb v0.23.0` and `rustybuzz v0.20.1`
#
# Exit 0 = no font-stack crate is reachable from aprender-viz's `raster` feature.
# Exit 1 = at least one is, or the check itself could not run (see rule 2 below).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CRATE_TOML="$REPO_ROOT/crates/aprender-viz/Cargo.toml"
CARGO_LOCK="$REPO_ROOT/Cargo.lock"

# The banned set, defined ONCE — the self-test and the live check both read this list.
# Two lists is the defect this repository already has a guard against (see
# scripts/ci/libm-ban-live.sh).
#
# BANNED_EXACT: crate names banned outright.
# BANNED_PREFIX: any crate whose name STARTS WITH one of these is banned. Prefix, not
# substring — a crate that merely contains "font" in the middle of its name (an icon-font
# helper, say) is not part of resvg's font stack and must not be flagged. See
# arm (d) of --self-test.
BANNED_EXACT="fontdb rustybuzz ttf-parser"
BANNED_PREFIX="font"

# check_tree: runs the live `cargo tree` command from the repo root and prints, one per
# line, the name of every banned crate it finds. Returns 0 (clean) or 1 (found >=1, or the
# command itself was unusable).
#
# Takes the command to run as "$@" so --self-test can substitute a canned command that
# prints a fixed tree instead of invoking cargo, and a distinguishing REASON on stderr for
# each of the two ways this can fail loud.
check_tree() {
  local out rc
  set +e
  out=$("$@" -p aprender-viz --features raster -e normal --prefix none 2>&1)
  rc=$?
  set -e

  if [ "$rc" -ne 0 ]; then
    printf 'raster-fontdb-ban: FAIL — cargo tree itself failed (exit %s), not "clean":\n' "$rc" >&2
    printf '%s\n' "$out" >&2
    return 1
  fi

  if [ -z "$out" ]; then
    printf 'raster-fontdb-ban: FAIL — cargo tree printed NOTHING. An empty dependency\n' >&2
    printf '  graph is not evidence of a clean one; it means the command, the package\n' >&2
    printf '  name, or the feature name is wrong. Treating this as clean would let the\n' >&2
    printf '  ban silently stop checking anything.\n' >&2
    return 1
  fi

  local found=""
  local name
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    name=$(printf '%s\n' "$line" | awk '{print $1}')
    [ -n "$name" ] || continue
    for exact in $BANNED_EXACT; do
      if [ "$name" = "$exact" ]; then
        found="$found
$name"
      fi
    done
    for prefix in $BANNED_PREFIX; do
      case "$name" in
        "$prefix"*)
          found="$found
$name"
          ;;
      esac
    done
  done <<EOF
$out
EOF

  if [ -n "$found" ]; then
    # dedupe while preserving a stable order ($found is newline-separated; no word
    # splitting needed, so it is safe to double-quote)
    local uniq
    uniq=$(printf '%s\n' "$found" | sort -u)
    printf 'raster-fontdb-ban: FAIL — font-stack crate(s) reachable from aprender-viz raster:\n' >&2
    printf '%s\n' "$uniq" | while IFS= read -r n; do
      [ -n "$n" ] && printf '  %s\n' "$n" >&2
    done
    printf 'raster-fontdb-ban: resvg default-features is compiling a font stack in. See\n' >&2
    printf '  crates/aprender-viz/Cargo.toml, the resvg dependency line (APEX-001 EV-2d).\n' >&2
    return 1
  fi

  return 0
}

live_cargo_tree() {
  (cd "$REPO_ROOT" && cargo tree "$@")
}

run_default() {
  if check_tree live_cargo_tree; then
    printf 'raster-fontdb-ban: ok — no font-stack crate reachable from aprender-viz raster.\n'
    return 0
  fi
  return 1
}

# ---------------------------------------------------------------------------------------
# --self-test
# ---------------------------------------------------------------------------------------

self_test() {
  local pass=0
  local fail=0
  local rows=""

  add_row() {
    local case="$1" expected="$2" actual="$3"
    local ok="FAIL"
    if [ "$expected" = "$actual" ]; then
      ok="ok"
      pass=$((pass + 1))
    else
      ok="FAIL"
      fail=$((fail + 1))
    fi
    rows="$rows
$case | $expected | $actual | $ok"
  }

  # --- arm (a): the control on the controls. The tree as shipped must be GREEN. ---------
  local actual_a="RED"
  if check_tree live_cargo_tree >/dev/null 2>&1; then
    actual_a="GREEN"
  fi
  add_row "(a) as-shipped tree" "GREEN" "$actual_a"

  # --- arm (b): planting `resvg` default features back on must go RED and name fontdb. --
  local toml_backup lock_backup toml_sha_before lock_sha_before toml_sha_after
  toml_backup=$(mktemp)
  lock_backup=$(mktemp)
  cp "$CRATE_TOML" "$toml_backup"
  cp "$CARGO_LOCK" "$lock_backup"
  toml_sha_before=$(sha256sum "$CRATE_TOML" | awk '{print $1}')
  lock_sha_before=$(sha256sum "$CARGO_LOCK" | awk '{print $1}')

  restore_b() {
    cp "$toml_backup" "$CRATE_TOML"
    cp "$lock_backup" "$CARGO_LOCK"
    rm -f "$toml_backup" "$lock_backup"
  }
  trap restore_b RETURN

  sed -i.bak \
    's/resvg = { version = "0.45", optional = true, default-features = false }/resvg = { version = "0.45", optional = true }/' \
    "$CRATE_TOML"
  rm -f "$CRATE_TOML.bak"

  toml_sha_after=$(sha256sum "$CRATE_TOML" | awk '{print $1}')
  if [ "$toml_sha_after" = "$toml_sha_before" ]; then
    printf 'raster-fontdb-ban: FAIL — the sed plant for arm (b) matched nothing; the\n' >&2
    printf '  Cargo.toml resvg line is not what this script expects. Update the sed to\n' >&2
    printf '  match the current line, or arm (b) passes for the wrong reason.\n' >&2
    add_row "(b) plant landed" "changed" "unchanged"
    add_row "(b) planted tree RED, names fontdb" "RED+fontdb" "not-run"
  else
    add_row "(b) plant landed" "changed" "changed"

    # Route the planted tree through the SAME check_tree used by the live check and by
    # arms (a)/(c)/(d) — one matcher, one list, read by everyone.
    local out_b actual_b
    actual_b="GREEN"
    if out_b=$(check_tree live_cargo_tree 2>&1); then
      actual_b="GREEN"
    else
      actual_b="RED"
    fi
    if [ "$actual_b" = "RED" ] && printf '%s\n' "$out_b" | grep -q 'fontdb'; then
      add_row "(b) planted tree RED, names fontdb" "RED+fontdb" "RED+fontdb"
    else
      add_row "(b) planted tree RED, names fontdb" "RED+fontdb" "$actual_b"
    fi
  fi

  restore_b
  trap - RETURN

  local toml_sha_restored lock_sha_restored
  toml_sha_restored=$(sha256sum "$CRATE_TOML" | awk '{print $1}')
  lock_sha_restored=$(sha256sum "$CARGO_LOCK" | awk '{print $1}')
  if [ "$toml_sha_restored" = "$toml_sha_before" ] && [ "$lock_sha_restored" = "$lock_sha_before" ]; then
    add_row "(b) checkout restored byte-identical" "identical" "identical"
  else
    add_row "(b) checkout restored byte-identical" "identical" "DIFFERS"
  fi

  # --- arm (c): an empty tree is RED, and RED for the distinct "empty" reason. ----------
  empty_cargo_tree() { printf ''; }
  local actual_c="unknown"
  local msg_c
  msg_c=$(check_tree empty_cargo_tree 2>&1) && actual_c="GREEN" || actual_c="RED"
  if [ "$actual_c" = "RED" ] && printf '%s\n' "$msg_c" | grep -q 'printed NOTHING'; then
    actual_c="RED-empty-reason"
  fi
  add_row "(c) empty cargo-tree output" "RED-empty-reason" "$actual_c"

  # --- arm (d): a crate that merely CONTAINS "font" but does not START with it is NOT ---
  # banned. We match by prefix (name STARTS WITH "font"), not substring, so an unrelated
  # crate like `iconfont-sys` (an icon-font helper, "font" appears mid-name) must be
  # classified NOT-banned. A substring matcher would wrongly flag it; that is the failure
  # mode this arm exists to catch.
  unrelated_cargo_tree() { printf 'iconfont-sys v1.0.0\nserde v1.0.0\n'; }
  local actual_d="unknown"
  if check_tree unrelated_cargo_tree >/dev/null 2>&1; then
    actual_d="GREEN-not-banned"
  else
    actual_d="RED-wrongly-banned"
  fi
  add_row "(d) iconfont-sys (contains, not prefixed) not banned" "GREEN-not-banned" "$actual_d"

  printf 'raster-fontdb-ban --self-test:\n'
  printf 'case | expected | actual | ok\n'
  printf '%s\n' "$rows" | tail -n +2

  printf 'raster-fontdb-ban --self-test: %s ok, %s FAIL\n' "$pass" "$fail"

  if [ "$fail" -ne 0 ]; then
    return 1
  fi
  return 0
}

main() {
  if [ "${1:-}" = "--self-test" ]; then
    self_test
    exit $?
  fi
  run_default
  exit $?
}

main "$@"
