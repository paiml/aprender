#!/usr/bin/env bash
# The cascade's facade ORDER gate must arm on every spelling cargo accepts.
#
# WHY THIS GUARD EXISTS (aprender#2628)
# -------------------------------------
# `facade_upstream_ready` in scripts/cascade-publish.sh decides whether a
# crates.io compatibility facade may be uploaded yet. A facade resolves its
# upstream FROM THE REGISTRY, so publishing it first yields a crate that cannot
# compile for anyone -- on an append-only registry.
#
# The original implementation read the requirement with a line-shaped `sed` and
# returned READY when it could not parse one. MEASURED: cargo treats the inline
# and multi-line dependency tables as identical, the sed understands only the
# inline one, and so a semantically null reformat DISARMED the gate silently.
#
# This guard is the falsifier for that class: it exercises the REAL functions
# (extracted from cascade-publish.sh, never re-implemented -- if they are
# renamed or deleted, extraction fails and this guard fails) against both
# spellings plus the two boundary cases.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CASCADE="$REPO_ROOT/scripts/cascade-publish.sh"
echo "=== cascade facade order gate (check_facade_order_gate.sh) ==="

[ -f "$CASCADE" ] || { echo "FAIL: $CASCADE not found"; exit 1; }

# Extract the two functions under test. Re-implementing them here would test a
# copy, which is the defect this repo keeps filing (see #2585).
WORK=$(mktemp -d)
# SEC011: never let the trap loose on an unvalidated path. `set -e` would catch
# a failing mktemp, but the cleanup must be safe on its own terms rather than
# relying on that.
case "$WORK" in
  /tmp/*|/var/folders/*) : ;;
  *) echo "FAIL: mktemp -d returned an unexpected path '$WORK'; refusing to clean up"; exit 1 ;;
esac
trap 'rm -rf "${WORK:?}"' EXIT
sed -n '/^facade_upstream_of() {/,/^}/p'   "$CASCADE" >  "$WORK/fns.sh"
sed -n '/^FACADE_EXPECTS_UPSTREAM=/p'      "$CASCADE" >> "$WORK/fns.sh"
for fn in facade_upstream_of FACADE_EXPECTS_UPSTREAM; do
  grep -q "$fn" "$WORK/fns.sh" || { echo "FAIL: could not extract '$fn' from cascade-publish.sh"; exit 1; }
done
# shellcheck disable=SC1091
. "$WORK/fns.sh"

# NOTE on the `|| true` on every facade_upstream_of call below: under
# `set -e`, a resolver whose last command legitimately evaluates false (e.g. a
# `[ -n "$x" ] && echo`) makes the enclosing assignment fail and KILLS this
# script mid-run -- it then exits non-zero having printed no FAIL line at all.
# That was observed while mutation-testing this very guard: the pre-#2628 sed
# resolver returns non-zero for a manifest it cannot parse, and the guard died
# after row 1 instead of reporting rows 2-8. A guard that fails without a
# diagnostic is the defect this repo keeps filing, so the failure of a resolver
# is captured as DATA (an empty string) and judged by the row, never by set -e.
rc=0
pass() { printf 'ok    %s\n' "$1"; }
fail() { printf 'FAIL  %s\n' "$1"; rc=1; }

# Build a throwaway two-crate workspace: one upstream, one facade. The facade's
# dependency spelling is the variable under test.
mk_case() {
  local dir=$1 spelling=$2
  mkdir -p "$dir/up/src" "$dir/facade/src"
  cat > "$dir/Cargo.toml" <<EOF
[workspace]
members = ["up", "facade"]
resolver = "2"
EOF
  cat > "$dir/up/Cargo.toml" <<EOF
[package]
name = "gate-upstream"
version = "0.63.0"
edition = "2021"
EOF
  : > "$dir/up/src/lib.rs"
  {
    printf '[package]\nname = "gate-facade"\nversion = "0.4.0"\nedition = "2021"\n\n'
    case "$spelling" in
      inline) printf '[dependencies]\nupstream = { path = "../up", version = "0.63.0", package = "gate-upstream" }\n' ;;
      multi)  printf '[dependencies.upstream]\npath = "../up"\nversion = "0.63.0"\npackage = "gate-upstream"\n' ;;
      none)   : ;;
    esac
  } > "$dir/facade/Cargo.toml"
  : > "$dir/facade/src/lib.rs"
}

# --- ROW 1/2: both spellings cargo accepts must RESOLVE identically ----------
for spelling in inline multi; do
  d="$WORK/$spelling"; mk_case "$d" "$spelling"
  got=$(facade_upstream_of "$d/facade/Cargo.toml" "gate-facade" || true)
  if [ "$got" = "gate-upstream 0.63.0" ]; then
    pass "'$spelling' dependency spelling resolves to: $got"
  else
    fail "'$spelling' dependency spelling resolved to '<$got>', expected 'gate-upstream 0.63.0' (#2628)"
  fi
done

# --- ROW 3: a facade with genuinely no upstream resolves to nothing ----------
d="$WORK/none"; mk_case "$d" none
got=$(facade_upstream_of "$d/facade/Cargo.toml" "gate-facade" || true)
if [ -z "$got" ]; then
  pass "a facade with no dependencies resolves to no upstream"
else
  fail "a facade with no dependencies resolved '<$got>', expected nothing"
fi

# --- ROW 4: the EXPECTATION list is the thing that makes row 3 safe ----------
# Row 3 alone is satisfied by a resolver that always returns nothing -- which is
# exactly the #2628 defect. What excludes that outcome is the positive list.
for c in provable-contracts provable-contracts-macros; do
  case " $FACADE_EXPECTS_UPSTREAM " in
    *" $c "*) pass "$c is declared to REQUIRE an upstream" ;;
    *)        fail "$c is missing from FACADE_EXPECTS_UPSTREAM — an unparseable manifest would read as READY (#2628)" ;;
  esac
done
case " $FACADE_EXPECTS_UPSTREAM " in
  *" provable-contracts-cli "*)
    fail "provable-contracts-cli must NOT be in FACADE_EXPECTS_UPSTREAM — it is the lib-only signpost facade with no dependencies (check_facade_compat.sh R3)" ;;
  *) pass "provable-contracts-cli is correctly exempt (signpost facade, no dependencies)" ;;
esac

# --- ROW 5: the real facades in this tree must still resolve -----------------
for c in provable-contracts provable-contracts-macros; do
  m="$REPO_ROOT/crates/facades/$c/Cargo.toml"
  if [ ! -f "$m" ]; then fail "$c manifest missing at $m"; continue; fi
  got=$(facade_upstream_of "$m" "$c" || true)
  if [ -n "$got" ]; then pass "$c resolves its upstream in-tree: $got"
  else fail "$c resolved NO upstream in-tree — the gate would return READY (#2628)"; fi
done

echo
if [ "$rc" -eq 0 ]; then
  echo "PASS: the facade order gate arms on every spelling cargo accepts, and"
  echo "      cannot be disarmed by a manifest it fails to parse."
else
  echo "FAILED: the cascade could publish a facade before its upstream."
fi
exit "$rc"
