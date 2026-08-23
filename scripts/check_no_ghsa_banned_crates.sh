#!/usr/bin/env bash
# check_no_ghsa_banned_crates.sh — Cargo.lock may not contain a crate version
# carrying a KNOWN advisory that the RustSec database does not publish.
#
# WHY THIS EXISTS
# ---------------
# `cargo audit` and `cargo deny check advisories` both read ONE database:
# RustSec. GitHub's advisory database (GHSA), which Dependabot reads, is a
# strict superset for several ecosystems. When an advisory exists only in GHSA,
# every advisory gate in this repo is green while the vulnerable crate sits in
# Cargo.lock.
#
# That is not hypothetical. #2531: `parquet ^57` pulls `thrift 0.17.0`, which
# carries CVE-2026-43868 / GHSA-2f9f-gq7v-9h6m (excessive-size memory
# allocation, fixed in thrift 0.23.0). RustSec has NO thrift advisory at all --
#
#     git -C ~/.cargo/advisory-db grep -il thrift -- crates   # returns nothing
#
# -- so `cargo deny check advisories` printed "advisories ok" while the crate
# was in the graph. It was found by a Dependabot alert on a DOWNSTREAM repo
# (paiml-mcp-agent-toolkit #66), i.e. by a tool this repo does not run, about
# this repo's dependency. The advisory surface had a hole exactly the width of
# "GHSA minus RustSec", and nothing here could see into it.
#
# This guard closes that hole with an explicit, reviewable table. It is
# deliberately NOT a general vulnerability scanner: entries are added by hand
# when a GHSA-only advisory is found, and REMOVED when RustSec picks the
# advisory up (at which point cargo-deny owns it and a duplicate here is dead
# weight, exactly like a dead deny.toml exemption -- see
# scripts/check_deny_exemptions_live.sh).
#
# Text-only: reads Cargo.lock, builds nothing.
#
#   bash scripts/check_no_ghsa_banned_crates.sh
#   bash scripts/check_no_ghsa_banned_crates.sh --self-test
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------------------------------------------------------------------
# THE TABLE
#
#   name|first_fixed_version|advisory-id|reason
#
# A locked version STRICTLY LESS THAN first_fixed_version fails the guard.
# Every entry must name a real advisory and a real fixed version; "we do not
# like this crate" belongs in deny.toml [bans], not here.
# ---------------------------------------------------------------------------
GHSA_TABLE="
thrift|0.23.0|GHSA-2f9f-gq7v-9h6m (CVE-2026-43868)|Memory allocation with excessive size value. Not in RustSec. Entered via parquet <=58 (parquet >=59 dropped thrift for an in-tree codec); see #2531.
"

# version_lt A B -> true when A < B under `sort -V` ordering.
version_lt() {
  [ "$1" != "$2" ] && [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n 1)" = "$1" ]
}

# scan_lockfile <path> -> prints one "name version advisory reason" line per
# violation; returns 0 when clean, 1 when any violation was found, 2 when the
# lockfile could not be read (a scan that read nothing certifies nothing).
scan_lockfile() {
  lockfile="$1"
  if [ ! -r "$lockfile" ]; then
    printf 'UNREADABLE %s\n' "$lockfile"
    return 2
  fi

  # Emit "name<TAB>version" for every [[package]] block.
  pkgs="$(awk '
    /^\[\[package\]\]/ { name=""; version=""; next }
    /^name = / { gsub(/^name = "|"$/, ""); name=$0; next }
    /^version = / { gsub(/^version = "|"$/, ""); version=$0;
                    if (name != "") print name "\t" version; next }
  ' "$lockfile")"

  if [ -z "$pkgs" ]; then
    printf 'EMPTY %s\n' "$lockfile"
    return 2
  fi

  found=0
  while IFS='|' read -r banned_name fixed advisory reason; do
    [ -z "${banned_name:-}" ] && continue
    while IFS=$'\t' read -r pkg_name pkg_version; do
      [ "$pkg_name" = "$banned_name" ] || continue
      if version_lt "$pkg_version" "$fixed"; then
        printf '%s %s %s %s\n' "$pkg_name" "$pkg_version" "$advisory" "$reason"
        found=1
      fi
    done <<< "$pkgs"
  done <<< "$GHSA_TABLE"

  return "$found"
}

# ---------------------------------------------------------------------------
# --self-test: prove the matcher can turn RED and can turn GREEN.
#
# A guard that has never been shown failing is a guard that may be structurally
# incapable of failing. Both directions are asserted against synthetic
# lockfiles, so the case table runs on every CI invocation rather than only
# when someone remembers to mutate the real tree.
# ---------------------------------------------------------------------------
self_test() {
  printf '=== case table: check_no_ghsa_banned_crates.sh ===\n'
  tmp="$(mktemp -d)"
  trap 'rm -rf "${tmp:?}"' RETURN

  make_lock() {
    printf '# synthetic\n[[package]]\nname = "%s"\nversion = "%s"\n' "$1" "$2" > "$tmp/Cargo.lock"
  }

  fails=0
  assert() {  # assert <label> <expected_rc> <actual_rc>
    if [ "$2" -eq "$3" ]; then
      printf '  ok   %-46s rc=%s\n' "$1" "$3"
    else
      printf '  FAIL %-46s expected rc=%s got rc=%s\n' "$1" "$2" "$3"
      fails=$((fails + 1))
    fi
  }

  # MUST MATCH (rc=1): the exact version #2531 found, and anything below the fix.
  make_lock thrift 0.17.0; scan_lockfile "$tmp/Cargo.lock" > /dev/null 2>&1; assert 'thrift 0.17.0 (the #2531 version)' 1 $?
  make_lock thrift 0.22.9; scan_lockfile "$tmp/Cargo.lock" > /dev/null 2>&1; assert 'thrift 0.22.9 (just below the fix)'  1 $?
  make_lock thrift 0.9.0;  scan_lockfile "$tmp/Cargo.lock" > /dev/null 2>&1; assert 'thrift 0.9.0 (not lexically < 0.23)' 1 $?

  # MUST NOT MATCH (rc=0): the fixed version, a later one, and an unrelated crate.
  make_lock thrift 0.23.0; scan_lockfile "$tmp/Cargo.lock" > /dev/null 2>&1; assert 'thrift 0.23.0 (the fixed version)'   0 $?
  make_lock thrift 1.0.0;  scan_lockfile "$tmp/Cargo.lock" > /dev/null 2>&1; assert 'thrift 1.0.0 (above the fix)'        0 $?
  make_lock serde 1.0.0;   scan_lockfile "$tmp/Cargo.lock" > /dev/null 2>&1; assert 'serde 1.0.0 (unrelated crate)'       0 $?

  # VACUITY (rc=2): a scan that parsed nothing must not report "clean".
  : > "$tmp/Cargo.lock";   scan_lockfile "$tmp/Cargo.lock" > /dev/null 2>&1; assert 'empty lockfile is NOT clean'         2 $?
  rm -f "$tmp/Cargo.lock"; scan_lockfile "$tmp/Cargo.lock" > /dev/null 2>&1; assert 'missing lockfile is NOT clean'       2 $?

  if [ "$fails" -gt 0 ]; then
    printf '\nFAIL: %s case(s) failed. The guard does not do what it claims.\n' "$fails"
    return 1
  fi
  printf 'PASS: all cases behave as declared.\n'
  return 0
}

if [ "${1:-}" = "--self-test" ]; then
  self_test
  exit $?
fi

printf '=== no GHSA-only vulnerable crate in Cargo.lock (check_no_ghsa_banned_crates.sh) ===\n'

entries="$(printf '%s\n' "$GHSA_TABLE" | grep -c '|' || true)"
if [ "$entries" -lt 1 ]; then
  printf 'FAIL (vacuity): the GHSA table parsed 0 entries. Fix the table, not this check.\n'
  exit 1
fi

violations="$(scan_lockfile "$REPO_ROOT/Cargo.lock")"
rc=$?

if [ "$rc" -eq 2 ]; then
  printf 'FAIL: could not read Cargo.lock -- %s\n' "$violations"
  exit 1
fi

if [ "$rc" -ne 0 ]; then
  printf '\nFAIL: Cargo.lock contains a crate version with a known advisory that\n'
  printf 'RustSec does NOT carry, so `cargo deny check advisories` cannot see it:\n\n'
  printf '%s\n' "$violations" | sed 's|^|  |'
  printf '\nUpgrade the dependency that pulls it. Do not add it to deny.toml:\n'
  printf 'deny.toml exemptions are keyed by RUSTSEC id and there is no id to key.\n'
  exit 1
fi

printf 'PASS: %s GHSA-only advisory table entry/entries checked, none present in Cargo.lock.\n' "$entries"
exit 0
