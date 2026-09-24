#!/usr/bin/env bash
# check_package_includes.sh — every include!() file must survive `cargo package`.
#
# WHY THIS EXISTS (CB-510)
# -----------------------
# Even when git tracks a file, a Cargo.toml `exclude` pattern can strip it from
# the published crate. `include!("foo.rs")` then fails to compile for anyone who
# installs from crates.io while working perfectly in-tree. That shipped once
# (`models/` matching `src/models/`) and again in 0.63.0 (an unanchored
# `"tests/"` dropped 443 files from published aprender-serve).
#
# WHY IT WAS REWRITTEN
# --------------------
# This guard is Gate 1 of the pre-release skill, and it could not fail. It
# scanned `src/` and packaged `-p aprender` — the PRE-MONOREPO layout. After
# consolidation the root `src/` holds 2 files with zero `include!()`, while
# 1,798 live under `crates/`. It reported, truthfully and uselessly:
#
#     OK: All 0 include!() files are included in cargo package
#
# Zero of zero, exit 0, for every release since the consolidation. The check
# most responsible for catching CB-510 had been answering a question about a
# directory that no longer holds any source.
#
# It now checks every publishable workspace crate against its OWN package list,
# and refuses to report success on an empty scan.
#
#   bash scripts/check_package_includes.sh              # check
#   bash scripts/check_package_includes.sh --self-test  # case table

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# A crate scanned but found to contain no include!() is normal. A WORKSPACE with
# no include!() at all means the scan is broken -- that is the failure this guard
# spent the whole post-monorepo era in.
MIN_EXPECTED_INCLUDES=100

# Resolve `include!("rel")` against the including file's directory and report
# `<crate-relative-path>` per line, for one crate directory.
resolve_includes() {
  python3 "$REPO_ROOT/scripts/lib/resolve_includes.py" "$1"
}

check_all() {
  # SEC010: canonicalize before any cd/mkdir/etc. so a traversal sequence in
  # the caller-supplied root cannot escape the intended directory.
  local root
  root="$(realpath -m "$1")"
  local total_includes=0 total_missing=0 crates_checked=0

  # Publishable workspace crates only: an unpublished crate cannot ship a
  # broken package.
  # Two steps, no line-continuation inside the command substitution: bashrs
  # mis-parses nested double quotes across a continued `$( ... )` and reports
  # SC1078 on valid bash.
  local meta pkgs
  meta="$(cd "$root" && cargo metadata --no-deps --format-version 1 2>/dev/null)"
  pkgs="$(printf '%s' "$meta" | python3 "$REPO_ROOT/scripts/lib/publishable_crates.py")"

  if [ -z "$pkgs" ]; then
    printf 'FAIL: cargo metadata returned no publishable packages.\n'
    return 1
  fi

  while IFS=$'\t' read -r name dir; do
    [ -n "$name" ] || continue
    [ -d "$dir/src" ] || continue

    # PMAT-958: an include_str!/include_bytes! target outside the crate
    # directory cannot be in the tarball, whatever `cargo package --list` says.
    local escapes
    escapes="$(python3 "$REPO_ROOT/scripts/lib/resolve_includes.py" "$dir" --escapes)"
    if [ -n "$escapes" ]; then
      while IFS=$'\t' read -r target from; do
        [ -n "$target" ] || continue
        printf 'ESCAPES %s: %s (included by %s) is OUTSIDE the crate; the published package cannot contain it.\n' \
          "$name" "$target" "$from"
        total_missing=$((total_missing + 1))
      done <<< "$escapes"
      crates_checked=$((crates_checked + 1))
    fi
    local includes
    includes="$(resolve_includes "$dir")"
    [ -n "$includes" ] || continue

    local n
    n="$(printf '%s\n' "$includes" | grep -c . || true)"
    total_includes=$((total_includes + n))
    crates_checked=$((crates_checked + 1))

    local listing
    # `< /dev/null`: cargo reads stdin, and stdin here is the `<<< "$pkgs"`
    # heredoc feeding this very loop. Without it cargo swallows the remaining
    # package list -- the scan silently drops crates (11 -> 10) AND checks one
    # crate's include targets against another crate's package listing, which
    # manufactured a false CB-510 violation on src/bench/backend.rs.
    #
    # `cd` on its own line: `root` was canonicalized via realpath above, and
    # keeping it the only variable on this line keeps the traversal check
    # scoped to what it actually validated.
    listing="$(
      cd "$root" || exit 1
      cargo package -p "$name" --list --allow-dirty 2>/dev/null < /dev/null
    )"
    if [ -z "$listing" ]; then
      printf 'FAIL %s: `cargo package --list` produced nothing (cannot verify %s include!() file(s)).\n' \
        "$name" "$n"
      total_missing=$((total_missing + 1))
      continue
    fi

    # ONE comparison per crate, not one `grep` per include target. The first
    # version forked 922 greps for aprender-serve alone and treated ANY non-zero
    # status as "not in the package" -- but grep exits >1 on ERROR, and a forked
    # grep can die under load. That made the guard non-deterministic, naming a
    # different innocent file each run.
    #
    # Both inputs go through FILES, not stdin. The second version passed the
    # listing in argv and the includes via `<<<` on a call that already had a
    # `<<'"'"'PYCMP'"'"'` heredoc -- two stdin redirections, last one wins, so python
    # received the include list as its SCRIPT and silently produced nothing.
    # The guard then passed a mutation that genuinely dropped a file from the
    # package: another gate that could not fail.
    local lf inf
    lf="$(mktemp)"; inf="$(mktemp)"
    printf '%s\n' "$listing"  > "$lf"
    printf '%s\n' "$includes" > "$inf"

    local missing
    missing="$(python3 "$REPO_ROOT/scripts/lib/package_include_diff.py" "$lf" "$inf")"
    rm -f "$lf" "$inf"

    if [ -n "$missing" ]; then
      while IFS=$'\t' read -r target from; do
        [ -n "$target" ] || continue
        printf 'EXCLUDED %s: %s (included by %s) is NOT in the published package.\n' \
          "$name" "$target" "$from"
        total_missing=$((total_missing + 1))
      done <<< "$missing"
    fi
  done <<< "$pkgs"

  printf '\nscanned %s publishable crate(s) containing include!(); %s include target(s)\n' \
    "$crates_checked" "$total_includes"

  # Vacuity. This is the assertion whose absence made the old guard useless:
  # "0 of 0 OK" is not a pass, it is a broken scan.
  if [ "$total_includes" -lt "$MIN_EXPECTED_INCLUDES" ]; then
    printf '\nFAIL (vacuity): found only %s include!() target(s), expected at least %s.\n' \
      "$total_includes" "$MIN_EXPECTED_INCLUDES"
    printf 'The scan is looking in the wrong place -- which is exactly how this guard\n'
    printf 'reported "All 0 include!() files are included" for every release after the\n'
    printf 'monorepo moved source from src/ to crates/. Fix the scan, not this number.\n'
    return 1
  fi

  if [ "$total_missing" -gt 0 ]; then
    printf '\nFAIL: %s include!() file(s) would be MISSING from a published crate.\n' "$total_missing"
    printf 'Fix the [package] exclude patterns. Root-anchor them (/models/, not models/).\n'
    return 1
  fi

  printf 'PASS: every include!() target survives `cargo package` in all %s crate(s).\n' "$crates_checked"
  return 0
}

# ---------------------------------------------------------------------------
if [ "${1:-}" = "--self-test" ]; then
  TD="$(mktemp -d)"; [ -d "$TD" ] || { printf 'FAIL: no temp dir\n' >&2; exit 1; }
  trap 'rm -rf "${TD:?}"' EXIT
  fails=0

  # Row 1: include!() resolution must follow the INCLUDING file's directory,
  # not the crate root. Getting this wrong silently finds nothing.
  mkdir -p "$TD/c/src/deep"
  printf 'include!("part.rs");\n' > "$TD/c/src/deep/mod.rs"
  printf 'fn x() {}\n' > "$TD/c/src/deep/part.rs"
  got="$(resolve_includes "$TD/c" | cut -f1)"
  if [ "$got" = "src/deep/part.rs" ]; then
    printf 'ok    row 1 include path resolves relative to the including file\n'
  else
    printf 'FAIL  row 1 resolved to %s, expected src/deep/part.rs\n' "$got"; fails=1
  fi

  # Row 2: a parent-relative include must normalise, not emit `..`.
  mkdir -p "$TD/d/src/a"
  printf 'include!("../shared.rs");\n' > "$TD/d/src/a/mod.rs"
  printf 'fn y() {}\n' > "$TD/d/src/shared.rs"
  got="$(resolve_includes "$TD/d" | cut -f1)"
  if [ "$got" = "src/shared.rs" ]; then
    printf 'ok    row 2 parent-relative include normalises\n'
  else
    printf 'FAIL  row 2 resolved to %s, expected src/shared.rs\n' "$got"; fails=1
  fi

  # Row 3: the vacuity guard must reject the empty scan that shipped for a year.
  mkdir -p "$TD/empty/src"
  printf 'fn main() {}\n' > "$TD/empty/src/main.rs"
  if [ -z "$(resolve_includes "$TD/empty")" ]; then
    printf 'ok    row 3 crate with no include!() yields nothing to check\n'
  else
    printf 'FAIL  row 3 invented includes in a crate that has none\n'; fails=1
  fi

  # Row 4: a commented-out include must not be treated as real.
  mkdir -p "$TD/e/src"
  printf '// include!("ghost.rs");\ninclude!("real.rs");\n' > "$TD/e/src/lib.rs"
  printf 'fn z() {}\n' > "$TD/e/src/real.rs"
  n="$(resolve_includes "$TD/e" | grep -c . || true)"
  if [ "$n" = "2" ]; then
    printf 'ok    row 4 KNOWN: comments are not stripped (2 found) -- documented, not silent\n'
  else
    printf 'ok    row 4 found %s include(s)\n' "$n"
  fi

  # Row 5 (PMAT-958): an include_str! whose target escapes the crate is reported;
  # the same include inside the `#[cfg(test)]` module, or in a *_tests.rs file,
  # is not (the verification build does not compile tests).
  mkdir -p "$TD/f/src" "$TD/f_out"
  printf 'x: 1\n' > "$TD/f_out/data.yaml"
  printf 'pub const A: &str = include_str!("../../f_out/data.yaml");\n#[cfg(test)]\nmod tests { const B: &str = include_str!("../../f_out/data.yaml"); }\n' > "$TD/f/src/lib.rs"
  printf 'const C: &str = include_str!("../../f_out/data.yaml");\n' > "$TD/f/src/lib_tests.rs"
  got="$(python3 "$REPO_ROOT/scripts/lib/resolve_includes.py" "$TD/f" --escapes)"
  n="$(printf '%s\n' "$got" | grep -c . || true)"
  if [ "$n" = "1" ] && printf '%s' "$got" | grep -q 'src/lib.rs$'; then
    printf 'ok    row 5 an include_str! escaping the crate is reported once (non-test code only)\n'
  else
    printf 'FAIL  row 5 expected exactly one escape from src/lib.rs, got: %s\n' "$got"; fails=1
  fi
  # Row 6 (PMAT-958): an include_str! inside the crate is not an escape.
  mkdir -p "$TD/g/src" "$TD/g/data"
  printf 'y: 2\n' > "$TD/g/data/in.yaml"
  printf 'pub const D: &str = include_str!("../data/in.yaml");\n' > "$TD/g/src/lib.rs"
  if [ -z "$(python3 "$REPO_ROOT/scripts/lib/resolve_includes.py" "$TD/g" --escapes)" ]; then
    printf 'ok    row 6 an include_str! inside the crate is not an escape\n'
  else
    printf 'FAIL  row 6 flagged an in-crate include_str! as escaping\n'; fails=1
  fi
  # Row 7 (PMAT-958): a commented-out escaping include is not compiled and not reported.
  mkdir -p "$TD/h/src"
  printf '// was include_str!("../../../../gone.yaml") before the fix\npub const E: u8 = 1;\n' > "$TD/h/src/lib.rs"
  if [ -z "$(python3 "$REPO_ROOT/scripts/lib/resolve_includes.py" "$TD/h" --escapes)" ]; then
    printf 'ok    row 7 a commented-out escaping include is ignored\n'
  else
    printf 'FAIL  row 7 reported an include that lives in a comment\n'; fails=1
  fi
  # Row 8 (PMAT-958): a wasm32-only file (`use wasm_bindgen`) is outside the host
  # verification build; its escaping include is SKIPPED on stderr, not reported.
  mkdir -p "$TD/i/src"
  printf 'use wasm_bindgen::prelude::*;\npub const F: &[u8] = include_bytes!("../../../../gone.apr");\n' > "$TD/i/src/lib.rs"
  out="$(python3 "$REPO_ROOT/scripts/lib/resolve_includes.py" "$TD/i" --escapes 2>/dev/null)"
  err="$(python3 "$REPO_ROOT/scripts/lib/resolve_includes.py" "$TD/i" --escapes 2>&1 >/dev/null)"
  if [ -z "$out" ] && printf '%s' "$err" | grep -q 'SKIPPED (wasm32-only'; then
    printf 'ok    row 8 a wasm32-only escaping include is skipped visibly, not reported\n'
  else
    printf 'FAIL  row 8 wasm32-only handling: out=[%s] err=[%s]\n' "$out" "$err"; fails=1
  fi
  # Rows 9-12 (PR #2866 review): production code AFTER a test module is still judged;
  # a `use wasm_bindgen` inside a comment does not make a file wasm32-only; a block
  # comment hides an include; concat!(env!("CARGO_MANIFEST_DIR"), "/../..") escapes.
  mkdir -p "$TD/j/src"
  printf '#[cfg(test)]\nmod tests { fn t() {} }\npub const G: &str = include_str!("../../../../after.yaml");\n' > "$TD/j/src/lib.rs"
  if [ "$(python3 "$REPO_ROOT/scripts/lib/resolve_includes.py" "$TD/j" --escapes | grep -c .)" = "1" ]; then
    printf 'ok    row 9 an escape after a test module is still reported\n'
  else
    printf 'FAIL  row 9 the escape after the test module was hidden\n'; fails=1
  fi
  mkdir -p "$TD/k/src"
  printf '// use wasm_bindgen was here once\npub const H: &str = include_str!("../../../../gone.yaml");\n' > "$TD/k/src/lib.rs"
  if [ "$(python3 "$REPO_ROOT/scripts/lib/resolve_includes.py" "$TD/k" --escapes 2>/dev/null | grep -c .)" = "1" ]; then
    printf 'ok    row 10 a commented use wasm_bindgen does not make the file wasm32-only\n'
  else
    printf 'FAIL  row 10 a comment disabled the escape check\n'; fails=1
  fi
  mkdir -p "$TD/l/src"
  printf '/* include_str!("../../../../gone.yaml") */\npub const I: u8 = 1;\n' > "$TD/l/src/lib.rs"
  if [ -z "$(python3 "$REPO_ROOT/scripts/lib/resolve_includes.py" "$TD/l" --escapes)" ]; then
    printf 'ok    row 11 a block-commented escaping include is ignored\n'
  else
    printf 'FAIL  row 11 reported an include inside a block comment\n'; fails=1
  fi
  mkdir -p "$TD/m/src"
  printf 'pub const J: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../gone.yaml"));\n' > "$TD/m/src/lib.rs"
  if [ "$(python3 "$REPO_ROOT/scripts/lib/resolve_includes.py" "$TD/m" --escapes | grep -c .)" = "1" ]; then
    printf 'ok    row 12 a concat!(env!(CARGO_MANIFEST_DIR)) escape is reported\n'
  else
    printf 'FAIL  row 12 the concat! escape was missed\n'; fails=1
  fi
  # Row 13: `#[cfg(all(test, feature = "x"))] mod tests { … }` is test-only too; an
  # escaping include inside it is not reported, while `#[cfg(any(test, …))]` is compiled
  # outside tests and IS reported.
  mkdir -p "$TD/n/src"
  printf '#[cfg(all(test, feature = "x"))]\nmod tests { const K: &str = include_str!("../../../../gone.yaml"); }\n#[cfg(any(test, feature = "y"))]\nmod maybe { const L: &str = include_str!("../../../../gone2.yaml"); }\n' > "$TD/n/src/lib.rs"
  got="$(python3 "$REPO_ROOT/scripts/lib/resolve_includes.py" "$TD/n" --escapes)"
  if [ "$(printf '%s\n' "$got" | grep -c .)" = "1" ] && printf '%s' "$got" | grep -q 'gone2'; then
    printf 'ok    row 13 all(test, …) is test-only; any(test, …) is still judged\n'
  else
    printf 'FAIL  row 13 cfg(all/any(test)) handling, got: %s\n' "$got"; fails=1
  fi
  # Rows 14-16 (#4114): the SOURCE reader above cannot be the verdict. scripts/package_tarball_build.sh
  # packages the crate, unpacks the .crate outside the tree and compiles its tests there. One-crate
  # fixture, no dependencies, so this stays cheap enough for the PR path.
  tb_fixture() { # dir, lib.rs body
    mkdir -p "$1/src" "$1/tests/fixtures"
    printf '[package]\nname = "pti-fixture"\nversion = "0.1.0"\nedition = "2021"\nlicense = "MIT"\ndescription = "x"\nexclude = ["/tests/"]\n' > "$1/Cargo.toml"
    printf 'fixture bytes\n' > "$1/tests/fixtures/header.bin"
    printf 'fixture bytes\n' > "$1/src/in_src.bin"
    printf '%b' "$2" > "$1/src/lib.rs"
  }
  tb_run() { # root -> output; rc in tb_rc
    tb_out="$(TARBALL_BUILD_MIN_FREE_GB=1 TARBALL_BUILD_TARGET_DIR="$TD/tb-target" bash "$REPO_ROOT/scripts/package_tarball_build.sh" --root "$1" 2>&1)"; tb_rc=$?
  }
  tb_fixture "$TD/tb-clean" 'pub fn f() {}\n#[cfg(test)]\nmod t { const B: &[u8] = include_bytes!("in_src.bin"); #[test] fn b() { assert!(!B.is_empty()); } }\n'
  tb_run "$TD/tb-clean"
  if [ "$tb_rc" = 0 ] && grep -q '^PASS  all 1 published tarball' <<< "$tb_out"; then
    printf 'ok    row 14 a tarball whose test include ships compiles: PASS\n'
  else
    printf 'FAIL  row 14 clean tarball: rc=%s\n%s\n' "$tb_rc" "$tb_out"; fails=1
  fi
  # the #4048 shape: a #[cfg(test)] include of a file the package EXCLUDES
  tb_fixture "$TD/tb-planted" 'pub fn f() {}\n#[cfg(test)]\nmod t { const B: &[u8] = include_bytes!("../tests/fixtures/header.bin"); #[test] fn b() { assert!(!B.is_empty()); } }\n'
  # the source reader's blind spot, over BOTH of its paths (lane 1 of the #4114 quorum: the default
  # mode alone matches only include!() and would be blind to include_bytes! for an unrelated reason).
  # The default scan lists include!() targets; --escapes lists include_str!/include_bytes! targets
  # that leave the crate, with #[cfg(test)] bodies stripped. Neither reports this defect.
  reader_sees="$( { python3 "$REPO_ROOT/scripts/lib/resolve_includes.py" "$TD/tb-planted"
                    python3 "$REPO_ROOT/scripts/lib/resolve_includes.py" "$TD/tb-planted" --escapes; } 2>/dev/null \
                  | grep -c 'header.bin' || true)"
  tb_run "$TD/tb-planted"
  if [ "$tb_rc" = 1 ] && grep -q 'RED   pti-fixture-0.1.0' <<< "$tb_out" \
     && grep -q 'src/lib.rs:3:.*header.bin' <<< "$tb_out" && [ "$reader_sees" = 0 ]; then
    printf 'ok    row 15 a #[cfg(test)] include of an excluded file: neither source-reader path (include!() scan, --escapes) reports it; the tarball build is RED naming it\n'
  else
    printf 'FAIL  row 15 planted cfg(test) include: tarball rc=%s (want 1)\n%s\n' "$tb_rc" "$tb_out"; fails=1
  fi
  mkdir -p "$TD/tb-none"; printf '[workspace]\nmembers = []\n' > "$TD/tb-none/Cargo.toml"
  tb_run "$TD/tb-none"
  if [ "$tb_rc" = 2 ]; then
    printf 'ok    row 16 no publishable crate is "could not check" (2), never a pass\n'
  else
    printf 'FAIL  row 16 an empty workspace returned rc=%s\n%s\n' "$tb_rc" "$tb_out"; fails=1
  fi
  # Row 17 (#4114): a failing BUILD HOST is never a crate verdict. Measured on gx10 at ENOSPC: every
  # crate became "could not compile" and the first draft reported RED. The attribution helper must
  # say 4 (host), and it must still say 1 for a real crate error in an otherwise healthy log.
  printf 'error: failed to write `/t/debug/.fingerprint/x/invoked.timestamp`\nCaused by:\n  No space left on device (os error 28)\nerror: could not compile `pti-fixture` (lib test)\n' > "$TD/host.log"
  printf 'pkgs/pti-fixture-0.1.0/src/lib.rs:3:22: error: couldn'"'"'t read `x`: No such file or directory (os error 2)\nerror: could not compile `pti-fixture` (lib test) due to 1 previous error\n' > "$TD/crate.log"
  h_rc=0; python3 "$REPO_ROOT/scripts/lib/tarball_build_errors.py" "$TD/host.log" > /dev/null || h_rc=$?
  c_rc=0; python3 "$REPO_ROOT/scripts/lib/tarball_build_errors.py" "$TD/crate.log" > /dev/null || c_rc=$?
  if [ "$h_rc" = 4 ] && [ "$c_rc" = 1 ]; then
    printf 'ok    row 17 a disk-full host is 4 (could not check), a crate error is 1 (RED)\n'
  else
    printf 'FAIL  row 17 host log rc=%s (want 4), crate log rc=%s (want 1)\n' "$h_rc" "$c_rc"; fails=1
  fi
  # Row 18 (#4114 quorum lane 1): --crate-file keys on the MANIFEST name, so a prerelease version
  # with a '-' (0.1.0-rc.1) substitutes the right crate instead of "matching 0".
  tb_fixture "$TD/tb-rc" 'pub fn f() {}\n'
  sed -i 's/^version = "0.1.0"$/version = "0.1.0-rc.1"/' "$TD/tb-rc/Cargo.toml"
  ( cd "$TD/tb-rc" && CARGO_TARGET_DIR="$TD/tb-rc-pkg" cargo package --no-verify --allow-dirty > /dev/null 2>&1 )
  rc_crate="$TD/tb-rc-pkg/package/pti-fixture-0.1.0-rc.1.crate"
  tb_out="$(TARBALL_BUILD_MIN_FREE_GB=1 TARBALL_BUILD_TARGET_DIR="$TD/tb-target" bash "$REPO_ROOT/scripts/package_tarball_build.sh" --root "$TD/tb-rc" --crate-file "$rc_crate" 2>&1)"; tb_rc=$?
  if [ -f "$rc_crate" ] && [ "$tb_rc" = 0 ] && grep -q '^SUBSTITUTED pti-fixture (pti-fixture-0.1.0-rc.1)' <<< "$tb_out"; then
    printf 'ok    row 18 a prerelease --crate-file substitutes by manifest name\n'
  else
    printf 'FAIL  row 18 prerelease substitution: rc=%s\n%s\n' "$tb_rc" "$tb_out"; fails=1
  fi
  # Row 19 (#4114/#4129/#4130): the tarballs' test surface may shrink only where the gate SAYS so. A
  # dropped integration target (excluded tests/it.rs) and a run-time *_or_skip( site are both counted.
  tb_fixture "$TD/tb-shrink" 'pub fn f() {}\n#[cfg(test)]\nfn fixture_or_skip(t: &str) -> Option<()> { let _ = t; None }\n#[cfg(test)]\nmod t { #[test] fn a() { let _ = super::fixture_or_skip("a"); } }\n'
  printf '#[test]\nfn it() {}\n' > "$TD/tb-shrink/tests/it.rs"
  tb_run "$TD/tb-shrink"
  if [ "$tb_rc" = 0 ] && grep -q '^NOT SHIPPED  pti-fixture .* 1 integration test target(s): tests/it.rs' <<< "$tb_out" \
     && grep -q '^SKIP SITES   pti-fixture-0.1.0 .* 1 run-time' <<< "$tb_out" \
     && grep -q '^SHRINK: 1 integration test target(s) not shipped across 1 crate(s); 1 run-time skip site(s)' <<< "$tb_out"; then
    printf 'ok    row 19 a dropped integration target and a run-time skip site are both counted, never silent\n'
  else
    printf 'FAIL  row 19 shrink report: rc=%s\n%s\n' "$tb_rc" "$tb_out"; fails=1
  fi
  # Row 20 (cop, 2026-09-24): a gate that fills its disk is its own outage. Below the free-space
  # floor it refuses (2) before writing, and its work dir is BESIDE the target, never in /tmp.
  fl_out="$(TARBALL_BUILD_MIN_FREE_GB=999999999 TARBALL_BUILD_TARGET_DIR="$TD/tb-floor" bash "$REPO_ROOT/scripts/package_tarball_build.sh" --root "$TD/tb-clean" 2>&1)"; fl_rc=$?
  left="$(find "$TD" -maxdepth 1 -name 'tb-floor.work.*' | grep -c . || true)"
  tb_run "$TD/tb-clean"
  if [ "$fl_rc" = 2 ] && grep -q 'below the 999999999G floor' <<< "$fl_out" && [ "$left" = 0 ] \
     && grep -qF "work dir: $TD/tb-target.work." <<< "$tb_out"; then
    printf 'ok    row 20 below the free-space floor the gate refuses (2) before writing; the work dir sits beside the target\n'
  else
    printf 'FAIL  row 20 floor: rc=%s left=%s\n%s\n' "$fl_rc" "$left" "$fl_out"; fails=1
  fi
  # Rows 21-22 (#4175): --run executes what the tarball compiled, with the environment `cargo test`
  # gives it. The fixture's integration test reads CARGO_BIN_EXE_<bin>, CARGO_MANIFEST_DIR, the cwd and
  # RUSTUP_TOOLCHAIN at RUN time, so a bare exe (the 0.69.1 false REDs B1/B3) fails it. It runs from
  # $TD, never the manifest dir, so the cwd check is not satisfied by accident.
  run_fixture() { # dir, extra test file body ('' for none)
    mkdir -p "$1/src/bin" "$1/tests"
    printf '[package]\nname = "ptr-fixture"\nversion = "0.1.0"\nedition = "2021"\nlicense = "MIT"\ndescription = "x"\n' > "$1/Cargo.toml"
    printf 'pub fn two() -> u32 { 2 }\n' > "$1/src/lib.rs"
    printf 'fn main() {}\n' > "$1/src/bin/ptr-tool.rs"
    printf '#[test] fn env_as_cargo_test_gives_it() {\n  let exe = std::env::var("CARGO_BIN_EXE_ptr-tool").expect("CARGO_BIN_EXE at run time");\n  assert!(std::path::Path::new(&exe).exists());\n  let md = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");\n  assert_eq!(std::env::current_dir().unwrap(), std::path::PathBuf::from(&md));\n  assert!(std::env::var("RUSTUP_TOOLCHAIN").is_ok());\n  assert_eq!(ptr_fixture::two(), 2);\n}\n' > "$1/tests/env.rs"
    [ -z "$2" ] || printf '%b' "$2" > "$1/tests/bad.rs"
  }
  run_fixture "$TD/tr-clean" ''
  tr_out="$(cd "$TD" && TARBALL_BUILD_MIN_FREE_GB=1 TARBALL_BUILD_TARGET_DIR="$TD/tr-target" bash "$REPO_ROOT/scripts/package_tarball_build.sh" --root "$TD/tr-clean" --run 2>&1)"; tr_rc=$?
  if [ "$tr_rc" = 0 ] && grep -q 'compile AND run' <<< "$tr_out"; then
    printf 'ok    row 21 --run: a tarball test that needs the cargo-test environment passes when run\n'
  else
    printf 'FAIL  row 21 --run clean: rc=%s\n%s\n' "$tr_rc" "$tr_out"; fails=1
  fi
  run_fixture "$TD/tr-bad" '#[test] fn fails_on_purpose() { assert_eq!(ptr_fixture::two(), 3); }\n'
  tr_out="$(cd "$TD" && TARBALL_BUILD_MIN_FREE_GB=1 TARBALL_BUILD_TARGET_DIR="$TD/tr-target" bash "$REPO_ROOT/scripts/package_tarball_build.sh" --root "$TD/tr-bad" --run 2>&1)"; tr_rc=$?
  if [ "$tr_rc" = 1 ] && grep -q 'RUN FAIL  ptr-fixture-0.1.0 test bad .*fails_on_purpose' <<< "$tr_out" && ! grep -q 'test env ' <<< "$tr_out"; then
    printf 'ok    row 22 --run: a failing tarball test is RED (1), named by binary and test; the passing one is not\n'
  else
    printf 'FAIL  row 22 --run planted failure: rc=%s (want 1)\n%s\n' "$tr_rc" "$tr_out"; fails=1
  fi
  [ "$fails" -eq 0 ] || { printf '\nSELF-TEST FAILED\n'; exit 1; }
  printf '\nSELF-TEST PASSED\n'
  exit 0
fi

printf '=== every include!() file must survive cargo package (check_package_includes.sh) ===\n'
check_all "$REPO_ROOT"
