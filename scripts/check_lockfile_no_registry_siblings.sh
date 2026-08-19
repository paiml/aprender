#!/usr/bin/env bash
# check_lockfile_no_registry_siblings.sh — no in-tree crate name may resolve
# from crates.io in Cargo.lock.
#
# WHY THIS EXISTS
# ---------------
# `check_workspace_siblings_pathed.sh` scans MANIFESTS: it proves no Cargo.toml
# declares an in-tree crate from the registry. That is necessary and it passes.
# It is not sufficient, because a dependency we legitimately take from crates.io
# can drag an in-tree name in TRANSITIVELY, and no manifest in this repo mentions
# it. The lockfile is where that shows up, and nothing read the lockfile.
#
# Live at the time of writing: `whisper-apr = "0.2"` (declared optional by
# apr-cli, aprender-orchestrate and aprender-rag) resolves crates.io `aprender
# 0.27.8` — the monorepo depending on a published copy of ITSELF, 36 minors
# behind the workspace — which pulls registry `trueno`, `realizar`,
# `trueno-quant`, `batuta-common`, `renacer-core`. Separately `bashrs` pulls
# `batuta-common` through the non-optional dependency in aprender-compute-xtask.
#
# WHY NOT `cargo tree --duplicates`
# ---------------------------------
# That is the instrument the sibling guard's own header recommends, and it is
# structurally blind here. `--duplicates` reports one PACKAGE name resolved at
# two versions. But `trueno`, `realizar` and `batuta-common` are `[lib]` names —
# the packages are `aprender-compute`, `aprender-serve`, `aprender-common`. The
# lockfile therefore contains exactly ONE `trueno` package (the registry one), so
# it is not a duplicate and never will be. Measured: `--duplicates` catches 0 of
# 8 under default features and 2 of 8 under `--all-features`, which no workflow
# passes.
#
# The lockfile check is also feature-independent — Cargo.lock records optional
# dependencies regardless of feature selection, which is exactly why it sees the
# seven that `cargo tree` cannot reach without `--all-features`.
#
#   bash scripts/check_lockfile_no_registry_siblings.sh              # check
#   bash scripts/check_lockfile_no_registry_siblings.sh --self-test  # case table

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIB_DIR="${REPO_ROOT}/scripts/lib"
NAME_AWK="${LIB_DIR}/workspace_sibling_names.awk"
LOCK_PY="${LIB_DIR}/lockfile_registry_packages.py"

# Ratchet. Every entry is a registry package sharing a name with an in-tree
# crate: a supply-chain and semver hazard, and for `aprender` a self-referential
# published-crate cycle. The list may only SHRINK. Closing them needs the
# transitive source cut (vendor whisper-apr per APR-MONO, or drop its published
# `aprender` dependency), which is why this ratchets rather than fails outright.
BASELINE_FILE="${REPO_ROOT}/scripts/lockfile_registry_siblings_baseline.txt"

# Names the workspace defines: package names, plus `[lib] name =` aliases. Same
# extractor the manifest guard uses, so the two cannot disagree about what
# "in-tree" means.
intree_names() {
  local root="$1" f
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    if [ "$f" = "$root/Cargo.toml" ]; then
      awk -v ROOT=1 -f "$NAME_AWK" "$f"
    else
      awk -v ROOT=0 -f "$NAME_AWK" "$f"
    fi
  done < <(find "$root/crates" -name Cargo.toml -not -path '*/target/*' 2>/dev/null; printf '%s\n' "$root/Cargo.toml") | sort -u
}

# Registry-sourced package names in the lockfile, one per line.
registry_names() {
  python3 "$LOCK_PY" "$1"
}

collisions() {
  local root="$1" names lock
  names="$(intree_names "$root")"
  lock="$(registry_names "$root/Cargo.lock")"
  comm -12 <(printf '%s\n' "$names") <(printf '%s\n' "$lock" | sort -u)
}

# ---------------------------------------------------------------------------
if [ "${1:-}" = "--self-test" ]; then
  # Fixtures live in scripts/lib/lockfile_cases/ rather than inline heredocs:
  # bashrs parses an embedded heredoc as shell, so TOML `name = "serde"` reads as
  # SC1007 "space after =" -- 21 phantom errors. Same reason the awk and python
  # helpers are separate files.
  CASES="${LIB_DIR}/lockfile_cases"
  fails=0

  got="$(registry_names "$CASES/mixed.lock" | sort | tr '\n' ' ')"
  if [ "$got" = "aprender serde trueno " ]; then
    printf 'ok    row 1 registry packages extracted; path package excluded\n'
  else
    printf 'FAIL  row 1 got [%s], expected [aprender serde trueno ]\n' "$got"; fails=1
  fi

  if [ -z "$(registry_names "$CASES/all_local.lock")" ]; then
    printf 'ok    row 2 no registry sources yields nothing\n'
  else
    printf 'FAIL  row 2 invented a registry package\n'; fails=1
  fi

  got="$(registry_names "$CASES/source_after_local.lock" | tr '\n' ' ')"
  if [ "$got" = "remote-thing " ]; then
    printf 'ok    row 3 source does not leak to the preceding package\n'
  else
    printf 'FAIL  row 3 got [%s], expected [remote-thing ]\n' "$got"; fails=1
  fi

  if [ -z "$(registry_names "$CASES/git_source.lock")" ]; then
    printf 'ok    row 4 git source is not a registry source\n'
  else
    printf 'FAIL  row 4 counted a git dependency as registry\n'; fails=1
  fi

  [ "$fails" -eq 0 ] || { printf '\nSELF-TEST FAILED\n'; exit 1; }
  printf '\nSELF-TEST PASSED (4/4)\n'
  exit 0
fi

# ---------------------------------------------------------------------------
printf '=== no workspace-local crate name may resolve from crates.io (check_lockfile_no_registry_siblings.sh) ===\n'

if [ ! -f "$REPO_ROOT/Cargo.lock" ]; then
  printf 'FAIL: Cargo.lock is missing; this guard needs the committed lockfile.\n'
  exit 1
fi

NAMES="$(intree_names "$REPO_ROOT")"
LOCKNAMES="$(registry_names "$REPO_ROOT/Cargo.lock")"
n_names="$(printf '%s\n' "$NAMES" | grep -c . || true)"
n_lock="$(printf '%s\n' "$LOCKNAMES" | sort -u | grep -c . || true)"

# Vacuity, both sides. An empty name set or an unparsed lockfile would make the
# intersection trivially empty — the precise way a guard like this reports clean
# while measuring nothing.
if [ "$n_names" -lt 200 ]; then
  printf '\nFAIL (vacuity): only %s workspace-local crate name(s) found, expected 200+.\n' "$n_names"
  printf 'The name extractor is broken. Fix it rather than this number.\n'
  exit 1
fi
if [ "$n_lock" -lt 500 ]; then
  printf '\nFAIL (vacuity): only %s registry package(s) parsed from Cargo.lock, expected 500+.\n' "$n_lock"
  printf 'The lockfile parser is broken. Fix it rather than this number.\n'
  exit 1
fi

FOUND="$(collisions "$REPO_ROOT")"
count="$(printf '%s\n' "$FOUND" | grep -c . || true)"

printf '%s workspace-local name(s), %s registry package(s) in Cargo.lock\n' "$n_names" "$n_lock"

if [ "${1:-}" = "--update" ]; then
  printf '%s\n' "$FOUND" | grep . > "$BASELINE_FILE" || : > "$BASELINE_FILE"
  printf 'baseline set to %s collision(s)\n' "$count"
  exit 0
fi

if [ ! -f "$BASELINE_FILE" ]; then
  printf 'FAIL: %s missing. Run --update once to establish it.\n' "$BASELINE_FILE"
  exit 1
fi
baseline_count="$(grep -c . "$BASELINE_FILE" || true)"

printf '%s collision(s), baseline %s\n' "$count" "$baseline_count"

if [ "$count" -gt "$baseline_count" ]; then
  printf '\nFAIL: registry copies of workspace-local crates grew %s -> %s.\n' "$baseline_count" "$count"
  printf 'A crates.io package now shares a name with a workspace crate. Cargo will\n'
  printf 'happily compile both, and their types are mutually incompatible.\n\n'
  comm -13 <(sort "$BASELINE_FILE") <(printf '%s\n' "$FOUND" | grep . | sort) | sed 's|^|  NEW: |'
  printf '\nFind the transitive source with:  cargo tree -i <name>@<version>\n'
  exit 1
fi

if [ "$count" -lt "$baseline_count" ]; then
  printf '\nImproved: %s -> %s. Run --update to record it.\n' "$baseline_count" "$count"
fi

printf '\nPASS (ratcheted). Currently resolved from crates.io despite being workspace-local:\n'
printf '%s\n' "$FOUND" | grep . | sed 's|^|  |'
exit 0
