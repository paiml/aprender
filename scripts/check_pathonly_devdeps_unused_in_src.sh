#!/usr/bin/env bash
# check_pathonly_devdeps_unused_in_src.sh - src/ may not reference a dev-dep
# that `cargo publish` deletes.
#
# THE CLASS. A dev-dependency written as `{ path = "..." }` with no version,
# no git and no workspace inheritance carries NO SOURCE. `cargo publish` omits
# it from the published manifest entirely -- that omission is deliberate and is
# what lets a dev-dep cycle (aprender-compute -> aprender-core) be legal. But
# the crate's own `#[cfg(test)]` code is published with it. If that code names
# the stripped crate, the published tarball cannot compile its own tests.
#
# In-tree every build resolves the path, so `ci / gate`, `workspace-test` and
# every nightly are green on the same commit. The ONLY instrument that can see
# it is the clean room, which runs downstream of publish -- and it was red for
# eight consecutive runs (2026-09-08..2026-09-15) while v0.66.0 and v0.67.0
# shipped over it, so its signal never reached a PR. #3305.
#
# THE RULE. For each workspace member: any dev-dep with `path` and no source is
# invisible post-publish, so its extern name must not appear in that member's
# src/. Members may keep such deps (the cycle needs them) -- they may not USE
# them from src/. The fix is normally `{ workspace = true }`, which carries the
# version and survives the strip.
#
# THE BASELINE. Five other crates carried this defect when the guard was written
# -- they are latent, not benign: clean-room only ever compiled aprender-compute,
# so nothing had surfaced them. They are recorded in the baseline beside this
# script, keyed by (manifest, alias) and NOT by file:line, because line numbers
# drift inside a single PR. The ratchet is SHRINK-ONLY: a pair absent from the
# baseline fails, and a baseline pair with no remaining violation also fails, so
# a fix must delete its row in the same commit. Draining it to empty is #3306.
#
# Runs bare, no arguments, from anywhere in the repo. `--selftest` runs the case
# table only. A bare run does BOTH: the case table first, then the tree.
set -euo pipefail

REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
BASELINE="$REPO/scripts/pathonly_devdeps_baseline.txt"

# Fixture dir for the case table. Global so the EXIT trap can still see it, and
# validated before the rm: an empty or unset name must never reach `rm -rf`.
FIXTURES=""
cleanup() {
  if [ -n "${FIXTURES:-}" ] && [ -d "${FIXTURES:-}" ]; then
    rm -rf "${FIXTURES:?refusing to rm an empty path}"
  fi
  return 0
}
trap cleanup EXIT

# scan <root-dir> -> prints "manifest|alias|src-file:line:text" per violation
scan() {
  python3 - "$1" <<'PY'
import os, re, sys, tomllib

root = sys.argv[1]
KINDS = ("dev-dependencies",)

def sourceless(spec):
    """True when the dep carries a path and nothing that survives publish."""
    if not isinstance(spec, dict):
        return False
    if "path" not in spec:
        return False
    return not any(k in spec for k in ("version", "git", "workspace"))

for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = [d for d in dirnames if d not in (".git", "target", "node_modules")]
    if "Cargo.toml" not in filenames:
        continue
    manifest = os.path.join(dirpath, "Cargo.toml")
    src = os.path.join(dirpath, "src")
    if not os.path.isdir(src):
        continue
    try:
        with open(manifest, "rb") as fh:
            data = tomllib.load(fh)
    except (tomllib.TOMLDecodeError, OSError):
        continue
    aliases = [a for k in KINDS for a, s in (data.get(k) or {}).items() if sourceless(s)]
    if not aliases:
        continue
    pats = {a: re.compile(r"(?:\buse\s+|\bextern\s+crate\s+|\b)" + re.escape(a.replace("-", "_")) + r"\s*(?:::|;)")
            for a in aliases}
    for sdir, sdirs, sfiles in os.walk(src):
        sdirs[:] = [d for d in sdirs if d != "target"]
        for name in sfiles:
            if not name.endswith(".rs"):
                continue
            path = os.path.join(sdir, name)
            try:
                lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
            except OSError:
                continue
            for n, line in enumerate(lines, 1):
                stripped = line.lstrip()
                if stripped.startswith("//"):
                    continue
                for alias, pat in pats.items():
                    if pat.search(line):
                        rel_m = os.path.relpath(manifest, root)
                        rel_s = os.path.relpath(path, root)
                        print(f"{rel_m}|{alias}|{rel_s}:{n}:{stripped[:100]}")
PY
}

# ── case table: every row is a fixture the regex/selector must classify ──
selftest() {
  local tmp rc=0 out
  FIXTURES="$(mktemp -d)"
  if [ -z "$FIXTURES" ] || [ ! -d "$FIXTURES" ]; then
    printf 'FAIL: mktemp -d produced no directory\n'
    return 1
  fi
  tmp="$FIXTURES"

  mk() { # mk <name> <dep-line> <src-line>
    mkdir -p "$tmp/$1/src"
    printf '[package]\nname = "%s"\nversion = "0.1.0"\n\n[dev-dependencies]\n%s\n' "$1" "$2" > "$tmp/$1/Cargo.toml"
    printf '%s\n' "$3" > "$tmp/$1/src/lib.rs"
  }

  # MUST MATCH -- sourceless path dev-dep, referenced from src/
  mk hit_use      'sib = { path = "../sib" }'                        'use sib::thing;'
  mk hit_qualified 'sib = { path = "../sib" }'                       'pub use sib::engine::rng::SimRng;'
  mk hit_underscore 'two-words = { path = "../tw" }'                 'use two_words::x;'
  mk hit_extern   'sib = { path = "../sib" }'                        'extern crate sib;'

  # MUST NOT MATCH -- these all survive `cargo publish`, or are not a reference
  mk ok_versioned 'sib = { path = "../sib", version = "0.1.0" }'     'use sib::thing;'
  mk ok_workspace 'sib = { workspace = true }'                       'use sib::thing;'
  mk ok_git       'sib = { git = "https://x/y", path = "../sib" }'   'use sib::thing;'
  mk ok_registry  'sib = "1.0"'                                      'use sib::thing;'
  mk ok_unused    'sib = { path = "../sib" }'                        'pub fn f() {}'
  mk ok_comment   'sib = { path = "../sib" }'                        '// use sib::thing;'
  mk ok_substring 'sib = { path = "../sib" }'                        'use sibling::thing;'

  out="$(scan "$tmp" || true)"
  for row in hit_use hit_qualified hit_underscore hit_extern; do
    printf '%s\n' "$out" | grep -q "^$row/Cargo.toml|" || { printf 'FAIL: missed %s\n' "$row"; rc=1; }
  done
  for row in ok_versioned ok_workspace ok_git ok_registry ok_unused ok_comment ok_substring; do
    if printf '%s\n' "$out" | grep -q "^$row/Cargo.toml|"; then printf 'FAIL: false positive on %s\n' "$row"; rc=1; fi
  done
  cleanup
  FIXTURES=""
  [ "$rc" -eq 0 ] && printf 'PASS: case table (4 must-match, 7 must-not-match)\n'
  return "$rc"
}

main() {
  if [ "${1:-}" = "--selftest" ]; then selftest; return $?; fi
  selftest || return 1

  local out pairs rc=0 new stale
  out="$(scan "$REPO" || true)"
  pairs="$(printf '%s\n' "$out" | awk -F'|' 'NF>=2 {print $1"|"$2}' | sort -u)"

  # anything not in the baseline is a NEW violation
  new="$(comm -23 <(printf '%s\n' "$pairs" | grep -v '^$' || true) \
                  <(grep -vE '^\s*(#|$)' "$BASELINE" | sort -u))"
  # a baseline row with no violation left must be deleted in the same commit
  stale="$(comm -13 <(printf '%s\n' "$pairs" | grep -v '^$' || true) \
                    <(grep -vE '^\s*(#|$)' "$BASELINE" | sort -u))"

  if [ -n "$new" ]; then
    rc=1
    printf 'FAIL: src/ references a dev-dep that `cargo publish` deletes (#3305)\n\n'
    printf '%s\n' "$new" | while IFS='|' read -r manifest alias; do
      printf '  %s declares %s path-only, and src/ uses it:\n' "$manifest" "$alias"
      printf '%s\n' "$out" | awk -F'|' -v m="$manifest" -v a="$alias" \
        '$1==m && $2==a {print "      " $3}' | head -5
    done
    printf '\nFix: give the dep a source in the manifest -- normally { workspace = true }.\n'
    printf 'A dep that must stay path-only (a genuine publish cycle) may not be used from src/.\n\n'
  fi

  if [ -n "$stale" ]; then
    rc=1
    printf 'FAIL: the baseline is shrink-only -- these rows have no violation left\n'
    printf 'and must be deleted from %s in this commit:\n' "scripts/pathonly_devdeps_baseline.txt"
    printf '%s\n' "$stale" | sed 's/^/  /'
    printf '\n'
  fi

  if [ "$rc" -eq 0 ]; then
    printf 'OK: no NEW src/ reference to a publish-stripped dev-dep (%s baselined pair(s), #3306)\n' \
      "$(grep -cvE '^\s*(#|$)' "$BASELINE")"
  fi
  return "$rc"
}

main "$@"
