#!/usr/bin/env bash
# check_pathonly_devdeps_unused_in_src.sh - src/ may not reference a dev-dep
# that publishing to crates.io deletes.
#
# THE CLASS. A dev-dependency written as `{ path = "..." }` with no version,
# no git and no workspace inheritance carries NO SOURCE. The publish step omits
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
# WHO RUNS IT, AND WHY THIS FILE NEVER WRITES THE BUILD TOOL'S NAME WITH A SPACE
# AFTER IT (#3644). guard_tree.sh classifies a guard as cargo-using by the
# substring CARGO_RE, comments included; `guard-tree` runs only the cargo-free
# population (--no-cargo) and `guard-cargo` names its guards by hand. This file
# said the tool's name in three comments and one FAIL string, was classified
# cargo-using, was named by nothing, and RAN NOWHERE from the day it was
# written (2026-09-15) -- while check_guards_are_wired.sh reported PASS, blinded
# by a step name. It invokes no build tool: python over the manifests, then
# grep. Now it is in the --no-cargo population and guard-tree runs it.
#
# ENV IS NEVER A CODE VERDICT (#3644, the same shape as #3626's guard). The
# scanner needs a TOML reader: tomllib (python 3.11+) or tomli. The intel
# runner host has python 3.10.12 and neither; the old module-level import died
# with a traceback, `out=$(scan ... || true)` swallowed it, and the case table
# said "FAIL: missed hit_use" -- the regex reported broken by an interpreter
# that never ran it. Now the scanner ends with a `SCAN-DONE manifests=N`
# trailer that the shell REQUIRES; anything without it (no reader, a dead
# interpreter, no python3) is ENV rc=2 naming the interpreter and the runner --
# never rc=1, never "no violations".
#
# Runs bare, no arguments, from anywhere in the repo. `--selftest` runs the case
# table only. A bare run does BOTH: the case table first, then the tree.
#   PATHONLY_GUARD_PYTHON=<interpreter>   test seam: a dead one reproduces the intel shape
#   PATHONLY_GUARD_FORCE_NO_TOML=1        test seam: both readers fail to import, for real
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

# scan <root-dir> -> prints "manifest|alias|src-file:line:text" per violation.
# rc 0 with the trailer consumed; rc 2 (ENV) when the scanner did not finish.
scan() {
    local out rc=0 interp="${PATHONLY_GUARD_PYTHON:-python3}"
    out=$(PATHONLY_GUARD_RUNNER="${RUNNER_NAME:-unknown}" "$interp" - "$1" 2>&1 <<'PY'
import os, re, sys

runner = os.environ.get("PATHONLY_GUARD_RUNNER", "unknown")
if os.environ.get("PATHONLY_GUARD_FORCE_NO_TOML") == "1":
    sys.modules["tomllib"] = None   # the imports below then raise for real
    sys.modules["tomli"] = None

toml = None
for name in ("tomllib", "tomli"):
    try:
        toml = __import__(name)
        break
    except ImportError:
        continue
if toml is None:
    print("ENV: no TOML reader on this interpreter (python %d.%d, need tomllib >= 3.11 or tomli) runner=%s -- "
          "the manifests were not read; this is not 'no violations'" % (sys.version_info[0], sys.version_info[1], runner))
    sys.exit(2)

root = sys.argv[1]
KINDS = ("dev-dependencies",)
manifests = 0

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
            data = toml.load(fh)
    except (toml.TOMLDecodeError, OSError):
        continue
    manifests += 1
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
# positive evidence that the scan RAN TO THE END; the shell refuses any output without it
print("SCAN-DONE manifests=%d reader=%s" % (manifests, toml.__name__))
PY
    ) || rc=$?
    case "$out" in
        *"SCAN-DONE manifests="*) ;;
        *)
            printf 'ENV: the scanner did not finish (interpreter=%s rc=%s runner=%s) -- not a pass, not a code verdict\n' \
                "$interp" "$rc" "${RUNNER_NAME:-unknown}" >&2
            [ -z "$out" ] || printf '%s\n' "$out" | sed 's/^/     /' >&2
            return 2 ;;
    esac
    # the trailer is evidence, not a row
    printf '%s\n' "$out" | grep -v '^SCAN-DONE ' || true
    return 0
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

  # MUST NOT MATCH -- these all survive publishing, or are not a reference
  mk ok_versioned 'sib = { path = "../sib", version = "0.1.0" }'     'use sib::thing;'
  mk ok_workspace 'sib = { workspace = true }'                       'use sib::thing;'
  mk ok_git       'sib = { git = "https://x/y", path = "../sib" }'   'use sib::thing;'
  mk ok_registry  'sib = "1.0"'                                      'use sib::thing;'
  mk ok_unused    'sib = { path = "../sib" }'                        'pub fn f() {}'
  mk ok_comment   'sib = { path = "../sib" }'                        '// use sib::thing;'
  mk ok_substring 'sib = { path = "../sib" }'                        'use sibling::thing;'

  # ENV from the scanner is ENV here: rc 2, never "missed hit_use" (#3644)
  local env_rc=0
  out="$(scan "$tmp")" || env_rc=$?
  if [ "$env_rc" -ne 0 ]; then
    printf 'ENV: the case table could not be measured (scan rc=%s)\n' "$env_rc"
    cleanup; FIXTURES=""
    return 2
  fi

  # Piping a producer into a quiet grep is banned here
  # (check_no_pipe_into_grep_q.sh): grep exits on its first match and the
  # producer dies of SIGPIPE, which under `pipefail` is 141 -- a false RED, or a
  # silent PASS depending on which side the shell reads. The detector is textual,
  # so even naming the pattern in a comment counts as a site: this comment
  # deliberately does not spell it. A case glob against the newline-delimited blob needs no
  # pipe and no subshell, and stays line-anchored via the leading newline.
  row_present() { # row_present <row>  -> 0 if the scan reported that fixture
    case $'\n'"$out" in
      *$'\n'"$1/Cargo.toml|"*) return 0 ;;
      *) return 1 ;;
    esac
  }

  for row in hit_use hit_qualified hit_underscore hit_extern; do
    row_present "$row" || { printf 'FAIL: missed %s\n' "$row"; rc=1; }
  done
  for row in ok_versioned ok_workspace ok_git ok_registry ok_unused ok_comment ok_substring; do
    if row_present "$row"; then printf 'FAIL: false positive on %s\n' "$row"; rc=1; fi
  done

  # THE DEATH ROWS (#3644): every way the scanner can fail to run must come
  # back as ENV rc=2, never as a row verdict. A must-match fixture is scanned
  # each time, so a wrong classification would have to invent "missed hit_use"
  # (rc 1) or "no violations" (rc 0) out of an interpreter that never ran.
  death() { # death <label> <want-rc> <env-assignments...>
    local label=$1 want=$2 got=0; shift 2
    ( export "$@"; scan "$tmp" > /dev/null 2>&1 ) || got=$?
    if [ "$got" = "$want" ]; then printf 'ok    death  rc=%s  %s\n' "$got" "$label"
    else printf 'FAIL  death  rc=%s (wanted %s)  %s\n' "$got" "$want" "$label"; rc=1; fi
  }
  death 'both TOML readers absent (the intel shape) -> ENV rc=2'   2 PATHONLY_GUARD_FORCE_NO_TOML=1
  death 'interpreter exits 1 with no output -> ENV rc=2, never 1' 2 PATHONLY_GUARD_PYTHON=/bin/false
  death 'no interpreter at all (rc=127) -> ENV rc=2'              2 PATHONLY_GUARD_PYTHON="$tmp/no-such-python"
  death 'the working interpreter, as the control -> rc=0'         0 PATHONLY_GUARD_PYTHON="${PATHONLY_GUARD_PYTHON:-python3}"
  # The rows above call scan() directly. This one runs THE WHOLE GUARD under the
  # intel shape, because the call site is where the old `|| true` swallowed the
  # death: the exit must be 2 and no row verdict may be printed. (Guarded
  # against recursion; the nested run has no death rows of its own.)
  if [ -z "${PATHONLY_GUARD_NESTED:-}" ]; then
    local nested_out nested_rc=0
    nested_out=$(PATHONLY_GUARD_NESTED=1 PATHONLY_GUARD_FORCE_NO_TOML=1 bash "$0" --selftest 2>&1) || nested_rc=$?
    case "$nested_rc:$nested_out" in
      2:*"missed hit_"*|2:*"false positive"*)
        printf 'FAIL  death  the whole guard under the intel shape printed a row verdict beside its ENV\n'; rc=1 ;;
      2:*) printf 'ok    death  rc=2  the whole guard under the intel shape exits ENV with no row verdict\n' ;;
      *)   printf 'FAIL  death  rc=%s (wanted 2)  the whole guard under the intel shape\n' "$nested_rc"; rc=1 ;;
    esac
  fi

  cleanup
  FIXTURES=""
  [ "$rc" -eq 0 ] && printf 'PASS: case table (4 must-match, 7 must-not-match, 5 death rows)\n'
  return "$rc"
}

main() {
  if [ "${1:-}" = "--selftest" ]; then selftest; return $?; fi
  local st_rc=0
  selftest || st_rc=$?
  [ "$st_rc" -eq 0 ] || return "$st_rc"

  local out pairs rc=0 new stale scan_rc=0
  out="$(scan "$REPO")" || scan_rc=$?
  if [ "$scan_rc" -ne 0 ]; then
    printf 'ENV: the tree was not scanned (rc=%s); the baseline is neither new nor stale, it is UNMEASURED\n' "$scan_rc"
    return 2
  fi
  pairs="$(printf '%s\n' "$out" | awk -F'|' 'NF>=2 {print $1"|"$2}' | sort -u)"

  # anything not in the baseline is a NEW violation
  new="$(comm -23 <(printf '%s\n' "$pairs" | grep -v '^$' || true) \
                  <(grep -vE '^\s*(#|$)' "$BASELINE" | sort -u))"
  # a baseline row with no violation left must be deleted in the same commit
  stale="$(comm -13 <(printf '%s\n' "$pairs" | grep -v '^$' || true) \
                    <(grep -vE '^\s*(#|$)' "$BASELINE" | sort -u))"

  if [ -n "$new" ]; then
    rc=1
    printf 'FAIL: src/ references a dev-dep that publishing deletes (#3305)\n\n'
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
