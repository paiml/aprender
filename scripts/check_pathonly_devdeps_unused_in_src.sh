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
# A RUNNER WITHOUT THE TOOL IS FLEET STATE, NOT RED (#3692). #3644 put this guard in
# guard-tree's population, and on the intel hosts the rc 2 above redded `ci / gate` for
# every PR -- against guard_tree.sh's rule that "a runner without the tool must not red
# every PR". So exactly two cases are UNMEASURED, exit 0, surfaced under the PASS row
# (#3651): no interpreter at all, and an interpreter with no TOML reader (the scanner says
# so and exits 3). Everything else that stops the scan -- a crash, a non-zero exit with no
# trailer, an exit 3 WITHOUT the no-reader line -- stays ENV rc 2, RED. Only the missing
# tool is fleet state, the line #3671 drew for pv.
#
# Runs bare, no arguments, from anywhere in the repo. `--selftest` runs the case
# table only. A bare run does BOTH: the case table first, then the tree.
#   PATHONLY_GUARD_PYTHON=<interpreter>   test seam: a missing one is fleet state, a dead one is ENV
#   PATHONLY_GUARD_FORCE_NO_TOML=1        test seam: both readers fail to import, for real (the intel shape)
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
# rc 0 with the trailer consumed; rc 3 (fleet state: the runner lacks the tool, an
# UNMEASURED line on stderr); rc 2 (ENV, RED) when the scanner did not finish.
scan() {
    local out rc=0 interp="${PATHONLY_GUARD_PYTHON:-python3}"
    if ! command -v "$interp" > /dev/null 2>&1; then
        printf 'UNMEASURED runner=%s reason=no-interpreter interpreter=%s -- this runner has no %s, so the manifests were not read; fleet state, not a pass (#3692)\n' \
            "${RUNNER_NAME:-unknown}" "$interp" "$interp" >&2
        return 3
    fi
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

class FallbackUnread(ValueError):
    pass

_HDR = re.compile(r"^\[\s*([^\[\]]+?)\s*\]\s*(#.*)?$")
_KEY = re.compile(r"""^(?:"([A-Za-z0-9_-]+)"|([A-Za-z0-9_-]+))(?:\.([A-Za-z0-9_-]+))?\s*=\s*(.*)$""")
_INLINE_KEY = re.compile(r"(?:^\{|,)\s*([A-Za-z0-9_-]+)\s*(?:\.[A-Za-z0-9_-]+\s*)*=")

def _strip_comment(val):
    out, q = [], None
    for ch in val:
        if q:
            if ch == q:
                q = None
        elif ch in "\"'":
            q = ch
        elif ch == "#":
            break
        out.append(ch)
    if q:
        raise FallbackUnread("unterminated string")
    return "".join(out).strip()

def devdeps_fallback(text):
    """The top-level [dev-dependencies] of a Cargo.toml, for a python with no TOML reader
    (#3863: python 3.10 on intel-clean-room-*). It returns only what sourceless() reads --
    each dep's KEY NAMES (path/version/git/workspace), never a value it would have to
    guess -- and raises FallbackUnread on any shape it does not parse, so the manifest is
    counted unread, never read as "no deps"."""
    deps, table, sub, depth = {}, None, None, 0
    for raw in text.splitlines():
        line = raw.strip()
        if depth:
            # continuation of a multi-line array: its lines are values, never keys
            depth += _strip_comment(line).count("[") - _strip_comment(line).count("]")
            continue
        if not line or line.startswith("#"):
            continue
        if line.startswith("[["):
            table = None
            continue
        if line.startswith("["):
            m = _HDR.match(line)
            if not m:
                raise FallbackUnread("header %r" % line)
            name = re.sub(r"\s+", "", m.group(1))
            if name == "dev-dependencies":
                table, sub = "dd", None
            elif name.startswith("dev-dependencies."):
                table, sub = "sub", name.split(".", 1)[1].strip("\"'")
                if "." in sub:
                    raise FallbackUnread("header %r" % line)
                deps.setdefault(sub, {})
            else:
                table = None
            continue
        if table is None:
            continue
        m = _KEY.match(line)
        if not m:
            raise FallbackUnread("line %r" % line)
        key, dotted, val = m.group(1) or m.group(2), m.group(3), _strip_comment(m.group(4))
        if '"""' in val or "\'\'\'" in val:
            raise FallbackUnread("multi-line string %r" % line)
        if table == "sub":
            if dotted:
                raise FallbackUnread("line %r" % line)
            deps[sub][key] = True
            depth = max(0, val.count("[") - val.count("]"))
        elif dotted:
            spec = deps.setdefault(key, {})
            if not isinstance(spec, dict):
                raise FallbackUnread("line %r" % line)
            spec[dotted] = True
        elif val[:1] in "\"'":
            deps[key] = val
        elif val.startswith("{"):
            if not val.endswith("}"):
                raise FallbackUnread("multi-line inline table %r" % line)
            deps[key] = {k: True for k in _INLINE_KEY.findall(val)}
        else:
            raise FallbackUnread("line %r" % line)
    return {"dev-dependencies": deps} if deps else {}

class _Fallback:
    __name__ = "fallback"
    TOMLDecodeError = FallbackUnread

    @staticmethod
    def load(fh):
        return devdeps_fallback(fh.read().decode("utf-8"))

if toml is None:
    toml = _Fallback()   # #3863: MEASURE on python 3.10, do not stand down to UNMEASURED

def sourceless(spec):
    """True when the dep carries a path and nothing that survives publish."""
    if not isinstance(spec, dict):
        return False
    if "path" not in spec:
        return False
    return not any(k in spec for k in ("version", "git", "workspace"))

root = sys.argv[1]
KINDS = ("dev-dependencies",)

if os.environ.get("PATHONLY_GUARD_EQUIV") == "1":
    # #3863: the fallback must read every manifest of the tree exactly as the real reader
    # does -- each top-level dev-dep's key names. Only a python WITH a real reader can say.
    if toml.__name__ == "fallback":
        print("EQUIV-UNMEASURED no real TOML reader to compare against")
        print("SCAN-DONE manifests=0 reader=fallback")
        sys.exit(0)
    checked = sourceless_n = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (".git", "target", "node_modules")]
        if "Cargo.toml" not in filenames:
            continue
        manifest = os.path.join(dirpath, "Cargo.toml")
        try:
            with open(manifest, "rb") as fh:
                raw = fh.read()
            want = toml.loads(raw.decode("utf-8")).get("dev-dependencies") or {}
        except (toml.TOMLDecodeError, OSError, UnicodeDecodeError):
            continue
        checked += 1
        try:
            got = devdeps_fallback(raw.decode("utf-8")).get("dev-dependencies") or {}
        except FallbackUnread as exc:
            print("EQUIV-MISMATCH %s unread (%s)" % (os.path.relpath(manifest, root), exc))
            continue
        norm = lambda m: {k: (sorted(v) if isinstance(v, dict) else "string") for k, v in m.items()}
        if norm(want) != norm(got):
            print("EQUIV-MISMATCH %s tomllib=%s fallback=%s" % (os.path.relpath(manifest, root), norm(want), norm(got)))
        sourceless_n += sum(1 for v in want.values() if sourceless(v))
    print("EQUIV-DONE manifests=%d sourceless=%d" % (checked, sourceless_n))
    print("SCAN-DONE manifests=%d reader=%s" % (checked, toml.__name__))
    sys.exit(0)

manifests = 0
unread = []
rows = []   # printed only after the unread check, so an UNMEASURED scan emits no row verdict


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
    except FallbackUnread as exc:
        unread.append("%s (%s)" % (os.path.relpath(manifest, root), exc))
        continue
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
                        rows.append(f"{rel_m}|{alias}|{rel_s}:{n}:{stripped[:100]}")
if unread:
    # the fallback met a shape it does not parse: those manifests were NOT read, so the
    # scan is not a verdict. Fleet state (the runner lacks a full reader), never a pass.
    print("ENV: no TOML reader on this interpreter (python %d.%d, need tomllib >= 3.11 or tomli) runner=%s -- "
          "the fallback reader could not read %d manifest(s): %s; this is not 'no violations'"
          % (sys.version_info[0], sys.version_info[1], runner, len(unread), "; ".join(unread[:3])))
    sys.exit(3)
for row in rows:
    print(row)
# positive evidence that the scan RAN TO THE END; the shell refuses any output without it
print("SCAN-DONE manifests=%d reader=%s" % (manifests, toml.__name__))
PY
    ) || rc=$?
    # fleet state needs BOTH the scanner's own no-reader line AND its exit 3; either alone is ENV
    case "$rc:$out" in
        3:"ENV: no TOML reader on this interpreter"*)
            printf 'UNMEASURED runner=%s reason=no-toml-reader -- %s; fleet state, not a pass (#3692)\n' \
                "${RUNNER_NAME:-unknown}" "${out#ENV: }" >&2
            return 3 ;;
    esac
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

  # THE SHAPES THE FALLBACK READER MUST GET RIGHT (#3863) -- scanned under both readers below
  mk hit_subtable  $'[dev-dependencies.sib]\npath = "../sib"'        'use sib::thing;'
  mk hit_trailing  'sib = { path = "../sib" } # version = "1"'       'use sib::thing;'
  mk hit_dotted   'sib.path = "../sib"'                              'use sib::thing;'
  mk hit_quoted    '"sib" = { path = "../sib", features = ["a", "b"] }' 'use sib::thing;'
  mk ok_sub_versioned $'[dev-dependencies.sib]\npath = "../sib"\nversion = "0.1"' 'use sib::thing;'
  mk ok_dotted_ws  'sib.workspace = true'                            'use sib::thing;'
  mk ok_array_then $'sib = { path = "../sib" }\n\n[features]\nx = [\n  "y",\n]' 'pub fn f() {}'

  # ENV from the scanner is ENV here: rc 2, never "missed hit_use" (#3644).
  # Fleet state (rc 3) passes through: scan already printed the UNMEASURED line (#3692).
  local env_rc=0 out_real out_fb
  out_real="$(scan "$tmp")" || env_rc=$?
  if [ "$env_rc" -eq 3 ]; then
    cleanup; FIXTURES=""
    return 3
  fi
  if [ "$env_rc" -ne 0 ]; then
    printf 'ENV: the case table could not be measured (scan rc=%s)\n' "$env_rc"
    cleanup; FIXTURES=""
    return 2
  fi
  # The same table under the fallback reader, the one python 3.10 runs (#3863). It MUST
  # measure: an UNMEASURED here is the intel shape standing down again, so it is RED.
  env_rc=0
  out_fb="$(PATHONLY_GUARD_FORCE_NO_TOML=1 scan "$tmp" 2>&1)" || env_rc=$?
  if [ "$env_rc" -ne 0 ]; then
    printf 'FAIL: the fallback reader did not measure the case table (scan rc=%s): %s\n' "$env_rc" "$out_fb"
    cleanup; FIXTURES=""
    return 1
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

  for reader in real fallback; do
    if [ "$reader" = real ]; then out="$out_real"; else out="$out_fb"; fi
    for row in hit_use hit_qualified hit_underscore hit_extern hit_subtable hit_trailing hit_dotted hit_quoted; do
      row_present "$row" || { printf 'FAIL: missed %s (reader=%s)\n' "$row" "$reader"; rc=1; }
    done
    for row in ok_versioned ok_workspace ok_git ok_registry ok_unused ok_comment ok_substring \
               ok_sub_versioned ok_dotted_ws ok_array_then; do
      if row_present "$row"; then printf 'FAIL: false positive on %s (reader=%s)\n' "$row" "$reader"; rc=1; fi
    done
  done
  out="$out_real"

  # A shape the fallback does not parse is NOT "no deps": the whole scan is UNMEASURED
  # (rc 3, the no-reader line), and it prints no row -- not even the must-match beside it.
  local unread_root="$tmp/unread-root" ur_rc=0 ur_out
  mkdir -p "$unread_root/multi/src" "$unread_root/hit/src"
  printf '[package]\nname = "multi"\n\n[dev-dependencies]\nsib = {\n  path = "../sib" }\n' > "$unread_root/multi/Cargo.toml"
  printf 'use sib::x;\n' > "$unread_root/multi/src/lib.rs"
  printf '[package]\nname = "hit"\n\n[dev-dependencies]\nsib = { path = "../sib" }\n' > "$unread_root/hit/Cargo.toml"
  printf 'use sib::x;\n' > "$unread_root/hit/src/lib.rs"
  ur_out="$(PATHONLY_GUARD_FORCE_NO_TOML=1 scan "$unread_root" 2>&1)" || ur_rc=$?
  case "$ur_rc:$ur_out" in
    *"Cargo.toml|sib|"*) printf 'FAIL  unread  the fallback printed a row beside an unread manifest\n'; rc=1 ;;
    3:*"UNMEASURED"*"reason=no-toml-reader"*"multi/Cargo.toml"*)
      printf 'ok    unread rc=3  a manifest the fallback cannot parse makes the scan UNMEASURED, never a verdict\n' ;;
    *) printf 'FAIL  unread rc=%s (wanted 3 + UNMEASURED naming multi/Cargo.toml): %s\n' "$ur_rc" "$ur_out"; rc=1 ;;
  esac
  rm -rf "${unread_root:?}"

  # EQUIVALENCE ON THE REAL TREE (#3863): wherever a real reader exists, the fallback must
  # read every manifest's dev-dep key names exactly as it does. Vacuity guard: the tree
  # must yield manifests AND sourceless deps, else the comparison proved nothing.
  if [ -z "${PATHONLY_GUARD_NESTED:-}" ] && [ -z "${PATHONLY_GUARD_FORCE_NO_TOML:-}" ]; then
    local eq_out eq_rc=0
    eq_out="$(PATHONLY_GUARD_EQUIV=1 scan "$REPO" 2>&1)" || eq_rc=$?
    case "$eq_rc:$eq_out" in
      0:*"EQUIV-MISMATCH"*)
        printf 'FAIL  equiv  the fallback reads the tree differently from the real reader:\n'
        printf '%s\n' "$eq_out" | grep '^EQUIV-MISMATCH' | head -5 | sed 's/^/      /'; rc=1 ;;
      0:*"EQUIV-UNMEASURED"*) printf '~     equiv  no real TOML reader here; the equivalence row is measured on the other runners\n' ;;
      0:*"EQUIV-DONE manifests=0 "*|0:*"EQUIV-DONE manifests="*" sourceless=0"*)
        printf 'FAIL  equiv  vacuous: %s\n' "$eq_out"; rc=1 ;;
      0:*"EQUIV-DONE manifests="*)
        printf 'ok    equiv  %s -- fallback == real reader on every manifest\n' "$(printf '%s\n' "$eq_out" | grep '^EQUIV-DONE')" ;;
      *) printf 'FAIL  equiv  rc=%s: %s\n' "$eq_rc" "$eq_out"; rc=1 ;;
    esac
  fi

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
  # An interpreter that exits 3 WITHOUT the no-reader line is not fleet state (#3692):
  # the exit code alone must not buy an UNMEASURED.
  printf '#!/bin/sh\necho "Traceback (most recent call last): boom" >&2\nexit 3\n' > "$tmp/exit3-python"
  chmod 755 "$tmp/exit3-python"
  death 'both TOML readers absent (the intel shape) -> the fallback MEASURES, rc=0 (#3863)'  0 PATHONLY_GUARD_FORCE_NO_TOML=1
  death 'no interpreter at all -> UNMEASURED rc=3, fleet state (#3692)'                     3 PATHONLY_GUARD_PYTHON="$tmp/no-such-python"
  death 'interpreter exits 1 with no output -> ENV rc=2 (RED), never fleet state'          2 PATHONLY_GUARD_PYTHON=/bin/false
  death 'interpreter exits 3 without the no-reader line -> ENV rc=2 (RED)'                 2 PATHONLY_GUARD_PYTHON="$tmp/exit3-python"
  death 'the working interpreter, as the control -> rc=0'                                  0 PATHONLY_GUARD_PYTHON="${PATHONLY_GUARD_PYTHON:-python3}"
  # The rows above call scan() directly. These run THE WHOLE GUARD, because the call
  # site is where the old `|| true` swallowed the death. Under the intel shape the guard
  # MEASURES with the fallback reader: exit 0, its OK line, no UNMEASURED (#3863 -- it
  # was UNMEASURED reason=no-toml-reader on intel-clean-room-4, #3692). Under a dead
  # interpreter: exit 2, still RED. (Guarded against recursion; the nested run has no
  # death rows of its own.)
  if [ -z "${PATHONLY_GUARD_NESTED:-}" ]; then
    local nested_out nested_rc=0
    nested_out=$(PATHONLY_GUARD_NESTED=1 PATHONLY_GUARD_FORCE_NO_TOML=1 RUNNER_NAME=intel-probe bash "$0" 2>&1) || nested_rc=$?
    case "$nested_rc:$nested_out" in
      "UNMEASURED runner="*|*$'\n'"UNMEASURED runner="*|*:"UNMEASURED runner="*)
        printf 'FAIL  death  the whole guard under the intel shape stood down to UNMEASURED (#3863): %s\n' "$nested_out"; rc=1 ;;
      0:*"OK: no NEW"*)
        printf 'ok    death  rc=0  the whole guard under the intel shape MEASURES the tree with the fallback reader (#3863)\n' ;;
      *)   printf 'FAIL  death  rc=%s (wanted 0 + OK)  the whole guard under the intel shape: %s\n' "$nested_rc" "$nested_out"; rc=1 ;;
    esac
    nested_rc=0
    PATHONLY_GUARD_NESTED=1 PATHONLY_GUARD_PYTHON=/bin/false bash "$0" > /dev/null 2>&1 || nested_rc=$?
    if [ "$nested_rc" -eq 2 ]; then printf 'ok    death  rc=2  the whole guard under a dead interpreter is ENV, RED\n'
    else printf 'FAIL  death  rc=%s (wanted 2)  the whole guard under a dead interpreter\n' "$nested_rc"; rc=1; fi
  fi

  cleanup
  FIXTURES=""
  [ "$rc" -eq 0 ] && printf 'PASS: case table (8 must-match, 10 must-not-match, x2 readers; 1 unread row; 7 death rows; tree equivalence)\n'
  return "$rc"
}

main() {
  local st_rc=0
  selftest || st_rc=$?
  # fleet state (#3692): the UNMEASURED line is already printed; exit 0, and no tree verdict
  [ "$st_rc" -eq 3 ] && return 0
  [ "${1:-}" = "--selftest" ] && return "$st_rc"
  [ "$st_rc" -eq 0 ] || return "$st_rc"

  local out pairs rc=0 new stale scan_rc=0
  out="$(scan "$REPO")" || scan_rc=$?
  [ "$scan_rc" -eq 3 ] && return 0
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
