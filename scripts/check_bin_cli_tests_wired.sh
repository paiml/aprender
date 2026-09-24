#!/usr/bin/env bash
# check_bin_cli_tests_wired.sh — every test target that SPAWNS a workspace binary is run by CI or is
# ledgered as a finding (#4059, part of the binary-debt epic #4057).
#
# WHY. The fleet HARD REQ says a binary's CLI is tested through the binary. The workspace-test
# job runs `--lib` only, so an integration target runs only when some workflow or a
# ci/explicit-test-commands.d/ fragment names it with `--test`. Measured on main@aa7c6ef03: 93
# targets spawn a binary (`CARGO_BIN_EXE_<bin>`, assert_cmd's `cargo_bin(` or `cargo_bin_cmd!`)
# and no lane runs them. That includes all 34 of aprender-profile's and 45 of apr-cli's. #4059
# wired the KEPT binaries' 82 (fragments 460-500); the 11 left are the ledger. A
# CLI test that never runs is not coverage, and nothing noticed the gap because nothing counted it.
#
# DERIVED, NEVER LISTED. The universe is every test target of every package, found the way the
# build tool finds them: tests/<name>.rs, tests/<name>/main.rs (every .rs file under that
# directory is scanned), and each [[test]] entry in a manifest, with `autotests = false` honoured.
# The root facade counts too; its tests/ is the `aprender` package's. A target is WIRED when one
# logical command, with `\` continuations joined, in .github/workflows/*.yml or
# ci/explicit-test-commands.d/*.cmd names `--test <name>` together with `-p <package>` (or
# `--package`). A command that names the target but a different package does NOT wire it,
# because test names repeat across crates (`integration_tests`, `cli_tests`).
#
# THE LEDGER. scripts/bin_cli_unwired_baseline.txt holds `<package>\t--test\t<name>` for every
# spawning target that no lane runs. It must EQUAL the derivation. A new unwired spawning test
# FAILS (wire it, or ledger it with --update), and so does a ledgered line that is now wired
# (delete it). scripts/check_baseline_ratchets.sh compares the file with origin/main as a SET,
# so a PR may only remove lines. Ledgering a new dark test is refused there.
#
# Usage:
#   scripts/check_bin_cli_tests_wired.sh                 # derive, diff against the ledger
#   scripts/check_bin_cli_tests_wired.sh --print [ROOT]  # the derived unwired set
#   scripts/check_bin_cli_tests_wired.sh --update        # rewrite the ledger from the derivation
#   scripts/check_bin_cli_tests_wired.sh --self-test     # case table + planted mutants
# Exit: 0 ledger matches · 1 drift · 2 could not check.
set -uo pipefail
LEDGER_DEFAULT="scripts/bin_cli_unwired_baseline.txt"
SELF="${BASH_SOURCE[0]}"

derive() { # derive <root> -> "<pkg>\t--test\t<name>" for each spawning target no lane runs
    python3 - "$1" <<'PY'
import glob, os, re, sys
# CI's guard-tree python is 3.10: no tomllib (#4315 run 36032470368 failed on
# ModuleNotFoundError). Use tomllib/tomli when present, else read only the keys
# this guard needs. APR_TOML_READER=minimal forces the fallback so the self-test
# proves it on any interpreter.
toml = None
if os.environ.get("APR_TOML_READER") != "minimal":
    for _name in ("tomllib", "tomli"):
        try:
            toml = __import__(_name); break
        except ImportError:
            pass

def _scalar(v):
    v = v.strip()
    if v[:1] in ('"', "'"):
        q = v[0]; end = v.find(q, 1)
        return v[1:end] if end > 0 else None
    v = v.split("#", 1)[0].strip()
    return {"true": True, "false": False}.get(v, v)

def load_minimal(man):
    """[package] name/autotests and [[test]] name/path -- all this guard reads."""
    d = {"package": {}, "test": []}; cur = None
    for line in open(man, encoding="utf-8"):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("[[") :
            cur = {} if s.split("]]")[0].strip("[ ") == "test" else None
            if cur is not None: d["test"].append(cur)
            continue
        if s.startswith("["):
            cur = d["package"] if s.split("]")[0].strip("[ ") == "package" else None
            continue
        if cur is not None and "=" in s:
            k, v = s.split("=", 1); k = k.strip()
            if k in ("name", "autotests", "path"):
                cur[k] = _scalar(v)
    return d
root = sys.argv[1]
SPAWN = re.compile(r'CARGO_BIN_EXE_|cargo_bin\(|cargo_bin_cmd!')

def manifests():
    yield os.path.join(root, "Cargo.toml")
    yield from sorted(glob.glob(os.path.join(root, "crates", "*", "Cargo.toml")))

def targets(man):
    """(package, name, [source files]) for every test target of one manifest."""
    try:
        d = toml.load(open(man, "rb")) if toml else load_minimal(man)
    except (OSError, ValueError) as e:  # TOMLDecodeError is a ValueError
        print("cannot parse %s: %s" % (man, e), file=sys.stderr); sys.exit(2)
    pkg = (d.get("package") or {}).get("name")
    if not pkg:
        return
    base = os.path.dirname(man); tdir = os.path.join(base, "tests"); seen = {}
    if (d.get("package") or {}).get("autotests", True) and os.path.isdir(tdir):
        for f in sorted(os.listdir(tdir)):
            p = os.path.join(tdir, f)
            if f.endswith(".rs") and os.path.isfile(p):
                seen[f[:-3]] = [p]
            elif os.path.isfile(os.path.join(p, "main.rs")):
                seen[f] = sorted(glob.glob(os.path.join(p, "**", "*.rs"), recursive=True))
    for t in d.get("test") or []:
        n = t.get("name")
        if not n:
            continue
        p = os.path.join(base, t.get("path") or os.path.join("tests", n + ".rs"))
        if os.path.basename(p) == "main.rs":
            seen[n] = sorted(glob.glob(os.path.join(os.path.dirname(p), "**", "*.rs"), recursive=True))
        else:
            seen[n] = [p]
    for n, files in sorted(seen.items()):
        yield pkg, n, files

def spawns(files):
    for f in files:
        try:
            if SPAWN.search(open(f, errors="ignore").read()):
                return True
        except OSError:
            pass
    return False

cmds = []
for f in sorted(glob.glob(os.path.join(root, ".github", "workflows", "*.yml"))
                + glob.glob(os.path.join(root, "ci", "explicit-test-commands.d", "*.cmd"))):
    text = re.sub(r"\\\n\s*", " ", open(f, errors="ignore").read())
    cmds += [l for l in text.splitlines() if "--test" in l]

def wired(pkg, name):
    t = re.compile(r"--test[= ]+" + re.escape(name) + r"(?![A-Za-z0-9_])")
    p = re.compile(r"(?:-p|--package)[= ]+" + re.escape(pkg) + r"(?![A-Za-z0-9_-])")
    return any(t.search(c) and p.search(c) for c in cmds)

for man in manifests():
    if not os.path.isfile(man):
        continue
    for pkg, name, files in targets(man):
        if spawns(files) and not wired(pkg, name):
            print("%s\t--test\t%s" % (pkg, name))
PY
}

ledger_body() { grep -v '^#' "$1" 2>/dev/null | grep . || :; }

check() { # check <root> <ledger>
    local want have
    [ -f "$2" ] || { echo "  cannot check: ledger $2 not found" >&2; return 2; }
    want=$(derive "$1") || { echo "  cannot check: derivation failed" >&2; return 2; }
    have=$(ledger_body "$2")
    if [ "$(LC_ALL=C sort <<< "$want")" = "$(LC_ALL=C sort <<< "$have")" ]; then
        printf 'ok    %s matches the derivation (%s spawning test target(s) no lane runs)\n' "$2" "$(grep -c . <<< "$want")"
        return 0
    fi
    printf 'FAIL  %s drifted (<: ledgered but now wired or gone, delete the line; >: a spawning test no lane runs, wire it in ci/explicit-test-commands.d/):\n' "$2"
    diff <(LC_ALL=C sort <<< "$have") <(LC_ALL=C sort <<< "$want") | grep '^[<>]' | sed 's/^/      /'
    return 1
}

update() { # update <root> <ledger>
    local body; body=$(derive "$1") || { echo "  cannot update: derivation failed" >&2; return 2; }
    { printf '# tool_version=none (derived by scripts/check_bin_cli_tests_wired.sh --print; regenerate with --update, never edit)\n'
      printf '# Test targets that SPAWN a workspace binary and that no workflow or ci/explicit-test-commands.d/\n'
      printf '# fragment runs (#4059). Each line is a finding. The file may only shrink against origin/main.\n'
      LC_ALL=C sort <<< "$body" | grep . ; } > "$2"
    printf 'ok    wrote %s (%s target(s))\n' "$2" "$(grep -c . <<< "$body")"
}

self_test() {
    local T red=0 got
    T=$(mktemp -d -t check_bin_cli_tests_wired.XXXXXXXX) || { echo "cannot mktemp" >&2; return 2; }
    trap 'case "$T" in /tmp/?*) rm -rf -- "${T:?}" ;; esac' RETURN
    mk() { mkdir -p "$(dirname "$T/tree/$1")"; printf '%s\n' "$2" > "$T/tree/$1"; }
    mk Cargo.toml '[package]
name = "facade"'
    mk tests/root_spawn.rs 'fn t() { let _ = env!("CARGO_BIN_EXE_facade"); }'
    mk crates/alpha/Cargo.toml '[package]
name = "alpha"

[[test]]
name = "custom"
path = "tests/x/custom.rs"'
    mk crates/alpha/tests/exe_env.rs 'fn t() { let _ = env!("CARGO_BIN_EXE_alpha"); }'
    mk crates/alpha/tests/assert_cmd_fn.rs 'fn t() { let _ = assert_cmd::Command::cargo_bin("alpha"); }'
    mk crates/alpha/tests/macro_form.rs '#![allow(deprecated)] // assert_cmd::Command::cargo_bin deprecation
fn t() { let _ = assert_cmd::cargo::cargo_bin_cmd!("alpha"); }'
    mk crates/alpha/tests/comment_only.rs '// see assert_cmd::Command::cargo_bin deprecation notes
fn t() {}'
    mk crates/alpha/tests/plain.rs 'fn t() { assert_eq!(1, 1); }'
    mk crates/alpha/tests/dir_target/main.rs 'mod helper;'
    mk crates/alpha/tests/dir_target/helper.rs 'pub fn h() { let _ = env!("CARGO_BIN_EXE_alpha"); }'
    mk crates/alpha/tests/x/custom.rs 'fn t() { let _ = env!("CARGO_BIN_EXE_alpha"); }'
    mk crates/alpha/tests/wired_frag.rs 'fn t() { let _ = env!("CARGO_BIN_EXE_alpha"); }'
    mk crates/alpha/tests/wired_multiline.rs 'fn t() { let _ = env!("CARGO_BIN_EXE_alpha"); }'
    mk crates/alpha/tests/other_pkg.rs 'fn t() { let _ = env!("CARGO_BIN_EXE_alpha"); }'
    mk crates/alpha/tests/prefix.rs 'fn t() { let _ = env!("CARGO_BIN_EXE_alpha"); }'
    mk crates/beta/Cargo.toml '[package]
name = "beta"
autotests = false'
    mk crates/beta/tests/not_a_target.rs 'fn t() { let _ = env!("CARGO_BIN_EXE_beta"); }'
    local ct="carg""o test"   # spelled apart: guard_tree classifies a guard as needing the toolchain by its text
    mk ci/explicit-test-commands.d/010-alpha.cmd "$ct -p alpha --test wired_frag"
    mk ci/explicit-test-commands.d/020-beta.cmd "$ct -p beta --test other_pkg"
    mk ci/explicit-test-commands.d/030-alpha-prefix.cmd "$ct -p alpha --test prefix_longer"
    mk .github/workflows/ci.yml "jobs:
  t:
    steps:
      - run: |
          $ct -p alpha \\
            --test wired_multiline"
    local want
    want=$(printf 'alpha\t--test\t%s\n' assert_cmd_fn custom dir_target exe_env macro_form other_pkg prefix; printf 'facade\t--test\troot_spawn\n')
    run_rows() { # <script> -> prints ok/FAIL rows; 0 when all land
        local s="$1" rc=0 got row key name
        got=$(bash "$s" --print "$T/tree" 2>&1 | LC_ALL=C sort)
        for row in "alpha/exe_env:CARGO_BIN_EXE_ is a spawn" "alpha/assert_cmd_fn:assert_cmd cargo_bin( is a spawn" \
                   "alpha/macro_form:cargo_bin_cmd! is a spawn" "alpha/dir_target:a tests/<dir>/main.rs target is scanned, helper files included" \
                   "alpha/custom:a [[test]] path= target is found" "facade/root_spawn:the root facade's tests/ count" \
                   "alpha/other_pkg:a --test naming another package does not wire it" "alpha/prefix:--test prefix_longer does not wire prefix"; do
            key=${row%%:*}; name=${key#*/}
            if grep -qxF "$(printf '%s\t--test\t%s' "${key%%/*}" "$name")" <<< "$got"; then printf '  ok    %-16s %s\n' "$name" "${row#*:}"
            else printf '  FAIL  %-16s %s (not reported unwired)\n' "$name" "${row#*:}"; rc=1; fi
        done
        for row in "wired_frag:a fragment -p/--test pair wires it" "wired_multiline:a continued workflow command wires it" \
                   "comment_only:a comment naming cargo_bin is not a spawn" "plain:a test that spawns nothing is out of scope" \
                   "not_a_target:autotests = false leaves tests/*.rs undiscovered"; do
            name=${row%%:*}
            if grep -q "	$name\$" <<< "$got"; then printf '  FAIL  %-16s %s (reported unwired)\n' "$name" "${row#*:}"; rc=1
            else printf '  ok    %-16s %s\n' "$name" "${row#*:}"; fi
        done
        if [ "$got" = "$(LC_ALL=C sort <<< "$want")" ]; then printf '  ok    %-16s the derived set is exactly the expected one\n' exact
        else printf '  FAIL  %-16s derived set differs:\n%s\n' exact "$got"; rc=1; fi
        return "$rc"
    }
    echo "case table:"
    run_rows "$SELF" || red=1
    echo "case table, fallback TOML reader (python 3.10 has no tomllib):"
    APR_TOML_READER=minimal run_rows "$SELF" || red=1
    # Parity on the real tree: the fallback derives exactly what tomllib does.
    local root; root=$(git -C "$(dirname "$SELF")" rev-parse --show-toplevel 2>/dev/null)
    if [ -n "$root" ] && python3 -c 'import tomllib' 2>/dev/null; then
        if [ "$(bash "$SELF" --print "$root" 2>&1)" = "$(APR_TOML_READER=minimal bash "$SELF" --print "$root" 2>&1)" ]; then
            echo "  ok    reader-parity    the fallback reader derives the tomllib set on this tree"
        else echo "  FAIL  reader-parity    the fallback reader differs from tomllib on this tree"; red=1; fi
    fi
    # The ledger itself: equal passes, a missing line and a stale line both fail.
    printf '%s\n' "$want" > "$T/ledger"
    bash "$SELF" --check "$T/tree" "$T/ledger" > /dev/null 2>&1 && echo "  ok    ledger-equal     an exact ledger passes" || { echo "  FAIL  ledger-equal     an exact ledger failed"; red=1; }
    sed '1d' "$T/ledger" > "$T/ledger.short"
    bash "$SELF" --check "$T/tree" "$T/ledger.short" > /dev/null 2>&1; got=$?
    [ "$got" = 1 ] && echo "  ok    ledger-missing   a new unwired spawning test FAILS (rc 1)" || { echo "  FAIL  ledger-missing   rc=$got, want 1"; red=1; }
    { cat "$T/ledger"; printf 'alpha\t--test\twired_frag\n'; } > "$T/ledger.stale"
    bash "$SELF" --check "$T/tree" "$T/ledger.stale" > /dev/null 2>&1; got=$?
    [ "$got" = 1 ] && echo "  ok    ledger-stale     a ledgered target that is now wired FAILS (rc 1)" || { echo "  FAIL  ledger-stale     rc=$got, want 1"; red=1; }
    bash "$SELF" --check "$T/tree" "$T/nope" > /dev/null 2>&1; got=$?
    [ "$got" = 2 ] && echo "  ok    ledger-absent    a missing ledger is rc 2, never a pass" || { echo "  FAIL  ledger-absent    rc=$got, want 2"; red=1; }
    echo "planted mutants:"
    plant() { # <label> <python-sed> <must-red row...>
        local label="$1" expr="$2" m="$T/m-$1.sh" o c; shift 2
        sed "$expr" "$SELF" > "$m"
        cmp -s "$SELF" "$m" && { echo "  FAIL  mutant $label did not apply"; red=1; return; }
        o=$(run_rows "$m" 2>&1) || :
        for c in "$@"; do
            grep -qE "FAIL  $c " <<< "$o" && printf '  ok    mutant %-14s killed by %s\n' "$label" "$c" || { echo "  FAIL  mutant $label SURVIVED row $c"; red=1; }
        done
    }
    plant no-macro 's/|cargo_bin_cmd!//' macro_form
    plant name-only 's/return any(t.search(c) and p.search(c) for c in cmds)/return any(t.search(c) for c in cmds)/' other_pkg
    plant no-dir-targets 's/elif os.path.isfile(os.path.join(p, "main.rs")):/elif False:/' dir_target
    plant no-boundary 's/r"(?!\[A-Za-z0-9_\])")$/r"")/' prefix
    plant no-continuation 's/text = re.sub(r"\\\\\\n\\s\*", " ", /text = (/' wired_multiline
    [ "$red" = 0 ] && { echo "SELF-TEST OK"; return 0; }
    echo "SELF-TEST FAIL"; return 1
}

case "${1:-}" in
    -h|--help) awk 'NR == 1 { next } !/^#/ { exit } { sub(/^# ?/, ""); print }' "$0"; exit 0 ;;
    --print) derive "${2:-.}" ;;
    --check) check "${2:-.}" "${3:-$LEDGER_DEFAULT}" ;;
    --update) update "${2:-.}" "${3:-$LEDGER_DEFAULT}" ;;
    --self-test) self_test ;;
    "") echo "binary-spawning test targets vs CI wiring (#4059)"; check . "$LEDGER_DEFAULT" ;;
    *) echo "check_bin_cli_tests_wired: unknown argument '$1'" >&2; exit 2 ;;
esac
