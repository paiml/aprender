#!/usr/bin/env bash
# check_include_files.sh — Verify all include!() referenced files are tracked by git
#
# Prevents the CB-510 bug where .gitignore hid src/models/ part files from git,
# causing crates.io publishes to silently exclude required source files.
#
# Usage: ./scripts/check_include_files.sh [--self-test]
# Exit 0 if all OK, exit 1 if any include!() files are untracked/missing,
# exit 2 if it could not measure (the scan failed or found nothing to check).
#
# The scan is anchored at the repo root (INCLUDE_ROOT overrides it, for the case
# table). It used to scan `src/ crates/` relative to the CALLER's cwd: run from
# anywhere else, grep failed on both paths, 0 files were checked, and it printed
# "OK: All 0 include!() files are tracked by git" with exit 0 (fail-closed sweep).

set -uo pipefail
# guard_tree.sh probes `--help` to decide whether to run a self-test. Answer it before any work:
# a probe that fell through to the body ran this whole guard a second time, serially (#4046).
case "${1:-}" in -h|--help) printf 'usage: bash scripts/check_include_files.sh (no arguments: scans every include!() target)\n'; exit 0 ;; esac

if [ "${1:-}" = "--self-test" ]; then
    T=$(mktemp -d "${TMPDIR:-/tmp}/incl-selftest.XXXXXX") || exit 2
    trap 'rm -rf "${T:?}"' EXIT
    n=0; red=0
    row() { # row <want rc> <label> <root>
        local want=$1 label=$2 root=$3 rc=0
        n=$((n + 1))
        INCLUDE_ROOT="$root" bash "$0" >"$T/out.$n" 2>&1 || rc=$?
        if [ "$rc" = "$want" ]; then printf 'ok    row %s rc=%s  %s\n' "$n" "$rc" "$label"
        else printf 'FAIL  row %s rc=%s (wanted %s)  %s\n' "$n" "$rc" "$want" "$label"; sed 's/^/        /' "$T/out.$n"; red=1; fi
    }
    mk() { # mk <dir>: a git repo with src/ and crates/c/src/, one tracked include!
        mkdir -p "$1/src" "$1/crates/c/src"
        printf 'include!("part.rs");\n' > "$1/crates/c/src/lib.rs"
        printf 'fn f() {}\n' > "$1/crates/c/src/part.rs"
        printf 'fn main() {}\n' > "$1/src/lib.rs"
        ( cd "$1" && git init -q . && git add -A && git -c core.hooksPath=/dev/null -c user.email=t@t -c user.name=t commit -qm x )
    }
    mk "$T/good"; row 0 "a tracked include! target" "$T/good"
    mk "$T/untracked"; printf 'include!("new.rs");\n' >> "$T/untracked/crates/c/src/lib.rs"; printf 'fn g() {}\n' > "$T/untracked/crates/c/src/new.rs"
    row 1 "an untracked include! target (CB-510)" "$T/untracked"
    mk "$T/missing"; printf 'include!("gone.rs");\n' >> "$T/missing/crates/c/src/lib.rs"
    row 1 "an include! target that does not exist" "$T/missing"
    mkdir -p "$T/empty"; row 2 "no src/ or crates/ to scan: exit 2, never 'OK: All 0'" "$T/empty"
    mk "$T/none"; printf 'fn h() {}\n' > "$T/none/crates/c/src/lib.rs"
    row 2 "a tree with zero include! directives: 0 checked is exit 2" "$T/none"
    mk "$T/nosrc"; rm -rf "${T:?}/nosrc/src"
    row 2 "one scan root missing (src/): the grep error is exit 2" "$T/nosrc"
    printf '%s/%s rows\n' "$((n - red))" "$n"
    [ "$red" = 0 ] || exit 1
    exit 0
fi

ROOT="${INCLUDE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT" || { echo "UNMEASURED: cannot cd to $ROOT"; exit 2; }

# Scan into a variable so grep's status is ours to read: 2 is an error (a scan
# root is missing), 1 is no include!() anywhere. Neither is a pass.
listing=$(grep -rn 'include!(' src/ crates/ --include='*.rs')
grc=$?
if [ "$grc" -ne 0 ]; then
    echo "UNMEASURED: grep over src/ crates/ in $ROOT exited $grc (2 = a scan root is missing, 1 = no include!() at all)"
    exit 2
fi

errors=0
checked=0

# Find all include!() directives in Rust source files
while IFS=: read -r file line content; do
    # Extract the included filename from include!("filename.rs")
    # POSIX ERE, not `grep -P`: -P is missing on BSD/busybox grep, and its error was
    # swallowed here, so every line was skipped and the scan went vacuous.
    included=$(printf '%s\n' "$content" | sed -nE 's/.*include!\([[:space:]]*"([^"]+)"[[:space:]]*\).*/\1/p')
    [ -z "$included" ] && continue

    # Resolve relative to the directory containing the source file
    dir=$(dirname "$file")
    resolved="$dir/$included"

    checked=$((checked + 1))

    # Check file exists
    if [ ! -f "$resolved" ]; then
        echo "MISSING: $resolved (referenced by $file:$line)"
        errors=$((errors + 1))
        continue
    fi

    # Check file is tracked by git (not gitignored)
    if ! git ls-files --error-unmatch "$resolved" >/dev/null 2>&1; then
        # Double-check: is it ignored?
        if git check-ignore -q "$resolved" 2>/dev/null; then
            echo "GITIGNORED: $resolved (referenced by $file:$line)"
        else
            echo "UNTRACKED: $resolved (referenced by $file:$line)"
        fi
        errors=$((errors + 1))
    fi
done < <(printf '%s\n' "$listing" | grep -v '/target/' | grep -v '#\[')

if [ "$checked" -eq 0 ]; then
    echo "UNMEASURED: 0 include!() targets resolved from $(printf '%s\n' "$listing" | wc -l) matching line(s); refusing 'OK: All 0'"
    exit 2
fi

if [ "$errors" -gt 0 ]; then
    echo ""
    echo "FAIL: $errors include!() files are missing, gitignored, or untracked out of $checked checked"
    echo "Fix: git add the files and check .gitignore / Cargo.toml exclude patterns"
    exit 1
else
    echo "OK: All $checked include!() files are tracked by git"
    exit 0
fi
