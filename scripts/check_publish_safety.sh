#!/usr/bin/env bash
# check_publish_safety.sh — Prevent P0 publish regressions from shipping to crates.io
#
# Bug classes caught:
#   1. Tracked symlinks (mode 120000) — broke cargo install for all external users
#   2. with_file_name() anti-pattern — misses hash-prefixed companion files
#   3. .pmat dev tool artifacts — shipped DB files and caches to crates.io
#   4. Large binary files in packages — bloated crate downloads
#   5. Hardcoded local paths — break on other machines
#
# Refs: PMAT-SQI (symlinks), GAP-UX-002 (companion lookup), CB-510 (gitignore)
# Contract: contracts/publish-safety-v1.yaml
#
# Usage: bash scripts/check_publish_safety.sh [--self-test]
# Exit 0 if all OK, exit 1 if any checks fail, exit 2 if it cannot measure (the
# package listings Checks 2/8/9 read could not be taken).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# pkg_list <pkg>: the file list `cargo package` would publish, or rc 1 with the
# reason. Checks 2, 8 and 9 used to run `cargo package -p … --list 2>/dev/null`
# inline and read an empty result as clean. Check 2 even omitted --allow-dirty,
# so on any dirty tree cargo refused, the listing was empty, and the
# .cargo/config leak check printed OK without looking (fail-closed sweep).
# PUBLISH_SAFETY_CARGO stands in for cargo in the case table only.
pkg_list() {
    local pkg=$1 out err rc=0
    err=$(mktemp) || return 1
    out=$("${PUBLISH_SAFETY_CARGO:-cargo}" package -p "$pkg" --list --allow-dirty 2>"$err" </dev/null) || rc=$?
    if [ "$rc" -ne 0 ]; then
        printf 'UNMEASURED: cargo package -p %s --list exited %s:\n' "$pkg" "$rc" >&2
        tail -n 3 "$err" | sed 's/^/    /' >&2
        rm -f "${err:?}"
        return 1
    fi
    rm -f "${err:?}"
    if [ -z "$out" ]; then
        printf 'UNMEASURED: cargo package -p %s --list listed no files\n' "$pkg" >&2
        return 1
    fi
    printf '%s\n' "$out"
}

if [ "${1:-}" = "--self-test" ]; then
    T=$(mktemp -d "${TMPDIR:-/tmp}/pubsafe-selftest.XXXXXX") || exit 2
    trap 'rm -rf "${T:?}"' EXIT
    n=0; red=0
    row() { # row <want rc> <label> <cmd...>
        local want=$1 label=$2 rc=0; shift 2
        n=$((n + 1))
        "$@" >"$T/out.$n" 2>&1 || rc=$?
        if [ "$rc" = "$want" ]; then printf 'ok    row %s rc=%s  %s\n' "$n" "$rc" "$label"
        else printf 'FAIL  row %s rc=%s (wanted %s)  %s\n' "$n" "$rc" "$want" "$label"; sed 's/^/        /' "$T/out.$n"; red=1; fi
    }
    fake() { # fake <name> <rc> <stdout>: a stand-in cargo
        printf '#!/usr/bin/env bash\nprintf "%%b" %q\nexit %s\n' "$3" "$2" > "$T/$1"
        chmod +x "$T/$1"
    }
    fake dirty 101 ''
    fake empty 0 ''
    fake ok 0 'Cargo.toml\nsrc/lib.rs\n'
    row 1 "pkg_list: cargo refuses (dirty tree, offline: rc 101) is not a listing" \
        env PUBLISH_SAFETY_CARGO="$T/dirty" bash -c ". \"\$1\" --lib-only && pkg_list aprender" _ "$0"
    row 1 "pkg_list: an empty listing is not a clean one" \
        env PUBLISH_SAFETY_CARGO="$T/empty" bash -c ". \"\$1\" --lib-only && pkg_list aprender" _ "$0"
    row 0 "pkg_list: a real listing passes through" \
        env PUBLISH_SAFETY_CARGO="$T/ok" bash -c ". \"\$1\" --lib-only && pkg_list aprender" _ "$0"
    row 2 "whole gate: cargo cannot list the package -> exit 2, never 'OK'" \
        env PUBLISH_SAFETY_CARGO="$T/dirty" bash "$0"
    row 2 "whole gate: an empty listing -> exit 2, never 'OK (0 files)'" \
        env PUBLISH_SAFETY_CARGO="$T/empty" bash "$0"
    # Row 4's rc alone would also be 2 for an unrelated early exit: pin the reason.
    if ! grep -q 'UNMEASURED: cargo package -p aprender' "$T/out.4"; then
        printf 'FAIL  row 4 exited 2 without naming the unmeasured package\n'; red=1
    fi
    printf '%s/%s rows\n' "$((n - red))" "$n"
    [ "$red" = 0 ] || exit 1
    exit 0
fi
[ "${1:-}" = "--lib-only" ] && return 0

cd "$REPO_ROOT" || { echo "UNMEASURED: cannot cd to $REPO_ROOT"; exit 2; }

errors=0
checked=0

echo "Publish safety gate..."

# The two listings Checks 2, 8 and 9 read, taken once. Without them those checks
# cannot answer, so the gate stops here rather than print three OKs.
LIST_aprender=$(pkg_list aprender) || { echo "UNMEASURED: no package listing for aprender; publish safety cannot be judged"; exit 2; }
LIST_apr_cli=$(pkg_list apr-cli) || { echo "UNMEASURED: no package listing for apr-cli; publish safety cannot be judged"; exit 2; }
listing_of() {
    case "$1" in
        aprender) printf '%s\n' "$LIST_aprender" ;;
        apr-cli) printf '%s\n' "$LIST_apr_cli" ;;
    esac
}

# Check 1: No tracked symlinks (P0: symlinks to build dirs broke all users)
echo -n "  Symlink check... "
symlinks=$(git ls-files -s | grep "^120000" || true)
checked=$((checked + 1))
if [ -n "$symlinks" ]; then
    echo "FAIL"
    echo "FAIL: Tracked symlinks found in git:"
    echo "$symlinks"
    echo "Fix: git rm --cached <symlink-path>"
    errors=$((errors + 1))
else
    echo "OK"
fi

# Check 2: No .cargo/config.toml in any workspace package
echo -n "  Cargo config leak check... "
checked=$((checked + 1))
config_leak=0
for pkg in aprender apr-cli; do
    if listing_of "$pkg" | grep -q '\.cargo/config'; then
        if [ "$config_leak" -eq 0 ]; then
            echo "FAIL"
        fi
        echo "FAIL: .cargo/config.toml found in $pkg package"
        echo "Fix: add .cargo/config.toml to Cargo.toml exclude and run git rm --cached .cargo/config.toml"
        config_leak=1
    fi
done
if [ "$config_leak" -gt 0 ]; then
    errors=$((errors + 1))
else
    echo "OK"
fi

# Check 3: No with_file_name("tokenizer.json"|"config.json") anti-pattern in apr-cli
echo -n "  Companion lookup check... "
checked=$((checked + 1))
bad_lookups=$(grep -rn 'with_file_name\s*(.*"tokenizer\.json"' crates/apr-cli/src/ 2>/dev/null || true)
bad_lookups2=$(grep -rn 'with_file_name\s*(.*"config\.json"' crates/apr-cli/src/ 2>/dev/null || true)
bad_all="${bad_lookups}${bad_lookups2}"
if [ -n "$bad_all" ]; then
    echo "FAIL"
    echo "FAIL: with_file_name() used for companion file lookup (use find_sibling_file instead):"
    echo "$bad_all"
    echo "Fix: replace path.with_file_name(\"tokenizer.json\") with find_sibling_file(path, \"tokenizer.json\")"
    errors=$((errors + 1))
else
    echo "OK"
fi

# Check 4: find_sibling_file is actually used (sanity check — regression if all removed)
echo -n "  find_sibling_file usage check... "
checked=$((checked + 1))
sibling_count=$(grep -rn 'find_sibling_file' crates/apr-cli/src/ 2>/dev/null | wc -l)
if [ "$sibling_count" -lt 1 ]; then
    echo "FAIL"
    echo "FAIL: No find_sibling_file() calls found in apr-cli (expected at least 1)"
    echo "This suggests companion file lookup was removed or broken"
    errors=$((errors + 1))
else
    echo "OK ($sibling_count usages)"
fi

# Check 5: No .pmat dev tool artifacts tracked by git (CB-510 class)
echo -n "  Dev tool artifacts check... "
checked=$((checked + 1))
pmat_tracked=$(git ls-files | grep '\.pmat/' || true)
if [ -n "$pmat_tracked" ]; then
    echo "FAIL"
    count=$(echo "$pmat_tracked" | wc -l)
    echo "FAIL: $count .pmat/ files tracked by git (these ship to crates.io):"
    echo "$pmat_tracked" | head -10
    echo "Fix: git rm --cached <paths> and ensure **/.pmat/ is in .gitignore"
    errors=$((errors + 1))
else
    echo "OK"
fi

# Check 6: No large or binary files in publishable packages
#
# This check had two holes, and the defect it is credited with catching (a 28 MB
# test.apr) would slip through both today:
#
#   1. It iterated `for pkg in aprender apr-cli` -- 2 of the 72 publishable
#      crates. The 0.63.0 CB-510 recurrence was in aprender-serve, which is not
#      one of them. A 2 MB blob planted in aprender-core passed cleanly.
#   2. Its name and comment say ">1MB" and it never measured a size. It grepped
#      an extension list, so a 100 MB .txt passed and a 1 KB .bin failed. The
#      threshold in the title did not exist in the code.
#
# It then acquired a THIRD hole, of exactly the same species as the first
# (aprender#2559): "every publishable crate" meant one `cargo metadata` on the
# ROOT workspace, and `crates/facades/` is a second workspace `exclude`d from
# the root. Its three crates are published to crates.io and were never size- or
# hygiene-scanned -- and could not be, because `cargo package -p
# provable-contracts` from the repo root is rc=101, "did not match any
# packages". The `|| continue` below turned that into a silent skip.
#
# Now: scripts/lib/cascade_universe.py -- every publishable crate in every
# workspace, each with the manifest path needed to reach it -- an actual byte
# threshold, and a count that must be accounted for rather than skipped.
echo -n "  Package size check... "
checked=$((checked + 1))
large_files=""
skipped_pkgs=""
scanned_pkgs=0
MAX_PACKAGED_BYTES=1048576

universe=$(python3 "${REPO_ROOT:-.}/scripts/lib/cascade_universe.py" "${REPO_ROOT:-.}")

while IFS=$'\t' read -r pkg _ver manifest _ws; do
    [ -n "$pkg" ] || continue
    # --manifest-path, not -p: `-p` cannot name a crate outside this workspace.
    listing=$(cargo package --manifest-path "$manifest" --list --allow-dirty 2>/dev/null < /dev/null) \
        || { skipped_pkgs="${skipped_pkgs} ${pkg}"; continue; }
    if [ -z "$listing" ]; then
        skipped_pkgs="${skipped_pkgs} ${pkg}"
        continue
    fi
    pkg_dir=$(dirname "$manifest")
    scanned_pkgs=$((scanned_pkgs + 1))

    while IFS= read -r f; do
        [ -n "$f" ] || continue
        # SIZE, not extension. The old rule failed on an extension list and
        # never measured anything, which is both too strict and too loose: it
        # would reject a 77-byte golden .apr fixture while passing a 3.6 MB
        # .pmat-baseline.json. Every real instance of this defect -- the 28 MB
        # test.apr, and the 5.4 MB of baseline JSON found when this check was
        # first pointed at all 72 crates -- is a SIZE problem.
        if [ -f "$pkg_dir/$f" ]; then
            sz=$(stat -c%s "$pkg_dir/$f" 2>/dev/null || echo 0)
            if [ "$sz" -gt "$MAX_PACKAGED_BYTES" ]; then
                large_files="${large_files}${pkg}: ${f} (${sz} bytes > ${MAX_PACKAGED_BYTES})\n"
            fi
        fi
    done <<< "$listing"
done <<< "$universe"

if [ -n "$large_files" ]; then
    echo "FAIL"
    echo "FAIL: large or binary files found in published packages:"
    echo -e "$large_files"
    echo "Fix: add to Cargo.toml [package] exclude (root-anchored) or .gitignore"
    errors=$((errors + 1))
elif [ "$scanned_pkgs" -lt 70 ]; then
    # Vacuity, in the direction this check has now failed twice: a scan that
    # covered almost nothing reports no large files and reads as a clean pass.
    # An unscannable crate is a RESULT, not a skip.
    echo "FAIL"
    echo "FAIL: only $scanned_pkgs package(s) were actually scanned, expected 70+."
    echo "Unscannable:${skipped_pkgs:- (none named)}"
    echo "The ENUMERATION is broken, not the packages."
    errors=$((errors + 1))
else
    echo "OK ($scanned_pkgs packages)"
fi

# Check 7: No hardcoded /home/ paths in non-test production source
echo -n "  Hardcoded paths check... "
checked=$((checked + 1))
# Only check non-test source files that ship to users
# Exclude: test files, cfg(test) blocks, #[ignore] tests, examples
bad_paths=$(grep -rn '/home/' crates/apr-cli/src/ src/lib.rs src/traits.rs src/primitives/ src/format/ src/text/ \
    --include='*.rs' 2>/dev/null \
    | grep -v '_test' | grep -v 'mod tests' | grep -v '#\[test' | grep -v '#\[ignore' \
    | grep -v 'falsification' | grep -v '// ' \
    || true)
if [ -n "$bad_paths" ]; then
    echo "WARN"
    echo "WARN: Hardcoded /home/ paths found in source (may break other users):"
    echo "$bad_paths" | head -5
else
    echo "OK"
fi

# Check 8: No CI/CD or dev infrastructure in publishable packages
echo -n "  Package hygiene check... "
checked=$((checked + 1))
hygiene_fail=0
for pkg in aprender apr-cli; do
    leaked=$(listing_of "$pkg" \
        | grep -E '\.github/|\.githooks/|\.pmat-metrics|Dockerfile' || true)
    if [ -n "$leaked" ]; then
        if [ "$hygiene_fail" -eq 0 ]; then
            echo "FAIL"
        fi
        echo "FAIL: Dev infrastructure files found in $pkg package:"
        echo "$leaked" | head -5
        hygiene_fail=1
    fi
done
if [ "$hygiene_fail" -gt 0 ]; then
    errors=$((errors + 1))
else
    echo "OK"
fi

# Check 9: Package file count sanity (catch accidental bloat regression)
echo -n "  Package file count check... "
checked=$((checked + 1))
aprender_count=$(listing_of aprender | wc -l)
# Threshold: current is ~1520. Alert if it grows by >200 files (new bloat).
if [ "$aprender_count" -gt 1800 ]; then
    echo "WARN"
    echo "WARN: aprender package has $aprender_count files (threshold: 1800)"
    echo "Check for new directories that should be excluded in Cargo.toml"
else
    echo "OK ($aprender_count files)"
fi

# Summary
echo ""
if [ "$errors" -gt 0 ]; then
    echo "FAIL: $errors publish safety checks failed out of $checked"
    echo "See: contracts/publish-safety-v1.yaml"
    exit 1
else
    echo "OK: All $checked publish safety checks passed"
fi
