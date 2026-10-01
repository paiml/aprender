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
# Usage: bash scripts/check_publish_safety.sh
# Exit 0 if all OK, exit 1 if any checks fail.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

errors=0
checked=0

echo "Publish safety gate..."

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
    if grep -q '\.cargo/config' <<< "$(cargo package -p "$pkg" --list 2>/dev/null)" ; then
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
    leaked=$(cargo package -p "$pkg" --list --allow-dirty 2>/dev/null \
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
aprender_count=$(cargo package -p aprender --list --allow-dirty 2>/dev/null | wc -l)
# Threshold: current is ~1520. Alert if it grows by >200 files (new bloat).
if [ "$aprender_count" -gt 1800 ]; then
    echo "WARN"
    echo "WARN: aprender package has $aprender_count files (threshold: 1800)"
    echo "Check for new directories that should be excluded in Cargo.toml"
else
    echo "OK ($aprender_count files)"
fi

# Check 10: every apr-cli feature that GATES A SUBCOMMAND must have a root-facade
# passthrough. The published crate is `aprender`, not `apr-cli`, so a feature with
# no passthrough cannot be enabled by any `cargo install aprender` invocation —
# `--features <name>` errors out and `--features full` silently omits it. Found by
# `cargo install aprender --features setfit` failing while every CI SetFit test
# passed, because CI tests `-p <crate> --features setfit` and never the facade.
# The decision surface is the facade's [features] table, so that is what is scanned.
echo -n "  Facade feature passthrough check... "
checked=$((checked + 1))
# The scanned set is apr-cli's [features] TABLE — the closed set cargo itself
# reads — not `#[cfg(feature = ...)]` occurrences in dispatch*.rs. Scanning source
# was wrong twice over: it missed `hf-hub` and `safetensors-compare` (they gate
# `apr publish` / `apr compare-hf` from commands/*.rs, never dispatch*.rs), and it
# caught `setfit` — the defect that motivated this check — only by luck, because
# its cfgs happened to also appear there. A source regex also cannot see
# `all(...)`/`any(...)` wrappers, and `[a-z-]+` silently skips any feature name
# containing a digit or underscore. The table has none of those failure modes.
#
# A feature is reachable by a `cargo install aprender` user if it has a facade
# passthrough OR is in apr-cli's `default` (which the facade inherits).
# Exempt, with reasons:
#   default   — the meta-feature itself, not a capability.
#   full      — the aggregate; it IS a passthrough.
#   dev       — `apr mono` internal maintenance subcommands.
#   dhat-heap — heap-profiling build, not a shipped capability.
#   code      — retained for backwards-compat; the subcommand is no longer gated.
passthrough_exempt="default full dev dhat-heap code"
missing_passthrough=""
apr_cli_features=$(awk '/^\[features\]/{f=1;next} /^\[/{f=0} f && /^[a-zA-Z0-9_-]+ *=/{sub(/ *=.*/,""); print}' \
    crates/apr-cli/Cargo.toml 2>/dev/null | sort -u)
# The WHOLE `default = [...]` array, which may wrap across lines. `grep -m1`
# read only the first line, so the day someone reformats that array every
# feature in it turns into a bogus "missing passthrough" FAIL. awk collects
# from `default = [` to the closing `]` instead.
apr_cli_default=$(awk '/^default *= *\[/{c=1} c{printf "%s", $0; if (/\]/) exit}' \
    crates/apr-cli/Cargo.toml 2>/dev/null)
# Here-string, not a pipe: a piped `while read` runs in a subshell and
# `missing_passthrough` would not survive the loop, silently passing the gate.
while IFS= read -r feat; do
    [ -n "$feat" ] || continue
    case " $passthrough_exempt " in
        *" $feat "*) continue ;;
        *) ;;
    esac
    # Reachable because the facade inherits apr-cli's default feature set.
    case "$apr_cli_default" in
        *"\"$feat\""*) continue ;;
        *) ;;
    esac
    # The passthrough must both exist AND forward to apr-cli/<feat>; a bare
    # `feat = []` would satisfy `cargo --features feat` while enabling nothing.
    if ! grep -qE "^${feat} = \[.*\"apr-cli/${feat}\".*\]" Cargo.toml; then
        missing_passthrough="$missing_passthrough $feat"
    fi
done <<< "$apr_cli_features"
if [ -n "$missing_passthrough" ]; then
    echo "FAIL"
    echo "FAIL: apr-cli features gate subcommands but have no root-facade passthrough:$missing_passthrough"
    echo "  Effect: 'cargo install aprender --features <name>' errors; '--features full' silently omits it."
    echo "  Fix: add the feature to the root Cargo.toml features table, forwarding to cli and apr-cli/<name>."
    errors=$((errors + 1))
else
    echo "OK"
fi

# Check 11: mirrored contracts must match the repo-root catalog byte-for-byte.
#
# A handful of contracts are tracked TWICE: once under the crate that consumes them
# at build time, once in the repo-root contracts/ catalog that `pv` and humans read.
# The duplication is forced — a published crate cannot reach ../../contracts/ (cargo
# package only includes files under the crate dir), and a tracked symlink is rejected
# by check 1 (PMAT-SQI). So both copies stay and this check keeps them equal.
#
# The failure it prevents is silent: the two files are indistinguishable by name and
# only the crate-local one has any effect, so editing the repo-root copy changes
# nothing while every gate stays green against the stale file.
#
# Scoped by BASENAME COLLISION, not by an allowlist: a crate-local contract is a
# mirror exactly when a file of the same basename exists in the root catalog.
# Crate-local contracts with no root counterpart (three unrelated matmul-v1.yaml
# among them) are genuinely distinct documents and are ignored.
echo -n "  Mirrored contract sync check... "
checked=$((checked + 1))
# DERIVED, not hand-listed: a crate-local contract is a mirror exactly when a
# file of the same basename exists in the root catalog. An allowlist would fail
# the same silent way the thing it guards fails — add a 12th mirror, forget the
# script, nothing goes red. Deriving cannot go stale.
#
# Measured over the tree when this was written: 11 mirrors, 0 drift, 49
# crate-local contracts with no root counterpart (correctly ignored). The
# staging crate is the vendored provable-contracts UPSTREAM corpus — a different
# project whose ~43 files merely share basenames — so it is skipped wholesale.
#
# MIRROR_FLOOR is the vacuity guard. Everything below is relative to the CWD,
# and `find` on a missing tree prints nothing and exits quietly, so without a
# floor a wrong CWD or a renamed directory reports `OK (0 mirrors)` — the exact
# shape of CR-02 (a zero-match filter exiting 0 while printing "ok"). The floor
# is a LOWER bound, not the exact count, so adding a 12th mirror does not turn
# this red for the wrong reason.
MIRROR_FLOOR=11
mirrored_contracts=$(find crates -maxdepth 3 -path '*/contracts/*.yaml' \
    -not -path '*/aprender-contracts-staging/*' -not -path '*/target/*' 2>/dev/null | sort)
mirror_drift=""
# Here-string, not a pipe: a piped `while read` runs in a subshell and
# `mirror_drift` would not survive the loop, silently passing the gate.
mirror_count=0
while IFS= read -r mirror; do
    [ -n "$mirror" ] || continue
    # Parameter expansion, not $(basename): one fork per file for nothing.
    root_copy="contracts/${mirror##*/}"
    # No root counterpart => a crate-local-only contract, not a mirror.
    [ -f "$root_copy" ] || continue
    mirror_count=$((mirror_count + 1))
    # Accumulate with a '|' separator rather than embedded newlines, then expand
    # at print time: a multi-line string assignment trips the shell linter here.
    if ! cmp -s "$mirror" "$root_copy"; then
        mirror_drift="${mirror_drift}|    DRIFTED: $mirror != $root_copy"
    fi
done <<< "$mirrored_contracts"
if [ -n "$mirror_drift" ]; then
    echo "FAIL"
    echo "FAIL: mirrored contracts out of sync with the repo-root catalog:"
    printf '%s\n' "$mirror_drift" | tr '|' '\n' | grep -v '^$'
    echo "  Only the crate-local copy is read at build time, so a repo-root-only edit has NO effect."
    echo "  Fix: copy whichever side you edited over the other, then rebuild so codegen re-runs."
    errors=$((errors + 1))
elif [ "$mirror_count" -lt "$MIRROR_FLOOR" ]; then
    echo "FAIL"
    echo "FAIL: only $mirror_count mirrored contracts found, expected at least $MIRROR_FLOOR."
    echo "  A comparison that compared nothing passes for free. Either this ran from the wrong"
    echo "  directory (every path here is CWD-relative), or mirrors were genuinely removed."
    echo "  If they were removed on purpose, lower MIRROR_FLOOR in this script deliberately."
    errors=$((errors + 1))
else
    echo "OK ($mirror_count mirrors)"
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
