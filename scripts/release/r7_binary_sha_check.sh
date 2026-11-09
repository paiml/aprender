#!/usr/bin/env bash
# r7_binary_sha_check.sh -- promote preflight: every R7 receipt's binary sha256 must equal a fresh build of the candidate (aprender#6679)
#
#   bash scripts/release/r7_binary_sha_check.sh --receipts DIR (--fresh-sha HEX64 | --build WORKTREE [--target-dir DIR])
#
# DIR holds the R7 receipts (<host>-gpu.meta.json, found at any depth). Each must carry .apr_binary_sha256 (64 hex).
# --fresh-sha gives the sha256 of a fresh `apr`; --build compiles `apr` from WORKTREE into its OWN target dir
# (default: a fresh mktemp dir, removed on exit) and hashes it.
# Exit 0 = every receipt names the fresh binary. Exit 1 = RED: a mismatch, a receipt with no/invalid sha, or no receipt
# at all (all RED, never not_measured). Exit 2 = caller error. Nothing is waived and no receipt is written or edited.
# PER HOST AND ARCH: builds are not reproducible across hosts (lambda x86_64 vs gx10 arm64 differ). Each host builds the
# candidate itself and judges that host's receipts: run this once per host with --receipts pointing at that host's receipts.
set -uo pipefail
receipts="" fresh="" wt="" td="" scratch=""
die() { printf 'r7_binary_sha_check: %s\n' "$*" >&2; exit 2; }
while [ $# -gt 0 ]; do
    case "$1" in
        --receipts) receipts=${2:-}; shift 2 ;;
        --fresh-sha) fresh=${2:-}; shift 2 ;;
        --build) wt=${2:-}; shift 2 ;;
        --target-dir) td=${2:-}; shift 2 ;;
        *) die "unknown argument '$1'" ;;
    esac
done
[ -n "$receipts" ] && [ -d "$receipts" ] || die "--receipts DIR is required and must be a directory"
[ -n "$fresh" ] || [ -n "$wt" ] || die "one of --fresh-sha or --build is required"
[ -z "$fresh" ] || [ -z "$wt" ] || die "--fresh-sha and --build are exclusive"
if [ -n "$wt" ]; then
    [ -d "$wt" ] || die "--build WORKTREE '$wt' is not a directory"
    scratch=$(mktemp -d) || die "mktemp failed"
    trap 'rm -rf -- "${scratch:?}"' EXIT
    [ -n "$td" ] || td="$scratch/target"
    if ! (cd "$wt" && cargo build --release --locked --bin apr --target-dir "$td") > "$scratch/build.log" 2>&1; then
        tail -n 5 "$scratch/build.log" >&2
        printf 'RED   R7 binary sha: fresh build of apr failed in %s\n' "$wt"; exit 1
    fi
    fresh=$(sha256sum -- "$td/release/apr" | cut -d' ' -f1) || die "cannot hash the fresh build"
fi
[[ $fresh =~ ^[0-9a-f]{64}$ ]] || die "fresh sha '$fresh' is not 64 lowercase hex"
bad=0 n=0
while IFS= read -r f; do
    n=$((n + 1))
    got=$(jq -r '.apr_binary_sha256 // empty' -- "$f" 2> /dev/null)
    if ! [[ $got =~ ^[0-9a-f]{64}$ ]]; then
        printf 'RED   R7 binary sha: %s has no valid apr_binary_sha256\n' "$f"; bad=$((bad + 1))
    elif [ "$got" != "$fresh" ]; then
        printf 'RED   R7 binary sha: %s names %s, a fresh build is %s\n' "$f" "${got:0:12}" "${fresh:0:12}"; bad=$((bad + 1))
    fi
done < <(find "$receipts" -name '*-gpu.meta.json' -type f | sort)
if [ "$n" -eq 0 ]; then printf 'RED   R7 binary sha: no *-gpu.meta.json receipt under %s\n' "$receipts"; exit 1; fi
[ "$bad" -eq 0 ] || exit 1
printf 'ok    R7 binary sha: %d receipt(s) name the fresh build %s\n' "$n" "${fresh:0:12}"
