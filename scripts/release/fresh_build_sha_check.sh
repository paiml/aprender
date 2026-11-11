#!/usr/bin/env bash
# fresh_build_sha_check.sh -- promote preflight (aprender#6679, operator ruling "R7 GREEN accepted on receipt provenance; add
# fresh-build sha check as promote preflight"). The R7 GPU receipts name the sha256 of the `apr` they measured; this builds `apr`
# FRESH from the candidate sha (clean worktree, own target dir) and requires every receipt to name that same binary.
#
#   fresh_build_sha_check.sh --sha SHA --repo DIR --receipts DIR [--builder CMD] [--work DIR]
#
# receipts = DIR/*-gpu.meta.json, each carrying .binary_sha256 (or .apr.binary_sha256). CMD is run as `CMD WORKTREE TARGETDIR`
# and prints the built binary path (default: cargo build --release --bin apr). rc 0 = every receipt matches the fresh build;
# rc 1 = RED: a mismatch, a receipt without a sha, no receipts, or a build that produced nothing. Never rc 0 on not_measured.
# Builds are compared per host and arch: a receipt measured on another arch than this build host is RED with that reason named,
# not skipped (reproducibility across hosts is not assumed).
set -uo pipefail
sha="" repo="" rdir="" builder="" work=""
while [ $# -gt 0 ]; do
    case $1 in
        --sha) sha=${2:-}; shift 2 ;; --repo) repo=${2:-}; shift 2 ;; --receipts) rdir=${2:-}; shift 2 ;;
        --builder) builder=${2:-}; shift 2 ;; --work) work=${2:-}; shift 2 ;;
        *) echo "fresh_build_sha_check: unknown arg $1" >&2; exit 2 ;;
    esac
done
[[ $sha =~ ^[0-9a-f]{40}$ ]] && [ -d "$repo" ] && [ -d "$rdir" ] || { echo "fresh_build_sha_check: RED usage/unreadable input" >&2; exit 1; }
own=0; [ -n "$work" ] || { work=$(mktemp -d) || exit 1; own=1; }
cleanup() { git -C "$repo" worktree remove --force "$work/wt" > /dev/null 2>&1; [ "$own" = 0 ] || rm -rf -- "${work:?}"; }
trap cleanup EXIT
shopt -s nullglob; recs=("$rdir"/*-gpu.meta.json); shopt -u nullglob
[ "${#recs[@]}" -gt 0 ] || { echo "RED fresh-build: no R7 *-gpu.meta.json receipts in $rdir (not a pass)"; exit 1; }
git -C "$repo" worktree add -q --detach "$work/wt" "$sha" > /dev/null 2>&1 || { echo "RED fresh-build: cannot make a clean worktree of ${sha:0:9}"; exit 1; }
if [ -n "$builder" ]; then bin=$(bash -c "$builder \"\$1\" \"\$2\"" _ "$work/wt" "$work/target" 2> "$work/build.log")
else bin=$(cd "$work/wt" && CARGO_TARGET_DIR="$work/target" cargo build -q --release --bin apr > "$work/build.log" 2>&1 && printf %s "$work/target/release/apr"); fi
[ -n "$bin" ] && [ -f "$bin" ] || { echo "RED fresh-build: no apr binary built from ${sha:0:9} ($(tail -n 1 "$work/build.log" 2> /dev/null | cut -c1-100))"; exit 1; }
want=$(sha256sum -- "$bin" | cut -d" " -f1)
rc=0
for f in "${recs[@]}"; do
    got=$(jq -r ".binary_sha256 // .apr.binary_sha256 // empty" "$f" 2> /dev/null)
    if [ -z "$got" ]; then echo "RED fresh-build: $(basename "$f") names no binary_sha256 (missing is RED, never not_measured)"; rc=1
    elif [ "$got" != "$want" ]; then echo "RED fresh-build: $(basename "$f") binary ${got:0:12} != fresh build ${want:0:12} of ${sha:0:9} on $(uname -m)"; rc=1
    else echo "ok fresh-build: $(basename "$f") matches ${want:0:12}"; fi
done
exit $rc
