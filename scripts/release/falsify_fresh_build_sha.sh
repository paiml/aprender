#!/usr/bin/env bash
# falsify_fresh_build_sha.sh -- three-case table + mutants for fresh_build_sha_check.sh (aprender#6679)
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || exit 2
SRC="${FB_SRC:-$HERE/fresh_build_sha_check.sh}"
if [ "${1:-}" = --mutants ]; then
    T=$(mktemp -d) || exit 2; trap "rm -rf -- \"\${T:?}\"" EXIT; miss=0
    mut() { sed "$2" "$SRC" > "$T/m.sh"; cmp -s "$SRC" "$T/m.sh" && { echo "MUTANT $1 did not apply" >&2; miss=$((miss+1)); return; }
            FB_SRC="$T/m.sh" bash "$0" > /dev/null 2>&1 && { echo "MUTANT $1 SURVIVED" >&2; miss=$((miss+1)); }; }
    mut mismatch-passes "s/elif \[ \"\$got\" != \"\$want\" \]/elif false/"
    mut missing-sha-passes "s/if \[ -z \"\$got\" \]; then/if false; then/"
    mut no-receipts-passes "s/\[ \"\${#recs\[@\]}\" -gt 0 \] ||/true ||/"
    echo "falsify_fresh_build_sha --mutants: $miss not caught"; [ "$miss" = 0 ]; exit
fi
TMP=$(mktemp -d) || exit 2; trap "rm -rf -- \"\${TMP:?}\"" EXIT
pass=0 fail=0
expect() { if [ "$2" = "$3" ]; then pass=$((pass+1)); else fail=$((fail+1)); echo "FAIL: $1: want $2, got $3" >&2; fi; }
R="$TMP/r"; git init -q "$R"; git -C "$R" -c user.email=f@x -c user.name=f -c commit.gpgsign=false -c core.hooksPath=/dev/null commit -q --allow-empty -m c0
SHA=$(git -C "$R" rev-parse HEAD)
printf "#!/bin/sh\necho fresh\n" > "$TMP/apr"; chmod +x "$TMP/apr"
printf "#!/usr/bin/env bash\nprintf %%s \"%s\"\n" "$TMP/apr" > "$TMP/builder.sh"
GOOD=$(sha256sum "$TMP/apr" | cut -d" " -f1)
run() { bash "$SRC" --sha "$SHA" --repo "$R" --receipts "$1" --builder "bash $TMP/builder.sh" "${@:2}" > "$TMP/out" 2>&1; echo $?; }
mk() { rm -rf "$TMP/$1"; mkdir -p "$TMP/$1"; }
mk m; printf "{\"binary_sha256\":\"%s\"}\n" "$GOOD" > "$TMP/m/lambda-gpu.meta.json"; printf "{\"apr\":{\"binary_sha256\":\"%s\"}}\n" "$GOOD" > "$TMP/m/gx10-gpu.meta.json"
expect match-passes 0 "$(run "$TMP/m")"
mk x; printf "{\"binary_sha256\":\"%s\"}\n" "$GOOD" > "$TMP/x/lambda-gpu.meta.json"; printf "{\"binary_sha256\":\"%064d\"}\n" 7 > "$TMP/x/gx10-gpu.meta.json"
expect mismatch-red 1 "$(run "$TMP/x")"; grep -q "!= fresh build" "$TMP/out" && pass=$((pass+1)) || { fail=$((fail+1)); echo "FAIL: mismatch not named" >&2; }
mk n; printf "{\"models\":[]}\n" > "$TMP/n/lambda-gpu.meta.json"
expect missing-sha-red 1 "$(run "$TMP/n")"; grep -q "names no binary_sha256" "$TMP/out" && pass=$((pass+1)) || { fail=$((fail+1)); echo "FAIL: missing not named" >&2; }
mk e; expect no-receipts-red 1 "$(run "$TMP/e")"
printf "#!/usr/bin/env bash\nexit 0\n" > "$TMP/builder.sh"; expect build-nothing-red 1 "$(run "$TMP/m")"
printf "falsify_fresh_build_sha: %d held, %d failed\n" "$pass" "$fail"; [ "$fail" = 0 ]
