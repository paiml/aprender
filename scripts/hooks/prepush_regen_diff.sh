#!/usr/bin/env bash
# prepush_regen_diff.sh - RS-6 (operator C49): `scripts/contracts_gate.sh regen` must leave
# contracts/contracts.nt + contracts/shapes.ttl byte-identical to what is committed, else the
# push is refused (the merge_group `guards` job fails on exactly this diff).
#   bash scripts/hooks/prepush_regen_diff.sh              # check (run by .githooks/pre-push)
#   bash scripts/hooks/prepush_regen_diff.sh --self-test  # case table
# REGEN_CMD overrides the regen command (self-test only). rc 0 clean, 1 diff, 2 regen could not run.
set -uo pipefail
GENERATED=(contracts/contracts.nt contracts/shapes.ttl)

check() {
    local root=$1 regen=${REGEN_CMD:-"bash scripts/contracts_gate.sh regen"}
    ( cd "$root" && eval "$regen" > /dev/null 2>&1 ) || { echo "RS-6 ENV: regen failed to run (rc!=0) -- not measured" >&2; return 2; }
    if git -C "$root" diff --quiet -- "${GENERATED[@]}"; then
        echo "RS-6 ok: regen leaves ${GENERATED[*]} unchanged"; return 0
    fi
    echo "RS-6 FAIL: regen changed a generated file; commit contracts/contracts.nt contracts/shapes.ttl" >&2
    git -C "$root" diff --stat -- "${GENERATED[@]}" >&2
    return 1
}

self_test() {
    local d rc bad=0
    d=$(mktemp -d) || return 2
    git -C "$d" init -q && mkdir -p "$d/contracts" && printf 'a\n' > "$d/contracts/contracts.nt" && printf 's\n' > "$d/contracts/shapes.ttl" \
        && git -C "$d" add -A && git -C "$d" -c user.email=t@t -c user.name=t commit -q -m i || return 2
    REGEN_CMD=true check "$d" > /dev/null 2>&1; rc=$?; [ "$rc" -eq 0 ] || { echo "FAIL row clean: rc=$rc"; bad=1; }
    REGEN_CMD="printf 'b\n' >> contracts/contracts.nt" check "$d" > /dev/null 2>&1; rc=$?; [ "$rc" -eq 1 ] || { echo "FAIL row drift: rc=$rc"; bad=1; }
    git -C "$d" checkout -q -- contracts
    REGEN_CMD=false check "$d" > /dev/null 2>&1; rc=$?; [ "$rc" -eq 2 ] || { echo "FAIL row regen-broken: rc=$rc"; bad=1; }
    rm -rf -- "${d:?}"
    [ "$bad" -eq 0 ] && echo "self-test: PASS"
    return "$bad"
}

case "${1:-}" in
    --self-test) self_test ;;
    *) check "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" ;;
esac
