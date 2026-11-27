# shellcheck shell=bash
# crux_apr_sha.sh -- the sha256 of the apr binary an R7 CRUX receipt measured (#6679). Sourced by
# scripts/crux_inference_dogfood.sh and scripts/crux_sweep_shards.sh; the case table is
# scripts/check_crux_receipt_binary_sha.sh. Every refusal prints why and returns 1.

# crux_apr_sha256 APR -> the 64-hex sha256 of the file APR names (a path, or a name on PATH)
crux_apr_sha256() {
    local bin sum
    bin=$(command -v -- "${1:-}") && [ -f "$bin" ] || { echo "apr binary '${1:-}' does not resolve to a file" >&2; return 1; }
    sum=$(sha256sum < "$bin") || { echo "cannot hash $bin" >&2; return 1; }
    sum=${sum%% *}
    [[ $sum =~ ^[0-9a-f]{64}$ ]] || { echo "sha256sum of $bin printed no digest" >&2; return 1; }
    printf '%s\n' "$sum"
}

# crux_stamp_apr_sha META APR PINNED -> set META's .apr.binary_sha256 to PINNED, the sha taken when
# APR was pinned. APR is hashed again: a binary replaced during the run is not the one measured.
crux_stamp_apr_sha() {
    local meta=$1 now pinned=${3:-}
    now=$(crux_apr_sha256 "$2") || return 1
    [ -n "$pinned" ] && [ "$now" = "$pinned" ] || { echo "apr changed during the run: pinned '$pinned', now '$now'" >&2; return 1; }
    jq --arg s "$pinned" 'if (.apr | type) == "object" then .apr.binary_sha256 = $s else error("meta has no apr object") end' \
        "$meta" > "$meta.stamped" || return 1
    mv -- "$meta.stamped" "$meta"
}

# crux_one_apr_sha META... -> 0 when every shard meta names the same binary sha. Shards that ALL
# predate the stamp agree on "none" (--merge-only on an old run); their receipt names no sha.
crux_one_apr_sha() {
    [ $# -gt 0 ] || { echo "no shard meta to compare" >&2; return 1; }
    local shas
    shas=$(jq -r '.apr.binary_sha256 // "none"' "$@") || { echo "a shard meta is unreadable" >&2; return 1; }
    shas=$(sort -u <<< "$shas")
    [ "$(wc -l <<< "$shas")" -eq 1 ] || { echo "shards were measured on different apr binaries: $(tr '\n' ' ' <<< "$shas")" >&2; return 1; }
}
