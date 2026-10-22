#!/usr/bin/env bash
# seed_registry.sh -- keep crates.io off the merge path (aprender#5751, PMAT-5751).
#
# WHY: the cargo registry mounted into a job is keyed per PR (x shard). A new PR, and every
# merge-queue ref, starts with an EMPTY registry, so each shard downloads its whole dependency
# set from crates.io before it compiles a line. A WAN brownout then fails a merge job
# ("spurious network error: Timeout was reached ... linux-raw-sys / libsqlite3-sys", run
# 38363869044). Retrying hides the dependency; this removes it: a cold registry is seeded from a
# warm host-local copy, so cargo finds the crates on disk and only a genuinely NEW crate version
# touches the network.
#
#   seed_registry.sh seed    <registry dir> <seed dir>   cold registry <- seed. Never fails the job.
#   seed_registry.sh publish <registry dir> <seed dir>   warm registry -> seed (atomic rename).
#
# Only index/ and cache/ travel (src/ is re-extracted by cargo). A registry is COLD when it holds
# no cached .crate. seed never overwrites a warm registry. publish refuses a registry with no
# cached crate, so an empty one cannot replace a good seed. A missing seed is a cold start, exit 0.
set -uo pipefail

crate_count() { find "$1/cache" -name '*.crate' 2>/dev/null | head -n 1000 | wc -l; }

cmd_seed() {
    local reg="$1" seed="$2"
    if [ "$(crate_count "$reg")" -gt 0 ]; then
        printf 'seed_registry: %s is warm; left alone\n' "$reg"; return 0
    fi
    if [ "$(crate_count "$seed")" -eq 0 ]; then
        printf 'seed_registry: no seed at %s; cold start (crates.io will be read)\n' "$seed"; return 0
    fi
    mkdir -p "$reg" || return 0
    local d
    for d in index cache; do
        [ -d "$seed/$d" ] || continue
        cp -a --reflink=auto "$seed/$d" "$reg/" 2>/dev/null \
            || printf 'seed_registry: warning: could not copy %s/%s\n' "$seed" "$d"
    done
    printf 'seed_registry: seeded %s from %s (%s crates)\n' "$reg" "$seed" "$(crate_count "$reg")"
    return 0
}

cmd_publish() {
    local reg="$1" seed="$2" n tmp
    n=$(crate_count "$reg")
    if [ "$n" -eq 0 ]; then
        printf 'seed_registry: refusing to publish %s: no cached crates\n' "$reg" >&2; return 1
    fi
    if [ "$n" -lt "$(crate_count "$seed")" ]; then
        printf 'seed_registry: %s has fewer crates than the seed; seed kept\n' "$reg"; return 0
    fi
    mkdir -p "$(dirname -- "$seed")" || return 1
    tmp="${seed}.tmp.$$"
    mkdir -p "$tmp" && cp -a --reflink=auto "$reg/index" "$reg/cache" "$tmp/" || { rm -rf "${tmp:?}"; return 1; }
    rm -rf "${seed:?}.old" && { [ ! -d "$seed" ] || mv "$seed" "${seed}.old"; } && mv "$tmp" "$seed" || return 1
    rm -rf "${seed:?}.old"
    printf 'seed_registry: published %s crates to %s\n' "$n" "$seed"
}

case "${1:-}" in
    seed)    [ "$#" -eq 3 ] || { echo 'usage: seed <registry> <seed>' >&2; exit 64; }; cmd_seed "$2" "$3" ;;
    publish) [ "$#" -eq 3 ] || { echo 'usage: publish <registry> <seed>' >&2; exit 64; }; cmd_publish "$2" "$3" ;;
    *) echo 'usage: seed_registry.sh seed|publish <registry> <seed>' >&2; exit 64 ;;
esac
