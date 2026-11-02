#!/usr/bin/env bash
# package_lane.sh -- graph-ordered `cargo package` lane: publish order from `cargo metadata`, never from TIERS[] (0.73.0 plan N1/A3, #3892)
#
#   bash scripts/release/package_lane.sh --out DIR [--sha <40-hex>] [--list-only] [--only a,b] [--min-crates N]
#   bash scripts/release/package_lane.sh --verify-order order.txt           # the A3 order check, reusable by tag CI
#
#   --out DIR         evidence dir, OUTSIDE the repo (a dir inside it dirties the tree and cargo refuses). Gets:
#                       order.txt      one crate name per line, every crate after all its in-set dependencies
#                       crates/*.crate the `cargo package --no-verify --locked` output, packaged in that order
#                       SHA256SUMS     sha256 of order.txt and of every .crate (`sha256sum -c` from inside DIR)
#                       package-lane.json  receipt {lane,sha,status:green,...}: written LAST and ONLY when all of the above held
#                       logs/<crate>.log   the cargo output of each crate
#   --list-only       derive and verify the order, write order.txt only: no cargo package, no SHA256SUMS, no receipt
#   --sha SHA         refuse unless the repo HEAD is exactly this sha (the receipt is bound to it)
#   --only a,b        restrict the set to these crates (a small real run); an unknown name is an error
#   --min-crates N    vacuity floor; default 70 for the real two-workspace enumeration, 1 otherwise
#   --root DIR        repo to enumerate and package (default: the repo this script lives in)
#   --metadata FILE   use this `cargo metadata --no-deps --format-version 1` document instead of running cargo (repeatable;
#                     this is how scripts/release/falsify_package_lane.sh feeds fixtures)
#   CARGO=...         the cargo to run (default `cargo`); the falsifier stubs it
#
# The graph: every publishable crate (`publish` unset or listing crates-io) of the root workspace AND of crates/facades
# (a second, excluded workspace); an edge for each normal and build dependency on another crate in the set. Dev-dependencies
# carry no edge: they are stripped on publish when path-only and would make a cycle out of every `foo-test-support`.
# Ties break by (level, name), so the order is deterministic. A cycle is an error, not an order.
#
# No `--allow-dirty`: the nightly packages a clean checkout of the pick sha, and a dirty one is refused by cargo.
# exit: 0 done - 1 a check or a package failed (reason printed, no receipt) - 2 usage
set -euo pipefail

PROG=package_lane
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$HERE/../.." && pwd)
out="" sha="" list_only=0 only="" min_crates="" verify_file="" meta_given=0
META=()
SCRATCH=""

usage() { printf '%s: %s\n' "$PROG" "$*" >&2; exit 2; }
die() { printf '%s: %s\n' "$PROG" "$*" >&2; exit 1; }
cleanup() { if [[ -n "$SCRATCH" ]]; then rm -rf -- "${SCRATCH:?}"; fi; }
trap cleanup EXIT

while [[ $# -gt 0 ]]; do
    case "$1" in
        --list-only) list_only=1; shift; continue ;;
        -h | --help) sed -n '2,32p' "${BASH_SOURCE[0]}"; exit 0 ;;
    esac
    [[ $# -ge 2 ]] || usage "$1 needs a value"
    case "$1" in
        --out) out=$2 ;;
        --sha) sha=$2 ;;
        --only) only=$2 ;;
        --min-crates) min_crates=$2 ;;
        --root) root=$2 ;;
        --verify-order) verify_file=$2 ;;
        --metadata) META+=("$2"); meta_given=1 ;;
        *) usage "unknown argument: $1" ;;
    esac
    shift 2
done

command -v jq > /dev/null 2>&1 || usage "jq is not installed"
if [[ -z "$verify_file" && -z "$out" ]]; then usage "--out is required"; fi
if [[ -n "$min_crates" && ! "$min_crates" =~ ^[0-9]+$ ]]; then usage "--min-crates must be a number"; fi
if [[ -n "$sha" && ! "$sha" =~ ^[0-9a-f]{40}$ ]]; then usage "--sha must be a full 40-hex sha"; fi
SCRATCH=$(mktemp -d)

# The graph, as one JSON document. Input: the slurped metadata documents. $only: null or a list of names.
JQ_GRAPH=$(
    cat << 'JQ'
def publishable: .publish == null or (.publish | index("crates-io") != null);
([.[].packages[] | select(publishable)] | unique_by(.name)) as $all
| ($all | map(.name)) as $allnames
| ($all | map(select($only == null or (.name as $n | $only | index($n) != null)))) as $pk
| ($pk | map(.name)) as $names
| ($names | map({key: ., value: true}) | from_entries) as $set
| ($pk | map(. as $p | {key: $p.name, value: ([$p.dependencies[] | select(.kind != "dev" and .name != $p.name and $set[.name] == true) | .name] | unique)}) | from_entries) as $deps
| ($pk | map(. as $p | [$p.dependencies[] | select(.kind == "dev" and .name != $p.name and $set[.name] == true) | "\($p.name) -> \(.name)"]) | add // [] | unique) as $dev
| ($names | length) as $n
| (reduce range(0; $n + 1) as $_ ({lv: ($names | map({key: ., value: 0}) | from_entries), ch: false};
      . as $s
      | ($names | map({key: ., value: ([$deps[.][] | $s.lv[.] + 1] | max // 0)}) | from_entries) as $nl
      | {lv: $nl, ch: ($nl != $s.lv)})) as $fix
| {
    order: (if $fix.ch then [] else ($names | sort_by([$fix.lv[.], .])) end),
    cyclic: (if $fix.ch then [$names[] | select($fix.lv[.] >= $n)] else [] end),
    unknown_only: (($only // []) - $allnames),
    levels: $fix.lv,
    deps: $deps,
    dev_edges: $dev,
    versions: ($pk | map({key: .name, value: .version}) | from_entries),
    manifests: ($pk | map({key: .name, value: .manifest_path}) | from_entries)
  }
JQ
)

# Every violation of the order file against the graph, one per line; no output = the order is valid.
JQ_VERIFY=$(
    cat << 'JQ'
. as $g
| $ord as $o
| ($o | to_entries | reverse | map({key: .value, value: .key}) | from_entries) as $pos
| [
    ($o | group_by(.) | map(select(length > 1)) | .[] | "duplicate entry: \(.[0])"),
    ($o[] | select($g.deps[.] == null) | "not a publishable crate of the set: \(.)"),
    ($g.deps | keys[] | select($pos[.] == null) | "missing from the order: \(.)"),
    ($o[] | . as $c | select($g.deps[$c] != null) | $g.deps[$c][] | select($pos[.] != null and $pos[.] > $pos[$c]) | "\($c) (#\($pos[$c] + 1)) is listed before its dependency \(.) (#\($pos[.] + 1))")
  ] | unique | .[]
JQ
)

gather_metadata() {
    local i=0 m
    for m in "$root/Cargo.toml" "$root/crates/facades/Cargo.toml"; do
        [[ -f "$m" ]] || continue
        i=$((i + 1))
        "${CARGO:-cargo}" metadata --no-deps --format-version 1 --manifest-path "$m" > "$SCRATCH/meta$i.json" || die "cargo metadata failed for $m"
        META+=("$SCRATCH/meta$i.json")
    done
    [[ "$i" -gt 0 ]] || die "no Cargo.toml under $root"
}

# validate every given metadata document before jq sees it: an unreadable or non-metadata file must not read as an empty set
check_metadata() {
    local f
    for f in "${META[@]}"; do
        jq -e '.packages | type == "array"' "$f" > /dev/null 2>&1 || die "not a cargo metadata document: $f"
    done
}

build_graph() {
    local only_json=null
    if [[ -n "$only" ]]; then only_json=$(jq -nc --arg s "$only" '$s | split(",") | map(select(length > 0))'); fi
    GRAPH=$(jq -s --argjson only "$only_json" "$JQ_GRAPH" "${META[@]}") || die "cannot build the dependency graph"
    local bad
    bad=$(jq -r '.unknown_only | join(" ")' <<< "$GRAPH")
    [[ -z "$bad" ]] || die "--only names no publishable crate: $bad"
    bad=$(jq -r '.cyclic | join(" ")' <<< "$GRAPH")
    [[ -z "$bad" ]] || die "dependency cycle (or downstream of one) among: $bad"
    local n floor
    n=$(jq '.order | length' <<< "$GRAPH")
    floor=${min_crates:-1}
    if [[ -z "$min_crates" && "$meta_given" -eq 0 && -z "$only" ]]; then floor=70; fi
    [[ "$n" -ge 1 ]] || die "vacuity: the graph has no publishable crate"
    [[ "$n" -ge "$floor" ]] || die "vacuity: enumerated $n publishable crate(s), expected at least $floor"
}

# verify_order FILE: 0 when every crate is listed after all of its in-set dependencies, exactly once, and none is missing
verify_order() {
    local file=$1 ord viol
    [[ -s "$file" ]] || die "order file missing or empty: $file"
    ord=$(jq -R 'select(length > 0 and (startswith("#") | not))' "$file" | jq -s .)
    viol=$(jq -r --argjson ord "$ord" "$JQ_VERIFY" <<< "$GRAPH")
    if [[ -n "$viol" ]]; then
        printf '%s: ORDER REFUSED %s\n%s\n' "$PROG" "$file" "$viol" >&2
        return 1
    fi
    printf '%s: order ok: %s crate(s) in %s\n' "$PROG" "$(jq length <<< "$ord")" "$file" >&2
}

# SHA256SUMS over order.txt and the given crates (paths relative to $out, so `sha256sum -c` works from inside the evidence dir), then check it
sums_in_out() {
    cd "$out" || return 1
    sha256sum -- order.txt "$@" > SHA256SUMS && sha256sum --quiet -c SHA256SUMS
}

package_all() {
    local head c ver man f line
    local files=()
    git -C "$root" rev-parse --git-dir > /dev/null 2>&1 || die "$root is not a git checkout: no sha to bind the receipt to"
    head=$(git -C "$root" rev-parse HEAD)
    if [[ -n "$sha" && "$sha" != "$head" ]]; then die "HEAD is ${head:0:9}, not the requested ${sha:0:9}"; fi
    install -d -- "$out/crates" "$out/logs"   # $out is realpath-resolved and checked to lie outside the repo above
    while IFS=$'\t' read -r c ver man; do
        f="$SCRATCH/target/package/$c-$ver.crate"
        if ! "${CARGO:-cargo}" package --no-verify --locked --manifest-path "$man" --target-dir "$SCRATCH/target" > "$out/logs/$c.log" 2>&1; then
            tail -n 20 "$out/logs/$c.log" >&2
            die "cargo package failed for $c (log: $out/logs/$c.log); no SHA256SUMS, no receipt"
        fi
        [[ -s "$f" ]] || die "cargo package reported success for $c but wrote no $c-$ver.crate"
        cp -- "$f" "$out/crates/"
        files+=("crates/$c-$ver.crate")
        printf '%s: packaged %s %s\n' "$PROG" "$c" "$ver" >&2
    done < <(jq -r '.order[] as $c | [$c, .versions[$c], .manifests[$c]] | @tsv' <<< "$GRAPH")
    (sums_in_out "${files[@]}") || die "SHA256SUMS does not verify"
    line=$(sha256sum < "$out/order.txt")
    jq -n --arg s "$head" --argjson n "${#files[@]}" --arg o "${line%% *}" --arg f "$(sha256sum < "$out/SHA256SUMS" | cut -d' ' -f1)" \
        '{lane: "package-order", sha: $s, status: "green", crates: $n, order_sha256: $o, sums_sha256: $f}' > "$out/package-lane.json"
    printf '%s: packaged %s crate(s) for %s into %s\n' "$PROG" "${#files[@]}" "${head:0:9}" "$out" >&2
}

[[ "$meta_given" -eq 1 ]] || gather_metadata
check_metadata
build_graph

if [[ -n "$verify_file" ]]; then
    verify_order "$verify_file"
    exit $?
fi

abs_out=$(realpath -m -- "$out")
case "$abs_out/" in
    "$(realpath -m -- "$root")"/*) usage "--out must be outside the repo ($abs_out is inside $root): evidence there dirties the tree" ;;
    *) ;;
esac
out="$abs_out"
mkdir -p -- "$out"
rm -rf -- "${out:?}/crates" "${out:?}/logs"
rm -f -- "${out:?}/order.txt" "${out:?}/SHA256SUMS" "${out:?}/package-lane.json"
jq -r '.order[]' <<< "$GRAPH" > "$out/order.txt"
verify_order "$out/order.txt" || die "derived order failed its own check"
printf '%s: %s dev-dependency edge(s) inside the set carry no ordering\n' "$PROG" "$(jq '.dev_edges | length' <<< "$GRAPH")" >&2
if [[ "$list_only" -eq 1 ]]; then
    printf '%s: --list-only: wrote %s only\n' "$PROG" "$out/order.txt" >&2
    exit 0
fi
package_all
