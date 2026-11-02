#!/usr/bin/env bash
# publish_train.sh -- publish the nightly package-lane evidence to crates.io, in order, resumable (0.73.0 plan N4/A1/A4)
#
#   bash scripts/release/publish_train.sh --evidence DIR --sha <40-hex> [--dry-run] [options]
#
#   --evidence DIR     the package_lane.sh --out dir: order.txt, crates/*.crate, SHA256SUMS, package-lane.json
#   --sha SHA          the tag/pick sha. The receipt must be green for exactly this sha and the repo HEAD must be it. Required.
#   --dry-run          do every check and the rebuild comparison, then PRINT what would be published. No `cargo publish`, no
#                      token needed, no polling. Reads the public index only to say which crates it would skip.
#   --root DIR         the checkout at --sha that `cargo publish` packages from (default: the repo this script lives in)
#   --index-url URL    sparse index to poll (default https://index.crates.io)
#   --poll-max N       index reads per crate before giving up (default 60)
#   --poll-interval S  seconds between reads (default 5; 0 allowed). Total wait per crate is at most N*S, never a tight loop.
#   --no-rebuild-check skip the bytes comparison (see "What is published" below); default is to compare
#   --receipt FILE     write {sha,dry_run,published,skipped,seconds} here when the run completes
#   --metadata FILE    use this `cargo metadata --no-deps --format-version 1` document (repeatable; the falsifier's fixtures)
#   CARGO=... CURL=... PACKAGE_LANE=...   the cargo, curl and package_lane.sh to run (the falsifier stubs them)
#   CARGO_REGISTRY_TOKEN   the crates.io token. The ONLY way a token is accepted: there is no --token flag, and a real run
#                          without the variable is refused before anything is published. It is never printed or logged.
#
# Order of work (nothing is uploaded until every check has passed):
#   1. the evidence: `sha256sum -c SHA256SUMS`; every .crate in crates/ is listed in SHA256SUMS and in order.txt; every crate in
#      order.txt has exactly one .crate, whose version equals the manifest's; package-lane.json is green, for --sha, and its
#      order_sha256 / sums_sha256 / crates agree with the files; the repo HEAD is --sha.
#   2. the order: `package_lane.sh --verify-order order.txt` against the graph (a crate listed before a dependency is refused).
#   3. the bytes (see below).
#   4. for each crate in order.txt order: already visible in the index -> skip; else `cargo publish --no-verify --locked`, then poll
#      the index until that exact version is visible (bounded) BEFORE the next crate. "already uploaded / already exists" from
#      cargo is success, but the version must still show in the index. Any other failure stops the train (exit 1); rerun resumes
#      at the first crate the index does not yet show (A4).
#
# What is published. `cargo publish` has no flag to upload a prebuilt .crate: it packages again from --root. So the evidence
# binds what is uploaded only through step 3: every crate is re-packaged (`cargo package --no-verify --locked`) and its sha256
# must equal the evidence .crate, else exit 1 before the first upload. Measured 2026-11-02 on cargo 1.93: the same crate packaged
# twice, into two target dirs, and again after `touch` of a source file, gives identical bytes (cargo fixes mtimes), so the
# comparison is sound on the same cargo + the same sha. A different cargo on the publish runner may break it; --no-rebuild-check
# turns it off, and that choice belongs to the operator (open question recorded in the W5 report).
#
# FINDING (W2's open question) -- versioned dev-dependencies on unpublished siblings:
#   * cargo: `cargo package`/`publish` strips a dev-dependency that has a path and NO version; it KEEPS one that has a version
#     (verified 2026-11-02: aprender-gpu's path-only dev-deps on aprender-simulate / aprender-test-lib are absent from the packaged
#     Cargo.toml). `cargo metadata` shows the stripped kind as req="*" with a path; of the dev-edges inside this set, the versioned
#     ones are aprender-compute -> {cuda-edge, simulate, solve, sparse} and {core, orchestrate, present-terminal, serve, train}
#     -> contracts (req ^0.72.3), all of which exist on crates.io at 0.72.3.
#   * crates.io (server side, publish endpoint) looks up every dependency of every kind by crate NAME and refuses with "no known
#     crate named X" when the crate has never been published; it does not require a published version that satisfies the
#     requirement for a dev-dependency (only `*` wildcards are refused). So a kept, versioned dev-dep on a sibling is accepted as
#     long as the sibling has been published at least once under any version, and it adds no ordering constraint within a train.
#     Consequence: package_lane.sh is right to give dev-edges no ordering weight; the only case that needs care is a crate that
#     has NEVER been published and is some other crate's versioned dev-dependency -- publish it first by hand or add the edge.
#     This server-side half is from the crates.io source/behaviour, not from a live upload (publishing to test it is not
#     reversible); the W7 rehearsal on a scratch tag is where it is proven. The falsifier does not claim it:
#     it cannot be tested without crates.io.
#
# exit: 0 done (or dry-run ok) - 1 a check or a publish failed (reason printed) - 2 usage / no token
set -euo pipefail

PROG=publish_train
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$HERE/../.." && pwd)
evid="" sha="" dry=0 index_url="https://index.crates.io" poll_max=60 poll_int=5 rebuild=1 receipt="" meta_given=0
META=()
SCRATCH=""
PUBLISHED=()
SKIPPED=()

usage() { printf '%s: %s\n' "$PROG" "$*" >&2; exit 2; }
die() { printf '%s: %s\n' "$PROG" "$*" >&2; exit 1; }
log() { printf '%s: %s\n' "$PROG" "$*" >&2; }
cleanup() { if [[ -n "$SCRATCH" ]]; then rm -rf -- "${SCRATCH:?}"; fi; }
trap cleanup EXIT

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) dry=1; shift; continue ;;
        --no-rebuild-check) rebuild=0; shift; continue ;;
        -h | --help) sed -n '2,12p' "${BASH_SOURCE[0]}"; exit 0 ;;
    esac
    [[ $# -ge 2 ]] || usage "$1 needs a value"
    case "$1" in
        --evidence) evid=$2 ;;
        --sha) sha=$2 ;;
        --root) root=$2 ;;
        --index-url) index_url=$2 ;;
        --poll-max) poll_max=$2 ;;
        --poll-interval) poll_int=$2 ;;
        --receipt) receipt=$2 ;;
        --metadata) META+=("$2"); meta_given=1 ;;
        *) usage "unknown argument: $1" ;;
    esac
    shift 2
done

command -v jq > /dev/null 2>&1 || usage "jq is not installed"
[[ -n "$evid" ]] || usage "--evidence is required"
[[ "$sha" =~ ^[0-9a-f]{40}$ ]] || usage "--sha must be a full 40-hex sha"
[[ "$poll_max" =~ ^[1-9][0-9]*$ ]] || usage "--poll-max must be a positive number"
[[ "$poll_int" =~ ^[0-9]+$ ]] || usage "--poll-interval must be a number"
[[ -d "$evid" ]] || usage "--evidence is not a directory: $evid"
if [[ "$dry" -eq 0 && -z "${CARGO_REGISTRY_TOKEN:-}" ]]; then usage "CARGO_REGISTRY_TOKEN is not set: a real run publishes with the environment token only"; fi
evid=$(cd "$evid" && pwd)
SCRATCH=$(mktemp -d)

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

# crate name from a .crate file name: everything before the first -<digits>.<digits>.<digits> (crate names carry no dots)
crate_name_of() { local b=${1##*/}; b=${b%.crate}; printf '%s\n' "$b" | sed -E 's/-[0-9]+\.[0-9]+\.[0-9]+.*$//'; }

# ---- step 1: the evidence
check_evidence() {
    local f c n want have exp_c f_count list
    for f in order.txt SHA256SUMS package-lane.json; do [[ -s "$evid/$f" ]] || die "evidence incomplete: $evid/$f missing or empty"; done
    [[ -d "$evid/crates" ]] || die "evidence incomplete: $evid/crates missing"
    (cd "$evid" && sha256sum --quiet -c SHA256SUMS > /dev/null 2>&1) || die "SHA256SUMS does not verify (a listed file is missing or changed)"
    log "SHA256SUMS verifies"
    jq -e --arg s "$sha" '.lane == "package-order" and .status == "green" and .sha == $s' "$evid/package-lane.json" > /dev/null 2>&1 \
        || die "package-lane.json is not a green package-order receipt for $sha"
    want=$(jq -r '.order_sha256' "$evid/package-lane.json"); have=$(sha256sum < "$evid/order.txt"); have=${have%% *}
    [[ "$want" == "$have" ]] || die "order.txt does not match the receipt's order_sha256"
    want=$(jq -r '.sums_sha256' "$evid/package-lane.json"); have=$(sha256sum < "$evid/SHA256SUMS"); have=${have%% *}
    [[ "$want" == "$have" ]] || die "SHA256SUMS does not match the receipt's sums_sha256"
    ORDER=()
    while IFS= read -r c; do
        [[ -z "$c" || "$c" == \#* ]] && continue
        ORDER+=("$c")
    done < "$evid/order.txt"
    n=${#ORDER[@]}
    [[ "$n" -ge 1 ]] || die "order.txt lists no crate"
    [[ "$(jq -r '.crates' "$evid/package-lane.json")" == "$n" ]] || die "receipt says $(jq -r '.crates' "$evid/package-lane.json") crates, order.txt lists $n"
    # every .crate on disk is in order.txt and in SHA256SUMS; every order entry has exactly one .crate
    list=$(printf '%s\n' "${ORDER[@]}")
    f_count=0
    for f in "$evid"/crates/*.crate; do
        [[ -e "$f" ]] || break
        f_count=$((f_count + 1))
        c=$(crate_name_of "$f")
        grep -qxF -- "$c" <<< "$list" || die "crates/${f##*/} is not listed in order.txt"
        grep -qE -- "^[0-9a-f]{64}  crates/${f##*/}\$" "$evid/SHA256SUMS" || die "crates/${f##*/} is not covered by SHA256SUMS"
    done
    for c in "${ORDER[@]}"; do
        exp_c=0
        for f in "$evid"/crates/*.crate; do
            [[ -e "$f" ]] || break
            if [[ "$(crate_name_of "$f")" == "$c" ]]; then exp_c=$((exp_c + 1)); fi
        done
        [[ "$exp_c" -ge 1 ]] || die "crate missing from the evidence: no crates/$c-*.crate"
        [[ "$exp_c" -eq 1 ]] || die "more than one crates/$c-*.crate in the evidence"
    done
    [[ "$f_count" -eq "$n" ]] || die "evidence has $f_count .crate file(s) for $n crate(s) in order.txt"
    local head
    git -C "$root" rev-parse --git-dir > /dev/null 2>&1 || die "$root is not a git checkout"
    head=$(git -C "$root" rev-parse HEAD)
    [[ "$head" == "$sha" ]] || die "repo HEAD is ${head:0:9}, not $sha: cargo publish would package a different tree"
    log "evidence ok: $n crate(s), receipt green for ${sha:0:9}"
}

# ---- step 2: the order, against the graph
check_order() {
    local args=("--verify-order" "$evid/order.txt") m
    for m in "${META[@]}"; do args+=("--metadata" "$m"); done
    bash "${PACKAGE_LANE:-$HERE/package_lane.sh}" "${args[@]}" || die "order.txt is not a valid dependency order"
}

# name -> "version<TAB>manifest" from the metadata; the .crate version must equal it
resolve() {
    local c=$1 row f ver
    row=$(jq -s -r --arg c "$c" '[.[].packages[] | select(.name == $c)] | first | if . == null then empty else [.version, .manifest_path] | @tsv end' "${META[@]}")
    [[ -n "$row" ]] || die "$c has no manifest in the cargo metadata"
    ver=${row%%$'\t'*}
    f="$evid/crates/$c-$ver.crate"
    [[ -s "$f" ]] || die "evidence holds a different version of $c than the manifest ($ver): no crates/$c-$ver.crate"
    printf '%s\n' "$row"
}

# ---- step 3: the bytes
check_bytes() {
    local c row ver man want have
    for c in "${ORDER[@]}"; do
        row=$(resolve "$c"); ver=${row%%$'\t'*}; man=${row#*$'\t'}
        [[ "$rebuild" -eq 1 ]] || continue
        "${CARGO:-cargo}" package --no-verify --locked --manifest-path "$man" --target-dir "$SCRATCH/target" > "$SCRATCH/pkg-$c.log" 2>&1 \
            || { tail -n 10 "$SCRATCH/pkg-$c.log" >&2; die "re-packaging $c failed"; }
        have=$(sha256sum < "$SCRATCH/target/package/$c-$ver.crate" 2> /dev/null) || die "re-packaging $c wrote no $c-$ver.crate"
        want=$(sha256sum < "$evid/crates/$c-$ver.crate")
        [[ "${have%% *}" == "${want%% *}" ]] || die "$c-$ver.crate re-packages to different bytes than the evidence (${have:0:12} vs ${want:0:12}): refusing to publish what was not verified"
    done
    if [[ "$rebuild" -eq 1 ]]; then log "bytes ok: every crate re-packages to its evidence hash"; else log "bytes NOT compared (--no-rebuild-check)"; fi
}

# ---- step 4: index + publish
index_path() {
    local n=$1
    n=$(printf '%s' "$n" | tr '[:upper:]' '[:lower:]')
    case ${#n} in
        1) printf '1/%s' "$n" ;;
        2) printf '2/%s' "$n" ;;
        3) printf '3/%s/%s' "${n:0:1}" "$n" ;;
        *) printf '%s/%s/%s' "${n:0:2}" "${n:2:2}" "$n" ;;
    esac
}

# index_has CRATE VERSION: 0 visible and not yanked, 1 absent, 2 unreadable
index_has() {
    local c=$1 v=$2 resp code body
    resp=$("${CURL:-curl}" -sS --max-time 30 -H "User-Agent: aprender-publish-train (https://github.com/paiml/aprender)" -w '\n%{http_code}' "$index_url/$(index_path "$c")" 2> /dev/null) || return 2
    code=$(printf '%s' "$resp" | tail -n 1)
    body=$(printf '%s' "$resp" | sed '$d')
    case "$code" in
        200) printf '%s' "$body" | jq -s -e --arg v "$v" 'any(.[]; .vers == $v and .yanked != true)' > /dev/null 2>&1 && return 0 || return 1 ;;
        404) return 1 ;;
        *) return 2 ;;
    esac
}

# wait_visible CRATE VERSION: bounded polling, a sleep between reads
wait_visible() {
    local c=$1 v=$2 i
    for ((i = 1; i <= poll_max; i++)); do
        if index_has "$c" "$v"; then
            log "index shows $c $v (read $i)"
            return 0
        fi
        if [[ "$i" -lt "$poll_max" ]]; then sleep "$poll_int"; fi
    done
    return 1
}

publish_one() {
    local c=$1 ver=$2 man=$3 rc=0
    "${CARGO:-cargo}" publish --no-verify --locked --manifest-path "$man" --target-dir "$SCRATCH/target" > "$SCRATCH/pub-$c.log" 2>&1 || rc=$?
    if [[ "$rc" -ne 0 ]]; then
        if grep -qiE 'already (uploaded|exists)' "$SCRATCH/pub-$c.log"; then
            log "$c $ver was already uploaded: continuing"
        else
            tail -n 15 "$SCRATCH/pub-$c.log" >&2
            die "cargo publish failed for $c $ver (rc=$rc); the train stops here, rerun resumes"
        fi
    fi
}

train() {
    local c row ver man rc
    for c in "${ORDER[@]}"; do
        row=$(resolve "$c"); ver=${row%%$'\t'*}; man=${row#*$'\t'}
        rc=0; index_has "$c" "$ver" || rc=$?
        if [[ "$rc" -eq 0 ]]; then
            log "skip $c $ver: already in the index"
            SKIPPED+=("$c")
            continue
        fi
        if [[ "$dry" -eq 1 ]]; then
            log "dry-run: would publish $c $ver"
            continue
        fi
        log "publishing $c $ver"
        publish_one "$c" "$ver" "$man"
        wait_visible "$c" "$ver" || die "$c $ver never appeared in the index after $poll_max read(s) x ${poll_int}s: stopping before its dependents"
        PUBLISHED+=("$c")
    done
}

write_receipt() {
    [[ -n "$receipt" ]] || return 0
    jq -n --arg s "$sha" --argjson d "$dry" --argjson sec "$SECONDS" \
        --argjson p "$(printf '%s\n' "${PUBLISHED[@]:-}" | jq -R 'select(length > 0)' | jq -s .)" \
        --argjson k "$(printf '%s\n' "${SKIPPED[@]:-}" | jq -R 'select(length > 0)' | jq -s .)" \
        '{lane: "publish-train", sha: $s, dry_run: ($d == 1), published: $p, skipped: $k, seconds: $sec}' > "$receipt"
}

[[ "$meta_given" -eq 1 ]] || gather_metadata
ORDER=()
check_evidence
check_order
check_bytes
train
write_receipt
log "done: ${#PUBLISHED[@]} published, ${#SKIPPED[@]} already in the index$([[ "$dry" -eq 1 ]] && printf ' (dry-run: nothing published)')"
