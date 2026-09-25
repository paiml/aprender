#!/usr/bin/env bash
# promote_rc.sh — promote vX.Y.Z-rc.N to vX.Y.Z with NO rebuild (#4286, RC-DOGFOOD-001 §4.4)
#
#   bash scripts/release/promote_rc.sh vX.Y.Z-rc.N [--dry-run] [--work DIR]
#   bash scripts/release/promote_rc.sh --self-test
#
# The final's assets ARE the rc's bytes: the bytes that were dogfooded are the bytes that
# ship. Cargo stays at X.Y.Z in the tree (cop ruling, 2026-09-24), but binary-release.yml
# stamps an rc build to X.Y.Z-rc.N (stamp_rc_version.sh, #4110), so the promoted final
# prints `apr X.Y.Z-rc.N (<sha>)` — the rc it was promoted from, as #4286 asks.
# asset_version_check.sh accepts exactly that on vX.Y.Z when <sha> is the final tag's
# commit, which step 6 below guarantees. Only the asset NAME changes:
#
#   1. read the rc release: a prerelease, carrying all eighteen assets
#      (scripts/check_release_assets.sh), its tag resolved to a commit
#   2. download every asset and verify each tarball against its .sha256; a missing
#      .sha256 or a mismatch refuses
#   3. stage each tarball under its final name (`-vX.Y.Z-rc.N-` -> `-vX.Y.Z-`), with
#      a .sha256 whose hash EQUALS the rc's
#   4. create release vX.Y.Z as a DRAFT at the rc's commit, and upload
#   5. read every final asset back and assert its sha256 equals the rc's. A mismatch
#      exits 1 and deletes nothing: the draft stays for a human to inspect
#   6. publish the draft, then check that the tag vX.Y.Z points at the rc's commit
#
# The tarball's top directory keeps the rc label (apr-vX.Y.Z-rc.N-<target>-<variant>/):
# renaming it would change the bytes. install.sh, the binary-release smoke and
# autopilot's host check therefore take the one apr-* directory an archive holds.
#
# Publishing with a personal token fires `release: published` in binary-release.yml.
# Its `assets` job sees all eighteen present and skips every build, so nothing
# overwrites these bytes. Verify and smoke still run, on the promoted bytes.
#
# Exit: 0 promoted (or the dry run passed) · 1 refused / a check failed · 2 ENV/usage.
set -uo pipefail
PROG=promote_rc
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
API="https://api.github.com/repos/${GITHUB_REPOSITORY:-paiml/aprender}"
UPLOADS="https://uploads.github.com/repos/${GITHUB_REPOSITORY:-paiml/aprender}"

die()  { echo "$PROG: $*" >&2; exit 1; }
env_die() { echo "$PROG: ENV — $*" >&2; exit 2; }

# ── pure / offline parts (the case table runs these on real files) ─────────────

# final_of RC — vX.Y.Z-rc.N -> vX.Y.Z; empty on anything else
final_of() {
    if [[ $1 =~ ^(v[0-9]+\.[0-9]+\.[0-9]+)-rc\.[0-9]+$ ]]; then echo "${BASH_REMATCH[1]}"; fi
}

# final_name RC FINAL NAME — the asset's final name; empty unless NAME carries
# `-RC-` exactly once (an asset from some other tag is not renamed, it is refused)
final_name() {
    local rc=$1 final=$2 name=$3 rest
    rest=${name#*"-$rc-"}
    [ "$rest" != "$name" ] || return 0
    [ "${rest#*"$rc"}" = "$rest" ] || return 0
    echo "${name%%"-$rc-"*}-$final-$rest"
}

sha_of() { sha256sum "$1" | awk '{print $1}'; }

# stage RC FINAL SRC DST — verify every tarball in SRC against its .sha256, copy it
# into DST under its final name, and write the final .sha256 carrying the SAME hash.
# Prints `<final tarball>\t<hash>` per asset. Refuses (returns 1) on: an asset whose
# name does not carry the rc, a tarball without a .sha256, a .sha256 without its
# tarball, a hash mismatch, or nothing to promote.
stage() {
    local rc=$1 final=$2 src=$3 dst=$4 f base fin want have n=0
    for f in "$src"/*; do
        [ -f "$f" ] || continue
        base=${f##*/}
        fin=$(final_name "$rc" "$final" "$base")
        if [ -z "$fin" ]; then echo "$PROG: refuse: $base does not carry -$rc-" >&2; return 1; fi
        case "$base" in
            *.sha256)
                if [ ! -f "${f%.sha256}" ]; then echo "$PROG: refuse: $base has no tarball" >&2; return 1; fi
                continue ;;
        esac
        if [ ! -f "$f.sha256" ]; then echo "$PROG: refuse: $base has no .sha256" >&2; return 1; fi
        want=$(awk 'NR==1{print $1}' "$f.sha256")
        have=$(sha_of "$f")
        if [ "$want" != "$have" ]; then echo "$PROG: refuse: $base is $have, its .sha256 says $want" >&2; return 1; fi
        cp -- "$f" "$dst/$fin" || return 1
        printf '%s  %s\n' "$have" "$fin" > "$dst/$fin.sha256" || return 1
        printf '%s\t%s\n' "$fin" "$have"
        n=$((n + 1))
    done
    if [ "$n" -eq 0 ]; then echo "$PROG: refuse: nothing to promote in $src" >&2; return 1; fi
}

# readback MANIFEST DIR — every final tarball in DIR, and the hash its .sha256 states,
# equal the rc's hash recorded in MANIFEST. Prints each mismatch; returns 1 on any.
readback() {
    local manifest=$1 dir=$2 fin want bad=0 have stated
    while IFS=$'\t' read -r fin want; do
        have=$( [ -f "$dir/$fin" ] && sha_of "$dir/$fin" )
        stated=$( [ -f "$dir/$fin.sha256" ] && awk 'NR==1{print $1}' "$dir/$fin.sha256" )
        if [ "$have" != "$want" ]; then echo "$PROG: MISMATCH $fin is '${have:-missing}', the rc's bytes are $want"; bad=1; fi
        if [ "$stated" != "$want" ]; then echo "$PROG: MISMATCH $fin.sha256 states '${stated:-missing}', the rc's is $want"; bad=1; fi
    done < "$manifest"
    [ "$bad" -eq 0 ]
}

self_test() {
    local fail=0 w rc=v0.70.0-rc.3 final=v0.70.0 got
    w=$(mktemp -d) || return 2
    # shellcheck disable=SC2064
    trap "rm -rf -- '$w'" RETURN
    row() { # want(0|1) reason why cmd... — a refusal must refuse for ITS reason, not a later check's
        local want=$1 reason=$2 why=$3; shift 3
        if "$@" > "$w/out" 2>&1; then got=0; else got=1; fi
        if [ "$got" = "$want" ] && { [ -z "$reason" ] || grep -qF -- "$reason" "$w/out"; }; then echo "  ok   $why"
        else echo "  FAIL $why (wanted rc $want and '$reason', got rc $got)"; sed 's/^/       | /' "$w/out"; fail=1; fi
    }
    # a fixture rc: two real tarballs and their .sha256 files
    fixture() { # dir
        local d=$1 a
        mkdir -p "$d/pkg"
        for a in "apr-$rc-x86_64-unknown-linux-gnu-cpu" "pv-$rc-x86_64-unknown-linux-musl"; do
            mkdir -p "$d/pkg/$a"; printf 'bytes of %s\n' "$a" > "$d/pkg/$a/bin"
            tar czf "$d/$a.tar.gz" -C "$d/pkg" "$a"
            (cd "$d" && sha256sum "$a.tar.gz" > "$a.tar.gz.sha256")
        done
        rm -rf -- "${d:?}/pkg"
    }
    echo "$PROG self-test: names"
    [ "$(final_of v0.70.0-rc.3)" = v0.70.0 ] && echo "  ok   final_of strips -rc.N" || { echo "  FAIL final_of"; fail=1; }
    [ -z "$(final_of v0.70.0)" ] && echo "  ok   a final tag is not an rc" || { echo "  FAIL final_of v0.70.0"; fail=1; }
    [ -z "$(final_of v0.70.0-rc.)" ] && echo "  ok   -rc. without N is not an rc" || { echo "  FAIL final_of -rc."; fail=1; }
    [ "$(final_name $rc $final "apr-$rc-aarch64-unknown-linux-gnu-cuda.tar.gz")" = "apr-$final-aarch64-unknown-linux-gnu-cuda.tar.gz" ] \
        && echo "  ok   final_name renames only the version" || { echo "  FAIL final_name"; fail=1; }
    [ -z "$(final_name $rc $final "apr-v0.70.0-rc.30-x.tar.gz")" ] && echo "  ok   rc.30 is not rc.3" || { echo "  FAIL rc.30 renamed as rc.3"; fail=1; }
    [ -z "$(final_name $rc $final "apr-$rc-$rc-x.tar.gz")" ] && echo "  ok   a doubled label is refused" || { echo "  FAIL doubled label"; fail=1; }

    echo "$PROG self-test: stage + readback on real files"
    mkdir -p "$w/good" "$w/good.out"; fixture "$w/good"
    row 0 "tar.gz" "rc -> final: every tarball staged, same bytes" stage $rc $final "$w/good" "$w/good.out"
    mkdir -p "$w/good.out2"; stage $rc $final "$w/good" "$w/good.out2" > "$w/manifest" 2>/dev/null
    row 0 "" "read-back: final bytes and stated hashes equal the rc's" readback "$w/manifest" "$w/good.out2"
    got=$(cmp -s "$w/good/apr-$rc-x86_64-unknown-linux-gnu-cpu.tar.gz" "$w/good.out2/apr-$final-x86_64-unknown-linux-gnu-cpu.tar.gz" && echo same)
    [ "$got" = same ] && echo "  ok   the final tarball is byte-identical to the rc's (cmp)" || { echo "  FAIL bytes differ"; fail=1; }

    mkdir -p "$w/tamper" "$w/tamper.out"; fixture "$w/tamper"
    printf 'x' >> "$w/tamper/pv-$rc-x86_64-unknown-linux-musl.tar.gz"
    row 1 "its .sha256 says" "a tampered rc asset refuses" stage $rc $final "$w/tamper" "$w/tamper.out"

    mkdir -p "$w/nosha" "$w/nosha.out"; fixture "$w/nosha"
    rm -f -- "$w/nosha/pv-$rc-x86_64-unknown-linux-musl.tar.gz.sha256"
    row 1 "has no .sha256" "a missing .sha256 refuses" stage $rc $final "$w/nosha" "$w/nosha.out"

    mkdir -p "$w/orphan" "$w/orphan.out"; fixture "$w/orphan"
    rm -f -- "$w/orphan/pv-$rc-x86_64-unknown-linux-musl.tar.gz"
    row 1 "has no tarball" "a .sha256 without its tarball refuses" stage $rc $final "$w/orphan" "$w/orphan.out"

    mkdir -p "$w/foreign" "$w/foreign.out"; fixture "$w/foreign"
    cp -- "$w/foreign/apr-$rc-x86_64-unknown-linux-gnu-cpu.tar.gz" "$w/foreign/apr-v0.69.9-x86_64-unknown-linux-gnu-cpu.tar.gz"
    row 1 "does not carry" "an asset from another tag refuses" stage $rc $final "$w/foreign" "$w/foreign.out"

    mkdir -p "$w/empty" "$w/empty.out"
    row 1 "nothing to promote" "nothing to promote refuses" stage $rc $final "$w/empty" "$w/empty.out"

    cp -- "$w/good.out2/apr-$final-x86_64-unknown-linux-gnu-cpu.tar.gz" "$w/swap"
    printf 'y' >> "$w/good.out2/apr-$final-x86_64-unknown-linux-gnu-cpu.tar.gz"
    row 1 "MISMATCH apr-" "read-back: an uploaded final that differs from the rc fails" readback "$w/manifest" "$w/good.out2"
    cp -- "$w/swap" "$w/good.out2/apr-$final-x86_64-unknown-linux-gnu-cpu.tar.gz"
    printf '%s  x\n' "$(printf 0%.0s {1..64})" > "$w/good.out2/pv-$final-x86_64-unknown-linux-musl.tar.gz.sha256"
    row 1 "states '0000" "read-back: a final .sha256 stating another hash fails" readback "$w/manifest" "$w/good.out2"
    rm -f -- "$w/good.out2/pv-$final-x86_64-unknown-linux-musl.tar.gz.sha256"
    row 1 "states 'missing'" "read-back: a missing final .sha256 fails" readback "$w/manifest" "$w/good.out2"

    echo "$PROG self-test: the jq reads of the GitHub answers (#4352)"
    prc_io_rows || fail=1
    # MUTANTS of the reads, built from THIS file's I/O section: each must turn a row red
    local m a b why src head tail mut marker=$'\n# ── GitHub I/O'
    src=$(cat -- "${BASH_SOURCE[0]}")
    for m in 1 2 3; do
        case $m in
            1) a='if key("type") == "commit" then'; b='if true then'; why='an annotated rc tag taken as the commit' ;;
            2) a=' // error("no release named \($t)")'; b=''; why='a missing rc release read as empty' ;;
            3) a='draft: true, prerelease: false'; b='draft: false, prerelease: false'; why='the final published before its read-back' ;;
        esac
        head=${src%%"$marker"*}; tail=${src#"$head"}
        mut="$w/mut$m.sh"; printf '%s\n' "$head${tail/"$a"/"$b"}" > "$mut"
        if [ "$head" = "$src" ] || [ "${tail/"$a"/}" = "$tail" ]; then
            echo "  FAIL mutant $m ($why) was not built: its anchor is not in the I/O section"; fail=1
        elif bash -c ". '$mut' --source-only; prc_io_rows" > /dev/null 2>&1; then
            echo "  FAIL mutant $m ($why) survived the rows"; fail=1
        else echo "  ok   mutant $m ($why) is killed"; fi
    done

    if [ "$fail" -eq 0 ]; then echo "$PROG self-test: PASS"; return 0; fi
    echo "$PROG self-test: FAIL"; return 1
}

# ── GitHub I/O ─────────────────────────────────────────────────────────────────

api() { # METHOD PATH [JSON] — prints the body; fails on a non-2xx
    local m=$1 p=$2
    if [ "$#" -ge 3 ]; then
        curl -sSf -X "$m" -H "Authorization: Bearer $TOKEN" -H "Accept: application/vnd.github+json" --data "$3" "$API/$p"
    else
        curl -sSf -X "$m" -H "Authorization: Bearer $TOKEN" -H "Accept: application/vnd.github+json" "$API/$p"
    fi
}
# jq, not python3 (#4352). The reads keep the python semantics the parity table pinned: a
# missing key or a non-object is an error, a JSON null prints "None" as str(None) did, and a
# python-falsy draft is not a draft. iter: python iterates an empty dict or str as nothing.
PRC_JQ='def obj: if type == "object" then . else error("not an object") end;
def key($k): obj | if has($k) then .[$k] else error("missing key \($k)") end;
def pystr: if . == null then "None" elif . == true then "True" elif . == false then "False"
    elif type == "string" then . else tojson end;
def truthy: . != null and . != false and . != 0 and . != "" and . != [] and . != {};
def str: if type == "string" then . else error("not a string") end;
def iter: if . == {} or . == "" then [] elif type == "array" then . else error("not a list") end;'
json() { local f=$1; shift; jq -r "$@" "$PRC_JQ $f"; }   # json FILTER [jq --arg ...]
prc_assets_tsv() { json '[key("assets") | iter | .[] | (key("id") | pystr) + "\t" + (key("name") | str)] | join("\n")'; }
prc_release_named() { json 'first(iter | .[] | select(key("tag_name") == $t)) // error("no release named \($t)")' -c --arg t "$1"; }
prc_draft_ids() { json '[iter | .[] | select((key("draft") | truthy) and key("tag_name") == $t) | key("id") | pystr] | join(" ")' --arg t "$1"; }
prc_ref_commit() { json 'key("object") | if key("type") == "commit" then key("sha") | pystr else "TAG:" + (key("url") | str) end'; }
prc_draft_body() {  # prc_draft_body FINAL COMMIT BODY -> the POST /releases payload
    jq -nc --arg t "$1" --arg c "$2" --arg b "$3" \
        '{tag_name: $t, target_commitish: $c, name: $t, draft: true, prerelease: false, body: $b}'
}
# prc_io_rows -- the reads against fixture answers; one line per row, 1 if any is wrong.
prc_io_rows() {
    local bad=0 got
    r() { if [ "$3" = "$1" ]; then printf '  ok   %s\n' "$2"; else printf '  FAIL %s\n       want: %q\n       got:  %q\n' "$2" "$1" "$3"; bad=1; fi; }
    local list='[{"tag_name":"v0.70.0-rc.2","id":1,"draft":false,"prerelease":true},{"tag_name":"v0.70.0","id":7,"draft":true},{"tag_name":"v0.70.0","id":8,"draft":false},{"tag_name":"v0.70.0","id":9,"draft":true}]'
    r 1 'the rc release is found by name in the listing (drafts included)' "$(printf '%s' "$list" | prc_release_named v0.70.0-rc.2 | jq .id)"
    got=$(printf '%s' "$list" | prc_release_named v0.70.0-rc.9 2>/dev/null; echo "rc=$?")
    r rc=5 'an rc absent from the listing is an error, never an empty release' "$got"
    r '7 9' 'every DRAFT of the final is named, a published one is not' "$(printf '%s' "$list" | prc_draft_ids v0.70.0)"
    r True 'a still-draft rc reads True (promote compares to False and refuses)' "$(printf '%s' '{"draft":true}' | json 'key("draft") | pystr')"
    r abc 'a lightweight tag is its commit' "$(printf '%s' '{"object":{"type":"commit","sha":"abc"}}' | prc_ref_commit)"
    r TAG:https://api/t 'an annotated tag is marked for dereference' \
        "$(printf '%s' '{"object":{"type":"tag","sha":"t","url":"https://api/t"}}' | prc_ref_commit)"
    r $'1\tapr.tar.gz\n2\tapr.tar.gz.sha256' 'the asset listing' \
        "$(printf '%s' '{"assets":[{"id":1,"name":"apr.tar.gz"},{"id":2,"name":"apr.tar.gz.sha256"}]}' | prc_assets_tsv)"
    printf '%s' '{"message":"Not Found"}' | prc_assets_tsv > /dev/null 2>&1; r 5 'an answer with no assets key is an error' "$?"
    got=$(prc_draft_body v0.70.0 abc $'b "q"\n' | jq -c '[.tag_name, .target_commitish, .name, .draft, .prerelease, .body]')
    r '["v0.70.0","abc","v0.70.0",true,false,"b \"q\"\n"]' 'the final is created as a DRAFT at the rc commit' "$got"
    return "$bad"
}
download_assets() { # RELEASE_JSON DIR — fails closed: an unparseable or empty listing is an error
    local id name list
    list=$(printf '%s' "$1" | prc_assets_tsv) || return 1
    [ -n "$list" ] || { echo "$PROG: the release lists no assets" >&2; return 1; }
    while IFS=$'\t' read -r id name; do
        curl -sSfL -H "Authorization: Bearer $TOKEN" -H "Accept: application/octet-stream" \
            -o "$2/$name" "$API/releases/assets/$id" || return 1
    done <<< "$list"
}

promote() {
    local rc=$1 dry=$2 work=$3 final rel commit tag_json rid out f drafts body
    final=$(final_of "$rc")
    [ -n "$final" ] || { echo "$PROG: usage: '$rc' is not vX.Y.Z-rc.N" >&2; exit 2; }
    TOKEN=${GH_TOKEN:-${GITHUB_TOKEN:-}}
    [ -n "$TOKEN" ] || env_die "no GH_TOKEN/GITHUB_TOKEN"
    command -v jq > /dev/null && command -v sha256sum > /dev/null || env_die "needs jq and sha256sum"

    # 1. the rc
    # by listing: an rc is a DRAFT until rc_fleet_stage.sh has it on every fleet host (#4327),
    # and releases/tags/ never returns a draft -- a still-draft rc must be refused by name
    rel=$(api GET "releases?per_page=100" | prc_release_named "$rc") \
        || env_die "cannot read release $rc (not among the newest 100 releases)"
    [ "$(printf '%s' "$rel" | json 'key("draft") | pystr')" = False ] || die "refuse: $rc is still a DRAFT -- it is not on every fleet host yet (scripts/release/rc_fleet_stage.sh $rc --publish, #4327)"
    [ "$(printf '%s' "$rel" | json 'key("prerelease") | pystr')" = True ] || die "refuse: $rc is not a prerelease"
    if api GET "releases/tags/$final" > /dev/null 2>&1; then die "refuse: release $final already exists (a draft from a failed run is deleted by hand, after reading it)"; fi
    drafts=$(api GET "releases?per_page=100" | prc_draft_ids "$final") \
        || env_die "cannot list releases"
    [ -z "$drafts" ] || die "refuse: draft release(s) $drafts of $final already exist: read them and delete them by hand"
    tag_json=$(api GET "git/ref/tags/$rc") || env_die "cannot read tag $rc"
    commit=$(printf '%s' "$tag_json" | prc_ref_commit)
    if [ "${commit#TAG:}" != "$commit" ]; then
        commit=$(curl -sSf -H "Authorization: Bearer $TOKEN" "${commit#TAG:}" | json 'key("object") | key("sha") | pystr') || env_die "cannot dereference $rc"
    fi
    echo "$PROG: $rc is commit $commit"
    bash "$ROOT/scripts/check_release_assets.sh" "$rc" || die "refuse: $rc does not carry every asset"

    # 2 + 3. download, verify, stage
    mkdir -p "$work/rc" "$work/final" "$work/back" || env_die "cannot create $work"
    download_assets "$rel" "$work/rc" || env_die "download from $rc failed"
    stage "$rc" "$final" "$work/rc" "$work/final" > "$work/manifest" || die "refuse: $rc's assets do not verify"
    echo "$PROG: staged $(wc -l < "$work/manifest") tarball(s) under $final names, hashes equal to the rc's"
    if [ "$dry" = 1 ]; then echo "$PROG: --dry-run: nothing written. Staged in $work/final"; return 0; fi

    # 4. draft final at the rc's commit, upload
    body="Promoted from $rc with no rebuild (RC-DOGFOOD-001 §4.4, #4286).

Every asset is the byte-identical rc asset, renamed; each .sha256 states the rc's hash. The tarball's top directory keeps the rc label.

Commit: $commit"
    out=$(api POST releases "$(prc_draft_body "$final" "$commit" "$body")") \
        || die "could not create the draft release $final"
    rid=$(printf '%s' "$out" | json 'key("id") | pystr')
    for f in "$work/final"/*; do
        curl -sSf -X POST -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/octet-stream" \
            --data-binary @"$f" "$UPLOADS/releases/$rid/assets?name=${f##*/}" > /dev/null \
            || die "upload of ${f##*/} failed; draft $rid left as is"
    done

    # 5. read back
    rel=$(api GET "releases/$rid") || die "cannot read the draft back"
    download_assets "$rel" "$work/back" || die "read-back download failed; draft $rid left as is"
    readback "$work/manifest" "$work/back" || die "read-back does not equal the rc's bytes; draft $rid left, nothing deleted"
    echo "$PROG: read-back: every final asset equals the rc's bytes"

    # 6. publish, then the tag must be the rc's commit
    api PATCH "releases/$rid" '{"draft":false,"make_latest":"true"}' > /dev/null || die "publish of draft $rid failed"
    [ "$(api GET "git/ref/tags/$final" | json 'key("object") | key("sha") | pystr')" = "$commit" ] \
        || die "published, but tag $final does not point at $commit: inspect before announcing"
    echo "$PROG: PROMOTED $rc -> $final at $commit"
}

main() {
    local rc='' dry=0 work=''
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --self-test) self_test; return ;;
            --dry-run) dry=1 ;;
            --work) work=${2:-}; shift ;;
            -h|--help) sed -n '2,31p' "${BASH_SOURCE[0]}"; return 0 ;;
            -*) echo "$PROG: unknown flag $1" >&2; return 2 ;;
            *) rc=$1 ;;
        esac
        shift
    done
    [ -n "$rc" ] || { echo "$PROG: usage: promote_rc.sh vX.Y.Z-rc.N [--dry-run] [--work DIR] | --self-test" >&2; return 2; }
    [ -n "$work" ] || work=$(mktemp -d) || env_die "mktemp"
    promote "$rc" "$dry" "$work"
}

# Sourced with --source-only (the self-test's mutants): define the functions, run nothing.
if [ "${1:-}" = --source-only ]; then return 0 2>/dev/null || exit 0; fi
main "$@"
