#!/usr/bin/env bash
# attach_release_installer.sh -- put scripts/install.sh on a published release as the asset
# `install.sh`, so https://github.com/paiml/aprender/releases/latest/download/install.sh resolves
# (#3546; PP-066 R-6 names that URL). Run by the `attach` job of deploy-apr-install.yml on every
# `release: published`, next to the job that publishes the same file to paiml.com/apr/install.sh.
#
#   bash scripts/attach_release_installer.sh v0.71.0            # 0 attached/identical/skipped · 1 refused · 2 ENV
#   bash scripts/attach_release_installer.sh --dry-run v0.71.0  # the same reads, no write
#   bash scripts/attach_release_installer.sh --self-test        # offline case table (stub curl)
#
# INSTALLER_SRC=FILE attaches FILE instead of scripts/install.sh (the job passes the TAG's copy).
#
# RULES
#   - Only a release that is neither a draft nor a prerelease gets the asset: that is exactly the
#     set /releases/latest/download/ serves. An rc MUST NOT carry it: scripts/release/promote_rc.sh
#     downloads every rc asset and refuses one whose name does not carry -<rc>-, so an install.sh
#     on an rc would stop its promotion. The nightly is a prerelease too. A skipped release is rc 0
#     with the reason printed: nothing was owed.
#   - Never overwrite, never delete. A release that already serves install.sh is compared byte for
#     byte with INSTALLER_SRC: identical is rc 0 with nothing uploaded, different is rc 1 with
#     nothing touched. The copy on a published release is the one that shipped.
#   - Read back. After the upload the asset is downloaded by its id and compared; a mismatch, or an
#     upload response naming another file or size, is rc 1.
#   - ENV is rc 2 and never a pass: no curl/jq, an unreadable INSTALLER_SRC, no token for a write,
#     the release unreadable (a draft is invisible to releases/tags/ -- also rc 2, fail closed),
#     or the presence probe answering anything but 200/404.
#
# Presence is probed on the public download URL, not the release's `assets` list, so the answer
# does not depend on how many other assets the release carries or on any page size.
#
# No python3 (C301): curl + jq, as check_release_assets.sh reads releases since #4352.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROG=attach_release_installer
NAME=install.sh
SRC="${INSTALLER_SRC:-$ROOT/scripts/install.sh}"
REPO="${GITHUB_REPOSITORY:-paiml/aprender}"
TOKEN="${GITHUB_TOKEN:-${GH_TOKEN:-}}"
API="https://api.github.com/repos/$REPO"
UPLOADS="https://uploads.github.com/repos/$REPO"
DOWNLOAD="https://github.com/$REPO/releases/download"

env_fail() { printf '%s: ENV -- %s (this is NOT a pass)\n' "$PROG" "$1" >&2; return 2; }
refuse() { printf '%s: REFUSED -- %s\n' "$PROG" "$1" >&2; return 1; }

api_get() { # api_get URL -> body; a non-2xx is curl's own non-zero rc
    if [ -n "$TOKEN" ]; then
        curl -sSf -H "Authorization: Bearer $TOKEN" -H "Accept: application/vnd.github+json" "$1"
    else
        curl -sSf -H "Accept: application/vnd.github+json" "$1"
    fi
}

attach() { # attach TAG DRY(0|1)
    local tag="$1" dry="$2" work rel meta rid draft pre code up size aid
    command -v curl > /dev/null 2>&1 || { env_fail "curl is not on PATH"; return; }
    command -v jq > /dev/null 2>&1 || { env_fail "jq is not on PATH"; return; }
    [ -r "$SRC" ] && [ -s "$SRC" ] || { env_fail "installer $SRC is missing or empty"; return; }
    work=$(mktemp -d) || { env_fail "mktemp failed"; return; }
    # shellcheck disable=SC2064
    trap "rm -rf '$work'" RETURN

    rel=$(api_get "$API/releases/tags/$tag") \
        || { env_fail "no published release $tag could be read (a draft is not served by /releases/latest and is never attached)"; return; }
    meta=$(printf '%s' "$rel" | jq -er --arg t "$tag" \
        'if .tag_name != $t then error("tag") else "\(.id)\t\(.draft)\t\(.prerelease)" end' 2> /dev/null) \
        || { env_fail "the release payload for $tag did not parse, or names another tag"; return; }
    IFS=$'\t' read -r rid draft pre <<< "$meta"
    case "$rid" in '' | *[!0-9]*) env_fail "release id '$rid' is not a number"; return ;; esac
    if [ "$draft" != false ] || [ "$pre" != false ]; then
        printf '%s: not attached -- %s is draft=%s prerelease=%s; /releases/latest never serves it, and promote_rc refuses an rc asset without -<rc>- in its name\n' \
            "$PROG" "$tag" "$draft" "$pre"
        return 0
    fi

    code=$(curl -sSL -o "$work/present" -w '%{http_code}' "$DOWNLOAD/$tag/$NAME") \
        || { env_fail "the presence probe of $DOWNLOAD/$tag/$NAME failed"; return; }
    case "$code" in
        200)
            if cmp -s -- "$SRC" "$work/present"; then
                printf '%s: %s already serves %s, byte-identical to %s; nothing uploaded\n' "$PROG" "$tag" "$NAME" "${SRC#"$ROOT"/}"
                return 0
            fi
            refuse "$tag already serves a $NAME that differs from ${SRC#"$ROOT"/}; nothing deleted, nothing uploaded"
            return ;;
        404) ;;
        *) env_fail "the presence probe of $DOWNLOAD/$tag/$NAME answered HTTP $code"; return ;;
    esac

    if [ "$dry" = 1 ]; then
        printf '%s: dry run -- %s does not serve %s; would attach %s to release %s\n' "$PROG" "$tag" "$NAME" "${SRC#"$ROOT"/}" "$rid"
        return 0
    fi
    [ -n "$TOKEN" ] || { env_fail "no GITHUB_TOKEN/GH_TOKEN to upload with"; return; }

    up=$(curl -sSf -X POST -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/octet-stream" \
        --data-binary "@$SRC" "$UPLOADS/releases/$rid/assets?name=$NAME") \
        || { refuse "the upload of $NAME to $tag failed"; return; }
    size=$(wc -c < "$SRC")
    size=$((size + 0))
    aid=$(printf '%s' "$up" | jq -er --arg n "$NAME" --argjson s "$size" \
        'if .name == $n and .size == $s then .id else error("mismatch") end' 2> /dev/null) \
        || { refuse "the upload response does not name $NAME at $size bytes"; return; }
    case "$aid" in '' | *[!0-9]*) refuse "asset id '$aid' is not a number"; return ;; esac
    curl -sSfL -H "Authorization: Bearer $TOKEN" -H "Accept: application/octet-stream" \
        -o "$work/readback" "$API/releases/assets/$aid" \
        || { refuse "the read-back of asset $aid failed"; return; }
    cmp -s -- "$SRC" "$work/readback" \
        || { refuse "MISMATCH: asset $aid on $tag differs from ${SRC#"$ROOT"/} after upload"; return; }
    printf '%s: attached %s to %s (asset %s, %s bytes), read back byte-identical\n' "$PROG" "$NAME" "$tag" "$aid" "$size"
    return 0
}

# --self-test: a stub `curl` first on PATH serves fixtures from $STUB and logs every call, so
# each row checks the verdict AND which writes happened. No network.
self_test() {
    local work n=0 red=0
    work=$(mktemp -d) || return 2
    # shellcheck disable=SC2064
    trap "rm -rf '$work'" RETURN
    mkdir -p "$work/bin"
    cat > "$work/bin/curl" << 'STUBEOF'
#!/usr/bin/env bash
out="" method=GET data="" url=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        -o) out="$2"; shift ;;
        -X) method="$2"; shift ;;
        --data-binary) data="${2#@}"; shift ;;
        -H | -w) shift ;;
        http*) url="$1" ;;
    esac
    shift
done
printf '%s %s\n' "$method" "$url" >> "$STUB/calls"
emit() { if [ -n "$out" ]; then cat -- "$1" > "$out"; else cat -- "$1"; fi; }
case "$method $url" in
    "GET "*/releases/tags/*)
        [ -f "$STUB/release.json" ] || exit 22
        emit "$STUB/release.json" ;;
    "GET "*/releases/download/*)
        [ -f "$STUB/probe.code" ] || exit 6
        code=$(cat "$STUB/probe.code")
        if [ "$code" = 200 ]; then cat -- "$STUB/present" > "$out"; fi
        printf '%s' "$code"
        if [ -f "$STUB/probe.exit" ]; then exit "$(cat "$STUB/probe.exit")"; fi ;;
    "POST "*)
        [ ! -f "$STUB/upload.fail" ] || exit 22
        cp -- "$data" "$STUB/uploaded"
        if [ -f "$STUB/upload.json" ]; then emit "$STUB/upload.json"
        else printf '{"id":99,"name":"install.sh","size":%s}' "$(wc -c < "$data")"; fi ;;
    "GET "*/releases/assets/*)
        if [ -f "$STUB/readback" ]; then emit "$STUB/readback"; else emit "$STUB/uploaded"; fi ;;
    *) exit 22 ;;
esac
STUBEOF
    chmod +x "$work/bin/curl"
    printf '#!/bin/sh\necho installer v1\n' > "$work/install.sh"
    printf '#!/bin/sh\necho installer v0\n' > "$work/other.sh"
    local final='{"id":42,"tag_name":"v9.9.9","draft":false,"prerelease":false,"assets":[]}'

    # stage DIR [release-json] [probe-code] -- a fresh fixture directory
    stage() {
        rm -rf "${1:?}"; mkdir -p "$1"; : > "$1/calls"
        if [ -n "${2:-}" ]; then printf '%s' "$2" > "$1/release.json"; fi
        if [ -n "${3:-}" ]; then printf '%s' "$3" > "$1/probe.code"; fi
    }
    # row WANT-RC WANT-POSTS LABEL DIR [args...] -- run the script against DIR's fixtures
    row() {
        local want=$1 posts=$2 label=$3 dir=$4 rc=0 got
        shift 4
        n=$((n + 1))
        STUB="$dir" PATH="$work/bin:$PATH" INSTALLER_SRC="$work/install.sh" GITHUB_TOKEN="${ROW_AUTH-t}" GH_TOKEN="" \
            bash "$0" "$@" > "$dir/out" 2>&1 || rc=$?
        got=$(grep -c '^POST ' "$dir/calls")
        if [ "$rc" = "$want" ] && [ "$got" = "$posts" ] && ! grep -q '^DELETE ' "$dir/calls"; then
            printf 'ok    row %-2s rc=%s posts=%s  %s\n' "$n" "$rc" "$got" "$label"
        else
            printf 'FAIL  row %-2s rc=%s (want %s) posts=%s (want %s)  %s\n' "$n" "$rc" "$want" "$got" "$posts" "$label"
            sed 's/^/        /' "$dir/out"
            red=$((red + 1))
        fi
    }
    local d="$work/s"

    stage "$d" "$final" 404
    row 0 1 "a final without install.sh gets it, read back identical" "$d" v9.9.9
    cmp -s "$work/install.sh" "$d/uploaded" || { echo "FAIL  the uploaded bytes are not INSTALLER_SRC"; red=$((red + 1)); }

    stage "$d" "$final" 200; cp "$work/install.sh" "$d/present"
    row 0 0 "a final already serving the identical file: nothing uploaded" "$d" v9.9.9

    stage "$d" "$final" 200; cp "$work/other.sh" "$d/present"
    row 1 0 "a final serving a DIFFERENT install.sh refuses, never overwrites" "$d" v9.9.9

    stage "$d" '{"id":42,"tag_name":"v9.9.9-rc.1","draft":false,"prerelease":true,"assets":[]}' 404
    row 0 0 "a prerelease (an rc, the nightly) is never attached" "$d" v9.9.9-rc.1
    grep -q '^GET .*/releases/download/' "$d/calls" && { echo "FAIL  a prerelease was probed"; red=$((red + 1)); }

    stage "$d" '{"id":42,"tag_name":"v9.9.9","draft":true,"prerelease":false,"assets":[]}' 404
    row 0 0 "a draft payload is never attached" "$d" v9.9.9

    stage "$d" "" 404
    row 2 0 "no readable release (absent, or a draft hidden by releases/tags/) is ENV" "$d" v9.9.9

    stage "$d" 'not json' 404
    row 2 0 "an unparseable release payload is ENV" "$d" v9.9.9

    stage "$d" '{"id":42,"tag_name":"v9.9.8","draft":false,"prerelease":false}' 404
    row 2 0 "a payload naming another tag is ENV" "$d" v9.9.9

    stage "$d" "$final" 500
    row 2 0 "a presence probe answering 500 is ENV, not absent" "$d" v9.9.9

    stage "$d" "$final"
    row 2 0 "a presence probe that cannot connect is ENV" "$d" v9.9.9

    stage "$d" "$final" 200; printf '#!/bin/sh\n' > "$d/present"; echo 18 > "$d/probe.exit"
    row 2 0 "a 200 cut off mid-transfer is ENV, not a differing file" "$d" v9.9.9

    stage "$d" "$final" 404; printf 'tampered\n' > "$d/readback"
    row 1 1 "a read-back that differs from the source refuses" "$d" v9.9.9

    stage "$d" "$final" 404; printf '{"id":99,"name":"install.sh.1","size":1}' > "$d/upload.json"
    row 1 1 "an upload response naming another file or size refuses" "$d" v9.9.9

    stage "$d" "$final" 404; : > "$d/upload.fail"
    row 1 1 "a failed upload refuses" "$d" v9.9.9

    stage "$d" "$final" 404
    ROW_AUTH='' row 2 0 "no token: nothing is uploaded, ENV" "$d" v9.9.9

    stage "$d" "$final" 404
    row 0 0 "--dry-run reads and writes nothing" "$d" --dry-run v9.9.9

    if grep -vE '^[[:space:]]*#' "$0" | grep -qE '(-X|--request)[[:space:]]*"?(DELETE|PATCH|PUT)'; then
        echo "FAIL  the script spells a DELETE/PATCH/PUT: it must only ever add an asset"; red=$((red + 1))
    fi

    printf '%s: %s rows, %s red\n' "$PROG" "$n" "$red"
    [ "$red" -eq 0 ] && [ "$n" -gt 0 ]
}

case "${1:-}" in
    --self-test) self_test; exit $? ;;
    --dry-run) [ -n "${2:-}" ] || { echo "usage: $0 [--dry-run] TAG | --self-test" >&2; exit 2; }; attach "$2" 1; exit $? ;;
    '' | -*) echo "usage: $0 [--dry-run] TAG | --self-test" >&2; exit 2 ;;
    *) attach "$1" 0; exit $? ;;
esac
