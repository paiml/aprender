#!/usr/bin/env bash
# archive_fetch.sh -- LAN cache for the FLOW-010 P1 nextest archive (ws-archive). Transport only.
#
# WHY. Every workspace-test-shard downloaded the ~541 MB ws-archive artifact from GitHub's blob store. Run 39595316228:
# kfocus shard 9 spent 14:19-14:43Z on it (0.4 MB/s, load 0.03) while intel shards finished in 6-7 min.
#
#   publish ROOT PORT KEY OUTDIR PTRDIR   archive job: copy OUTDIR/{ws.tar.zst,bind.json} to ROOT/KEY/ (atomic rename), write
#                                         SHA256SUM there and PTRDIR/ptr.json {cache_url,sha256,key}. ROOT empty = disabled:
#                                         ptr.json carries cache_url "" so every shard falls back. Always rc 0 on a disabled cache.
#   fetch PTR.json DEST                   shard: curl the files from cache_url, verify sha256 of ws.tar.zst against the POINTER's
#                                         sha256 (which travels as a GitHub artifact, not from the cache host). Prints
#                                         `hit=1` / `hit=0 reason=...` and appends it to $GITHUB_OUTPUT. Always rc 0: a miss means
#                                         "download the GitHub artifact", the same bytes.
#
# NOT a second verdict path: archive_bind_check.sh still runs on DEST whichever way the bytes arrived (sha, expr, flags,
# toolchain), so a stale or wrong cache file is RED exactly as before. No continue-on-error.
# Usage:  bash scripts/ci/archive_fetch.sh publish|fetch ... | --self-test
set -euo pipefail

emit() { # hit=1|hit=0 reason=...
    printf 'ws-archive %s\n' "$*"
    case "$1" in hit=1) h=1 ;; *) h=0 ;; esac
    [ -z "${GITHUB_OUTPUT:-}" ] || printf 'hit=%s\n' "$h" >> "$GITHUB_OUTPUT"
}

publish() {
    local root="$1" port="$2" key="$3" out="$4" ptr="$5" tmp sum host
    mkdir -p "$ptr"
    if [ -z "$root" ]; then
        jq -n '{cache_url:"",sha256:"",key:""}' > "$ptr/ptr.json"; echo "ws-archive publish: WS_ARCHIVE_CACHE_ROOT unset, cache disabled"; return 0
    fi
    mkdir -p "$root"
    tmp="$root/.tmp-$key-$$"; rm -rf "${tmp:?}"; mkdir -p "$tmp"
    cp "$out/ws.tar.zst" "$out/bind.json" "$tmp/"
    sum="$(sha256sum "$tmp/ws.tar.zst" | cut -d' ' -f1)"
    printf '%s  ws.tar.zst\n' "$sum" > "$tmp/SHA256SUM"
    rm -rf "${root:?}/$key"; mv "$tmp" "$root/$key"
    host="$(hostname -f 2> /dev/null || hostname)"
    jq -n --arg u "http://$host:$port/$key" --arg s "$sum" --arg k "$key" '{cache_url:$u,sha256:$s,key:$k}' > "$ptr/ptr.json"
    echo "ws-archive publish: $root/$key sha256=$sum"
}

fetch() {
    local ptr="$1" dest="$2" url sum got t0
    [ -f "$ptr" ] || { emit hit=0 reason=no-pointer; return 0; }
    url="$(jq -r '.cache_url // ""' "$ptr")"; sum="$(jq -r '.sha256 // ""' "$ptr")"
    [ -n "$url" ] && [ -n "$sum" ] || { emit hit=0 reason=cache-disabled; return 0; }
    mkdir -p "$dest"; t0="$(date +%s)"
    if ! curl -fsS --connect-timeout 5 --max-time "${WS_ARCHIVE_FETCH_MAX_S:-300}" -o "$dest/ws.tar.zst" "$url/ws.tar.zst" 2> "$dest/.curl.err" \
        || ! curl -fsS --connect-timeout 5 --max-time 60 -o "$dest/bind.json" "$url/bind.json" 2>> "$dest/.curl.err"; then
        emit hit=0 "reason=unreachable url=$url"; rm -f "$dest/ws.tar.zst" "$dest/bind.json" "$dest/.curl.err"; return 0
    fi
    rm -f "$dest/.curl.err"
    if [ "${AF_MUTANT:-}" != nosum ]; then # test-only mutant: skip the checksum
        got="$(sha256sum "$dest/ws.tar.zst" | cut -d' ' -f1)"
        if [ "$got" != "$sum" ]; then emit hit=0 "reason=sha256-mismatch url=$url"; rm -f "$dest/ws.tar.zst" "$dest/bind.json"; return 0; fi
    fi
    emit hit=1 "source=lan url=$url seconds=$(( $(date +%s) - t0 ))"
}

self_test() {
    local td fails=0 out
    td="$(mktemp -d)"; trap 'rm -rf "${td:?}"' RETURN
    mkdir -p "$td/out"; head -c 100000 /dev/urandom > "$td/out/ws.tar.zst"; echo '{"tier":"quick"}' > "$td/out/bind.json"
    want() { # LABEL EXPECT(hit=1|hit=0) -- reads `fetch` output on stdin
        local o; o="$(cat)"
        if printf '%s' "$o" | /usr/bin/grep -q "^ws-archive $2"; then echo "ok    $1"; else echo "FAIL  $1: wanted $2, got: $o"; fails=1; fi
    }
    publish "$td/cache" 8080 K1 "$td/out" "$td/ptr" > /dev/null
    [ "$(jq -r .sha256 "$td/ptr/ptr.json")" = "$(sha256sum "$td/out/ws.tar.zst" | cut -d' ' -f1)" ] && echo "ok    publish writes the archive's sha256 into the pointer" || { echo "FAIL  pointer sha256"; fails=1; }
    # file:// stands in for http://host:port; curl treats both the same way
    jq --arg u "file://$td/cache/K1" '.cache_url=$u' "$td/ptr/ptr.json" > "$td/ptr/good.json"
    fetch "$td/ptr/good.json" "$td/d1" | want "hit: intact cache file verifies" "hit=1"
    cmp -s "$td/d1/ws.tar.zst" "$td/out/ws.tar.zst" && echo "ok    hit: bytes identical to the archive" || { echo "FAIL  hit bytes differ"; fails=1; }
    printf 'X' | dd of="$td/cache/K1/ws.tar.zst" bs=1 seek=500 conv=notrunc 2> /dev/null
    fetch "$td/ptr/good.json" "$td/d2" | want "miss: one corrupted byte is a sha256 mismatch" "hit=0 reason=sha256-mismatch"
    [ ! -e "$td/d2/ws.tar.zst" ] && echo "ok    miss: the corrupt file is removed" || { echo "FAIL  corrupt file kept"; fails=1; }
    jq '.cache_url="http://127.0.0.1:1/x"' "$td/ptr/good.json" > "$td/ptr/dead.json"
    fetch "$td/ptr/dead.json" "$td/d3" | want "miss: unreachable host" "hit=0 reason=unreachable"
    publish "$td/cache" 8080 K1 "$td/out" "$td/ptr" > /dev/null; jq --arg u "file://$td/cache/K1" '.cache_url=$u|.sha256="00"' "$td/ptr/ptr.json" > "$td/ptr/wrong.json"
    fetch "$td/ptr/wrong.json" "$td/d4" | want "miss: pointer sha256 differs from the cache file" "hit=0 reason=sha256-mismatch"
    publish "" 8080 K1 "$td/out" "$td/ptr0" > /dev/null
    fetch "$td/ptr0/ptr.json" "$td/d5" | want "miss: cache disabled (var unset)" "hit=0 reason=cache-disabled"
    fetch "$td/nope.json" "$td/d6" | want "miss: no pointer" "hit=0 reason=no-pointer"
    # MUTANT: with the checksum skipped, the corrupt case must NOT be a clean miss -> this self-test goes RED
    if [ -z "${AF_MUTANT:-}" ]; then
        if AF_MUTANT=nosum bash "$0" --self-test > /dev/null 2>&1; then echo "FAIL  mutant: --self-test stayed green with the checksum skipped"; fails=1; else echo "ok    red   MUTANT: skipping the checksum turns --self-test RED"; fi
    fi
    return "$fails"
}

case "${1:-}" in
    publish) [ "$#" -eq 6 ] || { echo "usage: $0 publish ROOT PORT KEY OUTDIR PTRDIR" >&2; exit 2; }; publish "$2" "$3" "$4" "$5" "$6" ;;
    fetch) [ "$#" -eq 3 ] || { echo "usage: $0 fetch PTR.json DEST" >&2; exit 2; }; fetch "$2" "$3" ;;
    --self-test) self_test; exit $? ;;
    *) echo "usage: $0 publish|fetch|--self-test" >&2; exit 2 ;;
esac
