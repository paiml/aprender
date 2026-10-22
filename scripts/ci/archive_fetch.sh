#!/usr/bin/env bash
# archive_fetch.sh -- LAN cache for the FLOW-010 P1 nextest archive (ws-archive). Transport only.
#
# WHY. Every workspace-test-shard downloaded the ~541 MB ws-archive artifact from GitHub's blob store. Run 39595316228:
# kfocus shard 9 spent 14:19-14:43Z on it (0.4 MB/s, load 0.03) while intel shards finished in 6-7 min.
#
#   publish ROOT PORT KEY OUTDIR PTRDIR   archive job: copy EVERY file of OUTDIR (ws.tar.zst, bind.json, tree.N.list.json, ...) to
#                                         ROOT/KEY/ (atomic rename) and write PTRDIR/ptr.json {cache_url,key,files:{name:sha256}}.
#                                         Disabled (ROOT empty, or OUTDIR has no ws.tar.zst = a non-quick tier) writes a pointer
#                                         with cache_url "" and rc 0, so every shard falls back.
#   fetch PTR.json DEST                   shard: curl every listed file from cache_url and verify EACH sha256 against the POINTER
#                                         (a GitHub artifact, not the cache host). Prints `ws-archive hit=1 ...` / `hit=0 reason=...`
#                                         and appends hit=N to $GITHUB_OUTPUT. A miss (including a malformed pointer) means
#                                         "download the GitHub artifact", the same bytes; rc 0.
#
# NOT a second verdict path: archive_bind_check.sh still runs on DEST whichever way the bytes arrived, so a stale or wrong cache
# file is RED exactly as before. No continue-on-error.
# Usage:  bash scripts/ci/archive_fetch.sh publish|fetch ... | --self-test
set -euo pipefail

emit() { # hit=1|hit=0 reason=...
    local h=0
    printf 'ws-archive %s\n' "$*"
    case "$1" in hit=1) h=1 ;; *) h=0 ;; esac
    [ -z "${GITHUB_OUTPUT:-}" ] || printf 'hit=%s\n' "$h" >> "$GITHUB_OUTPUT"
}

disabled() { # PTRDIR WHY
    mkdir -p "$1"
    jq -n '{cache_url:"",key:"",files:{}}' > "$1/ptr.json"
    echo "ws-archive publish: cache disabled ($2)"
}

publish() {
    local root="$1" port="$2" key="$3" out="$4" ptr="$5" tmp host f b files='{}'
    if [ -z "$root" ]; then disabled "$ptr" "WS_ARCHIVE_CACHE_ROOT unset"; return 0; fi
    if [ "${AF_MUTANT:-}" != notier ] && [ ! -f "$out/ws.tar.zst" ]; then disabled "$ptr" "no ws.tar.zst: not the quick tier"; return 0; fi
    case "$key" in "" | .* | */* | *[!A-Za-z0-9._-]*) echo "archive_fetch: refusing key '$key'" >&2; return 2 ;; esac
    mkdir -p "$root" "$ptr"
    tmp="$root/.tmp-$key-$$"
    rm -rf "${tmp:?}"
    mkdir -p "$tmp"
    for f in "$out"/*; do
        [ -f "$f" ] || continue
        b="$(basename "$f")"
        if [ "${AF_MUTANT:-}" = notree ]; then case "$b" in tree.*) continue ;; esac; fi
        cp "$f" "$tmp/"
        files="$(printf '%s' "$files" | jq --arg n "$b" --arg s "$(sha256sum "$f" | cut -d' ' -f1)" '.[$n]=$s')"
    done
    rm -rf "${root:?}/${key:?}"
    mv "$tmp" "$root/$key"
    host="$(hostname -f 2> /dev/null || hostname)"
    jq -n --arg u "http://$host:$port/$key" --arg k "$key" --argjson f "$files" '{cache_url:$u,key:$k,files:$f}' > "$ptr/ptr.json"
    echo "ws-archive publish: $root/$key ($(printf '%s' "$files" | jq 'length') files)"
}

clean() { # DEST NAMES
    local n
    for n in $2; do rm -f "${1:?}/${n:?}"; done
}

fetch() {
    local ptr="$1" dest="$2" url names n got want t0
    [ -f "$ptr" ] || { emit hit=0 reason=no-pointer; return 0; }
    url="$(jq -r '.cache_url // ""' "$ptr" 2> /dev/null)" || { emit hit=0 reason=malformed-pointer; return 0; }
    names="$(jq -r '(.files // {}) | keys[]' "$ptr" 2> /dev/null)" || { emit hit=0 reason=malformed-pointer; return 0; }
    if [ -z "$url" ] || [ -z "$names" ]; then emit hit=0 reason=cache-disabled; return 0; fi
    mkdir -p "$dest"
    t0="$(date +%s)"
    for n in $names; do
        case "$n" in */* | .*) emit hit=0 reason=bad-name; clean "$dest" "$names"; return 0 ;; esac
        if ! curl -fsS --connect-timeout 5 --max-time "${WS_ARCHIVE_FETCH_MAX_S:-300}" -o "$dest/$n" "$url/$n" 2> /dev/null; then
            emit hit=0 "reason=unreachable url=$url"; clean "$dest" "$names"; return 0
        fi
        if [ "${AF_MUTANT:-}" != nosum ]; then # test-only mutant: skip the checksum
            got="$(sha256sum "$dest/$n" | cut -d' ' -f1)"
            want="$(jq -r --arg n "$n" '.files[$n]' "$ptr")"
            if [ "$got" != "$want" ]; then emit hit=0 "reason=sha256-mismatch file=$n url=$url"; clean "$dest" "$names"; return 0; fi
        fi
    done
    emit hit=1 "source=lan url=$url files=$(printf '%s\n' "$names" | wc -l) seconds=$(( $(date +%s) - t0 ))"
}

# The pointer must actually travel from the archive job to the shards (B4). Prints FAIL lines; rc 1 when any.
wiring() { # CI_YML SECTIONS_YML
    local ci="$1" sec="$2" bad=0
    # fat_driver stages section uploads under fat/artifacts/<name>/; only a real ci.yml upload step reaches other jobs
    if ! /usr/bin/grep -q 'fat/artifacts/ws-archive-ptr/' "$ci"; then echo "FAIL  ci.yml never uploads fat/artifacts/ws-archive-ptr/"; bad=1; fi
    # without merge-multiple the pointer lands at ws-archive-ptr/ws-archive-ptr/ptr.json and every fetch is a no-pointer miss
    if ! awk '/name: ws-archive-ptr/{f=1} f&&/merge-multiple: true/{ok=1} END{exit !ok}' "$sec"; then echo "FAIL  the shard's ws-archive-ptr download lacks merge-multiple: true"; bad=1; fi
    return "$bad"
}

self_test() {
    local td fails=0 root m
    td="$(mktemp -d)"
    trap 'rm -rf "${td:?}"' RETURN
    mkdir -p "$td/out"
    head -c 100000 /dev/urandom > "$td/out/ws.tar.zst"
    echo '{"tier":"quick"}' > "$td/out/bind.json"
    echo '{"n":1}' > "$td/out/tree.3.list.json"
    # B1: want reads a FILE, so fails is set in this shell, never in a pipe's subshell
    want() { # LABEL EXPECT OUTPUT-FILE
        if /usr/bin/grep -q "^ws-archive $2" "$3"; then echo "ok    $1"; else echo "FAIL  $1: wanted '$2', got: $(cat "$3")"; fails=1; fi
    }
    run_fetch() { fetch "$1" "$2" > "$td/fetch.out"; }
    publish "$td/cache" 8080 K1 "$td/out" "$td/ptr" > /dev/null
    if [ "$(jq -r '.files | keys | join(",")' "$td/ptr/ptr.json")" = "bind.json,tree.3.list.json,ws.tar.zst" ]; then
        echo "ok    publish lists every file of the archive dir (incl. tree.N.list.json)"
    else echo "FAIL  pointer files: $(cat "$td/ptr/ptr.json")"; fails=1; fi
    # file:// stands in for http://host:port; curl treats both the same way
    jq --arg u "file://$td/cache/K1" '.cache_url=$u' "$td/ptr/ptr.json" > "$td/ptr/good.json"
    run_fetch "$td/ptr/good.json" "$td/d1"
    want "hit: intact cache verifies" "hit=1" "$td/fetch.out"
    if cmp -s "$td/d1/ws.tar.zst" "$td/out/ws.tar.zst" && [ -f "$td/d1/tree.3.list.json" ] && [ -f "$td/d1/bind.json" ]; then
        echo "ok    hit: archive bytes identical, bind.json and tree.3.list.json present"
    else echo "FAIL  hit: files differ or absent"; fails=1; fi
    printf 'X' | dd of="$td/cache/K1/ws.tar.zst" bs=1 seek=500 conv=notrunc 2> /dev/null
    run_fetch "$td/ptr/good.json" "$td/d2"
    want "miss: one corrupted byte is a sha256 mismatch" "hit=0 reason=sha256-mismatch" "$td/fetch.out"
    if [ ! -e "$td/d2/ws.tar.zst" ] && [ ! -e "$td/d2/bind.json" ]; then echo "ok    miss: every fetched file is removed"; else echo "FAIL  files kept after a miss"; fails=1; fi
    jq '.cache_url="http://127.0.0.1:1/x"' "$td/ptr/good.json" > "$td/ptr/dead.json"
    run_fetch "$td/ptr/dead.json" "$td/d3"
    want "miss: unreachable host" "hit=0 reason=unreachable" "$td/fetch.out"
    echo '{not json' > "$td/ptr/bad.json"
    run_fetch "$td/ptr/bad.json" "$td/d4"
    want "miss: malformed pointer (rc 0, not a crash)" "hit=0 reason=malformed-pointer" "$td/fetch.out"
    publish "" 8080 K1 "$td/out" "$td/ptr0" > /dev/null
    run_fetch "$td/ptr0/ptr.json" "$td/d5"
    want "miss: cache disabled (var unset)" "hit=0 reason=cache-disabled" "$td/fetch.out"
    run_fetch "$td/nope.json" "$td/d6"
    want "miss: no pointer" "hit=0 reason=no-pointer" "$td/fetch.out"
    # B2: a non-quick tier leaves only bind.json in OUTDIR -> a disabled pointer, rc 0, no cache dir
    mkdir -p "$td/nonquick"
    echo '{"tier":"full"}' > "$td/nonquick/bind.json"
    if publish "$td/cache2" 8080 K2 "$td/nonquick" "$td/ptr2" > /dev/null 2>&1 && [ "$(jq -r .cache_url "$td/ptr2/ptr.json")" = "" ] && [ ! -d "$td/cache2/K2" ]; then
        echo "ok    publish on a non-quick tier is a disabled pointer, rc 0"
    else echo "FAIL  non-quick tier publish"; fails=1; fi
    # key guard: an empty key must never reach rm -rf on the cache root
    if publish "$td/cache3" 8080 "" "$td/out" "$td/ptr3" > /dev/null 2>&1; then echo "FAIL  empty key accepted"; fails=1; else echo "ok    publish refuses an empty key"; fi
    # B4: wiring against the real files, then each wiring mutant
    root="$(git rev-parse --show-toplevel)"
    if m="$(wiring "$root/.github/workflows/ci.yml" "$root/ci/sections.yml")"; then
        echo "ok    wiring: the pointer is uploaded and downloaded merge-multiple"
    else echo "FAIL  wiring as committed: $m"; fails=1; fi
    sed 's#fat/artifacts/ws-archive-ptr/#fat/artifacts/ws-archive/#' "$root/.github/workflows/ci.yml" > "$td/ci.mut"
    if wiring "$td/ci.mut" "$root/ci/sections.yml" > /dev/null; then echo "FAIL  wiring mutant (no ptr upload) stayed GREEN"; fails=1; else echo "ok    red   wiring MUTANT: ci.yml without the pointer upload"; fi
    sed '/merge-multiple: true/d' "$root/ci/sections.yml" > "$td/sec.mut"
    if wiring "$root/.github/workflows/ci.yml" "$td/sec.mut" > /dev/null; then echo "FAIL  wiring mutant (no merge-multiple) stayed GREEN"; fails=1; else echo "ok    red   wiring MUTANT: pointer download without merge-multiple"; fi
    # each behaviour mutant must turn this self-test RED (nosum: B-sum, notier: B2, notree: B3)
    if [ -z "${AF_MUTANT:-}" ]; then
        for m in nosum notier notree; do
            if AF_MUTANT=$m bash "$0" --self-test > /dev/null 2>&1; then echo "FAIL  MUTANT $m: --self-test stayed green"; fails=1; else echo "ok    red   MUTANT $m turns --self-test RED"; fi
        done
        # B1: a deliberately failing want must exit non-zero (the old pipe form lost it)
        if AF_MUTANT=wantfail bash "$0" --self-test > /dev/null 2>&1; then echo "FAIL  MUTANT wantfail: a failing want exited 0"; fails=1; else echo "ok    red   MUTANT wantfail: a failing want exits non-zero"; fi
    elif [ "$AF_MUTANT" = wantfail ]; then
        echo "{}" > "$td/w.out"
        want "forced failure" "hit=1" "$td/w.out"
    fi
    return "$fails"
}

case "${1:-}" in
    publish) [ "$#" -eq 6 ] || { echo "usage: $0 publish ROOT PORT KEY OUTDIR PTRDIR" >&2; exit 2; }; publish "$2" "$3" "$4" "$5" "$6" ;;
    fetch) [ "$#" -eq 3 ] || { echo "usage: $0 fetch PTR.json DEST" >&2; exit 2; }; fetch "$2" "$3" ;;
    --self-test) self_test; exit $? ;;
    *) echo "usage: $0 publish|fetch|--self-test" >&2; exit 2 ;;
esac
