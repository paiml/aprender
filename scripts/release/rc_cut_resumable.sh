#!/usr/bin/env bash
# rc_cut_resumable.sh -- one command cuts an rc end to end, and the SAME command finishes a cut a
# killed session left half done (#4327, RESUMABLE-CUT)
#
#   rc_cut_resumable.sh --tag vX.Y.Z-rc.N --sha <40-hex>    tag -> assets -> every host -> publish
#   rc_cut_resumable.sh --self-test                         the kill falsifier (fake GitHub + fleet)
#
# Steps, each receipted in the cut's ledger (lib_cut_receipt.sh) BEFORE it acts:
#   1. rc_tag_main.sh      tag, DRAFT, dispatch binary-release.yml (a tag this cut wrote is resumed, never re-cut)
#   2. wait (bounded) until check_release_assets.sh reports the full asset set on the draft
#   3. rc_fleet_stage.sh --publish   every host installs + verifies, then the draft is published
#      (a host already done is re-verified only; an asset already on a host at its sha256 is not copied)
#
# A session killed anywhere is finished by a NEW session running the same command: 0 manual steps.
# The ledger lives outside any worktree (${XDG_STATE_HOME:-~/.local/state}/rc-cut/<tag>.tsv, or
# $RC_CUT_LEDGER), so the new session finds it from any checkout on the cutting box (lambda).
#
#   RC_CUT_ASSET_WAIT=<s>   how long step 2 waits for binary-release.yml (default 5400)
#
# EXIT 0 published on every host · 1 refused or held (the tagger/stager says why) · 2 usage/ENV,
#      or the assets never completed within RC_CUT_ASSET_WAIT.
set -uo pipefail
PROG=rc_cut_resumable
HERE=${RC_CUT_HERE:-"$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"}
. "$HERE/lib_cut_receipt.sh" || { echo "$PROG: cannot source lib_cut_receipt.sh" >&2; exit 2; }

fault_point() {  # the self-test's kill -9 (RC_CUT_FAULT=<name>); a live run refuses the variable
    [ "${RC_CUT_FAULT:-}" = "$1" ] || return 0
    echo "$PROG: RC_CUT_FAULT=$1: kill -9 $$" >&2; kill -9 $$ $BASHPID
}

run_cut() {
    local tag=$1 sha=$2 t0 rc deadline
    t0=$(date +%s)
    export RC_CUT_LEDGER=${RC_CUT_LEDGER:-$(receipt_file "$tag")}
    echo "$PROG: ledger $RC_CUT_LEDGER"
    bash "$HERE/rc_tag_main.sh" --tag "$tag" --sha "$sha"; rc=$?
    [ "$rc" = 0 ] || { echo "$PROG: the tagger stopped (rc $rc): nothing staged. Re-run this command to resume." >&2; return 1; }

    deadline=$((t0 + ${RC_CUT_ASSET_WAIT:-5400}))
    until bash "$HERE/../check_release_assets.sh" "$tag" > /dev/null 2>&1; do
        [ "$(date +%s)" -lt "$deadline" ] || { echo "$PROG: $tag lacks assets after ${RC_CUT_ASSET_WAIT:-5400}s (scripts/check_release_assets.sh $tag)" >&2; return 2; }
        sleep "${RC_CUT_POLL:-30}"
    done
    fault_point before:deploy   # the operator's falsifier: tagged, nothing deployed

    RC_FLEET_COMMIT=${RC_FLEET_COMMIT:-$sha} bash "$HERE/rc_fleet_stage.sh" "$tag" --publish; rc=$?
    echo "$PROG: $tag finished rc $rc in $(( $(date +%s) - t0 ))s"
    return "$rc"
}

self_test() {
    local fail=0 d me tag=v9.9.9-rc.3 sha=1234567890abcdef1234567890abcdef12345678 h rc got
    local hosts='good1 good2 good3'
    me="$HERE/$(basename -- "${BASH_SOURCE[0]}")"
    fkey() { printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '_'; }
    d=$(mktemp -d) || return 2
    world() {  # a fresh fake GitHub (every judge input green) + a fresh fake fleet of three hosts
        rm -rf -- "${d:?}/api" "${d:?}/hosts" "${d:?}/ledger.tsv" "${d:?}/scp.log" "${d:?}/gh.log"
        mkdir -p "$d/api/get" "$d/api/code" "$d/bin" "$d/assets" "$d/pkg"
        printf '{"status":"identical"}' > "$d/api/get/$(fkey "compare/$sha...main")"
        printf '{"check_runs":[{"name":"ci / gate","status":"completed","conclusion":"success","started_at":"2026-09-25T20:00:00Z"}]}' \
            > "$d/api/get/$(fkey "commits/$sha/check-runs?check_name=ci%20%2F%20gate&per_page=100")"
        printf '{"content":"%s"}' "$(printf '[workspace.package]\nversion = "9.9.9"\n' | base64 -w0)" > "$d/api/get/$(fkey "contents/Cargo.toml?ref=$sha")"
        printf '[]' > "$d/api/get/$(fkey "git/matching-refs/tags/$tag")"
        echo 201 > "$d/api/code/$(fkey git/refs)"; echo 201 > "$d/api/code/releases"
        echo 204 > "$d/api/code/$(fkey actions/workflows/binary-release.yml/dispatches)"
        printf '# measured 2026-09-25T20:00:00Z\nlambda-labs\tapr\tGREEN\tprobe\n' > "$d/cells.tsv"
        : > "$d/hosts.tsv"
        for h in $hosts; do
            mkdir -p "$d/hosts/$h/.cargo/bin"
            printf '#!/bin/sh\necho "apr 9.9.8 (aaaaaaaaa)"\n' > "$d/hosts/$h/.cargo/bin/apr"; chmod +x "$d/hosts/$h/.cargo/bin/apr"
            printf '%s\t%s\tx86_64-unknown-linux-gnu-cpu\tx86_64-unknown-linux-gnu\n' "$h" "$h" >> "$d/hosts.tsv"
        done
        local n a
        for n in apr pv; do
            a="$n-$tag-x86_64-unknown-linux-gnu-cpu.tar.gz"; [ "$n" = pv ] && a="pv-$tag-x86_64-unknown-linux-gnu.tar.gz"
            rm -rf -- "${d:?}/pkg/x"; mkdir -p "$d/pkg/x"
            if [ "$n" = apr ]; then printf '#!/bin/sh\necho "apr 9.9.9-rc.3 (123456789)"\n' > "$d/pkg/x/apr"
            else printf '#!/bin/sh\necho "pv 9.9.9-rc.3 (aprender provable-contracts verifier)"\n' > "$d/pkg/x/pv"; fi
            chmod +x "$d/pkg/x/$n"; tar -czf "$d/assets/$a" -C "$d/pkg" x
            (cd "$d/assets" && sha256sum "$a" > "$a.sha256")
        done
        cat > "$d/bin/ssh" <<EOF
#!/usr/bin/env bash
while [ "\${1:-}" = -o ]; do shift 2; done
h=\$1; shift
[ -d "$d/hosts/\$h" ] || { echo "ssh: \$h: No route to host" >&2; exit 255; }
export HOME="$d/hosts/\$h"; export PATH="\$HOME/.cargo/bin:/usr/bin:/bin"
cd "\$HOME" && eval "\$*"
EOF
        cat > "$d/bin/scp" <<EOF
#!/usr/bin/env bash
args=(); for a in "\$@"; do case "\$a" in -q|-o|BatchMode=yes) ;; *) args+=("\$a") ;; esac; done
dst=\${args[-1]}; unset 'args[-1]'; h=\${dst%%:*}; p=\${dst#*:}
for a in "\${args[@]}"; do echo "\$h \$(basename "\$a")" >> "$d/scp.log"; done
cp -- "\${args[@]}" "$d/hosts/\$h/\$p"
EOF
        echo true > "$d/isdraft"
        cat > "$d/bin/gh" <<EOF
#!/usr/bin/env bash
echo "gh \$*" >> "$d/gh.log"
case "\$1 \$2" in
  "release download") while [ \$# -gt 0 ]; do [ "\$1" = -D ] && dst=\$2; shift; done; cp "$d/assets/"* "\$dst/" ;;
  "release view") case "\$*" in *isDraft*) cat "$d/isdraft" ;; *) echo "notes" ;; esac ;;
  "release edit") echo false > "$d/isdraft" ;;
esac
EOF
        chmod +x "$d/bin/"*
        bash "$HERE/../check_release_assets.sh" --list "$tag" > "$d/complete.txt"
    }
    session() {  # session <script>: ONE fresh session, sharing nothing with the last but the fakes and the ledger
        rm -rf -- "${d:?}/work"; mkdir -p "$d/work"
        env PATH="$d/bin:$PATH" TMPDIR="$d" RC_CUT_FAULT="${F:-}" RC_CUT_LEDGER="$d/ledger.tsv" \
            RC_TAG_FAKE="$d/api" RC_TAG_CELLS="$d/cells.tsv" RC_TAG_NOW="$(date -u -d 2026-09-25T21:00:00Z +%s)" \
            RELEASE_ASSETS_FIXTURE="$d/complete.txt" RC_FLEET_HOSTS_FILE="$d/hosts.tsv" RC_FLEET_WORK="$d/work" \
            RC_FLEET_TODAY=2026-09-25 RC_CUT_ASSET_WAIT=5 RC_CUT_POLL=1 ${HERE_OVERRIDE:+RC_CUT_HERE=$HERE_OVERRIDE} \
            bash "$1" --tag "$tag" --sha "$sha" > "$d/out" 2>&1
    }
    finished() {  # finished <label>: 1 tag, 1 draft, 1 dispatch, 0 duplicate copies, published, every host on the rc sha
        local posts dup on=0 v
        posts=$(cut -d' ' -f2 "$d/api/posts" 2>/dev/null | sort | uniq -c | awk '{printf "%s:%s ", $2, $1}')
        dup=$(sort "$d/scp.log" 2>/dev/null | uniq -d | wc -l)
        for h in $hosts; do v=$("$d/hosts/$h/.cargo/bin/apr" --version); [ "$v" = "apr 9.9.9-rc.3 (123456789)" ] && on=$((on + 1)); done
        if [ "$posts" = "actions/workflows/binary-release.yml/dispatches:1 git/refs:1 releases:1 " ] && [ "$dup" = 0 ] \
            && [ "$(cat "$d/isdraft")" = false ] && [ "$(grep -c -- '--draft=false' "$d/gh.log")" = 1 ] && [ "$on" = 3 ]; then
            echo "  ok   $1: 1 tag, 1 draft, 1 dispatch, 0 duplicate copies, 1 publish, 3/3 hosts on 123456789"
        else echo "  FAIL $1: posts '$posts', duplicate copies $dup, hosts on the rc $on/3, isDraft $(cat "$d/isdraft")"; sed 's/^/       /' "$d/out" | tail -n 8; fail=1; fi
    }

    echo "$PROG self-test: the kill falsifier (#4327 RESUMABLE-CUT)"
    world; F='' session "$me"; rc=$?
    [ "$rc" = 0 ] || { echo "  FAIL an uninterrupted cut: rc $rc"; tail -n 5 "$d/out"; fail=1; }
    finished 'an uninterrupted cut'

    world; { F=before:deploy session "$me"; rc=$?; } 2>/dev/null
    if [ "$rc" = 137 ] && [ ! -s "$d/scp.log" ] && grep -q '^POST git/refs' "$d/api/posts"; then echo "  ok   killed -9 after the tag, before the first deploy (tagged, 0 copies)"
    else echo "  FAIL the kill did not land between tag and deploy (rc $rc)"; fail=1; fi
    F='' session "$me"; rc=$?
    [ "$rc" = 0 ] || { echo "  FAIL the fresh session: rc $rc"; tail -n 5 "$d/out"; fail=1; }
    finished 'a fresh session after the kill'
    grep -q 'resume v9.9.9-rc.3' "$d/out" || { echo "  FAIL the fresh session did not RESUME the tag"; fail=1; }

    for pt in after:tag after:draft after:install:good2 after:publish; do
        world; { F=$pt session "$me"; } 2>/dev/null; F='' session "$me"; rc=$?
        [ "$rc" = 0 ] || { echo "  FAIL killed at $pt, fresh session rc $rc"; tail -n 5 "$d/out"; fail=1; }
        finished "killed at $pt, then a fresh session"
    done
    for v in RC_CUT_FAULT; do   # the fault hook on a live run (no fakes) is refused before any call
        env -u RC_TAG_FAKE "$v=before:deploy" GH_TOKEN=unused bash "$me" --tag "$tag" --sha "$sha" > /dev/null 2>&1; rc=$?
        if [ "$rc" = 2 ]; then echo "  ok   live run with $v set: refused (rc 2)"; else echo "  FAIL live run with $v set: rc $rc"; fail=1; fi
    done

    echo "$PROG self-test: mutant"
    # the ledger made per-session (a new session cannot see the old one's rows): the fresh session
    # must now fail -- the tag exists and nothing says it is this cut's -- or the resume is not the ledger's
    mkdir -p "$d/m"; cp -- "$HERE"/*.sh "$HERE/fleet-waivers.tsv" "$d/m/"
    sed 's|^    export RC_CUT_LEDGER=.*|    export RC_CUT_LEDGER=$(mktemp -u)|' "$me" > "$d/m/$(basename -- "$me")"
    if cmp -s "$d/m/$(basename -- "$me")" "$me"; then echo "  FAIL ledger mutant not built: the anchor moved"; fail=1
    else
        world; { F=before:deploy HERE_OVERRIDE=$d/m session "$d/m/$(basename -- "$me")"; } 2>/dev/null
        HERE_OVERRIDE=$d/m session "$d/m/$(basename -- "$me")"; rc=$?
        if [ "$rc" != 0 ] && [ "$(grep -c '^POST git/refs' "$d/api/posts")" = 1 ] && [ ! -s "$d/scp.log" ]; then
            echo "  ok   mutant without a shared ledger strands the killed cut (rc $rc, never re-tags): the ledger is what resumes"
        else echo "  FAIL mutant without a shared ledger still finished (rc $rc)"; fail=1; fi
    fi
    rm -rf -- "${d:?}"
    if [ "$fail" = 0 ]; then echo "$PROG self-test: PASS"; else echo "$PROG self-test: FAIL"; fi
    return "$fail"
}

main() {
    local tag='' sha=''
    while [ $# -gt 0 ]; do
        case "$1" in
            --self-test) self_test; return $? ;;
            --tag) tag=${2:-}; shift 2 || return 2 ;;
            --sha) sha=${2:-}; shift 2 || return 2 ;;
            -h | --help) sed -n '2,22p' "${BASH_SOURCE[0]}"; return 0 ;;
            *) echo "$PROG: unknown argument $1" >&2; return 2 ;;
        esac
    done
    if [ -n "${RC_CUT_FAULT:-}${RC_CUT_HERE:-}" ] && [ -z "${RC_TAG_FAKE:-}" ]; then
        echo "$PROG: RC_CUT_FAULT/RC_CUT_HERE are self-test only" >&2; return 2
    fi
    [ -n "$tag" ] && [ -n "$sha" ] || { echo "$PROG: --tag vX.Y.Z-rc.N and --sha <40-hex> are required" >&2; return 2; }
    run_cut "$tag" "$sha"
}

main "$@"
