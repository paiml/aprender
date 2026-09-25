#!/usr/bin/env bash
# rc_fleet_stage.sh -- an rc is published only when every fleet host already runs it (#4327, #4328 C2)
#
#   rc_fleet_stage.sh vX.Y.Z-rc.N               stage + verify on every host; print the receipt
#   rc_fleet_stage.sh vX.Y.Z-rc.N --publish     ...and on N/N flip the DRAFT to a visible prerelease
#   rc_fleet_stage.sh --self-test               verdict table + mutants + a fake fleet (no network)
#
# Operator, verbatim (2026-09-24): "THE INSTANT we do a release candidate it needs to be on all
# hardware either before or same time. change process". v0.69.3-rc.2 was published while
# lambda still ran rc.1, intel ran an older apr, and mini had no darwin asset to install: the
# fleet install ran on an hourly timer that nothing tied to the cut.
#
# THE NEW ORDER. The rc tagger creates the rc as a DRAFT release (rc_cut.sh did, until #4314
# was superseded by rc = tag on a queue-green main; the tagger now owns that step) and binary-release.yml attaches the
# assets to it (a draft is invisible to install.sh and to the fleet poller). This script then,
# for every host in the fleet (lambda, gx10, yoga, intel, mini; jetson is retired, #4328 C7):
#   1. copies the host's apr asset (+ .sha256), and pv's on linux, onto the host;
#   2. installs it: the host's own `fleet-bins install <tag> <dir> <commit>` (infra 620249d8)
#      when it has one,
#      otherwise the built-in atomic install below (sha256 checked, rename over the binary
#      PATH resolves, the previous copy kept beside it);
#   3. verifies what PATH now resolves: `apr --version` must equal `apr <X.Y.Z-rc.N> (<sha>)`
#      of THIS tag's commit (asset_version_check.sh, #4110), and `pv --version` its version.
# It publishes only when every host passes. A failed host holds the rc. An unreachable host
# holds it too -- never skipped silently (#4328 C7) -- unless a dated waiver names it:
#   RC_FLEET_WAIVERS=FILE   lines `<host>\t<until YYYY-MM-DD>\t<reason>`, valid through `until`
# A waiver never covers a host that was reached and failed.
#
# The per-host receipt (host, arch, tag, sha, path resolved, verdict) is printed, written to
# $RC_FLEET_RECEIPT, and appended to the release notes on publish (#4327 step 5).
#
# Runs on lambda as the operator user: it needs ssh to the fleet and an authenticated gh (it
# reads a draft). The fleet boxes and the CI runners have neither.
#
# EXIT  0 every host passes (published with --publish) · 1 held: a host failed or was not
#       reached · 2 usage/ENV: the release, its assets or the tag commit could not be read.
set -uo pipefail
PROG=rc_fleet_stage
HERE=${RC_FLEET_HERE:-"$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"}  # a mutant copy runs from a tmp dir
REPO=${RC_FLEET_REPO:-paiml/aprender}
SSH=${RC_FLEET_SSH:-ssh}
SCP=${RC_FLEET_SCP:-scp}

# host  ssh-target(- = this box)  apr asset suffix  pv asset suffix (- = none ships)
FLEET_DEFAULT='lambda	-	x86_64-unknown-linux-gnu-cuda	x86_64-unknown-linux-gnu
gx10	gx10	aarch64-unknown-linux-gnu-cuda	aarch64-unknown-linux-gnu
yoga	yoga	x86_64-unknown-linux-gnu-cuda	x86_64-unknown-linux-gnu
intel	intel	x86_64-unknown-linux-gnu-cpu	x86_64-unknown-linux-gnu
mini	mini	aarch64-apple-darwin-cpu	-'

# stage_verdict -- pure. stdin: rows `<host>\t<pass|fail|unreachable>`; $1 waivers text; $2 today
# (YYYY-MM-DD). Prints `publish` or `hold <why; why>`. Returns 0 for both.
stage_verdict() {
    local waivers=$1 today=$2 host state why='' n=0 until
    while IFS=$'\t' read -r host state _; do
        [ -n "$host" ] || continue
        n=$((n + 1))
        case "$state" in
            pass) ;;
            fail) why+="$host failed; " ;;
            unreachable)
                until=$(printf '%s\n' "$waivers" | awk -F'\t' -v h="$host" '$1==h {print $2}' | sort | tail -n 1)
                if [ -n "$until" ] && [[ "$until" > "$today" || "$until" = "$today" ]]; then :
                else why+="$host unreachable (no waiver through $today); "; fi ;;
            *) why+="$host state '$state'; " ;;
        esac
    done
    [ "$n" -gt 0 ] || why='no hosts; '
    if [ -z "$why" ]; then echo publish; else echo "hold ${why%; }"; fi
}

# ---- the remote side: one bash script, run on the host with its login PATH ------------
# args: <dir, relative to $HOME> <bin> <asset>. Prints one line: INSTALLED <path> | <REASON> <detail>.
INSTALL_SH='set -u
d=$HOME/$1 bin=$2 asset=$3
cd "$d" || { echo "NO-DIR $d"; exit 1; }
want=$(cut -d" " -f1 "$asset.sha256" 2>/dev/null)
if command -v sha256sum >/dev/null 2>&1; then got=$(sha256sum "$asset" | cut -d" " -f1); else got=$(shasum -a 256 "$asset" | cut -d" " -f1); fi
[ -n "$want" ] && [ "$got" = "$want" ] || { echo "SHA-MISMATCH $asset want=$want got=$got"; exit 1; }
rm -rf -- "${d:?}/x" && mkdir x && tar -xzf "$asset" -C x || { echo "UNTAR $asset"; exit 1; }
new=$(find x -type f -name "$bin" | head -n 1)
[ -n "$new" ] || { echo "NO-BINARY $bin in $asset"; exit 1; }
chmod 0755 "$new"
tgt=$(command -v "$bin" 2>/dev/null || true)
case "$tgt" in /*) ;; *) tgt=$HOME/.cargo/bin/$bin; mkdir -p "$HOME/.cargo/bin" ;; esac
dir=$(dirname "$tgt"); run=
if [ ! -w "$dir" ]; then
    if sudo -n true 2>/dev/null; then run="sudo -n"; else echo "NOT-WRITABLE $dir (no passwordless sudo)"; exit 1; fi
fi
[ -e "$tgt" ] && $run cp -p "$tgt" "$dir/.$bin.rc-stage.prev"
$run cp "$new" "$dir/.$bin.rc-stage.new" && $run mv -f "$dir/.$bin.rc-stage.new" "$tgt" || { echo "SWAP $tgt"; exit 1; }
echo "INSTALLED $tgt"'

rx() {  # rx <ssh-target> <command string> -> runs it under a login bash on the host
    if [ "$1" = - ]; then bash -lc "$2"; else "$SSH" -o BatchMode=yes -o ConnectTimeout=10 "$1" "bash -lc $(printf '%q' "$2")"; fi
}
rx_script() {  # rx_script <ssh-target> <args...> < script
    local t=$1; shift
    if [ "$t" = - ]; then bash -l -s -- "$@"; else "$SSH" -o BatchMode=yes -o ConnectTimeout=10 "$t" "bash -l -s -- $(printf '%q ' "$@")"; fi
}
put() {  # put <ssh-target> <remote dir, relative to $HOME> <files...>
    local t=$1 d=$2; shift 2
    if [ "$t" = - ]; then mkdir -p "$HOME/$d" && cp -- "$@" "$HOME/$d/"; else rx "$t" "mkdir -p \"\$HOME/$d\"" && "$SCP" -q -o BatchMode=yes "$@" "$t:$d/"; fi
}

# stage_host <name> <ssh> <apr suffix> <pv suffix> <tag> <commit> <assets dir>
# prints one receipt row: host arch tag sha path verdict detail
stage_host() {
    local h=$1 t=$2 as=$3 ps=$4 tag=$5 commit=$6 ad=$7 rdir out arch path line pvline detail='' ok=1 f
    rdir=.cache/rc-stage/$tag
    if ! arch=$(rx "$t" 'uname -sm' 2>/dev/null | tr ' ' '-') || [ -z "$arch" ]; then
        printf '%s\t?\t%s\t%s\t-\tunreachable\tssh %s failed\n' "$h" "$tag" "${commit:0:9}" "$t"; return 0
    fi
    local staged=()  # bin:asset pairs copied onto the host
    for f in "apr:$as" "pv:$ps"; do
        local bin=${f%%:*} sfx=${f#*:} asset
        [ "$sfx" = - ] && { detail+="$bin: no asset ships for this host; "; continue; }
        asset="$bin-$tag-$sfx.tar.gz"
        [ -f "$ad/$asset" ] && [ -f "$ad/$asset.sha256" ] || { ok=0; detail+="$bin: $asset not downloaded; "; continue; }
        put "$t" "$rdir" "$ad/$asset" "$ad/$asset.sha256" || { ok=0; detail+="$bin: copy failed; "; continue; }
        staged+=("$f")
    done
    if [ "${#staged[@]}" -gt 0 ] && rx "$t" 'command -v fleet-bins' > /dev/null 2>&1; then
        # infra's installer (620249d8): verifies each .sha256, swaps atomically, records the
        # commit, never downgrades. Its rc is the verdict, not its prose.
        out=$(rx "$t" "fleet-bins install $(printf '%q' "$tag") \"\$HOME/$rdir\" $commit" 2>&1) \
            || { ok=0; detail+="fleet-bins install: $(printf '%s' "$out" | tail -n 1); "; }
    else
        for f in "${staged[@]}"; do
            out=$(rx_script "$t" "$rdir" "${f%%:*}" "${f%%:*}-$tag-${f#*:}.tar.gz" <<< "$INSTALL_SH" 2>&1 | tail -n 1)
            case "$out" in INSTALLED*) ;; *) ok=0; detail+="${f%%:*}: install: $out; " ;; esac
        done
    fi
    # Verify what PATH resolves NOW -- not the file just written: a shadow earlier on PATH
    # would keep serving the old build (lambda's ~/.cargo/bin +no-git copy, #4327).
    path=$(rx "$t" 'command -v apr' 2>/dev/null | tail -n 1)
    line=$(rx "$t" 'apr --version' 2>/dev/null | head -n 1)
    if ! out=$(bash "$HERE/asset_version_check.sh" "$tag" "$commit" "$line" 2>&1); then ok=0; detail+="apr: $out; "; fi
    if [ "$ps" != - ]; then
        pvline=$(rx "$t" 'pv --version' 2>/dev/null | head -n 1)
        [ "$(printf '%s' "$pvline" | cut -d' ' -f1,2)" = "pv ${tag#v}" ] || { ok=0; detail+="pv: '$pvline' is not pv ${tag#v}; "; }
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$h" "$arch" "$tag" "${commit:0:9}" "${path:--}" "$([ "$ok" = 1 ] && echo pass || echo fail)" "${detail%; }"
}

tag_commit() {  # the commit a tag names (annotated tags dereferenced)
    [ -n "${RC_FLEET_COMMIT:-}" ] && { echo "$RC_FLEET_COMMIT"; return 0; }
    local refs
    refs=$(git ls-remote "https://github.com/$REPO" "refs/tags/$1" "refs/tags/$1^{}") || return 2
    printf '%s\n' "$refs" | awk '/\^\{\}$/ {d=$1} {p=$1} END {print (d != "" ? d : p)}'
}

stage() {
    local tag=$1 publish=$2 commit work rows verdict today hosts receipt waivers='' n pass
    [[ "$tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+-rc\.[0-9]+$ ]] || { echo "$PROG: $tag is not vX.Y.Z-rc.N" >&2; return 2; }
    commit=$(tag_commit "$tag") && [[ "$commit" =~ ^[0-9a-f]{40}$ ]] || { echo "$PROG: cannot resolve the commit of $tag" >&2; return 2; }
    bash "$HERE/../check_release_assets.sh" "$tag" > /dev/null || { echo "$PROG: $tag does not carry every asset yet (scripts/check_release_assets.sh $tag)" >&2; return 2; }
    work=${RC_FLEET_WORK:-$(mktemp -d)} || return 2
    gh release download "$tag" -R "$REPO" -D "$work" --clobber -p "apr-$tag-*" -p "pv-$tag-*" > /dev/null 2>&1 \
        || { echo "$PROG: cannot download the assets of $tag" >&2; return 2; }
    hosts=${RC_FLEET_HOSTS_FILE:+$(grep -vE '^\s*(#|$)' "$RC_FLEET_HOSTS_FILE")}
    hosts=${hosts:-$FLEET_DEFAULT}
    [ -n "${RC_FLEET_WAIVERS:-}" ] && waivers=$(grep -vE '^\s*(#|$)' "$RC_FLEET_WAIVERS")
    rows=$(while IFS=$'\t' read -r h t as ps; do stage_host "$h" "$t" "$as" "$ps" "$tag" "$commit" "$work"; done <<< "$hosts")
    today=${RC_FLEET_TODAY:-$(date -u +%F)}
    verdict=$(printf '%s\n' "$rows" | awk -F'\t' '{print $1 "\t" $6}' | stage_verdict "$waivers" "$today")
    n=$(printf '%s\n' "$rows" | grep -c .); pass=$(printf '%s\n' "$rows" | awk -F'\t' '$6=="pass"' | grep -c .)
    receipt=${RC_FLEET_RECEIPT:-$work/receipt.md}
    {
        echo "### Fleet staging of $tag at ${commit:0:9}: $pass/$n hosts (#4327)"
        echo
        echo "| host | arch | tag | sha | path resolved | verdict | detail |"
        echo "|---|---|---|---|---|---|---|"
        printf '%s\n' "$rows" | awk -F'\t' '{printf "| %s | %s | %s | %s | %s | %s | %s |\n", $1,$2,$3,$4,$5,($6=="pass"?"PASS":toupper($6)),$7}'
        echo
        echo "verdict: $verdict"
    } > "$receipt"
    cat "$receipt"
    case "$verdict" in
        publish) ;;
        *) echo "$PROG: HELD -- $tag stays a draft" >&2; return 1 ;;
    esac
    [ "$publish" = 1 ] || { echo "(no --publish: $tag stays a draft)"; return 0; }
    local body
    body=$(gh release view "$tag" -R "$REPO" --json body -q .body) || { echo "$PROG: cannot read the notes of $tag" >&2; return 2; }
    printf '%s\n\n%s\n' "$body" "$(cat "$receipt")" > "$work/notes.md"
    gh release edit "$tag" -R "$REPO" --draft=false --prerelease --latest=false --notes-file "$work/notes.md" > /dev/null \
        || { echo "$PROG: every host passed but publishing $tag failed" >&2; return 2; }
    echo "PUBLISHED $tag: on $pass/$n fleet hosts before it was visible"
}

# ---- self-test ---------------------------------------------------------------------
self_test() {
    local fail=0 got d mut rc log
    v() {  # v <want> <label> <rows> [waivers] [today]
        got=$(printf '%b' "$3" | stage_verdict "${4:-}" "${5:-2026-09-24}")
        if [ "$got" = "$1" ]; then echo "  ok   $2"; else printf '  FAIL %s\n       want: %s\n       got:  %s\n' "$2" "$1" "$got"; fail=1; fi
    }
    echo "$PROG self-test: stage_verdict"
    v publish 'every host passes -> publish' 'lambda\tpass\ngx10\tpass\nmini\tpass\n'
    v 'hold gx10 failed' 'ONE host failing holds the rc' 'lambda\tpass\ngx10\tfail\nmini\tpass\n'
    v 'hold intel unreachable (no waiver through 2026-09-24)' 'an unreachable host is RED, never skipped (#4328 C7)' 'lambda\tpass\nintel\tunreachable\n'
    v publish 'a dated waiver through today covers an unreachable host' 'lambda\tpass\nintel\tunreachable\n' $'intel\t2026-09-24\tmaintenance'
    v 'hold intel unreachable (no waiver through 2026-09-24)' 'an expired waiver covers nothing' 'lambda\tpass\nintel\tunreachable\n' $'intel\t2026-09-23\tmaintenance'
    v 'hold intel failed' 'a waiver never covers a host that was reached and failed' 'lambda\tpass\nintel\tfail\n' $'intel\t2026-12-31\tx'
    v 'hold no hosts' 'no hosts is not N/N' ''
    v "hold yoga state 'maybe'" 'an unknown state holds' 'yoga\tmaybe\n'

    # MUTANTS: the refusal made a no-op must PUBLISH past the bad host, or the rows above are
    # not what refuses it. (Deleting the arm alone is not a mutant: the catch-all holds too.)
    mutant() {  # mutant <label> <sed expr> <rows>
        mut=$(mktemp) || return 2
        sed "$2" "${BASH_SOURCE[0]}" > "$mut"
        if cmp -s "$mut" "${BASH_SOURCE[0]}"; then echo "  FAIL mutant '$1' not built: the anchor moved"; fail=1
        else
            got=$(printf '%b' "$3" | bash -c ". '$mut' --source-only; stage_verdict '' 2026-09-24")
            if [ "$got" = publish ]; then echo "  ok   mutant '$1' publishes past the bad host: the refusal is load-bearing"
            else echo "  FAIL mutant '$1' still says '$got': something else refuses"; fail=1; fi
        fi
        rm -f -- "$mut"
    }
    mutant 'fail arm is a no-op' 's/fail) why+="\$host failed; " ;;/fail) ;;/' 'lambda\tpass\ngx10\tfail\n'
    mutant 'unreachable counts as pass' 's/^            unreachable)$/            unreachable) continue ;; x)/' 'lambda\tpass\nintel\tunreachable\n'

    # A FAKE FLEET, end to end: three hosts behind a fake ssh/scp/gh. The operator's
    # acceptance (#4327): "a cut where one host fails verify does NOT publish".
    d=$(mktemp -d) || return 2
    local tag=v9.9.9-rc.3 sha=1234567890abcdef1234567890abcdef12345678 h
    mkdir -p "$d/bin" "$d/assets" "$d/pkg"
    for h in good1 good2 bad; do mkdir -p "$d/hosts/$h/.cargo/bin"; done
    fake_bin() {  # fake_bin <name> <version line> -> a tarball + .sha256 in $d/assets
        local n=$1 a
        a="$n-$tag-x86_64-unknown-linux-gnu-${3:-cpu}.tar.gz"; [ "$n" = pv ] && a="pv-$tag-x86_64-unknown-linux-gnu.tar.gz"
        rm -rf -- "${d:?}/pkg/x"; mkdir -p "$d/pkg/x"
        printf '#!/bin/sh\necho "%s"\n' "$2" > "$d/pkg/x/$n"; chmod +x "$d/pkg/x/$n"
        tar -czf "$d/assets/$a" -C "$d/pkg" x
        (cd "$d/assets" && sha256sum "$a" > "$a.sha256")
    }
    fake_bin apr "apr 9.9.9-rc.3 (123456789)"
    fake_bin pv "pv 9.9.9-rc.3 (aprender provable-contracts verifier)"
    # the bad host runs an old apr and a fleet-bins that SAYS installed and swaps nothing:
    # only verifying what PATH serves afterwards can see it (the installer is not trusted)
    mkdir -p "$d/hosts/bad/shadow"; printf '#!/bin/sh\necho "apr 9.9.9 (v9.9.9+no-git)"\n' > "$d/hosts/bad/shadow/apr"
    printf '#!/bin/sh\necho "apr INSTALLED $2"\n' > "$d/hosts/bad/shadow/fleet-bins"
    # ...and an already-current pv, so apr's verify is the ONLY thing that can hold it
    printf '#!/bin/sh\necho "pv 9.9.9-rc.3 (aprender provable-contracts verifier)"\n' > "$d/hosts/bad/shadow/pv"; chmod +x "$d/hosts/bad/shadow/"*
    log=$d/gh.log
    # fake ssh: `ssh -o .. -o .. HOST CMD` runs CMD with HOME and PATH of that host
    cat > "$d/bin/ssh" <<EOF
#!/usr/bin/env bash
while [ "\${1:-}" = -o ]; do shift 2; done
h=\$1; shift
[ -d "$d/hosts/\$h" ] || { echo "ssh: connect to host \$h: No route to host" >&2; exit 255; }
export HOME="$d/hosts/\$h"; export PATH="\$HOME/shadow:\$HOME/.cargo/bin:/usr/bin:/bin"
cd "\$HOME" && eval "\$*"
EOF
    cat > "$d/bin/scp" <<EOF
#!/usr/bin/env bash
args=(); for a in "\$@"; do case "\$a" in -q|-o|BatchMode=yes) ;; *) args+=("\$a") ;; esac; done
dst=\${args[-1]}; unset 'args[-1]'; h=\${dst%%:*}; p=\${dst#*:}
cp -- "\${args[@]}" "$d/hosts/\$h/\$p"
EOF
    cat > "$d/bin/gh" <<EOF
#!/usr/bin/env bash
echo "gh \$*" >> "$log"
case "\$1 \$2" in
  "release download") while [ \$# -gt 0 ]; do [ "\$1" = -D ] && dst=\$2; shift; done; cp "$d/assets/"* "\$dst/" ;;
  "release view") echo "notes" ;;
  "release edit") ;;
esac
EOF
    chmod +x "$d/bin/"*
    printf '%s\n' "apr-$tag-x86_64-unknown-linux-gnu-cpu.tar.gz" > "$d/assets.list"
    run_fleet() {  # run_fleet <hosts...> -> rc of a --publish run over those hosts (of $self)
        printf '' > "$log"; : > "$d/hosts.tsv"
        for h in "$@"; do printf '%s\t%s\tx86_64-unknown-linux-gnu-cpu\tx86_64-unknown-linux-gnu\n' "$h" "$h" >> "$d/hosts.tsv"; done
        PATH="$d/bin:$PATH" RC_FLEET_COMMIT=$sha RC_FLEET_HOSTS_FILE="$d/hosts.tsv" RC_FLEET_WORK="$d/work" \
            RELEASE_ASSETS_FIXTURE="$d/complete.txt" RC_FLEET_TODAY=2026-09-24 \
            RC_FLEET_HERE=$HERE bash "$self" "$tag" --publish > "$d/out.txt" 2>&1
    }
    bash "$HERE/../check_release_assets.sh" --list "$tag" > "$d/complete.txt"
    mkdir -p "$d/work"
    e2e() {  # e2e <want rc> <want edit 0|1> <label> <hosts...>
        local wrc=$1 wedit=$2 label=$3; shift 3
        rm -rf -- "${d:?}/work"; mkdir -p "$d/work"
        run_fleet "$@"; rc=$?
        local edited=0; grep -q '^gh release edit' "$log" && edited=1
        if [ "$rc" = "$wrc" ] && [ "$edited" = "$wedit" ]; then echo "  ok   $label (rc $rc, published=$edited)"
        else echo "  FAIL $label: rc $rc (want $wrc), published=$edited (want $wedit)"; sed 's/^/       /' "$d/out.txt"; fail=1; fi
    }
    echo "$PROG self-test: a fake fleet, end to end"
    local self=${BASH_SOURCE[0]}
    e2e 0 1 'every host installs and verifies -> the draft is published' good1 good2
    grep -q -- '--draft=false' "$log" || { echo "  FAIL publish did not flip --draft=false"; fail=1; }
    grep -q '| good2 | .* | PASS |' "$d/work/receipt.md" || { echo "  FAIL receipt has no PASS row for good2"; fail=1; }
    e2e 1 0 'ONE host whose PATH resolves an old apr -> NOT published (operator acceptance, #4327)' good1 bad good2
    grep -q 'bad.*FAIL.*wants 9.9.9-rc.3' "$d/work/receipt.md" || { echo "  FAIL the bad host's row does not name the version it served"; fail=1; }
    e2e 1 0 'an unreachable host -> NOT published (#4328 C7)' good1 gone
    # MUTANT: the per-host verify deleted -- the lying installer's host must now get published
    self=$d/mutant.sh
    sed 's/if ! out=$(bash "$HERE\/asset_version_check.sh"/if false \&\& out=$(bash "$HERE\/asset_version_check.sh"/' "${BASH_SOURCE[0]}" > "$self"
    if cmp -s "$self" "${BASH_SOURCE[0]}"; then echo "  FAIL verify mutant not built: the anchor moved"; fail=1
    else e2e 0 1 'mutant (asset_version_check deleted) publishes past the lying host: the verify is what holds it' good1 bad good2; fi
    rm -rf -- "${d:?}"
    if [ "$fail" = 0 ]; then echo "$PROG self-test: PASS"; else echo "$PROG self-test: FAIL"; fi
    return "$fail"
}

main() {
    local tag='' publish=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --self-test) self_test; return $? ;;
            --source-only) return 0 ;;
            --publish) publish=1; shift ;;
            -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}"; return 0 ;;
            v*) tag=$1; shift ;;
            *) echo "$PROG: unknown argument $1" >&2; return 2 ;;
        esac
    done
    [ -n "$tag" ] || { echo "usage: $PROG vX.Y.Z-rc.N [--publish] | --self-test" >&2; return 2; }
    stage "$tag" "$publish"
}

if [ "${1:-}" = --source-only ]; then return 0 2>/dev/null || exit 0; fi
main "$@"
