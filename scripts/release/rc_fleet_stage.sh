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
# THE NEW ORDER. rc_cut.sh creates the rc as a DRAFT and binary-release.yml attaches the
# assets to it (a draft is invisible to install.sh and to the fleet poller). This script then,
# for every host in the fleet (lambda, gx10, yoga, intel, mini; jetson is retired, #4328 C7):
#   1. copies the apr and pv assets (+ .sha256) the fleet catalogue installs on that host (#4509);
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

# host  ssh-target(- = this box)  arch. WHICH asset each host runs is NOT kept here: a hand copy
# of it said intel ran -cpu while the fleet catalogue installs -wgpu there, and mini ran no pv
# while the catalogue ships the darwin one (#4509). fleet_table derives both from the catalogue
# fleet-bins installs from (infra machines/fleet-hosts/fleet-bins/fleet-bins.tsv, deployed at
#   RC_FLEET_CATALOGUE, default ~/.config/fleet-bins/fleet-bins.tsv),
# so the rc is staged, verified and floor-measured on the build each host actually runs.
FLEET_HOSTS='lambda	-	x86_64
gx10	gx10	aarch64
yoga	yoga	x86_64
intel	intel	x86_64
mini	mini	aarch64'

# catalogue_suffix <catalogue> <bin> <host> <arch> -- pure. The asset suffix (between
# `<bin>-<tag>-` and `.tar.gz`) of the paiml/aprender <bin> row whose hosts list names <host>;
# empty when no row does, `?` when that row's asset is not `<bin>-{tag}-<suffix>.tar.gz`.
catalogue_suffix() {
    awk -F'\t' -v b="$2" -v h="$3" -v a="$4" '
        $1 == b && $2 == "paiml/aprender" && index("," $3 ",", "," h ",") {
            s = $5; p = b "-{tag}-"
            if (substr(s, 1, length(p)) != p || s !~ /\.tar\.gz$/) { print "?"; exit }
            s = substr(s, length(p) + 1); s = substr(s, 1, length(s) - 7)
            while ((i = index(s, "{arch}")) > 0) s = substr(s, 1, i - 1) a substr(s, i + 6)
            print s; exit
        }' "$1"
}

# fleet_table <catalogue> -> the host table `host\tssh-target\tapr suffix\tpv suffix(- = none)`.
# Fails closed (2): an unreadable catalogue, a host it ships no apr to, a row it cannot parse.
fleet_table() {
    local cat=$1 h t a as ps
    [ -f "$cat" ] && [ -r "$cat" ] || { echo "$PROG: cannot read the fleet catalogue $cat" >&2; return 2; }
    while IFS=$'\t' read -r h t a; do
        as=$(catalogue_suffix "$cat" apr "$h" "$a"); ps=$(catalogue_suffix "$cat" pv "$h" "$a")
        [ -n "$as" ] || { echo "$PROG: the fleet catalogue $cat ships no apr to $h" >&2; return 2; }
        [ "$as" != '?' ] && [ "$ps" != '?' ] || { echo "$PROG: cannot parse the apr/pv asset of $h in $cat" >&2; return 2; }
        printf '%s\t%s\t%s\t%s\n' "$h" "$t" "$as" "${ps:--}"
    done <<< "$FLEET_HOSTS"
}

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
# bashrs disable-next-line=SEC010
cd "$d" || { echo "NO-DIR $d"; exit 1; }
want=$(cut -d" " -f1 "$asset.sha256" 2>/dev/null)
if command -v sha256sum >/dev/null 2>&1; then got=$(sha256sum "$asset" | cut -d" " -f1); else got=$(shasum -a 256 "$asset" | cut -d" " -f1); fi
[ -n "$want" ] && [ "$got" = "$want" ] || { echo "SHA-MISMATCH $asset want=$want got=$got"; exit 1; }
# bashrs disable-next-line=SEC010
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

# ---- the decode floor (#4273): the rc vs the previous release line, on one host -----------
# Run on the floor host AFTER the rc is installed there. args: <dir, relative to $HOME>
# <prev asset> <model> <n> <max tokens> <gpu wait s>. Prints `rc|prev<TAB><bench JSON on one
# line>` per run, interleaved under ONE gpu-q ticket, or one `ERR <why>` line.
DECODE_SH=$(cat <<'EOF'
set -u
d=$HOME/$1 asset=$2 model=$3 n=$4 mt=$5 w=$6
# bashrs disable-next-line=SEC010
cd "$d" || { echo "ERR no dir $d"; exit 0; }
want=$(cut -d" " -f1 "$asset.sha256" 2>/dev/null)
if command -v sha256sum >/dev/null 2>&1; then got=$(sha256sum "$asset" | cut -d" " -f1); else got=$(shasum -a 256 "$asset" | cut -d" " -f1); fi
[ -n "$want" ] && [ "$got" = "$want" ] || { echo "ERR SHA-MISMATCH $asset"; exit 0; }
# bashrs disable-next-line=SEC010
rm -rf -- "${d:?}/x" && mkdir x && tar -xzf "$asset" -C x || { echo "ERR UNTAR $asset"; exit 0; }
prev=$(find "$d/x" -type f -name apr | head -n 1); cur=$(command -v apr)
[ -n "$prev" ] && [ -n "$cur" ] || { echo "ERR no apr binary (prev='$prev' rc='$cur')"; exit 0; }
chmod 0755 "$prev"
case "$model" in "~/"*) model=$HOME/${model#"~/"} ;; esac
[ -f "$model" ] || { echo "ERR no model $model on this host"; exit 0; }
command -v gpu-q > /dev/null 2>&1 || { echo "ERR no gpu-q on this host"; exit 0; }
loop='for i in $(seq "$1"); do for s in rc prev; do b=$2; [ "$s" = prev ] && b=$3
o=$("$b" bench "$4" --json --max-tokens "$5" --warmup 1 --iterations 3 2>/dev/null)
printf "%s\t%s\n" "$s" "$(printf %s "$o" | tr -d "\n")"; done; done'
GPUQ_WAIT=$w gpu-q --prio 1 -- bash -c "$loop" _ "$n" "$cur" "$prev" "$model" "$mt"
q=$?
[ "$q" = 75 ] && echo "ERR the GPU lock stayed held for ${w}s (gpu-q 75)"
exit 0
EOF
)

# decode_floor_row <tag> <commit> <assets dir> <hosts table> <stage rows> -> one receipt row,
# host `decode-floor`: pass | fail (never waivable) | unreachable (not measured; only a dated
# `decode-floor` waiver covers it). Thresholds and the model live in scripts/perf-matrix.yaml.
decode_floor_row() {
    local tag=$1 commit=$2 ad=$3 hosts=$4 rows=$5 df="$HERE/decode_floor.py" fh t as prev asset rdir out err state detail
    fh=${RC_FLEET_FLOOR_HOST:-$(python3 "$df" get host 2>/dev/null)}
    row() { printf 'decode-floor\ton %s\t%s\t%s\t-\t%s\t%s\n' "${fh:-?}" "$tag" "${commit:0:9}" "$1" "$2"; }
    [ -n "$fh" ] || { row unreachable "no release_gates.decode_floor.host in scripts/perf-matrix.yaml"; return 0; }
    IFS=$'\t' read -r _ t as _ < <(printf '%s\n' "$hosts" | awk -F'\t' -v h="$fh" '$1==h')
    [ -n "${t:-}" ] || { row unreachable "floor host $fh is not in the fleet table"; return 0; }
    [ "$(printf '%s\n' "$rows" | awk -F'\t' -v h="$fh" '$1==h {print $6}')" = pass ] \
        || { row unreachable "$fh did not stage $tag, so its decode was not measured"; return 0; }
    prev=$(gh release list -R "$REPO" --exclude-drafts --limit 200 --json tagName -q '.[].tagName' 2>/dev/null \
        | python3 "$df" prev-tag "$tag") || { row unreachable "no previous release line to measure against"; return 0; }
    asset="apr-$prev-$as.tar.gz"
    mkdir -p "$ad/prev"
    gh release download "$prev" -R "$REPO" -D "$ad/prev" --clobber -p "$asset" -p "$asset.sha256" > /dev/null 2>&1
    [ -f "$ad/prev/$asset" ] && [ -f "$ad/prev/$asset.sha256" ] || { row unreachable "cannot download $asset"; return 0; }
    rdir=.cache/rc-stage/$tag/prev
    put "$t" "$rdir" "$ad/prev/$asset" "$ad/prev/$asset.sha256" || { row unreachable "copy of $asset to $fh failed"; return 0; }
    out=$(rx_script "$t" "$rdir" "$asset" "$(python3 "$df" get model)" "$(python3 "$df" get n)" \
        "$(python3 "$df" get max_tokens)" "$(python3 "$df" get gpu_wait)" <<< "$DECODE_SH" 2>/dev/null)
    err=$(printf '%s\n' "$out" | sed -n 's/^ERR //p' | head -n 1)
    [ -z "$err" ] || { row unreachable "$err"; return 0; }
    IFS=$'\t' read -r state detail < <(printf '%s\n' "$out" | python3 "$df" verdict --commit "$commit")
    case "$state" in
        pass|fail) ;;
        unmeasured) state=unreachable ;;
        *) detail="decode_floor.py said '$state'"; state=fail ;;
    esac
    row "$state" "vs $prev: $detail"
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
    if [ -n "${RC_FLEET_HOSTS_FILE:-}" ]; then hosts=$(grep -vE '^\s*(#|$)' "$RC_FLEET_HOSTS_FILE")
    else hosts=$(fleet_table "${RC_FLEET_CATALOGUE:-$HOME/.config/fleet-bins/fleet-bins.tsv}") || return 2; fi
    commit=$(tag_commit "$tag") && [[ "$commit" =~ ^[0-9a-f]{40}$ ]] || { echo "$PROG: cannot resolve the commit of $tag" >&2; return 2; }
    bash "$HERE/../check_release_assets.sh" "$tag" > /dev/null || { echo "$PROG: $tag does not carry every asset yet (scripts/check_release_assets.sh $tag)" >&2; return 2; }
    work=${RC_FLEET_WORK:-$(mktemp -d)} || return 2
    gh release download "$tag" -R "$REPO" -D "$work" --clobber -p "apr-$tag-*" -p "pv-$tag-*" > /dev/null 2>&1 \
        || { echo "$PROG: cannot download the assets of $tag" >&2; return 2; }
    [ -n "${RC_FLEET_WAIVERS:-}" ] && waivers=$(grep -vE '^\s*(#|$)' "$RC_FLEET_WAIVERS")
    rows=$(while IFS=$'\t' read -r h t as ps; do stage_host "$h" "$t" "$as" "$ps" "$tag" "$commit" "$work"; done <<< "$hosts")
    rows+=$'\n'$(decode_floor_row "$tag" "$commit" "$work" "$hosts" "$rows")
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

    # THE HOST TABLE IS THE FLEET CATALOGUE'S (#4509): these rows mirror fleet-bins.tsv, where
    # intel runs the -wgpu apr and mini the darwin pv. A hand table said -cpu and none.
    local cd cf ft
    cd=$(mktemp -d) || return 1; cf=$cd/fleet-bins.tsv
    t() {  # t <want rc> <label> <want table|-> [catalogue] [script]
        ft=$(bash -c 'source "$1" --source-only; fleet_table "$2"' _ "${5:-${BASH_SOURCE[0]}}" "${4:-$cf}" 2> "$cd/err"); rc=$?
        if [ "$rc" = "$1" ] && { [ "$3" = - ] || [ "$ft" = "$(printf '%b' "$3")" ]; }; then echo "  ok   $2"
        else printf '  FAIL %s: rc %s (want %s)\n' "$2" "$rc" "$1"; printf '%s\n' "$ft" | sed 's/^/       /'; sed 's/^/       /' "$cd/err"; fail=1; fi
    }
    printf '%b' 'apr\tpaiml/aprender\tlambda,gx10,yoga\t^v\tapr-{tag}-{arch}-unknown-linux-gnu-cuda.tar.gz\t-\n' \
        'pv\tpaiml/aprender\tlambda,gx10,yoga,intel\t^v\tpv-{tag}-{arch}-unknown-linux-gnu.tar.gz\t-\n' \
        'apr\tpaiml/aprender\tintel\t^v\tapr-{tag}-x86_64-unknown-linux-gnu-wgpu.tar.gz\t-\n' \
        'apr\tpaiml/aprender\tmini\t^v\tapr-{tag}-aarch64-apple-darwin-cpu.tar.gz\t-\n' \
        'pv\tpaiml/aprender\tmini\t^v\tpv-{tag}-aarch64-apple-darwin.tar.gz\t-\n' \
        'apr\tpaiml/forjar\tintel,mini\t^v\tapr-{tag}-decoy.tar.gz\t-\n' > "$cf"
    local want='lambda\t-\tx86_64-unknown-linux-gnu-cuda\tx86_64-unknown-linux-gnu\ngx10\tgx10\taarch64-unknown-linux-gnu-cuda\taarch64-unknown-linux-gnu\nyoga\tyoga\tx86_64-unknown-linux-gnu-cuda\tx86_64-unknown-linux-gnu\nintel\tintel\tx86_64-unknown-linux-gnu-wgpu\tx86_64-unknown-linux-gnu\nmini\tmini\taarch64-apple-darwin-cpu\taarch64-apple-darwin'
    t 0 'catalogue -> intel stages -wgpu, mini stages the darwin pv, {arch} per host' "$want"
    grep -v 'intel,mini' "$cf" | sed 's/lambda,gx10,yoga,intel/lambda,gx10,yoga/' > "$cd/nopv.tsv"
    t 0 'a host no pv row covers -> pv "-" (none ships), not a guess' "${want/intel\\tintel\\tx86_64-unknown-linux-gnu-wgpu\\tx86_64-unknown-linux-gnu/intel\\tintel\\tx86_64-unknown-linux-gnu-wgpu\\t-}" "$cd/nopv.tsv"
    grep -v 'wgpu' "$cf" > "$cd/noapr.tsv"
    t 2 'a host the catalogue ships no apr to -> refused, never a default' - "$cd/noapr.tsv"
    sed 's/aarch64-apple-darwin.tar.gz/aarch64-apple-darwin.zip/' "$cf" > "$cd/odd.tsv"
    t 2 'an asset not <bin>-{tag}-<suffix>.tar.gz -> refused' - "$cd/odd.tsv"
    t 2 'no catalogue -> refused (fail closed)' - "$cd/absent.tsv"
    [ "$(RC_FLEET_CATALOGUE=$cd/absent.tsv bash "${BASH_SOURCE[0]}" v9.9.9-rc.3 2>&1)" = "$PROG: cannot read the fleet catalogue $cd/absent.tsv" ] \
        && echo "  ok   stage() without RC_FLEET_HOSTS_FILE reads the catalogue, and stops before the network" \
        || { echo "  FAIL stage() does not take its host table from RC_FLEET_CATALOGUE"; fail=1; }
    # MUTANT: the host match dropped -- the first apr row wins everywhere, intel gets -cuda
    sed 's/ && index("," $3 ",", "," h ",") {/ {/' "${BASH_SOURCE[0]}" > "$cd/mutant.sh"
    if cmp -s "$cd/mutant.sh" "${BASH_SOURCE[0]}"; then echo "  FAIL host-match mutant not built: the anchor moved"; fail=1
    else t 0 'mutant (host match dropped) still yields a table' - "$cf" "$cd/mutant.sh"
        [ "$ft" != "$(printf '%b' "$want")" ] && echo "  ok   host-match mutant killed" || { echo "  FAIL host-match mutant survived"; fail=1; }
    fi
    rm -rf -- "${cd:?}"
    # A FAKE FLEET, end to end: three hosts behind a fake ssh/scp/gh. The operator's
    # acceptance (#4327): "a cut where one host fails verify does NOT publish".
    d=$(mktemp -d) || return 2
    local tag=v9.9.9-rc.3 sha=1234567890abcdef1234567890abcdef12345678 h
    mkdir -p "$d/bin" "$d/assets" "$d/pkg"
    # bashrs disable-next-line=SEC010
    for h in good1 good2 bad; do mkdir -p "$d/hosts/$h/.cargo/bin"; done
    fake_bin() {  # fake_bin <name> <version line> [tok/s for `apr bench`] [tag] -> a tarball + .sha256 in $d/assets
        local n=$1 a t=${4:-$tag}
        a="$n-$t-x86_64-unknown-linux-gnu-cpu.tar.gz"; [ "$n" = pv ] && a="pv-$t-x86_64-unknown-linux-gnu.tar.gz"
        rm -rf -- "${d:?}/pkg/x"; mkdir -p "$d/pkg/x"
        # `apr bench` answers with a CUDA receipt whose build_commit is its version line's sha
        printf '#!/bin/sh\n[ "$1" = bench ] && { echo '"'"'{"tokens_per_second": %s,\n "provenance": {"compute_class": "cuda", "build_commit": "%s"}}'"'"'; exit 0; }\necho "%s"\n' \
            "${3:-100}" "$(printf '%s' "$2" | sed -n 's/.*(\([0-9a-f]*\)).*/\1/p')" "$2" > "$d/pkg/x/$n"; chmod +x "$d/pkg/x/$n"
        tar -czf "$d/assets/$a" -C "$d/pkg" x
        (cd "$d/assets" && sha256sum "$a" > "$a.sha256")
    }
    fake_bin apr "apr 9.9.9-rc.3 (123456789)"
    fake_bin apr "apr 9.8.0 (abcdef123)" 100 v9.8.0  # the previous release line the floor measures against
    # the floor host (good1) has the matrix's model and a gpu-q that just runs the command
    mkdir -p "$d/hosts/good1/shadow" "$d/hosts/good1/models"
    : > "$d/hosts/good1/$(python3 "$HERE/decode_floor.py" get model | sed 's|^~/||')"
    printf '#!/bin/sh\nshift 3; exec "$@"\n' > "$d/hosts/good1/shadow/gpu-q"; chmod +x "$d/hosts/good1/shadow/gpu-q"
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
# bashrs disable-next-line=SEC001
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
  "release list") printf 'v9.9.9-rc.2\nv9.8.0\nv9.7.1\n' ;;
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
            RC_FLEET_FLOOR_HOST=good1 RC_FLEET_HERE=$HERE bash "$self" "$tag" --publish > "$d/out.txt" 2>&1
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
    grep -q '| decode-floor | on good1 | .* | PASS | vs v9.8.0: rc 100.0 tok/s vs prev 100.0' "$d/work/receipt.md" \
        || { echo "  FAIL receipt has no measured decode-floor PASS row"; sed 's/^/       /' "$d/work/receipt.md"; fail=1; }
    e2e 1 0 'ONE host whose PATH resolves an old apr -> NOT published (operator acceptance, #4327)' good1 bad good2
    grep -q 'bad.*FAIL.*wants 9.9.9-rc.3' "$d/work/receipt.md" || { echo "  FAIL the bad host's row does not name the version it served"; fail=1; }
    e2e 1 0 'an unreachable host -> NOT published (#4328 C7)' good1 gone
    e2e 1 0 'the floor host absent from the fleet -> NOT published (never skipped)' good2
    grep -q '| decode-floor | .* | UNREACHABLE | floor host good1 is not in the fleet table' "$d/work/receipt.md" \
        || { echo "  FAIL the absent floor host is not named"; fail=1; }
    fake_bin apr "apr 9.9.9-rc.3 (123456789)" 5  # the #4273 shape: the rc decodes 20x slower
    e2e 1 0 'an rc decoding 20x slower than the previous line -> NOT published (#4273)' good1 good2
    grep -q '| decode-floor | .* | FAIL | vs v9.8.0: rc 5.0 tok/s vs prev 100.0 tok/s = 0.050x' "$d/work/receipt.md" \
        || { echo "  FAIL the slow rc's floor row does not say why"; sed 's/^/       /' "$d/work/receipt.md"; fail=1; }
    # MUTANT: the floor row dropped from the verdict -- the slow rc must now get published
    self=$d/mutant-floor.sh
    sed 's/^    rows+=$.\\n.$(decode_floor_row /    : $(decode_floor_row /' "${BASH_SOURCE[0]}" > "$self"
    if cmp -s "$self" "${BASH_SOURCE[0]}"; then echo "  FAIL floor mutant not built: the anchor moved"; fail=1
    else e2e 0 1 'mutant (floor row not in the verdict) publishes the slow rc: the row is what holds it' good1 good2; fi
    self=${BASH_SOURCE[0]}
    fake_bin apr "apr 9.9.9-rc.3 (123456789)"
    # MUTANT: the per-host verify deleted -- the lying installer's host must now get published
    self=$d/mutant.sh
    sed 's/if ! out=$(bash "$HERE\/asset_version_check.sh"/if false \&\& out=$(bash "$HERE\/asset_version_check.sh"/' "${BASH_SOURCE[0]}" > "$self"
    if cmp -s "$self" "${BASH_SOURCE[0]}"; then echo "  FAIL verify mutant not built: the anchor moved"; fail=1
    else e2e 0 1 'mutant (asset verify deleted) publishes past the lying host: the verify is what holds it' good1 bad good2; fi
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
