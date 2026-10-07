#!/usr/bin/env bash
# models_nightly.sh -- the models lane's nightly producer (#4721, 0702-P4). The nightly train reads one
# lane per release gate; the models lane (models_t1.sh's GPU-host ladder, preflight R7) had no producer
# on main, so it printed not_measured every night. This is that producer.
#
# THE SAME ENTRY POINT RELEASE DAY RUNS. --run executes models_t1.sh AS IT IS AT C (main's head):
# both GPU-host ladder legs (one local, one over SSH), its own build, its own binary proof, its own judge. Nothing
# here measures a model; this file only runs C's measurer and carries its result to the train.
#
# WHY A RELAY. The GPU hosts are not Actions runners, and the train reads only Actions runs on main.
#   GPU host --run      measure C with C's models_t1.sh, classify, write a bundle to <work>/pending/<C>
#   GPU host --publish  commit pending bundles to branch nightly-evidence as models/<C>/ (never forced)
#   CI       --relay    models-nightly.yml, at C: fetch models/<C>, verify it, replay the verdict from the
#                       raw evidence, and hand the state to the `models` job. No bundle, a bundle that
#                       fails a check, or a replay that disagrees is not_measured: never a pass.
#   --run and --publish are separate so that the measuring step never sees the push credential: the
#   host's units keep the credential out of the measuring sandbox (the infra ticket, not this file).
#
# THE VERDICT (classify, run identically on the GPU host and in the relay, over the bundle's own files):
#   green         models_t1 exit 0 and its GO line at C, and both receipts bound to "apr <v> (<sha9>)"
#                 with executed > 0 and red == 0
#   red           a bound receipt with red cells, a build failure, a binary that is not C, a receipt
#                 measured by another binary, a ladder that exited 0/1 with no receipt, or the judge
#                 finding red/missing cells on two bound receipts. Red dominates, as in the judge.
#   not_measured  everything else: a host unreachable, a disk refusal (ENV), a judge DECLINE or DEFER
#                 (the judge's own "not green, not red"), a ladder decline, a timeout, an unknown exit,
#                 or a red seen while another models_t1.sh was running (it shares the remote leg's rel-* dirs).
#
# NO REGISTRY TOKEN: --run and --publish refuse (exit 3) where one is reachable -- nightly_train.sh's
# no_token, plus $HOME/.cargo/credentials{,.toml} even when CARGO_HOME points elsewhere.
# PLANTED FAILURE: --relay --plant-red (workflow_dispatch plant_red) turns the lane red whatever the
# bundle says, so the train is shown to read this lane red. A planted run on C IS the newest run on C.
#
# usage:
#   models_nightly.sh --run --repo DIR --work DIR [--commit SHA] [--timeout SECONDS]
#   models_nightly.sh --publish --repo DIR --work DIR
#   models_nightly.sh --relay --commit SHA --out DIR [--evidence DIR] [--plant-red]   (cwd: a checkout)
#   models_nightly.sh --tickets --commit SHA --evidence BUNDLE --out FILE   open/update the issue of each red row (gh)
#   models_nightly.sh --self-test   the case table (fixture repos, no network, no GPU)
#   models_nightly.sh --mutants     each planted mutant must turn the case table RED
# exit: 0 done (a bundle written, published or relayed, whatever its state) · 2 failed · 3 refused
set -uo pipefail
SCRIPT_PATH=$(readlink -f -- "$0")
EV_BRANCH=nightly-evidence
KEEP=${MODELS_NIGHTLY_KEEP:-30}   # bundles kept on the branch tip (history keeps the rest)
TMO=21600                         # --timeout default: models_t1 gets six hours, then it is not_measured

say() { printf 'MODELS-NIGHTLY %s\n' "$*"; }
refuse() { say "REFUSED: $*"; exit 3; }
die() { say "FAILED: $*"; exit 2; }
# A commit id names models/<C>/ and pending/<C>/, so it must be one: 40 hex, never a path.
commit_id() {
    case $1 in *..*|*/*) return 1 ;; esac
    [[ $1 =~ ^[0-9a-f]{40}$ ]]
}

# nightly_train.sh's no_token, verbatim, plus the default credentials a redirected CARGO_HOME hides
no_token() {
    local ch="${CARGO_HOME:-$HOME/.cargo}"
    env | awk -F '=' '$1 == "CARGO_REGISTRY_TOKEN" || $1 ~ /^CARGO_REGISTRIES_[A-Za-z0-9_]+_TOKEN$/ { f = 1 } END { exit f }' || return 1
    [ ! -e "$ch/credentials" ] && [ ! -e "$ch/credentials.toml" ] || return 1
    [ ! -r "$HOME/.cargo/credentials" ] && [ ! -r "$HOME/.cargo/credentials.toml" ] || return 1
    ! grep -qsE '^[[:space:]]*token[[:space:]]*=' "$ch/config" "$ch/config.toml"
}

# ws_version < Cargo.toml -> [workspace.package] version, or nothing
ws_version() {
    awk '/^\[/ { s = ($0 == "[workspace.package]") }
         s && /^version[[:space:]]*=/ { sub(/^version[[:space:]]*=[[:space:]]*"/, ""); sub(/".*$/, ""); print; exit }'
}

# vget KEY < verdict -> the value of KEY=
vget() { awk -v k="$1" 'index($0, k "=") == 1 { print substr($0, length(k) + 2); exit }'; }

# foreign_t1 OWN-PGID < "pgid args" lines -> every models_t1.sh process outside OWN-PGID
foreign_t1() { awk -v own="$1" '$1 != own && $0 ~ /[ \/]models_t1\.sh( |$)/'; }
# The process table foreign_t1 reads. MODELS_NIGHTLY_PS_TABLE (a file of "pgid args" lines) is for
# the self-test only: on a host where anyone is running a models_t1.sh fixture, the real table would
# refuse every e2e row. The host units never set it.
ps_table() { if [ -n "${MODELS_NIGHTLY_PS_TABLE:-}" ]; then cat -- "$MODELS_NIGHTLY_PS_TABLE"; else ps -eo pgid=,args=; fi; }

receipt_bound() { [ -f "$1" ] && jq -es --arg w "$2" 'length == 1 and (.[0] | type == "object" and .apr_version == $w)' -- "$1" > /dev/null 2>&1; }
receipt_red() { jq -es 'length == 1 and (.[0].red | type == "number" and . > 0)' -- "$1" > /dev/null 2>&1; }
receipt_clean() { jq -es 'length == 1 and (.[0] | (.red | type == "number" and . == 0) and (.executed | type == "number" and . > 0))' -- "$1" > /dev/null 2>&1; }

# classify BUNDLE RC VERSION SHA9 -> "<state>\t<reason>", from the bundle's files only
classify() {
    local b=$1 rc=$2 sha9=$4 want="apr $3 ($4)" log="$1/models-t1.log" h bound=0 clean=0 red="" line
    for h in lambda gx10; do
        receipt_bound "$b/$h.json" "$want" || continue
        bound=$((bound + 1))
        if receipt_red "$b/$h.json"; then red="${red:+$red; }$h receipt has red cells"
        elif receipt_clean "$b/$h.json"; then clean=$((clean + 1)); fi
    done
    line=$(grep -E '^MODELS (lambda|gx10) NO-GO: (no receipt -- (BUILD-FAILED|NOT-THE-RELEASE|model_ladder\.sh wrote no receipt \(rc [01]\)$)|the receipt was measured by )' -- "$log" 2> /dev/null | head -n 1)
    [ -z "$line" ] || red="${red:+$red; }${line#MODELS }"
    if [ "$bound" = 2 ] && grep -qE '^MODELS NO-GO: the judge found red or missing cells' -- "$log" 2> /dev/null; then
        red="${red:+$red; }the judge found red or missing cells on two bound receipts"
    fi
    if [ -n "$red" ] && [ -f "$b/concurrent" ]; then
        printf 'not_measured\tred while another models_t1.sh ran (%s): %s\n' "$(head -n 1 -- "$b/concurrent")" "$red"; return 0
    fi
    if [ -n "$red" ]; then printf 'red\t%s\n' "$red"; return 0; fi
    if [ "$rc" = 0 ] && [ "$clean" = 2 ] && grep -qE "^MODELS GO on lambda and gx10 at $sha9: " -- "$log" 2> /dev/null; then
        printf 'green\tmodels_t1 GO on lambda and gx10, both receipts bound to %s\n' "$want"; return 0
    fi
    case $rc in
        124|137) line="models_t1 timed out (rc $rc)" ;;
        0) line="models_t1 exit 0, but the bundle does not hold its GO line and two clean receipts bound to $want" ;;
        *) line=$(grep -E '^MODELS( lambda| gx10)? NO-GO: ' -- "$log" 2> /dev/null | head -n 1)
           line=${line:-models_t1 exit $rc with no NO-GO line} ;;
    esac
    printf 'not_measured\t%s\n' "$line"
}

# seal BUNDLE COMMIT SHA9 VERSION ENTRY RC -> classify the bundle, write its verdict and SHA256SUMS
seal() {
    local b=$1 cls
    cls=$(classify "$b" "$6" "$4" "$3")
    printf 'commit=%s\nsha9=%s\nversion=%s\nentry=%s\nt1_rc=%s\nstate=%s\nreason=%s\n' \
        "$2" "$3" "$4" "$5" "$6" "${cls%%$'\t'*}" "${cls#*$'\t'}" > "$b/verdict" || return 1
    (cd "$b" && find . -maxdepth 1 -type f ! -name SHA256SUMS -printf '%f\n' | LC_ALL=C sort | xargs -r sha256sum -- > SHA256SUMS)
}

# verify BUNDLE COMMIT -> "<state>\t<reason>"; the CI checkout supplies C's version and models_t1 blob
verify() {
    local b=$1 c=$2 v listed have ver blob sha9 replay rec
    [ -f "$b/verdict" ] && [ -f "$b/SHA256SUMS" ] || { printf 'not_measured\tno bundle for %s on %s\n' "${c:0:9}" "$EV_BRANCH"; return 0; }
    (cd "$b" && sha256sum --strict --quiet -c SHA256SUMS > /dev/null 2>&1) \
        || { printf 'not_measured\tthe bundle fails its own SHA256SUMS\n'; return 0; }
    listed=$(awk '{ print $2 }' "$b/SHA256SUMS" | LC_ALL=C sort)
    have=$(cd "$b" && find . -maxdepth 1 -type f ! -name SHA256SUMS -printf '%f\n' | LC_ALL=C sort)
    [ "$listed" = "$have" ] || { printf 'not_measured\tthe bundle holds a file its SHA256SUMS does not list\n'; return 0; }
    v=$(vget commit < "$b/verdict")
    [ "$v" = "$c" ] || { printf 'not_measured\tthe bundle measured %s, not %s\n' "${v:0:9}" "${c:0:9}"; return 0; }
    ver=$(git show "$c:Cargo.toml" 2> /dev/null | ws_version)
    v=$(vget version < "$b/verdict")
    [ -n "$ver" ] && [ "$v" = "$ver" ] || { printf 'not_measured\tthe bundle measured version %s, C is %s\n' "$v" "${ver:-unreadable}"; return 0; }
    blob=$(git rev-parse -q --verify "$c:scripts/release/models_t1.sh")
    v=$(vget entry < "$b/verdict")
    [ -n "$blob" ] && [ "$v" = "scripts/release/models_t1.sh@$blob" ] \
        || { printf 'not_measured\tthe bundle was not measured by C'"'"'s models_t1.sh (%s)\n' "$v"; return 0; }
    sha9=$(vget sha9 < "$b/verdict")
    case $sha9 in "${c:0:9}"*) ;; *) printf 'not_measured\tthe bundle sha9 %s is not a prefix of C\n' "$sha9"; return 0 ;; esac
    replay=$(classify "$b" "$(vget t1_rc < "$b/verdict")" "$ver" "$sha9")
    rec=$(vget state < "$b/verdict")
    [ "${replay%%$'\t'*}" = "$rec" ] || { printf 'not_measured\tthe replay says %s, the bundle recorded %s\n' "${replay%%$'\t'*}" "$rec"; return 0; }
    printf '%s\n' "$replay"
}

relay() { # relay COMMIT OUT EVIDENCE PLANT
    local c=$1 out=$2 ev=$3 cls state reason bundle=
    mkdir -p -- "$out" || die "cannot create $out"
    if [ -z "$ev" ]; then
        ev="$out/models/$c"
        if git fetch -q --no-tags --depth=1 origin "+refs/heads/$EV_BRANCH:refs/remotes/origin/$EV_BRANCH" 2> /dev/null \
            && git rev-parse -q --verify "refs/remotes/origin/$EV_BRANCH:models/$c" > /dev/null; then
            git archive --format=tar "refs/remotes/origin/$EV_BRANCH" "models/$c" > "$out/bundle.tar" \
                && tar -xf "$out/bundle.tar" -C "$out" && rm -f -- "$out/bundle.tar"
        fi
    fi
    cls=$(verify "$ev" "$c")
    state=${cls%%$'\t'*}; reason=${cls#*$'\t'}
    # bundle= is the dir readiness-nightly reads as artifact models-t1: a verified green or red only. Never
    # not_measured (its files proved nothing) and never a planted run (a plant is not a measurement).
    case $state in green|red) [ -n "$4" ] || bundle=$ev ;; esac
    [ -z "$4" ] || { reason="PLANTED red (plant_red), the bundle said $state: $reason"; state=red; }
    reason=$(printf '%s' "$reason" | tr -d '\r\n')
    if [ -n "${GITHUB_OUTPUT:-}" ]; then printf 'state=%s\nreason=%s\nbundle=%s\n' "$state" "$reason" "$bundle" >> "$GITHUB_OUTPUT" || die "cannot write GITHUB_OUTPUT"; fi
    say "RELAY $state at ${c:0:9}: $reason"
}

# red_rows BUNDLE -> "host<TAB>row<TAB>why", one line per red row of a red bundle: each rung a receipt
# marks green:false; a red with no rung to name (a build failure, a binary that is not C, no receipt)
# is one "lane" row, so a red never goes unticketed.
red_rows() {
    local b=$1 h rows=""
    for h in lambda gx10; do
        [ -f "$b/$h.json" ] || continue
        rows+=$(jq -r --arg h "$h" '(.rungs // [])[] | select(.green == false)
            | [$h, (.id // .file // "unnamed-rung"), ("rung not green (" + (.file // "no file") + ")")] | @tsv' -- "$b/$h.json" 2> /dev/null)$'\n'
    done
    rows=$(printf '%s' "$rows" | awk 'NF' | head -n "$TICKETS_MAX")
    if [ -n "$rows" ]; then printf '%s\n' "$rows"
    else printf 'all\tlane\t%s\n' "$(vget reason < "$b/verdict" 2> /dev/null | tr -d '\t')"; fi
}

# tickets BUNDLE COMMIT OUT: the standing release policy sends every red nightly row to a ticket. For
# a verified red bundle, each red row opens the open issue titled "models-nightly red: <row> on <host>"
# or comments on it, once per commit (the marker models-nightly@<C>: both cron slots relay the same C).
# OUT gets "host<TAB>row<TAB>#N" per row, the list a release's known failures name. A nightly row never
# stops a release; a read or write that fails is a FAILED run of this step, never a ticket.
TICKETS_MAX=20
tickets() {
    local b=$1 c=$2 out=$3 gh=${MODELS_NIGHTLY_GH:-gh} host row why title n j body mark st=0
    : > "$out" || die "cannot write $out"
    if [ "$(vget state < "$b/verdict" 2> /dev/null)" != red ]; then say "TICKETS none: the bundle at ${c:0:9} is not red"; return 0; fi
    mark="models-nightly@$c"
    while IFS=$'\t' read -r host row why; do
        title="models-nightly red: $row on $host"
        if ! j=$("$gh" issue list --state open --limit 50 --search "\"$title\" in:title" --json number,title); then
            say "NOT-MEASURED: the issue search for '$title' failed"; st=1; continue
        fi
        n=$(printf '%s' "$j" | jq -r --arg t "$title" '[.[] | select(.title == $t) | .number] | min // empty' 2> /dev/null)
        body="Red in the models nightly at ${c:0:9}: $why. Under the standing release policy this row cannot stop a release; the release notes list it as a known failure with this ticket until it is green. $mark"
        if [ -n "$n" ]; then
            if ! j=$("$gh" issue view "$n" --json body,comments); then say "NOT-MEASURED: reading #$n failed"; st=1; continue; fi
            if printf '%s' "$j" | grep -qF -- "$mark"; then say "TICKET kept #$n: $title (already names ${c:0:9})"
            elif "$gh" issue comment "$n" --body "$body" > /dev/null; then say "TICKET updated #$n: $title"
            else say "NOT-MEASURED: commenting on #$n failed"; st=1; continue; fi
        else
            if ! n=$("$gh" issue create --title "$title" --body "$body"); then say "NOT-MEASURED: opening '$title' failed"; st=1; continue; fi
            n=${n##*/}
            [[ $n =~ ^[0-9]+$ ]] || { say "NOT-MEASURED: opening '$title' printed no issue url"; st=1; continue; }
            say "TICKET opened #$n: $title"
        fi
        printf '%s\t%s\t#%s\n' "$host" "$row" "$n" >> "$out"
    done < <(red_rows "$b")
    [ "$st" = 0 ] || die "a ticket read or write failed; the rows above without TICKET have none"
    say "TICKETS $(wc -l < "$out") row(s) at ${c:0:9} -> $out"
}

# already_measured REPO COMMIT WORK -> 0 when a green or red verdict for COMMIT is pending or published
already_measured() {
    local s
    for s in "$(vget state 2> /dev/null < "$3/pending/$2/verdict")" \
             "$(git -C "$1" cat-file -p "refs/remotes/origin/$EV_BRANCH:models/$2/verdict" 2> /dev/null | vget state)"; do
        case $s in green|red) say "SKIP: ${2:0:9} is already measured $s"; return 0 ;; esac
    done
    return 1
}

run() { # run REPO WORK COMMIT TIMEOUT
    local repo=$1 work=$2 c=$3 tmo=$4 wt out b ver sha9 blob tpid spid rc f cls
    no_token || refuse "a registry token is reachable (env, CARGO_HOME or \$HOME/.cargo): this producer holds none"
    exec 9> "$work/.lock" || die "cannot open $work/.lock"
    flock -n 9 || refuse "another models_nightly.sh holds $work/.lock"
    f=$(ps_table | foreign_t1 0 | head -n 1)
    [ -z "$f" ] || refuse "another models_t1.sh is running (a release's models step shares the GPU hosts): $f"
    git -C "$repo" fetch -q --no-tags origin +refs/heads/main:refs/remotes/origin/main || die "git fetch of main failed in $repo"
    git -C "$repo" fetch -q --no-tags origin "+refs/heads/$EV_BRANCH:refs/remotes/origin/$EV_BRANCH" 2> /dev/null || :
    [ -n "$c" ] || c=$(git -C "$repo" rev-parse refs/remotes/origin/main) || die "origin/main does not resolve"
    c=$(git -C "$repo" rev-parse -q --verify "$c^{commit}") || refuse "'$3' is not a commit in $repo"
    commit_id "$c" || refuse "'$c' is not a 40-hex commit id"
    ! already_measured "$repo" "$c" "$work" || return 0
    wt="$work/wt"; out="$work/out"
    git -C "$repo" worktree remove --force "$wt" > /dev/null 2>&1; rm -rf -- "${wt:?}"; git -C "$repo" worktree prune
    git -C "$repo" worktree add -q --detach "$wt" "$c" || die "worktree at ${c:0:9} failed"
    ver=$(ws_version < "$wt/Cargo.toml")
    [[ $ver =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || refuse "C's [workspace.package] version '$ver' is not X.Y.Z"
    grep -qE '^version\.workspace[[:space:]]*=[[:space:]]*true' -- "$wt/crates/apr-cli/Cargo.toml" \
        || refuse "apr-cli does not take the workspace version at C: its apr --version would not read $ver"
    blob=$(git -C "$wt" rev-parse -q --verify HEAD:scripts/release/models_t1.sh) || refuse "C has no scripts/release/models_t1.sh"
    sha9=$(git -C "$wt" rev-parse --short=9 HEAD)
    rm -rf -- "${out:?}"; mkdir -p -- "$out" "$work/target" || die "cannot create $out"
    : > "$work/foreign.log"
    # MODELS_T1_SCOPE=none: the nightly judges the FULL ladder, never a release scope or the standing
    # CRUX-smoke release policy. Its red rows are what the policy sends to tickets; a scoped nightly
    # would hide them.
    say "RUN models_t1.sh $ver at $sha9 (timeout ${tmo}s)"
    (cd "$wt" && export CARGO_TARGET_DIR="$work/target" MODELS_T1_SCOPE=none && exec timeout --kill-after=60 "$tmo" bash scripts/release/models_t1.sh "$ver" "$c" "$out") \
        > "$work/models-t1.log" 2>&1 9>&- &
    tpid=$!
    # timeout leads its own process group: any other models_t1.sh seen while it runs is foreign
    (while kill -0 "$tpid" 2> /dev/null; do ps_table | foreign_t1 "$tpid" >> "$work/foreign.log"; sleep 20; done) > /dev/null 2>&1 9>&- &
    spid=$!
    wait "$tpid"; rc=$?
    kill "$spid" 2> /dev/null; wait "$spid" 2> /dev/null
    b="$work/pending/$c"
    rm -rf -- "${b:?}"; mkdir -p -- "$b" || die "cannot create $b"
    tail -n 400 -- "$work/models-t1.log" > "$b/models-t1.log"
    for f in lambda gx10; do
        [ ! -f "$out/$f.json" ] || cp -- "$out/$f.json" "$b/$f.json"
        [ ! -f "$out/$f.log" ] || tail -n 200 -- "$out/$f.log" > "$b/$f.log.tail"
    done
    [ ! -f "$out/judge.log" ] || tail -n 200 -- "$out/judge.log" > "$b/judge.log"
    [ ! -s "$work/foreign.log" ] || head -n 1 -- "$work/foreign.log" > "$b/concurrent"
    seal "$b" "$c" "$sha9" "$ver" "scripts/release/models_t1.sh@$blob" "$rc" || die "cannot seal $b"
    git -C "$repo" worktree remove --force "$wt" > /dev/null 2>&1
    cls=$(vget state < "$b/verdict")
    say "RUN $cls at $sha9 ($ver, models_t1 rc $rc): $(vget reason < "$b/verdict")"
}

publish() { # publish REPO WORK
    local repo=$1 work=$2 p c n base idx tree new try msg
    no_token || refuse "a registry token is reachable (env, CARGO_HOME or \$HOME/.cargo): this producer holds none"
    exec 9> "$work/.lock" || die "cannot open $work/.lock"
    flock -n 9 || refuse "another models_nightly.sh holds $work/.lock"
    n=$(find "$work/pending" -mindepth 2 -maxdepth 2 -name verdict 2> /dev/null | wc -l)
    [ "$n" -gt 0 ] || { say "PUBLISH: nothing pending"; return 0; }
    export GIT_INDEX_FILE="$work/evidence.index"
    for try in 1 2 3; do
        git -C "$repo" fetch -q --no-tags origin "+refs/heads/$EV_BRANCH:refs/remotes/origin/$EV_BRANCH" 2> /dev/null || :
        base=$(git -C "$repo" rev-parse -q --verify "refs/remotes/origin/$EV_BRANCH^{commit}") || base=""
        rm -f -- "$GIT_INDEX_FILE"
        if [ -n "$base" ]; then git -C "$repo" read-tree "$base"; else git -C "$repo" read-tree --empty; fi || die "read-tree failed"
        idx="$work/INDEX"; msg="models nightly:"
        { [ -n "$base" ] && git -C "$repo" cat-file -p "$base:models/INDEX" 2> /dev/null; } > "$idx"
        for p in "$work"/pending/*/; do
            c=$(basename -- "$p")
            [[ $c =~ ^[0-9a-f]{40}$ ]] && [ -f "$p/verdict" ] || continue
            git -C "$repo" rm -r -q --cached --ignore-unmatch -- "models/$c" > /dev/null || die "index rm failed"
            for f in "$p"*; do
                new=$(git -C "$repo" hash-object -w -- "$f") \
                    && git -C "$repo" update-index --add --cacheinfo "100644,$new,models/$c/${f##*/}" || die "cannot stage $f"
            done
            awk -v c="$c" '$1 != c' "$idx" > "$idx.new" && printf '%s %s\n' "$c" "$(vget state < "$p/verdict")" >> "$idx.new" && mv -- "$idx.new" "$idx"
            msg="$msg ${c:0:9} $(vget state < "$p/verdict")"
        done
        tail -n "$KEEP" -- "$idx" > "$idx.new" && mv -- "$idx.new" "$idx"
        git -C "$repo" ls-files -- models/ | awk -F/ 'NF == 3 { print $2 }' | LC_ALL=C sort -u | while IFS= read -r c; do
            awk -v c="$c" '$1 == c { f = 1 } END { exit !f }' "$idx" || git -C "$repo" rm -r -q --cached -- "models/$c" > /dev/null
        done
        new=$(git -C "$repo" hash-object -w -- "$idx") && git -C "$repo" update-index --add --cacheinfo "100644,$new,models/INDEX" || die "cannot stage INDEX"
        tree=$(git -C "$repo" write-tree) || die "write-tree failed"
        if [ -n "$base" ]; then new=$(git -C "$repo" commit-tree "$tree" -p "$base" -m "$msg")
        else new=$(git -C "$repo" commit-tree "$tree" -m "$msg"); fi || die "commit-tree failed (is a git identity configured?)"
        if git -C "$repo" push -q origin "$new:refs/heads/$EV_BRANCH"; then
            rm -f -- "$GIT_INDEX_FILE"
            for p in "$work"/pending/*/; do rm -rf -- "${p:?}"; done
            say "PUBLISHED $EV_BRANCH ${new:0:9}:$msg"; return 0
        fi
        say "push rejected (try $try of 3): refetching $EV_BRANCH"
    done
    die "push to $EV_BRANCH rejected three times; the bundles stay pending for the next publish"
}

# ---------------------------------------------------------------- self-test ------------------------------------------
# fixture bundle: cfx DIR LAMBDA GX10 LOG-LINE... ; a receipt spec is ok | red | wrong | none | bare
ST_V=9.8.7; ST_S=abcdef012; ST_WANT="apr $ST_V ($ST_S)"
cfx() {
    local d=$1 h spec; mkdir -p -- "$d"; shift
    for h in lambda gx10; do
        spec=$1; shift
        case $spec in
            ok)    printf '{"apr_version":"apr %s (%s)","executed":3,"red":0}\n' "$ST_V" "$ST_S" > "$d/$h.json" ;;
            red)   printf '{"apr_version":"apr %s (%s)","executed":3,"red":2}\n' "$ST_V" "$ST_S" > "$d/$h.json" ;;
            wrong) printf '{"apr_version":"apr 0.0.1 (000000000)","executed":3,"red":0}\n' > "$d/$h.json" ;;
            bare)  printf '{"apr_version":"apr %s (%s)"}\n' "$ST_V" "$ST_S" > "$d/$h.json" ;;
            none)  ;;
        esac
    done
    printf '%s\n' "$@" > "$d/models-t1.log"
}
GO_LINE="MODELS GO on lambda and gx10 at $ST_S: the judge passed both receipts ($ST_WANT)"

self_test() {
    local tmp pass=0 fail=0 d o rc c1 c2 c3 work repo bare clone
    tmp=$(mktemp -d) || exit 3
    tmp=$(cd "$tmp" && pwd -P)
    export HOME="$tmp/home" CARGO_HOME="$tmp/cargo"; mkdir -p "$HOME" "$CARGO_HOME"; unset CARGO_REGISTRY_TOKEN GITHUB_OUTPUT
    while read -r o; do unset "$o"; done < <(env | awk -F "=" '$1 ~ /^CARGO_REGISTRIES_[A-Za-z0-9_]+_TOKEN$/ { print $1 }')
    export GIT_AUTHOR_NAME=st GIT_AUTHOR_EMAIL=st@invalid GIT_COMMITTER_NAME=st GIT_COMMITTER_EMAIL=st@invalid GIT_CONFIG_NOSYSTEM=1
    # row NAME EXPECT-RC NEEDLE FORBID -- CMD...
    row() {
        local name="$1" expect="$2" needle="$3" forbid="$4" o rc; shift 5
        o="$("$@" 2>&1)"; rc=$?
        if [ "$rc" != "$expect" ]; then printf '  BROKE %-48s expected exit %s got %s\n%s\n' "$name" "$expect" "$rc" "$o"; fail=$((fail + 1)); return 0; fi
        case "$o" in *"$needle"*) ;; *) printf '  BROKE %-48s never said: %s\n%s\n' "$name" "$needle" "$o"; fail=$((fail + 1)); return 0 ;; esac
        if [ -n "$forbid" ]; then case "$o" in *"$forbid"*) printf '  BROKE %-48s said: %s\n' "$name" "$forbid"; fail=$((fail + 1)); return 0 ;; esac; fi
        printf '  ok    %-48s exit=%s\n' "$name" "$rc"; pass=$((pass + 1))
    }
    cl() { local s; s=$(classify "$1" "$2" "$ST_V" "$ST_S"); printf "state=%s reason=%s\n" "${s%%$'\t'*}" "${s#*$'\t'}"; }
    G="state=green reason=" R="state=red reason=" N="state=not_measured reason="

    # -- classify: the verdict table
    d="$tmp/c/go"; cfx "$d" ok ok "$GO_LINE"
    row go_two_clean_bound_receipts_is_green 0 "$G" "" -- cl "$d" 0
    d="$tmp/c/go-gx10-missing"; cfx "$d" ok none "$GO_LINE"
    row go_line_without_gx10_receipt_is_not_measured 0 "$N" "$G" -- cl "$d" 0
    d="$tmp/c/go-bare"; cfx "$d" ok bare "$GO_LINE"
    row go_receipt_without_red_count_is_not_measured 0 "$N" "$G" -- cl "$d" 0
    d="$tmp/c/go-other-sha"; cfx "$d" ok ok "MODELS GO on lambda and gx10 at 999999999: the judge passed"
    row go_line_for_another_commit_is_not_measured 0 "$N" "$G" -- cl "$d" 0
    d="$tmp/c/go-rc1"; cfx "$d" ok ok "$GO_LINE"
    row go_line_with_nonzero_exit_is_not_green 0 "$N" "$G" -- cl "$d" 1
    d="$tmp/c/build"; cfx "$d" none ok "MODELS lambda NO-GO: no receipt -- BUILD-FAILED (rc 3)" "MODELS NO-GO: the judge found red or missing cells (rc 1):"
    row build_failure_is_red 0 "${R}lambda NO-GO: no receipt -- BUILD-FAILED" "" -- cl "$d" 1
    d="$tmp/c/notrel"; cfx "$d" ok none "MODELS gx10 NO-GO: no receipt -- NOT-THE-RELEASE: 'apr 1' (want 'apr 2') (rc 3)"
    row binary_not_c_is_red 0 "${R}gx10 NO-GO: no receipt -- NOT-THE-RELEASE" "" -- cl "$d" 1
    d="$tmp/c/wrongbin"; cfx "$d" wrong ok "MODELS lambda NO-GO: the receipt was measured by 'apr 0.0.1 (000000000)', not 'apr $ST_V ($ST_S)'"
    row receipt_by_another_binary_is_red 0 "${R}lambda NO-GO: the receipt was measured by" "" -- cl "$d" 1
    d="$tmp/c/judge-red"; cfx "$d" ok ok "MODELS NO-GO: the judge found red or missing cells (rc 1):"
    row judge_red_on_two_bound_receipts_is_red 0 "${R}the judge found red or missing cells on two bound" "" -- cl "$d" 1
    d="$tmp/c/ssh"; cfx "$d" ok none "MODELS gx10 NO-GO: no receipt -- unreachable over SSH (ssh rc 255)" "MODELS NO-GO: the judge found red or missing cells (rc 1):"
    row unreachable_gx10_is_not_measured 0 "${N}MODELS gx10 NO-GO: no receipt -- unreachable" "$R" -- cl "$d" 1
    d="$tmp/c/ssh-red"; cfx "$d" red none "MODELS gx10 NO-GO: no receipt -- unreachable over SSH (ssh rc 255)"
    row red_cells_on_one_host_are_red_alone 0 "${R}lambda receipt has red cells" "" -- cl "$d" 1
    d="$tmp/c/decline"; cfx "$d" ok ok "MODELS NO-GO: the judge DECLINED (rc 2), and a decline is not a pass: DEFERRED 1 row(s)"
    row judge_decline_or_defer_is_not_measured 0 "${N}MODELS NO-GO: the judge DECLINED" "$R" -- cl "$d" 1
    d="$tmp/c/ladder2"; cfx "$d" ok none "MODELS gx10 NO-GO: no receipt -- model_ladder.sh wrote no receipt (rc 2)"
    row ladder_decline_without_receipt_is_not_measured 0 "$N" "$R" -- cl "$d" 1
    d="$tmp/c/ladder1"; cfx "$d" ok none "MODELS gx10 NO-GO: no receipt -- model_ladder.sh wrote no receipt (rc 1)"
    row ladder_exit_1_without_receipt_is_red 0 "${R}gx10 NO-GO: no receipt -- model_ladder.sh wrote no receipt (rc 1)" "" -- cl "$d" 1
    d="$tmp/c/ladder137"; cfx "$d" ok none "MODELS gx10 NO-GO: no receipt -- model_ladder.sh wrote no receipt (rc 137)"
    row ladder_killed_without_receipt_is_not_measured 0 "$N" "$R" -- cl "$d" 1
    d="$tmp/c/env"; cfx "$d" ok none "MODELS gx10 NO-GO: no receipt -- ENV: 3 GiB usable under x, a fresh cuda release target needs 5 GiB -- refused before building (rc 4)"
    row disk_refusal_env_is_not_measured 0 "$N" "$R" -- cl "$d" 2
    d="$tmp/c/fetch"; cfx "$d" ok none "MODELS gx10 NO-GO: no receipt -- FETCH-FAILED: abc is not reachable from origin/main there (rc 3)"
    row gx10_fetch_failure_is_not_measured 0 "$N" "$R" -- cl "$d" 1
    d="$tmp/c/timeout"; cfx "$d" ok none
    row timeout_is_not_measured 0 "${N}models_t1 timed out (rc 124)" "" -- cl "$d" 124
    d="$tmp/c/timeout-red"; cfx "$d" red none
    row timeout_with_a_red_receipt_is_red 0 "${R}lambda receipt has red cells" "" -- cl "$d" 124
    d="$tmp/c/unknown"; cfx "$d" none none "something else"
    row unknown_exit_is_not_measured 0 "${N}models_t1 exit 1 with no NO-GO line" "" -- cl "$d" 1
    d="$tmp/c/concurrent"; cfx "$d" ok ok "MODELS NO-GO: the judge found red or missing cells (rc 1):"; echo "4242 bash x/models_t1.sh 0.70.2" > "$d/concurrent"
    row red_beside_a_concurrent_models_t1_is_not_measured 0 "${N}red while another models_t1.sh ran" "" -- cl "$d" 1

    # -- foreign_t1 and no_token
    row foreign_models_t1_is_seen 0 "77 bash scripts/release/models_t1.sh" "" -- foreign_t1 5 <<< $'77 bash scripts/release/models_t1.sh 0.70.2 abc out\n5 bash scripts/release/models_t1.sh 9.8.7 c out\n78 vim models_t1.sh.log'
    row own_group_and_lookalikes_are_not_foreign 0 "" "models_t1" -- foreign_t1 5 <<< $'5 bash scripts/release/models_t1.sh 9.8.7 c out\n78 tail -f out/models_t1.sh.log'
    row no_token_holds_in_a_clean_env 0 "" "" -- no_token
    row env_registry_token_refused 1 "" "" -- env CARGO_REGISTRY_TOKEN=x bash -c "$(declare -f no_token); no_token"
    row named_registry_token_refused 1 "" "" -- env CARGO_REGISTRIES_X_TOKEN=x bash -c "$(declare -f no_token); no_token"
    mkdir -p "$tmp/home2/.cargo"; : > "$tmp/home2/.cargo/credentials.toml"
    row home_credentials_refused_under_another_cargo_home 1 "" "" -- env HOME="$tmp/home2" bash -c "$(declare -f no_token); no_token"
    mkdir -p "$tmp/cargo3"; printf '[registry]\ntoken = "x"\n' > "$tmp/cargo3/config.toml"
    row config_token_refused 1 "" "" -- env CARGO_HOME="$tmp/cargo3" bash -c "$(declare -f no_token); no_token"

    # -- end to end: a fixture repo whose C carries a stub models_t1.sh, a bare origin, --run/--publish/--relay
    bare="$tmp/origin.git"; repo="$tmp/repo"; work="$tmp/work"; clone="$tmp/clone"
    git init -q --bare -b main "$bare" && git init -q -b main "$tmp/src" || exit 3
    mkdir -p "$tmp/src/scripts/release" "$tmp/src/crates/apr-cli" "$work"
    printf '[workspace]\nmembers = []\n\n[workspace.package]\nversion = "%s"\n' "$ST_V" > "$tmp/src/Cargo.toml"
    printf '[package]\nname = "apr-cli"\nversion.workspace = true\n' > "$tmp/src/crates/apr-cli/Cargo.toml"
    cat > "$tmp/src/scripts/release/models_t1.sh" <<'STUB'
#!/usr/bin/env bash
ver=$1; out=$3; sha9=$(git rev-parse --short=9 HEAD); want="apr $ver ($sha9)"
[ -z "${STUB_MARK:-}" ] || : > "$STUB_MARK"
echo "MODELS_T1_SCOPE=${MODELS_T1_SCOPE-unset}"
red=0; [ "${STUB_MODE:-green}" != red ] || red=2
[ "${STUB_MODE:-green}" != slow ] || exec sleep 30
for h in lambda gx10; do printf '{"apr_version":"%s","executed":3,"red":%s}\n' "$want" "$red" > "$out/$h.json"; echo "leg $h" > "$out/$h.log"; done
echo "judge ran" > "$out/judge.log"
if [ "$red" = 0 ]; then echo "MODELS GO on lambda and gx10 at $sha9: the judge passed both receipts ($want)"; exit 0; fi
echo "MODELS NO-GO: the judge found red or missing cells (rc 1):"; exit 1
STUB
    git -C "$tmp/src" add -A && git -C "$tmp/src" commit -q -m c1 && git -C "$tmp/src" push -q "$bare" main && git clone -q "$bare" "$repo" || exit 3
    c1=$(git -C "$tmp/src" rev-parse HEAD)
    : > "$tmp/ps.none"; export MODELS_NIGHTLY_PS_TABLE="$tmp/ps.none"   # hermetic: see ps_table
    nightly() { bash "$SCRIPT_PATH" "$@"; }
    row e2e_run_green 0 "RUN green at ${c1:0:9}" "" -- nightly --run --repo "$repo" --work "$work"
    row e2e_run_judges_the_full_ladder 0 "MODELS_T1_SCOPE=none" "" -- cat "$work/models-t1.log"
    row e2e_bundle_is_sealed 0 "OK" "FAILED" -- bash -c "cd '$work/pending/$c1' && sha256sum -c SHA256SUMS"
    row e2e_second_run_on_c_skips_pending 0 "SKIP: ${c1:0:9} is already measured green" "RUN " -- env STUB_MARK="$tmp/mark" bash "$SCRIPT_PATH" --run --repo "$repo" --work "$work"
    row e2e_skip_ran_no_measurer 1 "" "" -- test -e "$tmp/mark"
    row e2e_publish 0 "PUBLISHED $EV_BRANCH" "" -- nightly --publish --repo "$repo" --work "$work"
    row e2e_publish_cleared_pending 0 "" "verdict" -- find "$work/pending" -name verdict
    row e2e_publish_never_touched_main 0 "$c1" "" -- git -C "$bare" rev-parse refs/heads/main
    git clone -q "$bare" "$clone" || exit 3
    row e2e_relay_green 0 "RELAY green at ${c1:0:9}" "" -- bash -c "cd '$clone' && GITHUB_OUTPUT='$tmp/gho' bash '$SCRIPT_PATH' --relay --commit $c1 --out '$tmp/relay1'"
    row e2e_relay_wrote_github_output 0 "state=green" "" -- cat "$tmp/gho"
    row e2e_relay_green_gives_its_bundle 0 "bundle=$tmp/relay1/models/$c1" "" -- cat "$tmp/gho"
    row e2e_models_t1_root_holds_the_receipts 0 "" "" -- test -f "$tmp/relay1/models/$c1/lambda.json" -a -f "$tmp/relay1/models/$c1/gx10.json"
    row e2e_relay_plant_red 0 "RELAY red at ${c1:0:9}: PLANTED red" "RELAY green" -- bash -c "cd '$clone' && GITHUB_OUTPUT='$tmp/gho2' bash '$SCRIPT_PATH' --relay --commit $c1 --out '$tmp/relay2' --plant-red"
    row e2e_planted_red_gives_no_bundle 0 "bundle=" "bundle=/" -- cat "$tmp/gho2"
    row e2e_published_skip 0 "SKIP: ${c1:0:9} is already measured green" "RUN " -- nightly --run --repo "$repo" --work "$work"
    echo red >> "$tmp/src/Cargo.toml"; git -C "$tmp/src" commit -q -am c2 && git -C "$tmp/src" push -q "$bare" main || exit 3
    c2=$(git -C "$tmp/src" rev-parse HEAD)
    row e2e_run_red 0 "RUN red at ${c2:0:9}" "" -- env STUB_MODE=red bash "$SCRIPT_PATH" --run --repo "$repo" --work "$work"
    row e2e_publish_keeps_the_last_n 0 "PUBLISHED" "" -- env MODELS_NIGHTLY_KEEP=1 bash "$SCRIPT_PATH" --publish --repo "$repo" --work "$work"
    git -C "$clone" fetch -q origin main && git -C "$clone" checkout -q "$c2" || exit 3
    row e2e_relay_red 0 "RELAY red at ${c2:0:9}" "" -- bash -c "cd '$clone' && GITHUB_OUTPUT='$tmp/gho4' bash '$SCRIPT_PATH' --relay --commit $c2 --out '$tmp/relay3'"
    row e2e_relay_red_gives_its_bundle 0 "bundle=$tmp/relay3/models/$c2" "" -- cat "$tmp/gho4"
    row e2e_pruned_bundle_is_not_measured 0 "RELAY not_measured at ${c1:0:9}: no bundle" "" -- bash -c "cd '$clone' && bash '$SCRIPT_PATH' --relay --commit $c1 --out '$tmp/relay4'"
    row e2e_relay_on_unmeasured_c_is_not_measured 0 "RELAY not_measured" "RELAY green" -- bash -c "cd '$clone' && GITHUB_OUTPUT='$tmp/gho3' bash '$SCRIPT_PATH' --relay --commit $c2 --out '$tmp/relay5' --evidence '$tmp/absent'"
    row e2e_not_measured_gives_no_bundle 0 "state=not_measured" "bundle=/" -- cat "$tmp/gho3"
    echo slow >> "$tmp/src/Cargo.toml"; git -C "$tmp/src" commit -q -am c3 && git -C "$tmp/src" push -q "$bare" main || exit 3
    c3=$(git -C "$tmp/src" rev-parse HEAD)
    row e2e_timeout_is_not_measured 0 "RUN not_measured at ${c3:0:9}" "" -- env STUB_MODE=slow bash "$SCRIPT_PATH" --run --repo "$repo" --work "$work" --timeout 2
    row e2e_not_measured_is_measured_again 0 "RUN green at ${c3:0:9}" "SKIP" -- nightly --run --repo "$repo" --work "$work"
    row e2e_token_refused 3 "REFUSED: a registry token" "RUN " -- env CARGO_REGISTRY_TOKEN=x bash "$SCRIPT_PATH" --run --repo "$repo" --work "$work"
    row e2e_publish_token_refused 3 "REFUSED: a registry token" "PUBLISHED" -- env CARGO_REGISTRY_TOKEN=x bash "$SCRIPT_PATH" --publish --repo "$repo" --work "$work"
    mkdir -p "$tmp/fake"; printf 'sleep 30\n' > "$tmp/fake/models_t1.sh"
    bash "$tmp/fake/models_t1.sh" > /dev/null 2>&1 & bg=$!
    printf '4242 bash /r/scripts/release/models_t1.sh 1.2.3 c /o\n' > "$tmp/ps.one"
    row e2e_foreign_in_the_table_refused 3 "running (a release's models step shares the GPU hosts): 4242 bash /r/scripts" "RUN " -- env MODELS_NIGHTLY_PS_TABLE="$tmp/ps.one" bash "$SCRIPT_PATH" --run --repo "$repo" --work "$work" --commit "$c2"
    row e2e_foreign_models_t1_refused_by_ps 3 "REFUSED: another models_t1.sh is running" "RUN " -- env -u MODELS_NIGHTLY_PS_TABLE bash "$SCRIPT_PATH" --run --repo "$repo" --work "$work" --commit "$c2"
    kill "$bg" 2> /dev/null; wait "$bg" 2> /dev/null
    (exec 9> "$work/.lock"; flock 9; : > "$tmp/locked"; exec sleep 30) > /dev/null 2>&1 & bg=$!
    for _ in 1 2 3 4 5 6 7 8 9 10; do [ -e "$tmp/locked" ] && break; sleep 1; done
    row e2e_lock_held_refused 3 "REFUSED: another models_nightly.sh holds" "RUN " -- nightly --run --repo "$repo" --work "$work" --commit "$c2"
    kill "$bg" 2> /dev/null; wait "$bg" 2> /dev/null

    # -- verify: the relay's checks on a sealed bundle (the clone holds C1..C3)
    git -C "$clone" fetch -q origin main || exit 3
    d="$tmp/v/good"; mkdir -p "$d"; cfx "$d" none none
    rl() { (cd "$clone" && bash "$SCRIPT_PATH" --relay --commit "$1" --out "$tmp/rl" --evidence "$2"); }
    mkb() { # mkb NAME RC STATE-OVERRIDE ; a bundle at C1 with green-shaped evidence
        local b s9; b=$(realpath -m -- "$tmp/v/$1"); s9=$(git -C "$clone" rev-parse --short=9 "$c1")
        rm -rf -- "${b:?}"; mkdir -p "$b"
        printf '{"apr_version":"apr %s (%s)","executed":3,"red":0}\n' "$ST_V" "$s9" | tee "$b/lambda.json" > "$b/gx10.json"
        printf 'MODELS GO on lambda and gx10 at %s: the judge passed both receipts\n' "$s9" > "$b/models-t1.log"
        seal "$b" "$c1" "$s9" "$ST_V" "scripts/release/models_t1.sh@$(git -C "$clone" rev-parse "$c1:scripts/release/models_t1.sh")" "$2"
    }
    reseal() { (cd "$1" && find . -maxdepth 1 -type f ! -name SHA256SUMS -printf '%f\n' | LC_ALL=C sort | xargs sha256sum -- > SHA256SUMS); }
    mkb good 0
    row verify_sealed_green_bundle 0 "RELAY green" "" -- rl "$c1" "$tmp/v/good"
    mkb tamper 0; sed -i 's/"red":0/"red":5/' "$tmp/v/tamper/gx10.json"
    row verify_tampered_receipt_is_not_measured 0 "fails its own SHA256SUMS" "RELAY green" -- rl "$c1" "$tmp/v/tamper"
    mkb extra 0; echo x > "$tmp/v/extra/unlisted"
    row verify_unlisted_file_is_not_measured 0 "does not list" "RELAY green" -- rl "$c1" "$tmp/v/extra"
    mkb other 0
    row verify_bundle_for_another_commit_is_not_measured 0 "the bundle measured ${c1:0:9}, not ${c2:0:9}" "RELAY green" -- rl "$c2" "$tmp/v/other"
    mkb ver 0; sed -i 's/^version=.*/version=1.2.3/' "$tmp/v/ver/verdict"; reseal "$tmp/v/ver"
    row verify_other_version_is_not_measured 0 "measured version 1.2.3" "RELAY green" -- rl "$c1" "$tmp/v/ver"
    mkb entry 0; sed -i 's/^entry=.*/entry=scripts\/release\/models_t1.sh@0000/' "$tmp/v/entry/verdict"; reseal "$tmp/v/entry"
    row verify_other_entry_point_is_not_measured 0 "not measured by C's models_t1.sh" "RELAY green" -- rl "$c1" "$tmp/v/entry"
    mkb replay 0; sed -i 's/"red":0/"red":1/' "$tmp/v/replay/lambda.json"; reseal "$tmp/v/replay"
    row verify_replay_disagreeing_is_not_measured 0 "the replay says red, the bundle recorded green" "RELAY green" -- rl "$c1" "$tmp/v/replay"
    mkb rc1 1
    row verify_recorded_not_measured_stays 0 "RELAY not_measured" "RELAY green" -- rl "$c1" "$tmp/v/rc1"

    # -- tickets: every red nightly row opens or updates its ticket, through a gh stub that logs each call
    cat > "$tmp/gh" <<'GH'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$FXGH_LOG"
case "$1 $2" in
    "issue list") [ "${FXGH_FAIL:-}" != list ] || exit 1; cat -- "$FXGH_LIST" ;;
    "issue view") cat -- "$FXGH_VIEW" ;;
    "issue comment") [ "${FXGH_FAIL:-}" != comment ] || exit 1 ;;
    "issue create") echo "https://github.invalid/o/r/issues/77" ;;
    *) exit 9 ;;
esac
GH
    chmod +x "$tmp/gh"
    tkb() { # tkb NAME STATE LAMBDA-RUNGS-JSON -> a bundle dir
        local d="$tmp/tk/$1"; mkdir -p -- "$d"
        printf 'state=%s\nreason=fixture %s\n' "$2" "$2" > "$d/verdict"
        [ -z "$3" ] || printf '{"red":1,"rungs":%s}\n' "$3" > "$d/lambda.json"
    }
    tk() { # tk BUNDLE LIST-JSON VIEW-JSON [FAIL] -> tickets' output, then the out file and the gh calls
        printf '%s\n' "$2" > "$tmp/tk-list"; printf '%s\n' "$3" > "$tmp/tk-view"; : > "$tmp/tk-log"
        ( export MODELS_NIGHTLY_GH="$tmp/gh" FXGH_LOG="$tmp/tk-log" FXGH_LIST="$tmp/tk-list" FXGH_VIEW="$tmp/tk-view" FXGH_FAIL="${4:-}"
          tickets "$1" "$c1" "$tmp/tk-out" ); local rc=$?
        printf 'OUT %s\n' "$(tr '\t\n' '| ' < "$tmp/tk-out")"; printf 'GH %s\n' "$(tr '\n' ';' < "$tmp/tk-log")"
        return "$rc"
    }
    R1='[{"id":"fx-1","file":"a.gguf","green":false},{"id":"fx-2","file":"b.gguf","green":true}]'
    tkb green green "$R1"
    row tickets_green_bundle_touches_no_issue 0 "TICKETS none" "issue" -- tk "$tmp/tk/green" '[]' '{}'
    tkb red red "$R1"
    row tickets_red_rung_opens_its_issue 0 "OUT lambda|fx-1|#77 " "fx-2" -- tk "$tmp/tk/red" '[]' '{}'
    row tickets_open_issue_gets_a_comment 0 "TICKET updated #5" "issue create" -- tk "$tmp/tk/red" \
        '[{"number":4,"title":"models-nightly red: fx-10 on lambda"},{"number":5,"title":"models-nightly red: fx-1 on lambda"}]' '{"body":"x","comments":[]}'
    row tickets_same_commit_is_not_commented_twice 0 "TICKET kept #5" "issue comment" -- tk "$tmp/tk/red" \
        '[{"number":5,"title":"models-nightly red: fx-1 on lambda"}]' "{\"body\":\"x\",\"comments\":[{\"body\":\"models-nightly@$c1\"}]}"
    tkb lane red ""
    row tickets_red_with_no_rung_is_a_lane_row 0 "models-nightly red: lane on all" "" -- tk "$tmp/tk/lane" '[]' '{}'
    row tickets_failed_search_fails_the_step 2 "NOT-MEASURED: the issue search" "TICKET opened" -- tk "$tmp/tk/red" '[]' '{}' list
    row tickets_failed_comment_fails_the_step 2 "NOT-MEASURED: commenting on #5" "OUT lambda" -- tk "$tmp/tk/red" \
        '[{"number":5,"title":"models-nightly red: fx-1 on lambda"}]' '{"body":"x","comments":[]}' comment

    printf -- '--- %s/%s rows ---\n' "$pass" "$((pass + fail))"
    rm -rf -- "${tmp:?}"
    [ "$fail" -eq 0 ]
}

# each planted mutant must change the file, still parse, and turn at least one row RED
MN_MUTANTS='m01_red_receipt_ignored	s/receipt_red "\$b\/\$h.json"; then red=/false; then red=/
m02_green_without_go_line	s/\[ "\$clean" = 2 \] && grep -qE "\^MODELS GO/[ "$clean" = 2 ] || grep -qE "^MODELS GO/
m03_judge_red_without_two_bound	s/if \[ "\$bound" = 2 \] && grep -qE/if grep -qE/
m04_ladder_decline_counted_red	s/\(rc \[01\]/(rc [012]/
m05_build_failure_not_red	s/[(]BUILD-FAILED[|]NOT-THE-RELEASE[|]/(NOT-THE-RELEASE|/
m06_sums_unchecked	s/sha256sum --strict --quiet -c SHA256SUMS > \/dev\/null 2>&1/true/
m07_commit_mismatch_ignored	s/\[ "\$v" = "\$c" \] [|][|]/true ||/
m08_replay_unchecked	s/\[ "\$\{replay%%\$.\\t.\*\}" = "\$rec" \] [|][|]/true ||/
m09_home_credentials_ignored	/\[ ! -r "\$HOME\/.cargo\/credentials" \]/d
m10_plant_ignored	s/\[ -z "\$4" \] [|][|] \{ reason="PLANTED/true || { reason="PLANTED/
m11_measured_c_measured_again	s/^    ! already_measured "\$repo" "\$c" "\$work" [|][|] return 0$/    :/
m12_foreign_run_ignored	s/\[ -z "\$f" \] [|][|] refuse "another models_t1/true || refuse "another models_t1/
m13_concurrent_red_kept	s/if \[ -n "\$red" \] && \[ -f "\$b\/concurrent" \]; then/if false; then/
m14_timeout_counted_red	s/124[|]137[)] line="models_t1 timed out/124|137) printf "red\\tx\\n"; return 0; line="/
m15_version_unchecked	s/\[ -n "\$ver" \] && \[ "\$v" = "\$ver" \] [|][|]/true ||/
m16_entry_unchecked	s/\[ -n "\$blob" \] && \[ "\$v" = "scripts\/release\/models_t1.sh@\$blob" \] \\$/true \\/
m17_unlisted_file_ignored	s/\[ "\$listed" = "\$have" \] [|][|]/true ||/
m18_token_guard_dropped_from_run	0,/^    no_token [|][|] refuse/s/^    no_token [|][|] refuse/    true || refuse/
m19_prune_kept_everything	s/^        tail -n "\$KEEP" -- "\$idx"/        cat -- "$idx"/
m20_real_ps_never_read	s/else ps -eo pgid=,args=; fi/else :; fi/
m21_plant_feeds_readiness	s/\[ -n "\$4" \] [|][|] bundle=\$ev/bundle=$ev/
m22_unmeasured_feeds_readiness	s/case \$state in green[|]red[)] \[/case $state in *) [/
m23_nightly_judges_a_scope	s/ MODELS_T1_SCOPE=none \&\&/ \&\&/
m24_green_rungs_ticketed	s/select\(.green == false\)/select(.green != null)/
m25_existing_issue_ignored	s/^        if \[ -n "\$n" \]; then$/        if false; then/
m26_commented_every_slot	s/grep -qF -- "\$mark"; then say/false; then say/
m27_ticket_failure_passes	s/^    \[ "\$st" = 0 \] [|][|] die "a ticket/    true || die "a ticket/'
mutants() {
    local tmp pass=0 fail=0 name expr o rc
    tmp=$(mktemp -d) || exit 3
    cp -- "$SCRIPT_PATH" "$tmp/models_nightly.sh"
    if ! bash "$tmp/models_nightly.sh" --self-test > /dev/null 2>&1; then printf '  BROKE %-44s the unmutated script is not green in the mutant dir\n' baseline; rm -rf -- "${tmp:?}"; return 1; fi
    while IFS=$'\t' read -r name expr; do
        [ -n "$name" ] || continue
        sed -E -e "$expr" "$SCRIPT_PATH" > "$tmp/models_nightly.sh"
        if cmp -s "$SCRIPT_PATH" "$tmp/models_nightly.sh"; then printf '  BROKE %-44s changed nothing: its pattern no longer matches\n' "$name"; fail=$((fail + 1)); continue; fi
        if ! bash -n "$tmp/models_nightly.sh" 2> /dev/null; then printf '  BROKE %-44s does not parse\n' "$name"; fail=$((fail + 1)); continue; fi
        o=$(bash "$tmp/models_nightly.sh" --self-test 2>&1); rc=$?
        case "$rc:$o" in
            0:*) printf '  BROKE %-44s SURVIVED: the case table stayed green\n' "$name"; fail=$((fail + 1)) ;;
            *"  BROKE "*) printf '  ok    %-44s killed, %s row(s) broke\n' "$name" "$(printf '%s\n' "$o" | awk '/^  BROKE /{n++} END{print n+0}')"; pass=$((pass + 1)) ;;
            *) printf '  BROKE %-44s exit %s with no broken row: not a kill\n' "$name" "$rc"; fail=$((fail + 1)) ;;
        esac
    done < <(printf '%s\n' "$MN_MUTANTS")
    printf -- '--- %s/%s mutants killed ---\n' "$pass" "$((pass + fail))"
    rm -rf -- "${tmp:?}"
    [ "$fail" -eq 0 ]
}

# ---------------------------------------------------------------- main -----------------------------------------------
MODE=${1:-}; [ $# -eq 0 ] || shift
case $MODE in
    --self-test) self_test; exit $? ;;
    --mutants) mutants; exit $? ;;
    --run|--publish|--relay|--tickets) ;;
    *) echo "usage: models_nightly.sh --run|--publish|--relay|--tickets|--self-test|--mutants (see the header)" >&2; exit 2 ;;
esac
REPO=""; WORK=""; COMMIT=""; OUT=""; EVIDENCE=""; PLANT=""
while [ $# -gt 0 ]; do
    case $1 in
        --repo|--work|--commit|--out|--evidence|--timeout)
            [ $# -ge 2 ] || { echo "models_nightly: $1 needs a value" >&2; exit 2; }
            case $1 in
                --repo) REPO=$2 ;; --work) WORK=$2 ;; --commit) COMMIT=$2 ;;
                --out) OUT=$2 ;; --evidence) EVIDENCE=$2 ;; --timeout) TMO=$2 ;;
            esac
            shift 2 ;;
        --plant-red) PLANT=1; shift ;;
        *) echo "models_nightly: unknown argument '$1'" >&2; exit 2 ;;
    esac
done
[[ $TMO =~ ^[1-9][0-9]*$ ]] || { echo "models_nightly: --timeout '$TMO' is not a positive integer" >&2; exit 2; }
case $MODE in
    --run|--publish)
        [ -n "$REPO" ] && [ -n "$WORK" ] || { echo "models_nightly: $MODE needs --repo and --work" >&2; exit 2; }
        git -C "$REPO" rev-parse --git-dir > /dev/null 2>&1 || { echo "models_nightly: '$REPO' is not a git repository" >&2; exit 2; }
        mkdir -p -- "$WORK" && WORK=$(cd "$WORK" && pwd -P) || { echo "models_nightly: cannot use --work '$WORK'" >&2; exit 2; }
        if [ "$MODE" = --run ]; then run "$REPO" "$WORK" "$COMMIT" "$TMO"; else publish "$REPO" "$WORK"; fi ;;
    --relay)
        [[ $COMMIT =~ ^[0-9a-f]{40}$ ]] && [ -n "$OUT" ] || { echo "models_nightly: --relay needs --commit <40-hex sha> and --out" >&2; exit 2; }
        relay "$COMMIT" "$OUT" "$EVIDENCE" "$PLANT" ;;
    --tickets)
        [[ $COMMIT =~ ^[0-9a-f]{40}$ ]] && [ -n "$OUT" ] && [ -d "$EVIDENCE" ] || { echo "models_nightly: --tickets needs --commit <40-hex sha>, --evidence <bundle dir> and --out <file>" >&2; exit 2; }
        tickets "$EVIDENCE" "$COMMIT" "$OUT" ;;
esac
