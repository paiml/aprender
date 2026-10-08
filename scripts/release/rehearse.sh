#!/usr/bin/env bash
# rehearse.sh — the nightly release rehearsal (APR-071 spec row B1, H10; E1 #3998).
#
# B1 (rule a, C332): every night on main's head, the same script and the same policy file as release
# day run everything except the tag push and the uploads. Green means exit 0, every lane measured on
# that night's commit, no rerun. The 3 nights before release day are green; a red night resets the
# count (nightly_greens.sh counts this job's scheduled runs on main).
#
# H10: a rehearsal makes no write outside its own state directory. Enforced, not promised:
#   * every stage runs with the write guard (lib_write_guard.sh) first on PATH. A write is refused
#     (exit 97) and recorded; one WRITE row makes the night red;
#   * git's `origin` is a bare repository INSIDE the state dir, pinned at the night's commit C, so a
#     `git fetch origin main` in any release script sees C all night, and no git remote reaches GitHub;
#   * CARGO_HOME is a state-dir copy of the real one WITHOUT credentials.toml, so `cargo publish`
#     could not authenticate even if a call got past the guard;
#   * TMPDIR and RELEASE_AP (the release scripts' state dir) are inside it.
#   The one shared thing is cargo's download cache (CARGO_HOME/registry and /git are symlinks to the
#   real ones): a crate download lands there as on any build. That is a cache fill, never a release
#   write -- nothing in it is read by GitHub, crates.io or another host.
#
# THE NIGHT. The release script's own stages, in release-day order (STAGES below). A stage whose
# producer stage did not finish green is `unreached`, never run on made-up inputs. Release day stops
# at the first red; the rehearsal keeps going wherever a stage's inputs exist, so one night names
# every red it can reach.
#
# THE D-LEDGER (spec §1, D1..D7; §4 note under B5: "The rehearsal's first night must be red, naming
# every one of D1 to D7 that is still open. A rehearsal that is green while one of them is open is
# not the release path."). Each D-row names the stage(s) its gate line sits in and the line the
# release scripts print when they stop on it. A D-row is CLEAR only when every one of its stages ran
# green on the night (and, for D6 and D7, the night's own trace or bump commit shows the fix). An
# unreached stage clears nothing. Every D-row that is not CLEAR is named in the verdict.
#
# USAGE
#   rehearse.sh --run --state DIR [--commit SHA] [--version V] [--source REPO] [--stages a,b]
#       DIR must not exist (or be empty). SHA defaults to the source's HEAD; V to the lowest open
#       milestone above the workspace version. --stages runs a subset (a development aid: the
#       stages it skips are unreached, so such a night is never green).
#   rehearse.sh --judge DIR      the verdict of a finished night (the --run prints it too)
#   rehearse.sh --classify TOOL CWD [ARG...]   READ or WRITE <why>, as the guard decides it
#   rehearse.sh --selftest       the case table: guard must-refuse/must-pass, the stub end to end,
#                                the judge on fixture nights (both polarities)
#   rehearse.sh --mutants        each planted mutant must turn --selftest RED
#
# EXIT  0 GREEN · 1 RED · 2 not_measured (no night to judge, or the box cannot run one) · 3 caller error
set -uo pipefail

SCRIPT_PATH=$(realpath -- "${BASH_SOURCE[0]}") || exit 2
SCRIPT_DIR=$(dirname -- "$SCRIPT_PATH")
REPO=paiml/aprender
GUARDED="git gh cargo ssh scp sftp curl"
# The models a bump commit's Co-Authored-By trailer may name: the ones the fleet runs (D7).
FLEET_MODELS='Claude Opus 5\.5|Claude Sonnet 5\.5|Claude Haiku 5\.5'
# The paragraph a person writes into the CHANGELOG between `prepare_bump.sh V` and `--ship`.
MARK='<!-- one-paragraph summary of the train: EDIT BEFORE --ship -->'
SUMMARY='Release rehearsal (B1): no train summary. This bump is never pushed.'

# shellcheck source=scripts/release/lib_write_guard.sh
. "$SCRIPT_DIR/lib_write_guard.sh" || exit 2

# STAGES: name | needs (a stage that must have finished green, or -) | command, run from the clone root.
# $V is the train's version. ap:<step> is autopilot.sh run for that one step: its setup is re-entrant.
stages_table() {
    cat <<'STAGES'
t2|-|bash scripts/release/t2_preflight.sh "$V"
bump|-|bash scripts/release/prepare_bump.sh "$V"
summary|bump|stage_summary
ship|summary|bash scripts/release/prepare_bump.sh "$V" --ship
ap:deep|ship|bash scripts/release/autopilot.sh "$V" rehearsal deep deep
ap:dogfood|ship|bash scripts/release/autopilot.sh "$V" rehearsal dogfood dogfood
ap:models|ship|bash scripts/release/autopilot.sh "$V" rehearsal models models
ap:readiness|ap:models|bash scripts/release/autopilot.sh "$V" rehearsal readiness readiness
ap:tag|ship|bash scripts/release/autopilot.sh "$V" rehearsal tag tag
ap:cleanroom|ap:tag|bash scripts/release/autopilot.sh "$V" rehearsal cleanroom cleanroom
ap:assets|ap:tag|bash scripts/release/autopilot.sh "$V" rehearsal assets assets
ap:preflight|ap:tag|bash scripts/release/autopilot.sh "$V" rehearsal preflight preflight
ap:publish|ap:tag|bash scripts/release/autopilot.sh "$V" rehearsal publish publish
ap:dryrun|ap:tag|bash scripts/release/autopilot.sh "$V" rehearsal dryrun dryrun
cascade|ap:preflight|bash scripts/cascade-publish.sh --rehearse
STAGES
}

# DLEDGER: row | stages its gate line sits in (comma list) | the line the scripts print when they stop
# on it (ERE) | the defect, as the spec states it. D6 and D7 also have a trace check (dledger_extra).
dledger_table() {
    cat <<'DLEDGER'
D1|ap:preflight,cascade|FAIL +R4 HEAD .* is not an ancestor of origin/release/|preflight R4 requires origin/release/<V>, which no script creates
D2|t2,ap:dogfood|\[FAIL\] declared:check_model_ladder|the dogfood ladder gate runs with no CRUX receipt dir, beside the models lane that writes them
D3|ap:tag|NOT_MEASURED: no completed .* run on [0-9a-f]+ or a version-only parent|the coverage gate's version-only rule does not admit evidence/crux/<V>/prompt-certification*.json
D4|cascade|FAIL +R7 |the cascade's own preflight runs without the CRUX receipts
D5|ship|no T-2 GO receipt for origin/main|prepare_bump.sh --ship needs a T-2 GO that cannot exist
D6|bump,ship|-|the freeze (carry_milestone_items.sh) is a hand step: the bump script never runs it
D7|ship|-|the bump commit's trailer names a model the fleet does not use
DLEDGER
}

die3() { printf 'rehearse.sh: caller error: %s\n' "$*" >&2; exit 3; }

# ------------------------------------------------------------------ the night --
# install_guard STATE -> stubs + a credential-free CARGO_HOME under STATE; prints the env file path
install_guard() {
    local st=$1 ch bin real t f
    ch="$st/cargo-home"; bin="$ch/bin"
    mkdir -p "$bin" "$st/tmp" "$st/ap" "$st/logs" || return 2
    : > "$st/calls.tsv" || return 2
    local src_home=${CARGO_HOME:-$HOME/.cargo}
    for f in registry git config.toml config; do
        [ -e "$src_home/$f" ] && ln -sfn "$src_home/$f" "$ch/$f"
    done
    if [ -d "$src_home/bin" ]; then
        for f in "$src_home"/bin/*; do
            [ -e "$f" ] || continue
            ln -sfn "$f" "$bin/${f##*/}"
        done
    fi
    {
        printf 'export WG_STATE=%q WG_CALLS=%q\n' "$st" "$st/calls.tsv"
        for t in $GUARDED; do
            real=$(PATH="$src_home/bin:$PATH" type -P "$t") || real=""
            printf 'export WG_REAL_%s=%q\n' "$(printf '%s' "$t" | tr '[:lower:]' '[:upper:]')" "$real"
            rm -f "${bin:?}/${t:?}"
            printf '#!/usr/bin/env bash\n# rehearsal write guard stub for %s (rehearse.sh install_guard)\n. %q || exit 98\nwg_stub %s "$@"\n' \
                "$t" "$SCRIPT_DIR/lib_write_guard.sh" "$t" > "$bin/$t" && chmod +x "$bin/$t" || return 2
        done
        printf 'export CARGO_HOME=%q TMPDIR=%q RELEASE_AP=%q RELEASE_REHEARSAL=1\n' "$ch" "$st/tmp" "$st/ap"
        printf 'export PATH=%q:"$PATH"\n' "$bin"
        printf 'unset CARGO_REGISTRY_TOKEN\n'
    } > "$st/guard.env" || return 2
    printf '%s\n' "$st/guard.env"
}

# stage_summary: the CHANGELOG paragraph a person writes between the bump and --ship, as a fixed one
stage_summary() {
    local f="$RELEASE_AP/bump/CHANGELOG.md" s
    [ -f "$f" ] || { echo "no $f: the bump stage left no tree"; return 1; }
    s=$(cat -- "$f"; printf x) || return 1
    s=${s%x}
    [[ $s == *"$MARK"* ]] || { echo "the CHANGELOG has no placeholder to replace"; return 1; }
    printf '%s' "${s/"$MARK"/"$SUMMARY"}" > "$f" || return 1
    echo "summary: placeholder replaced with the rehearsal's fixed paragraph"
}

# default_version SOURCE -> the next release day's version: the lowest V above the workspace version
# with an open "EPIC: release train V" issue -- the epic the release scripts themselves resolve
# (lib_release_params.sh release_epic_number). A version with no such epic cannot be cut by them.
default_version() {
    local cur
    cur=$(sed -n '/^\[workspace.package\]/,/^\[/s/^version *= *"\([^"]*\)".*/\1/p' "$1/Cargo.toml" | head -1)
    [ -n "$cur" ] || return 1
    gh issue list --repo "$REPO" --label epic --state open --limit 200 --json title --jq '.[].title' 2>/dev/null \
        | sed -nE 's/^EPIC: release train ([0-9]+\.[0-9]+\.[0-9]+)( .*)?$/\1/p' | { cat; printf '%s\n' "$cur"; } \
        | sort -uV | awk -v c="$cur" 'f { print; exit } $0 == c { f = 1 }' | grep .
}

run_night() {
    local st="" commit="" v="" src="" only="" env name needs cmd rc start
    while [ $# -gt 0 ]; do
        case $1 in
            --state) st=${2:-}; shift 2 ;;
            --commit) commit=${2:-}; shift 2 ;;
            --version) v=${2:-}; shift 2 ;;
            --source) src=${2:-}; shift 2 ;;
            --stages) only=${2:-}; shift 2 ;;
            *) die3 "unknown option $1" ;;
        esac
    done
    [ -n "$st" ] || die3 "--state DIR is required"
    [ ! -e "$st" ] || [ -z "$(ls -A -- "$st" 2>/dev/null)" ] || die3 "--state $st exists and is not empty"
    mkdir -p "$st" || exit 2
    st=$(realpath -- "$st") || exit 2
    src=${src:-$(cd "$SCRIPT_DIR/../.." && pwd)}
    commit=$(git -C "$src" rev-parse --verify "${commit:-HEAD}^{commit}") || die3 "no commit ${commit:-HEAD} in $src"
    [ -n "$v" ] || v=$(default_version "$src") || true
    [[ $v =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "not_measured: no train version (pass --version)"; exit 2; }
    # origin: a bare repo inside the state dir, main pinned at C. The clone fetches from it. Both borrow
    # the source's objects read-only (--shared: alternates, no copy); every new object lands in the state dir.
    local srcgit
    srcgit=$(git -C "$src" rev-parse --path-format=absolute --git-common-dir) || exit 2
    git clone -q --bare --shared -- "$srcgit" "$st/origin.git" || exit 2
    git -C "$st/origin.git" update-ref refs/heads/main "$commit" || exit 2
    git clone -q --shared -- "$st/origin.git" "$st/clone" || exit 2
    git -C "$st/clone" checkout -q --detach "$commit" || exit 2
    git -C "$st/clone" config user.name "release rehearsal" && git -C "$st/clone" config user.email "rehearsal@invalid" || exit 2
    env=$(install_guard "$st") || exit 2
    printf 'C=%s\nV=%s\nSOURCE=%s\nSTARTED=%s\n' "$commit" "$v" "$src" "$(date -u +%FT%TZ)" > "$st/night.env"
    printf 'stage\trc\tcommit\tseconds\n' > "$st/stages.tsv"
    while IFS='|' read -r name needs cmd; do
        [ -n "$name" ] || continue
        if [ -n "$only" ] && [[ ",$only," != *",$name,"* ]]; then continue; fi
        if [ "$needs" != - ] && ! awk -F'\t' -v n="$needs" '$1 == n && $2 == 0 { f = 1 } END { exit !f }' "$st/stages.tsv"; then
            printf '%s\tunreached\t-\t0\n' "$name" >> "$st/stages.tsv"; continue
        fi
        start=$(date +%s)
        ( cd "$st/clone" || exit 2
          # shellcheck disable=SC1090
          . "$env" || exit 2
          export WG_STAGE=$name V=$v
          eval "$cmd" ) > "$st/logs/${name//:/_}.log" 2>&1 < /dev/null
        rc=$?
        printf '%s\t%s\t%s\t%s\n' "$name" "$rc" "$(stage_commit "$st" "$name" "$commit")" "$(( $(date +%s) - start ))" >> "$st/stages.tsv"
    done < <(stages_table)
    record_bump "$st" "$commit"
    judge "$st"
}

# stage_commit STATE STAGE C -> the commit the stage measured: C before the bump, the bump commit after
stage_commit() {
    case $2 in
        t2|bump|summary|ship) printf '%s' "$3" ;;
        *) git -C "$1/ap/bump" rev-parse HEAD 2>/dev/null || printf '%s' - ;;
    esac
}

# record_bump STATE C -> bump.msg (the bump commit's message) and bump.parent, when the ship stage made one
record_bump() {
    local b="$1/ap/bump" h
    h=$(git -C "$b" rev-parse HEAD 2>/dev/null) || return 0
    [ "$h" != "$2" ] || return 0
    git -C "$b" log -1 --format=%B "$h" > "$1/bump.msg" 2>/dev/null
    git -C "$b" rev-parse "$h^" > "$1/bump.parent" 2>/dev/null
    printf '%s\n' "$h" > "$1/bump.commit"
}

# ------------------------------------------------------------------ the judge --
# judge STATE -> the verdict on stdout; rc 0 GREEN, 1 RED, 2 not_measured
judge() {
    local st=$1 c v reds=0 name rc commit sec row stg sig what
    [ -f "$st/night.env" ] && [ -f "$st/stages.tsv" ] && [ -f "$st/calls.tsv" ] \
        || { echo "not_measured: $st holds no night (night.env, stages.tsv, calls.tsv)"; return 2; }
    c=$(sed -n 's/^C=//p' "$st/night.env"); v=$(sed -n 's/^V=//p' "$st/night.env")
    [ -n "$c" ] || { echo "not_measured: night.env names no commit"; return 2; }
    local bumpc="" bparent=""
    [ -f "$st/bump.commit" ] && bumpc=$(cat "$st/bump.commit")
    [ -f "$st/bump.parent" ] && bparent=$(cat "$st/bump.parent")
    printf 'REHEARSAL %s on %s\n' "$v" "$c"
    while IFS='|' read -r name _ _; do
        [ -n "$name" ] || continue
        row=$(awk -F'\t' -v n="$name" 'NR > 1 && $1 == n { r = $0 } END { print r }' "$st/stages.tsv")
        if [ -z "$row" ]; then printf 'RED   stage %-13s not run\n' "$name"; reds=$((reds + 1)); continue; fi
        IFS=$'\t' read -r _ rc commit sec <<< "$row"
        case $rc in
            0) if [ "$commit" = "$c" ] || { [ -n "$bumpc" ] && [ "$commit" = "$bumpc" ] && [ "$bparent" = "$c" ]; }; then
                   printf 'ok    stage %-13s %ss\n' "$name" "$sec"
               else
                   printf 'RED   stage %-13s measured on %s, not on %s or its bump\n' "$name" "$commit" "$c"; reds=$((reds + 1))
               fi ;;
            unreached) printf 'RED   stage %-13s unreached (its producer stage was not green)\n' "$name"; reds=$((reds + 1)) ;;
            *) printf 'RED   stage %-13s exit %s: %s\n' "$name" "$rc" "$(stage_tail "$st" "$name")"; reds=$((reds + 1)) ;;
        esac
    done < <(stages_table)
    local writes
    writes=$(awk -F'\t' '$3 == "WRITE" || $3 == "MISSING"' "$st/calls.tsv")
    if [ -n "$writes" ]; then
        printf '%s\n' "$writes" | awk -F'\t' '{ printf "RED   %s in stage %s: %s (by %s)\n", $3, $1, $4, $5 }'
        reds=$((reds + $(printf '%s\n' "$writes" | wc -l)))
    fi
    while IFS='|' read -r row stg sig what; do
        [ -n "$row" ] || continue
        dl=$(dledger_row "$st" "$row" "$stg" "$sig")
        case $dl in
            CLEAR*) printf 'ok    %s clear\n' "$row" ;;
            *) printf 'RED   %s %s -- %s\n' "$row" "$dl" "$what"; reds=$((reds + 1)) ;;
        esac
    done < <(dledger_table)
    if [ "$reds" -eq 0 ]; then echo "VERDICT GREEN: every stage green on $c, 0 writes, D1..D7 clear"; return 0; fi
    echo "VERDICT RED: $reds red line(s)"
    return 1
}

stage_tail() {
    local f="$1/logs/${2//:/_}.log"
    [ -f "$f" ] || { printf 'no log'; return; }
    grep -E '^(STOP|FAIL|RED|die|ERROR|⛔)' "$f" | tail -1 | cut -c1-200 | grep . || tail -1 "$f" | cut -c1-200
}

# dledger_row STATE ROW STAGES SIG -> "CLEAR" | "OPEN <evidence>" | "NOT CLEARED <why>"
dledger_row() {
    local st=$1 row=$2 sig=$4 s rc hit="" notrun=""
    local -a ss
    IFS=, read -r -a ss <<< "$3"
    for s in "${ss[@]}"; do
        rc=$(awk -F'\t' -v n="$s" 'NR > 1 && $1 == n { r = $2 } END { print r }' "$st/stages.tsv")
        if [ "$sig" != - ] && [ -f "$st/logs/${s//:/_}.log" ]; then
            hit=$(grep -m1 -E -- "$sig" "$st/logs/${s//:/_}.log" | cut -c1-160)
            [ -n "$hit" ] && { echo "OPEN in $s: $hit"; return; }
        fi
        [ "$rc" = 0 ] || notrun="$notrun $s(${rc:-not run})"
    done
    case $row in
        D6) dledger_d6 "$st" "$notrun"; return ;;
        D7) dledger_d7 "$st" "$notrun"; return ;;
    esac
    if [ -n "$notrun" ]; then echo "NOT CLEARED: its stage did not finish green:$notrun"; else echo CLEAR; fi
}

# D6: the freeze must run inside the bump script -- the guard's trace shows carry_milestone_items.sh
# calling out during the bump or ship stage. A bump that finished green with no such call is the
# defect, measured; a bump that did not finish proves nothing either way.
dledger_d6() {
    local st=$1 notrun=$2
    if awk -F'\t' '($1 == "bump" || $1 == "ship") && $5 ~ /carry_milestone_items\.sh/ { f = 1 } END { exit !f }' "$st/calls.tsv"; then
        if [ -n "$notrun" ]; then echo "NOT CLEARED: the freeze ran, but its stage did not finish green:$notrun"; else echo CLEAR; fi
    elif awk -F'\t' 'NR > 1 && $1 == "bump" && $2 == 0 { f = 1 } END { exit !f }' "$st/stages.tsv"; then
        echo "OPEN: the bump finished and carry_milestone_items.sh made no call in it"
    else
        echo "NOT CLEARED: the bump did not finish green"
    fi
}

# D7: the bump commit's Co-Authored-By trailers must each name a model the fleet runs.
dledger_d7() {
    local st=$1 notrun=$2 bad
    [ -f "$st/bump.msg" ] || { echo "NOT CLEARED: the ship stage made no bump commit"; return; }
    bad=$(grep -iE '^Co-Authored-By: *Claude' "$st/bump.msg" | grep -vE "^Co-Authored-By: *($FLEET_MODELS) <" | head -1)
    if [ -n "$bad" ]; then echo "OPEN in the bump commit: $bad"
    elif ! grep -qiE '^Co-Authored-By: *Claude' "$st/bump.msg"; then echo "OPEN in the bump commit: no Co-Authored-By trailer"
    elif [ -n "$notrun" ]; then echo "NOT CLEARED: the trailer is right, but its stage did not finish green:$notrun"
    else echo CLEAR; fi
}

# ------------------------------------------------------------------ case table --
selftest_cleanup() { [ -n "${tmp:-}" ] && case "$tmp" in ?*/tmp.?*) rm -rf -- "${tmp:?}" ;; esac; }

selftest() {
    local tmp pass=0 fail=0 st el
    tmp="$(mktemp -d)"
    case "$tmp" in ?*/tmp.?*) : ;; *) echo "mktemp gave '${tmp:-}', refusing"; return 2 ;; esac
    trap selftest_cleanup RETURN
    st="$tmp/st"; el="$tmp/else"
    mkdir -p "$st/clone" "$st/tmp" "$el" || return 2
    # c WANT TOOL CWD ARG... : the guard's verdict on one call
    c() {
        local want=$1 tool=$2 cwd=$3 got
        shift 3
        got=$(cd "$cwd" && CARGO_HOME="${CH:-$HOME/.cargo}" wg_classify "$tool" "$st" "$cwd" "$@")
        if [ "${got%% *}" = "$want" ]; then pass=$((pass + 1))
        else printf '  BROKE %-5s %s %s -> %s\n' "$want" "$tool" "$*" "$got"; fail=$((fail + 1)); fi
    }
    # must refuse: every write H10 names, and unknowns
    c WRITE git "$st/clone" push origin main
    c WRITE git "$st/clone" push --tags
    c WRITE git "$el" -C "$st/clone" push
    c WRITE git "$st/clone" -c user.name=x push origin HEAD:refs/heads/x
    c WRITE git "$st/clone" tag -a v1 -m x
    c WRITE git "$st/clone" tag v1
    c WRITE git "$st/clone" tag v1 HEAD
    c WRITE git "$st/clone" tag -d v1
    c WRITE git "$st/clone" tag -f v1 HEAD
    c WRITE git "$st/clone" tag -s v1 -m x
    c WRITE git "$st/clone" tag --sort=-creatordate v1
    c WRITE git "$el" commit -m x
    c WRITE git "$el" fetch -q origin main
    c WRITE git "$st/clone" -C "$el" commit -m x
    c WRITE git "$st/clone" --git-dir="$el/.git" update-ref refs/heads/x HEAD
    c WRITE git "$el" worktree add "$st/wt"
    c WRITE git "$el" branch release-1
    c WRITE git "$st/clone" config --global user.name x
    c WRITE git "$el" frobnicate
    c WRITE gh "$st/clone" release create v1 --draft
    c WRITE gh "$st/clone" release edit v1 --draft=false
    c WRITE gh "$st/clone" release upload v1 a.tgz
    c WRITE gh "$st/clone" release delete v1
    c WRITE gh "$st/clone" workflow run binary-release.yml --ref v1
    c WRITE gh "$st/clone" workflow run clean-room.yml -f ref=v1
    c WRITE gh "$st/clone" issue create --title x
    c WRITE gh "$st/clone" issue comment 1 --body x
    c WRITE gh "$st/clone" issue close 1
    c WRITE gh "$st/clone" issue edit 1 --milestone 0.71.0
    c WRITE gh "$st/clone" pr create --base main
    c WRITE gh "$st/clone" pr merge 1 --squash --auto
    c WRITE gh "$st/clone" pr comment 1 --body x
    c WRITE gh "$st/clone" pr edit 1 --milestone x
    c WRITE gh "$st/clone" pr checkout 1
    c WRITE gh "$st/clone" run rerun 1 --failed
    c WRITE gh "$st/clone" run cancel 1
    c WRITE gh "$st/clone" label create x
    c WRITE gh "$st/clone" secret set X
    c WRITE gh "$st/clone" auth login
    c WRITE gh "$st/clone" frobnicate now
    c WRITE gh "$st/clone" api -X PATCH repos/o/r/milestones/1 -f state=closed
    c WRITE gh "$st/clone" api repos/o/r/issues -f title=x
    c WRITE gh "$st/clone" api repos/o/r/issues/1/comments -F body=@f
    c WRITE gh "$st/clone" api repos/o/r/git/refs --input ref.json
    c WRITE gh "$st/clone" api --method=DELETE repos/o/r/git/refs/tags/v1
    c WRITE gh "$st/clone" api --method DELETE repos/o/r/git/refs/tags/v1
    c WRITE gh "$st/clone" api -XPOST repos/o/r/dispatches
    c WRITE gh "$st/clone" api -X put repos/o/r/x
    c WRITE gh "$st/clone" api graphql -f 'query=mutation { closeIssue(input: {}) { clientMutationId } }'
    c WRITE cargo "$st/clone" publish -p aprender
    c WRITE cargo "$st/clone" +stable publish
    c WRITE cargo "$st/clone" --locked publish --allow-dirty
    c WRITE cargo "$st/clone" yank --version 1.0.0 aprender
    c WRITE cargo "$st/clone" owner --add x aprender
    c WRITE cargo "$st/clone" login
    CH="$el" c WRITE cargo "$st/clone" install aprender
    c WRITE cargo "$st/clone" install --root "$el" aprender
    c WRITE curl "$st/clone" -X POST https://api.github.com/x
    c WRITE curl "$st/clone" -XDELETE https://api.github.com/x
    c WRITE curl "$st/clone" --request PATCH https://x
    c WRITE curl "$st/clone" -d x=1 https://x
    c WRITE curl "$st/clone" --data-binary @f https://x
    c WRITE curl "$st/clone" --json '{}' https://x
    c WRITE curl "$st/clone" -F f=@a https://x
    c WRITE curl "$st/clone" -T a.tgz https://uploads.github.com/x
    c WRITE ssh "$st/clone" -o BatchMode=yes gx10 bash -s
    c WRITE scp "$st/clone" a gx10:b
    c WRITE sftp "$st/clone" gx10
    c WRITE rsync "$st/clone" a b
    # must pass: the reads the release scripts make, and local writes inside the state dir
    c READ git "$el" status --porcelain
    c READ git "$el" -C "$el" rev-parse HEAD
    c READ git "$el" log -1 --format=%H
    c READ git "$el" tag -l 'v*'
    c READ git "$el" tag --list --sort=-creatordate
    c READ git "$el" tag --points-at HEAD
    c READ git "$el" tag --sort=-creatordate
    c READ git "$el" tag --contains abc123
    c READ git "$el" tag
    c READ git "$el" ls-remote origin
    c READ git "$el" merge-base --is-ancestor a b
    c READ git "$el" config --get user.name
    c READ git "$el" config --global --get user.name
    c READ git "$el" branch --show-current
    c READ git "$el" branch
    c READ git "$el" worktree list
    c READ git "$el" remote -v
    c READ git "$el" symbolic-ref HEAD
    c READ git "$el" --version
    c READ git "$st/clone" commit -qm x
    c READ git "$st/clone" fetch -q origin main
    c READ git "$st/clone" worktree add -q -b release-1 "$st/ap/bump" origin/main
    c READ git "$el" -C "$st/clone" checkout -q --detach HEAD
    c READ git "$st/clone" add -A
    c READ gh "$st/clone" pr view 1 --json mergeCommit -q .mergeCommit.oid
    c READ gh "$st/clone" pr list --state merged --search x --json number,title
    c READ gh "$st/clone" pr checks 1
    c READ gh "$st/clone" run list -w clean-room.yml --json databaseId
    c READ gh "$st/clone" run view 1 --log
    c READ gh "$st/clone" run download 1 -n x -D "$st/tmp/x"
    c READ gh "$st/clone" release view v1 --json assets
    c READ gh "$st/clone" workflow view x
    c READ gh "$st/clone" issue view 1 --json state
    c READ gh "$st/clone" api repos/o/r/milestones
    c READ gh "$st/clone" api --paginate repos/o/r/issues -q .number
    c READ gh "$st/clone" api -X GET search/issues -f q=x
    c READ gh "$st/clone" api -H 'If-None-Match: "x"' repos/o/r/actions/runs
    c READ gh "$st/clone" api graphql -f 'query=query { viewer { login } }'
    c READ gh "$st/clone" --version
    c READ cargo "$st/clone" build --release
    c READ cargo "$st/clone" metadata --no-deps --format-version 1
    c READ cargo "$st/clone" publish --dry-run -p aprender
    c READ cargo "$st/clone" publish -n -p aprender
    c READ cargo "$st/clone" +nightly test --lib
    c READ cargo "$st/clone" install --root "$st/tmp/x" aprender --version =1.0.0
    CH="$st/cargo-home" c READ cargo "$st/clone" install aprender
    c READ curl "$st/clone" -fsSL https://x
    c READ curl "$st/clone" -X GET https://x
    c READ curl "$st/clone" -H 'If-None-Match: x' -o "$st/tmp/f" https://x
    c READ curl "$st/clone" -sSf -D "$st/tmp/h" https://x
    printf '  %s guard rows\n' "$((pass))"
    selftest_stub
    selftest_judge
    printf -- '--- %s/%s rows ---\n' "$pass" "$((pass + fail))"
    [ "$fail" -eq 0 ]
}

# The stub end to end: a planted write in a stage is refused, recorded and turns the night red;
# a read runs the real tool. Also the by-path cargo call prepare_bump.sh makes.
selftest_stub() {
    local s="$tmp/stub" envf o rc
    mkdir -p "$s" || return 2
    envf=$(install_guard "$s") || { printf '  BROKE install_guard failed\n'; fail=$((fail + 1)); return; }
    t() { # t NAME WANT_RC WANT_OUT CMD
        local name=$1 wrc=$2 wout=$3; shift 3
        rc=0; o=$( . "$envf"; export WG_STAGE=plant; cd "$s" && eval "$*" 2>&1 ) || rc=$?
        if [ "$rc" = "$wrc" ] && [[ $o == *"$wout"* ]]; then pass=$((pass + 1))
        else printf '  BROKE %-44s rc=%s (want %s) out=%s\n' "$name" "$rc" "$wrc" "${o:0:160}"; fail=$((fail + 1)); fi
    }
    t a_planted_push_is_refused 97 "REHEARSAL WRITE REFUSED (H10): git push" 'git push origin main'
    t a_planted_tag_is_refused 97 "git tag" 'git tag -a v9.9.9 -m x'
    t a_planted_release_is_refused 97 "gh release create" 'gh release create v9.9.9 --draft'
    t cargo_by_path_is_guarded 97 "cargo publish" '"$CARGO_HOME/bin/cargo" publish -p x'
    t a_read_runs_the_real_tool 0 "git version" 'git --version'
    t credentials_are_not_copied 1 "" 'test -e "$CARGO_HOME/credentials.toml" || test -e "$CARGO_HOME/credentials"'
    t the_registry_token_is_unset 0 "unset" 'echo ${CARGO_REGISTRY_TOKEN:-unset}'
    if awk -F'\t' '$1 == "plant" && $2 == "git" && $3 == "WRITE" && $6 ~ /^push / { f = 1 } END { exit !f }' "$s/calls.tsv"; then
        pass=$((pass + 1))
    else printf '  BROKE %-44s calls.tsv: %s\n' the_refused_push_is_recorded "$(tr '\n' '|' < "$s/calls.tsv")"; fail=$((fail + 1)); fi
    if awk -F'\t' '$1 == "plant" && $2 == "git" && $3 == "READ" && $6 ~ /^--version/ { f = 1 } END { exit !f }' "$s/calls.tsv"; then
        pass=$((pass + 1))
    else printf '  BROKE %-44s calls.tsv has no READ row\n' the_read_is_recorded; fail=$((fail + 1)); fi
}

# fixture_night DIR -> a night where every stage is green on C, the trace shows the freeze inside the
# bump, and the bump commit carries a fleet trailer: the anti-vacuity arm (it must be GREEN).
fixture_night() {
    local d=$1 name
    mkdir -p "$d/logs" || return 2
    printf 'C=c0ffee\nV=0.71.0\n' > "$d/night.env"
    printf 'stage\trc\tcommit\tseconds\n' > "$d/stages.tsv"
    while IFS='|' read -r name _ _; do
        case $name in t2|bump|summary|ship) printf '%s\t0\tc0ffee\t1\n' "$name" ;; *) printf '%s\t0\tb0b\t1\n' "$name" ;; esac >> "$d/stages.tsv"
        printf 'ok\n' > "$d/logs/${name//:/_}.log"
    done < <(stages_table)
    printf 'bump\tgh\tREAD\t-\tbash scripts/release/carry_milestone_items.sh 0.71.0 --dry-run\tapi repos/o/r/milestones\n' > "$d/calls.tsv"
    printf 'b0b\n' > "$d/bump.commit"; printf 'c0ffee\n' > "$d/bump.parent"
    printf 'release: 0.71.0\n\nPmat-Ticket: PMAT-1\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n' > "$d/bump.msg"
}

selftest_judge() {
    local n=0 d o rc
    j() { # j NAME WANT_RC WANT_OUT MUTATION
        local name=$1 wrc=$2 wout=$3 mut=$4
        n=$((n + 1)); d="$tmp/night$n"
        fixture_night "$d" || return 2
        eval "$mut"
        rc=0; o=$(judge "$d" 2>&1) || rc=$?
        if [ "$rc" = "$wrc" ] && [[ $o == *"$wout"* ]]; then pass=$((pass + 1))
        else printf '  BROKE %-48s rc=%s (want %s): %s\n' "$name" "$rc" "$wrc" "$(printf '%s' "$o" | grep -E 'RED|VERDICT' | head -3 | tr '\n' '|')"; fail=$((fail + 1)); fi
    }
    j a_fully_measured_night_is_green 0 "VERDICT GREEN" ':'
    j one_write_row_is_red 1 "RED   WRITE in stage ap:tag: git push" \
        'printf "ap:tag\tgit\tWRITE\tgit push\tbash autopilot.sh\tpush origin v0.71.0\n" >> "$d/calls.tsv"'
    j a_missing_real_tool_is_red 1 "RED   MISSING" \
        'printf "ap:deep\tcurl\tMISSING\tno real curl\tbash x\t-fsSL u\n" >> "$d/calls.tsv"'
    j a_red_stage_is_red 1 "RED   stage ap:deep       exit 1: STOP deep failed" \
        'sed -i "s/^ap:deep\t0/ap:deep\t1/" "$d/stages.tsv"; printf "STOP deep failed\n" > "$d/logs/ap_deep.log"'
    j an_unreached_stage_is_red 1 "RED   stage ap:cleanroom  unreached" \
        'sed -i "s/^ap:cleanroom\t0\tb0b/ap:cleanroom\tunreached\t-/" "$d/stages.tsv"'
    j a_stage_never_run_is_red 1 "RED   stage cascade       not run" \
        'sed -i "/^cascade\t/d" "$d/stages.tsv"'
    j a_stage_on_another_commit_is_red 1 "measured on beef, not on c0ffee" \
        'sed -i "s/^ap:assets\t0\tb0b/ap:assets\t0\tbeef/" "$d/stages.tsv"'
    j a_bump_not_on_c_is_red 1 "RED   stage ap:deep       measured on b0b" \
        'printf "beef\n" > "$d/bump.parent"'
    j d1_line_names_d1 1 "RED   D1 OPEN in ap:preflight: FAIL  R4 HEAD" \
        'printf "FAIL  R4 HEAD abc is not an ancestor of origin/release/0.71.0 (or that ref does not exist)\n" > "$d/logs/ap_preflight.log"'
    j d2_line_names_d2 1 "RED   D2 OPEN in t2:" \
        'printf "  [FAIL] declared:check_model_ladder  exit=1\n" > "$d/logs/t2.log"'
    j d3_line_names_d3 1 "RED   D3 OPEN in ap:tag" \
        'printf "FAIL  NOT_MEASURED: no completed coverage-nightly.yml run on b0b or a version-only parent of it -- x\n" > "$d/logs/ap_tag.log"'
    j d4_line_names_d4 1 "RED   D4 OPEN in cascade" \
        'printf "FAIL  R7 model matrix: no receipt\n" > "$d/logs/cascade.log"'
    j d5_line_names_d5 1 "RED   D5 OPEN in ship" \
        'printf "STOP no T-2 GO receipt for origin/main c0ffee (x) -- run: t2\n" > "$d/logs/ship.log"'
    j d6_no_freeze_call_is_open 1 "RED   D6 OPEN: the bump finished" ': > "$d/calls.tsv"'
    j d6_red_bump_is_not_cleared 1 "RED   D6 NOT CLEARED: the bump did not finish green" \
        ': > "$d/calls.tsv"; sed -i "s/^bump\t0/bump\t1/" "$d/stages.tsv"'
    j d6_freeze_in_another_stage_is_open 1 "RED   D6 OPEN" 'sed -i "s/^bump\t/ap:tag\t/" "$d/calls.tsv"'
    j d7_fable_trailer_is_open 1 "RED   D7 OPEN in the bump commit: Co-Authored-By: Claude Fable 5.1" \
        'sed -i "s/Claude Opus 5.5/Claude Fable 5.1/" "$d/bump.msg"'
    j d7_no_trailer_is_open 1 "RED   D7 OPEN in the bump commit: no Co-Authored-By" \
        'sed -i "/^Co-Authored-By/d" "$d/bump.msg"'
    j d7_no_bump_commit_is_not_cleared 1 "RED   D7 NOT CLEARED: the ship stage made no bump commit" \
        'rm -f "$d/bump.msg"'
    j a_d_row_on_an_unreached_stage_is_not_cleared 1 "RED   D1 NOT CLEARED: its stage did not finish green: ap:preflight(unreached)" \
        'sed -i "s/^ap:preflight\t0\tb0b/ap:preflight\tunreached\t-/" "$d/stages.tsv"'
    j no_night_is_not_measured 2 "not_measured" 'rm -f "$d/stages.tsv"'
    printf '  %s judge rows\n' "$n"
}

# ------------------------------------------------------------------ mutants --
# Each mutant is "name sed-script". It must change this file, still parse, and turn --selftest RED with at
# least one BROKE row. A pattern that no longer matches is reported, never skipped. Mutants of the guard
# library are applied to a copy of it that a copy of this script sources.
mutants() {
    local tmp pass=0 fail=0 name file expr dir o rc
    tmp="$(mktemp -d)"
    case "$tmp" in ?*/tmp.?*) : ;; *) echo "mktemp gave '${tmp:-}', refusing"; return 2 ;; esac
    trap selftest_cleanup RETURN
    while read -r name file expr; do
        [ -n "$name" ] || continue
        dir="$tmp/$name"; mkdir -p "$dir"
        cp -- "$SCRIPT_PATH" "$dir/rehearse.sh"; cp -- "$SCRIPT_DIR/lib_write_guard.sh" "$dir/lib_write_guard.sh"
        sed -i -e "$expr" "$dir/$file"
        if cmp -s "$dir/$file" "$SCRIPT_DIR/$file"; then
            printf '  BROKE %-40s changed nothing: its pattern no longer matches\n' "$name"; fail=$((fail + 1)); continue
        fi
        if ! bash -n "$dir/$file" 2>/dev/null; then
            printf '  BROKE %-40s does not parse: a RED from it would prove nothing\n' "$name"; fail=$((fail + 1)); continue
        fi
        rc=0; o="$(bash "$dir/rehearse.sh" --selftest < /dev/null 2>&1)" || rc=$?
        case "$rc:$o" in
            0:*) printf '  BROKE %-40s SURVIVED: the case table stayed green\n' "$name"; fail=$((fail + 1)) ;;
            *"  BROKE "*) printf '  ok    %-40s killed, %s row(s) broke\n' "$name" "$(printf '%s\n' "$o" | awk '/^  BROKE /{n++} END{print n+0}')"; pass=$((pass + 1)) ;;
            *) printf '  BROKE %-40s exit %s with no broken row: not a kill\n' "$name" "$rc"; fail=$((fail + 1)) ;;
        esac
    done <<'MUTANTS'
push_is_a_read               lib_write_guard.sh  s/push|send-email|request-pull|svn|p4) echo "WRITE git \$sub"/send-email|request-pull|svn|p4) echo "WRITE git $sub"/
tag_list_flag_ignored        lib_write_guard.sh  s/-l|--list|-v|--verify) list=1 ;;/-v|--verify) list=1 ;;/
tag_positional_is_a_read     lib_write_guard.sh  s/\*) pos=1 ;;/*) ;;/
tag_value_flags_not_skipped  lib_write_guard.sh  s/--contains|--no-contains|--points-at|--merged|--no-merged|--sort|--format|--color|--column) skip=1 ;;/--zz) skip=1 ;;/
outside_state_is_inside      lib_write_guard.sh  s/\[ "\$p" = "\$s" \] || \[ "\${p#"\$s"\/}" != "\$p" \]/true/
dash_c_ignored               lib_write_guard.sh  s/-C) dir=\$(cd/-Q) dir=$(cd/
git_dir_ignored              lib_write_guard.sh  s/--git-dir=\*) gitdir=\${a#--git-dir=}; shift ;;/--git-dir=*) shift ;;/
global_config_is_local       lib_write_guard.sh  s/case \$a in --global|--system)/case $a in --zz)/
unknown_git_is_a_read        lib_write_guard.sh  /^wg_git_read() {$/,/^}$/s/^        \*) return 1 ;;$/        *) return 0 ;;/
gh_body_is_a_get             lib_write_guard.sh  s/method=POST; else method=GET/method=GET; else method=GET/
gh_method_ignored            lib_write_guard.sh  s/-X|--method) skip=m ;;/-X|--method) skip=1 ;;/
gh_mutation_is_a_read        lib_write_guard.sh  s/echo "WRITE gh api graphql mutation"/echo READ/
gh_unknown_is_a_read         lib_write_guard.sh  s/\*) echo "WRITE gh \$cmd\${sub:+ \$sub}" ;;/*) echo READ ;;/
gh_release_create_read       lib_write_guard.sh  s/release:view|release:list|release:download/release:view|release:list|release:download|release:create/
gh_workflow_run_read         lib_write_guard.sh  s/|workflow:view|workflow:list|/|workflow:view|workflow:list|workflow:run|/
cargo_publish_read           lib_write_guard.sh  s/echo "WRITE cargo publish (no --dry-run)"/echo READ/
cargo_install_anywhere       lib_write_guard.sh  s/echo "WRITE cargo install into \$root (outside the state dir)"/echo READ/
cargo_toolchain_is_sub       lib_write_guard.sh  s/+\*|-q|--quiet/-q|--quiet/
curl_body_is_a_read          lib_write_guard.sh  s/-d|-d?\*|--data\*/-zz|--zz/
ssh_is_a_read                lib_write_guard.sh  s/ssh|scp|sftp) v="WRITE/ssh|scp|sftp) v="READ/
stub_runs_a_write            lib_write_guard.sh  s/^        READ) exec "\$real" "\$@" ;;$/        *) exec "$real" "$@" ;;/
stub_records_nothing         lib_write_guard.sh  s/>> "\${WG_CALLS:?}"/> \/dev\/null/
credentials_copied           rehearse.sh         s/for f in registry git config.toml config; do/for f in registry git config.toml config credentials.toml; do/
token_kept                   rehearse.sh         s/printf 'unset CARGO_REGISTRY_TOKEN\\n'/printf 'export CARGO_REGISTRY_TOKEN=set\\n'/
cargo_home_bin_not_guarded   rehearse.sh         s/rm -f "\${bin:?}\/\${t:?}"/[ -e "${bin:?}\/${t:?}" ] \&\& continue/
writes_not_counted           rehearse.sh         s/\$3 == "WRITE" || \$3 == "MISSING"/$3 == "MISSING"/
missing_not_counted          rehearse.sh         s/\$3 == "WRITE" || \$3 == "MISSING"/$3 == "WRITE"/
unreached_is_green           rehearse.sh         s/unreached) printf 'RED   stage %-13s unreached/unreached) continue; printf 'RED   stage %-13s unreached/
other_commit_is_green        rehearse.sh         s/if \[ "\$commit" = "\$c" \] ||/if true ||/
bump_parent_not_checked      rehearse.sh         s/\&\& \[ "\$bparent" = "\$c" \]; }/; }/
dledger_sig_ignored          rehearse.sh         s/\[ -n "\$hit" \] \&\& { echo "OPEN in \$s: \$hit"; return; }/:/
unreached_d_row_clears       rehearse.sh         s/\[ "\$rc" = 0 \] || notrun="\$notrun \$s(\${rc:-not run})"/:/
d6_any_stage_counts          rehearse.sh         s/(\$1 == "bump" || \$1 == "ship") \&\& \$5/$5/
d6_absent_clears             rehearse.sh         s/echo "OPEN: the bump finished and carry_milestone_items.sh made no call in it"/echo CLEAR/
d6_red_bump_opens            rehearse.sh         s/NR > 1 \&\& \$1 == "bump" \&\& \$2 == 0/NR > 1 \&\& $1 == "bump"/
d7_fable_allowed             rehearse.sh         s/^FLEET_MODELS='Claude Opus 5\\.5|/FLEET_MODELS='Claude Fable 5\\.1|Claude Opus 5\\.5|/
d7_no_trailer_clears         rehearse.sh         s/echo "OPEN in the bump commit: no Co-Authored-By trailer"/echo CLEAR/
green_with_reds              rehearse.sh         s/if \[ "\$reds" -eq 0 \]; then echo "VERDICT GREEN/if true; then echo "VERDICT GREEN/
MUTANTS
    printf -- '--- %s/%s mutants killed ---\n' "$pass" "$((pass + fail))"
    [ "$fail" -eq 0 ]
}

case "${1:-}" in
    --run) shift; run_night "$@" ;;
    --judge) [ -n "${2:-}" ] || die3 "--judge DIR"; judge "$2" ;;
    --classify) [ $# -ge 3 ] || die3 "--classify TOOL CWD [ARG...]"; t=$2; d=$3; shift 3; wg_classify "$t" "${WG_STATE:-/nonexistent-state}" "$d" "$@" ;;
    --selftest) selftest ;;
    --mutants) mutants ;;
    -h|--help) awk 'NR > 1 && /^set -uo pipefail$/ { exit } NR > 1' "$0" ;;
    *) die3 "usage: rehearse.sh --run|--judge|--classify|--selftest|--mutants (see --help)" ;;
esac
