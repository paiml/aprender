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
#   rehearse.sh --run --state DIR [--commit SHA] [--version V] [--source REPO] [--stages a,b] [--in-run RUN]
#       DIR must not exist (or be empty). SHA defaults to the source's HEAD; V to the lowest open
#       milestone above the workspace version. --stages runs a subset (a development aid: the
#       stages it skips are unreached, so such a night is never green). The lanes stage reads the
#       train on SHA; RUN is the workflow run whose lanes job ran the in-repo lanes at SHA.
#   rehearse.sh --pick-c --cache DIR [--repo DIR] [--remote NAME]
#       C on stdout (B1 Q1): the newest first-parent main commit the models crux bundle (its INDEX on
#       nightly-evidence) and an infra clean-room run (release_lanes.sh measured-cpu) both measured.
#       EXIT 0 · 2 not_measured (a read failed, or no commit both measured: the count resets) · 3.
#   rehearse.sh --judge DIR      the verdict of a finished night (the --run prints it too)
#   rehearse.sh --streak --commit REV --as-of YYYY-MM-DD --cache DIR [--repo DIR] [--history FILE] [--release-path]
#       the three counting nights a pass on REV needs (C345 #4, Q9). Each night records the tree id of
#       scripts/release and the blob ids of the stop list and the policy file; a night counts only when
#       its ids equal REV's (release day's), so any change to them resets the count. The count is
#       nightly_greens.sh's over this workflow's scheduled runs on main: a dispatched day run never counts.
#       Ready prints one RECEIPT line per counting night. EXIT 0 ready · 1 not ready · 2 · 3 as below.
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
# C345 #4: "Each night's receipt records the git tree id of scripts/release plus the blob ids of the stop list and
# the policy file. Three counting nights carry identical ids, equal to release day's. Any change resets the count."
IDS_TREE=scripts/release
IDS_STOP=contracts/release-ready-v1.yaml
IDS_POLICY=contracts/model-capability-ladder-v1.yaml
# The workflow whose scheduled runs on main are the counting nights, and the floor its one REST read keeps.
WORKFLOW=release-rehearsal-nightly.yml
RATE_FLOOR=1000
# B1 Q1: C is the newest first-parent main commit that both the models crux bundle and an infra clean-room run
# measured. The bundle and its INDEX sit on EVIDENCE_BRANCH under MODELS_DIR; the walk reads main's newest PICK_DEPTH
# first-parent commits (about five days of main in early October 2026), and a commit further back is no night's C.
EVIDENCE_BRANCH=nightly-evidence
MODELS_DIR=models-crux
PICK_DEPTH=60

# shellcheck source=scripts/release/lib_write_guard.sh
. "$SCRIPT_DIR/lib_write_guard.sh" || exit 2

# STAGES: name | needs (a stage that must have finished green, or -) | command, run from the clone root.
# $V is the train's version. ap:<step> is autopilot.sh run for that one step: its setup is re-entrant.
stages_table() {
    cat <<'STAGES'
lanes|-|env -u INBOX bash scripts/release/nightly_train.sh --out "$RELEASE_REHEARSAL_TRAIN" --commit "$RELEASE_REHEARSAL_C" ${RELEASE_REHEARSAL_IN_RUN:+--in-run "$RELEASE_REHEARSAL_IN_RUN"}
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
# install_guard STATE [C [RUN]] -> stubs + a credential-free CARGO_HOME under STATE; prints the env file path.
# C is the night's commit: the release scripts' rehearsal seams (lib_rehearsal.sh) read the night's train
# bundle in STATE/train for it. RUN is the workflow run whose lanes job ran the in-repo lanes at C.
install_guard() {
    local st=$1 c=${2:-} run=${3:-} ch bin real t f
    ch="$st/cargo-home"; bin="$ch/bin"
    mkdir -p "$bin" "$st/tmp" "$st/ap" "$st/logs" "$st/train" || return 2
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
        printf 'export RELEASE_REHEARSAL_C=%q RELEASE_REHEARSAL_TRAIN=%q RELEASE_REHEARSAL_IN_RUN=%q\n' "$c" "$st/train" "$run"
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
    local st="" commit="" v="" src="" only="" inrun="" env name needs cmd rc start
    while [ $# -gt 0 ]; do
        case $1 in
            --state) st=${2:-}; shift 2 ;;
            --commit) commit=${2:-}; shift 2 ;;
            --version) v=${2:-}; shift 2 ;;
            --source) src=${2:-}; shift 2 ;;
            --stages) only=${2:-}; shift 2 ;;
            --in-run) inrun=${2:-}; shift 2 ;;
            *) die3 "unknown option $1" ;;
        esac
    done
    [ -z "$inrun" ] || [[ $inrun =~ ^[1-9][0-9]*$ ]] || die3 "--in-run must be a run id, not '$inrun'"
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
    env=$(install_guard "$st" "$commit" "$inrun") || exit 2
    night_env "$st" "$commit" "$v" "$src" || exit 2
    printf 'stage\trc\tcommit\tseconds\n' > "$st/stages.tsv"
    while IFS='|' read -r name needs cmd; do
        [ -n "$name" ] || continue
        if [ -n "$only" ] && [[ ",$only," != *",$name,"* ]]; then continue; fi
        if [ "$needs" != - ] && ! awk -F'\t' -v n="$needs" '$1 == n && $2 == 0 { f = 1 } END { exit !f }' "$st/stages.tsv"; then
            printf '%s\tunreached\t-\t0\n' "$name" >> "$st/stages.tsv"; continue
        fi
        start="$SECONDS"
        run_stage "$st" "$env" "$name" "$cmd" "$v"; rc=$?
        if [ "$name" = ship ] && [ "$rc" = 0 ] && ! handoff_bump "$st" "$env" "$commit"; then
            printf 'STOP rehearsal: the ship stage made no bump commit on %s that the release steps can read\n' "$commit" >> "$st/logs/ship.log"
            rc=1
        fi
        printf '%s\t%s\t%s\t%s\n' "$name" "$rc" "$(stage_commit "$st" "$name" "$commit")" "$(( SECONDS - start ))" >> "$st/stages.tsv"
    done < <(stages_table)
    record_bump "$st" "$commit"
    judge "$st"
}

# ap_gate_files DIR -> one "name<TAB>mtime<TAB>sha256" line per gate file directly in DIR, C-sorted
# (STATUS and autopilot.log are carried by byte offset instead). run_stage diffs two of these: a file
# that is new, or whose content or mtime changed, is the stage's own. Content, not an mtime stamp: mtime
# ticks are coarse (one jiffy), so a file a fast stage writes in the stamp's own tick reads as not newer.
ap_gate_files() {
    local f t
    find "$1" -maxdepth 1 -type f ! -name STATUS ! -name autopilot.log -printf '%f\t%T@\n' |
        while IFS=$'\t' read -r f t; do
            printf '%s\t%s\t%s\n' "$f" "$t" "$(sha256sum < "$1/$f")"
        done | LC_ALL=C sort
}

# run_stage STATE ENV NAME CMD V -> the stage's exit status. Its log is its own output, then what it
# added to the release scripts' state: each gate's output file there (preflight.log, tag-coverage.log,
# ...), then what autopilot.sh `say`s into RELEASE_AP's autopilot.log and STATUS, not stdout -- STATUS
# last, so the stage's own STOP line is the last stop line in the log (stage_tail).
# The verdict and the D-ledger read the log, so a stop autopilot.sh printed only there still counts.
run_stage() {
    local st=$1 env=$2 name=$3 cmd=$4 v=$5 log rc f s0 l0 before
    log="$st/logs/${name//:/_}.log"
    s0=$(stat -c %s -- "$st/ap/STATUS" 2>/dev/null) || s0=0
    l0=$(stat -c %s -- "$st/ap/autopilot.log" 2>/dev/null) || l0=0
    before=$(ap_gate_files "$st/ap" 2>/dev/null)
    ( cd "$st/clone" || exit 2
      # shellcheck disable=SC1090
      . "$env" || exit 2
      export WG_STAGE=$name V=$v
      if [ "$cmd" = stage_summary ]; then stage_summary; else bash -c -- "$cmd"; fi ) > "$log" 2>&1 < /dev/null
    rc=$?
    {
        while IFS= read -r f; do
            printf '== RELEASE_AP/%s ==\n' "$f"; cat -- "$st/ap/$f"
        done < <(ap_gate_files "$st/ap" | LC_ALL=C comm -13 <(printf '%s\n' "$before") - | cut -f1)
        [ -f "$st/ap/autopilot.log" ] && { printf '== RELEASE_AP/autopilot.log (this stage) ==\n'; tail -c +"$((l0 + 1))" -- "$st/ap/autopilot.log"; }
        [ -f "$st/ap/STATUS" ] && { printf '== RELEASE_AP/STATUS (this stage) ==\n'; tail -c +"$((s0 + 1))" -- "$st/ap/STATUS"; }
    } >> "$log" 2>/dev/null
    return "$rc"
}

# handoff_bump STATE ENV C -> rc 0 when the ship stage's local bump commit is one commit on C and is now
# the release commit: the in-state origin's main moves to it, as the merged bump PR moves main on
# release day (B1 quorum Q2 (a)), and autopilot.sh reads it as RELEASE_REHEARSAL_MC. Every write here
# is inside the state dir.
handoff_bump() {
    local st=$1 env=$2 c=$3 h
    h=$(git -C "$st/ap/bump" rev-parse --verify -q HEAD) || return 1
    [ "$(git -C "$st/ap/bump" rev-parse --verify -q "$h^")" = "$c" ] || return 1
    git -C "$st/clone" update-ref refs/rehearsal/bump "$h" || return 1
    git -C "$st/origin.git" fetch -q "$st/clone" "+refs/rehearsal/bump:refs/heads/main" || return 1
    printf 'export RELEASE_REHEARSAL_MC=%q\n' "$h" >> "$env"
}

# stage_commit STATE STAGE C -> the commit the stage measured: C before the bump, the bump commit after
stage_commit() {
    case $2 in
        lanes|t2|bump|summary|ship) printf '%s' "$3" ;;
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

# ------------------------------------------------------------------ the ids --
# release_ids REPO REV -> "tree=<id> stop=<id> policy=<id>" for REV in REPO's git (C345 #4). rc 1: REV lacks one
# of the three, the reason on stderr, so its ids equal no tree that has them; rc 2: REPO has no commit REV.
release_ids() {
    local g=$1 r=$2 c k name p want id out=""
    c=$(git -C "$g" rev-parse -q --verify "$r^{commit}") || { echo "no commit $r in $g" >&2; return 2; }
    for k in "tree:$IDS_TREE:tree" "stop:$IDS_STOP:blob" "policy:$IDS_POLICY:blob"; do
        IFS=: read -r name p want <<< "$k"
        id=$(git -C "$g" rev-parse -q --verify "$c:$p") && [ "$(git -C "$g" cat-file -t "$id")" = "$want" ] \
            || { echo "$c has no $want $p" >&2; return 1; }
        out="$out${out:+ }$name=$id"
    done
    printf '%s\n' "$out"
}

# night_env ST C V REPO -> ST/night.env: the night's commit, version, source and start, then C's ids:
# IDS=<ids>, or IDS=- and IDS_WHY=<reason> (the judge turns that red)
night_env() {
    local ids
    printf 'C=%s\nV=%s\nSOURCE=%s\nSTARTED=%s\n' "$2" "$3" "$4" "$(date -u -d "@${SOURCE_DATE_EPOCH:-$(date +%s)}" +%FT%TZ)" > "$1/night.env" || return 2
    if ids=$(release_ids "$4" "$2" 2>&1); then printf 'IDS=%s\n' "$ids"
    else printf 'IDS=-\nIDS_WHY=%s\n' "${ids%%$'\n'*}"; fi >> "$1/night.env"
}

# fetch_runs CACHE OUT -> OUT: the workflow's scheduled runs on main, nightly_greens.sh's six columns then head_sha.
# One REST read, sent with the cached ETag (a 304 reuses CACHE/runs.json). rc 2 and the reason on stderr when
# GitHub cannot be read or the floor is reached: not_measured, never "no nights".
fetch_runs() {
    local cache=$1 out=$2 relpath=${3:-0} lim rem fl st hdr=()
    read -r lim rem <<< "$(gh api rate_limit --jq '"\(.resources.core.limit) \(.resources.core.remaining)"' 2>/dev/null)"
    case "$lim:$rem" in *[!0-9:]*|:*|*:) echo "rate_limit unreadable" >&2; return 2 ;; esac
    # a fifth of the token's own hourly limit, capped at RATE_FLOOR, as nightly_train.sh's read keeps. The pass start
    # (--release-path) is the release path, which GH-1 lets call below 1000: no floor there (quorum 09:34Z, C345 Q9 Q3 B);
    # a refused or rate-limited read is still rc 2 for it.
    fl=$((lim / 5)); [ "$fl" -le "$RATE_FLOOR" ] || fl=$RATE_FLOOR
    if [ "$relpath" = 1 ]; then fl=0; fi
    [ "$rem" -ge "$fl" ] || { echo "core remaining $rem under $fl" >&2; return 2; }
    [ -s "$cache/runs.etag" ] && [ -s "$cache/runs.json" ] && hdr=(-H "If-None-Match: $(cat "$cache/runs.etag")")
    gh api -i "${hdr[@]}" "repos/$REPO/actions/workflows/$WORKFLOW/runs?branch=main&event=schedule&per_page=60" > "$cache/runs.http" 2>/dev/null
    st=$(awk 'NR == 1 { print $2; exit }' "$cache/runs.http")
    case $st in
        200) awk 'f { print } /^\r?$/ { f = 1 }' "$cache/runs.http" > "$cache/runs.json" || return 2
             awk 'tolower($1) == "etag:" { sub(/\r$/, ""); sub(/^[^:]*: */, ""); print; exit }' "$cache/runs.http" > "$cache/runs.etag" ;;
        304) [ -s "$cache/runs.json" ] || { echo "HTTP 304 and no cached runs" >&2; return 2; } ;;
        *) echo "workflow runs read: HTTP ${st:-none}" >&2; return 2 ;;
    esac
    { printf 'run_id\tcreated_at\tbranch\tevent\tconclusion\tattempt\thead_sha\n'
      jq -r '.workflow_runs | if type == "array" then .[] else error("no workflow_runs") end
             | [.id, .created_at, .head_branch, .event, (.conclusion // ""), .run_attempt, .head_sha] | @tsv' "$cache/runs.json"
    } > "$out" 2>/dev/null || { echo "the runs read is not a workflow_runs list" >&2; return 2; }
}

# streak --commit REV --as-of YYYY-MM-DD --cache DIR [--repo DIR] [--history FILE] -> the three counting nights'
# receipts a pass on REV needs (C345 #4, Q9). REV's ids are release day's. A scheduled night on main counts only
# when its head carries the same ids; any other is a reset, judged as a red night. The count is nightly_greens.sh's.
# The history is FILE (seven columns, as fetch_runs writes) or one read of the workflow's runs. Writes only in DIR.
# rc 0 ready (one RECEIPT line per counting night) · 1 not ready · 2 not_measured · 3 caller error
streak() {
    local rev="" repo="" hist="" asof="" cache="" relpath=0 target rc head ids o hdr
    while [ $# -gt 0 ]; do
        case $1 in
            --commit|--repo|--history|--as-of|--cache)
                [ $# -ge 2 ] || die3 "$1 needs a value"
                case $1 in --commit) rev=$2 ;; --repo) repo=$2 ;; --history) hist=$2 ;; --as-of) asof=$2 ;; *) cache=$2 ;; esac
                shift 2 ;;
            --release-path) relpath=1; shift ;;
            *) die3 "unknown option $1" ;;
        esac
    done
    [ -n "$rev" ] || die3 "--commit REV is required: the commit the pass starts on"
    [[ $asof =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || die3 "--as-of YYYY-MM-DD is required: a count is made for a named date"
    [ -n "$cache" ] || die3 "--cache DIR is required: the read and the history are written there and nowhere else"
    mkdir -p "$cache" || exit 2
    repo=${repo:-$(cd "$SCRIPT_DIR/../.." && pwd)}
    rev=$(git -C "$repo" rev-parse -q --verify "$rev^{commit}") || { echo "not_measured: no commit $rev in $repo"; return 2; }
    target=$(release_ids "$repo" "$rev" 2>&1) || { echo "not_measured: the pass commit's ids cannot be read: $target"; return 2; }
    printf 'TARGET %s %s\n' "$rev" "$target"
    if [ -z "$hist" ]; then
        hist="$cache/runs.tsv"
        o=$(fetch_runs "$cache" "$hist" "$relpath" 2>&1) || { echo "not_measured: $o"; return 2; }
    fi
    hdr=$(head -n 1 -- "$hist" 2>/dev/null)
    [ "$hdr" = "$(printf 'run_id\tcreated_at\tbranch\tevent\tconclusion\tattempt\thead_sha')" ] \
        || { echo "not_measured: $hist is not a seven-column run history"; return 2; }
    : > "$cache/heads.tsv"
    while IFS= read -r head; do
        rc=0; ids=$(release_ids "$repo" "$head" 2>&1) || rc=$?
        case $rc in
            0) printf '%s\t%s\n' "$head" "$ids" >> "$cache/heads.tsv" ;;
            1) printf '%s\t-\t%s\n' "$head" "$ids" >> "$cache/heads.tsv" ;;
            *) echo "not_measured: a counted night's head is not in $repo ($ids): fetch main first"; return 2 ;;
        esac
    done < <(awk -F'\t' 'NR > 1 && $3 == "main" && $4 == "schedule" { print $7 }' "$hist" | LC_ALL=C sort -u)
    # a night whose ids are not the target's is a reset: its run is judged failure, whatever it ended as
    awk -F'\t' -v OFS='\t' -v target="$target" '
        function diff(a, b,    x, y, i, out) {
            if (a == "-") return b
            split(a, x, " "); split(target, y, " "); out = ""
            for (i = 1; i <= 3; i++) if (x[i] != y[i]) out = out (out == "" ? "" : ", ") substr(y[i], 1, index(y[i], "=") - 1)
            return out " differ"
        }
        FILENAME == ARGV[1] { ids[$1] = $2; why[$1] = $3; next }
        FNR == 1 { print $1, $2, $3, $4, $5, $6; next }
        $3 == "main" && $4 == "schedule" && ids[$7] != target {
            printf "RESET run %s (night %s, head %s): %s\n", $1, substr($2, 1, 10), $7, diff(ids[$7], why[$7]) > "/dev/stderr"
            $5 = "failure"
        }
        { print $1, $2, $3, $4, $5, $6 }' "$cache/heads.tsv" "$hist" > "$cache/history.tsv" 2> "$cache/resets.txt" \
        || { echo "not_measured: the history could not be read"; return 2; }
    cat -- "$cache/resets.txt"
    rc=0; o=$(bash "$SCRIPT_DIR/nightly_greens.sh" --check release-rehearsal --history "$cache/history.tsv" --as-of "$asof") || rc=$?
    printf '%s\n' "$o"
    [ "$rc" = 0 ] || return "$rc"
    # the three receipts: the run IDs nightly_greens.sh printed, each with its night, head and ids
    o=$(printf '%s\n' "$o" | sed -n 's/^ok .* ready: .*: runs\(\( [0-9][0-9]*\)*\) (.*/\1/p')
    awk -F'\t' -v runs="$o" 'BEGIN { split(runs, r, " "); for (i in r) want[r[i]] = 1 }
        FILENAME == ARGV[1] { ids[$1] = $2; next }
        FNR > 1 && ($1 in want) { printf "RECEIPT run %s night %s head %s %s\n", $1, substr($2, 1, 10), $7, ids[$7]; n++ }
        END { exit (n != 3) }' "$cache/heads.tsv" "$hist" \
        || { echo "not_measured: nightly_greens.sh said ready, but its run IDs are not three counted runs in $hist"; return 2; }
}

# ------------------------------------------------------------------ the pick --
# models_measured REPO REF CACHE -> "sha TAB green|red" for each commit the models INDEX on REF says the crux bundle
# measured. INDEX lines are `<40-hex> green|red|not_measured`, one per commit, and a not_measured line is not a
# measurement. A green or red line counts only when its bundle MODELS_DIR/<sha>/ is on REF with a verdict whose state
# is the same. rc 2 and the reason on stderr: no INDEX, a line out of that format, a commit with two lines, or a
# measured line its bundle does not bear out. A reader that cannot trust its input reads nothing, never a part of it.
models_measured() {
    local g=$1 ref=$2 f="$3/INDEX" re='^([0-9a-f]{40}) (green|red|not_measured)$' line n=0 sha state v
    local -A seen=()
    git -C "$g" cat-file blob "$ref:$MODELS_DIR/INDEX" > "$f" 2>/dev/null || { echo "not_measured: models: $EVIDENCE_BRANCH has no blob $MODELS_DIR/INDEX" >&2; return 2; }
    while IFS= read -r line || [ -n "$line" ]; do
        n=$((n + 1))
        [[ $line =~ $re ]] || { echo "not_measured: models: $MODELS_DIR/INDEX line $n is not '<40-hex> green|red|not_measured': ${line:0:80}" >&2; return 2; }
        sha=${BASH_REMATCH[1]}; state=${BASH_REMATCH[2]}
        [ -z "${seen[$sha]:-}" ] || { echo "not_measured: models: $MODELS_DIR/INDEX has two lines for $sha" >&2; return 2; }
        seen[$sha]=1
        [ "$state" != not_measured ] || continue
        v=$(git -C "$g" cat-file blob "$ref:$MODELS_DIR/$sha/verdict" 2>/dev/null | awk 'index($0, "state=") == 1 { print substr($0, 7); exit }')
        [ "$v" = "$state" ] || { echo "not_measured: models: $MODELS_DIR/INDEX says $state for $sha, and its bundle's verdict says ${v:-nothing}" >&2; return 2; }
        printf '%s\t%s\n' "$sha" "$state"
    done < "$f"
}

# cpu_measured CACHE -> release_lanes.sh measured-cpu: "sha TAB green|red TAB run" for each commit an infra clean-room
# run measured; rc 2 not_measured, its reason on stderr
cpu_measured() { bash "$SCRIPT_DIR/release_lanes.sh" measured-cpu --cache "$1"; }

# pick_c --cache DIR [--repo DIR] [--remote NAME] -> C on stdout, rc 0 (B1 Q1). Fetches REMOTE's main and evidence
# branch into REPO's refs/rehearsal/ (whole fetches, never shallow; a shallow REPO is refused, its first-parent list is
# cut), reads both measured sets, and walks main's newest PICK_DEPTH first-parent commits, newest first: the first one
# both sides measured is C, printed with a receipt on stderr. A red on either side is a measurement. rc 2 and the
# reason on stderr: a fetch or a reader read nothing, or no commit in the walk was measured by both. That night is
# not_measured, and the count resets. rc 3 a caller error.
pick_c() {
    local g="" remote=origin cache="" rows line sha fp k=0 cre=$'^([0-9a-f]{40})\t(green|red)\t([0-9]+)$'
    local -A models=() cpu=()
    while [ $# -gt 0 ]; do
        case $1 in
            --repo) g=${2:-}; shift 2 ;;
            --remote) remote=${2:-}; shift 2 ;;
            --cache) cache=${2:-}; shift 2 ;;
            *) die3 "unknown option $1" ;;
        esac
    done
    [ -n "$cache" ] || die3 "--pick-c needs --cache DIR"
    g=${g:-$(cd "$SCRIPT_DIR/../.." && pwd)}
    mkdir -p -- "$cache" || return 2
    [ "$(git -C "$g" rev-parse --is-shallow-repository 2>/dev/null)" = false ] || { echo "not_measured: $g is not a full clone (shallow, or no git): its first-parent list is cut" >&2; return 2; }
    git -C "$g" fetch -q --no-tags "$remote" "+refs/heads/main:refs/rehearsal/main" || { echo "not_measured: main was not fetched from $remote" >&2; return 2; }
    git -C "$g" fetch -q --no-tags "$remote" "+refs/heads/$EVIDENCE_BRANCH:refs/rehearsal/$EVIDENCE_BRANCH" || { echo "not_measured: $EVIDENCE_BRANCH was not fetched from $remote (absent, or the read failed): no models bundle was read" >&2; return 2; }
    rows=$(models_measured "$g" "refs/rehearsal/$EVIDENCE_BRANCH" "$cache") || return 2
    if [ -n "$rows" ]; then
        while IFS=$'\t' read -r sha line; do models[$sha]=$line; done <<< "$rows"
    fi
    rows=$(cpu_measured "$cache/cpu") || { echo "not_measured: cleanroom-cpu: release_lanes.sh measured-cpu read nothing (its reason is above)" >&2; return 2; }
    if [ -n "$rows" ]; then
        while IFS= read -r line; do
            [[ $line =~ $cre ]] || { echo "not_measured: cleanroom-cpu: measured-cpu printed a line that is not 'sha TAB green|red TAB run': ${line:0:80}" >&2; return 2; }
            cpu[${BASH_REMATCH[1]}]="${BASH_REMATCH[2]} run ${BASH_REMATCH[3]}"
        done <<< "$rows"
    fi
    fp=$(git -C "$g" rev-list --first-parent -n "$PICK_DEPTH" refs/rehearsal/main) || { echo "not_measured: main's first-parent list was not read" >&2; return 2; }
    while read -r sha; do
        if [ -n "${models[$sha]:-}" ] && [ -n "${cpu[$sha]:-}" ]; then
            printf 'pick: C %s (main~%s): models-crux %s, cleanroom-cpu %s\n' "$sha" "$k" "${models[$sha]}" "${cpu[$sha]}" >&2
            printf '%s\n' "$sha"; return 0
        fi
        k=$((k + 1))
    done <<< "$fp"
    echo "not_measured: no commit in main's newest $PICK_DEPTH first-parent commits was measured by both the models bundle (${#models[@]} commits) and an infra clean-room run (${#cpu[@]} commits)" >&2
    return 2
}

# ------------------------------------------------------------------ the judge --
# judge STATE -> the verdict on stdout; rc 0 GREEN, 1 RED, 2 not_measured
judge() {
    local st=$1 c v reds=0 name rc commit sec row stg sig what ids
    [ -f "$st/night.env" ] && [ -f "$st/stages.tsv" ] && [ -f "$st/calls.tsv" ] \
        || { echo "not_measured: $st holds no night (night.env, stages.tsv, calls.tsv)"; return 2; }
    c=$(sed -n 's/^C=//p' "$st/night.env"); v=$(sed -n 's/^V=//p' "$st/night.env")
    [ -n "$c" ] || { echo "not_measured: night.env names no commit"; return 2; }
    local bumpc="" bparent=""
    [ -f "$st/bump.commit" ] && bumpc=$(cat "$st/bump.commit")
    [ -f "$st/bump.parent" ] && bparent=$(cat "$st/bump.parent")
    printf 'REHEARSAL %s on %s\n' "$v" "$c"
    ids=$(sed -n 's/^IDS=//p' "$st/night.env")
    if [[ $ids =~ ^tree=[0-9a-f]{40,64}\ stop=[0-9a-f]{40,64}\ policy=[0-9a-f]{40,64}$ ]]; then printf 'IDS   %s\n' "$ids"
    else
        printf 'RED   ids not recorded (%s): C345 #4, a night without its ids cannot count\n' "$(sed -n 's/^IDS_WHY=//p' "$st/night.env" | grep . || echo 'night.env has no IDS line')"
        reds=$((reds + 1))
    fi
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
    if [ "$reds" -eq 0 ]; then echo "VERDICT GREEN: every stage green on $c, its ids recorded, 0 writes, D1..D7 clear"; return 0; fi
    echo "VERDICT RED: $reds red line(s)"
    return 1
}

stage_tail() {
    local f="$1/logs/${2//:/_}.log"
    [ -f "$f" ] || { printf 'no log'; return; }
    grep -E '^([0-9T:Z-]+ )?(STOP|FAIL|RED|die|ERROR|⛔)' "$f" | tail -1 | cut -c1-200 | grep . || tail -1 "$f" | cut -c1-200
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
    selftest_seams
    selftest_judge
    selftest_streak
    selftest_pick
    selftest_lanes_argv
    printf -- '--- %s/%s rows ---\n' "$pass" "$((pass + fail))"
    [ "$fail" -eq 0 ]
}

# snip_run TEXT: run one of this file's own literal case-table setup snippets in THIS shell, so it sees the
# case table's functions and variables, exactly as the `eval` it replaces did. The text is written to a file
# under the selftest's $tmp and sourced; it is never an argument or an input from outside this file.
snip_run() {
    printf '%s\n' "$1" > "${tmp:?}/snip.sh" || return 2
    . "${tmp:?}/snip.sh"
}

# The stub end to end: a planted write in a stage is refused, recorded and turns the night red;
# a read runs the real tool. Also the by-path cargo call prepare_bump.sh makes.
selftest_stub() {
    local s="$tmp/stub" envf o rc
    mkdir -p "$s" || return 2
    envf=$(install_guard "$s") || { printf '  BROKE install_guard failed\n'; fail=$((fail + 1)); return; }
    t() { # t NAME WANT_RC WANT_OUT CMD
        local name=$1 wrc=$2 wout=$3; shift 3
        rc=0; o=$( . "$envf"; export WG_STAGE=plant; cd "$s" && bash -c "$*" 2>&1 ) || rc=$?
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

# The release scripts' rehearsal seams: the lane reader they read in place of a dispatch
# (lib_rehearsal.sh), a stage log that carries what autopilot.sh said only into RELEASE_AP, and the
# hand-off that makes the ship stage's bump commit the in-state origin's main.
selftest_seams() {
    local s="$tmp/seams" c=c0ffee00c0ffee00c0ffee00c0ffee00c0ffee00 n=0 o rc st h
    mkdir -p "$s" || return 2
    lrow() { # lrow LANE STATE RUN_ID RUN_HEAD REASON -> one bundle row, nightly_train.sh's columns
        printf '%s\tgate\t%s\tx.yml\t%s\t%s\tsuccess\t1\t-\t-\t%s\t0\n' "$@"
    }
    bundle() { # bundle DAY BUNDLE_C ROW...
        local f="$s/l$n/train/$1/bundle.tsv" bc=$2
        mkdir -p -- "$s/l$((n))/train/$1" || return 2; shift 2
        { printf 'lane\tkind\tstate\tproducer\trun_id\trun_head\tconclusion\tattempt\tstarted\tended\treason\thours_lost\n'
          printf '%s\n' "$@"; printf '# C\t%s\n' "$bc"; } > "$f"
    }
    l() { # l NAME WANT_RC WANT_OUT LANE SETUP -- rehearsal_lane for C over a fresh train dir
        local name=$1 wrc=$2 wout=$3 lane=$4
        n=$((n + 1)); mkdir -p -- "$s/l$((n))/train" || return 2
        snip_run "$5"
        rc=0; o=$( . "$SCRIPT_DIR/lib_rehearsal.sh" || exit 9
                   RELEASE_REHEARSAL_TRAIN="$s/l$n/train" RELEASE_REHEARSAL_C=${LC-$c} rehearsal_lane "$lane" 2>&1 ) || rc=$?
        if [ "$rc" = "$wrc" ] && [[ $o == *"$wout"* ]]; then pass=$((pass + 1))
        else printf '  BROKE %-44s rc=%s (want %s) out=%s\n' "$name" "$rc" "$wrc" "${o:0:160}"; fail=$((fail + 1)); fi
    }
    l lane_green_on_c_reads_its_run 0 4242 cleanroom-cpu \
        'bundle d "$c" "$(lrow cleanroom-cpu green 4242 $c -)"'
    l lane_red_is_not_green 1 "red: job failed" cleanroom-cpu 'bundle d "$c" "$(lrow cleanroom-cpu red 4242 $c "job failed")"'
    l lane_not_measured_is_not_green 1 "not_measured: no producer" assets 'bundle d "$c" "$(lrow assets not_measured - - "no producer")"'
    l lane_green_on_another_head_is_not 1 "not_measured: cleanroom-cpu read green on beef" cleanroom-cpu \
        'bundle d "$c" "$(lrow cleanroom-cpu green 4242 beef -)"'
    l lane_green_without_run_id_is_not 1 "read green with no run id" cleanroom-cpu 'bundle d "$c" "$(lrow cleanroom-cpu green - $c -)"'
    l lane_missing_row_is_not 1 "the train bundle has no assets row" assets 'bundle d "$c" "$(lrow cleanroom-cpu green 4242 $c -)"'
    l lane_bundle_for_another_c_is_not 1 "the train bundle is for beef" cleanroom-cpu 'bundle d beef "$(lrow cleanroom-cpu green 4242 $c -)"'
    l lane_two_bundles_is_not 1 "2 train bundles" cleanroom-cpu \
        'bundle d1 "$c" "$(lrow cleanroom-cpu green 4242 $c -)"; bundle d2 "$c" "$(lrow cleanroom-cpu green 4243 $c -)"'
    l lane_no_bundle_is_not 1 "0 train bundles" cleanroom-cpu ':'
    LC="" l lane_no_c_is_not 1 "RELEASE_REHEARSAL_C is unset" cleanroom-cpu 'bundle d "$c" "$(lrow cleanroom-cpu green 4242 $c -)"'
    # run_stage: autopilot.sh `say`s into RELEASE_AP/STATUS, not stdout; this stage's STOP and the gate
    # log it wrote reach the stage log, and an earlier stage's STATUS lines and untouched files do not.
    # rewritten.log keeps its mtime (epoch 0) across the rewrite: only its content says the stage wrote it
    st="$s/rs"; mkdir -p "$st/clone" "$st/logs" "$st/ap" || return 2
    printf '2026-10-07T00:00:00Z STOP an earlier stage\n' > "$st/ap/STATUS"
    printf 'stale\n' > "$st/ap/old-gate.log"
    printf 'before-stage\n' > "$st/ap/rewritten.log"; touch -d @0 -- "$st/ap/rewritten.log"
    printf 'export RELEASE_AP=%q\n' "$st/ap" > "$st/env"
    rc=0; run_stage "$st" "$st/env" ap:tag 'printf "2026-10-08T00:00:00Z STOP the tag gate refused\n" >> "$RELEASE_AP/STATUS"; printf "FAIL  NOT_MEASURED: x\n" > "$RELEASE_AP/tag-coverage.log"; printf "this-stage\n" > "$RELEASE_AP/rewritten.log"; touch -d @0 -- "$RELEASE_AP/rewritten.log"; exit 1' 0.0.0 || rc=$?
    o=$(stage_tail "$st" ap:tag)
    if [ "$rc" = 1 ] && [ "$o" = "2026-10-08T00:00:00Z STOP the tag gate refused" ] \
        && grep -qx 'FAIL  NOT_MEASURED: x' "$st/logs/ap_tag.log" && ! grep -q 'an earlier stage' "$st/logs/ap_tag.log" \
        && ! grep -q stale "$st/logs/ap_tag.log" && grep -qx this-stage "$st/logs/ap_tag.log"; then pass=$((pass + 1))
    else printf '  BROKE %-44s rc=%s tail=%s log=%s\n' run_stage_carries_this_stages_status "$rc" "$o" "$(tr '\n' '|' < "$st/logs/ap_tag.log" | cut -c1-200)"; fail=$((fail + 1)); fi
    # handoff_bump: one bump commit on C becomes origin's main and RELEASE_REHEARSAL_MC; none, or two, do not
    hb() { # hb NAME WANT_RC COMMITS -> a state dir with C, an in-state origin, a clone and its bump worktree
        local name=$1 wrc=$2 k=$3 i cc mc
        st="$s/hb-$name"; mkdir -p "$st/src" "$st/ap" || return 2
        g() { git -c user.name=t -c user.email=t@t -c commit.gpgsign=false -c core.hooksPath=/dev/null "$@"; }
        g -C "$st/src" init -q -b main && g -C "$st/src" commit -q --allow-empty -m C \
            && g clone -q --bare "$st/src" "$st/origin.git" && g clone -q "$st/origin.git" "$st/clone" \
            && g -C "$st/clone" worktree add -q --detach "$st/ap/bump" HEAD || return 2
        cc=$(g -C "$st/clone" rev-parse HEAD); : > "$st/env"
        for ((i = 0; i < k; i++)); do g -C "$st/ap/bump" commit -q --allow-empty -m "bump $i" || return 2; done
        h=$(g -C "$st/ap/bump" rev-parse HEAD)
        rc=0; handoff_bump "$st" "$st/env" "$cc" || rc=$?
        mc=$(g -C "$st/origin.git" rev-parse refs/heads/main)
        local ok=0
        if [ "$wrc" = 0 ]; then
            [ "$rc" = 0 ] && [ "$mc" = "$h" ] && grep -qx "export RELEASE_REHEARSAL_MC=$h" "$st/env" && ok=1
        else
            [ "$rc" != 0 ] && [ "$mc" = "$cc" ] && [ ! -s "$st/env" ] && ok=1
        fi
        if [ "$ok" = 1 ]; then pass=$((pass + 1))
        else printf '  BROKE %-44s rc=%s (want %s) main=%s bump=%s env=%s\n' "$name" "$rc" "$wrc" "${mc:0:9}" "${h:0:9}" "$(tr '\n' '|' < "$st/env")"; fail=$((fail + 1)); fi
    }
    hb handoff_one_bump_commit_becomes_main 0 1
    hb handoff_no_bump_commit_refuses 1 0
    hb handoff_two_commits_on_c_refuses 1 2
    # cascade-publish.sh --rehearse outside a rehearsal is refused before it reads anything (exit 2).
    # A missing cascade script is a BROKE row, never a skip.
    local cas=${REHEARSE_CASCADE:-$SCRIPT_DIR/../cascade-publish.sh} o
    rc=0; o=$(env -u RELEASE_REHEARSAL bash "$cas" --rehearse 2>&1 < /dev/null) || rc=$?
    if [ "$rc" = 2 ] && [[ $o == *"--rehearse runs only under RELEASE_REHEARSAL=1"* ]]; then pass=$((pass + 1))
    else printf '  BROKE %-44s rc=%s (want 2) %s\n' cascade_rehearse_outside_rehearsal_refuses "$rc" "${o:0:160}"; fail=$((fail + 1)); fi
    # Both libs are SOURCED into the release scripts, so a file-scope `set` would change autopilot.sh's
    # own options on release day (CLAUDE.md, "a sourced library must be option-neutral").
    local lib hits
    for lib in lib_write_guard.sh lib_rehearsal.sh; do
        hits=$(grep -nE '^set[[:space:]]+[-+]' "$SCRIPT_DIR/$lib" 2>&1) || true
        if [ -f "$SCRIPT_DIR/$lib" ] && [ -z "$hits" ]; then pass=$((pass + 1))
        else printf '  BROKE %-44s %s\n' "${lib%.sh}_option_neutral" "${hits:-missing $SCRIPT_DIR/$lib}"; fail=$((fail + 1)); fi
    done
    printf '  %s seam rows\n' "$((n + 7))"
}

# fixture_night DIR -> a night where every stage is green on C, the trace shows the freeze inside the
# bump, and the bump commit carries a fleet trailer: the anti-vacuity arm (it must be GREEN).
fixture_night() {
    local d=$1 name
    mkdir -p "$d/logs" || return 2
    printf 'C=c0ffee\nV=0.71.0\nIDS=tree=%s stop=%s policy=%s\n' "$(printf a%.0s {1..40})" "$(printf b%.0s {1..40})" "$(printf c%.0s {1..40})" > "$d/night.env"
    printf 'stage\trc\tcommit\tseconds\n' > "$d/stages.tsv"
    while IFS='|' read -r name _ _; do
        case $name in lanes|t2|bump|summary|ship) printf '%s\t0\tc0ffee\t1\n' "$name" ;; *) printf '%s\t0\tb0b\t1\n' "$name" ;; esac >> "$d/stages.tsv"
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
        snip_run "$mut"
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
    j a_night_without_ids_is_red 1 "RED   ids not recorded (night.env has no IDS line)" 'sed -i "/^IDS=/d" "$d/night.env"'
    j a_night_whose_ids_failed_is_red 1 "RED   ids not recorded (c0ffee has no blob contracts/release-ready-v1.yaml)" \
        'sed -i "s|^IDS=.*|IDS=-\nIDS_WHY=c0ffee has no blob contracts/release-ready-v1.yaml|" "$d/night.env"'
    j a_short_id_is_red 1 "RED   ids not recorded" 'sed -i "s/^IDS=tree=a/IDS=tree=/" "$d/night.env"'
    printf '  %s judge rows\n' "$n"
}

# --streak on a fixture repository: one commit per id path changed, and run histories on them. The pass commit's
# ids are release day's; a night on other ids resets the count. Then night_env on the same commits, and the read
# itself through a planted gh: one call, its ETag sent back, a 304 reused, the floor kept.
selftest_streak() {
    local g="$tmp/sr" k=0 p0=$((pass + fail)) o rc c0 c1 c2 c3 c4 c5 c6 none=0123456789abcdef0123456789abcdef01234567
    sg() { git -C "$g" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@invalid -c commit.gpgsign=false "$@"; }
    mkdir -p "$g/scripts/release" "$g/contracts" && sg init -q || return 2
    printf 'a\n' > "$g/scripts/release/a.sh"; printf 'p\n' > "$g/$IDS_POLICY"
    sg add -A && sg commit -qm c0 && c0=$(sg rev-parse HEAD) || return 2
    printf 's\n' > "$g/$IDS_STOP"; sg add -A && sg commit -qm c1 && c1=$(sg rev-parse HEAD) || return 2
    printf 'x\n' > "$g/README"; sg add -A && sg commit -qm c2 && c2=$(sg rev-parse HEAD) || return 2
    printf 'b\n' > "$g/scripts/release/a.sh"; sg commit -qam c3 && c3=$(sg rev-parse HEAD) || return 2
    sg checkout -q --detach "$c2" && printf 't\n' > "$g/$IDS_STOP" && sg commit -qam c4 && c4=$(sg rev-parse HEAD) || return 2
    sg checkout -q --detach "$c2" && printf 'q\n' > "$g/$IDS_POLICY" && sg commit -qam c5 && c5=$(sg rev-parse HEAD) || return 2
    sg checkout -q --detach "$c2" && sg rm -q "$IDS_STOP" && mkdir -p "$g/$IDS_STOP" && printf 's\n' > "$g/$IDS_STOP/x" \
        && sg add -A && sg commit -qm c6 && c6=$(sg rev-parse HEAD) || return 2
    hist() { # hist FILE ROW... with ROW = run|YYYY-MM-DD|event|conclusion|attempt|head
        local f=$1 r id dd ev co at hd; shift
        printf 'run_id\tcreated_at\tbranch\tevent\tconclusion\tattempt\thead_sha\n' > "$f"
        for r in "$@"; do
            IFS='|' read -r id dd ev co at hd <<< "$r"
            printf '%s\t%sT05:20:00Z\tmain\t%s\t%s\t%s\t%s\n' "$id" "$dd" "$ev" "$co" "$at" "$hd" >> "$f"
        done
    }
    sr() { # sr NAME WANT_RC WANT_OUT REV ROW...
        local name=$1 wrc=$2 wout=$3 rev=$4; shift 4
        k=$((k + 1)); hist "$tmp/h$k.tsv" "$@"
        rc=0; o=$(streak --commit "$rev" --repo "$g" --history "$tmp/h$k.tsv" --as-of 2026-10-08 --cache "$tmp/sc$k" 2>&1) || rc=$?
        if [ "$rc" = "$wrc" ] && [[ $o == *"$wout"* ]]; then pass=$((pass + 1))
        else printf '  BROKE %-48s rc=%s (want %s): %s\n' "$name" "$rc" "$wrc" "$(printf '%s' "$o" | grep -v '^TARGET' | head -3 | tr '\n' '|')"; fail=$((fail + 1)); fi
    }
    local n6="1|2026-10-06|schedule|success|1" n7="2|2026-10-07|schedule|success|1" n8="3|2026-10-08|schedule|success|1"
    sr three_nights_on_the_same_ids_are_ready 0 "RECEIPT run 1 night 2026-10-06 head $c1 tree=" "$c2" "$n6|$c1" "$n7|$c2" "$n8|$c2"
    sr a_release_scripts_change_after_them_resets 1 "RESET run 3 (night 2026-10-08, head $c2): tree differ" "$c3" "$n6|$c1" "$n7|$c2" "$n8|$c2"
    sr a_stop_list_change_resets 1 "RESET run 3 (night 2026-10-08, head $c2): stop differ" "$c4" "$n6|$c1" "$n7|$c2" "$n8|$c2"
    sr a_policy_file_change_resets 1 "RESET run 3 (night 2026-10-08, head $c2): policy differ" "$c5" "$n6|$c1" "$n7|$c2" "$n8|$c2"
    sr a_night_on_other_ids_breaks_the_streak 1 "not ready: night 2026-10-07 was red (run 2, failure)" "$c2" "$n6|$c2" "$n7|$c3" "$n8|$c2"
    sr a_night_without_the_stop_list_resets 1 "head $c0): $c0 has no blob $IDS_STOP" "$c2" "$n6|$c1" "$n7|$c2" "$n8|$c0"
    sr a_head_not_in_the_repo_is_not_measured 2 "not_measured: a counted night's head is not in" "$c2" "$n6|$c1" "$n7|$c2" "$n8|$none"
    sr a_pass_commit_without_the_stop_list_is_not_measured 2 "not_measured: the pass commit's ids cannot be read" "$c0" "$n6|$c0" "$n7|$c0" "$n8|$c0"
    sr a_stop_list_that_is_a_directory_is_not_measured 2 "has no blob $IDS_STOP" "$c6" "$n6|$c2" "$n7|$c2" "$n8|$c2"
    sr day_runs_do_not_count 1 "not ready" "$c2" "$n6|$c1" "$n7|$c2" "4|2026-10-08|workflow_dispatch|success|1|$none"
    sr a_running_night_on_the_same_ids_waits 0 "RECEIPT run 2 night 2026-10-07" "$c2" "0|2026-10-05|schedule|success|1|$c1" "$n6|$c1" "$n7|$c2" "3|2026-10-08|schedule||1|$c2"
    sr a_running_night_on_other_ids_is_a_reset 1 "not ready: night 2026-10-08 was red (run 3, failure)" "$c2" "0|2026-10-05|schedule|success|1|$c1" "$n6|$c1" "$n7|$c2" "3|2026-10-08|schedule||1|$c3"
    printf 'run_id\tcreated_at\tbranch\tevent\tconclusion\tattempt\n' > "$tmp/h6col.tsv"
    rc=0; o=$(streak --commit "$c2" --repo "$g" --history "$tmp/h6col.tsv" --as-of 2026-10-08 --cache "$tmp/sc6col" 2>&1) || rc=$?
    if [ "$rc" = 2 ] && [[ $o == *"not a seven-column run history"* ]]; then pass=$((pass + 1))
    else printf '  BROKE %-48s rc=%s (want 2): %s\n' a_six_column_history_is_not_measured "$rc" "$o"; fail=$((fail + 1)); fi
    rc=0; o=$(streak --commit "$c2" --repo "$g" --cache "$tmp/scx" 2>&1) || rc=$?
    if [ "$rc" = 3 ]; then pass=$((pass + 1)); else printf '  BROKE %-48s rc=%s (want 3)\n' a_count_with_no_date_is_a_caller_error "$rc"; fail=$((fail + 1)); fi
    # night_env: the ids of the night's commit, or IDS=- with the reason, which the judge turns red
    ne() { # ne NAME C WANT_LINE
        mkdir -p "$tmp/ne$k" && night_env "$tmp/ne$k" "$2" 0.71.0 "$g" || return 2
        if grep -qxF -- "$3" "$tmp/ne$k/night.env"; then pass=$((pass + 1))
        else printf '  BROKE %-48s want "%s": %s\n' "$1" "$3" "$(grep '^IDS' "$tmp/ne$k/night.env" | tr '\n' '|')"; fail=$((fail + 1)); fi
        k=$((k + 1))
    }
    ne a_night_records_its_commits_ids "$c1" "IDS=$(release_ids "$g" "$c1")"
    ne a_night_without_the_stop_list_records_why "$c0" "IDS_WHY=$c0 has no blob $IDS_STOP"
    local d="$tmp/ne-judge"
    fixture_night "$d" && night_env "$d" "$c0" 0.71.0 "$g" || return 2
    rc=0; o=$(judge "$d" 2>&1) || rc=$?
    if [ "$rc" = 1 ] && [[ $o == *"RED   ids not recorded ($c0 has no blob $IDS_STOP)"* ]]; then pass=$((pass + 1))
    else printf '  BROKE %-48s rc=%s (want 1)\n' a_night_on_a_tree_without_the_list_is_red "$rc"; fail=$((fail + 1)); fi
    # the read: a planted gh answers 200 with an ETag, then 304 to that ETag; the rate floor is kept
    local gs="$tmp/ghs"
    mkdir -p "$gs/bin" || return 2
    jq -n --arg a "$c1" --arg b "$c2" '{workflow_runs: [
        {id: 3, created_at: "2026-10-08T05:20:00Z", head_branch: "main", event: "schedule", conclusion: "success", run_attempt: 1, head_sha: $b},
        {id: 2, created_at: "2026-10-07T05:20:00Z", head_branch: "main", event: "schedule", conclusion: "success", run_attempt: 1, head_sha: $b},
        {id: 1, created_at: "2026-10-06T05:20:00Z", head_branch: "main", event: "schedule", conclusion: "success", run_attempt: 1, head_sha: $a},
        {id: 4, created_at: "2026-10-09T05:20:00Z", head_branch: "main", event: "schedule", conclusion: null, run_attempt: 1, head_sha: $b}]}' > "$gs/body.json" || return 2
    cat > "$gs/bin/gh" <<'GH'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$GHS/calls"
case " $* " in *" rate_limit "*) printf '%s\n' "${GHS_RATE:-5000 4000}"; exit 0 ;; esac
case "${GHS_RATE:-}" in *" 0") printf 'HTTP/2.0 403 Forbidden\r\n\r\n{"message": "API rate limit exceeded"}'; exit 1 ;; esac
for a in "$@"; do [ "$a" = 'If-None-Match: "e1"' ] && { printf 'HTTP/2.0 304 Not Modified\r\n\r\n'; exit 0; }; done
printf 'HTTP/2.0 200 OK\r\nEtag: "e1"\r\nContent-Type: application/json\r\n\r\n'; cat "$GHS/body.json"
GH
    chmod +x "$gs/bin/gh" || return 2
    rd() { # rd NAME WANT_RC WANT_OUT WANT_RUNS_CALLS WANT_ETAG_CALLS [RATE [FLAG]]
        rc=0; o=$(PATH="$gs/bin:$PATH" GHS="$gs" GHS_RATE="${6:-5000 4000}" streak --commit "$c2" --repo "$g" --as-of 2026-10-08 --cache "$gs/cache" ${7:+"$7"} 2>&1) || rc=$?
        local runs etag
        runs=$(grep -c 'actions/workflows/release-rehearsal-nightly.yml/runs?branch=main&event=schedule' "$gs/calls" 2>/dev/null)
        etag=$(grep -c 'If-None-Match: "e1"' "$gs/calls" 2>/dev/null)
        if [ "$rc" = "$2" ] && [[ $o == *"$3"* ]] && [ "${runs:-0}" = "$4" ] && [ "${etag:-0}" = "$5" ]; then pass=$((pass + 1))
        else printf '  BROKE %-48s rc=%s (want %s) runs=%s etag=%s: %s\n' "$1" "$rc" "$2" "${runs:-0}" "${etag:-0}" "$(printf '%s' "$o" | grep -v '^TARGET' | head -2 | tr '\n' '|')"; fail=$((fail + 1)); fi
    }
    rd the_first_read_is_one_call 0 "RECEIPT run 3 night 2026-10-08 head $c2" 1 0
    rd the_second_read_sends_the_etag_and_reuses_a_304 0 "RECEIPT run 2 night 2026-10-07 head $c2" 2 1
    rd a_read_under_the_floor_is_not_measured 2 "not_measured: core remaining 999 under 1000" 2 1 "5000 999"
    rd the_pass_start_reads_under_the_floor 0 "RECEIPT run 3 night 2026-10-08 head $c2" 3 2 "5000 999" --release-path
    rd a_refused_read_at_the_pass_start_is_still_not_measured 2 "not_measured: workflow runs read: HTTP 403" 4 3 "5000 0" --release-path
    printf '  %s streak rows\n' "$((pass + fail - p0))"
}

# --pick-c on a fixture origin: main is m0..m5 on its first parents, with a side commit s1 merged at m2 and dated after
# m1, and each row writes the evidence branch it needs. A full clone fetches from it; measured-cpu's lines are planted
# in place of the infra read, except in the last read row, which runs the real release_lanes.sh with no token.
selftest_pick() {
    local p="$tmp/pick" p0=$((pass + fail)) o e rc t m0 m1 s1 m2 m3 m4 m5 idxa cpua
    pg() { git -C "$p/origin.git" -c user.name=t -c user.email=t@invalid "$@"; }
    mkdir -p "$p" && git init -q --bare "$p/origin.git" && t=$(pg mktree < /dev/null) || return 2
    cm() { # cm DATE MSG PARENT... -> a commit of the empty tree, dated DATE (epoch seconds)
        local d=$1 m=$2 x a=(); shift 2
        for x in "$@"; do a+=(-p "$x"); done
        GIT_COMMITTER_DATE="@$d +0000" GIT_AUTHOR_DATE="@$d +0000" pg commit-tree "$t" -m "$m" "${a[@]}"
    }
    m0=$(cm 100 m0) && m1=$(cm 200 m1 "$m0") && s1=$(cm 300 s1 "$m0") && m2=$(cm 400 m2 "$m1" "$s1") \
        && m3=$(cm 500 m3 "$m2") && m4=$(cm 600 m4 "$m3") && m5=$(cm 700 m5 "$m4") || return 2
    pg update-ref refs/heads/main "$m5" && pg symbolic-ref HEAD refs/heads/main || return 2
    git clone -q -- "$p/origin.git" "$p/clone" 2>/dev/null || return 2
    ev() { # ev INDEX SHA:STATE... -> origin's evidence branch: models-crux/INDEX (none when INDEX is -) and one bundle per
        local i=$1 b v tb l mc root c # SHA:STATE, models-crux/<SHA>/verdict saying state=STATE
        shift
        l=""
        if [ "$i" != - ]; then i=$(printf '%s' "$i" | pg hash-object -w --stdin) || return 2; l=$(printf '100644 blob %s\tINDEX' "$i"); fi
        for b in "$@"; do
            v=$(printf 'state=%s\nreason=fixture\n' "${b#*:}" | pg hash-object -w --stdin) || return 2
            tb=$(printf '100644 blob %s\tverdict\n' "$v" | pg mktree) || return 2
            l="$l${l:+$'\n'}$(printf '040000 tree %s\t%s' "$tb" "${b%%:*}")"
        done
        mc=$(printf '%s\n' "$l" | sed '/^$/d' | pg mktree) && root=$(printf '040000 tree %s\t%s\n' "$mc" models-crux | pg mktree) \
            && c=$(pg commit-tree "$root" -m evidence) && pg update-ref refs/heads/nightly-evidence "$c"
    }
    pk() { # pk NAME WANT_RC WANT_STDOUT WANT_STDERR [ARG...] -- pick_c on the clone, measured-cpu answering CPU, rc CPU_RC
        local name=$1 wrc=$2 wout=$3 werr=$4; shift 4
        rc=0; o=$( cpu_measured() { [ -z "${CPU:-}" ] || printf '%s\n' "$CPU"; return "${CPU_RC:-0}"; }
                   pick_c --repo "$p/clone" --cache "$p/cache" "$@" 2> "$p/err" ) || rc=$?
        e=$(cat -- "$p/err")
        if [ "$rc" = "$wrc" ] && [ "$o" = "$wout" ] && [[ $e == *"$werr"* ]]; then pass=$((pass + 1))
        else printf '  BROKE %-48s rc=%s (want %s) out=%s err=%s\n' "$name" "$rc" "$wrc" "${o:0:41}" "$(printf '%s' "$e" | tail -n 2 | tr '\n' '|' | cut -c1-200)"; fail=$((fail + 1)); fi
    }
    idxa=$(printf '%s not_measured\n%s red\n%s green\n%s green\n' "$m5" "$m3" "$s1" "$m1")
    cpua=$(printf '%s\tgreen\t11\n%s\tred\t12\n%s\tgreen\t13\n%s\tgreen\t14\n%s\tred\t15\n' "$m5" "$m4" "$s1" "$m2" "$m1")
    ev "$idxa" "$m5:not_measured" "$m3:red" "$s1:green" "$m1:green" || return 2
    CPU=$cpua pk c_is_the_newest_first_parent_commit_both_measured 0 "$m1" "pick: C $m1 (main~4): models-crux green, cleanroom-cpu red run 15"
    CPU=$cpua PICK_DEPTH=4 pk the_walk_is_bounded 2 "" "no commit in main's newest 4 first-parent commits was measured by both"
    ev "$m5 red" "$m5:red" || return 2
    CPU=$(printf '%s\tgreen\t21' "$m5") pk a_red_models_bundle_on_the_head_is_c 0 "$m5" "pick: C $m5 (main~0): models-crux red, cleanroom-cpu green run 21"
    ev "$m1 green" || return 2
    CPU=$cpua pk a_measured_line_with_no_bundle_is_not_measured 2 "" "INDEX says green for $m1, and its bundle's verdict says nothing"
    ev "$m1 green" "$m1:red" || return 2
    CPU=$cpua pk a_bundle_that_says_otherwise_is_not_measured 2 "" "INDEX says green for $m1, and its bundle's verdict says red"
    ev "$(printf '%s green\n%s pass' "$m1" "$m3")" "$m1:green" "$m3:red" || return 2
    CPU=$cpua pk an_index_line_out_of_format_is_not_measured 2 "" "INDEX line 2 is not '<40-hex> green|red|not_measured': $m3 pass"
    ev "$(printf '%s green\n%s red' "$m1" "$m1")" "$m1:green" || return 2
    CPU=$cpua pk two_index_lines_for_a_commit_are_not_measured 2 "" "INDEX has two lines for $m1"
    ev - "$m1:green" || return 2
    CPU=$cpua pk no_index_is_not_measured 2 "" "nightly-evidence has no blob models-crux/INDEX"
    ev "$idxa" "$m5:not_measured" "$m3:red" "$s1:green" "$m1:green" || return 2
    CPU=$cpua CPU_RC=2 pk a_not_measured_cpu_read_is_not_measured 2 "" "cleanroom-cpu: release_lanes.sh measured-cpu read nothing"
    CPU=$(printf '%s green 14' "$m1") pk a_cpu_line_out_of_format_is_not_measured 2 "" "measured-cpu printed a line that is not 'sha TAB green|red TAB run': $m1 green 14"
    CPU=$cpua pk an_unreachable_remote_is_not_measured 2 "" "main was not fetched from nowhere" --remote nowhere
    pg update-ref -d refs/heads/nightly-evidence || return 2
    CPU=$cpua pk an_evidence_branch_gone_is_not_read_from_an_old_fetch 2 "" "nightly-evidence was not fetched from origin"
    ev "$idxa" "$m5:not_measured" "$m3:red" "$s1:green" "$m1:green" || return 2
    git clone -q -- "$p/origin.git" "$p/shallow" 2>/dev/null && printf '%s\n' "$m3" > "$p/shallow/.git/shallow" || return 2
    CPU=$cpua pk a_shallow_clone_is_not_measured 2 "" "is not a full clone (shallow, or no git)" --repo "$p/shallow"
    rc=0; o=$( unset INFRA_TOKEN; pick_c --repo "$p/clone" --cache "$p/cache" 2>&1 ) || rc=$?
    if [ "$rc" = 2 ] && [[ $o == *"not_measured: no infra read token (INFRA_ACTIONS_READ)"* ]]; then pass=$((pass + 1))
    else printf '  BROKE %-48s rc=%s (want 2): %s\n' the_cpu_side_is_release_lanes_measured_cpu "$rc" "$(printf '%s' "$o" | tail -n 2 | tr '\n' '|')"; fail=$((fail + 1)); fi
    rc=0; o=$( pick_c --repo "$p/clone" 2>&1 ) || rc=$?
    if [ "$rc" = 3 ] && [[ $o == *"--pick-c needs --cache DIR"* ]]; then pass=$((pass + 1))
    else printf '  BROKE %-48s rc=%s (want 3): %s\n' a_pick_without_a_cache_is_a_caller_error "$rc" "$o"; fail=$((fail + 1)); fi
    printf '  %s pick rows\n' "$((pass + fail - p0))"
}

# The night hands its train C and its own run: run_night --stages lanes on a fixture source whose nightly_train.sh prints
# its arguments. The lanes stage passes --commit C always, and --in-run RUN only when the night has a run.
selftest_lanes_argv() {
    local g="$tmp/ln" p0=$((pass + fail)) k=0 c b rl sc rt o rc
    lg() { git -C "$g" -c user.name=t -c user.email=t@invalid "$@"; }
    mkdir -p "$g" && git init -q "$g" || return 2
    printf '%s\n' "printf 'arg %s\\n' \"\$@\"" > "$tmp/ln-train.sh" || return 2
    b=$(lg hash-object -w -- "$tmp/ln-train.sh") && rl=$(printf '100644 blob %s\tnightly_train.sh\n' "$b" | lg mktree) \
        && sc=$(printf '040000 tree %s\trelease\n' "$rl" | lg mktree) && rt=$(printf '040000 tree %s\tscripts\n' "$sc" | lg mktree) \
        && c=$(lg commit-tree "$rt" -m lanes) && lg update-ref refs/heads/main "$c" && lg symbolic-ref HEAD refs/heads/main || return 2
    la() { # la NAME RUN -- the lanes stage's arguments, with --in-run RUN when RUN is set
        local name=$1 run=$2 st want got
        k=$((k + 1)); st="$tmp/ln-night$k"
        o=$( run_night --state "$st" --source "$g" --commit "$c" --version 9.9.9 --stages lanes ${run:+--in-run "$run"} 2>&1 )
        st=$(realpath -- "$st") || st="$tmp/ln-night$k"
        want="--out $st/train --commit $c${run:+ --in-run $run} "
        got=$(sed -n 's/^arg //p' "$st/logs/lanes.log" 2>/dev/null | tr '\n' ' ')
        if [ "$got" = "$want" ]; then pass=$((pass + 1))
        else printf '  BROKE %-48s got "%s" want "%s": %s\n' "$name" "$got" "$want" "$(printf '%s' "$o" | tail -n 2 | tr '\n' '|')"; fail=$((fail + 1)); fi
    }
    la the_lanes_stage_reads_c_in_the_nights_run 4242
    la the_lanes_stage_without_a_run_reads_c ""
    rc=0; o=$( run_night --state "$tmp/ln-bad" --source "$g" --commit "$c" --version 9.9.9 --stages lanes --in-run 12x 2>&1 ) || rc=$?
    if [ "$rc" = 3 ] && [[ $o == *"--in-run must be a run id"* ]] && [ ! -e "$tmp/ln-bad" ]; then pass=$((pass + 1))
    else printf '  BROKE %-48s rc=%s (want 3): %s\n' an_in_run_that_is_not_a_run_id_is_a_caller_error "$rc" "$o"; fail=$((fail + 1)); fi
    printf '  %s night rows\n' "$((pass + fail - p0))"
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
        dir=$(mktemp -d "${tmp:?}/m.XXXXXX") || return 2
        cp -- "$SCRIPT_PATH" "$dir/rehearse.sh"; cp -- "$SCRIPT_DIR/lib_write_guard.sh" "$SCRIPT_DIR/lib_rehearsal.sh" "$SCRIPT_DIR/nightly_greens.sh" "$SCRIPT_DIR/release_lanes.sh" "$dir/"
        sed -i -e "$expr" "$dir/$file"
        if cmp -s "$dir/$file" "$SCRIPT_DIR/$file"; then
            printf '  BROKE %-40s changed nothing: its pattern no longer matches\n' "$name"; fail=$((fail + 1)); continue
        fi
        if ! bash -n "$dir/$file" 2>/dev/null; then
            printf '  BROKE %-40s does not parse: a RED from it would prove nothing\n' "$name"; fail=$((fail + 1)); continue
        fi
        rc=0; o="$(REHEARSE_CASCADE="$SCRIPT_DIR/../cascade-publish.sh" bash "$dir/rehearse.sh" --selftest < /dev/null 2>&1)" || rc=$?
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
lane_unset_env_reads         lib_rehearsal.sh    s/\[ -n "\$train" \] \&\& \[ -n "\$c" \] ||/true ||/
lane_bundle_count_ignored    lib_rehearsal.sh    s/\[ "\$n" = 1 \] ||/true ||/
lane_bundle_c_ignored        lib_rehearsal.sh    s/\[ "\$bc" = "\$c" \] ||/true ||/
lane_red_is_green            lib_rehearsal.sh    s/\[ "\$state" = green \] ||/true ||/
lane_other_head_green        lib_rehearsal.sh    s/\[ "\$head" = "\$c" \] ||/true ||/
lane_no_run_id_green         lib_rehearsal.sh    s/\[\[ \$run =~ \^\[0-9\]+\$ \]\] ||/true ||/
stage_status_whole_file      rehearse.sh         s/tail -c +"\$((s0 + 1))" -- "\$st\/ap\/STATUS"/cat -- "$st\/ap\/STATUS"/
stage_status_not_captured    rehearse.sh         s/\[ -f "\$st\/ap\/STATUS" \] \&\& {/false \&\& {/
stage_gate_files_dropped     rehearse.sh         s/done < <(ap_gate_files "\$st\/ap" | /done < <(true | /
stage_old_files_kept         rehearse.sh         s/LC_ALL=C comm -13 <(printf '%s\\n' "\$before") - | cut -f1)/cut -f1)/
stage_content_ignored        rehearse.sh         s/"\$(sha256sum < "\$1\/\$f")"/-/
handoff_any_parent           rehearse.sh         s/\[ "\$(git -C "\$st\/ap\/bump" rev-parse --verify -q "\$h^")" = "\$c" \] || return 1/:/
handoff_main_not_moved       rehearse.sh         s/git -C "\$st\/origin.git" fetch -q "\$st\/clone" "+refs\/rehearsal\/bump:refs\/heads\/main" || return 1/:/
lib_rehearsal_sets_errexit   lib_rehearsal.sh    s/^rehearsal_lane() {$/set -e\nrehearsal_lane() {/
lib_guard_sets_option        lib_write_guard.sh  s/^wg_inside() {$/set +H\nwg_inside() {/
handoff_mc_not_exported      rehearse.sh         s/printf 'export RELEASE_REHEARSAL_MC=%q\\n' "\$h" >> "\$env"/:/
ids_reset_skipped            rehearse.sh         s/\$5 = "failure"/$5 = $5/
ids_tree_not_release_scripts rehearse.sh         s/^IDS_TREE=scripts\/release$/IDS_TREE=contracts/
ids_policy_is_the_stop_list  rehearse.sh         s/^IDS_POLICY=contracts\/model-capability-ladder-v1.yaml$/IDS_POLICY=contracts\/release-ready-v1.yaml/
ids_type_unchecked           rehearse.sh         s/ \&\& \[ "\$(git -C "\$g" cat-file -t "\$id")" = "\$want" \]//
ids_reset_names_nothing      rehearse.sh         s/if (x\[i\] != y\[i\]) out = out/if (0) out = out/
missing_head_is_a_reset      rehearse.sh         s/^            1) printf '%s\\t-\\t%s\\n' "\$head"/            1|2) printf '%s\\t-\\t%s\\n' "$head"/
day_runs_get_ids             rehearse.sh         s/NR > 1 \&\& \$3 == "main" \&\& \$4 == "schedule" { print \$7 }/NR > 1 { print $7 }/
six_column_history_read      rehearse.sh         s/\[ "\$hdr" = "\$(printf 'run_id/[ -n "$hdr" ] || [ "$hdr" = "$(printf 'run_id/
receipts_not_printed         rehearse.sh         s/printf "RECEIPT run %s night/printf "RCPT run %s night/
etag_not_sent                rehearse.sh         s/\[ -s "\$cache\/runs.etag" \] \&\& \[ -s "\$cache\/runs.json" \] \&\& hdr=/false \&\& hdr=/
rate_floor_ignored           rehearse.sh         s/\[ "\$rem" -ge "\$fl" \] || { echo "core remaining/true || { echo "core remaining/
night_ids_dropped            rehearse.sh         s/then printf 'IDS=%s\\n' "\$ids"/then :/
judge_ids_unchecked          rehearse.sh         s/if \[\[ \$ids =~ \^tree=/if true || [[ $ids =~ ^tree=/
release_path_keeps_floor     rehearse.sh         s/if \[ "\$relpath" = 1 \]; then fl=0; fi/:/
nights_lose_the_floor        rehearse.sh         s/relpath=\${3:-0}/relpath=1/
pick_not_measured_line_counts rehearse.sh         s/\[ "\$state" != not_measured \] || continue/true || continue/
pick_bundle_unchecked        rehearse.sh         s/\[ "\$v" = "\$state" \] || {/true || {/
pick_not_first_parent        rehearse.sh         s/rev-list --first-parent -n "\$PICK_DEPTH"/rev-list -n "$PICK_DEPTH"/
pick_unbounded               rehearse.sh         s/ -n "\$PICK_DEPTH" refs\/rehearsal\/main/ refs\/rehearsal\/main/
pick_models_red_refused      rehearse.sh         s/ (green|red|not_measured)\$'/ (green|not_measured)$'/
pick_cpu_red_refused         rehearse.sh         s/\\t(green|red)\\t/\\t(green)\\t/
pick_models_side_unchecked   rehearse.sh         s/if \[ -n "\${models\[\$sha\]:-}" \] && /if /
pick_cpu_side_unchecked      rehearse.sh         s/ && \[ -n "\${cpu\[\$sha\]:-}" \]; then/; then/
pick_stale_evidence_read     rehearse.sh         s/refs\/rehearsal\/\$EVIDENCE_BRANCH" || {/refs\/rehearsal\/$EVIDENCE_BRANCH" || true || {/
pick_main_fetch_ignored      rehearse.sh         s/"+refs\/heads\/main:refs\/rehearsal\/main" || {/"+refs\/heads\/main:refs\/rehearsal\/main" || true || {/
pick_index_state_unchecked   rehearse.sh         s/ (green|red|not_measured)\$'/ ([a-z_]+)$'/
pick_duplicates_allowed      rehearse.sh         s/\[ -z "\${seen\[\$sha\]:-}" \] || {/true || {/
pick_cpu_rc_ignored          rehearse.sh         s/rows=\$(cpu_measured "\$cache\/cpu") || {/rows=$(cpu_measured "$cache\/cpu") || true || {/
pick_cpu_rows_unvalidated    rehearse.sh         s/\[\[ \$line =~ \$cre \]\] || {/true || {/
pick_shallow_allowed         rehearse.sh         s/--is-shallow-repository 2>\/dev\/null)" = false \] || {/--is-shallow-repository 2>\/dev\/null)" = false ] || true || {/
pick_cpu_reader_not_called   rehearse.sh         s/^cpu_measured() { bash .*; }$/cpu_measured() { :; }/
pick_c_not_printed           rehearse.sh         s/printf '%s\\n' "\$sha"; return 0/return 0/
pick_cache_optional          rehearse.sh         s/^    \[ -n "\$cache" \] || die3 "--pick-c needs --cache DIR"$/    :/
pick_index_last_line_dropped rehearse.sh         s/while IFS= read -r line || \[ -n "\$line" \]; do/while IFS= read -r line; do/
pick_no_index_read_as_empty  rehearse.sh         s/\/INDEX" > "\$f" 2>\/dev\/null || {/\/INDEX" > "$f" 2>\/dev\/null || true || {/
pick_wrong_models_dir        rehearse.sh         s/^MODELS_DIR=models-crux$/MODELS_DIR=models/
pick_wrong_evidence_branch   rehearse.sh         s/^EVIDENCE_BRANCH=nightly-evidence$/EVIDENCE_BRANCH=evidence/
pick_receipt_depth_wrong     rehearse.sh         s/"\$sha" "\$k" "\${models\[\$sha\]}"/"$sha" 0 "${models[$sha]}"/
lanes_stage_no_commit        rehearse.sh         s/ --commit "\$RELEASE_REHEARSAL_C"//
lanes_stage_in_run_dropped   rehearse.sh         s/ \${RELEASE_REHEARSAL_IN_RUN:+--in-run "\$RELEASE_REHEARSAL_IN_RUN"}//
lanes_stage_in_run_always    rehearse.sh         s/\${RELEASE_REHEARSAL_IN_RUN:+--in-run "\$RELEASE_REHEARSAL_IN_RUN"}/--in-run "$RELEASE_REHEARSAL_IN_RUN"/
guard_env_drops_in_run       rehearse.sh         s/ RELEASE_REHEARSAL_IN_RUN=%q\\n' "\$c" "\$st\/train" "\$run"/\\n' "$c" "$st\/train"/
run_night_drops_in_run       rehearse.sh         s/install_guard "\$st" "\$commit" "\$inrun"/install_guard "$st" "$commit"/
in_run_unvalidated           rehearse.sh         s/^    \[ -z "\$inrun" \] || \[\[ \$inrun =~ .*must be a run id.*$/    :/
MUTANTS
    printf -- '--- %s/%s mutants killed ---\n' "$pass" "$((pass + fail))"
    [ "$fail" -eq 0 ]
}

case "${1:-}" in
    --run) shift; run_night "$@" ;;
    --judge) [ -n "${2:-}" ] || die3 "--judge DIR"; judge "$2" ;;
    --classify) [ $# -ge 3 ] || die3 "--classify TOOL CWD [ARG...]"; t=$2; d=$3; shift 3; wg_classify "$t" "${WG_STATE:-/nonexistent-state}" "$d" "$@" ;;
    --selftest) selftest ;;
    --streak) shift; streak "$@" ;;
    --pick-c) shift; pick_c "$@" ;;
    --mutants) mutants ;;
    -h|--help) awk 'NR > 1 && /^set -uo pipefail$/ { exit } NR > 1' "$0" ;;
    *) die3 "usage: rehearse.sh --run|--judge|--streak|--pick-c|--classify|--selftest|--mutants (see --help)" ;;
esac
