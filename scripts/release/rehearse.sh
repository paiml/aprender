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
# green on the night (and, for D7, the night's own bump commit shows the fix). An unreached stage
# clears nothing. Every D-row that is not CLEAR is named in the verdict.
# D3 and D6 are retired: #4967 took their checks off release day, and the rehearsal runs what release
# day runs. D3 was the tag step's coverage gate; #4932 fixed its version-only rule, and the
# dogfood step's coverage row still stops the night on a missing receipt. D6 was the milestone
# freeze, which no release-day script runs, so a row demanding it would hold every night red.
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
#       its ids equal REV's (release day's), so any change to them resets the count. A night is keyed on
#       the C its own record names (the release-rehearsal artifact), never on the run's head_sha. The count is
#       nightly_greens.sh's over this workflow's scheduled runs on main: a dispatched day run never counts.
#       Ready prints one RECEIPT line per counting night. EXIT 0 ready · 1 not ready · 2 · 3 as below.
#   rehearse.sh --classify TOOL CWD [ARG...]   READ or WRITE <why>, as the guard decides it
#   rehearse.sh --selftest       the case table: guard must-refuse/must-pass, the stub end to end,
#                                the judge on fixture nights (both polarities), and the rehearsal
#                                workflow's wiring against the train that reads its run (wiring())
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
# The workflow whose scheduled runs on main are the counting nights, and the floor its REST reads keep.
WORKFLOW=release-rehearsal-nightly.yml
RATE_FLOOR=1000
# That workflow's file, and the train that reads its run (--in-run): wiring() holds the one to the other.
REHEARSAL_WF="$SCRIPT_DIR/../../.github/workflows/$WORKFLOW"
TRAIN_SCRIPT="$SCRIPT_DIR/nightly_train.sh"
# A night's own record: the artifact its rehearse job uploads (night.txt, rehearsal/night.env, ...). The streak reads
# it for the night's C, newest night first, and reads none past the night that makes nightly_greens.sh's NEED.
RECORD_ARTIFACT=release-rehearsal
STREAK_NEED=3
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
# stage_summary and stage_cascade are functions below, run in the stage's own shell.
stages_table() {
    cat <<'STAGES'
lanes|-|env -u INBOX bash scripts/release/nightly_train.sh --out "$RELEASE_REHEARSAL_TRAIN" --commit "$RELEASE_REHEARSAL_C" ${RELEASE_REHEARSAL_IN_RUN:+--in-run "$RELEASE_REHEARSAL_IN_RUN"}
t2|-|bash scripts/release/t2_preflight.sh "$V"
bump|-|bash scripts/release/prepare_bump.sh "$V"
summary|bump|stage_summary
ship|summary|bash scripts/release/prepare_bump.sh "$V" --ship
ap:dogfood|ship|bash scripts/release/autopilot.sh "$V" rehearsal dogfood dogfood
ap:models|ship|bash scripts/release/autopilot.sh "$V" rehearsal models models
ap:readiness|ap:models|bash scripts/release/autopilot.sh "$V" rehearsal readiness readiness
ap:tag|ship|bash scripts/release/autopilot.sh "$V" rehearsal tag tag
ap:cleanroom|ap:tag|bash scripts/release/autopilot.sh "$V" rehearsal cleanroom cleanroom
ap:assets|ap:tag|bash scripts/release/autopilot.sh "$V" rehearsal assets assets
ap:preflight|ap:tag|bash scripts/release/autopilot.sh "$V" rehearsal preflight preflight
ap:publish|ap:tag|bash scripts/release/autopilot.sh "$V" rehearsal publish publish
ap:dryrun|ap:tag|bash scripts/release/autopilot.sh "$V" rehearsal dryrun dryrun
cascade|ap:preflight|stage_cascade
STAGES
}

# DLEDGER: row | stages its gate line sits in (comma list) | the line the scripts print when they stop
# on it (ERE) | the defect, as the spec states it. D7 also has a trace check (dledger_d7).
dledger_table() {
    cat <<'DLEDGER'
D1|ap:preflight,cascade|FAIL +R4 HEAD .* is an ancestor of neither origin/main nor |preflight R4 refuses: the release commit is on neither origin/main nor origin/release/<V>
D2|t2,ap:dogfood|\[FAIL\] declared:check_model_ladder|the dogfood ladder gate runs with no CRUX receipt dir, beside the models lane that writes them
D4|cascade|FAIL +R7 |the cascade's own preflight runs without the CRUX receipts
D5|ship|no T-2 GO receipt for origin/main|prepare_bump.sh --ship needs a T-2 GO that cannot exist
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

# ap_policy_fn AUTOPILOT -> the text of its ap_policy_applies function, empty when it has none
ap_policy_fn() { awk '/^ap_policy_applies\(\) \{/,/^\}/' "$1" 2>/dev/null; }

# stage_cascade: cascade-publish.sh --rehearse, run as release day's cascade step runs the drain
# (autopilot.sh step 6). It runs from the release worktree at the bump commit, where the preflight
# step ran; from the clone it judged C, not the commit the tag step names. Under the standing
# release policy it is handed the CRUX receipts the models step measured and the bump's
# certification. Without them R7 refuses (D4). Release day's own ap_policy_applies judges the
# policy, read from the release commit's autopilot.sh.
stage_cascade() {
    local w="$RELEASE_AP/wt" h fn pol
    h=$(git -C "$w" rev-parse -q --verify HEAD 2>/dev/null) || h=""
    if [ -z "$h" ] || [ "$h" != "${RELEASE_REHEARSAL_MC:-}" ]; then
        echo "STOP rehearsal: the release worktree $w is at ${h:-no commit}, not the bump commit ${RELEASE_REHEARSAL_MC:-(unset)}"
        return 1
    fi
    cd -- "$w" || return 1
    fn=$(ap_policy_fn scripts/release/autopilot.sh)
    [ -n "$fn" ] || { echo "STOP rehearsal: the release commit's autopilot.sh has no ap_policy_applies"; return 1; }
    pol=$( . <(printf '%s\n' "$fn") && ap_policy_applies "$V" ) \
        || { echo "STOP rehearsal: the standing release policy cannot be judged for $V"; return 1; }
    case $pol in
        1) MODEL_LADDER_CRUX_DIR="$RELEASE_AP/models-t1" CRUX_CERT="$w/evidence/crux/$V/prompt-certification.json" \
               bash scripts/cascade-publish.sh --rehearse ;;
        0) bash scripts/cascade-publish.sh --rehearse ;;
        *) echo "STOP rehearsal: ap_policy_applies printed '$pol' for $V, not 1 or 0"; return 1 ;;
    esac
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
# added to the release scripts' state: each gate's output file there (preflight.log, publish-dryrun.log,
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
      case $cmd in stage_summary|stage_cascade) "$cmd" ;; *) bash -c -- "$cmd" ;; esac ) > "$log" 2>&1 < /dev/null
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

# rate_ok RELPATH -> rc 0 while the core allowance is above the floor; rc 2 and the reason on stderr otherwise.
rate_ok() {
    local relpath=$1 lim rem fl
    read -r lim rem <<< "$(gh api rate_limit --jq '"\(.resources.core.limit) \(.resources.core.remaining)"' 2>/dev/null)"
    case "$lim:$rem" in *[!0-9:]*|:*|*:) echo "rate_limit unreadable" >&2; return 2 ;; esac
    # a fifth of the token's own hourly limit, capped at RATE_FLOOR, as nightly_train.sh's read keeps. The pass start
    # (--release-path) is the release path, which GH-1 lets call below 1000: no floor there (quorum 09:34Z, C345 Q9 Q3 B);
    # a refused or rate-limited read is still rc 2 for it.
    fl=$((lim / 5)); [ "$fl" -le "$RATE_FLOOR" ] || fl=$RATE_FLOOR
    if [ "$relpath" = 1 ]; then fl=0; fi
    [ "$rem" -ge "$fl" ] || { echo "core remaining $rem under $fl" >&2; return 2; }
}

# fetch_runs CACHE OUT -> OUT: the workflow's scheduled runs on main, nightly_greens.sh's six columns then head_sha.
# One REST read, sent with the cached ETag (a 304 reuses CACHE/runs.json). rc 2 and the reason on stderr when
# GitHub cannot be read or the floor is reached: not_measured, never "no nights".
fetch_runs() {
    local cache=$1 out=$2 relpath=${3:-0} st hdr=()
    rate_ok "$relpath" || return 2
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

# night_record CACHE RUN ATTEMPT -> CACHE/rec/RUN.ATTEMPT: that run's RECORD_ARTIFACT as uploaded (night.txt,
# rehearsal/night.env, ...). It is gh run download: two REST calls, the run's artifact list then the zip (measured
# with GH_DEBUG=api on gh 2.85.0; the blob the zip redirects to is not the API). A run attempt's artifact never
# changes, so a record is kept and never read again: 0 calls after the first. rc 2 and the reason on stderr when it
# cannot be read (none uploaded, expired after its 30 days, refused): not_measured, never a red night.
night_record() {
    local t o
    mkdir -p "$1/rec" && t=$(mktemp -d "$1/rec/.dl.XXXXXX") || return 2
    o=$(gh run download "$2" -R "$REPO" -n "$RECORD_ARTIFACT" -D "$t/a" 2>&1) \
        || { echo "$RECORD_ARTIFACT of run $2 cannot be read: ${o##*$'\n'}" >&2; return 2; }
    mv -- "$t/a" "$1/rec/$2.$3" || { echo "$RECORD_ARTIFACT of run $2 extracted nothing" >&2; return 2; }
    rmdir -- "$t" 2>/dev/null || :
}

# record_c REC -> the night's C, rc 0. rc 1 and the reason on stdout when the record names no C, or its two sources
# disagree: night.env's C= as uploaded, and the judge's "REHEARSAL <v> on <C>" in night.txt, which the judge printed
# from night.env after the last stage. So the second is not independent of the first: it catches a night.env changed
# after the judge, or a night.txt from another night. Both are in the one record, so the cross-check costs no call.
record_c() {
    local e t
    e=$(sed -n 's/^C=//p' "$1/rehearsal/night.env" 2>/dev/null)
    t=$(sed -n 's/^REHEARSAL [^ ]* on //p' "$1/night.txt" 2>/dev/null)
    [[ $e =~ ^[0-9a-f]{40}$ ]] || { printf 'its night.env names no C, or more than one'; return 1; }
    [ "$t" = "$e" ] || { printf 'night.env names C %s, night.txt %s' "$e" "${t:-no C}"; return 1; }
    printf '%s' "$e"
}

# ids_diff IDS TARGET -> which of tree, stop and policy differ
ids_diff() {
    local a b i out=""
    read -ra a <<< "$1"; read -ra b <<< "$2"
    for i in 0 1 2; do [ "${a[i]}" = "${b[i]}" ] || out="$out${out:+, }${b[i]%%=*}"; done
    printf '%s differ' "$out"
}

# streak --commit REV --as-of YYYY-MM-DD --cache DIR [--repo DIR] [--history FILE] -> the three counting nights'
# receipts a pass on REV needs (C345 #4, Q9). REV's ids are release day's. A scheduled night on main is keyed on its
# own C, the commit its record names, never on the run's head_sha (the pick sets C behind main's head). It counts
# only when C carries the same ids; any other, and a record that names no C or two, is a reset, judged as a red
# night. The count is nightly_greens.sh's. The history is FILE (seven columns, as fetch_runs writes) or one read of
# the workflow's runs; the records are CACHE/rec (night_record). Writes only in DIR.
# rc 0 ready (one RECEIPT line per counting night) · 1 not ready · 2 not_measured · 3 caller error
streak() {
    local rev="" repo="" hist="" asof="" cache="" relpath=0 target rc ids o hdr
    local id day co at hd prev="" last="" n=0 red=0 cut="" rated=0 c why
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
    # the walk, newest night first, up to --as-of. An attempt-1 green is keyed on its record's C; a red or retried run
    # is red whatever its C, and a running or void one is nightly_greens.sh's to judge from its conclusion alone. The
    # walk ends after the night that is red or makes STREAK_NEED counting nights: no older night can change the
    # verdict, and its record may be past the artifact's 30 days. The history judged ends there too, so total= counts
    # the walked nights only.
    : > "$cache/nights.tsv"; : > "$cache/resets.tsv"
    while IFS=$'\t' read -r id day co at hd; do
        if [ "$day" != "$prev" ]; then
            if [ "$red" = 1 ] || [ "$n" -ge "$STREAK_NEED" ]; then cut=$day; break; fi
            prev=$day
        fi
        case $co:$at in
            success:1) ;;
            success:*|failure:*|timed_out:*|startup_failure:*) red=1; continue ;;
            *) continue ;;
        esac
        if [ ! -d "$cache/rec/$id.$at" ]; then
            if [ "$rated" = 0 ]; then o=$(rate_ok "$relpath" 2>&1) || { echo "not_measured: night $day run $id: $o"; return 2; }; rated=1; fi
            o=$(night_record "$cache" "$id" "$at" 2>&1) || { echo "not_measured: night $day run $id: $o"; return 2; }
        fi
        if c=$(record_c "$cache/rec/$id.$at"); then
            rc=0; ids=$(release_ids "$repo" "$c" 2>&1) || rc=$?
            case $rc in
                0) if [ "$ids" = "$target" ]; then
                       printf '%s\t%s\t%s\n' "$id" "$c" "$ids" >> "$cache/nights.tsv"
                       [ "$day" = "$last" ] || n=$((n + 1)); last=$day; continue
                   fi
                   why=$(ids_diff "$ids" "$target") ;;
                1) why=$ids ;;
                *) echo "not_measured: night $day run $id: its C $c is not in $repo ($ids): fetch main first"; return 2 ;;
            esac
        else why=$c; c=-
        fi
        # a night whose C is not on the target's ids is a reset: its run is judged failure, whatever it ended as
        printf 'RESET run %s (night %s, C %s): %s\n' "$id" "$day" "$c" "$why"
        printf '%s\n' "$id" >> "$cache/resets.tsv"; red=1
    done < <(awk -F'\t' -v OFS='\t' -v asof="$asof" 'NR > 1 && $3 == "main" && $4 == "schedule" && substr($2, 1, 10) <= asof {
                 print $2, $1, substr($2, 1, 10), ($5 == "" ? "-" : $5), $6, $7 }' "$hist" | LC_ALL=C sort -r | cut -f 2-)
    awk -F'\t' -v OFS='\t' -v cut="$cut" '
        FILENAME == ARGV[1] { reset[$1] = 1; next }
        FNR == 1 { print $1, $2, $3, $4, $5, $6; next }
        $3 == "main" && $4 == "schedule" && cut != "" && substr($2, 1, 10) <= cut { next }
        $3 == "main" && $4 == "schedule" && ($1 in reset) { $5 = "failure" }
        { print $1, $2, $3, $4, $5, $6 }' "$cache/resets.tsv" "$hist" > "$cache/history.tsv" \
        || { echo "not_measured: the history could not be read"; return 2; }
    rc=0; o=$(bash "$SCRIPT_DIR/nightly_greens.sh" --check release-rehearsal --history "$cache/history.tsv" --as-of "$asof") || rc=$?
    printf '%s\n' "$o"
    [ "$rc" = 0 ] || return "$rc"
    # the three receipts: the run IDs nightly_greens.sh printed, each with its night, its C, the run's head and the ids
    o=$(printf '%s\n' "$o" | sed -n 's/^ok .* ready: .*: runs\(\( [0-9][0-9]*\)*\) (.*/\1/p')
    awk -F'\t' -v runs="$o" 'BEGIN { split(runs, r, " "); for (i in r) want[r[i]] = 1 }
        FILENAME == ARGV[1] { c[$1] = $2; ids[$1] = $3; next }
        FNR > 1 && ($1 in want) && ($1 in c) { printf "RECEIPT run %s night %s C %s head %s %s\n", $1, substr($2, 1, 10), c[$1], $7, ids[$1]; n++ }
        END { exit (n != 3) }' "$cache/nights.tsv" "$hist" \
        || { echo "not_measured: nightly_greens.sh said ready, but its run IDs are not three keyed nights in $hist"; return 2; }
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
    if [ "$reds" -eq 0 ]; then echo "VERDICT GREEN: every stage green on $c, its ids recorded, 0 writes, every D-row clear"; return 0; fi
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
        D7) dledger_d7 "$st" "$notrun"; return ;;
    esac
    if [ -n "$notrun" ]; then echo "NOT CLEARED: its stage did not finish green:$notrun"; else echo CLEAR; fi
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

# wiring WORKFLOW TRAIN -> 0 when the rehearsal workflow is wired the way nightly_train.sh --in-run reads its run. The
# train finds its lane jobs by NAME ("<INRUN_CALLER> / <job>", and INRUN_MODELS exactly) and a job's proof of C by
# STEP NAME (its two capture prefixes, then C), so a renamed job or step is a lane that reads not_measured every
# night, and a job of the rehearsal's own named "<INRUN_CALLER> / x" would vote as a lane. The models job's conclusion
# is the models lane: it checks out and asserts C, its relay step may not fail quietly, the relayed step runs only on
# a bundle the relay named and passes only on green or red, and a step fails the job on anything but green. The night
# needs both callers, or the train reads a run whose lanes still run, and hands the train C and its own run.
wiring() {
    local wf=$1 tr=$2 bad="" at='${{ needs.pick.outputs.c }}' from caller wfl models lk mk rk b found caps
    [ -f "$wf" ] || { printf '%s is missing' "$wf"; return 1; }
    [ -f "$tr" ] || { printf '%s is missing' "$tr"; return 1; }
    wr_tv() { sed -n "s/^$1=\"\\([^\"]*\\)\"\$/\\1/p" "$tr"; }
    from=$(wr_tv INRUN_FROM); caller=$(wr_tv INRUN_CALLER); wfl=$(wr_tv INRUN_WF); models=$(wr_tv INRUN_MODELS)
    { [ -n "$caller" ] && [ -n "$wfl" ] && [ -n "$models" ]; } || { printf 'miswired: the train names no INRUN_CALLER, INRUN_WF or INRUN_MODELS'; return 1; }
    [ "$from" = ".github/workflows/$WORKFLOW" ] || bad="$bad train(INRUN_FROM=$from)"
    # the train's capture prefixes, in its column order: the assert, then the relay
    mapfile -t caps < <(grep -o 'capture("^[^"(]*(?<s>\[0-9a-f\]{40})\$")' "$tr" | sed 's/^capture("^//; s/(?<s>.*$//')
    [ "${#caps[@]}" -eq 2 ] || { printf 'miswired: the train has %s step-name captures, not 2' "${#caps[@]}"; return 1; }
    # wr_named NAME -> the key of each job whose name: is NAME; wr_with ERE -> the key of each job with a line matching ERE
    wr_named() { awk -v n="$1" '/^jobs:$/ { j = 1; next } !j { next } /^  [A-Za-z0-9_-]+:$/ { k = substr($1, 1, length($1) - 1); next }
        /^    name: / { v = $0; sub(/^    name: /, "", v); sub(/[ \t]+#.*$/, "", v); if (v == n || (P != "" && index(v, P) == 1)) print k }' P="${2:-}" "$wf"; }
    wr_with() { awk -v re="$1" '/^jobs:$/ { j = 1; next } !j { next } /^  [A-Za-z0-9_-]+:$/ { k = substr($1, 1, length($1) - 1); next } $0 ~ re && !(k in s) { s[k]; print k }' "$wf"; }
    # wr_step BLOCK LINE -> the step of a job block (from its `      - ` line to the next) holding the exact line LINE
    wr_step() { awk -v l="$2" '/^      - / { if (hit) exit; s = "" } { s = s $0 "\n" } $0 == l { hit = 1 } END { if (hit) printf "%s", s }' <<< "$1"; }
    wr_one() { [ -n "$1" ] && [ "$1" = "${1%%$'\n'*}" ]; }
    lk=$(wr_named "$caller"); mk=$(wr_named "$models")
    found=$(wr_named "" "$caller / "); [ -z "$found" ] || bad="$bad $caller(prefix:$(printf '%s' "$found" | tr '\n' ','))"
    if wr_one "$lk"; then b=$(job_block "$wf" "$lk")
        grep -qxF "    uses: ./$wfl" <<< "$b" || bad="$bad $caller(uses)"
        grep -qxF "      ref: $at" <<< "$b" || bad="$bad $caller(ref-C)"
        grep -qxF "      caller: $caller" <<< "$b" || bad="$bad $caller(caller)"
    else bad="$bad $caller(name)"; fi
    if wr_one "$mk"; then b=$(job_block "$wf" "$mk")
        grep -qxF "      C: $at" <<< "$b" || bad="$bad $models(C)"
        grep -qxF "          ref: $at" <<< "$b" || bad="$bad $models(checkout-at-C)"
        grep -qxF '          [ "$h" = "$C" ] || { echo "::error::HEAD is $h, not C $C"; exit 1; }' <<< "$(wr_step "$b" "      - name: ${caps[0]}$at")" \
            || bad="$bad $models(assert)"
        grep -qE '^        run: setsid --wait bash scripts/release/models_nightly\.sh --relay --commit "\$C" .*--measure crux$' <<< "$(wr_step "$b" "        id: relay")" \
            || bad="$bad $models(relay)"
        ! grep -q 'continue-on-error' <<< "$b" || bad="$bad $models(continue-on-error)"
        found=$(wr_step "$b" "      - name: ${caps[1]}$at")
        if [ -z "$found" ]; then bad="$bad $models(relayed)"
        else
            grep -qxF "        if: steps.relay.outputs.bundle != ''" <<< "$found" || bad="$bad $models(relayed-if)"
            { grep -qxF '          STATE: ${{ steps.relay.outputs.state }}' <<< "$found" \
                && grep -qF '          case "$STATE" in green|red) ;; *) ' <<< "$found"; } || bad="$bad $models(relayed-state)"
        fi
        grep -qxF '          [ "$STATE" = green ] || { echo "::error::models RED on C $C: $REASON"; exit 1; }' <<< "$b" || bad="$bad $models(green-only)"
    else bad="$bad $models(name)"; fi
    rk=$(wr_with '^ +bash scripts/release/rehearse\.sh --run ')
    if wr_one "$rk"; then b=$(job_block "$wf" "$rk")
        found=$(sed -n 's/^    needs: \[\(.*\)\]$/\1/p' <<< "$b" | tr -d ' ' | tr ',' '\n')
        { grep -qx pick <<< "$found" && grep -qxF "$lk" <<< "$found" && grep -qxF "$mk" <<< "$found"; } || bad="$bad $rk(needs)"
        grep -qxF "      C: $at" <<< "$b" || bad="$bad $rk(C)"
        grep -qE '^ +bash scripts/release/rehearse\.sh --run .*--commit "\$C" --in-run "\$GITHUB_RUN_ID"( |$)' <<< "$b" || bad="$bad $rk(in-run)"
    else bad="$bad night(run)"; fi
    if [ -n "$bad" ]; then printf 'miswired:%s' "$bad"; return 1; fi
    printf '"%s / <job>" of %s and "%s", on C by "%s<C>" and "%s<C>"' "$caller" "$wfl" "$models" "${caps[0]}" "${caps[1]}"
}
# job_block WORKFLOW JOB -> the lines of job JOB, from `  JOB:` to the next job
job_block() {
    awk -v j="  $2:" '$0 == j { f = 1; next } f && /^  [^ #]/ { exit } f' "$1"
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
    selftest_cascade
    selftest_judge
    selftest_streak
    selftest_pick
    selftest_lanes_argv
    selftest_wiring
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
    local s="$tmp/stub" src="$tmp/src-home" envf o rc f
    mkdir -p "$s" "$src" || return 2
    # The source CARGO_HOME HAS credentials, so credentials_are_not_copied can fail on a host whose own
    # CARGO_HOME has none (a clean-room runner). The rest links to the real one, so reads still resolve.
    for f in registry git config.toml config bin; do
        [ -e "${CARGO_HOME:-$HOME/.cargo}/$f" ] && ln -sfn "${CARGO_HOME:-$HOME/.cargo}/$f" "$src/$f"
    done
    printf 'token = "planted"\n' > "$src/credentials.toml" && printf 'token = "planted"\n' > "$src/credentials" || return 2
    envf=$(CARGO_HOME="$src" install_guard "$s") || { printf '  BROKE install_guard failed\n'; fail=$((fail + 1)); return; }
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

# The cascade stage (D4): its command from the stage table, through run_stage, over a release worktree
# at the bump commit. The fixture's cascade-publish.sh prints the call it got and, under the policy,
# refuses R7 without the CRUX receipts as the preflight does. The judge reads that stage log: the
# receipts handed down clear D4, and the bare call opens it. The last row holds stage_cascade to what
# autopilot.sh's own cascade step hands the drain.
selftest_cascade() {
    local s="$tmp/cascade" n=0 cmd st w rc o d a r
    cmd=$(stages_table | awk -F'|' '$1 == "cascade" { print $3 }')
    cs() { # cs NAME WANT_RC WANT_OUT POLICY SETUP [CMD] -- @W@ and @AP@ in WANT_OUT are the worktree and RELEASE_AP
        local name=$1 wrc=$2 wout=$3 pol=$4 h
        n=$((n + 1)); st="$s/c$n"; w="$st/ap/wt"
        wout=${wout//@W@/$w}; wout=${wout//@AP@/$st/ap}
        mkdir -p "$st/clone" "$st/logs" "$w/scripts/release" || return 2
        git -c init.defaultBranch=main init -q "$w" \
            && git -C "$w" -c user.name=t -c user.email=t@t -c commit.gpgsign=false -c core.hooksPath=/dev/null commit -q --allow-empty -m bump \
            && h=$(git -C "$w" rev-parse HEAD) || return 2
        printf '%s\n' "$pol" > "$w/fx-policy"
        cat > "$w/scripts/release/autopilot.sh" <<'FX'
#!/usr/bin/env bash
ap_policy_applies() {
    [ "${1:-}" = 0.71.0 ] || { echo "asked about ${1:-no version}" >&2; return 2; }
    case $(cat fx-policy) in 1|0|yes) cat fx-policy ;; *) echo "the policy cannot be read" >&2; return 2 ;; esac
}
FX
        cat > "$w/scripts/cascade-publish.sh" <<'FX'
#!/usr/bin/env bash
printf 'CALL pwd=%s args=%s dir=%s cert=%s\n' "$PWD" "$*" "${MODEL_LADDER_CRUX_DIR:-}" "${CRUX_CERT:-}"
if [ "$(cat fx-policy)" = 1 ] && { [ -z "${MODEL_LADDER_CRUX_DIR:-}" ] || [ -z "${CRUX_CERT:-}" ]; }; then
    echo "FAIL  R7 STANDING RELEASE POLICY: no CRUX receipt dir or certification"; exit 1
fi
echo "ok    R7 CRUX smoke"
FX
        printf 'export RELEASE_AP=%q RELEASE_REHEARSAL_MC=%q\n' "$st/ap" "$h" > "$st/env"
        snip_run "$5"
        rc=0; run_stage "$st" "$st/env" cascade "${6:-$cmd}" 0.71.0 || rc=$?
        o=$(cat -- "$st/logs/cascade.log" 2>&1)
        if [ "$rc" = "$wrc" ] && [[ $o == *"$wout"* ]] && { [[ $name != *_stops ]] || [[ $o != *"CALL "* ]]; }; then pass=$((pass + 1))
        else printf '  BROKE %-44s rc=%s (want %s) log=%s\n' "$name" "$rc" "$wrc" "$(printf '%s' "$o" | tr '\n' '|' | cut -c1-200)"; fail=$((fail + 1)); fi
    }
    dj() { # dj NAME WANT_RC WANT_OUT -- the judge over a green fixture night carrying the last cs row's cascade log
        n=$((n + 1)); d="$s/night$n"
        fixture_night "$d" && cp -- "$st/logs/cascade.log" "$d/logs/cascade.log" || return 2
        rc=0; o=$(judge "$d" 2>&1) || rc=$?
        if [ "$rc" = "$2" ] && [[ $o == *"$3"* ]]; then pass=$((pass + 1))
        else printf '  BROKE %-44s rc=%s (want %s): %s\n' "$1" "$rc" "$2" "$(printf '%s' "$o" | grep -E 'RED|VERDICT' | head -3 | tr '\n' '|')"; fail=$((fail + 1)); fi
    }
    cs cascade_policy_hands_the_crux_receipts 0 \
        "CALL pwd=@W@ args=--rehearse dir=@AP@/models-t1 cert=@W@/evidence/crux/0.71.0/prompt-certification.json" 1 ':'
    dj d4_clear_with_the_receipts 0 "VERDICT GREEN"
    cs cascade_bare_call_refuses_r7 1 "FAIL  R7 STANDING RELEASE POLICY" 1 ':' 'cd -- "$RELEASE_AP/wt" && bash scripts/cascade-publish.sh --rehearse'
    dj d4_open_on_the_bare_call 1 "RED   D4 OPEN in cascade"
    cs cascade_off_policy_runs_bare 0 "CALL pwd=@W@ args=--rehearse dir= cert=" 0 ':'
    cs cascade_policy_not_judged_stops 1 "STOP rehearsal: the standing release policy cannot be judged for 0.71.0" x ':'
    cs cascade_policy_neither_1_nor_0_stops 1 "ap_policy_applies printed 'yes' for 0.71.0" yes ':'
    cs cascade_no_policy_judge_stops 1 "autopilot.sh has no ap_policy_applies" 1 ': > "$w/scripts/release/autopilot.sh"'
    cs cascade_worktree_on_another_commit_stops 1 "not the bump commit beef" 1 'printf "export RELEASE_REHEARSAL_MC=beef\n" >> "$st/env"'
    cs cascade_no_bump_commit_stops 1 "not the bump commit (unset)" 1 'printf "export RELEASE_REHEARSAL_MC=\n" >> "$st/env"'
    cs cascade_no_worktree_stops 1 "is at no commit" 1 'printf "export RELEASE_AP=%q\n" "$st/none" >> "$st/env"'
    # release day's hand-down: autopilot.sh's cascade step and stage_cascade name the same receipts
    a=$(awk '/^if run_step cascade; then$/,/^fi$/' "$SCRIPT_DIR/autopilot.sh" | grep -oE 'MODEL_LADDER_CRUX_DIR=[^ ]+ CRUX_CERT=[^ ]+')
    r=$(awk '/^stage_cascade\(\) \{/,/^\}/' "$SCRIPT_PATH" | grep -oE 'MODEL_LADDER_CRUX_DIR=[^ ]+ CRUX_CERT=[^ ]+')
    local ap_from='$AP/' ap_to='$RELEASE_AP/' wt_from='$WT/' wt_to='$w/'
    a=${a//"$ap_from"/"$ap_to"}; a=${a//"$wt_from"/"$wt_to"}
    n=$((n + 1))
    if [ -n "$a" ] && [ "$(printf '%s\n' "$a" | wc -l)" = 1 ] && [ "$a" = "$r" ] && [ -n "$(ap_policy_fn "$SCRIPT_DIR/autopilot.sh")" ]; then pass=$((pass + 1))
    else printf '  BROKE %-44s autopilot=%s rehearse=%s policy_fn=%s\n' cascade_hands_what_release_day_hands "${a:-none}" "${r:-none}" \
        "$(ap_policy_fn "$SCRIPT_DIR/autopilot.sh" | head -1)"; fail=$((fail + 1)); fi
    printf '  %s cascade rows\n' "$n"
}

# preflight_r4_fail_line -> the R4 FAIL line check_publish_preflight.sh prints, rendered from its own echo with a
# planted HEAD and release ref. Prints nothing when the preflight has no such echo, so D1's rows break.
preflight_r4_fail_line() {
    local t
    t=$(sed -n 's/^ *echo "\(FAIL  R4 HEAD .*\)"$/\1/p' "$SCRIPT_DIR/../check_publish_preflight.sh" | head -1)
    [ -n "$t" ] || return 0
    t=${t//'${head:0:9}'/abc123def}; t=${t//'$release_ref'/origin/release/0.71.0}
    printf '%s\n' "$t"
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
    printf 'bump\tgit\tREAD\t-\tbash scripts/release/prepare_bump.sh 0.71.0\tlog -1\n' > "$d/calls.tsv"
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
        'printf "ap:dogfood\tcurl\tMISSING\tno real curl\tbash x\t-fsSL u\n" >> "$d/calls.tsv"'
    j a_red_stage_is_red 1 "RED   stage ap:dogfood    exit 1: STOP dogfood failed" \
        'sed -i "s/^ap:dogfood\t0/ap:dogfood\t1/" "$d/stages.tsv"; printf "STOP dogfood failed\n" > "$d/logs/ap_dogfood.log"'
    j an_unreached_stage_is_red 1 "RED   stage ap:cleanroom  unreached" \
        'sed -i "s/^ap:cleanroom\t0\tb0b/ap:cleanroom\tunreached\t-/" "$d/stages.tsv"'
    j a_stage_never_run_is_red 1 "RED   stage cascade       not run" \
        'sed -i "/^cascade\t/d" "$d/stages.tsv"'
    j a_stage_on_another_commit_is_red 1 "measured on beef, not on c0ffee" \
        'sed -i "s/^ap:assets\t0\tb0b/ap:assets\t0\tbeef/" "$d/stages.tsv"'
    j a_bump_not_on_c_is_red 1 "RED   stage ap:dogfood    measured on b0b" \
        'printf "beef\n" > "$d/bump.parent"'
    # D1's planted line is read from the preflight itself, so a reworded R4 FAIL line breaks this row instead of
    # leaving the signature dead (#5059: the row sought "is not an ancestor of origin/release/" for two days
    # after #4934 reworded R4). An ok R4 line never opens D1.
    j d1_line_names_d1 1 "RED   D1 OPEN in ap:preflight: FAIL  R4 HEAD" \
        'preflight_r4_fail_line > "$d/logs/ap_preflight.log"'
    j d1_in_cascade_names_d1 1 "RED   D1 OPEN in cascade: FAIL  R4 HEAD" \
        'preflight_r4_fail_line > "$d/logs/cascade.log"'
    j d1_ok_r4_line_is_clear 0 "ok    D1 clear" \
        'printf "ok    R4 HEAD is an ancestor of origin/main\n" > "$d/logs/ap_preflight.log"'
    j d2_line_names_d2 1 "RED   D2 OPEN in t2:" \
        'printf "  [FAIL] declared:check_model_ladder  exit=1\n" > "$d/logs/t2.log"'
    j d4_line_names_d4 1 "RED   D4 OPEN in cascade" \
        'printf "FAIL  R7 model matrix: no receipt\n" > "$d/logs/cascade.log"'
    j d5_line_names_d5 1 "RED   D5 OPEN in ship" \
        'printf "STOP no T-2 GO receipt for origin/main c0ffee (x) -- run: t2\n" > "$d/logs/ship.log"'
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

# --streak on a fixture repository: one commit per id path changed, and run histories on them, each night with its
# record (the artifact's night.env and night.txt). The pass commit's ids are release day's; a night is keyed on its
# record's C, never its head, and a night on other ids, or with no C, or two, resets the count. Then night_env on the
# same commits, and the reads themselves through a planted gh: the runs read in one call, its ETag sent back and a 304
# reused, each record read once, the floor kept.
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
    # the planted gh: rate_limit prints GHS_RATE; the runs read answers 200 with an ETag, then 304 to that ETag; gh run
    # download extracts GHS/art/RUN where gh would, or fails as gh does when the run has no such artifact
    local gs="$tmp/ghs"
    mkdir -p "$gs/bin" || return 2
    cat > "$gs/bin/gh" <<'GH'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$GHS/calls"
case " $* " in *" rate_limit "*) printf '%s\n' "${GHS_RATE:-5000 4000}"; exit 0 ;; esac
if [ "$1 $2" = "run download" ]; then
    run=$3 dir="" name="" repo=""; shift 3
    while [ $# -ge 2 ]; do case $1 in -D) dir=$2 ;; -n) name=$2 ;; -R) repo=$2 ;; esac; shift 2; done
    [ "$repo" = paiml/aprender ] && [ "$name" = release-rehearsal ] && [ -n "$dir" ] && [ -d "$GHS/art/$run" ] \
        || { echo "no artifact matches any of the names or patterns provided" >&2; exit 1; }
    mkdir -p "$dir" && cp -R "$GHS/art/$run/." "$dir/"; exit
fi
case "${GHS_RATE:-}" in *" 0") printf 'HTTP/2.0 403 Forbidden\r\n\r\n{"message": "API rate limit exceeded"}'; exit 1 ;; esac
for a in "$@"; do [ "$a" = 'If-None-Match: "e1"' ] && { printf 'HTTP/2.0 304 Not Modified\r\n\r\n'; exit 0; }; done
printf 'HTTP/2.0 200 OK\r\nEtag: "e1"\r\nContent-Type: application/json\r\n\r\n'; cat "$GHS/body.json"
GH
    chmod +x "$gs/bin/gh" || return 2
    nrec() { # nrec DIR C_ENV C_TXT: a night's record as its artifact holds it; "-" leaves that file without a C
        mkdir -p "$1/rehearsal" || return 2
        { [ "$2" = - ] || printf 'C=%s\n' "$2"; printf 'V=0.71.0\nSOURCE=.\n'; } > "$1/rehearsal/night.env" || return 2
        { [ "$3" = - ] || printf 'REHEARSAL 0.71.0 on %s\n' "$3"; printf 'VERDICT GREEN: planted\n'; } > "$1/night.txt"
    }
    hist() { # hist FILE CACHE GHS ROW... with ROW = run|YYYY-MM-DD|event|conclusion|attempt|head[|record], the record
        # C (default: head) kept in CACHE, C_ENV/C_TXT (the two sources apart), noc (night.env names no C), art (gh
        # holds it, C = head) or none (no artifact at all)
        local f=$1 cd=$2 gd=$3 r id dd ev co at hd rs; shift 3
        printf 'run_id\tcreated_at\tbranch\tevent\tconclusion\tattempt\thead_sha\n' > "$f" || return 2
        for r in "$@"; do
            IFS='|' read -r id dd ev co at hd rs <<< "$r"
            printf '%s\t%sT05:20:00Z\tmain\t%s\t%s\t%s\t%s\n' "$id" "$dd" "$ev" "$co" "$at" "$hd" >> "$f"
            case ${rs:-$hd} in
                none) ;;
                noc) nrec "$cd/rec/$id.$at" - "$hd" ;;
                art) nrec "$gd/art/$id" "$hd" "$hd" ;;
                */*) nrec "$cd/rec/$id.$at" "${rs%/*}" "${rs#*/}" ;;
                *) nrec "$cd/rec/$id.$at" "${rs:-$hd}" "${rs:-$hd}" ;;
            esac || return 2
        done
    }
    sr() { # sr NAME WANT_RC WANT_OUT REV ROW... ; SR_RATE and SR_FLAG set the planted rate and a flag
        local name=$1 wrc=$2 wout=$3 rev=$4; shift 4
        k=$((k + 1))
        mkdir -p "$tmp/sg$k" && hist "$tmp/h$k.tsv" "$tmp/sc$k" "$tmp/sg$k" "$@" \
            || { printf '  BROKE %-48s fixture\n' "$name"; fail=$((fail + 1)); return; }
        rc=0; o=$(PATH="$gs/bin:$PATH" GHS="$tmp/sg$k" GHS_RATE="${SR_RATE:-5000 4000}" streak --commit "$rev" --repo "$g" \
            --history "$tmp/h$k.tsv" --as-of 2026-10-08 --cache "$tmp/sc$k" ${SR_FLAG:+"$SR_FLAG"} 2>&1) || rc=$?
        if [ "$rc" = "$wrc" ] && [[ $o == *"$wout"* ]]; then pass=$((pass + 1))
        else printf '  BROKE %-48s rc=%s (want %s): %s\n' "$name" "$rc" "$wrc" "$(printf '%s' "$o" | grep -v '^TARGET' | head -3 | tr '\n' '|')"; fail=$((fail + 1)); fi
    }
    local n6="1|2026-10-06|schedule|success|1" n7="2|2026-10-07|schedule|success|1" n8="3|2026-10-08|schedule|success|1"
    sr three_nights_on_the_same_ids_are_ready 0 "RECEIPT run 1 night 2026-10-06 C $c1 head $c1 tree=" "$c2" "$n6|$c1" "$n7|$c2" "$n8|$c2"
    sr a_release_scripts_change_after_them_resets 1 "RESET run 3 (night 2026-10-08, C $c2): tree differ" "$c3" "$n6|$c1" "$n7|$c2" "$n8|$c2"
    sr a_stop_list_change_resets 1 "RESET run 3 (night 2026-10-08, C $c2): stop differ" "$c4" "$n6|$c1" "$n7|$c2" "$n8|$c2"
    sr a_policy_file_change_resets 1 "RESET run 3 (night 2026-10-08, C $c2): policy differ" "$c5" "$n6|$c1" "$n7|$c2" "$n8|$c2"
    sr a_night_on_other_ids_breaks_the_streak 1 "not ready: night 2026-10-07 was red (run 2, failure)" "$c2" "$n6|$c2" "$n7|$c3" "$n8|$c2"
    sr a_night_without_the_stop_list_resets 1 "C $c0): $c0 has no blob $IDS_STOP" "$c2" "$n6|$c1" "$n7|$c2" "$n8|$c0"
    sr a_c_not_in_the_repo_is_not_measured 2 "not_measured: night 2026-10-08 run 3: its C $none is not in" "$c2" "$n6|$c1" "$n7|$c2" "$n8|$c2|$none"
    sr a_pass_commit_without_the_stop_list_is_not_measured 2 "not_measured: the pass commit's ids cannot be read" "$c0" "$n6|$c0" "$n7|$c0" "$n8|$c0"
    sr a_stop_list_that_is_a_directory_is_not_measured 2 "has no blob $IDS_STOP" "$c6" "$n6|$c2" "$n7|$c2" "$n8|$c2"
    sr day_runs_do_not_count 1 "not ready" "$c2" "$n6|$c1" "$n7|$c2" "4|2026-10-08|workflow_dispatch|success|1|$none"
    sr a_running_night_on_the_same_ids_waits 0 "RECEIPT run 2 night 2026-10-07" "$c2" "0|2026-10-05|schedule|success|1|$c1" "$n6|$c1" "$n7|$c2" "3|2026-10-08|schedule||1|$c2"
    # keyed on C: the head is the workflow's commit, C the one the night measured
    sr a_night_counts_on_its_c_whatever_its_head 0 "RECEIPT run 3 night 2026-10-08 C $c2 head $c3" "$c2" "$n6|$c3|$c1" "$n7|$c3|$c2" "$n8|$c3|$c2"
    sr a_running_night_waits_whatever_its_head 0 "RECEIPT run 2 night 2026-10-07" "$c2" "0|2026-10-05|schedule|success|1|$c1" "$n6|$c1" "$n7|$c2" "3|2026-10-08|schedule||1|$c3|none"
    sr a_record_without_c_resets 1 "RESET run 2 (night 2026-10-07, C -): its night.env names no C" "$c2" "$n6|$c2" "$n7|$c2|noc" "$n8|$c2"
    sr sources_that_disagree_reset 1 "RESET run 2 (night 2026-10-07, C -): night.env names C $c2, night.txt $c1" "$c2" "$n6|$c2" "$n7|$c2|$c2/$c1" "$n8|$c2"
    sr a_missing_record_is_not_measured 2 "not_measured: night 2026-10-07 run 2: release-rehearsal of run 2 cannot be read: no artifact" "$c2" "$n6|$c2" "$n7|$c2|none" "$n8|$c2"
    sr a_night_past_the_third_is_not_read 0 "(streak=3 total=3" "$c2" "0|2026-10-05|schedule|success|1|$c2|none" "$n6|$c2" "$n7|$c2" "$n8|$c2"
    sr a_red_night_needs_no_record 1 "not ready: night 2026-10-07 was red (run 2, failure)" "$c2" "$n6|$c2|none" "2|2026-10-07|schedule|failure|1|$c2|none" "$n8|$c2"
    sr a_green_at_attempt_two_needs_no_record 1 "not ready: night 2026-10-07 was red (run 2, success only at attempt 2" "$c2" "$n6|$c2|none" "2|2026-10-07|schedule|success|2|$c2|none" "$n8|$c2"
    sr a_reset_ends_the_walk 1 "not ready: night 2026-10-07 was red (run 2, failure)" "$c2" "$n6|$c2|none" "$n7|$c3" "$n8|$c2"
    SR_RATE="5000 999" sr a_record_read_under_the_floor_is_not_measured 2 "not_measured: night 2026-10-06 run 1: core remaining 999 under 1000" "$c2" "$n6|$c2|art" "$n7|$c2" "$n8|$c2"
    SR_RATE="5000 999" SR_FLAG="--release-path" sr the_pass_start_reads_a_record_under_the_floor 0 "RECEIPT run 1 night 2026-10-06 C $c2 head $c2" "$c2" "$n6|$c2|art" "$n7|$c2" "$n8|$c2"
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
    # the reads: every run's head is c3, on other ids; each record names the C its night measured
    jq -n --arg a "$c1" --arg b "$c2" --arg h "$c3" '{workflow_runs: [
        {id: 3, created_at: "2026-10-08T05:20:00Z", head_branch: "main", event: "schedule", conclusion: "success", run_attempt: 1, head_sha: $h},
        {id: 2, created_at: "2026-10-07T05:20:00Z", head_branch: "main", event: "schedule", conclusion: "success", run_attempt: 1, head_sha: $h},
        {id: 1, created_at: "2026-10-06T05:20:00Z", head_branch: "main", event: "schedule", conclusion: "success", run_attempt: 1, head_sha: $h},
        {id: 4, created_at: "2026-10-09T05:20:00Z", head_branch: "main", event: "schedule", conclusion: null, run_attempt: 1, head_sha: $h}]}' > "$gs/body.json" || return 2
    nrec "$gs/art/3" "$c2" "$c2" && nrec "$gs/art/2" "$c2" "$c2" && nrec "$gs/art/1" "$c1" "$c1" || return 2
    rd() { # rd NAME WANT_RC WANT_OUT WANT_RUNS_CALLS WANT_ETAG_CALLS WANT_RECORD_READS [RATE [FLAG]]
        rc=0; o=$(PATH="$gs/bin:$PATH" GHS="$gs" GHS_RATE="${7:-5000 4000}" streak --commit "$c2" --repo "$g" --as-of 2026-10-08 --cache "$gs/cache" ${8:+"$8"} 2>&1) || rc=$?
        local runs etag dl
        runs=$(grep -c 'actions/workflows/release-rehearsal-nightly.yml/runs?branch=main&event=schedule' "$gs/calls" 2>/dev/null)
        etag=$(grep -c 'If-None-Match: "e1"' "$gs/calls" 2>/dev/null)
        dl=$(grep -c '^run download ' "$gs/calls" 2>/dev/null)
        if [ "$rc" = "$2" ] && [[ $o == *"$3"* ]] && [ "${runs:-0}" = "$4" ] && [ "${etag:-0}" = "$5" ] && [ "${dl:-0}" = "$6" ]; then pass=$((pass + 1))
        else printf '  BROKE %-48s rc=%s (want %s) runs=%s etag=%s records=%s: %s\n' "$1" "$rc" "$2" "${runs:-0}" "${etag:-0}" "${dl:-0}" "$(printf '%s' "$o" | grep -v '^TARGET' | head -2 | tr '\n' '|')"; fail=$((fail + 1)); fi
    }
    rd the_first_read_is_one_call_and_a_record_a_night 0 "RECEIPT run 3 night 2026-10-08 C $c2 head $c3" 1 0 3
    rd the_second_read_sends_the_etag_and_keeps_the_records 0 "RECEIPT run 2 night 2026-10-07 C $c2 head $c3" 2 1 3
    rd a_read_under_the_floor_is_not_measured 2 "not_measured: core remaining 999 under 1000" 2 1 3 "5000 999"
    rd the_pass_start_reads_under_the_floor 0 "RECEIPT run 1 night 2026-10-06 C $c1 head $c3" 3 2 3 "5000 999" --release-path
    rd a_refused_read_at_the_pass_start_is_still_not_measured 2 "not_measured: workflow runs read: HTTP 403" 4 3 3 "5000 0" --release-path
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

# The rehearsal workflow against the train that reads its run: the real pair is wired, and each miswiring is refused
# for its own reason.
selftest_wiring() {
    local p0=$((pass + fail)) o
    wrow() { # wrow NAME wf|train NEEDLE SED -> wiring() refuses the pair with SED applied to that one file, naming NEEDLE
        local w=$REHEARSAL_WF t=$TRAIN_SCRIPT src out
        if [ "$2" = wf ]; then src=$w; out=$tmp/wr.yml; w=$out; else src=$t; out=$tmp/wr-train.sh; t=$out; fi
        sed -e "$4" "$src" > "$out" 2>/dev/null
        if cmp -s "$src" "$out"; then printf '  BROKE %-48s the fixture edit changed nothing\n' "$1"; fail=$((fail + 1))
        elif o=$(wiring "$w" "$t"); then printf '  BROKE %-48s accepted: %s\n' "$1" "$o"; fail=$((fail + 1))
        elif [[ $o != *"$3"* ]]; then printf '  BROKE %-48s refused, but not for %s: %s\n' "$1" "$3" "$o"; fail=$((fail + 1))
        else pass=$((pass + 1)); fi
    }
    if o=$(wiring "$REHEARSAL_WF" "$TRAIN_SCRIPT"); then pass=$((pass + 1))
    else printf '  BROKE %-48s %s\n' the_rehearsal_workflow_is_wired_as_its_train_reads_it "$o"; fail=$((fail + 1)); fi
    local mb='/^  models:$/,/^  [a-z]/'
    wrow a_renamed_models_job_is_refused wf 'models(name)' 's/^    name: models   #/    name: models-crux   #/'
    wrow a_renamed_lanes_caller_is_refused wf 'lanes(name)' 's/^    name: lanes   #/    name: lane   #/'
    wrow a_lanes_caller_input_of_another_name_is_refused wf 'lanes(caller)' 's/^      caller: lanes$/      caller: lane/'
    wrow lanes_called_at_another_ref_are_refused wf 'lanes(ref-C)' 's/^      ref: \${{ needs\.pick\.outputs\.c }}$/      ref: main/'
    wrow lanes_calling_another_workflow_are_refused wf 'lanes(uses)' 's#^    uses: \./\.github/workflows/release-lanes-nightly\.yml$#    uses: ./.github/workflows/x.yml#'
    wrow a_job_of_its_own_named_as_a_lane_is_refused wf 'lanes(prefix:rehearse' 's/^    name: rehearse$/    name: lanes \/ rehearse/'
    wrow a_renamed_relayed_step_is_refused wf 'models(relayed)' 's/^      - name: Relayed the models-crux bundle of /      - name: Relayed the bundle of /'
    wrow a_renamed_models_assert_is_refused wf 'models(assert)' "${mb}s/^      - name: Assert HEAD is /      - name: Assert head is /"
    wrow a_models_assert_that_asserts_nothing_is_refused wf 'models(assert)' "${mb}"'s/\[ "\$h" = "\$C" \] ||/true ||/'
    wrow a_relayed_step_run_on_no_bundle_is_refused wf 'models(relayed-if)' "/^      - name: Relayed/,/^      - /s/^        if: steps\\.relay\\.outputs\\.bundle != ''\$/        if: always()/"
    wrow a_relayed_step_passing_any_state_is_refused wf 'models(relayed-state)' 's/case "\$STATE" in green|red) ;;/case "$STATE" in *) ;;/'
    wrow a_models_job_that_succeeds_on_red_is_refused wf 'models(green-only)' 's/\[ "\$STATE" = green \] || {/[ -n "$STATE" ] || {/'
    wrow a_relay_that_may_fail_quietly_is_refused wf 'models(continue-on-error)' 's/^        id: relay$/&\n        continue-on-error: true/'
    wrow a_relay_of_the_ladder_is_refused wf 'models(relay)' 's/--measure crux$/--measure ladder/'
    wrow a_relay_step_under_another_id_is_refused wf 'models(relay)' 's/^        id: relay$/        id: crux/'
    wrow a_models_job_on_another_c_is_refused wf 'models(C)' "${mb}"'s/^      C: \${{ needs\.pick\.outputs\.c }}$/      C: ${{ github.sha }}/'
    wrow a_models_checkout_of_another_commit_is_refused wf 'models(checkout-at-C)' "${mb}"'s/^          ref: \${{ needs\.pick\.outputs\.c }}$/          ref: ${{ github.sha }}/'
    wrow a_night_that_does_not_wait_for_models_is_refused wf 'rehearse(needs)' 's/^    needs: \[pick, lanes, models\]$/    needs: [pick, lanes]/'
    wrow a_night_on_another_c_is_refused wf 'rehearse(C)' '/^  rehearse:$/,$s/^      C: \${{ needs\.pick\.outputs\.c }}$/      C: ${{ github.sha }}/'
    wrow a_night_without_its_own_run_is_refused wf 'rehearse(in-run)' 's/ --in-run "\$GITHUB_RUN_ID"//'
    wrow a_train_reading_another_workflow_is_refused train 'train(INRUN_FROM=' 's#^INRUN_FROM="\.github/workflows/release-rehearsal-nightly\.yml"$#INRUN_FROM=".github/workflows/x.yml"#'
    wrow a_train_reading_another_caller_is_refused train 'lane(name)' 's/^INRUN_CALLER="lanes"$/INRUN_CALLER="lane"/'
    wrow a_train_reading_another_models_job_is_refused train 'models-crux(name)' 's/^INRUN_MODELS="models"$/INRUN_MODELS="models-crux"/'
    wrow a_train_capturing_another_assert_is_refused train 'models(assert)' 's/capture("^Assert HEAD is (?<s>/capture("^Asserted (?<s>/'
    wrow a_train_capturing_another_relay_is_refused train 'models(relayed)' 's/capture("^Relayed the models-crux bundle of (?<s>/capture("^Relayed models-crux of (?<s>/'
    wrow a_second_job_named_models_is_refused wf 'models(name)' 's/^    name: rehearse$/    name: models/'
    wrow a_workflow_that_runs_no_night_is_refused wf 'night(run)' 's/bash scripts\/release\/rehearse\.sh --run /bash scripts\/release\/rehearse.sh --judge /'
    wrow a_train_with_one_capture_is_refused train 'step-name captures, not 2' 's/capture("^Relayed the models-crux bundle of (?<s>\[0-9a-f\]{40})\$")/test("^Relayed")/'
    wrow a_train_naming_no_models_job_is_refused train 'names no INRUN' 's/^INRUN_MODELS="models"$/INRUN_MODELS=models/'
    if o=$(wiring "$tmp/no-such.yml" "$TRAIN_SCRIPT" 2>&1); then printf '  BROKE %-48s accepted: %s\n' a_missing_workflow_is_refused "$o"; fail=$((fail + 1))
    elif [[ $o == *"no-such.yml is missing"* ]]; then pass=$((pass + 1)); else printf '  BROKE %-48s %s\n' a_missing_workflow_is_refused "$o"; fail=$((fail + 1)); fi
    if o=$(wiring "$REHEARSAL_WF" "$tmp/no-such.sh" 2>&1); then printf '  BROKE %-48s accepted: %s\n' a_missing_train_is_refused "$o"; fail=$((fail + 1))
    elif [[ $o == *"no-such.sh is missing"* ]]; then pass=$((pass + 1)); else printf '  BROKE %-48s %s\n' a_missing_train_is_refused "$o"; fail=$((fail + 1)); fi
    printf '  %s wiring rows\n' "$((pass + fail - p0))"
}

# ------------------------------------------------------------------ mutants --
# Each mutant is "name sed-script". It must change this file, still parse, and turn --selftest RED with at
# least one BROKE row. A pattern that no longer matches is reported, never skipped. Mutants of the guard
# library are applied to a copy of it that a copy of this script sources. Each copy is laid out as the repo is, with
# the train and the workflow that wiring() reads and the autopilot.sh the cascade rows read, and the unmutated copy
# must be green there first: a copy missing
# a file would turn every mutant RED and prove nothing.
mutants() {
    local tmp pass=0 fail=0 name file expr dir o rc
    tmp="$(mktemp -d)"
    case "$tmp" in ?*/tmp.?*) : ;; *) echo "mktemp gave '${tmp:-}', refusing"; return 2 ;; esac
    trap selftest_cleanup RETURN
    mdir() { # mdir -> DIR/scripts/release of a fresh copy of what the case table reads
        local d
        d=$(mktemp -d "${tmp:?}/m.XXXXXX") && mkdir -p "$d/scripts/release" "$d/.github/workflows" \
            && cp -- "$SCRIPT_PATH" "$SCRIPT_DIR/lib_write_guard.sh" "$SCRIPT_DIR/lib_rehearsal.sh" "$SCRIPT_DIR/nightly_greens.sh" \
                "$SCRIPT_DIR/release_lanes.sh" "$SCRIPT_DIR/autopilot.sh" "$TRAIN_SCRIPT" "$d/scripts/release/" \
            && cp -- "$SCRIPT_DIR/../check_publish_preflight.sh" "$d/scripts/" \
            && cp -- "$REHEARSAL_WF" "$d/.github/workflows/" && printf '%s/scripts/release' "$d"
    }
    dir=$(mdir) || return 2
    rc=0; o="$(REHEARSE_CASCADE="$SCRIPT_DIR/../cascade-publish.sh" bash "$dir/rehearse.sh" --selftest < /dev/null 2>&1)" || rc=$?
    if [ "$rc" != 0 ]; then
        printf '  BROKE %-40s the unmutated copy is not green in the mutant dir (exit %s): %s\n' baseline "$rc" "$(printf '%s\n' "$o" | grep -m 3 '^  BROKE ' | tr '\n' '|')"
        return 1
    fi
    while read -r name file expr; do
        [ -n "$name" ] || continue
        dir=$(mdir) || return 2
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
credentials_legacy_copied    rehearse.sh         s/for f in registry git config.toml config; do/for f in registry git config.toml config credentials; do/
token_kept                   rehearse.sh         s/printf 'unset CARGO_REGISTRY_TOKEN\\n'/printf 'export CARGO_REGISTRY_TOKEN=set\\n'/
cargo_home_bin_not_guarded   rehearse.sh         s/rm -f "\${bin:?}\/\${t:?}"/[ -e "${bin:?}\/${t:?}" ] \&\& continue/
writes_not_counted           rehearse.sh         s/\$3 == "WRITE" || \$3 == "MISSING"/$3 == "MISSING"/
missing_not_counted          rehearse.sh         s/\$3 == "WRITE" || \$3 == "MISSING"/$3 == "WRITE"/
unreached_is_green           rehearse.sh         s/unreached) printf 'RED   stage %-13s unreached/unreached) continue; printf 'RED   stage %-13s unreached/
other_commit_is_green        rehearse.sh         s/if \[ "\$commit" = "\$c" \] ||/if true ||/
bump_parent_not_checked      rehearse.sh         s/\&\& \[ "\$bparent" = "\$c" \]; }/; }/
dledger_sig_ignored          rehearse.sh         s/\[ -n "\$hit" \] \&\& { echo "OPEN in \$s: \$hit"; return; }/:/
d1_sig_old_text              rehearse.sh         s/R4 HEAD \.\* is an ancestor of neither origin\/main nor /R4 HEAD .* is not an ancestor of origin\/release\//
d1_preflight_reworded        ../check_publish_preflight.sh s/R4 HEAD \${head:0:9} is an ancestor of neither/R4 HEAD ${head:0:9} is on neither/
unreached_d_row_clears       rehearse.sh         s/\[ "\$rc" = 0 \] || notrun="\$notrun \$s(\${rc:-not run})"/:/
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
ids_reset_names_nothing      rehearse.sh         s/\[ "\${a\[i\]}" = "\${b\[i\]}" \] ||/true ||/
missing_c_is_a_reset         rehearse.sh         s/^                1) why=\$ids ;;/                1|2) why=$ids ;;/
day_runs_get_ids             rehearse.sh         s/NR > 1 \&\& \$3 == "main" \&\& \$4 == "schedule" \&\& substr(\$2, 1, 10) <= asof {/NR > 1 \&\& substr($2, 1, 10) <= asof {/
c_keyed_on_head              rehearse.sh         s/ids=\$(release_ids "\$repo" "\$c" 2>\&1)/ids=$(release_ids "$repo" "$hd" 2>\&1)/
record_without_c_counts      rehearse.sh         s/\[\[ \$e =~ \^\[0-9a-f\]{40}\$ \]\] ||/true ||/
sources_not_compared         rehearse.sh         s/\[ "\$t" = "\$e" \] ||/true ||/
missing_record_is_a_reset    rehearse.sh         s/o=\$(night_record "\$cache" "\$id" "\$at" 2>\&1) ||/o=$(night_record "$cache" "$id" "$at" 2>\&1) || true ||/
records_not_kept             rehearse.sh         s/if \[ ! -d "\$cache\/rec\/\$id.\$at" \]; then/if true; then/
record_repo_dropped          rehearse.sh         s/gh run download "\$2" -R "\$REPO" -n/gh run download "$2" -n/
artifact_name_changed        rehearse.sh         s/^RECORD_ARTIFACT=release-rehearsal$/RECORD_ARTIFACT=release-rehearsal-night/
record_layout_changed        rehearse.sh         s/"\$1\/rehearsal\/night.env" 2>\/dev\/null/"$1\/night.env" 2>\/dev\/null/
record_floor_skipped         rehearse.sh         s/if \[ "\$rated" = 0 \]; then o=\$(rate_ok/if false; then o=$(rate_ok/
record_floor_off_the_pass_start rehearse.sh         s/o=\$(rate_ok "\$relpath" 2>\&1)/o=$(rate_ok 0 2>\&1)/
walk_past_the_need           rehearse.sh         s/\[ "\$n" -ge "\$STREAK_NEED" \]/[ "$n" -gt "$STREAK_NEED" ]/
need_lowered                 rehearse.sh         s/^STREAK_NEED=3$/STREAK_NEED=2/
walk_past_a_red              rehearse.sh         s/if \[ "\$red" = 1 \] || \[ "\$n"/if false || [ "$n"/
reset_not_red                rehearse.sh         s/>> "\$cache\/resets.tsv"; red=1/>> "$cache\/resets.tsv"; red=0/
failure_does_not_end_walk    rehearse.sh         s/success:\*|failure:\*|timed_out/success:*|timed_out/
retried_does_not_end_walk    rehearse.sh         s/success:\*|failure:\*/failure:*/
pending_night_read           rehearse.sh         s/^            success:1) ;;/            success:1|-:1) ;;/
cut_not_applied              rehearse.sh         s/cut != "" \&\& substr(\$2, 1, 10) <= cut { next }/cut != "" \&\& 0 { next }/
receipt_names_head_as_c      rehearse.sh         s/substr(\$2, 1, 10), c\[\$1\], \$7, ids/substr($2, 1, 10), $7, $7, ids/
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
wiring_missing_wf_unnamed    rehearse.sh         s/\[ -f "\$wf" \] || { printf '%s is missing' "\$wf"; return 1; }/:/
wiring_missing_train_unnamed rehearse.sh         s/\[ -f "\$tr" \] || { printf '%s is missing' "\$tr"; return 1; }/:/
wiring_train_names_unchecked rehearse.sh         s/|| { printf 'miswired: the train names no INRUN_CALLER/|| true || { printf 'x/
wiring_train_from_unchecked  rehearse.sh         s/\[ "\$from" = ".github\/workflows\/\$WORKFLOW" \] || bad=/true || bad=/
wiring_capture_count_free    rehearse.sh         s/\[ "\${#caps\[@\]}" -eq 2 \] || {/true || {/
wiring_names_keep_comments   rehearse.sh         s/ sub(\/\[ \\t\]+#\.\*\$\/, "", v);//
wiring_step_runs_past_its_end rehearse.sh        s/{ if (hit) exit; s = "" }/{ s = "" }/
wiring_one_takes_two_jobs    rehearse.sh         s/wr_one() { .*/wr_one() { [ -n "$1" ]; }/
wiring_own_lane_names_free   rehearse.sh         s/\[ -z "\$found" \] || bad="\$bad \$caller(prefix/true || bad="$bad $caller(prefix/
wiring_caller_uses_free      rehearse.sh         s/|| bad="\$bad \$caller(uses)"/|| :/
wiring_caller_ref_free       rehearse.sh         s/|| bad="\$bad \$caller(ref-C)"/|| :/
wiring_caller_input_free     rehearse.sh         s/|| bad="\$bad \$caller(caller)"/|| :/
wiring_caller_name_unnamed   rehearse.sh         s/    else bad="\$bad \$caller(name)"; fi/    fi/
wiring_models_c_free         rehearse.sh         s/|| bad="\$bad \$models(C)"/|| :/
wiring_models_checkout_free  rehearse.sh         s/|| bad="\$bad \$models(checkout-at-C)"/|| :/
wiring_models_assert_free    rehearse.sh         s/|| bad="\$bad \$models(assert)"/|| :/
wiring_assert_read_as_relay  rehearse.sh         s/"      - name: \${caps\[0\]}\$at"/"      - name: ${caps[1]}$at"/
wiring_models_relay_free     rehearse.sh         s/|| bad="\$bad \$models(relay)"/|| :/
wiring_relay_may_fail        rehearse.sh         s/|| bad="\$bad \$models(continue-on-error)"/|| :/
wiring_relayed_unnamed       rehearse.sh         s/if \[ -z "\$found" \]; then bad="\$bad \$models(relayed)"/if false; then :/
wiring_relayed_if_free       rehearse.sh         s/|| bad="\$bad \$models(relayed-if)"/|| :/
wiring_relayed_state_free    rehearse.sh         s/|| bad="\$bad \$models(relayed-state)"/|| :/
wiring_green_only_free       rehearse.sh         s/|| bad="\$bad \$models(green-only)"/|| :/
wiring_models_name_unnamed   rehearse.sh         s/    else bad="\$bad \$models(name)"; fi/    fi/
wiring_needs_models_free     rehearse.sh         s/ && grep -qxF "\$mk" <<< "\$found"//
wiring_night_c_free          rehearse.sh         s/|| bad="\$bad \$rk(C)"/|| :/
wiring_night_in_run_free     rehearse.sh         s/|| bad="\$bad \$rk(in-run)"/|| :/
wiring_night_run_unnamed     rehearse.sh         s/    else bad="\$bad night(run)"; fi/    fi/
cascade_bare_under_policy    rehearse.sh         s/1) MODEL_LADDER_CRUX_DIR=[^ ]* CRUX_CERT=[^ ]* /1) /
cascade_cert_not_this_v      rehearse.sh         s/CRUX_CERT="\$w\/evidence\/crux\/\$V\//CRUX_CERT="$w\/evidence\/crux\//
cascade_crux_dir_elsewhere   rehearse.sh         s/MODEL_LADDER_CRUX_DIR="\$RELEASE_AP\/models-t1"/MODEL_LADDER_CRUX_DIR="$RELEASE_AP\/models"/
cascade_runs_in_the_clone    rehearse.sh         s/^    cd -- "\$w" || return 1$/    :/
cascade_worktree_unchecked   rehearse.sh         s/if \[ -z "\$h" \] || \[ "\$h" != "\${RELEASE_REHEARSAL_MC:-}" \]; then/if false; then/
cascade_odd_policy_bare      rehearse.sh         s/^        0) bash scripts\/cascade-publish.sh --rehearse ;;$/        *) bash scripts\/cascade-publish.sh --rehearse ;;/
cascade_policy_not_about_v   rehearse.sh         s/ap_policy_applies "\$V" ) \\$/ap_policy_applies ) \\/
cascade_policy_rc_ignored    rehearse.sh         s/ap_policy_applies "\$V" ) \\$/ap_policy_applies "$V"; : ) \\/
cascade_stage_from_clone     rehearse.sh         s/^cascade|ap:preflight|stage_cascade$/cascade|ap:preflight|bash scripts\/cascade-publish.sh --rehearse/
cascade_stage_as_text        rehearse.sh         s/stage_summary|stage_cascade) "\$cmd" ;;/stage_summary) "$cmd" ;;/
ap_policy_fn_renamed         autopilot.sh        s/^ap_policy_applies() {$/ap_policy_judge() {/
ap_cascade_other_receipts    autopilot.sh        /^if run_step cascade; then$/,/^fi$/s/models-t1/models-x/
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
