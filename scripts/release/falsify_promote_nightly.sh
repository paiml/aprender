#!/usr/bin/env bash
# falsify_promote_nightly.sh -- the case table and mutants for promote_nightly.sh (N26 / FLOW-008, PMAT-5306)
#
#   bash scripts/release/falsify_promote_nightly.sh                 # every refusal row + the green rows
#   bash scripts/release/falsify_promote_nightly.sh --mutants       # the table plus one mutant per check
#   PROMOTE_SCRIPT=/path/to/mutant.sh bash ... falsify_promote_nightly.sh   # run the table on another copy
#
# Every row builds a throwaway origin + clone under mktemp and runs the script against it. Nothing
# here touches the real repository, the network, or a real tag: the "tag" a --execute row makes
# lives in a temp clone and is pushed to a temp bare origin.
#
# Exit 0 = every row held (and, with --mutants, every mutant was caught). 1 = a row or mutant failed.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || exit 2
ROOT=$(cd "$HERE/../.." && pwd) || exit 2
SUBJECT="${PROMOTE_SCRIPT:-$HERE/promote_nightly.sh}"
TMP=$(mktemp -d) || exit 2
trap 'rm -rf -- "${TMP:?}"' EXIT

export GIT_AUTHOR_NAME=falsify GIT_AUTHOR_EMAIL=f@x GIT_COMMITTER_NAME=falsify GIT_COMMITTER_EMAIL=f@x
pass=0
fail=0
W="" S="" R=""
git_q() { git -c user.email=f@x -c user.name=falsify -c commit.gpgsign=false -c core.hooksPath=/dev/null "$@"; }

# template: origin (bare) + clone with a green release commit on main. Built once, copied per row.
build_template() {
    local t="$TMP/tpl" w
    mkdir -p "$t" && git init -q --bare "$t/origin.git" || return 1
    git clone -q "$t/origin.git" "$t/work" 2>/dev/null || return 1
    w="$t/work"
    git -C "$w" config core.hooksPath /dev/null   # fixture clone only: the real tag guard is stubbed above
    git -C "$w" checkout -q -b main
    mkdir -p "$w/scripts/release" "$w/scripts/hooks" "$w/evidence/crux/0.72.2" "$w/contracts"
    cat > "$w/Cargo.toml" <<'TOML'
[package]
name = "facade"
version = "0.72.2"

[workspace.dependencies]
foo = { version = "1.0.0" }

[workspace.package]
version = "0.72.2"
edition = "2021"
TOML
    # the gate scripts the promote script calls are the commit's own; these stand in for them
    cat > "$w/scripts/release/tag_coverage_gate.sh" <<'SH'
#!/usr/bin/env bash
[ "${1:-}" = "--resolve" ] || exit 2
[ -n "${2:-}" ] || exit 2
exit "$(cat "$(dirname "$0")/../../fx-coverage-rc")"
SH
    printf '0\n' > "$w/fx-coverage-rc"
    cat > "$w/scripts/hooks/pre-push-tags.sh" <<'SH'
#!/usr/bin/env bash
printf 'ARM %s\n' "$*" >> "${FX_CALLS:?}"
exit "$(cat "$(dirname "$0")/../../fx-arm-rc")"
SH
    printf '0\n' > "$w/fx-arm-rc"
    printf 'fx-calls\n' > "$w/.gitignore"
    # the REAL policy library and ladder: the policy is judged by the same code autopilot uses
    cp -r "$ROOT/scripts/lib" "$w/scripts/lib" || return 1
    cp "$ROOT/contracts/model-capability-ladder-v1.yaml" "$w/contracts/" || return 1
    printf '{"apr_commit":"x"}\n' > "$w/evidence/crux/0.72.2/prompt-certification.json"
    git_q -C "$w" add -A && git_q -C "$w" commit -q -m "release commit" || return 1
    git_q -C "$w" push -q origin main 2>/dev/null || return 1
    git -C "$t/origin.git" symbolic-ref HEAD refs/heads/main
    git -C "$w" rev-parse HEAD > "$t/sha"
}

rcpt() {
    case "$1" in
        clean-room) printf '{"lane":"clean-room","sha":"%s","status":"%s","run_id":"4242424242","job":"clean-room (aprender)","conclusion":"success","tested_shas":["%s"]}\n' "$3" "$2" "$3" > "$R/$1.json" ;;
        models) printf '{"lane":"models","sha":"%s","status":"%s","hosts":["lambda","gx10"],"go":"MODELS GO (CRUX smoke) on lambda and gx10 at %s:"}\n' "$3" "$2" "${3:0:9}" > "$R/$1.json" ;;
        *) printf '{"lane":"%s","sha":"%s","status":"%s"}\n' "$1" "$3" "$2" > "$R/$1.json" ;;
    esac
}
all_green() { local lane; for lane in clean-room models publish-dry-run; do rcpt "$lane" green "$S"; done; }
# bump: commit the working change in $W on main, push, and rebind S and every receipt to the new head
bump() {
    git_q -C "$W" add -A && git_q -C "$W" commit -q -m "$1" && git_q -C "$W" push -q origin main 2>/dev/null
    S=$(git -C "$W" rev-parse HEAD)
    all_green
}

# fx NAME -> fresh copy of the template at $TMP/NAME; sets W (clone), S (sha), R (receipts dir)
fx() {
    local d="$TMP/$1"
    cp -a "$TMP/tpl" "$d" || return 1
    git -C "$d/work" remote set-url origin "$d/origin.git"
    W="$d/work" S=$(cat "$d/sha") R="$d/receipts"
    export FX_CALLS="$d/calls"
    mkdir -p "$R"
    all_green
}

# row NAME WANT_RC WANT_TEXT [args...] -- run the subject; judge rc and that WANT_TEXT is in the output ('' = any)
row() {
    local name=$1 want=$2 text=$3 out rc
    shift 3
    out=$(cd "$W" && bash "$SUBJECT" "$@" 2>&1); rc=$?
    if [ "$rc" = "$want" ] && { [ -z "$text" ] || [[ $out == *"$text"* ]]; }; then
        pass=$((pass + 1))
    else
        printf '  BROKE %-60s rc=%s (want %s) want-text=[%s] out=[%s]\n' "$name" "$rc" "$want" "$text" "$(printf '%s' "$out" | tr '\n' '|' | cut -c1-240)"
        fail=$((fail + 1))
    fi
}
# holds NAME CMD... -- a post-condition judged after a row
holds() {
    local name=$1
    shift
    if "$@"; then pass=$((pass + 1)); else printf '  BROKE %-60s post-condition\n' "$name"; fail=$((fail + 1)); fi
}
origin_of() { printf '%s/origin.git\n' "$(dirname "$W")"; }
no_local_tag() { [ -z "$(git -C "$W" tag -l 'v*')" ]; }
no_origin_tag() { [ -z "$(git -C "$(origin_of)" tag -l 'v*')" ]; }
no_calls() { [ ! -s "$FX_CALLS" ]; }
tag_at_sha() { [ "$(git -C "$1" rev-parse 'v0.72.2^{commit}' 2>/dev/null)" = "$S" ]; }
tag_subject_ok() { [ "$(git -C "$W" tag -l --format='%(contents:subject)' v0.72.2)" = "aprender v0.72.2" ]; }
armed_once() { [ "$(cat "$FX_CALLS" 2>/dev/null)" = "ARM --arm-release v0.72.2 120" ]; }

table() {
    local lane a
    # ---- green, dry-run ----
    fx green
    row "dry-run, all green: promote decision + exact tag command" 0 "git tag -a v0.72.2 -m \"aprender v0.72.2\" $S" --sha "$S" --receipts "$R"
    holds "dry-run made no local tag" no_local_tag
    holds "dry-run pushed no tag" no_origin_tag
    holds "dry-run armed nothing" no_calls
    row "dry-run names the arm command" 0 "pre-push-tags.sh --arm-release v0.72.2 120" --sha "$S" --receipts "$R"
    row "--tag matching the version is accepted" 0 "PROMOTE" --sha "$S" --receipts "$R" --tag v0.72.2

    # ---- usage ----
    fx u1
    row "no --sha is a usage error" 2 "" --receipts "$R"
    row "no --receipts is a usage error" 2 "" --sha "$S"
    row "short sha refused" 2 "40" --sha "${S:0:12}" --receipts "$R"
    row "uppercase sha refused" 2 "" --sha "$(printf '%s' "$S" | tr a-f A-F)" --receipts "$R"
    row "unknown flag refused" 2 "" --sha "$S" --receipts "$R" --force

    # ---- (a) the sha ----
    fx offmain
    git -C "$W" checkout -q -b side
    printf 'x\n' > "$W/side.txt"
    git_q -C "$W" add side.txt && git_q -C "$W" commit -q -m side
    S=$(git -C "$W" rev-parse HEAD)
    all_green
    row "sha not on origin/main refused" 1 "not on origin/main" --sha "$S" --receipts "$R"
    row "off-main sha with --execute refused" 1 "not on origin/main" --sha "$S" --receipts "$R" --execute
    holds "off-main --execute made no tag" no_local_tag
    holds "off-main --execute armed nothing" no_calls

    fx headmismatch
    git_q -C "$W" commit -q --allow-empty -m next && git_q -C "$W" push -q origin main 2>/dev/null
    row "checkout HEAD ahead of the sha: judged on the sha, promotes" 0 "PROMOTE" --sha "$S" --receipts "$R"

    fx dirty
    printf '# edit\n' >> "$W/Cargo.toml"
    row "dirty tracked Cargo.toml in the checkout is not what is judged" 0 "PROMOTE" --sha "$S" --receipts "$R"

    # a hidden edit to a gate script must not decide the run: the gates run from a fresh worktree of the sha
    for flag in --assume-unchanged --skip-worktree; do
        fx "hide$flag"
        printf '1\n' > "$W/fx-coverage-rc"; git_q -C "$W" add fx-coverage-rc; git_q -C "$W" commit -q -m "cov red" && git_q -C "$W" push -q origin main 2>/dev/null
        S=$(git -C "$W" rev-parse HEAD); all_green
        printf '0\n' > "$W/fx-coverage-rc"
        git -C "$W" update-index "$flag" fx-coverage-rc
        row "hidden ($flag) edit of a gate input cannot green the run" 1 "coverage" --sha "$S" --receipts "$R"
        row "hidden ($flag) edit with --execute makes no tag" 1 "coverage" --sha "$S" --receipts "$R" --execute
        holds "hidden ($flag) --execute made no tag" no_local_tag
        holds "hidden ($flag) --execute pushed no tag" no_origin_tag
    done
    fx hidearm
    printf '1\n' > "$W/fx-arm-rc"; git_q -C "$W" add fx-arm-rc; git_q -C "$W" commit -q -m "arm red" && git_q -C "$W" push -q origin main 2>/dev/null
    S=$(git -C "$W" rev-parse HEAD); all_green
    printf '0\n' > "$W/fx-arm-rc"; git -C "$W" update-index --assume-unchanged fx-arm-rc
    row "hidden edit of the arm hook cannot arm" 1 "arm" --sha "$S" --receipts "$R" --execute
    holds "hidden arm edit pushed no tag" no_origin_tag

    fx excluded
    mkdir -p "$W/.git/info"; printf "stray\n" > "$W/stray.txt"; printf "stray.txt\n" >> "$W/.git/info/exclude"; printf 'u\n' > "$W/untracked.txt"
    row "untracked and excluded files do not decide the run" 0 "PROMOTE" --sha "$S" --receipts "$R"

    fx notacommit
    row "unknown sha refused" 1 "" --sha 0123456789abcdef0123456789abcdef01234567 --receipts "$R"

    # ---- (a) version and tag ----
    fx vmis
    row "--tag that differs from the workspace version refused" 1 "does not match" --sha "$S" --receipts "$R" --tag v0.72.3
    row "--tag without the v prefix refused" 1 "does not match" --sha "$S" --receipts "$R" --tag 0.72.2

    fx nover
    printf '[package]\nversion = "9.9.9"\n\n[workspace.package]\nedition = "2021"\n' > "$W/Cargo.toml"
    bump "no workspace version"
    row "no [workspace.package] version refused ([package] 9.9.9 is not read)" 1 "workspace.package" --sha "$S" --receipts "$R"

    fx prerelease
    sed -i 's/^version = "0.72.2"/version = "0.72.2-rc.1"/' "$W/Cargo.toml"
    bump "rc version"
    row "a pre-release version is not a final promote" 1 "X.Y.Z" --sha "$S" --receipts "$R"

    fx tagloc
    git_q -C "$W" tag -a v0.72.2 -m old "$S"
    row "tag exists locally refused" 1 "already exists locally" --sha "$S" --receipts "$R"
    row "tag exists locally, --execute refused" 1 "already exists locally" --sha "$S" --receipts "$R" --execute
    holds "no ARM when the tag exists" no_calls
    holds "no push when the tag exists" no_origin_tag

    fx tagorig
    git_q -C "$W" tag -a v0.72.2 -m old "$S"
    git_q -C "$W" push -q origin v0.72.2 2>/dev/null
    git_q -C "$W" tag -d v0.72.2 > /dev/null
    row "tag exists only on origin refused" 1 "already exists on origin" --sha "$S" --receipts "$R"

    fx noorigin
    git -C "$W" remote set-url origin "$TMP/does-not-exist.git"
    row "origin unreachable refused" 1 "origin" --sha "$S" --receipts "$R"

    # ---- (b) receipts ----
    for lane in clean-room models publish-dry-run; do
        fx "miss-$lane"
        rm -f -- "$R/$lane.json"
        row "missing $lane receipt refused" 1 "$lane" --sha "$S" --receipts "$R"
        fx "red-$lane"
        rcpt "$lane" red "$S"
        row "red $lane receipt refused" 1 "$lane" --sha "$S" --receipts "$R"
        fx "nm-$lane"
        rcpt "$lane" not_measured "$S"
        row "not_measured $lane receipt refused" 1 "$lane" --sha "$S" --receipts "$R"
        fx "other-$lane"
        rcpt "$lane" green "$(printf 'a%.0s' $(seq 1 40))"
        row "$lane receipt bound to another sha refused" 1 "$lane" --sha "$S" --receipts "$R"
        fx "junk-$lane"
        printf 'not json{' > "$R/$lane.json"
        row "unreadable $lane receipt refused" 1 "$lane" --sha "$S" --receipts "$R"
        fx "wronglane-$lane"
        printf '{"lane":"other","sha":"%s","status":"green"}\n' "$S" > "$R/$lane.json"
        row "$lane receipt naming another lane refused" 1 "$lane" --sha "$S" --receipts "$R"
        fx "execred-$lane"
        rcpt "$lane" red "$S"
        row "red $lane receipt with --execute refused" 1 "$lane" --sha "$S" --receipts "$R" --execute
        holds "red $lane --execute made no tag" no_local_tag
    done
    fx nodir
    row "receipts directory absent refused" 1 "receipts" --sha "$S" --receipts "$TMP/nope"

    fx cov
    printf '1\n' > "$W/fx-coverage-rc"
    bump "coverage gate red"
    row "coverage gate (tag_coverage_gate.sh --resolve) refusing refuses" 1 "coverage" --sha "$S" --receipts "$R"

    fx nocert
    git_q -C "$W" rm -q evidence/crux/0.72.2/prompt-certification.json
    bump "no certification"
    row "no prompt-certification.json at the sha refused" 1 "prompt-certification" --sha "$S" --receipts "$R"

    fx badcert
    printf 'not json{' > "$W/evidence/crux/0.72.2/prompt-certification.json"
    bump "bad certification"
    row "unreadable prompt-certification.json refused" 1 "prompt-certification" --sha "$S" --receipts "$R"

    fx noladder
    git_q -C "$W" rm -q contracts/model-capability-ladder-v1.yaml
    bump "no ladder"
    row "release policy that cannot be judged refused" 1 "policy" --sha "$S" --receipts "$R"

    fx oldver
    sed -i 's/^version = "0.72.2"/version = "0.60.0"/' "$W/Cargo.toml"
    mkdir -p "$W/evidence/crux/0.60.0" && cp "$W/evidence/crux/0.72.2/prompt-certification.json" "$W/evidence/crux/0.60.0/"
    bump "old version"
    row "a version the standing policy does not cover refused" 1 "policy" --sha "$S" --receipts "$R"


    # ---- quorum findings: one document, the models go-string, the clean-room run, the facade, untracked files ----
    for lane in clean-room models publish-dry-run; do
        fx "twodoc-rg-$lane"
        { printf '{"lane":"%s","sha":"%s","status":"red"}\n' "$lane" "$S"; cat "$R/$lane.json"; } > "$R/two"; mv "$R/two" "$R/$lane.json"
        row "two-document $lane receipt (red then green) refused" 1 "$lane" --sha "$S" --receipts "$R"
        fx "twodoc-gr-$lane"
        { cat "$R/$lane.json"; printf '{"lane":"%s","sha":"%s","status":"red"}\n' "$lane" "$S"; } > "$R/two"; mv "$R/two" "$R/$lane.json"
        row "two-document $lane receipt (green then red) refused" 1 "$lane" --sha "$S" --receipts "$R"
    done
    fx mgo-sha
    jq --arg g "MODELS GO (CRUX smoke) on lambda and gx10 at 123456789:" '.go = $g' "$R/models.json" > "$R/m" && mv "$R/m" "$R/models.json"
    row "models go-string at another sha9 refused" 1 "models" --sha "$S" --receipts "$R"
    fx mgo-none
    jq 'del(.go)' "$R/models.json" > "$R/m" && mv "$R/m" "$R/models.json"
    row "models receipt without the go-string refused (a bare green is not the gate)" 1 "models" --sha "$S" --receipts "$R"
    fx mgo-host
    jq --arg g "MODELS GO (CRUX smoke) on lambda at ${S:0:9}:" '.go = $g | .hosts = ["lambda"]' "$R/models.json" > "$R/m" && mv "$R/m" "$R/models.json"
    row "models receipt for one host only refused" 1 "models" --sha "$S" --receipts "$R"
    fx mgo-host2
    jq '.hosts = ["lambda"]' "$R/models.json" > "$R/m" && mv "$R/m" "$R/models.json"
    row "models hosts list missing gx10 refused" 1 "models" --sha "$S" --receipts "$R"
    fx cr-norun
    jq 'del(.run_id)' "$R/clean-room.json" > "$R/m" && mv "$R/m" "$R/clean-room.json"
    row "clean-room receipt without a run id refused" 1 "clean-room" --sha "$S" --receipts "$R"
    fx cr-badrun
    jq '.run_id = "abc"' "$R/clean-room.json" > "$R/m" && mv "$R/m" "$R/clean-room.json"
    row "clean-room run id that is not a number refused" 1 "clean-room" --sha "$S" --receipts "$R"
    fx cr-concl
    jq '.conclusion = "failure"' "$R/clean-room.json" > "$R/m" && mv "$R/m" "$R/clean-room.json"
    row "clean-room job conclusion not success refused" 1 "clean-room" --sha "$S" --receipts "$R"
    fx cr-job
    jq '.job = "clean-room (other)"' "$R/clean-room.json" > "$R/m" && mv "$R/m" "$R/clean-room.json"
    row "clean-room job that is not clean-room (aprender) refused" 1 "clean-room" --sha "$S" --receipts "$R"
    fx cr-tested
    jq '.tested_shas = ["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"]' "$R/clean-room.json" > "$R/m" && mv "$R/m" "$R/clean-room.json"
    row "clean-room that tested another sha refused" 1 "clean-room" --sha "$S" --receipts "$R"
    fx cr-tested2
    jq --arg s "$S" '.tested_shas = [$s, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"]' "$R/clean-room.json" > "$R/m" && mv "$R/m" "$R/clean-room.json"
    row "clean-room that tested more than exactly the sha refused" 1 "clean-room" --sha "$S" --receipts "$R"

    fx facade
    sed -i '0,/^version = "0.72.2"/s//version = "9.9.9"/' "$W/Cargo.toml"
    bump "facade version differs"
    row "root [package] version differing from [workspace.package] refused" 1 "facade" --sha "$S" --receipts "$R"
    fx facade-ws
    printf '[package]\nname = "facade"\nversion.workspace = true\n\n[workspace.package]\nversion = "0.72.2"\n' > "$W/Cargo.toml"
    bump "facade inherits"
    row "[package] version.workspace = true is accepted" 0 "PROMOTE" --sha "$S" --receipts "$R"
    fx facade-none
    printf '[package]\nname = "facade"\n\n[workspace.package]\nversion = "0.72.2"\n' > "$W/Cargo.toml"
    bump "facade no version"
    row "[package] without any version refused" 1 "facade" --sha "$S" --receipts "$R"

    # ---- duplicate keys and run ids ----
    for lane in clean-room models publish-dry-run; do
        fx "dupkey-$lane"
        jq -c . "$R/$lane.json" | sed 's/"status":"green"/"status":"red","status":"green"/' > "$R/m" && mv "$R/m" "$R/$lane.json"
        row "$lane receipt with a duplicate status key (red, green) refused" 1 "$lane" --sha "$S" --receipts "$R"
        fx "dupkey2-$lane"
        jq -c . "$R/$lane.json" | sed 's/"status":"green"/"status":"green","status":"red"/' > "$R/m" && mv "$R/m" "$R/$lane.json"
        row "$lane receipt with a duplicate status key (green, red) refused" 1 "$lane" --sha "$S" --receipts "$R"
    done
    fx dupsha
    jq -c . "$R/models.json" | sed "s/\"sha\":\"$S\"/\"sha\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\",\"sha\":\"$S\"/" > "$R/m" && mv "$R/m" "$R/models.json"
    row "models receipt with a duplicate sha key refused" 1 "models" --sha "$S" --receipts "$R"
    fx dupgo
    jq -c . "$R/models.json" | sed 's/"hosts":\["lambda","gx10"\]/"hosts":["lambda"],"hosts":["lambda","gx10"]/' > "$R/m" && mv "$R/m" "$R/models.json"
    row "models receipt with a duplicate array-valued key refused" 1 "models" --sha "$S" --receipts "$R"
    for bad in '4.2e1' '"42\n"' '"0"' '"042"' '"-5"' '42' '"4 2"' 'null'; do
        fx "runid-$bad"
        jq --argjson v "$bad" '.run_id = $v' "$R/clean-room.json" > "$R/m" && mv "$R/m" "$R/clean-room.json"
        row "clean-room run_id $bad refused" 1 "run_id" --sha "$S" --receipts "$R"
    done

    # ---- (c) --execute ----
    fx exec
    a=$(cd "$W" && bash "$SUBJECT" --sha "$S" --receipts "$R" --execute 2>&1); a_rc=$?
    holds "execute: all green exits 0" test "$a_rc" = 0
    holds "execute: local annotated tag at the sha" tag_at_sha "$W"
    holds "execute: tag message is 'aprender v0.72.2'" tag_subject_ok
    holds "execute: armed once via the existing guard for 120 s" armed_once
    holds "execute: tag pushed to origin" tag_at_sha "$(origin_of)"
    holds "execute output names the Binary Release dispatch" test -n "$(grep -F 'binary-release.yml' <<< "$a")"
    holds "execute output names the manual publish (RP-001)" test -n "$(grep -F 'RP-001' <<< "$a")"
    row "second --execute is refused: the tag exists" 1 "already exists" --sha "$S" --receipts "$R" --execute

    fx armfail
    printf '1\n' > "$W/fx-arm-rc"
    bump "arm fails"
    row "arming failure stops before the push" 1 "arm" --sha "$S" --receipts "$R" --execute
    holds "arming failure: nothing pushed" no_origin_tag

    # ---- the script body: promote never builds, tests, publishes, dispatches, deletes or force-pushes ----
    a=$(grep -v '^[[:space:]]*#' "$SUBJECT" | grep -v 'printf' | grep -cE '(^|[[:space:];|&])car''go[[:space:]]+(build|test|publish)\b')
    holds "script body holds no cargo build/test/publish" test "$a" = 0
    a=$(grep -v '^[[:space:]]*#' "$SUBJECT" | grep -v 'printf' | grep -cE 'gh[[:space:]]+(workflow|release)|tag[[:space:]]+-d|push[[:space:]]+(-f|--force)')
    holds "script body holds no workflow dispatch, release, tag delete or force push" test "$a" = 0
}

run_table() {
    pass=0
    fail=0
    build_template || { echo "falsify_promote_nightly: cannot build the fixture" >&2; return 2; }
    table
    printf 'falsify_promote_nightly: %s rows held, %s broke\n' "$pass" "$fail"
    [ "$fail" -eq 0 ]
}

mutants() {
    local m names bad=0 mut n=0
    names=$(grep -o '# CHK:[a-z_]*' "$SUBJECT" | sort -u | sed 's/# CHK://')
    [ -n "$names" ] || { echo "falsify_promote_nightly: no CHK markers in $SUBJECT -- nothing to mutate" >&2; return 1; }
    for m in $names; do
        n=$((n + 1))
        mut="$TMP/mutant-$m.sh"
        # replace the marked call with a no-op: the check is dropped, nothing else changes
        sed "s/^\([[:space:]]*\)[^#]*# CHK:$m\$/\1: # dropped CHK:$m/" "$SUBJECT" > "$mut"
        if cmp -s "$SUBJECT" "$mut"; then printf '  MUTANT %-22s did not apply\n' "$m"; bad=$((bad + 1)); continue; fi
        if PROMOTE_SCRIPT="$mut" bash "${BASH_SOURCE[0]}" > "$TMP/mutant-$m.log" 2>&1; then
            printf '  MUTANT %-22s SURVIVED (no row went red)\n' "$m"; bad=$((bad + 1))
        else
            printf '  mutant %-22s caught\n' "$m"
        fi
    done
    printf 'falsify_promote_nightly: %s of %s mutants not caught\n' "$bad" "$n"
    [ "$bad" -eq 0 ]
}

case "${1:-}" in
    "") run_table ;;
    --mutants) run_table && mutants ;;
    *) echo "usage: falsify_promote_nightly.sh [--mutants]" >&2; exit 2 ;;
esac
