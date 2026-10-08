#!/usr/bin/env bash
# prepare_bump.sh — §4.2 of docs/specifications/06x-release-schedule.md: the bump and the
# CHANGELOG section. Two invocations, because the section needs a human-quality summary:
#   prepare_bump.sh <version>          worktree at origin/main, bump-version.sh <version> + --check,
#                                      CHANGELOG [<version>] drafted from the PRs merged since the last
#                                      tag (with a placeholder --ship refuses)
#   prepare_bump.sh <version> --ship   PR body judged by §6 R-2 (#3699), model-ladder receipts for <version>
#                                      required (#3708), pre-push checks, commit, push,
#                                      PR in milestone <version> with auto-merge; prints the autopilot
#                                      launch line for that PR. Case table: scripts/check_release_bump_pr_body.sh
# The train's identity is DERIVED (#3618, lib_release_params.sh): milestone, epic, last tag and the
# state dir AP come from GitHub and the repo, never from literals.
# §4.1 (freeze: open milestone items move to the next milestone with slipped_from:) is done by hand before this.
set -uo pipefail
die() { printf 'STOP %s\n' "$*" >&2; exit 1; }
# The conventional-commit group of a merged PR: fix -> Fixed, feat -> Added, anything else -> Changed.
PB_JQ_GROUP='def group: ((.title | capture("^(?<k>\\w+)(\\([^)]*\\))?!?:").k | ascii_downcase) // "")
    | if . == "fix" then "Fixed" elif . == "feat" then "Added" else "Changed" end;'

# pb_section MERGED_JSON VERSION MARK OUT -> the CHANGELOG [VERSION] draft in OUT; prints the tally
pb_section() {
    local day; day=$(date -u +%F) || return 1  # bashrs disable-line=DET002
    jq -r --arg v "$2" --arg mark "$3" --arg day "$day" "$PB_JQ_GROUP"'
        [sort_by(.number)[] | {g: group, l: "- \(.title) (#\(.number))"}] as $p
        | ["## [\($v)] - \($day)", "", $mark, ""]
          + ([["Added", "Fixed", "Changed"][] as $g | [$p[] | select(.g == $g) | .l]
              | if . == [] then [] else ["### \($g)", ""] + . + [""] end] | add)
        | .[:-1][]' "$1" > "$4" || return 1
    jq -r "$PB_JQ_GROUP"' [.[] | group] as $g
        | "\(length) merged PRs since the last tag: "
          + (["Added", "Fixed", "Changed"] | map(. as $n | "\($n) \([$g[] | select(. == $n)] | length)") | join(", "))' "$1"
}

# pb_splice CHANGELOG SECTION -> SECTION inserted under the one `## [Unreleased]` line; 1 if not exactly one
pb_splice() {
    local s sec rest anchor=$'## [Unreleased]\n'
    s=$(cat -- "$1" && printf x) || return 1
    s=${s%x}
    sec=$(cat -- "$2") || return 1
    # python's text mode read \r\n and a lone \r as \n; keep that (#4352)
    s=${s//$'\r\n'/$'\n'}; s=${s//$'\r'/$'\n'}; sec=${sec//$'\r\n'/$'\n'}; sec=${sec//$'\r'/$'\n'}
    sec=${sec%"${sec##*[!$'\n']}"}
    rest=${s#*"$anchor"}
    if [ "$rest" = "$s" ] || [[ $rest == *"$anchor"* ]]; then
        echo "CHANGELOG has no single [Unreleased] anchor" >&2; return 1
    fi
    printf '%s' "${s%%"$anchor"*}$anchor"$'\n'"$sec"$'\n'"$rest" > "$1"
}

# pb_policy_covers ROOT VERSION -> rc 0 the standing release policy (ladder.release_policy) covers VERSION,
# 1 it does not, 2 the policy could not be read (never "not covered").
pb_policy_covers() {
    local root=$1 v=$2
    ( . "$root/scripts/lib/release_policy.sh" || exit 2
      t=$(mktemp) || exit 2
      release_policy_ladder "$root/contracts/model-capability-ladder-v1.yaml" "$v" > "$t"; r=$?
      lad=$(cat "$t"); rm -f "${t:?}"
      [ -z "$lad" ] || [ "$lad" = "$root/contracts/model-capability-ladder-v1.yaml" ] || rm -f "${lad:?}"
      [ "$r" = 0 ] || { echo "release policy: $RP_WHY" >&2; exit 2; }
      [ "$RP_APPLIES" = 1 ] )
}
# pb_carry_cert ROOT VERSION BASE -> under the standing release policy, ROOT/evidence/crux/VERSION/ holds the CRUX
# prompt certification the models lane requires at the release commit (models_t1.sh crux mode). 0.70.1 carried it
# by hand: the newest earlier version's certification, apr_commit set to the tree it was cut from, the inventory
# verbatim. A certification for a different prompt set is not carried: that needs a new certification run.
# rc 0 carried, present already, or not covered; 1 refused (no earlier certification, prompts changed);
# 2 the policy or a file could not be read.
pb_carry_cert() {
    local root=$1 v=$2 base=$3 dst src prev p pv vc lad got want
    pb_policy_covers "$root" "$v" || { [ $? = 1 ] && { echo "CERT not carried: the standing release policy does not cover $v"; return 0; }; return 2; }
    dst="$root/evidence/crux/$v"
    if [ -f "$dst/prompt-certification.json" ]; then echo "CERT present: $dst/prompt-certification.json"; return 0; fi
    . "$root/scripts/lib/release_policy.sh" || return 2
    vc=$(rp_core "$v") || return 2
    prev=""
    for p in "$root"/evidence/crux/*/prompt-certification.json; do
        [ -f "$p" ] && [ -f "${p%.json}-inventory.json" ] || continue
        pv=${p%/prompt-certification.json}; pv=${pv##*/}
        pv=$(rp_core "$pv") || continue
        [ "$pv" != "$vc" ] && rp_ge "$vc" "$pv" || continue
        [ -z "$prev" ] || [ "$(printf '%s\n%s\n' "${prev%%	*}" "$pv" | sort -V | tail -n 1)" = "$pv" ] && prev="$pv	$p"
    done
    [ -n "$prev" ] || { echo "CERT refused: no earlier evidence/crux/<version>/prompt-certification.json below $v to carry" >&2; return 1; }
    src=${prev#*	}
    want=$(jq -er .prompts_sha256 "$src") && p=$(jq -er .prompts "$src") || { echo "CERT refused: $src names no prompts or prompts_sha256" >&2; return 2; }
    got=$(sha256sum < "$root/$p" 2>/dev/null | cut -d' ' -f1) || got=""
    [ "$got" = "$want" ] || { echo "CERT refused: $p is not the prompt set $src certified (sha256 ${got:-unreadable}, certified $want) -- run a new certification" >&2; return 1; }
    mkdir -p "$dst" && cp -- "${src%.json}-inventory.json" "$dst/prompt-certification-inventory.json" \
        && jq --arg c "$base" '.apr_commit = $c' "$src" > "$dst/prompt-certification.json" || return 2
    echo "CERT carried: $src -> $dst (apr_commit $base)"
}

# pb_trailers MODEL AGENT -> the bump commit's trailer block; 1 (and nothing printed) when MODEL is not
# a model the fleet runs or AGENT is not a session name. D7 (#4928): the trailer was the literal
# "Claude Fable 5.1", a model no release worker runs, so every bump commit named the wrong author.
# MODEL and AGENT come from RELEASE_MODEL and RELEASE_AGENT; the defaults are the release worker's.
PB_FLEET_MODELS='Claude Opus 5.5, Claude Sonnet 5.5, Claude Haiku 5.5'
pb_trailers() {
    case "$1" in
        'Claude Opus 5.5'|'Claude Sonnet 5.5'|'Claude Haiku 5.5') ;;
        *) echo "commit trailer model '$1' is not one the fleet runs ($PB_FLEET_MODELS)" >&2; return 1 ;;
    esac
    [[ $2 =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "commit trailer agent '$2' is not a session name" >&2; return 1; }
    printf 'Co-Authored-By: %s <noreply@anthropic.com>\nAgent: %s\n' "$1" "$2"
}

pb_self_test() {
    local d fail=0 got; d=$(mktemp -d) || return 2
    printf '%s' '[{"number":4,"title":"fixup: d"},{"number":1,"title":"Fix(x)!: a"},{"number":2,"title":"feat: b"},{"number":3,"title":"chore: c"}]' > "$d/m.json"
    got=$(pb_section "$d/m.json" 9.9.9 MARK "$d/s.md")
    if [ "$got" = "4 merged PRs since the last tag: Added 1, Fixed 1, Changed 2" ]; then echo "  ok   section tally groups fix/feat/other"
    else echo "  FAIL section tally: $got"; fail=1; fi
    got=$(pb_trailers 'Claude Opus 5.5' apr-x 2>/dev/null | tr '\n' '|')
    if [ "$got" = "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>|Agent: apr-x|" ]; then echo "  ok   trailers: a fleet model and the agent"
    else echo "  FAIL trailers: $got"; fail=1; fi
    for got in 'Claude Fable 5.1|apr-x' 'Claude Opus|apr-x' 'Claude Opus 5.5 |apr-x' 'Claude Opus 5.5|' 'Claude Opus 5.5|a b'; do
        if pb_trailers "${got%%|*}" "${got#*|}" > /dev/null 2>&1; then echo "  FAIL trailers accepted '$got'"; fail=1
        else echo "  ok   trailers: refused '$got'"; fi
    done
    got=$(grep -v '^## \[' "$d/s.md" | tr '\n' '|')
    if [ "$got" = "|MARK||### Added||- feat: b (#2)||### Fixed||- Fix(x)!: a (#1)||### Changed||- chore: c (#3)|- fixup: d (#4)|" ]; then echo "  ok   section: Added, Fixed, Changed in order, PRs by number, fixup is not fix"
    else echo "  FAIL section body: $got"; fail=1; fi
    printf '# C\n## [Unreleased]\nold\n' > "$d/c.md"; printf 'NEW\n\n' > "$d/n.md"
    if pb_splice "$d/c.md" "$d/n.md" && [ "$(tr '\n' '|' < "$d/c.md")" = "# C|## [Unreleased]||NEW|old|" ]; then echo "  ok   splice lands under the one [Unreleased] line"
    else echo "  FAIL splice: $(tr '\n' '|' < "$d/c.md")"; fail=1; fi
    printf '## [Unreleased]\n## [Unreleased]\n' > "$d/c.md"
    if pb_splice "$d/c.md" "$d/n.md" 2>/dev/null; then echo "  FAIL splice accepted two [Unreleased] anchors"; fail=1; else echo "  ok   splice refuses two anchors"; fi
    printf '# C\n' > "$d/c.md"
    if pb_splice "$d/c.md" "$d/n.md" 2>/dev/null; then echo "  FAIL splice accepted no anchor"; fail=1; else echo "  ok   splice refuses no anchor"; fi
    # pb_carry_cert: a fixture tree with this checkout's policy reader, a ladder per case and certifications
    local r lib rc
    lib="$(cd "$(dirname "$0")/.." && pwd)/lib"
    pb_cert_tree() { # pb_cert_tree DIR POLICY(covers|none|bad) VERSIONS... -> DIR with a certification per VERSION
        local t=$1 pol=$2 x; shift 2
        mkdir -p "$t/scripts/lib" "$t/contracts" && cp -- "$lib/release_policy.sh" "$lib"/release_policy_*.awk "$t/scripts/lib/" || return 2
        printf 'P1\n' > "$t/prompts.json"
        {   printf 'ladder:\n'
            [ "$pol" = none ] || printf '  release_policy:\n    name: crux-smoke\n    since: "0.0.0"\n    date: "d"\n    quote: "q"\n    hosts: [lambda, gx10]\n    thinking: ["off"]\n    larger_rows: nightly\n    red_row_needs: ticket\n    ticket_owner: "#1"\n'
            [ "$pol" != covers ] || printf '    release_notes: known_failures\n'
            printf '  emergency_scopes:\n'
        } > "$t/contracts/model-capability-ladder-v1.yaml"
        for x in "$@"; do
            mkdir -p "$t/evidence/crux/$x"
            printf '{"prompts":"prompts.json","prompts_sha256":"%s","apr_commit":"old-%s"}\n' "$(sha256sum < "$t/prompts.json" | cut -d' ' -f1)" "$x" > "$t/evidence/crux/$x/prompt-certification.json"
            printf '{"inv":"%s"}\n' "$x" > "$t/evidence/crux/$x/prompt-certification-inventory.json"
        done
    }
    r="$d/c1"; pb_cert_tree "$r" covers 0.69.1 0.70.1 0.9.0 0.72.0 || return 2
    pb_carry_cert "$r" 0.71.0 base1 > /dev/null 2>&1; rc=$?
    if [ "$rc" = 0 ] && [ "$(jq -r .apr_commit "$r/evidence/crux/0.71.0/prompt-certification.json" 2>/dev/null)" = base1 ] \
        && [ "$(jq -r .inv "$r/evidence/crux/0.71.0/prompt-certification-inventory.json" 2>/dev/null)" = 0.70.1 ]; then
        echo "  ok   cert: covered 0.71.0 carries the newest earlier one (0.70.1, not 0.9.0 or 0.72.0), apr_commit = the base"
    else echo "  FAIL cert carry (rc $rc): $(cat "$r/evidence/crux/0.71.0/"*.json 2>/dev/null | tr '\n' ' ')"; fail=1; fi
    r="$d/c2"; pb_cert_tree "$r" none 0.70.1 || return 2
    pb_carry_cert "$r" 0.71.0 base1 > /dev/null 2>&1; rc=$?
    if [ "$rc" = 0 ] && [ ! -e "$r/evidence/crux/0.71.0" ]; then echo "  ok   cert: no policy -> nothing carried"
    else echo "  FAIL cert: uncovered version (rc $rc) wrote evidence/crux/0.71.0"; fail=1; fi
    r="$d/c3"; pb_cert_tree "$r" covers 0.70.1 0.71.0 || return 2
    pb_carry_cert "$r" 0.71.0 base1 > /dev/null 2>&1; rc=$?
    if [ "$rc" = 0 ] && [ "$(jq -r .apr_commit "$r/evidence/crux/0.71.0/prompt-certification.json")" = old-0.71.0 ]; then echo "  ok   cert: one already present is kept as it is"
    else echo "  FAIL cert: a present certification was rewritten (rc $rc)"; fail=1; fi
    r="$d/c4"; pb_cert_tree "$r" covers 0.70.1 || return 2
    printf 'P2\n' > "$r/prompts.json"
    pb_carry_cert "$r" 0.71.0 base1 > /dev/null 2>&1; rc=$?
    if [ "$rc" = 1 ] && [ ! -e "$r/evidence/crux/0.71.0" ]; then echo "  ok   cert: a changed prompt set is refused, not carried"
    else echo "  FAIL cert: changed prompts gave rc $rc"; fail=1; fi
    r="$d/c5"; pb_cert_tree "$r" covers 0.72.0 || return 2
    pb_carry_cert "$r" 0.71.0 base1 > /dev/null 2>&1; rc=$?
    if [ "$rc" = 1 ] && [ ! -e "$r/evidence/crux/0.71.0" ]; then echo "  ok   cert: only a later version's -> refused (never carried backwards)"
    else echo "  FAIL cert: no earlier certification gave rc $rc"; fail=1; fi
    r="$d/c6"; pb_cert_tree "$r" bad 0.70.1 || return 2
    pb_carry_cert "$r" 0.71.0 base1 > /dev/null 2>&1; rc=$?
    if [ "$rc" = 2 ] && [ ! -e "$r/evidence/crux/0.71.0" ]; then echo "  ok   cert: an unreadable policy is rc 2, not \"no policy\""
    else echo "  FAIL cert: unreadable policy gave rc $rc"; fail=1; fi
    rm -rf -- "${d:?}"
    if [ "$fail" -eq 0 ]; then echo "prepare_bump self-test: PASS"; else echo "prepare_bump self-test: FAIL"; fi
    return "$fail"
}
if [ "${1:-}" = --self-test ]; then pb_self_test; exit $?; fi
# D4/D5/D6/D7 (PMAT-3459): $0-derived root and CARGO_HOME-relative cargo. Root resolved
# BEFORE any cd. NOT `git rev-parse --show-toplevel` — refused on a bind-mounted tree (#3586).
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)" || die "cannot resolve the repo root from $0"
# shellcheck source=scripts/release/lib_release_params.sh
. "$REPO_ROOT/scripts/release/lib_release_params.sh" || exit 2
release_params "${1:-}" "$REPO_ROOT" || die "usage: prepare_bump.sh <version> [--ship]"
MS=$V
EPIC=$(release_epic_number) || die "release epic for $V: cannot resolve exactly one"
LAST_TAG=$(release_last_tag "$REPO_ROOT") || die "no release tag below $T"
B="$AP/bump"; BR="release-$V"; MARK='<!-- one-paragraph summary of the train: EDIT BEFORE --ship -->'
export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"

if [ "${2:-}" != "--ship" ]; then
  cd "$REPO_ROOT" || die "no repo"
  git fetch -q origin main || die "fetch failed"
  [ -e "$B" ] && die "$B exists: review it, or 'git worktree remove' it to start over"
  git worktree add -q -b "$BR" "$B" origin/main || die "worktree add failed"
  cd "$B" || die "cd $B"
  bash scripts/bump-version.sh "$V" > "$AP/bump.log" 2>&1 || die "bump-version.sh $V failed ($AP/bump.log)"
  bash scripts/bump-version.sh --check >> "$AP/bump.log" 2>&1 || die "bump-version.sh --check failed after the bump"
  pb_carry_cert "$PWD" "$V" "$(git rev-parse HEAD)" || die "the CRUX prompt certification for $V could not be carried (see above)"
  since=$(git log -1 --format=%cI "$LAST_TAG") || die "no tag $LAST_TAG"
  gh pr list --repo $REPO --state merged --search "merged:>=$since" --limit 500 --json number,title > "$AP/merged.json" || die "gh pr list failed"
  pb_section "$AP/merged.json" "$V" "$MARK" "$AP/section.md" || die "CHANGELOG draft from $AP/merged.json failed"
  pb_splice CHANGELOG.md "$AP/section.md" || die "CHANGELOG splice failed"
  printf 'REVIEW %s/CHANGELOG.md [%s]: replace the placeholder with the train summary, curate the bullets, then run: %s %s --ship\n' "$B" "$V" "$0" "$V"
  exit 0
fi

cd "$B" || die "no $B (run without --ship first)"
TRAILERS=$(pb_trailers "${RELEASE_MODEL:-Claude Opus 5.5}" "${RELEASE_AGENT:-release-autopilot}") ||
    die "the bump commit's trailer is refused (D7, #4928); nothing committed, pushed or opened"
# T-2 PREFLIGHT BEFORE THE BUMP (operator 2026-09-17): the bump PR is refused without a GO receipt for its parent sha.
parent=$(git rev-parse origin/main); rcpt="$AP/preflight-$parent.verdict"
[ -f "$rcpt" ] && grep -q '^GO ' "$rcpt" || die "no T-2 GO receipt for origin/main $parent ($rcpt) — run: $REPO_ROOT/scripts/release/t2_preflight.sh $V"
grep -qF "$MARK" CHANGELOG.md && die "CHANGELOG [$V] still carries the EDIT placeholder"
bash scripts/bump-version.sh --check > /dev/null 2>&1 || die "bump-version.sh --check failed"
awk -v h="## [$V]" 'index($0, h) == 1 {f = 1; next} f && /^## \[/ {exit} f' CHANGELOG.md > "$AP/release_notes.md"
[ -s "$AP/release_notes.md" ] || die "the CHANGELOG [$V] section is empty"
# THE PR BODY IS BUILT AND JUDGED BEFORE ANYTHING IS COMMITTED OR PUSHED (#3699). It embeds the
# CHANGELOG section, which cites the epic and context refs ("refs #N", "(#N layer 1)"), and
# ci.yml's §6 R-2 step (check_pr_closes_issue.sh) fails a body citing an open issue with no
# closing keyword and no keep-open line. That hit 0.68.2 (#3498) and 0.69.0 (#3698), each fixed
# by hand -- and because that step reads the frozen event payload, a body edit cost a cancel, a
# force-cancel and a close+reopen. The keep-open line names exactly what the guard's own
# classifier (--list-owed) says is owed, so the two cannot disagree. Its reason is #3699's own text;
# it carries no `#`, so it cannot add a closing reference: one that closed the epic on this PR's
# merge would be the #3400 landmine again. Nothing owed, no line.
CLOSES_GUARD="$REPO_ROOT/scripts/check_pr_closes_issue.sh"
{ printf 'Release bump for **%s** (06x release schedule §4.2): `bump-version.sh %s` across every workspace, and the CHANGELOG section below.\n\nWhen this merges, `scripts/release/autopilot.sh` runs the rest of the train, fail-closed (APR-RELEASE-001 §4): T-1 deep run, pre-publish dogfood, tag + release, `clean-room.yml` dispatched on the tag with the run id recorded (T-3), all release assets checked with `scripts/check_release_assets.sh`, publish preflight, the crates.io cascade (T-4, automated per operator 2026-09-13), `install.sh --version` receipts on intel and gx10 plus the CUDA asset receipts on gx10 and yoga, and the epic and milestone close.\n\n' "$V" "$V"; cat "$AP/release_notes.md"; } > "$AP/pr_body.md"
owed=$(bash "$CLOSES_GUARD" --list-owed --body "$AP/pr_body.md") || die "check_pr_closes_issue.sh --list-owed could not read $AP/pr_body.md"
owed=$(printf '%s' "$owed" | tr '\n' ' ')
[ -z "$owed" ] || printf '\nkeep-open: %s -- cited by the CHANGELOG for context; each closes via its own PR; the release EPIC closes at T-4\n' "$owed" >> "$AP/pr_body.md"
printf '\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n' >> "$AP/pr_body.md"
bash "$CLOSES_GUARD" --body "$AP/pr_body.md" > "$AP/r2.log" 2>&1 || die "the bump PR body fails §6 R-2; nothing committed, pushed or opened ($AP/r2.log, $AP/pr_body.md)"
# THE MODEL-LADDER RECEIPTS FOR THE NEW VERSION RIDE ON THE BUMP (#3708). The dogfood's
# check_model_ladder row reads evidence/dogfood/models/<V>/<host>.json for the version being cut;
# 0.68.2 committed them on its bump by hand (#3498) and 0.69.0 did not, so the row could not be
# measured until after the tag. This script does not PRODUCE them (scripts/model_ladder.sh, on each
# required host, with an apr built from this tree): it refuses without them, judged by the SAME
# judge the dogfood runs. `git add -A` below commits whatever the judge read, unless it is ignored.
# Under the standing release policy the bump carries no smoke receipts: CRUX smoke is measured at T-1
# on the merged release commit (autopilot.sh t1_models), and the tag step and preflight R7 judge those
# receipts. A receipt committed here would be from a pre-bump binary. The certification still rides.
pb_policy_covers "$PWD" "$V"; prc=$?
case $prc in
  0) [ -f "evidence/crux/$V/prompt-certification.json" ] \
       || die "the standing release policy covers $V but the bump tree has no evidence/crux/$V/prompt-certification.json"
     echo "LADDER not judged at the bump: the standing release policy covers $V -- CRUX smoke is measured at T-1 on the release commit" | tee "$AP/ladder.log" ;;
  1) bash scripts/check_model_ladder.sh --version "$V" > "$AP/ladder.log" 2>&1 || {
       grep -E '^(FAIL|decline)' "$AP/ladder.log" >&2
       die "model-ladder receipts for $V are not green in the bump tree; nothing committed, pushed or opened ($AP/ladder.log)"
     } ;;
  *) die "the standing release policy cannot be judged for $V; nothing committed, pushed or opened" ;;
esac
ignored=$(git ls-files --others --ignored --exclude-standard -- "evidence/dogfood/models/$V" "evidence/crux/$V")
[ -z "$ignored" ] || die "model-ladder receipts for $V are gitignored, so the bump would not commit them: $ignored"
cargo_bin() { "${CARGO_HOME:-$HOME/.cargo}"/bin/cargo "$@"; }
cargo_bin fmt --all -- --check > /dev/null 2>&1 || die "cargo fmt --check failed"
cargo_bin deny check advisories > "$AP/deny.log" 2>&1 || die "cargo deny check advisories failed ($AP/deny.log)"
cargo_bin nextest --version > /dev/null 2>&1; cargo_bin "test" --quiet --package aprender-contracts --lib > "$AP/contracts.log" 2>&1 || die "aprender-contracts lib tests failed ($AP/contracts.log)"
git add -A
git commit -q -F - <<MSG || die "commit refused"
release: $V

bump-version.sh $V (every workspace, facades included; --check green) and the CHANGELOG [$V] section,
per docs/specifications/06x-release-schedule.md §4.2. After this merges, scripts/release/autopilot.sh runs APR-RELEASE-001 §4 T-1..T-4 + close:
deep (T-1, local), pre-publish dogfood, tag + release, clean-room.yml dispatched on the tag (T-3, run id recorded), assets by command, preflight, cascade (T-4, automated), install, host + installer receipts, close.

Pmat-Ticket: PMAT-$EPIC
$TRAILERS
MSG
git push -q -u origin "$BR" || die "push failed"
url=$(gh pr create --repo $REPO --base main --head "$BR" --milestone "$MS" --title "release: $V" --body-file "$AP/pr_body.md") || die "gh pr create failed"
n=${url##*/}
ARM="$REPO_ROOT/scripts/arm_pr_automerge.sh"; [ -f "$ARM" ] || die "no $ARM"
armed=0; for _ in $(seq 1 30); do bash "$ARM" "$n" > "$AP/arm.log" 2>&1; rc=$?; [ $rc -eq 0 ] && { armed=1; break; }; [ $rc -eq 3 ] || break; sleep 120; done
[ $armed = 1 ] && printf 'ARMED #%s (guard-tree green on the head)\n' "$n" || printf 'WARN not armed: %s\n' "$(tail -1 "$AP/arm.log")"
printf 'BUMP PR #%s (%s)\nlaunch: setsid nohup %s/scripts/release/autopilot.sh %s %s > /dev/null 2>&1 < /dev/null & disown\n' "$n" "$url" "$REPO_ROOT" "$V" "$n"
