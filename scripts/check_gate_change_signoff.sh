#!/usr/bin/env bash
# check_gate_change_signoff.sh - a PR that changes what a check accepts
# (stricter included) needs a maintainer sign-off and two non-author review
# receipts, each naming the PR's HEAD sha (#4815).
#
# Evidence for #4815: 17 PRs merged between 2026-10-04 07:40Z and 2026-10-05
# 16:00Z changed workflows, check scripts, release scripts, contracts or test
# ignore attributes with no sign-off. Two of them (#4771, #4762) loosened a
# check and had to be undone (#4778, #4772). Nothing on the PR path asked
# whether a diff was a gate change, so nothing asked for the sign-off.
#
# Two steps:
#   1. CLASSIFY the three-dot diff base...head (`git diff --no-renames -U0`).
#      Every rule in GATE_RULES below is a row; a diff is a gate change when
#      at least one row matches. PATH rows match a changed file's path. LINE
#      rows match an added or removed line in a file whose path matches.
#   2. For a gate change, read the evidence (JSON Lines, one record per line):
#        {"kind":"signoff","head_sha":"<40 hex>","by":"<who>","role":"maintainer"}
#        {"kind":"review","head_sha":"<40 hex>","reviewer":"<who>","author":"<who>","verdict":"PASS"}
#      PASS needs >=1 maintainer sign-off AND >=2 distinct PASS reviewers who
#      are not the PR author, every record's head_sha EQUAL to the head sha
#      (full 40 hex, no prefix match). A record naming an older head counts
#      for nothing: the sign-off is for the diff that was read, not the PR.
#      Identities must be GitHub logins (letters, digits, "-"), compared
#      case-insensitively. The sign-off is NOT author-excluded: the
#      maintainer may also be the PR author; only reviews are.
#      Where the evidence comes from (PR comment, check run, signed ref) is
#      the wiring step and is out of scope here; this script only judges it.
#
# Modes. This is a NEW gate, so it lands as a ratchet:
#   --mode report (DEFAULT)  a refusal prints "would be RED" and exits 0.
#   --mode enforce           a refusal exits 1.
# Making it blocking is a ruling, not a code change here.
#
# Exit codes: 0 pass (or report-mode refusal), 1 refuse (enforce only),
#             2 usage error or NOT_MEASURED (diff unreadable, jq missing,
#               evidence unparseable) in either mode - a gate that cannot
#               read its input never reports PASS.
#
# usage:
#   check_gate_change_signoff.sh --base SHA --head SHA [--evidence FILE]
#                                [--author LOGIN] [--mode report|enforce]
#   check_gate_change_signoff.sh --classify < diff     # print matched rule ids
#   check_gate_change_signoff.sh --self-test           # also the bare run
#
# --self-test runs the case table (classifier must-match / must-not-match rows
# built from the #4815 PRs, plus the sign-off cases), then plants one mutant
# per GATE_RULES row (the row removed) and requires that row's own probe case
# to turn non-gate. A row with no probe case is dead and fails the test.

set -euo pipefail

SELF_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

usage() {
    printf 'usage: %s --base SHA --head SHA [--evidence FILE] [--author LOGIN] [--mode report|enforce] | --classify | --self-test\n' "$(basename "$0")" >&2
    exit 2
}

# id@kind@path ERE@line ERE (LINE rows only). Order is the print order.
GATE_RULES='
P-WORKFLOW@path@^\.github/@
P-CI@path@^ci/@
P-CHECK-SCRIPT@path@^scripts/([^/]*(check|guard|gate|ratchet|lint|release|criteria|conformance|receipt|quorum|pr_review|mutants|coverage|run_tests|nextest|_pin|_bin\.sh)[^/]*|ci_[^/]*)$@
P-ACCEPT-LIST@path@(^|/)[^/]*(allowlist|skips|acknowledged|waiver)[^/]*$@
P-GUARD-TEST@path@^scripts/tests/@
P-RELEASE@path@^scripts/release/@
P-CONTRACT@path@^contracts/@
P-BASELINE@path@(^|/)[^/]*baseline[^/]*$@
P-GATE-CONFIG@path@^(Makefile|\.config/nextest\.toml|\.?rustfmt\.toml|\.pmat-gates\.toml|\.?clippy\.toml|deny\.toml|rust-toolchain(\.toml)?|\.cargo/config\.toml)$@
P-CHECKER-SRC@path@^crates/aprender-contracts(-cli|-macros)?/src/@
L-TEST-IGNORE@line@\.rs$@^[-+][[:space:]]*#\[(ignore|cfg_attr\(.*ignore)
L-LINT-LEVEL@line@(^|/)Cargo\.toml$@^[-+][[:space:]]*[A-Za-z0-9_:-]+[[:space:]]*=[[:space:]]*(\{.*level[[:space:]]*=[[:space:]]*)?"(allow|warn|deny|forbid)"
'

# classify: unified diff on stdin -> matched rule ids, one per line, sorted
# unique. GATE_RULE_DROP=<id> removes one row (the self-test's planted mutant).
classify() {
    local drop="${GATE_RULE_DROP:-}" path="" line r id kind pre lre hits="" h n
    local -a rows=()
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        [ "${line%%@*}" = "$drop" ] && continue
        rows+=("$line")
    done <<< "$GATE_RULES"
    while IFS= read -r line; do
        case "$line" in
            'diff --git '*)
                # --no-renames: both halves name the same path X, so the
                # header is `a/X b/X` (2|X|+5 chars) or, when git quotes a
                # name, `"a/X" "b/X"` (2|X|+7 inside the outer quotes). Split
                # by length, not on " b/", which a file name may contain.
                h="${line#diff --git }"
                if [ "${h:0:1}" = '"' ]; then h="${h#\"}"; h="${h%\"}"; n=$(( (${#h} - 7) / 2 )); else n=$(( (${#h} - 5) / 2 )); fi
                if [ "$n" -gt 0 ]; then path="${h:2:n}"; else path="$h"; fi
                for r in "${rows[@]}"; do
                    IFS='@' read -r id kind pre lre <<< "$r"
                    if [ "$kind" = path ] && [[ "$path" =~ $pre ]]; then hits+="$id"$'\n'; fi
                done ;;
            '+++ '*|'--- '*) ;;
            [-+]*)
                for r in "${rows[@]}"; do
                    IFS='@' read -r id kind pre lre <<< "$r"
                    if [ "$kind" = line ] && [[ "$path" =~ $pre ]] && [[ "$line" =~ $lre ]]; then hits+="$id"$'\n'; fi
                done ;;
        esac
    done
    if [ -n "$hits" ]; then printf '%s' "$hits" | sort -u; fi
    return 0
}

# judge HEAD AUTHOR EVIDENCE_FILE -> prints "maint=<n> reviews=<n>", rc 2 on
# unparseable evidence. A missing or empty file is "no evidence", not an error.
# Identities are strings compared case-insensitively (GitHub logins are). The
# PR author comes from --author, never from the record; a record's own
# "author" field can only exclude more. GATE_JUDGE_MUTANT=no-head|no-unique|
# no-author is the self-test's planted mutant seam.
judge() {
    local head="$1" author="$2" ev="$3" m r lg='test("^[A-Za-z0-9][A-Za-z0-9-]*$")' hc='.head_sha==$h' uq='unique |' ax='. != ($a|ascii_downcase)'
    case "${GATE_JUDGE_MUTANT:-}" in no-head) hc='true' ;; no-unique) uq='' ;; no-author) ax='true' ;; esac
    if [ -z "$ev" ] || [ ! -s "$ev" ]; then printf 'maint=0 reviews=0\n'; return 0; fi
    m=$(jq -s --arg h "$head" "[.[] | select(.kind==\"signoff\" and .role==\"maintainer\" and $hc and (.by|type)==\"string\" and (.by|$lg))] | length" "$ev" 2>/dev/null) || return 2
    r=$(jq -s --arg h "$head" --arg a "$author" "[.[] | select(.kind==\"review\" and .verdict==\"PASS\" and $hc and (.reviewer|type)==\"string\" and (.reviewer|$lg) and ((.author|type)!=\"string\" or (.reviewer|ascii_downcase)!=(.author|ascii_downcase))) | .reviewer | ascii_downcase | select($ax)] | $uq length" "$ev" 2>/dev/null) || return 2
    printf 'maint=%s reviews=%s\n' "$m" "$r"
}

# decide RULES(newline list) JUDGE_LINE -> PASS:<why> | REFUSE:<why>
# GATE_JUDGE_MUTANT=one-review lowers the review threshold (mutant seam).
decide() {
    local rules="$1" j="$2" m r need=2
    [ "${GATE_JUDGE_MUTANT:-}" != one-review ] || need=1
    if [ -z "$rules" ]; then printf 'PASS:not a gate change\n'; return 0; fi
    m="${j#maint=}"; m="${m%% *}"; r="${j##*reviews=}"
    if [ "$m" -ge 1 ] && [ "$r" -ge "$need" ]; then printf 'PASS:signed off on head\n'
    else printf 'REFUSE:maintainer sign-off %s/1, non-author reviews %s/2 on head\n' "$m" "$r"; fi
}

# d PATH [LINE...] -> a one-file unified diff (self-test fixture)
d() {
    local p="$1"; shift
    printf 'diff --git a/%s b/%s\n--- a/%s\n+++ b/%s\n@@ -1 +1 @@\n' "$p" "$p" "$p" "$p"
    if [ $# -gt 0 ]; then printf '%s\n' "$@"; fi
    printf '+x\n'
}

self_test() {
    local tmp fails=0 cases=0 got want
    tmp=$(mktemp -d); GATE_TMP="$tmp"; trap '[ -z "${GATE_TMP:-}" ] || rm -rf "${GATE_TMP:?}"' EXIT
    ck() { # name want(rule id | -) diff-text
        cases=$((cases + 1))
        got=$(printf '%s\n' "$3" | classify | tr '\n' ' '); got="${got% }"
        want="$2"; [ "$want" != - ] || want=""
        if [ "$got" != "$want" ]; then printf 'FAIL classify %s: want [%s] got [%s]\n' "$1" "$want" "$got" >&2; fails=$((fails + 1)); fi
    }
    # must-match: modelled on the #4815 PRs, each matching ONLY its row.
    ck wf-4730       P-WORKFLOW     "$(d .github/workflows/pr-review-quorum.yml)"
    ck actionlint    P-WORKFLOW     "$(d .github/actionlint.yaml)"
    ck ci-4656       P-CI           "$(d ci/sections.yml)"
    ck check-4770    P-CHECK-SCRIPT "$(d scripts/check_pv_pin.sh)"
    ck mutgate-4782  P-CHECK-SCRIPT "$(d scripts/mutants_diff_gate.sh)"
    ck release-sh    P-CHECK-SCRIPT "$(d scripts/release.sh)"
    ck tool-pin      P-CHECK-SCRIPT "$(d scripts/pv_bin.sh)"
    ck allowlist     P-ACCEPT-LIST  "$(d scripts/duplicate_bin_names_allowlist.txt)"
    ck ack-list      P-ACCEPT-LIST  "$(d scripts/unwired_capabilities_acknowledged.txt)"
    ck guardtest     P-GUARD-TEST   "$(d scripts/tests/guard_tree_test.sh)"
    ck release-4738  P-RELEASE      "$(d scripts/release/coverage_gate.sh)"
    ck contract      P-CONTRACT     "$(d contracts/github-entities-v1.yaml)"
    ck baseline      P-BASELINE     "$(d evidence/ratchets/mutants-baseline.json)"
    ck makefile      P-GATE-CONFIG  "$(d Makefile)"
    ck pmatgates     P-GATE-CONFIG  "$(d .pmat-gates.toml)"
    ck checker-4773  P-CHECKER-SRC  "$(d crates/aprender-contracts/src/ontology/extract/example.rs)"
    ck ignore-4771   L-TEST-IGNORE  "$(d crates/aprender-serve/src/gpu/tests/streaming.rs '+    #[ignore = "FLAKE-0 #4769"]')"
    ck unignore-4778 L-TEST-IGNORE  "$(d crates/aprender-serve/src/gpu/tests/streaming.rs '-    #[ignore = "FLAKE-0 #4769"]')"
    ck cfgattr-ign   L-TEST-IGNORE  "$(d crates/aprender-core/src/a.rs '+#[cfg_attr(miri, ignore)]')"
    ck lint-level    L-LINT-LEVEL   "$(d Cargo.toml '+unwrap_used = "allow"')"
    ck lint-table    L-LINT-LEVEL   "$(d Cargo.toml '-pedantic = { level = "warn", priority = -1 }')"
    ck gh-action     P-WORKFLOW     "$(d .github/actions/setup/action.yml)"
    ck nextest-cfg   P-GATE-CONFIG  "$(d .config/nextest.toml)"
    ck quoted-path   P-CHECK-SCRIPT "$(printf 'diff --git "a/scripts/check_\\303\\251.sh" "b/scripts/check_\\303\\251.sh"\n+x\n')"
    ck b-in-name     P-RELEASE      "$(d 'scripts/release/x b/y.sh')"
    # must-not-match: code-only PRs.
    ck b-in-name-neg -              "$(d 'docs/x b/scripts/check_y.sh')"
    # KNOWN FALSE POSITIVES, pinned on purpose: a prose-only edit of a contract
    # (the shapes of #4676/#4677: a note:/owed_by: measurement record) still
    # flags. The path rule cannot tell a record from a threshold, and erring
    # strict is the ruled default; if a contract-field rule replaces it, these
    # two rows are the ones that must flip to "-".
    ck fp-4676-prose P-CONTRACT     "$(d contracts/thinking-budgets-v1.yaml '-      Generated 8,901 chars without closing at 2048 on the bench host. Whether a larger budget' '+      Generated 8,901 chars without closing at 2048 on the bench host (#3907). MEASURED since')"
    ck fp-4677-prose P-CONTRACT     "$(d contracts/thinking-budgets-v1.yaml '-      Measured on the bench host (RTX 4090, cuda), #3948 receipt' '+      Measured on the bench host (RTX 4090, cuda), re-measured receipt')"
    ck serve-4276    -              "$(d crates/aprender-serve/src/gguf/prefill.rs '+    let n = 1;')"
    ck readme-4467   -              "$(d README.md)"
    ck dep-4632      -              "$(d Cargo.toml '-wasmtime = "47.0.4"' '+wasmtime = "48.0.3"')"
    ck roadmap       -              "$(d docs/roadmaps/roadmap.yaml)"
    ck receipt       -              "$(d evidence/pr-review/4778/d5e0/receipt.intoto.jsonl)"
    ck new-test      -              "$(d crates/aprender-core/src/b.rs '+    #[test]' '+    fn ignores_blank() {}')"
    ck comment-ign   -              "$(d crates/aprender-core/src/c.rs '+    // ignore trailing whitespace')"
    ck install-sh    -              "$(d scripts/install.sh)"
    ck version-bump  -              "$(d Cargo.toml '-version = "0.70.1"' '+version = "0.70.2"')"

    # sign-off cases: #4815's four (a-d), then the receipt rules.
    local H=1111111111111111111111111111111111111111 OLD=2222222222222222222222222222222222222222
    local gate=P-WORKFLOW ev="$tmp/ev.jsonl"
    so() { printf '{"kind":"signoff","head_sha":"%s","by":"maint","role":"maintainer"}\n' "$1"; }
    rv() { printf '{"kind":"review","head_sha":"%s","reviewer":"%s","author":"author","verdict":"%s"}\n' "$1" "$2" "${3:-PASS}"; }
    sk() { # name rules want(PASS|REFUSE) [pr-author]; evidence already in $ev
        cases=$((cases + 1))
        local j
        if j=$(judge "$H" "${4:-author}" "$ev"); then got=$(decide "$2" "$j"); got="${got%%:*}"; else got=NOT_MEASURED; fi
        if [ "$got" != "$3" ]; then printf 'FAIL signoff %s%s: want %s got %s\n' "${GATE_JUDGE_MUTANT:+mutant $GATE_JUDGE_MUTANT survived on }" "$1" "$3" "$got" >&2; fails=$((fails + 1)); fi
    }
    : > "$ev";                                           sk a-no-signoff        "$gate" REFUSE
    { so "$OLD"; rv "$OLD" r1; rv "$OLD" r2; } > "$ev";  sk b-old-head          "$gate" REFUSE
    { so "$H"; rv "$H" r1; rv "$H" r2; } > "$ev";        sk c-on-head           "$gate" PASS
    : > "$ev";                                           sk d-non-gate          ""      PASS
    { so "$H"; rv "$H" r1; rv "$OLD" r2; } > "$ev";      sk one-review-stale    "$gate" REFUSE
    { so "$H"; rv "$H" r1; rv "$H" r1; } > "$ev";        sk same-reviewer-x2    "$gate" REFUSE
    { so "$H"; rv "$H" r1; rv "$H" author; } > "$ev";    sk author-reviews      "$gate" REFUSE
    { so "$H"; rv "$H" r1; rv "$H" r2 FAIL; } > "$ev";   sk failing-review      "$gate" REFUSE
    { rv "$H" r1; rv "$H" r2; } > "$ev";                 sk reviews-no-signoff  "$gate" REFUSE
    { so "${H:0:12}"; rv "$H" r1; rv "$H" r2; } > "$ev"; sk short-sha-signoff   "$gate" REFUSE
    { so "$H"; rv "$H" r1; rv "$H" pr-owner; } > "$ev";  sk author-via-flag     "$gate" REFUSE pr-owner
    { so "$H"; rv "$H" r1; rv "$H" author; } > "$ev";    sk author-via-record   "$gate" REFUSE someone
    { so "$H"; rv "$H" r1; rv "$H" PR-Owner; } > "$ev";  sk author-case         "$gate" REFUSE pr-owner
    { so "$H"; rv "$H" r1; rv "$H" R1; } > "$ev";        sk reviewer-case-dup   "$gate" REFUSE
    { so "$H"; rv "$H" r1; rv "$H" 'r1 '; } > "$ev";      sk reviewer-padded-dup "$gate" REFUSE
    { printf '{"kind":"signoff","head_sha":"%s","by":" ","role":"maintainer"}\n' "$H"; rv "$H" r1; rv "$H" r2; } > "$ev"; sk blank-signoff "$gate" REFUSE
    { so "$H"; printf '{"kind":"review","head_sha":"%s","reviewer":[%s],"verdict":"PASS"}\n' "$H" 1 "$H" 2; } > "$ev"
    sk nonstring-reviewer "$gate" REFUSE
    { printf '{"kind":"signoff","head_sha":"%s","by":1,"role":"maintainer"}\n' "$H"; rv "$H" r1; rv "$H" r2; } > "$ev"
    sk nonstring-signoff "$gate" REFUSE
    { printf '{"kind":"signoff","head_sha":"%s","by":"m","role":"reviewer"}\n' "$H"; rv "$H" r1; rv "$H" r2; } > "$ev"
    sk signoff-not-maintainer "$gate" REFUSE
    { so "$H"; printf '{"kind":"review","head_sha":"%s","reviewer":"r%s","verdict":"PASS"}\n' "$H" 1 "$H" 2; } > "$ev"
    sk review-without-author-field "$gate" PASS
    cases=$((cases + 1))
    printf 'not json\n' > "$ev"
    if judge "$H" author "$ev" >/dev/null 2>&1; then printf 'FAIL signoff garbage-evidence: judged instead of NOT_MEASURED\n' >&2; fails=$((fails + 1)); fi

    # planted judge mutants: each must flip its killing case to PASS.
    { so "$OLD"; rv "$OLD" r1; rv "$OLD" r2; } > "$ev";  GATE_JUDGE_MUTANT=no-head    sk b-old-head       "$gate" PASS
    { so "$H"; rv "$H" r1; rv "$H" r1; } > "$ev";        GATE_JUDGE_MUTANT=no-unique  sk same-reviewer-x2 "$gate" PASS
    { so "$H"; rv "$H" r1; rv "$H" pr-owner; } > "$ev";  GATE_JUDGE_MUTANT=no-author  sk author-via-flag  "$gate" PASS pr-owner
    { so "$H"; rv "$H" r1; rv "$H" r2 FAIL; } > "$ev";   GATE_JUDGE_MUTANT=one-review sk failing-review   "$gate" PASS

    # end to end through main(): a scratch repo, a gate commit G and a
    # non-gate commit N on base B; exit codes are the contract.
    local repo="$tmp/repo" B G N
    {
        git init -q "$repo" && gc() { git -C "$repo" -c user.name=self-test -c user.email=self-test@example.invalid "$@"; } &&
        gc commit -q --allow-empty -m base && B=$(git -C "$repo" rev-parse HEAD) &&
        mkdir -p "$repo/.github/workflows" && printf 'x\n' > "$repo/.github/workflows/x.yml" &&
        gc add -A && gc commit -q -m gate && G=$(git -C "$repo" rev-parse HEAD) &&
        gc checkout -q -b non-gate "$B" && printf 'x\n' > "$repo/README.md" &&
        gc add -A && gc commit -q -m doc && N=$(git -C "$repo" rev-parse HEAD)
    } >/dev/null 2>&1 || { printf 'FAIL main: could not build the scratch repo (git)\n' >&2; fails=$((fails + 1)); B=0; G=0; N=0; }
    mr() { # name want-rc args...
        local name="$1" want="$2" rc; shift 2; cases=$((cases + 1))
        if (cd "$repo" && "$SELF_PATH" "$@" >"$tmp/out" 2>&1); then rc=0; else rc=$?; fi
        if [ "$rc" != "$want" ]; then printf 'FAIL main %s: want rc %s got %s (%s)\n' "$name" "$want" "$rc" "$(tail -1 "$tmp/out")" >&2; fails=$((fails + 1)); fi
    }
    : > "$ev"
    mr report-none   0 --base "$B" --head "$G" --evidence "$ev" --author pr-owner
    if ! grep -q 'would be RED' "$tmp/out"; then printf 'FAIL main report-none: no "would be RED" line\n' >&2; fails=$((fails + 1)); fi
    mr enforce-none  1 --base "$B" --head "$G" --evidence "$ev" --author pr-owner --mode enforce
    git -C "$repo" config diff.noprefix true && git -C "$repo" config diff.mnemonicPrefix true
    mr noprefix-cfg  1 --base "$B" --head "$G" --evidence "$ev" --author pr-owner --mode enforce
    git -C "$repo" config --unset diff.noprefix; git -C "$repo" config --unset diff.mnemonicPrefix
    mr nongate       0 --base "$B" --head "$N" --mode enforce
    { so "$B"; rv "$B" r1; rv "$B" r2; } > "$ev"
    mr enforce-old   1 --base "$B" --head "$G" --evidence "$ev" --author pr-owner --mode enforce
    { so "$G"; rv "$G" r1; rv "$G" r2; } > "$ev"
    mr enforce-head  0 --base "$B" --head "$G" --evidence "$ev" --author pr-owner --mode enforce
    mr no-author     2 --base "$B" --head "$G" --evidence "$ev" --mode enforce
    mr missing-ev    2 --base "$B" --head "$G" --evidence "$tmp/absent" --author pr-owner
    mr short-head    2 --base "$B" --head "${G:0:12}" --evidence "$ev" --author pr-owner
    mr bad-mode      2 --base "$B" --head "$G" --evidence "$ev" --author pr-owner --mode strict

    # planted mutants: drop each row; its probe must be matched by that row
    # alone, and must turn non-gate once the row is gone.
    local id row probe
    while IFS= read -r row; do
        [ -n "$row" ] || continue
        id="${row%%@*}"; cases=$((cases + 1))
        case "$id" in
            P-WORKFLOW)     probe="$(d .github/workflows/ci.yml)" ;;
            P-CI)           probe="$(d ci/sections.yml)" ;;
            P-CHECK-SCRIPT) probe="$(d scripts/check_x.sh)" ;;
            P-GUARD-TEST)   probe="$(d scripts/tests/x_test.sh)" ;;
            P-ACCEPT-LIST)  probe="$(d docs/x-waiver.yaml)" ;;
            P-RELEASE)      probe="$(d scripts/release/x.sh)" ;;
            P-CONTRACT)     probe="$(d contracts/x-v1.yaml)" ;;
            P-BASELINE)     probe="$(d evidence/ratchets/x-baseline.json)" ;;
            P-GATE-CONFIG)  probe="$(d deny.toml)" ;;
            P-CHECKER-SRC)  probe="$(d crates/aprender-contracts-cli/src/main.rs)" ;;
            L-TEST-IGNORE)  probe="$(d crates/x/src/t.rs '+#[ignore]')" ;;
            L-LINT-LEVEL)   probe="$(d Cargo.toml '+x = "deny"')" ;;
            *) printf 'FAIL mutant %s: row has no probe case (dead row)\n' "$id" >&2; fails=$((fails + 1)); continue ;;
        esac
        got=$(printf '%s\n' "$probe" | classify | tr '\n' ' ')
        if [ "${got% }" != "$id" ]; then printf 'FAIL mutant %s: probe not matched by its row alone [%s]\n' "$id" "$got" >&2; fails=$((fails + 1)); fi
        got=$(printf '%s\n' "$probe" | GATE_RULE_DROP="$id" classify)
        if [ -n "$got" ]; then printf 'FAIL mutant %s: survived (row dropped, probe still gate: %s)\n' "$id" "$got" >&2; fails=$((fails + 1)); fi
    done <<< "$GATE_RULES"

    rm -rf "${tmp:?}"
    if [ "$fails" -ne 0 ]; then printf 'self-test FAILED: %s of %s case(s).\n' "$fails" "$cases" >&2; return 1; fi
    if [ "$cases" -lt 84 ]; then printf 'self-test VACUOUS: %s case(s) ran, fewer than the 84 this table shipped with.\n' "$cases" >&2; return 1; fi
    printf 'self-test OK: %s case(s).\n' "$cases"
}

main() {
    local base="" head="" ev="" author="" mode=report act=check
    if [ $# -eq 0 ]; then act=self; fi
    while [ $# -gt 0 ]; do
        case "$1" in
            --self-test) act=self; shift ;;
            --classify) act=classify; shift ;;
            --base) [ $# -ge 2 ] || usage; base="$2"; shift 2 ;;
            --head) [ $# -ge 2 ] || usage; head="$2"; shift 2 ;;
            --evidence) [ $# -ge 2 ] || usage; ev="$2"; shift 2 ;;
            --author) [ $# -ge 2 ] || usage; author="$2"; shift 2 ;;
            --mode) [ $# -ge 2 ] || usage; mode="$2"; shift 2 ;;
            -h|--help) sed -n '2,/^$/s/^# \{0,1\}//p' "$0"; exit 0 ;;
            *) usage ;;
        esac
    done
    command -v jq >/dev/null 2>&1 || { printf 'NOT_MEASURED: jq not found\n' >&2; exit 2; }
    case "$act" in
        self) self_test; exit $? ;;
        classify) classify; exit 0 ;;
    esac
    case "$mode" in report|enforce) ;; *) usage ;; esac
    [[ "$base" =~ ^[0-9a-f]{40}$ && "$head" =~ ^[0-9a-f]{40}$ ]] || { printf 'usage: --base and --head must be full 40-hex shas\n' >&2; exit 2; }
    if [ -n "$ev" ] && [ ! -r "$ev" ]; then printf 'NOT_MEASURED: evidence file %s unreadable\n' "$ev" >&2; exit 2; fi
    local diff rules j verdict
    diff=$(git -c core.quotePath=true diff --no-renames --no-ext-diff --no-color --src-prefix=a/ --dst-prefix=b/ -U0 "$base...$head") || { printf 'NOT_MEASURED: git diff %s...%s failed\n' "$base" "$head" >&2; exit 2; }
    rules=$(printf '%s\n' "$diff" | classify)
    if [ -n "$rules" ] && [ -z "$author" ]; then printf 'NOT_MEASURED: a gate change needs --author to exclude the PR author from its reviewers\n' >&2; exit 2; fi
    j=$(judge "$head" "$author" "$ev") || { printf 'NOT_MEASURED: evidence %s is not JSON Lines\n' "$ev" >&2; exit 2; }
    verdict=$(decide "$rules" "$j")
    printf 'head=%s gate-change=%s rules=%s %s\n' "$head" "$([ -n "$rules" ] && echo yes || echo no)" "$(printf '%s' "$rules" | tr '\n' ',')" "$j"
    case "$verdict" in
        PASS:*) printf 'PASS: %s\n' "${verdict#PASS:}"; exit 0 ;;
        REFUSE:*)
            if [ "$mode" = enforce ]; then printf 'REFUSE: %s\n' "${verdict#REFUSE:}"; exit 1; fi
            printf 'REPORT: would be RED in enforce mode: %s\n' "${verdict#REFUSE:}"; exit 0 ;;
    esac
}

main "$@"
