#!/usr/bin/env bash
# ci_test_tier.sh — decide which test tier a CI run owes (BSE-17, PMAT-1077).
#
#   quick  pull_request: the touched crates + direct reverse dependents
#          (scripts/gate_touched_crates.sh) PLUS every test target that reads
#          the tree (scripts/tree_reader_tests.txt, derived and checked by
#          scripts/check_tree_reader_tests.sh — drift is ENV, exit 2, never a
#          quick tier over a stale registry). ALSO merge_group and push: see
#          THE QUEUE MIRRORS THE PR below.
#   full   schedule, workflow_dispatch — that is where FULL lives now
#          (coverage-nightly, full-nightly, the pre-publish dogfood; PMAT-1098
#          67-E2 decision D-1) — plus any event whose diff touches a ROOT
#          manifest (rule (ii)) or whose own diff cannot be derived.
#
# THE QUEUE MIRRORS THE PR (PMAT-1098 67-E2, #3084). A rebase is not new
# evidence about a PR's diff: it is the same diff on a new base. This script
# used to answer `full` for every merge_group whose tree had moved and for
# every push to main, so the queue paid a ~1h full workspace run each time main
# moved under a PR — the single largest cost in the merge queue. Now:
#   merge_group, tree moved: the queue ref's first parent is main's tip and its
#     second is the PR head, so HEAD^1..HEAD IS the PR's diff on the new base.
#     Re-derive the PR's own selection from it and run the tier the PR ran.
#   push to main: the same, over the push's own diff (HEAD^1..HEAD for a queue
#     merge; origin/main@{1}..HEAD for a non-merge tip; neither -> full).
#   rule (ii): a diff touching a ROOT Cargo.toml / Cargo.lock /
#     rust-toolchain.toml keeps `full` at the PR, in the queue and on push —
#     dependency bumps are where compile-level integration breaks. The rule is
#     gate_touched_crates.sh's own ("root Cargo.toml/Cargo.lock/
#     rust-toolchain.toml touched -> full workspace check"), cited not copied.
#   rule (i): a selection OVER THE CAP (gate_touched_crates.sh CAP=3, rule text
#     "selection of N crate(s) exceeds cap") is no longer an hour of full
#     workspace tests. It is `quick` over the TOUCHED crates + the tree readers
#     plus one extra output line, check_workspace=1, which ci.yml turns into a
#     single `cargo check --workspace --all-targets --locked` — the
#     compile-level integration of every reverse dependent, in minutes.
#   reuse  merge_group only: HEAD^{tree} equals the PR head's tree AND that
#          head's workspace-test check-run concluded success — the same tree
#          measured twice is the definition of waste. Any doubt -> full.
#
# Output: KEY=VALUE lines — tier, crates (space list), targets (crate:--lib,
# crate:--lib:module, crate:--bins or crate:--test:name; space list),
# check_workspace (1, only under rule (i)), reason, cite (the PR head sha on
# reuse).
# Feature-gated suites (model-tests, setfit, ...) belong to the full tier only;
# the quick tier runs default features. Exit 2 on ENV (unknown event, registry
# drift); 0 otherwise. `--self-test` runs the case table.
#
# `--filterset '<targets>'` (or the same list on stdin) prints the cargo-nextest
# FILTERSET expression that selects exactly those targets — one clause per token,
# UNIONed with `|`. It is the whole of PMAT-1098 (#3084): the quick tier used to
# expand `targets` into a `&&` chain of one `cargo nextest run -p CRATE ...` per
# crate — 26 cargo invocations, 26 compiles of the shared dependency graph, run
# serially. Measured 55 min on a one-file YAML PR (run 34449608126) and killed at
# the 60-minute step timeout under fleet load on #3063 (#3070). One invocation
# over the filterset builds the union ONCE and runs the tests in parallel.
#   crate:--lib        -> (package(crate) & kind(lib))
#   crate:--lib:MOD    -> (package(crate) & kind(lib) & test(/^MOD::/))
#   crate:--bins       -> (package(crate) & kind(bin))    [bin-only crates: no lib target]
#   crate:--test:NAME  -> binary_id(crate::NAME)
# The binary-id forms are nextest's own, verified on cargo-nextest 0.9.132 against
# this workspace (`cargo nextest list --message-format json`): a lib suite's id is
# the bare package name, an integration target's is `package::target`, a bin's is
# `package::bin/name`. `kind(lib)`/`kind(bin)` are equality matches on those kinds,
# which is why the lib and bins tokens do not need to name the binary at all.
#
# PMAT-3120 — the registry names MODULES, not crates. A `crate:--lib` token ran
# the crate's WHOLE lib suite for the sake of one reader: measured 88.08% of all
# quick-tier test-seconds for a single reader FILE. The registry therefore
# carries a third column, the module that holds the reader
# (scripts/tree_reader_tests.txt, derived by scripts/check_tree_reader_tests.sh):
#   crate<TAB>--lib<TAB>module::path   a lib MODULE     -> crate:--lib:module::path
#   crate<TAB>--lib<TAB><root>         the crate root   -> crate:--lib
#   crate<TAB>--lib                    the WHOLE lib (the module was unresolvable;
#                                      the derivation said so, WARN unresolved-include)
# and the module token narrows the clause with `test(/^module::/)` — nextest
# matches a test's full path, so the `^module::` anchor selects that module and
# its descendants and nothing else. A crate that carries a whole-lib row WINS
# over its own module rows: the registry asking for the whole lib is never
# answered with a narrower atom. `targets=` keeps its shape (a space list of
# tokens), so ci.yml's one `--filterset` call needed no change at all.
#
#   --tier-of-record [--tsv F]   print the 80/20 tier of record (default
#          evidence/fleet/test-tier.tsv, spec §6.2/§6.3) as ONE nextest filterset
#          plus tier_of_record_{tests,modules,packages,seconds}. Exit 1 — never a
#          silent full run and never a silent empty set — when the table is
#          missing, headerless, malformed, has zero pr rows, or has a pr row with
#          an empty crate/module.
#   --union-touched              with --tier-of-record and --event pull_request
#          --comparand REF (or --diff-from FILE): the tier-of-record filterset OR
#          the quick tier's touched crates OR its tree-reader targets, so a PR
#          always runs its own crates' full lib tests plus the cross-tree 20%.
#          When the quick tier falls closed the FULL suite runs and no filterset
#          is emitted.
set -euo pipefail

EVENT=""; COMPARAND=""; DIFF_FROM=""; PR_HEAD=""; PR_CONCLUSION=""; REGISTRY="scripts/tree_reader_tests.txt"; ROOT="."
# The sibling scripts and the registry always come from the checkout this script
# runs in (cwd = repo root, in ci.yml and in the case table alike); --repo-root
# re-points only the GIT queries — HEAD, its parents, their trees — which is how
# the case table hands this a throwaway queue-shaped repository.
TREE="."
TSV="evidence/fleet/test-tier.tsv"; TIER_OF_RECORD=0; UNION=0
while [ $# -gt 0 ]; do
    case "$1" in
        --event) EVENT=$2; shift 2 ;;
        --comparand) COMPARAND=$2; shift 2 ;;
        --diff-from) DIFF_FROM=$2; shift 2 ;;
        --pr-head) PR_HEAD=$2; shift 2 ;;
        --pr-head-conclusion) PR_CONCLUSION=$2; shift 2 ;;
        --registry) REGISTRY=$2; shift 2 ;;
        --repo-root) ROOT=$2; shift 2 ;;
        --tsv) TSV=$2; shift 2 ;;
        --tier-of-record) TIER_OF_RECORD=1; shift ;;
        --union-touched) UNION=1; shift ;;
        --self-test) SELF_TEST=1; shift ;;
        # optional operand: `--filterset 'a:--lib b:--test:c'`, or nothing and the
        # list comes from stdin (how ci.yml pipes steps.tier.outputs.targets in).
        --filterset) FILTERSET=1; shift; if [ $# -gt 0 ]; then FS_TARGETS=$1; shift; fi ;;
        *) printf 'usage: %s --event EVENT [--comparand REF] [--diff-from FILE] [--pr-head SHA --pr-head-conclusion C] [--registry FILE] [--repo-root DIR] | --filterset [TARGETS] | --tier-of-record [--tsv FILE] [--union-touched --event pull_request --comparand REF] | --self-test\n' "$0" >&2; exit 2 ;;
    esac
done

targets_from_registry() { # -> space list crate:--lib | crate:--lib:module | crate:--bins | crate:--test:name
    # Two passes in one awk: a crate with a WHOLE-lib row (2 columns, or the
    # module `<root>`) collapses its own module rows into that one token —
    # a narrower atom than the registry asks for would silently stop running
    # the reader the unresolved-include warning was about.
    grep -v '^#' "$1" | grep -v '^[[:space:]]*$' | awk -F"\t" '
        { rows[++n] = $0
          if ($2 == "--lib" && (NF < 3 || $3 == "" || $3 == "<root>")) whole[$1] = 1 }
        END { for (i = 1; i <= n; i++) { split(rows[i], f, "\t")
                  if (f[2] == "--test") t = f[1] ":--test:" f[3]
                  else if (f[2] == "--lib" && !(f[1] in whole)) t = f[1] ":--lib:" f[3]
                  else if (f[2] == "--lib") t = f[1] ":--lib"
                  else t = f[1] ":" f[2]
                  if (!(t in seen)) { seen[t] = 1; printf "%s%s", (m++ ? " " : ""), t } } }'
}

re_escape() { # regex-escape a module path for a nextest test(/^.../) atom; `::` is not special, so it stays literal
    printf '%s' "$1" | sed 's#[].^$*+?()[{}|\\/]#\\&#g'
}

filterset_from_targets() { # <space list of crate:--lib|crate:--bins|crate:--test:NAME> -> nextest -E expression
    local t clause expr=""
    for t in $1; do
        case "$t" in
            *:--lib)    clause="(package(${t%:--lib}) & kind(lib))" ;;
            # PMAT-3120: a module token. nextest matches the test's full path, so
            # `^module::` is that module and its descendants — and nothing else.
            *:--lib:*)  clause="(package(${t%%:--lib:*}) & kind(lib) & test(/^$(re_escape "${t#*:--lib:}")::/))" ;;
            *:--bins)   clause="(package(${t%:--bins}) & kind(bin))" ;;
            *:--test:*) clause="binary_id(${t%%:--test:*}::${t#*:--test:})" ;;
            # Never a silent drop: a token this does not understand is a
            # tree-reader target that would stop running while the step stayed
            # green — exactly the darkness scripts/tree_reader_tests.txt exists
            # to end. ENV, exit 2, name the token.
            *) printf 'ENV: unrecognised target token "%s" — expected crate:--lib, crate:--bins or crate:--test:NAME\n' "$t" >&2; return 2 ;;
        esac
        expr="${expr:+$expr | }$clause"
    done
    # An empty -E is not "select nothing", it is `cargo nextest run --workspace`
    # with no filter at all. Refuse rather than run the full tier by accident.
    [ -n "$expr" ] || { printf 'ENV: no targets given — refusing to emit an empty filterset (nextest would then select the WHOLE workspace)\n' >&2; return 2; }
    printf '%s\n' "$expr"
}

# The tier a given touched-path set owes. ONE function for all three events
# that carry a diff (pull_request, merge_group on a moved main, push to main):
# the queue and main must not answer a different question about the same diff
# than the PR did, and one code path is how that stays true.
selection() { # $1 = reason prefix (names the event and how the diff was derived), $2 = diff file ("" -> gate_touched_crates' own git diff)
    local prefix=$1 diff=$2 chk sel crates rule touched nreg
    if ! chk=$(bash "$TREE/scripts/check_tree_reader_tests.sh" 2>&1); then printf 'ENV: %s\n' "$chk" >&2; return 2; fi
    sel=$(bash "$TREE/scripts/gate_touched_crates.sh" --print-selection ${COMPARAND:+--comparand "$COMPARAND"} ${diff:+--diff-from "$diff"} 2>/dev/null | tail -1)
    crates=$(printf '%s' "$sel" | sed -n 's/^selection=[a-z]* crates=\(.*\) rule=.*$/\1/p'); rule=${sel#*rule=}
    # Tokens, not lines: since PMAT-3120 a crate's module rows can collapse into
    # one whole-lib token, so a line count would over-report what actually runs.
    nreg=$(targets_from_registry "$TREE/$REGISTRY" | wc -w | tr -d ' ')
    case "$sel" in
        selection=full*)
            # gate_touched_crates.sh prints `selection=full` for BOTH fail-closed
            # rules and distinguishes them only in the rule text, so this reads
            # the rule rather than duplicating either test (that script is the
            # owner of both; PMAT-1098 67-E2 rule (i)/(ii)).
            case "$rule" in
                *"exceeds cap"*)
                    # Rule (i): over the cap is not a reason to spend an hour. The
                    # TOUCHED crates run their own tests (the reverse dependents
                    # are what blew the cap, and a test-level run of all of them is
                    # the expensive part), every tree reader still runs, and
                    # check_workspace=1 buys the compile-level integration of the
                    # whole workspace in one `cargo check` step.
                    touched=$(bash "$TREE/scripts/gate_touched_crates.sh" --dry-run ${diff:+--diff-from "$diff"} 2>/dev/null | sed -n 's/^gate_touched_crates: touched crate(s): //p')
                    if [ "$touched" = "(none)" ]; then touched=""; fi
                    printf 'tier=quick\ncrates=%s\ntargets=%s\ncheck_workspace=1\nreason=%s: %s -- rule (i): the touched crate(s) run their tests and ONE cargo check --workspace --all-targets covers every reverse dependent at compile level; plus %s tree-reader target(s) from %s\n' \
                        "$touched" "$(targets_from_registry "$TREE/$REGISTRY")" "$prefix" "$rule" "$nreg" "$REGISTRY" ;;
                # Rule (ii): a ROOT manifest stays full at the PR, in the queue and
                # on push alike — a dependency bump is exactly where compile-level
                # integration breaks, and its blast radius is the whole workspace.
                *) printf 'tier=full\nreason=%s: %s\n' "$prefix" "$rule" ;;
            esac ;;
        selection=quick*|selection=none*) printf 'tier=quick\ncrates=%s\ntargets=%s\nreason=%s: %s; plus %s tree-reader target(s) from %s\n' "$crates" "$(targets_from_registry "$TREE/$REGISTRY")" "$prefix" "$rule" "$nreg" "$REGISTRY" ;;
        *) printf 'ENV: gate_touched_crates --print-selection gave "%s"\n' "$sel" >&2; return 2 ;;
    esac
}

# The diff between a base revision and HEAD, as paths, into a file. Two-dot on
# purpose: for a merge or a squash the base IS an ancestor of HEAD, so
# base..HEAD and base...HEAD name the same tree comparison, and the two-dot form
# needs no merge-base — which a shallow CI checkout usually cannot compute.
diff_into() { # $1 = out file, $2 = base rev
    git -C "$ROOT" diff --name-only "$2" HEAD > "$1" 2>/dev/null
}

decide() {
    local df rc=0
    case "$EVENT" in
        schedule|workflow_dispatch)
            printf 'tier=full\nreason=%s event: the whole workspace, every feature-gated suite -- FULL lives here (coverage-nightly, full-nightly, the pre-publish dogfood; 67-E2 decision D-1)\n' "$EVENT" ;;
        push)
            # A push to main is the merge queue landing ONE PR, so the push's own
            # diff is HEAD^1..HEAD. It used to be a flat `full`, which is how every
            # landing paid an hour for work the PR and the queue had both already
            # measured.
            local base how
            if git -C "$ROOT" rev-parse -q --verify 'HEAD^2' >/dev/null 2>&1; then
                base='HEAD^1'; how="the merge commit's own diff, HEAD^1..HEAD"
            elif git -C "$ROOT" rev-parse -q --verify 'origin/main@{1}' >/dev/null 2>&1; then
                base='origin/main@{1}'; how='a non-merge tip diffed against the previous main, origin/main@{1}..HEAD'
            else
                printf 'tier=full\nreason=push: neither a merge commit (no HEAD^2) nor a previous origin/main in the reflog -- the pushed diff cannot be derived, so this falls closed to full\n'; return 0
            fi
            df=$(mktemp "${TMPDIR:-/tmp}/ci-tier-diff.XXXXXX")
            if ! diff_into "$df" "$base"; then rm -f "$df"; printf 'tier=full\nreason=push: git diff %s..HEAD failed -- the pushed diff cannot be derived, so this falls closed to full\n' "$base"; return 0; fi
            selection "push: $how" "$df" || rc=$?
            rm -f "$df"; return $rc ;;
        merge_group)
            if [ -z "$PR_HEAD" ]; then printf 'tier=full\nreason=merge_group without a PR head to compare against\n'; return 0; fi
            local ht pt
            ht=$(git -C "$ROOT" rev-parse 'HEAD^{tree}' 2>/dev/null || true)
            pt=$(git -C "$ROOT" rev-parse "${PR_HEAD}^{tree}" 2>/dev/null || true)
            if [ -z "$ht" ] || [ -z "$pt" ]; then printf 'tier=full\nreason=merge_group: a tree could not be resolved (HEAD=%s pr-head=%s)\n' "${ht:-?}" "${pt:-?}"; return 0; fi
            if [ "$ht" != "$pt" ]; then
                # THE QUEUE MIRRORS THE PR: main moved, the PR's diff did not.
                if ! git -C "$ROOT" rev-parse -q --verify 'HEAD^2' >/dev/null 2>&1; then
                    printf 'tier=full\nreason=merge_group: main moved under the PR (queue tree %s != PR head tree %s) and the queue ref is not a merge commit, so the PR diff cannot be re-derived -- fail closed\n' "${ht:0:9}" "${pt:0:9}"; return 0
                fi
                df=$(mktemp "${TMPDIR:-/tmp}/ci-tier-diff.XXXXXX")
                if ! diff_into "$df" 'HEAD^1'; then rm -f "$df"; printf 'tier=full\nreason=merge_group: main moved under the PR but git diff HEAD^1..HEAD failed, so the PR diff cannot be re-derived -- fail closed\n'; return 0; fi
                selection "merge_group: main moved under the PR (queue tree ${ht:0:9} != PR head tree ${pt:0:9}); the PR's own selection re-derived on the queue ref" "$df" || rc=$?
                rm -f "$df"; return $rc
            fi
            if [ "$PR_CONCLUSION" != "success" ]; then printf 'tier=full\nreason=merge_group: same tree but the PR head'"'"'s workspace-test concluded %s, not success\n' "${PR_CONCLUSION:-unknown}"; return 0; fi
            printf 'tier=reuse\ncite=%s\nreason=merge_group: HEAD^{tree} %s equals PR head %s^{tree}, whose workspace-test succeeded — the same tree measured twice\n' "$PR_HEAD" "${ht:0:9}" "${PR_HEAD:0:9}" ;;
        pull_request)
            selection "pull_request" "$DIFF_FROM" || return $? ;;
        *) printf 'ENV: unknown event "%s" — refusing to guess a tier\n' "$EVENT" >&2; return 2 ;;
    esac
}

tier_of_record() { # -> filterset= + tier_of_record_* KEY=VALUE lines; 1 on an unusable table
    local out rc=0
    out=$(python3 "$TREE/scripts/lib/test_tier.py" filterset --tsv "$TSV" 2>&1) || rc=$?
    if [ "$rc" != 0 ]; then printf 'DATA: %s\n' "$out" >&2; return 1; fi
    printf '%s\n' "$out"
}

union_touched() { # the tier of record OR the touched crates OR the tree-reader targets; 1 on an unusable table, 2 on ENV
    local tor dec crates targets expr pkgs trfs
    tor=$(tier_of_record) || return 1
    dec=$(decide) || return $?
    printf '%s\n' "$dec"
    if printf '%s\n' "$dec" | grep -qx 'tier=full'; then
        # Fail OPEN, towards more tests: the full suite runs, so a filterset would
        # only be a narrowing nobody asked for.
        printf 'union_touched_crates=\n'
        printf '%s\n' "$tor" | grep -v '^filterset='
        return 0
    fi
    crates=$(printf '%s\n' "$dec" | sed -n 's/^crates=//p')
    targets=$(printf '%s\n' "$dec" | sed -n 's/^targets=//p')
    expr=$(printf '%s\n' "$tor" | sed -n 's/^filterset=//p')
    # The quick tier's own targets go through the SAME token->clause translation
    # ci.yml uses, so the union can never disagree with the step about what a
    # token means. Exactly one filterset= line is printed, or a consumer would
    # have to guess which of two it owed.
    trfs=""; if [ -n "$targets" ]; then trfs=$(filterset_from_targets "$targets") || return 2; fi
    pkgs=$(printf '%s' "$crates" | awk '{ for (i = 1; i <= NF; i++) printf "%spackage(=%s)", (i > 1 ? " | " : ""), $i }')
    case "$crates" in
        "") printf 'filterset=%s' "$expr" ;;
        *)  printf 'filterset=%s | (%s)' "$expr" "$pkgs" ;;
    esac
    if [ -n "$trfs" ]; then printf ' | %s\n' "$trfs"; else printf '\n'; fi
    printf 'union_touched_crates=%s\n' "$crates"
    printf '%s\n' "$tor" | grep -v '^filterset='
}

self_test() {
    local td n=0 red=0 out rc leaf
    td=$(mktemp -d "${TMPDIR:-/tmp}/ci-tier.XXXXXX"); trap 'rm -rf "${td:?}"' RETURN
    # One decision, many rows. Each full decision re-runs check_tree_reader_tests.sh
    # (~23s, it re-derives the registry from the sources), so a fixture is DECIDED
    # once into a file and every assertion about it replays that file with its exit
    # code. Same output, same rc — the rows are not weakened, only the wall clock.
    cap() { local name=$1 r=0; shift; "$@" > "$td/$name.out" 2>&1 || r=$?; printf '%s' "$r" > "$td/$name.rc"; }
    replay() { printf 'cat "%s/%s.out"; exit "$(cat "%s/%s.rc")"' "$td" "$1" "$td" "$1"; }
    row() { local want=$1 label=$2 pat=$3; shift 3; n=$((n + 1)); rc=0; out=$("$@" 2>&1) || rc=$?
        if [ "$rc" = "$want" ] && printf '%s\n' "$out" | grep -qE -- "$pat"; then printf 'ok    row %-2s rc=%s  %s\n' "$n" "$rc" "$label"
        else printf 'FAIL  row %-2s rc=%s (wanted %s, must match /%s/)  %s\n' "$n" "$rc" "$want" "$pat" "$label"; printf '%s\n' "$out" | sed 's/^/        /'; red=1; fi; }
    T=$0
    # File-shaped assertions. `row` can call a shell FUNCTION, which keeps the
    # quoting out of `bash -c` and gives an exact-text verdict.
    contains_all() { # FILE PAT... -> ALL-PRESENT, or the first missing pattern and rc 1
        local f=$1 pat
        shift
        for pat in "$@"; do
            if ! grep -qF -- "$pat" "$f"; then printf 'MISSING %s\n' "$pat"; return 1; fi
        done
        printf 'ALL-PRESENT\n'
    }
    lacks() { if grep -qF -- "$2" "$1"; then printf 'STILL-PRESENT %s\n' "$2"; return 1; fi; printf 'ABSENT\n'; }
    full_ok() { if ! grep -qx 'tier=full' "$1"; then printf 'NOT-FULL\n'; return 1; fi
        if grep -q '^filterset=' "$1"; then printf 'FULL-WITH-A-FILTERSET\n'; return 1; fi; printf 'FULL-NO-FILTERSET\n'; }
    fs_of() { sed -n 's/^targets=//p' "$1" | bash "$T" --filterset; } # a captured decision's targets -> the expression ci.yml runs
    clauses_vs_tokens() { # FILE -> EQUAL | DIFFER: one clause per TOKEN, nothing dropped, nothing invented
        local tg ntok ncl
        tg=$(sed -n 's/^targets=//p' "$1"); ntok=$(printf '%s' "$tg" | wc -w | tr -d ' ')
        ncl=$(printf '%s' "$tg" | bash "$T" --filterset | tr '|' '\n' | wc -l | tr -d ' ')
        if [ "$ntok" = "$ncl" ]; then printf 'EQUAL\n'; else printf 'DIFFER tokens=%s clauses=%s\n' "$ntok" "$ncl"; fi
    }
    all_translated() { # FILE -> NONE-LEFT: every registry token became a clause
        local e
        e=$(fs_of "$1")
        if [ -z "$e" ]; then printf 'EMPTY\n'; elif printf '%s' "$e" | grep -q ':--'; then printf 'LEFTOVER\n'; else printf 'NONE-LEFT\n'; fi
    }
    row 0 "schedule -> full (FULL still lives on the nightlies — decision D-1)" '^tier=full' bash "$T" --event schedule
    row 0 "workflow_dispatch -> full" '^tier=full' bash "$T" --event workflow_dispatch
    row 2 "unknown event -> ENV (exit 2), never a guess" 'refusing to guess' bash "$T" --event release
    printf 'scripts/foo.sh\n' > "$td/d-scripts.txt"
    cap pr-scripts bash "$T" --event pull_request --diff-from "$td/d-scripts.txt"
    row 0 "pull_request, scripts-only diff -> quick with NO crates" '^crates=$' bash -c "$(replay pr-scripts)"
    row 0 "  ...and the tree-reader targets are in it (readme_contract, the reader that bit #3039)" 'aprender-core:--test:readme_contract' bash -c "$(replay pr-scripts)"
    row 0 "  ...including the lib module whose unit test reads a baseline (aprender-contracts)" 'aprender-contracts:--lib:' bash -c "$(replay pr-scripts)"
    printf 'Cargo.toml\n' > "$td/d-root.txt"
    row 0 "pull_request, root Cargo.toml touched -> full (fail closed)" '^tier=full' bash "$T" --event pull_request --diff-from "$td/d-root.txt"
    printf 'crates/aprender-core/src/lib.rs\n' > "$td/d-core.txt"
    cap pr-cap bash "$T" --event pull_request --diff-from "$td/d-core.txt"
    row 0 "pull_request over the cap (aprender-core) -> quick, NOT an hour of full workspace tests" '^tier=quick' bash -c "$(replay pr-cap)"
    row 0 "  ...with check_workspace=1: ONE cargo check --workspace covers every reverse dependent (rule (i))" '^check_workspace=1$' bash -c "$(replay pr-cap)"
    row 0 "  ...and the touched crate itself still runs its tests" '^crates=.*aprender-core' bash -c "$(replay pr-cap)"
    row 0 "pull_request WITHIN the cap emits NO check_workspace line (the workspace check is not free)" '^NO-CHECK-WORKSPACE$' bash -c "if bash '$T' --event pull_request --diff-from '$td/d-leaf.txt' | grep -q '^check_workspace='; then echo HAS-CHECK-WORKSPACE; else echo NO-CHECK-WORKSPACE; fi"
    leaf=$(cargo metadata --no-deps --format-version 1 2>/dev/null | jq -r '[.packages[]|select(.manifest_path|test("/crates/"))] | (map(.name) - (map(.dependencies[]?.name)|unique)) | sort | .[0]')
    printf 'crates/%s/src/lib.rs\n' "$leaf" > "$td/d-leaf.txt"
    row 0 "pull_request, a leaf crate ($leaf) touched -> quick with that crate" "^crates=.*$leaf" bash "$T" --event pull_request --diff-from "$td/d-leaf.txt"
    # registry drift is ENV for the quick tier
    cp scripts/tree_reader_tests.txt "$td/reg.bak"; printf 'zeta\t--lib\n' >> scripts/tree_reader_tests.txt
    row 2 "pull_request with a drifted registry -> ENV (exit 2): no quick tier over a stale list" 'drifted' bash "$T" --event pull_request --diff-from "$td/d-scripts.txt"
    cp "$td/reg.bak" scripts/tree_reader_tests.txt
    # merge_group rows on a throwaway repo: same tree (empty commit) vs different tree
    git init -q "$td/repo"; ( cd "$td/repo" && git -c user.name=t -c user.email=t@t commit -q --allow-empty -m base && printf 'a\n' > f && git add f && git -c user.name=t -c user.email=t@t commit -q -m one && git -c user.name=t -c user.email=t@t commit -q --allow-empty -m "queue merge, same tree" )
    same=$(git -C "$td/repo" rev-parse HEAD~1)
    row 0 "merge_group, same tree + PR head workspace-test success -> reuse, citing the head" "^cite=$same" bash "$T" --event merge_group --repo-root "$td/repo" --pr-head "$same" --pr-head-conclusion success
    row 0 "merge_group, same tree but PR head conclusion failure -> full" 'concluded failure' bash "$T" --event merge_group --repo-root "$td/repo" --pr-head "$same" --pr-head-conclusion failure
    diff1=$(git -C "$td/repo" rev-parse HEAD~2)
    row 0 "merge_group, different tree on a ref that is NOT a merge -> full (the PR diff cannot be re-derived)" 'not a merge commit' bash "$T" --event merge_group --repo-root "$td/repo" --pr-head "$diff1" --pr-head-conclusion success
    row 0 "merge_group without a PR head -> full" 'without a PR head' bash "$T" --event merge_group --repo-root "$td/repo"
    # PMAT-1098 67-E2 (#3084): THE QUEUE MIRRORS THE PR. A rebase is not new
    # evidence about the PR's diff, it is the same diff on a new base — so a
    # merge_group whose tree moved re-derives the PR's OWN selection from the
    # queue ref (first parent = main's tip, second = the PR head, so HEAD^1..HEAD
    # IS the PR's diff) and runs the tier the PR ran. Same for a push to main.
    # Both used to cost the full hour every time main moved.
    mkqueue() { # $1 dir, $2 touched path -> HEAD = merge(main tip, PR head), a queue ref's shape
        local d=$1 f=$2
        git init -q -b main "$d"
        ( cd "$d" \
          && export GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@t GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@t \
          && export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.hooksPath GIT_CONFIG_VALUE_0=/dev/null \
          && git commit -q --allow-empty -m base \
          && git branch pr \
          && printf 'moved\n' > main-moved.txt && git add -A && git commit -q -m "main moved under the PR" \
          && git checkout -q pr && mkdir -p "$(dirname "$f")" && printf 'x\n' > "$f" && git add -A && git commit -q -m "the PR" \
          && git checkout -q main && git merge -q --no-ff -m "queue merge" pr )
    }
    mkqueue "$td/q-leaf" "crates/$leaf/src/lib.rs"
    mkqueue "$td/q-root" "Cargo.toml"
    mkqueue "$td/q-cap"  "crates/aprender-core/src/lib.rs"
    qh() { git -C "$1" rev-parse pr; }
    cap mg-leaf bash "$T" --event merge_group --repo-root "$td/q-leaf" --pr-head "$(qh "$td/q-leaf")" --pr-head-conclusion success
    row 0 "merge_group, different tree, the PR ran quick -> quick, not full (main moved is not new evidence)" '^tier=quick' bash -c "$(replay mg-leaf)"
    row 0 "  ...with the SAME crates the PR ran ($leaf), re-derived from HEAD^1..HEAD on the queue ref" "^crates=.*$leaf" bash -c "$(replay mg-leaf)"
    row 0 "  ...and the reason names the re-derivation, not a guess" 're-derived on the queue ref' bash -c "$(replay mg-leaf)"
    row 0 "  ...and the tree-reader targets ride along (readme_contract, the reader that bit #3039)" 'aprender-core:--test:readme_contract' bash -c "$(replay mg-leaf)"
    cap mg-root bash "$T" --event merge_group --repo-root "$td/q-root" --pr-head "$(qh "$td/q-root")" --pr-head-conclusion success
    row 0 "merge_group, the PR touched a ROOT manifest -> full at the queue too (rule (ii))" '^tier=full' bash -c "$(replay mg-root)"
    row 0 "  ...citing the root-manifest rule, so the escalation is auditable" 'root Cargo.toml' bash -c "$(replay mg-root)"
    cap mg-cap bash "$T" --event merge_group --repo-root "$td/q-cap" --pr-head "$(qh "$td/q-cap")" --pr-head-conclusion success
    row 0 "merge_group over the cap -> quick + check_workspace=1 (rule (i)), not full" '^check_workspace=1$' bash -c "$(replay mg-cap)"
    cap push-leaf bash "$T" --event push --repo-root "$td/q-leaf"
    row 0 "push (a queue merge of one PR) -> quick with the PUSH own diff, not the whole workspace" "^crates=.*$leaf" bash -c "$(replay push-leaf)"
    row 0 "  ...and the reason names HEAD^1..HEAD, so the diff is auditable" 'HEAD\^1' bash -c "$(replay push-leaf)"
    cap push-root bash "$T" --event push --repo-root "$td/q-root"
    row 0 "push touching a ROOT manifest -> full on main too (rule (ii))" '^tier=full' bash -c "$(replay push-root)"
    cap push-cap bash "$T" --event push --repo-root "$td/q-cap"
    row 0 "push over the cap -> quick + check_workspace=1 (rule (i))" '^check_workspace=1$' bash -c "$(replay push-cap)"
    row 0 "push whose diff cannot be derived (root commit, no reflog) -> full (fail closed)" '^tier=full' bash -c "d=$td/p-root; git init -q -b main \"\$d\"; git -C \"\$d\" -c user.name=t -c user.email=t@t commit -q --allow-empty -m only; bash '$T' --event push --repo-root \"\$d\""
    # MUTANT: a copy whose queue branch ignores the re-derived diff and always
    # says full is exactly today's behaviour — the rows above must lose the crates.
    sed 's|^\( *\)selection "merge_group|\1printf "tier=full\\nreason=MUTANT\\n"; return 0; selection "merge_group|' "$T" > "$td/mutant-queue.sh"
    row 0 "mutant queue branch (always full on a moved main) loses the crates — the rows discriminate" 'MUTANT-FULL' bash -c "if bash '$td/mutant-queue.sh' --event merge_group --repo-root '$td/q-leaf' --pr-head '$(qh "$td/q-leaf")' --pr-head-conclusion success | grep -q '^tier=quick'; then echo MUTANT-QUICK; else echo MUTANT-FULL; fi"
    # --filterset (PMAT-1098, #3084): the quick tier's 26-way `&&` chain of
    # per-crate cargo invocations became ONE build graph + one nextest run over a
    # filterset. These rows pin the token->clause translation in BOTH polarities:
    # every recognised token becomes exactly one clause, and anything else is ENV
    # (exit 2) — a token silently dropped here is a tree-reader target that stops
    # running while the step stays green, which is the failure mode this whole
    # registry exists to prevent.
    row 0 "--filterset: a lib token -> (package & kind(lib))" '^\(package\(apr-cli\) & kind\(lib\)\)$' bash "$T" --filterset 'apr-cli:--lib'
    row 0 "--filterset: a test token -> binary_id(crate::name), nextest's id for an integration target" '^binary_id\(aprender-core::readme_contract\)$' bash "$T" --filterset 'aprender-core:--test:readme_contract'
    row 0 "--filterset: a bins token -> (package & kind(bin)); a bin-only crate has NO lib target" '^\(package\(aprender-compute-xtask\) & kind\(bin\)\)$' bash "$T" --filterset 'aprender-compute-xtask:--bins'
    row 0 "--filterset: two tokens are UNIONed with |" 'kind\(lib\)\) \| binary_id\(' bash "$T" --filterset 'apr-cli:--lib aprender-core:--test:readme_contract'
    row 2 "--filterset: an unknown token -> ENV (exit 2), never a silently dropped target" 'unrecognised target token' bash "$T" --filterset 'apr-cli:--doc'
    row 2 "--filterset: no targets -> ENV (exit 2); an empty -E would select the WHOLE workspace" 'empty filterset' bash "$T" --filterset ''
    row 0 "--filterset reads stdin and translates EVERY registry token (no ':--' survives)" '^NONE-LEFT$' all_translated "$td/pr-scripts.out"
    # Per TOKEN, not per registry line: since PMAT-3120 a crate's module rows can
    # collapse into one whole-lib token, so a line count would not be the thing
    # that runs.
    row 0 "--filterset over the registry: one clause per TOKEN (nothing dropped, nothing invented)" '^EQUAL$' clauses_vs_tokens "$td/pr-scripts.out"
    # MUTANT: a copy that drops the tree-reader targets from the quick tier must lose readme_contract — the falsifier discriminates
    sed 's/targets=%s\\n/targets=\\n/; s/"\$(targets_from_registry "\$TREE\/\$REGISTRY")" //' "$T" > "$td/mutant.sh"
    row 0 "mutant without tree-reader targets loses readme_contract (proves the inclusion is load-bearing)" 'MUTANT-LOST' bash -c "if bash '$td/mutant.sh' --event pull_request --diff-from '$td/d-scripts.txt' | grep -q readme_contract; then echo MUTANT-KEPT; else echo MUTANT-LOST; fi"
    # --- PMAT-3119: the tier of record as a filterset. Hermetic: committed fixture + temp copies, no cargo.
    local FX GOLD tor_expr realn
    FX="tests/fixtures/test_tier/tier-small.tsv"; GOLD="tests/fixtures/test_tier/tier-small.filterset.txt"
    tor_expr=$(sed 's/^filterset=//' "$GOLD")
    bash "$T" --tier-of-record --tsv "$FX" > "$td/fx.out" 2>&1 || true
    grep '^filterset=' "$td/fx.out" > "$td/fx-fs.txt" || true
    row 0 "fixture TSV -> the committed GOLDEN filterset, textually (2 lib modules grouped, 1 in another crate, 1 binary)" '^$' diff "$GOLD" "$td/fx-fs.txt"
    row 0 "  ...tier_of_record_tests=8 (the nightly/full rows' 11 tests excluded)" '^tier_of_record_tests=8$' cat "$td/fx.out"
    row 0 "  ...tier_of_record_modules=4" '^tier_of_record_modules=4$' cat "$td/fx.out"
    row 0 "  ...tier_of_record_packages=3" '^tier_of_record_packages=3$' cat "$td/fx.out"
    row 0 "  ...tier_of_record_seconds=10.75 at 2dp (not the table's 152.75)" '^tier_of_record_seconds=10\.75$' cat "$td/fx.out"
    # MUTATION: drop one pr row -> a DIFFERENT filterset, missing exactly that module
    grep -v '^crateA::guard' "$FX" > "$td/mut.tsv"
    bash "$T" --tier-of-record --tsv "$td/mut.tsv" > "$td/mut.out" 2>&1 || true
    grep '^filterset=' "$td/mut.out" > "$td/mut-fs.txt" || true
    row 1 "MUTATION: one pr row deleted -> the filterset DIFFERS from the golden (every row is load-bearing)" '^<' diff "$GOLD" "$td/mut-fs.txt"
    row 0 "  ...the deleted module is absent from it" '^ABSENT$' lacks "$td/mut-fs.txt" "guard"
    row 0 "  ...and the other three atoms survive" '^ALL-PRESENT$' contains_all "$td/mut-fs.txt" "dense" "test(/^(light)::/)" "binary(=readme_contract)"
    row 0 "  ...with modules=3" '^tier_of_record_modules=3$' cat "$td/mut.out"
    row 0 "  ...and tests=6 (the deleted row's 2 tests are gone from the budget too)" '^tier_of_record_tests=6$' cat "$td/mut.out"
    # exit-1 contract: never a silent full run, never a silent empty filterset
    row 1 "missing TSV -> exit 1" 'not readable' bash "$T" --tier-of-record --tsv "$td/absent.tsv"
    head -1 "$FX" > "$td/hdr.tsv"
    row 1 "header-only TSV -> exit 1 (zero pr rows)" 'zero tier=pr rows' bash "$T" --tier-of-record --tsv "$td/hdr.tsv"
    awk -F"\t" 'NR == 1 || $8 != "pr"' "$FX" > "$td/nopr.tsv"
    row 1 "rows but no pr row -> exit 1 (an empty filterset would run nothing)" 'zero tier=pr rows' bash "$T" --tier-of-record --tsv "$td/nopr.tsv"
    awk -F"\t" 'NR > 1' "$FX" > "$td/nohdr.tsv"
    row 1 "no header row -> exit 1" 'header is not' bash "$T" --tier-of-record --tsv "$td/nohdr.tsv"
    { head -1 "$FX"; printf 'crateZ::m\tm\t\t1\t1.00\t1\t0\tpr\tlib\n'; } > "$td/nocrate.tsv"
    row 1 "a pr row with an empty crate -> exit 1 (no half-formed atom)" 'empty crate/module' bash "$T" --tier-of-record --tsv "$td/nocrate.tsv"
    { head -1 "$FX"; printf 'crateZ::\t\tcrateZ\t1\t1.00\t1\t0\tpr\tlib\n'; } > "$td/nomodule.tsv"
    row 1 "a pr row with an empty module -> exit 1" 'empty crate/module' bash "$T" --tier-of-record --tsv "$td/nomodule.tsv"
    # the real table of record: rc 0 and a module count DERIVED from the file
    realn=$(awk -F"\t" 'NR > 1 && $8 == "pr"' evidence/fleet/test-tier.tsv | wc -l | tr -d ' ')
    row 0 "the real table of record -> rc 0 and modules=$realn, derived from the TSV" "^tier_of_record_modules=$realn\$" bash "$T" --tier-of-record
    # --union-touched: misuse is exit 2, never a quiet tier-of-record-only run
    row 2 "--union-touched without --tier-of-record -> exit 2" 'requires --tier-of-record' bash "$T" --union-touched --event pull_request --diff-from "$td/d-scripts.txt"
    row 2 "--union-touched on a push event -> exit 2" 'requires --event pull_request' bash "$T" --tier-of-record --union-touched --event push
    row 2 "--union-touched without a comparand/diff -> exit 2" 'requires --comparand' bash "$T" --tier-of-record --union-touched --event pull_request
    bash "$T" --tier-of-record --tsv "$FX" --union-touched --event pull_request --diff-from "$td/d-leaf.txt" > "$td/u-leaf.out" 2>&1 || true
    row 0 "--union-touched, leaf crate touched -> the tier of record OR that crate's package() atom" '^ALL-PRESENT$' contains_all "$td/u-leaf.out" "filterset=$tor_expr | (package(=$leaf)" "union_touched_crates=$leaf"
    bash "$T" --tier-of-record --tsv "$FX" --union-touched --event pull_request --diff-from "$td/d-root.txt" > "$td/u-root.out" 2>&1 || true
    row 0 "--union-touched when the quick tier falls closed -> tier=full and NO filterset (fail open to more tests)" '^FULL-NO-FILTERSET$' full_ok "$td/u-root.out"
    row 0 "--union-touched folds the tree-reader targets through the SAME token->clause translation, into ONE filterset= line" '^1$' \
        bash -c "grep -c '^filterset=' '$td/u-leaf.out'"
    row 0 "  ...so the union carries a tree-reader MODULE atom too, not just the touched packages" 'kind\(lib\) & test\(/\^' cat "$td/u-leaf.out"
    # --- PMAT-3120: the registry's 3-column lib rows. Hermetic: a committed
    # hand-written registry (one of each column shape) + two goldens. The token
    # grammar is the whole of this ticket, so both goldens are asserted TEXTUALLY.
    local RFX
    RFX="tests/fixtures/tree_reader/registry-small.txt"
    cap r3 bash "$T" --event pull_request --diff-from "$td/d-scripts.txt" --registry "$RFX"
    grep '^targets=' "$td/r3.out" > "$td/r3-tg.txt" || true
    fs_of "$td/r3.out" > "$td/r3-fs.txt" 2>&1 || true
    row 0 "3-column registry -> the committed GOLDEN targets=: a module row becomes crate:--lib:module, <root> and an unresolvable row become crate:--lib" \
        '^$' diff "tests/fixtures/tree_reader/registry-small.targets.txt" "$td/r3-tg.txt"
    row 0 "  ...and those tokens translate to the committed GOLDEN filterset — ONE clause per token, UNIONed (this is what ci.yml runs)" \
        '^$' diff "tests/fixtures/tree_reader/registry-small.filterset.txt" "$td/r3-fs.txt"
    row 0 "  ...a module token -> kind(lib) narrowed by an ANCHORED test(/^M::/) atom" \
        'package\(crateB\) & kind\(lib\) & test\(/\^deep::leaf::/\)' cat "$td/r3-fs.txt"
    row 0 "  ...a --test row is still binary_id(crate::name), untouched by PMAT-3120" 'binary_id\(crateC::it\)' cat "$td/r3-fs.txt"
    row 0 "  ...a --bins row is still kind(bin), never kind(lib)" 'package\(crateD\) & kind\(bin\)' cat "$td/r3-fs.txt"
    row 0 "  ...crateE carries BOTH a whole-lib row and a module row: the whole lib WINS (never a narrower atom than the registry asks for)" \
        'package\(crateE\) & kind\(lib\)\)$' cat "$td/r3-fs.txt"
    row 0 "  ...and crateE's module atom is NOT emitted alongside it" '^ABSENT$' lacks "$td/r3-fs.txt" "commands::x"
    # MUTATION: collapse every lib row to the whole crate (the pre-PMAT-3120 shape)
    # -> the filterset loses the module atoms. This row proves the 3rd column is load-bearing.
    awk -F"\t" -v OFS="\t" '/^#/ { print; next } $2 == "--lib" { print $1, $2; next } { print }' "$RFX" | LC_ALL=C sort -u > "$td/flat-reg.txt"
    cap r2 bash "$T" --event pull_request --diff-from "$td/d-scripts.txt" --registry "$td/flat-reg.txt"
    fs_of "$td/r2.out" > "$td/r2-fs.txt" 2>&1 || true
    row 1 "MUTATION: the module column dropped from every lib row -> the filterset DIFFERS from the golden" '^[<>]' \
        diff "tests/fixtures/tree_reader/registry-small.filterset.txt" "$td/r2-fs.txt"
    row 0 "  ...crateB becomes the WHOLE lib (the 88%-of-test-seconds shape this ticket removes)" '^ABSENT$' lacks "$td/r2-fs.txt" "deep::leaf"
    # the REAL registry, through the real decision captured above
    fs_of "$td/pr-scripts.out" > "$td/real-fs.txt" 2>&1 || true
    row 0 "the real registry -> targets= names the MODULE that holds the reader, not the crate (aprender-contracts lint::strict_test_binding)" \
        '^targets=.*aprender-contracts:--lib:lint::strict_test_binding' cat "$td/pr-scripts.out"
    row 0 "  ...whose clause is anchored at ^ inside the lib, so sibling modules do NOT run" '^ALL-PRESENT$' \
        contains_all "$td/real-fs.txt" "(package(aprender-contracts) & kind(lib) & test(/^lint::strict_test_binding::/))"
    row 0 "  ...while apr-cli, whose registry carries an UNRESOLVED whole-lib row, still runs its WHOLE lib (fail open)" '^ALL-PRESENT$' \
        contains_all "$td/real-fs.txt" "(package(apr-cli) & kind(lib))"
    row 0 "  ...with no narrower apr-cli module atom beside it" '^ABSENT$' lacks "$td/real-fs.txt" "package(apr-cli) & kind(lib) & test"
    printf '\n%s checks, %s failed\n' "$n" "$red"; [ "$red" -eq 0 ]
}

if [ "${SELF_TEST:-0}" = 1 ]; then self_test; exit $?; fi
if [ "${FILTERSET:-0}" = 1 ]; then
    if [ -z "${FS_TARGETS+x}" ]; then FS_TARGETS=$(cat); fi
    filterset_from_targets "$FS_TARGETS"; exit $?
fi
if [ "$UNION" = 1 ]; then
    if [ "$TIER_OF_RECORD" != 1 ]; then
        printf 'usage: --union-touched requires --tier-of-record\n' >&2
        exit 2
    fi
    if [ "$EVENT" != "pull_request" ]; then
        printf 'usage: --union-touched requires --event pull_request (got "%s")\n' "$EVENT" >&2
        exit 2
    fi
    if [ -z "$COMPARAND$DIFF_FROM" ]; then
        printf 'usage: --union-touched requires --comparand REF (or --diff-from FILE)\n' >&2
        exit 2
    fi
    URC=0; union_touched || URC=$?; exit "$URC"
fi
if [ "$TIER_OF_RECORD" = 1 ]; then tier_of_record || exit 1; exit 0; fi
[ -n "$EVENT" ] || { printf 'usage: %s --event EVENT ...\n' "$0" >&2; exit 2; }
decide
