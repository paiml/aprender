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
# Output: KEY=VALUE lines — tier, crates (space list), targets (crate:--lib or
# crate:--test:name, space list), check_workspace (1, only under rule (i)),
# reason, cite (the PR head sha on reuse).
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
#   crate:--bins       -> (package(crate) & kind(bin))    [bin-only crates: no lib target]
#   crate:--test:NAME  -> binary_id(crate::NAME)
# The binary-id forms are nextest's own, verified on cargo-nextest 0.9.132 against
# this workspace (`cargo nextest list --message-format json`): a lib suite's id is
# the bare package name, an integration target's is `package::target`, a bin's is
# `package::bin/name`. `kind(lib)`/`kind(bin)` are equality matches on those kinds,
# which is why the lib and bins tokens do not need to name the binary at all.
set -euo pipefail

EVENT=""; COMPARAND=""; DIFF_FROM=""; PR_HEAD=""; PR_CONCLUSION=""; REGISTRY="scripts/tree_reader_tests.txt"; ROOT="."
# The sibling scripts and the registry always come from the checkout this script
# runs in (cwd = repo root, in ci.yml and in the case table alike); --repo-root
# re-points only the GIT queries — HEAD, its parents, their trees — which is how
# the case table hands this a throwaway queue-shaped repository.
TREE="."
while [ $# -gt 0 ]; do
    case "$1" in
        --event) EVENT=$2; shift 2 ;;
        --comparand) COMPARAND=$2; shift 2 ;;
        --diff-from) DIFF_FROM=$2; shift 2 ;;
        --pr-head) PR_HEAD=$2; shift 2 ;;
        --pr-head-conclusion) PR_CONCLUSION=$2; shift 2 ;;
        --registry) REGISTRY=$2; shift 2 ;;
        --repo-root) ROOT=$2; shift 2 ;;
        --self-test) SELF_TEST=1; shift ;;
        # optional operand: `--filterset 'a:--lib b:--test:c'`, or nothing and the
        # list comes from stdin (how ci.yml pipes steps.tier.outputs.targets in).
        --filterset) FILTERSET=1; shift; if [ $# -gt 0 ]; then FS_TARGETS=$1; shift; fi ;;
        *) printf 'usage: %s --event EVENT [--comparand REF] [--diff-from FILE] [--pr-head SHA --pr-head-conclusion C] [--registry FILE] [--repo-root DIR] | --filterset [TARGETS] | --self-test\n' "$0" >&2; exit 2 ;;
    esac
done

targets_from_registry() { # -> space list crate:--lib | crate:--test:name
    grep -v '^#' "$1" | grep -v '^[[:space:]]*$' | awk -F"\t" '{ if ($2=="--test") printf "%s:--test:%s ", $1, $3; else printf "%s:%s ", $1, $2 }' | sed 's/ $//'
}

filterset_from_targets() { # <space list of crate:--lib|crate:--bins|crate:--test:NAME> -> nextest -E expression
    local t clause expr=""
    for t in $1; do
        case "$t" in
            *:--lib)    clause="(package(${t%:--lib}) & kind(lib))" ;;
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
    nreg=$(grep -vc '^#' "$TREE/$REGISTRY")
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
    row 0 "schedule -> full (FULL still lives on the nightlies — decision D-1)" '^tier=full' bash "$T" --event schedule
    row 0 "workflow_dispatch -> full" '^tier=full' bash "$T" --event workflow_dispatch
    row 2 "unknown event -> ENV (exit 2), never a guess" 'refusing to guess' bash "$T" --event release
    printf 'scripts/foo.sh\n' > "$td/d-scripts.txt"
    row 0 "pull_request, scripts-only diff -> quick with NO crates" '^crates=$' bash "$T" --event pull_request --diff-from "$td/d-scripts.txt"
    row 0 "  ...and the tree-reader targets are in it (readme_contract, the reader that bit #3039)" 'aprender-core:--test:readme_contract' bash "$T" --event pull_request --diff-from "$td/d-scripts.txt"
    row 0 "  ...including the lib target whose unit test reads a baseline (aprender-contracts)" 'aprender-contracts:--lib' bash "$T" --event pull_request --diff-from "$td/d-scripts.txt"
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
    row 0 "--filterset reads stdin and translates EVERY registry token (no ':--' survives)" '^NONE-LEFT$' bash -c "e=\$(bash '$T' --event pull_request --diff-from '$td/d-scripts.txt' | sed -n 's/^targets=//p' | bash '$T' --filterset); if [ -z \"\$e\" ]; then echo EMPTY; elif printf '%s' \"\$e\" | grep -q ':--'; then echo LEFTOVER; else echo NONE-LEFT; fi"
    row 0 "--filterset over the registry: one clause per registry line (nothing dropped, nothing invented)" '^EQUAL$' bash -c "reg=\$(grep -vc '^#' scripts/tree_reader_tests.txt); n=\$(bash '$T' --event pull_request --diff-from '$td/d-scripts.txt' | sed -n 's/^targets=//p' | bash '$T' --filterset | tr '|' '\n' | wc -l); [ \"\$reg\" = \"\$n\" ] && echo EQUAL || echo \"DIFFER registry=\$reg clauses=\$n\""
    # MUTANT: a copy that drops the tree-reader targets from the quick tier must lose readme_contract — the falsifier discriminates
    sed 's/targets=%s\\n/targets=\\n/; s/"\$(targets_from_registry "\$TREE\/\$REGISTRY")" //' "$T" > "$td/mutant.sh"
    row 0 "mutant without tree-reader targets loses readme_contract (proves the inclusion is load-bearing)" 'MUTANT-LOST' bash -c "if bash '$td/mutant.sh' --event pull_request --diff-from '$td/d-scripts.txt' | grep -q readme_contract; then echo MUTANT-KEPT; else echo MUTANT-LOST; fi"
    printf '\n%s checks, %s failed\n' "$n" "$red"; [ "$red" -eq 0 ]
}

if [ "${SELF_TEST:-0}" = 1 ]; then self_test; exit $?; fi
if [ "${FILTERSET:-0}" = 1 ]; then
    if [ -z "${FS_TARGETS+x}" ]; then FS_TARGETS=$(cat); fi
    filterset_from_targets "$FS_TARGETS"; exit $?
fi
[ -n "$EVENT" ] || { printf 'usage: %s --event EVENT ...\n' "$0" >&2; exit 2; }
decide
