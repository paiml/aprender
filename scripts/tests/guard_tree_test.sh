#!/usr/bin/env bash
#
# guard_tree_test.sh -- acceptance test for scripts/guard_tree.sh (BSE-01,
# PMAT-1062).
#
# Asserts:
#   1. guard_tree.sh runs EVERY guard in a fixture and reports EVERY
#      failure, not just the first (no fail-fast), exiting 1.
#   2. A fail-fast MUTANT of guard_tree.sh (stops at the first failure, the
#      exact regression this script exists to catch) is shown to go RED
#      against the very same fixture and assertion -- proving check #1
#      actually discriminates rather than passing by construction.
#   3. `--list` equals `git ls-files 'scripts/check_*.sh'` over the real
#      repo.
#   4. `--list --no-cargo` has the same count as
#      `git ls-files 'scripts/check_*.sh' | grep -LE '(^|[^a-z_-])cargo '`.
#   5. The summary line reads `N checks, M failed`.
#   8. `--dry-run` DECIDES and executes nothing: over a fixture holding two
#      guards that exit 1 it must exit 0 and print no FAIL row.
#   9. The release-time skip is read from `check_no_timing_in_required.sh
#      --list` and ONLY from there -- planting a stub registry skips a guard,
#      removing it runs the guard again (the control).
#  10. The argument/env skip is read from the workflows -- an invocation with
#      an argument skips, a BARE invocation of the same guard does not.
#  11. Over the real repo, `--no-cargo --dry-run` skips guards for BOTH
#      reasons, because the two come from two oracles and either can go empty.
#  12. `check_guards_are_wired.sh` counts a dispatched guard as wired, and a
#      mutant whose dispatcher_wired() returns nothing -- the pre-BSE-01
#      behaviour that reported 29 CI-run guards as dark -- fails its own case
#      table.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1
GUARD_TREE="$REPO_ROOT/scripts/guard_tree.sh"

total=0
failed=0

pass_row() {
    total=$((total + 1))
    printf 'PASS  %s\n' "$1"
}

fail_row() {
    total=$((total + 1))
    failed=$((failed + 1))
    printf 'FAIL  %s\n' "$1"
    [ -n "${2:-}" ] && printf '      | %s\n' "$2"
}

cleanup_dirs=""
on_exit() {
    for d in $cleanup_dirs; do
        rm -rf "${d:?on_exit: refusing to rm -rf an empty path}"
    done
}
trap on_exit EXIT

# ---------------------------------------------------------------------------
# Fixture: a throwaway git repo carrying a real copy of guard_tree.sh plus
# one passing and two deliberately failing check_*.sh guards.
# ---------------------------------------------------------------------------
make_fixture() {
    dir="$(mktemp -d)" || exit 1
    # An EMPTY template dir: `git init` otherwise inherits this machine's global
    # template hooks, and a pre-commit hook that fails would leave the fixture
    # with no commit at all -- `git ls-files` then answers nothing and every row
    # below passes over an empty universe.
    mkdir -p "$dir/.empty-git-template"
    git -C "$dir" init -q --template="$dir/.empty-git-template"
    git -C "$dir" config user.email test@example.invalid
    git -C "$dir" config user.name "guard_tree_test"
    mkdir -p "$dir/scripts"
    cp "$GUARD_TREE" "$dir/scripts/guard_tree.sh"
    chmod +x "$dir/scripts/guard_tree.sh"

    cat >"$dir/scripts/check_good.sh" <<'SH'
#!/usr/bin/env bash
exit 0
SH

    cat >"$dir/scripts/check_fail_alpha.sh" <<'SH'
#!/usr/bin/env bash
echo "check_fail_alpha: deliberately red"
exit 1
SH

    cat >"$dir/scripts/check_fail_beta.sh" <<'SH'
#!/usr/bin/env bash
echo "check_fail_beta: deliberately red"
exit 1
SH

    chmod +x "$dir"/scripts/check_*.sh

    # A guard WITHOUT the executable bit. Ten of the 96 tracked guards are
    # mode 100644 (git ls-files -s 'scripts/check_*.sh'), and ci.yml has
    # always invoked every guard as `bash scripts/check_X.sh`, so the mode
    # never mattered there. A runner that execs the path directly turns those
    # ten into "Permission denied" FAIL rows -- a red row for a guard that
    # was never run, which is the one thing a run-all runner may not do.
    # Created AFTER the blanket chmod above so the mode is the assertion.
    cat >"$dir/scripts/check_noexec.sh" <<'SH'
#!/usr/bin/env bash
echo "check_noexec: passes, and carries no executable bit"
exit 0
SH
    chmod 644 "$dir/scripts/check_noexec.sh"

    git -C "$dir" add -A
    git -C "$dir" -c commit.gpgsign=false commit -q -m fixture
    printf '%s\n' "$dir"
}

# make_fail_fast_mutant SRC DST -- rewrite guard_tree.sh so it exits on the
# FIRST failed guard instead of collecting every result. This is the exact
# shape of the bug guard_tree.sh exists to fix (GitHub Actions steps stopping
# at the first red step), reintroduced deliberately so we can prove the
# assertion above would have caught it.
make_fail_fast_mutant() {
    src="$1"
    dst="$2"
    sed 's/failed=\$((failed + 1))/failed=$((failed + 1)); exit 1/' "$src" >"$dst"
    chmod +x "$dst"
}

# ---------------------------------------------------------------------------
# 1. Run-all / report-all, against the real guard_tree.sh.
# ---------------------------------------------------------------------------
fixture="$(make_fixture)"
cleanup_dirs="$cleanup_dirs $fixture"

out="$(cd "$fixture" && bash scripts/guard_tree.sh 2>&1)"
rc=$?
n_alpha="$(grep -c 'check_fail_alpha.sh' <<<"$out")"
n_beta="$(grep -c 'check_fail_beta.sh' <<<"$out")"

if [ "$rc" -eq 1 ] && [ "${n_alpha:-0}" -gt 0 ] && [ "${n_beta:-0}" -gt 0 ]; then
    pass_row "runs every guard and reports every failure (both names present, exit 1)"
else
    fail_row "runs every guard and reports every failure" \
        "rc=$rc n_alpha=${n_alpha:-0} n_beta=${n_beta:-0}"
fi

# ---------------------------------------------------------------------------
# 1b. A guard without the executable bit must be RUN, not reported FAIL.
#     ci.yml invokes every guard as `bash scripts/check_X.sh`; the runner must
#     use the same form, or the ten mode-644 guards in this repo become red
#     rows nobody executed.
# ---------------------------------------------------------------------------
n_noexec_pass="$(grep -c '^PASS  scripts/check_noexec\.sh \[run\]$' <<<"$out")"
n_noexec_denied="$(grep -c 'check_noexec.*[Pp]ermission denied' <<<"$out")"

if [ "${n_noexec_pass:-0}" -gt 0 ] && [ "${n_noexec_denied:-0}" -eq 0 ]; then
    pass_row "a mode-644 guard is run through bash, not exec'd (PASS row, no EACCES)"
else
    fail_row "a mode-644 guard is run through bash, not exec'd" \
        "pass_rows=${n_noexec_pass:-0} permission_denied=${n_noexec_denied:-0}"
fi

# ---------------------------------------------------------------------------
# 2. The fail-fast mutation must go RED against the SAME fixture and the
#    SAME assertion (poka-yoke: prove #1 actually discriminates).
# ---------------------------------------------------------------------------
mutant="$fixture/scripts/guard_tree_failfast.sh"
make_fail_fast_mutant "$fixture/scripts/guard_tree.sh" "$mutant"

mutant_out="$(cd "$fixture" && bash scripts/guard_tree_failfast.sh 2>&1)"
m_alpha="$(grep -c 'check_fail_alpha.sh' <<<"$mutant_out")"
m_beta="$(grep -c 'check_fail_beta.sh' <<<"$mutant_out")"

# The mutant stops at the FIRST failing guard git ls-files hands back, so it
# can never report BOTH fail_alpha and fail_beta -- if it did, the mutation
# failed to reintroduce fail-fast and this row itself must fail loudly.
if [ "${m_alpha:-0}" -gt 0 ] && [ "${m_beta:-0}" -gt 0 ]; then
    fail_row "fail-fast mutation must go RED (mutation did not discriminate)" \
        "mutant reported both names: m_alpha=${m_alpha:-0} m_beta=${m_beta:-0}"
else
    pass_row "fail-fast mutation goes RED (misses at least one failing guard name)"
fi

# ---------------------------------------------------------------------------
# 3. --list equals git ls-files 'scripts/check_*.sh' over the real repo.
# ---------------------------------------------------------------------------
real_list="$(cd "$REPO_ROOT" && git ls-files 'scripts/check_*.sh' | sort)"
guard_list="$(cd "$REPO_ROOT" && bash scripts/guard_tree.sh --list | sort)"

if [ "$real_list" = "$guard_list" ]; then
    pass_row "--list equals git ls-files 'scripts/check_*.sh'"
else
    fail_row "--list equals git ls-files 'scripts/check_*.sh'" "lists differ"
fi

# ---------------------------------------------------------------------------
# 4. --list --no-cargo count matches the grep -LE count, over the real repo.
# ---------------------------------------------------------------------------
expected_no_cargo="$(cd "$REPO_ROOT" && git ls-files 'scripts/check_*.sh' \
    | xargs -r grep -LE '(^|[^a-z_-])cargo ' | wc -l | tr -d ' ')"
actual_no_cargo="$(cd "$REPO_ROOT" && bash scripts/guard_tree.sh --list --no-cargo \
    | wc -l | tr -d ' ')"

if [ "$expected_no_cargo" = "$actual_no_cargo" ]; then
    pass_row "--no-cargo count matches grep -LE '(^|[^a-z_-])cargo ' ($actual_no_cargo)"
else
    fail_row "--no-cargo count matches grep -LE count" \
        "expected=$expected_no_cargo actual=$actual_no_cargo"
fi

# ---------------------------------------------------------------------------
# 5. Summary line reads "N checks, M failed".
# ---------------------------------------------------------------------------
n_summary="$(grep -cE '^[0-9]+ checks, [0-9]+ failed$' <<<"$out")"
if [ "${n_summary:-0}" -gt 0 ]; then
    pass_row 'summary line reads "N checks, M failed"'
else
    fail_row 'summary line reads "N checks, M failed"' "no matching line in: $out"
fi

# ---------------------------------------------------------------------------
# 6. PMAT-238 (paiml/infra#435): the container CARGO_HOME mount must carry
#    registry/ together with .package-cache and .package-cache-mutate.
#
#    A registry mounted WITHOUT both lock files is infra#77: every container
#    takes a private flock while they all write one shared registry, so the
#    lock serialises nothing and the registry corrupts instead. The two legs
#    below are (a) ci.yml names the three together, and (b) the job that runs
#    the cargo guards contains no `:/usr/local/cargo/registry` mount -- (b) is
#    the discriminating one: restoring the registry-only mount turns it red.
#
#    grep -c over a captured variable throughout. `producer | grep -q`
#    SIGPIPEs the producer under pipefail and can read a real match as a
#    false negative.
# ---------------------------------------------------------------------------
CI_YML="$REPO_ROOT/.github/workflows/ci.yml"
ci_text="$(cat "$CI_YML")"

n_together="$(grep -cE 'registry/.*\.package-cache.*\.package-cache-mutate' <<<"$ci_text")"
if [ "${n_together:-0}" -gt 0 ]; then
    pass_row "ci.yml names registry/, .package-cache and .package-cache-mutate together"
else
    fail_row "ci.yml names registry/, .package-cache and .package-cache-mutate together" \
        "no line names all three (matches=${n_together:-0})"
fi

# The guard-cargo job body: from its key to the next top-level job key.
guard_cargo_job="$(awk '/^  guard-cargo:/{f=1} f&&/^  [a-z][a-z0-9_-]*:/&&!/^  guard-cargo:/{f=0} f' <<<"$ci_text")"
n_job="$(grep -c 'guard-cargo:' <<<"$guard_cargo_job")"
n_registry_only="$(grep -c -- ':/usr/local/cargo/registry' <<<"$guard_cargo_job")"
n_cargo_home="$(grep -c -- '-e CARGO_HOME=' <<<"$guard_cargo_job")"

if [ "${n_job:-0}" -eq 1 ] && [ "${n_registry_only:-0}" -eq 0 ] && [ "${n_cargo_home:-0}" -gt 0 ]; then
    pass_row "guard-cargo mounts the whole CARGO_HOME, never registry/ alone"
else
    fail_row "guard-cargo mounts the whole CARGO_HOME, never registry/ alone" \
        "job_found=${n_job:-0} registry_only_mounts=${n_registry_only:-0} cargo_home_env=${n_cargo_home:-0}"
fi

# ---------------------------------------------------------------------------
# 7. The two guard jobs must not resolve to one target dir. workspace-test
#    mounts run-<RUN_ID>; guard-cargo must mount run-<RUN_ID>-guards, or the
#    dep-info race (aprender#2822) is back.
# ---------------------------------------------------------------------------
n_suffix="$(grep -c 'run-\${{ github.run_id }}-guards' <<<"$guard_cargo_job")"
if [ "${n_suffix:-0}" -gt 0 ]; then
    pass_row "guard-cargo has its own target dir suffix (run-<RUN_ID>-guards)"
else
    fail_row "guard-cargo has its own target dir suffix (run-<RUN_ID>-guards)" \
        "matches=${n_suffix:-0}"
fi

# ---------------------------------------------------------------------------
# 8. --dry-run decides and executes NOTHING.
#
#    The fixture holds two guards that exit 1. A dry run over it must exit 0
#    and print no FAIL row at all -- that is the only evidence available that
#    nothing ran, and it is exactly the evidence the real run produces the
#    opposite of two rows above.
# ---------------------------------------------------------------------------
dry_out="$(cd "$fixture" && bash scripts/guard_tree.sh --dry-run 2>&1)"
dry_rc=$?
n_dry_fail="$(grep -c '^FAIL' <<<"$dry_out")"
n_dry_run="$(grep -c '^run: ' <<<"$dry_out")"

if [ "$dry_rc" -eq 0 ] && [ "${n_dry_fail:-0}" -eq 0 ] && [ "${n_dry_run:-0}" -ge 4 ]; then
    pass_row "--dry-run decides without executing (rc=0, no FAIL row, $n_dry_run run: rows)"
else
    fail_row "--dry-run decides without executing" \
        "rc=$dry_rc fail_rows=${n_dry_fail:-0} run_rows=${n_dry_run:-0}"
fi

# ---------------------------------------------------------------------------
# 9. The release-time skip is DERIVED from check_no_timing_in_required.sh
#    --list, not from a list living in guard_tree.sh.
#
#    Planted stub registry naming check_fail_alpha.sh => that guard is skipped
#    and does NOT fail the run. Remove the stub => it is back to a FAIL row.
#    The control is the whole point: a runner that skipped everything would
#    pass the first half on its own.
# ---------------------------------------------------------------------------
cat >"$fixture/scripts/check_no_timing_in_required.sh" <<'SH'
#!/usr/bin/env bash
if [ "${1:-}" = "--list" ]; then
    printf 'check_fail_alpha.sh\n'
    exit 0
fi
exit 0
SH
reg_out="$(cd "$fixture" && bash scripts/guard_tree.sh --dry-run 2>&1)"
n_reg_skip="$(grep -c '^skipped: scripts/check_fail_alpha\.sh -- release-time (check_no_timing_in_required)$' <<<"$reg_out")"
rm -f "$fixture/scripts/check_no_timing_in_required.sh"
ctl_out="$(cd "$fixture" && bash scripts/guard_tree.sh --dry-run 2>&1)"
n_ctl_run="$(grep -c '^run: scripts/check_fail_alpha\.sh$' <<<"$ctl_out")"

if [ "${n_reg_skip:-0}" -eq 1 ] && [ "${n_ctl_run:-0}" -eq 1 ]; then
    pass_row "release-time skip is read from check_no_timing_in_required.sh --list (and only from it)"
else
    fail_row "release-time skip is read from check_no_timing_in_required.sh --list" \
        "with_registry_skip=${n_reg_skip:-0} without_registry_run=${n_ctl_run:-0}"
fi

# ---------------------------------------------------------------------------
# 10. The argument/env skip is DERIVED from the workflows themselves.
#
#     A guard whose only invocation carries an argument cannot be run bare by
#     this runner (check_beat_measurements.sh <beat-log> is the live case). The
#     control is the same workflow with the argument removed: a BARE invocation
#     must NOT skip it, or the runner would quietly stop running every guard
#     any workflow happens to name.
#
#     The bare control is written to ci.yml (not e.g. nightly.yml): since
#     PMAT-1062 a BARE invocation in a workflow OTHER than ci.yml, with none
#     in ci.yml, is the wired-elsewhere case (rows 13-17 below) and would
#     confound this row. ci.yml naming the guard for itself is neither
#     arg-wired nor wired-elsewhere -- it is the plain "this dispatcher's own
#     workflow also runs it bare" case, which must still run here.
# ---------------------------------------------------------------------------
mkdir -p "$fixture/.github/workflows"
printf 'jobs:\n  n:\n    steps:\n      - run: bash scripts/check_fail_beta.sh "$LOG"\n' \
    >"$fixture/.github/workflows/nightly.yml"
arg_out="$(cd "$fixture" && bash scripts/guard_tree.sh --dry-run 2>&1)"
n_arg_skip="$(grep -c '^skipped: scripts/check_fail_beta\.sh -- wired-with-args in nightly\.yml$' <<<"$arg_out")"

rm -f "$fixture/.github/workflows/nightly.yml"
printf 'jobs:\n  n:\n    steps:\n      - run: bash scripts/check_fail_beta.sh\n' \
    >"$fixture/.github/workflows/ci.yml"
bare_out="$(cd "$fixture" && bash scripts/guard_tree.sh --dry-run 2>&1)"
n_bare_run="$(grep -c '^run: scripts/check_fail_beta\.sh$' <<<"$bare_out")"
rm -rf "${fixture:?row 10: refusing to rm -rf an empty path}/.github"

if [ "${n_arg_skip:-0}" -eq 1 ] && [ "${n_bare_run:-0}" -eq 1 ]; then
    pass_row "argument-wired skip is derived from the workflows (a BARE invocation still runs)"
else
    fail_row "argument-wired skip is derived from the workflows" \
        "with_arg_skip=${n_arg_skip:-0} bare_run=${n_bare_run:-0}"
fi

# ---------------------------------------------------------------------------
# 11. Over the REAL repo, --no-cargo --dry-run skips both populations.
#
#     A count alone would pass on three skips of one kind; both reasons must
#     appear, because they come from two different oracles and either can go
#     silently empty.
# ---------------------------------------------------------------------------
real_dry="$(cd "$REPO_ROOT" && bash scripts/guard_tree.sh --no-cargo --dry-run 2>&1)"
n_skipped="$(grep -c '^skipped: ' <<<"$real_dry")"
n_release="$(grep -c 'release-time (check_no_timing_in_required)' <<<"$real_dry")"
n_argwired="$(grep -c 'wired-with-args in ' <<<"$real_dry")"

if [ "${n_skipped:-0}" -ge 3 ] && [ "${n_release:-0}" -ge 1 ] && [ "${n_argwired:-0}" -ge 1 ]; then
    pass_row "real --no-cargo --dry-run: $n_skipped skipped ($n_release release-time, $n_argwired arg-wired)"
else
    fail_row "real --no-cargo --dry-run skips both populations" \
        "skipped=${n_skipped:-0} release=${n_release:-0} argwired=${n_argwired:-0}"
fi

# ---------------------------------------------------------------------------
# 12. The wiring oracle knows the dispatcher, and a mutant that does not goes
#     RED against the SAME case table.
#
#     check_guards_are_wired.sh --self-test rows 3/4 assert that a workflow
#     whose ONLY content is `bash scripts/guard_tree.sh --no-cargo` wires the
#     derived subset. The mutant neuters dispatcher_wired() to return nothing
#     -- the pre-BSE-01 behaviour that reported 29 CI-run guards as dark -- and
#     must fail that table. Without this row, row 3 passing proves nothing.
# ---------------------------------------------------------------------------
wired_guard="$REPO_ROOT/scripts/check_guards_are_wired.sh"
wired_out="$(bash "$wired_guard" --self-test 2>&1)"
wired_rc=$?

mutdir="$(mktemp -d)" || exit 1
cleanup_dirs="$cleanup_dirs $mutdir"
mkdir -p "$mutdir/scripts"
cp "$REPO_ROOT/scripts/guard_tree.sh" "$mutdir/scripts/guard_tree.sh"
sed 's#^    \[ -f "\$root/scripts/guard_tree.sh" \] || return 0$#    return 0#' \
    "$wired_guard" >"$mutdir/scripts/check_guards_are_wired.sh"
mut_out="$(bash "$mutdir/scripts/check_guards_are_wired.sh" --self-test 2>&1)"
mut_rc=$?
n_mut_row3="$(grep -c '^FAIL  row 3' <<<"$mut_out")"

if [ "$wired_rc" -eq 0 ] && [ "$mut_rc" -ne 0 ] && [ "${n_mut_row3:-0}" -gt 0 ]; then
    pass_row "check_guards_are_wired.sh knows the dispatcher; the no-dispatcher mutant goes RED"
else
    fail_row "check_guards_are_wired.sh knows the dispatcher, mutant goes RED" \
        "real_rc=$wired_rc mutant_rc=$mut_rc mutant_row3_fail=${n_mut_row3:-0}"
fi

# ---------------------------------------------------------------------------
# 13-16. The wired-elsewhere skip (PMAT-1062): a guard bare-invoked ONLY in a
#     workflow other than ci.yml, and never in ci.yml at all, is
#     `skipped: ... wired-elsewhere <workflow>` -- one row, not a failure.
#
#     13. bare-invoked only in book.yml            -> skipped: wired-elsewhere
#     14. the SAME guard also bare-invoked in ci.yml -> run: (ci.yml still
#         wires it for itself; another workflow naming it too changes nothing)
#     15. invoked by no workflow at all (dark)      -> run: (BSE-01's intent)
#     16. a mutant that neuters the new rule         -> case 13 goes RED,
#         proving 13 actually discriminates rather than passing by
#         construction (same poka-yoke shape as check #2 above).
# ---------------------------------------------------------------------------
mkdir -p "$fixture/.github/workflows"
printf 'jobs:\n  b:\n    steps:\n      - run: bash scripts/check_good.sh\n' \
    >"$fixture/.github/workflows/book.yml"

elsewhere_out="$(cd "$fixture" && bash scripts/guard_tree.sh --dry-run 2>&1)"
n_elsewhere_skip="$(grep -c '^skipped: scripts/check_good\.sh -- wired-elsewhere book\.yml$' <<<"$elsewhere_out")"

if [ "${n_elsewhere_skip:-0}" -eq 1 ]; then
    pass_row "13: bare-invoked only outside ci.yml -> skipped: wired-elsewhere <workflow>"
else
    fail_row "13: bare-invoked only outside ci.yml -> skipped: wired-elsewhere" \
        "matches=${n_elsewhere_skip:-0} out=$elsewhere_out"
fi

printf 'jobs:\n  c:\n    steps:\n      - run: bash scripts/check_good.sh\n' \
    >"$fixture/.github/workflows/ci.yml"
also_ci_out="$(cd "$fixture" && bash scripts/guard_tree.sh --dry-run 2>&1)"
n_also_ci_run="$(grep -c '^run: scripts/check_good\.sh$' <<<"$also_ci_out")"
n_also_ci_skip="$(grep -c 'check_good\.sh -- wired-elsewhere' <<<"$also_ci_out")"
rm -f "$fixture/.github/workflows/ci.yml"

if [ "${n_also_ci_run:-0}" -eq 1 ] && [ "${n_also_ci_skip:-0}" -eq 0 ]; then
    pass_row "14: also bare-invoked in ci.yml still runs (not wired-elsewhere)"
else
    fail_row "14: also bare-invoked in ci.yml still runs" \
        "run_rows=${n_also_ci_run:-0} skip_rows=${n_also_ci_skip:-0}"
fi

rm -rf "${fixture:?row 15: refusing to rm -rf an empty path}/.github"
dark_out="$(cd "$fixture" && bash scripts/guard_tree.sh --dry-run 2>&1)"
n_dark_run="$(grep -c '^run: scripts/check_good\.sh$' <<<"$dark_out")"

if [ "${n_dark_run:-0}" -eq 1 ]; then
    pass_row "15: invoked by no workflow at all still runs (dark, BSE-01 intent)"
else
    fail_row "15: invoked by no workflow at all still runs" "run_rows=${n_dark_run:-0}"
fi

mkdir -p "$fixture/.github/workflows"
printf 'jobs:\n  b:\n    steps:\n      - run: bash scripts/check_good.sh\n' \
    >"$fixture/.github/workflows/book.yml"
no_elsewhere_mutant="$fixture/scripts/guard_tree_no_elsewhere.sh"
sed 's/wf="\$(wired_elsewhere_in "\$base")"/wf=""/' "$fixture/scripts/guard_tree.sh" \
    >"$no_elsewhere_mutant"
chmod +x "$no_elsewhere_mutant"
mutant_elsewhere_out="$(cd "$fixture" && bash scripts/guard_tree_no_elsewhere.sh --dry-run 2>&1)"
n_mut_elsewhere_skip="$(grep -c 'check_good\.sh -- wired-elsewhere' <<<"$mutant_elsewhere_out")"
rm -rf "${fixture:?row 16: refusing to rm -rf an empty path}/.github"

if [ "${n_mut_elsewhere_skip:-0}" -eq 0 ]; then
    pass_row "16: mutant that drops the wired-elsewhere rule goes RED on case 13"
else
    fail_row "16: mutant that drops the wired-elsewhere rule goes RED on case 13" \
        "mutant still emitted a wired-elsewhere skip: ${n_mut_elsewhere_skip:-0}"
fi

# ---------------------------------------------------------------------------
# 17. Over the REAL repo, --no-cargo --dry-run lists check_book_cli_parity.sh
#     as wired-elsewhere -- the concrete finding this ticket exists to fix
#     (book.yml owns it, has no cargo/apr on the bare --no-cargo runner).
# ---------------------------------------------------------------------------
real_elsewhere="$(cd "$REPO_ROOT" && bash scripts/guard_tree.sh --no-cargo --dry-run 2>&1)"
n_real_book="$(grep -c '^skipped: scripts/check_book_cli_parity\.sh -- wired-elsewhere book\.yml$' <<<"$real_elsewhere")"

if [ "${n_real_book:-0}" -eq 1 ]; then
    pass_row "17: real --no-cargo --dry-run reports check_book_cli_parity.sh as wired-elsewhere book.yml"
else
    fail_row "17: real --no-cargo --dry-run reports check_book_cli_parity.sh as wired-elsewhere" \
        "matches=${n_real_book:-0}"
fi

# ---------------------------------------------------------------------------
# 18-20. The unwired-baseline skip (PMAT-1062): a guard NAMED in
#     scripts/unwired_guards_baseline.txt -- check_guards_are_wired.sh's own
#     shrink-only ledger of already-accepted unwired guards -- is
#     `skipped: ... unwired-baseline (shrink-only ledger)`, not run.
#
#     18. planting the ledger with the guard's name -> skipped
#     19. the control: removing it from the ledger -> run: (a runner that
#         skipped everything would pass 18 on its own without this)
#     20. a mutant that ignores the ledger entirely -> case 18 goes RED,
#         proving 18 actually discriminates (same poka-yoke shape as #2, #16)
# ---------------------------------------------------------------------------
printf 'check_fail_alpha.sh\n' >"$fixture/scripts/unwired_guards_baseline.txt"
ledger_out="$(cd "$fixture" && bash scripts/guard_tree.sh --dry-run 2>&1)"
n_ledger_skip="$(grep -c '^skipped: scripts/check_fail_alpha\.sh -- unwired-baseline (shrink-only ledger)$' <<<"$ledger_out")"

if [ "${n_ledger_skip:-0}" -eq 1 ]; then
    pass_row "18: a guard named in unwired_guards_baseline.txt -> skipped: unwired-baseline"
else
    fail_row "18: a guard named in unwired_guards_baseline.txt -> skipped: unwired-baseline" \
        "matches=${n_ledger_skip:-0} out=$ledger_out"
fi

rm -f "$fixture/scripts/unwired_guards_baseline.txt"
noledger_out="$(cd "$fixture" && bash scripts/guard_tree.sh --dry-run 2>&1)"
n_noledger_run="$(grep -c '^run: scripts/check_fail_alpha\.sh$' <<<"$noledger_out")"

if [ "${n_noledger_run:-0}" -eq 1 ]; then
    pass_row "19: removing the guard from the ledger runs it again (control)"
else
    fail_row "19: removing the guard from the ledger runs it again" \
        "run_rows=${n_noledger_run:-0}"
fi

printf 'check_fail_alpha.sh\n' >"$fixture/scripts/unwired_guards_baseline.txt"
no_ledger_mutant="$fixture/scripts/guard_tree_no_ledger.sh"
sed 's/if is_unwired_baseline "\$base"; then/if false; then/' "$fixture/scripts/guard_tree.sh" \
    >"$no_ledger_mutant"
chmod +x "$no_ledger_mutant"
mutant_ledger_out="$(cd "$fixture" && bash scripts/guard_tree_no_ledger.sh --dry-run 2>&1)"
n_mut_ledger_skip="$(grep -c 'check_fail_alpha\.sh -- unwired-baseline' <<<"$mutant_ledger_out")"
rm -f "$fixture/scripts/unwired_guards_baseline.txt"

if [ "${n_mut_ledger_skip:-0}" -eq 0 ]; then
    pass_row "20: mutant that ignores the ledger goes RED on case 18"
else
    fail_row "20: mutant that ignores the ledger goes RED on case 18" \
        "mutant still emitted an unwired-baseline skip: ${n_mut_ledger_skip:-0}"
fi

# ---------------------------------------------------------------------------
# 21. Over the REAL repo, --dry-run (the full universe, since every ledgered
#     guard here happens to be cargo-classified and so never enters the
#     --no-cargo subset) lists at least one ledgered guard as unwired-baseline.
# ---------------------------------------------------------------------------
real_ledger="$(cd "$REPO_ROOT" && bash scripts/guard_tree.sh --dry-run 2>&1)"
n_real_ledger="$(grep -c -- '-- unwired-baseline (shrink-only ledger)$' <<<"$real_ledger")"

if [ "${n_real_ledger:-0}" -ge 1 ]; then
    pass_row "21: real --dry-run reports at least one ledgered guard as unwired-baseline ($n_real_ledger)"
else
    fail_row "21: real --dry-run reports at least one ledgered guard as unwired-baseline" \
        "matches=${n_real_ledger:-0}"
fi

printf '%d checks, %d failed\n' "$total" "$failed"
if [ "$failed" -gt 0 ]; then
    exit 1
fi
exit 0
