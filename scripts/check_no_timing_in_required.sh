#!/usr/bin/env bash
#
# check_no_timing_in_required.sh — a release-time timing gate may never become a
# required PR check (aprender#2671, PARITY-004).
#
# WHY THIS EXISTS
# ---------------
# Eleven wall-clock assertions have failed in a required check in this repo.
# All three obvious remediations were tried and all three failed:
#
#   widen the tolerance   -> the gate stops detecting the thing it exists for
#   rewrite it as a ratio -> one such rewrite BLOCKED ALL 9 OPEN PRs
#   #[ignore] the flake   -> banned; a disabled gate is a deleted gate
#
# The remedy that worked is placement, not tuning: a timing assertion belongs at
# RELEASE time, where a human is already reading a verdict and a slow host costs
# one re-run rather than nine blocked authors.
#
# Until now that placement was protected by A COMMENT.
# scripts/unwired_guards_baseline.txt says check_multiplatform_dogfood.sh is
# "a RELEASE gate, not a PR gate. Deliberately unwired from CI" — and nothing
# mechanical stopped the next well-meaning PR from wiring it. Policy is what
# failed the other eleven times. This is the poka-yoke.
#
# THE REGISTRY IS THE UNIVERSE, and it is declared here rather than inferred, so
# that adding a timing gate without deciding its placement is impossible: a new
# entry must be classified, and an unclassified bench/timing guard is caught by
# PART 2 below.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

# Guards that assert something about DURATION or THROUGHPUT. Each runs at
# release time (Gate 12 of .claude/skills/pre-release/SKILL.md) or in a
# non-required nightly lane. None may appear in a required check.
RELEASE_TIME_ONLY="
check_multiplatform_dogfood.sh
check_bench_receipt.sh
check_perf041_marker.sh
"

# --list — the registry, one basename per line, and nothing else.
#
# WHY A LISTING MODE EXISTS
# -------------------------
# scripts/guard_tree.sh runs every guard in the universe from ONE required CI
# step. Without a way to ASK this file which guards are release-time, that
# dispatcher would have to carry a second copy of the registry — two
# hand-maintained lists with nothing tying them together, which is the exact
# root cause behind bashrs#266 and the four stale python-census paragraphs in
# paiml/infra. So the registry is declared once, here, and read from there.
#
# This must stay the FIRST thing this script does: guard_tree.sh calls it, and
# PART 1 below calls guard_tree.sh. The recursion terminates only because
# --list returns before reaching that call.
if [ "${1:-}" = "--list" ]; then
    for guard in $RELEASE_TIME_ONLY; do
        printf '%s\n' "$guard"
    done
    exit 0
fi

# The workflows whose jobs are REQUIRED status checks on main.
REQUIRED_WORKFLOWS="
.github/workflows/ci.yml
.github/workflows/pr-gate.yml
"
# ci/explicit-test-commands.d/*.cmd are not workflows, but ci.yml's REQUIRED
# workspace-test job executes every one of them (PMAT-3313). A release-time guard
# placed in a fragment runs in a required check exactly as if ci.yml named it.
for frag in ci/explicit-test-commands.d/*.cmd; do
    [ -f "$frag" ] && REQUIRED_WORKFLOWS="$REQUIRED_WORKFLOWS
$frag"
done

rc=0
printf -- '--- no timing gate in a required check ------------------------------\n'

# ── PART 1a: what a required workflow reaches by DISPATCH, naming nothing ───
#
# NAME-MATCHING ALONE STOPPED BEING SUFFICIENT THE DAY ci.yml GOT A RUNNER.
# BSE-01 replaced ~53 `bash scripts/check_X.sh` steps with one
# `bash scripts/guard_tree.sh --no-cargo` step in a REQUIRED job. That step
# names no guard at all, so the `grep -q "$guard" "$wf"` below — this gate's
# entire mechanism — passes over it while the dispatcher happily executes
# whatever its universe contains. check_bench_receipt.sh and
# check_perf041_marker.sh ARE in that cargo-free universe: without this part,
# both would have run in a required check with this gate reporting PASS, which
# is precisely the placement failure the header calls the poka-yoke for.
#
# The set is DERIVED, never listed: `guard_tree.sh --dry-run <mode>` prints the
# decision it will act on, and guard_tree.sh builds its skip set by calling
# `check_no_timing_in_required.sh --list` — this file. One registry, read from
# both ends. Delete the skip in guard_tree.sh and this part turns RED.
DISPATCHED=""
dispatch_modes=""
for wf in $REQUIRED_WORKFLOWS; do
    [ -f "$wf" ] || continue
    while IFS= read -r l; do
        [ -n "$l" ] || continue
        case "$l" in
            *--no-cargo*)   dispatch_modes="$dispatch_modes no-cargo" ;;
            *--cargo-only*) dispatch_modes="$dispatch_modes cargo-only" ;;
            *)              dispatch_modes="$dispatch_modes all" ;;
        esac
    done <<< "$(sed 's/#.*$//' "$wf" | grep -E "(^|[[:space:];&|(])((ba)?sh[[:space:]]+|\./)?[^[:space:]]*guard_tree\.sh([[:space:]]|$|['\"])" || true)"
done
dispatch_modes=$(printf '%s\n' $dispatch_modes | sort -u)
for mode in $dispatch_modes; do
    case "$mode" in
        no-cargo)   dflag="--no-cargo" ;;
        cargo-only) dflag="--cargo-only" ;;
        *)          dflag="" ;;
    esac
    dout=$(bash scripts/guard_tree.sh --dry-run $dflag 2>/dev/null) || dout=""
    # VACUITY: a dispatcher that answers nothing is not a dispatcher that runs
    # nothing. Refuse to grade it rather than sweep clean over an empty answer.
    if [ -z "$(printf '%s\n' "$dout" | grep -c . )" ] || [ "$(printf '%s\n' "$dout" | grep -c '^run: ')" -eq 0 ]; then
        printf 'FAIL  a required workflow dispatches guard_tree.sh %s, and that\n' "${dflag:-(all)}"
        printf '      dispatcher reported no run set. An oracle that answers nothing\n'
        printf '      cannot clear anything; fix the dispatcher, not this check.\n'
        rc=1
        continue
    fi
    DISPATCHED="$DISPATCHED $(printf '%s\n' "$dout" | sed -n 's|^run: ||p' | xargs -r -n1 basename | tr '\n' ' ')"
done

# ── PART 1: the registry may not appear in a required workflow ──────────────
printf 'PART 1 — release-time gates stay out of the required workflows\n'
checked=0
for guard in $RELEASE_TIME_ONLY; do
    [ -f "scripts/$guard" ] || {
        printf 'FAIL  registry names scripts/%s, which does not exist. A registry\n' "$guard"
        printf '      entry pointing at nothing is a rule guarding an empty universe.\n'
        rc=1; continue
    }
    checked=$((checked + 1))
    for wf in $REQUIRED_WORKFLOWS; do
        [ -f "$wf" ] || continue
        if grep -q "$guard" "$wf" 2>/dev/null; then
            printf 'FAIL  %s is named in %s\n' "$guard" "$wf"
            printf '      That workflow carries a REQUIRED status check. Eleven wall-clock\n'
            printf '      assertions have failed there; one ratio rewrite blocked all 9 open\n'
            printf '      PRs. Run it at release time (Gate 12) or in a nightly lane.\n'
            rc=1
        fi
    done
    case " $DISPATCHED " in
        *" $guard "*)
            printf 'FAIL  %s is RUN by scripts/guard_tree.sh from a required workflow.\n' "$guard"
            printf '      Nothing names it — the dispatcher reached it by universe. A\n'
            printf '      release-time gate reached by dispatch is still in a required\n'
            printf '      check. guard_tree.sh reads this registry via --list and must\n'
            printf '      emit `skipped: ... release-time` for it.\n'
            rc=1
            ;;
    esac
done

# VACUITY: a registry that names nothing sweeps clean.
if [ "$checked" -lt 2 ]; then
    printf 'FAIL  the registry resolved %s guard(s); at least 2 are required. A\n' "$checked"
    printf '      shrinking registry silently narrows what this gate protects.\n'
    rc=1
elif [ "$rc" -eq 0 ]; then
    printf 'ok    %s release-time gate(s), none named in a required workflow\n' "$checked"
fi

# ── PART 2: a timing guard outside the registry is unclassified ─────────────
#
# Without this, the gate protects only what someone remembered to list, which is
# the guard's-universe-from-the-wrong-side failure this repo keeps finding.
printf '\nPART 2 — every bench/timing guard is classified\n'
# Guards whose SUBJECT is timing placement rather than a duration. The name
# heuristic below cannot tell "asserts a duration" from "asserts that no
# duration is asserted" -- and this guard matched ITSELF on its first wired
# run. Listed explicitly, with the reason, rather than loosening the pattern.
META_GUARDS="
check_bench_protocol.sh
check_bench_threshold.sh
check_llama_pin.sh
check_no_fabricated_baselines.sh
check_no_timing_in_required.sh
check_perf_claims_cite_receipts.sh
check_perf_concurrency_groups.sh
check_perf_gate_selftest_scoped.sh
check_perf_matrix_schema.sh
check_perf_receipt_fields_have_producers.sh
"
# check_perf_gate_selftest_scoped.sh (#3676) matched on `perf`. It reads no clock:
# it takes a git tree diff against origin/main, decides whether perf_gate.sh's
# DERIVED input set was touched, and if so runs `perf_gate.sh --selftest` -- the
# fixture-driven case table that was ALREADY an explicit required step in
# guard-tree before #3676 moved it behind this scope. A statement about which
# files changed, not about duration.
# check_perf_claims_cite_receipts.sh (PERF-010) matches `check_*perf*.sh` and
# turned this guard RED the moment the file appeared — verified before wiring
# anything, not after:
#   PART 2 FAIL  bench/timing guard(s) not in the registry:
#                check_perf_claims_cite_receipts.sh          rc=1
# It reads no clock. It greps markdown for a speed comparison and asserts an
# evidence/ receipt path is cited beside it, which is a statement about CITATION,
# not about duration — the same reason check_bench_protocol.sh and
# check_llama_pin.sh sit here. Registering it in RELEASE_TIME_ONLY instead would
# be actively wrong: that list BANS a guard from the required workflows, and
# APR-PERF-GATE-001 §4.1 puts the claim guards in the merge phase precisely
# because they are text-only and deterministic.
# check_perf_receipt_fields_have_producers.sh matched on `perf`, and asserts no
# duration of any kind: it reads a YAML classification and the text of two
# reader scripts. Its one timing-shaped token is the word `timeouts` inside a
# comment, naming a RECEIPT FIELD rather than a bound. Verified by reading it
# rather than by trusting the name, which is what the heuristic above cannot do.
# check_perf_matrix_schema.sh matched on `perf` and reads only YAML: host
# classes, threshold_class/author coverage, cell statuses, anchor resolution. It
# compares no quantity to a clock.
# check_perf_concurrency_groups.sh matched on `perf` and reads only workflow
# YAML, asserting that a bench job declares a concurrency group. It is a
# statement about ISOLATION, not about duration.
# check_perf041_marker.sh is the opposite case and is in RELEASE_TIME_ONLY
# above: it compares a marker's `started_utc` against the matrix's
# `witness.max_age_days`, which IS a duration, and it belongs to the release
# surface (root Cargo.toml `[package.metadata.dogfood] gates`) rather than to a
# required workflow.

unclassified=""
while IFS= read -r f; do
    base=$(basename "$f")
    meta_flat=" $(printf '%s' "$META_GUARDS" | tr '\n' ' ') "
    case "$meta_flat" in *" $base "*) continue ;; esac
    # Normalise: the registry is newline-separated, so a space-delimited
    # `case` match silently never fires. Caught by this guard's own PART 2 on
    # its first run, reporting a guard that IS registered as unclassified.
    registry_flat=" $(printf '%s' "$RELEASE_TIME_ONLY" | tr '\n' ' ') "
    case "$registry_flat" in *" $base "*) continue ;; esac
    unclassified="$unclassified $base"
done < <(
    # TRACKED *and* UNTRACKED. A `git ls-files`-only universe lets a brand-new
    # guard pass until the moment it is committed — which is exactly how
    # check_bench_threshold.sh slipped through its own PR (it was green when
    # run pre-`git add`, and failed the instant it became tracked). Same shape
    # as SHIM-2644-03, where an UNTRACKED copy of the runner passed a
    # `git ls-files` check whose whole purpose was catching a second copy.
    { git ls-files 'scripts/check_*bench*.sh' 'scripts/check_*timing*.sh' \
                   'scripts/check_*throughput*.sh' 'scripts/check_*perf*.sh' 2>/dev/null
      find scripts -maxdepth 1 -type f \
           \( -name 'check_*bench*.sh' -o -name 'check_*timing*.sh' \
              -o -name 'check_*throughput*.sh' -o -name 'check_*perf*.sh' \) 2>/dev/null
    } | sort -u)

if [ -n "$unclassified" ]; then
    printf 'FAIL  bench/timing guard(s) not in the registry:%s\n' "$unclassified"
    printf '      Add each to RELEASE_TIME_ONLY (and keep it out of the required\n'
    printf '      workflows), or rename it if it asserts no duration.\n'
    rc=1
else
    printf 'ok    no unclassified bench/timing guard\n'
fi

# ── PART 3: the TRANSITIVE path through the Makefile ───────────────────────
#
# A required workflow that runs `make tier3` reaches every recipe tier3 reaches.
# Naming the guard directly in ci.yml is the obvious spelling; routing it
# through a make target is the one that gets past a workflow-only scan.
#
# The scope extension is re-mutated here rather than assumed: the standing scar
# (check_apr_bin_pinned.sh, #2360) is that extending a guard's SCOPE requires
# re-proving it in the NEW scope, because the old proof does not transfer — and
# the Makefile's `\t@cmd` form is exactly what a pattern written for `run:`
# lines misses.
printf '\nPART 3 — the transitive path: make targets a required workflow invokes\n'
if [ ! -f Makefile ]; then
    printf 'FAIL  no Makefile — this part scanned nothing, which is not a pass\n'
    rc=1
else
    ENTRY_TARGETS=$(grep -ohE 'make [a-z][a-z0-9_-]*' $REQUIRED_WORKFLOWS 2>/dev/null \
                    | awk '{print $2}' | sort -u)
    if [ -z "$ENTRY_TARGETS" ]; then
        printf 'ok    the required workflows invoke no make target\n'
    else
        hits=""
        for t in $ENTRY_TARGETS; do
            # The recipe body: everything indented under `target:` until the
            # next unindented line. Captures `\t@cmd` as well as `\tcmd`.
            recipe=$(awk -v t="$t" '$0 ~ "^"t":" {f=1;next} /^[^\t]/{f=0} f' Makefile)
            for guard in $RELEASE_TIME_ONLY; do
                case "$recipe" in
                    *"$guard"*) hits="$hits $t->$guard" ;;
                esac
            done
        done
        if [ -n "$hits" ]; then
            printf 'FAIL  a required workflow reaches a release-time gate through make:%s\n' "$hits"
            printf '      Transitive is still required. Move it to a release-only target.\n'
            rc=1
        else
            printf 'ok    %s entry target(s) checked, none reaches a release-time gate\n' \
                "$(printf '%s\n' $ENTRY_TARGETS | grep -c .)"
        fi
    fi
fi

# ── PART 4: no lib test asserts a TRUNCATED duration is positive (#3703) ────
#
# The parts above know timing GUARDS (scripts). They cannot see a timing
# ASSERTION inside a `--lib` test, which every required lane and the clean-room
# B2 gate run, so this guard passed while `assert!(stats.generation_time_ms > 0
# …)` sat in aprender-verify-ml. `generation_time_ms` is `elapsed.as_millis()`:
# 0 on a host that finishes in under a millisecond, and it stopped the v0.69.0
# clean-room on gx10-pool1.
#
# The shape: an assertion that an INTEGER duration (`.as_millis()`,
# `.as_micros()`, `.as_secs()`, or a name ending `_ms` / `_us` / `_secs`) is
# `> 0`, `>= 1` or `!= 0`. A float comparison (`> 0.0` on `as_secs_f64()`)
# does not truncate and is not this shape. Comment lines are not assertions.
# Each hit is either FIXED or recorded, with its reason, in
# scripts/wallclock_assert_baseline.txt, keyed by the assertion's TEXT with an
# occurrence count, never by `<path>:<line>`. A coordinate key goes stale on
# any edit above the line, and the shrink-only ratchet refuses the re-pointed
# entry, so an unrelated one-line edit in any file holding a recorded line
# would turn this red with no remedy short of editing this guard. The ratchet
# is `keyed`: no text may appear and no count may rise, so a new assertion of
# this shape fails and so does a copy of a recorded one. A count above the
# tree's is stale and fails too: lower it in the commit that fixed the line.
printf '\nPART 4 — no lib test asserts a truncated duration is positive (#3703)\n'
WALLCLOCK_RE='assert(_ne)?!\(.*(\.as_millis\(\)|\.as_micros\(\)|\.as_secs\(\)|\b[a-z_]*(_ms|_us|_secs)\b)[[:space:]]*(>[[:space:]]*0([^.0-9x]|$)|>=[[:space:]]*1([^.0-9]|$)|!=[[:space:]]*0([^.0-9x]|$))'
WALLCLOCK_BASELINE=scripts/wallclock_assert_baseline.txt
wc_matches() { # wc_matches <source line> -> 0 when it is the #3703 shape (never a comment)
    [[ "$1" =~ ^[[:space:]]*// ]] && return 1
    grep -qE "$WALLCLOCK_RE" <<<"$1"
}
# The pattern ships its case table (CLAUDE.md verification discipline #7).
wc_table_bad=0
while IFS='|' read -r want line; do
    [ -n "$want" ] || continue
    if wc_matches "$line"; then got=match; else got=no; fi
    if [ "$got" != "$want" ]; then
        printf 'FAIL  PART 4 case table: %s (wanted %s): %s\n' "$got" "$want" "$line"
        wc_table_bad=1
    fi
done <<'ROWS'
match|        assert!(stats.generation_time_ms > 0 || stats.total_generated < 10);
match|    assert!(meta.age().as_millis() >= 1);
match|    assert!(event.duration.as_micros() > 0);
match|        assert!(result.total_time_ms > 0);
match|    assert!(hunt_result.duration_ms > 0);
match|    assert!(elapsed.as_secs() != 0);
match|    assert!(snap.timestamp_ms > 0);
no|        assert!(result.total_time_ms > 0.0);
no|    assert!(elapsed.as_secs_f64() > 0.0);
no|    assert_eq!(stats.total_generated, programs.len());
no|    assert!(count > 0);
no|    assert!(items_ms_total > 0);
no|    // assert!(stats.generation_time_ms > 0) -- documented, not asserted
no|    assert!(result.total_time_ms >= 10);
no|    let ok = elapsed_ms > 0;
ROWS
if [ "$wc_table_bad" -ne 0 ]; then
    printf '      the PART 4 pattern disagrees with its own case table -- no scan result is trusted\n'
    rc=1
elif [ ! -f "$WALLCLOCK_BASELINE" ]; then
    printf 'FAIL  %s is missing -- the recorded hits are the other half of this check\n' "$WALLCLOCK_BASELINE"
    rc=1
else
    wc_files=$(git ls-files -- 'crates/*/src/*.rs' 'crates/*/src/**/*.rs' 2>/dev/null | grep -c .)
    # Every hit as `<path>:<line><TAB><text, tabs flattened, trimmed>`. The text is the key.
    wc_sites=$(git ls-files -z -- 'crates/*/src/*.rs' 'crates/*/src/**/*.rs' 2>/dev/null \
        | xargs -0 grep -nHE "$WALLCLOCK_RE" 2>/dev/null \
        | while IFS= read -r h; do
            f=${h%%:*}; rest=${h#*:}; n=${rest%%:*}; text=${rest#*:}
            wc_matches "$text" || continue
            text=${text//$'\t'/ }
            text=${text#"${text%%[![:space:]]*}"}
            text=${text%"${text##*[![:space:]]}"}
            printf '%s:%s\t%s\n' "$f" "$n" "$text"
          done | LC_ALL=C sort -u)
    wc_data=$(grep -vE '^[[:space:]]*(#|$)' "$WALLCLOCK_BASELINE")
    wc_malformed=$(printf '%s\n' "$wc_data" | grep . | grep -vE $'^[^\t]+\t[1-9][0-9]*$')
    # NEW: a text the tree holds more often than recorded (absent = 0). STALE: the reverse.
    # FILENAME, not NR == FNR: with an emptied ledger NR == FNR stays true through the
    # sites, reads every hit as a recorded entry, and passes them all.
    wc_new=$(LC_ALL=C awk -F'\t' 'FILENAME == ARGV[1] { want[$1] = $2 + 0; next }
        { have[$2]++; at[$2] = at[$2] "\n            " $1 }
        END { for (t in have) if (have[t] > want[t] + 0)
                printf "%s  (%d in tree, %d recorded)%s\n", t, have[t], want[t] + 0, at[t] }' \
        <(printf '%s\n' "$wc_data" | grep .) <(printf '%s\n' "$wc_sites" | grep .))
    wc_stale=$(LC_ALL=C awk -F'\t' 'FILENAME == ARGV[1] { want[$1] = $2 + 0; next }
        { have[$2]++ }
        END { for (t in want) if (want[t] > have[t] + 0)
                printf "%s  (%d recorded, %d in tree)\n", t, want[t], have[t] + 0 }' \
        <(printf '%s\n' "$wc_data" | grep .) <(printf '%s\n' "$wc_sites" | grep .))
    if [ "$wc_files" -eq 0 ]; then
        printf 'FAIL  PART 4 scanned 0 files under crates/*/src -- git ls-files answered nothing,\n'
        printf '      and a scan of nothing clears nothing\n'
        rc=1
    fi
    if [ -n "$wc_malformed" ]; then
        printf 'FAIL  %s has line(s) that are not TEXT, one TAB, then a count of at least 1:\n' "$WALLCLOCK_BASELINE"
        printf '%s\n' "$wc_malformed" | sed 's/^/        /'
        rc=1
    fi
    if [ -n "$wc_new" ]; then
        printf 'FAIL  a lib test asserts a truncated duration is positive -- fails on a fast host (#3703):\n'
        printf '%s\n' "$wc_new" | sed 's/^/        /'
        printf '      Assert what the code promises (a count, a state, the declared value),\n'
        printf '      never that elapsed time is positive. A copy of a recorded line counts too.\n'
        rc=1
    fi
    if [ -n "$wc_stale" ]; then
        printf 'FAIL  %s records more than the tree holds; lower or delete these in this commit:\n' "$WALLCLOCK_BASELINE"
        printf '%s\n' "$wc_stale" | sed 's/^/        /'
        rc=1
    fi
    [ "$wc_files" -gt 0 ] && [ -z "$wc_malformed" ] && [ -z "$wc_new" ] && [ -z "$wc_stale" ] &&
        printf 'ok    %s hit(s) in %s files, all %s recorded text(s) in %s; no new one\n' \
            "$(printf '%s\n' "$wc_sites" | grep -c .)" "$wc_files" \
            "$(printf '%s\n' "$wc_data" | grep -c .)" "$WALLCLOCK_BASELINE"
    # shrink-only against the merge-base: no text may appear, no count may rise (keyed)
    # shellcheck source=scripts/lib_baseline_ratchet.sh
    if . scripts/lib_baseline_ratchet.sh; then
        baseline_ratchet_check "$(pwd)" "$WALLCLOCK_BASELINE" keyed || rc=1
    else
        printf 'FAIL  scripts/lib_baseline_ratchet.sh could not be sourced -- the ratchet did not run\n'
        rc=1
    fi
fi

printf '\n'
if [ "$rc" -eq 0 ]; then
    printf 'PASS  no timing assertion can reach a required status check.\n'
else
    printf 'FAIL  see rows above (#2671). Placement, not tuning, is the remedy.\n'
fi
exit "$rc"
