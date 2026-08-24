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
"

# The workflows whose jobs are REQUIRED status checks on main.
REQUIRED_WORKFLOWS="
.github/workflows/ci.yml
.github/workflows/pr-gate.yml
"

rc=0
printf -- '--- no timing gate in a required check ------------------------------\n'

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
check_no_timing_in_required.sh
check_no_fabricated_baselines.sh
"

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
done < <(git ls-files 'scripts/check_*bench*.sh' 'scripts/check_*timing*.sh' \
                     'scripts/check_*throughput*.sh' 'scripts/check_*perf*.sh' 2>/dev/null)

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

printf '\n'
if [ "$rc" -eq 0 ]; then
    printf 'PASS  no timing assertion can reach a required status check.\n'
else
    printf 'FAIL  see rows above (#2671). Placement, not tuning, is the remedy.\n'
fi
exit "$rc"
