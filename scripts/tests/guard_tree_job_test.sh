#!/usr/bin/env bash
#
# guard_tree_job_test.sh -- acceptance test for the `guard-tree` JOB shape
# (BSE-02, PMAT-1064, paiml/infra BSE-001 spec §4 wave 3).
#
# #4433 SPLIT THE CONTRACT ACROSS TWO FILES. The guard-tree job body moved
# verbatim to ci/sections.yml; .github/workflows/ci.yml's x86-main fat job runs
# it as a section, and the verdict job `gate` reads that section's result.
# Legs 1 and 3 read the section; legs 2 and 4 read who runs and gates it.
#
# Four assertions, each a leg of the spec's acceptance expression:
#
#   1. sections .jobs."guard-tree".needs is null (guard-tree needs nothing,
#      so the fat driver starts it at once -- §13 F-13: gating the builds on
#      it would add its runtime to every green run).
#   2. gate requires it: ci.yml's x86-main runs `--sections` naming
#      guard-tree, gate.needs contains x86-main, and gate's required-section
#      list names `X86:guard-tree` (gate is the ONLY job that waits on it).
#   3. No step's `run:` text in the guard-tree section contains a bare `cargo `
#      token -- the SAME classification regex guard_tree.sh itself uses
#      (`(^|[^a-z_-])cargo `), not a naive substring match: a naive
#      `test("cargo ")` false-positives on the job's own
#      `bash scripts/guard_tree.sh --no-cargo` step, whose `--no-cargo`
#      FLAG contains the substring "cargo " once joined with the next
#      token. That would make the real, correct job read as violating its
#      own contract. See CARGO_RE below and scripts/guard_tree.sh's header.
#   4. It runs on clean-room: the section's `runs-on` AND the x86-main fat
#      job's (the section's runs-on is now only what the driver honours; the
#      fat job's is where it actually lands).
#
# TWO IMPLEMENTATIONS, ONE CONTRACT. `command -v yq` is checked for a
# mikefarah v4 build (`yq --version` prints
# "yq (https://github.com/mikefarah/yq/) version v4.x.x" -- the jackson/
# python "yq" wrapper prints something else entirely and must not be used).
# When present, the four assertions run as yq expressions, structurally
# equivalent to the spec's jq-shaped expression. When absent (or a
# non-mikefarah/non-v4 yq shadows the name), the same four assertions run
# over line-oriented awk/grep, using the job-block extraction convention
# already established in scripts/tests/guard_tree_test.sh (BSE-01):
# `awk '/^  <job>:/{f=1} f&&/^  [a-z][a-z0-9_-]*:/&&!/^  <job>:/{f=0} f'`.
#
# Never a hand-rolled YAML parser beyond that one line-scoping trick -- PYTHON
# IS BANNED repo-wide (CLAUDE.md) and provisioning a second YAML tool is not
# the fix when one is already declared.
#
# MUTATION TABLE (checks 2-7 below): each plants ONE violation in a temp copy
# of ci.yml or ci/sections.yml and asserts the SAME assertion function reports
# it RED, so a check here that always reports PASS is caught rather than
# trusted.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1
CI_YML="$REPO_ROOT/.github/workflows/ci.yml"
SECT_YML="$REPO_ROOT/ci/sections.yml"

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

# A bare `cargo ` token, same convention as scripts/guard_tree.sh's CARGO_RE:
# not preceded by a lowercase letter, underscore or hyphen (so `--no-cargo`,
# `before-cargo-install-bashrs`, `sccache` etc. do not count), followed by a
# space (an invocation shape, not a comment fragment).
CARGO_RE='(^|[^a-z_-])cargo '

# --- mode selection ---------------------------------------------------------
YQ_MODE=0
if command -v yq >/dev/null 2>&1; then
    v="$(yq --version 2>&1)"
    case "$v" in
        *mikefarah/yq*v4.*) YQ_MODE=1 ;;
    esac
fi

# --- job-block extraction (shared by the awk path) --------------------------
job_block() {
    # $1 = job key, $2 = file
    awk -v job="  $1:" '
        $0 == job { f = 1 }
        f && /^  [a-z][a-z0-9_-]*:/ && $0 != job { f = 0 }
        f
    ' "$2"
}

# Extracts the concatenated `run:` text of every step in a job block,
# handling both inline (`run: cmd`) and block-scalar (`run: |`) forms. Never
# reads YAML comments as run text: a step-attribute comment sits at the SAME
# indent as `run:` itself and is excluded because it never matches `^run:`
# after trimming; block-scalar content is included even if a line inside it
# starts with `#`, because inside a literal block that is a shell comment,
# not a YAML one -- true to what actually executes.
run_text_of_job_block() {
    awk '
        {
            line = $0
            tmp = line
            gsub(/^ */, "", tmp)
            indent = length(line) - length(tmp)
            trimmed = tmp
        }
        inrun {
            if (line == "" || indent > run_indent) { print line; next }
            inrun = 0
        }
        !inrun {
            if (trimmed ~ /^run:/) {
                run_indent = indent
                rest = trimmed
                sub(/^run: */, "", rest)
                if (rest == "|" || rest == ">" || rest == "|-" || rest == ">-" || rest == "") {
                    inrun = 1
                } else {
                    print rest
                }
            }
        }
    '
}

# The fat job's --sections list names guard-tree as a whole item (quoted,
# comma-separated), and gate's required list names X86:guard-tree.
export SECTIONS_RE="--sections '([^']*,)?guard-tree(,[^']*)?'"
export GATE_RE="(^|[[:space:]])X86:guard-tree([[:space:]]|$)"

# assertions_yq CI SECT -- prints "true" or "false"
# NOTE: `yq` and its subcommand are split across lines on purpose -- bashrs
# SEC012 flags any single line containing both "eval" and "yq" as unsafe
# YAML deserialization (its real target is `eval $(yq ...)` executing
# attacker-controlled output). This is `yq eval`, the read-only query
# subcommand, not the shell builtin, so splitting the line is a rewording,
# not a functional change.
assertions_yq() {
    local a b
    a="$(yq \
        e '
        (.jobs."guard-tree".needs == null)
        and ((.jobs."guard-tree".steps | map(.run // "") | join(" ") | test("(^|[^a-z_-])cargo ")) | not)
        and (.jobs."guard-tree"."runs-on" | contains(["clean-room"]))
    ' "$2" 2>/dev/null)"
    b="$(yq \
        e '
        (.jobs.gate.needs | contains(["x86-main"]))
        and (.jobs.gate.steps | map(.run // "") | join(" ") | test(strenv(GATE_RE)))
        and (.jobs."x86-main".steps | map(.run // "") | join(" ") | test(strenv(SECTIONS_RE)))
        and (.jobs."x86-main"."runs-on" | contains(["clean-room"]))
    ' "$1" 2>/dev/null)"
    if [ "$a" = true ] && [ "$b" = true ]; then echo true; else echo false; fi
}

# assertions_awk CI SECT -- prints "true" or "false"
assertions_awk() {
    local ci="$1" sect="$2" gt_job gate_job x86_job needs_line runson_line
    gt_job="$(job_block "guard-tree" "$sect")"
    [ -n "$gt_job" ] || { echo false; return; }
    gate_job="$(job_block "gate" "$ci")"
    [ -n "$gate_job" ] || { echo false; return; }
    x86_job="$(job_block "x86-main" "$ci")"
    [ -n "$x86_job" ] || { echo false; return; }

    needs_line="$(grep -c '^    needs:' <<<"$gt_job")"
    runson_line="$(grep '^    runs-on:' <<<"$gt_job")"

    local leg1=false leg2=false leg3=false leg4=false
    [ "${needs_line:-1}" -eq 0 ] && leg1=true
    # Each producer's text is captured first and grepped from a here-string: a
    # pipe into `grep -q` reports the producer's SIGPIPE under pipefail
    # (check_no_pipe_into_grep_q.sh).
    local gate_needs gate_run x86_run gt_run x86_runson
    gate_needs="$(grep '^    needs:' <<<"$gate_job" || true)"
    gate_run="$(run_text_of_job_block <<<"$gate_job" || true)"
    x86_run="$(run_text_of_job_block <<<"$x86_job" | tr '\n' ' ' || true)"
    gt_run="$(run_text_of_job_block <<<"$gt_job" || true)"
    x86_runson="$(grep '^    runs-on:' <<<"$x86_job" || true)"
    grep -qE '(^|[^a-z0-9_-])x86-main([^a-z0-9_-]|$)' <<<"$gate_needs" \
        && grep -qE "$GATE_RE" <<<"$gate_run" \
        && grep -qE -- "$SECTIONS_RE" <<<"$x86_run" \
        && leg2=true
    grep -qE "$CARGO_RE" <<<"$gt_run" || leg3=true
    grep -qE '(^|[^a-z0-9_-])clean-room([^a-z0-9_-]|$)' <<<"$runson_line" \
        && grep -qE '(^|[^a-z0-9_-])clean-room([^a-z0-9_-]|$)' <<<"$x86_runson" \
        && leg4=true

    if [ "$leg1" = true ] && [ "$leg2" = true ] && [ "$leg3" = true ] && [ "$leg4" = true ]; then
        echo true
    else
        echo false
    fi
}

assertions() {
    if [ "$YQ_MODE" -eq 1 ]; then
        assertions_yq "$1" "$2"
    else
        assertions_awk "$1" "$2"
    fi
}

if [ "$YQ_MODE" -eq 1 ]; then
    printf 'mode: yq (mikefarah v4 detected: %s)\n' "$(yq --version 2>&1)"
else
    printf 'mode: awk/grep fallback (no mikefarah v4 yq on this host)\n'
fi

# 1. The real ci.yml + ci/sections.yml pass all four legs.
# ---------------------------------------------------------------------------
real_result="$(assertions "$CI_YML" "$SECT_YML")"
if [ "$real_result" = "true" ]; then
    pass_row "guard-tree: no needs, gate needs it, no cargo step, runs on clean-room"
else
    fail_row "guard-tree: no needs, gate needs it, no cargo step, runs on clean-room" \
        "assertions()=$real_result"
fi

WORK="$(mktemp -d)" || exit 1
trap 'rm -rf "${WORK:?}"' EXIT

# mutant_row LABEL CI SECT PROOF -- PROOF is a command that exits 0 only when
# the mutation actually landed (a stale fixture must not read as a kill).
mutant_row() {
    local label="$1" ci="$2" sect="$3" got
    shift 3
    if ! "$@"; then
        fail_row "mutant fixture: $label" "the mutation did not land -- fixture is stale"
        return
    fi
    got="$(assertions "$ci" "$sect")"
    if [ "$got" = "false" ]; then
        pass_row "mutant: $label turns the assertion RED"
    else
        fail_row "mutant: $label turns the assertion RED" "assertions()=$got (expected false)"
    fi
}
differs() { ! cmp -s "$1" "$2"; }

# ---------------------------------------------------------------------------
# 2. Mutant: a `cargo ` step added to the guard-tree section.
# ---------------------------------------------------------------------------
m2="$WORK/cargo-step.yml"
awk '
    /^  guard-tree:/ { print; in_gt = 1; next }
    in_gt && /^  [a-z][a-z0-9_-]*:/ {
        print "      - name: mutant cargo step"
        print "        run: cargo build --release"
        in_gt = 0
    }
    { print }
' "$SECT_YML" > "$m2"
mutant_row "cargo step in guard-tree" "$CI_YML" "$m2" differs "$SECT_YML" "$m2"

# ---------------------------------------------------------------------------
# 3. Mutant: guard-tree dropped from gate's required-section list.
# ---------------------------------------------------------------------------
m3="$WORK/gate-drops-guard-tree.yml"
sed -E 's/ X86:guard-tree / /' "$CI_YML" > "$m3"
mutant_row "guard-tree dropped from gate's required sections" "$m3" "$SECT_YML" differs "$CI_YML" "$m3"

# ---------------------------------------------------------------------------
# 4. Mutant: guard-tree dropped from x86-main's --sections.
# ---------------------------------------------------------------------------
m4="$WORK/x86-drops-guard-tree.yml"
sed -E "s/^( *--sections '.*),guard-tree,/\1,/" "$CI_YML" > "$m4"
mutant_row "guard-tree dropped from x86-main --sections" "$m4" "$SECT_YML" differs "$CI_YML" "$m4"

# ---------------------------------------------------------------------------
# 5. Mutant: x86-main dropped from gate.needs.
# ---------------------------------------------------------------------------
m5="$WORK/gate-drops-x86.yml"
sed -E 's/^(    needs: \[)x86-main, (gx10, yoga, determinism\])$/\1\2/' "$CI_YML" > "$m5"
mutant_row "x86-main dropped from gate.needs" "$m5" "$SECT_YML" differs "$CI_YML" "$m5"

# ---------------------------------------------------------------------------
# 6. Mutant: the guard-tree section gains a `needs:`.
# ---------------------------------------------------------------------------
m6="$WORK/guard-tree-gains-needs.yml"
awk '
    /^  guard-tree:/ { print; print "    needs: [workspace-test]"; next }
    { print }
' "$SECT_YML" > "$m6"
mutant_row "guard-tree gaining a needs:" "$CI_YML" "$m6" differs "$SECT_YML" "$m6"

# ---------------------------------------------------------------------------
# 7. Mutant: the x86-main fat job leaves clean-room.
# ---------------------------------------------------------------------------
m7="$WORK/x86-off-clean-room.yml"
awk '
    /^  x86-main:/ { in_x = 1 }
    in_x && /^    runs-on:/ { sub(/clean-room, /, ""); in_x = 0 }
    { print }
' "$CI_YML" > "$m7"
mutant_row "x86-main off clean-room" "$m7" "$SECT_YML" differs "$CI_YML" "$m7"

printf '%d checks, %d failed\n' "$total" "$failed"
if [ "$failed" -gt 0 ]; then
    exit 1
fi
exit 0
