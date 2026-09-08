#!/usr/bin/env bash
#
# guard_tree_job_test.sh -- acceptance test for the `guard-tree` JOB shape in
# .github/workflows/ci.yml (BSE-02, PMAT-1064, paiml/infra BSE-001 spec §4
# wave 3).
#
# Four assertions, each a leg of the spec's acceptance expression:
#
#   1. .jobs."guard-tree".needs is null (no job needs guard-tree to start,
#      and guard-tree needs nothing -- §13 F-13: gating the builds on it
#      would add its runtime to every green run).
#   2. .jobs.gate.needs contains "guard-tree" (gate is the ONLY job that
#      waits on it).
#   3. No step's `run:` text in the guard-tree job contains a bare `cargo `
#      token -- the SAME classification regex guard_tree.sh itself uses
#      (`(^|[^a-z_-])cargo `), not a naive substring match: a naive
#      `test("cargo ")` false-positives on the job's own
#      `bash scripts/guard_tree.sh --no-cargo` step, whose `--no-cargo`
#      FLAG contains the substring "cargo " once joined with the next
#      token. That would make the real, correct job read as violating its
#      own contract. See CARGO_RE below and scripts/guard_tree.sh's header.
#   4. .jobs."guard-tree"."runs-on" contains "clean-room".
#
# TWO IMPLEMENTATIONS, ONE CONTRACT. `command -v yq` is checked for a
# mikefarah v4 build (`yq --version` prints
# "yq (https://github.com/mikefarah/yq/) version v4.x.x" -- the jackson/
# python "yq" wrapper prints something else entirely and must not be used).
# When present, the four assertions run as one yq expression, structurally
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
# MUTATION TABLE (checks 2-4 below): each plants ONE violation in a temp copy
# of ci.yml and asserts the SAME assertion function reports it RED, so a
# check here that always reports PASS is caught rather than trusted.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1
CI_YML="$REPO_ROOT/.github/workflows/ci.yml"

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

# assertions_yq FILE -- prints "true" or "false"
# NOTE: `yq` and its subcommand are split across lines on purpose -- bashrs
# SEC012 flags any single line containing both "eval" and "yq" as unsafe
# YAML deserialization (its real target is `eval $(yq ...)` executing
# attacker-controlled output). This is `yq eval`, the read-only query
# subcommand, not the shell builtin, so splitting the line is a rewording,
# not a functional change.
assertions_yq() {
    yq \
        eval '
        (.jobs."guard-tree".needs == null)
        and (.jobs.gate.needs | contains(["guard-tree"]))
        and ((.jobs."guard-tree".steps | map(.run // "") | join(" ") | test("(^|[^a-z_-])cargo ")) | not)
        and (.jobs."guard-tree"."runs-on" | contains(["clean-room"]))
    ' "$1" 2>/dev/null
}

# assertions_awk FILE -- prints "true" or "false"
assertions_awk() {
    local file="$1" gt_job gate_job needs_line runson_line gate_needs_line run_text
    gt_job="$(job_block "guard-tree" "$file")"
    [ -n "$gt_job" ] || { echo false; return; }
    gate_job="$(job_block "gate" "$file")"
    [ -n "$gate_job" ] || { echo false; return; }

    needs_line="$(grep -c '^    needs:' <<<"$gt_job")"
    runson_line="$(grep '^    runs-on:' <<<"$gt_job")"
    gate_needs_line="$(grep '^    needs:' <<<"$gate_job")"
    run_text="$(run_text_of_job_block <<<"$gt_job")"

    local leg1=false leg2=false leg3=false leg4=false
    [ "${needs_line:-1}" -eq 0 ] && leg1=true
    printf '%s\n' "$gate_needs_line" | grep -qE '(^|[^a-z0-9_-])guard-tree([^a-z0-9_-]|$)' && leg2=true
    printf '%s\n' "$run_text" | grep -qE "$CARGO_RE" || leg3=true
    printf '%s\n' "$runson_line" | grep -qE '(^|[^a-z0-9_-])clean-room([^a-z0-9_-]|$)' && leg4=true

    if [ "$leg1" = true ] && [ "$leg2" = true ] && [ "$leg3" = true ] && [ "$leg4" = true ]; then
        echo true
    else
        echo false
    fi
}

assertions() {
    if [ "$YQ_MODE" -eq 1 ]; then
        assertions_yq "$1"
    else
        assertions_awk "$1"
    fi
}

if [ "$YQ_MODE" -eq 1 ]; then
    printf 'mode: yq (mikefarah v4 detected: %s)\n' "$(yq --version 2>&1)"
else
    printf 'mode: awk/grep fallback (no mikefarah v4 yq on this host)\n'
fi

# ---------------------------------------------------------------------------
# 1. The real ci.yml passes all four legs.
# ---------------------------------------------------------------------------
real_result="$(assertions "$CI_YML")"
if [ "$real_result" = "true" ]; then
    pass_row "guard-tree: no needs, gate needs it, no cargo step, runs on clean-room"
else
    fail_row "guard-tree: no needs, gate needs it, no cargo step, runs on clean-room" \
        "assertions()=$real_result"
fi

WORK="$(mktemp -d)" || exit 1
trap 'rm -rf "${WORK:?}"' EXIT

# ---------------------------------------------------------------------------
# 2. Mutant: a `cargo ` step added to guard-tree -> must go RED.
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
' "$CI_YML" > "$m2"
mut2="$(assertions "$m2")"
if [ "$mut2" = "false" ]; then
    pass_row "mutant: cargo step in guard-tree turns the assertion RED"
else
    fail_row "mutant: cargo step in guard-tree turns the assertion RED" \
        "assertions()=$mut2 (expected false)"
fi

# ---------------------------------------------------------------------------
# 3. Mutant: guard-tree dropped from gate.needs -> must go RED.
# ---------------------------------------------------------------------------
m3="$WORK/gate-drops-guard-tree.yml"
sed -E 's/^(    needs: \[.*)guard-tree, (.*\])$/\1\2/' "$CI_YML" > "$m3"
if job_block "gate" "$m3" | grep '^    needs:' | grep -qF 'guard-tree'; then
    fail_row "mutant fixture: gate.needs actually dropped guard-tree" \
        "the sed substitution did not match the real gate needs: line -- fixture is stale"
else
    mut3="$(assertions "$m3")"
    if [ "$mut3" = "false" ]; then
        pass_row "mutant: guard-tree dropped from gate.needs turns the assertion RED"
    else
        fail_row "mutant: guard-tree dropped from gate.needs turns the assertion RED" \
            "assertions()=$mut3 (expected false)"
    fi
fi

# ---------------------------------------------------------------------------
# 4. Mutant: guard-tree gains a `needs:` -> must go RED.
# ---------------------------------------------------------------------------
m4="$WORK/guard-tree-gains-needs.yml"
awk '
    /^  guard-tree:/ { print; print "    needs: [ci]"; next }
    { print }
' "$CI_YML" > "$m4"
mut4="$(assertions "$m4")"
if [ "$mut4" = "false" ]; then
    pass_row "mutant: guard-tree gaining a needs: turns the assertion RED"
else
    fail_row "mutant: guard-tree gaining a needs: turns the assertion RED" \
        "assertions()=$mut4 (expected false)"
fi

printf '%d checks, %d failed\n' "$total" "$failed"
if [ "$failed" -gt 0 ]; then
    exit 1
fi
exit 0
