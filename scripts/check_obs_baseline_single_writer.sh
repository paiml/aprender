#!/usr/bin/env bash
#
# check_obs_baseline_single_writer.sh — APR-OBS-001 §4 rule 4 / R-10 (OBS-07, aprender#4494).
#
# The release-tag perf baseline has ONE writer: the release train. This guard is RED when
#   1. anything but scripts/release/autopilot.sh invokes `obs_baseline.sh write`
#      (shell, Makefiles, CI yaml, Rust, TOML; comment lines are not invocations);
#   2. anything under crates/ names the baseline files or the writer — apr (and the arbiter,
#      which consumes apr's crates) is the subject being measured and never writes its own
#      baseline (R-10);
#   3. the release train does NOT invoke the writer: a single writer that is absent is no
#      writer, and the rule would be silently empty (R-2).
#
#   bash scripts/check_obs_baseline_single_writer.sh [ROOT]   # default: this repo
#   bash scripts/check_obs_baseline_single_writer.sh --self-test
set -euo pipefail

WRITER="scripts/release/autopilot.sh"
EXEMPT_RE='^scripts/(obs_baseline\.sh|check_obs_baseline_single_writer\.sh)$'
SURFACE_RE='(\.sh|\.bash|\.mk|\.rs|\.toml|\.ya?ml|(^|/)Makefile)$'
# `obs_baseline.sh` (optionally quoted, optionally after a path or $VAR) followed by the word write
INVOKE_RE='obs_baseline\.sh["'"'"']?[[:space:]]+(["'"'"'])?write([^[:alnum:]_-]|$)'
COMMENT_RE='^[[:space:]]*(#|//)'
SUBJECT_RE='obs_baseline\.sh|apr-perf-baseline-v1\.(v|[$][{]?[A-Za-z_])'

# invokes <line> -> 0 if the line is a (non-comment) writer invocation
invokes() { ! grep -Eq -- "$COMMENT_RE" <<<"$1" && grep -Eq -- "$INVOKE_RE" <<<"$1"; }

scan() { # scan <root> -> prints one line per violation; exit 1 if any
    local root="$1" hit f rest n line bad=0 writer_calls=0 rc=0 hits subj
    # git grep finds the candidates (tracked files only); the per-line rules are applied below
    hits=$(git -C "$root" grep -nIE -e "$INVOKE_RE" -- .) || rc=$?
    [ "$rc" -le 1 ] || { printf 'obs single-writer: git grep failed (exit %d)\n' "$rc"; return 2; }
    while IFS= read -r hit; do
        [ -n "$hit" ] || continue
        f="${hit%%:*}"; rest="${hit#*:}"; n="${rest%%:*}"; line="${rest#*:}"
        [[ "$f" =~ $SURFACE_RE ]] || continue
        [[ "$f" =~ $EXEMPT_RE ]] && continue
        invokes "$line" || continue
        if [ "$f" = "$WRITER" ]; then
            writer_calls=$((writer_calls + 1))
        else
            printf 'RED second-writer %s:%s: %s\n' "$f" "$n" "$line"; bad=1
        fi
    done <<<"$hits"
    rc=0; subj=$(git -C "$root" grep -lIE -e "$SUBJECT_RE" -- crates) || rc=$?
    [ "$rc" -le 1 ] || { printf 'obs single-writer: git grep failed (exit %d)\n' "$rc"; return 2; }
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        printf 'RED subject-writes-baseline %s: crates/ must never name the baseline or its writer (R-10)\n' "$f"; bad=1
    done <<<"$subj"
    if [ "$writer_calls" -eq 0 ]; then
        printf 'RED no-writer %s does not invoke obs_baseline.sh write: rule 4 would have no baseline (R-2)\n' "$WRITER"; bad=1
    fi
    [ "$bad" -eq 0 ] || return 1
    printf 'GREEN obs baseline single writer: %s (%d call)\n' "$WRITER" "$writer_calls"
}

self_test() {
    set +e  # every case records its own status
    local pass=0 fail=0 s
    ok() { if [ "$2" = 0 ]; then pass=$((pass + 1)); printf 'ok   %s\n' "$1"; else fail=$((fail + 1)); printf 'FAIL %s\n' "$1"; fi; }
    # case table: the line matcher
    local -a MUST=(
        'bash scripts/obs_baseline.sh write --tag v1'
        '  "$REPO_ROOT/scripts/obs_baseline.sh" write \'
        "run: bash scripts/obs_baseline.sh 'write' --store x"
        'obs_baseline.sh write'
        '    cmd="scripts/obs_baseline.sh    write --tag v"'
        'Command::new("bash").args(["scripts/obs_baseline.sh write"])'
        '\tbash scripts/obs_baseline.sh write --tag $(TAG)'
    )
    local -a MUST_NOT=(
        '# bash scripts/obs_baseline.sh write --tag v1'
        '   // obs_baseline.sh write'
        'bash scripts/obs_baseline.sh check --store x'
        'bash scripts/obs_baseline.sh --self-test'
        'bash scripts/obs_baseline.sh writer'
        'bash scripts/obs_baseline.sh write-ahead'
        'obs_baseline.sh.bak write'
        'echo obs_baseline.shwrite'
    )
    for s in "${MUST[@]}"; do invokes "$(printf '%b' "$s")"; ok "must-match: $s" $?; done
    for s in "${MUST_NOT[@]}"; do ! invokes "$s"; ok "must-not-match: $s" $?; done

    # tree fixtures: the scan itself
    local T out rc
    T=$(mktemp -d "${TMPDIR:-/tmp}/obs-sw.XXXXXX")
    # shellcheck disable=SC2064
    trap "rm -rf -- '${T:?}'" RETURN
    fixture() { # fixture <name> -> a git tree whose release train is the writer
        mkdir -p "$T/$1/scripts/release" "$T/$1/crates/apr-cli/src" "$T/$1/ci"
        printf 'step_baseline() {\n  bash "$REPO_ROOT/scripts/obs_baseline.sh" write --tag "$T"\n}\n' > "$T/$1/scripts/release/autopilot.sh"
        printf '#!/bin/bash\n# the writer\n' > "$T/$1/scripts/obs_baseline.sh"
        printf 'fn main() {}\n' > "$T/$1/crates/apr-cli/src/main.rs"
        printf 'steps:\n  - run: bash scripts/obs_baseline.sh check --store s\n' > "$T/$1/ci/sections.yml"
        git -C "$T/$1" init -q && git -C "$T/$1" add -A
    }
    tree_case() { # tree_case <id> <want-rc> <want-grep|-> <name>
        rc=0; out=$(scan "$T/$4") || rc=$?
        [ "$rc" = "$2" ] && { [ "$3" = - ] || grep -q -- "$3" <<<"$out"; }; ok "tree: $1 (exit $rc)" $?
    }
    fixture clean; tree_case clean-green 0 GREEN clean
    fixture nightly; printf 'bash scripts/obs_baseline.sh write --tag v9\n' > "$T/nightly/ci/nightly.sh"
    git -C "$T/nightly" add -A; tree_case nightly-writer-red 1 'second-writer ci/nightly.sh:1' nightly
    fixture mk; printf 'baseline:\n\tbash scripts/obs_baseline.sh write --tag $(TAG)\n' > "$T/mk/Makefile"
    git -C "$T/mk" add -A; tree_case makefile-tab-red 1 'second-writer Makefile:2' mk
    fixture crate; printf 'const P: &str = "apr-perf-baseline-v1.v0.70.0.json";\n' > "$T/crate/crates/apr-cli/src/b.rs"
    git -C "$T/crate" add -A; tree_case crate-names-baseline-red 1 'subject-writes-baseline crates/apr-cli/src/b.rs' crate
    fixture crate2; printf 'let f = format!("apr-perf-baseline-v1.${tag}.json");\n' > "$T/crate2/crates/apr-cli/src/b.rs"
    git -C "$T/crate2" add -A; tree_case crate-templated-name-red 1 subject-writes-baseline crate2
    fixture ctr; printf '// see contracts/apr-perf-baseline-v1.yaml\n' > "$T/ctr/crates/apr-cli/src/c.rs"
    git -C "$T/ctr" add -A; tree_case crate-cites-contract-green 0 GREEN ctr
    fixture gone; printf 'step_close() { :; }\n' > "$T/gone/scripts/release/autopilot.sh"
    git -C "$T/gone" add -A; tree_case no-writer-red 1 'no-writer' gone
    fixture cmt; printf '# never: bash scripts/obs_baseline.sh write\n' > "$T/cmt/ci/x.sh"
    git -C "$T/cmt" add -A; tree_case comment-green 0 GREEN cmt
    fixture untracked; printf 'bash scripts/obs_baseline.sh write\n' > "$T/untracked/ci/y.sh"
    tree_case untracked-not-scanned 0 GREEN untracked

    printf 'check_obs_baseline_single_writer self-test: %d/%d passed\n' "$pass" "$((pass + fail))"
    [ "$fail" -eq 0 ] && [ "$pass" -gt 0 ]
}

case "${1:-}" in
    --self-test) self_test ;;
    -h|--help) sed -n '2,17p' "$0" ;;
    *) scan "${1:-$(git rev-parse --show-toplevel)}" ;;
esac
