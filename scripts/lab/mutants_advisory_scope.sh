#!/usr/bin/env bash
# scripts/lab/mutants_advisory_scope.sh -- #6601 (spec sweep, PR-REVIEW SKILL.md §3.D). LAB/nightly check, never inside a PR,
# merge-queue or release job (C324). The spec line: Rust-source mutation (`cargo mutants --in-diff`, MUTANT_BUDGET=40) is
# ADVISORY; only the bash-guard row (100% kill on a guard's committed mutation set) blocks. This check holds that to account:
#   1. row    the SKILL.md §3.D table has a "Rust source" row with `cargo mutants --in-diff`, MUTANT_BUDGET=40 and "advisory".
#   2. guard  the "check_*.sh / contracts falsifier" row stays "yes" (blocking) -- advisory must not leak into the guard row.
#   3. scope  no workflow but mutants-nightly.yml invokes `cargo mutants`: the advisory run is nightly-only.
# Usage: scripts/lab/mutants_advisory_scope.sh | --self-test      Exit: 0 green, 1 RED, 2 cannot measure
# Overrides (tests): SKILL WORKFLOWS_DIR
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SKILL="${SKILL:-$ROOT/.claude/skills/pr-review/SKILL.md}"
WORKFLOWS_DIR="${WORKFLOWS_DIR:-$ROOT/.github/workflows}"
ALLOWED="mutants-nightly.yml"

run() {
    local red=0 rust guard f hits
    [ -f "$SKILL" ] || { echo "NO-MEASURE: $SKILL does not exist" >&2; return 2; }
    [ -d "$WORKFLOWS_DIR" ] || { echo "NO-MEASURE: $WORKFLOWS_DIR does not exist" >&2; return 2; }
    rust="$(grep -m1 -E '^\| Rust source \|' "$SKILL" || true)"
    guard="$(grep -m1 -E '^\| `scripts/check_\*\.sh`' "$SKILL" || true)"
    [ -n "$rust" ] && [ -n "$guard" ] || { echo "NO-MEASURE: §3.D table rows not found in $SKILL" >&2; return 2; }
    if grep -q 'cargo mutants --in-diff' <<<"$rust" && grep -Eq 'MUTANT_BUDGET=40([^0-9]|$)' <<<"$rust" \
        && grep -Eq '\| advisory \|$' <<<"$rust"; then echo "  ok    row    Rust-source mutation: --in-diff, MUTANT_BUDGET=40, advisory"
    else echo "  RED   row    Rust-source row is not (--in-diff, MUTANT_BUDGET=40, advisory): $rust"; red=1; fi
    if grep -Eq '\| \*\*yes\*\* \|$' <<<"$guard"; then echo "  ok    guard  bash-guard mutation row stays blocking"
    else echo "  RED   guard  bash-guard row no longer blocking: $guard"; red=1; fi
    hits=""
    for f in "$WORKFLOWS_DIR"/*.yml "$WORKFLOWS_DIR"/*.yaml; do
        [ -f "$f" ] || continue
        [ "$(basename "$f")" = "$ALLOWED" ] && continue
        if grep -Eq 'cargo[ -]mutants' "$f"; then hits="$hits $(basename "$f")"; fi
    done
    if [ -z "$hits" ]; then echo "  ok    scope  only $ALLOWED invokes cargo mutants"
    else echo "  RED   scope  cargo mutants outside the nightly:$hits"; red=1; fi
    if [ "$red" -eq 0 ]; then echo "GREEN mutants-advisory scope"; else echo "RED mutants-advisory scope"; fi
    return "$red"
}

self_test() {
    local d rc fail=0
    d="$(mktemp -d /tmp/mutadv.XXXXXX)"
    mkdir -p "$d/wf"; cp "$SKILL" "$d/SKILL.md"; printf 'run: cargo mutants\n' > "$d/wf/$ALLOWED"; printf 'run: cargo test\n' > "$d/wf/ci.yml"
    expect() { # name want-rc
        set +e; SKILL="$d/SKILL.md" WORKFLOWS_DIR="$d/wf" "$0" >/dev/null 2>&1; rc=$?; set -e
        if [ "$rc" = "$2" ]; then echo "  pass  $1 (rc=$rc)"; else echo "  FAIL  $1: want $2 got $rc"; fail=1; fi
    }
    expect "real skill + clean workflows is GREEN" 0
    cp "$SKILL" "$d/SKILL.md"; sed -i -E '/^\| Rust source \|/ s/\| advisory \|$/| **yes** |/' "$d/SKILL.md"; expect "Rust row promoted to blocking is RED" 1
    cp "$SKILL" "$d/SKILL.md"; sed -i -E '/^\| Rust source \|/ s/MUTANT_BUDGET=40/MUTANT_BUDGET=400/' "$d/SKILL.md"; expect "budget changed is RED" 1
    cp "$SKILL" "$d/SKILL.md"; sed -i -E '/^\| `scripts\/check_/ s/\| \*\*yes\*\* \|$/| advisory |/' "$d/SKILL.md"; expect "guard row demoted is RED" 1
    cp "$SKILL" "$d/SKILL.md"; printf 'run: cargo mutants --in-diff x\n' >> "$d/wf/ci.yml"; expect "cargo mutants in ci.yml is RED" 1
    printf 'run: cargo test\n' > "$d/wf/ci.yml"; : > "$d/SKILL.md"; expect "empty SKILL is NO-MEASURE" 2
    cp "$SKILL" "$d/SKILL.md"; mv "$d/wf" "$d/wf.gone"; expect "missing workflows dir is NO-MEASURE" 2
    rm -rf "${d:?}"
    return "$fail"
}

if [ "${1:-}" = "--self-test" ]; then self_test; else run; fi
