#!/usr/bin/env bash
# #6721-class falsifier: the guards that refuse a mktemp -d outside /tmp must also accept
# "${TMPDIR:-/tmp}" children (RS-5 per-runner temp). --mutants proves each file's check can fail.
set -euo pipefail
cd "$(dirname "$0")/../.."
FILES=(scripts/check_cascade_clean_room_gate.sh scripts/check_comparator_one_client.sh
    scripts/check_ladder_write_errors.sh scripts/check_perf_receipt_fields_have_producers.sh
    scripts/perf_gate.sh scripts/perf_receipt_sign.sh scripts/spec_conformance.sh
    scripts/release/carry_milestone_items.sh scripts/check_thresholds_in_matrix.sh
    scripts/check_perf_matrix_schema.sh
    scripts/check_apr_bin_resolution.sh scripts/check_cascade_converges.sh scripts/check_crux_greedy_rows.sh
    scripts/check_lockfile_current.sh scripts/check_nextest_config_keys.sh scripts/model_ladder.sh
    scripts/check_arch_constraints_contract_covers_fallback.sh scripts/check_bin_cli_tests_wired.sh
    scripts/check_cascade_covers_all_crates.sh scripts/check_crux_inference_judge.sh
    scripts/check_crux_ollama_in_lock.sh scripts/check_crux_serve_code.sh
    scripts/check_facade_order_gate.sh scripts/check_guards_nightly_manifest.sh
    scripts/crux_inference_dogfood.sh
    scripts/check_perf_concurrency_groups.sh scripts/check_no_hand_rolled_parsers.sh
    scripts/check_publishable_deps_publishable.sh scripts/check_pv_bin_resolution.sh
    scripts/check_publish_preflight.sh scripts/include_fmt_ratchet.sh
    scripts/check_moe_routes_through_one_dispatch.sh scripts/check_no_competing_harnesses.sh
    scripts/check_perf041_marker.sh scripts/check_pr_ont_delta.sh scripts/check_readme_numeric_claims.sh
    scripts/check_serve_probe_evidence.sh scripts/check_tokenizer_nonascii_parity.sh
    scripts/ci/nextest_list_wrapper.sh scripts/dogfood_surfaces.sh scripts/lab/comparative_regex_agreement.sh
    scripts/lint_ratchet.sh scripts/parity_receipt_denominator.sh
    scripts/tests/model_ladder_diagnostic_not_truncated_test.sh)

# ok <file> : a case arm lists "${TMPDIR:-/tmp}", and the /tmp arm is still there
ok() { grep -qE '\|"\$\{TMPDIR:-/tmp\}"/(\?)?\*[|)]' "$1" && grep -qE '/tmp/(\?)?\*' "$1"; }

# fast behaviour: the cascade gate completes with a scratch dir outside /tmp
# outside_tmp : a scratch base that is NOT under /tmp, so the base-only arm is what lets the guard pass
outside_tmp() { mktemp -d -p "${XDG_CACHE_HOME:-$HOME/.cache}"; }

behave() {
    local d; d=$(outside_tmp); trap 'rm -rf "${d:?}"' RETURN
    mkdir "$d/scratch"
    TMPDIR=$d/scratch bash scripts/check_cascade_clean_room_gate.sh >/dev/null 2>&1
}

# no-widening table: the arm is `/tmp/?*|"${TMPDIR:-/tmp}"/?*` -- children of the base only, never a wider set.
# row <name> <want 0|1> <TMPDIR value> <path>
arm() { local p=$2; (unset TMPDIR; [ -n "$1" ] && TMPDIR=$1; case "$p" in /tmp/?*|"${TMPDIR:-/tmp}"/?*) exit 0 ;; *) exit 1 ;; esac); }
table() {
    local bad=0 n w t p rc
    while IFS='|' read -r n w t p; do
        arm "$t" "$p" && rc=0 || rc=1
        [ "$rc" = "$w" ] || { echo "FAIL row: $n (want $w got $rc)"; bad=1; }
    done <<'ROWS'
child of /tmp|0||/tmp/x
child of TMPDIR|0|/runner/t|/runner/t/x
TMPDIR itself is not a child|1|/runner/t|/runner/t
unrelated path|1|/runner/t|/etc/x
sibling prefix|1|/runner/t|/runner/tx/y
empty TMPDIR falls back to /tmp only|1||/etc/x
root TMPDIR does not widen|1|/|/etc/x
trailing-slash TMPDIR fails closed|1|/runner/t/|/runner/t/x
ROWS
    return "$bad"
}

# the ladder guard removes a scratch dir under a TMPDIR outside /tmp (e5: cleanup arm leaked it)
ladder_cleanup() {
    local d; d=$(outside_tmp); trap 'rm -rf "${d:?}"' RETURN
    mkdir "$d/scratch"
    TMPDIR=$d/scratch bash scripts/check_ladder_write_errors.sh >/dev/null 2>&1 || true
    [ -z "$(ls -A "$d/scratch")" ] || { echo "FAIL: ladder guard leaked a scratch dir under TMPDIR"; return 1; }
}

check() {
    local f bad=0
    for f in "${FILES[@]}"; do ok "$f" || { echo "FAIL: $f has no \${TMPDIR:-/tmp} arm"; bad=1; }; done
    return "$bad"
}

if [ "${1:-}" = "--mutants" ]; then
    check || exit 1
    table || exit 1
    ladder_cleanup || exit 1
    d=$(mktemp -d); trap 'rm -rf "${d:?}"' EXIT
    for f in "${FILES[@]}"; do
        sed 's#|"\${TMPDIR:-/tmp}"/#|#' "$f" > "$d/m.sh"
        cmp -s "$f" "$d/m.sh" && { echo "FAIL: mutant of $f did not change it"; exit 1; }
        if ok "$d/m.sh"; then echo "FAIL: mutant of $f survived"; exit 1; fi
        echo "mutant $f: killed"
    done
    exit 0
fi
check && table && behave && ladder_cleanup && echo "PASS: ${#FILES[@]} guards accept \${TMPDIR:-/tmp}"
