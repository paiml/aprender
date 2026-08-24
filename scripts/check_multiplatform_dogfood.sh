#!/usr/bin/env bash
# Every release must be dogfooded on EVERY supported platform, not just the one
# the release engineer happens to be sitting at.
#
# WHY THIS EXISTS. The 0.64.0 pre-release sweep ran `cargo install aprender` on
# four hosts for the first time. The published crate had NEVER been verified on
# either arm64 platform. What that bought, in one afternoon:
#
#   aprender#2567  Q4_K GEMV -- the hottest kernel in quantized inference -- has
#                  ZERO aarch64 SIMD, and `matmul_q4k_f32_parallel` on non-x86 is
#                  a direct call to the SERIAL scalar routine. Numbers correct,
#                  speed wrong, so no correctness gate could ever have caught it.
#   aprender#2568  The OOM guard reads /proc/meminfo and `.unwrap_or(u64::MAX)`,
#                  so on macOS the threshold is ~12.8 EXABYTES and the guard can
#                  never fire. Its only test self-skips with
#                  `cfg!(target_os = "linux")` -- the platform where it is broken.
#   aprender#2572  `block v0.1.6` faces future-rustc rejection and sits under
#                  wgpu->metal, the ONLY GPU backend macOS has. Entirely absent
#                  from the Linux dependency graph.
#
# Every one of those is invisible from a single host BY CONSTRUCTION. That is the
# argument: this is not diligence theatre, it is the only way to see this class.
#
# WHAT THIS GATE CHECKS. Not that someone ran a sweep -- that a dated RECEIPT
# exists for each host, for the version being cut. A receipt is evidence; a
# checklist tick is not.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

# The supported platform matrix. A host is here because it is a DISTINCT
# combination of ISA, OS and accelerator -- not because we happen to own it.
#   lambda      x86_64 Linux  + RTX 4090 (sm_89)     consumer x86, AVX2
#   intel       x86_64 Linux  + no discrete GPU      Xeon, AVX-512 + VNNI
#   gx10        aarch64 Linux + GB10 (sm_121)        ARM server, unified memory
#   mini        arm64 macOS   + Metal                Apple silicon, no /proc, APFS
HOSTS="lambda intel gx10 mini"

VERSION="$(awk -F'"' '/^version *=/{print $2; exit}' Cargo.toml)"
DIR="evidence/dogfood/$VERSION"
# ONE validator, shared with scripts/check_bench_receipt.sh. Two readers of one
# schema is the divergence class #2640 exists to close.
REPO_BENCH_VALIDATOR="scripts/lib/bench_receipt.py"
rc=0

printf -- '--- multi-platform dogfood receipts for %s -------------------------\n' "$VERSION"

# ANTI-VACUITY, TWO LAYERS. The first layer alone was UNSOUND and the hole was
# found by adversarial review before this gate ever landed:
#
#   `HOSTS=...` and the floor `-lt 4` were BOTH literals in THIS file, so ONE
#   commit editing both to two exited 0 -- measured, rc=0 with the matrix
#   silently halved. The original mutation evidence proved the floor FIRES at 2;
#   it never proved the floor could not be LOWERED. That is the same defect the
#   competitive-parity ledger spent five rounds on:
#
#       ANY STATE THE AUTHOR WRITES AND THE GATE READS CAN BE MOVED IN THE SAME
#       COMMIT.
#
# Layer 2 is the fix, and it is the parity resolution: the comparand lives on
# PROTECTED main. `main` requires a PR, review and passing CI to change, so the
# host list as it exists at origin/main is the one prior state this branch
# cannot rewrite in its own PR. The matrix may GROW freely and may never SHRINK.
n_hosts=$(printf '%s\n' $HOSTS | grep -c .)

# Layer 1: an absolute floor. Weak on its own (editable here), kept because it
# is the only thing that works during bootstrap, when main has no copy yet.
if [ "$n_hosts" -lt 4 ]; then
    printf 'FAIL  the platform matrix has %s host(s); at least 4 are required.\n' "$n_hosts"
    printf '      A shrinking matrix silently narrows what "verified" means.\n'
    exit 1
fi

# Layer 2: the matrix at origin/main is the floor, and this file cannot move it.
main_hosts=$(git show origin/main:scripts/check_multiplatform_dogfood.sh 2>/dev/null \
             | sed -n 's/^HOSTS="\(.*\)"$/\1/p' | head -1)
if [ -z "$main_hosts" ]; then
    # BOOTSTRAP, and it is self-limiting rather than renewable: reachable only
    # while this script does not exist at the protected ref. Once it lands, this
    # branch is unreachable forever. A renewable bootstrap is the fifth hat of
    # `registry: true` and is exactly what parity round 5 was caught on.
    printf '!     BOOTSTRAP: this gate is not yet on origin/main, so the matrix\n'
    printf '      has no protected floor this run. It gains one the moment this\n'
    printf '      script lands.\n'
else
    missing=""
    for h in $main_hosts; do
        case " $HOSTS " in *" $h "*) ;; *) missing="$missing $h" ;; esac
    done
    if [ -n "$missing" ]; then
        printf 'FAIL  the matrix DROPPED host(s) present at origin/main:%s\n' "$missing"
        printf '      The host list at the protected ref is the floor. Editing it in\n'
        printf '      the same commit that reads it is the defect this layer exists\n'
        printf '      to close -- growing the matrix is free, shrinking it is not.\n'
        exit 1
    fi
    printf 'ok    matrix covers every host present at origin/main (%s)\n' \
        "$(printf '%s\n' $main_hosts | grep -c .)"
fi

for h in $HOSTS; do
    f="$DIR/$h.json"
    if [ ! -f "$f" ]; then
        printf 'FAIL  %-7s no receipt at %s\n' "$h" "$f"
        printf '        run the dogfood on %s and record its receipt before cutting.\n' "$h"
        rc=1
        continue
    fi

    got_ver=$(python3 -c "import json,sys;print(json.load(open('$f')).get('version_tested',''))" 2>/dev/null)
    inst=$(python3 -c "import json,sys;print(json.load(open('$f')).get('install_rc','MISSING'))" 2>/dev/null)
    when=$(python3 -c "import json,sys;print(json.load(open('$f')).get('date',''))" 2>/dev/null)

    if [ "$got_ver" != "$VERSION" ]; then
        printf 'FAIL  %-7s receipt is for %s, this cut is %s -- STALE\n' "$h" "${got_ver:-<none>}" "$VERSION"
        printf '        a receipt from a previous release says nothing about this one.\n'
        rc=1
    elif [ "$inst" != "0" ]; then
        printf 'FAIL  %-7s install_rc=%s -- `cargo install aprender` did not succeed\n' "$h" "$inst"
        rc=1
    else
        printf 'ok    %-7s %s verified %s (install rc=0)\n' "$h" "$VERSION" "${when:-undated}"
    fi

    # ── the bench block (PARITY-003, aprender#2670) ────────────────────────
    #
    # CPU-CLASS, apr-vs-apr, NO COMPARATOR — and that is a decision, not an
    # omission. `cargo install aprender` builds CPU-only on every host in this
    # matrix (crates/apr-cli/Cargo.toml `default` carries no cuda, no wgpu),
    # while llama.cpp runs CUDA on lambda/gx10 and Metal on mini. A ratio here
    # would read ~0.05-0.10, nobody would red a release over it (correctly),
    # the row would go EXISTENCE-ONLY and the threshold would never arm — the
    # exact shape this gate itself had before #2658. The tree already documents
    # that collapse at crates/apr-cli/src/dispatch.rs:165.
    #
    # An apr-vs-apr self-ratchet catches OUR regressions, which is the goal,
    # without inventing a number nobody will act on. The comparator ratio lives
    # in the pre-publish phase where --features cuda exists (#2677).
    #
    # ABSENT is not FAIL yet: the block arrives with the first release cut
    # after this lands, and a gate that fails for a version that predates it is
    # a gate nobody can satisfy. It is REPORTed so the absence is visible.
    if ! python3 "$REPO_BENCH_VALIDATOR" --has-bench "$f" >/dev/null 2>&1; then
        printf 'REPORT %-6s no bench block yet — arrives with the first cut after #2670\n' "$h"
    elif python3 "$REPO_BENCH_VALIDATOR" --bench "$f" >/dev/null 2>&1; then
        bmed=$(python3 "$REPO_BENCH_VALIDATOR" --bench-median "$f" 2>/dev/null)
        printf 'ok    %-7s bench: median %s ms, CPU-class self-ratchet\n' "$h" "${bmed:-?}"
    else
        printf 'FAIL  %-7s bench block present but INVALID:\n' "$h"
        python3 "$REPO_BENCH_VALIDATOR" --bench "$f" 2>&1 | sed 's/^/          /'
        rc=1
    fi
done

if [ "$rc" -eq 0 ]; then
    printf '\nPASS  all %s platforms carry a receipt for %s.\n' "$n_hosts" "$VERSION"
else
    printf '\nFAIL  see rows above. Record receipts under %s/ as <host>.json with at\n' "$DIR"
    printf '      least: {"host","arch","version_tested","date","install_rc"}.\n'
fi
exit "$rc"
