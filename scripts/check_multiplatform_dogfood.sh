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
rc=0

printf -- '--- multi-platform dogfood receipts for %s -------------------------\n' "$VERSION"

# ANTI-VACUITY. An empty matrix would make every check below pass over nothing,
# which is exactly how a gate ends up reporting success while discriminating
# nothing. Assert the universe is non-trivial before trusting any verdict from it.
n_hosts=$(printf '%s\n' $HOSTS | grep -c .)
if [ "$n_hosts" -lt 4 ]; then
    printf 'FAIL  the platform matrix has %s host(s); at least 4 are required.\n' "$n_hosts"
    printf '      A shrinking matrix silently narrows what "verified" means.\n'
    exit 1
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
done

if [ "$rc" -eq 0 ]; then
    printf '\nPASS  all %s platforms carry a receipt for %s.\n' "$n_hosts" "$VERSION"
else
    printf '\nFAIL  see rows above. Record receipts under %s/ as <host>.json with at\n' "$DIR"
    printf '      least: {"host","arch","version_tested","date","install_rc"}.\n'
fi
exit "$rc"
