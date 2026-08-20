#!/usr/bin/env bash
#
# check_facade_compat.sh — the crates.io compatibility facades must still be a
# drop-in replacement for the crate names APR-MONO renamed.
#
# WHY THIS EXISTS (aprender#2546)
# -------------------------------
# `provable-contracts{,-macros,-cli}` were renamed to `aprender-contracts*`.
# The old names stayed on crates.io, frozen at 0.3.1, with ~10K downloads and
# at least one production pin. paiml/infra pinned `provable-contracts-cli =
# "0.3.1"`; it resolved cleanly and installed a `pv` sixty versions behind. No
# error, no warning. It surfaced only because 0.3.1 predates the
# `safety`/`liveness` proof-obligation kinds, so the guard read as BROKEN
# rather than OUT OF DATE.
#
# crates.io has no rename mechanism (rust-lang/crates.io#2902), so the names
# are carried forward as facades under crates/facades/. Hosting them HERE and
# not in the archived paiml/provable-contracts is the point: a shape change in
# `aprender-contracts` breaks the facade in the SAME CI run instead of rotting
# silently, which is the exact failure mode that produced the issue.
#
# WHAT IT CHECKS, AND WHY EACH HALF IS NECESSARY
# ----------------------------------------------
# STRUCTURE (scripts/lib/facade_facts.py, read from `cargo metadata`)
#   R1 lib/bin target names match the crate each facade fronts
#   R2 no facade is an aprender workspace member -- MEASURED, as a member:
#        warning: output filename collision at
#          <target>/debug/libprovable_contracts.rlib
#          = note: this may become a hard error in the future; see
#                  https://github.com/rust-lang/cargo/issues/6313
#      and the facade `pv` overwrites the real `pv` in the shared target dir
#   R3 the upstream version requirement equals the workspace version --
#      `cargo set-version` cannot reach an excluded manifest, so a release
#      would leave the facades pinned to a version that no longer exists
#   R4 the compat corpus is non-empty (a gate that passes on n=0 is a fail mode)
#   R5 crates/facades/Cargo.lock is current -- check_lockfile_current.sh resolves
#      the root workspace only and cannot see a second lockfile
#   R6 the 28 vendored 0.3.1 programs match their published sha256s
#
# COMPATIBILITY (rustc)
#   Identical export LISTS do not imply identical SIGNATURES. The 28 example
#   programs published INSIDE provable-contracts 0.3.1 are vendored verbatim
#   and 27 are compiled against the facade. They are real downstream consumer
#   code: they call into 20 re-exported modules by name and destructure the
#   return types, so a drifted signature is a compile error. The five 0.3.1
#   attribute macros are INVOKED through the macros facade, which a
#   `proc-macro = true` facade could not do at all.
#
# MUTATION CONTROL (always, not just in --self-test)
#   `--features __facade_probe_mutant` compiles an arm naming an item the
#   facade does not export. That build MUST fail. A compile gate that has only
#   ever been green is indistinguishable from one that never invoked rustc.
#
# CURRENCY
#   The facade `pv --version` must report the WORKSPACE version, not the
#   facade's own. That is the whole remedy for #2546: `cargo install
#   provable-contracts-cli` has to yield a current tool, not a shim.
#
# WHY BASH AND NOT pv
# -------------------
# `pv` is the contract CLI: its universe is contracts/**.yaml. The subject
# here is Cargo manifests and rustc compilation of a vendored corpus -- there
# is no contract to validate and no pv subcommand shape for "build this
# workspace". The pv-native half IS present:
# contracts/provable-contracts-facade-v1.yaml states the promise, is checked by
# `pv validate`, and binds its falsification tests to this script -- the same
# division of labour as scripts/check_contract_test_binding.sh, which is a bash
# guard whose job is to RUN pv.
#
#   bash scripts/check_facade_compat.sh              # check
#   bash scripts/check_facade_compat.sh --self-test  # case table, text-only
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FACTS="${REPO_ROOT}/scripts/lib/facade_facts.py"
CASES="${REPO_ROOT}/scripts/lib/facade_cases"
FACADE_WS="${REPO_ROOT}/crates/facades"

# Isolated from the aprender target dir on purpose. The facade lib and the
# crate it fronts share a lib name, and the facade bin shares the name `pv`;
# a shared directory lets one overwrite the other's uplifted artifact. A
# shadowed artifact is worse than a missing one -- edits look effective and
# change nothing.
# A SUBDIRECTORY of the ambient target dir, so it inherits whatever caching the
# host has arranged and is swept by `cargo clean`, while staying a distinct
# directory (no uplift collision). If cargo metadata cannot be read -- a lock
# held by a concurrent build has produced an empty document here -- the fallback
# is repo-local. Both candidates satisfy the property that matters, which is
# "not the aprender target dir"; only cache locality differs, so the fallback
# is safe rather than silent.
# ---------------------------------------------------------------------------
if [ "${1:-}" = "--self-test" ]; then
    # Must-match / must-not-match table for the structural checker. Fixtures are
    # committed JSON under scripts/lib/facade_cases/ rather than inline
    # heredocs: bashrs parses an embedded heredoc as shell.
    fails=0

    run_case() {  # name expect_rc root facade needle
        local name="$1" want="$2" root="$3" fac="$4" needle="$5" out rc
        out="$( python3 "$FACTS" "$CASES/$root" "$CASES/$fac" 2>&1 )"; rc=$?
        if [ "$rc" != "$want" ]; then
            printf 'FAIL  %s: exit %s, expected %s\n' "$name" "$rc" "$want"; fails=1; return
        fi
        if [ -n "$needle" ] && ! grep -q -- "$needle" <<< "$out"; then
            printf 'FAIL  %s: exit %s as expected but did not name %s\n' "$name" "$rc" "$needle"
            fails=1; return
        fi
        printf 'ok    %s\n' "$name"
    }

    run_case 'row 1 a conforming pair passes'              0 root_good.json facade_good.json ''
    run_case 'row 2 upstream version drift is REJECTED'    1 root_good.json facade_version_drift.json 'FAIL  R3'
    run_case 'row 3 a wrong lib name is REJECTED'          1 root_good.json facade_wrong_lib_name.json 'FAIL  R1'
    run_case 'row 4 facade as workspace member is REJECTED' 1 root_facade_is_member.json facade_good.json 'FAIL  R2'
    run_case 'row 5 an empty compat corpus is REJECTED'    1 root_good.json facade_empty_corpus.json 'FAIL  R4'

    # Row 6 probes the checker itself: the corpus floor must be a real floor.
    out="$( python3 "$FACTS" "$CASES/root_good.json" "$CASES/facade_good.json" 9999 2>&1 )"; rc=$?
    if [ "$rc" = 1 ] && grep -q 'FAIL  R4' <<< "$out"; then
        printf 'ok    row 6 corpus floor is honoured (min 9999 rejects a 27-target corpus)\n'
    else
        printf 'FAIL  row 6 corpus floor ignored: exit %s\n' "$rc"; fails=1
    fi

    [ "$fails" -eq 0 ] || { printf '\nSELF-TEST FAILED\n'; exit 1; }
    printf '\nSELF-TEST PASSED (6/6)\n'
    exit 0
fi

# ---------------------------------------------------------------------------
printf '=== crates.io compatibility facades (check_facade_compat.sh) ===\n\n'
rc=0

ROOT_MD="$(mktemp)"; FAC_MD="$(mktemp)"
trap 'rm -f "${ROOT_MD:?}" "${FAC_MD:?}"' EXIT
( cd "$REPO_ROOT" && cargo metadata --no-deps --format-version 1 ) > "$ROOT_MD" 2>/dev/null
( cd "$FACADE_WS" && cargo metadata --no-deps --format-version 1 ) > "$FAC_MD" 2>/dev/null
# VACUOUS guard: an empty or unparsable metadata document must be RED, never a
# silent pass over nothing. Copied from check_contract_test_binding.sh, where a
# missing measurement had to be made to fail rather than read as clean.
if [ ! -s "$ROOT_MD" ] || [ ! -s "$FAC_MD" ]; then
    printf 'VACUOUS: cargo metadata produced no document; nothing was checked.\n'
    exit 1
fi

TD="$( python3 "$FACTS" --target-dir "$ROOT_MD" )/facades"
printf 'facade target dir: %s\n\n' "$TD"

printf -- '--- STRUCTURE ------------------------------------------------------\n'
python3 "$FACTS" "$ROOT_MD" "$FAC_MD" || rc=1

# crates/facades/Cargo.lock is outside the aprender workspace, so
# check_lockfile_current.sh -- which resolves the root workspace only -- does
# not see it. Same defect, same instrument: `--locked` fails exactly when the
# lock and the manifests disagree, in about a second, with no build.
if ( cd "$FACADE_WS" && cargo metadata --format-version 1 --locked ) >/dev/null 2>&1; then
    printf 'ok    R5 crates/facades/Cargo.lock matches the facade manifests (--locked)\n'
else
    printf 'FAIL  R5 crates/facades/Cargo.lock is stale -- run `cargo metadata` in crates/facades\n'
    rc=1
fi

# R6. The vendored corpus is a fixed record of the published 0.3.1 API. If it
# can be edited, "0.3.1 code still compiles" degrades into "whatever is in this
# directory still compiles", which is a tautology. Note `cargo fmt` inside
# crates/facades WILL rewrite four of these files -- they predate this
# workspace's rustfmt settings -- so this is a live hazard, not a hypothetical.
CORPUS="${FACADE_WS}/provable-contracts/compat/0.3.1"
if ( cd "$CORPUS/examples" && sha256sum -c ../SHA256SUMS ) >/dev/null 2>&1; then
    printf 'ok    R6 all 28 vendored 0.3.1 programs match their published checksums\n'
else
    printf 'FAIL  R6 the vendored 0.3.1 corpus has been modified:\n'
    ( cd "$CORPUS/examples" && sha256sum -c ../SHA256SUMS 2>&1 | grep -v ': OK$' | sed 's/^/        /' )
    rc=1
fi

printf -- '\n--- COMPATIBILITY: 0.3.1 consumer code against the facade -----------\n'
if ( cd "$FACADE_WS" && cargo check --quiet --workspace --all-targets --target-dir "$TD" ) 2>&1 \
        | grep -vE '^(warning: [a-z-]+@|    Finished|\s*$)'; then :; fi
if ( cd "$FACADE_WS" && cargo check --quiet --workspace --all-targets --target-dir "$TD" ) >/dev/null 2>&1; then
    printf 'ok    27 vendored 0.3.1 examples + probes compile against the facades\n'
else
    printf 'FAIL  0.3.1 consumer code no longer compiles against the facades\n'; rc=1
fi

printf -- '\n--- BEHAVIOUR: the five 0.3.1 attribute macros still expand ---------\n'
if ( cd "$FACADE_WS" && cargo test --quiet --workspace --target-dir "$TD" ) >/dev/null 2>&1; then
    printf 'ok    compat_invoke + compat_probe pass\n'
else
    printf 'FAIL  compat targets do not pass -- run `cargo test` in crates/facades\n'; rc=1
fi

printf -- '\n--- MUTATION CONTROL: a broken re-export must be RED ----------------\n'
for pkg in provable-contracts provable-contracts-macros; do
    if ( cd "$FACADE_WS" && cargo check --quiet -p "$pkg" --test compat_probe \
            --features __facade_probe_mutant --target-dir "$TD" ) >/dev/null 2>&1; then
        printf 'FAIL  %s: the mutant arm COMPILED. The compile gate above proves nothing.\n' "$pkg"
        rc=1
    else
        printf 'ok    %s: naming an unexported item fails the build\n' "$pkg"
    fi
done

WS_VER="$(awk -F'\"' '/^version *=/{print $2; exit}' Cargo.toml)"
printf -- '\n--- CURRENCY: the facade installs the CURRENT pv --------------------\n'
WANT="$( python3 "$FACTS" --version-of "$ROOT_MD" aprender-contracts-cli )"
( cd "$FACADE_WS" && cargo build --quiet -p provable-contracts-cli --bin pv --target-dir "$TD" ) >/dev/null 2>&1
GOT="$( "$TD/debug/pv" --version 2>/dev/null )"
if [ "$GOT" = "pv $WANT" ]; then
    printf 'ok    facade `pv --version` reports `%s` -- the workspace version, not 0.4.0\n' "$GOT"
else
    printf 'FAIL  facade pv reported [%s], expected [pv %s]\n' "$GOT" "$WANT"; rc=1
fi


printf -- '\n--- PUBLISH ORDER: what building here does NOT prove ------------------\n'
# STAGED CHECK, same shape as pre-release Gate 5 (aprender#2543): it cannot pass
# before the cascade reaches the upstream crate, and saying so is the point.
#
# Everything above builds inside THIS workspace, where `upstream` resolves
# through its `path`. A published consumer resolves it from the REGISTRY. Those
# are different builds, and only the second is what `cargo install
# provable-contracts-cli` actually does.
#
# The CLI facade calls `aprender_contracts_cli::run()`. That symbol lives in a
# lib.rs added on THIS branch, so it is NOT in any already-published version.
# A facade whose constraint names an already-published version therefore cannot
# compile for a real consumer no matter how green this script is -- which is
# exactly the failure an adversarial review found, and exactly what building
# through the path dep hides.
for pkg in provable-contracts provable-contracts-macros provable-contracts-cli; do
    up_ver=$(awk -F'"' '/^upstream *=/{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+\.[0-9]+\.[0-9]+$/){print $i; exit}}' \
        "crates/facades/$pkg/Cargo.toml" 2>/dev/null)
    [ -n "$up_ver" ] || { printf 'ok    %s: no pinned upstream version\n' "$pkg"; continue; }
    if [ "$up_ver" = "$WS_VER" ]; then
        printf 'ok    %s: upstream pinned to the workspace version (%s)\n' "$pkg" "$up_ver"
    else
        printf 'FAIL  %s: upstream pinned to %s but workspace is %s -- a facade must\n' "$pkg" "$up_ver" "$WS_VER"
        printf '      track the version published from THIS tree, or it resolves an older\n'
        printf '      registry copy that lacks the symbols it calls.\n'
        rc=1
    fi
done
# LAST PUBLISHED WITHOUT THE LIB. aprender-contracts-cli 0.63.0 is on crates.io
# as a BIN-ONLY crate: `pub fn run()` lives in a lib.rs added after it shipped.
# So a facade constrained to 0.63.0 resolves a registry copy with no such symbol
# and cannot compile for a consumer, however green the path-dep build is.
#
# This is a STAGE verdict, not a defect -- the same shape as pre-release Gate 5
# (#2543), which also cannot pass before the version bump. Reporting it as a
# FAILURE here would block a PR over an ordering fact and train people to ignore
# the guard; reporting it as a PASS would be the lie. So: named, not fatal.
CLI_LAST_BINONLY="0.63.0"
cli_ver=$(awk -F'"' '/^upstream *=/{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+\.[0-9]+\.[0-9]+$/){print $i; exit}}' \
    crates/facades/provable-contracts-cli/Cargo.toml 2>/dev/null)
if [ "$cli_ver" = "$CLI_LAST_BINONLY" ]; then
    printf 'STAGE provable-contracts-cli is NOT YET PUBLISHABLE (expected at this stage).\n'
    printf '      It pins upstream %s, which is on crates.io BIN-ONLY -- `pub fn run()`\n' "$cli_ver"
    printf '      was added after that release, so a registry consumer gets a crate\n'
    printf '      without the symbol the facade calls. Publishable once the workspace\n'
    printf '      bumps and aprender-contracts-cli ships a version carrying the lib.\n'
    printf '      Do NOT publish this facade in the 0.63.0 cascade.\n'
else
    printf 'ok    provable-contracts-cli pins %s (> the bin-only %s) -- carries the lib\n' \
        "$cli_ver" "$CLI_LAST_BINONLY"
fi
printf 'note  PUBLISH ORDER IS A HARD CONSTRAINT, not a preference:\n'
printf '        1. aprender-contracts, -macros, -cli   (carry the lib + symbols)\n'
printf '        2. provable-contracts{,-macros,-cli}   (the facades, which resolve #1\n'
printf '           from the registry)\n'
printf '      Publishing a facade before its upstream yields a crate that cannot\n'
printf '      compile for anyone. This script CANNOT verify step 2 offline -- it has\n'
printf '      no registry to resolve against -- so treat a green run here as\n'
printf '      "structurally correct", never as "publishable".\n'

printf '\n'
if [ "$rc" -eq 0 ]; then
    printf 'PASS  the facades are still a drop-in for provable-contracts 0.3.1\n'
else
    printf 'FAIL  see rows above\n'
fi
exit "$rc"
