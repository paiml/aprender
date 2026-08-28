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
#   R7 NO facade declares a `[[bin]]` -- aprender#2558, below
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
# SURFACE / SIGNPOST / ROUTES / CURRENCY  (rewritten by aprender#2558)
#   This block used to build the facade's own `pv` and assert `pv --version`
#   equalled the workspace version. The CLI facade no longer ships a binary, so
#   that promise is void -- FOUR crates declared a bin named `pv` (the crates.io
#   pipe viewer, /usr/bin/pv, aprender-contracts-cli, and this facade), all
#   writing ~/.cargo/bin/pv. `cargo install` FAILS CLOSED on that collision
#   (exit 101, first binary survives), so a duplicate BLOCKS an install. At 463
#   downloads against the library's 57K, the CLI facade is the one that yields.
#
#   The promise is REPLACED, not deleted -- deleting a check to make a change
#   pass is a move this repo has already regretted twice:
#     C1 SURFACE   `cargo build --bin pv` in the facade workspace must FAIL,
#                  with that exact cargo message and not some other error
#     C2 SIGNPOST  description, README, lib.rs and build.rs must each name both
#                  working routes, and the lib must carry a #[deprecated]
#     C3 ROUTES    those routes must be REAL: aprender-contracts-cli declares
#                  [[bin]] name = "pv", and apr-cli declares the `Pv` subcommand.
#                  Without C3 the crate is a well-worded dead end.
#     CURRENCY     the crate the redirect points at is at the workspace version.
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
# ---------------------------------------------------------------------------
# cargo exits non-zero for two categories of reason, and only one of them is
# about the code under test. aprender#2574 taught this for the `cargo test`
# block below -- "the reader could not tell a signature drift from an OOM-killed
# rustc" -- but the fix stopped at that one block and left this residual, which
# is the shape a published-artifact audit keeps finding: the earlier fix was
# real and incomplete.
#
# On 2026-08-27 the residual came due. rustc was un-spawnable on a clean-room
# runner (`could not execute process ... (os error 2)`) while compiling
# `equivalent` -- a crate that cannot fail to compile -- and this gate reported
# `0.3.1 consumer code no longer compiles against the facades`. `gate`
# hard-requires guard-runner-labels, so a runner fault presented as a facade
# regression and blocked every open PR behind a phantom.
#
# ENV still exits 1. A gate that goes green on "we could not tell" is the
# defect class this repo names most often. The claim just has to be true.
classify_cargo_failure() {  # $1=logfile -> ENV | CODE
    # Anchored on cargo's OWN framing of a spawn/IO/resource fault, never on a
    # bare phrase: `No such file or directory` also appears in the legitimate
    # rustc diagnostic for a source file the repo really is missing, and that
    # is CODE. See row C7.
    if grep -qE 'could not execute process|could not parse/generate dep info|No space left on device|signal: (9|15)|failed to acquire package cache lock|error: failed to download|Connection refused|Temporary failure in name resolution' "$1"; then
        printf 'ENV\n'
    else
        printf 'CODE\n'
    fi
}

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

    # aprender#2558. Row 7 is the one that would have caught the state this
    # branch started in: a facade declaring `[[bin]] name = "pv"` alongside
    # aprender-contracts-cli's. Row 8 is its companion — the signpost facade
    # must not re-acquire the upstream dependency, because depending on a
    # BIN-ONLY published crate is the E0433 that made the bin form
    # unpublishable in the first place (#2553).
    run_case 'row 7 a facade declaring a [[bin]] is REJECTED' 1 root_good.json facade_cli_declares_bin.json 'FAIL  R7'
    run_case 'row 8 the signpost facade depending on upstream is REJECTED' 1 root_good.json facade_cli_depends_upstream.json 'FAIL  R3'

    # ---- Rows C1..C7: the failure CLASSIFIER -------------------------------
    # Rows 1-8 above probe the python structural checker only. The classifier
    # is a NEW surface, so it gets its own must-match/must-not-match table
    # rather than inheriting rows 1-8's green: extending a guard's scope
    # requires re-mutating in the new scope, the old proof does not transfer.
    # Guard regexes in this repo have been wrong five times; every one was
    # caught by a table like this and none by review.
    run_classify() {  # name want fixture
        local name="$1" want="$2" got
        got="$( classify_cargo_failure "$CASES/$3" )"
        if [ "$got" = "$want" ]; then
            printf 'ok    %s -> %s\n' "$name" "$got"
        else
            printf 'FAIL  %s: classified %s, expected %s\n' "$name" "$got" "$want"; fails=1
        fi
    }
    run_classify 'C1 rustc un-spawnable (the 2026-08-27 outage)' ENV  log_env_rustc_missing.txt
    run_classify 'C2 ENOSPC via dep-info'                        ENV  log_env_enospc.txt
    run_classify 'C3 OOM-killed rustc (signal 9)'                ENV  log_env_oom_kill.txt
    run_classify 'C4 package cache lock unavailable'             ENV  log_env_cache_lock.txt
    run_classify 'C5 a genuine unresolved import'                CODE log_code_unresolved_import.txt
    run_classify 'C6 a genuine missing method'                   CODE log_code_no_method.txt
    # C7 is the discrimination case and the reason the pattern is anchored on
    # cargo's spawn framing rather than on the bare phrase: rustc says
    # "No such file or directory (os error 2)" for a source file the repo is
    # actually missing, and that is a real defect the gate must still report.
    run_classify 'C7 missing SOURCE file is CODE, not ENV'       CODE log_code_missing_source.txt

    [ "$fails" -eq 0 ] || { printf '\nSELF-TEST FAILED\n'; exit 1; }
    printf '\nSELF-TEST PASSED (8/8 structural + 7/7 classifier)\n'
    exit 0
fi

# ---------------------------------------------------------------------------
printf '=== crates.io compatibility facades (check_facade_compat.sh) ===\n\n'
rc=0

ROOT_MD="$(mktemp)"; FAC_MD="$(mktemp)"; BINLOG="$(mktemp)"; TESTLOG="$(mktemp)"
CHECKLOG="$(mktemp)"
trap 'rm -f "${ROOT_MD:?}" "${FAC_MD:?}" "${BINLOG:?}" "${TESTLOG:?}" "${CHECKLOG:?}"' EXIT
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
# ONE invocation, exactly as the block below. This ran cargo TWICE: the first
# for display, the second -- output to /dev/null -- for the verdict. So the text
# shown to the reader was produced by a different run, under a different machine
# state, than the one that decided. #2574 states the rule six lines down; this
# block predated it. `rc=$?` is read directly and never through a pipe.
if ( cd "$FACADE_WS" && cargo check --quiet --workspace --all-targets --target-dir "$TD" ) > "$CHECKLOG" 2>&1; then
    printf 'ok    27 vendored 0.3.1 examples + probes compile against the facades\n'
elif [ "$( classify_cargo_failure "$CHECKLOG" )" = 'ENV' ]; then
    printf 'ENV   cargo check could not run to a verdict on this host -- this is a\n'
    printf '      runner fault, NOT evidence that the facades regressed. Triage the\n'
    printf '      runner (toolchain, disk, OOM), then re-run. cargo output (%s lines):\n' \
        "$( wc -l < "$CHECKLOG" | tr -d ' ' )"
    sed 's/^/      | /' "$CHECKLOG"
    rc=1
else
    printf 'FAIL  0.3.1 consumer code no longer compiles against the facades\n'
    printf '      cargo output follows (%s lines):\n' "$( wc -l < "$CHECKLOG" | tr -d ' ' )"
    sed 's/^/      | /' "$CHECKLOG"
    rc=1
fi

printf -- '\n--- BEHAVIOUR: the five 0.3.1 attribute macros still expand ---------\n'
# The output goes to a FILE, not to /dev/null, and the failure branch prints it.
# aprender#2574: this line discarded stdout AND stderr, so when it went red the
# whole gate emitted one context-free line -- `FAIL compat targets do not pass`
# -- and the reader could not tell a signature drift from an OOM-killed rustc
# without re-running the build by hand on a box under a different load. A gate
# that fails without emitting its diagnostic costs more time than it saves.
# One invocation only: re-running cargo to "get the output" would re-run the
# tests under a different machine state than the one that decided the verdict.
if ( cd "$FACADE_WS" && cargo test --quiet --workspace --target-dir "$TD" ) > "$TESTLOG" 2>&1; then
    printf 'ok    compat_invoke + compat_probe pass\n'
else
    printf 'FAIL  compat targets do not pass -- run `cargo test` in crates/facades\n'
    printf '      cargo output follows (%s lines):\n' "$( wc -l < "$TESTLOG" | tr -d ' ' )"
    sed 's/^/      | /' "$TESTLOG"
    rc=1
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

# --------------------------------------------------------------------------
# WHAT REPLACED THE CURRENCY CHECK, AND WHY IT IS NOT A DELETION
# --------------------------------------------------------------------------
# Until aprender#2558 this block built `-p provable-contracts-cli --bin pv` and
# asserted `pv --version` equalled the workspace version. Its promise was
# "`cargo install provable-contracts-cli` yields a CURRENT tool, not a shim".
#
# That promise is now void because the CLI facade deliberately ships NO binary:
# FOUR crates declared a bin named `pv` -- the crates.io pipe viewer, /usr/bin/pv,
# aprender-contracts-cli, and this facade. MEASURED: `cargo install` FAILS CLOSED
# on that collision (exit 101, first binary survives), so a second claimant BLOCKS
# the install rather than clobbering it -- which is worse for the facade's own
# purpose, since it obstructs the upgrade it exists to enable. The 463-download
# facade is the one that yields.
#
# Deleting the block would be the move this repo has made twice and regretted.
# So the promise is REPLACED, not dropped. The new promise is weaker but it is a
# real promise and it is the one users actually need:
#
#   `cargo install provable-contracts-cli` now FAILS -- and the failure is
#   SIGNPOSTED. Every surface a stranded user can see names a route that works.
#
# Three rows, each falsifiable:
#   C1 the facade really ships no binary  (behavioural: `--bin pv` must FAIL)
#   C2 every signpost surface names both routes  (text, four files)
#   C3 the routes named are REAL  (aprender-contracts-cli declares bin `pv`;
#      apr-cli declares the `Pv` subcommand)
# C3 is the one that stops this degenerating into a nicely-worded dead end.
printf -- '\n--- SURFACE: the CLI facade ships NO binary (aprender#2558) ----------\n'
if ( cd "$FACADE_WS" && cargo build --quiet -p provable-contracts-cli --bin pv \
        --target-dir "$TD" ) > "$BINLOG" 2>&1; then
    printf 'FAIL  C1 the facade still BUILDS a bin named `pv`. It must not: that name is\n'
    printf '      claimed by aprender-contracts-cli, by the crates.io pipe viewer and by\n'
    printf '      /usr/bin/pv. `cargo install` FAILS CLOSED on that collision (exit 101),\n'
    printf '      so a second claimant BLOCKS the install rather than clobbering it.\n'
    rc=1
elif grep -q 'no bin target named `pv`' "$BINLOG"; then
    printf 'ok    C1 `cargo build --bin pv` fails with "no bin target named `pv`"\n'
else
    # A build that failed for some OTHER reason would let C1 pass on a
    # compile error and prove nothing about the bin surface.
    printf 'FAIL  C1 the build failed, but NOT with "no bin target named `pv`" --\n'
    printf '      this row proves nothing until that is fixed:\n'
    sed 's/^/        /' "$BINLOG" | head -12
    rc=1
fi

printf -- '\n--- SIGNPOST: a failed install must not be a dead end ----------------\n'
# 463 downloads/month land on `there are no binaries to install`. Each of these
# four surfaces is somewhere such a user can plausibly look, so each must carry
# the redirect. Checked as text because that is what the user reads.
CLI_FACADE="${REPO_ROOT}/crates/facades/provable-contracts-cli"
for f in Cargo.toml README.md src/lib.rs build.rs; do
    miss=""
    grep -q 'cargo install aprender-contracts-cli' "$CLI_FACADE/$f" || miss="$miss cargo-install-route"
    grep -q 'apr pv' "$CLI_FACADE/$f" || miss="$miss apr-pv-route"
    if [ -z "$miss" ]; then
        printf 'ok    C2 %s names both routes\n' "$f"
    else
        printf 'FAIL  C2 provable-contracts-cli/%s is missing:%s\n' "$f" "$miss"; rc=1
    fi
done
# The `#[deprecated]` note specifically -- a doc comment is not a compiler
# diagnostic, and the note is the only signpost that reaches a consumer who
# reads nothing.
if grep -q '#\[deprecated' "$REPO_ROOT/crates/facades/provable-contracts-cli/src/lib.rs"; then
    printf 'ok    C2 the lib carries a #[deprecated] item, so a caller is warned by rustc\n'
else
    printf 'FAIL  C2 no #[deprecated] in the CLI facade lib -- the only signpost a\n'
    printf '      non-reader receives is gone\n'; rc=1
fi

printf -- '\n--- ROUTES ARE REAL: the redirect must point somewhere ---------------\n'
# C3. Without this the crate is a well-worded dead end: it would keep passing C2
# while naming a tool that no longer exists. Read from cargo metadata and the
# apr command enum, not from prose.
if python3 "$FACTS" --has-bin "$ROOT_MD" aprender-contracts-cli pv; then
    printf 'ok    C3 route 1 is real: aprender-contracts-cli declares [[bin]] name = "pv"\n'
else
    printf 'FAIL  C3 route 1 is DEAD: aprender-contracts-cli declares no bin named `pv`,\n'
    printf '      but the facade tells 463 downloads/month to install it\n'; rc=1
fi
if grep -q 'Pv(aprender_contracts_cli::cli::Commands)' \
        "$REPO_ROOT/crates/apr-cli/src/commands_enum.rs"; then
    printf 'ok    C3 route 2 is real: apr-cli declares the `Pv` subcommand (`apr pv`)\n'
else
    printf 'FAIL  C3 route 2 is DEAD: no `Pv` variant in crates/apr-cli/src/commands_enum.rs,\n'
    printf '      but the facade names `apr pv` as a replacement\n'; rc=1
fi

# CURRENCY did not disappear -- it moved to the crate that now owns the name.
printf -- '\n--- CURRENCY: the tool the facade points AT is the current one -------\n'
WANT="$( python3 "$FACTS" --version-of "$ROOT_MD" aprender-contracts-cli )"
if [ "$WANT" = "$WS_VER" ]; then
    printf 'ok    aprender-contracts-cli is at %s, the workspace version\n' "$WANT"
else
    printf 'FAIL  aprender-contracts-cli is at %s but the workspace is at %s -- the\n' "$WANT" "$WS_VER"
    printf '      redirect would install a version this tree never published\n'; rc=1
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
# THE STAGE BLOCKER THIS BRANCH USED TO CARRY, AND WHY IT IS GONE
# ---------------------------------------------------------------
# It read: "provable-contracts-cli is NOT YET PUBLISHABLE. It pins upstream
# 0.63.0, which is on crates.io BIN-ONLY -- `pub fn run()` was added after that
# release, so a registry consumer gets a crate without the symbol the facade
# calls." That was correct. VERIFIED against the real registry, 2026-08-21:
#
#   $ curl -s https://crates.io/api/v1/crates/aprender-contracts-cli/0.63.0
#     → num 0.63.0  has_lib False  bin_names ['pv']  yanked False
#   $ cargo check   # scratch crate, dep `= "0.63.0"` resolved from the REGISTRY
#     error[E0433]: cannot find module or crate `upstream` in this scope   rc=101
#
# Going lib-only dissolved it rather than waiting it out: the facade calls
# run() from nowhere and depends on aprender-contracts-cli not at all, so there
# is no symbol to be missing and no version to be too old. Row R3 (SIGNPOST)
# above is what keeps that true -- it FAILS if the dependency comes back.
#
# The row is kept, inverted, as a REGRESSION DETECTOR. A dependency reappearing
# on this crate would silently restore the blocker, and a deleted check would
# not notice.
cli_ver=$(awk -F'"' '/^upstream *=/{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+\.[0-9]+\.[0-9]+$/){print $i; exit}}' \
    crates/facades/provable-contracts-cli/Cargo.toml 2>/dev/null)
if [ -z "$cli_ver" ]; then
    printf 'ok    provable-contracts-cli has NO upstream pin -- it is a lib-only signpost\n'
    printf '      with zero dependencies, so it is publishable at ANY point in the\n'
    printf '      cascade. The bin-only-0.63.0 STAGE blocker (#2553) is dissolved, not\n'
    printf '      deferred.\n'
else
    printf 'FAIL  provable-contracts-cli pins upstream %s. It must not depend on\n' "$cli_ver"
    printf '      aprender-contracts-cli at all: that crate is published BIN-ONLY at\n'
    printf '      0.63.0 (has_lib false), so a registry consumer gets E0433 and the\n'
    printf '      facade cannot compile for anyone. See aprender#2553/#2558.\n'
    rc=1
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
