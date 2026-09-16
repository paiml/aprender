#!/usr/bin/env bash
# check_backend_registry.sh — R-0b (#3002, PMAT-1073, PP-066 claim 1 / C11):
# a backend DECISION in apr-cli reads the registry, never `cfg!(feature = …)`.
#
#   bash scripts/check_backend_registry.sh            # same as --static
#   bash scripts/check_backend_registry.sh --static   # 0 clean · 1 a cfg read leaked · 2 env
#   bash scripts/check_backend_registry.sh --dir DIR  # the same scan over DIR
#   bash scripts/check_backend_registry.sh --self-test # case table, both polarities
#
# THE RULE. `cfg!(any(feature = "cuda"|"wgpu"))` decides, at COMPILE time, what
# a build can do. A binary built without cuda then reported success for `--gpu`
# and ran on CPU (aprender#2696) — the decision was `cfg!`, and the HOST it ran
# on never entered into it. R-0a made every backend a registry ENTRY (Ready /
# Unavailable(reason), with a source); R-0b routes every apr-cli backend
# decision AND every build-capability report through that registry. So under
# crates/apr-cli/src there is ZERO `cfg!(… feature = "cuda"|"wgpu")` outside
# registry.rs (the one module that owns the crate's reading of the feature set,
# inside its `#[cfg(feature = "inference")]` gate) and outside comments.
#
# NOT IN SCOPE: `cfg!(feature = "inference"|"training"|"cuda-batch")` — those
# are not backend kinds; the regex is anchored to cuda|wgpu.
#
# WHY A BARE INVOCATION IS THE GATE, AND NOT AN ARGUMENT-WIRED WORKFLOW STEP.
# scripts/guard_tree.sh (ci.yml `guard-runner-labels`) runs every tracked
# `scripts/check_*.sh` — BARE, plus `--self-test` when `--help` advertises one.
# A guard whose gate hides behind a flag would be run bare by that dispatcher,
# print usage, exit 2, and be a RED row for a guard that never ran. So no-arg IS
# `--static`; `--static` stays as the explicit spelling the contract cites.
#
# WHY grep AND NOT ripgrep. An earlier draft used `rg` and exited 2 (ENV)
# without it. No `scripts/check_*.sh` in this repo invokes `rg`, no workflow
# provisions ripgrep, and guard-runner-labels runs on `[self-hosted, Linux,
# clean-room]` — so that self-test would have been an env-death inside
# guard_tree. The regex is ERE, so `grep -rnE` is the same scan with no new
# runner dependency (pr-3041-split-plan.md decision D4b).
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PROG=check_backend_registry
RE='cfg!\((any\()?[^)]*feature *= *"(cuda|wgpu)"'

usage() {
    printf 'usage: %s [--static] | --dir DIR | --self-test | --help\n' "$PROG"
    printf '  no argument is the same gate as --static.\n'
    printf '  --self-test runs the case table (both polarities).\n'
}

# scan DIR -- print `file:line:code` for every BACKEND cfg! read that is not in
# registry.rs and not a comment. rc 2 (ENV) when DIR is missing; rc 0 otherwise,
# hits or not, so the caller decides -- a scan that reports "no hits" and
# "could not look" with the same status is the vacuity this repo keeps paying for.
#
# NEVER `producer | grep -q`: under `pipefail` a SIGPIPEd producer reads as a
# failure and a real match can come back a false negative (aprender
# feedback_sigpipe_pipefail_false_red). Every filter below is a `<<<` on a
# variable, and every one is `|| x=''` because a grep that matches NOTHING
# exits 1 and would kill this script under `set -e`.
scan() {
    local dir=$1 raw='' kept='' code=''
    [ -d "$dir" ] || { printf '%s: ENV - %s missing\n' "$PROG" "$dir" >&2; return 2; }
    raw=$(grep -rnE --include='*.rs' -e "$RE" -- "$dir" 2>/dev/null) || raw=''
    [ -n "$raw" ] || return 0
    # registry.rs owns the crate's ONE reading of the feature set.
    kept=$(grep -vE '/registry\.rs:[0-9]+:' <<<"$raw") || kept=''
    [ -n "$kept" ] || return 0
    # a line whose first non-space is // or * (so also ///, //!) is prose.
    code=$(grep -vE '^[^:]*:[0-9]+:[[:space:]]*(//|\*)' <<<"$kept") || code=''
    if [ -n "$code" ]; then printf '%s\n' "$code"; fi
    return 0
}

# gate DIR -- 0 clean, 1 a backend cfg! read leaked, 2 env.
gate() {
    local dir=$1 hits='' rc=0
    hits=$(scan "$dir") || rc=$?
    [ "$rc" = 0 ] || return "$rc"
    if [ -n "$hits" ]; then
        printf 'FAIL  a backend decision reads cfg!(feature = "cuda"|"wgpu") outside registry.rs:\n%s\n' "$hits"
        printf '  route it through crate::registry (R-0b, #3002): resolve() / compiled() / build_has_accelerator().\n'
        return 1
    fi
    printf 'PASS  no cfg!(feature = "cuda"|"wgpu") backend read in %s outside registry.rs\n' "$dir"
    return 0
}

# The scratch tree is a GLOBAL, because the EXIT trap that removes it fires
# AFTER self_test has returned: a `local TD` is out of scope by then and
# `set -u` turns the cleanup into an "unbound variable" death -- which exits 1
# after the case table has already printed `10/10 rows`. A guard whose own
# teardown can invert its verdict is the defect class it exists to catch.
TD=""
# shellcheck disable=SC2317  # reached only through the EXIT trap
cleanup() { case "${TD:-}" in *bereg.*) rm -rf -- "${TD:?}" ;; esac; }

self_test() {
    local n=0 red=0
    TD=$(mktemp -d "${TMPDIR:-/tmp}/bereg.XXXXXX")
    trap cleanup EXIT
    mkdir -p "$TD/src"

    row() { # row WANT LABEL CMD...
        local want=$1 label=$2 rc=0
        n=$((n + 1)); shift 2
        "$@" >"$TD/o.$n" 2>&1 || rc=$?
        if [ "$rc" = "$want" ]; then
            printf 'ok    row %-2s rc=%s %s\n' "$n" "$rc" "$label"
        else
            printf 'FAIL  row %-2s rc=%s (want %s) %s\n' "$n" "$rc" "$want" "$label"
            sed 's/^/        /' "$TD/o.$n"; red=1
        fi
    }
    row_says() { # row_says WANT NEEDLE LABEL CMD... -- status AND what it named
        local want=$1 needle=$2 label=$3
        shift 3
        row "$want" "$label" "$@"
        if [ "$(grep -c -F -- "$needle" "$TD/o.$n")" = 0 ]; then
            printf 'FAIL  row %-2s did not name `%s` %s\n' "$n" "$needle" "$label"
            sed 's/^/        /' "$TD/o.$n"; red=1
        fi
    }

    # --- a synthetic tree, both polarities -------------------------------
    printf 'fn f() { let _ = crate::registry::compiled("cuda"); }\n' >"$TD/src/a.rs"
    printf '//! a comment mentioning cfg!(feature = "cuda") teaches the old spelling\n' >"$TD/src/b.rs"
    printf 'pub fn compiled(k:&str)->bool{ cfg!(feature="cuda") || cfg!(feature="wgpu") }\n' >"$TD/src/registry.rs"
    printf 'fn ok(){ let _ = cfg!(feature="inference"); }\n' >"$TD/src/c.rs"
    row 0 "a tree that routes decisions through the registry is clean" \
        bash "$0" --dir "$TD/src"
    printf 'fn bad(){ if cfg!(feature="cuda") { } }\n' >"$TD/src/leak.rs"
    row_says 1 'leak.rs' "one cfg!(feature=cuda) backend read outside registry.rs is RED, by name" \
        bash "$0" --dir "$TD/src"
    printf 'fn bad(){ if cfg!(any(feature="cuda", feature="wgpu")) { } }\n' >"$TD/src/leak.rs"
    row 1 "the any(...) spelling is the same read and is RED too" \
        bash "$0" --dir "$TD/src"
    rm -f "$TD/src/leak.rs"
    row 0 "removing the leak returns to clean" bash "$0" --dir "$TD/src"
    printf '    /// pass `cfg!(feature = "wgpu")` from the call site\n' >"$TD/src/doc.rs"
    row 0 "a doc comment quoting the old spelling is prose, not a read" \
        bash "$0" --dir "$TD/src"
    rm -f "$TD/src/doc.rs"

    # --- the modes themselves -------------------------------------------
    row 2 "a missing directory is ENV (2), never a silent PASS" \
        bash "$0" --dir "$TD/nope"
    row 2 "an unknown argument is refused, never run as the default gate" \
        bash "$0" --statik
    row_says 0 'self-test' "--help advertises the self-test guard_tree.sh probes for" \
        bash "$0" --help

    # --- THE REAL TREE, and the mutation that turns it RED ---------------
    # A synthetic tree proves the scanner. Only the real one proves the CLAIM,
    # and only a mutation of the real one proves the claim could fail.
    local REALSRC="$ROOT/crates/apr-cli/src"
    if [ -d "$REALSRC" ]; then
        cp -r -- "$REALSRC" "$TD/real"
        row 0 "crates/apr-cli/src itself is clean (this is what --static asserts)" \
            bash "$0" --dir "$TD/real"
        # MUTATION: put the pre-R-0b compile-time read back at accel.rs's
        # build_has_accelerator -- the exact line #2696 was decided by.
        if [ -f "$TD/real/accel.rs" ]; then
            printf 'fn mutant() -> bool { cfg!(any(feature = "cuda", feature = "wgpu")) }\n' \
                >>"$TD/real/accel.rs"
            row_says 1 'accel.rs' \
                "MUTATION: cfg!(any(feature=cuda,wgpu)) back in accel.rs turns --static RED, naming the file" \
                bash "$0" --dir "$TD/real"
        else
            printf 'FAIL  row --  accel.rs missing from the copied tree\n'; red=1
        fi
    else
        printf 'FAIL  row --  ENV %s missing\n' "$REALSRC"; red=1
    fi

    printf '%s/%s rows\n' "$((n - red))" "$n"
    [ "$red" = 0 ] || return 1
    return 0
}

# ARGUMENT VALIDATION BEFORE ANY WORK. A typo that falls through into the
# default gate tells a caller who believed they ran the case table that it
# PASSED (the shape check_no_claim_literals.sh grew this same guard for).
case "${1:-}" in
    '' | --static) gate "$ROOT/crates/apr-cli/src" ;;
    --dir)         gate "${2:?--dir needs a directory}" ;;
    --self-test)   self_test ;;
    -h | --help)   usage ;;
    *)             printf 'unknown arg: %s\n' "$1" >&2; usage >&2; exit 2 ;;
esac
