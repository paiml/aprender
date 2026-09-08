#!/usr/bin/env bash
# check_backend_registry.sh — R-0b (#3002, PMAT-1073, PP-066 claim 1 / C11):
# a backend DECISION in apr-cli reads the registry, never `cfg!(feature = …)`.
#
#   bash scripts/check_backend_registry.sh --static   # 0 clean · 1 a cfg read leaked · 2 env
#   bash scripts/check_backend_registry.sh --self-test # case table, both polarities
#
# THE RULE. `cfg!(any(feature = "cuda"|"wgpu"))` decides, at compile time, what
# a build can do. A binary built without cuda then reported success for
# `--gpu` and ran on CPU (aprender#2696) — the decision was `cfg!`, and the
# host it ran on never entered into it. R-0a made every backend a registry
# ENTRY (Ready / Unavailable(reason), with a source); R-0b routes every apr-cli
# backend decision AND every build-capability report through that registry.
# So under crates/apr-cli/src there is ZERO `cfg!(… feature = "cuda"|"wgpu")`
# outside registry.rs (the module that owns the crate's one reading of the
# feature set, inside its `#[cfg(feature = "inference")]` gate) and comments.
#
# NOT in scope: `cfg!(feature = "inference"|"training"|"cuda-batch")` — those
# are not backend kinds; the regex is anchored to cuda|wgpu.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROG=check_backend_registry
RE='cfg!\((any\()?[^)]*feature *= *"(cuda|wgpu)"'

scan() { # scan <dir> -> prints offending file:line (code only), excludes registry.rs and comment lines
    local dir=$1
    command -v rg >/dev/null 2>&1 || { printf '%s: ENV - ripgrep missing\n' "$PROG" >&2; return 2; }
    [ -d "$dir" ] || { printf '%s: ENV - %s missing\n' "$PROG" "$dir" >&2; return 2; }
    # -n line numbers; strip registry.rs; strip lines whose first non-space is // or *
    rg -n --no-heading -e "$RE" -g '*.rs' "$dir" 2>/dev/null \
        | grep -v -E '/registry\.rs:' \
        | grep -v -E ':[0-9]+: *(//|\*|///)' || true
}

if [ "${1:-}" = "--self-test" ]; then
    TD=$(mktemp -d "${TMPDIR:-/tmp}/bereg.XXXXXX")
    cleanup() { case "$TD" in *bereg.*) rm -rf -- "$TD" ;; esac; }
    trap cleanup EXIT
    mkdir -p "$TD/apr-cli/src"
    n=0; red=0
    row() { local want=$1 label=$2 rc=0; n=$((n+1)); shift 2; "$@" >"$TD/o.$n" 2>&1 || rc=$?; if [ "$rc" = "$want" ]; then printf 'ok    row %-2s rc=%s %s\n' "$n" "$rc" "$label"; else printf 'FAIL  row %-2s rc=%s (want %s) %s\n' "$n" "$rc" "$want" "$label"; sed 's/^/        /' "$TD/o.$n"; red=1; fi; }
    # clean tree
    printf 'fn f() { let _ = crate::registry::compiled("cuda"); }\n' > "$TD/apr-cli/src/a.rs"
    printf '//! a comment mentioning cfg!(feature = "cuda") teaches the old spelling\n' > "$TD/apr-cli/src/b.rs"
    printf 'pub fn compiled(k:&str)->bool{ cfg!(feature="cuda") || cfg!(feature="wgpu") }\n' > "$TD/apr-cli/src/registry.rs"
    printf 'fn ok(){ let _ = cfg!(feature="inference"); }\n' > "$TD/apr-cli/src/c.rs"
    row 0 "a tree that routes decisions through the registry is clean" bash "$0" --dir "$TD/apr-cli/src"
    # a leak
    printf 'fn bad(){ if cfg!(any(feature="cuda", feature="wgpu")) { } }\n' > "$TD/apr-cli/src/leak.rs"
    row 1 "one cfg!(feature=cuda|wgpu) backend read outside registry.rs is RED" bash "$0" --dir "$TD/apr-cli/src"
    rm -f "$TD/apr-cli/src/leak.rs"
    row 0 "removing the leak returns to clean" bash "$0" --dir "$TD/apr-cli/src"
    printf '%s/%s rows\n' "$((n-red))" "$n"; [ "$red" = 0 ] || exit 1; exit 0
fi

DIR="$ROOT/crates/apr-cli/src"
[ "${1:-}" = "--dir" ] && DIR="${2:?}"
[ "${1:-}" = "--static" ] || [ "${1:-}" = "--dir" ] || { printf 'usage: %s --static | --self-test\n' "$PROG" >&2; exit 2; }
hits=$(scan "$DIR") || { rc=$?; [ "$rc" = 2 ] && exit 2; }
if [ -n "$hits" ]; then
    printf 'FAIL  a backend decision reads cfg!(feature = "cuda"|"wgpu") outside registry.rs:\n%s\n' "$hits"
    printf '  route it through crate::registry (R-0b, #3002): resolve() / compiled() / build_has_accelerator().\n'
    exit 1
fi
printf 'PASS  no cfg!(feature = "cuda"|"wgpu") backend read in crates/apr-cli/src outside registry.rs\n'
