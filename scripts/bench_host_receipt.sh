#!/usr/bin/env bash
#
# bench_host_receipt.sh — add the CPU-class bench block to this host's receipt
# (PARITY-003, aprender#2670).
#
# RUN THIS ON THE HOST, AFTER `cargo install aprender`, in the same sitting as
# install_rc. It measures the PUBLISHED artifact, which is the only thing a
# post-publish receipt may speak about.
#
# CPU-class, apr-vs-apr, NO COMPARATOR. `cargo install aprender` builds CPU-only
# on every host in the matrix while llama.cpp runs CUDA or Metal, so a ratio
# here is uncomputable rather than merely unwise: it reads ~0.05-0.10 and the
# threshold never arms. The comparator ratio lives pre-publish (#2677).
#
# NEVER PATH for the binary under test: a bare `apr` on the release host
# resolved to a 49-day-old 0.60.0 during the 0.64.0 sweep, shadowing a fresh
# install, because ~/.local/bin precedes ~/.cargo/bin. $APR_UNDER_TEST or the
# cargo root, and the receipt records which.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

HOST="${1:-}"
MODEL="${2:-}"
if [ -z "$HOST" ] || [ -z "$MODEL" ]; then
    printf 'usage: bench_host_receipt.sh <host> <model-path>\n' >&2
    printf '  host  one of the names in check_multiplatform_dogfood.sh HOSTS\n' >&2
    exit 2
fi

VERSION=$(awk -F'"' '/^version *=/{print $2; exit}' Cargo.toml)
RECEIPT="evidence/dogfood/$VERSION/$HOST.json"
[ -f "$RECEIPT" ] || { printf 'FAIL  no receipt at %s — run the install sweep first\n' "$RECEIPT" >&2; exit 1; }
[ -f "$MODEL" ]   || { printf 'FAIL  no model at %s\n' "$MODEL" >&2; exit 1; }

APR="${APR_UNDER_TEST:-}"
if [ -z "$APR" ]; then
    APR="$(cargo install --list 2>/dev/null >/dev/null; printf '%s' "${CARGO_HOME:-$HOME/.cargo}/bin/apr")"
fi
[ -x "$APR" ] || { printf 'FAIL  no installed apr at %s\n' "$APR" >&2; exit 1; }

printf 'measuring %s on %s with %s\n' "$("$APR" --version 2>&1 | head -1)" "$HOST" "$MODEL"

tmp=$(mktemp) || exit 2
trap 'rm -f "${tmp:?}"' EXIT
if ! "$APR" bench "$MODEL" --json > "$tmp" 2>/dev/null; then
    printf 'FAIL  apr bench exited non-zero; NOT writing a bench block.\n' >&2
    printf '      A receipt that records a failed measurement as a measurement is\n' >&2
    printf '      the fabricated-measurement shape (F12).\n' >&2
    exit 1
fi

# The bench JSON must validate BEFORE it is written into the receipt. A block
# that could not stand alone must not be laundered by nesting.
if ! python3 scripts/lib/bench_receipt.py "$tmp" >/dev/null 2>&1; then
    printf 'FAIL  apr bench output does not validate:\n' >&2
    python3 scripts/lib/bench_receipt.py "$tmp" 2>&1 | sed 's/^/      /' >&2
    exit 1
fi

RECEIPT="$RECEIPT" BENCH="$tmp" python3 - <<'PY'
import json, os
receipt_path = os.environ["RECEIPT"]
with open(receipt_path, encoding="utf-8") as h:
    receipt = json.load(h)
with open(os.environ["BENCH"], encoding="utf-8") as h:
    receipt["bench"] = json.load(h)
with open(receipt_path, "w", encoding="utf-8") as h:
    json.dump(receipt, h, indent=2)
    h.write("\n")
print("  wrote bench block into %s" % receipt_path)
PY

python3 scripts/lib/bench_receipt.py --bench "$RECEIPT" >/dev/null 2>&1 || {
    printf 'FAIL  the written receipt does not validate\n' >&2; exit 1; }
printf 'ok    bench block written and validated\n'
