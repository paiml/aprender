#!/usr/bin/env bash
# advisory_positive_control.sh -- the scanner of record must SEE RUSTSEC-2026-0378 and -0379 on the wasmtime 48.0.6 lock
# (PMAT-5792, ADG-INV-001 / FALSIFY-ADG-003). A real `cargo deny --all-features check advisories` runs against a committed
# lock fixture (fixtures/adv-positive-control/, stored as *.fixture so cargo does not treat the directory as a workspace
# member), with a deny.toml that carries no ignores. Exit 0 only when both ids are reported; cargo-deny missing or a
# scan that names neither id is a failure, never a skip.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fx="$here/fixtures/adv-positive-control"
d=$(mktemp -d)
trap 'rm -rf -- "${d:?}"' EXIT

command -v cargo-deny >/dev/null || { echo "advisory_positive_control: cargo-deny is not installed -- the control cannot run" >&2; exit 1; }
mkdir -p "$d/src"
cp "$fx/Cargo.toml.fixture" "$d/Cargo.toml"
cp "$fx/Cargo.lock.fixture" "$d/Cargo.lock"
: >"$d/src/lib.rs"
printf '[advisories]\nversion = 2\n' >"$d/deny.toml"

out=$(cd "$d" && cargo deny --locked --all-features --format json check --config "$d/deny.toml" advisories 2>&1) || true
for id in RUSTSEC-2026-0378 RUSTSEC-2026-0379; do
    if ! grep -qF -- "$id" <<<"$out"; then
        echo "advisory_positive_control: FAIL -- cargo deny on the wasmtime 48.0.6 lock did not report $id" >&2
        printf '%s\n' "$out" | tail -5 >&2
        exit 1
    fi
done
echo "advisory_positive_control: ok -- cargo deny reports RUSTSEC-2026-0378 and -0379 on the 48.0.6 lock"
