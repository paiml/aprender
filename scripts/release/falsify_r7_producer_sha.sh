#!/usr/bin/env bash
# falsify_r7_producer_sha.sh -- the R7 receipt producer (crux_sweep_shards.sh) stamps apr_binary_sha256 (aprender#6679)
# Re-merges a fixture shard with --merge-only; the merged <host>-<backend>.meta.json must carry the sha256 of --apr.
# The judge's own verdict on the fixture is not under test (its rc is ignored); the stamped meta is.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2
t=$(mktemp -d "${TMPDIR:-/tmp}/r7prod.XXXXXX") || exit 2
trap 'rm -rf "$t"' EXIT
mkdir -p "$t/out" "$t/w"
printf 'fake-apr-bytes' > "$t/apr"; chmod +x "$t/apr"
printf '{"models":[{"sha256":"aa"}]}' > "$t/w/meta.json"
: > "$t/w/manifest.jsonl"
printf 's1\t0\t%s\n' "$t/w" > "$t/out/shards.tsv"
want=$(sha256sum "$t/apr" | cut -d' ' -f1)
bash scripts/crux_sweep_shards.sh 0.0.0 --host fx --apr "$t/apr" --out "$t/out" --models-dir "$t/w" --merge-only > "$t/log" 2>&1
got=$(jq -r '.apr_binary_sha256 // empty' "$t/out/fx-gpu.meta.json" 2> /dev/null)
if [ "$got" = "$want" ]; then echo "PASS merged meta carries apr_binary_sha256"; exit 0; fi
echo "FAIL merged meta apr_binary_sha256='${got:-absent}', want $want"; tail -3 "$t/log"; exit 1
