#!/usr/bin/env bash
# chat_template_inventory_parity.sh -- pre-publish dogfood gate (#3755)
#
# apr renders a model file's OWN chat template (EmbeddedChatTemplate). The lib test
# chat_template_embedded_oracle proves that byte-equal to HF's apply_chat_template on
# COMMITTED fixtures. This gate proves it on the release host's REAL inventory: every
# GGUF in the model directory has its template extracted and rendered by transformers
# (scripts/render_chat_template_reference.py), and the same oracle test runs against
# those fresh fixtures (APR_CHAT_TEMPLATE_FIXTURES). A template apr cannot reproduce byte
# for byte, a model dir with no template, or a test that did not read the fresh fixtures
# is a FAIL.
#
# Listed in Cargo.toml [package.metadata.dogfood].gates: only
# `scripts/dogfood.sh --phase pre-publish` runs it, on the host that has the models.
# Nothing on the PR path runs it (the PR-time half is the lib test on the fixtures).
#
# Usage: bash scripts/chat_template_inventory_parity.sh [MODELS_DIR]   (default ~/models)
set -uo pipefail
root=$(cd "$(dirname "$0")/.." && pwd) || exit 2
cd "$root" || exit 2
default_models="$HOME/models"
models="${1:-${APR_MODELS_DIR:-"$default_models"}}"
[ -d "$models" ] || { printf 'FAIL  model directory %s does not exist\n' "$models"; exit 1; }

out=$(mktemp -d) || exit 2
case "$out" in /tmp/?*|"${TMPDIR:-/tmp}"/?*) ;; *) printf 'FAIL  mktemp gave %s\n' "$out"; exit 2 ;; esac
trap 'rm -rf -- "${out:?}"' EXIT

python3 scripts/render_chat_template_reference.py --models "$models" --out "$out" > "$out/render.log" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then
    printf 'FAIL  reference rendering (transformers) rc=%s:\n' "$rc"
    sed 's/^/      /' "$out/render.log"
    exit 1
fi
templates=$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$out/index.json") || templates=0
if [ "$templates" -lt 1 ]; then
    printf 'FAIL  no GGUF in %s ships a chat template: nothing was compared\n' "$models"
    exit 1
fi

APR_CHAT_TEMPLATE_FIXTURES="$out" cargo test -p aprender-serve --lib -- \
    chat_template::chat_template_embedded_oracle::embedded_rendering_is_byte_equal_to_hf --exact --nocapture \
    > "$out/test.log" 2>&1
rc=$?
# The test prints the directory it read: a pass on the committed fixtures is not a pass here.
if ! grep -qF "from $out" "$out/test.log"; then
    printf 'FAIL  the oracle did not read the fresh fixtures in %s (rc=%s)\n' "$out" "$rc"
    tail -20 "$out/test.log" | sed 's/^/      /'
    exit 1
fi
if [ "$rc" -ne 0 ]; then
    printf 'FAIL  apr does not reproduce the inventory templates byte for byte:\n'
    grep -E 'differ from|first difference|apr:|hf:' "$out/test.log" | head -40 | sed 's/^/      /'
    exit 1
fi
printf 'PASS  %s inventory template(s) from %s rendered byte-equal to %s\n' \
    "$templates" "$models" "$(grep -m1 -o 'transformers [0-9.]*' "$out/render.log")"
