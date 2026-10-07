#!/usr/bin/env bash
# apr_embed_probe.sh - APR-EMBED-001 EG-0(a): run every verb in a probe table
# against one model file with the tree-pinned `apr`, recording the exit code and
# the first non-empty output line of each. Measurement only: no verdict, no gate.
#
# Usage: bash scripts/apr_embed_probe.sh <model.gguf> <bf16-sibling.gguf> <out.tsv>
# Table:  docs/audits/APR-EMBED-001/EG-0/probe-verbs.tsv
set -euo pipefail

MODEL="${1:?model file}"
BF16="${2:?bf16 sibling}"
OUT="${3:?output tsv}"
TABLE="docs/audits/APR-EMBED-001/EG-0/probe-verbs.tsv"
TIMEOUT_S="${APR_EMBED_PROBE_TIMEOUT:-120}"

. scripts/apr_bin.sh || exit 1

LOGS="${OUT%.tsv}.logs"
mkdir -p "${LOGS}"
SCRATCH="$(mktemp -d)"
trap 'rm -rf "${SCRATCH}"' EXIT

printf '# apr=%s model_sha256=%s timeout_s=%s\n' \
    "$("${APR}" --version)" "$(sha256sum "${MODEL}" | cut -d' ' -f1)" "${TIMEOUT_S}" > "${OUT}"
printf 'verb\texit\tfirst_line\n' >> "${OUT}"

n=0
while IFS=$'\t' read -r verb args; do
    case "${verb}" in ''|'#'*) continue ;; esac
    o="${SCRATCH}/${verb}"
    mkdir -p "${o}"
    args="${args//\{M\}/${MODEL}}"
    args="${args//\{B\}/${BF16}}"
    args="${args//\{D\}/$(dirname "${MODEL}")}"
    args="${args//\{O\}/${o}}"
    # Word-splitting of the substituted args is intended: the table holds argv.
    # shellcheck disable=SC2086
    rc=0
    timeout "${TIMEOUT_S}" "${APR}" "${verb}" ${args} < /dev/null > "${LOGS}/${verb}.log" 2>&1 || rc=$?
    first="$(grep -m1 -vE '^[[:space:]]*$|^[━─╭╰│]+$' "${LOGS}/${verb}.log" | tr '\t' ' ' | cut -c1-200 || true)"
    printf '%s\t%s\t%s\n' "${verb}" "${rc}" "${first}" >> "${OUT}"
    n=$((n + 1))
done < "${TABLE}"

# A probe that ran zero verbs measured nothing.
if [ "${n}" -eq 0 ]; then
    echo "apr_embed_probe: 0 verbs probed from ${TABLE}" >&2
    exit 1
fi
echo "apr_embed_probe: ${n} verbs -> ${OUT}"
