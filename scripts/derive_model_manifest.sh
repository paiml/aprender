#!/usr/bin/env bash
# derive_model_manifest.sh — evidence/models/supported.yaml is DERIVED from what the
# tree names, never typed (PP-066 row L0-1, #2971, PMAT-1065, card item 1).
#
# Sources (in this order; every citation is file:line):
#   README.md · docs/BEATS.md · book/src/**/*.md · evidence/dogfood/*/*.json ·
#   scripts/perf-matrix.yaml
# A model is a name matching /qwen[0-9.]*-?[a-z]*-?[0-9]+(\.[0-9])?b(-[a-z0-9.]+)*/ (case-
# insensitive) with the quantisation suffix and the .gguf extension stripped; each
# entry records where it is cited. The manifest is what C14 (check_model_parity.sh
# --manifest) iterates and what check_readme_claims.sh holds the README to.
#
#   bash scripts/derive_model_manifest.sh            # (re)write evidence/models/supported.yaml
#   bash scripts/derive_model_manifest.sh --check    # 0 the committed manifest equals the derivation · 1 DRIFT (diff printed) · 2 ENV
#   bash scripts/derive_model_manifest.sh --self-test
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROG=derive_model_manifest
OUT="${MANIFEST_OUT:-$ROOT/evidence/models/supported.yaml}"
SRC_ROOT="${MANIFEST_SRC_ROOT:-$ROOT}"

derive() { # derive <src root> -> yaml on stdout
    python3 - "$1" <<'PY'
import sys, os, re, glob, json
root = sys.argv[1]
pat = re.compile(r"qwen[0-9](?:\.[0-9])?(?:-coder|-instruct)?-[0-9]+(?:\.[0-9])?b(?:-instruct)?", re.I)
sources = ["README.md", "docs/BEATS.md"] + sorted(glob.glob(os.path.join(root, "book/src/**/*.md"), recursive=True)) \
        + sorted(glob.glob(os.path.join(root, "evidence/dogfood/*/*.json"))) + ["scripts/perf-matrix.yaml"]
found = {}
for f in sources:
    p = f if os.path.isabs(f) else os.path.join(root, f)
    if not os.path.isfile(p): continue
    rel = os.path.relpath(p, root)
    try: text = open(p, encoding="utf-8", errors="replace").read()
    except OSError: continue
    for i, line in enumerate(text.split("\n"), 1):
        for m in pat.finditer(line):
            name = m.group(0).lower()
            found.setdefault(name, []).append(f"{rel}:{i}")
lines = ["# evidence/models/supported.yaml — DERIVED by scripts/derive_model_manifest.sh; do not edit by hand (L0-1, #2971).",
         "# A model is in the manifest iff a shipped document names it; every entry cites where.",
         "schema: apr-supported-models/v1",
         "models:"]
for name in sorted(found):
    cites = found[name]; fam = re.match(r"(qwen[0-9](?:\.[0-9])?)", name).group(1)
    size = re.search(r"-([0-9]+(?:\.[0-9])?b)", name).group(1)
    lines.append(f"- name: {name}")
    lines.append(f"  family: {fam}")
    lines.append(f"  size: {size}")
    lines.append(f"  cited_by_count: {len(cites)}")
    lines.append("  cited_by:")
    for c in cites[:12]: lines.append(f"  - {c}")
    if len(cites) > 12: lines.append(f"  # … and {len(cites)-12} more")
print("\n".join(lines))
PY
}

if [ "${1:-}" = "--self-test" ]; then
    TD=$(mktemp -d "${TMPDIR:-/tmp}/manifest.XXXXXX"); trap 'rm -rf "${TD:?}"' EXIT
    mkdir -p "$TD/docs" "$TD/book/src" "$TD/evidence/dogfood/x" "$TD/scripts"
    printf '# apr\nRuns Qwen2.5-Coder-1.5B-Instruct and qwen2.5-coder-7b-instruct-q4_k_m.gguf\n' > "$TD/README.md"
    printf 'model: qwen3.5-0.8b-q4_k_m.gguf\n' > "$TD/scripts/perf-matrix.yaml"
    printf '{"model":"qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"}\n' > "$TD/evidence/dogfood/x/h.json"
    n=0; red=0
    t() { local want=$1 label=$2 got=$3; n=$((n + 1)); if [ "$got" = "$want" ]; then printf 'ok    row %-2s %s\n' "$n" "$label"; else printf 'FAIL  row %-2s got=%s wanted=%s  %s\n' "$n" "$got" "$want" "$label"; red=1; fi; }
    y=$(derive "$TD")
    t 3 "three distinct models derived from README, perf-matrix and a dogfood receipt" "$(printf '%s\n' "$y" | grep -c '^- name:')"
    t "qwen2.5-coder-1.5b-instruct" "case-folded, quant suffix and .gguf stripped" "$(printf '%s\n' "$y" | grep -oE '^- name: qwen2.5-coder-1.5b[a-z-]*' | head -1 | sed 's/^- name: //')"
    t 2 "the 1.5B is cited twice (README + dogfood)" "$(printf '%s\n' "$y" | awk '/^- name: qwen2.5-coder-1.5b-instruct$/{f=1} f&&/cited_by_count:/{print $2; exit}')"
    printf '%s\n' "$y" > "$TD/committed.yaml"
    n=$((n + 1)); if MANIFEST_SRC_ROOT="$TD" MANIFEST_OUT="$TD/committed.yaml" bash "$0" --check >/dev/null 2>&1; then printf 'ok    row %-2s --check PASSES on an identical committed manifest\n' "$n"; else printf 'FAIL  row %-2s --check failed on an identical manifest\n' "$n"; red=1; fi
    printf -- '- name: qwen9-99b\n  family: qwen9\n  size: 99b\n  cited_by_count: 0\n  cited_by: []\n' >> "$TD/committed.yaml"
    n=$((n + 1)); if MANIFEST_SRC_ROOT="$TD" MANIFEST_OUT="$TD/committed.yaml" bash "$0" --check >/dev/null 2>&1; then printf 'FAIL  row %-2s --check passed a HAND-TYPED entry (the registered mutation)\n' "$n"; red=1; else printf 'ok    row %-2s a hand-typed entry nothing cites is DRIFT (rc 1)\n' "$n"; fi
    n=$((n + 1)); if MANIFEST_SRC_ROOT="$TD" MANIFEST_OUT="$TD/absent.yaml" bash "$0" --check >/dev/null 2>&1; then printf 'FAIL  row %-2s --check passed with no manifest\n' "$n"; red=1; else printf 'ok    row %-2s a missing manifest is not a pass\n' "$n"; fi
    printf '%s/%s rows\n' "$((n - red))" "$n"; [ "$red" = 0 ] || exit 1; exit 0
fi
command -v python3 >/dev/null || { printf '%s: ENV - python3 missing\n' "$PROG" >&2; exit 2; }
if [ "${1:-}" = "--check" ]; then
    [ -f "$OUT" ] || { printf 'FAIL  %s: %s is missing — run without --check to derive it\n' "$PROG" "${OUT#"$ROOT"/}"; exit 1; }
    if diff -u "$OUT" <(derive "$SRC_ROOT") > "${TMPDIR:-/tmp}/manifest-diff.$$"; then printf 'PASS  %s: %s equals its derivation (%s models)\n' "$PROG" "${OUT#"$ROOT"/}" "$(grep -c '^- name:' "$OUT")"; rm -f "${TMPDIR:-/tmp}/manifest-diff.$$"; exit 0; fi
    printf 'FAIL  %s: DRIFT — %s differs from its derivation (the manifest is derived, never typed):\n' "$PROG" "${OUT#"$ROOT"/}"; head -40 "${TMPDIR:-/tmp}/manifest-diff.$$" | sed 's/^/  /'; rm -f "${TMPDIR:-/tmp}/manifest-diff.$$"; exit 1
fi
mkdir -p "$(dirname "$OUT")"; derive "$SRC_ROOT" > "$OUT.tmp" && mv "$OUT.tmp" "$OUT"; printf 'wrote %s (%s models)\n' "${OUT#"$ROOT"/}" "$(grep -c '^- name:' "$OUT")"
