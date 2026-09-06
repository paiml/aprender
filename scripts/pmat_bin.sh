# pmat_bin.sh - resolve THE analyser this repository's gates are measured with.
#
# Source it, never execute it:
#     . scripts/pmat_bin.sh || exit 1
#     "$PMAT" analyze hardcoded-paths -p . -f json
#
# WHY (PMAT-1059, G-10, #2999). A ratchet's number is a property of the tree AND
# the instrument. scripts/hardcoded_path_shipped_baseline.txt held 277 under an
# unnamed version; 3.37.0 and 3.38.0 both count 317 on the same tree. The day
# paiml/infra pinned the fleet at 3.37.0 (machines/intel/forjar.yaml, PMAT-231)
# every PR went red. So the version is pinned HERE, in one line, and bumped only
# by its own PR after forjar moves the fleet (the R4 sequence in PP-066).
#
# OPTION-NEUTRAL BY CONSTRUCTION: no `set` in this file; failure is the return
# status (scripts/check_sourced_libs_option_neutral.sh). Same shape as
# scripts/pv_bin.sh and scripts/apr_bin.sh.
#
# Resolution order:
#   1. $PMAT_BIN_OVERRIDE                   (tests inject a fixture binary)
#   2. $HOME/.local/pmat/$PMAT_PIN/bin/pmat (a versioned root, never PATH):
#        cargo install --version "$PMAT_PIN" --locked --root ~/.local/pmat/$PMAT_PIN pmat
#   3. the first `pmat` on PATH             (accepted ONLY if it reports the pin;
#                                            PMAT_BIN_NO_FALLBACK=1 disables 2-3)
# Whatever is found must print `pmat $PMAT_PIN`; anything else is refused with
# both versions named. Exports PMAT and PMAT_VERSION.
PMAT_PIN="3.37.0"

pmat_bin_resolve() {
    # an ARRAY, not a word-split string: sourced from zsh (no word splitting) a string
    # would be one non-executable "candidate" and the pin would look uninstalled
    local cand tried="" v onpath="" cands=()
    if [ -n "${PMAT_BIN_OVERRIDE:-}" ]; then cands+=("$PMAT_BIN_OVERRIDE"); fi
    if [ "${PMAT_BIN_NO_FALLBACK:-0}" != 1 ]; then
        cands+=("${HOME}/.local/pmat/${PMAT_PIN}/bin/pmat")
        onpath=$(command -v pmat 2>/dev/null || true)
        if [ -n "$onpath" ]; then cands+=("$onpath"); fi
    fi
    for cand in ${cands[@]+"${cands[@]}"}; do
        if [ ! -x "$cand" ]; then continue; fi
        v=$("$cand" --version 2>/dev/null | head -n1 | awk '{print $2}')
        if [ "$v" = "$PMAT_PIN" ]; then
            PMAT="$cand"; PMAT_VERSION="$v"; export PMAT PMAT_VERSION; return 0
        fi
        tried="$tried ${cand}=${v:-unparseable}"
    done
    printf 'pmat_bin.sh: no analyser at the pin %s (tried:%s). Install it into a versioned root:\n' "$PMAT_PIN" "${tried:- none}" >&2
    printf '  cargo install --version "%s" --locked --root ~/.local/pmat/%s pmat\n' "$PMAT_PIN" "$PMAT_PIN" >&2
    return 1
}
pmat_bin_resolve
