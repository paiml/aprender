# pmat_pin.sh - the ONE definition of how the pmat pin is read from tools.toml (PMAT-6131).
#
# Source it: `. scripts/pmat_pin.sh`, then `pmat_pin_read <tools.toml>` prints X.Y.Z (the
# `version` under [pmat]) and nothing else when the line is absent or not exactly three
# dot-separated numbers. Callers treat empty output as an unreadable pin and fail closed.
# scripts/pmat_bin.sh and the check_hardcoded_paths.sh self-test both call this, so the
# production reader and its test cannot disagree on what a version is.
#
# OPTION-NEUTRAL: no `set` here; a sourced library fails by return status
# (scripts/check_sourced_libs_option_neutral.sh).
pmat_pin_read() { # pmat_pin_read <tools.toml> -> X.Y.Z or nothing
    sed -nE '/^\[pmat\]/,/^\[/{s/^version[[:space:]]*=[[:space:]]*"([0-9]+\.[0-9]+\.[0-9]+)"[[:space:]]*(#.*)?$/\1/p;}' "$1" 2>/dev/null | head -n1
}
