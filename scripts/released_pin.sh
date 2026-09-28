# released_pin.sh — RELEASED MODE for scripts/pv_bin.sh (N-1 gate tools).
#
# Source it, never execute it. pv_bin.sh sources it itself when
# PV_BIN_REQUIRE=released; callers do not.
#
# WHY THIS EXISTS. Operator, 2026-09-28 17:15Z (N-1): gates use only
# already-released tool features, so the FINAL path runs its contract gate on a
# RELEASED pv, never on the pv this tree builds. pv_bin.sh had two modes:
#   head     — build pv at HEAD and require `pv --version` == the tree's
#              declared version. A released pv is always an OLDER version than
#              the tree it gates (0.69.4 vs 0.70.0), so it is refused as STALE —
#              including through the PV_BIN override, which is version-checked.
#   nightly  — require the arbiter's nightly manifest. A release is not a nightly.
# So there was no way to hand a gate a released pv at all.
#
# THE AUTHORITY is the fleet pin forjar writes on every converged host:
#     ${PV_BIN_RELEASED_PIN:-$HOME/.config/fleet/pv.pin}   (a bare version, e.g. 0.69.4)
# with an optional sibling <pin>.sha256 (64 hex). Nothing here touches the network.
#
# ACCEPT iff ALL hold:
#   R1 the pin is readable and is a bare release version X.Y.Z. A pre-release
#      or build suffix (-rc.1, -dev, +sha) is not a release and is refused.
#   R2 the pin is STRICTLY OLDER than the version this tree declares (numeric
#      X.Y.Z). A pin equal to the tree's version is exactly what a HEAD dev build
#      prints, so it proves nothing about being released; newer is nonsense.
#   R3 the binary is PV_BIN if set, else the first regular executable among
#      ${PV_BIN_RELEASED_CANDIDATES:-/opt/fleet-bin/bin/pv:$HOME/.cargo/bin/pv}.
#   R4 its FIRST `--version` line has the pin as field 2, exactly, and carries
#      the identity marker. A dev build reports the tree's version, and a
#      binary with a suffix reports that suffix: both fail equality.
#   R5 if <pin>.sha256 exists, sha256(binary) equals it.
# Everything else REFUSES, with rc 1 and a message naming what was found and
# what was wanted. There is no fallback to HEAD provenance and no fallback to
# PATH: a released-mode caller that cannot prove the release gets nothing.
#
# The case table and its planted mutants: scripts/check_released_pin.sh --self-test.
#
# OPTION-NEUTRAL: sets no shell options (see scripts/check_sourced_libs_option_neutral.sh).
# Failure is by return status only.

RELEASED_PIN_API=1

released_pin_refuse() {
    {
        printf 'RELEASED PIN REFUSED: %s\n' "$1"
        printf '  pin      : %s\n' "${PV_BIN_RELEASED_PIN:-${HOME:-}/.config/fleet/pv.pin}"
        printf '  PV_BIN_REQUIRE=released runs ONLY the released pv the fleet pin names\n'
        printf '  (N-1). Converge the host (forjar), or unset PV_BIN_REQUIRE for HEAD mode.\n'
    } >&2
    return 1
}

# released_pin_older A B -> 0 iff numeric X.Y.Z of A < numeric X.Y.Z of B.
# Any suffix on B (-dev, -rc.1) is dropped: 0.70.0-rc.1 declares the 0.70.0 line.
released_pin_older() {
    awk -v a="$1" -v b="$2" 'BEGIN {
        sub(/[-+].*/, "", a); sub(/[-+].*/, "", b)
        na = split(a, x, "."); nb = split(b, y, ".")
        if (na != 3 || nb != 3) exit 2
        for (i = 1; i <= 3; i++) {
            if (x[i] !~ /^[0-9]+$/ || y[i] !~ /^[0-9]+$/) exit 2
            if (x[i] + 0 < y[i] + 0) exit 0
            if (x[i] + 0 > y[i] + 0) exit 1
        }
        exit 1
    }'
}

# released_pin_resolve DECLARED IDENTITY -> prints the binary on stdout, rc 0; else rc 1.
released_pin_resolve() {
    rp_declared="$1"
    rp_identity="$2"
    rp_pin_file="${PV_BIN_RELEASED_PIN:-${HOME:-}/.config/fleet/pv.pin}"
    if [ ! -r "$rp_pin_file" ]; then
        released_pin_refuse "no readable pin at $rp_pin_file"
        return 1
    fi
    rp_pin=$(tr -d '[:space:]' < "$rp_pin_file") || rp_pin=""
    case "$rp_pin" in
        '' | *[!0-9.]* | .* | *. | *..*)
            released_pin_refuse "pin '$rp_pin' is not a bare release version X.Y.Z (a -rc/-dev/+sha build is not a release)"
            return 1 ;;
    esac
    if [ -z "$rp_declared" ]; then
        released_pin_refuse "could not read the version this tree declares"
        return 1
    fi
    rp_cmp=0
    released_pin_older "$rp_pin" "$rp_declared" || rp_cmp=$?
    if [ "$rp_cmp" -eq 2 ]; then
        released_pin_refuse "pin '$rp_pin' or declared '$rp_declared' is not X.Y.Z"
        return 1
    fi
    if [ "$rp_cmp" -ne 0 ]; then
        released_pin_refuse "pin $rp_pin is not older than this tree's $rp_declared, so a HEAD build would satisfy it: that is not a released pv"
        return 1
    fi
    rp_bin=""
    if [ -n "${PV_BIN:-}" ]; then
        rp_bin="$PV_BIN"
    else
        rp_rest="${PV_BIN_RELEASED_CANDIDATES:-/opt/fleet-bin/bin/pv:${HOME:-}/.cargo/bin/pv}:"
        while [ -n "$rp_rest" ]; do
            rp_c="${rp_rest%%:*}"
            rp_rest="${rp_rest#*:}"
            if [ -n "$rp_c" ] && [ -f "$rp_c" ] && [ -x "$rp_c" ]; then
                rp_bin="$rp_c"
                break
            fi
        done
    fi
    if [ -z "$rp_bin" ] || [ ! -f "$rp_bin" ] || [ ! -x "$rp_bin" ]; then
        released_pin_refuse "no executable pv (PV_BIN='${PV_BIN:-}', candidates='${PV_BIN_RELEASED_CANDIDATES:-/opt/fleet-bin/bin/pv:${HOME:-}/.cargo/bin/pv}')"
        return 1
    fi
    rp_all=$("$rp_bin" --version 2>&1) || rp_all=""
    rp_first=$(printf '%s\n' "$rp_all" | awk 'NR==1{print; exit}') || rp_first=""
    rp_ver=$(printf '%s\n' "$rp_all" | awk 'NR==1{print $2; exit}') || rp_ver=""
    if [ "$rp_ver" != "$rp_pin" ]; then
        released_pin_refuse "$rp_bin reports '${rp_ver:-?}', the pin is $rp_pin (a dev build reports the tree's version)"
        return 1
    fi
    case "$rp_first" in
        *"$rp_identity"*) ;;
        *)
            released_pin_refuse "$rp_bin does not identify as $rp_identity (first --version line: $rp_first)"
            return 1 ;;
    esac
    if [ -e "$rp_pin_file.sha256" ]; then
        rp_want=$(tr -d '[:space:]' < "$rp_pin_file.sha256") || rp_want=""
        rp_got=$(sha256sum "$rp_bin" 2>/dev/null | awk '{print $1}') || rp_got=""
        case "$rp_want" in
            '' | *[!0-9a-f]*) released_pin_refuse "$rp_pin_file.sha256 is not 64 lowercase hex"; return 1 ;;
        esac
        if [ "${#rp_want}" -ne 64 ] || [ "$rp_got" != "$rp_want" ]; then
            released_pin_refuse "sha256($rp_bin)=${rp_got:-?} != pinned $rp_want"
            return 1
        fi
    fi
    printf '%s\n' "$rp_bin"
    return 0
}
