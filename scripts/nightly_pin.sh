# nightly_pin.sh — NIGHTLY MODE for scripts/apr_bin.sh and scripts/pv_bin.sh (#4186).
#
# Source it, never execute it. apr_bin.sh / pv_bin.sh source it themselves when
# APR_BIN_REQUIRE=nightly / PV_BIN_REQUIRE=nightly; callers do not.
#
# WHY THIS EXISTS. Operator, 2026-09-24: "lambda-labs should always dogfood
# latest soveirgn stack nightly binaries and never run older releases: this is
# P0". apr_bin.sh and pv_bin.sh prove a binary came from THIS TREE at HEAD. That
# is the right question in a dev worktree or on a PR, and the wrong one on a
# dogfood host, where the question is "is this THE green nightly?". The fleet
# inventory on #4186 measured the gap: of the apr first on PATH, lambda ran
# 0.68.2 from crates.io, yoga ran 0.63.0 (six weeks old) and had no pv at all,
# and every gate there was free to call that green.
#
# THE AUTHORITY is the arbiter's nightly manifest (schema
# aprender-nightly-manifest/v1, produced by #4189, owned by aprender-48),
# read from a LOCAL copy the installer writes next to the binaries:
#     ${APR_NIGHTLY_MANIFEST:-$HOME/.cache/aprender/nightly-manifest.json}
# Nothing here touches the network: this runs inside every gate.
#
# ACCEPT iff ALL hold (accept rule v1, agreed with aprender-48):
#   - manifest present, parses, schema is v1, generated_at within max age
#     (APR_NIGHTLY_MAX_AGE_H, default 36) and not in the future
#   - targets[<host triple>] exists with a 40-hex green_sha
#   - the tool has a tools[] entry and is not denylisted
#   - sha256(binary) == tools[tool].bin_sha256            (always)
#   - if tools[tool].version_sha is non-null: it equals green_sha AND the
#     sha in the binary's `--version` is a prefix of green_sha
# Everything else REFUSES. There is no fallback to HEAD provenance and no
# fallback to "whatever is on PATH": a nightly-mode caller that cannot prove
# the nightly gets rc 1 and a message naming what it found and what it wanted.
#
# The case table lives in scripts/check_nightly_pin.sh (--self-test), which also
# re-runs it against mutants of this file.
#
# OPTION-NEUTRAL: sets no shell options (see scripts/check_sourced_libs_option_neutral.sh).
# Failure is by return status only.

NIGHTLY_PIN_SCHEMA='aprender-nightly-manifest/v1'

nightly_pin_refuse() {
    {
        printf 'NIGHTLY PIN REFUSED (%s): %s\n' "$1" "$2"
        printf '  manifest : %s\n' "$(nightly_pin_manifest_path)"
        printf '  this host runs ONLY the latest green nightly (#4186). Install it with the\n'
        printf '  fleet installer, or unset %s to use HEAD provenance in a dev tree.\n' "$3"
    } >&2
    return 1
}

nightly_pin_manifest_path() {
    printf '%s\n' "${APR_NIGHTLY_MANIFEST:-$HOME/.cache/aprender/nightly-manifest.json}"
}

nightly_pin_triple() {
    [ "$(uname -s 2>/dev/null)" = "Linux" ] || return 1
    case "$(uname -m 2>/dev/null)" in
        x86_64) printf 'x86_64-unknown-linux-gnu\n' ;;
        aarch64 | arm64) printf 'aarch64-unknown-linux-gnu\n' ;;
        *) return 1 ;;
    esac
}

# nightly_pin_check TOOL BIN REQUIRE_VAR -> 0 accept, 1 refuse (reason on stderr)
nightly_pin_check() {
    np_tool="$1"
    np_bin="$2"
    np_var="$3"
    np_m=$(nightly_pin_manifest_path)

    command -v jq >/dev/null 2>&1 || { nightly_pin_refuse "$np_tool" "jq not found; cannot read the manifest" "$np_var"; return 1; }
    [ -f "$np_m" ] && [ -r "$np_m" ] || { nightly_pin_refuse "$np_tool" "MISSING MANIFEST" "$np_var"; return 1; }
    jq -e 'type == "object"' "$np_m" >/dev/null 2>&1 || { nightly_pin_refuse "$np_tool" "MALFORMED MANIFEST (not a JSON object)" "$np_var"; return 1; }

    np_schema=$(jq -r '.schema // empty' "$np_m" 2>/dev/null) || np_schema=""
    [ "$np_schema" = "$NIGHTLY_PIN_SCHEMA" ] || { nightly_pin_refuse "$np_tool" "UNKNOWN SCHEMA '$np_schema' (want $NIGHTLY_PIN_SCHEMA)" "$np_var"; return 1; }

    np_gen=$(jq -r '.generated_at // empty' "$np_m" 2>/dev/null) || np_gen=""
    np_gen_s=$(date -d "$np_gen" +%s 2>/dev/null) || np_gen_s=""  # bashrs disable-line=DET002
    case "$np_gen_s" in '' | *[!0-9]*) nightly_pin_refuse "$np_tool" "MALFORMED MANIFEST (generated_at '$np_gen')" "$np_var"; return 1 ;; esac
    np_now=$(date +%s)  # bashrs disable-line=DET002 (staleness is wall-clock by definition)
    np_max_h="${APR_NIGHTLY_MAX_AGE_H:-36}"
    case "$np_max_h" in '' | *[!0-9]*) nightly_pin_refuse "$np_tool" "APR_NIGHTLY_MAX_AGE_H='$np_max_h' is not a whole number of hours" "$np_var"; return 1 ;; esac
    if [ "$np_gen_s" -gt $((np_now + 300)) ]; then
        nightly_pin_refuse "$np_tool" "MANIFEST FROM THE FUTURE (generated_at $np_gen)" "$np_var"; return 1
    fi
    if [ $((np_now - np_gen_s)) -gt $((np_max_h * 3600)) ]; then
        nightly_pin_refuse "$np_tool" "STALE MANIFEST (generated_at $np_gen, older than ${np_max_h}h: the arbiter has not run)" "$np_var"; return 1
    fi

    np_triple=$(nightly_pin_triple) || { nightly_pin_refuse "$np_tool" "UNSUPPORTED HOST $(uname -s)/$(uname -m)" "$np_var"; return 1; }
    np_green=$(jq -r --arg t "$np_triple" '.targets[$t].green_sha // empty' "$np_m" 2>/dev/null) || np_green=""
    printf '%s' "$np_green" | grep -Eqx '[0-9a-f]{40}' || { nightly_pin_refuse "$np_tool" "NO GREEN NIGHTLY for $np_triple (green_sha '$np_green')" "$np_var"; return 1; }

    # denylist entries may be tool names, shas, or objects naming either.
    np_denied=$(jq -r --arg tool "$np_tool" --arg g "$np_green" '
        [.denylist[]? | if type == "string" then . else (.tool // .bin // .sha // empty) end]
        | map(select(. == $tool or . == $g)) | length' "$np_m" 2>/dev/null) || np_denied=""
    [ "$np_denied" = "0" ] || { nightly_pin_refuse "$np_tool" "DENYLISTED by the manifest (or denylist unreadable: '$np_denied')" "$np_var"; return 1; }

    [ -f "$np_bin" ] && [ -x "$np_bin" ] || { nightly_pin_refuse "$np_tool" "no executable '$np_tool' to check ('$np_bin')" "$np_var"; return 1; }
    np_have_hash=$(sha256sum "$np_bin" 2>/dev/null | awk '{print $1}') || np_have_hash=""
    printf '%s' "$np_have_hash" | grep -Eqx '[0-9a-f]{64}' || { nightly_pin_refuse "$np_tool" "cannot hash $np_bin" "$np_var"; return 1; }
    np_first=$("$np_bin" --version 2>&1 </dev/null | head -n 1) || np_first=""

    # The tool's entries: `apr` and its build variants (`apr@cuda`, ...). The
    # installed binary must BE one of them, byte for byte.
    np_keys=$(jq -r --arg t "$np_triple" --arg tool "$np_tool" '
        .targets[$t].tools // {} | keys[] | select(. == $tool or startswith($tool + "@"))' "$np_m" 2>/dev/null) || np_keys=""
    [ -n "$np_keys" ] || { nightly_pin_refuse "$np_tool" "no nightly '$np_tool' in the manifest for $np_triple" "$np_var"; return 1; }
    np_key=$(jq -r --arg t "$np_triple" --arg tool "$np_tool" --arg h "$np_have_hash" '
        .targets[$t].tools | to_entries[]
        | select((.key == $tool or (.key | startswith($tool + "@"))) and .value.bin_sha256 == $h)
        | .key' "$np_m" 2>/dev/null | head -n 1) || np_key=""
    if [ -z "$np_key" ]; then
        np_want=$(jq -r --arg t "$np_triple" --arg tool "$np_tool" '
            .targets[$t].tools | to_entries[]
            | select(.key == $tool or (.key | startswith($tool + "@")))
            | "\(.key)=\(.value.bin_sha256)"' "$np_m" 2>/dev/null | tr '\n' ' ')
        nightly_pin_refuse "$np_tool" "NOT THE NIGHTLY: $np_bin reports '$np_first' (sha256 $np_have_hash); the manifest's green $np_triple build is ${np_green} ($np_want)" "$np_var"
        return 1
    fi
    np_denied=$(jq -r --arg k "$np_key" '
        [.denylist[]? | if type == "string" then . else (.tool // .bin // .sha // empty) end]
        | map(select(. == $k)) | length' "$np_m" 2>/dev/null) || np_denied=""
    [ "$np_denied" = "0" ] || { nightly_pin_refuse "$np_tool" "'$np_key' is DENYLISTED by the manifest" "$np_var"; return 1; }
    np_vsha=$(jq -r --arg t "$np_triple" --arg k "$np_key" '.targets[$t].tools[$k].version_sha // empty' "$np_m" 2>/dev/null) || np_vsha=""

    if [ -n "$np_vsha" ]; then
        [ "$np_vsha" = "$np_green" ] || { nightly_pin_refuse "$np_tool" "MALFORMED MANIFEST (tools.$np_key.version_sha $np_vsha != green_sha $np_green)" "$np_var"; return 1; }
        np_bsha=$(printf '%s\n' "$np_first" | sed -n 's/.*(\([0-9a-f]\{7,40\}\)).*/\1/p')
        case "$np_bsha" in
            '') nightly_pin_refuse "$np_tool" "binary --version carries no sha ('$np_first') but the manifest says it must" "$np_var"; return 1 ;;
        esac
        case "$np_green" in
            "$np_bsha"*) ;;
            *) nightly_pin_refuse "$np_tool" "binary --version sha $np_bsha is not green_sha $np_green" "$np_var"; return 1 ;;
        esac
    fi
    return 0
}

# nightly_pin_resolve TOOL OVERRIDE_VAR REQUIRE_VAR -> prints the accepted path.
# The candidate is the explicit override if set, else the first TOOL on PATH:
# on a dogfood host "what runs" IS what PATH resolves, so that is what must be
# the nightly.
nightly_pin_resolve() {
    np_r_tool="$1"
    np_r_override="$2"
    np_r_var="$3"
    case "$np_r_override" in '' | *[!A-Za-z0-9_]*) nightly_pin_refuse "$np_r_tool" "bad override name '$np_r_override'" "$np_r_var"; return 1 ;; esac
    eval "np_r_bin=\${$np_r_override:-}"  # bashrs disable-line=SEC001 (name validated above; bash+zsh portable)
    if [ -z "$np_r_bin" ]; then
        np_r_bin=$(command -v "$np_r_tool" 2>/dev/null) || np_r_bin=""
        case "$np_r_bin" in /*) ;; *) np_r_bin="" ;; esac
    fi
    [ -n "$np_r_bin" ] || { nightly_pin_refuse "$np_r_tool" "no '$np_r_tool' on PATH and $np_r_override unset" "$np_r_var"; return 1; }
    nightly_pin_check "$np_r_tool" "$np_r_bin" "$np_r_var" || return 1
    printf '%s\n' "$np_r_bin"
}

# nightly_pin_mode REQUIRE_VAR -> 0 nightly mode on, 1 off (HEAD provenance),
# 2 set to something unknown: the caller must refuse, never guess.
#
# DEFAULT ON A FLEET HOST. Operator, 2026-09-24: "we need nightly binaries on
# fleet from nightly build; the end." A host is a fleet host when the marker
# ${APR_FLEET_MARKER:-$HOME/.config/aprender/fleet-nightly} exists (the fleet
# installer writes it). There, an UNSET REQUIRE_VAR means nightly. Two
# exceptions, both explicit:
#   - REQUIRE_VAR=head opts one invocation into HEAD provenance (a dev tree
#     testing its own build);
#   - inside a GitHub Actions job (GITHUB_ACTIONS=true AND a numeric
#     GITHUB_RUN_ID, both of which the runner sets) the marker is ignored: a PR
#     job tests the PR's code, which is by construction never the nightly, and
#     the self-hosted runners share the host user with the marker. A nightly
#     job sets REQUIRE_VAR=nightly itself. A bare GITHUB_ACTIONS=true (exported
#     while debugging) does not qualify, and the skip is never silent: it
#     prints a notice, so a shell that inherited a runner's env says so.
nightly_pin_mode() {
    case "$1" in '' | *[!A-Za-z0-9_]*) printf 'NIGHTLY PIN REFUSED: bad mode variable name %s\n' "$1" >&2; return 2 ;; esac
    eval "np_mode=\${$1:-}"  # bashrs disable-line=SEC001 (name validated above; bash+zsh portable)
    if [ -z "$np_mode" ] && [ -e "${APR_FLEET_MARKER:-$HOME/.config/aprender/fleet-nightly}" ]; then
        np_actions=0
        if [ "${GITHUB_ACTIONS:-}" = "true" ]; then
            case "${GITHUB_RUN_ID:-}" in '' | *[!0-9]*) ;; *) np_actions=1 ;; esac
        fi
        if [ "$np_actions" = 1 ]; then
            printf 'nightly pin: fleet marker ignored inside GitHub Actions run %s (HEAD provenance); set %s=nightly to pin\n' "$GITHUB_RUN_ID" "$1" >&2
        else
            np_mode=nightly
        fi
    fi
    case "$np_mode" in
        '' | head) return 1 ;;
        nightly) return 0 ;;
        *) printf 'NIGHTLY PIN REFUSED: %s=%s is not a known mode (nightly|head)\n' "$1" "$np_mode" >&2; return 2 ;;
    esac
}
