#!/usr/bin/env bash
# check_tool_versions.sh — ASSERT the fleet-pinned pmat/bashrs versions.
# Never install. (BSE-10a, PMAT-1066)
#
# WHY THIS EXISTS
# ----------------
# guard-cargo ran, per job:
#
#   cargo install bashrs --locked --quiet || true
#   command -v pmat > /dev/null 2>&1 || cargo install pmat --locked --quiet || true
#
# unpinned, PATH-first, and FAIL-OPEN: the trailing `|| true` swallowed a
# failed install and carried on with whatever was already on PATH, so a
# broken crates.io publish or a bad network day silently downgraded the
# linter/complexity gate to whatever version happened to survive, without
# ever failing the job that depends on it. aprender separately forbids
# self-hosted jobs from installing into the shared ~/.cargo/bin
# (scripts/check_cargo_install_private_root.sh, aprender#2353) — the fleet
# pin (infra machines/intel/forjar.yaml: stack-tool-pmat, stack-tool-bashrs,
# applied to every clean-room runner) is the only installer this repo may
# rely on.
#
# So this guard does not install anything. It reads the pin this repo
# expects (tools.toml) and asserts the binary already on PATH matches it,
# failing CLOSED — mismatch or absent are both a hard failure, never a
# fallback.
#
#   bash scripts/check_tool_versions.sh                       # check
#   bash scripts/check_tool_versions.sh --self-test            # case table
#   bash scripts/check_tool_versions.sh --audit-workflow FILE  # scan a workflow

set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_TOML="${REPO_ROOT}/tools.toml"
TOOLS="pmat bashrs"

# pinned_version TOOL TOMLFILE -- reads the `[TOOL]` section's `version = "X"`
# line. Prints nothing if the section or key is absent.
pinned_version() { # pinned_version TOOL TOMLFILE
    local tool=$1 f=$2
    awk -v want="[$tool]" '
        $0 == want { insec = 1; next }
        /^\[/      { insec = 0 }
        insec && $1 == "version" {
            v = $0
            sub(/^[^"]*"/, "", v)
            sub(/".*$/, "", v)
            print v
            exit
        }
    ' "$f" 2>/dev/null
}

# installed_version TOOL -- runs `TOOL --version`, takes the first line, and
# prints its second whitespace-separated field ("pmat 3.37.0" -> "3.37.0").
# Prints nothing if the tool is not on PATH.
installed_version() { # installed_version TOOL
    local tool=$1
    if ! command -v "$tool" > /dev/null 2>&1; then
        printf ''
        return 0
    fi
    "$tool" --version 2>/dev/null | head -1 | awk '{print $2}'
}

# check_one TOOL TOMLFILE -- prints one table row and returns 1 on a missing
# pin, a missing binary, or a version mismatch. Never installs anything.
check_one() { # check_one TOOL TOMLFILE
    local tool=$1 f=$2 pinned found
    pinned=$(pinned_version "$tool" "$f")
    if [ -z "$pinned" ]; then
        printf 'FAIL  %-8s no pin in %s\n' "$tool" "$f"
        return 1
    fi
    found=$(installed_version "$tool")
    if [ -z "$found" ]; then
        printf 'FAIL  %-8s pinned %-10s found <absent>\n' "$tool" "$pinned"
        printf 'pinned \xe2\x89\xa0 found: %s %s <absent>\n' "$tool" "$pinned"
        return 1
    fi
    if [ "$pinned" != "$found" ]; then
        printf 'FAIL  %-8s pinned %-10s found %s\n' "$tool" "$pinned" "$found"
        printf 'pinned \xe2\x89\xa0 found: %s %s %s\n' "$tool" "$pinned" "$found"
        return 1
    fi
    printf 'ok    %-8s pinned %-10s found %s\n' "$tool" "$pinned" "$found"
    return 0
}

# audit_workflow FILE -- REDs a workflow that still installs pmat/bashrs, or
# that calls this script's own invocation in a way that can fail open.
audit_workflow() { # audit_workflow FILE
    local f=$1 bad=0 hits
    if [ ! -f "$f" ]; then
        printf 'ENV: %s does not exist -- cannot judge an absent workflow\n' "$f"
        return 2
    fi
    hits=$(grep -cE 'cargo install (pmat|bashrs)' "$f")
    if [ "$hits" -gt 0 ]; then
        printf 'FAIL  %s: %s line(s) still `cargo install` pmat/bashrs -- assert the fleet pin, do not install\n' "$f" "$hits"
        bad=1
    fi
    if grep -E 'check_tool_versions\.sh' "$f" | grep -qE '\|\|[[:space:]]*(true|echo)'; then
        printf 'FAIL  %s: check_tool_versions.sh invocation is fail-open (`|| true` / `|| echo`)\n' "$f"
        bad=1
    fi
    if [ "$bad" -eq 0 ]; then
        printf 'PASS  %s: no cargo install of pmat/bashrs, check_tool_versions.sh is fail-closed\n' "$f"
    fi
    return "$bad"
}

# ---------------------------------------------------------------------------
if [ "${1:-}" = "--self-test" ]; then
    fails=0
    n=0
    TD=$(mktemp -d "${TMPDIR:-/tmp}/tool-versions-test.XXXXXX") || exit 2
    case "$TD" in
        /tmp/*|/var/folders/*) : ;;
        *) printf 'FAIL: mktemp -d gave %s, refusing to use it\n' "${TD:-<empty>}"; exit 1 ;;
    esac
    trap 'rm -rf "${TD:?}"' EXIT

    cat > "$TD/tools.toml" <<'TOML'
[pmat]
version = "1.2.3"

[bashrs]
version = "4.5.6"
TOML

    mkdir -p "$TD/bin-right" "$TD/bin-wrong" "$TD/bin-empty"
    # Absolute-path shebang: the row below runs these under a PATH that
    # contains ONLY the shim dir, and `/usr/bin/env bash` resolves `bash`
    # via PATH too -- so an env-shebang shim silently fails to execute
    # ("No such file or directory") instead of proving the version check.
    # A mock answers ONLY to `--version`: a bare invocation prints usage and
    # exits 2, so the spec's named mutation for BSE-10a (strip `--version`
    # from installed_version) turns every GREEN row RED. With unconditional
    # mocks that mutant passed 6/6 (measured 2026-09-07, PR #3037 review).
    for spec in "bin-right/pmat=pmat 1.2.3" "bin-right/bashrs=bashrs 4.5.6" \
                "bin-wrong/pmat=pmat 9.9.9" "bin-wrong/bashrs=bashrs 9.9.9"; do
        printf '#!/bin/bash\n[ "${1:-}" = "--version" ] || { printf "usage\\n"; exit 2; }\nprintf "%s\\n"\n' \
            "${spec#*=}" > "$TD/${spec%%=*}"
    done
    chmod +x "$TD"/bin-right/* "$TD"/bin-wrong/*

    row() { # row WANT_RC LABEL PATHVAL TOOL
        local want=$1 label=$2 pathval=$3 tool=$4 rc
        n=$((n + 1))
        # check_one's own plumbing (awk, head) is an external binary too, and
        # must resolve even while the shim dir shadows the real pmat/bashrs --
        # so /usr/bin:/bin rides along; it carries no pmat/bashrs of its own.
        ( PATH="$pathval:/usr/bin:/bin" check_one "$tool" "$TD/tools.toml" > /dev/null 2>&1 )
        rc=$?
        if [ "$rc" = "$want" ]; then
            printf 'ok    row %-2s rc=%s  %s\n' "$n" "$rc" "$label"
        else
            printf 'FAIL  row %-2s rc=%s (wanted %s)  %s\n' "$n" "$rc" "$want" "$label"
            fails=1
        fi
    }

    row 0 "pmat: right version on PATH -> GREEN"   "$TD/bin-right" pmat
    row 0 "bashrs: right version on PATH -> GREEN" "$TD/bin-right" bashrs
    row 1 "pmat: wrong version on PATH -> RED"     "$TD/bin-wrong" pmat
    row 1 "bashrs: wrong version on PATH -> RED"   "$TD/bin-wrong" bashrs
    row 1 "pmat: absent from PATH -> RED"          "$TD/bin-empty" pmat
    row 1 "bashrs: absent from PATH -> RED"        "$TD/bin-empty" bashrs

    [ "$fails" -eq 0 ] || { printf '\nSELF-TEST FAILED\n'; exit 1; }
    printf '\nSELF-TEST PASSED (%s/%s rows)\n' "$n" "$n"
    exit 0
fi

if [ "${1:-}" = "--audit-workflow" ]; then
    if [ -z "${2:-}" ]; then
        printf 'usage: %s --audit-workflow FILE\n' "$0" >&2
        exit 2
    fi
    audit_workflow "$2"
    exit $?
fi

if [ -n "${1:-}" ]; then
    printf 'usage: %s [--self-test|--audit-workflow FILE]\n' "$0" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
printf '=== pinned tool versions must match what is on PATH (check_tool_versions.sh) ===\n'

if [ ! -f "$TOOLS_TOML" ]; then
    printf 'FAIL: %s does not exist. This guard is checking nothing.\n' "$TOOLS_TOML"
    exit 1
fi

fails=0
n=0
for tool in $TOOLS; do
    n=$((n + 1))
    check_one "$tool" "$TOOLS_TOML" || fails=$((fails + 1))
done

printf '%s/%s checks, %s failed\n' "$((n - fails))" "$n" "$fails"
[ "$fails" -eq 0 ]
