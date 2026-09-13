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


# check_headers TOMLFILE -- every `# tool_version=<tool> <version>` header on a
# ratchet baseline (BSE-10a, lib_baseline_ratchet.sh) that names a PINNED tool
# must carry the pinned version. Two mirrors of one fleet pin drifted apart on
# 2026-09-07 (tools.toml 3.39.0, complexity_baseline.txt 3.37.0) and the D2
# complexity ratchet refused every merge as an instrument mismatch; this makes
# the pin bump and the header move together or fail here.
check_headers() { # check_headers TOMLFILE
    local f=$1 bad=0 n=0 total=0 b tool ver pinned
    # VACUITY FLOOR (#3217). `n` below counts only baselines naming a PINNED
    # tool, and nothing required that count to hold. Delete the header from
    # shell_lint_baseline.txt — which is exactly what
    # `check_shell_lint_ratchet.sh --update` did — and this row goes from
    #     3 baseline header(s) name a pinned tool; all agree with tools.toml
    # to
    #     2 baseline header(s) name a pinned tool; all agree with tools.toml
    # and still PASSES. A third of the universe vanished and the audit reported
    # agreement over what was left.
    #
    # The floor is not a magic number, because a magic number rots the moment a
    # baseline is added or retired. Every baseline must DECLARE its instrument,
    # in one of the two spellings this repo uses:
    #     # tool_version=<tool> <version>   lib_baseline_ratchet.sh, 16 files
    #     pmat_version: <version>           the richer count/version/basis
    #                                       schema of hardcoded_path_shipped
    # A count whose analyser nobody recorded is not a baseline — it is a number.
    # Adding one now goes RED on the commit that adds it.
    for b in scripts/*baseline*.txt; do
        [ -f "$b" ] || continue
        total=$((total + 1))
        grep -qE '^#[[:space:]]*tool_version=|^[[:space:]]*pmat_version:' "$b" && continue
        printf 'FAIL  header   %s declares no instrument — a count with no\n' "$b"
        printf '               analyser version behind it is not a baseline\n'
        bad=1
    done
    if [ "$total" -eq 0 ]; then
        printf 'FAIL  header   no baseline files found — this audit swept nothing\n'
        return 1
    fi
    for b in scripts/*baseline*.txt; do
        [ -f "$b" ] || continue
        read -r tool ver < <(sed -n 's/^#[[:space:]]*tool_version=\([a-z]*\) \([0-9][0-9.]*\).*/\1 \2/p' "$b" | head -1)
        [ -n "$tool" ] || continue
        case " $TOOLS " in *" $tool "*) : ;; *) continue ;; esac
        n=$((n + 1))
        pinned=$(pinned_version "$tool" "$f")
        if [ "$ver" != "$pinned" ]; then
            printf 'FAIL  header   %s records %s %s, tools.toml pins %s\n' "$b" "$tool" "$ver" "$pinned"
            bad=1
        fi
    done
    printf '%s of %s baseline(s) name a pinned tool; %s\n' "$n" "$total" \
        "$([ "$bad" -eq 0 ] && echo 'all declare an instrument and agree with tools.toml' || echo 'MISMATCH')"
    return "$bad"
}
# audit_workflow FILE -- REDs a workflow that still installs pmat/bashrs, or
# that calls this script's own invocation in a way that can fail open.
audit_workflow() { # audit_workflow FILE
    local f=$1 bad=0 hits
    if [ ! -f "$f" ]; then
        printf 'ENV: %s does not exist -- cannot judge an absent workflow\n' "$f"
        return 2
    fi
    # COMMENTS ARE NOT CODE. A line whose first non-space character is `#` is a
    # shell comment inside `run: |` (or a YAML comment outside it) and installs
    # nothing. Counting it means a step that EXPLAINS why it no longer installs
    # trips the guard that wanted it to stop — measured 2026-09-13, when the
    # comment `# Was: \`cargo install pmat … || true\`` kept book.yml RED after the
    # install itself was gone. The wrong remedy is to reword the comment; a
    # detector that cannot tell prose from code teaches people to hide the prose.
    hits=$(grep -vE '^[[:space:]]*#' "$f" | grep -cE 'cargo install (pmat|bashrs)')
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

# audit_all_workflows -- every .github/workflows/*.yml, not one named by hand.
#
# `--audit-workflow FILE` shipped with this guard and was invoked by NOTHING:
# `grep -rn audit-workflow .github/ scripts/` found only this script's own usage
# strings. A detector that no caller runs is the same nothing as no detector, and
# it is why book.yml kept `cargo install bashrs` for the whole life of the policy
# it violates. The universe is a glob, so a workflow added tomorrow is audited the
# day it lands and nobody has to remember to add it.
audit_all_workflows() {
    _wfdir="$REPO_ROOT/.github/workflows"
    [ -d "$_wfdir" ] || { printf 'FAIL: %s does not exist; the workflow audit checked nothing.\n' "$_wfdir"; return 1; }
    _seen=0
    _bad=0
    for _wf in "$_wfdir"/*.yml "$_wfdir"/*.yaml; do
        [ -f "$_wf" ] || continue
        _seen=$((_seen + 1))
        audit_workflow "$_wf" || _bad=$((_bad + 1))
    done
    # Vacuity: a sweep that audited no file is not a clean sweep.
    if [ "$_seen" -eq 0 ]; then
        printf 'FAIL: no workflow files under %s; the audit is vacuous.\n' "$_wfdir"
        return 1
    fi
    printf '%s workflow(s) audited, %s still installing pmat/bashrs\n' "$_seen" "$_bad"
    [ "$_bad" -eq 0 ]
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
    # header rows run from a scratch tree whose scripts/ holds only fixtures.
    mkdir -p "$TD/tree/scripts"
    printf '# tool_version=pmat 1.2.3\n1\n' > "$TD/tree/scripts/a_baseline.txt"
    printf '# tool_version=none (grep)\n1\n' > "$TD/tree/scripts/b_baseline.txt"
    n=$((n + 1)); if ( cd "$TD/tree" && check_headers "$TD/tools.toml" > /dev/null 2>&1 ); then printf 'ok    row %-2s rc=0  baseline headers at the pin -> GREEN\n' "$n"; else printf 'FAIL  row %-2s  headers at the pin should be GREEN\n' "$n"; fails=1; fi
    printf '# tool_version=pmat 1.2.2\n1\n' > "$TD/tree/scripts/a_baseline.txt"
    n=$((n + 1)); if ( cd "$TD/tree" && check_headers "$TD/tools.toml" > /dev/null 2>&1 ); then printf 'FAIL  row %-2s  a header behind the pin should be RED\n' "$n"; fails=1; else printf 'ok    row %-2s rc=1  a baseline header behind the pin -> RED\n' "$n"; fi

    # --- the vacuity floor (#3217) ------------------------------------------
    # These three rows are the ones that were missing. Without them the audit
    # above passed on a SHRINKING universe: deleting a header took it from
    # "3 baseline header(s) ... all agree" to "2 ... all agree", green both
    # times. Each row here removes something and requires RED.
    printf '# tool_version=pmat 1.2.3\n1\n' > "$TD/tree/scripts/a_baseline.txt"
    printf '1\n' > "$TD/tree/scripts/c_baseline.txt"
    n=$((n + 1)); if ( cd "$TD/tree" && check_headers "$TD/tools.toml" > /dev/null 2>&1 ); then printf 'FAIL  row %-2s  a baseline declaring NO instrument must be RED\n' "$n"; fails=1; else printf 'ok    row %-2s rc=1  a baseline with no instrument header -> RED\n' "$n"; fi
    # The floor must accept BOTH spellings, or it would force a false edit on
    # hardcoded_path_shipped_baseline.txt, whose count/pmat_version/basis schema
    # records the instrument more strictly than the one-line header does.
    printf 'count: 1\npmat_version: INVALID\nbasis: UNMEASURED\n' > "$TD/tree/scripts/c_baseline.txt"
    n=$((n + 1)); if ( cd "$TD/tree" && check_headers "$TD/tools.toml" > /dev/null 2>&1 ); then printf 'ok    row %-2s rc=0  the `pmat_version:` schema also declares an instrument\n' "$n"; else printf 'FAIL  row %-2s  the count/version/basis schema must be accepted\n' "$n"; fails=1; fi
    rm -f "$TD/tree/scripts/c_baseline.txt"
    # Sweeping nothing is the oldest way for an audit to be green.
    mkdir -p "$TD/empty/scripts"
    n=$((n + 1)); if ( cd "$TD/empty" && check_headers "$TD/tools.toml" > /dev/null 2>&1 ); then printf 'FAIL  row %-2s  an audit that found NO baselines must not pass\n' "$n"; fails=1; else printf 'ok    row %-2s rc=1  no baseline files at all -> RED (vacuity)\n' "$n"; fi

    # --- audit_workflow: prose vs code, both polarities ------------------------
    # The regex shipped without a case table and immediately produced a false
    # positive on a comment. Every row here is a shape that was, or could be,
    # mistaken for the other.
    mkdir -p "$TD/wf"
    printf 'jobs:\n  a:\n    steps:\n      - run: |\n          cargo install bashrs --locked\n' > "$TD/wf/installs.yml"
    n=$((n + 1)); if audit_workflow "$TD/wf/installs.yml" > /dev/null 2>&1; then printf 'FAIL  row %-2s  a real `cargo install bashrs` must be RED\n' "$n"; fails=1; else printf 'ok    row %-2s rc=1  a real cargo install bashrs -> RED\n' "$n"; fi
    printf 'jobs:\n  a:\n    steps:\n      - run: |\n          # Was: `cargo install pmat --locked || true`, now the pin\n          bash scripts/check_tool_versions.sh\n' > "$TD/wf/comment.yml"
    n=$((n + 1)); if audit_workflow "$TD/wf/comment.yml" > /dev/null 2>&1; then printf 'ok    row %-2s rc=0  a COMMENT naming cargo install pmat -> GREEN\n' "$n"; else printf 'FAIL  row %-2s  a comment installs nothing and must be GREEN\n' "$n"; fails=1; fi
    printf 'jobs:\n  a:\n    steps:\n      - run: |\n          cargo install --path crates/apr-cli --locked\n' > "$TD/wf/pathinstall.yml"
    n=$((n + 1)); if audit_workflow "$TD/wf/pathinstall.yml" > /dev/null 2>&1; then printf 'ok    row %-2s rc=0  `cargo install --path` (our own binary) -> GREEN\n' "$n"; else printf 'FAIL  row %-2s  a --path install is not a pinned-tool install\n' "$n"; fails=1; fi
    printf 'jobs:\n  a:\n    steps:\n      - run: |\n          cargo install cargo-llvm-cov --locked\n' > "$TD/wf/other.yml"
    n=$((n + 1)); if audit_workflow "$TD/wf/other.yml" > /dev/null 2>&1; then printf 'ok    row %-2s rc=0  installing a DIFFERENT tool -> GREEN\n' "$n"; else printf 'FAIL  row %-2s  only pmat/bashrs are pinned here\n' "$n"; fails=1; fi
    printf 'jobs:\n  a:\n    steps:\n      - run: bash scripts/check_tool_versions.sh || true\n' > "$TD/wf/failopen.yml"
    n=$((n + 1)); if audit_workflow "$TD/wf/failopen.yml" > /dev/null 2>&1; then printf 'FAIL  row %-2s  a `|| true` on the guard must be RED\n' "$n"; fails=1; else printf 'ok    row %-2s rc=1  fail-open invocation of the guard -> RED\n' "$n"; fi
    # Vacuity of the sweep itself: an empty workflows dir is not a clean sweep.
    n=$((n + 1)); if ( REPO_ROOT="$TD/emptyrepo"; mkdir -p "$REPO_ROOT/.github/workflows"; audit_all_workflows > /dev/null 2>&1 ); then printf 'FAIL  row %-2s  a sweep over zero workflows must be RED\n' "$n"; fails=1; else printf 'ok    row %-2s rc=1  zero workflows audited -> RED (vacuity)\n' "$n"; fi

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
n=$((n + 1))
check_headers "$TOOLS_TOML" || fails=$((fails + 1))
n=$((n + 1))
audit_all_workflows || fails=$((fails + 1))

printf '%s/%s checks, %s failed\n' "$((n - fails))" "$n" "$fails"
[ "$fails" -eq 0 ]
