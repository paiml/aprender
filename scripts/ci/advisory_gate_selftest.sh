#!/usr/bin/env bash
# advisory_gate_selftest.sh -- FALSIFY-ADG-003/004 (contracts/advisory-gate-diff-v1.yaml): run
# advisory_gate.sh against a stub `cargo` and read what it asked cargo to do.
#   ADG-003  every advisories scan carries --all-features (the optional-feature crate is in scope)
#   ADG-004  ONE advisory-db fetch; both scans read it with --disable-fetch and the same --config
# usage: advisory_gate_selftest.sh [GATE_SCRIPT]   (default: the sibling advisory_gate.sh)
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GATE_SRC="${1:-$HERE/advisory_gate.sh}"

run_gate() { # prints the stub's argv log
    local d repo log
    d=$(mktemp -d)
    # shellcheck disable=SC2064
    trap "rm -rf -- '${d:?}'" RETURN
    repo="$d/repo"; log="$d/cargo.log"; mkdir -p "$repo" "$d/bin" "$d/home" "$d/scripts"
    cp "$GATE_SRC" "$d/scripts/advisory_gate.sh"; cp "$HERE/advisory_diff.sh" "$d/scripts/advisory_diff.sh"
    # `cargo deny check --help` is the version probe: it is answered, never logged (STUB_OLD=1 plays cargo-deny <= 0.19)
    printf '#!/usr/bin/env bash\ncase "$*" in *--help*) [ "${STUB_OLD:-0}" = 1 ] && echo -- --config; exit 0;; esac\necho "$*" >>"%s"\nexit 0\n' "$log" >"$d/bin/cargo"; chmod +x "$d/bin/cargo"
    (
        cd "$repo"
        git init -q . && printf '[advisories]\n' >deny.toml && git add deny.toml
        # plumbing, not `git commit`: a throwaway repo must not run the global commit hooks
        git update-ref refs/remotes/origin/main "$(git -c user.name=t -c user.email=t@t commit-tree -m base "$(git write-tree)")"
        git update-ref HEAD refs/remotes/origin/main
        PATH="$d/bin:$PATH" HOME="$d/home" TMPDIR="$d" GITHUB_EVENT_NAME=pull_request \
            bash "$d/scripts/advisory_gate.sh" >/dev/null 2>&1
    ) || { echo "gate-did-not-pass"; return 0; }
    cat "$log"
}

verdict_of() { # stdin: stub log -> "ok" or the first defect
    local log fetch checks nall nfd ncfg
    log=$(cat)
    [ "$log" != gate-did-not-pass ] || { echo "gate did not pass against the stub"; return; }
    fetch=$(grep -cE '^deny (--config [^ ]+ )?fetch' <<<"$log" || true)
    checks=$(grep -c 'check' <<<"$(grep -vE '^deny (--config [^ ]+ )?fetch' <<<"$log")" || true)
    nall=$(grep -c -- '--all-features' <<<"$(grep -vE '^deny (--config [^ ]+ )?fetch' <<<"$log")" || true)
    nfd=$(grep -cE -- '--disable-fetch|--offline' <<<"$(grep -vE '^deny (--config [^ ]+ )?fetch' <<<"$log")" || true)
    ncfg=$(grep -vE '^deny (--config [^ ]+ )?fetch' <<<"$log" | grep -o -- '--config [^ ]*' | sort -u | wc -l)
    [ "$checks" -eq 2 ] || { echo "expected 2 scans, saw $checks"; return; }
    [ "$nall" -eq 2 ] || { echo "ADG-003: $nall of 2 scans carry --all-features"; return; }
    [ "$fetch" -eq 1 ] || { echo "ADG-004: $fetch advisory-db fetches, want 1"; return; }
    [ "$nfd" -eq 2 ] || { echo "ADG-004: $nfd of 2 scans carry --disable-fetch/--offline"; return; }
    [ "$ncfg" -eq 1 ] || { echo "ADG-004: scans use $ncfg different --config files"; return; }
    echo ok
}

main() {
    local bad=0 d m got
    d=$(mktemp -d)
    # shellcheck disable=SC2064
    trap "rm -rf -- '${d:?}'" EXIT
    got=$(run_gate | verdict_of)
    [ "$got" = ok ] || { echo "SELF-TEST FAIL: real gate: $got" >&2; bad=1; }
    got=$(STUB_OLD=1 run_gate | verdict_of)
    [ "$got" = ok ] || { echo "SELF-TEST FAIL: real gate vs cargo-deny <= 0.19 flags: $got" >&2; bad=1; }
    mutant() { # name sed-expr want-substring
        sed -e "$2" "$GATE_SRC" >"$d/m.sh"
        cmp -s "$d/m.sh" "$GATE_SRC" && { echo "SELF-TEST FAIL: mutant '$1' changed nothing" >&2; bad=1; return; }
        m=$(run_gate_with "$d/m.sh" | verdict_of)
        [[ "$m" == *"$3"* ]] || { echo "SELF-TEST FAIL: mutant '$1' stayed green: $m" >&2; bad=1; }
    }
    mutant 'scan without --all-features' 's/ --all-features//' 'ADG-003'
    mutant 'scan reads the net (no --disable-fetch)' 's/ --disable-fetch//;s/ --offline//' 'ADG-004'
    mutant 'second fetch before the base scan' 's|^BASERC=|cargo deny fetch --config "$W/deny.toml" db; BASERC=|' 'ADG-004'
    # ADG-003 on the other two scan sites: the push path and the exemption-liveness guard.
    local f
    for f in "$GATE_SRC" "${EXEMPT_SRC:-scripts/check_deny_exemptions_live.sh}"; do
        if grep -v printf "$f" | grep -E '^[^#]*cargo deny .*check .*advisories' | grep -qv -- '--all-features'; then
            echo "SELF-TEST FAIL: ADG-003: $f scans advisories without --all-features" >&2; bad=1
        fi
    done
    [ "$bad" -eq 0 ] && echo "advisory_gate self-test: 2 real (cargo-deny 0.19 and 0.20 flags) + 3 mutants + 2 sites ok"
    return "$bad"
}

run_gate_with() { GATE_SRC="$1"; run_gate; }
main
