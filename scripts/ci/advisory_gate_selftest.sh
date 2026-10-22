#!/usr/bin/env bash
# advisory_gate_selftest.sh -- FALSIFY-ADG-003/004 (contracts/advisory-gate-diff-v1.yaml): run
# advisory_gate.sh against a stub `cargo` and read what it asked cargo to do.
#   ADG-003  every advisories scan carries --all-features (the optional-feature crate is in scope)
#   ADG-004  ONE advisory-db fetch; both scans read it (--disable-fetch on cargo-deny <= 0.19, --offline on 0.20) with the same --config
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
    printf '#!/usr/bin/env bash\nexec "$(dirname "$0")/cargo" deny "$@"\n' >"$d/bin/cargo-deny"; chmod +x "$d/bin/cargo-deny"
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

gate_rc() { # prints the gate's EXIT CODE under a scenario: STUB_PR_FIND=1 (the PR tree's scan reports an error finding, rc 1),
    # WIDEN=1 (the PR adds an advisory id to deny.toml ignore), EV=<event>, MG=<MG_BASE_SHA>
    local d repo rc=0
    d=$(mktemp -d)
    # shellcheck disable=SC2064
    trap "rm -rf -- '${d:?}'" RETURN
    repo="$d/repo"; mkdir -p "$repo" "$d/bin" "$d/home" "$d/scripts"
    cp "$GATE_SRC" "$d/scripts/advisory_gate.sh"; cp "$HERE/advisory_diff.sh" "$d/scripts/advisory_diff.sh"
    cat >"$d/bin/cargo" <<'STUB'
#!/usr/bin/env bash
case "$*" in *--help*) exit 0;; esac
case "$* " in
    *" check "*)
        if [ -n "${STUB_ALL_RC:-}" ]; then exit "$STUB_ALL_RC"; fi
        if [ "${STUB_PR_FIND:-0}" = 1 ] && [[ "$PWD" != *advgate-base-* ]]; then
            echo '{"type":"diagnostic","fields":{"code":"vulnerability","severity":"error","advisory":{"id":"RUSTSEC-2026-0378"},"graphs":[{"Krate":{"name":"wasmtime","version":"48.0.6"}}]}}' >&2
            exit 1
        fi ;;
esac
exit 0
STUB
    chmod +x "$d/bin/cargo"
    printf '#!/usr/bin/env bash\nexec "$(dirname "$0")/cargo" deny "$@"\n' >"$d/bin/cargo-deny"; chmod +x "$d/bin/cargo-deny"
    (
        cd "$repo"
        git init -q . && printf '[advisories]\n' >deny.toml && git add deny.toml
        git update-ref refs/remotes/origin/main "$(git -c user.name=t -c user.email=t@t commit-tree -m base "$(git write-tree)")"
        git update-ref HEAD refs/remotes/origin/main
        if [ "${WIDEN:-0}" = 1 ]; then printf 'ignore = ["RUSTSEC-2026-0378"]\n' >>deny.toml; fi
        PATH="$d/bin:$PATH" HOME="$d/home" TMPDIR="$d" GITHUB_EVENT_NAME="${EV:-pull_request}" MG_BASE_SHA="${MG:-}" \
            STUB_PR_FIND="${STUB_PR_FIND:-0}" bash "$d/scripts/advisory_gate.sh" >/dev/null 2>&1
    ) || rc=$?
    echo "$rc"
}

# scenario table: name, wanted exit code, env assignments. Each row also names the advisory_diff.sh call that decides it.
scenarios() { # prints the first defect, or ok
    local got
    got=$(gate_rc); [ "$got" = 0 ] || { echo "clean PR: gate exit $got, want 0"; return; }
    got=$(STUB_PR_FIND=1 gate_rc); [ "$got" = 1 ] || { echo "verdict: PR-introduced advisory: gate exit $got, want 1"; return; }
    got=$(WIDEN=1 gate_rc); [ "$got" = 1 ] || { echo "ignores-added: PR widens deny.toml ignore: gate exit $got, want 1"; return; }
    got=$(EV=merge_group MG='' gate_rc); [ "$got" = 75 ] || { echo "base-ref: merge_group without base_sha: gate exit $got, want 75"; return; }
    got=$(EV=push STUB_ALL_RC=1 gate_rc); [ "$got" = 1 ] || { echo "push-path: a failing full advisories check: gate exit $got, want 1"; return; }
    echo ok
}

verdict_of() { # [old|new] (cargo-deny flavour the stub played); stdin: stub log -> "ok" or the first defect
    local log fetch checks nall nfd ncfg
    log=$(cat)
    [ "$log" != gate-did-not-pass ] || { echo "gate did not pass against the stub"; return; }
    fetch=$(grep -cE '^deny (--config [^ ]+ )?fetch' <<<"$log" || true)
    checks=$(grep -c 'check' <<<"$(grep -vE '^deny (--config [^ ]+ )?fetch' <<<"$log")" || true)
    nall=$(grep -c -- '--all-features' <<<"$(grep -vE '^deny (--config [^ ]+ )?fetch' <<<"$log")" || true)
    nfd=$(grep -cE -- '--disable-fetch' <<<"$(grep -vE '^deny (--config [^ ]+ )?fetch' <<<"$log")" || true)
    ncfg=$(grep -vE '^deny (--config [^ ]+ )?fetch' <<<"$log" | grep -o -- '--config [^ ]*' | sort -u | wc -l)
    [ "$checks" -eq 2 ] || { echo "expected 2 scans, saw $checks"; return; }
    [ "$nall" -eq 2 ] || { echo "ADG-003: $nall of 2 scans carry --all-features"; return; }
    [ "$fetch" -eq 1 ] || { echo "ADG-004: $fetch advisory-db fetches, want 1"; return; }
    if [ "${1:-new}" = old ]; then [ "$nfd" -eq 2 ] || { echo "ADG-004: $nfd of 2 scans carry --disable-fetch"; return; }; fi
    # 0.20 (m70, #5793): both scans run --offline (one advisory-db head), and `cargo fetch --locked` runs on BOTH trees BEFORE the first scan,
    # else --offline on a cold registry cache exits rc=1 with no diagnostics. The stub log is in call order.
    if [ "${1:-new}" != old ]; then
        [ "$(grep -c -- '--offline' <<<"$log")" -eq 2 ] || { echo "ADG-004: not both scans carry --offline"; return; }
        [ "$(grep -c '^fetch --locked$' <<<"$log")" -eq 2 ] || { echo "ADG-004: cargo fetch --locked is not run once per tree"; return; }
        [ "$(grep -n '^fetch --locked$' <<<"$log" | tail -1 | cut -d: -f1)" -lt "$(grep -n -- '--offline' <<<"$log" | head -1 | cut -d: -f1)" ] || { echo "ADG-004: cargo fetch --locked runs after a scan"; return; }
    fi
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
    got=$(STUB_OLD=1 run_gate | verdict_of old)
    [ "$got" = ok ] || { echo "SELF-TEST FAIL: real gate vs cargo-deny <= 0.19 flags: $got" >&2; bad=1; }
    mutant() { # name sed-expr want-substring
        sed -e "$2" "$GATE_SRC" >"$d/m.sh"
        cmp -s "$d/m.sh" "$GATE_SRC" && { echo "SELF-TEST FAIL: mutant '$1' changed nothing" >&2; bad=1; return; }
        m=$(STUB_OLD="${5:-0}" run_gate_with "$d/m.sh" | verdict_of "${4:-new}")
        [[ "$m" == *"$3"* ]] || { echo "SELF-TEST FAIL: mutant '$1' stayed green: $m" >&2; bad=1; }
    }
    mutant 'scan without --all-features' 's/ --all-features//' 'ADG-003'
    mutant 'scan reads the net (no --disable-fetch)' 's/ --disable-fetch//' 'ADG-004' old 1
    mutant 'scan not offline (0.20)' 's/--locked --offline --all-features/--locked --all-features/' 'ADG-004'
    mutant 'no cargo fetch --locked before the scans (0.20)' 's/cargo fetch --locked)/true)/' 'ADG-004'
    mutant 'second fetch before the base scan' 's|^BASERC=|cargo-deny fetch --config "$W/deny.toml" db; BASERC=|' 'ADG-004'
    # The gate's three advisory_diff.sh calls (base-ref, ignores-added, verdict) each decide a scenario: replaced by `true`, each must go red.
    got=$(scenarios); [ "$got" = ok ] || { echo "SELF-TEST FAIL: real gate scenarios: $got" >&2; bad=1; }
    smut() { # name sed-expr want-substring
        sed -e "$2" "$GATE_SRC" >"$d/s.sh"
        cmp -s "$d/s.sh" "$GATE_SRC" && { echo "SELF-TEST FAIL: mutant '$1' changed nothing" >&2; bad=1; return; }
        m=$(GATE_SRC="$d/s.sh" scenarios)
        [[ "$m" == *"$3"* ]] || { echo "SELF-TEST FAIL: mutant '$1' stayed green: $m" >&2; bad=1; }
    }
    smut 'verdict call replaced by true' 's|^bash "\$GATE" verdict .*|true|' 'verdict:'
    smut 'ignores-added call replaced by true' 's|^bash "\$GATE" ignores-added .*|true|' 'ignores-added:'
    smut 'base-ref call replaced by true' 's|^bash "\$HERE/advisory_diff.sh" base-ref .*|true|' 'base-ref:'
    smut 'push-path full check replaced by exit 0' 's|^    \*) exec cargo-deny --all-features check advisories ;;|    *) exit 0 ;;|' 'push-path:'
    # ADG-003 on the other two scan sites: the push path and the exemption-liveness guard.
    local f
    for f in "$GATE_SRC" "${EXEMPT_SRC:-scripts/check_deny_exemptions_live.sh}"; do
        [ -f "$f" ] || { echo "SELF-TEST FAIL: ADG-003: $f does not exist, so its scan sites cannot be judged" >&2; bad=1; continue; }
        scans=$(grep -v 'printf' "$f" | grep -E '^[^#]*cargo[ -]deny .*check .*advisories' || true)
        if [ -n "$scans" ] && grep -qv -- '--all-features' <<<"$scans"; then
            echo "SELF-TEST FAIL: ADG-003: $f scans advisories without --all-features" >&2; bad=1
        fi
    done
    [ "$bad" -eq 0 ] && echo "advisory_gate self-test: 2 real (cargo-deny 0.19 and 0.20 flags) + 5 scenarios + 9 mutants + 2 sites ok"
    return "$bad"
}

run_gate_with() { GATE_SRC="$1"; run_gate; }
main
