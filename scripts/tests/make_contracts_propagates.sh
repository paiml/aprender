#!/usr/bin/env bash
#
# make_contracts_propagates.sh -- PVL-001 EV-4 (aprender#4168): `make contracts`
# propagates pv's exit status, and so does every other step of that recipe.
#
# THE DEFECT. The Makefile runs with `.ONESHELL:` and `.SHELLFLAGS := -o
# pipefail -c` (no -e), so a multi-line recipe is ONE shell script whose status
# is its LAST line's. `pv lint contracts/ | tail -5` failing printed five lines
# and the gate exited 0; so did a failing census, extraction, README sync and
# provenance lint. Separately, the Makefile's own `PV_BIN := cargo run ...`
# overrode an exported PV_BIN -- the one override scripts/pv_bin.sh honours --
# and handed pv_bin.sh the string "cargo run ...".
#
# HOW THIS TESTS IT, CARGO-FREE. The `contracts`, `contract-test` and
# `contract-audit` recipes and the three shell-setting lines are EXTRACTED from
# the real Makefile -- never retyped -- into a throwaway git repo, next to the
# REAL scripts/pv_bin.sh and stubs for everything else. One env var, FAIL_AT,
# makes exactly one step fail with a distinctive rc (7; cargo's 101). Each row
# asserts the VERDICT, not just "non-zero": make's own `] Error <rc>` line
# naming the step's rc (a missing Makefile is non-zero too), and that the NEXT
# step's header never printed (the recipe stopped where it failed).
#
# Every mechanism of the fix is then removed, one at a time, and the row that
# mechanism exists for must go RED -- a table never shown capable of failing
# is evidence of nothing.
#
# One row runs the REAL Makefile in the REAL tree: an exported PV_BIN must reach
# pv_bin.sh (it stops at the first step, so it writes nothing to the tree).
#
# Usage: bash scripts/tests/make_contracts_propagates.sh    (exit 0 = all rows green)

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1
MAKEFILE="$REPO_ROOT/Makefile"
TMP="$(mktemp -d)" || exit 1
if [ -n "${KEEP:-}" ]; then echo "KEEP: fixture left at $TMP" >&2; else trap 'rm -rf "$TMP"' EXIT; fi

total=0
failed=0
row() { # row <name> <0|1 verdict of the check>
    total=$((total + 1))
    if [ "$2" -eq 0 ]; then
        printf 'ok    %s\n' "$1"
    else
        failed=$((failed + 1))
        printf 'FAIL  %s\n' "$1"
    fi
}

# ---------------------------------------------------------------------------
# Extraction from the real Makefile. Each piece must be found exactly once, or
# the fixture would silently test something other than what ships.
# ---------------------------------------------------------------------------
header_line() { # header_line <ERE> -> the single matching line, or die
    n=$(grep -cE "$1" "$MAKEFILE") || n=0
    if [ "$n" != 1 ]; then
        echo "BROKE: expected exactly one line matching /$1/ in Makefile, found $n" >&2
        exit 2
    fi
    grep -E "$1" "$MAKEFILE"
}
recipe() { # recipe <target> -> the target line and its tab-indented body
    out=$(awk -v t="$1:" 'index($0, t) == 1 { f = 1; print; next }
                         f && /^\t/ { print; next }
                         f { exit }' "$MAKEFILE")
    if [ -z "$out" ] || [ "$(printf '%s\n' "$out" | wc -l)" -lt 3 ]; then
        echo "BROKE: could not extract the '$1' recipe from Makefile" >&2
        exit 2
    fi
    printf '%s\n' "$out"
}

{
    header_line '^SHELL[[:space:]]*:='
    header_line '^\.SHELLFLAGS[[:space:]]*:='
    header_line '^\.ONESHELL:'
    echo ".PHONY: contracts contract-test contract-audit"
    echo
    recipe contracts
    echo
    recipe contract-test
    echo
    recipe contract-audit
} > "$TMP/Makefile.shipped" || exit 2

DECLARED=$(awk -F'"' '/^version[[:space:]]*=/{print $2; exit}' "$REPO_ROOT/Cargo.toml")
[ -n "$DECLARED" ] || { echo "BROKE: no workspace version in Cargo.toml" >&2; exit 2; }

# ---------------------------------------------------------------------------
# The fixture: a git repo holding the extracted Makefile, the REAL pv_bin.sh,
# and stubs. FAIL_AT picks the one step that fails.
# ---------------------------------------------------------------------------
FX="$TMP/fx"
mkdir -p "$FX/scripts" "$FX/contracts" "$TMP/bin" || exit 2
cp "$REPO_ROOT/scripts/pv_bin.sh" "$FX/scripts/pv_bin.sh" || exit 2
printf '[workspace.package]\nversion = "%s"\n' "$DECLARED" > "$FX/Cargo.toml"
printf '{"n_files": 3}\n' > "$FX/contracts/census.json"
cp "$FX/contracts/census.json" "$TMP/census.golden"
for s in readme_sync:readme lint-provenance:provenance check_census_derived:derived; do
    printf '#!/usr/bin/env bash\n[ "${FAIL_AT:-}" = %s ] && { echo "STUB %s fails" >&2; exit 7; }\nexit 0\n' \
        "${s#*:}" "${s%%:*}" > "$FX/scripts/${s%%:*}.sh"
done
chmod +x "$FX/scripts/"*.sh || exit 2
( cd "$FX" && git init -q . && git add -A \
    && git -c user.name=t -c user.email=t@t commit -qm fixture ) || exit 2

# The fake pv. It identifies as the aprender verifier at the declared version,
# so the REAL pv_bin.sh accepts it; FAIL_AT=<subcommand> makes that one fail.
cat > "$TMP/bin/pv" <<EOF
#!/usr/bin/env bash
case "\$1" in
  --version) echo "pv $DECLARED (aprender provable-contracts verifier)"; exit 0 ;;
esac
if [ "\${FAIL_AT:-}" = "\$1" ]; then echo "FAKE pv \$1 fails"; exit 7; fi
if [ "\${FAIL_AT:-}" = audit-first ] && [ "\$1" = audit ] && [ "\$2" = c1 ]; then echo "FAKE pv audit c1 fails"; exit 7; fi
case "\$1" in
  census) cat "$TMP/census.golden" ;;
  audit)  echo "AUDIT \$2" ;;
  *)      echo "FAKE pv \$1 ok" ;;
esac
exit 0
EOF
# A binary pv_bin.sh must REFUSE: right name, wrong version.
printf '#!/usr/bin/env bash\necho "pv 0.0.1 (aprender provable-contracts verifier)"\nexit 0\n' > "$TMP/bin/stale-pv"
# cargo, stubbed: the fixture proves exit propagation, not the engine tests.
cat > "$TMP/bin/cargo" <<'EOF'
#!/usr/bin/env bash
if [ "${FAIL_AT:-}" = cargo ]; then echo "test result: FAILED. 0 passed; 1 failed"; exit 101; fi
echo "test result: ok. 3 passed; 0 failed"
EOF
chmod +x "$TMP/bin/pv" "$TMP/bin/stale-pv" "$TMP/bin/cargo"

# run <makefile> <target> <FAIL_AT> [PV_BIN] -> sets OUT and RC
run() {
    OUT=$(cd "$FX" && git checkout -q -- contracts/census.json \
        && PATH="$TMP/bin:$PATH" FAIL_AT="$3" PV_BIN="${4:-$TMP/bin/pv}" \
           make -s -f "$1" "$2" PV_CARGO_RUN="$TMP/bin/pv" CONTRACTS="c1 c2 c3" BINDING=b 2>&1)
    RC=$?
}
# expect_stop <makefile> <target> <FAIL_AT> <rc> <header that must NOT print> [PV_BIN]
# -> 0 iff make exits 2 with `] Error <rc>` and the next step never started.
expect_stop() {
    run "$1" "$2" "$3" "${6:-}"
    [ "$RC" -eq 2 ] || return 1
    grep -qE "\] Error $4\$" <<<"$OUT" || return 1
    grep -qF -- "$5" <<<"$OUT" && return 1
    return 0
}

M="$TMP/Makefile.shipped"

# ---------------------------------------------------------------------------
# The case table, against the recipes as shipped.
# ---------------------------------------------------------------------------
run "$M" contracts none
{ [ "$RC" -eq 0 ] && grep -qF 'test result: ok' <<<"$OUT"; }
row "contracts: every step passes -> rc 0, and the last step ran" $?

expect_stop "$M" contracts lint 7 '== census'
row "contracts: pv lint fails (rc 7, through | tail) -> Error 7, census never starts" $?

expect_stop "$M" contracts census 7 '== graph'
row "contracts: pv census fails -> Error 7, graph never starts" $?
expect_stop "$M" contracts derived 7 '== graph'
row "contracts: check_census_derived fails -> Error 7, graph never starts" $?

expect_stop "$M" contracts extract 7 '== README'
row "contracts: pv extract --check fails -> Error 7, README never starts" $?

expect_stop "$M" contracts readme 7 '== provenance'
row "contracts: readme_sync fails -> Error 7, provenance never starts" $?

expect_stop "$M" contracts provenance 7 '== contract engine tests'
row "contracts: lint-provenance fails -> Error 7, engine tests never start" $?

expect_stop "$M" contracts cargo 101 '@@unreachable@@'
row "contracts: engine tests fail (rc 101, through grep | tail) -> Error 101" $?

expect_stop "$M" contracts none 1 '== census' "$TMP/bin/stale-pv"
{ [ $? -eq 0 ] && grep -qF 'STALE pv BINARY' <<<"$OUT"; }
row "contracts: pv_bin.sh refuses a stale PV_BIN -> Error 1 naming it, census never starts" $?

expect_stop "$M" contract-test cargo 101 'Contract tests passed'
row "contract-test: cargo test fails -> Error 101, no 'passed' line" $?

expect_stop "$M" contract-audit audit-first 7 'Binding audit complete'
{ [ $? -eq 0 ] && [ "$(printf '%s\n' "$OUT" | grep -cE '^(AUDIT c[23]|FAKE pv audit c1 fails)$')" -eq 3 ]; }
row "contract-audit: the FIRST of three audits fails -> Error 7, and all three still ran" $?

# ---------------------------------------------------------------------------
# The same rows against mutants: each mechanism removed must turn its row RED.
# mutant <name> <sed expression> -> path of the mutated Makefile, or die if the
# expression changed nothing (a mutant identical to the original proves nothing).
# ---------------------------------------------------------------------------
mutant() {
    sed -E "$2" "$M" > "$TMP/Makefile.$1"
    if cmp -s "$M" "$TMP/Makefile.$1"; then
        echo "BROKE: mutant '$1' did not change the Makefile" >&2
        exit 2
    fi
    printf '%s\n' "$TMP/Makefile.$1"
}

m=$(mutant no-errexit '/^contracts:/,/^$/{/^\t@set -e$/d}') || exit 2
# The provenance row, not readme_sync's: since #3569 readme_sync also runs inside
# the census step, whose own `|| exit` stops the recipe with or without errexit.
expect_stop "$m" contracts provenance 7 '== contract engine tests'
went_red=$?; [ "$went_red" -ne 0 ]; row "mutant: contracts without set -e -> the lint-provenance row goes RED" $?

m=$(mutant no-exit-on-lint '/lint contracts\//s/ \|\| exit$//') || exit 2
expect_stop "$m" contracts none 1 '== census' "$TMP/bin/stale-pv"
went_red=$?; [ "$went_red" -ne 0 ]; row "mutant: the lint line without || exit -> the stale-PV_BIN row goes RED" $?

# The d8 shape: `exit $$rc` in a BRACE group runs in the recipe's own shell, so a
# PASSING lint exits 0 there and census, graph, README, provenance and the engine
# tests never run -- a gate that always passes. Only a subshell confines the exit.
m=$(mutant brace-lint '/lint contracts\//{s/pv_bin\.sh \&\& \( /pv_bin.sh \&\& { /;s/exit \$\$rc \) \|\| exit$/exit $$rc; } || exit/}') || exit 2
run "$m" contracts none
{ [ "$RC" -eq 0 ] && grep -qF 'test result: ok' <<<"$OUT"; }
went_red=$?; [ "$went_red" -ne 0 ]; row "mutant: the lint step's exit in a { } group, not ( ) -> the every-step-passes row goes RED" $?

m=$(mutant no-pipefail 's/-o pipefail //') || exit 2
expect_stop "$m" contracts cargo 101 '@@unreachable@@'
went_red=$?; [ "$went_red" -ne 0 ]; row "mutant: .SHELLFLAGS without pipefail -> the engine-tests row (cargo | grep | tail) goes RED; pv lint no longer pipes (d8)" $?

m=$(mutant no-accumulator 's/ \|\| rc=\$\$\?;/;/') || exit 2
expect_stop "$m" contract-audit audit-first 7 'Binding audit complete'
went_red=$?; [ "$went_red" -ne 0 ]; row "mutant: contract-audit without its rc accumulator -> the audit row goes RED" $?

# ---------------------------------------------------------------------------
# The REAL Makefile in the REAL tree: an exported PV_BIN reaches pv_bin.sh. The
# fake fails `lint`, the first step, so nothing in the tree is written.
# Mutant: the pre-fix variable name, which shadows the exported PV_BIN.
# ---------------------------------------------------------------------------
real_row() { # real_row <makefile> -> 0 iff Error 7 from the fake's lint
    OUT=$(cd "$REPO_ROOT" && FAIL_AT=lint PV_BIN="$TMP/bin/pv" make -s -f "$1" contracts 2>&1)
    RC=$?
    [ "$RC" -eq 2 ] && grep -qE '\] Error 7$' <<<"$OUT" \
        && grep -qF 'FAKE pv lint fails' <<<"$OUT"
}
real_row "$MAKEFILE"
row "real tree: exported PV_BIN reaches pv_bin.sh -> the fake's lint rc 7 is make's Error 7" $?

sed -E 's/^PV_CARGO_RUN :=/PV_BIN :=/' "$MAKEFILE" > "$TMP/Makefile.shadow"
if cmp -s "$MAKEFILE" "$TMP/Makefile.shadow"; then
    echo "BROKE: shadow mutant did not change the Makefile" >&2
    exit 2
fi
real_row "$TMP/Makefile.shadow"
went_red=$?; [ "$went_red" -ne 0 ]; row "mutant: a makefile PV_BIN := assignment -> the real-tree row goes RED" $?

echo "make_contracts_propagates: $total rows, $failed failed"
[ "$failed" -eq 0 ]
