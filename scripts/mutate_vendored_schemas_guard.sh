#!/usr/bin/env bash
# mutate_vendored_schemas_guard.sh - prove check_vendored_schemas.sh turns RED.
#
# PRREV-002. A guard's own GREEN is worth nothing until someone has watched it
# go RED for each defect it claims to catch. `--self-test` inside the guard
# covers its POSITIVE CONTROLS; this covers the real checks, by mutating a
# throwaway copy of the tree and requiring exit 1 from every mutant.
#
# M0 is the discrimination case: the UNMUTATED copy must be GREEN. Without it,
# "the guard exits 1 unconditionally" scores a perfect kill rate.
#
# Every mutant also PROVES ITS MUTATION ENGAGED before its verdict is read.
# The first draft of M4 used `jq '.$defs...'`, which is a jq syntax error; jq
# exited non-zero, the tree was never mutated, the guard correctly returned 0,
# and the harness recorded "MUTANT SURVIVED". A guard verdict read from a
# mutation that never applied is the same defect class as a benchmark that
# never ran the comparator.
#
# Exit 0 = baseline GREEN and every mutant RED.
# Exit 1 = a mutant survived, a mutation failed to engage, or the baseline
#          was not GREEN.
# Exit 2 = the environment could not run the guard at all.

set -euo pipefail

cd "$(dirname "$0")/.." || exit 2
REPO="$PWD"

INTOTO="schemas/in-toto-statement-v1.json"
SARIF="schemas/sarif-2.1.0.json"

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK:?}"' EXIT

killed=0
survived=0
notengaged=0

# Stage a throwaway copy of everything the guard reads.
stage() {
    local dir="$WORK/$1"
    rm -rf "${dir:?}"
    mkdir -p "$dir"
    cp -r "$REPO/schemas" "$REPO/scripts" "$REPO/tests" "$dir/"
    printf '%s' "$dir"
}

# Re-record MANIFEST + sources.json so a mutant tests ONE thing. Without this,
# every schema edit trips the integrity check first and the later checks are
# never exercised - a kill for the wrong reason.
rehash() {
    local dir="$1" h
    ( cd "$dir" && sha256sum "$INTOTO" "$SARIF" > schemas/MANIFEST.sha256 )
    h="$( cd "$dir" && sha256sum "$INTOTO" | cut -d' ' -f1 )"
    ( cd "$dir" && jq --arg h "$h" '(.schemas[] | select(.path=="'"$INTOTO"'") | .sha256) = $h' schemas/sources.json > s.tmp && mv s.tmp schemas/sources.json )
}

# run <name> <expected-exit> <dir>
run_guard() {
    local name="$1" want="$2" dir="$3" rc=0
    ( cd "$dir" && ./scripts/check_vendored_schemas.sh ) > "$WORK/$name.out" 2>&1 || rc=$?
    if [ "$rc" -eq "$want" ]; then
        printf '  %-34s exit=%s  as required\n' "$name" "$rc"
        killed=$((killed + 1))
        return 0
    fi
    printf '  %-34s exit=%s  WANTED %s  <-- SURVIVED\n' "$name" "$rc" "$want"
    sed 's/^/      /' "$WORK/$name.out" | head -25
    survived=$((survived + 1))
    return 1
}

# engaged <name> <predicate-command...> - the mutation must be observable.
engaged() {
    local name="$1"; shift
    if "$@"; then
        return 0
    fi
    printf '  %-34s MUTATION DID NOT ENGAGE - verdict not read\n' "$name"
    notengaged=$((notengaged + 1))
    return 1
}

absent_from() {
    local needle="$1" file="$2" n
    n="$(grep -c "$needle" "$file" || true)"
    [ "${n:-0}" -eq 0 ]
}

dir_has_no_files() {
    local n
    n="$(find "$1" -type f | wc -l)"
    [ "$n" -eq 0 ]
}

file_content_is() {
    local got
    got="$(cat "$1")"
    [ "$got" = "$2" ]
}

printf 'mutate_vendored_schemas_guard.sh - PRREV-002\n\n'

# --- M0: discrimination case -------------------------------------------------
d="$(stage M0)"
run_guard "M0-baseline-unmutated" 0 "$d" || true

# --- M1: a vendored schema is edited ----------------------------------------
d="$(stage M1)"
printf '\n' >> "$d/$SARIF"
if engaged M1 test -s "$d/$SARIF"; then
    run_guard "M1-schema-byte-appended" 1 "$d" || true
fi

# --- M2: the manifest stops covering a schema -------------------------------
d="$(stage M2)"
grep -v 'sarif-2.1.0.json' "$d/schemas/MANIFEST.sha256" > "$d/m.tmp"
mv "$d/m.tmp" "$d/schemas/MANIFEST.sha256"
if engaged M2 absent_from sarif-2.1.0.json "$d/schemas/MANIFEST.sha256"; then
    run_guard "M2-manifest-entry-dropped" 1 "$d" || true
fi

# --- M3: the two hash records drift apart -----------------------------------
d="$(stage M3)"
sed -i 's/c3b4bb2d6093897483348925aaa73af03b3e3f4bd4ca38cef26dcb4212a2682e/0000000000000000000000000000000000000000000000000000000000000000/' "$d/schemas/sources.json"
if engaged M3 grep -q '0000000000000000000000000000000000000000000000000000000000000000' "$d/schemas/sources.json"; then
    run_guard "M3-sources-hash-drift" 1 "$d" || true
fi

# --- M4: a $ref is pointed at the network -----------------------------------
d="$(stage M4)"
jq '.["$defs"]["typeUri"] = {"$ref": "https://json-schema.org/draft/2020-12/schema"}' \
   "$d/$INTOTO" > "$d/t.json"
mv "$d/t.json" "$d/$INTOTO"
rehash "$d"
if engaged M4 grep -q 'json-schema.org/draft/2020-12/schema' "$d/$INTOTO"; then
    run_guard "M4-remote-ref-introduced" 1 "$d" || true
fi

# --- M5: a fixture is moved to the wrong side of the line -------------------
d="$(stage M5)"
mv "$d/tests/fixtures/schemas/intoto/reject/digest-uppercase-hex.json" \
   "$d/tests/fixtures/schemas/intoto/accept/"
if engaged M5 test -f "$d/tests/fixtures/schemas/intoto/accept/digest-uppercase-hex.json"; then
    run_guard "M5-reject-fixture-relabelled" 1 "$d" || true
fi

# --- M6/M7: the table loses a discrimination side ---------------------------
d="$(stage M6)"
rm -f "$d"/tests/fixtures/schemas/sarif/accept/*
if engaged M6 dir_has_no_files "$d/tests/fixtures/schemas/sarif/accept"; then
    run_guard "M6-accept-fixtures-deleted" 1 "$d" || true
fi

d="$(stage M7)"
rm -f "$d"/tests/fixtures/schemas/sarif/reject/*
if engaged M7 dir_has_no_files "$d/tests/fixtures/schemas/sarif/reject"; then
    run_guard "M7-reject-fixtures-deleted" 1 "$d" || true
fi

# --- M8: the schema is replaced by one that accepts anything ----------------
d="$(stage M8)"
printf 'true\n' > "$d/$INTOTO"
rehash "$d"
if engaged M8 file_content_is "$d/$INTOTO" true; then
    run_guard "M8-schema-accepts-everything" 1 "$d" || true
fi

printf '\nmutants: %s as-required, %s survived, %s never engaged\n' "$killed" "$survived" "$notengaged"
if [ "$survived" -ne 0 ] || [ "$notengaged" -ne 0 ]; then
    printf 'FAIL: check_vendored_schemas.sh is not proved to turn RED for every defect it names.\n'
    exit 1
fi
printf 'PASS: baseline GREEN, every mutant RED, every mutation observed to engage.\n'
exit 0
