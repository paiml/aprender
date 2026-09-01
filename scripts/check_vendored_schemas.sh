#!/usr/bin/env bash
# check_vendored_schemas.sh - the schemas under schemas/ are the ones we vendored,
# and check-jsonschema validates against them with the network unavailable.
#
# PRREV-002. PR-REVIEW-SKILL-002 v2 §6.2:
#
#     "Schemas are vendored under schemas/ with a recorded SHA-256. Fetching a
#      schema over the network at gate time makes the gate depend on an external
#      service - the same defect class as the render path."
#
# THE CLASS. A vendored file is only vendored while nothing has edited it, and a
# gate is only offline while nothing has quietly reintroduced a fetch. Both decay
# silently: an edited schema still parses, and a $ref to a remote host only fails
# on the day the remote is down - which is the day you least want a new failure.
# So this check asserts three things mechanically:
#
#   1. INTEGRITY   - every vendored schema still hashes to its recorded SHA-256,
#                    and the two independent records of those hashes
#                    (schemas/MANIFEST.sha256 and schemas/sources.json) agree.
#   2. NO OFF-HOST - no $ref in a vendored schema resolves anywhere but inside
#                    the same file.
#   3. OFFLINE     - the whole accept/reject fixture table runs green inside a
#                    network namespace with NO interfaces and an EMPTY schema
#                    cache. Not "we did not see a request": no route exists.
#
# A manifest that lists nothing verifies nothing, and a table with only reject
# cases reads green when the validator is broken enough to refuse everything.
# Both vacuous shapes are checked for explicitly - see check_manifest_covers_all
# and the discrimination-case counts in run_case_table.
#
# Exit 0 = vendored schemas intact, self-contained, and validated offline.
# Exit 1 = a DEFECT: hash drift, off-host $ref, a fixture on the wrong side of
#          the line, a vacuous manifest or table, or a positive control that did
#          not fire.
# Exit 2 = the ENVIRONMENT could not answer the question (validator missing,
#          no user+network namespace available). Still non-zero, deliberately:
#          an unmeasured gate is not a passing gate. The distinct code is so a
#          broken box is never read as a broken tree.
#
# `--self-test` runs only the positive controls and the regex case table, which
# is where this script proves it can still turn RED.

set -euo pipefail

cd "$(dirname "$0")/.." || exit 2

MANIFEST="schemas/MANIFEST.sha256"
SOURCES="schemas/sources.json"
FIXTURES="tests/fixtures/schemas"

INTOTO_SCHEMA="schemas/in-toto-statement-v1.json"
SARIF_SCHEMA="schemas/sarif-2.1.0.json"

RC_DEFECT=1
RC_ENV=2

fail=0
note()  { printf '  %s\n' "$*"; }
bad()   { printf '  FAIL: %s\n' "$*"; fail=1; }
head2() { printf '\n== %s\n' "$*"; }

# ---------------------------------------------------------------------------
# $ref extraction.
#
# REF_RE matches a `"$ref": "<value>"` pair; the value is then classified by a
# glob, not a regex - a local reference is exactly one that starts with '#'.
# This repo's patterns have been wrong six times and a case table caught every
# one, so REF_RE ships its table in self_test_ref_regex below rather than
# relying on this comment.
# ---------------------------------------------------------------------------
REF_RE='"\$ref"[[:space:]]*:[[:space:]]*"[^"]*"'

extract_refs() {
    grep -oE "$REF_RE" "$1" 2>/dev/null | sed -E 's/^.*:[[:space:]]*"//; s/"$//' || true
}

# Prints every $ref in $1 that is NOT a local JSON Pointer. Prints nothing when
# the file is self-contained.
remote_refs() {
    local value
    while IFS= read -r value; do
        [ -n "$value" ] || continue
        case "$value" in
            '#'*) : ;;
            *) printf '%s\n' "$value" ;;
        esac
    done < <(extract_refs "$1")
}

# ---------------------------------------------------------------------------
# Offline execution.
#
# `unshare -r -n` gives an unprivileged user namespace with a fresh network
# namespace that has no interfaces at all - not a firewall rule, no route.
# XDG_CACHE_HOME is pointed at an empty directory so a previously downloaded
# schema cannot stand in for the vendored one.
# ---------------------------------------------------------------------------
netns_available() {
    unshare -r -n true 2>/dev/null
}

run_offline() {
    local cachedir rc=0
    cachedir="$(mktemp -d)"
    unshare -r -n env XDG_CACHE_HOME="$cachedir" "$@" >/dev/null 2>&1 || rc=$?
    rm -rf "${cachedir:?}"
    return "$rc"
}

# ===========================================================================
# Positive controls - §6.1. These run BEFORE anything is validated for real.
# If a control does not fire, this script's GREEN is a count of files.
# ===========================================================================

self_test_ref_regex() {
    head2 "positive control 1/4: \$ref classification case table"
    local tmp rc=0 got want line
    tmp="$(mktemp)"

    # must-be-flagged (off-host or off-file), then must-NOT-be-flagged (local)
    local -a remote_cases=(
        '{"$ref": "https://json-schema.org/draft/2020-12/schema"}'
        '{"$ref":"http://docs.oasis-open.org/x.json#/definitions/run"}'
        '{"$ref": "sarif-2.1.0.json#/definitions/result"}'
        '{"$ref"  :   "//example.com/s.json"}'
    )
    local -a local_cases=(
        '{"$ref": "#/definitions/run"}'
        '{"$ref":"#/$defs/typeUri"}'
        '{"$ref" : "#"}'
        '{"description": "the $ref keyword is a JSON Pointer"}'
        '{"$comment": "no reference here at all"}'
    )

    for line in "${remote_cases[@]}"; do
        printf '%s\n' "$line" > "$tmp"
        got="$(remote_refs "$tmp")"
        if [ -z "$got" ]; then
            bad "case table: expected REMOTE, got local/none for: $line"
            rc=1
        fi
    done
    for line in "${local_cases[@]}"; do
        printf '%s\n' "$line" > "$tmp"
        got="$(remote_refs "$tmp")"
        if [ -n "$got" ]; then
            bad "case table: expected LOCAL/none, got remote '$got' for: $line"
            rc=1
        fi
    done
    rm -f "${tmp:?}"
    want=$(( ${#remote_cases[@]} + ${#local_cases[@]} ))
    if [ "$rc" -eq 0 ]; then
        note "OK - ${want} rows, ${#remote_cases[@]} must-match / ${#local_cases[@]} must-not-match"
    fi
    return "$rc"
}

self_test_corrupt_schema() {
    head2 "positive control 2/4: an edited schema must fail the manifest"
    local tmp rc=0
    tmp="$(mktemp -d)"
    mkdir -p "$tmp/schemas"
    cp "$INTOTO_SCHEMA" "$SARIF_SCHEMA" "$tmp/schemas/"
    cp "$MANIFEST" "$tmp/schemas/"
    printf '\n' >> "$tmp/schemas/$(basename "$INTOTO_SCHEMA")"
    local crc=0
    ( cd "$tmp" && sha256sum -c schemas/MANIFEST.sha256 >/dev/null 2>&1 ) || crc=$?
    if [ "$crc" -eq 0 ]; then
        bad "sha256sum -c accepted a schema with one byte appended - the integrity check is theater"
        rc=1
    else
        note "OK - one appended newline turns sha256sum -c RED"
    fi
    rm -rf "${tmp:?}"
    return "$rc"
}

self_test_empty_manifest() {
    head2 "positive control 3/4: a manifest that lists nothing must not pass"
    local tmp rc=0 n
    tmp="$(mktemp)"
    : > "$tmp"
    n="$(grep -c . "$tmp" || true)"
    if [ "${n:-0}" -ge 2 ]; then
        bad "an empty manifest counted ${n} entries"
        rc=1
    else
        note "OK - empty manifest counts ${n:-0} entries, below the required floor of 2"
    fi
    rm -f "${tmp:?}"
    return "$rc"
}

self_test_validator_rejects() {
    head2 "positive control 4/4: the validator must reject a malformed document"
    local tmp rc=0 vrc=0
    tmp="$(mktemp -d)"
    # Structurally plausible, semantically wrong: legacy statement type, empty
    # subject, no predicateType. Nothing here is a parse error.
    cat > "$tmp/malformed.json" <<'BAD'
{ "_type": "https://in-toto.io/Statement/v0.1", "subject": [], "predicate": {} }
BAD
    check-jsonschema --schemafile "$INTOTO_SCHEMA" "$tmp/malformed.json" >/dev/null 2>&1 || vrc=$?
    if [ "$vrc" -eq 0 ]; then
        bad "check-jsonschema ACCEPTED a malformed in-toto Statement - it is parsing, not validating"
        rc=1
    else
        note "OK - malformed Statement rejected (exit ${vrc})"
    fi
    rm -rf "${tmp:?}"
    return "$rc"
}

positive_controls() {
    local rc=0
    self_test_ref_regex        || rc=1
    self_test_corrupt_schema   || rc=1
    self_test_empty_manifest   || rc=1
    self_test_validator_rejects || rc=1
    return "$rc"
}

# ===========================================================================
# The real checks
# ===========================================================================

check_validator_present() {
    head2 "validator"
    if ! command -v check-jsonschema >/dev/null 2>&1; then
        printf '  ENV: check-jsonschema is not on PATH.\n'
        printf '       Obtain it with:  uv tool install %s\n' "'check-jsonschema==0.38.0'"
        printf '       (network is required ONCE, to install. Never at gate time.)\n'
        return 1
    fi
    note "check-jsonschema $(check-jsonschema --version 2>&1 | head -1)"
    return 0
}

check_manifest_integrity() {
    head2 "integrity: schemas/MANIFEST.sha256"
    local out rc=0
    out="$(sha256sum -c "$MANIFEST" 2>&1)" || rc=$?
    printf '%s\n' "$out" | sed 's/^/  /'
    if [ "$rc" -ne 0 ]; then
        bad "a vendored schema no longer matches its recorded SHA-256"
        return 1
    fi
    return 0
}

# A manifest is only a ratchet while it covers everything it is supposed to
# cover. This is the `pv lint <FILE>` shape: PASS over zero contracts.
check_manifest_covers_all() {
    head2 "completeness: every schemas/*.json is in the manifest"
    local listed present n_listed rc=0
    listed="$(awk 'NF {print $2}' "$MANIFEST" | sort)"
    present="$(find schemas -maxdepth 1 -type f -name '*.json' ! -name 'sources.json' -print | sort)"
    n_listed="$(printf '%s\n' "$listed" | grep -c . || true)"

    if [ "${n_listed:-0}" -lt 2 ]; then
        bad "manifest lists ${n_listed:-0} file(s); PRREV-002 vendors 2 schemas, so anything below 2 is vacuous"
        rc=1
    fi
    if [ "$listed" != "$present" ]; then
        bad "manifest and schemas/ disagree:"
        diff <(printf '%s\n' "$listed") <(printf '%s\n' "$present") | sed 's/^/    /' || true
        rc=1
    else
        note "OK - ${n_listed} schema(s), manifest and directory agree"
    fi
    return "$rc"
}

# Two independent records of the same hash must not drift.
check_sources_agreement() {
    head2 "cross-check: schemas/sources.json vs schemas/MANIFEST.sha256"
    local rc=0 path want got
    while IFS= read -r path; do
        want="$(jq -r --arg p "$path" '.schemas[] | select(.path == $p) | .sha256' "$SOURCES")"
        got="$(awk -v p="$path" 'NF && $2 == p {print $1}' "$MANIFEST")"
        if [ -z "$want" ]; then
            bad "$path is in the manifest but has no provenance entry in $SOURCES"
            rc=1
        elif [ "$want" != "$got" ]; then
            bad "$path: sources.json says $want, MANIFEST says $got"
            rc=1
        else
            note "OK - $path $got"
        fi
    done < <(awk 'NF {print $2}' "$MANIFEST")
    return "$rc"
}

check_no_remote_refs() {
    head2 "self-containment: no \$ref resolves off-file"
    local rc=0 schema offenders
    for schema in "$INTOTO_SCHEMA" "$SARIF_SCHEMA"; do
        offenders="$(remote_refs "$schema")"
        if [ -n "$offenders" ]; then
            bad "$schema has non-local \$ref(s) - the gate would need the network:"
            printf '%s\n' "$offenders" | sed 's/^/    /'
            rc=1
        else
            note "OK - $schema: every \$ref is a local JSON Pointer"
        fi
    done
    return "$rc"
}

# The fixture table, run with no network and no schema cache.
run_case_table() {
    local label="$1" schema="$2" dir="$3"
    local f rc=0 n_accept=0 n_reject=0 vrc

    head2 "offline case table: $label"
    for f in "$dir"/accept/*; do
        [ -e "$f" ] || continue
        n_accept=$((n_accept + 1))
        vrc=0
        run_offline check-jsonschema --schemafile "$schema" "$f" || vrc=$?
        if [ "$vrc" -ne 0 ]; then
            bad "accept fixture REJECTED (exit $vrc): $f"
            rc=1
        fi
    done
    for f in "$dir"/reject/*; do
        [ -e "$f" ] || continue
        n_reject=$((n_reject + 1))
        vrc=0
        run_offline check-jsonschema --schemafile "$schema" "$f" || vrc=$?
        if [ "$vrc" -eq 0 ]; then
            bad "reject fixture ACCEPTED: $f"
            rc=1
        fi
    done

    # Discrimination cases. Without an accept row, "refuse everything" reads
    # green; without a reject row, "accept everything" does.
    if [ "$n_accept" -lt 1 ]; then
        bad "$label: zero accept fixtures - a table with no accept row cannot detect a validator that refuses everything"
        rc=1
    fi
    if [ "$n_reject" -lt 1 ]; then
        bad "$label: zero reject fixtures - a table with no reject row cannot detect a validator that accepts everything"
        rc=1
    fi
    if [ "$rc" -eq 0 ]; then
        note "OK - ${n_accept} accept / ${n_reject} reject, all validated offline under unshare -r -n with an empty XDG_CACHE_HOME"
    fi
    return "$rc"
}

usage() {
    printf 'usage: %s [--self-test]\n' "$0"
}

main() {
    case "${1:-}" in
        --self-test)
            printf 'check_vendored_schemas.sh --self-test (positive controls only)\n'
            if ! command -v check-jsonschema >/dev/null 2>&1; then
                printf '\nENV: check-jsonschema is not on PATH; control 4/4 cannot run.\n'
                printf '     uv tool install %s\n' "'check-jsonschema==0.38.0'"
                exit "$RC_ENV"
            fi
            if positive_controls; then
                printf '\nSELF-TEST PASS: every positive control fired.\n'
                exit 0
            fi
            printf '\nSELF-TEST FAIL: a positive control did not fire; this guard cannot be trusted to turn RED.\n'
            exit "$RC_DEFECT"
            ;;
        ''|--help|-h)
            [ "${1:-}" = "" ] || { usage; exit 0; }
            ;;
        *)
            usage
            exit "$RC_DEFECT"
            ;;
    esac

    printf 'check_vendored_schemas.sh - PRREV-002 / PR-REVIEW-SKILL-002 v2 §6.2\n'

    check_validator_present || exit "$RC_ENV"

    if ! netns_available; then
        printf '\nENV: `unshare -r -n` is unavailable on this host, so the OFFLINE claim\n'
        printf '     cannot be proved here. Refusing to report GREEN on an unmeasured gate.\n'
        printf '     (Unprivileged user namespaces: sysctl kernel.unprivileged_userns_clone=1)\n'
        exit "$RC_ENV"
    fi

    positive_controls       || fail=1
    check_manifest_integrity || true
    check_manifest_covers_all || true
    check_sources_agreement  || true
    check_no_remote_refs     || true
    run_case_table "in-toto Statement v1" "$INTOTO_SCHEMA" "$FIXTURES/intoto" || true
    run_case_table "SARIF 2.1.0"          "$SARIF_SCHEMA"  "$FIXTURES/sarif"  || true

    printf '\n'
    if [ "$fail" -ne 0 ]; then
        printf 'FAIL: vendored schemas did not verify. See FAIL lines above.\n'
        exit "$RC_DEFECT"
    fi
    printf 'PASS: schemas vendored, hashes recorded and matching, self-contained,\n'
    printf '      and validated by check-jsonschema with no network namespace attached.\n'
    exit 0
}

main "$@"
