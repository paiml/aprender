#!/usr/bin/env bash
#
# check_gate5_stage.sh — a release gate must name the stage it is valid at.
#
# WHY THIS EXISTS (#2543)
# -----------------------
# Gate 5 of the pre-release skill runs `cargo package -p apr-cli`. `cargo
# package` re-resolves every dependency against crates.io, so a workspace
# sibling resolves to its ALREADY-PUBLISHED copy, not to the tree. Before the
# version bump the published copy IS the workspace version, so the verify build
# compiles today's apr-cli against yesterday's aprender/entrenar_lora/realizar
# and reports dozens of E0432/E0433 SYMBOL-not-found errors:
#
#     error[E0432]: unresolved import `aprender::format::q4k_output_size_estimate`
#     error[E0432]: unresolved import `entrenar_lora::plan_with_rank`
#     error[E0433]: could not find `CancelToken` in `generate`
#
# Those symbols are real post-0.63.0 additions. Nothing is broken — the gate is
# simply UNSATISFIABLE at that stage. A release engineer meeting it cold reads
# the errors as a release blocker and aborts the cut. That has happened.
#
# The constraint is ORDERING, so the remedy is to make the ordering explicit and
# machine-checked rather than to code around the gate.
#
# WHAT IT CHECKS (the ratchet)
# ----------------------------
# For every `cargo package -p <crate>` named in the pre-release skill, EITHER
# that crate has zero workspace-sibling (non-dev) dependencies — in which case
# the gate is meaningful at any stage — OR the skill carries an explicit,
# machine-readable declaration of the stage it is valid at:
#
#     STAGE-PRECONDITION: cargo package -p <crate> requires stage <STAGE>
#
# with <STAGE> one of MEANINGFUL | PRE_BUMP | POST_BUMP_PRE_CASCADE |
# CASCADE_READY. The declaration must also be TRUE against the dependency graph:
# claiming MEANINGFUL for a crate that has siblings, or a non-MEANINGFUL stage
# for a crate that has none, is a failure. An assertion that cannot exclude an
# outcome is not an assertion.
#
# Text + `cargo metadata --no-deps --offline` only. No build, no network.
#
# THE CLASSIFIER (the release engineer's tool)
# --------------------------------------------
#   bash scripts/check_gate5_stage.sh --explain apr-cli
#
# compares the workspace version V against each sibling's crates.io max version
# and prints one of:
#
#   MEANINGFUL             no workspace-sibling deps; Gate 5 is valid at any stage
#   PRE_BUMP               V is already published, so siblings resolve to the
#                          STALE published API — symbol-not-found is EXPECTED
#   POST_BUMP_PRE_CASCADE  V is ahead of the registry; unsatisfiable until the
#                          siblings publish at V
#   CASCADE_READY          every sibling is live at V; Gate 5 is now meaningful
#
# `--explain` needs the network (or a fixture, see the test hooks below). The
# default mode and `--self-test` never touch it.
#
#   bash scripts/check_gate5_stage.sh              # the ratchet
#   bash scripts/check_gate5_stage.sh --self-test  # case table
#   bash scripts/check_gate5_stage.sh --explain <crate>
#
# TEST HOOKS: GATE5_METADATA_JSON and GATE5_REGISTRY_FIXTURE exist so
# --self-test can drive the same code paths hermetically. CI sets neither.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKILL_FILE="${GATE5_SKILL_FILE:-${REPO_ROOT}/.claude/skills/pre-release/SKILL.md}"
STAGES="MEANINGFUL PRE_BUMP POST_BUMP_PRE_CASCADE CASCADE_READY"
# The exact cargo diagnostic for the post-bump state. The apostrophe is built
# from its octal code rather than written literally: bashrs cannot parse an
# apostrophe inside a double-quoted string (SC1078 false positive, the class
# scripts/check_shell_lint_ratchet.sh documents), and scripts/*.sh is gated on
# a shrink-only bashrs error count. Reproducing the string EXACTLY matters --
# #2543 predicted the wrong one ("no matching package named"), which is the
# error for a crate that was never published at all.
APOS=$(printf '\047')
CARGO_MISMATCH_MSG="failed to select a version ... candidate versions found which didn${APOS}t match"

die() { printf '%s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# `name<TAB>version<TAB>sibling_count<TAB>sibling,names` for every workspace
# member. Non-dev (normal + build) workspace-sibling dependencies only: those
# are what `cargo package`'s verify build must resolve from the registry.
# ---------------------------------------------------------------------------
sibling_table() {
    local json="${GATE5_METADATA_JSON:-}"
    if [ -z "$json" ]; then
        json="${GUARD_TMP}/metadata.json"
        # No --offline: the sibling guards in this same CI job (check_contract_
        # enforcement.sh) run exactly this form and are proven to work there.
        # `--no-deps` resolves nothing, so it does not reach the network anyway.
        ( cd "$REPO_ROOT" && cargo metadata --no-deps --format-version 1 ) \
            > "$json" 2>"${GUARD_TMP}/metadata.err" \
            || die "check_gate5_stage: cargo metadata failed; see ${GUARD_TMP}/metadata.err"
    fi
    python3 - "$json" <<'PY'
import json, sys
meta = json.load(open(sys.argv[1]))
members = {p["name"] for p in meta["packages"]}
for p in sorted(meta["packages"], key=lambda p: p["name"]):
    sib = sorted({
        d["name"] for d in p.get("dependencies", [])
        if d.get("kind") in (None, "build") and d["name"] in members
    })
    print("\t".join([p["name"], p["version"], str(len(sib)), ",".join(sib)]))
PY
}

field() { printf '%s\n' "$1" | cut -f"$2"; }

row_for() {  # row_for <crate> <table>
    printf '%s\n' "$2" | awk -F'\t' -v n="$1" '$1 == n { print; exit }'
}

# Crates named by a Gate-5-class `cargo package -p <crate>` command, deduped.
# STAGE-PRECONDITION lines quote the command too; they are the declaration, not
# the gate, so they are excluded here.
# NOTE the ORDER: the STAGE-PRECONDITION lines are dropped by LINE first. Doing
# it after `grep -o` cannot work — the extracted fragment is `cargo package -p X`
# and never begins with the marker, so the filter would be a silent no-op and a
# declaration alone would masquerade as the gate it is declaring about.
gate5_command_lines() {
    grep -nE 'cargo package -p [A-Za-z0-9_-]+' "$1" | grep -v 'STAGE-PRECONDITION:'
}
gate5_crates() {
    gate5_command_lines "$1" \
        | grep -oE 'cargo package -p [A-Za-z0-9_-]+' \
        | awk '{print $NF}' | LC_ALL=C sort -u
}

# `crate<TAB>stage` for every STAGE-PRECONDITION declaration in the file.
declarations() {
    grep -oE 'STAGE-PRECONDITION: cargo package -p [A-Za-z0-9_-]+ requires stage [A-Z_]+' "$1" \
        | awk '{print $5 "\t" $NF}' | LC_ALL=C sort -u
}
declared_stage() {  # declared_stage <crate> <decls>
    printf '%s\n' "$2" | awk -F'\t' -v n="$1" '$1 == n { print $2; exit }'
}

stage_known() {
    case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac
}

# ---------------------------------------------------------------------------
# crates.io max version of $1, or the empty string when unpublished/unreachable.
# ---------------------------------------------------------------------------
registry_max_version() {
    if [ -n "${GATE5_REGISTRY_FIXTURE:-}" ]; then
        awk -F'\t' -v n="$1" '$1 == n { print $2; exit }' "$GATE5_REGISTRY_FIXTURE"
        return 0
    fi
    curl -s --max-time 20 -H 'User-Agent: aprender check_gate5_stage (release tooling)' \
        "https://crates.io/api/v1/crates/$1" 2>/dev/null \
        | python3 -c 'import json,sys
try:
    d = json.load(sys.stdin)
    v = d.get("crate", {}).get("max_version", "")
except Exception:
    v = ""
print(v)' 2>/dev/null
}

# classify <crate> <table>  -> prints the stage keyword, or the empty string if
# the crate is not a workspace member.
classify() {
    local crate="$1" table="$2" row ver count sibs own s m all_at_v
    row="$(row_for "$crate" "$table")"
    [ -n "$row" ] || return 0
    ver="$(field "$row" 2)"; count="$(field "$row" 3)"; sibs="$(field "$row" 4)"
    if [ "$count" -eq 0 ]; then printf 'MEANINGFUL\n'; return 0; fi

    own="$(registry_max_version "$crate")"
    # V is already on the registry: the siblings this build resolves are the
    # published copies of the SAME version number, i.e. the stale API.
    if [ "$own" = "$ver" ]; then printf 'PRE_BUMP\n'; return 0; fi

    all_at_v=1
    for s in ${sibs//,/ }; do
        m="$(registry_max_version "$s")"
        [ "$m" = "$ver" ] || { all_at_v=0; break; }
    done
    if [ "$all_at_v" -eq 1 ]; then printf 'CASCADE_READY\n'; else printf 'POST_BUMP_PRE_CASCADE\n'; fi
}

# ---------------------------------------------------------------------------
# The ratchet. Returns 1 on any violation. Factored out so --self-test drives
# exactly the code CI runs.
# ---------------------------------------------------------------------------
check_predicate() {  # check_predicate <skill_file> <table>
    local skill="$1" table="$2" decls crates crate row count stage n=0 bad=0

    [ -f "$skill" ] || { printf 'FAIL: skill file not found: %s\n' "$skill"; return 1; }

    crates="$(gate5_crates "$skill")"
    n="$(printf '%s\n' "$crates" | grep -c . || true)"
    # n=0 is a FAIL mode, not a pass: Gate 5's existence is the thing guarded.
    # "0 cases OK" is how a gate goes dark.
    if [ "$n" -eq 0 ]; then
        printf 'FAIL (vacuity): %s names no `cargo package -p <crate>` command.\n' "$skill"
        printf '       Gate 5 has been renamed, reworded or deleted. This guard measured NOTHING.\n'
        return 1
    fi

    decls="$(declarations "$skill")"

    while IFS= read -r crate; do
        [ -n "$crate" ] || continue
        row="$(row_for "$crate" "$table")"
        if [ -z "$row" ]; then
            printf 'FAIL: Gate 5 names `cargo package -p %s`, which is not a workspace member.\n' "$crate"
            bad=1; continue
        fi
        count="$(field "$row" 3)"
        stage="$(declared_stage "$crate" "$decls")"

        if [ "$count" -eq 0 ]; then
            if [ -n "$stage" ] && [ "$stage" != "MEANINGFUL" ]; then
                printf 'FAIL: %s has 0 workspace-sibling deps but is declared %s.\n' "$crate" "$stage"
                printf '      Gate 5 is stage-independent for it; declare MEANINGFUL or drop the line.\n'
                bad=1
            fi
            continue
        fi

        if [ -z "$stage" ]; then
            printf 'FAIL: Gate 5 names %s (%s workspace-sibling deps) with no precondition stated.\n' \
                "$crate" "$count"
            printf '      `cargo package` resolves those %s siblings from crates.io, so the gate is\n' "$count"
            printf '      unsatisfiable before they are published at the workspace version (#2543).\n'
            printf '      Add to %s:\n' "$skill"
            printf '        STAGE-PRECONDITION: cargo package -p %s requires stage CASCADE_READY\n' "$crate"
            bad=1; continue
        fi
        if ! stage_known "$stage"; then
            printf 'FAIL: %s declares unknown stage %s (expected one of: %s).\n' "$crate" "$stage" "$STAGES"
            bad=1; continue
        fi
        if [ "$stage" = "MEANINGFUL" ]; then
            printf 'FAIL: %s is declared MEANINGFUL but has %s workspace-sibling deps.\n' "$crate" "$count"
            printf '      MEANINGFUL means "valid at any stage" and is false here.\n'
            bad=1; continue
        fi
        printf 'ok   %s: %s sibling dep(s), declared %s\n' "$crate" "$count" "$stage"
    done <<< "$crates"

    # An orphan declaration is drift in the other direction: it names a crate no
    # Gate-5 command mentions, or a stage keyword the classifier cannot produce.
    while IFS=$'\t' read -r crate stage; do
        [ -n "$crate" ] || continue
        if ! stage_known "$stage"; then
            printf 'FAIL: STAGE-PRECONDITION for %s declares unknown stage %s.\n' "$crate" "$stage"
            bad=1
        fi
        if [ -z "$(row_for "$crate" "$table")" ]; then
            printf 'FAIL: STAGE-PRECONDITION names %s, which is not a workspace member.\n' "$crate"
            bad=1
        fi
    done <<< "$decls"

    return "$bad"
}

GUARD_TMP="$(mktemp -d)" || exit 1
trap 'rm -rf "${GUARD_TMP:?}"' EXIT

# ---------------------------------------------------------------------------
case "${1:-}" in
--self-test)
    fails=0
    FX="$GUARD_TMP/fx"; mkdir -p "$FX"

    # Two workspaces, identical shape, different version. V=1.0.0 is the
    # pre-bump world; V=2.0.0 is post-bump.
    write_meta() {  # write_meta <path> <version>
        cat > "$1" <<META
{"packages":[
 {"name":"leafcrate","version":"$2","dependencies":[]},
 {"name":"bigcrate","version":"$2","dependencies":[
   {"name":"leafcrate","kind":null},
   {"name":"serde","kind":null},
   {"name":"leafcrate","kind":"dev"}]}
]}
META
    }
    write_meta "$FX/meta-v1.json" "1.0.0"
    write_meta "$FX/meta-v2.json" "2.0.0"
    TABLE1="$(GATE5_METADATA_JSON="$FX/meta-v1.json" sibling_table)"
    TABLE2="$(GATE5_METADATA_JSON="$FX/meta-v2.json" sibling_table)"

    row() {  # row <label> <expected_rc> <skill_text> [table]
        local label="$1" want="$2" text="$3" tbl="${4:-$TABLE1}" got out
        printf '%s\n' "$text" > "$FX/skill.md"
        out="$(check_predicate "$FX/skill.md" "$tbl" 2>&1)"; got=$?
        if [ "$got" -eq "$want" ]; then
            printf 'ok    %s (rc=%s)\n' "$label" "$got"
        else
            printf 'FAIL  %s: rc=%s, expected %s\n%s\n' "$label" "$got" "$want" "$out"
            fails=1
        fi
    }

    # Dev-deps must not count: bigcrate's dev edge on leafcrate is deliberate in
    # the fixture. If it counted, leafcrate would show 0 either way and row 5
    # would still pass — the count is asserted directly instead.
    lc="$(row_for leafcrate "$TABLE1")"; bc="$(row_for bigcrate "$TABLE1")"
    if [ "$(field "$lc" 3)" = "0" ] && [ "$(field "$bc" 3)" = "1" ]; then
        printf 'ok    row 0 sibling counts: leafcrate=0, bigcrate=1 (dev edge and serde excluded)\n'
    else
        printf 'FAIL  row 0 sibling counts: leafcrate=%s bigcrate=%s, expected 0 and 1\n' \
            "$(field "$lc" 3)" "$(field "$bc" 3)"; fails=1
    fi

    row "row 1 zero-sibling crate needs no declaration" 0 \
        'Gate 5: run `cargo package -p leafcrate --allow-dirty`.'
    row "row 2 sibling crate with NO declaration is REJECTED" 1 \
        'Gate 5: run `cargo package -p bigcrate --allow-dirty`.'
    row "row 3 sibling crate WITH a declaration passes (doc arm reads the text)" 0 \
        'Gate 5: run `cargo package -p bigcrate --allow-dirty`.
STAGE-PRECONDITION: cargo package -p bigcrate requires stage CASCADE_READY'
    row "row 4 declaring MEANINGFUL for a sibling crate is REJECTED" 1 \
        'Gate 5: run `cargo package -p bigcrate --allow-dirty`.
STAGE-PRECONDITION: cargo package -p bigcrate requires stage MEANINGFUL'
    row "row 5 declaring a non-MEANINGFUL stage for a leaf crate is REJECTED" 1 \
        'Gate 5: run `cargo package -p leafcrate --allow-dirty`.
STAGE-PRECONDITION: cargo package -p leafcrate requires stage PRE_BUMP'
    row "row 6 n=0 (no Gate-5 command at all) is a FAIL, not a silent pass" 1 \
        'Gate 5 was renamed and no longer runs cargo package.'
    # A declaration must not be mistaken for the command it declares about. The
    # first cut of this script filtered STAGE-PRECONDITION AFTER `grep -o`, where
    # the filter is a no-op: a file holding only the declaration read as a
    # satisfied gate. This row is that mutation, frozen.
    row "row 6b a STAGE-PRECONDITION line is NOT itself a Gate-5 command (n=0)" 1 \
        'Gate 5 no longer runs it.
STAGE-PRECONDITION: cargo package -p bigcrate requires stage CASCADE_READY'
    row "row 7 a Gate-5 command naming a non-member is REJECTED" 1 \
        'Gate 5: run `cargo package -p nosuchcrate --allow-dirty`.'
    row "row 8 an unknown stage keyword is REJECTED" 1 \
        'Gate 5: run `cargo package -p bigcrate --allow-dirty`.
STAGE-PRECONDITION: cargo package -p bigcrate requires stage LATER'

    # ---- classifier: it must be able to produce more than one verdict. -----
    # A classifier that returned a constant would satisfy every row above, so
    # this arm asserts DISTINCT verdicts from distinct registry states.
    printf 'leafcrate\t1.0.0\nbigcrate\t1.0.0\n' > "$FX/reg-prebump.tsv"
    printf 'leafcrate\t1.0.0\nbigcrate\t1.0.0\n' > "$FX/reg-postbump.tsv"
    printf 'leafcrate\t2.0.0\nbigcrate\t1.0.0\n' > "$FX/reg-cascade.tsv"

    cls() { GATE5_REGISTRY_FIXTURE="$1" classify "$2" "$3"; }
    declare -a verdicts=()
    expect_cls() {  # expect_cls <label> <want> <fixture> <crate> <table>
        local got; got="$(cls "$3" "$4" "$5")"
        verdicts+=("$got")
        if [ "$got" = "$2" ]; then printf 'ok    %s -> %s\n' "$1" "$got"
        else printf 'FAIL  %s -> %s, expected %s\n' "$1" "$got" "$2"; fails=1; fi
    }
    expect_cls "row 9  leaf crate, any registry state" MEANINGFUL "$FX/reg-prebump.tsv"  leafcrate "$TABLE1"
    expect_cls "row 10 V=1.0.0 already published (pre-bump)" PRE_BUMP "$FX/reg-prebump.tsv" bigcrate "$TABLE1"
    expect_cls "row 11 V=2.0.0, siblings still at 1.0.0" POST_BUMP_PRE_CASCADE "$FX/reg-postbump.tsv" bigcrate "$TABLE2"
    expect_cls "row 12 V=2.0.0, every sibling live at 2.0.0" CASCADE_READY "$FX/reg-cascade.tsv" bigcrate "$TABLE2"

    distinct=$(printf '%s\n' "${verdicts[@]}" | LC_ALL=C sort -u | grep -c .)
    if [ "$distinct" -ge 3 ]; then
        printf 'ok    row 13 non-vacuity: classifier produced %s distinct verdicts\n' "$distinct"
    else
        printf 'FAIL  row 13 non-vacuity: only %s distinct verdict(s) — a constant-return classifier\n' "$distinct"
        fails=1
    fi

    [ "$fails" -eq 0 ] || { printf '\nSELF-TEST FAILED\n'; exit 1; }
    printf '\nSELF-TEST PASSED (15 rows: 10 predicate, 4 classifier, 1 non-vacuity control)\n'
    exit 0
    ;;
--explain)
    crate="${2:-}"
    [ -n "$crate" ] || die "usage: check_gate5_stage.sh --explain <crate>"
    TABLE="$(sibling_table)" || exit 1
    row="$(row_for "$crate" "$TABLE")"
    [ -n "$row" ] || die "check_gate5_stage: $crate is not a workspace member"
    verdict="$(classify "$crate" "$TABLE")"
    printf '%s\n' "$verdict"
    printf '  crate               %s %s\n' "$crate" "$(field "$row" 2)"
    printf '  sibling deps        %s\n' "$(field "$row" 3)"
    case "$verdict" in
    MEANINGFUL)
        printf '  Gate 5 is valid at ANY stage for this crate: it resolves nothing from the\n'
        printf '  workspace, so `cargo package -p %s` measures the real tarball.\n' "$crate" ;;
    PRE_BUMP)
        printf '  The workspace version is ALREADY on crates.io, so the verify build resolves\n'
        printf '  the published siblings — the stale API at the same version number. Symbol\n'
        printf '  errors (E0432/E0433) here are EXPECTED, not a release blocker (#2543).\n'
        printf '  Bump the version, then re-check.\n' ;;
    POST_BUMP_PRE_CASCADE)
        printf '  The version is bumped but at least one sibling is not published at it yet, so\n'
        printf '  cargo reports [%s]. Expected until the cascade reaches those crates.\n' "$CARGO_MISMATCH_MSG"
        printf '  That is NOT the same error as `no matching package named X found`, which means\n'
        printf '  the crate was never published at all.\n' ;;
    CASCADE_READY)
        printf '  Every sibling is live at the workspace version. Gate 5 is meaningful NOW and a\n'
        printf '  failure here is a real defect.\n' ;;
    esac
    exit 0
    ;;
"") : ;;
*) die "usage: check_gate5_stage.sh [--self-test | --explain <crate>]" ;;
esac

printf '=== Gate 5 must name the stage it is valid at (check_gate5_stage.sh, #2543) ===\n'
printf 'skill file: %s\n' "$SKILL_FILE"
TABLE="$(sibling_table)" || exit 1
members="$(printf '%s\n' "$TABLE" | grep -c . || true)"
if [ "$members" -lt 20 ]; then
    die "VACUOUS: cargo metadata reported only $members workspace members; the scan is broken."
fi
gate5_command_lines "$SKILL_FILE" | sed 's/^/  gate5 cmd: /'
if check_predicate "$SKILL_FILE" "$TABLE"; then
    printf 'PASS\n'
    exit 0
fi
printf '\nSee `bash scripts/check_gate5_stage.sh --explain <crate>` for the stage this tree is at.\n'
exit 1
