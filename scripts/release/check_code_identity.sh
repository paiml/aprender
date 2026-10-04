#!/usr/bin/env bash
# check_code_identity.sh — case table and planted mutants for the ONE code identity H
# (#4673, BLD-002 row R1; contract: contracts/code-identity-v1.yaml).
#
#   bash scripts/release/check_code_identity.sh --selftest   # case table; exit 0 green, 1 a row BROKE, 2 not_measured
#   bash scripts/release/check_code_identity.sh --mutants    # every planted mutant must turn a row BROKE
#
# Every row but two runs in a temporary git repository. agrees_with_readiness reads
# this repository's last 30 first-parent pairs; one_home reads this repository's scripts/.
# The decision surface rows drive the publish preflight's REAL rule_r7_scope body.
#
# Overrides (the mutant harness uses them): CODE_IDENTITY_LIB, CODE_IDENTITY_PREFLIGHT,
# CODE_IDENTITY_READINESS.
set -uo pipefail

PROG="$(basename -- "$0")"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
LIB="${CODE_IDENTITY_LIB:-$SCRIPT_DIR/lib_code_identity.sh}"
PREFLIGHT="${CODE_IDENTITY_PREFLIGHT:-$REPO/scripts/check_publish_preflight.sh}"
READINESS="${CODE_IDENTITY_READINESS:-$SCRIPT_DIR/release_readiness.sh}"

pass=0; fail=0; nm=0
row() { # row NAME EXPECT ACTUAL [detail]
    if [ "$2" = "$3" ]; then
        printf '  ok    %-40s %s\n' "$1" "${4:-}"; pass=$((pass + 1))
    else
        printf '  BROKE %-40s expected %s, got %s %s\n' "$1" "$2" "$3" "${4:-}"; fail=$((fail + 1))
    fi
}
not_measured() { printf '  NOT_MEASURED %-33s %s\n' "$1" "$2"; nm=$((nm + 1)); }

g() { git -C "$1" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t "${@:2}"; }
commit() { g "$1" add -A >/dev/null 2>&1; g "$1" commit -q --allow-empty -m "$2" >/dev/null 2>&1; g "$1" rev-parse HEAD; }

# fixture REPO-DIR -> a repo with code, a binding yaml, the perf matrix, evidence and a bump tool
fixture() {
    local d="$1"
    mkdir -p "$d/crates/a/src" "$d/contracts/a" "$d/scripts" "$d/evidence/x"
    git init -q "$d"
    printf '[package]\nname = "a"\nversion = "1.0.0"\n' > "$d/Cargo.toml"
    printf 'pub fn f() {}\n' > "$d/crates/a/src/lib.rs"
    printf 'k: 1\n' > "$d/contracts/a/binding.yaml"
    printf 'm: 1\n' > "$d/scripts/perf-matrix.yaml"
    printf '{}\n' > "$d/evidence/x/r.json"
    printf '#!/usr/bin/env bash\nsed -i "s/^version = .*/version = \\"$1\\"/" Cargo.toml\n' > "$d/scripts/bump-version.sh"
    commit "$d" base >/dev/null
}

h() { (cd "$1" && code_identity "$2"); }
same() { (cd "$1" && code_identity_same "$2" "$3"); echo $?; }

# r7 REPO CUT -> runs the preflight's own rule_r7_scope body (HEAD vs CUT) with a judge that
# says yes; prints "rc=<n>" then its output
r7() {
    local d="$1" cut="$2" body judge
    body="$(sed -n '/^rule_r7_scope() {$/,/^}$/p' "$PREFLIGHT")"
    [ -n "$body" ] || { echo "rc=NOFN"; return; }
    judge="$d/.judge.sh"
    printf '#!/usr/bin/env bash\necho "OPERATOR EMERGENCY SCOPE crux-smoke satisfied"\n' > "$judge"
    (
        # shellcheck disable=SC1090
        . "$LIB" || exit 9
        eval "$body"
        SCOPE=crux-smoke CUT_COMMIT="$cut"
        out="$(rule_r7_scope "$d" 1.0.0 "$judge" 2>&1)"; rc=$?
        printf 'rc=%s\n%s\n' "$rc" "$out"
    )
}

selftest() {
    # shellcheck disable=SC1090
    . "$LIB" || { echo "FAIL  cannot source $LIB"; exit 2; }
    local tmp d b c e out
    tmp="$(mktemp -d "${TMPDIR:-/tmp}/check-code-identity.XXXXXX")" || exit 2
    # shellcheck disable=SC2064
    trap "rm -rf -- '${tmp:?}'" EXIT
    printf -- '--- %s: code identity H ---\n' "$PROG"

    d="$tmp/r"; fixture "$d"; b="$(g "$d" rev-parse HEAD)"

    printf 'k: 2\n' > "$d/contracts/a/binding.yaml"; c="$(commit "$d" binding)"
    row binding_yaml_edit_changes_H 1 "$(same "$d" "$b" "$c")" "(a build.rs input outside crates/)"
    printf 'm: 2\n' > "$d/scripts/perf-matrix.yaml"; e="$(commit "$d" perf)"
    row perf_matrix_edit_changes_H 1 "$(same "$d" "$c" "$e")"
    b="$e"; printf '{"a":1}\n' > "$d/evidence/x/r.json"; printf 'n\n' > "$d/evidence/new"; c="$(commit "$d" evidence)"
    row evidence_edit_keeps_H 0 "$(same "$d" "$b" "$c")" "(recording a result never changes H)"
    b="$c"; mkdir -p "$d/crates/a/evidence"; printf 'x\n' > "$d/crates/a/evidence/f"; c="$(commit "$d" nested)"
    row nested_evidence_dir_changes_H 1 "$(same "$d" "$b" "$c")" "(only the ROOT evidence/ is excluded)"
    b="$c"; mkdir -p "$d/evidence-old"; printf 'x\n' > "$d/evidence-old/f"; printf 'x\n' > "$d/evidence.md"; c="$(commit "$d" lookalike)"
    row evidence_lookalike_changes_H 1 "$(same "$d" "$b" "$c")" "(evidence-old/, evidence.md)"
    b="$c"; chmod +x "$d/crates/a/src/lib.rs"; c="$(commit "$d" mode)"
    row mode_change_changes_H 1 "$(same "$d" "$b" "$c")"
    b="$c"; g "$d" mv scripts/perf-matrix.yaml scripts/perf-matrix-renamed.yaml; c="$(commit "$d" rename)"
    row rename_changes_H 1 "$(same "$d" "$b" "$c")" "(same bytes, new path)"
    b="$c"; printf 'x\n' > "$d/evidence/"$'a\nb\tc'; c="$(commit "$d" nlpath)"
    row newline_tab_path_under_evidence_keeps_H 0 "$(same "$d" "$b" "$c")"
    b="$c"; printf 'x\n' > "$d/"$'x\nevidence'; c="$(commit "$d" nlpath2)"
    row newline_path_ending_in_evidence_changes_H 1 "$(same "$d" "$b" "$c")"
    b="$c"; printf 'x\n' > "$d/"$'evidence\nfoo'; c="$(commit "$d" nlpath3)"
    row newline_path_starting_with_evidence_changes_H 1 "$(same "$d" "$b" "$c")"
    b="$c"; out="$(h "$d" HEAD)"
    printf 'pub fn dirty() {}\n' >> "$d/crates/a/src/lib.rs"; printf 'u\n' > "$d/untracked.rs"
    printf 's\n' > "$d/staged.rs"; g "$d" add staged.rs >/dev/null
    row untracked_and_uncommitted_ignored "$out" "$(h "$d" HEAD)"
    g "$d" reset -q --hard >/dev/null; rm -f -- "$d/untracked.rs"
    c="$(g "$d" commit-tree -m twin "$(g "$d" rev-parse 'HEAD^{tree}')")"
    row same_tree_two_commits_same_H 0 "$(same "$d" "$b" "$c")" "(H is of the code, not the commit)"
    case "$out" in H=????????????????????????????????????????????????????????????????) e=64hex ;; *) e="$out" ;; esac
    row identity_is_H_64hex 64hex "$e"
    (cd "$d" && code_identity 0123456789abcdef0123456789abcdef01234567 >/dev/null); row unknown_rev_not_measured 2 "$?"
    row unknown_rev_same_not_measured 2 "$(same "$d" HEAD no-such-rev)"
    row unknown_first_rev_same_not_measured 2 "$(same "$d" no-such-rev HEAD)"
    (cd "$d" && _code_identity_hash 0123456789abcdef0123456789abcdef01234567 >/dev/null); row missing_object_hash_not_measured 2 "$?"

    # code_identity_bump_of: a version-only bump reuses H; anything more does not; no tool = not_measured
    d="$tmp/bump"; fixture "$d"; b="$(g "$d" rev-parse HEAD)"
    sed -i 's/^version = .*/version = "1.0.1"/' "$d/Cargo.toml"; c="$(commit "$d" bump)"
    (cd "$d" && code_identity_bump_of "$b" "$c" 1.0.1); row version_only_bump_reuses 0 "$?"
    printf 'pub fn more() {}\n' >> "$d/crates/a/src/lib.rs"; e="$(commit "$d" bump+code)"
    (cd "$d" && code_identity_bump_of "$b" "$e" 1.0.1); row bump_plus_code_edit_does_not 1 "$?"
    g "$d" rm -q scripts/bump-version.sh >/dev/null; e="$(commit "$d" notool)"
    (cd "$d" && code_identity_bump_of "$e" "$c" 1.0.1); row bump_tool_missing_not_measured 2 "$?"
    row bump_left_no_worktree 1 "$(g "$d" worktree list | wc -l)"

    # DECISION SURFACE: the publish preflight's emergency rule R7 asks the one identity
    printf -- '--- decision surfaces ---\n'
    d="$tmp/r7"; fixture "$d"; b="$(g "$d" rev-parse HEAD)"
    printf 'k: 9\n' > "$d/contracts/a/binding.yaml"; commit "$d" binding >/dev/null
    out="$(r7 "$d" "$b")"
    row r7_binding_yaml_after_cut_refuses rc=1 "$(head -n 1 <<< "$out")" "(today's four paths call this the same code)"
    d="$tmp/r7ev"; fixture "$d"; b="$(g "$d" rev-parse HEAD)"
    printf '{"r":1}\n' > "$d/evidence/x/r.json"; commit "$d" receipts >/dev/null
    out="$(r7 "$d" "$b")"
    row r7_evidence_after_cut_passes rc=0 "$(head -n 1 <<< "$out")"

    # one home: no script keeps a private definition of "same code"
    out="$(grep -rnE "diff --quiet[^|;]*(:\(exclude\)evidence|-- crates src Cargo\.toml Cargo\.lock)" \
        "$REPO/scripts" --include='*.sh' 2>/dev/null | grep -vE "/(check|lib)_code_identity\.sh:" \
        | sed -e "s|^$REPO/||" -e 's/:.*//' | sort -u | tr '\n' ' ')"
    [ -f "$PREFLIGHT" ] && out="$out$(grep -lE 'diff --quiet[^|;]*-- crates src Cargo\.toml Cargo\.lock' "$PREFLIGHT" 2>/dev/null | sed 's|.*/|override:|')"
    [ -f "$READINESS" ] && out="$out$(grep -lE "diff --quiet[^|;]*:\(exclude\)evidence" "$READINESS" 2>/dev/null | sed 's|.*/|override:|')"
    row one_home_no_private_same_code_rule "" "${out% }"

    # the readiness rule and H agree on this repository's history
    local n=0 dis=0 p
    n="$(git -C "$REPO" rev-list --first-parent --count HEAD 2>/dev/null)" || n=0
    if [ "$n" -gt 30 ]; then
        n=0
        for c in $(git -C "$REPO" rev-list --first-parent -n 30 HEAD); do
            p="$(git -C "$REPO" rev-parse "$c^1")"
            (cd "$REPO" && code_identity_same "$p" "$c"); e=$?
            git -C "$REPO" diff --quiet "$p" "$c" -- . ':(exclude)evidence'
            [ "$e" = "$?" ] || dis=$((dis + 1)); n=$((n + 1))
        done
        row agrees_with_readiness "0/30" "$dis/$n" "(disagreements over the last 30 first-parent pairs)"
    else
        not_measured agrees_with_readiness "fewer than 31 first-parent commits (shallow clone?)"
    fi

    printf -- '--- %s/%s rows, %s not_measured ---\n' "$pass" "$((pass + fail))" "$nm"
    [ "$fail" -eq 0 ] || exit 1
    [ "$nm" -eq 0 ] || exit 2
    exit 0
}

# mutant NAME FILE(lib|preflight|readiness) OLD NEW -> applies to a copy; a kill needs a BROKE row
mutants() {
    local tmp k=0 t=0 survivors=""
    tmp="$(mktemp -d "${TMPDIR:-/tmp}/check-code-identity-mut.XXXXXX")" || exit 2
    # shellcheck disable=SC2064
    trap "rm -rf -- '${tmp:?}'" EXIT
    mutant() {
        local name="$1" which="$2" old="$3" new="$4" src dst out rc body
        case "$which" in
            lib) src="$LIB" ;; preflight) src="$PREFLIGHT" ;; readiness) src="$READINESS" ;;
        esac
        dst="$tmp/$name.sh"; t=$((t + 1))
        body="$(cat -- "$src"; printf x)"; body="${body%x}"
        printf '%s' "${body/"$old"/"$new"}" > "$dst"
        if cmp -s "$src" "$dst"; then printf '  NOT-APPLIED %s\n' "$name"; survivors="$survivors $name"; return; fi
        if ! bash -n "$dst" 2>/dev/null; then printf '  NO-PARSE    %s\n' "$name"; survivors="$survivors $name"; return; fi
        out="$(env "CODE_IDENTITY_$(tr '[:lower:]' '[:upper:]' <<< "$which")=$dst" bash "$0" --selftest 2>&1)"; rc=$?
        if [ "$rc" = 1 ] && grep -q '^  BROKE ' <<< "$out"; then
            printf '  killed      %-34s by %s\n' "$name" "$(grep -m1 '^  BROKE ' <<< "$out" | awk '{print $2}')"; k=$((k + 1))
        else
            printf '  SURVIVED    %-34s rc %s\n' "$name" "$rc"; survivors="$survivors $name"
        fi
    }
    printf -- '--- %s --mutants ---\n' "$PROG"
    mutant drop_evidence_filter lib '\tevidence(/|$)' '\t__never__(/|$)'
    mutant evidence_glob_no_slash lib '\tevidence(/|$)' '\tevidence'
    mutant evidence_unanchored lib "\$'^[^\\t]*\\tevidence(/|\$)'" "'evidence(/|\$)'"
    mutant evidence_dollar_multiline lib "\$'^[^\\t]*\\tevidence(/|\$)'" "\$'^[^\\t]*\\tevidence(/|\$|[[:space:]])'"
    mutant ignores_root lib 'local rev="${1:-}" root="${2:-.}"' 'local rev="${1:-}" root=.'
    mutant hash_the_index lib 'git "$@" ls-tree -r -z --full-tree "$treeish"' 'git "$@" ls-files -s -z'
    mutant drop_path_column lib '            sha256sum' "            sed -z 's/\\t.*//' | sha256sum"
    mutant drop_mode_column lib 'LC_ALL=C grep -zvE' "sed -z 's/^[0-7]* //' | LC_ALL=C grep -zvE"
    mutant unknown_rev_falls_back lib '"${rev}^{commit}" 2>/dev/null)" || return 2' '"${rev}^{commit}" 2>/dev/null)" || c=HEAD'
    mutant same_on_not_measured lib 'a="$(code_identity "${1:-}" "${3:-.}")" || return 2' 'a="$(code_identity "${1:-}" "${3:-.}")" || return 0'
    mutant pipestatus_ignored lib '|| exit 2' '|| :'
    mutant bump_always_same lib 'if [ "H=$hm" = "$hb" ]' 'if true'
    mutant bump_no_tool_is_different lib '    rc=2
    if git worktree' '    rc=1
    if git worktree'
    mutant r7_back_to_four_paths preflight 'code_identity_same "$cut" "$head" "$root"' 'git -C "$root" diff --quiet "$cut" "$head" -- crates src Cargo.toml Cargo.lock'
    mutant readiness_private_rule readiness 'code_identity_same "$x" "$commit" "$root"' 'git -C "$root" diff --quiet "$x" "$commit" -- . '"':(exclude)evidence'"
    printf -- '--- %s/%s mutants killed ---\n' "$k" "$t"
    [ "$k" = "$t" ] || { printf 'FAIL  survived:%s\n' "$survivors"; exit 1; }
}

case "${1:-}" in
    --selftest) selftest ;;
    --mutants) mutants ;;
    *) printf 'usage: %s --selftest | --mutants\n' "$PROG" >&2; exit 2 ;;
esac
