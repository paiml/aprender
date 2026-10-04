#!/usr/bin/env bash
# bumped_publish_paths.sh — measure what publish would upload on the BUMPED tree, and refuse a
# release whose bump commit would upload something else.
#
#   bumped_publish_paths.sh night <V> --out FILE [--root R] [--rev REV]
#   bumped_publish_paths.sh release <V> --night FILE [--root R] [--rev BUMP]
#   bumped_publish_paths.sh --selftest | --mutants
#
# WHY. A nightly that measures the publish path on commit C measures C at its OLD version, but
# release publishes the bump commit of C. So the night bumps C first and measures there:
#   night    tag_on_bump_sandbox.sh (#4692, reused unchanged) clones R at REV with no remote, runs
#            that commit's own scripts/bump-version.sh <V>, commits, and calls this script inside
#            the clone (as its judge). Here, every publishable crate of every workspace gets
#            `cargo package --list` — the file set `cargo publish` packs and uploads — and the
#            sorted "crate<TAB>path" lines go to FILE with a header (version, source, tree).
#   release  the same listing at the real bump commit BUMP (a no-remote clone), compared with the
#            night's FILE for the same V. Equal = PASS. Any path added or dropped = REFUSE.
#
# NO UPLOAD IS POSSIBLE. Neither mode runs `cargo publish`; the selftest row source_has_no_upload
# fails if this file ever does. And both modes refuse to START (rc 3) when a registry token is
# reachable: CARGO_REGISTRY_TOKEN, any CARGO_REGISTRIES_<NAME>_TOKEN, a credential provider
# setting, or a credentials file in CARGO_HOME. Run it with a token-less CARGO_HOME.
#
# Exit: 0 PASS (release) / measured (night); 1 REFUSE; 2 NOT_MEASURED (any step that cannot run:
# no clone, a bump that fails, cargo or jq missing, a crate that cannot be listed, a universe
# under BPR_MIN_CRATES, a night file for another version — never a pass, L25); 3 caller error or
# refused to start. The release check is a NEW REFUSAL: built, wired nowhere, until operator
# sign-off. Report-only nights first (L31).
set -uo pipefail
PROG=bumped_publish_paths
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SELF="$HERE/$(basename "${BASH_SOURCE[0]}")"
SANDBOX=${BPR_SANDBOX:-$HERE/tag_on_bump_sandbox.sh}
CARGO=${BPR_CARGO:-cargo}
MIN_CRATES=${BPR_MIN_CRATES:-70}
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_OBJECT_DIRECTORY GIT_ALTERNATE_OBJECT_DIRECTORIES \
    GIT_COMMON_DIR GIT_NAMESPACE GIT_PREFIX
TMP=""
cleanup() { [ -z "$TMP" ] || rm -rf "${TMP:?}"; TMP=""; }
trap cleanup EXIT

usage() { echo "$PROG: usage: night <V> --out FILE [--root R] [--rev REV] | release <V> --night FILE [--root R] [--rev BUMP] | --selftest | --mutants" >&2; exit 3; }
nm() { echo "NOT_MEASURED $PROG: $*"; exit 2; }

# token_reachable: prints what it found and returns 0 when any registry credential is reachable
token_reachable() {
    local n ch=${CARGO_HOME:-$HOME/.cargo} f hit=1
    [ -z "${CARGO_REGISTRY_TOKEN:-}" ] || { echo "env CARGO_REGISTRY_TOKEN"; hit=0; }
    for n in $(compgen -e); do
        case $n in
            CARGO_REGISTRIES_*_TOKEN) [ -z "${!n}" ] || { echo "env $n"; hit=0; } ;;
            CARGO_REGISTRY_CREDENTIAL_PROVIDER|CARGO_REGISTRY_GLOBAL_CREDENTIAL_PROVIDERS|CARGO_REGISTRIES_*_CREDENTIAL_PROVIDER)
                [ -z "${!n}" ] || { echo "env $n"; hit=0; } ;;
        esac
    done
    for f in "$ch/credentials.toml" "$ch/credentials"; do [ ! -e "$f" ] || { echo "file CARGO_HOME/$(basename "$f")"; hit=0; }; done
    return "$hit"
}
refuse_if_token() {
    local t
    if t=$(token_reachable); then
        echo "REFUSED-TO-START $PROG: a registry credential is reachable ($(echo "$t" | tr '\n' ',' | sed 's/,$//')); run with a token-less CARGO_HOME"
        exit 3
    fi
}

# listing <tree-dir>: sorted "crate<TAB>path" for every publishable crate of every workspace; rc 2 on any gap
listing() {
    local c=$1 ws m name man n=0 out
    command -v jq > /dev/null || { echo "jq is not installed" >&2; return 2; }
    command -v "$CARGO" > /dev/null || { echo "cargo ($CARGO) is not installed" >&2; return 2; }
    out=$(mktemp) || return 2
    for ws in Cargo.toml crates/facades/Cargo.toml; do
        [ -f "$c/$ws" ] || { [ "$ws" != Cargo.toml ] && continue; echo "no root Cargo.toml" >&2; rm -f -- "${out:?}"; return 2; }
        m=$(cd "$c" && "$CARGO" metadata --no-deps --format-version 1 --manifest-path "$c/$ws" 2> /dev/null) \
            || { echo "cargo metadata failed for $ws" >&2; rm -f -- "${out:?}"; return 2; }
        while IFS=$'\t' read -r name man; do
            [ -n "$name" ] || continue
            grep -q -F -x -e "$name" "$out.names" 2> /dev/null && continue
            echo "$name" >> "$out.names"; n=$((n + 1))
            (cd "$(dirname "$man")" && "$CARGO" package --list --manifest-path "$man") > "$out.one" 2> /dev/null \
                || { echo "cargo package --list failed for $name" >&2; rm -f -- "${out:?}" "${out:?}.names" "${out:?}.one"; return 2; }
            [ -s "$out.one" ] || { echo "cargo package --list printed nothing for $name" >&2; rm -f -- "${out:?}" "${out:?}.names" "${out:?}.one"; return 2; }
            awk -v N="$name" 'NF { printf "%s\t%s\n", N, $0 }' "$out.one" >> "$out"
        done < <(printf '%s\n' "$m" | jq -r '.packages[] | select(.publish == null or (.publish | length) > 0) | [.name, .manifest_path] | @tsv')
    done
    if [ "$n" -lt "$MIN_CRATES" ]; then
        echo "the universe is $n publishable crate(s), under the floor of $MIN_CRATES" >&2
        rm -f -- "${out:?}" "${out:?}.names" "${out:?}.one"; return 2
    fi
    LC_ALL=C sort -u "$out"
    rm -f -- "${out:?}" "${out:?}.names" "${out:?}.one"
}

# header <dir> <V> <source-sha>: the night file's header lines
header() {
    printf '# bumped-publish-paths v1\n# version: %s\n# source: %s\n# tree: %s\n' "$2" "$3" "$(git -C "$1" rev-parse 'HEAD^{tree}')"
}

# measure (the sandbox's judge role): list the clone, write BPR_OUT
measure() {
    local v=$1 c=$2 body src
    [ -n "${BPR_OUT:-}" ] || nm "measure role without BPR_OUT"
    src=$(git -C "$c" rev-parse 'HEAD^') || nm "the bumped clone has no parent commit"
    body=$(listing "$c") || nm "the bumped tree could not be listed"
    { header "$c" "$v" "$src"; printf '# crates: %s\n# paths: %s\n' "$(printf '%s\n' "$body" | cut -f1 | sort -u | wc -l | tr -d ' ')" \
        "$(printf '%s\n' "$body" | wc -l | tr -d ' ')"; printf '%s\n' "$body"; } > "$BPR_OUT" || nm "could not write $BPR_OUT"
    echo "measured: $(grep -c -v '^#' "$BPR_OUT") path(s) of the tree bumped to $v"
}

night() {
    local v=$1 out=$2 root=$3 rev=$4 rc=0
    refuse_if_token
    [ -f "$SANDBOX" ] || nm "the sandbox $SANDBOX is not there"
    rm -f -- "${out:?}"
    BPR_ROLE=measure BPR_OUT=$out TOB_JUDGE=$SELF bash "$SANDBOX" "$v" --root "$root" --rev "$rev" || rc=$?
    [ "$rc" = 0 ] || { echo "NOT_MEASURED $PROG: the sandbox run ended rc=$rc"; exit 2; }
    [ -s "$out" ] && grep -q -x -F -e "# version: $v" "$out" || nm "the sandbox ended 0 but wrote no measurement for $v"
    echo "night: $out"
}

release() {
    local v=$1 nf=$2 root=$3 rev=$4 h hv c body d
    refuse_if_token
    [ -s "$nf" ] || nm "no night measurement at $nf"
    grep -q -x -F -e '# bumped-publish-paths v1' "$nf" || nm "$nf is not a bumped-publish-paths v1 file"
    grep -q -x -F -e "# version: $v" "$nf" || nm "the night measured $(sed -n 's/^# version: //p' "$nf" | head -1), not $v"
    h=$(git -C "$root" rev-parse -q --verify "$rev^{commit}" 2> /dev/null) || nm "$rev does not resolve to a commit"
    hv=$(git -C "$root" show "$h:Cargo.toml" 2> /dev/null | awk -F'"' '/^\[workspace\.package\]/{w=1;next} /^\[/{w=0} w&&/^version[ \t]*=/{print $2; exit}')
    [ "$hv" = "$v" ] || nm "${h:0:12} carries ${hv:-no version}, not $v: it is not the bump commit"
    TMP=$(mktemp -d) || { TMP=""; nm "no scratch directory"; }
    c="$TMP/r"
    { git clone -q --shared --no-checkout "$root" "$c" 2> /dev/null && git -C "$c" remote remove origin \
        && git -C "$c" -c advice.detachedHead=false checkout -q --detach "$h"; } || nm "no clone of ${h:0:12}"
    body=$(listing "$c") || nm "the bump commit ${h:0:12} could not be listed"
    d=$(diff <(grep -v '^#' "$nf") <(printf '%s\n' "$body") | grep -E '^[<>]' | sed -e 's/^</- night only:/' -e 's/^>/+ release only:/')
    if [ -n "$d" ]; then
        printf '%s\n' "$d" | head -n 20
        echo "REFUSE $PROG: the bump commit ${h:0:12} would publish $(printf '%s\n' "$d" | wc -l | tr -d ' ') path(s) other than the night measured for $v"
        exit 1
    fi
    echo "PASS $PROG: ${h:0:12} publishes the $(printf '%s\n' "$body" | wc -l | tr -d ' ') path(s) the night measured for $v (night tree $(sed -n 's/^# tree: //p' "$nf" | cut -c1-12), bump tree $(git -C "$c" rev-parse 'HEAD^{tree}' | cut -c1-12))"
}

# ---------------------------------------------------------------------------------------------
# selftest: fixture repos with a stub bump procedure and a stub cargo (metadata + package --list)
mkfix() {   # mkfix DIR: root at 1.2.2, crates a and b publishable, c publish=false, a stub bump
    local d=$1
    git init -q "$d" && git -C "$d" config user.email fixture && git -C "$d" config user.name t \
        && git -C "$d" config core.hooksPath /dev/null && git -C "$d" config commit.gpgsign false || return 1
    printf '[workspace]\nmembers = ["crates/a", "crates/b", "crates/c"]\n\n[workspace.package]\nversion = "1.2.2"\n' > "$d/Cargo.toml"
    mkdir -p "$d/scripts" "$d/crates/a/src" "$d/crates/b/src" "$d/crates/c/src"
    printf '%s\n' 'sed -i "s/^version = \".*\"/version = \"$1\"/" Cargo.toml' > "$d/scripts/bump-version.sh"
    for x in a b c; do printf '[package]\nname = "%s"\n' "$x" > "$d/crates/$x/Cargo.toml"; echo "// $x" > "$d/crates/$x/src/lib.rs"; done
    git -C "$d" add -A && git -C "$d" commit -qm init && git -C "$d" remote add origin https://example.invalid/r.git
}
mkstub() {  # mkstub FILE: cargo metadata lists a, b (publishable) and c (publish = []); package --list lists the crate dir
    cat > "$1" <<'STUB'
#!/usr/bin/env bash
mp=""; a=("$@")
for ((i = 0; i < ${#a[@]}; i++)); do [ "${a[i]}" = --manifest-path ] && mp=${a[i+1]}; done
case $1 in
  metadata)
    [ "${STUB_META_FAIL:-0}" = 1 ] && exit 101
    r=$(dirname "$mp"); [ "$(basename "$mp")" = Cargo.toml ] && [ -d "$r/crates/a" ] || { echo '{"packages":[]}'; exit 0; }
    printf '{"packages":[{"name":"a","manifest_path":"%s/crates/a/Cargo.toml","publish":null},{"name":"b","manifest_path":"%s/crates/b/Cargo.toml","publish":["crates-io"]},{"name":"c","manifest_path":"%s/crates/c/Cargo.toml","publish":[]}]}\n' "$r" "$r" "$r" ;;
  package)
    [ "${STUB_PKG_FAIL:-}" = "$(basename "$(dirname "$mp")")" ] && { echo Cargo.toml; exit 101; }
    (cd "$(dirname "$mp")" && find . -type f | sed 's|^\./||' | LC_ALL=C sort; echo Cargo.toml.orig) ;;
  *) exit 101 ;;
esac
STUB
    chmod +x "$1"
}
# bumpcommit DIR V [extra-cmd]: a release bump commit on DIR's HEAD, optionally with one more change
bumpcommit() {
    (cd "$1" && bash scripts/bump-version.sh "$2" && { [ -z "${3:-}" ] || eval "$3"; } && git add -A && git commit -qm "v$2") > /dev/null 2>&1
}
selftest() {
    local subj=${1:-$SELF} quiet=${2:-} d fail=0 n=0 got rc b0
    command -v git > /dev/null && command -v jq > /dev/null || { echo "$PROG selftest: needs git and jq"; return 2; }
    [ -f "$SANDBOX" ] || { echo "$PROG selftest: no sandbox at $SANDBOX"; return 2; }
    d=$(mktemp -d) || return 2
    mkstub "$d/cargo"; mkdir -p "$d/home-clean" "$d/home-cred"; : > "$d/home-cred/credentials.toml"
    run() { got=$(env -u CARGO_REGISTRY_TOKEN CARGO_HOME="$d/home-clean" BPR_CARGO="$d/cargo" BPR_MIN_CRATES="${MIN:-2}" BPR_SANDBOX="$SANDBOX" "$@" 2>&1); rc=$?; }
    check() { n=$((n + 1))
        if [ "$rc" = "$2" ] && [[ "$got" == *"$3"* ]]; then [ -n "$quiet" ] || echo "  ok   $1 (rc=$rc)"
        else echo "  FAIL $1: want rc=$2 '$3', got rc=$rc: $(printf '%s' "$got" | tr '\n' '|' | cut -c1-240)"; fail=$((fail + 1)); fi; }
    mkfix "$d/r"; b0=$(git -C "$d/r" for-each-ref --format='%(refname) %(objectname)'; git -C "$d/r" rev-parse HEAD)
    run bash "$subj" night 1.2.3 --out "$d/n.txt" --root "$d/r"; check night_measures 0 "night: $d/n.txt"
    got=$(cat "$d/n.txt" 2> /dev/null); rc=0; check night_lists_publishable_only 0 $'a\tsrc/lib.rs'
    if grep -q -P '^c\t' "$d/n.txt" 2> /dev/null; then rc=1; got="publish=false crate c listed"; else rc=0; got="c absent"; fi; check night_skips_publish_false 0 "c absent"
    got=$(grep -c -x -F -e '# version: 1.2.3' "$d/n.txt" 2> /dev/null); rc=0; check night_header_version 0 1
    got="$(git -C "$d/r" for-each-ref --format='%(refname) %(objectname)'; git -C "$d/r" rev-parse HEAD)"
    [ "$got" = "$b0" ] && { rc=0; got=untouched; } || rc=1; check night_source_untouched 0 untouched
    run env CARGO_REGISTRY_TOKEN=x bash "$subj" night 1.2.3 --out "$d/t.txt" --root "$d/r"; check night_refuses_env_token 3 "REFUSED-TO-START"
    run env CARGO_REGISTRIES_ALT_TOKEN=x bash "$subj" night 1.2.3 --out "$d/t.txt" --root "$d/r"; check night_refuses_named_registry_token 3 "CARGO_REGISTRIES_ALT_TOKEN"
    run env CARGO_REGISTRY_CREDENTIAL_PROVIDER=cargo:token bash "$subj" night 1.2.3 --out "$d/t.txt" --root "$d/r"; check night_refuses_credential_provider 3 "REFUSED-TO-START"
    run env CARGO_HOME="$d/home-cred" bash "$subj" night 1.2.3 --out "$d/t.txt" --root "$d/r"; check night_refuses_credentials_file 3 "credentials.toml"
    run env STUB_PKG_FAIL=b bash "$subj" night 1.2.3 --out "$d/t.txt" --root "$d/r"; check night_package_fails_nm 2 "NOT_MEASURED"
    run env STUB_META_FAIL=1 bash "$subj" night 1.2.3 --out "$d/t.txt" --root "$d/r"; check night_metadata_fails_nm 2 "cargo metadata failed"
    MIN=3 run bash "$subj" night 1.2.3 --out "$d/t.txt" --root "$d/r"; check night_universe_under_floor_nm 2 "NOT_MEASURED"
    run bash "$subj" night 1.2.2 --out "$d/t.txt" --root "$d/r"; check night_nothing_to_bump_nm 2 "NOT_MEASURED"
    git -C "$d" clone -q "$d/r" "$d/eq" && bumpcommit "$d/eq" 1.2.3
    run bash "$subj" release 1.2.3 --night "$d/n.txt" --root "$d/eq"; check release_same_paths_pass 0 "PASS"
    git -C "$d" clone -q "$d/r" "$d/add" && bumpcommit "$d/add" 1.2.3 'echo x > crates/a/src/extra.rs'
    run bash "$subj" release 1.2.3 --night "$d/n.txt" --root "$d/add"; check release_extra_path_refuses 1 $'+ release only: a\tsrc/extra.rs'
    git -C "$d" clone -q "$d/r" "$d/del" && bumpcommit "$d/del" 1.2.3 'git rm -q crates/b/src/lib.rs'
    run bash "$subj" release 1.2.3 --night "$d/n.txt" --root "$d/del"; check release_dropped_path_refuses 1 $'- night only: b\tsrc/lib.rs'
    run bash "$subj" release 1.2.3 --night "$d/n.txt" --root "$d/r"; check release_not_bump_commit_nm 2 "not the bump commit"
    run bash "$subj" release 1.2.4 --night "$d/n.txt" --root "$d/eq"; check release_night_other_version_nm 2 "not 1.2.4"
    run bash "$subj" release 1.2.3 --night "$d/none.txt" --root "$d/eq"; check release_night_missing_nm 2 "NOT_MEASURED"
    git -C "$d" clone -q "$d/r" "$d/v4" && bumpcommit "$d/v4" 1.2.4
    run bash "$subj" release 1.2.4 --night "$d/n.txt" --root "$d/v4"; check release_night_for_other_bump_nm 2 "the night measured 1.2.3, not 1.2.4"
    printf '%s\n' 'printf "# bumped-publish-paths v1\n# version: %s\n" "$1" > "$BPR_OUT"; exit 1' > "$d/sbx-late-fail.sh"
    run env BPR_SANDBOX="$d/sbx-late-fail.sh" bash "$subj" night 1.2.3 --out "$d/t.txt" --root "$d/r"; check night_sandbox_fails_after_write_nm 2 "the sandbox run ended rc=1"
    run env CARGO_REGISTRY_TOKEN=x bash "$subj" release 1.2.3 --night "$d/n.txt" --root "$d/eq"; check release_refuses_token 3 "REFUSED-TO-START"
    run env STUB_PKG_FAIL=a bash "$subj" release 1.2.3 --night "$d/n.txt" --root "$d/eq"; check release_package_fails_nm 2 "NOT_MEASURED"
    run bash "$subj" release 1.2.3 --root "$d/eq"; check caller_error_no_night 3 "usage"
    if grep -v -E '^[[:space:]]*#' "$subj" | grep -q -E '(cargo|CARGO"?\}?) +publish'; then rc=1; got="an upload verb"; else rc=0; got="no upload verb"; fi
    check source_has_no_upload 0 "no upload verb"
    rm -rf -- "${d:?}"
    echo "$PROG selftest: $((n - fail))/$n rows pass"
    [ "$fail" = 0 ]
}

# mutants: each drops one check; a mutant that does not apply, or breaks syntax, is an ERROR, not a survivor
MUTANTS='M01 env token ignored@@[ -z "${CARGO_REGISTRY_TOKEN:-}" ] || { echo "env CARGO_REGISTRY_TOKEN"; hit=0; }@@:
M02 named registry token ignored@@CARGO_REGISTRIES_*_TOKEN) [ -z "${!n}" ] || { echo "env $n"; hit=0; } ;;@@CARGO_REGISTRIES_*_TOKEN) : ;;
M03 credentials file ignored@@[ ! -e "$f" ] || { echo "file CARGO_HOME/$(basename "$f")"; hit=0; }@@:
M04 refusal does not stop@@        exit 3
    fi@@        :
    fi
M05 provider ignored@@[ -z "${!n}" ] || { echo "env $n"; hit=0; } ;;
        esac@@: ;;
        esac
M06 diff ignored@@    if [ -n "$d" ]; then@@    if false; then
M07 night version not checked@@grep -q -x -F -e "# version: $v" "$nf" || nm@@true || nm
M08 bump commit not checked@@[ "$hv" = "$v" ] || nm@@true || nm
M09 package failure passes@@(cd "$(dirname "$man")" && "$CARGO" package --list --manifest-path "$man") > "$out.one" 2> /dev/null \@@{ (cd "$(dirname "$man")" && "$CARGO" package --list --manifest-path "$man") || true; } > "$out.one" 2> /dev/null \
M10 universe floor dropped@@if [ "$n" -lt "$MIN_CRATES" ]; then@@if false; then
M11 publish=false listed@@select(.publish == null or (.publish | length) > 0)@@select(true)
M12 night sandbox failure passes@@[ "$rc" = 0 ] || { echo "NOT_MEASURED $PROG: the sandbox run ended rc=$rc"; exit 2; }@@:
M13 metadata failure passes@@|| { echo "cargo metadata failed for $ws" >&2; rm -f -- "${out:?}"; return 2; }@@|| m=""'
mutants() {
    local tmp line name from to k=0 t=0 e=0
    tmp=$(mktemp -d) || return 2
    # multi-line mutants: split MUTANTS on lines that start with M<nn>
    printf '%s\n' "$MUTANTS" | awk '/^M[0-9][0-9] /{ if (b != "") print b; b = $0; next } { b = b "\\n" $0 } END { if (b != "") print b }' > "$tmp/list"
    while IFS= read -r line; do
        name=${line%%@@*}; from=${line#*@@}; to=${from#*@@}; from=${from%%@@*}
        from=$(printf '%b' "$from"); to=$(printf '%b' "$to"); t=$((t + 1))
        RR_FROM=$from RR_TO=$to awk 'BEGIN { RS = "\001" } { i = index($0, ENVIRON["RR_FROM"]); if (!i) exit 1
            printf "%s%s%s", substr($0, 1, i - 1), ENVIRON["RR_TO"], substr($0, i + length(ENVIRON["RR_FROM"])) }' "$SELF" > "$tmp/m.sh"
        if [ "${PIPESTATUS[0]:-0}" != 0 ] || cmp -s "$SELF" "$tmp/m.sh" || ! bash -n "$tmp/m.sh" 2> /dev/null; then
            echo "ERROR    $name (did not apply or broke syntax)"; e=$((e + 1)); continue
        fi
        if selftest "$tmp/m.sh" quiet > "$tmp/out" 2>&1; then echo "SURVIVED $name"
        else k=$((k + 1)); echo "killed   $name ($(grep -c 'FAIL' "$tmp/out") rows red)"; fi
    done < "$tmp/list"
    rm -rf -- "${tmp:?}"
    echo "$PROG mutants: killed=$k total=$t errors=$e"
    [ "$k" = "$t" ] && [ "$e" = 0 ]
}

main() {
    local mode=${1:-} v="" out="" nf="" root=. rev=HEAD
    case $mode in
        --selftest) selftest; return ;;
        --mutants) mutants; return ;;
        night|release) ;;
        *) [ "${BPR_ROLE:-}" = measure ] || usage
           v=${1:-}; [ "${2:-}" = --root ] && [ -n "${3:-}" ] || nm "measure role needs <V> --root <clone>"
           measure "$v" "$3"; return ;;
    esac
    shift; v=${1:-}; [ "$#" -gt 0 ] && shift
    [[ "$v" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || usage
    while [ "$#" -gt 0 ]; do
        case $1 in
            --out) out=${2:-}; shift 2 || usage ;;
            --night) nf=${2:-}; shift 2 || usage ;;
            --root) root=${2:-}; shift 2 || usage ;;
            --rev) rev=${2:-}; shift 2 || usage ;;
            *) usage ;;
        esac
    done
    if [ "$mode" = night ]; then [ -n "$out" ] || usage; night "$v" "$out" "$root" "$rev"
    else [ -n "$nf" ] || usage; release "$v" "$nf" "$root" "$rev"; fi
}
[ "${BPR_SOURCED:-0}" = 1 ] || main "$@"
