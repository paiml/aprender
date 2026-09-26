#!/usr/bin/env bash
# check_bin_names_aprender.sh -- every shipped [[bin]] is named aprender-* (#4430).
#
# WHY. The 0.70.0 all-binaries gate ships every workspace [[bin]] (the nightly
# derives the list from the workspace metadata, #4189). Before #4430, eleven of them
# carried pre-monorepo names -- alimentar, presentar, ptop, score, simular,
# trueno-rag, trueno-zram, verificar, apr-qa, apr-qa-readme-sync,
# apr-corpus-ingest -- so a fleet box's PATH mixed three naming schemes, and
# `score` and `ptop` claimed generic names that other tools also use.
#
# THE RULE. A bin target, in the root workspace OR the `exclude`d facade
# workspace (crates/facades), passes only if it is:
#   - `apr` or `pv`, the two user-facing names (apr-mono-binary-rule-v1); or
#   - aprender-<word>[-<word>...], lowercase; or
#   - listed with its package in scripts/bin_names_pending_fold.txt, meaning it
#     is being folded into `apr` and will not be renamed.
# A pending-fold row whose bin is gone is RED too, so that list only shrinks.
#
# Bins are read from the Cargo.toml manifests with the toolchain's own target
# inference, never from a grep of [[bin]]: an auto-discovered src/bin/*.rs has
# no stanza (verificar was one). Explicit [[bin]]s, autobins (src/main.rs as
# the package name, src/bin/<x>.rs, src/bin/<x>/main.rs, minus any name or path
# an explicit bin claims), `autobins = false`, and workspace member globs minus
# `exclude` are all modelled. The manifests are read by an awk reader of the
# TOML subset they use; a form it does not model (an inline `bin = [...]`
# array) is ENV (2), never a silent skip. No interpreter and no toolchain: the
# fleet declares python out of guards (the sovereign-ci image has none), and
# needing no toolchain keeps this guard in the cargo-free subset that
# guard_tree.sh runs in the guard-tree job, so it is wired by existing.
#
#   bash scripts/check_bin_names_aprender.sh              # check the tree
#   bash scripts/check_bin_names_aprender.sh --list       # the shipped bin set (asset list)
#   bash scripts/check_bin_names_aprender.sh --self-test  # case table + mutants
#   bash scripts/check_bin_names_aprender.sh --check-tree PENDING ROOT   # check another tree
#
# Exit: 0 pass · 1 a bin is misnamed, a row is stale, or the scan is vacuous · 2 ENV.
set -uo pipefail
export LC_ALL=C
shopt -s nullglob
SELF="${BASH_SOURCE[0]}"
REPO_ROOT="$(cd "$(dirname "$SELF")/.." && pwd)"
PENDING="${REPO_ROOT}/scripts/bin_names_pending_fold.txt"
NAME_RE='^aprender-[a-z0-9]+(-[a-z0-9]+)*$'
FS=$'\037'

# toml_scan <Cargo.toml> -- the TOML subset target inference needs, one record
# per line, fields split by \037: W (the file has [workspace]), M <member>,
# X <exclude>, P <package name>, A <autobins value>, B <bin name> <bin path>
# (either may be empty), U <line> (a form this reader does not model).
toml_scan() {
    awk -v q="'" -v fs="$FS" '
    function strip(s,   o, i, c, inq) {
        o = ""; inq = ""
        for (i = 1; i <= length(s); i++) {
            c = substr(s, i, 1)
            if (inq == "") { if (c == "#") break; if (c == "\"" || c == q) inq = c }
            else if (c == inq) inq = ""
            o = o c
        }
        return o
    }
    function first(s) {
        if (match(s, "\"[^\"]*\"|" q "[^" q "]*" q)) return substr(s, RSTART + 1, RLENGTH - 2)
        return ""
    }
    function all(tag, s) {
        while (match(s, "\"[^\"]*\"|" q "[^" q "]*" q)) {
            print tag fs substr(s, RSTART + 1, RLENGTH - 2)
            s = substr(s, RSTART + RLENGTH)
        }
    }
    function flush() { if (inbin) print "B" fs bn fs bp; inbin = 0 }
    {
        line = strip($0)
        if (arr != "") {
            buf = buf " " line
            if (index(line, "]")) { all(arr, buf); arr = "" }
            next
        }
        if (line ~ /^[ \t]*\[/) {
            flush()
            h = line; gsub(/[][ \t]/, "", h)
            if (line ~ /^[ \t]*\[\[/) {
                sec = "[[" h "]]"
                if (h == "bin") { inbin = 1; bn = ""; bp = "" }
            } else {
                sec = h
                if (h == "workspace") print "W"
            }
            next
        }
        if (line !~ /^[ \t]*[A-Za-z0-9_."-]+[ \t]*=/) next
        key = line; sub(/[ \t]*=.*/, "", key); gsub(/[ \t]/, "", key)
        val = line; sub(/^[^=]*=[ \t]*/, "", val)
        if (key == "bin" || key ~ /^package\./ || key ~ /^workspace\./) { print "U" fs $0; next }
        if (sec == "package" && key == "name") print "P" fs first(val)
        else if (sec == "package" && key == "autobins") { gsub(/[ \t]/, "", val); print "A" fs val }
        else if (sec == "[[bin]]" && key == "name") bn = first(val)
        else if (sec == "[[bin]]" && key == "path") bp = first(val)
        else if (sec == "workspace" && (key == "members" || key == "exclude")) {
            tag = (key == "members") ? "M" : "X"
            if (val !~ /^\[/) { print "U" fs $0; next }
            if (index(val, "]")) all(tag, val)
            else { arr = tag; buf = val }
        }
    }
    END { flush(); if (arr != "") print "U" fs "unterminated array" }' "$1"
}

normpath() { realpath -ms -- "$1"; }

# package_bins <dir> <label> -- `label<FS>bin<FS>package<FS>manifest` per bin target.
package_bins() {
    local d=$1 label=$2 man="$1/Cargo.toml" rec tag f1 f2 name='' auto=true e bn rel p taken
    local -a names=() paths=() cands=()
    [ -f "$man" ] || { printf 'ENV: %s: no such manifest\n' "$man" >&2; return 2; }
    rec=$(toml_scan "$man") || { printf 'ENV: %s: unreadable\n' "$man" >&2; return 2; }
    while IFS="$FS" read -r tag f1 f2; do
        case "$tag" in
            U) printf 'ENV: %s: a form this reader does not model: %s\n' "$man" "$f1" >&2; return 2 ;;
            P) name=$f1 ;;
            A) [ "$f1" = false ] && auto=false ;;
        esac
    done <<< "$rec"
    grep -q "^P" <<< "$rec" || return 0 # a virtual manifest: no package, no bins
    [ -n "$name" ] || { printf 'ENV: %s: [package] has no name\n' "$man" >&2; return 2; }
    while IFS="$FS" read -r tag f1 f2; do
        [ "$tag" = B ] || continue
        bn=${f1:-$name}
        names+=("$bn")
        paths+=("$(normpath "$d/${f2:-src/bin/$bn.rs}")")
        printf '%s%s%s%s%s%s%s\n' "$label" "$FS" "$bn" "$FS" "$name" "$FS" "$man"
    done <<< "$rec"
    [ "$auto" = true ] || return 0
    cands=("$name${FS}src/main.rs")
    for e in "$d"/src/bin/*; do
        e=${e##*/}
        case "$e" in
            *.rs) cands+=("${e%.rs}${FS}src/bin/$e") ;;
            *) cands+=("$e${FS}src/bin/$e/main.rs") ;;
        esac
    done
    for e in "${cands[@]}"; do
        bn=${e%%"$FS"*}; rel=${e#*"$FS"}
        [ -f "$d/$rel" ] || continue
        taken=0
        for p in "${names[@]}"; do [ "$p" = "$bn" ] && taken=1; done
        for p in "${paths[@]}"; do [ "$p" = "$(normpath "$d/$rel")" ] && taken=1; done
        [ "$taken" = 0 ] && printf '%s%s%s%s%s%s%s\n' "$label" "$FS" "$bn" "$FS" "$name" "$FS" "$man"
    done
    return 0
}

# workspace_bins <ws dir> <label> -- package_bins over the workspace members.
workspace_bins() {
    local ws=$1 label=$2 rec tag f1 f2 m h d x skip
    local -a dirs=() excl=() done_dirs=()
    rec=$(toml_scan "$ws/Cargo.toml" 2> /dev/null) || { printf 'ENV: %s/Cargo.toml: unreadable\n' "$ws" >&2; return 2; }
    grep -q '^W$' <<< "$rec" || { printf 'ENV: %s/Cargo.toml has no [workspace]\n' "$ws" >&2; return 2; }
    while IFS="$FS" read -r tag f1 f2; do
        case "$tag" in
            U) printf 'ENV: %s/Cargo.toml: a form this reader does not model: %s\n' "$ws" "$f1" >&2; return 2 ;;
            X) excl+=("$(normpath "$ws/$f1")") ;;
            M) case "$f1" in
                   *[*?[]*) for h in $ws/$f1; do dirs+=("$(normpath "$h")"); done ;;
                   *) dirs+=("$(normpath "$ws/$f1")") ;;
               esac ;;
        esac
    done <<< "$rec"
    [ "${#dirs[@]}" -gt 0 ] || { printf 'ENV: %s/Cargo.toml: [workspace] has no members\n' "$ws" >&2; return 2; }
    for d in "${dirs[@]}"; do
        skip=0
        for x in "${excl[@]}" "${done_dirs[@]}"; do [ "$x" = "$d" ] && skip=1; done
        [ "$skip" = 1 ] && continue
        done_dirs+=("$d")
        package_bins "$d" "$label" || return 2
    done
    return 0
}

# collect <root> -- every bin target of the root workspace and the facade
# workspace, each workspace announced by a `SCAN<FS><label>` line so that a
# scanned workspace with no bins still counts as scanned.
collect() {
    workspace_bins "$1" root || return 2
    printf 'SCAN%sroot\n' "$FS"
    if [ -f "$1/crates/facades/Cargo.toml" ]; then
        workspace_bins "$1/crates/facades" facades || return 2
        printf 'SCAN%sfacades\n' "$FS"
    fi
    return 0
}

# check_tree <pending-fold file> <root> -- 0 ok, 1 red, 2 ENV.
check_tree() {
    local pend_file=$1 root=$2 recs line n=0 bad=0 nbins=0 label name pkg man
    local -A pend=() seen=() names=() labels=()
    local -a w
    [ -f "$pend_file" ] || { printf 'ENV: %s: no such pending-fold file\n' "$pend_file"; return 2; }
    while IFS= read -r line || [ -n "$line" ]; do
        n=$((n + 1))
        line=${line%%#*}
        read -r -a w <<< "$line"
        [ "${#w[@]}" = 0 ] && continue
        [ "${#w[@]}" = 2 ] || { printf 'ENV: %s:%s: want `<bin> <package>`, got %s\n' "$pend_file" "$n" "${w[*]}"; return 2; }
        pend[${w[0]}]=${w[1]}
    done < "$pend_file"
    recs=$(collect "$root" 2>&1) || { printf '%s\n' "$recs"; return 2; }
    while IFS="$FS" read -r label name pkg man; do
        [ -n "$label" ] || continue
        if [ "$label" = SCAN ]; then
            labels[$name]=1
            continue
        fi
        nbins=$((nbins + 1))
        names[$name]=1
        seen[$name]="${seen[$name]:-} $pkg "
    done <<< "$recs"
    if [ -z "${labels[facades]:-}" ]; then
        line="${!labels[*]}"
        printf 'W1 facade workspace not in the scan (scanned: %s)\n' "${line:-none}"; bad=1
    fi
    if [ -z "${names[apr]:-}" ]; then
        printf 'W2 %s bin target(s) found and none is `apr`: the scan is not this tree\n' "$nbins"; bad=1
    fi
    while IFS="$FS" read -r label name pkg man; do
        [ -n "$label" ] && [ "$label" != SCAN ] || continue
        if [ "$name" = apr ] || [ "$name" = pv ] || [[ $name =~ $NAME_RE ]]; then continue; fi
        if [ "${pend[$name]:-}" = "$pkg" ]; then
            printf 'ok   `%s` (%s) is pending fold into `apr`, not a rename\n' "$name" "$pkg"; continue
        fi
        printf 'N  `%s` (%s, %s) is not aprender-*: rename it, or list it in the pending-fold file if it is being folded into apr\n' "$name" "$pkg" "$man"
        bad=1
    done <<< "$recs"
    for name in "${!pend[@]}"; do
        case "${seen[$name]:-}" in
            *" ${pend[$name]} "*) ;;
            *) printf 'P  pending-fold row `%s %s` names no bin in the tree: the fold landed or the row is wrong -- delete the row\n' "$name" "${pend[$name]}"; bad=1 ;;
        esac
    done
    [ "$bad" = 0 ] && printf 'ok   %s bin name(s) across %s workspace(s): every one is aprender-*, apr/pv, or pending fold (%s)\n' \
        "${#names[@]}" "${#labels[@]}" "${#pend[@]}"
    return "$bad"
}

# list_tree <root> -- the shipped bin set, one name per line.
list_tree() {
    local recs out
    recs=$(collect "$1") || return 2
    out=$(grep -v "^SCAN$FS" <<< "$recs" | cut -d "$FS" -f2 | sort -u | grep .) || { echo 'ENV: no bin targets found' >&2; return 2; }
    printf '%s\n' "$out"
}

self_test() {
    local d n=0 fail=0 chk=$SELF
    d=$(mktemp -d) || return 2
    # shellcheck disable=SC2064
    trap "rm -rf \"${d:?}\"" RETURN
    printf 'apr-qa aprender-qa-cli\n' > "$d/pend"
    : > "$d/pend0"

    # ws <dir> <bin:package>... [-- <bin:package>...] -- a root workspace with
    # one explicit [[bin]] per spec, and (after `--`) a facade workspace.
    ws() {
        local t=$1 s pk sep='' fac=0 root='' facs=''
        shift
        mkdir -p "$t"
        for s in "$@"; do
            if [ "$s" = -- ]; then fac=1; continue; fi
            pk=${s#*:}
            if [ "$fac" = 1 ]; then
                mkdir -p "$t/crates/facades/$pk"; facs="$facs\"$pk\", "
                printf '[[bin]]\nname = "%s"\npath = "src/%s.rs"\n' "${s%%:*}" "${s%%:*}" >> "$t/crates/facades/$pk/Cargo.toml.bins"
            else
                mkdir -p "$t/crates/$pk"; root="$root$sep\"crates/$pk\""; sep=', '
                printf '[[bin]]\nname = "%s"\npath = "src/%s.rs"\n' "${s%%:*}" "${s%%:*}" >> "$t/crates/$pk/Cargo.toml.bins"
            fi
        done
        printf '[workspace]\nmembers = [%s]\nexclude = ["crates/facades"]\n' "$root" > "$t/Cargo.toml"
        [ "$fac" = 1 ] && printf '[workspace]\nmembers = [%s]\n' "${facs%, }" > "$t/crates/facades/Cargo.toml"
        for s in "$t"/crates/*/Cargo.toml.bins "$t"/crates/facades/*/Cargo.toml.bins; do
            pk=${s%/Cargo.toml.bins}
            { printf '[package]\nname = "%s"\nautobins = false\n' "${pk##*/}"; cat "$s"; } > "$pk/Cargo.toml"
        done
    }
    ws "$d/good" apr:apr-cli apr:aprender pv:aprender-contracts-cli aprender-data:aprender-data apr-qa:aprender-qa-cli -- pv:provable-contracts-cli
    ws "$d/rogue" apr:apr-cli alimentar:aprender-data apr-qa:aprender-qa-cli -- pv:provable-contracts-cli
    ws "$d/aprfoo" apr:apr-cli apr-foo:apr-cli apr-qa:aprender-qa-cli -- pv:provable-contracts-cli
    ws "$d/bare" apr:apr-cli aprender-:x apr-qa:aprender-qa-cli -- pv:provable-contracts-cli
    ws "$d/nohyph" apr:apr-cli aprenderx:x apr-qa:aprender-qa-cli -- pv:provable-contracts-cli
    ws "$d/upper" apr:apr-cli Aprender-Data:x apr-qa:aprender-qa-cli -- pv:provable-contracts-cli
    ws "$d/moved" apr:apr-cli apr-qa:some-other-crate -- pv:provable-contracts-cli
    ws "$d/nofold" apr:apr-cli aprender-data:aprender-data -- pv:provable-contracts-cli
    ws "$d/facrogue" apr:apr-cli apr-qa:aprender-qa-cli -- pv:provable-contracts-cli trueno-rag:facade
    ws "$d/nofac" apr:apr-cli aprender-data:aprender-data apr-qa:aprender-qa-cli
    ws "$d/noapr" aprender-data:aprender-data apr-qa:aprender-qa-cli -- pv:provable-contracts-cli
    ws "$d/facempty" apr:apr-cli aprender-data:aprender-data apr-qa:aprender-qa-cli
    mkdir -p "$d/facempty/crates/facades/pc"; printf '[workspace]\nmembers = ["pc"]\n' > "$d/facempty/crates/facades/Cargo.toml"
    printf '[package]\nname = "pc"\n' > "$d/facempty/crates/facades/pc/Cargo.toml"
    ws "$d/inline" apr:apr-cli apr-qa:aprender-qa-cli -- pv:provable-contracts-cli
    printf 'bin = [{ name = "hidden" }]\n' >> "$d/inline/crates/apr-cli/Cargo.toml"
    ws "$d/nomanifest" apr:apr-cli apr-qa:aprender-qa-cli -- pv:provable-contracts-cli
    sed -i 's|members = \[|members = ["crates/ghost", |' "$d/nomanifest/Cargo.toml"

    # tree <dir> -- manifests read the way the toolchain infers targets: root
    # (members, one excluded), facades, an auto-discovered src/bin/*.rs, a
    # src/bin/<x>/main.rs, an explicit [[bin]] that claims an inferred path, and
    # autobins = false; members and a comment span several lines.
    tree() {
        local t=$1
        mkdir -p "$t/src/bin" "$t/crates/a/src/bin/aprender-dir" "$t/crates/b/src/bin" \
            "$t/crates/c/src/bin" "$t/crates/x/src" "$t/crates/facades/pc/src/bin"
        printf '[package]\nname = "aprender"\n[package.metadata.x]\nname = "not-a-package"\n[workspace]\nmembers = [\n    ".",\n    # a comment, "crates/ghost"\n    "crates/a", "crates/b",\n    '"'crates/c'"',\n]\nexclude = ["crates/x", "crates/facades"]\n' > "$t/Cargo.toml"
        : > "$t/src/bin/apr.rs"
        printf '[package]\nname = "a"\n' > "$t/crates/a/Cargo.toml"
        : > "$t/crates/a/src/bin/aprender-a.rs"; : > "$t/crates/a/src/bin/aprender-dir/main.rs"
        printf '[package]\nname = "b"\n\n[[bin]]\nname = "aprender-b" # was old\npath = "src/bin/old.rs"\n' > "$t/crates/b/Cargo.toml"
        : > "$t/crates/b/src/bin/old.rs"
        printf '[package]\nname = "c"\nautobins = false\n' > "$t/crates/c/Cargo.toml"
        : > "$t/crates/c/src/bin/rogue.rs"
        printf '[package]\nname = "x"\n' > "$t/crates/x/Cargo.toml"; : > "$t/crates/x/src/main.rs"
        printf '[workspace]\nmembers = ["pc"]\n' > "$t/crates/facades/Cargo.toml"
        printf '[package]\nname = "pc"\n' > "$t/crates/facades/pc/Cargo.toml"; : > "$t/crates/facades/pc/src/bin/pv.rs"
    }
    tree "$d/t_good"
    tree "$d/t_auto"; : > "$d/t_auto/crates/a/src/bin/trueno-rag.rs"
    tree "$d/t_dir"; mkdir -p "$d/t_dir/crates/a/src/bin/simular"; : > "$d/t_dir/crates/a/src/bin/simular/main.rs"
    tree "$d/t_main"; : > "$d/t_main/crates/a/src/main.rs"
    tree "$d/t_fac"; : > "$d/t_fac/crates/facades/pc/src/bin/presentar.rs"
    tree "$d/t_claim"; : > "$d/t_claim/crates/b/src/bin/aprender-b.rs"; mv "$d/t_claim/crates/b/src/bin/old.rs" "$d/t_claim/crates/b/src/bin/not-old.rs"
    tree "$d/t_glob"; sed -i '/"crates\/a", "crates\/b",/d; s|^    '"'crates/c'"',|    "crates/*",|' "$d/t_glob/Cargo.toml"
    tree "$d/t_globx"; sed -i '/"crates\/a", "crates\/b",/d; s|^    '"'crates/c'"',|    "crates/*",|; s|"crates/x", ||' "$d/t_globx/Cargo.toml"
    tree "$d/t_nows"; printf '[package]\nname = "aprender"\n' > "$d/t_nows/Cargo.toml"

    row() { # row <want rc> <needle|-> <label> <pending> <root>
        local want=$1 needle=$2 label=$3 out rc
        n=$((n + 1))
        out=$(bash "$chk" --check-tree "$4" "$5" 2>&1); rc=$?
        if [ "$rc" != "$want" ]; then
            printf 'FAIL %-66s rc=%s want %s\n%s\n' "$label" "$rc" "$want" "$out"; fail=1
        elif [ "$needle" != - ] && ! grep -qF -- "$needle" <<< "$out"; then
            printf 'FAIL %-66s did not name %s\n%s\n' "$label" "$needle" "$out"; fail=1
        else printf 'ok   %s\n' "$label"; fi
    }
    rows() {
        row 0 '4 bin name(s) across 2' 'aprender-*, apr, pv and a pending-fold bin pass' "$d/pend" "$d/good"
        row 1 '`alimentar`' 'a pre-monorepo name (alimentar) is RED' "$d/pend" "$d/rogue"
        row 1 '`apr-foo`' 'apr-<x> is not the apr exemption' "$d/pend" "$d/aprfoo"
        row 1 '`aprender-`' 'a bare `aprender-` is RED' "$d/pend" "$d/bare"
        row 1 '`aprenderx`' 'aprender without the hyphen is RED' "$d/pend" "$d/nohyph"
        row 1 '`Aprender-Data`' 'upper case is RED' "$d/pend" "$d/upper"
        row 1 '`apr-qa` (some-other-crate,' 'a pending-fold name in ANOTHER package is RED' "$d/pend" "$d/moved"
        row 1 'P  pending-fold row `apr-qa' 'a stale pending-fold row (fold landed) is RED' "$d/pend" "$d/nofold"
        row 1 '`apr-qa`' 'without its pending-fold row, apr-qa is RED' "$d/pend0" "$d/good"
        row 1 '`trueno-rag`' 'a rogue bin in the FACADE workspace is RED' "$d/pend" "$d/facrogue"
        row 1 'W1 facade' 'a scan without the facade workspace is RED' "$d/pend" "$d/nofac"
        row 0 'across 2 workspace(s)' 'a facade workspace with no bins still counts as scanned' "$d/pend" "$d/facempty"
        row 1 'W2' 'a scan with no `apr` bin is RED (not this tree)' "$d/pend" "$d/noapr"
        row 2 'ENV' 'an inline `bin = [...]` is ENV (2), never skipped' "$d/pend" "$d/inline"
        row 2 'ENV' 'a member with no manifest is ENV (2)' "$d/pend" "$d/nomanifest"
        row 2 'ENV' 'a missing pending-fold file is ENV (2)' "$d/absent" "$d/good"
        row 0 '5 bin name(s)' 'tree: explicit, auto, dir, apr, pv; excluded + autobins=false unseen' "$d/pend0" "$d/t_good"
        row 1 '`trueno-rag`' 'tree: an auto-discovered src/bin/*.rs is RED' "$d/pend0" "$d/t_auto"
        row 1 '`simular`' 'tree: an auto-discovered src/bin/<x>/main.rs is RED' "$d/pend0" "$d/t_dir"
        row 1 '`a`' 'tree: src/main.rs ships as the package name -- RED' "$d/pend0" "$d/t_main"
        row 1 '`presentar`' 'tree: a rogue bin in the facade workspace is RED' "$d/pend0" "$d/t_fac"
        row 1 '`not-old`' 'tree: an explicit bin claims its NAME, not every src/bin file' "$d/pend0" "$d/t_claim"
        row 0 '5 bin name(s)' 'tree: a glob member list honours exclude' "$d/pend0" "$d/t_glob"
        row 1 '`x`' 'tree: the same glob without the exclude scans x -- RED' "$d/pend0" "$d/t_globx"
        row 2 'ENV' 'tree: a root with no [workspace] is ENV (2)' "$d/pend0" "$d/t_nows"
    }
    rows

    # MUTANTS of this script's reader and checker: each must turn a row red.
    local m got
    while IFS= read -r m; do
        n=$((n + 1))
        sed "$m" "$SELF" > "$d/mut.sh"
        if cmp -s "$SELF" "$d/mut.sh"; then printf 'FAIL mutant did not apply: %s\n' "$m"; fail=1; continue; fi
        got=$(chk="$d/mut.sh"; rows 2>&1)
        if grep -q '^FAIL' <<< "$got"; then printf 'ok   mutant killed: %s\n' "$m"
        else printf 'FAIL mutant SURVIVED: %s\n' "$m"; fail=1; fi
    done <<'MUTANTS'
s/\[ "\$name" = apr \] || \[ "\$name" = pv \] || \[\[ \$name =~ \$NAME_RE \]\]/true/
s/^NAME_RE=.*/NAME_RE='^apr'/
s/if \[ "\${pend\[\$name\]:-}" = "\$pkg" \]; then/if [ -n "${pend[$name]:-}" ]; then/
s/\*) printf 'P  pending-fold row/*) : printf 'P  pending-fold row/
s/if \[ -z "\${labels\[facades\]:-}" \]; then/if false; then/
s/for x in "\${excl\[@\]}" "\${done_dirs\[@\]}"; do/for x in "${done_dirs[@]}"; do/
s/A) \[ "\$f1" = false \] \&\& auto=false ;;/A) ;;/
s/\*) cands+=("\$e\${FS}src\/bin\/\$e\/main.rs") ;;/*) ;;/
s/cands=("\$name\${FS}src\/main.rs")/cands=()/
s/for p in "\${paths\[@\]}"; do \[ "\$p" = "\$(normpath "\$d\/\$rel")" \] \&\& taken=1; done/:/
s/\*\[\*?\[\]\*) for h in \$ws\/\$f1; do dirs+=("\$(normpath "\$h")"); done ;;/*[*?[]*) dirs+=("$(normpath "$ws\/$f1")") ;;/
s/workspace_bins "\$1\/crates\/facades" facades || return 2/:/
s/printf 'SCAN%sfacades\\n' "\$FS"/:/
s/U) printf 'ENV: %s: a form this reader does not model: %s\\n' "\$man" "\$f1" >\&2; return 2 ;;/U) ;;/
s/if (inq == "") { if (c == "#") break;/if (inq == "") {/
s/if (index(line, "\]")) { all(arr, buf); arr = "" }/if (index(line, "]")) { arr = "" }/
s/if (h == "workspace") print "W"/if (0) print "W"/
MUTANTS
    if [ "$fail" = 0 ]; then echo "check_bin_names_aprender self-test: ${n}/${n} pass"; return 0; fi
    echo "check_bin_names_aprender self-test: FAIL"; return 1
}

case "${1:-}" in
    -h|--help) sed -n '2,/^set -uo pipefail/p' "$SELF" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    --self-test) self_test; exit $? ;;
    --list) list_tree "$REPO_ROOT"; exit $? ;;
    --check-tree) [ $# = 3 ] || { echo "usage: $0 --check-tree PENDING ROOT" >&2; exit 2; }
        check_tree "$2" "$3"; exit $? ;;
    "") ;;
    *) echo "check_bin_names_aprender: unknown argument $1" >&2; exit 2 ;;
esac

for tool in awk realpath grep sed sort cut; do
    command -v "$tool" > /dev/null 2>&1 || { echo "check_bin_names_aprender: ENV $tool missing" >&2; exit 2; }
done
echo "=== every shipped [[bin]] is aprender-* (check_bin_names_aprender.sh, #4430) ==="
check_tree "$PENDING" "$REPO_ROOT"
