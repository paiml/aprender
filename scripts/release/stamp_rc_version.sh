#!/usr/bin/env bash
# stamp_rc_version.sh -- an rc asset's crate version IS its tag: 0.69.3-rc.2, never bare 0.69.3
# (#4110, #4256, #4290)
#
#   bash scripts/release/stamp_rc_version.sh TREE TAG    # rewrite TREE's workspace to TAG's version
#   bash scripts/release/stamp_rc_version.sh --self-test # case table + mutants on a scratch workspace
#
# WHY. The rc is tagged on the commit CI gated, whose Cargo.toml says X.Y.Z. Built as is, every
# v0.69.3-rc.N asset printed `apr 0.69.3 (...)`: rc.1, rc.2 and the final were indistinguishable,
# and every CARGO_PKG_VERSION-keyed record (the F2 forward receipt, #4290) collided across them.
# Operator 2026-09-24: "we need actual version numbers" / "version number needs release canidate
# info in it". binary-release.yml runs this on the checked-out tag tree before `cargo build
# --locked`, so CARGO_PKG_VERSION itself is X.Y.Z-rc.N.
#
# WHAT IS REWRITTEN, and only when the workspace version is exactly TAG's X.Y.Z:
#   - every workspace member manifest (and the root): `version = "X.Y.Z"` on its own line;
#   - every dependency line that names a `path =` and a version requirement "X.Y.Z" / "=X.Y.Z" /
#     "^X.Y.Z" / "~X.Y.Z": a plain "0.69.3" requirement does NOT match 0.69.3-rc.2 (semver: a
#     pre-release only satisfies a requirement naming that same pre-release), so the pins move too;
#   - Cargo.lock `[[package]]` entries at X.Y.Z WITHOUT a `source` (workspace crates only; a
#     registry crate that happens to share the number keeps it).
# A final tag (vX.Y.Z) is a no-op. A tag whose X.Y.Z is not the workspace version is refused:
# stamping it would publish a binary claiming a version its source never was.
# Cargo itself is the oracle: the build that follows runs `--locked`, and the self-test runs
# `cargo metadata --locked --offline` on the stamped scratch workspace.
#
# EXIT 0 stamped (or a final tag: nothing to do) · 1 refused · 2 usage / unreadable tree.
set -uo pipefail
PROG=stamp_rc_version

# stamp(): bash + POSIX awk, no python3 (#4352). Parity with the python it replaced is
# the table in #4352; every rule below names the python behaviour it keeps.

# sr_repr S -> S as python's repr() prints a str (quotes, backslash, \n \t \r).
sr_repr() {
    local s=$1 q="'"
    s=${s//\\/\\\\}
    case $s in *"'"*) case $s in *'"'*) s=${s//\'/\\\'} ;; *) q='"' ;; esac ;; esac
    s=${s//$'\n'/\\n}; s=${s//$'\t'/\\t}; s=${s//$'\r'/\\r}
    printf '%s%s%s' "$q" "$s" "$q"
}

# sr_read FILE -> SR_TEXT = its text as python's text mode reads it (\r\n and \r -> \n), or
# rc 1 with SR_ERR = the OSError python would print.
sr_read() {
    local f=$1
    if [ -d "$f" ]; then SR_ERR="[Errno 21] Is a directory: $(sr_repr "$f")"; return 1; fi
    if [ ! -e "$f" ]; then SR_ERR="[Errno 2] No such file or directory: $(sr_repr "$f")"; return 1; fi
    if [ ! -r "$f" ]; then SR_ERR="[Errno 13] Permission denied: $(sr_repr "$f")"; return 1; fi
    SR_TEXT=$(cat -- "$f" && printf x) || { SR_ERR="cannot read $(sr_repr "$f")"; return 1; }
    SR_TEXT=${SR_TEXT%x}
    case $SR_TEXT in *$'\r'*) SR_TEXT=${SR_TEXT//$'\r\n'/$'\n'}; SR_TEXT=${SR_TEXT//$'\r'/$'\n'} ;; esac
    return 0
}

# sr_write FILE AWKOUT TEXT_IT_CAME_FROM: awk ends every line with \n; drop the last one when
# the source had no final newline, so the file keeps its shape (python writes s back as-is).
sr_write() {
    local out=$2
    case $3 in '' | *$'\n') ;; *) out=${out%$'\n'} ;; esac
    printf '%s' "$out" > "$1"
}

# sr_normpath P -> posixpath.normpath(P)
sr_normpath() {
    local p=$1 init='' c r='' n=0
    local -a out=()
    case $p in '') echo .; return ;; ///*) init=/ ;; //*) init=// ;; /*) init=/ ;; esac
    local IFS=/
    set -f
    for c in $p; do
        case $c in '' | .) continue ;; esac
        if [ "$c" != .. ] || { [ -z "$init" ] && [ "$n" -eq 0 ]; } || { [ "$n" -gt 0 ] && [ "${out[n-1]}" = .. ]; }; then
            out[n]=$c; n=$((n + 1))
        elif [ "$n" -gt 0 ]; then
            n=$((n - 1)); unset "out[$n]"
        fi
    done
    set +f
    [ "$n" -gt 0 ] && r="${out[*]}"
    r=$init$r
    echo "${r:-.}"
}

# The root's [workspace.package] version: a `version = "..."` line at column 0 after the header
# and before the next line that starts with `[`; a later header is searched when one misses.
SR_AWK_WS='
/^\[workspace\.package\][[:space:]]*$/ { ins = 1; next }
ins && /^\[/ { ins = 0 }
ins && match($0, /^version[[:space:]]*=[[:space:]]*"[^"]+"/) {
    v = substr($0, 1, RLENGTH); sub(/^[^"]*"/, "", v); sub(/"$/, "", v); print v; exit
}'

# `KEY = [ ... ]`: `#` to end of line dropped FIRST, then up to the first `]`, then every
# double-quoted string, scanned as re.findall(r'"([^"]+)"') does. The python cut at the first
# `]` even inside a comment: #4219's `# ... every [[bin]]` ended `members` after 3 entries and
# the rc tree failed `cargo metadata` (aprender-core "^0.69.0" vs 0.69.0-rc.1). Single quotes are
# NOT read -- the python did not read them either (#4352 notes the latent defect).
SR_AWK_LIST='
{ t = t "\n" $0 }
END {
    if (!match(t, "\n" key "[[:space:]]*=[[:space:]]*\\[")) exit
    n = split(substr(t, RSTART + RLENGTH), L, "\n"); body = ""
    for (j = 1; j <= n; j++) { x = L[j]; h = index(x, "#"); if (h) x = substr(x, 1, h - 1); body = body (j > 1 ? "\n" : "") x }
    e = index(body, "]"); if (!e) exit
    body = substr(body, 1, e - 1)
    p = 1
    while ((q = index(substr(body, p), "\"")) > 0) {
        o = p + q - 1; c = index(substr(body, o + 1), "\"")
        if (!c) break
        if (c == 1) { p = o + 1; continue }
        print substr(body, o + 1, c - 1); p = o + c + 1
    }
}'

# One manifest, line by line, in the python order own -> dep -> dep2:
#   own   column-0 version = "BASE"                       -> NEW
#   dep   the LAST pin `\bversion\s*=\s*"[=^~]?N(.N){0,2}"` with a `\bpath\s*=` before it
#   dep2  the LAST such pin with a `\bpath\s*=` after its closing quote
# A pin is rewritten only when its dotted parts are a prefix of BASE's; a pin that is not
# still COUNTS (subn counted matches, not changes). Counts go to the file `cf`.
SR_AWK_MANIFEST='
function isw(ch) { return ch ~ /[A-Za-z0-9_]/ }
function kw(s, p, k,   r) {  # 0, or the index just past the `=` of `\bk\s*=` at p
    if (substr(s, p, length(k)) != k) return 0
    if (p > 1 && isw(substr(s, p - 1, 1))) return 0
    r = substr(s, p + length(k))
    if (!match(r, /^[[:space:]]*=/)) return 0
    return p + length(k) + RLENGTH
}
function scan(s,   p, i, pos, e, qo, op) {  # fills V*/NV (valid pins) and minpe/maxps (path=)
    NV = 0; minpe = 0; maxps = 0; p = 1
    while ((i = index(substr(s, p), "path")) > 0) {
        pos = p + i - 1; e = kw(s, pos, "path")
        if (e) { if (!minpe || e < minpe) minpe = e; if (pos > maxps) maxps = pos }
        p = pos + 1
    }
    p = 1
    while ((i = index(substr(s, p), "version")) > 0) {
        pos = p + i - 1; p = pos + 1
        e = kw(s, pos, "version"); if (!e) continue
        if (!match(substr(s, e), /^[[:space:]]*"/)) continue
        qo = e + RLENGTH
        if (!match(substr(s, qo), /^[=^~]?[0-9]+(\.[0-9]+)?(\.[0-9]+)?"/)) continue
        op = (substr(s, qo, 1) ~ /[=^~]/) ? 1 : 0
        NV++; VS[NV] = pos; VN[NV] = qo + op; VL[NV] = RLENGTH - 1 - op; VA[NV] = qo + RLENGTH
    }
}
function pin(s, k,   num, a, b, n, j) {
    num = substr(s, VN[k], VL[k]); n = split(num, a, "."); split(base, b, ".")
    for (j = 1; j <= n; j++) if ((a[j] "") != (b[j] "")) return s
    return substr(s, 1, VN[k] - 1) new substr(s, VN[k] + VL[k])
}
{
    s = $0
    if (match(s, /^version[[:space:]]*=[[:space:]]*"/) && substr(s, RLENGTH + 1, length(base) + 1) == base "\"") {
        s = substr(s, 1, RLENGTH) new substr(s, RLENGTH + 1 + length(base)); nown++
    }
    scan(s); hit = 0
    if (minpe) for (k = NV; k >= 1; k--) if (VS[k] >= minpe) { hit = k; break }
    if (hit) { s = pin(s, hit); ndep++ }  # dep: path= before the pin
    scan(s); hit = 0
    for (k = NV; k >= 1; k--) if (maxps && maxps >= VA[k]) { hit = k; break }
    if (hit) { s = pin(s, hit); ndep++ }  # dep2: path= after the pin
    print s
}
END { print nown + 0, ndep + 0 > cf }'

# Cargo.lock: python split on "\n[[package]]\n" (left to right, non-overlapping) and, in every
# block after the first with no "\nsource = " (a source on the block'"'"'s FIRST line is not seen),
# replaced lines that are exactly version = "BASE". Lines are held one behind so the last line
# is known: a final [[package]] with no newline after it is not a separator.
SR_AWK_LOCK='
function flush(   j, src) {
    if (bi > 0) {
        src = 0
        for (j = 2; j <= bn; j++) if (index(bl[j], "source = ") == 1) src = 1
        if (!src) for (j = 1; j <= bn; j++) if (bl[j] == "version = \"" base "\"") { bl[j] = "version = \"" new "\""; nl++ }
    }
    for (j = 1; j <= bn; j++) print bl[j]
    bn = 0
}
function take(line, islast) {
    k++
    if (line == "[[package]]" && k > 1 && !islast && !prevsep) { flush(); print line; bi++; prevsep = 1; return }
    prevsep = 0; bl[++bn] = line
}
{ if (havep) take(pend, 0); pend = $0; havep = 1 }
END { if (havep) take(pend, nolf); flush(); print nl + 0 > cf }'

stamp() {  # stamp TREE TAG
    local tree=$1 tag=$2 base new rc_n root text have mem exc pat full d f e keep
    local n_own=0 n_dep=0 n_lock=0 a c cf out nolf
    local -x LC_ALL=C
    if [[ $tag =~ ^v([0-9]+\.[0-9]+\.[0-9]+)(-rc\.([0-9]+))?$ ]]; then
        base=${BASH_REMATCH[1]}; rc_n=${BASH_REMATCH[2]}; new=${tag#v}
    else
        echo "refuse: tag $(sr_repr "$tag") is not vX.Y.Z or vX.Y.Z-rc.N"; return 1
    fi
    case $tree in */) root=${tree}Cargo.toml ;; *) root=$tree/Cargo.toml ;; esac
    sr_read "$root" || { echo "usage: $SR_ERR"; return 2; }
    text=$SR_TEXT
    have=$(printf '%s' "$text" | awk "$SR_AWK_WS")
    if [ -z "$have" ]; then echo "usage: no [workspace.package] version in the root Cargo.toml"; return 2; fi
    if [ "$have" = "$new" ]; then echo "ok $new: already stamped"; return 0; fi
    if [ "$have" != "$base" ]; then echo "refuse: tag $tag names $base, the workspace is $have"; return 1; fi
    if [ -z "$rc_n" ]; then echo "ok $new: a final tag, nothing to stamp"; return 0; fi
    local -a manifests=("$root") excluded=() kept=()
    mem=$(printf '%s' "$text" | awk -v key=members "$SR_AWK_LIST")
    exc=$(printf '%s' "$text" | awk -v key=exclude "$SR_AWK_LIST")
    local oldng; oldng=$(shopt -p nullglob)
    shopt -s nullglob
    while IFS= read -r pat; do
        [ -n "$pat" ] || continue
        case $pat in /*) full='' ;; *) case $tree in */) full=$tree ;; *) full=$tree/ ;; esac ;; esac
        local IFS=
        for d in "$full"$pat; do
            case $d in */) f=${d}Cargo.toml ;; *) f=$d/Cargo.toml ;; esac
            [ -f "$f" ] && manifests+=("$f")
        done
        unset IFS
    done <<< "$mem"
    eval "$oldng"
    while IFS= read -r e; do
        [ -n "$e" ] || continue
        case $e in /*) excluded+=("$(sr_normpath "$e")") ;; *) case $tree in */) excluded+=("$(sr_normpath "$tree$e")") ;; *) excluded+=("$(sr_normpath "$tree/$e")") ;; esac ;; esac
    done <<< "$exc"
    for f in "${manifests[@]}"; do
        keep=1
        if [ "${#excluded[@]}" -gt 0 ]; then
            d=$(sr_normpath "$f")
            for e in "${excluded[@]}"; do case $d in "$e"/*) keep=0; break ;; esac; done
        fi
        [ "$keep" = 1 ] && kept+=("$f")
    done
    manifests=("${kept[@]}")
    cf=$(mktemp) || return 1
    for f in "${manifests[@]}"; do
        sr_read "$f" || { echo "$PROG: $SR_ERR" >&2; rm -f -- "$cf"; return 1; }
        out=$(printf '%s' "$SR_TEXT" | awk -v base="$base" -v new="$new" -v cf="$cf" "$SR_AWK_MANIFEST"; printf x)
        read -r a c < "$cf"
        sr_write "$f" "${out%x}" "$SR_TEXT"
        n_own=$((n_own + a)); n_dep=$((n_dep + c))
    done
    case $tree in */) f=${tree}Cargo.lock ;; *) f=$tree/Cargo.lock ;; esac
    if [ -f "$f" ]; then
        sr_read "$f" || { echo "$PROG: $SR_ERR" >&2; rm -f -- "$cf"; return 1; }
        nolf=0; case $SR_TEXT in '' | *$'\n') ;; *) nolf=1 ;; esac
        out=$(printf '%s' "$SR_TEXT" | awk -v base="$base" -v new="$new" -v cf="$cf" -v nolf="$nolf" "$SR_AWK_LOCK"; printf x)
        read -r n_lock < "$cf"
        sr_write "$f" "${out%x}" "$SR_TEXT"
    fi
    rm -f -- "$cf"
    echo "stamped $base -> $new: ${#manifests[@]} manifests, $n_own package versions, $n_dep path pins, $n_lock lock entries"
    [ "$n_own" -gt 0 ]
}

self_test() {
    local d fail=0 got rc row want name
    command -v cargo > /dev/null || { echo "$PROG self-test: needs cargo (it is the oracle)"; return 2; }
    d=$(mktemp -d) || return 2
    # a scratch workspace shaped like the real one: [workspace.package] version, a member that
    # inherits it, one that declares its own, a path dep with a caret pin and one with =, an
    # excluded nested crate, and a lock entry for a REGISTRY crate at the same number. `members`
    # carries a `]` in a comment before its glob, the #4219 shape that cut the list short
    mkws() {
        local w=$1; rm -rf -- "${w:?}"; mkdir -p "$w/crates/a/src" "$w/crates/b/src" "$w/crates/c/src" "$w/crates/d/src" "$w/crates/x/src"
        printf '[workspace]\nmembers = [\n    "crates/a", # every [[bin]] prints the SHA (#4219)\n    "crates/*",\n]\nexclude = [\n    "crates/x", # nested\n]\nresolver = "2"\n\n[workspace.package]\nversion = "0.69.3"\nedition = "2021"\n\n[workspace.dependencies]\nb = { path = "crates/b", version = "0.69.3" }\n' > "$w/Cargo.toml"
        printf '[package]\nname = "a"\nversion.workspace = true\nedition = "2021"\n\n[dependencies]\nb = { workspace = true }\nd = { path = "../d", version = "0.69" }\nc = { version = "=0.69.3", path = "../c" }\n' > "$w/crates/a/Cargo.toml"
        printf '[package]\nname = "b"\nversion = "0.69.3"\nedition = "2021"\n' > "$w/crates/b/Cargo.toml"
        printf '[package]\nname = "c"\nversion = "0.69.3"\nedition = "2021"\n' > "$w/crates/c/Cargo.toml"; : > "$w/crates/c/src/lib.rs"
        printf '[package]\nname = "d"\nversion = "0.69.3"\nedition = "2021"\n' > "$w/crates/d/Cargo.toml"; : > "$w/crates/d/src/lib.rs"
        printf '[package]\nname = "x"\nversion = "0.69.3"\nedition = "2021"\n[workspace]\n' > "$w/crates/x/Cargo.toml"
        : > "$w/crates/a/src/lib.rs"; : > "$w/crates/b/src/lib.rs"; : > "$w/crates/x/src/lib.rs"
        (cd "$w" && cargo generate-lockfile --offline -q 2> /dev/null) || return 1
    }
    # versions WT -> "a=<v> b=<v>" from cargo, resolving --locked; "LOCKED-FAIL" when cargo refuses
    versions() {
        (cd "$1" && cargo metadata --locked --offline --format-version 1 2> /dev/null) \
        | jq -r '[.packages[] | .name + "=" + .version] | sort | join(" ")' 2> /dev/null \
        || echo LOCKED-FAIL
    }
    expect() {  # expect ROW WANT GOT
        if [ "$3" = "$2" ]; then echo "  ok   $1"; else printf '  FAIL %s\n       want: %s\n       got:  %s\n' "$1" "$2" "$3"; fail=1; fi
    }
    echo "$PROG self-test: case table (cargo metadata --locked is the oracle)"
    mkws "$d/w" || { echo "  FAIL cannot build the scratch workspace"; rm -rf -- "${d:?}"; return 2; }
    stamp "$d/w" v0.69.3-rc.2 > "$d/out"; rc=$?
    expect "rc tag stamps and exits 0" 0 "$rc"
    expect "cargo resolves the stamped workspace --locked, every member at the rc" "a=0.69.3-rc.2 b=0.69.3-rc.2 c=0.69.3-rc.2 d=0.69.3-rc.2" "$(versions "$d/w")"
    expect "the excluded nested crate keeps its version" 'version = "0.69.3"' "$(grep '^version' "$d/w/crates/x/Cargo.toml")"
    mkws "$d/w"; printf '\n[[package]]\nname = "zzz-registry"\nversion = "0.69.3"\nsource = "registry+https://github.com/rust-lang/crates.io-index"\n' >> "$d/w/Cargo.lock"
    stamp "$d/w" v0.69.3-rc.2 > /dev/null
    expect "a registry lock entry at the same number is untouched" 'version = "0.69.3"' "$(grep -A1 'name = "zzz-registry"' "$d/w/Cargo.lock" | tail -1)"
    mkws "$d/w"; stamp "$d/w" v0.69.3-rc.2 > /dev/null
    stamp "$d/w" v0.69.3-rc.2 > /dev/null; expect "re-stamping the same rc is idempotent" 0 "$?"
    mkws "$d/w"; stamp "$d/w" v0.69.3 > /dev/null; rc=$?
    expect "a final tag is a no-op (exit 0, workspace unchanged)" "0 a=0.69.3 b=0.69.3 c=0.69.3 d=0.69.3" "$rc $(versions "$d/w")"
    for row in "1 v0.69.4-rc.1 a tag for another version is refused" "1 v0.69.3-beta.1 a non-rc prerelease is refused" "1 0.69.3-rc.1 a tag without v is refused"; do
        mkws "$d/w"; want=${row%% *}; row=${row#* }; name=${row%% *}
        stamp "$d/w" "$name" > /dev/null; expect "${row#* }" "$want" "$?"
    done
    # MUTANTS, built from THIS file: each drops one rewrite, and cargo must refuse the result.
    # Without them the table could pass on a stamper that only renamed the packages.
    local src a1 b1 pre
    src=$(cat -- "${BASH_SOURCE[0]}"; printf x); src=${src%x}; pre=${src%%$'\n'"self_test() {"*}
    for row in 'lock|bl[j] = "version = \"" new "\""; nl++|nl++' \
               'path-first pins|{ s = pin(s, hit); ndep++ }  # dep:|{ ndep++ }  # dep:' \
               'version-first pins|{ s = pin(s, hit); ndep++ }  # dep2:|{ ndep++ }  # dep2:' \
               'partial (X.Y) pins|[0-9]+(\.[0-9]+)?(\.[0-9]+)?"/|[0-9]+\.[0-9]+\.[0-9]+"/' \
               'members comments (#4219)|h = index(x, "#"); if (h) x = substr(x, 1, h - 1); |'; do
        IFS='|' read -r name a1 b1 <<< "$row"
        case $pre in *"$a1"*) ;; *) echo "  FAIL mutant $name: anchor moved, re-anchor it"; fail=1; continue ;; esac
        printf '%s' "${src/"$a1"/"$b1"}" > "$d/mut.sh"
        mkws "$d/w"; bash "$d/mut.sh" "$d/w" v0.69.3-rc.2 > /dev/null
        got=$(versions "$d/w")
        case "$got" in *LOCKED-FAIL*) echo "  ok   mutant ($name rewrite dropped): cargo refuses the tree";;
                       *) echo "  FAIL mutant ($name rewrite dropped) survived: $got"; fail=1;; esac
    done
    rm -rf -- "${d:?}"
    if [ "$fail" -eq 0 ]; then echo "$PROG self-test: PASS"; return 0; fi
    echo "$PROG self-test: FAIL"; return 1
}

case "${1:-}" in
    --self-test) self_test ;;
    -h|--help) sed -n '2,28p' "${BASH_SOURCE[0]}" ;;
    *) if [ "$#" -ne 2 ] || [ ! -d "$1" ]; then echo "$PROG: usage: TREE TAG | --self-test" >&2; exit 2; fi
       stamp "$1" "$2" ;;
esac
