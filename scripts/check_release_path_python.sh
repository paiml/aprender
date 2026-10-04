#!/usr/bin/env bash
# check_release_path_python.sh -- Python on the release path: the inventory, and a guard that keeps it
# from growing (BLD-002 R12).
# WHY       The release path starts Python in many places: the CI section driver, the ladder judge, the
#           CRUX scope, the publish universe and more. Moving them off Python is ticketed work, and until
#           it lands no new Python may join the path. Nothing listed those places, so nothing could tell a
#           new one from an old one.
# THE PATH  Every tracked file reached from the ENTRIES below by a reference the release machinery
#           follows: a file path in a script, a workflow or a Makefile recipe; a make target; a Python
#           import of a module that sits next to the importer or in scripts/lib/. Files are read at a git
#           revision, never from the work tree.
# KINDS     python: a .py file, or a file whose first line is a #! line naming python. code: .sh .bash
#           .mk, a .yml or .yaml under .github/ or ci/ (the CI definitions), a Makefile target, or a file
#           with any other #! line; its lines are scanned.
#           data: anything else, every other YAML file (contracts, roadmaps, fixtures) included: a script
#           reads it, nothing runs it, so it is on the path but never scanned.
# MEASURES  python files on the path, by path and blob; interpreter lines (a code line, not a comment,
#           that names python, python3, python3.N, pytest, pip, pip3, pipx, uv or uvx); and references
#           to a .py that the walk cannot find in the tree.
# THE RULE  Base and head are walked in one run, by this script's own entries and scanner. RED when a
#           Python file joins the path (one that only moved, byte for byte, has not joined), when the
#           path's interpreter lines rise in number, when references to a .py the walk cannot find rise
#           in number, or when an entry is gone at head.
# NOT READ  The clean-room job (another repository runs it) and the GPU-host ladder and CRUX legs (host
#           scripts, not files in this repository). Every inventory prints them as not read.
# EXIT      0 GREEN, or the inventory printed; 1 RED, with a line for every finding; 2 not measured
#           (the tree, a blob or the scanner failed: never GREEN); 3 caller error
# USAGE     check_release_path_python.sh --inventory [--rev REV] [--repo DIR]
#           check_release_path_python.sh --check --base REV [--head REV] [--repo DIR]
#           check_release_path_python.sh --selftest | --mutants | -h
set -uo pipefail
export LC_ALL=C

PROG="${0##*/}"
SCRIPT_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/${BASH_SOURCE[0]##*/}"
WORK=''

# step(s) of the release (BLD-002 R0's step list) | the entry: a file, a directory (every tracked file
# under it), or Makefile:<target>
ENTRIES='S01       scripts/bump-version.sh
S01-S13   scripts/release/
S02-S03   .github/workflows/ci.yml
S05-S07   scripts/model_ladder.sh
S05-S07   scripts/check_model_ladder.sh
S09       scripts/dogfood.sh
S10       scripts/check_publish_preflight.sh
S12       Makefile:publish
S13       .github/workflows/binary-release.yml
S13       .github/workflows/rc-cut.yml'
NOT_READ='S04       the clean-room job: another repository runs it
S05-S07   the GPU-host ladder and CRUX legs: host scripts, not files in this repository'

caller_error() { printf 'FAIL  RELPY %s: caller error: %s\n' "$PROG" "$*"; exit 3; }
cleanup() { case "$WORK" in ?*/tmp.?*) rm -rf -- "${WORK:?}" ;; esac; }
trap cleanup EXIT
# A signal ends the script through exit, so the EXIT trap still removes the work directory.
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

# One file (or one Makefile target) in, one record per line out:
#   K kind | N lines | I lineno text | T token root-relative referrer-relative bare-name | M target | P module
# (an empty T field prints as ".", which names no file, so bash can split the record on tabs)
# The program reads a whole blob: kind comes from the name and the first line, then only code lines
# (interpreter lines, make targets, path tokens) or python lines (imports, path tokens) are looked at.
read -r -d '' SCAN_AWK <<'AWK'
function trim(s) { sub(/^[ \t]+/, "", s); sub(/[ \t]+$/, "", s); return s }
# PATH -> the path with "./", "//" and ".." folded; "" when ".." climbs above the root
function fold(p,    n, i, a, out, k, j) {
    n = split(p, a, "/"); k = 0
    for (i = 1; i <= n; i++) {
        if (a[i] == "" || a[i] == ".") continue
        if (a[i] == "..") { if (k == 0) return ""; k--; continue }
        out[++k] = a[i]
    }
    p = ""
    for (j = 1; j <= k; j++) p = p (j > 1 ? "/" : "") out[j]
    return p
}
function tokens(s,    t, u, k, n, i, w, c, ws, c1, c2, bare) {
    # $VAR, ${...}, $(...) and ${{ ... }} stand for some directory, so each one becomes "/"
    t = s
    for (k = 0; k < 8; k++) {
        u = t
        gsub(/\$\{\{[^}]*\}\}/, "/", t)
        gsub(/\$\{[^{}]*\}/, "/", t)
        gsub(/\$\([^()]*\)/, "/", t)
        if (t == u) break
    }
    gsub(/\$[A-Za-z_][A-Za-z0-9_]*/, "/", t)
    gsub(/\$[0-9@*#?!$-]/, "/", t)
    n = split(t, ws, /[^A-Za-z0-9_.\/-]+/)
    for (i = 1; i <= n; i++) {
        w = ws[i]
        sub(/[.]+$/, "", w)
        if (w == "" || (w in seen)) continue
        c = w
        while (c ~ /^(\.\/|\/)/) c = (substr(c, 1, 2) == "./") ? substr(c, 3) : substr(c, 2)
        if (w !~ /[A-Za-z0-9_-]\.(sh|bash|py|yml|yaml)$/ && c !~ /^(scripts|ci|\.github)\//) continue
        seen[w] = 1
        c1 = fold(c)
        c2 = (dir == "") ? c1 : fold(dir "/" c)
        bare = (c ~ /\//) ? "" : c
        printf "T\t%s\t%s\t%s\t%s\n", w, (c1 == "" ? "." : c1), (c2 == "" ? "." : c2), (bare == "" ? "." : bare)
    }
}
function maketargets(s,    rest, tail, n, i, w, words, ended) {
    rest = s
    while (match(rest, /(^|[^A-Za-z0-9_.-])(make|\$\(MAKE\)|\$\{MAKE\})[ \t]+/)) {
        tail = substr(rest, RSTART + RLENGTH)
        rest = tail
        n = split(tail, words, /[ \t]+/)
        for (i = 1; i <= n; i++) {
            w = words[i]
            if (w ~ /^-(C|f|-directory|-file|-makefile)/) break
            if (w ~ /^-/ || w ~ /=/ || w ~ /^[0-9]+$/) continue
            ended = (w ~ /(;|&|\||\)|>)$/)
            gsub(/^["'(]+|["');&|>]+$/, "", w)
            if (w ~ /^[a-z][A-Za-z0-9_.-]*$/) printf "M\t%s\n", w
            else break
            if (ended) break
        }
    }
}
function code_line(s, lineno,    t) {
    if (s ~ /^[ \t]*(#|$)/) return
    if (s ~ /(^|[^A-Za-z0-9_.-]|:-)(python(3(\.[0-9]+)?)?|pytest|pip3?|pipx|uvx?)([^A-Za-z0-9_.-]|$)/)
        { t = substr(trim(s), 1, 160); gsub(/\t/, " ", t); printf "I\t%d\t%s\n", lineno, t }
    maketargets(s)
    tokens(s)
}
function imports(s,    m, n, i, a) {
    if (match(s, /^[ \t]*from[ \t]+[.]*[A-Za-z_][A-Za-z0-9_]*/)) {
        m = substr(s, RSTART, RLENGTH); sub(/^[ \t]*from[ \t]+[.]*/, "", m)
        printf "P\t%s\n", m
    } else if (s ~ /^[ \t]*from[ \t]+[.]+[ \t]+import[ \t]/) {
        m = s; sub(/^[ \t]*from[ \t]+[.]+[ \t]+import[ \t]+/, "", m); sub(/#.*/, "", m); gsub(/[()]/, "", m)
        n = split(m, a, ",")
        for (i = 1; i <= n; i++) { sub(/^[ \t]+/, "", a[i]); sub(/[ \t].*$/, "", a[i]); if (a[i] ~ /^[A-Za-z_][A-Za-z0-9_]*$/) printf "P\t%s\n", a[i] }
    } else if (s ~ /^[ \t]*import[ \t]/) {
        m = s; sub(/^[ \t]*import[ \t]+/, "", m); sub(/#.*/, "", m)
        n = split(m, a, ",")
        for (i = 1; i <= n; i++) { sub(/^[ \t]+/, "", a[i]); sub(/[ \t.].*$/, "", a[i]); if (a[i] ~ /^[A-Za-z_][A-Za-z0-9_]*$/) printf "P\t%s\n", a[i] }
    }
}
BEGIN {
    # the caller passes these through the environment: awk -v would make a backslash in a path an escape
    name = ENVIRON["RELPY_NAME"]; dir = ENVIRON["RELPY_DIR"]; target = ENVIRON["RELPY_TARGET"]
    if (target != "") kind = "make"
    else if (name ~ /\.py$/) kind = "python"
    else if (name ~ /\.(sh|bash|mk)$/ || (name ~ /\.(yml|yaml)$/ && name ~ /^(\.github|ci)\//)) kind = "code"
    else kind = ""
    inrec = 0; found = 0
}
NR == 1 && target == "" {
    if ($0 ~ /^#!/ && $0 ~ /(python|uv run)/) kind = "python"
    else if (kind == "" && $0 ~ /^#!/) kind = "code"
    else if (kind == "") { kind = "data"; exit }
}
kind == "python" { if ($0 !~ /^[ \t]*#/) { imports($0); tokens($0) }; next }
kind == "code" { code_line($0, NR); next }
kind == "make" {
    if ($0 ~ /^\t/) {
        if (inrec) { s = substr($0, 2); sub(/^[@+-]+/, "", s); code_line(s, NR) }
        next
    }
    if ($0 ~ /^[ \t]*$/ || $0 ~ /^#/) next
    inrec = 0
    if ($0 ~ /^[^ \t#][^=:]*:([^=]|$)/) {
        i = index($0, ":"); names = substr($0, 1, i - 1); rest = substr($0, i + 1)
        sub(/^:/, "", rest)
        n = split(names, nm, /[ \t]+/)
        for (k = 1; k <= n; k++) if (nm[k] == target) inrec = 1
        if (inrec) {
            found = 1
            recipe = ""
            if (index(rest, ";") > 0) { recipe = substr(rest, index(rest, ";") + 1); rest = substr(rest, 1, index(rest, ";") - 1) }
            sub(/#.*/, "", rest); gsub(/\|/, " ", rest)
            n = split(rest, pr, /[ \t]+/)
            for (k = 1; k <= n; k++) if (pr[k] ~ /^[A-Za-z0-9_.\/-]+$/) printf "M\t%s\n", pr[k]
            if (recipe != "") code_line(recipe, NR)
        }
    }
    next
}
END {
    if (kind == "make" && !found) kind = "none"
    if (kind == "") kind = "data"
    printf "K\t%s\n", kind
    printf "N\t%d\n", NR
}
AWK

# Base and head walk files in, run from the work directory; the findings, then one verdict line, out.
# Exit 1 on any RED. Lists are kept in file order, so the output is the same on every run.
read -r -d '' CMP_AWK <<'AWK'
FILENAME == "base/py.tsv" { bn++; bord[bn] = $1; bpy[$1] = $2; bsum += $3; next }
FILENAME == "head/py.tsv" { hn++; hord[hn] = $1; hpy[$1] = $2; hpl[$1] = $3; hvia[$1] = $4; hsum += $3; next }
FILENAME == "base/interp.tsv" { bi++; bnode[$1]++; next }
FILENAME == "head/interp.tsv" { hi++; if (!($1 in hnode)) hnord[++hnn] = $1; hnode[$1]++; next }
FILENAME == "base/unres.tsv" { bu++; bun[$1 "\t" $2] = 1; next }
FILENAME == "head/unres.tsv" { hu++; huord[hu] = $1 "\t" $2; next }
FILENAME == "base/entries.tsv" { be[$1 " " $2] = $3; next }
FILENAME == "head/entries.tsv" { hen++; heord[hen] = $1 " " $2; he[$1 " " $2] = $3; next }
END {
    red = 0
    for (i = 1; i <= hn; i++) {
        p = hord[i]
        if (p in bpy) continue
        # a newcomer with the bytes of a base file that is no longer on the path only moved; each base
        # file can be claimed once, so a copy whose source is still on the path has joined
        moved = ""
        for (j = 1; j <= bn; j++) { q = bord[j]; if (!(q in hpy) && !(q in claimed) && bpy[q] == hpy[p]) { moved = q; break } }
        if (moved != "") { claimed[moved] = 1; printf "info  moved: %s -> %s (the same bytes)\n", moved, p }
        else { printf "RED   new Python file on the release path: %s (%d lines; reached via %s)\n", p, hpl[p], hvia[p]; red++ }
    }
    for (j = 1; j <= bn; j++) { q = bord[j]; if (!(q in hpy) && !(q in claimed)) printf "info  left the release path: %s\n", q }
    if (hi > bi) {
        printf "RED   interpreter lines on the release path rose %d -> %d\n", bi, hi; red++
        for (i = 1; i <= hnn; i++) { n = hnord[i]; if (hnode[n] > bnode[n] + 0) printf "        %s %d -> %d\n", n, bnode[n] + 0, hnode[n] }
    } else if (hi < bi) printf "info  interpreter lines on the release path fell %d -> %d\n", bi, hi
    if (hu > bu) {
        printf "RED   references to a .py the walk cannot find rose %d -> %d\n", bu, hu; red++
        for (i = 1; i <= hu; i++) if (!(huord[i] in bun)) { split(huord[i], kv, "\t"); printf "        %s names %s\n", kv[1], kv[2] }
    }
    for (i = 1; i <= hen; i++) {
        k = heord[i]
        if (be[k] == "read" && he[k] != "read") { printf "RED   entry gone at head: %s\n", k; red++ }
        else if (he[k] == "read" && be[k] != "read") printf "info  entry missing at base: %s\n", k
    }
    s = sprintf("(python_files %d->%d, python_lines %d->%d, interp_lines %d->%d, unresolved_py %d->%d)", bn, hn, bsum, hsum, bi, hi, bu, hu)
    if (red) { printf "RED   RELPY %d finding(s) base=%s head=%s %s\n", red, b12, h12, s; exit 1 }
    printf "GREEN RELPY base=%s head=%s: no new Python on the release path %s\n", b12, h12, s
}
AWK

# git in the repository under test; paths print as they are, never C-quoted
g() { git -C "$REPO" -c core.quotepath=off "$@"; }

# NODE REFERRER: put NODE on the walk once; the first file to reach it is the one remembered
enq() {
    if [ -n "${seen[$1]+x}" ]; then return 0; fi
    seen[$1]=1
    via[$1]="$2"
    queue+=("$1")
}

# NODE TOKEN ROOT-RELATIVE REFERRER-RELATIVE BARE-NAME: put the files a path token names on the walk.
# A directory with an action.yml is a local action. A bare name is looked up under scripts/, ci/ and
# .github/ only when the path forms miss. A .py token that names no tracked file is remembered.
resolve() {
    local c f hit=0
    for c in "$3" "$4"; do
        if [ -n "${blob[$c]+x}" ]; then enq "$c" "$1"; hit=1
        elif [ -n "${blob[$c/action.yml]+x}" ]; then enq "$c/action.yml" "$1"; hit=1
        elif [ -n "${blob[$c/action.yaml]+x}" ]; then enq "$c/action.yaml" "$1"; hit=1
        fi
    done
    if [ "$hit" = 0 ] && [ -n "${bybase[$5]+x}" ]; then
        while IFS= read -r f; do
            if [ -n "$f" ]; then enq "$f" "$1"; hit=1; fi
        done <<< "${bybase[$5]}"
    fi
    if [ "$hit" = 0 ]; then case "$2" in *.py) printf '%s\t%s\n' "$1" "$2" >> "$out/unres.tsv" ;; esac; fi
    return 0
}

# NODE DIR MODULE: an import names a module beside the importer or in scripts/lib/; any other module is
# the standard library or an installed package, not a file in this repository
module() {
    local c
    for c in "${2:+$2/}$3.py" "${2:+$2/}$3/__init__.py" "scripts/lib/$3.py" "scripts/lib/$3/__init__.py"; do
        if [ -n "${blob[$c]+x}" ]; then enq "$c" "$1"; return 0; fi
    done
    return 0
}

# REV OUT: walk the release path at REV into OUT: entries.tsv (step entry read|missing), nodes.tsv
# (node kind via), py.tsv (node blob lines via), interp.tsv (node line text) and unres.tsv (node token).
# Returns 2, having said why, when the tree, a blob or the scanner could not be read.
walk() {
    local rev="$1" out="$2" meta path step entry node name target d b kind nlines tag f1 f2 f3 f4 p state qi=0
    local -A blob=() bybase=() seen=() via=() nkind=() dirn=()
    local -a queue=() order=()
    if ! mkdir -p -- "$out"; then printf 'FAIL  RELPY not measured: cannot make %s\n' "$out"; return 2; fi
    : > "$out/entries.tsv"; : > "$out/nodes.tsv"; : > "$out/py.tsv"; : > "$out/interp.tsv"; : > "$out/unres.tsv"
    if ! g ls-tree -r -z --full-tree "$rev" > "$out/tree.z"; then
        printf 'FAIL  RELPY not measured: git ls-tree %s failed\n' "$rev"; return 2
    fi
    while IFS=$'\t' read -r -d '' meta path; do
        case "$meta" in *' blob '*) ;; *) continue ;; esac
        blob[$path]="${meta##* }"; order+=("$path")
        case "$path" in scripts/*|ci/*|.github/*) bybase[${path##*/}]+="$path"$'\n' ;; esac
    done < "$out/tree.z"
    while read -r step entry; do
        if [ -z "$entry" ]; then continue; fi
        case "$entry" in
            Makefile:*) enq "$entry" "entry $step" ;;
            */) for p in "${order[@]}"; do
                    case "$p" in "$entry"?*) enq "$p" "entry $step"; dirn[$entry]=$(( ${dirn[$entry]:-0} + 1 )) ;; esac
                done ;;
            *) if [ -n "${blob[$entry]+x}" ]; then enq "$entry" "entry $step"; fi ;;
        esac
    done <<< "$ENTRIES"
    while [ "$qi" -lt "${#queue[@]}" ]; do
        node="${queue[$qi]}"; qi=$((qi + 1))
        case "$node" in
            Makefile:*) name=Makefile; target="${node#Makefile:}"; d='' ;;
            */*) name="$node"; target=''; d="${node%/*}" ;;
            *) name="$node"; target=''; d='' ;;
        esac
        b="${blob[$name]:-}"; kind=none; nlines=0
        if [ -n "$b" ]; then
            if ! g cat-file blob "$b" > "$out/cur"; then
                printf 'FAIL  RELPY not measured: cannot read %s (blob %s) at %s\n' "$name" "$b" "$rev"; return 2
            fi
            if ! RELPY_NAME="$name" RELPY_DIR="$d" RELPY_TARGET="$target" awk "$SCAN_AWK" < "$out/cur" > "$out/rec"; then
                printf 'FAIL  RELPY not measured: the scanner failed on %s at %s\n' "$node" "$rev"; return 2
            fi
            while IFS=$'\t' read -r tag f1 f2 f3 f4; do
                case "$tag" in
                    K) kind="$f1" ;;
                    N) nlines="$f1" ;;
                    I) printf '%s\t%s\t%s\n' "$node" "$f1" "$f2" >> "$out/interp.tsv" ;;
                    T) resolve "$node" "$f1" "$f2" "$f3" "$f4" ;;
                    M) enq "Makefile:$f1" "$node"; if [ -n "${blob[$f1]+x}" ]; then enq "$f1" "$node"; fi ;;
                    P) module "$node" "$d" "$f1" ;;
                esac
            done < "$out/rec"
        fi
        nkind[$node]="$kind"
        printf '%s\t%s\t%s\n' "$node" "$kind" "${via[$node]}" >> "$out/nodes.tsv"
        if [ "$kind" = python ]; then printf '%s\t%s\t%s\t%s\n' "$node" "$b" "$nlines" "${via[$node]}" >> "$out/py.tsv"; fi
    done
    while read -r step entry; do
        if [ -z "$entry" ]; then continue; fi
        state=missing
        case "$entry" in
            Makefile:*) if [ "${nkind[$entry]:-none}" = make ]; then state=read; fi ;;
            */) if [ "${dirn[$entry]:-0}" -ge 1 ]; then state=read; fi ;;
            *) if [ -n "${blob[$entry]+x}" ]; then state=read; fi ;;
        esac
        printf '%s\t%s\t%s\n' "$step" "$entry" "$state" >> "$out/entries.tsv"
    done <<< "$ENTRIES"
    return 0
}

# DIR: one line of counts over a walk's files
counts() {
    local nr
    nr="$(printf '%s\n' "$NOT_READ" | awk 'NF { n++ } END { print n + 0 }')"
    (cd -- "$1" && awk -F'\t' -v nr="$nr" '
        FILENAME == "entries.tsv" { e++; if ($3 == "read") r++; else m++ }
        FILENAME == "nodes.tsv" { if ($2 != "none") n++ }
        FILENAME == "py.tsv" { pf++; pl += $3 }
        FILENAME == "interp.tsv" { il++ }
        FILENAME == "unres.tsv" { u++ }
        END { printf "entries=%d read=%d missing=%d not_read=%d nodes=%d python_files=%d python_lines=%d interp_lines=%d unresolved_py=%d\n", e + nr, r, m, nr, n, pf, pl, il, u }
    ' entries.tsv nodes.tsv py.tsv interp.tsv unres.tsv)
}

# make the work directory that the EXIT trap removes
mkwork() {
    WORK="$(mktemp -d)" || { WORK=''; printf 'FAIL  RELPY not measured: mktemp -d failed\n'; exit 2; }
    case "$WORK" in ?*/tmp.?*) ;; *) printf 'FAIL  RELPY not measured: mktemp -d gave %s\n' "$WORK"; WORK=''; exit 2 ;; esac
}

# REV: print the release path at REV, then its counts
inventory() {
    local sha
    sha="$(g rev-parse --verify --quiet "$1^{commit}")" || caller_error "'$1' names no commit in $REPO"
    mkwork
    walk "$sha" "$WORK/at" || exit 2
    printf 'RELPY inventory rev=%s\n' "${sha:0:12}"
    printf 'entries (state  step  entry)\n'
    awk -F'\t' '{ printf "  %-9s %-9s %s\n", ($3 == "read" ? "read" : "MISSING"), $1, $2 }' < "$WORK/at/entries.tsv"
    printf '%s\n' "$NOT_READ" | awk 'NF { s = $1; t = $0; sub(/^[^ ]+ +/, "", t); printf "  %-9s %-9s %s\n", "not read", s, t }'
    printf 'python files (lines  path  via)\n'
    awk -F'\t' '{ printf "  %5d  %s  via %s\n", $3, $1, $4 }' < "$WORK/at/py.tsv"
    printf 'interpreter lines (node:line  text)\n'
    awk -F'\t' '{ printf "  %s:%s  %s\n", $1, $2, $3 }' < "$WORK/at/interp.tsv"
    printf 'references to a .py the walk cannot find (node  token)\n'
    awk -F'\t' '{ printf "  %s  %s\n", $1, $2 }' < "$WORK/at/unres.tsv"
    printf 'RELPY-INVENTORY rev=%s %s\n' "${sha:0:12}" "$(counts "$WORK/at")"
}

# BASE HEAD: walk both in this run, by this script's entries and scanner, and compare
check() {
    local bs hs rc=0
    bs="$(g rev-parse --verify --quiet "$1^{commit}")" || caller_error "'$1' names no commit in $REPO"
    hs="$(g rev-parse --verify --quiet "$2^{commit}")" || caller_error "'$2' names no commit in $REPO"
    mkwork
    walk "$bs" "$WORK/base" || exit 2
    walk "$hs" "$WORK/head" || exit 2
    printf 'RELPY check base=%s head=%s\n' "${bs:0:12}" "${hs:0:12}"
    (cd -- "$WORK" && awk -F'\t' -v b12="${bs:0:12}" -v h12="${hs:0:12}" "$CMP_AWK" base/py.tsv head/py.tsv \
        base/interp.tsv head/interp.tsv base/unres.tsv head/unres.tsv base/entries.tsv head/entries.tsv) || rc=$?
    exit "$rc"
}

selftest_cleanup() { case "${tmp:-}" in ''|/) return 0 ;; *) rm -rf -- "${tmp:?}" ;; esac; }

selftest() {
    local tmp pass=0 fail=0 r b obj
    tmp="$(mktemp -d)"
    case "$tmp" in ?*/tmp.?*) : ;; *) echo "mktemp gave '${tmp:-}', refusing"; return 2 ;; esac
    # SEC011: validate before rm -rf. $tmp is this function's own mktemp -d (checked above).
    trap selftest_cleanup RETURN
    export HOME="$tmp" GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null
    unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_OBJECT_DIRECTORY GIT_ALTERNATE_OBJECT_DIRECTORIES GIT_COMMON_DIR
    r="$tmp/repo"
    fx() { git -C "$r" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t -c commit.gpgsign=false "$@"; }
    put() { mkdir -p -- "$r/$(dirname -- "$1")" && cat > "$r/$1"; }
    swap() { # FILE OLD NEW: replace the first OLD on each line
        OLD="$2" NEW="$3" awk '{ i = index($0, ENVIRON["OLD"]); if (i) $0 = substr($0, 1, i - 1) ENVIRON["NEW"] substr($0, i + length(ENVIRON["OLD"])); print }' "$r/$1" > "$tmp/e" && cat -- "$tmp/e" > "$r/$1"
    }
    drop() { OLD="$2" awk 'index($0, ENVIRON["OLD"]) == 0' "$r/$1" > "$tmp/e" && cat -- "$tmp/e" > "$r/$1"; }
    variant() { fx checkout -q -b "$1" main; }
    commit() { fx add -A && fx commit -q -m "$1"; }
    row() { # name expect-rc needle -- script-args ... ; a needle that starts with ! must NOT be said, and
        # one that starts with = is the whole output
        local name="$1" expect="$2" needle="$3" o rc=0
        shift 3
        if [ "${1:-}" = -- ]; then shift; fi
        o="$(timeout 20 bash "$SCRIPT_PATH" "$@" < /dev/null 2>&1)" || rc=$?
        if [ "$rc" != "$expect" ]; then printf '  BROKE %-66s expected exit %s got %s\n%s\n' "$name" "$expect" "$rc" "$o"; fail=$((fail + 1)); return 0; fi
        case "$needle" in
            '!'*)
                case "$o" in
                    *"${needle#!}"*) printf '  BROKE %-66s exit %s but said: %s\n%s\n' "$name" "$rc" "${needle#!}" "$o"; fail=$((fail + 1)) ;;
                    *) printf '  ok    %-66s exit=%s\n' "$name" "$rc"; pass=$((pass + 1)) ;;
                esac ;;
            '='*)
                if [ "$o" = "${needle#=}" ]; then printf '  ok    %-66s exit=%s\n' "$name" "$rc"; pass=$((pass + 1))
                else printf '  BROKE %-66s exit %s but did not say exactly: %s\n%s\n' "$name" "$rc" "${needle#=}" "$o"; fail=$((fail + 1)); fi ;;
            *)
                case "$o" in
                    *"$needle"*) printf '  ok    %-66s exit=%s\n' "$name" "$rc"; pass=$((pass + 1)) ;;
                    *) printf '  BROKE %-66s exit %s but never said: %s\n%s\n' "$name" "$rc" "$needle" "$o"; fail=$((fail + 1)) ;;
                esac ;;
        esac
    }

    # The fixture: every entry, a script reached through a variable path, a module imported beside its
    # importer, a module run by its bare name from a make recipe, a workflow step, a data file and a
    # comment that both say python3, and a script off the path that runs Python.
    git init -q -b main "$r" || return 2
    printf '#!/usr/bin/env bash\necho %s\n' bump | put scripts/bump-version.sh
    put scripts/release/autopilot.sh <<'FX'
#!/usr/bin/env bash
. "$REPO_ROOT/scripts/release/lib.sh"
bash scripts/dogfood.sh
make publish
python3 -c 'print(1)'
FX
    printf '# python3 is not run from here\ntrue\n' | put scripts/release/lib.sh
    printf 'python3 scripts/tools/offpath.py\n' | put scripts/release/publish-order.txt
    printf '#!/usr/bin/env bash\necho %s\n' dogfood | put scripts/dogfood.sh
    printf '#!/usr/bin/env bash\necho %s\n' preflight | put scripts/check_publish_preflight.sh
    printf '#!/usr/bin/env bash\necho %s\n' ladder | put scripts/model_ladder.sh
    put scripts/check_model_ladder.sh <<'FX'
#!/usr/bin/env bash
HERE="$(cd "$(dirname "$0")" && pwd)"
python3 "$HERE/lib/judge.py"
FX
    printf 'import helper\nprint("judge")\n' | put scripts/lib/judge.py
    printf 'X = 1\n' | put scripts/lib/helper.py
    printf 'print("python3 is here")\n' | put scripts/lib/universe.py
    printf 'publish: prep\n\tcd scripts/lib && python3 universe.py\nprep:\n\t@echo prep\ndanger:\n\tpython3 scripts/tools/offpath.py\n' | put Makefile
    put .github/workflows/ci.yml <<'FX'
name: ci
on: [push]
jobs:
  gate:
    runs-on: any
    steps:
      - run: bash scripts/ci/run.sh
FX
    printf '#!/usr/bin/env bash\necho %s\n' run | put scripts/ci/run.sh
    printf 'name: %s\n' binary-release | put .github/workflows/binary-release.yml
    printf 'name: %s\n' rc-cut | put .github/workflows/rc-cut.yml
    printf '#!/usr/bin/env bash\npython3 scripts/tools/offpath.py\n' | put scripts/tools/offpath.sh
    printf 'print("off")\n' | put scripts/tools/offpath.py
    commit base || return 2

    # Each variant is a branch off main with one change.
    variant v-newpy && printf 'print("new")\n' | put scripts/lib/newtool.py &&
        printf '"$HERE/lib/newtool.py" --run\n' >> "$r/scripts/check_model_ladder.sh" && commit v-newpy || return 2
    variant v-inline && printf 'python3 -m json.tool < /dev/null\n' >> "$r/scripts/dogfood.sh" && commit v-inline || return 2
    variant v-reach-offpath && printf 'bash scripts/tools/offpath.sh\n' >> "$r/scripts/dogfood.sh" && commit v-reach-offpath || return 2
    variant v-fell && drop scripts/release/autopilot.sh 'python3 -c' && commit v-fell || return 2
    variant v-offpath-py && printf 'python3 -V\n' >> "$r/scripts/tools/offpath.sh" && commit v-offpath-py || return 2
    variant v-comment && printf '# python3 used to run here\n  # and python3 here\n' >> "$r/scripts/release/autopilot.sh" &&
        commit v-comment || return 2
    variant v-rename-entry && fx mv scripts/dogfood.sh scripts/dogfood2.sh &&
        swap scripts/release/autopilot.sh scripts/dogfood.sh scripts/dogfood2.sh && commit v-rename-entry || return 2
    variant v-pyvar && printf '"${PY:-python3}" scripts/lib/universe.py\n' >> "$r/scripts/dogfood.sh" && commit v-pyvar || return 2
    variant v-sibling-import && printf 'import extra\n' >> "$r/scripts/lib/judge.py" && printf 'Z = 3\n' | put scripts/lib/extra.py &&
        commit v-sibling-import || return 2
    variant v-make-recursion &&
        printf 'publish: prep\n\tcd scripts/lib && python3 universe.py\n\t$(MAKE) sub\nprep:\n\t@echo prep\nsub:\n\tpython3 scripts/tools/offpath.py\ndanger:\n\tpython3 scripts/tools/offpath.py\n' | put Makefile &&
        commit v-make-recursion || return 2
    variant v-move && fx mv scripts/lib/helper.py scripts/lib/helper2.py && swap scripts/lib/judge.py 'import helper' 'import helper2' &&
        commit v-move || return 2
    variant v-shebang && printf '#!/usr/bin/env python3\nprint(1)\n' | put scripts/release/newtool && commit v-shebang || return 2
    variant v-swap && fx rm -q scripts/lib/helper.py && swap scripts/lib/judge.py 'import helper' 'import other' &&
        printf 'Y = 2\n' | put scripts/lib/other.py && commit v-swap || return 2
    variant v-copy && cp -- "$r/scripts/lib/helper.py" "$r/scripts/lib/helper_copy.py" &&
        printf 'import helper_copy\n' >> "$r/scripts/lib/judge.py" && commit v-copy || return 2
    variant v-unres && printf '"$HERE/lib/ghost.py" --run\n' >> "$r/scripts/check_model_ladder.sh" && commit v-unres || return 2
    variant v-action && printf '      - uses: ./.github/actions/setup\n' >> "$r/.github/workflows/ci.yml" &&
        printf 'name: setup\nruns:\n  using: composite\n  steps:\n    - run: python3 -m pip --version\n      shell: bash\n' |
        put .github/actions/setup/action.yml && commit v-action || return 2
    variant v-make-C && printf 'make -C docs danger\n' >> "$r/scripts/dogfood.sh" && commit v-make-C || return 2
    variant v-uv && printf 'uv tool install check-jsonschema\n' >> "$r/scripts/dogfood.sh" &&
        printf '      - uses: astral-sh/setup-uv@v6\n' >> "$r/.github/workflows/ci.yml" && commit v-uv || return 2
    variant v-data-yaml && printf 'pv validate contracts/demo.yaml\n' >> "$r/scripts/release/autopilot.sh" &&
        printf 'id: demo\nrun: python3 scripts/tools/offpath.py\n' | put contracts/demo.yaml && commit v-data-yaml || return 2
    variant v-ci-yaml && printf '      - run: bash scripts/ci/run.sh ci/sections.yml\n' >> "$r/.github/workflows/ci.yml" &&
        printf 'gate:\n  run: python3 -m pytest\n' | put ci/sections.yml && commit v-ci-yaml || return 2
    # a copy of the repository with one blob of main deleted: nothing in it can be measured
    cp -a -- "$r" "$tmp/broken" || return 2
    b="$(git -C "$tmp/broken" rev-parse main:scripts/dogfood.sh)" || return 2
    obj="$tmp/broken/.git/objects/${b:0:2}/${b:2}"
    rm -f -- "${obj:?}"

    local -a at=(--repo "$r")
    row 'inventory: the counts of the fixture path' 0 ' entries=12 read=10 missing=0 not_read=2 nodes=17 python_files=3 python_lines=4 interp_lines=3 unresolved_py=0' -- --inventory --rev main "${at[@]}"
    row 'inventory: a module run by its bare name from a make recipe' 0 'scripts/lib/universe.py  via Makefile:publish' -- --inventory --rev main "${at[@]}"
    row 'inventory: a script path relative to the script that names it' 0 'scripts/lib/judge.py  via scripts/check_model_ladder.sh' -- --inventory --rev main "${at[@]}"
    row 'inventory: a module imported beside its importer' 0 'scripts/lib/helper.py  via scripts/lib/judge.py' -- --inventory --rev main "${at[@]}"
    row 'inventory: a data file is on the path and never scanned' 0 '!offpath' -- --inventory --rev main "${at[@]}"
    row 'inventory: a comment that names python is not an interpreter line' 0 '!scripts/release/lib.sh:' -- --inventory --rev main "${at[@]}"
    row 'inventory: a python string in a .py is not an interpreter line' 0 '!universe.py:' -- --inventory --rev main "${at[@]}"
    row 'inventory: the steps no file in this repository runs' 0 'not read  S04       the clean-room job' -- --inventory --rev main "${at[@]}"
    row 'inventory: an entry the tree does not have is printed' 0 'MISSING   S09       scripts/dogfood.sh' -- --inventory --rev v-rename-entry "${at[@]}"
    row 'check: a tree against itself is GREEN' 0 'GREEN RELPY' -- --check --base main --head main "${at[@]}"
    row 'FALSIFIER: a new .py run by its path, the word python never said' 1 'RED   new Python file on the release path: scripts/lib/newtool.py' -- --check --base main --head v-newpy "${at[@]}"
    row 'a python line added to a release script' 1 'RED   interpreter lines on the release path rose 3 -> 4' -- --check --base main --head v-inline "${at[@]}"
    row 'a release script now starts a script that runs Python' 1 'RED   new Python file on the release path: scripts/tools/offpath.py' -- --check --base main --head v-reach-offpath "${at[@]}"
    row 'a python line removed is GREEN, and said' 0 'info  interpreter lines on the release path fell 3 -> 2' -- --check --base main --head v-fell "${at[@]}"
    row 'python added to a script off the path is GREEN' 0 'interp_lines 3->3, unresolved_py 0->0)' -- --check --base main --head v-offpath-py "${at[@]}"
    row 'comments that name python are GREEN' 0 'interp_lines 3->3, unresolved_py 0->0)' -- --check --base main --head v-comment "${at[@]}"
    row 'an entry renamed away is RED' 1 'RED   entry gone at head: S09 scripts/dogfood.sh' -- --check --base main --head v-rename-entry "${at[@]}"
    row 'an entry that appears at head is GREEN, and said' 0 'info  entry missing at base: S09 scripts/dogfood.sh' -- --check --base v-rename-entry --head main "${at[@]}"
    row 'the interpreter behind a default, ${PY:-python3}' 1 'RED   interpreter lines on the release path rose 3 -> 4' -- --check --base main --head v-pyvar "${at[@]}"
    row 'a module imported beside a release .py' 1 'RED   new Python file on the release path: scripts/lib/extra.py' -- --check --base main --head v-sibling-import "${at[@]}"
    row 'a make target reached by $(MAKE) from a release target' 1 'RED   new Python file on the release path: scripts/tools/offpath.py' -- --check --base main --head v-make-recursion "${at[@]}"
    row 'a .py that moved, byte for byte, is GREEN' 0 'info  moved: scripts/lib/helper.py -> scripts/lib/helper2.py' -- --check --base main --head v-move "${at[@]}"
    row 'an extensionless file with a python #! line' 1 'RED   new Python file on the release path: scripts/release/newtool' -- --check --base main --head v-shebang "${at[@]}"
    row 'a .py swapped for one with other bytes is RED' 1 'RED   new Python file on the release path: scripts/lib/other.py' -- --check --base main --head v-swap "${at[@]}"
    row 'a copy whose source is still on the path is RED' 1 'RED   new Python file on the release path: scripts/lib/helper_copy.py' -- --check --base main --head v-copy "${at[@]}"
    row 'a reference to a .py that is not tracked' 1 'RED   references to a .py the walk cannot find rose 0 -> 1' -- --check --base main --head v-unres "${at[@]}"
    row 'a local action reached from a workflow step' 1 '.github/actions/setup/action.yml 0 -> 1' -- --check --base main --head v-action "${at[@]}"
    row 'make -C runs another Makefile: no root target is followed' 0 'interp_lines 3->3, unresolved_py 0->0)' -- --check --base main --head v-make-C "${at[@]}"
    row 'a YAML file a release script reads is data, never scanned' 0 'python_files 3->3, python_lines 4->4, interp_lines 3->3, unresolved_py 0->0)' -- --check --base main --head v-data-yaml "${at[@]}"
    row 'a CI definition under ci/ is scanned like a workflow' 1 'ci/sections.yml 0 -> 1' -- --check --base main --head v-ci-yaml "${at[@]}"
    row 'uv is an interpreter word; setup-uv is not' 1 'RED   interpreter lines on the release path rose 3 -> 4' -- --check --base main --head v-uv "${at[@]}"
    row 'a blob the walk cannot read: not measured, never GREEN' 2 'FAIL  RELPY not measured: cannot read scripts/dogfood.sh' -- --inventory --rev main --repo "$tmp/broken"
    row 'a check that cannot read a blob: not measured' 2 'not measured' -- --check --base main --head main --repo "$tmp/broken"
    row 'caller: --check without --base' 3 '--check needs --base' -- --check "${at[@]}"
    row 'caller: a rev that names no commit' 3 'names no commit' -- --inventory --rev no-such-rev "${at[@]}"
    row 'caller: an unknown argument' 3 'unknown argument' -- --frobnicate
    row 'caller: --repo that is not a repository' 3 'is not a git repository' -- --inventory --repo "$tmp/none"
    row 'caller: two commands' 3 'two commands' -- --inventory --check
    row 'caller: --rev with --check' 3 '--rev goes with --inventory' -- --check --base main --rev main "${at[@]}"
    row 'caller: an option with no value' 3 '--base needs a value' -- --check --base
    row 'caller: no command' 3 'no command' --
    printf -- '--- %s/%s rows ---\n' "$pass" "$((pass + fail))"
    [ "$fail" -eq 0 ]
}

mutants() {
    local tmp pass=0 fail=0 name expr copy o rc
    tmp="$(mktemp -d)"
    case "$tmp" in ?*/tmp.?*) : ;; *) echo "mktemp gave '${tmp:-}', refusing"; return 2 ;; esac
    # SEC011: validate before rm -rf. $tmp is this function's own mktemp -d (checked above).
    trap selftest_cleanup RETURN
    while read -r name expr; do
        [ -n "$name" ] || continue
        copy="$tmp/$name.sh"
        sed -e "$expr" "$SCRIPT_PATH" > "$copy"
        if cmp -s "$SCRIPT_PATH" "$copy"; then
            printf '  BROKE %-34s changed nothing: its pattern no longer matches\n' "$name"; fail=$((fail + 1)); continue
        fi
        if ! bash -n "$copy" 2>/dev/null; then
            printf '  BROKE %-34s does not parse: a RED from it would prove nothing\n' "$name"; fail=$((fail + 1)); continue
        fi
        rc=0; o="$(bash "$copy" --selftest < /dev/null 2>&1)" || rc=$?
        case "$rc:$o" in
            0:*) printf '  BROKE %-34s SURVIVED: the case table stayed green\n' "$name"; fail=$((fail + 1)) ;;
            *"syntax error"*|*"command not found"*|*"unbound variable"*) printf '  BROKE %-34s the mutant does not run: a RED from it proves nothing\n' "$name"; fail=$((fail + 1)) ;;
            *"  BROKE "*) printf '  ok    %-34s killed, %s row(s) broke\n' "$name" "$(printf '%s\n' "$o" | awk '/^  BROKE /{n++} END{print n+0}')"; pass=$((pass + 1)) ;;
            *) printf '  BROKE %-34s exit %s with no broken row: not a kill\n%s\n' "$name" "$rc" "$o"; fail=$((fail + 1)) ;;
        esac
    done <<'MUTANTS'
interpreter_word_blind      s/:-)(python(3/:-)(pythonQ(3/
comment_lines_counted       s@if (s ~ /^\[ \\t\]\*(#|\$)/) return@if (0) return@
python_strings_counted      s/{ imports(\$0); tokens(\$0) }/{ imports($0); code_line($0, NR) }/
data_files_scanned          s/{ kind = "data"; exit }/{ kind = "code" }/
shebang_not_read            s@if (\$0 ~ /^#!/ && \$0 ~ /(python|uv run)/) kind = "python"@if (0) kind = "python"@
recipe_never_ends           s/^    inrec = 0$/    inrec = inrec/
make_C_followed             s@if (w ~ /^-(C|f|-directory|-file|-makefile)/) break@if (0) break@
referrer_path_dropped       s/for c in "\$3" "\$4"; do/for c in "$3"; do/
bare_name_dropped           s/if \[ "\$hit" = 0 \] && \[ -n "\${bybase\[\$5\]+x}" \]; then/if false; then/
action_dir_dropped          s@blob\[\$c/action.yml\]+x@blob[$c/action.ymlX]+x@
imports_dropped             s/P) module "\$node" "\$d" "\$f1" ;;/P) : ;;/
make_targets_dropped        s/M) enq "Makefile:\$f1" "\$node";/M) :;/
new_python_never_flagged    s/if (p in bpy) continue/continue/
copy_claims_moved           s/if (!(q in hpy) && !(q in claimed) && /if (/
move_ignores_bytes          s/bpy\[q\] == hpy\[p\]/1/
interp_ratchet_off          s/if (hi > bi) {/if (0) {/
unresolved_ratchet_off      s/if (hu > bu) {/if (0) {/
gone_entry_ignored          s/if (be\[k\] == "read" && he\[k\] != "read")/if (0)/
unread_blob_ignored         s@if ! g cat-file blob "\$b" > "\$out/cur"; then@g cat-file blob "$b" > "$out/cur"; if false; then@
yaml_outside_ci_scanned     s@ && name ~ /^(\\.github|ci)\\//@@
ci_yaml_not_code            s@(\\.github|ci)@(\\.github)@
uv_word_blind               s@|pipx|uvx?)@|pipx)@
MUTANTS
    printf -- '--- %s/%s mutants killed ---\n' "$pass" "$((pass + fail))"
    [ "$fail" -eq 0 ]
}

REPO=.
cmd='' rev=HEAD base='' head=HEAD o_rev='' o_base='' o_head='' o_repo=''
while [ "$#" -gt 0 ]; do
    case "$1" in
        --inventory|--check|--selftest|--mutants|-h|--help)
            if [ -n "$cmd" ]; then caller_error "two commands: $cmd and $1"; fi
            cmd="$1" ;;
        --rev|--base|--head|--repo)
            if [ "$#" -lt 2 ]; then caller_error "$1 needs a value"; fi
            case "$1" in
                --rev) rev="$2"; o_rev=1 ;;
                --base) base="$2"; o_base=1 ;;
                --head) head="$2"; o_head=1 ;;
                --repo) REPO="$2"; o_repo=1 ;;
            esac
            shift ;;
        *) caller_error "unknown argument '$1' (try -h)" ;;
    esac
    shift
done
case "$cmd" in
    --inventory)
        if [ -n "$o_base$o_head" ]; then caller_error "--base and --head go with --check"; fi
        git -C "$REPO" rev-parse --git-dir > /dev/null 2>&1 || caller_error "--repo '$REPO' is not a git repository"
        inventory "$rev" ;;
    --check)
        if [ -n "$o_rev" ]; then caller_error "--rev goes with --inventory; --check takes --base and --head"; fi
        if [ -z "$o_base" ]; then caller_error "--check needs --base REV"; fi
        git -C "$REPO" rev-parse --git-dir > /dev/null 2>&1 || caller_error "--repo '$REPO' is not a git repository"
        check "$base" "$head" ;;
    '') caller_error "no command (try -h)" ;;
    *)
        if [ -n "$o_rev$o_base$o_head$o_repo" ]; then caller_error "--rev, --base, --head and --repo go with --inventory or --check"; fi
        case "$cmd" in
            --selftest) selftest ;;
            --mutants) mutants ;;
            *) awk 'NR > 1 && /^set -uo pipefail$/ { exit } NR > 1' "$SCRIPT_PATH" ;;
        esac ;;
esac
