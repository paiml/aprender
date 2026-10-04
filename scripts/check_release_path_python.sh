#!/usr/bin/env bash
# check_release_path_python.sh -- Python on the release path: the inventory, and a guard that keeps it
# from growing (BLD-002 R12).
# WHY       The release path starts Python in many places: the CI section driver, the ladder judge, the
#           CRUX scope, the publish universe and more. Moving them off Python is ticketed work, and until
#           it lands no new Python may join the path. Nothing listed those places, so nothing could tell a
#           new one from an old one.
# THE PATH  Every tracked file reached from the ENTRIES below by a reference the release machinery
#           follows: a file path in a script, a workflow or a Makefile recipe (a name with a known
#           extension, any path with a directory in it, or a bare file name under scripts/, ci/ or
#           .github/; a directory given to an interpreter reaches every .py under it); a make target,
#           from a recipe, a prerequisite or $(MAKE), read from the Makefile that -C DIR, -f FILE or a
#           cd DIR before the make names, and the default goal of a bare make; the lines of a Makefile
#           outside every rule; a local action; a module run with python -m or imported in a python -c
#           program or a heredoc fed to python; a Python import (dotted, relative, or a name after
#           "from X import") of a module that sits next to the importer or in scripts/lib/. Files are
#           read at a git revision, never from the work tree.
# KINDS     python: a .py file, or a file whose first line is a #! line naming python or uv run. code:
#           .sh .bash .mk, a .yml or .yaml under .github/ or ci/ (the CI definitions), an action.yml or
#           action.yaml anywhere, a symlink (its blob is the path it points to), or a file with any
#           other #! line; make: a Makefile target; the lines of code and make nodes are scanned.
#           data: anything else, every other YAML file (contracts, roadmaps, fixtures) included: a script
#           reads it, nothing runs it, so it is on the path but never scanned.
# MEASURES  python files on the path, by path and blob; interpreter uses (in a code line, not a comment:
#           each word python, python3, python3.N, python2, pytest, pip, pip3, pipx, pipenv, uv, uvx,
#           poetry, pdm, hatch, tox, nox, conda, pypy, pypy3, twine, mkdocs, pre-commit, jupyter,
#           ipython, maturin, sphinx-build, flake8, mypy, ruff or virtualenv, and each expansion of a PY,
#           PYTHON, PIP, PYTEST, PYBIN, PYEXE or PYCMD variable, counts once); and references to a .py
#           that the walk cannot find in the tree.
# THE RULE  Base and head are walked in one run, by this script's own entries and scanner. RED when a
#           Python file joins the path (one that only moved, byte for byte, has not joined), when the
#           path's interpreter uses rise in number, when references to a .py the walk cannot find rise
#           in number, or when an entry is gone at head.
# NOT READ  The clean-room job (another repository runs it) and the GPU-host ladder and CRUX legs (host
#           scripts, not files in this repository). Every inventory prints them as not read.
# EXIT      0 GREEN, or the inventory printed; 1 RED, with a line for every finding; 2 not measured
#           (the tree, a blob or the scanner failed, or no entry is read: never GREEN); 3 caller error
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
#   K kind | N lines | I lineno text | T token root-relative referrer-relative bare-name
#   | M target makefile-root-relative makefile-referrer-relative | P module | R path | D module path
# (an empty T field prints as ".", which names no file, so bash can split the record on tabs)
# The program reads a whole blob: kind comes from the name and the first line, then only code lines
# (interpreter uses, make targets, path tokens) or python lines (imports, path tokens) are looked at.
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
function tokens(s, li,    t, u, k, n, i, w, c, ws, c1, c2, bare) {
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
        # a name with a known extension, or any name written with a "/" in it (an extensionless tool, a
        # local action anywhere, "$HERE/tool", "./tool"); a path that names no tracked file is dropped
        # by the caller
        if (w !~ /[A-Za-z0-9_-]\.(sh|bash|mk|py|yml|yaml)$/ && w !~ /\/[A-Za-z0-9_.-]/) continue
        seen[w] = 1
        c1 = fold(c)
        c2 = (dir == "") ? c1 : fold(dir "/" c)
        bare = (c ~ /\//) ? "" : c
        printf "T\t%s\t%s\t%s\t%s\n", w, (c1 == "" ? "." : c1), (c2 == "" ? "." : c2), (bare == "" ? "." : bare)
        # on a line that runs an interpreter, a directory argument (pytest DIR, python3 PKGDIR) runs the
        # Python under it; the directory a cd changes to does not
        if (li && ws[i - 1] != "cd") printf "Y\t%s\t%s\n", (c1 == "" ? "." : c1), (c2 == "" ? "." : c2)
    }
}
# one make target record: TARGET in the Makefile MF, which may be named from the root or from the
# directory of the file that runs make
function mkrec(t, mf,    m1, m2) {
    m1 = fold(mf); m2 = (dir == "") ? m1 : fold(dir "/" mf)
    printf "M\t%s\t%s\t%s\n", t, (m1 == "" ? "." : m1), (m2 == "" ? "." : m2)
}
function maketargets(s,    rest, tail, n, i, w, words, ended, nt, tg, odd, k, cd, mf, v, pre, cdpre, o, c) {
    rest = s
    while (match(rest, /(^|[^A-Za-z0-9_.-])(g?make|\$\(MAKE\)|\$\{MAKE\}|"?\$MAKE"?|"\$\{MAKE\}")([ \t]+|$|\))/)) {
        tail = substr(rest, RSTART + RLENGTH)
        pre = substr(rest, 1, RSTART)
        rest = tail
        # "cd DIR && make" and "cd DIR; make" run make in DIR; the Makefile beside the caller is kept too,
        # since the cd may sit in another subshell of the line
        cdpre = ""
        while (match(pre, /(^|[^A-Za-z0-9_.-])cd[ \t]+[^ \t;&|()]+[ \t]*(&&|;)/)) {
            cdpre = substr(pre, RSTART, RLENGTH); pre = substr(pre, RSTART + RLENGTH)
            sub(/^[^c]*cd[ \t]+/, "", cdpre); sub(/[ \t]*(&&|;)$/, "", cdpre); gsub(/["']/, "", cdpre)
        }
        if (cdpre ~ /\$/) cdpre = ""
        n = split(tail, words, /[ \t]+/)
        # -C DIR and -f FILE, before or after the targets, name the Makefile the targets are read from;
        # with no target, make runs the default goal
        nt = 0; odd = 0; cd = ""; mf = ""
        for (i = 1; i <= n; i++) {
            w = words[i]
            if (w == "") continue
            # an operator or a redirection ends the make command
            if (w ~ /^([0-9]*[<>]|&&|\|\|?|;|&|\))/) break
            # an expansion such as $(nproc) or "$JOBS" is an option value, not a target
            if (w ~ /^["']?\$/) { o = gsub(/\(/, "(", w); c = gsub(/\)/, ")", w); if (c > o || w ~ /[;&|>]["']?$/) break; continue }
            ended = (w ~ /(;|&|\||\)|>)$/)
            if (w ~ /^(-C|--directory|-f|--file|--makefile)$/) { v = words[++i]; ended = (v ~ /(;|&|\||\)|>)$/) }
            else if (w ~ /^(-C|-f)./) v = substr(w, 3)
            else if (w ~ /^--(directory|file|makefile)=/) v = substr(w, index(w, "=") + 1)
            else v = ""
            if (v != "") {
                gsub(/^["']+|["');&|>]+$/, "", v)
                if (w ~ /^-(C|-directory)/) cd = v; else mf = v
                if (ended) break
                continue
            }
            if (w ~ /^-/ || w ~ /=/ || w ~ /^[0-9]+$/) { if (ended) break; continue }
            gsub(/^["'(]+|["');&|>]+$/, "", w)
            if (w ~ /^[A-Za-z0-9_][A-Za-z0-9_.-]*$/) tg[++nt] = w
            else { odd = 1; break }
            if (ended) break
        }
        if (mf == "") mf = (cd != "") ? "Makefile" : (kind == "make" ? name : "Makefile")
        if (cd != "") mf = cd "/" mf
        if (nt == 0 && !odd) tg[++nt] = "<default>"
        for (k = 1; k <= nt; k++) { mkrec(tg[k], mf); if (cdpre != "") mkrec(tg[k], cdpre "/" mf) }
    }
}
# S -> how many interpreter uses S names. Each interpreter word counts once, so a second call on a line
# counts; so does each expansion of an interpreter variable, one whose name is PY, PYTHON, PIP or PYTEST
# (either case), maybe with digits, a prefix ending in _ or a suffix starting with _: $PYTHON, ${PY},
# $(PYTHON), $HOST_PYTHON3, ${PY_BIN:-x}. ${PY:-python3} names both, so it counts twice.
function interp_words(s,    n, t) {
    n = 0
    # one space either side, so the first and the last word each find a boundary
    t = " " s " "
    while (match(t, /([^A-Za-z0-9_.-]|:-|\$\{[A-Za-z0-9_]+[-=?+])(python([0-9]+(\.[0-9]+)?)?|pytest|pip[0-9]?|pipx|pipenv|uvx?|poetry|pdm|hatch|tox|nox|conda|pypy([0-9]+(\.[0-9]+)?)?|twine|mkdocs|pre-commit|jupyter|ipython|maturin|sphinx-build|flake8|mypy|ruff|virtualenv)[^A-Za-z0-9_.-]/)) {
        n++; t = substr(t, RSTART + RLENGTH - 1)
    }
    t = s " "
    while (match(t, /\$[{(]?([A-Za-z0-9]+_)*(PY|PYTHON|PIP|PYTEST|PYBIN|PYEXE|PYCMD|py|python|pip|pytest)[0-9]*(_[A-Za-z0-9_]*)?[^A-Za-z0-9_]/)) {
        n++; t = substr(t, RSTART + RLENGTH)
    }
    return n
}
# the modules a command line hands to Python by name: "-m a.b" runs a/b.py (or its __main__.py), and
# each import in a "-c '...'" program loads a module; one "D a/b" record each
function pymods(s,    t, m, n, i, a, seg, n2, k2, q) {
    t = s
    while (match(t, /(^|[ \t])-m[ \t]+[A-Za-z_][A-Za-z0-9_.]*/)) {
        m = substr(t, RSTART, RLENGTH); t = substr(t, RSTART + RLENGTH)
        sub(/^[ \t]*-m[ \t]+/, "", m); gsub(/\./, "/", m); printf "D\t%s\n", m
    }
    if (!match(s, /(^|[ \t])-c[ \t]+/)) return
    t = substr(s, RSTART + RLENGTH); q = substr(t, 1, 1)
    # a quote the line does not close leaves the program open: the lines up to the closing quote are Python
    if ((q == "\"" || q == "'") && index(substr(t, 2), q) == 0) pyq = q
    gsub(/["']/, " ", t)
    n = split(t, seg, ";")
    for (i = 1; i <= n; i++) {
        m = seg[i]
        if (match(m, /^[ \t]*from[ \t]+[A-Za-z_][A-Za-z0-9_.]*[ \t]+import[ \t]/)) {
            sub(/^[ \t]*from[ \t]+/, "", m); sub(/[ \t].*$/, "", m); gsub(/\./, "/", m); printf "D\t%s\n", m
        } else if (m ~ /^[ \t]*import[ \t]/) {
            sub(/^[ \t]*import[ \t]+/, "", m)
            n2 = split(m, a, ",")
            for (k2 = 1; k2 <= n2; k2++) {
                sub(/^[ \t]+/, "", a[k2]); sub(/[ \t].*$/, "", a[k2])
                if (a[k2] ~ /^[A-Za-z_][A-Za-z0-9_.]*$/) { gsub(/\./, "/", a[k2]); printf "D\t%s\n", a[k2] }
            }
        }
    }
}
function code_line(s, lineno,    t, n, k, m) {
    # the body of a python3 - <<END heredoc, or of a python3 -c " program left open, is Python: its imports
    # are followed until the END line, or the line that closes the quote
    if (pyhd != "") { if (trim(s) == pyhd) pyhd = ""; else pyline(s) }
    else if (pyq != "") { k = index(s, pyq); if (k) { pyline(substr(s, 1, k - 1)); pyq = "" } else pyline(s) }
    if (s ~ /^[ \t]*(#|$)/) return
    n = interp_words(s)
    if (n) { t = substr(trim(s), 1, 160); gsub(/\t/, " ", t); for (k = 1; k <= n; k++) printf "I\t%d\t%s\n", lineno, t }
    maketargets(s)
    tokens(s, n)
    pymods(s)
    if (match(s, /(^|[^A-Za-z0-9_.-])(python([0-9]+(\.[0-9]+)?)?|uv[ \t]+run)([ \t][^<|;&]*)?<<-?[ \t]*["']?[A-Za-z_][A-Za-z0-9_]*/)) {
        m = substr(s, RSTART, RLENGTH); sub(/^.*<<-?[ \t]*["']?/, "", m); pyhd = m
    }
}
# one line of Python: its imports, which may follow a ";" or the colon of try:, else:, if ...: and the like
function pyline(s,    ns, seg, k, x) {
    if (s ~ /^[ \t]*#/) return
    if (impx != "") { imports(s); return }
    ns = split(s, seg, ";")
    for (k = 1; k <= ns; k++) {
        x = seg[k]
        sub(/^[ \t]*(try|else|finally|except[^:]*|if[ \t][^:]*|elif[ \t][^:]*|with[ \t][^:]*|for[ \t][^:]*|while[ \t][^:]*):[ \t]*/, "", x)
        imports(x)
    }
}
# MOD (dotted, a leading dot for each level of a relative import) -> one record per module it can load:
# "P a", "P a/b" for a.b (the package, then the submodule); a relative import resolves against the
# importer's directory here and prints "R <root-relative path>"
function emitmod(m,    nd, i, n, a, p, base) {
    nd = 0; while (substr(m, 1, 1) == ".") { nd++; m = substr(m, 2) }
    if (nd) { base = dir; for (i = 1; i < nd; i++) base = base "/.." }
    n = split(m, a, ".")
    p = ""
    for (i = 1; i <= n; i++) {
        if (a[i] !~ /^[A-Za-z_][A-Za-z0-9_]*$/) return
        p = p (i > 1 ? "/" : "") a[i]
        if (!nd) printf "P\t%s\n", p
        else if (fold(base "/" p) != "") printf "R\t%s\n", fold(base "/" p)
    }
}
# the names after "from X import": each may be a submodule of X
function fromnames(x, m,    n, i, a) {
    sub(/#.*/, "", m); gsub(/[()\\]/, " ", m)
    n = split(m, a, ",")
    for (i = 1; i <= n; i++) {
        sub(/^[ \t]+/, "", a[i]); sub(/[ \t].*$/, "", a[i])
        if (a[i] ~ /^[A-Za-z_][A-Za-z0-9_]*$/) emitmod(x (x ~ /^[.]*$/ ? "" : ".") a[i])
    }
}
function imports(s,    m, x, n, i, a) {
    # the rest of a parenthesised or continued "from X import (" list
    if (impx != "") { m = s; sub(/#.*/, "", m); impx_end =(m ~ /\)/ || (m !~ /\\[ \t]*$/ && impx_paren == 0)); fromnames(impx, m); if (impx_end) impx = ""; return }
    if (match(s, /^[ \t]*from[ \t]+[.]*([A-Za-z_][A-Za-z0-9_.]*)?[ \t]+import[ \t]/)) {
        x = substr(s, RSTART, RLENGTH); sub(/^[ \t]*from[ \t]+/, "", x); sub(/[ \t]+import[ \t]$/, "", x)
        m = substr(s, RSTART + RLENGTH)
        if (x !~ /^[.]*$/) emitmod(x)
        fromnames(x, m)
        sub(/#.*/, "", m)
        if (m ~ /\(/ && m !~ /\)/) { impx = x; impx_paren = 1 }
        else if (m ~ /\\[ \t]*$/) { impx = x; impx_paren = 0 }
    } else if (s ~ /^[ \t]*import[ \t]/) {
        m = s; sub(/^[ \t]*import[ \t]+/, "", m); sub(/#.*/, "", m)
        n = split(m, a, ",")
        for (i = 1; i <= n; i++) { sub(/^[ \t]+/, "", a[i]); sub(/[ \t].*$/, "", a[i]); emitmod(a[i]) }
    }
}
# the prerequisites of a rule: each word that can name a target
function prereqs(r,    n, k, pr) {
    sub(/#.*/, "", r); gsub(/[|\\]/, " ", r)
    n = split(r, pr, /[ \t]+/)
    for (k = 1; k <= n; k++) if (pr[k] ~ /^[A-Za-z0-9_.\/-]+$/) mkrec(pr[k], name)
}
BEGIN {
    # the caller passes these through the environment: awk -v would make a backslash in a path an escape
    name = ENVIRON["RELPY_NAME"]; dir = ENVIRON["RELPY_DIR"]; target = ENVIRON["RELPY_TARGET"]
    if (target == "<all>") kind = "code"
    else if (target != "") kind = "make"
    else if (name ~ /\.py$/) kind = "python"
    else if (name ~ /\.(sh|bash|mk)$/ || (name ~ /\.(yml|yaml)$/ && name ~ /^(\.github|ci)\//)) kind = "code"
    # a local action runs wherever it sits
    else if (name ~ /(^|\/)action\.ya?ml$/) kind = "code"
    # a symlink: its blob is the path it points to, which is followed
    else if (ENVIRON["RELPY_LINK"] != "") kind = "code"
    else kind = ""
    inrec = 0; found = 0; rcont = 0; pcont = 0; impx = ""; gcont = 0; dflt = 0; pyhd = ""; pyq = ""
}
NR == 1 && target == "" {
    if ($0 ~ /^#!/ && $0 ~ /(python|uv run)/) kind = "python"
    else if (kind == "" && $0 ~ /^#!/) kind = "code"
    else if (kind == "") { kind = "data"; exit }
}
# an import may follow a ";" or the colon of try:, else:, if ...: and the like
kind == "python" {
    if ($0 ~ /^[ \t]*#/) next
    pyline($0)
    tokens($0)
    next
}
kind == "code" { code_line($0, NR); next }
# the lines of a Makefile outside every rule (include, variables, $(shell ...), conditionals) run when
# make reads the file, whichever target is asked for: the "<global>" node scans them
kind == "make" && target == "<global>" {
    found = 1
    if (gcont) { gcont = ($0 ~ /\\$/); next }
    if ($0 ~ /^\t/) { gcont = ($0 ~ /\\$/); next }
    if ($0 ~ /^[ \t]*(#|$)/) next
    if ($0 ~ /^[^ \t#][^=:]*:([^=]|$)/) next
    # an included fragment that is not a .mk is read whole, as a .mk is
    if ($0 ~ /^[ \t]*-?s?include[ \t]/) { s = $0; sub(/^[ \t]*-?s?include[ \t]+/, "", s); sub(/#.*/, "", s); n = split(s, inc, /[ \t]+/); for (k = 1; k <= n; k++) if (inc[k] != "" && inc[k] !~ /\.mk$/ && inc[k] !~ /\$/) mkrec("<all>", inc[k]) }
    code_line($0, NR)
    next
}
kind == "make" {
    # a recipe line continued with a backslash goes on, whatever the next line starts with
    if (inrec && rcont) { s = $0; sub(/^\t/, "", s); rcont = (s ~ /\\$/); code_line(s, NR); next }
    # a rule line continued with a backslash: the next line holds more prerequisites
    if (inrec && pcont) { pcont = ($0 ~ /\\$/); prereqs($0); next }
    if ($0 ~ /^\t/) {
        if (inrec) { s = substr($0, 2); sub(/^[@+-]+/, "", s); rcont = (s ~ /\\$/); code_line(s, NR) }
        next
    }
    if ($0 ~ /^[ \t]*$/ || $0 ~ /^#/) next
    # a conditional does not end a recipe: make keeps the lines inside it in the rule
    if ($0 ~ /^[ \t]*(ifdef|ifndef|ifeq|ifneq|else|endif)([ \t(]|$)/) next
    inrec = 0
    if ($0 ~ /^[^ \t#][^=:]*:([^=]|$)/) {
        i = index($0, ":"); names = substr($0, 1, i - 1); rest = substr($0, i + 1)
        sub(/^:/, "", rest)
        n = split(names, nm, /[ \t]+/)
        for (k = 1; k <= n; k++) if (nm[k] == target) inrec = 1
        # the default goal is the first rule whose target is not special (.PHONY and the like) or a pattern
        # it is followed under its own name, so a recipe already on the path is not counted twice
        if (target == "<default>" && !dflt && nm[1] !~ /^[.]/ && nm[1] !~ /%/) { dflt = 1; found = 1; mkrec(nm[1], name); exit }
        if (inrec) {
            found = 1
            recipe = ""
            if (index(rest, ";") > 0) { recipe = substr(rest, index(rest, ";") + 1); rest = substr(rest, 1, index(rest, ";") - 1) }
            else pcont = (rest ~ /\\$/)
            prereqs(rest)
            if (recipe != "") { rcont = (recipe ~ /\\$/); code_line(recipe, NR) }
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
        printf "RED   interpreter uses on the release path rose %d -> %d\n", bi, hi; red++
        for (i = 1; i <= hnn; i++) { n = hnord[i]; if (hnode[n] > bnode[n] + 0) printf "        %s %d -> %d\n", n, bnode[n] + 0, hnode[n] }
    } else if (hi < bi) printf "info  interpreter uses on the release path fell %d -> %d\n", bi, hi
    if (hu > bu) {
        printf "RED   references to a .py the walk cannot find rose %d -> %d\n", bu, hu; red++
        for (i = 1; i <= hu; i++) if (!(huord[i] in bun)) { split(huord[i], kv, "\t"); printf "        %s names %s\n", kv[1], kv[2] }
    }
    for (i = 1; i <= hen; i++) {
        k = heord[i]
        if (be[k] == "read" && he[k] != "read") { printf "RED   entry gone at head: %s\n", k; red++ }
        else if (he[k] == "read" && be[k] != "read") printf "info  entry missing at base: %s\n", k
    }
    s = sprintf("(python_files %d->%d, python_lines %d->%d, interp_uses %d->%d, unresolved_py %d->%d)", bn, hn, bsum, hsum, bi, hi, bu, hu)
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
    if [ "$hit" = 0 ]; then case "$2" in *.py) printf '%s\t%s\n' "$1" "$2" >> "$out/unres.tsv" || werr=1 ;; esac; fi
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

# NODE PATH: a relative import, already resolved against the importer's directory
relmodule() {
    local c
    for c in "$2.py" "$2/__init__.py"; do
        if [ -n "${blob[$c]+x}" ]; then enq "$c" "$1"; return 0; fi
    done
    return 0
}

# NODE DIR PATH: a module Python is handed by name (-m, or an import in a -c program); it is looked
# for from the root, beside the file that names it and in scripts/lib/
runmodule() {
    local c p
    for p in "$3" "${2:+$2/}$3" "scripts/lib/$3"; do
        for c in "$p.py" "$p/__main__.py" "$p/__init__.py"; do
            if [ -n "${blob[$c]+x}" ]; then enq "$c" "$1"; return 0; fi
        done
    done
    return 0
}

# REV OUT: walk the release path at REV into OUT: entries.tsv (step entry read|missing), nodes.tsv
# (node kind via), py.tsv (node blob lines via), interp.tsv (node line text, once per interpreter use) and unres.tsv (node token).
# Returns 2, having said why, when the tree, a blob or the scanner could not be read.
walk() {
    local rev="$1" out="$2" meta path step entry node name target d b kind nlines tag f1 f2 f3 f4 mf p c state qi=0 werr=0 nread=0
    local -A blob=() bybase=() seen=() via=() nkind=() dirn=() pyunder=() link=()
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
        # a symlink is read as the path it points to
        case "$meta" in 120000\ *) link[$path]=1 ;; esac
        # every directory above a .py, for a directory an interpreter is given
        case "$path" in *.py) p="$path"; while [[ "$p" == */* ]]; do p="${p%/*}"; pyunder[$p]+="$path"$'\n'; done ;; esac
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
            *:*) name="${node%:*}"; target="${node##*:}"; d=''; case "$name" in */*) d="${name%/*}" ;; esac
                 if [ "$target" != "<global>" ] && [ "$target" != "<all>" ]; then enq "$name:<global>" "$node"; fi ;;
            */*) name="$node"; target=''; d="${node%/*}" ;;
            *) name="$node"; target=''; d='' ;;
        esac
        b="${blob[$name]:-}"; kind=none; nlines=0
        if [ -n "$b" ]; then
            if ! g cat-file blob "$b" > "$out/cur"; then
                printf 'FAIL  RELPY not measured: cannot read %s (blob %s) at %s\n' "$name" "$b" "$rev"; return 2
            fi
            if ! RELPY_NAME="$name" RELPY_DIR="$d" RELPY_TARGET="$target" RELPY_LINK="${link[$name]:-}" awk "$SCAN_AWK" < "$out/cur" > "$out/rec"; then
                printf 'FAIL  RELPY not measured: the scanner failed on %s at %s\n' "$node" "$rev"; return 2
            fi
            while IFS=$'\t' read -r tag f1 f2 f3 f4; do
                case "$tag" in
                    K) kind="$f1" ;;
                    N) nlines="$f1" ;;
                    I) printf '%s\t%s\t%s\n' "$node" "$f1" "$f2" >> "$out/interp.tsv" || werr=1 ;;
                    T) resolve "$node" "$f1" "$f2" "$f3" "$f4" ;;
                    M) mf="$f2"; if [ -z "${blob[$f2]+x}" ] && [ -n "${blob[$f3]+x}" ]; then mf="$f3"; fi
                       # with no Makefile, make reads a GNUmakefile or a makefile
                       if [ -z "${blob[$mf]+x}" ]; then
                           case "$mf" in
                               Makefile|*/Makefile)
                                   for c in "${mf%Makefile}GNUmakefile" "${mf%Makefile}makefile"; do
                                       if [ -n "${blob[$c]+x}" ]; then mf="$c"; break; fi
                                   done ;;
                           esac
                       fi
                       enq "$mf:$f1" "$node"; if [ -n "${blob[$f1]+x}" ]; then enq "$f1" "$node"; fi ;;
                    P) module "$node" "$d" "$f1" ;;
                    R) relmodule "$node" "$f1" ;;
                    D) runmodule "$node" "$d" "$f1" ;;
                    Y) for c in "$f1" "$f2"; do
                           if [ -z "${blob[$c]+x}" ] && [ -n "${pyunder[$c]+x}" ]; then
                               while IFS= read -r p; do
                                   if [ -n "$p" ]; then enq "$p" "$node"; fi
                               done <<< "${pyunder[$c]}"
                               break
                           fi
                       done ;;
                esac
            done < "$out/rec"
        fi
        nkind[$node]="$kind"
        printf '%s\t%s\t%s\n' "$node" "$kind" "${via[$node]}" >> "$out/nodes.tsv" || werr=1
        if [ "$kind" = python ]; then printf '%s\t%s\t%s\t%s\n' "$node" "$b" "$nlines" "${via[$node]}" >> "$out/py.tsv" || werr=1; fi
    done
    while read -r step entry; do
        if [ -z "$entry" ]; then continue; fi
        state=missing
        case "$entry" in
            Makefile:*) if [ "${nkind[$entry]:-none}" = make ]; then state=read; fi ;;
            */) if [ "${dirn[$entry]:-0}" -ge 1 ]; then state=read; fi ;;
            *) if [ -n "${blob[$entry]+x}" ]; then state=read; fi ;;
        esac
        if [ "$state" = read ]; then nread=$((nread + 1)); fi
        printf '%s\t%s\t%s\n' "$step" "$entry" "$state" >> "$out/entries.tsv" || werr=1
    done <<< "$ENTRIES"
    # a record that could not be written lowers a count, so a failed write is not measured
    if [ "$werr" != 0 ]; then printf 'FAIL  RELPY not measured: cannot write the walk of %s into %s\n' "$rev" "$out"; return 2; fi
    # a revision where no entry is read (another repository, a tree with no release scripts) measured nothing
    if [ "$nread" = 0 ]; then printf 'FAIL  RELPY not measured: no entry of the release path is read at %s\n' "$rev"; return 2; fi
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
        END { printf "entries=%d read=%d missing=%d not_read=%d nodes=%d python_files=%d python_lines=%d interp_uses=%d unresolved_py=%d\n", e + nr, r, m, nr, n, pf, pl, il, u }
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
    printf 'interpreter uses (node:line  text)\n'
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
    # round 1 of the review: each is a way Python could join the path that the walk did not see
    variant v-recipe-cond &&
        printf 'publish: prep\n\tcd scripts/lib && python3 universe.py\nifdef X\n\tpython3 -V\nendif\nprep:\n\t@echo prep\ndanger:\n\tpython3 scripts/tools/offpath.py\n' | put Makefile &&
        commit v-recipe-cond || return 2
    variant v-prereq-cont &&
        printf 'publish: prep \\\n  extra\n\tcd scripts/lib && python3 universe.py\nprep:\n\t@echo prep\nextra:\n\tpython3 -V\ndanger:\n\tpython3 scripts/tools/offpath.py\n' | put Makefile &&
        commit v-prereq-cont || return 2
    variant v-recipe-cont &&
        printf 'publish: prep\n\tcd scripts/lib && python3 universe.py && \\\necho more; python3 -V\nprep:\n\t@echo prep\ndanger:\n\tpython3 scripts/tools/offpath.py\n' | put Makefile &&
        commit v-recipe-cont || return 2
    variant v-make-upper && printf 'make Release\n' >> "$r/scripts/dogfood.sh" &&
        printf 'publish: prep\n\tcd scripts/lib && python3 universe.py\nprep:\n\t@echo prep\nRelease:\n\tpython3 -V\ndanger:\n\tpython3 scripts/tools/offpath.py\n' | put Makefile &&
        commit v-make-upper || return 2
    variant v-make-after && printf 'make danger -C docs\n' >> "$r/scripts/dogfood.sh" && commit v-make-after || return 2
    variant v-extless && printf 'tools/gen --run\n' >> "$r/scripts/dogfood.sh" &&
        printf '#!/usr/bin/env python3\nprint(1)\n' | put tools/gen && commit v-extless || return 2
    variant v-uv-shebang && printf '#!/usr/bin/env -S uv run --script\nprint(1)\n' | put scripts/release/uvtool && commit v-uv-shebang || return 2
    variant v-action-out && printf '      - uses: ./actions/py\n' >> "$r/.github/workflows/ci.yml" &&
        printf 'name: py\nruns:\n  using: composite\n  steps:\n    - run: python3 -V\n      shell: bash\n' |
        put actions/py/action.yml && commit v-action-out || return 2
    variant v-dotted && printf 'from pkg.sub import x\n' >> "$r/scripts/lib/judge.py" && printf '' | put scripts/lib/pkg/__init__.py &&
        printf 'x = 1\n' | put scripts/lib/pkg/sub.py && commit v-dotted || return 2
    variant v-relative && printf 'from . import newmod\n' >> "$r/scripts/lib/judge.py" && printf 'N = 1\n' | put scripts/lib/newmod.py &&
        commit v-relative || return 2
    variant v-paren && printf 'from pkg2 import (\n    a,\n    b,\n)\n' >> "$r/scripts/lib/judge.py" &&
        printf 'B = 1\n' | put scripts/lib/pkg2/b.py && commit v-paren || return 2
    variant v-pyvar2 && printf '"$PYTHON" -m json.tool < x\necho "${PIPESTATUS[0]}" "$COPY" "$PYPI_URL"\n' >> "$r/scripts/dogfood.sh" &&
        commit v-pyvar2 || return 2
    variant v-twice && swap scripts/release/autopilot.sh "python3 -c 'print(1)'" "python3 -c 'print(1)'; python3 -V" && commit v-twice || return 2
    variant v-two-copies && fx mv scripts/lib/helper.py scripts/lib/helper2.py &&
        cp -- "$r/scripts/lib/helper2.py" "$r/scripts/lib/helper3.py" &&
        swap scripts/lib/judge.py 'import helper' 'import helper2, helper3' && commit v-two-copies || return 2
    variant v-cycle && printf '#!/usr/bin/env bash\nbash scripts/release/b.sh\n' | put scripts/release/a.sh &&
        printf '#!/usr/bin/env bash\nbash scripts/release/a.sh\n' | put scripts/release/b.sh && commit v-cycle || return 2
    # round 2 of the review
    variant v-here-tool && printf '"$HERE/ladder-tool" --x\n' >> "$r/scripts/dogfood.sh" &&
        printf '#!/usr/bin/env python3\nprint(1)\n' | put scripts/ladder-tool && commit v-here-tool || return 2
    variant v-make-C-walk && printf 'make -C cookbooks/x test\n' >> "$r/scripts/dogfood.sh" &&
        printf 'test:\n\tpython3 -V\n' | put cookbooks/x/Makefile && commit v-make-C-walk || return 2
    variant v-make-f && printf 'make -f mk/Release.mk ship\n' >> "$r/scripts/dogfood.sh" &&
        printf 'ship:\n\tpython3 -V\n' | put mk/Release.mk && commit v-make-f || return 2
    variant v-include &&
        printf 'include mk/rel.mk\npublish: prep\n\tcd scripts/lib && python3 universe.py\nprep:\n\t@echo prep\ndanger:\n\tpython3 scripts/tools/offpath.py\n' | put Makefile &&
        printf 'rel:\n\tpython3 -V\n' | put mk/rel.mk && commit v-include || return 2
    variant v-shell-assign &&
        printf 'VERSION := $(shell python3 -V)\npublish: prep\n\tcd scripts/lib && python3 universe.py\nprep:\n\t@echo prep\ndanger:\n\tpython3 scripts/tools/offpath.py\n' | put Makefile &&
        commit v-shell-assign || return 2
    variant v-default-goal && printf 'make -j8\n' >> "$r/scripts/dogfood.sh" &&
        printf '.PHONY: all danger\nall:\n\tpython3 -V\npublish: prep\n\tcd scripts/lib && python3 universe.py\nprep:\n\t@echo prep\ndanger:\n\tpython3 scripts/tools/offpath.py\n' | put Makefile &&
        commit v-default-goal || return 2
    variant v-dash-m && printf 'python3 -m scripts.lib.runmod\n' >> "$r/scripts/dogfood.sh" &&
        printf 'R = 1\n' | put scripts/lib/runmod.py && commit v-dash-m || return 2
    variant v-dash-c && printf 'python3 -c "import cmod; cmod.go()"\n' >> "$r/scripts/dogfood.sh" &&
        printf 'C = 1\n' | put scripts/lib/cmod.py && commit v-dash-c || return 2
    variant v-semi-import && printf 'import os; import extra3\n' >> "$r/scripts/lib/judge.py" &&
        printf 'E = 3\n' | put scripts/lib/extra3.py && commit v-semi-import || return 2
    variant v-try-import && printf 'try: import extra4\nexcept ImportError: pass\n' >> "$r/scripts/lib/judge.py" &&
        printf 'E = 4\n' | put scripts/lib/extra4.py && commit v-try-import || return 2
    variant v-comment-paren && printf 'from pkg3 import (\n    a,  # old (legacy)\n    b,\n)\n' >> "$r/scripts/lib/judge.py" &&
        printf 'B = 3\n' | put scripts/lib/pkg3/b.py && commit v-comment-paren || return 2
    variant v-bslash && printf 'from pkg4 import a, \\\n    b\n' >> "$r/scripts/lib/judge.py" &&
        printf 'B = 4\n' | put scripts/lib/pkg4/b.py && commit v-bslash || return 2
    variant v-runners && printf 'twine upload x\n"$PYBIN" t\n' >> "$r/scripts/dogfood.sh" && commit v-runners || return 2
    variant v-action-yaml && printf '      - uses: ./actions/py2\n' >> "$r/.github/workflows/ci.yml" &&
        printf 'name: py2\nruns:\n  using: composite\n  steps:\n    - run: python3 -V\n      shell: bash\n' |
        put actions/py2/action.yaml && commit v-action-yaml || return 2
    variant v-symlink && printf 'L = 1\n' | put scripts/tools/linked.py &&
        ln -s ../tools/linked.py "$r/scripts/release/link.sh" && commit v-symlink || return 2
    variant v-multi-bare && printf 'bash dup.sh\n' >> "$r/scripts/dogfood.sh" &&
        printf 'python3 -V\n' | put scripts/a/dup.sh && printf 'python3 -V\n' | put scripts/b/dup.sh && commit v-multi-bare || return 2
    # round 3 of the review
    variant v-heredoc && printf "python3 - <<'PY'\nfrom hdmod import x\nPY\n" >> "$r/scripts/dogfood.sh" &&
        printf 'x = 1\n' | put scripts/hdmod.py && commit v-heredoc || return 2
    variant v-dash-c-multi && printf 'python3 -c "\nimport cmod2\ncmod2.go()"\n' >> "$r/scripts/dogfood.sh" &&
        printf 'C = 2\n' | put scripts/cmod2.py && commit v-dash-c-multi || return 2
    variant v-make-nproc && printf 'make -j $(nproc) gen\n' >> "$r/scripts/dogfood.sh" &&
        printf 'gen:\n\tpython3 -V\n' >> "$r/Makefile" && commit v-make-nproc || return 2
    variant v-dir-arg && printf 'python3 -m pytest tools/tests\n' >> "$r/scripts/dogfood.sh" &&
        printf 'def test_x():\n    pass\n' | put tools/tests/test_x.py && commit v-dir-arg || return 2
    variant v-cd-make && printf 'cd cookbooks/y && make test\n' >> "$r/scripts/dogfood.sh" &&
        printf 'test:\n\tpython3 -V\n' | put cookbooks/y/Makefile && commit v-cd-make || return 2
    variant v-include-frag &&
        printf 'include Makefile.common\npublish: prep\n\tcd scripts/lib && python3 universe.py\nprep:\n\t@echo prep\ndanger:\n\tpython3 scripts/tools/offpath.py\n' | put Makefile &&
        printf 'V := $(shell python3 -V)\n' | put Makefile.common && commit v-include-frag || return 2
    variant v-gnumakefile && printf 'make -C docs html\n' >> "$r/scripts/dogfood.sh" &&
        printf 'html:\n\tpython3 -V\n' | put docs/GNUmakefile && commit v-gnumakefile || return 2
    variant v-make-andand && printf 'make && echo ok\n' >> "$r/scripts/dogfood.sh" &&
        printf 'all:\n\tpython3 -V\npublish: prep\n\tcd scripts/lib && python3 universe.py\nprep:\n\t@echo prep\ndanger:\n\tpython3 scripts/tools/offpath.py\n' | put Makefile &&
        commit v-make-andand || return 2
    variant v-default-dup && printf 'make\n' >> "$r/scripts/dogfood.sh" && commit v-default-dup || return 2
    variant v-link-extless && printf 'L = 2\n' | put scripts/tools/linked2.py &&
        ln -s ../tools/linked2.py "$r/scripts/release/runner" && commit v-link-extless || return 2
    variant v-noentries && fx rm -r -q scripts .github Makefile && printf 'r\n' | put README && commit v-noentries || return 2
    variant v-words2 && printf 'maturin build\nruff check .\n' >> "$r/scripts/dogfood.sh" && commit v-words2 || return 2
    # the work tree is left dirty: an edit to a release script and an untracked release script, both
    # running Python; every row reads at a revision, so neither may show
    printf 'python3 -V\n' >> "$r/scripts/dogfood.sh" && printf 'python3 -V\n' > "$r/scripts/release/dirty.sh" || return 2
    # a copy of the repository with one blob of main deleted: nothing in it can be measured
    cp -a -- "$r" "$tmp/broken" || return 2
    b="$(git -C "$tmp/broken" rev-parse main:scripts/dogfood.sh)" || return 2
    obj="$tmp/broken/.git/objects/${b:0:2}/${b:2}"
    rm -f -- "${obj:?}"

    local -a at=(--repo "$r")
    row 'inventory: the counts of the fixture path' 0 ' entries=12 read=10 missing=0 not_read=2 nodes=18 python_files=3 python_lines=4 interp_uses=3 unresolved_py=0' -- --inventory --rev main "${at[@]}"
    row 'inventory: a module run by its bare name from a make recipe' 0 'scripts/lib/universe.py  via Makefile:publish' -- --inventory --rev main "${at[@]}"
    row 'inventory: a script path relative to the script that names it' 0 'scripts/lib/judge.py  via scripts/check_model_ladder.sh' -- --inventory --rev main "${at[@]}"
    row 'inventory: a module imported beside its importer' 0 'scripts/lib/helper.py  via scripts/lib/judge.py' -- --inventory --rev main "${at[@]}"
    row 'inventory: a data file is on the path and never scanned' 0 '!offpath' -- --inventory --rev main "${at[@]}"
    row 'inventory: a comment that names python is not an interpreter use' 0 '!scripts/release/lib.sh:' -- --inventory --rev main "${at[@]}"
    row 'inventory: a python string in a .py is not an interpreter use' 0 '!universe.py:' -- --inventory --rev main "${at[@]}"
    row 'inventory: the steps no file in this repository runs' 0 'not read  S04       the clean-room job' -- --inventory --rev main "${at[@]}"
    row 'inventory: an entry the tree does not have is printed' 0 'MISSING   S09       scripts/dogfood.sh' -- --inventory --rev v-rename-entry "${at[@]}"
    row 'check: a tree against itself is GREEN' 0 'GREEN RELPY' -- --check --base main --head main "${at[@]}"
    row 'FALSIFIER: a new .py run by its path, the word python never said' 1 'RED   new Python file on the release path: scripts/lib/newtool.py' -- --check --base main --head v-newpy "${at[@]}"
    row 'a python line added to a release script' 1 'RED   interpreter uses on the release path rose 3 -> 4' -- --check --base main --head v-inline "${at[@]}"
    row 'a release script now starts a script that runs Python' 1 'RED   new Python file on the release path: scripts/tools/offpath.py' -- --check --base main --head v-reach-offpath "${at[@]}"
    row 'a python line removed is GREEN, and said' 0 'info  interpreter uses on the release path fell 3 -> 2' -- --check --base main --head v-fell "${at[@]}"
    row 'python added to a script off the path is GREEN' 0 'interp_uses 3->3, unresolved_py 0->0)' -- --check --base main --head v-offpath-py "${at[@]}"
    row 'comments that name python are GREEN' 0 'interp_uses 3->3, unresolved_py 0->0)' -- --check --base main --head v-comment "${at[@]}"
    row 'an entry renamed away is RED' 1 'RED   entry gone at head: S09 scripts/dogfood.sh' -- --check --base main --head v-rename-entry "${at[@]}"
    row 'an entry that appears at head is GREEN, and said' 0 'info  entry missing at base: S09 scripts/dogfood.sh' -- --check --base v-rename-entry --head main "${at[@]}"
    row 'the interpreter behind a default, ${PY:-python3}' 1 'RED   interpreter uses on the release path rose 3 -> 5' -- --check --base main --head v-pyvar "${at[@]}"
    row 'a module imported beside a release .py' 1 'RED   new Python file on the release path: scripts/lib/extra.py' -- --check --base main --head v-sibling-import "${at[@]}"
    row 'a make target reached by $(MAKE) from a release target' 1 'RED   new Python file on the release path: scripts/tools/offpath.py' -- --check --base main --head v-make-recursion "${at[@]}"
    row 'a .py that moved, byte for byte, is GREEN' 0 'info  moved: scripts/lib/helper.py -> scripts/lib/helper2.py' -- --check --base main --head v-move "${at[@]}"
    row 'an extensionless file with a python #! line' 1 'RED   new Python file on the release path: scripts/release/newtool' -- --check --base main --head v-shebang "${at[@]}"
    row 'a .py swapped for one with other bytes is RED' 1 'RED   new Python file on the release path: scripts/lib/other.py' -- --check --base main --head v-swap "${at[@]}"
    row 'a copy whose source is still on the path is RED' 1 'RED   new Python file on the release path: scripts/lib/helper_copy.py' -- --check --base main --head v-copy "${at[@]}"
    row 'a reference to a .py that is not tracked' 1 'RED   references to a .py the walk cannot find rose 0 -> 1' -- --check --base main --head v-unres "${at[@]}"
    row 'a local action reached from a workflow step' 1 '.github/actions/setup/action.yml 0 -> 2' -- --check --base main --head v-action "${at[@]}"
    row 'make -C runs another Makefile: no root target is followed' 0 'interp_uses 3->3, unresolved_py 0->0)' -- --check --base main --head v-make-C "${at[@]}"
    row 'a YAML file a release script reads is data, never scanned' 0 'python_files 3->3, python_lines 4->4, interp_uses 3->3, unresolved_py 0->0)' -- --check --base main --head v-data-yaml "${at[@]}"
    row 'a CI definition under ci/ is scanned like a workflow' 1 'ci/sections.yml 0 -> 2' -- --check --base main --head v-ci-yaml "${at[@]}"
    row 'uv is an interpreter word; setup-uv is not' 1 'RED   interpreter uses on the release path rose 3 -> 4' -- --check --base main --head v-uv "${at[@]}"
    row 'a recipe line inside a make conditional is still in the recipe' 1 'RED   interpreter uses on the release path rose 3 -> 4' -- --check --base main --head v-recipe-cond "${at[@]}"
    row 'a prerequisite on a continued rule line is followed' 1 'Makefile:extra 0 -> 1' -- --check --base main --head v-prereq-cont "${at[@]}"
    row 'a recipe line continued with a backslash is still in the recipe' 1 'RED   interpreter uses on the release path rose 3 -> 4' -- --check --base main --head v-recipe-cont "${at[@]}"
    row 'a make target whose name starts with a capital is followed' 1 'Makefile:Release 0 -> 1' -- --check --base main --head v-make-upper "${at[@]}"
    row 'make TARGET -C DIR runs another Makefile: no root target' 0 'interp_uses 3->3, unresolved_py 0->0)' -- --check --base main --head v-make-after "${at[@]}"
    row 'an extensionless file outside scripts/, run by its path' 1 'RED   new Python file on the release path: tools/gen' -- --check --base main --head v-extless "${at[@]}"
    row 'a file whose #! line runs uv run is Python' 1 'RED   new Python file on the release path: scripts/release/uvtool' -- --check --base main --head v-uv-shebang "${at[@]}"
    row 'a local action outside .github/ is followed and scanned' 1 'actions/py/action.yml 0 -> 1' -- --check --base main --head v-action-out "${at[@]}"
    row 'a dotted import reaches the submodule' 1 'RED   new Python file on the release path: scripts/lib/pkg/sub.py' -- --check --base main --head v-dotted "${at[@]}"
    row 'from . import NAME reaches the module beside the importer' 1 'RED   new Python file on the release path: scripts/lib/newmod.py' -- --check --base main --head v-relative "${at[@]}"
    row 'a parenthesised import list over several lines' 1 'RED   new Python file on the release path: scripts/lib/pkg2/b.py' -- --check --base main --head v-paren "${at[@]}"
    row 'an interpreter variable counts; PIPESTATUS, COPY, PYPI_URL do not' 1 'RED   interpreter uses on the release path rose 3 -> 4' -- --check --base main --head v-pyvar2 "${at[@]}"
    row 'a second python call on a counted line counts' 1 'RED   interpreter uses on the release path rose 3 -> 4' -- --check --base main --head v-twice "${at[@]}"
    row 'one base file covers one move: the second copy has joined' 1 'RED   new Python file on the release path: scripts/lib/helper3.py' -- --check --base main --head v-two-copies "${at[@]}"
    row 'a cycle of scripts ends the walk' 0 'interp_uses 3->3, unresolved_py 0->0)' -- --check --base main --head v-cycle "${at[@]}"
    row 'a tool named as "$HERE/tool", no extension' 1 'RED   new Python file on the release path: scripts/ladder-tool' -- --check --base main --head v-here-tool "${at[@]}"
    row 'make -C DIR TARGET reads the target from DIR/Makefile' 1 'cookbooks/x/Makefile:test 0 -> 1' -- --check --base main --head v-make-C-walk "${at[@]}"
    row 'make -f FILE TARGET reads the target from FILE' 1 'mk/Release.mk:ship 0 -> 1' -- --check --base main --head v-make-f "${at[@]}"
    row 'a Makefile include is followed' 1 'mk/rel.mk 0 -> 1' -- --check --base main --head v-include "${at[@]}"
    row 'a $(shell python3) outside every rule counts' 1 'Makefile:<global> 0 -> 1' -- --check --base main --head v-shell-assign "${at[@]}"
    row 'make with no target runs the default goal, .PHONY skipped' 1 'Makefile:all 0 -> 1' -- --check --base main --head v-default-goal "${at[@]}"
    row 'the default goal is never a .PHONY line' 1 '!offpath.py' -- --check --base main --head v-default-goal "${at[@]}"
    row 'python3 -m a.b reaches a/b.py' 1 'RED   new Python file on the release path: scripts/lib/runmod.py' -- --check --base main --head v-dash-m "${at[@]}"
    row 'an import in a python3 -c program is followed' 1 'RED   new Python file on the release path: scripts/lib/cmod.py' -- --check --base main --head v-dash-c "${at[@]}"
    row 'an import after a ";" is followed' 1 'RED   new Python file on the release path: scripts/lib/extra3.py' -- --check --base main --head v-semi-import "${at[@]}"
    row 'an import after try: is followed' 1 'RED   new Python file on the release path: scripts/lib/extra4.py' -- --check --base main --head v-try-import "${at[@]}"
    row 'a ) in a comment does not end an import list' 1 'RED   new Python file on the release path: scripts/lib/pkg3/b.py' -- --check --base main --head v-comment-paren "${at[@]}"
    row 'an import list continued with a backslash' 1 'RED   new Python file on the release path: scripts/lib/pkg4/b.py' -- --check --base main --head v-bslash "${at[@]}"
    row 'twine and "$PYBIN" are interpreter uses' 1 'RED   interpreter uses on the release path rose 3 -> 5' -- --check --base main --head v-runners "${at[@]}"
    row 'a local action with an action.yaml' 1 'actions/py2/action.yaml 0 -> 1' -- --check --base main --head v-action-yaml "${at[@]}"
    row 'a symlink is read as the path it points to' 1 'RED   new Python file on the release path: scripts/tools/linked.py' -- --check --base main --head v-symlink "${at[@]}"
    row 'a bare name that matches two files reaches both' 1 'RED   interpreter uses on the release path rose 3 -> 5' -- --check --base main --head v-multi-bare "${at[@]}"
    row 'an import in a python3 heredoc is followed' 1 'RED   new Python file on the release path: scripts/hdmod.py' -- --check --base main --head v-heredoc "${at[@]}"
    row 'an import in a python3 -c program over many lines' 1 'RED   new Python file on the release path: scripts/cmod2.py' -- --check --base main --head v-dash-c-multi "${at[@]}"
    row 'make -j $(nproc) TARGET keeps the target' 1 'Makefile:gen 0 -> 1' -- --check --base main --head v-make-nproc "${at[@]}"
    row 'a directory given to an interpreter runs the Python under it' 1 'RED   new Python file on the release path: tools/tests/test_x.py' -- --check --base main --head v-dir-arg "${at[@]}"
    row 'cd DIR && make TARGET reads DIR/Makefile' 1 'cookbooks/y/Makefile:test 0 -> 1' -- --check --base main --head v-cd-make "${at[@]}"
    row 'an include of a file that is not .mk is read whole' 1 'Makefile.common:<all> 0 -> 1' -- --check --base main --head v-include-frag "${at[@]}"
    row 'make -C DIR with only a GNUmakefile there' 1 'docs/GNUmakefile:html 0 -> 1' -- --check --base main --head v-gnumakefile "${at[@]}"
    row 'make && x runs the default goal' 1 'Makefile:all 0 -> 1' -- --check --base main --head v-make-andand "${at[@]}"
    row 'the default goal is counted once' 0 'interp_uses 3->3' -- --check --base main --head v-default-dup "${at[@]}"
    row 'a symlink with no extension is read as the path it points to' 1 'RED   new Python file on the release path: scripts/tools/linked2.py' -- --check --base main --head v-link-extless "${at[@]}"
    row 'maturin and ruff are interpreter uses' 1 'RED   interpreter uses on the release path rose 3 -> 5' -- --check --base main --head v-words2 "${at[@]}"
    row 'a tree with no entry read: not measured, never GREEN' 2 'no entry of the release path is read' -- --check --base v-noentries --head v-noentries "${at[@]}"
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
interpreter_word_blind      s/(python(\[0-9\]+/(pythonQ([0-9]+/
second_use_uncounted        s/n++; t = substr(t, RSTART + RLENGTH - 1)/n++; t = ""/
interp_variable_blind       s/(PY|PYTHON|PIP|PYTEST|PYBIN|PYEXE|PYCMD|py|python|pip|pytest)/(PYQ)/
comment_lines_counted       s@if (s ~ /^\[ \\t\]\*(#|\$)/) return@if (0) return@
python_strings_counted      s/^    tokens(\$0)$/    code_line($0, NR)/
data_files_scanned          s/{ kind = "data"; exit }/{ kind = "code" }/
shebang_not_read            s@if (\$0 ~ /^#!/ && \$0 ~ /(python|uv run)/) kind = "python"@if (0) kind = "python"@
uv_shebang_blind            s@(python|uv run)/) kind@(python)/) kind@
recipe_never_ends           s/^    inrec = 0$/    inrec = inrec/
conditional_ends_recipe     /(ifdef|ifndef|ifeq|ifneq|else|endif)(/d
recipe_continuation_dropped s/if (inrec && rcont) {/if (0) {/
prereq_continuation_dropped s/if (inrec && pcont) {/if (0) {/
make_target_lowercase_only  s/if (w ~ \/^\[A-Za-z0-9_\]\[A-Za-z0-9_.-\]\*\$\/) tg/if (w ~ \/^[a-z0-9_][A-Za-z0-9_.-]*$\/) tg/
path_token_needs_extension  s@ && w !~ /\\/\[A-Za-z0-9_.-\]/) continue@) continue@
referrer_path_dropped       s/for c in "\$3" "\$4"; do/for c in "$3"; do/
bare_name_dropped           s/if \[ "\$hit" = 0 \] && \[ -n "\${bybase\[\$5\]+x}" \]; then/if false; then/
action_dir_dropped          s@blob\[\$c/action.yml\]+x@blob[$c/action.ymlX]+x@
action_yml_anywhere_dropped s@action\\.ya?ml\$/) kind@actionQ.ya?ml$/) kind@
imports_dropped             s/P) module "\$node" "\$d" "\$f1" ;;/P) : ;;/
dotted_prefix_dropped       s/if (!nd) printf "P/if (!nd \&\& i == 1) printf "P/
relative_import_dropped     s/R) relmodule "\$node" "\$f1" ;;/R) : ;;/
from_names_dropped          s/^        fromnames(x, m)$/        x = x/
paren_import_dropped        s/{ impx = x; impx_paren = 1 }/{ impx_paren = 1 }/
make_targets_dropped        s/enq "\$mf:\$f1" "\$node";/:;/
cycle_unguarded             s/if \[ -n "\${seen\[\$1\]+x}" \]; then return 0; fi/:/
worktree_read               s@if ! g cat-file blob "\$b" > "\$out/cur"; then@if ! cat -- "$REPO/$name" > "$out/cur"; then@
new_python_never_flagged    s/if (p in bpy) continue/continue/
copy_claims_moved           s/if (!(q in hpy) && !(q in claimed) && /if (/
claim_not_kept              s/if (!(q in hpy) && !(q in claimed) && bpy/if (!(q in hpy) \&\& bpy/
move_ignores_bytes          s/bpy\[q\] == hpy\[p\]/1/
interp_ratchet_off          s/if (hi > bi) {/if (0) {/
unresolved_ratchet_off      s/if (hu > bu) {/if (0) {/
gone_entry_ignored          s/if (be\[k\] == "read" && he\[k\] != "read")/if (0)/
unread_blob_ignored         s@if ! g cat-file blob "\$b" > "\$out/cur"; then@g cat-file blob "$b" > "$out/cur"; if false; then@
yaml_outside_ci_scanned     s@ && name ~ /^(\\.github|ci)\\//@@
ci_yaml_not_code            s@(\\.github|ci)@(\\.github)@
uv_word_blind               s@|pipenv|uvx?|@|pipenv|@
make_C_ignored              s@if (cd != "") mf = cd "/" mf@if (0) mf = mf@
make_f_ignored              s@if (w ~ /^-(C|-directory)/) cd = v; else mf = v@if (w ~ /^-(C|-directory)/) cd = v@
global_lines_skipped        s@^kind == "make" && target == "<global>" {@kind == "make" \&\& target == "<global>" { next@
global_node_never_queued    s@if \[ "\$target" != "<global>" \] && \[@if false \&\& [@
default_goal_dropped        s/if (nt == 0 && !odd) tg\[++nt\]/if (0) tg[++nt]/
default_goal_takes_phony    s@nm\[1\] !~ /^\[.\]/ && @@
dash_m_blind                s/(^|\[ \\t\])-m\[ \\t\]+\[A-Za-z_\]/(^|[ \\t])-mQ[ \\t]+[A-Za-z_]/
dash_c_blind                s@if (!match(s, /(^|\[ \\t\])-c\[ \\t\]+/)) return@return@
run_module_dropped          s/D) runmodule "\$node" "\$d" "\$f1" ;;/D) : ;;/
semicolon_import_blind      s/ns = split(s, seg, ";")/ns = split(s, seg, "\\n")/
try_import_blind            s/sub(\/^\[ \\t\]\*(try|else/sub(\/^[ \\t]*(tryQ|elseQ/
comment_paren_ends_list     s@m = s; sub(/#\.\*/, "", m); impx_end@m = s; impx_end@
backslash_import_dropped    s@else if (m ~ /\\\\\[ \\t\]\*\$/) {@else if (0) {@
runner_words_blind          s/|twine|mkdocs|/|mkdocs|/
pybin_blind                 s/|PYBIN|PYEXE|/|PYEXE|/
action_yaml_dropped         s@blob\[\$c/action.yaml\]+x@blob[$c/action.yamlX]+x@
bare_name_first_only        s/then enq "\$f" "\$1"; hit=1; fi/then enq "$f" "$1"; hit=1; break; fi/
heredoc_import_blind        s/if (pyhd != "") {/if (0) {/
dash_c_multiline_blind      s/index(substr(t, 2), q) == 0) pyq = q/index(substr(t, 2), q) == 0) pyq = ""/
make_expansion_ends_targets /an expansion such as/{n;s/; continue }$/; break }/}
make_operator_not_end       /an operator or a redirection/{n;s/) break$/) w = w/}
dir_arg_dropped             s/Y) for c in "\$f1" "\$f2"; do/Y) for c in; do/
cd_make_ignored             s/if (cdpre != "") mkrec/if (0) mkrec/
include_frag_dropped        s@mkrec("<all>", inc\[k\])@k = k@
gnumakefile_ignored         s/for c in "\${mf%Makefile}GNUmakefile" "\${mf%Makefile}makefile"; do/for c in; do/
default_goal_twice          s/mkrec(nm\[1\], name); exit/inrec = 1/
extless_link_ignored        s/else if (ENVIRON\["RELPY_LINK"\] != "") kind = "code"/else if (0) kind = "code"/
words2_blind                s/|ipython|maturin|/|ipython|/
no_entry_read_green         s/if \[ "\$nread" = 0 \]; then/if false; then/
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
