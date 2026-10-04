# keys.awk: raw call records (shim log + trace.awk output) -> Python entry points.
# Usage: awk -v root=REPO -v tmpd=TMPDIR -f keys.awk SHEBANGS RECORDS
#   tmpd:     the TMPDIR the lane ran with (--run stores it in meta), so the key
#             does not depend on the environment --count runs in.
#   SHEBANGS: "file <TAB> interpreter-basename" for every exec'd file whose #!
#             line names python*, uv or uvx (python_runcount.sh builds it).
#   RECORDS:  src <TAB> cwd <TAB> caller argv <TAB> argv [<TAB> file]
# Output, one line per call that ran Python: key <TAB> src <TAB> caller
# An entry point is WHAT ran, so one key per distinct thing:
#   script:<path>   a file python ran (repo-relative when under root; ~tmp/<name>
#                   under a temp dir, whose mktemp name changes every run)
#   -m:<module>     python -m
#   -c@<caller>     inline code, keyed by the script that holds it
#   stdin@<caller>  code on stdin: a heredoc or a pipe (`python3 -`, bare python3)
#   info@<caller>   python -V / --version / -h
#   uv:<sub>:<arg>  uv run x.py, uv pip ... ; uvx:<tool>
# The caller is the basename of the script the parent shell runs ("bash -c" for
# inline shell), else the parent's program name.
BEGIN { FS = "\t"; US = sprintf("%c", 31); sub(/\/+$/, "", tmpd) }
function base(p) { sub(/\/+$/, "", p); sub(/^.*\//, "", p); return p }
function tmpath(p) { return p ~ /^\/(var\/)?tmp\// || (tmpd != "" && index(p, tmpd "/") == 1) }
# A name made fresh each run (mktemp, $$) gets one stable spelling, or every run
# would count a "new" entry point. Each run of letters and digits r, after the
# text before it, is replaced by X when it is random by the strength asked for:
#   0 (any name)          a pid or a time: 5+ digits; the 10 X's of tmp.XXXXXXXXXX
#   1 (a dir in the repo) also .<6+ chars holding a digit or a capital (mktemp -d d.XXXXXX)
#   2 (a temp-dir name)   also 6+ chars holding a digit or a capital, or any 10+
# A random suffix of only lower-case letters (about 1 in 200 for six X's) keeps its name.
function randish(r, lvl, before) {
    if (r ~ /^[0-9]+$/ && length(r) >= 5) return 1
    if (length(r) == 10 && before ~ /(^|[^A-Za-z0-9])tmp\.$/) return 1
    if (lvl >= 1 && length(r) >= 6 && r ~ /[0-9A-Z]/ && (lvl == 2 || before ~ /\.$/)) return 1
    return lvl == 2 && length(r) >= 10
}
function stable(s, lvl,    out, r) {
    out = ""
    while (match(s, /[A-Za-z0-9]+/)) {
        r = substr(s, RSTART, RLENGTH)
        out = out substr(s, 1, RSTART - 1)
        out = out (randish(r, lvl, out) ? "X" : r)
        s = substr(s, RSTART + RLENGTH)
    }
    return out s
}
function norm(p, cwdir,    n, i, parts, out, k, rel) {
    if (p !~ /^\//) p = cwdir "/" p
    n = split(p, parts, "/"); k = 0
    for (i = 1; i <= n; i++) {
        if (parts[i] == "" || parts[i] == ".") continue
        if (parts[i] == "..") { if (k > 0) k--; continue }
        out[++k] = parts[i]
    }
    p = ""
    for (i = 1; i <= k; i++) p = p "/" out[i]
    rel = (root != "" && index(p, root "/") == 1)
    # A file under a temp dir has a fresh name each run (mktemp): key it by its
    # basename, or every run would count a "new" entry point.
    if (!rel && tmpath(p)) return "~tmp/" stable(base(p), 2)
    if (!rel) return p == "" ? "/" : p
    # In the repo, a mktemp dir is collapsed, and so is a pid in the file name.
    n = split(substr(p, length(root) + 2), parts, "/"); p = ""
    for (i = 1; i < n; i++) p = p stable(parts[i], 1) "/"
    return p stable(parts[n], 0)
}
function interp(name) { return name ~ /^python[0-9.]*t?$/ || name == "uv" || name == "uvx" }
function cname(p) { return stable(base(p), tmpath(p) ? 2 : 0) }
function caller_of(s,    a, n, b, i) {
    if (s == "?" || s == "") return "?"
    n = split(s, a, US); b = base(a[1])
    if (b ~ /^(ba|da|z|k)?sh$/) {
        for (i = 2; i <= n; i++) {
            if (a[i] ~ /^-[A-Za-z]*c$/) return b " -c"   # -c, -ec, -lc ...
            if (a[i] ~ /^[-+]/) { if (a[i] == "-o" || a[i] == "+o") i++; continue }
            return cname(a[i])
        }
    }
    return cname(a[1])
}
function py_key(a, n, cwdir, who,    i, x, f, j) {
    for (i = 2; i <= n; i++) {
        x = a[i]
        if (x == "-") return "stdin@" who
        if (x == "-V" || x == "--version" || x == "-h" || x == "--help") return "info@" who
        if (x == "-W" || x == "-X" || x == "--check-hash-based-pycs") { i++; continue }
        if (x ~ /^-[WX]./) continue   # -Wonce, -Xutf8: the value is joined, not a -c or -m
        if (x ~ /^-[A-Za-z]/) {
            # Short flags join (-Bc, -Im) and so does a value (-mjson.tool): the
            # first c or m in the token decides, and what follows an m is the module.
            f = substr(x, 2); j = match(f, /[cm]/)
            if (!j) continue
            if (substr(f, j, 1) == "c") return "-c@" who
            f = substr(f, j + 1)
            return "-m:" (f != "" ? f : a[i + 1])
        }
        if (x ~ /^-/) continue
        return "script:" norm(x, cwdir)
    }
    return "stdin@" who
}
function uv_key(a, n, cwdir, tool,    i, x, sub1) {
    sub1 = ""
    for (i = 2; i <= n; i++) {
        x = a[i]
        # An option whose value is the next word; the value is not the script.
        if (x ~ /^--(with|with-requirements|with-editable|python|project|directory|from|index|index-url|default-index|extra-index-url|find-links|constraints?|overrides?|requirements?|build-constraints?|cache-dir|config-file|config-setting|extra|group|only-group|no-group|package|env-file|color|index-strategy|keyring-provider|resolution|prerelease|exclude-newer|link-mode|python-platform|python-version|python-preference|allow-insecure-host|refresh-package|reinstall-package|upgrade-package|no-binary-package|no-build-package|no-build-isolation-package)$/) { i++; continue }
        if (x ~ /^-[pwcrifCP]$/) { i++; continue }
        if (x ~ /^-/) continue
        if (tool == "uvx") return "uvx:" x
        if (sub1 == "") { sub1 = x; continue }
        if (x ~ /\.py$/ || x ~ /\//) x = "script:" norm(x, cwdir)
        return "uv:" sub1 ":" x
    }
    return (tool == "uvx" ? "uvx:" : "uv:" sub1)
}
FILENAME == ARGV[1] { sb[$1] = $2; next }   # NR == FNR is wrong when SHEBANGS is empty
{
    src = $1; cwdir = $2; who = caller_of($3)
    n = split($4, a, US); name = base(a[1]); file = (NF >= 5 ? $5 : "")
    if (src == "trace" && !interp(name) && interp(base(file))) name = base(file)
    if (src == "trace" && !interp(name)) {
        if (!(file in sb)) next
        # A script run by its own #! line: the entry point is the script.
        printf "script:%s\t%s\t%s\n", norm(file, "/"), src, who; next
    }
    if (!interp(name)) next
    if (name == "uv" || name == "uvx") key = uv_key(a, n, cwdir, name)
    else key = py_key(a, n, cwdir, who)
    printf "%s\t%s\t%s\n", key, src, who
}
