# keys.awk: raw call records (shim log + trace.awk output) -> Python entry points.
# Usage: awk -v root=REPO -f keys.awk SHEBANGS RECORDS
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
BEGIN { FS = "\t"; US = sprintf("%c", 31); tmpd = ENVIRON["TMPDIR"]; sub(/\/+$/, "", tmpd) }
function base(p) { sub(/\/+$/, "", p); sub(/^.*\//, "", p); return p }
# mktemp names (tmp.XXXXXXXXXX) change every run; one stable spelling keeps keys comparable.
function stable(s) { gsub(/tmp\.[A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9]/, "tmp.X", s); return s }
function norm(p, cwdir,    n, i, parts, out, k) {
    if (p !~ /^\//) p = cwdir "/" p
    n = split(p, parts, "/"); k = 0
    for (i = 1; i <= n; i++) {
        if (parts[i] == "" || parts[i] == ".") continue
        if (parts[i] == "..") { if (k > 0) k--; continue }
        out[++k] = parts[i]
    }
    p = ""
    for (i = 1; i <= k; i++) p = p "/" out[i]
    if (root != "" && index(p, root "/") == 1) p = substr(p, length(root) + 2)
    # A file under a temp dir has a fresh name each run (mktemp): key it by its
    # basename, or every run would count a "new" entry point.
    if (p ~ /^\/(var\/)?tmp\// || (tmpd != "" && index(p, tmpd "/") == 1)) p = "~tmp/" stable(base(p))
    return p == "" ? "/" : p
}
function interp(name) { return name ~ /^python[0-9.]*$/ || name == "uv" || name == "uvx" }
function caller_of(s,    a, n, b, i) {
    if (s == "?" || s == "") return "?"
    s = stable(s)
    n = split(s, a, US); b = base(a[1])
    if (b ~ /^(ba|da|z|k)?sh$/) {
        for (i = 2; i <= n; i++) {
            if (a[i] == "-c") return b " -c"
            if (a[i] ~ /^[-+]/) { if (a[i] == "-o" || a[i] == "+o") i++; continue }
            return base(a[i])
        }
    }
    return b
}
function py_key(a, n, cwdir, who,    i, x) {
    for (i = 2; i <= n; i++) {
        x = a[i]
        if (x == "-") return "stdin@" who
        if (x == "-V" || x == "--version" || x == "-h" || x == "--help") return "info@" who
        if (x == "-W" || x == "-X" || x == "--check-hash-based-pycs") { i++; continue }
        if (x ~ /^-[A-Za-z]*c/) return "-c@" who
        if (x ~ /^-[A-Za-z]*m$/) return "-m:" a[i + 1]
        if (x ~ /^-[A-Za-z]*m./) { sub(/^-[A-Za-z]*m/, "", x); return "-m:" x }
        if (x ~ /^-/) continue
        return "script:" norm(x, cwdir)
    }
    return "stdin@" who
}
function uv_key(a, n, cwdir, tool,    i, x, sub1) {
    sub1 = ""
    for (i = 2; i <= n; i++) {
        x = a[i]
        if (x ~ /^--(with|python|project|directory|from|index-url|extra|group|package|env-file)$/ || x == "-p" || x == "-w") { i++; continue }
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
