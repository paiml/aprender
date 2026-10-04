# trace.awk: an strace log (python_runcount.sh --run) -> one record per python,
# pypy, uv or uvx PROCESS IMAGE (an execve up to the process's exit or its next
# execve):
#   trace <TAB> pid <TAB> kind <TAB> value <TAB> caller
#     kind script  value = the absolute path of the entry point it opened
#     kind module  value = M, the dotted module name the entry point names
#     kind inline  value = -   (it opened no file its argv names: -c, stdin, -V)
#     kind uv      value = uv or uvx
#   caller = the absolute path of the nearest entry point up the process tree
#            (the lane script, a python script, ...), or exe:<file> with none
# No python or uv option is parsed. The interpreter is the exec'd FILE (its
# basename, or the first word of its #! line), never argv[0]. The entry point is
# the first non-directory file the image opens that an argv word names:
#   script  the word, resolved by the cwd at exec, is the opened path;
#   module  the opened path is M.py, M/__init__.py, M/__main__.py or
#           __pycache__/<last>.<tag>.pyc, and a word is M or a flag cluster
#           ending in M (-mjson.tool); the longest M wins.
# An image replaced by execve before it opened its entry point (the shim, env)
# is not a process of its own and emits nothing; a forked copy without an exec
# emits nothing.
# Input: strace -f -q -s 4096 -e trace=execve,execveat,?open,openat,chdir,fchdir,
#        clone,clone3,fork,vfork  (-q, not -qq: "+++ exited" ends a pid's lifetime)
# Variables: -v cwd0=<the run's working dir> -v shimdir=<the run's shim dir>
# Usage: awk -v cwd0=DIR -v shimdir=DIR -f trace.awk LOG LOG   (the same log twice)
BEGIN { PYRE = "^(python|pypy)[0-9.]*[a-z]*$"; UVRE = "^uvx?$"; IDRE = "^[A-Za-z_][A-Za-z0-9_]*$" }
function base(p) { sub(/.*\//, "", p); return p }
function dir(p) { if (p !~ /\//) return ""; sub(/\/[^\/]*$/, "", p); return p }
function norm(p, cwd,    n, i, parts, out, k) {
    if (p !~ /^\//) p = cwd "/" p
    n = split(p, parts, "/"); k = 0
    for (i = 1; i <= n; i++) {
        if (parts[i] == "" || parts[i] == ".") continue
        if (parts[i] == "..") { if (k > 0) k--; continue }
        out[++k] = parts[i]
    }
    p = ""
    for (i = 1; i <= k; i++) p = p "/" out[i]
    return p == "" ? "/" : p
}
# dec: one strace string literal (quotes included) -> its bytes. Octal escapes
# (\303) are decoded, and \" and \\. \n, \t and the rest stay escaped: a record
# is one line of tab-separated fields.
function dec(s,    out, i, c, j, v) {
    s = substr(s, 2, length(s) - 2)
    if (index(s, "\\") == 0) return s
    out = ""
    while ((i = index(s, "\\")) > 0) {
        out = out substr(s, 1, i - 1); c = substr(s, i + 1, 1)
        if (c ~ /[0-7]/) {
            v = 0
            for (j = 1; j <= 3 && substr(s, i + j, 1) ~ /[0-7]/; j++) v = v * 8 + substr(s, i + j, 1)
            out = out ((v == 9 || v == 10) ? substr(s, i, j) : sprintf("%c", v)); s = substr(s, i + j)
        } else if (c == "\"" || c == "\\") {
            out = out c; s = substr(s, i + 2)
        } else {
            out = out "\\" c; s = substr(s, i + 2)
        }
    }
    return out s
}
function strs(s, a,    n) { # every string literal in s, decoded, in order -> a[1..n]
    n = 0
    while (match(s, /"([^"\\]|\\.)*"/)) { a[++n] = dec(substr(s, RSTART, RLENGTH)); s = substr(s, RSTART + RLENGTH) }
    return n
}
function after(s) { match(s, /"([^"\\]|\\.)*"/); return substr(s, RSTART + RLENGTH) } # the text after the first literal
function result(s) { if (s !~ /\) += -?[0-9]+$/) return -1; sub(/^.*\) += /, "", s); return s + 0 }
function hashbang(f,    line, r, t) { # the first word of f's #! line; "" if none; "?" if f cannot be read
    if (f in hb) return hb[f]
    r = (getline line < f); close(f)
    if (r < 0) return hb[f] = "?"
    if (r == 0 || line !~ /^#!/) return hb[f] = ""
    sub(/^#![ \t]*/, "", line); split(line, t, /[ \t]+/)
    return hb[f] = t[1]
}
function getfd(pid, fd) { # a pid's fd table, falling back to its parent's (inherited at fork)
    while (pid != "") { if ((pid, fd) in fdp) return fdp[pid, fd]; pid = par[pid] }
    return ""
}
function seen(pid,    p, k) {
    if (pid in cwd) return
    k = pid SUBSEP (life[pid] + 0)
    p = (k in par1) ? par1[k] : ""
    if (p != "") seen(p)
    par[pid] = p; img[pid] = 0
    cwd[pid] = (p != "" ? cwd[p] : cwd0)
    ctx[pid] = (p != "" ? ctx[p] : "")
    cexe[pid] = (p == "" ? "?" : (img[p] ? base(exe[p]) : cexe[p]))
}
function emit(pid, kind, val) { printf "trace\t%s\t%s\t%s\t%s\n", pid, kind, val, caller[pid] }
function finish(pid, why) { # why: exit, exec (the image is replaced) or end (still running)
    if (!img[pid]) return
    img[pid] = 0
    if (id[pid] == "unk") { # a #! script gone before --count: counted only if its interpreter read it
        if (ekind[pid] == "script" && entry[pid] == exe[pid]) emit(pid, "script", entry[pid])
    } else if (id[pid] == "py") {
        if (ekind[pid] != "") emit(pid, ekind[pid], entry[pid])
        else if (why != "exec") emit(pid, "inline", "-")
    } else if (id[pid] == "uv" && why != "exec") emit(pid, "uv", uvname[pid])
}
function do_exec(pid, rest,    a, n, f, d, fd, i, k, h, start) {
    n = strs(rest, a)
    if (n < 1) return
    d = cwd[pid]
    if (rest ~ /^execveat\([0-9]+,/) { fd = rest; sub(/^execveat\(/, "", fd); sub(/,.*$/, "", fd); d = getfd(pid, fd) }
    if (d == "") return
    f = (a[1] == "" ? d : norm(a[1], d))
    finish(pid, "exec")
    img[pid] = 1; exe[pid] = f; entry[pid] = ""; ekind[pid] = ""; id[pid] = ""
    caller[pid] = (ctx[pid] != "" ? ctx[pid] : "exe:" cexe[pid])
    k = 0; start = 3   # a[1] is the file, a[2] is argv[0]: neither is a word the image was handed
    if (base(f) ~ PYRE) id[pid] = "py"
    else if (base(f) ~ UVRE) { id[pid] = "uv"; uvname[pid] = base(f) }
    else {
        h = hashbang(f)
        if (h == "?") id[pid] = "unk"
        else if (base(h) ~ PYRE) id[pid] = "py"
        else if (base(h) ~ UVRE) { id[pid] = "uv"; uvname[pid] = base(h) }
        # a #! script: the interpreter is handed the script's own path
        if (h != "") { k++; tok[pid, k] = f; raw[pid, k] = f }
    }
    for (i = start; i <= n; i++) {
        if (a[i] == "") continue
        k++; raw[pid, k] = a[i]; tok[pid, k] = norm(a[i], cwd[pid])
    }
    ntok[pid] = k
}
function modname(pid, p,    last, d, m, best, i, t, L) { # the longest module name an argv word gives p
    if (p ~ /\/__pycache__\/[^\/]+\.pyc$/) { last = base(p); sub(/\..*$/, "", last); d = dir(dir(p)) }
    else if (p ~ /\.py$/) { last = base(p); sub(/\.py$/, "", last); d = dir(p) }
    else return ""
    if (last == "__init__" || last == "__main__") { last = base(d); d = dir(d) }
    if (last !~ IDRE) return ""
    best = ""; m = last
    while (1) {
        L = length(m)
        for (i = 1; i <= ntok[pid]; i++) {
            t = raw[pid, i]
            if (t == m || (t ~ /^-/ && length(t) > L && substr(t, length(t) - L + 1) == m)) best = m
        }
        if (d == "" || base(d) !~ IDRE) break
        m = base(d) "." m; d = dir(d)
    }
    return best
}
function do_open(pid, rest,    a, d, fd, p, r, i, m) {
    if (strs(rest, a) < 1) return
    r = result(rest)
    if (r < 0) return
    d = cwd[pid]
    if (rest ~ /^openat\([0-9]+,/) { fd = rest; sub(/^openat\(/, "", fd); sub(/,.*$/, "", fd); d = getfd(pid, fd) }
    if (a[1] !~ /^\// && d == "") return
    p = norm(a[1], d)
    fdp[pid, r] = p
    if (after(rest) ~ /O_DIRECTORY/) return
    if (!img[pid] || ekind[pid] != "") return
    if (shimdir != "" && index(p, shimdir "/") == 1) return   # the shim is never an entry point
    for (i = 1; i <= ntok[pid]; i++) if (tok[pid, i] == p) { entry[pid] = p; ekind[pid] = "script"; ctx[pid] = p; return }
    if (id[pid] == "py" && (m = modname(pid, p)) != "") { entry[pid] = m; ekind[pid] = "module"; ctx[pid] = p }
}
# Pass 1 (the log is read twice): who forked whom. strace may print a child's
# lines before its parent's clone result, so the tree must be known up front.
# The kernel reuses pids, so the tree is kept per LIFETIME: pid P after its Nth
# exit line is a new process, with its own parent, image and working dir.
NR == FNR {
    if ($0 ~ /^[0-9]+ +\+\+\+ (exited|killed) /) { life1[$1]++; next }
    if ($0 ~ / (clone3?|v?fork)\(/ || $0 ~ /<\.\.\. (clone3?|v?fork) resumed>/) if ($0 ~ /= [0-9]+$/) {
        c = $0; sub(/^.*= /, "", c); if (c + 0 > 0) par1[c, life1[c] + 0] = $1
    }
    next
}
{
    pid = $1; seen(pid)
    rest = $0; sub(/^[0-9]+ +/, "", rest)
}
# A call strace split around another pid's line: keep the first half, read it whole at its resumed line.
rest ~ / <unfinished \.\.\.>$/ { sub(/ <unfinished \.\.\.>$/, "", rest); pend[pid] = rest; next }
rest ~ /^<\.\.\. [a-z0-9_]+ resumed>/ {
    if (!(pid in pend)) next
    sub(/^<\.\.\. [a-z0-9_]+ resumed>/, "", rest); rest = pend[pid] rest; delete pend[pid]
}
# exit: the image ends; a later process with this pid starts afresh.
rest ~ /^\+\+\+ (exited|killed) / {
    finish(pid, "exit")
    delete cwd[pid]; delete ctx[pid]; delete par[pid]; delete pend[pid]; life[pid]++; next
}
rest ~ /^execve(at)?\(/ { if (result(rest) == 0) do_exec(pid, rest); next }
rest ~ /^(open|openat)\(/ { do_open(pid, rest); next }
rest ~ /^chdir\(/ { if (result(rest) == 0 && strs(rest, cd) > 0) cwd[pid] = norm(cd[1], cwd[pid]); next }
rest ~ /^fchdir\(/ {
    if (result(rest) == 0) { fd = rest; sub(/^fchdir\(/, "", fd); sub(/\).*$/, "", fd); d = getfd(pid, fd); if (d != "") cwd[pid] = d }
    next
}
# fork family: the child inherits cwd and caller. Resumed lines carry the result.
rest ~ /^(clone3?|v?fork)\(/ { child = result(rest); if (child > 0) seen(child); next }
END { for (p in img) finish(p, "end") }
