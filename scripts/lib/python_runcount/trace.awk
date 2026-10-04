# trace.awk: an strace log (python_runcount.sh --run) -> one raw record per
# successful execve, in the shim log's format plus the exec'd file:
#   trace <TAB> cwd <TAB> caller argv <TAB> argv <TAB> file (absolute)
# argv elements are joined by 0x1f. The caller is the PARENT process's argv at
# the time of the call -- what the shim reads from /proc/$PPID/cmdline. A forked
# child inherits its parent's argv and working dir until it execs or chdirs.
# Every exec is emitted, not only python ones: an absolute-shebang script is
# exec'd by its own path, and keys.awk decides from its #! line.
# Input: strace -f -q -s 4096 -e trace=execve,chdir,clone,clone3,fork,vfork
#        (-q, not -qq: the "+++ exited" lines mark where a pid's lifetime ends)
# Variables: -v cwd0=<the working dir the traced command started in>.
# Usage: awk -v cwd0=DIR -f trace.awk LOG LOG   (the same log twice)
BEGIN { US = sprintf("%c", 31) }
function norm(p, base,    n, i, parts, out, k) {
    if (p !~ /^\//) p = base "/" p
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
function unq(s) { # strace's quoted string -> raw (\" and \\ only)
    sub(/^"/, "", s); sub(/"(\.\.\.)?$/, "", s)
    gsub(/\\"/, "\"", s); gsub(/\\\\/, "\\", s)
    return s
}
function seen(pid,    p, k) {
    if (pid in cwd) return
    k = pid SUBSEP (life[pid] + 0)
    p = (k in par1) ? par1[k] : ""
    if (p != "") seen(p)
    cwd[pid] = (p != "" ? cwd[p] : cwd0); argv[pid] = (p != "" ? argv[p] : "?"); par[pid] = p
}
function do_exec(pid, body,    file, args, n, i, a, joined) {
    # body: execve("FILE", ["a", "b"], 0x... /* n vars */
    file = body; sub(/^execve\("/, "", file); sub(/", \[.*$/, "", file)
    args = body; sub(/^execve\("[^"]*", \[/, "", args); sub(/\], 0x[0-9a-f]+.*$/, "", args)
    n = split(args, a, /", "/); joined = ""
    for (i = 1; i <= n; i++) joined = joined (i > 1 ? US : "") unq((i > 1 ? "\"" : "") a[i] (i < n ? "\"" : ""))
    file = norm(unq("\"" file "\""), cwd[pid])
    argv[pid] = joined
    printf "trace\t%s\t%s\t%s\t%s\n", cwd[pid], (par[pid] != "" ? argv[par[pid]] : "?"), joined, file
}
# Pass 1 (the log is read twice): who forked whom. strace may print a child's
# exec before its parent's clone result, so the tree must be known up front.
# The kernel reuses pids, so the tree is kept per LIFETIME: pid P after its Nth
# exit line is a new process, with its own parent, argv and working dir.
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
# exit: the pid's lifetime ends; a later process with this pid starts afresh.
rest ~ /^\+\+\+ (exited|killed) / { delete cwd[pid]; delete argv[pid]; delete par[pid]; delete pend[pid]; life[pid]++; next }
# execve: whole line, or <unfinished ...> + <... execve resumed>
rest ~ /^execve\(/ && rest ~ /<unfinished \.\.\.>$/ { pend[pid] = rest; next }
rest ~ /^<\.\.\. execve resumed>/ {
    if ((pid in pend) && rest ~ /= 0$/) do_exec(pid, pend[pid])
    delete pend[pid]; next
}
rest ~ /^execve\(/ && rest ~ /\) += 0$/ { do_exec(pid, rest); next }
# chdir: whole line only (bash's cd is not interleaved in practice)
rest ~ /^chdir\(/ && rest ~ /\) += 0$/ {
    d = rest; sub(/^chdir\(/, "", d); sub(/\) += 0$/, "", d)
    cwd[pid] = norm(unq(d), cwd[pid]); next
}
# fork family: the child inherits argv and cwd. Resumed lines carry the result.
(rest ~ /^(clone3?|v?fork)\(/ || rest ~ /^<\.\.\. (clone3?|v?fork) resumed>/) && rest ~ /= [0-9]+$/ {
    child = rest; sub(/^.*= /, "", child)
    if (child + 0 > 0) seen(child)
    next
}
