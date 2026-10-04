# keys.awk: process images (trace.awk) -> one entry point per image:
#   key <TAB> source <TAB> caller
# Inputs, in this order:
#   TRACKED  the repo's tracked files, one repo-relative path a line (git ls-files)
#   HASHES   path <TAB> sha12 of each script file still on disk at --count
#   SHIMLOG  the shim's records: "shim pid name" and "hash pid path sha12"
#   IMAGES   trace.awk's records: "trace pid kind value caller"
# Variables: -v root=<physical repo root> -v tmpd=<the run's TMPDIR; empty = /tmp>
# Keys:
#   script:REL        a tracked file, by its repo path
#   script:#SHA       an untracked file in the repo or a temp dir, by its content
#                     (the shim's hash at call time, else the hash at --count)
#   script:gone@WHO   such a file nobody hashed and that is gone at --count
#   script:ABS        any other file, by its absolute path
#   -m:M              a module
#   inline@WHO        code on -c or stdin, by the entry point that ran it
#   uv@WHO, uvx@WHO   uv, by its call site (its options are never read)
# WHO is a caller: its repo path if tracked, ~tmp if a temp or untracked repo
# file (a fresh name each run), else its basename, or exe:<file> as given.
# The source is shim+trace when the shim logged that pid (the shim execs the
# real tool, so the pid is the same process), else trace.
BEGIN { FS = "\t"; if (tmpd != "") sub(/\/+$/, "", tmpd) }
FILENAME == ARGV[1] { tracked[$0] = 1; next }
FILENAME == ARGV[2] { nowhash[$1] = $2; next }
FILENAME == ARGV[3] {
    if ($1 == "shim" && NF == 3) shim[$2] = $3
    else if ($1 == "hash" && NF == 4) callh[$2, norm($3)] = $4
    next
}
function norm(p,    n, i, parts, out, k) { # /a/./b/../c -> /a/c (the shim logs $PWD/arg as given)
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
function rel(p) { return (root != "" && index(p, root "/") == 1) ? substr(p, length(root) + 2) : "" }
function temp(p) { return index(p, (tmpd != "" ? tmpd : "/tmp") "/") == 1 } # mktemp's own rule: $TMPDIR, else /tmp
function cname(c,    r) {
    if (c ~ /^exe:/) return c
    r = rel(c)
    if (r != "" && (r in tracked)) return r
    if (r != "" || temp(c)) return "~tmp"
    sub(/.*\//, "", c); return c
}
function skey(pid, p, who,    r) {
    r = rel(p)
    if (r != "" && (r in tracked)) return "script:" r
    if (r != "" || temp(p)) {
        if ((pid, p) in callh) return "script:#" callh[pid, p]
        if (p in nowhash) return "script:#" nowhash[p]
        return "script:gone@" who
    }
    return "script:" p
}
FILENAME == ARGV[4] {
    if ($1 != "trace" || NF != 5) { printf "keys.awk: bad image record: %s\n", $0 > "/dev/stderr"; bad = 1; exit 1 }
    src = ($2 in shim) ? "shim+trace" : "trace"
    who = cname($5)
    if ($3 == "script") k = skey($2, $4, who)
    else if ($3 == "module") k = "-m:" $4
    else if ($3 == "inline") k = "inline@" who
    else if ($3 == "uv") k = $4 "@" who
    else { printf "keys.awk: unknown kind %s\n", $3 > "/dev/stderr"; bad = 1; exit 1 }
    printf "%s\t%s\t%s\n", k, src, who
    next
}
END { if (bad) exit 1 }
