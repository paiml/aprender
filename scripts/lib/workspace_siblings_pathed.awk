# workspace_siblings_pathed.awk — find in-tree crates declared from crates.io.
#
# Driven by scripts/check_workspace_siblings_pathed.sh, which supplies:
#   -v FILE=<repo-relative path>   for the report
#   -v NAMEFILE=<file>             one in-tree crate name per line
#
# Emits one violation per line: FILE \t line \t crate \t class \t text
#
# SECTION-SCOPED ON PURPOSE. `[features]` carries lines indistinguishable from
# dependency declarations by shape alone —
#     trueno-integration = ["trueno", "native"]
# — so a line-oriented grep would have to guess. Tracking the current section
# means the parser never sees them. `[patch.crates-io]` is out of scope for the
# same kind of reason: a patch entry IS the redirection.

function strip_comment(s,   out, i, c, inq) {
    out = ""; inq = 0
    for (i = 1; i <= length(s); i++) {
        c = substr(s, i, 1)
        if (c == "\"") inq = 1 - inq
        if (c == "#" && inq == 0) break
        out = out c
    }
    return out
}

function is_dep_sec(s) {
    if (s ~ /^\[(workspace\.)?(dependencies|dev-dependencies|build-dependencies)\]$/) return 1
    if (s ~ /^\[target\.[^]]*\.(dependencies|dev-dependencies|build-dependencies)\]$/) return 1
    return 0
}

# `[dependencies.foo]` / `[target.<cfg>.dev-dependencies.foo]`.
# Split at the LAST dot: a crate name never contains one, while a target cfg
# string can.
function dep_subtable(s,   i, last, head, tail) {
    last = 0
    for (i = length(s); i >= 1; i--) {
        if (substr(s, i, 1) == ".") { last = i; break }
    }
    if (last == 0) return ""
    head = substr(s, 1, last - 1) "]"
    tail = substr(s, last + 1)
    sub(/\]$/, "", tail)
    if (tail !~ /^[A-Za-z0-9_-]+$/) return ""
    if (!is_dep_sec(head)) return ""
    return tail
}

# Classify a declaration's right-hand side (or a sub-table's accumulated fields).
# "" == legal.
#
# A `version` NEXT TO a `path` is legal and common: cargo builds from the path
# and uses the version only when publishing. The defect is a version with no
# local source at all.
function verdict(val) {
    if (val ~ /workspace[[:space:]]*=[[:space:]]*true/) return ""
    if (val ~ /(^|[^A-Za-z0-9_])path[[:space:]]*=/)     return ""
    if (val ~ /(^|[^A-Za-z0-9_])git[[:space:]]*=/)      return "GIT"
    if (val ~ /^[[:space:]]*["']/)                      return "REGISTRY"
    if (val ~ /(^|[^A-Za-z0-9_])version[[:space:]]*=/)  return "REGISTRY"
    return ""
}

# `compute = { package = "trueno", version = "0.16" }` renames the key, so the
# key alone would not match the name set. The `package` value is the real crate.
function renamed_pkg(val,   v) {
    if (val !~ /(^|[^A-Za-z0-9_])package[[:space:]]*=/) return ""
    v = val
    sub(/^.*[^A-Za-z0-9_]package[[:space:]]*=[[:space:]]*/, "", v)
    sub(/^[[:space:]]*package[[:space:]]*=[[:space:]]*/, "", v)
    gsub(/^["']/, "", v)
    sub(/["'].*$/, "", v)
    return v
}

function report(name, lineno, cls, text) {
    printf "%s\t%s\t%s\t%s\t%s\n", FILE, lineno, name, cls, text
}

# A `[dependencies.x]` sub-table is only decidable once every field has been
# read, i.e. at the next section header or at EOF.
function flush_sub(   cls, pkg) {
    if (sub_name == "") return
    cls = verdict(sub_body)
    pkg = renamed_pkg(sub_body)
    if (cls != "" && (sub_name in intree || (pkg != "" && pkg in intree)))
        report(sub_name, sub_line, cls, "[dependencies." sub_name "]" sub_body)
    sub_name = ""; sub_body = ""; sub_line = 0
}

BEGIN {
    while ((getline n < NAMEFILE) > 0) if (n != "") intree[n] = 1
    close(NAMEFILE)
    sec = ""; sub_name = ""; sub_body = ""; sub_line = 0; depsec = 0
}

{ line = strip_comment($0) }

line ~ /^[[:space:]]*$/ { next }

line ~ /^[[:space:]]*\[/ {
    flush_sub()
    hdr = line
    gsub(/[[:space:]]/, "", hdr)
    sec = hdr
    depsec = is_dep_sec(hdr)
    if (!depsec) {
        s = dep_subtable(hdr)
        if (s != "") { sub_name = s; sub_body = ""; sub_line = FNR }
    }
    next
}

sub_name != "" { sub_body = sub_body " " line; next }

depsec == 1 {
    t = line
    sub(/^[[:space:]]*/, "", t)
    if (t !~ /^[A-Za-z0-9_.-]+[[:space:]]*=/) next

    key = t
    sub(/[[:space:]]*=.*$/, "", key)
    rhs = t
    sub(/^[A-Za-z0-9_.-]+[[:space:]]*=[[:space:]]*/, "", rhs)

    # Dotted-key form: `serde.workspace = true`, `serde.version = "1"`.
    if (key ~ /\./) {
        field = key
        sub(/^[^.]*\./, "", field)
        sub(/\..*$/, "", key)
        rhs = field " = " rhs
    }

    cls = verdict(rhs)
    if (cls == "") next
    pkg = renamed_pkg(rhs)
    if (key in intree || (pkg != "" && pkg in intree))
        report(key, FNR, cls, t)
}

END { flush_sub() }
