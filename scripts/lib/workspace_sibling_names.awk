# workspace_sibling_names.awk — emit the names a manifest serves IN-TREE.
#
# Driven by scripts/check_workspace_siblings_pathed.sh. Run with -v ROOT=1 for
# the workspace root manifest, -v ROOT=0 for a member.
#
# Three sources, and the second is the load-bearing one:
#
#   [package] name   aprender-compute   — what cargo calls the package
#   [lib] name       trueno             — what `use` sees, and what a crates.io
#                                         crate would COLLIDE with
#   root [workspace.dependencies] keys that carry a `path =` — in-tree by
#                                         construction, whatever they are named
#
# Package and lib names diverge throughout this tree (aprender-compute/trueno,
# aprender-db/trueno_db, aprender-serve/realizar), so a name set built from
# `[package] name` alone would not contain "trueno" and would sail straight past
# `trueno = "0.16"` — the exact declaration the guard exists to stop.
#
# Every name is emitted in BOTH spellings: Rust lib names use `_`, the crates.io
# package they shadow publishes with `-` (lib `trueno_db` vs crate `trueno-db`).

function unquote(line,   v) {
    v = line
    sub(/^[^=]*=[[:space:]]*/, "", v)
    sub(/[[:space:]]*#.*$/, "", v)
    gsub(/^["']|["'][[:space:]]*$/, "", v)
    return v
}

function emit(n,   a, b) {
    if (n == "") return
    a = n; gsub(/_/, "-", a)
    b = n; gsub(/-/, "_", b)
    print a
    print b
}

/^[[:space:]]*#/  { next }

/^[[:space:]]*\[/ {
    sec = $0
    sub(/[[:space:]]*#.*$/, "", sec)
    gsub(/[[:space:]]/, "", sec)
    next
}

sec == "[package]" && /^[[:space:]]*name[[:space:]]*=/ { emit(unquote($0)); next }
sec == "[lib]"     && /^[[:space:]]*name[[:space:]]*=/ { emit(unquote($0)); next }

sec == "[workspace.dependencies]" && ROOT == 1 \
    && /^[[:space:]]*[A-Za-z0-9_-]+[[:space:]]*=/ && /path[[:space:]]*=/ {
    k = $0
    sub(/^[[:space:]]*/, "", k)
    sub(/[[:space:]]*=.*$/, "", k)
    emit(k)
}
