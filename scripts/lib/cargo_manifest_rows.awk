# cargo_manifest_rows.awk — the Cargo.toml keys check_binary_debt.sh judges, as rows.
#
# The guard used to read manifests with python's tomllib. The clean-room runners carry
# python 3.10, which has no tomllib, so on every one of them the guard printed UNMEASURED
# and exited 0: it never judged a tree in CI. This reader needs only awk.
#
# It reads a SUBSET of TOML, and only the keys the guard uses. Anything else in those
# keys is REFUSED (a "REFUSE file:line: why" line on stderr, exit 2), never guessed:
#
#   [package]           name = "<s>"   autobins = true|false   version = "<s>"
#   [package.metadata]  cargo-fuzz = true|false
#   [[bin]]             name = "<s>"   path = "<s>"
#   [workspace]         members = [ "<s>", ... ]   exclude = [ "<s>", ... ]   (multi-line, # comments)
#   [workspace.package] version = "<s>"
#
# "<s>" is a one-line basic string with no escapes, or a one-line literal string.
# Multi-line strings and arrays in OTHER keys are skipped whole, so a "[[bin]]" line inside
# a description can never be read as a table header.
#
# Output, tab-separated, one per line:
#   haspkg | name V | autobins V | pkgversion V | pkgversion_other | wsversion V | fuzz V
#   members V | exclude V | bintable N | bin N name V | bin N path V          (N counts [[bin]] tables from 1)

function refuse(why) {
    printf "REFUSE %s:%d: %s\n", FILENAME, FNR, why > "/dev/stderr"
    bad = 1
    exit 2
}

# scan(s): walk s once, honouring "..." and '...' strings. Sets CODE (s with any trailing
# comment removed), DEPTH_DELTA ([ minus ], outside strings), OPEN_ML (a """ or ''' was left
# open). Basic strings with a backslash are refused by the callers that read them.
function scan(s,    i, c, n, q, out) {
    CODE = ""; DEPTH_DELTA = 0; OPEN_ML = ""
    n = length(s); q = ""; out = ""
    for (i = 1; i <= n; i++) {
        c = substr(s, i, 1)
        if (q == "") {
            if (substr(s, i, 3) == "\"\"\"" || substr(s, i, 3) == "'''") {
                q = substr(s, i, 3); out = out q; i += 2; continue
            }
            if (c == "#") break
            if (c == "\"" || c == "'") q = c
            else if (c == "[") DEPTH_DELTA++
            else if (c == "]") DEPTH_DELTA--
        } else if (length(q) == 3) {
            if (substr(s, i, 3) == q) { out = out q; i += 2; q = ""; continue }
        } else if (q == "\"" && c == "\\") {
            out = out c; i++; c = substr(s, i, 1)
        } else if (c == q) {
            q = ""
        }
        out = out c
    }
    if (length(q) == 3) OPEN_ML = q
    CODE = out
}

function trim(s) { sub(/^[ \t]+/, "", s); sub(/[ \t]+$/, "", s); return s }

# str(v): the value of a one-line string, or refuse.
function str(v, key) {
    v = trim(v)
    if (v ~ /^"[^"\\]*"$/ || v ~ /^'[^']*'$/) return substr(v, 2, length(v) - 2)
    refuse(key " is not a one-line string without escapes: " v)
}

function boolean(v, key) {
    v = trim(v)
    if (v == "true" || v == "false") return v
    refuse(key " is not true or false: " v)
}

# A [workspace] array: the strings in it, and nothing else.
function emit_array(kind, body,    rest, m) {
    rest = body
    sub(/^[ \t]*\[/, "", rest); sub(/\][ \t]*$/, "", rest)
    while (match(rest, /^[ \t,]*("[^"\\]*"|'[^']*')/)) {
        m = substr(rest, RSTART, RLENGTH)
        sub(/^[ \t,]*/, "", m)
        print kind "\t" substr(m, 2, length(m) - 2)
        rest = substr(rest, RSTART + RLENGTH)
    }
    if (rest !~ /^[ \t,]*$/) refuse(kind " array holds something other than strings: " trim(rest))
}

BEGIN { table = ""; bins = 0; ml = ""; depth = 0; acc = ""; acc_key = "" }

{
    line = $0
    # Inside a multi-line string another key opened: skip to its close.
    if (ml != "") {
        if (index(line, ml) == 0) next
        line = substr(line, index(line, ml) + 3); ml = ""
        scan(line); if (OPEN_ML != "") ml = OPEN_ML
        next
    }
    scan(line)
    # Inside a multi-line array: accumulate members/exclude, skip any other key's array.
    if (depth > 0) {
        depth += DEPTH_DELTA
        if (acc_key != "") acc = acc " " CODE
        if (depth == 0 && acc_key != "") { emit_array(acc_key, acc); acc_key = ""; acc = "" }
        next
    }
    code = trim(CODE)
    if (code == "") next

    if (code ~ /^\[/) {
        h = code; gsub(/[ \t]/, "", h)
        # A quoted FIRST key could spell package, bin or workspace: refuse it. A quote later
        # in the header ([target.'cfg(unix)'.dependencies]) is some other table.
        if (h ~ /^\[\[?["']/) refuse("quoted table header: " code)
        if (h ~ /["']/) { table = "other"; next }
        if (h == "[[bin]]") { table = "bin"; bins++; print "bintable\t" bins; next }
        if (h ~ /^\[\[/) { table = "other"; next }
        table = substr(h, 2, length(h) - 2)
        if (table == "package") print "haspkg"
        next
    }

    eq = index(code, "=")
    if (eq == 0) refuse("not a key = value line: " code)
    key = trim(substr(code, 1, eq - 1)); val = trim(substr(code, eq + 1))
    if (OPEN_ML != "") ml = OPEN_ML

    if (table == "" && (key ~ /^(package|bin|workspace)([.]|$)/)) refuse("top-level " key " is a form this reader does not read")

    if (table == "package") {
        if (key == "name") print "name\t" str(val, "package.name")
        else if (key == "autobins") print "autobins\t" boolean(val, "package.autobins")
        else if (key == "version") {
            if (val ~ /^["']/) print "pkgversion\t" str(val, "package.version"); else print "pkgversion_other"
        }
        else if (key ~ /^version[.]/) print "pkgversion_other"
        else if (key ~ /^(name|autobins)[.]/ || key ~ /^metadata[.]"?cargo-fuzz/ || (key == "metadata" && val ~ /cargo-fuzz/))
            refuse("package." key " is a form this reader does not read")
    } else if (table == "package.metadata") {
        if (key == "cargo-fuzz") print "fuzz\t" boolean(val, "package.metadata.cargo-fuzz")
        else if (key ~ /^"?cargo-fuzz/) refuse("package.metadata." key " is a form this reader does not read")
    } else if (table == "bin") {
        if (key == "name") print "bin\t" bins "\tname\t" str(val, "bin.name")
        else if (key == "path") print "bin\t" bins "\tpath\t" str(val, "bin.path")
    } else if (table == "workspace") {
        if (key == "members" || key == "exclude") {
            if (val !~ /^\[/) refuse("workspace." key " is not an array")
            if (DEPTH_DELTA > 0) { depth = DEPTH_DELTA; acc_key = key; acc = val; next }
            emit_array(key, val)
        }
    } else if (table == "workspace.package") {
        if (key == "version") print "wsversion\t" str(val, "workspace.package.version")
    }
    # Any other key's multi-line array is skipped whole.
    if (DEPTH_DELTA > 0) { depth = DEPTH_DELTA; acc_key = "" }
}
