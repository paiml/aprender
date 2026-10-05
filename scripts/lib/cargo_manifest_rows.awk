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
function scan(s,    i, c, n, q, out, e) {
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
            if (q == "\"\"\"" && c == "\\") { out = out c; i++; c = substr(s, i, 1) }
            else if (substr(s, i, 3) == q) { e = ml_quotes(s, i, q); out = out substr(s, i, e); i += e - 1; q = ""; continue }
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

# ml_quotes(s, i, q): the length of the closing delimiter at s[i]: q, plus up to two more of its quote
# character, which TOML reads as content ("""a"""" is the string a").
function ml_quotes(s, i, q,    e) {
    e = 3
    while (e < 5 && substr(s, i + e, 1) == substr(q, 1, 1)) e++
    return e
}

# ml_end(s, q): the position just past the """ or ''' that closes a multi-line string open
# before s, or 0 when s does not close it. In a """ string a backslash escapes the next character.
function ml_end(s, q,    i, n) {
    n = length(s)
    for (i = 1; i <= n; i++) {
        if (q == "\"\"\"" && substr(s, i, 1) == "\\") { i++; continue }
        if (substr(s, i, 3) == q) return i + ml_quotes(s, i, q)
    }
    return 0
}

function trim(s) { sub(/^[ \t]+/, "", s); sub(/[ \t]+$/, "", s); return s }

# str(v): the value of a one-line string, or refuse.
function str(v, key) {
    v = trim(v)
    # A tab is legal inside a TOML string, but the rows are tab-separated.
    if (index(v, "\t")) refuse(key " holds a tab: " v)
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
        if (index(m, "\t")) refuse(kind " array holds a tab: " m)
        print kind "\t" substr(m, 2, length(m) - 2)
        rest = substr(rest, RSTART + RLENGTH)
    }
    if (rest !~ /^[ \t,]*$/) refuse(kind " array holds something other than strings: " trim(rest))
}

BEGIN { table = ""; bins = 0; ml = ""; depth = 0; acc = ""; acc_key = "" }

{
    line = $0
    # A CRLF line ending is a newline to TOML; any other carriage return is not TOML.
    sub(/\r$/, "", line)
    if (index(line, "\r")) refuse("a carriage return inside a line")
    # Inside a multi-line string another key opened: skip to its close, then read the rest of
    # the line as usual (inside an array that rest can hold the closing ]).
    if (ml != "") {
        p = ml_end(line, ml)
        if (p == 0) next
        line = substr(line, p); ml = ""
    }
    scan(line)
    # Inside a multi-line array: accumulate members/exclude, skip any other key's array.
    if (depth > 0) {
        depth += DEPTH_DELTA
        if (OPEN_ML != "") ml = OPEN_ML
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
        # A quote in the SECOND key ([package."metadata"], [workspace."package"]) could spell a table this
        # reader reads; a quote deeper ([package.metadata."docs.rs"]) is some other table.
        if (h ~ /^\[\[?(package|workspace|bin)[.]["']/) refuse("quoted key in a package, workspace or bin header: " code)
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
    # TOML allows whitespace around the dots of a dotted key: "metadata . cargo-fuzz" is metadata.cargo-fuzz.
    gsub(/[ \t]*[.][ \t]*/, ".", key)
    if (OPEN_ML != "") ml = OPEN_ML

    if (table == "" && (key ~ /^["']?(package|bin|workspace)["']?([.]|$)/)) refuse("top-level " key " is a form this reader does not read")
    # A quoted key, or a dotted name/path/autobins/members/exclude, in a table this reader reads could spell a key it
    # reads under another form; refuse it rather than skip it.
    if (table ~ /^(package|package[.]metadata|bin|workspace|workspace[.]package)$/ && (key ~ /["']/ || key ~ /^(name|path|autobins|members|exclude)[ \t]*[.]/))
        refuse(table "." key " is a form this reader does not read")

    if (table == "package") {
        if (key == "name") print "name\t" str(val, "package.name")
        else if (key == "autobins") print "autobins\t" boolean(val, "package.autobins")
        else if (key == "version") {
            if (val ~ /^["']/) print "pkgversion\t" str(val, "package.version"); else print "pkgversion_other"
        }
        else if (key ~ /^version[.]/) print "pkgversion_other"
        else if (key ~ /^(name|autobins)[.]/ || key ~ /^metadata[.]["']?cargo-fuzz/ || (key == "metadata" && val ~ /cargo-fuzz/))
            refuse("package." key " is a form this reader does not read")
    } else if (table == "package.metadata") {
        if (key == "cargo-fuzz") print "fuzz\t" boolean(val, "package.metadata.cargo-fuzz")
    } else if (table == "bin") {
        if (key == "name") print "bin\t" bins "\tname\t" str(val, "bin.name")
        else if (key == "path") print "bin\t" bins "\tpath\t" str(val, "bin.path")
    } else if (table == "workspace") {
        if (key ~ /^["']?package([.]|["']?$)/) refuse("workspace." key " is a form this reader does not read")
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
