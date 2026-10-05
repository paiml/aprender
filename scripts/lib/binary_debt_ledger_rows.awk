# binary_debt_ledger_rows.awk — the binary-debt ledger's four sections, as rows.
#
# check_binary_debt.sh read the ledger with python's yaml. This reads the one shape the
# ledger is written in, and REFUSES anything else in those sections (a "REFUSE file:line:
# why" line on stderr, exit 2), so a ledger edited into another YAML form turns the guard
# RED instead of being half-read:
#
#   classes: [A, B + C, ...]                 a one-line flow sequence of plain scalars
#   ceilings:
#     current: {binary_debt: N, legacy_names: N}
#     releases:
#       - {release: X.Y.Z, binary_debt: N, legacy_names: N, armed: true|false}
#   legacy_names:
#     - {name: S, sunset: null|S}
#   binaries:
#     - {crate: S, bin: S, class: S, ...}    every other key is carried and ignored
#
# Values may be plain or "double"/'single' quoted; commas inside quotes are kept.
# Output, tab-separated:
#   class C | current BD LN | release R BD LN ARMED | legacy NAME SUNSET | row CRATE BIN CLASS
# SUNSET is empty when python reads it as falsy (absent, null, ~, "", false). ARMED is true only
# for an UNQUOTED YAML 1.1 true (PyYAML reads it as bool True; the guard tests `is True`).

function refuse(why) {
    printf "REFUSE %s:%d: %s\n", FILENAME, FNR, why > "/dev/stderr"
    refused = 1
    exit 2
}

function trim(s) { sub(/^[ \t]+/, "", s); sub(/[ \t]+$/, "", s); return s }

function unquote(v) {
    v = trim(v); QUOTED = 0
    if (v ~ /^"[^"\\]*"$/ || v ~ /^'[^']*'$/) { QUOTED = 1; return substr(v, 2, length(v) - 2) }
    if (v ~ /^["']/) refuse("a quoted value this reader does not read: " v)
    return v
}

# flow(s): parse "{k: v, k: v}" (an optional trailing # comment) into F[k], and FQ[k] = 1 when that
# value was quoted (PyYAML reads a quoted "true" or "null" as a string, not a bool or null); refuse otherwise.
function flow(s,    i, c, n, q, part, parts, np, k, colon, key) {
    split("", F); split("", FQ)
    s = trim(s)
    if (s !~ /^\{/) refuse("not a one-line flow map: " s)
    n = length(s); q = ""; part = ""; np = 0
    for (i = 2; i <= n; i++) {
        c = substr(s, i, 1)
        if (q == "") {
            if (c == "\"" || c == "'") q = c
            else if (c == "," ) { parts[++np] = part; part = ""; continue }
            else if (c == "}") break
            else if (c == "{" || c == "[") refuse("a nested collection in a flow map: " s)
        } else if (c == q) q = ""
        part = part c
    }
    if (i > n) refuse("an unclosed flow map: " s)
    if (trim(substr(s, i + 1)) !~ /^(#.*)?$/) refuse("text after a flow map: " s)
    if (trim(part) != "") parts[++np] = part
    for (k = 1; k <= np; k++) {
        colon = index(parts[k], ":")
        if (colon == 0) refuse("a flow-map entry without a key: " parts[k])
        key = trim(substr(parts[k], 1, colon - 1))
        F[key] = unquote(substr(parts[k], colon + 1)); FQ[key] = QUOTED
    }
}

function need(key, where) { if (!(key in F)) refuse(where " row has no " key) }

# An integer field: an UNQUOTED decimal with no leading zero. PyYAML reads 010 as octal 8 and 08
# or '05' as a string, so those are refused rather than read as decimal.
function int_of(key, what,    v) {
    v = F[key]
    if (FQ[key] || v !~ /^(0|[1-9][0-9]*)$/) refuse(what " is not an unquoted non-negative integer: " v)
    return v + 0
}

# A string field written plain must be one PyYAML also reads as a string: not a bool or null word,
# and not a number, date or other form starting with a digit or sign (an X.Y.Z version is a string).
function typed_word(v) {
    return v ~ /^(true|True|TRUE|false|False|FALSE|yes|Yes|YES|no|No|NO|on|On|ON|off|Off|OFF|null|Null|NULL|~)$/ ||
        (v ~ /^[-+.0-9]/ && v !~ /^[0-9]+\.[0-9]+\.[0-9]+([-+][0-9A-Za-z.-]*)?$/)
}
function plain_str(key, where) {
    if (!FQ[key] && typed_word(F[key])) refuse(where "." key " is not a string to YAML (quote it): " F[key])
    return F[key]
}

# armed binds only when PyYAML reads a bool True: an UNQUOTED YAML 1.1 true word (the guard tests `is True`).
function yaml_true(key) { return !FQ[key] && F[key] ~ /^(true|True|TRUE|yes|Yes|YES|on|On|ON)$/ }

/^[ \t]*(#.*)?$/ { next }

/^[^ \t-]/ {
    sec = ""; sub_ = ""
    if ($0 ~ /^classes:/) {
        v = $0; sub(/^classes:[ \t]*/, "", v); sub(/[ \t]+#.*$/, "", v); v = trim(v)
        if (v !~ /^\[[^]\[{}"']*\]$/) refuse("classes is not a one-line flow sequence of plain scalars: " v)
        v = substr(v, 2, length(v) - 2); nc = split(v, cs, ",")
        for (j = 1; j <= nc; j++) if (trim(cs[j]) != "") {
            if (typed_word(trim(cs[j]))) refuse("a class is not a string to YAML: " trim(cs[j]))
            print "class\t" trim(cs[j]); ncls++
        }
        if (!ncls) refuse("classes is empty")
        seen["classes"] = 1
    } else if ($0 ~ /^ceilings:[ \t]*(#.*)?$/) { sec = "ceilings"; seen["ceilings"] = 1 }
    else if ($0 ~ /^legacy_names:[ \t]*(#.*)?$/) sec = "legacy"
    else if ($0 ~ /^binaries:[ \t]*(#.*)?$/) { sec = "binaries"; seen["binaries"] = 1 }
    else if ($0 ~ /^(classes|ceilings|legacy_names|binaries):/) refuse("a section in a form this reader does not read: " $0)
    next
}

sec == "ceilings" {
    if ($0 ~ /^  current:/) {
        v = $0; sub(/^  current:/, "", v); flow(v)
        need("binary_debt", "ceilings.current"); need("legacy_names", "ceilings.current")
        print "current\t" int_of("binary_debt", "ceilings.current.binary_debt") "\t" int_of("legacy_names", "ceilings.current.legacy_names")
        seen["current"] = 1
    } else if ($0 ~ /^  releases:[ \t]*(#.*)?$/) sub_ = "releases"
    else if ($0 ~ /^    - / && sub_ == "releases") {
        v = $0; sub(/^    - /, "", v); flow(v)
        need("release", "ceilings.releases"); need("binary_debt", "ceilings.releases"); need("legacy_names", "ceilings.releases")
        print "release\t" plain_str("release", "ceilings.releases") "\t" int_of("binary_debt", "release binary_debt") "\t" int_of("legacy_names", "release legacy_names") "\t" (yaml_true("armed") ? "true" : "false")
    } else refuse("a ceilings line this reader does not read: " $0)
    next
}

sec == "legacy" {
    v = $0; sub(/^  - /, "", v); flow(v)
    if (F["name"] == "") refuse("a legacy_names row with no or an empty name: " v)
    s = F["sunset"]
    # The guard counts a row whose sunset is falsy to python: absent, null, "", false. An unquoted
    # value that YAML could read as a number (0, 0.0, 1e3) is refused rather than guessed.
    if (!FQ["sunset"]) {
        if (s ~ /^(null|Null|NULL|~|false|False|FALSE|no|No|NO|off|Off|OFF)$/) s = ""
        else if (s ~ /^[-+.0-9]/ && s !~ /^[0-9]+\.[0-9]+\.[0-9]+([-+].*)?$/) refuse("a sunset YAML may read as a number: " s)
    }
    print "legacy\t" plain_str("name", "legacy_names") "\t" s
    next
}

sec == "binaries" {
    v = $0; sub(/^  - /, "", v); flow(v)
    # flow() refuses any line that is not "  - {...}". A missing key reads as "", so one test covers both.
    if (F["crate"] == "" || F["bin"] == "" || F["class"] == "") refuse("a binaries row with no or an empty crate, bin or class: " v)
    print "row\t" plain_str("crate", "binaries") "\t" plain_str("bin", "binaries") "\t" plain_str("class", "binaries")
    next
}

END {
    if (refused) exit 2
    if (FNR == 0) refuse("the ledger is empty")
    if (!("classes" in seen)) refuse("no classes")
    if (!("binaries" in seen)) refuse("no binaries")
    if (!("current" in seen)) refuse("no ceilings.current")
}
