# ci_guard_yaml.awk -- the block-YAML reader behind ci_guard_steps.sh (#4415).
# JSON on stdout; rc 2 + "line N: why" on stderr for anything outside the subset
# (see the header of ci_guard_steps.sh).
function refuse(i, why) {
    if (!ERR) printf "line %d: %s\n", i, why > "/dev/stderr"
    ERR = 1
    exit 2
}
function ind(s) { match(s, /^ */); return RLENGTH }
function skip(s) { return s ~ /^ *$/ || s ~ /^ *#/ }
function sig(i) { while (i <= N && skip(L[i])) i++; return i }
function js(s,   out, p, n, ch) {
    # a character walk: awks disagree on backslashes in a gsub replacement
    # (mawk turns "\\\\" into one), so no replacement string is used
    out = ""; n = length(s)
    for (p = 1; p <= n; p++) {
        ch = substr(s, p, 1)
        if (ch == "\\") out = out "\\\\"
        else if (ch == "\"") out = out "\\\""
        else if (ch == "\n") out = out "\\n"
        else if (ch == "\t") out = out "\\t"
        else if (ch == "\r") out = out "\\r"
        else out = out ch
    }
    return "\"" out "\""
}
function isdash(c) { return c ~ /^-( |$)/ }
# KEY/REST from a mapping line body c; 0 when c is not "key: ..." / "key:"
function splitkey(c, i,   p, q) {
    if (c ~ /^["\x27]/) {
        q = substr(c, 1, 1)
        p = index(substr(c, 2), q)
        if (!p) return 0
        if (substr(c, p + 2) !~ /^:( |$)/) return 0
        KEY = substr(c, 2, p - 1); REST = substr(c, p + 3)
        if (q == "\"" && KEY ~ /\\/) refuse(i, "escape in a double-quoted key")
        if (q == "\x27") gsub(/\x27\x27/, "\x27", KEY)
    } else {
        if (c ~ /^[\[\{|>&*!%@`?#]/) return 0
        p = match(c, /:( |$)/)
        if (!p) return 0
        KEY = substr(c, 1, p - 1); REST = substr(c, p + 1)
        if (KEY ~ / #/) return 0
        sub(/ +$/, "", KEY)
        if (KEY == "<<") refuse(i, "merge key <<")
    }
    sub(/^ +/, "", REST)
    return 1
}
# strip " # comment" / leading-space from what follows a closed scalar
function tail_ok(t) { return t ~ /^ *$/ || t ~ /^ +#/ }
function plain_json(s) {
    if (s ~ /^(~|null|Null|NULL)$/) return "null"
    if (s ~ /^(true|True|TRUE)$/) return "true"
    if (s ~ /^(false|False|FALSE)$/) return "false"
    if (s ~ /^-?(0|[1-9][0-9]*)$/) return s
    return js(s)
}
# one scalar s that is complete on line i: plain, single- or double-quoted
function scalar(s, i,   q, p, out, ch, n, t) {
    if (s ~ /^[&*!%@`]/) refuse(i, "anchor, alias, tag or reserved indicator: " s)
    if (s ~ /^\x27/) {
        out = ""; p = 2; n = length(s)
        while (1) {
            if (p > n) refuse(i, "unclosed single-quoted scalar (multi-line scalars are not read)")
            ch = substr(s, p, 1)
            if (ch == "\x27") {
                if (substr(s, p + 1, 1) == "\x27") { out = out "\x27"; p += 2; continue }
                break
            }
            out = out ch; p++
        }
        SCTAIL = substr(s, p + 1)
        return js(out)
    }
    if (s ~ /^"/) {
        out = ""; p = 2; n = length(s)
        while (1) {
            if (p > n) refuse(i, "unclosed double-quoted scalar (multi-line scalars are not read)")
            ch = substr(s, p, 1)
            if (ch == "\"") break
            if (ch == "\\") {
                t = substr(s, p + 1, 1)
                if (t == "n") out = out "\n"
                else if (t == "t") out = out "\t"
                else if (t == "r") out = out "\r"
                else if (t == "\\" || t == "\"" || t == "/") out = out t
                else refuse(i, "unsupported escape \\" t)
                p += 2; continue
            }
            out = out ch; p++
        }
        SCTAIL = substr(s, p + 1)
        return js(out)
    }
    SCTAIL = ""
    p = match(s, / #/)
    if (p) s = substr(s, 1, p - 1)
    sub(/ +$/, "", s)
    if (s ~ /: / || s ~ /:$/) refuse(i, "a mapping inside a plain scalar: " s)
    return plain_json(s)
}
function flowseq(s, i,   out, p, n, ch, item, q, first, depth) {
    out = "["; first = 1; p = 2; n = length(s)
    while (1) {
        while (substr(s, p, 1) == " ") p++
        if (p > n) refuse(i, "unclosed [ flow list (multi-line flow is not read)")
        ch = substr(s, p, 1)
        if (ch == "]") { p++; break }
        if (ch == "[" || ch == "{") refuse(i, "nested flow collection")
        if (ch == "\x27" || ch == "\"") {
            item = scalar(substr(s, p), i)
            p = n - length(SCTAIL) + 1
        } else {
            q = p
            while (p <= n && substr(s, p, 1) !~ /[],]/) p++
            item = substr(s, q, p - q); sub(/ +$/, "", item)
            if (item ~ /^[&*!%@`#]/ || item ~ /: /) refuse(i, "unsupported flow item: " item)
            item = plain_json(item)
        }
        out = out (first ? "" : ",") item; first = 0
        while (substr(s, p, 1) == " ") p++
        ch = substr(s, p, 1)
        if (ch == ",") { p++; continue }
        if (ch == "]") { p++; break }
        refuse(i, "bad flow list")
    }
    if (!tail_ok(substr(s, p))) refuse(i, "text after ]")
    return out "]"
}
function block(i, n, hdr,   chomp, j, bi, out, k, last, line, li) {
    if (hdr ~ /^>/) refuse(i, "folded > scalar")
    chomp = substr(hdr, 2, 1)
    if (chomp ~ /[0-9]/ || substr(hdr, 3, 1) ~ /[0-9]/) refuse(i, "block indentation indicator")
    if (chomp != "-" && chomp != "+") chomp = ""
    if (!tail_ok(substr(hdr, chomp == "" ? 2 : 3))) refuse(i, "text after a block scalar header")
    bi = -1; k = 0; j = i + 1
    while (j <= N) {
        line = L[j]
        if (line ~ /^ *$/) { B[++k] = (bi >= 0 && length(line) > bi) ? substr(line, bi + 1) : ""; j++; continue }
        li = ind(line)
        if (li <= n) break
        if (bi < 0) bi = li
        if (li < bi) { if (line ~ /^ *#/) break; refuse(j, "block scalar line less indented than its first line") }
        B[++k] = substr(line, bi + 1); j++
    }
    NEXT = j
    last = k
    if (chomp != "+") while (last > 0 && B[last] == "") last--
    out = ""
    for (j = 1; j <= last; j++) out = out B[j] (j < last ? "\n" : "")
    if (chomp == "+") { out = ""; for (j = 1; j <= k; j++) out = out B[j] "\n" }
    else if (chomp == "" && last > 0) out = out "\n"
    return js(out)
}
# the value of a key / item on line i whose own indent is n; rest = text after "key:" / "- "
function value(rest, i, n, seqok,   j, v) {
    if (rest == "" || rest ~ /^#/) {
        j = sig(i + 1)
        if (j <= N && ind(L[j]) > n) return node(j, ind(L[j]))
        if (seqok && j <= N && ind(L[j]) == n && isdash(substr(L[j], n + 1))) return seq(j, n)
        NEXT = i + 1
        return "null"
    }
    if (rest ~ /^[|>]/) return block(i, n, rest)
    if (rest ~ /^\[/) v = flowseq(rest, i)
    else if (rest ~ /^\{/) {
        if (rest !~ /^\{ *\}/ || !tail_ok(substr(rest, index(rest, "}") + 1))) refuse(i, "flow mapping { } (only {} is read)")
        v = "{}"
    } else {
        v = scalar(rest, i)
        if (!tail_ok(SCTAIL)) refuse(i, "text after a quoted scalar")
    }
    j = sig(i + 1)
    if (j <= N && ind(L[j]) > n) refuse(j, "continuation line (multi-line scalars are not read)")
    NEXT = i + 1
    return v
}
function node(i, n,   c) {
    c = substr(L[i], n + 1)
    if (isdash(c)) return seq(i, n)
    if (splitkey(c, i)) return map(i, n)
    refuse(i, "expected a key: value or a - item")
}
function map(i, n,   out, first, id, c, li) {
    out = "{"; first = 1; id = ++MAPS
    while (1) {
        i = sig(i)
        if (i > N) break
        li = ind(L[i])
        if (li < n) break
        if (li > n) refuse(i, "unexpected indentation")
        c = substr(L[i], n + 1)
        if (isdash(c)) {
            if (n == 0) refuse(i, "a - item beside top-level keys")
            break
        }
        if (!splitkey(c, i)) refuse(i, "expected key: value")
        if ((id, KEY) in SEEN) refuse(i, "duplicate key " KEY)
        SEEN[id, KEY] = 1
        out = out (first ? "" : ",") js(KEY) ":"; first = 0
        out = out value(REST, i, n, 1)
        i = NEXT
    }
    NEXT = i
    return out "}"
}
function seq(i, n,   out, first, c, rest, m, li) {
    out = "["; first = 1
    while (1) {
        i = sig(i)
        if (i > N) break
        li = ind(L[i])
        if (li < n) break
        if (li > n) refuse(i, "unexpected indentation")
        c = substr(L[i], n + 1)
        if (!isdash(c)) break
        rest = substr(c, 2)
        m = n + 1 + ind(rest)
        sub(/^ +/, "", rest)
        out = out (first ? "" : ","); first = 0
        if (rest == "" || rest ~ /^#/) out = out value(rest, i, n, 0)
        else if (isdash(rest)) refuse(i, "nested - - item")
        else if (splitkey(rest, i)) {
            L[i] = sprintf("%" m "s", "") rest
            out = out map(i, m)
        } else out = out value(rest, i, n, 0)
        i = NEXT
    }
    NEXT = i
    return out "]"
}
{
    if ($0 ~ /^ *\t/) { printf "line %d: tab in indentation\n", NR > "/dev/stderr"; ERR = 1; exit 2 }
    sub(/\r$/, "")
    L[++N] = $0
}
END {
    if (ERR) exit 2
    i = sig(1)
    if (i <= N && L[i] ~ /^---( |$)/) i = sig(i + 1)
    if (i > N) { print "{}"; exit 0 }
    if (ind(L[i]) != 0) refuse(i, "the document must be a top-level mapping")
    out = map(i, 0)
    if (sig(NEXT) <= N) refuse(sig(NEXT), "content after the top-level mapping (--- or ...)")
    print out
}
