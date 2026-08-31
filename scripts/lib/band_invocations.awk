# band_invocations.awk -- emit the logical lines of a shell/Make file in which
# `test llm bench ` stands at QUOTE DEPTH ZERO, i.e. where the shell would
# EXECUTE it.
#
# Extracted from scripts/check_band_client_tokenizer.sh so the guard reads as
# shell. Inlined, the program's own braces and quotes also defeated `bashrs`,
# which lost the enclosing function boundary and reported SC2168 on a `local`
# that was correctly inside one.
#
# WHY QUOTE DEPTH AND NOT A SUBSTRING. Three drafts of that guard reddened on
# their own error strings: `fail "... an `apr test llm bench --band` ..."` is a
# message and `printf '%s\n' '"$APR" test llm bench ...'` is a datum. Exempting
# the file by name is how a guard stops seeing the file it most needs to watch,
# so this asks the only question that separates them. State CARRIES ACROSS
# LINES, because the second physical line of a multi-line message is still
# inside its opening quote.

# Quote state CARRIES ACROSS LINES. Without that, the second physical
# line of a multi-line "..." message reads as depth 0 and a sentence
# about the flag becomes an invocation of it -- which is exactly how the
# first two drafts of this guard reddened on their own error strings.
function walk(s,   i, c, n, hit) {
    n = length(s)
    for (i = 1; i <= n; i++) {
        c = substr(s, i, 1)
        if (c == "\\" && !SQ) { i++; continue }
        if (c == "\x27" && !DQ) { SQ = !SQ; continue }
        if (c == "\"" && !SQ) { DQ = !DQ; continue }
        if (!SQ && !DQ && substr(s, i, 15) == "test llm bench ") hit = 1
    }
    return hit
}
!SQ && !DQ && !inhd && /^[[:space:]]*#/ { next }
!SQ && !DQ && !inhd && /<<-?[\x27"][A-Za-z_][A-Za-z0-9_]*[\x27"]/ {
    match($0, /<<-?[\x27"][A-Za-z_][A-Za-z0-9_]*[\x27"]/)
    tag = substr($0, RSTART, RLENGTH)
    gsub(/^<<-?[\x27"]|[\x27"]$/, "", tag)
    inhd = 1; hdtag = tag; next
}
inhd { if ($0 ~ "^[[:space:]]*" hdtag "[[:space:]]*$") inhd = 0; next }
{ buf = buf $0
  if (walk($0)) found = 1
  if ($0 ~ /\\[[:space:]]*$/) { sub(/\\[[:space:]]*$/, " ", buf); next }
  if (found) print buf
  buf = ""; found = 0 }
END { if (buf != "" && found) print buf }
