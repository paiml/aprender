# dogfood_cites.awk -- the ONE reader of `evidence_path` in the dogfood surface
# ledger (#7370). check_dogfood_coverage.sh uses it for freshness, for the
# material-change caps and for the cite check, so the three cannot disagree.
# POSIX awk (gawk and mawk). Run with LC_ALL=C.
#
# The grammar is closed: text that does not match it is an error, never skipped.
#   field := cite *(", " cite) [" (" note ")"]
#   cite  := path [":" range *("," range)]
#   range := N ["-" M]                         with 1 <= N <= M
#   path  := a repo-relative path ending .rs .toml .yml .yaml .md .sh or .py;
#            no leading "/", no "//", no "." or ".." segment
#   note  := free text; its first "(" closes at the end of the field. A word in
#            it that names a file with one of those extensions, or that is a
#            name followed by a line number (Makefile:5, x#L5), must be a cite.
# NOT READ: other numbers in a note (":397-437", "16,732"), and a note word
# naming a file of another extension without a line number.
#
# Output, tab separated; <row> is the ledger line number:
#   P <row> <path>              the leading path of the row (see field())
#   W <row> <path>              a whole-file cite
#   C <row> <path> <lo> <hi>    one cited line range
#   E <row> <message>           an error: every consumer fails closed on it
#   N <rows>                    printed last: rows read

# csv(line, out): split one CSV line into out[1..n]; -1 = broken quoting.
function csv(line, out,    n, i, c, q, closed, cur, len) {
  n = 0; cur = ""; q = 0; closed = 0; len = length(line)
  for (i = 1; i <= len; i++) {
    c = substr(line, i, 1)
    if (q) {
      if (c != "\"") cur = cur c
      else if (substr(line, i + 1, 1) == "\"") { cur = cur c; i++ }
      else { q = 0; closed = 1 }
    } else if (c == ",") { out[++n] = cur; cur = ""; closed = 0 }
    else if (closed) return -1
    else if (c == "\"" && cur == "") q = 1
    else cur = cur c
  }
  if (q) return -1
  out[++n] = cur
  return n
}

function err(msg) { printf "E\t%d\t%s\n", NR, msg }

function is_cite(s) {
  return s ~ /^[A-Za-z0-9_.\/-]+\.(rs|toml|yml|yaml|md|sh|py)(:[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*)?$/
}

# cite(s): report one string that already matches is_cite().
function cite(s, leading,    path, spec, k, i, parts, se, m, lo, hi) {
  k = index(s, ":")
  path = k ? substr(s, 1, k - 1) : s
  spec = k ? substr(s, k + 1) : ""
  if (("/" path "/") ~ /\/\// || ("/" path "/") ~ /\/\.\.?\//) {
    err("path is not a plain repo-relative path: " path); return
  }
  if (spec == "") { printf "W\t%d\t%s\n", NR, path; return }
  k = split(spec, parts, ",")
  for (i = 1; i <= k; i++) {
    m = split(parts[i], se, "-")
    lo = se[1] + 0; hi = (m == 2 ? se[2] + 0 : lo)
    if (lo < 1 || lo > hi) err(path ":" parts[i] " is out of range or reversed")
    else printf "C\t%d\t%s\t%d\t%d\n", NR, path, lo, hi
  }
}

# note(s): s starts with "(". Its first "(" must close at the end of the field.
function note(s,    i, c, depth, len, k, i2, words, w) {
  len = length(s); depth = 0
  for (i = 1; i <= len; i++) {
    c = substr(s, i, 1)
    if (c == "(") depth++
    else if (c == ")") { depth--; if (depth == 0) break }
  }
  if (depth != 0 || i != len) {
    err("malformed cite field: the note must close at the end of the field: " s); return
  }
  k = split(substr(s, 2, len - 2), words, /[ \t]+/)
  for (i2 = 1; i2 <= k; i2++) {
    w = words[i2]
    sub(/^[([]+/, "", w)
    sub(/[]);,.:]+$/, "", w)
    if (is_cite(w)) cite(w)
    else if (w ~ /\.(rs|toml|yml|yaml|md|sh|py)([^A-Za-z0-9_]|$)/ ||
             w ~ /[A-Za-z][A-Za-z0-9_.\/-]*(:[Ll]?[0-9]|#[Ll][0-9])/ ||
             w ~ /([.\/][A-Za-z0-9_.\/-]*|file)#[0-9]/)
      err("malformed cite in a note: " w)
  }
}

function field(p,    k, head, rest, n, i, parts) {
  k = index(p, " (")
  head = k ? substr(p, 1, k - 1) : p
  rest = k ? substr(p, k + 1) : ""
  n = split(head, parts, ", ")
  # P is the leading path even when its line spec is bad: the E record is what
  # fails that row; P only says the row names a file.
  if (match(parts[1], /^[A-Za-z0-9_.\/-]+\.(rs|toml|yml|yaml|md|sh|py)/))
    printf "P\t%d\t%s\n", NR, substr(parts[1], 1, RLENGTH)
  for (i = 1; i <= n; i++) {
    if (is_cite(parts[i])) cite(parts[i])
    else err("malformed cite: " parts[i])
  }
  if (rest != "") note(rest)
}

{ sub(/\r$/, "") }

NR == 1 {
  ncol = csv($0, h)
  for (i = 1; i <= ncol; i++) if (h[i] == "evidence_path") col = i
  next
}

$0 == "" { next }

{
  rows++
  nf = csv($0, f)
  if (nf < 0) { err("broken CSV quoting (a field may not span lines)"); next }
  # An unquoted comma inside a cite shifts the rest of the row: fail closed.
  if (nf != ncol) { err((nf > ncol ? "more" : "fewer") " fields than the header (quote a field that holds commas)"); next }
  p = f[col]
  sub(/^[ \t]+/, "", p); sub(/[ \t]+$/, "", p)
  if (p == "") { err("empty evidence_path (a row with no cite is not evidence)"); next }
  field(p)
}

END {
  if (!col) { print "dogfood_cites.awk: no evidence_path column in the header" > "/dev/stderr"; exit 2 }
  printf "N\t%d\n", rows
}
