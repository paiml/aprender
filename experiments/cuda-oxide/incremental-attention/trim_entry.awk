# PMAT-883: trim a full NVPTX module .ptx to a standalone module = the PTX header
# (.version/.target/.address_size) + the single brace-balanced
# `.visible .entry <entry>` block. Pass the entry name via `-v entry=<name>`.
/^\.version|^\.target|^\.address_size/ { print; next }
$0 ~ ("\\.visible \\.entry " entry "\\(") { capture = 1 }
capture {
    print
    n = gsub(/{/, "{"); depth += n
    m = gsub(/}/, "}"); depth -= m
    if (n > 0) seen = 1
    if (seen && depth == 0) { capture = 0 }
}
