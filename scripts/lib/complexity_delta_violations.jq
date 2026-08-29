# Emit one TSV row per complexity violation: file, function, rule, value.
#
# $tb is the basename of the temporary sibling the gate materialised; $rb is the
# real basename it stands in for. Only the analysed file's own path needs
# rewriting — a violation pmat reaches through `include!` is already reported
# against the real sibling, identically on both sides of the comparison, so it
# must be left alone.
#
# Lives in its own file so bashrs lints the gate as shell instead of parsing a
# jq program as one.
.violations[]?
| ((.file // "") | sub("^\\./"; "")) as $p
| (if $p == $tb then $rb
   elif ($p | endswith("/" + $tb)) then (($p | .[0:(($p | length) - ($tb | length))]) + $rb)
   else $p end) as $np
| [$np, (.function // "?"), (.rule // "?"), ((.value // 0) | tostring)]
| @tsv
