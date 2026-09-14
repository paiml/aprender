# feature_leak_edges.jq -- owned by scripts/check_feature_gated_test_leak.sh
#
# In:  `cargo metadata --no-deps --format-version 1`, plus $gatesraw, a TSV of
#      <crate>\t<feature> naming every feature that gates an integration target.
# Out: TSV <enabler>\t<dep kind>\t<gated crate>\t<feature>\t<how>
#
# It lives in its OWN FILE rather than inline in the guard because bashrs lints
# the guard as shell and reads jq's `$r[0]` / `$deps[]` as unbraced array
# expansions -- nine SC1087 errors on a program that is not shell at all. A
# false positive silenced by rewriting the code is worse than the finding; a
# false positive removed by putting the jq where jq belongs is just correct.
  ($gatesraw | split("\n") | map(select(length > 0) | split("\t"))
    | reduce .[] as $r ({}; .[$r[0]] = ((.[$r[0]] // []) + [$r[1]]))) as $gates
| .packages[]
| . as $a
| ($a.dependencies | map({ alias: (.rename // .name), real: .name,
                           kind: (if (.kind == null or .kind == "") then "normal" else .kind end),
                           feats: (.features // []) })) as $deps
| (
    # (i) the dependency names the feature directly: features = ["browser"]
    ( $deps[] | . as $d | $d.feats[]
      | select(($gates[$d.real] // []) | index(.))
      | [$a.name, $d.kind, $d.real, ., "features=[] on the dependency"] )
  ,
    # (ii) the enabler's own feature table forwards it: browser = ["probar/browser"].
    # Both spellings matter -- covering only (i) misses aprender-test-cli, which
    # reaches aprender-test-lib/browser exactly this way.
    ( ($a.features // {}) | to_entries[] | . as $f | $f.value[]
      | select(contains("/"))
      | (split("/")) as $parts
      | ($parts[0] | rtrimstr("?")) as $dn
      | ($parts[1] | ltrimstr("?")) as $fe
      | ($deps[] | select(.alias == $dn)) as $d
      | select(($gates[$d.real] // []) | index($fe))
      | [$a.name, $d.kind, $d.real, $fe, ("feature " + $f.key + " -> " + $dn + "/" + $fe)] )
  )
# A crate arming its OWN gated target is not a leak: it owns that target.
| select(.[0] != .[2])
| @tsv
