# dogfood_transports.jq — reads [package.metadata.transports] for scripts/dogfood.sh's
# transport gates (#4377: this was a python3 heredoc in dogfood.sh).
#
# Input:  `cargo metadata --no-deps --format-version 1` JSON.  Arg: --arg crate <name>.
# Output (jq -r), one line each:
#   NOPKG <a,b,…>                          no package of that name, and not exactly one package
#   NODECL                                 no (or an empty) transports table
#   MANIFEST <path|-> <regen|->            when `manifest` is a table
#   BADSHAPE <key>                         a transport declared as a bare boolean
#   DECL <key> <e2e|-> <feat,…|-> <src|->  one per transport, keys in sorted order;
#                                          <src> is the src_path of the test target named by e2e
# A malformed declaration (transports not a table, a transport that is not a table,
# features that are not a list of strings) is a jq error. The caller maps that to META_ERROR.
#
# Usage:  jq -r --arg crate "$CRATE" -f scripts/lib/dogfood_transports.jq <cargo-metadata.json>

def falsy: . == null or . == false or . == 0 or . == "" or . == [] or . == {};
def or_dash: if falsy then "-" else tostring end;
def need($k): if type == "object" and has($k) then .[$k] else error("missing \($k)") end;
def feats:
  if . == null then []
  elif type == "array" and all(.[]; type == "string") then .
  else error("features must be a list of strings") end
  | join(",");

# Every line is collected before any is printed, so an error part-way prints
# nothing, and the caller sees META_ERROR alone, never half a plan with it.
[(.packages // []) as $ps
| (first($ps[] | select(.name == $crate)) // (if ($ps | length) == 1 then $ps[0] else null end)) as $pkg
| if $pkg == null then "NOPKG " + ([$ps[].name] | sort | join(","))
  else
    (reduce ($pkg | need("targets"))[] as $t ({};
       if any($t.kind[]; . == "test") then .[$t.name] = $t.src_path else . end)) as $tests
    | ($pkg.metadata | if falsy then {} else . end
       | if type != "object" then error("metadata must be a table") else .transports end) as $tp
    | if ($tp | falsy) then "NODECL"
      elif ($tp | type) != "object" then error("transports must be a table")
      else
        ($tp.manifest | if type == "object" then "MANIFEST \(.path | or_dash) \(.regen | or_dash)" else empty end),
        ($tp | to_entries | sort_by(.key)[] | select(.key != "manifest")
          | .key as $k | .value as $v
          | if ($v | type) == "boolean" then "BADSHAPE \($k)"
            else ($v | if falsy then {} else . end) as $o
            | if ($o | type) != "object" then error("transport \($k) must be a table")
              else ($o.e2e // "") as $tgt
              | "DECL \($k) \($tgt | or_dash) \($o.features | feats | or_dash) \(if ($tgt | type) == "string" then ($tests[$tgt] // "-") else "-" end)"
              end
            end)
      end
  end] | .[]
