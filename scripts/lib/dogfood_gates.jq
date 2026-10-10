# dogfood_gates.jq — the ONE parser of [package.metadata.dogfood] (#4377: was dogfood_gates.py).
#
# Two consumers read this declaration, and until #2644 each had its own parser:
#   - scripts/dogfood.sh EXECUTES every declared string with `bash "$dg_path"`;
#   - scripts/check_verifier_pinning.sh SCANS the declared gates for unpinned verifiers.
# The guard's scrape saw only `"...*.sh"`. So a gate declared as `scripts/check_x`
# (no suffix), or written with TOML literal quotes, was RUN by the release and never
# SCANNED (#2644 audit CI-3, VP-05). One parser, shared, closes that. #4377 ported it
# from Python to jq, because no step of the dogfood gate path may need Python.
#
# Input:  `cargo metadata --no-deps --format-version 1` JSON.  Arg: --arg crate <name>.
# Output (jq -r): one line, or one line per gate.
#   NOPKG       no package of that name in the metadata (and not exactly one package)
#   NODECL      no [package.metadata.dogfood] table
#   BADSHAPE    `gates` is not a list, or holds a non-string / blank entry
#   EMPTY       `gates = []`, which claims there are no gates; it is not a pass
#   GATE <path> one per declared gate, in declaration order
#
# Usage:  jq -r --arg crate "$CRATE" -f scripts/lib/dogfood_gates.jq <cargo-metadata.json>
# Case table: scripts/check_dogfood_gates_parser.sh --self-test.

def select_package($name):
  (.packages // []) as $ps
  | ([$ps[] | select(.name == $name)] | first)
    // (if ($ps | length) == 1 then $ps[0] else null end);

select_package($crate) as $pkg
| if $pkg == null then "NOPKG"
  else ($pkg.metadata // {} | if type == "object" then .dogfood else null end) as $decl
  | if ($decl | type) != "object" then "NODECL"
    else $decl.gates as $gates
    | if ($gates | type) != "array" then "BADSHAPE"
      elif ($gates | length) == 0 then "EMPTY"
      elif any($gates[]; (type != "string") or ((. | gsub("^\\s+|\\s+$"; "")) == "")) then "BADSHAPE"
      else $gates[] | "GATE " + gsub("^\\s+|\\s+$"; "")
      end
    end
  end
