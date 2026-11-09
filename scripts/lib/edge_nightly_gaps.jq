# edge_nightly_gaps.jq: the CRUX rows of the #5254 edge gap list, from one
# crux_inference_dogfood.sh receipt. One line per verb where apr missed a cell,
# ranked by cells missed, then one line for a comparator that refused.
def top_reason: group_by(.) | max_by(length) | .[0] // null;

([.cells[] | {v: .key.verb, a: .engines.apr}]
 | group_by(.v)
 | map({verb: .[0].v,
        n: length,
        ok: ([.[] | select(.a.correct == true)] | length),
        wrong: ([.[] | select(.a.answered == true and .a.correct == false)] | length),
        unans: ([.[] | select(.a.answered != true)] | length),
        why: ([.[] | select(.a.answered != true) | .a.why // "no reason given"] | top_reason)})
 | map(. + {miss: (.n - .ok)})
 | sort_by(-.miss)
 | .[]
 | select(.miss > 0)
 | "CRUX `\(.verb)`: apr correct on \(.ok) of \(.n); \(.wrong) wrong, \(.unans) unanswered"
   + (if .why then " (top reason: \(.why))" else "" end)),
([.cells[] | .engines.ollama // empty | select(.answered != true) | .why // "no reason given"]
 | if length > 0 then "comparator ollama refused \(length) cells: \(top_reason)" else empty end)
