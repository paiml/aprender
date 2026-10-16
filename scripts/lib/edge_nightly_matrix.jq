# edge_nightly_matrix.jq: the verb x device matrix of the #5254 edge nightly.
# Input: null; --slurpfile gpu <crux gpu receipt> --slurpfile cpu <crux cpu
# receipt> (either may be []). One markdown row per verb: for each device, apr
# and ollama as correct/answered/cells over a FIXED verb list. ollama has no
# `code` verb (n/a). With --arg zero 1, instead one line per (device, engine,
# verb) that measured nothing: no cells, nothing answered (an empty output), or
# a known gap (`apr code` on cpu, #3978). Each is a missing measurement, never
# a quiet RED and never an exemption.
def tally($r; $engine; $v):
  [$r[0].cells[]? | select(.key.verb == $v)] as $c
  | {n: ($c | length),
     answered: ([$c[] | .engines[$engine] // {} | select(.answered == true)] | length),
     correct: ([$c[] | .engines[$engine] // {} | select(.correct == true)] | length)};
def offered($engine; $v): $engine == "apr" or $v != "code";
# `apr code` spawns a GPU server whatever the lane: its cpu cell is a known
# gap, still reported as unmeasured, never exempted.
def blocked($engine; $v; $dev): $engine == "apr" and $v == "code" and $dev == "cpu";
def cell($r; $engine; $v; $dev):
  if offered($engine; $v) | not then "n/a"
  elif blocked($engine; $v; $dev) then "not measured (#3978)"
  else tally($r; $engine; $v) as $t
    | if $t.n == 0 then "not run" else "\($t.correct)/\($t.answered)/\($t.n)" end
  end;

# The verbs the nightly asks for, fixed: a verb absent from a receipt is a
# missing measurement, not a shorter table.
["run", "chat", "serve run", "serve stream", "code"] as $verbs
| if $zero == "1" then
    $verbs[] as $v
    | (["gpu", $gpu], ["cpu", $cpu]) as [$dev, $r]
    | ("apr", "ollama") as $engine
    | select(offered($engine; $v))
    | tally($r; $engine; $v) as $t
    | if blocked($engine; $v; $dev) then "\($engine) `\($v)` on \($dev): not measured, it has no cpu lane (#3978)"
      elif $t.n == 0 then "\($engine) `\($v)` on \($dev): no cells run"
      elif $t.answered == 0 then "\($engine) answered 0 of \($t.n) `\($v)` cells on \($dev) (empty output)"
      else empty end
  else
    "| verb | GPU apr | GPU ollama | CPU apr | CPU ollama |",
    "|---|---|---|---|---|",
    ($verbs[] as $v
     | "| `\($v)` | \(cell($gpu; "apr"; $v; "gpu")) | \(cell($gpu; "ollama"; $v; "gpu")) | \(cell($cpu; "apr"; $v; "cpu")) | \(cell($cpu; "ollama"; $v; "cpu")) |")
  end
