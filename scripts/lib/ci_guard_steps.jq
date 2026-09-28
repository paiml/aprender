# ci_guard_steps.jq -- the jq definitions behind ci_guard_steps.sh (#4415).
# Prepended to every query; the bash side binds $runner $sfx $status $expr.
def rx: gsub("(?<c>[.^$*+?()\\[\\]{}|\\\\])"; "\\\(.c)");
def calls_runner($job): ((.run // "") | tostring) | test("(^|\\s)" + ($runner | rx) + "\\s+" + ($job | rx) + "(\\s|$)");
def setup_end: (map(.id == "guard-setup") | index(true)) // -1;
def fail_fast: (.if == null) or ((.if | tostring) | test($status) | not);
def name_of: .name // .uses // "(unnamed)";
def steps_for($job; $ci):
    (.[$job + $sfx]) as $man
    | ( if $man != null then [($man.steps // []) | to_entries[] | {idx: "m\(.key)", step: .value}] else [] end )
    + ( if $man == null or ($ci | not) then
          (.[$job].steps // []) as $s | ($s | setup_end) as $e
          | [$s | to_entries[] | select(.key > $e and (.value | calls_runner($job) | not)
                and (((.value.if // "") | tostring) | contains("always") | not)) | {idx: "\(.key)", step: .value}]
        else [] end );
# "<setup index> <fail-fast steps after it>" for check-run-all
def fail_fast_count($job):
    .[$job].steps as $s | ($s | setup_end) as $e
    | "\($e) \([$s[($e + 1):][] | select(fail_fast)] | length)";
# one finding per line: what the runner could not honour in <job>-steps
def manifest_findings($job; $known):
    ($job + $sfx) as $m | .[$m] as $man | (.[$job].steps // []) as $own
    | ($known | split(", ")) as $kn
    | if $man == null then
        if any($own[]; calls_runner($job)) then "\($job): a step runs `bash \($runner) \($job)` but there is no \($m) job" else empty end
      else
        (if $man.if != false then "\($m): a manifest job needs `if: false`, or GitHub runs its steps a second time" else empty end),
        (($man.steps // []) | to_entries[] | .key as $i | .value as $s | ($s.name // "step \($i)") as $n
          | ((($s | keys) - ["name", "run", "env"]) as $x | if ($x | length) > 0 then "\($m) '\($n)': \($x | join(", ")) -- the runner cannot honour it" else empty end),
            (if (($s.run // "") | tostring) == "" then "\($m) '\($n)': no run:" else empty end),
            (if (($s.run // "") | tostring | test($expr)) then "\($m) '\($n)': ${{ }} in run: -- put it in env:" else empty end),
            (($s.env // {}) | to_entries[] | .key as $k | (.value | tostring) | [match($expr; "g").captures[0].string][]
              | select(. as $e | $kn | index([$e]) | not)
              | "\($m) '\($n)': env \($k) uses ${{ \(.) }}, which the runner cannot resolve (known: \($known))")),
        (if (($man.steps // []) | length) == 0 then "\($m): no steps" else empty end),
        (if any($own[]; calls_runner($job)) then empty else "\($job): no step runs `bash \($runner) \($job)`, so \($m) never runs" end)
      end;
# a manifest whose base job is gone runs nowhere
def orphans:
    . as $all | to_entries[] | select((.key | endswith($sfx)) and .value.if == false)
    | (.key | .[:length - ($sfx | length)]) as $b | select($all | has($b) | not)
    | "\(.key): a manifest with no job \($b) to run it";
# the run loop's record for one step: NUL-terminated fields
#   idx name run if has_uses has_run <n env> then n x (key value)
def step_record:
    [ .idx, (.step | name_of | tostring), (.step.run // "" | tostring), (.step.if // "" | tostring),
      (if .step | has("uses") then "1" else "" end), (if .step.run != null then "1" else "" end),
      ((.step.env // {}) | length | tostring),
      ((.step.env // {}) | to_entries[] | .key, (.value | if . == null then "" else tostring end)) ]
    | map(. + "\u0000") | add;
