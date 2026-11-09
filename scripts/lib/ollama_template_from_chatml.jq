# ollama_template_from_chatml.jq: an ollama Go TEMPLATE built from a GGUF's
# OWN `tokenizer.chat_template` (#5254). Input: `apr inspect <gguf> --json`.
#
# ollama 0.17.6 imports a Qwen3.5 GGUF with TEMPLATE {{ .Prompt }}: it cannot
# translate that Jinja, so it would complete raw text while apr chats. The
# translation here covers ONE shape and refuses everything else (empty output):
#   - ChatML turns: `<|im_start|>{role}\n{content}<|im_end|>\n`
#   - an `add_generation_prompt` block holding either one string literal (the
#     assistant prefix), or three: the prefix, then the `enable_thinking` true
#     branch, then the else branch (Qwen3 / Qwen3.5).
# The assistant prefix and both thinking suffixes are the template's own
# literals, so an ollama turn renders the same text apr renders from the same
# file. `$think` ("true" when the harness will pass --think) keeps both
# branches behind `.Think`; otherwise only the else branch, which is what Jinja
# renders with enable_thinking unset, so ollama cannot default to thinking.
def unescape: gsub("\\\\n"; "\n") | gsub("\\\\'"; "'");
def literals: [scan("\\{\\{-?\\s*'((?:[^'\\\\]|\\\\.)*)'\\s*-?\\}\\}") | .[0] | unescape];

(.. | objects | .["tokenizer.chat_template"]? // empty | strings) as $t
| select($t | test("<\\|im_start\\|>") and test("<\\|im_end\\|>"))
| ($t | split("if add_generation_prompt") | if length == 2 then .[1] else null end) as $gen
| select($gen != null)
| ($gen | literals) as $l
| select($l[0] // "" | startswith("<|im_start|>assistant"))
# Branch order is read, not trusted: the thinking branch only opens <think>,
# the other closes it. A template written the other way round is refused.
| select(($l | length) != 3 or (($l[1] | contains("</think>") | not) and ($l[2] | contains("</think>"))))
| "{{- range .Messages }}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n{{ end }}"
  + $l[0]
  + (if ($l | length) == 1 then ""
    elif ($l | length) == 3 and ($gen | test("enable_thinking")) and $think == "true" then
      "{{ if .Think }}" + $l[1] + "{{ else }}" + $l[2] + "{{ end }}"
    elif ($l | length) == 3 and ($gen | test("enable_thinking")) then $l[2]
    else error("unrecognised add_generation_prompt shape") end)
# ollama trims a TEMPLATE's trailing whitespace, which would drop the "\n\n"
# after </think> that Jinja renders. Keep it inside a Go string action.
| capture("^(?<body>[\\s\\S]*?)(?<tail>\\s*)$")
| .body + if .tail == "" then "" else "{{ \"" + (.tail | gsub("\n"; "\\n")) + "\" }}" end
