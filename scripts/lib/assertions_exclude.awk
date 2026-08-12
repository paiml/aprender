# assertions_exclude.awk — the scanner for check_assertions_exclude.sh.
#
# Emits `path<TAB>line<TAB>domain<TAB>classes` for every in-scope assertion that
# admits more than one outcome class. Lives in its own file rather than inline in
# the shell script: bashrs parses an embedded awk program as shell and reported 96
# phantom errors (SC1065 "don't declare function parameters" against awk's local
# -variable convention), which would have buried a real finding later.
#
# Exit 3 signals an unrecognised StatusCode name — the caller must treat that as a
# hard failure, never as "no findings".

function classify_http(name,   c) {
  if (name ~ /^(OK|CREATED|ACCEPTED|NON_AUTHORITATIVE_INFORMATION|NO_CONTENT|RESET_CONTENT|PARTIAL_CONTENT|MULTI_STATUS|ALREADY_REPORTED|IM_USED)$/) return "2xx"
  if (name ~ /^(MULTIPLE_CHOICES|MOVED_PERMANENTLY|FOUND|SEE_OTHER|NOT_MODIFIED|USE_PROXY|TEMPORARY_REDIRECT|PERMANENT_REDIRECT)$/) return "3xx"
  if (name ~ /^(BAD_REQUEST|UNAUTHORIZED|PAYMENT_REQUIRED|FORBIDDEN|NOT_FOUND|METHOD_NOT_ALLOWED|NOT_ACCEPTABLE|PROXY_AUTHENTICATION_REQUIRED|REQUEST_TIMEOUT|CONFLICT|GONE|LENGTH_REQUIRED|PRECONDITION_FAILED|PAYLOAD_TOO_LARGE|URI_TOO_LONG|UNSUPPORTED_MEDIA_TYPE|RANGE_NOT_SATISFIABLE|EXPECTATION_FAILED|IM_A_TEAPOT|MISDIRECTED_REQUEST|UNPROCESSABLE_ENTITY|LOCKED|FAILED_DEPENDENCY|UPGRADE_REQUIRED|PRECONDITION_REQUIRED|TOO_MANY_REQUESTS|REQUEST_HEADER_FIELDS_TOO_LARGE|UNAVAILABLE_FOR_LEGAL_REASONS)$/) return "4xx"
  if (name ~ /^(INTERNAL_SERVER_ERROR|NOT_IMPLEMENTED|BAD_GATEWAY|SERVICE_UNAVAILABLE|GATEWAY_TIMEOUT|HTTP_VERSION_NOT_SUPPORTED|VARIANT_ALSO_NEGOTIATES|INSUFFICIENT_STORAGE|LOOP_DETECTED|NOT_EXTENDED|NETWORK_AUTHENTICATION_REQUIRED)$/) return "5xx"
  if (name ~ /^(CONTINUE|SWITCHING_PROTOCOLS|PROCESSING)$/) return "1xx"
  return "UNKNOWN:" name
}

# Remove string literals, char literals and line comments so that neither
# `||` nor parens inside them are ever counted. Returns the code-only text.
function strip(s,   out, i, n, c, d, instr, inchr, esc) {
  out = ""; n = length(s); instr = 0; inchr = 0; esc = 0
  for (i = 1; i <= n; i++) {
    c = substr(s, i, 1)
    if (instr) {
      if (esc) { esc = 0 }
      else if (c == "\\") { esc = 1 }
      else if (c == "\"") { instr = 0 }
      continue
    }
    if (inchr) {
      if (esc) { esc = 0 }
      else if (c == "\\") { esc = 1 }
      else if (c == "'") { inchr = 0 }
      continue
    }
    if (c == "\"") { instr = 1; out = out " "; continue }
    # A char literal only when it closes within 4 chars; otherwise it is a
    # lifetime (`&'a str`) and must be left alone.
    if (c == "'") {
      d = substr(s, i, 5)
      if (d ~ /^'\\?.'/) { inchr = 1; out = out " "; continue }
      out = out c; continue
    }
    if (c == "/" && substr(s, i+1, 1) == "/") break
    out = out c
  }
  return out
}

function balance(s,   i, n, c, b) {
  b = 0; n = length(s)
  for (i = 1; i <= n; i++) {
    c = substr(s, i, 1)
    if (c == "(") b++
    else if (c == ")") b--
  }
  return b
}

# Decide the verdict for one accumulated assertion body.
function verdict(t, path, ln,   m, name, cls, nhttp, http, nres, res, nexit, ex, k, keys, rest, num) {
  if (index(t, "||") == 0) return
  if (t !~ /StatusCode::|\.status\(|is_ok\(|is_err\(|\.success\(|\.failure\(|\.code\(|exit_code|is_success\(/) return

  # --- HTTP domain -------------------------------------------------------
  delete http; nhttp = 0
  rest = t
  while (match(rest, /StatusCode::[A-Z][A-Z0-9_]*/)) {
    name = substr(rest, RSTART + 12, RLENGTH - 12)
    cls = classify_http(name)
    if (cls ~ /^UNKNOWN:/) {
      printf("%s\t%s\tUNKNOWN-STATUS\t%s\n", path, ln, name) > "/dev/stderr"
      UNKNOWN_SEEN = 1
    } else if (!(cls in http)) { http[cls] = 1; nhttp++ }
    rest = substr(rest, RSTART + RLENGTH)
  }
  # numeric status comparisons: `.as_u16() == 400`, `status == 503`
  rest = t
  while (match(rest, /(as_u16\(\)|\.status\(\)|status)[^;|&]*==[ \t]*[1-5][0-9][0-9]\b/)) {
    m = substr(rest, RSTART, RLENGTH)
    if (match(m, /[1-5][0-9][0-9]$/)) {
      num = substr(m, RSTART, RLENGTH) + 0
      cls = int(num / 100) "xx"
      if (!(cls in http)) { http[cls] = 1; nhttp++ }
    }
    rest = substr(rest, RSTART + RLENGTH)
  }

  # --- Result / success domain ------------------------------------------
  delete res; nres = 0
  if (t ~ /is_ok\(|\.success\(|is_success\(/) { res["ok"] = 1; nres++ }
  if (t ~ /is_err\(|\.failure\(/)             { res["err"] = 1; nres++ }

  # --- process exit domain ----------------------------------------------
  delete ex; nexit = 0
  if (t ~ /\.code\(|exit_code/) {
    rest = t
    while (match(rest, /(Some\(|==[ \t]*)-?[0-9]+/)) {
      m = substr(rest, RSTART, RLENGTH)
      if (match(m, /-?[0-9]+$/)) {
        num = substr(m, RSTART, RLENGTH) + 0
        cls = (num == 0) ? "zero" : "nonzero"
        if (!(cls in ex)) { ex[cls] = 1; nexit++ }
      }
      rest = substr(rest, RSTART + RLENGTH)
    }
    if (t ~ /\.code\(\)[ \t]*\.is_none\(|== *None/) { if (!("none" in ex)) { ex["none"] = 1; nexit++ } }
  }

  if (nhttp > 1) { keys = ""; for (k in http) keys = keys (keys == "" ? "" : ",") k; printf("%s\t%s\thttp\t%s\n", path, ln, keys); return }
  if (nres  > 1) { printf("%s\t%s\tresult\tok,err\n", path, ln); return }
  if (nexit > 1) { keys = ""; for (k in ex) keys = keys (keys == "" ? "" : ",") k; printf("%s\t%s\texit\t%s\n", path, ln, keys); return }
}

BEGIN { acc = ""; depth = 0; startln = 0; UNKNOWN_SEEN = 0 }
FNR == 1 { if (depth != 0) { acc = ""; depth = 0 } }
{
  code = strip($0)
  if (depth == 0) {
    # Find an assertion opener on this line.
    if (match(code, /(^|[^A-Za-z0-9_])(prop_assert|debug_assert|assert)!\(/)) {
      startln = FNR
      acc = substr(code, RSTART)
      depth = balance(acc)
      if (depth <= 0) { verdict(acc, FILENAME, startln); acc = ""; depth = 0 }
      else { lines = 1 }
    }
  } else {
    acc = acc " " code
    lines++
    depth = balance(acc)
    if (depth <= 0) { verdict(acc, FILENAME, startln); acc = ""; depth = 0 }
    else if (lines > 80) { acc = ""; depth = 0 }   # runaway: give up on this candidate
  }
}
END { if (UNKNOWN_SEEN) exit 3 }
