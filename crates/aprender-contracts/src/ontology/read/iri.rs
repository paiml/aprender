//! RFC 3986 §5.2 reference resolution, the way Turtle resolves a relative IRI against the base. No
//! normalization beyond §5.2.4's dot-segment removal: the scheme, host and percent-encodings are kept as written.

/// The five components of a URI reference (RFC 3986 appendix B). An absent component is `None`; an empty one
/// is `Some("")`, and the two differ when the reference is recomposed.
#[derive(Debug, Default, PartialEq, Eq)]
struct Parts<'a> {
    scheme: Option<&'a str>,
    authority: Option<&'a str>,
    path: &'a str,
    query: Option<&'a str>,
    fragment: Option<&'a str>,
}

/// Whether `s` has a scheme, which is what N-Triples requires of every IRI.
#[must_use]
pub fn is_absolute(s: &str) -> bool {
    split(s).scheme.is_some()
}

/// `rel` resolved against `base` (§5.2.2, strict).
#[must_use]
pub fn resolve(base: &str, rel: &str) -> String {
    let b = split(base);
    let r = split(rel);
    if r.scheme.is_some() {
        return recompose(
            r.scheme,
            r.authority,
            &remove_dot_segments(r.path),
            r.query,
            r.fragment,
        );
    }
    if r.authority.is_some() {
        return recompose(
            b.scheme,
            r.authority,
            &remove_dot_segments(r.path),
            r.query,
            r.fragment,
        );
    }
    let (path, query) = if r.path.is_empty() {
        (b.path.to_string(), r.query.or(b.query))
    } else if r.path.starts_with('/') {
        (remove_dot_segments(r.path), r.query)
    } else {
        (remove_dot_segments(&merge(&b, r.path)), r.query)
    };
    recompose(b.scheme, b.authority, &path, query, r.fragment)
}

/// Appendix B: `^(([^:/?#]+):)?(//([^/?#]*))?([^?#]*)(\?([^#]*))?(#(.*))?`.
fn split(s: &str) -> Parts<'_> {
    let (rest, fragment) = match s.split_once('#') {
        Some((a, f)) => (a, Some(f)),
        None => (s, None),
    };
    let (rest, query) = match rest.split_once('?') {
        Some((a, q)) => (a, Some(q)),
        None => (rest, None),
    };
    let (scheme, rest) = split_scheme(rest);
    let (authority, path) = match rest.strip_prefix("//") {
        Some(after) => {
            let end = after.find('/').unwrap_or(after.len());
            (Some(&after[..end]), &after[end..])
        }
        None => (None, rest),
    };
    Parts {
        scheme,
        authority,
        path,
        query,
        fragment,
    }
}

/// `([^:/?#]+):` at the start. `rest` has no `?` or `#` left, so only `/` can end the scheme early.
fn split_scheme(s: &str) -> (Option<&str>, &str) {
    match s.find([':', '/']) {
        Some(i) if i > 0 && s.as_bytes()[i] == b':' => (Some(&s[..i]), &s[i + 1..]),
        _ => (None, s),
    }
}

/// §5.2.3.
fn merge(base: &Parts<'_>, rel_path: &str) -> String {
    if base.authority.is_some() && base.path.is_empty() {
        return format!("/{rel_path}");
    }
    match base.path.rfind('/') {
        Some(i) => format!("{}{rel_path}", &base.path[..=i]),
        None => rel_path.to_string(),
    }
}

/// §5.2.4.
fn remove_dot_segments(path: &str) -> String {
    let mut input = path.to_string();
    let mut output = String::new();
    while !input.is_empty() {
        dot_step(&mut input, &mut output);
    }
    output
}

/// One step of §5.2.4's loop: rules A to E, in order.
fn dot_step(input: &mut String, output: &mut String) {
    if let Some(rest) = input
        .strip_prefix("../")
        .or_else(|| input.strip_prefix("./"))
    {
        *input = rest.to_string();
    } else if input.starts_with("/./") || input == "/." {
        input.replace_range(..2, "");
        if input.is_empty() {
            input.push('/');
        }
    } else if input.starts_with("/../") || input == "/.." {
        input.replace_range(..3, "");
        if input.is_empty() {
            input.push('/');
        }
        output.truncate(output.rfind('/').unwrap_or(0));
    } else if input == "." || input == ".." {
        input.clear();
    } else {
        let skip = usize::from(input.starts_with('/'));
        let end = input[skip..].find('/').map_or(input.len(), |i| i + skip);
        output.push_str(&input[..end]);
        input.replace_range(..end, "");
    }
}

/// §5.3.
fn recompose(
    scheme: Option<&str>,
    authority: Option<&str>,
    path: &str,
    query: Option<&str>,
    fragment: Option<&str>,
) -> String {
    let mut s = String::new();
    if let Some(x) = scheme {
        s.push_str(x);
        s.push(':');
    }
    if let Some(x) = authority {
        s.push_str("//");
        s.push_str(x);
    }
    s.push_str(path);
    if let Some(x) = query {
        s.push('?');
        s.push_str(x);
    }
    if let Some(x) = fragment {
        s.push('#');
        s.push_str(x);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RFC 3986 §5.4.1 and §5.4.2, every example, against the RFC's own base.
    #[test]
    fn rfc3986_section_5_4_examples() {
        let base = "http://a/b/c/d;p?q";
        let cases = [
            ("g:h", "g:h"),
            ("g", "http://a/b/c/g"),
            ("./g", "http://a/b/c/g"),
            ("g/", "http://a/b/c/g/"),
            ("/g", "http://a/g"),
            ("//g", "http://g"),
            ("?y", "http://a/b/c/d;p?y"),
            ("g?y", "http://a/b/c/g?y"),
            ("#s", "http://a/b/c/d;p?q#s"),
            ("g#s", "http://a/b/c/g#s"),
            ("g?y#s", "http://a/b/c/g?y#s"),
            (";x", "http://a/b/c/;x"),
            ("g;x", "http://a/b/c/g;x"),
            ("g;x?y#s", "http://a/b/c/g;x?y#s"),
            ("", "http://a/b/c/d;p?q"),
            (".", "http://a/b/c/"),
            ("./", "http://a/b/c/"),
            ("..", "http://a/b/"),
            ("../", "http://a/b/"),
            ("../g", "http://a/b/g"),
            ("../..", "http://a/"),
            ("../../", "http://a/"),
            ("../../g", "http://a/g"),
            ("../../../g", "http://a/g"),
            ("../../../../g", "http://a/g"),
            ("/./g", "http://a/g"),
            ("/../g", "http://a/g"),
            ("g.", "http://a/b/c/g."),
            (".g", "http://a/b/c/.g"),
            ("g..", "http://a/b/c/g.."),
            ("..g", "http://a/b/c/..g"),
            ("./../g", "http://a/b/g"),
            ("./g/.", "http://a/b/c/g/"),
            ("g/./h", "http://a/b/c/g/h"),
            ("g/../h", "http://a/b/c/h"),
            ("g;x=1/./y", "http://a/b/c/g;x=1/y"),
            ("g;x=1/../y", "http://a/b/c/y"),
            ("g?y/./x", "http://a/b/c/g?y/./x"),
            ("g?y/../x", "http://a/b/c/g?y/../x"),
            ("g#s/./x", "http://a/b/c/g#s/./x"),
            ("g#s/../x", "http://a/b/c/g#s/../x"),
            ("http:g", "http:g"),
        ];
        for (rel, want) in cases {
            assert_eq!(resolve(base, rel), want, "resolve({rel:?})");
        }
    }

    #[test]
    fn a_base_with_an_authority_and_no_path_gets_a_slash() {
        assert_eq!(resolve("http://a", "b"), "http://a/b");
        assert_eq!(resolve("file:///x/y.ttl", ""), "file:///x/y.ttl");
        assert_eq!(resolve("file:///x/y.ttl", "#a"), "file:///x/y.ttl#a");
    }

    #[test]
    fn only_a_scheme_makes_an_iri_absolute() {
        assert!(is_absolute("http://a/b"));
        assert!(is_absolute("urn:x"));
        assert!(!is_absolute("/a/b"));
        assert!(!is_absolute("a/b:c"));
        assert!(!is_absolute(""));
        assert!(!is_absolute("#x"));
    }
}
