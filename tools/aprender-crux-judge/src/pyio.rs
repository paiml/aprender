//! File reads with CPython's semantics: `open()`'s argument checks and
//! OSError text, strict and `errors="replace"` UTF-8 decoding, universal
//! newlines, and the 8192-byte chunking that line iteration decodes in (it
//! decides the *position* a UnicodeDecodeError names).

use std::io::Read;

use crate::pyerr::{PyErr, PyResult};
use crate::pyval::{py_strip, repr_str, Val};

/// `TextIOWrapper._CHUNK_SIZE`.
const CHUNK: usize = 8192;

/// The str a Python `open(path)` call accepts. Divergence: Python opens an
/// `int`/`bool` argument as a file descriptor; the port raises TypeError.
pub fn path_arg(v: &Val) -> PyResult<String> {
    match v {
        Val::Str(s) => {
            if s.contains('\0') {
                return Err(PyErr::value("embedded null byte"));
            }
            Ok(s.clone())
        }
        other => Err(PyErr::type_err(format!(
            "expected str, bytes or os.PathLike object, not {}",
            other.type_name()
        ))),
    }
}

/// `OSError(errno, strerror, filename)` as `str()` prints it.
pub fn os_err(e: &std::io::Error, path: &str) -> PyErr {
    let Some(errno) = e.raw_os_error() else {
        return PyErr::new("OSError", format!("{e}: {}", repr_str(path)));
    };
    let full = e.to_string();
    let suffix = format!(" (os error {errno})");
    let strerror = full.strip_suffix(&suffix).unwrap_or(&full);
    let kind = match errno {
        2 => "FileNotFoundError",
        21 => "IsADirectoryError",
        1 | 13 => "PermissionError",
        20 => "NotADirectoryError",
        17 => "FileExistsError",
        _ => "OSError",
    };
    PyErr::new(
        kind,
        format!("[Errno {errno}] {strerror}: {}", repr_str(path)),
    )
}

/// `open(path, "rb").read()`.
pub fn read_bytes(path: &str) -> PyResult<Vec<u8>> {
    let mut f = std::fs::File::open(path).map_err(|e| os_err(&e, path))?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).map_err(|e| os_err(&e, path))?;
    Ok(buf)
}

/// `_load_bytes(v)`: `open(v, "rb").read()` for any value.
pub fn load_bytes(v: &Val) -> PyResult<Vec<u8>> {
    read_bytes(&path_arg(v)?)
}

/// `"\r\n"` and `"\r"` become `"\n"`.
pub fn universal_newlines(s: &str) -> String {
    if !s.contains('\r') {
        return s.to_string();
    }
    s.replace("\r\n", "\n").replace('\r', "\n")
}

/// `bytes.decode("utf-8", "replace")`: one U+FFFD per maximal invalid
/// subpart, which is what CPython emits.
pub fn decode_lossy(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

/// The UnicodeDecodeError CPython raises for the first bad sequence in `b`.
fn decode_error(b: &[u8], e: &std::str::Utf8Error) -> PyErr {
    let start = e.valid_up_to();
    let (end, reason) = match e.error_len() {
        None => (b.len(), "unexpected end of data"),
        Some(n) => {
            let lead = b[start];
            let reason = if n == 1 && !(0xc2..=0xf4).contains(&lead) {
                "invalid start byte"
            } else {
                "invalid continuation byte"
            };
            (start + n, reason)
        }
    };
    let msg = if end - start == 1 {
        format!(
            "'utf-8' codec can't decode byte 0x{:02x} in position {start}: {reason}",
            b[start]
        )
    } else {
        format!(
            "'utf-8' codec can't decode bytes in position {start}-{}: {reason}",
            end - 1
        )
    };
    PyErr::new("UnicodeDecodeError", msg)
}

/// `bytes.decode("utf-8")` (strict, final).
pub fn decode_strict(b: &[u8]) -> PyResult<String> {
    match std::str::from_utf8(b) {
        Ok(s) => Ok(s.to_string()),
        Err(e) => Err(decode_error(b, &e)),
    }
}

/// `read_text(path)` of the judge: falsy path → `""`, OSError → `""`,
/// errors="replace", universal newlines. TypeError and ValueError (an
/// embedded NUL) propagate, as in the original.
pub fn read_text(v: &Val) -> PyResult<String> {
    if !v.truthy() {
        return Ok(String::new());
    }
    let path = path_arg(v)?;
    match read_bytes(&path) {
        Ok(b) => Ok(universal_newlines(&decode_lossy(&b))),
        Err(e) if e.is_os() => Ok(String::new()),
        Err(e) => Err(e),
    }
}

/// `open(path, encoding="utf-8").read()`.
pub fn read_text_strict(path: &str) -> PyResult<String> {
    let b = read_bytes(path)?;
    Ok(universal_newlines(&decode_strict(&b)?))
}

/// Decode `data` incrementally: before EOF, an incomplete trailing sequence
/// is held back in `pending` for the next chunk.
fn decode_chunk(data: &[u8], eof: bool, pending: &mut Vec<u8>) -> PyResult<String> {
    match std::str::from_utf8(data) {
        Ok(s) => Ok(s.to_string()),
        Err(e) if e.error_len().is_none() && !eof => {
            let ok = e.valid_up_to();
            *pending = data[ok..].to_vec();
            Ok(std::str::from_utf8(&data[..ok])
                .expect("valid prefix")
                .to_string())
        }
        Err(e) => Err(decode_error(data, &e)),
    }
}

/// IncrementalNewlineDecoder(translate=True): a `\r` at the end of a chunk
/// waits for the next one, in case a `\n` follows.
fn translate_chunk(decoded: &str, eof: bool, pending_cr: &mut bool) -> String {
    let mut out = String::new();
    if *pending_cr {
        out.push('\r');
        *pending_cr = false;
    }
    out.push_str(decoded);
    if !eof && out.ends_with('\r') {
        out.pop();
        *pending_cr = true;
    }
    universal_newlines(&out)
}

/// Hand every complete non-blank line in `text` to `f`, leaving the
/// unterminated tail in `text`.
fn emit_lines(text: &mut String, f: &mut dyn FnMut(&str) -> PyResult<()>) -> PyResult<()> {
    while let Some(nl) = text.find('\n') {
        let line: String = text.drain(..=nl).collect();
        if !py_strip(&line).is_empty() {
            f(&line)?;
        }
    }
    Ok(())
}

/// `for line in open(path, encoding="utf-8"): if line.strip(): f(line)`,
/// decoding in 8192-byte chunks the way TextIOWrapper does, so a decode
/// error surfaces after the lines before its chunk were handled and names a
/// position relative to that chunk.
pub fn for_each_nonblank_line(path: &str, f: &mut dyn FnMut(&str) -> PyResult<()>) -> PyResult<()> {
    let mut fh = std::fs::File::open(path).map_err(|e| os_err(&e, path))?;
    let mut pending: Vec<u8> = Vec::new();
    let mut text = String::new();
    let mut pending_cr = false;
    loop {
        let mut chunk = vec![0u8; CHUNK];
        let n = read_full(&mut fh, &mut chunk).map_err(|e| os_err(&e, path))?;
        chunk.truncate(n);
        let eof = n == 0;
        let mut data = std::mem::take(&mut pending);
        data.extend_from_slice(&chunk);
        let decoded = decode_chunk(&data, eof, &mut pending)?;
        text.push_str(&translate_chunk(&decoded, eof, &mut pending_cr));
        emit_lines(&mut text, f)?;
        if eof {
            if !text.is_empty() && !py_strip(&text).is_empty() {
                f(&text)?;
            }
            return Ok(());
        }
    }
}

/// `read1(8192)` on a regular file returns a full chunk unless at EOF.
fn read_full(fh: &mut std::fs::File, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut got = 0;
    while got < buf.len() {
        match fh.read(&mut buf[got..]) {
            Ok(0) => break,
            Ok(n) => got += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(got)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_error_messages() {
        let cases: [(&[u8], &str); 5] = [
            (
                b"ab\xff",
                "'utf-8' codec can't decode byte 0xff in position 2: invalid start byte",
            ),
            (
                b"a\xe2\x41",
                "'utf-8' codec can't decode byte 0xe2 in position 1: invalid continuation byte",
            ),
            (
                b"\xf0\x90\x41",
                "'utf-8' codec can't decode bytes in position 0-1: invalid continuation byte",
            ),
            (
                b"a\xe2\x82",
                "'utf-8' codec can't decode bytes in position 1-2: unexpected end of data",
            ),
            (
                b"\xc0\x80",
                "'utf-8' codec can't decode byte 0xc0 in position 0: invalid start byte",
            ),
        ];
        for (b, want) in cases {
            assert_eq!(decode_strict(b).unwrap_err().msg, want);
        }
    }
}
