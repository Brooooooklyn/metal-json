//! The scalar string unescaper — ONE implementation shared by the
//! `cpu-reference` oracle (`reference::stage6_strings`) and the GPU
//! pipeline's long-string fixup pass (`gpu::strings::patch_long_strings`).
//!
//! This module is deliberately **not** feature-gated: the GPU backend's
//! long-string valve re-runs flagged strings on the CPU, and that re-run
//! must be the bit-exact reference semantics whether or not the
//! `cpu-reference` feature is compiled in. Keeping a single function here
//! (instead of a gated copy + a duplicate) makes divergence impossible.
//!
//! Full escape handling per RFC 8259 — the K11 kernel
//! (`shaders/13_strings.metal`) mirrors this function exactly:
//!
//! - `\"` `\\` `\/` `\b` `\f` `\n` `\r` `\t`;
//! - `\u` + 4 hex digits (case-insensitive), including UTF-16 surrogate
//!   pairs; lone/inverted surrogates are rejected; the NUL escape
//!   (`\u` + `0000`) is legal and produces an interior NUL byte;
//! - unescaped control characters (`0x00..0x20`) are rejected;
//! - any other `\x` escape is rejected.

use crate::error::{Error, Result, SyntaxErrorKind};

/// Read 4 hex digits (case-insensitive) at `raw[at..at+4]`.
fn hex4(raw: &[u8], at: usize) -> Option<u32> {
    if at + 4 > raw.len() {
        return None;
    }
    let mut value = 0u32;
    for &b in &raw[at..at + 4] {
        value = value * 16 + (b as char).to_digit(16)?;
    }
    Some(value)
}

/// Unescape one raw string body (the bytes between the quotes, escapes
/// still escaped). `base` is the absolute offset of `raw[0]` in the input,
/// for error reporting; escape errors point at the backslash.
///
/// # Errors
///
/// [`SyntaxErrorKind::InvalidStringEscape`] (at the backslash of the bad
/// escape — bad designator, bad/short hex, lone or inverted surrogates) or
/// [`SyntaxErrorKind::ControlCharacterInString`] (at the raw control
/// byte). No other error is ever produced.
pub(crate) fn unescape(raw: &[u8], base: u32) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(raw.len());
    let mut i = 0usize;
    while i < raw.len() {
        let b = raw[i];
        if b < 0x20 {
            // Raw control characters must be escaped (kills
            // n_string_unescaped_tab.json / _newline / _ctrl_char).
            return Err(Error::Syntax {
                offset: u64::from(base) + i as u64,
                kind: SyntaxErrorKind::ControlCharacterInString,
            });
        }
        if b != b'\\' {
            // UTF-8 continuation bytes were validated in stage 1; copy
            // verbatim. (An unescaped `"` cannot appear: the extent ends at
            // the first unescaped quote.)
            out.push(b);
            i += 1;
            continue;
        }

        let escape_error = || Error::Syntax {
            offset: u64::from(base) + i as u64,
            kind: SyntaxErrorKind::InvalidStringEscape,
        };
        // A trailing lone backslash cannot occur via stages 1-3 (it would
        // have escaped the closing quote), but stay graceful for direct use.
        let designator = *raw.get(i + 1).ok_or_else(escape_error)?;
        match designator {
            b'"' | b'\\' | b'/' => {
                out.push(designator);
                i += 2;
            }
            b'b' => {
                out.push(0x08);
                i += 2;
            }
            b'f' => {
                out.push(0x0C);
                i += 2;
            }
            b'n' => {
                out.push(0x0A);
                i += 2;
            }
            b'r' => {
                out.push(0x0D);
                i += 2;
            }
            b't' => {
                out.push(0x09);
                i += 2;
            }
            b'u' => {
                let first = hex4(raw, i + 2).ok_or_else(escape_error)?;
                let (code_point, consumed) = match first {
                    0xD800..=0xDBFF => {
                        // High surrogate: must be chased by a low-surrogate
                        // escape (kills n_string_incomplete_surrogate.json
                        // and n_string_1_surrogate_then_escape_u1.json).
                        if raw.get(i + 6) != Some(&b'\\') || raw.get(i + 7) != Some(&b'u') {
                            return Err(escape_error());
                        }
                        let low = hex4(raw, i + 8).ok_or_else(escape_error)?;
                        if !(0xDC00..=0xDFFF).contains(&low) {
                            return Err(escape_error());
                        }
                        (0x10000 + ((first - 0xD800) << 10) + (low - 0xDC00), 12)
                    }
                    // Lone / inverted low surrogate.
                    0xDC00..=0xDFFF => return Err(escape_error()),
                    _ => (first, 6),
                };
                let ch = char::from_u32(code_point)
                    .expect("surrogate-free code point <= U+10FFFF is a valid char");
                let mut buf = [0u8; 4];
                out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                i += consumed;
            }
            // Unknown escapes (kills n_string_escape_x.json,
            // n_string_backslash_00.json, n_string_escaped_emoji.json).
            _ => return Err(escape_error()),
        }
    }
    Ok(out)
}
