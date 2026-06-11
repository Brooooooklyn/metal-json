//! Stage 6 — string validation + unescaping.
//!
//! Scalar oracle for GPU kernel **K11** (string unescape: fast no-escape
//! path + thread-per-string escape path). The M4 kernel unit tests run K11
//! and this function on identical token streams and diff the unescaped
//! bytes.
//!
//! For every `QuoteOpen`/`QuoteClose` token pair the raw extent is
//! `input[open+1 .. close]` (the GPU gets the same extent for free because
//! the two quote tokens are adjacent). Full escape handling per RFC 8259:
//!
//! - `\"` `\\` `\/` `\b` `\f` `\n` `\r` `\t`;
//! - `\u` + 4 hex digits (case-insensitive), including UTF-16 surrogate
//!   pairs (a high surrogate `D800..=DBFF` followed by a low surrogate
//!   `DC00..=DFFF` combines into one code point, e.g. U+1F600 😀). Lone
//!   high surrogates, lone low surrogates and inverted pairs are rejected;
//!   the NUL escape (`\u` + `0000`) is **legal** and produces an interior
//!   NUL byte (which is why string records carry an explicit length);
//! - unescaped control characters (`0x00..0x20`) are rejected;
//! - any other `\x` escape is rejected.
//!
//! Each output record also carries the byte offset its
//! `[u32 LE length][content][NUL]` record will start at in the string
//! buffer. Offsets follow the tape-format-v1 allocation scheme (see
//! `docs/tape-format.md`): the **exclusive prefix sum of `raw_len + 5`**
//! over strings in document order, where `raw_len` is the byte count
//! between the quotes in the *input* (before unescaping) and `+5` covers
//! the 4-byte length prefix plus the NUL terminator. The GPU computes
//! these offsets in the K7 scan from token positions alone — *before*
//! unescaping runs — which is exactly why the scheme uses raw lengths.
//! Unescaped content is always ≤ raw content, so a record whose escapes
//! shrank it leaves a **gap** before the next slot; the reference
//! zero-fills gaps (in [`emit_tape`](super::emit_tape)) so its output is
//! deterministic, while GPU gap bytes are unspecified (kernel diff tests
//! compare per-record bytes + tape offsets, never gap bytes).

use super::tokens::{Token, TokenKind};
use crate::error::{Error, Result, SyntaxErrorKind};
use crate::tape::{STRING_RECORD_HEADER_BYTES, STRING_RECORD_TRAILER_BYTES};

/// One unescaped string, tagged with its `QuoteOpen` token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnescapedString {
    /// Index into the stage-2 token stream of the `QuoteOpen` token.
    pub token_index: u32,
    /// Fully unescaped UTF-8 content (may contain interior NULs).
    pub bytes: Vec<u8>,
    /// Byte offset of this record in the string buffer: the exclusive
    /// prefix sum of `raw_len + 5` over the preceding strings (document
    /// order). This is what the `"` tape word stores.
    pub record_offset: u64,
    /// Byte count between the quotes in the input, before unescaping.
    /// The record's allocated slot is `raw_len + 5` bytes; `bytes.len()`
    /// never exceeds `raw_len`, and the difference is gap space.
    pub raw_len: u32,
}

/// Stage 6: validate + unescape every string literal, in token order.
///
/// # Errors
///
/// [`SyntaxErrorKind::InvalidStringEscape`] (at the backslash of the bad
/// escape), [`SyntaxErrorKind::ControlCharacterInString`] (at the raw
/// control byte), or — only when driven with a token stream stage 3 has
/// not vetted — [`SyntaxErrorKind::UnterminatedString`].
pub fn stage6_strings(tokens: &[Token], input: &[u8]) -> Result<Vec<UnescapedString>> {
    let mut out = Vec::new();
    let mut next_offset: u64 = 0;
    for (i, tok) in tokens.iter().enumerate() {
        if tok.kind != TokenKind::QuoteOpen {
            continue;
        }
        let close = tokens
            .get(i + 1)
            .filter(|t| t.kind == TokenKind::QuoteClose)
            .ok_or(Error::Syntax {
                offset: u64::from(tok.pos),
                kind: SyntaxErrorKind::UnterminatedString,
            })?;
        let raw = &input[tok.pos as usize + 1..close.pos as usize];
        let bytes = unescape(raw, tok.pos + 1)?;
        debug_assert!(bytes.len() <= raw.len(), "unescaping never grows a string");
        let record_offset = next_offset;
        // Slot size uses the RAW length (offsets must be derivable from
        // token positions alone, before unescaping), not bytes.len().
        next_offset +=
            (STRING_RECORD_HEADER_BYTES + raw.len() + STRING_RECORD_TRAILER_BYTES) as u64;
        out.push(UnescapedString {
            token_index: u32::try_from(i).expect("more than u32::MAX tokens"),
            bytes,
            record_offset,
            raw_len: u32::try_from(raw.len()).expect("string longer than u32::MAX bytes"),
        });
    }
    Ok(out)
}

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

/// Unescape one raw string body. `base` is the absolute offset of `raw[0]`
/// in the input, for error reporting; escape errors point at the backslash.
fn unescape(raw: &[u8], base: u32) -> Result<Vec<u8>> {
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

#[cfg(test)]
mod tests {
    use super::super::classify::stage1_classify;
    use super::super::tokens::stage2_tokens;
    use super::*;

    fn run(input: &[u8]) -> Result<Vec<UnescapedString>> {
        let tokens = stage2_tokens(&stage1_classify(input).unwrap(), input);
        stage6_strings(&tokens, input)
    }

    /// `\u` + `hex` escape text, built at runtime: the literal sequence
    /// must not appear in this source file (editor/tooling layers may
    /// resolve it like a JSON escape).
    fn u_esc(hex: &str) -> String {
        format!("{}u{hex}", '\\')
    }

    /// A quoted JSON string literal assembled from `parts`.
    fn quoted(parts: &[&str]) -> Vec<u8> {
        let mut s = String::from("\"");
        for p in parts {
            s.push_str(p);
        }
        s.push('"');
        s.into_bytes()
    }

    /// Unescape a single root string literal.
    fn content(input: &[u8]) -> Result<Vec<u8>> {
        run(input).map(|mut v| {
            assert_eq!(v.len(), 1, "fixture must contain exactly one string");
            v.pop().unwrap().bytes
        })
    }

    fn expect_err(input: &[u8], kind: SyntaxErrorKind) -> u64 {
        match run(input) {
            Err(Error::Syntax { offset, kind: k }) => {
                assert_eq!(k, kind, "{:?}", String::from_utf8_lossy(input));
                offset
            }
            other => panic!(
                "expected {kind:?} for {:?}, got {other:?}",
                String::from_utf8_lossy(input)
            ),
        }
    }

    #[test]
    fn plain_strings_pass_through() {
        assert_eq!(content(br#""hello""#).unwrap(), b"hello");
        assert_eq!(content(br#""""#).unwrap(), b"");
        assert_eq!(
            content("\"héllo 😀\"".as_bytes()).unwrap(),
            "héllo 😀".as_bytes()
        );
        // DEL (0x7F) is >= 0x20: legal unescaped (y_string_with_del_character).
        assert_eq!(content(b"\"a\x7Fb\"").unwrap(), b"a\x7Fb");
    }

    #[test]
    fn simple_escapes() {
        assert_eq!(
            content(br#""\" \\ \/ \b \f \n \r \t""#).unwrap(),
            b"\" \\ / \x08 \x0C \n \r \t"
        );
    }

    #[test]
    fn unicode_escapes() {
        assert_eq!(content(&quoted(&[&u_esc("0041")])).unwrap(), b"A");
        assert_eq!(content(&quoted(&[&u_esc("00e9")])).unwrap(), "é".as_bytes());
        // Case-insensitive hex.
        assert_eq!(content(&quoted(&[&u_esc("00E9")])).unwrap(), "é".as_bytes());
        assert_eq!(
            content(&quoted(&[&u_esc("2603")])).unwrap(),
            "\u{2603}".as_bytes() // snowman
        );
        // Highest BMP code point.
        assert_eq!(
            content(&quoted(&[&u_esc("FFFF")])).unwrap(),
            "\u{FFFF}".as_bytes()
        );
    }

    #[test]
    fn nul_escape_is_legal_json() {
        let input = quoted(&["a", &u_esc("0000"), "b"]);
        assert_eq!(content(&input).unwrap(), b"a\0b");
    }

    #[test]
    fn surrogate_pairs_combine() {
        // U+1F600 😀
        let pair = quoted(&[&u_esc("D83D"), &u_esc("DE00")]);
        assert_eq!(content(&pair).unwrap(), "\u{1F600}".as_bytes());
        // Lowercase hex digits work too.
        let pair = quoted(&[&u_esc("d83d"), &u_esc("de00")]);
        assert_eq!(content(&pair).unwrap(), "\u{1F600}".as_bytes());
        // U+10FFFF — the very last code point.
        let pair = quoted(&[&u_esc("DBFF"), &u_esc("DFFF")]);
        assert_eq!(content(&pair).unwrap(), "\u{10FFFF}".as_bytes());
    }

    #[test]
    fn bad_escapes_are_rejected_at_the_backslash() {
        use SyntaxErrorKind::InvalidStringEscape;
        // Offsets are absolute input offsets of the backslash.
        assert_eq!(expect_err(br#""\x41""#, InvalidStringEscape), 1); // n_string_escape_x
        assert_eq!(
            expect_err(&quoted(&[&u_esc("12")]), InvalidStringEscape),
            1 // short hex: only two digits before the closing quote
        );
        assert_eq!(expect_err(br#""\uZZZZ""#, InvalidStringEscape), 1); // bad hex
        assert_eq!(expect_err(br#""ab\q""#, InvalidStringEscape), 3);
        // Lone high surrogate (n_string_incomplete_surrogate.json).
        expect_err(&quoted(&[&u_esc("D800")]), InvalidStringEscape);
        // High surrogate chased by a non-surrogate escape
        // (n_string_1_surrogate_then_escape_u1.json).
        expect_err(
            &quoted(&[&u_esc("D800"), &u_esc("0041")]),
            InvalidStringEscape,
        );
        // High surrogate chased by a plain character.
        expect_err(&quoted(&[&u_esc("D800"), "x"]), InvalidStringEscape);
        // Lone low surrogate / inverted pair.
        expect_err(&quoted(&[&u_esc("DC00")]), InvalidStringEscape);
        expect_err(
            &quoted(&[&u_esc("DE00"), &u_esc("D83D")]),
            InvalidStringEscape,
        );
    }

    #[test]
    fn raw_control_characters_are_rejected() {
        use SyntaxErrorKind::ControlCharacterInString;
        assert_eq!(expect_err(b"\"a\tb\"", ControlCharacterInString), 2); // n_string_unescaped_tab
        expect_err(b"\"a\nb\"", ControlCharacterInString);
        expect_err(b"\"a\x01b\"", ControlCharacterInString);
        expect_err(b"\"a\x1Fb\"", ControlCharacterInString);
    }

    #[test]
    fn record_offsets_are_the_raw_length_prefix_sum() {
        // ["a","bc",""] — slots of raw_len+5: 1+5=6, 2+5=7, 0+5=5; the
        // offsets are the exclusive prefix sum: 0, 6, 13.
        let records = run(br#"["a","bc",""]"#).unwrap();
        let offsets: Vec<u64> = records.iter().map(|r| r.record_offset).collect();
        assert_eq!(offsets, vec![0, 6, 13]);
        let raw_lens: Vec<u32> = records.iter().map(|r| r.raw_len).collect();
        assert_eq!(raw_lens, vec![1, 2, 0]);
        let token_indices: Vec<u32> = records.iter().map(|r| r.token_index).collect();
        assert_eq!(token_indices, vec![1, 4, 7]);
    }

    #[test]
    fn shrinking_escapes_do_not_move_the_next_offset() {
        // A surrogate-pair escape: 12 raw bytes -> 4 content bytes. The
        // next slot still starts at the RAW-length prefix sum (4+12+1=17),
        // leaving an 8-byte gap after the shrunk record — offsets depend
        // only on token positions, never on unescape results.
        let mut input = b"[".to_vec();
        input.extend_from_slice(&quoted(&[&u_esc("D83D"), &u_esc("DE00")]));
        input.extend_from_slice(b",\"x\"]");
        let records = run(&input).unwrap();
        assert_eq!(records[0].bytes, "\u{1F600}".as_bytes());
        assert_eq!(records[0].raw_len, 12);
        assert_eq!(records[1].record_offset, (4 + 12 + 1) as u64);
    }

    #[test]
    fn keys_and_values_both_get_records() {
        let records = run(br#"{"k":"v"}"#).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].bytes, b"k");
        assert_eq!(records[1].bytes, b"v");
    }

    #[test]
    fn strings_spanning_bitmap_word_seams_unescape_fine() {
        // Escape sequence straddling byte 64.
        let mut input = b"\"".to_vec();
        input.extend(std::iter::repeat_n(b'a', 62)); // bytes 1..=62
        input.extend_from_slice(br#"\n"#); // backslash at 63, 'n' at 64
        input.extend_from_slice(b"b\"");
        let records = run(&input).unwrap();
        let mut want = vec![b'a'; 62];
        want.push(b'\n');
        want.push(b'b');
        assert_eq!(records[0].bytes, want);
    }

    #[test]
    fn unterminated_string_is_graceful_when_driven_directly() {
        // Stage 3 normally rejects this stream first.
        assert_eq!(expect_err(b"\"abc", SyntaxErrorKind::UnterminatedString), 0);
    }
}
