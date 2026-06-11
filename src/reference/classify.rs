//! Stage 1 — byte classification bitmaps + UTF-8 validation.
//!
//! Scalar oracle for GPU kernel **K1** (classify + escape + UTF-8) and the
//! quote-parity carry that **K2** propagates between chunks. The M2 kernel
//! unit tests run K1 and this function on identical inputs and diff the
//! bitmap words.
//!
//! Per 64-byte input word this stage produces two `u64` bitmaps
//! (bit `i` of word `w` describes input byte `w * 64 + i`):
//!
//! - [`Bitmaps::quote_real`]: `"` bytes that are *not* escaped. Escapes are
//!   resolved exactly like simdjson: a quote is escaped iff it is preceded
//!   by an odd-length run of backslashes (modeled here as a per-byte state
//!   machine, which carries across the 64-byte word seams for free — the
//!   GPU kernel must reproduce that carry explicitly).
//! - [`Bitmaps::candidates`]: structural operators (`{` `}` `[` `]` `:`
//!   `,`) plus *scalar starts* — the first byte of any run of
//!   non-whitespace, non-operator, non-quote bytes. Candidates are computed
//!   **without** knowing what is inside a string: bits inside string
//!   literals are present here and masked away by stage 2's in-string mask.
//!
//! UTF-8 validation is full Lemire-style: overlong encodings, UTF-16
//! surrogates (U+D800..U+DFFF), leads above U+10FFFF, truncated sequences
//! and stray continuation bytes are all rejected with [`Error::Utf8`] at
//! the offset of the **first byte of the first invalid sequence** (the same
//! offset `core::str::from_utf8`'s `valid_up_to` reports; a unit test pins
//! that equivalence).

use crate::error::{Error, Result};

/// Per-64-byte-word classification bitmaps from [`stage1_classify`].
///
/// Both vectors have `input_len.div_ceil(64)` words; bits at positions
/// `>= input_len` in the final word are always zero.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bitmaps {
    /// Unescaped `"` bytes (escape-resolved quote bitmap).
    pub quote_real: Vec<u64>,
    /// Structural operators | scalar starts, **pre** in-string masking.
    pub candidates: Vec<u64>,
    /// Length in bytes of the input the bitmaps describe.
    pub input_len: usize,
}

/// Is `b` JSON insignificant whitespace?
#[inline]
pub(crate) fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

/// Is `b` a structural operator?
#[inline]
pub(crate) fn is_op(b: u8) -> bool {
    matches!(b, b'{' | b'}' | b'[' | b']' | b':' | b',')
}

/// Stage 1: classify every input byte into bitmaps and validate UTF-8.
///
/// # Errors
///
/// [`Error::Utf8`] with the offset of the first invalid byte sequence.
pub fn stage1_classify(input: &[u8]) -> Result<Bitmaps> {
    validate_utf8(input)?;

    let words = input.len().div_ceil(64);
    let mut quote_real = vec![0u64; words];
    let mut candidates = vec![0u64; words];

    // Escape state machine: `escaped_next` is true when the *previous* byte
    // was a backslash that starts (or extends to odd length) an escape.
    // Equivalent to simdjson's odd-backslash-run resolution, including the
    // carry across 64-byte word seams.
    let mut escaped_next = false;
    // True when the previous byte allows a scalar run to start here
    // (start of input, whitespace, operator, or a real quote).
    let mut prev_allows_start = true;

    for (i, &b) in input.iter().enumerate() {
        let escaped = escaped_next;
        escaped_next = !escaped && b == b'\\';

        let quote = b == b'"' && !escaped;
        let op = is_op(b);
        let ws = is_ws(b);
        // Everything else — including backslashes and *escaped* quotes — is
        // scalar-class. Inside strings these bits are garbage by design;
        // stage 2 masks them.
        let scalar = !quote && !op && !ws;

        let bit = 1u64 << (i % 64);
        if quote {
            quote_real[i / 64] |= bit;
        }
        if op || (scalar && prev_allows_start) {
            candidates[i / 64] |= bit;
        }
        prev_allows_start = ws || op || quote;
    }

    Ok(Bitmaps {
        quote_real,
        candidates,
        input_len: input.len(),
    })
}

/// Full Lemire-style UTF-8 validation.
///
/// Rejects, with the offset of the first byte of the offending sequence:
/// - stray continuation bytes (`0x80..=0xBF` in lead position);
/// - overlong 2-byte leads `0xC0`/`0xC1`;
/// - overlong 3-byte sequences (`0xE0` followed by `< 0xA0`);
/// - UTF-16 surrogates (`0xED` followed by `>= 0xA0`);
/// - overlong 4-byte sequences (`0xF0` followed by `< 0x90`);
/// - code points above U+10FFFF (`0xF4` followed by `> 0x8F`, leads
///   `0xF5..=0xFF`);
/// - truncated sequences (at EOF or interrupted by a non-continuation).
fn validate_utf8(input: &[u8]) -> Result<()> {
    let n = input.len();
    let mut i = 0;
    while i < n {
        let b = input[i];
        if b < 0x80 {
            i += 1;
            continue;
        }
        let err = Err(Error::Utf8 { offset: i as u64 });
        // (continuation byte count, allowed range for the second byte)
        let (cont, lo, hi) = match b {
            0xC2..=0xDF => (1, 0x80, 0xBF),
            0xE0 => (2, 0xA0, 0xBF), // reject overlong 3-byte
            0xE1..=0xEC | 0xEE..=0xEF => (2, 0x80, 0xBF),
            0xED => (2, 0x80, 0x9F), // reject surrogates
            0xF0 => (3, 0x90, 0xBF), // reject overlong 4-byte
            0xF1..=0xF3 => (3, 0x80, 0xBF),
            0xF4 => (3, 0x80, 0x8F), // reject > U+10FFFF
            // 0x80..=0xC1: stray continuation / overlong 2-byte lead;
            // 0xF5..=0xFF: would encode > U+10FFFF.
            _ => return err,
        };
        if i + cont >= n {
            return err; // truncated at EOF
        }
        let second = input[i + 1];
        if second < lo || second > hi {
            return err;
        }
        for k in 2..=cont {
            if !matches!(input[i + k], 0x80..=0xBF) {
                return err;
            }
        }
        i += cont + 1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bit(words: &[u64], i: usize) -> bool {
        words[i / 64] >> (i % 64) & 1 == 1
    }

    fn classify(input: &[u8]) -> Bitmaps {
        stage1_classify(input).expect("valid UTF-8 fixture")
    }

    #[test]
    fn empty_input_produces_empty_bitmaps() {
        let b = classify(b"");
        assert!(b.quote_real.is_empty());
        assert!(b.candidates.is_empty());
        assert_eq!(b.input_len, 0);
    }

    #[test]
    fn operators_and_scalar_starts_are_candidates() {
        //                0123456789
        let b = classify(b"{ true:12}");
        // ops at 0, 6, 9; scalar starts at 2 ('t') and 7 ('1').
        for i in [0, 2, 6, 7, 9] {
            assert!(bit(&b.candidates, i), "candidate bit {i}");
        }
        // continuation bytes of runs are not candidates, nor is whitespace
        for i in [1, 3, 4, 5, 8] {
            assert!(!bit(&b.candidates, i), "non-candidate bit {i}");
        }
        assert!(b.quote_real.iter().all(|&w| w == 0));
    }

    #[test]
    fn scalar_start_after_each_boundary_kind() {
        // After start-of-input, ws, op, and a closing quote.
        //                0123456789
        let b = classify(b"x y,z\"q\"w");
        for i in [0, 2, 4, 6, 8] {
            // 'x' (start), 'y' (after ws), 'z' (after op), 'q' (after open
            // quote — garbage-by-design, masked by stage 2), 'w' (after
            // close quote).
            assert!(bit(&b.candidates, i), "scalar start bit {i}");
        }
    }

    #[test]
    fn quote_bitmap_resolves_escapes() {
        // "a\"b"  — the inner quote is escaped, outer two are real.
        let input = br#""a\"b""#;
        let b = classify(input);
        assert!(bit(&b.quote_real, 0));
        assert!(!bit(&b.quote_real, 3), "escaped quote must not be real");
        assert!(bit(&b.quote_real, 5));

        // "a\\" — even backslash run: the closing quote IS real.
        let input = br#""a\\""#;
        let b = classify(input);
        assert!(bit(&b.quote_real, 0));
        assert!(bit(&b.quote_real, 4));

        // "a\\\" — odd run of 3: quote escaped.
        let input = br#""a\\\""#;
        let b = classify(input);
        assert!(!bit(&b.quote_real, 5));
    }

    #[test]
    fn escape_carry_crosses_the_64_byte_word_seam() {
        // Backslash at byte 63, quote at byte 64: the escape state must
        // carry from bitmap word 0 into word 1.
        let mut input = vec![b' '; 63];
        input.push(b'\\'); // byte 63
        input.push(b'"'); // byte 64
        let b = classify(&input);
        assert_eq!(b.quote_real.len(), 2);
        assert!(!bit(&b.quote_real, 64), "quote at 64 is escaped via carry");

        // Even run straddling the seam: backslashes at 62..=63, quote at 64
        // is real.
        let mut input = vec![b' '; 62];
        input.extend_from_slice(br#"\\""#);
        let b = classify(&input);
        assert!(bit(&b.quote_real, 64), "even-run quote at 64 is real");

        // Backslash at 63 escaping a backslash at 64; quote at 65 is real.
        let mut input = vec![b' '; 63];
        input.extend_from_slice(br#"\\""#);
        let b = classify(&input);
        assert!(bit(&b.quote_real, 65));
    }

    #[test]
    fn candidates_are_computed_pre_in_string_masking() {
        // Ops and scalar starts *inside* the string literal still set bits;
        // stage 2 is responsible for masking them.
        //                0123456789
        let b = classify(b"\"a b,{:}\"");
        assert!(bit(&b.candidates, 1), "in-string scalar start 'a'");
        assert!(bit(&b.candidates, 3), "in-string scalar start 'b'");
        for i in [4, 5, 6, 7] {
            assert!(bit(&b.candidates, i), "in-string op at {i}");
        }
        assert!(bit(&b.quote_real, 0));
        assert!(bit(&b.quote_real, 8));
    }

    #[test]
    fn utf8_accepts_valid_sequences() {
        for s in [
            "".as_bytes(),
            b"plain ascii { } [ ] 123",
            "héllo wörld".as_bytes(),
            "\u{7FF}\u{800}\u{FFFD}\u{10000}\u{10FFFF}".as_bytes(),
            "😀 emoji".as_bytes(),
            b"\xF4\x8F\xBF\xBF", // U+10FFFF, the very last code point
            b"\xED\x9F\xBF",     // U+D7FF, just below the surrogate range
            b"\xEE\x80\x80",     // U+E000, just above the surrogate range
        ] {
            assert!(stage1_classify(s).is_ok(), "rejected valid UTF-8 {s:?}");
        }
    }

    #[test]
    fn utf8_rejects_with_first_offending_byte_offset() {
        // (input, expected offset of the first invalid sequence)
        let cases: &[(&[u8], u64)] = &[
            (b"\x80", 0),                   // stray continuation byte
            (b"ab\x80", 2),                 // ... after valid prefix
            (b"\xC0\xAF", 0),               // overlong 2-byte
            (b"\xC1\xBF", 0),               // overlong 2-byte
            (b"\xC2", 0),                   // truncated at EOF
            (b"\xC2x", 0),                  // truncated by non-continuation
            (b"\xE0\x80\x80", 0),           // overlong 3-byte
            (b"\xE0\x9F\xBF", 0),           // overlong 3-byte (max overlong)
            (b"\xED\xA0\x80", 0),           // surrogate U+D800
            (b"\xED\xBF\xBF", 0),           // surrogate U+DFFF
            (b"\xE2\x82", 0),               // truncated 3-byte at EOF
            (b"\xF0\x80\x80\x80", 0),       // overlong 4-byte
            (b"\xF0\x8F\xBF\xBF", 0),       // overlong 4-byte (max overlong)
            (b"\xF4\x90\x80\x80", 0),       // > U+10FFFF
            (b"\xF5\x80\x80\x80", 0),       // lead can never appear
            (b"\xFF", 0),                   // lead can never appear
            (b"{\"k\": \"\xE0\x80x\"}", 7), // mid-document
        ];
        for &(input, want) in cases {
            match stage1_classify(input) {
                Err(Error::Utf8 { offset }) => {
                    assert_eq!(offset, want, "offset for {input:?}");
                }
                other => panic!("expected Utf8 error for {input:?}, got {other:?}"),
            }
        }
    }

    #[test]
    fn utf8_offsets_match_std_from_utf8() {
        // The contract "offset = first byte of the invalid sequence" is
        // pinned to core::str::from_utf8's valid_up_to.
        let cases: &[&[u8]] = &[
            b"valid ascii",
            b"caf\xC3\xA9",
            b"\x80",
            b"ab\xC0\xAF cd",
            b"x\xED\xA0\x80y",
            b"\xF4\x90\x80\x80",
            b"abc\xE2\x82",
            b"\xC2",
            b"ok \xF0\x9F\x98\x80 then \xFF bad",
        ];
        for &input in cases {
            let ours = validate_utf8(input);
            match core::str::from_utf8(input) {
                Ok(_) => assert!(ours.is_ok(), "{input:?}"),
                Err(e) => match ours {
                    Err(Error::Utf8 { offset }) => {
                        assert_eq!(offset, e.valid_up_to() as u64, "{input:?}");
                    }
                    other => panic!("expected Utf8 error for {input:?}, got {other:?}"),
                },
            }
        }
    }
}
