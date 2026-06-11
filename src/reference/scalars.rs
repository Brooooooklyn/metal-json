//! Stage 5 — scalar (number + literal) parsing.
//!
//! Scalar oracle for GPU kernel **K10** (number parse). The M4 kernel unit
//! tests run K10 and this function on identical token streams and diff the
//! parsed values bit-for-bit.
//!
//! Number handling follows the tape contract (`docs/tape-format.md`):
//!
//! 1. full JSON number grammar validation
//!    (`-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?`, and the grammar
//!    must consume the **entire** scalar run);
//! 2. type selection mirroring simdjson: integer literal that fits `i64` →
//!    [`ScalarValue::Int64`]; else fits `u64` → [`ScalarValue::UInt64`];
//!    everything else → [`ScalarValue::Double`];
//! 3. doubles via `str::parse::<f64>`, which is correctly rounded — this is
//!    the oracle bit pattern. (The GPU's Eisel-Lemire fast path and its
//!    CPU-fixup slow-path classification arrive in M4; the reference only
//!    has to produce the correct bits.) Grammar-valid numbers whose value
//!    overflows the finite `f64` range (e.g. `1e400`) are rejected like
//!    simdjson rejects infinities; underflow to `0.0` (e.g. `1e-400`) is
//!    accepted.
//!
//! Literals re-use the same byte check stage 3 applies (`true`/`false`/
//! `null`, byte-exact, followed by a non-scalar byte), so this stage can be
//! driven directly by kernel tests without stage 3 running first.

use super::classify::{is_op, is_ws};
use super::tokens::{Token, TokenKind};
use crate::error::{Error, Result, SyntaxErrorKind};

/// A parsed scalar value, ready for tape emission.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalarValue {
    /// Integer literal that fits `i64` (tape tag `l`).
    Int64(i64),
    /// Integer literal in `i64::MAX+1 ..= u64::MAX` (tape tag `u`).
    UInt64(u64),
    /// Everything else (tape tag `d`), correctly rounded.
    Double(f64),
    /// `true`.
    True,
    /// `false`.
    False,
    /// `null`.
    Null,
}

/// A scalar token's parsed value, tagged with the token it came from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParsedScalar {
    /// Index into the stage-2 token stream of the `ScalarStart` token.
    pub token_index: u32,
    /// The parsed value.
    pub value: ScalarValue,
}

/// Stage 5: parse every `ScalarStart` token into a typed value.
///
/// Returns one [`ParsedScalar`] per `ScalarStart` token, in token order.
///
/// # Errors
///
/// [`SyntaxErrorKind::InvalidNumber`] / [`SyntaxErrorKind::InvalidLiteral`]
/// / [`SyntaxErrorKind::UnexpectedToken`] at the scalar's byte offset, for
/// the first (in token order) offending scalar.
pub fn stage5_scalars(tokens: &[Token], input: &[u8]) -> Result<Vec<ParsedScalar>> {
    let mut out = Vec::new();
    for (i, tok) in tokens.iter().enumerate() {
        if tok.kind != TokenKind::ScalarStart {
            continue;
        }
        let pos = tok.pos as usize;
        let value = match input[pos] {
            b'-' | b'0'..=b'9' => parse_number(scalar_run(input, pos), tok.pos)?,
            b't' | b'f' | b'n' => check_literal(input, pos)?,
            _ => {
                // Stage 3 rejects these first in the full pipeline; kept
                // here so the stage is safe to drive directly.
                return Err(Error::Syntax {
                    offset: u64::from(tok.pos),
                    kind: SyntaxErrorKind::UnexpectedToken,
                });
            }
        };
        out.push(ParsedScalar {
            token_index: u32::try_from(i).expect("more than u32::MAX tokens"),
            value,
        });
    }
    Ok(out)
}

/// The scalar run starting at `pos`: every byte up to (exclusive) the next
/// whitespace, operator, `"`, or end of input.
#[inline]
pub(crate) fn scalar_run(input: &[u8], pos: usize) -> &[u8] {
    let end = input[pos..]
        .iter()
        .position(|&b| is_ws(b) || is_op(b) || b == b'"')
        .map_or(input.len(), |n| pos + n);
    &input[pos..end]
}

/// Byte-exact `true`/`false`/`null` check at `pos` (first byte must be
/// `t`/`f`/`n`), including the boundary rule: the byte after the literal
/// must not extend the scalar run (kills `truee`, `nulll`, …).
///
/// Shared between stage 3 (Layer-1 literal validation; kills
/// `n_object_bad_value.json` `["x", truth]` and
/// `n_incomplete_true.json`-style cases) and stage 5 (value production).
///
/// # Errors
///
/// [`SyntaxErrorKind::InvalidLiteral`] at `pos`.
pub(crate) fn check_literal(input: &[u8], pos: usize) -> Result<ScalarValue> {
    let (text, value): (&[u8], ScalarValue) = match input[pos] {
        b't' => (b"true", ScalarValue::True),
        b'f' => (b"false", ScalarValue::False),
        b'n' => (b"null", ScalarValue::Null),
        _ => unreachable!("check_literal called on a non-literal first byte"),
    };
    let end = pos + text.len();
    let ok = input.len() >= end
        && &input[pos..end] == text
        && (end == input.len() || is_ws(input[end]) || is_op(input[end]) || input[end] == b'"');
    if ok {
        Ok(value)
    } else {
        Err(Error::Syntax {
            offset: pos as u64,
            kind: SyntaxErrorKind::InvalidLiteral,
        })
    }
}

/// Validate the full JSON number grammar over `run` and parse it with the
/// tape contract's type selection. `pos` is the run's byte offset, used for
/// error reporting.
fn parse_number(run: &[u8], pos: u32) -> Result<ScalarValue> {
    let err = || Error::Syntax {
        offset: u64::from(pos),
        kind: SyntaxErrorKind::InvalidNumber,
    };

    // --- Grammar: -?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)? ---------
    let negative = run.first() == Some(&b'-');
    let mut i = usize::from(negative);

    let int_start = i;
    while i < run.len() && run[i].is_ascii_digit() {
        i += 1;
    }
    let int_digits = &run[int_start..i];
    if int_digits.is_empty() {
        return Err(err()); // "-", "-x", and runs like "x12" never reach here
    }
    if int_digits.len() > 1 && int_digits[0] == b'0' {
        return Err(err()); // leading zero: "012", "-012", "00"
    }

    let mut is_double = false;
    if i < run.len() && run[i] == b'.' {
        is_double = true;
        i += 1;
        let frac_start = i;
        while i < run.len() && run[i].is_ascii_digit() {
            i += 1;
        }
        if i == frac_start {
            return Err(err()); // "1.", "1.e5"
        }
    }
    if i < run.len() && matches!(run[i], b'e' | b'E') {
        is_double = true;
        i += 1;
        if i < run.len() && matches!(run[i], b'+' | b'-') {
            i += 1;
        }
        let exp_start = i;
        while i < run.len() && run[i].is_ascii_digit() {
            i += 1;
        }
        if i == exp_start {
            return Err(err()); // "1e", "1e+"
        }
    }
    if i != run.len() {
        return Err(err()); // trailing junk in the run: "1x", "1.2.3", "0x1"
    }

    // --- Type selection (tape contract / simdjson parity) -----------------
    if !is_double {
        // Accumulate into u128 with overflow detection; >39 digits simply
        // falls through to the double path.
        let mut magnitude: Option<u128> = Some(0);
        for &d in int_digits {
            magnitude = magnitude
                .and_then(|m| m.checked_mul(10))
                .and_then(|m| m.checked_add(u128::from(d - b'0')));
        }
        if let Some(m) = magnitude {
            if negative {
                if m <= 1 << 63 {
                    // -(2^63) == i64::MIN still fits.
                    #[allow(clippy::cast_possible_truncation)]
                    return Ok(ScalarValue::Int64((m as i128).wrapping_neg() as i64));
                }
            } else if let Ok(v) = i64::try_from(m) {
                return Ok(ScalarValue::Int64(v));
            } else if let Ok(v) = u64::try_from(m) {
                return Ok(ScalarValue::UInt64(v));
            }
        }
        // Out-of-range integer literal: fall through to the double path.
    }

    let text = core::str::from_utf8(run).expect("number grammar admits only ASCII");
    let value: f64 = text.parse().map_err(|_| err())?;
    if value.is_infinite() {
        return Err(err()); // simdjson rejects out-of-range doubles
    }
    Ok(ScalarValue::Double(value))
}

#[cfg(test)]
mod tests {
    use super::super::classify::stage1_classify;
    use super::super::tokens::stage2_tokens;
    use super::*;

    /// Parse `input` as a lone root scalar through stages 1–2 then 5.
    fn scalar(input: &[u8]) -> Result<ScalarValue> {
        let tokens = stage2_tokens(&stage1_classify(input).unwrap(), input);
        assert_eq!(tokens.len(), 1, "fixture {input:?} must be one scalar");
        stage5_scalars(&tokens, input).map(|v| v[0].value)
    }

    fn expect_double(input: &[u8]) -> f64 {
        match scalar(input) {
            Ok(ScalarValue::Double(d)) => d,
            other => panic!("expected Double for {input:?}, got {other:?}"),
        }
    }

    #[test]
    fn integer_fast_path_selects_int64() {
        let cases: &[(&[u8], i64)] = &[
            (b"0", 0),
            (b"-0", 0), // "-0" is an integer literal: l 0 (simdjson parity)
            (b"42", 42),
            (b"-1", -1),
            (b"9223372036854775807", i64::MAX),
            (b"-9223372036854775808", i64::MIN),
        ];
        for &(input, want) in cases {
            assert_eq!(
                scalar(input).unwrap(),
                ScalarValue::Int64(want),
                "{input:?}"
            );
        }
    }

    #[test]
    fn big_positive_integers_select_uint64() {
        let cases: &[(&[u8], u64)] = &[
            (b"9223372036854775808", 9_223_372_036_854_775_808), // i64::MAX + 1
            (b"18446744073709551615", u64::MAX),
        ];
        for &(input, want) in cases {
            assert_eq!(
                scalar(input).unwrap(),
                ScalarValue::UInt64(want),
                "{input:?}"
            );
        }
    }

    #[test]
    fn out_of_integer_range_falls_to_double() {
        // u64::MAX + 1
        assert_eq!(
            expect_double(b"18446744073709551616").to_bits(),
            1.844_674_407_370_955_2e19_f64.to_bits()
        );
        // i64::MIN - 1
        assert_eq!(
            expect_double(b"-9223372036854775809").to_bits(),
            (-9.223_372_036_854_776e18_f64).to_bits()
        );
        // Way past u128 too (>39 digits).
        let huge = b"99999999999999999999999999999999999999999999999999";
        assert_eq!(expect_double(huge).to_bits(), 1e50_f64.to_bits());
    }

    #[test]
    fn fraction_or_exponent_selects_double() {
        for input in [
            &b"2.5"[..],
            b"-0.0",
            b"1e1",
            b"1E+5",
            b"1e-5",
            b"0.5e10",
            b"1e308",
            b"5e-324",
            b"0e0",
            b"0.1",
            b"1e23",
            b"123456789012345678901234567890.5",
        ] {
            let text = core::str::from_utf8(input).unwrap();
            let want: f64 = text.parse().unwrap();
            assert_eq!(
                expect_double(input).to_bits(),
                want.to_bits(),
                "bits for {text}"
            );
        }
    }

    #[test]
    fn negative_zero_double_keeps_its_sign_bit() {
        assert_eq!(expect_double(b"-0.0").to_bits(), (-0.0f64).to_bits());
        assert_ne!(expect_double(b"-0.0").to_bits(), 0.0f64.to_bits());
        // ... while integer "-0" is Int64(0), checked above.
    }

    #[test]
    fn underflow_is_zero_but_overflow_is_an_error() {
        assert_eq!(expect_double(b"1e-400").to_bits(), 0.0f64.to_bits());
        assert_eq!(expect_double(b"-1e-400").to_bits(), (-0.0f64).to_bits());
        for input in [&b"1e400"[..], b"-1e400", b"1e309", b"2e308"] {
            assert!(
                matches!(
                    scalar(input),
                    Err(Error::Syntax {
                        kind: SyntaxErrorKind::InvalidNumber,
                        ..
                    })
                ),
                "{input:?} must be rejected like simdjson rejects infinities"
            );
        }
    }

    #[test]
    fn number_grammar_rejections() {
        // Each kills its JSONTestSuite n_number_* relative:
        let cases: &[&[u8]] = &[
            b"01",        // n_number_with_leading_zero.json
            b"-01",       // n_number_neg_int_starting_with_zero.json
            b"00",        // n_number_0_capital_E? (leading zero family)
            b"-",         // n_number_minus_sign_with_trailing_garbage-ish
            b"1.",        // n_number_real_without_fractional_part.json
            b"1e",        // n_number_with_alpha_char family ("1e" truncated)
            b"1e+",       // n_number_1e+.json? (empty exponent)
            b"1eE2",      // n_number_1eE2.json
            b"0e",        // n_number_0e.json? (empty exponent)
            b"0x1",       // n_number_hex_1_digit.json
            b"1.2.3",     // two fraction parts
            b"1x",        // trailing junk in the run
            b"-x",        // no digits after the sign
            b"--1",       // n_number_minus_minus? (double sign)
            b"1e1.5",     // junk after the exponent digits
            b"123\x00",   // NUL extends the scalar run -> junk
            b"-Infinity", // n_number_minus_infinity.json
        ];
        for &input in cases {
            assert!(
                matches!(
                    scalar(input),
                    Err(Error::Syntax {
                        kind: SyntaxErrorKind::InvalidNumber,
                        ..
                    })
                ),
                "{input:?} must fail the number grammar"
            );
        }
    }

    #[test]
    fn number_error_offset_is_the_token_position() {
        let input = br#"[1, 2.5, 0x1]"#;
        let tokens = stage2_tokens(&stage1_classify(input).unwrap(), input);
        match stage5_scalars(&tokens, input) {
            Err(Error::Syntax { offset, kind }) => {
                assert_eq!(offset, 9);
                assert_eq!(kind, SyntaxErrorKind::InvalidNumber);
            }
            other => panic!("expected InvalidNumber, got {other:?}"),
        }
    }

    #[test]
    fn numbers_at_end_of_input_parse() {
        // No delimiter after the number: the run ends at EOF.
        assert_eq!(scalar(b"42").unwrap(), ScalarValue::Int64(42));
        assert_eq!(expect_double(b"2.5").to_bits(), 2.5f64.to_bits());
    }

    #[test]
    fn literals_parse_and_garbage_is_rejected() {
        assert_eq!(scalar(b"true").unwrap(), ScalarValue::True);
        assert_eq!(scalar(b"false").unwrap(), ScalarValue::False);
        assert_eq!(scalar(b"null").unwrap(), ScalarValue::Null);
        for input in [&b"tru"[..], b"truee", b"nul", b"nulll", b"fals", b"nan"] {
            assert!(
                matches!(
                    scalar(input),
                    Err(Error::Syntax {
                        kind: SyntaxErrorKind::InvalidLiteral,
                        ..
                    })
                ),
                "{input:?} must be an invalid literal"
            );
        }
    }

    #[test]
    fn literal_boundaries_accept_delimiters() {
        // `[true]` — the ']' terminates the literal run.
        let input = b"[true]";
        let tokens = stage2_tokens(&stage1_classify(input).unwrap(), input);
        let scalars = stage5_scalars(&tokens, input).unwrap();
        assert_eq!(scalars.len(), 1);
        assert_eq!(scalars[0].value, ScalarValue::True);
        assert_eq!(scalars[0].token_index, 1);
    }

    #[test]
    fn capitalized_or_alien_scalars_are_unexpected_tokens() {
        for input in [&b"True"[..], b"NaN", b"*", b"'a'"] {
            assert!(
                matches!(
                    scalar(input),
                    Err(Error::Syntax {
                        kind: SyntaxErrorKind::UnexpectedToken,
                        ..
                    })
                ),
                "{input:?}"
            );
        }
    }

    #[test]
    fn parsed_scalars_carry_their_token_index() {
        let input = b"[1,true,2.5]";
        let tokens = stage2_tokens(&stage1_classify(input).unwrap(), input);
        let scalars = stage5_scalars(&tokens, input).unwrap();
        let indices: Vec<u32> = scalars.iter().map(|s| s.token_index).collect();
        assert_eq!(indices, vec![1, 3, 5]);
        assert_eq!(scalars[0].value, ScalarValue::Int64(1));
        assert_eq!(scalars[1].value, ScalarValue::True);
        assert_eq!(scalars[2].value, ScalarValue::Double(2.5));
    }
}
