//! Scalar CPU reference pipeline (`cpu-reference` feature): the correctness
//! **oracle**. It produces the exact target tape of `docs/tape-format.md`;
//! every GPU kernel/stage of M2-M4 is diffed against it.
//!
//! # Architecture: one pure function per GPU stage
//!
//! The pipeline is deliberately split into separate pure functions with
//! inspectable intermediate outputs, mirroring the planned GPU kernels, so
//! the M2-M4 kernel unit tests can run "GPU kernel K vs reference stage K"
//! on identical inputs:
//!
//! | Reference stage | GPU kernels | Output |
//! |---|---|---|
//! | [`stage1_classify`] | K1 (+K2 carry) | [`Bitmaps`]: escape-resolved quotes, candidates; UTF-8 verdict |
//! | [`stage2_tokens`] | K3/K5 (+K2/K4) | `Vec<`[`Token`]`>` via quote-parity prefix-XOR masking |
//! | [`stage3_validate_local`] | K6 | Layer-1 verdict + tape footprints + [`SkeletonRecord`]s |
//! | [`stage4_structure`] | K8/K9 | depths, counting-sort order, pair map, comma context, child counts |
//! | [`stage5_scalars`] | K10 | parsed numbers (`i64`/`u64`/`f64` per the tape contract) + literals |
//! | [`stage6_strings`] | K11 | unescaped string records + stringbuf offsets |
//! | [`emit_tape`] | K7/K12/K13 | ([`TapeBuffer`], [`StringBuffer`]) |
//!
//! [`parse`] wires them together; performance is explicitly *not* a goal —
//! clarity and correctness are.
//!
//! # Error policy (documented deviation from the GPU)
//!
//! The GPU reduces all errors with `atomic_min` over
//! `(offset << 32) | code`, so the **globally** earliest byte offset wins.
//! The reference instead reports the first error in **stage order**
//! (1 → 3 → 4 → 5 → 6), applying earliest-offset-wins only *within* stage
//! 4 (the one stage that discovers errors out of document order). On
//! multi-error documents the two backends may therefore disagree about
//! *which* error is reported — never about *whether* parsing fails.
//! Differential tests compare `Ok`/`Err` and, on `Ok`, the tape bytes.
//! Relatedly, Layer 1 is context-free, so top-level trailing content like
//! `{} {}` surfaces as `MissingComma` rather than
//! [`Error::TrailingContent`] (which is reported for separator-led
//! trailing content such as `{},1`).

mod classify;
mod emit;
mod scalars;
mod strings;
mod structure;
mod tokens;
mod validate;

pub use classify::{Bitmaps, stage1_classify};
pub use emit::emit_tape;
pub use scalars::{ParsedScalar, ScalarValue, stage5_scalars};
pub use strings::{UnescapedString, stage6_strings};
pub use structure::{NO_MATCH, Stage4Output, stage4_structure};
pub use tokens::{Token, TokenKind, stage2_tokens};
pub use validate::{SkeletonRecord, Stage3Output, stage3_validate_local};

use crate::error::{Error, Result};
use crate::parser::ParserOptions;
use crate::tape::{StringBuffer, TapeBuffer};

/// Maximum input size the reference pipeline accepts.
///
/// Token positions and tape indices are `u32` (matching the GPU layouts and
/// the 32-bit container-index field of the tape contract). The tape never
/// exceeds `input_len + 3` words, so capping a comfortable margin below
/// `u32::MAX` keeps every index representable.
pub const MAX_INPUT_BYTES: u64 = u32::MAX as u64 - 64;

/// Parse `input` to a tape-format-v1 `(tape, string buffer)` pair with the
/// scalar CPU pipeline.
///
/// This is the contract `Parser::parse` (the `Backend::CpuReference` arm)
/// builds a [`Document`](crate::Document) from.
///
/// # Errors
///
/// - [`Error::InputTooLarge`] above [`MAX_INPUT_BYTES`];
/// - [`Error::Utf8`] from stage 1;
/// - [`Error::Syntax`] (including `EmptyInput` for empty / whitespace-only
///   input) from stages 3-6;
/// - [`Error::DepthLimit`] (nesting beyond `opts.max_depth`) and
///   [`Error::TrailingContent`] from stage 4.
pub fn parse(input: &[u8], opts: &ParserOptions) -> Result<(TapeBuffer, StringBuffer)> {
    if input.len() as u64 > MAX_INPUT_BYTES {
        return Err(Error::InputTooLarge {
            len: input.len() as u64,
            max: MAX_INPUT_BYTES,
        });
    }
    let bitmaps = stage1_classify(input)?;
    let tokens = stage2_tokens(&bitmaps, input);
    let stage3 = stage3_validate_local(&tokens, input)?;
    let stage4 = stage4_structure(&stage3.skeleton, opts.max_depth)?;
    let scalars = stage5_scalars(&tokens, input)?;
    let strings = stage6_strings(&tokens, input)?;
    Ok(emit_tape(&tokens, &stage3, &stage4, &scalars, &strings))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::Document;
    use crate::error::SyntaxErrorKind;
    use crate::value::{Value, ValueKind};

    fn parse_doc(input: &[u8]) -> Result<Document> {
        let (tape, strings) = parse(input, &ParserOptions::default())?;
        Ok(Document::from_parts(tape, strings))
    }

    #[test]
    fn empty_and_whitespace_only_inputs_are_empty_input_errors() {
        for input in [&b""[..], b" ", b"\t\n\r ", b"\n"] {
            assert!(
                matches!(
                    parse(input, &ParserOptions::default()),
                    Err(Error::Syntax {
                        offset: 0,
                        kind: SyntaxErrorKind::EmptyInput,
                    })
                ),
                "{input:?}"
            );
        }
    }

    #[test]
    fn root_scalars_parse_to_minimal_tapes() {
        let doc = parse_doc(b"42").unwrap();
        assert_eq!(doc.root().as_i64(), Some(42));

        let doc = parse_doc(b"true").unwrap();
        assert_eq!(doc.root().as_bool(), Some(true));

        let doc = parse_doc(b"\"x\"").unwrap();
        assert_eq!(doc.root().as_str(), Some("x"));

        let doc = parse_doc(b"-0.0").unwrap();
        assert_eq!(
            doc.root().as_f64().map(f64::to_bits),
            Some((-0.0f64).to_bits())
        );

        // Surrounding whitespace is fine.
        let doc = parse_doc(b"  null \n").unwrap();
        assert!(doc.root().is_null());
    }

    #[test]
    fn nesting_to_the_depth_limit_parses_and_one_past_fails() {
        let nest = |depth: usize| {
            let mut s = "[".repeat(depth);
            s.push_str(&"]".repeat(depth));
            s.into_bytes()
        };
        assert!(parse_doc(&nest(1024)).is_ok());
        match parse(&nest(1025), &ParserOptions::default()) {
            Err(Error::DepthLimit { offset, limit }) => {
                assert_eq!(offset, 1024);
                assert_eq!(limit, 1024);
            }
            other => panic!("expected DepthLimit, got {other:?}"),
        }
        // The limit is an option.
        let opts = ParserOptions {
            max_depth: 2,
            ..ParserOptions::default()
        };
        assert!(parse(&nest(2), &opts).is_ok());
        assert!(matches!(
            parse(&nest(3), &opts),
            Err(Error::DepthLimit { limit: 2, .. })
        ));
    }

    #[test]
    fn rejection_smoke_across_stages() {
        // One representative per stage; exhaustive cases live in the
        // per-stage test modules.
        let cases: &[&[u8]] = &[
            b"{\"k\": \"\xE0\x80\"}", // stage 1: UTF-8
            b"[1 2]",                 // stage 3: adjacency
            b"\"abc",                 // stage 3: unterminated string
            b"[1",                    // stage 4: balance
            b"1,2",                   // stage 4: trailing content
            br#"{"a":1,"b"}"#,        // stage 4: comma context
            b"[01]",                  // stage 5: number grammar
            br#"["\q"]"#,             // stage 6: bad escape
        ];
        for &input in cases {
            assert!(
                parse(input, &ParserOptions::default()).is_err(),
                "{:?} must fail",
                String::from_utf8_lossy(input)
            );
        }
    }

    #[test]
    fn duplicate_keys_are_all_on_the_tape() {
        let doc = parse_doc(br#"{"k":1,"k":2}"#).unwrap();
        let entries: Vec<(&str, i64)> = doc
            .root()
            .entries()
            .map(|(k, v)| (k, v.as_i64().unwrap()))
            .collect();
        assert_eq!(entries, vec![("k", 1), ("k", 2)]);
        // get() picks the first, simdjson-style.
        assert_eq!(doc.root().get("k").unwrap().as_i64(), Some(1));
    }

    /// Recursively compare our tape walk against serde_json's parse of the
    /// same document.
    fn assert_matches_serde(ours: Value<'_>, serde: &serde_json::Value) {
        match serde {
            serde_json::Value::Null => assert!(ours.is_null()),
            serde_json::Value::Bool(b) => assert_eq!(ours.as_bool(), Some(*b)),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    assert_eq!(ours.kind(), ValueKind::Int64);
                    assert_eq!(ours.as_i64(), Some(i));
                } else if let Some(u) = n.as_u64() {
                    assert_eq!(ours.kind(), ValueKind::UInt64);
                    assert_eq!(ours.as_u64(), Some(u));
                } else {
                    assert_eq!(ours.kind(), ValueKind::Double);
                    assert_eq!(
                        ours.as_f64().map(f64::to_bits),
                        n.as_f64().map(f64::to_bits),
                        "f64 bits must match str::parse exactly"
                    );
                }
            }
            serde_json::Value::String(s) => assert_eq!(ours.as_str(), Some(s.as_str())),
            serde_json::Value::Array(items) => {
                assert_eq!(ours.kind(), ValueKind::Array);
                assert_eq!(ours.len(), Some(items.len()));
                let elements: Vec<Value<'_>> = ours.elements().collect();
                assert_eq!(elements.len(), items.len());
                for (v, s) in elements.iter().zip(items) {
                    assert_matches_serde(*v, s);
                }
            }
            serde_json::Value::Object(members) => {
                assert_eq!(ours.kind(), ValueKind::Object);
                assert_eq!(ours.len(), Some(members.len()));
                // preserve_order is enabled: both sides are in doc order.
                let entries: Vec<(&str, Value<'_>)> = ours.entries().collect();
                assert_eq!(entries.len(), members.len());
                for ((our_key, our_value), (serde_key, serde_value)) in entries.iter().zip(members)
                {
                    assert_eq!(our_key, serde_key);
                    assert_matches_serde(*our_value, serde_value);
                }
            }
        }
    }

    #[test]
    fn differential_against_serde_json() {
        let bs = '\\';
        let docs: Vec<String> = vec![
            r#"{"a":[1,2.5],"b":"x"}"#.to_owned(),
            r#"[0, -1, 9223372036854775807, -9223372036854775808,
                18446744073709551615, 18446744073709551616, 1e-5, -0.0,
                0.1, 1e23, 5e-324, 1.7976931348623157e308]"#
                .to_owned(),
            r#"{"nested":{"deep":[[[{"x":[null,true,false]}]]]},"":""}"#.to_owned(),
            // Escapes (built without literal backslash-u sequences in this
            // source file): \" \\ \n \t, NUL, an astral surrogate pair.
            format!(
                r#"["{bs}"quote{bs}" {bs}{bs} {bs}n {bs}t", "{bs}u0000interior", "{bs}uD83D{bs}uDE00"]"#
            ),
            // Whitespace torture.
            "\t{\n\"k\"\r:\t[\n1 ,\r2\t]\n}\r".to_owned(),
            // 200 sibling members (unique keys: serde's map would collapse
            // duplicates; duplicate-key tape behavior is tested separately).
            {
                let members: Vec<String> =
                    (0..200).map(|i| format!(r#""k{i}":{}"#, i % 7)).collect();
                format!("{{{}}}", members.join(","))
            },
        ];
        for json in &docs {
            let doc = parse_doc(json.as_bytes()).unwrap_or_else(|e| {
                panic!("reference failed on {json:?}: {e}");
            });
            let serde: serde_json::Value = serde_json::from_str(json).unwrap();
            assert_matches_serde(doc.root(), &serde);
        }
    }

    #[test]
    fn differential_rejects_what_serde_rejects() {
        // Quick agreement check on the reject side too (full JSONTestSuite
        // runs land with the M1 verification harness).
        let cases: &[&[u8]] = &[
            b"",
            b"[",
            b"]",
            b"{",
            b"[1,]",
            b"{\"a\":1,}",
            b"[01]",
            b"\"a",
            b"[\"a\" 1]",
            b"nul",
            b"{} []",
            b"1 2",
        ];
        for &input in cases {
            assert!(
                serde_json::from_slice::<serde_json::Value>(input).is_err(),
                "fixture bug: serde accepts {input:?}"
            );
            assert!(
                parse(input, &ParserOptions::default()).is_err(),
                "we must reject {input:?}"
            );
        }
    }

    #[test]
    fn string_records_pack_in_document_order() {
        let (tape, strings) = parse(br#"{"k":"v"}"#, &ParserOptions::default()).unwrap();
        // r { "k "v } r
        assert_eq!(tape.len(), 6);
        assert_eq!(strings.record_str(crate::tape::string_offset(tape[2])), "k");
        assert_eq!(strings.record_str(crate::tape::string_offset(tape[3])), "v");
        assert_eq!(strings.len(), 6 + 6); // two [len u32][1 byte][NUL] records
    }

    #[test]
    fn interior_nul_from_escape_survives_the_round_trip() {
        let bs = '\\';
        let json = format!(r#"["a{bs}u0000b"]"#);
        let doc = parse_doc(json.as_bytes()).unwrap();
        let value = doc.root().at(0).unwrap();
        assert_eq!(value.as_str(), Some("a\0b"));
    }

    #[test]
    fn trailing_content_after_root_value() {
        assert!(matches!(
            parse(b"{},1", &ParserOptions::default()),
            Err(Error::TrailingContent { offset: 2 })
        ));
        // Starter-led trailing content dies in Layer 1 (documented policy).
        assert!(matches!(
            parse(b"{} {}", &ParserOptions::default()),
            Err(Error::Syntax {
                kind: SyntaxErrorKind::MissingComma,
                offset: 3,
            })
        ));
    }
}
