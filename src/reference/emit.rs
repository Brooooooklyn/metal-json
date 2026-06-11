//! Tape emission — assembling the stage outputs into the final tape.
//!
//! Scalar oracle for GPU kernels **K7** (spine scan of tape footprints →
//! tape positions), **K12** (container tape words via the pair map) and
//! **K13** (root words / finalize). The interesting property mirrored from
//! the GPU: emission needs **no stack** — every token's tape position is
//! known up front from the footprint prefix sum, so container open words
//! are written complete (end index + child count) in a single pass, never
//! patched.
//!
//! Layout produced is tape format v1 exactly as `docs/tape-format.md`
//! specifies; string records land at the offsets stage 6 precomputed (the
//! exclusive prefix sum of `raw_len + 5` in document order — the K7 scan).
//! When an escape shrank a string, the slack before the next slot (and
//! after the last record) is a **gap**: the reference zero-fills every gap
//! byte so its output is deterministic, the total buffer size being the
//! full `Σ (raw_len + 5)`. GPU gap bytes are unspecified; kernel diff
//! tests compare per-record bytes + tape offsets only.

use super::scalars::{ParsedScalar, ScalarValue};
use super::strings::UnescapedString;
use super::structure::Stage4Output;
use super::tokens::{Token, TokenKind};
use super::validate::Stage3Output;
use crate::tape::{
    STRING_RECORD_HEADER_BYTES, STRING_RECORD_TRAILER_BYTES, StringBuffer, TAG_END_ARRAY,
    TAG_END_OBJECT, TAG_START_ARRAY, TAG_START_OBJECT, TapeBuffer, double_bits, int64_bits,
    make_close, make_double_marker, make_false, make_final_root, make_int64_marker, make_null,
    make_open, make_root, make_string, make_true, make_uint64_marker,
};

/// Assemble the final `(tape, string buffer)` pair from the outputs of
/// stages 2-6 (which must all describe the **same** validated token
/// stream).
///
/// # Panics
///
/// On internally inconsistent stage outputs (mismatched token indices,
/// missing pair map entries). [`parse`](super::parse) can never trigger
/// this: stages 3-6 have validated everything by the time emission runs.
#[must_use]
pub fn emit_tape(
    tokens: &[Token],
    stage3: &Stage3Output,
    stage4: &Stage4Output,
    scalars: &[ParsedScalar],
    strings: &[UnescapedString],
) -> (TapeBuffer, StringBuffer) {
    let n = tokens.len();
    assert_eq!(stage3.footprints.len(), n, "footprint per token");

    // Tape position of each token: 1 (for tape[0], the root word) plus the
    // exclusive prefix sum of the footprints. On the GPU this is the K7
    // spine scan.
    let mut tape_pos = vec![0u32; n];
    let mut running: u32 = 1;
    for (t, fp) in stage3.footprints.iter().enumerate() {
        tape_pos[t] = running;
        running += fp;
    }
    let final_root_index = u64::from(running);

    // token index -> skeleton index (for pair-map lookups).
    let mut skel_of_token = vec![u32::MAX; n];
    for (si, rec) in stage3.skeleton.iter().enumerate() {
        skel_of_token[rec.token_index as usize] =
            u32::try_from(si).expect("skeleton larger than the token stream");
    }
    let partner_tape_pos = |token_index: usize| -> u32 {
        let si = skel_of_token[token_index] as usize;
        let partner_si = stage4.match_index[si] as usize;
        let partner_token = stage3.skeleton[partner_si].token_index as usize;
        tape_pos[partner_token]
    };

    // Total string-buffer size: Σ (raw_len + 5) — equivalently, one past
    // the last allocated slot. (On the GPU this is the K7 scan total.)
    let stringbuf_size = strings.last().map_or(0, |record| {
        record.record_offset
            + (STRING_RECORD_HEADER_BYTES + record.raw_len as usize + STRING_RECORD_TRAILER_BYTES)
                as u64
    });

    let mut tape = TapeBuffer::with_capacity(running as usize + 1);
    let mut stringbuf =
        StringBuffer::with_capacity(usize::try_from(stringbuf_size).expect("stringbuf size"));
    tape.push(make_root(final_root_index));

    let mut next_scalar = scalars.iter();
    let mut next_string = strings.iter();

    for (t, tok) in tokens.iter().enumerate() {
        debug_assert!(
            matches!(
                tok.kind,
                TokenKind::QuoteClose | TokenKind::Colon | TokenKind::Comma
            ) || tape.len() == tape_pos[t] as usize,
            "emission position drifted from the K7 prefix sum"
        );
        match tok.kind {
            TokenKind::LBrace | TokenKind::LBracket => {
                let tag = if tok.kind == TokenKind::LBrace {
                    TAG_START_OBJECT
                } else {
                    TAG_START_ARRAY
                };
                let si = skel_of_token[t] as usize;
                // One past the matching close word; complete on first write.
                let end_index = partner_tape_pos(t) + 1;
                tape.push(make_open(tag, end_index, stage4.child_counts[si]));
            }
            TokenKind::RBrace | TokenKind::RBracket => {
                let tag = if tok.kind == TokenKind::RBrace {
                    TAG_END_OBJECT
                } else {
                    TAG_END_ARRAY
                };
                tape.push(make_close(tag, partner_tape_pos(t)));
            }
            TokenKind::QuoteOpen => {
                let record = next_string.next().expect("stage6 record per QuoteOpen");
                assert_eq!(record.token_index as usize, t, "string/token mismatch");
                // Place the record at its prefix-sum offset; if the previous
                // record shrank, the gap up to here is zero-filled.
                let offset = stringbuf.append_record_at(record.record_offset, &record.bytes);
                tape.push(make_string(offset));
            }
            TokenKind::ScalarStart => {
                let scalar = next_scalar.next().expect("stage5 value per ScalarStart");
                assert_eq!(scalar.token_index as usize, t, "scalar/token mismatch");
                match scalar.value {
                    ScalarValue::Int64(v) => {
                        tape.push(make_int64_marker());
                        tape.push(int64_bits(v));
                    }
                    ScalarValue::UInt64(v) => {
                        tape.push(make_uint64_marker());
                        tape.push(v);
                    }
                    ScalarValue::Double(v) => {
                        tape.push(make_double_marker());
                        tape.push(double_bits(v));
                    }
                    ScalarValue::True => {
                        tape.push(make_true());
                    }
                    ScalarValue::False => {
                        tape.push(make_false());
                    }
                    ScalarValue::Null => {
                        tape.push(make_null());
                    }
                }
            }
            // No tape words of their own.
            TokenKind::QuoteClose | TokenKind::Colon | TokenKind::Comma => {}
        }
    }

    // Zero-fill the trailing gap if the last record shrank: the reference
    // buffer is always exactly Σ (raw_len + 5) bytes.
    stringbuf.pad_to(stringbuf_size);

    let last = tape.push(make_final_root());
    debug_assert_eq!(
        last as u64, final_root_index,
        "final root lands at tape[len-1]"
    );
    (tape, stringbuf)
}

#[cfg(test)]
mod tests {
    use super::super::classify::stage1_classify;
    use super::super::scalars::stage5_scalars;
    use super::super::strings::stage6_strings;
    use super::super::structure::stage4_structure;
    use super::super::tokens::stage2_tokens;
    use super::super::validate::stage3_validate_local;
    use super::*;
    use crate::parser::DEFAULT_MAX_DEPTH;

    /// Drive all stages and emit.
    fn emit(input: &[u8]) -> (TapeBuffer, StringBuffer) {
        let bitmaps = stage1_classify(input).unwrap();
        let tokens = stage2_tokens(&bitmaps, input);
        let s3 = stage3_validate_local(&tokens, input).unwrap();
        let s4 = stage4_structure(&s3.skeleton, DEFAULT_MAX_DEPTH).unwrap();
        let scalars = stage5_scalars(&tokens, input).unwrap();
        let strings = stage6_strings(&tokens, input).unwrap();
        emit_tape(&tokens, &s3, &s4, &scalars, &strings)
    }

    /// THE pipeline-level pin of docs/tape-format.md's worked example —
    /// the same expected words the `worked_example_matches_tape_format_doc`
    /// test in src/tape.rs builds by hand.
    #[test]
    fn worked_example_full_pipeline() {
        let (tape, strings) = emit(br#"{"a":[1,2.5],"b":"x\n"}"#);
        let expected: [u64; 13] = [
            0x7200_0000_0000_000C, // [0]  r -> 12
            0x7B00_0002_0000_000C, // [1]  { end=12 count=2
            0x2200_0000_0000_0000, // [2]  " offset=0  ("a")
            0x5B00_0002_0000_0009, // [3]  [ end=9 count=2
            0x6C00_0000_0000_0000, // [4]  l
            0x0000_0000_0000_0001, // [5]  1
            0x6400_0000_0000_0000, // [6]  d
            0x4004_0000_0000_0000, // [7]  2.5
            0x5D00_0000_0000_0003, // [8]  ] open=3
            0x2200_0000_0000_0006, // [9]  " offset=6  ("b")
            0x2200_0000_0000_000C, // [10] " offset=12 ("x\n")
            0x7D00_0000_0000_0001, // [11] } open=1
            0x7200_0000_0000_0000, // [12] r -> 0
        ];
        assert_eq!(tape.as_words(), &expected);
        // 20 bytes = slots of raw_len+5: 6 ("a") + 6 ("b") + 8 ("x\n",
        // raw len 3). The \n escape shrank the last record to 7 bytes, so
        // its slot ends with one zero gap byte.
        let expected_strings: [u8; 20] = [
            0x01, 0x00, 0x00, 0x00, 0x61, 0x00, // "a"
            0x01, 0x00, 0x00, 0x00, 0x62, 0x00, // "b"
            0x02, 0x00, 0x00, 0x00, 0x78, 0x0A, 0x00, // "x\n"
            0x00, // gap
        ];
        assert_eq!(strings.as_bytes(), &expected_strings);
    }

    /// Gap policy, pinned byte for byte: interior and trailing gaps are
    /// zero-filled, offsets come from the raw-length prefix sum.
    #[test]
    fn shrunk_records_leave_zero_filled_gaps() {
        // ["\n","x","\t"] — raw lens 2, 1, 2 → slots 7, 6, 7 at offsets
        // 0, 7, 13; every escape shrinks its record by one byte.
        let (tape, strings) = emit(br#"["\n","x","\t"]"#);
        assert_eq!(tape.as_words()[2], make_string(0));
        assert_eq!(tape.as_words()[3], make_string(7));
        assert_eq!(tape.as_words()[4], make_string(13));
        let expected: [u8; 20] = [
            0x01, 0x00, 0x00, 0x00, 0x0A, 0x00, // "\n" record (6 bytes)
            0x00, // interior gap
            0x01, 0x00, 0x00, 0x00, 0x78, 0x00, // "x" record (6 bytes)
            0x01, 0x00, 0x00, 0x00, 0x09, 0x00, // "\t" record (6 bytes)
            0x00, // trailing gap
        ];
        assert_eq!(strings.as_bytes(), &expected);
        // The records themselves decode exactly through the gaps.
        assert_eq!(strings.record_bytes(0), b"\n");
        assert_eq!(strings.record_bytes(7), b"x");
        assert_eq!(strings.record_bytes(13), b"\t");
    }

    /// A surrogate-pair escape shrinks 12 raw bytes to 4 content bytes:
    /// the next offset is unmoved and the 8 slack bytes are zeroed.
    #[test]
    fn surrogate_pair_gap_is_eight_zero_bytes() {
        let bs = '\\';
        let json = format!(r#"["{bs}uD83D{bs}uDE00","x"]"#);
        let (tape, strings) = emit(json.as_bytes());
        // Slot 0: raw len 12 → 17 bytes; slot 1 at 17.
        assert_eq!(tape.as_words()[2], make_string(0));
        assert_eq!(tape.as_words()[3], make_string(17));
        assert_eq!(strings.len(), 17 + 6);
        assert_eq!(strings.record_str(0), "\u{1F600}");
        // Record 0 occupies 4+4+1 = 9 bytes; bytes 9..17 are the gap.
        assert_eq!(&strings.as_bytes()[9..17], &[0u8; 8]);
        assert_eq!(strings.record_str(17), "x");
    }

    #[test]
    fn root_number() {
        let (tape, strings) = emit(b"42");
        assert_eq!(
            tape.as_words(),
            &[
                make_root(3),
                make_int64_marker(),
                int64_bits(42),
                make_final_root(),
            ]
        );
        assert!(strings.is_empty());
    }

    #[test]
    fn root_literals_and_string() {
        let (tape, _) = emit(b"true");
        assert_eq!(
            tape.as_words(),
            &[make_root(2), make_true(), make_final_root()]
        );

        let (tape, _) = emit(b"null");
        assert_eq!(
            tape.as_words(),
            &[make_root(2), make_null(), make_final_root()]
        );

        let (tape, strings) = emit(b"\"x\"");
        assert_eq!(
            tape.as_words(),
            &[make_root(2), make_string(0), make_final_root()]
        );
        assert_eq!(strings.record_str(0), "x");
    }

    #[test]
    fn negative_zero_double_bits_on_tape() {
        let (tape, _) = emit(b"-0.0");
        assert_eq!(tape.as_words()[1], make_double_marker());
        assert_eq!(tape.as_words()[2], (-0.0f64).to_bits());
    }

    #[test]
    fn uint64_value_word_is_the_raw_value() {
        let (tape, _) = emit(b"18446744073709551615");
        assert_eq!(tape.as_words()[1], make_uint64_marker());
        assert_eq!(tape.as_words()[2], u64::MAX);
    }

    #[test]
    fn sibling_containers_open_and_close_words() {
        // [[],{}]  — tokens: [0 [1 ]2 ,3 {4 }5 ]6
        // tape:    r [ [ ] { } ] r  (indices 0..=7)
        let (tape, _) = emit(b"[[],{}]");
        let words = tape.as_words();
        assert_eq!(words.len(), 8);
        assert_eq!(words[0], make_root(7));
        assert_eq!(words[1], make_open(TAG_START_ARRAY, 7, 2)); // outer
        assert_eq!(words[2], make_open(TAG_START_ARRAY, 4, 0)); // inner []
        assert_eq!(words[3], make_close(TAG_END_ARRAY, 2));
        assert_eq!(words[4], make_open(TAG_START_OBJECT, 6, 0)); // inner {}
        assert_eq!(words[5], make_close(TAG_END_OBJECT, 4));
        assert_eq!(words[6], make_close(TAG_END_ARRAY, 1));
        assert_eq!(words[7], make_final_root());
    }

    #[test]
    fn duplicate_keys_are_kept_verbatim() {
        // The tape does no deduplication (simdjson parity): both members
        // appear, in document order, and the count says 2.
        let (tape, strings) = emit(br#"{"a":1,"a":2}"#);
        let words = tape.as_words();
        // r { "a l 1 "a l 2 } r — 10 words.
        assert_eq!(words.len(), 10);
        assert_eq!(words[1], make_open(TAG_START_OBJECT, 9, 2));
        assert_eq!(
            strings.record_str(crate::tape::string_offset(words[2])),
            "a"
        );
        assert_eq!(
            strings.record_str(crate::tape::string_offset(words[5])),
            "a"
        );
        assert_eq!(int64_bits(1), words[4]);
        assert_eq!(int64_bits(2), words[7]);
        assert_eq!(words[8], make_close(TAG_END_OBJECT, 1));
    }

    #[test]
    fn empty_string_values_get_records() {
        let (tape, strings) = emit(br#"["",""]"#);
        let words = tape.as_words();
        assert_eq!(words[2], make_string(0));
        assert_eq!(words[3], make_string(5)); // slot = raw_len 0 + 5
        assert_eq!(strings.record_bytes(0), b"");
        assert_eq!(strings.record_bytes(5), b"");
    }
}
