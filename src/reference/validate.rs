//! Stage 3 — Layer-1 (local) validation + tape footprints + skeleton.
//!
//! Scalar oracle for GPU kernel **K6** (local validation, tape footprints,
//! and the skeleton/string/scalar lists). The M3 kernel unit tests run K6
//! and this function on identical token streams and diff footprints,
//! skeleton records and accept/reject verdicts.
//!
//! Layer 1 is **context-free**: every rule looks at a fixed window of the
//! token stream, which is what makes it data-parallel on the GPU (one
//! thread per token). It comprises exactly the checks the design assigns to
//! K6 — each documented below with a JSONTestSuite `n_*.json` case it
//! kills:
//!
//! 1. **The token-order table** ([`pair_allowed`]): which token kind may
//!    follow which, including virtual start/end boundaries.
//! 2. **The colon 4-token rule** ([`colon_rule`]): a `:` at token index `i`
//!    requires the window `(i-3, i-2, i-1, i)` to be
//!    (`{` or `,`, `QuoteOpen`, `QuoteClose`, `:`) — i.e. a colon is always
//!    preceded by a complete string key that itself follows `{` or `,`.
//!    Kills `n_array_colon_instead_of_comma.json` (`["": 1]`, the token
//!    before the key is `[`) and `{"a":"b":"c"}`-style double members (the
//!    token before the second key is `:`).
//! 3. **The object-first-member rule** ([`first_member_rule`]): when `{` is
//!    followed by a complete key pair, the next token must be `:`. Kills
//!    `n_object_comma_instead_of_colon.json` (`{"x", null}`) and
//!    `{"a"}`-style key-without-value objects.
//! 4. **Literal byte checks** (via [`check_literal`]): `true` / `false` /
//!    `null` byte-exact with a clean boundary. Kills
//!    `n_object_bad_value.json` (`["x", truth]`) and
//!    `n_structure_capitalized_True.json`-adjacent cases (a capital `T`
//!    scalar first byte is rejected as an unexpected token).
//!
//! What Layer 1 deliberately cannot see (no container context): colon /
//! comma *placement relative to the enclosing container* and bracket
//! pairing — that is stage 4 (Layer 2). One consequence to know about:
//! top-level trailing content such as `{} {}` is killed here by the
//! ender→starter ban and therefore surfaces as
//! [`SyntaxErrorKind::MissingComma`], not
//! [`Error::TrailingContent`](crate::Error::TrailingContent); separator-led
//! trailing content (`{},`) is detected by stage 4.
//!
//! Besides the verdict, this stage computes per-token **tape footprints**
//! (how many tape words the token will emit; stage `emit_tape` turns the
//! prefix sum of these into tape positions — on the GPU that prefix sum is
//! the K7 spine scan) and the **skeleton**: the structural subsequence
//! (brackets + colons + commas) that stage 4 sorts by depth.

use super::scalars::check_literal;
use super::tokens::{Token, TokenKind};
use crate::error::{Error, Result, SyntaxErrorKind};

/// One structural token (bracket, colon or comma) of the skeleton.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SkeletonRecord {
    /// Index of this element in the stage-2 token stream.
    pub token_index: u32,
    /// Byte offset in the input.
    pub pos: u32,
    /// The structural byte: one of `{` `}` `[` `]` `:` `,`. Open/close
    /// brackets of the same type differ by exactly `0x06` (`{`^`}` ==
    /// `[`^`]` == `0x06`), which stage 4's pair matching exploits.
    pub byte: u8,
}

/// Stage 3 outputs: per-token tape footprints + the skeleton.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stage3Output {
    /// `footprints[t]` = number of tape words token `t` emits:
    /// brackets and string opens 1, literals 1, numbers 2 (marker + value
    /// word), everything else 0.
    pub footprints: Vec<u32>,
    /// Brackets + colons + commas in document order.
    pub skeleton: Vec<SkeletonRecord>,
}

/// Stage 3: Layer-1 validation; on success returns footprints + skeleton.
///
/// # Errors
///
/// The first rule violation in token order, as [`Error::Syntax`] (kinds:
/// `EmptyInput`, `UnexpectedToken`, `MissingComma`, `MissingColon`,
/// `UnterminatedString`, `InvalidLiteral`, `UnbalancedBrackets`).
pub fn stage3_validate_local(tokens: &[Token], input: &[u8]) -> Result<Stage3Output> {
    if tokens.is_empty() {
        // Kills n_structure_no_data.json (``) and n_single_space.json (` `).
        return Err(Error::Syntax {
            offset: 0,
            kind: SyntaxErrorKind::EmptyInput,
        });
    }

    let mut footprints = Vec::with_capacity(tokens.len());
    let mut skeleton = Vec::new();

    for (i, tok) in tokens.iter().enumerate() {
        let prev = if i == 0 { None } else { Some(tokens[i - 1]) };
        check_adjacent(prev, Some(*tok), input.len())?;

        let mut push_skeleton = |byte: u8| {
            skeleton.push(SkeletonRecord {
                token_index: u32::try_from(i).expect("more than u32::MAX tokens"),
                pos: tok.pos,
                byte,
            });
        };

        match tok.kind {
            TokenKind::LBrace => {
                first_member_rule(tokens, i, input.len())?;
                push_skeleton(b'{');
                footprints.push(1);
            }
            TokenKind::LBracket => {
                push_skeleton(b'[');
                footprints.push(1);
            }
            TokenKind::RBrace => {
                push_skeleton(b'}');
                footprints.push(1);
            }
            TokenKind::RBracket => {
                push_skeleton(b']');
                footprints.push(1);
            }
            TokenKind::Colon => {
                colon_rule(tokens, i)?;
                push_skeleton(b':');
                footprints.push(0);
            }
            TokenKind::Comma => {
                push_skeleton(b',');
                footprints.push(0);
            }
            TokenKind::QuoteOpen => footprints.push(1),
            TokenKind::QuoteClose => footprints.push(0),
            TokenKind::ScalarStart => match input[tok.pos as usize] {
                // Numbers occupy two tape words (marker + value).
                b'-' | b'0'..=b'9' => footprints.push(2),
                b't' | b'f' | b'n' => {
                    check_literal(input, tok.pos as usize)?;
                    footprints.push(1);
                }
                // Kills n_structure_capitalized_True.json (`[True]`),
                // n_object_key_with_single_quotes.json (`{key: 'value'}`,
                // at `k`), n_array_star_inside.json (`[*]`), and any other
                // byte that cannot begin a JSON scalar.
                _ => {
                    return Err(Error::Syntax {
                        offset: u64::from(tok.pos),
                        kind: SyntaxErrorKind::UnexpectedToken,
                    });
                }
            },
        }
    }

    // Virtual end-of-input boundary.
    check_adjacent(tokens.last().copied(), None, input.len())?;

    Ok(Stage3Output {
        footprints,
        skeleton,
    })
}

/// THE token-order table: may `next` follow `prev`? `None` is the virtual
/// start (for `prev`) / end (for `next`) of the token stream.
///
/// Quote tokens come in adjacent open/close pairs by stage-2 construction,
/// so `QuoteClose` rows/columns only encode what may surround a complete
/// string.
fn pair_allowed(prev: Option<TokenKind>, next: Option<TokenKind>) -> bool {
    use TokenKind::*;
    match prev {
        // Start of input: any value starter. Bans `]` `}` `:` `,` first —
        // kills n_structure_end_array.json (`]`) and
        // n_structure_angle_bracket_.-style stray separators (`,` / `:`).
        None => next.is_none_or(TokenKind::is_value_start),

        // After `{`: a key string or `}`. Bans scalars/containers as keys —
        // kills n_object_unquoted_key.json (`{a: "b"}`),
        // n_object_non_string_key.json (`{1:1}`),
        // n_object_bracket_key.json (`{[: "x"}`), n_object_missing_key.json
        // (`{:"b"}`), `{,}`; bans end-of-input — unclosed lone `{`.
        Some(LBrace) => matches!(next, Some(QuoteOpen | RBrace)),

        // After `[`: any value starter or an immediate `]`. Bans `,` first —
        // kills n_array_comma_and_number.json (`[,1]`) and
        // n_array_just_comma.json (`[,]`); bans `}` — kills
        // n_structure_open_array_close_object-style `[}`; bans end-of-input
        // — kills n_structure_lone-open-bracket.json (`[`).
        Some(LBracket) => {
            matches!(next, Some(RBracket)) || next.is_some_and(TokenKind::is_value_start)
        }

        // After `:` or `,`: a value must start. Bans `]` after `,` — kills
        // n_array_extra_comma.json (`["",]`); bans `}` — kills
        // n_object_trailing_comma.json (`{"id":0,}`); bans `,,` — kills
        // n_array_double_comma.json (`[1,,2]`); bans `::` — kills
        // n_object_double_colon.json (`{"x"::"b"}`); bans `:}` — kills
        // `{"a":}`; bans end-of-input — kills `{"a":` and `[1,`.
        Some(Colon | Comma) => next.is_some_and(TokenKind::is_value_start),

        // An open quote is only ever followed by its close quote; at
        // end-of-input the string never terminated — kills
        // n_structure_unclosed_string-style `"abc`.
        Some(QuoteOpen) => matches!(next, Some(QuoteClose)),

        // After a complete string: separator, container close, or end of
        // input (root string). `Colon` is allowed ONLY here — a string is
        // the only legal key, so every other X→`:` pair is banned, killing
        // n_array_items_separated_by_semicolon.json (`[1:2]`).
        Some(QuoteClose) => matches!(next, None | Some(Colon | Comma | RBrace | RBracket)),

        // After a scalar or container close: separator, close, or end of
        // input. The ender→starter ban here kills
        // n_array_1_true_without_comma.json (`[1 true]`),
        // n_array_inner_array_no_comma.json (`[3[4]]`),
        // n_structure_double_array.json (`[][]`),
        // n_structure_object_with_trailing_garbage.json
        // (`{"a": true} "x"`), and n_structure_array_trailing_garbage.json
        // (`[1]x`). (n_object_missing_semicolon.json `{"a" "b"}` and
        // n_object_missing_colon.json `{"a" b}` violate this ban too, but
        // the object-first-member rule fires first, as MissingColon.)
        Some(ScalarStart | RBrace | RBracket) => {
            matches!(next, None | Some(Comma | RBrace | RBracket))
        }
    }
}

/// Apply [`pair_allowed`]; on a banned pair, pick the most descriptive
/// error kind and offset.
fn check_adjacent(prev: Option<Token>, next: Option<Token>, input_len: usize) -> Result<()> {
    use TokenKind::*;
    if pair_allowed(prev.map(|t| t.kind), next.map(|t| t.kind)) {
        return Ok(());
    }
    Err(match (prev, next) {
        // A string that never closed: report at its opening quote.
        (Some(p), _) if p.kind == QuoteOpen => Error::Syntax {
            offset: u64::from(p.pos),
            kind: SyntaxErrorKind::UnterminatedString,
        },
        // Input ends right after an open bracket: report the bracket.
        (Some(p), None) if matches!(p.kind, LBrace | LBracket) => Error::Syntax {
            offset: u64::from(p.pos),
            kind: SyntaxErrorKind::UnbalancedBrackets,
        },
        // Two values back to back: a comma is missing between them (at top
        // level this is trailing content; Layer 1 cannot tell the
        // difference — see the module docs).
        (Some(p), Some(n)) if p.kind.is_value_end() && n.kind.is_value_start() => Error::Syntax {
            offset: u64::from(n.pos),
            kind: SyntaxErrorKind::MissingComma,
        },
        (_, Some(n)) => Error::Syntax {
            offset: u64::from(n.pos),
            kind: SyntaxErrorKind::UnexpectedToken,
        },
        (_, None) => Error::Syntax {
            offset: input_len as u64,
            kind: SyntaxErrorKind::UnexpectedToken,
        },
    })
}

/// The colon 4-token rule: tokens `(i-3, i-2, i-1, i)` must be
/// (`{` or `,`, `QuoteOpen`, `QuoteClose`, `:`). The `i-1`/`i-2` half is
/// already enforced by the token-order table (only `QuoteClose` may precede
/// `:`, and a `QuoteClose` always follows its `QuoteOpen`), so only the
/// `i-3` token is checked here.
///
/// Kills `["": 1]` (`i-3` is `[`), `{"a":"b":"c"}` (`i-3` of the second
/// colon is `:`), and a root-level `"a":1` (no `i-3` at all).
fn colon_rule(tokens: &[Token], i: usize) -> Result<()> {
    if i >= 3 && matches!(tokens[i - 3].kind, TokenKind::LBrace | TokenKind::Comma) {
        Ok(())
    } else {
        Err(Error::Syntax {
            offset: u64::from(tokens[i].pos),
            kind: SyntaxErrorKind::UnexpectedToken,
        })
    }
}

/// The object-first-member rule: when `{` at token `i` is followed by a
/// complete key pair (`QuoteOpen`, `QuoteClose`), token `i+3` must be `:`.
///
/// Kills `{"x", null}` (n_object_comma_instead_of_colon.json), `{"a"}`,
/// and `{"a"` (key pair, then end of input). Incomplete key pairs are left
/// to the token-order table (`{"abc` must report the unterminated string,
/// not a missing colon).
fn first_member_rule(tokens: &[Token], i: usize, input_len: usize) -> Result<()> {
    let key_pair = tokens
        .get(i + 1)
        .is_some_and(|t| t.kind == TokenKind::QuoteOpen)
        && tokens
            .get(i + 2)
            .is_some_and(|t| t.kind == TokenKind::QuoteClose);
    if !key_pair {
        return Ok(()); // `{}` (fine) or a banned pair the table reports
    }
    match tokens.get(i + 3) {
        Some(t) if t.kind == TokenKind::Colon => Ok(()),
        Some(t) => Err(Error::Syntax {
            offset: u64::from(t.pos),
            kind: SyntaxErrorKind::MissingColon,
        }),
        None => Err(Error::Syntax {
            offset: input_len as u64,
            kind: SyntaxErrorKind::MissingColon,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::super::classify::stage1_classify;
    use super::super::tokens::stage2_tokens;
    use super::*;

    fn run(input: &[u8]) -> Result<Stage3Output> {
        let tokens = stage2_tokens(&stage1_classify(input).unwrap(), input);
        stage3_validate_local(&tokens, input)
    }

    fn expect_kind(input: &[u8], kind: SyntaxErrorKind) -> u64 {
        match run(input) {
            Err(Error::Syntax { offset, kind: k }) => {
                assert_eq!(k, kind, "kind for {:?}", String::from_utf8_lossy(input));
                offset
            }
            other => panic!(
                "expected Syntax {{ kind: {kind:?} }} for {:?}, got {other:?}",
                String::from_utf8_lossy(input)
            ),
        }
    }

    #[test]
    fn worked_example_footprints_and_skeleton() {
        let out = run(br#"{"a":[1,2.5],"b":"x\n"}"#).unwrap();
        // {  "a  ":  :  [  1  ,  2.5  ]  ,  "b  ":  :  "x  ":  }
        assert_eq!(
            out.footprints,
            vec![1, 1, 0, 0, 1, 2, 0, 2, 1, 0, 1, 0, 0, 1, 0, 1]
        );
        let bytes: Vec<u8> = out.skeleton.iter().map(|r| r.byte).collect();
        assert_eq!(bytes, b"{:[,],:}".to_vec());
        let token_indices: Vec<u32> = out.skeleton.iter().map(|r| r.token_index).collect();
        assert_eq!(token_indices, vec![0, 3, 4, 6, 8, 9, 12, 15]);
        let positions: Vec<u32> = out.skeleton.iter().map(|r| r.pos).collect();
        assert_eq!(positions, vec![0, 4, 5, 7, 11, 12, 16, 22]);
    }

    #[test]
    fn valid_token_sequences_pass() {
        for input in [
            &br#"{"a":{}}"#[..],
            br#"[[],[[]]]"#,
            br#"42"#,
            br#""root string""#,
            br#"true"#,
            br#"{"k":[1,{"n":null},"s"],"e":{}}"#,
            br#"[1,2]"#,
            br#"{"":0}"#, // empty key is fine
        ] {
            assert!(run(input).is_ok(), "{:?}", String::from_utf8_lossy(input));
        }
    }

    #[test]
    fn empty_and_whitespace_only_inputs() {
        // n_structure_no_data.json / n_single_space.json
        assert_eq!(expect_kind(b"", SyntaxErrorKind::EmptyInput), 0);
        assert_eq!(expect_kind(b" \t\n\r", SyntaxErrorKind::EmptyInput), 0);
    }

    #[test]
    fn token_order_table_rejections() {
        use SyntaxErrorKind::*;
        // (input, expected kind) — each is the n_*.json shape named in the
        // pair_allowed docs.
        let cases: &[(&[u8], SyntaxErrorKind)] = &[
            (b"]", UnexpectedToken), // n_structure_end_array
            (b"}", UnexpectedToken),
            (b",1", UnexpectedToken),
            (b"{a: 1}", UnexpectedToken),      // n_object_unquoted_key
            (b"{1:1}", UnexpectedToken),       // n_object_non_string_key
            (br#"{[: "x"}"#, UnexpectedToken), // n_object_bracket_key
            (br#"{:"b"}"#, UnexpectedToken),   // n_object_missing_key
            (b"{,}", UnexpectedToken),
            (b"[,1]", UnexpectedToken), // n_array_comma_and_number
            (b"[,]", UnexpectedToken),  // n_array_just_comma
            (b"[}", UnexpectedToken),
            (b"[1,,2]", UnexpectedToken),        // n_array_double_comma
            (br#"{"x"::"b"}"#, UnexpectedToken), // n_object_double_colon
            (br#"{"a":}"#, UnexpectedToken),
            (br#"["",]"#, UnexpectedToken),     // n_array_extra_comma
            (br#"{"id":0,}"#, UnexpectedToken), // n_object_trailing_comma
            (b"[1:2]", UnexpectedToken),        // n_array_items_separated_by_semicolon
            (b"[1 true]", MissingComma),        // n_array_1_true_without_comma
            (b"[3[4]]", MissingComma),          // n_array_inner_array_no_comma
            (br#"["a" "b"]"#, MissingComma),    // strings without a comma
            // In an object the first-member rule reports these first:
            (br#"{"a" "b"}"#, MissingColon), // n_object_missing_semicolon
            (br#"{"a" b}"#, MissingColon),   // n_object_missing_colon
            (b"[][]", MissingComma),         // n_structure_double_array
            (br#"{"a": true} "x""#, MissingComma), // trailing garbage
            (b"[1]x", MissingComma),         // n_structure_array_trailing_garbage
            (b"null null", MissingComma),
            (b"[1,", UnexpectedToken),      // comma then end of input
            (br#"{"a":"#, UnexpectedToken), // colon then end of input
        ];
        for &(input, kind) in cases {
            expect_kind(input, kind);
        }
    }

    #[test]
    fn unclosed_opens_at_end_of_input() {
        // Reported at the open bracket's own position.
        assert_eq!(expect_kind(b"{", SyntaxErrorKind::UnbalancedBrackets), 0);
        assert_eq!(expect_kind(b"[", SyntaxErrorKind::UnbalancedBrackets), 0);
        assert_eq!(expect_kind(b"[[", SyntaxErrorKind::UnbalancedBrackets), 1);
    }

    #[test]
    fn unterminated_strings_report_the_open_quote() {
        assert_eq!(
            expect_kind(b"\"abc", SyntaxErrorKind::UnterminatedString),
            0
        );
        assert_eq!(
            expect_kind(br#"{"a"#, SyntaxErrorKind::UnterminatedString),
            1
        );
        assert_eq!(
            expect_kind(br#"["a", "b"#, SyntaxErrorKind::UnterminatedString),
            6
        );
    }

    #[test]
    fn colon_4_token_rule() {
        use SyntaxErrorKind::*;
        // n_array_colon_instead_of_comma.json: token before the key is `[`.
        assert_eq!(expect_kind(br#"["": 1]"#, UnexpectedToken), 3);
        // Second colon's i-3 token is the first colon.
        expect_kind(br#"{"a":"b":"c"}"#, UnexpectedToken);
        // Root-level colon: no i-3 token at all.
        assert_eq!(expect_kind(br#""a":1"#, UnexpectedToken), 3);
    }

    #[test]
    fn object_first_member_rule() {
        use SyntaxErrorKind::*;
        // n_object_comma_instead_of_colon.json
        assert_eq!(expect_kind(br#"{"x", null}"#, MissingColon), 4);
        // Key with no value at all.
        expect_kind(br#"{"a"}"#, MissingColon);
        // Key pair then end of input: offset = input length.
        assert_eq!(expect_kind(br#"{"a""#, MissingColon), 4);
        // NOT this rule's job: a later member without a colon
        // (`{"foo":1, "a"}`) is the stage-4 comma-context rule.
        assert!(run(br#"{"foo":1, "a"}"#).is_ok());
    }

    #[test]
    fn literal_byte_checks() {
        use SyntaxErrorKind::*;
        // n_object_bad_value.json-style
        assert_eq!(expect_kind(br#"["x", truth]"#, InvalidLiteral), 6);
        expect_kind(b"[tru]", InvalidLiteral);
        expect_kind(b"false0", InvalidLiteral);
        // Bad scalar first bytes (n_structure_capitalized_True.json etc.)
        assert_eq!(expect_kind(b"[True]", UnexpectedToken), 1);
        expect_kind(br#"{'a':0}"#, UnexpectedToken); // n_object_single_quote
        expect_kind(b"[*]", UnexpectedToken); // n_array_star_inside
        expect_kind(b"\xEF\xBB\xBF{}", UnexpectedToken); // UTF-8 BOM
    }

    #[test]
    fn balanced_close_after_scalar_is_layer_2s_problem() {
        // `1]` is adjacency-legal (scalar then close); the unmatched close
        // is stage 4's job. Stage 3 must accept it.
        assert!(run(b"1]").is_ok());
        assert!(run(b"{}}").is_ok());
        // Same for separators at depth 0 (`{},` / `1,2` / `{},{}`):
        // adjacency-legal, killed by stage 4's depth-0 separator check.
        assert!(run(b"{},{}").is_ok());
        assert!(run(b"1,2").is_ok());
        // ... but back-to-back values without a separator die here.
        assert_eq!(expect_kind(b"{}{}", SyntaxErrorKind::MissingComma), 2);
    }

    #[test]
    fn root_scalars_pass_with_correct_footprints() {
        assert_eq!(run(b"42").unwrap().footprints, vec![2]);
        assert_eq!(run(b"true").unwrap().footprints, vec![1]);
        assert_eq!(run(b"\"x\"").unwrap().footprints, vec![1, 0]);
        assert!(run(b"42").unwrap().skeleton.is_empty());
    }
}
