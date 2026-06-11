//! Stage 2 — token extraction with in-string masking.
//!
//! Scalar oracle for GPU kernels **K3** (token mask via prefix-XOR) and
//! **K5** (token scatter), including the quote-parity carry between words
//! that the **K2** spine scan provides on the GPU. The M2 kernel unit tests
//! run K3/K5 and this function on identical inputs and diff the token
//! streams.
//!
//! The algorithm is modeled word-for-word on the GPU formulation:
//!
//! 1. For each 64-byte word, the **inclusive prefix-XOR** of the
//!    [`quote_real`](super::Bitmaps::quote_real) bits (the shift ladder
//!    `m ^= m << 1; m ^= m << 2; …; m ^= m << 32`), seeded with the quote
//!    parity carried in from all previous words, yields the in-string mask:
//!    bit `i` is 1 iff byte `i` is the opening quote of a string or inside
//!    one.
//! 2. Candidate bits surviving `& !mask` become operator / scalar-start
//!    tokens (candidates inside strings vanish here).
//! 3. Quote bits become [`QuoteOpen`](TokenKind::QuoteOpen) when their
//!    inclusive-mask bit is 1 (odd parity: this quote starts a string) and
//!    [`QuoteClose`](TokenKind::QuoteClose) when it is 0. By construction
//!    quote tokens strictly alternate open/close and an open's close is the
//!    very next token (everything between them is masked) — stage 3 relies
//!    on that adjacency.
//! 4. The carry for the next word is the running parity of all quote bits.
//!
//! This stage cannot fail: an unpaired quote simply produces a trailing
//! [`QuoteOpen`](TokenKind::QuoteOpen), which stage 3 rejects as an
//! unterminated string.

use super::classify::Bitmaps;

/// What a token is. `QuoteOpen`/`QuoteClose` come in adjacent pairs; one
/// pair per string literal. `ScalarStart` marks the first byte of a number
/// or `true`/`false`/`null` literal (or garbage — stages 3/5 decide).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenKind {
    /// `{`
    LBrace,
    /// `}`
    RBrace,
    /// `[`
    LBracket,
    /// `]`
    RBracket,
    /// `:`
    Colon,
    /// `,`
    Comma,
    /// `"` starting a string literal.
    QuoteOpen,
    /// `"` ending a string literal.
    QuoteClose,
    /// First byte of a scalar run outside any string.
    ScalarStart,
}

impl TokenKind {
    /// Token kinds that can begin a JSON value.
    #[inline]
    #[must_use]
    pub fn is_value_start(self) -> bool {
        matches!(
            self,
            Self::LBrace | Self::LBracket | Self::QuoteOpen | Self::ScalarStart
        )
    }

    /// Token kinds that can end a JSON value.
    #[inline]
    #[must_use]
    pub fn is_value_end(self) -> bool {
        matches!(
            self,
            Self::RBrace | Self::RBracket | Self::QuoteClose | Self::ScalarStart
        )
    }
}

/// One extracted token: byte position + kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Token {
    /// Byte offset of the token in the input.
    pub pos: u32,
    /// What the token is.
    pub kind: TokenKind,
}

/// Inclusive prefix-XOR within a 64-bit word: output bit `i` is the parity
/// of input bits `0..=i`. The GPU computes the same thing with this exact
/// shift ladder (in `uint2` halves per the spike-B decision).
#[inline]
fn prefix_xor(mut x: u64) -> u64 {
    x ^= x << 1;
    x ^= x << 2;
    x ^= x << 4;
    x ^= x << 8;
    x ^= x << 16;
    x ^= x << 32;
    x
}

/// Stage 2: extract the document-order token stream from stage 1's bitmaps.
///
/// `input` must be the same bytes `bitmaps` were computed from (it is read
/// only at candidate positions, to classify the operator byte).
///
/// # Panics
///
/// If `input` is longer than `u32::MAX` bytes ([`parse`](super::parse)
/// rejects such inputs with [`Error::InputTooLarge`](crate::Error) first).
#[must_use]
pub fn stage2_tokens(bitmaps: &Bitmaps, input: &[u8]) -> Vec<Token> {
    let mut tokens = Vec::new();
    // Quote parity entering the current word: true = inside a string.
    // On the GPU this is the K2 spine-scan output for each chunk.
    let mut carry = false;

    for (w, (&quotes, &candidates)) in bitmaps
        .quote_real
        .iter()
        .zip(&bitmaps.candidates)
        .enumerate()
    {
        let seed = if carry { u64::MAX } else { 0 };
        let mask = prefix_xor(quotes) ^ seed;

        // Operator / scalar-start tokens: candidates outside strings.
        // (At non-quote positions the inclusive and exclusive masks agree.)
        // Quote bits always produce a token; their kind comes from parity.
        let mut bits = (candidates & !mask & !quotes) | quotes;
        while bits != 0 {
            let bit = bits.trailing_zeros();
            bits &= bits - 1;
            let pos = w * 64 + bit as usize;
            let kind = if quotes >> bit & 1 == 1 {
                if mask >> bit & 1 == 1 {
                    TokenKind::QuoteOpen
                } else {
                    TokenKind::QuoteClose
                }
            } else {
                match input[pos] {
                    b'{' => TokenKind::LBrace,
                    b'}' => TokenKind::RBrace,
                    b'[' => TokenKind::LBracket,
                    b']' => TokenKind::RBracket,
                    b':' => TokenKind::Colon,
                    b',' => TokenKind::Comma,
                    _ => TokenKind::ScalarStart,
                }
            };
            tokens.push(Token {
                pos: u32::try_from(pos).expect("input longer than u32::MAX bytes"),
                kind,
            });
        }

        carry ^= quotes.count_ones() & 1 == 1;
    }

    tokens
}

#[cfg(test)]
mod tests {
    use super::super::classify::stage1_classify;
    use super::*;

    fn tokens_of(input: &[u8]) -> Vec<Token> {
        stage2_tokens(&stage1_classify(input).unwrap(), input)
    }

    fn tok(pos: u32, kind: TokenKind) -> Token {
        Token { pos, kind }
    }

    #[test]
    fn worked_example_token_stream() {
        use TokenKind::*;
        // The docs/tape-format.md example document.
        let tokens = tokens_of(br#"{"a":[1,2.5],"b":"x\n"}"#);
        assert_eq!(
            tokens,
            vec![
                tok(0, LBrace),
                tok(1, QuoteOpen),
                tok(3, QuoteClose),
                tok(4, Colon),
                tok(5, LBracket),
                tok(6, ScalarStart), // 1
                tok(7, Comma),
                tok(8, ScalarStart), // 2.5
                tok(11, RBracket),
                tok(12, Comma),
                tok(13, QuoteOpen),
                tok(15, QuoteClose),
                tok(16, Colon),
                tok(17, QuoteOpen),
                tok(21, QuoteClose),
                tok(22, RBrace),
            ]
        );
    }

    #[test]
    fn everything_inside_a_string_is_masked() {
        use TokenKind::*;
        // Ops, scalars and whitespace inside the literal produce no tokens.
        let tokens = tokens_of(br#""{[: ,]} true 5""#);
        assert_eq!(tokens, vec![tok(0, QuoteOpen), tok(15, QuoteClose)]);
    }

    #[test]
    fn quote_parity_decides_open_vs_close() {
        use TokenKind::*;
        let tokens = tokens_of(br#""a","b""#);
        assert_eq!(
            tokens,
            vec![
                tok(0, QuoteOpen),
                tok(2, QuoteClose),
                tok(3, Comma),
                tok(4, QuoteOpen),
                tok(6, QuoteClose),
            ]
        );
    }

    #[test]
    fn escaped_quote_does_not_close_the_string() {
        use TokenKind::*;
        let tokens = tokens_of(br#""a\"b""#);
        assert_eq!(tokens, vec![tok(0, QuoteOpen), tok(5, QuoteClose)]);
    }

    #[test]
    fn unpaired_quote_yields_a_lone_quote_open() {
        let tokens = tokens_of(b"\"abc");
        assert_eq!(tokens, vec![tok(0, TokenKind::QuoteOpen)]);
    }

    #[test]
    fn string_spanning_word_seams_masks_the_later_words() {
        use TokenKind::*;
        // Open quote in word 0, ops in word 1, close quote in word 2: the
        // parity carry must keep masking across both seams.
        let mut input = b"[\"".to_vec();
        input.extend(std::iter::repeat_n(b'x', 70)); // bytes 2..=71
        input.extend_from_slice(b"{:,}"); // in-string ops, word 1
        input.extend(std::iter::repeat_n(b'y', 60)); // pushes close into word 2
        input.extend_from_slice(b"\"]");
        let close_pos = u32::try_from(input.len() - 2).unwrap();
        let tokens = tokens_of(&input);
        assert_eq!(
            tokens,
            vec![
                tok(0, LBracket),
                tok(1, QuoteOpen),
                tok(close_pos, QuoteClose),
                tok(close_pos + 1, RBracket),
            ]
        );
    }

    #[test]
    fn quote_at_word_seam_keeps_parity() {
        use TokenKind::*;
        // Close quote exactly at byte 64 (bit 0 of word 1).
        let mut input = b"\"".to_vec();
        input.extend(std::iter::repeat_n(b'a', 63)); // bytes 1..=63
        input.push(b'"'); // byte 64
        input.extend_from_slice(b":1"); // ops after the string, word 1
        let tokens = tokens_of(&input);
        assert_eq!(
            tokens,
            vec![
                tok(0, QuoteOpen),
                tok(64, QuoteClose),
                tok(65, Colon),
                tok(66, ScalarStart),
            ]
        );
    }

    #[test]
    fn root_scalars_are_single_tokens() {
        for input in [&b"-0.0"[..], b"true", b"false", b"null", b"42"] {
            let tokens = tokens_of(input);
            assert_eq!(tokens, vec![tok(0, TokenKind::ScalarStart)], "{input:?}");
        }
    }

    #[test]
    fn whitespace_only_input_has_no_tokens() {
        assert!(tokens_of(b"").is_empty());
        assert!(tokens_of(b" \t\r\n ").is_empty());
    }
}
