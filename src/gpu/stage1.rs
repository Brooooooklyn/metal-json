//! Stage 1 on the GPU: bitmaps + spine scans + token extraction (K1–K5).
//!
//! # Command-buffer shape
//!
//! ```text
//! CB1: K1  classify_escape_utf8   1 thread / 64 B word: quote + candidate
//!                                 bitmaps (uint2), escape carries via raw
//!                                 look-back, UTF-8 validation, chunk quote
//!                                 partials
//!      K1b escape_carry_fixup     the valve: repairs words whose backslash
//!                                 look-back hit the 4096 B cap (near-free
//!                                 uniform early-out when none did)
//!      K2  spine_quote_scan       1 threadgroup: chunk quote carries +
//!                                 quote_total (odd ⇒ error word)
//!      K3  token_mask             1 threadgroup / 64 KiB chunk: in-string
//!                                 mask via prefix-XOR + parity carries;
//!                                 candidates → tokens in place; chunk token
//!                                 partials
//!      K4  spine_token_scan       1 threadgroup: chunk token carries +
//!                                 token_total
//!   ── commit, wait: CPU reads the header. If it carries an error
//!      (invalid UTF-8 / odd quote count) the input is REJECTED here:
//!      CB2 never runs and the token outputs stay empty (see
//!      [`Stage1Output`]). Otherwise the CPU allocates tok_pos/tok_kind
//!      at exactly token_total entries ──
//! CB2: K5  token_scatter          1 threadgroup / chunk: writes tok_pos +
//!                                 tok_kind by globally dense rank
//!   ── commit, wait ──
//! ```
//!
//! K5 lives in its own tiny command buffer because its output buffers
//! cannot exist until the CPU has read `token_total` from CB1 — the plan's
//! exact-size allocation rule (never an `input_len`-proportional guess).
//! The cost is one extra `waitUntilCompleted` round trip (~50–160 µs per
//! spike C), which M5 may fold away by keeping K5 at the head of CB2/CB3
//! work; correctness-first for M2.
//!
//! Within each command buffer a single serial encoder orders the dispatches
//! (see `src/metal/batch.rs`); across CB1 → CB2 the `commit_and_wait` is the
//! synchronization point.
//!
//! The bit-exact spec for everything here is `reference::stage1_classify` +
//! `reference::stage2_tokens`; `tests/kernels.rs` diffs the two backends on
//! identical inputs.

use crate::error::{Error, Result};
use crate::metal::{Dispatch, MetalContext, MjParams, THREADGROUP_SIZE};
use crate::stage::{Stage, Stage1Buffers};

/// `MjErrorCode` values relevant to stage 1. Mirror `shaders/common.h`.
pub const ERR_UTF8: u32 = 1;
/// Unterminated string (odd total quote count), packed by K2 with
/// offset = `input_len` so any byte-addressed UTF-8 error wins `atomic_min`.
pub const ERR_STRING: u32 = 6;

/// Everything stage 1 produces, copied back into plain `Vec`s for test
/// ergonomics (the parser integration in M3+ reads the `Stage1Buffers`
/// directly instead).
///
/// Bitmap words are `u64` read straight from the GPU `uint2 (lo, hi)`
/// buffers — identical layouts on little-endian — and diff directly against
/// the reference `Bitmaps` (`crate::reference::Bitmaps`) vectors.
///
/// # Rejection contract
///
/// When [`error`](Self::error) is `Some`, stage 1 has **rejected** the
/// input (invalid UTF-8 or an odd quote count) and downstream outputs are
/// never produced: K5 is skipped, [`tok_pos`](Self::tok_pos) and
/// [`tok_kind`](Self::tok_kind) are empty, and
/// [`token_total`](Self::token_total) is 0 — mirroring how the eventual
/// parser aborts on stage-1 errors. The bitmaps, `quote_total` and
/// `carry_overflow_count` are still returned (CB1 produced them before the
/// error was observable; the valve tests assert the repaired quote bitmap
/// on such inputs), but consumers must treat a rejected input as having no
/// token stream.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Stage1Output {
    /// Escape-resolved real-quote bitmap (one u64 per 64-byte word).
    pub quote_real: Vec<u64>,
    /// Token bitmap: `(candidates & !in_string & !quote_real) | quote_real`.
    pub tokens: Vec<u64>,
    /// Token byte positions, document order, exactly `token_total` entries.
    /// Empty when [`error`](Self::error) is `Some` (rejection contract).
    pub tok_pos: Vec<u32>,
    /// Token kinds; `reference::TokenKind` discriminants (a test pins them).
    /// Empty when [`error`](Self::error) is `Some` (rejection contract).
    pub tok_kind: Vec<u8>,
    /// Total real quotes (odd ⇒ unterminated string ⇒ `error` is set).
    pub quote_total: u64,
    /// Total tokens (`== tok_pos.len()`; 0 on rejected inputs).
    pub token_total: u64,
    /// Words whose escape look-back hit the 4096-byte cap (valve engaged).
    pub carry_overflow_count: u64,
    /// First error, packed `(byte_offset << 32) | code`, or `None`.
    pub error: Option<u64>,
    /// Input length the run described.
    pub input_len: usize,
}

impl Stage1Output {
    /// Decode [`error`](Self::error) as `(byte_offset, code)`.
    #[must_use]
    pub fn error_offset_code(&self) -> Option<(u64, u32)> {
        self.error.map(|e| (e >> 32, e as u32))
    }
}

/// The six stage-1 kernels with their lazily-built, cached pipelines.
/// Create once and reuse across parses (PSO creation is the expensive part).
#[derive(Debug)]
pub struct Stage1 {
    classify: Stage,
    fixup: Stage,
    spine_quote: Stage,
    token_mask: Stage,
    spine_token: Stage,
    scatter: Stage,
}

impl Stage1 {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            classify: Stage::new("classify_escape_utf8"),
            fixup: Stage::new("escape_carry_fixup"),
            spine_quote: Stage::new("spine_quote_scan"),
            token_mask: Stage::new("token_mask"),
            spine_token: Stage::new("spine_token_scan"),
            scatter: Stage::new("token_scatter"),
        }
    }

    /// Run stage 1 over `input` on freshly allocated buffers and read the
    /// results back. See the module docs for the CB1/CB2 shape and
    /// [`Stage1Output`] for the rejection contract: a stage-1 error (invalid
    /// UTF-8, odd quotes) means the input is rejected — K5 is skipped and
    /// the token outputs are empty.
    ///
    /// # Errors
    ///
    /// GPU plumbing failures only ([`Error::InputTooLarge`],
    /// [`Error::BufferAlloc`], pipeline/command-buffer errors). Input
    /// *content* problems (invalid UTF-8, odd quotes) are **data**, reported
    /// in [`Stage1Output::error`] — stage 3 (M3) turns them into [`Error`]s.
    pub fn run(&self, ctx: &MetalContext, input: &[u8]) -> Result<Stage1Output> {
        self.run_timed(ctx, input).map(|(out, _)| out)
    }

    /// [`run`](Self::run), additionally returning the summed GPU execution
    /// time of CB1 + CB2 in seconds (per-command-buffer `GPUEndTime −
    /// GPUStartTime`; zero when the device reports no timestamps; CB1 only
    /// when the input is rejected). Coarse whole-stage timing for the
    /// manual sanity test in `tests/kernels.rs`; per-kernel breakdowns are
    /// the M5 `timing` feature's job.
    ///
    /// # Errors
    ///
    /// As [`run`](Self::run).
    pub fn run_timed(&self, ctx: &MetalContext, input: &[u8]) -> Result<(Stage1Output, f64)> {
        let mut bufs = Stage1Buffers::new(ctx, input)?;
        self.run_with_buffers(ctx, &mut bufs)
    }

    /// [`run_timed`](Self::run_timed) over caller-prepared buffers (the
    /// path `tests/kernels.rs` uses to prove the poisoned-buffer reset).
    ///
    /// `bufs` must satisfy the [`Stage1Buffers`] zero/init preconditions —
    /// i.e. be freshly constructed or re-armed with
    /// [`Stage1Buffers::reset_for_reuse`]. Running twice on the same
    /// buffers without a reset accumulates into stale chunk counts and a
    /// stale header and produces garbage.
    ///
    /// # Errors
    ///
    /// As [`run`](Self::run).
    pub fn run_with_buffers(
        &self,
        ctx: &MetalContext,
        bufs: &mut Stage1Buffers,
    ) -> Result<(Stage1Output, f64)> {
        let input_len = bufs.input_len();
        let words = bufs.words();
        let chunks = bufs.chunks();
        if words == 0 {
            // Zero-byte input: nothing to dispatch; empty outputs, no error
            // (EmptyInput is a stage-3 grammar verdict, not a stage-1 one).
            return Ok((Stage1Output::default(), 0.0));
        }
        let mut gpu_seconds = 0.0f64;

        // The cooperative kernels (threadgroup scans, chunk-aligned simd
        // reductions) are written for full 256-thread groups; the dispatch
        // helper would silently shrink groups if a PSO capped below that.
        for stage in [
            &self.classify,
            &self.fixup,
            &self.spine_quote,
            &self.token_mask,
            &self.spine_token,
            &self.scatter,
        ] {
            let max = stage.pipeline(ctx)?.max_total_threads_per_threadgroup();
            assert!(
                max >= THREADGROUP_SIZE,
                "kernel `{}` supports only {max} threads/threadgroup (< {THREADGROUP_SIZE})",
                stage.name()
            );
        }

        let word_params = MjParams {
            input_len: input_len as u64,
            element_count: words as u64,
            ..Default::default()
        };
        let chunk_params = MjParams {
            input_len: input_len as u64,
            element_count: chunks as u64,
            ..Default::default()
        };

        // --- CB1: K1 → valve → K2 → K3 → K4, one commit, one wait ---------
        {
            let mut batch = ctx.batch()?;
            let h_input = batch.bind_read(&bufs.input);
            let h_quote = batch.bind_write(&mut bufs.bm_quote);
            let h_tok = batch.bind_write(&mut bufs.bm_tok);
            let h_escape = batch.bind_write(&mut bufs.escape_info);
            let h_qcounts = batch.bind_write(&mut bufs.chunk_quote_counts);
            let h_tcounts = batch.bind_write(&mut bufs.chunk_token_counts);
            let h_header = batch.bind_write(&mut bufs.header);

            self.classify.encode(
                &mut batch,
                &[h_input, h_quote, h_tok, h_escape, h_qcounts, h_header],
                Some(&word_params),
                Dispatch::Threadgroups(words.div_ceil(THREADGROUP_SIZE)),
            )?;
            self.fixup.encode(
                &mut batch,
                &[h_input, h_escape, h_quote, h_tok, h_qcounts, h_header],
                Some(&word_params),
                Dispatch::Threadgroups(chunks),
            )?;
            self.spine_quote.encode(
                &mut batch,
                &[h_qcounts, h_header],
                Some(&chunk_params),
                Dispatch::Threadgroups(1),
            )?;
            self.token_mask.encode(
                &mut batch,
                &[h_quote, h_tok, h_qcounts, h_tcounts],
                Some(&word_params),
                Dispatch::Threadgroups(chunks),
            )?;
            self.spine_token.encode(
                &mut batch,
                &[h_tcounts, h_header],
                Some(&chunk_params),
                Dispatch::Threadgroups(1),
            )?;
            gpu_seconds += batch.commit_and_wait_timed()?;
        }

        // --- CB1 → CPU sync point ------------------------------------------
        let header = bufs.read_header();

        // Rejection contract (see Stage1Output): CB1 detected invalid UTF-8
        // or an odd quote count, so the input is rejected and downstream
        // outputs are never produced — skip the token allocation, K5/CB2 and
        // the token readback entirely. The bitmaps and totals CB1 already
        // wrote are returned for the valve/bitmap tests; token outputs stay
        // empty (token_total = 0), mirroring how the eventual parser aborts.
        if let Some((offset, code)) = header.first_error() {
            let output = Stage1Output {
                quote_real: bufs.bm_quote.as_slice::<u64>().to_vec(),
                tokens: bufs.bm_tok.as_slice::<u64>().to_vec(),
                tok_pos: Vec::new(),
                tok_kind: Vec::new(),
                quote_total: header.quote_total,
                token_total: 0,
                carry_overflow_count: header.carry_overflow_count,
                error: Some((offset << 32) | u64::from(code)),
                input_len,
            };
            return Ok((output, gpu_seconds));
        }

        // --- exact-size token allocation ------------------------------------
        let token_total = usize::try_from(header.token_total).expect("token_total fits usize");
        if token_total > input_len {
            // A token occupies at least one input byte; anything else means
            // the GPU pipeline corrupted its own header.
            return Err(Error::CommandBuffer {
                message: format!(
                    "stage1 header reports {token_total} tokens for {input_len} input bytes"
                ),
            });
        }
        bufs.alloc_tokens(ctx, token_total)?;

        // --- CB2: K5 scatter into the exact-size buffers --------------------
        if token_total > 0 {
            let mut batch = ctx.batch()?;
            let h_input = batch.bind_read(&bufs.input);
            let h_quote = batch.bind_read(&bufs.bm_quote);
            let h_tok = batch.bind_read(&bufs.bm_tok);
            let h_qcounts = batch.bind_read(&bufs.chunk_quote_counts);
            let h_tcounts = batch.bind_read(&bufs.chunk_token_counts);
            let h_pos = batch.bind_write(bufs.tok_pos.as_mut().expect("allocated above"));
            let h_kind = batch.bind_write(bufs.tok_kind.as_mut().expect("allocated above"));
            self.scatter.encode(
                &mut batch,
                &[h_input, h_quote, h_tok, h_qcounts, h_tcounts, h_pos, h_kind],
                Some(&word_params),
                Dispatch::Threadgroups(chunks),
            )?;
            gpu_seconds += batch.commit_and_wait_timed()?;
        }

        let output = Stage1Output {
            quote_real: bufs.bm_quote.as_slice::<u64>().to_vec(),
            tokens: bufs.bm_tok.as_slice::<u64>().to_vec(),
            tok_pos: bufs
                .tok_pos
                .as_ref()
                .map(|b| b.as_slice::<u32>().to_vec())
                .unwrap_or_default(),
            tok_kind: bufs
                .tok_kind
                .as_ref()
                .map(|b| b.as_slice::<u8>().to_vec())
                .unwrap_or_default(),
            quote_total: header.quote_total,
            token_total: header.token_total,
            carry_overflow_count: header.carry_overflow_count,
            // No error: the rejection early-out above already returned.
            error: None,
            input_len,
        };
        Ok((output, gpu_seconds))
    }
}

impl Default for Stage1 {
    fn default() -> Self {
        Self::new()
    }
}

/// One-shot convenience over [`Stage1::run`] (builds the pipelines each
/// call; tests that run many inputs should hold a [`Stage1`] instead).
pub fn run_stage1(ctx: &MetalContext, input: &[u8]) -> Result<Stage1Output> {
    Stage1::new().run(ctx, input)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// GPU gating, as in tests/smoke.rs: skip without a device unless
    /// `METAL_JSON_REQUIRE_GPU=1` makes that a hard failure.
    fn ctx_or_skip(test: &str) -> Option<MetalContext> {
        match MetalContext::new() {
            Ok(ctx) => Some(ctx),
            Err(err) => {
                if std::env::var_os("METAL_JSON_REQUIRE_GPU").is_some_and(|v| v == "1") {
                    panic!("METAL_JSON_REQUIRE_GPU=1 but no usable Metal device: {err}");
                }
                eprintln!("SKIP {test}: no usable Metal device here ({err})");
                None
            }
        }
    }

    #[test]
    fn empty_input_produces_empty_outputs() {
        let Some(ctx) = ctx_or_skip("empty_input_produces_empty_outputs") else {
            return;
        };
        let out = run_stage1(&ctx, b"").unwrap();
        assert_eq!(out, Stage1Output::default());
    }

    #[test]
    fn whitespace_only_input_has_no_tokens() {
        let Some(ctx) = ctx_or_skip("whitespace_only_input_has_no_tokens") else {
            return;
        };
        let out = run_stage1(&ctx, b" \t\r\n ").unwrap();
        assert_eq!(out.quote_real, vec![0]);
        assert_eq!(out.tokens, vec![0]);
        assert!(out.tok_pos.is_empty());
        assert!(out.tok_kind.is_empty());
        assert_eq!(out.quote_total, 0);
        assert_eq!(out.token_total, 0);
        assert_eq!(out.error, None);
    }

    /// `{"a":1}` — every bitmap bit and token hand-computed.
    #[test]
    fn tiny_object_bitmaps_and_tokens_are_exact() {
        let Some(ctx) = ctx_or_skip("tiny_object_bitmaps_and_tokens_are_exact") else {
            return;
        };
        let out = run_stage1(&ctx, br#"{"a":1}"#).unwrap();
        // bytes: 0:{ 1:" 2:a 3:" 4:: 5:1 6:}
        assert_eq!(out.quote_real, vec![0b1010]);
        // tokens = ops {0,4,6} + quotes {1,3} + scalar start {5}
        assert_eq!(out.tokens, vec![0b111_1011]);
        assert_eq!(out.tok_pos, vec![0, 1, 3, 4, 5, 6]);
        // LBrace, QuoteOpen, QuoteClose, Colon, ScalarStart, RBrace
        assert_eq!(out.tok_kind, vec![0, 6, 7, 4, 8, 1]);
        assert_eq!(out.quote_total, 2);
        assert_eq!(out.token_total, 6);
        assert_eq!(out.carry_overflow_count, 0);
        assert_eq!(out.error, None);
    }

    /// Quote at byte 0 + an escaped quote inside the literal.
    #[test]
    fn escaped_quote_does_not_close_the_string() {
        let Some(ctx) = ctx_or_skip("escaped_quote_does_not_close_the_string") else {
            return;
        };
        let out = run_stage1(&ctx, br#""a\"b""#).unwrap();
        // bytes: 0:" 1:a 2:\ 3:" 4:b 5:"  — quote at 3 is escaped.
        assert_eq!(out.quote_real, vec![0b10_0001]);
        assert_eq!(out.tokens, vec![0b10_0001]);
        assert_eq!(out.tok_pos, vec![0, 5]);
        assert_eq!(out.tok_kind, vec![6, 7]); // QuoteOpen, QuoteClose
        assert_eq!(out.error, None);
    }

    /// Backslash at byte 63 escaping a quote at byte 64: the escape carry
    /// must cross the word seam via the raw look-back.
    #[test]
    fn escape_carry_crosses_the_word_seam() {
        let Some(ctx) = ctx_or_skip("escape_carry_crosses_the_word_seam") else {
            return;
        };
        let mut input = vec![b' '; 63];
        input.push(b'\\'); // byte 63
        input.push(b'"'); // byte 64, escaped via the carry
        let out = run_stage1(&ctx, &input).unwrap();
        assert_eq!(out.quote_real, vec![0, 0], "quote at 64 must be escaped");
        // The backslash at 63 is a scalar start (space before it); the
        // escaped quote at 64 continues that scalar run.
        assert_eq!(out.tok_pos, vec![63]);
        assert_eq!(out.tok_kind, vec![8]); // ScalarStart
        assert_eq!(out.quote_total, 0);
        assert_eq!(out.error, None);
    }

    /// Odd total quote count must set the error word (UnterminatedString
    /// surfaces in M3; stage 1 packs MJ_ERR_STRING at offset input_len) and
    /// REJECT the input: K5 is skipped, so no token outputs are produced.
    #[test]
    fn odd_quote_total_sets_the_error_word_and_rejects_the_input() {
        let Some(ctx) = ctx_or_skip("odd_quote_total_sets_the_error_word_and_rejects_the_input")
        else {
            return;
        };
        let out = run_stage1(&ctx, b"\"abc").unwrap();
        assert_eq!(out.quote_total, 1);
        assert_eq!(out.error_offset_code(), Some((4, ERR_STRING)));
        // The unpaired quote is still visible in the CB1 quote bitmap...
        assert_eq!(out.quote_real, vec![0b1]);
        // ...but the rejection contract keeps the token outputs empty.
        assert!(out.tok_pos.is_empty());
        assert!(out.tok_kind.is_empty());
        assert_eq!(out.token_total, 0);
    }

    /// Invalid UTF-8 reports the offset of the first byte of the first
    /// invalid sequence — exactly core::str::from_utf8's valid_up_to.
    #[test]
    fn utf8_error_offset_is_exact() {
        let Some(ctx) = ctx_or_skip("utf8_error_offset_is_exact") else {
            return;
        };
        let stage1 = Stage1::new();
        let cases: &[(&[u8], u64)] = &[
            (b"ab\x80", 2),
            (b"\xC2", 0),
            (b"{\"k\": \"\xE0\x80x\"}", 7),
            (b"\xED\xA0\x80", 0), // surrogate
            (b"\xF4\x90\x80\x80", 0),
        ];
        for &(input, offset) in cases {
            let out = stage1.run(&ctx, input).unwrap();
            let (got_offset, code) = out.error_offset_code().expect("error must be set");
            assert_eq!(code, ERR_UTF8, "{input:?}");
            assert_eq!(got_offset, offset, "{input:?}");
        }
    }

    /// The valve: a backslash wall longer than the look-back cap must flag
    /// words AND still produce the bit-exact quote bitmap after fix-up.
    #[test]
    fn backslash_wall_engages_the_valve_and_stays_correct() {
        let Some(ctx) = ctx_or_skip("backslash_wall_engages_the_valve_and_stays_correct") else {
            return;
        };
        let stage1 = Stage1::new();
        // Odd wall: the quote right after 8193 backslashes is escaped.
        // Even wall: the quote after 8192 backslashes is real.
        for (wall, escaped) in [(8193usize, true), (8192usize, false)] {
            let mut input = vec![b'\\'; wall];
            input.push(b'"');
            let out = stage1.run(&ctx, &input).unwrap();
            assert!(
                out.carry_overflow_count > 0,
                "wall of {wall} must hit the look-back cap"
            );
            let quote_bit = out.quote_real[wall / 64] >> (wall % 64) & 1;
            assert_eq!(quote_bit, u64::from(!escaped), "wall of {wall}");
            assert_eq!(out.quote_total, u64::from(!escaped), "wall of {wall}");
        }
    }

    /// Discriminant lock: the MJ_TOK_* constants written by K5 must match
    /// the reference TokenKind declaration order.
    #[cfg(feature = "cpu-reference")]
    #[test]
    fn token_kind_discriminants_match_the_reference() {
        use crate::reference::TokenKind;
        let expected: [(TokenKind, u8); 9] = [
            (TokenKind::LBrace, 0),
            (TokenKind::RBrace, 1),
            (TokenKind::LBracket, 2),
            (TokenKind::RBracket, 3),
            (TokenKind::Colon, 4),
            (TokenKind::Comma, 5),
            (TokenKind::QuoteOpen, 6),
            (TokenKind::QuoteClose, 7),
            (TokenKind::ScalarStart, 8),
        ];
        for (kind, value) in expected {
            assert_eq!(kind as u8, value, "{kind:?}");
        }
    }
}
