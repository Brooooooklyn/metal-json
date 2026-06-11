//! Stage 2 on the GPU: Layer-1 validation, tape footprints and the
//! structure lists (K6–K7 + K6b) — the M3 CB2 extension.
//!
//! # Command-buffer shape
//!
//! ```text
//! CB1: K1–K4                      (see crate::gpu::stage1)
//!   ── commit, wait: CPU sync 1 reads the header. Stage-1 errors
//!      (invalid UTF-8 / odd quote count) REJECT the input here; an empty
//!      token stream is the EmptyInput verdict (decided on the CPU — there
//!      is nothing to dispatch). Otherwise tok_pos/tok_kind are allocated
//!      at exactly token_total entries, plus the token-chunk scratch ──
//! CB2: K5  token_scatter          (M2) tok_pos + tok_kind by dense rank
//!      K6  token_validate_footprint
//!                                 1 threadgroup / 1024-token chunk: every
//!                                 Layer-1 rule of reference stage 3 + per-
//!                                 chunk partial counts (tape words, string
//!                                 slot bytes, skeleton/string/scalar
//!                                 records) + one min-reduced error word
//!                                 per chunk — no device atomics
//!      K7  spine3                 1 threadgroup: 5-component exclusive
//!                                 scan over the chunk partials (uint4 +
//!                                 u64), totals → header, error fold
//!   ── commit, wait: CPU sync 2 reads the header. A Layer-1 error REJECTS
//!      the input: CB2b never runs and the skeleton/list/tape_ofs outputs
//!      are never produced. Otherwise the skeleton / string / scalar lists
//!      are allocated at exactly the K7 totals ──
//! CB2b: K6b apply_tape_offsets    1 threadgroup / chunk: tape_ofs[token]
//!                                 (+1 root-prologue base) and the
//!                                 document-order scatter of the three
//!                                 lists via chunk carries + in-chunk ranks
//!   ── commit, wait ──
//! ```
//!
//! K6b lives in its own command buffer for the same reason K5 does in M2:
//! its output buffers cannot exist until the CPU has read the K7 totals —
//! the plan's exact-size allocation rule. Correctness-first; M5 may fold
//! the extra sync (~50–160 µs per spike C) into CB3's head.
//!
//! The bit-exact spec for the validation rules and every intermediate is
//! `reference::stage3_validate_local` (src/reference/validate.rs); the
//! in-module tests diff the two backends on identical inputs. The tape
//! position base (+1 for the root prologue word) is pinned by reference
//! `emit_tape` (src/reference/emit.rs).
//!
//! # Error-class contract (M3 split)
//!
//! By the end of this stage the GPU pipeline catches exactly the error
//! classes reference stages 1–3 catch: UTF-8 and odd-quote (CB1, carried
//! forward), every Layer-1 syntax rule, and `EmptyInput`. Depth / balance /
//! Layer-2 context / `TrailingContent` are CB3's job (the rest of M3);
//! scalar-content errors (number grammar, string escapes, control
//! characters) stay CPU-reference-only until M4.

use crate::error::{Error, Result};
use crate::metal::{Dispatch, MetalContext, MjHeader, MjParams, THREADGROUP_SIZE};
use crate::stage::{Stage, Stage1Buffers, Stage2Buffers};

use super::stage1::{Stage1, Stage1Output};

/// `MjErrorCode` values produced by stage 2 (K6 + the CPU EmptyInput
/// verdict). Mirror `shaders/common.h` — keep in sync (a test pins them).
///
/// The numeric order of the K6 codes is the **same-offset tie-break
/// contract**: K6 evaluates every Layer-1 rule in parallel and min-reduces
/// packed `(offset << 32) | code` words, so when two rules fire at the same
/// byte offset the smaller code must be the one the reference oracle (which
/// checks rules in token-iteration order) reports. See `MjErrorCode` in
/// `shaders/common.h` for the case analysis; the constants here are listed
/// in that contract's order.
pub const ERR_MISSING_COLON: u32 = 16;
/// See [`ERR_MISSING_COLON`].
pub const ERR_MISSING_COMMA: u32 = 17;
/// See [`ERR_MISSING_COLON`].
pub const ERR_UNEXPECTED_TOKEN: u32 = 18;
/// See [`ERR_MISSING_COLON`].
pub const ERR_INVALID_LITERAL: u32 = 19;
/// See [`ERR_MISSING_COLON`].
pub const ERR_UNBALANCED: u32 = 20;
/// Unterminated string from the adjacency table. Never produced on the GPU
/// today: CB1 rejects odd quote totals first (as `ERR_STRING` at offset
/// `input_len`), and an even total makes every `QuoteOpen` adjacent to its
/// close. Kept so the kernel rule table is a complete reference mirror.
pub const ERR_UNTERMINATED_STRING: u32 = 21;
/// Empty (or whitespace-only) input: `token_total == 0`. A CPU-side verdict
/// — there are no tokens to dispatch K6 over — packed at offset 0 exactly
/// like the reference's `EmptyInput` error.
pub const ERR_EMPTY_INPUT: u32 = 22;

/// Everything stage 2 produces, copied back into plain `Vec`s for test
/// ergonomics (the parser integration in late M3 reads the buffers
/// directly), mirroring [`Stage1Output`].
///
/// # Rejection contract
///
/// When [`error`](Self::error) is `Some`, the pipeline has **rejected** the
/// input and outputs after the failing stage are never produced:
///
/// - a stage-1 error (UTF-8 / odd quotes) leaves [`stage1`](Self::stage1)
///   with its own rejection contract applied (no token outputs) and every
///   stage-2 output empty;
/// - `ERR_EMPTY_INPUT` (no tokens) leaves every stage-2 output empty;
/// - a Layer-1 error (K6) keeps the stage-1 token outputs — stage 1
///   accepted the input — but K6b never runs: `tape_ofs`, the skeleton and
///   the lists stay empty, and the totals are not meaningful.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Stage2Output {
    /// The stage-1 view of the same run (bitmaps, token stream, totals).
    pub stage1: Stage1Output,
    /// Tape position of every token: `1 +` the exclusive prefix sum of the
    /// tape footprints (the `+1` is the root prologue word at `tape[0]`,
    /// per reference `emit_tape`). Defined for every token; footprint-0
    /// tokens (colon, comma, quote-close) share their successor's offset.
    pub tape_ofs: Vec<u32>,
    /// Skeleton record field 1: stage-2 token index per structural element
    /// (brackets + colons + commas), document order. Field-for-field equal
    /// to reference `SkeletonRecord::token_index` (the record is
    /// struct-of-arrays on the GPU; values are bit-identical).
    pub skeleton_token_index: Vec<u32>,
    /// Skeleton record field 2: byte offset (`SkeletonRecord::pos`).
    pub skeleton_pos: Vec<u32>,
    /// Skeleton record field 3: the structural byte, one of `{}[]:,`
    /// (`SkeletonRecord::byte`).
    pub skeleton_byte: Vec<u8>,
    /// `QuoteOpen` token indices, document order (the M4 string work list).
    pub string_tokens: Vec<u32>,
    /// `ScalarStart` token indices, document order (the M4 scalar work
    /// list).
    pub scalar_tokens: Vec<u32>,
    /// K7 total of the per-token tape footprints. The final tape is
    /// `tape_word_total + 2` words (root prologue + final root).
    pub tape_word_total: u64,
    /// K7 total string-buffer size: `Σ (raw_len + 5)` in document order.
    pub stringbuf_total: u64,
    /// First error, packed `(byte_offset << 32) | code`, or `None`. Codes:
    /// stage-1 classes carried forward (`ERR_UTF8`, `ERR_STRING`), the K6
    /// Layer-1 codes ([`ERR_MISSING_COLON`]..[`ERR_UNBALANCED`]), or
    /// [`ERR_EMPTY_INPUT`].
    pub error: Option<u64>,
}

impl Stage2Output {
    /// Decode [`error`](Self::error) as `(byte_offset, code)`.
    #[must_use]
    pub fn error_offset_code(&self) -> Option<(u64, u32)> {
        self.error.map(|e| (e >> 32, e as u32))
    }

    /// A rejected output: `stage1` as given, everything else empty.
    fn rejected(stage1: Stage1Output, packed_error: u64) -> Self {
        Self {
            stage1,
            error: Some(packed_error),
            ..Self::default()
        }
    }
}

/// The stage-2 kernels (K6, K7, K6b) plus the composed [`Stage1`], with
/// lazily-built cached pipelines. Create once and reuse across parses.
#[derive(Debug)]
pub struct Stage2 {
    stage1: Stage1,
    validate: Stage,
    spine3: Stage,
    apply: Stage,
}

impl Stage2 {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            stage1: Stage1::new(),
            validate: Stage::new("token_validate_footprint"),
            spine3: Stage::new("spine3"),
            apply: Stage::new("apply_tape_offsets"),
        }
    }

    /// Run the pipeline through Layer-1 validation (CB1 → CB2 → CB2b) over
    /// `input` on freshly allocated buffers and read the results back. See
    /// the module docs for the command-buffer shape and [`Stage2Output`]
    /// for the rejection contract.
    ///
    /// # Errors
    ///
    /// GPU plumbing failures only ([`Error::InputTooLarge`],
    /// [`Error::BufferAlloc`], pipeline/command-buffer errors). Input
    /// *content* problems are **data**, reported in
    /// [`Stage2Output::error`].
    pub fn run(&self, ctx: &MetalContext, input: &[u8]) -> Result<Stage2Output> {
        let mut bufs1 = Stage1Buffers::new(ctx, input)?;
        self.run_with_buffers(ctx, &mut bufs1)
    }

    /// [`run`](Self::run) over caller-prepared stage-1 buffers (which must
    /// satisfy the [`Stage1Buffers`] zero/init preconditions). The stage-2
    /// scratch ([`Stage2Buffers`]) is allocated internally once the token
    /// count is known.
    ///
    /// # Errors
    ///
    /// As [`run`](Self::run).
    pub fn run_with_buffers(
        &self,
        ctx: &MetalContext,
        bufs1: &mut Stage1Buffers,
    ) -> Result<Stage2Output> {
        match self.run_to_lists(ctx, bufs1)? {
            Stage2Run::Rejected(out) => Ok(*out),
            Stage2Run::Accepted(run) => {
                let Stage2Accepted {
                    stage1,
                    bufs2,
                    header,
                    gpu_seconds: _,
                } = *run;
                Ok(Self::collect_outputs(stage1, &bufs2, &header))
            }
        }
    }

    /// Run the pipeline through CB2b (the lists exist on success) WITHOUT
    /// the test-oriented `Vec` readback, so the CB3 orchestration
    /// (`crate::gpu::stage3`) can keep using the GPU buffers directly.
    ///
    /// # Errors
    ///
    /// As [`run`](Self::run).
    pub(crate) fn run_to_lists(
        &self,
        ctx: &MetalContext,
        bufs1: &mut Stage1Buffers,
    ) -> Result<Stage2Run> {
        let input_len = bufs1.input_len();

        if bufs1.words() == 0 {
            // Zero-byte input: no tokens, nothing to dispatch — the
            // reference's EmptyInput verdict at offset 0.
            return Ok(Stage2Run::rejected(Stage2Output::rejected(
                Stage1Output::default(),
                u64::from(ERR_EMPTY_INPUT),
            )));
        }

        // The stage-2 cooperative kernels are written for full 256-thread
        // groups too (same invariant Stage1::run_cb1 asserts for K1-K5).
        for stage in [&self.validate, &self.spine3, &self.apply] {
            let max = stage.pipeline(ctx)?.max_total_threads_per_threadgroup();
            assert!(
                max >= THREADGROUP_SIZE,
                "kernel `{}` supports only {max} threads/threadgroup (< {THREADGROUP_SIZE})",
                stage.name()
            );
        }

        // --- CB1, then CPU sync 1 ------------------------------------------
        let mut gpu_seconds = self.stage1.run_cb1(ctx, bufs1)?;
        let header = bufs1.read_header();

        if let Some((offset, code)) = header.first_error() {
            // Stage-1 rejection (UTF-8 / odd quotes) carried forward: CB2
            // never runs (M2 contract).
            let packed = (offset << 32) | u64::from(code);
            let stage1 = Stage1Output::snapshot(bufs1, &header, Some(packed));
            return Ok(Stage2Run::rejected(Stage2Output::rejected(stage1, packed)));
        }

        let token_total = usize::try_from(header.token_total).expect("token_total fits usize");
        if token_total > input_len {
            // A token occupies at least one input byte (header corruption).
            return Err(Error::CommandBuffer {
                message: format!(
                    "stage1 header reports {token_total} tokens for {input_len} input bytes"
                ),
            });
        }
        if token_total == 0 {
            // Whitespace-only input: the reference's EmptyInput verdict at
            // offset 0. CPU-side — there are no tokens to dispatch K6 over.
            let stage1 = Stage1Output::snapshot(bufs1, &header, None);
            return Ok(Stage2Run::rejected(Stage2Output::rejected(
                stage1,
                u64::from(ERR_EMPTY_INPUT),
            )));
        }
        bufs1.alloc_tokens(ctx, token_total)?;
        let mut bufs2 = Stage2Buffers::new(ctx, token_total)?;
        let tok_chunks = bufs2.chunks();
        let word_chunks = bufs1.chunks();

        let word_params = MjParams {
            input_len: input_len as u64,
            element_count: bufs1.words() as u64,
            ..Default::default()
        };
        let token_params = MjParams {
            input_len: input_len as u64,
            element_count: token_total as u64,
            ..Default::default()
        };
        let tok_chunk_params = MjParams {
            input_len: input_len as u64,
            element_count: tok_chunks as u64,
            ..Default::default()
        };

        // --- CB2: K5 → K6 → K7, one commit, one wait ------------------------
        {
            let mut batch = ctx.batch()?;
            let h_input = batch.bind_read(&bufs1.input);
            let h_quote = batch.bind_read(&bufs1.bm_quote);
            let h_tok = batch.bind_read(&bufs1.bm_tok);
            let h_qcounts = batch.bind_read(&bufs1.chunk_quote_counts);
            let h_tcounts = batch.bind_read(&bufs1.chunk_token_counts);
            let h_pos = batch.bind_write(bufs1.tok_pos.as_mut().expect("allocated above"));
            let h_kind = batch.bind_write(bufs1.tok_kind.as_mut().expect("allocated above"));
            let h_header = batch.bind_write(&mut bufs1.header);
            let h_counts = batch.bind_write(&mut bufs2.chunk_counts);
            let h_sbytes = batch.bind_write(&mut bufs2.chunk_string_bytes);
            let h_error = batch.bind_write(&mut bufs2.chunk_error);

            self.stage1.scatter_stage().encode(
                &mut batch,
                &[h_input, h_quote, h_tok, h_qcounts, h_tcounts, h_pos, h_kind],
                Some(&word_params),
                Dispatch::Threadgroups(word_chunks),
            )?;
            self.validate.encode(
                &mut batch,
                &[h_input, h_pos, h_kind, h_counts, h_sbytes, h_error],
                Some(&token_params),
                Dispatch::Threadgroups(tok_chunks),
            )?;
            self.spine3.encode(
                &mut batch,
                &[h_counts, h_sbytes, h_error, h_header],
                Some(&tok_chunk_params),
                Dispatch::Threadgroups(1),
            )?;
            gpu_seconds += batch.commit_and_wait_timed()?;
        }

        // --- CB2 → CPU sync 2 ------------------------------------------------
        let header = bufs1.read_header();
        let stage1 = Stage1Output::snapshot(bufs1, &header, None);

        if let Some((offset, code)) = header.first_error() {
            // Layer-1 rejection: K6b (and CB3) never run; the list outputs
            // are never produced. Stage 1 accepted the input, so its token
            // outputs are kept (see Stage2Output's rejection contract).
            let packed = (offset << 32) | u64::from(code);
            return Ok(Stage2Run::rejected(Stage2Output::rejected(stage1, packed)));
        }

        let totals = ListTotals::checked(&header, token_total)?;

        // --- exact-size list allocation + CB2b: K6b --------------------------
        bufs2.alloc_lists(ctx, totals.skeleton, totals.strings, totals.scalars)?;
        {
            let mut batch = ctx.batch()?;
            let h_input = batch.bind_read(&bufs1.input);
            let h_pos = batch.bind_read(bufs1.tok_pos.as_ref().expect("allocated above"));
            let h_kind = batch.bind_read(bufs1.tok_kind.as_ref().expect("allocated above"));
            let h_counts = batch.bind_read(&bufs2.chunk_counts);
            let h_tape_ofs = batch.bind_write(&mut bufs2.tape_ofs);
            let h_skel_ti = batch.bind_write(bufs2.skel_token_index.as_mut().expect("just allocated"));
            let h_skel_pos = batch.bind_write(bufs2.skel_pos.as_mut().expect("just allocated"));
            let h_skel_byte = batch.bind_write(bufs2.skel_byte.as_mut().expect("just allocated"));
            let h_strings = batch.bind_write(bufs2.string_tokens.as_mut().expect("just allocated"));
            let h_scalars = batch.bind_write(bufs2.scalar_tokens.as_mut().expect("just allocated"));
            self.apply.encode(
                &mut batch,
                &[
                    h_input, h_pos, h_kind, h_counts, h_tape_ofs, h_skel_ti, h_skel_pos,
                    h_skel_byte, h_strings, h_scalars,
                ],
                Some(&token_params),
                Dispatch::Threadgroups(tok_chunks),
            )?;
            gpu_seconds += batch.commit_and_wait_timed()?;
        }

        Ok(Stage2Run::Accepted(Box::new(Stage2Accepted {
            stage1,
            bufs2,
            header,
            gpu_seconds,
        })))
    }

    /// The test-oriented `Vec` readback of an accepted run's outputs (the
    /// CB3 orchestration calls this too, so its riding-along stage-2 view
    /// is bit-identical to [`run`](Self::run)'s).
    pub(crate) fn collect_outputs(
        stage1: Stage1Output,
        bufs2: &Stage2Buffers,
        header: &MjHeader,
    ) -> Stage2Output {
        Stage2Output {
            stage1,
            tape_ofs: bufs2.tape_ofs.as_slice::<u32>().to_vec(),
            skeleton_token_index: read_u32s(bufs2.skel_token_index.as_ref()),
            skeleton_pos: read_u32s(bufs2.skel_pos.as_ref()),
            skeleton_byte: bufs2
                .skel_byte
                .as_ref()
                .map(|b| b.as_slice::<u8>().to_vec())
                .unwrap_or_default(),
            string_tokens: read_u32s(bufs2.string_tokens.as_ref()),
            scalar_tokens: read_u32s(bufs2.scalar_tokens.as_ref()),
            tape_word_total: header.tape_word_total,
            stringbuf_total: header.stringbuf_total,
            error: None,
        }
    }
}

/// Where a stage-2 run ended. Internal: lets the CB3 orchestration
/// (`crate::gpu::stage3`) continue from `Accepted` with live GPU buffers
/// instead of forcing [`Stage2Output`]'s readback.
pub(crate) enum Stage2Run {
    /// The pipeline rejected the input (stage-1 error, `EmptyInput`, or a
    /// Layer-1 error): the finished output with its rejection contract
    /// applied. Boxed to keep the enum small.
    Rejected(Box<Stage2Output>),
    /// Layer 1 passed: K6b ran and the skeleton / list buffers exist.
    Accepted(Box<Stage2Accepted>),
}

impl Stage2Run {
    fn rejected(out: Stage2Output) -> Self {
        Self::Rejected(Box::new(out))
    }
}

/// The accepted half of [`Stage2Run`]: everything CB3 needs.
pub(crate) struct Stage2Accepted {
    /// The stage-1 view (bitmaps, token stream, totals), already
    /// snapshotted at the CPU sync 2.
    pub(crate) stage1: Stage1Output,
    /// The CB2 buffers, lists allocated and written.
    pub(crate) bufs2: Stage2Buffers,
    /// The header as read at the CPU sync 2 (totals valid, no error).
    pub(crate) header: MjHeader,
    /// Summed GPU execution time of CB1 + CB2 + CB2b in seconds
    /// (per-command-buffer `GPUEndTime − GPUStartTime`; zero when the
    /// device reports no timestamps). Coarse whole-stage timing for
    /// `Stage3::run_timed`'s manual sanity test.
    pub(crate) gpu_seconds: f64,
}

impl Default for Stage2 {
    fn default() -> Self {
        Self::new()
    }
}

/// The K7 list totals, bounds-checked against `token_total` (every record
/// is keyed by a distinct token, so a violation means the GPU pipeline
/// corrupted its own header — surfaced as a plumbing error, like the
/// stage-1 token_total check).
struct ListTotals {
    skeleton: usize,
    strings: usize,
    scalars: usize,
}

impl ListTotals {
    fn checked(header: &MjHeader, token_total: usize) -> Result<Self> {
        let corrupt = |what: &str, got: u64| Error::CommandBuffer {
            message: format!("stage2 header reports {got} {what} for {token_total} tokens"),
        };
        let skeleton = usize::try_from(header.skeleton_total).expect("fits usize");
        let strings = usize::try_from(header.string_total).expect("fits usize");
        let scalars = usize::try_from(header.scalar_total).expect("fits usize");
        if skeleton > token_total {
            return Err(corrupt("skeleton records", header.skeleton_total));
        }
        // Every string is a QuoteOpen/QuoteClose token pair.
        if strings > token_total / 2 {
            return Err(corrupt("string records", header.string_total));
        }
        if scalars > token_total {
            return Err(corrupt("scalar records", header.scalar_total));
        }
        // Footprints are at most 2 tape words per token.
        if header.tape_word_total > 2 * token_total as u64 {
            return Err(corrupt("tape words", header.tape_word_total));
        }
        Ok(Self {
            skeleton,
            strings,
            scalars,
        })
    }
}

fn read_u32s(buffer: Option<&crate::metal::GpuBuffer>) -> Vec<u32> {
    buffer
        .map(|b| b.as_slice::<u32>().to_vec())
        .unwrap_or_default()
}

/// One-shot convenience over [`Stage2::run`] (builds the pipelines each
/// call; tests that run many inputs should hold a [`Stage2`] instead).
pub fn run_stage2(ctx: &MetalContext, input: &[u8]) -> Result<Stage2Output> {
    Stage2::new().run(ctx, input)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// GPU gating, as in stage1.rs: skip without a device unless
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

    /// Code-order lock: the same-offset tie-break contract (see the
    /// constants' docs and `shaders/common.h`) requires exactly this
    /// ordering; reordering any constant silently changes which error the
    /// GPU reports on multi-rule offsets like `{"a" "b"}`. Compile-time
    /// (`const` blocks): violating the contract fails the build.
    #[test]
    fn error_code_order_is_the_tie_break_contract() {
        const {
            assert!(ERR_MISSING_COLON < ERR_MISSING_COMMA);
            assert!(ERR_MISSING_COMMA < ERR_UNEXPECTED_TOKEN);
            assert!(ERR_UNEXPECTED_TOKEN < ERR_INVALID_LITERAL);
            assert!(ERR_INVALID_LITERAL < ERR_UNBALANCED);
            assert!(ERR_UNBALANCED < ERR_UNTERMINATED_STRING);
            assert!(ERR_UNTERMINATED_STRING < ERR_EMPTY_INPUT);
            // Stage-1 codes are disjoint from the Layer-1 block (the
            // rejection contract keeps them from ever competing in one
            // reduction, but the values must not collide for decoding).
            assert!(super::super::ERR_UTF8 < ERR_MISSING_COLON);
            assert!(super::super::ERR_STRING < ERR_MISSING_COLON);
        }
    }

    /// The docs/tape-format.md worked example, every output hand-computed:
    /// `{"a":[1,2.5],"b":"x\n"}`.
    ///
    /// Footprints per token (reference stage 3 test):
    ///   [1,1,0,0,1,2,0,2,1,0,1,0,0,1,0,1]
    /// so tape_ofs = 1 + exclusive prefix sum, and the totals match the
    /// tape-format doc's 13-word tape (11 + 2 root words) and 20-byte
    /// string buffer.
    #[test]
    fn worked_example_outputs_are_exact() {
        let Some(ctx) = ctx_or_skip("worked_example_outputs_are_exact") else {
            return;
        };
        let out = run_stage2(&ctx, br#"{"a":[1,2.5],"b":"x\n"}"#).unwrap();
        assert_eq!(out.error, None);
        assert_eq!(
            out.tape_ofs,
            vec![1, 2, 3, 3, 3, 4, 6, 6, 8, 9, 9, 10, 10, 10, 11, 11]
        );
        assert_eq!(out.skeleton_byte, b"{:[,],:}".to_vec());
        assert_eq!(out.skeleton_token_index, vec![0, 3, 4, 6, 8, 9, 12, 15]);
        assert_eq!(out.skeleton_pos, vec![0, 4, 5, 7, 11, 12, 16, 22]);
        assert_eq!(out.string_tokens, vec![1, 10, 13]);
        assert_eq!(out.scalar_tokens, vec![5, 7]);
        assert_eq!(out.tape_word_total, 11);
        // Slots of raw_len + 5: "a" → 6, "b" → 6, "x\n" (raw 3) → 8.
        assert_eq!(out.stringbuf_total, 20);
        // The stage-1 view rode along intact.
        assert_eq!(out.stage1.token_total, 16);
        assert_eq!(out.stage1.error, None);
    }

    #[test]
    fn root_scalars_and_strings() {
        let Some(ctx) = ctx_or_skip("root_scalars_and_strings") else {
            return;
        };
        let stage2 = Stage2::new();

        let out = stage2.run(&ctx, b"42").unwrap();
        assert_eq!(out.error, None);
        assert_eq!(out.tape_ofs, vec![1]);
        assert_eq!(out.tape_word_total, 2); // number: marker + value word
        assert_eq!(out.stringbuf_total, 0);
        assert!(out.skeleton_byte.is_empty());
        assert!(out.string_tokens.is_empty());
        assert_eq!(out.scalar_tokens, vec![0]);

        let out = stage2.run(&ctx, b"true").unwrap();
        assert_eq!(out.error, None);
        assert_eq!(out.tape_word_total, 1);
        assert_eq!(out.scalar_tokens, vec![0]);

        let out = stage2.run(&ctx, b"\"x\"").unwrap();
        assert_eq!(out.error, None);
        assert_eq!(out.tape_ofs, vec![1, 2]); // open emits 1 word, close 0
        assert_eq!(out.tape_word_total, 1);
        assert_eq!(out.stringbuf_total, 6); // raw_len 1 + 5
        assert_eq!(out.string_tokens, vec![0]);
        assert!(out.scalar_tokens.is_empty());
    }

    #[test]
    fn empty_and_whitespace_only_inputs_are_empty_input() {
        let Some(ctx) = ctx_or_skip("empty_and_whitespace_only_inputs_are_empty_input") else {
            return;
        };
        let stage2 = Stage2::new();
        for input in [&b""[..], b" \t\n\r", b" "] {
            let out = stage2.run(&ctx, input).unwrap();
            assert_eq!(
                out.error_offset_code(),
                Some((0, ERR_EMPTY_INPUT)),
                "{input:?}"
            );
            assert!(out.tape_ofs.is_empty(), "{input:?}");
            assert!(out.skeleton_byte.is_empty(), "{input:?}");
        }
    }

    /// Layer-1 rejections: the (offset, kind) pairs the reference oracle's
    /// own test suite pins (src/reference/validate.rs), including the
    /// same-offset tie-break cases the code-order contract exists for.
    #[test]
    fn layer1_rejections_report_reference_offsets_and_codes() {
        let Some(ctx) = ctx_or_skip("layer1_rejections_report_reference_offsets_and_codes")
        else {
            return;
        };
        let stage2 = Stage2::new();
        let cases: &[(&[u8], u64, u32)] = &[
            // token-order table
            (b"]", 0, ERR_UNEXPECTED_TOKEN),
            (b"}", 0, ERR_UNEXPECTED_TOKEN),
            (b",1", 0, ERR_UNEXPECTED_TOKEN),
            (b"{a: 1}", 1, ERR_UNEXPECTED_TOKEN), // unquoted key
            (b"[,]", 1, ERR_UNEXPECTED_TOKEN),
            (b"[1,,2]", 3, ERR_UNEXPECTED_TOKEN),
            (br#"["",]"#, 4, ERR_UNEXPECTED_TOKEN), // reported at the `]`
            (br#"{"a":}"#, 5, ERR_UNEXPECTED_TOKEN),
            (b"[1:2]", 2, ERR_UNEXPECTED_TOKEN),
            (b"[1 true]", 3, ERR_MISSING_COMMA),
            (b"[3[4]]", 2, ERR_MISSING_COMMA),
            (b"[][]", 2, ERR_MISSING_COMMA),
            (b"[1]x", 3, ERR_MISSING_COMMA),
            (b"null null", 5, ERR_MISSING_COMMA),
            (b"{}{}", 2, ERR_MISSING_COMMA),
            (b"[1,", 3, ERR_UNEXPECTED_TOKEN), // separator then end of input
            // unclosed opens at end of input (reported at the bracket)
            (b"{", 0, ERR_UNBALANCED),
            (b"[", 0, ERR_UNBALANCED),
            (b"[[", 1, ERR_UNBALANCED),
            // colon 4-token rule
            (br#"["": 1]"#, 3, ERR_UNEXPECTED_TOKEN),
            (br#""a":1"#, 3, ERR_UNEXPECTED_TOKEN),
            // object-first-member rule
            (br#"{"x", null}"#, 4, ERR_MISSING_COLON),
            (br#"{"a"}"#, 4, ERR_MISSING_COLON),
            (br#"{"a""#, 4, ERR_MISSING_COLON), // at input_len
            // literal byte checks
            (br#"["x", truth]"#, 6, ERR_INVALID_LITERAL),
            (b"[tru]", 1, ERR_INVALID_LITERAL),
            (b"false0", 0, ERR_INVALID_LITERAL),
            (b"truee", 0, ERR_INVALID_LITERAL),
            (b"[True]", 1, ERR_UNEXPECTED_TOKEN),
            (b"[*]", 1, ERR_UNEXPECTED_TOKEN),
            (b"\xEF\xBB\xBF{}", 0, ERR_UNEXPECTED_TOKEN), // UTF-8 BOM
            // SAME-OFFSET TIE-BREAKS (the code-order contract):
            // {"a" "b"}: first-member MissingColon@5 vs adjacency
            // MissingComma@5 — the reference fires the first-member rule
            // three iterations earlier.
            (br#"{"a" "b"}"#, 5, ERR_MISSING_COLON),
            // 1 *: adjacency MissingComma@2 vs scalar-first-byte
            // UnexpectedToken@2 — adjacency precedes kind rules.
            (b"1 *", 2, ERR_MISSING_COMMA),
            // 1 truth: adjacency MissingComma@2 vs InvalidLiteral@2.
            (b"1 truth", 2, ERR_MISSING_COMMA),
            // {"a"{: first-member MissingColon@4 vs adjacency
            // MissingComma@4 vs end-check Unbalanced@4.
            (br#"{"a"{"#, 4, ERR_MISSING_COLON),
            // 1{: adjacency MissingComma@1 vs end-check Unbalanced@1.
            (b"1{", 1, ERR_MISSING_COMMA),
            // {[: adjacency UnexpectedToken@1 vs end-check Unbalanced@1.
            (b"{[", 1, ERR_UNEXPECTED_TOKEN),
        ];
        for &(input, offset, code) in cases {
            let out = stage2.run(&ctx, input).unwrap();
            assert_eq!(
                out.error_offset_code(),
                Some((offset, code)),
                "{:?}",
                String::from_utf8_lossy(input)
            );
            // Rejection contract: K6b never ran.
            assert!(out.tape_ofs.is_empty(), "{input:?}: no tape_ofs");
            assert!(out.skeleton_byte.is_empty(), "{input:?}: no skeleton");
            assert!(out.string_tokens.is_empty(), "{input:?}: no string list");
            // ... but stage 1 accepted the input: tokens are kept.
            assert!(
                !out.stage1.tok_pos.is_empty(),
                "{input:?}: stage-1 tokens kept on Layer-1 rejection"
            );
        }
    }

    /// What Layer 1 deliberately cannot see (no container context) must
    /// PASS this stage — depth/balance/Layer-2 errors are CB3's job, and
    /// rejecting them here would report the wrong error class.
    #[test]
    fn layer2_problems_pass_layer1() {
        let Some(ctx) = ctx_or_skip("layer2_problems_pass_layer1") else {
            return;
        };
        let stage2 = Stage2::new();
        for input in [
            &b"1]"[..],   // close with nothing open
            b"{}}",       // stray close
            b"{},{}",     // depth-0 separator (trailing content)
            b"1,2",       // depth-0 separator
            b"[1}",       // mismatched bracket types
            b"[1",        // unclosed container (last token is a value end)
            br#"[1,"a":2]"#,    // colon inside an array
            br#"{"a":1,2}"#,    // object comma without a member colon
            br#"{"foo":1, "a"}"#, // later member without colon (stage 4)
        ] {
            let out = stage2.run(&ctx, input).unwrap();
            assert_eq!(
                out.error,
                None,
                "{:?} must pass Layer 1",
                String::from_utf8_lossy(input)
            );
            assert!(
                !out.skeleton_byte.is_empty(),
                "{:?}: skeleton produced for CB3",
                String::from_utf8_lossy(input)
            );
        }

        // Spot-check one: `1]` has a single skeleton record, the stray `]`.
        let out = stage2.run(&ctx, b"1]").unwrap();
        assert_eq!(out.skeleton_byte, b"]".to_vec());
        assert_eq!(out.skeleton_token_index, vec![1]);
        assert_eq!(out.skeleton_pos, vec![1]);
    }

    /// Stage-1 rejections carry forward unchanged (the M2 contract): CB2
    /// never runs, stage-2 outputs stay empty.
    #[test]
    fn stage1_rejections_carry_forward() {
        let Some(ctx) = ctx_or_skip("stage1_rejections_carry_forward") else {
            return;
        };
        let stage2 = Stage2::new();

        // Invalid UTF-8 at byte 2.
        let out = stage2.run(&ctx, b"ab\x80").unwrap();
        assert_eq!(out.error_offset_code(), Some((2, super::super::ERR_UTF8)));
        assert!(out.stage1.tok_pos.is_empty(), "stage-1 rejection contract");
        assert!(out.tape_ofs.is_empty());

        // Odd quote count: ERR_STRING at offset input_len (the documented
        // provisional offset; the reference reports the open quote — class
        // parity only, see stage1's K2 docs).
        let out = stage2.run(&ctx, b"\"abc").unwrap();
        assert_eq!(out.error_offset_code(), Some((4, super::super::ERR_STRING)));
        assert!(out.tape_ofs.is_empty());
    }

    /// Chunk-carry coverage: a document spanning several 1024-token chunks
    /// so the K7 carries and K6b in-chunk ranks both matter. Totals and a
    /// few spot offsets are hand-computed.
    #[test]
    fn multi_chunk_token_streams_scan_correctly() {
        let Some(ctx) = ctx_or_skip("multi_chunk_token_streams_scan_correctly") else {
            return;
        };
        // [0,0,...,0] with 3000 elements: tokens = 1 + 3000 scalars +
        // 2999 commas + 1 = 6001 → 6 token chunks.
        let n = 3000usize;
        let mut input = b"[".to_vec();
        for i in 0..n {
            if i > 0 {
                input.push(b',');
            }
            input.push(b'0');
        }
        input.push(b']');

        let out = run_stage2(&ctx, &input).unwrap();
        assert_eq!(out.error, None);
        assert_eq!(out.stage1.token_total, 1 + 2 * n as u64);
        // Footprints: [ = 1, each number = 2, commas = 0, ] = 1.
        assert_eq!(out.tape_word_total, 2 + 2 * n as u64);
        assert_eq!(out.stringbuf_total, 0);
        // Skeleton: [ + 2999 commas + ] in document order.
        assert_eq!(out.skeleton_byte.len(), n + 1);
        assert_eq!(out.skeleton_byte[0], b'[');
        assert_eq!(*out.skeleton_byte.last().unwrap(), b']');
        assert!(out.skeleton_byte[1..n].iter().all(|&b| b == b','));
        assert_eq!(out.scalar_tokens.len(), n);
        // tape_ofs: token 0 (`[`) at 1; scalar k (token 2k+1) at 2 + 2k;
        // comma k (token 2k+2) at 4 + 2k; `]` last at 2 + 2n.
        assert_eq!(out.tape_ofs[0], 1);
        for k in 0..n {
            assert_eq!(out.tape_ofs[2 * k + 1], 2 + 2 * k as u32, "scalar {k}");
        }
        assert_eq!(out.tape_ofs[2 * n], 2 + 2 * n as u32);
        // Scalar list is the odd token indices in order.
        assert!(
            out.scalar_tokens
                .iter()
                .enumerate()
                .all(|(k, &t)| t == 2 * k as u32 + 1)
        );
    }

    // --- vs the cpu-reference oracle --------------------------------------

    #[cfg(feature = "cpu-reference")]
    mod vs_reference {
        use super::*;
        use crate::reference::{
            SkeletonRecord, TokenKind, stage1_classify, stage2_tokens, stage3_validate_local,
        };
        use crate::{Error as CrateError, SyntaxErrorKind};

        /// The reference SyntaxErrorKind for each GPU code this stage can
        /// produce (a second lock on the constant mapping, from the Rust
        /// side).
        fn code_for_kind(kind: SyntaxErrorKind) -> u32 {
            match kind {
                SyntaxErrorKind::MissingColon => ERR_MISSING_COLON,
                SyntaxErrorKind::MissingComma => ERR_MISSING_COMMA,
                SyntaxErrorKind::UnexpectedToken => ERR_UNEXPECTED_TOKEN,
                SyntaxErrorKind::InvalidLiteral => ERR_INVALID_LITERAL,
                SyntaxErrorKind::UnbalancedBrackets => ERR_UNBALANCED,
                SyntaxErrorKind::UnterminatedString => ERR_UNTERMINATED_STRING,
                SyntaxErrorKind::EmptyInput => ERR_EMPTY_INPUT,
                other => panic!("reference stage 3 cannot produce {other:?}"),
            }
        }

        /// Run both backends on `input` and require agreement: clean inputs
        /// compare every stage-2 output bit-for-bit against reference
        /// stage 3 (+ the emit-defined tape positions); rejected inputs
        /// compare the verdict (offset + code), with the documented
        /// odd-quote offset exception.
        fn diff(stage2: &Stage2, ctx: &MetalContext, input: &[u8], label: &str) {
            let got = stage2
                .run(ctx, input)
                .unwrap_or_else(|e| panic!("{label}: GPU stage 2 failed: {e}"));

            let bitmaps = match stage1_classify(input) {
                Ok(bitmaps) => bitmaps,
                Err(CrateError::Utf8 { offset }) => {
                    assert_eq!(
                        got.error_offset_code(),
                        Some((offset, super::super::super::ERR_UTF8)),
                        "{label}: UTF-8 verdict"
                    );
                    return;
                }
                Err(other) => panic!("{label}: unexpected reference error {other:?}"),
            };
            let quote_total: u64 = bitmaps
                .quote_real
                .iter()
                .map(|w| u64::from(w.count_ones()))
                .sum();
            if quote_total % 2 == 1 {
                // Odd quotes are rejected in CB1 at offset input_len (the
                // documented provisional offset); the reference reports the
                // open quote's offset from stage 3 — class parity only.
                assert_eq!(
                    got.error_offset_code(),
                    Some((input.len() as u64, super::super::super::ERR_STRING)),
                    "{label}: odd-quote verdict"
                );
                if !input.is_empty() {
                    let tokens = stage2_tokens(&bitmaps, input);
                    assert!(
                        matches!(
                            stage3_validate_local(&tokens, input),
                            Err(CrateError::Syntax {
                                kind: SyntaxErrorKind::UnterminatedString,
                                ..
                            })
                        ),
                        "{label}: reference must agree the string is unterminated"
                    );
                }
                return;
            }

            let tokens = stage2_tokens(&bitmaps, input);
            match stage3_validate_local(&tokens, input) {
                Err(CrateError::Syntax { offset, kind }) => {
                    assert_eq!(
                        got.error_offset_code(),
                        Some((offset, code_for_kind(kind))),
                        "{label}: Layer-1 verdict for reference {kind:?}"
                    );
                    assert!(got.tape_ofs.is_empty(), "{label}: rejection contract");
                    assert!(got.skeleton_byte.is_empty(), "{label}: rejection contract");
                }
                Err(other) => panic!("{label}: unexpected reference error {other:?}"),
                Ok(s3) => {
                    assert_eq!(got.error, None, "{label}: spurious GPU error");

                    // tape_ofs == the reference emit's tape positions:
                    // 1 + exclusive prefix sum of the footprints.
                    let mut want_tape_ofs = Vec::with_capacity(tokens.len());
                    let mut running = 1u32;
                    for fp in &s3.footprints {
                        want_tape_ofs.push(running);
                        running += fp;
                    }
                    assert_eq!(got.tape_ofs, want_tape_ofs, "{label}: tape_ofs");
                    assert_eq!(
                        got.tape_word_total,
                        u64::from(running - 1),
                        "{label}: tape word total"
                    );

                    // Skeleton: field-for-field against SkeletonRecord.
                    let want: Vec<SkeletonRecord> = s3.skeleton;
                    let got_records: Vec<SkeletonRecord> = got
                        .skeleton_token_index
                        .iter()
                        .zip(&got.skeleton_pos)
                        .zip(&got.skeleton_byte)
                        .map(|((&token_index, &pos), &byte)| SkeletonRecord {
                            token_index,
                            pos,
                            byte,
                        })
                        .collect();
                    assert_eq!(got_records, want, "{label}: skeleton records");

                    // Work lists: the QuoteOpen / ScalarStart token indices
                    // in document order, exactly the reference's record
                    // keying (UnescapedString/ParsedScalar::token_index).
                    let want_strings: Vec<u32> = tokens
                        .iter()
                        .enumerate()
                        .filter(|(_, t)| t.kind == TokenKind::QuoteOpen)
                        .map(|(i, _)| u32::try_from(i).unwrap())
                        .collect();
                    let want_scalars: Vec<u32> = tokens
                        .iter()
                        .enumerate()
                        .filter(|(_, t)| t.kind == TokenKind::ScalarStart)
                        .map(|(i, _)| u32::try_from(i).unwrap())
                        .collect();
                    assert_eq!(got.string_tokens, want_strings, "{label}: string list");
                    assert_eq!(got.scalar_tokens, want_scalars, "{label}: scalar list");

                    // stringbuf_total == Σ (raw_len + 5) over the strings
                    // (the K7 scan total; reference stage 6 derives its
                    // record offsets from the same sums).
                    let want_stringbuf: u64 = want_strings
                        .iter()
                        .map(|&i| {
                            let open = tokens[i as usize].pos;
                            let close = tokens[i as usize + 1].pos;
                            u64::from(close - open - 1) + 5
                        })
                        .sum();
                    assert_eq!(
                        got.stringbuf_total, want_stringbuf,
                        "{label}: stringbuf total"
                    );
                }
            }
        }

        #[test]
        fn corpus_files_match_reference_stage3() {
            let Some(ctx) = ctx_or_skip("corpus_files_match_reference_stage3") else {
                return;
            };
            let stage2 = Stage2::new();
            let corpus = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
            let mut paths: Vec<_> = std::fs::read_dir(&corpus)
                .expect("corpus/ is checked in")
                .map(|e| e.expect("readable corpus entry").path())
                .filter(|p| p.extension().is_some_and(|e| e == "json"))
                .collect();
            paths.sort();
            assert!(!paths.is_empty(), "corpus/ must contain fixtures");
            for path in paths {
                let name = path.file_name().unwrap().to_string_lossy().into_owned();
                let bytes = std::fs::read(&path).expect("readable corpus fixture");
                diff(&stage2, &ctx, &bytes, &name);
            }
        }

        /// Every JSONTestSuite file, GPU stage 2 vs reference stages 1–3:
        /// the error-class split pinned across the whole suite. Files the
        /// reference rejects in stages 1–3 must produce the same packed
        /// verdict here; files it only rejects later (depth/balance/
        /// Layer-2 → CB3, scalar content → M4) must pass stage 2 with
        /// bit-identical skeleton/list/offset outputs.
        #[test]
        fn jsontestsuite_files_match_reference_stage3() {
            let Some(ctx) = ctx_or_skip("jsontestsuite_files_match_reference_stage3") else {
                return;
            };
            let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("data/JSONTestSuite/test_parsing");
            if !dir.is_dir() {
                eprintln!(
                    "SKIP jsontestsuite_files_match_reference_stage3: {} not fetched \
                     (scripts/fetch_jsontestsuite.sh)",
                    dir.display()
                );
                return;
            }
            let stage2 = Stage2::new();
            let mut paths: Vec<_> = std::fs::read_dir(&dir)
                .expect("readable test_parsing dir")
                .map(|e| e.expect("readable entry").path())
                .filter(|p| p.extension().is_some_and(|e| e == "json"))
                .collect();
            paths.sort();
            assert!(paths.len() >= 300, "the fetched suite has 318 files");
            for path in paths {
                let name = path.file_name().unwrap().to_string_lossy().into_owned();
                let bytes = std::fs::read(&path).expect("readable suite file");
                diff(&stage2, &ctx, &bytes, &name);
            }
        }

        #[test]
        fn structural_fixtures_match_reference_stage3() {
            let Some(ctx) = ctx_or_skip("structural_fixtures_match_reference_stage3") else {
                return;
            };
            let stage2 = Stage2::new();
            // Accepted documents, Layer-1 rejections (the reference test
            // suite's own fixtures), Layer-2-only problems, scalar torture,
            // empty inputs, and token streams crossing the 1024-token chunk
            // seam — one differential sweep.
            let mut cases: Vec<Vec<u8>> = [
                &br#"{"a":{}}"#[..],
                br#"[[],[[]]]"#,
                br#"{"k":[1,{"n":null},"s"],"e":{}}"#,
                br#"{"":0}"#,
                b"42",
                b"true",
                br#""root string""#,
                b"",
                b" \t\n\r",
                b"]",
                b"}",
                b",1",
                b"{a: 1}",
                b"{1:1}",
                br#"{[: "x"}"#,
                br#"{:"b"}"#,
                b"{,}",
                b"[,1]",
                b"[,]",
                b"[}",
                b"[1,,2]",
                br#"{"x"::"b"}"#,
                br#"{"a":}"#,
                br#"["",]"#,
                br#"{"id":0,}"#,
                b"[1:2]",
                b"[1 true]",
                b"[3[4]]",
                br#"["a" "b"]"#,
                br#"{"a" "b"}"#,
                br#"{"a" b}"#,
                b"[][]",
                br#"{"a": true} "x""#,
                b"[1]x",
                b"null null",
                b"[1,",
                br#"{"a":"#,
                b"{",
                b"[",
                b"[[",
                b"\"abc",
                br#"{"a"#,
                br#"["a", "b"#,
                br#"["": 1]"#,
                br#"{"a":"b":"c"}"#,
                br#""a":1"#,
                br#"{"x", null}"#,
                br#"{"a"}"#,
                br#"{"a""#,
                br#"{"foo":1, "a"}"#,
                br#"["x", truth]"#,
                b"[tru]",
                b"false0",
                b"[True]",
                br#"{'a':0}"#,
                b"[*]",
                b"\xEF\xBB\xBF{}",
                b"1]",
                b"{}}",
                b"{},{}",
                b"1,2",
                b"{}{}",
                b"[1}",
                b"[1",
                br#"[1,"a":2]"#,
                br#"{"a":1,2}"#,
                b"x",
                b"-0.0",
                b"{ true:12}",
            ]
            .iter()
            .map(|d| d.to_vec())
            .collect();

            // Token streams straddling the 1024-token chunk seam, both
            // accepted and rejected-late (the error sits in the second
            // chunk, so K7's cross-chunk error fold is exercised).
            let mut big = b"[".to_vec();
            for i in 0..900 {
                if i > 0 {
                    big.push(b',');
                }
                big.extend_from_slice(format!("\"s{i}\"").as_bytes());
            }
            big.push(b']'); // ~2701 tokens, strings in every chunk
            cases.push(big.clone());
            let mut bad = big.clone();
            let len = bad.len();
            bad[len - 2] = b'x'; // corrupt a late string into `"s899x` ...
            cases.push(bad);
            let mut late_literal = b"[".to_vec();
            for _ in 0..800 {
                late_literal.extend_from_slice(b"true,");
            }
            late_literal.extend_from_slice(b"tru]"); // InvalidLiteral in chunk 3
            cases.push(late_literal);

            for input in &cases {
                diff(
                    &stage2,
                    &ctx,
                    input,
                    &format!("{:?}", String::from_utf8_lossy(&input[..input.len().min(48)])),
                );
            }
        }
    }
}
