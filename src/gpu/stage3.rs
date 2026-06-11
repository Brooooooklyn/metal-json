//! Stage 3 on the GPU: depth scan, counting sort, pair/context and the M3
//! tape (container + root words) — the CB3 structure kernels over the
//! skeleton CB2 produced. `run_structure` (= [`Stage3::run`]) is the full
//! M3 pipeline runner.
//!
//! # Command-buffer shape
//!
//! ```text
//! CB1 → CPU sync 1 → CB2 (K5 K6 K7) → CPU sync 2 → CB2b (K6b)
//!                                  (see crate::gpu::stage2)
//!   ── CPU sync: the skeleton / lists exist; skeleton_total and
//!      tape_word_total are known, so the tape is exact-allocated HERE
//!      (tape_word_total + 2 words, zero-filled — the M3 hole convention)
//!      and CB3 needs no further sync. An empty skeleton (root scalar)
//!      makes CB3 a no-op: every stage-4 output is empty, and the tape —
//!      root words around scalar holes — is written by the CPU ──
//! CB3: depth_partials      1 threadgroup / 1024-element skeleton chunk:
//!                          ±1 bracket weight sums
//!      depth_spine         1 threadgroup: chunk depth carries
//!      depth_apply         per element: depth value (reference stage-4
//!                          phase-1 semantics), DepthLimit / underflow
//!                          checks → per-chunk error words
//!      [sort_hist → sort_matrix_scan → sort_scatter] × passes
//!                          K8: stable counting sort by depth, 5-bit digit
//!                          passes (1 pass for max_depth ≤ 32, 2 at the
//!                          1024 default — stage::sort_passes); stability
//!                          is what makes brackets alternate within a
//!                          depth group
//!      ctx_partials        K9 reduce: segmented walk-state summaries
//!      ctx_spine           1 threadgroup: walk-state chunk carries
//!      pair_ctx_apply      K9 apply: adjacent pairing + XOR 0x06 check +
//!                          pair map both directions, opener forward-fill,
//!                          Layer-2 separator checks, child counts
//!      emit_container_words K12: 1 thread/bracket — container open/close
//!                          tape words via pair-map + tape_ofs gathers,
//!                          child count saturated at 0xFFFFFF
//!      tape_root_words     K13: root prologue + final root words
//!      structure_finalize  1 threadgroup: error fold → header
//!   ── commit, wait: CPU sync 3 reads the header. A structural error
//!      (depth / balance / Layer-2 context / TrailingContent) REJECTS the
//!      input: the stage-3 outputs — including the tape — are never
//!      produced (the stage-2 outputs are kept — stage 2 accepted the
//!      input) ──
//! ```
//!
//! CB3 needs no internal CPU sync: every buffer size derives from
//! `skeleton_total` / `tape_word_total` (both known at the CPU sync 2) and
//! the sort pass count derives from the `max_depth` limit at encode time.
//!
//! The bit-exact spec for every output and error is
//! `reference::stage4_structure` (src/reference/structure.rs) plus, for the
//! tape words, `reference::emit_tape` (src/reference/emit.rs); the
//! in-module tests diff the two backends on identical inputs. The one
//! documented deviation (verdict-preserving, never visible in accepted
//! outputs) is the depth scan's unclamped prefix sum past the first
//! underflow — see `shaders/08_depth.metal`.
//!
//! # The M3 tape and its HOLE convention
//!
//! The tape this stage produces has `tape_word_total + 2` words: the root
//! prologue at `[0]`, the final root at `[len - 1]`, every container
//! open/close word complete (end index + saturated child count — never
//! patched), and **zero-word HOLES** at every scalar/string position. The
//! M4 kernels (K10 numbers, K11 strings) fill the holes; until then the
//! orchestration zero-fills the tape buffer after allocation (GpuBuffer
//! makes no contents guarantee) so holes read as `0u64` deterministically —
//! `0` is unambiguous, since every real tape word carries a nonzero ASCII
//! tag in its top byte. The string buffer is likewise M4's: no M3 kernel
//! writes it, so it is **not** allocated here — only its exact size
//! (`MjHeader::stringbuf_total`, already pinned against the reference by
//! the stage-2 tests) is carried in the output for M4's sync-2 allocation.
//!
//! # Error-class contract (end of M3 structure work)
//!
//! After this stage the GPU pipeline catches exactly the error classes
//! reference stages 1–4 catch: UTF-8 / odd-quote (CB1), Layer-1 syntax +
//! `EmptyInput` (CB2), and depth / balance / Layer-2 context /
//! `TrailingContent` (CB3). Scalar-content errors (number grammar, string
//! escapes, control characters) remain CPU-reference-only until M4 — the
//! `vs_reference` tests pin the split across JSONTestSuite.
//!
//! `max_depth` plumbing: the reference takes it from
//! [`ParserOptions::max_depth`](crate::ParserOptions); the GPU kernels read
//! the same value from `MjParams::reserved0`, set per dispatch by this
//! orchestration ([`Stage3::run`] uses the same [`DEFAULT_MAX_DEPTH`] the
//! reference defaults to).

use crate::error::Result;
use crate::metal::{Dispatch, GpuBuffer, MetalContext, MjParams, THREADGROUP_SIZE};
use crate::parser::DEFAULT_MAX_DEPTH;
use crate::stage::{Stage, Stage1Buffers, Stage3Buffers, sort_passes};
use crate::tape::{make_final_root, make_root};

use super::stage2::{Stage2, Stage2Accepted, Stage2Output, Stage2Run};

/// `MjErrorCode` values produced by CB3. Mirror `shaders/common.h` — keep
/// in sync (a test pins them). The CB3 same-offset tie-break: the only
/// pair of rules that can fire at one offset is a DepthLimit open that the
/// group walk also flags (two-opens / unclosed), and the reference records
/// the phase-1 DepthLimit first — so `ERR_DEPTH_LIMIT` must stay below
/// [`ERR_UNBALANCED`](super::ERR_UNBALANCED) (3 < 20; a const test locks
/// it).
pub const ERR_DEPTH_LIMIT: u32 = 3;
/// See [`ERR_DEPTH_LIMIT`].
pub const ERR_TRAILING_CONTENT: u32 = 4;

/// "No partner" marker in [`Stage3Output::match_index`] (separators).
/// Equal to `reference::NO_MATCH` (a test pins it).
pub const NO_MATCH: u32 = u32::MAX;

/// Everything stage 3 produces, copied back into plain `Vec`s for test
/// ergonomics, mirroring [`Stage2Output`]. The five structure vectors are
/// field-for-field the reference `Stage4Output`, all keyed by skeleton
/// index.
///
/// # Rejection contract
///
/// When [`error`](Self::error) is `Some`, the pipeline has rejected the
/// input and outputs after the failing stage are never produced:
///
/// - a stage-1 / stage-2 rejection leaves [`stage2`](Self::stage2) with its
///   own rejection contract applied and every stage-3 output empty;
/// - a CB3 structural error (depth / balance / Layer-2 / trailing content)
///   keeps the stage-2 outputs — stage 2 accepted the input — but the
///   stage-3 vectors and the [`tape`](Self::tape) stay empty.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Stage3Output {
    /// The stage-2 view of the same run (stage-1 view nested inside).
    pub stage2: Stage2Output,
    /// Depth of each skeleton element (root container = 1). Mirrors
    /// `Stage4Output::depths`.
    pub depths: Vec<u32>,
    /// Skeleton indices in (depth, document-order) order — the stable
    /// counting-sort output. Mirrors `Stage4Output::sorted_by_depth`.
    pub sorted_by_depth: Vec<u32>,
    /// For brackets: skeleton index of the matching bracket; [`NO_MATCH`]
    /// for separators. Mirrors `Stage4Output::match_index`.
    pub match_index: Vec<u32>,
    /// For separators: the enclosing opener byte (`{` or `[`); 0 for
    /// brackets. Mirrors `Stage4Output::context_opener`.
    pub context_opener: Vec<u8>,
    /// For open brackets: number of direct children; 0 for everything
    /// else. Mirrors `Stage4Output::child_counts`.
    pub child_counts: Vec<u32>,
    /// The M3 tape: `tape_word_total + 2` u64 words — root prologue at
    /// `[0]` (payload = final root index), final root at `[len - 1]`,
    /// every container open/close word bit-exact per `reference::emit_tape`
    /// (K12/K13), and **zero-word holes** at every scalar/string position
    /// (the M4 hole convention — see the module docs). Empty on rejected
    /// inputs (the rejection contract: the tape is a stage-3 output).
    pub tape: Vec<u64>,
    /// First error, packed `(byte_offset << 32) | code`, or `None`. Codes:
    /// everything stage 2 can report, plus [`ERR_DEPTH_LIMIT`],
    /// [`ERR_TRAILING_CONTENT`], and the CB3 uses of
    /// [`ERR_UNBALANCED`](super::ERR_UNBALANCED) /
    /// [`ERR_UNEXPECTED_TOKEN`](super::ERR_UNEXPECTED_TOKEN) /
    /// [`ERR_MISSING_COLON`](super::ERR_MISSING_COLON).
    pub error: Option<u64>,
}

impl Stage3Output {
    /// Decode [`error`](Self::error) as `(byte_offset, code)`.
    #[must_use]
    pub fn error_offset_code(&self) -> Option<(u64, u32)> {
        self.error.map(|e| (e >> 32, e as u32))
    }

    /// A rejected output: `stage2` as given, every stage-3 vector empty.
    fn rejected(stage2: Stage2Output, packed_error: u64) -> Self {
        Self {
            stage2,
            error: Some(packed_error),
            ..Self::default()
        }
    }
}

/// The CB3 kernels plus the composed [`Stage2`] (which composes
/// [`Stage1`](super::Stage1)), with lazily-built cached pipelines. Create
/// once and reuse across parses.
#[derive(Debug)]
pub struct Stage3 {
    stage2: Stage2,
    depth_partials: Stage,
    depth_spine: Stage,
    depth_apply: Stage,
    sort_hist: Stage,
    sort_matrix_scan: Stage,
    sort_scatter: Stage,
    ctx_partials: Stage,
    ctx_spine: Stage,
    pair_ctx_apply: Stage,
    emit_container: Stage,
    tape_root: Stage,
    finalize: Stage,
}

impl Stage3 {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            stage2: Stage2::new(),
            depth_partials: Stage::new("depth_partials"),
            depth_spine: Stage::new("depth_spine"),
            depth_apply: Stage::new("depth_apply"),
            sort_hist: Stage::new("sort_hist"),
            sort_matrix_scan: Stage::new("sort_matrix_scan"),
            sort_scatter: Stage::new("sort_scatter"),
            ctx_partials: Stage::new("ctx_partials"),
            ctx_spine: Stage::new("ctx_spine"),
            pair_ctx_apply: Stage::new("pair_ctx_apply"),
            emit_container: Stage::new("emit_container_words"),
            tape_root: Stage::new("tape_root_words"),
            finalize: Stage::new("structure_finalize"),
        }
    }

    /// Run the pipeline through the CB3 structure kernels over `input` on
    /// freshly allocated buffers with the default depth limit
    /// (`DEFAULT_MAX_DEPTH` = 1024, simdjson parity — the same default the
    /// reference's `ParserOptions` uses).
    ///
    /// # Errors
    ///
    /// GPU plumbing failures only; input *content* problems are **data**,
    /// reported in [`Stage3Output::error`].
    pub fn run(&self, ctx: &MetalContext, input: &[u8]) -> Result<Stage3Output> {
        self.run_with_max_depth(ctx, input, DEFAULT_MAX_DEPTH)
    }

    /// [`run`](Self::run), additionally returning the summed GPU execution
    /// time of every command buffer the pipeline committed (CB1 + CB2 +
    /// CB2b + CB3) in seconds (per-command-buffer `GPUEndTime −
    /// GPUStartTime`; zero when the device reports no timestamps; fewer
    /// command buffers when the input is rejected early). Coarse
    /// whole-pipeline timing for the manual sanity test in
    /// `tests/structure.rs`, mirroring [`Stage1::run_timed`](super::Stage1::run_timed); per-kernel
    /// breakdowns are the M5 `timing` feature's job.
    ///
    /// # Errors
    ///
    /// As [`run`](Self::run).
    pub fn run_timed(&self, ctx: &MetalContext, input: &[u8]) -> Result<(Stage3Output, f64)> {
        let mut bufs1 = Stage1Buffers::new(ctx, input)?;
        self.run_with_buffers_timed(ctx, &mut bufs1, DEFAULT_MAX_DEPTH)
    }

    /// [`run`](Self::run) with an explicit depth limit (mirrors
    /// `ParserOptions::max_depth`; kernels receive it via
    /// `MjParams::reserved0`).
    ///
    /// # Errors
    ///
    /// As [`run`](Self::run).
    pub fn run_with_max_depth(
        &self,
        ctx: &MetalContext,
        input: &[u8],
        max_depth: u32,
    ) -> Result<Stage3Output> {
        let mut bufs1 = Stage1Buffers::new(ctx, input)?;
        self.run_with_buffers(ctx, &mut bufs1, max_depth)
    }

    /// [`run_with_max_depth`](Self::run_with_max_depth) over
    /// caller-prepared stage-1 buffers (which must satisfy the
    /// [`Stage1Buffers`] zero/init preconditions). Stage-2 and stage-3
    /// scratch is allocated internally at the exact sizes the syncs report.
    ///
    /// # Errors
    ///
    /// As [`run`](Self::run).
    pub fn run_with_buffers(
        &self,
        ctx: &MetalContext,
        bufs1: &mut Stage1Buffers,
        max_depth: u32,
    ) -> Result<Stage3Output> {
        self.run_with_buffers_timed(ctx, bufs1, max_depth)
            .map(|(out, _)| out)
    }

    /// [`run_with_buffers`](Self::run_with_buffers), additionally returning
    /// the summed GPU command-buffer time in seconds (see
    /// [`run_timed`](Self::run_timed)).
    fn run_with_buffers_timed(
        &self,
        ctx: &MetalContext,
        bufs1: &mut Stage1Buffers,
        max_depth: u32,
    ) -> Result<(Stage3Output, f64)> {
        // --- CB1 → CB2 → CB2b (stage 2 owns the first two syncs) -----------
        let Stage2Accepted {
            stage1,
            bufs2,
            header,
            mut gpu_seconds,
        } = match self.stage2.run_to_lists(ctx, bufs1)? {
            Stage2Run::Rejected(out) => {
                let packed = out.error.expect("rejected runs carry an error");
                return Ok((Stage3Output::rejected(*out, packed), 0.0));
            }
            Stage2Run::Accepted(run) => *run,
        };
        let stage2_out = Stage2::collect_outputs(stage1, &bufs2, &header);

        let skeleton_total =
            usize::try_from(header.skeleton_total).expect("skeleton_total fits usize");
        // CPU sync 2 exact allocation: the tape is tape_word_total + 2 words
        // (root prologue + final root around the per-token words), known
        // before CB3 is encoded — no extra sync for K12/K13.
        let tape_words =
            usize::try_from(header.tape_word_total).expect("tape_word_total fits usize") + 2;

        if skeleton_total == 0 {
            // Root scalars have no structural tokens at all: stage 4 over an
            // empty skeleton is trivially clean (reference parity) and there
            // is nothing to dispatch — including K12 (no brackets). The tape
            // is just the two root words around the scalar's hole(s),
            // written CPU-side (reference `emit_tape` semantics; the M4
            // scalar kernels fill the holes).
            let mut tape = vec![0u64; tape_words];
            tape[0] = make_root(tape_words as u64 - 1);
            tape[tape_words - 1] = make_final_root();
            return Ok((
                Stage3Output {
                    stage2: stage2_out,
                    tape,
                    ..Stage3Output::default()
                },
                gpu_seconds,
            ));
        }

        // The CB3 cooperative kernels are written for full 256-thread
        // groups (same invariant the stage-1/2 orchestrations assert).
        for stage in [
            &self.depth_partials,
            &self.depth_spine,
            &self.depth_apply,
            &self.sort_hist,
            &self.sort_matrix_scan,
            &self.sort_scatter,
            &self.ctx_partials,
            &self.ctx_spine,
            &self.pair_ctx_apply,
            &self.finalize,
        ] {
            let max = stage.pipeline(ctx)?.max_total_threads_per_threadgroup();
            assert!(
                max >= THREADGROUP_SIZE,
                "kernel `{}` supports only {max} threads/threadgroup (< {THREADGROUP_SIZE})",
                stage.name()
            );
        }

        let passes = sort_passes(max_depth);
        let mut bufs3 = Stage3Buffers::new(ctx, skeleton_total, passes)?;
        let chunks = bufs3.chunks();
        let input_len = bufs1.input_len() as u64;

        // The tape buffer, zero-filled to establish the M3 hole convention
        // explicitly (GpuBuffer::alloc makes no contents guarantee; the M4
        // scalar/string kernels will overwrite every hole, at which point
        // the fill can be dropped). K12/K13 write the container/root words
        // into it inside CB3.
        let mut tape_buf = GpuBuffer::alloc(ctx, tape_words * size_of::<u64>())?;
        tape_buf.contents_mut().fill(0);

        let skel_params = MjParams {
            input_len,
            element_count: skeleton_total as u64,
            reserved0: u64::from(max_depth),
            reserved1: 0,
        };
        let chunk_params = MjParams {
            input_len,
            element_count: chunks as u64,
            reserved0: u64::from(max_depth),
            reserved1: 0,
        };
        let matrix_params = MjParams {
            input_len,
            element_count: (32 * chunks) as u64,
            reserved0: u64::from(max_depth),
            reserved1: 0,
        };
        let tape_params = MjParams {
            input_len,
            element_count: tape_words as u64, // K13: final root index + 1
            reserved0: u64::from(max_depth),
            reserved1: 0,
        };

        // --- CB3: one commit, one wait --------------------------------------
        {
            let mut batch = ctx.batch()?;
            let h_skel_ti =
                batch.bind_read(bufs2.skel_token_index.as_ref().expect("lists allocated"));
            let h_skel_pos = batch.bind_read(bufs2.skel_pos.as_ref().expect("lists allocated"));
            let h_skel_byte = batch.bind_read(bufs2.skel_byte.as_ref().expect("lists allocated"));
            let h_chunk_depth = batch.bind_write(&mut bufs3.chunk_depth);
            let h_depths = batch.bind_write(&mut bufs3.depths);
            let h_err = batch.bind_write(&mut bufs3.chunk_error);
            let h_hist = batch.bind_write(&mut bufs3.sort_hist);
            let h_sorted = batch.bind_write(&mut bufs3.sorted);
            let h_scratch = bufs3.sorted_scratch.as_mut().map(|b| batch.bind_write(b));
            let h_ctx = batch.bind_write(&mut bufs3.chunk_ctx);
            let h_match = batch.bind_write(&mut bufs3.match_index);
            let h_opener = batch.bind_write(&mut bufs3.context_opener);
            let h_child = batch.bind_write(&mut bufs3.child_counts);
            let h_tape_ofs = batch.bind_read(&bufs2.tape_ofs);
            let h_tape = batch.bind_write(&mut tape_buf);
            let h_header = batch.bind_write(&mut bufs1.header);

            // Depth scan: reduce → spine → apply.
            self.depth_partials.encode(
                &mut batch,
                &[h_skel_byte, h_chunk_depth],
                Some(&skel_params),
                Dispatch::Threadgroups(chunks),
            )?;
            self.depth_spine.encode(
                &mut batch,
                &[h_chunk_depth],
                Some(&chunk_params),
                Dispatch::Threadgroups(1),
            )?;
            self.depth_apply.encode(
                &mut batch,
                &[h_skel_byte, h_skel_pos, h_chunk_depth, h_depths, h_err],
                Some(&skel_params),
                Dispatch::Threadgroups(chunks),
            )?;

            // K8: LSD radix passes; pass 0 reads the implicit identity
            // ordering, later passes ping-pong so the LAST pass lands in
            // `sorted`.
            for pass in 0..passes {
                let to_sorted = (passes - 1 - pass).is_multiple_of(2);
                let h_out = if to_sorted {
                    h_sorted
                } else {
                    h_scratch.expect("scratch exists for multi-pass sorts")
                };
                // Pass 0's input binding is ignored (identity flag); bind
                // `sorted` as the placeholder.
                let h_in = if pass == 0 {
                    h_sorted
                } else if to_sorted {
                    h_scratch.expect("scratch exists for multi-pass sorts")
                } else {
                    h_sorted
                };
                let pass_params = MjParams {
                    input_len,
                    element_count: skeleton_total as u64,
                    reserved0: u64::from(max_depth),
                    reserved1: (5 * pass as u64) | if pass == 0 { 1 << 8 } else { 0 },
                };
                self.sort_hist.encode(
                    &mut batch,
                    &[h_depths, h_in, h_hist],
                    Some(&pass_params),
                    Dispatch::Threadgroups(chunks),
                )?;
                self.sort_matrix_scan.encode(
                    &mut batch,
                    &[h_hist],
                    Some(&matrix_params),
                    Dispatch::Threadgroups(1),
                )?;
                self.sort_scatter.encode(
                    &mut batch,
                    &[h_depths, h_in, h_hist, h_out],
                    Some(&pass_params),
                    Dispatch::Threadgroups(chunks),
                )?;
            }

            // K9: reduce → spine → apply over the sorted order, then the
            // error fold.
            self.ctx_partials.encode(
                &mut batch,
                &[h_sorted, h_depths, h_skel_byte, h_skel_ti, h_ctx],
                Some(&skel_params),
                Dispatch::Threadgroups(chunks),
            )?;
            self.ctx_spine.encode(
                &mut batch,
                &[h_ctx],
                Some(&chunk_params),
                Dispatch::Threadgroups(1),
            )?;
            self.pair_ctx_apply.encode(
                &mut batch,
                &[
                    h_sorted,
                    h_depths,
                    h_skel_byte,
                    h_skel_pos,
                    h_skel_ti,
                    h_ctx,
                    h_match,
                    h_opener,
                    h_child,
                    h_err,
                ],
                Some(&skel_params),
                Dispatch::Threadgroups(chunks),
            )?;

            // K12/K13: the tape's container + root words. Plain thread
            // grids — neither kernel uses cooperative scans. The serial
            // encoder orders K12 after pair_ctx_apply (it gathers through
            // match_index / child_counts).
            self.emit_container.encode(
                &mut batch,
                &[h_skel_ti, h_skel_byte, h_match, h_child, h_tape_ofs, h_tape],
                Some(&skel_params),
                Dispatch::Threads(skeleton_total),
            )?;
            self.tape_root.encode(
                &mut batch,
                &[h_tape],
                Some(&tape_params),
                Dispatch::Threads(1),
            )?;

            self.finalize.encode(
                &mut batch,
                &[h_err, h_header],
                Some(&chunk_params),
                Dispatch::Threadgroups(1),
            )?;
            gpu_seconds += batch.commit_and_wait_timed()?;
        }

        // --- CB3 → CPU sync 3 -------------------------------------------------
        let header = bufs1.read_header();
        if let Some((offset, code)) = header.first_error() {
            // Structural rejection: stage 2 accepted the input (its outputs
            // are kept), the stage-3 outputs are never produced.
            let packed = (offset << 32) | u64::from(code);
            return Ok((Stage3Output::rejected(stage2_out, packed), gpu_seconds));
        }

        Ok((
            Stage3Output {
                stage2: stage2_out,
                depths: bufs3.depths.as_slice::<u32>().to_vec(),
                sorted_by_depth: bufs3.sorted.as_slice::<u32>().to_vec(),
                match_index: bufs3.match_index.as_slice::<u32>().to_vec(),
                context_opener: bufs3.context_opener.as_slice::<u8>().to_vec(),
                child_counts: bufs3.child_counts.as_slice::<u32>().to_vec(),
                tape: tape_buf.as_slice::<u64>().to_vec(),
                error: None,
            },
            gpu_seconds,
        ))
    }
}

impl Default for Stage3 {
    fn default() -> Self {
        Self::new()
    }
}

/// The full M3 structure pipeline, one shot: CB1 → CB2 → CB2b → CB3
/// (depth scan, K8 sort, K9 pair/context, K12 container words, K13 root
/// words, error fold) over `input` at the default depth limit, with every
/// allocation exact-sized from the header totals at each CPU sync and
/// per-CB error short-circuits per the rejection contract. This is what
/// the parser integration will call; [`Stage3::run`] is the same runner on
/// a reusable [`Stage3`] (tests that run many inputs hold one).
///
/// # Errors
///
/// GPU plumbing failures only; input *content* problems are **data**,
/// reported in [`Stage3Output::error`].
pub(crate) fn run_structure(ctx: &MetalContext, input: &[u8]) -> Result<Stage3Output> {
    Stage3::new().run(ctx, input)
}

/// One-shot convenience over [`Stage3::run`] (builds the pipelines each
/// call; tests that run many inputs should hold a [`Stage3`] instead).
/// Identical to `run_structure`, exported for integration tests.
pub fn run_stage3(ctx: &MetalContext, input: &[u8]) -> Result<Stage3Output> {
    run_structure(ctx, input)
}

#[cfg(test)]
mod tests {
    use super::super::{
        ERR_EMPTY_INPUT, ERR_MISSING_COLON, ERR_MISSING_COMMA, ERR_UNBALANCED,
        ERR_UNEXPECTED_TOKEN,
    };
    use super::*;
    use crate::tape::{make_close, make_open};

    /// GPU gating, as in stage1.rs/stage2.rs: skip without a device unless
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

    fn assert_stage3_empty(out: &Stage3Output, label: &str) {
        assert!(out.depths.is_empty(), "{label}: no depths");
        assert!(out.sorted_by_depth.is_empty(), "{label}: no sort output");
        assert!(out.match_index.is_empty(), "{label}: no pair map");
        assert!(out.context_opener.is_empty(), "{label}: no context");
        assert!(out.child_counts.is_empty(), "{label}: no child counts");
        assert!(out.tape.is_empty(), "{label}: no tape");
    }

    /// Code-order lock for the CB3 tie-break (see [`ERR_DEPTH_LIMIT`]):
    /// a DepthLimit open the group walk also flags must resolve to the
    /// reference's phase-1 DepthLimit. Compile-time.
    #[test]
    fn cb3_error_code_order_is_the_tie_break_contract() {
        const {
            assert!(ERR_DEPTH_LIMIT < ERR_UNBALANCED);
            // Not contested (separators and brackets can't share an
            // offset), but the decoded classes must stay distinct from the
            // Layer-1 block.
            assert!(ERR_TRAILING_CONTENT < ERR_MISSING_COLON);
            assert!(ERR_DEPTH_LIMIT < ERR_TRAILING_CONTENT);
        }
    }

    /// The reference stage-4 worked example, every output hand-computed:
    /// `{"a":[{}]}` — skeleton `{ : [ { } ] }` (the same expectations the
    /// reference's own `depths_sort_pairs_and_counts_for_a_nested_doc`
    /// test pins).
    #[test]
    fn worked_example_structure_outputs_are_exact() {
        let Some(ctx) = ctx_or_skip("worked_example_structure_outputs_are_exact") else {
            return;
        };
        let out = run_stage3(&ctx, br#"{"a":[{}]}"#).unwrap();
        assert_eq!(out.error, None);
        assert_eq!(out.depths, vec![1, 1, 2, 3, 3, 2, 1]);
        assert_eq!(out.sorted_by_depth, vec![0, 1, 6, 2, 5, 3, 4]);
        assert_eq!(out.match_index, vec![6, NO_MATCH, 5, 4, 3, 2, 0]);
        assert_eq!(out.context_opener, vec![0, b'{', 0, 0, 0, 0, 0]);
        assert_eq!(out.child_counts, vec![1, 0, 1, 0, 0, 0, 0]);
        // The stage-2 view rode along intact.
        assert_eq!(out.stage2.skeleton_byte, b"{:[{}]}".to_vec());
        assert_eq!(out.stage2.error, None);
    }

    /// Counting-sort stability pinned by hand: `[[1,2],[3,4]]` — sibling
    /// containers' brackets and separators must stay in document order
    /// inside each depth group (the reference's own stability test).
    #[test]
    fn counting_sort_is_stable_within_depth_groups() {
        let Some(ctx) = ctx_or_skip("counting_sort_is_stable_within_depth_groups") else {
            return;
        };
        let out = run_stage3(&ctx, b"[[1,2],[3,4]]").unwrap();
        assert_eq!(out.error, None);
        // skeleton: [0 [1 ,2 ]3 ,4 [5 ,6 ]7 ]8
        assert_eq!(out.depths, vec![1, 2, 2, 2, 1, 2, 2, 2, 1]);
        assert_eq!(out.sorted_by_depth, vec![0, 4, 8, 1, 2, 3, 5, 6, 7]);
        assert_eq!(out.match_index[1], 3);
        assert_eq!(out.match_index[3], 1);
        assert_eq!(out.match_index[5], 7);
        assert_eq!(out.match_index[7], 5);
        assert_eq!(out.match_index[0], 8);
        assert_eq!(out.child_counts[0], 2);
        assert_eq!(out.child_counts[1], 2);
        assert_eq!(out.child_counts[5], 2);
    }

    /// THE tape pin: the docs/tape-format.md worked example,
    /// `{"a":[1,2.5],"b":"x\n"}` — container/root words bit-identical to
    /// the reference emit's 13-word tape (the same constants the
    /// `worked_example_full_pipeline` test in src/reference/emit.rs pins),
    /// with zero-word holes at every scalar/string position (M4's slots).
    #[test]
    fn worked_example_tape_words_are_exact() {
        let Some(ctx) = ctx_or_skip("worked_example_tape_words_are_exact") else {
            return;
        };
        let out = run_stage3(&ctx, br#"{"a":[1,2.5],"b":"x\n"}"#).unwrap();
        assert_eq!(out.error, None);
        let expected: [u64; 13] = [
            0x7200_0000_0000_000C, // [0]  r -> 12          (K13)
            0x7B00_0002_0000_000C, // [1]  { end=12 count=2 (K12)
            0,                     // [2]  hole: "a"
            0x5B00_0002_0000_0009, // [3]  [ end=9 count=2  (K12)
            0,                     // [4]  hole: l marker
            0,                     // [5]  hole: 1
            0,                     // [6]  hole: d marker
            0,                     // [7]  hole: 2.5
            0x5D00_0000_0000_0003, // [8]  ] open=3         (K12)
            0,                     // [9]  hole: "b"
            0,                     // [10] hole: "x\n"
            0x7D00_0000_0000_0001, // [11] } open=1         (K12)
            0x7200_0000_0000_0000, // [12] r -> 0           (K13)
        ];
        assert_eq!(out.tape, expected);
        assert_eq!(out.stage2.stringbuf_total, 20, "M4's stringbuf size");
    }

    /// Sibling containers: `[[],{}]` — all-container tape, no holes
    /// (mirrors the reference emit's own sibling-containers test).
    #[test]
    fn sibling_containers_tape_words() {
        let Some(ctx) = ctx_or_skip("sibling_containers_tape_words") else {
            return;
        };
        let out = run_stage3(&ctx, b"[[],{}]").unwrap();
        assert_eq!(out.error, None);
        let expected: [u64; 8] = [
            make_root(7),
            make_open(b'[', 7, 2), // outer
            make_open(b'[', 4, 0), // inner []
            make_close(b']', 2),
            make_open(b'{', 6, 0), // inner {}
            make_close(b'}', 4),
            make_close(b']', 1),
            make_final_root(),
        ];
        assert_eq!(out.tape, expected);
    }

    /// K12's child-count saturation at CONTAINER_COUNT_MAX (24 bits),
    /// driven directly through the TestHarness with a hand-built two-element
    /// skeleton (a >16.7M-child container would need a ~33 MB input): the
    /// open word's count field saturates, its end index is untouched, and
    /// unpaired brackets (match_index out of range — only reachable on
    /// rejected inputs) write nothing.
    #[test]
    fn container_count_saturates_at_24_bits_on_the_gpu() {
        use crate::metal::Binding;
        use crate::stage::TestHarness;
        use crate::tape::CONTAINER_COUNT_MAX;

        let harness = match TestHarness::new() {
            Ok(harness) => harness,
            Err(err) => {
                if std::env::var_os("METAL_JSON_REQUIRE_GPU").is_some_and(|v| v == "1") {
                    panic!("METAL_JSON_REQUIRE_GPU=1 but no usable Metal device: {err}");
                }
                eprintln!(
                    "SKIP container_count_saturates_at_24_bits_on_the_gpu: no device ({err})"
                );
                return;
            }
        };
        let k12 = crate::stage::Stage::new("emit_container_words");

        // Skeleton: `{` (token 0) paired with `}` (token 1), plus a third
        // stray open whose match_index is NO_MATCH (must not write).
        let skel_token_index = harness.upload::<u32>(&[0, 1, 2]).unwrap();
        let skel_byte = harness.upload::<u8>(b"{}[").unwrap();
        let match_index = harness.upload::<u32>(&[1, 0, NO_MATCH]).unwrap();
        let child_counts = harness
            .upload::<u32>(&[CONTAINER_COUNT_MAX + 1, 0, 5])
            .unwrap();
        let tape_ofs = harness.upload::<u32>(&[1, 2, 3]).unwrap();
        let mut tape = harness.alloc_zeroed::<u64>(5).unwrap();

        let params = MjParams {
            input_len: 0,
            element_count: 3,
            reserved0: u64::from(DEFAULT_MAX_DEPTH),
            reserved1: 0,
        };
        harness
            .run(
                &k12,
                &mut [
                    Binding::Read(&skel_token_index),
                    Binding::Read(&skel_byte),
                    Binding::Read(&match_index),
                    Binding::Read(&child_counts),
                    Binding::Read(&tape_ofs),
                    Binding::ReadWrite(&mut tape),
                ],
                Some(&params),
                Dispatch::Threads(3),
            )
            .unwrap();

        let words: Vec<u64> = harness.read_back(&tape);
        assert_eq!(
            words[1],
            make_open(b'{', 3, CONTAINER_COUNT_MAX),
            "count saturates, end index untouched (reference make_open parity)"
        );
        assert_eq!(
            words[1],
            crate::tape::make_open(b'{', 3, CONTAINER_COUNT_MAX + 1),
            "the Rust constructor saturates to the same word"
        );
        assert_eq!(words[2], make_close(b'}', 1));
        assert_eq!(words[3], 0, "unpaired bracket writes nothing");
        assert_eq!(words[0], 0);
        assert_eq!(words[4], 0);
    }

    #[test]
    fn root_scalars_have_empty_structure_outputs_and_root_word_tapes() {
        let Some(ctx) = ctx_or_skip("root_scalars_have_empty_structure_outputs_and_root_word_tapes")
        else {
            return;
        };
        let stage3 = Stage3::new();
        // (input, expected tape) — root words around zero-word scalar holes
        // (reference emit_tape's root-scalar tapes with the scalar words
        // blanked; numbers occupy two holes: marker + value).
        let cases: &[(&[u8], &[u64])] = &[
            (b"42", &[make_root(3), 0, 0, make_final_root()]),
            (b"true", &[make_root(2), 0, make_final_root()]),
            (b"\"x\"", &[make_root(2), 0, make_final_root()]),
        ];
        for &(input, want_tape) in cases {
            let out = stage3.run(&ctx, input).unwrap();
            assert_eq!(out.error, None, "{input:?}");
            // Structure vectors are empty (no skeleton), but the tape is
            // not: the CPU writes the root words around the M4 holes.
            assert!(out.depths.is_empty(), "{input:?}");
            assert!(out.sorted_by_depth.is_empty(), "{input:?}");
            assert!(out.match_index.is_empty(), "{input:?}");
            assert!(out.context_opener.is_empty(), "{input:?}");
            assert!(out.child_counts.is_empty(), "{input:?}");
            assert_eq!(out.tape, want_tape, "{input:?}: root-scalar tape");
            // The empty skeleton mirrors reference stage 4 on no input.
            assert!(out.stage2.skeleton_byte.is_empty());
        }
    }

    /// Structural rejections: the (offset, code) pairs the reference
    /// oracle's stage-4 test suite pins (src/reference/structure.rs),
    /// including the earliest-offset-across-phases case.
    #[test]
    fn structural_rejections_report_reference_offsets_and_codes() {
        let Some(ctx) = ctx_or_skip("structural_rejections_report_reference_offsets_and_codes")
        else {
            return;
        };
        let stage3 = Stage3::new();
        let cases: &[(&[u8], u64, u32)] = &[
            // unmatched closes (depth-scan underflow)
            (b"1]", 1, ERR_UNBALANCED),
            (b"{}}", 2, ERR_UNBALANCED),
            (b"[[]]]", 4, ERR_UNBALANCED),
            // unclosed opens (group-walk leftover, at the OPEN's offset)
            (b"[1", 0, ERR_UNBALANCED),
            (br#"{"a":1"#, 0, ERR_UNBALANCED),
            (b"[[1]", 0, ERR_UNBALANCED),
            // mismatched bracket types (xor 0x26, at the close)
            (b"[1}", 2, ERR_UNBALANCED),
            (br#"{"a":1]"#, 6, ERR_UNBALANCED),
            // depth-0 separators (trailing content)
            (b"{},1", 2, ERR_TRAILING_CONTENT),
            (b"1,2", 1, ERR_TRAILING_CONTENT),
            (br#"[""],1"#, 4, ERR_TRAILING_CONTENT),
            (b"{},{}", 2, ERR_TRAILING_CONTENT),
            // colon inside an array (Layer-2 context)
            (br#"[1,"a":2]"#, 6, ERR_UNEXPECTED_TOKEN),
            // object comma without a member colon
            (br#"{ "foo" : "bar", "a" }"#, 15, ERR_MISSING_COLON),
            (br#"{"a":1,2}"#, 6, ERR_MISSING_COLON),
            (br#"{"a":1,"b"}"#, 6, ERR_MISSING_COLON),
            (br#"{"a":1,[]}"#, 6, ERR_MISSING_COLON),
            // earliest offset wins across phases: underflow@2 vs depth-0
            // comma@3 vs unclosed open@4
            (b"{}},[1", 2, ERR_UNBALANCED),
        ];
        for &(input, offset, code) in cases {
            let out = stage3.run(&ctx, input).unwrap();
            assert_eq!(
                out.error_offset_code(),
                Some((offset, code)),
                "{:?}",
                String::from_utf8_lossy(input)
            );
            // Rejection contract: stage-3 outputs are never produced, but
            // stage 2 accepted the input so its outputs are kept.
            assert_stage3_empty(&out, "rejected");
            assert!(
                !out.stage2.skeleton_byte.is_empty(),
                "{input:?}: stage-2 outputs kept on a CB3 rejection"
            );
        }
        // ... while the well-formed sibling member passes.
        assert!(stage3.run(&ctx, br#"{"a":1,"b":2}"#).unwrap().error.is_none());
    }

    /// Depth limit: respects the explicit max_depth (reference stage-4
    /// `depth_limit_respects_max_depth`), at simdjson parity by default.
    #[test]
    fn depth_limit_respects_max_depth() {
        let Some(ctx) = ctx_or_skip("depth_limit_respects_max_depth") else {
            return;
        };
        let stage3 = Stage3::new();

        let out = stage3.run_with_max_depth(&ctx, b"[[[[]]]]", 4).unwrap();
        assert_eq!(out.error, None, "4 deep at limit 4");
        let out = stage3.run_with_max_depth(&ctx, b"[[[[]]]]", 3).unwrap();
        assert_eq!(
            out.error_offset_code(),
            Some((3, ERR_DEPTH_LIMIT)),
            "the 4th open bracket"
        );

        // simdjson parity at the default limit; nest(1024) also sweeps the
        // full 2-pass key range 0..=1023 with cross-chunk groups.
        let nest = |depth: usize| {
            let mut s = "[".repeat(depth);
            s.push_str(&"]".repeat(depth));
            s.into_bytes()
        };
        let out = stage3.run(&ctx, &nest(1024)).unwrap();
        assert_eq!(out.error, None, "1024 deep is exactly at the limit");
        let m = out.depths.len();
        assert_eq!(m, 2048);
        // nest(d): open i pairs close 2d-1-i; group keys sort to
        // [0, 2d-1, 1, 2d-2, ...].
        for i in 0..1024 {
            assert_eq!(out.depths[i], (i + 1) as u32);
            assert_eq!(out.match_index[i], (m - 1 - i) as u32);
            assert_eq!(out.match_index[m - 1 - i], i as u32);
            assert_eq!(out.sorted_by_depth[2 * i], i as u32);
            assert_eq!(out.sorted_by_depth[2 * i + 1], (m - 1 - i) as u32);
            let want = if i == 1023 { 0 } else { 1 };
            assert_eq!(out.child_counts[i], want, "children of open {i}");
        }

        let out = stage3.run(&ctx, &nest(1025)).unwrap();
        assert_eq!(
            out.error_offset_code(),
            Some((1024, ERR_DEPTH_LIMIT)),
            "the 1025th open bracket"
        );
    }

    /// The single-pass sort path (max_depth ≤ 32 → 1 digit) must agree
    /// with the default two-pass path on the same clean input.
    #[test]
    fn single_pass_and_two_pass_sorts_agree() {
        let Some(ctx) = ctx_or_skip("single_pass_and_two_pass_sorts_agree") else {
            return;
        };
        let stage3 = Stage3::new();
        let input = br#"{"a":[{"b":[1,2,{"c":[]}]},[[{}]]],"d":0}"#;
        assert_eq!(crate::stage::sort_passes(31), 1);
        assert_eq!(crate::stage::sort_passes(DEFAULT_MAX_DEPTH), 2);
        let one = stage3.run_with_max_depth(&ctx, input, 31).unwrap();
        let two = stage3.run(&ctx, input).unwrap();
        assert_eq!(one.error, None);
        assert_eq!(one.depths, two.depths);
        assert_eq!(one.sorted_by_depth, two.sorted_by_depth);
        assert_eq!(one.match_index, two.match_index);
        assert_eq!(one.context_opener, two.context_opener);
        assert_eq!(one.child_counts, two.child_counts);
        assert_eq!(one.tape, two.tape);
        // Boundary: depth == limit passes, one past fails, on the 1-pass path.
        let nest3 = b"[[[]]]";
        assert_eq!(
            stage3.run_with_max_depth(&ctx, nest3, 3).unwrap().error,
            None
        );
        assert_eq!(
            stage3
                .run_with_max_depth(&ctx, nest3, 2)
                .unwrap()
                .error_offset_code(),
            Some((2, ERR_DEPTH_LIMIT))
        );
    }

    /// Earlier-stage rejections carry forward unchanged: CB3 never runs,
    /// stage-3 outputs stay empty.
    #[test]
    fn earlier_stage_rejections_carry_forward() {
        let Some(ctx) = ctx_or_skip("earlier_stage_rejections_carry_forward") else {
            return;
        };
        let stage3 = Stage3::new();

        // Stage 1: invalid UTF-8.
        let out = stage3.run(&ctx, b"ab\x80").unwrap();
        assert_eq!(out.error_offset_code(), Some((2, super::super::ERR_UTF8)));
        assert_stage3_empty(&out, "utf8");

        // CPU verdict: empty input.
        let out = stage3.run(&ctx, b" \t\n").unwrap();
        assert_eq!(out.error_offset_code(), Some((0, ERR_EMPTY_INPUT)));
        assert_stage3_empty(&out, "empty");

        // CB2 Layer 1: missing comma; separator→end (reported at
        // input_len, the reference's virtual-end offset); lone open.
        for (input, offset, code) in [
            (&b"[1 true]"[..], 3, ERR_MISSING_COMMA),
            (br#"[""],"#, 5, ERR_UNEXPECTED_TOKEN),
            (b"[", 0, ERR_UNBALANCED),
        ] {
            let out = stage3.run(&ctx, input).unwrap();
            assert_eq!(
                out.error_offset_code(),
                Some((offset, code)),
                "{:?}",
                String::from_utf8_lossy(input)
            );
            assert_stage3_empty(&out, "layer1");
            // CB2's own rejection contract: no skeleton either.
            assert!(out.stage2.skeleton_byte.is_empty());
        }
    }

    /// The M3/M4 error-class split: scalar-content problems (number
    /// grammar, escapes, control characters) are NOT structure errors —
    /// they must pass CB3 with full structure outputs and stay
    /// CPU-reference-only until M4.
    #[test]
    fn scalar_content_problems_pass_the_structure_stage() {
        let Some(ctx) = ctx_or_skip("scalar_content_problems_pass_the_structure_stage") else {
            return;
        };
        let stage3 = Stage3::new();
        for input in [
            &b"[01]"[..],          // number grammar (reference stage 5)
            b"[1e+]",              // number grammar
            b"[-]",                // number grammar
            br#"["\q"]"#,          // bad escape (reference stage 6)
            b"[\"\x01\"]",         // raw control char in string (stage 6)
            br#"{"k":1e999}"#,     // overflow handling is stage 5's
        ] {
            let out = stage3.run(&ctx, input).unwrap();
            assert_eq!(
                out.error,
                None,
                "{:?} must pass the structure stage",
                String::from_utf8_lossy(input)
            );
            assert!(
                !out.match_index.is_empty(),
                "{:?}: structure outputs produced",
                String::from_utf8_lossy(input)
            );
        }
    }

    /// Multi-chunk skeletons: groups span 1024-element chunk seams in both
    /// the depth scan and the sorted order, so the spine carries and the
    /// segmented forward-fill must all cross chunks. Hand-computed shape:
    /// `[{"a":1},{"a":1},...]` with 1500 objects → skeleton
    /// `[` + 1500×`{ : }` + 1499×`,` + `]` = 6001 elements (6 chunks);
    /// the depth-1 group alone has 3001 elements.
    #[test]
    fn multi_chunk_skeletons_pair_and_count_correctly() {
        let Some(ctx) = ctx_or_skip("multi_chunk_skeletons_pair_and_count_correctly") else {
            return;
        };
        let n = 1500usize;
        let mut input = b"[".to_vec();
        for i in 0..n {
            if i > 0 {
                input.push(b',');
            }
            input.extend_from_slice(br#"{"a":1}"#);
        }
        input.push(b']');

        let out = run_stage3(&ctx, &input).unwrap();
        assert_eq!(out.error, None);
        let m = out.depths.len();
        assert_eq!(m, 2 + 3 * n + (n - 1));

        // Skeleton layout: [0; then object k at 1 + 4k = {, :, }; comma at
        // 4k + 4 (k < n-1); ] last.
        assert_eq!(out.child_counts[0], n as u32, "outer array children");
        assert_eq!(out.match_index[0], (m - 1) as u32);
        assert_eq!(out.match_index[m - 1], 0);
        assert_eq!(out.depths[0], 1);
        assert_eq!(out.depths[m - 1], 1);
        for k in 0..n {
            let open = 1 + 4 * k;
            assert_eq!(out.depths[open], 2, "object {k} open");
            assert_eq!(out.depths[open + 1], 2, "object {k} colon");
            assert_eq!(out.depths[open + 2], 2, "object {k} close");
            assert_eq!(out.match_index[open], (open + 2) as u32);
            assert_eq!(out.match_index[open + 2], open as u32);
            assert_eq!(out.child_counts[open], 1, "object {k} member count");
            assert_eq!(out.context_opener[open + 1], b'{', "object {k} colon ctx");
            if k + 1 < n {
                let comma = open + 3;
                assert_eq!(out.depths[comma], 1);
                assert_eq!(out.context_opener[comma], b'[', "comma {k} ctx");
                assert_eq!(out.match_index[comma], NO_MATCH);
            }
        }
        // Stable sort: the depth-1 group is [0, commas in order, ]; the
        // depth-2 group is the objects' brackets in document order.
        assert_eq!(out.sorted_by_depth[0], 0);
        assert_eq!(out.sorted_by_depth[n], (m - 1) as u32);
        assert_eq!(out.sorted_by_depth[n + 1], 1, "first depth-2 element");

        // The tape (K12/K13) across chunk seams. Footprints per object are
        // 5 ({ "a l value }), so object k's words sit at tape 2 + 5k:
        //   r [ ({ "a l 1 })×n ] r   — len = 5n + 4.
        assert_eq!(out.tape.len(), 5 * n + 4);
        assert_eq!(out.tape[0], make_root(5 * n as u64 + 3));
        assert_eq!(out.tape[1], make_open(b'[', 5 * n as u32 + 3, n as u32));
        assert_eq!(out.tape[5 * n + 2], make_close(b']', 1));
        assert_eq!(out.tape[5 * n + 3], make_final_root());
        for k in 0..n {
            let open = 2 + 5 * k;
            assert_eq!(
                out.tape[open],
                make_open(b'{', open as u32 + 5, 1),
                "object {k} open word"
            );
            assert_eq!(
                out.tape[open + 4],
                make_close(b'}', open as u32),
                "object {k} close word"
            );
            for hole in 1..=3 {
                assert_eq!(out.tape[open + hole], 0, "object {k} hole {hole}");
            }
        }
    }

    // --- vs the cpu-reference oracle --------------------------------------

    #[cfg(feature = "cpu-reference")]
    mod vs_reference {
        use super::*;
        use crate::reference::{
            Stage3Output as RefStage3Output, Stage4Output, Token, emit_tape, stage1_classify,
            stage2_tokens, stage3_validate_local, stage4_structure, stage5_scalars,
            stage6_strings,
        };
        use crate::{Error as CrateError, SyntaxErrorKind};

        /// The GPU code for each Layer-1 SyntaxErrorKind (mirrors the
        /// stage-2 test mapping).
        fn layer1_code(kind: SyntaxErrorKind) -> u32 {
            match kind {
                SyntaxErrorKind::MissingColon => ERR_MISSING_COLON,
                SyntaxErrorKind::MissingComma => ERR_MISSING_COMMA,
                SyntaxErrorKind::UnexpectedToken => ERR_UNEXPECTED_TOKEN,
                SyntaxErrorKind::InvalidLiteral => super::super::super::ERR_INVALID_LITERAL,
                SyntaxErrorKind::UnbalancedBrackets => ERR_UNBALANCED,
                SyntaxErrorKind::UnterminatedString => {
                    super::super::super::ERR_UNTERMINATED_STRING
                }
                SyntaxErrorKind::EmptyInput => ERR_EMPTY_INPUT,
                other => panic!("reference stage 3 cannot produce {other:?}"),
            }
        }

        /// The GPU (offset, code) for a reference stage-4 error.
        fn stage4_code(err: &CrateError) -> (u64, u32) {
            match err {
                CrateError::Syntax { offset, kind } => {
                    let code = match kind {
                        SyntaxErrorKind::UnbalancedBrackets => ERR_UNBALANCED,
                        SyntaxErrorKind::UnexpectedToken => ERR_UNEXPECTED_TOKEN,
                        SyntaxErrorKind::MissingColon => ERR_MISSING_COLON,
                        other => panic!("reference stage 4 cannot produce {other:?}"),
                    };
                    (*offset, code)
                }
                CrateError::DepthLimit { offset, .. } => (*offset, ERR_DEPTH_LIMIT),
                CrateError::TrailingContent { offset } => (*offset, ERR_TRAILING_CONTENT),
                other => panic!("reference stage 4 cannot produce {other:?}"),
            }
        }

        /// Run both backends on `input` with the same `max_depth` and
        /// require agreement: inputs that pass reference stages 1–4
        /// compare every structure vector bit-for-bit against
        /// `Stage4Output`; rejected inputs compare the packed verdict,
        /// with the documented odd-quote offset exception.
        fn diff(stage3: &Stage3, ctx: &MetalContext, input: &[u8], max_depth: u32, label: &str) {
            let got = stage3
                .run_with_max_depth(ctx, input, max_depth)
                .unwrap_or_else(|e| panic!("{label}: GPU stage 3 failed: {e}"));

            // Reference stage 1.
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
                // Odd quotes reject in CB1 at offset input_len (documented
                // provisional offset — class parity only, see stage 1).
                assert_eq!(
                    got.error_offset_code(),
                    Some((input.len() as u64, super::super::super::ERR_STRING)),
                    "{label}: odd-quote verdict"
                );
                return;
            }

            // Reference stages 2–3 (Layer 1).
            let tokens = stage2_tokens(&bitmaps, input);
            let s3 = match stage3_validate_local(&tokens, input) {
                Err(CrateError::Syntax { offset, kind }) => {
                    assert_eq!(
                        got.error_offset_code(),
                        Some((offset, layer1_code(kind))),
                        "{label}: Layer-1 verdict for reference {kind:?}"
                    );
                    assert_stage3_empty(&got, label);
                    return;
                }
                Err(other) => panic!("{label}: unexpected reference error {other:?}"),
                Ok(s3) => s3,
            };

            // Reference stage 4 — the CB3 spec.
            match stage4_structure(&s3.skeleton, max_depth) {
                Err(err) => {
                    let (offset, code) = stage4_code(&err);
                    assert_eq!(
                        got.error_offset_code(),
                        Some((offset, code)),
                        "{label}: structural verdict for reference {err:?}"
                    );
                    // Rejection contract: stage-2 outputs kept, stage-3
                    // outputs never produced.
                    assert_stage3_empty(&got, label);
                    assert_eq!(
                        got.stage2.skeleton_byte.len(),
                        s3.skeleton.len(),
                        "{label}: stage-2 outputs kept"
                    );
                }
                Ok(want) => {
                    assert_eq!(got.error, None, "{label}: spurious GPU error");
                    assert_eq!(got.depths, want.depths, "{label}: depths");
                    assert_eq!(
                        got.sorted_by_depth, want.sorted_by_depth,
                        "{label}: sorted_by_depth"
                    );
                    assert_eq!(got.match_index, want.match_index, "{label}: match_index");
                    assert_eq!(
                        got.context_opener, want.context_opener,
                        "{label}: context_opener"
                    );
                    assert_eq!(got.child_counts, want.child_counts, "{label}: child_counts");
                    diff_tape(&got, &tokens, &s3, &want, input, label);
                }
            }
        }

        /// The M3 tape oracle for an input that passed reference stages 1-4:
        ///
        /// 1. tape_ofs == the reference emit's tape positions (1 + the
        ///    exclusive footprint prefix sum);
        /// 2. tape length == tape_word_total + 2;
        /// 3. every container/root position holds the bit-exact reference
        ///    word (open: one-past-close + saturated count; close: open
        ///    index; root prologue/epilogue), rebuilt here straight from the
        ///    reference stage-3/4 outputs;
        /// 4. every other position — the scalar/string slots stages 5/6 own
        ///    in M4 — is a zero-word HOLE (the documented M3 convention);
        /// 5. when the FULL reference parse succeeds (scalar content clean),
        ///    the rebuilt words also match `reference::emit_tape`'s actual
        ///    tape at those positions and the lengths agree — so the
        ///    rebuild can never drift from the real emitter. (Inputs that
        ///    fail stages 5/6 still verify 1-4: the structure of the tape
        ///    is defined by stages 3/4 alone.)
        fn diff_tape(
            got: &Stage3Output,
            tokens: &[Token],
            s3: &RefStage3Output,
            want: &Stage4Output,
            input: &[u8],
            label: &str,
        ) {
            // 1) tape positions (the reference emit's tape_pos vector).
            let mut tape_pos = vec![0u32; tokens.len()];
            let mut running = 1u32;
            for (t, fp) in s3.footprints.iter().enumerate() {
                tape_pos[t] = running;
                running += fp;
            }
            assert_eq!(got.stage2.tape_ofs, tape_pos, "{label}: tape_ofs map");

            // 2) length: footprint total + the two root words.
            let len = running as usize + 1;
            assert_eq!(got.tape.len(), len, "{label}: tape length");

            // 3) container/root words, rebuilt from the reference outputs.
            let mut expected: Vec<Option<u64>> = vec![None; len];
            expected[0] = Some(make_root(u64::from(running)));
            expected[len - 1] = Some(make_final_root());
            for (si, rec) in s3.skeleton.iter().enumerate() {
                if rec.byte == b':' || rec.byte == b',' {
                    continue;
                }
                let partner_si = want.match_index[si] as usize;
                let partner_pos = tape_pos[s3.skeleton[partner_si].token_index as usize];
                let own = tape_pos[rec.token_index as usize] as usize;
                let word = if rec.byte == b'{' || rec.byte == b'[' {
                    make_open(rec.byte, partner_pos + 1, want.child_counts[si])
                } else {
                    make_close(rec.byte, partner_pos)
                };
                expected[own] = Some(word);
            }
            for (i, want_word) in expected.iter().enumerate() {
                match want_word {
                    Some(word) => {
                        assert_eq!(got.tape[i], *word, "{label}: tape[{i}]");
                    }
                    // 4) holes are zero words.
                    None => assert_eq!(got.tape[i], 0, "{label}: hole at tape[{i}]"),
                }
            }

            // 5) self-check the rebuild against the real emitter when the
            // scalar stages accept the input.
            if let (Ok(scalars), Ok(strings)) =
                (stage5_scalars(tokens, input), stage6_strings(tokens, input))
            {
                let (ref_tape, _) = emit_tape(tokens, s3, want, &scalars, &strings);
                assert_eq!(ref_tape.len(), got.tape.len(), "{label}: emit length");
                for (i, want_word) in expected.iter().enumerate() {
                    if let Some(word) = want_word {
                        assert_eq!(
                            ref_tape.as_words()[i],
                            *word,
                            "{label}: rebuilt word drifted from reference emit at tape[{i}]"
                        );
                    }
                }
            }
        }

        /// GPU NO_MATCH must be the reference's NO_MATCH.
        #[test]
        fn no_match_marker_matches_the_reference() {
            assert_eq!(NO_MATCH, crate::reference::NO_MATCH);
        }

        #[test]
        fn corpus_files_match_reference_stage4() {
            let Some(ctx) = ctx_or_skip("corpus_files_match_reference_stage4") else {
                return;
            };
            let stage3 = Stage3::new();
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
                diff(&stage3, &ctx, &bytes, DEFAULT_MAX_DEPTH, &name);
            }
        }

        /// Every JSONTestSuite file, GPU stages 1–3 vs reference stages
        /// 1–4: the M3 error-class split pinned across the whole suite.
        /// Files the reference rejects in stages 1–4 must produce the same
        /// packed verdict; files it only rejects in stages 5–6 (scalar
        /// content → M4) must pass with bit-identical structure outputs.
        #[test]
        fn jsontestsuite_files_match_reference_stage4() {
            let Some(ctx) = ctx_or_skip("jsontestsuite_files_match_reference_stage4") else {
                return;
            };
            let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("data/JSONTestSuite/test_parsing");
            if !dir.is_dir() {
                eprintln!(
                    "SKIP jsontestsuite_files_match_reference_stage4: {} not fetched \
                     (scripts/fetch_jsontestsuite.sh)",
                    dir.display()
                );
                return;
            }
            let stage3 = Stage3::new();
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
                diff(&stage3, &ctx, &bytes, DEFAULT_MAX_DEPTH, &name);
            }
        }

        #[test]
        fn structural_fixtures_match_reference_stage4() {
            let Some(ctx) = ctx_or_skip("structural_fixtures_match_reference_stage4") else {
                return;
            };
            let stage3 = Stage3::new();
            // Accepted documents (nesting shapes, sibling containers, empty
            // containers, root scalars), every reference stage-4 rejection
            // fixture, Layer-1 and stage-1 rejections (verdict carry), and
            // scalar-content cases that must pass — one differential sweep.
            let mut cases: Vec<Vec<u8>> = [
                // accepted
                &br#"{"a":[{}]}"#[..],
                br#"[[1,2],[3,4]]"#,
                br#"{"a":{}}"#,
                br#"[[],[[]]]"#,
                br#"{"k":[1,{"n":null},"s"],"e":{}}"#,
                br#"[{"x":0},2]"#,
                br#"{"a":1,"b":[1,2,3]}"#,
                br#"[1,{"a":1,"b":2},3]"#,
                br#"{"":0}"#,
                b"[]",
                b"{}",
                b"[[],[],[]]",
                b"42",
                br#""root string""#,
                // stage-4 rejections
                b"1]",
                b"{}}",
                b"[[]]]",
                b"[1",
                br#"{"a":1"#,
                b"[[1]",
                b"[1}",
                br#"{"a":1]"#,
                b"{},1",
                b"1,2",
                br#"[""],1"#,
                b"{},{}",
                br#"[1,"a":2]"#,
                br#"{ "foo" : "bar", "a" }"#,
                br#"{"a":1,2}"#,
                br#"{"a":1,"b"}"#,
                br#"{"a":1,[]}"#,
                br#"{"foo":1, "a"}"#,
                b"{}},[1",
                // earlier-stage rejections (verdict carry-forward)
                b"",
                b" \t\n\r",
                b"]",
                b"[1 true]",
                b"{\"a\":1,}",
                b"\"abc",
                b"\xEF\xBB\xBF{}",
                b"ab\x80",
                // scalar-content problems that must PASS this stage
                b"[01]",
                b"[1e+]",
                br#"["\q"]"#,
                b"[\"\x01\"]",
            ]
            .iter()
            .map(|d| d.to_vec())
            .collect();

            // Skeletons straddling the 1024-element chunk seam, accepted
            // and rejected-late (the error sits in a late chunk, exercising
            // the cross-chunk error fold and segmented carries).
            let wide = |n: usize, close: &[u8]| {
                let mut v = b"[".to_vec();
                for i in 0..n {
                    if i > 0 {
                        v.push(b',');
                    }
                    v.extend_from_slice(br#"{"k":[0]}"#);
                }
                v.extend_from_slice(close);
                v
            };
            cases.push(wide(400, b"]")); // ~2800 skeleton elements, clean
            cases.push(wide(400, b"}")); // mismatched close in the last chunk
            cases.push(wide(400, b"")); // unclosed outer array (error at 0)
            let mut deep = b"[".repeat(700);
            deep.extend_from_slice(b"1".as_slice());
            deep.extend_from_slice(&b"]".repeat(700));
            cases.push(deep); // 1400 elements, every depth group tiny

            for input in &cases {
                let label = format!(
                    "{:?}",
                    String::from_utf8_lossy(&input[..input.len().min(48)])
                );
                diff(&stage3, &ctx, input, DEFAULT_MAX_DEPTH, &label);
            }

            // Custom depth limits run through the same oracle, on both the
            // 1-pass and the 2-pass sort paths.
            for max_depth in [1, 2, 3, 31, 32, 33] {
                for input in [&b"[[[[]]]]"[..], br#"{"a":[{"b":[]}]}"#, b"[[1,2],[3,4]]"] {
                    let label = format!(
                        "max_depth={max_depth} {:?}",
                        String::from_utf8_lossy(input)
                    );
                    diff(&stage3, &ctx, input, max_depth, &label);
                }
            }
        }
    }
}
