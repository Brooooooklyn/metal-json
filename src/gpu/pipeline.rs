//! The full GPU parse pipeline — CB1 → CB2 → CB2b → CB3 with the M4
//! scalar kernels (K10 numbers, K11 strings) encoded into the same CB3
//! command buffer as the M3 structure kernels. This is what
//! [`Parser`](crate::Parser)'s `Backend::Gpu` arm drives; the per-milestone
//! runners ([`Stage3`], [`Numbers`](super::Numbers),
//! [`StringsStage`](super::StringsStage)) remain the narrower test
//! orchestrations this module evolved from (the design decision: rather
//! than growing `stage3.rs` further, the production composition lives
//! here and reuses `Stage3::encode_structure` so the CB3 structure block
//! cannot drift between the two).
//!
//! # Command-buffer shape
//!
//! ```text
//! CB1 → CPU sync 1 → CB2 (K5 K6 K7) → CPU sync 2 → CB2b (K6b)
//!                                  (see crate::gpu::stage2)
//!   ── CPU sync 2 knows every remaining size: the tape is exact-allocated
//!      at tape_word_total + 2 words, the string buffer at stringbuf_total
//!      bytes (the K7 raw-length prefix-sum total), the record-offset /
//!      fixup lists at string_total / scalar_total entries — CB3 needs no
//!      further sync ──
//! CB3: [structure kernels]   skeleton_total > 0 only: depth scan → K8
//!                            sort → K9 pair/context → K12 container words
//!                            → K13 root words → error fold
//!                            (Stage3::encode_structure — see gpu::stage3)
//!      [string kernels]      string_total > 0 only: record offsets → K11
//!                            unescape (records + `"` tape words; strings
//!                            over the long-string threshold → fixup list)
//!                            → error fold (see gpu::strings for the
//!                            kernel contract and the long-string valve)
//!      [number kernel]       scalar_total > 0 only: K10 numbers/literals
//!                            (l/u/d two-word + t/f/n one-word entries,
//!                            hard cases → fixup list; see gpu::numbers)
//!   ── commit, wait: CPU sync 3 reads the merged verdict, then patches
//!      the fixup-listed value words in the shared tape buffer via
//!      patch_number_fixups (re-parsing those few scalars on the CPU) and
//!      the fixup-listed long strings via patch_long_strings (re-running
//!      the shared reference unescaper into the precomputed record slots
//!      of the shared string buffer + their `"` tape words) ──
//! ```
//!
//! All three CB3 blocks consume only stage-1/2 outputs (token stream,
//! lists, `tape_ofs`) and write disjoint tape words (containers/roots vs
//! numbers/literals vs strings — each token's footprint words belong to
//! exactly one kernel), so their relative encode order is irrelevant; the
//! serial encoder only has to order the producer→consumer chains *within*
//! each block. A root scalar (`skeleton_total == 0`) skips the structure
//! block and gets its two root words written CPU-side into the shared tape
//! buffer (reference `emit_tape` semantics), exactly like the M3 runner.
//!
//! # Error contract (the M4 completion)
//!
//! After CB3 the GPU catches **every** error class the full reference
//! parse catches. The verdict is the packed minimum over three sources:
//!
//! - `header.error` — the CB1/CB2 carries plus the CB3 structure fold plus
//!   K11's string fold (both folds reuse `structure_finalize`, which
//!   min-folds on top of the header);
//! - K10's 32-bit offset cell, packed with `MJ_ERR_NUMBER` on the CPU;
//! - rejections from the number-fixup re-parse (overflow-to-infinity);
//! - rejections from the long-string-fixup re-parse (escape / control
//!   errors past the K11 threshold, at reference-exact offsets).
//!
//! On single-error documents this equals the reference verdict (code AND
//! offset, with the documented odd-quote offset exception). On multi-error
//! documents the GPU's globally-earliest-offset min can pick a different
//! error than the reference's stage-order walk — the documented
//! WHETHER-not-WHICH relaxation (see `crate::reference`'s error policy).
//!
//! # Rejection contract
//!
//! Any error short-circuits `Document` construction: [`GpuParse::Rejected`]
//! carries only the packed verdict and the tape/string buffers are dropped
//! unobserved. `decode_packed_error` maps the packed word to the public
//! [`Error`] enum (a completeness test pins every `MJ_ERR_*` code).
//!
//! # M5 perf contract (the former correctness-first deviations, resolved)
//!
//! - **No whole-buffer zero fills.** The tape buffer is *not* zero-filled:
//!   on accepted inputs every word is written by exactly one producer
//!   (containers/roots by K12/K13 or the CPU root write, numbers/literals
//!   by K10, strings by K11 — each token's footprint words belong to
//!   exactly one kernel), and rejected runs never observe the tape. The
//!   string buffer is not pre-zeroed either: K11 (and the long-string CPU
//!   valve) writes every record **and zero-fills each escape-shrunk slot's
//!   gap tail**, so the cost is proportional to escape shrinkage, never
//!   buffer size — gap bytes are deterministically zero on both backends
//!   (they are reachable through `StringBuffer::as_bytes`, so pooled
//!   previous-parse bytes must never survive there). The pool-poison tests
//!   in `tests/gpu_e2e.rs` pin both with whole-buffer equality.
//! - **No stage-1 snapshot.** `Stage2::run_to_lists` is lean: accepted runs
//!   hand over live GPU buffers, rejected runs a packed verdict; the
//!   test-runner `Vec` readbacks are rebuilt on demand outside this path.
//! - **Zero-copy `Document`.** [`GpuParseOutput`] hands the tape / string
//!   `GpuBuffer`s to the parser, which wraps them into GPU-backed
//!   [`TapeBuffer`](crate::tape::TapeBuffer) /
//!   [`StringBuffer`](crate::tape::StringBuffer) storage — no copy-out.
//! - **Pooled scratch.** [`GpuPipeline::run_pooled`] checks every buffer
//!   out of a [`ScratchPool`] and recycles the scratch before returning
//!   (the tape/string buffers return when the `Document` drops). Steady
//!   state does zero large allocations. All zero/init preconditions are
//!   explicit fills (`Stage1Buffers` chunk counts + header, the K10/K11
//!   counters below) — pooled contents are garbage by contract.
//! - **Zero-copy input.** [`GpuInput::External`] wraps caller-held
//!   page-aligned memory (`bytesNoCopy`); [`GpuInput::Bytes`] copies once
//!   into a pooled buffer; [`GpuInput::Pooled`] hands over a pool buffer
//!   the caller already filled (the `parse_file` read path — its one copy
//!   happens file→buffer, with no intermediate allocation).

use crate::error::{Error, Result, SyntaxErrorKind};
use crate::metal::{Dispatch, GpuBuffer, MetalContext, MjParams, THREADGROUP_SIZE};
use crate::pool::{Alloc, ScratchPool};
use crate::stage::{Stage, Stage1Buffers, Stage3Buffers, sort_passes};
use crate::tape::{make_final_root, make_root};

use super::numbers::{NO_NUMBER_ERROR, pack_number_error, patch_number_fixups};
use super::strings::patch_long_strings;
use super::stage2::{Stage2Accepted, Stage2Run};
use super::stage3::{Stage3, StructureDims};
use super::{
    ERR_DEPTH_LIMIT, ERR_EMPTY_INPUT, ERR_INVALID_LITERAL, ERR_MISSING_COLON, ERR_MISSING_COMMA,
    ERR_NUMBER, ERR_STRING, ERR_STRING_CONTROL, ERR_STRING_ESCAPE, ERR_TRAILING_CONTENT,
    ERR_UNBALANCED, ERR_UNEXPECTED_TOKEN, ERR_UNTERMINATED_STRING, ERR_UTF8,
};

/// `MJ_ERR_SYNTAX` — a reserved legacy placeholder in `shaders/common.h`
/// that no kernel produces. Mapped (to a generic `UnexpectedToken`) so the
/// error decoding is total over the `MjErrorCode` space.
const ERR_SYNTAX_RESERVED: u32 = 2;

/// How the input reaches the GPU.
///
/// Internal/unstable (exposed for the integration tests, like the rest of
/// [`crate::gpu`]); the supported surface is `Parser::parse` /
/// `Parser::parse_aligned` / `Parser::parse_file`.
#[derive(Debug)]
pub enum GpuInput<'a> {
    /// Plain bytes: copied **once** into a pooled, space-padded GPU buffer
    /// (the only copy on this path).
    Bytes(&'a [u8]),
    /// A caller-prepared zero-copy buffer (already wrapped via
    /// [`GpuBuffer::from_page_aligned`], typically over an
    /// [`AlignedInput`](crate::AlignedInput) or an mmap). `len` is the
    /// document length; the wrapped memory must satisfy
    /// `from_page_aligned`'s contract for the duration of the parse, plus
    /// the stage-1 padding invariant: bytes `len..len.next_multiple_of(64)`
    /// are ASCII spaces.
    External {
        /// The `bytesNoCopy` buffer over the caller's pages.
        buffer: GpuBuffer,
        /// Document length in bytes.
        len: usize,
    },
    /// A buffer checked out of **this parse's pool** that the caller
    /// already filled with the document bytes plus the stage-1 padding
    /// (bytes `len..len.next_multiple_of(64)` are ASCII spaces) — the
    /// `Parser::parse_file` read-into-pool path. Unlike
    /// [`External`](Self::External) it returns to the pool with the rest
    /// of the stage-1 scratch.
    Pooled {
        /// The checked-out, filled, space-padded buffer.
        buffer: GpuBuffer,
        /// Document length in bytes.
        len: usize,
    },
}

/// Where a full GPU parse ended.
#[derive(Debug)]
pub enum GpuParse {
    /// The input was rejected; the packed `(byte_offset << 32) | code`
    /// verdict (decode with `decode_packed_error` / the parser). Per the
    /// rejection contract the tape and string buffers are never observed.
    Rejected(u64),
    /// The input parsed; the finished buffers.
    Accepted(GpuParseOutput),
}

/// The finished buffers of an accepted GPU parse — everything `Document`
/// construction needs.
#[derive(Debug)]
pub struct GpuParseOutput {
    /// The complete tape: container/root words (K12/K13 or the CPU root
    /// write), number/literal entries (K10, fixup value words CPU-patched),
    /// string words (K11).
    pub tape: GpuBuffer,
    /// The string buffer (`[u32 LE len][content][NUL]` records at the
    /// raw-length prefix-sum offsets); `None` when the document has no
    /// strings (`stringbuf_total == 0`). Gap bytes (escape-shrunk slot
    /// tails) are zero-filled by K11 / the long-string valve, so the whole
    /// buffer is deterministic and byte-equal to the reference backend's.
    pub stringbuf: Option<GpuBuffer>,
    /// Token indices that took the K10 hard-case fixup path, sorted
    /// ascending (diagnostic; the value words are already patched).
    pub fixup_tokens: Vec<u32>,
    /// String-list indices that took the K11 long-string fixup path
    /// (`raw_len > LONG_STRING_THRESHOLD`), sorted ascending (diagnostic;
    /// the records and tape words are already patched).
    pub long_string_fixups: Vec<u32>,
}

/// K11's CB3 buffers (sizes known at CPU sync 2).
struct StringBuffers {
    record_offsets: GpuBuffer,
    stringbuf: GpuBuffer,
    chunk_error: GpuBuffer,
    /// Long-string fixup append counter (device atomic, armed with 0).
    long_count: GpuBuffer,
    /// The long-string fixup index list, sized from the string count
    /// (worst case every string is long) — acceptable because entries are
    /// index-sized u32s, 4 bytes per string, never content-sized.
    long_list: GpuBuffer,
    chunks: usize,
}

/// K10's CB3 buffers (sizes known at CPU sync 2).
struct NumberBuffers {
    /// 32-bit min-reduced failing byte offset, armed with
    /// [`NO_NUMBER_ERROR`].
    err_min_pos: GpuBuffer,
    /// Device atomic append counter, armed with 0.
    fixup_count: GpuBuffer,
    /// Worst case every scalar takes the fixup path.
    fixup_tokens: GpuBuffer,
}

/// The full GPU parse pipeline with lazily-built cached pipelines: the
/// composed [`Stage3`] (structure kernels + stages 1–2) plus the M4 scalar
/// kernels. Create once and reuse across parses.
#[derive(Debug)]
pub struct GpuPipeline {
    stage3: Stage3,
    parse_numbers: Stage,
    string_offsets: Stage,
    strings_unescape: Stage,
}

impl GpuPipeline {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            stage3: Stage3::new(),
            parse_numbers: Stage::new("parse_numbers"),
            string_offsets: Stage::new("string_record_offsets"),
            strings_unescape: Stage::new("strings_unescape"),
        }
    }

    /// Run the full pipeline over `input` with a throwaway buffer pool:
    /// CB1 → CB2 → CB2b → CB3 (structure + K10 + K11) → fixup patch. See
    /// the module docs for the command-buffer shape and the
    /// error/rejection contracts. Tests and one-shot callers use this;
    /// the parser drives [`run_pooled`](Self::run_pooled) with its
    /// long-lived pool.
    ///
    /// # Errors
    ///
    /// GPU plumbing failures only ([`Error::InputTooLarge`],
    /// [`Error::BufferAlloc`], pipeline/command-buffer errors). Input
    /// *content* problems are **data**: [`GpuParse::Rejected`].
    pub fn run(&self, ctx: &MetalContext, input: &[u8], max_depth: u32) -> Result<GpuParse> {
        self.run_pooled(ctx, &ScratchPool::new(), GpuInput::Bytes(input), max_depth)
    }

    /// [`run`](Self::run) over an explicit [`ScratchPool`] and input
    /// source. Every buffer is checked out of `pool`; the scratch returns
    /// to it before this call finishes, the finished tape/string buffers
    /// return when their owner (the parser's `Document`) drops them back.
    ///
    /// # Errors
    ///
    /// As [`run`](Self::run).
    pub fn run_pooled(
        &self,
        ctx: &MetalContext,
        pool: &ScratchPool,
        input: GpuInput<'_>,
        max_depth: u32,
    ) -> Result<GpuParse> {
        super::timing::begin_parse();
        let alloc = Alloc::Pool(pool);
        let t = super::timing::start();
        let mut bufs1 = match input {
            GpuInput::Bytes(bytes) => {
                let bufs = Stage1Buffers::new_in(ctx, alloc, bytes)?;
                super::timing::record("stage1 alloc + input copy", t, 0.0);
                bufs
            }
            GpuInput::External { buffer, len } => {
                let bufs = Stage1Buffers::with_external_input(ctx, alloc, buffer, len)?;
                super::timing::record("stage1 alloc (zero-copy input)", t, 0.0);
                bufs
            }
            GpuInput::Pooled { buffer, len } => {
                let bufs = Stage1Buffers::with_pooled_input(ctx, alloc, buffer, len)?;
                super::timing::record("stage1 alloc (pooled input)", t, 0.0);
                bufs
            }
        };

        let result = self.drive(ctx, pool, &mut bufs1, max_depth);

        let t = super::timing::start();
        bufs1.recycle(pool);
        super::timing::record("scratch recycle (stage1)", t, 0.0);
        result
    }

    /// The pipeline body over prepared stage-1 buffers (which the caller
    /// recycles). Internally checks all further scratch out of `pool` and
    /// returns it before finishing — on every rejection path too; only the
    /// rare `?` plumbing errors let buffers drop-free instead.
    fn drive(
        &self,
        ctx: &MetalContext,
        pool: &ScratchPool,
        bufs1: &mut Stage1Buffers,
        max_depth: u32,
    ) -> Result<GpuParse> {
        let alloc = Alloc::Pool(pool);
        // --- CB1 → CB2 → CB2b (stage 2 owns the first two syncs) -----------
        let Stage2Accepted {
            bufs2,
            header,
            gpu_seconds: _,
        } = match self.stage3.stage2().run_to_lists(ctx, alloc, bufs1)? {
            Stage2Run::Rejected(rejection) => {
                return Ok(GpuParse::Rejected(rejection.packed));
            }
            Stage2Run::Accepted(run) => *run,
        };

        let t = super::timing::start();
        let token_total = bufs2.token_total();
        let skeleton_total =
            usize::try_from(header.skeleton_total).expect("skeleton_total fits usize");
        let string_total = usize::try_from(header.string_total).expect("string_total fits usize");
        let scalar_total = usize::try_from(header.scalar_total).expect("scalar_total fits usize");
        let stringbuf_total =
            usize::try_from(header.stringbuf_total).expect("stringbuf_total fits usize");
        let tape_words =
            usize::try_from(header.tape_word_total).expect("tape_word_total fits usize") + 2;
        let input_len = bufs1.input_len() as u64;

        // --- CPU sync 2 exact allocations -----------------------------------
        // The tape, NOT zero-filled (pooled contents are garbage by
        // contract): accepted runs overwrite every word — each token's
        // footprint words belong to exactly one CB3 kernel, the root words
        // to K13 or the CPU write below — and rejected runs never observe
        // the tape. Pinned by the pool-poison test in tests/gpu_e2e.rs.
        let mut tape_buf = alloc.buffer(ctx, tape_words * size_of::<u64>())?;

        if skeleton_total == 0 {
            // Root scalar: no structure dispatches run (K13 included), so
            // the two root words are CPU-written into the shared buffer
            // (reference emit_tape semantics); K10/K11 fill the hole(s).
            let words = tape_buf.as_mut_slice::<u64>();
            words[0] = make_root(tape_words as u64 - 1);
            words[tape_words - 1] = make_final_root();
        }

        let mut bufs3 = if skeleton_total > 0 {
            self.stage3.assert_threadgroup_support(ctx)?;
            Some(Stage3Buffers::new_in(
                ctx,
                alloc,
                skeleton_total,
                sort_passes(max_depth),
            )?)
        } else {
            None
        };

        let mut string_bufs = if string_total > 0 {
            for stage in [&self.string_offsets, &self.strings_unescape] {
                let max = stage.pipeline(ctx)?.max_total_threads_per_threadgroup();
                assert!(
                    max >= THREADGROUP_SIZE,
                    "kernel `{}` supports only {max} threads/threadgroup (< {THREADGROUP_SIZE})",
                    stage.name()
                );
            }
            let chunks = string_total.div_ceil(THREADGROUP_SIZE);
            // The string buffer, NOT pre-zeroed (pooled contents are
            // garbage by contract): K11 writes every record's
            // `[u32 LE len][content][NUL]` bytes AND zero-fills each
            // escape-shrunk slot's gap tail (the pinned gap policy), so
            // every reachable byte of an accepted parse is written —
            // without an O(buffer) memset.
            let stringbuf = alloc.buffer(ctx, stringbuf_total)?;
            // The long-string append counter gets its precondition
            // established explicitly (pooled contents are garbage), like
            // K10's fixup counter below.
            let mut long_count = alloc.buffer(ctx, size_of::<u32>())?;
            long_count.as_mut_slice::<u32>()[0] = 0;
            Some(StringBuffers {
                record_offsets: alloc.buffer(ctx, string_total * size_of::<u64>())?,
                stringbuf,
                chunk_error: alloc.buffer(ctx, chunks * size_of::<u64>())?,
                long_count,
                long_list: alloc.buffer(ctx, string_total * size_of::<u32>())?,
                chunks,
            })
        } else {
            None
        };

        let mut number_bufs = if scalar_total > 0 {
            // Accumulation targets get their preconditions established
            // explicitly (pooled contents are garbage by contract).
            let mut err_min_pos = alloc.buffer(ctx, size_of::<u32>())?;
            err_min_pos.as_mut_slice::<u32>()[0] = NO_NUMBER_ERROR;
            let mut fixup_count = alloc.buffer(ctx, size_of::<u32>())?;
            fixup_count.as_mut_slice::<u32>()[0] = 0;
            Some(NumberBuffers {
                err_min_pos,
                fixup_count,
                fixup_tokens: alloc.buffer(ctx, scalar_total * size_of::<u32>())?,
            })
        } else {
            None
        };

        super::timing::record("sync2: tape/scratch alloc", t, 0.0);

        // --- CB3: structure + strings + numbers, one commit, one wait -------
        let t = super::timing::start();
        if bufs3.is_some() || string_bufs.is_some() || number_bufs.is_some() {
            let mut batch = ctx.batch()?;
            let h_tape = batch.bind_write(&mut tape_buf);
            let h_header = batch.bind_write(&mut bufs1.header);

            if let Some(bufs3) = bufs3.as_mut() {
                self.stage3.encode_structure(
                    &mut batch,
                    &bufs2,
                    bufs3,
                    h_tape,
                    h_header,
                    &StructureDims {
                        input_len,
                        max_depth,
                        tape_words: tape_words as u64,
                    },
                )?;
            }

            // Bindings shared by the scalar kernels.
            let h_input = batch.bind_read(&bufs1.input);
            let h_pos = batch.bind_read(bufs1.tok_pos.as_ref().expect("tokens allocated"));
            let h_tape_ofs = batch.bind_read(&bufs2.tape_ofs);

            if let Some(sb) = string_bufs.as_mut() {
                let h_kind = batch.bind_read(bufs1.tok_kind.as_ref().expect("tokens allocated"));
                let h_counts = batch.bind_read(&bufs2.chunk_counts);
                let h_sbytes = batch.bind_read(&bufs2.chunk_string_bytes);
                let h_strings =
                    batch.bind_read(bufs2.string_tokens.as_ref().expect("lists allocated"));
                let h_offsets = batch.bind_write(&mut sb.record_offsets);
                let h_sbuf = batch.bind_write(&mut sb.stringbuf);
                let h_serr = batch.bind_write(&mut sb.chunk_error);
                let h_lcount = batch.bind_write(&mut sb.long_count);
                let h_llist = batch.bind_write(&mut sb.long_list);

                let token_params = MjParams {
                    input_len,
                    element_count: token_total as u64,
                    ..Default::default()
                };
                self.string_offsets.encode(
                    &mut batch,
                    &[h_pos, h_kind, h_counts, h_sbytes, h_offsets],
                    Some(&token_params),
                    Dispatch::Threadgroups(bufs2.chunks()),
                )?;
                let string_params = MjParams {
                    input_len,
                    element_count: string_total as u64,
                    reserved0: tape_words as u64,  // defensive tape bound
                    reserved1: token_total as u64, // defensive token bound
                };
                self.strings_unescape.encode(
                    &mut batch,
                    &[
                        h_input, h_pos, h_strings, h_offsets, h_tape_ofs, h_sbuf, h_tape, h_serr,
                        h_lcount, h_llist,
                    ],
                    Some(&string_params),
                    Dispatch::Threadgroups(sb.chunks),
                )?;
                // K11's error fold reuses structure_finalize: min-fold the
                // per-chunk words on top of header.error.
                let fold_params = MjParams {
                    input_len,
                    element_count: sb.chunks as u64,
                    ..Default::default()
                };
                self.stage3.finalize_stage().encode(
                    &mut batch,
                    &[h_serr, h_header],
                    Some(&fold_params),
                    Dispatch::Threadgroups(1),
                )?;
            }

            if let Some(nb) = number_bufs.as_mut() {
                let h_scalars =
                    batch.bind_read(bufs2.scalar_tokens.as_ref().expect("lists allocated"));
                let h_nerr = batch.bind_write(&mut nb.err_min_pos);
                let h_ncount = batch.bind_write(&mut nb.fixup_count);
                let h_nfix = batch.bind_write(&mut nb.fixup_tokens);
                let number_params = MjParams {
                    input_len,
                    element_count: scalar_total as u64,
                    ..Default::default()
                };
                self.parse_numbers.encode(
                    &mut batch,
                    &[
                        h_input, h_scalars, h_pos, h_tape_ofs, h_tape, h_nerr, h_ncount, h_nfix,
                    ],
                    Some(&number_params),
                    Dispatch::Threads(scalar_total),
                )?;
            }

            let cb3_gpu = batch.commit_and_wait_timed()?;
            super::timing::record("cb3 (structure + strings + numbers)", t, cb3_gpu);
        } else {
            super::timing::record("cb3 (skipped: root scalar)", t, 0.0);
        }

        // --- CPU sync 3: merged verdict + fixup patches ----------------------
        let t = super::timing::start();
        let header = bufs1.read_header();
        let mut error: Option<u64> = header.first_error().map(|(o, c)| (o << 32) | u64::from(c));
        fn merge(packed: u64, error: &mut Option<u64>) {
            *error = Some(error.map_or(packed, |e| e.min(packed)));
        }

        // The CPU view of the input for the fixup re-parses: shared
        // storage, so this is the same memory the kernels read (the copied
        // bytes on the Bytes path, the caller's pages on the External
        // path) — no separate slice needs to ride along.
        let input = &bufs1.input.contents()[..input_len as usize];

        // The K11 long-string patch: re-run the flagged strings through the
        // shared reference unescaper into their precomputed slots of the
        // shared string buffer (+ their `"` tape words). Done even when an
        // error is already known — a long string can reject at an EARLIER
        // offset, and the merged verdict is the packed minimum.
        let mut long_string_fixups = Vec::new();
        if let Some(sb) = string_bufs.as_mut() {
            // Kernel appends at most once per thread.
            let total = (sb.long_count.as_slice::<u32>()[0] as usize).min(string_total);
            long_string_fixups = sb.long_list.as_slice::<u32>()[..total].to_vec();
            long_string_fixups.sort_unstable();
            if let Some(patch_error) = patch_long_strings(
                input,
                bufs1
                    .tok_pos
                    .as_ref()
                    .expect("tokens allocated")
                    .as_slice::<u32>(),
                bufs2
                    .string_tokens
                    .as_ref()
                    .expect("lists allocated")
                    .as_slice::<u32>(),
                sb.record_offsets.as_slice::<u64>(),
                bufs2.tape_ofs.as_slice::<u32>(),
                &long_string_fixups,
                sb.stringbuf.contents_mut(),
                tape_buf.as_mut_slice::<u64>(),
            ) {
                merge(patch_error, &mut error);
            }
        }

        let mut fixup_tokens = Vec::new();
        if let Some(nb) = &mut number_bufs {
            let err_pos = nb.err_min_pos.as_slice::<u32>()[0];
            if err_pos != NO_NUMBER_ERROR {
                merge(pack_number_error(u64::from(err_pos)), &mut error);
            }
            // Kernel appends at most once per thread.
            let total = (nb.fixup_count.as_slice::<u32>()[0] as usize).min(scalar_total);
            fixup_tokens = nb.fixup_tokens.as_slice::<u32>()[..total].to_vec();
            fixup_tokens.sort_unstable();

            // Patch the 8-byte value words in the SHARED tape buffer in
            // place. Done even when an error is already known: a fixup
            // re-parse can reject at an EARLIER offset, and the merged
            // verdict is the packed minimum.
            let tok_pos = bufs1.tok_pos.as_ref().expect("tokens allocated");
            if let Some(patch_error) = patch_number_fixups(
                input,
                tok_pos.as_slice::<u32>(),
                bufs2.tape_ofs.as_slice::<u32>(),
                &fixup_tokens,
                tape_buf.as_mut_slice::<u64>(),
            ) {
                merge(patch_error, &mut error);
            }
        }

        super::timing::record("sync3: verdict + fixup patches", t, 0.0);

        // --- scratch back to the pool ----------------------------------------
        let t = super::timing::start();
        bufs2.recycle(pool);
        if let Some(bufs3) = bufs3 {
            bufs3.recycle(pool);
        }
        let stringbuf = string_bufs.map(|sb| {
            let StringBuffers {
                record_offsets,
                stringbuf,
                chunk_error,
                long_count,
                long_list,
                chunks: _,
            } = sb;
            for buf in [record_offsets, chunk_error, long_count, long_list] {
                pool.put_back(buf);
            }
            stringbuf
        });
        if let Some(nb) = number_bufs {
            let NumberBuffers {
                err_min_pos,
                fixup_count,
                fixup_tokens,
            } = nb;
            for buf in [err_min_pos, fixup_count, fixup_tokens] {
                pool.put_back(buf);
            }
        }
        super::timing::record("scratch recycle (stages 2-3)", t, 0.0);

        if let Some(packed) = error {
            // Rejection contract: the tape/string buffers are never
            // observed — straight back to the pool, Document construction
            // is short-circuited.
            pool.put_back(tape_buf);
            if let Some(buf) = stringbuf {
                pool.put_back(buf);
            }
            return Ok(GpuParse::Rejected(packed));
        }
        Ok(GpuParse::Accepted(GpuParseOutput {
            tape: tape_buf,
            stringbuf,
            fixup_tokens,
            long_string_fixups,
        }))
    }
}

impl Default for GpuPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Map a packed GPU error word (`(byte_offset << 32) | MjErrorCode`) to the
/// public [`Error`] enum. Total over the `MjErrorCode` space (every
/// `MJ_ERR_*` code in `shaders/common.h` + the K11 codes in
/// `shaders/13_strings.metal` — a completeness test pins the mapping);
/// unknown codes mean the GPU corrupted its own header and surface as the
/// internal [`Error::CommandBuffer`].
///
/// `max_depth` is the limit the parse ran with (the `DepthLimit` variant
/// reports it).
pub(crate) fn decode_packed_error(packed: u64, max_depth: u32) -> Error {
    let offset = packed >> 32;
    let code = packed as u32;
    let syntax = |kind: SyntaxErrorKind| Error::Syntax { offset, kind };
    match code {
        ERR_UTF8 => Error::Utf8 { offset },
        // Reserved legacy placeholder — no kernel produces it; mapped for
        // totality.
        ERR_SYNTAX_RESERVED => syntax(SyntaxErrorKind::UnexpectedToken),
        ERR_DEPTH_LIMIT => Error::DepthLimit {
            offset,
            limit: max_depth,
        },
        ERR_TRAILING_CONTENT => Error::TrailingContent { offset },
        ERR_NUMBER => syntax(SyntaxErrorKind::InvalidNumber),
        // CB1's odd-quote-total verdict; same class as the Layer-1 code,
        // reported at the documented provisional offset (input_len).
        ERR_STRING => syntax(SyntaxErrorKind::UnterminatedString),
        ERR_MISSING_COLON => syntax(SyntaxErrorKind::MissingColon),
        ERR_MISSING_COMMA => syntax(SyntaxErrorKind::MissingComma),
        ERR_UNEXPECTED_TOKEN => syntax(SyntaxErrorKind::UnexpectedToken),
        ERR_INVALID_LITERAL => syntax(SyntaxErrorKind::InvalidLiteral),
        ERR_UNBALANCED => syntax(SyntaxErrorKind::UnbalancedBrackets),
        ERR_UNTERMINATED_STRING => syntax(SyntaxErrorKind::UnterminatedString),
        ERR_EMPTY_INPUT => syntax(SyntaxErrorKind::EmptyInput),
        ERR_STRING_ESCAPE => syntax(SyntaxErrorKind::InvalidStringEscape),
        ERR_STRING_CONTROL => syntax(SyntaxErrorKind::ControlCharacterInString),
        other => Error::CommandBuffer {
            message: format!("GPU reported unknown error code {other} at byte {offset}"),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::{
        make_close, make_double_marker, make_false, make_int64_marker, make_null, make_open,
        make_string, make_true,
    };

    /// GPU gating, as in stage1/2/3: skip without a device unless
    /// `METAL_JSON_REQUIRE_GPU=1` makes that a hard failure.
    fn gpu_or_skip(test: &str) -> Option<(MetalContext, GpuPipeline)> {
        match MetalContext::new() {
            Ok(ctx) => Some((ctx, GpuPipeline::new())),
            Err(err) => {
                if std::env::var_os("METAL_JSON_REQUIRE_GPU").is_some_and(|v| v == "1") {
                    panic!("METAL_JSON_REQUIRE_GPU=1 but no usable Metal device: {err}");
                }
                eprintln!("SKIP {test}: no usable Metal device here ({err})");
                None
            }
        }
    }

    fn run(pipeline: &GpuPipeline, ctx: &MetalContext, input: &[u8]) -> GpuParse {
        pipeline
            .run(ctx, input, crate::parser::DEFAULT_MAX_DEPTH)
            .unwrap_or_else(|e| {
                panic!(
                    "GPU pipeline failed on {:?}: {e}",
                    String::from_utf8_lossy(&input[..input.len().min(60)])
                )
            })
    }

    fn accept(pipeline: &GpuPipeline, ctx: &MetalContext, input: &[u8]) -> GpuParseOutput {
        match run(pipeline, ctx, input) {
            GpuParse::Accepted(out) => out,
            GpuParse::Rejected(packed) => panic!(
                "{:?} must parse, got packed error {:?}",
                String::from_utf8_lossy(input),
                (packed >> 32, packed as u32)
            ),
        }
    }

    fn reject(pipeline: &GpuPipeline, ctx: &MetalContext, input: &[u8]) -> (u64, u32) {
        match run(pipeline, ctx, input) {
            GpuParse::Rejected(packed) => (packed >> 32, packed as u32),
            GpuParse::Accepted(_) => panic!(
                "{:?} must be rejected",
                String::from_utf8_lossy(input)
            ),
        }
    }

    /// THE tape pin: the docs/tape-format.md worked example with EVERY hole
    /// filled — container/root words (K12/K13), number entries (K10) and
    /// string words + records (K11) in one CB3.
    #[test]
    fn worked_example_full_tape_and_stringbuf() {
        let Some((ctx, pipeline)) = gpu_or_skip("worked_example_full_tape_and_stringbuf") else {
            return;
        };
        let out = accept(&pipeline, &ctx, br#"{"a":[1,2.5],"b":"x\n"}"#);
        let expected: [u64; 13] = [
            make_root(12),
            make_open(b'{', 12, 2),
            make_string(0),
            make_open(b'[', 9, 2),
            make_int64_marker(),
            1,
            make_double_marker(),
            2.5f64.to_bits(),
            make_close(b']', 3),
            make_string(6),
            make_string(12),
            make_close(b'}', 1),
            make_final_root(),
        ];
        assert_eq!(out.tape.as_slice::<u64>(), expected);
        let sb = out.stringbuf.expect("three strings");
        let bytes = sb.contents();
        assert_eq!(bytes.len(), 20);
        assert_eq!(&bytes[0..6], &[1, 0, 0, 0, b'a', 0]);
        assert_eq!(&bytes[6..12], &[1, 0, 0, 0, b'b', 0]);
        assert_eq!(&bytes[12..19], &[2, 0, 0, 0, b'x', 0x0A, 0]);
        // bytes[19] is the gap byte the `\n` escape shrank away —
        // zero-filled by K11's careful path (the pinned gap policy).
        assert_eq!(bytes[19], 0);
        assert!(out.fixup_tokens.is_empty());
    }

    /// Root scalars: the structure block is skipped (no skeleton), the
    /// root words are CPU-written, and K10/K11 still fill the hole(s).
    #[test]
    fn root_scalars_produce_complete_tapes() {
        let Some((ctx, pipeline)) = gpu_or_skip("root_scalars_produce_complete_tapes") else {
            return;
        };
        let out = accept(&pipeline, &ctx, b"42");
        assert_eq!(
            out.tape.as_slice::<u64>(),
            [make_root(3), make_int64_marker(), 42, make_final_root()]
        );
        assert!(out.stringbuf.is_none());

        let out = accept(&pipeline, &ctx, b"true");
        assert_eq!(
            out.tape.as_slice::<u64>(),
            [make_root(2), make_true(), make_final_root()]
        );

        let out = accept(&pipeline, &ctx, b" null \n");
        assert_eq!(
            out.tape.as_slice::<u64>(),
            [make_root(2), make_null(), make_final_root()]
        );

        let out = accept(&pipeline, &ctx, br#""x""#);
        assert_eq!(
            out.tape.as_slice::<u64>(),
            [make_root(2), make_string(0), make_final_root()]
        );
        let sb = out.stringbuf.expect("root string");
        assert_eq!(sb.contents(), &[1, 0, 0, 0, b'x', 0]);

        let out = accept(&pipeline, &ctx, b"-0.0");
        assert_eq!(
            out.tape.as_slice::<u64>(),
            [
                make_root(3),
                make_double_marker(),
                (-0.0f64).to_bits(),
                make_final_root()
            ]
        );

        let out = accept(&pipeline, &ctx, b"[true,false,null]");
        assert_eq!(
            out.tape.as_slice::<u64>(),
            [
                make_root(6),
                make_open(b'[', 6, 3),
                make_true(),
                make_false(),
                make_null(),
                make_close(b']', 1),
                make_final_root()
            ]
        );
    }

    /// One rejection per error class the merged CB3 verdict can carry,
    /// with the reference's (offset, code) — the M4 error contract.
    #[test]
    fn every_error_class_rejects_with_reference_offset_and_code() {
        let Some((ctx, pipeline)) =
            gpu_or_skip("every_error_class_rejects_with_reference_offset_and_code")
        else {
            return;
        };
        let cases: &[(&[u8], u64, u32)] = &[
            (b"ab\x80", 2, ERR_UTF8),                       // CB1 UTF-8
            (b"\"abc", 4, ERR_STRING),                      // CB1 odd quotes (offset = input_len)
            (b"", 0, ERR_EMPTY_INPUT),                      // CPU verdict
            (b" \t\n", 0, ERR_EMPTY_INPUT),                 // whitespace-only
            (b"[1 true]", 3, ERR_MISSING_COMMA),            // CB2 Layer 1
            (b"]", 0, ERR_UNEXPECTED_TOKEN),                // CB2 Layer 1
            (b"nan", 0, ERR_INVALID_LITERAL),               // CB2 Layer 1
            (br#"{"a" "b"}"#, 5, ERR_MISSING_COLON),        // CB2 Layer 1
            (b"[1", 0, ERR_UNBALANCED),                     // CB3 structure
            (b"{},1", 2, ERR_TRAILING_CONTENT),             // CB3 structure
            (b"[01]", 1, ERR_NUMBER),                       // K10
            (br#"{"k":1e999}"#, 5, ERR_NUMBER),             // K10 overflow
            (br#"["\q"]"#, 2, ERR_STRING_ESCAPE),           // K11
            (b"[\"a\x01b\"]", 3, ERR_STRING_CONTROL),       // K11
        ];
        for &(input, offset, code) in cases {
            assert_eq!(
                reject(&pipeline, &ctx, input),
                (offset, code),
                "{:?}",
                String::from_utf8_lossy(input)
            );
        }
        // DepthLimit (CB3 structure) at a custom limit.
        let GpuParse::Rejected(packed) = pipeline.run(&ctx, b"[[[[]]]]", 3).unwrap() else {
            panic!("depth 4 at limit 3 must reject");
        };
        assert_eq!((packed >> 32, packed as u32), (3, ERR_DEPTH_LIMIT));
    }

    /// Scalar errors merge by packed minimum across K10 and K11: the
    /// earliest byte offset wins regardless of which kernel found it.
    #[test]
    fn scalar_error_merge_is_earliest_offset_first() {
        let Some((ctx, pipeline)) = gpu_or_skip("scalar_error_merge_is_earliest_offset_first")
        else {
            return;
        };
        // String escape at 6 beats the number error at 14.
        assert_eq!(
            reject(&pipeline, &ctx, br#"{"a":"\q","b":01}"#),
            (6, ERR_STRING_ESCAPE)
        );
        // Number error at 1 beats the string escape at 5.
        assert_eq!(
            reject(&pipeline, &ctx, br#"[01,"\q"]"#),
            (1, ERR_NUMBER)
        );
    }

    /// The packed-code → public-Error mapping is total over the MjErrorCode
    /// space: every `MJ_ERR_*` constant in shaders/common.h and
    /// shaders/13_strings.metal decodes to a real (non-internal) variant,
    /// and the per-code variants are pinned. Parses the shader sources,
    /// like the layout-lock tests.
    #[test]
    fn error_code_mapping_is_complete_and_pinned() {
        let mut codes: Vec<(String, u32)> = Vec::new();
        for src in [
            include_str!("../../shaders/common.h"),
            include_str!("../../shaders/13_strings.metal"),
        ] {
            for line in src.lines() {
                let line = line.trim();
                let Some(rest) = line
                    .strip_prefix("MJ_ERR_")
                    .or_else(|| line.strip_prefix("constant constexpr uint MJ_ERR_"))
                else {
                    continue;
                };
                let Some((name, rest)) = rest.split_once('=') else {
                    continue;
                };
                let digits: String = rest
                    .trim_start()
                    .chars()
                    .take_while(char::is_ascii_digit)
                    .collect();
                let value: u32 = digits.parse().expect("MJ_ERR_ value parses");
                codes.push((format!("MJ_ERR_{}", name.trim()), value));
            }
        }
        assert_eq!(codes.len(), 15, "13 common.h codes + 2 K11 codes: {codes:?}");

        for (name, code) in &codes {
            let err = decode_packed_error((77 << 32) | u64::from(*code), 1024);
            assert!(
                !matches!(err, Error::CommandBuffer { .. }),
                "{name} ({code}) must map to a public error, got {err:?}"
            );
        }
        // Unknown codes are internal corruption.
        assert!(matches!(
            decode_packed_error((77 << 32) | 7, 1024),
            Error::CommandBuffer { .. }
        ));

        // Pin the variant per code.
        let kind = |code: u32| match decode_packed_error((9 << 32) | u64::from(code), 1024) {
            Error::Syntax { offset: 9, kind } => kind,
            other => panic!("code {code}: expected Syntax, got {other:?}"),
        };
        assert!(matches!(
            decode_packed_error((9 << 32) | u64::from(ERR_UTF8), 1024),
            Error::Utf8 { offset: 9 }
        ));
        assert!(matches!(
            decode_packed_error((9 << 32) | u64::from(ERR_DEPTH_LIMIT), 64),
            Error::DepthLimit {
                offset: 9,
                limit: 64
            }
        ));
        assert!(matches!(
            decode_packed_error((9 << 32) | u64::from(ERR_TRAILING_CONTENT), 1024),
            Error::TrailingContent { offset: 9 }
        ));
        assert_eq!(kind(ERR_SYNTAX_RESERVED), SyntaxErrorKind::UnexpectedToken);
        assert_eq!(kind(ERR_NUMBER), SyntaxErrorKind::InvalidNumber);
        assert_eq!(kind(ERR_STRING), SyntaxErrorKind::UnterminatedString);
        assert_eq!(kind(ERR_MISSING_COLON), SyntaxErrorKind::MissingColon);
        assert_eq!(kind(ERR_MISSING_COMMA), SyntaxErrorKind::MissingComma);
        assert_eq!(kind(ERR_UNEXPECTED_TOKEN), SyntaxErrorKind::UnexpectedToken);
        assert_eq!(kind(ERR_INVALID_LITERAL), SyntaxErrorKind::InvalidLiteral);
        assert_eq!(kind(ERR_UNBALANCED), SyntaxErrorKind::UnbalancedBrackets);
        assert_eq!(
            kind(ERR_UNTERMINATED_STRING),
            SyntaxErrorKind::UnterminatedString
        );
        assert_eq!(kind(ERR_EMPTY_INPUT), SyntaxErrorKind::EmptyInput);
        assert_eq!(kind(ERR_STRING_ESCAPE), SyntaxErrorKind::InvalidStringEscape);
        assert_eq!(
            kind(ERR_STRING_CONTROL),
            SyntaxErrorKind::ControlCharacterInString
        );
    }

    /// Halfway-point literals (≥ 20 truncated digits) inside a full
    /// document take the fixup path and come out CPU-patched, bit-exact.
    #[test]
    fn fixup_numbers_inside_full_documents_are_patched() {
        let Some((ctx, pipeline)) =
            gpu_or_skip("fixup_numbers_inside_full_documents_are_patched")
        else {
            return;
        };
        // halfway(1.0, 1.0 + ulp): 54 digits, ties to even -> 1.0 exactly.
        let halfway = "1.00000000000000011102230246251565404236316680908203125";
        let oracle: f64 = halfway.parse().unwrap();
        assert_eq!(oracle.to_bits(), 1.0f64.to_bits(), "fixture sanity");

        let json = format!(r#"{{"k":[{halfway},2],"s":"v"}}"#);
        let out = accept(&pipeline, &ctx, json.as_bytes());
        assert!(
            !out.fixup_tokens.is_empty(),
            "a truncated halfway literal must take the fixup path"
        );
        let tape = out.tape.as_slice::<u64>();
        // r { "k [ d <bits> l 2 ] "v } r
        assert_eq!(tape[4], make_double_marker());
        assert_eq!(tape[5], oracle.to_bits(), "CPU-patched value word");
        assert_eq!(tape[6], make_int64_marker());
        assert_eq!(tape[7], 2);
    }
}
