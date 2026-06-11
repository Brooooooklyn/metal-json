//! K11 on the GPU: string validation + unescaping — the M4 string stage,
//! run standalone over the CB2 lists (`shaders/13_strings.metal`).
//!
//! # Command-buffer shape (standalone M4 runner)
//!
//! ```text
//! CB1 → CPU sync 1 → CB2 (K5 K6 K7) → CPU sync 2 → CB2b (K6b)
//!                                  (see crate::gpu::stage2)
//!   ── CPU sync: string_total / stringbuf_total / tape_word_total are
//!      known, so the record-offset list, the string buffer and the tape
//!      are exact-allocated HERE ──
//! CBs: string_record_offsets  1 threadgroup / 1024-token chunk: per-string
//!                             record offsets — the K7 string-byte chunk
//!                             carries refined by an in-chunk scan into the
//!                             exclusive `raw_len + 5` prefix sum
//!                             (docs/tape-format.md's pinned offset policy)
//!      strings_unescape       K11: 1 thread / string-list entry — fast
//!                             16-byte-block path for escape-free spans,
//!                             sequential escape path (`\" \\ \/ \b \f \n
//!                             \r \t`, `\uXXXX` incl. surrogate pairs);
//!                             writes the [u32 LE len][content][NUL] record
//!                             and the `"` tape word at tape_ofs[token];
//!                             per-chunk min-reduced error words
//!      structure_finalize     (reused from CB3) error fold → header
//!   ── commit, wait: CPU sync reads the header. A string error REJECTS
//!      the input: the record/tape outputs are never produced (the stage-2
//!      outputs are kept — stage 2 accepted the input) ──
//! ```
//!
//! The bit-exact spec is `reference::stage6_strings`
//! (src/reference/strings.rs) plus, for the record/tape bytes, the string
//! arm of `reference::emit_tape`; the in-module tests diff the two
//! backends on identical inputs.
//!
//! This is a **standalone kernel runner** (the M4 deliverable for K11):
//! it composes [`Stage2`] — not [`Stage3`](super::Stage3) — because string
//! unescaping is a pure token-level stage, exactly like the reference's
//! `stage6_strings(tokens, input)`. The production composition,
//! [`crate::gpu::pipeline::GpuPipeline`], encodes the same dispatches into
//! the full pipeline's CB3 next to K10 and the structure kernels; nothing
//! differs beyond where the dispatches are encoded (and that the pipeline
//! zero-fills the string buffer instead of poisoning it — gap bytes stay
//! contractually unspecified either way).
//!
//! # The pinned gap policy (and why the runner poisons the string buffer)
//!
//! Record offsets are allocated by RAW length (`raw_len + 5` slots), so a
//! record whose escapes shrank it leaves a gap before the next slot. Gap
//! bytes are **unspecified** in GPU output (the kernel never writes them);
//! tests therefore compare tape words and per-record bytes
//! (`[u32 LE len][content][NUL]` at each offset) — never raw gap bytes.
//! To keep that contract honest, this runner pre-fills the string buffer
//! with a poison byte: a kernel that forgot to write a length byte, a
//! content byte or the NUL shows up as poison instead of being masked by
//! conveniently-zero fresh pages. (The CPU reference zero-fills gaps
//! instead — its output is deterministic by design.)
//!
//! # The long-string valve (the K10 fixup pattern, applied to K11)
//!
//! K11 is thread-per-string: without a valve one multi-MB string would
//! serialize the whole parse on a single lane (latency, potential
//! command-buffer timeout). Strings with `raw_len >`
//! [`LONG_STRING_THRESHOLD`] are therefore not unescaped on the GPU: the
//! kernel appends their string-list index to a long-string fixup list via
//! a device atomic counter, and after the command buffer completes
//! [`patch_long_strings`] re-runs exactly those strings through the shared
//! reference unescaper (`crate::unescape`), writing each record into the
//! string buffer at its precomputed offset and the `"` tape word at
//! `tape_ofs[token]` — composable on the CPU because record offsets are
//! pure token-position math (`raw_len + 5` prefix sums), independent of
//! content. Unescape errors from flagged strings merge into the packed
//! first-error fold exactly like fixup-driven number rejections.
//!
//! # Error contract
//!
//! After K11 the GPU catches both reference stage-6 error classes, with
//! the reference's exact offset (the leftmost bad byte of the first bad
//! string — per-thread left-to-right walks + the packed-min fold reproduce
//! the document-order-first verdict, since string extents are disjoint):
//!
//! - [`ERR_STRING_ESCAPE`] — `SyntaxErrorKind::InvalidStringEscape`, at
//!   the backslash (bad designator, bad/short hex, lone or inverted
//!   surrogates);
//! - [`ERR_STRING_CONTROL`] — `SyntaxErrorKind::ControlCharacterInString`,
//!   at the raw control byte (< 0x20).

use crate::error::{Error, Result, SyntaxErrorKind};
use crate::metal::{Dispatch, GpuBuffer, MetalContext, MjParams, THREADGROUP_SIZE};
use crate::stage::{Stage, Stage1Buffers};
use crate::tape::{STRING_RECORD_HEADER_BYTES, make_string};

use super::stage2::{Stage2, Stage2Accepted, Stage2Output, Stage2Run};

/// `MjErrorCode` value for `SyntaxErrorKind::InvalidStringEscape`. Mirrors
/// `MJ_ERR_STRING_ESCAPE` in `shaders/13_strings.metal` — keep in sync (a
/// test parses the shader and pins both constants). The M4 string codes
/// extend the `MjErrorCode` space past `MJ_ERR_EMPTY_INPUT` (22); they are
/// defined in the kernel file rather than `shaders/common.h` so the M4
/// scalar kernels land independently (fold into the enum at parser
/// integration). No same-offset tie-break constraint exists: one byte
/// offset names either a backslash or a control byte, never both, and K11
/// offsets never collide with other kernels' (disjoint token extents,
/// rejection contract).
pub const ERR_STRING_ESCAPE: u32 = 23;
/// `MjErrorCode` value for `SyntaxErrorKind::ControlCharacterInString`.
/// See [`ERR_STRING_ESCAPE`].
pub const ERR_STRING_CONTROL: u32 = 24;

/// The long-string valve threshold, in raw bytes between the quotes:
/// strings with `raw_len` STRICTLY ABOVE this are deferred to the CPU
/// patch pass ([`patch_long_strings`]) instead of being walked by one GPU
/// thread. 16384 (one 16 KiB page) is long enough that real-world strings
/// almost never cross it — the GPU keeps the whole hot path — yet short
/// enough that a single K11 lane never owns more than one page-sized walk
/// (never megabytes), bounding the kernel's serial tail. Mirrors
/// `MJ_LONG_STRING_THRESHOLD` in `shaders/13_strings.metal` — keep in sync
/// (a test parses the shader and pins both).
pub const LONG_STRING_THRESHOLD: u32 = 16384;

/// Pre-fill byte for the GPU string buffer (see the module docs: gap bytes
/// are unspecified, and poison keeps "kernel forgot to write" failures
/// from hiding behind zeroed fresh pages).
const STRINGBUF_POISON: u8 = 0xA5;

/// Everything the standalone K11 runner produces, copied back into plain
/// `Vec`s for test ergonomics, mirroring [`Stage2Output`] /
/// [`Stage3Output`](super::Stage3Output).
///
/// # Rejection contract
///
/// When [`error`](Self::error) is `Some`, the pipeline has rejected the
/// input and outputs after the failing stage are never produced:
///
/// - a stage-1 / stage-2 rejection leaves [`stage2`](Self::stage2) with
///   its own rejection contract applied and every string output empty;
/// - a K11 string error (escape / control character) keeps the stage-2
///   outputs — stage 2 accepted the input — but the
///   [`record_offsets`](Self::record_offsets) / [`stringbuf`](Self::stringbuf)
///   / [`tape`](Self::tape) outputs stay empty (the tape is never
///   observed — Document construction is short-circuited).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StringsOutput {
    /// The stage-2 view of the same run (stage-1 view nested inside).
    pub stage2: Stage2Output,
    /// Byte offset of each string's record in [`stringbuf`](Self::stringbuf),
    /// in document order (entry `s` describes `stage2.string_tokens[s]`).
    /// The exclusive prefix sum of `raw_len + 5` — bit-identical to
    /// reference `UnescapedString::record_offset`.
    pub record_offsets: Vec<u64>,
    /// The string buffer: a `[u32 LE len][content][NUL]` record at every
    /// offset in [`record_offsets`](Self::record_offsets). Gap bytes are
    /// UNSPECIFIED (this runner poisons them with `0xA5`); read records
    /// only through the offsets.
    pub stringbuf: Vec<u8>,
    /// The tape, `tape_word_total + 2` words: `"` words written by K11 at
    /// `tape_ofs[token]` for every string token, zero-word holes
    /// everywhere else (container/root/scalar words belong to K12/K13/K10,
    /// which this standalone runner does not dispatch).
    pub tape: Vec<u64>,
    /// String-list indices that took the long-string fixup path
    /// (`raw_len > LONG_STRING_THRESHOLD`), sorted ascending (the GPU
    /// appends in nondeterministic order; the runner sorts). Reported on
    /// rejected runs too (diagnostic: which strings the CPU re-ran),
    /// mirroring `NumbersOutput::fixup_tokens`.
    pub long_string_fixups: Vec<u32>,
    /// First error, packed `(byte_offset << 32) | code`, or `None`. Codes:
    /// everything stage 2 can report, plus [`ERR_STRING_ESCAPE`] and
    /// [`ERR_STRING_CONTROL`].
    pub error: Option<u64>,
}

impl StringsOutput {
    /// Decode [`error`](Self::error) as `(byte_offset, code)`.
    #[must_use]
    pub fn error_offset_code(&self) -> Option<(u64, u32)> {
        self.error.map(|e| (e >> 32, e as u32))
    }

    /// Content bytes of record `s`, read through its offset exactly like a
    /// consumer would: `[u32 LE len][content][NUL]`. Asserts the record is
    /// well-formed (in-bounds, NUL-terminated) — a test accessor.
    ///
    /// # Panics
    ///
    /// On a malformed record or out-of-range `s`.
    #[must_use]
    pub fn record_content(&self, s: usize) -> &[u8] {
        let offset = usize::try_from(self.record_offsets[s]).expect("offset fits usize");
        let header: [u8; STRING_RECORD_HEADER_BYTES] = self.stringbuf
            [offset..offset + STRING_RECORD_HEADER_BYTES]
            .try_into()
            .expect("4 header bytes");
        let len = u32::from_le_bytes(header) as usize;
        let content_start = offset + STRING_RECORD_HEADER_BYTES;
        let content = &self.stringbuf[content_start..content_start + len];
        assert_eq!(
            self.stringbuf[content_start + len],
            0,
            "record {s} must be NUL-terminated"
        );
        content
    }

    /// A rejected output: `stage2` as given, every string output empty.
    fn rejected(stage2: Stage2Output, packed_error: u64) -> Self {
        Self {
            stage2,
            error: Some(packed_error),
            ..Self::default()
        }
    }
}

/// The K11 kernels plus the composed [`Stage2`] (which composes
/// [`Stage1`](super::Stage1)), with lazily-built cached pipelines. Create
/// once and reuse across parses.
#[derive(Debug)]
pub struct StringsStage {
    stage2: Stage2,
    offsets: Stage,
    unescape: Stage,
    /// The CB3 error fold (`shaders/10_pair_ctx.metal`), reused verbatim:
    /// its contract — min-fold a `ulong` chunk-error array into
    /// `header.error` from one threadgroup — is exactly what K11's
    /// per-chunk error words need.
    finalize: Stage,
}

impl StringsStage {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            stage2: Stage2::new(),
            offsets: Stage::new("string_record_offsets"),
            unescape: Stage::new("strings_unescape"),
            finalize: Stage::new("structure_finalize"),
        }
    }

    /// Run the pipeline through K11 (CB1 → CB2 → CB2b → the string command
    /// buffer) over `input` on freshly allocated buffers and read the
    /// results back. See the module docs for the command-buffer shape and
    /// [`StringsOutput`] for the rejection contract.
    ///
    /// # Errors
    ///
    /// GPU plumbing failures only; input *content* problems are **data**,
    /// reported in [`StringsOutput::error`].
    pub fn run(&self, ctx: &MetalContext, input: &[u8]) -> Result<StringsOutput> {
        let mut bufs1 = Stage1Buffers::new(ctx, input)?;
        self.run_with_buffers(ctx, &mut bufs1)
    }

    /// [`run`](Self::run) over caller-prepared stage-1 buffers (which must
    /// satisfy the [`Stage1Buffers`] zero/init preconditions).
    ///
    /// # Errors
    ///
    /// As [`run`](Self::run).
    pub fn run_with_buffers(
        &self,
        ctx: &MetalContext,
        bufs1: &mut Stage1Buffers,
    ) -> Result<StringsOutput> {
        // --- CB1 → CB2 → CB2b (stage 2 owns the first two syncs) -----------
        let Stage2Accepted {
            stage1,
            bufs2,
            header,
            gpu_seconds: _,
        } = match self.stage2.run_to_lists(ctx, bufs1)? {
            Stage2Run::Rejected(out) => {
                let packed = out.error.expect("rejected runs carry an error");
                return Ok(StringsOutput::rejected(*out, packed));
            }
            Stage2Run::Accepted(run) => *run,
        };
        let stage2_out = Stage2::collect_outputs(stage1, &bufs2, &header);

        let token_total = bufs2.token_total();
        let string_total =
            usize::try_from(header.string_total).expect("string_total fits usize");
        let stringbuf_total =
            usize::try_from(header.stringbuf_total).expect("stringbuf_total fits usize");
        let tape_words =
            usize::try_from(header.tape_word_total).expect("tape_word_total fits usize") + 2;

        // The tape, zero-filled: the hole convention (every non-string word
        // stays a 0 hole in this standalone runner — K10/K12/K13 own them).
        let mut tape_buf = GpuBuffer::alloc(ctx, tape_words * size_of::<u64>())?;
        tape_buf.contents_mut().fill(0);

        if string_total == 0 {
            // Nothing to dispatch; stringbuf_total is 0 by construction.
            return Ok(StringsOutput {
                stage2: stage2_out,
                tape: tape_buf.as_slice::<u64>().to_vec(),
                ..StringsOutput::default()
            });
        }

        // The cooperative kernels are written for full 256-thread groups
        // (same invariant the stage-1/2/3 orchestrations assert).
        for stage in [&self.offsets, &self.unescape, &self.finalize] {
            let max = stage.pipeline(ctx)?.max_total_threads_per_threadgroup();
            assert!(
                max >= THREADGROUP_SIZE,
                "kernel `{}` supports only {max} threads/threadgroup (< {THREADGROUP_SIZE})",
                stage.name()
            );
        }

        // Exact-size allocations from the K7 totals (CPU sync 2).
        let mut record_offsets = GpuBuffer::alloc(ctx, string_total * size_of::<u64>())?;
        let mut stringbuf = GpuBuffer::alloc(ctx, stringbuf_total)?;
        stringbuf.contents_mut().fill(STRINGBUF_POISON); // see the module docs
        let str_chunks = string_total.div_ceil(THREADGROUP_SIZE);
        let mut chunk_error = GpuBuffer::alloc(ctx, str_chunks * size_of::<u64>())?;
        // The long-string fixup list (K10's fixup plumbing, mirrored).
        // Accumulation targets get their preconditions established
        // explicitly (GpuBuffer::alloc makes no contents guarantee).
        let mut long_count = GpuBuffer::alloc(ctx, size_of::<u32>())?;
        long_count.as_mut_slice::<u32>()[0] = 0;
        // Worst case every string is long, so size from the string count —
        // acceptable because entries are index-sized u32s (4 bytes per
        // string), never content-sized.
        let mut long_list = GpuBuffer::alloc(ctx, string_total * size_of::<u32>())?;

        let input_len = bufs1.input_len() as u64;
        let tok_chunks = bufs2.chunks();
        let token_params = MjParams {
            input_len,
            element_count: token_total as u64,
            ..Default::default()
        };
        let string_params = MjParams {
            input_len,
            element_count: string_total as u64,
            reserved0: tape_words as u64,  // defensive tape bound
            reserved1: token_total as u64, // defensive token bound
        };
        let fold_params = MjParams {
            input_len,
            element_count: str_chunks as u64,
            ..Default::default()
        };

        // --- the string command buffer: one commit, one wait ----------------
        {
            let mut batch = ctx.batch()?;
            let h_input = batch.bind_read(&bufs1.input);
            let h_pos = batch.bind_read(bufs1.tok_pos.as_ref().expect("stage 2 allocated tokens"));
            let h_kind =
                batch.bind_read(bufs1.tok_kind.as_ref().expect("stage 2 allocated tokens"));
            let h_counts = batch.bind_read(&bufs2.chunk_counts);
            let h_sbytes = batch.bind_read(&bufs2.chunk_string_bytes);
            let h_strings =
                batch.bind_read(bufs2.string_tokens.as_ref().expect("lists allocated"));
            let h_tape_ofs = batch.bind_read(&bufs2.tape_ofs);
            let h_offsets = batch.bind_write(&mut record_offsets);
            let h_sb = batch.bind_write(&mut stringbuf);
            let h_tape = batch.bind_write(&mut tape_buf);
            let h_err = batch.bind_write(&mut chunk_error);
            let h_lcount = batch.bind_write(&mut long_count);
            let h_llist = batch.bind_write(&mut long_list);
            let h_header = batch.bind_write(&mut bufs1.header);

            self.offsets.encode(
                &mut batch,
                &[h_pos, h_kind, h_counts, h_sbytes, h_offsets],
                Some(&token_params),
                Dispatch::Threadgroups(tok_chunks),
            )?;
            self.unescape.encode(
                &mut batch,
                &[
                    h_input, h_pos, h_strings, h_offsets, h_tape_ofs, h_sb, h_tape, h_err,
                    h_lcount, h_llist,
                ],
                Some(&string_params),
                Dispatch::Threadgroups(str_chunks),
            )?;
            self.finalize.encode(
                &mut batch,
                &[h_err, h_header],
                Some(&fold_params),
                Dispatch::Threadgroups(1),
            )?;
            batch.commit_and_wait()?;
        }

        // --- CPU sync: long-string patch + the merged K11 verdict ------------
        // Kernel appends at most once per thread.
        let long_total = (long_count.as_slice::<u32>()[0] as usize).min(string_total);
        let mut long_string_fixups = long_list.as_slice::<u32>()[..long_total].to_vec();
        long_string_fixups.sort_unstable();

        // Patch the flagged records/tape words in the SHARED buffers in
        // place (the production flow: CPU-visible after the wait). Done
        // even when the GPU already found an error: a long string can
        // reject at an EARLIER offset, and the merged verdict is the
        // packed minimum — string extents are disjoint, so the minimum is
        // the reference's document-order-first verdict.
        let raw_input_len = bufs1.input_len();
        let patch_error = patch_long_strings(
            &bufs1.input.contents()[..raw_input_len],
            bufs1
                .tok_pos
                .as_ref()
                .expect("stage 2 allocated tokens")
                .as_slice::<u32>(),
            bufs2
                .string_tokens
                .as_ref()
                .expect("lists allocated")
                .as_slice::<u32>(),
            record_offsets.as_slice::<u64>(),
            bufs2.tape_ofs.as_slice::<u32>(),
            &long_string_fixups,
            stringbuf.contents_mut(),
            tape_buf.as_mut_slice::<u64>(),
        );

        let header = bufs1.read_header();
        let header_error = header.first_error().map(|(o, c)| (o << 32) | u64::from(c));
        let error = match (header_error, patch_error) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        if let Some(packed) = error {
            // String rejection: stage 2 accepted the input (its outputs are
            // kept), the string outputs are never produced. The fixup list
            // stays reported (diagnostic).
            let mut out = StringsOutput::rejected(stage2_out, packed);
            out.long_string_fixups = long_string_fixups;
            return Ok(out);
        }

        Ok(StringsOutput {
            stage2: stage2_out,
            record_offsets: record_offsets.as_slice::<u64>().to_vec(),
            stringbuf: stringbuf.contents().to_vec(),
            tape: tape_buf.as_slice::<u64>().to_vec(),
            long_string_fixups,
            error: None,
        })
    }
}

impl Default for StringsStage {
    fn default() -> Self {
        Self::new()
    }
}

/// One-shot convenience over [`StringsStage::run`] (builds the pipelines
/// each call; tests that run many inputs should hold a [`StringsStage`]).
///
/// # Errors
///
/// As [`StringsStage::run`].
pub fn run_strings(ctx: &MetalContext, input: &[u8]) -> Result<StringsOutput> {
    StringsStage::new().run(ctx, input)
}

/// The CPU half of the K11 long-string valve (`patch_number_fixups`'s
/// sibling): for every flagged string-list entry, re-run the **shared
/// reference unescaper** (`crate::unescape` — the same function
/// `reference::stage6_strings` calls, so the two paths cannot diverge) and
/// write the `[u32 LE len][content][NUL]` record into the string-buffer
/// slice at its precomputed offset plus the `"` tape word at
/// `tape_ofs[token]`. Both writes land in the shared `MTLBuffer` contents
/// after `waitUntilCompleted` — the same memory-model situation as the
/// number-fixup value-word patches. Record offsets were computed by
/// `string_record_offsets` from token positions alone, so CPU-written
/// records compose exactly with GPU-written neighbors (the pinned
/// raw-length allocation; gap bytes in oversized slots stay unwritten,
/// per the gap policy).
///
/// Returns the earliest packed `(offset << 32) | code` among flagged
/// strings that reject ([`ERR_STRING_ESCAPE`] at the backslash,
/// [`ERR_STRING_CONTROL`] at the control byte — reference-exact offsets),
/// or `None` when every flagged string patched cleanly. Callers merge it
/// into the verdict by packed minimum, exactly like fixup-driven number
/// rejections; string extents are disjoint, so the minimum reproduces the
/// reference's document-order-first verdict. Patch order does not matter
/// (disjoint slots/tape words), so callers may pass the list in any order.
///
/// # Panics
///
/// If an index is out of range of the lists, a record/tape slot is out of
/// range of its buffer, or the unescaper reports an error class stage 6
/// cannot produce — all internal-contract violations (the fixup list is
/// produced by K11 from the CB2-vetted string list).
#[allow(clippy::too_many_arguments)] // mirrors the K11 kernel's flat buffer list
pub fn patch_long_strings(
    input: &[u8],
    tok_pos: &[u32],
    string_tokens: &[u32],
    record_offsets: &[u64],
    tape_ofs: &[u32],
    long_fixups: &[u32],
    stringbuf: &mut [u8],
    tape: &mut [u64],
) -> Option<u64> {
    let mut first_error: Option<u64> = None;
    for &s in long_fixups {
        let t = string_tokens[s as usize] as usize;
        // The close quote is the very next token (post-CB1 even quote
        // total) — the same adjacency K11 relies on.
        let open_pos = tok_pos[t] as usize;
        let close_pos = tok_pos[t + 1] as usize;
        let raw = &input[open_pos + 1..close_pos];
        let base = u32::try_from(open_pos + 1).expect("input is capped below u32::MAX");
        match crate::unescape::unescape(raw, base) {
            Ok(bytes) => {
                let rec = usize::try_from(record_offsets[s as usize]).expect("offset fits usize");
                let len = u32::try_from(bytes.len()).expect("string longer than u32::MAX bytes");
                stringbuf[rec..rec + STRING_RECORD_HEADER_BYTES]
                    .copy_from_slice(&len.to_le_bytes());
                let content = rec + STRING_RECORD_HEADER_BYTES;
                stringbuf[content..content + bytes.len()].copy_from_slice(&bytes);
                stringbuf[content + bytes.len()] = 0;
                tape[tape_ofs[t] as usize] = make_string(record_offsets[s as usize]);
            }
            Err(Error::Syntax { offset, kind }) => {
                let code = match kind {
                    SyntaxErrorKind::InvalidStringEscape => ERR_STRING_ESCAPE,
                    SyntaxErrorKind::ControlCharacterInString => ERR_STRING_CONTROL,
                    other => panic!("the unescaper cannot produce {other:?}"),
                };
                let packed = (offset << 32) | u64::from(code);
                first_error = Some(first_error.map_or(packed, |e| e.min(packed)));
            }
            Err(other) => panic!("the unescaper cannot produce {other:?}"),
        }
    }
    first_error
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::make_string;

    /// GPU gating, as in stage1/2/3: skip without a device unless
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

    /// `\u` + `hex` escape text, built at runtime: the literal sequence
    /// must not appear in this source file (editor/tooling layers may
    /// resolve it like a JSON escape). Mirrors the reference test helper.
    fn u_esc(hex: &str) -> String {
        format!("{}u{hex}", '\\')
    }

    /// A quoted JSON string literal assembled from `parts`.
    fn quoted(parts: &[&str]) -> Vec<u8> {
        let mut s = String::from("\"");
        for p in parts {
            s.push_str(p);
        }
        s.push('"');
        s.into_bytes()
    }

    fn assert_strings_empty(out: &StringsOutput, label: &str) {
        assert!(out.record_offsets.is_empty(), "{label}: no record offsets");
        assert!(out.stringbuf.is_empty(), "{label}: no string buffer");
        assert!(out.tape.is_empty(), "{label}: no tape");
    }

    /// All record contents, in document order.
    fn contents(out: &StringsOutput) -> Vec<Vec<u8>> {
        (0..out.record_offsets.len())
            .map(|s| out.record_content(s).to_vec())
            .collect()
    }

    /// Unescape a single root-string input and return its content bytes.
    fn root_content(stage: &StringsStage, ctx: &MetalContext, input: &[u8]) -> Vec<u8> {
        let out = stage.run(ctx, input).unwrap();
        assert_eq!(
            out.error,
            None,
            "{:?} must unescape cleanly",
            String::from_utf8_lossy(input)
        );
        assert_eq!(out.record_offsets, vec![0], "one root string record");
        // Root string: tape = [root hole, " word, root hole].
        assert_eq!(out.tape.len(), 3);
        assert_eq!(out.tape[1], make_string(0));
        out.record_content(0).to_vec()
    }

    fn expect_err(stage: &StringsStage, ctx: &MetalContext, input: &[u8], code: u32) -> u64 {
        let out = stage.run(ctx, input).unwrap();
        let (offset, got_code) = out
            .error_offset_code()
            .unwrap_or_else(|| panic!("expected an error for {:?}", String::from_utf8_lossy(input)));
        assert_eq!(
            got_code,
            code,
            "code for {:?}",
            String::from_utf8_lossy(input)
        );
        // Rejection contract: string outputs are never produced; stage 2
        // accepted the input, so its outputs are kept.
        assert_strings_empty(&out, "rejected");
        assert!(
            !out.stage2.string_tokens.is_empty(),
            "{:?}: stage-2 outputs kept on a K11 rejection",
            String::from_utf8_lossy(input)
        );
        offset
    }

    /// The Rust constants and the kernel's `MJ_ERR_*` definitions must
    /// agree, and the codes must extend (never collide with) the
    /// `MjErrorCode` space common.h defines (which tops out at
    /// MJ_ERR_EMPTY_INPUT = 22). Parses the shader source, like the
    /// `msl_header_layout_lock` test in src/tape.rs.
    #[test]
    fn msl_error_codes_match_the_rust_constants() {
        let src = include_str!("../../shaders/13_strings.metal");
        for (name, value) in [
            ("MJ_ERR_STRING_ESCAPE", ERR_STRING_ESCAPE),
            ("MJ_ERR_STRING_CONTROL", ERR_STRING_CONTROL),
            ("MJ_LONG_STRING_THRESHOLD", LONG_STRING_THRESHOLD),
        ] {
            let needle = format!("constant constexpr uint {name} = {value};");
            assert!(
                src.contains(&needle),
                "shaders/13_strings.metal must define `{needle}`"
            );
        }
        const {
            assert!(ERR_STRING_ESCAPE > super::super::ERR_EMPTY_INPUT);
            assert!(ERR_STRING_CONTROL > super::super::ERR_EMPTY_INPUT);
            assert!(ERR_STRING_ESCAPE != ERR_STRING_CONTROL);
        }
    }

    /// The docs/tape-format.md worked example, every output hand-computed:
    /// `{"a":[1,2.5],"b":"x\n"}` — record offsets 0/6/12 (raw-length
    /// prefix sum), record bytes, and the three `"` tape words dropped
    /// into the 13-word tape's holes (positions 2, 9, 10 per the stage-2
    /// tape_ofs pin) with every other word still a zero hole.
    #[test]
    fn worked_example_strings_fill_their_tape_holes() {
        let Some(ctx) = ctx_or_skip("worked_example_strings_fill_their_tape_holes") else {
            return;
        };
        let out = StringsStage::new()
            .run(&ctx, br#"{"a":[1,2.5],"b":"x\n"}"#)
            .unwrap();
        assert_eq!(out.error, None);
        assert_eq!(out.record_offsets, vec![0, 6, 12]);
        assert_eq!(out.stringbuf.len(), 20);
        assert_eq!(contents(&out), vec![b"a".to_vec(), b"b".to_vec(), b"x\n".to_vec()]);
        // Record bytes, exactly as the tape-format doc tabulates them
        // (offset 19 is the gap byte — unspecified, NOT compared).
        assert_eq!(&out.stringbuf[0..6], &[1, 0, 0, 0, b'a', 0]);
        assert_eq!(&out.stringbuf[6..12], &[1, 0, 0, 0, b'b', 0]);
        assert_eq!(&out.stringbuf[12..19], &[2, 0, 0, 0, b'x', 0x0A, 0]);
        // The tape: string words at 2/9/10, zero holes everywhere else
        // (this runner dispatches neither K10 nor K12/K13).
        assert_eq!(out.tape.len(), 13);
        let mut want = vec![0u64; 13];
        want[2] = make_string(0);
        want[9] = make_string(6);
        want[10] = make_string(12);
        assert_eq!(out.tape, want);
        assert_eq!(out.stage2.string_tokens, vec![1, 10, 13]);
    }

    #[test]
    fn every_single_escape_and_unicode_escape_unescapes_exactly() {
        let Some(ctx) = ctx_or_skip("every_single_escape_and_unicode_escape_unescapes_exactly")
        else {
            return;
        };
        let stage = StringsStage::new();

        // All eight single-character escapes in one string.
        assert_eq!(
            root_content(&stage, &ctx, br#""\" \\ \/ \b \f \n \r \t""#),
            b"\" \\ / \x08 \x0C \n \r \t"
        );
        // Plain strings pass through; DEL (0x7F) is legal unescaped.
        assert_eq!(root_content(&stage, &ctx, br#""hello""#), b"hello");
        assert_eq!(root_content(&stage, &ctx, b"\"a\x7Fb\""), b"a\x7Fb");
        assert_eq!(
            root_content(&stage, &ctx, "\"héllo 😀\"".as_bytes()),
            "héllo 😀".as_bytes()
        );

        // \uXXXX across the UTF-8 width classes, both hex cases.
        let cases: &[(&str, &[u8])] = &[
            ("0041", b"A"),
            ("00e9", "é".as_bytes()),
            ("00E9", "é".as_bytes()),
            ("2603", "\u{2603}".as_bytes()),
            ("FFFF", "\u{FFFF}".as_bytes()),
        ];
        for &(hex, want) in cases {
            assert_eq!(
                root_content(&stage, &ctx, &quoted(&[&u_esc(hex)])),
                want,
                "{hex}"
            );
        }

        // Interior NUL via the legal backslash-u-0000 escape: the
        // what makes it representable.
        assert_eq!(
            root_content(&stage, &ctx, &quoted(&["a", &u_esc("0000"), "b"])),
            b"a\0b"
        );
    }

    #[test]
    fn surrogate_pairs_combine_up_to_u10ffff() {
        let Some(ctx) = ctx_or_skip("surrogate_pairs_combine_up_to_u10ffff") else {
            return;
        };
        let stage = StringsStage::new();
        let cases: &[(&str, &str, &str)] = &[
            ("D83D", "DE00", "\u{1F600}"), // 😀
            ("d83d", "de00", "\u{1F600}"), // lowercase hex
            ("d834", "dd1e", "\u{1D11E}"), // 𝄞
            ("D800", "DC00", "\u{10000}"), // first supplementary code point
            ("DBFF", "DFFF", "\u{10FFFF}"), // the very last code point
        ];
        for &(hi, lo, want) in cases {
            assert_eq!(
                root_content(&stage, &ctx, &quoted(&[&u_esc(hi), &u_esc(lo)])),
                want.as_bytes(),
                "{hi}/{lo}"
            );
        }
    }

    /// Reference-exact rejection offsets and codes: bad escapes point at
    /// the backslash, control characters at the raw byte — the
    /// (offset, kind) pairs reference stage 6's own test suite pins.
    #[test]
    fn rejections_report_reference_offsets_and_codes() {
        let Some(ctx) = ctx_or_skip("rejections_report_reference_offsets_and_codes") else {
            return;
        };
        let stage = StringsStage::new();

        // Invalid escapes (ERR_STRING_ESCAPE at the backslash).
        assert_eq!(expect_err(&stage, &ctx, br#""\x41""#, ERR_STRING_ESCAPE), 1);
        assert_eq!(
            expect_err(&stage, &ctx, &quoted(&[&u_esc("12")]), ERR_STRING_ESCAPE),
            1, // short hex: only two digits before the closing quote
        );
        assert_eq!(expect_err(&stage, &ctx, br#""\uZZZZ""#, ERR_STRING_ESCAPE), 1);
        assert_eq!(expect_err(&stage, &ctx, br#""ab\q""#, ERR_STRING_ESCAPE), 3);
        // Lone high surrogate; high chased by a non-surrogate escape; high
        // chased by a plain character; lone low; inverted pair.
        for parts in [
            vec![u_esc("D800")],
            vec![u_esc("D800"), u_esc("0041")],
            vec![u_esc("D800"), "x".to_owned()],
            vec![u_esc("DC00")],
            vec![u_esc("DE00"), u_esc("D83D")],
        ] {
            let part_refs: Vec<&str> = parts.iter().map(String::as_str).collect();
            let input = quoted(&part_refs);
            assert_eq!(
                expect_err(&stage, &ctx, &input, ERR_STRING_ESCAPE),
                1,
                "{parts:?}"
            );
        }

        // Raw control characters (ERR_STRING_CONTROL at the byte).
        assert_eq!(
            expect_err(&stage, &ctx, b"\"a\tb\"", ERR_STRING_CONTROL),
            2
        );
        assert_eq!(
            expect_err(&stage, &ctx, b"\"a\nb\"", ERR_STRING_CONTROL),
            2
        );
        assert_eq!(
            expect_err(&stage, &ctx, b"\"a\x01b\"", ERR_STRING_CONTROL),
            2
        );
        assert_eq!(
            expect_err(&stage, &ctx, b"\"a\x1Fb\"", ERR_STRING_CONTROL),
            2
        );

        // Inside containers, and the document-order-first fold: the
        // earliest bad byte of the first bad string wins.
        assert_eq!(
            expect_err(&stage, &ctx, br#"["ok","\q"]"#, ERR_STRING_ESCAPE),
            7
        );
        assert_eq!(
            expect_err(&stage, &ctx, br#"["\q","\p"]"#, ERR_STRING_ESCAPE),
            2
        );
        // A control char in string 1 beats an escape error in string 2.
        assert_eq!(
            expect_err(&stage, &ctx, b"[\"a\x06b\",\"\\q\"]", ERR_STRING_CONTROL),
            3
        );
    }

    /// Escapes, controls and surrogate pairs swept across every offset
    /// around the 16-byte fast-path block seams (0..=33 covers two full
    /// blocks plus both edges), so a special byte lands at every position
    /// of the vector scan: first byte, mid-block, last byte, block
    /// boundary, and in the < 16-byte tail.
    #[test]
    fn fast_path_seams_handle_specials_at_every_offset() {
        let Some(ctx) = ctx_or_skip("fast_path_seams_handle_specials_at_every_offset") else {
            return;
        };
        let stage = StringsStage::new();

        for k in 0..=33usize {
            // \n escape at raw offset k.
            let mut input = b"\"".to_vec();
            input.extend(std::iter::repeat_n(b'a', k));
            input.extend_from_slice(br"\n");
            input.extend(std::iter::repeat_n(b'b', 40));
            input.push(b'"');
            let mut want = vec![b'a'; k];
            want.push(b'\n');
            want.extend(std::iter::repeat_n(b'b', 40));
            assert_eq!(root_content(&stage, &ctx, &input), want, "escape at {k}");

            // Surrogate-pair escape straddling the seam at raw offset k.
            let mut input = b"\"".to_vec();
            input.extend(std::iter::repeat_n(b'a', k));
            input.extend_from_slice(u_esc("D83D").as_bytes());
            input.extend_from_slice(u_esc("DE00").as_bytes());
            input.extend(std::iter::repeat_n(b'c', 20));
            input.push(b'"');
            let mut want = vec![b'a'; k];
            want.extend_from_slice("\u{1F600}".as_bytes());
            want.extend(std::iter::repeat_n(b'c', 20));
            assert_eq!(root_content(&stage, &ctx, &input), want, "pair at {k}");

            // Raw control byte at raw offset k: rejected at 1 + k (the
            // content starts one past the open quote).
            let mut input = b"\"".to_vec();
            input.extend(std::iter::repeat_n(b'a', k));
            input.push(0x01);
            input.extend(std::iter::repeat_n(b'b', 8));
            input.push(b'"');
            assert_eq!(
                expect_err(&stage, &ctx, &input, ERR_STRING_CONTROL),
                1 + k as u64,
                "control at {k}"
            );
        }
    }

    /// A string whose escape straddles the 64-byte bitmap word seam (the
    /// reference's own stage-6 seam test).
    #[test]
    fn strings_spanning_bitmap_word_seams_unescape_fine() {
        let Some(ctx) = ctx_or_skip("strings_spanning_bitmap_word_seams_unescape_fine") else {
            return;
        };
        let stage = StringsStage::new();
        let mut input = b"\"".to_vec();
        input.extend(std::iter::repeat_n(b'a', 62)); // bytes 1..=62
        input.extend_from_slice(br"\n"); // backslash at 63, 'n' at 64
        input.extend_from_slice(b"b\"");
        let mut want = vec![b'a'; 62];
        want.push(b'\n');
        want.push(b'b');
        assert_eq!(root_content(&stage, &ctx, &input), want);
    }

    #[test]
    fn empty_strings_get_empty_records() {
        let Some(ctx) = ctx_or_skip("empty_strings_get_empty_records") else {
            return;
        };
        let stage = StringsStage::new();
        assert_eq!(root_content(&stage, &ctx, br#""""#), b"");

        // Three empty strings: 5-byte slots, offsets 0/5/10.
        let out = stage.run(&ctx, br#"["","",""]"#).unwrap();
        assert_eq!(out.error, None);
        assert_eq!(out.record_offsets, vec![0, 5, 10]);
        assert_eq!(out.stringbuf.len(), 15);
        for s in 0..3 {
            assert_eq!(out.record_content(s), b"", "record {s}");
        }
    }

    /// 8 KB with no escape anywhere: the 16-byte-block fast path end to
    /// end, including raw multi-byte UTF-8 passthrough. Timing is printed
    /// for the perf-cliff note, not asserted.
    #[test]
    fn fast_path_8kb_string() {
        let Some(ctx) = ctx_or_skip("fast_path_8kb_string") else {
            return;
        };
        let stage = StringsStage::new();
        let mut body = String::new();
        while body.len() < 8192 {
            body.push_str("abcdefgh é→😀 0123");
        }
        let input = quoted(&[&body]);
        let started = std::time::Instant::now();
        let content = root_content(&stage, &ctx, &input);
        eprintln!(
            "fast_path_8kb_string: {} raw bytes in {:?} (whole pipeline)",
            body.len(),
            started.elapsed()
        );
        assert_eq!(content, body.as_bytes());
    }

    /// 8 KB where EVERY unit is an escape: the sequential slow path on one
    /// thread (the documented v1 perf cliff — timing printed, correctness
    /// asserted). Expected bytes come from serde_json's own unescaper.
    #[test]
    fn slow_path_8kb_heavily_escaped_string() {
        let Some(ctx) = ctx_or_skip("slow_path_8kb_heavily_escaped_string") else {
            return;
        };
        let stage = StringsStage::new();
        let piece = format!(
            "{}{}{}{}{}{}{}",
            u_esc("D83D"),
            u_esc("DE00"),
            r"\n\t\\",
            "\\\"", // the \" escape (a raw string cannot hold a quote)
            u_esc("0041"),
            u_esc("0000"),
            r"\/"
        );
        let mut body = String::new();
        while body.len() < 8192 {
            body.push_str(&piece);
        }
        let input = quoted(&[&body]);
        let want: String = serde_json::from_slice(&input).expect("valid JSON string");
        let started = std::time::Instant::now();
        let content = root_content(&stage, &ctx, &input);
        eprintln!(
            "slow_path_8kb_heavily_escaped_string: {} raw bytes in {:?} (whole pipeline; \
             thread-per-string cliff documented in shaders/13_strings.metal)",
            body.len(),
            started.elapsed()
        );
        assert_eq!(content, want.as_bytes());
    }

    /// The long-string valve boundary: raw_len exactly AT the threshold
    /// stays on the GPU (no fixups); one byte over takes the CPU patch
    /// path — with identical record/tape output either way.
    #[test]
    fn long_string_valve_threshold_boundary() {
        let Some(ctx) = ctx_or_skip("long_string_valve_threshold_boundary") else {
            return;
        };
        let stage = StringsStage::new();
        let at = LONG_STRING_THRESHOLD as usize;

        // raw_len == threshold: GPU path.
        let body = "a".repeat(at);
        let out = stage.run(&ctx, &quoted(&[&body])).unwrap();
        assert_eq!(out.error, None);
        assert!(
            out.long_string_fixups.is_empty(),
            "at-threshold strings stay on the GPU"
        );
        assert_eq!(out.record_content(0), body.as_bytes());
        assert_eq!(out.tape[1], make_string(0));

        // raw_len == threshold + 1: the valve.
        let body = "a".repeat(at + 1);
        let out = stage.run(&ctx, &quoted(&[&body])).unwrap();
        assert_eq!(out.error, None);
        assert_eq!(
            out.long_string_fixups,
            vec![0],
            "just-over strings take the valve"
        );
        assert_eq!(out.record_content(0), body.as_bytes());
        assert_eq!(out.tape[1], make_string(0));

        // Just over WITH an escape (raw_len = threshold + 1 via the 2-byte
        // \n): the CPU unescape shrinks the record inside its slot.
        let body = format!("{}{}", "a".repeat(at - 1), r"\n");
        let out = stage.run(&ctx, &quoted(&[&body])).unwrap();
        assert_eq!(out.error, None);
        assert_eq!(out.long_string_fixups, vec![0]);
        let mut want = vec![b'a'; at - 1];
        want.push(b'\n');
        assert_eq!(out.record_content(0), want);
    }

    /// Long strings with an error PAST the threshold reject on the CPU
    /// patch path with the reference's exact (offset, code), merged into
    /// the packed verdict like fixup-driven number rejections; the fixup
    /// list stays reported on the rejection (diagnostic).
    #[test]
    fn long_string_valve_rejects_with_reference_offsets() {
        let Some(ctx) = ctx_or_skip("long_string_valve_rejects_with_reference_offsets") else {
            return;
        };
        let stage = StringsStage::new();
        let n = LONG_STRING_THRESHOLD as usize + 100;

        // Bad escape at raw offset n (absolute 1 + n: content starts one
        // past the open quote).
        let mut input = b"\"".to_vec();
        input.extend(std::iter::repeat_n(b'a', n));
        input.extend_from_slice(br"\q");
        input.extend(std::iter::repeat_n(b'b', 8));
        input.push(b'"');
        let out = stage.run(&ctx, &input).unwrap();
        assert_eq!(
            out.error_offset_code(),
            Some(((1 + n) as u64, ERR_STRING_ESCAPE)),
            "escape error past the threshold, at the backslash"
        );
        assert_eq!(out.long_string_fixups, vec![0], "the valve was exercised");
        assert_strings_empty(&out, "long escape rejection");

        // Raw control byte past the threshold.
        let mut input = b"\"".to_vec();
        input.extend(std::iter::repeat_n(b'a', n));
        input.push(0x01);
        input.push(b'"');
        let out = stage.run(&ctx, &input).unwrap();
        assert_eq!(
            out.error_offset_code(),
            Some(((1 + n) as u64, ERR_STRING_CONTROL)),
            "control error past the threshold, at the byte"
        );
        assert_eq!(out.long_string_fixups, vec![0]);
        assert_strings_empty(&out, "long control rejection");

        // Document-order-first across the GPU/CPU split: a SHORT bad
        // string after a LONG bad string — the long string's earlier
        // offset must win the merged fold.
        let mut input = b"[\"".to_vec();
        input.extend(std::iter::repeat_n(b'a', n));
        input.extend_from_slice(b"\\q\",\"\\p\"]");
        let out = stage.run(&ctx, &input).unwrap();
        assert_eq!(
            out.error_offset_code(),
            Some(((2 + n) as u64, ERR_STRING_ESCAPE)),
            "the long string's earlier backslash wins the merge"
        );
        assert_eq!(out.long_string_fixups, vec![0]);
    }

    /// Long + short strings interleaved: CPU-patched records land in their
    /// precomputed slots without disturbing the GPU-written neighbors —
    /// offsets and gaps hold around the patched records.
    #[test]
    fn long_and_short_strings_interleave_correctly() {
        let Some(ctx) = ctx_or_skip("long_and_short_strings_interleave_correctly") else {
            return;
        };
        let stage = StringsStage::new();
        let t = LONG_STRING_THRESHOLD as usize;

        let long_clean = "x".repeat(t + 7);
        let long_escaped = format!("{}{}", "y".repeat(t), r"\n\t"); // raw t + 4, shrinks by 2
        let bodies: [&str; 5] = ["a", &long_clean, r"b\n", &long_escaped, "cc"];
        let mut input = b"[".to_vec();
        let mut want_offsets = Vec::new();
        let mut offset = 0u64;
        for (i, &body) in bodies.iter().enumerate() {
            if i > 0 {
                input.push(b',');
            }
            input.extend_from_slice(&quoted(&[body]));
            want_offsets.push(offset);
            offset += body.len() as u64 + 5; // raw_len + 5 slots
        }
        input.push(b']');

        let out = stage.run(&ctx, &input).unwrap();
        assert_eq!(out.error, None);
        assert_eq!(
            out.long_string_fixups,
            vec![1, 3],
            "exactly the two long strings took the valve"
        );
        assert_eq!(out.record_offsets, want_offsets);
        assert_eq!(out.stringbuf.len() as u64, offset);
        let mut want_yy = vec![b'y'; t];
        want_yy.extend_from_slice(b"\n\t");
        let want_contents: Vec<Vec<u8>> = vec![
            b"a".to_vec(),
            long_clean.clone().into_bytes(),
            b"b\n".to_vec(),
            want_yy,
            b"cc".to_vec(),
        ];
        assert_eq!(contents(&out), want_contents);
        // Every tape word points at its slot (CPU-patched and GPU-written
        // alike).
        for (s, &tok) in out.stage2.string_tokens.iter().enumerate() {
            assert_eq!(
                out.tape[out.stage2.tape_ofs[tok as usize] as usize],
                make_string(want_offsets[s]),
                "tape word of record {s}"
            );
        }
    }

    /// Shrinking escapes create gaps but never move later offsets: slots
    /// are allocated from RAW lengths (token positions alone), so the
    /// records after a shrunk string land exactly where the prefix sum
    /// says — offset independence.
    #[test]
    fn shrunk_records_leave_gaps_and_later_offsets_hold() {
        let Some(ctx) = ctx_or_skip("shrunk_records_leave_gaps_and_later_offsets_hold") else {
            return;
        };
        let stage = StringsStage::new();
        // String 0: 12 raw bytes -> 4 content bytes (8-byte gap).
        // String 1: "x" (1 raw). String 2: "yy" (2 raw).
        let mut input = b"[".to_vec();
        input.extend_from_slice(&quoted(&[&u_esc("D83D"), &u_esc("DE00")]));
        input.extend_from_slice(b",\"x\",\"yy\"]");
        let out = stage.run(&ctx, &input).unwrap();
        assert_eq!(out.error, None);
        // Slots: 12+5=17, 1+5=6, 2+5=7 -> offsets 0, 17, 23.
        assert_eq!(out.record_offsets, vec![0, 17, 23]);
        assert_eq!(out.stringbuf.len(), 30);
        assert_eq!(out.record_content(0), "\u{1F600}".as_bytes());
        assert_eq!(out.record_content(1), b"x");
        assert_eq!(out.record_content(2), b"yy");
        // The record bytes around the gap: [4][😀][NUL] then unspecified
        // gap bytes (NOT compared — the pinned policy), then record 1.
        assert_eq!(&out.stringbuf[0..4], &[4, 0, 0, 0]);
        assert_eq!(&out.stringbuf[4..8], "\u{1F600}".as_bytes());
        assert_eq!(out.stringbuf[8], 0);
        assert_eq!(&out.stringbuf[17..23], &[1, 0, 0, 0, b'x', 0]);
        // Tape words point at the slot offsets.
        let t0 = out.stage2.string_tokens[0] as usize;
        let t1 = out.stage2.string_tokens[1] as usize;
        let t2 = out.stage2.string_tokens[2] as usize;
        assert_eq!(out.tape[out.stage2.tape_ofs[t0] as usize], make_string(0));
        assert_eq!(out.tape[out.stage2.tape_ofs[t1] as usize], make_string(17));
        assert_eq!(out.tape[out.stage2.tape_ofs[t2] as usize], make_string(23));
    }

    /// The duplicate-key corpus file: all three `"k"`s (and both `"d"`s,
    /// both `"x"`s) get records, verbatim, in document order — the tape
    /// does no deduplication (simdjson parity).
    #[test]
    fn duplicate_keys_corpus_keeps_every_record() {
        let Some(ctx) = ctx_or_skip("duplicate_keys_corpus_keeps_every_record") else {
            return;
        };
        let input = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus/duplicate_keys.json"),
        )
        .expect("corpus fixture");
        let out = StringsStage::new().run(&ctx, &input).unwrap();
        assert_eq!(out.error, None);
        let want: &[&str] = &[
            "k", "k", "k", "other", "d", "d", "arr", "x", "first", "x", "second",
        ];
        assert_eq!(
            contents(&out),
            want.iter().map(|s| s.as_bytes().to_vec()).collect::<Vec<_>>()
        );
        // No escapes in this file: raw == content, so the offsets are the
        // content-length prefix sum.
        let mut offset = 0u64;
        for (s, w) in want.iter().enumerate() {
            assert_eq!(out.record_offsets[s], offset, "record {s}");
            offset += w.len() as u64 + 5;
        }
        assert_eq!(out.stringbuf.len() as u64, offset);
    }

    /// The unicode-key corpus file: raw multi-byte UTF-8 keys (incl.
    /// astral 😀/😁 and the empty key) pass through the fast path intact.
    #[test]
    fn unicode_keys_corpus_round_trips() {
        let Some(ctx) = ctx_or_skip("unicode_keys_corpus_round_trips") else {
            return;
        };
        let input = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus/unicode_keys.json"),
        )
        .expect("corpus fixture");
        let out = StringsStage::new().run(&ctx, &input).unwrap();
        assert_eq!(out.error, None);
        let want: &[&str] = &[
            "héllo",
            "日本語のキー",
            "😀",
            "éscaped",
            "",
            "ключ",
            "中",
            "値",
            "😁 paired",
        ];
        assert_eq!(
            contents(&out),
            want.iter().map(|s| s.as_bytes().to_vec()).collect::<Vec<_>>()
        );
        let mut offset = 0u64;
        for (s, w) in want.iter().enumerate() {
            assert_eq!(out.record_offsets[s], offset, "record {s}");
            offset += w.len() as u64 + 5;
        }
    }

    /// Token streams spanning several 1024-token chunks, strings in every
    /// chunk (so the K7 chunk carries, the in-chunk offset scan and the
    /// per-string-chunk error fold all cross seams), every 7th string
    /// escaped (so raw lengths and content lengths diverge).
    #[test]
    fn multi_chunk_string_lists_offset_and_unescape_correctly() {
        let Some(ctx) = ctx_or_skip("multi_chunk_string_lists_offset_and_unescape_correctly")
        else {
            return;
        };
        let n = 900usize; // 2701 tokens -> 3 token chunks; 900 strings -> 4 string chunks
        let mut input = b"[".to_vec();
        let mut want_contents: Vec<Vec<u8>> = Vec::with_capacity(n);
        let mut want_offsets: Vec<u64> = Vec::with_capacity(n);
        let mut offset = 0u64;
        for i in 0..n {
            if i > 0 {
                input.push(b',');
            }
            input.push(b'"');
            let body = format!("s{i}");
            input.extend_from_slice(body.as_bytes());
            let raw_len = if i % 7 == 0 {
                input.extend_from_slice(br"\n");
                let mut c = body.clone().into_bytes();
                c.push(b'\n');
                want_contents.push(c);
                body.len() + 2
            } else {
                let len = body.len();
                want_contents.push(body.into_bytes());
                len
            };
            input.push(b'"');
            want_offsets.push(offset);
            offset += raw_len as u64 + 5;
        }
        input.push(b']');

        let out = StringsStage::new().run(&ctx, &input).unwrap();
        assert_eq!(out.error, None);
        assert_eq!(out.record_offsets, want_offsets);
        assert_eq!(out.stringbuf.len() as u64, offset);
        assert_eq!(contents(&out), want_contents);
        // Every string token's tape word, and zero holes everywhere else.
        let mut want_tape = vec![0u64; out.tape.len()];
        for (s, &t) in out.stage2.string_tokens.iter().enumerate() {
            let pos = out.stage2.tape_ofs[t as usize] as usize;
            want_tape[pos] = make_string(out.record_offsets[s]);
        }
        assert_eq!(out.tape, want_tape);

        // The same shape rejected LATE: corrupt the final string (`"s899"`
        // becomes `"s8\q"`) so the error sits in the last string chunk
        // (cross-chunk error fold).
        let mut bad = input.clone();
        let len = bad.len();
        bad[len - 4] = b'\\';
        bad[len - 3] = b'q';
        let out = StringsStage::new().run(&ctx, &bad).unwrap();
        let (off, code) = out.error_offset_code().expect("late escape error");
        assert_eq!(code, ERR_STRING_ESCAPE);
        assert_eq!(off as usize, len - 4, "the backslash of the last string");
    }

    /// Earlier-stage rejections carry forward unchanged: the string
    /// kernels never run, string outputs stay empty.
    #[test]
    fn earlier_stage_rejections_carry_forward() {
        let Some(ctx) = ctx_or_skip("earlier_stage_rejections_carry_forward") else {
            return;
        };
        let stage = StringsStage::new();

        // Stage 1: invalid UTF-8.
        let out = stage.run(&ctx, b"ab\x80").unwrap();
        assert_eq!(out.error_offset_code(), Some((2, super::super::ERR_UTF8)));
        assert_strings_empty(&out, "utf8");

        // Stage 1: odd quote count (the documented provisional offset).
        let out = stage.run(&ctx, b"\"abc").unwrap();
        assert_eq!(
            out.error_offset_code(),
            Some((4, super::super::ERR_STRING))
        );
        assert_strings_empty(&out, "odd quotes");

        // CB2 Layer 1: missing comma.
        let out = stage.run(&ctx, b"[1 true]").unwrap();
        assert_eq!(
            out.error_offset_code(),
            Some((3, super::super::ERR_MISSING_COMMA))
        );
        assert_strings_empty(&out, "layer1");

        // CPU verdict: empty input.
        let out = stage.run(&ctx, b" \t\n").unwrap();
        assert_eq!(
            out.error_offset_code(),
            Some((0, super::super::ERR_EMPTY_INPUT))
        );
        assert_strings_empty(&out, "empty");
    }

    /// Number-grammar problems are K10's job, not K11's: documents whose
    /// only flaw is a bad number must pass the string stage with correct
    /// records (the M4 error-class split).
    #[test]
    fn number_problems_pass_the_string_stage() {
        let Some(ctx) = ctx_or_skip("number_problems_pass_the_string_stage") else {
            return;
        };
        let stage = StringsStage::new();
        let out = stage.run(&ctx, br#"["a",01]"#).unwrap();
        assert_eq!(out.error, None, "bad number grammar is not a string error");
        assert_eq!(contents(&out), vec![b"a".to_vec()]);

        let out = stage.run(&ctx, b"[1e+]").unwrap();
        assert_eq!(out.error, None);
        assert!(out.record_offsets.is_empty());
        assert_eq!(out.tape.len() as u64, out.stage2.tape_word_total + 2);
        assert!(out.tape.iter().all(|&w| w == 0), "no strings, all holes");
    }

    // --- vs the cpu-reference oracle --------------------------------------

    #[cfg(feature = "cpu-reference")]
    mod vs_reference {
        use super::super::super::{
            ERR_EMPTY_INPUT, ERR_INVALID_LITERAL, ERR_MISSING_COLON, ERR_MISSING_COMMA,
            ERR_STRING, ERR_UNBALANCED, ERR_UNEXPECTED_TOKEN, ERR_UNTERMINATED_STRING, ERR_UTF8,
        };
        use super::*;
        use crate::reference::{
            stage1_classify, stage2_tokens, stage3_validate_local, stage6_strings,
        };
        use crate::tape::STRING_RECORD_HEADER_BYTES;
        use crate::{Error as CrateError, SyntaxErrorKind};

        /// The GPU code for each Layer-1 SyntaxErrorKind (mirrors the
        /// stage-2/3 test mapping).
        fn layer1_code(kind: SyntaxErrorKind) -> u32 {
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

        /// The GPU code for each stage-6 SyntaxErrorKind.
        fn string_code(kind: SyntaxErrorKind) -> u32 {
            match kind {
                SyntaxErrorKind::InvalidStringEscape => ERR_STRING_ESCAPE,
                SyntaxErrorKind::ControlCharacterInString => ERR_STRING_CONTROL,
                other => panic!("reference stage 6 cannot produce {other:?}"),
            }
        }

        /// Run both backends on `input` and require agreement.
        ///
        /// K11 is a token-level stage, exactly like reference
        /// `stage6_strings(tokens, input)`: the oracle here is reference
        /// stages 1–3 (the runner's acceptance domain) then stage 6 —
        /// structure (stage 4) is deliberately NOT consulted, mirroring
        /// what the runner dispatches. Inputs that pass stages 1–3 and 6
        /// compare every record offset, every record's
        /// `[u32 LE len][content][NUL]` bytes, every `"` tape word at
        /// `tape_ofs[token]`, and that all remaining tape positions are
        /// zero holes — gap bytes are never compared (the pinned policy).
        /// Rejected inputs compare the packed verdict, with the documented
        /// odd-quote offset exception.
        fn diff(stage: &StringsStage, ctx: &MetalContext, input: &[u8], label: &str) {
            let got = stage
                .run(ctx, input)
                .unwrap_or_else(|e| panic!("{label}: GPU strings stage failed: {e}"));

            // Reference stage 1.
            let bitmaps = match stage1_classify(input) {
                Ok(bitmaps) => bitmaps,
                Err(CrateError::Utf8 { offset }) => {
                    assert_eq!(
                        got.error_offset_code(),
                        Some((offset, ERR_UTF8)),
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
                // provisional offset — verdict parity only).
                assert_eq!(
                    got.error_offset_code(),
                    Some((input.len() as u64, ERR_STRING)),
                    "{label}: odd-quote verdict"
                );
                let tokens = stage2_tokens(&bitmaps, input);
                assert!(
                    stage3_validate_local(&tokens, input).is_err(),
                    "{label}: reference must also reject an odd-quote input"
                );
                return;
            }

            // Reference stages 2–3 (Layer 1 — the runner's acceptance gate).
            let tokens = stage2_tokens(&bitmaps, input);
            if let Err(err) = stage3_validate_local(&tokens, input) {
                let CrateError::Syntax { offset, kind } = err else {
                    panic!("{label}: unexpected reference error {err:?}");
                };
                assert_eq!(
                    got.error_offset_code(),
                    Some((offset, layer1_code(kind))),
                    "{label}: Layer-1 verdict for reference {kind:?}"
                );
                assert_strings_empty(&got, label);
                return;
            }

            // Reference stage 6 — the K11 spec.
            match stage6_strings(&tokens, input) {
                Err(CrateError::Syntax { offset, kind }) => {
                    assert_eq!(
                        got.error_offset_code(),
                        Some((offset, string_code(kind))),
                        "{label}: string verdict for reference {kind:?}"
                    );
                    // Rejection contract: stage-2 outputs kept, string
                    // outputs never produced.
                    assert_strings_empty(&got, label);
                    assert!(
                        !got.stage2.string_tokens.is_empty(),
                        "{label}: stage-2 outputs kept"
                    );
                }
                Err(other) => panic!("{label}: unexpected reference error {other:?}"),
                Ok(records) => {
                    assert_eq!(got.error, None, "{label}: spurious GPU error");
                    assert_eq!(
                        got.record_offsets.len(),
                        records.len(),
                        "{label}: record count"
                    );
                    assert_eq!(
                        got.tape.len() as u64,
                        got.stage2.tape_word_total + 2,
                        "{label}: tape length"
                    );
                    assert_eq!(
                        got.stringbuf.len() as u64,
                        got.stage2.stringbuf_total,
                        "{label}: string buffer size"
                    );
                    // The list keying: entry s describes the same string in
                    // both backends.
                    let want_tokens: Vec<u32> =
                        records.iter().map(|r| r.token_index).collect();
                    assert_eq!(
                        got.stage2.string_tokens, want_tokens,
                        "{label}: string token list"
                    );

                    let mut is_string_pos = vec![false; got.tape.len()];
                    for (s, rec) in records.iter().enumerate() {
                        assert_eq!(
                            got.record_offsets[s], rec.record_offset,
                            "{label}: offset of record {s}"
                        );
                        // Per-record bytes at the offset (never gap bytes).
                        let off = usize::try_from(rec.record_offset).unwrap();
                        let header: [u8; STRING_RECORD_HEADER_BYTES] = got.stringbuf
                            [off..off + STRING_RECORD_HEADER_BYTES]
                            .try_into()
                            .unwrap();
                        assert_eq!(
                            u32::from_le_bytes(header) as usize,
                            rec.bytes.len(),
                            "{label}: length prefix of record {s}"
                        );
                        let content = off + STRING_RECORD_HEADER_BYTES;
                        assert_eq!(
                            &got.stringbuf[content..content + rec.bytes.len()],
                            &rec.bytes[..],
                            "{label}: content of record {s}"
                        );
                        assert_eq!(
                            got.stringbuf[content + rec.bytes.len()],
                            0,
                            "{label}: NUL of record {s}"
                        );
                        // The `"` tape word at tape_ofs[token].
                        let pos = got.stage2.tape_ofs[rec.token_index as usize] as usize;
                        assert_eq!(
                            got.tape[pos],
                            make_string(rec.record_offset),
                            "{label}: tape word of record {s}"
                        );
                        is_string_pos[pos] = true;
                    }
                    // Everything K11 does not own is an untouched zero hole.
                    for (i, &word) in got.tape.iter().enumerate() {
                        if !is_string_pos[i] {
                            assert_eq!(word, 0, "{label}: hole at tape[{i}]");
                        }
                    }
                }
            }
        }

        #[test]
        fn corpus_files_match_reference_stage6() {
            let Some(ctx) = ctx_or_skip("corpus_files_match_reference_stage6") else {
                return;
            };
            let stage = StringsStage::new();
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
                diff(&stage, &ctx, &bytes, &name);
            }
        }

        /// Every JSONTestSuite file, GPU K11 vs reference stage 6: every
        /// record byte and tape word on accepted strings, verdict + code +
        /// offset on rejected ones (the n_string_* files in particular).
        #[test]
        fn jsontestsuite_files_match_reference_stage6() {
            let Some(ctx) = ctx_or_skip("jsontestsuite_files_match_reference_stage6") else {
                return;
            };
            let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("data/JSONTestSuite/test_parsing");
            if !dir.is_dir() {
                eprintln!(
                    "SKIP jsontestsuite_files_match_reference_stage6: {} not fetched \
                     (scripts/fetch_jsontestsuite.sh)",
                    dir.display()
                );
                return;
            }
            let stage = StringsStage::new();
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
                diff(&stage, &ctx, &bytes, &name);
            }
        }

        #[test]
        fn string_fixtures_match_reference_stage6() {
            let Some(ctx) = ctx_or_skip("string_fixtures_match_reference_stage6") else {
                return;
            };
            let stage = StringsStage::new();
            // Escape torture, gap shapes, seam straddles, rejections in
            // every class, plus structurally-odd-but-token-clean documents
            // (K11's domain is the token stream) — one differential sweep.
            let mut cases: Vec<Vec<u8>> = vec![
                br#""""#.to_vec(),
                br#""hello""#.to_vec(),
                br#""\" \\ \/ \b \f \n \r \t""#.to_vec(),
                quoted(&[&u_esc("0000")]),
                quoted(&[&u_esc("0041"), "mid", &u_esc("00e9")]),
                quoted(&[&u_esc("d83d"), &u_esc("de00")]),
                quoted(&[&u_esc("DBFF"), &u_esc("DFFF")]),
                quoted(&[&u_esc("D800")]),
                quoted(&[&u_esc("DC00")]),
                quoted(&[&u_esc("D800"), &u_esc("0041")]),
                quoted(&[&u_esc("DE00"), &u_esc("D83D")]),
                quoted(&[&u_esc("12")]),
                br#""\uZZZZ""#.to_vec(),
                br#""\x41""#.to_vec(),
                br#""ab\q""#.to_vec(),
                b"\"a\tb\"".to_vec(),
                b"\"a\x01b\"".to_vec(),
                b"\"a\x1F\"".to_vec(),
                b"\"a\x7Fb\"".to_vec(),
                "\"héllo 😀\"".as_bytes().to_vec(),
                br#"["", "x", ""]"#.to_vec(),
                br#"{"k":"v","k":"v2"}"#.to_vec(),
                br#"["ok","\q"]"#.to_vec(),
                br#"["\q","\p"]"#.to_vec(),
                b"[\"a\x06b\",\"\\q\"]".to_vec(),
                // Strings in documents that fail structure (stage 4) or
                // numbers (stage 5) — K11's outputs are defined regardless
                // (its domain is the token stream, like reference stage 6).
                br#"["a","b""#.to_vec(), // unclosed array: stages 1-3 accept
                br#"{"a":"b"}"#.to_vec(),
                br#"["a",01]"#.to_vec(),
                br#"{"k":1e999}"#.to_vec(),
                // Multi-error: a bad escape AND a bad number; the string
                // kernel only sees its own class.
                br#"[01,"\q"]"#.to_vec(),
                // Empty input / whitespace.
                b"".to_vec(),
                b" \t\n\r".to_vec(),
            ];

            // Escape/control/pair sweeps across the 16-byte seams.
            for k in 0..=17usize {
                let pad = "a".repeat(k);
                cases.push(quoted(&[&pad, r"\n", "bb"]));
                cases.push(quoted(&[&pad, &u_esc("D83D"), &u_esc("DE00"), "c"]));
                let mut ctl = b"\"".to_vec();
                ctl.extend(std::iter::repeat_n(b'a', k));
                ctl.push(0x02);
                ctl.extend_from_slice(b"b\"");
                cases.push(ctl);
            }
            // The 64-byte bitmap seam.
            let mut seam = b"\"".to_vec();
            seam.extend(std::iter::repeat_n(b'a', 62));
            seam.extend_from_slice(br"\nb");
            seam.push(b'"');
            cases.push(seam);
            // A shrunk record followed by more strings (gap + offsets).
            let mut gap = b"[".to_vec();
            gap.extend_from_slice(&quoted(&[&u_esc("D83D"), &u_esc("DE00")]));
            gap.extend_from_slice(b",\"x\",\"yy\"]");
            cases.push(gap);
            // Long strings: fast path and heavy escapes.
            let mut long_clean = String::new();
            while long_clean.len() < 8192 {
                long_clean.push_str("abcdefgh é→😀 0123");
            }
            cases.push(quoted(&[&long_clean]));
            let mut long_escaped = String::new();
            while long_escaped.len() < 8192 {
                long_escaped.push_str(&u_esc("D83D"));
                long_escaped.push_str(&u_esc("DE00"));
                long_escaped.push_str(r"\n\t\\");
                long_escaped.push_str("\\\""); // the \" escape
                long_escaped.push_str(&u_esc("0000"));
            }
            cases.push(quoted(&[&long_escaped]));
            // Multi-chunk: 900 strings, every 7th escaped.
            let mut big = b"[".to_vec();
            for i in 0..900 {
                if i > 0 {
                    big.push(b',');
                }
                big.push(b'"');
                big.extend_from_slice(format!("s{i}").as_bytes());
                if i % 7 == 0 {
                    big.extend_from_slice(br"\n");
                }
                big.push(b'"');
            }
            big.push(b']');
            cases.push(big.clone());
            // ... and the same with a late error (last string chunk).
            let mut bad = big;
            let len = bad.len();
            bad[len - 4] = b'\\';
            bad[len - 3] = b'q';
            cases.push(bad);

            for input in &cases {
                let label = format!(
                    "{:?}",
                    String::from_utf8_lossy(&input[..input.len().min(48)])
                );
                diff(&stage, &ctx, input, &label);
            }
        }

        /// Random strings with random escape density (valid AND invalid
        /// pieces — lone surrogates, bad designators, raw control bytes),
        /// assembled into array documents and diffed against the
        /// reference: accepted inputs roundtrip every record byte + tape
        /// word, rejected ones match verdict + code + offset.
        #[test]
        fn proptest_random_strings_match_reference() {
            use proptest::prelude::*;
            use proptest::test_runner::{Config, TestRunner};

            let Some(ctx) = ctx_or_skip("proptest_random_strings_match_reference") else {
                return;
            };
            let stage = StringsStage::new();

            // One raw piece of a string literal's body (escapes still
            // escaped). Mostly valid; the rare invalid pieces exercise
            // verdict parity.
            let piece = prop_oneof![
                // plain text (no quote/backslash/control)
                4 => "[a-zA-Z0-9 _.,:;<>~é中😀%-]{0,12}".prop_map(String::into_bytes),
                // the eight simple escapes
                2 => proptest::sample::select(vec![
                    &br#"\""#[..], &br"\\"[..], &br"\/"[..], &br"\b"[..],
                    &br"\f"[..], &br"\n"[..], &br"\r"[..], &br"\t"[..],
                ]).prop_map(<[u8]>::to_vec),
                // \uXXXX over the whole BMP — surrogate halves included,
                // so lone/inverted surrogates occur naturally
                2 => (0u32..=0xFFFF).prop_map(|cp| format!("{}u{cp:04X}", '\\').into_bytes()),
                // a correct surrogate pair for a supplementary code point
                1 => (0x10000u32..=0x10FFFF).prop_map(|cp| {
                    let c = cp - 0x10000;
                    format!(
                        "{}u{:04X}{}u{:04X}",
                        '\\', 0xD800 + (c >> 10), '\\', 0xDC00 + (c & 0x3FF)
                    ).into_bytes()
                }),
                // a random escape designator (usually invalid)
                1 => proptest::char::range(' ', '~').prop_map(|c| vec![b'\\', c as u8]),
                // a raw control byte (always invalid)
                1 => (0u8..0x20).prop_map(|b| vec![b]),
            ];
            let string = proptest::collection::vec(piece, 0..6).prop_map(|pieces| {
                let mut s = vec![b'"'];
                for p in pieces {
                    s.extend_from_slice(&p);
                }
                s.push(b'"');
                s
            });
            let doc = proptest::collection::vec(string, 0..8).prop_map(|strings| {
                let mut d = vec![b'['];
                for (i, s) in strings.iter().enumerate() {
                    if i > 0 {
                        d.push(b',');
                    }
                    d.extend_from_slice(s);
                }
                d.push(b']');
                d
            });

            let mut runner = TestRunner::new(Config {
                cases: 64,
                ..Config::default()
            });
            runner
                .run(&doc, |input| {
                    diff(&stage, &ctx, &input, "proptest");
                    Ok(())
                })
                .unwrap();
        }
    }
}
