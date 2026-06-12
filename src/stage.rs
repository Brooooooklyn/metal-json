//! Per-stage GPU pipeline abstraction, single-stage test harness, and the
//! stage-1 (CB1) scratch-buffer container.
//!
//! A [`Stage`] is one GPU kernel: a function name plus its lazily-built
//! [`Pipeline`], with an [`encode`](Stage::encode) that appends one dispatch
//! to a [`CommandBatch`]. The parser composes stages into command buffers;
//! [`TestHarness`] runs ONE stage in isolation against Rust slices, which is
//! how `tests/kernels.rs` diffs each kernel bit-for-bit against the
//! `cpu-reference` oracle (`src/reference/`).
//!
//! Internal/unstable: exposed publicly so integration tests can drive
//! kernels directly (like [`crate::metal`]), but not part of the supported
//! API surface.

use crate::error::{Error, Result};
use crate::metal::{
    Binding, BoundBuffer, CommandBatch, Dispatch, GpuBuffer, MetalContext, MjHeader, MjParams,
    Pipeline, Pod,
};
use crate::pool::{Alloc, ScratchPool};

// --- Pipeline geometry (mirrors shaders/common.h — keep in sync) ------------

/// Input bytes per bitmap word and per K1/K3/K5 thread.
/// Mirrors `MJ_WORD_BYTES`. The 64 B/thread grain is a spike-C decision:
/// giant-grid scheduling costs ~2.4 ns/threadgroup, so thread-per-byte
/// designs are out, and 64 bytes is one 64-bit bitmap word per thread.
pub const WORD_BYTES: usize = 64;

/// Bitmap words per spine chunk. Mirrors `MJ_CHUNK_WORDS`. K1/K3 emit one
/// popcount partial per chunk; the K2/K4 spine scans run as a single
/// threadgroup over those partials.
pub const CHUNK_WORDS: usize = 1024;

/// Input bytes per spine chunk (64 KiB).
pub const CHUNK_BYTES: usize = WORD_BYTES * CHUNK_WORDS;

/// Tokens per CB2 spine chunk and per K6/K6b threadgroup. Mirrors
/// `MJ_TOK_CHUNK_TOKENS`. One 256-thread threadgroup covers one chunk at
/// 4 tokens/thread (the K3/K5 shape transplanted to the token domain): K6
/// emits one partial record per chunk, the K7 spine scan runs as a single
/// threadgroup over those partials, K6b consumes the scanned carries.
pub const TOKEN_CHUNK_TOKENS: usize = 1024;

/// Skeleton elements per CB3 spine chunk and per depth/K8/K9 threadgroup.
/// Mirrors `MJ_SKEL_CHUNK_ELEMS` (the K6 shape in the skeleton domain).
pub const SKELETON_CHUNK_ELEMS: usize = 1024;

/// Number of 5-bit digit passes the K8 counting sort needs to keep every
/// *clean* sort key distinct, derived from the `max_depth` limit at encode
/// time (the pass count must be known before the GPU observes any depth).
///
/// Clean inputs only carry depths `1..=max_depth` (depth-0 separators are
/// `TrailingContent` errors, deeper nesting is a `DepthLimit` error — both
/// reject the input, discarding the sort output), so the sort key is
/// `depth - 1` in `0..=max_depth - 1` (`mj_sort_key` in
/// `shaders/common.h`) and the pass count is `ceil(bits(max_depth-1)/5)`:
/// 1 pass for limits up to 32, 2 passes for the 1024 default. Error-input
/// depths clamp INTO this key range rather than growing it (overflow
/// depths share `key_max` but stay inert in K9 — the `mj_sort_key`
/// contract), so the pass count never has to cover a key past
/// `max_depth - 1`.
#[must_use]
pub fn sort_passes(max_depth: u32) -> usize {
    let key_max = max_depth.max(1) - 1;
    let bits = (32 - key_max.leading_zeros()).max(1) as usize;
    bits.div_ceil(5)
}

/// K1 escape-carry look-back cap in bytes. Mirrors `MJ_ESCAPE_LOOKBACK_CAP`.
///
/// The valve: a K1 thread resolving whether its word starts escaped peeks
/// backward at the raw input for the preceding backslash run, at most this
/// many bytes. A run still going at the cap (adversarial "backslash wall")
/// bumps [`MjHeader::carry_overflow_count`] instead of looking further, and
/// a tiny sequential fix-up pass (GPU kernel, or CPU fallback over the
/// flagged words) corrects those words before K3 consumes the quote bitmap.
pub const ESCAPE_LOOKBACK_CAP: usize = 4096;

/// Maximum input size the GPU pipeline accepts. Token positions (`tok_pos`)
/// and tape indices are `u32`; identical to `reference::MAX_INPUT_BYTES`
/// (a test pins the equality).
pub const MAX_INPUT_BYTES: u64 = u32::MAX as u64 - 64;

// --- Stage -------------------------------------------------------------------

/// One GPU kernel as a pipeline stage: name + lazily-created pipeline-state
/// object + dispatch encoding.
///
/// The pipeline is built from the [`MetalContext`] on first use and cached,
/// so a `Stage` can be constructed (`const`) before any device exists and
/// stages the parser never runs (e.g. the escape fix-up valve on benign
/// input) never pay PSO creation.
#[derive(Debug)]
pub struct Stage {
    name: &'static str,
    pipeline: core::cell::OnceCell<Pipeline>,
}

impl Stage {
    /// A stage for the kernel function `name` (must exist in the shader
    /// library; surfaced as [`Error::KernelNotFound`] on first use if not).
    #[must_use]
    pub const fn new(name: &'static str) -> Self {
        Self {
            name,
            pipeline: core::cell::OnceCell::new(),
        }
    }

    /// Kernel function name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// The stage's pipeline, built from `ctx` on first call and cached.
    pub fn pipeline(&self, ctx: &MetalContext) -> Result<&Pipeline> {
        if self.pipeline.get().is_none() {
            // OnceCell is !Sync, so &self cannot be shared across threads
            // here; a racing/lost `set` is impossible.
            let _ = self.pipeline.set(Pipeline::new(ctx, self.name)?);
        }
        Ok(self.pipeline.get().expect("pipeline just initialized"))
    }

    /// Append this stage as one dispatch to `batch`, binding the declared
    /// buffers at `[[buffer(0..n)]]` in slice order (+ optional `MjParams`
    /// at index `n`).
    pub fn encode(
        &self,
        batch: &mut CommandBatch<'_>,
        buffers: &[BoundBuffer],
        params: Option<&MjParams>,
        work: Dispatch,
    ) -> Result<()> {
        let pipeline = self.pipeline(batch.ctx())?;
        batch.dispatch(pipeline, buffers, params, work);
        Ok(())
    }
}

// --- TestHarness ---------------------------------------------------------------

/// Runs ONE stage in isolation: upload arbitrary input/intermediate buffers
/// from Rust slices, dispatch, read outputs back as `Vec<T>`. This is what
/// the per-kernel unit tests in `tests/kernels.rs` drive — GPU kernel K vs
/// reference stage K on identical inputs.
#[derive(Debug)]
pub struct TestHarness {
    ctx: MetalContext,
}

impl TestHarness {
    /// Create a harness on the system default Metal device.
    pub fn new() -> Result<Self> {
        Ok(Self {
            ctx: MetalContext::new()?,
        })
    }

    /// The underlying context (for ad-hoc pipelines/batches in tests).
    #[must_use]
    pub fn ctx(&self) -> &MetalContext {
        &self.ctx
    }

    /// Allocate a GPU buffer holding exactly `data` (an input or a
    /// hand-crafted intermediate).
    pub fn upload<T: Pod>(&self, data: &[T]) -> Result<GpuBuffer> {
        let mut buffer = GpuBuffer::alloc(&self.ctx, size_of_val(data))?;
        buffer.write_from(data);
        Ok(buffer)
    }

    /// Allocate a zero-initialized GPU buffer for `count` elements of `T`
    /// (an output, or a zeroed intermediate). Zeroing is explicit —
    /// [`GpuBuffer::alloc`] makes no contents guarantee.
    pub fn alloc_zeroed<T: Pod>(&self, count: usize) -> Result<GpuBuffer> {
        let mut buffer = GpuBuffer::alloc(&self.ctx, count * size_of::<T>())?;
        buffer.contents_mut().fill(0);
        Ok(buffer)
    }

    /// Dispatch `stage` once over `work`, synchronously, with the M0
    /// [`Binding`] read/write soundness model. Encodes through a
    /// single-entry [`CommandBatch`], so every kernel test also exercises
    /// the production encoding path.
    pub fn run(
        &self,
        stage: &Stage,
        bindings: &mut [Binding<'_>],
        params: Option<&MjParams>,
        work: Dispatch,
    ) -> Result<()> {
        let mut batch = self.ctx.batch()?;
        let mut handles = Vec::with_capacity(bindings.len());
        for binding in bindings.iter_mut() {
            handles.push(match binding {
                Binding::Read(buffer) => batch.bind_read(buffer),
                Binding::ReadWrite(buffer) => batch.bind_write(buffer),
            });
        }
        stage.encode(&mut batch, &handles, params, work)?;
        batch.commit_and_wait()
    }

    /// Copy a buffer's contents back into a fresh `Vec<T>`.
    #[must_use]
    pub fn read_back<T: Pod>(&self, buffer: &GpuBuffer) -> Vec<T> {
        buffer.as_slice::<T>().to_vec()
    }
}

// --- Stage1Buffers ----------------------------------------------------------------

/// All GPU buffers for stage 1 of the pipeline (CB1: K1-K4, then K5 in CB2
/// after the exact-size token allocation). Sizes derive from the input
/// length alone; only `tok_pos`/`tok_kind` wait for the CB1 → CPU sync that
/// reads the exact token count from the header.
///
/// Size formulas (`n` = input bytes):
///
/// | buffer               | size (bytes)        | contents                              |
/// |----------------------|---------------------|---------------------------------------|
/// | `input`              | `words * 64`        | input copy, space-padded              |
/// | `bm_quote`           | `words * 8`         | escape-resolved real-quote bitmap     |
/// | `bm_tok`             | `words * 8`         | K1: candidates → K3: tokens (in place)|
/// | `escape_info`        | `words * 1`         | K1 carries + valve flags per word     |
/// | `chunk_quote_counts` | `chunks * 4`        | K1 partials → K2 scan (in place)      |
/// | `chunk_token_counts` | `chunks * 4`        | K3 partials → K4 scan (in place)      |
/// | `header`             | 128                 | [`MjHeader`]                          |
/// | `tok_pos`            | `token_total * 4`   | u32 byte positions (post-CB1)         |
/// | `tok_kind`           | `token_total * 1`   | u8 `TokenKind` discriminants          |
///
/// with `words = ceil(n / 64)` ([`WORD_BYTES`]) and
/// `chunks = ceil(words / 1024)` ([`CHUNK_WORDS`]).
///
/// Bitmaps are u64 words stored as `uint2 (lo, hi)` on the GPU — on
/// little-endian that IS the u64 layout, so tests read `bm_quote`/`bm_tok`
/// as `&[u64]` and diff directly against the reference
/// `Bitmaps` (`crate::reference::Bitmaps`) vectors.
///
/// # Zero/init preconditions (stage-1 invariant)
///
/// Some buffers are **accumulated into** by the kernels (read-modify-write),
/// so stage 1 is only correct when they start in a known state. Fresh
/// `MTLBuffer`s happen to arrive zero-filled today (zeroed VM pages), but
/// that is an allocator accident — [`GpuBuffer::alloc`] guarantees nothing,
/// and a pooled/reused buffer (planned for M5) keeps its old contents. The
/// constructor therefore establishes every precondition **explicitly**, and
/// [`reset_for_reuse`](Self::reset_for_reuse) re-establishes them for a
/// fresh parse over the same allocations:
///
/// | buffer               | precondition       | accumulating kernel(s)                       |
/// |----------------------|--------------------|-----------------------------------------------|
/// | `chunk_quote_counts` | all zero           | K1 `atomic_add`s simdgroup partials (skipping zero totals); the K1b valve `atomic_add`s repair deltas |
/// | `chunk_token_counts` | all zero           | none today — K3 plain-stores every entry — zeroed defensively so the buffer is pool-safe |
/// | `header`             | [`MjHeader::new`]  | K1 `atomic_add`s `carry_overflow_count`, `atomic_min`s the UTF-8 scratch sentinel; K2 `min`-folds `error` |
///
/// `bm_quote`, `bm_tok` and `escape_info` carry **no** precondition (K1
/// overwrites every word unconditionally), `input` is fully written by the
/// constructor, and `tok_pos`/`tok_kind` are fully written by the K5
/// scatter (dense ranks). The poisoned-buffer test in `tests/kernels.rs`
/// (`poisoned_buffers_reset_to_a_fresh_parse_state`) pins this invariant
/// against allocator behavior.
#[derive(Debug)]
pub struct Stage1Buffers {
    /// The input bytes, copied and padded with ASCII spaces (0x20) to a
    /// whole number of 64-byte words so K1 reads full words. Space padding
    /// classifies as whitespace — no spurious quote/candidate bits, no
    /// spurious scalar starts after a trailing scalar, and a multi-byte
    /// UTF-8 sequence truncated at EOF still fails (space is not a
    /// continuation byte) at the same offset the reference reports.
    /// Kernels still mask the tail word by `input_len` (defense in depth).
    ///
    /// The zero-copy path ([`Stage1Buffers::with_external_input`]) instead
    /// wraps caller-held page-aligned memory via
    /// [`GpuBuffer::from_page_aligned`]; that caller guarantees the same
    /// space-padding invariant for the tail word.
    pub input: GpuBuffer,
    /// True when `input` wraps caller-owned memory
    /// ([`with_external_input`](Self::with_external_input)): it must never
    /// be returned to a pool (the wrapped memory's lifetime belongs to the
    /// caller).
    input_external: bool,
    input_len: usize,
    words: usize,
    chunks: usize,
    /// K1 output: escape-resolved real-quote bitmap, one u64 per word.
    pub bm_quote: GpuBuffer,
    /// K1 writes the candidate bitmap (structural chars | scalar starts)
    /// here; K3 overwrites it **in place** with the token bitmap
    /// `(candidates & ~in_string) | quote_real` — the plan's bm_cand/bm_tok
    /// aliasing, saving `words * 8` bytes of traffic and footprint.
    pub bm_tok: GpuBuffer,
    /// One byte per word, written by K1 and read by the `escape_carry_fixup`
    /// valve kernel. Bit assignments (mirror `MJ_CARRY_*` in
    /// `shaders/common.h` — keep in sync): bit 0 = the prev-escaped carry K1
    /// used, bit 1 = the prev-allows-scalar-start carry K1 used, bit 2 = the
    /// backslash look-back hit [`ESCAPE_LOOKBACK_CAP`] (bit 0 is a guess),
    /// bit 3 = the word follows a `"` whose own look-back hit the cap
    /// (bit 1 is a guess).
    pub escape_info: GpuBuffer,
    /// One u32 per chunk. K1 writes per-chunk popcounts of `bm_quote`; the
    /// K2 spine scan (one threadgroup) rewrites them in place as exclusive
    /// prefix sums — the quote-rank carry whose low bit seeds each chunk's
    /// in-string parity in K3 — and writes the grand total to
    /// [`MjHeader::quote_total`].
    ///
    /// **Must be all-zero before K1** (see the struct docs): K1 and the K1b
    /// valve accumulate with `atomic_add`, and K1 skips zero partials
    /// entirely. Enforced by [`new`](Self::new) and
    /// [`reset_for_reuse`](Self::reset_for_reuse).
    pub chunk_quote_counts: GpuBuffer,
    /// One u32 per chunk. K3 writes per-chunk token popcounts; the K4 spine
    /// scan rewrites them as exclusive prefix sums — the token-rank carry
    /// K5 adds to its in-word prefix popcount — and writes the total to
    /// [`MjHeader::token_total`].
    ///
    /// Zeroed by [`new`](Self::new) and
    /// [`reset_for_reuse`](Self::reset_for_reuse). K3 plain-stores every
    /// entry today, so this is defense in depth rather than a hard
    /// precondition (see the struct docs).
    pub chunk_token_counts: GpuBuffer,
    /// One [`MjHeader`]. **Must be [`MjHeader::new`] before CB1** (see the
    /// struct docs): K1 `atomic_add`s the carry-overflow counter and
    /// `atomic_min`s the UTF-8 scratch cell, and K2 `min`-folds the error
    /// word — all relative to the initialized state. Enforced by
    /// [`new`](Self::new) and [`reset_for_reuse`](Self::reset_for_reuse).
    pub header: GpuBuffer,
    /// K5 output: token byte positions (u32), exactly `token_total` of
    /// them. `None` until [`alloc_tokens`](Self::alloc_tokens) applies the
    /// post-CB1 count.
    pub tok_pos: Option<GpuBuffer>,
    /// K5 output: token kinds (u8, `reference::TokenKind` discriminants in
    /// declaration order), exactly `token_total`. `None` until
    /// [`alloc_tokens`](Self::alloc_tokens).
    pub tok_kind: Option<GpuBuffer>,
}

impl Stage1Buffers {
    /// Allocate every input-length-derived buffer, copy `input` into a
    /// space-padded GPU buffer, and **explicitly establish the zero/init
    /// preconditions** (chunk count buffers zero-filled, header set to
    /// [`MjHeader::new`]) — see the struct docs; fresh allocations arriving
    /// zeroed is an accident this constructor must not rely on.
    ///
    /// # Errors
    ///
    /// [`Error::InputTooLarge`] above [`MAX_INPUT_BYTES`];
    /// [`Error::BufferAlloc`] if the device is out of memory.
    pub fn new(ctx: &MetalContext, input: &[u8]) -> Result<Self> {
        Self::new_in(ctx, Alloc::Direct, input)
    }

    /// [`new`](Self::new) with an explicit buffer source (the production
    /// parse path passes the parser's pool; the buffers come back with the
    /// previous parse's contents, which is why every precondition below is
    /// an explicit fill).
    pub(crate) fn new_in(ctx: &MetalContext, alloc: Alloc<'_>, input: &[u8]) -> Result<Self> {
        check_input_len(input.len() as u64)?;
        let words = input.len().div_ceil(WORD_BYTES);
        let mut input_buf = alloc.buffer(ctx, words * WORD_BYTES)?;
        let bytes = input_buf.contents_mut();
        bytes[..input.len()].copy_from_slice(input);
        bytes[input.len()..].fill(b' ');
        Self::assemble(ctx, alloc, input_buf, input.len(), false)
    }

    /// Build the stage-1 set around a caller-provided input buffer — the
    /// zero-copy path ([`GpuBuffer::from_page_aligned`] over an
    /// [`AlignedInput`](crate::AlignedInput) or an mmap). No input byte is
    /// copied; only the scratch buffers are allocated (from `alloc`).
    ///
    /// The caller guarantees, in addition to `from_page_aligned`'s
    /// contract, that bytes `input_len..input_len.next_multiple_of(64)` of
    /// the wrapped region are **ASCII spaces** — the same padding invariant
    /// [`new`](Self::new) writes (kernel tail words must classify the
    /// padding as whitespace).
    pub(crate) fn with_external_input(
        ctx: &MetalContext,
        alloc: Alloc<'_>,
        input_buf: GpuBuffer,
        input_len: usize,
    ) -> Result<Self> {
        check_input_len(input_len as u64)?;
        Self::assemble(ctx, alloc, input_buf, input_len, true)
    }

    fn assemble(
        ctx: &MetalContext,
        alloc: Alloc<'_>,
        input_buf: GpuBuffer,
        input_len: usize,
        input_external: bool,
    ) -> Result<Self> {
        let words = input_len.div_ceil(WORD_BYTES);
        let chunks = words.div_ceil(CHUNK_WORDS);

        // No zero/init needed: K1 overwrites every word of these.
        let bm_quote = alloc.buffer(ctx, words * size_of::<u64>())?;
        let bm_tok = alloc.buffer(ctx, words * size_of::<u64>())?;
        let escape_info = alloc.buffer(ctx, words)?;
        // Kernel accumulation targets: establish the documented zero/init
        // preconditions explicitly (a few bytes per 64 KiB chunk — cheap).
        let mut chunk_quote_counts = alloc.buffer(ctx, chunks * size_of::<u32>())?;
        let mut chunk_token_counts = alloc.buffer(ctx, chunks * size_of::<u32>())?;
        let mut header = alloc.buffer(ctx, size_of::<MjHeader>())?;
        chunk_quote_counts.contents_mut().fill(0);
        chunk_token_counts.contents_mut().fill(0);
        header.as_mut_slice::<MjHeader>()[0] = MjHeader::new();

        Ok(Self {
            input: input_buf,
            input_external,
            input_len,
            words,
            chunks,
            bm_quote,
            bm_tok,
            escape_info,
            chunk_quote_counts,
            chunk_token_counts,
            header,
            tok_pos: None,
            tok_kind: None,
        })
    }

    /// Unpadded input length in bytes.
    #[must_use]
    pub fn input_len(&self) -> usize {
        self.input_len
    }

    /// Bitmap words: `ceil(input_len / 64)`. The K1/K3/K5 grid size.
    #[must_use]
    pub fn words(&self) -> usize {
        self.words
    }

    /// Spine chunks: `ceil(words / 1024)`. The K2/K4 scan length.
    #[must_use]
    pub fn chunks(&self) -> usize {
        self.chunks
    }

    /// Read the header back (CB1 → CPU sync point).
    #[must_use]
    pub fn read_header(&self) -> MjHeader {
        self.header.as_slice::<MjHeader>()[0]
    }

    /// Re-arm the same allocations for a fresh parse: re-establishes every
    /// zero/init precondition from the struct docs (chunk count buffers
    /// zero-filled, header back to [`MjHeader::new`]) and drops the token
    /// buffers (the next parse re-allocates them at its exact token count).
    /// `input`, the bitmaps and `escape_info` need no reset — K1 overwrites
    /// them. Cheap: 4 bytes per 64 KiB chunk plus the 128-byte header.
    pub fn reset_for_reuse(&mut self) {
        self.chunk_quote_counts.contents_mut().fill(0);
        self.chunk_token_counts.contents_mut().fill(0);
        self.header.as_mut_slice::<MjHeader>()[0] = MjHeader::new();
        self.tok_pos = None;
        self.tok_kind = None;
    }

    /// Apply the exact token count read from [`MjHeader::token_total`]
    /// after CB1: allocates `tok_pos` (`token_count * 4` bytes) and
    /// `tok_kind` (`token_count` bytes) — exact-size, never a worst-case
    /// `input_len`-proportional guess.
    pub fn alloc_tokens(&mut self, ctx: &MetalContext, token_count: usize) -> Result<()> {
        self.alloc_tokens_in(ctx, Alloc::Direct, token_count)
    }

    /// [`alloc_tokens`](Self::alloc_tokens) with an explicit buffer source.
    /// No preconditions: the K5 scatter writes every entry (dense ranks).
    pub(crate) fn alloc_tokens_in(
        &mut self,
        ctx: &MetalContext,
        alloc: Alloc<'_>,
        token_count: usize,
    ) -> Result<()> {
        self.tok_pos = Some(alloc.buffer(ctx, token_count * size_of::<u32>())?);
        self.tok_kind = Some(alloc.buffer(ctx, token_count)?);
        Ok(())
    }

    /// Return every pool-eligible buffer to `pool`. An external input
    /// buffer ([`with_external_input`](Self::with_external_input)) wraps
    /// caller memory and is dropped instead of pooled.
    pub(crate) fn recycle(self, pool: &ScratchPool) {
        let Self {
            input,
            input_external,
            bm_quote,
            bm_tok,
            escape_info,
            chunk_quote_counts,
            chunk_token_counts,
            header,
            tok_pos,
            tok_kind,
            ..
        } = self;
        if !input_external {
            pool.put_back(input);
        }
        for buf in [
            bm_quote,
            bm_tok,
            escape_info,
            chunk_quote_counts,
            chunk_token_counts,
            header,
        ] {
            pool.put_back(buf);
        }
        if let Some(buf) = tok_pos {
            pool.put_back(buf);
        }
        if let Some(buf) = tok_kind {
            pool.put_back(buf);
        }
    }
}

/// The shared `MAX_INPUT_BYTES` guard for both input paths.
fn check_input_len(len: u64) -> Result<()> {
    if len > MAX_INPUT_BYTES {
        return Err(Error::InputTooLarge {
            len,
            max: MAX_INPUT_BYTES,
        });
    }
    Ok(())
}

// --- Stage2Buffers ----------------------------------------------------------------

/// All GPU buffers specific to the CB2 structure extension (K6 validate →
/// K7 spine3, then K6b apply after the CPU sync 2). Sizes derive from
/// `token_total` (known after the CB1 → CPU sync); only the skeleton /
/// string / scalar list buffers wait for the second sync that reads the K7
/// totals from the header — the same exact-size allocation rule as
/// `tok_pos`/`tok_kind` after CB1.
///
/// Size formulas (`t` = token count, `c` = `ceil(t / 1024)`
/// ([`TOKEN_CHUNK_TOKENS`])):
///
/// | buffer               | size (bytes)         | contents                                  |
/// |----------------------|----------------------|-------------------------------------------|
/// | `chunk_counts`       | `c * 16`             | uint4 (tape words, skeleton, string, scalar) partials → K7 scan (in place) |
/// | `chunk_string_bytes` | `c * 8`              | u64 `Σ raw_len + 5` partials → K7 scan (in place) |
/// | `chunk_error`        | `c * 8`              | per-chunk min packed error → K7 fold       |
/// | `tape_ofs`           | `t * 4`              | u32 tape position per token (K6b)          |
/// | `skel_token_index`   | `skeleton_total * 4` | u32 token index per skeleton record (post-K7) |
/// | `skel_pos`           | `skeleton_total * 4` | u32 byte position per skeleton record (post-K7) |
/// | `skel_byte`          | `skeleton_total`     | u8 structural byte `{}[]:,` (post-K7)      |
/// | `string_tokens`      | `string_total * 4`   | u32 `QuoteOpen` token indices (post-K7)    |
/// | `scalar_tokens`      | `scalar_total * 4`   | u32 `ScalarStart` token indices (post-K7)  |
///
/// # Zero/init preconditions
///
/// None. Unlike the stage-1 chunk count buffers (which K1 accumulates into
/// with atomics), every buffer here is **fully overwritten** by exactly one
/// single-writer kernel pass: K6's thread 0 plain-stores every chunk entry
/// of the three chunk buffers, and K6b's dense document-order ranks write
/// every entry of `tape_ofs` and the lists exactly once (the K7 totals the
/// lists were allocated from come from the same per-token classification).
/// Constructed fresh per parse in M3; the M5 buffer pool can reuse these
/// without resets.
#[derive(Debug)]
pub struct Stage2Buffers {
    token_total: usize,
    chunks: usize,
    /// K6 output: one `uint4` of partial counts per token chunk —
    /// (tape words, skeleton records, string records, scalar records).
    /// K7 rewrites it **in place** as four exclusive prefix sums (the chunk
    /// carries K6b adds to its in-chunk ranks) and stores the grand totals
    /// in [`MjHeader::tape_word_total`] / `skeleton_total` / `string_total`
    /// / `scalar_total`.
    pub chunk_counts: GpuBuffer,
    /// K6 output: one u64 string-slot byte sum (`Σ raw_len + 5`) per token
    /// chunk. 64-bit because a single string literal can exceed `u32`. K7
    /// rewrites it in place as an exclusive prefix sum (consumed by the M4
    /// string kernel) and stores the total in [`MjHeader::stringbuf_total`].
    pub chunk_string_bytes: GpuBuffer,
    /// K6 output: one packed `(offset << 32) | code` minimum per token
    /// chunk (`u64::MAX` = no error in the chunk), reduced deterministically
    /// in threadgroup memory — never device atomics. K7 min-folds these
    /// into [`MjHeader::error`].
    pub chunk_error: GpuBuffer,
    /// K6b output: tape position of every token — `1 +` the exclusive
    /// prefix sum of the tape footprints, the `+1` being the root prologue
    /// word at `tape[0]` (reference `emit_tape` seeds its running position
    /// at 1). Allocated up front (`token_total` is already exact); written
    /// only if Layer-1 validation passes (rejection contract).
    pub tape_ofs: GpuBuffer,
    /// K6b output: skeleton record field 1 of 3 — the stage-2 token index
    /// of each bracket / colon / comma, document order. Field values are
    /// bit-identical to the reference `SkeletonRecord.token_index`; the
    /// record is materialized struct-of-arrays (the reference's
    /// `{u32, u32, u8}` struct would pad to 12 bytes per record on the
    /// GPU). `None` until [`alloc_lists`](Self::alloc_lists).
    pub skel_token_index: Option<GpuBuffer>,
    /// K6b output: skeleton record field 2 — byte offset in the input
    /// (`SkeletonRecord.pos`). `None` until [`alloc_lists`](Self::alloc_lists).
    pub skel_pos: Option<GpuBuffer>,
    /// K6b output: skeleton record field 3 — the structural byte, one of
    /// `{ } [ ] : ,` (`SkeletonRecord.byte`; open/close brackets of one
    /// type differ by exactly `0x06`, which CB3's pair matching exploits).
    /// `None` until [`alloc_lists`](Self::alloc_lists).
    pub skel_byte: Option<GpuBuffer>,
    /// K6b output: `QuoteOpen` token indices in document order — the M4
    /// string kernel's work list (mirrors the reference
    /// `UnescapedString.token_index` order). `None` until
    /// [`alloc_lists`](Self::alloc_lists).
    pub string_tokens: Option<GpuBuffer>,
    /// K6b output: `ScalarStart` token indices in document order — the M4
    /// number/literal kernel's work list (mirrors the reference
    /// `ParsedScalar.token_index` order). `None` until
    /// [`alloc_lists`](Self::alloc_lists).
    pub scalar_tokens: Option<GpuBuffer>,
}

impl Stage2Buffers {
    /// Allocate every `token_total`-derived buffer (`token_total` must be
    /// the exact [`MjHeader::token_total`] from the CB1 sync, > 0 — empty
    /// token streams are an `EmptyInput` verdict decided on the CPU and
    /// never reach the CB2 kernels).
    ///
    /// # Errors
    ///
    /// [`Error::BufferAlloc`] if the device is out of memory.
    pub fn new(ctx: &MetalContext, token_total: usize) -> Result<Self> {
        Self::new_in(ctx, Alloc::Direct, token_total)
    }

    /// [`new`](Self::new) with an explicit buffer source (no zero/init
    /// preconditions — see the struct docs — so pooled reuse needs no
    /// resets).
    pub(crate) fn new_in(
        ctx: &MetalContext,
        alloc: Alloc<'_>,
        token_total: usize,
    ) -> Result<Self> {
        let chunks = token_total.div_ceil(TOKEN_CHUNK_TOKENS);
        Ok(Self {
            token_total,
            chunks,
            chunk_counts: alloc.buffer(ctx, chunks * 4 * size_of::<u32>())?,
            chunk_string_bytes: alloc.buffer(ctx, chunks * size_of::<u64>())?,
            chunk_error: alloc.buffer(ctx, chunks * size_of::<u64>())?,
            tape_ofs: alloc.buffer(ctx, token_total * size_of::<u32>())?,
            skel_token_index: None,
            skel_pos: None,
            skel_byte: None,
            string_tokens: None,
            scalar_tokens: None,
        })
    }

    /// Token count these buffers were sized for.
    #[must_use]
    pub fn token_total(&self) -> usize {
        self.token_total
    }

    /// Token spine chunks: `ceil(token_total / 1024)`. The K6/K6b grid size
    /// and the K7 scan length.
    #[must_use]
    pub fn chunks(&self) -> usize {
        self.chunks
    }

    /// Apply the exact list sizes read from the K7 totals after the CPU
    /// sync 2: allocates the three skeleton arrays plus the string / scalar
    /// work lists — exact-size, never an `input_len`-proportional guess.
    ///
    /// # Errors
    ///
    /// [`Error::BufferAlloc`] if the device is out of memory.
    pub fn alloc_lists(
        &mut self,
        ctx: &MetalContext,
        skeleton_total: usize,
        string_total: usize,
        scalar_total: usize,
    ) -> Result<()> {
        self.alloc_lists_in(ctx, Alloc::Direct, skeleton_total, string_total, scalar_total)
    }

    /// [`alloc_lists`](Self::alloc_lists) with an explicit buffer source.
    pub(crate) fn alloc_lists_in(
        &mut self,
        ctx: &MetalContext,
        alloc: Alloc<'_>,
        skeleton_total: usize,
        string_total: usize,
        scalar_total: usize,
    ) -> Result<()> {
        self.skel_token_index = Some(alloc.buffer(ctx, skeleton_total * size_of::<u32>())?);
        self.skel_pos = Some(alloc.buffer(ctx, skeleton_total * size_of::<u32>())?);
        self.skel_byte = Some(alloc.buffer(ctx, skeleton_total)?);
        self.string_tokens = Some(alloc.buffer(ctx, string_total * size_of::<u32>())?);
        self.scalar_tokens = Some(alloc.buffer(ctx, scalar_total * size_of::<u32>())?);
        Ok(())
    }

    /// Return every buffer to `pool`.
    pub(crate) fn recycle(self, pool: &ScratchPool) {
        let Self {
            chunk_counts,
            chunk_string_bytes,
            chunk_error,
            tape_ofs,
            skel_token_index,
            skel_pos,
            skel_byte,
            string_tokens,
            scalar_tokens,
            ..
        } = self;
        for buf in [chunk_counts, chunk_string_bytes, chunk_error, tape_ofs] {
            pool.put_back(buf);
        }
        for buf in [skel_token_index, skel_pos, skel_byte, string_tokens, scalar_tokens]
            .into_iter()
            .flatten()
        {
            pool.put_back(buf);
        }
    }
}

// --- Stage3Buffers ----------------------------------------------------------------

/// Bytes per `MjCtxState` entry (the K9 segmented-scan chunk summary /
/// carry record). Mirrors `struct MjCtxState` in `shaders/10_pair_ctx.metal`
/// — eight `u32` fields, keep in sync.
pub const CTX_STATE_BYTES: usize = 32;

/// All GPU buffers specific to CB3 (depth scan → K8 counting sort → K9
/// pair/context → error fold). Every size derives from `skeleton_total`
/// (read at the CPU sync 2) and the sort pass count
/// ([`sort_passes`]`(max_depth)`), both known before CB3 is encoded — no
/// further CPU sync is needed inside CB3.
///
/// Size formulas (`m` = skeleton elements, `c` = `ceil(m / 1024)`
/// ([`SKELETON_CHUNK_ELEMS`])):
///
/// | buffer            | size (bytes) | contents                                        |
/// |-------------------|--------------|--------------------------------------------------|
/// | `chunk_depth`     | `c * 8`      | i64 chunk weight sums → depth spine carries (in place) |
/// | `depths`          | `m * 4`      | u32 depth per skeleton element                   |
/// | `chunk_error`     | `c * 8`      | per-chunk min packed CB3 error (depth scan, then K9 folds on top) |
/// | `sort_hist`       | `32 * c * 4` | bucket-major digit histogram → matrix scan (in place), per pass |
/// | `max_key`         | `4`          | max clamped sort key (depth scan) → K8 shallow-pass skip |
/// | `sorted`          | `m * 4`      | the final (depth, document-order) ordering       |
/// | `sorted_scratch`  | `m * 4`      | radix ping-pong; `None` for single-pass sorts    |
/// | `chunk_ctx`       | `c * 32`     | K9 segmented-scan summaries → carries (in place) |
/// | `match_index`     | `m * 4`      | partner skeleton index per bracket (`NO_MATCH` for separators) |
/// | `context_opener`  | `m * 1`      | enclosing opener byte per separator, 0 for brackets |
/// | `child_counts`    | `m * 4`      | direct children per open bracket, 0 otherwise    |
///
/// # Zero/init preconditions
///
/// None. `chunk_depth`, `chunk_error`, `sort_hist` and `chunk_ctx` are
/// fully plain-stored by their producer kernels each dispatch; `max_key`
/// is zero-stored by `depth_spine` before `depth_apply` folds into it; the sort
/// scatter writes a permutation (every slot exactly once); `depths` is
/// fully written by `depth_apply`; and `match_index` / `context_opener` /
/// `child_counts` are fully written by `pair_ctx_apply` on every input
/// whose outputs are ever read (unpaired-open entries can only be stale on
/// inputs CB3 rejects, and the rejection contract discards those outputs).
#[derive(Debug)]
pub struct Stage3Buffers {
    skeleton_total: usize,
    chunks: usize,
    passes: usize,
    /// Depth-scan chunk partials, rewritten in place by `depth_spine` as
    /// the signed depth entering each chunk. i64: the running depth across
    /// chunks can exceed `i32` at max input size.
    pub chunk_depth: GpuBuffer,
    /// Depth of every skeleton element (root container = 1), bit-identical
    /// to reference `Stage4Output::depths` on accepted inputs.
    pub depths: GpuBuffer,
    /// One packed `(offset << 32) | code` minimum per skeleton chunk:
    /// `depth_apply` plain-stores the depth-scan candidates, then
    /// `pair_ctx_apply` min-folds its own on top (same chunk index space),
    /// and `structure_finalize` folds the buffer into `MjHeader::error`.
    pub chunk_error: GpuBuffer,
    /// The 32 × chunks bucket-major digit histogram of the current sort
    /// pass, rewritten in place by `sort_matrix_scan` as global output
    /// slots. Re-produced from scratch each pass.
    pub sort_hist: GpuBuffer,
    /// One `u32`: the maximum clamped sort key over all elements,
    /// zero-stored by `depth_spine` and max-folded by `depth_apply` each
    /// parse (so pooled reuse needs no reset). K8 passes whose digits are
    /// all zero under it degenerate to a stable identity copy.
    pub max_key: GpuBuffer,
    /// Skeleton indices in (depth, document-order) order — the final K8
    /// output, bit-identical to reference `Stage4Output::sorted_by_depth`.
    pub sorted: GpuBuffer,
    /// Radix ping-pong partner of [`sorted`](Self::sorted); only allocated
    /// when the sort needs more than one pass.
    pub sorted_scratch: Option<GpuBuffer>,
    /// K9 segmented-scan chunk summaries ([`CTX_STATE_BYTES`] each),
    /// rewritten in place by `ctx_spine` as exclusive walk-state carries.
    pub chunk_ctx: GpuBuffer,
    /// For brackets: skeleton index of the matching bracket; `NO_MATCH`
    /// (u32::MAX) for separators. Mirrors `Stage4Output::match_index`.
    pub match_index: GpuBuffer,
    /// For separators: the enclosing opener byte (`{` or `[`); 0 for
    /// brackets. Mirrors `Stage4Output::context_opener`.
    pub context_opener: GpuBuffer,
    /// For open brackets: number of direct children (full width; K12
    /// saturates). Mirrors `Stage4Output::child_counts`.
    pub child_counts: GpuBuffer,
}

impl Stage3Buffers {
    /// Allocate every CB3 buffer for `skeleton_total` elements and `passes`
    /// sort passes ([`sort_passes`]). `skeleton_total` must be > 0 — an
    /// empty skeleton (root scalar) skips CB3 entirely.
    ///
    /// # Errors
    ///
    /// [`Error::BufferAlloc`] if the device is out of memory.
    pub fn new(ctx: &MetalContext, skeleton_total: usize, passes: usize) -> Result<Self> {
        Self::new_in(ctx, Alloc::Direct, skeleton_total, passes)
    }

    /// [`new`](Self::new) with an explicit buffer source (no zero/init
    /// preconditions — see the struct docs — so pooled reuse needs no
    /// resets).
    pub(crate) fn new_in(
        ctx: &MetalContext,
        alloc: Alloc<'_>,
        skeleton_total: usize,
        passes: usize,
    ) -> Result<Self> {
        assert!(skeleton_total > 0, "empty skeletons never dispatch CB3");
        assert!(passes > 0, "the sort always runs at least one pass");
        let chunks = skeleton_total.div_ceil(SKELETON_CHUNK_ELEMS);
        Ok(Self {
            skeleton_total,
            chunks,
            passes,
            chunk_depth: alloc.buffer(ctx, chunks * size_of::<i64>())?,
            depths: alloc.buffer(ctx, skeleton_total * size_of::<u32>())?,
            chunk_error: alloc.buffer(ctx, chunks * size_of::<u64>())?,
            sort_hist: alloc.buffer(ctx, 32 * chunks * size_of::<u32>())?,
            max_key: alloc.buffer(ctx, size_of::<u32>())?,
            sorted: alloc.buffer(ctx, skeleton_total * size_of::<u32>())?,
            sorted_scratch: if passes > 1 {
                Some(alloc.buffer(ctx, skeleton_total * size_of::<u32>())?)
            } else {
                None
            },
            chunk_ctx: alloc.buffer(ctx, chunks * CTX_STATE_BYTES)?,
            match_index: alloc.buffer(ctx, skeleton_total * size_of::<u32>())?,
            context_opener: alloc.buffer(ctx, skeleton_total)?,
            child_counts: alloc.buffer(ctx, skeleton_total * size_of::<u32>())?,
        })
    }

    /// Return every buffer to `pool`.
    pub(crate) fn recycle(self, pool: &ScratchPool) {
        let Self {
            chunk_depth,
            depths,
            chunk_error,
            sort_hist,
            max_key,
            sorted,
            sorted_scratch,
            chunk_ctx,
            match_index,
            context_opener,
            child_counts,
            ..
        } = self;
        for buf in [
            chunk_depth,
            depths,
            chunk_error,
            sort_hist,
            max_key,
            sorted,
            chunk_ctx,
            match_index,
            context_opener,
            child_counts,
        ] {
            pool.put_back(buf);
        }
        if let Some(buf) = sorted_scratch {
            pool.put_back(buf);
        }
    }

    /// Skeleton element count these buffers were sized for.
    #[must_use]
    pub fn skeleton_total(&self) -> usize {
        self.skeleton_total
    }

    /// Skeleton spine chunks: `ceil(skeleton_total / 1024)`. The CB3
    /// per-chunk grid size and the spine scan lengths.
    #[must_use]
    pub fn chunks(&self) -> usize {
        self.chunks
    }

    /// Counting-sort passes these buffers were sized for.
    #[must_use]
    pub fn passes(&self) -> usize {
        self.passes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_geometry_constants_are_consistent() {
        assert_eq!(CHUNK_BYTES, WORD_BYTES * CHUNK_WORDS);
        assert_eq!(CHUNK_BYTES, 65536);
        // The look-back cap must cover whole words.
        assert_eq!(ESCAPE_LOOKBACK_CAP % WORD_BYTES, 0);
        // One 256-thread group, 4 tokens/thread, covers one token chunk.
        assert_eq!(TOKEN_CHUNK_TOKENS, crate::metal::THREADGROUP_SIZE * 4);
        assert_eq!(SKELETON_CHUNK_ELEMS, crate::metal::THREADGROUP_SIZE * 4);
    }

    #[test]
    fn sort_pass_counts_cover_the_clean_key_range() {
        // Keys are depth-1 in 0..max_depth; each pass covers 5 bits.
        assert_eq!(sort_passes(1), 1);
        assert_eq!(sort_passes(31), 1);
        assert_eq!(sort_passes(32), 1, "key_max 31 still fits one digit");
        assert_eq!(sort_passes(33), 2);
        assert_eq!(sort_passes(1024), 2, "the simdjson-parity default");
        assert_eq!(sort_passes(1025), 3);
        assert_eq!(sort_passes(u32::MAX), 7);
        // Degenerate limit: still dispatches a (vacuous) single pass.
        assert_eq!(sort_passes(0), 1);
    }

    #[cfg(feature = "cpu-reference")]
    #[test]
    fn max_input_matches_the_reference_pipeline() {
        assert_eq!(MAX_INPUT_BYTES, crate::reference::MAX_INPUT_BYTES);
    }

    #[test]
    fn stage_is_const_constructible_and_named() {
        // `Stage::new` must stay a const fn; a `const` *item* of an
        // interior-mutable type would re-instantiate (and re-create the
        // PSO) on every use, so stages live in regular bindings.
        const fn make() -> Stage {
            Stage::new("classify_escape_utf8")
        }
        let k1 = make();
        assert_eq!(k1.name(), "classify_escape_utf8");
    }
}
