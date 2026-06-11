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
    /// (an output, or a zeroed intermediate).
    pub fn alloc_zeroed<T: Pod>(&self, count: usize) -> Result<GpuBuffer> {
        GpuBuffer::alloc(&self.ctx, count * size_of::<T>())
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
/// | `header`             | 64                  | [`MjHeader`]                          |
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
    /// This is the copied path; the zero-copy mmap path
    /// ([`GpuBuffer::from_page_aligned`]) arrives with the `Parser`
    /// integration.
    pub input: GpuBuffer,
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
    pub chunk_quote_counts: GpuBuffer,
    /// One u32 per chunk. K3 writes per-chunk token popcounts; the K4 spine
    /// scan rewrites them as exclusive prefix sums — the token-rank carry
    /// K5 adds to its in-word prefix popcount — and writes the total to
    /// [`MjHeader::token_total`].
    pub chunk_token_counts: GpuBuffer,
    /// One [`MjHeader`], initialized by the constructor (error =
    /// [`MjHeader::NO_ERROR`], counts zero).
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
    /// Allocate every input-length-derived buffer and initialize the
    /// header; copies `input` into a space-padded GPU buffer.
    ///
    /// # Errors
    ///
    /// [`Error::InputTooLarge`] above [`MAX_INPUT_BYTES`];
    /// [`Error::BufferAlloc`] if the device is out of memory.
    pub fn new(ctx: &MetalContext, input: &[u8]) -> Result<Self> {
        if input.len() as u64 > MAX_INPUT_BYTES {
            return Err(Error::InputTooLarge {
                len: input.len() as u64,
                max: MAX_INPUT_BYTES,
            });
        }
        let words = input.len().div_ceil(WORD_BYTES);
        let chunks = words.div_ceil(CHUNK_WORDS);

        let mut input_buf = GpuBuffer::alloc(ctx, words * WORD_BYTES)?;
        let bytes = input_buf.contents_mut();
        bytes[..input.len()].copy_from_slice(input);
        bytes[input.len()..].fill(b' ');

        let bm_quote = GpuBuffer::alloc(ctx, words * size_of::<u64>())?;
        let bm_tok = GpuBuffer::alloc(ctx, words * size_of::<u64>())?;
        let escape_info = GpuBuffer::alloc(ctx, words)?;
        let chunk_quote_counts = GpuBuffer::alloc(ctx, chunks * size_of::<u32>())?;
        let chunk_token_counts = GpuBuffer::alloc(ctx, chunks * size_of::<u32>())?;
        let mut header = GpuBuffer::alloc(ctx, size_of::<MjHeader>())?;
        header.as_mut_slice::<MjHeader>()[0] = MjHeader::new();

        Ok(Self {
            input: input_buf,
            input_len: input.len(),
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

    /// Re-initialize the header for a fresh parse over the same buffers.
    pub fn reset_header(&mut self) {
        self.header.as_mut_slice::<MjHeader>()[0] = MjHeader::new();
    }

    /// Apply the exact token count read from [`MjHeader::token_total`]
    /// after CB1: allocates `tok_pos` (`token_count * 4` bytes) and
    /// `tok_kind` (`token_count` bytes) — exact-size, never a worst-case
    /// `input_len`-proportional guess.
    pub fn alloc_tokens(&mut self, ctx: &MetalContext, token_count: usize) -> Result<()> {
        self.tok_pos = Some(GpuBuffer::alloc(ctx, token_count * size_of::<u32>())?);
        self.tok_kind = Some(GpuBuffer::alloc(ctx, token_count)?);
        Ok(())
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
