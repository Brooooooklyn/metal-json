//! Safe wrapper layer over `objc2-metal`.
//!
//! This module is the **only** place in the crate that touches raw `objc2*`
//! types; everything above it works with [`MetalContext`], [`Pipeline`],
//! [`GpuBuffer`] and [`CommandBatch`].
//!
//! Internal/unstable: exposed publicly so integration tests and spikes can
//! drive kernels directly, but not part of the supported API surface.

mod batch;
mod buffer;
mod context;
mod pipeline;

pub use batch::{BoundBuffer, CommandBatch};
pub use buffer::{GpuBuffer, PAGE_SIZE, Pod};
pub use context::MetalContext;
pub use pipeline::Pipeline;

use crate::error::Result;

/// Threads per threadgroup for 1-D dispatches.
/// Mirrors `THREADGROUP_SIZE` in `shaders/common.h`.
pub const THREADGROUP_SIZE: usize = 256;

/// Kernel launch parameters, bound by value (`setBytes`) after the data
/// buffers. Mirrors `struct MjParams` in `shaders/common.h` — keep the
/// layouts in sync.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MjParams {
    /// Total input bytes.
    pub input_len: u64,
    /// Elements processed by this dispatch (per-thread bound check).
    pub element_count: u64,
    /// M2+: chunk counts, token counts, ...
    pub reserved0: u64,
    pub reserved1: u64,
}

/// Per-parse result header the kernels write and the CPU reads between
/// command buffers. Mirrors `struct MjHeader` in `shaders/common.h` — keep
/// the layouts in sync (eight `u64` fields, 64 bytes, no padding; a layout
/// test pins it).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MjHeader {
    /// Packed `(byte_offset << 32) | MjErrorCode` reduced on the GPU with a
    /// 64-bit `atomic_min`, so the earliest byte offset wins
    /// deterministically (the code breaks ties). [`MjHeader::NO_ERROR`]
    /// (all ones — any real error is smaller) when no kernel reported one.
    pub error: u64,
    /// K2 spine scan: total real-quote count over the whole input. Odd
    /// total ⇒ `UnterminatedString`.
    pub quote_total: u64,
    /// K4 spine scan: total token count. The CPU reads this after CB1 to
    /// size `tok_pos`/`tok_kind` exactly.
    pub token_total: u64,
    /// K1 escape valve: number of words whose backslash-run look-back hit
    /// the 4096-byte cap (`MJ_ESCAPE_LOOKBACK_CAP`) and need the sequential
    /// fix-up pass before K3.
    pub carry_overflow_count: u64,
    /// Reserved; pads the header to 64 bytes. The low 32 bits of
    /// `reserved[0]` are GPU scratch (`MjHeaderDev.utf8_error_offset`):
    /// K1 threads reduce UTF-8 error offsets there with a 32-bit
    /// `atomic_min` — 64-bit atomics are an Apple9+ device feature the
    /// embedded metallib cannot assume — and K2 thread 0 folds the winner
    /// into `error`. Initialized to the `u32::MAX` no-error sentinel.
    pub reserved: [u64; 4],
}

impl MjHeader {
    /// `error` value meaning "no error reported": all ones, so every real
    /// packed error wins the `atomic_min`. Mirrors `MJ_HEADER_NO_ERROR`.
    pub const NO_ERROR: u64 = u64::MAX;

    /// The UTF-8 scratch cell's "no error" sentinel, pre-set in the low 32
    /// bits of `reserved[0]`. Mirrors `MJ_NO_UTF8_ERROR`.
    pub const NO_UTF8_ERROR_SCRATCH: u64 = u32::MAX as u64;

    /// The initial header the CPU writes before CB1: no error, zero counts,
    /// UTF-8 scratch armed with its sentinel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            error: Self::NO_ERROR,
            quote_total: 0,
            token_total: 0,
            carry_overflow_count: 0,
            reserved: [Self::NO_UTF8_ERROR_SCRATCH, 0, 0, 0],
        }
    }

    /// Decode the packed error word as `(byte_offset, code)`, or `None`
    /// when no kernel reported an error.
    #[must_use]
    pub fn first_error(&self) -> Option<(u64, u32)> {
        (self.error != Self::NO_ERROR).then_some((self.error >> 32, self.error as u32))
    }
}

impl Default for MjHeader {
    fn default() -> Self {
        Self::new()
    }
}

/// Grid shape for a 1-D dispatch.
#[derive(Clone, Copy, Debug)]
pub enum Dispatch {
    /// `dispatchThreads:` with exactly this many threads. Apple GPUs support
    /// non-uniform threadgroup sizes, so any count works; kernels still
    /// bound-check against `MjParams::element_count`.
    Threads(usize),
    /// `dispatchThreadgroups:` with this many groups of the threadgroup size.
    Threadgroups(usize),
}

/// How a kernel accesses a bound buffer.
///
/// A buffer the kernel writes must be bound [`ReadWrite`](Binding::ReadWrite),
/// which takes `&mut GpuBuffer` — the exclusive borrow statically rules out
/// any live CPU slice view (`contents`/`as_slice`) across the dispatch that
/// mutates it. The binding mode must match what the kernel actually does;
/// kernels and their call sites both live in this crate, so a mismatch is an
/// internal bug, not caller error. [`CommandBatch`] extends the same model to
/// multi-dispatch command buffers (see `batch.rs`).
pub enum Binding<'a> {
    /// The kernel only reads this buffer.
    Read(&'a GpuBuffer),
    /// The kernel may write this buffer.
    ReadWrite(&'a mut GpuBuffer),
}

impl Binding<'_> {
    fn buffer(&self) -> &GpuBuffer {
        match self {
            Binding::Read(b) => b,
            Binding::ReadWrite(b) => b,
        }
    }
}

impl MetalContext {
    /// Encode and run **one** compute dispatch synchronously, as a
    /// single-entry [`CommandBatch`]: pipeline + buffers (+ optional
    /// `MjParams` at the index after the last buffer) → dispatch → commit →
    /// wait, surfacing any command-buffer error.
    ///
    /// Convenience for smoke tests and single-kernel unit tests; the real
    /// pipeline encodes its CB1/CB2/CB3 through [`MetalContext::batch`]
    /// directly.
    pub fn dispatch(
        &self,
        pipeline: &Pipeline,
        bindings: &[Binding<'_>],
        params: Option<&MjParams>,
        work: Dispatch,
    ) -> Result<()> {
        let mut batch = self.batch()?;
        let mut handles = Vec::with_capacity(bindings.len());
        for binding in bindings {
            // Registered via the shared-borrow path even for ReadWrite
            // bindings: `&Binding` cannot release its inner `&mut`. Sound
            // here because this call is fully synchronous — the caller's
            // borrows (including the exclusive one inside every ReadWrite
            // binding) are held across encode + commit + wait, which is
            // exactly the guarantee bind_write exists to provide.
            handles.push(batch.bind_read(binding.buffer()));
        }
        batch.dispatch(pipeline, &handles, params, work);
        batch.commit_and_wait()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Layout lock against `shaders/common.h` (`struct MjParams` /
    /// `struct MjHeader`): sizes, alignment and field offsets must match
    /// the MSL definitions exactly.
    #[test]
    fn mj_params_layout_matches_common_h() {
        assert_eq!(size_of::<MjParams>(), 32);
        assert_eq!(align_of::<MjParams>(), 8);
        assert_eq!(core::mem::offset_of!(MjParams, input_len), 0);
        assert_eq!(core::mem::offset_of!(MjParams, element_count), 8);
        assert_eq!(core::mem::offset_of!(MjParams, reserved0), 16);
        assert_eq!(core::mem::offset_of!(MjParams, reserved1), 24);
    }

    #[test]
    fn mj_header_layout_matches_common_h() {
        assert_eq!(size_of::<MjHeader>(), 64);
        assert_eq!(align_of::<MjHeader>(), 8);
        assert_eq!(core::mem::offset_of!(MjHeader, error), 0);
        assert_eq!(core::mem::offset_of!(MjHeader, quote_total), 8);
        assert_eq!(core::mem::offset_of!(MjHeader, token_total), 16);
        assert_eq!(core::mem::offset_of!(MjHeader, carry_overflow_count), 24);
        assert_eq!(core::mem::offset_of!(MjHeader, reserved), 32);
    }

    #[test]
    fn mj_header_error_packing_roundtrips() {
        let fresh = MjHeader::new();
        assert_eq!(fresh.error, MjHeader::NO_ERROR);
        assert_eq!(fresh.first_error(), None);

        // (offset << 32) | code, exactly what mj_pack_error produces.
        let packed = MjHeader {
            error: (123_456u64 << 32) | 2, // MJ_ERR_SYNTAX at byte 123456
            ..MjHeader::new()
        };
        assert_eq!(packed.first_error(), Some((123_456, 2)));

        // atomic_min semantics: the earlier offset packs to the smaller word.
        let early = (10u64 << 32) | 6;
        let late = (11u64 << 32) | 1;
        assert!(early < late, "earlier offset must win atomic_min");
    }
}
