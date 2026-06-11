//! Safe wrapper layer over `objc2-metal`.
//!
//! This module is the **only** place in the crate that touches raw `objc2*`
//! types; everything above it works with [`MetalContext`], [`Pipeline`] and
//! [`GpuBuffer`].
//!
//! Internal/unstable: exposed publicly so integration tests and spikes can
//! drive kernels directly, but not part of the supported API surface.

mod buffer;
mod context;
mod pipeline;

pub use buffer::{GpuBuffer, PAGE_SIZE, Pod};
pub use context::MetalContext;
pub use pipeline::Pipeline;

use core::ffi::c_void;
use core::ptr::NonNull;

use objc2_metal::{
    MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLSize,
};

use crate::error::{Error, Result};

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
    /// M1+: chunk counts, token counts, ...
    pub reserved0: u64,
    pub reserved1: u64,
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
/// internal bug, not caller error.
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
    /// Encode and run one compute dispatch synchronously:
    /// command buffer → encoder → pipeline + buffers (+ optional `MjParams`
    /// at the index after the last buffer) → dispatch → commit →
    /// `waitUntilCompleted`, then surface any command-buffer error.
    ///
    /// M0 helper: one dispatch per command buffer is plenty for smoke tests
    /// and per-kernel unit tests. Multi-dispatch command-buffer encoding
    /// (CB1/CB2/CB3 of the real pipeline) arrives with `stage.rs` in M2.
    pub fn dispatch(
        &self,
        pipeline: &Pipeline,
        bindings: &[Binding<'_>],
        params: Option<&MjParams>,
        work: Dispatch,
    ) -> Result<()> {
        let cmd_buf = self
            .queue()
            .commandBuffer()
            .ok_or_else(|| Error::CommandBuffer {
                message: "failed to create command buffer".to_owned(),
            })?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer {
                message: "failed to create compute command encoder".to_owned(),
            })?;

        encoder.setComputePipelineState(pipeline.state());
        for (index, binding) in bindings.iter().enumerate() {
            // SAFETY: the buffer is retained by the caller for the duration
            // of this synchronous call; offset 0 is always in bounds.
            unsafe { encoder.setBuffer_offset_atIndex(Some(binding.buffer().raw()), 0, index) };
        }
        if let Some(params) = params {
            let ptr = NonNull::from(params).cast::<c_void>();
            // SAFETY: `ptr` points at a live MjParams for the duration of the
            // call; setBytes copies the data into the command stream.
            unsafe {
                encoder.setBytes_length_atIndex(ptr, size_of::<MjParams>(), bindings.len());
            }
        }

        let group_width = pipeline
            .max_total_threads_per_threadgroup()
            .min(THREADGROUP_SIZE);
        let per_group = MTLSize {
            width: group_width,
            height: 1,
            depth: 1,
        };
        let grid = |width: usize| MTLSize {
            width,
            height: 1,
            depth: 1,
        };
        match work {
            Dispatch::Threads(n) => {
                encoder.dispatchThreads_threadsPerThreadgroup(grid(n), per_group);
            }
            Dispatch::Threadgroups(n) => {
                encoder.dispatchThreadgroups_threadsPerThreadgroup(grid(n), per_group);
            }
        }
        encoder.endEncoding();

        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        if cmd_buf.status() == MTLCommandBufferStatus::Error {
            let message = cmd_buf
                .error()
                .map(|e| e.localizedDescription().to_string())
                .unwrap_or_else(|| "unknown command buffer error".to_owned());
            return Err(Error::CommandBuffer { message });
        }
        Ok(())
    }
}
