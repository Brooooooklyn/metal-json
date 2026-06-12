//! Multi-dispatch command-buffer encoding: the builder behind the pipeline's
//! CB1/CB2/CB3 command buffers (N kernel dispatches, one commit, one wait).
//!
//! # Encoder strategy (verified choice)
//!
//! One [`CommandBatch`] encodes **all** of its dispatches into a single
//! `MTLComputeCommandEncoder` created with the default dispatch type,
//! `MTLDispatchTypeSerial`. Per the Metal documentation, a serial-dispatch
//! encoder executes its dispatches in encoding order with the equivalent of
//! a memory barrier between successive `dispatchThreadgroups` calls, so
//! producer→consumer chains over the same buffers (K1 writes the bitmaps,
//! K3 reads them) need no explicit barriers and no per-dispatch encoders.
//! Metal's implicit hazard tracking on (default, tracked) `MTLBuffer`s
//! orders access *across* encoders and command buffers; *within* one
//! encoder it is the serial dispatch type that provides the guarantee — an
//! `MTLDispatchTypeConcurrent` encoder would make
//! `memoryBarrierWithScope:` our responsibility. One encoder per batch is
//! also the cheapest shape to encode: spike C (docs/spikes.md) measured
//! ~14-29 µs of CPU encode cost per tiny dispatch and ~50-160 µs per extra
//! `waitUntilCompleted` round trip, so the batch exists precisely to pay
//! the encoder + sync cost once per command buffer instead of once per
//! kernel. Revisit (concurrent type + explicit barriers for provably
//! independent dispatches) only if M5 profiling shows the serial barriers
//! on the critical path.
//!
//! # Binding soundness model
//!
//! The M0 rule ([`Binding`](super::Binding)) extends to batches: a buffer
//! some dispatch **writes** must be registered with
//! [`CommandBatch::bind_write`], which takes `&mut GpuBuffer`; read-only
//! buffers register with [`CommandBatch::bind_read`] (`&GpuBuffer`). The
//! batch is invariant in its borrow lifetime and holds every registration
//! until [`commit_and_wait`](CommandBatch::commit_and_wait) consumes it, so
//! the borrow checker statically rules out any CPU slice view of a
//! GPU-written buffer — and any CPU mutation of a GPU-read buffer — being
//! live while the GPU may be touching it:
//!
//! ```compile_fail,E0502
//! use metal_json::metal::{GpuBuffer, MetalContext};
//!
//! fn race(ctx: &MetalContext) -> metal_json::Result<()> {
//!     let mut out = GpuBuffer::alloc(ctx, 4)?;
//!     let mut batch = ctx.batch()?;
//!     let _h = batch.bind_write(&mut out);
//!     let view = out.as_slice::<u32>(); // ERROR: still mutably borrowed by `batch`
//!     batch.commit_and_wait()?;
//!     let _ = view;
//!     Ok(())
//! }
//! ```
//!
//! As in M0, the registration mode must match what the kernels actually do;
//! kernels and call sites both live in this crate, so a mismatch is an
//! internal bug, not caller error.

use core::cell::Cell;
use core::ffi::c_void;
use core::marker::PhantomData;
use core::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLSize,
};

use super::{Dispatch, GpuBuffer, MetalContext, MjParams, Pipeline, THREADGROUP_SIZE};
use crate::error::{Error, Result};

/// Whether `METAL_JSON_SPLIT_KERNELS=1` put batches into the
/// measurement-only one-command-buffer-per-dispatch mode (checked once).
#[cfg(feature = "timing")]
fn split_kernels_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("METAL_JSON_SPLIT_KERNELS").is_some_and(|v| v == "1"))
}

/// Handle to a buffer registered with [`CommandBatch::bind_read`] /
/// [`CommandBatch::bind_write`]: an index into that batch's resource table.
///
/// It is *not* a kernel buffer-bind slot — the position of the handle in a
/// [`CommandBatch::dispatch`] call's `buffers` slice decides the
/// `[[buffer(i)]]` index for that dispatch, so one registration can sit at
/// different slots in different dispatches (K1 writes the bitmap K3 reads).
///
/// Handles are only meaningful for the batch that created them; passing a
/// handle to a different batch binds the wrong buffer or panics
/// (internal-crate misuse, like a wrong `Binding` mode).
#[derive(Clone, Copy, Debug)]
pub struct BoundBuffer {
    index: usize,
}

/// A command buffer being built: N compute dispatches sharing one serial
/// encoder, committed and awaited as a unit. See the module docs for the
/// encoder strategy and the binding soundness model.
///
/// # Example (CB-style chain: two dependent dispatches, one sync)
///
/// ```no_run
/// use metal_json::metal::{Dispatch, GpuBuffer, MetalContext, MjParams, Pipeline};
///
/// # fn main() -> metal_json::Result<()> {
/// let ctx = MetalContext::new()?;
/// let add = Pipeline::new(&ctx, "smoke_add")?;
/// let a = GpuBuffer::alloc(&ctx, 1024)?;
/// let b = GpuBuffer::alloc(&ctx, 1024)?;
/// let mut mid = GpuBuffer::alloc(&ctx, 1024)?;
/// let mut out = GpuBuffer::alloc(&ctx, 1024)?;
/// let params = MjParams { input_len: 1024, element_count: 256, ..Default::default() };
///
/// let mut batch = ctx.batch()?;
/// let ha = batch.bind_read(&a);
/// let hb = batch.bind_read(&b);
/// let hmid = batch.bind_write(&mut mid);
/// let hout = batch.bind_write(&mut out);
/// batch.dispatch(&add, &[ha, hb, hmid], Some(&params), Dispatch::Threads(256));
/// // The serial encoder guarantees this dispatch sees `mid` fully written.
/// batch.dispatch(&add, &[hmid, hb, hout], Some(&params), Dispatch::Threads(256));
/// batch.commit_and_wait()?;
/// let sums = out.as_slice::<u32>(); // fine: the batch (and its borrows) are gone
/// # let _ = sums;
/// # Ok(())
/// # }
/// ```
pub struct CommandBatch<'env> {
    ctx: &'env MetalContext,
    cmd_buf: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    resources: Vec<&'env ProtocolObject<dyn MTLBuffer>>,
    finished: bool,
    /// Makes the batch **invariant** in `'env`. Without this the batch
    /// would be covariant (shrinkable) in `'env`, letting borrowck end the
    /// registered borrows before `commit_and_wait` — which would re-admit
    /// exactly the CPU-view-during-GPU-write race the model exists to ban.
    _invariant: PhantomData<Cell<&'env ()>>,
}

impl MetalContext {
    /// Start building a multi-dispatch command buffer.
    pub fn batch(&self) -> Result<CommandBatch<'_>> {
        let cmd_buf = self
            .queue()
            .commandBuffer()
            .ok_or_else(|| Error::CommandBuffer {
                message: "failed to create command buffer".to_owned(),
            })?;
        let encoder = cmd_buf
            .computeCommandEncoder() // default MTLDispatchTypeSerial
            .ok_or_else(|| Error::CommandBuffer {
                message: "failed to create compute command encoder".to_owned(),
            })?;
        Ok(CommandBatch {
            ctx: self,
            cmd_buf,
            encoder,
            resources: Vec::new(),
            finished: false,
            _invariant: PhantomData,
        })
    }
}

impl<'env> CommandBatch<'env> {
    /// The context this batch encodes for (handy for lazy pipeline lookup
    /// while the batch holds no borrow of itself).
    pub fn ctx(&self) -> &'env MetalContext {
        self.ctx
    }

    /// Register a buffer that every dispatch in this batch only **reads**.
    /// The shared borrow lasts until the batch is consumed, so the CPU can
    /// keep reading it but cannot mutate it mid-batch.
    pub fn bind_read(&mut self, buffer: &'env GpuBuffer) -> BoundBuffer {
        self.push(buffer)
    }

    /// Register a buffer that some dispatch in this batch **writes** (it
    /// may also be read, by the same or other dispatches). Takes the
    /// exclusive borrow for the whole batch lifetime: no CPU view of the
    /// buffer can coexist with the batch. Register each buffer once, with
    /// the strongest access any dispatch needs.
    pub fn bind_write(&mut self, buffer: &'env mut GpuBuffer) -> BoundBuffer {
        self.push(buffer)
    }

    fn push(&mut self, buffer: &'env GpuBuffer) -> BoundBuffer {
        let index = self.resources.len();
        self.resources.push(buffer.raw());
        BoundBuffer { index }
    }

    /// Encode one compute dispatch: `pipeline` with `buffers` bound at
    /// `[[buffer(0..n)]]` in slice order, optional `MjParams` by value at
    /// index `n`, over the given grid. Execution order between dispatches
    /// is encoding order (serial encoder, see module docs).
    ///
    /// # Panics
    ///
    /// If a [`BoundBuffer`] does not belong to this batch (index out of
    /// range — programmer error, like a wrong `Binding` mode in M0).
    pub fn dispatch(
        &mut self,
        pipeline: &Pipeline,
        buffers: &[BoundBuffer],
        params: Option<&MjParams>,
        work: Dispatch,
    ) {
        self.encoder.setComputePipelineState(pipeline.state());
        for (slot, handle) in buffers.iter().enumerate() {
            let raw = self.resources[handle.index];
            // SAFETY: the buffer is borrowed by this batch for 'env, which
            // outlives the synchronous commit_and_wait; offset 0 is always
            // in bounds.
            unsafe { self.encoder.setBuffer_offset_atIndex(Some(raw), 0, slot) };
        }
        if let Some(params) = params {
            let ptr = NonNull::from(params).cast::<c_void>();
            // SAFETY: `ptr` points at a live MjParams for the duration of
            // the call; setBytes copies the data into the command stream.
            unsafe {
                self.encoder
                    .setBytes_length_atIndex(ptr, size_of::<MjParams>(), buffers.len());
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
                self.encoder
                    .dispatchThreads_threadsPerThreadgroup(grid(n), per_group);
            }
            Dispatch::Threadgroups(n) => {
                self.encoder
                    .dispatchThreadgroups_threadsPerThreadgroup(grid(n), per_group);
            }
        }

        #[cfg(feature = "timing")]
        if split_kernels_enabled() {
            self.split_after_dispatch(pipeline.name());
        }
    }

    /// Measurement-only (`METAL_JSON_SPLIT_KERNELS=1` + the `timing`
    /// feature): cut the command buffer right after a dispatch — end the
    /// encoder, commit, wait, record the kernel's GPU execution time — and
    /// start a fresh command buffer + encoder for the batch's remaining
    /// dispatches. Each dispatch binds its PSO/buffers itself, so a fresh
    /// encoder needs no rebinding. Serial-encoder ordering is preserved
    /// (the wait is a stronger barrier); only wall time inflates (one sync
    /// per kernel), which the per-phase gap columns expose separately.
    #[cfg(feature = "timing")]
    fn split_after_dispatch(&mut self, kernel: &str) {
        self.encoder.endEncoding();
        self.cmd_buf.commit();
        self.cmd_buf.waitUntilCompleted();
        let gpu = (self.cmd_buf.GPUEndTime() - self.cmd_buf.GPUStartTime()).max(0.0);
        crate::gpu::timing::record_kernel(kernel, gpu);
        self.cmd_buf = self
            .ctx
            .queue()
            .commandBuffer()
            .expect("split-kernel mode: fresh command buffer");
        self.encoder = self
            .cmd_buf
            .computeCommandEncoder()
            .expect("split-kernel mode: fresh compute encoder");
    }

    /// End encoding, commit, block until the GPU finishes, and surface any
    /// command-buffer error. Consumes the batch, releasing every registered
    /// borrow — only then can the CPU read the outputs.
    pub fn commit_and_wait(self) -> Result<()> {
        self.commit_and_wait_timed().map(|_| ())
    }

    /// [`commit_and_wait`](Self::commit_and_wait), additionally returning
    /// the command buffer's GPU execution time in seconds (`GPUEndTime −
    /// GPUStartTime`; clamps to zero when the device reports no timestamps,
    /// e.g. for a command buffer with no GPU work — the spike-C caveat).
    /// Coarse whole-CB timing for sanity tests; per-kernel breakdowns are
    /// the M5 `timing` feature's job.
    pub fn commit_and_wait_timed(mut self) -> Result<f64> {
        self.encoder.endEncoding();
        self.finished = true; // Drop must not endEncoding twice
        self.cmd_buf.commit();
        self.cmd_buf.waitUntilCompleted();

        if self.cmd_buf.status() == MTLCommandBufferStatus::Error {
            let message = self
                .cmd_buf
                .error()
                .map(|e| e.localizedDescription().to_string())
                .unwrap_or_else(|| "unknown command buffer error".to_owned());
            return Err(Error::CommandBuffer { message });
        }
        Ok((self.cmd_buf.GPUEndTime() - self.cmd_buf.GPUStartTime()).max(0.0))
    }
}

impl Drop for CommandBatch<'_> {
    fn drop(&mut self) {
        if !self.finished {
            // An abandoned batch (early `?` return) must still end its
            // encoder before the command buffer is released; the
            // never-committed command buffer is then simply discarded.
            self.encoder.endEncoding();
        }
    }
}

impl std::fmt::Debug for CommandBatch<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommandBatch")
            .field("resources", &self.resources.len())
            .field("finished", &self.finished)
            .finish_non_exhaustive()
    }
}
