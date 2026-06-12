//! `MTLBuffer` wrapper. Everything is `StorageModeShared`: on Apple Silicon
//! unified memory the CPU and GPU see the same pages, so "upload"/"download"
//! are plain slice accesses.

use core::ffi::c_void;
use core::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use super::MetalContext;
use crate::error::{Error, Result};

/// Apple Silicon VM page size; `newBufferWithBytesNoCopy` requires
/// page-aligned memory in page-multiple lengths.
pub const PAGE_SIZE: usize = 16384;

/// A shared-storage GPU buffer with a logical length in bytes.
///
/// The logical length may be smaller than the underlying `MTLBuffer`
/// allocation (which is page-granular for no-copy buffers); slice accessors
/// only ever expose the logical `len` bytes.
pub struct GpuBuffer {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    len: usize,
}

impl GpuBuffer {
    /// Allocate a shared buffer of `len` bytes.
    ///
    /// Contents are **unspecified**: freshly created `MTLBuffer`s happen to
    /// arrive zero-filled today (the OS hands out zeroed VM pages), but that
    /// is an allocator accident, not part of this contract — a pooled/reused
    /// buffer (planned for M5) keeps its previous contents. Callers that
    /// need a known starting state must fill it explicitly (as
    /// [`Stage1Buffers::new`](crate::stage::Stage1Buffers::new) does for the
    /// buffers its kernels accumulate into).
    pub fn alloc(ctx: &MetalContext, len: usize) -> Result<Self> {
        // Metal returns nil for zero-length buffers; keep the allocation
        // non-empty and let `len` carry the logical size.
        let alloc_len = len.max(1);
        let raw = ctx
            .device()
            .newBufferWithLength_options(alloc_len, MTLResourceOptions::StorageModeShared)
            .ok_or(Error::BufferAlloc { bytes: alloc_len })?;
        Ok(Self { raw, len })
    }

    /// Wrap caller-owned, page-aligned memory zero-copy
    /// (`newBufferWithBytesNoCopy`). This is how mmap'd input files become
    /// GPU-visible without any copy on unified memory.
    ///
    /// `len` is the logical byte length; the wrapped region is
    /// `len.next_multiple_of(PAGE_SIZE)` bytes starting at `ptr`.
    ///
    /// # Safety
    ///
    /// The caller must guarantee, for the entire lifetime of the returned
    /// `GpuBuffer` (and of any command buffer using it):
    ///
    /// - `ptr` is aligned to [`PAGE_SIZE`] (16384 bytes);
    /// - the region `ptr .. ptr + len.next_multiple_of(PAGE_SIZE)` is one
    ///   valid, readable + writable allocation (e.g. from `mmap` or
    ///   page-aligned `vm_allocate`/`posix_memalign`);
    /// - the memory stays alive and is not unmapped/freed — no deallocator
    ///   block is registered, ownership stays with the caller;
    /// - the CPU does not mutate the region while a command buffer that
    ///   accesses it is executing.
    pub unsafe fn from_page_aligned(
        ctx: &MetalContext,
        ptr: NonNull<u8>,
        len: usize,
    ) -> Result<Self> {
        // Layout problems are checked at runtime and reported as structured
        // errors; the `unsafe` contract covers only what cannot be checked
        // (allocation validity, lifetime, and GPU/CPU mutation exclusion).
        if !(ptr.as_ptr() as usize).is_multiple_of(PAGE_SIZE) {
            return Err(Error::InvalidBufferLayout {
                message: format!(
                    "pointer {:p} is not {PAGE_SIZE}-byte page-aligned",
                    ptr.as_ptr()
                ),
            });
        }
        let wrapped_len = len
            .checked_next_multiple_of(PAGE_SIZE)
            .ok_or_else(|| Error::InvalidBufferLayout {
                message: format!("length {len} overflows when rounded up to a page multiple"),
            })?
            .max(PAGE_SIZE);
        if wrapped_len > isize::MAX as usize {
            return Err(Error::InvalidBufferLayout {
                message: format!("wrapped length {wrapped_len} exceeds isize::MAX"),
            });
        }
        // SAFETY: caller upholds the invariants documented above; passing no
        // deallocator means Metal never frees the memory.
        let raw = unsafe {
            ctx.device()
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    ptr.cast::<c_void>(),
                    wrapped_len,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
        }
        .ok_or(Error::BufferAlloc { bytes: wrapped_len })?;
        Ok(Self { raw, len })
    }

    /// Logical length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Capacity in bytes: the length of the underlying `MTLBuffer`
    /// allocation (page-granular for pooled / no-copy buffers). Always
    /// `>= len()`.
    pub(crate) fn capacity(&self) -> usize {
        self.raw.length()
    }

    /// Re-aim the logical length at `len` bytes of the existing allocation
    /// (the buffer-pool reuse path: capacity is kept, the logical view
    /// shrinks or grows within it). Contents are whatever the previous user
    /// left — exactly the [`alloc`](Self::alloc) non-guarantee.
    ///
    /// # Panics
    /// If `len` exceeds [`capacity`](Self::capacity) (programmer error; the
    /// pool only hands out buffers it checked).
    pub(crate) fn set_len(&mut self, len: usize) {
        assert!(
            len <= self.capacity(),
            "set_len({len}) exceeds the {}-byte allocation",
            self.capacity()
        );
        self.len = len;
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// CPU view of the buffer contents.
    ///
    /// Soundness: a kernel that writes this buffer must bind it
    /// [`Binding::ReadWrite`](super::Binding::ReadWrite), which takes
    /// `&mut GpuBuffer` — the borrow checker therefore guarantees no slice
    /// returned here is live across a dispatch that mutates the buffer.
    pub fn contents(&self) -> &[u8] {
        // SAFETY: shared-storage buffer; `contents()` is valid for at least
        // `length()` >= `self.len` bytes for the lifetime of `self.raw`.
        unsafe { core::slice::from_raw_parts(self.base_ptr(), self.len) }
    }

    /// Mutable CPU view of the buffer contents. Same synchronization caveat
    /// as [`contents`](Self::contents); `&mut self` rules out CPU-side
    /// aliasing.
    pub fn contents_mut(&mut self) -> &mut [u8] {
        // SAFETY: as `contents`, plus exclusive `&mut self`.
        unsafe { core::slice::from_raw_parts_mut(self.base_ptr(), self.len) }
    }

    /// Typed view: the buffer as `&[T]`.
    ///
    /// # Panics
    /// If the logical length is not a multiple of `size_of::<T>()` or the
    /// base pointer is not aligned for `T` (programmer error).
    pub fn as_slice<T: Pod>(&self) -> &[T] {
        let (ptr, n) = self.typed_parts::<T>();
        // SAFETY: Pod types tolerate any bit pattern; size/alignment checked.
        unsafe { core::slice::from_raw_parts(ptr as *const T, n) }
    }

    /// Typed mutable view: the buffer as `&mut [T]`.
    ///
    /// # Panics
    /// Same conditions as [`as_slice`](Self::as_slice).
    pub fn as_mut_slice<T: Pod>(&mut self) -> &mut [T] {
        let (ptr, n) = self.typed_parts::<T>();
        // SAFETY: as `as_slice`, plus exclusive `&mut self`.
        unsafe { core::slice::from_raw_parts_mut(ptr, n) }
    }

    /// Copy `src` into the start of the buffer.
    ///
    /// # Panics
    /// If `src` does not fit.
    pub fn write_from<T: Pod>(&mut self, src: &[T]) {
        let dst = self.as_mut_slice::<T>();
        dst[..src.len()].copy_from_slice(src);
    }

    pub(crate) fn raw(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.raw
    }

    fn base_ptr(&self) -> *mut u8 {
        self.raw.contents().cast::<u8>().as_ptr()
    }

    fn typed_parts<T: Pod>(&self) -> (*mut T, usize) {
        let size = size_of::<T>();
        assert!(
            self.len.is_multiple_of(size),
            "GpuBuffer length {} is not a multiple of size_of::<{}>() = {}",
            self.len,
            core::any::type_name::<T>(),
            size
        );
        let ptr = self.base_ptr();
        assert!(
            (ptr as usize).is_multiple_of(align_of::<T>()),
            "GpuBuffer base pointer is not aligned for {}",
            core::any::type_name::<T>()
        );
        (ptr.cast::<T>(), self.len / size)
    }
}

// SAFETY (audited, M5): `GpuBuffer` wraps a retained `MTLBuffer` plus a
// plain `usize`. Sending or sharing it across threads is sound because:
//
// - Metal framework objects are documented thread-safe ("Metal and
//   Threading": all objects except command *encoders* — which this type
//   never holds — may be used from multiple threads), and ObjC
//   retain/release is itself thread-safe; the missing auto-traits on
//   `Retained<ProtocolObject<dyn MTLBuffer>>` are a bindings limitation,
//   not a Metal one (the wgpu Metal backend relies on the same property).
// - CPU-side aliasing of the shared-storage contents is governed entirely
//   by the `&self`/`&mut self` borrow discipline of the accessors
//   (`contents`/`as_slice` vs `contents_mut`/`as_mut_slice`), which the
//   borrow checker enforces across threads exactly as on one thread.
// - GPU-side hazards are governed by the `CommandBatch` borrow model
//   (buffers a dispatch writes are held `&mut` until the synchronous
//   `commit_and_wait` returns), which is thread-agnostic.
//
// This is what lets `Document` own GPU-backed tape/string buffers and stay
// `Send + Sync`, and lets the parser's buffer pool be shared via
// `Mutex<Vec<GpuBuffer>>` (drops, and therefore pool returns, can happen on
// any thread).
unsafe impl Send for GpuBuffer {}
// SAFETY: see the `Send` audit above; `&GpuBuffer` only exposes read access
// to the shared-storage contents.
unsafe impl Sync for GpuBuffer {}

impl std::fmt::Debug for GpuBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBuffer").field("len", &self.len).finish()
    }
}

/// Plain-old-data marker for typed buffer views: any bit pattern is a valid
/// value, no padding, no pointers. Sealed; implemented only for the primitive
/// types the kernels exchange.
///
/// # Safety
/// Implementors must be valid for every bit pattern and contain no padding.
pub unsafe trait Pod: Copy + 'static + private::Sealed {}

mod private {
    pub trait Sealed {}
}

macro_rules! impl_pod {
    ($($t:ty),*) => {
        $(
            impl private::Sealed for $t {}
            // SAFETY: primitive integer/float types are valid for all bit
            // patterns and have no padding.
            unsafe impl Pod for $t {}
        )*
    };
}

impl_pod!(u8, u16, u32, u64, i8, i16, i32, i64, f32);

impl private::Sealed for super::MjHeader {}
// SAFETY: `MjHeader` is a #[repr(C)] struct of eight u64 fields: no padding,
// no pointers, and every bit pattern is a valid value.
unsafe impl Pod for super::MjHeader {}
