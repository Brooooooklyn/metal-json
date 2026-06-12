//! [`AlignedInput`]: caller-held input the GPU consumes **zero-copy**.
//!
//! `MTLBuffer newBufferWithBytesNoCopy` requires 16 KiB page alignment and
//! page-multiple lengths; the stage-1 kernels additionally rely on the
//! bytes between `len` and the next 64-byte word boundary classifying as
//! whitespace (the same ASCII-space padding `Stage1Buffers::new` writes on
//! the copied path — it keeps trailing scalars terminated and truncated
//! UTF-8 sequences failing at the reference offset). `AlignedInput` bakes
//! all three invariants into construction, so
//! [`Parser::parse_aligned`](crate::Parser::parse_aligned) can wrap it for
//! the GPU without copying a single input byte.

use core::ops::Deref;
use core::ptr::NonNull;

use crate::metal::PAGE_SIZE;

/// An owned, page-aligned, space-padded copy of a JSON document — the input
/// layout the zero-copy GPU path consumes directly. Build it once (outside
/// any timed region), parse from it as many times as needed via
/// [`Parser::parse_aligned`](crate::Parser::parse_aligned).
///
/// Invariants (established by construction, relied upon by the parser):
///
/// - the allocation starts 16 KiB-aligned and spans a whole number of
///   16 KiB pages (`MTLBuffer bytesNoCopy` layout);
/// - bytes `len..capacity` are ASCII spaces (kernel tail-word padding).
pub struct AlignedInput {
    ptr: NonNull<u8>,
    len: usize,
    capacity: usize,
}

impl AlignedInput {
    /// Copy `bytes` into a fresh page-aligned allocation with a space-filled
    /// tail.
    ///
    /// # Panics
    ///
    /// Panics if the allocation fails (matching `Vec` semantics).
    #[must_use]
    pub fn from_slice(bytes: &[u8]) -> Self {
        let capacity = bytes.len().next_multiple_of(PAGE_SIZE).max(PAGE_SIZE);
        let layout = std::alloc::Layout::from_size_align(capacity, PAGE_SIZE)
            .expect("page-aligned layout is always valid");
        // SAFETY: `layout` has nonzero size.
        let ptr = unsafe { std::alloc::alloc(layout) };
        let Some(ptr) = NonNull::new(ptr) else {
            std::alloc::handle_alloc_error(layout);
        };
        // SAFETY: the allocation is `capacity` bytes; the two writes cover
        // disjoint ranges `[0, len)` and `[len, capacity)`.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.as_ptr(), bytes.len());
            std::ptr::write_bytes(ptr.as_ptr().add(bytes.len()), b' ', capacity - bytes.len());
        }
        Self {
            ptr,
            len: bytes.len(),
            capacity,
        }
    }

    /// The document bytes (without the padding tail).
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `[0, len)` is initialized by construction and owned.
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Document length in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// True for an empty document.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The page-aligned base pointer (the whole `capacity()`-byte region is
    /// valid and initialized).
    pub(crate) fn base_ptr(&self) -> NonNull<u8> {
        self.ptr
    }
}

impl Deref for AlignedInput {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl Drop for AlignedInput {
    fn drop(&mut self) {
        let layout = std::alloc::Layout::from_size_align(self.capacity, PAGE_SIZE)
            .expect("layout validated at construction");
        // SAFETY: allocated in `from_slice` with this exact layout.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), layout) };
    }
}

// SAFETY: `AlignedInput` uniquely owns its allocation and has no interior
// mutability; it is an owned byte buffer like `Vec<u8>`.
unsafe impl Send for AlignedInput {}
// SAFETY: shared access only exposes `&[u8]` reads.
unsafe impl Sync for AlignedInput {}

impl std::fmt::Debug for AlignedInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlignedInput")
            .field("len", &self.len)
            .field("capacity", &self.capacity)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_slice_is_aligned_padded_and_roundtrips() {
        let input = AlignedInput::from_slice(b"{\"a\":1}");
        assert_eq!(&*input, b"{\"a\":1}");
        assert_eq!(input.len(), 7);
        assert!(!input.is_empty());
        assert_eq!(input.base_ptr().as_ptr() as usize % PAGE_SIZE, 0);
        assert_eq!(input.capacity % PAGE_SIZE, 0);
        // The whole tail is ASCII spaces (the kernel padding invariant).
        let full =
            unsafe { core::slice::from_raw_parts(input.base_ptr().as_ptr(), input.capacity) };
        assert!(full[input.len()..].iter().all(|&b| b == b' '));
    }

    #[test]
    fn empty_input_still_allocates_one_padded_page() {
        let input = AlignedInput::from_slice(b"");
        assert!(input.is_empty());
        assert_eq!(input.capacity, PAGE_SIZE);
        let full =
            unsafe { core::slice::from_raw_parts(input.base_ptr().as_ptr(), input.capacity) };
        assert!(full.iter().all(|&b| b == b' '));
    }

    #[test]
    fn page_multiple_inputs_get_no_extra_page() {
        let bytes = vec![b'7'; PAGE_SIZE];
        let input = AlignedInput::from_slice(&bytes);
        assert_eq!(input.capacity, PAGE_SIZE);
        assert_eq!(input.len(), PAGE_SIZE);
    }

    #[test]
    fn aligned_input_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<AlignedInput>();
    }
}
