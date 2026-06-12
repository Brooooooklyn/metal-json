//! The M5 buffer pool: steady-state parses make **zero** large allocations.
//!
//! A [`ScratchPool`] is a flat free list of shared-storage [`GpuBuffer`]s.
//! Every buffer the GPU pipeline needs — stage-1/2/3 scratch, the K10/K11
//! scratch, and the tape / string buffers themselves — is
//! [`checkout`](ScratchPool::checkout)-ed per parse and returned with
//! [`put_back`](ScratchPool::put_back): the scratch at the end of the
//! parse, the tape/string buffers when the [`Document`](crate::Document)
//! that owns them drops (it holds an `Arc<ScratchPool>` handle, so the
//! buffers can never return — let alone be reused — while the document is
//! alive).
//!
//! # Sizing policy: grow-and-keep
//!
//! Capacities are rounded up to whole 16 KiB pages and **kept**: the pool
//! never shrinks or frees. `checkout` picks the smallest free buffer whose
//! capacity fits (best fit) and re-aims its logical length; a miss
//! allocates a fresh page-rounded buffer that joins the pool when returned.
//! Parsing the same-shaped input repeatedly therefore reaches a steady
//! state where every checkout is a hit (this is exactly what criterion
//! measures); parsing a larger input grows the pooled capacities once and
//! keeps them. Callers that interleave wildly different input sizes pay the
//! high-water-mark footprint — drop the `Parser` (and its documents) to
//! release everything.
//!
//! # Contents are garbage by contract
//!
//! A pooled buffer keeps its previous user's bytes. This is the documented
//! [`GpuBuffer::alloc`] non-guarantee made *real*: every zero/init
//! precondition in the pipeline is established explicitly by the code that
//! needs it (`Stage1Buffers`' chunk counts + header, the K10/K11 atomic
//! counters), and everything else is fully overwritten before it is read.
//! [`poison_free_buffers`](ScratchPool::poison_free_buffers) lets tests pin
//! that invariant by pre-filling the free list with a poison byte.
//!
//! Internal/unstable: exposed publicly so integration tests can drive the
//! pool directly (like [`crate::metal`] / [`crate::stage`]), but not part
//! of the supported API surface.

use std::sync::Mutex;

use crate::error::Result;
use crate::metal::{GpuBuffer, MetalContext, PAGE_SIZE};

/// A shared free list of reusable GPU buffers. See the module docs for the
/// sizing policy and the contents contract.
#[derive(Debug, Default)]
pub struct ScratchPool {
    free: Mutex<Vec<GpuBuffer>>,
}

impl ScratchPool {
    /// An empty pool.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Check out a buffer with logical length `bytes` (contents
    /// unspecified). Best-fit reuse from the free list; a miss allocates a
    /// fresh buffer with page-rounded capacity. The Metal allocation on a
    /// miss happens outside the lock.
    ///
    /// # Errors
    ///
    /// [`crate::Error::BufferAlloc`] if the device is out of memory.
    pub fn checkout(&self, ctx: &MetalContext, bytes: usize) -> Result<GpuBuffer> {
        let capacity = bytes.next_multiple_of(PAGE_SIZE).max(PAGE_SIZE);
        {
            let mut free = self.free.lock().expect("pool lock never poisoned");
            let mut best: Option<(usize, usize)> = None; // (index, capacity)
            for (i, buf) in free.iter().enumerate() {
                let cap = buf.capacity();
                if cap >= capacity && best.is_none_or(|(_, c)| cap < c) {
                    best = Some((i, cap));
                }
            }
            if let Some((i, _)) = best {
                let mut buf = free.swap_remove(i);
                buf.set_len(bytes);
                return Ok(buf);
            }
        }
        let mut buf = GpuBuffer::alloc(ctx, capacity)?;
        buf.set_len(bytes);
        Ok(buf)
    }

    /// Return a buffer to the free list (grow-and-keep: capacity is
    /// retained forever, contents are left as-is).
    pub fn put_back(&self, buf: GpuBuffer) {
        self.free
            .lock()
            .expect("pool lock never poisoned")
            .push(buf);
    }

    /// Number of buffers currently in the free list (tests / diagnostics).
    #[must_use]
    pub fn free_len(&self) -> usize {
        self.free.lock().expect("pool lock never poisoned").len()
    }

    /// Fill every free buffer's **whole capacity** with `byte` — the poison
    /// hook for tests pinning the "pooled contents are garbage" contract
    /// (parse → poison → parse again must produce identical documents).
    pub fn poison_free_buffers(&self, byte: u8) {
        let mut free = self.free.lock().expect("pool lock never poisoned");
        for buf in free.iter_mut() {
            let len = buf.len();
            buf.set_len(buf.capacity());
            buf.contents_mut().fill(byte);
            buf.set_len(len);
        }
    }
}

/// Where pipeline buffers come from: a fresh Metal allocation per buffer
/// (the per-milestone test runners) or a [`ScratchPool`] checkout (the
/// production parse path). Passing this down keeps one constructor per
/// buffer set instead of parallel pooled/direct variants.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Alloc<'a> {
    /// `GpuBuffer::alloc` per buffer (exact capacity, freed on drop).
    Direct,
    /// Checkout from (and, for scratch, eventually return to) the pool.
    Pool(&'a ScratchPool),
}

impl Alloc<'_> {
    /// Produce a buffer with logical length `bytes` (contents unspecified
    /// either way — [`GpuBuffer::alloc`]'s documented non-guarantee).
    pub(crate) fn buffer(&self, ctx: &MetalContext, bytes: usize) -> Result<GpuBuffer> {
        match self {
            Alloc::Direct => GpuBuffer::alloc(ctx, bytes),
            Alloc::Pool(pool) => pool.checkout(ctx, bytes),
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// Steady state: returning a buffer and checking out the same size hits
    /// the free list (no growth); a larger request misses and grows.
    #[test]
    fn checkout_reuses_returned_capacity_best_fit() {
        let Some(ctx) = ctx_or_skip("checkout_reuses_returned_capacity_best_fit") else {
            return;
        };
        let pool = ScratchPool::new();
        let small = pool.checkout(&ctx, 100).unwrap();
        let big = pool.checkout(&ctx, PAGE_SIZE + 1).unwrap();
        assert_eq!(small.len(), 100);
        assert_eq!(small.capacity(), PAGE_SIZE);
        assert_eq!(big.capacity(), 2 * PAGE_SIZE);
        pool.put_back(small);
        pool.put_back(big);
        assert_eq!(pool.free_len(), 2);

        // Best fit: a 50-byte request takes the 1-page buffer, not 2 pages.
        let again = pool.checkout(&ctx, 50).unwrap();
        assert_eq!(again.len(), 50);
        assert_eq!(again.capacity(), PAGE_SIZE);
        assert_eq!(pool.free_len(), 1);

        // The remaining (2-page) buffer serves anything that fits it.
        let two = pool.checkout(&ctx, PAGE_SIZE + 5).unwrap();
        assert_eq!(two.capacity(), 2 * PAGE_SIZE);
        assert_eq!(pool.free_len(), 0);
    }

    /// Pooled contents are garbage by contract: a poisoned free buffer
    /// comes back with the poison intact (no hidden zeroing anywhere).
    #[test]
    fn poison_survives_checkout() {
        let Some(ctx) = ctx_or_skip("poison_survives_checkout") else {
            return;
        };
        let pool = ScratchPool::new();
        let buf = pool.checkout(&ctx, 64).unwrap();
        pool.put_back(buf);
        pool.poison_free_buffers(0xDB);
        let buf = pool.checkout(&ctx, 32).unwrap();
        assert!(buf.contents().iter().all(|&b| b == 0xDB));
    }

    /// The pool is shared across threads (the Document-drop return path).
    #[test]
    fn pool_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ScratchPool>();
        assert_send_sync::<std::sync::Arc<ScratchPool>>();
    }
}
