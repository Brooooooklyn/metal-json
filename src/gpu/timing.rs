//! Coarse parse-phase timing for the GPU pipeline, behind the `timing`
//! feature.
//!
//! The M5 question this answers: **where does the wall time of a large
//! parse go** — per-command-buffer GPU execution (`GPUEndTime −
//! GPUStartTime`) vs the CPU-side gaps between them (encode/commit/sync
//! latency, exact-size allocations, zero fills, snapshot copies, fixup
//! patches, tape copy-out)?
//!
//! With the feature enabled, every [`GpuPipeline::run`]
//! (and the `Parser::parse` copy-out around it) records one `Phase` per
//! pipeline segment into a thread-local; `take_parse_timings` hands the
//! finished list to the caller (see `examples/parse_breakdown.rs`).
//! Without the feature every helper is an inlined no-op and the pipeline
//! pays nothing.
//!
//! This is deliberately *coarse* — whole command buffers, not per-kernel
//! counters. Per-kernel `MTLCounterSampleBuffer` breakdowns can layer on
//! top once the phase level points at a GPU-bound command buffer; today's
//! wall is CPU-side (copy-out / zero fills / snapshot copies), which this
//! granularity already exposes.
//!
//! [`GpuPipeline::run`]: super::pipeline::GpuPipeline::run

/// One timed pipeline segment.
#[cfg(feature = "timing")]
#[derive(Debug, Clone, Copy)]
pub struct Phase {
    /// Segment name (stable identifiers, e.g. `"cb1 (K1-K4)"`).
    pub name: &'static str,
    /// Wall-clock seconds the CPU spent in the segment (for command-buffer
    /// segments: encode + commit + `waitUntilCompleted`).
    pub wall_seconds: f64,
    /// GPU execution seconds within the segment (`GPUEndTime −
    /// GPUStartTime` of its command buffer); 0 for CPU-only segments. The
    /// difference `wall − gpu` is the CPU-side gap.
    pub gpu_seconds: f64,
}

/// The phases of one full parse, in execution order.
#[cfg(feature = "timing")]
#[derive(Debug, Clone, Default)]
pub struct ParseTimings {
    /// Execution-order segments; names are unique within one parse.
    pub phases: Vec<Phase>,
}

#[cfg(feature = "timing")]
mod imp {
    use super::{ParseTimings, Phase};
    use std::cell::RefCell;
    use std::time::Instant;

    thread_local! {
        static CURRENT: RefCell<Option<ParseTimings>> = const { RefCell::new(None) };
        static KERNELS: RefCell<Vec<(String, f64)>> = const { RefCell::new(Vec::new()) };
    }

    /// Record one kernel's GPU execution time (the `CommandBatch`
    /// split-kernel measurement mode, `METAL_JSON_SPLIT_KERNELS=1`).
    /// Appended in dispatch order; duplicate names (multi-pass sorts) stay
    /// separate entries.
    pub(crate) fn record_kernel(name: &str, gpu_seconds: f64) {
        KERNELS.with(|k| k.borrow_mut().push((name.to_owned(), gpu_seconds)));
    }

    /// Take the per-kernel GPU times recorded since the last take (empty
    /// unless `METAL_JSON_SPLIT_KERNELS=1` split the command buffers).
    #[must_use]
    pub fn take_kernel_timings() -> Vec<(String, f64)> {
        KERNELS.with(|k| std::mem::take(&mut *k.borrow_mut()))
    }

    /// A running phase stopwatch (wraps the start `Instant`).
    #[derive(Debug)]
    pub struct PhaseTimer(Instant);

    /// Reset the thread-local recording (called at the top of every
    /// `GpuPipeline::run`); the previous parse's timings are discarded if
    /// never taken.
    pub(crate) fn begin_parse() {
        CURRENT.with(|c| *c.borrow_mut() = Some(ParseTimings::default()));
    }

    /// Start timing a phase.
    pub(crate) fn start() -> PhaseTimer {
        PhaseTimer(Instant::now())
    }

    /// Finish a phase: wall time from the timer, GPU seconds from the
    /// command buffer (0 for CPU-only phases).
    pub(crate) fn record(name: &'static str, timer: PhaseTimer, gpu_seconds: f64) {
        let wall_seconds = timer.0.elapsed().as_secs_f64();
        CURRENT.with(|c| {
            if let Some(t) = c.borrow_mut().as_mut() {
                t.phases.push(Phase {
                    name,
                    wall_seconds,
                    gpu_seconds,
                });
            }
        });
    }

    /// Take the timings of the most recent parse on this thread (`None`
    /// when no parse ran since the last take).
    #[must_use]
    pub fn take_parse_timings() -> Option<ParseTimings> {
        CURRENT.with(|c| c.borrow_mut().take())
    }
}

#[cfg(not(feature = "timing"))]
mod imp {
    /// No-op stand-in; the optimizer erases every call.
    #[derive(Debug)]
    pub struct PhaseTimer;

    #[inline(always)]
    pub(crate) fn begin_parse() {}

    #[inline(always)]
    pub(crate) fn start() -> PhaseTimer {
        PhaseTimer
    }

    #[inline(always)]
    pub(crate) fn record(name: &'static str, timer: PhaseTimer, gpu_seconds: f64) {
        let _ = (name, timer, gpu_seconds);
    }
}

// `PhaseTimer` itself stays module-internal: call sites only ever hold it
// through `start()`'s return value and hand it straight back to `record`.
pub(crate) use imp::{begin_parse, record, start};

#[cfg(feature = "timing")]
pub(crate) use imp::record_kernel;

#[cfg(feature = "timing")]
pub use imp::{take_kernel_timings, take_parse_timings};

#[cfg(all(test, feature = "timing"))]
mod tests {
    use super::*;

    #[test]
    fn record_take_roundtrip_and_reset() {
        begin_parse();
        record("a", start(), 0.25);
        record("b", start(), 0.0);
        let t = take_parse_timings().expect("recording was begun");
        assert_eq!(t.phases.len(), 2);
        assert_eq!(t.phases[0].name, "a");
        assert!((t.phases[0].gpu_seconds - 0.25).abs() < 1e-12);
        assert_eq!(t.phases[1].name, "b");
        assert!(t.phases[1].wall_seconds >= 0.0);
        // Taken: gone until the next begin_parse.
        assert!(take_parse_timings().is_none());
        // Records outside a parse are dropped, not panics.
        record("stray", start(), 0.0);
        assert!(take_parse_timings().is_none());
        // A new parse starts clean.
        begin_parse();
        record("c", start(), 0.0);
        assert_eq!(take_parse_timings().unwrap().phases.len(), 1);
    }
}
