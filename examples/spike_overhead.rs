//! SPIKE C (M0): fixed pipeline overhead.
//!
//! The parser plan executes 3 command buffers with ~14 total compute
//! dispatches and 2 CPU syncs (`waitUntilCompleted` between CBs). This spike
//! measures what that shape costs in fixed overhead on this machine:
//!
//! 1. empty command buffer commit+wait round trip;
//! 2. one CB with 14 trivial dispatches (tiny kernel, 1 threadgroup);
//! 3. the planned shape: CB1(4)+wait, CB2(3)+wait, CB3(7)+wait, tiny kernels;
//! 4. same shape, realistic 16Mi-thread grids, two variants:
//!    4a. no memory traffic (pure scheduling);
//!    4b. each dispatch reads+writes a 64 MiB dummy buffer.
//!
//! Methodology: 3 warmup runs, then median of 20 timed runs. GPU time is the
//! sum of command-buffer `GPUEndTime - GPUStartTime` deltas; wall time is host
//! `Instant` around encode→commit→wait (what a parse actually pays). Falls
//! back to wall-only if GPU timestamps are unavailable.
//!
//! Run: `cargo run --release --example spike_overhead`
//!
//! Self-contained on purpose: compiles its own MSL via
//! `newLibraryWithSource:` and encodes raw command buffers, because the spike
//! needs multi-dispatch CBs and GPU timestamps that the M0 wrapper layer does
//! not (and should not yet) expose.

use core::ffi::c_void;
use core::ptr::NonNull;
use std::time::Instant;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLibrary, MTLResourceOptions, MTLSize,
};

/// 64 MiB dummy buffer = 16Mi u32 elements, one thread each.
const BIG_ELEMS: usize = 16 * 1024 * 1024;
const THREADGROUP_SIZE: usize = 256;
/// Dispatches per command buffer in the planned pipeline shape.
const SHAPE: [usize; 3] = [4, 3, 7];
const WARMUP_RUNS: usize = 3;
const TIMED_RUNS: usize = 20;

const MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Tiny kernel: 1 threadgroup of near-zero work. Thread 0 bumps a sink word so
// the dispatch cannot be elided.
kernel void spike_tiny(
    device uint* sink [[buffer(0)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid == 0) {
        sink[0] = sink[0] + 1u;
    }
}

// Realistic grid size, no memory traffic: isolates the cost of scheduling a
// 16Mi-thread dispatch from the cost of moving bytes.
kernel void spike_nop_grid(
    device uint* sink [[buffer(0)]],
    constant uint& element_count [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= element_count) {
        return;
    }
    if (gid == 0) {
        sink[0] = sink[0] + 1u;
    }
}

// Realistic grid size, trivial body with real traffic: one u32 read + write
// per thread over the 64 MiB dummy buffer.
kernel void spike_touch(
    device uint* data [[buffer(0)]],
    constant uint& element_count [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= element_count) {
        return;
    }
    data[gid] = data[gid] + 1u;
}
"#;

type Pso = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type Buf = Retained<ProtocolObject<dyn MTLBuffer>>;
/// One pipeline stage: encodes its dispatches into the given command buffer.
type Stage<'a> = &'a dyn Fn(&ProtocolObject<dyn MTLCommandBuffer>);

fn mtl_size(width: usize) -> MTLSize {
    MTLSize {
        width,
        height: 1,
        depth: 1,
    }
}

struct Gpu {
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    tiny: Pso,
    nop_grid: Pso,
    touch: Pso,
    /// 4-byte sink for the tiny / nop kernels.
    sink: Buf,
    /// 64 MiB dummy data buffer.
    big: Buf,
}

impl Gpu {
    fn new() -> Self {
        let device = MTLCreateSystemDefaultDevice().expect("no Metal device");
        println!(
            "device: {}  (BIG buffer: {} MiB, {} threads/dispatch at size)",
            device.name(),
            BIG_ELEMS * size_of::<u32>() / (1024 * 1024),
            BIG_ELEMS
        );
        let queue = device.newCommandQueue().expect("newCommandQueue");
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(MSL), None)
            .unwrap_or_else(|e| panic!("MSL compile failed: {}", e.localizedDescription()));
        let pso = |name: &str| -> Pso {
            let f = library
                .newFunctionWithName(&NSString::from_str(name))
                .unwrap_or_else(|| panic!("kernel `{name}` not found"));
            device
                .newComputePipelineStateWithFunction_error(&f)
                .unwrap_or_else(|e| {
                    panic!("PSO `{name}` failed: {}", e.localizedDescription())
                })
        };
        let tiny = pso("spike_tiny");
        let nop_grid = pso("spike_nop_grid");
        let touch = pso("spike_touch");
        let sink = device
            .newBufferWithLength_options(size_of::<u32>(), MTLResourceOptions::StorageModeShared)
            .expect("sink buffer");
        let big = device
            .newBufferWithLength_options(
                BIG_ELEMS * size_of::<u32>(),
                MTLResourceOptions::StorageModeShared,
            )
            .expect("big buffer");
        Self {
            queue,
            tiny,
            nop_grid,
            touch,
            sink,
            big,
        }
    }

    /// Encode `n` one-threadgroup dispatches of the tiny kernel. Pipeline
    /// state and buffer are re-bound per dispatch to mimic 14 distinct
    /// kernels.
    fn encode_tiny(&self, cb: &ProtocolObject<dyn MTLCommandBuffer>, n: usize) {
        let enc = cb.computeCommandEncoder().expect("computeCommandEncoder");
        for _ in 0..n {
            enc.setComputePipelineState(&self.tiny);
            // SAFETY: `sink` outlives the synchronous commit+wait below.
            unsafe { enc.setBuffer_offset_atIndex(Some(&self.sink), 0, 0) };
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                mtl_size(1),
                mtl_size(THREADGROUP_SIZE),
            );
        }
        enc.endEncoding();
    }

    /// Encode `n` full-grid (16Mi threads) dispatches of `pso` over the big
    /// buffer.
    fn encode_grid(&self, cb: &ProtocolObject<dyn MTLCommandBuffer>, pso: &Pso, n: usize) {
        let enc = cb.computeCommandEncoder().expect("computeCommandEncoder");
        let element_count = BIG_ELEMS as u32;
        for _ in 0..n {
            enc.setComputePipelineState(pso);
            // SAFETY: buffers outlive the synchronous commit+wait below;
            // setBytes copies `element_count` into the command stream.
            unsafe {
                let buf0: &ProtocolObject<dyn MTLBuffer> =
                    if core::ptr::eq(&**pso, &*self.nop_grid) {
                        &self.sink
                    } else {
                        &self.big
                    };
                enc.setBuffer_offset_atIndex(Some(buf0), 0, 0);
                enc.setBytes_length_atIndex(
                    NonNull::from(&element_count).cast::<c_void>(),
                    size_of::<u32>(),
                    1,
                );
            }
            enc.dispatchThreads_threadsPerThreadgroup(
                mtl_size(BIG_ELEMS),
                mtl_size(THREADGROUP_SIZE),
            );
        }
        enc.endEncoding();
    }

    /// Create+encode+commit+wait one CB per stage; return (wall seconds,
    /// summed GPU seconds if timestamps were available for every CB).
    fn run_sequence(&self, stages: &[Stage<'_>]) -> (f64, Option<f64>) {
        let t0 = Instant::now();
        let mut gpu = Some(0.0_f64);
        for stage in stages {
            let cb = self.queue.commandBuffer().expect("commandBuffer");
            stage(&cb);
            cb.commit();
            cb.waitUntilCompleted();
            assert_ne!(
                cb.status(),
                MTLCommandBufferStatus::Error,
                "command buffer failed: {:?}",
                cb.error().map(|e| e.localizedDescription().to_string())
            );
            let start = cb.GPUStartTime();
            let end = cb.GPUEndTime();
            gpu = match (gpu, end > start && start > 0.0) {
                (Some(acc), true) => Some(acc + (end - start)),
                _ => None, // timestamps unavailable -> wall-only fallback
            };
        }
        (t0.elapsed().as_secs_f64(), gpu)
    }
}

#[derive(Clone, Copy)]
struct Stats {
    wall_us: f64,
    /// Median GPU time; None if any run lacked timestamps.
    gpu_us: Option<f64>,
    wall_min_us: f64,
    wall_max_us: f64,
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.total_cmp(b));
    let n = samples.len();
    if n.is_multiple_of(2) {
        (samples[n / 2 - 1] + samples[n / 2]) / 2.0
    } else {
        samples[n / 2]
    }
}

fn measure(gpu: &Gpu, stages: &[Stage<'_>]) -> Stats {
    for _ in 0..WARMUP_RUNS {
        gpu.run_sequence(stages);
    }
    let mut walls = Vec::with_capacity(TIMED_RUNS);
    let mut gpus = Vec::with_capacity(TIMED_RUNS);
    let mut gpu_ok = true;
    for _ in 0..TIMED_RUNS {
        let (wall, gpu_time) = gpu.run_sequence(stages);
        walls.push(wall * 1e6);
        match gpu_time {
            Some(g) => gpus.push(g * 1e6),
            None => gpu_ok = false,
        }
    }
    let wall_min_us = walls.iter().copied().fold(f64::INFINITY, f64::min);
    let wall_max_us = walls.iter().copied().fold(0.0, f64::max);
    Stats {
        wall_us: median(&mut walls),
        gpu_us: gpu_ok.then(|| median(&mut gpus)),
        wall_min_us,
        wall_max_us,
    }
}

fn print_row(label: &str, s: Stats) {
    let gpu = s
        .gpu_us
        .map_or_else(|| "      n/a".to_owned(), |g| format!("{g:9.1}"));
    println!(
        "{label:<46} {:9.1} {gpu} {:9.1} {:9.1}",
        s.wall_us, s.wall_min_us, s.wall_max_us
    );
}

fn main() {
    let gpu = Gpu::new();
    println!(
        "warmup {WARMUP_RUNS} runs, median of {TIMED_RUNS}; all times in microseconds\n"
    );
    println!(
        "{:<46} {:>9} {:>9} {:>9} {:>9}",
        "scenario", "wall_med", "gpu_med", "wall_min", "wall_max"
    );

    // (1) empty command buffer commit+wait round trip.
    let s1 = measure(&gpu, &[&|_cb: &ProtocolObject<dyn MTLCommandBuffer>| {}]);
    print_row("1  empty CB, commit+wait", s1);

    // (2) one CB with 14 tiny dispatches.
    let tiny14 = |cb: &ProtocolObject<dyn MTLCommandBuffer>| gpu.encode_tiny(cb, 14);
    let s2 = measure(&gpu, &[&tiny14]);
    print_row("2  one CB, 14 tiny dispatches", s2);

    // (3) planned shape: CB1(4)+wait, CB2(3)+wait, CB3(7)+wait, tiny kernels.
    let t4 = |cb: &ProtocolObject<dyn MTLCommandBuffer>| gpu.encode_tiny(cb, SHAPE[0]);
    let t3 = |cb: &ProtocolObject<dyn MTLCommandBuffer>| gpu.encode_tiny(cb, SHAPE[1]);
    let t7 = |cb: &ProtocolObject<dyn MTLCommandBuffer>| gpu.encode_tiny(cb, SHAPE[2]);
    let s3 = measure(&gpu, &[&t4, &t3, &t7]);
    print_row("3  3 CBs (4+3+7 tiny) + 3 waits", s3);

    // (4a) same shape, 16Mi-thread grids, no memory traffic.
    let n4 = |cb: &ProtocolObject<dyn MTLCommandBuffer>| gpu.encode_grid(cb, &gpu.nop_grid, SHAPE[0]);
    let n3 = |cb: &ProtocolObject<dyn MTLCommandBuffer>| gpu.encode_grid(cb, &gpu.nop_grid, SHAPE[1]);
    let n7 = |cb: &ProtocolObject<dyn MTLCommandBuffer>| gpu.encode_grid(cb, &gpu.nop_grid, SHAPE[2]);
    let s4a = measure(&gpu, &[&n4, &n3, &n7]);
    print_row("4a 3 CBs (4+3+7), 16Mi-thread nop grids", s4a);

    // (4b) same shape, each dispatch touches the 64 MiB buffer (r+w).
    let g4 = |cb: &ProtocolObject<dyn MTLCommandBuffer>| gpu.encode_grid(cb, &gpu.touch, SHAPE[0]);
    let g3 = |cb: &ProtocolObject<dyn MTLCommandBuffer>| gpu.encode_grid(cb, &gpu.touch, SHAPE[1]);
    let g7 = |cb: &ProtocolObject<dyn MTLCommandBuffer>| gpu.encode_grid(cb, &gpu.touch, SHAPE[2]);
    let s4b = measure(&gpu, &[&g4, &g3, &g7]);
    print_row("4b 3 CBs (4+3+7), 64MiB read+write each", s4b);

    // Derived numbers (median wall times).
    println!("\nderived (from medians):");
    println!(
        "  per-dispatch overhead within one CB (tiny):   {:7.1} us  ((2-1)/14)",
        (s2.wall_us - s1.wall_us) / 14.0
    );
    println!(
        "  extra cost of 2 more CB+wait round trips:     {:7.1} us  (3-2, i.e. {:.1} us per sync)",
        s3.wall_us - s2.wall_us,
        (s3.wall_us - s2.wall_us) / 2.0
    );
    println!(
        "  16Mi-thread scheduling cost per dispatch:     {:7.1} us  ((4a-3)/14, no memory)",
        (s4a.wall_us - s3.wall_us) / 14.0
    );
    let traffic_bytes = (SHAPE.iter().sum::<usize>() * 2 * BIG_ELEMS * size_of::<u32>()) as f64;
    if let Some(g) = s4b.gpu_us {
        println!(
            "  4b effective bandwidth (14 x 128MiB r+w):     {:7.1} GB/s (GPU time)",
            traffic_bytes / (g * 1e-6) / 1e9
        );
    }
    println!(
        "  fixed overhead of planned shape (3, wall):    {:7.1} us = {:.3} ms",
        s3.wall_us,
        s3.wall_us / 1000.0
    );
    for cpu_gbs in [3.0_f64, 5.0, 7.0] {
        println!(
            "  crossover vs CPU parse at {cpu_gbs:.0} GB/s:           {:7.2} MB  (input where CPU time = shape overhead)",
            s3.wall_us * 1e-6 * cpu_gbs * 1e9 / 1e6
        );
    }
}
