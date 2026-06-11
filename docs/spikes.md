# M0 Spike Results

Measured on the target machine: Apple M5 Max (unified memory), macOS Metal runtime, Apple metal toolchain 32023.883 (`xcrun -sdk macosx metal`), rustc/cargo 1.95.0. All spikes are standalone `cargo run --release --example ...` programs driving `objc2-metal` directly with self-contained MSL compiled via `newLibraryWithSource`; no `src/` wrappers or `shaders/` files were touched. These numbers drive design decisions in M2 (bitmap kernels K1/K3/K5) and M4 (number kernel K10); see the [Decisions](#decisions) section at the end.

---

## Spike A — 64×64→128-bit multiplication strategy for the Eisel-Lemire f64 kernel (M4)

**Example:** [`examples/spike_mulhi.rs`](/Users/brooklyn/workspace/github/metal-json/examples/spike_mulhi.rs)

### Question

(1) Does `mulhi(ulong, ulong)` compile in MSL on this machine? (2) Throughput of full 128-bit products (hi+lo) via `mulhi(ulong)` vs a 4-limb 32-bit mul/mulhi formulation? (3) Are `ulong` add/shift/popcount natively fine?

### Method

Standalone example `/Users/brooklyn/workspace/github/metal-json/examples/spike_mulhi.rs` (`cargo run --release --example spike_mulhi`), self-contained: drives objc2-metal directly with its own MSL string compiled via `newLibraryWithSource` (no wrapper changes, `src/` untouched). N = 2^26 = 67,108,864 `ulong` elements (512 MiB per input array; 1.5 GiB traffic per dispatch: read a, read b, write out). Each multiply kernel runs R dependent rounds of a full 128-bit multiply with hi mixed into both next operands (`x = lo^hi; y += hi|1`) to defeat DCE and force the real dependency chain; R=8 and R=32 per variant, so `(t_r32 - t_r8)/24 rounds` isolates pure ALU cost with memory traffic cancelled.

Three variants:

- **mulhi64** = `mulhi(ulong)` + 64-bit mul
- **limbs32** = pure 4-limb 32-bit (4× mul + 4× `mulhi(uint)` + explicit carry adds)
- **limbs64** = mixed portable umul128 (4× `(ulong)uint*uint` partial products, 64-bit accumulation, simdjson-style)

Timing: 3 warmup runs, median of 7 timed runs using command-buffer GPUStartTime/GPUEndTime deltas (GPU timestamps were available on every run; host fallback never triggered). Every kernel output verified against a CPU u128 oracle on a 65,536-element prefix (131,072 for the uint kernel). Q3: 32-round dependent add/rotate/xor/popcount chains on `ulong` (N elements) vs `uint` (2N elements over the same buffers, identical byte traffic). Two full runs executed to confirm stability; numbers below from the second run (first run within ~2%).

### Numbers

Device: Apple M5 Max.

**Q1:** `mulhi(ulong, ulong)` compiles: **YES**, both runtime `newLibraryWithSource` and offline `xcrun metal` (MSL 3.2).

**Q2** medians (7 runs):

| Variant | r8 time | r8 mul128/s | r8 GB/s | r32 time | r32 mul128/s | ALU-isolated mul128/s (Δ(r32−r8)/24) |
|---|---|---|---|---|---|---|
| mulhi64 | 2.944 ms | 1.82e11 | 547 | 7.233 ms | 2.97e11 | 3.76e11 |
| limbs32 | 2.902 ms | 1.85e11 | 555 | 7.114 ms | 3.02e11 | 3.82e11 |
| limbs64 | 2.936 ms | 1.83e11 | 549 | 5.602 ms | 3.83e11 | **6.04e11** |

All three identical at r8: memory-bound, saturating ~546 GB/s M5 Max bandwidth. ALU-isolated: limbs64 wins by ~1.6× and the gap reproduced across both runs (first run: 3.76e11 / 3.88e11 / 6.05e11).

**Q3:** ops_chain64 5.464 ms = 3.93e11 rounds/s; ops_chain32 3.371 ms = 1.27e12 rounds/s; u64 = 0.29–0.31× of u32 round throughput (clean 2×32 lowering would be 0.5×; this chain is rotate/popcount-heavy and 64-bit rotates lower to ~3× uint cost; plain add/xor look like clean 2×).

### Recommendation

The M4 Eisel-Lemire number kernel should use an explicit inline `umul128` helper in the **limbs64** style: split operands into 32-bit halves, form the four `(ulong)uint*uint` partial products, and assemble hi and lo with 64-bit adds/shifts (exactly simdjson's portable umul128 fallback). Do NOT use `mulhi(ulong)` + `x*y` when both product words are needed: it compiles and is correct, but it is ~1.6× slower ALU-wise (3.76e11 vs 6.04e11 mul128/s) because the two opaque builtins each redo partial-product work, while the explicit limb form lets the compiler compute the 4 partial products once and derive both words. The pure 32-bit carry-chain formulation (limbs32) buys nothing over `mulhi(ulong)` (3.82e11) — the explicit bool-carry adds cost as much as the builtin saves — so there is no reason to drop to uint state. Keep mantissa/product state as `ulong`: ulong add/xor/shift lower cleanly to 2×32 ops (no cliff), just avoid gratuitous 64-bit rotates/popcounts in hot loops where a uint formulation exists (~3× uint cost, not 2×). `mulhi(ulong)` remains fine for the rare places that need ONLY the high word, though limbs64-derived hi is equally fast there.

### Caveats

1. The 1.6× limbs64 win only matters under ALU pressure: at 8 rounds per element every variant was identically memory-bound at ~545–555 GB/s, and the real K10 number kernel does ~1–2 mul128 per number amid heavy memory traffic, so formulation choice likely won't move end-to-end numbers much — it is still free perf, take it.
2. ALU-isolated rates come from the r32−r8 delta (memory cancels), not from a pure-register kernel; occupancy/latency-hiding differences between variants are folded in.
3. The Q3 0.29× figure is specific to a rotate+popcount-heavy chain; it is the worst case for u64, not a general "u64 is 3.4× slow" claim.
4. Compiler-version dependent: Apple metal 32023.883 / macOS Metal runtime on M5 Max; re-run the spike if the toolchain changes.
5. No `src/` or `shaders/` files were touched; the example is fully self-contained and CPU-oracle-verified on every run.

---

## Spike B — `ulong` vs `uint2` bitmap words for K1/K3/K5 (M2)

**Example:** [`examples/spike_wordsize.rs`](/Users/brooklyn/workspace/github/metal-json/examples/spike_wordsize.rs)

### Question

Should the K1/K3/K5 bitmap kernels (M2) keep their 64-byte-chunk bitmaps as a single `ulong`, or as `uint2` (lo/hi pair with explicit carry across the 32-bit boundary) on Apple GPUs with 32-bit ALUs?

### Method

Standalone example `examples/spike_wordsize.rs`, self-contained (own MSL string compiled at runtime via `newLibraryWithSource` through raw objc2-metal; no wrapper or shader files touched). Two kernels do identical K1-style work per thread: load 64 input bytes as 4× `uint4`, build quote/backslash/structural bitmaps via per-byte compares and shifts, then run the representative bitmap-op sequence — simdjson `find_escaped` (incl. the 64-bit add with carry-out) plus the 6-step prefix-xor shift ladder plus masking shifts/ands — written naturally in each word size (variant 1: `ulong`; variant 2: `uint2` with hand-written cross-word shift/carry plumbing). 16Mi chunks × 64 B = 1 GiB input read + 128 MiB checksum output per run, dispatched as one 16M-thread grid (tg=256) on the Apple M5 Max. Timing: 3 warm-ups, median of 10 runs, command-buffer GPUEndTime−GPUStartTime (GPU timestamps were available; host fallback never triggered). Correctness: both variants' 8-byte-per-chunk outputs compared bit-for-bit and spot-checked against a scalar CPU model. A second config repeats the bitmap-op sequence 8× with a per-round dependency to amplify ALU cost in case the 1× config were memory-bound.

### Numbers

Apple M5 Max, 1 GiB input, median of 10 (two independent process runs shown as run1 / run2):

| Config | ulong | uint2 | uint2/ulong ratio |
|---|---|---|---|
| rounds=1 (representative K1 mix) | 4.507 / 4.943 ms = 238.2 / 217.2 GB/s input | 2.198 / 2.180 ms = 488.6 / 492.6 GB/s input (549.6 / 554.2 GB/s incl. output traffic) | 2.05× / 2.27× |
| rounds=8 (ALU-amplified) | 5.351 / 5.423 ms = 200.6 / 198.0 GB/s | 2.715 / 2.715 ms = 395.4 / 395.4 GB/s | 1.97× / 2.00× |

Outputs matched bit-for-bit between variants in every config (xor checksum `0xce0d4c1148e5abc2` for rounds=1, `0x5fac31ae3e051f89` for rounds=8, identical across runs) and matched the scalar CPU model on sampled chunks.

### Recommendation

Use **uint2** (32-bit lo/hi pairs with explicit carries) for the M2 bitmap kernels. uint2 is ~2× faster than ulong on this machine in the exact K1-style mix. At rounds=1 the uint2 variant reaches ~550 GB/s of total memory traffic — i.e. it turns the kernel memory-bandwidth-bound — while the ulong variant is ALU-bound at under half that, so the emulated 64-bit ops cost more than the entire memory traffic of the kernel. The ~2× gap persists at 8× ALU amplification, so kernels with more bitmap ops per byte (K3/K5) benefit at least as much. Write the 64-bit add-with-carry (`find_escaped`) and the prefix-xor ladder explicitly in two 32-bit words as done in spike_bitmap_uint2; a thin uint2 helper set (`shl64`, `add64`-with-carry, `prefix_xor64_u2`) is all that is needed.

### Caveats

1. Each variant is written the way real kernel code would be written in that word size and compiled by the Metal runtime compiler; the measured gap therefore includes whatever the compiler does (or fails to do) when lowering loop-carried ulong shifts/adds — that is the realistic comparison, but a hand-tuned ulong formulation could narrow it somewhat.
2. The ALU work is data-independent, so the 1 GiB of splitmix64 random bytes (not real JSON) does not bias the timing; only checksum values depend on data.
3. uint2 at rounds=1 appears pinned at memory bandwidth (~550 GB/s total traffic), so the true ALU advantage of uint2 is likely larger than the observed 2× there (rounds=8, ~2.0×, is the cleaner ALU-bound comparison).
4. ulong medians wobbled ~10% between process runs (4.5–4.9 ms); uint2 was stable within 1%. The conclusion is insensitive to this noise.
5. Numbers are from one M5 Max; per the plan, the design ports between word sizes with constant changes only if other hardware disagrees.

---

## Spike C — Fixed overhead of the 3-CB / ~14-dispatch / 2-CPU-sync pipeline shape

**Example:** [`examples/spike_overhead.rs`](/Users/brooklyn/workspace/github/metal-json/examples/spike_overhead.rs)

### Question

What is the fixed overhead of the planned 3-command-buffer / ~14-dispatch / 2-CPU-sync pipeline shape on this machine (M5 Max), and does it confirm the plan's assumption of ~0.1–0.5 ms fixed overhead with a CPU-parsing crossover at a few MB?

### Method

Standalone example `/Users/brooklyn/workspace/github/metal-json/examples/spike_overhead.rs` (`cargo run --release --example spike_overhead`). Self-contained: compiles its own 3-kernel MSL string via `newLibraryWithSource` (no changes to `src/` wrappers, which keep device/queue pub(crate)). Kernels: `spike_tiny` (1 threadgroup, thread 0 bumps a sink word), `spike_nop_grid` (16Mi threads, bound-check only, no memory traffic), `spike_touch` (16Mi threads, one u32 read+write each over a 64 MiB shared buffer).

Scenarios:

1. empty CB commit+waitUntilCompleted;
2. one CB with 14 tiny dispatches;
3. planned shape CB1(4)+wait, CB2(3)+wait, CB3(7)+wait with tiny kernels;
4. (a) same shape with 16Mi-thread nop grids (pure scheduling at size); (b) same shape with 64 MiB read+write per dispatch (14 × 128 MiB traffic).

Timing: 3 warmup runs then median of 20; GPU time = sum of per-CB GPUEndTime−GPUStartTime, wall time = host `Instant` around encode→commit→wait (what a parse pays); automatic wall-only fallback when timestamps are absent (only the empty CB, which does no GPU work). Serial compute encoder, PSO+buffers re-bound per dispatch to mimic 14 distinct kernels. Verified across 3 process runs for stability.

### Numbers

ACTUAL measured medians on Apple M5 Max (ranges across 3 process runs of median-of-20):

| Scenario | Wall | GPU |
|---|---|---|
| (1) empty CB commit+wait | 18.7–20.1 µs | n/a (zero GPU work) |
| (2) one CB, 14 tiny dispatches | 219–421 µs (min 138, occasional outliers to ~1.1 ms) | 29–53 µs |
| (3) planned shape, 3 CBs (4+3+7 tiny) + 3 waits | 497–533 µs | 31–38 µs |
| (4a) shape with 16Mi-thread nop grids | 2.71–2.92 ms | 2.21–2.38 ms |
| (4b) shape with 64 MiB read+write per dispatch | 3.78–3.92 ms | 3.19–3.27 ms |

Scenario 3: nearly all of the 0.5 ms is CPU-side encode/commit/sync latency, not GPU execution. Derived: ~14–29 µs per tiny dispatch within one CB; splitting one CB into three with waits adds ~95–315 µs (~50–160 µs per extra sync round trip). 4a: ~158–170 µs per 16Mi-thread dispatch of pure scheduling (~2.4 ns per 256-wide threadgroup, 65536 groups). 4b: 574–590 GB/s effective bandwidth over 1.88 GB of traffic.

Crossover where CPU parse time equals the 0.50–0.53 ms shape overhead: 1.5–1.6 MB at 3 GB/s, 2.5–2.7 MB at 5 GB/s, 3.5–3.7 MB at 7 GB/s.

### Recommendation

**CONFIRM** the plan's assumption, at the top of its band: the full 3-CB/14-dispatch/2-sync shape costs ~0.50–0.53 ms wall on this machine (assumed 0.1–0.5 ms), and the crossover vs CPU parsing is ~1.5–3.7 MB depending on baseline speed — "a few MB" holds. Three adjustments for later milestones:

- (a) Overhead is CPU-sync dominated (GPU busy time for trivial work is only ~30–50 µs of the ~500 µs), so each `waitUntilCompleted` round trip costs ~50–160 µs; collapsing CB2+CB3 into one CB (1 sync instead of 2) would shave ~0.1–0.3 ms and pull the crossover under ~2 MB if it ever matters, but it is not required for the ≥100 MB headline target where 0.5 ms is noise (<1% at 100 MB).
- (b) Giant-grid scheduling is real: a 16Mi-thread dispatch costs ~160 µs even with an empty body (~2.4 ns/threadgroup), so kernels should process a grain of ≥16–64 bytes per thread (the planned 64-byte-per-thread bitmap kernels imply ~1M threads ⇒ ~10 µs/dispatch, negligible — keep it that way, avoid thread-per-byte designs).
- (c) Use medians in all benches: low-occupancy wall times jitter up to 4× (power-state ramping), max outliers 1–2.3 ms on tiny workloads.

### Caveats

1. GPUStartTime/GPUEndTime return 0 for empty command buffers (no GPU work), so scenario 1 reports wall time only; the documented host-timing fallback engaged exactly there.
2. Tiny-workload wall medians vary ~2× between process runs (219–421 µs for scenario 2) due to GPU power-state ramping at near-zero occupancy; ranges across 3 runs are reported instead of a single number, and conclusions use the stable scenario-3 number (497–533 µs).
3. The 574–590 GB/s in 4b may be slightly flattered by the 64 MiB working set partially fitting the system-level cache; treat it as an upper bound, not sustained DRAM bandwidth.
4. The 14 dispatches re-bind the same PSO; real pipelines bind 14 distinct PSOs, which could add a few µs of CPU encode time but not change the conclusion.

---

## Decisions

These spike results translate into the following binding design decisions:

### (a) M2 bitmap word type: `uint2`

K1/K3/K5 store and manipulate their 64-byte-chunk bitmaps as `uint2` (lo/hi 32-bit pairs with explicit carries), not `ulong`. Spike B measured ~2× kernel speedup (uint2 reaches the ~550 GB/s memory-bandwidth ceiling; ulong is ALU-bound at under half). Implement a thin shared helper set in `shaders/common.h` — `shl64`, `add64`-with-carry (for `find_escaped`), `prefix_xor64_u2` — and write the prefix-xor ladder and carry plumbing explicitly in two 32-bit words.

### (b) M4 mulhi strategy: explicit limbs64 `umul128`

The Eisel-Lemire kernel (K10) computes 128-bit products with an explicit inline `umul128` in the limbs64 style: split into 32-bit halves, four `(ulong)uint*uint` partial products, assemble hi/lo with 64-bit adds/shifts (simdjson's portable fallback). This is ~1.6× faster ALU-wise than `mulhi(ulong)` + `x*y` (6.04e11 vs 3.76e11 mul128/s) and no worse anywhere. Mantissa/product state stays `ulong` (add/xor/shift lower cleanly to 2×32); avoid 64-bit rotates/popcounts in hot loops (~3× uint cost). `mulhi(ulong)` is acceptable only where solely the high word is needed.

### (c) Fixed-overhead budget and CPU/GPU crossover: confirmed

Fixed pipeline overhead is **~0.50–0.53 ms** wall for the planned 3-CB / 14-dispatch / 2-sync shape (top of the plan's assumed 0.1–0.5 ms band), dominated by CPU sync (~50–160 µs per `waitUntilCompleted`), not GPU work (~30–50 µs). Estimated CPU/GPU crossover: **~1.5–3.7 MB** (1.5–1.6 MB vs a 3 GB/s CPU baseline, 2.5–2.7 MB at 5 GB/s, 3.5–3.7 MB at 7 GB/s) — the plan's "a few MB" claim holds and the bench report (M5) should document it at these sizes. Budget rules going forward: keep ≥64 bytes/thread grain in big-grid kernels (giant-grid scheduling costs ~2.4 ns/threadgroup); optionally merge CB2+CB3 later to shave ~0.1–0.3 ms if small-input latency ever matters; use medians in all benches (tiny-workload wall times jitter up to 4× from power-state ramping). At the ≥100 MB headline target, 0.5 ms overhead is <1% — noise.
