# Parsing huge JSON on an Apple GPU, faster than simdjson

> [metal-json GitHub repo](https://github.com/Brooooooklyn/metal-json) — full write-up in [docs/parsing-json-on-an-apple-gpu.md](https://github.com/Brooooooklyn/metal-json/blob/main/docs/parsing-json-on-an-apple-gpu.md)

JSON parsing is supposed to be the worst possible workload for a GPU: branchy grammar, quotes whose meaning depends on every byte before them, brackets matched by a stack — the textbook example of sequential state.

I built it anyway. metal-json parses standard JSON to a simdjson-compatible 64-bit tape using Metal compute kernels on Apple Silicon. On large documents it beats C++ simdjson — the parser it borrows its output format from. On small ones it loses, and I want to show that first:

| dataset      | size      | metal-json | simdjson-cpp | winner     |
| ------------ | --------- | ---------- | ------------ | ---------- |
| twitter      | 0.6 MiB   | 1.287 ms   | 0.159 ms     | simdjson   |
| twitter_4m   | 4.0 MiB   | 1.435 ms   | 1.385 ms     | simdjson   |
| twitter_16m  | 16.0 MiB  | 3.383 ms   | 6.129 ms     | metal-json |
| twitter_100m | 100.0 MiB | 16.386 ms  | 33.772 ms    | metal-json |
| twitter_512m | 512.0 MiB | 71.447 ms  | 193.711 ms   | metal-json |

Criterion medians on an M5 Max, simdjson v4.6.4 at `-O3 -mcpu=native`, both parsers verified to produce bit-identical tapes before timing. The crossover is ~4.3 MiB: below it the GPU's fixed dispatch/sync overhead (~0.5 ms, measured before any kernel existed) dominates; above it metal-json wins 2.1–2.7x. If your documents are small, use a CPU parser. This library is for big ones.

## Why it works at all

simdjson already showed the branchy half is solvable: don't branch over bytes, compute bitmaps over them. Those bit tricks port straight to Metal — one GPU thread per 64-byte word instead of one loop iteration. Apple Silicon kills the discrete-GPU objection too: unified memory means the GPU reads the input pages in place and the CPU reads the tape in place. Zero copies in either direction.

The part with no CPU counterpart is bracket matching. simdjson does it with a sequential stage and a stack. A GPU needs a different idea:

**A stable sort by depth makes every matching bracket pair adjacent.**

Take an exclusive prefix sum of bracket weights (+1 open, −1 close) to get each element's depth — embarrassingly parallel. Then note: within one depth, brackets of distinct containers cannot interleave (to open a second container at depth *d* you must first return to *d−1*, closing the first). So within a depth group, in document order, brackets strictly alternate open/close:

```
input:                  [   {   :   }   ,   [   ]   ]
depth:                  1   2   2   2   1   2   2   1

stable sort by depth:
  depth-1 group:  [ , ]        <- outer array
  depth-2 group:  { : } [ ]    <- inner containers, pairs now adjacent
```

One LSD counting sort (5-bit digits, two passes at the default 1024 depth limit, "STABILITY IS CORRECTNESS-CRITICAL" shouted in the source) buys pairing, balance errors at exact offsets, separator context, and child counts — everything the stack would have computed, with no stack and no recursion anywhere in the pipeline.

## What the rest took

- **No 64-bit ALUs:** every hot bitmap is a `uint2` pair with hand-written carries — measured 2x faster than emulated `ulong` (489 vs 238 GB/s).
- **No fp64 either:** the float kernel computes f64 *bit patterns* with pure integer math — Eisel–Lemire over a 651-entry table of 128-bit powers of five. Hard cases go to a fixup list the CPU re-parses after the GPU finishes.
- **Adversarial inputs get valves, not limits:** a backslash wall caps escape look-back at 4096 bytes with a repair kernel; an 8 MB escaped string that serialized one GPU lane for 950 ms now takes the CPU valve — 83 ms. Both costs sit inside every benchmark number.
- **Verification:** every GPU kernel is diffed bit-for-bit against a scalar CPU oracle; the oracle is validated against serde_json and JSONTestSuite (95/95 accept, 188/188 reject — including 100,000 unclosed `[`s that stack-overflow recursive parsers).

## The honest part

The fixed overhead never goes away — that's physics, not a TODO. The same curve shows up end-to-end: deserializing a 4 MiB corpus into typed Rust structs via serde, plain serde_json beats the GPU path ~2.5x. The crossover isn't a confession; it's the product spec. Big documents, 2.1–2.7x. Small ones, use simdjson.

Everything is reproducible: `cargo run -p xtask -- bench-report` on any M-series Mac regenerates the whole table.
