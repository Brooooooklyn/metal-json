# metal-json

JSON parsing on the Apple Silicon GPU. metal-json parses standard JSON
documents to a simdjson-equivalent typed tape — parsed `i64`/`u64`/`f64`
numbers, validated + unescaped strings, full grammar validation — entirely
through a Metal compute pipeline, and beats C++ simdjson on the large
documents in our benchmark sweep.

## Results

Median throughput, parse-to-tape + identical stats walk for both headline
contenders, criterion medians on an Apple M5 Max (macOS 26.5, simdjson
v4.6.4 vendored, `-O3`). Full table, provenance, and methodology:
[`docs/bench-report.md`](docs/bench-report.md).

| dataset      | size      | metal-json GB/s | simdjson (C++) GB/s | speedup   |
|--------------|----------:|----------------:|--------------------:|----------:|
| twitter      | 0.6 MiB   | 0.49            | 3.96                | 0.12×     |
| citm_catalog | 1.6 MiB   | 1.90            | 4.36                | 0.44×     |
| canada       | 2.1 MiB   | 2.18            | 1.54                | **1.42×** |
| twitter_100m | 100 MiB   | 6.40            | 3.11                | **2.06×** |
| twitter_256m | 256 MiB   | 7.13            | 3.26                | **2.19×** |
| twitter_512m | 512 MiB   | 7.51            | 2.77                | **2.71×** |

**The claim this data supports:** metal-json is **2.1–2.7× faster** than
C++ simdjson (DOM tape API) on large documents in our benchmark sweep —
100–512 MiB inputs, twitter-shaped (deterministic expansions of the
twitter template), measured on one Apple M5 Max — with every CPU-side cost
of the hybrid pipeline (syncs, allocations, fixup passes) inside the timed
region. Document shape matters: number-dense canada already favors
metal-json at just 2.1 MiB (1.42×), while citm-style documents favor
simdjson at the small sizes we measured — non-twitter shapes were only
benchmarked at 0.6–2.1 MiB, and broader large-document shape coverage is
future work. Parse-only (without the stats walk both sides run), the
512 MiB parse sustains ~11–12 GB/s of wall throughput (the exact figure
moves ~1 GB/s between sessions; see the breakdown in the report).

**The crossover caveat:** the GPU pipeline pays a roughly fixed ~0.5–0.9 ms
of dispatch/sync overhead per parse. On this machine the measured crossover
on the twitter sweep is **≈ 4–5 MiB**: below that simdjson wins (up to ~8×
faster on the 0.6 MiB twitter.json), above it metal-json wins and the gap
widens with size. If your documents are small, use a CPU parser; this
library is for big ones.

## Quick start

```rust
use metal_json::Parser;

fn main() -> Result<(), metal_json::Error> {
    let parser = Parser::new()?; // Metal device + pipelines, reusable

    let doc = parser.parse(br#"{"name":"meow","tags":[1,2.5]}"#)?;
    let root = doc.root();
    assert_eq!(root.get("name").unwrap().as_str(), Some("meow"));
    assert_eq!(root.get("tags").unwrap().at(1).unwrap().as_f64(), Some(2.5));

    // Big files: read once, straight into a pooled page-aligned buffer.
    // let doc = parser.parse_file("big.json")?;
    Ok(())
}
```

`Parser` is reusable (device, pipeline states, and a buffer pool are built
once); `Document` is self-contained and returns its buffers to the pool on
drop. For repeated parsing without input copies, fill a parser-provided
page-aligned buffer (`AlignedInput`) and call `parse_aligned`. Zero-copy
*file* input exists as `unsafe fn parse_file_mmap` (mmap →
`bytesNoCopy`): it is `unsafe` because the file must not be truncated or
modified for the duration of the parse — truncating a mapped file can
`SIGBUS` the process — which is exactly why the safe `parse_file` copies.

## What it parses

- **Full standard JSON** (RFC 8259) — objects, arrays, strings with all
  escapes (`\uXXXX` + surrogate pairs), numbers, literals; strict UTF-8
  validation; no extensions, no relaxed mode. Error parity with the scalar
  reference oracle across JSONTestSuite (318/318 two-way).
- Numbers follow simdjson's type policy (`i64` → `u64` → `f64`) with
  bit-exact f64s (Eisel-Lemire on the GPU; rare hard roundings re-parsed on
  the CPU from a GPU-built fixup list).
- Nesting depth limit 1024 (simdjson parity), configurable via
  `ParserOptions::max_depth`.
- Maximum input size 4 GiB − 65 bytes (`u32` tape/token indices).

## Requirements

- **macOS on Apple Silicon** (unified memory is the point: input bytes map
  into the GPU with `MTLBuffer bytesNoCopy`, output tapes live in shared
  storage — zero copies either way).
- Xcode Metal toolchain for the AOT shader build
  (`xcodebuild -downloadComponent MetalToolchain`), or build with
  `--features runtime-shaders` to compile MSL at runtime instead.

## Architecture

The kernels (K1–K13, plus a K6b offset-scatter pass) run across four
command buffers; each CPU sync reads back exact sizes so every allocation is
exact-fit. Bitmaps are built 64 input bytes per thread; scans are
hierarchical reduce→spine→apply (no decoupled look-back — Apple GPUs give no
forward-progress guarantee).

```text
input bytes ── mmap / page-aligned ──> MTLBuffer (bytesNoCopy, zero copy)
     │
 CB1 │ K1 classify+escape+UTF-8 → K2 spine scan → K3 in-string mask → K4 spine scan
     ├── CPU sync 1: token count → exact-size token/scratch allocations
 CB2 │ K5 token scatter → K6 local validation + tape footprints → K7 spine scan
     ├── CPU sync 2: tape/stringbuf/list sizes → exact-fit allocations
 CB2b│ K6b scatter per-token tape offsets (after this sync every CB3 size is known)
 CB3 │ K8 counting sort by depth → K9 bracket pairing + container context
     │ K10 numbers (Eisel-Lemire f64 bit patterns) → K11 string unescape
     │ K12 container tape words → K13 root/finalize + error reduce
     ├── CPU: error verdict; rare fixups (hard float roundings, >16 KiB strings)
     ▼
 Document: typed tape + string buffer (shared storage, read directly by the CPU)
```

Design notes that matter for performance and correctness:

- **Bracket matching without sorting networks**: a stable counting sort by
  depth makes matching brackets adjacent — one sort yields pairing *and*
  comma/colon container context.
- **No fp64 on Apple GPUs**: the number kernel computes f64 *bit patterns*
  with 64-bit integer math (portable `umul128` via 32-bit limbs — measured
  fastest in `docs/spikes.md`).
- **Valves for adversarial inputs**: backslash walls and >16 KiB strings
  divert to CPU fixup lists instead of serializing a GPU lane; their cost is
  included in every benchmark number.
- **Errors are values**: structured `Error` with byte offsets (`Syntax`,
  `Utf8`, `DepthLimit`, `TrailingContent`, …); earliest error wins
  deterministically via atomic min-reduction. Invalid input never panics.

## Benchmarks

```sh
cargo run -p xtask -- fetch-data                                  # canonical datasets
cargo run -p xtask -- gen-data --template twitter --size 512m     # large variants
cargo run -p xtask -- bench-report                                # criterion + report
```

`bench-report` regenerates [`docs/bench-report.md`](docs/bench-report.md):
GB/s medians for metal-json / C++ simdjson (vendored FFI, DCE-defeating tape
walk on both sides) / Rust simd-json / serde_json, dataset sha256s, the size
sweep with the CPU/GPU crossover, and a per-kernel time breakdown
(`--features timing`).

## Status

All milestones complete:

- **M0** — workspace, AOT `.metal` → metallib build, Metal wrapper layer,
  decision spikes (`docs/spikes.md`).
- **M1** — tape format v1 (`docs/tape-format.md`), `Document`/`Value` API,
  scalar CPU reference oracle (`cpu-reference` feature).
- **M2** — GPU stage 1: classify/escape/UTF-8 bitmaps, spine scans, token
  extraction (K1–K5).
- **M3** — GPU structure: validation, depth sort, bracket pairing, container
  tape (K6–K9, K12–K13); JSONTestSuite parity.
- **M4** — GPU scalars: Eisel-Lemire numbers + string unescape (K10–K11);
  full GPU tape, differential-tested against serde_json bit-for-bit.
- **M5** — zero-copy input + pooled buffers, vendored simdjson benchmark
  harness, per-kernel timing, optimization to the headline numbers above.

Tests run with `MTL_SHADER_VALIDATION=1`; the GPU backend is diffed against
the CPU oracle on JSONTestSuite, proptest-generated documents, and number/
string torture corpora.

## License

MIT OR Apache-2.0
