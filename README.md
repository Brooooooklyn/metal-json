# metal-json

GPU JSON parser on Apple Metal. Parses standard JSON documents to a
simdjson-equivalent typed tape (parsed numbers, validated + unescaped
strings, full grammar validation) on the Apple Silicon GPU.

## Why

Apple unified memory removes the host↔device copy cost that cripples
discrete-GPU parsers: input files map straight into `MTLBuffer`s with
`bytesNoCopy`, zero copy. The pipeline touches roughly 9 bytes of memory per
input byte, which on an M-series Max translates to a realistic 24–40 GB/s —
versus ~3–7 GB/s for CPU SIMD parsers like simdjson. The goal: **faster than
C++ simdjson parse-to-tape on large inputs (100MB–1GB)**, with the CPU/GPU
crossover point documented honestly.

## Status: WIP (milestone M4 — the GPU parser is end-to-end)

Working today:

- **The full GPU parser**: `Parser::new()?` (acquires the Metal device;
  `Backend::Gpu` is the default whenever the machine has one — without a
  device the default resolves to the `cpu-reference` oracle when that
  feature is compiled in, while an *explicit* `Backend::Gpu` is never
  second-guessed) → `parser.parse(&bytes)?` → `Document` / `Value`
  navigation. All 13 kernels (classify/escape/UTF-8 → scans → token
  scatter → validation/footprints → depth sort → bracket pairing →
  container tape words → number parse with Eisel-Lemire f64 bit patterns +
  CPU fixups for the hard roundings → string validation/unescape, with the
  same CPU-fixup valve for rare >16 KiB strings so one giant string never
  serializes a parse on a single GPU lane) across 3 command buffers with
  exact-size allocations at each CPU sync.
- Every error class at reference parity (JSONTestSuite 318/318 two-way vs
  the scalar `cpu-reference` oracle, which remains available as an explicit
  backend behind the `cpu-reference` feature).
- AOT shader pipeline: `shaders/*.metal` → `metal_json.metallib` in build.rs,
  embedded into the binary (`runtime-shaders` feature switches to runtime MSL
  compilation with `METAL_JSON_SHADER_DIR` hot reload).
- Safe wrapper layer over `objc2-metal` (`src/metal/`): context, compute
  pipelines, shared-storage buffers, multi-dispatch command batches.

M5 (benchmarks vs vendored C++ simdjson, buffer pooling, zero-copy input
and `Document`s, per-kernel timing) is next. Full design:
[`docs/superpowers/specs/2026-06-10-metal-json-design.md`](docs/superpowers/specs/2026-06-10-metal-json-design.md).

## Requirements

- macOS on Apple Silicon.
- Full Xcode with the Metal toolchain
  (`xcodebuild -downloadComponent MetalToolchain`) — or build with
  `--features runtime-shaders` to skip the AOT step.

## License

MIT OR Apache-2.0
