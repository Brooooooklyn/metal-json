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

## Status: WIP (milestone M0 — scaffold)

Working today:

- AOT shader pipeline: `shaders/*.metal` → `metal_json.metallib` in build.rs,
  embedded into the binary (`runtime-shaders` feature switches to runtime MSL
  compilation with `METAL_JSON_SHADER_DIR` hot reload).
- Safe wrapper layer over `objc2-metal` (`src/metal/`): context, compute
  pipelines, shared-storage buffers, synchronous dispatch helper.
- GPU smoke tests (`cargo test`) proving 32-bit and 64-bit integer kernels
  end to end.

The parser itself (`Parser` / `Document` / `Value`) lands over milestones
M1–M5. Full design:
[`docs/superpowers/specs/2026-06-10-metal-json-design.md`](docs/superpowers/specs/2026-06-10-metal-json-design.md).

## Requirements

- macOS on Apple Silicon.
- Full Xcode with the Metal toolchain
  (`xcodebuild -downloadComponent MetalToolchain`) — or build with
  `--features runtime-shaders` to skip the AOT step.

## License

MIT OR Apache-2.0
