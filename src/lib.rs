//! # metal-json
//!
//! GPU JSON parser on Apple Metal: parses standard JSON documents to a
//! simdjson-equivalent typed tape on Apple Silicon GPUs, exploiting unified
//! memory (`bytesNoCopy`, zero copy) to beat CPU SIMD parsers on large
//! (MBs–GB) inputs.
//!
//! ## Status: M4 complete — the GPU parser is end-to-end
//!
//! [`Parser`] parses real documents on the GPU by default: `Parser::new()`
//! acquires the Metal device and [`Parser::parse`] drives the full
//! CB1→CB2→CB3 pipeline ([`gpu::GpuPipeline`]) to a complete
//! tape-format-v1 [`Document`]. What exists today: the build pipeline (AOT
//! `.metal` → metallib in build.rs, or runtime MSL compilation behind the
//! `runtime-shaders` feature), the safe wrapper layer over `objc2-metal`
//! in [`metal`] (including multi-dispatch [`metal::CommandBatch`]
//! encoding), the **tape format v1** foundation in [`tape`] (constants +
//! encode/decode helpers + buffers, locked to `shaders/tape_types.h` by a
//! layout test; spec in `docs/tape-format.md`), the full `cpu-reference`
//! oracle behind `Backend::CpuReference`, the kernel infrastructure in
//! [`stage`], the **stage-1 kernels K1–K5** (classify + escape + UTF-8,
//! escape valve, spine scans, token mask, token scatter — [`gpu::Stage1`]),
//! the **CB2 extension K6/K7/K6b** (Layer-1 validation + tape footprints,
//! the 5-component spine scan, tape offsets and the skeleton/string/scalar
//! lists — [`gpu::Stage2`]), the **CB3 structure kernels** (depth scan, K8
//! stable counting sort, K9 pairing/context/child counts, K12 container
//! tape words, K13 root words — [`gpu::Stage3`]), and the **M4 scalar
//! kernels**: K10 numbers/literals (Eisel-Lemire f64 bit patterns over the
//! verified 128-bit pow5 table, integer fast paths, hard-case CPU fixups —
//! [`gpu::Numbers`]) and K11 strings (vectorized no-escape fast path,
//! full escape/surrogate validation + unescape — [`gpu::StringsStage`]),
//! all composed by [`gpu::GpuPipeline`] with every error class at
//! reference parity (JSONTestSuite 318/318 two-way vs the oracle). Every
//! kernel/stage is diffed bit-for-bit against the `cpu-reference` oracle.
//! M5 (benchmarks, buffer pooling, zero-copy input + `Document`s,
//! per-kernel timing) is next — see
//! `docs/superpowers/specs/2026-06-10-metal-json-design.md` for the design.
//!
//! ## Feature flags
//!
//! - `runtime-shaders` — compile MSL at runtime instead of embedding an
//!   AOT-built metallib; honors `METAL_JSON_SHADER_DIR` for hot reload.
//!   Also the fallback for machines without the Xcode Metal toolchain.
//! - `cpu-reference` — scalar CPU oracle backend (M1).
//! - `timing` — per-kernel GPU timing via `MTLCounterSampleBuffer` (M5).

mod error;

pub mod document;
pub mod gpu;
pub mod metal;
pub mod parser;
pub mod stage;
pub mod tape;
pub mod value;

#[cfg(feature = "cpu-reference")]
pub mod reference;

pub use document::Document;
pub use error::{Error, Result, SyntaxErrorKind};
pub use parser::{Backend, Parser, ParserOptions};
pub use value::{Value, ValueKind};
