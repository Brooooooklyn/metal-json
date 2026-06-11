//! # metal-json
//!
//! GPU JSON parser on Apple Metal: parses standard JSON documents to a
//! simdjson-equivalent typed tape on Apple Silicon GPUs, exploiting unified
//! memory (`bytesNoCopy`, zero copy) to beat CPU SIMD parsers on large
//! (MBs–GB) inputs.
//!
//! ## Status: M3 (work in progress)
//!
//! What exists today: the build pipeline (AOT `.metal` → metallib in
//! build.rs, or runtime MSL compilation behind the `runtime-shaders`
//! feature), the safe wrapper layer over `objc2-metal` in [`metal`]
//! (including multi-dispatch [`metal::CommandBatch`] encoding), the
//! [`Error`] type, GPU smoke tests, the **tape format v1** foundation in
//! [`tape`] (constants + encode/decode helpers + buffers, locked to
//! `shaders/tape_types.h` by a layout test; spec in `docs/tape-format.md`),
//! the full `cpu-reference` oracle behind [`Parser`]/[`Document`]/[`Value`],
//! the M2 kernel infrastructure in [`stage`] (per-stage encode abstraction,
//! single-stage test harness, stage-1 buffer container) with the
//! `shaders/bitmap_u2.h` uint2 64-bit vocabulary and its GPU self-test, and
//! the **stage-1 GPU kernels K1–K5** (classify + escape + UTF-8, escape
//! valve, spine scans, token mask, token scatter) orchestrated by
//! [`gpu::Stage1`] and diffed bit-for-bit against the oracle in
//! `tests/kernels.rs`, the **M3 CB2 extension K6/K7/K6b** (Layer-1
//! validation + tape footprints, the 5-component spine scan, tape offsets
//! and the skeleton/string/scalar lists) orchestrated by [`gpu::Stage2`]
//! and diffed against reference stage 3 in its module tests, and the **M3
//! CB3 structure kernels** (hierarchical depth scan, K8 stable counting
//! sort by depth, K9 pairing/context/child counts, K12 container tape
//! words, K13 root words, error fold) orchestrated by [`gpu::Stage3`] —
//! the full M3 structure pipeline, producing the tape's container/root
//! words around zero-word scalar/string holes — and diffed against
//! reference stages 4 and `emit_tape`. The remaining work (M4 scalar
//! kernels filling the holes, parser integration) lands over M4–M5 — see
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
