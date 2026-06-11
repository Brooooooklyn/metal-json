//! GPU pipeline orchestration: composes the kernels in `shaders/` into the
//! command buffers of the parse pipeline.
//!
//! M2 scope: [`stage1`] — CB1 (classify → escape valve → spine → token mask
//! → spine) plus the K5 token scatter, producing bitmaps and the token
//! stream that the structural stages consume.
//!
//! M3 scope: [`stage2`] — the CB2 extension (K5 → K6 Layer-1
//! validation/footprints → K7 multi-component spine, then K6b after the
//! second CPU sync), producing the skeleton, the string/scalar work lists,
//! per-token tape offsets and the tape/string-buffer totals — and
//! [`stage3`] — the CB3 structure kernels (depth scan, K8 stable counting
//! sort by depth, K9 pairing/context/child counts, K12 container tape
//! words, K13 root words, error fold), producing the depth vector, the
//! sorted skeleton, the pair map, the separator contexts, the per-container
//! child counts and the M3 tape (container/root words around zero-word
//! scalar/string holes — the M4 kernels fill those). `stage3::run_structure`
//! is the full M3 pipeline runner.
//!
//! M4 scope: [`numbers`] (K10: number grammar + Eisel-Lemire + literals,
//! hard cases → CPU fixup) and [`strings`] (K11: record offsets +
//! validation/unescape) as standalone kernel runners, plus [`pipeline`] —
//! the **full GPU parse pipeline** that encodes K10/K11 into the same CB3
//! as the structure kernels and completes the error contract. The parser's
//! `Backend::Gpu` drives [`pipeline::GpuPipeline`]; everything else here is
//! a narrower per-milestone test orchestration.
//!
//! Internal/unstable: exposed publicly so integration tests can drive the
//! pipeline directly (like [`crate::metal`] and [`crate::stage`]), but not
//! part of the supported API surface.

pub mod numbers;
pub mod pipeline;
pub mod stage1;
pub mod stage2;
pub mod stage3;
pub mod strings;

pub use numbers::{ERR_NUMBER, Numbers, NumbersOutput, patch_number_fixups, run_numbers};
pub use pipeline::{GpuParse, GpuParseOutput, GpuPipeline};
pub use stage1::{ERR_STRING, ERR_UTF8, Stage1, Stage1Output, run_stage1};
pub use stage2::{
    ERR_EMPTY_INPUT, ERR_INVALID_LITERAL, ERR_MISSING_COLON, ERR_MISSING_COMMA, ERR_UNBALANCED,
    ERR_UNEXPECTED_TOKEN, ERR_UNTERMINATED_STRING, Stage2, Stage2Output, run_stage2,
};
pub use stage3::{
    ERR_DEPTH_LIMIT, ERR_TRAILING_CONTENT, NO_MATCH, Stage3, Stage3Output, run_stage3,
};
pub use strings::{
    ERR_STRING_CONTROL, ERR_STRING_ESCAPE, StringsOutput, StringsStage, run_strings,
};
