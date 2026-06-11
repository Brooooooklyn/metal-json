//! Structured error types. Parsing never panics or aborts on bad input;
//! every failure maps to a variant here, with byte offsets where they exist.
//!
//! GPU kernels report errors as a packed `(offset << 32) | code` u64 reduced
//! with `atomic_min` so the earliest error wins deterministically; the codes
//! mirror `MjErrorCode` in `shaders/common.h`.

/// Crate-wide result alias.
pub type Result<T, E = Error> = core::result::Result<T, E>;

/// All the ways metal-json can fail.
///
/// Several variants are not constructed yet — they belong to pipeline stages
/// that land in M1–M4 (see the design spec in `docs/superpowers/specs/`).
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
#[allow(dead_code)]
pub enum Error {
    /// Input is not valid UTF-8.
    #[error("invalid UTF-8 sequence at byte {offset}")]
    Utf8 { offset: u64 },

    /// Input violates the JSON grammar.
    #[error("JSON syntax error at byte {offset}: {kind}")]
    Syntax { offset: u64, kind: SyntaxErrorKind },

    /// Container nesting exceeds the supported depth (1024, simdjson parity).
    #[error("nesting depth exceeds the limit of {limit} at byte {offset}")]
    DepthLimit { offset: u64, limit: u32 },

    /// Non-whitespace bytes after the top-level value.
    #[error("trailing content after the top-level JSON value at byte {offset}")]
    TrailingContent { offset: u64 },

    /// Input larger than the parser supports.
    #[error("input of {len} bytes exceeds the maximum supported size of {max} bytes")]
    InputTooLarge { len: u64, max: u64 },

    /// No Metal device is available on this system.
    #[error("no Metal GPU device available (MTLCreateSystemDefaultDevice returned nil)")]
    NoDevice,

    /// The Metal device refused to create a command queue.
    #[error("failed to create a Metal command queue")]
    NoCommandQueue,

    /// Loading the precompiled metallib (or compiled source) into a
    /// `MTLLibrary` failed.
    #[error("failed to load Metal shader library: {message}")]
    LibraryLoad { message: String },

    /// Runtime MSL compilation failed (`runtime-shaders` feature).
    #[error("Metal shader compilation failed: {message}")]
    ShaderCompile { message: String },

    /// A kernel function is missing from the shader library.
    #[error("kernel function `{name}` not found in the Metal shader library")]
    KernelNotFound { name: String },

    /// Compute pipeline state creation failed.
    #[error("failed to create compute pipeline for kernel `{name}`: {message}")]
    PipelineCreate { name: String, message: String },

    /// GPU buffer allocation failed.
    #[error("failed to allocate a GPU buffer of {bytes} bytes")]
    BufferAlloc { bytes: usize },

    /// A committed command buffer completed with an error.
    #[error("GPU command buffer failed: {message}")]
    CommandBuffer { message: String },

    /// Filesystem error (e.g. `parse_file` mmap).
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Fine-grained classification for [`Error::Syntax`].
///
/// Placeholder set until M1/M3 pin down the exact kernel-side codes; kept in
/// sync with `MjErrorCode` in `shaders/common.h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[allow(dead_code)]
pub enum SyntaxErrorKind {
    /// A token that cannot follow the previous token (adjacency rule).
    UnexpectedToken,
    /// `{`/`[` without a matching `}`/`]`, or vice versa.
    UnbalancedBrackets,
    /// Malformed `true` / `false` / `null`.
    InvalidLiteral,
    /// Number violates the JSON number grammar.
    InvalidNumber,
    /// Invalid `\` escape (including bad `\uXXXX` / lone surrogate).
    InvalidStringEscape,
    /// String missing its closing quote.
    UnterminatedString,
    /// Unescaped control character (0x00..0x1F) inside a string.
    ControlCharacterInString,
    /// Object member missing the `:` separator.
    MissingColon,
    /// Missing or misplaced `,` between values.
    MissingComma,
    /// Empty (or whitespace-only) input.
    EmptyInput,
}

impl core::fmt::Display for SyntaxErrorKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let msg = match self {
            Self::UnexpectedToken => "unexpected token",
            Self::UnbalancedBrackets => "unbalanced brackets",
            Self::InvalidLiteral => "invalid literal",
            Self::InvalidNumber => "invalid number",
            Self::InvalidStringEscape => "invalid string escape",
            Self::UnterminatedString => "unterminated string",
            Self::ControlCharacterInString => "unescaped control character in string",
            Self::MissingColon => "missing ':' in object member",
            Self::MissingComma => "missing ',' between values",
            Self::EmptyInput => "empty input",
        };
        f.write_str(msg)
    }
}
