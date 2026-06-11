//! `Parser`: the reusable entry point.
//!
//! M1 surface: options ([`ParserOptions`], [`Backend`]) plus
//! [`Parser::parse`] / [`Parser::parse_file`]. The CPU reference backend
//! (`cpu-reference` feature) runs the scalar oracle pipeline in
//! [`crate::reference`]; the GPU backend is a stub until the Metal pipeline
//! lands over M2–M4, at which point `Parser` also grows the device /
//! pipeline-state / buffer-pool state and the zero-copy input paths
//! (`alloc_input` / `parse_aligned`, mmap-backed `parse_file`) from the
//! design spec.

use std::path::Path;

use crate::document::Document;
use crate::error::{Error, Result};

/// Container nesting limit applied by default (simdjson parity).
pub const DEFAULT_MAX_DEPTH: u32 = 1024;

/// Which pipeline executes a parse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Backend {
    /// The Metal GPU pipeline.
    ///
    /// **Not implemented yet** — it lands over M2–M4. Until then, parsing
    /// with this backend returns [`Error::CommandBuffer`] with the message
    /// `"GPU backend lands in M2-M4"`.
    Gpu,
    /// The scalar CPU oracle pipeline — produces the bit-identical target
    /// tape and is the correctness reference the GPU kernels are diffed
    /// against.
    #[cfg(feature = "cpu-reference")]
    CpuReference,
}

/// M1-interim default: [`Backend::CpuReference`] when the `cpu-reference`
/// feature is enabled (the only backend that can parse today), otherwise
/// [`Backend::Gpu`] (the documented stub until M2–M4). Once the GPU
/// pipeline lands, `Gpu` becomes the unconditional default.
impl Default for Backend {
    fn default() -> Self {
        #[cfg(feature = "cpu-reference")]
        {
            Self::CpuReference
        }
        #[cfg(not(feature = "cpu-reference"))]
        {
            Self::Gpu
        }
    }
}

/// Parse configuration.
///
/// Construct via [`Default`] and override fields (the struct is
/// `#[non_exhaustive]`; more knobs — buffer-pool sizing, etc. — arrive with
/// the GPU pipeline):
///
/// ```
/// use metal_json::ParserOptions;
///
/// let mut opts = ParserOptions::default();
/// opts.max_depth = 64;
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ParserOptions {
    /// Maximum container nesting depth; deeper input fails with
    /// [`Error::DepthLimit`]. Defaults to [`DEFAULT_MAX_DEPTH`] (1024,
    /// simdjson parity).
    pub max_depth: u32,
    /// Backend selection; defaults to [`Backend::default()`] — during M1
    /// that is [`Backend::CpuReference`] with the `cpu-reference` feature
    /// enabled (so a default-constructed parser can actually parse), and
    /// the [`Backend::Gpu`] stub otherwise.
    pub backend: Backend,
}

impl Default for ParserOptions {
    fn default() -> Self {
        Self {
            max_depth: DEFAULT_MAX_DEPTH,
            backend: Backend::default(),
        }
    }
}

/// A reusable JSON parser.
///
/// In M1 this only carries [`ParserOptions`]; from M2 it also owns the
/// Metal device, pipeline states and buffer pool (created once, reused
/// across parses — the reason construction is fallible).
#[derive(Debug, Clone)]
pub struct Parser {
    opts: ParserOptions,
}

impl Parser {
    /// Create a parser with default options.
    ///
    /// M1-interim behavior: with the `cpu-reference` feature enabled the
    /// default backend is [`Backend::CpuReference`], so this parser parses
    /// real documents today. Without the feature the default is the
    /// [`Backend::Gpu`] stub, which fails every parse until the Metal
    /// pipeline lands (M2–M4).
    pub fn new() -> Result<Self> {
        Self::with_options(ParserOptions::default())
    }

    /// Create a parser with explicit [`ParserOptions`].
    pub fn with_options(opts: ParserOptions) -> Result<Self> {
        Ok(Self { opts })
    }

    /// The options this parser was created with.
    #[must_use]
    pub fn options(&self) -> &ParserOptions {
        &self.opts
    }

    /// Parse a JSON document from a byte slice.
    ///
    /// # Errors
    ///
    /// Any parse failure from the selected backend (syntax, UTF-8, depth,
    /// ...). With [`Backend::Gpu`] this currently always returns
    /// [`Error::CommandBuffer`] — the GPU pipeline lands in M2–M4.
    pub fn parse(&self, json: &[u8]) -> Result<Document> {
        match self.opts.backend {
            Backend::Gpu => {
                // The GPU pipeline consumes `json` from M2 on.
                let _ = json;
                Err(Error::CommandBuffer {
                    message: "GPU backend lands in M2-M4".to_owned(),
                })
            }
            #[cfg(feature = "cpu-reference")]
            Backend::CpuReference => {
                let (tape, strings) = crate::reference::parse(json, &self.opts)?;
                Ok(Document::from_parts(tape, strings))
            }
        }
    }

    /// Parse a JSON file.
    ///
    /// M1 reads the file into memory and delegates to
    /// [`parse`](Self::parse); the mmap-backed zero-copy path arrives with
    /// the GPU backend (M2+).
    ///
    /// # Errors
    ///
    /// [`Error::Io`] if the file cannot be read, otherwise as
    /// [`parse`](Self::parse).
    pub fn parse_file(&self, path: impl AsRef<Path>) -> Result<Document> {
        let bytes = std::fs::read(path)?;
        self.parse(&bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_options() {
        let opts = ParserOptions::default();
        assert_eq!(opts.max_depth, 1024);
        // M1-interim default backend: the CPU oracle when available,
        // otherwise the (stub) GPU backend.
        #[cfg(feature = "cpu-reference")]
        assert_eq!(opts.backend, Backend::CpuReference);
        #[cfg(not(feature = "cpu-reference"))]
        assert_eq!(opts.backend, Backend::Gpu);
    }

    #[test]
    fn new_uses_default_options() {
        let parser = Parser::new().unwrap();
        assert_eq!(parser.options().max_depth, DEFAULT_MAX_DEPTH);
        assert_eq!(parser.options().backend, Backend::default());
    }

    #[test]
    fn with_options_stores_the_options() {
        let opts = ParserOptions {
            max_depth: 32,
            ..ParserOptions::default()
        };
        let parser = Parser::with_options(opts).unwrap();
        assert_eq!(parser.options().max_depth, 32);
    }

    #[test]
    fn gpu_backend_is_a_documented_stub_until_m2() {
        let opts = ParserOptions {
            backend: Backend::Gpu,
            ..ParserOptions::default()
        };
        let parser = Parser::with_options(opts).unwrap();
        let err = parser.parse(b"null").unwrap_err();
        match err {
            Error::CommandBuffer { message } => {
                assert_eq!(message, "GPU backend lands in M2-M4");
            }
            other => panic!("expected CommandBuffer error, got {other:?}"),
        }
    }

    /// Public-API check of the M1-interim default: a default-constructed
    /// parser must be able to parse (finding: an unusable `Parser::new`).
    #[cfg(feature = "cpu-reference")]
    #[test]
    fn default_parser_parses_null_and_roundtrips() {
        let parser = Parser::new().unwrap();
        let doc = parser.parse(b"null").expect("default parser must parse");
        assert!(doc.root().is_null());
        // Minimal valid tape: r, n, r.
        assert_eq!(doc.tape().len(), 3);
    }

    /// A nested document through the default constructor, end to end.
    #[cfg(feature = "cpu-reference")]
    #[test]
    fn default_parser_parses_a_nested_document() {
        let parser = Parser::new().unwrap();
        let doc = parser
            .parse(br#"{"a":[1,2.5],"b":"x\n"}"#)
            .expect("default parser must parse");
        let root = doc.root();
        assert_eq!(root.len(), Some(2));
        let a = root.get("a").unwrap();
        assert_eq!(a.at(0).unwrap().as_i64(), Some(1));
        assert_eq!(a.at(1).unwrap().as_f64(), Some(2.5));
        assert_eq!(root.get("b").unwrap().as_str(), Some("x\n"));
    }

    #[test]
    fn parse_file_surfaces_io_errors() {
        let parser = Parser::new().unwrap();
        let err = parser
            .parse_file("/nonexistent/metal-json-no-such-file.json")
            .unwrap_err();
        assert!(matches!(err, Error::Io(_)), "got {err:?}");
    }
}
