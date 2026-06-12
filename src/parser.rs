//! `Parser`: the reusable entry point.
//!
//! From M4 on the default backend is the **Metal GPU pipeline**
//! ([`Backend::Gpu`]) on every machine that has a usable Metal device:
//! [`Parser::new`] acquires the system default device once and every
//! [`Parser::parse`] drives the full CB1→CB2→CB3 pipeline in
//! [`crate::gpu::pipeline`], building a [`Document`] from the finished
//! tape/string buffers.
//!
//! # Default-backend policy (pinned by `default_backend_policy` below)
//!
//! [`Backend::default`] — and therefore [`ParserOptions::default`], i.e.
//! the selection happens at **options-construction time**, via a cached
//! one-per-process device probe — resolves to:
//!
//! 1. [`Backend::Gpu`] when a Metal device is creatable on this machine;
//! 2. else `Backend::CpuReference` when the `cpu-reference` feature is
//!    compiled in (the scalar oracle parses correctly anywhere — an
//!    ergonomic default for non-Metal hosts, M1 parity);
//! 3. else [`Backend::Gpu`] anyway, so [`Parser::new`] surfaces a clear
//!    [`Error::NoDevice`].
//!
//! An **explicitly selected** `Backend::Gpu` is GPU-strict: construction
//! fails with [`Error::NoDevice`] rather than silently falling back — an
//! explicit choice is never second-guessed (and the reference backend
//! stays what it is: the bit-exact oracle the GPU is diffed against, not
//! a runtime escape hatch).
//!
//! # The M5 fast path (what a steady-state parse costs)
//!
//! - **Buffer pool**: every parse checks its buffers out of the parser's
//!   [`ScratchPool`] (shared by clones); scratch returns at the end of the
//!   parse, the tape/string buffers when their [`Document`] drops. Steady
//!   state does **zero** large allocations; capacities grow-and-keep (see
//!   `crate::pool`).
//! - **Zero-copy `Document`**: the tape and string buffer stay in the
//!   shared GPU memory the kernels wrote; [`Document`] reads them in place.
//! - **Zero-copy input**: [`Parser::parse_aligned`] (caller-held
//!   [`AlignedInput`]) and the **unsafe** [`Parser::parse_file_mmap`]
//!   (private file mapping; the caller guarantees the file is not
//!   truncated or modified mid-parse) wrap the input pages with
//!   `bytesNoCopy` — no input byte is copied. [`Parser::parse`] and the
//!   safe [`Parser::parse_file`] copy once, into a pooled buffer
//!   (`parse_file` reads the file directly into it, no intermediate
//!   allocation).

use std::path::Path;
use std::rc::Rc;
use std::sync::{Arc, OnceLock};

use crate::document::Document;
use crate::error::Result;
// Referenced by doc links (and the test module); the parse paths construct
// errors through `decode_packed_error` / the reference pipeline.
#[allow(unused_imports)]
use crate::error::Error;
use crate::gpu::pipeline::{GpuInput, GpuParse, GpuPipeline, decode_packed_error};
use crate::input::AlignedInput;
use crate::metal::{GpuBuffer, MetalContext};
use crate::pool::ScratchPool;
use crate::stage::{MAX_INPUT_BYTES, WORD_BYTES};
use crate::tape::{StringBuffer, TapeBuffer};

/// Container nesting limit applied by default (simdjson parity).
pub const DEFAULT_MAX_DEPTH: u32 = 1024;

/// Which pipeline executes a parse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Backend {
    /// The Metal GPU pipeline (the default whenever a device exists).
    /// Construction requires a Metal device: [`Parser::with_options`]
    /// returns [`Error::NoDevice`] (or another device/library error) when
    /// none is usable — explicit selection is GPU-strict, never silently
    /// downgraded (see the module-level default-backend policy).
    Gpu,
    /// The scalar CPU oracle pipeline — produces the bit-identical target
    /// tape and is the correctness reference the GPU pipeline is diffed
    /// against. Needs no Metal device.
    #[cfg(feature = "cpu-reference")]
    CpuReference,
}

/// Whether a usable Metal device exists on this machine. Probed at most
/// once per process (a throwaway [`MetalContext`] is created and dropped;
/// the result is cached in a `OnceLock`), so calling
/// [`ParserOptions::default`] repeatedly stays cheap.
fn metal_device_available() -> bool {
    static PROBE: OnceLock<bool> = OnceLock::new();
    *PROBE.get_or_init(|| MetalContext::new().is_ok())
}

/// The default-backend policy with the device probe injected — separated
/// from [`metal_device_available`] so the probed-false branch is unit
/// testable on machines that do have a GPU. See the module docs for the
/// policy's rationale.
fn resolve_default_backend(gpu_available: bool) -> Backend {
    if !gpu_available {
        #[cfg(feature = "cpu-reference")]
        return Backend::CpuReference;
    }
    // A device is available — or there is no CPU oracle compiled in, in
    // which case Gpu is still the default and Parser construction surfaces
    // a clear Error::NoDevice.
    Backend::Gpu
}

/// The module-level default-backend policy: [`Backend::Gpu`] when a Metal
/// device is creatable on this machine (cached one-shot probe), else
/// `Backend::CpuReference` when the `cpu-reference` feature is enabled,
/// else [`Backend::Gpu`] (which surfaces [`Error::NoDevice`]). Resolution
/// happens here — i.e. at [`ParserOptions::default`] time, not at parse
/// time.
impl Default for Backend {
    fn default() -> Self {
        resolve_default_backend(metal_device_available())
    }
}

/// Parse configuration.
///
/// Construct via [`Default`] and override fields (the struct is
/// `#[non_exhaustive]`; more knobs — buffer-pool sizing, etc. — arrive with
/// M5):
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
    /// Backend selection; defaults to [`Backend::default`] (the documented
    /// device-probe policy — [`Backend::Gpu`] whenever this machine has a
    /// Metal device).
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

/// The per-parser GPU state: the Metal device/queue/library plus the
/// pipeline's lazily-built cached pipeline-state objects. Created once in
/// [`Parser::with_options`], shared by clones (`Rc` — the lazily-cached
/// pipeline states are not `Sync`, so neither is `Parser`; the audited
/// `Send`/`Sync` story is M5 work alongside the buffer pool), reused
/// across parses.
#[derive(Debug)]
struct GpuState {
    ctx: MetalContext,
    pipeline: GpuPipeline,
    /// The M5 buffer pool. `Arc` because every GPU-backed [`Document`]
    /// holds a handle (its tape/string buffers return here on drop, and
    /// the pool must outlive any document even if the parser is gone).
    pool: Arc<ScratchPool>,
}

/// A reusable JSON parser.
///
/// Owns the Metal device and pipeline states for the [`Backend::Gpu`]
/// backend (created once — the reason construction is fallible — and
/// shared by clones); the M5 buffer pool will live here too. Not yet
/// `Send`/`Sync` (see `GpuState`).
#[derive(Debug, Clone)]
pub struct Parser {
    opts: ParserOptions,
    /// Present iff the backend is [`Backend::Gpu`].
    gpu: Option<Rc<GpuState>>,
}

impl Parser {
    /// Create a parser with default options (the [`Backend::default`]
    /// policy: GPU when a Metal device exists, the CPU oracle when it does
    /// not and `cpu-reference` is compiled in).
    ///
    /// # Errors
    ///
    /// [`Error::NoDevice`] when no Metal device is usable and no CPU
    /// fallback is compiled in (plus the other device/library construction
    /// failures). An explicit `Backend::Gpu` in [`Parser::with_options`]
    /// is never downgraded — only the *default* selection probes.
    pub fn new() -> Result<Self> {
        Self::with_options(ParserOptions::default())
    }

    /// Create a parser with explicit [`ParserOptions`].
    ///
    /// # Errors
    ///
    /// As [`new`](Self::new) for [`Backend::Gpu`]; infallible for
    /// `Backend::CpuReference`.
    pub fn with_options(opts: ParserOptions) -> Result<Self> {
        let gpu = match opts.backend {
            Backend::Gpu => Some(Rc::new(GpuState {
                ctx: MetalContext::new()?,
                pipeline: GpuPipeline::new(),
                pool: Arc::new(ScratchPool::new()),
            })),
            #[cfg(feature = "cpu-reference")]
            Backend::CpuReference => None,
        };
        Ok(Self { opts, gpu })
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
    /// Any parse failure from the selected backend: [`Error::Utf8`],
    /// [`Error::Syntax`], [`Error::DepthLimit`], [`Error::TrailingContent`],
    /// [`Error::InputTooLarge`] — plus, on the GPU backend, the internal
    /// device/command-buffer failures. Both backends reject the same
    /// inputs; on multi-error documents they may differ in *which* error
    /// they report (the GPU reports the earliest byte offset, the reference
    /// walks in stage order — the documented WHETHER-not-WHICH relaxation).
    pub fn parse(&self, json: &[u8]) -> Result<Document> {
        match self.opts.backend {
            Backend::Gpu => {
                let gpu = self.gpu.as_ref().expect("Gpu backend constructed its state");
                let parse = gpu.pipeline.run_pooled(
                    &gpu.ctx,
                    &gpu.pool,
                    GpuInput::Bytes(json),
                    self.opts.max_depth,
                )?;
                self.finish_gpu(parse)
            }
            #[cfg(feature = "cpu-reference")]
            Backend::CpuReference => {
                let (tape, strings) = crate::reference::parse(json, &self.opts)?;
                Ok(Document::from_parts(tape, strings))
            }
        }
    }

    /// Parse a JSON document and deserialize it into an owned serde data
    /// model.
    ///
    /// This convenience path returns only `T`, so `T` must own its data.
    /// Use [`Document::deserialize`](crate::Document::deserialize) or
    /// [`crate::serde::from_document`] when `T` contains borrowed string
    /// fields.
    ///
    /// # Errors
    ///
    /// Any parse failure from [`parse`](Self::parse), or a serde
    /// deserialization error if the parsed document does not match `T`.
    #[cfg(feature = "serde")]
    pub fn parse_deserialize<T>(&self, json: &[u8]) -> Result<T>
    where
        T: ::serde::de::DeserializeOwned,
    {
        let doc = self.parse(json)?;
        Ok(crate::serde::from_document(&doc)?)
    }

    /// Parse from a caller-held [`AlignedInput`] — the **zero-copy** input
    /// path: the input pages are wrapped with `MTLBuffer bytesNoCopy` and
    /// read by the GPU in place. Build the `AlignedInput` once, parse from
    /// it as often as needed.
    ///
    /// On the `cpu-reference` backend this is equivalent to
    /// [`parse`](Self::parse) (the oracle reads any slice).
    ///
    /// # Errors
    ///
    /// As [`parse`](Self::parse).
    pub fn parse_aligned(&self, input: &AlignedInput) -> Result<Document> {
        match self.opts.backend {
            Backend::Gpu => {
                let gpu = self.gpu.as_ref().expect("Gpu backend constructed its state");
                // SAFETY: `AlignedInput` guarantees a 16 KiB-aligned,
                // page-multiple, readable+writable allocation with a
                // space-padded tail; the wrapper is dropped inside
                // `run_pooled` (with the parse's stage-1 buffers), strictly
                // before the `&input` borrow ends, and no `&mut` to the
                // input can exist while that borrow is live.
                let buffer = unsafe {
                    GpuBuffer::from_page_aligned(&gpu.ctx, input.base_ptr(), input.len())?
                };
                let parse = gpu.pipeline.run_pooled(
                    &gpu.ctx,
                    &gpu.pool,
                    GpuInput::External {
                        buffer,
                        len: input.len(),
                    },
                    self.opts.max_depth,
                )?;
                self.finish_gpu(parse)
            }
            #[cfg(feature = "cpu-reference")]
            Backend::CpuReference => self.parse(input),
        }
    }

    /// Parse a JSON file. On the GPU backend the file is read **once**,
    /// directly into a pooled page-aligned GPU buffer (no intermediate
    /// allocation), space-padded like every other input path — the same
    /// one-copy cost as [`parse`](Self::parse). The returned [`Document`]
    /// never borrows the input.
    ///
    /// This function is safe against concurrent modification of the file:
    /// it reads through ordinary file I/O, so a racing truncation surfaces
    /// as [`Error::Io`], never a crash. Callers who can guarantee the file
    /// is not modified during the parse can use the zero-copy
    /// [`parse_file_mmap`](Self::parse_file_mmap) instead.
    ///
    /// # Errors
    ///
    /// [`Error::Io`] if the file cannot be opened or read (including a
    /// concurrent truncation mid-read), otherwise as
    /// [`parse`](Self::parse).
    pub fn parse_file(&self, path: impl AsRef<Path>) -> Result<Document> {
        match self.opts.backend {
            Backend::Gpu => self.parse_file_gpu_copy(path.as_ref()),
            #[cfg(feature = "cpu-reference")]
            Backend::CpuReference => {
                let bytes = std::fs::read(path)?;
                self.parse(&bytes)
            }
        }
    }

    /// Parse a JSON file via a private (copy-on-write) mmap — the
    /// **zero-copy** file input path on the GPU backend: the mapped pages
    /// are wrapped with `MTLBuffer bytesNoCopy` and read by the GPU in
    /// place, the tail of the last 64-byte word space-padded in the
    /// private mapping (the file is never written). The mapping is held
    /// for the parse duration only; the returned [`Document`] never
    /// borrows the input. Behaves identically to
    /// [`parse_file`](Self::parse_file) otherwise — including on the
    /// `cpu-reference` backend, where it reads the file normally (the
    /// oracle parses any slice; no mapping is created).
    ///
    /// # Safety
    ///
    /// The file must not be **truncated or modified** — by this process or
    /// any other — for the entire duration of the call. A `MAP_PRIVATE`
    /// (copy-on-write) mapping does not protect against concurrent
    /// truncation: if the file shrinks while the parse (CPU padding write,
    /// GPU kernels, or the CPU fixup passes) touches a page that no longer
    /// has file backing, the process receives `SIGBUS` — i.e. undefined
    /// behavior, not an `Err`. Concurrent *writes* that keep the length
    /// can also tear the mapped bytes mid-parse, violating the memory
    /// model the parse relies on. If you cannot rule both out, use the
    /// safe, copying [`parse_file`](Self::parse_file).
    ///
    /// # Errors
    ///
    /// [`Error::Io`] if the file cannot be opened or mapped, otherwise as
    /// [`parse`](Self::parse).
    pub unsafe fn parse_file_mmap(&self, path: impl AsRef<Path>) -> Result<Document> {
        match self.opts.backend {
            Backend::Gpu => self.parse_file_gpu_mmap(path.as_ref()),
            #[cfg(feature = "cpu-reference")]
            Backend::CpuReference => {
                let bytes = std::fs::read(path)?;
                self.parse(&bytes)
            }
        }
    }

    /// The safe `parse_file` body: open, size-check, then read straight
    /// into a pooled space-padded buffer (one copy, no intermediate `Vec`)
    /// and hand it to the pipeline as [`GpuInput::Pooled`].
    fn parse_file_gpu_copy(&self, path: &Path) -> Result<Document> {
        use std::io::Read;

        let gpu = self.gpu.as_ref().expect("Gpu backend constructed its state");
        let mut file = std::fs::File::open(path)?;
        let file_len = file.metadata()?.len();
        if file_len > MAX_INPUT_BYTES {
            return Err(Error::InputTooLarge {
                len: file_len,
                max: MAX_INPUT_BYTES,
            });
        }
        let len = usize::try_from(file_len).expect("checked against MAX_INPUT_BYTES");
        if len == 0 {
            // The empty verdict needs no GPU (and no buffer).
            return self.parse(&[]);
        }

        // The one copy: file → pooled GPU buffer, padded with ASCII spaces
        // to the 64-byte word boundary (the stage-1 invariant every input
        // path establishes). A concurrent truncation makes read_exact
        // return UnexpectedEof → Error::Io — never a fault.
        let padded_len = len.next_multiple_of(WORD_BYTES);
        let mut buffer = gpu.pool.checkout(&gpu.ctx, padded_len)?;
        let contents = buffer.contents_mut();
        file.read_exact(&mut contents[..len])?;
        contents[len..].fill(b' ');

        let parse = gpu.pipeline.run_pooled(
            &gpu.ctx,
            &gpu.pool,
            GpuInput::Pooled { buffer, len },
            self.opts.max_depth,
        )?;
        self.finish_gpu(parse)
    }

    /// The `parse_file_mmap` body (the caller upholds the no-modification
    /// contract; see `parse_file_mmap`'s `# Safety`).
    fn parse_file_gpu_mmap(&self, path: &Path) -> Result<Document> {
        let gpu = self.gpu.as_ref().expect("Gpu backend constructed its state");
        let file = std::fs::File::open(path)?;
        let file_len = file.metadata()?.len();
        if file_len > MAX_INPUT_BYTES {
            return Err(Error::InputTooLarge {
                len: file_len,
                max: MAX_INPUT_BYTES,
            });
        }
        let len = usize::try_from(file_len).expect("checked against MAX_INPUT_BYTES");
        if len == 0 {
            // mmap rejects zero-length mappings; the empty verdict needs no
            // GPU anyway.
            return self.parse(&[]);
        }

        // Map privately (copy-on-write): the kernels need the bytes between
        // `len` and the next 64-byte word boundary to be ASCII spaces (the
        // same padding the copied path writes), and a COW mapping lets that
        // tail be written without touching the file. The padded length
        // never crosses into a page beyond the file's last page (64 divides
        // the page size), so every touched byte is file-backed — as long as
        // the caller-guaranteed no-truncation contract holds.
        let padded_len = len.next_multiple_of(WORD_BYTES);
        let mut map = unsafe { memmap2::MmapOptions::new().len(padded_len).map_copy(&file) }?;
        map[len..padded_len].fill(b' ');

        let ptr = core::ptr::NonNull::new(map.as_mut_ptr()).expect("mmap never returns null");
        // SAFETY: mmap returns page-aligned memory; the mapping covers
        // `len.next_multiple_of(PAGE_SIZE)` bytes of one valid
        // readable+writable (MAP_PRIVATE) allocation; `map` outlives the
        // wrapper, which is dropped inside `run_pooled`; the CPU does not
        // touch the mapping after the space-fill above. The caller of
        // `parse_file_mmap` guarantees the underlying file is neither
        // truncated nor modified for the duration of the call (unfaulted
        // pages of a COW mapping otherwise SIGBUS on access).
        let buffer = unsafe { GpuBuffer::from_page_aligned(&gpu.ctx, ptr, len)? };
        let parse = gpu.pipeline.run_pooled(
            &gpu.ctx,
            &gpu.pool,
            GpuInput::External { buffer, len },
            self.opts.max_depth,
        )?;
        // `map` stays alive until here — past the GPU work and the wrapper's
        // drop — then unmaps.
        drop(map);
        self.finish_gpu(parse)
    }

    /// Shared GPU epilogue: decode rejections; wrap accepted tape/string
    /// buffers into the zero-copy [`Document`] (the buffers return to the
    /// pool when it drops).
    fn finish_gpu(&self, parse: GpuParse) -> Result<Document> {
        let gpu = self.gpu.as_ref().expect("Gpu backend constructed its state");
        match parse {
            GpuParse::Rejected(packed) => Err(decode_packed_error(packed, self.opts.max_depth)),
            GpuParse::Accepted(out) => {
                let tape = TapeBuffer::from_gpu(out.tape, Arc::clone(&gpu.pool));
                let strings = match out.stringbuf {
                    Some(buf) => StringBuffer::from_gpu(buf, Arc::clone(&gpu.pool)),
                    None => StringBuffer::new(),
                };
                Ok(Document::from_parts(tape, strings))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::SyntaxErrorKind;

    /// GPU gating, as everywhere else: skip without a device unless
    /// `METAL_JSON_REQUIRE_GPU=1` makes that a hard failure.
    fn gpu_parser_or_skip(test: &str) -> Option<Parser> {
        match Parser::new() {
            // `Parser::new` succeeding is not enough: when the device probe
            // fails (no device, or METAL_JSON_DISABLE_GPU=1) the default
            // backend falls back to the CPU reference — these tests want
            // the GPU specifically.
            Ok(parser) if parser.options().backend == Backend::Gpu => Some(parser),
            Ok(_) => {
                if std::env::var_os("METAL_JSON_REQUIRE_GPU").is_some_and(|v| v == "1") {
                    panic!("METAL_JSON_REQUIRE_GPU=1 but the default backend is not Gpu");
                }
                eprintln!("SKIP {test}: no usable Metal device here (CPU fallback default)");
                None
            }
            Err(err) => {
                if std::env::var_os("METAL_JSON_REQUIRE_GPU").is_some_and(|v| v == "1") {
                    panic!("METAL_JSON_REQUIRE_GPU=1 but no usable Metal device: {err}");
                }
                eprintln!("SKIP {test}: no usable Metal device here ({err})");
                None
            }
        }
    }

    #[test]
    fn default_options() {
        let opts = ParserOptions::default();
        assert_eq!(opts.max_depth, 1024);
        // Backend selection happens at options-default time, per the
        // pinned policy (default_backend_policy below).
        assert_eq!(opts.backend, Backend::default());
    }

    /// THE default-backend policy pin, with **no silent skips**: the
    /// resolution function is asserted for both probe outcomes (the
    /// probed-false branch injected — faking device absence end-to-end is
    /// not attempted), and the live default is asserted to follow whatever
    /// the real probe says, including a working end-to-end parse.
    #[test]
    fn default_backend_policy() {
        // Probe injected: a machine with a device always defaults to Gpu.
        assert_eq!(resolve_default_backend(true), Backend::Gpu);
        // Probe injected: no device falls back to the CPU oracle when it
        // is compiled in, else stays Gpu (surfacing NoDevice clearly).
        #[cfg(feature = "cpu-reference")]
        assert_eq!(resolve_default_backend(false), Backend::CpuReference);
        #[cfg(not(feature = "cpu-reference"))]
        assert_eq!(resolve_default_backend(false), Backend::Gpu);

        // The live default follows the real probe — asserted on BOTH
        // probe outcomes (no early return).
        if metal_device_available() {
            assert_eq!(Backend::default(), Backend::Gpu);
            let parser = Parser::new().expect("probe says a device exists");
            assert_eq!(parser.options().backend, Backend::Gpu);
            let doc = parser.parse(b"[1,2]").expect("default backend parses");
            assert_eq!(doc.root().len(), Some(2));
        } else {
            if std::env::var_os("METAL_JSON_REQUIRE_GPU").is_some_and(|v| v == "1") {
                panic!("METAL_JSON_REQUIRE_GPU=1 but the device probe failed");
            }
            assert_eq!(Backend::default(), resolve_default_backend(false));
            #[cfg(feature = "cpu-reference")]
            {
                let parser =
                    Parser::new().expect("CpuReference default construction is infallible");
                assert_eq!(parser.options().backend, Backend::CpuReference);
                let doc = parser.parse(b"[1,2]").expect("fallback backend parses");
                assert_eq!(doc.root().len(), Some(2));
            }
        }
    }

    /// Explicit `Backend::Gpu` is GPU-strict: whatever the default policy
    /// does, explicitly selecting the GPU must construct iff a device
    /// exists (no silent fallback in either direction).
    #[test]
    fn explicit_gpu_backend_is_never_second_guessed() {
        let opts = ParserOptions {
            backend: Backend::Gpu,
            ..ParserOptions::default()
        };
        match Parser::with_options(opts) {
            Ok(parser) => {
                assert!(
                    metal_device_available(),
                    "explicit Gpu constructed without a device?"
                );
                assert_eq!(parser.options().backend, Backend::Gpu);
            }
            Err(err) => {
                assert!(
                    !metal_device_available(),
                    "explicit Gpu failed despite a usable device: {err}"
                );
            }
        }
    }

    #[test]
    fn new_uses_default_options() {
        let Some(parser) = gpu_parser_or_skip("new_uses_default_options") else {
            return;
        };
        assert_eq!(parser.options().max_depth, DEFAULT_MAX_DEPTH);
        assert_eq!(parser.options().backend, Backend::Gpu);
    }

    #[test]
    fn with_options_stores_the_options() {
        if gpu_parser_or_skip("with_options_stores_the_options").is_none() {
            return;
        }
        let opts = ParserOptions {
            max_depth: 32,
            ..ParserOptions::default()
        };
        let parser = Parser::with_options(opts).expect("device exists: probe succeeded above");
        assert_eq!(parser.options().max_depth, 32);
    }

    /// A default-constructed parser parses real documents end to end on
    /// the GPU.
    #[test]
    fn default_parser_parses_the_worked_example() {
        let Some(parser) = gpu_parser_or_skip("default_parser_parses_the_worked_example") else {
            return;
        };
        let doc = parser
            .parse(br#"{"a":[1,2.5],"b":"x\n"}"#)
            .expect("default parser must parse");
        let root = doc.root();
        assert_eq!(root.len(), Some(2));
        let a = root.get("a").unwrap();
        assert_eq!(a.at(0).unwrap().as_i64(), Some(1));
        assert_eq!(a.at(1).unwrap().as_f64(), Some(2.5));
        assert_eq!(root.get("b").unwrap().as_str(), Some("x\n"));
        // The docs/tape-format.md worked example, bit-for-bit.
        let expected: [u64; 13] = [
            0x7200_0000_0000_000C,
            0x7B00_0002_0000_000C,
            0x2200_0000_0000_0000,
            0x5B00_0002_0000_0009,
            0x6C00_0000_0000_0000,
            1,
            0x6400_0000_0000_0000,
            0x4004_0000_0000_0000,
            0x5D00_0000_0000_0003,
            0x2200_0000_0000_0006,
            0x2200_0000_0000_000C,
            0x7D00_0000_0000_0001,
            0x7200_0000_0000_0000,
        ];
        assert_eq!(doc.tape(), expected);
    }

    /// Root scalars and whitespace-padded roots produce correct full tapes
    /// through the GPU backend (the CB3-structure-skipped path).
    #[test]
    fn root_scalar_documents_parse_on_the_gpu() {
        let Some(parser) = gpu_parser_or_skip("root_scalar_documents_parse_on_the_gpu") else {
            return;
        };
        let doc = parser.parse(b"42").unwrap();
        assert_eq!(doc.root().as_i64(), Some(42));
        assert_eq!(doc.tape().len(), 4);

        let doc = parser.parse(b"true").unwrap();
        assert_eq!(doc.root().as_bool(), Some(true));
        assert_eq!(doc.tape().len(), 3);

        let doc = parser.parse(b"  null \n").unwrap();
        assert!(doc.root().is_null());

        let doc = parser.parse(br#""xA""#).unwrap();
        assert_eq!(doc.root().as_str(), Some("xA"));

        let doc = parser.parse(b"-0.0").unwrap();
        assert_eq!(
            doc.root().as_f64().map(f64::to_bits),
            Some((-0.0f64).to_bits())
        );
    }

    /// Empty / whitespace-only input is the reference's EmptyInput verdict.
    #[test]
    fn empty_and_whitespace_only_inputs_error_as_empty_input() {
        let Some(parser) =
            gpu_parser_or_skip("empty_and_whitespace_only_inputs_error_as_empty_input")
        else {
            return;
        };
        for input in [&b""[..], b" ", b" \t\n\r"] {
            match parser.parse(input) {
                Err(Error::Syntax {
                    offset: 0,
                    kind: SyntaxErrorKind::EmptyInput,
                }) => {}
                other => panic!("{input:?}: expected EmptyInput at 0, got {other:?}"),
            }
        }
    }

    /// One public-error pin per GPU error class (the packed-code mapping's
    /// integration-level counterpart; the completeness test lives in
    /// `crate::gpu::pipeline`).
    #[test]
    fn gpu_errors_map_to_the_public_error_enum() {
        let Some(parser) = gpu_parser_or_skip("gpu_errors_map_to_the_public_error_enum") else {
            return;
        };
        match parser.parse(b"ab\x80") {
            Err(Error::Utf8 { offset: 2 }) => {}
            other => panic!("utf8: {other:?}"),
        }
        match parser.parse(b"[1 true]") {
            Err(Error::Syntax {
                offset: 3,
                kind: SyntaxErrorKind::MissingComma,
            }) => {}
            other => panic!("missing comma: {other:?}"),
        }
        match parser.parse(b"[1") {
            Err(Error::Syntax {
                offset: 0,
                kind: SyntaxErrorKind::UnbalancedBrackets,
            }) => {}
            other => panic!("unbalanced: {other:?}"),
        }
        match parser.parse(b"{},1") {
            Err(Error::TrailingContent { offset: 2 }) => {}
            other => panic!("trailing: {other:?}"),
        }
        match parser.parse(b"[01]") {
            Err(Error::Syntax {
                offset: 1,
                kind: SyntaxErrorKind::InvalidNumber,
            }) => {}
            other => panic!("number: {other:?}"),
        }
        match parser.parse(br#"["\q"]"#) {
            Err(Error::Syntax {
                offset: 2,
                kind: SyntaxErrorKind::InvalidStringEscape,
            }) => {}
            other => panic!("escape: {other:?}"),
        }
        match parser.parse(b"[\"a\x01\"]") {
            Err(Error::Syntax {
                offset: 3,
                kind: SyntaxErrorKind::ControlCharacterInString,
            }) => {}
            other => panic!("control: {other:?}"),
        }
        // Odd quotes: UnterminatedString at the documented provisional
        // offset (input_len; the reference points at the open quote).
        match parser.parse(b"\"abc") {
            Err(Error::Syntax {
                offset: 4,
                kind: SyntaxErrorKind::UnterminatedString,
            }) => {}
            other => panic!("unterminated: {other:?}"),
        }
        // DepthLimit carries the configured limit.
        let deep = Parser::with_options(ParserOptions {
            max_depth: 3,
            ..ParserOptions::default()
        })
        .expect("device exists: gpu_parser_or_skip succeeded above");
        match deep.parse(b"[[[[]]]]") {
            Err(Error::DepthLimit {
                offset: 3,
                limit: 3,
            }) => {}
            other => panic!("depth: {other:?}"),
        }
    }

    #[test]
    fn parse_file_surfaces_io_errors() {
        let Some(parser) = gpu_parser_or_skip("parse_file_surfaces_io_errors") else {
            return;
        };
        let err = parser
            .parse_file("/nonexistent/metal-json-no-such-file.json")
            .unwrap_err();
        assert!(matches!(err, Error::Io(_)), "got {err:?}");
    }

    // --- vs the cpu-reference oracle (the M4 integration differential) ----

    #[cfg(feature = "cpu-reference")]
    mod vs_reference {
        use super::*;
        use crate::value::{Value, ValueKind};

        fn cpu_parser() -> Parser {
            let opts = ParserOptions {
                backend: Backend::CpuReference,
                ..ParserOptions::default()
            };
            Parser::with_options(opts).expect("CPU reference parser construction cannot fail")
        }

        fn cpu_parser_with_depth(max_depth: u32) -> Parser {
            let opts = ParserOptions {
                max_depth,
                backend: Backend::CpuReference,
            };
            Parser::with_options(opts).expect("CPU reference parser construction cannot fail")
        }

        /// Walk two documents' value trees and require equality: kinds,
        /// scalar values (doubles bit-for-bit), string contents, array
        /// elements in order, object members in order (keys included —
        /// duplicates and all).
        fn assert_values_eq(gpu: Value<'_>, cpu: Value<'_>, path: &str) {
            assert_eq!(gpu.kind(), cpu.kind(), "{path}: kind");
            match cpu.kind() {
                ValueKind::Null => assert!(gpu.is_null(), "{path}"),
                ValueKind::Bool => assert_eq!(gpu.as_bool(), cpu.as_bool(), "{path}"),
                ValueKind::Int64 => assert_eq!(gpu.as_i64(), cpu.as_i64(), "{path}"),
                ValueKind::UInt64 => assert_eq!(gpu.as_u64(), cpu.as_u64(), "{path}"),
                ValueKind::Double => assert_eq!(
                    gpu.as_f64().map(f64::to_bits),
                    cpu.as_f64().map(f64::to_bits),
                    "{path}: f64 bits"
                ),
                ValueKind::String => assert_eq!(gpu.as_str(), cpu.as_str(), "{path}"),
                ValueKind::Array => {
                    assert_eq!(gpu.len(), cpu.len(), "{path}: array len");
                    for (i, (g, c)) in gpu.elements().zip(cpu.elements()).enumerate() {
                        assert_values_eq(g, c, &format!("{path}[{i}]"));
                    }
                }
                ValueKind::Object => {
                    assert_eq!(gpu.len(), cpu.len(), "{path}: object len");
                    for (i, ((gk, gv), (ck, cv))) in
                        gpu.entries().zip(cpu.entries()).enumerate()
                    {
                        assert_eq!(gk, ck, "{path}: key #{i}");
                        assert_values_eq(gv, cv, &format!("{path}.{ck}"));
                    }
                }
            }
        }

        /// Both backends on one input: same verdict; on acceptance the
        /// tapes are bit-identical (string offsets included — the pinned
        /// raw-length allocation makes them equal) and the value walks
        /// agree.
        fn diff_doc(gpu: &Parser, cpu: &Parser, input: &[u8], label: &str) {
            match (gpu.parse(input), cpu.parse(input)) {
                (Ok(gpu_doc), Ok(cpu_doc)) => {
                    assert_eq!(
                        gpu_doc.tape(),
                        cpu_doc.tape(),
                        "{label}: tape words must be bit-identical"
                    );
                    assert_values_eq(gpu_doc.root(), cpu_doc.root(), label);
                }
                (Err(_), Err(_)) => {} // verdict parity; WHICH may differ
                (Ok(_), Err(e)) => panic!("{label}: GPU accepted, reference rejected ({e})"),
                (Err(e), Ok(_)) => panic!("{label}: GPU rejected ({e}), reference accepted"),
            }
        }

        /// Every corpus fixture through both backends: Document equality
        /// by full tree walk + bit-identical tapes.
        #[test]
        fn corpus_documents_match_the_reference_backend() {
            let Some(gpu) = gpu_parser_or_skip("corpus_documents_match_the_reference_backend")
            else {
                return;
            };
            let cpu = cpu_parser();
            let corpus = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
            let mut paths: Vec<_> = std::fs::read_dir(&corpus)
                .expect("corpus/ is checked in")
                .map(|e| e.expect("readable corpus entry").path())
                .filter(|p| p.extension().is_some_and(|e| e == "json"))
                .collect();
            paths.sort();
            assert!(!paths.is_empty(), "corpus/ must contain fixtures");
            for path in paths {
                let name = path.file_name().unwrap().to_string_lossy().into_owned();
                let bytes = std::fs::read(&path).expect("readable corpus fixture");
                let doc = gpu
                    .parse(&bytes)
                    .unwrap_or_else(|e| panic!("{name}: corpus fixture must parse on GPU: {e}"));
                let cpu_doc = cpu.parse(&bytes).expect("corpus parses on the reference");
                assert_eq!(doc.tape(), cpu_doc.tape(), "{name}: tape");
                assert_values_eq(doc.root(), cpu_doc.root(), &name);
            }
        }

        /// Fixup-path numbers inside full documents: halfway-point literals
        /// (≥ 20 truncated digits) embedded among other values come out
        /// bit-identical to the reference.
        #[test]
        fn fixup_numbers_inside_full_documents_match_the_reference() {
            let Some(gpu) =
                gpu_parser_or_skip("fixup_numbers_inside_full_documents_match_the_reference")
            else {
                return;
            };
            let cpu = cpu_parser();
            // halfway(1.0, next-up), halfway(0, smallest subnormal) and the
            // 2.2250738585072011e-308 long form: hard-case roundings the
            // kernel must punt to the CPU patch (pinned to take the fixup
            // path in src/gpu/pipeline.rs / src/gpu/numbers.rs).
            let json = format!(
                r#"{{"ties":[{},-{}],"subnormal":{},"mix":[1,"s",2.5e10]}}"#,
                "1.00000000000000011102230246251565404236316680908203125",
                "1.00000000000000011102230246251565404236316680908203125",
                "2.22507385850720113605740979670913197593481954635164564e-308",
            );
            diff_doc(&gpu, &cpu, json.as_bytes(), "fixup document");
            // And the values really are what str::parse says.
            let doc = gpu.parse(json.as_bytes()).unwrap();
            let ties = doc.root().get("ties").unwrap();
            assert_eq!(ties.at(0).unwrap().as_f64(), Some(1.0));
            assert_eq!(
                ties.at(1).unwrap().as_f64().map(f64::to_bits),
                Some((-1.0f64).to_bits())
            );
        }

        /// A multi-error document: both backends reject (WHETHER parity);
        /// WHICH error differs by design — the GPU reports the earliest
        /// byte offset (the string escape), the reference its stage order
        /// (numbers before strings) — pinning the documented relaxation.
        #[test]
        fn multi_error_documents_reject_on_both_backends() {
            let Some(gpu) = gpu_parser_or_skip("multi_error_documents_reject_on_both_backends")
            else {
                return;
            };
            let cpu = cpu_parser();
            let input = br#"{"a":"\q","b":01}"#;
            match gpu.parse(input) {
                Err(Error::Syntax {
                    offset: 6,
                    kind: SyntaxErrorKind::InvalidStringEscape,
                }) => {}
                other => panic!("GPU: earliest offset (escape at 6), got {other:?}"),
            }
            match cpu.parse(input) {
                Err(Error::Syntax {
                    offset: 14,
                    kind: SyntaxErrorKind::InvalidNumber,
                }) => {}
                other => panic!("reference: stage order (number at 14), got {other:?}"),
            }
            // Same-class multi-error: identical verdicts.
            diff_doc(&gpu, &cpu, br#"[01, 0x2]"#, "two number errors");
        }

        /// Single-error inputs: code AND offset parity per class (the
        /// two-way collapse of the M3 split at the public-API level).
        #[test]
        fn single_error_inputs_match_reference_code_and_offset() {
            let Some(gpu) =
                gpu_parser_or_skip("single_error_inputs_match_reference_code_and_offset")
            else {
                return;
            };
            let cpu = cpu_parser();
            let cases: &[&[u8]] = &[
                b"ab\x80",            // Utf8
                b"[1 true]",          // MissingComma
                br#"{"a":1,2}"#,      // MissingColon
                b"]",                 // UnexpectedToken
                b"nul",               // InvalidLiteral
                b"[1",                // UnbalancedBrackets
                b"{},1",              // TrailingContent
                b"[01]",              // InvalidNumber
                br#"{"k":1e999}"#,    // InvalidNumber (overflow)
                br#"["\q"]"#,         // InvalidStringEscape
                b"[\"a\x01\"]",       // ControlCharacterInString
                b"",                  // EmptyInput
            ];
            for &input in cases {
                let gpu_err = gpu.parse(input).expect_err("rejects on GPU");
                let cpu_err = cpu.parse(input).expect_err("rejects on reference");
                assert_eq!(
                    format!("{gpu_err:?}"),
                    format!("{cpu_err:?}"),
                    "{:?}: error parity",
                    String::from_utf8_lossy(input)
                );
            }
            // Depth limit parity, custom limit.
            let gpu_deep = Parser::with_options(ParserOptions {
                max_depth: 2,
                ..ParserOptions::default()
            })
            .expect("device exists");
            let cpu_deep = cpu_parser_with_depth(2);
            let gpu_err = gpu_deep.parse(b"[[[]]]").expect_err("too deep");
            let cpu_err = cpu_deep.parse(b"[[[]]]").expect_err("too deep");
            assert_eq!(format!("{gpu_err:?}"), format!("{cpu_err:?}"));
        }

        /// Duplicate keys ride the tape verbatim on the GPU too.
        #[test]
        fn duplicate_keys_match_the_reference() {
            let Some(gpu) = gpu_parser_or_skip("duplicate_keys_match_the_reference") else {
                return;
            };
            let cpu = cpu_parser();
            diff_doc(&gpu, &cpu, br#"{"k":1,"k":2,"k":3}"#, "duplicate keys");
            let doc = gpu.parse(br#"{"k":1,"k":2,"k":3}"#).unwrap();
            assert_eq!(doc.root().len(), Some(3));
            assert_eq!(doc.root().get("k").unwrap().as_i64(), Some(1));
        }
    }
}
