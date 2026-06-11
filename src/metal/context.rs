//! Device + queue + shader library bundle. Created once, reused for every
//! parse (pipeline-state objects hang off the library, see `pipeline.rs`).

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary};

use crate::error::{Error, Result};

/// Owns the Metal device, the command queue and the compiled shader library.
///
/// This is the root object of the GPU side; everything else
/// ([`Pipeline`](super::Pipeline), [`GpuBuffer`](super::GpuBuffer)) is created
/// from it.
pub struct MetalContext {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
}

// NOTE on threading: `Retained<ProtocolObject<dyn MTL*>>` does not implement
// Send/Sync in objc2-metal even though Metal objects are documented
// thread-safe (the wgpu Metal backend relies on this). When `Parser` needs
// Send + Sync (M1), add thin audited wrapper types here — not before.

impl MetalContext {
    /// Grab the system default device, create a command queue and load the
    /// shader library (embedded metallib, or runtime-compiled MSL when the
    /// `runtime-shaders` feature is enabled).
    pub fn new() -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice().ok_or(Error::NoDevice)?;
        let queue = device.newCommandQueue().ok_or(Error::NoCommandQueue)?;
        let library = load_library(&device)?;
        Ok(Self {
            device,
            queue,
            library,
        })
    }

    pub(crate) fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    pub(crate) fn queue(&self) -> &ProtocolObject<dyn MTLCommandQueue> {
        &self.queue
    }

    pub(crate) fn library(&self) -> &ProtocolObject<dyn MTLLibrary> {
        &self.library
    }
}

impl std::fmt::Debug for MetalContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalContext")
            .field("device", &self.device.name().to_string())
            .finish_non_exhaustive()
    }
}

/// AOT path: the metallib produced by build.rs is embedded in the binary and
/// handed to Metal as dispatch data (`newLibraryWithData:`).
#[cfg(not(feature = "runtime-shaders"))]
fn load_library(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>> {
    static METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/metal_json.metallib"));
    // DispatchData::from_bytes copies; fine for a ~KB..MB library blob.
    let data = dispatch2::DispatchData::from_bytes(METALLIB);
    device
        .newLibraryWithData_error(&data)
        .map_err(|e| Error::LibraryLoad {
            message: e.localizedDescription().to_string(),
        })
}

/// Runtime path: compile MSL source with `newLibraryWithSource:options:error:`.
///
/// That API has no include-path support, so a tiny textual preprocessor
/// inlines `#include "X.h"` directives first. Sources are embedded at build
/// time with `include_str!`; setting `METAL_JSON_SHADER_DIR` to the shaders/
/// directory overrides them at runtime for hot-reload iteration.
#[cfg(feature = "runtime-shaders")]
fn load_library(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>> {
    let source = runtime::assemble_source()?;
    let ns_source = objc2_foundation::NSString::from_str(&source);
    device
        .newLibraryWithSource_options_error(&ns_source, None)
        .map_err(|e| Error::ShaderCompile {
            message: e.localizedDescription().to_string(),
        })
}

#[cfg(feature = "runtime-shaders")]
mod runtime {
    use std::path::PathBuf;

    use crate::error::{Error, Result};

    /// Shader sources embedded at compile time. Keep in sync with the files
    /// under shaders/ — build.rs emits rerun-if-changed for all of them.
    const EMBEDDED: &[(&str, &str)] = &[
        ("common.h", include_str!("../../shaders/common.h")),
        ("tape_types.h", include_str!("../../shaders/tape_types.h")),
        (
            "00_smoke.metal",
            include_str!("../../shaders/00_smoke.metal"),
        ),
    ];

    fn shader_dir_override() -> Option<PathBuf> {
        std::env::var_os("METAL_JSON_SHADER_DIR").map(PathBuf::from)
    }

    fn read_source(name: &str) -> Result<String> {
        if let Some(dir) = shader_dir_override() {
            return std::fs::read_to_string(dir.join(name)).map_err(|e| Error::ShaderCompile {
                message: format!("METAL_JSON_SHADER_DIR: cannot read `{name}`: {e}"),
            });
        }
        EMBEDDED
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, src)| (*src).to_owned())
            .ok_or_else(|| Error::ShaderCompile {
                message: format!("no embedded shader source named `{name}`"),
            })
    }

    /// Names of the `.metal` translation units to compile (sorted, so kernel
    /// numbering is deterministic).
    fn metal_unit_names() -> Result<Vec<String>> {
        let mut names: Vec<String> = if let Some(dir) = shader_dir_override() {
            let entries = std::fs::read_dir(&dir).map_err(|e| Error::ShaderCompile {
                message: format!(
                    "METAL_JSON_SHADER_DIR: cannot read dir {}: {e}",
                    dir.display()
                ),
            })?;
            entries
                .filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| n.ends_with(".metal"))
                .collect()
        } else {
            EMBEDDED
                .iter()
                .map(|(n, _)| (*n).to_owned())
                .filter(|n| n.ends_with(".metal"))
                .collect()
        };
        names.sort();
        Ok(names)
    }

    /// Concatenate every `.metal` unit with `#include "X"` textually inlined.
    /// Duplicate header inclusion across units is harmless: the inlined
    /// headers keep their `#ifndef` guards and the result is compiled as a
    /// single translation unit.
    pub(super) fn assemble_source() -> Result<String> {
        let mut out = String::new();
        for name in metal_unit_names()? {
            let mut stack = vec![name.clone()];
            inline_includes(&name, &mut out, &mut stack)?;
            out.push('\n');
        }
        Ok(out)
    }

    fn inline_includes(name: &str, out: &mut String, stack: &mut Vec<String>) -> Result<()> {
        let src = read_source(name)?;
        for line in src.lines() {
            if let Some(included) = parse_quoted_include(line) {
                if stack.iter().any(|s| s == included) {
                    // Include cycle; the #ifndef guard would terminate it at
                    // MSL-compile time anyway, but cut it off here for speed.
                    continue;
                }
                stack.push(included.to_owned());
                inline_includes(included, out, stack)?;
                stack.pop();
            } else {
                out.push_str(line);
                out.push('\n');
            }
        }
        Ok(())
    }

    /// Match `#include "file"` (leaves `#include <metal_stdlib>` alone).
    fn parse_quoted_include(line: &str) -> Option<&str> {
        let rest = line.trim_start().strip_prefix('#')?.trim_start();
        let rest = rest.strip_prefix("include")?.trim_start();
        let rest = rest.strip_prefix('"')?;
        let end = rest.find('"')?;
        Some(&rest[..end])
    }
}
