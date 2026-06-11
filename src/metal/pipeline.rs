//! Compute pipeline-state wrapper. One `Pipeline` per kernel function,
//! created once at `Parser::new` time and reused across parses.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLLibrary};

use super::MetalContext;
use crate::error::{Error, Result};

/// A compiled compute pipeline for a single kernel function.
pub struct Pipeline {
    state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    name: String,
}

impl Pipeline {
    /// Look up `kernel_name` in the context's shader library and build a
    /// `MTLComputePipelineState` for it.
    pub fn new(ctx: &MetalContext, kernel_name: &str) -> Result<Self> {
        let function = ctx
            .library()
            .newFunctionWithName(&NSString::from_str(kernel_name))
            .ok_or_else(|| Error::KernelNotFound {
                name: kernel_name.to_owned(),
            })?;
        let state = ctx
            .device()
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| Error::PipelineCreate {
                name: kernel_name.to_owned(),
                message: e.localizedDescription().to_string(),
            })?;
        Ok(Self {
            state,
            name: kernel_name.to_owned(),
        })
    }

    // TODO(M2+): `Pipeline::with_constants(ctx, name, &MTLFunctionConstantValues)`
    // for function-constant specialization (newFunctionWithName:constantValues:error:).

    /// Kernel function name this pipeline was built from.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Hardware limit for threads per threadgroup with this pipeline.
    pub fn max_total_threads_per_threadgroup(&self) -> usize {
        self.state.maxTotalThreadsPerThreadgroup()
    }

    pub(crate) fn state(&self) -> &ProtocolObject<dyn MTLComputePipelineState> {
        &self.state
    }
}

impl std::fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipeline")
            .field("name", &self.name)
            .finish()
    }
}
