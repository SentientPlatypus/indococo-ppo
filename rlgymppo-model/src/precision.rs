//! Training-time numeric precision for network forward passes.
//!
//! [`ComputePrecision::Bf16`] mirrors PyTorch autocast for the layers this
//! crate builds: activations and weights are cast to `bf16` for each matrix
//! multiplication, while the parameters (the optimizer's master weights),
//! normalization layers and the network output stay in `f32`. Losses,
//! log-softmax and PPO ratios are therefore always computed in `f32`.

/// Precision used for one network's matrix multiplications.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ComputePrecision {
    /// Full `f32` (or TF32 when the backend enables it).
    #[default]
    F32,
    /// `bf16` matrix multiplications with `f32` parameters and output.
    Bf16,
}

/// Per-network precision for the PPO training step.
///
/// Rollout inference, value estimation and exported weights are unaffected:
/// they always run in `f32`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TrainPrecision {
    pub policy: ComputePrecision,
    pub critic: ComputePrecision,
}
