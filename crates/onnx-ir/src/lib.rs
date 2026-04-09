#[macro_use]
extern crate derive_new;

mod external_data;
mod graph_state;
pub mod ir;
pub mod node;
mod phases;
mod pipeline;
mod processor;
mod proto_conversion;
mod protos;
mod registry;
mod simplify;
mod tensor_store;

// Public API - only expose essentials
pub use ir::*;
pub use node::*;
pub use pipeline::{Error, OnnxGraphBuilder};

/// Generated protobuf binding for the ONNX `TensorProto` message.
///
/// Re-exported so sibling crates (e.g. `onnx-official-tests`) can decode
/// `.pb` reference tensors without rebuilding the bindings themselves.
/// The rest of the generated `protos` module is intentionally kept
/// private; if more types need exposing, prefer adding narrow re-exports
/// here over making the whole module public.
pub use protos::TensorProto;
