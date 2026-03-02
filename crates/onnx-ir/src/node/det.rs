//! # Det Operation
//!
//! Computes the determinant of a square matrix or batches of square matrices.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Det.html>
//!
//! ## Type Constraints
//!
//! T: Floating-point tensor types (float16, float, double, bfloat16)
//!
//! ## Opset Versions
//! - **Opset 11+**: Initial version
//! - **Opset 22+**: Added bfloat16 support
//!
//! ## Implementation Notes
//!
//! The determinant is computed using PLU decomposition:
//! `det(A) = sign(P) * det(L) * det(U) = sign(P) * product_of_diagonal(U)`
//! since `det(L) = 1` (L has ones on its diagonal).
//!
//! The sign of the permutation is computed by counting inversions in the permutation vector.
//! An inversion is a pair `(i, j)` where `i < j` but `p[i] > p[j]`.
//!
//! Both 2D (non-batched) and batched inputs (`[*, M, M]`) are supported. For batched inputs
//! the determinant is computed per-matrix using a loop over batch dimensions.
//!
//! **Limitation**: Singular matrices (det = 0) cause a panic from `lu_decomposition`
//! since partial pivoting encounters a zero pivot. A native `det()` function in Burn
//! would handle this case correctly (tracked as a separate issue in the Burn repo).

use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, Node, RawNode, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError, validate_opset,
};

/// Node representation for Det operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct DetNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
}

/// Node processor for Det operation
pub(crate) struct DetProcessor;

impl NodeProcessor for DetProcessor {
    type Config = ();

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 11,
            max_opset: None,
            inputs: InputSpec::Exact(1),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        validate_opset(opset, 11)?;

        let (rank, dtype) = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => {
                if tensor.rank < 2 {
                    return Err(ProcessError::Custom(format!(
                        "Det: input must have rank >= 2 (got rank {})",
                        tensor.rank
                    )));
                }
                (tensor.rank, tensor.dtype)
            }
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };

        // Output rank is input rank minus 2 (the last two dims are the matrix dims)
        if rank == 2 {
            // 2D input → scalar output
            node.outputs[0].ty = ArgType::ScalarTensor(dtype);
        } else {
            node.outputs[0].ty = ArgType::Tensor(TensorType {
                dtype,
                rank: rank - 2,
                static_shape: None,
            });
        }

        Ok(())
    }

    fn build_node(&self, builder: RawNode, _opset: usize) -> Node {
        Node::Det(DetNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, DType, NodeType};
    use crate::node::test_utils::TestNodeBuilder;
    use crate::processor::OutputPreferences;

    #[test]
    fn test_det_2d_infer_types() {
        let mut node = TestNodeBuilder::new(NodeType::Det, "test_det")
            .input_tensor_f32("X", 2, None)
            .output_tensor_f32("Y", 0, None)
            .build();

        let processor = DetProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 11, &prefs).unwrap();

        assert!(matches!(node.outputs[0].ty, ArgType::ScalarTensor(DType::F32)));
    }

    #[test]
    fn test_det_3d_infer_types() {
        let mut node = TestNodeBuilder::new(NodeType::Det, "test_det")
            .input_tensor_f32("X", 3, None)
            .output_tensor_f32("Y", 0, None)
            .build();

        let processor = DetProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 11, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 1);
                assert_eq!(t.dtype, DType::F32);
            }
            _ => panic!("Expected Tensor output for 3D input"),
        }
    }

    #[test]
    fn test_det_rank_too_low() {
        let mut node = TestNodeBuilder::new(NodeType::Det, "test_det")
            .input_tensor_f32("X", 1, None)
            .output_tensor_f32("Y", 0, None)
            .build();

        let processor = DetProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 11, &prefs);
        assert!(result.is_err());
    }
}
