//! # HammingWindow
//!
//! Generates a Hamming window as described in the paper
//! <https://ieeexplore.ieee.org/document/1455106>.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__HammingWindow.html>
//!
//! ## Opset Versions
//! - **Opset 17**: Initial version with size input, periodic and output_datatype attributes.

use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, DType, Node, RawNode, TensorDataExt, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError, validate_opset,
};
use crate::proto_conversion::element_type_from_proto;

/// Configuration for the HammingWindow operation.
#[derive(Debug, Clone)]
pub struct HammingWindowConfig {
    /// If true, returns a periodic window. If false, returns a symmetric window.
    pub periodic: bool,
    /// The output data type.
    pub output_dtype: DType,
    /// The window size (must be a compile-time constant).
    pub size: usize,
}

impl Default for HammingWindowConfig {
    fn default() -> Self {
        Self {
            periodic: true,
            output_dtype: DType::F32,
            size: 0,
        }
    }
}

/// Node representation for HammingWindow operation.
#[derive(Debug, Clone, NodeBuilder)]
pub struct HammingWindowNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: HammingWindowConfig,
}

pub(crate) struct HammingWindowProcessor;

/// Resolve the output_datatype attribute (default: FLOAT).
fn resolve_output_dtype(node: &RawNode) -> Result<DType, ProcessError> {
    let dtype = match node.attrs.get("output_datatype") {
        Some(val) => {
            let dt_i32 = val.clone().into_i32();
            element_type_from_proto(dt_i32).map_err(|e| ProcessError::InvalidAttribute {
                name: "output_datatype".to_string(),
                reason: e,
            })?
        }
        None => DType::F32,
    };

    // ONNX spec constrains T2 to float types
    if !matches!(dtype, DType::F16 | DType::BF16 | DType::F32 | DType::F64) {
        return Err(ProcessError::InvalidAttribute {
            name: "output_datatype".to_string(),
            reason: format!("must be a float type, got {dtype:?}"),
        });
    }

    Ok(dtype)
}

impl NodeProcessor for HammingWindowProcessor {
    type Config = HammingWindowConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 17,
            max_opset: None,
            inputs: InputSpec::Exact(1),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn lift_constants(&self, node: &mut RawNode, _opset: usize) -> Result<(), ProcessError> {
        if !node.inputs.is_empty() && node.inputs[0].is_constant() {
            node.inputs[0].to_static()?;
        }
        Ok(())
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        validate_opset(opset, 17)?;

        // Validate input is an integer scalar
        let input = &node.inputs[0];
        let is_valid_scalar = input.ty.is_scalar()
            || matches!(&input.ty, ArgType::Tensor(t) if t.rank == 0)
            || matches!(&input.ty, ArgType::Tensor(t) if t.rank == 1
                && t.static_shape.as_ref().is_some_and(|s| s == &[Some(1)]));
        if !is_valid_scalar {
            return Err(ProcessError::TypeMismatch {
                expected: "scalar input (int32 or int64)".to_string(),
                actual: format!("{:?}", input.ty),
            });
        }

        let input_dtype = input.ty.elem_type();
        if !matches!(input_dtype, DType::I32 | DType::I64) {
            return Err(ProcessError::TypeMismatch {
                expected: "int32 or int64".to_string(),
                actual: format!("{:?}", input_dtype),
            });
        }

        let output_dtype = resolve_output_dtype(node)?;

        // Validate size is a compile-time constant and extract static shape
        let static_shape = match node.inputs[0].value() {
            Some(data) => {
                let val = data.scalar_i64().map_err(|e| ProcessError::TypeMismatch {
                    expected: "scalar integer for size".to_string(),
                    actual: format!("{e}"),
                })?;
                if val < 0 {
                    return Err(ProcessError::Custom(
                        "HammingWindow: size must be non-negative".to_string(),
                    ));
                }
                Some(vec![Some(val as usize)])
            }
            None => {
                return Err(ProcessError::Custom(
                    "HammingWindow: size must be a compile-time constant".to_string(),
                ));
            }
        };

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: output_dtype,
            rank: 1,
            static_shape,
        });

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let periodic = node
            .attrs
            .get("periodic")
            .map(|v| v.clone().into_i64() != 0)
            .unwrap_or(true);

        let output_dtype = resolve_output_dtype(node)?;

        // Extract size from the constant input
        let size = match node.inputs[0].value() {
            Some(data) => {
                let val = data.scalar_i64().map_err(|e| ProcessError::TypeMismatch {
                    expected: "scalar integer for size".to_string(),
                    actual: format!("{e}"),
                })?;
                if val < 0 {
                    return Err(ProcessError::Custom(
                        "HammingWindow: size must be non-negative".to_string(),
                    ));
                }
                val as usize
            }
            None => {
                return Err(ProcessError::Custom(
                    "HammingWindow: size must be a compile-time constant".to_string(),
                ));
            }
        };

        Ok(HammingWindowConfig {
            periodic,
            output_dtype,
            size,
        })
    }

    fn build_node(&self, mut builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        // Drop the size input: it's baked into config and codegen has no runtime inputs.
        builder.inputs.clear();

        Node::HammingWindow(HammingWindowNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::TestNodeBuilder;
    use crate::processor::OutputPreferences;

    #[test]
    fn test_hamming_window_default() {
        let mut node = TestNodeBuilder::new(NodeType::HammingWindow, "test_hamming")
            .input_scalar_tensor_i64("size", Some(10))
            .output_tensor_f32("output", 0, None)
            .build_with_graph_data(17);

        let processor = HammingWindowProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 17, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.dtype, DType::F32);
                assert_eq!(t.rank, 1);
                assert_eq!(t.static_shape, Some(vec![Some(10)]));
            }
            _ => panic!("Expected Tensor output"),
        }
    }

    #[test]
    fn test_hamming_window_double_output() {
        let mut node = TestNodeBuilder::new(NodeType::HammingWindow, "test_hamming")
            .input_scalar_tensor_i64("size", Some(8))
            .output_tensor_f32("output", 0, None)
            .attr_int("output_datatype", 11) // DOUBLE
            .build_with_graph_data(17);

        let processor = HammingWindowProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 17, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.dtype, DType::F64);
                assert_eq!(t.rank, 1);
                assert_eq!(t.static_shape, Some(vec![Some(8)]));
            }
            _ => panic!("Expected Tensor output"),
        }
    }

    #[test]
    fn test_hamming_window_symmetric() {
        let node = TestNodeBuilder::new(NodeType::HammingWindow, "test_hamming")
            .input_scalar_tensor_i64("size", Some(10))
            .output_tensor_f32("output", 0, None)
            .attr_int("periodic", 0)
            .build_with_graph_data(17);

        let processor = HammingWindowProcessor;
        let config = processor.extract_config(&node, 17).unwrap();
        assert!(!config.periodic);
        assert_eq!(config.size, 10);
        assert_eq!(config.output_dtype, DType::F32);
    }

    #[test]
    fn test_hamming_window_periodic() {
        let node = TestNodeBuilder::new(NodeType::HammingWindow, "test_hamming")
            .input_scalar_tensor_i64("size", Some(8))
            .output_tensor_f32("output", 0, None)
            .attr_int("periodic", 1)
            .build_with_graph_data(17);

        let processor = HammingWindowProcessor;
        let config = processor.extract_config(&node, 17).unwrap();
        assert!(config.periodic);
        assert_eq!(config.size, 8);
    }

    #[test]
    fn test_hamming_window_non_constant_size() {
        let mut node = TestNodeBuilder::new(NodeType::HammingWindow, "test_hamming")
            .input_scalar_i64("size")
            .output_tensor_f32("output", 0, None)
            .build();

        let processor = HammingWindowProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 17, &prefs);
        assert!(
            matches!(result, Err(ProcessError::Custom(ref msg)) if msg.contains("compile-time constant")),
            "Expected compile-time constant error, got: {result:?}"
        );
    }

    #[test]
    fn test_hamming_window_negative_size() {
        let mut node = TestNodeBuilder::new(NodeType::HammingWindow, "test_hamming")
            .input_scalar_tensor_i64("size", Some(-5))
            .output_tensor_f32("output", 0, None)
            .build_with_graph_data(17);

        let processor = HammingWindowProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 17, &prefs);
        assert!(
            matches!(result, Err(ProcessError::Custom(ref msg)) if msg.contains("non-negative")),
            "Expected non-negative error, got: {result:?}"
        );
    }

    #[test]
    fn test_hamming_window_integer_output_dtype_rejected() {
        let mut node = TestNodeBuilder::new(NodeType::HammingWindow, "test_hamming")
            .input_scalar_tensor_i64("size", Some(10))
            .output_tensor_f32("output", 0, None)
            .attr_int("output_datatype", 7) // INT64
            .build_with_graph_data(17);

        let processor = HammingWindowProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 17, &prefs);
        assert!(
            matches!(result, Err(ProcessError::InvalidAttribute { .. })),
            "Expected invalid attribute error for integer output dtype, got: {result:?}"
        );
    }

    #[test]
    fn test_hamming_window_opset_too_low() {
        let mut node = TestNodeBuilder::new(NodeType::HammingWindow, "test_hamming")
            .input_scalar_tensor_i64("size", Some(10))
            .output_tensor_f32("output", 0, None)
            .build_with_graph_data(16);

        let processor = HammingWindowProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(result.is_err());
    }
}
