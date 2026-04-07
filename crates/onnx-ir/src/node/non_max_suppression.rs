//! # NonMaxSuppression
//!
//! Filters bounding boxes by score and overlap (IoU), keeping the highest-scoring
//! non-overlapping boxes per class.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__NonMaxSuppression.html>
//!
//! ## Opset Versions
//! - **Opset 10**: Initial version
//! - **Opset 11**: No changes to this operator

use derive_new::new;

use burn_tensor::DType;
use onnx_ir_derive::NodeBuilder;

use crate::{
    ArgType, Argument, Node, RawNode, TensorDataExt, TensorType,
    processor::{
        ArgPreference, InputPreferences, InputSpec, NodeProcessor, NodeSpec, OutputPreferences,
        OutputSpec, ProcessError,
    },
};

/// How bounding box coordinates are encoded.
#[derive(Debug, Clone, Default)]
pub enum BoxFormat {
    /// Corner format: `[y1, x1, y2, x2]` where `(y1, x1)` and `(y2, x2)` are
    /// the coordinates of any diagonal pair of box corners.
    #[default]
    Corner,
    /// Center format: `[x_center, y_center, width, height]`.
    Center,
}

/// Configuration for the ONNX NonMaxSuppression operator.
#[derive(Debug, Clone, Default, new)]
pub struct NonMaxSuppressionConfig {
    /// How bounding box coordinates are encoded in the input tensor.
    pub center_point_box: BoxFormat,
}

impl TryFrom<i64> for NonMaxSuppressionConfig {
    type Error = ProcessError;

    fn try_from(value: i64) -> Result<Self, Self::Error> {
        let box_format = match value {
            0 => BoxFormat::Corner,
            1 => BoxFormat::Center,
            _ => {
                return Err(ProcessError::InvalidAttribute {
                    name: "center_point_box".to_string(),
                    reason: format!("expected 0 or 1, got {value}"),
                });
            }
        };

        Ok(Self {
            center_point_box: box_format,
        })
    }
}

/// ONNX NonMaxSuppression node — filters bounding boxes by score and IoU overlap.
///
/// Outputs a `[num_selected, 3]` tensor of `[batch_index, class_index, box_index]` triples.
#[derive(Debug, Clone, NodeBuilder)]
pub struct NonMaxSuppressionNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: NonMaxSuppressionConfig,
}

pub(crate) struct NonMaxSuppressionProcessor;

impl NodeProcessor for NonMaxSuppressionProcessor {
    type Config = NonMaxSuppressionConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 10,
            max_opset: None,
            inputs: InputSpec::Range(2, 5),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn input_preferences(
        &self,
        node: &RawNode,
        _opset: usize,
    ) -> Result<Option<InputPreferences>, ProcessError> {
        let mut prefs = InputPreferences::new();

        for index in 2..=4 {
            if let Some(input) = node.get_input(index) {
                prefs = prefs.add(input.name.clone(), ArgPreference::ScalarNative);
            }
        }

        Ok(Some(prefs))
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        self.extract_config(node, opset)?;
        validate_non_max_suppression_inputs(node)?;

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: DType::I64,
            rank: 2,
            static_shape: Some(vec![None, Some(3)]),
        });

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError>
    where
        Self: Sized,
    {
        node.attrs
            .get("center_point_box")
            .map(|value| value.clone().into_i64())
            .unwrap_or(0)
            .try_into()
    }

    fn build_node(&self, builder: RawNode, _opset: usize) -> Node
    where
        Self: Sized,
    {
        // Construct config infallibly — infer_types() already validated center_point_box.
        let center_point_box = builder
            .attrs
            .get("center_point_box")
            .map(|value| value.clone().into_i64())
            .unwrap_or(0);

        let box_format = match center_point_box {
            1 => BoxFormat::Center,
            _ => BoxFormat::Corner,
        };

        Node::NonMaxSuppression(NonMaxSuppressionNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config: NonMaxSuppressionConfig {
                center_point_box: box_format,
            },
        })
    }
}

fn validate_non_max_suppression_inputs(node: &RawNode) -> Result<(), ProcessError> {
    let boxes = node
        .get_input(0)
        .ok_or_else(|| ProcessError::MissingInput("boxes".to_string()))?;
    let scores = node
        .get_input(1)
        .ok_or_else(|| ProcessError::MissingInput("scores".to_string()))?;

    let boxes_tensor = expect_float_tensor(boxes, "boxes", 3)?;
    let scores_tensor = expect_float_tensor(scores, "scores", 3)?;

    validate_boxes_shape(boxes_tensor)?;
    validate_scores_shape(scores_tensor)?;
    validate_batch_and_box_counts(boxes_tensor, scores_tensor)?;

    for (index, input_name) in [
        (2usize, "max_output_boxes_per_class"),
        (3usize, "iou_threshold"),
        (4usize, "score_threshold"),
    ] {
        if let Some(input) = node.get_input(index) {
            validate_optional_input(input, input_name, index)?;
        }
    }

    Ok(())
}

fn expect_float_tensor<'a>(
    arg: &'a Argument,
    name: &str,
    expected_rank: usize,
) -> Result<&'a TensorType, ProcessError> {
    let tensor = match &arg.ty {
        ArgType::Tensor(tensor) => tensor,
        other => {
            return Err(ProcessError::TypeMismatch {
                expected: format!("{name} to be a rank-{expected_rank} float tensor"),
                actual: other.to_string(),
            });
        }
    };

    if tensor.rank != expected_rank {
        return Err(ProcessError::Custom(format!(
            "NonMaxSuppression input '{name}' must be rank {expected_rank}, got rank {}",
            tensor.rank
        )));
    }

    if !tensor.dtype.is_float() {
        return Err(ProcessError::TypeMismatch {
            expected: format!("{name} to have a floating-point dtype"),
            actual: format!("{:?}", tensor.dtype),
        });
    }

    Ok(tensor)
}

fn validate_boxes_shape(boxes: &TensorType) -> Result<(), ProcessError> {
    if let Some(shape) = boxes.static_shape.as_ref()
        && let Some(Some(last_dim)) = shape.get(2)
        && *last_dim != 4
    {
        return Err(ProcessError::Custom(format!(
            "NonMaxSuppression input 'boxes' must have shape [batch, num_boxes, 4], got last dimension {last_dim}"
        )));
    }

    Ok(())
}

fn validate_scores_shape(scores: &TensorType) -> Result<(), ProcessError> {
    if let Some(shape) = scores.static_shape.as_ref()
        && shape.len() != 3
    {
        return Err(ProcessError::Custom(format!(
            "NonMaxSuppression input 'scores' must have shape [batch, num_classes, num_boxes], got rank {}",
            shape.len()
        )));
    }

    Ok(())
}

fn validate_batch_and_box_counts(
    boxes: &TensorType,
    scores: &TensorType,
) -> Result<(), ProcessError> {
    let Some(boxes_shape) = boxes.static_shape.as_ref() else {
        return Ok(());
    };
    let Some(scores_shape) = scores.static_shape.as_ref() else {
        return Ok(());
    };

    if let (Some(Some(box_batches)), Some(Some(score_batches))) =
        (boxes_shape.first(), scores_shape.first())
        && box_batches != score_batches
    {
        return Err(ProcessError::Custom(format!(
            "NonMaxSuppression inputs 'boxes' and 'scores' must agree on batch dimension, got {box_batches} and {score_batches}"
        )));
    }

    if let (Some(Some(num_boxes)), Some(Some(score_boxes))) =
        (boxes_shape.get(1), scores_shape.get(2))
        && num_boxes != score_boxes
    {
        return Err(ProcessError::Custom(format!(
            "NonMaxSuppression inputs 'boxes' and 'scores' must agree on num_boxes, got {num_boxes} and {score_boxes}"
        )));
    }

    Ok(())
}

fn validate_optional_input(arg: &Argument, name: &str, index: usize) -> Result<(), ProcessError> {
    let dtype_description = match index {
        2 => "int64",
        3 | 4 => "floating-point",
        _ => {
            return Err(ProcessError::Custom(format!(
                "NonMaxSuppression: unexpected optional input index {index}"
            )));
        }
    };

    let dtype_valid = |dtype: DType| match index {
        2 => dtype == DType::I64,
        3 | 4 => dtype.is_float(),
        _ => false,
    };

    match &arg.ty {
        ArgType::ScalarNative(dtype) | ArgType::ScalarTensor(dtype) => {
            if !dtype_valid(*dtype) {
                return Err(ProcessError::TypeMismatch {
                    expected: format!("{name} to have {dtype_description} dtype"),
                    actual: format!("{dtype:?}"),
                });
            }
        }
        ArgType::Tensor(tensor) if tensor.rank == 1 => {
            if !dtype_valid(tensor.dtype) {
                return Err(ProcessError::TypeMismatch {
                    expected: format!("{name} to have {dtype_description} dtype"),
                    actual: format!("{:?}", tensor.dtype),
                });
            }

            if let Some(shape) = tensor.static_shape.as_ref()
                && let Some(Some(length)) = shape.first()
                && *length != 1
            {
                return Err(ProcessError::Custom(format!(
                    "NonMaxSuppression optional input '{name}' must be scalar-like; rank-1 tensors must have length 1, got length {length}"
                )));
            }
        }
        other => {
            return Err(ProcessError::TypeMismatch {
                expected: format!("{name} to be ScalarNative, ScalarTensor, or rank-1 tensor"),
                actual: other.to_string(),
            });
        }
    }

    // Per spec, iou_threshold must be in [0, 1]. Validate when statically known.
    if index == 3
        && let Some(data) = arg.value()
        && let Ok(v) = data.scalar_f32()
        && !(0.0..=1.0).contains(&v)
    {
        return Err(ProcessError::Custom(format!(
            "NonMaxSuppression iou_threshold must be in [0, 1], got {v}"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::NodeType,
        node::test_utils::TestNodeBuilder,
        processor::{ArgPreference, OutputPreferences},
    };

    fn base_builder() -> TestNodeBuilder {
        TestNodeBuilder::new(NodeType::NonMaxSuppression, "nms")
            .input_tensor_f32("boxes", 3, Some(vec![1, 6, 4]))
            .input_tensor_f32("scores", 3, Some(vec![1, 1, 6]))
            .output_default("selected_indices")
    }

    #[test]
    fn input_preferences_skip_missing_optional_placeholder() {
        let node = base_builder()
            .input_scalar_i64("max_output_boxes_per_class")
            .add_input("", ArgType::ScalarNative(DType::F32))
            .input_scalar_f32("score_threshold")
            .build();

        let prefs = NonMaxSuppressionProcessor
            .input_preferences(&node, 11)
            .unwrap()
            .unwrap();

        assert!(matches!(
            prefs.get("max_output_boxes_per_class"),
            [ArgPreference::ScalarNative]
        ));
        assert!(prefs.get("").is_empty());
        assert!(matches!(
            prefs.get("score_threshold"),
            [ArgPreference::ScalarNative]
        ));
    }

    #[test]
    fn infer_types_sets_rank_2_i64_output() {
        let mut node = base_builder()
            .input_scalar_i64("max_output_boxes_per_class")
            .input_scalar_f32("iou_threshold")
            .input_scalar_f32("score_threshold")
            .build();

        NonMaxSuppressionProcessor
            .infer_types(&mut node, 11, &OutputPreferences::new())
            .unwrap();

        assert_eq!(
            node.outputs[0].ty,
            ArgType::Tensor(TensorType {
                dtype: DType::I64,
                rank: 2,
                static_shape: Some(vec![None, Some(3)]),
            })
        );
    }

    #[test]
    fn infer_types_accepts_missing_middle_optional_input() {
        let mut node = base_builder()
            .input_scalar_i64("max_output_boxes_per_class")
            .add_input("", ArgType::ScalarNative(DType::F32))
            .input_scalar_f32("score_threshold")
            .build();

        NonMaxSuppressionProcessor
            .infer_types(&mut node, 11, &OutputPreferences::new())
            .unwrap();
    }

    #[test]
    fn infer_types_rejects_invalid_center_point_box() {
        let mut node = base_builder().attr_int("center_point_box", 2).build();

        let err = NonMaxSuppressionProcessor
            .infer_types(&mut node, 11, &OutputPreferences::new())
            .unwrap_err();

        assert!(matches!(err, ProcessError::InvalidAttribute { .. }));
    }

    #[test]
    fn infer_types_rejects_invalid_boxes_rank() {
        let mut node = TestNodeBuilder::new(NodeType::NonMaxSuppression, "nms")
            .input_tensor_f32("boxes", 2, Some(vec![6, 4]))
            .input_tensor_f32("scores", 3, Some(vec![1, 1, 6]))
            .output_default("selected_indices")
            .build();

        let err = NonMaxSuppressionProcessor
            .infer_types(&mut node, 11, &OutputPreferences::new())
            .unwrap_err();

        assert!(
            err.to_string().contains("boxes"),
            "expected boxes validation error, got {err}"
        );
    }

    #[test]
    fn infer_types_rejects_non_scalar_like_optional_input() {
        let mut node = base_builder()
            .input_tensor_i64("max_output_boxes_per_class", 1, Some(vec![2]))
            .build();

        let err = NonMaxSuppressionProcessor
            .infer_types(&mut node, 11, &OutputPreferences::new())
            .unwrap_err();

        assert!(
            err.to_string().contains("length 1"),
            "expected scalar-like optional input error, got {err}"
        );
    }

    #[test]
    fn infer_types_rejects_out_of_range_iou_threshold() {
        let mut node = base_builder()
            .input_scalar_i64("max_output_boxes_per_class")
            .input_tensor_f32_data("iou_threshold", vec![1.5], vec![1])
            .input_scalar_f32("score_threshold")
            .build_with_graph_data(11);

        let err = NonMaxSuppressionProcessor
            .infer_types(&mut node, 11, &OutputPreferences::new())
            .unwrap_err();

        assert!(
            err.to_string().contains("[0, 1]"),
            "expected iou_threshold range error, got {err}"
        );
    }

    #[test]
    fn build_node_keeps_center_point_box_config() {
        let mut node = base_builder().attr_int("center_point_box", 1).build();
        NonMaxSuppressionProcessor
            .infer_types(&mut node, 11, &OutputPreferences::new())
            .unwrap();

        let built = NonMaxSuppressionProcessor.build_node(node, 11);
        let Node::NonMaxSuppression(node) = built else {
            panic!("expected NonMaxSuppression node");
        };

        assert!(matches!(node.config.center_point_box, BoxFormat::Center));
    }
}
