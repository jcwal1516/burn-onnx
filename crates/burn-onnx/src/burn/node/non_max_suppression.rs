use onnx_ir::non_max_suppression::{BoxFormat, NonMaxSuppressionNode};

use super::prelude::*;

fn compile_error_tokens(message: impl Into<String>) -> TokenStream {
    let message = message.into();
    quote! {
        compile_error!(#message);
    }
}

fn optional_scalar_extract(
    arg: Option<&Argument>,
    scope: &mut ScopeAtPosition<'_>,
) -> Result<Option<TokenStream>, String> {
    let Some(arg) = arg.filter(|arg| !arg.is_optional()) else {
        return Ok(None);
    };

    let tokens = scope.arg(arg);
    Ok(Some(match &arg.ty {
        ArgType::ScalarNative(_) => tokens,
        ArgType::ScalarTensor(dtype) => on_device_to_native(tokens, dtype),
        ArgType::Tensor(tensor) if tensor.rank == 1 => on_device_to_native(tokens, &tensor.dtype),
        other => {
            return Err(format!(
                "NonMaxSuppression optional input type should be validated in onnx-ir, got {other:?}"
            ));
        }
    }))
}

/// Remap ONNX box coords to `[x1, y1, x2, y2]` corner boxes as a `[num_boxes, 4]` tensor.
/// Corner format stores `[y1, x1, y2, x2]`; center format stores `[cx, cy, w, h]`.
fn corner_boxes_expr(format: &BoxFormat) -> TokenStream {
    match format {
        BoxFormat::Corner => quote! {{
            let __y1_raw: Tensor<B, 1> = __boxes_batch_raw
                .clone()
                .slice([0..__num_boxes, 0..1])
                .reshape([__num_boxes]);
            let __x1_raw: Tensor<B, 1> = __boxes_batch_raw
                .clone()
                .slice([0..__num_boxes, 1..2])
                .reshape([__num_boxes]);
            let __y2_raw: Tensor<B, 1> = __boxes_batch_raw
                .clone()
                .slice([0..__num_boxes, 2..3])
                .reshape([__num_boxes]);
            let __x2_raw: Tensor<B, 1> = __boxes_batch_raw
                .clone()
                .slice([0..__num_boxes, 3..4])
                .reshape([__num_boxes]);
            let __x1 = __x1_raw.clone().min_pair(__x2_raw.clone());
            let __y1 = __y1_raw.clone().min_pair(__y2_raw.clone());
            let __x2 = __x1_raw.max_pair(__x2_raw);
            let __y2 = __y1_raw.max_pair(__y2_raw);
            Tensor::cat(
                alloc::vec![
                    __x1.unsqueeze_dim(1),
                    __y1.unsqueeze_dim(1),
                    __x2.unsqueeze_dim(1),
                    __y2.unsqueeze_dim(1),
                ],
                1,
            )
        }},
        BoxFormat::Center => quote! {{
            let __cx: Tensor<B, 1> = __boxes_batch_raw
                .clone()
                .slice([0..__num_boxes, 0..1])
                .reshape([__num_boxes]);
            let __cy: Tensor<B, 1> = __boxes_batch_raw
                .clone()
                .slice([0..__num_boxes, 1..2])
                .reshape([__num_boxes]);
            let __w: Tensor<B, 1> = __boxes_batch_raw
                .clone()
                .slice([0..__num_boxes, 2..3])
                .reshape([__num_boxes]);
            let __h: Tensor<B, 1> = __boxes_batch_raw
                .clone()
                .slice([0..__num_boxes, 3..4])
                .reshape([__num_boxes]);
            let __half_w = __w / 2.0f32;
            let __half_h = __h / 2.0f32;
            let __x1 = __cx.clone() - __half_w.clone();
            let __y1 = __cy.clone() - __half_h.clone();
            let __x2 = __cx + __half_w;
            let __y2 = __cy + __half_h;
            Tensor::cat(
                alloc::vec![
                    __x1.unsqueeze_dim(1),
                    __y1.unsqueeze_dim(1),
                    __x2.unsqueeze_dim(1),
                    __y2.unsqueeze_dim(1),
                ],
                1,
            )
        }},
    }
}

impl NodeCodegen for NonMaxSuppressionNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::vision::Nms");
        imports.register("burn::vision::NmsOptions");
    }

    fn extra_trait_bounds(&self) -> Vec<String> {
        vec!["burn::vision::VisionBackend".to_string()]
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let [boxes_arg, scores_arg, ..] = self.inputs.as_slice() else {
            return compile_error_tokens(format!(
                "NonMaxSuppression node '{}' expects at least 2 inputs, got {}",
                self.name,
                self.inputs.len()
            ));
        };
        let [output_arg] = self.outputs.as_slice() else {
            return compile_error_tokens(format!(
                "NonMaxSuppression node '{}' expects exactly 1 output, got {}",
                self.name,
                self.outputs.len()
            ));
        };

        if !matches!(output_arg.ty, ArgType::Tensor(_)) {
            return compile_error_tokens(format!(
                "NonMaxSuppression node '{}' expects tensor output, got {:?}",
                self.name, output_arg.ty
            ));
        }

        let output = arg_to_ident(output_arg);

        let boxes = scope.arg(boxes_arg);
        let scores = scope.arg(scores_arg);

        let max_output_extract = match optional_scalar_extract(self.inputs.get(2), scope) {
            Ok(Some(extract)) => extract,
            Ok(None) => quote! { 0i64 },
            Err(message) => return compile_error_tokens(message),
        };
        let iou_extract = match optional_scalar_extract(self.inputs.get(3), scope) {
            Ok(Some(extract)) => quote! { #extract as f32 },
            Ok(None) => quote! { 0.0f32 },
            Err(message) => return compile_error_tokens(message),
        };
        let score_extract = match optional_scalar_extract(self.inputs.get(4), scope) {
            Ok(Some(extract)) => quote! { Some(#extract as f32) },
            Ok(None) => quote! { None },
            Err(message) => return compile_error_tokens(message),
        };
        let corner_boxes_expr = corner_boxes_expr(&self.config.center_point_box);

        // ONNX NonMaxSuppression operates per-batch per-class and returns [M, 3] triples
        // of (batch_index, class_index, box_index). We delegate the core NMS to
        // burn-vision's Nms trait which handles single-class [N,4] + [N] -> [M] indices.
        quote! {
            let #output = {
                let __max_output_boxes_per_class: i64 = #max_output_extract;
                let __iou_threshold: f32 = #iou_extract;
                let __score_threshold: Option<f32> = #score_extract;
                let __device = #boxes.device();
                let [__num_batches, __num_boxes, _] = #boxes.shape().dims();
                let [_, __num_classes, _] = #scores.shape().dims();
                let mut __selected_chunks: alloc::vec::Vec<Tensor<B, 2, Int>> =
                    alloc::vec::Vec::new();

                // ONNX: max_output_boxes_per_class == 0 means "select no indices"
                // burn-vision: max_output_boxes == 0 means "unlimited"
                if __max_output_boxes_per_class > 0 {
                    let __max_output_boxes = core::convert::TryFrom::try_from(
                        __max_output_boxes_per_class,
                    )
                    .unwrap_or(usize::MAX);
                    for __b in 0..__num_batches {
                        let __boxes_batch_raw: Tensor<B, 2> = #boxes
                            .clone()
                            .slice([__b..(__b + 1), 0..__num_boxes, 0..4])
                            .reshape([__num_boxes, 4]);
                        let __corner_boxes: Tensor<B, 2> = #corner_boxes_expr;
                        let __scores_batch: Tensor<B, 2> = #scores
                            .clone()
                            .slice([__b..(__b + 1), 0..__num_classes, 0..__num_boxes])
                            .reshape([__num_classes, __num_boxes]);

                        for __c in 0..__num_classes {
                            let __class_scores: Tensor<B, 1> = __scores_batch
                                .clone()
                                .slice([__c..(__c + 1), 0..__num_boxes])
                                .reshape([__num_boxes]);

                            let __nms_opts = NmsOptions {
                                iou_threshold: __iou_threshold,
                                score_threshold: __score_threshold.unwrap_or(f32::NEG_INFINITY),
                                max_output_boxes: __max_output_boxes,
                            };

                            let __kept: Tensor<B, 1, Int> =
                                __corner_boxes.clone().nms(__class_scores, __nms_opts);
                            let [__num_kept] = __kept.shape().dims();

                            if __num_kept > 0 {
                                let __batch_indices = Tensor::<B, 1, Int>::from_data(
                                    burn::tensor::TensorData::new(
                                        alloc::vec![__b as i64; __num_kept],
                                        [__num_kept],
                                    ),
                                    (&__device, burn::tensor::DType::I64),
                                );
                                let __class_indices = Tensor::<B, 1, Int>::from_data(
                                    burn::tensor::TensorData::new(
                                        alloc::vec![__c as i64; __num_kept],
                                        [__num_kept],
                                    ),
                                    (&__device, burn::tensor::DType::I64),
                                );

                                __selected_chunks.push(
                                    Tensor::cat(
                                        alloc::vec![
                                            __batch_indices.unsqueeze_dim(1),
                                            __class_indices.unsqueeze_dim(1),
                                            __kept
                                                .cast(burn::tensor::DType::I64)
                                                .unsqueeze_dim(1),
                                        ],
                                        1,
                                    ),
                                );
                            }
                        }
                    }
                }

                if __selected_chunks.is_empty() {
                    Tensor::<B, 2, Int>::from_data(
                        burn::tensor::TensorData::new(
                            alloc::vec::Vec::<i64>::new(),
                            [0, 3],
                        ),
                        (&__device, burn::tensor::DType::I64),
                    )
                } else {
                    Tensor::cat(__selected_chunks, 0)
                }
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use super::*;
    use insta::assert_snapshot;
    use onnx_ir::non_max_suppression::{NonMaxSuppressionConfig, NonMaxSuppressionNodeBuilder};

    #[test]
    fn test_nms_corner_scalar_native() {
        let config = NonMaxSuppressionConfig::new(BoxFormat::Corner);
        let node = NonMaxSuppressionNodeBuilder::new("nms")
            .input_tensor("boxes", 3, DType::F32)
            .input_tensor("scores", 3, DType::F32)
            .input_scalar("max_output_boxes_per_class", DType::I64)
            .input_scalar("iou_threshold", DType::F32)
            .input_scalar("score_threshold", DType::F32)
            .output_tensor("selected_indices", 2, DType::I64)
            .config(config)
            .build();

        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"
        pub fn forward(
            &self,
            boxes: Tensor<B, 3>,
            scores: Tensor<B, 3>,
            max_output_boxes_per_class: i64,
            iou_threshold: f32,
            score_threshold: f32,
        ) -> Tensor<B, 2, Int> {
            let selected_indices = {
                let __max_output_boxes_per_class: i64 = max_output_boxes_per_class;
                let __iou_threshold: f32 = iou_threshold as f32;
                let __score_threshold: Option<f32> = Some(score_threshold as f32);
                let __device = boxes.device();
                let [__num_batches, __num_boxes, _] = boxes.shape().dims();
                let [_, __num_classes, _] = scores.shape().dims();
                let mut __selected_chunks: alloc::vec::Vec<Tensor<B, 2, Int>> = alloc::vec::Vec::new();
                if __max_output_boxes_per_class > 0 {
                    let __max_output_boxes = core::convert::TryFrom::try_from(
                            __max_output_boxes_per_class,
                        )
                        .unwrap_or(usize::MAX);
                    for __b in 0..__num_batches {
                        let __boxes_batch_raw: Tensor<B, 2> = boxes
                            .clone()
                            .slice([__b..(__b + 1), 0..__num_boxes, 0..4])
                            .reshape([__num_boxes, 4]);
                        let __corner_boxes: Tensor<B, 2> = {
                            let __y1_raw: Tensor<B, 1> = __boxes_batch_raw
                                .clone()
                                .slice([0..__num_boxes, 0..1])
                                .reshape([__num_boxes]);
                            let __x1_raw: Tensor<B, 1> = __boxes_batch_raw
                                .clone()
                                .slice([0..__num_boxes, 1..2])
                                .reshape([__num_boxes]);
                            let __y2_raw: Tensor<B, 1> = __boxes_batch_raw
                                .clone()
                                .slice([0..__num_boxes, 2..3])
                                .reshape([__num_boxes]);
                            let __x2_raw: Tensor<B, 1> = __boxes_batch_raw
                                .clone()
                                .slice([0..__num_boxes, 3..4])
                                .reshape([__num_boxes]);
                            let __x1 = __x1_raw.clone().min_pair(__x2_raw.clone());
                            let __y1 = __y1_raw.clone().min_pair(__y2_raw.clone());
                            let __x2 = __x1_raw.max_pair(__x2_raw);
                            let __y2 = __y1_raw.max_pair(__y2_raw);
                            Tensor::cat(
                                alloc::vec![
                                    __x1.unsqueeze_dim(1), __y1.unsqueeze_dim(1), __x2
                                    .unsqueeze_dim(1), __y2.unsqueeze_dim(1),
                                ],
                                1,
                            )
                        };
                        let __scores_batch: Tensor<B, 2> = scores
                            .clone()
                            .slice([__b..(__b + 1), 0..__num_classes, 0..__num_boxes])
                            .reshape([__num_classes, __num_boxes]);
                        for __c in 0..__num_classes {
                            let __class_scores: Tensor<B, 1> = __scores_batch
                                .clone()
                                .slice([__c..(__c + 1), 0..__num_boxes])
                                .reshape([__num_boxes]);
                            let __nms_opts = NmsOptions {
                                iou_threshold: __iou_threshold,
                                score_threshold: __score_threshold.unwrap_or(f32::NEG_INFINITY),
                                max_output_boxes: __max_output_boxes,
                            };
                            let __kept: Tensor<B, 1, Int> = __corner_boxes
                                .clone()
                                .nms(__class_scores, __nms_opts);
                            let [__num_kept] = __kept.shape().dims();
                            if __num_kept > 0 {
                                let __batch_indices = Tensor::<
                                    B,
                                    1,
                                    Int,
                                >::from_data(
                                    burn::tensor::TensorData::new(
                                        alloc::vec![__b as i64; __num_kept],
                                        [__num_kept],
                                    ),
                                    (&__device, burn::tensor::DType::I64),
                                );
                                let __class_indices = Tensor::<
                                    B,
                                    1,
                                    Int,
                                >::from_data(
                                    burn::tensor::TensorData::new(
                                        alloc::vec![__c as i64; __num_kept],
                                        [__num_kept],
                                    ),
                                    (&__device, burn::tensor::DType::I64),
                                );
                                __selected_chunks
                                    .push(
                                        Tensor::cat(
                                            alloc::vec![
                                                __batch_indices.unsqueeze_dim(1), __class_indices
                                                .unsqueeze_dim(1), __kept.cast(burn::tensor::DType::I64)
                                                .unsqueeze_dim(1),
                                            ],
                                            1,
                                        ),
                                    );
                            }
                        }
                    }
                }
                if __selected_chunks.is_empty() {
                    Tensor::<
                        B,
                        2,
                        Int,
                    >::from_data(
                        burn::tensor::TensorData::new(alloc::vec::Vec::<i64>::new(), [0, 3]),
                        (&__device, burn::tensor::DType::I64),
                    )
                } else {
                    Tensor::cat(__selected_chunks, 0)
                }
            };
            selected_indices
        }
        ");
    }

    #[test]
    fn test_nms_corner_scalar_tensor() {
        let config = NonMaxSuppressionConfig::new(BoxFormat::Corner);
        let node = NonMaxSuppressionNodeBuilder::new("nms")
            .input_tensor("boxes", 3, DType::F32)
            .input_tensor("scores", 3, DType::F32)
            .input_scalar_tensor("max_output_boxes_per_class", DType::I64)
            .input_scalar_tensor("iou_threshold", DType::F32)
            .input_scalar_tensor("score_threshold", DType::F32)
            .output_tensor("selected_indices", 2, DType::I64)
            .config(config)
            .build();

        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"
        pub fn forward(
            &self,
            boxes: Tensor<B, 3>,
            scores: Tensor<B, 3>,
            max_output_boxes_per_class: Tensor<B, 1, Int>,
            iou_threshold: Tensor<B, 1>,
            score_threshold: Tensor<B, 1>,
        ) -> Tensor<B, 2, Int> {
            let selected_indices = {
                let __max_output_boxes_per_class: i64 = max_output_boxes_per_class
                    .into_scalar()
                    .elem::<i64>();
                let __iou_threshold: f32 = iou_threshold.into_scalar().elem::<f32>() as f32;
                let __score_threshold: Option<f32> = Some(
                    score_threshold.into_scalar().elem::<f32>() as f32,
                );
                let __device = boxes.device();
                let [__num_batches, __num_boxes, _] = boxes.shape().dims();
                let [_, __num_classes, _] = scores.shape().dims();
                let mut __selected_chunks: alloc::vec::Vec<Tensor<B, 2, Int>> = alloc::vec::Vec::new();
                if __max_output_boxes_per_class > 0 {
                    let __max_output_boxes = core::convert::TryFrom::try_from(
                            __max_output_boxes_per_class,
                        )
                        .unwrap_or(usize::MAX);
                    for __b in 0..__num_batches {
                        let __boxes_batch_raw: Tensor<B, 2> = boxes
                            .clone()
                            .slice([__b..(__b + 1), 0..__num_boxes, 0..4])
                            .reshape([__num_boxes, 4]);
                        let __corner_boxes: Tensor<B, 2> = {
                            let __y1_raw: Tensor<B, 1> = __boxes_batch_raw
                                .clone()
                                .slice([0..__num_boxes, 0..1])
                                .reshape([__num_boxes]);
                            let __x1_raw: Tensor<B, 1> = __boxes_batch_raw
                                .clone()
                                .slice([0..__num_boxes, 1..2])
                                .reshape([__num_boxes]);
                            let __y2_raw: Tensor<B, 1> = __boxes_batch_raw
                                .clone()
                                .slice([0..__num_boxes, 2..3])
                                .reshape([__num_boxes]);
                            let __x2_raw: Tensor<B, 1> = __boxes_batch_raw
                                .clone()
                                .slice([0..__num_boxes, 3..4])
                                .reshape([__num_boxes]);
                            let __x1 = __x1_raw.clone().min_pair(__x2_raw.clone());
                            let __y1 = __y1_raw.clone().min_pair(__y2_raw.clone());
                            let __x2 = __x1_raw.max_pair(__x2_raw);
                            let __y2 = __y1_raw.max_pair(__y2_raw);
                            Tensor::cat(
                                alloc::vec![
                                    __x1.unsqueeze_dim(1), __y1.unsqueeze_dim(1), __x2
                                    .unsqueeze_dim(1), __y2.unsqueeze_dim(1),
                                ],
                                1,
                            )
                        };
                        let __scores_batch: Tensor<B, 2> = scores
                            .clone()
                            .slice([__b..(__b + 1), 0..__num_classes, 0..__num_boxes])
                            .reshape([__num_classes, __num_boxes]);
                        for __c in 0..__num_classes {
                            let __class_scores: Tensor<B, 1> = __scores_batch
                                .clone()
                                .slice([__c..(__c + 1), 0..__num_boxes])
                                .reshape([__num_boxes]);
                            let __nms_opts = NmsOptions {
                                iou_threshold: __iou_threshold,
                                score_threshold: __score_threshold.unwrap_or(f32::NEG_INFINITY),
                                max_output_boxes: __max_output_boxes,
                            };
                            let __kept: Tensor<B, 1, Int> = __corner_boxes
                                .clone()
                                .nms(__class_scores, __nms_opts);
                            let [__num_kept] = __kept.shape().dims();
                            if __num_kept > 0 {
                                let __batch_indices = Tensor::<
                                    B,
                                    1,
                                    Int,
                                >::from_data(
                                    burn::tensor::TensorData::new(
                                        alloc::vec![__b as i64; __num_kept],
                                        [__num_kept],
                                    ),
                                    (&__device, burn::tensor::DType::I64),
                                );
                                let __class_indices = Tensor::<
                                    B,
                                    1,
                                    Int,
                                >::from_data(
                                    burn::tensor::TensorData::new(
                                        alloc::vec![__c as i64; __num_kept],
                                        [__num_kept],
                                    ),
                                    (&__device, burn::tensor::DType::I64),
                                );
                                __selected_chunks
                                    .push(
                                        Tensor::cat(
                                            alloc::vec![
                                                __batch_indices.unsqueeze_dim(1), __class_indices
                                                .unsqueeze_dim(1), __kept.cast(burn::tensor::DType::I64)
                                                .unsqueeze_dim(1),
                                            ],
                                            1,
                                        ),
                                    );
                            }
                        }
                    }
                }
                if __selected_chunks.is_empty() {
                    Tensor::<
                        B,
                        2,
                        Int,
                    >::from_data(
                        burn::tensor::TensorData::new(alloc::vec::Vec::<i64>::new(), [0, 3]),
                        (&__device, burn::tensor::DType::I64),
                    )
                } else {
                    Tensor::cat(__selected_chunks, 0)
                }
            };
            selected_indices
        }
        ");
    }

    #[test]
    fn test_nms_corner_rank1_tensor() {
        let config = NonMaxSuppressionConfig::new(BoxFormat::Corner);
        let node = NonMaxSuppressionNodeBuilder::new("nms")
            .input_tensor("boxes", 3, DType::F32)
            .input_tensor("scores", 3, DType::F32)
            .input_tensor("max_output_boxes_per_class", 1, DType::I64)
            .input_tensor("iou_threshold", 1, DType::F32)
            .input_tensor("score_threshold", 1, DType::F32)
            .output_tensor("selected_indices", 2, DType::I64)
            .config(config)
            .build();

        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"
        pub fn forward(
            &self,
            boxes: Tensor<B, 3>,
            scores: Tensor<B, 3>,
            max_output_boxes_per_class: Tensor<B, 1, Int>,
            iou_threshold: Tensor<B, 1>,
            score_threshold: Tensor<B, 1>,
        ) -> Tensor<B, 2, Int> {
            let selected_indices = {
                let __max_output_boxes_per_class: i64 = max_output_boxes_per_class
                    .into_scalar()
                    .elem::<i64>();
                let __iou_threshold: f32 = iou_threshold.into_scalar().elem::<f32>() as f32;
                let __score_threshold: Option<f32> = Some(
                    score_threshold.into_scalar().elem::<f32>() as f32,
                );
                let __device = boxes.device();
                let [__num_batches, __num_boxes, _] = boxes.shape().dims();
                let [_, __num_classes, _] = scores.shape().dims();
                let mut __selected_chunks: alloc::vec::Vec<Tensor<B, 2, Int>> = alloc::vec::Vec::new();
                if __max_output_boxes_per_class > 0 {
                    let __max_output_boxes = core::convert::TryFrom::try_from(
                            __max_output_boxes_per_class,
                        )
                        .unwrap_or(usize::MAX);
                    for __b in 0..__num_batches {
                        let __boxes_batch_raw: Tensor<B, 2> = boxes
                            .clone()
                            .slice([__b..(__b + 1), 0..__num_boxes, 0..4])
                            .reshape([__num_boxes, 4]);
                        let __corner_boxes: Tensor<B, 2> = {
                            let __y1_raw: Tensor<B, 1> = __boxes_batch_raw
                                .clone()
                                .slice([0..__num_boxes, 0..1])
                                .reshape([__num_boxes]);
                            let __x1_raw: Tensor<B, 1> = __boxes_batch_raw
                                .clone()
                                .slice([0..__num_boxes, 1..2])
                                .reshape([__num_boxes]);
                            let __y2_raw: Tensor<B, 1> = __boxes_batch_raw
                                .clone()
                                .slice([0..__num_boxes, 2..3])
                                .reshape([__num_boxes]);
                            let __x2_raw: Tensor<B, 1> = __boxes_batch_raw
                                .clone()
                                .slice([0..__num_boxes, 3..4])
                                .reshape([__num_boxes]);
                            let __x1 = __x1_raw.clone().min_pair(__x2_raw.clone());
                            let __y1 = __y1_raw.clone().min_pair(__y2_raw.clone());
                            let __x2 = __x1_raw.max_pair(__x2_raw);
                            let __y2 = __y1_raw.max_pair(__y2_raw);
                            Tensor::cat(
                                alloc::vec![
                                    __x1.unsqueeze_dim(1), __y1.unsqueeze_dim(1), __x2
                                    .unsqueeze_dim(1), __y2.unsqueeze_dim(1),
                                ],
                                1,
                            )
                        };
                        let __scores_batch: Tensor<B, 2> = scores
                            .clone()
                            .slice([__b..(__b + 1), 0..__num_classes, 0..__num_boxes])
                            .reshape([__num_classes, __num_boxes]);
                        for __c in 0..__num_classes {
                            let __class_scores: Tensor<B, 1> = __scores_batch
                                .clone()
                                .slice([__c..(__c + 1), 0..__num_boxes])
                                .reshape([__num_boxes]);
                            let __nms_opts = NmsOptions {
                                iou_threshold: __iou_threshold,
                                score_threshold: __score_threshold.unwrap_or(f32::NEG_INFINITY),
                                max_output_boxes: __max_output_boxes,
                            };
                            let __kept: Tensor<B, 1, Int> = __corner_boxes
                                .clone()
                                .nms(__class_scores, __nms_opts);
                            let [__num_kept] = __kept.shape().dims();
                            if __num_kept > 0 {
                                let __batch_indices = Tensor::<
                                    B,
                                    1,
                                    Int,
                                >::from_data(
                                    burn::tensor::TensorData::new(
                                        alloc::vec![__b as i64; __num_kept],
                                        [__num_kept],
                                    ),
                                    (&__device, burn::tensor::DType::I64),
                                );
                                let __class_indices = Tensor::<
                                    B,
                                    1,
                                    Int,
                                >::from_data(
                                    burn::tensor::TensorData::new(
                                        alloc::vec![__c as i64; __num_kept],
                                        [__num_kept],
                                    ),
                                    (&__device, burn::tensor::DType::I64),
                                );
                                __selected_chunks
                                    .push(
                                        Tensor::cat(
                                            alloc::vec![
                                                __batch_indices.unsqueeze_dim(1), __class_indices
                                                .unsqueeze_dim(1), __kept.cast(burn::tensor::DType::I64)
                                                .unsqueeze_dim(1),
                                            ],
                                            1,
                                        ),
                                    );
                            }
                        }
                    }
                }
                if __selected_chunks.is_empty() {
                    Tensor::<
                        B,
                        2,
                        Int,
                    >::from_data(
                        burn::tensor::TensorData::new(alloc::vec::Vec::<i64>::new(), [0, 3]),
                        (&__device, burn::tensor::DType::I64),
                    )
                } else {
                    Tensor::cat(__selected_chunks, 0)
                }
            };
            selected_indices
        }
        ");
    }

    #[test]
    fn test_nms_center_scalar_native() {
        let config = NonMaxSuppressionConfig::new(BoxFormat::Center);
        let node = NonMaxSuppressionNodeBuilder::new("nms")
            .input_tensor("boxes", 3, DType::F32)
            .input_tensor("scores", 3, DType::F32)
            .input_scalar("max_output_boxes_per_class", DType::I64)
            .input_scalar("iou_threshold", DType::F32)
            .input_scalar("score_threshold", DType::F32)
            .output_tensor("selected_indices", 2, DType::I64)
            .config(config)
            .build();

        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"
        pub fn forward(
            &self,
            boxes: Tensor<B, 3>,
            scores: Tensor<B, 3>,
            max_output_boxes_per_class: i64,
            iou_threshold: f32,
            score_threshold: f32,
        ) -> Tensor<B, 2, Int> {
            let selected_indices = {
                let __max_output_boxes_per_class: i64 = max_output_boxes_per_class;
                let __iou_threshold: f32 = iou_threshold as f32;
                let __score_threshold: Option<f32> = Some(score_threshold as f32);
                let __device = boxes.device();
                let [__num_batches, __num_boxes, _] = boxes.shape().dims();
                let [_, __num_classes, _] = scores.shape().dims();
                let mut __selected_chunks: alloc::vec::Vec<Tensor<B, 2, Int>> = alloc::vec::Vec::new();
                if __max_output_boxes_per_class > 0 {
                    let __max_output_boxes = core::convert::TryFrom::try_from(
                            __max_output_boxes_per_class,
                        )
                        .unwrap_or(usize::MAX);
                    for __b in 0..__num_batches {
                        let __boxes_batch_raw: Tensor<B, 2> = boxes
                            .clone()
                            .slice([__b..(__b + 1), 0..__num_boxes, 0..4])
                            .reshape([__num_boxes, 4]);
                        let __corner_boxes: Tensor<B, 2> = {
                            let __cx: Tensor<B, 1> = __boxes_batch_raw
                                .clone()
                                .slice([0..__num_boxes, 0..1])
                                .reshape([__num_boxes]);
                            let __cy: Tensor<B, 1> = __boxes_batch_raw
                                .clone()
                                .slice([0..__num_boxes, 1..2])
                                .reshape([__num_boxes]);
                            let __w: Tensor<B, 1> = __boxes_batch_raw
                                .clone()
                                .slice([0..__num_boxes, 2..3])
                                .reshape([__num_boxes]);
                            let __h: Tensor<B, 1> = __boxes_batch_raw
                                .clone()
                                .slice([0..__num_boxes, 3..4])
                                .reshape([__num_boxes]);
                            let __half_w = __w / 2.0f32;
                            let __half_h = __h / 2.0f32;
                            let __x1 = __cx.clone() - __half_w.clone();
                            let __y1 = __cy.clone() - __half_h.clone();
                            let __x2 = __cx + __half_w;
                            let __y2 = __cy + __half_h;
                            Tensor::cat(
                                alloc::vec![
                                    __x1.unsqueeze_dim(1), __y1.unsqueeze_dim(1), __x2
                                    .unsqueeze_dim(1), __y2.unsqueeze_dim(1),
                                ],
                                1,
                            )
                        };
                        let __scores_batch: Tensor<B, 2> = scores
                            .clone()
                            .slice([__b..(__b + 1), 0..__num_classes, 0..__num_boxes])
                            .reshape([__num_classes, __num_boxes]);
                        for __c in 0..__num_classes {
                            let __class_scores: Tensor<B, 1> = __scores_batch
                                .clone()
                                .slice([__c..(__c + 1), 0..__num_boxes])
                                .reshape([__num_boxes]);
                            let __nms_opts = NmsOptions {
                                iou_threshold: __iou_threshold,
                                score_threshold: __score_threshold.unwrap_or(f32::NEG_INFINITY),
                                max_output_boxes: __max_output_boxes,
                            };
                            let __kept: Tensor<B, 1, Int> = __corner_boxes
                                .clone()
                                .nms(__class_scores, __nms_opts);
                            let [__num_kept] = __kept.shape().dims();
                            if __num_kept > 0 {
                                let __batch_indices = Tensor::<
                                    B,
                                    1,
                                    Int,
                                >::from_data(
                                    burn::tensor::TensorData::new(
                                        alloc::vec![__b as i64; __num_kept],
                                        [__num_kept],
                                    ),
                                    (&__device, burn::tensor::DType::I64),
                                );
                                let __class_indices = Tensor::<
                                    B,
                                    1,
                                    Int,
                                >::from_data(
                                    burn::tensor::TensorData::new(
                                        alloc::vec![__c as i64; __num_kept],
                                        [__num_kept],
                                    ),
                                    (&__device, burn::tensor::DType::I64),
                                );
                                __selected_chunks
                                    .push(
                                        Tensor::cat(
                                            alloc::vec![
                                                __batch_indices.unsqueeze_dim(1), __class_indices
                                                .unsqueeze_dim(1), __kept.cast(burn::tensor::DType::I64)
                                                .unsqueeze_dim(1),
                                            ],
                                            1,
                                        ),
                                    );
                            }
                        }
                    }
                }
                if __selected_chunks.is_empty() {
                    Tensor::<
                        B,
                        2,
                        Int,
                    >::from_data(
                        burn::tensor::TensorData::new(alloc::vec::Vec::<i64>::new(), [0, 3]),
                        (&__device, burn::tensor::DType::I64),
                    )
                } else {
                    Tensor::cat(__selected_chunks, 0)
                }
            };
            selected_indices
        }
        ");
    }

    #[test]
    fn test_nms_minimal_inputs() {
        let node = NonMaxSuppressionNodeBuilder::new("nms")
            .input_tensor("boxes", 3, DType::F32)
            .input_tensor("scores", 3, DType::F32)
            .output_tensor("selected_indices", 2, DType::I64)
            .config(NonMaxSuppressionConfig::default())
            .build();

        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"
        pub fn forward(&self, boxes: Tensor<B, 3>, scores: Tensor<B, 3>) -> Tensor<B, 2, Int> {
            let selected_indices = {
                let __max_output_boxes_per_class: i64 = 0i64;
                let __iou_threshold: f32 = 0.0f32;
                let __score_threshold: Option<f32> = None;
                let __device = boxes.device();
                let [__num_batches, __num_boxes, _] = boxes.shape().dims();
                let [_, __num_classes, _] = scores.shape().dims();
                let mut __selected_chunks: alloc::vec::Vec<Tensor<B, 2, Int>> = alloc::vec::Vec::new();
                if __max_output_boxes_per_class > 0 {
                    let __max_output_boxes = core::convert::TryFrom::try_from(
                            __max_output_boxes_per_class,
                        )
                        .unwrap_or(usize::MAX);
                    for __b in 0..__num_batches {
                        let __boxes_batch_raw: Tensor<B, 2> = boxes
                            .clone()
                            .slice([__b..(__b + 1), 0..__num_boxes, 0..4])
                            .reshape([__num_boxes, 4]);
                        let __corner_boxes: Tensor<B, 2> = {
                            let __y1_raw: Tensor<B, 1> = __boxes_batch_raw
                                .clone()
                                .slice([0..__num_boxes, 0..1])
                                .reshape([__num_boxes]);
                            let __x1_raw: Tensor<B, 1> = __boxes_batch_raw
                                .clone()
                                .slice([0..__num_boxes, 1..2])
                                .reshape([__num_boxes]);
                            let __y2_raw: Tensor<B, 1> = __boxes_batch_raw
                                .clone()
                                .slice([0..__num_boxes, 2..3])
                                .reshape([__num_boxes]);
                            let __x2_raw: Tensor<B, 1> = __boxes_batch_raw
                                .clone()
                                .slice([0..__num_boxes, 3..4])
                                .reshape([__num_boxes]);
                            let __x1 = __x1_raw.clone().min_pair(__x2_raw.clone());
                            let __y1 = __y1_raw.clone().min_pair(__y2_raw.clone());
                            let __x2 = __x1_raw.max_pair(__x2_raw);
                            let __y2 = __y1_raw.max_pair(__y2_raw);
                            Tensor::cat(
                                alloc::vec![
                                    __x1.unsqueeze_dim(1), __y1.unsqueeze_dim(1), __x2
                                    .unsqueeze_dim(1), __y2.unsqueeze_dim(1),
                                ],
                                1,
                            )
                        };
                        let __scores_batch: Tensor<B, 2> = scores
                            .clone()
                            .slice([__b..(__b + 1), 0..__num_classes, 0..__num_boxes])
                            .reshape([__num_classes, __num_boxes]);
                        for __c in 0..__num_classes {
                            let __class_scores: Tensor<B, 1> = __scores_batch
                                .clone()
                                .slice([__c..(__c + 1), 0..__num_boxes])
                                .reshape([__num_boxes]);
                            let __nms_opts = NmsOptions {
                                iou_threshold: __iou_threshold,
                                score_threshold: __score_threshold.unwrap_or(f32::NEG_INFINITY),
                                max_output_boxes: __max_output_boxes,
                            };
                            let __kept: Tensor<B, 1, Int> = __corner_boxes
                                .clone()
                                .nms(__class_scores, __nms_opts);
                            let [__num_kept] = __kept.shape().dims();
                            if __num_kept > 0 {
                                let __batch_indices = Tensor::<
                                    B,
                                    1,
                                    Int,
                                >::from_data(
                                    burn::tensor::TensorData::new(
                                        alloc::vec![__b as i64; __num_kept],
                                        [__num_kept],
                                    ),
                                    (&__device, burn::tensor::DType::I64),
                                );
                                let __class_indices = Tensor::<
                                    B,
                                    1,
                                    Int,
                                >::from_data(
                                    burn::tensor::TensorData::new(
                                        alloc::vec![__c as i64; __num_kept],
                                        [__num_kept],
                                    ),
                                    (&__device, burn::tensor::DType::I64),
                                );
                                __selected_chunks
                                    .push(
                                        Tensor::cat(
                                            alloc::vec![
                                                __batch_indices.unsqueeze_dim(1), __class_indices
                                                .unsqueeze_dim(1), __kept.cast(burn::tensor::DType::I64)
                                                .unsqueeze_dim(1),
                                            ],
                                            1,
                                        ),
                                    );
                            }
                        }
                    }
                }
                if __selected_chunks.is_empty() {
                    Tensor::<
                        B,
                        2,
                        Int,
                    >::from_data(
                        burn::tensor::TensorData::new(alloc::vec::Vec::<i64>::new(), [0, 3]),
                        (&__device, burn::tensor::DType::I64),
                    )
                } else {
                    Tensor::cat(__selected_chunks, 0)
                }
            };
            selected_indices
        }
        ");
    }

    #[test]
    fn test_nms_missing_middle_optional_input() {
        let node = NonMaxSuppressionNodeBuilder::new("nms")
            .input_tensor("boxes", 3, DType::F32)
            .input_tensor("scores", 3, DType::F32)
            .input_scalar("max_output_boxes_per_class", DType::I64)
            .input_scalar("", DType::F32)
            .input_scalar("score_threshold", DType::F32)
            .output_tensor("selected_indices", 2, DType::I64)
            .config(NonMaxSuppressionConfig::default())
            .build();

        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"
        pub fn forward(
            &self,
            boxes: Tensor<B, 3>,
            scores: Tensor<B, 3>,
            max_output_boxes_per_class: i64,
            score_threshold: f32,
        ) -> Tensor<B, 2, Int> {
            let selected_indices = {
                let __max_output_boxes_per_class: i64 = max_output_boxes_per_class;
                let __iou_threshold: f32 = 0.0f32;
                let __score_threshold: Option<f32> = Some(score_threshold as f32);
                let __device = boxes.device();
                let [__num_batches, __num_boxes, _] = boxes.shape().dims();
                let [_, __num_classes, _] = scores.shape().dims();
                let mut __selected_chunks: alloc::vec::Vec<Tensor<B, 2, Int>> = alloc::vec::Vec::new();
                if __max_output_boxes_per_class > 0 {
                    let __max_output_boxes = core::convert::TryFrom::try_from(
                            __max_output_boxes_per_class,
                        )
                        .unwrap_or(usize::MAX);
                    for __b in 0..__num_batches {
                        let __boxes_batch_raw: Tensor<B, 2> = boxes
                            .clone()
                            .slice([__b..(__b + 1), 0..__num_boxes, 0..4])
                            .reshape([__num_boxes, 4]);
                        let __corner_boxes: Tensor<B, 2> = {
                            let __y1_raw: Tensor<B, 1> = __boxes_batch_raw
                                .clone()
                                .slice([0..__num_boxes, 0..1])
                                .reshape([__num_boxes]);
                            let __x1_raw: Tensor<B, 1> = __boxes_batch_raw
                                .clone()
                                .slice([0..__num_boxes, 1..2])
                                .reshape([__num_boxes]);
                            let __y2_raw: Tensor<B, 1> = __boxes_batch_raw
                                .clone()
                                .slice([0..__num_boxes, 2..3])
                                .reshape([__num_boxes]);
                            let __x2_raw: Tensor<B, 1> = __boxes_batch_raw
                                .clone()
                                .slice([0..__num_boxes, 3..4])
                                .reshape([__num_boxes]);
                            let __x1 = __x1_raw.clone().min_pair(__x2_raw.clone());
                            let __y1 = __y1_raw.clone().min_pair(__y2_raw.clone());
                            let __x2 = __x1_raw.max_pair(__x2_raw);
                            let __y2 = __y1_raw.max_pair(__y2_raw);
                            Tensor::cat(
                                alloc::vec![
                                    __x1.unsqueeze_dim(1), __y1.unsqueeze_dim(1), __x2
                                    .unsqueeze_dim(1), __y2.unsqueeze_dim(1),
                                ],
                                1,
                            )
                        };
                        let __scores_batch: Tensor<B, 2> = scores
                            .clone()
                            .slice([__b..(__b + 1), 0..__num_classes, 0..__num_boxes])
                            .reshape([__num_classes, __num_boxes]);
                        for __c in 0..__num_classes {
                            let __class_scores: Tensor<B, 1> = __scores_batch
                                .clone()
                                .slice([__c..(__c + 1), 0..__num_boxes])
                                .reshape([__num_boxes]);
                            let __nms_opts = NmsOptions {
                                iou_threshold: __iou_threshold,
                                score_threshold: __score_threshold.unwrap_or(f32::NEG_INFINITY),
                                max_output_boxes: __max_output_boxes,
                            };
                            let __kept: Tensor<B, 1, Int> = __corner_boxes
                                .clone()
                                .nms(__class_scores, __nms_opts);
                            let [__num_kept] = __kept.shape().dims();
                            if __num_kept > 0 {
                                let __batch_indices = Tensor::<
                                    B,
                                    1,
                                    Int,
                                >::from_data(
                                    burn::tensor::TensorData::new(
                                        alloc::vec![__b as i64; __num_kept],
                                        [__num_kept],
                                    ),
                                    (&__device, burn::tensor::DType::I64),
                                );
                                let __class_indices = Tensor::<
                                    B,
                                    1,
                                    Int,
                                >::from_data(
                                    burn::tensor::TensorData::new(
                                        alloc::vec![__c as i64; __num_kept],
                                        [__num_kept],
                                    ),
                                    (&__device, burn::tensor::DType::I64),
                                );
                                __selected_chunks
                                    .push(
                                        Tensor::cat(
                                            alloc::vec![
                                                __batch_indices.unsqueeze_dim(1), __class_indices
                                                .unsqueeze_dim(1), __kept.cast(burn::tensor::DType::I64)
                                                .unsqueeze_dim(1),
                                            ],
                                            1,
                                        ),
                                    );
                            }
                        }
                    }
                }
                if __selected_chunks.is_empty() {
                    Tensor::<
                        B,
                        2,
                        Int,
                    >::from_data(
                        burn::tensor::TensorData::new(alloc::vec::Vec::<i64>::new(), [0, 3]),
                        (&__device, burn::tensor::DType::I64),
                    )
                } else {
                    Tensor::cat(__selected_chunks, 0)
                }
            };
            selected_indices
        }
        ");
    }

    #[test]
    fn test_nms_missing_trailing_optional_input() {
        let node = NonMaxSuppressionNodeBuilder::new("nms")
            .input_tensor("boxes", 3, DType::F32)
            .input_tensor("scores", 3, DType::F32)
            .input_scalar("max_output_boxes_per_class", DType::I64)
            .input_scalar("iou_threshold", DType::F32)
            .output_tensor("selected_indices", 2, DType::I64)
            .config(NonMaxSuppressionConfig::default())
            .build();

        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"
        pub fn forward(
            &self,
            boxes: Tensor<B, 3>,
            scores: Tensor<B, 3>,
            max_output_boxes_per_class: i64,
            iou_threshold: f32,
        ) -> Tensor<B, 2, Int> {
            let selected_indices = {
                let __max_output_boxes_per_class: i64 = max_output_boxes_per_class;
                let __iou_threshold: f32 = iou_threshold as f32;
                let __score_threshold: Option<f32> = None;
                let __device = boxes.device();
                let [__num_batches, __num_boxes, _] = boxes.shape().dims();
                let [_, __num_classes, _] = scores.shape().dims();
                let mut __selected_chunks: alloc::vec::Vec<Tensor<B, 2, Int>> = alloc::vec::Vec::new();
                if __max_output_boxes_per_class > 0 {
                    let __max_output_boxes = core::convert::TryFrom::try_from(
                            __max_output_boxes_per_class,
                        )
                        .unwrap_or(usize::MAX);
                    for __b in 0..__num_batches {
                        let __boxes_batch_raw: Tensor<B, 2> = boxes
                            .clone()
                            .slice([__b..(__b + 1), 0..__num_boxes, 0..4])
                            .reshape([__num_boxes, 4]);
                        let __corner_boxes: Tensor<B, 2> = {
                            let __y1_raw: Tensor<B, 1> = __boxes_batch_raw
                                .clone()
                                .slice([0..__num_boxes, 0..1])
                                .reshape([__num_boxes]);
                            let __x1_raw: Tensor<B, 1> = __boxes_batch_raw
                                .clone()
                                .slice([0..__num_boxes, 1..2])
                                .reshape([__num_boxes]);
                            let __y2_raw: Tensor<B, 1> = __boxes_batch_raw
                                .clone()
                                .slice([0..__num_boxes, 2..3])
                                .reshape([__num_boxes]);
                            let __x2_raw: Tensor<B, 1> = __boxes_batch_raw
                                .clone()
                                .slice([0..__num_boxes, 3..4])
                                .reshape([__num_boxes]);
                            let __x1 = __x1_raw.clone().min_pair(__x2_raw.clone());
                            let __y1 = __y1_raw.clone().min_pair(__y2_raw.clone());
                            let __x2 = __x1_raw.max_pair(__x2_raw);
                            let __y2 = __y1_raw.max_pair(__y2_raw);
                            Tensor::cat(
                                alloc::vec![
                                    __x1.unsqueeze_dim(1), __y1.unsqueeze_dim(1), __x2
                                    .unsqueeze_dim(1), __y2.unsqueeze_dim(1),
                                ],
                                1,
                            )
                        };
                        let __scores_batch: Tensor<B, 2> = scores
                            .clone()
                            .slice([__b..(__b + 1), 0..__num_classes, 0..__num_boxes])
                            .reshape([__num_classes, __num_boxes]);
                        for __c in 0..__num_classes {
                            let __class_scores: Tensor<B, 1> = __scores_batch
                                .clone()
                                .slice([__c..(__c + 1), 0..__num_boxes])
                                .reshape([__num_boxes]);
                            let __nms_opts = NmsOptions {
                                iou_threshold: __iou_threshold,
                                score_threshold: __score_threshold.unwrap_or(f32::NEG_INFINITY),
                                max_output_boxes: __max_output_boxes,
                            };
                            let __kept: Tensor<B, 1, Int> = __corner_boxes
                                .clone()
                                .nms(__class_scores, __nms_opts);
                            let [__num_kept] = __kept.shape().dims();
                            if __num_kept > 0 {
                                let __batch_indices = Tensor::<
                                    B,
                                    1,
                                    Int,
                                >::from_data(
                                    burn::tensor::TensorData::new(
                                        alloc::vec![__b as i64; __num_kept],
                                        [__num_kept],
                                    ),
                                    (&__device, burn::tensor::DType::I64),
                                );
                                let __class_indices = Tensor::<
                                    B,
                                    1,
                                    Int,
                                >::from_data(
                                    burn::tensor::TensorData::new(
                                        alloc::vec![__c as i64; __num_kept],
                                        [__num_kept],
                                    ),
                                    (&__device, burn::tensor::DType::I64),
                                );
                                __selected_chunks
                                    .push(
                                        Tensor::cat(
                                            alloc::vec![
                                                __batch_indices.unsqueeze_dim(1), __class_indices
                                                .unsqueeze_dim(1), __kept.cast(burn::tensor::DType::I64)
                                                .unsqueeze_dim(1),
                                            ],
                                            1,
                                        ),
                                    );
                            }
                        }
                    }
                }
                if __selected_chunks.is_empty() {
                    Tensor::<
                        B,
                        2,
                        Int,
                    >::from_data(
                        burn::tensor::TensorData::new(alloc::vec::Vec::<i64>::new(), [0, 3]),
                        (&__device, burn::tensor::DType::I64),
                    )
                } else {
                    Tensor::cat(__selected_chunks, 0)
                }
            };
            selected_indices
        }
        ");
    }
}
