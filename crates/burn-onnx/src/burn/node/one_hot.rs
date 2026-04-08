use super::prelude::*;
use crate::burn::TensorKind;

impl NodeCodegen for onnx_ir::one_hot::OneHotNode {
    fn inputs(&self) -> &[Argument] {
        // Only the first input (indices) is a dynamic tensor
        // depth and values are either static or runtime inputs
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        // Runtime depth/values are bound to `__onehot_*` locals inside a
        // prelude block so (1) the main one_hot_fill call stays readable
        // and (2) multiple OneHot nodes in the same forward() can't
        // collide on the temporary names.
        let mut prelude = TokenStream::new();

        let depth_expr = match &self.config.depth {
            onnx_ir::one_hot::OneHotDepthInput::Static(d) => quote! { #d },
            onnx_ir::one_hot::OneHotDepthInput::Runtime(r) => {
                let arg = &self.inputs[r.input_index];
                match &arg.ty {
                    ArgType::ScalarNative(_) => {
                        let ident = arg_to_ident(arg);
                        prelude.extend(quote! {
                            let __onehot_depth: usize = #ident as usize;
                        });
                    }
                    ArgType::ScalarTensor(_) | ArgType::Tensor(_) => {
                        let tensor = scope.arg(arg);
                        prelude.extend(quote! {
                            let __onehot_depth: usize = {
                                let __data = #tensor.to_data().convert::<i64>();
                                __data.as_slice::<i64>().unwrap()[0] as usize
                            };
                        });
                    }
                    other => {
                        panic!("OneHot depth must be a scalar or rank-1 tensor, got {other:?}")
                    }
                }
                quote! { __onehot_depth }
            }
        };

        let (on_value, off_value) = match &self.config.values {
            onnx_ir::one_hot::OneHotValuesInput::Static(v) => {
                let off = v[0];
                let on = v[1];
                (quote! { #on }, quote! { #off })
            }
            onnx_ir::one_hot::OneHotValuesInput::Runtime(r) => {
                let arg = &self.inputs[r.input_index];
                let tensor = scope.arg(arg);
                // The values input is a rank-1 tensor [off_value, on_value].
                // Burn's `one_hot_fill` signature pins `on_value`/`off_value`
                // to concrete `f32`, so we have to narrow to f32 here even
                // though ONNX's T2 constraint allows any numeric dtype.
                // This silently rounds int64 values above 2^24 - tracked in
                // #317 as a followup; resolving it properly requires either
                // an overload of `one_hot_fill` that's generic over
                // `E: ElementConversion` or a tensor-input one_hot path
                // that never materializes on/off as host scalars.
                prelude.extend(quote! {
                    let (__onehot_off, __onehot_on): (f32, f32) = {
                        let __data = #tensor.to_data().convert::<f32>();
                        let __slice = __data.as_slice::<f32>().unwrap();
                        (__slice[0], __slice[1])
                    };
                });
                (quote! { __onehot_on }, quote! { __onehot_off })
            }
        };

        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());
        let num_classes = depth_expr;

        let axis = self.config.axis;

        // Determine input and output tensor kinds
        let input_arg = self.inputs.first().unwrap();
        let output_arg = self.outputs.first().unwrap();

        let input_kind = match &input_arg.ty {
            ArgType::Tensor(t) => TensorKind::from(t.dtype),
            _ => panic!("Expected tensor input"),
        };

        let output_kind = match &output_arg.ty {
            ArgType::Tensor(t) => TensorKind::from(t.dtype),
            _ => panic!("Expected tensor output"),
        };

        let output_dtype = output_arg.ty.elem_type();
        let output_dtype_tokens = output_dtype.to_tokens();

        let input_dtype = input_arg.ty.elem_type();

        // Build the `one_hot_fill` call as a trailing expression (no
        // `let #output = ...;`). Wrapping the prelude + expression inside
        // a single block scopes the `__onehot_*` temporaries so multiple
        // OneHot nodes in the same forward() don't collide.
        let body: TokenStream = match (input_kind, output_kind) {
            (TensorKind::Int, TensorKind::Int) | (TensorKind::Float, TensorKind::Float) => {
                if input_dtype == output_dtype {
                    quote! {
                        #input.one_hot_fill(#num_classes, #on_value, #off_value, #axis)
                    }
                } else {
                    quote! {
                        #input.one_hot_fill(#num_classes, #on_value, #off_value, #axis).cast(#output_dtype_tokens)
                    }
                }
            }
            (TensorKind::Int, TensorKind::Float) => {
                quote! {
                    #input.one_hot_fill(#num_classes, #on_value, #off_value, #axis).float().cast(#output_dtype_tokens)
                }
            }
            (TensorKind::Float, TensorKind::Int) => {
                quote! {
                    #input.one_hot_fill(#num_classes, #on_value, #off_value, #axis).int().cast(#output_dtype_tokens)
                }
            }
            (TensorKind::Int, TensorKind::Bool) | (TensorKind::Float, TensorKind::Bool) => {
                quote! {
                    #input.one_hot_fill(#num_classes, #on_value, #off_value, #axis).bool()
                }
            }
            (TensorKind::Bool, _) => panic!("Input should be numeric"),
        };

        if prelude.is_empty() {
            // Preserve the legacy flat form for the common static case so
            // snapshots stay readable and we don't introduce a redundant
            // block wrapper.
            quote! {
                let #output = #body;
            }
        } else {
            quote! {
                let #output = {
                    #prelude
                    #body
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::{BoolStore, DType};
    use insta::assert_snapshot;
    use onnx_ir::one_hot::{OneHotConfig, OneHotDepthInput, OneHotNodeBuilder, OneHotValuesInput};

    #[test]
    fn test_one_hot() {
        let config = OneHotConfig::new(
            OneHotDepthInput::Static(10),
            OneHotValuesInput::Static([0.0, 1.0]),
            -1,
        );
        let node = OneHotNodeBuilder::new("onehot1")
            .input_tensor("indices", 1, DType::I32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, indices: Tensor<B, 1, Int>) -> Tensor<B, 2> {
            let output = indices
                .one_hot_fill(10usize, 1f32, 0f32, -1i64)
                .float()
                .cast(burn::tensor::DType::F32);
            output
        }
        ");
    }

    #[test]
    fn test_one_hot_int_to_int() {
        let config = OneHotConfig::new(
            OneHotDepthInput::Static(5),
            OneHotValuesInput::Static([0.0, 1.0]),
            -1,
        );
        let node = OneHotNodeBuilder::new("onehot2")
            .input_tensor("indices", 1, DType::I32)
            .output_tensor("output", 2, DType::I32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, indices: Tensor<B, 1, Int>) -> Tensor<B, 2, Int> {
            let output = indices.one_hot_fill(5usize, 1f32, 0f32, -1i64);
            output
        }
        ");
    }

    #[test]
    fn test_one_hot_float_to_float() {
        let config = OneHotConfig::new(
            OneHotDepthInput::Static(5),
            OneHotValuesInput::Static([0.0, 1.0]),
            0,
        );
        let node = OneHotNodeBuilder::new("onehot3")
            .input_tensor("indices", 1, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, indices: Tensor<B, 1>) -> Tensor<B, 2> {
            let output = indices.one_hot_fill(5usize, 1f32, 0f32, 0i64);
            output
        }
        ");
    }

    #[test]
    fn test_one_hot_float_to_int() {
        let config = OneHotConfig::new(
            OneHotDepthInput::Static(5),
            OneHotValuesInput::Static([0.0, 1.0]),
            0,
        );
        let node = OneHotNodeBuilder::new("onehot4")
            .input_tensor("indices", 1, DType::F32)
            .output_tensor("output", 2, DType::I32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, indices: Tensor<B, 1>) -> Tensor<B, 2, Int> {
            let output = indices
                .one_hot_fill(5usize, 1f32, 0f32, 0i64)
                .int()
                .cast(burn::tensor::DType::I32);
            output
        }
        ");
    }

    #[test]
    fn test_one_hot_int_to_bool() {
        let config = OneHotConfig::new(
            OneHotDepthInput::Static(5),
            OneHotValuesInput::Static([0.0, 1.0]),
            -1,
        );
        let node = OneHotNodeBuilder::new("onehot5")
            .input_tensor("indices", 1, DType::I32)
            .output_tensor("output", 2, DType::Bool(BoolStore::Native))
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, indices: Tensor<B, 1, Int>) -> Tensor<B, 2, Bool> {
            let output = indices.one_hot_fill(5usize, 1f32, 0f32, -1i64).bool();
            output
        }
        ");
    }

    #[test]
    fn test_one_hot_runtime_depth_and_values() {
        // Opset 11+ gives OneHot all three inputs as tensors, which lands
        // here as Runtime depth and Runtime values.
        //
        // Depth dtype here intentionally mirrors the real ONNX backend
        // test `test_onehot_negative_indices`, which declares depth as
        // ONNX FLOAT (type 1) — the OneHot spec's T2 constraint allows any
        // numeric scalar, not just integers, and this path must accept
        // whatever the upstream model uses. See
        // `test_one_hot_runtime_depth_int64` below for the i64 variant.
        let config = OneHotConfig::new(
            OneHotDepthInput::Runtime(onnx_ir::ir::RuntimeInputRef::new("depth".to_string(), 1)),
            OneHotValuesInput::Runtime(onnx_ir::ir::RuntimeInputRef::new("values".to_string(), 2)),
            -1,
        );
        let node = OneHotNodeBuilder::new("onehot_rt")
            .input_tensor("indices", 1, DType::I64)
            .input_scalar_tensor("depth", DType::F32)
            .input_tensor("values", 1, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            indices: Tensor<B, 1, Int>,
            depth: Tensor<B, 1>,
            values: Tensor<B, 1>,
        ) -> Tensor<B, 2> {
            let output = {
                let __onehot_depth: usize = {
                    let __data = depth.to_data().convert::<i64>();
                    __data.as_slice::<i64>().unwrap()[0] as usize
                };
                let (__onehot_off, __onehot_on): (f32, f32) = {
                    let __data = values.to_data().convert::<f32>();
                    let __slice = __data.as_slice::<f32>().unwrap();
                    (__slice[0], __slice[1])
                };
                indices
                    .one_hot_fill(__onehot_depth, __onehot_on, __onehot_off, -1i64)
                    .float()
                    .cast(burn::tensor::DType::F32)
            };
            output
        }
        ");
    }

    #[test]
    fn test_one_hot_runtime_depth_int64() {
        // Companion to test_one_hot_runtime_depth_and_values: some ONNX
        // models (and the spec's T2 constraint allows it) pass depth as
        // an i64 tensor rather than f32. The generated prelude should
        // produce identical code thanks to `to_data().convert::<i64>()`
        // being dtype-agnostic.
        let config = OneHotConfig::new(
            OneHotDepthInput::Runtime(onnx_ir::ir::RuntimeInputRef::new("depth".to_string(), 1)),
            OneHotValuesInput::Static([0.0, 1.0]),
            -1,
        );
        let node = OneHotNodeBuilder::new("onehot_rt_i64")
            .input_tensor("indices", 1, DType::I64)
            .input_scalar_tensor("depth", DType::I64)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            indices: Tensor<B, 1, Int>,
            depth: Tensor<B, 1, Int>,
        ) -> Tensor<B, 2> {
            let output = {
                let __onehot_depth: usize = {
                    let __data = depth.to_data().convert::<i64>();
                    __data.as_slice::<i64>().unwrap()[0] as usize
                };
                indices
                    .one_hot_fill(__onehot_depth, 1f32, 0f32, -1i64)
                    .float()
                    .cast(burn::tensor::DType::F32)
            };
            output
        }
        ");
    }

    #[test]
    fn test_one_hot_float_to_bool() {
        let config = OneHotConfig::new(
            OneHotDepthInput::Static(5),
            OneHotValuesInput::Static([0.0, 1.0]),
            0,
        );
        let node = OneHotNodeBuilder::new("onehot6")
            .input_tensor("indices", 1, DType::F32)
            .output_tensor("output", 2, DType::Bool(BoolStore::Native))
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, indices: Tensor<B, 1>) -> Tensor<B, 2, Bool> {
            let output = indices.one_hot_fill(5usize, 1f32, 0f32, 0i64).bool();
            output
        }
        ");
    }
}
