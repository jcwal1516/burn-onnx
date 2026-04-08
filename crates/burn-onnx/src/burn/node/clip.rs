use super::prelude::*;

/// Build a token stream that evaluates to an `f64` bound at runtime for a
/// single Clip `min`/`max` input.
///
/// Static bounds are inlined as literals. Runtime bounds come from one of
/// the node's inputs, which is either a native scalar (use directly) or a
/// scalar tensor (extract via `into_scalar().elem()`). Everything is
/// coerced to `f64` so the caller can feed both `min` and `max` into
/// Burn's `clamp(E, E)` without dtype mixing.
fn clip_bound_expr(
    bound: &Option<onnx_ir::node::clip::ClipInput>,
    inputs: &[Argument],
    scope: &mut ScopeAtPosition<'_>,
) -> Option<TokenStream> {
    match bound {
        None => None,
        Some(onnx_ir::node::clip::ClipInput::Static(v)) => {
            let v = *v;
            Some(quote! { #v })
        }
        Some(onnx_ir::node::clip::ClipInput::Runtime(r)) => {
            let arg = &inputs[r.input_index];
            match &arg.ty {
                ArgType::ScalarNative(_) => {
                    let ident = arg_to_ident(arg);
                    Some(quote! { (#ident as f64) })
                }
                ArgType::ScalarTensor(dtype) => {
                    let tensor = scope.arg(arg);
                    let native = on_device_to_native(quote! { #tensor }, dtype);
                    Some(quote! { (#native as f64) })
                }
                other => panic!(
                    "Clip min/max must be a scalar (ScalarNative or ScalarTensor), got {other:?}"
                ),
            }
        }
    }
}

impl NodeCodegen for onnx_ir::clip::ClipNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let output = arg_to_ident(self.outputs.first().unwrap());

        // Extract bound expressions first so the `match (min_expr, max_expr)`
        // below is a straight token-stream assembly rather than a nested
        // branch on the enum structure.
        let min_expr = clip_bound_expr(&self.config.min, &self.inputs, scope);
        let max_expr = clip_bound_expr(&self.config.max, &self.inputs, scope);
        let input = scope.arg(self.inputs.first().unwrap());

        match (min_expr, max_expr) {
            (Some(min), Some(max)) => quote! {
                let #output = {
                    let __clip_min = #min;
                    let __clip_max = #max;
                    #input.clamp(__clip_min, __clip_max)
                };
            },
            (Some(min), None) => quote! {
                let #output = {
                    let __clip_min = #min;
                    #input.clamp_min(__clip_min)
                };
            },
            (None, Some(max)) => quote! {
                let #output = {
                    let __clip_max = #max;
                    #input.clamp_max(__clip_max)
                };
            },
            (None, None) => panic!("Clip node must have at least one min or max value"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::clip::{ClipConfig, ClipNode, ClipNodeBuilder};
    use onnx_ir::node::clip::ClipInput;

    fn create_clip_node(name: &str, min: Option<f64>, max: Option<f64>) -> ClipNode {
        let config = ClipConfig {
            min: min.map(ClipInput::Static),
            max: max.map(ClipInput::Static),
        };

        ClipNodeBuilder::new(name)
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_clip_both_bounds() {
        let node = create_clip_node("clip1", Some(-1.0), Some(1.0));
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = {
                let __clip_min = -1f64;
                let __clip_max = 1f64;
                input.clamp(__clip_min, __clip_max)
            };
            output
        }
        ");
    }

    #[test]
    fn test_clip_min_only() {
        let node = create_clip_node("clip1", Some(0.0), None);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = {
                let __clip_min = 0f64;
                input.clamp_min(__clip_min)
            };
            output
        }
        ");
    }

    #[test]
    fn test_clip_max_only() {
        let node = create_clip_node("clip1", None, Some(10.0));
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = {
                let __clip_max = 10f64;
                input.clamp_max(__clip_max)
            };
            output
        }
        ");
    }

    #[test]
    fn test_clip_runtime_min_scalar_tensor() {
        // ONNX gives us Clip(x, min_tensor) where min_tensor arrives here
        // as `ArgType::ScalarTensor`, which onnx-ir models as a rank-1
        // tensor with shape [1] (see ArgType::rank() for ScalarTensor).
        // Burn's clamp_min takes a native scalar, so we extract it via
        // .into_scalar().elem::<T>() and cast to f64.
        let config = ClipConfig {
            min: Some(ClipInput::Runtime(onnx_ir::ir::RuntimeInputRef::new(
                "min_val".to_string(),
                1,
            ))),
            max: None,
        };
        let node = ClipNodeBuilder::new("clip1")
            .input_tensor("input", 2, DType::F32)
            .input_scalar_tensor("min_val", DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>, min_val: Tensor<B, 1>) -> Tensor<B, 2> {
            let output = {
                let __clip_min = (min_val.into_scalar().elem::<f32>() as f64);
                input.clamp_min(__clip_min)
            };
            output
        }
        ");
    }

    #[test]
    fn test_clip_runtime_both_scalar_tensors() {
        let config = ClipConfig {
            min: Some(ClipInput::Runtime(onnx_ir::ir::RuntimeInputRef::new(
                "min_val".to_string(),
                1,
            ))),
            max: Some(ClipInput::Runtime(onnx_ir::ir::RuntimeInputRef::new(
                "max_val".to_string(),
                2,
            ))),
        };
        let node = ClipNodeBuilder::new("clip1")
            .input_tensor("input", 2, DType::F32)
            .input_scalar_tensor("min_val", DType::F32)
            .input_scalar_tensor("max_val", DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<B, 2>,
            min_val: Tensor<B, 1>,
            max_val: Tensor<B, 1>,
        ) -> Tensor<B, 2> {
            let output = {
                let __clip_min = (min_val.into_scalar().elem::<f32>() as f64);
                let __clip_max = (max_val.into_scalar().elem::<f32>() as f64);
                input.clamp(__clip_min, __clip_max)
            };
            output
        }
        ");
    }

    #[test]
    fn test_clip_runtime_min_scalar_native() {
        let config = ClipConfig {
            min: Some(ClipInput::Runtime(onnx_ir::ir::RuntimeInputRef::new(
                "min_val".to_string(),
                1,
            ))),
            max: None,
        };
        let node = ClipNodeBuilder::new("clip1")
            .input_tensor("input", 2, DType::F32)
            .input_scalar("min_val", DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>, min_val: f32) -> Tensor<B, 2> {
            let output = {
                let __clip_min = (min_val as f64);
                input.clamp_min(__clip_min)
            };
            output
        }
        ");
    }
}
