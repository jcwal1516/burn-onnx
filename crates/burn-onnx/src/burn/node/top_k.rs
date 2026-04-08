use super::prelude::*;

impl NodeCodegen for onnx_ir::topk::TopKNode {
    fn inputs(&self) -> &[Argument] {
        // Filter inputs only dynamic and constant
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        // TopK has 2 outputs: values and indices
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        // TopK has 2 outputs
        let values_output = arg_to_ident(&self.outputs[0]);
        let indices_output = arg_to_ident(&self.outputs[1]);

        let axis = self.config.axis.to_tokens();

        // Runtime `k` can reach this path in three shapes: a native scalar
        // (if an earlier pass has already scalarized it), a rank-0 scalar
        // tensor, or ONNX opset-10+'s rank-1 single-element tensor. Lower
        // whichever form to a `usize` local in a prelude block so (1) the
        // main topk_with_indices call stays readable and (2) multiple
        // TopK nodes in the same forward() can't collide on `__topk_k`.
        let (prelude, k) = match &self.config.k {
            onnx_ir::topk::TopKInput::Static(k_value) => (TokenStream::new(), k_value.to_tokens()),
            onnx_ir::topk::TopKInput::Runtime(r) => {
                let arg = &self.inputs[r.input_index];
                let prelude = match &arg.ty {
                    ArgType::ScalarNative(_) => {
                        let ident = arg_to_ident(arg);
                        quote! { let __topk_k: usize = #ident as usize; }
                    }
                    ArgType::ScalarTensor(_) | ArgType::Tensor(_) => {
                        let tensor = scope.arg(arg);
                        quote! {
                            let __topk_k: usize = {
                                let __data = #tensor.to_data().convert::<i64>();
                                __data.as_slice::<i64>().unwrap()[0] as usize
                            };
                        }
                    }
                    other => panic!("TopK k must be a scalar or rank-1 tensor, got {other:?}"),
                };
                (prelude, quote! { __topk_k })
            }
        };

        let input = scope.arg(self.inputs.first().unwrap());

        if prelude.is_empty() {
            quote! {
                let (#values_output, #indices_output) = #input.topk_with_indices(#k, #axis);
            }
        } else {
            quote! {
                let (#values_output, #indices_output) = {
                    #prelude
                    #input.topk_with_indices(#k, #axis)
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::topk::{TopKConfig, TopKInput, TopKNodeBuilder};

    #[test]
    fn test_top_k() {
        let config = TopKConfig::new(1, TopKInput::Static(5));
        let node = TopKNodeBuilder::new("topk1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("values", 2, DType::F32)
            .output_tensor("indices", 2, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2, Int>) {
            let (values, indices) = input.topk_with_indices(5, 1);
            (values, indices)
        }
        ");
    }

    #[test]
    fn test_top_k_runtime_k() {
        // Opset 10+ passes k as a runtime 1D single-element tensor.
        let config = TopKConfig::new(
            1,
            TopKInput::Runtime(onnx_ir::ir::RuntimeInputRef::new("k".to_string(), 1)),
        );
        let node = TopKNodeBuilder::new("topk_rt")
            .input_tensor("input", 2, DType::F32)
            .input_tensor("k", 1, DType::I64)
            .output_tensor("values", 2, DType::F32)
            .output_tensor("indices", 2, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<B, 2>,
            k: Tensor<B, 1, Int>,
        ) -> (Tensor<B, 2>, Tensor<B, 2, Int>) {
            let (values, indices) = {
                let __topk_k: usize = {
                    let __data = k.to_data().convert::<i64>();
                    __data.as_slice::<i64>().unwrap()[0] as usize
                };
                input.topk_with_indices(__topk_k, 1)
            };
            (values, indices)
        }
        ");
    }
}
