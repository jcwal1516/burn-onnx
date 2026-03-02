use super::prelude::*;

impl NodeCodegen for onnx_ir::node::argmax::ArgMaxNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        // NOTE: select_last_index=1 is not supported (will panic during conversion)
        let axis = self.config.axis.to_tokens();
        let input_arg = self.inputs.first().unwrap();
        let output_arg = self.outputs.first().unwrap();

        let input = scope.arg(input_arg);
        let output = arg_to_ident(output_arg);

        // argmax uses the backend's default int element type (I32 on GPU, I64 on NdArray).
        // Cast to the ONNX-specified dtype to avoid DTypeMismatch on GPU backends.
        match &output_arg.ty {
            onnx_ir::ir::ArgType::Tensor(tensor) => {
                let output_dtype = tensor.dtype.to_tokens();
                if self.config.keepdims {
                    quote! {
                        let #output = #input.argmax(#axis).cast(#output_dtype);
                    }
                } else {
                    let output_rank = tensor.rank;
                    quote! {
                        let argmax_result = #input.argmax(#axis);
                        let #output = argmax_result.squeeze_dim::<#output_rank>(#axis).cast(#output_dtype);
                    }
                }
            }
            onnx_ir::ir::ArgType::ScalarTensor(dtype) => {
                let output_dtype = dtype.to_tokens();
                quote! {
                    let #output = #input.argmax(#axis).reshape([1]).cast(#output_dtype);
                }
            }
            onnx_ir::ir::ArgType::ScalarNative(_) => {
                // Extracted to native scalar, no cast needed
                quote! {
                    let argmax_result = #input.argmax(#axis);
                    let #output = argmax_result.into_scalar().elem::<i64>();
                }
            }
            _ => panic!("ArgMax output must be Tensor or Scalar"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::argmax::{ArgMaxConfig, ArgMaxNodeBuilder};

    #[test]
    fn test_argmax_keepdims() {
        let config = ArgMaxConfig::new(1, true);
        let node = ArgMaxNodeBuilder::new("argmax1")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3, Int> {
            let output = input.argmax(1).cast(burn::tensor::DType::I64);
            output
        }
        ");
    }

    #[test]
    fn test_argmax_no_keepdims() {
        let config = ArgMaxConfig::new(2, false);
        let node = ArgMaxNodeBuilder::new("argmax2")
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 3, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 3, Int> {
            let argmax_result = input.argmax(2);
            let output = argmax_result.squeeze_dim::<3usize>(2).cast(burn::tensor::DType::I64);
            output
        }
        ");
    }

    #[test]
    fn test_argmax_scalar_tensor_output() {
        let config = ArgMaxConfig::new(0, false);
        let node = ArgMaxNodeBuilder::new("argmax3")
            .input_tensor("input", 1, DType::F32)
            .output_scalar_tensor("output", DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 1>) -> Tensor<B, 1, Int> {
            let output = input.argmax(0).reshape([1]).cast(burn::tensor::DType::I64);
            output
        }
        ");
    }

    #[test]
    fn test_argmax_scalar_native_output() {
        let config = ArgMaxConfig::new(0, false);
        let node = ArgMaxNodeBuilder::new("argmax4")
            .input_tensor("input", 1, DType::F32)
            .output_scalar("output", DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 1>) -> i64 {
            let argmax_result = input.argmax(0);
            let output = argmax_result.into_scalar().elem::<i64>();
            output
        }
        ");
    }
}
