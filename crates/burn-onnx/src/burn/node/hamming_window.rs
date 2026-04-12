use super::prelude::*;

impl NodeCodegen for onnx_ir::node::hamming_window::HammingWindowNode {
    fn inputs(&self) -> &[Argument] {
        // HammingWindow has no runtime inputs; size is baked into config
        &[]
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, _scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let output = arg_to_ident(self.outputs.first().unwrap());
        let size = self.config.size;
        let periodic = self.config.periodic;
        let output_dtype = self.config.output_dtype.to_tokens();

        quote! {
            let #output = hamming_window::<B>(#size, #periodic, &self.device)
                .cast(#output_dtype);
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::tensor::signal::hamming_window");
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::hamming_window::{HammingWindowConfig, HammingWindowNodeBuilder};

    #[test]
    fn test_hamming_window_periodic() {
        let config = HammingWindowConfig {
            periodic: true,
            output_dtype: DType::F32,
            size: 10,
        };
        let node = HammingWindowNodeBuilder::new("hamming1")
            .output_tensor("output", 1, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self) -> Tensor<B, 1> {
            let output = hamming_window::<B>(10usize, true, &self.device)
                .cast(burn::tensor::DType::F32);
            output
        }
        ");
    }

    #[test]
    fn test_hamming_window_symmetric() {
        let config = HammingWindowConfig {
            periodic: false,
            output_dtype: DType::F64,
            size: 8,
        };
        let node = HammingWindowNodeBuilder::new("hamming1")
            .output_tensor("output", 1, DType::F64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self) -> Tensor<B, 1> {
            let output = hamming_window::<B>(8usize, false, &self.device)
                .cast(burn::tensor::DType::F64);
            output
        }
        ");
    }
}
