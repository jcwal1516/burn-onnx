use super::prelude::*;

impl NodeCodegen for onnx_ir::node::det::DetNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input_arg = self.inputs.first().unwrap();
        let output_arg = self.outputs.first().unwrap();
        let input = scope.arg(input_arg);
        let output = arg_to_ident(output_arg);

        let input_rank = match &input_arg.ty {
            ArgType::Tensor(t) => t.rank,
            _ => panic!("Det input must be a tensor"),
        };

        match input_rank {
            2 => {
                // 2D non-batched: compute determinant using LU decomposition
                // det(A) = product_of_diagonal(U) * sign_of_permutation(P)
                // where PA = LU (PLU decomposition)
                quote! {
                    let #output = {
                        let (lu_mat, permutations) =
                            burn::tensor::linalg::lu_decomposition(#input);
                        let det_u =
                            burn::tensor::linalg::diag::<B, 2usize, 1usize, Float>(lu_mat)
                                .prod();
                        let n = permutations.dims()[0];
                        let perm_f = permutations.float();
                        let perm_col = perm_f.clone().reshape([n, 1]);
                        let perm_row = perm_f.reshape([1, n]);
                        let inv_count = (perm_row - perm_col)
                            .triu(1i64)
                            .lower_elem(0.0f32)
                            .int()
                            .sum();
                        let parity = inv_count.remainder_scalar(2i64);
                        let device = det_u.device();
                        let dtype = det_u.dtype();
                        let sign = Tensor::<B, 1, Float>::from_data_dtype(
                            [1.0f32, -1.0f32],
                            &device,
                            dtype,
                        )
                        .select(0, parity);
                        det_u * sign
                    };
                }
            }
            _ => {
                // Batched inputs: loop over all batch dimensions using LU decomposition
                // per-matrix. This is a fallback; a native batched det in Burn would be
                // more efficient.
                let batch_rank = input_rank - 2;
                quote! {
                    let #output = {
                        let shape = #input.shape();
                        let batch_size: usize = shape.dims[..#batch_rank].iter().product();
                        let m = shape.dims[#batch_rank];
                        let flat = #input.reshape([batch_size, m, m]);
                        let mut dets: Vec<Tensor<B, 1, Float>> = Vec::with_capacity(batch_size);
                        for i in 0..batch_size {
                            let matrix: Tensor<B, 2, Float> =
                                flat.clone().slice([i..(i + 1), 0..m, 0..m]).squeeze::<2>(0);
                            let (lu_mat, permutations) =
                                burn::tensor::linalg::lu_decomposition(matrix);
                            let det_u =
                                burn::tensor::linalg::diag::<B, 2usize, 1usize, Float>(lu_mat)
                                    .prod();
                            let n = permutations.dims()[0];
                            let perm_f = permutations.float();
                            let perm_col = perm_f.clone().reshape([n, 1]);
                            let perm_row = perm_f.reshape([1, n]);
                            let inv_count = (perm_row - perm_col)
                                .triu(1i64)
                                .lower_elem(0.0f32)
                                .int()
                                .sum();
                            let parity = inv_count.remainder_scalar(2i64);
                            let device = det_u.device();
                            let dtype = det_u.dtype();
                            let sign = Tensor::<B, 1, Float>::from_data_dtype(
                                [1.0f32, -1.0f32],
                                &device,
                                dtype,
                            )
                            .select(0, parity);
                            dets.push(det_u * sign);
                        }
                        let flat_out = Tensor::cat(dets, 0);
                        let batch_shape: Vec<usize> = shape.dims[..#batch_rank].to_vec();
                        flat_out.reshape(batch_shape)
                    };
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::det::DetNodeBuilder;

    #[test]
    fn test_det_2d_forward() {
        let node = DetNodeBuilder::new("det1")
            .input_tensor("input", 2, DType::F32)
            .output_scalar_tensor("output", DType::F32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 1> {
            let output = {
                let (lu_mat, permutations) = burn::tensor::linalg::lu_decomposition(input);
                let det_u = burn::tensor::linalg::diag::<B, 2usize, 1usize, Float>(lu_mat)
                    .prod();
                let n = permutations.dims()[0];
                let perm_f = permutations.float();
                let perm_col = perm_f.clone().reshape([n, 1]);
                let perm_row = perm_f.reshape([1, n]);
                let inv_count = (perm_row - perm_col).triu(1i64).lower_elem(0.0f32).int().sum();
                let parity = inv_count.remainder_scalar(2i64);
                let device = det_u.device();
                let dtype = det_u.dtype();
                let sign = Tensor::<
                    B,
                    1,
                    Float,
                >::from_data_dtype([1.0f32, -1.0f32], &device, dtype)
                    .select(0, parity);
                det_u * sign
            };
            output
        }
        ");
    }
}
