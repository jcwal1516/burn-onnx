use super::prelude::*;
use crate::burn::argument_helpers::{elem_cast_tokens, tensor_type_tokens};
use onnx_ir::node::einsum::ParsedEinsum;

/// Compute a product of dimension expressions, or `1usize` when empty.
fn dims_product(dims: &[proc_macro2::TokenStream]) -> proc_macro2::TokenStream {
    if dims.is_empty() {
        quote! { 1usize }
    } else {
        let mut result = dims[0].clone();
        for d in &dims[1..] {
            result = quote! { #result * #d };
        }
        result
    }
}

fn compile_error_tokens(message: impl Into<String>) -> TokenStream {
    let message = message.into();
    quote! {
        compile_error!(#message);
    }
}

fn find_axis_positions(order: &[char], labels: &[char]) -> Option<Vec<usize>> {
    order
        .iter()
        .map(|axis| labels.iter().position(|current| current == axis))
        .collect()
}

fn scalar_native_to_tensor(expr: TokenStream, dtype: DType) -> TokenStream {
    let dtype_tokens = dtype.to_tokens();

    // Promote through a wide host literal and let Burn cast to the requested dtype.
    if matches!(dtype, DType::F16 | DType::BF16) {
        quote! {
            Tensor::<B, 1>::from_data(
                burn::tensor::TensorData::from([(#expr).to_f64()]),
                (&*self.device, #dtype_tokens)
            )
        }
    } else if matches!(dtype, DType::F32) {
        quote! {
            Tensor::<B, 1>::from_data(
                burn::tensor::TensorData::from([f64::from(#expr)]),
                (&*self.device, #dtype_tokens)
            )
        }
    } else if matches!(dtype, DType::F64) {
        quote! {
            Tensor::<B, 1>::from_data(
                burn::tensor::TensorData::from([#expr]),
                (&*self.device, #dtype_tokens)
            )
        }
    } else if dtype.is_int() || dtype.is_uint() {
        quote! {
            Tensor::<B, 1, burn::tensor::Int>::from_data(
                burn::tensor::TensorData::from([#expr as i64]),
                (&*self.device, #dtype_tokens)
            )
        }
    } else if dtype.is_bool() {
        quote! {
            Tensor::<B, 1, burn::tensor::Bool>::from_data(
                burn::tensor::TensorData::from([#expr]),
                (&*self.device, #dtype_tokens)
            )
        }
    } else {
        compile_error_tokens(format!("Einsum does not support scalar dtype {:?}", dtype))
    }
}

fn einsum_operand_expr(
    node_name: &str,
    equation: &str,
    side: &str,
    labels: &[char],
    arg: &Argument,
    expr: TokenStream,
) -> Result<TokenStream, TokenStream> {
    match &arg.ty {
        ArgType::Tensor(_) => Ok(expr),
        ArgType::ScalarTensor(_) if labels.is_empty() => Ok(expr),
        ArgType::ScalarNative(dtype) if labels.is_empty() => {
            Ok(scalar_native_to_tensor(expr, *dtype))
        }
        ArgType::ScalarTensor(_) | ArgType::ScalarNative(_) => Err(compile_error_tokens(format!(
            "Einsum node '{}' uses scalar {} input for non-empty term '{}' in '{}'",
            node_name,
            side,
            labels.iter().collect::<String>(),
            equation
        ))),
        _ => Err(compile_error_tokens(format!(
            "Einsum node '{}' requires tensor-compatible {} input, got {:?}",
            node_name, side, arg.ty
        ))),
    }
}

impl NodeCodegen for onnx_ir::node::einsum::EinsumNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let [lhs_arg, rhs_arg] = self.inputs.as_slice() else {
            return compile_error_tokens(format!(
                "Einsum node '{}' expects exactly 2 inputs, got {}",
                self.name,
                self.inputs.len()
            ));
        };
        let [output_arg] = self.outputs.as_slice() else {
            return compile_error_tokens(format!(
                "Einsum node '{}' expects exactly 1 output, got {}",
                self.name,
                self.outputs.len()
            ));
        };

        let lhs = scope.arg(lhs_arg);
        let rhs = scope.arg(rhs_arg);
        let output = arg_to_ident(output_arg);

        let parsed = match ParsedEinsum::parse(&self.config.equation) {
            Ok(parsed) => parsed,
            Err(error) => {
                return compile_error_tokens(format!(
                    "Einsum node '{}' has an invalid equation '{}': {}",
                    self.name, self.config.equation, error
                ));
            }
        };

        let output_rank = parsed.output.len();
        let result_rank = if output_rank == 0 { 1 } else { output_rank };

        let lhs = match einsum_operand_expr(
            &self.name,
            &self.config.equation,
            "lhs",
            &parsed.lhs,
            lhs_arg,
            lhs,
        ) {
            Ok(lhs) => lhs,
            Err(error) => return error,
        };
        let rhs = match einsum_operand_expr(
            &self.name,
            &self.config.equation,
            "rhs",
            &parsed.rhs,
            rhs_arg,
            rhs,
        ) {
            Ok(rhs) => rhs,
            Err(error) => return error,
        };

        let batch = parsed.batch_axes();
        let contract = parsed.contraction_axes();
        let free_lhs = parsed.free_lhs_axes();
        let free_rhs = parsed.free_rhs_axes();

        let lhs_perm_order: Vec<char> = batch
            .iter()
            .chain(free_lhs.iter())
            .chain(contract.iter())
            .copied()
            .collect();
        let lhs_perm = match find_axis_positions(&lhs_perm_order, &parsed.lhs) {
            Some(perm) => perm,
            None => {
                return compile_error_tokens(format!(
                    "Einsum node '{}' references an unknown lhs axis in '{}'",
                    self.name, self.config.equation
                ));
            }
        };

        let rhs_perm_order: Vec<char> = batch
            .iter()
            .chain(contract.iter())
            .chain(free_rhs.iter())
            .copied()
            .collect();
        let rhs_perm = match find_axis_positions(&rhs_perm_order, &parsed.rhs) {
            Some(perm) => perm,
            None => {
                return compile_error_tokens(format!(
                    "Einsum node '{}' references an unknown rhs axis in '{}'",
                    self.name, self.config.equation
                ));
            }
        };

        let matmul_output_order: Vec<char> = batch
            .iter()
            .chain(free_lhs.iter())
            .chain(free_rhs.iter())
            .copied()
            .collect();
        let output_perm = match find_axis_positions(&parsed.output, &matmul_output_order) {
            Some(perm) => perm,
            None => {
                return compile_error_tokens(format!(
                    "Einsum node '{}' references an unknown output axis in '{}'",
                    self.name, self.config.equation
                ));
            }
        };

        let n_batch = batch.len();
        let n_free_lhs = free_lhs.len();
        let n_contract = contract.len();
        let n_free_rhs = free_rhs.len();

        let lhs_perm_rank = parsed.lhs.len();
        let rhs_perm_rank = parsed.rhs.len();

        let lhs_is_identity = lhs_perm.iter().enumerate().all(|(i, &v)| i == v);
        let rhs_is_identity = rhs_perm.iter().enumerate().all(|(i, &v)| i == v);
        let output_is_identity = output_perm.iter().enumerate().all(|(i, &v)| i == v);
        let needs_reshape = n_batch + n_free_lhs != 1
            || n_batch + n_free_rhs != 1
            || n_batch + n_contract != 1
            || n_batch > 1;

        // Fast path for plain matrix multiplication.
        if lhs_is_identity
            && rhs_is_identity
            && output_is_identity
            && !needs_reshape
            && n_batch == 0
        {
            return quote! {
                let #output = #lhs.matmul(#rhs);
            };
        }

        // Fast path for the common rank-3 batched case.
        if lhs_is_identity
            && rhs_is_identity
            && output_is_identity
            && n_batch == 1
            && n_free_lhs == 1
            && n_contract == 1
            && n_free_rhs == 1
        {
            return quote! {
                let #output = #lhs.matmul(#rhs);
            };
        }

        let lhs_perm_expr = if lhs_is_identity {
            lhs.clone()
        } else {
            let perm_dims: Vec<proc_macro2::TokenStream> =
                lhs_perm.iter().map(|&d| quote! { #d }).collect();
            quote! { #lhs.permute([#(#perm_dims),*]) }
        };

        let rhs_perm_expr = if rhs_is_identity {
            rhs.clone()
        } else {
            let perm_dims: Vec<proc_macro2::TokenStream> =
                rhs_perm.iter().map(|&d| quote! { #d }).collect();
            quote! { #rhs.permute([#(#perm_dims),*]) }
        };

        // Flatten grouped axes to a batched `[B, M, K] x [B, K, N]` matmul.
        let lhs_batch_dims: Vec<proc_macro2::TokenStream> = (0..n_batch)
            .map(|i| quote! { einsum_lhs_shape[#i] })
            .collect();
        let lhs_free_dims: Vec<proc_macro2::TokenStream> = (n_batch..n_batch + n_free_lhs)
            .map(|i| quote! { einsum_lhs_shape[#i] })
            .collect();
        let lhs_contract_dims: Vec<proc_macro2::TokenStream> = (n_batch + n_free_lhs
            ..lhs_perm_rank)
            .map(|i| quote! { einsum_lhs_shape[#i] })
            .collect();

        let rhs_free_dims: Vec<proc_macro2::TokenStream> = (n_batch + n_contract..rhs_perm_rank)
            .map(|i| quote! { einsum_rhs_shape[#i] })
            .collect();

        let batch_product = dims_product(&lhs_batch_dims);
        let m_product = dims_product(&lhs_free_dims);
        let k_product = dims_product(&lhs_contract_dims);
        let n_product = dims_product(&rhs_free_dims);

        let batch_shape_dims: Vec<proc_macro2::TokenStream> = (0..n_batch)
            .map(|i| quote! { einsum_lhs_shape[#i] })
            .collect();
        let free_lhs_shape_dims: Vec<proc_macro2::TokenStream> = (n_batch..n_batch + n_free_lhs)
            .map(|i| quote! { einsum_lhs_shape[#i] })
            .collect();
        let free_rhs_shape_dims: Vec<proc_macro2::TokenStream> = (n_batch + n_contract
            ..rhs_perm_rank)
            .map(|i| quote! { einsum_rhs_shape[#i] })
            .collect();

        let out_shape_dims: Vec<proc_macro2::TokenStream> = batch_shape_dims
            .iter()
            .chain(free_lhs_shape_dims.iter())
            .chain(free_rhs_shape_dims.iter())
            .cloned()
            .collect();
        let needs_lhs_shape = !lhs_batch_dims.is_empty()
            || !lhs_free_dims.is_empty()
            || !lhs_contract_dims.is_empty()
            || !batch_shape_dims.is_empty()
            || !free_lhs_shape_dims.is_empty();
        let needs_rhs_shape = !rhs_free_dims.is_empty() || !free_rhs_shape_dims.is_empty();

        let lhs_3d_ty = tensor_type_tokens(3, &lhs_arg.ty.elem_type());
        let rhs_3d_ty = tensor_type_tokens(3, &rhs_arg.ty.elem_type());
        let result_ty = tensor_type_tokens(result_rank, &output_arg.ty.elem_type());

        let result_expr = if output_is_identity {
            quote! { einsum_result }
        } else {
            let out_perm_dims: Vec<proc_macro2::TokenStream> =
                output_perm.iter().map(|&d| quote! { #d }).collect();
            quote! { einsum_result.permute([#(#out_perm_dims),*]) }
        };
        let final_output_expr = match &output_arg.ty {
            ArgType::Tensor(tensor) => {
                if tensor.rank != output_rank {
                    return compile_error_tokens(format!(
                        "Einsum node '{}' expected tensor output rank {} but got {}",
                        self.name, output_rank, tensor.rank
                    ));
                }
                result_expr
            }
            ArgType::ScalarTensor(_) => {
                if output_rank != 0 {
                    return compile_error_tokens(format!(
                        "Einsum node '{}' expected scalar tensor output for rank-0 equation '{}'",
                        self.name, self.config.equation
                    ));
                }
                result_expr
            }
            ArgType::ScalarNative(dtype) => {
                if output_rank != 0 {
                    return compile_error_tokens(format!(
                        "Einsum node '{}' expected scalar output for rank-0 equation '{}'",
                        self.name, self.config.equation
                    ));
                }
                let cast = elem_cast_tokens(dtype);
                quote! { (#result_expr).into_scalar()#cast }
            }
            _ => {
                return compile_error_tokens(format!(
                    "Einsum node '{}' does not support output type {:?}",
                    self.name, output_arg.ty
                ));
            }
        };
        let lhs_shape_binding = if needs_lhs_shape {
            quote! { let einsum_lhs_shape = einsum_lhs.dims(); }
        } else {
            quote! {}
        };
        let rhs_shape_binding = if needs_rhs_shape {
            quote! { let einsum_rhs_shape = einsum_rhs.dims(); }
        } else {
            quote! {}
        };
        let result_reshape_expr = if output_rank == 0 {
            quote! { .reshape([1usize]) }
        } else {
            quote! { .reshape([#(#out_shape_dims),*]) }
        };

        quote! {
            let #output = {
                let einsum_lhs = #lhs_perm_expr;
                let einsum_rhs = #rhs_perm_expr;
                #lhs_shape_binding
                #rhs_shape_binding
                let einsum_lhs_3d: #lhs_3d_ty =
                    einsum_lhs.reshape([#batch_product, #m_product, #k_product]);
                let einsum_rhs_3d: #rhs_3d_ty =
                    einsum_rhs.reshape([#batch_product, #k_product, #n_product]);
                let einsum_result: #result_ty = einsum_lhs_3d
                    .matmul(einsum_rhs_3d)
                    #result_reshape_expr;
                #final_output_expr
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::einsum::{EinsumConfig, EinsumNodeBuilder};

    #[test]
    fn test_einsum_simple_matmul() {
        let node = EinsumNodeBuilder::new("einsum1")
            .input_tensor("lhs", 2, DType::F32)
            .input_tensor("rhs", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(EinsumConfig {
                equation: "ij,jk->ik".to_string(),
            })
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, lhs: Tensor<B, 2>, rhs: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = lhs.matmul(rhs);
            output
        }
        ");
    }

    #[test]
    fn test_einsum_batch_matmul() {
        let node = EinsumNodeBuilder::new("einsum2")
            .input_tensor("lhs", 3, DType::F32)
            .input_tensor("rhs", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(EinsumConfig {
                equation: "bij,bjk->bik".to_string(),
            })
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, lhs: Tensor<B, 3>, rhs: Tensor<B, 3>) -> Tensor<B, 3> {
            let output = lhs.matmul(rhs);
            output
        }
        ");
    }

    #[test]
    fn test_einsum_simple_matmul_int() {
        let node = EinsumNodeBuilder::new("einsum_int")
            .input_tensor("lhs", 2, DType::I32)
            .input_tensor("rhs", 2, DType::I32)
            .output_tensor("output", 2, DType::I32)
            .config(EinsumConfig {
                equation: "ij,jk->ik".to_string(),
            })
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            lhs: Tensor<B, 2, Int>,
            rhs: Tensor<B, 2, Int>,
        ) -> Tensor<B, 2, Int> {
            let output = lhs.matmul(rhs);
            output
        }
        ");
    }

    #[test]
    fn test_einsum_sam_pattern() {
        let node = EinsumNodeBuilder::new("einsum3")
            .input_tensor("r_q", 4, DType::F32)
            .input_tensor("r_h", 3, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(EinsumConfig {
                equation: "bhwc,hkc->bhwk".to_string(),
            })
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, r_q: Tensor<B, 4>, r_h: Tensor<B, 3>) -> Tensor<B, 4> {
            let output = {
                let einsum_lhs = r_q.permute([1usize, 0usize, 2usize, 3usize]);
                let einsum_rhs = r_h.permute([0usize, 2usize, 1usize]);
                let einsum_lhs_shape = einsum_lhs.dims();
                let einsum_rhs_shape = einsum_rhs.dims();
                let einsum_lhs_3d: Tensor<B, 3> = einsum_lhs
                    .reshape([
                        einsum_lhs_shape[0usize],
                        einsum_lhs_shape[1usize] * einsum_lhs_shape[2usize],
                        einsum_lhs_shape[3usize],
                    ]);
                let einsum_rhs_3d: Tensor<B, 3> = einsum_rhs
                    .reshape([
                        einsum_lhs_shape[0usize],
                        einsum_lhs_shape[3usize],
                        einsum_rhs_shape[2usize],
                    ]);
                let einsum_result: Tensor<B, 4> = einsum_lhs_3d
                    .matmul(einsum_rhs_3d)
                    .reshape([
                        einsum_lhs_shape[0usize],
                        einsum_lhs_shape[1usize],
                        einsum_lhs_shape[2usize],
                        einsum_rhs_shape[2usize],
                    ]);
                einsum_result.permute([1usize, 0usize, 2usize, 3usize])
            };
            output
        }
        "#);
    }

    #[test]
    fn test_einsum_outer_product() {
        let node = EinsumNodeBuilder::new("einsum4")
            .input_tensor("a", 1, DType::F32)
            .input_tensor("b", 1, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(EinsumConfig {
                equation: "i,j->ij".to_string(),
            })
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, a: Tensor<B, 1>, b: Tensor<B, 1>) -> Tensor<B, 2> {
            let output = {
                let einsum_lhs = a;
                let einsum_rhs = b;
                let einsum_lhs_shape = einsum_lhs.dims();
                let einsum_rhs_shape = einsum_rhs.dims();
                let einsum_lhs_3d: Tensor<B, 3> = einsum_lhs
                    .reshape([1usize, einsum_lhs_shape[0usize], 1usize]);
                let einsum_rhs_3d: Tensor<B, 3> = einsum_rhs
                    .reshape([1usize, 1usize, einsum_rhs_shape[0usize]]);
                let einsum_result: Tensor<B, 2> = einsum_lhs_3d
                    .matmul(einsum_rhs_3d)
                    .reshape([einsum_lhs_shape[0usize], einsum_rhs_shape[0usize]]);
                einsum_result
            };
            output
        }
        "#);
    }

    #[test]
    fn test_einsum_outer_product_int_preserves_tensor_kind() {
        let node = EinsumNodeBuilder::new("einsum_outer_int")
            .input_tensor("a", 1, DType::I32)
            .input_tensor("b", 1, DType::I32)
            .output_tensor("output", 2, DType::I32)
            .config(EinsumConfig {
                equation: "i,j->ij".to_string(),
            })
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, a: Tensor<B, 1, Int>, b: Tensor<B, 1, Int>) -> Tensor<B, 2, Int> {
            let output = {
                let einsum_lhs = a;
                let einsum_rhs = b;
                let einsum_lhs_shape = einsum_lhs.dims();
                let einsum_rhs_shape = einsum_rhs.dims();
                let einsum_lhs_3d: Tensor<B, 3, Int> = einsum_lhs
                    .reshape([1usize, einsum_lhs_shape[0usize], 1usize]);
                let einsum_rhs_3d: Tensor<B, 3, Int> = einsum_rhs
                    .reshape([1usize, 1usize, einsum_rhs_shape[0usize]]);
                let einsum_result: Tensor<B, 2, Int> = einsum_lhs_3d
                    .matmul(einsum_rhs_3d)
                    .reshape([einsum_lhs_shape[0usize], einsum_rhs_shape[0usize]]);
                einsum_result
            };
            output
        }
        "#);
    }

    #[test]
    fn test_einsum_accepts_scalar_native_lhs() {
        let node = EinsumNodeBuilder::new("einsum_scalar_lhs")
            .input_scalar("scale", DType::F32)
            .input_tensor("rhs", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(EinsumConfig {
                equation: ",ij->ij".to_string(),
            })
            .build();
        let code = codegen_forward_default(&node);

        assert!(code.contains("scale: f32"));
        assert!(code.contains("from_data("));
        assert!(code.contains("TensorData::from([f64::from(scale)])"));
        assert!(code.contains("reshape([1usize, 1usize, 1usize])"));
        assert!(code.contains("einsum_rhs_shape[0usize]"));
        assert!(code.contains("einsum_rhs_shape[1usize]"));
    }

    #[test]
    fn test_einsum_accepts_float16_scalar_native_lhs() {
        let node = EinsumNodeBuilder::new("einsum_scalar_lhs_f16")
            .input_scalar("scale", DType::F16)
            .input_tensor("rhs", 2, DType::F16)
            .output_tensor("output", 2, DType::F16)
            .config(EinsumConfig {
                equation: ",ij->ij".to_string(),
            })
            .build();
        let code = codegen_forward_default(&node);

        assert!(code.contains("scale: half::f16"));
        assert!(code.contains("TensorData::from([(scale).to_f64()])"));
    }

    #[test]
    fn test_einsum_accepts_scalar_tensor_rhs() {
        let node = EinsumNodeBuilder::new("einsum_scalar_rhs")
            .input_tensor("lhs", 2, DType::F32)
            .input_scalar_tensor("scale", DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(EinsumConfig {
                equation: "ij,->ij".to_string(),
            })
            .build();
        let code = codegen_forward_default(&node);

        assert!(code.contains("scale: Tensor<B, 1>"));
        assert!(code.contains("let einsum_rhs = scale;"));
        assert!(code.contains("reshape([1usize, 1usize, 1usize])"));
        assert!(code.contains("einsum_lhs_shape[0usize]"));
        assert!(code.contains("einsum_lhs_shape[1usize]"));
    }

    #[test]
    fn test_einsum_scalar_scalar_can_return_scalar_tensor_output() {
        let node = EinsumNodeBuilder::new("einsum_scalar_scalar")
            .input_scalar("lhs", DType::F32)
            .input_scalar_tensor("rhs", DType::F32)
            .output_scalar_tensor("output", DType::F32)
            .config(EinsumConfig {
                equation: ",->".to_string(),
            })
            .build();
        let code = codegen_forward_default(&node);

        assert!(code.contains("-> Tensor<B, 1>"));
        assert!(code.contains("let einsum_lhs ="));
        assert!(code.contains("from_data("));
        assert!(code.contains("let einsum_rhs = rhs;"));
        assert!(code.contains("let einsum_result: Tensor<B, 1>"));
        assert!(code.contains(".reshape([1usize])"));
    }

    #[test]
    fn test_einsum_scalar_scalar_native_output() {
        let node = EinsumNodeBuilder::new("einsum_scalar_scalar_native")
            .input_scalar("lhs", DType::F32)
            .input_scalar("rhs", DType::F32)
            .output_scalar("output", DType::F32)
            .config(EinsumConfig {
                equation: ",->".to_string(),
            })
            .build();
        let code = codegen_forward_default(&node);

        assert!(code.contains("-> f32"));
        assert!(code.contains("TensorData::from([f64::from(lhs)])"));
        assert!(code.contains("TensorData::from([f64::from(rhs)])"));
        assert!(code.contains("let einsum_result: Tensor<B, 1>"));
        assert!(code.contains(".reshape([1usize])"));
        assert!(code.contains("(einsum_result).into_scalar().elem::<f32>()"));
    }

    #[test]
    fn test_einsum_decomposition_temporaries_are_scoped() {
        let node = EinsumNodeBuilder::new("einsum_scope")
            .input_tensor("lhs", 2, DType::F32)
            .input_tensor("einsum_rhs", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(EinsumConfig {
                equation: "ij,kj->ik".to_string(),
            })
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, lhs: Tensor<B, 2>, einsum_rhs: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = {
                let einsum_lhs = lhs;
                let einsum_rhs = einsum_rhs.permute([1usize, 0usize]);
                let einsum_lhs_shape = einsum_lhs.dims();
                let einsum_rhs_shape = einsum_rhs.dims();
                let einsum_lhs_3d: Tensor<B, 3> = einsum_lhs
                    .reshape([1usize, einsum_lhs_shape[0usize], einsum_lhs_shape[1usize]]);
                let einsum_rhs_3d: Tensor<B, 3> = einsum_rhs
                    .reshape([1usize, einsum_lhs_shape[1usize], einsum_rhs_shape[1usize]]);
                let einsum_result: Tensor<B, 2> = einsum_lhs_3d
                    .matmul(einsum_rhs_3d)
                    .reshape([einsum_lhs_shape[0usize], einsum_rhs_shape[1usize]]);
                einsum_result
            };
            output
        }
        "#);
    }

    #[test]
    fn test_einsum_sam_pattern_2() {
        let node = EinsumNodeBuilder::new("einsum5")
            .input_tensor("r_q", 4, DType::F32)
            .input_tensor("r_w", 3, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(EinsumConfig {
                equation: "bhwc,wkc->bhwk".to_string(),
            })
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, r_q: Tensor<B, 4>, r_w: Tensor<B, 3>) -> Tensor<B, 4> {
            let output = {
                let einsum_lhs = r_q.permute([2usize, 0usize, 1usize, 3usize]);
                let einsum_rhs = r_w.permute([0usize, 2usize, 1usize]);
                let einsum_lhs_shape = einsum_lhs.dims();
                let einsum_rhs_shape = einsum_rhs.dims();
                let einsum_lhs_3d: Tensor<B, 3> = einsum_lhs
                    .reshape([
                        einsum_lhs_shape[0usize],
                        einsum_lhs_shape[1usize] * einsum_lhs_shape[2usize],
                        einsum_lhs_shape[3usize],
                    ]);
                let einsum_rhs_3d: Tensor<B, 3> = einsum_rhs
                    .reshape([
                        einsum_lhs_shape[0usize],
                        einsum_lhs_shape[3usize],
                        einsum_rhs_shape[2usize],
                    ]);
                let einsum_result: Tensor<B, 4> = einsum_lhs_3d
                    .matmul(einsum_rhs_3d)
                    .reshape([
                        einsum_lhs_shape[0usize],
                        einsum_lhs_shape[1usize],
                        einsum_lhs_shape[2usize],
                        einsum_rhs_shape[2usize],
                    ]);
                einsum_result.permute([1usize, 2usize, 0usize, 3usize])
            };
            output
        }
        "#);
    }
}
