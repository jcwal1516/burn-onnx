use proc_macro2::TokenStream;
use quote::quote;

/// Prepend leading unsqueeze dimensions to `expr` so its rank matches `target_rank`.
/// Returns `expr` unchanged when `expr_rank >= target_rank`.
pub(crate) fn leading_broadcast(
    expr: TokenStream,
    expr_rank: usize,
    target_rank: usize,
) -> TokenStream {
    if expr_rank >= target_rank {
        return expr;
    }
    let num_dims = target_rank - expr_rank;
    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
    quote! { (#expr).unsqueeze_dims(&[#(#dims),*]) }
}

pub(crate) fn align_rhs_for_lhs_rank(
    rhs_expr: TokenStream,
    lhs_rank: usize,
    rhs_rank: usize,
    axis: Option<i64>,
) -> TokenStream {
    if lhs_rank <= rhs_rank {
        return rhs_expr;
    }

    if rhs_rank == 1 && lhs_rank > 1 {
        let axis = axis.unwrap_or(1);
        let axis_norm = if axis < 0 {
            (lhs_rank as i64 + axis) as usize
        } else {
            axis as usize
        };

        let dims: Vec<isize> = (0..lhs_rank)
            .filter(|&i| i != axis_norm)
            .map(|i| i as isize)
            .collect();
        quote! { (#rhs_expr).unsqueeze_dims(&[#(#dims),*]) }
    } else {
        let num_dims = lhs_rank - rhs_rank;
        let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
        quote! { (#rhs_expr).unsqueeze_dims(&[#(#dims),*]) }
    }
}
