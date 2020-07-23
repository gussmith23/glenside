use egg::RecExpr;
use glenside::language::{ComputeType, Language};

type Expr = RecExpr<Language>;

fn access_tensor_literal(expr: &mut Expr, name: &str, access_axis: usize) -> u32 {
    // <usize>
    let usize_literal_id = expr.add(Language::Usize(access_axis));
    // <tensor>
    let tensor_literal_id = expr.add(Language::Symbol(name.to_string()));
    // (access-tensor <tensor>)
    let access_tensor_id = expr.add(Language::AccessTensor(tensor_literal_id));
    // (access (access-tensor <tensor>) <axis>)
    let access_id = expr.add(Language::Access([access_tensor_id, usize_literal_id]));

    access_id
}

fn compute(expr: &mut Expr, compute_type: ComputeType, a_id: u32, b_id: u32) -> u32 {
    let compute_type_id = expr.add(Language::ComputeType(compute_type));
    let pair_id = expr.add(Language::AccessPair([a_id, b_id]));
    expr.add(Language::Compute([compute_type_id, pair_id]))
}

/// Inference-time batch normalization. Adds in mean and multiplies by variance.
/// This means that the mean should be negated beforehand and the reciprocal of
/// the sqrt of the variance (times epsilon) should be taken.
fn batch_norm_inference(
    expr: &mut Expr,
    data_id: u32,
    gamma_name: &str,
    beta_name: &str,
    mean_name: &str,
    var_name: &str,
) -> u32 {
    // TODO(@gussmith) Fix hardcoded access here
    // TODO(@gussmith) It doesn't matter how things are paired in elwise ops!
    // That is, regardless of the access axes of the two inputs, the result will
    // be the same. How to encode that?
    // I could wrap each argument in an (access <..> 0), then do access-pair,
    // and then wrap the compute result in an (access <..> axis), where we
    // determine axis based on the situation. I'm not sure what a good default
    // axis is.

    // <gamma tensor>
    let gamma_id = access_tensor_literal(expr, gamma_name, 3);
    // <beta tensor>
    let beta_id = access_tensor_literal(expr, beta_name, 3);
    // <mean tensor>
    let mean_id = access_tensor_literal(expr, mean_name, 3);
    // <var tensor>
    let var_id = access_tensor_literal(expr, var_name, 3);

    let data_id = compute(expr, ComputeType::ElementwiseAdd, data_id, mean_id);
    let data_id = compute(expr, ComputeType::ElementwiseMul, data_id, var_id);
    let data_id = compute(expr, ComputeType::ElementwiseMul, data_id, gamma_id);
    let data_id = compute(expr, ComputeType::ElementwiseAdd, data_id, beta_id);

    data_id
}

#[test]
fn resnet50() {
    let mut expr = RecExpr::default();

    // layout: C x H x W
    let data = access_tensor_literal(&mut expr, "image", 3);

    let _data = batch_norm_inference(
        &mut expr,
        data,
        "bn0_gamma",
        "bn0_beta",
        "bn0_mean",
        "bn0_var",
    );

    #[rustfmt::skip]
    assert_eq!(
        expr.pretty(80),
"(compute
  elementwise-add
  (access-pair
    (compute
      elementwise-mul
      (access-pair
        (compute
          elementwise-mul
          (access-pair
            (compute
              elementwise-add
              (access-pair
                (access (access-tensor image) 3)
                (access (access-tensor bn0_mean) 3)))
            (access (access-tensor bn0_var) 3)))
        (access (access-tensor bn0_gamma) 3)))
    (access (access-tensor bn0_beta) 3)))");
}
