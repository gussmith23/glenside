use egg::RecExpr;
use glenside::language::{ComputeType, Language, PadType};

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

/// Computes over an access
fn compute(expr: &mut Expr, compute_type: ComputeType, access_id: u32) -> u32 {
    let compute_type_id = expr.add(Language::ComputeType(compute_type));
    expr.add(Language::Compute([compute_type_id, access_id]))
}

/// Pairs the two accesses before computing over them
fn compute_pair(expr: &mut Expr, compute_type: ComputeType, a_id: u32, b_id: u32) -> u32 {
    let pair_id = expr.add(Language::AccessPair([a_id, b_id]));
    compute(expr, compute_type, pair_id)
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

    let data_id = compute_pair(expr, ComputeType::ElementwiseAdd, data_id, mean_id);
    let data_id = compute_pair(expr, ComputeType::ElementwiseMul, data_id, var_id);
    let data_id = compute_pair(expr, ComputeType::ElementwiseMul, data_id, gamma_id);
    let data_id = compute_pair(expr, ComputeType::ElementwiseAdd, data_id, beta_id);

    data_id
}

fn conv2d(
    expr: &mut Expr,
    data_id: u32,
    weights_name: &str,
    (stride_h, stride_w): (usize, usize),
    (pad_h, pad_w): (usize, usize),
) -> u32 {
    let usize_1_id = expr.add(Language::Usize(1));
    let usize_2_id = expr.add(Language::Usize(2));
    let usize_3_id = expr.add(Language::Usize(3));
    let (usize_stride_h, usize_stride_w) = (
        expr.add(Language::Usize(stride_h)),
        expr.add(Language::Usize(stride_w)),
    );
    let (usize_pad_h, usize_pad_w) = (
        expr.add(Language::Usize(pad_h)),
        expr.add(Language::Usize(pad_w)),
    );

    let zero_pad_id = expr.add(Language::PadType(PadType::ZeroPadding));

    let weights_id = expr.add(Language::Symbol(weights_name.to_string()));
    let shape_of_id = expr.add(Language::ShapeOf([weights_id]));
    let slice_shape_id = expr.add(Language::SliceShape([shape_of_id, usize_1_id]));

    let data_id = expr.add(Language::Access([data_id, usize_3_id]));
    let pad_h_id = expr.add(Language::AccessPad([
        data_id,
        zero_pad_id,
        usize_1_id,
        usize_pad_h,
        usize_pad_h,
    ]));
    let pad_w_id = expr.add(Language::AccessPad([
        pad_h_id,
        zero_pad_id,
        usize_2_id,
        usize_pad_w,
        usize_pad_w,
    ]));

    let access_windows_id = expr.add(Language::AccessWindows([
        pad_w_id,
        slice_shape_id,
        usize_stride_h,
        usize_stride_w,
    ]));

    let access_weights_id = access_tensor_literal(expr, weights_name, 1);

    let access_cartesian_product_id = expr.add(Language::AccessCartesianProduct([
        access_weights_id,
        access_windows_id,
    ]));

    let compute_dot_product_id =
        compute(expr, ComputeType::DotProduct, access_cartesian_product_id);

    compute_dot_product_id
}

#[test]
fn resnet50() {
    // TODO(@gussmith23) delete this comment
    //             filter_list = [64, 256, 512, 1024, 2048]

    let mut expr = RecExpr::default();

    // layout: C x H x W
    let data = access_tensor_literal(&mut expr, "image", 3);

    let data = batch_norm_inference(
        &mut expr,
        data,
        "bn0_gamma",
        "bn0_beta",
        "bn0_mean",
        "bn0_var",
    );

    // conv0_weights should be 3 in, 64 out, kernel size 3x3
    let _data = conv2d(&mut expr, data, "conv0_weights", (1, 1), (1, 1));

    #[rustfmt::skip]
    assert_eq!(
        expr.pretty(80),
"(compute
  dot-product
  (access-cartesian-product
    (access (access-tensor conv0_weights) 1)
    (access-windows
      (access-pad
        (access-pad
          (access
            (compute
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
                (access (access-tensor bn0_beta) 3)))
            3)
          zero-padding
          1
          1
          1)
        zero-padding
        2
        1
        1)
      (slice-shape (shape-of conv0_weights) 1)
      1
      1)))");
}
