mod common;

use approx::AbsDiffEq;
use common::load_npy;
use egg::RecExpr;
use glenside::language::{
    interpreter::interpret, interpreter::Environment, interpreter::Value, ComputeType, Language,
    PadType,
};

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

fn access_insert_axis(expr: &mut Expr, access_id: u32, axis: usize) -> u32 {
    let usize_id = expr.add(Language::Usize(axis));
    let insert_axis_id = expr.add(Language::AccessInsertAxis([access_id, usize_id]));
    insert_axis_id
}

fn get_access_shape(expr: &mut Expr, access_id: u32) -> u32 {
    let get_access_shape_id = expr.add(Language::GetAccessShape(access_id));
    get_access_shape_id
}

fn access_broadcast(expr: &mut Expr, access_id: u32, shape_id: u32) -> u32 {
    expr.add(Language::AccessBroadcast([access_id, shape_id]))
}

/// Inference-time batch normalization. Adds in mean and multiplies by variance.
/// This means that the mean should be negated beforehand and the reciprocal of
/// the sqrt of the variance (plus epsilon) should be taken.
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

    let shape_of_data = get_access_shape(expr, data_id);

    // <gamma tensor>
    let gamma_id = access_tensor_literal(expr, gamma_name, 1);
    let gamma_id = access_insert_axis(expr, gamma_id, 0);
    let gamma_id = access_insert_axis(expr, gamma_id, 2);
    let gamma_id = access_insert_axis(expr, gamma_id, 3);
    let gamma_id = access_broadcast(expr, gamma_id, shape_of_data);
    // <beta tensor>
    let beta_id = access_tensor_literal(expr, beta_name, 1);
    let beta_id = access_insert_axis(expr, beta_id, 0);
    let beta_id = access_insert_axis(expr, beta_id, 2);
    let beta_id = access_insert_axis(expr, beta_id, 3);
    let beta_id = access_broadcast(expr, beta_id, shape_of_data);
    // <mean tensor>
    let mean_id = access_tensor_literal(expr, mean_name, 1);
    let mean_id = access_insert_axis(expr, mean_id, 0);
    let mean_id = access_insert_axis(expr, mean_id, 2);
    let mean_id = access_insert_axis(expr, mean_id, 3);
    let mean_id = access_broadcast(expr, mean_id, shape_of_data);
    // <var tensor>
    let var_id = access_tensor_literal(expr, var_name, 1);
    let var_id = access_insert_axis(expr, var_id, 0);
    let var_id = access_insert_axis(expr, var_id, 2);
    let var_id = access_insert_axis(expr, var_id, 3);
    let var_id = access_broadcast(expr, var_id, shape_of_data);

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

fn relu(expr: &mut Expr, data_id: u32) -> u32 {
    compute(expr, ComputeType::ReLU, data_id)
}

/// h_index/w_index: dimension index of h/w dimensions
fn pad(
    expr: &mut Expr,
    data_id: u32,
    padding: (usize, usize),
    h_index: usize,
    w_index: usize,
) -> u32 {
    let h_axis_index_id = expr.add(Language::Usize(h_index));
    let w_axis_index_id = expr.add(Language::Usize(w_index));
    let zero_pad_id = expr.add(Language::PadType(PadType::ZeroPadding));
    let (usize_pad_h_id, usize_pad_w_id) = (
        expr.add(Language::Usize(padding.0)),
        expr.add(Language::Usize(padding.1)),
    );
    let pad_h_id = expr.add(Language::AccessPad([
        data_id,
        zero_pad_id,
        h_axis_index_id,
        usize_pad_h_id,
        usize_pad_h_id,
    ]));
    let pad_w_id = expr.add(Language::AccessPad([
        pad_h_id,
        zero_pad_id,
        w_axis_index_id,
        usize_pad_w_id,
        usize_pad_w_id,
    ]));

    pad_w_id
}

fn access(expr: &mut Expr, data_id: u32, access_axis: usize) -> u32 {
    let axis_id = expr.add(Language::Usize(access_axis));
    expr.add(Language::Access([data_id, axis_id]))
}

fn max_pool2d(
    expr: &mut Expr,
    data_id: u32,
    pool_size: (usize, usize),
    strides: (usize, usize),
    padding: (usize, usize),
) -> u32 {
    let usize_pool_size_c_id = expr.add(Language::Usize(1));
    let usize_pool_size_h_id = expr.add(Language::Usize(pool_size.0));
    let usize_pool_size_w_id = expr.add(Language::Usize(pool_size.1));
    let window_shape_id = expr.add(Language::Shape(Box::new([
        usize_pool_size_c_id,
        usize_pool_size_h_id,
        usize_pool_size_w_id,
    ])));

    let usize_stride_h_id = expr.add(Language::Usize(strides.0));
    let usize_stride_w_id = expr.add(Language::Usize(strides.1));

    let data_id = pad(expr, data_id, padding, 1, 2);
    // TODO(@gussmith23) Change this when you add batch dim
    let data_id = access(expr, data_id, 3);

    let access_windows_id = expr.add(Language::AccessWindows([
        data_id,
        window_shape_id,
        usize_stride_h_id,
        usize_stride_w_id,
    ]));

    let compute_id = compute(expr, ComputeType::ReduceMax, access_windows_id);

    compute_id
}

/// See https://github.com/apache/incubator-tvm/blob/0a1c4c2174e1c4a04ca6e40cd90cdf7c2ef1d90a/python/tvm/relay/testing/resnet.py
#[test]
fn resnet50() {
    // TODO(@gussmith23) delete this comment
    //             filter_list = [64, 256, 512, 1024, 2048]

    let mut expr = RecExpr::default();

    // layout: C x H x W
    let data = access_tensor_literal(&mut expr, "image", 4);

    let data = batch_norm_inference(
        &mut expr,
        data,
        "bn_data_gamma",
        "bn_data_beta",
        "bn_data_moving_mean_negated",
        "bn_data_moving_var_reciprocal_sqrt_plus_epsilon",
    );

    println!("{}", expr.pretty(80));
    let mut env = Environment::<f32>::default();
    env.insert("image", load_npy("data/resnet/image.npy"));
    for var in &[
        "image",
        "result",
        "bn_data_moving_mean_negated",
        "bn_data_moving_var_reciprocal_sqrt_plus_epsilon",
        "bn_data_gamma",
        "bn_data_beta",
    ] {
        env.insert(var, load_npy(&format!("data/resnet/{}.npy", var)));
    }
    let result = match interpret(&expr, data as usize, &env) {
        Value::Access(a) => a,
        _ => panic!(),
    };
    let true_result = load_npy::<f32>("data/resnet/result.npy");
    assert_eq!(result.tensor.shape(), true_result.shape());
    assert!(result.tensor.abs_diff_eq(&true_result, 1e-60));

    // conv0_weight should be 3 in, 64 out, kernel size 3x3
    let data = conv2d(&mut expr, data, "conv0_weight", (1, 1), (1, 1));

    let data = batch_norm_inference(
        &mut expr,
        data,
        "bn0_gamma",
        "bn0_beta",
        "bn0_mean",
        "bn0_var",
    );

    let data = relu(&mut expr, data);

    let _data = max_pool2d(&mut expr, data, (3, 3), (2, 2), (1, 1));
}
