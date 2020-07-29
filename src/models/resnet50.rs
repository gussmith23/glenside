use crate::language::{ComputeType, Language, PadType};
use egg::RecExpr;

type Expr = RecExpr<Language>;

fn residual_unit(
    expr: &mut Expr,
    data_id: u32,
    stride: (usize, usize),
    dim_match: bool,
    bottle_neck: bool,
    name: &str,
) -> u32 {
    if bottle_neck {
        let bn1_id = batch_norm_inference(
            expr,
            data_id,
            format!("{}_bn1_gamma", name).as_str(),
            format!("{}_bn1_beta", name).as_str(),
            format!("{}_bn1_moving_mean", name).as_str(),
            format!("{}_bn1_moving_var", name).as_str(),
        );
        let act1_id = relu(expr, bn1_id);
        let conv1_id = conv2d(
            expr,
            act1_id,
            format!("{}_conv1_weight", name).as_str(),
            stride,
            (0, 0),
        );
        let bn2_id = batch_norm_inference(
            expr,
            conv1_id,
            format!("{}_bn2_gamma", name).as_str(),
            format!("{}_bn2_beta", name).as_str(),
            format!("{}_bn2_moving_mean", name).as_str(),
            format!("{}_bn2_moving_var", name).as_str(),
        );
        let act2_id = relu(expr, bn2_id);
        let conv2_id = conv2d(
            expr,
            act2_id,
            format!("{}_conv2_weight", name).as_str(),
            (1, 1),
            (1, 1),
        );
        let bn3_id = batch_norm_inference(
            expr,
            conv2_id,
            format!("{}_bn3_gamma", name).as_str(),
            format!("{}_bn3_beta", name).as_str(),
            format!("{}_bn3_moving_mean", name).as_str(),
            format!("{}_bn3_moving_var", name).as_str(),
        );
        let act3_id = relu(expr, bn3_id);
        let conv3_id = conv2d(
            expr,
            act3_id,
            format!("{}_conv3_weight", name).as_str(),
            (1, 1),
            (0, 0),
        );
        let shortcut_id = if dim_match {
            data_id
        } else {
            // TODO(@gussmith23) Padding correct here?
            conv2d(
                expr,
                act1_id,
                format!("{}_sc_weight", name).as_str(),
                stride,
                (0, 0),
            )
        };
        let access_data_id = access(expr, conv3_id, 0);
        let access_sc_id = access(expr, shortcut_id, 0);
        compute_pair(
            expr,
            ComputeType::ElementwiseAdd,
            access_data_id,
            access_sc_id,
        )
    } else {
        todo!("not implemented, not needed for CIFAR10 Resnet50")
    }
}

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
    // TODO(@gussmith23) Layout assumption
    let gamma_id = access_tensor_literal(expr, gamma_name, 1);
    let gamma_id = access_insert_axis(expr, gamma_id, 0);
    let gamma_id = access_insert_axis(expr, gamma_id, 0);
    let gamma_id = access_insert_axis(expr, gamma_id, 0);
    let gamma_id = access_broadcast(expr, gamma_id, shape_of_data);
    // <beta tensor>
    // TODO(@gussmith23) Layout assumption
    let beta_id = access_tensor_literal(expr, beta_name, 1);
    let beta_id = access_insert_axis(expr, beta_id, 0);
    let beta_id = access_insert_axis(expr, beta_id, 0);
    let beta_id = access_insert_axis(expr, beta_id, 0);
    let beta_id = access_broadcast(expr, beta_id, shape_of_data);
    // <mean tensor>
    // TODO(@gussmith23) Layout assumption
    let mean_id = access_tensor_literal(expr, mean_name, 1);
    let mean_id = access_insert_axis(expr, mean_id, 0);
    let mean_id = access_insert_axis(expr, mean_id, 0);
    let mean_id = access_insert_axis(expr, mean_id, 0);
    let mean_id = access_broadcast(expr, mean_id, shape_of_data);
    // <var tensor>
    // TODO(@gussmith23) Layout assumption
    let var_id = access_tensor_literal(expr, var_name, 1);
    let var_id = access_insert_axis(expr, var_id, 0);
    let var_id = access_insert_axis(expr, var_id, 0);
    let var_id = access_insert_axis(expr, var_id, 0);
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
    let usize_0_id = expr.add(Language::Usize(0));
    let usize_1_id = expr.add(Language::Usize(1));
    let usize_2_id = expr.add(Language::Usize(2));
    let usize_3_id = expr.add(Language::Usize(3));
    let usize_4_id = expr.add(Language::Usize(4));
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
    let weights_shape_id = expr.add(Language::ShapeOf([weights_id]));
    // TODO(@gussmith23) Assuming output channels is the last dimension
    let weights_shape_id = expr.add(Language::ShapeRemoveAxis([weights_shape_id, usize_3_id]));
    // TODO(@gussmith23) Assuming batch is the first dimension
    let weights_shape_id = expr.add(Language::ShapeInsertAxis([weights_shape_id, usize_0_id]));

    // Access data at last location, assuming 4 dimensions.
    // TODO(@gussmith23) Assuming 4 dimensions
    let data_id = expr.add(Language::Access([data_id, usize_4_id]));
    // TODO(@gussmith23) Layout assumption
    let pad_h_id = expr.add(Language::AccessPad([
        data_id,
        zero_pad_id,
        usize_1_id,
        usize_pad_h,
        usize_pad_h,
    ]));
    // TODO(@gussmith23) Layout assumption
    let pad_w_id = expr.add(Language::AccessPad([
        pad_h_id,
        zero_pad_id,
        usize_2_id,
        usize_pad_w,
        usize_pad_w,
    ]));

    // TODO(@gussmith23) Layout assumption
    let stride_shape_id = expr.add(Language::Shape(Box::new([
        // N
        usize_1_id,
        // H
        usize_stride_h,
        // W
        usize_stride_w,
        // C
        usize_1_id,
    ])));
    let access_windows_id = expr.add(Language::AccessWindows([
        pad_w_id,
        weights_shape_id,
        stride_shape_id,
    ]));
    // TODO(@gussmith23) Layout assumption
    // Squeeze both the (now collapsed to 1) input channels dimension and the
    // batch dimension of the filters.
    let access_windows_id = expr.add(Language::AccessSqueeze([access_windows_id, usize_3_id]));
    let access_windows_id = expr.add(Language::AccessSqueeze([access_windows_id, usize_3_id]));
    // Access at the start of the filters.
    let access_windows_id = expr.add(Language::Access([access_windows_id, usize_3_id]));

    // Access at H.
    // TODO(@gussmith23) Layout assumption
    let access_weights_id = access_tensor_literal(expr, weights_name, 0);
    // Weights are in HWIO. Move O to first to get access pat [O] [H, W, I]
    // TODO(@gussmith23) Layout assumption
    let access_weights_id = expr.add(Language::AccessMoveAxis([
        access_weights_id,
        usize_3_id,
        usize_0_id,
    ]));

    let access_cartesian_product_id = expr.add(Language::AccessCartesianProduct([
        access_weights_id,
        access_windows_id,
    ]));

    let compute_dot_product_id =
        compute(expr, ComputeType::DotProduct, access_cartesian_product_id);

    // TODO(@gussmith23) Layout assumption
    // Move old output/new input channels dimension to the back.
    let data_id = expr.add(Language::AccessMoveAxis([
        compute_dot_product_id,
        usize_0_id,
        usize_3_id,
    ]));

    data_id
}

fn relu(expr: &mut Expr, data_id: u32) -> u32 {
    compute(expr, ComputeType::ReLU, data_id)
}

/// h_index/w_index: dimension index of h/w dimensions
fn _pad(
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

fn _max_pool2d(
    expr: &mut Expr,
    data_id: u32,
    pool_size: (usize, usize),
    strides: (usize, usize),
    padding: (usize, usize),
) -> u32 {
    let usize_1_id = expr.add(Language::Usize(1));
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

    let data_id = _pad(expr, data_id, padding, 1, 2);
    // TODO(@gussmith23) Change this when you add batch dim
    let data_id = access(expr, data_id, 3);

    // TODO(@gussmith23) Hardcoded to CHW.
    let stride_shape_id = expr.add(Language::Shape(Box::new([
        usize_1_id,
        usize_stride_h_id,
        usize_stride_w_id,
    ])));
    let access_windows_id = expr.add(Language::AccessWindows([
        data_id,
        window_shape_id,
        stride_shape_id,
    ]));

    let compute_id = compute(expr, ComputeType::ReduceMax, access_windows_id);

    compute_id
}

pub fn resnet50_cifar10_nhwc_hwio() -> (RecExpr<Language>, u32) {
    let mut expr = RecExpr::default();

    let data = access_tensor_literal(&mut expr, "image", 4);

    let data = batch_norm_inference(
        &mut expr,
        data,
        "bn_data_gamma",
        "bn_data_beta",
        "bn_data_moving_mean_negated",
        "bn_data_moving_var_reciprocal_sqrt_plus_epsilon",
    );

    let data = conv2d(&mut expr, data, "conv0_weight", (1, 1), (1, 1));

    let data = residual_unit(&mut expr, data, (1, 1), false, true, "stage1_unit1");

    (expr, data)
}
