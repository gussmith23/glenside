mod common;

use approx::AbsDiffEq;
use common::load_npy;
use egg::RecExpr;
use glenside::language::{interpreter::interpret, interpreter::Environment, interpreter::Value};

use glenside::models::resnet50::*;

/// See https://github.com/apache/incubator-tvm/blob/0a1c4c2174e1c4a04ca6e40cd90cdf7c2ef1d90a/python/tvm/relay/testing/resnet.py
#[test]
fn interpret_incomplete_resnet50_cifar10_nhwc_hwio() {
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

    let mut env = Environment::<f32>::default();
    let image = load_npy("data/resnet/image.npy");
    assert_eq!(image.shape(), &[1, 32, 32, 3]);
    env.insert("image", image);
    for var in &[
        "result",
        "bn_data_moving_mean_negated",
        "bn_data_moving_var_reciprocal_sqrt_plus_epsilon",
        "bn_data_gamma",
        "bn_data_beta",
        "conv0_weight",
        "stage1_unit1_conv1_weight",
        "stage1_unit1_conv2_weight",
        "stage1_unit1_conv3_weight",
    ] {
        env.insert(var, load_npy(&format!("data/resnet/{}.npy", var)));
    }
    let result = match interpret(&expr, data.into(), &env) {
        Value::Access(a) => a,
        _ => panic!(),
    };
    let true_result = load_npy::<f32>("data/resnet/result.npy");
    assert_eq!(result.tensor.shape(), true_result.shape());
    assert!(result.tensor.abs_diff_eq(&true_result, 5e-7));

    let _data = residual_unit(&mut expr, data, (1, 1), false, true, "stage1_unit1");
}
