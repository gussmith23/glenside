#![cfg(feature = "tvm")]

use egg::EGraph;
use egg::Extractor;
use egg::Pattern;
use egg::Runner;
use egg::Searcher;
use glenside::extraction::MonolithicCostFunction;
use glenside::language::rewrites::PadLocation;
use glenside::language::rewrites::PadSliceStrategy;
use glenside::language::rewrites::SliceConcatenateStrategy;
use glenside::language::MyAnalysis;
use glenside::language::PadType;
use std::collections::HashMap;

// Mobilenet, simplified for inference (so batch norms are removed).
// Generate with:
// ```python3
// import tvm
// from tvm import relay
// from tvm.relay.testing.mobilenet import get_workload
//
// mod, _ = get_workload()
// mod = relay.transform.SimplifyInference()(mod)
// print(mod.astext())
// ```
//
// TODO(@gussmith23) Shouldn't always panic.
// Panics at the moment because we can't actually handle the size of mobilenet.
#[should_panic]
#[test]
fn mobilenet_try_to_run_rewrites() {
    let relay = r#"
#[version = "0.0.5"]
def @main(%data: Tensor[(1, 3, 224, 224), float32], %conv_block_1_conv_weight: Tensor[(32, 3, 3, 3), float32], %conv_block_1_bn_gamma: Tensor[(32), float32], %conv_block_1_bn_beta: Tensor[(32), float32], %conv_block_1_bn_moving_mean: Tensor[(32), float32], %conv_block_1_bn_moving_var: Tensor[(32), float32], %separable_conv_block_1_weight: Tensor[(32, 1, 3, 3), float32], %separable_conv_block_1_bn1_gamma: Tensor[(32), float32], %separable_conv_block_1_bn1_beta: Tensor[(32), float32], %separable_conv_block_1_bn1_moving_mean: Tensor[(32), float32], %separable_conv_block_1_bn1_moving_var: Tensor[(32), float32], %separable_conv_block_1_conv2_weight: Tensor[(64, 32, 1, 1), float32], %separable_conv_block_1_bn2_gamma: Tensor[(64), float32], %separable_conv_block_1_bn2_beta: Tensor[(64), float32], %separable_conv_block_1_bn2_moving_mean: Tensor[(64), float32], %separable_conv_block_1_bn2_moving_var: Tensor[(64), float32], %separable_conv_block_2_weight: Tensor[(64, 1, 3, 3), float32], %separable_conv_block_2_bn1_gamma: Tensor[(64), float32], %separable_conv_block_2_bn1_beta: Tensor[(64), float32], %separable_conv_block_2_bn1_moving_mean: Tensor[(64), float32], %separable_conv_block_2_bn1_moving_var: Tensor[(64), float32], %separable_conv_block_2_conv2_weight: Tensor[(128, 64, 1, 1), float32], %separable_conv_block_2_bn2_gamma: Tensor[(128), float32], %separable_conv_block_2_bn2_beta: Tensor[(128), float32], %separable_conv_block_2_bn2_moving_mean: Tensor[(128), float32], %separable_conv_block_2_bn2_moving_var: Tensor[(128), float32], %separable_conv_block_3_weight: Tensor[(128, 1, 3, 3), float32], %separable_conv_block_3_bn1_gamma: Tensor[(128), float32], %separable_conv_block_3_bn1_beta: Tensor[(128), float32], %separable_conv_block_3_bn1_moving_mean: Tensor[(128), float32], %separable_conv_block_3_bn1_moving_var: Tensor[(128), float32], %separable_conv_block_3_conv2_weight: Tensor[(128, 128, 1, 1), float32], %separable_conv_block_3_bn2_gamma: Tensor[(128), float32], %separable_conv_block_3_bn2_beta: Tensor[(128), float32], %separable_conv_block_3_bn2_moving_mean: Tensor[(128), float32], %separable_conv_block_3_bn2_moving_var: Tensor[(128), float32], %separable_conv_block_4_weight: Tensor[(128, 1, 3, 3), float32], %separable_conv_block_4_bn1_gamma: Tensor[(128), float32], %separable_conv_block_4_bn1_beta: Tensor[(128), float32], %separable_conv_block_4_bn1_moving_mean: Tensor[(128), float32], %separable_conv_block_4_bn1_moving_var: Tensor[(128), float32], %separable_conv_block_4_conv2_weight: Tensor[(256, 128, 1, 1), float32], %separable_conv_block_4_bn2_gamma: Tensor[(256), float32], %separable_conv_block_4_bn2_beta: Tensor[(256), float32], %separable_conv_block_4_bn2_moving_mean: Tensor[(256), float32], %separable_conv_block_4_bn2_moving_var: Tensor[(256), float32], %separable_conv_block_5_weight: Tensor[(256, 1, 3, 3), float32], %separable_conv_block_5_bn1_gamma: Tensor[(256), float32], %separable_conv_block_5_bn1_beta: Tensor[(256), float32], %separable_conv_block_5_bn1_moving_mean: Tensor[(256), float32], %separable_conv_block_5_bn1_moving_var: Tensor[(256), float32], %separable_conv_block_5_conv2_weight: Tensor[(256, 256, 1, 1), float32], %separable_conv_block_5_bn2_gamma: Tensor[(256), float32], %separable_conv_block_5_bn2_beta: Tensor[(256), float32], %separable_conv_block_5_bn2_moving_mean: Tensor[(256), float32], %separable_conv_block_5_bn2_moving_var: Tensor[(256), float32], %separable_conv_block_6_weight: Tensor[(256, 1, 3, 3), float32], %separable_conv_block_6_bn1_gamma: Tensor[(256), float32], %separable_conv_block_6_bn1_beta: Tensor[(256), float32], %separable_conv_block_6_bn1_moving_mean: Tensor[(256), float32], %separable_conv_block_6_bn1_moving_var: Tensor[(256), float32], %separable_conv_block_6_conv2_weight: Tensor[(512, 256, 1, 1), float32], %separable_conv_block_6_bn2_gamma: Tensor[(512), float32], %separable_conv_block_6_bn2_beta: Tensor[(512), float32], %separable_conv_block_6_bn2_moving_mean: Tensor[(512), float32], %separable_conv_block_6_bn2_moving_var: Tensor[(512), float32], %separable_conv_block_7_weight: Tensor[(512, 1, 3, 3), float32], %separable_conv_block_7_bn1_gamma: Tensor[(512), float32], %separable_conv_block_7_bn1_beta: Tensor[(512), float32], %separable_conv_block_7_bn1_moving_mean: Tensor[(512), float32], %separable_conv_block_7_bn1_moving_var: Tensor[(512), float32], %separable_conv_block_7_conv2_weight: Tensor[(512, 512, 1, 1), float32], %separable_conv_block_7_bn2_gamma: Tensor[(512), float32], %separable_conv_block_7_bn2_beta: Tensor[(512), float32], %separable_conv_block_7_bn2_moving_mean: Tensor[(512), float32], %separable_conv_block_7_bn2_moving_var: Tensor[(512), float32], %separable_conv_block_8_weight: Tensor[(512, 1, 3, 3), float32], %separable_conv_block_8_bn1_gamma: Tensor[(512), float32], %separable_conv_block_8_bn1_beta: Tensor[(512), float32], %separable_conv_block_8_bn1_moving_mean: Tensor[(512), float32], %separable_conv_block_8_bn1_moving_var: Tensor[(512), float32], %separable_conv_block_8_conv2_weight: Tensor[(512, 512, 1, 1), float32], %separable_conv_block_8_bn2_gamma: Tensor[(512), float32], %separable_conv_block_8_bn2_beta: Tensor[(512), float32], %separable_conv_block_8_bn2_moving_mean: Tensor[(512), float32], %separable_conv_block_8_bn2_moving_var: Tensor[(512), float32], %separable_conv_block_9_weight: Tensor[(512, 1, 3, 3), float32], %separable_conv_block_9_bn1_gamma: Tensor[(512), float32], %separable_conv_block_9_bn1_beta: Tensor[(512), float32], %separable_conv_block_9_bn1_moving_mean: Tensor[(512), float32], %separable_conv_block_9_bn1_moving_var: Tensor[(512), float32], %separable_conv_block_9_conv2_weight: Tensor[(512, 512, 1, 1), float32], %separable_conv_block_9_bn2_gamma: Tensor[(512), float32], %separable_conv_block_9_bn2_beta: Tensor[(512), float32], %separable_conv_block_9_bn2_moving_mean: Tensor[(512), float32], %separable_conv_block_9_bn2_moving_var: Tensor[(512), float32], %separable_conv_block_10_weight: Tensor[(512, 1, 3, 3), float32], %separable_conv_block_10_bn1_gamma: Tensor[(512), float32], %separable_conv_block_10_bn1_beta: Tensor[(512), float32], %separable_conv_block_10_bn1_moving_mean: Tensor[(512), float32], %separable_conv_block_10_bn1_moving_var: Tensor[(512), float32], %separable_conv_block_10_conv2_weight: Tensor[(512, 512, 1, 1), float32], %separable_conv_block_10_bn2_gamma: Tensor[(512), float32], %separable_conv_block_10_bn2_beta: Tensor[(512), float32], %separable_conv_block_10_bn2_moving_mean: Tensor[(512), float32], %separable_conv_block_10_bn2_moving_var: Tensor[(512), float32], %separable_conv_block_11_weight: Tensor[(512, 1, 3, 3), float32], %separable_conv_block_11_bn1_gamma: Tensor[(512), float32], %separable_conv_block_11_bn1_beta: Tensor[(512), float32], %separable_conv_block_11_bn1_moving_mean: Tensor[(512), float32], %separable_conv_block_11_bn1_moving_var: Tensor[(512), float32], %separable_conv_block_11_conv2_weight: Tensor[(512, 512, 1, 1), float32], %separable_conv_block_11_bn2_gamma: Tensor[(512), float32], %separable_conv_block_11_bn2_beta: Tensor[(512), float32], %separable_conv_block_11_bn2_moving_mean: Tensor[(512), float32], %separable_conv_block_11_bn2_moving_var: Tensor[(512), float32], %separable_conv_block_12_weight: Tensor[(512, 1, 3, 3), float32], %separable_conv_block_12_bn1_gamma: Tensor[(512), float32], %separable_conv_block_12_bn1_beta: Tensor[(512), float32], %separable_conv_block_12_bn1_moving_mean: Tensor[(512), float32], %separable_conv_block_12_bn1_moving_var: Tensor[(512), float32], %separable_conv_block_12_conv2_weight: Tensor[(1024, 512, 1, 1), float32], %separable_conv_block_12_bn2_gamma: Tensor[(1024), float32], %separable_conv_block_12_bn2_beta: Tensor[(1024), float32], %separable_conv_block_12_bn2_moving_mean: Tensor[(1024), float32], %separable_conv_block_12_bn2_moving_var: Tensor[(1024), float32], %separable_conv_block_13_weight: Tensor[(1024, 1, 3, 3), float32], %separable_conv_block_13_bn1_gamma: Tensor[(1024), float32], %separable_conv_block_13_bn1_beta: Tensor[(1024), float32], %separable_conv_block_13_bn1_moving_mean: Tensor[(1024), float32], %separable_conv_block_13_bn1_moving_var: Tensor[(1024), float32], %separable_conv_block_13_conv2_weight: Tensor[(1024, 1024, 1, 1), float32], %separable_conv_block_13_bn2_gamma: Tensor[(1024), float32], %separable_conv_block_13_bn2_beta: Tensor[(1024), float32], %separable_conv_block_13_bn2_moving_mean: Tensor[(1024), float32], %separable_conv_block_13_bn2_moving_var: Tensor[(1024), float32], %fc_weight: Tensor[(1000, 1024), float32], %fc_bias: Tensor[(1000), float32]) -> Tensor[(1, 1000), float32] {
  %0 = nn.conv2d(%data, %conv_block_1_conv_weight, strides=[2, 2], padding=[1, 1, 1, 1], channels=32, kernel_size=[3, 3]) /* ty=Tensor[(1, 32, 112, 112), float32] */;
  %1 = add(%conv_block_1_bn_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(32), float32] */;
  %2 = sqrt(%1) /* ty=Tensor[(32), float32] */;
  %3 = divide(1f /* ty=float32 */, %2) /* ty=Tensor[(32), float32] */;
  %4 = multiply(%3, %conv_block_1_bn_gamma) /* ty=Tensor[(32), float32] */;
  %5 = expand_dims(%4, axis=1, num_newaxis=2) /* ty=Tensor[(32, 1, 1), float32] */;
  %6 = multiply(%0, %5) /* ty=Tensor[(1, 32, 112, 112), float32] */;
  %7 = negative(%conv_block_1_bn_moving_mean) /* ty=Tensor[(32), float32] */;
  %8 = multiply(%7, %4) /* ty=Tensor[(32), float32] */;
  %9 = add(%8, %conv_block_1_bn_beta) /* ty=Tensor[(32), float32] */;
  %10 = expand_dims(%9, axis=1, num_newaxis=2) /* ty=Tensor[(32, 1, 1), float32] */;
  %11 = add(%6, %10) /* ty=Tensor[(1, 32, 112, 112), float32] */;
  %12 = nn.relu(%11) /* ty=Tensor[(1, 32, 112, 112), float32] */;
  %13 = nn.conv2d(%12, %separable_conv_block_1_weight, padding=[1, 1, 1, 1], groups=32, channels=32, kernel_size=[3, 3]) /* ty=Tensor[(1, 32, 112, 112), float32] */;
  %14 = add(%separable_conv_block_1_bn1_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(32), float32] */;
  %15 = sqrt(%14) /* ty=Tensor[(32), float32] */;
  %16 = divide(1f /* ty=float32 */, %15) /* ty=Tensor[(32), float32] */;
  %17 = multiply(%16, %separable_conv_block_1_bn1_gamma) /* ty=Tensor[(32), float32] */;
  %18 = expand_dims(%17, axis=1, num_newaxis=2) /* ty=Tensor[(32, 1, 1), float32] */;
  %19 = multiply(%13, %18) /* ty=Tensor[(1, 32, 112, 112), float32] */;
  %20 = negative(%separable_conv_block_1_bn1_moving_mean) /* ty=Tensor[(32), float32] */;
  %21 = multiply(%20, %17) /* ty=Tensor[(32), float32] */;
  %22 = add(%21, %separable_conv_block_1_bn1_beta) /* ty=Tensor[(32), float32] */;
  %23 = expand_dims(%22, axis=1, num_newaxis=2) /* ty=Tensor[(32, 1, 1), float32] */;
  %24 = add(%19, %23) /* ty=Tensor[(1, 32, 112, 112), float32] */;
  %25 = nn.relu(%24) /* ty=Tensor[(1, 32, 112, 112), float32] */;
  %26 = nn.conv2d(%25, %separable_conv_block_1_conv2_weight, padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1]) /* ty=Tensor[(1, 64, 112, 112), float32] */;
  %27 = add(%separable_conv_block_1_bn2_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(64), float32] */;
  %28 = sqrt(%27) /* ty=Tensor[(64), float32] */;
  %29 = divide(1f /* ty=float32 */, %28) /* ty=Tensor[(64), float32] */;
  %30 = multiply(%29, %separable_conv_block_1_bn2_gamma) /* ty=Tensor[(64), float32] */;
  %31 = expand_dims(%30, axis=1, num_newaxis=2) /* ty=Tensor[(64, 1, 1), float32] */;
  %32 = multiply(%26, %31) /* ty=Tensor[(1, 64, 112, 112), float32] */;
  %33 = negative(%separable_conv_block_1_bn2_moving_mean) /* ty=Tensor[(64), float32] */;
  %34 = multiply(%33, %30) /* ty=Tensor[(64), float32] */;
  %35 = add(%34, %separable_conv_block_1_bn2_beta) /* ty=Tensor[(64), float32] */;
  %36 = expand_dims(%35, axis=1, num_newaxis=2) /* ty=Tensor[(64, 1, 1), float32] */;
  %37 = add(%32, %36) /* ty=Tensor[(1, 64, 112, 112), float32] */;
  %38 = nn.relu(%37) /* ty=Tensor[(1, 64, 112, 112), float32] */;
  %39 = nn.conv2d(%38, %separable_conv_block_2_weight, strides=[2, 2], padding=[1, 1, 1, 1], groups=64, channels=64, kernel_size=[3, 3]) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %40 = add(%separable_conv_block_2_bn1_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(64), float32] */;
  %41 = sqrt(%40) /* ty=Tensor[(64), float32] */;
  %42 = divide(1f /* ty=float32 */, %41) /* ty=Tensor[(64), float32] */;
  %43 = multiply(%42, %separable_conv_block_2_bn1_gamma) /* ty=Tensor[(64), float32] */;
  %44 = expand_dims(%43, axis=1, num_newaxis=2) /* ty=Tensor[(64, 1, 1), float32] */;
  %45 = multiply(%39, %44) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %46 = negative(%separable_conv_block_2_bn1_moving_mean) /* ty=Tensor[(64), float32] */;
  %47 = multiply(%46, %43) /* ty=Tensor[(64), float32] */;
  %48 = add(%47, %separable_conv_block_2_bn1_beta) /* ty=Tensor[(64), float32] */;
  %49 = expand_dims(%48, axis=1, num_newaxis=2) /* ty=Tensor[(64, 1, 1), float32] */;
  %50 = add(%45, %49) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %51 = nn.relu(%50) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %52 = nn.conv2d(%51, %separable_conv_block_2_conv2_weight, padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1]) /* ty=Tensor[(1, 128, 56, 56), float32] */;
  %53 = add(%separable_conv_block_2_bn2_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(128), float32] */;
  %54 = sqrt(%53) /* ty=Tensor[(128), float32] */;
  %55 = divide(1f /* ty=float32 */, %54) /* ty=Tensor[(128), float32] */;
  %56 = multiply(%55, %separable_conv_block_2_bn2_gamma) /* ty=Tensor[(128), float32] */;
  %57 = expand_dims(%56, axis=1, num_newaxis=2) /* ty=Tensor[(128, 1, 1), float32] */;
  %58 = multiply(%52, %57) /* ty=Tensor[(1, 128, 56, 56), float32] */;
  %59 = negative(%separable_conv_block_2_bn2_moving_mean) /* ty=Tensor[(128), float32] */;
  %60 = multiply(%59, %56) /* ty=Tensor[(128), float32] */;
  %61 = add(%60, %separable_conv_block_2_bn2_beta) /* ty=Tensor[(128), float32] */;
  %62 = expand_dims(%61, axis=1, num_newaxis=2) /* ty=Tensor[(128, 1, 1), float32] */;
  %63 = add(%58, %62) /* ty=Tensor[(1, 128, 56, 56), float32] */;
  %64 = nn.relu(%63) /* ty=Tensor[(1, 128, 56, 56), float32] */;
  %65 = nn.conv2d(%64, %separable_conv_block_3_weight, padding=[1, 1, 1, 1], groups=128, channels=128, kernel_size=[3, 3]) /* ty=Tensor[(1, 128, 56, 56), float32] */;
  %66 = add(%separable_conv_block_3_bn1_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(128), float32] */;
  %67 = sqrt(%66) /* ty=Tensor[(128), float32] */;
  %68 = divide(1f /* ty=float32 */, %67) /* ty=Tensor[(128), float32] */;
  %69 = multiply(%68, %separable_conv_block_3_bn1_gamma) /* ty=Tensor[(128), float32] */;
  %70 = expand_dims(%69, axis=1, num_newaxis=2) /* ty=Tensor[(128, 1, 1), float32] */;
  %71 = multiply(%65, %70) /* ty=Tensor[(1, 128, 56, 56), float32] */;
  %72 = negative(%separable_conv_block_3_bn1_moving_mean) /* ty=Tensor[(128), float32] */;
  %73 = multiply(%72, %69) /* ty=Tensor[(128), float32] */;
  %74 = add(%73, %separable_conv_block_3_bn1_beta) /* ty=Tensor[(128), float32] */;
  %75 = expand_dims(%74, axis=1, num_newaxis=2) /* ty=Tensor[(128, 1, 1), float32] */;
  %76 = add(%71, %75) /* ty=Tensor[(1, 128, 56, 56), float32] */;
  %77 = nn.relu(%76) /* ty=Tensor[(1, 128, 56, 56), float32] */;
  %78 = nn.conv2d(%77, %separable_conv_block_3_conv2_weight, padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1]) /* ty=Tensor[(1, 128, 56, 56), float32] */;
  %79 = add(%separable_conv_block_3_bn2_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(128), float32] */;
  %80 = sqrt(%79) /* ty=Tensor[(128), float32] */;
  %81 = divide(1f /* ty=float32 */, %80) /* ty=Tensor[(128), float32] */;
  %82 = multiply(%81, %separable_conv_block_3_bn2_gamma) /* ty=Tensor[(128), float32] */;
  %83 = expand_dims(%82, axis=1, num_newaxis=2) /* ty=Tensor[(128, 1, 1), float32] */;
  %84 = multiply(%78, %83) /* ty=Tensor[(1, 128, 56, 56), float32] */;
  %85 = negative(%separable_conv_block_3_bn2_moving_mean) /* ty=Tensor[(128), float32] */;
  %86 = multiply(%85, %82) /* ty=Tensor[(128), float32] */;
  %87 = add(%86, %separable_conv_block_3_bn2_beta) /* ty=Tensor[(128), float32] */;
  %88 = expand_dims(%87, axis=1, num_newaxis=2) /* ty=Tensor[(128, 1, 1), float32] */;
  %89 = add(%84, %88) /* ty=Tensor[(1, 128, 56, 56), float32] */;
  %90 = nn.relu(%89) /* ty=Tensor[(1, 128, 56, 56), float32] */;
  %91 = nn.conv2d(%90, %separable_conv_block_4_weight, strides=[2, 2], padding=[1, 1, 1, 1], groups=128, channels=128, kernel_size=[3, 3]) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %92 = add(%separable_conv_block_4_bn1_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(128), float32] */;
  %93 = sqrt(%92) /* ty=Tensor[(128), float32] */;
  %94 = divide(1f /* ty=float32 */, %93) /* ty=Tensor[(128), float32] */;
  %95 = multiply(%94, %separable_conv_block_4_bn1_gamma) /* ty=Tensor[(128), float32] */;
  %96 = expand_dims(%95, axis=1, num_newaxis=2) /* ty=Tensor[(128, 1, 1), float32] */;
  %97 = multiply(%91, %96) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %98 = negative(%separable_conv_block_4_bn1_moving_mean) /* ty=Tensor[(128), float32] */;
  %99 = multiply(%98, %95) /* ty=Tensor[(128), float32] */;
  %100 = add(%99, %separable_conv_block_4_bn1_beta) /* ty=Tensor[(128), float32] */;
  %101 = expand_dims(%100, axis=1, num_newaxis=2) /* ty=Tensor[(128, 1, 1), float32] */;
  %102 = add(%97, %101) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %103 = nn.relu(%102) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %104 = nn.conv2d(%103, %separable_conv_block_4_conv2_weight, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]) /* ty=Tensor[(1, 256, 28, 28), float32] */;
  %105 = add(%separable_conv_block_4_bn2_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(256), float32] */;
  %106 = sqrt(%105) /* ty=Tensor[(256), float32] */;
  %107 = divide(1f /* ty=float32 */, %106) /* ty=Tensor[(256), float32] */;
  %108 = multiply(%107, %separable_conv_block_4_bn2_gamma) /* ty=Tensor[(256), float32] */;
  %109 = expand_dims(%108, axis=1, num_newaxis=2) /* ty=Tensor[(256, 1, 1), float32] */;
  %110 = multiply(%104, %109) /* ty=Tensor[(1, 256, 28, 28), float32] */;
  %111 = negative(%separable_conv_block_4_bn2_moving_mean) /* ty=Tensor[(256), float32] */;
  %112 = multiply(%111, %108) /* ty=Tensor[(256), float32] */;
  %113 = add(%112, %separable_conv_block_4_bn2_beta) /* ty=Tensor[(256), float32] */;
  %114 = expand_dims(%113, axis=1, num_newaxis=2) /* ty=Tensor[(256, 1, 1), float32] */;
  %115 = add(%110, %114) /* ty=Tensor[(1, 256, 28, 28), float32] */;
  %116 = nn.relu(%115) /* ty=Tensor[(1, 256, 28, 28), float32] */;
  %117 = nn.conv2d(%116, %separable_conv_block_5_weight, padding=[1, 1, 1, 1], groups=256, channels=256, kernel_size=[3, 3]) /* ty=Tensor[(1, 256, 28, 28), float32] */;
  %118 = add(%separable_conv_block_5_bn1_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(256), float32] */;
  %119 = sqrt(%118) /* ty=Tensor[(256), float32] */;
  %120 = divide(1f /* ty=float32 */, %119) /* ty=Tensor[(256), float32] */;
  %121 = multiply(%120, %separable_conv_block_5_bn1_gamma) /* ty=Tensor[(256), float32] */;
  %122 = expand_dims(%121, axis=1, num_newaxis=2) /* ty=Tensor[(256, 1, 1), float32] */;
  %123 = multiply(%117, %122) /* ty=Tensor[(1, 256, 28, 28), float32] */;
  %124 = negative(%separable_conv_block_5_bn1_moving_mean) /* ty=Tensor[(256), float32] */;
  %125 = multiply(%124, %121) /* ty=Tensor[(256), float32] */;
  %126 = add(%125, %separable_conv_block_5_bn1_beta) /* ty=Tensor[(256), float32] */;
  %127 = expand_dims(%126, axis=1, num_newaxis=2) /* ty=Tensor[(256, 1, 1), float32] */;
  %128 = add(%123, %127) /* ty=Tensor[(1, 256, 28, 28), float32] */;
  %129 = nn.relu(%128) /* ty=Tensor[(1, 256, 28, 28), float32] */;
  %130 = nn.conv2d(%129, %separable_conv_block_5_conv2_weight, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]) /* ty=Tensor[(1, 256, 28, 28), float32] */;
  %131 = add(%separable_conv_block_5_bn2_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(256), float32] */;
  %132 = sqrt(%131) /* ty=Tensor[(256), float32] */;
  %133 = divide(1f /* ty=float32 */, %132) /* ty=Tensor[(256), float32] */;
  %134 = multiply(%133, %separable_conv_block_5_bn2_gamma) /* ty=Tensor[(256), float32] */;
  %135 = expand_dims(%134, axis=1, num_newaxis=2) /* ty=Tensor[(256, 1, 1), float32] */;
  %136 = multiply(%130, %135) /* ty=Tensor[(1, 256, 28, 28), float32] */;
  %137 = negative(%separable_conv_block_5_bn2_moving_mean) /* ty=Tensor[(256), float32] */;
  %138 = multiply(%137, %134) /* ty=Tensor[(256), float32] */;
  %139 = add(%138, %separable_conv_block_5_bn2_beta) /* ty=Tensor[(256), float32] */;
  %140 = expand_dims(%139, axis=1, num_newaxis=2) /* ty=Tensor[(256, 1, 1), float32] */;
  %141 = add(%136, %140) /* ty=Tensor[(1, 256, 28, 28), float32] */;
  %142 = nn.relu(%141) /* ty=Tensor[(1, 256, 28, 28), float32] */;
  %143 = nn.conv2d(%142, %separable_conv_block_6_weight, strides=[2, 2], padding=[1, 1, 1, 1], groups=256, channels=256, kernel_size=[3, 3]) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %144 = add(%separable_conv_block_6_bn1_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(256), float32] */;
  %145 = sqrt(%144) /* ty=Tensor[(256), float32] */;
  %146 = divide(1f /* ty=float32 */, %145) /* ty=Tensor[(256), float32] */;
  %147 = multiply(%146, %separable_conv_block_6_bn1_gamma) /* ty=Tensor[(256), float32] */;
  %148 = expand_dims(%147, axis=1, num_newaxis=2) /* ty=Tensor[(256, 1, 1), float32] */;
  %149 = multiply(%143, %148) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %150 = negative(%separable_conv_block_6_bn1_moving_mean) /* ty=Tensor[(256), float32] */;
  %151 = multiply(%150, %147) /* ty=Tensor[(256), float32] */;
  %152 = add(%151, %separable_conv_block_6_bn1_beta) /* ty=Tensor[(256), float32] */;
  %153 = expand_dims(%152, axis=1, num_newaxis=2) /* ty=Tensor[(256, 1, 1), float32] */;
  %154 = add(%149, %153) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %155 = nn.relu(%154) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %156 = nn.conv2d(%155, %separable_conv_block_6_conv2_weight, padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %157 = add(%separable_conv_block_6_bn2_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(512), float32] */;
  %158 = sqrt(%157) /* ty=Tensor[(512), float32] */;
  %159 = divide(1f /* ty=float32 */, %158) /* ty=Tensor[(512), float32] */;
  %160 = multiply(%159, %separable_conv_block_6_bn2_gamma) /* ty=Tensor[(512), float32] */;
  %161 = expand_dims(%160, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %162 = multiply(%156, %161) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %163 = negative(%separable_conv_block_6_bn2_moving_mean) /* ty=Tensor[(512), float32] */;
  %164 = multiply(%163, %160) /* ty=Tensor[(512), float32] */;
  %165 = add(%164, %separable_conv_block_6_bn2_beta) /* ty=Tensor[(512), float32] */;
  %166 = expand_dims(%165, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %167 = add(%162, %166) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %168 = nn.relu(%167) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %169 = nn.conv2d(%168, %separable_conv_block_7_weight, padding=[1, 1, 1, 1], groups=512, channels=512, kernel_size=[3, 3]) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %170 = add(%separable_conv_block_7_bn1_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(512), float32] */;
  %171 = sqrt(%170) /* ty=Tensor[(512), float32] */;
  %172 = divide(1f /* ty=float32 */, %171) /* ty=Tensor[(512), float32] */;
  %173 = multiply(%172, %separable_conv_block_7_bn1_gamma) /* ty=Tensor[(512), float32] */;
  %174 = expand_dims(%173, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %175 = multiply(%169, %174) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %176 = negative(%separable_conv_block_7_bn1_moving_mean) /* ty=Tensor[(512), float32] */;
  %177 = multiply(%176, %173) /* ty=Tensor[(512), float32] */;
  %178 = add(%177, %separable_conv_block_7_bn1_beta) /* ty=Tensor[(512), float32] */;
  %179 = expand_dims(%178, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %180 = add(%175, %179) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %181 = nn.relu(%180) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %182 = nn.conv2d(%181, %separable_conv_block_7_conv2_weight, padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %183 = add(%separable_conv_block_7_bn2_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(512), float32] */;
  %184 = sqrt(%183) /* ty=Tensor[(512), float32] */;
  %185 = divide(1f /* ty=float32 */, %184) /* ty=Tensor[(512), float32] */;
  %186 = multiply(%185, %separable_conv_block_7_bn2_gamma) /* ty=Tensor[(512), float32] */;
  %187 = expand_dims(%186, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %188 = multiply(%182, %187) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %189 = negative(%separable_conv_block_7_bn2_moving_mean) /* ty=Tensor[(512), float32] */;
  %190 = multiply(%189, %186) /* ty=Tensor[(512), float32] */;
  %191 = add(%190, %separable_conv_block_7_bn2_beta) /* ty=Tensor[(512), float32] */;
  %192 = expand_dims(%191, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %193 = add(%188, %192) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %194 = nn.relu(%193) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %195 = nn.conv2d(%194, %separable_conv_block_8_weight, padding=[1, 1, 1, 1], groups=512, channels=512, kernel_size=[3, 3]) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %196 = add(%separable_conv_block_8_bn1_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(512), float32] */;
  %197 = sqrt(%196) /* ty=Tensor[(512), float32] */;
  %198 = divide(1f /* ty=float32 */, %197) /* ty=Tensor[(512), float32] */;
  %199 = multiply(%198, %separable_conv_block_8_bn1_gamma) /* ty=Tensor[(512), float32] */;
  %200 = expand_dims(%199, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %201 = multiply(%195, %200) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %202 = negative(%separable_conv_block_8_bn1_moving_mean) /* ty=Tensor[(512), float32] */;
  %203 = multiply(%202, %199) /* ty=Tensor[(512), float32] */;
  %204 = add(%203, %separable_conv_block_8_bn1_beta) /* ty=Tensor[(512), float32] */;
  %205 = expand_dims(%204, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %206 = add(%201, %205) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %207 = nn.relu(%206) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %208 = nn.conv2d(%207, %separable_conv_block_8_conv2_weight, padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %209 = add(%separable_conv_block_8_bn2_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(512), float32] */;
  %210 = sqrt(%209) /* ty=Tensor[(512), float32] */;
  %211 = divide(1f /* ty=float32 */, %210) /* ty=Tensor[(512), float32] */;
  %212 = multiply(%211, %separable_conv_block_8_bn2_gamma) /* ty=Tensor[(512), float32] */;
  %213 = expand_dims(%212, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %214 = multiply(%208, %213) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %215 = negative(%separable_conv_block_8_bn2_moving_mean) /* ty=Tensor[(512), float32] */;
  %216 = multiply(%215, %212) /* ty=Tensor[(512), float32] */;
  %217 = add(%216, %separable_conv_block_8_bn2_beta) /* ty=Tensor[(512), float32] */;
  %218 = expand_dims(%217, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %219 = add(%214, %218) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %220 = nn.relu(%219) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %221 = nn.conv2d(%220, %separable_conv_block_9_weight, padding=[1, 1, 1, 1], groups=512, channels=512, kernel_size=[3, 3]) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %222 = add(%separable_conv_block_9_bn1_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(512), float32] */;
  %223 = sqrt(%222) /* ty=Tensor[(512), float32] */;
  %224 = divide(1f /* ty=float32 */, %223) /* ty=Tensor[(512), float32] */;
  %225 = multiply(%224, %separable_conv_block_9_bn1_gamma) /* ty=Tensor[(512), float32] */;
  %226 = expand_dims(%225, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %227 = multiply(%221, %226) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %228 = negative(%separable_conv_block_9_bn1_moving_mean) /* ty=Tensor[(512), float32] */;
  %229 = multiply(%228, %225) /* ty=Tensor[(512), float32] */;
  %230 = add(%229, %separable_conv_block_9_bn1_beta) /* ty=Tensor[(512), float32] */;
  %231 = expand_dims(%230, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %232 = add(%227, %231) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %233 = nn.relu(%232) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %234 = nn.conv2d(%233, %separable_conv_block_9_conv2_weight, padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %235 = add(%separable_conv_block_9_bn2_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(512), float32] */;
  %236 = sqrt(%235) /* ty=Tensor[(512), float32] */;
  %237 = divide(1f /* ty=float32 */, %236) /* ty=Tensor[(512), float32] */;
  %238 = multiply(%237, %separable_conv_block_9_bn2_gamma) /* ty=Tensor[(512), float32] */;
  %239 = expand_dims(%238, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %240 = multiply(%234, %239) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %241 = negative(%separable_conv_block_9_bn2_moving_mean) /* ty=Tensor[(512), float32] */;
  %242 = multiply(%241, %238) /* ty=Tensor[(512), float32] */;
  %243 = add(%242, %separable_conv_block_9_bn2_beta) /* ty=Tensor[(512), float32] */;
  %244 = expand_dims(%243, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %245 = add(%240, %244) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %246 = nn.relu(%245) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %247 = nn.conv2d(%246, %separable_conv_block_10_weight, padding=[1, 1, 1, 1], groups=512, channels=512, kernel_size=[3, 3]) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %248 = add(%separable_conv_block_10_bn1_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(512), float32] */;
  %249 = sqrt(%248) /* ty=Tensor[(512), float32] */;
  %250 = divide(1f /* ty=float32 */, %249) /* ty=Tensor[(512), float32] */;
  %251 = multiply(%250, %separable_conv_block_10_bn1_gamma) /* ty=Tensor[(512), float32] */;
  %252 = expand_dims(%251, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %253 = multiply(%247, %252) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %254 = negative(%separable_conv_block_10_bn1_moving_mean) /* ty=Tensor[(512), float32] */;
  %255 = multiply(%254, %251) /* ty=Tensor[(512), float32] */;
  %256 = add(%255, %separable_conv_block_10_bn1_beta) /* ty=Tensor[(512), float32] */;
  %257 = expand_dims(%256, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %258 = add(%253, %257) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %259 = nn.relu(%258) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %260 = nn.conv2d(%259, %separable_conv_block_10_conv2_weight, padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %261 = add(%separable_conv_block_10_bn2_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(512), float32] */;
  %262 = sqrt(%261) /* ty=Tensor[(512), float32] */;
  %263 = divide(1f /* ty=float32 */, %262) /* ty=Tensor[(512), float32] */;
  %264 = multiply(%263, %separable_conv_block_10_bn2_gamma) /* ty=Tensor[(512), float32] */;
  %265 = expand_dims(%264, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %266 = multiply(%260, %265) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %267 = negative(%separable_conv_block_10_bn2_moving_mean) /* ty=Tensor[(512), float32] */;
  %268 = multiply(%267, %264) /* ty=Tensor[(512), float32] */;
  %269 = add(%268, %separable_conv_block_10_bn2_beta) /* ty=Tensor[(512), float32] */;
  %270 = expand_dims(%269, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %271 = add(%266, %270) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %272 = nn.relu(%271) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %273 = nn.conv2d(%272, %separable_conv_block_11_weight, padding=[1, 1, 1, 1], groups=512, channels=512, kernel_size=[3, 3]) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %274 = add(%separable_conv_block_11_bn1_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(512), float32] */;
  %275 = sqrt(%274) /* ty=Tensor[(512), float32] */;
  %276 = divide(1f /* ty=float32 */, %275) /* ty=Tensor[(512), float32] */;
  %277 = multiply(%276, %separable_conv_block_11_bn1_gamma) /* ty=Tensor[(512), float32] */;
  %278 = expand_dims(%277, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %279 = multiply(%273, %278) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %280 = negative(%separable_conv_block_11_bn1_moving_mean) /* ty=Tensor[(512), float32] */;
  %281 = multiply(%280, %277) /* ty=Tensor[(512), float32] */;
  %282 = add(%281, %separable_conv_block_11_bn1_beta) /* ty=Tensor[(512), float32] */;
  %283 = expand_dims(%282, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %284 = add(%279, %283) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %285 = nn.relu(%284) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %286 = nn.conv2d(%285, %separable_conv_block_11_conv2_weight, padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %287 = add(%separable_conv_block_11_bn2_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(512), float32] */;
  %288 = sqrt(%287) /* ty=Tensor[(512), float32] */;
  %289 = divide(1f /* ty=float32 */, %288) /* ty=Tensor[(512), float32] */;
  %290 = multiply(%289, %separable_conv_block_11_bn2_gamma) /* ty=Tensor[(512), float32] */;
  %291 = expand_dims(%290, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %292 = multiply(%286, %291) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %293 = negative(%separable_conv_block_11_bn2_moving_mean) /* ty=Tensor[(512), float32] */;
  %294 = multiply(%293, %290) /* ty=Tensor[(512), float32] */;
  %295 = add(%294, %separable_conv_block_11_bn2_beta) /* ty=Tensor[(512), float32] */;
  %296 = expand_dims(%295, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %297 = add(%292, %296) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %298 = nn.relu(%297) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %299 = nn.conv2d(%298, %separable_conv_block_12_weight, strides=[2, 2], padding=[1, 1, 1, 1], groups=512, channels=512, kernel_size=[3, 3]) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %300 = add(%separable_conv_block_12_bn1_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(512), float32] */;
  %301 = sqrt(%300) /* ty=Tensor[(512), float32] */;
  %302 = divide(1f /* ty=float32 */, %301) /* ty=Tensor[(512), float32] */;
  %303 = multiply(%302, %separable_conv_block_12_bn1_gamma) /* ty=Tensor[(512), float32] */;
  %304 = expand_dims(%303, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %305 = multiply(%299, %304) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %306 = negative(%separable_conv_block_12_bn1_moving_mean) /* ty=Tensor[(512), float32] */;
  %307 = multiply(%306, %303) /* ty=Tensor[(512), float32] */;
  %308 = add(%307, %separable_conv_block_12_bn1_beta) /* ty=Tensor[(512), float32] */;
  %309 = expand_dims(%308, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %310 = add(%305, %309) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %311 = nn.relu(%310) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %312 = nn.conv2d(%311, %separable_conv_block_12_conv2_weight, padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1]) /* ty=Tensor[(1, 1024, 7, 7), float32] */;
  %313 = add(%separable_conv_block_12_bn2_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(1024), float32] */;
  %314 = sqrt(%313) /* ty=Tensor[(1024), float32] */;
  %315 = divide(1f /* ty=float32 */, %314) /* ty=Tensor[(1024), float32] */;
  %316 = multiply(%315, %separable_conv_block_12_bn2_gamma) /* ty=Tensor[(1024), float32] */;
  %317 = expand_dims(%316, axis=1, num_newaxis=2) /* ty=Tensor[(1024, 1, 1), float32] */;
  %318 = multiply(%312, %317) /* ty=Tensor[(1, 1024, 7, 7), float32] */;
  %319 = negative(%separable_conv_block_12_bn2_moving_mean) /* ty=Tensor[(1024), float32] */;
  %320 = multiply(%319, %316) /* ty=Tensor[(1024), float32] */;
  %321 = add(%320, %separable_conv_block_12_bn2_beta) /* ty=Tensor[(1024), float32] */;
  %322 = expand_dims(%321, axis=1, num_newaxis=2) /* ty=Tensor[(1024, 1, 1), float32] */;
  %323 = add(%318, %322) /* ty=Tensor[(1, 1024, 7, 7), float32] */;
  %324 = nn.relu(%323) /* ty=Tensor[(1, 1024, 7, 7), float32] */;
  %325 = nn.conv2d(%324, %separable_conv_block_13_weight, padding=[1, 1, 1, 1], groups=1024, channels=1024, kernel_size=[3, 3]) /* ty=Tensor[(1, 1024, 7, 7), float32] */;
  %326 = add(%separable_conv_block_13_bn1_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(1024), float32] */;
  %327 = sqrt(%326) /* ty=Tensor[(1024), float32] */;
  %328 = divide(1f /* ty=float32 */, %327) /* ty=Tensor[(1024), float32] */;
  %329 = multiply(%328, %separable_conv_block_13_bn1_gamma) /* ty=Tensor[(1024), float32] */;
  %330 = expand_dims(%329, axis=1, num_newaxis=2) /* ty=Tensor[(1024, 1, 1), float32] */;
  %331 = multiply(%325, %330) /* ty=Tensor[(1, 1024, 7, 7), float32] */;
  %332 = negative(%separable_conv_block_13_bn1_moving_mean) /* ty=Tensor[(1024), float32] */;
  %333 = multiply(%332, %329) /* ty=Tensor[(1024), float32] */;
  %334 = add(%333, %separable_conv_block_13_bn1_beta) /* ty=Tensor[(1024), float32] */;
  %335 = expand_dims(%334, axis=1, num_newaxis=2) /* ty=Tensor[(1024, 1, 1), float32] */;
  %336 = add(%331, %335) /* ty=Tensor[(1, 1024, 7, 7), float32] */;
  %337 = nn.relu(%336) /* ty=Tensor[(1, 1024, 7, 7), float32] */;
  %338 = nn.conv2d(%337, %separable_conv_block_13_conv2_weight, padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1]) /* ty=Tensor[(1, 1024, 7, 7), float32] */;
  %339 = add(%separable_conv_block_13_bn2_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(1024), float32] */;
  %340 = sqrt(%339) /* ty=Tensor[(1024), float32] */;
  %341 = divide(1f /* ty=float32 */, %340) /* ty=Tensor[(1024), float32] */;
  %342 = multiply(%341, %separable_conv_block_13_bn2_gamma) /* ty=Tensor[(1024), float32] */;
  %343 = expand_dims(%342, axis=1, num_newaxis=2) /* ty=Tensor[(1024, 1, 1), float32] */;
  %344 = multiply(%338, %343) /* ty=Tensor[(1, 1024, 7, 7), float32] */;
  %345 = negative(%separable_conv_block_13_bn2_moving_mean) /* ty=Tensor[(1024), float32] */;
  %346 = multiply(%345, %342) /* ty=Tensor[(1024), float32] */;
  %347 = add(%346, %separable_conv_block_13_bn2_beta) /* ty=Tensor[(1024), float32] */;
  %348 = expand_dims(%347, axis=1, num_newaxis=2) /* ty=Tensor[(1024, 1, 1), float32] */;
  %349 = add(%344, %348) /* ty=Tensor[(1, 1024, 7, 7), float32] */;
  %350 = nn.relu(%349) /* ty=Tensor[(1, 1024, 7, 7), float32] */;
  %351 = nn.global_avg_pool2d(%350) /* ty=Tensor[(1, 1024, 1, 1), float32] */;
  %352 = nn.batch_flatten(%351) /* ty=Tensor[(1, 1024), float32] */;
  %353 = nn.dense(%352, %fc_weight, units=1000) /* ty=Tensor[(1, 1000), float32] */;
  %354 = nn.bias_add(%353, %fc_bias) /* ty=Tensor[(1, 1000), float32] */;
  nn.softmax(%354) /* ty=Tensor[(1, 1000), float32] */
}
"#;
    let module = tvm::ir::module::IRModule::parse("", relay);

    let (expr, shapes_vec) = glenside::language::from_relay::from_relay(&module);

    let mut env = HashMap::default();
    for (k, v) in &shapes_vec {
        env.insert(k.clone(), v.clone());
    }

    // TODO(@gussmith23) Include some simple simplifying rewrites
    // If we add some very basic rewrites here, then $glenside_str
    // won't need to exactly match what's actually produced by
    // from_relay.py. It can be simpler (e.g. collapsing accesses).
    let mut egraph = EGraph::new(MyAnalysis {
        name_to_shape: env.clone(),
    });
    let _id = egraph.add_expr(&expr);

    let rws = vec![
        glenside::language::rewrites::flatten_unflatten_any_access(),
        glenside::language::rewrites::bubble_reshape_through_cartesian_product(),
        glenside::language::rewrites::bubble_reshape_through_compute_dot_product(),
        glenside::language::rewrites::bubble_access_concatenate_through_access_cartesian_product_not_item_axis_left(),
        glenside::language::rewrites::bubble_access_concatenate_through_access_cartesian_product_not_item_axis_right(),
        glenside::language::rewrites::bubble_access_concatenate_through_access_cartesian_product_same_item_axis(),
        glenside::language::rewrites::bubble_access_concatenate_through_compute_dot_product_item_axis(),
        glenside::language::rewrites::bubble_access_concatenate_through_compute_dot_product_not_item_axis(),
        glenside::language::rewrites::bubble_access_slice_through_access_pad_inequal_axes(),
        glenside::language::rewrites::systolic_array(),
        glenside::language::rewrites::pad_slice_accesses(
            0,
            PadSliceStrategy::PadToClosestMultipleOf {
                multiple_of: 64,
                pad_location: PadLocation::End,
                pad_type: PadType::ZeroPadding,
            },
        ),
        glenside::language::rewrites::pad_slice_accesses(
            1,
            PadSliceStrategy::PadToClosestMultipleOf {
                multiple_of: 64,
                pad_location: PadLocation::End,
                pad_type: PadType::ZeroPadding,
            },
        ),
        glenside::language::rewrites::slice_concatenate_accesses(
            0,
            SliceConcatenateStrategy::DivideInto { segment_size: 64 },
        ),
        glenside::language::rewrites::slice_concatenate_accesses(
            1,
            SliceConcatenateStrategy::DivideInto { segment_size: 64 },
        ),
        glenside::language::rewrites::bubble_access_slice_through_access_cartesian_product_not_item_axis_left(),
        glenside::language::rewrites::bubble_access_slice_through_access_cartesian_product_not_item_axis_right(),
        glenside::language::rewrites::bubble_access_slice_through_access_cartesian_product_same_item_axis(),
        glenside::language::rewrites::bubble_access_slice_through_compute_dot_product_not_item_axis(),
        glenside::language::rewrites::bubble_access_slice_through_compute_dot_product_item_axis_not_tuple_axis(),
    ];

    // TODO(@gussmith23) This is starting to become a flaky test...
    // I know the correct program can be found, but it takes time.
    let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
        .with_egraph(egraph)
        .with_time_limit(std::time::Duration::from_secs(10))
        .with_node_limit(500000)
        .with_iter_limit(40)
        .run(&rws);

    runner.print_report();

    // Did any tensorization happen?
    assert!(
        "(systolic-array ?a ?b ?c ?d)"
            .parse::<Pattern<_>>()
            .unwrap()
            .search(&runner.egraph)
            .len()
            > 0
    );

    // Did tensorization to 64x64 happen? (Harder than tensorizing to just
    // anything)
    assert!(
        "(systolic-array 64 64 ?c ?d)"
            .parse::<Pattern<_>>()
            .unwrap()
            .search(&runner.egraph)
            .len()
            > 0
    );

    // Can we extract something that can be turned into a hardware design?
    let _ex = Extractor::new(
        &runner.egraph,
        MonolithicCostFunction {
            systolic_array_configuration: (16, 128),
            egraph: &runner.egraph,
            prefer_systolic_arrays_with_blocking: false,
        },
    );
    // TODO(@gussmith23) This is overflowing the stack
    // See https://github.com/egraphs-good/egg/issues/50
    // let (cost, _) = ex.find_best(id);
    // assert!(cost < usize::MAX);
}
