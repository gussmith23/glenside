#![cfg(feature = "tvm")]

use egg::EGraph;
use glenside::language::MyAnalysis;
use std::collections::HashMap;

// ResNet18, simplified for inference (so batch norms are removed).
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
/// Can we parse (but not run) resnet18?
#[test]
fn resnet18_relay_to_glenside() {
    test_logger::ensure_env_logger_initialized();
    let relay = r#"
#[version = "0.0.5"]
def @main(%data: Tensor[(1, 3, 224, 224), float32], %bn_data_gamma: Tensor[(3), float32], %bn_data_beta: Tensor[(3), float32], %bn_data_moving_mean: Tensor[(3), float32], %bn_data_moving_var: Tensor[(3), float32], %conv0_weight: Tensor[(64, 3, 7, 7), float32], %bn0_gamma: Tensor[(64), float32], %bn0_beta: Tensor[(64), float32], %bn0_moving_mean: Tensor[(64), float32], %bn0_moving_var: Tensor[(64), float32], %stage1_unit1_bn1_gamma: Tensor[(64), float32], %stage1_unit1_bn1_beta: Tensor[(64), float32], %stage1_unit1_bn1_moving_mean: Tensor[(64), float32], %stage1_unit1_bn1_moving_var: Tensor[(64), float32], %stage1_unit1_conv1_weight: Tensor[(64, 64, 3, 3), float32], %stage1_unit1_bn2_gamma: Tensor[(64), float32], %stage1_unit1_bn2_beta: Tensor[(64), float32], %stage1_unit1_bn2_moving_mean: Tensor[(64), float32], %stage1_unit1_bn2_moving_var: Tensor[(64), float32], %stage1_unit1_conv2_weight: Tensor[(64, 64, 3, 3), float32], %stage1_unit1_sc_weight: Tensor[(64, 64, 1, 1), float32], %stage1_unit2_bn1_gamma: Tensor[(64), float32], %stage1_unit2_bn1_beta: Tensor[(64), float32], %stage1_unit2_bn1_moving_mean: Tensor[(64), float32], %stage1_unit2_bn1_moving_var: Tensor[(64), float32], %stage1_unit2_conv1_weight: Tensor[(64, 64, 3, 3), float32], %stage1_unit2_bn2_gamma: Tensor[(64), float32], %stage1_unit2_bn2_beta: Tensor[(64), float32], %stage1_unit2_bn2_moving_mean: Tensor[(64), float32], %stage1_unit2_bn2_moving_var: Tensor[(64), float32], %stage1_unit2_conv2_weight: Tensor[(64, 64, 3, 3), float32], %stage2_unit1_bn1_gamma: Tensor[(64), float32], %stage2_unit1_bn1_beta: Tensor[(64), float32], %stage2_unit1_bn1_moving_mean: Tensor[(64), float32], %stage2_unit1_bn1_moving_var: Tensor[(64), float32], %stage2_unit1_conv1_weight: Tensor[(128, 64, 3, 3), float32], %stage2_unit1_bn2_gamma: Tensor[(128), float32], %stage2_unit1_bn2_beta: Tensor[(128), float32], %stage2_unit1_bn2_moving_mean: Tensor[(128), float32], %stage2_unit1_bn2_moving_var: Tensor[(128), float32], %stage2_unit1_conv2_weight: Tensor[(128, 128, 3, 3), float32], %stage2_unit1_sc_weight: Tensor[(128, 64, 1, 1), float32], %stage2_unit2_bn1_gamma: Tensor[(128), float32], %stage2_unit2_bn1_beta: Tensor[(128), float32], %stage2_unit2_bn1_moving_mean: Tensor[(128), float32], %stage2_unit2_bn1_moving_var: Tensor[(128), float32], %stage2_unit2_conv1_weight: Tensor[(128, 128, 3, 3), float32], %stage2_unit2_bn2_gamma: Tensor[(128), float32], %stage2_unit2_bn2_beta: Tensor[(128), float32], %stage2_unit2_bn2_moving_mean: Tensor[(128), float32], %stage2_unit2_bn2_moving_var: Tensor[(128), float32], %stage2_unit2_conv2_weight: Tensor[(128, 128, 3, 3), float32], %stage3_unit1_bn1_gamma: Tensor[(128), float32], %stage3_unit1_bn1_beta: Tensor[(128), float32], %stage3_unit1_bn1_moving_mean: Tensor[(128), float32], %stage3_unit1_bn1_moving_var: Tensor[(128), float32], %stage3_unit1_conv1_weight: Tensor[(256, 128, 3, 3), float32], %stage3_unit1_bn2_gamma: Tensor[(256), float32], %stage3_unit1_bn2_beta: Tensor[(256), float32], %stage3_unit1_bn2_moving_mean: Tensor[(256), float32], %stage3_unit1_bn2_moving_var: Tensor[(256), float32], %stage3_unit1_conv2_weight: Tensor[(256, 256, 3, 3), float32], %stage3_unit1_sc_weight: Tensor[(256, 128, 1, 1), float32], %stage3_unit2_bn1_gamma: Tensor[(256), float32], %stage3_unit2_bn1_beta: Tensor[(256), float32], %stage3_unit2_bn1_moving_mean: Tensor[(256), float32], %stage3_unit2_bn1_moving_var: Tensor[(256), float32], %stage3_unit2_conv1_weight: Tensor[(256, 256, 3, 3), float32], %stage3_unit2_bn2_gamma: Tensor[(256), float32], %stage3_unit2_bn2_beta: Tensor[(256), float32], %stage3_unit2_bn2_moving_mean: Tensor[(256), float32], %stage3_unit2_bn2_moving_var: Tensor[(256), float32], %stage3_unit2_conv2_weight: Tensor[(256, 256, 3, 3), float32], %stage4_unit1_bn1_gamma: Tensor[(256), float32], %stage4_unit1_bn1_beta: Tensor[(256), float32], %stage4_unit1_bn1_moving_mean: Tensor[(256), float32], %stage4_unit1_bn1_moving_var: Tensor[(256), float32], %stage4_unit1_conv1_weight: Tensor[(512, 256, 3, 3), float32], %stage4_unit1_bn2_gamma: Tensor[(512), float32], %stage4_unit1_bn2_beta: Tensor[(512), float32], %stage4_unit1_bn2_moving_mean: Tensor[(512), float32], %stage4_unit1_bn2_moving_var: Tensor[(512), float32], %stage4_unit1_conv2_weight: Tensor[(512, 512, 3, 3), float32], %stage4_unit1_sc_weight: Tensor[(512, 256, 1, 1), float32], %stage4_unit2_bn1_gamma: Tensor[(512), float32], %stage4_unit2_bn1_beta: Tensor[(512), float32], %stage4_unit2_bn1_moving_mean: Tensor[(512), float32], %stage4_unit2_bn1_moving_var: Tensor[(512), float32], %stage4_unit2_conv1_weight: Tensor[(512, 512, 3, 3), float32], %stage4_unit2_bn2_gamma: Tensor[(512), float32], %stage4_unit2_bn2_beta: Tensor[(512), float32], %stage4_unit2_bn2_moving_mean: Tensor[(512), float32], %stage4_unit2_bn2_moving_var: Tensor[(512), float32], %stage4_unit2_conv2_weight: Tensor[(512, 512, 3, 3), float32], %bn1_gamma: Tensor[(512), float32], %bn1_beta: Tensor[(512), float32], %bn1_moving_mean: Tensor[(512), float32], %bn1_moving_var: Tensor[(512), float32], %fc1_weight: Tensor[(1000, 512), float32], %fc1_bias: Tensor[(1000), float32]) -> Tensor[(1, 1000), float32] {
  %0 = add(%bn_data_moving_var, 2e-05f /* ty=float32 */) /* ty=Tensor[(3), float32] */;
  %1 = sqrt(%0) /* ty=Tensor[(3), float32] */;
  %2 = divide(1f /* ty=float32 */, %1) /* ty=Tensor[(3), float32] */;
  %3 = expand_dims(%2, axis=1, num_newaxis=2) /* ty=Tensor[(3, 1, 1), float32] */;
  %4 = multiply(%data, %3) /* ty=Tensor[(1, 3, 224, 224), float32] */;
  %5 = negative(%bn_data_moving_mean) /* ty=Tensor[(3), float32] */;
  %6 = multiply(%5, %2) /* ty=Tensor[(3), float32] */;
  %7 = add(%6, %bn_data_beta) /* ty=Tensor[(3), float32] */;
  %8 = expand_dims(%7, axis=1, num_newaxis=2) /* ty=Tensor[(3, 1, 1), float32] */;
  %9 = add(%4, %8) /* ty=Tensor[(1, 3, 224, 224), float32] */;
  %10 = nn.conv2d(%9, %conv0_weight, strides=[2, 2], padding=[3, 3, 3, 3], channels=64, kernel_size=[7, 7]) /* ty=Tensor[(1, 64, 112, 112), float32] */;
  %11 = add(%bn0_moving_var, 2e-05f /* ty=float32 */) /* ty=Tensor[(64), float32] */;
  %12 = sqrt(%11) /* ty=Tensor[(64), float32] */;
  %13 = divide(1f /* ty=float32 */, %12) /* ty=Tensor[(64), float32] */;
  %14 = multiply(%13, %bn0_gamma) /* ty=Tensor[(64), float32] */;
  %15 = expand_dims(%14, axis=1, num_newaxis=2) /* ty=Tensor[(64, 1, 1), float32] */;
  %16 = multiply(%10, %15) /* ty=Tensor[(1, 64, 112, 112), float32] */;
  %17 = negative(%bn0_moving_mean) /* ty=Tensor[(64), float32] */;
  %18 = multiply(%17, %14) /* ty=Tensor[(64), float32] */;
  %19 = add(%18, %bn0_beta) /* ty=Tensor[(64), float32] */;
  %20 = expand_dims(%19, axis=1, num_newaxis=2) /* ty=Tensor[(64, 1, 1), float32] */;
  %21 = add(%16, %20) /* ty=Tensor[(1, 64, 112, 112), float32] */;
  %22 = nn.relu(%21) /* ty=Tensor[(1, 64, 112, 112), float32] */;
  %23 = nn.max_pool2d(%22, pool_size=[3, 3], strides=[2, 2], padding=[1, 1, 1, 1]) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %24 = add(%stage1_unit1_bn1_moving_var, 2e-05f /* ty=float32 */) /* ty=Tensor[(64), float32] */;
  %25 = sqrt(%24) /* ty=Tensor[(64), float32] */;
  %26 = divide(1f /* ty=float32 */, %25) /* ty=Tensor[(64), float32] */;
  %27 = multiply(%26, %stage1_unit1_bn1_gamma) /* ty=Tensor[(64), float32] */;
  %28 = expand_dims(%27, axis=1, num_newaxis=2) /* ty=Tensor[(64, 1, 1), float32] */;
  %29 = multiply(%23, %28) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %30 = negative(%stage1_unit1_bn1_moving_mean) /* ty=Tensor[(64), float32] */;
  %31 = multiply(%30, %27) /* ty=Tensor[(64), float32] */;
  %32 = add(%31, %stage1_unit1_bn1_beta) /* ty=Tensor[(64), float32] */;
  %33 = expand_dims(%32, axis=1, num_newaxis=2) /* ty=Tensor[(64, 1, 1), float32] */;
  %34 = add(%29, %33) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %35 = nn.relu(%34) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %36 = nn.conv2d(%35, %stage1_unit1_conv1_weight, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3]) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %37 = add(%stage1_unit1_bn2_moving_var, 2e-05f /* ty=float32 */) /* ty=Tensor[(64), float32] */;
  %38 = sqrt(%37) /* ty=Tensor[(64), float32] */;
  %39 = divide(1f /* ty=float32 */, %38) /* ty=Tensor[(64), float32] */;
  %40 = multiply(%39, %stage1_unit1_bn2_gamma) /* ty=Tensor[(64), float32] */;
  %41 = expand_dims(%40, axis=1, num_newaxis=2) /* ty=Tensor[(64, 1, 1), float32] */;
  %42 = multiply(%36, %41) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %43 = negative(%stage1_unit1_bn2_moving_mean) /* ty=Tensor[(64), float32] */;
  %44 = multiply(%43, %40) /* ty=Tensor[(64), float32] */;
  %45 = add(%44, %stage1_unit1_bn2_beta) /* ty=Tensor[(64), float32] */;
  %46 = expand_dims(%45, axis=1, num_newaxis=2) /* ty=Tensor[(64, 1, 1), float32] */;
  %47 = add(%42, %46) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %48 = nn.relu(%47) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %49 = nn.conv2d(%48, %stage1_unit1_conv2_weight, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3]) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %50 = nn.conv2d(%35, %stage1_unit1_sc_weight, padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1]) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %51 = add(%49, %50) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %52 = add(%stage1_unit2_bn1_moving_var, 2e-05f /* ty=float32 */) /* ty=Tensor[(64), float32] */;
  %53 = sqrt(%52) /* ty=Tensor[(64), float32] */;
  %54 = divide(1f /* ty=float32 */, %53) /* ty=Tensor[(64), float32] */;
  %55 = multiply(%54, %stage1_unit2_bn1_gamma) /* ty=Tensor[(64), float32] */;
  %56 = expand_dims(%55, axis=1, num_newaxis=2) /* ty=Tensor[(64, 1, 1), float32] */;
  %57 = multiply(%51, %56) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %58 = negative(%stage1_unit2_bn1_moving_mean) /* ty=Tensor[(64), float32] */;
  %59 = multiply(%58, %55) /* ty=Tensor[(64), float32] */;
  %60 = add(%59, %stage1_unit2_bn1_beta) /* ty=Tensor[(64), float32] */;
  %61 = expand_dims(%60, axis=1, num_newaxis=2) /* ty=Tensor[(64, 1, 1), float32] */;
  %62 = add(%57, %61) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %63 = nn.relu(%62) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %64 = nn.conv2d(%63, %stage1_unit2_conv1_weight, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3]) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %65 = add(%stage1_unit2_bn2_moving_var, 2e-05f /* ty=float32 */) /* ty=Tensor[(64), float32] */;
  %66 = sqrt(%65) /* ty=Tensor[(64), float32] */;
  %67 = divide(1f /* ty=float32 */, %66) /* ty=Tensor[(64), float32] */;
  %68 = multiply(%67, %stage1_unit2_bn2_gamma) /* ty=Tensor[(64), float32] */;
  %69 = expand_dims(%68, axis=1, num_newaxis=2) /* ty=Tensor[(64, 1, 1), float32] */;
  %70 = multiply(%64, %69) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %71 = negative(%stage1_unit2_bn2_moving_mean) /* ty=Tensor[(64), float32] */;
  %72 = multiply(%71, %68) /* ty=Tensor[(64), float32] */;
  %73 = add(%72, %stage1_unit2_bn2_beta) /* ty=Tensor[(64), float32] */;
  %74 = expand_dims(%73, axis=1, num_newaxis=2) /* ty=Tensor[(64, 1, 1), float32] */;
  %75 = add(%70, %74) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %76 = nn.relu(%75) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %77 = nn.conv2d(%76, %stage1_unit2_conv2_weight, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3]) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %78 = add(%77, %51) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %79 = add(%stage2_unit1_bn1_moving_var, 2e-05f /* ty=float32 */) /* ty=Tensor[(64), float32] */;
  %80 = sqrt(%79) /* ty=Tensor[(64), float32] */;
  %81 = divide(1f /* ty=float32 */, %80) /* ty=Tensor[(64), float32] */;
  %82 = multiply(%81, %stage2_unit1_bn1_gamma) /* ty=Tensor[(64), float32] */;
  %83 = expand_dims(%82, axis=1, num_newaxis=2) /* ty=Tensor[(64, 1, 1), float32] */;
  %84 = multiply(%78, %83) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %85 = negative(%stage2_unit1_bn1_moving_mean) /* ty=Tensor[(64), float32] */;
  %86 = multiply(%85, %82) /* ty=Tensor[(64), float32] */;
  %87 = add(%86, %stage2_unit1_bn1_beta) /* ty=Tensor[(64), float32] */;
  %88 = expand_dims(%87, axis=1, num_newaxis=2) /* ty=Tensor[(64, 1, 1), float32] */;
  %89 = add(%84, %88) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %90 = nn.relu(%89) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %91 = nn.conv2d(%90, %stage2_unit1_conv1_weight, strides=[2, 2], padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3]) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %92 = add(%stage2_unit1_bn2_moving_var, 2e-05f /* ty=float32 */) /* ty=Tensor[(128), float32] */;
  %93 = sqrt(%92) /* ty=Tensor[(128), float32] */;
  %94 = divide(1f /* ty=float32 */, %93) /* ty=Tensor[(128), float32] */;
  %95 = multiply(%94, %stage2_unit1_bn2_gamma) /* ty=Tensor[(128), float32] */;
  %96 = expand_dims(%95, axis=1, num_newaxis=2) /* ty=Tensor[(128, 1, 1), float32] */;
  %97 = multiply(%91, %96) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %98 = negative(%stage2_unit1_bn2_moving_mean) /* ty=Tensor[(128), float32] */;
  %99 = multiply(%98, %95) /* ty=Tensor[(128), float32] */;
  %100 = add(%99, %stage2_unit1_bn2_beta) /* ty=Tensor[(128), float32] */;
  %101 = expand_dims(%100, axis=1, num_newaxis=2) /* ty=Tensor[(128, 1, 1), float32] */;
  %102 = add(%97, %101) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %103 = nn.relu(%102) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %104 = nn.conv2d(%103, %stage2_unit1_conv2_weight, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3]) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %105 = nn.conv2d(%90, %stage2_unit1_sc_weight, strides=[2, 2], padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1]) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %106 = add(%104, %105) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %107 = add(%stage2_unit2_bn1_moving_var, 2e-05f /* ty=float32 */) /* ty=Tensor[(128), float32] */;
  %108 = sqrt(%107) /* ty=Tensor[(128), float32] */;
  %109 = divide(1f /* ty=float32 */, %108) /* ty=Tensor[(128), float32] */;
  %110 = multiply(%109, %stage2_unit2_bn1_gamma) /* ty=Tensor[(128), float32] */;
  %111 = expand_dims(%110, axis=1, num_newaxis=2) /* ty=Tensor[(128, 1, 1), float32] */;
  %112 = multiply(%106, %111) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %113 = negative(%stage2_unit2_bn1_moving_mean) /* ty=Tensor[(128), float32] */;
  %114 = multiply(%113, %110) /* ty=Tensor[(128), float32] */;
  %115 = add(%114, %stage2_unit2_bn1_beta) /* ty=Tensor[(128), float32] */;
  %116 = expand_dims(%115, axis=1, num_newaxis=2) /* ty=Tensor[(128, 1, 1), float32] */;
  %117 = add(%112, %116) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %118 = nn.relu(%117) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %119 = nn.conv2d(%118, %stage2_unit2_conv1_weight, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3]) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %120 = add(%stage2_unit2_bn2_moving_var, 2e-05f /* ty=float32 */) /* ty=Tensor[(128), float32] */;
  %121 = sqrt(%120) /* ty=Tensor[(128), float32] */;
  %122 = divide(1f /* ty=float32 */, %121) /* ty=Tensor[(128), float32] */;
  %123 = multiply(%122, %stage2_unit2_bn2_gamma) /* ty=Tensor[(128), float32] */;
  %124 = expand_dims(%123, axis=1, num_newaxis=2) /* ty=Tensor[(128, 1, 1), float32] */;
  %125 = multiply(%119, %124) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %126 = negative(%stage2_unit2_bn2_moving_mean) /* ty=Tensor[(128), float32] */;
  %127 = multiply(%126, %123) /* ty=Tensor[(128), float32] */;
  %128 = add(%127, %stage2_unit2_bn2_beta) /* ty=Tensor[(128), float32] */;
  %129 = expand_dims(%128, axis=1, num_newaxis=2) /* ty=Tensor[(128, 1, 1), float32] */;
  %130 = add(%125, %129) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %131 = nn.relu(%130) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %132 = nn.conv2d(%131, %stage2_unit2_conv2_weight, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3]) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %133 = add(%132, %106) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %134 = add(%stage3_unit1_bn1_moving_var, 2e-05f /* ty=float32 */) /* ty=Tensor[(128), float32] */;
  %135 = sqrt(%134) /* ty=Tensor[(128), float32] */;
  %136 = divide(1f /* ty=float32 */, %135) /* ty=Tensor[(128), float32] */;
  %137 = multiply(%136, %stage3_unit1_bn1_gamma) /* ty=Tensor[(128), float32] */;
  %138 = expand_dims(%137, axis=1, num_newaxis=2) /* ty=Tensor[(128, 1, 1), float32] */;
  %139 = multiply(%133, %138) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %140 = negative(%stage3_unit1_bn1_moving_mean) /* ty=Tensor[(128), float32] */;
  %141 = multiply(%140, %137) /* ty=Tensor[(128), float32] */;
  %142 = add(%141, %stage3_unit1_bn1_beta) /* ty=Tensor[(128), float32] */;
  %143 = expand_dims(%142, axis=1, num_newaxis=2) /* ty=Tensor[(128, 1, 1), float32] */;
  %144 = add(%139, %143) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %145 = nn.relu(%144) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %146 = nn.conv2d(%145, %stage3_unit1_conv1_weight, strides=[2, 2], padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3]) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %147 = add(%stage3_unit1_bn2_moving_var, 2e-05f /* ty=float32 */) /* ty=Tensor[(256), float32] */;
  %148 = sqrt(%147) /* ty=Tensor[(256), float32] */;
  %149 = divide(1f /* ty=float32 */, %148) /* ty=Tensor[(256), float32] */;
  %150 = multiply(%149, %stage3_unit1_bn2_gamma) /* ty=Tensor[(256), float32] */;
  %151 = expand_dims(%150, axis=1, num_newaxis=2) /* ty=Tensor[(256, 1, 1), float32] */;
  %152 = multiply(%146, %151) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %153 = negative(%stage3_unit1_bn2_moving_mean) /* ty=Tensor[(256), float32] */;
  %154 = multiply(%153, %150) /* ty=Tensor[(256), float32] */;
  %155 = add(%154, %stage3_unit1_bn2_beta) /* ty=Tensor[(256), float32] */;
  %156 = expand_dims(%155, axis=1, num_newaxis=2) /* ty=Tensor[(256, 1, 1), float32] */;
  %157 = add(%152, %156) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %158 = nn.relu(%157) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %159 = nn.conv2d(%158, %stage3_unit1_conv2_weight, padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3]) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %160 = nn.conv2d(%145, %stage3_unit1_sc_weight, strides=[2, 2], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %161 = add(%159, %160) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %162 = add(%stage3_unit2_bn1_moving_var, 2e-05f /* ty=float32 */) /* ty=Tensor[(256), float32] */;
  %163 = sqrt(%162) /* ty=Tensor[(256), float32] */;
  %164 = divide(1f /* ty=float32 */, %163) /* ty=Tensor[(256), float32] */;
  %165 = multiply(%164, %stage3_unit2_bn1_gamma) /* ty=Tensor[(256), float32] */;
  %166 = expand_dims(%165, axis=1, num_newaxis=2) /* ty=Tensor[(256, 1, 1), float32] */;
  %167 = multiply(%161, %166) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %168 = negative(%stage3_unit2_bn1_moving_mean) /* ty=Tensor[(256), float32] */;
  %169 = multiply(%168, %165) /* ty=Tensor[(256), float32] */;
  %170 = add(%169, %stage3_unit2_bn1_beta) /* ty=Tensor[(256), float32] */;
  %171 = expand_dims(%170, axis=1, num_newaxis=2) /* ty=Tensor[(256, 1, 1), float32] */;
  %172 = add(%167, %171) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %173 = nn.relu(%172) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %174 = nn.conv2d(%173, %stage3_unit2_conv1_weight, padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3]) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %175 = add(%stage3_unit2_bn2_moving_var, 2e-05f /* ty=float32 */) /* ty=Tensor[(256), float32] */;
  %176 = sqrt(%175) /* ty=Tensor[(256), float32] */;
  %177 = divide(1f /* ty=float32 */, %176) /* ty=Tensor[(256), float32] */;
  %178 = multiply(%177, %stage3_unit2_bn2_gamma) /* ty=Tensor[(256), float32] */;
  %179 = expand_dims(%178, axis=1, num_newaxis=2) /* ty=Tensor[(256, 1, 1), float32] */;
  %180 = multiply(%174, %179) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %181 = negative(%stage3_unit2_bn2_moving_mean) /* ty=Tensor[(256), float32] */;
  %182 = multiply(%181, %178) /* ty=Tensor[(256), float32] */;
  %183 = add(%182, %stage3_unit2_bn2_beta) /* ty=Tensor[(256), float32] */;
  %184 = expand_dims(%183, axis=1, num_newaxis=2) /* ty=Tensor[(256, 1, 1), float32] */;
  %185 = add(%180, %184) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %186 = nn.relu(%185) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %187 = nn.conv2d(%186, %stage3_unit2_conv2_weight, padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3]) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %188 = add(%187, %161) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %189 = add(%stage4_unit1_bn1_moving_var, 2e-05f /* ty=float32 */) /* ty=Tensor[(256), float32] */;
  %190 = sqrt(%189) /* ty=Tensor[(256), float32] */;
  %191 = divide(1f /* ty=float32 */, %190) /* ty=Tensor[(256), float32] */;
  %192 = multiply(%191, %stage4_unit1_bn1_gamma) /* ty=Tensor[(256), float32] */;
  %193 = expand_dims(%192, axis=1, num_newaxis=2) /* ty=Tensor[(256, 1, 1), float32] */;
  %194 = multiply(%188, %193) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %195 = negative(%stage4_unit1_bn1_moving_mean) /* ty=Tensor[(256), float32] */;
  %196 = multiply(%195, %192) /* ty=Tensor[(256), float32] */;
  %197 = add(%196, %stage4_unit1_bn1_beta) /* ty=Tensor[(256), float32] */;
  %198 = expand_dims(%197, axis=1, num_newaxis=2) /* ty=Tensor[(256, 1, 1), float32] */;
  %199 = add(%194, %198) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %200 = nn.relu(%199) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %201 = nn.conv2d(%200, %stage4_unit1_conv1_weight, strides=[2, 2], padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3]) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %202 = add(%stage4_unit1_bn2_moving_var, 2e-05f /* ty=float32 */) /* ty=Tensor[(512), float32] */;
  %203 = sqrt(%202) /* ty=Tensor[(512), float32] */;
  %204 = divide(1f /* ty=float32 */, %203) /* ty=Tensor[(512), float32] */;
  %205 = multiply(%204, %stage4_unit1_bn2_gamma) /* ty=Tensor[(512), float32] */;
  %206 = expand_dims(%205, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %207 = multiply(%201, %206) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %208 = negative(%stage4_unit1_bn2_moving_mean) /* ty=Tensor[(512), float32] */;
  %209 = multiply(%208, %205) /* ty=Tensor[(512), float32] */;
  %210 = add(%209, %stage4_unit1_bn2_beta) /* ty=Tensor[(512), float32] */;
  %211 = expand_dims(%210, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %212 = add(%207, %211) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %213 = nn.relu(%212) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %214 = nn.conv2d(%213, %stage4_unit1_conv2_weight, padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3]) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %215 = nn.conv2d(%200, %stage4_unit1_sc_weight, strides=[2, 2], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %216 = add(%214, %215) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %217 = add(%stage4_unit2_bn1_moving_var, 2e-05f /* ty=float32 */) /* ty=Tensor[(512), float32] */;
  %218 = sqrt(%217) /* ty=Tensor[(512), float32] */;
  %219 = divide(1f /* ty=float32 */, %218) /* ty=Tensor[(512), float32] */;
  %220 = multiply(%219, %stage4_unit2_bn1_gamma) /* ty=Tensor[(512), float32] */;
  %221 = expand_dims(%220, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %222 = multiply(%216, %221) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %223 = negative(%stage4_unit2_bn1_moving_mean) /* ty=Tensor[(512), float32] */;
  %224 = multiply(%223, %220) /* ty=Tensor[(512), float32] */;
  %225 = add(%224, %stage4_unit2_bn1_beta) /* ty=Tensor[(512), float32] */;
  %226 = expand_dims(%225, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %227 = add(%222, %226) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %228 = nn.relu(%227) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %229 = nn.conv2d(%228, %stage4_unit2_conv1_weight, padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3]) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %230 = add(%stage4_unit2_bn2_moving_var, 2e-05f /* ty=float32 */) /* ty=Tensor[(512), float32] */;
  %231 = sqrt(%230) /* ty=Tensor[(512), float32] */;
  %232 = divide(1f /* ty=float32 */, %231) /* ty=Tensor[(512), float32] */;
  %233 = multiply(%232, %stage4_unit2_bn2_gamma) /* ty=Tensor[(512), float32] */;
  %234 = expand_dims(%233, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %235 = multiply(%229, %234) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %236 = negative(%stage4_unit2_bn2_moving_mean) /* ty=Tensor[(512), float32] */;
  %237 = multiply(%236, %233) /* ty=Tensor[(512), float32] */;
  %238 = add(%237, %stage4_unit2_bn2_beta) /* ty=Tensor[(512), float32] */;
  %239 = expand_dims(%238, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %240 = add(%235, %239) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %241 = nn.relu(%240) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %242 = nn.conv2d(%241, %stage4_unit2_conv2_weight, padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3]) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %243 = add(%242, %216) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %244 = add(%bn1_moving_var, 2e-05f /* ty=float32 */) /* ty=Tensor[(512), float32] */;
  %245 = sqrt(%244) /* ty=Tensor[(512), float32] */;
  %246 = divide(1f /* ty=float32 */, %245) /* ty=Tensor[(512), float32] */;
  %247 = multiply(%246, %bn1_gamma) /* ty=Tensor[(512), float32] */;
  %248 = expand_dims(%247, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %249 = multiply(%243, %248) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %250 = negative(%bn1_moving_mean) /* ty=Tensor[(512), float32] */;
  %251 = multiply(%250, %247) /* ty=Tensor[(512), float32] */;
  %252 = add(%251, %bn1_beta) /* ty=Tensor[(512), float32] */;
  %253 = expand_dims(%252, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %254 = add(%249, %253) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %255 = nn.relu(%254) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %256 = nn.global_avg_pool2d(%255) /* ty=Tensor[(1, 512, 1, 1), float32] */;
  %257 = nn.batch_flatten(%256) /* ty=Tensor[(1, 512), float32] */;
  %258 = nn.dense(%257, %fc1_weight, units=1000) /* ty=Tensor[(1, 1000), float32] */;
  %259 = nn.bias_add(%258, %fc1_bias, axis=-1) /* ty=Tensor[(1, 1000), float32] */;
  nn.softmax(%259) /* ty=Tensor[(1, 1000), float32] */
}
"#;

    let module = tvm::ir::module::IRModule::parse("", relay).unwrap();

    let (expr, shapes_vec, _) = glenside::language::from_relay::from_relay(&module, false, &vec![]);

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

    egraph.add_expr(&expr);
}
