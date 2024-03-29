#![cfg(feature = "tvm")]
use egg::{EGraph, Extractor, Runner};
use glenside::extraction::AcceleratorCostFunction;
use glenside::language::MyAnalysis;
use std::collections::HashMap;

#[test]
fn test_conv1d_flexmatch() {
    let relay = r#"
    #[version = "0.0.5"]
    def @main(%data: Tensor[(1, 3, 32), float32], %weights: Tensor[(8, 3, 3), float32]) -> Tensor[(1, 8, 19), float32] {
        nn.conv1d(%data, %weights, strides=[2], padding=[3, 4]) /* ty=Tensor[(1, 8, 19), float32] */
    }
    "#;
    // let relay = r#"
    // #[version = "0.0.5"]
    // def @main(%data: Tensor[(1, 3, 32, 32), float32], %weights: Tensor[(2, 3, 16, 16), float32]) -> Tensor[(1, 2, 13, 13), float32] {
    //     nn.conv2d(%data, %weights, strides=[2, 2], padding=[4, 4, 4, 4]) /* ty=Tensor[(1, 2, 13, 13), float32] */
    // }
    // "#;
    let module = tvm::ir::module::IRModule::parse("", relay).unwrap();
    let (expr, shape_info, dtype_info) = glenside::language::from_relay::from_relay(
        &module,
        false,
        &glenside::language::RELAY_OPS.into(),
    );
    let mut env = HashMap::default();
    for (name, shape) in &shape_info {
        env.insert(name.clone(), shape.clone());
    }
    let mut egraph = EGraph::new(MyAnalysis {
        name_to_shape: env.clone(),
        name_to_dtype: dtype_info.iter().cloned().collect(),
    });
    let mut rws = vec![
        glenside::language::rewrites::flatten_unflatten_any_access(),
        glenside::language::rewrites::access_reshape_to_relay(),
        glenside::language::rewrites::bubble_reshape_through_cartesian_product(),
        glenside::language::rewrites::bubble_reshape_through_compute_dot_product(),
        glenside::language::rewrites::dot_product_with_vta(),
    ];
    rws.append(&mut glenside::language::rewrites::relay_to_glenside_rewrites());
    let id = egraph.add_expr(&expr);
    egraph.rebuild();
    let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
        .with_egraph(egraph)
        .with_time_limit(std::time::Duration::from_secs(5))
        .with_node_limit(500000)
        .with_iter_limit(40)
        .run(&rws);
    let extractor = Extractor::new(
        &runner.egraph,
        AcceleratorCostFunction(runner.egraph.total_size() as f64),
    );
    let (_cost, best) = extractor.find_best(id);
    // let json_dump = best.serialize();
    let model = best.pretty(80);
    println!("{}", model);
    println!("{}", _cost);
}
