#![cfg(feature = "tvm")]
use egg::{EGraph, Extractor, Runner};
use glenside::extraction::AcceleratorCostFunction;
use glenside::language::MyAnalysis;
use std::collections::HashMap;
use std::path::PathBuf;

#[test]
fn test_resnet_flexmatch() {
    let filename = PathBuf::from(format!(
        "{}/models/resnet.relay",
        env!("CARGO_MANIFEST_DIR")
    ));
    let relay = std::fs::read_to_string(&filename).unwrap();
    let module = tvm::ir::module::IRModule::parse("", relay).unwrap();
    let (expr, shape_info, dtypes_info) = glenside::language::from_relay::from_relay(
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
        name_to_dtype: dtypes_info.iter().cloned().collect(),
    });
    let mut rws = vec![
        // glenside::language::rewrites::bubble_reshape_through_linear_generalized(),
        glenside::language::rewrites::access_reshape_to_relay(),
        glenside::language::rewrites::linear_layer_accelerator_rewrites(),
        glenside::language::rewrites::flatten_unflatten_any_access(),
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
    let _model = best.pretty(80);
    // println!("{}", model);
    // println!("{}", _cost);
}
