#![cfg(feature = "tvm")]

use egg::EGraph;
use glenside::language::MyAnalysis;
use std::collections::HashMap;
use std::path::PathBuf;

#[test]
fn parse_ssd_r34() {
    let filename = PathBuf::from(format!(
        "{}/models/ssd_r34.relay",
        env!("CARGO_MANIFEST_DIR")
    ));
    let relay = std::fs::read_to_string(&filename).unwrap();
    let module = tvm::ir::module::IRModule::parse("", relay).unwrap();

    let (expr, shapes_vec, dtypes_vec) = glenside::language::from_relay::from_relay(
        &module,
        true,
        &glenside::language::RELAY_OPS.into(),
    );

    let mut env = HashMap::default();
    for (k, v) in &shapes_vec {
        env.insert(k.clone(), v.clone());
    }

    let mut egraph = EGraph::new(MyAnalysis {
        name_to_shape: env.clone(),
        name_to_dtype: dtypes_vec.into_iter().collect(),
    });
    egraph.add_expr(&expr);
}
