#![cfg(feature = "tvm")]

use std::{collections::HashMap, path::PathBuf};

use egg::EGraph;
use glenside::language::MyAnalysis;

#[test]
fn transformer() {
    let filename = PathBuf::from(format!(
        "{}/models/transformer.relay",
        env!("CARGO_MANIFEST_DIR")
    ));
    let relay = std::fs::read_to_string(&filename).unwrap();
    let module = tvm::ir::module::IRModule::parse("", relay).unwrap();

    let (expr, shapes_vec, dtypes_vec, _) = glenside::language::from_relay::from_relay(
        &module,
        false,
        &vec![glenside::language::RelayOperator::RelayStridedSlice],
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
