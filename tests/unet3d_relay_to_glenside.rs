#![cfg(feature = "tvm")]

use egg::EGraph;
use glenside::language::{MyAnalysis, RelayOperator};
use std::collections::HashMap;
use std::path::PathBuf;

#[test]
#[should_panic="not yet implemented: nn.conv3d_transpose operator not implemented"]
fn parse_unet3d() {
    let filename = PathBuf::from(format!(
        "{}/models/unet3d.relay",
        env!("CARGO_MANIFEST_DIR")
    ));
    let relay = std::fs::read_to_string(&filename).unwrap();
    let module = tvm::ir::module::IRModule::parse("", relay).unwrap();

    let (expr, shapes_vec, dtypes_vec) = glenside::language::from_relay::from_relay(
        &module,
        true,
        &vec![RelayOperator::RelayConv3D]
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
