#![cfg(feature = "tvm")]

use egg::EGraph;
use glenside::language::MyAnalysis;
use std::collections::HashMap;
use std::path::PathBuf;

// Mobilenet, simplified for inference (so batch norms are simplified).
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

// Mobilenet
// Generate with:
// ```python3
// import tvm
// from tvm import relay
// from tvm.relay.testing.mobilenet import get_workload
//
// mod, _ = get_workload()
// print(mod.astext())
// ```
#[test]
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
        &vec![]
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
