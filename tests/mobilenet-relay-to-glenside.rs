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
#[test]
fn parse_mobilenet_simplified_for_inference() {
    let filename = PathBuf::from(format!(
        "{}/models/mobilenet-simplified-for-inference.relay",
        env!("CARGO_MANIFEST_DIR")
    ));
    let relay = std::fs::read_to_string(&filename).unwrap();
    let module = tvm::ir::module::IRModule::parse("", relay);

    let (expr, shapes_vec) = glenside::language::from_relay::from_relay(&module, false, &vec![]);

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
fn parse_mobilenet() {
    let filename = PathBuf::from(format!(
        "{}/models/mobilenet.relay",
        env!("CARGO_MANIFEST_DIR")
    ));
    let relay = std::fs::read_to_string(&filename).unwrap();
    let module = tvm::ir::module::IRModule::parse("", relay);

    let (expr, shapes_vec) = glenside::language::from_relay::from_relay(
        &module,
        true,
        &vec![glenside::language::RelayOperator::BatchNormInference],
    );

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
