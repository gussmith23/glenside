#![cfg(feature = "tvm")]

use egg::EGraph;
use glenside::language::{MyAnalysis, MyAnalysisData};
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
    let id = egraph.add_expr(&expr);
    assert_eq!(
        vec![vec![1, 4, 15130], vec![1, 81, 15130]],
        match &egraph[id].data {
            MyAnalysisData::Tuple(a) => match a.as_slice() {
                [MyAnalysisData::AccessPattern(t1), MyAnalysisData::AccessPattern(t2)] =>
                    vec![t1.as_vec(), t2.as_vec()],
                _ => panic!(),
            },
            _ => panic!(),
        }
    );
}
