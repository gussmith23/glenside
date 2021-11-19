#![cfg(feature = "tvm")]

use std::{collections::HashMap, path::PathBuf};

use egg::EGraph;
use glenside::language::{MyAnalysis, MyAnalysisData};

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
        &vec![
            glenside::language::RelayOperator::RelayStridedSlice,
            glenside::language::RelayOperator::RelaySoftmax,
            glenside::language::RelayOperator::RelayAdd,
            glenside::language::RelayOperator::RelayDropout,
            glenside::language::RelayOperator::RelayMultiply,
        ],
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
        vec![20, 32, 256],
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => a.as_vec(),
            _ => panic!(),
        }
    );

    // Currently, these checks won't work, as merge() is not fully working. Mike
    // has a hack where he manually figures this out later.

    // let runner = Runner::default()
    //     .with_egraph(egraph)
    //     .with_node_limit(1000000)
    //     .run(vec![&glenside::language::rewrites::dot_product_with_vta()]);
    // runner.print_report();

    // assert!(
    //     "(accelerator-call vta-dense ?x ?w ?shape)"
    //         .parse::<Pattern<_>>()
    //         .unwrap()
    //         .search(&runner.egraph)
    //         .len()
    //         > 0
    // );

    // let matches ="(accelerator-call vta-dense ?x ?w ?shape)"
    //         .parse::<Pattern<_>>()
    //         .unwrap()
    //         .search(&runner.egraph);
    // assert!(matches.len() > 0);
    // println!("{:#?}", &runner.egraph[matches[0].eclass]);
    // assert!(matches.iter().all(|m| match &runner.egraph[m.eclass].data {
    //     MyAnalysisData::AccessPattern(a) => a.contains_accelerator_calls,
    //     _ => panic!(),
    // }));

    // let matches ="(access-insert-axis (accelerator-call vta-dense ?x ?w ?shape) 0)"
    //         .parse::<Pattern<_>>()
    //         .unwrap()
    //         .search(&runner.egraph);
    // assert!(matches.len() > 0);
    // println!("{:#?}", &runner.egraph[matches[0].eclass]);
    // println!("{:#?}", &runner.egraph[runner.egraph[matches[0].eclass].nodes[0].children()[0]]);
    // assert!(matches.iter().any(|m| match &runner.egraph[m.eclass].data {
    //     MyAnalysisData::AccessPattern(a) => a.contains_accelerator_calls,
    //     _ => panic!(),
    // }));

    // let matches ="(access-concatenate (access-insert-axis (accelerator-call vta-dense ?x ?w ?shape) 0) ?second ?dim)"
    //         .parse::<Pattern<_>>()
    //         .unwrap()
    //         .search(&runner.egraph);
    // assert!(matches.len() > 0);
    // println!("{:#?}", &runner.egraph[matches[0].eclass]);
    // assert!(matches.iter().all(|m| match &runner.egraph[m.eclass].data {
    //     MyAnalysisData::AccessPattern(a) => a.contains_accelerator_calls,
    //     _ => panic!(),
    // }));
}
