#![cfg(feature = "tvm")]
use egg::{EGraph, Extractor, Runner};
use glenside::extraction::AcceleratorCostFunction;
use glenside::language::{serialize_analysis_data, MyAnalysis};
use std::collections::HashMap;
use std::path::PathBuf;

#[test]
fn test_resmlp() {
    let filename = PathBuf::from(format!(
        "{}/models/resmlp.relay",
        env!("CARGO_MANIFEST_DIR")
    ));
    let relay = std::fs::read_to_string(&filename).unwrap();
    let module = tvm::ir::module::IRModule::parse("", relay).unwrap();
    let (expr, shape_info, dtype_info, equiv_worklist) =
        glenside::language::from_relay::from_relay(&module, false, &vec![]);
    let mut env = HashMap::default();
    for (name, shape) in &shape_info {
        env.insert(name.clone(), shape.clone());
    }
    let mut egraph = EGraph::new(MyAnalysis {
        name_to_shape: env.clone(),
        name_to_dtype: dtype_info.iter().cloned().collect(),
    });
    let rws = vec![
        //    glenside::language::rewrites::bubble_reshape_through_compute_dot_product(),
        glenside::language::rewrites::bubble_reshape_through_linear_generalized(),
        glenside::language::rewrites::linear_layer_accelerator_rewrites(),
        glenside::language::rewrites::access_reshape_to_relay(),
    ];
    let (id, id_map) = egraph.add_expr_with_record(&expr);
    for (left, right) in equiv_worklist {
        if let (Some(&new_left), Some(&new_right)) = (id_map.get(&left), id_map.get(&right)) {
            egraph.union(new_left, new_right);
        }
    }
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
    // let model = best.pretty(80);
    println!("{}", best.pretty(80));
    // let output_file = PathBuf::from(format!("{}/models/resmlp-rewrite", env!("CARGO_MANIFEST_DIR")));
    // let _ = std::fs::write(output_file, model).unwrap();
    let json_dump = best.serialize();
    let output_file = PathBuf::from(format!(
        "{}/models/resmlp-dump.json",
        env!("CARGO_MANIFEST_DIR")
    ));
    let _ = std::fs::write(output_file, json_dump.to_string()).unwrap();
    egraph = EGraph::new(MyAnalysis {
        name_to_shape: env.clone(),
        name_to_dtype: dtype_info.into_iter().collect(),
    });
    let (_, id_map) = egraph.add_expr_with_record(&best);
    let mut native_map = HashMap::new();
    for (k, v) in id_map.into_iter() {
        native_map.insert(k, v);
    }
    let _data_json_dump = serialize_analysis_data(&egraph, &native_map);
}
