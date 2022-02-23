#![cfg(feature = "tvm")]
use egg::{EGraph, Extractor, Runner};
use glenside::extraction::AcceleratorCostFunction;
use glenside::language::{serialize_analysis_data, MyAnalysis};
use std::collections::HashMap;
use std::path::PathBuf;

#[test]
fn test_conv1d_flexmatch() {
    // let relay = r#"
    // #[version = "0.0.5"]
    // def @main(%data: Tensor[(1, 3, 32), float32], %weights: Tensor[(8, 3, 3), float32]) -> Tensor[(1, 8, 19), float32] {
    //     nn.conv1d(%data, %weights, strides=[2], padding=[3, 4]) /* ty=Tensor[(1, 8, 19), float32] */
    // }
    // "#;
    let relay = r#"
    #[version = "0.0.5"]
    def @main(%data: Tensor[(1, 3, 32, 32), float32], %weights: Tensor[(2, 3, 16, 16), float32]) -> Tensor[(1, 2, 13, 13), float32] {
        nn.conv2d(%data, %weights, strides=[2, 2], padding=[4, 4, 4, 4]) /* ty=Tensor[(1, 2, 13, 13), float32] */
    }
    "#;
    // let filename = PathBuf::from(format!(
    //     "{}/models/resmlp.relay",
    //     env!("CARGO_MANIFEST_DIR")
    // ));
    // let relay = std::fs::read_to_string(&filename).unwrap();
    let module = tvm::ir::module::IRModule::parse("", relay).unwrap();
    let (expr, shape_info, equiv_worklist) =
        glenside::language::from_relay::from_relay(&module, false, &vec![]);
    let mut env = HashMap::default();
    for (name, shape) in &shape_info {
        env.insert(name.clone(), shape.clone());
    }
    // println!("{:?}", env);
    let mut egraph = EGraph::new(MyAnalysis {
        name_to_shape: env.clone(),
    });
    let rws = vec![
        glenside::language::rewrites::flatten_unflatten_any_access(),
        glenside::language::rewrites::access_reshape_to_relay(),
        glenside::language::rewrites::bubble_reshape_through_cartesian_product(),
        glenside::language::rewrites::bubble_reshape_through_compute_dot_product(),
        // glenside::language::rewrites::flatten_dot_product_to_dense(),
        glenside::language::rewrites::dot_product_with_vta(),
        // glenside::language::rewrites::dot_product_to_linear(),
        // glenside::language::rewrites::linear_layer_accelerator_rewrites(),
        //    glenside::language::rewrites::bubble_reshape_through_compute_dot_product(),
        //    glenside::language::rewrites::bubble_reshape_through_linear_generalized(),
        //    glenside::language::rewrites::linear_layer_accelerator_rewrites(),
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
    let extractor = Extractor::new(&runner.egraph, AcceleratorCostFunction {});
    let (_cost, best) = extractor.find_best(id);
    // let json_dump = best.serialize();
    let model = best.pretty(80);
    println!("{}", model);
    println!("{}", _cost);
    let json_dump = best.serialize();
    let output_file = PathBuf::from(format!("{}/models/conv2d.json", env!("CARGO_MANIFEST_DIR")));
    let _ = std::fs::write(output_file, json_dump.to_string()).unwrap();
    egraph = EGraph::new(MyAnalysis {
        name_to_shape: env.clone(),
    });
    let (_, id_map) = egraph.add_expr_with_record(&best);
    let mut native_map = HashMap::new();
    for (k, v) in id_map.into_iter() {
        native_map.insert(k, v);
    }
    let data_json_dump = serialize_analysis_data(&egraph, &native_map);
    let data_output = PathBuf::from(format!(
        "{}/models/conv2d_data.json",
        env!("CARGO_MANIFEST_DIR")
    ));
    let _ = std::fs::write(data_output, data_json_dump.to_string()).unwrap();
    // let output_file = PathBuf::from(format!("{}/models/resmlp-rewrite", env!("CARGO_MANIFEST_DIR")));
    // let _ = std::fs::write(output_file, model).unwrap();
    assert!(false);
}
