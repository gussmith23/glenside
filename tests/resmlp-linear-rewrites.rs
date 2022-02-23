#![cfg(feature = "tvm")]

use egg::{EGraph, Runner, Extractor};
use glenside::language::MyAnalysis;
use glenside::extraction::{AcceleratorCostFunction};
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
    let (expr, shape_info, equiv_worklist) = glenside::language::from_relay::from_relay(&module, false, &vec![]);
    let mut env = HashMap::default();
    for (name, shape) in &shape_info {
        env.insert(name.clone(), shape.clone());
    }
    let mut egraph = EGraph::new(MyAnalysis {
        name_to_shape: env.clone(),
    });
    let rws = vec![
       glenside::language::rewrites::bubble_reshape_through_compute_dot_product(),
       glenside::language::rewrites::bubble_access_slice_through_access_pad_inequal_axes(),
       glenside::language::rewrites::bubble_reshape_through_linear_generalized(),
       glenside::language::rewrites::linear_layer_accelerator_rewrites(),
    ];
    let (id, id_map) = egraph.add_expr_with_record(&expr);
    for (left, right) in equiv_worklist {
        if let (Some(&new_left), Some(&new_right)) = (id_map.get(&left), id_map.get(&right)) {
            egraph.union(new_left, new_right);
        } else {
            let nodes = expr.as_ref();
            println!("{:?} v.s. {:?}", nodes[usize::from(left)], nodes[usize::from(right)]);
        }
    }
} 