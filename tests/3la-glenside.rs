#![cfg(feature = "tvm")]

use egg::{EGraph, Extractor, Runner};
use glenside::extraction::AcceleratorCostFunction;
use glenside::language::MyAnalysis;
use std::collections::HashMap;

#[test]
#[ignore = "Mike says this is handled by the ResMLP test."]
fn test_3la_glenside_linear_rewrite() {
    let prog_frag = r#"
    #[version = "0.0.5"]
    def @main(%data: Tensor[(10, 8), float32], %weight: Tensor[(16, 8), float32], %bias: Tensor[(16), float32]) -> Tensor[(1, 10, 16), float32] {
        %0 = nn.dense(%data, %weight, units=None) /* ty=Tensor[(10, 16), float32] */;
        %1 = reshape(%0, newshape=[1, 10, 16]) /* ty=Tensor[(1, 10, 16), float32] */;
        add(%1, %bias) /* ty=Tensor[(1, 10, 16), float32] */
      }
    "#;

    /*
    let rewritten_prog = r#"
    #[version = "0.0.5"]
        def @main(%x: Tensor[(10, 8), float32], %w: Tensor[(16, 8), float32], %bias: Tensor[(16), float32]) -> Tensor[(1, 10, 16), float32] {
        %0 = nn.dense(%x, %w, units=None) /* ty=Tensor[(10, 16), float32] */
;
    %1 = nn.bias_add(%0, %bias) /* ty=Tensor[(10, 16), float32] */
;
    reshape(%1, newshape=[1, 10, 16]) /* ty=Tensor[(1, 10, 16), float32] */
        }
    "#;

    let linear_pattern = r#"
    #[version = "0.0.5"]
    def @main(%data: Tensor[(10, 8), float32], %weight: Tensor[(16, 8), float32], %bias: Tensor[(16), float32]) -> Tensor[(10, 16), float32] {
    %0 = nn.dense(%data, %weight, units=None) /* ty=Tensor[(10, 16), float32] */
;
    nn.bias_add(%0, %bias) /* ty=Tensor[(10, 16), float32] */
      }
    "#;*/

    let prog_frag_mod = tvm::ir::IRModule::parse("", prog_frag).unwrap();
    let (expr, shape_info, dtypes_info, equiv_worklist) =
        glenside::language::from_relay::from_relay(&prog_frag_mod, false, &vec![]);

    let mut env = HashMap::default();
    for (name, shape) in &shape_info {
        env.insert(name.clone(), shape.clone());
    }
    let mut egraph = EGraph::new(MyAnalysis {
        name_to_shape: env.clone(),
        name_to_dtype: dtypes_info.into_iter().collect(),
    });
    let mut rws = vec![glenside::language::rewrites::linear_layer_accelerator_rewrites()];
    rws.extend(glenside::language::rewrites::bubble_reshape_through_linear_generalized());
    let (id, id_map) = egraph.add_expr_with_record(&expr);
    for (left, right) in equiv_worklist {
        if let (Some(&new_left), Some(&new_right)) = (id_map.get(&left), id_map.get(&right)) {
            egraph.union(new_left, new_right);
        } else {
            let nodes = expr.as_ref();
            println!(
                "{:?} v.s. {:?}",
                nodes[usize::from(left)],
                nodes[usize::from(right)]
            );
        }
    }
    let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
        .with_egraph(egraph)
        .with_time_limit(std::time::Duration::from_secs(5))
        .with_node_limit(500000)
        .with_iter_limit(40)
        .run(&rws);
    println!("Finished");
    runner
        .egraph
        .dot(&|_x, _y| true)
        .to_svg("/home/dh63/marlowe/smoke-test/glenside/render_egraph.svg")
        .unwrap();
    println!("{}", runner.egraph.record().to_record_instructions(id));
    let extractor = Extractor::new(
        &runner.egraph,
        AcceleratorCostFunction(runner.egraph.total_size() as f64),
    );
    let (_cost, best) = extractor.find_best(id);
    println!("{}", best.pretty(80));
}
