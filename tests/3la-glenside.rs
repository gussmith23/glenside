#![cfg(feature = "tvm")]

use egg::EGraph;
use glenside::language::MyAnalysis;
use std::collections::HashMap;

#[test]
fn test_3la_glenside_linear_rewrite() {
    let prog_frag = r#"
    def @main(%data: Tensor[(10, 8), float32], %weight: Tensor[(16, 8), float32], %bias: Tensor[(16), float32]) -> Tensor[(1, 10, 16), float32] {
        %0 = nn.dense(%data, %weight, units=None) /* ty=Tensor[(10, 16), float32] */;
        %1 = reshape(%0, newshape=[1, 10, 16]) /* ty=Tensor[(1, 10, 16), float32] */;
        add(%1, %bias) /* ty=Tensor[(1, 10, 16), float32] */
      }
    "#;

    let linear_pattern = r#"
    def @main(%data: Tensor[(10, 8), float32], %weight: Tensor[(16, 8), float32], %bias: Tensor[(16), float32]) -> Tensor[(10, 16), float32] {
        %0 = nn.dense(%data, %weight, units=None) /* ty=Tensor[(10, 16), float32] */;
        nn.bias_add(%0, %bias) /* ty=Tensor[(10, 16), float32] */
      }
    "#;

    let prog_frag_mod = tvm::ir::IRModule::parse("", prog_frag);
    let (expr, shape_info) = glenside::language::from_relay::from_relay(&prog_frag_mod, false, &vec![]);

    let mut env = HashMap::default();
    for (name, shape) in &shape_info {
        env.insert(name, shape);
    }
    let mut egraph = EGraph::new(MyAnalysis {
        name_to_shape: env.clone(),
    });
    let rws = vec![
        glenside::language::rewrites::bubble_reshape_through_linear()
    ]
    let id_prog = egraph.add_expr(&expr);
    let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
        .with_egraph(egraph)
        .with_time_limit(std::time::Duration::from_secs(10))
        .with_node_limit(500000)
        .with_iter_limit(40)
        .run(&rws);

    runner.print_report();
}