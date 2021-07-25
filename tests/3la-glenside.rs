#![cfg(feature = "tvm")]

use egg::{EGraph, Runner};
use glenside::language::MyAnalysis;
use std::collections::HashMap;

#[test]
fn test_3la_glenside_linear_rewrite() {
    let prog_frag = r#"
    #[version = "0.0.5"]
    def @main(%data: Tensor[(10, 8), float32], %weight: Tensor[(16, 8), float32], %bias: Tensor[(16), float32]) -> Tensor[(1, 10, 16), float32] {
        %0 = nn.dense(%data, %weight, units=None) /* ty=Tensor[(10, 16), float32] */;
        %1 = reshape(%0, newshape=[1, 10, 16]) /* ty=Tensor[(1, 10, 16), float32] */;
        add(%1, %bias) /* ty=Tensor[(1, 10, 16), float32] */
      }
    "#;

    let linear_pattern = r#"
    #[version = "0.0.5"]
    def @main(%data: Tensor[(10, 8), float32], %weight: Tensor[(16, 8), float32], %bias: Tensor[(16), float32]) -> Tensor[(10, 16), float32] {
        %0 = nn.dense(%data, %weight, units=None) /* ty=Tensor[(10, 16), float32] */;
        nn.bias_add(%0, %bias) /* ty=Tensor[(10, 16), float32] */
      }
    "#;

    let prog_frag_mod = tvm::ir::IRModule::parse("", prog_frag).unwrap();
    let (expr, shape_info, equiv_worklist) = glenside::language::from_relay::from_relay(&prog_frag_mod, false, &vec![]);

    let mut env = HashMap::default();
    for (name, shape) in &shape_info {
        env.insert(name.clone(), shape.clone());
    }
    let mut egraph = EGraph::new(MyAnalysis {
        name_to_shape: env.clone(),
    });
    // let mut egraph = EGraph::new(MyAnalysis {name_to_shape: HashMap::default()});
    let rws = vec![
        // glenside::language::rewrites::bubble_reshape_through_linear()
        glenside::language::rewrites::relay_dense_rewrite()
    ];
    // let id_prog = egraph.add_expr(&expr);
    // let expr = "(relay-operator-call 
    //             relay-bias-add 
    //             (relay-operator-call 
    //                 relay-dense
    //                 (access x 1)
    //                 (access w 1))
    //             (access bias 0))".parse().unwrap();
    egraph.add_expr(&expr);
    for (left, right, left_expr, right_expr) in equiv_worklist {
        // println!("left: {}({}), right: {}({})", egraph.find(left), left, egraph.find(right), right);
        if let (Some(left_id), Some(right_id)) = (egraph.lookup(left_expr.clone()), egraph.lookup(right_expr.clone())) {
            println!("left id: {}; right id: {}", left_id, right_id);
            egraph.union(left_id, right_id);
        }
        // egraph.union(egraph.find(left), egraph.find(right));
    }
    egraph.rebuild();
    // let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
    //     .with_egraph(egraph)
    //     .with_time_limit(std::time::Duration::from_secs(10))
    //     .with_node_limit(500000)
    //     .with_iter_limit(40)
    //     .run(&rws);

    // // runner.print_report();
    // println!("egraph:\n");
    // println!("{}", runner.egraph.dot());
    egraph.dot().to_png("/home/dh63/marlowe/smoke-test/glenside/render_egraph.png").unwrap();
}