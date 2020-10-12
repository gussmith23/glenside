#![cfg(feature = "tvm")]

use egg::EGraph;
use egg::Extractor;
use egg::Pattern;
use egg::Runner;
use egg::Searcher;
use glenside::extraction::ilp::create_generic_egraph_lp_model;
use glenside::extraction::ilp::into_recexpr;
use glenside::extraction::MonolithicCostFunction;
use glenside::language::rewrites::PadLocation;
use glenside::language::rewrites::PadSliceStrategy;
use glenside::language::rewrites::SliceConcatenateStrategy;
use glenside::language::MyAnalysis;
use glenside::language::PadType;
use lp_modeler::dsl::LpBinary;
use lp_modeler::dsl::LpExpression;
use lp_modeler::dsl::LpObjective;
use lp_modeler::solvers::GurobiSolver;
use lp_modeler::solvers::SolverTrait;
use std::collections::HashMap;
use std::path::PathBuf;

// Mobilenet, simplified for inference (so batch norms are removed).
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
//
// TODO(@gussmith23) Shouldn't always panic.
// Panics at the moment because we can't actually handle the size of mobilenet.
#[should_panic]
#[test]
fn mobilenet_try_to_run_rewrites() {
    let filename = PathBuf::from(format!(
        "{}/models/mobilenet-simplified-for-inference.relay",
        env!("CARGO_MANIFEST_DIR")
    ));
    let relay = std::fs::read_to_string(&filename).unwrap();
    let module = tvm::ir::module::IRModule::parse("", relay);

    let (expr, shapes_vec) = glenside::language::from_relay::from_relay(&module, false);

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
    let id = egraph.add_expr(&expr);

    let rws = vec![
        glenside::language::rewrites::flatten_unflatten_any_access(),
        glenside::language::rewrites::bubble_reshape_through_cartesian_product(),
        glenside::language::rewrites::bubble_reshape_through_compute_dot_product(),
        glenside::language::rewrites::bubble_access_concatenate_through_access_cartesian_product_not_item_axis_left(),
        glenside::language::rewrites::bubble_access_concatenate_through_access_cartesian_product_not_item_axis_right(),
        glenside::language::rewrites::bubble_access_concatenate_through_access_cartesian_product_same_item_axis(),
        glenside::language::rewrites::bubble_access_concatenate_through_compute_dot_product_item_axis(),
        glenside::language::rewrites::bubble_access_concatenate_through_compute_dot_product_not_item_axis(),
        glenside::language::rewrites::bubble_access_slice_through_access_pad_inequal_axes(),
        glenside::language::rewrites::systolic_array(),
        glenside::language::rewrites::pad_slice_accesses(
            0,
            PadSliceStrategy::PadToClosestMultipleOf {
                multiple_of: 64,
                pad_location: PadLocation::End,
                pad_type: PadType::ZeroPadding,
            },
        ),
        glenside::language::rewrites::pad_slice_accesses(
            1,
            PadSliceStrategy::PadToClosestMultipleOf {
                multiple_of: 64,
                pad_location: PadLocation::End,
                pad_type: PadType::ZeroPadding,
            },
        ),
        glenside::language::rewrites::slice_concatenate_accesses(
            0,
            SliceConcatenateStrategy::DivideInto { segment_size: 64 },
        ),
        glenside::language::rewrites::slice_concatenate_accesses(
            1,
            SliceConcatenateStrategy::DivideInto { segment_size: 64 },
        ),
        glenside::language::rewrites::bubble_access_slice_through_access_cartesian_product_not_item_axis_left(),
        glenside::language::rewrites::bubble_access_slice_through_access_cartesian_product_not_item_axis_right(),
        glenside::language::rewrites::bubble_access_slice_through_access_cartesian_product_same_item_axis(),
        glenside::language::rewrites::bubble_access_slice_through_compute_dot_product_not_item_axis(),
        glenside::language::rewrites::bubble_access_slice_through_compute_dot_product_item_axis_not_tuple_axis(),
    ];

    // TODO(@gussmith23) This is starting to become a flaky test...
    // I know the correct program can be found, but it takes time.
    let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
        .with_egraph(egraph)
        .with_time_limit(std::time::Duration::from_secs(10))
        .with_node_limit(500000)
        .with_iter_limit(40)
        .run(&rws);

    runner.print_report();

    println!("hi");
    let mut model =
        create_generic_egraph_lp_model(&runner.egraph, &[id], "mobilenet", LpObjective::Minimize);
    println!("hi");

    // TODO(@gussmith23) Figure out a better way to create optimization func
    // TODO(@gussmith23) This is written this way b/c of the stack overflowing
    let mut expr: LpExpression = LpExpression::EmptyExpr;
    for (_, name) in &model.bq_names {
        expr = expr + LpBinary::new(&name);
    }
    println!("after loop");
    model.problem += expr;
    println!("hi");
    let solver = GurobiSolver::new();
    let result = solver.run(&model.problem);

    println!("hi");
    let var_values = match result.unwrap() {
        (lp_modeler::solvers::Status::Optimal, var_values) => var_values,
        _ => panic!(),
    };
    println!("{:#?}", var_values);

    let out_expr = into_recexpr(&model, &var_values, &[id]);
    println!("{}", out_expr.pretty(80));

    // Did any tensorization happen?
    assert!(
        "(systolic-array ?a ?b ?c ?d)"
            .parse::<Pattern<_>>()
            .unwrap()
            .search(&runner.egraph)
            .len()
            > 0
    );

    // Did tensorization to 64x64 happen? (Harder than tensorizing to just
    // anything)
    assert!(
        "(systolic-array 64 64 ?c ?d)"
            .parse::<Pattern<_>>()
            .unwrap()
            .search(&runner.egraph)
            .len()
            > 0
    );

    // Can we extract something that can be turned into a hardware design?
    let _ex = Extractor::new(
        &runner.egraph,
        MonolithicCostFunction {
            systolic_array_configuration: (16, 128),
            egraph: &runner.egraph,
            prefer_systolic_arrays_with_blocking: false,
        },
    );

    // TODO(@gussmith23) This is overflowing the stack
    // See https://github.com/egraphs-good/egg/issues/50
    // let (cost, _) = ex.find_best(id);
    // assert!(cost < usize::MAX);
}
