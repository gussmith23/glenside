#![cfg(feature = "tvm")]
#![cfg(feature = "cplex")]

use egg::EGraph;
use egg::Pattern;
use egg::Runner;
use egg::Searcher;
use glenside::extraction::ilp::create_generic_egraph_lp_model;
use glenside::extraction::ilp::into_recexpr;
use glenside::language::rewrites::PadLocation;
use glenside::language::rewrites::PadSliceStrategy;
use glenside::language::MyAnalysis;
use glenside::language::PadType;
use log::info;
use rplex::Env;
use rplex::ObjectiveType;
use std::collections::HashMap;
use std::path::PathBuf;

#[test]
fn mobilenet_end_to_end() {
    test_logger::ensure_env_logger_initialized();

    let filename = PathBuf::from(format!(
        "{}/models/mobilenet.relay",
        env!("CARGO_MANIFEST_DIR")
    ));
    let relay = std::fs::read_to_string(&filename).unwrap();
    let module = tvm::ir::module::IRModule::parse("", relay);
    info!("parsed relay source to IRModule");

    let (expr, shapes_vec) = glenside::language::from_relay::from_relay(&module, true);
    info!("ingested Relay code into Glenside");

    let mut env = HashMap::default();
    for (k, v) in &shapes_vec {
        env.insert(k.clone(), v.clone());
    }

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
        glenside::language::rewrites::systolic_array_with_blocking(64,64),
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
        glenside::language::rewrites::bubble_access_slice_through_access_cartesian_product_not_item_axis_left(),
        glenside::language::rewrites::bubble_access_slice_through_access_cartesian_product_not_item_axis_right(),
        glenside::language::rewrites::bubble_access_slice_through_access_cartesian_product_same_item_axis(),
        glenside::language::rewrites::bubble_access_slice_through_compute_dot_product_not_item_axis(),
        glenside::language::rewrites::bubble_access_slice_through_compute_dot_product_item_axis_not_tuple_axis(),
    ];

    let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
        .with_egraph(egraph)
        .with_time_limit(std::time::Duration::from_secs(10))
        .with_node_limit(500000)
        .with_iter_limit(40)
        .run(&rws);

    runner.print_report();
    info!("rewrites complete");

    let env = Env::new().unwrap();
    let mut model = create_generic_egraph_lp_model(&env, &runner.egraph, &[id], "mobilenet");
    model
        .problem
        .set_objective_type(ObjectiveType::Minimize)
        .unwrap();
    info!("ilp problem created");

    let result = model.problem.solve().unwrap();
    info!("ilp problem solved");

    assert!(result.objective > 0.0);

    let _out_expr = into_recexpr(&model, &result.variables);
    info!("Glenside expression extracted using solution of ILP problem");

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
}
