use egg::{EGraph, Pattern, Runner, Searcher};
use glenside::language::*;

#[test]
fn regular_multilayer_perceptron() {
    test_logger::ensure_env_logger_initialized();

    let program = "
     (compute dot-product
      (access-cartesian-product
       (access
        (compute dot-product
         (access-cartesian-product
          (access
           (compute dot-product
            (access-cartesian-product
             (access (access-tensor v-32) 0)
             (access-move-axis (access (access-tensor t-32-64) 1) 1 0)
            )
           )
           0
          )
          (access-move-axis (access (access-tensor t-64-128) 1) 1 0)
         )
        )
        0
       )
       (access-move-axis (access (access-tensor t-128-16) 1) 1 0)
      )
     )
     "
    .parse()
    .unwrap();

    // We will achieve this tensorization with a set of program rewrites:
    let rws = vec![
        // These rewrites tile the program automatically.
        // Specifically, they cut every input tensor in half and concatenate
        // them back together, along both axes (0 and 1). Currently, we don't
        // break down axes that are less than 16 in length, but we can adjust
        // that later.
        rewrites::slice_concatenate_tensor_accesses(0, 16),
        rewrites::slice_concatenate_tensor_accesses(1, 16),
        // This rewrite collapses multiple slice operators (introduced by the
        // above rewrites) into one.
        rewrites::collapse_nested_access_slices(),
        // These rewrites bubble concatenate operators (also introduced by the
        // split rewrites) up to the top of the program.
        // We need to get the concatenate operators to the top so that we can
        // identify places where we can map in hardware.
        // For example, if we see:
        // (map-dot-product
        //  (cartesian-product (concatenate ...) (concatenate ...))
        // )
        // ...we don't have a hardware atom that does this. But if we can
        // rewrite it to:
        // (concatenate
        //  (map-dot-product
        //   (cartesian-product <a 1x16 vector> <a 16x16 tensor>)
        //  )
        //  (map-dot-product
        //   (cartesian-product <a 1x16 vector> <a 16x16 tensor>)
        //  )
        // )
        // ...then we can map in systolic arrays for the (map-dot-product...)
        // expressions, and the top-level concatenate will be handled by the compiler.
        rewrites::bubble_access_concatenate_through_access(),
        rewrites::bubble_access_concatenate_through_access_slice(),
        rewrites::bubble_access_concatenate_through_access_move_axis(),
        rewrites::bubble_access_concatenate_through_access_cartesian_product_not_item_axis_left(),
        rewrites::bubble_access_concatenate_through_access_cartesian_product_not_item_axis_right(),
        rewrites::bubble_access_concatenate_through_access_cartesian_product_same_item_axis(),
        rewrites::bubble_access_concatenate_through_compute_dot_product_item_axis(),
        rewrites::bubble_access_concatenate_through_compute_dot_product_not_item_axis(),
        // Finally, this rewrite tensorizes!
        // It identifies patterns that we have hardware atoms for. Right now, it
        // finds:
        // (map-dot-product
        //  (cartesian-product <a 1xrows vector> (cols <a rowsxcols tensor>))
        // )
        // and rewrites it to:
        // (bsg-systolic-array <rows> <cols> < 1xrows vector> <a rowsxcols tensor>)
        rewrites::systolic_array(),
    ];

    // Run the rewrites over the egraph.
    let mut egraph = EGraph::new(MyAnalysis);
    let id = egraph.add_expr(&program);
    let runner = Runner::<_, _, ()>::new(MyAnalysis)
        .with_egraph(egraph)
        .with_node_limit(100_000)
        .with_time_limit(std::time::Duration::from_secs(60))
        .with_iter_limit(40)
        .run(&rws);
    println!(
        "Stopped after {} iterations, reason: {:?}",
        runner.iterations.len(),
        runner.stop_reason
    );

    // Find the monolithic program.
    "(systolic-array 16 128
      (access
       (systolic-array 128 64
        (access
         (systolic-array 64 32
          (access (access-tensor v-32) 0)
          (access-move-axis (access (access-tensor t-32-64) 1) 1 0)
         )
         0
        )
        (access-move-axis (access (access-tensor t-64-128) 1) 1 0)
       )
       0
      )
      (access-move-axis (access (access-tensor t-128-16) 1) 1 0)
     )
     "
    .parse::<Pattern<_>>()
    .unwrap()
    .search_eclass(&runner.egraph, id)
    .expect("Did not find expected program");

    // TODO(@gussmith23) add more checks. I should be using extraction to do this,
    // instead of writing these out by hand.
}
