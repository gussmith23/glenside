use egg::{EGraph, Pattern, Runner, Searcher};
use glenside::language::*;

#[test]
fn regular_small_multilayer_perceptron() {
    let program = "
     (map-dot-product
      (cartesian-product
       (map-dot-product
        (cartesian-product
         v-32
         (move-axis t-32-32 1 0)
        )
       )
       (move-axis t-32-32 1 0)
      )
     )
     "
    .parse()
    .unwrap();

    let rws = vec![
        rewrites::split(0, 16, true),
        rewrites::split(1, 16, true),
        rewrites::collapse_nested_slices(),
        rewrites::bubble_concatenate_through_move_axis(),
        rewrites::bubble_concatenate_through_cartesian_product_not_last_axis_left(),
        rewrites::bubble_concatenate_through_cartesian_product_not_last_axis_right(),
        rewrites::bubble_concatenate_through_cartesian_product_last_axis(),
        rewrites::bubble_concatenate_through_map_dot_product_not_last_axis(),
        rewrites::bubble_concatenate_through_map_dot_product_last_axis(),
        rewrites::slice_move_axis_composition_commutative(),
        rewrites::systolic_array_vector_matrix(),
    ];

    // Run the rewrites over the egraph.
    let mut egraph = EGraph::new(MyAnalysis);
    let id = egraph.add_expr(&program);
    let runner = Runner::<_, _, ()>::new(MyAnalysis)
        .with_egraph(egraph)
        .run(&rws);
    println!(
        "Stopped after {} iterations, reason: {:?}",
        runner.iterations.len(),
        runner.stop_reason
    );

    "(concatenate
      (elementwise-add
       (bsg-systolic-array 16 16
        (slice
         (concatenate
          (elementwise-add
           (bsg-systolic-array 16 16
            (slice v-32 0 0 16)
            (slice (slice t-32-32 1 0 16) 0 0 16)
           )
           (bsg-systolic-array 16 16
            (slice v-32 0 16 32)
            (slice (slice t-32-32 1 0 16) 0 16 32)
           )
          )
          (elementwise-add
           (bsg-systolic-array 16 16
            (slice v-32 0 0 16)
            (slice (slice t-32-32 1 16 32) 0 0 16)
           )
           (bsg-systolic-array 16 16
            (slice v-32 0 16 32)
            (slice (slice t-32-32 1 16 32) 0 16 32)
           )
          )
          0
         )
         0 0 16
        )
        (slice (slice t-32-32 1 0 16) 0 0 16)
       )
       (bsg-systolic-array 16 16
        (slice
         (concatenate
          (elementwise-add
           (bsg-systolic-array 16 16
            (slice v-32 0 0 16)
            (slice (slice t-32-32 1 0 16) 0 0 16)
           )
           (bsg-systolic-array 16 16
            (slice v-32 0 16 32)
            (slice (slice t-32-32 1 0 16) 0 16 32)
           )
          )
          (elementwise-add
           (bsg-systolic-array 16 16
            (slice v-32 0 0 16)
            (slice (slice t-32-32 1 16 32) 0 0 16)
           )
           (bsg-systolic-array 16 16
            (slice v-32 0 16 32)
            (slice (slice t-32-32 1 16 32) 0 16 32)
           )
          )
          0
         )
         0 16 32
        )
        (slice (slice t-32-32 1 0 16) 0 16 32)
       )
      )
      (elementwise-add
       (bsg-systolic-array 16 16
        (slice
         (concatenate
          (elementwise-add
           (bsg-systolic-array 16 16
            (slice v-32 0 0 16)
            (slice (slice t-32-32 1 0 16) 0 0 16)
           )
           (bsg-systolic-array 16 16
            (slice v-32 0 16 32)
            (slice (slice t-32-32 1 0 16) 0 16 32)
           )
          )
          (elementwise-add
           (bsg-systolic-array 16 16
            (slice v-32 0 0 16)
            (slice (slice t-32-32 1 16 32) 0 0 16)
           )
           (bsg-systolic-array 16 16
            (slice v-32 0 16 32)
            (slice (slice t-32-32 1 16 32) 0 16 32)
           )
          )
          0
         )
         0 0 16
        )
        (slice (slice t-32-32 1 16 32) 0 0 16)
       )
       (bsg-systolic-array 16 16
        (slice
         (concatenate
          (elementwise-add
           (bsg-systolic-array 16 16
            (slice v-32 0 0 16)
            (slice (slice t-32-32 1 0 16) 0 0 16)
           )
           (bsg-systolic-array 16 16
            (slice v-32 0 16 32)
            (slice (slice t-32-32 1 0 16) 0 16 32)
           )
          )
          (elementwise-add
           (bsg-systolic-array 16 16
            (slice v-32 0 0 16)
            (slice (slice t-32-32 1 16 32) 0 0 16)
           )
           (bsg-systolic-array 16 16
            (slice v-32 0 16 32)
            (slice (slice t-32-32 1 16 32) 0 16 32)
           )
          )
          0
         )
         0 16 32
        )
        (slice (slice t-32-32 1 16 32) 0 16 32)
       )
      )
      0)"
    .parse::<Pattern<_>>()
    .unwrap()
    .search_eclass(&runner.egraph, id)
    .expect("Did not find expected program");

    // TODO(@gussmith23) Find other programs
}
