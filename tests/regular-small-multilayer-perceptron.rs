use egg::{EGraph, Pattern, Runner, Searcher};
use glenside::language::*;

#[test]
fn regular_small_multilayer_perceptron() {
    let program = "
     (compute dot-product
      (access-cartesian-product
       (access
        (compute dot-product
         (access-cartesian-product
          (access (access-tensor v-32) 0)
          (access-move-axis (access (access-tensor t-32-32) 1) 1 0)
         )
        )
        0
       )
       (access-move-axis (access (access-tensor t-32-32) 1) 1 0)
      )
     )
     "
    .parse()
    .unwrap();

    let rws = vec![
        rewrites::slice_concatenate_accesses(0, 16),
        rewrites::slice_concatenate_accesses(1, 16),
        rewrites::collapse_nested_access_slices(),
        rewrites::bubble_access_concatenate_through_access(),
        rewrites::bubble_access_concatenate_through_access_slice(),
        rewrites::bubble_access_concatenate_through_access_move_axis(),
        rewrites::bubble_access_concatenate_through_access_cartesian_product_not_item_axis_left(),
        rewrites::bubble_access_concatenate_through_access_cartesian_product_not_item_axis_right(),
        rewrites::bubble_access_concatenate_through_access_cartesian_product_same_item_axis(),
        rewrites::bubble_access_concatenate_through_compute_dot_product_item_axis(),
        rewrites::bubble_access_concatenate_through_compute_dot_product_not_item_axis(),
        rewrites::systolic_array(),
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

    let matches = "
(access-concatenate
 (compute reduce-sum
  (access-pair
   (systolic-array 16 16
    (access
     (compute reduce-sum
      (access-pair
       (systolic-array 16 16
        (access-slice (access (access-tensor v-32) 0) 0 0 16)
        (access-move-axis
         (access-slice
          (access-slice
           (access (access-tensor t-32-32) 1)
           0 0 16
          )
          1 0 16
         )
         1 0
        )
       )
       (systolic-array 16 16
        (access-slice (access (access-tensor v-32) 0) 0 16 32)
        (access-move-axis
         (access-slice
          (access-slice
           (access (access-tensor t-32-32) 1)
           0 16 32
          )
          1 0 16
         )
         1 0
        )
       )
      )
     )
     0
    )
    (access-move-axis
     (access-slice
      (access-slice
       (access (access-tensor t-32-32) 1)
       0 0 16
      )
      1 0 16
     )
     1 0
    )
   )
   (systolic-array 16 16
    (access
     (compute reduce-sum
      (access-pair
       (systolic-array 16 16
        (access-slice (access (access-tensor v-32) 0) 0 0 16)
        (access-move-axis
         (access-slice
          (access-slice
           (access (access-tensor t-32-32) 1)
           0 0 16
          )
          1 16 32
         )
         1 0
        )
       )
       (systolic-array 16 16
        (access-slice (access (access-tensor v-32) 0) 0 16 32)
        (access-move-axis
         (access-slice
          (access-slice
           (access (access-tensor t-32-32) 1)
           0 16 32
          )
          1 16 32
         )
         1 0
        )
       )
      )
     )
     0
    )
    (access-move-axis
     (access-slice
      (access-slice
       (access (access-tensor t-32-32) 1)
       0 16 32
      )
      1 0 16
     )
     1 0
    )
   )
  )
 )
 (compute reduce-sum
  (access-pair
   (systolic-array 16 16
    (access
     (compute reduce-sum
      (access-pair
       (systolic-array 16 16
        (access-slice (access (access-tensor v-32) 0) 0 0 16)
        (access-move-axis
         (access-slice
          (access-slice
           (access (access-tensor t-32-32) 1)
           0 0 16
          )
          1 0 16
         )
         1 0
        )
       )
       (systolic-array 16 16
        (access-slice (access (access-tensor v-32) 0) 0 16 32)
        (access-move-axis
         (access-slice
          (access-slice
           (access (access-tensor t-32-32) 1)
           0 16 32
          )
          1 0 16
         )
         1 0
        )
       )
      )
     )
     0
    )
    (access-move-axis
     (access-slice
      (access-slice
       (access (access-tensor t-32-32) 1)
       0 0 16
      )
      1 16 32
     )
     1 0
    )
   )
   (systolic-array 16 16
    (access
     (compute reduce-sum
      (access-pair
       (systolic-array 16 16
        (access-slice (access (access-tensor v-32) 0) 0 0 16)
        (access-move-axis
         (access-slice
          (access-slice
           (access (access-tensor t-32-32) 1)
           0 0 16
          )
          1 16 32
         )
         1 0
        )
       )
       (systolic-array 16 16
        (access-slice (access (access-tensor v-32) 0) 0 16 32)
        (access-move-axis
         (access-slice
          (access-slice
           (access (access-tensor t-32-32) 1)
           0 16 32
          )
          1 16 32
         )
         1 0
        )
       )
      )
     )
     0
    )
    (access-move-axis
     (access-slice
      (access-slice
       (access (access-tensor t-32-32) 1)
       0 16 32
      )
      1 16 32
     )
     1 0
    )
   )
  )
 )
 0
)
    "
    .parse::<Pattern<_>>()
    .unwrap()
    .search_eclass(&runner.egraph, id)
    .expect("Did not find expected program");

    assert_eq!(matches.substs.len(), 1);
}
