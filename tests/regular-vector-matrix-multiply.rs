use egg::{Pattern, Searcher};
use glenside::language::*;

#[test]
fn regular_vector_matrix_multiply() {
    let program = "
     (map-dot-product
      (cartesian-product
       v-32
       (cols t-32-32)
      )
     )
     "
    .parse()
    .unwrap();

    let rws = vec![
        rewrites::split(0, 16),
        rewrites::split(1, 16),
        rewrites::collapse_nested_slices(),
        rewrites::bubble_concat_through_rows_axis_0(),
        rewrites::bubble_concat_through_rows_axis_1(),
        rewrites::bubble_concat_through_cols_axis_0(),
        rewrites::bubble_concat_through_cols_axis_1(),
        rewrites::bubble_concat_through_cartesian_product_not_last_axis_left(),
        rewrites::bubble_concat_through_cartesian_product_not_last_axis_right(),
        rewrites::bubble_concat_through_cartesian_product_last_axis(),
        rewrites::bubble_concat_through_map_dot_product_not_last_axis(),
        rewrites::bubble_concat_through_map_dot_product_last_axis(),
        rewrites::systolic_array_vector_matrix(),
    ];

    let (egraph, id) = egg::EGraph::<Language, Meta>::from_expr(&program);
    let runner = egg::Runner::new().with_egraph(egraph).run(&rws);
    println!(
        "Stopped after {} iterations, reason: {:?}",
        runner.iterations.len(),
        runner.stop_reason
    );

    // Search for the expected program. Everything before this was just my
    // incremental debugging. It could all be removed, but I figured I'll keep
    // it---more checks don't hurt!
    "(concat
      (elementwise-add
       (map-dot-product
        (cartesian-product
         (slice v-32 0 16)
         (cols (slice t-32-32 0 16 0 16))
        )
       )
       (map-dot-product
        (cartesian-product
         (slice v-32 16 32)
         (cols (slice t-32-32 16 32 0 16))
        )
       )

      )
      (elementwise-add
       (map-dot-product
        (cartesian-product
         (slice v-32 0 16)
         (cols (slice t-32-32 0 16 16 32))
        )
       )
       (map-dot-product
        (cartesian-product
         (slice v-32 16 32)
         (cols (slice t-32-32 16 32 16 32))
        )
       )
      )
      0)"
    .parse::<Pattern<_>>()
    .unwrap()
    .search_eclass(&runner.egraph, id)
    .expect("Did not find expected program");

    // Check that the program got tensorized.
    "(concat
      (elementwise-add
       (bsg-systolic-array 16 16
        (slice v-32 0 16)
        (slice t-32-32 0 16 0 16)
       )
       (bsg-systolic-array 16 16
        (slice v-32 16 32)
        (slice t-32-32 16 32 0 16)
       )
      )
      (elementwise-add
       (bsg-systolic-array 16 16
        (slice v-32 0 16)
        (slice t-32-32 0 16 16 32)
       )
       (bsg-systolic-array 16 16
        (slice v-32 16 32)
        (slice t-32-32 16 32 16 32)
       )
      )
      0)"
    .parse::<Pattern<_>>()
    .unwrap()
    .search_eclass(&runner.egraph, id)
    .expect("Did not find expected program");
}
