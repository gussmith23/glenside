use egg::{Pattern, Searcher};
use glenside::language::*;

#[test]
fn regular_vector_matrix_multiply() {

    // Our initial program: A 1x32 X 32x32 vector--matrix multiply.
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

    // All the rewrites needed to tensorize this program.
    let rws = vec![
        // These rewrites tile the program automatically.
        rewrites::split(0, 16),
        rewrites::split(1, 16),

        // This rewrite collapses multiple slice operators into one.
        rewrites::collapse_nested_slices(),

        // These rewrites bubble concatenate operators up to the top of the
        // program.
        rewrites::bubble_concat_through_rows_axis_0(),
        rewrites::bubble_concat_through_rows_axis_1(),
        rewrites::bubble_concat_through_cols_axis_0(),
        rewrites::bubble_concat_through_cols_axis_1(),
        rewrites::bubble_concat_through_cartesian_product_not_last_axis_left(),
        rewrites::bubble_concat_through_cartesian_product_not_last_axis_right(),
        rewrites::bubble_concat_through_cartesian_product_last_axis(),
        rewrites::bubble_concat_through_map_dot_product_not_last_axis(),
        rewrites::bubble_concat_through_map_dot_product_last_axis(),

        // Finally, this rewrite tensorizes!
        rewrites::systolic_array_vector_matrix(),
    ];

    // Run the rewrites over the egraph.
    let (egraph, id) = egg::EGraph::<Language, Meta>::from_expr(&program);
    let runner = egg::Runner::new().with_egraph(egraph).run(&rws);
    println!(
        "Stopped after {} iterations, reason: {:?}",
        runner.iterations.len(),
        runner.stop_reason
    );

    // Search for one of the expected "tensorizable" programs. By
    // "tensorizable", I mean that this program has clear locations where
    // hardware can be used (specifically, each map-dot-product can be replaced
    // by 16x16 systolic arrays).
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

    // But that's not all!
    // The egraph will find a ton of hardware combinations.

    // Check that the program got tensorized.
    "(concat
      (bsg-systolic-array 32 16
       v-32
       (slice t-32-32 0 32 0 16)
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
