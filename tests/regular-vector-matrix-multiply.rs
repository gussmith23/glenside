use egg::{Pattern, Searcher};
use glenside::language::*;

#[test]
fn regular_vector_matrix_multiply() {
    // Our initial program: A 1x32 X 32x32 vector--matrix multiply.
    // (v-32 is our vector, while t-32-32 is our matrix (or tensor))
    // Our goal is to tile this program up into 1x16 X 16x16 computations and
    // tensorize it to 16x16 systolic arrays. Let's go!
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

    // We will achieve this tensorization with a set of program rewrites:
    let rws = vec![
        // These rewrites tile the program automatically.
        // Specifically, they cut every tensor (including intermediate tensors!)
        // in half and concatenate them back together, along both axes (0 and
        // 1). Currently, we don't break down axes that are less than 16 in
        // length, but we can adjust that later.
        rewrites::split(0, 16),
        rewrites::split(1, 16),
        // This rewrite collapses multiple slice operators (introduced by the
        // above rewrites) into one.
        rewrites::collapse_nested_slices(),
        // These rewrites bubble concatenate operators (also introduced by the
        // split rewrites) up to the top of the program.
        // We need to get the concatenate operators to the top so that we can
        // identify places where we can map in hardware.
        // For example, if we see:
        // (map-dot-product
        //  (cartesian-product (concat ...) (concat ...))
        // )
        // ...we don't have a hardware atom that does this. But if we can
        // rewrite it to:
        // (concat
        //  (map-dot-product
        //   (cartesian-product <a 1x16 vector> <a 16x16 tensor>)
        //  )
        //  (map-dot-product
        //   (cartesian-product <a 1x16 vector> <a 16x16 tensor>)
        //  )
        // )
        // ...then we can map in systolic arrays for the (map-dot-product...)
        // expressions, and the top-level concat will be handled by the compiler.
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
        // It identifies patterns that we have hardware atoms for. Right now, it
        // finds:
        // (map-dot-product
        //  (cartesian-product <a 1xrows vector> (cols <a rowsxcols tensor>))
        // )
        // and rewrites it to:
        // (bsg-systolic-array <rows> <cols> < 1xrows vector> <a rowsxcols tensor>)
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
