use egg::{Pattern, Searcher};
use glenside::language::*;

#[test]
fn regular_multilayer_perceptron() {
    let program = "
     (map-dot-product
      (cartesian-product
       (map-dot-product
        (cartesian-product
         (map-dot-product
          (cartesian-product
           v-32
           (cols t-32-64)
          )
         )
         (cols t-64-128)
        )
       )
       (cols t-128-16)
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
        rewrites::split(0, 16, false),
        rewrites::split(1, 16, false),
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

    // Find the monolithic program.
    "(bsg-systolic-array 128 16
      (bsg-systolic-array 64 128
       (bsg-systolic-array 32 64 v-32 t-32-64)
       t-64-128
      )
      t-128-16
     )
     "
    .parse::<Pattern<_>>()
    .unwrap()
    .search_eclass(&runner.egraph, id)
    .expect("Did not find expected program");

    // TODO(gus) add more checks. I should be using extraction to do this,
    // instead of writing these out by hand.
}
