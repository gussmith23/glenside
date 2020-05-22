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
        rewrites::bubble_concat_through_rows_axis_0(),
        rewrites::bubble_concat_through_rows_axis_1(),
        rewrites::bubble_concat_through_cols_axis_0(),
        rewrites::bubble_concat_through_cols_axis_1(),
        rewrites::bubble_concat_through_cartesian_product_not_last_axis_left(),
        rewrites::bubble_concat_through_cartesian_product_not_last_axis_right(),
        rewrites::bubble_concat_through_cartesian_product_last_axis(),
        rewrites::bubble_concat_through_map_dot_product_not_last_axis(),
        rewrites::systolic_array_vector_matrix(),
    ];

    let (egraph, id) = egg::EGraph::<Language, Meta>::from_expr(&program);
    let runner = egg::Runner::new().with_egraph(egraph).run(&rws);
    println!(
        "Stopped after {} iterations, reason: {:?}",
        runner.iterations.len(),
        runner.stop_reason
    );

    println!("{:?}", runner.egraph[id]);
}
