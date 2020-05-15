#[test]
fn test_32_32_matmul_16_16_systolic_array() {
    use glenside::language::*;

    let rws = vec![
        rewrites::split_concat(),
        rewrites::bubble_concat_through_rows_axis_0(),
        rewrites::bubble_concat_through_rows_axis_1(),
        rewrites::bubble_concat_through_cols_axis_0(),
        rewrites::bubble_concat_through_cols_axis_1(),
        rewrites::bubble_concat_through_cartesian_product_axis_0_0(),
        rewrites::rewrite_nonmatching_cartesian_product_concat(),
    ];

    let program = "
     (map-dot-product
      (cartesian-product
       (rows single-matrix-multiply-input-a)
       (cols single-matrix-multiply-input-b)
      )
     )
     "
    .parse()
    .unwrap();

    let (egraph, _) = egg::EGraph::<Language, Meta>::from_expr(&program);
    let runner = egg::Runner::new().with_egraph(egraph).run(&rws);
    println!(
        "Stopped after {} iterations, reason: {:?}",
        runner.iterations.len(),
        runner.stop_reason
    );
}
