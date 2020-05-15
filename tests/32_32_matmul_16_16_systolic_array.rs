#[test]
fn test_32_32_matmul_16_16_systolic_array() {
    use glenside::language::rewrites::*;
    use glenside::language::*;

    let rws = vec![
        // TODO(gus) damn it, I still think that usize-halve won't even be enough.
        // TODO(gus) the if statements actually run backwards.
        //egg::rewrite!("split-x"; "?a" => {SplitApplier{axis: 0, a:"?a".parse().unwrap()}} if dimension_greater_than("?a", 0, 16) if dimension_is_even("?a", 0) if has_shape("?a")),
        //egg::rewrite!("split-y"; "?a" => {SplitApplier{axis: 1, a:"?a".parse().unwrap()}} if dimension_greater_than("?a", 1, 16) if dimension_is_even("?a", 1) if has_shape("?a")),
        egg::rewrite!("split-concat"; "?a" => {SplitConcatApplier{a:"?a".parse().unwrap()}} if has_shape("?a") if is_symbol("?a")),
        egg::rewrite!("bubble-concat-through-rows-axis-0"; "(rows (concat ?a ?b 0))"
                      => "(concat (rows ?a) (rows ?b) 0)"),
        egg::rewrite!("bubble-concat-through-rows-axis-1"; "(rows (concat ?a ?b 1))"
                      => "(concat (rows ?a) (rows ?b) 1)"),
        egg::rewrite!("bubble-concat-through-cols-axis-0"; "(cols (concat ?a ?b 0))"
                      => "(concat (cols ?a) (cols ?b) 1)"),
        egg::rewrite!("bubble-concat-through-cols-axis-1"; "(cols (concat ?a ?b 1))"
                      => "(concat (cols ?a) (cols ?b) 0)"),
        // TODO(gus) this isn't the only way this could be done.
        // Also there's gotta be a name for this in terms of algebraic rules
        // TODO(gus) would it make our pattern-matching life easier if (1) we
        // put the axes at the start of concat and (2) we used cons cells?
        egg::rewrite!("bubble-concat-through-cartesian-product-axes-0-0";
                      "(cartesian-product (concat ?a1 ?a2 0) (concat ?b1 ?b2 0))"
                      // TODO(gus) check this
                      => "(concat
                           (concat (cartesian-product ?a1 ?b1)
                                   (cartesian-product ?a1 ?b2) 1)
                           (concat (cartesian-product ?a2 ?b1)
                                   (cartesian-product ?a2 ?b2) 1)
                           0)"
        ),
        egg::rewrite!(
        "rewrite-nonmatching-cartesian-product-concat";
        "(cartesian-product
              (concat ?a1 ?a2 0)
              (concat ?b1 ?b2 1)
             )" =>
        {RewriteNonMatchingCartConcatApplier{
            a1:"?a1".parse().unwrap(),
            a2:"?a2".parse().unwrap(),
            a_axis:0,
            b1:"?b1".parse().unwrap(),
            b2:"?b2".parse().unwrap(),
            b_axis:1,
        }}),
        // egg::rewrite!("bubble-concat-through-cartesian-product"; "(cartesian-product (concat ?a ?b ?c ?d ?axis) (concat ?e ?f ?g ?h ?axis))" =>
        // // TODO(gus) I think this one's where the magic happens :)
        // {BubbleConcatThroughCartesianProductApplier{
        //     a:"?a".parse().unwrap(),
        //     b:"?b".parse().unwrap(),
        //     c:"?c".parse().unwrap(),
        //     d:"?d".parse().unwrap(),
        //     e:"?e".parse().unwrap(),
        //     f:"?f".parse().unwrap(),
        //     g:"?g".parse().unwrap(),
        //     h:"?h".parse().unwrap(),
        //     axis:"?axis".parse().unwrap(),

        // }}),
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
    egraph
        .dot()
        .to_svg("single-matrix-multiply-before-rewrites.svg")
        .unwrap();
    let runner = egg::Runner::new().with_egraph(egraph).run(&rws);
    runner
        .egraph
        .dot()
        .to_svg("single-matrix-multiply-after-rewrites.svg")
        .unwrap();

    // let out = interpret_eclass(
    //     &runner.egraph,
    //     &runner.egraph[id],
    //     &env,
    //     &mut MemoizationMap::new(),
    // );
    // let out = unpack_interpreter_output(out);
    // assert!(out_true.abs_diff_eq(&out, 1e-8));
}
