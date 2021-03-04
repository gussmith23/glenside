#![cfg(feature = "tvm")]

use egg::Pattern;
use egg::Searcher;
use egg::{RecExpr, Runner};
use glenside::language::rewrites::*;
use glenside::language::*;
use std::collections::HashMap;
use std::str::FromStr;

#[test]
fn conv2d_im2col_tensorize_to_smaller_array_with_padding_and_slicing() {
    #[cfg(not(feature = "run-on-github-actions"))]
    pub const EGG_SEARCH_TIME_SECS: u64 = 60;
    #[cfg(feature = "run-on-github-actions")]
    pub const EGG_SEARCH_TIME_SECS: u64 = 180;

    pub const IMAGE_SHAPE: &[usize] = &[1, 32, 32, 3];
    pub const KERNEL_SHAPE: &[usize] = &[3, 3, 3, 8];

    let mut expr = RecExpr::from_str("(access-tensor image)").unwrap();
    let data_id = expr.as_ref().len() - 1;
    let weights_id = expr.add(Language::Symbol("weights".to_string()));
    let weights_id = expr.add(Language::AccessTensor(weights_id));

    let _conv2d_id = crate::from_relay::conv2d(
        &mut expr,
        data_id.into(),
        IMAGE_SHAPE,
        weights_id,
        KERNEL_SHAPE,
        &[1, 1],
        &[1, 1, 1, 1],
        &[1, 1],
        1,
        "NHWC",
        "HWIO",
        "",
    );

    let mut map = HashMap::default();
    // batch, height, width, channels
    map.insert("image".to_string(), vec![1, 32, 32, 3]);
    // kernel height, kernel width, in channels, out channels
    map.insert("weights".to_string(), vec![3, 3, 3, 8]);

    let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis { name_to_shape: map });
    let id = egraph.add_expr(&expr);

    let rws = vec![
        glenside::language::rewrites::flatten_unflatten_any_access(),
        glenside::language::rewrites::bubble_reshape_through_cartesian_product(),
        glenside::language::rewrites::bubble_reshape_through_compute_dot_product(),
        glenside::language::rewrites::bubble_access_concatenate_through_access_cartesian_product_not_item_axis_left(),
        glenside::language::rewrites::bubble_access_concatenate_through_access_cartesian_product_not_item_axis_right(),
        glenside::language::rewrites::bubble_access_concatenate_through_access_cartesian_product_same_item_axis(),
        glenside::language::rewrites::bubble_access_concatenate_through_compute_dot_product_item_axis(),
        glenside::language::rewrites::bubble_access_concatenate_through_compute_dot_product_not_item_axis(),
        glenside::language::rewrites::bubble_access_slice_through_access_pad_inequal_axes(),
        glenside::language::rewrites::systolic_array(),
        glenside::language::rewrites::pad_slice_accesses(
            0,
            PadSliceStrategy::PadToClosestMultipleOf {
                multiple_of: 64,
                pad_location: PadLocation::End,
                pad_type: PadType::ZeroPadding,
            },
        ),
        glenside::language::rewrites::pad_slice_accesses(
            1,
            PadSliceStrategy::PadToClosestMultipleOf {
                multiple_of: 64,
                pad_location: PadLocation::End,
                pad_type: PadType::ZeroPadding,
            },
        ),
        glenside::language::rewrites::slice_concatenate_accesses(
            0,
            SliceConcatenateStrategy::DivideInto { segment_size: 64 },
        ),
        glenside::language::rewrites::slice_concatenate_accesses(
            1,
            SliceConcatenateStrategy::DivideInto { segment_size: 64 },
        ),
        glenside::language::rewrites::bubble_access_slice_through_access_cartesian_product_not_item_axis_left(),
        glenside::language::rewrites::bubble_access_slice_through_access_cartesian_product_not_item_axis_right(),
        glenside::language::rewrites::bubble_access_slice_through_access_cartesian_product_same_item_axis(),
        glenside::language::rewrites::bubble_access_slice_through_compute_dot_product_not_item_axis(),
        glenside::language::rewrites::bubble_access_slice_through_compute_dot_product_item_axis_not_tuple_axis(),
    ];

    // TODO(@gussmith23) This is starting to become a flaky test...
    // I know the correct program can be found, but it takes time.
    let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
        .with_egraph(egraph)
        .with_time_limit(std::time::Duration::from_secs(EGG_SEARCH_TIME_SECS))
        .with_node_limit(500000)
        .with_iter_limit(40)
        .run(&rws);

    runner.print_report();

    let (cost, expr) = egg::Extractor::new(
        &runner.egraph,
        glenside::extraction::MonolithicCostFunction {
            egraph: &runner.egraph,
            systolic_array_configuration: (64, 64),
            prefer_systolic_arrays_with_blocking: false,
        },
    )
    .find_best(id);

    println!("{}", expr.pretty(80));
    println!("{:?}", cost);

    assert!(cost < glenside::extraction::MonolithicCostFunction::INFINITY_VALUE);

    "
(access-transpose
 (access-transpose
  (compute
   dot-product
   (access-cartesian-product
    (access-reshape
     (access-flatten
      (access (access-transpose (access-tensor weights) (list 3 2 0 1)) 1))
     (access-shape (shape 8) (shape 3 3 3)))
    (access-reshape
     (access-flatten
      (access
       (access-squeeze
        (access-squeeze
         (access-windows
          (access
           (access-pad
            (access-pad (access-transpose (access-tensor image) (list 0 3 1 2)) zero-padding 2 1 1)
            zero-padding
            3
            1
            1)
           4)
          (shape 1 3 3 3)
          (shape 1 1 1 1))
         4)
        1)
       3))
     (access-shape (shape 1 32 32) (shape 3 3 3)))))
  (list 1 0 2 3))
 (list 0 2 3 1))"
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
}
