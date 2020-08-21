use egg::{Pattern, RecExpr, Runner, Searcher};
use glenside::language::rewrites::*;
use glenside::language::*;
use std::collections::HashMap;
use std::str::FromStr;

#[test]
fn conv2d_im2col_tensorize_to_smaller_array_with_padding_and_slicing() {
    test_logger::ensure_env_logger_initialized();

    let mut expr = RecExpr::from_str("(access-tensor image)").unwrap();
    let id = expr.as_ref().len() - 1;
    let _conv2d_id =
        glenside::models::resnet50::conv2d(&mut expr, id as u32, "weights", (1, 1), (1, 1));

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
        .with_time_limit(std::time::Duration::from_secs(40))
        .with_node_limit(500000)
        .with_iter_limit(40)
        .run(&rws);

    runner.print_report();

    // Search for continued tensorization
    let _matches = "
 (access-transpose
  (access-reshape
   (access-slice
    (access-concatenate
     (access-concatenate
      (access-concatenate
      (access-concatenate
      (access-concatenate
      (access-concatenate

      (compute dot-product
       (access-cartesian-product
        (access-pad
         (access-pad
          (access-flatten
           (access
            (access-transpose (access (access-tensor weights) 0)
             (list 3 0 1 2)
            )
            1
           )
          )
          zero-padding 0 ?2 ?3
         )
         zero-padding 1 ?8 ?9
        )
        (access-concatenate ?21 ?19 ?20)
       )
      )


       ?a
       ?42
      )
       (systolic-array 64 64 ?37 ?38)
       ?39
      )
       (systolic-array 64 64 ?34 ?35)
       ?36
      )
       (systolic-array 64 64 ?31 ?32)
       ?30
      )
      (systolic-array 64 64 ?27 ?28)
      ?29
     )
     (systolic-array 64 64 ?22 ?23)
     ?18
    )
    ?24 ?25 ?26
   )
   ?shape
  )
  ?list
 )
            "
    .parse::<Pattern<_>>()
    .unwrap()
    .search_eclass(&runner.egraph, id)
    .unwrap();
    println!(
        "{:#?}",
        runner.egraph[_matches.substs[0]["?a".parse().unwrap()]]
    );

    // Find whole program
    let _matches = "
 (access-transpose
  (access-reshape
   (access-slice
    (access-concatenate
     (compute dot-product
      (access-cartesian-product
       (access-pad
        (access-pad
         (access-flatten
          (access
           (access-transpose (access (access-tensor weights) 0)
            (list 3 0 1 2)
           )
           1
          )
         )
         zero-padding 0 ?2 ?3
        )
        zero-padding 1 ?8 ?9
       )
       (access-concatenate ?21 ?19 ?20)
      )
     )
     (systolic-array 64 64 ?22 ?23)
     ?18
    )
    ?24 ?25 ?26
   )
   ?shape
  )
  ?list
 )
            "
    .parse::<Pattern<_>>()
    .unwrap()
    .search_eclass(&runner.egraph, id)
    .unwrap();

    // Check for tensorization
    let matches = "
  (access-concatenate
   (compute dot-product
    (access-cartesian-product
     (access-pad
      (access-pad
       (access-flatten
        (access
         (access-transpose (access (access-tensor weights) 0)
          (list 3 0 1 2)
         )
         1
        )
       )
       zero-padding 0 ?2 ?3
      )
      zero-padding 1 ?8 ?9
     )
     (access-concatenate ?21 ?19 ?20)
    )
   )
   (systolic-array 64 64 ?22 ?23)
   ?18
  )
            "
    .parse::<Pattern<_>>()
    .unwrap()
    .search(&runner.egraph);
    assert!(matches.len() > 0);

    // Check that the access-concatenate gets bubbled up
    let matches = "
  (compute dot-product
   (access-concatenate
    (access-cartesian-product
     (access-pad
      (access-pad
       (access-flatten
        (access
         (access-transpose (access (access-tensor weights) 0)
          (list 3 0 1 2)
         )
         1
        )
       )
       zero-padding 0 ?2 ?3
      )
      zero-padding 1 ?8 ?9
     )
     (access-concatenate ?21 ?19 ?20)
    )
    (access-cartesian-product
     (access-pad
      (access-pad
       (access-flatten
        (access
         (access-transpose (access (access-tensor weights) 0)
          (list 3 0 1 2)
         )
         1
        )
       )
       zero-padding 0 ?2 ?3
      )
      zero-padding 1 ?8 ?9
     )
     (access-slice
      (access-pad
       (access-pad
        (access-flatten
         (access
          (access-squeeze
           (access-squeeze
            (access-windows
             (access-pad
              (access-pad (access (access-tensor image) 4) zero-padding 1 1 1)
               zero-padding
               2
               1
               1)
              (shape-insert-axis (shape-remove-axis (shape-of weights) 3) 0)
             (shape 1 1 1 1))
           3)
          3)
         3)
        )
        zero-padding 0 ?10 ?11
       )
       zero-padding 1 ?12 ?13
      )
      ?14 ?15 ?16
     )
    )
    ?18
   )
  )
            "
    .parse::<Pattern<_>>()
    .unwrap()
    .search(&runner.egraph);
    assert!(matches.len() > 0);

    // The image gets flattened, padded, and sliced up; the weights get
    // padded (but not sliced); now we need to see whether the concatenates
    // get moved around correctly.
    let matches = "
  (compute dot-product
   (access-cartesian-product
    (access-pad
     (access-pad
      (access-flatten
       (access
        (access-transpose (access (access-tensor weights) 0)
         (list 3 0 1 2)
        )
        1
       )
      )
      zero-padding 0 ?2 ?3
     )
     zero-padding 1 ?8 ?9
    )
    (access-concatenate
     ?17
     (access-slice
      (access-pad
       (access-pad
        (access-flatten
         (access
          (access-squeeze
           (access-squeeze
            (access-windows
             (access-pad
              (access-pad (access (access-tensor image) 4) zero-padding 1 1 1)
               zero-padding
               2
               1
               1)
              (shape-insert-axis (shape-remove-axis (shape-of weights) 3) 0)
             (shape 1 1 1 1))
           3)
          3)
         3)
        )
        zero-padding 0 ?10 ?11
       )
       zero-padding 1 ?12 ?13
      )
      ?14 ?15 ?16
     )
     ?18
    )
   )
  )
            "
    .parse::<Pattern<_>>()
    .unwrap()
    .search(&runner.egraph);
    assert!(matches.len() > 0);

    let matches = "
(access-transpose
 (access-reshape
  (compute dot-product
   (access-cartesian-product
    (access-slice
     ?25
     ?26
     ?27
     ?24
    )
    (access-slice
     (access-slice
      (access-pad
       ?36 zero-padding ?37 ?38 ?39
      )
      ?33 ?34 ?35
     )
     ?29
     ?30
     ?31
    )
   )
  )
  ?shape
 )
 ?21
)
            "
    .parse::<Pattern<_>>()
    .unwrap()
    .search(&runner.egraph);
    assert!(matches.len() > 0);
    //println!("{:#?}", runner.egraph[matches[0].substs[0]["?25".parse().unwrap()]]);

    // Check that concatenate gets bubbled through the compute
    let matches = "
  (access-concatenate
   (compute dot-product
    (access-cartesian-product
     (access-pad
      (access-pad
       (access-flatten
        (access
         (access-transpose (access (access-tensor weights) 0)
          (list 3 0 1 2)
         )
         1
        )
       )
       zero-padding 0 ?2 ?3
      )
      zero-padding 1 ?8 ?9
     )
     (access-concatenate ?21 ?19 ?20)
    )
   )
   (compute dot-product
    (access-cartesian-product
     (access-pad
      (access-pad
       (access-flatten
        (access
         (access-transpose (access (access-tensor weights) 0)
          (list 3 0 1 2)
         )
         1
        )
       )
       zero-padding 0 ?2 ?3
      )
      zero-padding 1 ?8 ?9
     )
     (access-slice
      (access-pad
       (access-pad
        (access-flatten
         (access
          (access-squeeze
           (access-squeeze
            (access-windows
             (access-pad
              (access-pad (access (access-tensor image) 4) zero-padding 1 1 1)
               zero-padding
               2
               1
               1)
              (shape-insert-axis (shape-remove-axis (shape-of weights) 3) 0)
             (shape 1 1 1 1))
           3)
          3)
         3)
        )
        zero-padding 0 ?10 ?11
       )
       zero-padding 1 ?12 ?13
      )
      ?14 ?15 ?16
     )
    )
   )
   ?18
  )
            "
    .parse::<Pattern<_>>()
    .unwrap()
    .search(&runner.egraph);
    assert!(matches.len() > 0);

    // Check that the access-concatenate gets bubbled up
    let matches = "
  (compute dot-product
   (access-concatenate
    (access-cartesian-product
     (access-pad
      (access-pad
       (access-flatten
        (access
         (access-transpose (access (access-tensor weights) 0)
          (list 3 0 1 2)
         )
         1
        )
       )
       zero-padding 0 ?2 ?3
      )
      zero-padding 1 ?8 ?9
     )
     (access-concatenate ?21 ?19 ?20)
    )
    (access-cartesian-product
     (access-pad
      (access-pad
       (access-flatten
        (access
         (access-transpose (access (access-tensor weights) 0)
          (list 3 0 1 2)
         )
         1
        )
       )
       zero-padding 0 ?2 ?3
      )
      zero-padding 1 ?8 ?9
     )
     (access-slice
      (access-pad
       (access-pad
        (access-flatten
         (access
          (access-squeeze
           (access-squeeze
            (access-windows
             (access-pad
              (access-pad (access (access-tensor image) 4) zero-padding 1 1 1)
               zero-padding
               2
               1
               1)
              (shape-insert-axis (shape-remove-axis (shape-of weights) 3) 0)
             (shape 1 1 1 1))
           3)
          3)
         3)
        )
        zero-padding 0 ?10 ?11
       )
       zero-padding 1 ?12 ?13
      )
      ?14 ?15 ?16
     )
    )
    ?18
   )
  )
            "
    .parse::<Pattern<_>>()
    .unwrap()
    .search(&runner.egraph);
    assert!(matches.len() > 0);

    // The image gets flattened, padded, and sliced up; the weights get
    // padded (but not sliced); now we need to see whether the concatenates
    // get moved around correctly.
    let matches = "
  (compute dot-product
   (access-cartesian-product
    (access-pad
     (access-pad
      (access-flatten
       (access
        (access-transpose (access (access-tensor weights) 0)
         (list 3 0 1 2)
        )
        1
       )
      )
      zero-padding 0 ?2 ?3
     )
     zero-padding 1 ?8 ?9
    )
    (access-concatenate
     ?17
     (access-slice
      (access-pad
       (access-pad
        (access-flatten
         (access
          (access-squeeze
           (access-squeeze
            (access-windows
             (access-pad
              (access-pad (access (access-tensor image) 4) zero-padding 1 1 1)
               zero-padding
               2
               1
               1)
              (shape-insert-axis (shape-remove-axis (shape-of weights) 3) 0)
             (shape 1 1 1 1))
           3)
          3)
         3)
        )
        zero-padding 0 ?10 ?11
       )
       zero-padding 1 ?12 ?13
      )
      ?14 ?15 ?16
     )
     ?18
    )
   )
  )
            "
    .parse::<Pattern<_>>()
    .unwrap()
    .search(&runner.egraph);
    assert!(matches.len() > 0);

    let matches = "
(access-transpose
 (access-reshape
  (compute dot-product
   (access-cartesian-product
    (access-slice
     ?25
     ?26
     ?27
     ?24
    )
    (access-slice
     (access-slice
      (access-pad
       ?36 zero-padding ?37 ?38 ?39
      )
      ?33 ?34 ?35
     )
     ?29
     ?30
     ?31
    )
   )
  )
  ?shape
 )
 ?21
)
            "
    .parse::<Pattern<_>>()
    .unwrap()
    .search(&runner.egraph);
    assert!(matches.len() > 0);
    //println!("{:#?}", runner.egraph[matches[0].substs[0]["?25".parse().unwrap()]]);

    let matches = "
(systolic-array 64 64 ?c ?d)
            "
    .parse::<Pattern<_>>()
    .unwrap()
    .search(&runner.egraph);
    assert!(matches.len() > 0);

    let matches = "
(access-slice
 (access-slice
  (access-concatenate
   ?15
   (access-cartesian-product
    (access-pad
     (access-pad
      (access-flatten
       (access ?a 1)
      )
      zero-padding 0 0 56
     )
     zero-padding 1 0 37
    )
    (access-slice
     (access-pad
      (access-pad
       (access-flatten
        ?10
       )
       zero-padding 0 0 0
      )
      zero-padding 1 0 37
     )
     ?17 ?18 ?19
    )
   )
   ?13
  )
  0 ?3 ?4
 )
 1 ?8 ?9
)
            "
    .parse::<Pattern<_>>()
    .unwrap()
    .search(&runner.egraph);
    assert!(matches.len() > 0);

    let matches = "
(access-slice
 (access-slice
  (access-cartesian-product
   (access-pad
    (access-pad
     (access-flatten
      (access ?a 1)
     )
     zero-padding 0 0 56
    )
    zero-padding 1 0 37
   )
   (access-pad
    (access-pad
     (access-flatten
      ?10
     )
     zero-padding 0 0 0
    )
    zero-padding 1 0 37
   )
  )
  0 ?3 ?4
 )
 1 ?8 ?9
)

            "
    .parse::<Pattern<_>>()
    .unwrap()
    .search(&runner.egraph);
    assert!(matches.len() > 0);
    //println!("{:#?}", runner.egraph[matches[0].substs[0]["?11".parse().unwrap()]].data);
    //panic!()//
    let (cost, expr) = egg::Extractor::new(
        &runner.egraph,
        glenside::extraction::MonolithicCostFunction {
            egraph: &runner.egraph,
            systolic_array_configuration: (64, 64),
        },
    )
    .find_best(id);
    // If this call causes a stack overflow, it's because you're getting an
    // infinite loop due to the cost function. If it can only find things
    // that are usize::MAX, it might get stuck traversing an infinite loop.

    println!("{}", expr.pretty(80));
    println!("{:?}", cost);

    assert!(cost < std::usize::MAX);
}
