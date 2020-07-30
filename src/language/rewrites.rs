use super::{Language, MyAnalysis, MyAnalysisData};
use egg::{rewrite, Applier, ConditionalApplier, EGraph, Id, Pattern, Rewrite, Subst, Var};
use ndarray::Dimension;

type EG = EGraph<Language, MyAnalysis>;
type RW = Rewrite<Language, MyAnalysis>;

fn is_access() -> impl Fn(&mut EG, egg::Id, &egg::Subst) -> bool {
    move |egraph, id, _| match &egraph[id].data {
        MyAnalysisData::AccessPattern(_) => true,
        _ => false,
    }
}

fn same_item_axis(
    axis0: Var,
    access0: Var,
    axis1: Var,
    access1: Var,
) -> impl Fn(&mut EG, egg::Id, &egg::Subst) -> bool {
    move |egraph, _id, subst| {
        let (a0, a1) = match (&egraph[subst[access0]].data, &egraph[subst[access1]].data) {
            (MyAnalysisData::AccessPattern(a0), MyAnalysisData::AccessPattern(a1)) => (a0, a1),
            _ => panic!(),
        };
        let axis0 = MyAnalysis::get_usize(subst[axis0], egraph);
        let axis1 = MyAnalysis::get_usize(subst[axis1], egraph);

        axis0 >= a0.shape.ndim()
            && axis1 >= a1.shape.ndim()
            && axis0 - a0.shape.ndim() == axis1 - a1.shape.ndim()
    }
}

fn not_item_axis(axis: Var, access: Var) -> impl Fn(&mut EG, egg::Id, &egg::Subst) -> bool {
    move |egraph, _id, subst| match &egraph[subst[access]].data {
        MyAnalysisData::AccessPattern(a) => {
            MyAnalysis::get_usize(subst[axis], egraph) < a.shape.ndim()
        }
        _ => panic!(),
    }
}

fn item_axis(axis: Var, access: Var) -> impl Fn(&mut EG, egg::Id, &egg::Subst) -> bool {
    move |egraph, _id, subst| match &egraph[subst[access]].data {
        MyAnalysisData::AccessPattern(a) => {
            MyAnalysis::get_usize(subst[axis], egraph) >= a.shape.ndim()
        }
        _ => panic!(),
    }
}

// TODO(@gussmith23) I think I should make this a conditional applier, and fold in
// checks to make sure it has a shape and that it's an input
pub fn has_shape(var: &'static str) -> impl Fn(&mut EG, egg::Id, &egg::Subst) -> bool {
    let var = var.parse().unwrap();
    move |egraph, _, subst| match &egraph[subst[var]].data {
        MyAnalysisData::Legacy(s) => !s.shape.is_none(),
        _ => panic!(),
    }
}
/// short_circuit lets us return early if we don't actually care about the
/// result of this check. This is the easiest way I could find to do this using
/// egg's conditional appliers.
/// TODO(@gussmith23) make this cleaner
pub fn is_symbol(
    short_circuit: bool,
    var: &'static str,
) -> impl Fn(&mut EG, egg::Id, &egg::Subst) -> bool {
    let var = var.parse().unwrap();
    // TODO(@gussmith23) should this be `all` or `any` or something else entirely?
    move |egraph, _, subst| {
        if short_circuit {
            true
        } else {
            egraph[subst[var]]
                .nodes
                .iter()
                .map(|enode| match enode {
                    Language::Symbol(_) => true,
                    _ => false,
                })
                .all(|x| x)
        }
    }
}
fn has_axis(var: &'static str, axis: usize) -> impl Fn(&mut EG, egg::Id, &egg::Subst) -> bool {
    let var = var.parse().unwrap();
    move |egraph, _, subst| axis < MyAnalysis::get_shape(subst[var], egraph).ndim()
}
fn dimension_greater_than(
    var: &'static str,
    axis: usize,
    greater_than: usize,
) -> impl Fn(&mut EG, egg::Id, &egg::Subst) -> bool {
    let var = var.parse().unwrap();
    move |egraph, _, subst| MyAnalysis::get_shape(subst[var], egraph)[axis] > greater_than
}
fn dimension_is_even(
    var: &'static str,
    axis: usize,
) -> impl Fn(&mut EG, egg::Id, &egg::Subst) -> bool {
    let var = var.parse().unwrap();
    move |egraph, _, subst| MyAnalysis::get_shape(subst[var], egraph)[axis] % 2 == 0
}

// TODO(@gussmith23) not sure all this should be public.
pub struct RewriteNonMatchingCartConcatenateApplier {
    pub a1: egg::Var,
    pub a2: egg::Var,
    pub a_axis: usize,
    pub b1: egg::Var,
    pub b2: egg::Var,
    pub b_axis: usize,
}
impl egg::Applier<Language, MyAnalysis> for RewriteNonMatchingCartConcatenateApplier {
    fn apply_one(
        &self,
        _egraph: &mut EG,
        _id: egg::Id,
        _subst: &egg::Subst,
    ) -> std::vec::Vec<egg::Id> {
        // For now, just want to handle these cases.
        assert!(self.a_axis == 0 || self.a_axis == 1);
        assert!(self.b_axis == 0 || self.b_axis == 1);
        assert_ne!(self.a_axis, self.b_axis);

        // We will break up the as into smaller chunks and the bs into
        // smaller chunks, so that they all match in size.
        // The goal is to have the innermost concatenates be along axis 0, and
        // the outermost concatenates to be along axis 1. Additionally, the goal
        // is that the result should only involve cartesian products of
        // concatenates, where the left and right concatenate use the same axis.
        // Then, existing rewrites can be used to bubble the concatenates up
        // through the cartesian products.

        // Each a needs to be split into 4; each b needs to be split into 4.

        // First we want to construct all of the concatenates along the 1 axis.
        // These will become our innermost concatenates.
        // One of these is already concatenateted along the 1 axis!

        // TODO(@gussmith23) left off here, I think I should actually do something
        // simpler here and just rewrite the two concatenates that are the
        // children of this cartesian product.
        // It needs some information from elsewhere in the graph, though,
        // that's the tough thing.

        // So we're going to slice-and-concatenate all 4 tensors. We'll slice the
        // as based on the bs size, and slice the bs based on the as size.
        // TODO(@gussmith23) I could write an even simpler rewrite rule that slices
        // more indiscriminately, everywhere. Right now I'm using some
        // context clue (the overarching cartesian product) to only apply
        // this where needed.

        // All I actually want to do is to rewrite that second concatenate.
        //  (cartesian-product
        //   (concatenate ?a1 ?a2 0)
        //   (concatenate ?b1 ?b2 1)
        //  )
        //  (cartesian-product
        //   (concatenate ?a1 ?a2 0)
        //   (concatenate (concatenate (slice ?b1) (slice ?b1)  0)
        //  )
        //

        vec![]
    }
}

struct SplitApplier {
    axis: usize,
}
impl egg::Applier<Language, MyAnalysis> for SplitApplier {
    fn apply_one(
        &self,
        egraph: &mut EG,
        id: egg::Id,
        _subst: &egg::Subst,
    ) -> std::vec::Vec<egg::Id> {
        let shape: ndarray::IxDyn = MyAnalysis::get_shape(id, egraph).clone();
        assert_eq!(shape[self.axis] % 2, 0);
        let low_bound = 0;
        let low_bound_id = egraph.add(Language::Usize(low_bound));
        let high_bound = shape[self.axis];
        let high_bound_id = egraph.add(Language::Usize(high_bound));
        let middle_bound = high_bound / 2;
        let middle_bound_id = egraph.add(Language::Usize(middle_bound));

        let axis_id = egraph.add(Language::Usize(self.axis));

        let slice_0_id = egraph.add(Language::Slice([
            id,
            axis_id,
            low_bound_id,
            middle_bound_id,
        ]));
        let slice_1_id = egraph.add(Language::Slice([
            id,
            axis_id,
            middle_bound_id,
            high_bound_id,
        ]));

        let id: egg::Id = egraph.add(Language::Concatenate([slice_0_id, slice_1_id, axis_id]));

        vec![id]
    }
}

pub fn split(
    axis: usize,
    dimension_greater_than: usize,
    split_all_nodes: bool,
) -> Rewrite<Language, MyAnalysis> {
    rewrite!(format!("split-axis-{}", axis); "?a" =>
                  {SplitApplier{axis: axis}}
             if is_symbol(split_all_nodes, "?a")
             if has_shape("?a")
             if has_axis("?a", axis)
             if dimension_is_even("?a", axis)
             if self::dimension_greater_than("?a", axis, dimension_greater_than))
}

pub fn collapse_nested_slices() -> Rewrite<Language, MyAnalysis> {
    struct CollapseNestedSlicesApplier {
        low0: Var,
        high0: Var,
        low1: Var,
        high1: Var,
    }
    impl Applier<Language, MyAnalysis> for CollapseNestedSlicesApplier {
        fn apply_one(&self, egraph: &mut EG, eclass: Id, subst: &Subst) -> Vec<Id> {
            let low0: usize = MyAnalysis::get_usize(subst[self.low0], egraph);
            let high0: usize = MyAnalysis::get_usize(subst[self.high0], egraph);
            let low1: usize = MyAnalysis::get_usize(subst[self.low1], egraph);
            let high1: usize = MyAnalysis::get_usize(subst[self.high1], egraph);

            let new_low: usize = low0 + low1;
            assert!(high1 - low1 <= high0 - low0);
            let new_high: usize = new_low + (high1 - low1);

            format!("(slice ?t ?axis {} {})", new_low, new_high)
                .parse::<Pattern<Language>>()
                .unwrap()
                .apply_one(egraph, eclass, subst)
        }
    }
    rewrite!("collapse-nested-slices";
    "(slice (slice ?t ?axis ?low0 ?high0) ?axis ?low1 ?high1)" =>
    { CollapseNestedSlicesApplier {
        low0: "?low0".parse().unwrap(),
        low1: "?low1".parse().unwrap(),
        high0: "?high0".parse().unwrap(),
        high1: "?high1".parse().unwrap(),
    }})
}

pub fn bubble_concatenate_through_move_axis() -> Rewrite<Language, MyAnalysis> {
    struct MoveAxisApplier {
        concatenate_axis: Var,
        src_axis: Var,
        dst_axis: Var,
    }
    impl Applier<Language, MyAnalysis> for MoveAxisApplier {
        fn apply_one(&self, egraph: &mut EG, eclass: Id, subst: &Subst) -> Vec<Id> {
            let original_concatenate_axis: usize =
                MyAnalysis::get_usize(subst[self.concatenate_axis], egraph);
            let src_axis: usize = MyAnalysis::get_usize(subst[self.src_axis], egraph);
            let dst_axis: usize = MyAnalysis::get_usize(subst[self.dst_axis], egraph);

            // If the move now happens /before/ the concatenate, we have to
            // figure out what the new axis for the concatenate is.
            // TODO(@gussmith23) Would be nice to have a more principled system of
            // keeping track of axes. This is where Remy's relational algebra
            // stuff could be really useful!
            let new_concatenate_axis: usize = if (original_concatenate_axis < src_axis
                && original_concatenate_axis < dst_axis)
                || (original_concatenate_axis > src_axis && original_concatenate_axis > dst_axis)
            {
                // Axis is unaffected if it's not between src and dst.
                original_concatenate_axis
            } else if original_concatenate_axis == src_axis {
                dst_axis
            } else if original_concatenate_axis < src_axis && original_concatenate_axis >= dst_axis
            {
                original_concatenate_axis + 1
            } else if original_concatenate_axis > src_axis && original_concatenate_axis <= dst_axis
            {
                original_concatenate_axis - 1
            } else {
                unreachable!()
            };

            format!(
                "(concatenate
                      (move-axis ?a ?src-axis ?dst-axis)
                      (move-axis ?b ?src-axis ?dst-axis) {})",
                new_concatenate_axis
            )
            .parse::<Pattern<_>>()
            .unwrap()
            .apply_one(egraph, eclass, subst)
        }
    }
    rewrite!("bubble-concatenate-through-move-axis";
        "(move-axis (concatenate ?a ?b ?concatenate-axis) ?src-axis ?dst-axis)" =>
    {
        MoveAxisApplier {
            concatenate_axis: "?concatenate-axis".parse().unwrap(),
            src_axis:"?src-axis".parse().unwrap(),
            dst_axis:"?dst-axis".parse().unwrap()
        }
    })
}

/// Whether an axis is the last axis of a given tensor
fn last_axis(
    var: &'static str,
    axis: &'static str,
) -> impl Fn(&mut EG, egg::Id, &egg::Subst) -> bool {
    let var = var.parse().unwrap();
    let axis_id = axis.parse().unwrap();
    move |egraph, _, subst| {
        MyAnalysis::get_usize(subst[axis_id], egraph)
            == MyAnalysis::get_shape(subst[var], egraph).ndim() - 1
    }
}
fn not_last_axis(
    var: &'static str,
    axis: &'static str,
) -> impl Fn(&mut EG, egg::Id, &egg::Subst) -> bool {
    move |egraph, id, subst| !(last_axis(var, axis)(egraph, id, subst))
}
fn same_number_of_dimensions(
    a: &'static str,
    b: &'static str,
) -> impl Fn(&mut EG, egg::Id, &egg::Subst) -> bool {
    let a = a.parse().unwrap();
    let b = b.parse().unwrap();
    move |egraph, _, subst| {
        MyAnalysis::get_shape(subst[a], egraph).ndim()
            == MyAnalysis::get_shape(subst[b], egraph).ndim()
    }
}

// TODO(@gussmith23) naming
pub fn bubble_concatenate_through_cartesian_product_not_last_axis_left(
) -> Rewrite<Language, MyAnalysis> {
    rewrite!("bubble-concatenate-through-cartesian-product-not-last-axis-left";
                  "(cartesian-product (concatenate ?t1 ?t2 ?axis) ?right)" =>
                  "(concatenate
                    (cartesian-product ?t1 ?right)
                    (cartesian-product ?t2 ?right)
                    ?axis)"
                  if not_last_axis("?t1", "?axis")
                  // This should always be true, for now. Just making extra sure
                  if same_number_of_dimensions("?t1", "?t2"))
}

struct BubbleConcatenateThroughCartesianProductNotLastAxisRightApplier {
    left: Var,
    axis: Var,
}
impl Applier<Language, MyAnalysis>
    for BubbleConcatenateThroughCartesianProductNotLastAxisRightApplier
{
    fn apply_one(&self, egraph: &mut EG, matched_id: Id, subst: &Subst) -> Vec<Id> {
        // cart-prod [a1, ..., an, c] [b1, ..., bm, c]
        // = [a1, ..., an, b1, ..., bm, 2, c]
        // So the axis gets shifted over by the a1, ..., an added in.
        let left_shape = MyAnalysis::get_shape(subst[self.left], egraph);
        let left_shape_length: usize = left_shape.as_array_view().len();
        let old_axis: usize = MyAnalysis::get_usize(subst[self.axis], egraph);
        let new_axis = old_axis + left_shape_length - 1;

        let applier: Pattern<Language> = format!(
            "(concatenate
                    (cartesian-product ?left ?t1)
                    (cartesian-product ?left ?t2)
                    {})",
            new_axis
        )
        .parse()
        .unwrap();

        applier.apply_one(egraph, matched_id, subst)
    }
}

// TODO(@gussmith23) naming
pub fn bubble_concatenate_through_cartesian_product_not_last_axis_right(
) -> Rewrite<Language, MyAnalysis> {
    rewrite!("bubble-concatenate-through-cartesian-product-not-last-axis-right";
    "(cartesian-product ?left (concatenate ?t1 ?t2 ?axis))" =>
    {
        ConditionalApplier {
            applier: ConditionalApplier {
                applier:
                BubbleConcatenateThroughCartesianProductNotLastAxisRightApplier {
                    left: "?left".parse().unwrap(),
                    axis: "?axis".parse().unwrap(),
                },
                condition: not_last_axis("?t1", "?axis")
            },
            condition: same_number_of_dimensions("?t1", "?t2")
        }
    })
}

struct BubbleConcatenateThroughCartesianProductLastAxisApplier {
    // Note that we're assuming a1's shape is the same as a2; same with b1 and
    // b2.
    a1: Var,
    b1: Var,
}
impl Applier<Language, MyAnalysis> for BubbleConcatenateThroughCartesianProductLastAxisApplier {
    fn apply_one(&self, egraph: &mut EG, matched_id: Id, subst: &Subst) -> Vec<Id> {
        // cart-prod [a1, ..., an, c] [b1, ..., bm, c]
        // = [a1, ..., an, b1, ..., bm, 2, c]
        // axis1 and axis2 both point to their c dimension.
        let a_shape = MyAnalysis::get_shape(subst[self.a1], egraph);
        let a_shape_length: usize = a_shape.as_array_view().len();
        let b_shape = MyAnalysis::get_shape(subst[self.b1], egraph);
        let b_shape_length: usize = b_shape.as_array_view().len();
        let new_axis = a_shape_length - 1 // skip [a1, ..., an]
            + b_shape_length - 1          // skip [b1, ..., bm]
            + 1; // skip [2]

        // TODO
        let applier: Pattern<Language> = format!(
            // "(concatenate
            //   (concatenate
            //    (cartesian-product ?a1 ?b1)
            //    (cartesian-product ?a1 ?b2)
            //    {0})
            //   (concatenate
            //    (cartesian-product ?a2 ?b1)
            //    (cartesian-product ?a2 ?b2)
            //    {0})
            //  {0})",
            "(concatenate
              (cartesian-product ?a1 ?b1)
              (cartesian-product ?a2 ?b2)
             {0})",
            new_axis
        )
        .parse()
        .unwrap();

        applier.apply_one(egraph, matched_id, subst)
    }
}

// TODO(@gussmith23) naming
pub fn bubble_concatenate_through_cartesian_product_last_axis() -> Rewrite<Language, MyAnalysis> {
    // TODO(@gussmith23) I think we need more checks here, to make sure that the sizes
    // actually line up correctly.
    rewrite!("bubble-concatenate-through-cartesian-product-last-axis";
    "(cartesian-product (concatenate ?a1 ?a2 ?axis1) (concatenate ?b1 ?b2 ?axis2))" =>

    {
        ConditionalApplier {
            condition: same_number_of_dimensions("?a1", "?a2"),
            applier: ConditionalApplier {
                condition: last_axis("?a1", "?axis1"),
                applier:                       ConditionalApplier {
                    condition: same_number_of_dimensions("?b1", "?b2"),
                    applier: ConditionalApplier {
                        condition: last_axis("?b1", "?axis2"),
                        applier: BubbleConcatenateThroughCartesianProductLastAxisApplier {
                            a1: "?a1".parse().unwrap(),
                            b1: "?b1".parse().unwrap(),
                        }
                    }
                }

            }
        }
    })
}

pub fn bubble_concatenate_through_cartesian_product_axis_0_0() -> Rewrite<Language, MyAnalysis> {
    // TODO(@gussmith23) this isn't the only way this could be done.
    // Also there's gotta be a name for this in terms of algebraic rules
    // TODO(@gussmith23) would it make our pattern-matching life easier if (1) we
    // put the axes at the start of concatenate and (2) we used cons cells?
    rewrite!("bubble-concatenate-through-cartesian-product-axes-0-0";
                  "(cartesian-product (concatenate ?a1 ?a2 0) (concatenate ?b1 ?b2 0))"
                  // TODO(@gussmith23) check this
                  => "(concatenate
                           (concatenate (cartesian-product ?a1 ?b1)
                                   (cartesian-product ?a1 ?b2) 1)
                           (concatenate (cartesian-product ?a2 ?b1)
                                   (cartesian-product ?a2 ?b2) 1)
                           0)"
    )
}

pub fn rewrite_nonmatching_cartesian_product_concatenate() -> Rewrite<Language, MyAnalysis> {
    rewrite!(
    "rewrite-nonmatching-cartesian-product-concatenate";
    "(cartesian-product
              (concatenate ?a1 ?a2 0)
              (concatenate ?b1 ?b2 1)
             )" =>
    {RewriteNonMatchingCartConcatenateApplier{
        a1:"?a1".parse().unwrap(),
        a2:"?a2".parse().unwrap(),
        a_axis:0,
        b1:"?b1".parse().unwrap(),
        b2:"?b2".parse().unwrap(),
        b_axis:1,
    }})
}

pub fn bubble_concatenate_through_map_dot_product_not_last_axis() -> Rewrite<Language, MyAnalysis> {
    rewrite!(

        "bubble-concatenate-through-map-dot-product-not-last-axis";
        "(map-dot-product
          (concatenate ?left ?right ?axis)
         )" =>
        "(concatenate
          (map-dot-product ?left)
          (map-dot-product ?right)
         ?axis)"
            if not_last_axis("?left", "?axis")
            // This should always be true, for now. Just making extra sure
            if same_number_of_dimensions("?left", "?right")
    )
}

pub fn bubble_concatenate_through_map_dot_product_last_axis() -> Rewrite<Language, MyAnalysis> {
    rewrite!(

        "bubble-concatenate-through-map-dot-product-last-axis";
        "(map-dot-product
          (concatenate ?left ?right ?axis)
         )" =>
            "(elementwise-add
              (map-dot-product ?left)
              (map-dot-product ?right)
             )"
            if last_axis("?left", "?axis")
            // This should always be true, for now. Just making extra sure
            if same_number_of_dimensions("?left", "?right")
    )
}

pub fn slice_move_axis_composition_commutative() -> Rewrite<Language, MyAnalysis> {
    struct SliceMoveAxisCompositionCommutativeApplier {
        move_axis_src: Var,
        move_axis_dest: Var,
        slice_axis: Var,
    }
    impl Applier<Language, MyAnalysis> for SliceMoveAxisCompositionCommutativeApplier {
        fn apply_one(&self, egraph: &mut EG, matched_id: Id, subst: &Subst) -> Vec<Id> {
            let src_axis: usize = MyAnalysis::get_usize(subst[self.move_axis_src], egraph);
            let dst_axis: usize = MyAnalysis::get_usize(subst[self.move_axis_dest], egraph);
            let old_slice_axis: usize = MyAnalysis::get_usize(subst[self.slice_axis], egraph);
            let new_slice_axis = if (old_slice_axis < src_axis && old_slice_axis < dst_axis)
                || (old_slice_axis > src_axis && old_slice_axis > dst_axis)
            {
                // Axis is unaffected if it's not between src and dst.
                old_slice_axis
            } else if old_slice_axis == src_axis {
                dst_axis
            } else if old_slice_axis < src_axis && old_slice_axis >= dst_axis {
                old_slice_axis + 1
            } else if old_slice_axis > src_axis && old_slice_axis <= dst_axis {
                old_slice_axis - 1
            } else {
                unreachable!()
            };

            format!(
                "(move-axis (slice ?tensor {} ?bottom ?top) ?src ?dest)",
                new_slice_axis
            )
            .parse::<Pattern<Language>>()
            .unwrap()
            .apply_one(egraph, matched_id, subst)
        }
    }
    rewrite!(
        "slice-move-axis-composition-commutative";
        "(slice (move-axis ?tensor ?src ?dest) ?axis ?bottom ?top)" =>
        { SliceMoveAxisCompositionCommutativeApplier {
            move_axis_src: "?src".parse().unwrap(),
            move_axis_dest: "?dest".parse().unwrap(),
            slice_axis: "?axis".parse().unwrap(),
        }}
    )
}

pub fn systolic_array_vector_matrix() -> Rewrite<Language, MyAnalysis> {
    struct SystolicArrayApplier {
        a: Var,
        b: Var,
    }
    impl Applier<Language, MyAnalysis> for SystolicArrayApplier {
        fn apply_one(&self, egraph: &mut EG, eclass: Id, subst: &Subst) -> Vec<Id> {
            let a_shape = MyAnalysis::get_shape(subst[self.a], egraph);
            let b_shape = MyAnalysis::get_shape(subst[self.b], egraph);
            assert_eq!(a_shape.as_array_view().len(), 1);
            assert_eq!(b_shape.as_array_view().len(), 2);
            let rows: usize = b_shape.as_array_view()[0];
            let cols: usize = b_shape.as_array_view()[1];

            let pattern: Pattern<Language> =
                format!("(bsg-systolic-array {} {} ?a ?b)", rows, cols)
                    .parse()
                    .unwrap();

            pattern.apply_one(egraph, eclass, subst)
        }
    }

    rewrite!("systolic-array";
             // TODO(@gussmith23) should check that ?a is a vector.
             "(map-dot-product (cartesian-product ?a (move-axis ?b 1 0)))" =>
             {SystolicArrayApplier{a: "?a".parse().unwrap(), b: "?b".parse().unwrap(),}})
}

// pub fn flatten_unflatten_access_windows() -> RW {
//     rewrite!("access-windows-to-im2col";
//              "(access-windows ?access ?kernel-shape ?stride-0 ?stride-1)" =>
//              "(access-reshape
//                (access-flatten
//                 (access-windows ?access ?kernel-shape ?stride-0 ?stride-1)
//                )
//                (get-access-shape
//                 (access-windows ?access ?kernel-shape ?stride-0 ?stride-1)
//                )
//               )")
// }

pub fn flatten_unflatten_any_access() -> RW {
    rewrite!("flatten-unflatten-all-accesses";
             "?access" =>
             "(access-reshape
               (access-flatten
                ?access
               )
               (get-access-shape
                ?access
               )
              )"
             if is_access())
}

pub fn bubble_reshape_through_cartesian_product() -> RW {
    fn access_item_shapes_equal(
        left: Var,
        right: Var,
    ) -> impl Fn(&mut EG, egg::Id, &egg::Subst) -> bool {
        move |egraph, _, subst| match (&egraph[subst[left]].data, &egraph[subst[right]].data) {
            (MyAnalysisData::AccessPattern(left), MyAnalysisData::AccessPattern(right)) => {
                left.item_shape == right.item_shape
            }
            _ => false,
        }
    }
    // TODO(@gussmith23) Calculate new cartesian product shape by hand?
    // The fact that we're using get-access-shape in the rewrite feels bad,
    // somehow. I'm not fully sure why.
    // Pro: we don't have to duplicate the logic for figuring out the shape of a
    // cartesian product. Using get-access-shape just extracts the shape from
    // the access-cartesian-product.
    // Con: I'm not sure. If there's an existing bug, it'll just propagate? I
    // guess I should just make sure the cartesian product definition is right,
    // and that get-access-shape works as expected!
    rewrite!("bubble-reshape-through-cartesian-product";
             "(access-cartesian-product
               (access-reshape
                ?left-access
                ?left-shape
               )
               (access-reshape
                ?right-access
                ?right-shape
               )
              )" =>
              "(access-reshape
                (access-cartesian-product
                 ?left-access
                 ?right-access
                )
                (get-access-shape
                 (access-cartesian-product
                  (access-reshape
                   ?left-access
                   ?left-shape
                  )
                  (access-reshape
                   ?right-access
                   ?right-shape
                  )
                 )
                )
               )"
             if access_item_shapes_equal("?left-access".parse().unwrap(),
                                         "?right-access".parse().unwrap()))
}

pub fn bubble_reshape_through_compute_dot_product() -> RW {
    fn is_dot_product(op: Var) -> impl Fn(&mut EG, egg::Id, &egg::Subst) -> bool {
        move |egraph, _, subst| match &egraph[subst[op]].data {
            MyAnalysisData::ComputeType(c) => *c == super::language::ComputeType::DotProduct,
            _ => false,
        }
    }
    struct ApplierImpl(Var);
    impl Applier<Language, MyAnalysis> for ApplierImpl {
        fn apply_one(&self, egraph: &mut EG, eclass: Id, subst: &Subst) -> Vec<Id> {
            let a = match &egraph[subst[self.0]].data {
                MyAnalysisData::AccessPattern(a) => a,
                _ => panic!(),
            };

            format!(
                "(access-reshape (compute ?op ?a) (access-shape (shape {}) (shape)))",
                a.shape
                    .as_array_view()
                    .iter()
                    .map(|u: &usize| { format!("{}", u) })
                    .collect::<Vec<_>>()
                    .join(" ")
            )
            .parse::<Pattern<Language>>()
            .unwrap()
            .apply_one(egraph, eclass, subst)
        }
    }
    rewrite!("bubble-reshape-through-compute";
             "(compute ?op (access-reshape ?a ?shape))" =>
             { ApplierImpl("?shape".parse().unwrap()) }
             if is_dot_product("?op".parse().unwrap()))
}

pub fn systolic_array() -> Rewrite<Language, MyAnalysis> {
    fn constrain_access(
        access: Var,
        constraint: impl Fn(&super::language::AccessPatternData) -> bool,
    ) -> impl Fn(&mut EG, egg::Id, &egg::Subst) -> bool {
        move |egraph, _, subst| match &egraph[subst[access]].data {
            MyAnalysisData::AccessPattern(a) => constraint(a),
            _ => false,
        }
    }
    struct ApplierImpl {
        a: Var,
        b: Var,
    }
    impl Applier<Language, MyAnalysis> for ApplierImpl {
        fn apply_one(&self, egraph: &mut EG, eclass: Id, subst: &Subst) -> Vec<Id> {
            let (a, b) = match (&egraph[subst[self.a]].data, &egraph[subst[self.b]].data) {
                (MyAnalysisData::AccessPattern(a), MyAnalysisData::AccessPattern(b)) => (a, b),
                _ => panic!(),
            };
            assert_eq!(a.item_shape.ndim(), 1);
            assert_eq!(b.shape.ndim(), 1);
            assert_eq!(b.item_shape.ndim(), 1);
            let rows: usize = b.item_shape.slice()[0];
            let cols: usize = b.shape.slice()[0];

            let pattern: Pattern<Language> = format!(
                "(systolic-array {} {}
                          ?access-1
                          (access (access-transpose ?access-2 (list 1 0)) 0)
                         )",
                rows, cols
            )
            .parse()
            .unwrap();

            pattern.apply_one(egraph, eclass, subst)
        }
    }

    rewrite!("systolic-array";
             "(compute dot-product
               (access-cartesian-product
                ?access-1
                ?access-2
               )
              )
             " =>
             { ApplierImpl{a: "?access-1".parse().unwrap(), b: "?access-2".parse().unwrap(),}}
             if constrain_access("?access-1".parse().unwrap(),
                                 |a| a.shape.ndim() <= 1 && a.item_shape.ndim() == 1)
             if constrain_access("?access-2".parse().unwrap(),
                                 |a| a.shape.ndim() == 1 && a.item_shape.ndim() == 1))
}

pub fn slice_concatenate_accesses(
    axis: usize,
    dimension_greater_than: usize,
) -> Rewrite<Language, MyAnalysis> {
    fn access_has_axis(axis: usize) -> impl Fn(&mut EG, egg::Id, &egg::Subst) -> bool {
        move |egraph, id, _subst| match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => axis < a.shape.ndim() + a.item_shape.ndim(),
            _ => panic!(),
        }
    }

    fn access_dimension_greater_than(
        axis: usize,
        greater_than: usize,
    ) -> impl Fn(&mut EG, egg::Id, &egg::Subst) -> bool {
        move |egraph, id, _subst| match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                if axis < a.shape.ndim() {
                    a.shape[axis] > greater_than
                } else {
                    a.item_shape[axis - a.shape.ndim()] > greater_than
                }
            }
            _ => panic!(),
        }
    }

    fn access_dimension_is_even(axis: usize) -> impl Fn(&mut EG, egg::Id, &egg::Subst) -> bool {
        move |egraph, id, _subst| match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                if axis < a.shape.ndim() {
                    a.shape[axis] % 2 == 0
                } else {
                    a.item_shape[axis - a.shape.ndim()] % 2 == 0
                }
            }
            _ => panic!(),
        }
    }

    struct ApplierImpl {
        axis: usize,
    }
    impl Applier<Language, MyAnalysis> for ApplierImpl {
        fn apply_one(
            &self,
            egraph: &mut EG,
            id: egg::Id,
            subst: &egg::Subst,
        ) -> std::vec::Vec<egg::Id> {
            let shape = match &egraph[id].data {
                MyAnalysisData::AccessPattern(a) => a,
                _ => panic!(),
            };
            assert!(self.axis < shape.shape.ndim() + shape.item_shape.ndim());
            let low_bound = 0;
            let high_bound = if self.axis < shape.shape.ndim() {
                shape.shape[self.axis]
            } else {
                shape.item_shape[self.axis - shape.shape.ndim()]
            };
            assert_eq!(high_bound % 2, 0);
            let middle_bound = high_bound / 2;

            format!(
                "(access-concatenate (access-slice ?a {} {} {}) (access-slice ?a {} {} {}) {})",
                self.axis, low_bound, middle_bound, self.axis, middle_bound, high_bound, self.axis
            )
            .parse::<Pattern<_>>()
            .unwrap()
            .apply_one(egraph, id, subst)
        }
    }
    rewrite!(format!("slice-concatenate-access-axis-{}", axis);
             "?a" => { ApplierImpl {axis: axis} }
             // TODO(@gussmith) Wouldn't need this if we had an "access" op
             // We could have an access op that takes the access type, the
             // access config, and then a number of arguments. I.e. access-1
             // would take 1, access-2 would take 2, etc. Then we wouldn't need
             // this check.
             // TODO(@gussmith) Need to limit what gets sliced
             // I used to do this with the old split() rewrite. It's a little
             // trickier now with accesses, because you have to match only on
             // accesses to tensor literals. Not impossible, obviously. I'm just
             // being lazy right now.
             if is_access()
             if access_has_axis(axis)
             if access_dimension_is_even(axis)
             if access_dimension_greater_than(axis, dimension_greater_than))
}

pub fn slice_concatenate_tensor_accesses(
    axis: usize,
    dimension_greater_than: usize,
) -> Rewrite<Language, MyAnalysis> {
    fn access_has_axis(axis: usize) -> impl Fn(&mut EG, egg::Id, &egg::Subst) -> bool {
        move |egraph, id, _subst| match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => axis < a.shape.ndim() + a.item_shape.ndim(),
            _ => panic!(),
        }
    }

    fn access_dimension_greater_than(
        axis: usize,
        greater_than: usize,
    ) -> impl Fn(&mut EG, egg::Id, &egg::Subst) -> bool {
        move |egraph, id, _subst| match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                if axis < a.shape.ndim() {
                    a.shape[axis] > greater_than
                } else {
                    a.item_shape[axis - a.shape.ndim()] > greater_than
                }
            }
            _ => panic!(),
        }
    }

    fn access_dimension_is_even(axis: usize) -> impl Fn(&mut EG, egg::Id, &egg::Subst) -> bool {
        move |egraph, id, _subst| match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                if axis < a.shape.ndim() {
                    a.shape[axis] % 2 == 0
                } else {
                    a.item_shape[axis - a.shape.ndim()] % 2 == 0
                }
            }
            _ => panic!(),
        }
    }

    struct ApplierImpl {
        axis: usize,
    }
    impl Applier<Language, MyAnalysis> for ApplierImpl {
        fn apply_one(
            &self,
            egraph: &mut EG,
            id: egg::Id,
            subst: &egg::Subst,
        ) -> std::vec::Vec<egg::Id> {
            let shape = match &egraph[id].data {
                MyAnalysisData::AccessPattern(a) => a,
                _ => panic!(),
            };
            assert!(self.axis < shape.shape.ndim() + shape.item_shape.ndim());
            let low_bound = 0;
            let high_bound = if self.axis < shape.shape.ndim() {
                shape.shape[self.axis]
            } else {
                shape.item_shape[self.axis - shape.shape.ndim()]
            };
            assert_eq!(high_bound % 2, 0);
            let middle_bound = high_bound / 2;

            format!(
                "(access-concatenate
                  (access-slice (access-tensor ?a) {} {} {})
                  (access-slice (access-tensor ?a) {} {} {})
                  {}
                 )",
                self.axis, low_bound, middle_bound, self.axis, middle_bound, high_bound, self.axis
            )
            .parse::<Pattern<_>>()
            .unwrap()
            .apply_one(egraph, id, subst)
        }
    }
    rewrite!(format!("slice-concatenate-tensor-access-axis-{}", axis);
             "(access-tensor ?a)" => { ApplierImpl {axis: axis} }
             if access_has_axis(axis)
             if access_dimension_is_even(axis)
             if access_dimension_greater_than(axis, dimension_greater_than))
}

// TODO(@gussmith) Can also implement a collapse_nested_concatenate
pub fn collapse_nested_access_slices() -> Rewrite<Language, MyAnalysis> {
    struct ApplierImpl {
        low0: Var,
        high0: Var,
        low1: Var,
        high1: Var,
    }
    impl Applier<Language, MyAnalysis> for ApplierImpl {
        fn apply_one(&self, egraph: &mut EG, eclass: Id, subst: &Subst) -> Vec<Id> {
            let low0: usize = MyAnalysis::get_usize(subst[self.low0], egraph);
            let high0: usize = MyAnalysis::get_usize(subst[self.high0], egraph);
            let low1: usize = MyAnalysis::get_usize(subst[self.low1], egraph);
            let high1: usize = MyAnalysis::get_usize(subst[self.high1], egraph);

            let new_low: usize = low0 + low1;
            assert!(high1 - low1 <= high0 - low0);
            let new_high: usize = new_low + (high1 - low1);

            format!("(access-slice ?t ?axis {} {})", new_low, new_high)
                .parse::<Pattern<_>>()
                .unwrap()
                .apply_one(egraph, eclass, subst)
        }
    }
    rewrite!("collapse-nested-slices";
    "(access-slice (access-slice ?t ?axis ?low0 ?high0) ?axis ?low1 ?high1)" =>
    { ApplierImpl {
        low0: "?low0".parse().unwrap(),
        low1: "?low1".parse().unwrap(),
        high0: "?high0".parse().unwrap(),
        high1: "?high1".parse().unwrap(),
    }})
}

pub fn access_slice_access_move_axis_composition_commutative() -> Rewrite<Language, MyAnalysis> {
    struct ApplierImpl {
        move_axis_src: Var,
        move_axis_dest: Var,
        slice_axis: Var,
    }
    impl Applier<Language, MyAnalysis> for ApplierImpl {
        fn apply_one(&self, egraph: &mut EG, matched_id: Id, subst: &Subst) -> Vec<Id> {
            let src_axis: usize = MyAnalysis::get_usize(subst[self.move_axis_src], egraph);
            let dst_axis: usize = MyAnalysis::get_usize(subst[self.move_axis_dest], egraph);
            let old_slice_axis: usize = MyAnalysis::get_usize(subst[self.slice_axis], egraph);
            let new_slice_axis = if (old_slice_axis < src_axis && old_slice_axis < dst_axis)
                || (old_slice_axis > src_axis && old_slice_axis > dst_axis)
            {
                // Axis is unaffected if it's not between src and dst.
                old_slice_axis
            } else if old_slice_axis == dst_axis {
                src_axis
            } else if old_slice_axis <= src_axis && old_slice_axis > dst_axis {
                old_slice_axis - 1
            } else if old_slice_axis >= src_axis && old_slice_axis < dst_axis {
                old_slice_axis + 1
            } else {
                unreachable!()
            };

            format!(
                "(access-move-axis (access-slice ?tensor {} ?bottom ?top) ?src ?dest)",
                new_slice_axis
            )
            .parse::<Pattern<Language>>()
            .unwrap()
            .apply_one(egraph, matched_id, subst)
        }
    }
    rewrite!(
        "access-slice-access-move-axis-composition-commutative";
        "(access-slice (access-move-axis ?tensor ?src ?dest) ?axis ?bottom ?top)" =>
        { ApplierImpl {
            move_axis_src: "?src".parse().unwrap(),
            move_axis_dest: "?dest".parse().unwrap(),
            slice_axis: "?axis".parse().unwrap(),
        }}
    )
}

pub fn bubble_access_concatenate_through_access_move_axis() -> Rewrite<Language, MyAnalysis> {
    struct ApplierImpl {
        concatenate_axis: Var,
        src_axis: Var,
        dst_axis: Var,
    }
    impl Applier<Language, MyAnalysis> for ApplierImpl {
        fn apply_one(&self, egraph: &mut EG, eclass: Id, subst: &Subst) -> Vec<Id> {
            let original_concatenate_axis: usize =
                MyAnalysis::get_usize(subst[self.concatenate_axis], egraph);
            let src_axis: usize = MyAnalysis::get_usize(subst[self.src_axis], egraph);
            let dst_axis: usize = MyAnalysis::get_usize(subst[self.dst_axis], egraph);

            // If the move now happens /before/ the concatenate, we have to
            // figure out what the new axis for the concatenate is.
            // TODO(@gussmith23) Would be nice to have a more principled system of
            // keeping track of axes. This is where Remy's relational algebra
            // stuff could be really useful!
            let new_concatenate_axis: usize = if (original_concatenate_axis < src_axis
                && original_concatenate_axis < dst_axis)
                || (original_concatenate_axis > src_axis && original_concatenate_axis > dst_axis)
            {
                // Axis is unaffected if it's not between src and dst.
                original_concatenate_axis
            } else if original_concatenate_axis == src_axis {
                dst_axis
            } else if original_concatenate_axis < src_axis && original_concatenate_axis >= dst_axis
            {
                original_concatenate_axis + 1
            } else if original_concatenate_axis > src_axis && original_concatenate_axis <= dst_axis
            {
                original_concatenate_axis - 1
            } else {
                unreachable!()
            };

            format!(
                "(access-concatenate
                      (access-move-axis ?a ?src-axis ?dst-axis)
                      (access-move-axis ?b ?src-axis ?dst-axis) {})",
                new_concatenate_axis
            )
            .parse::<Pattern<_>>()
            .unwrap()
            .apply_one(egraph, eclass, subst)
        }
    }
    rewrite!("bubble-access-concatenate-through-access-move-axis";
        "(access-move-axis (access-concatenate ?a ?b ?concatenate-axis) ?src-axis ?dst-axis)" =>
    {
        ApplierImpl {
            concatenate_axis: "?concatenate-axis".parse().unwrap(),
            src_axis:"?src-axis".parse().unwrap(),
            dst_axis:"?dst-axis".parse().unwrap()
        }
    })
}

pub fn bubble_access_concatenate_through_access_cartesian_product_not_item_axis_left(
) -> Rewrite<Language, MyAnalysis> {
    rewrite!("bubble-access-concatenate-through-access-cartesian-product-not-item-axis-left";
             "(access-cartesian-product (access-concatenate ?t1 ?t2 ?axis) ?right)" =>
             "(access-concatenate
               (access-cartesian-product ?t1 ?right)
               (access-cartesian-product ?t2 ?right)
               ?axis
              )"
             if not_item_axis("?axis".parse().unwrap(), "?t1".parse().unwrap()))
}

pub fn bubble_access_concatenate_through_access_cartesian_product_not_item_axis_right(
) -> Rewrite<Language, MyAnalysis> {
    struct ApplierImpl {
        axis: Var,
        left: Var,
    }
    impl Applier<Language, MyAnalysis> for ApplierImpl {
        fn apply_one(&self, egraph: &mut EG, eclass: Id, subst: &Subst) -> Vec<Id> {
            let left = match &egraph[subst[self.left]].data {
                MyAnalysisData::AccessPattern(a) => a,
                _ => panic!(),
            };
            let axis = MyAnalysis::get_usize(subst[self.axis], egraph);
            format!(
                "(access-concatenate
                   (access-cartesian-product ?left ?t1)
                   (access-cartesian-product ?left ?t2)
                   {}
                  )",
                axis + left.shape.ndim()
            )
            .parse::<Pattern<_>>()
            .unwrap()
            .apply_one(egraph, eclass, subst)
        }
    }
    rewrite!("bubble-access-concatenate-through-access-cartesian-product-not-item-axis-right";
                 "(access-cartesian-product ?left (access-concatenate ?t1 ?t2 ?axis))" =>
             { ApplierImpl {
                 axis: "?axis".parse().unwrap(),
                 left: "?left".parse().unwrap(),
             } }
                 if not_item_axis("?axis".parse().unwrap(), "?t1".parse().unwrap()))
}

pub fn bubble_access_concatenate_through_access_cartesian_product_same_item_axis(
) -> Rewrite<Language, MyAnalysis> {
    struct ApplierImpl {
        axis0: Var,
        access1: Var,
    }
    impl Applier<Language, MyAnalysis> for ApplierImpl {
        fn apply_one(&self, egraph: &mut EG, matched_id: Id, subst: &Subst) -> Vec<Id> {
            let a1 = match &egraph[subst[self.access1]].data {
                MyAnalysisData::AccessPattern(a) => a,
                _ => panic!(),
            };
            let axis0 = MyAnalysis::get_usize(subst[self.axis0], egraph);

            let new_axis = axis0 + a1.shape.ndim() + 1;

            format!(
                "(access-concatenate
                    (access-cartesian-product ?t1 ?t3)
                    (access-cartesian-product ?t2 ?t4)
                    {})",
                new_axis
            )
            .parse::<Pattern<_>>()
            .unwrap()
            .apply_one(egraph, matched_id, subst)
        }
    }
    rewrite!("bubble-access-concatenate-through-access-cartesian-product-same-axis";
             "(access-cartesian-product (access-concatenate ?t1 ?t2 ?axis0) (access-concatenate ?t3 ?t4 ?axis1))" =>
             { ApplierImpl {
                 axis0: "?axis0".parse().unwrap(),
                 access1: "?t3".parse().unwrap()
             }}
             if same_item_axis("?axis0".parse().unwrap(), "?t1".parse().unwrap(), "?axis1".parse().unwrap(), "?t3".parse().unwrap())
    )
}

/// When bubbling up through (compute dot-product ...), what you do depends on
/// whether you're concatenating along an item axis or not. If you are not, it's
/// easy: you just bubble it straight through. If you are, then you are
/// concatenating along an axis that's getting reduced in the reduction sum. So
/// we need to explicitly insert another reduction.
pub fn bubble_access_concatenate_through_compute_dot_product_not_item_axis(
) -> Rewrite<Language, MyAnalysis> {
    rewrite!("bubble-access-concatenate-through-compute-dot-product-not-item-axis";
             "(compute dot-product (access-concatenate ?a0 ?a1 ?axis))" =>
             "(access-concatenate
               (compute dot-product ?a0)
               (compute dot-product ?a1)
               ?axis
              )"
             if not_item_axis("?axis".parse().unwrap(), "?a0".parse().unwrap()))
}

pub fn bubble_access_concatenate_through_compute_dot_product_item_axis(
) -> Rewrite<Language, MyAnalysis> {
    rewrite!("bubble-access-concatenate-through-compute-dot-product-item-axis";
             "(compute dot-product (access-concatenate ?a0 ?a1 ?axis))" =>
             "(compute reduce-sum
               (access-pair
                (compute dot-product ?a0)
                (compute dot-product ?a1)
               )
              )"
             if item_axis("?axis".parse().unwrap(), "?a0".parse().unwrap()))
}

pub fn bubble_access_concatenate_through_access() -> Rewrite<Language, MyAnalysis> {
    rewrite!("bubble-access-concatenate-through-access";
             "(access (access-concatenate ?a0 ?a1 ?concatenate-axis) ?access-axis)" =>
             "(access-concatenate
               (access ?a0 ?access-axis)
               (access ?a1 ?access-axis)
               ?concatenate-axis
              )")
}

pub fn bubble_access_concatenate_through_access_slice() -> Rewrite<Language, MyAnalysis> {
    struct ApplierImpl {
        a0: Var,
        low: Var,
        high: Var,
        concatenate_axis: Var,
        slice_axis: Var,
    }
    impl Applier<Language, MyAnalysis> for ApplierImpl {
        fn apply_one(&self, egraph: &mut EG, matched_id: Id, subst: &Subst) -> Vec<Id> {
            let a0_shape = match &egraph[subst[self.a0]].data {
                MyAnalysisData::AccessPattern(a) => a,
                _ => panic!(),
            };
            let low = MyAnalysis::get_usize(subst[self.low], egraph);
            let high = MyAnalysis::get_usize(subst[self.high], egraph);
            let concatenate_axis = MyAnalysis::get_usize(subst[self.concatenate_axis], egraph);
            let slice_axis = MyAnalysis::get_usize(subst[self.slice_axis], egraph);

            let a0_dim_value = if concatenate_axis < a0_shape.shape.ndim() {
                a0_shape.shape[concatenate_axis]
            } else {
                a0_shape.item_shape[concatenate_axis - a0_shape.shape.ndim()]
            };

            if slice_axis != concatenate_axis {
                "(access-concatenate
                  (access-slice ?a0 ?slice-axis ?low ?high)
                  (access-slice ?a1 ?slice-axis ?low ?high)
                  ?concatenate-axis
                 )
                "
                .parse::<Pattern<_>>()
                .unwrap()
                .apply_one(egraph, matched_id, subst)
            } else if low < a0_dim_value && high <= a0_dim_value {
                // only in a0
                "(access-slice ?a0 ?slice-axis ?low ?high)"
                    .parse::<Pattern<_>>()
                    .unwrap()
                    .apply_one(egraph, matched_id, subst)
            } else if low >= a0_dim_value && high >= a0_dim_value {
                // only in a1
                // Adjust low/high indices
                let (low, high) = (low - a0_dim_value, high - a0_dim_value);
                format!("(access-slice ?a1 ?slice-axis {} {})", low, high)
                    .parse::<Pattern<_>>()
                    .unwrap()
                    .apply_one(egraph, matched_id, subst)
            } else if low < a0_dim_value && high >= a0_dim_value {
                // split between a0 and a1
                // Adjust slice indices
                let a0_low = low;
                let a0_high = a0_dim_value;
                let a1_low = 0;
                let a1_high = high - a0_dim_value;
                format!(
                    "(access-concatenate
                          (access-slice ?a0 ?slice-axis {} {})
                          (access-slice ?a1 ?slice-axis {} {})
                          ?concatenate-axis
                         )
                ",
                    a0_low, a0_high, a1_low, a1_high
                )
                .parse::<Pattern<_>>()
                .unwrap()
                .apply_one(egraph, matched_id, subst)
            } else {
                unreachable!()
            }
        }
    }
    rewrite!("bubble-access-concatenate-through-access-slice";
    "(access-slice (access-concatenate ?a0 ?a1 ?concatenate-axis) ?slice-axis ?low ?high)" =>
    {
        ApplierImpl {
            a0: "?a0".parse().unwrap(),
            low: "?low".parse().unwrap(),
            high: "?high".parse().unwrap(),
            concatenate_axis: "?concatenate-axis".parse().unwrap(),
            slice_axis: "?slice-axis".parse().unwrap(),
        }
    })
}

#[cfg(test)]
mod tests {

    use super::*;
    use egg::{Pattern, Runner, Searcher};
    use ndarray::IxDyn;

    #[test]
    fn split() {
        test_logger::ensure_env_logger_initialized();

        let program = "t-32-32".parse().unwrap();

        let rws = vec![
            super::split(0, 16, true),
            super::split(1, 16, true),
            super::collapse_nested_slices(),
        ];

        let mut egraph = EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        egraph.add_expr(&program);
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        assert_eq!(
            "(slice (slice t-32-32 1 0 16) 0 0 16)"
                .parse::<Pattern<_>>()
                .unwrap()
                .search(&runner.egraph)
                .len(),
            1
        );

        assert_eq!(
            "(slice (slice t-32-32 1 16 32) 0 0 16)"
                .parse::<Pattern<_>>()
                .unwrap()
                .search(&runner.egraph)
                .len(),
            1
        );

        assert_eq!(
            "(slice (slice t-32-32 1 0 16) 0 16 32)"
                .parse::<Pattern<_>>()
                .unwrap()
                .search(&runner.egraph)
                .len(),
            1
        );

        assert_eq!(
            "(slice (slice t-32-32 1 16 32) 0 16 32)"
                .parse::<Pattern<_>>()
                .unwrap()
                .search(&runner.egraph)
                .len(),
            1
        );
    }

    #[test]
    fn slice_move_axis() {
        test_logger::ensure_env_logger_initialized();

        let program = "(slice (move-axis t-32-32 0 1) 0 0 16)".parse().unwrap();

        let rws = vec![super::slice_move_axis_composition_commutative()];

        let mut egraph = EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        egraph.add_expr(&program);
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        assert_eq!(
            "(move-axis (slice t-32-32 1 0 16) 0 1)"
                .parse::<Pattern<_>>()
                .unwrap()
                .search(&runner.egraph)
                .len(),
            1
        );

        let program = "(slice (move-axis t-32-32 1 1) 0 0 16)".parse().unwrap();

        let rws = vec![super::slice_move_axis_composition_commutative()];

        let mut egraph = EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        egraph.add_expr(&program);
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        assert_eq!(
            "(move-axis (slice t-32-32 0 0 16) 1 1)"
                .parse::<Pattern<_>>()
                .unwrap()
                .search(&runner.egraph)
                .len(),
            1
        );

        let program = "(slice (move-axis t-32-32 0 0) 1 0 16)".parse().unwrap();

        let rws = vec![super::slice_move_axis_composition_commutative()];

        let mut egraph = EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        egraph.add_expr(&program);
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        assert_eq!(
            "(move-axis (slice t-32-32 1 0 16) 0 0)"
                .parse::<Pattern<_>>()
                .unwrap()
                .search(&runner.egraph)
                .len(),
            1
        );
    }

    #[test]
    fn flatten_unflatten_access_windows() {
        test_logger::ensure_env_logger_initialized();

        let program = "
         (access-windows
          (access (access-tensor t-3-32-32) 3)
          (slice-shape (shape-of t-8-3-3-3) 1)
          (shape 1 1 2)
         )
         "
        .parse()
        .unwrap();

        let rws = vec![super::flatten_unflatten_any_access()];
        let mut egraph = EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        egraph.add_expr(&program);
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        assert_eq!(
            "
            (access-reshape
             (access-flatten
              (access-windows
               (access (access-tensor t-3-32-32) 3)
               (slice-shape (shape-of t-8-3-3-3) 1)
               (shape 1 1 2)
              )
             )
             (get-access-shape
              (access-windows
               (access (access-tensor t-3-32-32) 3)
               (slice-shape (shape-of t-8-3-3-3) 1)
               (shape 1 1 2)
              )
             )
            )
            "
            .parse::<Pattern<_>>()
            .unwrap()
            .search(&runner.egraph)
            .len(),
            1
        );
    }

    #[test]
    fn bubble_reshape_through_cartesian_product() {
        test_logger::ensure_env_logger_initialized();

        let program = "
         (access-cartesian-product
          (access (access-tensor t-8-3-3-3) 1)
          (access-squeeze
           (access-windows
            (access (access-tensor t-3-32-32) 3)
            (slice-shape (shape-of t-8-3-3-3) 1)
            (shape 1 1 1)
           )
           0
          )
         )
        "
        .parse()
        .unwrap();
        let rws = vec![
            super::flatten_unflatten_any_access(),
            super::bubble_reshape_through_cartesian_product(),
        ];
        let mut egraph = EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        let matches = "
            (access-reshape
             (access-cartesian-product
              (access-flatten
               (access (access-tensor t-8-3-3-3) 1)
              )
              (access-flatten
               (access-squeeze
                (access-windows
                 (access (access-tensor t-3-32-32) 3)
                 (slice-shape (shape-of t-8-3-3-3) 1)
                 (shape 1 1 1)
                )
                0
               )
              )
             )
             ?shape
            )
            "
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);

        // ?shape should be what we expect. This is kind of over-testing, in the
        // sense that, if we find the above pattern, then the analysis data is
        // guaranteed (for now) to match, and so the shapes should be the same,
        // but I figure over-testing is better than under-testing (given that
        // "for now").
        match &runner.egraph[matches.substs[0]["?shape".parse().unwrap()]].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.item_shape, IxDyn(&[2, 3, 3, 3]));
                assert_eq!(a.shape, IxDyn(&[8, 30, 30]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn bubble_reshape_through_dot_product() {
        test_logger::ensure_env_logger_initialized();

        let program = "
         (compute dot-product
          (access-reshape (access (access-tensor t-1024-2-256) 1) (access-shape (shape 32 32) (shape 2 16 16)))
         )
        "
        .parse()
        .unwrap();
        let rws = vec![super::bubble_reshape_through_compute_dot_product()];
        let mut egraph = EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        let matches = "
            (access-reshape
             (compute dot-product
              (access (access-tensor t-1024-2-256) 1)
             )
             ?shape
            )
            "
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);

        match &runner.egraph[matches.substs[0]["?shape".parse().unwrap()]].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.item_shape, IxDyn(&[]));
                assert_eq!(a.shape, IxDyn(&[32, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn conv2d_im2col_systolic_array() {
        let program = "
         (compute dot-product
          (access-cartesian-product
           (access (access-tensor t-8-3-3-3) 1)
           (access-squeeze
            (access-windows
             (access (access-tensor t-3-32-32) 3)
             (slice-shape (shape-of t-8-3-3-3) 1)
             (shape 1 1 1)
            )
            0
           )
          )
         )
        "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);

        let rws = vec![
            super::flatten_unflatten_any_access(),
            super::bubble_reshape_through_cartesian_product(),
            super::bubble_reshape_through_compute_dot_product(),
            super::systolic_array(),
        ];

        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        let matches = "
            (access-reshape
             (systolic-array 27 900
              (access-flatten (access (access-tensor t-8-3-3-3) 1))
              (access
               (access-transpose
                (access-flatten
                 (access-squeeze
                  (access-windows
                   (access (access-tensor t-3-32-32) 3)
                   (slice-shape (shape-of t-8-3-3-3) 1)
                   (shape 1 1 1)
                  )
                  0
                 )
                )
                (list 1 0)
               )
               0
              )
             )
             ?shape
            )
            "
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn slice_concatenate_accesses() {
        test_logger::ensure_env_logger_initialized();

        let program = "(access (access-tensor t-32-32) 1)".parse().unwrap();

        let rws = vec![
            super::slice_concatenate_accesses(0, 16),
            super::slice_concatenate_accesses(1, 16),
            super::collapse_nested_access_slices(),
        ];

        let mut egraph = EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        egraph.add_expr(&program);
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        assert_eq!(
            "(access-slice (access-slice (access (access-tensor t-32-32) 1) 1 0 16) 0 0 16)"
                .parse::<Pattern<_>>()
                .unwrap()
                .search(&runner.egraph)
                .len(),
            1
        );

        assert_eq!(
            "(access-slice (access-slice (access (access-tensor t-32-32) 1) 1 16 32) 0 0 16)"
                .parse::<Pattern<_>>()
                .unwrap()
                .search(&runner.egraph)
                .len(),
            1
        );

        assert_eq!(
            "(access-slice (access-slice (access (access-tensor t-32-32) 1) 1 0 16) 0 16 32)"
                .parse::<Pattern<_>>()
                .unwrap()
                .search(&runner.egraph)
                .len(),
            1
        );

        assert_eq!(
            "(access-slice (access-slice (access (access-tensor t-32-32) 1) 1 16 32) 0 16 32)"
                .parse::<Pattern<_>>()
                .unwrap()
                .search(&runner.egraph)
                .len(),
            1
        );
    }

    #[test]
    fn access_slice_access_move_axis_composition_commutative_0() {
        let program = "
        (access-slice (access-move-axis (access (access-tensor t-3-32-32) 1) 2 0) 0 16 32)
        "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        let rws = vec![super::access_slice_access_move_axis_composition_commutative()];
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        let matches =
            "(access-move-axis (access-slice (access (access-tensor t-3-32-32) 1) 2 16 32) 2 0)"
                .parse::<Pattern<_>>()
                .unwrap()
                .search_eclass(&runner.egraph, id)
                .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn access_slice_access_move_axis_composition_commutative_1() {
        let program = "
        (access-slice (access-move-axis (access (access-tensor t-3-32-32) 1) 2 0) 1 0 2)
        "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        let rws = vec![super::access_slice_access_move_axis_composition_commutative()];
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        let matches =
            "(access-move-axis (access-slice (access (access-tensor t-3-32-32) 1) 0 0 2) 2 0)"
                .parse::<Pattern<_>>()
                .unwrap()
                .search_eclass(&runner.egraph, id)
                .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn access_slice_access_move_axis_composition_commutative_2() {
        let program = "
        (access-slice (access-move-axis (access (access-tensor t-3-32-32) 1) 1 0) 2 0 16)
        "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        let rws = vec![super::access_slice_access_move_axis_composition_commutative()];
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        let matches =
            "(access-move-axis (access-slice (access (access-tensor t-3-32-32) 1) 2 0 16) 1 0)"
                .parse::<Pattern<_>>()
                .unwrap()
                .search_eclass(&runner.egraph, id)
                .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn access_slice_access_move_axis_composition_commutative_3() {
        let program =
            "(access-slice (access-move-axis (access (access-tensor t-3-32-32) 1) 0 1) 1 0 2)"
                .parse()
                .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        let rws = vec![super::access_slice_access_move_axis_composition_commutative()];
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        let matches =
            "(access-move-axis (access-slice (access (access-tensor t-3-32-32) 1) 0 0 2) 0 1)"
                .parse::<Pattern<_>>()
                .unwrap()
                .search_eclass(&runner.egraph, id)
                .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn bubble_access_concatenate_through_access_move_axis_0() {
        let program = "(access-move-axis (access-concatenate (access (access-tensor t-3-32-32) 1) (access (access-tensor t-3-32-32) 1) 0) 0 2)"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        let rws = vec![super::bubble_access_concatenate_through_access_move_axis()];
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        let matches =
            "(access-concatenate (access-move-axis (access (access-tensor t-3-32-32) 1) 0 2) (access-move-axis (access (access-tensor t-3-32-32) 1) 0 2) 2)"
            .parse::<Pattern<_>>()
            .unwrap()
            .search_eclass(&runner.egraph, id)
            .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn bubble_access_concatenate_through_access_move_axis_1() {
        let program = "(access-move-axis (access-concatenate (access (access-tensor t-3-32-32) 1) (access (access-tensor t-3-32-32) 1) 1) 0 2)"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        let rws = vec![super::bubble_access_concatenate_through_access_move_axis()];
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        let matches =
            "(access-concatenate (access-move-axis (access (access-tensor t-3-32-32) 1) 0 2) (access-move-axis (access (access-tensor t-3-32-32) 1) 0 2) 0)"
            .parse::<Pattern<_>>()
            .unwrap()
            .search_eclass(&runner.egraph, id)
            .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn bubble_access_concatenate_through_access_move_axis_2() {
        let program = "(access-move-axis (access-concatenate (access (access-tensor t-3-32-32) 1) (access (access-tensor t-3-32-32) 1) 2) 0 1)"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        let rws = vec![super::bubble_access_concatenate_through_access_move_axis()];
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        let matches =
            "(access-concatenate (access-move-axis (access (access-tensor t-3-32-32) 1) 0 1) (access-move-axis (access (access-tensor t-3-32-32) 1) 0 1) 2)"
            .parse::<Pattern<_>>()
            .unwrap()
            .search_eclass(&runner.egraph, id)
            .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn bubble_access_concatenate_through_access_move_axis_3() {
        let program = "(access-move-axis (access-concatenate (access (access-tensor t-3-32-32) 1) (access (access-tensor t-3-32-32) 1) 1) 2 0)"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        let rws = vec![super::bubble_access_concatenate_through_access_move_axis()];
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        let matches =
            "(access-concatenate (access-move-axis (access (access-tensor t-3-32-32) 1) 2 0) (access-move-axis (access (access-tensor t-3-32-32) 1) 2 0) 2)"
            .parse::<Pattern<_>>()
            .unwrap()
            .search_eclass(&runner.egraph, id)
            .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn bubble_access_concatenate_through_access_cartesian_product_left() {
        let program = "
             (access-cartesian-product
              (access-concatenate (access (access-tensor t-3-32-32) 1) (access (access-tensor t-3-32-32) 1) 0)
              (access-concatenate (access (access-tensor t-3-32-32) 1) (access (access-tensor t-3-32-32) 1) 0)
             )"
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        let rws = vec![
            super::bubble_access_concatenate_through_access_cartesian_product_not_item_axis_left(),
        ];
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        let matches = "
             (access-concatenate
              (access-cartesian-product
               (access (access-tensor t-3-32-32) 1)
               (access-concatenate (access (access-tensor t-3-32-32) 1) (access (access-tensor t-3-32-32) 1) 0)
              )
              (access-cartesian-product
               (access (access-tensor t-3-32-32) 1)
               (access-concatenate (access (access-tensor t-3-32-32) 1) (access (access-tensor t-3-32-32) 1) 0)
              )
              0
             )"
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn bubble_access_concatenate_through_access_cartesian_product_right() {
        let program = "
             (access-cartesian-product
              (access-concatenate (access (access-tensor t-3-32-32) 1) (access (access-tensor t-3-32-32) 1) 0)
              (access-concatenate (access (access-tensor t-3-32-32) 1) (access (access-tensor t-3-32-32) 1) 0)
             )"
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        let rws = vec![
            super::bubble_access_concatenate_through_access_cartesian_product_not_item_axis_right(),
        ];
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        let matches = "
             (access-concatenate
              (access-cartesian-product
               (access-concatenate (access (access-tensor t-3-32-32) 1) (access (access-tensor t-3-32-32) 1) 0)
               (access (access-tensor t-3-32-32) 1)
              )
              (access-cartesian-product
               (access-concatenate (access (access-tensor t-3-32-32) 1) (access (access-tensor t-3-32-32) 1) 0)
               (access (access-tensor t-3-32-32) 1)
              )
              1
             )"
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn bubble_access_concatenate_through_access_cartesian_product_same_item_axis_0() {
        let program = "
             (access-cartesian-product
              (access-concatenate (access (access-tensor t-3-32-32) 1) (access (access-tensor t-3-32-32) 1) 1)
              (access-concatenate (access (access-tensor t-3-32-32) 1) (access (access-tensor t-3-32-32) 1) 1)
             )"
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        let rws = vec![
            super::bubble_access_concatenate_through_access_cartesian_product_same_item_axis(),
        ];
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        let matches = "
             (access-concatenate
              (access-cartesian-product
               (access (access-tensor t-3-32-32) 1)
               (access (access-tensor t-3-32-32) 1)
              )
              (access-cartesian-product
               (access (access-tensor t-3-32-32) 1)
               (access (access-tensor t-3-32-32) 1)
              )
              3
             )"
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn bubble_access_concatenate_through_access_cartesian_product_same_item_axis_1() {
        let program = "
             (access-cartesian-product
              (access-concatenate (access (access-tensor t-3-32-32) 0) (access (access-tensor t-3-32-32) 0) 2)
              (access-concatenate (access (access-tensor t-3-32-32) 0) (access (access-tensor t-3-32-32) 0) 2)
             )"
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        let rws = vec![
            super::bubble_access_concatenate_through_access_cartesian_product_same_item_axis(),
        ];
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        let matches = "
             (access-concatenate
              (access-cartesian-product
               (access (access-tensor t-3-32-32) 0)
               (access (access-tensor t-3-32-32) 0)
              )
              (access-cartesian-product
               (access (access-tensor t-3-32-32) 0)
               (access (access-tensor t-3-32-32) 0)
              )
              3
             )"
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn bubble_access_concatenate_through_compute_dot_product_not_item_access() {
        let program = "
             (compute dot-product
              (access-concatenate
               (access (access-tensor t-3-32-32) 1)
               (access (access-tensor t-3-32-32) 1)
               0
              )
             )"
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        let rws =
            vec![super::bubble_access_concatenate_through_compute_dot_product_not_item_axis()];
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        let matches = "
             (access-concatenate
              (compute dot-product (access (access-tensor t-3-32-32) 1))
              (compute dot-product (access (access-tensor t-3-32-32) 1))
              0
             )
             "
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn bubble_access_concatenate_through_compute_dot_product_item_access_0() {
        let program = "
             (compute dot-product
              (access-concatenate
               (access (access-tensor t-3-32-32) 1)
               (access (access-tensor t-3-32-32) 1)
               1
              )
             )"
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        let rws = vec![super::bubble_access_concatenate_through_compute_dot_product_item_axis()];
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        let matches = "
             (compute reduce-sum
              (access-pair
               (compute dot-product (access (access-tensor t-3-32-32) 1))
               (compute dot-product (access (access-tensor t-3-32-32) 1))
              )
             )
             "
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn bubble_access_concatenate_through_compute_dot_product_item_access_1() {
        let program = "
             (compute dot-product
              (access-concatenate
               (access (access-tensor t-3-32-32) 1)
               (access (access-tensor t-3-32-32) 1)
               2
              )
             )"
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        let rws = vec![super::bubble_access_concatenate_through_compute_dot_product_item_axis()];
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        let matches = "
             (compute reduce-sum
              (access-pair
               (compute dot-product (access (access-tensor t-3-32-32) 1))
               (compute dot-product (access (access-tensor t-3-32-32) 1))
              )
             )
             "
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn bubble_access_concatenate_through_access() {
        let program = "
             (access
              (access-concatenate
               (access (access-tensor t-3-32-32) 1)
               (access (access-tensor t-3-32-32) 1)
               2
              )
              0
             )"
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        let rws = vec![super::bubble_access_concatenate_through_access()];
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        let matches = "
             (access-concatenate
              (access
               (access (access-tensor t-3-32-32) 1)
               0
              )
              (access
               (access (access-tensor t-3-32-32) 1)
               0
              )
              2
             )
             "
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn bubble_access_concatenate_through_access_slice_0() {
        let program = "
             (access-slice
              (access-concatenate
               (access (access-tensor t-32-32) 1)
               (access (access-tensor t-32-64) 1)
               1
              )
              1 0 16
             )"
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        let rws = vec![super::bubble_access_concatenate_through_access_slice()];
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        let matches = "(access-slice (access (access-tensor t-32-32) 1) 1 0 16)"
            .parse::<Pattern<_>>()
            .unwrap()
            .search_eclass(&runner.egraph, id)
            .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn bubble_access_concatenate_through_access_slice_1() {
        let program = "
             (access-slice
              (access-concatenate
               (access (access-tensor t-32-32) 1)
               (access (access-tensor t-32-64) 1)
               1
              )
              1 48 64
             )"
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        let rws = vec![super::bubble_access_concatenate_through_access_slice()];
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        let matches = "(access-slice (access (access-tensor t-32-64) 1) 1 16 32)"
            .parse::<Pattern<_>>()
            .unwrap()
            .search_eclass(&runner.egraph, id)
            .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn bubble_access_concatenate_through_access_slice_2() {
        let program = "
             (access-slice
              (access-concatenate
               (access (access-tensor t-32-32) 1)
               (access (access-tensor t-32-64) 1)
               1
              )
              1 16 48
             )"
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        let rws = vec![super::bubble_access_concatenate_through_access_slice()];
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        let matches = "
            (access-concatenate
             (access-slice
              (access (access-tensor t-32-32) 1)
              1 16 32
             )
             (access-slice
              (access (access-tensor t-32-64) 1)
              1 0 16
             )
             1
            )"
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn slice_concatenate_tensor_accesses() {
        let program = "(access-tensor t-32-32)".parse().unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        let rws = vec![
            super::slice_concatenate_tensor_accesses(0, 1),
            super::slice_concatenate_tensor_accesses(1, 1),
        ];
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        let matches = "
            (access-concatenate
             (access-slice
              (access-tensor t-32-32)
              0 0 16
             )
             (access-slice
              (access-tensor t-32-32)
              0 16 32
             )
             0
            )"
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);

        let matches = "
            (access-concatenate
             (access-slice
              (access-tensor t-32-32)
              1 0 16
             )
             (access-slice
              (access-tensor t-32-32)
              1 16 32
             )
             1
            )"
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);

        let matches = "
            (access-concatenate
             (access-slice
              (access-concatenate
               (access-slice
                (access-tensor t-32-32)
                1 0 16
               )
               (access-slice
                (access-tensor t-32-32)
                1 16 32
               )
               1
              )
              0 0 16
             )
             (access-slice
              (access-tensor t-32-32)
              0 16 32
             )
             0
            )"
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }
}
