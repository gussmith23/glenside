use std::str::FromStr;

use crate::language::{from_relay, ShapeData};

use super::{Language, MyAnalysis, MyAnalysisData, PadType, RangeSet2};
use egg::{
    rewrite, Applier, ConditionalApplier, EGraph, ENodeOrVar, Id, Pattern, Rewrite, Subst, Var,
};
use egg::{PatternAst, RecExpr};
use itertools::Itertools;
use ndarray::Dimension;
use ndarray::IxDyn;

type EG = EGraph<Language, MyAnalysis>;
type RW = Rewrite<Language, MyAnalysis>;

fn constrain_vars(
    vars: Vec<Var>,
    constraint: impl Fn(Vec<MyAnalysisData>) -> bool,
) -> impl Fn(&mut EG, egg::Id, &egg::Subst) -> bool {
    move |egraph, _, subst| {
        constraint(
            vars.iter()
                .map(|var| &egraph[subst[*var]].data)
                .cloned()
                .collect::<Vec<_>>(),
        )
    }
}

fn match_shape_data(data: &MyAnalysisData) -> Vec<usize> {
    match data {
        MyAnalysisData::Shape(x) => x.shape.slice().to_vec(),
        MyAnalysisData::AccessPattern(access) => access.shape.slice().to_vec(),
        _ => panic!("not enough info for rewriting"),
    }
}

fn constrain_access(
    access: Var,
    constraint: impl Fn(&super::language::AccessPatternData) -> bool,
) -> impl Fn(&mut EG, egg::Id, &egg::Subst) -> bool {
    move |egraph, _, subst| match &egraph[subst[access]].data {
        MyAnalysisData::AccessPattern(a) => constraint(a),
        _ => false,
    }
}

fn access_has_axis(axis: usize) -> impl Fn(&mut EG, egg::Id, &egg::Subst) -> bool {
    move |egraph, id, _subst| match &egraph[id].data {
        MyAnalysisData::AccessPattern(a) => axis < a.shape.ndim() + a.item_shape.ndim(),
        _ => panic!(),
    }
}

fn is_access() -> impl Fn(&mut EG, egg::Id, &egg::Subst) -> bool {
    move |egraph, id, _| match &egraph[id].data {
        MyAnalysisData::AccessPattern(_) => true,
        _ => false,
    }
}

/// True if a list is equal to 0..len(list)
fn list_is_0_through_len(list: Var) -> impl Fn(&mut EG, egg::Id, &egg::Subst) -> bool {
    move |egraph, _id, subst| {
        let list = match &egraph[subst[list]].data {
            MyAnalysisData::List(l) => l,
            _ => panic!(),
        };

        *list == (0..list.len()).collect::<Vec<_>>()
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
        MyAnalysisData::Shape(_) => true,
        _ => false,
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

pub fn flatten_dot_product_to_dense() -> RW {
    rewrite!("flatten-dot-product-to-dense";
        "(compute dot-product (access-cartesian-product 
                                (access-flatten ?x) 
                                (access-flatten ?w)))"
        => "(relay-operator-call relay-dense (access-flatten ?x) (access-flatten ?w))")
}

pub fn relay_dense_rewrite() -> RW {
    // struct RelayOperatorRewriteApplier(Var);
    // impl Applier<Language, MyAnalysis> for RelayOperatorRewriteApplier {
    //     fn apply_one(
    //         &self,
    //         egraph: &mut EG,
    //         id: egg::Id,
    //         subst: &egg::Subst,
    //     ) -> std::vec::Vec<egg::Id> {

    //     }
    // }
    rewrite! ("dense-rewrites"; 
                "(relay-operator-call relay-dense ?access-x ?access-w)" 
                => "(compute dot-product (access-cartesian-product ?access-x ?access-w))")
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
    struct ApplierImpl(Var);
    impl Applier<Language, MyAnalysis> for ApplierImpl {
        fn apply_one(&self, egraph: &mut EG, eclass: Id, subst: &Subst) -> Vec<Id> {
            let shape = match &egraph[subst[self.0]].data {
                MyAnalysisData::AccessPattern(a) => a,
                _ => panic!(),
            };

            format!(
                "(access-reshape
                      (access-flatten
                       ?access
                      )
                      (access-shape
                       (shape {})
                       (shape {})
                      )
                     )",
                shape
                    .shape
                    .slice()
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(" "),
                shape
                    .item_shape
                    .slice()
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            )
            .parse::<Pattern<_>>()
            .unwrap()
            .apply_one(egraph, eclass, subst)
        }
    }
    rewrite!("flatten-unflatten-all-accesses";
             "?access" =>
             { ApplierImpl("?access".parse().unwrap()) }
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

    struct ApplierImpl {
        left_shape: Var,
        right_shape: Var,
    }
    impl Applier<Language, MyAnalysis> for ApplierImpl {
        fn apply_one(&self, egraph: &mut EG, eclass: Id, subst: &Subst) -> Vec<Id> {
            let left_shape = match &egraph[subst[self.left_shape]].data {
                MyAnalysisData::AccessPattern(a) => a,
                _ => panic!(),
            };
            let right_shape = match &egraph[subst[self.right_shape]].data {
                MyAnalysisData::AccessPattern(a) => a,
                _ => panic!(),
            };

            // TODO(@gussmith23) Duplicated logic for cartprod shape calculation

            assert_eq!(
                left_shape.item_shape, right_shape.item_shape,
                "Cartesian product argument shapes must match"
            );

            let new_shape = IxDyn(
                left_shape
                    .shape
                    .as_array_view()
                    .iter()
                    .cloned()
                    .chain(right_shape.shape.as_array_view().iter().cloned())
                    .collect::<Vec<usize>>()
                    .as_slice(),
            );
            let new_item_shape = IxDyn(
                std::iter::once(2)
                    .chain(left_shape.item_shape.as_array_view().iter().cloned())
                    .collect::<Vec<usize>>()
                    .as_slice(),
            );

            assert_eq!(
                new_shape.as_array_view().iter().product::<usize>()
                    * new_item_shape.as_array_view().iter().product::<usize>(),
                left_shape.shape.as_array_view().iter().product::<usize>()
                    * right_shape.shape.as_array_view().iter().product::<usize>()
                    * 2
                    * left_shape
                        .item_shape
                        .as_array_view()
                        .iter()
                        .product::<usize>()
            );

            format!(
                "(access-reshape
                      (access-cartesian-product
                       ?left-access
                       ?right-access
                      )
                      (access-shape
                       (shape {})
                       (shape {})
                      )
                     )",
                new_shape
                    .slice()
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(" "),
                new_item_shape
                    .slice()
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            )
            .parse::<Pattern<_>>()
            .unwrap()
            .apply_one(egraph, eclass, subst)
        }
    }

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
             { ApplierImpl {
                 left_shape: "?left-shape".parse().unwrap(),
                 right_shape: "?right-shape".parse().unwrap(),
             }}
             if access_item_shapes_equal("?left-access".parse().unwrap(),
                                         "?right-access".parse().unwrap()))
}

/// More general rewrite
/// because it's using the properties of Glenside expressions
///

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

pub fn conv2d_on_hlscnn() -> RW {
    fn is_one(g: Var) -> impl Fn(&mut EG, egg::Id, &egg::Subst) -> bool {
        move |egraph, _, subst| match &egraph[subst[g]].data {
            MyAnalysisData::Usize(group) => *group == 1 as usize,
            _ => false,
        }
    }
    rewrite!("conv2d-on-hlscnn";
            "(relay-operator-call relay-conv2d ?data ?kernel ?strides ?padding ?group ?channels ?kshape ?layout ?klayout)"
            =>
            "(accelerator-call hlscnn-conv2d ?data ?kernel ?strides ?padding ?group ?channels ?kshape ?layout ?klayout (shape 0))"
            if is_one("?group".parse().unwrap()))
}

pub fn access_reshape_to_relay() -> RW {
    rewrite!("access-reshape-to-reshape";
        "(access-reshape ?access (access-shape ?shape (shape)))" => "(relay-operator-call relay-reshape ?access ?shape)")
}

pub fn dot_product_with_vta() -> RW {
    fn dim_supported(x: Var) -> impl Fn(&mut EG, egg::Id, &egg::Subst) -> bool {
        move |egraph, _, subst| match &egraph[subst[x]].data {
            MyAnalysisData::AccessPattern(access) => {
                access.shape.ndim() + access.item_shape.ndim() == 2
            }
            MyAnalysisData::Shape(shape) => shape.shape.ndim() == 2,
            _ => false,
        }
    }
    rewrite!("dot-product-on-vta";
        "(compute dot-product (access-cartesian-product ?x ?w))"
        => "(accelerator-call vta-dense ?x ?w (shape 0))"
            if dim_supported("?x".parse().unwrap())
            if dim_supported("?w".parse().unwrap()))
}

pub fn dot_product_to_linear() -> RW {
    struct ApplierImpl(Var, Var);
    impl Applier<Language, MyAnalysis> for ApplierImpl {
        fn apply_one(&self, egraph: &mut EG, eclass: Id, subst: &Subst) -> Vec<Id> {
            // let x_shape = match_shape_data(&egraph[subst[self.0]].data);
            let w_shape = match_shape_data(&egraph[subst[self.1]].data);
            format!(
                "(accelerator-call flex-linear ?x ?w (constant-tensor 0 (shape 1 {})) (shape 0))",
                w_shape[1]
            )
            .parse::<Pattern<_>>()
            .unwrap()
            .apply_one(egraph, eclass, subst)
        }
    }
    rewrite!("dot-product-to-linear";
        "(compute dot-product (access-cartesian-product (access ?x 1) (access ?w 1)))"
        => {ApplierImpl("?x".parse().unwrap(), "?w".parse().unwrap())})
}

pub fn lstm_to_flexasr() -> RW {
    use std::path::PathBuf;
    let pattern = {
        let filename = PathBuf::from(format!(
            "{}/models/lstm-for-pldi-pattern.relay",
            env!("CARGO_MANIFEST_DIR")
        ));
        let relay = std::fs::read_to_string(&filename).unwrap();
        let module = tvm::ir::module::IRModule::parse("", relay).unwrap();

        // The pattern in the Glenside language.
        let (orig_pattern, _, _, _) = crate::language::from_relay::from_relay(
            &module,
            false,
            // Has to stay the same as the list above...
            &vec![
                crate::language::RelayOperator::RelaySigmoid,
                crate::language::RelayOperator::RelayTanh,
                crate::language::RelayOperator::RelayLogSoftmax,
                crate::language::RelayOperator::RelayAdd,
            ],
        );

        let pattern_ast = egg::RecExpr::from(
            orig_pattern
                .as_ref()
                .iter()
                .map(|enode| {
                    // We have a single Var in this pattern: it's the "%x"
                    // argument to the pattern. In the pattern compiled to
                    // Glenside, it looks like (access-tensor x).
                    if let crate::language::Language::AccessTensor(id) = enode {
                        if let crate::language::Language::Symbol(v) = &orig_pattern[*id] {
                            if v == "x" {
                                return egg::ENodeOrVar::Var(Var::from_str("?x".into()).unwrap());
                            }
                        }
                    }
                    // Construct the ENode-type node in the pattern AST by first
                    // recursively converting the children of this node.
                    egg::ENodeOrVar::ENode(enode.clone())
                })
                .collect::<Vec<_>>(),
        );

        // Here, we don't use any Vars. This means we won't bind anything with
        // this pattern, BUT the pattern should be much faster according to Max.
        // let pattern_ast = RecExpr::from(
        //     orig_pattern
        //         .as_ref()
        //         .iter()
        //         .map(|enode| ENodeOrVar::ENode(enode.clone()))
        //         .collect::<Vec<_>>(),
        // );

        Pattern::from(pattern_ast)
    };
    struct LSTMApplier;
    impl Applier<Language, MyAnalysis> for LSTMApplier {
        fn apply_one(
            &self,
            egraph: &mut EGraph<Language, MyAnalysis>,
            eclass: Id,
            subst: &Subst,
        ) -> Vec<Id> {
            let out_shape = match &egraph[eclass].data {
                MyAnalysisData::AccessPattern(access) => access.as_vec(),
                _ => panic!("invalid access pattern for LSTM"),
            };
            format!("(accelerator-call flex-lstm ?x hidden0 hidden1 rnn_weight_ih_l0 rnn_weight_hh_l0 rnn_bias_ih_l0 rnn_bias_hh_l0 (shape {}))", out_shape.into_iter().map(|x| x.to_string()).join(" "))
            .parse::<Pattern<_>>().unwrap().apply_one(egraph, eclass, subst)
        }
    }
    rewrite!("flex-lstm"; 
        { pattern } => { LSTMApplier {} })
}

/// Model rewrite
/// If we know how to implement them (a computation) in relay
/// 1. To have two equivalent implementations for a computation
///    the example below is linear layer
///         (reshape (bias_add (dense ?x ?w) ?bias) ?shape)
///     <=> (add (reshape (dense ?x ?w) ?shape) ?bias)
/// 2. Call the Glenside compiler to compile both implementation
///    This will give us two Glenside patterns
/// 3. Rewrite from lhs to rhs

pub fn bubble_reshape_through_linear_generalized() -> Vec<RW> {
    fn can_broadcast(x: Var) -> impl Fn(&mut EG, egg::Id, &egg::Subst) -> bool {
        move |egraph, _, subst| match &egraph[subst[x]].data {
            MyAnalysisData::AccessPattern(access) => {
                access.shape.ndim() + access.item_shape.ndim() == 1
            }
            MyAnalysisData::Shape(shape) => shape.shape.ndim() == 1,
            _ => false,
        }
    }
    struct ApplierImpl(Var);
    impl Applier<Language, MyAnalysis> for ApplierImpl {
        fn apply_one(&self, egraph: &mut EG, eclass: Id, subst: &Subst) -> Vec<Id> {
            let shape_data = match &egraph[subst[self.0]].data {
                MyAnalysisData::Shape(s) => s,
                _ => panic!("not a valid shape data"),
            };
            format!("(access-reshape 
                        (compute elementwise-add 
                            (access-pair 
                                (access (compute dot-product (access-cartesian-product (access ?x 1) (access ?w 1))) 0) 
                                (access (access-broadcast (access-insert-axis ?bias 0) 
                                        (access-shape (shape {} {}) (shape))) 0)))
                        (access-shape ?shape (shape)))", shape_data.shape[1], shape_data.shape[2])
                        .parse::<Pattern<Language>>().unwrap().apply_one(egraph, eclass, subst)
        }
    }
    vec![
        rewrite!("bubble-reshape-through-linear";
            "(compute elementwise-add 
                (access-pair 
                    (access 
                        (access-reshape 
                            (compute dot-product 
                                (access-cartesian-product (access ?x 1) 
                                (access ?w 1))) 
                                (access-shape ?shape (shape)))
                    0) 
                    (access 
                        (access-broadcast 
                            (access-insert-axis (access-insert-axis ?bias 0) 0)
                            (access-shape ?shape (shape))) 0)))"
            =>
            { ApplierImpl("?shape".parse().unwrap()) }),
        rewrite!("bubble-reshape-through-linear-relay";
                    "(relay-operator-call relay-add
                                        (relay-operator-call relay-reshape
                                            (relay-operator-call relay-dense ?x ?w)
                                            ?shape)
                                        ?bias)"
            =>      "(relay-operator-call relay-reshape
                                          (relay-operator-call relay-bias-add
                                            (relay-operator-call relay-dense ?x ?w)
                                            ?bias
                                            1)
                                        ?shape)"
                    if can_broadcast("?bias".parse().unwrap())),
        rewrite!("add-to-bias-add";
                "(relay-operator-call relay-add ?x ?b)"
                => "(relay-operator-call relay-bias-add ?x ?b 1)"
                    if can_broadcast("?b".parse().unwrap())),
    ]
}

pub fn bubble_reshape_through_linear() -> RW {
    // fn same_op_expr(op1 : Var, op2 : Var, expr1 : Var, expr2 : Var) -> impl Fn(&mut EG, egg::Id, &egg::Subst) -> bool {
    //     move |egraph, _, subst| egraph.find(subst[op1]) == egraph.find(subst[op2]) && egraph.find(subst[expr1]) == egraph.find(subst[expr2])
    // }
    rewrite!("bubble-reshape-through-linear";
            "(compute elementwise-add 
                (access-pair 
                    (access 
                        (access-reshape 
                            (compute dot-product 
                                (access-cartesian-product (access ?x 1) 
                                (access ?w 1))) 
                                (access-shape ?shape (shape))) 
                    0) 
                    (access 
                        (access-broadcast 
                            (access-insert-axis (access-insert-axis ?bias 0) 0)
                            (access-shape ?shape (shape))) 0)))"
            =>
            "(access-reshape 
                (compute elementwise-add 
                    (access-pair 
                        (access (compute dot-product (access-cartesian-product (access (access-tensor ?x) 1) (access (access-tensor ?w) 1))) 0) 
                        (access (access-broadcast (access-insert-axis (access-tensor ?bias) 0) 
                                (access-shape (shape 10 16) (shape))) 0)))
            (access-shape ?shape (shape)))")
}

/// 1. the user of the accelerator will give us a pattern written in Relay
///    (bias_add (dense ?x ?w) ?bias)
/// 2. Compile this pattern to a Glenside version pattern
/// 3. Add the following rewrite: from the Glenside version of the pattern to an accelerator call

pub fn linear_layer_accelerator_rewrites() -> RW {
    rewrite!("linear-to-flexnlp-relay";
        "(relay-operator-call relay-bias-add
            (relay-operator-call relay-dense ?x ?w)
            ?bias
            ?axis)"
        =>
        "(accelerator-call flex-linear ?x ?w ?bias (shape 0))")
}

/// Experimental rewrite to convert Glenside matmuls into Relay denses. Pretty
/// straightforward; only experimental b/c adding the night before PLDI
/// deadline.
pub fn glenside_matmul_to_relay_dense() -> RW {
    rewrite!("glenside_matmul_to_relay_dense";
             "(compute dot-product (access-cartesian-product ?x ?w))"
             => "(relay-operator-call relay-dense ?x ?w)"
            if constrain_access("?w".parse().unwrap(),
                                |v| v.as_vec().len() == 2)
            if constrain_access("?x".parse().unwrap(),
                                |v| v.as_vec().len() == 2))
}

/// Experimental rewrite to add a bias add on any dense.
pub fn add_bias_add_to_dense() -> RW {
    struct ApplierImpl;
    impl Applier<Language, MyAnalysis> for ApplierImpl {
        fn apply_one(
            &self,
            egraph: &mut EGraph<Language, MyAnalysis>,
            eclass: Id,
            subst: &Subst,
        ) -> Vec<Id> {
            let shape_str = match &egraph[eclass].data {
                MyAnalysisData::AccessPattern(a) => {
                    // The bias that is added should be a vector. By default in
                    // Relay, it should match the length of axis 1. In our case
                    // it doesn't really matter, because it's 0, but we need to
                    // make the shapes match, so we assume we're matching the
                    // size of dim 1.
                    assert_eq!(a.as_vec().len(), 2);
                    usize::to_string(&a.as_vec()[1])
                }
                MyAnalysisData::Shape(s) => s.shape.slice().iter().map(usize::to_string).join(" "),
                _ => panic!(),
            };

            format!(
                "(relay-operator-call relay-bias-add 
                  (relay-operator-call relay-dense ?x ?w)
                  (relay-operator-call relay-zeros (shape {}))
                  1)",
                shape_str
            )
            .parse::<Pattern<_>>()
            .unwrap()
            .apply_one(egraph, eclass, subst)
        }
    }
    rewrite!("add_bias_add_to_dense";
             "(relay-operator-call relay-dense ?x ?w)"
             => { ApplierImpl })
}

/// Tensorizes a computation to an externally-blocked systolic array.
///
/// `rows` and `cols` define the size of the systolic array to map to. This
/// rewrite will map any matrix multiplication MxN X NxO to this size systolic
/// array as long as:
///  - N is a multiple of `rows`, and
///  - O is a multiple of `columns`.
pub fn systolic_array_with_blocking(rows: usize, cols: usize) -> Rewrite<Language, MyAnalysis> {
    struct ApplierImpl {
        rows: usize,
        cols: usize,
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

            let pattern: Pattern<Language> = format!(
                "(systolic-array-with-blocking {} {}
                          ?access-1
                          (access (access-transpose ?access-2 (list 1 0)) 0)
                         )",
                self.rows, self.cols
            )
            .parse()
            .unwrap();

            pattern.apply_one(egraph, eclass, subst)
        }
    }

    rewrite!(format!("systolic-array-with-blocking-{}-{}", rows, cols);
             "(compute dot-product
               (access-cartesian-product
                ?access-1
                ?access-2
               )
              )
             " =>
             { ApplierImpl{rows, cols, a: "?access-1".parse().unwrap(), b: "?access-2".parse().unwrap(),}}
             // Remember: the accesses look like [M] [N] and [O] [N] for an
             // MxN X NxO multiplication.
             if constrain_access("?access-1".parse().unwrap(),
                                 move |a| a.shape.ndim() <= 1 && a.item_shape.ndim() == 1
                                          && a.item_shape.slice()[0] % rows == 0)
             if constrain_access("?access-2".parse().unwrap(),
                                 move |a| a.shape.ndim() == 1 && a.item_shape.ndim() == 1
                                          && a.shape.slice()[0] % cols == 0))
}

pub fn systolic_array() -> Rewrite<Language, MyAnalysis> {
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

pub enum SliceConcatenateStrategy {
    /// Divides the axis by `divisor`; does not divide anything less than or
    /// equal to `limit`.
    DivideBy {
        divisor: usize,
        limit: usize,
    },
    DivideInto {
        segment_size: usize,
    },
    /// Slice into a dimension just once at the beginning, if there's at least
    /// `segment_size` remaining in the dimension.
    // TODO(@gussmith23) Test this
    SliceOnce {
        segment_size: usize,
    },
}

pub fn slice_concatenate_accesses(
    axis: usize,
    strategy: SliceConcatenateStrategy,
) -> Rewrite<Language, MyAnalysis> {
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

    fn access_dimension_divisible_by(
        axis: usize,
        divisor: usize,
    ) -> impl Fn(&mut EG, egg::Id, &egg::Subst) -> bool {
        move |egraph, id, _subst| match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => a[axis] % divisor == 0,
            _ => panic!(),
        }
    }

    struct DivideByApplier {
        axis: usize,
        divisor: usize,
    }
    impl Applier<Language, MyAnalysis> for DivideByApplier {
        fn apply_one(
            &self,
            egraph: &mut EG,
            id: egg::Id,
            _subst: &egg::Subst,
        ) -> std::vec::Vec<egg::Id> {
            let shape = match &egraph[id].data {
                MyAnalysisData::AccessPattern(a) => a,
                _ => panic!(),
            };
            assert!(self.axis < shape.shape.ndim() + shape.item_shape.ndim());
            let dim_value = if self.axis < shape.shape.ndim() {
                shape.shape[self.axis]
            } else {
                shape.item_shape[self.axis - shape.shape.ndim()]
            };
            assert_eq!(dim_value % self.divisor, 0);

            let axis_id = egraph.add(Language::Usize(self.axis));

            let top_concat_id = (0..self.divisor)
                .map(|segment_index| {
                    let low_bound = segment_index * (dim_value / self.divisor);
                    let high_bound = low_bound + (dim_value / self.divisor);
                    let low_bound_id = egraph.add(Language::Usize(low_bound));
                    let high_bound_id = egraph.add(Language::Usize(high_bound));
                    egraph.add(Language::AccessSlice([
                        id,
                        axis_id,
                        low_bound_id,
                        high_bound_id,
                    ]))
                })
                .collect::<Vec<_>>()
                .iter()
                .fold(None, |prev_concat_id, this_slice_id| match prev_concat_id {
                    None => Some(*this_slice_id),
                    Some(prev_concat_id) => Some(egraph.add(Language::AccessConcatenate([
                        prev_concat_id,
                        *this_slice_id,
                        axis_id,
                    ]))),
                })
                .unwrap();

            vec![top_concat_id]
        }
    }

    // TODO(@gussmith23) This could be combined with the applier above.
    // Their implementations are nearly identical.
    struct DivideIntoApplier {
        axis: usize,
        segment_size: usize,
    }
    impl Applier<Language, MyAnalysis> for DivideIntoApplier {
        fn apply_one(
            &self,
            egraph: &mut EG,
            id: egg::Id,
            _subst: &egg::Subst,
        ) -> std::vec::Vec<egg::Id> {
            let shape = match &egraph[id].data {
                MyAnalysisData::AccessPattern(a) => a,
                _ => panic!(),
            };
            assert!(self.axis < shape.shape.ndim() + shape.item_shape.ndim());
            let dim_value = if self.axis < shape.shape.ndim() {
                shape.shape[self.axis]
            } else {
                shape.item_shape[self.axis - shape.shape.ndim()]
            };
            assert_eq!(dim_value % self.segment_size, 0);

            let axis_id = egraph.add(Language::Usize(self.axis));

            let top_concat_id = (0..(dim_value / self.segment_size))
                .map(|segment_index| {
                    let low_bound = segment_index * self.segment_size;
                    let high_bound = low_bound + self.segment_size;
                    let low_bound_id = egraph.add(Language::Usize(low_bound));
                    let high_bound_id = egraph.add(Language::Usize(high_bound));
                    egraph.add(Language::AccessSlice([
                        id,
                        axis_id,
                        low_bound_id,
                        high_bound_id,
                    ]))
                })
                .collect::<Vec<_>>()
                .iter()
                .fold(None, |prev_concat_id, this_slice_id| match prev_concat_id {
                    None => Some(*this_slice_id),
                    Some(prev_concat_id) => Some(egraph.add(Language::AccessConcatenate([
                        prev_concat_id,
                        *this_slice_id,
                        axis_id,
                    ]))),
                })
                .unwrap();

            vec![top_concat_id]
        }
    }

    struct SliceOnceApplier {
        axis: usize,
        segment_size: usize,
    }
    impl Applier<Language, MyAnalysis> for SliceOnceApplier {
        fn apply_one(
            &self,
            egraph: &mut EG,
            id: egg::Id,
            _subst: &egg::Subst,
        ) -> std::vec::Vec<egg::Id> {
            let access = match &egraph[id].data {
                MyAnalysisData::AccessPattern(a) => a,
                _ => panic!(),
            };

            format!(
                "
             (access-concatenate
              (access-slice ?a {axis} 0 {segment_size})
              (access-slice ?a {axis} {segment_size} {dim_length})
              {axis}
             )
             ",
                segment_size = self.segment_size,
                dim_length = access[self.axis],
                axis = self.axis
            )
            .parse::<Pattern<_>>()
            .unwrap()
            .apply_one(egraph, id, _subst)
        }
    }

    match strategy {
        SliceConcatenateStrategy::DivideBy { divisor, limit } => {
            rewrite!(format!("slice-concatenate-access-axis-{}-divide-by-{}-limit-{}", axis, divisor, limit);
                     "?a" => { DivideByApplier {axis: axis, divisor: divisor} }
                     // TODO(@gussmith) Wouldn't need this if we had "access" op
                     // We could have an access op that takes the access type,
                     // the access config, and then a number of arguments. I.e.
                     // access-1 would take 1, access-2 would take 2, etc. Then
                     // we wouldn't need this check.
                     // TODO(@gussmith) Need to limit what gets sliced
                     // I used to do this with the old split() rewrite. It's a
                     // little trickier now with accesses, because you have to
                     // match only on accesses to tensor literals. Not
                     // impossible, obviously. I'm just being lazy right now.
                     if is_access()
                     if access_has_axis(axis)
                     if access_dimension_divisible_by(axis, divisor)
                     if access_dimension_greater_than(axis, limit))
        }
        SliceConcatenateStrategy::DivideInto { segment_size } => {
            rewrite!(format!("slice-concatenate-access-axis-{}-divide-into-{}", axis, segment_size);
                     "?a" => { DivideIntoApplier {axis: axis, segment_size:segment_size} }
                     if is_access()
                     if access_has_axis(axis)
                     if access_dimension_divisible_by(axis, segment_size))
        }
        SliceConcatenateStrategy::SliceOnce { segment_size } => {
            rewrite!(format!("slice-concatenate-access-axis-{}-slice-once-{}", axis, segment_size);
                     "?a" => { SliceOnceApplier {axis, segment_size} }
                     if is_access()
                     if access_has_axis(axis)
                     // Only slice if there's  at least `segment_size` to slice off.
                     if constrain_access("?a".parse().unwrap(), move |access| access[axis] > segment_size))
        }
    }
}

pub fn slice_concatenate_tensor_accesses(
    axis: usize,
    dimension_greater_than: usize,
) -> Rewrite<Language, MyAnalysis> {
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

pub fn access_slice_access_transpose_composition_commutative() -> Rewrite<Language, MyAnalysis> {
    struct ApplierImpl {
        axis_list: Var,
        slice_axis: Var,
    }
    impl Applier<Language, MyAnalysis> for ApplierImpl {
        fn apply_one(&self, egraph: &mut EG, matched_id: Id, subst: &Subst) -> Vec<Id> {
            let axis_list = match &egraph[subst[self.axis_list]].data {
                MyAnalysisData::List(l) => l,
                _ => panic!("expected list"),
            };
            let old_slice_axis: usize = MyAnalysis::get_usize(subst[self.slice_axis], egraph);
            let new_slice_axis = axis_list[old_slice_axis];
            format!(
                "(access-transpose (access-slice ?tensor {} ?bottom ?top) ?list)",
                new_slice_axis
            )
            .parse::<Pattern<Language>>()
            .unwrap()
            .apply_one(egraph, matched_id, subst)
        }
    }
    rewrite!(
        "access-slice-access-transpose-composition-commutative";
        "(access-slice (access-transpose ?tensor ?list) ?axis ?bottom ?top)" =>
        { ApplierImpl {
            axis_list: "?list".parse().unwrap(),
            slice_axis: "?axis".parse().unwrap(),
        }}
    )
}

pub fn bubble_access_concatenate_through_access_transpose() -> Rewrite<Language, MyAnalysis> {
    struct ApplierImpl {
        concatenate_axis: Var,
        axis_list: Var,
    }
    impl Applier<Language, MyAnalysis> for ApplierImpl {
        fn apply_one(&self, egraph: &mut EG, eclass: Id, subst: &Subst) -> Vec<Id> {
            let original_concatenate_axis: usize =
                MyAnalysis::get_usize(subst[self.concatenate_axis], egraph);
            let axis_list = match &egraph[subst[self.axis_list]].data {
                MyAnalysisData::List(l) => l,
                _ => panic!("Expected list"),
            };
            let new_concatenate_axis = axis_list
                .iter()
                .position(|axis| *axis == original_concatenate_axis)
                .expect(
                    format!(
                        "Did not find axis {} in list of axes {:?}",
                        original_concatenate_axis, axis_list
                    )
                    .as_str(),
                );

            format!(
                "(access-concatenate
                      (access-transpose ?a ?list)
                      (access-transpose ?b ?list) {})",
                new_concatenate_axis
            )
            .parse::<Pattern<_>>()
            .unwrap()
            .apply_one(egraph, eclass, subst)
        }
    }
    rewrite!("bubble-access-concatenate-through-access-transpose";
        "(access-transpose (access-concatenate ?a ?b ?concatenate-axis) ?list)" =>
    {
        ApplierImpl {
            concatenate_axis: "?concatenate-axis".parse().unwrap(),
            axis_list: "?list".parse().unwrap()
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

pub fn collapse_nested_transposes() -> Rewrite<Language, MyAnalysis> {
    struct ApplierImpl {
        inner_list: Var,
        outer_list: Var,
    }
    impl Applier<Language, MyAnalysis> for ApplierImpl {
        fn apply_one(&self, egraph: &mut EG, matched_id: Id, subst: &Subst) -> Vec<Id> {
            let inner_list = match &egraph[subst[self.inner_list]].data {
                MyAnalysisData::List(l) => l,
                _ => panic!(),
            };
            let outer_list = match &egraph[subst[self.outer_list]].data {
                MyAnalysisData::List(l) => l,
                _ => panic!(),
            };

            assert_eq!(inner_list.len(), outer_list.len());

            let new_list = outer_list
                .iter()
                .map(|outer_index| inner_list[*outer_index])
                .collect::<Vec<_>>();

            format!(
                "(access-transpose ?a (list {}))",
                itertools::join(new_list.iter().map(|v| v.to_string()), " ")
            )
            .parse::<Pattern<_>>()
            .unwrap()
            .apply_one(egraph, matched_id, subst)
        }
    }
    rewrite!("collapse-nested-transposes";
             "(access-transpose (access-transpose ?a ?inner-list) ?outer-list)" =>
             { ApplierImpl {
                 inner_list: "?inner-list".parse().unwrap(),
                 outer_list: "?outer-list".parse().unwrap(),
             } }
    )
}

pub fn remove_trivial_transpose() -> Rewrite<Language, MyAnalysis> {
    rewrite!("remove-trivial-transpose";
             "(access-transpose ?a ?list)" => "?a"
             if list_is_0_through_len("?list".parse().unwrap())
    )
}

pub fn collapse_nested_accesses() -> Rewrite<Language, MyAnalysis> {
    rewrite!("collapse-nested-accesses";
             "(access (access ?a ?unused) ?new)" => "(access ?a ?new)")
}

#[derive(Copy, Clone)]
pub enum PadLocation {
    End,
}
impl std::fmt::Display for PadLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                PadLocation::End => "end",
            }
        )
    }
}
pub enum PadSliceStrategy {
    PadToClosestMultipleOf {
        multiple_of: usize,
        pad_location: PadLocation,
        pad_type: PadType,
    },
    PadToMultiplesOf {
        multiples_of: usize,
        limit: usize,
        pad_location: PadLocation,
        pad_type: PadType,
    },
}

pub fn pad_slice_accesses(
    axis: usize,
    strategy: PadSliceStrategy,
) -> Rewrite<Language, MyAnalysis> {
    struct PadToClosestMultipleOfApplier {
        axis: usize,
        multiple_of: usize,
        pad_location: PadLocation,
        pad_type: PadType,
    }
    impl Applier<Language, MyAnalysis> for PadToClosestMultipleOfApplier {
        fn apply_one(
            &self,
            egraph: &mut EG,
            id: egg::Id,
            _subst: &egg::Subst,
        ) -> std::vec::Vec<egg::Id> {
            let access = match &egraph[id].data {
                MyAnalysisData::AccessPattern(a) => a,
                _ => panic!(),
            };

            let dim_val = access[self.axis];
            let pad_to = closest_multiple(self.multiple_of, access[self.axis]);

            let pad_before = match self.pad_location {
                PadLocation::End => 0,
            };
            let pad_after = match self.pad_location {
                PadLocation::End => pad_to - dim_val,
            };
            assert!(
                (pad_before > 0 && pad_before < self.multiple_of)
                    || (pad_after > 0 && pad_after < self.multiple_of)
            );
            let low = match self.pad_location {
                PadLocation::End => 0,
            };
            let high = match self.pad_location {
                PadLocation::End => dim_val,
            };

            format!(
                "(access-slice
                  (access-pad
                   ?a
                   {} {} {} {}
                  )
                  {} {} {}
                 )",
                self.pad_type, self.axis, pad_before, pad_after, self.axis, low, high
            )
            .parse::<Pattern<_>>()
            .unwrap()
            .apply_one(egraph, id, _subst)
        }
    }
    struct PadToMultiplesOfApplier {
        axis: usize,
        limit: usize,
        multiples_of: usize,
        pad_location: PadLocation,
        pad_type: PadType,
    }
    impl Applier<Language, MyAnalysis> for PadToMultiplesOfApplier {
        fn apply_one(
            &self,
            egraph: &mut EG,
            id: egg::Id,
            _subst: &egg::Subst,
        ) -> std::vec::Vec<egg::Id> {
            let access = match &egraph[id].data {
                MyAnalysisData::AccessPattern(a) => a,
                _ => panic!(),
            };

            let dim_val = access[self.axis];
            let mut ids = Vec::new();
            let mut pad_to = closest_greater_multiple(self.multiples_of, access[self.axis]);

            while pad_to <= self.limit {
                let pad_before = match self.pad_location {
                    PadLocation::End => 0,
                };
                let pad_after = match self.pad_location {
                    PadLocation::End => pad_to - dim_val,
                };
                let low = match self.pad_location {
                    PadLocation::End => 0,
                };
                let high = match self.pad_location {
                    PadLocation::End => dim_val,
                };

                let pad_before_id = egraph.add(Language::Usize(pad_before));
                let pad_after_id = egraph.add(Language::Usize(pad_after));
                let low_id = egraph.add(Language::Usize(low));
                let high_id = egraph.add(Language::Usize(high));
                let axis_id = egraph.add(Language::Usize(self.axis));
                let pad_type_id = egraph.add(Language::PadType(self.pad_type));

                let access_pad_id = egraph.add(Language::AccessPad([
                    id,
                    pad_type_id,
                    axis_id,
                    pad_before_id,
                    pad_after_id,
                ]));
                let access_slice_id = egraph.add(Language::AccessSlice([
                    access_pad_id,
                    axis_id,
                    low_id,
                    high_id,
                ]));

                ids.push(access_slice_id);

                pad_to += self.multiples_of;
            }

            ids
        }
    }

    /// Given closest_to and multiple_of, returns n such that n >= closest_to,
    /// n % multiple_of == 0, and n-closest_to < multiple_of.
    ///
    /// for closest_to in 0..100 {
    ///   for multiple_of in 0..100 {
    ///     let closest_multiple = closest_multiple(multiple_of, closest_to);
    ///     assert!(closest_multiple >= closest_to);
    ///     assert_eq!(closest_multiple % multiple_of, 0);
    ///     assert!(closest_multiple - closest_to < multiple_of);
    ///   }
    /// }
    ///
    /// TODO(@gussmith23) Test this
    fn closest_multiple(multiple_of: usize, closest_to: usize) -> usize {
        closest_to + ((multiple_of - (closest_to % multiple_of)) % multiple_of)
    }
    fn closest_greater_multiple(multiple_of: usize, closest_to: usize) -> usize {
        let multiple = closest_multiple(multiple_of, closest_to);
        if multiple == closest_to {
            multiple + multiple_of
        } else {
            multiple
        }
    }

    match strategy {
        PadSliceStrategy::PadToClosestMultipleOf {
            multiple_of,
            pad_location,
            pad_type,
        } => rewrite!(
            format!("pad-slice-accesses-axis-{}-pad-to-nearest-multiple-of-{}-location-{}",
                    axis, multiple_of, pad_location);
            "?a" =>
            {
                PadToClosestMultipleOfApplier{
                    axis,
                    multiple_of,
                    pad_location,
                    pad_type
                }
            }
            if is_access()
            if access_has_axis(axis)
            // Don't run if it's already divisible.
            if constrain_access("?a".parse().unwrap(), move |access| access[axis] % multiple_of != 0)
        ),
        PadSliceStrategy::PadToMultiplesOf {
            multiples_of,
            limit,
            pad_location,
            pad_type,
        } => rewrite!(
            format!("pad-slice-accesses-axis-{}-pad-to-nearest-multiple-of-{}-limit-{}-location-{}",
                            axis, multiples_of, limit, pad_location);
                            "?a" =>
            { PadToMultiplesOfApplier{
                limit,
                axis,
                multiples_of,
                pad_location,
                pad_type

            } }
                            if  is_access()
                            if access_has_axis(axis)
                            if constrain_access("?a".parse().unwrap(), move |a| {
                                    closest_greater_multiple(a[axis], multiples_of) <= limit
                            })

        ),
    }
}

pub fn bubble_access_slice_through_access_pad_inequal_axes() -> Rewrite<Language, MyAnalysis> {
    rewrite! {"bubble-access-slice-through-access-pad-inequal-axes";
        "(access-pad
               (access-slice ?a ?slice-axis ?low ?high)
               ?padding ?pad-axis ?pad-before ?pad-after
              )" =>
        "(access-slice
               (access-pad ?a ?padding ?pad-axis ?pad-before ?pad-after)
               ?slice-axis ?low ?high
              )"
    if constrain_vars(vec!["?slice-axis".parse().unwrap(), "?pad-axis".parse().unwrap()], |data| {
        assert_eq!(data.len(), 2);
        let slice_axis = match &data[0] {
            MyAnalysisData::Usize(l) => *l,
            _ =>panic!(),
        };
        let pad_axis = match &data[1] {
            MyAnalysisData::Usize(l) => *l,
            _ =>panic!(),
        };
        slice_axis != pad_axis
    })}
}

pub fn bubble_access_slice_through_access_cartesian_product_not_item_axis_left(
) -> Rewrite<Language, MyAnalysis> {
    rewrite!("bubble-access-slice-through-access-cartesian-product-not-item-axis-left";
             "(access-cartesian-product (access-slice ?a ?axis ?low ?high) ?right)" =>
             "(access-slice
               (access-cartesian-product ?a ?right)
               ?axis
               ?low
               ?high
              )"
             if not_item_axis("?axis".parse().unwrap(), "?a".parse().unwrap()))
}

pub fn bubble_access_slice_through_access_cartesian_product_not_item_axis_right(
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
                "(access-slice
                  (access-cartesian-product ?left ?a)
                  {}
                  ?low
                  ?high
                 )",
                axis + left.shape.ndim()
            )
            .parse::<Pattern<_>>()
            .unwrap()
            .apply_one(egraph, eclass, subst)
        }
    }
    rewrite!("bubble-access-slice-through-access-cartesian-product-not-item-axis-right";
                 "(access-cartesian-product ?left (access-slice ?a ?axis ?low ?high))" =>
             { ApplierImpl {
                 axis: "?axis".parse().unwrap(),
                 left: "?left".parse().unwrap(),
             } }
                 if not_item_axis("?axis".parse().unwrap(), "?a".parse().unwrap()))
}

pub fn bubble_access_slice_through_access_cartesian_product_same_item_axis(
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
                "(access-slice
                    (access-cartesian-product ?a0 ?a1)
                    {} ?low ?high)",
                new_axis
            )
            .parse::<Pattern<_>>()
            .unwrap()
            .apply_one(egraph, matched_id, subst)
        }
    }
    rewrite! {"bubble-access-slice-through-access-cartesian-product-same-item-axis";
    "(access-cartesian-product
               (access-slice ?a0 ?axis0 ?low ?high)
               (access-slice ?a1 ?axis1 ?low ?high)
              )" =>
    { ApplierImpl {
        axis0: "?axis0".parse().unwrap(),
        access1: "?a1".parse().unwrap()
    }}
              if same_item_axis("?axis0".parse().unwrap(), "?a0".parse().unwrap(), "?axis1".parse().unwrap(), "?a1".parse().unwrap())
    if constrain_vars(vec!["?a0".parse().unwrap(), "?a1".parse().unwrap(), "?axis0".parse().unwrap(), "?axis1".parse().unwrap()], |data| {
        let a0 = match &data[0] {
            MyAnalysisData::AccessPattern(a) =>a,
            _ => panic!(),
        };
        let a1 = match &data[1] {
            MyAnalysisData::AccessPattern(a) =>a,
            _ => panic!(),
        };
        let axis0 = match &data[2] {
            MyAnalysisData::Usize(l) => *l,
            _ => panic!(),
        };
        let axis1 = match &data[3] {
            MyAnalysisData::Usize(l) => *l,
            _ => panic!(),
        };

        // The unsliced dimensions must be cartesian-product-compatible (i.e. equal)
        a0[axis0] == a1[axis1]
    })}
}

/// If we're not slicing in an axis that's being computed over, then removing
/// the slice has no potential to effect the computation. It will just result in
/// a larger computation, with some of the data being sliced away.
pub fn bubble_access_slice_through_compute_dot_product_not_item_axis(
) -> Rewrite<Language, MyAnalysis> {
    rewrite!("bubble-access-slice-through-compute-dot-product-not-item-axis";
             "(compute dot-product (access-slice ?a ?axis ?low ?high))" =>
             "(access-slice
               (compute dot-product ?a)
               ?axis ?low ?high
              )"
             if not_item_axis("?axis".parse().unwrap(), "?a".parse().unwrap()))
}

/// If we're slicing in an item axis that isn't the tuple axis (i.e. the first
/// item axis), then computing before slicing has the potential to affect the
/// computation, as it is adding data. So we need to be able to prove that this
/// region will not affect the computation.
pub fn bubble_access_slice_through_compute_dot_product_item_axis_not_tuple_axis(
) -> Rewrite<Language, MyAnalysis> {
    rewrite!("bubble-access-slice-through-compute-dot-product-item-axis";
             "(compute dot-product (access-slice ?a ?axis ?low ?high))" =>
             "(compute dot-product ?a)"
             if item_axis("?axis".parse().unwrap(), "?a".parse().unwrap())
             // This checks that everything outside of the sliced region in
             // this axis is zero.
             if constrain_vars(vec!["?a".parse().unwrap(), "?axis".parse().unwrap(), "?low".parse().unwrap(), "?high".parse().unwrap()],
                               |data| {
                                   let a = match &data[0] {
                                       MyAnalysisData::AccessPattern(a) =>a,
                                       _ => panic!(),
                                   };
                                   let axis = match &data[1] {
                                       MyAnalysisData::Usize(l) => *l,
                                       _ => panic!(),
                                   };
                                   let low = match &data[2] {
                                       MyAnalysisData::Usize(l) => *l,
                                       _ => panic!(),
                                   };
                                   let high = match &data[3] {
                                       MyAnalysisData::Usize(l) => *l,
                                       _ => panic!(),
                                   };

                                   a.zero_regions.get(&axis).map_or(false,
                                                                    |range_set|{
                                                                        range_set.covered((0, low))
                                                                            &&
                                                                            range_set.covered((high, a[axis]))
                                                                    })
                               })

    )
}

pub fn bubble_access_transpose_through_access_pad() -> Rewrite<Language, MyAnalysis> {
    struct ApplierImpl {
        list: Var,
        pad_axis: Var,
    }
    impl Applier<Language, MyAnalysis> for ApplierImpl {
        fn apply_one(&self, egraph: &mut EG, matched_id: Id, subst: &Subst) -> Vec<Id> {
            let list = match &egraph[subst[self.list]].data {
                MyAnalysisData::List(l) => l.clone(),
                _ => panic!(),
            };
            let pad_axis = MyAnalysis::get_usize(subst[self.pad_axis], egraph);

            format!(
                "(access-transpose (access-pad ?a ?pad-type {} ?pad-before ?pad-after) ?list)",
                list[pad_axis]
            )
            .parse::<Pattern<_>>()
            .unwrap()
            .apply_one(egraph, matched_id, subst)
        }
    }

    rewrite!("bubble-access-transpose-through-access-pad";
    "(access-pad
      (access-transpose ?a ?list)
      ?pad-type ?pad-axis ?pad-before ?pad-after
     )" => {
         ApplierImpl {
             list: "?list".parse().unwrap(),
             pad_axis: "?pad-axis".parse().unwrap(),
         }
     })
}

pub fn systolic_array_conv2d_nchw_oihw_with_blocking(
    rows: usize,
    cols: usize,
) -> Rewrite<Language, MyAnalysis> {
    struct ApplierImpl {
        rows: usize,
        cols: usize,
    }
    impl Applier<Language, MyAnalysis> for ApplierImpl {
        fn apply_one(&self, egraph: &mut EG, matched_id: Id, subst: &Subst) -> Vec<Id> {
            format!("(systolic-array-conv2d-nchw-oihw-with-blocking {rows} {cols} ?weights ?data ?kh ?kw ?stride-h ?stride-w)",
                        rows = self.rows,
                        cols = self.cols)
                .parse::<Pattern<_>>()
                .unwrap()
                .apply_one(egraph, matched_id, subst)
        }
    }
    rewrite!(format!("systolic-array-conv2d-nchw-oihw-with-blocking-{}-{}", rows, cols);
    "
             (access-transpose
              (compute dot-product
               (access-cartesian-product
                (access ?weights 1)
                (access
                 (access-squeeze
                   (access-windows
                    (access ?data 1)
                    (shape ?c ?kh ?kw)
                    (shape 1 ?stride-h ?stride-w)
                   )
                  1
                 )
                 3
                )
               )
              )
              (list 1 0 2 3)
             )
" => {
                  ApplierImpl {rows, cols}
              }
             if constrain_access("?weights".parse().unwrap(),
                                 move |a| a.shape.ndim() + a.item_shape.ndim() == 4
                                 // Input channels divisible by rows, output
                                 // channels divisible by columns.
                                 && a[1] % rows == 0 && a[0] % cols == 0)
             if constrain_access("?data".parse().unwrap(),
                                 move |a| a.shape.ndim() + a.item_shape.ndim() == 4
                                 // Input channels divisible by rows
                                 && a[1] % rows == 0))
}

pub fn systolic_array_conv2d_nhwc_hwio_with_blocking(
    rows: usize,
    cols: usize,
) -> Rewrite<Language, MyAnalysis> {
    struct ApplierImpl {
        rows: usize,
        cols: usize,
    }
    impl Applier<Language, MyAnalysis> for ApplierImpl {
        fn apply_one(&self, egraph: &mut EG, matched_id: Id, subst: &Subst) -> Vec<Id> {
            format!("(systolic-array-conv2d-nhwc-hwio-with-blocking {rows} {cols} ?weights ?data ?kh ?kw ?stride-h ?stride-w)",
                        rows = self.rows,
                        cols = self.cols)
                .parse::<Pattern<_>>()
                .unwrap()
                .apply_one(egraph, matched_id, subst)
        }
    }
    rewrite!(format!("systolic-array-conv2d-nhwc-hwio-with-blocking-{}-{}", rows, cols);
    "       (access-transpose
             (systolic-array-conv2d-nchw-oihw-with-blocking
              ?rows ?cols
              (access-transpose ?weights (list 3 2 0 1))
              (access-transpose ?data (list 0 3 1 2))
              ?kh ?kw ?stride-h ?stride-w
             )
             (list 0 2 3 1)
            )" => {
                  ApplierImpl {rows, cols}
              }
             if move |egraph: &mut EGraph<Language, MyAnalysis>, _, subst: &Subst| match &egraph[subst["?rows".parse().unwrap()]].data {
                 MyAnalysisData::Usize(l) => *l == rows,
                 _ => panic!(),
             }
             if move |egraph: &mut EGraph<Language, MyAnalysis>, _, subst: &Subst| match &egraph[subst["?cols".parse().unwrap()]].data {
                 MyAnalysisData::Usize(l) => *l == cols,
                 _ => panic!(),
             }
             if constrain_access("?weights".parse().unwrap(),
                                 move |a| a.shape.ndim() + a.item_shape.ndim() == 4
                                 // Input channels divisible by rows, output
                                 // channels divisible by columns.
                                 && a[2] % rows == 0 && a[3] % cols == 0)
             if constrain_access("?data".parse().unwrap(),
                                 move |a| a.shape.ndim() + a.item_shape.ndim() == 4
                                 // Input channels divisible by rows
                                 && a[3] % rows == 0))
}

pub fn systolic_array_conv2d_im2col_nchw_oihw_with_blocking(
    rows: usize,
    cols: usize,
) -> Rewrite<Language, MyAnalysis> {
    struct ApplierImpl {
        rows: usize,
        cols: usize,
    }
    impl Applier<Language, MyAnalysis> for ApplierImpl {
        fn apply_one(&self, egraph: &mut EG, matched_id: Id, subst: &Subst) -> Vec<Id> {
            format!("(systolic-array-conv2d-im2col-nchw-oihw-with-blocking {rows} {cols} ?weights ?data ?kh ?kw ?stride-h ?stride-w)",
                        rows = self.rows,
                        cols = self.cols)
                .parse::<Pattern<_>>()
                .unwrap()
                .apply_one(egraph, matched_id, subst)
        }
    }
    rewrite!(format!("systolic-array-conv2d-im2col-nchw-oihw-with-blocking-{}-{}", rows, cols);
    "
            (access-transpose
             (access-reshape
              (compute dot-product
               (access-cartesian-product
                (access-flatten (access ?weights 1))
                (access-flatten
                 (access
                  (access-squeeze
                    (access-windows
                     (access ?data 1)
                     (shape ?c ?kh ?kw)
                     (shape 1 ?stride-h ?stride-w)
                    )
                   1
                  )
                  3
                 )
                )
               )
              )
              ?reshape-shape
             )
             (list 1 0 2 3)
            )" => {
                  ApplierImpl {rows, cols}
              }
             // TODO(@gussmith23) Any constraints on these?
             // There may not be many constraints here, because Scott's
             // implementing the tail padding himself.
             )
}

pub fn systolic_array_conv2d_im2col_nhwc_hwio_with_blocking(
    rows: usize,
    cols: usize,
) -> Rewrite<Language, MyAnalysis> {
    struct ApplierImpl {
        rows: usize,
        cols: usize,
    }
    impl Applier<Language, MyAnalysis> for ApplierImpl {
        fn apply_one(&self, egraph: &mut EG, matched_id: Id, subst: &Subst) -> Vec<Id> {
            format!("(systolic-array-conv2d-im2col-nhwc-hwio-with-blocking {rows} {cols} ?weights ?data ?kh ?kw ?stride-h ?stride-w)",
                        rows = self.rows,
                        cols = self.cols)
                .parse::<Pattern<_>>()
                .unwrap()
                .apply_one(egraph, matched_id, subst)
        }
    }
    rewrite!(format!("systolic-array-conv2d-im2col-nhwc-hwio-with-blocking-{}-{}", rows, cols);
    "       (access-transpose
             (systolic-array-conv2d-im2col-nchw-oihw-with-blocking
              ?rows ?cols
              (access-transpose ?weights (list 3 2 0 1))
              (access-transpose ?data (list 0 3 1 2))
              ?kh ?kw ?stride-h ?stride-w
             )
             (list 0 2 3 1)
            )" => {
                  ApplierImpl {rows, cols}
              }
             // TODO(@gussmith23) Any constraints on these?
             // There may not be many constraints here, because Scott's
             // implementing the tail padding himself.
    )
}

/// TODO(@gussmith23) This is a hack
/// This is pretty hyper-specific to how we currently implement conv2d when reading from Relay. That is, to implement conv2d, we transpose to NCHW
pub fn systolic_array_conv2d_im2col_fc_with_blocking(
    _rows: usize,
    _cols: usize,
) -> Rewrite<Language, MyAnalysis> {
    todo!()
}

/// Rewrite mapping maxpools to the FlexASR accelerator.
///
/// A single invocation of FlexASR's maxpool operator does the following:
/// Given a number of *timesteps* t and *hidden states* h, the input data looks
/// like:
/// ```text
/// [ [d_0_0, ..., d_0_h], ..., [d_t_0, ..., d_t_h] ]
/// ```
/// The maxpool computes the max between `d_0_i` and `d_1_i`, between `d_2_i`
/// and `d_3_i`, etc., for all `i`. The result is an array with the same number
/// of hidden states but half the number of timesteps. Because the number of
/// timesteps is halved, we require the timesteps to be divisible by 2.
///
/// Memory is laid out in the manner described above. Within FlexASR, Each
/// timestep is 128 bits: 16 hidden states, where each state is 8 bits. However,
/// FlexASR supports more than 16 hidden states. It also supports timesteps not
/// divisible by 2, though I don't think we're going to worry about supporting
/// that on the Glenside side for now, because all of our examples should be
/// divisible by 2.
///
/// Number of hidden states should be a multiple of 16.
///
/// Note how we transform the access pattern that is fed into `flexasr-maxpool`.
/// First, we transpose the access pattern, to indicate the "timestep-major"
/// (like row-major) layout in memory. Then, we re-access at dimension 0, to
/// indicate that the input data should be viewed as an opaque input tensor.
/// This re-access is not necessary, and moreso in place so as not to abuse
/// access pattern semantics.
pub fn flexasr_maxpool() -> Rewrite<Language, MyAnalysis> {
    rewrite!("flexasr-maxpool";
    "(compute reduce-max
      (access-windows ?a (shape 2) (shape 2)))" =>
    "(access
      (access-transpose
       (accelerator-call flex-maxpool
        (access (access-transpose ?a (list 1 0)) 0) (shape 0))
       (list 1 0))
      1)"
    if constrain_access("?a".parse().unwrap(), move |a| {
       // Hidden states divisible by 16.
       a.shape.ndim() == 1 && a.shape[0] % 16 == 0
       // This check is a bit redundant (access-windows providing a
       // length 1 stride/window shape means the compute dimensions
       // here must be len 1) but we include it just to be clear!
       && a.item_shape.ndim() == 1
    }))
}

/// Breaks a large reduce-max into smaller reduce-maxes which are then reduced
/// by the original reduce-max.
pub fn reassociate_max(window_len: usize, strides: usize) -> RW {
    // TODO(@gussmith23) explain why...
    assert!(strides <= window_len, "Strides > window_len will not work.");

    struct ApplierImpl {
        a: Var,
        window_len: usize,
        strides: usize,
    }
    impl Applier<Language, MyAnalysis> for ApplierImpl {
        fn apply_one(&self, egraph: &mut EG, matched_id: Id, subst: &Subst) -> Vec<Id> {
            // The dimension to re-access at, after we compute the new reduce-max.
            let reaccess_dim = match &egraph[subst[self.a]].data {
                MyAnalysisData::AccessPattern(a) => a.shape.ndim(),
                _ => panic!(),
            };
            format!(
                "(compute reduce-max 
                      (access
                       (compute reduce-max
                        (access-windows 
                         ?a
                         (shape {window_len})
                         (shape {strides})))
                       {reaccess_dim}))",
                window_len = self.window_len,
                strides = self.strides,
                reaccess_dim = reaccess_dim
            )
            .parse::<Pattern<_>>()
            .unwrap()
            .apply_one(egraph, matched_id, subst)
        }
    }

    rewrite!("reassociate-max";
     "(compute reduce-max ?a)" =>
     { ApplierImpl {
         a: "?a".parse().unwrap(),
         window_len,
         strides
        } }
     if constrain_access("?a".parse().unwrap(),
                         move |a| a.item_shape.ndim() == 1
                                    && a.item_shape[0] != 0
                                    && a.item_shape[0] % window_len == 0)
    )
}

/// Moves a reshape through a compute reduce-max. We do this by simply throwing
/// away the shape associated with the compute dimensions.
pub fn bubble_access_reshape_through_compute_reduce_max() -> RW {
    struct ApplierImpl {
        shape: Var,
    }
    impl Applier<Language, MyAnalysis> for ApplierImpl {
        fn apply_one(&self, egraph: &mut EG, matched_id: Id, subst: &Subst) -> Vec<Id> {
            let shape = match &egraph[subst[self.shape]].data {
                MyAnalysisData::AccessPattern(a) => a.shape.slice(),
                _ => panic!(),
            };
            format!(
                "(access-reshape
                  (compute reduce-max ?a)
                  (access-shape (shape {shape}) (shape)))",
                shape = shape.iter().map(usize::to_string).join(" ")
            )
            .parse::<Pattern<_>>()
            .unwrap()
            .apply_one(egraph, matched_id, subst)
        }
    }
    rewrite!("bubble-access-reshape-through-compute-reduce-max";
     "(compute reduce-max
       (access-reshape ?a ?shape))" =>
     { ApplierImpl {shape: "?shape".parse().unwrap()}})
}

pub fn simplify_multiple_accesses() -> RW {
    rewrite!("simplify-multiple-accesses";
     "(access (access ?a ?d0) ?d1)" => "(access ?a ?d1)")
}

pub fn simplify_multiple_transposes() -> RW {
    rewrite!("simplify-multiple-transposes";
    "(access-transpose (access-transpose ?a ?list1) ?list2)" =>
    "?a"
    if move |egraph: &mut EG, _, subst: &Subst| {
        let (l1, l2) = match (&egraph[subst["?list1".parse().unwrap()]].data,
                                &egraph[subst["?list2".parse().unwrap()]].data) {
            (MyAnalysisData::List(l1), MyAnalysisData::List(l2)) => (l1.clone(), l2.clone()),
            _ => panic!(),
        };

        assert_eq!(l1.len(), l2.len());

        // If we apply l2 to l1 and get back 0, 1, 2, ... then these transposes cancel!
        (0..l1.len()).collect::<Vec<_>>() == l2.iter().map(|i| l1[*i]).collect::<Vec<_>>()
    })
}

/// Both directions of this rewrite are trivial.
pub fn bubble_access_through_access_transpose() -> RW {
    rewrite!("bubble-access-through-access-transpose";
             "(access-transpose (access ?a ?dim) ?list)" => "(access (access-transpose ?a ?list) ?dim)")
}

/// Simplify away a reduce-max of a single element by rewriting it to a simple
/// reshape. I.e. a reduce-max over `((...), (1, ..., 1))` gets rewritten to a
/// reshape which reshapes to `((...), ())`. My previous version of this rewrite
/// was seeming to cause bugs; if things go wrong, disable this rewrite first!
pub fn simplify_reduce_max() -> RW {
    struct ApplierImpl(Var);
    impl Applier<Language, MyAnalysis> for ApplierImpl {
        fn apply_one(&self, egraph: &mut EG, matched_id: Id, subst: &Subst) -> Vec<Id> {
            let shape = match &egraph[subst[self.0]].data {
                MyAnalysisData::AccessPattern(a) => a.shape.slice(),
                _ => panic!(),
            };
            format!(
                "(access-reshape
                  ?a
                  (access-shape (shape {shape}) (shape)))",
                shape = shape.iter().map(usize::to_string).join(" ")
            )
            .parse::<Pattern<_>>()
            .unwrap()
            .apply_one(egraph, matched_id, subst)
        }
    }
    rewrite!("simplify-reduce-max";
     "(compute reduce-max ?a)" =>
     {ApplierImpl("?a".parse().unwrap())}
    if constrain_access("?a".parse().unwrap(), |access| {
        // Lets all of the following pass:
        // - `((...), ())`
        // - `((...), (1))`
        // - `((...), (1, 1, ..., 1))`
        access.item_shape.slice().iter().product::<usize>() == 1
    }))
}

pub fn simplify_multiple_access_reshapes() -> RW {
    rewrite!("simplify-multiple-access-reshapes";
     "(access-reshape (access-reshape ?a ?s0) ?s1)" => "(access-reshape ?a ?s1)")
}

pub fn conv2d_relay_to_glenside() -> RW {
    struct Impl {
        data: Var,
        weights: Var,
        strides: Var,
        padding: Var,
        group: Var,
        channel: Var,
        kernel_size: Var,
        activation_layout: Var,
        kernel_layout: Var,
    }
    impl Applier<Language, MyAnalysis> for Impl {
        fn apply_one(
            &self,
            egraph: &mut EGraph<Language, MyAnalysis>,
            _eclass: Id,
            subst: &Subst,
        ) -> Vec<Id> {
            let (
                data,
                weight,
                strides,
                padding,
                group,
                _channels,
                _kernel_size,
                activation_layout,
                kernel_layout,
            ) = match vec![
                &self.data,
                &self.weights,
                &self.strides,
                &self.padding,
                &self.group,
                &self.channel,
                &self.kernel_size,
                &self.activation_layout,
                &self.kernel_layout,
            ]
            .drain(..)
            .map(|v| &egraph[subst[*v]].data)
            .collect::<Vec<_>>()[..]
            {
                [MyAnalysisData::AccessPattern(data), MyAnalysisData::AccessPattern(weight), MyAnalysisData::Shape(strides), MyAnalysisData::Shape(padding), MyAnalysisData::Usize(group), MyAnalysisData::Usize(channels), MyAnalysisData::Shape(kernel_size), MyAnalysisData::RelayActivationLayout(act_layout), MyAnalysisData::RelayKernelLayout(_ker_layout)] => {
                    (
                        data,
                        weight,
                        strides,
                        padding,
                        *group,
                        channels,
                        kernel_size,
                        act_layout,
                        _ker_layout,
                    )
                }
                _ => todo!(),
            };

            assert_eq!(group, 1);
            assert_eq!(strides.shape.ndim(), 3);

            let mut expr = RecExpr::default();
            let data_id = expr.add(Language::Symbol("data_PLACEHOLDER".to_string()));
            let weights_id = expr.add(Language::Symbol("weights_PLACEHOLDER".to_string()));
            from_relay::conv2d(
                &mut expr,
                data_id,
                data.as_vec().as_slice(),
                weights_id,
                weight.as_vec().as_slice(),
                &strides.shape.slice()[1..3],
                padding.shape.slice(),
                &[1, 1],
                group,
                match activation_layout {
                    crate::language::RelayActivationLayout::NCHW => "NCHW",
                    crate::language::RelayActivationLayout::NHWC => "NHWC",
                },
                match kernel_layout {
                    crate::language::RelayKernelLayout::OIHW => "OIHW",
                    crate::language::RelayKernelLayout::HWIO => "HWIO",
                },
                "",
            );

            let pattern_ast = PatternAst::from(
                expr.as_ref()
                    .iter()
                    .map(|n| match n {
                        Language::Symbol(s) if s == "data_PLACEHOLDER" => {
                            ENodeOrVar::Var(self.data)
                        }
                        Language::Symbol(s) if s == "weights_PLACEHOLDER" => {
                            ENodeOrVar::Var(self.weights)
                        }
                        _ => ENodeOrVar::ENode(n.clone()),
                    })
                    .collect::<Vec<_>>(),
            );

            vec![egraph.add_instantiation(&pattern_ast, subst)]
        }
    }
    rewrite!("conv2d-relay-to-glenside";
    "(relay-operator-call relay-conv2d
       ?data
       ?weights
       ?strides
       ?padding
       ?group
       ?channel
       ?kernel-size
       ?activation-layout
       ?kernel-layout)" => {
          Impl {
            data: "?data".parse().unwrap(),
            weights: "?weights".parse().unwrap(),
            strides: "?strides".parse().unwrap(),
            padding: "?padding".parse().unwrap(),
            group: "?group".parse().unwrap(),
            channel: "?channel".parse().unwrap(),
            kernel_size: "?kernel-size".parse().unwrap(),
            activation_layout: "?activation-layout".parse().unwrap(),
            kernel_layout: "?kernel-layout".parse().unwrap(),
          }
      })
}

pub fn conv1d_relay_to_glenside() -> RW {
    struct Impl {
        data: Var,
        weights: Var,
        strides: Var,
        padding: Var,
    }
    impl Applier<Language, MyAnalysis> for Impl {
        fn apply_one(
            &self,
            egraph: &mut EGraph<Language, MyAnalysis>,
            _eclass: Id,
            subst: &Subst,
        ) -> Vec<Id> {
            let (data, weights, strides, padding) = match vec![
                self.data,
                self.weights,
                self.strides,
                self.padding,
            ]
            .drain(..)
            .map(|v| &egraph[subst[v]].data)
            .collect::<Vec<_>>()[..]
            {
                [MyAnalysisData::AccessPattern(data), MyAnalysisData::AccessPattern(weights), MyAnalysisData::Shape(ShapeData { shape: strides, .. }), MyAnalysisData::Shape(ShapeData { shape: padding, .. })] => {
                    (data, weights, strides, padding)
                }
                _ => todo!(),
            };

            let mut expr = RecExpr::default();
            let data_id = expr.add(Language::Symbol("data_PLACEHOLDER".to_string()));
            let weights_id = expr.add(Language::Symbol("weights_PLACEHOLDER".to_string()));
            from_relay::conv1d(
                &mut expr,
                data_id,
                data.as_vec().as_slice(),
                weights_id,
                weights.as_vec().as_slice(),
                &strides.slice(),
                padding.slice(),
                &[1],
                1,
                "NCW",
                "OIW",
                "",
            );

            let pattern_ast = PatternAst::from(
                expr.as_ref()
                    .iter()
                    .map(|n| match n {
                        Language::Symbol(s) if s == "data_PLACEHOLDER" => {
                            ENodeOrVar::Var(self.data)
                        }
                        Language::Symbol(s) if s == "weights_PLACEHOLDER" => {
                            ENodeOrVar::Var(self.weights)
                        }
                        _ => ENodeOrVar::ENode(n.clone()),
                    })
                    .collect::<Vec<_>>(),
            );

            vec![egraph.add_instantiation(&pattern_ast, subst)]
        }
    }
    let i = Impl {
        data: "?data".parse().unwrap(),
        weights: "?weights".parse().unwrap(),
        strides: "?strides".parse().unwrap(),
        padding: "?padding".parse().unwrap(),
    };
    rewrite!("conv1d-relay-to-glenside";
        { format!("(relay-operator-call relay-conv1d {} {} {} {})",
                    i.data, i.weights, i.strides, i.padding)
            .parse::<Pattern<_>>().unwrap() } => { i })
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::language::interpreter::interpret;
    use crate::language::{Language, MyAnalysis};
    use approx::AbsDiffEq;
    use egg::{EGraph, Pattern, RecExpr, Runner, Searcher};
    use ndarray::IxDyn;
    use ndarray_npy::{read_npy, write_npy};
    use ndarray_rand::{rand_distr::Uniform, RandomExt};
    use rand::{rngs::SmallRng, Rng, SeedableRng};
    use std::collections::HashMap;
    use std::io::Write;
    use std::process::Command;
    use std::str::FromStr;

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
          (access (access-tensor t-3-32-32) 0)
          (shape 3 3 3)
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
               (access (access-tensor t-3-32-32) 0)
               (shape 3 3 3)
               (shape 1 1 2)
              )
             )
             (access-shape
              (shape 1 30 15)
              (shape 3 3 3)
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
            (access (access-tensor t-3-32-32) 0)
            (shape 3 3 3)
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
                 (access (access-tensor t-3-32-32) 0)
                 (shape 3 3 3)
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
    fn conv1d_im2col_systolic_array() {
        let program = "
        (access-transpose
            (compute dot-product
              (access-cartesian-product
               (access (access-tensor weights) 1)
               (access-squeeze
                (access-windows
                 (access
                  (access-pad
                   (access-tensor data)
                   zero-padding
                   2 3 4
                  )
                  1
                 )
                 (shape 3 3)
                 (shape 1 2)
                )
                1
               )
              )
            )
            (list 1 0 2)
           )
        "
        .parse()
        .unwrap();

        let mut map = HashMap::new();
        map.insert("data".to_string(), vec![1, 3, 32]);
        map.insert("weights".to_string(), vec![8, 3, 3]);
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
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
        (access-transpose
         (access-reshape
          (systolic-array ?rows ?cols
            ?a
            ?b
          )
          ?shape
         )
         ?transpose-list
        )
            "
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn conv2d_im2col_systolic_array() {
        let program = "
         (compute dot-product
          (access-cartesian-product
           (access (access-tensor t-8-3-3-3) 1)
           (access-squeeze
            (access-windows
             (access (access-tensor t-3-32-32) 0)
             (shape 3 3 3)
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
                   (access (access-tensor t-3-32-32) 0)
                   (shape 3 3 3)
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
    fn slice_concatenate_accesses_divide_into_0() {
        test_logger::ensure_env_logger_initialized();

        let program = "(access (access-tensor t-32-32) 1)".parse().unwrap();

        let rws = vec![
            super::slice_concatenate_accesses(
                0,
                SliceConcatenateStrategy::DivideInto { segment_size: 8 },
            ),
            super::slice_concatenate_accesses(
                1,
                SliceConcatenateStrategy::DivideInto { segment_size: 8 },
            ),
            super::collapse_nested_access_slices(),
        ];

        let mut egraph = EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        egraph.add_expr(&program);
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        for dim0 in &[0, 8, 16, 24] {
            for dim1 in &[0, 8, 16, 24] {
                assert_eq!(
                    format!("(access-slice (access-slice (access (access-tensor t-32-32) 1) 1 {} {}) 0 {} {})", dim1, dim1+8, dim0, dim0+8)
                        .parse::<Pattern<_>>()
                        .unwrap()
                        .search(&runner.egraph)
                        .len(),
                    1
                );
            }
        }
    }

    #[test]
    fn slice_concatenate_accesses() {
        test_logger::ensure_env_logger_initialized();

        let program = "(access (access-tensor t-32-32) 1)".parse().unwrap();

        let rws = vec![
            super::slice_concatenate_accesses(
                0,
                SliceConcatenateStrategy::DivideBy {
                    divisor: 2,
                    limit: 16,
                },
            ),
            super::slice_concatenate_accesses(
                1,
                SliceConcatenateStrategy::DivideBy {
                    divisor: 2,
                    limit: 16,
                },
            ),
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
    fn access_slice_access_transpose_composition_commutative_0() {
        let program = "
(access-slice
 (access-transpose
  (access (access-tensor t-3-32-32) 1)
  (list 2 0 1)
 )
 0 16 32
)
        "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        let rws = vec![super::access_slice_access_transpose_composition_commutative()];
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        let matches = "
(access-transpose
 (access-slice
  (access (access-tensor t-3-32-32) 1)
  2 16 32
 )
 (list 2 0 1)
)"
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn access_slice_access_transpose_composition_commutative_1() {
        let program = "
(access-slice
 (access-transpose
  (access (access-tensor t-3-32-32) 1)
  (list 2 0 1)
 )
 1 0 2
)
        "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        let rws = vec![super::access_slice_access_transpose_composition_commutative()];
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        let matches = "
(access-transpose
 (access-slice
  (access (access-tensor t-3-32-32) 1)
  0 0 2
 )
 (list 2 0 1)
)"
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn access_slice_access_transpose_composition_commutative_2() {
        let program = "
(access-slice
 (access-transpose
  (access (access-tensor t-3-32-32) 1)
  (list 1 0 2)
 )
 2 0 16
)
        "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        let rws = vec![super::access_slice_access_transpose_composition_commutative()];
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        let matches = "
(access-transpose
 (access-slice (access (access-tensor t-3-32-32) 1) 2 0 16)
 (list 1 0 2)
)"
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn access_slice_access_transpose_composition_commutative_3() {
        let program = "
(access-slice
 (access-transpose
  (access (access-tensor t-3-32-32) 1)
  (list 1 0 2)
 )
 1 0 2
)"
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        let rws = vec![super::access_slice_access_transpose_composition_commutative()];
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        let matches = "
(access-transpose
 (access-slice (access (access-tensor t-3-32-32) 1) 0 0 2)
 (list 1 0 2)
)"
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn bubble_access_concatenate_through_access_transpose_0() {
        let program = "
(access-transpose
 (access-concatenate
  (access (access-tensor t-3-32-32) 1)
  (access (access-tensor t-3-32-32) 1)
  0
 )
 (list 1 2 0)
)"
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        let rws = vec![super::bubble_access_concatenate_through_access_transpose()];
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        let matches = "
(access-concatenate
 (access-transpose (access (access-tensor t-3-32-32) 1) (list 1 2 0))
 (access-transpose (access (access-tensor t-3-32-32) 1) (list 1 2 0))
 2
)"
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn bubble_access_concatenate_through_access_transpose_1() {
        let program = "
(access-transpose
 (access-concatenate
  (access (access-tensor t-3-32-32) 1)
  (access (access-tensor t-3-32-32) 1)
  1
 )
 (list 1 2 0)
)"
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        let rws = vec![super::bubble_access_concatenate_through_access_transpose()];
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        let matches = "
(access-concatenate
 (access-transpose (access (access-tensor t-3-32-32) 1) (list 1 2 0))
 (access-transpose (access (access-tensor t-3-32-32) 1) (list 1 2 0))
 0
)"
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn bubble_access_concatenate_through_access_transpose_2() {
        let program = "
(access-transpose
 (access-concatenate
  (access (access-tensor t-3-32-32) 1)
  (access (access-tensor t-3-32-32) 1)
  2
 )
 (list 1 0 2)
)"
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        let rws = vec![super::bubble_access_concatenate_through_access_transpose()];
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        let matches = "
(access-concatenate
 (access-transpose (access (access-tensor t-3-32-32) 1) (list 1 0 2))
 (access-transpose (access (access-tensor t-3-32-32) 1) (list 1 0 2))
 2
)"
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn bubble_access_concatenate_through_access_transpose_3() {
        let program = "
(access-transpose
 (access-concatenate
  (access (access-tensor t-3-32-32) 1)
  (access (access-tensor t-3-32-32) 1)
  1
 )
 (list 2 0 1)
)"
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        let rws = vec![super::bubble_access_concatenate_through_access_transpose()];
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        let matches = "
(access-concatenate
 (access-transpose (access (access-tensor t-3-32-32) 1) (list 2 0 1))
 (access-transpose (access (access-tensor t-3-32-32) 1) (list 2 0 1))
 2
)"
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
    fn bubble_access_concatenate_through_compute_dot_product_not_item_axis() {
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

    #[test]
    fn collapse_nested_transposes() {
        let program = "
             (access-transpose
              (access-transpose
               (access (access-tensor t) 1)
               (list 1 3 2 0)
              )
              (list 3 2 1 0)
             )"
        .parse()
        .unwrap();
        let mut map = HashMap::default();
        map.insert("t".to_string(), vec![1, 2, 3, 4]);
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        let rws = vec![super::collapse_nested_transposes()];
        let runner = Runner::<_, _, ()>::default().with_egraph(egraph).run(&rws);

        let matches = "(access-transpose (access (access-tensor t) 1) (list 0 2 3 1))"
            .parse::<Pattern<_>>()
            .unwrap()
            .search_eclass(&runner.egraph, id)
            .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn remove_trivial_transpose() {
        let program = "
              (access-transpose
               (access (access-tensor t) 1)
               (list 0 1 2 3)
              )
             "
        .parse()
        .unwrap();
        let mut map = HashMap::default();
        map.insert("t".to_string(), vec![1, 2, 3, 4]);
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        let rws = vec![super::remove_trivial_transpose()];
        let runner = Runner::<_, _, ()>::default().with_egraph(egraph).run(&rws);

        let matches = "(access (access-tensor t) 1)"
            .parse::<Pattern<_>>()
            .unwrap()
            .search_eclass(&runner.egraph, id)
            .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn collapse_nested_accesses() {
        let program = "(access (access (access-tensor t) 0) 1)".parse().unwrap();
        let mut map = HashMap::default();
        map.insert("t".to_string(), vec![1, 2, 3, 4]);
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        let rws = vec![super::collapse_nested_accesses()];
        let runner = Runner::<_, _, ()>::default().with_egraph(egraph).run(&rws);

        let matches = "(access (access-tensor t) 1)"
            .parse::<Pattern<_>>()
            .unwrap()
            .search_eclass(&runner.egraph, id)
            .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn pad_slice_accesses_pad_to_multiples() {
        let program = "(access (access-tensor t) 0)".parse().unwrap();
        let mut map = HashMap::default();
        map.insert("t".to_string(), vec![1, 2, 3, 4]);
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        let rws = vec![super::pad_slice_accesses(
            0,
            PadSliceStrategy::PadToMultiplesOf {
                limit: 16,
                multiples_of: 4,
                pad_location: PadLocation::End,
                pad_type: PadType::ZeroPadding,
            },
        )];
        let runner = Runner::<_, _, ()>::default().with_egraph(egraph).run(&rws);

        for dim_val in &[4, 8, 12, 16] {
            let matches = format!(
                "(access-slice
                  (access-pad
                   (access (access-tensor t) 0)
                   zero-padding
                   0 0 {}
                  )
                  0 0 1
                 )",
                dim_val - 1
            )
            .parse::<Pattern<_>>()
            .unwrap()
            .search_eclass(&runner.egraph, id)
            .unwrap();
            assert_eq!(matches.substs.len(), 1);
        }
    }

    /// This test tests the newer conv2d syntax, on a more realistically-shaped
    /// set of inputs.
    #[test]
    fn conv2d_im2col_systolic_array_1() {
        test_logger::ensure_env_logger_initialized();

        pub const IMAGE_SHAPE: &[usize] = &[1, 32, 32, 3];
        pub const KERNEL_SHAPE: &[usize] = &[3, 3, 3, 8];

        let mut expr = RecExpr::from_str("(access-tensor image)").unwrap();
        let data_id = expr.as_ref().len() - 1;
        let weights_id = expr.add(Language::Symbol("weights".to_string()));
        let weights_id = expr.add(Language::AccessTensor(weights_id));

        let _conv2d_id = crate::language::from_relay::conv2d(
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

        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&expr);

        let rws = vec![
            super::flatten_unflatten_any_access(),
            super::bubble_reshape_through_cartesian_product(),
            super::bubble_reshape_through_compute_dot_product(),
            super::systolic_array(),
        ];

        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        // Expected original program.
        let matches = "
(access-transpose
 (access-transpose
  (compute dot-product
   (access-cartesian-product
    (access
     (access-transpose (access-tensor weights) (list 3 2 0 1))
     1
    )
    (access
     (access-squeeze
       (access-windows
        (access
         (access-pad
          (access-pad
           (access-transpose (access-tensor image) (list 0 3 1 2))
           zero-padding
           2 1 1
          )
          zero-padding
          3 1 1
         )
         1
        )
        (shape 3 3 3)
        (shape 1 1 1)
       )
      1
     )
     3
    )
   )
  )
  (list 1 0 2 3)
 )
 (list 0 2 3 1)
)
            "
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);

        // Find the version with flattened and reshaped weight/image arguments
        let matches = "
(access-transpose
 (access-transpose
  (compute dot-product
   (access-cartesian-product
    (access-reshape
     (access-flatten
      (access
       (access-transpose (access-tensor weights) (list 3 2 0 1))
       1
      )
     )
     (access-shape (shape 8) (shape 3 3 3))
    )
    (access-reshape
     (access-flatten
      (access
       (access-squeeze
         (access-windows
          (access
           (access-pad
            (access-pad
             (access-transpose (access-tensor image) (list 0 3 1 2))
             zero-padding
             2 1 1
            )
            zero-padding
            3 1 1
           )
           1
          )
          (shape 3 3 3)
          (shape 1 1 1)
         )
        1
       )
       3
      )
     )
     (access-shape (shape 1 32 32) (shape 3 3 3))
    )
   )
  )
  (list 1 0 2 3)
 )
 (list 0 2 3 1)
)
            "
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);

        let matches = "
(access-transpose
 (access-transpose
  (access-reshape
   (systolic-array 27 1024
    (access-flatten
     (access
      (access-transpose (access-tensor weights) (list 3 2 0 1))
      1
     )
    )
    (access
     (access-transpose
      (access-flatten
       (access
        (access-squeeze
          (access-windows
           (access
            (access-pad
             (access-pad
              (access-transpose (access-tensor image) (list 0 3 1 2))
              zero-padding
              2 1 1
             )
             zero-padding
             3 1 1
            )
            1
           )
           (shape 3 3 3)
           (shape 1 1 1)
          )
         1
        )
        3
       )
      )
      (list 1 0)
     )
     0
    )
   )
   (access-shape (shape 8 1 32 32) (shape))
  )
  (list 1 0 2 3)
 )
 (list 0 2 3 1)
)
            "
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn bubble_access_slice_through_access_pad_inequal_axes() {
        let program = "
              (access-pad
               (access-slice
                (access-tensor t)
                0 0 8
               )
               zero-padding 1 2 2
              )
             "
        .parse()
        .unwrap();
        let mut map = HashMap::default();
        map.insert("t".to_string(), vec![8, 10]);
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        let rws = vec![super::bubble_access_slice_through_access_pad_inequal_axes()];
        let runner = Runner::<_, _, ()>::default().with_egraph(egraph).run(&rws);

        let matches = "
              (access-slice
               (access-pad
                (access-tensor t)
                zero-padding 1 2 2
               )
               0 0 8
              )
"
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn pad_slice_accesses_pad_to_closest_multiple() {
        let program = "(access (access-tensor t) 0)".parse().unwrap();
        let mut map = HashMap::default();
        map.insert("t".to_string(), vec![1, 2, 3, 4]);
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        let rws = vec![
            super::pad_slice_accesses(
                0,
                PadSliceStrategy::PadToClosestMultipleOf {
                    multiple_of: 3,
                    pad_location: PadLocation::End,
                    pad_type: PadType::ZeroPadding,
                },
            ),
            super::pad_slice_accesses(
                1,
                PadSliceStrategy::PadToClosestMultipleOf {
                    multiple_of: 3,
                    pad_location: PadLocation::End,
                    pad_type: PadType::ZeroPadding,
                },
            ),
            super::pad_slice_accesses(
                2,
                PadSliceStrategy::PadToClosestMultipleOf {
                    multiple_of: 3,
                    pad_location: PadLocation::End,
                    pad_type: PadType::ZeroPadding,
                },
            ),
            super::pad_slice_accesses(
                3,
                PadSliceStrategy::PadToClosestMultipleOf {
                    multiple_of: 3,
                    pad_location: PadLocation::End,
                    pad_type: PadType::ZeroPadding,
                },
            ),
        ];
        let runner = Runner::<_, _, ()>::default().with_egraph(egraph).run(&rws);

        for class in runner.egraph.classes() {
            match &class.data {
                MyAnalysisData::AccessPattern(a) => {
                    for &shape_dim_val in a.shape.slice() {
                        assert!(shape_dim_val <= 6);
                    }
                    for &item_shape_dim_val in a.item_shape.slice() {
                        assert!(item_shape_dim_val <= 6);
                    }
                }
                _ => (),
            }
        }

        runner.print_report();

        let matches = "
                 (access-slice
                  (access-pad
                   (access (access-tensor t) 0)
                   zero-padding
                   0 0 2
                  )
                  0 0 1
                 )"
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);

        let matches = "
                 (access-slice
                  (access-pad
                   (access (access-tensor t) 0)
                   zero-padding
                   1 0 1
                  )
                  1 0 2
                 )"
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);

        let matches = "
                 (access-slice
                  (access-pad
                   (access (access-tensor t) 0)
                   zero-padding
                   2 0 0
                  )
                  2 0 3
                 )"
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id);
        assert!(matches.is_none());

        let matches = "
                 (access-slice
                  (access-pad
                   (access (access-tensor t) 0)
                   zero-padding
                   3 0 2
                  )
                  3 0 4
                 )"
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);

        // Shouldn't pad up to the next highest multiple. I.e. length 1
        // shouldn't go to 6, just to 3.
        let matches = "
                 (access-slice
                  (access-pad
                   (access (access-tensor t) 0)
                   zero-padding
                   0 0 5
                  )
                  0 0 1
                 )"
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id);
        assert!(matches.is_none());
    }

    #[test]
    fn bubble_access_slice_through_access_cartesian_product_left() {
        let program = "
             (access-cartesian-product
              (access-slice (access (access-tensor a) 1) 0 0 2)
              (access (access-tensor b) 1)
             )"
        .parse()
        .unwrap();
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![4, 3, 3, 4]);
        map.insert("b".to_string(), vec![10, 3, 3, 4]);
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        let rws =
            vec![super::bubble_access_slice_through_access_cartesian_product_not_item_axis_left()];
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        let matches = "
             (access-slice
              (access-cartesian-product
               (access (access-tensor a) 1)
               (access (access-tensor b) 1)
              )
              0 0 2
             )"
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn bubble_access_slice_through_access_cartesian_product_right() {
        let program = "
             (access-cartesian-product
              (access (access-tensor b) 1)
              (access-slice (access (access-tensor a) 1) 0 0 2)
             )"
        .parse()
        .unwrap();
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![4, 3, 3, 4]);
        map.insert("b".to_string(), vec![10, 3, 3, 4]);
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        let rws =
            vec![super::bubble_access_slice_through_access_cartesian_product_not_item_axis_right()];
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        let matches = "
             (access-slice
              (access-cartesian-product
               (access (access-tensor b) 1)
               (access (access-tensor a) 1)
              )
              1 0 2
             )"
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn bubble_access_slice_through_access_cartesian_product_same_item_axis() {
        let program = "
             (access-cartesian-product
              (access-slice (access (access-tensor a) 2) 3 1 2)
              (access-slice (access (access-tensor b) 1) 2 1 2)
             )"
        .parse()
        .unwrap();
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![4, 16, 3, 3, 4]);
        map.insert("b".to_string(), vec![10, 3, 3, 4]);
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        let rws =
            vec![super::bubble_access_slice_through_access_cartesian_product_same_item_axis()];
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        let matches = "
             (access-slice
              (access-cartesian-product
               (access (access-tensor a) 2)
               (access (access-tensor b) 1)
              )
              5 1 2
             )"
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn bubble_access_slice_through_compute_dot_product_not_item_axis() {
        let program = "
             (compute dot-product
              (access-slice
               (access (access-tensor a) 1)
               0 2 3
              )
             )"
        .parse()
        .unwrap();
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![4, 16, 3, 3, 4]);
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        let rws = vec![super::bubble_access_slice_through_compute_dot_product_not_item_axis()];
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        let matches = "
             (access-slice
              (compute dot-product (access (access-tensor a) 1))
              0 2 3
             )
             "
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn bubble_access_slice_through_compute_dot_product_item_axis_not_tuple_axis() {
        let program = "
             (compute dot-product
              (access-slice
               (access-pad
                (access (access-tensor a) 1)
                zero-padding
                3 2 3
               )
               3 2 5
              )
             )"
        .parse()
        .unwrap();
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![4, 16, 3, 3, 4]);
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        let rws =
            vec![super::bubble_access_slice_through_compute_dot_product_item_axis_not_tuple_axis()];
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        let matches = "
              (compute dot-product
               (access-pad
                (access (access-tensor a) 1)
                zero-padding
                3 2 3
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
    fn systolic_array_with_blocking() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![32, 64]);
        map.insert("b".to_string(), vec![32, 64]);
        let program = "
         (compute dot-product
          (access-cartesian-product
           (access (access-tensor a) 1)
           (access (access-tensor b) 1)
          )
         )
        "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);

        let rws = vec![
            super::systolic_array_with_blocking(64, 32),
            super::systolic_array_with_blocking(32, 32),
            super::systolic_array_with_blocking(16, 32),
            super::systolic_array_with_blocking(32, 16),
            super::systolic_array_with_blocking(2, 2),
            // Shouldn't tensorize to this one.
            super::systolic_array_with_blocking(32, 3),
        ];

        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);
        match runner.stop_reason.unwrap() {
            egg::StopReason::Saturated => (),
            _ => panic!(),
        };

        let matches = "
          (systolic-array-with-blocking 64 32
           (access (access-tensor a) 1)
           (access (access-transpose  (access (access-tensor b) 1) (list 1 0)) 0)
          )
            "
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);

        let matches = "
          (systolic-array-with-blocking 32 32
           (access (access-tensor a) 1)
           (access (access-transpose  (access (access-tensor b) 1) (list 1 0)) 0)
          )
            "
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);

        let matches = "
          (systolic-array-with-blocking 16 32
           (access (access-tensor a) 1)
           (access (access-transpose  (access (access-tensor b) 1) (list 1 0)) 0)
          )
            "
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);

        let matches = "
          (systolic-array-with-blocking 32 16
           (access (access-tensor a) 1)
           (access (access-transpose  (access (access-tensor b) 1) (list 1 0)) 0)
          )
            "
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);

        let matches = "
          (systolic-array-with-blocking 2 2
           (access (access-tensor a) 1)
           (access (access-transpose  (access (access-tensor b) 1) (list 1 0)) 0)
          )
            "
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);

        let matches = "
          (systolic-array-with-blocking 32 3
           (access (access-tensor a) 1)
           (access (access-transpose  (access (access-tensor b) 1) (list 1 0)) 0)
          )
            "
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id);
        assert!(matches.is_none());
    }

    #[test]
    #[should_panic]
    fn conv2d_im2col_fc_systolic_array() {
        let data_shape = vec![1, 32, 32, 3];
        let kernel_shape = vec![3, 3, 3, 8];

        let mut expr = RecExpr::default();

        let data_id = expr.add(Language::Symbol("data".to_string()));
        let data_id = expr.add(Language::AccessTensor(data_id));

        let kernel_id = expr.add(Language::Symbol("kernel".to_string()));
        let kernel_id = expr.add(Language::AccessTensor(kernel_id));

        let _conv2d_id = crate::language::from_relay::conv2d(
            &mut expr,
            data_id,
            &data_shape,
            kernel_id,
            &kernel_shape,
            &[1, 1],
            &[1, 1, 1, 1],
            &[1, 1],
            1,
            "NHWC",
            "HWIO",
            "",
        );

        let mut map = HashMap::default();
        map.insert("data".to_string(), data_shape);
        map.insert("kernel".to_string(), kernel_shape);
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&expr);

        let rws = vec![
            // This rewrite enables im2col to be found.
            super::flatten_unflatten_any_access(),
            // These rewrites move the reshape into the right place.
            super::bubble_reshape_through_cartesian_product(),
            super::bubble_reshape_through_compute_dot_product(),
            // This rewrite tensorizes.
            super::systolic_array_conv2d_im2col_fc_with_blocking(32, 32),
        ];

        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);
        match runner.stop_reason.unwrap() {
            egg::StopReason::Saturated => (),
            _ => panic!(),
        };

        let matches = "
          (systolic-array-conv2d-im2col-fc-with-blocking 32 32
           (access (access-tensor data) 1)
           (access (access-tensor kernel) 0)
          )
            "
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    #[ignore = "ignored b/c broken during pldi push"]
    fn systolic_array_conv2d_nchw_oihw_with_blocking() {
        let data_shape = vec![1, 64, 32, 32]; // NCHW
        let kernel_shape = vec![128, 64, 3, 3]; // OIHW

        let mut expr = RecExpr::default();

        let data_id = expr.add(Language::Symbol("data".to_string()));
        let data_id = expr.add(Language::AccessTensor(data_id));

        let kernel_id = expr.add(Language::Symbol("kernel".to_string()));
        let kernel_id = expr.add(Language::AccessTensor(kernel_id));

        let _conv2d_id = crate::language::from_relay::conv2d(
            &mut expr,
            data_id,
            &data_shape,
            kernel_id,
            &kernel_shape,
            &[1, 1],
            &[1, 1, 1, 1],
            &[1, 1],
            1,
            "NCHW",
            "OIHW",
            "",
        );

        let mut map = HashMap::default();
        map.insert("data".to_string(), data_shape);
        map.insert("kernel".to_string(), kernel_shape);
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&expr);

        let rws = vec![
            super::systolic_array_conv2d_nchw_oihw_with_blocking(64, 32),
            super::systolic_array_conv2d_nchw_oihw_with_blocking(32, 32),
            super::systolic_array_conv2d_nchw_oihw_with_blocking(2, 2),
            super::systolic_array_conv2d_nchw_oihw_with_blocking(3, 2),
        ];

        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);
        match runner.stop_reason.unwrap() {
            egg::StopReason::Saturated => (),
            _ => panic!(),
        };

        let matches = "
          (systolic-array-conv2d-nchw-oihw-with-blocking
           64 32
           (access-tensor kernel)
           (access-pad (access-pad ?data zero-padding ?0 1 1) zero-padding ?1 1 1)
           3 3
           1 1
          )
            "
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);

        let _matches = "
          (systolic-array-conv2d-nchw-oihw-with-blocking
           32 32
           (access-tensor kernel)
           (access-pad (access-pad ?data zero-padding ?0 1 1) zero-padding ?1 1 1)
           3 3
           1 1
          )
            "
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        // I don't think this check makes sense.
        //assert_eq!(matches.substs.len(), 1);

        let _matches = "
          (systolic-array-conv2d-nchw-oihw-with-blocking
          2 2
           (access-tensor kernel)
           (access-pad (access-pad ?data zero-padding ?0 1 1) zero-padding ?1 1 1)
           3 3
           1 1
          )
            "
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        // I don't think this check makes sense.
        //assert_eq!(matches.substs.len(), 1);

        let matches = "
          (systolic-array-conv2d-nchw-oihw-with-blocking
          3 2
           (access-tensor kernel)
           (access-pad (access-pad ?data zero-padding ?0 1 1) zero-padding ?1 1 1)
           3 3
           1 1
          )
            "
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id);
        assert!(matches.is_none());
    }

    #[test]
    #[ignore = "ignored b/c broken during pldi push"]
    fn systolic_array_conv2d_nhwc_hwio_with_blocking() {
        let data_shape = vec![1, 32, 32, 64]; // NHWC
        let kernel_shape = vec![3, 3, 64, 128]; // HWIO

        let mut expr = RecExpr::default();

        let data_id = expr.add(Language::Symbol("data".to_string()));
        let data_id = expr.add(Language::AccessTensor(data_id));

        let kernel_id = expr.add(Language::Symbol("kernel".to_string()));
        let kernel_id = expr.add(Language::AccessTensor(kernel_id));

        let _conv2d_id = crate::language::from_relay::conv2d(
            &mut expr,
            data_id,
            &data_shape,
            kernel_id,
            &kernel_shape,
            &[1, 1],
            &[1, 1, 1, 1],
            &[1, 1],
            1,
            "NHWC",
            "HWIO",
            "",
        );

        let mut map = HashMap::default();
        map.insert("data".to_string(), data_shape);
        map.insert("kernel".to_string(), kernel_shape);
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&expr);

        let rws = vec![
            // This rewrite is needed to move the padding further "out" and the
            // transposes further "in", so that we can match on the layout
            // changes on the inputs.
            super::bubble_access_transpose_through_access_pad(),
            // These rewrites are needed to find the initial systolic array.
            super::systolic_array_conv2d_nchw_oihw_with_blocking(64, 32),
            super::systolic_array_conv2d_nchw_oihw_with_blocking(32, 32),
            super::systolic_array_conv2d_nchw_oihw_with_blocking(2, 2),
            super::systolic_array_conv2d_nchw_oihw_with_blocking(3, 2),
            super::systolic_array_conv2d_nhwc_hwio_with_blocking(64, 32),
            super::systolic_array_conv2d_nhwc_hwio_with_blocking(32, 32),
            super::systolic_array_conv2d_nhwc_hwio_with_blocking(2, 2),
            super::systolic_array_conv2d_nhwc_hwio_with_blocking(3, 2),
        ];

        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);
        match runner.stop_reason.unwrap() {
            egg::StopReason::Saturated => (),
            _ => panic!(),
        };

        let matches = "
          (systolic-array-conv2d-nhwc-hwio-with-blocking
           64 32
           (access-tensor kernel)
           (access-pad (access-pad ?data zero-padding ?0 1 1) zero-padding ?1 1 1)
           3 3
           1 1
          )
            "
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);

        let _matches = "
          (systolic-array-conv2d-nhwc-hwio-with-blocking
           32 32
           (access-tensor kernel)
           (access-pad (access-pad ?data zero-padding ?0 1 1) zero-padding ?1 1 1)
           3 3
           1 1
          )
            "
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        // I don't think this check makes sense.
        //assert_eq!(matches.substs.len(), 1);

        let _matches = "
          (systolic-array-conv2d-nhwc-hwio-with-blocking
          2 2
           (access-tensor kernel)
           (access-pad (access-pad ?data zero-padding ?0 1 1) zero-padding ?1 1 1)
           3 3
           1 1
          )
            "
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        // I don't think this check makes sense.
        //assert_eq!(matches.substs.len(), 1);

        let matches = "
          (systolic-array-conv2d-nhwc-hwio-with-blocking
          3 2
           (access-tensor kernel)
           (access-pad (access-pad ?data zero-padding ?0 1 1) zero-padding ?1 1 1)
           3 3
           1 1
          )
            "
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id);
        assert!(matches.is_none());
    }

    #[test]
    fn bubble_access_transpose_through_access_pad() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![32, 64, 3, 2]);
        let program = "
         (access-pad
          (access-transpose
           (access (access-tensor a) 1)
           (list 3 1 2 0)
          )
          zero-padding
          0 2 3
         )
        "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);

        let rws = vec![super::bubble_access_transpose_through_access_pad()];

        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);
        match runner.stop_reason.unwrap() {
            egg::StopReason::Saturated => (),
            _ => panic!(),
        };

        let matches = "
          (access-transpose
           (access-pad
            (access (access-tensor a) 1)
            zero-padding
            3 2 3
           )
           (list 3 1 2 0)
          )
            "
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn systolic_array_conv2d_im2col_nchw_oihw_with_blocking() {
        let data_shape = vec![1, 64, 32, 32]; // NCHW
        let kernel_shape = vec![128, 64, 3, 3]; // OIHW

        let mut expr = RecExpr::default();

        let data_id = expr.add(Language::Symbol("data".to_string()));
        let data_id = expr.add(Language::AccessTensor(data_id));

        let kernel_id = expr.add(Language::Symbol("kernel".to_string()));
        let kernel_id = expr.add(Language::AccessTensor(kernel_id));

        let _conv2d_id = crate::language::from_relay::conv2d(
            &mut expr,
            data_id,
            &data_shape,
            kernel_id,
            &kernel_shape,
            &[1, 1],
            &[1, 1, 1, 1],
            &[1, 1],
            1,
            "NCHW",
            "OIHW",
            "",
        );

        let mut map = HashMap::default();
        map.insert("data".to_string(), data_shape);
        map.insert("kernel".to_string(), kernel_shape);
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&expr);

        let rws = vec![
            // These are needed to find the im2col convolution
            super::flatten_unflatten_any_access(),
            super::bubble_reshape_through_cartesian_product(),
            super::bubble_reshape_through_compute_dot_product(),
            super::systolic_array_conv2d_im2col_nchw_oihw_with_blocking(64, 32),
            super::systolic_array_conv2d_im2col_nchw_oihw_with_blocking(32, 32),
            super::systolic_array_conv2d_im2col_nchw_oihw_with_blocking(2, 2),
            super::systolic_array_conv2d_im2col_nchw_oihw_with_blocking(3, 2),
        ];

        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        let matches = "
          (systolic-array-conv2d-im2col-nchw-oihw-with-blocking
           64 32
           (access-tensor kernel)
           (access-pad (access-pad ?data zero-padding ?0 1 1) zero-padding ?1 1 1)
           3 3
           1 1
          )
            "
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);

        let _matches = "
          (systolic-array-conv2d-im2col-nchw-oihw-with-blocking
           32 32
           (access-tensor kernel)
           (access-pad (access-pad ?data zero-padding ?0 1 1) zero-padding ?1 1 1)
           3 3
           1 1
          )
            "
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        // I don't think this check makes sense.
        //assert_eq!(matches.substs.len(), 1);

        let _matches = "
          (systolic-array-conv2d-im2col-nchw-oihw-with-blocking
          2 2
           (access-tensor kernel)
           (access-pad (access-pad ?data zero-padding ?0 1 1) zero-padding ?1 1 1)
           3 3
           1 1
          )
            "
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        // I don't think this check makes sense.
        //assert_eq!(matches.substs.len(), 1);

        let _matches = "
          (systolic-array-conv2d-im2col-nchw-oihw-with-blocking
          3 2
           (access-tensor kernel)
           (access-pad (access-pad ?data zero-padding ?0 1 1) zero-padding ?1 1 1)
           3 3
           1 1
          )
            "
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        // I don't think this check makes sense.
        //assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn systolic_array_conv2d_im2col_nhwc_hwio_with_blocking() {
        let data_shape = vec![1, 32, 32, 64]; // NHWC
        let kernel_shape = vec![3, 3, 64, 128]; // HWIO

        let mut expr = RecExpr::default();

        let data_id = expr.add(Language::Symbol("data".to_string()));
        let data_id = expr.add(Language::AccessTensor(data_id));

        let kernel_id = expr.add(Language::Symbol("kernel".to_string()));
        let kernel_id = expr.add(Language::AccessTensor(kernel_id));

        let _conv2d_id = crate::language::from_relay::conv2d(
            &mut expr,
            data_id,
            &data_shape,
            kernel_id,
            &kernel_shape,
            &[1, 1],
            &[1, 1, 1, 1],
            &[1, 1],
            1,
            "NHWC",
            "HWIO",
            "",
        );

        let mut map = HashMap::default();
        map.insert("data".to_string(), data_shape);
        map.insert("kernel".to_string(), kernel_shape);
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&expr);

        let rws = vec![
            // These are needed to find the im2col convolution
            super::flatten_unflatten_any_access(),
            super::bubble_reshape_through_cartesian_product(),
            super::bubble_reshape_through_compute_dot_product(),
            // These tensorize to nchw systolic arrays, which is needed to find
            // nhwc
            super::systolic_array_conv2d_im2col_nchw_oihw_with_blocking(64, 32),
            super::systolic_array_conv2d_im2col_nchw_oihw_with_blocking(32, 32),
            super::systolic_array_conv2d_im2col_nchw_oihw_with_blocking(2, 2),
            super::systolic_array_conv2d_im2col_nchw_oihw_with_blocking(3, 2),
            // This rewrite is needed to move the padding further "out" and the
            // transposes further "in", so that we can match on the layout
            // changes on the inputs.
            super::bubble_access_transpose_through_access_pad(),
            // These rewrites tensorize.
            super::systolic_array_conv2d_im2col_nhwc_hwio_with_blocking(64, 32),
            super::systolic_array_conv2d_im2col_nhwc_hwio_with_blocking(32, 32),
            super::systolic_array_conv2d_im2col_nhwc_hwio_with_blocking(2, 2),
            super::systolic_array_conv2d_im2col_nhwc_hwio_with_blocking(3, 2),
        ];

        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);
        runner.print_report();

        let matches = "
          (systolic-array-conv2d-im2col-nhwc-hwio-with-blocking
           64 32
           (access-tensor kernel)
           (access-pad (access-pad ?data zero-padding ?0 1 1) zero-padding ?1 1 1)
           3 3
           1 1
          )
            "
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);

        let _matches = "
          (systolic-array-conv2d-im2col-nhwc-hwio-with-blocking
           32 32
           (access-tensor kernel)
           (access-pad (access-pad ?data zero-padding ?0 1 1) zero-padding ?1 1 1)
           3 3
           1 1
          )
            "
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        // I don't think this check makes sense.
        //assert_eq!(matches.substs.len(), 1);

        let _matches = "
          (systolic-array-conv2d-im2col-nhwc-hwio-with-blocking
          2 2
           (access-tensor kernel)
           (access-pad (access-pad ?data zero-padding ?0 1 1) zero-padding ?1 1 1)
           3 3
           1 1
          )
            "
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        // I don't think this check makes sense.
        //assert_eq!(matches.substs.len(), 1);

        let _matches = "
          (systolic-array-conv2d-im2col-nhwc-hwio-with-blocking
          3 2
           (access-tensor kernel)
           (access-pad (access-pad ?data zero-padding ?0 1 1) zero-padding ?1 1 1)
           3 3
           1 1
          )
            "
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        // I don't think this check makes sense.
        //assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn flexasr_maxpool() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![32, 32]);
        let program = "
         (compute reduce-max (access-windows (access (access-tensor a) 1) (shape 2) (shape 2)))
        "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);

        let rws = vec![super::flexasr_maxpool()];

        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);
        match runner.stop_reason.unwrap() {
            egg::StopReason::Saturated => (),
            _ => panic!(),
        };

        let matches = "
            (access
             (access-transpose
              (accelerator-call flex-maxpool
               (access 
                (access-transpose 
                 (access (access-tensor a) 1)
                 (list 1 0))
                 0) ?shape)
              (list 1 0))
             1)"
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn reassociate_max() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![16, 4]);
        let program = "
         (compute reduce-max (access (access-tensor a) 1))
        "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);

        let rws = vec![super::reassociate_max(2, 2)];

        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        let matches = "
            (compute reduce-max
             (access
              (compute reduce-max
               (access-windows
                (access (access-tensor a) 1)
                (shape 2)
                (shape 2)))
              1))
             "
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn reassociate_max_maxpool_2d() {
        let mut map = HashMap::default();
        map.insert("data".to_string(), vec![16, 4]);
        let program = "
         (compute reduce-max 
          (access-windows 
           (access (access-tensor data) 1) 
           (shape 4) (shape 4)))
        "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);

        let rws = vec![super::reassociate_max(2, 2)];

        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        let matches = "
            (compute reduce-max
             (access
              (compute reduce-max
               (access-windows
                (access-windows 
                 (access (access-tensor data) 1) 
                 (shape 4) (shape 4))
                (shape 2)
                (shape 2)))
              2))
             "
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    /// Test in which we map a small 2D max pool layer to FlexASR. This is done
    /// by:
    ///
    /// 1. Flattening the input windows to vectors,
    /// 2. Converting the reduce-max computation to reduce in windows of two
    ///    (FlexASR-style) rather than reducing the entire vector all at once,
    /// 3. Mapping the new reduce-max computations to FlexASR.
    ///
    /// There are also various rewrites used for cleanup and exploration.
    #[test]
    fn flexasr_maxpool_split_tensorize() {
        // Very simple 2D max pool layer. We use small shapes here so that we
        // can write out the final expression by hand; the larger the reduction,
        // the larger the final expression. Note that this is the max pool
        // described in our original paper.
        let program = "
         (compute reduce-max 
          (access-windows 
           (access (access-tensor data) 2) 
           (shape 2 2) (shape 2 2)))
         "
        .parse()
        .unwrap();

        // Define the shape of the input data: batch, channels, height, width.
        let mut map = HashMap::default();
        map.insert("data".to_string(), vec![3, 16, 4, 4]);

        // Insert the expression into the egraph.
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);

        // Define our rewrites. These rewrites are what map the max pool to
        // FlexASR.
        let rws = vec![
            // Performs initial flattening of 2D pooling windows into vectors.
            super::flatten_unflatten_any_access(),
            // Splits a reduce-max into two reduce-maxes: the first reduce max
            // reduces the data in windows of size 2, striding by 2 (FlexASR
            // style) while the second reduce max just reduces the rest all at
            // once. This is the core rewrite for transforming max pools to a
            // format which can be mapped to FlexASR.
            super::reassociate_max(2, 2),
            // Tensorize.
            super::flexasr_maxpool(),
            //
            // The rest of the rewrites are needed for cleanup.
            //
            // Moves the access-reshape which results from the flatten-unflatten
            // rewrite up through the program.
            super::bubble_access_reshape_through_compute_reduce_max(),
            // Collapses adjacent operators.
            super::simplify_multiple_accesses(),
            super::simplify_multiple_transposes(),
            super::simplify_multiple_access_reshapes(),
            // Move access through access-transpose, to enable more collapsing.
            super::bubble_access_through_access_transpose(),
            // Remove the topmost reduce-max which becomes "trivial" (i.e. a max
            // over a single element) after we rewrite it multiple times to
            // FlexASR invocations.
            super::simplify_reduce_max(),
        ];

        // Run the rewrites.
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .with_iter_limit(7)
            .run(&rws);

        // Assert that we find the expected program in the egraph.
        //
        // The final program computes the original max pool with two invocations
        // of FlexASR.
        //
        // Reading from inside to out:
        // 1. We first form 2x2 windows over the data. These are the windows we
        //    want to reduce via the max operator.
        // 2. We then flatten those 2x2 windows into vectors of length 4.
        // 3. We transpose the data so that it's in the format expected by
        //    FlexASR.
        // 4. We compute multiple max pools on FlexASR.
        // 5. We transpose the data back to its original layout, and reshape it
        //    to its final reshape.
        //
        // Note that operators like access-reshape, access-flatten, and access
        // are operators which exist purely to keep the types in check in
        // Glenside. They do not involve actual data movement.
        let matches = "
         (access-reshape
          (access
           (access-transpose
            (accelerator-call flex-maxpool
             (access
              (accelerator-call flex-maxpool
               (access
                (access-transpose
                 (access-flatten
                  (access-windows
                   (access (access-tensor data) 2)
                   (shape 2 2)
                   (shape 2 2)))
                 (list 1 0))
                0) ?shape0)
              0) ?shape1)
            (list 1 0))
           1)
          (access-shape (shape 3 16 2 2) (shape)))
        "
        .parse::<Pattern<_>>()
        .unwrap()
        .search_eclass(&runner.egraph, id)
        .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn bubble_access_reshape_through_compute_reduce_max() {
        let mut map = HashMap::default();
        map.insert("data".to_string(), vec![16, 4]);

        let program = "
         (compute reduce-max
          (access-reshape 
           (access (access-tensor data) 1)
           (access-shape (shape 4 4) (shape 1 2 2))))"
            .parse()
            .unwrap();

        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);

        let rws = vec![super::bubble_access_reshape_through_compute_reduce_max()];

        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        let matches = "
         (access-reshape
          (compute reduce-max 
           (access (access-tensor data) 1))
          (access-shape (shape 4 4) (shape)))"
            .parse::<Pattern<_>>()
            .unwrap()
            .search_eclass(&runner.egraph, id)
            .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn simplify_multiple_transposes_0() {
        let mut map = HashMap::default();
        map.insert("data".to_string(), vec![2, 3, 4, 5]);

        let program = "
         (access-transpose (access-transpose (access-tensor data) (list 3 1 0 2)) (list 2 1 3 0))
         "
        .parse()
        .unwrap();

        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);

        let rws = vec![super::simplify_multiple_transposes()];

        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        let matches = "(access-tensor data)"
            .parse::<Pattern<_>>()
            .unwrap()
            .search_eclass(&runner.egraph, id)
            .unwrap();
        assert_eq!(matches.substs.len(), 1);
    }

    #[test]
    fn simplify_multiple_transposes_1() {
        let mut map = HashMap::default();
        map.insert("data".to_string(), vec![2, 3, 4, 5]);

        let program = "
         (access-transpose (access-transpose (access-tensor data) (list 3 1 0 2)) (list 2 3 1 0))
         "
        .parse()
        .unwrap();

        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);

        let rws = vec![super::simplify_multiple_transposes()];

        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .run(&rws);

        assert!("(access-tensor data)"
            .parse::<Pattern<_>>()
            .unwrap()
            .search_eclass(&runner.egraph, id)
            .is_none());
    }

    /// Creates a Relay-to-Glenside test
    /// The test does the following:
    ///  1. Converts $relay_str to glenside by running the from_relay.py script
    ///  2. Inserts the resulting Glenside code into an egraph
    ///  3. Searches the egraph for $glenside_str to ensure the expected program
    ///     exists
    /// $test_name: the name of the created test
    /// $relay_str: A string containing the Relay program to be converted
    /// $glenside_str: A string containing the expected resulting Glenside
    ///     expression
    /// $tol: Tolerance value.
    /// $rws: Expression evaluating to a vector of rewrites.
    macro_rules! test {
        ($test_name:ident, $tol:literal, $relay_str:expr, $glenside_str:expr, $rws: expr) => {
            // TODO(@gussmith23) Hardcoding to f32
            test!(
                $test_name,
                $tol,
                $relay_str,
                $glenside_str,
                "",
                Uniform::new(-1f32, 1f32),
                $rws
            );
        };
        ($test_name:ident, $tol:literal, $relay_str:expr, $glenside_str:expr, $optional_arg:literal, $rws: expr) => {
            // TODO(@gussmith23) Hardcoding to f32
            test!(
                $test_name,
                $tol,
                $relay_str,
                $glenside_str,
                $optional_arg,
                Uniform::new(-1f32, 1f32),
                $rws
            );
        };
        ($test_name:ident, $tol:literal, $relay_str:expr, $glenside_str:expr, $optional_arg:literal, $distribution:expr, $rws: expr) => {
            #[test]
            fn $test_name() {
                // The number of times to run each program and compare their
                // outputs.
                // TODO(@gussmith23) # random samples chosen arbitrarily
                const SAMPLES: usize = 3;

                // Random number generator for generating random tensors.
                const SEED: u64 = 23;
                let mut tensor_rng = SmallRng::seed_from_u64(SEED);

                let module = tvm::ir::module::IRModule::parse("", $relay_str).unwrap();

                let (expr, shapes_vec, dtypes_vec, _) =
                    from_relay::from_relay(&module, false, &vec![]);

                let mut env = HashMap::default();
                for (k, v) in &shapes_vec {
                    env.insert(k.clone(), v.clone());
                }

                // TODO(@gussmith23) Include some simple simplifying rewrites
                // If we add some very basic rewrites here, then $glenside_str
                // won't need to exactly match what's actually produced by
                // from_relay.py. It can be simpler (e.g. collapsing accesses).
                let mut egraph = EGraph::new(MyAnalysis {
                    name_to_shape: env.clone(),
                    name_to_dtype: dtypes_vec.into_iter().collect(),
                });
                let id = egraph.add_expr(&expr);

                let pattern = $glenside_str.parse::<Pattern<Language>>().unwrap();
                // The program should not be found at first.
                assert!(pattern.search_eclass(&egraph, id).is_none());

                // Run compilation rewrites.
                let runner = Runner::default().with_egraph(egraph).run($rws);

                // Program should now be found.
                assert!(pattern.search_eclass(&runner.egraph, id).is_some());

                let expr_to_interpret = RecExpr::from_str($glenside_str).unwrap();

                for _ in (0..SAMPLES) {
                    // Run interpreters and compare output.
                    let script_filepath = format!(
                        "{}/src/language/from_relay/run_relay.py",
                        env!("CARGO_MANIFEST_DIR")
                    );
                    // https://www.reddit.com/r/rust/comments/38jhva/piping_string_to_child_process_stdin/crvlqcd/?utm_source=reddit&utm_medium=web2x&context=3
                    // Output filename
                    // TODO(@gussmith23) Do we want this RNG to use SEED?
                    // I initially attempted to do this, but was running into issues
                    // (I think the same filename kept being generated b/c I wasn't
                    // using the RNG carefully...but maybe there's also something
                    // wrong w/ how I'm reading files!)
                    let output_filepath = std::env::temp_dir().join(format!(
                        "output-{}.npy",
                        rand::thread_rng()
                            .sample_iter(&rand::distributions::Alphanumeric)
                            .take(30)
                            .collect::<String>()
                    ));

                    let mut cmd = Command::new("python3");
                    cmd.arg(script_filepath);
                    if $optional_arg.len() > 0 {
                        cmd.arg($optional_arg);
                    }
                    cmd.arg("--npy_out_filepath");
                    cmd.arg(&output_filepath);
                    cmd.arg("--npy_arg_filepath");
                    cmd.stdin(std::process::Stdio::piped())
                        .stdout(std::process::Stdio::piped())
                        .stderr(std::process::Stdio::piped());
                    let mut env = HashMap::default();
                    for (name, shape) in shapes_vec.iter() {
                        // TODO(@gussmith23) output type assumption
                        let value = ndarray::ArrayD::<f32>::random_using(
                            shape.clone(),
                            $distribution,
                            &mut tensor_rng,
                        );
                        env.insert(name.as_str(), value.clone());
                        let filepath = std::env::temp_dir().join(format!(
                            "arg-{}.npy",
                            rand::thread_rng()
                                .sample_iter(&rand::distributions::Alphanumeric)
                                .take(30)
                                .collect::<String>()
                        ));
                        write_npy(&filepath, &value).unwrap();
                        cmd.arg(filepath);
                    }

                    let mut proc = cmd.spawn().ok().expect("Failed to spawn process");
                    proc.stdin
                        .as_mut()
                        .unwrap()
                        .write_all($relay_str.as_bytes())
                        .unwrap();
                    let output = proc.wait_with_output().unwrap();
                    // Check that it ran.
                    assert!(
                        output.status.success(),
                        "Running Relay code failed with code {:?}.\nstdout:\n{}\nstderr:\n{}",
                        output.status.code(),
                        std::str::from_utf8(output.stdout.as_slice())
                            .expect("Could not convert stderr to UTF8"),
                        std::str::from_utf8(output.stderr.as_slice())
                            .expect("Could not convert stderr to UTF8")
                    );

                    // TODO(@gussmith23) output type assumption
                    let relay_output: ndarray::ArrayD<f32> = read_npy(output_filepath).unwrap();
                    let interpreter_output = match interpret(
                        &expr_to_interpret,
                        expr_to_interpret.as_ref().len() - 1,
                        &env,
                    ) {
                        crate::language::interpreter::Value::Access(a) => a.tensor,
                        _ => panic!(),
                    };
                    assert!(
                        relay_output.abs_diff_eq(&interpreter_output, $tol),
                        "{:?}\nvs.\n{:?}",
                        relay_output,
                        interpreter_output
                    );
                }
            }
        };
    }

    test!(
        conv2d_relay_to_glenside,
        1e-5,
        r#"
#[version = "0.0.5"]
def @main(%data: Tensor[(1, 3, 32, 32), float32], %weights: Tensor[(8, 3, 3, 3), float32]) -> Tensor[(1, 8, 17, 12), float32] {
  nn.conv2d(%data, %weights, strides=[2, 3], padding=[1, 2, 3, 4]) /* ty=Tensor[(1, 8, 17, 12), float32] */
}
"#,
        r#"
(access-transpose
 (compute dot-product
  (access-cartesian-product
   (access (access-tensor weights) 1)
   (access
    (access-squeeze
     (access-windows
      (access
       (access-pad
        (access-pad
         (access-tensor data)
         zero-padding
         2 1 3
        )
        zero-padding
        3 2 4
       )
       1
      )
      (shape 3 3 3)
      (shape 1 2 3)
     )
     1
    )
    3
   )
  )
 )
 (list 1 0 2 3)
)
"#,
        &vec![super::conv2d_relay_to_glenside()]
    );

    test!(
        conv1d_relay_to_glenside,
        1e-6,
        r#"
    #[version = "0.0.5"]
    def @main(%data: Tensor[(1, 3, 32), float32], %weights: Tensor[(8, 3, 3), float32]) -> Tensor[(1, 8, 19), float32] {
        nn.conv1d(%data, %weights, strides=[2], padding=[3, 4]) /* ty=Tensor[(1, 8, 19), float32] */
    }
"#,
        r#"
(access-transpose
 (compute dot-product
   (access-cartesian-product
    (access (access-tensor weights) 1)
    (access-squeeze
     (access-windows
      (access
       (access-pad
        (access-tensor data)
        zero-padding
        2 3 4
       )
       1
      )
      (shape 3 3)
      (shape 1 2)
     )
     1
    )
   )
 )
 (list 1 0 2)
)
"#,
        &vec![super::conv1d_relay_to_glenside()]
    );
}
