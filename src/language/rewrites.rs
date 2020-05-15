use super::{Language, Meta};
use egg::Rewrite;
use ndarray::Dimension;

// TODO(gus) I think I should make this a conditional applier, and fold in
// checks to make sure it has a shape and that it's an input
pub struct SplitConcatApplier {
    pub a: egg::Var,
}
impl egg::Applier<Language, Meta> for SplitConcatApplier {
    fn apply_one(
        &self,
        egraph: &mut egg::EGraph<Language, Meta>,
        _id: egg::Id,
        subst: &egg::Subst,
    ) -> std::vec::Vec<egg::Id> {
        let a: egg::Id = subst[&self.a];
        let shape = egraph[a].metadata.shape.as_ref().unwrap().clone();
        //println!("{:?}", shape);

        assert_eq!(shape.as_array_view().len(), 2);
        assert_eq!(0, shape[0] % 16);
        assert_eq!(0, shape[1] % 16);

        let mut to_be_concatted_along_axis_0 = std::vec::Vec::default();
        for i in 0..shape[0] / 16 {
            let mut to_be_concatted_along_axis_1 = std::vec::Vec::default();
            for j in 0..shape[1] / 16 {
                use std::convert::TryInto;
                let x_slice_start = (16 * i).try_into().unwrap();
                let x_slice_end = (16 * (i + 1)).try_into().unwrap();
                let y_slice_start = (16 * j).try_into().unwrap();
                let y_slice_end = (16 * (j + 1)).try_into().unwrap();
                let x_slice_start_id: egg::Id =
                    egraph.add(egg::ENode::leaf(Language::Usize(x_slice_start)));
                let x_slice_end_id: egg::Id =
                    egraph.add(egg::ENode::leaf(Language::Usize(x_slice_end)));
                let y_slice_start_id: egg::Id =
                    egraph.add(egg::ENode::leaf(Language::Usize(y_slice_start)));
                let y_slice_end_id: egg::Id =
                    egraph.add(egg::ENode::leaf(Language::Usize(y_slice_end)));
                to_be_concatted_along_axis_1.push(egraph.add(egg::ENode::new(
                    Language::Slice,
                    vec![
                        a,
                        x_slice_start_id,
                        x_slice_end_id,
                        y_slice_start_id,
                        y_slice_end_id,
                    ],
                )));
            }
            // Args should be a list of the sliced arrays, plus the axis
            // along which to stitch them back together.
            let mut args: std::vec::Vec<egg::Id> = to_be_concatted_along_axis_1;
            args.push(egraph.add(egg::ENode::leaf(Language::Usize(1))));
            to_be_concatted_along_axis_0.push(egraph.add(egg::ENode::new(Language::Concat, args)));
        }
        let mut args: std::vec::Vec<egg::Id> = to_be_concatted_along_axis_0;
        args.push(egraph.add(egg::ENode::leaf(Language::Usize(0))));
        let concat_id: egg::Id = egraph.add(egg::ENode::new(Language::Concat, args));

        vec![concat_id]
    }
}
// TODO(gus) I think I should make this a conditional applier, and fold in
// checks to make sure it has a shape and that it's an input
pub fn has_shape(
    var: &'static str,
) -> impl Fn(&mut egg::EGraph<Language, Meta>, egg::Id, &egg::Subst) -> bool {
    let var = var.parse().unwrap();
    move |egraph, _, subst| !egraph[subst[&var]].metadata.shape.is_none()
}
pub fn is_symbol(
    var: &'static str,
) -> impl Fn(&mut egg::EGraph<Language, Meta>, egg::Id, &egg::Subst) -> bool {
    let var = var.parse().unwrap();
    // TODO(gus) should this be `all` or `any` or something else entirely?
    move |egraph, _, subst| {
        egraph[subst[&var]]
            .nodes
            .iter()
            .map(|enode| match enode.op {
                Language::Symbol(_) => true,
                _ => false,
            })
            .all(|x| x)
    }
}
fn dimension_greater_than(
    var: &'static str,
    axis: usize,
    greater_than: usize,
) -> impl Fn(&mut egg::EGraph<Language, Meta>, egg::Id, &egg::Subst) -> bool {
    let var = var.parse().unwrap();
    move |egraph, _, subst| {
        egraph[subst[&var]].metadata.shape.as_ref().unwrap()[axis] > greater_than
    }
}
fn dimension_is_even(
    var: &'static str,
    axis: usize,
) -> impl Fn(&mut egg::EGraph<Language, Meta>, egg::Id, &egg::Subst) -> bool {
    let var = var.parse().unwrap();
    move |egraph, _, subst| egraph[subst[&var]].metadata.shape.as_ref().unwrap()[axis] % 2 == 0
}

// TODO(gus) not sure all this should be public.
pub struct RewriteNonMatchingCartConcatApplier {
    pub a1: egg::Var,
    pub a2: egg::Var,
    pub a_axis: usize,
    pub b1: egg::Var,
    pub b2: egg::Var,
    pub b_axis: usize,
}
impl egg::Applier<Language, Meta> for RewriteNonMatchingCartConcatApplier {
    fn apply_one(
        &self,
        _egraph: &mut egg::EGraph<Language, Meta>,
        _id: egg::Id,
        _subst: &egg::Subst,
    ) -> std::vec::Vec<egg::Id> {
        // For now, just want to handle these cases.
        assert!(self.a_axis == 0 || self.a_axis == 1);
        assert!(self.b_axis == 0 || self.b_axis == 1);
        assert_ne!(self.a_axis, self.b_axis);

        // We will break up the as into smaller chunks and the bs into
        // smaller chunks, so that they all match in size.
        // The goal is to have the innermost concats be along axis 0, and
        // the outermost concats to be along axis 1. Additionally, the goal
        // is that the result should only involve cartesian products of
        // concats, where the left and right concat use the same axis.
        // Then, existing rewrites can be used to bubble the concats up
        // through the cartesian products.

        // Each a needs to be split into 4; each b needs to be split into 4.

        // First we want to construct all of the concats along the 1 axis.
        // These will become our innermost concats.
        // One of these is already concatted along the 1 axis!

        // TODO(gus) left off here, I think I should actually do something
        // simpler here and just rewrite the two concats that are the
        // children of this cartesian product.
        // It needs some information from elsewhere in the graph, though,
        // that's the tough thing.

        // So we're going to slice-and-concat all 4 tensors. We'll slice the
        // as based on the bs size, and slice the bs based on the as size.
        // TODO(gus) I could write an even simpler rewrite rule that slices
        // more indiscriminately, everywhere. Right now I'm using some
        // context clue (the overarching cartesian product) to only apply
        // this where needed.

        // All I actually want to do is to rewrite that second concat.
        //  (cartesian-product
        //   (concat ?a1 ?a2 0)
        //   (concat ?b1 ?b2 1)
        //  )
        //  (cartesian-product
        //   (concat ?a1 ?a2 0)
        //   (concat (concat (slice ?b1) (slice ?b1)  0)
        //  )
        //

        vec![]
    }
}

struct SplitApplier {
    a: egg::Var,
    axis: usize,
}
impl egg::Applier<Language, Meta> for SplitApplier {
    fn apply_one(
        &self,
        egraph: &mut egg::EGraph<Language, Meta>,
        id: egg::Id,
        _subst: &egg::Subst,
    ) -> std::vec::Vec<egg::Id> {
        let shape: ndarray::IxDyn = egraph[id].metadata.shape.as_ref().unwrap().clone();
        assert_eq!(shape[self.axis] % 2, 0);
        let low_bound = 0;
        let low_bound_id = egraph.add(egg::ENode::leaf(Language::Usize(low_bound)));
        let high_bound = shape[self.axis];
        let high_bound_id = egraph.add(egg::ENode::leaf(Language::Usize(high_bound)));
        let middle_bound = high_bound / 2;
        let middle_bound_id = egraph.add(egg::ENode::leaf(Language::Usize(middle_bound)));

        let mut slice_0_indices = std::vec::Vec::new();
        for i in 0..shape.as_array_view().len() {
            if i == self.axis {
                // If this is the axis we're splitting on, then access the
                // first half.
                slice_0_indices.push(low_bound_id);
                slice_0_indices.push(middle_bound_id);
            } else {
                // Otherwise, access the whole axis.
                slice_0_indices.push(egraph.add(egg::ENode::leaf(Language::Usize(0))));
                slice_0_indices.push(egraph.add(egg::ENode::leaf(Language::Usize(shape[i]))));
            }
        }

        let mut slice_1_indices = std::vec::Vec::new();
        for i in 0..shape.as_array_view().len() {
            if i == self.axis {
                // If this is the axis we're splitting on, then access the
                // second half.
                slice_1_indices.push(middle_bound_id);
                slice_1_indices.push(high_bound_id);
            } else {
                // Otherwise, access the whole axis.
                slice_1_indices.push(egraph.add(egg::ENode::leaf(Language::Usize(0))));
                slice_1_indices.push(egraph.add(egg::ENode::leaf(Language::Usize(shape[i]))));
            }
        }

        let mut slice_0_children = std::vec::Vec::new();
        slice_0_children.push(id);
        slice_0_children.append(&mut slice_0_indices);

        let mut slice_1_children = std::vec::Vec::new();
        slice_1_children.push(id);
        slice_1_children.append(&mut slice_1_indices);

        let slice_0_id = egraph.add(egg::ENode::new(Language::Slice, slice_0_children));
        let slice_1_id = egraph.add(egg::ENode::new(Language::Slice, slice_1_children));
        //println!("{:?}", egraph[slice_0_id]);
        //println!("{:?}", egraph[slice_1_id]);

        let axis_usize_id = egraph.add(egg::ENode::leaf(Language::Usize(self.axis)));

        // Add
        // (concat )
        let id: egg::Id = egraph.add(egg::ENode::new(
            Language::Concat,
            vec![slice_0_id, slice_1_id, axis_usize_id],
        ));
        vec![id]
    }
}

pub fn bubble_concat_through_cols_axis_1() -> Rewrite<Language, Meta> {
    egg::rewrite!("bubble-concat-through-cols-axis-1"; "(cols (concat ?a ?b 1))"
                  => "(concat (cols ?a) (cols ?b) 0)")
}

pub fn split(axis: usize, dimension_greater_than: usize) -> Rewrite<Language, Meta> {
    egg::rewrite!(format!("split-axis-{}", axis); "?a" =>
                  {SplitApplier{axis: axis, a:"?a".parse().unwrap()}}
                  if self::dimension_greater_than("?a", axis, dimension_greater_than)
                  if dimension_is_even("?a", axis)
                  if has_shape("?a"))
}

pub fn split_concat() -> Rewrite<Language, Meta> {
    egg::rewrite!("split-concat"; "?a" => {SplitConcatApplier{a:"?a".parse().unwrap()}} if has_shape("?a") if is_symbol("?a"))
}
pub fn bubble_concat_through_rows_axis_0() -> Rewrite<Language, Meta> {
    egg::rewrite!("bubble-concat-through-rows-axis-0"; "(rows (concat ?a ?b 0))"
                      => "(concat (rows ?a) (rows ?b) 0)")
}
pub fn bubble_concat_through_rows_axis_1() -> Rewrite<Language, Meta> {
    egg::rewrite!("bubble-concat-through-rows-axis-1"; "(rows (concat ?a ?b 1))"
                      => "(concat (rows ?a) (rows ?b) 1)")
}
pub fn bubble_concat_through_cols_axis_0() -> Rewrite<Language, Meta> {
    egg::rewrite!("bubble-concat-through-cols-axis-0"; "(cols (concat ?a ?b 0))"
                      => "(concat (cols ?a) (cols ?b) 1)")
}
pub fn bubble_concat_through_cartesian_product_axis_0_0() -> Rewrite<Language, Meta> {
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
    )
}
pub fn rewrite_nonmatching_cartesian_product_concat() -> Rewrite<Language, Meta> {
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
    }})
}

#[cfg(test)]
mod tests {

    // #[test]
    // fn test_split_concat() {
    //     todo!();
    // }

    // #[test]
    // fn test_split() {
    //     todo!();
    // }
}
