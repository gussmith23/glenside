use egg::define_language;
use ndarray::Dimension;

egg::define_language! {
    pub enum Language {
        Rows = "rows",
        Cols = "cols",
        CartesianProduct = "cartesian-product",
        // Map dot product:
        // for a tensor with shape
        // [a1, ..., an, 2, b],
        // the result is a new tensor with shape
        // [a1, ..., an]
        // Whose elements are the dot product of the two b-length vectors at
        // each position in the original array.
        MapDotProduct = "map-dot-product",
        BsgSystolicArray = "bsg-systolic-array-weight-stationary",
        // Slice into list/tensor/whatever we're calling them
        Slice = "slice",
        Concat = "concat",
        // TODO(gus) this will probably need to be signed at some point?
        ElementwiseAdd = "elementwise-add",
        Usize(usize),
        Symbol(String),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Meta {
    pub(super) shape: Option<ndarray::IxDyn>,
    pub(super) usize_value: Option<usize>,
}
impl egg::Metadata<Language> for Meta {
    type Error = ();

    fn merge(&self, other: &Self) -> Self {
        assert_eq!(self, other);
        self.clone()
    }

    fn make(egraph: &egg::EGraph<Language, Self>, enode: &egg::ENode<Language>) -> Self {
        // We only know the value in the case of a Num.
        use Language::*;
        match &enode.op {
            CartesianProduct => {
                // This cartesian product works a little differently from
                // before, given the new, simplified shape system.
                // It wants to pair up the very last dimension of the two
                // input arrays. I.e. it views the two input arrays as
                // having shapes
                // [a1, a2, ..., an, c]
                // [b1, b2, ..., bn, c]
                // And sees them essentially as two tensors of vectors:
                // input 1 is a [a1, ..., an] sized tensor of c-length vectors
                // similar for input 2.
                // So I think our only requirement is that the last dimension
                // is the same size. And then the resulting size is
                // [a1, ... an, b1, ..., bn, 2, c].
                // Originally I implemented it for input arrays with 2
                // dimensions. Going to try to allow for 1 dimension in both
                // tensors. Can generalize further later.
                assert_eq!(enode.children.len(), 2);
                let initial_shape_left: &ndarray::IxDyn =
                    &egraph[enode.children[0]].metadata.shape.as_ref().unwrap();
                assert!(initial_shape_left.as_array_view().len() >= 1);
                assert!(initial_shape_left.as_array_view().len() <= 2);
                let initial_shape_right: &ndarray::IxDyn =
                    &egraph[enode.children[1]].metadata.shape.as_ref().unwrap();
                assert!(initial_shape_left.as_array_view().len() >= 1);
                assert!(initial_shape_left.as_array_view().len() <= 2);
                assert_eq!(
                    initial_shape_left[initial_shape_left.as_array_view().len() - 1],
                    initial_shape_right[initial_shape_right.as_array_view().len() - 1],
                );

                // New shape is [a1, ..., an, b1, ..., bn, 2, c].
                let mut new_shape: Vec<usize> = initial_shape_left
                    .as_array_view()
                    .iter()
                    .take(initial_shape_left.as_array_view().len() - 1)
                    .copied()
                    .collect();
                new_shape.extend(
                    initial_shape_right
                        .as_array_view()
                        .iter()
                        .take(initial_shape_right.as_array_view().len() - 1),
                );
                new_shape.push(2);
                new_shape.push(initial_shape_left[initial_shape_left.as_array_view().len() - 1]);
                let new_shape: ndarray::IxDyn = ndarray::IxDyn(&new_shape[..]);
                assert_eq!(
                    new_shape.as_array_view().len(),
                    initial_shape_left.as_array_view().len() - 1
                        + initial_shape_right.as_array_view().len()
                        - 1
                        + 1
                        + 1
                );
                Meta {
                    shape: Some(new_shape),
                    usize_value: None,
                }
            }
            Rows => {
                assert_eq!(enode.children.len(), 1);
                let initial_shape: ndarray::IxDyn = egraph[enode.children[0]]
                    .metadata
                    .shape
                    .as_ref()
                    .unwrap()
                    .clone();
                // Doesn't have to be true in the future.
                assert_eq!(initial_shape.as_array_view().len(), 2);
                // Our new, simpler system makes this way easier!
                Meta {
                    shape: Some(initial_shape),
                    usize_value: None,
                }
            }
            Cols => {
                assert_eq!(enode.children.len(), 1);
                let mut initial_shape: ndarray::IxDyn = egraph[enode.children[0]]
                    .metadata
                    .shape
                    .as_ref()
                    .unwrap()
                    .clone();
                // Doesn't have to be true in the future.
                assert_eq!(initial_shape.as_array_view().len(), 2);

                // The column dimension gets moved first. For a two-dimensional
                // array, it's a transpose!
                let cols_val: usize = initial_shape[1];
                initial_shape[1] = initial_shape[0];
                initial_shape[0] = cols_val;

                Meta {
                    shape: Some(initial_shape),
                    usize_value: None,
                }
            }
            MapDotProduct => {
                assert_eq!(enode.children.len(), 1);
                let shape: &ndarray::IxDyn =
                    egraph[enode.children[0]].metadata.shape.as_ref().unwrap();

                assert!(shape.as_array_view().len() >= 3);
                assert_eq!(shape[shape.as_array_view().len() - 2], 2);

                let new_shape: ndarray::IxDyn = ndarray::IxDyn(
                    &shape
                        .as_array_view()
                        .iter()
                        .take(shape.as_array_view().len() - 2)
                        .copied()
                        .collect::<Vec<usize>>()[..],
                );

                Meta {
                    shape: Some(new_shape),
                    usize_value: None,
                }
            }
            BsgSystolicArray => panic!(),
            Slice => {
                let shape_to_be_sliced: &ndarray::IxDyn =
                    &egraph[enode.children[0]].metadata.shape.as_ref().unwrap();
                let slice_indices: std::vec::Vec<usize> = enode.children[1..]
                    .iter()
                    .map(|id| egraph[*id].metadata.usize_value.unwrap())
                    .collect();

                // For every dimension, there should be two slice indices:
                // ( [beginning, end) )
                // Note that this is a pretty restrictive syntax for now.
                assert_eq!(0, slice_indices.len() % 2);
                assert_eq!(
                    shape_to_be_sliced.as_array_view().len(),
                    slice_indices.len() / 2
                );

                let mut new_shape = shape_to_be_sliced.clone();

                for dim_i in 0..shape_to_be_sliced.as_array_view().len() {
                    let dim_val: usize = shape_to_be_sliced[dim_i];
                    let slice_start: usize = slice_indices[dim_i * 2];
                    let slice_end: usize = slice_indices[dim_i * 2 + 1];
                    use std::convert::TryInto;
                    assert!(slice_end <= dim_val.try_into().unwrap());
                    assert!(slice_start <= slice_end);
                    if slice_end - slice_start > 0 {
                        // If the slice actually needs to produce values...
                        assert!(slice_start < dim_val.try_into().unwrap());
                    }

                    new_shape[dim_i] = (slice_end - slice_start).try_into().unwrap();
                }

                Meta {
                    shape: Some(new_shape),
                    usize_value: None,
                }
            }
            Concat => {
                // Need at least two arrays and always need one axis
                assert!(enode.children.len() >= 3);
                let shapes: std::vec::Vec<&ndarray::IxDyn> = (0..(enode.children.len() - 1))
                    .map(|i| egraph[enode.children[i]].metadata.shape.as_ref().unwrap())
                    .collect();

                let concat_axis: usize = egraph[enode.children[enode.children.len() - 1]]
                    .metadata
                    .usize_value
                    .unwrap();

                assert!((0..shapes.len())
                    .all(|i| shapes[i].as_array_view().len() == shapes[0].as_array_view().len()));
                // The two shapes must be equal, except for along the concat
                // axis.
                assert!(
                    (0..shapes[0].as_array_view().len()).all(|i| i == concat_axis
                        || ((0..shapes.len()).all(|j| shapes[j][i] == shapes[0][i])))
                );

                let mut new_shape = shapes[0].clone();
                new_shape[concat_axis] += (1..shapes.len())
                    .map(|i| shapes[i][concat_axis])
                    .sum::<usize>();
                //println!("concat input shapes: {:?}", shapes);
                //println!("concat output shape: {:?}", new_shape);

                Meta {
                    shape: Some(new_shape),
                    usize_value: None,
                }
            }
            ElementwiseAdd => {
                assert!(enode.children.len() == 2);
                assert_eq!(
                    egraph[enode.children[0]].metadata.shape.as_ref().unwrap(),
                    egraph[enode.children[1]].metadata.shape.as_ref().unwrap()
                );

                Meta {
                    shape: egraph[enode.children[0]].metadata.shape.clone(),
                    usize_value: None,
                }
            }
            Usize(u) => Meta {
                shape: None,
                usize_value: Some(*u),
            },
            Symbol(name) => {
                //println!("Symbol");
                Meta {
                    shape: Some(ndarray::IxDyn(
                        &(match &name[..] {
                            "in" => vec![1, 784],
                            "w1" => vec![784, 512],
                            "w2" => vec![512, 512],
                            "w3" => vec![512, 10],
                            // TODO(gus) have to figure out a way around this.
                            // Max seems to think the tensors should just go
                            // into the egraph. I was hoping to have some kind
                            // of environment that we could wrap the egraph in
                            // (would have to be accessible from here), but Max
                            // doesn't have that nor does he plan to implement
                            // it.
                            //
                            // Update, Max is implementing something that will
                            // allow for this.
                            "single-matrix-multiply-input-a" => vec![32, 32],
                            "single-matrix-multiply-input-b" => vec![32, 32],
                            "v-32" => vec![32],
                            "t-32-32" => vec![32, 32],
                            _ => panic!("No shape defined for {}", name),
                        })[..],
                    )),
                    usize_value: None,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse() {
        "
         (map-dot-product
          (cartesian-product
           (rows single-matrix-multiply-input-a)
           (cols single-matrix-multiply-input-b)
          )
         )
         "
        .parse::<egg::RecExpr<Language>>()
        .unwrap();
    }

    #[test]
    fn test_cartesian_product_shape() {
        let program = "(cartesian-product
          v-32
          (cols t-32-32)
         )
         "
        .parse()
        .unwrap();
        let (egraph, id) = egg::EGraph::<Language, Meta>::from_expr(&program);
        assert_eq!(
            egraph[id].metadata.shape.as_ref().unwrap(),
            &ndarray::IxDyn(&[32, 2, 32])
        );

        let program = "(cartesian-product
          (rows t-32-32)
          v-32
         )
         "
        .parse()
        .unwrap();
        let (egraph, id) = egg::EGraph::<Language, Meta>::from_expr(&program);
        assert_eq!(
            egraph[id].metadata.shape.as_ref().unwrap(),
            &ndarray::IxDyn(&[32, 2, 32])
        );
    }
}
