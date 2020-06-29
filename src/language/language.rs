use egg::{define_language, merge_if_different, EGraph, Id};
use ndarray::{Dimension, IxDyn};
use std::str::FromStr;
use std::fmt::Display;

define_language! {
   pub enum Language {
        // (move-axis <tensor> <axis (usize)> <dest (usize)>)
        // Moves axis <axis> so that it is now axis <dest>.
        // Replaces the "rows" and "cols" operators.
        "move-axis" = MoveAxis([Id; 3]),

        // (cartesian-product <t0> <t1>)
        // Expects tensors of shape
        // [a1, ..., an, c]
        // [b1, ..., bm, c]
        // Outputs a tensor of shape
        // [a1, ..., an, b1, ..., bm, 2, c]
        // which represents the cartesian product of the c-length vectors stored
        // in the two tensors.
        "cartesian-product" = CartesianProduct([Id; 2]),

        // (map-dot-product <tensor>)
        // for a tensor with shape
        // [a1, ..., an, 2, c],
        // the result is a new tensor with shape
        // [a1, ..., an]
        // Whose elements are the dot product of the two c-length vectors at
        // each position in the original array.
        "map-dot-product" = MapDotProduct(Id),

        // (slice <tensor> <axis (usize)> <low (usize)> <high (usize)>)
        // Slices into <tensor> at axis <axis>, slicing the half-open range
        // [<low>, <high>).
        "slice" = Slice([Id; 4]),

        // (concatenate <t0> <t1> <axis (usize)>)
        // Concatenate tensors <t0> and <t1> along <axis>.
        "concatenate" = Concatenate([Id; 3]),


        // (elementwise-add <t0> <t1>)
        // TODO(@gussmith23) this will probably need to be signed at some point?
        // TODO(@gussmith23) ^^ what did I mean by this?
        "elementwise-add" = ElementwiseAdd([Id; 2]),

        // (bsg-systolic-array <rows (usize)> <cols (usize)> <t0> <t1>)
        // Represents a systolic array of size rows X cols, fed with tensors t0
        // and t1.
        // TODO(@gussmith23) do we need to specify rows and cols? You can infer these
        // from the size of the input, but it's also useful for searching.
        "bsg-systolic-array" = BsgSystolicArray([Id; 4]),

        Usize(usize),

        // pad-type: zero-padding
        // (No other options right now)
        PadType(PadType),
        Symbol(String),
    }
}

/// Specifies how to pick the values we pad with.
#[derive(Debug, PartialEq, Eq, Hash, Clone, PartialOrd, Ord)]
pub enum PadType {
    /// Pad with zeroes.
    ZeroPadding,
}
impl FromStr for PadType {
    type Err = ();
    fn from_str(input: &str) -> Result<PadType, Self::Err> {
        match input {
            "zero-padding" => Ok(PadType::ZeroPadding),
            _ => Err(()),
        }
    }
}
impl Display for PadType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                PadType::ZeroPadding => "zero-padding",
            }
        )
    }
}

// TODO(@gussmith23) Pick a better analysis name.
#[derive(Debug, Clone, PartialEq)]
pub struct MyAnalysisData {
    pub(crate) shape: Option<IxDyn>,
    pub(crate) usize_value: Option<usize>,
}
pub struct MyAnalysis;
impl MyAnalysis {
    pub(crate) fn get_usize(id: Id, egraph: &EGraph<Language, MyAnalysis>) -> usize {
        egraph[id].data.usize_value.unwrap()
    }
    pub(crate) fn get_shape(id: Id, egraph: &EGraph<Language, MyAnalysis>) -> &IxDyn {
        egraph[id].data.shape.as_ref().unwrap()
    }
}
impl egg::Analysis<Language> for MyAnalysis {
    type Data = MyAnalysisData;

    fn merge(&self, to: &mut Self::Data, from: Self::Data) -> bool {
        assert_eq!(*to, from);
        merge_if_different(to, from)
    }

    fn make(egraph: &EGraph<Language, Self>, enode: &Language) -> Self::Data {
        use Language::*;
        match enode {
            &MoveAxis([tensor_id, src_axis_id, dest_axis_id]) => {
                let mut new_shape = Self::get_shape(tensor_id, egraph).clone();
                let src_axis = Self::get_usize(src_axis_id, egraph);
                let dest_axis = Self::get_usize(dest_axis_id, egraph);

                assert!(src_axis < new_shape.as_array_view().len());
                assert!(dest_axis < new_shape.as_array_view().len());

                let tmp = new_shape[dest_axis];
                new_shape[dest_axis] = new_shape[src_axis];
                new_shape[src_axis] = tmp;

                Self::Data {
                    shape: Some(new_shape),
                    usize_value: None,
                }
            }
            &CartesianProduct([t0_id, t1_id]) => {
                let initial_shape_left: &IxDyn = Self::get_shape(t0_id, egraph);
                assert!(initial_shape_left.as_array_view().len() >= 1);
                assert!(initial_shape_left.as_array_view().len() <= 2);
                let initial_shape_right: &IxDyn = Self::get_shape(t1_id, egraph);
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
                Self::Data {
                    shape: Some(new_shape),
                    usize_value: None,
                }
            }
            &MapDotProduct(tensor_id) => {
                let shape: &IxDyn = Self::get_shape(tensor_id, egraph);

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

                Self::Data {
                    shape: Some(new_shape),
                    usize_value: None,
                }
            }
            &BsgSystolicArray([rows_id, cols_id, t0_id, t1_id]) => {
                egraph[rows_id].data.usize_value.unwrap();
                egraph[cols_id].data.usize_value.unwrap();
                let left_shape = Self::get_shape(t0_id, egraph);
                let right_shape = Self::get_shape(t1_id, egraph);
                let left_shape_len: usize = left_shape.as_array_view().len();
                let right_shape_len: usize = right_shape.as_array_view().len();

                // TODO(@gussmith23) check that the rows/cols params sizes are correct
                // given the input tensor shapes.

                // Assumptions I'm making right now.
                assert!(left_shape_len == 1 || left_shape_len == 2);
                assert_eq!(right_shape_len, 2);

                let new_shape: Vec<ndarray::Ix> = left_shape
                    .as_array_view()
                    .iter()
                    .cloned()
                    .take(left_shape.as_array_view().len() - 1)
                    .chain(right_shape.as_array_view().iter().cloned().skip(1))
                    .collect();

                Self::Data {
                    shape: Some(ndarray::IxDyn(&new_shape)),
                    usize_value: None,
                }
            }
            &Slice([tensor_id, axis_id, low_id, high_id]) => {
                let mut new_shape: IxDyn = Self::get_shape(tensor_id, egraph).clone();

                let axis: usize = Self::get_usize(axis_id, egraph);
                let low: usize = Self::get_usize(low_id, egraph);
                let high: usize = Self::get_usize(high_id, egraph);

                assert!(new_shape.as_array_view().len() > axis);
                assert!(low < new_shape[axis]);
                assert!(high <= new_shape[axis]);

                new_shape[axis] = high - low;

                Self::Data {
                    shape: Some(new_shape),
                    usize_value: None,
                }
            }
            &Concatenate([t0_id, t1_id, axis_id]) => {
                let axis = Self::get_usize(axis_id, egraph);
                let mut new_shape = Self::get_shape(t0_id, egraph).clone();
                let t1_shape = Self::get_shape(t1_id, egraph).clone();
                assert_eq!(
                    new_shape.as_array_view().len(),
                    t1_shape.as_array_view().len()
                );
                assert!(axis < t1_shape.as_array_view().len());
                new_shape[axis] += t1_shape[axis];

                Self::Data {
                    shape: Some(new_shape),
                    usize_value: None,
                }
            }
            &ElementwiseAdd([t0_id, t1_id]) => {
                assert_eq!(
                    Self::get_shape(t0_id, egraph),
                    Self::get_shape(t1_id, egraph)
                );

                Self::Data {
                    shape: Some(Self::get_shape(t0_id, egraph).clone()),
                    usize_value: None,
                }
            }
            Usize(u) => Self::Data {
                shape: None,
                usize_value: Some(*u),
            },
            Symbol(name) => {
                //println!("Symbol");
                Self::Data {
                    shape: Some(ndarray::IxDyn(
                        &(match &name[..] {
                            "in" => vec![1, 784],
                            "w1" => vec![784, 512],
                            "w2" => vec![512, 512],
                            "w3" => vec![512, 10],
                            // TODO(@gussmith23) have to figure out a way around this.
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
                            "t-32-64" => vec![32, 64],
                            "t-64-128" => vec![64, 128],
                            "t-128-16" => vec![128, 16],
                            _ => panic!("No shape defined for {}", name),
                        })[..],
                    )),
                    usize_value: None,
                }
            }
            PadType(_) => {
                todo!("Need to figure out how to represent pad type in metadata...I think I want to move to new egg version first...");
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
           single-matrix-multiply-input-a
           (move-axis single-matrix-multiply-input-b 1 0)
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
          (move-axis t-32-32 1 0)
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        assert_eq!(MyAnalysis::get_shape(id, &egraph), &IxDyn(&[32, 2, 32]));

        let program = "(cartesian-product
          (move-axis t-32-32 1 0)
          v-32
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        assert_eq!(MyAnalysis::get_shape(id, &egraph), &IxDyn(&[32, 2, 32]));
    }
}
