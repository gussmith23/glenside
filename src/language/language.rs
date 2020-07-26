use egg::{define_language, merge_if_different, EGraph, Id};
use itertools::multizip;
use ndarray::{s, Dimension, IxDyn};
use std::fmt::Display;
use std::str::FromStr;

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

        // (systolic-array <rows (usize)> <cols (usize)> <access-0> <access-1>)
        // Represents a systolic array of size rows X cols, fed with two
        // accesses.
        // TODO(@gussmith23) do we need to specify rows and cols? You can infer these
        // from the size of the input, but it's also useful for searching.
        "systolic-array" = SystolicArray([Id; 4]),

        // (access-windows <access> <filters-shape> <x-stride> <y-stride>)
        // Form the windows which will be convolved over.
        // TODO(@gussmith23) AccessWindows shouldn't be specific to filters.
        // AccessWindows is used in other contexts too, i.e. pooling.
        "access-windows" = AccessWindows([Id; 4]),

        // (shape-of <tensor>)
        // Returns the shape of the tensor.
        // TODO(@gussmith) Choose between ([Id; 1]) and (Id) and be consistent
        // When describing the arguments of a construct that takes a single Id
        // argument (like shape-of), we can use (Id) or ([Id; 1]). I'm not sure
        // which is better, but I should choose one and be consistent.
        "shape-of" = ShapeOf([Id; 1]),

        // (slice-shape <shape> <dim>)
        // Slices a shape by taking dimensions >= <dim>.
        "slice-shape" = SliceShape([Id; 2]),

        // (access <tensor> <dim>)
        // The most basic access pattern.
        // Let <tensor> have dims d0, .., dn.
        // Interprets <tensor> as a shaped list of shape d0, .., d(<dim>-1)
        // whose elements are of shape d<dim>, .., dn.
        "access" = Access([Id; 2]),

        // (access-move-axis <access> <axis (usize)> <dest (usize)>)
        // Move <axis> so it is now <dest>, shifting other axes as needed.
        "access-move-axis" = AccessMoveAxis([Id; 3]),

        // (access-cartesian-product <access1> <access2>)
        // Cartesian product access pattern.
        // Assume <access1> has shape
        // [a1, ..., an]
        // and <access2> has shape
        // [b1, ..., bm].
        // Both must have the same item shape,
        // [c1, ..., co]
        // Outputs a tensor of shape
        // [a1, ..., an, b1, ..., bm, 2, c1, ..., co]
        // which represents the cartesian product of the items in both accesses.
        "access-cartesian-product" = AccessCartesianProduct([Id; 2]),

        // (compute <compute-type> <access>)
        // Compute over the items in <access>.
        //
        // Compute types:
        //
        // dot-product
        // Expects an item shape of
        // [n, a0, ..., am]
        // Where n specifies the tuple multiplicity and [a0, ..., am] is the
        // shape of the tensors to be dot-producted with one another.
        "compute" = Compute([Id; 2]),

        // (get-access-shape <access>)
        // Returns the shape of the access.
        "get-access-shape" = GetAccessShape(Id),

        // (access-reshape <access> <shape>)
        // Reshapes the access to have the given
        "access-reshape" = AccessReshape([Id; 2]),

        // (access-flatten <access>)
        // Flattens the access's shape and item shape.
        "access-flatten" = AccessFlatten(Id),

        // (shape <usize>...)
        // Shape literal.
        "shape" = Shape(Box<[Id]>),

        // (access-shape <shape: shape> <item-shape: shape>)
        // Access shape literal.
        "access-shape" = AccessShape([Id;2]),

        // (access-slice <access> <axis (usize)> <low (usize)> <high (usize)>)
        // Slices into <access> at axis <axis>, slicing the half-open range
        // [<low>, <high>).
        // TODO(@gussmith23) Implement access-slice-item
        // If axis >= access.shape.ndim(), it slices into access.item_shape.
        // This is me being lazy and not wanting to implement separate
        // access-slice-shape and access-slice-item operators for right now.
        "access-slice" = AccessSlice([Id; 4]),

        // (access-concatenate <a0> <a1> <axis (usize)>)
        // Concatenate accesses <a0> and <a1> along <axis>.
        // TODO(@gussmith23) Implement access-concatenate-item
        // If axis >= access.shape.ndim(), it concatenates along dimensions in
        // access.item_shape.
        "access-concatenate" = AccessConcatenate([Id; 3]),

        // (access-pair <a0> <a1>)
        // Simply pair every item of a0 with every item of a1.
        "access-pair" = AccessPair([Id; 2]),

        // (access-shift-right <a0>)
        // Shifts a dimension from shape to item shape.
        "access-shift-right" = AccessShiftRight(Id),

        // (access-tensor <t>)
        // Access a tensor literal.
        "access-tensor" = AccessTensor(Id),

        // (access-pad <a>
        //             <pad-type (PadType)>
        //             <axis (usize)> <pad-before (usize)> <pad-after (usize)>)
        // Pads a tensor at the given axis.
        "access-pad" = AccessPad([Id; 5]),

        // (access-squeeze <a> <axis (usize)>)
        "access-squeeze" = AccessSqueeze([Id; 2]),

        // (access-insert-axis <a> <axis (usize)>)
        "access-insert-axis" = AccessInsertAxis([Id; 2]),

        Usize(usize),

        // pad-type: zero-padding
        // (No other options right now)
        PadType(PadType),

        ComputeType(ComputeType),

        Symbol(String),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ComputeType {
    DotProduct,
    ReduceSum,
    ReLU,
    /// Expects item shape of `a x b1 x .. x bn`. Performs an elementwise
    /// addition of the `a` tensors of size `b1 x .. x bn`.
    /// TODO(@gussmith) Multiple-arg compute feels clunky and ad-hoc.
    /// Should figure out an explicit way to define access multiple-stream
    /// access patterns.
    ElementwiseAdd,
    /// Expects item shape of `a x b1 x .. x bn`. Performs an elementwise
    /// multiplication of the `a` tensors of size `b1 x .. x bn`.
    ElementwiseMul,
    /// Takes the max across all elements in each item. Reduces any item shape
    /// to a scalar.
    ReduceMax,
}
impl FromStr for ComputeType {
    type Err = ();
    fn from_str(input: &str) -> Result<ComputeType, Self::Err> {
        match input {
            "dot-product" => Ok(ComputeType::DotProduct),
            "reduce-sum" => Ok(ComputeType::ReduceSum),
            "reduce-max" => Ok(ComputeType::ReduceMax),
            "relu" => Ok(ComputeType::ReLU),
            "elementwise-add" => Ok(ComputeType::ElementwiseAdd),
            "elementwise-mul" => Ok(ComputeType::ElementwiseMul),
            _ => Err(()),
        }
    }
}
impl Display for ComputeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                ComputeType::DotProduct => "dot-product",
                ComputeType::ReduceSum => "reduce-sum",
                ComputeType::ReduceMax => "reduce-max",
                ComputeType::ReLU => "relu",
                ComputeType::ElementwiseAdd => "elementwise-add",
                ComputeType::ElementwiseMul => "elementwise-mul",
            }
        )
    }
}

/// Specifies how to pick the values we pad with.
#[derive(Debug, PartialEq, Eq, Hash, Clone, PartialOrd, Ord, Copy)]
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
pub enum MyAnalysisData {
    Legacy(MyAnalysisDataLegacyData),
    AccessPattern(AccessPatternData),
    Shape(ShapeData),
    // TODO(@gussmith23) Needed?
    //Tensor(TensorData),
    ComputeType(ComputeType),
    PadType(PadType),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ShapeData {
    shape: IxDyn,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AccessPatternData {
    pub shape: IxDyn,
    pub item_shape: IxDyn,
}

// #[derive(Debug, Clone, PartialEq)]
// pub struct TensorData {
//     shape: IxDyn,
// }

// TODO(@gussmith23) Pick a better analysis name.
#[derive(Debug, Clone, PartialEq)]
pub struct MyAnalysisDataLegacyData {
    pub(crate) shape: Option<IxDyn>,
    pub(crate) usize_value: Option<usize>,
}
pub struct MyAnalysis;
impl MyAnalysis {
    pub(crate) fn get_usize(id: Id, egraph: &EGraph<Language, MyAnalysis>) -> usize {
        match &egraph[id].data {
            MyAnalysisData::Legacy(s) => s.usize_value.unwrap(),
            _ => panic!(),
        }
    }
    pub(crate) fn get_shape(id: Id, egraph: &EGraph<Language, MyAnalysis>) -> &IxDyn {
        match &egraph[id].data {
            MyAnalysisData::Legacy(s) => s.shape.as_ref().unwrap(),
            _ => panic!(),
        }
    }
    pub(crate) fn get_shape_of_value(id: Id, egraph: &EGraph<Language, MyAnalysis>) -> &IxDyn {
        match &egraph[id].data {
            MyAnalysisData::Shape(s) => &s.shape,
            _ => panic!(),
        }
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
            &AccessInsertAxis([access_id, axis_id]) => {
                let mut access = match &egraph[access_id].data {
                    MyAnalysisData::AccessPattern(a) => a.clone(),
                    _ => panic!(),
                };
                let axis = MyAnalysis::get_usize(axis_id, egraph);

                assert!(axis <= access.shape.ndim() + access.item_shape.ndim());

                if axis <= access.shape.ndim() {
                    access.shape = IxDyn(
                        access.shape.slice()[..axis]
                            .iter()
                            .cloned()
                            .chain(std::iter::once(1))
                            .chain(access.shape.slice()[axis..].iter().cloned())
                            .collect::<Vec<_>>()
                            .as_slice(),
                    );
                } else {
                    let n = access.shape.ndim();
                    access.item_shape = IxDyn(
                        access.item_shape.slice()[..axis - n]
                            .iter()
                            .cloned()
                            .chain(std::iter::once(1))
                            .chain(access.item_shape.slice()[axis - n..].iter().cloned())
                            .collect::<Vec<_>>()
                            .as_slice(),
                    );
                }

                MyAnalysisData::AccessPattern(access)
            }
            &AccessSqueeze([access_id, axis_id]) => {
                let mut access = match &egraph[access_id].data {
                    MyAnalysisData::AccessPattern(a) => a.clone(),
                    _ => panic!(),
                };
                let axis = MyAnalysis::get_usize(axis_id, egraph);
                use ndarray::RemoveAxis;
                if axis < access.shape.ndim() {
                    assert_eq!(access.shape[axis], 1);
                    access.shape = access.shape.remove_axis(ndarray::Axis(axis));
                } else {
                    assert_eq!(access.item_shape[axis - access.shape.ndim()], 1);
                    access.item_shape = access
                        .item_shape
                        .remove_axis(ndarray::Axis(axis - access.shape.ndim()));
                }

                MyAnalysisData::AccessPattern(access)
            }
            &AccessPad([access_id, pad_type_id, axis_id, pad_before_id, pad_after_id]) => {
                let mut access = match &egraph[access_id].data {
                    MyAnalysisData::AccessPattern(a) => a.clone(),
                    _ => panic!(),
                };
                let _pad_type = match &egraph[pad_type_id].data {
                    MyAnalysisData::PadType(t) => t,
                    _ => panic!(),
                };
                let axis = MyAnalysis::get_usize(axis_id, egraph);
                assert!(axis < access.shape.ndim() + access.item_shape.ndim());
                let pad_before = MyAnalysis::get_usize(pad_before_id, egraph);
                let pad_after = MyAnalysis::get_usize(pad_after_id, egraph);
                if axis < access.shape.ndim() {
                    access.shape[axis] += pad_before + pad_after;
                } else {
                    access.item_shape[axis - access.shape.ndim()] += pad_before + pad_after;
                };
                MyAnalysisData::AccessPattern(access)
            }
            &AccessTensor(t_id) => MyAnalysisData::AccessPattern(AccessPatternData {
                shape: match &egraph[t_id].data {
                    MyAnalysisData::Legacy(l) => l.shape.as_ref().unwrap().clone(),
                    _ => panic!(),
                },
                item_shape: IxDyn(&[]),
            }),
            &AccessShiftRight(a_id) => {
                let a = match &egraph[a_id].data {
                    MyAnalysisData::AccessPattern(a) => a,
                    _ => panic!(),
                };

                let combined = a
                    .shape
                    .as_array_view()
                    .iter()
                    .chain(a.item_shape.as_array_view().iter())
                    .cloned()
                    .collect::<Vec<_>>();
                MyAnalysisData::AccessPattern(AccessPatternData {
                    shape: IxDyn(&combined[..(a.shape.ndim().saturating_sub(1))]),
                    item_shape: IxDyn(&combined[(a.shape.ndim().saturating_sub(1))..]),
                })
            }
            &AccessPair([a0_id, a1_id]) => {
                let (a0, a1) = match (&egraph[a0_id].data, &egraph[a1_id].data) {
                    (MyAnalysisData::AccessPattern(a0), MyAnalysisData::AccessPattern(a1)) => {
                        (a0, a1)
                    }
                    _ => panic!(),
                };

                assert_eq!(a0.shape, a1.shape);
                assert_eq!(a0.item_shape, a1.item_shape);

                MyAnalysisData::AccessPattern(AccessPatternData {
                    shape: a0.shape.clone(),
                    item_shape: IxDyn(
                        std::iter::once(2)
                            .chain(a0.item_shape.as_array_view().iter().cloned())
                            .collect::<Vec<_>>()
                            .as_slice(),
                    ),
                })
            }
            &AccessMoveAxis([access_id, src_axis_id, dest_axis_id]) => {
                let (split_axis, shape) = match &egraph[access_id].data {
                    MyAnalysisData::AccessPattern(a) => (
                        a.shape.ndim(),
                        a.shape
                            .as_array_view()
                            .iter()
                            .chain(a.item_shape.as_array_view().iter())
                            .cloned()
                            .collect::<Vec<_>>(),
                    ),
                    _ => panic!(),
                };
                let src_axis = Self::get_usize(src_axis_id, egraph);
                let dest_axis = Self::get_usize(dest_axis_id, egraph);

                assert!(src_axis < shape.len());
                assert!(dest_axis < shape.len());

                let new_shape_order = if src_axis <= dest_axis {
                    shape[..src_axis]
                        .iter()
                        .chain(shape[src_axis + 1..=dest_axis].iter())
                        .chain(std::iter::once(&shape[src_axis]))
                        .chain(shape[dest_axis + 1..].iter())
                        .cloned()
                        .collect::<Vec<_>>()
                } else {
                    shape[..dest_axis]
                        .iter()
                        .chain(std::iter::once(&shape[src_axis]))
                        .chain(shape[dest_axis..src_axis].iter())
                        .chain(shape[src_axis + 1..].iter())
                        .cloned()
                        .collect::<Vec<_>>()
                };

                MyAnalysisData::AccessPattern(AccessPatternData {
                    shape: IxDyn(&new_shape_order[..split_axis]),
                    item_shape: IxDyn(&new_shape_order[split_axis..]),
                })
            }
            &AccessSlice([access_id, axis_id, low_id, high_id]) => {
                let mut new_access = match &egraph[access_id].data {
                    MyAnalysisData::AccessPattern(a) => a.clone(),
                    _ => panic!(),
                };

                let axis: usize = Self::get_usize(axis_id, egraph);
                let low: usize = Self::get_usize(low_id, egraph);
                let high: usize = Self::get_usize(high_id, egraph);

                assert!(new_access.shape.ndim() + new_access.item_shape.ndim() > axis);
                if axis < new_access.shape.ndim() {
                    assert!(low < new_access.shape[axis]);
                    assert!(high <= new_access.shape[axis]);
                    new_access.shape[axis] = high - low;
                } else {
                    assert!(low < new_access.item_shape[axis - new_access.shape.ndim()]);
                    assert!(high <= new_access.item_shape[axis - new_access.shape.ndim()]);
                    new_access.item_shape[axis - new_access.shape.ndim()] = high - low;
                }

                MyAnalysisData::AccessPattern(new_access)
            }
            &AccessConcatenate([a0_id, a1_id, axis_id]) => {
                let axis = Self::get_usize(axis_id, egraph);
                let mut new_access = match &egraph[a0_id].data {
                    MyAnalysisData::AccessPattern(a) => a.clone(),
                    _ => panic!(),
                };
                let a1 = match &egraph[a1_id].data {
                    MyAnalysisData::AccessPattern(a) => a,
                    _ => panic!(),
                };
                assert_eq!(new_access.shape.ndim(), a1.shape.ndim(),);
                assert_eq!(new_access.item_shape.ndim(), a1.item_shape.ndim(),);
                assert!(axis < a1.shape.ndim() + a1.item_shape.ndim());
                if axis < new_access.shape.ndim() {
                    new_access.shape[axis] += a1.shape[axis];
                } else {
                    new_access.item_shape[axis - new_access.shape.ndim()] +=
                        a1.item_shape[axis - new_access.shape.ndim()];
                }

                MyAnalysisData::AccessPattern(new_access)
            }
            &AccessShape([shape_id, item_shape_id]) => {
                MyAnalysisData::AccessPattern(AccessPatternData {
                    shape: match &egraph[shape_id].data {
                        MyAnalysisData::Shape(s) => s.shape.clone(),
                        _ => panic!(),
                    },
                    item_shape: match &egraph[item_shape_id].data {
                        MyAnalysisData::Shape(s) => s.shape.clone(),
                        _ => panic!(),
                    },
                })
            }
            Shape(list) => MyAnalysisData::Shape(ShapeData {
                shape: IxDyn(
                    list.iter()
                        .map(|id: &Id| MyAnalysis::get_usize(*id, egraph))
                        .collect::<Vec<_>>()
                        .as_slice(),
                ),
            }),
            &AccessReshape([access_id, access_shape_id]) => {
                let a = match &egraph[access_id].data {
                    MyAnalysisData::AccessPattern(a) => a,
                    _ => panic!("Expected an access as the first argument to access-reshape"),
                };
                let new_shape = match &egraph[access_shape_id].data {
                    MyAnalysisData::AccessPattern(a) => a,
                    _ => panic!(),
                };
                assert_eq!(
                    a.shape.as_array_view().iter().product::<usize>(),
                    new_shape.shape.as_array_view().iter().product::<usize>(),
                );
                assert_eq!(
                    a.item_shape.as_array_view().iter().product::<usize>(),
                    new_shape
                        .item_shape
                        .as_array_view()
                        .iter()
                        .product::<usize>(),
                );
                MyAnalysisData::AccessPattern(new_shape.clone())
            }
            &AccessFlatten(access_id) => {
                let a = match &egraph[access_id].data {
                    MyAnalysisData::AccessPattern(a) => a,
                    _ => panic!(),
                };
                MyAnalysisData::AccessPattern(AccessPatternData {
                    shape: IxDyn(&[a.shape.as_array_view().iter().product()]),
                    item_shape: IxDyn(&[a.item_shape.as_array_view().iter().product()]),
                })
            }
            &GetAccessShape(access_id) => egraph[access_id].data.clone(),
            ComputeType(t) => MyAnalysisData::ComputeType(t.clone()),
            &Compute([compute_type_id, access_id]) => {
                let compute_type = match &egraph[compute_type_id].data {
                    MyAnalysisData::ComputeType(t) => t,
                    _ => panic!("Argument 0 of {:?} should be a ComputeType", enode),
                };
                let a0 = match &egraph[access_id].data {
                    MyAnalysisData::AccessPattern(a0) => a0,
                    _ => panic!(),
                };

                match compute_type {
                    self::ComputeType::ElementwiseAdd | self::ComputeType::ElementwiseMul => {
                        assert!(a0.item_shape.ndim() >= 1);
                        MyAnalysisData::AccessPattern(AccessPatternData {
                            shape: a0.shape.clone(),
                            item_shape: IxDyn(&a0.item_shape.slice()[1..]),
                        })
                    }
                    self::ComputeType::DotProduct => {
                        // If it's =1, that's just a "dot product" of scalars,
                        // which is just a sum.
                        //
                        // Honestly, it could also be 0. It doesn't make much
                        // sense but it's not wrong. Can remove this later if we
                        // want those semantics.
                        assert!(a0.item_shape.ndim() >= 1);

                        // MyAnalysisData::Tensor(TensorData {
                        //     shape: a0.shape.clone(),
                        // })
                        MyAnalysisData::AccessPattern(AccessPatternData {
                            shape: a0.shape.clone(),
                            item_shape: IxDyn(&[]),
                        })
                    }
                    self::ComputeType::ReduceSum | self::ComputeType::ReduceMax => {
                        MyAnalysisData::AccessPattern(AccessPatternData {
                            shape: a0.shape.clone(),
                            item_shape: IxDyn(&[]),
                        })
                    }
                    self::ComputeType::ReLU => MyAnalysisData::AccessPattern(a0.clone()),
                }
            }
            &AccessCartesianProduct([a0_id, a1_id]) => {
                let (a0, a1) = match (&egraph[a0_id].data, &egraph[a1_id].data) {
                    (MyAnalysisData::AccessPattern(a0), MyAnalysisData::AccessPattern(a1)) => {
                        (a0, a1)
                    }
                    _ => panic!(),
                };
                assert_eq!(
                    a0.item_shape, a1.item_shape,
                    "Cartesian product argument shapes must match"
                );

                let new_shape = IxDyn(
                    a0.shape
                        .as_array_view()
                        .iter()
                        .cloned()
                        .chain(a1.shape.as_array_view().iter().cloned())
                        .collect::<Vec<usize>>()
                        .as_slice(),
                );
                let new_item_shape = IxDyn(
                    std::iter::once(2)
                        .chain(a0.item_shape.as_array_view().iter().cloned())
                        .collect::<Vec<usize>>()
                        .as_slice(),
                );

                assert_eq!(
                    new_shape.as_array_view().iter().product::<usize>()
                        * new_item_shape.as_array_view().iter().product::<usize>(),
                    a0.shape.as_array_view().iter().product::<usize>()
                        * a1.shape.as_array_view().iter().product::<usize>()
                        * 2
                        * a0.item_shape.as_array_view().iter().product::<usize>()
                );

                MyAnalysisData::AccessPattern(AccessPatternData {
                    shape: new_shape,
                    item_shape: new_item_shape,
                })
            }
            &SliceShape([shape_id, dim_id]) => {
                let shape = MyAnalysis::get_shape_of_value(shape_id, egraph);
                let dim = MyAnalysis::get_usize(dim_id, egraph);
                MyAnalysisData::Shape(ShapeData {
                    shape: IxDyn(shape.as_array_view().slice(s![dim..]).to_slice().unwrap()),
                })
            }
            &Access([tensor_or_access_id, dim_id]) => {
                // TODO(@gussmith23) How to access tensor literals?
                let dim = MyAnalysis::get_usize(dim_id, egraph);
                let shape = match &egraph[tensor_or_access_id].data {
                    MyAnalysisData::AccessPattern(a) => a
                        .shape
                        .as_array_view()
                        .iter()
                        .chain(a.item_shape.as_array_view().iter())
                        .cloned()
                        .collect::<Vec<_>>(),
                    _ => panic!(),
                };
                MyAnalysisData::AccessPattern(AccessPatternData {
                    shape: IxDyn(&shape[..dim]),
                    item_shape: IxDyn(&shape[dim..]),
                })
            }
            &MoveAxis([tensor_id, src_axis_id, dest_axis_id]) => {
                let mut new_shape = Self::get_shape(tensor_id, egraph).clone();
                let src_axis = Self::get_usize(src_axis_id, egraph);
                let dest_axis = Self::get_usize(dest_axis_id, egraph);

                assert!(src_axis < new_shape.as_array_view().len());
                assert!(dest_axis < new_shape.as_array_view().len());

                let tmp = new_shape[dest_axis];
                new_shape[dest_axis] = new_shape[src_axis];
                new_shape[src_axis] = tmp;

                MyAnalysisData::Legacy(MyAnalysisDataLegacyData {
                    shape: Some(new_shape),
                    usize_value: None,
                })
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
                MyAnalysisData::Legacy(MyAnalysisDataLegacyData {
                    shape: Some(new_shape),
                    usize_value: None,
                })
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

                MyAnalysisData::Legacy(MyAnalysisDataLegacyData {
                    shape: Some(new_shape),
                    usize_value: None,
                })
            }
            &BsgSystolicArray([rows_id, cols_id, t0_id, t1_id]) => {
                // Check that the rows and cols are usizes.
                let _unused = Self::get_usize(rows_id, egraph);
                let _unused = Self::get_usize(cols_id, egraph);

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

                MyAnalysisData::Legacy(MyAnalysisDataLegacyData {
                    shape: Some(ndarray::IxDyn(&new_shape)),
                    usize_value: None,
                })
            }
            &SystolicArray([rows_id, cols_id, a0_id, a1_id]) => {
                // Check that the rows and cols are usizes.
                let _unused = Self::get_usize(rows_id, egraph);
                let _unused = Self::get_usize(cols_id, egraph);

                let (a0, a1) = match (&egraph[a0_id].data, &egraph[a1_id].data) {
                    (MyAnalysisData::AccessPattern(a0), MyAnalysisData::AccessPattern(a1)) => {
                        (a0, a1)
                    }
                    _ => panic!(),
                };

                assert!(a0.shape.ndim() <= 1);
                assert_eq!(a0.item_shape.ndim(), 1);
                assert_eq!(a1.shape.ndim(), 1);
                assert_eq!(a1.item_shape.ndim(), 1);

                MyAnalysisData::AccessPattern(AccessPatternData {
                    shape: IxDyn(
                        a0.shape
                            .as_array_view()
                            .iter()
                            .chain(a1.shape.as_array_view().iter())
                            .cloned()
                            .collect::<Vec<_>>()
                            .as_slice(),
                    ),
                    item_shape: IxDyn(&[]),
                })
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

                MyAnalysisData::Legacy(MyAnalysisDataLegacyData {
                    shape: Some(new_shape),
                    usize_value: None,
                })
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

                MyAnalysisData::Legacy(MyAnalysisDataLegacyData {
                    shape: Some(new_shape),
                    usize_value: None,
                })
            }
            &ElementwiseAdd([t0_id, t1_id]) => {
                assert_eq!(
                    Self::get_shape(t0_id, egraph),
                    Self::get_shape(t1_id, egraph)
                );

                MyAnalysisData::Legacy(MyAnalysisDataLegacyData {
                    shape: Some(Self::get_shape(t0_id, egraph).clone()),
                    usize_value: None,
                })
            }
            Usize(u) => MyAnalysisData::Legacy(MyAnalysisDataLegacyData {
                shape: None,
                usize_value: Some(*u),
            }),
            Symbol(name) => {
                //println!("Symbol");
                MyAnalysisData::Legacy(MyAnalysisDataLegacyData {
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
                            // A 3-channel "image" in CHW format.
                            "t-3-32-32" => vec![3, 32, 32],
                            // An OIHW set of convolution filters.
                            "t-8-3-3-3" => vec![8, 3, 3, 3],
                            "t-1024-2-256" => vec![1024, 2, 256],
                            "t-1-2-3-4" => vec![1, 2, 3, 4],
                            _ => panic!("No shape defined for {}", name),
                        })[..],
                    )),
                    usize_value: None,
                })
            }
            PadType(t) => MyAnalysisData::PadType(*t),
            // TODO(@gussmith23) should take access, not a tensor.
            &AccessWindows([access_id, filters_shape_id, x_stride_id, y_stride_id]) => {
                let x_stride = MyAnalysis::get_usize(x_stride_id, egraph);
                let y_stride = MyAnalysis::get_usize(y_stride_id, egraph);
                let access = match &egraph[access_id].data {
                    MyAnalysisData::AccessPattern(a) => a,
                    _ => {
                        panic!("Expected an access pattern as the first argument to access-windows")
                    }
                };
                let filters_shape = MyAnalysis::get_shape_of_value(filters_shape_id, egraph);

                // TODO(@gussmith23) Figure out how to generalize access-windows
                // Should be able to generalize to other shapes.
                // TODO(@gussmith23) Add batch dimension
                // TODO(@gussmith23) Change layout to NHWC
                // This is totally hard coded right now for our # of dims AND
                // for our layout.
                // Scott needs batch dim and NHWC.
                // Expect it to be (access <tensor> 3).
                assert_eq!(access.shape.ndim(), 3);
                assert_eq!(access.item_shape.ndim(), 0);
                assert_eq!(filters_shape.ndim(), 3);

                let new_shape: Vec<usize> = multizip((
                    // channels, rows, cols dimensions of tensor shape
                    access.shape.slice().iter(),
                    // channels, rows, cols dimensions of filter shape
                    filters_shape.slice().iter(),
                    // TODO(@gussmith23) channels stride hardcoded to 1
                    &[1, x_stride, y_stride],
                ))
                .map(
                    |(&dim_len, &kernel_dim_len, &stride): (&usize, &usize, &usize)| {
                        let total_dim_len = dim_len;
                        assert!(total_dim_len >= kernel_dim_len);
                        let num_spots = total_dim_len - (kernel_dim_len - 1);
                        (num_spots + stride - 1) / stride
                    },
                )
                .collect();

                MyAnalysisData::AccessPattern(AccessPatternData {
                    shape: IxDyn(new_shape.clone().as_slice()),
                    item_shape: filters_shape.clone(),
                })
            }

            &ShapeOf([tensor_id]) => MyAnalysisData::Shape(ShapeData {
                shape: MyAnalysis::get_shape(tensor_id, egraph).clone(),
            }),
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

    #[test]
    fn access_windows() {
        // TODO(@gussmith23) Could probably clean this up with a for loop
        // Would make it easier to add more tests.

        let program = "
         (access-windows (access (access-tensor t-3-32-32) 3) (slice-shape (shape-of t-8-3-3-3) 1) 1 1)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1, 30, 30]));
                assert_eq!(a.item_shape, IxDyn(&[3, 3, 3]));
            }
            _ => panic!(),
        }

        let program = "
         (access-windows (access (access-tensor t-3-32-32) 3) (slice-shape (shape-of t-8-3-3-3) 1) 2 1)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1, 15, 30]));
                assert_eq!(a.item_shape, IxDyn(&[3, 3, 3]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn shape_of() {
        let program = "
         (shape-of t-3-32-32)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        assert_eq!(
            MyAnalysis::get_shape_of_value(id, &egraph),
            &IxDyn(&[3, 32, 32])
        );
    }

    #[test]
    fn access() {
        let program = "
         (access (access-tensor t-3-32-32) 0)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[3, 32, 32]));
            }
            _ => panic!(),
        }

        let program = "
         (access (access-tensor t-3-32-32) 2)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3, 32]));
                assert_eq!(a.item_shape, IxDyn(&[32]));
            }
            _ => panic!(),
        }

        let program = "
         (access (access-tensor t-3-32-32) 3)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3, 32, 32]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn reaccess() {
        let program = "
         (access (access (access-tensor t-3-32-32) 3) 0)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[3, 32, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    #[should_panic]
    fn access_invalid() {
        let program = "
         (access (access-tensor t-3-32-32) 4)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        egraph.add_expr(&program);
    }

    #[test]
    fn slice_shape() {
        let program = "
         (slice-shape (shape-of t-3-32-32) 2)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        assert_eq!(MyAnalysis::get_shape_of_value(id, &egraph), &IxDyn(&[32]));

        let program = "
         (slice-shape (shape-of t-3-32-32) 0)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        assert_eq!(
            MyAnalysis::get_shape_of_value(id, &egraph),
            &IxDyn(&[3, 32, 32])
        );
    }

    #[test]
    #[should_panic]
    fn slice_shape_invalid_slice() {
        let program = "
         (slice-shape (shape-of t-3-32-32) 10)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        assert_eq!(MyAnalysis::get_shape_of_value(id, &egraph), &IxDyn(&[]));
    }

    #[test]
    fn access_cartesian_product() {
        let program = "
         (access-cartesian-product
          (access (access-tensor v-32) 0)
          (access (access-tensor t-32-32) 1)
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32]));
                assert_eq!(a.item_shape, IxDyn(&[2, 32]));
            }
            _ => panic!(),
        }

        let program = "
         (access-cartesian-product
          (access (access-tensor t-32-32) 1)
          (access (access-tensor v-32) 0)
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32]));
                assert_eq!(a.item_shape, IxDyn(&[2, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_dot_product() {
        let program = "
         (compute dot-product (access (access-tensor t-3-32-32) 0))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }

        let program = "
         (compute dot-product (access (access-tensor t-3-32-32) 1))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }

        let program = "
         (compute dot-product (access (access-tensor t-3-32-32) 2))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3, 32]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    // This may not panic in the future, if we allow dot products over empty
    // tuples.
    #[should_panic]
    #[test]
    fn compute_dot_product_panic() {
        let program = "
         (compute dot-product (access (access-tensor t-3-32-32) 3))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3, 32]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn conv2d() {
        // The following TVM/Python code will compute the correct sizes of
        // cov2ds.
        //
        // import tvm
        // from tvm import relay
        //
        // mod = relay.module.Module.from_expr(
        //     relay.nn.conv2d(relay.var('x', shape=[1, 3, 32, 32]),
        //                     relay.var('weight', shape=[8, 3, 3, 3])))
        //
        // print(mod)

        let program = "
         (compute dot-product
          (access-cartesian-product
           (access (access-tensor t-8-3-3-3) 1)
           (access-squeeze
            (access-windows
             (access (access-tensor t-3-32-32) 3)
             (slice-shape (shape-of t-8-3-3-3) 1)
             1
             1
            )
            0
           )
          )
         )
        "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);

        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[8, 30, 30]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }

        let program = "
         (compute dot-product
          (access-cartesian-product
           (access (access-tensor t-8-3-3-3) 1)
           (access-squeeze
            (access-windows
             (access (access-tensor t-3-32-32) 3)
             (slice-shape (shape-of t-8-3-3-3) 1)
             1
             2
            )
            0
           )
          )
         )
        "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);

        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[8, 30, 15]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn flatten_reshape() {
        let program = "
         (access-reshape
          (access-flatten (access (access-tensor t-3-32-32) 2))
          (access-shape (shape 32 3) (shape 16 2))
         )
        "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);

        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32, 3]));
                assert_eq!(a.item_shape, IxDyn(&[16, 2]));
            }
            _ => panic!(),
        }
    }

    #[should_panic]
    #[test]
    fn flatten_reshape_panic() {
        let program = "
         (access-reshape
          (access-flatten (access (access-tensor t-3-32-32) 2))
          (access-shape (shape 1) (shape 16 2))
         )
        "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);

        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32, 3]));
                assert_eq!(a.item_shape, IxDyn(&[16, 2]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_slice_0() {
        let program = "(access-slice (access (access-tensor t-3-32-32) 1) 0 0 1)"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1]));
                assert_eq!(a.item_shape, IxDyn(&[32, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_slice_1() {
        let program = "(access-slice (access (access-tensor t-3-32-32) 1) 1 16 32)"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3]));
                assert_eq!(a.item_shape, IxDyn(&[16, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_slice_2() {
        let program = "(access-slice (access (access-tensor t-3-32-32) 2) 2 16 32)"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3, 32]));
                assert_eq!(a.item_shape, IxDyn(&[16]));
            }
            _ => panic!(),
        }
    }

    #[test]
    #[should_panic]
    fn access_slice_panic() {
        let program = "(access-slice (access (access-tensor t-3-32-32) 1) 3 16 32)"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3]));
                assert_eq!(a.item_shape, IxDyn(&[16, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_concatenate_0() {
        let program = "(access-concatenate (access (access-tensor t-3-32-32) 1) (access (access-tensor t-3-32-32) 1) 0)"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[6]));
                assert_eq!(a.item_shape, IxDyn(&[32, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_concatenate_1() {
        let program = "(access-concatenate (access (access-tensor t-3-32-32) 1) (access (access-tensor t-3-32-32) 1) 2)"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3]));
                assert_eq!(a.item_shape, IxDyn(&[32, 64]));
            }
            _ => panic!(),
        }
    }

    #[should_panic]
    #[test]
    fn access_concatenate_panic_0() {
        let program = "(access-concatenate (access (access-tensor t-3-32-32) 1) (access (access-tensor t-3-32-32) 1) 3)"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3]));
                assert_eq!(a.item_shape, IxDyn(&[32, 64]));
            }
            _ => panic!(),
        }
    }

    #[should_panic]
    #[test]
    fn access_concatenate_panic_1() {
        let program = "(access-concatenate (access (access-tensor t-3-32-32) 1) (access (access-tensor t-8-3-3-3) 1) 2)"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3]));
                assert_eq!(a.item_shape, IxDyn(&[32, 64]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_move_axis_0() {
        let program = "(access-move-axis (access (access-tensor t-3-32-32) 1) 0 2)"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32]));
                assert_eq!(a.item_shape, IxDyn(&[32, 3]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_move_axis_1() {
        let program = "(access-move-axis (access (access-tensor t-3-32-32) 1) 1 0)"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32]));
                assert_eq!(a.item_shape, IxDyn(&[3, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_move_axis_2() {
        let program = "(access-move-axis (access (access-tensor t-3-32-32) 1) 1 1)"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3]));
                assert_eq!(a.item_shape, IxDyn(&[32, 32]));
            }
            _ => panic!(),
        }
    }

    #[should_panic]
    #[test]
    fn access_move_axis_panic_0() {
        let program = "(access-move-axis (access (access-tensor t-3-32-32) 1) 3 1)"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3]));
                assert_eq!(a.item_shape, IxDyn(&[32, 32]));
            }
            _ => panic!(),
        }
    }

    #[should_panic]
    #[test]
    fn access_move_axis_panic_1() {
        let program = "(access-move-axis (access (access-tensor t-3-32-32) 1) 1 3)"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3]));
                assert_eq!(a.item_shape, IxDyn(&[32, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_reduce_sum_0() {
        let program = "
         (compute reduce-sum (access (access-tensor t-3-32-32) 0))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_reduce_sum_1() {
        let program = "
         (compute reduce-sum (access (access-tensor t-3-32-32) 2))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3, 32]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_pair() {
        let program = "
         (access-pair (access (access-tensor t-32-32) 1) (access (access-tensor t-32-32) 1))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32]));
                assert_eq!(a.item_shape, IxDyn(&[2, 32]));
            }
            _ => panic!(),
        }
    }

    #[should_panic]
    #[test]
    fn access_pair_panic() {
        let program = "
         (access-pair (access (access-tensor t-32-32) 0) (access (access-tensor t-32-32) 1))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32]));
                assert_eq!(a.item_shape, IxDyn(&[2, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_shift_right_0() {
        let program = "
         (access-shift-right (access (access-tensor t-32-32) 2))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32]));
                assert_eq!(a.item_shape, IxDyn(&[32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_shift_right_1() {
        let program = "
         (access-shift-right (access (access-tensor t-32-32) 1))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[32, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_shift_right_2() {
        let program = "
         (access-shift-right (access (access-tensor t-32-32) 0))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[32, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_relu() {
        let program = "
         (compute relu (access (access-tensor t-32-32) 0))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[32, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_elementwise_add_0() {
        let program = "
         (compute elementwise-add (access (access-tensor t-3-32-32) 0))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[32, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_elementwise_add_1() {
        let program = "
         (compute elementwise-add (access (access-tensor t-3-32-32) 1))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3]));
                assert_eq!(a.item_shape, IxDyn(&[32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_elementwise_add_2() {
        let program = "
         (compute elementwise-add (access (access-tensor t-3-32-32) 2))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3, 32]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[should_panic]
    #[test]
    fn compute_elementwise_add_panic() {
        let program = "
         (compute elementwise-add (access (access-tensor t-3-32-32) 3))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        egraph.add_expr(&program);
    }

    #[test]
    fn compute_elementwise_mul_0() {
        let program = "
         (compute elementwise-mul (access (access-tensor t-3-32-32) 0))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[32, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_elementwise_mul_1() {
        let program = "
         (compute elementwise-mul (access (access-tensor t-3-32-32) 1))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3]));
                assert_eq!(a.item_shape, IxDyn(&[32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_elementwise_mul_2() {
        let program = "
         (compute elementwise-mul (access (access-tensor t-3-32-32) 2))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3, 32]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[should_panic]
    #[test]
    fn compute_elementwise_mul_panic() {
        let program = "
         (compute elementwise-mul (access (access-tensor t-3-32-32) 3))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        egraph.add_expr(&program);
    }

    #[test]
    fn zero_padding() {
        let program = "zero-padding".parse().unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::PadType(PadType::ZeroPadding) => (),
            _ => panic!(),
        };
    }

    #[test]
    fn access_pad_0() {
        let program = "
         (access-pad (access (access-tensor t-32-32) 1) zero-padding 0 1 2)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[35]));
                assert_eq!(a.item_shape, IxDyn(&[32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_pad_1() {
        let program = "
         (access-pad (access (access-tensor t-32-32) 1) zero-padding 1 3 2)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32]));
                assert_eq!(a.item_shape, IxDyn(&[37]));
            }
            _ => panic!(),
        }
    }

    #[test]
    #[should_panic]
    fn access_pad_panic() {
        let program = "
         (access-pad (access (access-tensor t-32-32) 1) zero-padding 2 3 2)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        egraph.add_expr(&program);
    }

    #[test]
    fn compute_reduce_max_0() {
        let program = "
         (compute reduce-max (access (access-tensor t-3-32-32) 0))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_reduce_max_1() {
        let program = "
         (compute reduce-max (access (access-tensor t-3-32-32) 2))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3, 32]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_squeeze() {
        let program = "
         (access-squeeze (access (access-tensor t-1-2-3-4) 1) 0)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[2, 3, 4]));
            }
            _ => panic!(),
        }
    }

    #[test]
    #[should_panic]
    fn access_squeeze_panic() {
        let program = "
         (access-squeeze (access (access-tensor t-1-2-3-4) 1) 2)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        egraph.add_expr(&program);
    }

    #[test]
    fn access_insert_axis() {
        let program = "
         (access-insert-axis (access (access-tensor t-1-2-3-4) 1) 0)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1, 1]));
                assert_eq!(a.item_shape, IxDyn(&[2, 3, 4]));
            }
            _ => panic!(),
        }
    }

    #[test]
    #[should_panic]
    // TODO(@gussmith) More access-insert-axis tests
    fn access_insert_axis_panic() {
        let program = "
         (access-squeeze (access (access-tensor t-1-2-3-4) 1) 5)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis);
        egraph.add_expr(&program);
    }
}
