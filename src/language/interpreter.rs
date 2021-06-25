use super::language::{ComputeType, Language, PadType};
use egg::{Id, RecExpr};
use itertools::Itertools;
use ndarray::{s, Array, ArrayD, Dimension, IxDyn, Zip};
use num_traits::cast::AsPrimitive;
use num_traits::Pow;
use std::collections::hash_map::HashMap;
use std::iter::FromIterator;
use std::ops::Div;
use std::str::FromStr;

pub enum Value<DataType> {
    Tensor(ArrayD<DataType>),
    Access(Access<DataType>),
    Usize(usize),
    Shape(IxDyn),
    ComputeType(ComputeType),
    PadType(PadType),
    AccessShape(IxDyn, usize),
    List(Vec<usize>),
}

pub struct Access<DataType> {
    pub tensor: ArrayD<DataType>,
    pub access_axis: usize,
}

pub type Environment<'a, DataType> = HashMap<&'a str, ArrayD<DataType>>;

/// Simple wrapper over [`interpret`].
///
/// This was created for the web demo. Specifically, this lets us avoid having
/// the web demo depend on egg directly, thus avoiding versioning mismatches
/// between glenside's egg and the web demo's egg. There may be a much better
/// way to handle that, though.
///
/// ```
/// use glenside::language::interpreter::interpret_from_str;
/// use glenside::language::interpreter::Value;
/// use std::collections::HashMap;
/// use ndarray::Dimension;
///
/// match interpret_from_str::<i64>("(access-shape (shape 1 2) (shape 3 4))", &HashMap::default()) {
///     Value::AccessShape(shape, access_axis) => {
///         assert_eq!(shape.slice(), &[1, 2, 3, 4]);
///         assert_eq!(access_axis, 2);
///     }
///     _ => panic!(),
/// }
/// ```
pub fn interpret_from_str<DataType: 'static>(
    program: &str,
    env: &Environment<DataType>,
) -> Value<DataType>
where
    DataType: Copy
        + std::ops::Mul<Output = DataType>
        + std::ops::Div<Output = DataType>
        + std::ops::Neg<Output = DataType>
        + std::iter::Sum
        + num_traits::identities::One
        + num_traits::identities::Zero
        + std::cmp::PartialOrd
        + num_traits::Bounded
        + Exp
        + Sqrt
        + FromNotNanFloat64Literal
        + ndarray::ScalarOperand,
    usize: num_traits::cast::AsPrimitive<DataType>,
{
    let expr = RecExpr::<Language>::from_str(program).unwrap();

    interpret(&expr, expr.as_ref().len() - 1, env)
}

// TODO(@gussmith23) Interpreter stack overflows on large programs
// If I want to interpret something like a full resnet, then I will have to
// figure out a way around the stack overflows.
/// Interpret a Glenside expression
///
/// Generally, `DataType` can be inferred from the environment passed in. If
/// your expression doesn't actually use any tensor values, and the environment
/// is empty, you can choose an arbitrary type: e.g. `interpret::<i64>(...)`:
///
/// ```
/// use egg::RecExpr;
/// use glenside::language::Language;
/// use glenside::language::interpreter::interpret;
/// use glenside::language::interpreter::Value;
/// use std::str::FromStr;
/// use std::collections::HashMap;
/// use ndarray::Dimension;
///
/// let expr = RecExpr::<Language>::from_str("(access-shape (shape 1 2) (shape 3 4))").unwrap();
/// match interpret::<i64>(&expr, expr.as_ref().len() - 1, &HashMap::default()) {
///     Value::AccessShape(shape, access_axis) => {
///         assert_eq!(shape.slice(), &[1, 2, 3, 4]);
///         assert_eq!(access_axis, 2);
///     }
///     _ => panic!(),
/// }
/// ```
pub fn interpret<DataType: 'static>(
    expr: &RecExpr<Language>,
    index: usize,
    env: &Environment<DataType>,
) -> Value<DataType>
where
    DataType: Copy
        + std::ops::Mul<Output = DataType>
        + std::ops::Div<Output = DataType>
        + std::ops::Neg<Output = DataType>
        + std::iter::Sum
        + num_traits::identities::One
        + num_traits::identities::Zero
        + std::cmp::PartialOrd
        + num_traits::Bounded
        + Exp
        + Sqrt
        + FromNotNanFloat64Literal
        + ndarray::ScalarOperand,
    usize: num_traits::cast::AsPrimitive<DataType>,
{
    match &expr.as_ref()[index] {
        &Language::SystolicArrayConv2dIm2colNchwOihwWithBlocking(_) => todo!(),
        &Language::SystolicArrayConv2dIm2colNhwcHwioWithBlocking(_) => todo!(),
        &Language::SystolicArrayConv2dNchwOihwWithBlocking(_) => todo!(),
        &Language::SystolicArrayConv2dNhwcHwioWithBlocking(_) => todo!(),
        &Language::RelayOperatorCall(_) => todo!(),
        &Language::RelayOperator(_) => todo!(),
        &Language::RelayActivationLayout(_) => todo!(),
        &Language::RelayKernelLayout(_) => todo!(),
        &Language::ConstructTuple(_) => todo!(),
        &Language::TupleGetItem(_) => todo!(),
        &Language::AccessShape([shape_id, item_shape_id]) => {
            let shape = match interpret(expr, shape_id.into(), env) {
                Value::Shape(s) => s,
                _ => panic!(),
            };
            let item_shape = match interpret(expr, item_shape_id.into(), env) {
                Value::Shape(s) => s,
                _ => panic!(),
            };
            Value::AccessShape(
                IxDyn(
                    shape
                        .slice()
                        .iter()
                        .chain(item_shape.slice().iter())
                        .cloned()
                        .collect::<Vec<_>>()
                        .as_slice(),
                ),
                shape.ndim(),
            )
        }
        &Language::AccessSlice([access_id, axis_id, low_id, high_id]) => {
            let mut access = match interpret(expr, access_id.into(), env) {
                Value::Access(a) => a,
                _ => panic!(),
            };
            let axis = match interpret(expr, axis_id.into(), env) {
                Value::Usize(u) => u,
                _ => panic!(),
            };
            let low = match interpret(expr, low_id.into(), env) {
                Value::Usize(u) => u,
                _ => panic!(),
            };
            let high = match interpret(expr, high_id.into(), env) {
                Value::Usize(u) => u,
                _ => panic!(),
            };

            let mut slice_info: Vec<ndarray::SliceOrIndex> =
                std::iter::repeat(ndarray::SliceOrIndex::from(..))
                    .take(access.tensor.ndim())
                    .collect();
            slice_info[axis] = ndarray::SliceOrIndex::from(low..high);
            let slice_info = ndarray::SliceInfo::new(slice_info).unwrap();
            access.tensor = access
                .tensor
                .into_owned()
                .slice(slice_info.as_ref())
                .into_owned();

            Value::Access(access)
        }
        &Language::AccessConcatenate([a_id, b_id, axis_id]) => {
            let a = match interpret(expr, a_id.into(), env) {
                Value::Access(a) => a,
                _ => panic!(),
            };
            let b = match interpret(expr, b_id.into(), env) {
                Value::Access(a) => a,
                _ => panic!(),
            };
            let axis = match interpret(expr, axis_id.into(), env) {
                Value::Usize(u) => u,
                _ => panic!(),
            };

            assert_eq!(a.access_axis, b.access_axis);

            Value::Access(Access {
                tensor: ndarray::stack![ndarray::Axis(axis), a.tensor, b.tensor].into_dyn(),
                access_axis: a.access_axis,
            })
        }
        &Language::AccessLiteral(id) => match interpret(expr, id.into(), env) {
            Value::Tensor(t) => Value::Access(Access {
                tensor: t,
                access_axis: 0,
            }),
            _ => panic!(),
        },
        &Language::Literal(id) => match interpret(expr, id.into(), env) {
            t @ Value::Tensor(_) => t,
            _ => panic!(),
        },
        &Language::NotNanFloat64(v) => Value::Tensor(
            ndarray::arr0(DataType::from_not_nan_float_64_literal(v.into())).into_dyn(),
        ),
        &Language::AccessFlatten(access_id) => {
            let mut access = match interpret(expr, access_id.into(), env) {
                Value::Access(a) => a,
                _ => panic!(),
            };

            let shape = if access.access_axis <= 0 || access.access_axis >= access.tensor.ndim() {
                vec![access.tensor.shape().iter().product::<usize>()]
            } else {
                vec![
                    access.tensor.shape()[..access.access_axis]
                        .iter()
                        .product::<usize>(),
                    access.tensor.shape()[access.access_axis..]
                        .iter()
                        .product::<usize>(),
                ]
            };

            access.tensor = access.tensor.into_shape(shape).unwrap().into_dyn();

            Value::Access(access)
        }
        &Language::AccessTranspose([access_id, list_id]) => {
            let mut access = match interpret(expr, access_id.into(), env) {
                Value::Access(a) => a,
                _ => panic!(),
            };
            let list = match interpret(expr, list_id.into(), env) {
                Value::List(l) => l,
                _ => panic!(),
            };

            access.tensor = access.tensor.permuted_axes(list);
            Value::Access(access)
        }
        Language::List(list) => Value::List(
            list.iter()
                .map(|id: &Id| match interpret(expr, (*id).into(), env) {
                    Value::Usize(u) => u,
                    _ => panic!(),
                })
                .collect::<Vec<_>>(),
        ),
        &Language::AccessBroadcast([access_id, shape_id]) => {
            let mut access = match interpret(expr, access_id.into(), env) {
                Value::Access(a) => a,
                _ => panic!(),
            };
            let shape = match interpret(expr, shape_id.into(), env) {
                Value::AccessShape(s, _) => s,
                _ => panic!("Expected access shape as second argument to access-broadcast"),
            };

            assert_eq!(access.tensor.ndim(), shape.ndim());
            for (broadcast_from_dim, broadcast_to_dim) in
                access.tensor.shape().iter().zip(shape.slice().iter())
            {
                assert!(*broadcast_from_dim == 1 || broadcast_from_dim == broadcast_to_dim);
            }

            access.tensor = access.tensor.broadcast(shape).unwrap().to_owned();

            Value::Access(access)
        }
        &Language::AccessInsertAxis([access_id, axis_id]) => {
            let mut access = match interpret(expr, access_id.into(), env) {
                Value::Access(a) => a,
                _ => panic!(),
            };
            let axis = match interpret(expr, axis_id.into(), env) {
                Value::Usize(u) => u,
                _ => panic!(),
            };

            assert!(axis <= access.tensor.ndim());

            access.tensor = access.tensor.insert_axis(ndarray::Axis(axis));
            if axis <= access.access_axis {
                access.access_axis += 1;
            }

            Value::Access(access)
        }
        &Language::AccessPair([a0_id, a1_id]) => {
            let (a0, a1) = match (
                interpret(expr, a0_id.into(), env),
                interpret(expr, a1_id.into(), env),
            ) {
                (Value::Access(a0), Value::Access(a1)) => (a0, a1),
                _ => panic!("Expected both arguments to access-pair to be accesses"),
            };

            assert_eq!(a0.tensor.shape(), a1.tensor.shape());
            // TODO(@gussmith23) Trying out some new syntax...
            let access_axis = {
                assert_eq!(
                    a0.access_axis, a1.access_axis,
                    "Expected access axes to match in access-pair"
                );
                a0.access_axis
            };

            let tensor = ndarray::stack(
                ndarray::Axis(access_axis),
                &[
                    a0.tensor.insert_axis(ndarray::Axis(access_axis)).view(),
                    a1.tensor.insert_axis(ndarray::Axis(access_axis)).view(),
                ],
            )
            .unwrap();

            Value::Access(Access {
                tensor,
                access_axis,
            })
        }
        &Language::AccessSqueeze([access_id, axis_id]) => {
            let mut access = match interpret(expr, access_id.into(), env) {
                Value::Access(a) => a,
                _ => panic!(),
            };
            let axis = match interpret(expr, axis_id.into(), env) {
                Value::Usize(u) => u,
                _ => panic!(),
            };

            assert_eq!(
                access.tensor.shape()[axis],
                1,
                "Cannot squeeze an axis which is not equal to 1"
            );

            access.tensor = access.tensor.index_axis_move(ndarray::Axis(axis), 0);
            if axis < access.access_axis {
                access.access_axis -= 1;
            }

            Value::Access(access)
        }
        Language::PadType(t) => Value::PadType(*t),
        &Language::AccessPad([access_id, pad_type_id, axis_id, pad_before_id, pad_after_id]) => {
            let access = match interpret(expr, access_id.into(), env) {
                Value::Access(a) => a,
                _ => panic!(),
            };
            let pad_type = match interpret(expr, pad_type_id.into(), env) {
                Value::PadType(t) => t,
                _ => panic!(),
            };
            let axis = match interpret(expr, axis_id.into(), env) {
                Value::Usize(u) => u,
                _ => panic!(),
            };
            let pad_before = match interpret(expr, pad_before_id.into(), env) {
                Value::Usize(u) => u,
                _ => panic!(),
            };
            let pad_after = match interpret(expr, pad_after_id.into(), env) {
                Value::Usize(u) => u,
                _ => panic!(),
            };

            let mut before_shape = access.tensor.shape().to_vec();
            before_shape[axis] = pad_before;
            let mut after_shape = access.tensor.shape().to_vec();
            after_shape[axis] = pad_after;

            Value::Access(Access {
                tensor: ndarray::stack(
                    ndarray::Axis(axis),
                    &[
                        // TODO(@gussmith) What's going on here...
                        ndarray::ArrayD::from_elem(
                            before_shape,
                            match &pad_type {
                                PadType::ZeroPadding => DataType::zero(),
                                PadType::MinPadding => DataType::min_value(),
                            },
                        )
                        .to_owned()
                        .view(),
                        access.tensor.clone().view(),
                        ndarray::ArrayD::from_elem(
                            after_shape,
                            match &pad_type {
                                PadType::ZeroPadding => DataType::zero(),
                                PadType::MinPadding => DataType::min_value(),
                            },
                        )
                        .to_owned()
                        .view(),
                    ],
                )
                .unwrap(),
                access_axis: access.access_axis,
            })
        }
        Language::ComputeType(t) => Value::ComputeType(t.clone()),
        &Language::Compute([compute_type_id, access_id]) => {
            let compute_type = match interpret(expr, compute_type_id.into(), env) {
                Value::ComputeType(t) => t,
                _ => panic!(),
            };
            let access = match interpret(expr, access_id.into(), env) {
                Value::Access(a) => a,
                _ => panic!(),
            };

            match compute_type {
                ComputeType::ReduceMean => Value::Access(Access {
                    tensor: access
                        .tensor
                        .clone()
                        .into_shape(
                            access.tensor.shape()[..access.access_axis]
                                .iter()
                                .cloned()
                                .chain(std::iter::once(
                                    access.tensor.shape()[access.access_axis..]
                                        .iter()
                                        .cloned()
                                        .product(),
                                ))
                                .collect::<Vec<_>>()
                                .as_slice(),
                        )
                        .unwrap()
                        .sum_axis(ndarray::Axis(access.access_axis))
                        .div(
                            access.tensor.shape()[access.access_axis..]
                                .iter()
                                .product::<usize>()
                                .as_(),
                        ),
                    access_axis: access.access_axis,
                }),
                ComputeType::Softmax => {
                    assert_eq!(
                        access.access_axis,
                        access.tensor.ndim() - 1,
                        "Softmax over any axis other than the last is not implemented",
                    );

                    let shape = access.tensor.shape();
                    let mut exps = ndarray::Zip::from(&access.tensor).apply_collect(|v| v.exp());
                    let denominators = exps
                        .sum_axis(ndarray::Axis(access.tensor.ndim() - 1))
                        .insert_axis(ndarray::Axis(access.tensor.ndim() - 1));
                    ndarray::Zip::from(&mut exps)
                        .and(&denominators.broadcast(shape).unwrap())
                        .apply(|v, denom| *v = *v / *denom);

                    Value::Access(Access {
                        access_axis: access.access_axis,
                        tensor: exps,
                    })
                }
                ComputeType::ElementwiseDiv => Value::Access(Access {
                    access_axis: access.access_axis,
                    tensor: access
                        .tensor
                        .axis_iter(ndarray::Axis(access.access_axis))
                        // This is like a hacky version of fold_first. Struggled
                        // to use fold_first because it will expect the function
                        // to return an ArrayView, which is not what you want
                        // (or not even possible?)
                        .skip(1)
                        .fold(
                            // TODO(@gussmith23) More efficient way to do this?
                            access
                                .tensor
                                .axis_iter(ndarray::Axis(access.access_axis))
                                .next()
                                .expect("Cannot divide 0 arguments")
                                .into_owned(),
                            |acc, t| acc / t,
                        ),
                }),
                ComputeType::ElementwiseMul => Value::Access(Access {
                    access_axis: access.access_axis,
                    tensor: access
                        .tensor
                        .axis_iter(ndarray::Axis(access.access_axis))
                        .fold(
                            ndarray::ArrayBase::ones(
                                access.tensor.shape()[..access.access_axis]
                                    .iter()
                                    .cloned()
                                    .chain(
                                        access.tensor.shape()[access.access_axis + 1..]
                                            .iter()
                                            .cloned(),
                                    )
                                    .collect::<Vec<_>>()
                                    .as_slice(),
                            ),
                            |acc, t| acc * t,
                        ),
                }),
                ComputeType::ElementwiseAdd => Value::Access(Access {
                    access_axis: access.access_axis,
                    tensor: access
                        .tensor
                        .axis_iter(ndarray::Axis(access.access_axis))
                        .fold(
                            ndarray::ArrayBase::zeros(
                                access.tensor.shape()[..access.access_axis]
                                    .iter()
                                    .cloned()
                                    .chain(
                                        access.tensor.shape()[access.access_axis + 1..]
                                            .iter()
                                            .cloned(),
                                    )
                                    .collect::<Vec<_>>()
                                    .as_slice(),
                            ),
                            |acc, t| acc + t,
                        ),
                }),
                ComputeType::DotProduct => {
                    let reshaped = access
                        .tensor
                        .clone()
                        .into_shape(
                            std::iter::once(
                                access.tensor.shape()[..access.access_axis]
                                    .iter()
                                    .cloned()
                                    .product(),
                            )
                            .chain(access.tensor.shape()[access.access_axis..].iter().cloned())
                            .collect::<Vec<_>>(),
                        )
                        .unwrap();

                    let num_elements_per_vec: usize = access.tensor.shape()
                        [access.access_axis + 1..]
                        .iter()
                        .product();

                    let result = ndarray::arr1(
                        reshaped
                            .axis_iter(ndarray::Axis(0))
                            .map(|t| {
                                t.axis_iter(ndarray::Axis(0))
                                    .fold(
                                        ndarray::ArrayBase::ones([num_elements_per_vec]),
                                        |acc, vec| {
                                            let reshaped = vec
                                                .clone()
                                                .into_shape([num_elements_per_vec])
                                                .unwrap();

                                            ndarray::arr1(
                                                reshaped
                                                    .axis_iter(ndarray::Axis(0))
                                                    .zip(acc.axis_iter(ndarray::Axis(0)))
                                                    .map(|(a, b)| {
                                                        *a.into_scalar() * *b.into_scalar()
                                                    })
                                                    .collect::<Vec<_>>()
                                                    .as_slice(),
                                            )
                                        },
                                    )
                                    .sum()
                            })
                            .collect::<Vec<_>>()
                            .as_slice(),
                    );

                    let reshaped = result
                        .into_shape(&access.tensor.shape()[..access.access_axis])
                        .unwrap();

                    Value::Access(Access {
                        access_axis: reshaped.ndim(),
                        tensor: reshaped,
                    })
                }
                ComputeType::Negative => Value::Access(Access {
                    tensor: access.tensor.mapv(|v| v.neg()),
                    access_axis: access.access_axis,
                }),
                ComputeType::Sqrt => Value::Access(Access {
                    tensor: access.tensor.mapv(|v| v.sqrt()),
                    access_axis: access.access_axis,
                }),
                ComputeType::ReLU => Value::Access(Access {
                    tensor: access.tensor.mapv(|v| {
                        if v >= DataType::zero() {
                            v
                        } else {
                            DataType::zero()
                        }
                    }),
                    access_axis: access.access_axis,
                }),
                ComputeType::ReduceSum => Value::Access(Access {
                    tensor: access
                        .tensor
                        .clone()
                        .into_shape(
                            access.tensor.shape()[..access.access_axis]
                                .iter()
                                .cloned()
                                .chain(std::iter::once(
                                    access.tensor.shape()[access.access_axis..]
                                        .iter()
                                        .cloned()
                                        .product(),
                                ))
                                .collect::<Vec<_>>()
                                .as_slice(),
                        )
                        .unwrap()
                        .sum_axis(ndarray::Axis(access.access_axis)),
                    access_axis: access.access_axis,
                }),
                ComputeType::ReduceMax => Value::Access(Access {
                    tensor: access
                        .tensor
                        .clone()
                        .into_shape(
                            access.tensor.shape()[..access.access_axis]
                                .iter()
                                .cloned()
                                .chain(std::iter::once(
                                    access.tensor.shape()[access.access_axis..]
                                        .iter()
                                        .cloned()
                                        .product(),
                                ))
                                .collect::<Vec<_>>()
                                .as_slice(),
                        )
                        .unwrap()
                        .map_axis(ndarray::Axis(access.access_axis), |t| {
                            t.iter().fold(
                                DataType::min_value(),
                                |acc, v| if *v > acc { *v } else { acc },
                            )
                        }),
                    access_axis: access.access_axis,
                }),
            }
        }
        &Language::AccessCartesianProduct([a0_id, a1_id]) => {
            let (a0, a1) = match (
                interpret(expr, a0_id.into(), env),
                interpret(expr, a1_id.into(), env),
            ) {
                (Value::Access(a0), Value::Access(a1)) => (a0, a1),
                _ => panic!(),
            };

            assert_eq!(
                a0.tensor.shape()[a0.access_axis..],
                a1.tensor.shape()[a1.access_axis..],
                "Expected item shapes to match"
            );

            let reshaped_0 = a0
                .tensor
                .as_standard_layout()
                .into_shape(
                    std::iter::once(
                        a0.tensor.shape()[..a0.access_axis]
                            .iter()
                            .cloned()
                            .product(),
                    )
                    .chain(a0.tensor.shape()[a0.access_axis..].iter().cloned())
                    .collect::<Vec<_>>(),
                )
                .unwrap();
            let reshaped_1 = a1
                .tensor
                .clone()
                .into_shape(
                    std::iter::once(
                        a1.tensor.shape()[..a1.access_axis]
                            .iter()
                            .cloned()
                            .product(),
                    )
                    .chain(a1.tensor.shape()[a1.access_axis..].iter().cloned())
                    .collect::<Vec<_>>(),
                )
                .unwrap();

            let to_stack = reshaped_0
                .axis_iter(ndarray::Axis(0))
                .cartesian_product(reshaped_1.axis_iter(ndarray::Axis(0)))
                .map(|(t0, t1)| {
                    ndarray::stack(
                        ndarray::Axis(0),
                        &[
                            t0.insert_axis(ndarray::Axis(0)),
                            t1.insert_axis(ndarray::Axis(0)),
                        ],
                    )
                    .unwrap()
                    .insert_axis(ndarray::Axis(0))
                })
                .collect::<Vec<_>>();

            let unreshaped = ndarray::stack(
                ndarray::Axis(0),
                to_stack
                    .iter()
                    .map(|t| t.view())
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
            .unwrap();

            let reshaped = unreshaped
                .into_shape(
                    a0.tensor.shape()[..a0.access_axis]
                        .iter()
                        .cloned()
                        .chain(a1.tensor.shape()[..a1.access_axis].iter().cloned())
                        .chain(std::iter::once(2))
                        .chain(a0.tensor.shape()[a0.access_axis..].iter().cloned())
                        .collect::<Vec<_>>(),
                )
                .unwrap();

            Value::Access(Access {
                tensor: reshaped.into_dyn(),
                access_axis: a0.access_axis + a1.access_axis,
            })
        }
        &Language::Access([access_id, dim_id]) => {
            let access = match interpret(expr, access_id.into(), env) {
                Value::Access(a) => a,
                _ => panic!(),
            };
            let dim = match interpret(expr, dim_id.into(), env) {
                Value::Usize(u) => u,
                _ => panic!(),
            };

            assert!(dim <= access.tensor.ndim());

            Value::Access(Access {
                tensor: access.tensor,
                // TODO(@gussmith) Settle on vocab: "axis" or "dimension"?
                access_axis: dim,
            })
        }
        &Language::AccessWindows([access_id, filters_shape_id, stride_shape_id]) => {
            let access = match interpret(expr, access_id.into(), env) {
                Value::Access(a) => a,
                _ => panic!(),
            };
            let filters_shape = match interpret(expr, filters_shape_id.into(), env) {
                Value::Shape(s) => s,
                _ => panic!(),
            };
            let stride_shape = match interpret(expr, stride_shape_id.into(), env) {
                Value::Shape(s) => s,
                _ => panic!(),
            };

            assert_eq!(
                access.access_axis,
                access.tensor.ndim(),
                "access-windows access should be accessed at its last dimension"
            );
            assert_eq!(
                access.tensor.ndim(),
                stride_shape.ndim(),
                "access-windows access ndims should match stride ndims"
            );
            assert_eq!(
                filters_shape.ndim(),
                stride_shape.ndim(),
                "access-windows filters ndims should match stride ndims"
            );

            let out_shape = super::access_windows_resulting_shape(
                &IxDyn(access.tensor.shape()),
                &filters_shape,
                // Ignore striding for now; we will stride after we get the result
                // TODO(@gussmith23) More efficient striding?
                &IxDyn(
                    std::iter::repeat(1)
                        .take(filters_shape.ndim())
                        .collect::<Vec<_>>()
                        .as_slice(),
                ),
            );

            let mut result = ArrayD::<DataType>::zeros(
                out_shape
                    .iter()
                    .cloned()
                    .chain(std::iter::once(filters_shape.slice().iter().product()))
                    .collect::<Vec<_>>(),
            );

            Zip::from(result.genrows_mut())
                .and(access.tensor.windows(filters_shape.clone()))
                .apply(|mut result, windows| {
                    result.assign(&Array::from_iter(windows.iter().cloned()))
                });

            let mut result = result
                .into_shape(
                    out_shape
                        .iter()
                        .cloned()
                        .chain(filters_shape.slice().iter().cloned())
                        .collect::<Vec<_>>(),
                )
                .unwrap();

            for (axis, &stride) in (0..stride_shape.ndim()).zip(stride_shape.slice().iter()) {
                // TODO(@gussmith23) Interpreter window striding is inefficient
                // I wish we could do this in the windows() method of ndarray,
                // but they don't seem to support it.
                result = ndarray::stack(
                    ndarray::Axis(axis),
                    result
                        .axis_iter(ndarray::Axis(axis))
                        .step_by(stride)
                        // TODO(@gussmith23) This seems dumb
                        // Can we do an axis_iter that doesn't remove the axis?
                        .map(|t| t.insert_axis(ndarray::Axis(axis)))
                        .collect::<Vec<_>>()
                        .as_slice(),
                )
                .unwrap();
            }

            Value::Access(Access {
                tensor: result,
                // TODO(@gussmith23) Hardcoded
                // This already bit me. I forgot to update it when I changed the
                // access-windows semantics, and it took me a bit to find the
                // bug.
                // Actually, at this point, I'm pretty sure this is just wrong.
                access_axis: 3,
            })
        }
        Language::Shape(list) => Value::Shape(IxDyn(
            list.iter()
                .map(|id: &Id| match interpret(expr, (*id).into(), env) {
                    Value::Usize(u) => u,
                    _ => panic!(),
                })
                .collect::<Vec<_>>()
                .as_slice(),
        )),
        &Language::SliceShape([shape_id, slice_axis_id]) => match (
            interpret(expr, shape_id.into(), env),
            interpret(expr, slice_axis_id.into(), env),
        ) {
            (Value::Shape(s), Value::Usize(u)) => {
                Value::Shape(IxDyn(s.as_array_view().slice(s![u..]).to_slice().unwrap()))
            }
            _ => panic!(),
        },
        &Language::ShapeInsertAxis([shape_id, axis_id]) => match (
            interpret(expr, shape_id.into(), env),
            interpret(expr, axis_id.into(), env),
        ) {
            (Value::Shape(s), Value::Usize(u)) => {
                assert!(u <= s.ndim());
                Value::Shape(IxDyn(
                    s.slice()[..u]
                        .iter()
                        .chain(std::iter::once(&1))
                        .chain(s.slice()[u..].iter())
                        .cloned()
                        .collect::<Vec<_>>()
                        .as_slice(),
                ))
            }
            _ => panic!(),
        },
        &Language::ShapeRemoveAxis([shape_id, axis_id]) => match (
            interpret(expr, shape_id.into(), env),
            interpret(expr, axis_id.into(), env),
        ) {
            (Value::Shape(s), Value::Usize(u)) => {
                assert!(u < s.ndim(), "Invalid axis in shape-remove-axis");
                Value::Shape(IxDyn(
                    s.slice()[..u]
                        .iter()
                        .chain(s.slice()[u + 1..].iter())
                        .cloned()
                        .collect::<Vec<_>>()
                        .as_slice(),
                ))
            }
            _ => panic!(),
        },
        &Language::ShapeOf([tensor_id]) => match interpret(expr, tensor_id.into(), env) {
            Value::Tensor(t) => Value::Shape(IxDyn(t.shape())),
            _ => panic!(),
        },
        &Language::AccessTensor(tensor_id) => match interpret(expr, tensor_id.into(), env) {
            Value::Tensor(t) => Value::Access(Access {
                tensor: t,
                // TODO(@gussmith) Arbitrarily picked default access axis
                access_axis: 0,
            }),
            _ => panic!(),
        },
        Language::Symbol(s) => Value::Tensor(
            env.get(s.as_str())
                .unwrap_or_else(|| panic!("Symbol {} not in environment", s))
                .clone(),
        ),
        &Language::Usize(u) => Value::Usize(u),

        &Language::MoveAxis(_)
        | &Language::CartesianProduct(_)
        | &Language::MapDotProduct(_)
        | &Language::Slice(_)
        | &Language::Concatenate(_)
        | &Language::ElementwiseAdd(_)
        | &Language::BsgSystolicArray(_)
        | &Language::SystolicArray(_)
        | &Language::SystolicArrayWithBlocking(_)
        | &Language::AccessReshape(_)
        | &Language::AccessShiftRight(_) => todo!("{:?}", &expr.as_ref()[index]),
    }
}

/// Trait for types which can be converted to from Glenside literals.
pub trait FromNotNanFloat64Literal {
    /// Convert from ordered_float::NotNan<f64>
    fn from_not_nan_float_64_literal(value: ordered_float::NotNan<f64>) -> Self;
}

impl FromNotNanFloat64Literal for f64 {
    /// ```
    /// use glenside::language::interpreter::FromNotNanFloat64Literal;
    /// assert_eq!(
    ///     f64::from_not_nan_float_64_literal(
    ///         ordered_float::NotNan::new(std::f64::consts::PI).unwrap()
    ///     ),
    ///     std::f64::consts::PI
    /// );
    /// ```
    fn from_not_nan_float_64_literal(value: ordered_float::NotNan<f64>) -> Self {
        value.into_inner()
    }
}

impl FromNotNanFloat64Literal for f32 {
    /// ```
    /// use glenside::language::interpreter::FromNotNanFloat64Literal;
    /// assert_eq!(
    ///     f32::from_not_nan_float_64_literal(
    ///         ordered_float::NotNan::new(std::f64::consts::PI).unwrap()
    ///     ),
    ///     std::f64::consts::PI as f32
    /// );
    /// ```
    fn from_not_nan_float_64_literal(value: ordered_float::NotNan<f64>) -> Self {
        value.into_inner() as f32
    }
}

impl FromNotNanFloat64Literal for i64 {
    /// ```should_panic
    /// use glenside::language::interpreter::FromNotNanFloat64Literal;
    /// i64::from_not_nan_float_64_literal(
    ///     ordered_float::NotNan::new(std::f64::consts::PI).unwrap(),
    /// );
    /// ```
    fn from_not_nan_float_64_literal(_value: ordered_float::NotNan<f64>) -> Self {
        unreachable!()
    }
}

/// Trait for types which implement the exponential function.
pub trait Exp {
    /// Calculate exponential function
    fn exp(self) -> Self;
}

impl Exp for f64 {
    /// ```
    /// use glenside::language::interpreter::Exp;
    /// assert_eq!(1.234f64.exp(), 3.43494186080076);
    /// ```
    fn exp(self) -> Self {
        Pow::pow(std::f64::consts::E, self)
    }
}

impl Exp for f32 {
    /// ```
    /// use glenside::language::interpreter::Exp;
    /// assert_eq!(1.234f32.exp(), 3.4349418);
    /// ```
    fn exp(self) -> Self {
        Pow::pow(std::f32::consts::E, self)
    }
}

impl Exp for i64 {
    /// ```should_panic
    /// use glenside::language::interpreter::Exp;
    /// 0i64.exp();
    /// ```
    fn exp(self) -> Self {
        unreachable!()
    }
}

/// Trait for types which implement square root.
/// TODO(@gussmith23) Does this already exist somewhere?
pub trait Sqrt {
    /// Calculate square root.
    fn sqrt(self) -> Self;
}

impl Sqrt for f64 {
    /// ```
    /// use glenside::language::interpreter::Sqrt;
    /// assert_eq!(1.234f64.sqrt(), 1.1108555261599053);
    /// ```
    fn sqrt(self) -> Self {
        self.sqrt()
    }
}

impl Sqrt for f32 {
    /// ```
    /// use glenside::language::interpreter::Sqrt;
    /// assert_eq!(1.234f32.sqrt(), 1.1108555);
    /// ```
    fn sqrt(self) -> Self {
        self.sqrt()
    }
}

impl Sqrt for i64 {
    /// ```should_panic
    /// use glenside::language::interpreter::Sqrt;
    /// 5i64.sqrt();
    /// ```
    fn sqrt(self) -> Self {
        panic!()
    }
}

extern crate test;

#[cfg(test)]
mod tests {

    use super::*;
    use approx::AbsDiffEq;
    use ndarray::array;
    use std::str::FromStr;
    use test::Bencher;

    /// Creates a benchmark test for the interpreter
    /// The test does the following:
    ///  1. Parses $glenside_str as glenside expression
    ///  2. Creates a new Environment from the vector of (key, value) pairs if
    ///     present
    ///  3. Calls the interpreter with the glenside expression and env, and
    ///         passes the value to check_correct
    /// $test_name: the name of the created benchmark test
    /// $glenside_str: A string containing the Glenside program
    /// $env: An optional vector of 2-tuples of key, value pairs to put
    /// into the environment. If your environment is empty, do not pass
    /// an enviroment argument, otherwise you will get a compile time
    /// error.
    /// $check_correct: A closure with arguments (value) that checks for correctness
    /// $(#[$meta:meta]): Optional: attributes to add to test.
    macro_rules! benchmark_test {
        ($(#[$meta:meta])* $test_name: ident, $bench_name:ident, $glenside_str: expr, $env: expr, $check_correct: expr) => {
            $(#[$meta])*
            #[bench]
            fn $bench_name(b: &mut Bencher) {
                let mut env = Environment::new();
                for (key, value) in $env.into_iter() {
                    env.insert(key, value);
                }

                let expr = RecExpr::<Language>::from_str($glenside_str).unwrap();

                b.iter(|| {
                    // use black box to prevent compiler optimizations
                    let expr = test::black_box(&expr);
                    let env = test::black_box(&env);

                    interpret(&expr, expr.as_ref().len() - 1, &env);
                });
            }

            $(#[$meta])*
            #[test]
            fn $test_name() {
                let mut env = Environment::new();
                for (key, value) in $env.into_iter() {
                    env.insert(key, value);
                }

                let expr = RecExpr::<Language>::from_str($glenside_str).unwrap();

                let value = interpret(&expr, expr.as_ref().len() - 1, &env);
                $check_correct(value);
            }
        };
        ($(#[$meta:meta])* $test_name: ident, $bench_name:ident, $glenside_str: expr, $check_correct: expr) => {
            benchmark_test!($(#[$meta])* $test_name, $bench_name, $glenside_str, Vec::<(&str, ArrayD<f32>)>::new(), $check_correct);
        };
    }

    benchmark_test!(
        compute_elementwise_add_0,
        bench_compute_elementwise_add_0,
        "(compute elementwise-add
        (access (access-tensor t) 0)
        )",
        vec![(
            "t",
            array![[[1, -2], [3, 0]], [[-5, 6], [0, 8]], [[-9, 10], [11, 12]],].into_dyn(),
        )],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(access_axis, 0);
                    assert_eq!(
                        tensor,
                        array![[1 + -5 + -9, -2 + 6 + 10], [3 + 0 + 11, 0 + 8 + 12]].into_dyn()
                    );
                }
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        compute_elementwise_mul_0,
        bench_compute_elementwise_mul_0,
        "(compute elementwise-mul
        (access (access-tensor t) 0)
        )",
        vec![(
            "t",
            array![[[1, -2], [3, 0]], [[-5, 6], [0, 8]], [[-9, 10], [11, 12]],].into_dyn(),
        )],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(access_axis, 0);
                    assert_eq!(
                        tensor,
                        array![[1 * -5 * -9, -2 * 6 * 10], [3 * 0 * 11, 0 * 8 * 12]].into_dyn()
                    );
                }
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        compute_reduce_sum_0,
        bench_compute_reduce_sum_0,
        "(compute reduce-sum
        (access (access-tensor t) 0)
        )",
        vec![(
            "t",
            array![[[1, -2], [3, 0]], [[-5, 6], [0, 8]], [[-9, 10], [11, 12]],].into_dyn(),
        )],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(access_axis, 0);
                    assert_eq!(
                        tensor,
                        ndarray::arr0(1 + -2 + 3 + 0 + -5 + 6 + 0 + 8 + -9 + 10 + 11 + 12)
                            .into_dyn()
                    );
                }
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        compute_reduce_sum_1,
        bench_compute_reduce_sum_1,
        "(compute reduce-sum
        (access (access-tensor t) 1)
        )",
        vec![(
            "t",
            array![[[1, -2], [3, 0]], [[-5, 6], [0, 8]], [[-9, 10], [11, 12]],].into_dyn(),
        )],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(access_axis, 1);
                    assert_eq!(
                        tensor,
                        array![1 + -2 + 3 + 0, -5 + 6 + 0 + 8, -9 + 10 + 11 + 12].into_dyn()
                    );
                }
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        compute_reduce_sum_2,
        bench_compute_reduce_sum_2,
        "(compute reduce-sum
        (access (access-tensor t) 2)
        )",
        vec![(
            "t",
            array![[[1, -2], [3, 0]], [[-5, 6], [0, 8]], [[-9, 10], [11, 12]],].into_dyn(),
        )],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(access_axis, 2);
                    assert_eq!(
                        tensor,
                        array![[1 + -2, 3 + 0], [-5 + 6, 0 + 8], [-9 + 10, 11 + 12]].into_dyn()
                    );
                }
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        compute_reduce_sum_3,
        bench_compute_reduce_sum_3,
        "(compute reduce-sum
        (access (access-tensor t) 3)
        )",
        vec![(
            "t",
            array![[[1, -2], [3, 0]], [[-5, 6], [0, 8]], [[-9, 10], [11, 12]],].into_dyn(),
        )],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(access_axis, 3);
                    assert_eq!(
                        tensor,
                        array![[[1, -2], [3, 0]], [[-5, 6], [0, 8]], [[-9, 10], [11, 12]],]
                            .into_dyn(),
                    );
                }
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        compute_relu_0,
        bench_compute_relu_0,
        "(compute relu
        (access (access-tensor t) 0)
        )",
        vec![(
            "t",
            array![[[1, -2], [3, 0]], [[-5, 6], [0, 8]], [[-9, 10], [11, 12]],].into_dyn(),
        )],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(access_axis, 0);
                    assert_eq!(
                        tensor,
                        array![[[1, 0], [3, 0]], [[0, 6], [0, 8]], [[0, 10], [11, 12]],].into_dyn(),
                    );
                }
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        compute_relu_1,
        bench_compute_relu_1,
        "(compute relu
        (access (access-tensor t) 2)
        )",
        vec![(
            "t",
            array![[[1, -2], [3, 0]], [[-5, 6], [0, 8]], [[-9, 10], [11, 12]],].into_dyn(),
        )],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(access_axis, 2);
                    assert_eq!(
                        tensor,
                        array![[[1, 0], [3, 0]], [[0, 6], [0, 8]], [[0, 10], [11, 12]],].into_dyn(),
                    );
                }
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        compute_dot_product_0,
        bench_compute_dot_product_0,
        "(compute dot-product
        (access (access-tensor t) 0)
        )",
        vec![(
            "t",
            // 3 x 2 x 2
            array![[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]],].into_dyn(),
        )],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(tensor.shape(), &[] as &[usize]);
                    assert_eq!(access_axis, 0);
                    assert_eq!(
                        tensor,
                        ndarray::arr0(1 * 5 * 9 + 2 * 6 * 10 + 3 * 7 * 11 + 4 * 8 * 12).into_dyn()
                    );
                }
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        compute_dot_product_1,
        bench_compute_dot_product_1,
        "(compute dot-product
        (access (access-tensor t) 1)
        )",
        vec![(
            "t",
            // 3 x 2 x 2
            array![[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]],].into_dyn(),
        )],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(tensor.shape(), &[3]);
                    assert_eq!(access_axis, 1);
                    assert_eq!(
                        tensor,
                        array![11, 5 * 7 + 8 * 6, 9 * 11 + 10 * 12].into_dyn()
                    );
                }
                _ => panic!(),
            }
        }
    );
    benchmark_test!(
        compute_dot_product_2,
        bench_compute_dot_product_2,
        "(compute dot-product
            (access (access-tensor t) 2)
           )",
        vec![(
            "t",
            // 3 x 2 x 2
            array![[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]],].into_dyn(),
        )],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(tensor.shape(), &[3, 2]);
                    assert_eq!(access_axis, 2);
                    assert_eq!(
                        tensor,
                        array![[1 * 2, 3 * 4], [5 * 6, 7 * 8], [9 * 10, 11 * 12]].into_dyn()
                    );
                }
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        access_cartesian_product,
        bench_access_cartesian_product,
        "(access-cartesian-product
        (access (access-tensor t0) 2)
        (access (access-tensor t1) 2)
        )",
        vec![
            (
                "t0",
                // 3 x 2 x 2
                array![[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]],].into_dyn(),
            ),
            (
                "t1",
                // 2 x 2 x 2
                array![[[13, 14], [15, 16]], [[17, 18], [19, 20]]].into_dyn(),
            )
        ],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(tensor.shape(), &[3, 2, 2, 2, 2, 2]);
                    assert_eq!(access_axis, 4);
                    assert_eq!(
                        tensor.slice(s![0, 0, 0, 0, .., ..]),
                        array![[1, 2], [13, 14]]
                    );
                    assert_eq!(
                        tensor.slice(s![2, 0, 1, 0, .., ..]),
                        array![[9, 10], [17, 18]]
                    );
                }
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        access,
        bench_access,
        "(access (access-tensor t) 1)",
        vec![("t", array![[1., 2.], [3., 4.]].into_dyn())],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(tensor, array![[1., 2.], [3., 4.]].into_dyn());
                    assert_eq!(access_axis, 1);
                }
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        #[should_panic]
        access_panic,
        bench_access_panic,
        "access (access-tensor t) 3)",
        vec![("t", array![[1., 2.], [3., 4.]].into_dyn())],
        |value| value
    );

    benchmark_test!(
        access_windows,
        bench_access_windows,
        "(access-windows
            (access (access-tensor t) 3)
            (shape 3 2 2)
            (shape 1 1 1)
        )",
        vec![(
            "t",
            array![
                [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                [[10., 11., 12.], [13., 14., 15.], [16., 17., 18.]],
                [[19., 20., 21.], [22., 23., 24.], [25., 26., 27.]],
            ]
            .into_dyn()
        )],
        |value| {
            match value {
                Value::Access(a) => {
                    assert_eq!(a.access_axis, 3);
                    assert_eq!(a.tensor.shape(), &[1, 2, 2, 3, 2, 2]);
                    assert_eq!(
                        a.tensor.slice(s![0, 0, 0, .., .., ..]),
                        array![
                            [[1., 2.], [4., 5.]],
                            [[10., 11.], [13., 14.]],
                            [[19., 20.], [22., 23.]],
                        ]
                    );
                    assert_eq!(
                        a.tensor.slice(s![0, 1, 0, .., .., ..]),
                        array![
                            [[4., 5.], [7., 8.]],
                            [[13., 14.], [16., 17.]],
                            [[22., 23.], [25., 26.]],
                        ]
                    );
                }
                _ => panic!(),
            }
        }
    );

    benchmark_test!(shape, bench_shape, "(shape 1 2 3)", |value| {
        match value {
            Value::Shape(s) => assert_eq!(s, IxDyn(&[1, 2, 3])),
            _ => panic!(),
        }
    });

    benchmark_test!(
        slice_shape_0,
        bench_slice_shape_0,
        "(slice-shape (shape-of t) 0)",
        vec![("t", array![[1., 2.], [3., 4.]].into_dyn())],
        |value| {
            match value {
                Value::Shape(s) => assert_eq!(s, IxDyn(&[2, 2])),
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        slice_shape_1,
        bench_slice_shape_1,
        "(slice-shape (shape-of t) 1)",
        vec![("t", array![[1., 2.], [3., 4.]].into_dyn())],
        |value| {
            match value {
                Value::Shape(s) => assert_eq!(s, IxDyn(&[2])),
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        slice_shape_2,
        bench_slice_shape_2,
        "(slice-shape (shape-of t) 2)",
        vec![("t", array![[1., 2.], [3., 4.]].into_dyn())],
        |value| {
            match value {
                Value::Shape(s) => assert_eq!(s, IxDyn(&[])),
                _ => panic!(),
            }
        }
    );
    benchmark_test!(
        shape_insert_axis_0,
        bench_shape_insert_axis_0,
        "(shape-insert-axis (shape 2 3) 0)",
        |value| {
            match value {
                Value::Shape(s) => assert_eq!(s, IxDyn(&[1, 2, 3])),
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        shape_insert_axis_1,
        bench_shape_insert_axis_1,
        "(shape-insert-axis (shape 2 3) 1)",
        |value| {
            match value {
                Value::Shape(s) => assert_eq!(s, IxDyn(&[2, 1, 3])),
                _ => panic!(),
            }
        }
    );
    benchmark_test!(
        shape_insert_axis_2,
        bench_shape_insert_axis_2,
        "(shape-insert-axis (shape 2 3) 2)",
        |value| {
            match value {
                Value::Shape(s) => assert_eq!(s, IxDyn(&[2, 3, 1])),
                _ => panic!(),
            }
        }
    );
    benchmark_test!(
        #[should_panic]
        shape_insert_axis_panic,
        bench_shape_insert_axis_panic,
        "(shape-insert-axis (shape 2 3) 3)",
        |value| { value }
    );
    benchmark_test!(
        shape_remove_axis_0,
        bench_shape_remove_axis_0,
        "(shape-remove-axis (shape 1 2 3) 0)",
        |value| {
            match value {
                Value::Shape(s) => assert_eq!(s, IxDyn(&[2, 3])),
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        shape_remove_axis_1,
        bench_shape_remove_axis_1,
        "(shape-remove-axis (shape 1 2 3) 1)",
        |value| {
            match value {
                Value::Shape(s) => assert_eq!(s, IxDyn(&[1, 3])),
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        shape_remove_axis_2,
        bench_shape_remove_axis_2,
        "(shape-remove-axis (shape 1 2 3) 2)",
        |value| {
            match value {
                Value::Shape(s) => assert_eq!(s, IxDyn(&[1, 2])),
                _ => panic!(),
            }
        }
    );
    benchmark_test!(
        #[should_panic]
        shape_remove_axis_panic,
        bench_shape_remove_axis_panic,
        "(shape-remove-axis (shape 1 2 3) 3)",
        |value| { value }
    );

    benchmark_test!(
        shape_of,
        bench_shape_of,
        "(shape-of t)",
        vec![("t", array![[1., 2.], [3., 4.]].into_dyn())],
        |value| {
            match value {
                Value::Shape(s) => assert_eq!(s, IxDyn(&[2, 2])),
                _ => panic!(),
            }
        }
    );

    benchmark_test!(usize, bench_usize, "23", |value| {
        match value {
            Value::Usize(23) => (),
            _ => panic!(),
        }
    });
    benchmark_test!(
        symbol,
        bench_symbol,
        "t",
        vec![("t", array![[1., 2.], [3., 4.]].into_dyn())],
        |value| {
            match value {
                Value::Tensor(t) => assert_eq!(t, array![[1., 2.], [3., 4.]].into_dyn()),
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        access_tensor,
        bench_access_tensor,
        "(access-tensor t)",
        vec![("t", array![[1., 2.], [3., 4.]].into_dyn())],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(tensor, array![[1., 2.], [3., 4.]].into_dyn());
                    assert_eq!(access_axis, 0);
                }
                _ => panic!(),
            }
        }
    );

    benchmark_test!(pad_type, bench_pad_type, "zero-padding", |value| {
        match value {
            Value::PadType(PadType::ZeroPadding) => (),
            _ => panic!(),
        }
    });

    benchmark_test!(
        access_pad,
        bench_access_pad,
        "(access-pad (access-tensor t) zero-padding 0 2 4)",
        vec![("t", array![[1., 2.], [3., 4.]].into_dyn())],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(
                        tensor,
                        array![
                            [0., 0.],
                            [0., 0.],
                            [1., 2.],
                            [3., 4.],
                            [0., 0.],
                            [0., 0.],
                            [0., 0.],
                            [0., 0.]
                        ]
                        .into_dyn()
                    );
                    assert_eq!(access_axis, 0);
                }
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        compute_reduce_max_0,
        bench_compute_reduce_max_0,
        "(compute reduce-max
            (access (access-tensor t) 0)
           )",
        vec![(
            "t",
            array![[[1, -2], [3, 0]], [[-5, 6], [0, 8]], [[-9, 10], [11, 12]],].into_dyn()
        )],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(access_axis, 0);
                    assert_eq!(tensor, ndarray::arr0(12).into_dyn());
                }
                _ => panic!(),
            }
        }
    );
    benchmark_test!(
        compute_reduce_max_1,
        bench_compute_reduce_max_1,
        "(compute reduce-max
            (access (access-tensor t) 1)
           )",
        vec![(
            "t",
            array![[[1, -2], [3, 0]], [[-5, 6], [0, 8]], [[-9, 10], [11, 12]],].into_dyn()
        )],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(access_axis, 1);
                    assert_eq!(tensor, array![3, 8, 12].into_dyn());
                }
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        compute_reduce_max_2,
        bench_compute_reduce_max_2,
        "(compute reduce-max
            (access (access-tensor t) 2)
           )",
        vec![(
            "t",
            array![[[1, -2], [3, 0]], [[-5, 6], [0, 8]], [[-9, 10], [11, 12]],].into_dyn()
        )],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(access_axis, 2);
                    assert_eq!(tensor, array![[1, 3], [6, 8], [10, 12]].into_dyn());
                }
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        compute_reduce_max_3,
        bench_compute_reduce_max_3,
        "(compute reduce-max
            (access (access-tensor t) 3)
            )",
        vec![(
            "t",
            array![[[1, -2], [3, 0]], [[-5, 6], [0, 8]], [[-9, 10], [11, 12]],].into_dyn()
        )],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(access_axis, 3);
                    assert_eq!(
                        tensor,
                        array![[[1, -2], [3, 0]], [[-5, 6], [0, 8]], [[-9, 10], [11, 12]],]
                            .into_dyn(),
                    );
                }
                _ => panic!(),
            }
        }
    );
    benchmark_test!(
        access_squeeze_0,
        bench_access_squeeze_0,
        "(access-squeeze (access-tensor t) 0)",
        vec![("t", array![[1., 2.]].into_dyn())],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(tensor, array![1., 2.].into_dyn());
                    assert_eq!(access_axis, 0);
                }
                _ => panic!(),
            }
        }
    );
    benchmark_test!(
        access_squeeze_1,
        bench_access_squeeze_1,
        "(access-squeeze (access (access-tensor t) 1) 0)",
        vec![("t", array![[1., 2.]].into_dyn())],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(tensor, array![1., 2.].into_dyn());
                    assert_eq!(access_axis, 0);
                }
                _ => panic!(),
            }
        }
    );
    benchmark_test!(
        #[should_panic]
        access_squeeze_panic,
        bench_access_squeeze_panic,
        "(access-squeeze (access (access-tensor t) 1) 1)",
        vec![("t", array![[1., 2.]].into_dyn())],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(tensor, array![1., 2.].into_dyn());
                    assert_eq!(access_axis, 0);
                }
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        max_pool2d,
        bench_max_pool2d,
        "(compute reduce-max
            (access-windows (access (access-tensor t) 3) (shape 1 2 2) (shape 1 2 2))
           )",
        vec![(
            "t",
            array![
                [[1, -2, -4, 5], [3, 6, -8, 0]],
                [[-5, 6, -8, -10], [0, 0, 0, 8]],
                [[-9, -20, -15, 10], [-1, 2, 11, 12]],
            ]
            .into_dyn()
        )],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(access_axis, 3);
                    // Checking that output is CHW.
                    assert_eq!(tensor.shape(), [3, 1, 2]);
                    assert_eq!(tensor, array![[[6, 5]], [[6, 8]], [[2, 12]]].into_dyn(),);
                }
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        access_pair_0,
        bench_access_pair_0,
        "(access-pair (access (access-tensor a) 0) (access (access-tensor b) 0))",
        vec![
            ("a", array![[1, 2], [3, 4]].into_dyn()),
            ("b", array![[5, 6], [7, 8]].into_dyn())
        ],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(tensor.shape(), [2, 2, 2]);
                    assert_eq!(
                        tensor,
                        array![[[1, 2], [3, 4]], [[5, 6], [7, 8]]].into_dyn()
                    );
                    assert_eq!(access_axis, 0);
                }
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        access_pair_1,
        bench_access_pair_1,
        "(access-pair (access (access-tensor a) 1) (access (access-tensor b) 1))",
        vec![
            ("a", array![[1, 2], [3, 4]].into_dyn()),
            ("b", array![[5, 6], [7, 8]].into_dyn())
        ],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(tensor.shape(), [2, 2, 2]);
                    assert_eq!(
                        tensor,
                        array![[[1, 2], [5, 6]], [[3, 4], [7, 8]]].into_dyn()
                    );
                    assert_eq!(access_axis, 1);
                }
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        access_pair_2,
        bench_access_pair_2,
        "(access-pair (access (access-tensor a) 2) (access (access-tensor b) 2))",
        vec![
            ("a", array![[1, 2], [3, 4]].into_dyn()),
            ("b", array![[5, 6], [7, 8]].into_dyn())
        ],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(tensor.shape(), [2, 2, 2]);
                    assert_eq!(
                        tensor,
                        array![[[1, 5], [2, 6]], [[3, 7], [4, 8]]].into_dyn()
                    );
                    assert_eq!(access_axis, 2);
                }
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        #[should_panic]
        access_pair_panic,
        bench_access_pair_panic,
        "(access-pair (access (access-tensor a) 2) (access (access-tensor b) 2))",
        vec![
            ("a", array![[1], [3]].into_dyn()),
            ("b", array![[5, 6], [7, 8]].into_dyn())
        ],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(tensor.shape(), [2, 2, 2]);
                    assert_eq!(
                        tensor,
                        array![[[1, 5], [2, 6]], [[3, 7], [4, 8]]].into_dyn()
                    );
                    assert_eq!(access_axis, 2);
                }
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        access_insert_axis_0,
        bench_access_insert_axis_0,
        "(access-insert-axis (access (access-tensor t) 0) 0)",
        vec![("t", array![1, 2].into_dyn())],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(tensor, array![[1, 2]].into_dyn());
                    assert_eq!(access_axis, 1);
                }
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        access_insert_axis_1,
        bench_access_insert_axis_1,
        "(access-insert-axis (access (access-tensor t) 0) 1)",
        vec![("t", array![1, 2].into_dyn())],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(tensor, array![[1], [2]].into_dyn());
                    assert_eq!(access_axis, 0);
                }
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        access_broadcast,
        bench_access_broadcast,
        "(access-broadcast (access (access-tensor t) 0) (access-shape (shape 2 2) (shape)))",
        vec![("t", array![[1, 2]].into_dyn())],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(tensor, array![[1, 2], [1, 2]].into_dyn());
                    assert_eq!(access_axis, 0);
                }
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        #[should_panic]
        access_broadcast_panic,
        bench_access_broadcast_panic,
        "(access-broadcast (access (access-tensor t) 0) (shape 2 2 2))",
        vec![("t", array![[1, 2]].into_dyn())],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(tensor, array![[1, 2], [1, 2]].into_dyn());
                    assert_eq!(access_axis, 0);
                }
                _ => panic!(),
            }
        }
    );
    benchmark_test!(
        access_transpose_0,
        bench_access_transpose_0,
        "(access-transpose (access (access-tensor t) 0) (list 1 0))",
        vec![("t", array![[1, 2]].into_dyn())],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(tensor, array![[1], [2]].into_dyn());
                    assert_eq!(access_axis, 0);
                }
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        access_transpose_1,
        bench_access_transpose_1,
        "(access-transpose (access (access-tensor t) 0) (list 1 0))",
        vec![("t", array![[1, 2]].into_dyn())],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(tensor, array![[1], [2]].into_dyn());
                    assert_eq!(access_axis, 0);
                }
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        access_transpose_2,
        bench_access_transpose_2,
        "(access-transpose (access (access-tensor t) 0) (list 1 0))",
        vec![("t", array![[2, 3], [1, 2]].into_dyn())],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(tensor, array![[2, 1], [3, 2]].into_dyn());
                    assert_eq!(access_axis, 0);
                }
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        #[should_panic]
        access_transpose_panic_0,
        bench_access_transpose_panic_0,
        "(access-transpose (access (access-tensor t) 0) (list 1 0 2))",
        vec![("t", array![[2, 3], [1, 2]].into_dyn())],
        |value| { value }
    );
    benchmark_test!(
        #[should_panic]
        access_transpose_panic_1,
        bench_access_transpose_panic_1,
        "(access-transpose (access (access-tensor t) 0) (list 1 1))",
        vec![("t", array![[2, 3], [1, 2]].into_dyn())],
        |value| { value }
    );
    benchmark_test!(
        compute_softmax,
        bench_compute_softmax,
        "(compute softmax (access (access-tensor t) 1))",
        vec![(
            "t",
            array![[0.4597965, 0.8250755], [0.14535584, 0.16271448]].into_dyn()
        )],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert!(tensor.abs_diff_eq(
                        &array![[0.40968227, 0.5903177], [0.49566042, 0.5043395]].into_dyn(),
                        1e-7
                    ));
                    assert_eq!(access_axis, 1);
                }
                _ => panic!(),
            }
        }
    );
    benchmark_test!(
        access_flatten_0,
        bench_access_flatten_0,
        "(access-flatten (access (access-tensor t) 0))",
        vec![(
            "t",
            ndarray::ArrayD::from_shape_vec(
                vec![2, 2, 4, 6, 10],
                (0..2 * 2 * 4 * 6 * 10).collect(),
            )
            .unwrap(),
        )],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis: _,
                }) => {
                    assert_eq!(
                        tensor,
                        ndarray::ArrayD::from_shape_vec(
                            vec![2 * 2 * 4 * 6 * 10],
                            (0..2 * 2 * 4 * 6 * 10).collect(),
                        )
                        .unwrap(),
                    );
                }
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        access_flatten_1,
        bench_access_flatten_1,
        "(access-flatten (access (access-tensor t) 1))",
        vec![(
            "t",
            ndarray::ArrayD::from_shape_vec(
                vec![2, 2, 4, 6, 10],
                (0..2 * 2 * 4 * 6 * 10).collect(),
            )
            .unwrap(),
        )],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis: _,
                }) => {
                    assert_eq!(
                        tensor,
                        ndarray::ArrayD::from_shape_vec(
                            vec![2, 2 * 4 * 6 * 10],
                            (0..2 * 2 * 4 * 6 * 10).collect(),
                        )
                        .unwrap(),
                    );
                }
                _ => panic!(),
            }
        }
    );
    benchmark_test!(
        access_flatten_2,
        bench_access_flatten_2,
        "(access-flatten (access (access-tensor t) 2))",
        vec![(
            "t",
            ndarray::ArrayD::from_shape_vec(
                vec![2, 2, 4, 6, 10],
                (0..2 * 2 * 4 * 6 * 10).collect(),
            )
            .unwrap(),
        )],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis: _,
                }) => {
                    assert_eq!(
                        tensor,
                        ndarray::ArrayD::from_shape_vec(
                            vec![2 * 2, 4 * 6 * 10],
                            (0..2 * 2 * 4 * 6 * 10).collect(),
                        )
                        .unwrap(),
                    );
                }
                _ => panic!(),
            }
        }
    );
    benchmark_test!(
        access_flatten_3,
        bench_access_flatten_3,
        "(access-flatten (access (access-tensor t) 5))",
        vec![(
            "t",
            ndarray::ArrayD::from_shape_vec(
                vec![2, 2, 4, 6, 10],
                (0..2 * 2 * 4 * 6 * 10).collect(),
            )
            .unwrap(),
        )],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis: _,
                }) => {
                    assert_eq!(
                        tensor,
                        ndarray::ArrayD::from_shape_vec(
                            vec![2 * 2 * 4 * 6 * 10],
                            (0..2 * 2 * 4 * 6 * 10).collect(),
                        )
                        .unwrap(),
                    );
                }
                _ => panic!(),
            }
        }
    );
    benchmark_test!(
        compute_reduce_mean_0,
        bench_compute_reduce_mean_0,
        "(compute reduce-mean (access (access-tensor t) 0))",
        vec![(
            "t",
            array![[[1f64, 2f64], [3f64, 4f64]], [[5f64, 6f64], [7f64, 8f64]]].into_dyn()
        )],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(
                        tensor,
                        ndarray::arr0(
                            (1f64 + 2f64 + 3f64 + 4f64 + 5f64 + 6f64 + 7f64 + 8f64) / 8f64
                        )
                        .into_dyn(),
                    );
                    assert_eq!(access_axis, 0);
                }
                _ => panic!(),
            }
        }
    );
    benchmark_test!(
        compute_reduce_mean_1,
        bench_compute_reduce_mean_1,
        "(compute reduce-mean (access (access-tensor t) 1))",
        vec![(
            "t",
            array![[[1f64, 2f64], [3f64, 4f64]], [[5f64, 6f64], [7f64, 8f64]]].into_dyn()
        )],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(
                        tensor,
                        array![
                            (1f64 + 2f64 + 3f64 + 4f64) / 4f64,
                            (5f64 + 6f64 + 7f64 + 8f64) / 4f64
                        ]
                        .into_dyn(),
                    );
                    assert_eq!(access_axis, 1);
                }
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        compute_reduce_mean_2,
        bench_compute_reduce_mean_2,
        "(compute reduce-mean (access (access-tensor t) 2))",
        vec![(
            "t",
            array![[[1f64, 2f64], [3f64, 4f64]], [[5f64, 6f64], [7f64, 8f64]]].into_dyn()
        )],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(
                        tensor,
                        array![
                            [(1f64 + 2f64) / 2f64, (3f64 + 4f64) / 2f64],
                            [(5f64 + 6f64) / 2f64, (7f64 + 8f64) / 2f64]
                        ]
                        .into_dyn(),
                    );
                    assert_eq!(access_axis, 2);
                }
                _ => panic!(),
            }
        }
    );
    benchmark_test!(
        compute_reduce_mean_3,
        bench_compute_reduce_mean_3,
        "(compute reduce-mean (access (access-tensor t) 3))",
        vec![(
            "t",
            array![[[1f64, 2f64], [3f64, 4f64]], [[5f64, 6f64], [7f64, 8f64]]].into_dyn()
        )],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(
                        tensor,
                        array![[[1f64, 2f64], [3f64, 4f64]], [[5f64, 6f64], [7f64, 8f64]]]
                            .into_dyn(),
                    );
                    assert_eq!(access_axis, 3);
                }
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        access_pad_min_padding,
        bench_access_pad_min_padding,
        "(access-pad (access-tensor t) min-padding 0 2 4)",
        vec![("t", array![[1., 2.], [3., 4.]].into_dyn())],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(
                        tensor,
                        array![
                            [std::f64::MIN, std::f64::MIN],
                            [std::f64::MIN, std::f64::MIN],
                            [1., 2.],
                            [3., 4.],
                            [std::f64::MIN, std::f64::MIN],
                            [std::f64::MIN, std::f64::MIN],
                            [std::f64::MIN, std::f64::MIN],
                            [std::f64::MIN, std::f64::MIN]
                        ]
                        .into_dyn()
                    );
                    assert_eq!(access_axis, 0);
                }
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        compute_elementwise_div,
        bench_compute_elementwise_div,
        "(compute elementwise-div
            (access (access-tensor t) 0)
           )",
        vec![(
            "t",
            array![
                [[1f32, -2f32], [3f32, 1f32]],
                [[-5f32, 6f32], [1f32, 8f32]],
                [[-9f32, 10f32], [11f32, 12f32]],
            ]
            .into_dyn()
        )],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(access_axis, 0);
                    assert_eq!(
                        tensor,
                        array![
                            [1f32 / -5f32 / -9f32, -2f32 / 6f32 / 10f32],
                            [3f32 / 1f32 / 11f32, 1f32 / 8f32 / 12f32]
                        ]
                        .into_dyn()
                    );
                }
                _ => panic!(),
            }
        }
    );
    benchmark_test!(literal_0, bench_literal_0, "(literal 0.1234)", |value| {
        match value {
            Value::Tensor(t) => {
                assert_eq!(t, ndarray::arr0(0.1234).into_dyn());
            }
            _ => panic!(),
        }
    });

    benchmark_test!(
        access_literal,
        bench_access_literal,
        "(access-literal (literal 0.1234))",
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(tensor, ndarray::arr0(0.1234).into_dyn());
                    assert_eq!(access_axis, 0);
                }
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        compute_sqrt,
        bench_compute_sqrt,
        "(compute sqrt
            (access (access-tensor t) 0)
           )",
        vec![(
            "t",
            array![
                [[1f64, 2f64], [3f64, 0f64]],
                [[5f64, 6f64], [0f64, 8f64]],
                [[9f64, 10f64], [11f64, 12f64]],
            ]
            .into_dyn()
        )],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(access_axis, 0);
                    assert_eq!(
                        tensor,
                        array![
                            [[1f64.sqrt(), 2f64.sqrt()], [3f64.sqrt(), 0f64.sqrt()]],
                            [[5f64.sqrt(), 6f64.sqrt()], [0f64.sqrt(), 8f64.sqrt()]],
                            [[9f64.sqrt(), 10f64.sqrt()], [11f64.sqrt(), 12f64.sqrt()]],
                        ]
                        .into_dyn(),
                    );
                }
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        compute_negative,
        bench_compute_negative,
        "(compute negative
            (access (access-tensor t) 0)
           )",
        vec![(
            "t",
            array![
                [[1f32, -2f32], [3f32, 1f32]],
                [[-5f32, 6f32], [1f32, 8f32]],
                [[-9f32, 10f32], [11f32, 12f32]],
            ]
            .into_dyn()
        )],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(access_axis, 0);
                    assert_eq!(
                        tensor,
                        array![
                            [[-1f32, 2f32], [-3f32, -1f32]],
                            [[5f32, -6f32], [-1f32, -8f32]],
                            [[9f32, -10f32], [-11f32, -12f32]],
                        ]
                        .into_dyn()
                    );
                }
                _ => panic!(),
            }
        }
    );
    benchmark_test!(
        access_concatenate_0,
        bench_access_concatenate_0,
        "(access-concatenate (access (access-tensor t) 0) (access (access-tensor n) 0) 0)",
        vec![
            ("t", array![[1, 2]].into_dyn()),
            ("n", array![[1, 2], [3, 4]].into_dyn())
        ],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(tensor, array![[1, 2], [1, 2], [3, 4]].into_dyn());
                    assert_eq!(access_axis, 0);
                }
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        access_concatenate_1,
        bench_access_concatenate_1,
        "(access-concatenate (access (access-tensor t) 0) (access (access-tensor n) 0) 1)",
        vec![
            ("t", array![[1], [2]].into_dyn()),
            ("n", array![[1, 2], [3, 4]].into_dyn())
        ],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(tensor, array![[1, 1, 2], [2, 3, 4]].into_dyn());
                    assert_eq!(access_axis, 0);
                }
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        #[should_panic]
        access_concatenate_panic_0,
        bench_access_concatenate_panic_0,
        "(access-concatenate (access (access-tensor t) 0) (access (access-tensor n) 1) 1)",
        vec![
            ("t", array![[1], [2]].into_dyn()),
            ("n", array![[1, 2], [3, 4]].into_dyn())
        ],
        |value| { value }
    );
    benchmark_test!(
        #[should_panic]
        access_concatenate_panic_1,
        bench_access_concatenate_panic_1,
        "(access-concatenate (access (access-tensor t) 0) (access (access-tensor n) 0) 0)",
        vec![
            ("t", array![[1], [2]].into_dyn()),
            ("n", array![[1, 2], [3, 4]].into_dyn())
        ],
        |value| { value }
    );
    benchmark_test!(
        access_slice_0,
        bench_access_slice_0,
        "(access-slice (access (access-tensor t) 0) 0 0 1)",
        vec![("t", array![[1, 2], [3, 4]].into_dyn())],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(tensor, array![[1, 2]].into_dyn());
                    assert_eq!(access_axis, 0);
                }
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        access_slice_1,
        bench_access_slice_1,
        "(access-slice (access (access-tensor t) 0) 0 0 2)",
        vec![("t", array![[1, 2], [3, 4]].into_dyn())],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(tensor, array![[1, 2], [3, 4]].into_dyn());
                    assert_eq!(access_axis, 0);
                }
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        access_slice_2,
        bench_access_slice_2,
        "(access-slice (access (access-tensor t) 0) 1 1 2)",
        vec![("t", array![[1, 2], [3, 4]].into_dyn())],
        |value| {
            match value {
                Value::Access(Access {
                    tensor,
                    access_axis,
                }) => {
                    assert_eq!(tensor, array![[2], [4]].into_dyn());
                    assert_eq!(access_axis, 0);
                }
                _ => panic!(),
            }
        }
    );

    benchmark_test!(
        #[should_panic]
        access_slice_panic_0,
        bench_access_slice_panic_0,
        "(access-slice (access (access-tensor t) 0) 0 0 3)",
        vec![("t", array![[1, 2], [3, 4]].into_dyn())],
        |value| { value }
    );
    benchmark_test!(
        #[should_panic]
        access_slice_panic_1,
        bench_access_slice_panic_1,
        "(access-slice (access (access-tensor t) 0) 2 0 1)",
        vec![("t", array![[1, 2], [3, 4]].into_dyn())],
        |value| { value }
    );
    benchmark_test!(
        access_shape,
        bench_access_shape,
        "(access-shape (shape 1 2) (shape 3 4))",
        |value| {
            match value {
                Value::AccessShape(shape, access_axis) => {
                    assert_eq!(shape.slice(), &[1, 2, 3, 4]);
                    assert_eq!(access_axis, 2);
                }
                _ => panic!(),
            }
        }
    );
}
