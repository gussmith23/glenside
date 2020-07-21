use super::language::{ComputeType, Language};
use egg::RecExpr;
use itertools::Itertools;
use ndarray::{s, ArrayD, Dimension, IxDyn};
use std::collections::hash_map::HashMap;

pub enum Value<DataType> {
    Tensor(ArrayD<DataType>),
    Access(Access<DataType>),
    Usize(usize),
    Shape(IxDyn),
    ComputeType(ComputeType),
}

pub struct Access<DataType> {
    pub tensor: ArrayD<DataType>,
    pub access_axis: usize,
}

pub type Environment<'a, DataType> = HashMap<&'a str, ArrayD<DataType>>;

pub fn interpret<DataType>(
    expr: &RecExpr<Language>,
    index: usize,
    env: &Environment<DataType>,
) -> Value<DataType>
where
    DataType: Copy
        + std::ops::Mul<Output = DataType>
        + num_traits::identities::One
        + num_traits::identities::Zero
        + std::cmp::PartialOrd,
{
    match &expr.as_ref()[index] {
        Language::ComputeType(t) => Value::ComputeType(t.clone()),
        &Language::Compute([compute_type_id, access_id]) => {
            let compute_type = match interpret(expr, compute_type_id as usize, env) {
                Value::ComputeType(t) => t,
                _ => panic!(),
            };
            let access = match interpret(expr, access_id as usize, env) {
                Value::Access(a) => a,
                _ => panic!(),
            };

            match compute_type {
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
            }
        }
        &Language::AccessCartesianProduct([a0_id, a1_id]) => {
            let (a0, a1) = match (
                interpret(expr, a0_id as usize, env),
                interpret(expr, a1_id as usize, env),
            ) {
                (Value::Access(a0), Value::Access(a1)) => (a0, a1),
                _ => panic!(),
            };

            assert_eq!(
                a0.tensor.shape()[a0.access_axis..],
                a1.tensor.shape()[a1.access_axis..]
            );

            let reshaped_0 = a0
                .tensor
                .clone()
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
            let access = match interpret(expr, access_id as usize, env) {
                Value::Access(a) => a,
                _ => panic!(),
            };
            let dim = match interpret(expr, dim_id as usize, env) {
                Value::Usize(u) => u,
                _ => panic!(),
            };

            Value::Access(Access {
                tensor: access.tensor,
                // TODO(@gussmith) Settle on vocab: "axis" or "dimension"?
                access_axis: dim,
            })
        }
        &Language::AccessWindows([tensor_id, filters_shape_id, x_stride_id, y_stride_id]) => {
            let tensor = match interpret(expr, tensor_id as usize, env) {
                Value::Tensor(t) => t,
                _ => panic!(),
            };
            let filters_shape = match interpret(expr, filters_shape_id as usize, env) {
                Value::Shape(s) => s,
                _ => panic!(),
            };
            let x_stride = match interpret(expr, x_stride_id as usize, env) {
                Value::Usize(u) => u,
                _ => panic!(),
            };
            let y_stride = match interpret(expr, y_stride_id as usize, env) {
                Value::Usize(u) => u,
                _ => panic!(),
            };

            // Won't always have to be true. Just simplifying right now.
            assert_eq!(tensor.ndim(), 3);
            assert_eq!(filters_shape.ndim(), 3);

            assert_eq!(tensor.ndim(), filters_shape.ndim());
            assert_eq!(tensor.shape()[0], filters_shape[0]);

            // TODO(@gussmith) Need one central place for window-gen logic
            // I'm duplicating this logic between here and language.rs. It
            // should be centralized.
            let (tensor_x, tensor_y) = (tensor.shape()[1], tensor.shape()[2]);
            let (filters_x, filters_y) = (filters_shape[1], filters_shape[2]);
            let num_windows_x = ((tensor_x - (filters_x - 1)) + x_stride - 1) / x_stride;
            let num_windows_y = ((tensor_y - (filters_y - 1)) + y_stride - 1) / y_stride;

            let windows = (0..num_windows_x)
                .map(|x_window_index: usize| {
                    let window_start_x = x_window_index * x_stride;
                    let windows = (0..num_windows_y)
                        .map(|y_window_index: usize| {
                            let window_start_y = y_window_index * y_stride;

                            tensor
                                .slice(s![
                                    ..,
                                    window_start_x..window_start_x + filters_x,
                                    window_start_y..window_start_y + filters_y
                                ])
                                .insert_axis(ndarray::Axis(0))
                        })
                        .collect::<Vec<_>>();
                    ndarray::stack(
                        ndarray::Axis(0),
                        windows
                            .iter()
                            .map(|t| t.view())
                            .collect::<Vec<_>>()
                            .as_slice(),
                    )
                    .unwrap()
                    .insert_axis(ndarray::Axis(0))
                })
                .collect::<Vec<_>>();
            let out = ndarray::stack(
                ndarray::Axis(0),
                windows
                    .iter()
                    .map(|t| t.view())
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
            .unwrap();

            Value::Access(Access {
                tensor: out.into_dyn(),
                access_axis: 2,
            })
        }
        Language::Shape(list) => Value::Shape(IxDyn(
            list.iter()
                .map(|id: &u32| match interpret(expr, *id as usize, env) {
                    Value::Usize(u) => u,
                    _ => panic!(),
                })
                .collect::<Vec<_>>()
                .as_slice(),
        )),
        &Language::SliceShape([shape_id, slice_axis_id]) => match (
            interpret(expr, shape_id as usize, env),
            interpret(expr, slice_axis_id as usize, env),
        ) {
            (Value::Shape(s), Value::Usize(u)) => {
                Value::Shape(IxDyn(s.as_array_view().slice(s![u..]).to_slice().unwrap()))
            }
            _ => panic!(),
        },
        &Language::ShapeOf([tensor_id]) => match interpret(expr, tensor_id as usize, env) {
            Value::Tensor(t) => Value::Shape(IxDyn(t.shape())),
            _ => panic!(),
        },
        &Language::AccessTensor(tensor_id) => match interpret(expr, tensor_id as usize, env) {
            Value::Tensor(t) => Value::Access(Access {
                tensor: t,
                // TODO(@gussmith) Arbitrarily picked default access axis
                access_axis: 0,
            }),
            _ => panic!(),
        },
        Language::Symbol(s) => Value::Tensor(env[s.as_str()].clone()),
        &Language::Usize(u) => Value::Usize(u),
        _ => panic!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::str::FromStr;

    #[test]
    fn compute_reduce_sum_0() {
        let mut env = Environment::new();
        env.insert(
            "t",
            array![[[1, -2], [3, 0]], [[-5, 6], [0, 8]], [[-9, 10], [11, 12]],].into_dyn(),
        );

        let expr = RecExpr::<Language>::from_str(
            "(compute reduce-sum
              (access (access-tensor t) 0)
             )",
        )
        .unwrap();

        match interpret(&expr, expr.as_ref().len() - 1, &env) {
            Value::Access(Access {
                tensor,
                access_axis,
            }) => {
                assert_eq!(access_axis, 0);
                assert_eq!(
                    tensor,
                    ndarray::arr0(1 + -2 + 3 + 0 + -5 + 6 + 0 + 8 + -9 + 10 + 11 + 12).into_dyn()
                );
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_reduce_sum_1() {
        let mut env = Environment::new();
        env.insert(
            "t",
            array![[[1, -2], [3, 0]], [[-5, 6], [0, 8]], [[-9, 10], [11, 12]],].into_dyn(),
        );

        let expr = RecExpr::<Language>::from_str(
            "(compute reduce-sum
              (access (access-tensor t) 1)
             )",
        )
        .unwrap();

        match interpret(&expr, expr.as_ref().len() - 1, &env) {
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

    #[test]
    fn compute_reduce_sum_2() {
        let mut env = Environment::new();
        env.insert(
            "t",
            array![[[1, -2], [3, 0]], [[-5, 6], [0, 8]], [[-9, 10], [11, 12]],].into_dyn(),
        );

        let expr = RecExpr::<Language>::from_str(
            "(compute reduce-sum
              (access (access-tensor t) 2)
             )",
        )
        .unwrap();

        match interpret(&expr, expr.as_ref().len() - 1, &env) {
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

    #[test]
    fn compute_reduce_sum_3() {
        let mut env = Environment::new();
        env.insert(
            "t",
            array![[[1, -2], [3, 0]], [[-5, 6], [0, 8]], [[-9, 10], [11, 12]],].into_dyn(),
        );

        let expr = RecExpr::<Language>::from_str(
            "(compute reduce-sum
              (access (access-tensor t) 3)
             )",
        )
        .unwrap();

        match interpret(&expr, expr.as_ref().len() - 1, &env) {
            Value::Access(Access {
                tensor,
                access_axis,
            }) => {
                assert_eq!(access_axis, 3);
                assert_eq!(
                    tensor,
                    array![[[1, -2], [3, 0]], [[-5, 6], [0, 8]], [[-9, 10], [11, 12]],].into_dyn(),
                );
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_relu_0() {
        let mut env = Environment::new();
        env.insert(
            "t",
            array![[[1, -2], [3, 0]], [[-5, 6], [0, 8]], [[-9, 10], [11, 12]],].into_dyn(),
        );

        let expr = RecExpr::<Language>::from_str(
            "(compute relu
              (access (access-tensor t) 0)
             )",
        )
        .unwrap();

        match interpret(&expr, expr.as_ref().len() - 1, &env) {
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

    #[test]
    fn compute_relu_1() {
        let mut env = Environment::new();
        env.insert(
            "t",
            array![[[1, -2], [3, 0]], [[-5, 6], [0, 8]], [[-9, 10], [11, 12]],].into_dyn(),
        );

        let expr = RecExpr::<Language>::from_str(
            "(compute relu
              (access (access-tensor t) 2)
             )",
        )
        .unwrap();

        match interpret(&expr, expr.as_ref().len() - 1, &env) {
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

    #[test]
    fn compute_dot_product_0() {
        let mut env = Environment::new();
        env.insert(
            "t",
            // 3 x 2 x 2
            array![[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]],].into_dyn(),
        );

        let expr = RecExpr::<Language>::from_str(
            "(compute dot-product
              (access (access-tensor t) 0)
             )",
        )
        .unwrap();

        match interpret(&expr, expr.as_ref().len() - 1, &env) {
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

    #[test]
    fn compute_dot_product_1() {
        let mut env = Environment::new();
        env.insert(
            "t",
            // 3 x 2 x 2
            array![[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]],].into_dyn(),
        );

        let expr = RecExpr::<Language>::from_str(
            "(compute dot-product
              (access (access-tensor t) 1)
             )",
        )
        .unwrap();

        match interpret(&expr, expr.as_ref().len() - 1, &env) {
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

    #[test]
    fn compute_dot_product_2() {
        let mut env = Environment::new();
        env.insert(
            "t",
            // 3 x 2 x 2
            array![[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]],].into_dyn(),
        );

        let expr = RecExpr::<Language>::from_str(
            "(compute dot-product
              (access (access-tensor t) 2)
             )",
        )
        .unwrap();

        match interpret(&expr, expr.as_ref().len() - 1, &env) {
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

    #[test]
    fn access_cartesian_product() {
        let mut env = Environment::new();
        env.insert(
            "t0",
            // 3 x 2 x 2
            array![[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]],].into_dyn(),
        );
        env.insert(
            "t1",
            // 2 x 2 x 2
            array![[[13, 14], [15, 16]], [[17, 18], [19, 20]]].into_dyn(),
        );

        let expr = RecExpr::<Language>::from_str(
            "(access-cartesian-product
              (access (access-tensor t0) 2)
              (access (access-tensor t1) 2)
             )",
        )
        .unwrap();

        match interpret(&expr, expr.as_ref().len() - 1, &env) {
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
    #[test]
    fn access() {
        let mut env = Environment::new();
        env.insert("t", array![[1., 2.], [3., 4.]].into_dyn());

        let expr = RecExpr::<Language>::from_str("(access (access-tensor t) 1)").unwrap();
        match interpret(&expr, expr.as_ref().len() - 1, &env) {
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

    #[test]
    fn access_windows() {
        let mut env = Environment::new();
        env.insert(
            "t",
            array![
                [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                [[10., 11., 12.], [13., 14., 15.], [16., 17., 18.]],
                [[19., 20., 21.], [22., 23., 24.], [25., 26., 27.]],
            ]
            .into_dyn(),
        );

        let expr = RecExpr::<Language>::from_str(
            "
             (access-windows
              t
              (shape 3 2 2)
              1
              1
             )",
        )
        .unwrap();
        match interpret(&expr, expr.as_ref().len() - 1, &env) {
            Value::Access(a) => {
                assert_eq!(a.access_axis, 2);
                assert_eq!(a.tensor.shape(), &[2, 2, 3, 2, 2]);
                assert_eq!(
                    a.tensor.slice(s![0, 0, .., .., ..]),
                    array![
                        [[1., 2.], [4., 5.]],
                        [[10., 11.], [13., 14.]],
                        [[19., 20.], [22., 23.]],
                    ]
                );
                assert_eq!(
                    a.tensor.slice(s![1, 0, .., .., ..]),
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

    #[test]
    fn shape() {
        let expr = RecExpr::<Language>::from_str("(shape 1 2 3)").unwrap();
        match interpret(
            &expr,
            expr.as_ref().len() - 1,
            &Environment::<f32>::default(),
        ) {
            Value::Shape(s) => assert_eq!(s, IxDyn(&[1, 2, 3])),
            _ => panic!(),
        }
    }

    #[test]
    fn slice_shape_0() {
        let mut env = Environment::new();
        env.insert("t", array![[1., 2.], [3., 4.]].into_dyn());

        let expr = RecExpr::<Language>::from_str("(slice-shape (shape-of t) 0)").unwrap();
        match interpret(&expr, expr.as_ref().len() - 1, &env) {
            Value::Shape(s) => assert_eq!(s, IxDyn(&[2, 2])),
            _ => panic!(),
        }
    }

    #[test]
    fn slice_shape_1() {
        let mut env = Environment::new();
        env.insert("t", array![[1., 2.], [3., 4.]].into_dyn());

        let expr = RecExpr::<Language>::from_str("(slice-shape (shape-of t) 1)").unwrap();
        match interpret(&expr, expr.as_ref().len() - 1, &env) {
            Value::Shape(s) => assert_eq!(s, IxDyn(&[2])),
            _ => panic!(),
        }
    }

    #[test]
    fn slice_shape_2() {
        let mut env = Environment::new();
        env.insert("t", array![[1., 2.], [3., 4.]].into_dyn());

        let expr = RecExpr::<Language>::from_str("(slice-shape (shape-of t) 2)").unwrap();
        match interpret(&expr, expr.as_ref().len() - 1, &env) {
            Value::Shape(s) => assert_eq!(s, IxDyn(&[])),
            _ => panic!(),
        }
    }

    #[test]
    fn shape_of() {
        let mut env = Environment::new();
        env.insert("t", array![[1., 2.], [3., 4.]].into_dyn());

        let expr = RecExpr::<Language>::from_str("(shape-of t)").unwrap();
        match interpret(&expr, expr.as_ref().len() - 1, &env) {
            Value::Shape(s) => assert_eq!(s, IxDyn(&[2, 2])),
            _ => panic!(),
        }
    }

    #[test]
    fn usize() {
        let expr = RecExpr::<Language>::from_str("23").unwrap();
        match interpret(
            &expr,
            expr.as_ref().len() - 1,
            &Environment::<f32>::default(),
        ) {
            Value::Usize(23) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn symbol() {
        let mut env = Environment::new();
        env.insert("t", array![[1., 2.], [3., 4.]].into_dyn());

        let expr = RecExpr::<Language>::from_str("t").unwrap();
        match interpret(&expr, expr.as_ref().len() - 1, &env) {
            Value::Tensor(t) => assert_eq!(t, array![[1., 2.], [3., 4.]].into_dyn()),
            _ => panic!(),
        }
    }

    #[test]
    fn access_tensor() {
        let mut env = Environment::new();
        env.insert("t", array![[1., 2.], [3., 4.]].into_dyn());

        let expr = RecExpr::<Language>::from_str("(access-tensor t)").unwrap();
        match interpret(&expr, expr.as_ref().len() - 1, &env) {
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
}
