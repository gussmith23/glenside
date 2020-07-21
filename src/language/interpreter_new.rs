use super::language::Language;
use egg::RecExpr;
use ndarray::{s, ArrayD, Dimension, IxDyn};
use std::collections::hash_map::HashMap;

pub enum Value<DataType> {
    Tensor(ArrayD<DataType>),
    Access(Access<DataType>),
    Usize(usize),
    Shape(IxDyn),
}

pub struct Access<DataType> {
    pub tensor: ArrayD<DataType>,
    pub access_axis: usize,
}

pub type Environment<'a, DataType> = HashMap<&'a str, ArrayD<DataType>>;

pub fn interpret<DataType: Copy>(
    expr: &RecExpr<Language>,
    index: usize,
    env: &Environment<DataType>,
) -> Value<DataType> {
    match &expr.as_ref()[index] {
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
    use ndarray_npy::ReadableElement;
    use std::str::FromStr;

    fn load_npy<DataType: ReadableElement>(path: &str) -> ndarray::ArrayD<DataType> {
        ndarray_npy::read_npy::<_, ndarray::ArrayD<DataType>>(path).unwrap()
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

    #[test]
    #[should_panic]
    fn conv2d() {
        let expr = RecExpr::<Language>::from_str(
            "
         (compute dot-product
          (access-cartesian-product
           (access (access-tensor filters) 1)
           (access-windows
            activations
            (slice-shape (shape-of filters) 1)
            1
            1
           )
          )
         )
        ",
        )
        .unwrap();

        let filters = load_npy::<f32>(
            format!(
                "{}/{}",
                env!("CARGO_MANIFEST_DIR"),
                "data/conv2d_filters.npy"
            )
            .as_str(),
        );
        let activations = load_npy::<f32>(
            format!(
                "{}/{}",
                env!("CARGO_MANIFEST_DIR"),
                "data/conv2d_activations.npy"
            )
            .as_str(),
        );
        let result = load_npy::<f32>(
            format!(
                "{}/{}",
                env!("CARGO_MANIFEST_DIR"),
                "data/conv2d_result.npy"
            )
            .as_str(),
        );

        let mut env = Environment::new();
        env.insert("filters", filters);
        env.insert("activations", activations);

        match interpret(&expr, expr.as_ref().len() - 1, &env) {
            Value::Access(a) => {
                assert_eq!(a.tensor, result);
            }
            _ => panic!(),
        }
    }
}
