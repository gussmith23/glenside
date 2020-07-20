use super::language::Language;
use egg::RecExpr;
use ndarray::ArrayD;
use std::collections::hash_map::HashMap;

pub enum Value<DataType> {
    Tensor(ArrayD<DataType>),
    Access(Access<DataType>),
}

pub struct Access<DataType> {
    pub tensor: ArrayD<DataType>,
    pub access_axis: usize,
}

pub type Environment<'a, DataType> = HashMap<&'a str, ArrayD<DataType>>;

pub fn interpret<DataType: Clone>(
    expr: &RecExpr<Language>,
    index: usize,
    env: &Environment<DataType>,
) -> Value<DataType> {
    match &expr.as_ref()[index] {
        &Language::AccessTensor(tensor_id) => match interpret(expr, tensor_id as usize, env) {
            Value::Tensor(t) => Value::Access(Access {
                tensor: t,
                // TODO(@gussmith) Arbitrarily picked default access axis
                access_axis: 0,
            }),
            _ => panic!(),
        },
        Language::Symbol(s) => Value::Tensor(env[s.as_str()].clone()),
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
