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
    use std::str::FromStr;

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
