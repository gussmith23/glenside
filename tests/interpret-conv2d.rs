use glenside::language::interpreter_new::*;
use glenside::language::Language;
use egg::{RecExpr};
use ndarray_npy::ReadableElement;
use std::str::FromStr;

fn load_npy<DataType: ReadableElement>(path: &str) -> ndarray::ArrayD<DataType> {
    ndarray_npy::read_npy::<_, ndarray::ArrayD<DataType>>(path).unwrap()
}

#[test]
fn conv2d() {
    // TODO(@gussmith) Support batch dimension
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
    };
}
