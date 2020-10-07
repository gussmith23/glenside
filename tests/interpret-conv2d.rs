mod common;

use common::load_npy;
use egg::RecExpr;
use glenside::language::interpreter::*;
use glenside::language::Language;
use std::str::FromStr;

#[test]
fn interpret_conv2d() {
    // TODO(@gussmith) Support batch dimension
    let expr = RecExpr::<Language>::from_str(
        "
         (compute dot-product
          (access-cartesian-product
           (access (access-tensor filters) 1)
           (access-squeeze
            (access-windows
             (access (access-tensor activations) 3)
             (slice-shape (shape-of filters) 1)
             (shape 1 1 1)
            )
            0
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
    assert_eq!(filters.shape(), &[8, 4, 3, 3]);
    let activations = load_npy::<f32>(
        format!(
            "{}/{}",
            env!("CARGO_MANIFEST_DIR"),
            "data/conv2d_activations.npy"
        )
        .as_str(),
    );
    assert_eq!(activations.shape(), &[4, 32, 32]);
    let result = load_npy::<f32>(
        format!(
            "{}/{}",
            env!("CARGO_MANIFEST_DIR"),
            "data/conv2d_result.npy"
        )
        .as_str(),
    );
    assert_eq!(result.shape(), &[8, 30, 30]);

    let mut env = Environment::new();
    env.insert("filters", filters);
    env.insert("activations", activations);

    use approx::AbsDiffEq;
    match interpret(&expr, expr.as_ref().len() - 1, &env) {
        Value::Access(a) => {
            assert_eq!(a.tensor.shape(), result.shape());
            // TODO(@gussmith) Is this tolerance too big?
            assert!(a.tensor.abs_diff_eq(&result, 5e-6));
        }
        _ => panic!(),
    };
}
