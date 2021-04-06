use std::collections::HashSet;
use std::iter::FromIterator;

use super::language_new::Glenside;

use egg::RecExpr;
use ndarray::ArrayD;

type Dimension = crate::language::language_new::Dimension;

pub enum Value {
    Code(fn(Vec<ArrayD<f64>>) -> ArrayD<f64>),
    Usize(usize),
    String(String),
    Dimension(Dimension),
    DimensionSet(HashSet<Dimension>),
}

/// Interpret a Glenside expression
pub fn interpret(expr: &RecExpr<Glenside>, index: usize) -> Value {
    /// Helper macro for defining match arms.
    ///
    /// It's formatted this way so that we can use rustfmt to format macro
    /// invocations!
    macro_rules! make_interpreter_arm {
        ($expr:ident, match $ids:ident { ( $($p:pat),* $(,)? ) => $body:expr } ) => {
            {{
                match $ids.iter().map(|id| interpret($expr, usize::from(*id))).collect::<Vec<_>>().as_slice() {
                    [$($p),*] => $body,
                    _ => panic!("Does not type check"),
                }
            }}
        };
    }

    match &expr.as_ref()[index] {
        Glenside::Apply(_ids) => todo!(),
        Glenside::Input(ids) => make_interpreter_arm!(
            expr,
            match ids {
                (Value::String(_name), Value::DimensionSet(set)) => {
                    Value::DimensionSet(set.clone())
                }
            }
        ),
        Glenside::Dimension(ids) => make_interpreter_arm!(
            expr,
            match ids {
                (Value::String(name), Value::Usize(length)) => {
                    Value::Dimension((name.clone(), *length))
                }
            }
        ),
        Glenside::DimensionSet(ids) => {
            assert!(ids.is_sorted());
            Value::DimensionSet(HashSet::from_iter(ids.iter().map(|id| {
                match interpret(expr, usize::from(*id)) {
                    Value::Dimension(d) => d,
                    _ => panic!(),
                }
            })))
        }
        Glenside::DotProduct(ids) => make_interpreter_arm!(
            expr,
            match ids {
                (Value::DimensionSet(_i0), Value::DimensionSet(_i1)) => {
                    Value::Code(|_inputs| todo!("you ran the code!"))
                }
            }
        ),
        Glenside::Usize(u) => Value::Usize(u.clone()),
        Glenside::String(s) => Value::String(s.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::str::FromStr;

    /// Note that, if a program doesn't use a `DataType` explicitly, you must
    /// pass in a datatype as `$datatype`. It can be whatever you want.
    macro_rules! test {
        ($(#[$meta:meta])* $name:ident,
         $glenside_str:literal,
         match result { $result_pat:pat => $check_block:expr }) => {
            #[test]
            $(#[$meta])*
            fn $name() {
                let expr = RecExpr::from_str($glenside_str).unwrap();
                match interpret(&expr, expr.as_ref().len() - 1) {
                    $result_pat => $check_block,
                    _ => panic!(),
                }
            }
        };
    }

    macro_rules! code_test {
        ($(#[$meta:meta])* $name:ident,
         $glenside_str:literal,
         $($arg_val:expr),*,
         $check:expr
         ) => {
            #[test]
            $(#[$meta])*
            fn $name() {
                let expr = RecExpr::from_str($glenside_str).unwrap();
                match interpret(&expr, expr.as_ref().len() - 1) {
                    Value::Code(lambda) => $check(lambda(vec![$($arg_val),*])),
                    _ => panic!(),
                }
            }
        };
    }

    test!(
        dimension,
        "(d N 1)",
        match result {
            Value::Dimension((name, length)) => {
                assert_eq!(name, "N");
                assert_eq!(length, 1);
            }
        }
    );

    test!(
        dimension_identifier,
        "N",
        match result {
            Value::String(name) => {
                assert_eq!(name, "N");
            }
        }
    );

    test!(
        dimension_set,
        "(ds (d N 1) (d C 3) (d H 224) (d W 224))",
        match result {
            Value::DimensionSet(set) => {
                assert_eq!(
                    set,
                    HashSet::from_iter(
                        vec![
                            ("N".to_string(), 1),
                            ("C".to_string(), 3),
                            ("H".to_string(), 224),
                            ("W".to_string(), 224)
                        ]
                        .drain(..)
                    )
                );
            }
        }
    );

    test!(
        input,
        "(input test_input
          (ds
           (d N 1)
           (d C 3)
           (d H 32)
           (d W 64)
          )
         )",
        match result {
            Value::DimensionSet(set) => {
                assert_eq!(
                    set,
                    HashSet::from_iter(
                        vec![
                            ("N".to_string(), 1),
                            ("C".to_string(), 3),
                            ("H".to_string(), 32),
                            ("W".to_string(), 64)
                        ]
                        .drain(..)
                    )
                );
            }
        }
    );

    test!(
        #[should_panic = "not yet implemented: you ran the code!"]
        dot_product,
        "
        (dot-product
         (input
          input_MxN
          (ds
           (d M 16)
           (d N 32)
          )
         )
         (input
          input_NxO
          (ds
           (d N 32)
           (d O 64)
          )
         )
        )",
        match result {
            Value::Code(code) => {
                code(vec![]);
            }
        }
    );

    // TODO(@gussmith23)
    // code_test!(
    //     #[should_panic = "not yet implemented: you ran the code!"]
    //     dot_product,
    //     "(dot-product
    //       (input test_input_0
    //        (dimension-map-definition
    //         (dimension-definition M 16)
    //         (dimension-definition N 32)
    //        )
    //       )
    //       (input test_input_1
    //        (dimension-map-definition
    //         (dimension-definition N 32)
    //         (dimension-definition O 64)
    //        )
    //       )
    //      )",
    //     {
    //         // Arg 0
    //     },
    //     |_result| {}
    // );
}
