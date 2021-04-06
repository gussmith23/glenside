use std::collections::HashMap;

use super::language_new::Glenside;

use egg::RecExpr;
use ndarray::ArrayD;

pub enum Value {
    Code(fn(Vec<ArrayD<f64>>) -> ArrayD<f64>),
    Usize(usize),
    String(String),
    DimensionMap(HashMap<String, usize>),
    DimensionDefinition { name: String, length: usize },
    DimensionList(Vec<String>),
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
        Glenside::Input(ids) => make_interpreter_arm!(
            expr,
            match ids {
                (Value::String(_name), Value::DimensionMap(map)) => {
                    Value::DimensionMap(map.clone())
                }
            }
        ),
        Glenside::DimensionMapDefinition(ids) => {
            let mut out = HashMap::new();

            for (k, v) in ids
                .iter()
                .map(|id| match interpret(expr, usize::from(*id)) {
                    Value::DimensionDefinition { name, length } => (name, length),
                    _ => panic!(),
                })
                .collect::<Vec<_>>()
            {
                out.insert(k, v);
            }

            Value::DimensionMap(out)
        }
        Glenside::DimensionDefinition(ids) => make_interpreter_arm!(
            expr,
            match ids {
                (Value::String(name), Value::Usize(length)) => {
                    Value::DimensionDefinition {
                        name: name.clone(),
                        length: *length,
                    }
                }
            }
        ),
        Glenside::DimensionList(ids) => Value::DimensionList(
            ids.iter()
                .map(|id| match interpret(expr, usize::from(*id)) {
                    Value::String(name) => name,
                    _ => panic!(),
                })
                .collect::<Vec<_>>(),
        ),
        Glenside::Pair(_) => panic!("Remove pair"),
        Glenside::DotProduct(ids) => make_interpreter_arm!(
            expr,
            match ids {
                (Value::DimensionMap(_i0), Value::DimensionMap(_i1)) => {
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
        dimension_definition,
        "(dimension-definition N 1)",
        match result {
            Value::DimensionDefinition { name, length } => {
                assert_eq!(name, "N");
                assert_eq!(length, 1);
            }
        }
    );

    test!(
        dimension_map_definition,
        "(dimension-map-definition
            (dimension-definition N 1)
            (dimension-definition C 3)
            (dimension-definition H 32)
            (dimension-definition W 64)
        )",
        match result {
            Value::DimensionMap(map) => {
                assert_eq!(map.len(), 4);
                assert_eq!(map.get(&"N".to_string()), Some(&1));
                assert_eq!(map.get(&"C".to_string()), Some(&3));
                assert_eq!(map.get(&"H".to_string()), Some(&32));
                assert_eq!(map.get(&"W".to_string()), Some(&64));
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
        dimension_identifier_list,
        "(dimension-list N C H W)",
        match result {
            Value::DimensionList(list) => {
                assert_eq!(list, vec!["N", "C", "H", "W"]);
            }
        }
    );

    test!(
        input,
        "(input test_input
          (dimension-map-definition
           (dimension-definition N 1)
           (dimension-definition C 3)
           (dimension-definition H 32)
           (dimension-definition W 64)
          )
         )",
        match result {
            Value::DimensionMap(map) => {
                assert_eq!(map.len(), 4);
                assert_eq!(map.get(&"N".to_string()), Some(&1));
                assert_eq!(map.get(&"C".to_string()), Some(&3));
                assert_eq!(map.get(&"H".to_string()), Some(&32));
                assert_eq!(map.get(&"W".to_string()), Some(&64));
            }
        }
    );

    test!(
        #[should_panic = "not yet implemented: you ran the code!"]
        dot_product,
        "(dot-product
          (input test_input_0
           (dimension-map-definition
            (dimension-definition M 16)
            (dimension-definition N 32)
           )
          )
          (input test_input_1
           (dimension-map-definition
            (dimension-definition N 32)
            (dimension-definition O 64)
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
