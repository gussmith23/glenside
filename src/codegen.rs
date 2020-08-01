use crate::language::Language;
use crate::language::MyAnalysis;
use crate::language::MyAnalysisData;
use egg::EGraph;
use egg::Id;
use itertools::Itertools;
use ndarray::Dimension;
use ndarray::IxDyn;

type Expr = EGraph<Language, MyAnalysis>;

/// Finds all symbols in a program, and return their names.
pub fn find_vars(expr: &Expr, id: Id) -> Vec<String> {
    fn find_vars_recursive_helper(vec: &mut Vec<String>, expr: &Expr, id: Id) {
        match {
            assert_eq!(expr[id].nodes.len(), 1);
            &expr[id].nodes[0]
        } {
            Language::Symbol(s) => vec.push(s.to_string()),
            // Id
            &Language::AccessTensor(id) => {
                find_vars_recursive_helper(vec, expr, id);
            }
            // Box<[Id]>
            Language::List(ids) => {
                for id in ids.iter() {
                    find_vars_recursive_helper(vec, expr, *id);
                }
            }
            // [Id; 2]
            &Language::Access(ids) | &Language::AccessTranspose(ids) => {
                for id in ids.iter() {
                    find_vars_recursive_helper(vec, expr, *id);
                }
            }
            // [Id; 3]
            &Language::AccessMoveAxis(ids) => {
                for id in ids.iter() {
                    find_vars_recursive_helper(vec, expr, *id);
                }
            }
            // [Id; 4]
            &Language::SystolicArray(ids) => {
                for id in ids.iter() {
                    find_vars_recursive_helper(vec, expr, *id);
                }
            }
            &Language::Usize(_) => (),
            &Language::GetAccessShape(_)
            | &Language::AccessBroadcast(_)
            | &Language::AccessInsertAxis(_)
            | &Language::AccessPair(_)
            | &Language::AccessSqueeze(_)
            | Language::PadType(_)
            | &Language::AccessPad(_)
            | Language::ComputeType(_)
            | &Language::Compute(_)
            | &Language::AccessCartesianProduct(_)
            | &Language::AccessWindows(_)
            | Language::Shape(_)
            | &Language::SliceShape(_)
            | &Language::ShapeInsertAxis(_)
            | &Language::ShapeRemoveAxis(_)
            | &Language::ShapeOf(_)
            | &Language::MoveAxis(_)
            | &Language::CartesianProduct(_)
            | &Language::MapDotProduct(_)
            | &Language::Slice(_)
            | &Language::Concatenate(_)
            | &Language::ElementwiseAdd(_)
            | &Language::BsgSystolicArray(_)
            | &Language::AccessReshape(_)
            | &Language::AccessFlatten(_)
            | &Language::AccessShape(_)
            | &Language::AccessSlice(_)
            | &Language::AccessConcatenate(_)
            | &Language::AccessShiftRight(_) => panic!("{:#?} not implemented", expr[id].nodes[0]),
        }
    }

    let mut vec = Vec::default();
    find_vars_recursive_helper(&mut vec, expr, id);

    vec
}

/// Returns signature and code.
pub fn codegen(expr: &Expr, id: Id) -> (String, String) {
    let mut out = String::default();

    let mut signature = "void mlp(".to_string();
    signature.push_str("float * out, ");
    signature.push_str(
        find_vars(expr, id)
            .iter()
            .map(|var| format!("float * {}", var))
            .intersperse(", ".to_string())
            .chain(std::iter::once(")".to_string()))
            .collect::<String>()
            .as_str(),
    );

    out.push_str(signature.as_str());

    out.push_str("\n");
    out.push_str("{");
    out.push_str("\n");

    let mut body = String::default();
    codegen_recursive_helper(expr, id, &mut body).as_str();
    out.push_str(body.as_str());

    out.push_str("}");
    out.push_str("\n");

    (signature, out)
}

fn codegen_recursive_helper(expr: &Expr, id: Id, code: &mut String) -> String {
    match {
        assert_eq!(expr[id].nodes.len(), 1);
        &expr[id].nodes[0]
    } {
        Language::Symbol(s) => s.clone(),
        &Language::AccessTensor(symbol_id) => {
            let symbol = codegen_recursive_helper(expr, symbol_id, code);
            symbol
        }
        &Language::Access([access_tensor_id, axis_id]) => {
            let axis = MyAnalysis::get_usize(axis_id, expr);
            assert_eq!(axis, 0);
            codegen_recursive_helper(expr, access_tensor_id, code)
        }
        &Language::SystolicArray([rows_id, cols_id, a0_id, a1_id]) => {
            let rows = MyAnalysis::get_usize(rows_id, expr);
            let cols = MyAnalysis::get_usize(cols_id, expr);

            let (a0, a1) = match (&expr[a0_id].data, &expr[a1_id].data) {
                (MyAnalysisData::AccessPattern(a0), MyAnalysisData::AccessPattern(a1)) => (a0, a1),
                _ => panic!(),
            };

            assert_eq!(a1.shape, IxDyn(&[]));
            assert_eq!(a1.item_shape, IxDyn(&[rows, cols]));
            assert!(a0.shape.ndim() == 0 || a0.shape.ndim() == 1);
            assert_eq!(a0.item_shape, IxDyn(&[rows]));

            let this_access = match &expr[id].data {
                MyAnalysisData::AccessPattern(a) => a,
                _ => panic!(),
            };
            assert_eq!(this_access.shape.ndim(), 1);
            assert_eq!(this_access.item_shape.ndim(), 0);

            let s0 = codegen_recursive_helper(expr, a0_id, code);
            let s1 = codegen_recursive_helper(expr, a1_id, code);

            // TODO(@gussmith23) How to generate output buffer?
            // This seems like it might not be legal, just declaring it.
            code.push_str(format!("float out[{}];\n", this_access.shape.slice()[0]).as_str());

            code.push_str(
                format!(
                    "rtml_systolic_array_weight_stationary({}, {}, {}, {}, {}, {});\n",
                    "&out[0]", s0, rows, s1, rows, cols
                )
                .as_str(),
            );

            "out".to_string()
        }
        &Language::Usize(u) => format!("{}", u),
        &Language::GetAccessShape(_)
        | &Language::AccessTranspose(_)
        | &Language::AccessMoveAxis(_)
        | Language::List(_)
        | &Language::AccessBroadcast(_)
        | &Language::AccessInsertAxis(_)
        | &Language::AccessPair(_)
        | &Language::AccessSqueeze(_)
        | Language::PadType(_)
        | &Language::AccessPad(_)
        | Language::ComputeType(_)
        | &Language::Compute(_)
        | &Language::AccessCartesianProduct(_)
        | &Language::AccessWindows(_)
        | Language::Shape(_)
        | &Language::SliceShape(_)
        | &Language::ShapeInsertAxis(_)
        | &Language::ShapeRemoveAxis(_)
        | &Language::ShapeOf(_)
        | &Language::MoveAxis(_)
        | &Language::CartesianProduct(_)
        | &Language::MapDotProduct(_)
        | &Language::Slice(_)
        | &Language::Concatenate(_)
        | &Language::ElementwiseAdd(_)
        | &Language::BsgSystolicArray(_)
        | &Language::AccessReshape(_)
        | &Language::AccessFlatten(_)
        | &Language::AccessShape(_)
        | &Language::AccessSlice(_)
        | &Language::AccessConcatenate(_)
        | &Language::AccessShiftRight(_) => panic!("{:#?} not implemented", expr[id].nodes[0]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language::MyAnalysis;
    use egg::EGraph;
    use egg::RecExpr;
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::prelude::*;
    use std::process::Command;
    use std::str::FromStr;

    #[test]
    fn mlp() {
        const LIBRARY_FILENAME_C: &str = "mlp.c";
        const LIBRARY_FILENAME_O: &str = "mlp.o";
        const MAIN_FILENAME: &str = "main.c";

        // This is a simplified version of what's produced in the
        // regular-multilayer-perceptron test. It's simplified in that the
        // accesses are collapsed (transpose, move-axis) which are things that
        // we can achieve with rewrites.
        // TODO(@gussmith23) Rewrite to collapse move axis and transpose
        // TODO(@gussmith23) Change to just using transpose?
        let program = "
     (systolic-array 6 2
      (access
       (systolic-array 4 6
        (access
         (systolic-array 2 4
          (access (access-tensor input) 0)
          (access (access-tensor weight0) 0)
         )
         0
        )
        (access (access-tensor weight1) 0)
       )
       0
      )
      (access (access-tensor weight2) 0)
     )
     ";

        let mut map = HashMap::default();
        map.insert("input".to_string(), vec![2]);
        map.insert("weight0".to_string(), vec![2, 4]);
        map.insert("weight1".to_string(), vec![4, 6]);
        map.insert("weight2".to_string(), vec![6, 2]);

        let expr = RecExpr::from_str(program).unwrap();
        // Check that it "type checks"
        let mut egraph = EGraph::new(MyAnalysis { name_to_shape: map });
        let id = egraph.add_expr(&expr);

        let (signature, program) = codegen(&egraph, id);
        println!("{}", program);

        let mut file = File::create(LIBRARY_FILENAME_C).unwrap();
        file.write_all(program.as_bytes()).unwrap();

        let output = Command::new("gcc")
            .arg("-c")
            .arg(LIBRARY_FILENAME_C)
            .arg("-O0")
            .arg(format!("-o {}", LIBRARY_FILENAME_O))
            .output()
            .expect("Failed to compile with gcc");
        assert!(
            output.status.success(),
            "Compilation failed. stderr:\n{}",
            std::str::from_utf8(output.stderr.as_slice())
                .expect("Could not convert stderr to UTF8")
        );

        File::create(MAIN_FILENAME)
            .expect("Couldn't create main file")
            .write_all(
                format! {"
extern {};

float in  [2]    = {{1,2}};
float a   [2][2] = {{ {{1,2}}, {{3,4}} }};
float b   [2][2] = {{ {{1,2}}, {{3,4}} }};
float c   [2][2] = {{ {{1,2}}, {{3,4}} }};
float out [2][2] = {{ {{1,2}}, {{3,4}} }};

int main() {{
  mlp(&out[0][0], &in[0], &a[0][0], &b[0][0], &c[0][0]);
  return 0;
}}
",
                signature}
                .as_bytes(),
            )
            .expect("Couldn't write main file");

        let output = Command::new("gcc")
            .arg(MAIN_FILENAME)
            .arg(LIBRARY_FILENAME_O)
            .output()
            .expect("Failed to compile main file with gcc");
        assert!(
            output.status.success(),
            "Compilation failed. stderr:\n{}",
            std::str::from_utf8(output.stderr.as_slice())
                .expect("Could not convert stderr to UTF8")
        );

        let output = Command::new("./a.out")
            .output()
            .expect("Failed to run result");
        assert!(
            output.status.success(),
            "Main binary failed with code {}. stderr:\n{}",
            output.status.code().unwrap(),
            std::str::from_utf8(output.stderr.as_slice())
                .expect("Could not convert stderr to UTF8")
        );
    }
}
