use crate::language::Language;
use egg::RecExpr;
use itertools::Itertools;

type Expr = RecExpr<Language>;

/// Finds all symbols in a program, and return their names.
pub fn find_vars(expr: &Expr, id: usize) -> Vec<String> {
    fn find_vars_recursive_helper(vec: &mut Vec<String>, expr: &Expr, id: usize) {
        match &expr.as_ref()[id] {
            Language::Symbol(s) => vec.push(s.to_string()),
            // Id
            &Language::AccessTensor(id) => {
                find_vars_recursive_helper(vec, expr, id as usize);
            }
            // Box<[Id]>
            Language::List(ids) => {
                for id in ids.iter() {
                    find_vars_recursive_helper(vec, expr, *id as usize);
                }
            }
            // [Id; 2]
            &Language::Access(ids) | &Language::AccessTranspose(ids) => {
                for id in ids.iter() {
                    find_vars_recursive_helper(vec, expr, *id as usize);
                }
            }
            // [Id; 3]
            &Language::AccessMoveAxis(ids) => {
                for id in ids.iter() {
                    find_vars_recursive_helper(vec, expr, *id as usize);
                }
            }
            // [Id; 4]
            &Language::SystolicArray(ids) => {
                for id in ids.iter() {
                    find_vars_recursive_helper(vec, expr, *id as usize);
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
            | &Language::AccessShiftRight(_) => panic!("{:#?} not implemented", expr.as_ref()[id]),
        }
    }

    let mut vec = Vec::default();
    find_vars_recursive_helper(&mut vec, expr, id);

    todo!("Actually generate the code");

    vec
}

/// Returns signature and code.
pub fn codegen(expr: &Expr, id: usize) -> (String, String) {
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

    out.push_str("}");
    out.push_str("\n");

    (signature, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::prelude::*;
    use std::process::Command;
    use std::str::FromStr;

    #[test]
    fn mlp() {
        const LIBRARY_FILENAME_C: &str = "mlp.c";
        const LIBRARY_FILENAME_O: &str = "mlp.o";
        const MAIN_FILENAME: &str = "main.c";

        let program = "
     (systolic-array 128 16
      (access
       (systolic-array 64 128
        (access
         (systolic-array 32 64
          (access (access-tensor in) 0)
          (access
           (access-transpose
            (access-move-axis (access (access-tensor weight0) 1) 1 0)
            (list 1 0)
           )
           0
          )
         )
         0
        )
        (access
         (access-transpose
          (access-move-axis (access (access-tensor weight1) 1) 1 0)
          (list 1 0)
         )
         0
        )
       )
       0
      )
      (access
       (access-transpose
        (access-move-axis (access (access-tensor weight2) 1) 1 0)
        (list 1 0)
       )
       0
      )
    )
     ";

        let expr = RecExpr::from_str(program).unwrap();
        let (signature, program) = codegen(&expr, expr.as_ref().len() - 1);
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
