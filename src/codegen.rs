use crate::language::Language;
use egg::RecExpr;

type Expr = RecExpr<Language>;

pub fn codegen<Id>(_expr: &Expr, _id: Id) -> String {
    String::default()
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
(compute dot-product
 (access-cartesian-product
  (access
   (compute dot-product
    (access-cartesian-product
     (access
      (compute dot-product
       (access-cartesian-product
        (access (access-tensor v-32) 0)
        (access-move-axis (access (access-tensor t-32-64) 1) 1 0)
       )
      )
      0
     )
     (access-move-axis (access (access-tensor t-64-128) 1) 1 0)
    )
   )
   0
  )
  (access-move-axis (access (access-tensor t-128-16) 1) 1 0)
 )
)
     ";

        let expr = RecExpr::from_str(program).unwrap();
        let out = codegen(&expr, expr.as_ref().len() - 1);

        let mut file = File::create(LIBRARY_FILENAME_C).unwrap();
        file.write_all(out.as_bytes()).unwrap();

        let output = Command::new("gcc")
            .arg("-c")
            .arg(LIBRARY_FILENAME_C)
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
                b"
extern void mlp();

int main() {
  mlp();
  return 0;
}
",
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
