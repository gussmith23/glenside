use egg::EGraph;
use egg::RecExpr;
use glenside::codegen::*;
use glenside::language::MyAnalysis;
use std::collections::HashMap;
use std::fs::File;
use std::io::prelude::*;
use std::process::Command;
use std::str::FromStr;

#[test]
fn mlp() {
    // TODO(@gussmith23) This test should use temporary files
    const LIBRARY_FILENAME_C: &str = "mlp.c";
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

    // Get hardware design
    let (hw_map, _atoms) = create_hardware_design_no_sharing(&egraph);

    let code = codegen(&egraph, id, &hw_map, "mlp", "");
    println!("{}", code);

    let mut file = File::create(LIBRARY_FILENAME_C).unwrap();
    file.write_all(code.as_bytes()).unwrap();

    File::create(MAIN_FILENAME)
            .expect("Couldn't create main file")
            .write_all(
                format! {"
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include \"{}\"

// Generated with mlp_c_gen_helper.py
float input[2] = {{ 0.07359921, 0.77422889 }};
float weights_0[2][4] = {{ {{ 0.8101289 , 0.6391721 , 0.38471687, 0.43476797 }},
 {{ 0.19664564, 0.67206388, 0.23291401, 0.59172156 }} }};
float weights_1[4][6] = {{ {{ 0.98278998, 0.61484222, 0.67881745, 0.93680619, 0.86837485, 0.42437781 }},
 {{ 0.45952102, 0.15076425, 0.52289059, 0.45032712, 0.80552555, 0.3883881  }},
 {{ 0.08942791, 0.76285151, 0.6280041 , 0.31396817, 0.80541261, 0.86247733 }},
 {{ 0.28771746, 0.1725068 , 0.18405514, 0.33522804, 0.59388266, 0.53529322 }} }};
float weights_2[6][2] = {{ {{ 0.5403619 , 0.34246081 }},
 {{ 0.96203926, 0.50463116 }},
 {{ 0.21667461, 0.7079291  }},
 {{ 0.04013655, 0.3549699  }},
 {{ 0.4628509 , 0.52579893 }},
 {{ 0.28693871, 0.41747579 }} }};
float out[2] = {{ 0., 0. }};
float expected_out[2] = {{ 1.67773846, 2.05100039 }};

int main() {{
  mlp(&out[0], &input[0], &weights_0[0][0], &weights_1[0][0], &weights_2[0][0]);

  // Ensure result is what we expect.
  for (int dim_0 = 0; dim_0 < 2; ++dim_0) {{
    fprintf(stderr, \"%f ?= %f\\n\", out[dim_0],expected_out[dim_0]);
    assert(fabs(out[dim_0] - expected_out[dim_0]) < 0.00001);
  }}

  return 0;
}}
",
               LIBRARY_FILENAME_C }
                .as_bytes(),
            )
            .expect("Couldn't write main file");

    let output = Command::new("gcc")
        .arg("-Werror")
        .arg(MAIN_FILENAME)
        .arg(format!(
            "{}/data/codegen-mlp/{}",
            env!("CARGO_MANIFEST_DIR"),
            "rtml_systolic_array_weight_stationary.c"
        ))
        .output()
        .expect("Failed to compile main file with gcc");
    assert!(
        output.status.success(),
        "Compilation failed. stderr:\n{}",
        std::str::from_utf8(output.stderr.as_slice()).expect("Could not convert stderr to UTF8")
    );

    let output = Command::new("./a.out")
        .output()
        .expect("Failed to run result");
    assert!(
        output.status.success(),
        "Main binary failed with code {:?}. stderr:\n{}",
        output.status.code(),
        std::str::from_utf8(output.stderr.as_slice()).expect("Could not convert stderr to UTF8")
    );
}
