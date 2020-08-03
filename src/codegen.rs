use crate::hw_design_language::*;
use crate::language::Language;
use crate::language::MyAnalysis;
use crate::language::MyAnalysisData;
use egg::EGraph;
use egg::Id;
use itertools::Itertools;
use ndarray::Dimension;
use ndarray::IxDyn;
use std::collections::HashMap;

type Expr = EGraph<Language, MyAnalysis>;

static SYSTOLIC_ARRAY_SIGNATURE: &str = "
extern void rtml_systolic_array_weight_stationary(
  int hardware_id,
  float * out,
  float * activations,
  int activations_width,
  float * weights,
  int weights_width,
  int weights_height);
";

/// Create a hardware design from an expression, creating a unique atom for each
/// unique node (eclass or  id).
pub fn create_hardware_design_no_sharing(expr: &Expr) -> (HashMap<Id, usize>, Vec<Atom>) {
    let mut hw_id = 0;
    let mut map = HashMap::new();
    let mut atoms = Vec::new();

    for eclass in expr.classes() {
        assert_eq!(eclass.nodes.len(), 1);
        match &eclass.nodes[0] {
            &Language::SystolicArray([row_id, col_id, _, _]) => {
                hw_id += 1;
                let hw_id = hw_id - 1;

                let row = match {
                    assert_eq!(expr[row_id].nodes.len(), 1);
                    &expr[row_id].nodes[0]
                } {
                    Language::Usize(u) => u,
                    _ => panic!(),
                };
                let col = match {
                    assert_eq!(expr[col_id].nodes.len(), 1);
                    &expr[col_id].nodes[0]
                } {
                    Language::Usize(u) => u,
                    _ => panic!(),
                };

                map.insert(eclass.id, hw_id);
                atoms.push(Atom {
                    name: format!("multiplier{}", hw_id),
                    id: hw_id,
                    config: AtomConfig::SystolicArrayWeightStationary(
                        SystolicArrayWeightStationaryParams {
                            // TODO(@gussmith23) hardcoded datatype
                            dtype: DType::Fp32,
                            rows: *row,
                            cols: *col,
                        },
                    ),
                });
            }
            _ => (),
        }
    }

    (map, atoms)
}

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
pub fn codegen(
    expr: &Expr,
    id: Id,
    hw_map: &HashMap<Id, usize>,
    function_name: &str,
) -> (String, String) {
    let mut out = String::default();

    let mut declarations = String::default();
    let mut code = String::default();
    codegen_recursive_helper(expr, id, id, &mut declarations, &mut code, hw_map).as_str();

    out.push_str(declarations.as_str());
    out.push_str("\n");

    let mut signature = format!("void {}(", function_name);
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

    out.push_str(SYSTOLIC_ARRAY_SIGNATURE);
    out.push_str("\n");

    out.push_str(signature.as_str());

    out.push_str("\n");
    out.push_str("{");
    out.push_str("\n");

    out.push_str(code.as_str());

    out.push_str("}");
    out.push_str("\n");

    (signature, out)
}

fn codegen_recursive_helper(
    expr: &Expr,
    id: Id,
    top_level_id: Id,
    declarations: &mut String,
    code: &mut String,
    hw_map: &HashMap<Id, usize>,
) -> String {
    match {
        assert_eq!(expr[id].nodes.len(), 1);
        &expr[id].nodes[0]
    } {
        Language::Symbol(s) => s.clone(),
        &Language::AccessTensor(symbol_id) => {
            let symbol =
                codegen_recursive_helper(expr, symbol_id, top_level_id, declarations, code, hw_map);
            symbol
        }
        &Language::Access([access_tensor_id, axis_id]) => {
            let axis = MyAnalysis::get_usize(axis_id, expr);
            assert_eq!(axis, 0);
            codegen_recursive_helper(
                expr,
                access_tensor_id,
                top_level_id,
                declarations,
                code,
                hw_map,
            )
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

            let s0 =
                codegen_recursive_helper(expr, a0_id, top_level_id, declarations, code, hw_map);
            let s1 =
                codegen_recursive_helper(expr, a1_id, top_level_id, declarations, code, hw_map);

            let out_var_name = if id == top_level_id {
                "out".to_string()
            } else {
                let out_var_name = format!(
                    "systolic_array_{}_eclass_{}_out",
                    hw_map.get(&id).unwrap(),
                    id,
                );

                // TODO(@gussmith23) How to generate output buffer?
                // This seems like it might not be legal, just declaring it.
                // TODO(@gussmith23) how to assign unique names to each usage?
                declarations.push_str(
                    format!(
                        "float {}[{}];\n",
                        out_var_name,
                        this_access.shape.slice()[0]
                    )
                    .as_str(),
                );

                out_var_name
            };

            code.push_str(
                format!(
                    "rtml_systolic_array_weight_stationary(
                       // Hardware ID
                       {},
                       {}, {}, {}, {}, {}, {});\n",
                    hw_map.get(&id).unwrap(),
                    format!("&{}[0]", out_var_name,),
                    format!("&{}[0]", s0,),
                    rows,
                    format!("&{}[0]", s1,),
                    rows,
                    cols
                )
                .as_str(),
            );

            out_var_name
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

        // Get hardware design
        let (hw_map, _atoms) = create_hardware_design_no_sharing(&egraph);

        let (signature, program) = codegen(&egraph, id, &hw_map, "mlp");
        println!("{}", program);

        let mut file = File::create(LIBRARY_FILENAME_C).unwrap();
        file.write_all(program.as_bytes()).unwrap();

        let output = Command::new("gcc")
            .arg("-c")
            .arg(LIBRARY_FILENAME_C)
            .arg("-O0")
            .arg("-Werror")
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
#include <assert.h>
#include <stdio.h>
#include <math.h>

extern {};

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
                signature}
                .as_bytes(),
            )
            .expect("Couldn't write main file");

        let output = Command::new("gcc")
            .arg("-Werror")
            .arg(MAIN_FILENAME)
            .arg(LIBRARY_FILENAME_O)
            .arg("rtml_systolic_array_weight_stationary.c")
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
            "Main binary failed with code {:?}. stderr:\n{}",
            output.status.code(),
            std::str::from_utf8(output.stderr.as_slice())
                .expect("Could not convert stderr to UTF8")
        );
    }
}
