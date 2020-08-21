use crate::hw_design_language::*;
use crate::language::Language;
use crate::language::MyAnalysis;
use crate::language::MyAnalysisData;
use egg::EGraph;
use egg::Id;
use itertools::Itertools;
use ndarray::Dimension;
use ndarray::IxDyn;
use rand::Rng;
use std::collections::HashMap;

type Expr = EGraph<Language, MyAnalysis>;

static SYSTOLIC_ARRAY_SIGNATURE: &str = "
extern void rtml_systolic_array_weight_stationary(
  int hardware_id,
  float * out,
  float * activations,
  float * weights,
  int input_vector_size,
  int output_vector_size,
  int batch);
";

/// Creates a representation (currently just a string with a C declaration) of
/// an allocation, given the name, shape, dtype, and any prefix.
pub fn create_allocation_str(prefix: &str, name: &str, shape: &[usize], dtype: DType) -> String {
    let dtype_str = match dtype {
        DType::Fp32 => "float",
        _ => panic!(),
    };

    format!(
        "{} {} {}{};",
        prefix,
        dtype_str,
        name,
        itertools::join(shape.iter().map(|dim: &usize| format!("[{}]", *dim)), "")
    )
}

/// Creates a representation (currently just a string with a C definition) of
/// an allocation and an assignment.
/// ```
/// use glenside::hw_design_language::DType::Fp32;
/// use glenside::codegen::create_assignment_str;
/// assert_eq!(
///     create_assignment_str(
///         "my_prefix",
///         "a",
///         Fp32,
///         &ndarray::ArrayD::from_shape_vec(vec![2, 3], (0..6).collect())
///             .unwrap()
///             .view()
///     ),
///     "my_prefix float a[2][3] = {{0, 1, 2}, {3, 4, 5}};"
/// );
///
/// ```
pub fn create_assignment_str<A: std::fmt::Display>(
    prefix: &str,
    name: &str,
    dtype: DType,
    array: &ndarray::ArrayViewD<A>,
) -> String {
    let mut allocation_string = create_allocation_str(prefix, name, array.shape(), dtype);
    // TODO(@gussmith23) This is a hack
    // Instead of this, create_allocation_str shouldn't return something with a
    // ; at the end.
    assert_eq!(allocation_string.chars().last().unwrap(), ';');

    // Cut off the ;
    allocation_string.truncate(allocation_string.len() - 1);

    fn recursive_helper<A: std::fmt::Display>(array: &ndarray::ArrayViewD<A>) -> String {
        if array.ndim() == 0 {
            array.to_string()
        } else {
            format!(
                "{{{}}}",
                array
                    .axis_iter(ndarray::Axis(0))
                    .map(|a| recursive_helper(&a))
                    .join(", ")
            )
        }
    }

    format!("{} = {};", allocation_string, recursive_helper(array))
}

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
                    name: format!("systolic_array_{}", hw_id),
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

/// Returns c code.
// TODO(@gussmith23) Does not reason about ordering on hardware.
// TODO(@gussmith23) Hardcoded to float32
pub fn codegen(
    expr: &Expr,
    id: Id,
    hw_map: &HashMap<Id, usize>,
    function_name: &str,
    allocations_prefix: &str,
) -> String {
    let mut declarations = String::default();
    let mut code = String::default();
    codegen_recursive_helper(
        expr,
        id,
        id,
        allocations_prefix,
        &mut declarations,
        &mut code,
        hw_map,
    )
    .as_str();

    let mut signature = format!("void {}(", function_name);
    // TODO(@gussmith23) Assuming the output is a tensor
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

    let mut out = String::default();

    out.push_str(SYSTOLIC_ARRAY_SIGNATURE);
    out.push_str("\n");

    out.push_str(declarations.as_str());
    out.push_str("\n");

    out.push_str(signature.as_str());
    out.push_str("{");
    out.push_str("\n");

    out.push_str(code.as_str());

    out.push_str("}");
    out.push_str("\n");

    out
}

/// allocations_prefix: string to prefix all allocations/declarations with
fn codegen_recursive_helper(
    expr: &Expr,
    id: Id,
    top_level_id: Id,
    allocations_prefix: &str,
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
            let symbol = codegen_recursive_helper(
                expr,
                symbol_id,
                top_level_id,
                allocations_prefix,
                declarations,
                code,
                hw_map,
            );
            symbol
        }
        &Language::Access([access_tensor_id, axis_id]) => {
            let axis = MyAnalysis::get_usize(axis_id, expr);
            assert_eq!(axis, 0);
            codegen_recursive_helper(
                expr,
                access_tensor_id,
                top_level_id,
                allocations_prefix,
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

            let s0 = codegen_recursive_helper(
                expr,
                a0_id,
                top_level_id,
                allocations_prefix,
                declarations,
                code,
                hw_map,
            );
            let s1 = codegen_recursive_helper(
                expr,
                a1_id,
                top_level_id,
                allocations_prefix,
                declarations,
                code,
                hw_map,
            );

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
                // TODO(@gussmith23) Allocations should not be done ad-hoc
                declarations.push_str(
                    create_allocation_str(
                        allocations_prefix,
                        out_var_name.as_str(),
                        this_access
                            .shape
                            .slice()
                            .iter()
                            .chain(this_access.item_shape.slice().iter())
                            .cloned()
                            .collect::<Vec<_>>()
                            .as_slice(),
                        DType::Fp32,
                    )
                    .as_str(),
                );

                out_var_name
            };

            code.push_str(
                format!(
                    "rtml_systolic_array_weight_stationary(
                       {}, {}, {}, {}, {}, {}, {});\n",
                    // Hardware ID
                    hw_map.get(&id).unwrap(),
                    // Pointer to output
                    format!("&{}[0]", out_var_name,),
                    // Pointer to input vector
                    format!("&{}[0]", s0,),
                    // Pointer to input matrix
                    format!("&{}[0]", s1,),
                    // Length of input vector/size of input matrix dim 0
                    rows,
                    // Size of input matrix dim 1/length of output vector
                    cols,
                    // Batch size, or, the number of vectors to push through.
                    // Changing this is how we implement matrix-matrix
                    // multiplication.
                    1
                )
                .as_str(),
            );

            out_var_name
        }
        &Language::Usize(u) => format!("{}", u),
        &Language::AccessTranspose([access_id, list_id]) => {
            let access = match &expr[access_id].data {
                MyAnalysisData::AccessPattern(a) => a,
                _ => panic!(),
            };
            let original_shape = access
                .shape
                .slice()
                .iter()
                .chain(access.item_shape.slice().iter())
                .cloned()
                .collect::<Vec<_>>();

            let new_axis_order = match &expr[list_id].data {
                MyAnalysisData::List(l) => l,
                _ => panic!(),
            };

            assert_eq!(original_shape.len(), new_axis_order.len());

            let new_shape = new_axis_order
                .iter()
                .map(|i| original_shape[*i])
                .collect::<Vec<_>>();

            let access_var_name = codegen_recursive_helper(
                expr,
                access_id,
                top_level_id,
                allocations_prefix,
                declarations,
                code,
                hw_map,
            );

            // Declare transpose output
            // TODO(@gussmith23) Having to check this at every stage is not sustainable.
            let transpose_out_var_name: String = if id == top_level_id {
                "out".to_string()
            } else {
                // TODO(@gussmith23) Find a different way to name intermediates
                // Currently generating random strings. Not great IMO.
                let out = format!(
                    "transpose_out_{}",
                    rand::thread_rng()
                        .sample_iter(&rand::distributions::Alphanumeric)
                        .take(30)
                        .collect::<String>()
                );
                declarations.push_str(
                    create_allocation_str(
                        allocations_prefix,
                        out.as_str(),
                        original_shape.as_slice(),
                        DType::Fp32,
                    )
                    .as_str(),
                );
                out
            };

            let index_var_names = (0..original_shape.len())
                .map(|i| format!("i{}", i))
                .collect::<Vec<_>>();

            // Create a for loop for every dimension in the shape.
            for (dim_index, dim_len) in original_shape.iter().enumerate() {
                let index_var_name = &index_var_names[dim_index];
                code.push_str(
                    format!(
                        "
int {};
for ({} = 0; {} < {}; {}++) {{",
                        index_var_name, index_var_name, index_var_name, dim_len, index_var_name
                    )
                    .as_str(),
                );
            }

            let index_var_names_reordered = new_axis_order
                .iter()
                .map(|i| &index_var_names[*i])
                .collect::<Vec<_>>();

            // Within the innermost for loop: assign to the output at the
            // correct location.
            // We have indices i0..i(n-1), for each of the n axes.
            code.push_str(
                format!(
                    "
{}[{}] = {}[{}];",
                    transpose_out_var_name,
                    (0..original_shape.len())
                        .map(|i| format!(
                            "{}*({})",
                            index_var_names_reordered[i],
                            new_shape[i + 1..].iter().product::<usize>()
                        ))
                        .collect::<Vec<_>>()
                        .join(" + "),
                    access_var_name,
                    (0..original_shape.len())
                        .map(|i| format!(
                            "{}*({})",
                            index_var_names[i],
                            original_shape[i + 1..].iter().product::<usize>()
                        ))
                        .collect::<Vec<_>>()
                        .join(" + ")
                )
                .as_str(),
            );

            // Close each for loop
            for _ in original_shape.iter() {
                code.push_str("}");
            }

            transpose_out_var_name
        }
        &Language::GetAccessShape(_)
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
    use egg::RecExpr;
    use std::fs::File;
    use std::io::Write;
    use std::process::Command;
    use std::str::FromStr;

    #[test]
    fn transpose() {
        let shape = vec![1, 20, 300, 3];
        let permutation = vec![3, 1, 0, 2];
        let input = ndarray::ArrayD::from_shape_vec(
            shape.clone(),
            (0..shape.iter().product::<usize>()).collect(),
        )
        .unwrap();
        let input_transposed = input.clone().permuted_axes(permutation.clone());

        let expr = RecExpr::from_str(
            format!(
                "
(access-transpose (access-tensor t) (list {}))",
                permutation.iter().map(|x| x.to_string()).join(" ")
            )
            .as_str(),
        )
        .unwrap();

        let mut map = HashMap::default();
        map.insert("t".to_string(), shape.clone());

        let mut egraph = EGraph::new(MyAnalysis { name_to_shape: map });
        let id = egraph.add_expr(&expr);

        let code = codegen(&egraph, id, &HashMap::default(), "transpose", "");

        let main_code = format!(
            "
#include <assert.h>

{}
{}
{}
{}

int main() {{
  transpose(&out[0], &a[0]);

  int i;
  for (i = 0; i < {}; i++) {{
    assert(((float*)a_t)[i] == ((float*)out)[i]);
  }}
}}
",
            create_assignment_str("", "a", DType::Fp32, &input.view()),
            create_assignment_str("", "a_t", DType::Fp32, &input_transposed.view()),
            create_assignment_str(
                "",
                "out",
                DType::Fp32,
                &ndarray::ArrayD::<f32>::zeros(input_transposed.shape()).view()
            ),
            code,
            shape.iter().product::<usize>()
        );

        println!("{}", main_code);

        let main_c_filepath = std::env::temp_dir().with_file_name(format!(
            "transpose-test-{}.c",
            std::time::SystemTime::now().elapsed().unwrap().as_nanos()
        ));

        let binary_filepath = std::env::temp_dir().with_file_name(format!(
            "transpose-test-{}",
            std::time::SystemTime::now().elapsed().unwrap().as_nanos()
        ));
        println!("{}", binary_filepath.to_string_lossy());

        File::create(&main_c_filepath)
            .unwrap()
            .write_all(main_code.as_bytes())
            .unwrap();

        let result = Command::new("gcc")
            .arg("-g")
            .arg("-o")
            .arg(&binary_filepath)
            .arg(&main_c_filepath)
            .output()
            .unwrap();

        assert!(
            result.status.success(),
            "{}",
            std::str::from_utf8(result.stderr.as_slice())
                .expect("Could not convert stderr to UTF8")
        );

        let result = Command::new(&binary_filepath).output().unwrap();

        assert!(
            result.status.success(),
            "{}",
            std::str::from_utf8(result.stderr.as_slice())
                .expect("Could not convert stderr to UTF8")
        );
    }
}
