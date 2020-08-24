use crate::hw_design_language::*;
use crate::language::MyAnalysis;
use crate::language::MyAnalysisData;
use crate::language::{Language, PadType};
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
            &Language::AccessConcatenate(ids) => {
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
            // [Id; 5]
            &Language::AccessPad(ids) => {
                for id in ids.iter() {
                    find_vars_recursive_helper(vec, expr, *id);
                }
            }
            &Language::Usize(_) | &Language::PadType(_) => (),
            &Language::GetAccessShape(_)
            | &Language::AccessBroadcast(_)
            | &Language::AccessInsertAxis(_)
            | &Language::AccessPair(_)
            | &Language::AccessSqueeze(_)
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
    let out_symbol = codegen_recursive_helper(
        expr,
        id,
        id,
        allocations_prefix,
        &mut declarations,
        &mut code,
        hw_map,
    );

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

    // Copy value into "out" variable
    // Get length of array
    let length = match &expr[id].data {
        MyAnalysisData::AccessPattern(a) => a
            .shape
            .slice()
            .iter()
            .chain(a.item_shape.slice().iter())
            .product::<usize>(),
        _ => panic!(),
    };
    out.push_str(
        format!(
            "
for (int i = 0; i < {}; i++) {{
  ((float*)out)[i] = ((float*){})[i];
}}
",
            length, out_symbol
        )
        .as_str(),
    );

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
        &Language::Access([access_tensor_id, _axis_id]) => codegen_recursive_helper(
            expr,
            access_tensor_id,
            top_level_id,
            allocations_prefix,
            declarations,
            code,
            hw_map,
        ),
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
            assert!(this_access.shape.ndim() == 1 || this_access.shape.ndim() == 2);
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

            let out_var_name = format!(
                "systolic_array_{}_eclass_{}_out",
                hw_map.get(&id).unwrap(),
                id,
            );

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
                    if a0.shape.ndim() == 0 {
                        1
                    } else if a0.shape.ndim() == 1 {
                        a0.shape[0]
                    } else {
                        panic!()
                    }
                )
                .as_str(),
            );

            out_var_name
        }
        &Language::Usize(u) => format!("{}", u),
        &Language::AccessPad([access_id, pad_type_id, axis_id, pad_before_id, pad_after_id]) => {
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

            let pad_type = match &expr[pad_type_id].data {
                MyAnalysisData::PadType(t) => t,
                _ => panic!(),
            };

            let axis = MyAnalysis::get_usize(axis_id, expr);
            let pad_before = MyAnalysis::get_usize(pad_before_id, expr);
            let _pad_after = MyAnalysis::get_usize(pad_after_id, expr);

            let new_shape = match &expr[id].data {
                MyAnalysisData::AccessPattern(a) => a
                    .shape
                    .slice()
                    .iter()
                    .chain(a.item_shape.slice().iter())
                    .cloned()
                    .collect::<Vec<_>>(),
                _ => panic!(),
            };

            let access_var_name = codegen_recursive_helper(
                expr,
                access_id,
                top_level_id,
                allocations_prefix,
                declarations,
                code,
                hw_map,
            );

            let pad_out_var_name: String = {
                // TODO(@gussmith23) Find a different way to name intermediates
                // Currently generating random strings. Not great IMO.
                let out = format!(
                    "pad_out_{}",
                    rand::thread_rng()
                        .sample_iter(&rand::distributions::Alphanumeric)
                        .take(30)
                        .collect::<String>()
                );
                declarations.push_str(
                    create_allocation_str(
                        allocations_prefix,
                        out.as_str(),
                        new_shape.as_slice(),
                        DType::Fp32,
                    )
                    .as_str(),
                );
                out
            };

            let index_var_names = (0..new_shape.len())
                .map(|i| format!("i{}", i))
                .collect::<Vec<_>>();

            // Create a for loop for every dimension in the result shape.
            for (dim_index, dim_len) in new_shape.iter().enumerate() {
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

            // Within the innermost for loop: assign to the output at the
            // correct location.
            // We have indices i0..i(n-1), for each of the n axes.
            code.push_str(
                format!(
                    "
if (i{pad_axis} < {pad_before_index} || i{pad_axis} >= {pad_after_index}) {{
  {out_name}{out_index} = {pad_value};
}} else {{
  {out_name}{out_index} = {in_name}[{in_index}];
}}
",
                    pad_axis = axis,
                    pad_before_index = pad_before,
                    pad_after_index = pad_before + original_shape[axis],
                    out_name = pad_out_var_name,
                    out_index = (0..new_shape.len())
                        .map(|i| format!("[{}]", index_var_names[i],))
                        .collect::<Vec<_>>()
                        .join(""),
                    pad_value = match pad_type {
                        PadType::ZeroPadding => "0",
                    },
                    in_name = access_var_name,
                    in_index = (0..new_shape.len())
                        .map(|i| if i != axis {
                            format!(
                                "{}*({})",
                                index_var_names[i],
                                original_shape[i + 1..].iter().product::<usize>()
                            )
                        } else {
                            format!(
                                "({}-{})*({})",
                                index_var_names[i],
                                pad_before,
                                original_shape[i + 1..].iter().product::<usize>()
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(" + ")
                )
                .as_str(),
            );

            // Close each for loop
            for _ in original_shape.iter() {
                code.push_str("}");
            }

            pad_out_var_name
        }
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
            let transpose_out_var_name: String = {
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
((float*){})[{}] = {}[{}];",
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
        &Language::AccessReshape([access_id, _access_shape_id]) => {
            // Don't need to do anything for reshape at runtime.
            codegen_recursive_helper(
                expr,
                access_id,
                top_level_id,
                allocations_prefix,
                declarations,
                code,
                hw_map,
            )
        }
        &Language::AccessConcatenate([a0_id, a1_id, axis_id]) => {
            let axis = MyAnalysis::get_usize(axis_id, expr);
            let (a0, a1) = match (&expr[a0_id].data, &expr[a1_id].data) {
                (MyAnalysisData::AccessPattern(a0), MyAnalysisData::AccessPattern(a1)) => (a0, a1),
                _ => panic!(),
            };
            let arg_0_name = codegen_recursive_helper(
                expr,
                a0_id,
                top_level_id,
                allocations_prefix,
                declarations,
                code,
                hw_map,
            );
            let arg_1_name = codegen_recursive_helper(
                expr,
                a1_id,
                top_level_id,
                allocations_prefix,
                declarations,
                code,
                hw_map,
            );

            let concat_shape = match &expr[id].data {
                MyAnalysisData::AccessPattern(a) => a
                    .shape
                    .slice()
                    .iter()
                    .chain(a.item_shape.slice().iter())
                    .cloned()
                    .collect::<Vec<_>>(),
                _ => panic!("Expected access pattern"),
            };
            let a0_shape = a0
                .shape
                .slice()
                .iter()
                .chain(a0.item_shape.slice().iter())
                .cloned()
                .collect::<Vec<_>>();
            let a1_shape = a1
                .shape
                .slice()
                .iter()
                .chain(a1.item_shape.slice().iter())
                .cloned()
                .collect::<Vec<_>>();

            let out_var_name: String = {
                // TODO(@gussmith23) Find a different way to name intermediates
                // Currently generating random strings. Not great IMO.
                let out = format!(
                    "concat_out_{}",
                    rand::thread_rng()
                        .sample_iter(&rand::distributions::Alphanumeric)
                        .take(30)
                        .collect::<String>()
                );
                declarations.push_str(
                    create_allocation_str(
                        allocations_prefix,
                        out.as_str(),
                        concat_shape.as_slice(),
                        DType::Fp32,
                    )
                    .as_str(),
                );
                out
            };

            // Iterate over each dim in the output
            for (i, dim_val) in concat_shape.iter().enumerate() {
                code.push_str(
                    format!(
                        "
int i{i};
for (i{i} = 0; i{i} < {dim_val}; i{i} ++) {{
",
                        i = i,
                        dim_val = dim_val
                    )
                    .as_str(),
                );
            }

            code.push_str(
                format!(
                    "
if (i{} < {}) {{
  ((float*){})[{}] = {}[{}];
}} else {{
  ((float*){})[{}] = {}[{}];
}}
",
                    axis,
                    a0[axis],
                    out_var_name,
                    (0..(concat_shape.len()))
                        .map(|i| format!(
                            "i{}*{}",
                            i,
                            concat_shape[i + 1..].iter().product::<usize>()
                        ))
                        .join(" + "),
                    arg_0_name,
                    (0..(concat_shape.len()))
                        .map(|i| format!("i{}*{}", i, a0_shape[i + 1..].iter().product::<usize>()))
                        .join(" + "),
                    out_var_name,
                    (0..(concat_shape.len()))
                        .map(|i| format!(
                            "i{}*{}",
                            i,
                            concat_shape[i + 1..].iter().product::<usize>()
                        ))
                        .join(" + "),
                    arg_1_name,
                    (0..(concat_shape.len()))
                        .map(|i| if i != axis {
                            format!("i{}*{}", i, a1_shape[i + 1..].iter().product::<usize>())
                        } else {
                            format!(
                                "(i{}-{})*{}",
                                i,
                                a0_shape[i],
                                a1_shape[i + 1..].iter().product::<usize>()
                            )
                        })
                        .join(" + "),
                )
                .as_str(),
            );

            for _ in concat_shape.iter().enumerate() {
                code.push_str("}");
            }

            out_var_name
        }
        &Language::GetAccessShape(_)
        | Language::List(_)
        | &Language::AccessBroadcast(_)
        | &Language::AccessInsertAxis(_)
        | &Language::AccessPair(_)
        | &Language::AccessSqueeze(_)
        | Language::PadType(_)
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
        | &Language::AccessFlatten(_)
        | &Language::AccessShape(_)
        | &Language::AccessSlice(_)
        | &Language::AccessShiftRight(_) => panic!("{:#?} not implemented", expr[id].nodes[0]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egg::RecExpr;
    use std::fs::File;
    use std::io::Write;
    use std::path::PathBuf;
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

    #[test]
    fn concat() {
        let shape0 = vec![2, 10, 50, 3];
        let shape1 = vec![2, 3, 50, 3];
        let concat_axis = 1;

        let input0 = ndarray::ArrayD::from_shape_vec(
            shape0.clone(),
            (0..shape0.iter().product::<usize>()).collect(),
        )
        .unwrap();
        let input1 = ndarray::ArrayD::from_shape_vec(
            shape1.clone(),
            (0..shape1.iter().product::<usize>()).collect(),
        )
        .unwrap();
        let concatted =
            ndarray::stack(ndarray::Axis(concat_axis), &[input0.view(), input1.view()]).unwrap();

        let expr = RecExpr::from_str(
            format!(
                "
(access-concatenate (access-tensor t0) (access-tensor t1) {})",
                concat_axis
            )
            .as_str(),
        )
        .unwrap();

        let mut map = HashMap::default();
        map.insert("t0".to_string(), shape0.clone());
        map.insert("t1".to_string(), shape1.clone());

        let mut egraph = EGraph::new(MyAnalysis { name_to_shape: map });
        let id = egraph.add_expr(&expr);

        let code = codegen(&egraph, id, &HashMap::default(), "concatenate", "");

        let main_code = format!(
            "
#include <assert.h>

{}
{}
{}
{}
{}

int main() {{
  concatenate((float*) out, (float*) t0, (float*) t1);

  int i;
  for (i = 0; i < {}; i++) {{
    assert(((float*)a_t)[i] == ((float*)out)[i]);
  }}
}}
",
            create_assignment_str("", "t0", DType::Fp32, &input0.view()),
            create_assignment_str("", "t1", DType::Fp32, &input1.view()),
            create_assignment_str("", "a_t", DType::Fp32, &concatted.view()),
            create_assignment_str(
                "",
                "out",
                DType::Fp32,
                &ndarray::ArrayD::<f32>::zeros(concatted.shape()).view()
            ),
            code,
            concatted.shape().iter().product::<usize>()
        );

        let main_c_filepath = std::env::temp_dir().with_file_name(format!(
            "concatenate-test-{}.c",
            std::time::SystemTime::now().elapsed().unwrap().as_nanos()
        ));
        println!("{}", main_c_filepath.to_string_lossy());

        let binary_filepath = std::env::temp_dir().with_file_name(format!(
            "concatenate-test-{}",
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

    #[test]
    fn systolic_array() {
        let shape0 = vec![2, 10];
        let shape1 = vec![10, 15];

        let input0 = ndarray::ArrayD::from_shape_vec(
            shape0.clone(),
            (0..shape0.iter().product::<usize>()).collect(),
        )
        .unwrap()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();
        let input1 = ndarray::ArrayD::from_shape_vec(
            shape1.clone(),
            (0..shape1.iter().product::<usize>()).collect(),
        )
        .unwrap()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();
        let multiplied = input0.dot(&input1).into_dyn();

        let expr = RecExpr::from_str(
            "
(systolic-array 10 15
 (access (access-tensor t0) 1)
 (access (access-tensor t1) 0)
)",
        )
        .unwrap();

        let mut map = HashMap::default();
        map.insert("t0".to_string(), shape0.clone());
        map.insert("t1".to_string(), shape1.clone());

        let mut egraph = EGraph::new(MyAnalysis { name_to_shape: map });
        let id = egraph.add_expr(&expr);

        let mut hw_map = HashMap::default();
        hw_map.insert(id, 0);

        let code = codegen(&egraph, id, &hw_map, "systolic_array", "");

        let main_code = format!(
            "
#include <assert.h>
#include \"{}\"

{}
{}
{}
{}
{}

int main() {{
  systolic_array((float*) out, (float*) t0, (float*) t1);

  int i;
  for (i = 0; i < {}; i++) {{
    assert(((float*)result)[i] == ((float*)out)[i]);
  }}
}}
",
            PathBuf::from_str(
                format!(
                    "{}/{}/{}/{}",
                    env!("CARGO_MANIFEST_DIR"),
                    "data",
                    "codegen-mlp",
                    "rtml_systolic_array_weight_stationary.c"
                )
                .as_str()
            )
            .unwrap()
            .to_string_lossy(),
            create_assignment_str("", "t0", DType::Fp32, &input0.into_dyn().view()),
            create_assignment_str("", "t1", DType::Fp32, &input1.into_dyn().view()),
            create_assignment_str("", "result", DType::Fp32, &multiplied.view()),
            create_assignment_str(
                "",
                "out",
                DType::Fp32,
                &ndarray::ArrayD::<f32>::zeros(multiplied.shape()).view()
            ),
            code,
            multiplied.shape().iter().product::<usize>()
        );

        let main_c_filepath = std::env::temp_dir().with_file_name(format!(
            "systolic-array-test-{}.c",
            std::time::SystemTime::now().elapsed().unwrap().as_nanos()
        ));
        println!("{}", main_c_filepath.to_string_lossy());

        let binary_filepath = std::env::temp_dir().with_file_name(format!(
            "systolic-array-test-{}",
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

    #[test]
    fn pad() {
        let shape = vec![10, 20, 3, 45];
        let pad_axis = 2;
        let pad_before = 2;
        let pad_after = 3;
        let pad_type = PadType::ZeroPadding;

        let mut pad_before_shape = shape.clone();
        pad_before_shape[pad_axis] = pad_before;
        let mut pad_after_shape = shape.clone();
        pad_after_shape[pad_axis] = pad_after;

        let input = ndarray::ArrayD::from_shape_vec(
            shape.clone(),
            (0..shape.iter().product::<usize>()).collect(),
        )
        .unwrap();

        assert!(pad_type == PadType::ZeroPadding);

        let padded = ndarray::stack(
            ndarray::Axis(pad_axis),
            &[
                ndarray::ArrayD::zeros(pad_before_shape).view(),
                input.view(),
                ndarray::ArrayD::zeros(pad_after_shape).view(),
            ],
        )
        .unwrap();

        let expr = RecExpr::from_str(
            format!(
                "
(access-pad (access-tensor t) {} {} {} {})",
                pad_type, pad_axis, pad_before, pad_after
            )
            .as_str(),
        )
        .unwrap();

        let mut map = HashMap::default();
        map.insert("t".to_string(), shape.clone());

        let mut egraph = EGraph::new(MyAnalysis { name_to_shape: map });
        let id = egraph.add_expr(&expr);

        let code = codegen(&egraph, id, &HashMap::default(), "pad", "");

        let main_code = format!(
            "
#include <assert.h>

{}
{}
{}
{}

int main() {{
  pad(&out[0], &a[0]);

  int i;
  for (i = 0; i < {}; i++) {{
    assert(((float*)a_pad)[i] == ((float*)out)[i]);
  }}
}}
",
            create_assignment_str("", "a", DType::Fp32, &input.view()),
            create_assignment_str("", "a_pad", DType::Fp32, &padded.view()),
            create_assignment_str(
                "",
                "out",
                DType::Fp32,
                &ndarray::ArrayD::<f32>::zeros(padded.shape()).view()
            ),
            code,
            shape.iter().product::<usize>()
        );

        let main_c_filepath = std::env::temp_dir().with_file_name(format!(
            "pad-test-{}.c",
            std::time::SystemTime::now().elapsed().unwrap().as_nanos()
        ));

        let binary_filepath = std::env::temp_dir().with_file_name(format!(
            "pad-test-{}",
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
