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
// TODO(@gussmith23) Does not reason about ordering on hardware.
// TODO(@gussmith23) Hardcoded to float32
pub fn codegen(
    expr: &Expr,
    id: Id,
    hw_map: &HashMap<Id, usize>,
    function_name: &str,
) -> (String, String) {
    let mut out = String::default();

    let mut declarations = String::default();
    let mut code = String::default();
    codegen_recursive_helper(expr, id, id, "", &mut declarations, &mut code, hw_map).as_str();

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
