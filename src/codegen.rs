use crate::hw_design_language::*;
use crate::language::MyAnalysis;
use crate::language::MyAnalysisData;
use crate::language::RelayActivationLayout;
use crate::language::{Language, PadType, RelayOperator};
use egg::EGraph;
use egg::Id;
use itertools::Itertools;
use log::warn;
use ndarray::Dimension;
use ndarray::IxDyn;
use ndarray::{Array, ArrayD};
use ndarray_npy::{read_npy, write_npy};
use rand::Rng;
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::iter::FromIterator;
use std::process::Command;

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

/// Gives the signature of a C array given the datatype, name, and shape. Useful
/// for declaring arrays and array-type function arguments.
/// ```
/// assert_eq!(
///     glenside::codegen::c_array_string(
///         "my_array",
///         &[3, 4, 5],
///         glenside::hw_design_language::DType::Fp32
///     ),
///     "float my_array[3][4][5]"
/// );
/// ```
pub fn c_array_string(name: &str, shape: &[usize], dtype: DType) -> String {
    assert_eq!(DType::Fp32.to_c_type_string(), "float");
    format!(
        "{} {}{}",
        dtype.to_c_type_string(),
        name,
        itertools::join(shape.iter().map(|dim: &usize| format!("[{}]", *dim)), "")
    )
}

/// Creates a representation (currently just a string with a C declaration) of
/// an allocation, given the name, shape, dtype, and any prefix.
/// ```
/// assert_eq!(
///     glenside::codegen::c_allocation_string(
///         "__my_prefix",
///         "my_array",
///         &[3, 4, 5],
///         glenside::hw_design_language::DType::Fp32
///     ),
///     "__my_prefix float my_array[3][4][5];"
/// );
/// ```
pub fn c_allocation_string(prefix: &str, name: &str, shape: &[usize], dtype: DType) -> String {
    format!("{} {};", prefix, c_array_string(name, shape, dtype))
}

/// Creates a representation (currently just a string with a C definition) of
/// an allocation and an assignment.
/// ```
/// assert_eq!(
///     glenside::codegen::c_assignment_string(
///         "my_prefix",
///         "a",
///         glenside::hw_design_language::DType::Fp32,
///         &ndarray::ArrayD::from_shape_vec(vec![2, 3], (0..6).collect())
///             .unwrap()
///             .view()
///     ),
///     "my_prefix float a[2][3] = {{0, 1, 2}, {3, 4, 5}};"
/// );
///
/// ```
pub fn c_assignment_string<A: std::fmt::Display>(
    prefix: &str,
    name: &str,
    dtype: DType,
    array: &ndarray::ArrayViewD<A>,
) -> String {
    let mut allocation_string = c_allocation_string(prefix, name, array.shape(), dtype);
    // TODO(@gussmith23) This is a hack
    // Instead of this, c_allocation_string shouldn't return something with a
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

// TODO(@gussmith23) Turn on rustdoc lints
// https://doc.rust-lang.org/rustdoc/lints.html

/// Create a monolithic systolic array hardware design from an expression, where
/// we use just one systolic array.
/// (row, col): the rows/cols of the systolic array.
///
/// ```
/// use glenside::hw_design_language::*;
/// use glenside::language::*;
/// use glenside::codegen::*;
/// use egg::*;
/// use std::collections::HashMap;
/// use std::str::FromStr;
///
/// let expr = RecExpr::from_str(
///     "
///     (systolic-array 32 32
///      (access
///       (systolic-array 32 32
///        (access (access-tensor t0) 1)
///        (access (access-tensor t1) 0)
///       )
///       1
///      )
///      (access
///       (systolic-array 32 32
///        (access (access-tensor t2) 1)
///        (access (access-tensor t3) 0)
///       )
///       0
///      )
///     )",
/// )
/// .unwrap();
///
/// let mut map = HashMap::default();
/// map.insert("t0".to_string(), vec![32, 32]);
/// map.insert("t1".to_string(), vec![32, 32]);
/// map.insert("t2".to_string(), vec![32, 32]);
/// map.insert("t3".to_string(), vec![32, 32]);
///
/// let mut egraph = EGraph::new(MyAnalysis { name_to_shape: map });
/// egraph.add_expr(&expr);
///
/// let (hw_map, hw_design) = create_hardware_design_monolithic(&egraph, (32, 32));
///
/// assert_eq!(hw_design.len(), 1);
/// match &hw_design[0] {
///     Atom {
///         name,
///         id: 0,
///         config:
///             AtomConfig::SystolicArrayWeightStationary(SystolicArrayWeightStationaryParams {
///                 dtype: DType::Fp32,
///                 rows: 32,
///                 cols: 32,
///             }),
///     } => assert_eq!(*name, "systolic_array_0".to_string()),
///     _ => panic!(),
/// };
///
/// assert_eq!(hw_map.len(), 3);
/// for (_id, val) in hw_map.iter() {
///     assert_eq!(*val, 0);
/// }
/// ```
pub fn create_hardware_design_monolithic(
    expr: &Expr,
    (row, col): (usize, usize),
) -> (HashMap<Id, usize>, Vec<Atom>) {
    let hw_id = 0;
    let mut map = HashMap::new();
    let atoms = vec![Atom {
        name: format!("systolic_array_{}", hw_id),
        id: hw_id,
        config: AtomConfig::SystolicArrayWeightStationary(SystolicArrayWeightStationaryParams {
            // TODO(@gussmith23) hardcoded datatype
            dtype: DType::Fp32,
            rows: row,
            cols: col,
        }),
    }];

    for eclass in expr.classes() {
        assert_eq!(eclass.nodes.len(), 1);
        match &eclass.nodes[0] {
            // TODO(@gussmith23) Need to test w/ blocking
            &Language::SystolicArrayWithBlocking([row_id, col_id, _, _])
            | &Language::SystolicArray([row_id, col_id, _, _]) => {
                match {
                    assert_eq!(expr[row_id].nodes.len(), 1);
                    &expr[row_id].nodes[0]
                } {
                    Language::Usize(u) => assert_eq!(
                        *u, row,
                        "Found a systolic array with row size {} not equal to {}",
                        *u, row
                    ),
                    _ => panic!(),
                };
                match {
                    assert_eq!(expr[col_id].nodes.len(), 1);
                    &expr[col_id].nodes[0]
                } {
                    Language::Usize(u) => assert_eq!(
                        *u, col,
                        "Found a systolic array with col size {} not equal to {}",
                        *u, col
                    ),
                    _ => panic!(),
                };

                map.insert(eclass.id, hw_id);
            }
            _ => (),
        }
    }

    (map, atoms)
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
            // TODO(@gussmith23) Need to test w/ blocking
            &Language::SystolicArrayWithBlocking([row_id, col_id, _, _])
            | &Language::SystolicArray([row_id, col_id, _, _]) => {
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
    fn find_vars_recursive_helper(set: &mut HashSet<String>, expr: &Expr, id: Id) {
        match {
            assert_eq!(expr[id].nodes.len(), 1);
            &expr[id].nodes[0]
        } {
            Language::RelayOperator(_) => {}
            Language::RelayKernelLayout(_) => {}
            Language::RelayActivationLayout(_) => {}
            Language::Symbol(s) => {
                set.insert(s.to_string());
            }
            // Id
            &Language::AccessTensor(id) | &Language::AccessFlatten(id) => {
                find_vars_recursive_helper(set, expr, id);
            }
            // Box<[Id]>
            Language::RelayOperatorCall(ids)
            | Language::List(ids)
            | Language::Shape(ids)
            | Language::ConstructTuple(ids) => {
                for id in ids.iter() {
                    find_vars_recursive_helper(set, expr, *id);
                }
            }
            // [Id; 1]
            &Language::ShapeOf(ids) => {
                for id in ids.iter() {
                    find_vars_recursive_helper(set, expr, *id);
                }
            }
            // [Id; 2]
            &Language::Access(ids)
            | &Language::AccessTranspose(ids)
            | &Language::AccessReshape(ids)
            | &Language::ShapeInsertAxis(ids)
            | &Language::ShapeRemoveAxis(ids)
            | &Language::AccessShape(ids)
            | &Language::AccessSqueeze(ids)
            | &Language::TupleGetItem(ids) => {
                for id in ids.iter() {
                    find_vars_recursive_helper(set, expr, *id);
                }
            }
            // [Id; 3]
            &Language::AccessConcatenate(ids) | &Language::AccessWindows(ids) => {
                for id in ids.iter() {
                    find_vars_recursive_helper(set, expr, *id);
                }
            }
            // [Id; 4]
            &Language::SystolicArray(ids)
            | &Language::SystolicArrayWithBlocking(ids)
            | &Language::AccessSlice(ids) => {
                for id in ids.iter() {
                    find_vars_recursive_helper(set, expr, *id);
                }
            }
            // [Id; 5]
            &Language::AccessPad(ids) => {
                for id in ids.iter() {
                    find_vars_recursive_helper(set, expr, *id);
                }
            }
            &Language::NotNanFloat64(_) => {}
            &Language::Usize(_) | &Language::PadType(_) => (),
            &Language::Literal(_)
            | &Language::SystolicArrayConv2dIm2colNchwOihwWithBlocking(_)
            | &Language::SystolicArrayConv2dIm2colNhwcHwioWithBlocking(_)
            | &Language::SystolicArrayConv2dNchwOihwWithBlocking(_)
            | &Language::SystolicArrayConv2dNhwcHwioWithBlocking(_)
            | &Language::AccessLiteral(_)
            | &Language::AccessBroadcast(_)
            | &Language::AccessInsertAxis(_)
            | &Language::AccessPair(_)
            | Language::ComputeType(_)
            | &Language::Compute(_)
            | &Language::AccessCartesianProduct(_)
            | &Language::SliceShape(_)
            | &Language::MoveAxis(_)
            | &Language::CartesianProduct(_)
            | &Language::MapDotProduct(_)
            | &Language::Slice(_)
            | &Language::Concatenate(_)
            | &Language::ElementwiseAdd(_)
            | &Language::BsgSystolicArray(_)
            | &Language::AccessShiftRight(_) => panic!("{:#?} not implemented", expr[id].nodes[0]),
        }
    }

    let mut set = HashSet::default();
    find_vars_recursive_helper(&mut set, expr, id);

    Vec::from_iter(set.drain())
}

/// Generates a worklist for codegen recursively, given an egraph and an eclass
/// id to start from.
///
/// This is the easiest way to generate a worklist for codegen, and should
/// probably be used in most cases. When the egraph is very large (i.e. when
/// handling real models) the worklist will be generated using other methods, as
/// this recursive method will overflow the stack.
// TODO(@gussmith23) Could add doctests
pub fn generate_worklist_for_codegen(expr: &Expr, id: Id) -> Vec<Id> {
    fn add_to_worklist(id: Id, worklist: &mut Vec<Id>) {
        if !worklist.contains(&id) {
            worklist.push(id);
        }
    }
    fn helper(worklist: &mut Vec<Id>, expr: &Expr, id: Id) {
        match {
            assert_eq!(expr[id].nodes.len(), 1);
            &expr[id].nodes[0]
        } {
            // Id
            &Language::AccessTensor(id) | &Language::AccessFlatten(id) => {
                helper(worklist, expr, id);
            }
            // [Id; 1]
            &Language::ShapeOf(ids) => {
                for id in ids.iter() {
                    helper(worklist, expr, *id);
                }
            }
            // Box<[Id]>
            Language::RelayOperatorCall(ids)
            | Language::Shape(ids)
            | Language::List(ids)
            | Language::ConstructTuple(ids) => {
                for id in ids.iter() {
                    helper(worklist, expr, *id);
                }
            }
            // [Id; 2]
            &Language::Access(ids)
            | &Language::AccessTranspose(ids)
            | &Language::AccessShape(ids)
            | &Language::AccessReshape(ids)
            | &Language::ShapeInsertAxis(ids)
            | &Language::ShapeRemoveAxis(ids)
            | &Language::AccessSqueeze(ids)
            | &Language::TupleGetItem(ids) => {
                for id in ids.iter() {
                    helper(worklist, expr, *id);
                }
            }
            // [Id; 3]
            &Language::AccessConcatenate(ids) | &Language::AccessWindows(ids) => {
                for id in ids.iter() {
                    helper(worklist, expr, *id);
                }
            }
            // [Id; 4]
            &Language::SystolicArray(ids)
            | &Language::SystolicArrayWithBlocking(ids)
            | &Language::AccessSlice(ids) => {
                for id in ids.iter() {
                    helper(worklist, expr, *id);
                }
            }
            // [Id; 5]
            &Language::AccessPad(ids) => {
                for id in ids.iter() {
                    helper(worklist, expr, *id);
                }
            }

            Language::RelayOperator(_)
            | Language::RelayKernelLayout(_)
            | Language::RelayActivationLayout(_)
            | Language::Symbol(_)
            | &Language::NotNanFloat64(_)
            | &Language::Usize(_)
            | &Language::PadType(_) => (),

            &Language::Literal(_)
            | &Language::SystolicArrayConv2dIm2colNchwOihwWithBlocking(_)
            | &Language::SystolicArrayConv2dIm2colNhwcHwioWithBlocking(_)
            | &Language::SystolicArrayConv2dNchwOihwWithBlocking(_)
            | &Language::SystolicArrayConv2dNhwcHwioWithBlocking(_)
            | &Language::AccessLiteral(_)
            | &Language::AccessBroadcast(_)
            | &Language::AccessInsertAxis(_)
            | &Language::AccessPair(_)
            | Language::ComputeType(_)
            | &Language::Compute(_)
            | &Language::AccessCartesianProduct(_)
            | &Language::SliceShape(_)
            | &Language::MoveAxis(_)
            | &Language::CartesianProduct(_)
            | &Language::MapDotProduct(_)
            | &Language::Slice(_)
            | &Language::Concatenate(_)
            | &Language::ElementwiseAdd(_)
            | &Language::BsgSystolicArray(_)
            | &Language::AccessShiftRight(_) => panic!("{:#?} not implemented", expr[id].nodes[0]),
        }

        add_to_worklist(id, worklist);
    }

    let mut worklist = Vec::default();
    helper(&mut worklist, expr, id);

    worklist
}

/// Returns c code.
///
/// args: The signature will be `void <function_name>(float * out, float * <arg0>...)`
/// uninitialized_allocations_prefix: The prefix to use for buffer allocations
/// that do not need to be initialized. In the future, once we have literals in
/// the program, we will need to also include an initialized_allocations_prefix.
///
/// worklist: the eclasses to generate code for, in order. Generally, you should
/// use [`generate_worklist`].
///
/// assert_only_one_enode_per_eclass: If this is true, then panics if there is
/// more than one enode in any eclass. If false, it just emits a warning. Useful
/// in ILP extraction, when ILP might not have found an optimal assignment.
// TODO(@gussmith23) Does not reason about ordering on hardware.
// TODO(@gussmith23) Hardcoded to float32
pub fn codegen(
    expr: &Expr,
    id: Id,
    hw_map: &HashMap<Id, usize>,
    function_name: &str,
    uninitialized_allocations_prefix: &str,
    args: &Vec<&str>,
    worklist: &Vec<Id>,
    assert_only_one_enode_per_eclass: bool,
) -> String {
    let mut declarations = String::default();
    let mut code = String::default();
    let mut id_to_variable: HashMap<Id, String> = HashMap::default();

    for id in worklist {
        if let Some(var_name) = codegen_helper(
            expr,
            *id,
            uninitialized_allocations_prefix,
            &mut declarations,
            &mut code,
            hw_map,
            |_, id| {
                id_to_variable
                    .get(&id)
                    .unwrap_or_else(|| panic!("Id {} not found in map of already compiled expressions -- is your worklist ordered correctly?", id))
                    .clone()
            },
            assert_only_one_enode_per_eclass,
        ) {
            id_to_variable.insert(*id, var_name);
        }
    }

    let out_symbol = id_to_variable.get(&id).unwrap();

    let found_vars = find_vars(expr, id);
    for found_var in found_vars.iter() {
        assert!(
            args.contains(&found_var.as_str()),
            format!("Found {} in program, but did not find in `args`", found_var)
        );
    }

    let mut signature = format!("void {}(", function_name);

    // Output comes first
    signature.push_str(
        c_array_string(
            "out",
            match &expr[id].data {
                MyAnalysisData::AccessPattern(a) => a.as_vec(),
                _ => panic!("Assuming output is a tensor for now"),
            }
            .as_slice(),
            // TODO(@gussmith23) Assuming float32 output.
            DType::Fp32,
        )
        .as_str(),
    );

    signature.push_str(
        args.iter()
            .map(|var| {
                format!(
                    ", {}",
                    c_array_string(
                        var,
                        expr.analysis.name_to_shape[*var].as_slice(),
                        DType::Fp32,
                    )
                )
            })
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

/// uninitialized_allocations_prefix: The prefix to use for buffer allocations
/// that do not need to be initialized. In the future, once we have literals in
/// the program, we will need to also include an initialized_allocations_prefix.
///
/// get_c_variable_for_id: When the codegen needs the variable name of a
/// subexpression it depends on, it can use this function to get it.
///
/// Optionally returns a [`String`] containing the C variable name for the
/// variable which holds this expression's result. May return [`None`] if no
/// such variable exists, or if no code was actually generated for this
/// expression.
fn codegen_helper(
    expr: &Expr,
    id: Id,
    uninitialized_allocations_prefix: &str,
    declarations: &mut String,
    code: &mut String,
    hw_map: &HashMap<Id, usize>,
    get_c_variable_for_id: impl Fn(&Expr, Id) -> String,
    assert_only_one_enode_per_eclass: bool,
) -> Option<String> {
    match {
        if expr[id].nodes.len() > 1 {
            if assert_only_one_enode_per_eclass {
                panic!("eclass {} has {} variants", id, expr[id].nodes.len());
            } else {
                warn!(
                    "eclass {} has {} variants, defaulting to variant 0",
                    id,
                    expr[id].nodes.len()
                );
            }
        }
        &expr[id].nodes[0]
    } {
        Language::RelayOperatorCall(ids) => {
            let relay_op = match &expr[ids[0]].data {
                MyAnalysisData::RelayOperator(op) => op,
                _ => panic!(),
            };

            match relay_op {
                RelayOperator::RelayBatchNormInference => {
                    let data = get_c_variable_for_id(expr, ids[1]);
                    let gamma = get_c_variable_for_id(expr, ids[2]);
                    let beta = get_c_variable_for_id(expr, ids[3]);

                    let moving_mean = get_c_variable_for_id(expr, ids[4]);

                    let moving_var = get_c_variable_for_id(expr, ids[5]);

                    let axis = MyAnalysis::get_usize(ids[6], expr);

                    // expect NHWC format data
                    assert!(axis == 3, "expected NHWC format");
                    let epsilon = match &expr[ids[7]].data {
                        MyAnalysisData::Literal(l) => l,
                        _ => panic!(),
                    };

                    let new_shape = match &expr[id].data {
                        MyAnalysisData::AccessPattern(a) => a.as_vec(),
                        _ => panic!(),
                    };

                    let batchnorm_out: String = {
                        // TODO(@gussmith23) Find a different way to name intermediates
                        // Currently generating random strings. Not great IMO.
                        let out = format!(
                            "relay_op_batchnorminference_out_{}",
                            rand::thread_rng()
                                .sample_iter(&rand::distributions::Alphanumeric)
                                .take(30)
                                .collect::<String>()
                        );
                        declarations.push_str(
                            c_allocation_string(
                                uninitialized_allocations_prefix,
                                out.as_str(),
                                new_shape.as_slice(),
                                DType::Fp32,
                            )
                            .as_str(),
                        );
                        out
                    };

                    code.push_str(format!("
batchNormInference((float*) {X}, (float*) {Y}, {N}, {H}, {W}, {C}, (float*) {gamma}, (float*) {beta}, (float*) {moving_mean}, (float*) {moving_var}, {epsilon});
",                      X = data,
                        Y = batchnorm_out,
                        N = new_shape[0],
                        H = new_shape[1],
                        W = new_shape[2],
                        C = new_shape[3],
                        gamma = gamma,
                        beta = beta,
                        moving_mean = moving_mean,
                        moving_var = moving_var,
                        epsilon = epsilon
                    )
                    .as_str());

                    Some(batchnorm_out)
                }
                RelayOperator::RelaySoftmax => {
                    let data = get_c_variable_for_id(expr, ids[1]);

                    // TODO: axis not used since
                    // softmax c function does softmax over all values
                    let axis = MyAnalysis::get_usize(ids[2], expr);

                    let new_shape = match &expr[id].data {
                        MyAnalysisData::AccessPattern(a) => a.as_vec(),
                        _ => panic!(),
                    };

                    assert_eq!(
                        new_shape.len(),
                        2,
                        "softmax assumes 2 dimensions (batch_size, num_classes)"
                    );
                    assert_eq!(axis, 1, "softmax should be over axis = 1");
                    assert_eq!(new_shape[0], 1, "softmax only supports batch size = 1");

                    let softmax_out: String = {
                        // TODO(@gussmith23) Find a different way to name intermediates
                        // Currently generating random strings. Not great IMO.
                        let out = format!(
                            "relay_op_softmax_out_{}",
                            rand::thread_rng()
                                .sample_iter(&rand::distributions::Alphanumeric)
                                .take(30)
                                .collect::<String>()
                        );
                        declarations.push_str(
                            c_allocation_string(
                                uninitialized_allocations_prefix,
                                out.as_str(),
                                new_shape.as_slice(),
                                DType::Fp32,
                            )
                            .as_str(),
                        );
                        out
                    };

                    code.push_str(
                        format!(
                            "
softmax1D((float*) {X}, (float*) {Y}, {N});
",
                            X = data,
                            Y = softmax_out,
                            N = new_shape.iter().product::<usize>()
                        )
                        .as_str(),
                    );

                    Some(softmax_out)
                }
                RelayOperator::RelayReLU => {
                    let data = get_c_variable_for_id(expr, ids[1]);

                    let new_shape = match &expr[id].data {
                        MyAnalysisData::AccessPattern(a) => a.as_vec(),
                        _ => panic!(),
                    };

                    let relu_out: String = {
                        // TODO(@gussmith23) Find a different way to name intermediates
                        // Currently generating random strings. Not great IMO.
                        let out = format!(
                            "relay_op_relu_out_{}",
                            rand::thread_rng()
                                .sample_iter(&rand::distributions::Alphanumeric)
                                .take(30)
                                .collect::<String>()
                        );
                        declarations.push_str(
                            c_allocation_string(
                                uninitialized_allocations_prefix,
                                out.as_str(),
                                new_shape.as_slice(),
                                DType::Fp32,
                            )
                            .as_str(),
                        );
                        out
                    };

                    code.push_str(
                        format!(
                            "
relu((float*) {X}, (float*) {Y}, {N}, {H}, {W}, {C});
",
                            X = data,
                            Y = relu_out,
                            N = new_shape[0],
                            H = new_shape[1],
                            W = new_shape[2],
                            C = new_shape[3]
                        )
                        .as_str(),
                    );

                    Some(relu_out)
                }
                RelayOperator::RelayMaxPool2D => {
                    let data = get_c_variable_for_id(expr, ids[1]);

                    let _pool_size = MyAnalysis::get_shape_of_value(ids[2], expr);
                    let _strides = MyAnalysis::get_shape_of_value(ids[3], expr);
                    let _padding = MyAnalysis::get_shape_of_value(ids[4], expr);

                    let old_shape = match &expr[ids[1]].data {
                        MyAnalysisData::AccessPattern(a) => a.as_vec(),
                        _ => panic!(),
                    };
                    // TODO: currently only supports maxpool2d shape in resnet
                    assert!(old_shape[0] == 1 && old_shape[1] == 112 && old_shape[2] == 112 && old_shape[3] == 64,
                            "RelayMaxPool2d currently only works for resnet on tensors of (1, 112, 112, 64)");

                    let new_shape = match &expr[id].data {
                        MyAnalysisData::AccessPattern(a) => a.as_vec(),
                        _ => panic!(),
                    };
                    let maxpool2d_out: String = {
                        // TODO(@gussmith23) Find a different way to name intermediates
                        // Currently generating random strings. Not great IMO.
                        let out = format!(
                            "relay_op_maxpool2d_out_{}",
                            rand::thread_rng()
                                .sample_iter(&rand::distributions::Alphanumeric)
                                .take(30)
                                .collect::<String>()
                        );
                        declarations.push_str(
                            c_allocation_string(
                                uninitialized_allocations_prefix,
                                out.as_str(),
                                new_shape.as_slice(),
                                DType::Fp32,
                            )
                            .as_str(),
                        );
                        out
                    };

                    code.push_str(
                        format!(
                            "
maxpool2D3x3_resnet18_op6((float*) {X}, (float*) {Y});
",
                            X = data,
                            Y = maxpool2d_out
                        )
                        .as_str(),
                    );

                    Some(maxpool2d_out)
                }
                RelayOperator::RelayGlobalAvgPool2D => {
                    match &expr[ids[2]].data {
                        MyAnalysisData::RelayActivationLayout(l) => assert_eq!(
                            *l,
                            RelayActivationLayout::NHWC,
                            "Only supporting codegen for NHWC at the moment"
                        ),
                        _ => panic!(
                            "Expected third argument of RelayGlobalAvgPool2D call to be a layout"
                        ),
                    };

                    let data = get_c_variable_for_id(expr, ids[1]);

                    let old_shape = match &expr[ids[1]].data {
                        MyAnalysisData::AccessPattern(a) => a.as_vec(),
                        _ => panic!(),
                    };

                    let new_shape = match &expr[id].data {
                        MyAnalysisData::AccessPattern(a) => a.as_vec(),
                        _ => panic!(),
                    };
                    let globalavgpool2d_out: String = {
                        // TODO(@gussmith23) Find a different way to name intermediates
                        // Currently generating random strings. Not great IMO.
                        let out = format!(
                            "relay_op_globalavgpool2d_out_{}",
                            rand::thread_rng()
                                .sample_iter(&rand::distributions::Alphanumeric)
                                .take(30)
                                .collect::<String>()
                        );
                        declarations.push_str(
                            c_allocation_string(
                                uninitialized_allocations_prefix,
                                out.as_str(),
                                new_shape.as_slice(),
                                DType::Fp32,
                            )
                            .as_str(),
                        );
                        out
                    };

                    code.push_str(
                        format!(
                            "
globalAvgPool((float*) {X}, (float*) {Y}, {N}, {H}, {W}, {C});
",
                            X = data,
                            Y = globalavgpool2d_out,
                            N = old_shape[0],
                            H = old_shape[1],
                            W = old_shape[2],
                            C = old_shape[3]
                        )
                        .as_str(),
                    );

                    Some(globalavgpool2d_out)
                }
                RelayOperator::RelayBatchFlatten => {
                    let data = get_c_variable_for_id(expr, ids[1]);

                    // just a reshape, which is a no-op!
                    Some(data)
                }
                RelayOperator::RelayBiasAdd | RelayOperator::RelayAdd => {
                    let a = get_c_variable_for_id(expr, ids[1]);
                    let b = get_c_variable_for_id(expr, ids[2]);

                    let a_shape = match &expr[ids[1]].data {
                        MyAnalysisData::AccessPattern(a) => a.as_vec(),
                        _ => panic!(),
                    };
                    let b_shape = match &expr[ids[2]].data {
                        MyAnalysisData::AccessPattern(a) => a.as_vec(),
                        _ => panic!(),
                    };

                    let new_shape = match &expr[id].data {
                        MyAnalysisData::AccessPattern(a) => a.as_vec(),
                        _ => panic!(),
                    };

                    let add_out: String = {
                        // TODO(@gussmith23) Find a different way to name intermediates
                        // Currently generating random strings. Not great IMO.
                        let out = format!(
                            "relay_op_add_out_{}",
                            rand::thread_rng()
                                .sample_iter(&rand::distributions::Alphanumeric)
                                .take(30)
                                .collect::<String>()
                        );
                        declarations.push_str(
                            c_allocation_string(
                                uninitialized_allocations_prefix,
                                out.as_str(),
                                new_shape.as_slice(),
                                DType::Fp32,
                            )
                            .as_str(),
                        );
                        out
                    };

                    let out_shape_str = format!(
                        "out_shape_add_{}",
                        rand::thread_rng()
                            .sample_iter(&rand::distributions::Alphanumeric)
                            .take(30)
                            .collect::<String>()
                    );
                    let a_shape_str = format!(
                        "a_shape_add_{}",
                        rand::thread_rng()
                            .sample_iter(&rand::distributions::Alphanumeric)
                            .take(30)
                            .collect::<String>()
                    );
                    let b_shape_str = format!(
                        "b_shape_add_{}",
                        rand::thread_rng()
                            .sample_iter(&rand::distributions::Alphanumeric)
                            .take(30)
                            .collect::<String>()
                    );

                    code.push_str(
                        format!(
                            "
{}
{}
{}                            
add_with_broadcasting((float*) {out}, (float*) {X}, (float*) {Y}, (int*)  {out_shape}, {out_dims}, (int*) {a_shape}, {a_dims}, (int*) {b_shape}, {b_dims});
",
                            c_assignment_string("", &out_shape_str, DType::Int32, &Array::from(new_shape.clone()).into_dyn().view()),
                            c_assignment_string("", &a_shape_str, DType::Int32, &Array::from(a_shape.clone()).into_dyn().view()),
                            c_assignment_string("", &b_shape_str, DType::Int32, &Array::from(b_shape.clone()).into_dyn().view()),
                            out = add_out,
                            X = a,
                            Y = b,
                            out_shape = out_shape_str,
                            out_dims = new_shape.len(),
                            a_shape = a_shape_str,
                            a_dims = a_shape.len(),
                            b_shape = b_shape_str,
                            b_dims = b_shape.len(),
                        )
                        .as_str(),
                    );

                    Some(add_out)
                }
                RelayOperator::RelayLeakyReLU => todo!(),
                RelayOperator::RelaySigmoid => todo!(),
                RelayOperator::RelayAvgPool2D => todo!(),
                RelayOperator::RelayUpSampling => todo!(),
                RelayOperator::RelayMaximum => todo!(),
                RelayOperator::RelayMinimum => todo!(),
            }
        }
        &Language::AccessWindows([access_id, filters_shape_id, stride_shape_id]) => {
            let access = match &expr[access_id].data {
                MyAnalysisData::AccessPattern(a) => a,
                _ => panic!(),
            };
            let filters_shape = MyAnalysis::get_shape_of_value(filters_shape_id, expr);
            let stride_shape = MyAnalysis::get_shape_of_value(stride_shape_id, expr);

            // TODO(@gussmith23) Generalize AccessWindows to other accesses
            // Right now we expect item shape to be a scalar.
            assert_eq!(access.item_shape.ndim(), 0);

            let access_windows_shape = crate::language::access_windows_resulting_shape(
                &access.shape,
                &filters_shape,
                &stride_shape,
            );

            let access_windows_item_shape = filters_shape.clone();

            assert_eq!(access_windows_shape.len(), access_windows_item_shape.ndim());
            assert_eq!(stride_shape.ndim(), access_windows_shape.len());

            let access_windows_out_var_name: String = {
                // TODO(@gussmith23) Find a different way to name intermediates
                // Currently generating random strings. Not great IMO.
                let out = format!(
                    "access_windows_out_{}",
                    rand::thread_rng()
                        .sample_iter(&rand::distributions::Alphanumeric)
                        .take(30)
                        .collect::<String>()
                );
                declarations.push_str(
                    c_allocation_string(
                        uninitialized_allocations_prefix,
                        out.as_str(),
                        access_windows_shape
                            .iter()
                            .chain(access_windows_item_shape.slice().iter())
                            .cloned()
                            .collect::<Vec<_>>()
                            .as_slice(),
                        DType::Fp32,
                    )
                    .as_str(),
                );
                out
            };

            // TODO(@gussmith23) It would make our lives easier if we
            let access_var_name = get_c_variable_for_id(expr, access_id);

            // Create a for loop for every dimension in the result shape.
            for (dim_index, dim_len) in access_windows_shape.iter().enumerate() {
                let index_var_name = format!("shape_index_{}", dim_index);
                code.push_str(
                    format!(
                        "
for (int {index_var_name} = 0; {index_var_name} < {dim_len}; {index_var_name}++) {{",
                        index_var_name = index_var_name,
                        dim_len = dim_len,
                    )
                    .as_str(),
                );
            }

            // Create a for loop for every dimension in the result item shape.
            for (dim_index, dim_len) in access_windows_item_shape.slice().iter().enumerate() {
                let index_var_name = format!("item_shape_index_{}", dim_index);
                code.push_str(
                    format!(
                        "
for (int {index_var_name} = 0; {index_var_name} < {dim_len}; {index_var_name}++) {{",
                        index_var_name = index_var_name,
                        dim_len = dim_len,
                    )
                    .as_str(),
                );
            }

            // Within the innermost for loop: assign to the output at the
            // correct location.
            code.push_str(
                format!(
                    "
{out_name}{out_index} = {in_name}{in_index};
",
                    out_name = access_windows_out_var_name,
                    out_index = (0..access_windows_shape.len())
                        .map(|i| format!("[shape_index_{}]", i))
                        .chain(
                            (0..access_windows_item_shape.ndim())
                                .map(|i| format!("[item_shape_index_{}]", i))
                        )
                        .collect::<Vec<_>>()
                        .join(""),
                    in_name = access_var_name,
                    in_index = (0..access_windows_shape.len())
                        .map(|i| format!(
                            "[{shape_index}*{stride} + {item_shape_index}]",
                            shape_index = format!("shape_index_{}", i),
                            item_shape_index = format!("item_shape_index_{}", i),
                            stride = stride_shape[i],
                        ))
                        .collect::<Vec<_>>()
                        .join("")
                )
                .as_str(),
            );

            // close each for loop
            for _ in 0..(access_windows_shape.len() + access_windows_item_shape.ndim()) {
                code.push_str("}");
            }

            Some(access_windows_out_var_name)
        }
        &Language::AccessSlice([access_id, axis_id, low_id, high_id]) => {
            let original_shape = match &expr[access_id].data {
                MyAnalysisData::AccessPattern(a) => a.as_vec(),
                _ => panic!(),
            };
            let axis = MyAnalysis::get_usize(axis_id, expr);
            let low = MyAnalysis::get_usize(low_id, expr);
            let _high = MyAnalysis::get_usize(high_id, expr);
            let new_shape = match &expr[id].data {
                MyAnalysisData::AccessPattern(a) => a.as_vec(),
                _ => panic!(),
            };

            let access_var_name = get_c_variable_for_id(expr, access_id);

            let slice_out_var_name: String = {
                // TODO(@gussmith23) Find a different way to name intermediates
                // Currently generating random strings. Not great IMO.
                let out = format!(
                    "access_slice_out_{}",
                    rand::thread_rng()
                        .sample_iter(&rand::distributions::Alphanumeric)
                        .take(30)
                        .collect::<String>()
                );
                declarations.push_str(
                    c_allocation_string(
                        uninitialized_allocations_prefix,
                        out.as_str(),
                        new_shape.as_slice(),
                        DType::Fp32,
                    )
                    .as_str(),
                );
                out
            };

            // Create a for loop for every dimension in the result shape.
            for (i, dim_len) in new_shape.iter().enumerate() {
                code.push_str(
                    format!(
                        "
for (int i{i} = 0; i{i} < {limit}; i{i}++) {{",
                        i = i,
                        limit = dim_len,
                    )
                    .as_str(),
                );
            }

            // Within the innermost for loop: assign to the output at the
            // correct location.
            code.push_str(
                format!(
                    "
{out_name}{out_index} = ((float*){in_name})[{in_index}];
",
                    out_name = slice_out_var_name,
                    out_index = (0..new_shape.len())
                        .map(|i| format!("[i{}]", i,))
                        .collect::<String>(),
                    in_name = access_var_name,
                    in_index = (0..original_shape.len())
                        .map(|i| if i != axis {
                            format!("(i{})", i)
                        } else {
                            format!("(i{}+{})", i, low,)
                        })
                        .enumerate()
                        .map(|(i, s)| format!(
                            "{}{}",
                            s,
                            original_shape[i + 1..]
                                .iter()
                                .map(|i| format!("*{}", i))
                                .collect::<Vec<_>>()
                                .join("")
                        ))
                        .join(" + ")
                )
                .as_str(),
            );

            // Close each for loop
            for _ in new_shape.iter() {
                code.push_str("}");
            }

            Some(slice_out_var_name)
        }
        &Language::AccessTensor(symbol_id) => {
            assert_eq!(expr[symbol_id].nodes.len(), 1);
            match &expr[symbol_id].nodes[0] {
                Language::Symbol(s) => Some(s.clone()),
                _ => panic!("expected a symbol!"),
            }
        }
        &Language::Access([access_tensor_id, _axis_id]) => {
            Some(get_c_variable_for_id(expr, access_tensor_id))
        }
        &Language::SystolicArray([rows_id, cols_id, a0_id, a1_id])
        | &Language::SystolicArrayWithBlocking([rows_id, cols_id, a0_id, a1_id]) => {
            let rows = MyAnalysis::get_usize(rows_id, expr);
            let cols = MyAnalysis::get_usize(cols_id, expr);

            let (a0, a1) = match (&expr[a0_id].data, &expr[a1_id].data) {
                (MyAnalysisData::AccessPattern(a0), MyAnalysisData::AccessPattern(a1)) => (a0, a1),
                _ => panic!(),
            };

            assert_eq!(a1.shape, IxDyn(&[]));
            assert!(a0.shape.ndim() == 0 || a0.shape.ndim() == 1);

            // TODO(@gussmith23) These are just ripped from language.rs.
            // Don't want to duplicate checks. Maybe we just assume things check
            // out? Given that they're in the egraph?
            match {
                assert_eq!(expr[id].nodes.len(), 1);
                &expr[id].nodes[0]
            } {
                &Language::SystolicArray(_) => {
                    assert_eq!(a1.item_shape, IxDyn(&[rows, cols]));
                    assert_eq!(a0.item_shape, IxDyn(&[rows]));
                }
                &Language::SystolicArrayWithBlocking(_) => {
                    // Scott: The input vector size should be a multiple of
                    // the systolic array's height and the output vector
                    // size should be a multiple of the systolic array's
                    // width.
                    assert_eq!(a0.item_shape.ndim(), 1);
                    assert!(a0.item_shape.slice()[0] % rows == 0);
                    assert_eq!(a1.item_shape.ndim(), 2);
                    assert_eq!(a0.item_shape.slice()[0], a1.item_shape.slice()[0]);
                    assert!(a1.item_shape.slice()[1] % cols == 0);
                }
                _ => unreachable!(),
            }

            let this_access = match &expr[id].data {
                MyAnalysisData::AccessPattern(a) => a,
                _ => panic!(),
            };
            assert!(this_access.shape.ndim() == 1 || this_access.shape.ndim() == 2);
            assert_eq!(this_access.item_shape.ndim(), 0);

            let s0 = get_c_variable_for_id(expr, a0_id);
            let s1 = get_c_variable_for_id(expr, a1_id);

            let out_var_name = match {
                assert_eq!(expr[id].nodes.len(), 1);
                &expr[id].nodes[0]
            } {
                &Language::SystolicArray(_) => format!(
                    "systolic_array_{}_eclass_{}_out",
                    hw_map.get(&id).unwrap(),
                    id,
                ),
                &Language::SystolicArrayWithBlocking(_) => format!(
                    "systolic_array_with_blocking_{}_eclass_{}_out",
                    hw_map.get(&id).unwrap(),
                    id,
                ),
                _ => unreachable!(),
            };

            // TODO(@gussmith23) how to assign unique names to each usage?
            // TODO(@gussmith23) Allocations should not be done ad-hoc
            declarations.push_str(
                c_allocation_string(
                    uninitialized_allocations_prefix,
                    out_var_name.as_str(),
                    this_access
                        .shape
                        .slice()
                        .iter()
                        .chain(this_access.item_shape.slice().iter())
                        .cloned()
                        .collect::<Vec<_>>()
                        .as_slice(),
                    // TODO(@gussmith23) Datatype assumption
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
                    format!("(float*){}", out_var_name,),
                    // Pointer to input vector
                    format!("(float*){}", s0,),
                    // Pointer to input matrix
                    format!("(float*){}", s1,),
                    // Length of input vector/size of input matrix dim 0
                    a1.item_shape.slice()[0],
                    // Size of input matrix dim 1/length of output vector
                    a1.item_shape.slice()[1],
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

            Some(out_var_name)
        }
        &Language::Usize(u) => Some(format!("{}", u)),
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

            let access_var_name = get_c_variable_for_id(expr, access_id);

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
                    c_allocation_string(
                        uninitialized_allocations_prefix,
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
for (int {i} = 0; {i} < {limit}; {i}++) {{",
                        i = index_var_name,
                        limit = dim_len,
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
  {out_name}{out_index} = ((float*){in_name})[{in_index}];
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
                        PadType::MinPadding => todo!(),
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

            Some(pad_out_var_name)
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

            let new_shape = match &expr[id].data {
                MyAnalysisData::AccessPattern(a) => a.as_vec(),
                _ => panic!(),
            };

            let new_axis_order = match &expr[list_id].data {
                MyAnalysisData::List(l) => l,
                _ => panic!(),
            };

            assert_eq!(original_shape.len(), new_axis_order.len());

            let access_var_name = get_c_variable_for_id(expr, access_id);
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
                    c_allocation_string(
                        uninitialized_allocations_prefix,
                        out.as_str(),
                        new_shape.as_slice(),
                        DType::Fp32,
                    )
                    .as_str(),
                );
                out
            };

            // Create a for loop for every dimension in the input.
            for (dim_index, dim_len) in original_shape.iter().enumerate() {
                code.push_str(
                    format!(
                        "
for (int {i} = 0; {i} < {limit}; {i}++) {{",
                        i = format!("i{}", dim_index),
                        limit = dim_len
                    )
                    .as_str(),
                );
            }

            // Within the innermost for loop: assign to the output at the
            // correct location.
            code.push_str(
                format!(
                    "
{out_var_name}{out_indices} = ((float*){access_var_name}){access_indices};",
                    out_var_name = transpose_out_var_name,
                    out_indices = (0..original_shape.len())
                        .map(|i| format!("[i{}]", new_axis_order[i]))
                        .collect::<Vec<_>>()
                        .join(""),
                    access_var_name = access_var_name,
                    access_indices = format!(
                        "[{}]",
                        (0..original_shape.len())
                            .map(|i| format!(
                                "i{}{}",
                                i,
                                original_shape[i + 1..]
                                    .iter()
                                    .map(|i| format!("*{}", i))
                                    .collect::<Vec<_>>()
                                    .join("")
                            ))
                            .collect::<Vec<_>>()
                            .join(" + ")
                    )
                )
                .as_str(),
            );

            // Close each for loop
            for _ in original_shape.iter() {
                code.push_str("}");
            }

            Some(transpose_out_var_name)
        }
        &Language::AccessFlatten(access_id) => Some(get_c_variable_for_id(expr, access_id)),
        &Language::AccessReshape([access_id, _access_shape_id]) => {
            Some(get_c_variable_for_id(expr, access_id))
        }
        &Language::AccessSqueeze([access_id, _axis_id]) => {
            Some(get_c_variable_for_id(expr, access_id))
        }
        &Language::AccessConcatenate([a0_id, a1_id, axis_id]) => {
            let axis = MyAnalysis::get_usize(axis_id, expr);
            let (a0, _a1) = match (&expr[a0_id].data, &expr[a1_id].data) {
                (MyAnalysisData::AccessPattern(a0), MyAnalysisData::AccessPattern(a1)) => (a0, a1),
                _ => panic!(),
            };
            let arg_0_name = get_c_variable_for_id(expr, a0_id);
            let arg_1_name = get_c_variable_for_id(expr, a1_id);

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
                    c_allocation_string(
                        uninitialized_allocations_prefix,
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
for (int i{i} = 0; i{i} < {dim_val}; i{i} ++) {{
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
if (i{i} < {dim_len}) {{
  {out_var_name}{out_indices} = {arg_0_name}{arg_0_indices};
}} else {{
  {out_var_name}{out_indices} = {arg_1_name}{arg_1_indices};
}}
",
                    i = axis,
                    dim_len = a0[axis],
                    out_var_name = out_var_name,
                    out_indices = (0..(concat_shape.len()))
                        .map(|i| format!("[i{}]", i,))
                        .join(""),
                    arg_0_name = arg_0_name,
                    arg_0_indices = (0..(concat_shape.len()))
                        .map(|i| format!("[i{}]", i,))
                        .join(""),
                    arg_1_name = arg_1_name,
                    arg_1_indices = (0..(concat_shape.len()))
                        .map(|i| if i != axis {
                            format!("[i{}]", i)
                        } else {
                            format!("[i{}-{}]", i, a0_shape[i],)
                        })
                        .join(""),
                )
                .as_str(),
            );

            for _ in concat_shape.iter().enumerate() {
                code.push_str("}");
            }

            Some(out_var_name)
        }
        // TODO(@gussmith23) Needs test
        &Language::NotNanFloat64(not_nan) => Some(format!("{:.20}f", not_nan.into_inner())),

        // Constructs for which we shouldn't need to generate code.
        Language::RelayActivationLayout(_)
        | Language::RelayKernelLayout(_)
        | Language::Symbol(_)
        | Language::PadType(_)
        | Language::Shape(_)
        | Language::List(_)
        | &Language::ShapeInsertAxis(_)
        | &Language::ShapeRemoveAxis(_)
        | &Language::ShapeOf(_)
        | &Language::AccessShape(_)
        | Language::RelayOperator(_) => None,

        &Language::Literal(_)
        | &Language::ConstructTuple(_)
        | &Language::TupleGetItem(_)
        | &Language::SystolicArrayConv2dIm2colNchwOihwWithBlocking(_)
        | &Language::SystolicArrayConv2dIm2colNhwcHwioWithBlocking(_)
        | &Language::SystolicArrayConv2dNchwOihwWithBlocking(_)
        | &Language::SystolicArrayConv2dNhwcHwioWithBlocking(_)
        | &Language::AccessBroadcast(_)
        | &Language::AccessInsertAxis(_)
        | &Language::AccessPair(_)
        | Language::ComputeType(_)
        | &Language::Compute(_)
        | &Language::AccessCartesianProduct(_)
        | &Language::SliceShape(_)
        | &Language::MoveAxis(_)
        | &Language::CartesianProduct(_)
        | &Language::MapDotProduct(_)
        | &Language::Slice(_)
        | &Language::Concatenate(_)
        | &Language::ElementwiseAdd(_)
        | &Language::BsgSystolicArray(_)
        | &Language::AccessLiteral(_)
        | &Language::AccessShiftRight(_) => panic!("{:#?} not implemented", expr[id].nodes[0]),
    }
}

pub fn run_relay(
    env: &HashMap<String, ArrayD<f32>>,
    shapes_vec: &Vec<(String, Vec<usize>)>,
    relay_str: &str,
) -> ArrayD<f32> {
    let script_filepath = format!(
        "{}/src/language/from_relay/run_relay.py",
        env!("CARGO_MANIFEST_DIR")
    );
    // https://www.reddit.com/r/rust/comments/38jhva/piping_string_to_child_process_stdin/crvlqcd/?utm_source=reddit&utm_medium=web2x&context=3
    // Output filename
    // TODO(@gussmith23) Do we want this RNG to use SEED?
    // I initially attempted to do this, but was running into issues
    // (I think the same filename kept being generated b/c I wasn't
    // using the RNG carefully...but maybe there's also something
    // wrong w/ how I'm reading files!)
    let output_filepath = std::env::temp_dir().with_file_name(format!(
        "output-{}.npy",
        rand::thread_rng()
            .sample_iter(&rand::distributions::Alphanumeric)
            .take(30)
            .collect::<String>()
    ));

    let mut cmd = Command::new("python3");
    cmd.arg(script_filepath);
    cmd.arg(&output_filepath);
    cmd.stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    for (name, _) in shapes_vec.iter() {
        let value = env.get(name).unwrap();
        // TODO(@gussmith23) output type assumption
        let filepath = std::env::temp_dir().with_file_name(format!(
            "arg-{}.npy",
            rand::thread_rng()
                .sample_iter(&rand::distributions::Alphanumeric)
                .take(30)
                .collect::<String>()
        ));
        write_npy(&filepath, value).unwrap();
        cmd.arg(filepath);
    }

    let mut proc = cmd.spawn().ok().expect("Failed to spawn process");
    proc.stdin
        .as_mut()
        .unwrap()
        .write_all(relay_str.as_bytes())
        .unwrap();
    let output = proc.wait_with_output().unwrap();
    // Check that it ran.
    assert!(
        output.status.success(),
        "Running Relay code failed with code {:?}.\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        std::str::from_utf8(output.stdout.as_slice()).expect("Could not convert stderr to UTF8"),
        std::str::from_utf8(output.stderr.as_slice()).expect("Could not convert stderr to UTF8")
    );

    // TODO(@gussmith23) output type assumption
    let relay_output: ndarray::ArrayD<f32> = read_npy(output_filepath).unwrap();
    relay_output
}

#[cfg(test)]
mod tests {
    use super::*;
    use egg::RecExpr;
    use ndarray::ArrayD;
    use ndarray::{SliceInfo, SliceOrIndex};
    use ndarray_npy::{read_npy, write_npy};
    use ndarray_rand::{rand_distr::Uniform, RandomExt};
    use rand::{rngs::SmallRng, Rng, SeedableRng};
    use std::fs::File;
    use std::io::Write;
    use std::iter::FromIterator;
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

        let code = codegen(
            &egraph,
            id,
            &HashMap::default(),
            "transpose",
            "",
            &vec!["t"],
            &generate_worklist_for_codegen(&egraph, id),
            true,
        );

        let main_code = format!(
            "
#include <assert.h>

{}
{}
{}
{}

int main() {{
  transpose(out, a);

  for (int i = 0; i < {}; i++) {{
    assert(((float*)a_t)[i] == ((float*)out)[i]);
  }}
}}
",
            c_assignment_string("", "a", DType::Fp32, &input.view()),
            c_assignment_string("", "a_t", DType::Fp32, &input_transposed.view()),
            c_assignment_string(
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
            .arg("-Werror")
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

        let code = codegen(
            &egraph,
            id,
            &HashMap::default(),
            "concatenate",
            "",
            &vec!["t0", "t1"],
            &generate_worklist_for_codegen(&egraph, id),
            true,
        );

        let main_code = format!(
            "
#include <assert.h>

{}
{}
{}
{}
{}

int main() {{
  concatenate(out, t0, t1);

  for (int i = 0; i < {}; i++) {{
    assert(((float*)a_t)[i] == ((float*)out)[i]);
  }}
}}
",
            c_assignment_string("", "t0", DType::Fp32, &input0.view()),
            c_assignment_string("", "t1", DType::Fp32, &input1.view()),
            c_assignment_string("", "a_t", DType::Fp32, &concatted.view()),
            c_assignment_string(
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
            .arg("-Werror")
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

        let code = codegen(
            &egraph,
            id,
            &hw_map,
            "systolic_array",
            "",
            &vec!["t0", "t1"],
            &generate_worklist_for_codegen(&egraph, id),
            true,
        );

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
  systolic_array(out, t0, t1);

  for (int i = 0; i < {}; i++) {{
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
            c_assignment_string("", "t0", DType::Fp32, &input0.into_dyn().view()),
            c_assignment_string("", "t1", DType::Fp32, &input1.into_dyn().view()),
            c_assignment_string("", "result", DType::Fp32, &multiplied.view()),
            c_assignment_string(
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
            .arg("-Werror")
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

        let code = codegen(
            &egraph,
            id,
            &HashMap::default(),
            "pad",
            "",
            &vec!["t"],
            &generate_worklist_for_codegen(&egraph, id),
            true,
        );

        let main_code = format!(
            "
#include <assert.h>

{}
{}
{}
{}

int main() {{
  pad(out, a);

  for (int i = 0; i < {}; i++) {{
    assert(((float*)a_pad)[i] == ((float*)out)[i]);
  }}
}}
",
            c_assignment_string("", "a", DType::Fp32, &input.view()),
            c_assignment_string("", "a_pad", DType::Fp32, &padded.view()),
            c_assignment_string(
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
            .arg("-Werror")
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
    fn slice() {
        let shape = vec![32, 7, 100, 3];
        let slice_axis = 2;
        let low = 5;
        let high = 83;

        let input = ndarray::ArrayD::from_shape_vec(
            shape.clone(),
            (0..shape.iter().product::<usize>()).collect(),
        )
        .unwrap();

        let mut slices = Vec::from_iter(
            std::iter::repeat(SliceOrIndex::Slice {
                start: 0,
                end: None,
                step: 1,
            })
            .take(shape.len()),
        );
        slices[slice_axis] = SliceOrIndex::Slice {
            start: low,
            end: Some(high),
            step: 1,
        };
        let sliced = input.slice(
            &SliceInfo::<std::vec::Vec<ndarray::SliceOrIndex>, ndarray::IxDyn>::new(slices)
                .unwrap()
                .as_ref(),
        );

        let expr = RecExpr::from_str(
            format!(
                "
(access-slice (access-tensor t) {} {} {})",
                slice_axis, low, high
            )
            .as_str(),
        )
        .unwrap();

        let mut map = HashMap::default();
        map.insert("t".to_string(), shape.clone());

        let mut egraph = EGraph::new(MyAnalysis { name_to_shape: map });
        let id = egraph.add_expr(&expr);

        let code = codegen(
            &egraph,
            id,
            &HashMap::default(),
            "slice",
            "",
            &vec!["t"],
            &generate_worklist_for_codegen(&egraph, id),
            true,
        );

        let main_code = format!(
            "
#include <assert.h>

{}
{}
{}
{}

int main() {{
  slice(out, a);

  for (int i = 0; i < {}; i++) {{
    assert(((float*)a_sliced)[i] == ((float*)out)[i]);
  }}
}}
",
            c_assignment_string("", "a", DType::Fp32, &input.view()),
            c_assignment_string("", "a_sliced", DType::Fp32, &sliced.view()),
            c_assignment_string(
                "",
                "out",
                DType::Fp32,
                &ndarray::ArrayD::<f32>::zeros(sliced.shape()).view()
            ),
            code,
            sliced.shape().iter().product::<usize>()
        );

        let main_c_filepath = std::env::temp_dir().with_file_name(format!(
            "slice-test-{}.c",
            std::time::SystemTime::now().elapsed().unwrap().as_nanos()
        ));

        let binary_filepath = std::env::temp_dir().with_file_name(format!(
            "slice-test-{}",
            std::time::SystemTime::now().elapsed().unwrap().as_nanos()
        ));
        println!("{}", binary_filepath.to_string_lossy());

        File::create(&main_c_filepath)
            .unwrap()
            .write_all(main_code.as_bytes())
            .unwrap();

        let result = Command::new("gcc")
            .arg("-Werror")
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
    fn access_windows() {
        let shape = vec![3, 50, 27, 4];
        let filters_shape = vec![2, 9, 22, 3];
        let stride = vec![1, 10, 3, 1];

        let input = ndarray::ArrayD::from_shape_vec(
            shape.clone(),
            (0i64..shape.iter().map(|v| *v as i64).product::<i64>()).collect(),
        )
        .unwrap();

        let expr = RecExpr::from_str(
            format!(
                "
(access-windows
 (access (access-tensor t) {access_axis})
 (shape {filter_shapes})
 (shape {strides})
)",
                access_axis = shape.len(),
                filter_shapes = filters_shape
                    .iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(" "),
                strides = stride
                    .iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(" "),
            )
            .as_str(),
        )
        .unwrap();

        let mut map = HashMap::default();
        map.insert("t".to_string(), shape.clone());

        let mut env = HashMap::default();
        env.insert("t", input.clone());
        let out =
            match crate::language::interpreter::interpret(&expr, expr.as_ref().len() - 1, &env) {
                crate::language::interpreter::Value::Access(a) => a,
                _ => panic!(),
            };

        let mut egraph = EGraph::new(MyAnalysis { name_to_shape: map });
        let id = egraph.add_expr(&expr);

        let code = codegen(
            &egraph,
            id,
            &HashMap::default(),
            "access_windows",
            "",
            &vec!["t"],
            &generate_worklist_for_codegen(&egraph, id),
            true,
        );

        let main_code = format!(
            "
#include <assert.h>

{}
{}
{}
{}

int main() {{
  access_windows(out, a);

  for (int i = 0; i < {}; i++) {{
    assert(((float*)a_windows)[i] == ((float*)out)[i]);
  }}
}}
",
            c_assignment_string("", "a", DType::Fp32, &input.view()),
            c_assignment_string("", "a_windows", DType::Fp32, &out.tensor.view()),
            c_assignment_string(
                "",
                "out",
                DType::Fp32,
                &ndarray::ArrayD::<f32>::zeros(out.tensor.shape()).view()
            ),
            code,
            out.tensor.shape().iter().product::<usize>()
        );

        let main_c_filepath = std::env::temp_dir().with_file_name(format!(
            "access-windows-test-{}.c",
            std::time::SystemTime::now().elapsed().unwrap().as_nanos()
        ));

        let binary_filepath = std::env::temp_dir().with_file_name(format!(
            "access-windows-test-{}",
            std::time::SystemTime::now().elapsed().unwrap().as_nanos()
        ));
        println!("{}", binary_filepath.to_string_lossy());

        File::create(&main_c_filepath)
            .unwrap()
            .write_all(main_code.as_bytes())
            .unwrap();

        let result = Command::new("gcc")
            .arg("-Werror")
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
    fn access_flatten() {
        let shape = vec![3, 50, 27, 4];
        let access_axis = 2;

        let input = ndarray::ArrayD::from_shape_vec(
            shape.clone(),
            (0..shape.iter().product::<usize>()).collect(),
        )
        .unwrap();

        let expr = RecExpr::from_str(
            format!(
                "
(access-flatten
 (access (access-tensor t) {access_axis})
)",
                access_axis = access_axis
            )
            .as_str(),
        )
        .unwrap();

        let mut map = HashMap::default();
        map.insert("t".to_string(), shape.clone());

        let out = input
            .view()
            .into_shape((
                shape[0..access_axis].iter().product::<usize>(),
                shape[access_axis..].iter().product::<usize>(),
            ))
            .unwrap()
            .into_dyn();

        let mut egraph = EGraph::new(MyAnalysis { name_to_shape: map });
        let id = egraph.add_expr(&expr);

        let code = codegen(
            &egraph,
            id,
            &HashMap::default(),
            "access_flatten",
            "",
            &vec!["t"],
            &generate_worklist_for_codegen(&egraph, id),
            true,
        );

        let main_code = format!(
            "
#include <assert.h>

{}
{}
{}
{}

int main() {{
  access_flatten(out, a);

  for (int i = 0; i < {}; i++) {{
    assert(((float*)a_flattened)[i] == ((float*)out)[i]);
  }}
}}
",
            c_assignment_string("", "a", DType::Fp32, &input.view()),
            c_assignment_string("", "a_flattened", DType::Fp32, &out.view()),
            c_assignment_string(
                "",
                "out",
                DType::Fp32,
                &ndarray::ArrayD::<f32>::zeros(out.shape()).view()
            ),
            code,
            out.shape().iter().product::<usize>()
        );

        let main_c_filepath = std::env::temp_dir().with_file_name(format!(
            "access-flatten-test-{}.c",
            std::time::SystemTime::now().elapsed().unwrap().as_nanos()
        ));

        let binary_filepath = std::env::temp_dir().with_file_name(format!(
            "access-flatten-test-{}",
            std::time::SystemTime::now().elapsed().unwrap().as_nanos()
        ));
        println!("{}", binary_filepath.to_string_lossy());

        File::create(&main_c_filepath)
            .unwrap()
            .write_all(main_code.as_bytes())
            .unwrap();

        let result = Command::new("gcc")
            .arg("-Werror")
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
    #[should_panic]
    fn extract_monolithic_panic() {
        let expr = RecExpr::from_str(
            "
(systolic-array 32 32
 (access
  (systolic-array 64 32
   (access (access-tensor t0) 1)
   (access (access-tensor t1) 0)
  )
  1
 )
 (access
  (systolic-array 32 32
   (access (access-tensor t2) 1)
   (access (access-tensor t3) 0)
  )
  0
 )
)",
        )
        .unwrap();

        let mut map = HashMap::default();
        map.insert("t0".to_string(), vec![32, 64]);
        map.insert("t1".to_string(), vec![64, 32]);
        map.insert("t2".to_string(), vec![32, 32]);
        map.insert("t3".to_string(), vec![32, 32]);

        let mut egraph = EGraph::new(MyAnalysis { name_to_shape: map });
        egraph.add_expr(&expr);

        let (_hw_map, _hw_design) = create_hardware_design_monolithic(&egraph, (32, 32));
    }

    #[test]
    fn systolic_array_with_blocking() {
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
(systolic-array-with-blocking 2 5
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

        let code = codegen(
            &egraph,
            id,
            &hw_map,
            "systolic_array_with_blocking",
            "",
            &vec!["t0", "t1"],
            &generate_worklist_for_codegen(&egraph, id),
            true,
        );

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
  systolic_array_with_blocking(out, t0, t1);

  for (int i = 0; i < {}; i++) {{
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
            c_assignment_string("", "t0", DType::Fp32, &input0.into_dyn().view()),
            c_assignment_string("", "t1", DType::Fp32, &input1.into_dyn().view()),
            c_assignment_string("", "result", DType::Fp32, &multiplied.view()),
            c_assignment_string(
                "",
                "out",
                DType::Fp32,
                &ndarray::ArrayD::<f32>::zeros(multiplied.shape()).view()
            ),
            code,
            multiplied.shape().iter().product::<usize>()
        );

        let main_c_filepath = std::env::temp_dir().with_file_name(format!(
            "systolic-array-with-blocking-test-{}.c",
            std::time::SystemTime::now().elapsed().unwrap().as_nanos()
        ));
        println!("{}", main_c_filepath.to_string_lossy());

        let binary_filepath = std::env::temp_dir().with_file_name(format!(
            "systolic-array-with-blocking-test-{}",
            std::time::SystemTime::now().elapsed().unwrap().as_nanos()
        ));
        println!("{}", binary_filepath.to_string_lossy());

        File::create(&main_c_filepath)
            .unwrap()
            .write_all(main_code.as_bytes())
            .unwrap();

        let result = Command::new("gcc")
            .arg("-Werror")
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
    fn relay_op_add() {
        let relay = r#"
#[version = "0.0.5"]
def @main(%x: Tensor[(1, 16, 16, 3), float32], %y: Tensor[(1, 1, 3), float32]) {
  add(%x, %y)
}
"#;

        let module = tvm::ir::module::IRModule::parse("", relay).unwrap();

        let (expr, shapes_vec) = crate::language::from_relay::from_relay(
            &module,
            true,
            &vec![crate::language::RelayOperator::RelayAdd],
        );

        let mut env = HashMap::default();
        for (k, v) in &shapes_vec {
            env.insert(k.clone(), v.clone());
        }

        let mut egraph = EGraph::new(MyAnalysis {
            name_to_shape: env.clone(),
        });

        let id = egraph.add_expr(&expr);

        let x_input = ndarray::ArrayD::from_shape_vec(
            env.get("x").unwrap().clone(),
            (0..env.get("x").unwrap().iter().product::<usize>()).collect(),
        )
        .unwrap();
        let y_input = ndarray::ArrayD::from_shape_vec(
            env.get("y").unwrap().clone(),
            (0..env.get("y").unwrap().iter().product::<usize>()).collect(),
        )
        .unwrap();
        let result = (&x_input + &y_input).into_dyn();

        let code = codegen(
            &egraph,
            id,
            &HashMap::default(),
            "relay_add",
            "",
            &vec!["x", "y"],
            &generate_worklist_for_codegen(&egraph, id),
            true,
        );

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
    relay_add(out, x, y);

  for (int i = 0; i < {}; i++) {{
    assert(((float*)result)[i] == ((float*)out)[i]);
  }}
}}
",
            PathBuf::from_str(
                format!(
                    "{}/{}/{}",
                    env!("CARGO_MANIFEST_DIR"),
                    "c-files",
                    "relay-op-implementations.c"
                )
                .as_str()
            )
            .unwrap()
            .to_string_lossy(),
            c_assignment_string("", "x", DType::Fp32, &x_input.into_dyn().view()),
            c_assignment_string("", "y", DType::Fp32, &y_input.into_dyn().view()),
            c_assignment_string("", "result", DType::Fp32, &result.view()),
            c_assignment_string(
                "",
                "out",
                DType::Fp32,
                &ndarray::ArrayD::<f32>::zeros(result.shape()).view()
            ),
            code,
            result.shape().iter().product::<usize>()
        );

        let main_c_filepath = std::env::temp_dir().with_file_name(format!(
            "relay-op-add-test-{}.c",
            std::time::SystemTime::now().elapsed().unwrap().as_nanos()
        ));
        println!("{}", main_c_filepath.to_string_lossy());

        let binary_filepath = std::env::temp_dir().with_file_name(format!(
            "relay-op-add-test-{}",
            std::time::SystemTime::now().elapsed().unwrap().as_nanos()
        ));
        println!("{}", binary_filepath.to_string_lossy());

        File::create(&main_c_filepath)
            .unwrap()
            .write_all(main_code.as_bytes())
            .unwrap();

        let result = Command::new("gcc")
            .arg("-Werror")
            .arg("-g")
            .arg("-o")
            .arg(&binary_filepath)
            .arg(&main_c_filepath)
            .arg("-lm")
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
    fn relay_op_biasadd() {
        let relay = r#"
#[version = "0.0.5"]
def @main(%x: Tensor[(1, 1000), float32], %y: Tensor[(1000), float32]) {
  nn.bias_add(%x, %y)
}
"#;

        let module = tvm::ir::module::IRModule::parse("", relay).unwrap();

        let (expr, shapes_vec) = crate::language::from_relay::from_relay(
            &module,
            true,
            &vec![crate::language::RelayOperator::RelayBiasAdd],
        );

        let mut env = HashMap::default();
        for (k, v) in &shapes_vec {
            env.insert(k.clone(), v.clone());
        }

        let mut egraph = EGraph::new(MyAnalysis {
            name_to_shape: env.clone(),
        });

        let id = egraph.add_expr(&expr);

        let x_input = ndarray::ArrayD::from_shape_vec(
            env.get("x").unwrap().clone(),
            (0..env.get("x").unwrap().iter().product::<usize>()).collect(),
        )
        .unwrap();
        let y_input = ndarray::ArrayD::from_shape_vec(
            env.get("y").unwrap().clone(),
            (0..env.get("y").unwrap().iter().product::<usize>()).collect(),
        )
        .unwrap();
        let result = (&x_input + &y_input).into_dyn();

        let code = codegen(
            &egraph,
            id,
            &HashMap::default(),
            "relay_biasadd",
            "",
            &vec!["x", "y"],
            &generate_worklist_for_codegen(&egraph, id),
            true,
        );

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
    relay_biasadd(out, x, y);

  for (int i = 0; i < {}; i++) {{
    assert(((float*)result)[i] == ((float*)out)[i]);
  }}
}}
",
            PathBuf::from_str(
                format!(
                    "{}/{}/{}",
                    env!("CARGO_MANIFEST_DIR"),
                    "c-files",
                    "relay-op-implementations.c"
                )
                .as_str()
            )
            .unwrap()
            .to_string_lossy(),
            c_assignment_string("", "x", DType::Fp32, &x_input.into_dyn().view()),
            c_assignment_string("", "y", DType::Fp32, &y_input.into_dyn().view()),
            c_assignment_string("", "result", DType::Fp32, &result.view()),
            c_assignment_string(
                "",
                "out",
                DType::Fp32,
                &ndarray::ArrayD::<f32>::zeros(result.shape()).view()
            ),
            code,
            result.shape().iter().product::<usize>()
        );

        let main_c_filepath = std::env::temp_dir().with_file_name(format!(
            "relay-op-biasadd-test-{}.c",
            std::time::SystemTime::now().elapsed().unwrap().as_nanos()
        ));
        println!("{}", main_c_filepath.to_string_lossy());

        let binary_filepath = std::env::temp_dir().with_file_name(format!(
            "relay-op-biasadd-test-{}",
            std::time::SystemTime::now().elapsed().unwrap().as_nanos()
        ));
        println!("{}", binary_filepath.to_string_lossy());

        File::create(&main_c_filepath)
            .unwrap()
            .write_all(main_code.as_bytes())
            .unwrap();

        let result = Command::new("gcc")
            .arg("-Werror")
            .arg("-g")
            .arg("-o")
            .arg(&binary_filepath)
            .arg(&main_c_filepath)
            .arg("-lm")
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
    fn relay_op_batchnorm() {
        let relay = r#"
#[version = "0.0.5"]
def @main(%data: Tensor[(1, 2, 2, 16), float32], %bn_gamma: Tensor[(16), float32], %bn_beta: Tensor[(16), float32], %bn_mean: Tensor[(16), float32], %bn_var: Tensor[(16), float32]) -> (Tensor[(1, 2, 2, 16), float32], Tensor[(16), float32], Tensor[(16), float32]) {
  nn.batch_norm(%data, %bn_gamma, %bn_beta, %bn_mean, %bn_var, axis=3) /* ty=(Tensor[(1, 2, 2, 16), float32], Tensor[(16), float32], Tensor[(16), float32]) */
}
"#;
        let relay_to_run = r#"
#[version = "0.0.5"]
def @main(%data: Tensor[(1, 2, 2, 16), float32], %bn_gamma: Tensor[(16), float32], %bn_beta: Tensor[(16), float32], %bn_mean: Tensor[(16), float32], %bn_var: Tensor[(16), float32]) -> Tensor[(1, 2, 2, 16), float32] {
  %0 = add(%bn_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(16), float32] */;
  %1 = sqrt(%0) /* ty=Tensor[(16), float32] */;
  %2 = divide(1f /* ty=float32 */, %1) /* ty=Tensor[(16), float32] */;
  %3 = multiply(%2, %bn_gamma) /* ty=Tensor[(16), float32] */;
  %4 = multiply(%data, %3) /* ty=Tensor[(1, 2, 2, 16), float32] */;
  %5 = negative(%bn_mean) /* ty=Tensor[(16), float32] */;
  %6 = multiply(%5, %3) /* ty=Tensor[(16), float32] */;
  %7 = add(%6, %bn_beta) /* ty=Tensor[(16), float32] */;
  add(%4, %7) /* ty=Tensor[(1, 2, 2, 16), float32] */
}      
"#;
        // Random number generator for generating random tensors.
        const SEED: u64 = 23;
        let mut tensor_rng = SmallRng::seed_from_u64(SEED);

        let module = tvm::ir::module::IRModule::parse("", relay).unwrap();

        let (expr, shapes_vec) = crate::language::from_relay::from_relay(
            &module,
            true,
            &vec![crate::language::RelayOperator::RelayBatchNormInference],
        );

        let mut env = HashMap::default();
        let mut value_env = HashMap::default();
        for (k, v) in &shapes_vec {
            env.insert(k.clone(), v.clone());
            value_env.insert(
                k.clone(),
                ndarray::ArrayD::<f32>::random_using(
                    v.clone(),
                    Uniform::new(1f32, 2f32), // prevent NaN results
                    &mut tensor_rng,
                ),
            );
        }

        let mut egraph = EGraph::new(MyAnalysis {
            name_to_shape: env.clone(),
        });

        let id = egraph.add_expr(&expr);

        let result = run_relay(&value_env, &shapes_vec, relay_to_run);

        let code = codegen(
            &egraph,
            id,
            &HashMap::default(),
            "relay_batchnorm",
            "",
            &vec!["data", "bn_gamma", "bn_beta", "bn_mean", "bn_var"],
            &generate_worklist_for_codegen(&egraph, id),
            true,
        );
        let main_code = format!(
            "
#include <assert.h>
#include <math.h>
#include \"{}\"

{}
{}
{}
{}
{}
{}
{}
{}

int main() {{
  relay_batchnorm(out, data, bn_gamma, bn_beta, bn_mean, bn_var);

  for (int i = 0; i < {}; i++) {{
    assert(fabs(((float*)result)[i] - ((float*)out)[i]) < 0.00001);
  }}
}}
",
            PathBuf::from_str(
                format!(
                    "{}/{}/{}",
                    env!("CARGO_MANIFEST_DIR"),
                    "c-files",
                    "relay-op-implementations.c"
                )
                .as_str()
            )
            .unwrap()
            .to_string_lossy(),
            c_assignment_string(
                "",
                "data",
                DType::Fp32,
                &value_env.get("data").unwrap().view()
            ),
            c_assignment_string(
                "",
                "bn_gamma",
                DType::Fp32,
                &value_env.get("bn_gamma").unwrap().view()
            ),
            c_assignment_string(
                "",
                "bn_beta",
                DType::Fp32,
                &value_env.get("bn_beta").unwrap().view()
            ),
            c_assignment_string(
                "",
                "bn_mean",
                DType::Fp32,
                &value_env.get("bn_mean").unwrap().view()
            ),
            c_assignment_string(
                "",
                "bn_var",
                DType::Fp32,
                &value_env.get("bn_var").unwrap().view()
            ),
            c_assignment_string("", "result", DType::Fp32, &result.view()),
            c_assignment_string(
                "",
                "out",
                DType::Fp32,
                &ndarray::ArrayD::<f32>::zeros(result.shape()).view()
            ),
            code,
            result.shape().iter().product::<usize>()
        );

        let main_c_filepath = std::env::temp_dir().with_file_name(format!(
            "relay-op-batchnorm-test-{}.c",
            std::time::SystemTime::now().elapsed().unwrap().as_nanos()
        ));
        println!("{}", main_c_filepath.to_string_lossy());

        let binary_filepath = std::env::temp_dir().with_file_name(format!(
            "relay-op-batchnorm-test-{}",
            std::time::SystemTime::now().elapsed().unwrap().as_nanos()
        ));
        println!("{}", binary_filepath.to_string_lossy());

        File::create(&main_c_filepath)
            .unwrap()
            .write_all(main_code.as_bytes())
            .unwrap();

        let result = Command::new("gcc")
            .arg("-Werror")
            .arg("-g")
            .arg("-o")
            .arg(&binary_filepath)
            .arg(&main_c_filepath)
            .arg("-lm")
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
    fn relay_op_softmax() {
        let relay = r#"
#[version = "0.0.5"]
def @main(%data: Tensor[(1,100), float32]) -> Tensor[(1,100), float32] {
    nn.softmax(%data) /* ty=Tensor[(1,10), float32] */
}
"#;

        let module = tvm::ir::module::IRModule::parse("", relay).unwrap();

        let (expr, shapes_vec) = crate::language::from_relay::from_relay(
            &module,
            true,
            &vec![crate::language::RelayOperator::RelaySoftmax],
        );

        let mut env = HashMap::default();
        let mut value_env = HashMap::default();
        for (k, v) in &shapes_vec {
            env.insert(k.clone(), v.clone());
            value_env.insert(
                k.clone(),
                ndarray::ArrayD::from_shape_vec(
                    v.clone(),
                    (0..v.iter().product::<usize>()).map(|x| x as f32).collect(),
                )
                .unwrap(),
            );
        }

        let mut egraph = EGraph::new(MyAnalysis {
            name_to_shape: env.clone(),
        });

        let id = egraph.add_expr(&expr);

        let result_output = run_relay(&value_env, &shapes_vec, relay);

        let code = codegen(
            &egraph,
            id,
            &HashMap::default(),
            "relay_softmax",
            "",
            &vec!["data"],
            &generate_worklist_for_codegen(&egraph, id),
            true,
        );
        let main_code = format!(
            "
#include <assert.h>
#include <math.h>
#include \"{}\"

{}
{}
{}
{}

int main() {{
  relay_softmax(out, data);

  for (int i = 0; i < {}; i++) {{
    assert(fabs(((float*)result)[i] - ((float*)out)[i]) < 0.00001);
  }}
}}
",
            PathBuf::from_str(
                format!(
                    "{}/{}/{}",
                    env!("CARGO_MANIFEST_DIR"),
                    "c-files",
                    "relay-op-implementations.c"
                )
                .as_str()
            )
            .unwrap()
            .to_string_lossy(),
            c_assignment_string(
                "",
                "data",
                DType::Fp32,
                &value_env.get("data").unwrap().view()
            ),
            c_assignment_string("", "result", DType::Fp32, &result_output.view()),
            c_assignment_string(
                "",
                "out",
                DType::Fp32,
                &ndarray::ArrayD::<f32>::zeros(result_output.shape()).view()
            ),
            code,
            result_output.shape().iter().product::<usize>()
        );

        let main_c_filepath = std::env::temp_dir().with_file_name(format!(
            "relay-op-softmax-test-{}.c",
            std::time::SystemTime::now().elapsed().unwrap().as_nanos()
        ));
        println!("{}", main_c_filepath.to_string_lossy());

        let binary_filepath = std::env::temp_dir().with_file_name(format!(
            "relay-op-softmax-test-{}",
            std::time::SystemTime::now().elapsed().unwrap().as_nanos()
        ));
        println!("{}", binary_filepath.to_string_lossy());

        File::create(&main_c_filepath)
            .unwrap()
            .write_all(main_code.as_bytes())
            .unwrap();

        let result = Command::new("gcc")
            .arg("-Werror")
            .arg("-g")
            .arg("-o")
            .arg(&binary_filepath)
            .arg(&main_c_filepath)
            .arg("-lm")
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
    fn relay_op_relu() {
        let relay = r#"
#[version = "0.0.5"]
def @main(%x: Tensor[(1, 3, 3, 4), float32]) {
  nn.relu(%x)
}
"#;

        // Random number generator for generating random tensors.
        const SEED: u64 = 23;
        let mut tensor_rng = SmallRng::seed_from_u64(SEED);

        let module = tvm::ir::module::IRModule::parse("", relay).unwrap();

        let (expr, shapes_vec) = crate::language::from_relay::from_relay(
            &module,
            true,
            &vec![crate::language::RelayOperator::RelayReLU],
        );

        let mut env = HashMap::default();
        let mut value_env = HashMap::default();
        for (k, v) in &shapes_vec {
            env.insert(k.clone(), v.clone());
            value_env.insert(
                k.clone(),
                ndarray::ArrayD::<f32>::random_using(
                    v.clone(),
                    Uniform::new(-1f32, 1f32),
                    &mut tensor_rng,
                ),
            );
        }

        let mut egraph = EGraph::new(MyAnalysis {
            name_to_shape: env.clone(),
        });

        let id = egraph.add_expr(&expr);

        let result = run_relay(&value_env, &shapes_vec, relay);

        let code = codegen(
            &egraph,
            id,
            &HashMap::default(),
            "relay_relu",
            "",
            &vec!["x"],
            &generate_worklist_for_codegen(&egraph, id),
            true,
        );

        let main_code = format!(
            "
#include <assert.h>
#include \"{}\"

{}
{}
{}
{}

int main() {{
    relay_relu(out, x);

  for (int i = 0; i < {}; i++) {{
    assert(fabs(((float*)result)[i] - ((float*)out)[i]) < 0.00001);
  }}
}}
",
            PathBuf::from_str(
                format!(
                    "{}/{}/{}",
                    env!("CARGO_MANIFEST_DIR"),
                    "c-files",
                    "relay-op-implementations.c"
                )
                .as_str()
            )
            .unwrap()
            .to_string_lossy(),
            c_assignment_string("", "x", DType::Fp32, &value_env.get("x").unwrap().view()),
            c_assignment_string("", "result", DType::Fp32, &result.view()),
            c_assignment_string(
                "",
                "out",
                DType::Fp32,
                &ndarray::ArrayD::<f32>::zeros(result.shape()).view()
            ),
            code,
            result.shape().iter().product::<usize>()
        );

        let main_c_filepath = std::env::temp_dir().with_file_name(format!(
            "relay-op-relu-test-{}.c",
            std::time::SystemTime::now().elapsed().unwrap().as_nanos()
        ));
        println!("{}", main_c_filepath.to_string_lossy());

        let binary_filepath = std::env::temp_dir().with_file_name(format!(
            "relay-op-relu-test-{}",
            std::time::SystemTime::now().elapsed().unwrap().as_nanos()
        ));
        println!("{}", binary_filepath.to_string_lossy());

        File::create(&main_c_filepath)
            .unwrap()
            .write_all(main_code.as_bytes())
            .unwrap();

        let result = Command::new("gcc")
            .arg("-Werror")
            .arg("-g")
            .arg("-o")
            .arg(&binary_filepath)
            .arg(&main_c_filepath)
            .arg("-lm")
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
    fn relay_op_maxpool2d_resnet_3x3() {
        let relay = r#"
#[version = "0.0.5"]
def @main(%x: Tensor[(1, 112, 112, 64), float32]) -> Tensor[(1, 56, 56, 64), float32] {
  nn.max_pool2d(%x, pool_size=[3, 3], strides=[2, 2], padding=[1, 1, 1, 1], layout="NHWC") /* ty=Tensor[(1, 56, 56, 64), float32] */
}
"#;
        const SEED: u64 = 23;
        let mut tensor_rng = SmallRng::seed_from_u64(SEED);

        let module = tvm::ir::module::IRModule::parse("", relay.clone()).unwrap();

        let (expr, shapes_vec) = crate::language::from_relay::from_relay(
            &module,
            true,
            &vec![crate::language::RelayOperator::RelayMaxPool2D],
        );

        let mut env = HashMap::default();
        let mut value_env = HashMap::default();
        for (k, v) in &shapes_vec {
            env.insert(k.clone(), v.clone());
            value_env.insert(
                k.clone(),
                ndarray::ArrayD::<f32>::random_using(
                    v.clone(),
                    Uniform::new(-1f32, 1f32),
                    &mut tensor_rng,
                ),
            );
        }

        let mut egraph = EGraph::new(MyAnalysis {
            name_to_shape: env.clone(),
        });

        let id = egraph.add_expr(&expr);

        let result = run_relay(&value_env, &shapes_vec, relay);

        let code = codegen(
            &egraph,
            id,
            &HashMap::default(),
            "relay_maxpool",
            "",
            &vec!["x"],
            &generate_worklist_for_codegen(&egraph, id),
            true,
        );

        let main_code = format!(
            "
#include <assert.h>
#include <math.h>
#include \"{}\"

{}
{}
{}
{}

int main() {{
    relay_maxpool(out, x);

  for (int i = 0; i < {}; i++) {{
    assert(fabs(((float*)result)[i] - ((float*)out)[i]) < 0.00001);
  }}
}}
",
            PathBuf::from_str(
                format!(
                    "{}/{}/{}",
                    env!("CARGO_MANIFEST_DIR"),
                    "c-files",
                    "relay-op-implementations.c"
                )
                .as_str()
            )
            .unwrap()
            .to_string_lossy(),
            c_assignment_string("", "x", DType::Fp32, &value_env.get("x").unwrap().view()),
            c_assignment_string("", "result", DType::Fp32, &result.view()),
            c_assignment_string(
                "",
                "out",
                DType::Fp32,
                &ndarray::ArrayD::<f32>::zeros(result.shape()).view()
            ),
            code,
            result.shape().iter().product::<usize>()
        );

        let main_c_filepath = std::env::temp_dir().with_file_name(format!(
            "relay-op-maxpool-test-{}.c",
            std::time::SystemTime::now().elapsed().unwrap().as_nanos()
        ));
        println!("{}", main_c_filepath.to_string_lossy());

        let binary_filepath = std::env::temp_dir().with_file_name(format!(
            "relay-op-maxpool-test-{}",
            std::time::SystemTime::now().elapsed().unwrap().as_nanos()
        ));
        println!("{}", binary_filepath.to_string_lossy());

        File::create(&main_c_filepath)
            .unwrap()
            .write_all(main_code.as_bytes())
            .unwrap();

        let result = Command::new("gcc")
            .arg("-Werror")
            .arg("-g")
            .arg("-o")
            .arg(&binary_filepath)
            .arg(&main_c_filepath)
            .arg("-lm")
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
    fn relay_op_batchflatten() {
        let relay = r#"
#[version = "0.0.5"]
def @main(%x: Tensor[(1, 512, 1, 1), float32]) {
  nn.batch_flatten(%x)
}
"#;
        const SEED: u64 = 23;
        let mut tensor_rng = SmallRng::seed_from_u64(SEED);

        let module = tvm::ir::module::IRModule::parse("", relay).unwrap();

        let (expr, shapes_vec) = crate::language::from_relay::from_relay(
            &module,
            true,
            &vec![crate::language::RelayOperator::RelayBatchFlatten],
        );

        let mut env = HashMap::default();
        let mut value_env = HashMap::default();
        for (k, v) in &shapes_vec {
            env.insert(k.clone(), v.clone());
            value_env.insert(
                k.clone(),
                ndarray::ArrayD::<f32>::random_using(
                    v.clone(),
                    Uniform::new(-2f32, 2f32),
                    &mut tensor_rng,
                ),
            );
        }

        let mut egraph = EGraph::new(MyAnalysis {
            name_to_shape: env.clone(),
        });

        let id = egraph.add_expr(&expr);

        let result = run_relay(&value_env, &shapes_vec, relay);

        let code = codegen(
            &egraph,
            id,
            &HashMap::default(),
            "relay_maxpool",
            "",
            &vec!["x"],
            &generate_worklist_for_codegen(&egraph, id),
            true,
        );

        let main_code = format!(
            "
#include <assert.h>
#include <math.h>
#include \"{}\"

{}
{}
{}
{}

int main() {{
    relay_maxpool(out, x);

  for (int i = 0; i < {}; i++) {{
    assert(fabs(((float*)result)[i] - ((float*)out)[i]) < 0.00001);
  }}
}}
",
            PathBuf::from_str(
                format!(
                    "{}/{}/{}",
                    env!("CARGO_MANIFEST_DIR"),
                    "c-files",
                    "relay-op-implementations.c"
                )
                .as_str()
            )
            .unwrap()
            .to_string_lossy(),
            c_assignment_string("", "x", DType::Fp32, &value_env.get("x").unwrap().view()),
            c_assignment_string("", "result", DType::Fp32, &result.view()),
            c_assignment_string(
                "",
                "out",
                DType::Fp32,
                &ndarray::ArrayD::<f32>::zeros(result.shape()).view()
            ),
            code,
            result.shape().iter().product::<usize>()
        );

        let main_c_filepath = std::env::temp_dir().with_file_name(format!(
            "relay-op-maxpool-test-{}.c",
            std::time::SystemTime::now().elapsed().unwrap().as_nanos()
        ));
        println!("{}", main_c_filepath.to_string_lossy());

        let binary_filepath = std::env::temp_dir().with_file_name(format!(
            "relay-op-maxpool-test-{}",
            std::time::SystemTime::now().elapsed().unwrap().as_nanos()
        ));
        println!("{}", binary_filepath.to_string_lossy());

        File::create(&main_c_filepath)
            .unwrap()
            .write_all(main_code.as_bytes())
            .unwrap();

        let result = Command::new("gcc")
            .arg("-Werror")
            .arg("-g")
            .arg("-o")
            .arg(&binary_filepath)
            .arg(&main_c_filepath)
            .arg("-lm")
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
    fn relay_op_globalavgpool2d() {
        let relay = r#"
#[version = "0.0.5"]
def @main(%x: Tensor[(1, 7, 7, 512), float32]) -> Tensor[(1, 1, 1, 512), float32] {
    nn.global_avg_pool2d(%x, layout="NHWC") /* ty=Tensor[(1, 1, 1, 512), float32] */
}
"#;
        const SEED: u64 = 23;
        let mut tensor_rng = SmallRng::seed_from_u64(SEED);

        let module = tvm::ir::module::IRModule::parse("", relay.clone()).unwrap();

        let (expr, shapes_vec) = crate::language::from_relay::from_relay(
            &module,
            true,
            &vec![crate::language::RelayOperator::RelayGlobalAvgPool2D],
        );

        let mut env = HashMap::default();
        let mut value_env = HashMap::default();
        for (k, v) in &shapes_vec {
            env.insert(k.clone(), v.clone());
            value_env.insert(
                k.clone(),
                ndarray::ArrayD::<f32>::random_using(
                    v.clone(),
                    Uniform::new(-2f32, 2f32),
                    &mut tensor_rng,
                ),
            );
        }

        let mut egraph = EGraph::new(MyAnalysis {
            name_to_shape: env.clone(),
        });

        let id = egraph.add_expr(&expr);

        let result = run_relay(&value_env, &shapes_vec, relay);

        let code = codegen(
            &egraph,
            id,
            &HashMap::default(),
            "relay_globalavgpool2d",
            "",
            &vec!["x"],
            &generate_worklist_for_codegen(&egraph, id),
            true,
        );

        let main_code = format!(
            "
#include <assert.h>
#include <math.h>
#include \"{}\"

{}
{}
{}
{}

int main() {{
    relay_globalavgpool2d(out, x);

  for (int i = 0; i < {}; i++) {{
    assert(fabs(((float*)result)[i] - ((float*)out)[i]) < 0.00001);
  }}
}}
",
            PathBuf::from_str(
                format!(
                    "{}/{}/{}",
                    env!("CARGO_MANIFEST_DIR"),
                    "c-files",
                    "relay-op-implementations.c"
                )
                .as_str()
            )
            .unwrap()
            .to_string_lossy(),
            c_assignment_string("", "x", DType::Fp32, &value_env.get("x").unwrap().view()),
            c_assignment_string("", "result", DType::Fp32, &result.view()),
            c_assignment_string(
                "",
                "out",
                DType::Fp32,
                &ndarray::ArrayD::<f32>::zeros(result.shape()).view()
            ),
            code,
            result.shape().iter().product::<usize>()
        );

        let main_c_filepath = std::env::temp_dir().with_file_name(format!(
            "relay-op-globalavgpool2d-test-{}.c",
            std::time::SystemTime::now().elapsed().unwrap().as_nanos()
        ));
        println!("{}", main_c_filepath.to_string_lossy());

        let binary_filepath = std::env::temp_dir().with_file_name(format!(
            "relay-op-globalavgpool2d-test-{}",
            std::time::SystemTime::now().elapsed().unwrap().as_nanos()
        ));
        println!("{}", binary_filepath.to_string_lossy());

        File::create(&main_c_filepath)
            .unwrap()
            .write_all(main_code.as_bytes())
            .unwrap();

        let result = Command::new("gcc")
            .arg("-Werror")
            .arg("-g")
            .arg("-o")
            .arg(&binary_filepath)
            .arg(&main_c_filepath)
            .arg("-lm")
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
    fn relay_model_yolov3() {
        // Generate yolov3 with directions from:
        // https://tvm.apache.org/docs/tutorials/frontend/from_darknet.html
        let filename = PathBuf::from(format!(
            "{}/models/yolov3.relay",
            env!("CARGO_MANIFEST_DIR")
        ));
        let relay = std::fs::read_to_string(&filename).unwrap();
        const SEED: u64 = 23;
        let mut tensor_rng = SmallRng::seed_from_u64(SEED);

        let module = tvm::ir::module::IRModule::parse("", relay.clone()).unwrap();
        let (expr, shapes_vec) = crate::language::from_relay::from_relay(
            &module,
            true,
            &vec![
                crate::language::RelayOperator::RelayBatchNormInference,
                crate::language::RelayOperator::RelaySoftmax,
                crate::language::RelayOperator::RelayReLU,
                crate::language::RelayOperator::RelayLeakyReLU,
                crate::language::RelayOperator::RelayMaxPool2D,
                crate::language::RelayOperator::RelayGlobalAvgPool2D,
                crate::language::RelayOperator::RelayBatchFlatten,
                crate::language::RelayOperator::RelayBiasAdd,
                crate::language::RelayOperator::RelayAdd,
                crate::language::RelayOperator::RelaySigmoid,
            ],
        );
        let mut env = HashMap::default();
        let mut value_env = HashMap::default();
        for (k, v) in &shapes_vec {
            env.insert(k.clone(), v.clone());
            value_env.insert(
                k.clone(),
                ndarray::ArrayD::<f32>::random_using(
                    v.clone(),
                    Uniform::new(-2f32, 2f32),
                    &mut tensor_rng,
                ),
            );
        }

        let mut egraph = EGraph::new(MyAnalysis {
            name_to_shape: env.clone(),
        });

        let id = egraph.add_expr(&expr);

        let result = run_relay(&value_env, &shapes_vec, &relay);

        let code = codegen(
            &egraph,
            id,
            &HashMap::default(),
            "yolo",
            "",
            &vec!["x"],
            &generate_worklist_for_codegen(&egraph, id),
            true,
        );

        let main_code = format!(
            "
#include <assert.h>
#include <math.h>
#include \"{}\"

{}
{}
{}
{}

int main() {{
    yolo(out, x);

  for (int i = 0; i < {}; i++) {{
    assert(fabs(((float*)result)[i] - ((float*)out)[i]) < 0.00001);
  }}
}}
",
            PathBuf::from_str(
                format!(
                    "{}/{}/{}",
                    env!("CARGO_MANIFEST_DIR"),
                    "c-files",
                    "relay-op-implementations.c"
                )
                .as_str()
            )
            .unwrap()
            .to_string_lossy(),
            c_assignment_string("", "x", DType::Fp32, &value_env.get("x").unwrap().view()),
            c_assignment_string("", "result", DType::Fp32, &result.view()),
            c_assignment_string(
                "",
                "out",
                DType::Fp32,
                &ndarray::ArrayD::<f32>::zeros(result.shape()).view()
            ),
            code,
            result.shape().iter().product::<usize>()
        );

        let main_c_filepath = std::env::temp_dir().with_file_name(format!(
            "relay-op-yolo-test-{}.c",
            std::time::SystemTime::now().elapsed().unwrap().as_nanos()
        ));
        println!("{}", main_c_filepath.to_string_lossy());

        let binary_filepath = std::env::temp_dir().with_file_name(format!(
            "relay-op-yolo-test-{}",
            std::time::SystemTime::now().elapsed().unwrap().as_nanos()
        ));
        println!("{}", binary_filepath.to_string_lossy());

        File::create(&main_c_filepath)
            .unwrap()
            .write_all(main_code.as_bytes())
            .unwrap();

        let result = Command::new("gcc")
            .arg("-Werror")
            .arg("-g")
            .arg("-o")
            .arg(&binary_filepath)
            .arg(&main_c_filepath)
            .arg("-lm")
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
    fn relay_model_efficientnet_lite4_11() {
        // efficientnet onnx model source: https://github.com/onnx/models/blob/master/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx
        // imported into relay
        let filename = PathBuf::from(format!(
            "{}/models/efficientnet-lite4-11.relay",
            env!("CARGO_MANIFEST_DIR")
        ));
        let relay = std::fs::read_to_string(&filename).unwrap();
        const SEED: u64 = 23;
        let mut tensor_rng = SmallRng::seed_from_u64(SEED);

        let module = tvm::ir::module::IRModule::parse("", relay.clone()).unwrap();
        let (expr, shapes_vec) = crate::language::from_relay::from_relay(
            &module,
            true,
            &vec![
                crate::language::RelayOperator::RelayBatchNormInference,
                crate::language::RelayOperator::RelaySoftmax,
                crate::language::RelayOperator::RelayReLU,
                crate::language::RelayOperator::RelayMaxPool2D,
                crate::language::RelayOperator::RelayGlobalAvgPool2D,
                crate::language::RelayOperator::RelayBatchFlatten,
                crate::language::RelayOperator::RelayBiasAdd,
                crate::language::RelayOperator::RelayAdd,
            ],
        );
        let mut env = HashMap::default();
        let mut value_env = HashMap::default();
        for (k, v) in &shapes_vec {
            env.insert(k.clone(), v.clone());
            value_env.insert(
                k.clone(),
                ndarray::ArrayD::<f32>::random_using(
                    v.clone(),
                    Uniform::new(-2f32, 2f32),
                    &mut tensor_rng,
                ),
            );
        }

        let mut egraph = EGraph::new(MyAnalysis {
            name_to_shape: env.clone(),
        });

        let id = egraph.add_expr(&expr);

        let result = run_relay(&value_env, &shapes_vec, &relay);

        let code = codegen(
            &egraph,
            id,
            &HashMap::default(),
            "efficientnet",
            "",
            &vec!["x"],
            &generate_worklist_for_codegen(&egraph, id),
            true,
        );

        let main_code = format!(
            "
#include <assert.h>
#include <math.h>
#include \"{}\"

{}
{}
{}
{}

int main() {{
    efficientnet(out, x);

  for (int i = 0; i < {}; i++) {{
    assert(fabs(((float*)result)[i] - ((float*)out)[i]) < 0.00001);
  }}
}}
",
            PathBuf::from_str(
                format!(
                    "{}/{}/{}",
                    env!("CARGO_MANIFEST_DIR"),
                    "c-files",
                    "relay-op-implementations.c"
                )
                .as_str()
            )
            .unwrap()
            .to_string_lossy(),
            c_assignment_string("", "x", DType::Fp32, &value_env.get("x").unwrap().view()),
            c_assignment_string("", "result", DType::Fp32, &result.view()),
            c_assignment_string(
                "",
                "out",
                DType::Fp32,
                &ndarray::ArrayD::<f32>::zeros(result.shape()).view()
            ),
            code,
            result.shape().iter().product::<usize>()
        );

        let main_c_filepath = std::env::temp_dir().with_file_name(format!(
            "relay-op-efficientnet-test-{}.c",
            std::time::SystemTime::now().elapsed().unwrap().as_nanos()
        ));
        println!("{}", main_c_filepath.to_string_lossy());

        let binary_filepath = std::env::temp_dir().with_file_name(format!(
            "relay-op-efficientnet-test-{}",
            std::time::SystemTime::now().elapsed().unwrap().as_nanos()
        ));
        println!("{}", binary_filepath.to_string_lossy());

        File::create(&main_c_filepath)
            .unwrap()
            .write_all(main_code.as_bytes())
            .unwrap();

        let result = Command::new("gcc")
            .arg("-Werror")
            .arg("-g")
            .arg("-o")
            .arg(&binary_filepath)
            .arg(&main_c_filepath)
            .arg("-lm")
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
