use crate::hw_design_language::*;
use crate::language::MyAnalysis;
use crate::language::MyAnalysisData;
use crate::language::RelayActivationLayout;
use crate::language::{Language, PadType, RelayOperator};
use egg::EGraph;
use egg::Id;
use itertools::Itertools;
use ndarray::array;
use ndarray::Array;
use ndarray::Dimension;
use ndarray::IxDyn;
use ndarray::array;
use rand::Rng;
use std::cmp;
use std::collections::{HashMap, HashSet};
use std::iter::FromIterator;

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
            Language::RelayOperatorCall(ids) => {
                for id in ids.iter() {
                    find_vars_recursive_helper(set, expr, *id);
                }
            },
            Language::RelayOperator(_) => {

            },
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
            Language::List(ids) | Language::Shape(ids) => {
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
            | &Language::AccessSqueeze(ids) => {
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
            &Language::NotNanFloat64(_) => {

            }
            &Language::Usize(_) | &Language::PadType(_) => (),
            &Language::Literal(_)
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

/// Returns c code.
/// args: The signature will be `void <function_name>(float * out, float * <arg0>...)`
/// uninitialized_allocations_prefix: The prefix to use for buffer allocations
/// that do not need to be initialized. In the future, once we have literals in
/// the program, we will need to also include an initialized_allocations_prefix.
// TODO(@gussmith23) Does not reason about ordering on hardware.
// TODO(@gussmith23) Hardcoded to float32
pub fn codegen(
    expr: &Expr,
    id: Id,
    hw_map: &HashMap<Id, usize>,
    function_name: &str,
    uninitialized_allocations_prefix: &str,
    args: &Vec<&str>,
) -> String {
    let mut declarations = String::default();
    let mut code = String::default();
    let out_symbol = codegen_recursive_helper(
        expr,
        id,
        id,
        uninitialized_allocations_prefix,
        &mut declarations,
        &mut code,
        hw_map,
    );

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
fn codegen_recursive_helper(
    expr: &Expr,
    id: Id,
    top_level_id: Id,
    uninitialized_allocations_prefix: &str,
    declarations: &mut String,
    code: &mut String,
    hw_map: &HashMap<Id, usize>,
) -> String {
    match {
        assert_eq!(expr[id].nodes.len(), 1);
        &expr[id].nodes[0]
    } {
        Language::RelayOperatorCall(ids) => {
            let relay_op = match &expr[ids[0]].data {
                MyAnalysisData::RelayOperator(op) => op,
                _ => panic!()
            };

            match relay_op {
                RelayOperator::RelayBatchNormInference => {
                    let data = codegen_recursive_helper(
                        expr,
                        ids[1],
                        top_level_id,
                        uninitialized_allocations_prefix,
                        declarations,
                        code,
                        hw_map,
                    );
                    let gamma = codegen_recursive_helper(
                        expr,
                        ids[2],
                        top_level_id,
                        uninitialized_allocations_prefix,
                        declarations,
                        code,
                        hw_map,
                    );
                    let beta = codegen_recursive_helper(
                        expr,
                        ids[3],
                        top_level_id,
                        uninitialized_allocations_prefix,
                        declarations,
                        code,
                        hw_map,
                    );
                    
                    let moving_mean = codegen_recursive_helper(
                        expr,
                        ids[4],
                        top_level_id,
                        uninitialized_allocations_prefix,
                        declarations,
                        code,
                        hw_map,
                    );

                    let moving_var = codegen_recursive_helper(
                        expr,
                        ids[5],
                        top_level_id,
                        uninitialized_allocations_prefix,
                        declarations,
                        code,
                        hw_map,
                    );

                    let axis = MyAnalysis::get_usize(ids[6], expr);

                    // Pre-ISCA: expect NHWC format data
                    assert!(axis == 3, "expected NHWC format");
                    let epsilon = match &expr[ids[7]].data {
                        MyAnalysisData::Literal(l) => {
                            println!("shape of batchnorm inference epsilon: {:?}", &l.shape());
                            l
                        },
                        _ => panic!()
                    };

                    let new_shape = match &expr[ids[1]].data {
                        MyAnalysisData::AccessPattern(a) => a.as_vec(),
                        _ => panic!()
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
batchNormInference({X}, {Y}, {N}, {H}, {W}, {C}, {gamma}, {beta}, {moving_mean}, {moving_var}, {epsilon});
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

                    batchnorm_out
                },
                RelayOperator::RelaySoftmax => {
                    let data = codegen_recursive_helper(
                        expr,
                        ids[1],
                        top_level_id,
                        uninitialized_allocations_prefix,
                        declarations,
                        code,
                        hw_map,
                    );

                    let axis = MyAnalysis::get_usize(ids[2], expr);
                    
                    // Pre-ISCA: resnet only does softmax over (1,1000)
                    println!("{}", axis);
                    assert!(axis == 1, "expected NHWC format");

                    let new_shape = match &expr[ids[1]].data {
                        MyAnalysisData::AccessPattern(a) => a.as_vec(),
                        _ => panic!()
                    };

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

                    code.push_str(format!("
softmax1D({X}, {Y}, {N});
",                      X = data,
                        Y = softmax_out,
                        N = new_shape.iter().product::<usize>()
                    )
                    .as_str());

                    softmax_out
                },
                RelayOperator::RelayReLU => {
                    let data = codegen_recursive_helper(
                        expr,
                        ids[1],
                        top_level_id,
                        uninitialized_allocations_prefix,
                        declarations,
                        code,
                        hw_map,
                    );

                    let new_shape = match &expr[ids[1]].data {
                        MyAnalysisData::AccessPattern(a) => a.as_vec(),
                        _ => panic!()
                    };

                    // TODO: axis currently not used...
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

                    code.push_str(format!("
relu({X}, {Y}, {N}, {H}, {W}, {C});
",                      X = data,
                        Y = relu_out,
                        N = new_shape[0],
                        H = new_shape[1],
                        W = new_shape[2],
                        C = new_shape[3]
                    )
                    .as_str());

                    relu_out
                },
                RelayOperator::RelayMaxPool2D => {
                    let data = codegen_recursive_helper(
                        expr,
                        ids[1],
                        top_level_id,
                        uninitialized_allocations_prefix,
                        declarations,
                        code,
                        hw_map,
                    );

                    let pool_size = MyAnalysis::get_shape_of_value(ids[2], expr);
                    let strides = MyAnalysis::get_shape_of_value(ids[3], expr);
                    let padding = MyAnalysis::get_shape_of_value(ids[4], expr);

                    // TODO: currently hardcoded shape for max pool2d for resnet
                    let old_shape = match &expr[ids[1]].data {
                        MyAnalysisData::AccessPattern(a) => a.as_vec(),
                        _ => panic!()
                    };

                    let new_shape = vec![old_shape[0], 56, 56, old_shape[3]];
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

                    code.push_str(format!("
maxpool2D3x3_resnet18_op6({X}, {Y});
",                      X = data,
                        Y = data
                    )
                    .as_str());

                    maxpool2d_out
                },
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

                    let data = codegen_recursive_helper(
                        expr,
                        ids[1],
                        top_level_id,
                        uninitialized_allocations_prefix,
                        declarations,
                        code,
                        hw_map,
                    );

                    // TODO: support broadcasting
                    let old_shape = match &expr[ids[1]].data {
                        MyAnalysisData::AccessPattern(a) => a.as_vec(),
                        _ => panic!()
                    };

                    let new_shape = vec![old_shape[0], old_shape[3]];
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

                    code.push_str(format!("
globalAvgPool({X}, {Y}, {N}, {H}, {W}, {C});
",                      X = data,
                        Y = globalavgpool2d_out,
                        N = old_shape[0],
                        H = old_shape[1],
                        W = old_shape[2],
                        C = old_shape[3]
                    )
                    .as_str());

                    globalavgpool2d_out
                },
                RelayOperator::RelayBatchFlatten => {
                    let data = codegen_recursive_helper(
                        expr,
                        ids[1],
                        top_level_id,
                        uninitialized_allocations_prefix,
                        declarations,
                        code,
                        hw_map,
                    );

                    // just a reshape, which is a no-op!
                    data
                },
                RelayOperator::RelayBiasAdd => {
                    let data = codegen_recursive_helper(
                        expr,
                        ids[1],
                        top_level_id,
                        uninitialized_allocations_prefix,
                        declarations,
                        code,
                        hw_map,
                    );
                    let bias = codegen_recursive_helper(
                        expr,
                        ids[2],
                        top_level_id,
                        uninitialized_allocations_prefix,
                        declarations,
                        code,
                        hw_map,
                    );

                    // TODO: support broadcasting
                    let new_shape = match &expr[ids[1]].data {
                        MyAnalysisData::AccessPattern(a) => a.as_vec(),
                        _ => panic!()
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

                    code.push_str(format!("
add({X}, {Y}, {out}, {N}, {H}, {W}, {C});
",                      X = data,
                        Y = bias,
                        out = add_out,
                        N = new_shape[0],
                        H = new_shape[1],
                        W = new_shape[2],
                        C = new_shape[3]
                    )
                    .as_str());

                    add_out
                },
                RelayOperator::RelayAdd => {
                    let a = codegen_recursive_helper(
                        expr,
                        ids[1],
                        top_level_id,
                        uninitialized_allocations_prefix,
                        declarations,
                        code,
                        hw_map,
                    );
                    let b = codegen_recursive_helper(
                        expr,
                        ids[2],
                        top_level_id,
                        uninitialized_allocations_prefix,
                        declarations,
                        code,
                        hw_map,
                    );
                    
                    // TODO: support broadcasting
                    // TODO: cannot assume adding 4d tensors...
                    let new_shape = match &expr[ids[1]].data {
                        MyAnalysisData::AccessPattern(a) => a.as_vec(),
                        _ => panic!()
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

                    code.push_str(format!("
add({X}, {Y}, {out}, {N}, {H}, {W}, {C});
",                      X = a,
                        Y = b,
                        out = add_out,
                        N = new_shape[0],
                        H = new_shape[1],
                        W = new_shape[2],
                        C = new_shape[3]
                    )
                    .as_str());

                    add_out
                }
            }
        },
        Language::RelayActivationLayout(_) => panic!(),
        Language::RelayKernelLayout(_) => panic!(),
        Language::RelayOperator(_) => todo!(),
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
            let access_var_name = codegen_recursive_helper(
                expr,
                access_id,
                top_level_id,
                uninitialized_allocations_prefix,
                declarations,
                code,
                hw_map,
            );

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

            access_windows_out_var_name
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

            let access_var_name = codegen_recursive_helper(
                expr,
                access_id,
                top_level_id,
                uninitialized_allocations_prefix,
                declarations,
                code,
                hw_map,
            );

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
{out_name}{out_index} = {in_name}{in_index};
",
                    out_name = slice_out_var_name,
                    out_index = (0..new_shape.len())
                        .map(|i| format!("[i{}]", i,))
                        .collect::<String>(),
                    in_name = access_var_name,
                    in_index = (0..original_shape.len())
                        .map(|i| if i != axis {
                            format!("[i{}]", i)
                        } else {
                            format!("[i{}+{}]", i, low,)
                        })
                        .collect::<String>()
                )
                .as_str(),
            );

            // Close each for loop
            for _ in new_shape.iter() {
                code.push_str("}");
            }

            slice_out_var_name
        }
        Language::Symbol(s) => s.clone(),
        &Language::AccessTensor(symbol_id) => {
            let symbol = codegen_recursive_helper(
                expr,
                symbol_id,
                top_level_id,
                uninitialized_allocations_prefix,
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
            uninitialized_allocations_prefix,
            declarations,
            code,
            hw_map,
        ),
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

            let s0 = codegen_recursive_helper(
                expr,
                a0_id,
                top_level_id,
                uninitialized_allocations_prefix,
                declarations,
                code,
                hw_map,
            );
            let s1 = codegen_recursive_helper(
                expr,
                a1_id,
                top_level_id,
                uninitialized_allocations_prefix,
                declarations,
                code,
                hw_map,
            );

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
                uninitialized_allocations_prefix,
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

            let new_shape = match &expr[id].data {
                MyAnalysisData::AccessPattern(a) => a.as_vec(),
                _ => panic!(),
            };

            let new_axis_order = match &expr[list_id].data {
                MyAnalysisData::List(l) => l,
                _ => panic!(),
            };

            assert_eq!(original_shape.len(), new_axis_order.len());

            let access_var_name = codegen_recursive_helper(
                expr,
                access_id,
                top_level_id,
                uninitialized_allocations_prefix,
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
{out_var_name}{out_indices} = {access_var_name}{access_indices};",
                    out_var_name = transpose_out_var_name,
                    out_indices = (0..original_shape.len())
                        .map(|i| format!("[i{}]", new_axis_order[i]))
                        .collect::<Vec<_>>()
                        .join(""),
                    access_var_name = access_var_name,
                    access_indices = (0..original_shape.len())
                        .map(|i| format!("[i{}]", i))
                        .collect::<Vec<_>>()
                        .join("")
                )
                .as_str(),
            );

            // Close each for loop
            for _ in original_shape.iter() {
                code.push_str("}");
            }

            transpose_out_var_name
        }
        &Language::AccessFlatten(access_id) => {
            // Flatten doesn't do anything. Just copy the array verbatim.

            let out_shape = match &expr[id].data {
                MyAnalysisData::AccessPattern(a) => a.as_vec(),
                _ => panic!(),
            };
            let out_var_name: String = {
                // TODO(@gussmith23) Find a different way to name intermediates
                // Currently generating random strings. Not great IMO.
                let out = format!(
                    "access_flatten_out_{}",
                    rand::thread_rng()
                        .sample_iter(&rand::distributions::Alphanumeric)
                        .take(30)
                        .collect::<String>()
                );
                declarations.push_str(
                    c_allocation_string(
                        uninitialized_allocations_prefix,
                        out.as_str(),
                        out_shape.as_slice(),
                        DType::Fp32,
                    )
                    .as_str(),
                );
                out
            };

            let in_var_name = codegen_recursive_helper(
                expr,
                access_id,
                top_level_id,
                uninitialized_allocations_prefix,
                declarations,
                code,
                hw_map,
            );

            code.push_str(
                format!(
                    "
for (int i = 0; i < {limit}; ++i) {{
  ((float*){out_var_name})[i] = ((float*){in_var_name})[i];
}}",
                    out_var_name = out_var_name,
                    in_var_name = in_var_name,
                    limit = out_shape.iter().product::<usize>(),
                )
                .as_str(),
            );

            out_var_name
        }
        &Language::AccessReshape([access_id, _access_shape_id]) => {
            // Reshape doesn't do anything. Just copy the array verbatim.

            let out_shape = match &expr[id].data {
                MyAnalysisData::AccessPattern(a) => a.as_vec(),
                _ => panic!(),
            };
            let out_var_name: String = {
                // TODO(@gussmith23) Find a different way to name intermediates
                // Currently generating random strings. Not great IMO.
                let out = format!(
                    "access_reshape_out_{}",
                    rand::thread_rng()
                        .sample_iter(&rand::distributions::Alphanumeric)
                        .take(30)
                        .collect::<String>()
                );
                declarations.push_str(
                    c_allocation_string(
                        uninitialized_allocations_prefix,
                        out.as_str(),
                        out_shape.as_slice(),
                        DType::Fp32,
                    )
                    .as_str(),
                );
                out
            };

            let in_var_name = codegen_recursive_helper(
                expr,
                access_id,
                top_level_id,
                uninitialized_allocations_prefix,
                declarations,
                code,
                hw_map,
            );

            code.push_str(
                format!(
                    "
for (int i = 0; i < {limit}; ++i) {{
  ((float*){out_var_name})[i] = ((float*){in_var_name})[i];
}}",
                    out_var_name = out_var_name,
                    in_var_name = in_var_name,
                    limit = out_shape.iter().product::<usize>(),
                )
                .as_str(),
            );

            out_var_name
        }
        &Language::AccessSqueeze([access_id, _axis_id]) => {
            // Squeeze doesn't do anything. Just copy the array verbatim.

            let out_shape = match &expr[id].data {
                MyAnalysisData::AccessPattern(a) => a.as_vec(),
                _ => panic!(),
            };
            let out_var_name: String = {
                // TODO(@gussmith23) Find a different way to name intermediates
                // Currently generating random strings. Not great IMO.
                let out = format!(
                    "access_squeeze_out_{}",
                    rand::thread_rng()
                        .sample_iter(&rand::distributions::Alphanumeric)
                        .take(30)
                        .collect::<String>()
                );
                declarations.push_str(
                    c_allocation_string(
                        uninitialized_allocations_prefix,
                        out.as_str(),
                        out_shape.as_slice(),
                        DType::Fp32,
                    )
                    .as_str(),
                );
                out
            };

            let in_var_name = codegen_recursive_helper(
                expr,
                access_id,
                top_level_id,
                uninitialized_allocations_prefix,
                declarations,
                code,
                hw_map,
            );

            code.push_str(
                format!(
                    "
for (int i = 0; i < {limit}; ++i) {{
  ((float*){out_var_name})[i] = ((float*){in_var_name})[i];
}}",
                    out_var_name = out_var_name,
                    in_var_name = in_var_name,
                    limit = out_shape.iter().product::<usize>(),
                )
                .as_str(),
            );

            out_var_name
        }
        &Language::AccessConcatenate([a0_id, a1_id, axis_id]) => {
            let axis = MyAnalysis::get_usize(axis_id, expr);
            let (a0, _a1) = match (&expr[a0_id].data, &expr[a1_id].data) {
                (MyAnalysisData::AccessPattern(a0), MyAnalysisData::AccessPattern(a1)) => (a0, a1),
                _ => panic!(),
            };
            let arg_0_name = codegen_recursive_helper(
                expr,
                a0_id,
                top_level_id,
                uninitialized_allocations_prefix,
                declarations,
                code,
                hw_map,
            );
            let arg_1_name = codegen_recursive_helper(
                expr,
                a1_id,
                top_level_id,
                uninitialized_allocations_prefix,
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

            out_var_name
        }
        &Language::Literal(_)
        | &Language::NotNanFloat64(_)
        | Language::List(_)
        | &Language::AccessBroadcast(_)
        | &Language::AccessInsertAxis(_)
        | &Language::AccessPair(_)
        | Language::PadType(_)
        | Language::ComputeType(_)
        | &Language::Compute(_)
        | &Language::AccessCartesianProduct(_)
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
        | &Language::AccessShape(_)
        | &Language::AccessLiteral(_)
        | &Language::AccessShiftRight(_) => panic!("{:#?} not implemented", expr[id].nodes[0]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egg::RecExpr;
    use ndarray::{SliceInfo, SliceOrIndex};
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

        let code = codegen(&egraph, id, &HashMap::default(), "pad", "", &vec!["t"]);

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

        let code = codegen(&egraph, id, &HashMap::default(), "slice", "", &vec!["t"]);

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
        // TODO: do broadcasting
        let relay = r#"
#[version = "0.0.5"]
def @main(%x: Tensor[(1, 3, 3, 4), float32], %y: Tensor[(1, 3, 3, 4), float32]) {
  add(%x, %y)
}
"#;

        let module = tvm::ir::module::IRModule::parse("", relay).unwrap();

        let (expr, shapes_vec) = crate::language::from_relay::from_relay(&module, true, &vec![crate::language::RelayOperator::RelayAdd]);

        let mut env = HashMap::default();
        for (k, v) in &shapes_vec {
            env.insert(k.clone(), v.clone());
        }

        // TODO(@gussmith23) Include some simple simplifying rewrites
        // If we add some very basic rewrites here, then $glenside_str
        // won't need to exactly match what's actually produced by
        // from_relay.py. It can be simpler (e.g. collapsing accesses).
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
        );

        println!("{}", code);

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
                    "{}/{}/{}/{}",
                    env!("CARGO_MANIFEST_DIR"),
                    "data",
                    "codegen-mlp",
                    "opaque_relay_op.c"
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
            // .arg("-Werror")
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
    fn relay_op_batchnorm() {
        let relay = r#"
#[version = "0.0.5"]
def @main(%data: Tensor[(1, 32, 32, 16), float32], %bn_gamma: Tensor[(16), float32], %bn_beta: Tensor[(16), float32], %bn_mean: Tensor[(16), float32], %bn_var: Tensor[(16), float32]) -> (Tensor[(1, 32, 32, 16), float32], Tensor[(16), float32], Tensor[(16), float32]) {
    nn.batch_norm(%data, %bn_gamma, %bn_beta, %bn_mean, %bn_var, axis=3) /* ty=(Tensor[(1, 32, 32, 16), float32], Tensor[(16), float32], Tensor[(16), float32]) */
}
"#;

        let module = tvm::ir::module::IRModule::parse("", relay).unwrap();

        let (expr, shapes_vec) = crate::language::from_relay::from_relay(&module, true, &vec![crate::language::RelayOperator::RelayBatchNormInference]);

        let mut env = HashMap::default();
        for (k, v) in &shapes_vec {
            env.insert(k.clone(), v.clone());
        }

        // TODO(@gussmith23) Include some simple simplifying rewrites
        // If we add some very basic rewrites here, then $glenside_str
        // won't need to exactly match what's actually produced by
        // from_relay.py. It can be simpler (e.g. collapsing accesses).
        let mut egraph = EGraph::new(MyAnalysis {
            name_to_shape: env.clone(),
        });

        let id = egraph.add_expr(&expr);

        let data_input = ndarray::ArrayD::from_shape_vec(
            env.get("data").unwrap().clone(),
            (0..env.get("data").unwrap().iter().product::<usize>()).collect(),
        )
        .unwrap();
        let bn_gamma_input = ndarray::ArrayD::from_shape_vec(
            env.get("bn_gamma").unwrap().clone(),
            (0..env.get("bn_gamma").unwrap().iter().product::<usize>()).collect(),
        )
        .unwrap();
        let bn_beta_input = ndarray::ArrayD::from_shape_vec(
            env.get("bn_beta").unwrap().clone(),
            (0..env.get("bn_beta").unwrap().iter().product::<usize>()).collect(),
        )
        .unwrap();
        let bn_mean_input = ndarray::ArrayD::from_shape_vec(
            env.get("bn_mean").unwrap().clone(),
            (0..env.get("bn_mean").unwrap().iter().product::<usize>()).collect(),
        )
        .unwrap();
        let bn_var_input = ndarray::ArrayD::from_shape_vec(
            env.get("bn_var").unwrap().clone(),
            (0..env.get("bn_var").unwrap().iter().product::<usize>()).collect(),
        )
        .unwrap();
        let result_output = ndarray::ArrayD::<f32>::zeros(env.get("data").unwrap().clone());

        println!("{}", expr);

        let code = codegen(
            &egraph,
            id,
            &HashMap::default(),
            "relay_batchnorm",
            "",
            &vec!["data", "bn_gamma", "bn_beta", "bn_mean", "bn_var"],
        );
        // TODO: check out array with result array
        let main_code = format!(
            "
#include <assert.h>
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
    // assert(((float*)result)[i] == ((float*)out)[i]);
  }}
}}
",
            PathBuf::from_str(
                format!(
                    "{}/{}/{}/{}",
                    env!("CARGO_MANIFEST_DIR"),
                    "data",
                    "codegen-mlp",
                    "opaque_relay_op.c"
                )
                .as_str()
            )
            .unwrap()
            .to_string_lossy(),
            c_assignment_string("", "data", DType::Fp32, &data_input.into_dyn().view()),
            c_assignment_string("", "bn_gamma", DType::Fp32, &bn_gamma_input.into_dyn().view()),
            c_assignment_string("", "bn_beta", DType::Fp32, &bn_beta_input.into_dyn().view()),
            c_assignment_string("", "bn_mean", DType::Fp32, &bn_mean_input.into_dyn().view()),
            c_assignment_string("", "bn_var", DType::Fp32, &bn_var_input.into_dyn().view()),
            c_assignment_string("", "result", DType::Fp32, &result_output.view()),
            c_assignment_string(
                "",
                "out",
                DType::Fp32,
                &result_output.view()
            ),
            code,
            result_output.shape().iter().product::<usize>()
        );

        println!("{}", main_code);

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

        // TODO: find a better way to convert from C multidimensional array to pointer
        // rather than removing -Werror
        let result = Command::new("gcc")
            // .arg("-Werror")
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
    fn relay_op_softmax() {
        let relay = r#"
#[version = "0.0.5"]
def @main(%data: Tensor[(1,10), float32]) -> Tensor[(1,10), float32] {
    nn.softmax(%data) /* ty=Tensor[(1,10), float32] */
}
"#;

        let module = tvm::ir::module::IRModule::parse("", relay).unwrap();

        let (expr, shapes_vec) = crate::language::from_relay::from_relay(&module, true, &vec![crate::language::RelayOperator::RelaySoftmax]);

        let mut env = HashMap::default();
        for (k, v) in &shapes_vec {
            env.insert(k.clone(), v.clone());
        }

        // TODO(@gussmith23) Include some simple simplifying rewrites
        // If we add some very basic rewrites here, then $glenside_str
        // won't need to exactly match what's actually produced by
        // from_relay.py. It can be simpler (e.g. collapsing accesses).
        let mut egraph = EGraph::new(MyAnalysis {
            name_to_shape: env.clone(),
        });

        let id = egraph.add_expr(&expr);

        let data_input = ndarray::ArrayD::from_shape_vec(
            env.get("data").unwrap().clone(),
            (0..env.get("data").unwrap().iter().product::<usize>()).collect(),
        )
        .unwrap();
        let result_output = ndarray::ArrayD::<f32>::zeros(env.get("data").unwrap().clone());

        println!("{}", expr);

        let code = codegen(
            &egraph,
            id,
            &HashMap::default(),
            "relay_softmax",
            "",
            &vec!["data"],
        );
        // TODO: check out array with result array
        let main_code = format!(
            "
#include <assert.h>
#include \"{}\"

{}
{}
{}
{}

int main() {{
  relay_softmax(out, data);

  for (int i = 0; i < {}; i++) {{
    // assert(((float*)result)[i] == ((float*)out)[i]);
  }}
}}
",
            PathBuf::from_str(
                format!(
                    "{}/{}/{}/{}",
                    env!("CARGO_MANIFEST_DIR"),
                    "data",
                    "codegen-mlp",
                    "opaque_relay_op.c"
                )
                .as_str()
            )
            .unwrap()
            .to_string_lossy(),
            c_assignment_string("", "data", DType::Fp32, &data_input.into_dyn().view()),
            c_assignment_string("", "result", DType::Fp32, &result_output.view()),
            c_assignment_string(
                "",
                "out",
                DType::Fp32,
                &result_output.view()
            ),
            code,
            result_output.shape().iter().product::<usize>()
        );

        println!("{}", main_code);

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

        // TODO: find a better way to convert from C multidimensional array to pointer
        // rather than removing -Werror
        let result = Command::new("gcc")
            // .arg("-Werror")
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
    fn relay_op_relu() {
        let relay = r#"
#[version = "0.0.5"]
def @main(%x: Tensor[(1, 3, 3, 4), float32]) {
  nn.relu(%x)
}
"#;

        let module = tvm::ir::module::IRModule::parse("", relay).unwrap();

        let (expr, shapes_vec) = crate::language::from_relay::from_relay(&module, true, &vec![crate::language::RelayOperator::RelayReLU]);

        let mut env = HashMap::default();
        for (k, v) in &shapes_vec {
            env.insert(k.clone(), v.clone());
        }

        // TODO(@gussmith23) Include some simple simplifying rewrites
        // If we add some very basic rewrites here, then $glenside_str
        // won't need to exactly match what's actually produced by
        // from_relay.py. It can be simpler (e.g. collapsing accesses).
        let mut egraph = EGraph::new(MyAnalysis {
            name_to_shape: env.clone(),
        });

        let id = egraph.add_expr(&expr);

        let x_input = ndarray::ArrayD::from_shape_vec(
            env.get("x").unwrap().clone(),
            (0..env.get("x").unwrap().iter().product::<usize>()).collect(),
        )
        .unwrap();
        let result = x_input.clone().into_dyn();

        let code = codegen(
            &egraph,
            id,
            &HashMap::default(),
            "relay_relu",
            "",
            &vec!["x"],
        );

        println!("{}", code);

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
                    "opaque_relay_op.c"
                )
                .as_str()
            )
            .unwrap()
            .to_string_lossy(),
            c_assignment_string("", "x", DType::Fp32, &x_input.into_dyn().view()),
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
            // .arg("-Werror")
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
    fn relay_op_maxpool2d_resnet_3x3() {
        let relay = r#"
#[version = "0.0.5"]
def @main(%x: Tensor[(1, 112, 112, 64), float32]) -> Tensor[(1, 56, 56, 64), float32] {
  nn.max_pool2d(%x, pool_size=[3, 3], strides=[2, 2], padding=[1, 1, 1, 1], layout="NHWC") /* ty=Tensor[(1, 56, 56, 64), float32] */
}
"#;

        let module = tvm::ir::module::IRModule::parse("", relay).unwrap();

        let (expr, shapes_vec) = crate::language::from_relay::from_relay(&module, true, &vec![crate::language::RelayOperator::RelayMaxPool2D]);

        let mut env = HashMap::default();
        for (k, v) in &shapes_vec {
            env.insert(k.clone(), v.clone());
        }

        // TODO(@gussmith23) Include some simple simplifying rewrites
        // If we add some very basic rewrites here, then $glenside_str
        // won't need to exactly match what's actually produced by
        // from_relay.py. It can be simpler (e.g. collapsing accesses).
        let mut egraph = EGraph::new(MyAnalysis {
            name_to_shape: env.clone(),
        });

        let id = egraph.add_expr(&expr);

        let x_input = ndarray::ArrayD::from_shape_vec(
            env.get("x").unwrap().clone(),
            (0..env.get("x").unwrap().iter().product::<usize>()).collect(),
        )
        .unwrap();
        let result = ndarray::ArrayD::<f32>::zeros(vec![1, 56, 56, 64]);

        let code = codegen(
            &egraph,
            id,
            &HashMap::default(),
            "relay_maxpool",
            "",
            &vec!["x"],
        );

        println!("{}", code);

        let main_code = format!(
            "
#include <assert.h>
#include \"{}\"

{}
{}
{}
{}

int main() {{
    relay_maxpool(out, x);

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
                    "opaque_relay_op.c"
                )
                .as_str()
            )
            .unwrap()
            .to_string_lossy(),
            c_assignment_string("", "x", DType::Fp32, &x_input.into_dyn().view()),
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
            // .arg("-Werror")
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
