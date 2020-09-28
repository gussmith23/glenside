// TODO(@gussmith23) Make sure TVM feature flag is getting tested in CI
#[cfg(feature = "use-tvm")]
use crate::language::Language;
use egg::{Id, RecExpr};
use ordered_float::NotNan;
use std::collections::HashMap;
use std::convert::TryInto;
use tvm::ir::module::*;
use tvm::ir::relay::*;
use tvm::ir::tir::*;
use tvm::ir::ty::*;
use tvm::runtime::IsObjectRef;
use tvm::DataType;

use super::ComputeType;
use super::PadType;

// TODO(@gussmith23) Give glenside-expression-creation helpers a new home

/// Create a shape
///
/// ```
/// use glenside::language::from_relay::shape;
/// use egg::RecExpr;
///
/// let mut expr = RecExpr::default();
/// let id = shape(&mut expr, vec![1, 2, 3, 4]);
/// assert_eq!(expr.pretty(80), "(shape 1 2 3 4)");
/// ```
pub fn shape(expr: &mut RecExpr<Language>, vals: Vec<usize>) -> Id {
    let mut ids = Vec::default();
    for val in vals {
        ids.push(expr.add(Language::Usize(val)));
    }
    expr.add(Language::Shape(Box::from(ids.as_slice())))
}

/// Pad an access
///
/// ```
/// use std::str::FromStr;
/// use glenside::language::from_relay::access_pad;
/// use glenside::language::PadType;
/// use egg::RecExpr;
///
/// let mut expr = RecExpr::from_str("(access-tensor a)").unwrap();
/// let id = access_pad(&mut expr, 1, PadType::ZeroPadding, 2, 3, 4 );
/// assert_eq!(expr.pretty(80), "(access-pad (access-tensor a) zero-padding 2 3 4)");
/// ```
pub fn access_pad(
    expr: &mut RecExpr<Language>,
    id: Id,
    pad_type: PadType,
    axis: usize,
    pad_before: usize,
    pad_after: usize,
) -> Id {
    let axis_id = expr.add(Language::Usize(axis));
    let pad_before_id = expr.add(Language::Usize(pad_before));
    let pad_after_id = expr.add(Language::Usize(pad_after));
    let pad_type_id = expr.add(Language::PadType(pad_type));
    expr.add(Language::AccessPad([
        id,
        pad_type_id,
        axis_id,
        pad_before_id,
        pad_after_id,
    ]))
}

/// Given an access and axis, add access expression accessing access at axis
///
/// ```
/// use std::str::FromStr;
/// use glenside::language::from_relay::access;
/// use egg::RecExpr;
///
/// let mut expr = RecExpr::from_str("(access-tensor a)").unwrap();
/// let id = access(&mut expr, 1, 2);
/// assert_eq!(expr.pretty(80), "(access (access-tensor a) 2)");
/// ```
pub fn access(expr: &mut RecExpr<Language>, id: Id, axis: usize) -> Id {
    let axis_id = expr.add(Language::Usize(axis));
    expr.add(Language::Access([id, axis_id]))
}

/// Given an access and axis, add access expression accessing access at axis
///
/// ```
/// use std::str::FromStr;
/// use glenside::language::from_relay::access_insert_axis;
/// use egg::RecExpr;
///
/// let mut expr = RecExpr::from_str("(access-tensor a)").unwrap();
/// let id = access_insert_axis(&mut expr, 1, 2);
/// assert_eq!(expr.pretty(80), "(access-insert-axis (access-tensor a) 2)");
/// ```
pub fn access_insert_axis(expr: &mut RecExpr<Language>, id: Id, axis: usize) -> Id {
    let axis_id = expr.add(Language::Usize(axis));
    expr.add(Language::AccessInsertAxis([id, axis_id]))
}

/// Given the input access and compute type, add compute expression
///
/// ```
/// use egg::RecExpr;
/// use glenside::language::from_relay::compute;
/// use glenside::language::ComputeType;
/// use std::str::FromStr;
///
/// let mut expr = RecExpr::from_str("(access (access-tensor a) 2)").unwrap();
/// let id = compute(&mut expr, ComputeType::ReLU, 3);
/// assert_eq!(
///     expr.pretty(80),
///     "(compute relu (access (access-tensor a) 2))"
/// );
/// ```
pub fn compute(expr: &mut RecExpr<Language>, compute_type: ComputeType, access_id: Id) -> Id {
    let compute_type_id = expr.add(Language::ComputeType(compute_type));
    expr.add(Language::Compute([compute_type_id, access_id]))
}

/// Given input accesses and an axis, re-accesses and pairs both accesses
///
/// ```
/// use egg::RecExpr;
/// use glenside::language::from_relay::access_pair;
/// use glenside::language::from_relay::access;
/// use glenside::language::Language;
///
/// let mut expr = RecExpr::default();
/// let a_id = expr.add(Language::Symbol("a".to_string()));
/// let a_id = expr.add(Language::AccessTensor(a_id));
/// let b_id = expr.add(Language::Symbol("b".to_string()));
/// let b_id = expr.add(Language::AccessTensor(b_id));
/// let id = access_pair(&mut expr, a_id, b_id, 2);
/// assert_eq!(
///     expr.pretty(80),
///     "(access-pair (access (access-tensor a) 2) (access (access-tensor b) 2))"
/// );
/// ```
pub fn access_pair(
    expr: &mut RecExpr<Language>,
    access_a_id: Id,
    access_b_id: Id,
    axis: usize,
) -> Id {
    let a_id = access(expr, access_a_id, axis);
    let b_id = access(expr, access_b_id, axis);
    expr.add(Language::AccessPair([a_id, b_id]))
}

/// Get shape from type
pub fn shape_from_type(t: tvm::ir::ty::Type) -> Vec<usize> {
    let tensor_type = t
        .clone()
        .downcast::<tvm::ir::ty::TensorType>()
        .unwrap_or_else(|_| {
            panic!(
                "Expected type {:?} to have tensor type",
                *t.upcast::<tvm::runtime::ObjectRef>()
            )
        });
    assert_eq!(
        tensor_type.dtype.clone(),
        "float32".parse().unwrap(),
        "only supporting float32x1 at the moment"
    );
    let mut shape = Vec::<usize>::default();
    for j in 0..tensor_type.shape.len() {
        shape.push(
            tensor_type
                .shape
                .get(j as isize)
                .unwrap()
                .downcast::<tvm::ir::tir::IntImm>()
                .unwrap()
                .value
                .try_into()
                .unwrap(),
        );
    }
    shape
}

/// Convert Relay IRModule to Glenside RecExpr.
///
/// Returns the RecExpr, along with a Vec mapping symbols to their shapes.
/// Note that the shapes are a Vec rather than a HashMap to preserve ordering.
pub fn from_relay(module: &IRModule) -> (RecExpr<Language>, Vec<(String, Vec<usize>)>) {
    let main = module
        .lookup(module.get_global_var("main".to_string().into()).unwrap())
        .unwrap();
    let func = main.downcast::<tvm::ir::relay::Function>().unwrap();
    let mut names_and_shapes = Vec::default();
    for i in 0..func.params.len() {
        let var = func.params.get(i as isize).unwrap();
        let t = shape_from_type(var.type_annotation.clone());
        names_and_shapes.push((var.name_hint().as_str().unwrap().to_string(), t));
    }
    let mut glenside_expr = RecExpr::default();
    let mut worklist = Vec::default();
    create_worklist(func.body.clone(), &mut worklist);
    let mut map = HashMap::new();
    for (i, expr) in worklist.iter().enumerate() {
        map.insert(
            expr.clone(),
            compile_expression(expr.clone(), &mut glenside_expr, |expr| {
                *map.get(&expr).unwrap_or_else(|| {
                    panic!("Not found:\n{}", tvm::ir::expr::as_text(expr.clone()))
                })
            }),
        );
    }

    (glenside_expr, names_and_shapes)
}

/// Generates an ordered list of Relay expressions to compile.
///
/// Compiling large Relay expressions with naive recursion overflows the stack,
/// so we first recursively generate a worklist which we can then iterate over.
/// The main goal of the worklist is to make sure an expression comes *after*
/// its children in the worklist; otherwise, we can't compile the expression!
fn create_worklist(relay_expr: Expr, worklist: &mut Vec<Expr>) {
    fn add_to_worklist(expr: Expr, worklist: &mut Vec<Expr>) {
        if !worklist.contains(&expr) {
            worklist.push(expr.clone());
        }
    }
    if let Ok(_var) = relay_expr.clone().downcast::<tvm::ir::relay::Var>() {
        add_to_worklist(relay_expr.clone(), worklist);
    } else if let Ok(_constant) = relay_expr.clone().downcast::<tvm::ir::relay::Constant>() {
        add_to_worklist(relay_expr.clone(), worklist);
    } else if let Ok(call) = relay_expr.clone().downcast::<tvm::ir::relay::Call>() {
        for i in 0..call.args.len() {
            // Recursively add children (and their dependencies) to the worklist
            create_worklist(call.args.get(i.try_into().unwrap()).unwrap(), worklist);
        }
        add_to_worklist(relay_expr.clone(), worklist);
    } else {
        todo!()
    }
}

/// Compile a Relay expression to a Glenside [`RecExpr`]
///
/// `get_compiled_expression` is a function which `compile_expression` can use
/// to get the [`Id`]s of the expression's children, once they are compiled and
/// added to the [`RecExpr`]. This can be `compile_expression` itself, for a
/// naive recursive strategy, or some other function that e.g. accesses a
/// memoization map. `get_compiled_expression`'s signature may need to be
/// modified to actually support the naive recursive case.
///
fn compile_expression(
    relay_expr: Expr,
    glenside_expr: &mut RecExpr<Language>,
    // TODO(@gussmith23) Do we need to pass the recexpr into this closure?
    get_compiled_expression: impl Fn(Expr) -> Id,
) -> Id {
    if let Ok(var) = relay_expr.clone().downcast::<tvm::ir::relay::Var>() {
        let symbol_id = glenside_expr.add(Language::Symbol(var.name_hint().to_string()));
        glenside_expr.add(Language::AccessTensor(symbol_id))
    } else if let Ok(constant) = relay_expr.clone().downcast::<tvm::ir::relay::Constant>() {
        let tuple_type = constant
            .clone()
            .upcast::<Expr>()
            .checked_type
            .clone()
            .downcast::<TensorType>()
            .unwrap();
        assert_eq!(
            tuple_type.shape.len(),
            0,
            "Only scalar constants supported for now"
        );
        assert_eq!(
            tuple_type.dtype,
            "float32".parse().unwrap(),
            "Only float32x1 constants supported for now",
        );
        assert_eq!(
            constant.data.size(),
            4,
            "Only scalar constants supported for now"
        );
        // TODO(@gussmith23) This is broken at the moment
        // assert_eq!(
        //     constant.data.shape().unwrap().len(),
        //     0,
        //     "Only scalar constants supported for now"
        // );
        assert_eq!(
            constant.data.dtype(),
            "float32".parse().unwrap(),
            "Only float32x1 constants supported for now",
        );
        // TODO(@gussmith23) This is a hack
        // Jared and Max are working on ndarray at the moment.
        let value: f32 = unsafe { *(constant.data.as_dltensor().data as *const f32) };
        let literal_id = glenside_expr.add(Language::NotNanFloat64(
            NotNan::<f64>::new(value as f64).unwrap(),
        ));
        let literal_id = glenside_expr.add(Language::Literal(literal_id));
        let access_literal_id = glenside_expr.add(Language::AccessLiteral(literal_id));
        access_literal_id
    } else if let Ok(call) = relay_expr.clone().downcast::<tvm::ir::relay::Call>() {
        if let Ok(primitive_op) = call
            .op
            .clone()
            .upcast::<tvm::ir::expr::BaseExpr>()
            .downcast::<tvm::ir::op::Op>()
        {
            match primitive_op.name.as_str().unwrap() {
                "nn.softmax" => {
                    assert_eq!(call.args.len(), 1);
                    let data_id = get_compiled_expression(call.args.get(0).unwrap());
                    let attrs = call
                        .attrs
                        .clone()
                        .downcast::<tvm::ir::relay::attrs::nn::SoftmaxAttrs>()
                        .unwrap();
                    match attrs.axis {
                        -1 => {
                            let data_id = access(
                                glenside_expr,
                                data_id,
                                (call
                                    .args
                                    .get(0)
                                    .unwrap()
                                    .checked_type
                                    .clone()
                                    .downcast::<TensorType>()
                                    .unwrap()
                                    .shape
                                    .len()
                                    - 1)
                                .try_into()
                                .unwrap(),
                            );
                            compute(glenside_expr, ComputeType::Softmax, data_id)
                        }
                        other @ _ => todo!("Softmax with axis value {} not yet supported", other),
                    }
                }
                "nn.relu" | "sqrt" | "negative" => {
                    assert_eq!(call.args.len(), 1);
                    let data_id = get_compiled_expression(call.args.get(0).unwrap());
                    compute(
                        glenside_expr,
                        match primitive_op.name.as_str().unwrap() {
                            "nn.relu" => ComputeType::ReLU,
                            "sqrt" => ComputeType::Sqrt,
                            "negative" => ComputeType::Negative,
                            _ => unreachable!(),
                        },
                        data_id,
                    )
                }
                "nn.max_pool2d" => {
                    assert_eq!(call.args.len(), 1);
                    let attrs = call
                        .attrs
                        .clone()
                        .downcast::<tvm::ir::relay::attrs::nn::MaxPool2DAttrs>()
                        .unwrap();
                    assert_eq!(
                        call.args
                            .get(0)
                            .unwrap()
                            .checked_type
                            .clone()
                            .downcast::<TensorType>()
                            .unwrap()
                            .shape
                            .len(),
                        4
                    );
                    assert_eq!(attrs.pool_size.len(), 2);
                    assert_eq!(attrs.padding.len(), 4);
                    assert_eq!(attrs.strides.len(), 2);
                    assert_eq!(attrs.ceil_mode, false);

                    match attrs.layout.as_str().unwrap() {
                        "NCHW" => {
                            let data_id = get_compiled_expression(call.args.get(0).unwrap());
                            let data_id = access_pad(
                                glenside_expr,
                                data_id,
                                PadType::MinPadding,
                                2,
                                attrs
                                    .padding
                                    .get(0)
                                    .unwrap()
                                    .downcast::<IntImm>()
                                    .unwrap()
                                    .value as usize,
                                attrs
                                    .padding
                                    .get(2)
                                    .unwrap()
                                    .downcast::<IntImm>()
                                    .unwrap()
                                    .value as usize,
                            );
                            let data_id = access_pad(
                                glenside_expr,
                                data_id,
                                PadType::MinPadding,
                                3,
                                attrs
                                    .padding
                                    .get(1)
                                    .unwrap()
                                    .downcast::<IntImm>()
                                    .unwrap()
                                    .value as usize,
                                attrs
                                    .padding
                                    .get(3)
                                    .unwrap()
                                    .downcast::<IntImm>()
                                    .unwrap()
                                    .value as usize,
                            );
                            let data_id = access(glenside_expr, data_id, 4);

                            let stride_shape_id = shape(
                                glenside_expr,
                                vec![
                                    1,
                                    1,
                                    attrs
                                        .strides
                                        .get(0)
                                        .unwrap()
                                        .downcast::<IntImm>()
                                        .unwrap()
                                        .value as usize,
                                    attrs
                                        .strides
                                        .get(1)
                                        .unwrap()
                                        .downcast::<IntImm>()
                                        .unwrap()
                                        .value as usize,
                                ],
                            );
                            let pool_window_shape_id = shape(
                                glenside_expr,
                                vec![
                                    1,
                                    1,
                                    attrs
                                        .pool_size
                                        .get(0)
                                        .unwrap()
                                        .downcast::<IntImm>()
                                        .unwrap()
                                        .value as usize,
                                    attrs
                                        .pool_size
                                        .get(1)
                                        .unwrap()
                                        .downcast::<IntImm>()
                                        .unwrap()
                                        .value as usize,
                                ],
                            );

                            let data_id = glenside_expr.add(Language::AccessWindows([
                                data_id,
                                pool_window_shape_id,
                                stride_shape_id,
                            ]));

                            let data_id = access(glenside_expr, data_id, 4);

                            let data_id = compute(glenside_expr, ComputeType::ReduceMax, data_id);

                            data_id
                        }
                        other @ _ => todo!("layout {} not supported", other),
                    }
                }
                "nn.global_avg_pool2d" => {
                    let attrs = call
                        .attrs
                        .clone()
                        .downcast::<tvm::ir::relay::attrs::nn::GlobalPool2DAttrs>()
                        .unwrap();
                    assert_eq!(call.args.len(), 1);
                    assert_eq!(
                        attrs.layout, "NCHW",
                        "NCHW is the only layout currently supported"
                    );

                    match attrs.layout.as_str().unwrap() {
                        "NCHW" => {
                            let data_id = get_compiled_expression(call.args.get(0).unwrap());
                            let data_id = access(glenside_expr, data_id, 2);
                            let data_id = compute(glenside_expr, ComputeType::ReduceMean, data_id);
                            let data_id = access_insert_axis(glenside_expr, data_id, 2);
                            let data_id = access_insert_axis(glenside_expr, data_id, 3);
                            let data_id = access(glenside_expr, data_id, 2);
                            data_id
                        }
                        _ => todo!("layout not currently supported"),
                    }
                }
                "expand_dims" => {
                    let attrs = call
                        .attrs
                        .clone()
                        .downcast::<tvm::ir::relay::attrs::transform::ExpandDimsAttrs>()
                        .unwrap();
                    assert_eq!(call.args.len(), 1);

                    let mut data_id = get_compiled_expression(call.args.get(0).unwrap());

                    for _ in 0..attrs.num_newaxis {
                        data_id = access_insert_axis(
                            glenside_expr,
                            data_id,
                            attrs.axis.try_into().unwrap(),
                        )
                    }

                    data_id
                }
                "nn.dense" => {
                    let attrs = call
                        .attrs
                        .clone()
                        .downcast::<tvm::ir::relay::attrs::nn::DenseAttrs>()
                        .unwrap();
                    assert_eq!(call.args.len(), 2);
                    assert_eq!(
                        attrs.out_dtype,
                        // This datatype seems to indicate "null"?
                        DataType::new(3, 0, 0),
                        "Changing out_dtype not yet supported"
                    );
                    assert_eq!(
                        call.args
                            .get(0)
                            .unwrap()
                            .checked_type
                            .clone()
                            .downcast::<TensorType>()
                            .unwrap()
                            .shape
                            .len(),
                        2,
                        "Only supporting dense matrix multiplication of tensors with 2 dimensions"
                    );
                    assert_eq!(
                        call.args
                            .get(1)
                            .unwrap()
                            .checked_type
                            .clone()
                            .downcast::<TensorType>()
                            .unwrap()
                            .shape
                            .len(),
                        2,
                        "Only supporting dense matrix multiplication of tensors with 2 dimensions"
                    );

                    let data_id = get_compiled_expression(call.args.get(0).unwrap());
                    let weights_id = get_compiled_expression(call.args.get(1).unwrap());

                    let data_id = access(glenside_expr, data_id, 1);
                    let weights_id = access(glenside_expr, weights_id, 1);

                    let data_id =
                        glenside_expr.add(Language::AccessCartesianProduct([data_id, weights_id]));
                    compute(glenside_expr, ComputeType::DotProduct, data_id)
                }
                "add" | "multiply" | "divide" => {
                    assert_eq!(call.args.len(), 2);
                    let mut a_id = get_compiled_expression(call.args.get(0).unwrap());
                    let mut a_shape =
                        shape_from_type(call.args.get(0).unwrap().checked_type.clone());
                    let mut b_id = get_compiled_expression(call.args.get(1).unwrap());
                    let mut b_shape =
                        shape_from_type(call.args.get(1).unwrap().checked_type.clone());

                    while a_shape.len() < b_shape.len() {
                        a_id = access_insert_axis(glenside_expr, a_id, 0);
                        a_shape.insert(0, 1);
                    }

                    while b_shape.len() < a_shape.len() {
                        b_id = access_insert_axis(glenside_expr, b_id, 0);
                        b_shape.insert(0, 1);
                    }

                    assert!(a_shape.iter().zip(b_shape.iter()).map(|(a, b)| a <= b).all(|v| v) ||
                            a_shape.iter().zip(b_shape.iter()).map(|(a, b)| a >= b).all(|v| v),
                            "Can only handle simple broadcasts; all dims of a must be <= all dims of b (or vice-versa)");
                    if a_shape
                        .iter()
                        .zip(b_shape.iter())
                        .map(|(a, b)| a < b)
                        .any(|v| v)
                    {
                        let get_access_shape_id = glenside_expr.add(Language::GetAccessShape(b_id));
                        a_id = glenside_expr
                            .add(Language::AccessBroadcast([a_id, get_access_shape_id]));
                    } else if a_shape
                        .iter()
                        .zip(b_shape.iter())
                        .map(|(a, b)| a > b)
                        .any(|v| v)
                    {
                        let get_access_shape_id = glenside_expr.add(Language::GetAccessShape(a_id));
                        b_id = glenside_expr
                            .add(Language::AccessBroadcast([b_id, get_access_shape_id]));
                    }

                    let pair_id = access_pair(glenside_expr, a_id, b_id, 0);

                    match primitive_op.name.as_str().unwrap() {
                        "add" => compute(glenside_expr, ComputeType::ElementwiseAdd, pair_id),
                        "multiply" => compute(glenside_expr, ComputeType::ElementwiseMul, pair_id),
                        "divide" => compute(glenside_expr, ComputeType::ElementwiseDiv, pair_id),
                        _ => unreachable!(),
                    }
                }
                "nn.batch_flatten" => {
                    assert_eq!(call.args.len(), 1);
                    assert!(
                        call.args
                            .get(0)
                            .unwrap()
                            .checked_type
                            .clone()
                            .downcast::<TensorType>()
                            .unwrap()
                            .shape
                            .len()
                            >= 1
                    );

                    let data_id = get_compiled_expression(call.args.get(0).unwrap());
                    let data_id = access(glenside_expr, data_id, 1);
                    glenside_expr.add(Language::AccessFlatten(data_id))
                }
                "nn.bias_add" => {
                    assert_eq!(call.args.len(), 2);
                    assert_eq!(
                        call.args
                            .get(1)
                            .unwrap()
                            .checked_type
                            .clone()
                            .downcast::<TensorType>()
                            .unwrap()
                            .shape
                            .len(),
                        1,
                        "Only supporting vector biases at the moment"
                    );

                    let data_id = get_compiled_expression(call.args.get(0).unwrap());
                    let mut bias_id = get_compiled_expression(call.args.get(1).unwrap());

                    let attrs = call
                        .attrs
                        .clone()
                        .downcast::<tvm::ir::relay::attrs::nn::BiasAddAttrs>()
                        .unwrap();

                    // Insert axes before
                    for _ in 0..attrs.axis {
                        let zero_id = glenside_expr.add(Language::Usize(0));
                        bias_id = glenside_expr.add(Language::AccessInsertAxis([bias_id, zero_id]));
                    }

                    // Insert axes after
                    for axis in (attrs.axis + 1) as i64
                        ..call
                            .args
                            .get(0)
                            .unwrap()
                            .checked_type
                            .clone()
                            .downcast::<TensorType>()
                            .unwrap()
                            .shape
                            .len()
                    {
                        let axis_id = glenside_expr.add(Language::Usize(axis as usize));
                        bias_id = glenside_expr.add(Language::AccessInsertAxis([bias_id, axis_id]));
                    }

                    let get_shape_id = glenside_expr.add(Language::GetAccessShape(data_id));
                    let bias_id =
                        glenside_expr.add(Language::AccessBroadcast([bias_id, get_shape_id]));

                    let data_id = access_pair(glenside_expr, data_id, bias_id, 0);
                    let data_id = compute(glenside_expr, ComputeType::ElementwiseAdd, data_id);

                    data_id
                }
                "nn.conv2d" => {
                    assert_eq!(call.args.len(), 2);
                    let attrs = call
                        .attrs
                        .clone()
                        .downcast::<tvm::ir::relay::attrs::nn::Conv2DAttrs>()
                        .unwrap();

                    let data_id = get_compiled_expression(call.args.get(0).unwrap());
                    let data_shape =
                        shape_from_type(call.args.get(0).unwrap().checked_type.clone());
                    assert_eq!(data_shape.len(), 4);
                    let weights_id = get_compiled_expression(call.args.get(1).unwrap());
                    let weights_shape =
                        shape_from_type(call.args.get(1).unwrap().checked_type.clone());
                    assert_eq!(weights_shape.len(), 4);

                    assert_eq!(
                        attrs.data_layout, "NCHW",
                        "NCHW is the only layout supported at the moment"
                    );
                    assert_eq!(
                        attrs.kernel_layout, "OIHW",
                        "OIHW is the only layout supported at the moment"
                    );

                    assert_eq!(attrs.padding.len(), 4);
                    assert_eq!(attrs.dilation.len(), 2);
                    assert_eq!(
                        attrs
                            .dilation
                            .get(0)
                            .unwrap()
                            .downcast::<tvm::ir::tir::IntImm>()
                            .unwrap()
                            .value,
                        1
                    );
                    assert_eq!(
                        attrs
                            .dilation
                            .get(1)
                            .unwrap()
                            .downcast::<tvm::ir::tir::IntImm>()
                            .unwrap()
                            .value,
                        1
                    );
                    assert_eq!(attrs.groups, 1);
                    assert_eq!(attrs.out_layout, "");
                    assert_eq!(
                        attrs.out_dtype,
                        // TODO(@gussmith23) How to actually constrain this?
                        tvm::DataType::new(3, 0, 0)
                    );

                    let usize_1_id = glenside_expr.add(Language::Usize(1));

                    let mut list = Vec::default();
                    list.push(usize_1_id);
                    for v in weights_shape[1..].iter() {
                        list.push(glenside_expr.add(Language::Usize(*v as usize)));
                    }
                    let weights_shape_id =
                        glenside_expr.add(Language::Shape(Box::from(list.as_slice())));

                    let pad_axis_id = glenside_expr.add(Language::Usize(2));
                    let pad_before_id = glenside_expr.add(Language::Usize(
                        attrs
                            .padding
                            .get(0)
                            .unwrap()
                            .downcast::<IntImm>()
                            .unwrap()
                            .value as usize,
                    ));
                    let pad_after_id = glenside_expr.add(Language::Usize(
                        attrs
                            .padding
                            .get(2)
                            .unwrap()
                            .downcast::<IntImm>()
                            .unwrap()
                            .value as usize,
                    ));
                    let zero_padding_id =
                        glenside_expr.add(Language::PadType(PadType::ZeroPadding));
                    let data_id = glenside_expr.add(Language::AccessPad([
                        data_id,
                        zero_padding_id,
                        pad_axis_id,
                        pad_before_id,
                        pad_after_id,
                    ]));

                    let pad_axis_id = glenside_expr.add(Language::Usize(3));
                    let pad_before_id = glenside_expr.add(Language::Usize(
                        attrs
                            .padding
                            .get(1)
                            .unwrap()
                            .downcast::<IntImm>()
                            .unwrap()
                            .value as usize,
                    ));
                    let pad_after_id = glenside_expr.add(Language::Usize(
                        attrs
                            .padding
                            .get(3)
                            .unwrap()
                            .downcast::<IntImm>()
                            .unwrap()
                            .value as usize,
                    ));
                    let zero_padding_id =
                        glenside_expr.add(Language::PadType(PadType::ZeroPadding));
                    let data_id = glenside_expr.add(Language::AccessPad([
                        data_id,
                        zero_padding_id,
                        pad_axis_id,
                        pad_before_id,
                        pad_after_id,
                    ]));

                    let access_axis_id = glenside_expr.add(Language::Usize(4));
                    let data_id = glenside_expr.add(Language::Access([data_id, access_axis_id]));

                    let mut stride_list = Vec::default();
                    stride_list.push(glenside_expr.add(Language::Usize(1)));
                    stride_list.push(glenside_expr.add(Language::Usize(1)));
                    stride_list.push(
                        glenside_expr.add(Language::Usize(
                            attrs
                                .strides
                                .get(0)
                                .unwrap()
                                .downcast::<IntImm>()
                                .unwrap()
                                .value as usize,
                        )),
                    );
                    stride_list.push(
                        glenside_expr.add(Language::Usize(
                            attrs
                                .strides
                                .get(1)
                                .unwrap()
                                .downcast::<IntImm>()
                                .unwrap()
                                .value as usize,
                        )),
                    );
                    let stride_shape_id =
                        glenside_expr.add(Language::Shape(Box::from(stride_list.as_slice())));

                    let data_id = glenside_expr.add(Language::AccessWindows([
                        data_id,
                        weights_shape_id,
                        stride_shape_id,
                    ]));

                    let squeeze_axis_id = glenside_expr.add(Language::Usize(4));
                    let data_id =
                        glenside_expr.add(Language::AccessSqueeze([data_id, squeeze_axis_id]));
                    let squeeze_axis_id = glenside_expr.add(Language::Usize(1));
                    let data_id =
                        glenside_expr.add(Language::AccessSqueeze([data_id, squeeze_axis_id]));

                    let access_axis_id = glenside_expr.add(Language::Usize(3));
                    let data_id = glenside_expr.add(Language::Access([data_id, access_axis_id]));

                    let access_axis_id = glenside_expr.add(Language::Usize(1));
                    let weights_id =
                        glenside_expr.add(Language::Access([weights_id, access_axis_id]));

                    let data_id =
                        glenside_expr.add(Language::AccessCartesianProduct([weights_id, data_id]));

                    let compute_type_id =
                        glenside_expr.add(Language::ComputeType(ComputeType::DotProduct));
                    let data_id = glenside_expr.add(Language::Compute([compute_type_id, data_id]));

                    let mut transpose_list = Vec::default();
                    transpose_list.push(glenside_expr.add(Language::Usize(1)));
                    transpose_list.push(glenside_expr.add(Language::Usize(0)));
                    transpose_list.push(glenside_expr.add(Language::Usize(2)));
                    transpose_list.push(glenside_expr.add(Language::Usize(3)));
                    let transpose_list_id =
                        glenside_expr.add(Language::List(Box::from(transpose_list)));

                    let data_id =
                        glenside_expr.add(Language::AccessTranspose([data_id, transpose_list_id]));

                    data_id
                }
                _ => todo!(),
            }
        } else {
            todo!()
        }
    } else {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use crate::language::interpreter::interpret;
    use crate::language::{Language, MyAnalysis};
    use approx::AbsDiffEq;
    use egg::{EGraph, Pattern, Searcher};
    use ndarray_npy::{read_npy, write_npy};
    use ndarray_rand::{rand_distr::Uniform, RandomExt};
    use rand::{rngs::SmallRng, Rng, SeedableRng};
    use std::collections::HashMap;
    use std::io::Write;
    use std::process::Command;

    /// Creates a Relay-to-Glenside test
    /// The test does the following:
    ///  1. Converts $relay_str to glenside by running the from_relay.py script
    ///  2. Inserts the resulting Glenside code into an egraph
    ///  3. Searches the egraph for $glenside_str to ensure the expected program
    ///     exists
    /// $test_name: the name of the created test
    /// $relay_str: A string containing the Relay program to be converted
    /// $glenside_str: A string containing the expected resulting Glenside
    ///     expression
    macro_rules! test {
        ($test_name:ident, $tol:literal, $relay_str:expr, $glenside_str:expr) => {
            // TODO(@gussmith23) Hardcoding to f32
            test!(
                $test_name,
                $tol,
                $relay_str,
                $glenside_str,
                "",
                Uniform::new(-1f32, 1f32)
            );
        };
        ($test_name:ident, $tol:literal, $relay_str:expr, $glenside_str:expr, $optional_arg:literal) => {
            // TODO(@gussmith23) Hardcoding to f32
            test!(
                $test_name,
                $tol,
                $relay_str,
                $glenside_str,
                $optional_arg,
                Uniform::new(-1f32, 1f32)
            );
        };
        ($test_name:ident, $tol:literal, $relay_str:expr, $glenside_str:expr, $optional_arg:literal, $distribution:expr) => {
            #[test]
            fn $test_name() {
                // The number of times to run each program and compare their
                // outputs.
                // TODO(@gussmith23) # random samples chosen arbitrarily
                const SAMPLES: usize = 3;

                // Random number generator for generating random tensors.
                const SEED: u64 = 23;
                let mut tensor_rng = SmallRng::seed_from_u64(SEED);

                let module = tvm::ir::module::IRModule::parse("", $relay_str);

                let (expr, shapes_vec) = super::from_relay(&module);

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

                let pattern = $glenside_str.parse::<Pattern<Language>>().unwrap();
                assert!(pattern.search_eclass(&egraph, id).is_some());

                for _ in (0..SAMPLES) {
                    // Run interpreters and compare output.
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
                    if $optional_arg.len() > 0 {
                        cmd.arg($optional_arg);
                    }
                    cmd.arg(&output_filepath);
                    cmd.stdin(std::process::Stdio::piped())
                        .stdout(std::process::Stdio::piped())
                        .stderr(std::process::Stdio::piped());
                    let mut env = HashMap::default();
                    for (name, shape) in shapes_vec.iter() {
                        // TODO(@gussmith23) output type assumption
                        let value = ndarray::ArrayD::<f32>::random_using(
                            shape.clone(),
                            $distribution,
                            &mut tensor_rng,
                        );
                        env.insert(name.as_str(), value.clone());
                        let filepath = std::env::temp_dir().with_file_name(format!(
                            "arg-{}.npy",
                            rand::thread_rng()
                                .sample_iter(&rand::distributions::Alphanumeric)
                                .take(30)
                                .collect::<String>()
                        ));
                        write_npy(&filepath, &value).unwrap();
                        cmd.arg(filepath);
                    }

                    let mut proc = cmd.spawn().ok().expect("Failed to spawn process");
                    proc.stdin
                        .as_mut()
                        .unwrap()
                        .write_all($relay_str.as_bytes())
                        .unwrap();
                    let output = proc.wait_with_output().unwrap();
                    // Check that it ran.
                    assert!(
                        output.status.success(),
                        "Running Relay code failed with code {:?}.\nstdout:\n{}\nstderr:\n{}",
                        output.status.code(),
                        std::str::from_utf8(output.stdout.as_slice())
                            .expect("Could not convert stderr to UTF8"),
                        std::str::from_utf8(output.stderr.as_slice())
                            .expect("Could not convert stderr to UTF8")
                    );

                    // TODO(@gussmith23) output type assumption
                    let relay_output: ndarray::ArrayD<f32> = read_npy(output_filepath).unwrap();
                    let interpreter_output = match interpret(&expr, expr.as_ref().len() - 1, &env) {
                        crate::language::interpreter::Value::Access(a) => a.tensor,
                        _ => panic!(),
                    };
                    assert!(
                        relay_output.abs_diff_eq(&interpreter_output, $tol),
                        "{:?}\nvs.\n{:?}",
                        relay_output,
                        interpreter_output
                    );
                }
            }
        };
    }

    test!(
        negative,
        1e-60,
        r#"
#[version = "0.0.5"]
def @main(%x: Tensor[(1, 3, 32, 32), float32]) -> Tensor[(1, 3, 32, 32), float32] {
  negative(%x)
}
"#,
        r#"
(compute negative (access-tensor x))
"#
    );

    test!(
        sqrt,
        1e-60,
        r#"
#[version = "0.0.5"]
def @main(%x: Tensor[(1, 3, 32, 32), float32]) -> Tensor[(1, 3, 32, 32), float32] {
  sqrt(%x)
}
"#,
        r#"
(compute sqrt (access-tensor x))
"#,
        "",
        // TODO(@gussmith23) Hardcoding test to f32
        Uniform::new(0f32, 1f32)
    );

    test!(
        constant_0,
        1e-600,
        r#"
#[version = "0.0.5"]
def @main() -> float32 {
  0.01639530062675476f
}
"#,
        r#"
(access-literal (literal 0.01639530062675476))
"#
    );

    test!(
        divide,
        1e-60,
        r#"
#[version = "0.0.5"]
def @main(%x: Tensor[(1024, 1, 1), float32], %y: Tensor[(1, 1024, 7, 7), float32]) -> Tensor[(1, 1024, 7, 7), float32] {
  divide(%x, %y) /* ty=Tensor[(1, 1024, 7, 7), float32] */
}
"#,
        r#"
(compute elementwise-div
 (access-pair
  (access (access-broadcast (access-insert-axis (access-tensor x) 0) (get-access-shape (access-tensor y))) 0)
  (access (access-tensor y) 0)
 )
)
"#
    );

    test!(
        expand_dims,
        1e-60,
        r#"
#[version = "0.0.5"]
def @main(%data: Tensor[(1, 3, 32, 32), float32]) -> Tensor[(1, 3, 1, 1, 1, 32, 32), float32] {
  expand_dims(%data, axis=2, num_newaxis=3) /* ty=Tensor[(1, 3, 1, 1, 1, 32, 32), float32] */
}
"#,
        r#"
(access-insert-axis (access-insert-axis (access-insert-axis (access-tensor data) 2) 2) 2)
"#
    );

    test!(
        max_pool2d,
        1e-60,
        r#"
#[version = "0.0.5"]
def @main(%data: Tensor[(1, 3, 32, 32), float32]) -> Tensor[(1, 3, 17, 12), float32] {
  nn.max_pool2d(%data, pool_size=[3, 4], strides=[2, 3], padding=[1, 2, 3, 4]) /* ty=Tensor[(1, 3, 17, 12), float32] */
}
"#,
        r#"
(compute reduce-max
 (access
  (access-windows
   (access
    (access-pad
     (access-pad
      (access-tensor data)
      min-padding
      2 1 3
     )
     min-padding
     3 2 4
    )
    4
   )
   (shape 1 1 3 4)
   (shape 1 1 2 3)
  )
  4
 )
)
"#
    );

    test!(
        conv2d,
        1e-5,
        r#"
#[version = "0.0.5"]
def @main(%data: Tensor[(1, 3, 32, 32), float32], %weights: Tensor[(8, 3, 3, 3), float32]) -> Tensor[(1, 8, 17, 12), float32] {
  nn.conv2d(%data, %weights, strides=[2, 3], padding=[1, 2, 3, 4]) /* ty=Tensor[(1, 8, 17, 12), float32] */
}
"#,
        r#"
(access-transpose
 (compute dot-product
  (access-cartesian-product
   (access (access-tensor weights) 1)
   (access
    (access-squeeze
     (access-squeeze
      (access-windows
       (access
        (access-pad
         (access-pad
          (access-tensor data)
          zero-padding
          2 1 3
         )
         zero-padding
         3 2 4
        )
        4
       )
       (shape 1 3 3 3)
       (shape 1 1 2 3)
      )
      4
     )
     1
    )
    3
   )
  )
 )
 (list 1 0 2 3)
)
"#
    );

    test!(
        multiply,
        1e-60,
        r#"
#[version = "0.0.5"]
def @main(%x: Tensor[(1024, 1, 1), float32], %y: Tensor[(1, 1024, 7, 7), float32]) -> Tensor[(1, 1024, 7, 7), float32] {
  multiply(%x, %y) /* ty=Tensor[(1, 1024, 7, 7), float32] */
}
"#,
        r#"
(compute elementwise-mul
 (access-pair
  (access (access-broadcast (access-insert-axis (access-tensor x) 0) (get-access-shape (access-tensor y))) 0)
  (access (access-tensor y) 0)
 )
)
"#
    );

    test!(
        add,
        1e-60,
        r#"
#[version = "0.0.5"]
def @main(%x: Tensor[(1024, 1, 1), float32], %y: Tensor[(1, 1024, 7, 7), float32]) -> Tensor[(1, 1024, 7, 7), float32] {
  add(%x, %y) /* ty=Tensor[(1, 1024, 7, 7), float32] */
}
"#,
        r#"
(compute elementwise-add
 (access-pair
  (access (access-broadcast (access-insert-axis (access-tensor x) 0) (get-access-shape (access-tensor y))) 0)
  (access (access-tensor y) 0)
 )
)
"#
    );

    test!(
        relu,
        1e-60,
        r#"
#[version = "0.0.5"]
def @main(%x: Tensor[(1, 3, 32, 32), float32]) -> Tensor[(1, 3, 32, 32), float32] {
  nn.relu(%x)
}
"#,
        r#"
(compute relu (access-tensor x))
"#
    );

    test!(
        global_avg_pool2d,
        1e-60,
        r#"
#[version = "0.0.5"]
def @main(%x: Tensor[(1, 3, 32, 32), float32]) -> Tensor[(1, 3, 1, 1), float32] {
  nn.global_avg_pool2d(%x) /* ty=Tensor[(1, 3, 1, 1), float32] */
}
"#,
        r#"
(access
 (access-insert-axis
  (access-insert-axis
   (compute reduce-mean (access (access-tensor x) 2))
   2
  )
  3
 )
 2
)
"#
    );

    test!(
        batch_flatten,
        1e-60,
        r#"
#[version = "0.0.5"]
def @main(%x: Tensor[(2, 3, 3, 4, 100), float32]) -> Tensor[(2, 3600), float32] {
  nn.batch_flatten(%x)
}
"#,
        r#"
(access-flatten (access (access-tensor x) 1))
"#
    );

    // TODO(@gussmith23) Uncomment when they fix relay parser
    //     test!(
    //         dense,
    //         1e-5,
    //         r#"
    // #[version = "0.0.5"]
    // def @main(%data: Tensor[(16, 32), float32], %weights: Tensor[(64, 32), float32]) -> Tensor[(16, 64), float32] {
    //   nn.dense(%data, %weights) /* ty=Tensor[(16, 64), float32] */
    // }
    // "#,
    //         r#"
    // (compute dot-product
    //  (access-cartesian-product
    //   (access (access-tensor data) 1)
    //   (access (access-tensor weights) 1)
    //  )
    // )
    // "#,
    //     );

    test!(
        bias_add_axis_0,
        1e-60,
        r#"
#[version = "0.0.5"]
def @main(%x: Tensor[(3, 3), float32], %y: Tensor[(3), float32]) -> Tensor[(3, 3), float32] {
  nn.bias_add(%x, %y, axis=0)
}
"#,
        r#"
(compute elementwise-add
 (access-pair
  (access (access-tensor x) 0)
  (access
   (access-broadcast
    (access-insert-axis (access-tensor y) 1)
    (get-access-shape (access-tensor x))
   )
   0
  )
 )
)
"#
    );

    test!(
        bias_add_axis_1,
        1e-60,
        r#"
#[version = "0.0.5"]
def @main(%x: Tensor[(3, 3), float32], %y: Tensor[(3), float32]) -> Tensor[(3, 3), float32] {
  nn.bias_add(%x, %y, axis=1)
}
"#,
        r#"
(compute elementwise-add
 (access-pair
  (access (access-tensor x) 0)
  (access
   (access-broadcast
    (access-insert-axis (access-tensor y) 0)
    (get-access-shape (access-tensor x))
   )
   0
  )
 )
)
"#
    );

    test!(
        softmax_0,
        1e-7,
        r#"
#[version = "0.0.5"]
def @main(%x: Tensor[(3), float32]) -> Tensor[(3), float32] {
  nn.softmax(%x) /* ty=Tensor[(3), float32] */
}
"#,
        r#"
(compute softmax (access (access-tensor x) 0))
"#
    );

    test!(
        softmax_1,
        1e-7,
        r#"
#[version = "0.0.5"]
def @main(%x: Tensor[(3), float32]) -> Tensor[(3), float32] {
  %0 = nn.softmax(%x); /* ty=Tensor[(3), float32] */
  nn.softmax(%0) /* ty=Tensor[(3), float32] */
}
"#,
        r#"
(compute softmax (access (compute softmax (access (access-tensor x) 0)) 0))
"#
    );

    // Mobilenet, simplified for inference (so batch norms are removed).
    // Generate with:
    // ```python3
    // import tvm
    // from tvm import relay
    // from tvm.relay.testing.mobilenet import get_workload
    //
    // mod, _ = get_workload()
    // mod = relay.transform.SimplifyInference()(mod)
    // print(mod.astext())
    // ```
    /// Can we parse (but not run) mobilenet?
    #[test]
    fn parse_mobilenet_simplified_for_inference() {
        let relay = r#"
#[version = "0.0.5"]
def @main(%data: Tensor[(1, 3, 224, 224), float32], %conv_block_1_conv_weight: Tensor[(32, 3, 3, 3), float32], %conv_block_1_bn_gamma: Tensor[(32), float32], %conv_block_1_bn_beta: Tensor[(32), float32], %conv_block_1_bn_moving_mean: Tensor[(32), float32], %conv_block_1_bn_moving_var: Tensor[(32), float32], %separable_conv_block_1_weight: Tensor[(32, 1, 3, 3), float32], %separable_conv_block_1_bn1_gamma: Tensor[(32), float32], %separable_conv_block_1_bn1_beta: Tensor[(32), float32], %separable_conv_block_1_bn1_moving_mean: Tensor[(32), float32], %separable_conv_block_1_bn1_moving_var: Tensor[(32), float32], %separable_conv_block_1_conv2_weight: Tensor[(64, 32, 1, 1), float32], %separable_conv_block_1_bn2_gamma: Tensor[(64), float32], %separable_conv_block_1_bn2_beta: Tensor[(64), float32], %separable_conv_block_1_bn2_moving_mean: Tensor[(64), float32], %separable_conv_block_1_bn2_moving_var: Tensor[(64), float32], %separable_conv_block_2_weight: Tensor[(64, 1, 3, 3), float32], %separable_conv_block_2_bn1_gamma: Tensor[(64), float32], %separable_conv_block_2_bn1_beta: Tensor[(64), float32], %separable_conv_block_2_bn1_moving_mean: Tensor[(64), float32], %separable_conv_block_2_bn1_moving_var: Tensor[(64), float32], %separable_conv_block_2_conv2_weight: Tensor[(128, 64, 1, 1), float32], %separable_conv_block_2_bn2_gamma: Tensor[(128), float32], %separable_conv_block_2_bn2_beta: Tensor[(128), float32], %separable_conv_block_2_bn2_moving_mean: Tensor[(128), float32], %separable_conv_block_2_bn2_moving_var: Tensor[(128), float32], %separable_conv_block_3_weight: Tensor[(128, 1, 3, 3), float32], %separable_conv_block_3_bn1_gamma: Tensor[(128), float32], %separable_conv_block_3_bn1_beta: Tensor[(128), float32], %separable_conv_block_3_bn1_moving_mean: Tensor[(128), float32], %separable_conv_block_3_bn1_moving_var: Tensor[(128), float32], %separable_conv_block_3_conv2_weight: Tensor[(128, 128, 1, 1), float32], %separable_conv_block_3_bn2_gamma: Tensor[(128), float32], %separable_conv_block_3_bn2_beta: Tensor[(128), float32], %separable_conv_block_3_bn2_moving_mean: Tensor[(128), float32], %separable_conv_block_3_bn2_moving_var: Tensor[(128), float32], %separable_conv_block_4_weight: Tensor[(128, 1, 3, 3), float32], %separable_conv_block_4_bn1_gamma: Tensor[(128), float32], %separable_conv_block_4_bn1_beta: Tensor[(128), float32], %separable_conv_block_4_bn1_moving_mean: Tensor[(128), float32], %separable_conv_block_4_bn1_moving_var: Tensor[(128), float32], %separable_conv_block_4_conv2_weight: Tensor[(256, 128, 1, 1), float32], %separable_conv_block_4_bn2_gamma: Tensor[(256), float32], %separable_conv_block_4_bn2_beta: Tensor[(256), float32], %separable_conv_block_4_bn2_moving_mean: Tensor[(256), float32], %separable_conv_block_4_bn2_moving_var: Tensor[(256), float32], %separable_conv_block_5_weight: Tensor[(256, 1, 3, 3), float32], %separable_conv_block_5_bn1_gamma: Tensor[(256), float32], %separable_conv_block_5_bn1_beta: Tensor[(256), float32], %separable_conv_block_5_bn1_moving_mean: Tensor[(256), float32], %separable_conv_block_5_bn1_moving_var: Tensor[(256), float32], %separable_conv_block_5_conv2_weight: Tensor[(256, 256, 1, 1), float32], %separable_conv_block_5_bn2_gamma: Tensor[(256), float32], %separable_conv_block_5_bn2_beta: Tensor[(256), float32], %separable_conv_block_5_bn2_moving_mean: Tensor[(256), float32], %separable_conv_block_5_bn2_moving_var: Tensor[(256), float32], %separable_conv_block_6_weight: Tensor[(256, 1, 3, 3), float32], %separable_conv_block_6_bn1_gamma: Tensor[(256), float32], %separable_conv_block_6_bn1_beta: Tensor[(256), float32], %separable_conv_block_6_bn1_moving_mean: Tensor[(256), float32], %separable_conv_block_6_bn1_moving_var: Tensor[(256), float32], %separable_conv_block_6_conv2_weight: Tensor[(512, 256, 1, 1), float32], %separable_conv_block_6_bn2_gamma: Tensor[(512), float32], %separable_conv_block_6_bn2_beta: Tensor[(512), float32], %separable_conv_block_6_bn2_moving_mean: Tensor[(512), float32], %separable_conv_block_6_bn2_moving_var: Tensor[(512), float32], %separable_conv_block_7_weight: Tensor[(512, 1, 3, 3), float32], %separable_conv_block_7_bn1_gamma: Tensor[(512), float32], %separable_conv_block_7_bn1_beta: Tensor[(512), float32], %separable_conv_block_7_bn1_moving_mean: Tensor[(512), float32], %separable_conv_block_7_bn1_moving_var: Tensor[(512), float32], %separable_conv_block_7_conv2_weight: Tensor[(512, 512, 1, 1), float32], %separable_conv_block_7_bn2_gamma: Tensor[(512), float32], %separable_conv_block_7_bn2_beta: Tensor[(512), float32], %separable_conv_block_7_bn2_moving_mean: Tensor[(512), float32], %separable_conv_block_7_bn2_moving_var: Tensor[(512), float32], %separable_conv_block_8_weight: Tensor[(512, 1, 3, 3), float32], %separable_conv_block_8_bn1_gamma: Tensor[(512), float32], %separable_conv_block_8_bn1_beta: Tensor[(512), float32], %separable_conv_block_8_bn1_moving_mean: Tensor[(512), float32], %separable_conv_block_8_bn1_moving_var: Tensor[(512), float32], %separable_conv_block_8_conv2_weight: Tensor[(512, 512, 1, 1), float32], %separable_conv_block_8_bn2_gamma: Tensor[(512), float32], %separable_conv_block_8_bn2_beta: Tensor[(512), float32], %separable_conv_block_8_bn2_moving_mean: Tensor[(512), float32], %separable_conv_block_8_bn2_moving_var: Tensor[(512), float32], %separable_conv_block_9_weight: Tensor[(512, 1, 3, 3), float32], %separable_conv_block_9_bn1_gamma: Tensor[(512), float32], %separable_conv_block_9_bn1_beta: Tensor[(512), float32], %separable_conv_block_9_bn1_moving_mean: Tensor[(512), float32], %separable_conv_block_9_bn1_moving_var: Tensor[(512), float32], %separable_conv_block_9_conv2_weight: Tensor[(512, 512, 1, 1), float32], %separable_conv_block_9_bn2_gamma: Tensor[(512), float32], %separable_conv_block_9_bn2_beta: Tensor[(512), float32], %separable_conv_block_9_bn2_moving_mean: Tensor[(512), float32], %separable_conv_block_9_bn2_moving_var: Tensor[(512), float32], %separable_conv_block_10_weight: Tensor[(512, 1, 3, 3), float32], %separable_conv_block_10_bn1_gamma: Tensor[(512), float32], %separable_conv_block_10_bn1_beta: Tensor[(512), float32], %separable_conv_block_10_bn1_moving_mean: Tensor[(512), float32], %separable_conv_block_10_bn1_moving_var: Tensor[(512), float32], %separable_conv_block_10_conv2_weight: Tensor[(512, 512, 1, 1), float32], %separable_conv_block_10_bn2_gamma: Tensor[(512), float32], %separable_conv_block_10_bn2_beta: Tensor[(512), float32], %separable_conv_block_10_bn2_moving_mean: Tensor[(512), float32], %separable_conv_block_10_bn2_moving_var: Tensor[(512), float32], %separable_conv_block_11_weight: Tensor[(512, 1, 3, 3), float32], %separable_conv_block_11_bn1_gamma: Tensor[(512), float32], %separable_conv_block_11_bn1_beta: Tensor[(512), float32], %separable_conv_block_11_bn1_moving_mean: Tensor[(512), float32], %separable_conv_block_11_bn1_moving_var: Tensor[(512), float32], %separable_conv_block_11_conv2_weight: Tensor[(512, 512, 1, 1), float32], %separable_conv_block_11_bn2_gamma: Tensor[(512), float32], %separable_conv_block_11_bn2_beta: Tensor[(512), float32], %separable_conv_block_11_bn2_moving_mean: Tensor[(512), float32], %separable_conv_block_11_bn2_moving_var: Tensor[(512), float32], %separable_conv_block_12_weight: Tensor[(512, 1, 3, 3), float32], %separable_conv_block_12_bn1_gamma: Tensor[(512), float32], %separable_conv_block_12_bn1_beta: Tensor[(512), float32], %separable_conv_block_12_bn1_moving_mean: Tensor[(512), float32], %separable_conv_block_12_bn1_moving_var: Tensor[(512), float32], %separable_conv_block_12_conv2_weight: Tensor[(1024, 512, 1, 1), float32], %separable_conv_block_12_bn2_gamma: Tensor[(1024), float32], %separable_conv_block_12_bn2_beta: Tensor[(1024), float32], %separable_conv_block_12_bn2_moving_mean: Tensor[(1024), float32], %separable_conv_block_12_bn2_moving_var: Tensor[(1024), float32], %separable_conv_block_13_weight: Tensor[(1024, 1, 3, 3), float32], %separable_conv_block_13_bn1_gamma: Tensor[(1024), float32], %separable_conv_block_13_bn1_beta: Tensor[(1024), float32], %separable_conv_block_13_bn1_moving_mean: Tensor[(1024), float32], %separable_conv_block_13_bn1_moving_var: Tensor[(1024), float32], %separable_conv_block_13_conv2_weight: Tensor[(1024, 1024, 1, 1), float32], %separable_conv_block_13_bn2_gamma: Tensor[(1024), float32], %separable_conv_block_13_bn2_beta: Tensor[(1024), float32], %separable_conv_block_13_bn2_moving_mean: Tensor[(1024), float32], %separable_conv_block_13_bn2_moving_var: Tensor[(1024), float32], %fc_weight: Tensor[(1000, 1024), float32], %fc_bias: Tensor[(1000), float32]) -> Tensor[(1, 1000), float32] {
  %0 = nn.conv2d(%data, %conv_block_1_conv_weight, strides=[2, 2], padding=[1, 1, 1, 1], channels=32, kernel_size=[3, 3]) /* ty=Tensor[(1, 32, 112, 112), float32] */;
  %1 = add(%conv_block_1_bn_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(32), float32] */;
  %2 = sqrt(%1) /* ty=Tensor[(32), float32] */;
  %3 = divide(1f /* ty=float32 */, %2) /* ty=Tensor[(32), float32] */;
  %4 = multiply(%3, %conv_block_1_bn_gamma) /* ty=Tensor[(32), float32] */;
  %5 = expand_dims(%4, axis=1, num_newaxis=2) /* ty=Tensor[(32, 1, 1), float32] */;
  %6 = multiply(%0, %5) /* ty=Tensor[(1, 32, 112, 112), float32] */;
  %7 = negative(%conv_block_1_bn_moving_mean) /* ty=Tensor[(32), float32] */;
  %8 = multiply(%7, %4) /* ty=Tensor[(32), float32] */;
  %9 = add(%8, %conv_block_1_bn_beta) /* ty=Tensor[(32), float32] */;
  %10 = expand_dims(%9, axis=1, num_newaxis=2) /* ty=Tensor[(32, 1, 1), float32] */;
  %11 = add(%6, %10) /* ty=Tensor[(1, 32, 112, 112), float32] */;
  %12 = nn.relu(%11) /* ty=Tensor[(1, 32, 112, 112), float32] */;
  %13 = nn.conv2d(%12, %separable_conv_block_1_weight, padding=[1, 1, 1, 1], groups=32, channels=32, kernel_size=[3, 3]) /* ty=Tensor[(1, 32, 112, 112), float32] */;
  %14 = add(%separable_conv_block_1_bn1_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(32), float32] */;
  %15 = sqrt(%14) /* ty=Tensor[(32), float32] */;
  %16 = divide(1f /* ty=float32 */, %15) /* ty=Tensor[(32), float32] */;
  %17 = multiply(%16, %separable_conv_block_1_bn1_gamma) /* ty=Tensor[(32), float32] */;
  %18 = expand_dims(%17, axis=1, num_newaxis=2) /* ty=Tensor[(32, 1, 1), float32] */;
  %19 = multiply(%13, %18) /* ty=Tensor[(1, 32, 112, 112), float32] */;
  %20 = negative(%separable_conv_block_1_bn1_moving_mean) /* ty=Tensor[(32), float32] */;
  %21 = multiply(%20, %17) /* ty=Tensor[(32), float32] */;
  %22 = add(%21, %separable_conv_block_1_bn1_beta) /* ty=Tensor[(32), float32] */;
  %23 = expand_dims(%22, axis=1, num_newaxis=2) /* ty=Tensor[(32, 1, 1), float32] */;
  %24 = add(%19, %23) /* ty=Tensor[(1, 32, 112, 112), float32] */;
  %25 = nn.relu(%24) /* ty=Tensor[(1, 32, 112, 112), float32] */;
  %26 = nn.conv2d(%25, %separable_conv_block_1_conv2_weight, padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1]) /* ty=Tensor[(1, 64, 112, 112), float32] */;
  %27 = add(%separable_conv_block_1_bn2_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(64), float32] */;
  %28 = sqrt(%27) /* ty=Tensor[(64), float32] */;
  %29 = divide(1f /* ty=float32 */, %28) /* ty=Tensor[(64), float32] */;
  %30 = multiply(%29, %separable_conv_block_1_bn2_gamma) /* ty=Tensor[(64), float32] */;
  %31 = expand_dims(%30, axis=1, num_newaxis=2) /* ty=Tensor[(64, 1, 1), float32] */;
  %32 = multiply(%26, %31) /* ty=Tensor[(1, 64, 112, 112), float32] */;
  %33 = negative(%separable_conv_block_1_bn2_moving_mean) /* ty=Tensor[(64), float32] */;
  %34 = multiply(%33, %30) /* ty=Tensor[(64), float32] */;
  %35 = add(%34, %separable_conv_block_1_bn2_beta) /* ty=Tensor[(64), float32] */;
  %36 = expand_dims(%35, axis=1, num_newaxis=2) /* ty=Tensor[(64, 1, 1), float32] */;
  %37 = add(%32, %36) /* ty=Tensor[(1, 64, 112, 112), float32] */;
  %38 = nn.relu(%37) /* ty=Tensor[(1, 64, 112, 112), float32] */;
  %39 = nn.conv2d(%38, %separable_conv_block_2_weight, strides=[2, 2], padding=[1, 1, 1, 1], groups=64, channels=64, kernel_size=[3, 3]) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %40 = add(%separable_conv_block_2_bn1_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(64), float32] */;
  %41 = sqrt(%40) /* ty=Tensor[(64), float32] */;
  %42 = divide(1f /* ty=float32 */, %41) /* ty=Tensor[(64), float32] */;
  %43 = multiply(%42, %separable_conv_block_2_bn1_gamma) /* ty=Tensor[(64), float32] */;
  %44 = expand_dims(%43, axis=1, num_newaxis=2) /* ty=Tensor[(64, 1, 1), float32] */;
  %45 = multiply(%39, %44) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %46 = negative(%separable_conv_block_2_bn1_moving_mean) /* ty=Tensor[(64), float32] */;
  %47 = multiply(%46, %43) /* ty=Tensor[(64), float32] */;
  %48 = add(%47, %separable_conv_block_2_bn1_beta) /* ty=Tensor[(64), float32] */;
  %49 = expand_dims(%48, axis=1, num_newaxis=2) /* ty=Tensor[(64, 1, 1), float32] */;
  %50 = add(%45, %49) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %51 = nn.relu(%50) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %52 = nn.conv2d(%51, %separable_conv_block_2_conv2_weight, padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1]) /* ty=Tensor[(1, 128, 56, 56), float32] */;
  %53 = add(%separable_conv_block_2_bn2_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(128), float32] */;
  %54 = sqrt(%53) /* ty=Tensor[(128), float32] */;
  %55 = divide(1f /* ty=float32 */, %54) /* ty=Tensor[(128), float32] */;
  %56 = multiply(%55, %separable_conv_block_2_bn2_gamma) /* ty=Tensor[(128), float32] */;
  %57 = expand_dims(%56, axis=1, num_newaxis=2) /* ty=Tensor[(128, 1, 1), float32] */;
  %58 = multiply(%52, %57) /* ty=Tensor[(1, 128, 56, 56), float32] */;
  %59 = negative(%separable_conv_block_2_bn2_moving_mean) /* ty=Tensor[(128), float32] */;
  %60 = multiply(%59, %56) /* ty=Tensor[(128), float32] */;
  %61 = add(%60, %separable_conv_block_2_bn2_beta) /* ty=Tensor[(128), float32] */;
  %62 = expand_dims(%61, axis=1, num_newaxis=2) /* ty=Tensor[(128, 1, 1), float32] */;
  %63 = add(%58, %62) /* ty=Tensor[(1, 128, 56, 56), float32] */;
  %64 = nn.relu(%63) /* ty=Tensor[(1, 128, 56, 56), float32] */;
  %65 = nn.conv2d(%64, %separable_conv_block_3_weight, padding=[1, 1, 1, 1], groups=128, channels=128, kernel_size=[3, 3]) /* ty=Tensor[(1, 128, 56, 56), float32] */;
  %66 = add(%separable_conv_block_3_bn1_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(128), float32] */;
  %67 = sqrt(%66) /* ty=Tensor[(128), float32] */;
  %68 = divide(1f /* ty=float32 */, %67) /* ty=Tensor[(128), float32] */;
  %69 = multiply(%68, %separable_conv_block_3_bn1_gamma) /* ty=Tensor[(128), float32] */;
  %70 = expand_dims(%69, axis=1, num_newaxis=2) /* ty=Tensor[(128, 1, 1), float32] */;
  %71 = multiply(%65, %70) /* ty=Tensor[(1, 128, 56, 56), float32] */;
  %72 = negative(%separable_conv_block_3_bn1_moving_mean) /* ty=Tensor[(128), float32] */;
  %73 = multiply(%72, %69) /* ty=Tensor[(128), float32] */;
  %74 = add(%73, %separable_conv_block_3_bn1_beta) /* ty=Tensor[(128), float32] */;
  %75 = expand_dims(%74, axis=1, num_newaxis=2) /* ty=Tensor[(128, 1, 1), float32] */;
  %76 = add(%71, %75) /* ty=Tensor[(1, 128, 56, 56), float32] */;
  %77 = nn.relu(%76) /* ty=Tensor[(1, 128, 56, 56), float32] */;
  %78 = nn.conv2d(%77, %separable_conv_block_3_conv2_weight, padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1]) /* ty=Tensor[(1, 128, 56, 56), float32] */;
  %79 = add(%separable_conv_block_3_bn2_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(128), float32] */;
  %80 = sqrt(%79) /* ty=Tensor[(128), float32] */;
  %81 = divide(1f /* ty=float32 */, %80) /* ty=Tensor[(128), float32] */;
  %82 = multiply(%81, %separable_conv_block_3_bn2_gamma) /* ty=Tensor[(128), float32] */;
  %83 = expand_dims(%82, axis=1, num_newaxis=2) /* ty=Tensor[(128, 1, 1), float32] */;
  %84 = multiply(%78, %83) /* ty=Tensor[(1, 128, 56, 56), float32] */;
  %85 = negative(%separable_conv_block_3_bn2_moving_mean) /* ty=Tensor[(128), float32] */;
  %86 = multiply(%85, %82) /* ty=Tensor[(128), float32] */;
  %87 = add(%86, %separable_conv_block_3_bn2_beta) /* ty=Tensor[(128), float32] */;
  %88 = expand_dims(%87, axis=1, num_newaxis=2) /* ty=Tensor[(128, 1, 1), float32] */;
  %89 = add(%84, %88) /* ty=Tensor[(1, 128, 56, 56), float32] */;
  %90 = nn.relu(%89) /* ty=Tensor[(1, 128, 56, 56), float32] */;
  %91 = nn.conv2d(%90, %separable_conv_block_4_weight, strides=[2, 2], padding=[1, 1, 1, 1], groups=128, channels=128, kernel_size=[3, 3]) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %92 = add(%separable_conv_block_4_bn1_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(128), float32] */;
  %93 = sqrt(%92) /* ty=Tensor[(128), float32] */;
  %94 = divide(1f /* ty=float32 */, %93) /* ty=Tensor[(128), float32] */;
  %95 = multiply(%94, %separable_conv_block_4_bn1_gamma) /* ty=Tensor[(128), float32] */;
  %96 = expand_dims(%95, axis=1, num_newaxis=2) /* ty=Tensor[(128, 1, 1), float32] */;
  %97 = multiply(%91, %96) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %98 = negative(%separable_conv_block_4_bn1_moving_mean) /* ty=Tensor[(128), float32] */;
  %99 = multiply(%98, %95) /* ty=Tensor[(128), float32] */;
  %100 = add(%99, %separable_conv_block_4_bn1_beta) /* ty=Tensor[(128), float32] */;
  %101 = expand_dims(%100, axis=1, num_newaxis=2) /* ty=Tensor[(128, 1, 1), float32] */;
  %102 = add(%97, %101) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %103 = nn.relu(%102) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %104 = nn.conv2d(%103, %separable_conv_block_4_conv2_weight, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]) /* ty=Tensor[(1, 256, 28, 28), float32] */;
  %105 = add(%separable_conv_block_4_bn2_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(256), float32] */;
  %106 = sqrt(%105) /* ty=Tensor[(256), float32] */;
  %107 = divide(1f /* ty=float32 */, %106) /* ty=Tensor[(256), float32] */;
  %108 = multiply(%107, %separable_conv_block_4_bn2_gamma) /* ty=Tensor[(256), float32] */;
  %109 = expand_dims(%108, axis=1, num_newaxis=2) /* ty=Tensor[(256, 1, 1), float32] */;
  %110 = multiply(%104, %109) /* ty=Tensor[(1, 256, 28, 28), float32] */;
  %111 = negative(%separable_conv_block_4_bn2_moving_mean) /* ty=Tensor[(256), float32] */;
  %112 = multiply(%111, %108) /* ty=Tensor[(256), float32] */;
  %113 = add(%112, %separable_conv_block_4_bn2_beta) /* ty=Tensor[(256), float32] */;
  %114 = expand_dims(%113, axis=1, num_newaxis=2) /* ty=Tensor[(256, 1, 1), float32] */;
  %115 = add(%110, %114) /* ty=Tensor[(1, 256, 28, 28), float32] */;
  %116 = nn.relu(%115) /* ty=Tensor[(1, 256, 28, 28), float32] */;
  %117 = nn.conv2d(%116, %separable_conv_block_5_weight, padding=[1, 1, 1, 1], groups=256, channels=256, kernel_size=[3, 3]) /* ty=Tensor[(1, 256, 28, 28), float32] */;
  %118 = add(%separable_conv_block_5_bn1_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(256), float32] */;
  %119 = sqrt(%118) /* ty=Tensor[(256), float32] */;
  %120 = divide(1f /* ty=float32 */, %119) /* ty=Tensor[(256), float32] */;
  %121 = multiply(%120, %separable_conv_block_5_bn1_gamma) /* ty=Tensor[(256), float32] */;
  %122 = expand_dims(%121, axis=1, num_newaxis=2) /* ty=Tensor[(256, 1, 1), float32] */;
  %123 = multiply(%117, %122) /* ty=Tensor[(1, 256, 28, 28), float32] */;
  %124 = negative(%separable_conv_block_5_bn1_moving_mean) /* ty=Tensor[(256), float32] */;
  %125 = multiply(%124, %121) /* ty=Tensor[(256), float32] */;
  %126 = add(%125, %separable_conv_block_5_bn1_beta) /* ty=Tensor[(256), float32] */;
  %127 = expand_dims(%126, axis=1, num_newaxis=2) /* ty=Tensor[(256, 1, 1), float32] */;
  %128 = add(%123, %127) /* ty=Tensor[(1, 256, 28, 28), float32] */;
  %129 = nn.relu(%128) /* ty=Tensor[(1, 256, 28, 28), float32] */;
  %130 = nn.conv2d(%129, %separable_conv_block_5_conv2_weight, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]) /* ty=Tensor[(1, 256, 28, 28), float32] */;
  %131 = add(%separable_conv_block_5_bn2_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(256), float32] */;
  %132 = sqrt(%131) /* ty=Tensor[(256), float32] */;
  %133 = divide(1f /* ty=float32 */, %132) /* ty=Tensor[(256), float32] */;
  %134 = multiply(%133, %separable_conv_block_5_bn2_gamma) /* ty=Tensor[(256), float32] */;
  %135 = expand_dims(%134, axis=1, num_newaxis=2) /* ty=Tensor[(256, 1, 1), float32] */;
  %136 = multiply(%130, %135) /* ty=Tensor[(1, 256, 28, 28), float32] */;
  %137 = negative(%separable_conv_block_5_bn2_moving_mean) /* ty=Tensor[(256), float32] */;
  %138 = multiply(%137, %134) /* ty=Tensor[(256), float32] */;
  %139 = add(%138, %separable_conv_block_5_bn2_beta) /* ty=Tensor[(256), float32] */;
  %140 = expand_dims(%139, axis=1, num_newaxis=2) /* ty=Tensor[(256, 1, 1), float32] */;
  %141 = add(%136, %140) /* ty=Tensor[(1, 256, 28, 28), float32] */;
  %142 = nn.relu(%141) /* ty=Tensor[(1, 256, 28, 28), float32] */;
  %143 = nn.conv2d(%142, %separable_conv_block_6_weight, strides=[2, 2], padding=[1, 1, 1, 1], groups=256, channels=256, kernel_size=[3, 3]) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %144 = add(%separable_conv_block_6_bn1_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(256), float32] */;
  %145 = sqrt(%144) /* ty=Tensor[(256), float32] */;
  %146 = divide(1f /* ty=float32 */, %145) /* ty=Tensor[(256), float32] */;
  %147 = multiply(%146, %separable_conv_block_6_bn1_gamma) /* ty=Tensor[(256), float32] */;
  %148 = expand_dims(%147, axis=1, num_newaxis=2) /* ty=Tensor[(256, 1, 1), float32] */;
  %149 = multiply(%143, %148) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %150 = negative(%separable_conv_block_6_bn1_moving_mean) /* ty=Tensor[(256), float32] */;
  %151 = multiply(%150, %147) /* ty=Tensor[(256), float32] */;
  %152 = add(%151, %separable_conv_block_6_bn1_beta) /* ty=Tensor[(256), float32] */;
  %153 = expand_dims(%152, axis=1, num_newaxis=2) /* ty=Tensor[(256, 1, 1), float32] */;
  %154 = add(%149, %153) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %155 = nn.relu(%154) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %156 = nn.conv2d(%155, %separable_conv_block_6_conv2_weight, padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %157 = add(%separable_conv_block_6_bn2_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(512), float32] */;
  %158 = sqrt(%157) /* ty=Tensor[(512), float32] */;
  %159 = divide(1f /* ty=float32 */, %158) /* ty=Tensor[(512), float32] */;
  %160 = multiply(%159, %separable_conv_block_6_bn2_gamma) /* ty=Tensor[(512), float32] */;
  %161 = expand_dims(%160, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %162 = multiply(%156, %161) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %163 = negative(%separable_conv_block_6_bn2_moving_mean) /* ty=Tensor[(512), float32] */;
  %164 = multiply(%163, %160) /* ty=Tensor[(512), float32] */;
  %165 = add(%164, %separable_conv_block_6_bn2_beta) /* ty=Tensor[(512), float32] */;
  %166 = expand_dims(%165, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %167 = add(%162, %166) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %168 = nn.relu(%167) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %169 = nn.conv2d(%168, %separable_conv_block_7_weight, padding=[1, 1, 1, 1], groups=512, channels=512, kernel_size=[3, 3]) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %170 = add(%separable_conv_block_7_bn1_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(512), float32] */;
  %171 = sqrt(%170) /* ty=Tensor[(512), float32] */;
  %172 = divide(1f /* ty=float32 */, %171) /* ty=Tensor[(512), float32] */;
  %173 = multiply(%172, %separable_conv_block_7_bn1_gamma) /* ty=Tensor[(512), float32] */;
  %174 = expand_dims(%173, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %175 = multiply(%169, %174) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %176 = negative(%separable_conv_block_7_bn1_moving_mean) /* ty=Tensor[(512), float32] */;
  %177 = multiply(%176, %173) /* ty=Tensor[(512), float32] */;
  %178 = add(%177, %separable_conv_block_7_bn1_beta) /* ty=Tensor[(512), float32] */;
  %179 = expand_dims(%178, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %180 = add(%175, %179) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %181 = nn.relu(%180) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %182 = nn.conv2d(%181, %separable_conv_block_7_conv2_weight, padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %183 = add(%separable_conv_block_7_bn2_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(512), float32] */;
  %184 = sqrt(%183) /* ty=Tensor[(512), float32] */;
  %185 = divide(1f /* ty=float32 */, %184) /* ty=Tensor[(512), float32] */;
  %186 = multiply(%185, %separable_conv_block_7_bn2_gamma) /* ty=Tensor[(512), float32] */;
  %187 = expand_dims(%186, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %188 = multiply(%182, %187) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %189 = negative(%separable_conv_block_7_bn2_moving_mean) /* ty=Tensor[(512), float32] */;
  %190 = multiply(%189, %186) /* ty=Tensor[(512), float32] */;
  %191 = add(%190, %separable_conv_block_7_bn2_beta) /* ty=Tensor[(512), float32] */;
  %192 = expand_dims(%191, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %193 = add(%188, %192) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %194 = nn.relu(%193) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %195 = nn.conv2d(%194, %separable_conv_block_8_weight, padding=[1, 1, 1, 1], groups=512, channels=512, kernel_size=[3, 3]) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %196 = add(%separable_conv_block_8_bn1_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(512), float32] */;
  %197 = sqrt(%196) /* ty=Tensor[(512), float32] */;
  %198 = divide(1f /* ty=float32 */, %197) /* ty=Tensor[(512), float32] */;
  %199 = multiply(%198, %separable_conv_block_8_bn1_gamma) /* ty=Tensor[(512), float32] */;
  %200 = expand_dims(%199, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %201 = multiply(%195, %200) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %202 = negative(%separable_conv_block_8_bn1_moving_mean) /* ty=Tensor[(512), float32] */;
  %203 = multiply(%202, %199) /* ty=Tensor[(512), float32] */;
  %204 = add(%203, %separable_conv_block_8_bn1_beta) /* ty=Tensor[(512), float32] */;
  %205 = expand_dims(%204, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %206 = add(%201, %205) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %207 = nn.relu(%206) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %208 = nn.conv2d(%207, %separable_conv_block_8_conv2_weight, padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %209 = add(%separable_conv_block_8_bn2_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(512), float32] */;
  %210 = sqrt(%209) /* ty=Tensor[(512), float32] */;
  %211 = divide(1f /* ty=float32 */, %210) /* ty=Tensor[(512), float32] */;
  %212 = multiply(%211, %separable_conv_block_8_bn2_gamma) /* ty=Tensor[(512), float32] */;
  %213 = expand_dims(%212, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %214 = multiply(%208, %213) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %215 = negative(%separable_conv_block_8_bn2_moving_mean) /* ty=Tensor[(512), float32] */;
  %216 = multiply(%215, %212) /* ty=Tensor[(512), float32] */;
  %217 = add(%216, %separable_conv_block_8_bn2_beta) /* ty=Tensor[(512), float32] */;
  %218 = expand_dims(%217, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %219 = add(%214, %218) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %220 = nn.relu(%219) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %221 = nn.conv2d(%220, %separable_conv_block_9_weight, padding=[1, 1, 1, 1], groups=512, channels=512, kernel_size=[3, 3]) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %222 = add(%separable_conv_block_9_bn1_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(512), float32] */;
  %223 = sqrt(%222) /* ty=Tensor[(512), float32] */;
  %224 = divide(1f /* ty=float32 */, %223) /* ty=Tensor[(512), float32] */;
  %225 = multiply(%224, %separable_conv_block_9_bn1_gamma) /* ty=Tensor[(512), float32] */;
  %226 = expand_dims(%225, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %227 = multiply(%221, %226) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %228 = negative(%separable_conv_block_9_bn1_moving_mean) /* ty=Tensor[(512), float32] */;
  %229 = multiply(%228, %225) /* ty=Tensor[(512), float32] */;
  %230 = add(%229, %separable_conv_block_9_bn1_beta) /* ty=Tensor[(512), float32] */;
  %231 = expand_dims(%230, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %232 = add(%227, %231) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %233 = nn.relu(%232) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %234 = nn.conv2d(%233, %separable_conv_block_9_conv2_weight, padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %235 = add(%separable_conv_block_9_bn2_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(512), float32] */;
  %236 = sqrt(%235) /* ty=Tensor[(512), float32] */;
  %237 = divide(1f /* ty=float32 */, %236) /* ty=Tensor[(512), float32] */;
  %238 = multiply(%237, %separable_conv_block_9_bn2_gamma) /* ty=Tensor[(512), float32] */;
  %239 = expand_dims(%238, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %240 = multiply(%234, %239) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %241 = negative(%separable_conv_block_9_bn2_moving_mean) /* ty=Tensor[(512), float32] */;
  %242 = multiply(%241, %238) /* ty=Tensor[(512), float32] */;
  %243 = add(%242, %separable_conv_block_9_bn2_beta) /* ty=Tensor[(512), float32] */;
  %244 = expand_dims(%243, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %245 = add(%240, %244) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %246 = nn.relu(%245) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %247 = nn.conv2d(%246, %separable_conv_block_10_weight, padding=[1, 1, 1, 1], groups=512, channels=512, kernel_size=[3, 3]) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %248 = add(%separable_conv_block_10_bn1_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(512), float32] */;
  %249 = sqrt(%248) /* ty=Tensor[(512), float32] */;
  %250 = divide(1f /* ty=float32 */, %249) /* ty=Tensor[(512), float32] */;
  %251 = multiply(%250, %separable_conv_block_10_bn1_gamma) /* ty=Tensor[(512), float32] */;
  %252 = expand_dims(%251, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %253 = multiply(%247, %252) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %254 = negative(%separable_conv_block_10_bn1_moving_mean) /* ty=Tensor[(512), float32] */;
  %255 = multiply(%254, %251) /* ty=Tensor[(512), float32] */;
  %256 = add(%255, %separable_conv_block_10_bn1_beta) /* ty=Tensor[(512), float32] */;
  %257 = expand_dims(%256, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %258 = add(%253, %257) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %259 = nn.relu(%258) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %260 = nn.conv2d(%259, %separable_conv_block_10_conv2_weight, padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %261 = add(%separable_conv_block_10_bn2_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(512), float32] */;
  %262 = sqrt(%261) /* ty=Tensor[(512), float32] */;
  %263 = divide(1f /* ty=float32 */, %262) /* ty=Tensor[(512), float32] */;
  %264 = multiply(%263, %separable_conv_block_10_bn2_gamma) /* ty=Tensor[(512), float32] */;
  %265 = expand_dims(%264, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %266 = multiply(%260, %265) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %267 = negative(%separable_conv_block_10_bn2_moving_mean) /* ty=Tensor[(512), float32] */;
  %268 = multiply(%267, %264) /* ty=Tensor[(512), float32] */;
  %269 = add(%268, %separable_conv_block_10_bn2_beta) /* ty=Tensor[(512), float32] */;
  %270 = expand_dims(%269, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %271 = add(%266, %270) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %272 = nn.relu(%271) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %273 = nn.conv2d(%272, %separable_conv_block_11_weight, padding=[1, 1, 1, 1], groups=512, channels=512, kernel_size=[3, 3]) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %274 = add(%separable_conv_block_11_bn1_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(512), float32] */;
  %275 = sqrt(%274) /* ty=Tensor[(512), float32] */;
  %276 = divide(1f /* ty=float32 */, %275) /* ty=Tensor[(512), float32] */;
  %277 = multiply(%276, %separable_conv_block_11_bn1_gamma) /* ty=Tensor[(512), float32] */;
  %278 = expand_dims(%277, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %279 = multiply(%273, %278) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %280 = negative(%separable_conv_block_11_bn1_moving_mean) /* ty=Tensor[(512), float32] */;
  %281 = multiply(%280, %277) /* ty=Tensor[(512), float32] */;
  %282 = add(%281, %separable_conv_block_11_bn1_beta) /* ty=Tensor[(512), float32] */;
  %283 = expand_dims(%282, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %284 = add(%279, %283) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %285 = nn.relu(%284) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %286 = nn.conv2d(%285, %separable_conv_block_11_conv2_weight, padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %287 = add(%separable_conv_block_11_bn2_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(512), float32] */;
  %288 = sqrt(%287) /* ty=Tensor[(512), float32] */;
  %289 = divide(1f /* ty=float32 */, %288) /* ty=Tensor[(512), float32] */;
  %290 = multiply(%289, %separable_conv_block_11_bn2_gamma) /* ty=Tensor[(512), float32] */;
  %291 = expand_dims(%290, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %292 = multiply(%286, %291) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %293 = negative(%separable_conv_block_11_bn2_moving_mean) /* ty=Tensor[(512), float32] */;
  %294 = multiply(%293, %290) /* ty=Tensor[(512), float32] */;
  %295 = add(%294, %separable_conv_block_11_bn2_beta) /* ty=Tensor[(512), float32] */;
  %296 = expand_dims(%295, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %297 = add(%292, %296) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %298 = nn.relu(%297) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %299 = nn.conv2d(%298, %separable_conv_block_12_weight, strides=[2, 2], padding=[1, 1, 1, 1], groups=512, channels=512, kernel_size=[3, 3]) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %300 = add(%separable_conv_block_12_bn1_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(512), float32] */;
  %301 = sqrt(%300) /* ty=Tensor[(512), float32] */;
  %302 = divide(1f /* ty=float32 */, %301) /* ty=Tensor[(512), float32] */;
  %303 = multiply(%302, %separable_conv_block_12_bn1_gamma) /* ty=Tensor[(512), float32] */;
  %304 = expand_dims(%303, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %305 = multiply(%299, %304) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %306 = negative(%separable_conv_block_12_bn1_moving_mean) /* ty=Tensor[(512), float32] */;
  %307 = multiply(%306, %303) /* ty=Tensor[(512), float32] */;
  %308 = add(%307, %separable_conv_block_12_bn1_beta) /* ty=Tensor[(512), float32] */;
  %309 = expand_dims(%308, axis=1, num_newaxis=2) /* ty=Tensor[(512, 1, 1), float32] */;
  %310 = add(%305, %309) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %311 = nn.relu(%310) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %312 = nn.conv2d(%311, %separable_conv_block_12_conv2_weight, padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1]) /* ty=Tensor[(1, 1024, 7, 7), float32] */;
  %313 = add(%separable_conv_block_12_bn2_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(1024), float32] */;
  %314 = sqrt(%313) /* ty=Tensor[(1024), float32] */;
  %315 = divide(1f /* ty=float32 */, %314) /* ty=Tensor[(1024), float32] */;
  %316 = multiply(%315, %separable_conv_block_12_bn2_gamma) /* ty=Tensor[(1024), float32] */;
  %317 = expand_dims(%316, axis=1, num_newaxis=2) /* ty=Tensor[(1024, 1, 1), float32] */;
  %318 = multiply(%312, %317) /* ty=Tensor[(1, 1024, 7, 7), float32] */;
  %319 = negative(%separable_conv_block_12_bn2_moving_mean) /* ty=Tensor[(1024), float32] */;
  %320 = multiply(%319, %316) /* ty=Tensor[(1024), float32] */;
  %321 = add(%320, %separable_conv_block_12_bn2_beta) /* ty=Tensor[(1024), float32] */;
  %322 = expand_dims(%321, axis=1, num_newaxis=2) /* ty=Tensor[(1024, 1, 1), float32] */;
  %323 = add(%318, %322) /* ty=Tensor[(1, 1024, 7, 7), float32] */;
  %324 = nn.relu(%323) /* ty=Tensor[(1, 1024, 7, 7), float32] */;
  %325 = nn.conv2d(%324, %separable_conv_block_13_weight, padding=[1, 1, 1, 1], groups=1024, channels=1024, kernel_size=[3, 3]) /* ty=Tensor[(1, 1024, 7, 7), float32] */;
  %326 = add(%separable_conv_block_13_bn1_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(1024), float32] */;
  %327 = sqrt(%326) /* ty=Tensor[(1024), float32] */;
  %328 = divide(1f /* ty=float32 */, %327) /* ty=Tensor[(1024), float32] */;
  %329 = multiply(%328, %separable_conv_block_13_bn1_gamma) /* ty=Tensor[(1024), float32] */;
  %330 = expand_dims(%329, axis=1, num_newaxis=2) /* ty=Tensor[(1024, 1, 1), float32] */;
  %331 = multiply(%325, %330) /* ty=Tensor[(1, 1024, 7, 7), float32] */;
  %332 = negative(%separable_conv_block_13_bn1_moving_mean) /* ty=Tensor[(1024), float32] */;
  %333 = multiply(%332, %329) /* ty=Tensor[(1024), float32] */;
  %334 = add(%333, %separable_conv_block_13_bn1_beta) /* ty=Tensor[(1024), float32] */;
  %335 = expand_dims(%334, axis=1, num_newaxis=2) /* ty=Tensor[(1024, 1, 1), float32] */;
  %336 = add(%331, %335) /* ty=Tensor[(1, 1024, 7, 7), float32] */;
  %337 = nn.relu(%336) /* ty=Tensor[(1, 1024, 7, 7), float32] */;
  %338 = nn.conv2d(%337, %separable_conv_block_13_conv2_weight, padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1]) /* ty=Tensor[(1, 1024, 7, 7), float32] */;
  %339 = add(%separable_conv_block_13_bn2_moving_var, 1e-05f /* ty=float32 */) /* ty=Tensor[(1024), float32] */;
  %340 = sqrt(%339) /* ty=Tensor[(1024), float32] */;
  %341 = divide(1f /* ty=float32 */, %340) /* ty=Tensor[(1024), float32] */;
  %342 = multiply(%341, %separable_conv_block_13_bn2_gamma) /* ty=Tensor[(1024), float32] */;
  %343 = expand_dims(%342, axis=1, num_newaxis=2) /* ty=Tensor[(1024, 1, 1), float32] */;
  %344 = multiply(%338, %343) /* ty=Tensor[(1, 1024, 7, 7), float32] */;
  %345 = negative(%separable_conv_block_13_bn2_moving_mean) /* ty=Tensor[(1024), float32] */;
  %346 = multiply(%345, %342) /* ty=Tensor[(1024), float32] */;
  %347 = add(%346, %separable_conv_block_13_bn2_beta) /* ty=Tensor[(1024), float32] */;
  %348 = expand_dims(%347, axis=1, num_newaxis=2) /* ty=Tensor[(1024, 1, 1), float32] */;
  %349 = add(%344, %348) /* ty=Tensor[(1, 1024, 7, 7), float32] */;
  %350 = nn.relu(%349) /* ty=Tensor[(1, 1024, 7, 7), float32] */;
  %351 = nn.global_avg_pool2d(%350) /* ty=Tensor[(1, 1024, 1, 1), float32] */;
  %352 = nn.batch_flatten(%351) /* ty=Tensor[(1, 1024), float32] */;
  %353 = nn.dense(%352, %fc_weight, units=1000) /* ty=Tensor[(1, 1000), float32] */;
  %354 = nn.bias_add(%353, %fc_bias) /* ty=Tensor[(1, 1000), float32] */;
  nn.softmax(%354) /* ty=Tensor[(1, 1000), float32] */
}
"#;
        let module = tvm::ir::module::IRModule::parse("", relay);

        let (expr, shapes_vec) = super::from_relay(&module);

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
        egraph.add_expr(&expr);
    }
}
