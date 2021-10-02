// TODO(@gussmith23) Make sure TVM feature flag is getting tested in CI
#![cfg(feature = "tvm")]

use crate::language::Language;
use egg::{Id, RecExpr};
use ordered_float::NotNan;
use std::collections::{HashMap, HashSet};
use std::convert::TryInto;
use tvm::ir::module::*;
use tvm::ir::relay::*;
use tvm::ir::tir::*;
use tvm::ir::ty::*;
use tvm::runtime::IsObjectRef;

use super::ComputeType;
use super::PadType;
use super::RelayOperator;

pub fn list(expr: &mut RecExpr<Language>, list: &[usize]) -> Id {
    let mut id_list: Vec<Id> = Vec::default();
    for i in list {
        id_list.push(expr.add(Language::Usize(*i)));
    }
    expr.add(Language::List(id_list.into_boxed_slice()))
}

// TODO(@gussmith23) Give glenside-expression-creation helpers a new home
pub fn access_transpose(expr: &mut RecExpr<Language>, data_id: Id, transpose_list: &[usize]) -> Id {
    let transpose_list_id = list(expr, transpose_list);

    expr.add(Language::AccessTranspose([data_id, transpose_list_id]))
}
pub fn conv1d(
    expr: &mut RecExpr<Language>,
    data_id: Id,
    data_shape: &[usize],
    weights_id: Id, 
    weights_shape: &[usize],
    strides: &[usize],
    padding: &[usize],
    dilation: &[usize],
    groups: usize,
    data_layout: &str,
    kernel_layout: &str,
    out_layout: &str,
) -> Id {
    assert_eq!(data_shape.len(), 3);
    assert_eq!(weights_shape.len(), 3);
    assert_eq!(strides.len(), 1);
    assert_eq!(padding.len(), 2);
    assert_eq!(dilation.len(), 1);
    assert_eq!(groups, 1);

    assert!(
        &["NCW"].contains(&data_layout)
    );
    assert!(
        &["OIW"].contains(&kernel_layout)
    );
    // check if alternative layouts are correct 
    assert_eq!(dilation, [1]);
    assert_eq!(out_layout, ""); 
    /*VISHAL: not sure what this is; are we saying that we always want the output
    layout to be the same as data_layout */

    //TODO: Make syre data layout is corect (look at Conv2d shuffling for inspiration, or ask Mike which data layout
    // we need for the minimum and just only assert for that :)


    // let (data_id, data_shape) = match data_layout {
    //     "NCHW" => (data_id, Vec::from(data_shape)),
    //     "NHWC" => (
    //         access_transpose(expr, data_id, &[0, 3, 1, 2]),
    //         vec![data_shape[0], data_shape[3], data_shape[1], data_shape[2]],
    //     ),
    //     _ => unreachable!(),
    // };

    // // Transpose to OIHW
    // let (weights_id, weights_shape) = match kernel_layout {
    //     "OIHW" => (weights_id, Vec::from(weights_shape)),
    //     "HWIO" => (
    //         access_transpose(expr, weights_id, &[3, 2, 0, 1]),
    //         vec![
    //             weights_shape[3],
    //             weights_shape[2],
    //             weights_shape[0],
    //             weights_shape[1],
    //         ],
    //     ),
    //     _ => unreachable!(),
    // };


    let pad_axis_id = expr.add(Language::Usize(2));
    let access_dim_id = expr.add(Language::Usize(1));
    let pad_before_id = expr.add(Language::Usize(padding[0]));
    let pad_after_id = expr.add(Language::Usize(padding[1]));
    let zero_padding_id = expr.add(Language::PadType(PadType::ZeroPadding));
    // let data_id = expr.add(Language::Access([data_id, access_dim_id]));
    let data_id = expr.add(Language::AccessPad([
        data_id,
        zero_padding_id,
        pad_axis_id,
        pad_before_id,
        pad_after_id,
    ]));
    //gets the inner access-pad (or in the case of conv1d, the singular access-pad)

    //SKIP SECOND ACCESS-PAD (Conv1d is simpler)



    // SKIP ACCESS (Conv1d is easier; double check if we can do this)

    //TODO: Figure out how stride_list changes 
    let mut stride_list = Vec::default();
    stride_list.push(expr.add(Language::Usize(1)));
    stride_list.push(expr.add(Language::Usize(1)));
    stride_list.push(expr.add(Language::Usize(strides[0])));
    let stride_shape_id = expr.add(Language::Shape(Box::from(stride_list.as_slice())));
    
    let usize_o_id = expr.add(Language::Usize(1));
    let usize_c_id = expr.add(Language::Usize(weights_shape[1]));
    let usize_kw_id = expr.add(Language::Usize(weights_shape[2]));
    let weights_shape_id = expr.add(Language::Shape(Box::new([
        usize_o_id,
        usize_c_id,
        usize_kw_id,
    ])));
    // let data_id = expr.add(Language::Access([data_id, access_dim_id]));
    let data_id = expr.add(Language::AccessWindows([
        data_id,
        weights_shape_id,
        stride_shape_id,
    ]));
    let dim_id_3 = expr.add(Language::Usize(3));
    let data_id = expr.add(Language::AccessSqueeze([data_id, dim_id_3]));
    // data_id = cartProd (data_id, (access weights 1))
    let weights_id = expr.add(Language::Access([weights_id, access_dim_id]));
    let data_id = expr.add(Language::AccessCartesianProduct([weights_id, data_id]));
    let data_id = expr.add(Language::AccessSqueeze([data_id, access_dim_id]));

    let compute_type_id = expr.add(Language::ComputeType(ComputeType::DotProduct));
    let data_id = expr.add(Language::Compute([compute_type_id, data_id]));

    let data_id = access_transpose(expr, data_id, &[0, 2, 1]);

    data_id

}
pub fn conv2d(
    expr: &mut RecExpr<Language>,
    data_id: Id,
    data_shape: &[usize],
    weights_id: Id,
    weights_shape: &[usize],
    strides: &[usize],
    padding: &[usize],
    dilation: &[usize],
    groups: usize,
    // We get these from the shapes above. Remove?
    // _channels: usize,
    // _kernel_size: [usize; 2],
    data_layout: &str,
    kernel_layout: &str,
    out_layout: &str,
) -> Id {
    assert_eq!(data_shape.len(), 4);
    assert_eq!(weights_shape.len(), 4);
    assert_eq!(strides.len(), 2);
    assert_eq!(padding.len(), 4);
    assert_eq!(dilation.len(), 2);

    assert!(
        &["NCHW", "NHWC"].contains(&data_layout),
        "NCHW and NHWC are the only layouts supported at the moment"
    );
    assert!(
        ["OIHW", "HWIO"].contains(&kernel_layout),
        "OIHW and HWIO are the only layouts supported at the moment"
    );

    assert_eq!(dilation, [1, 1]);
    assert_eq!(out_layout, "");

    // Transpose to NCHW
    let (data_id, data_shape) = match data_layout {
        "NCHW" => (data_id, Vec::from(data_shape)),
        "NHWC" => (
            access_transpose(expr, data_id, &[0, 3, 1, 2]),
            vec![data_shape[0], data_shape[3], data_shape[1], data_shape[2]],
        ),
        _ => unreachable!(),
    };

    // Transpose to OIHW
    let (weights_id, weights_shape) = match kernel_layout {
        "OIHW" => (weights_id, Vec::from(weights_shape)),
        "HWIO" => (
            access_transpose(expr, weights_id, &[3, 2, 0, 1]),
            vec![
                weights_shape[3],
                weights_shape[2],
                weights_shape[0],
                weights_shape[1],
            ],
        ),
        _ => unreachable!(),
    };

    let pad_axis_id = expr.add(Language::Usize(2));
    let pad_before_id = expr.add(Language::Usize(padding[0]));
    let pad_after_id = expr.add(Language::Usize(padding[2]));
    let zero_padding_id = expr.add(Language::PadType(PadType::ZeroPadding));
    let data_id = expr.add(Language::AccessPad([
        data_id,
        zero_padding_id,
        pad_axis_id,
        pad_before_id,
        pad_after_id,
    ]));

    let pad_axis_id = expr.add(Language::Usize(3));
    let pad_before_id = expr.add(Language::Usize(padding[1]));
    let pad_after_id = expr.add(Language::Usize(padding[3]));
    let zero_padding_id = expr.add(Language::PadType(PadType::ZeroPadding));
    let data_id = expr.add(Language::AccessPad([
        data_id,
        zero_padding_id,
        pad_axis_id,
        pad_before_id,
        pad_after_id,
    ]));

    let access_axis_id = expr.add(Language::Usize(4));
    let data_id = expr.add(Language::Access([data_id, access_axis_id]));

    let mut stride_list = Vec::default();
    stride_list.push(expr.add(Language::Usize(1)));
    stride_list.push(expr.add(Language::Usize(1)));
    stride_list.push(expr.add(Language::Usize(strides[0])));
    stride_list.push(expr.add(Language::Usize(strides[1])));
    let stride_shape_id = expr.add(Language::Shape(Box::from(stride_list.as_slice())));

    let in_channels = data_shape[1];

    let data_id = match groups as usize {
        1 => {
            // Create the (shape ...) representing the kernel shapes
            let usize_1_id = expr.add(Language::Usize(1));
            let usize_c_id = expr.add(Language::Usize(weights_shape[1]));
            let usize_kh_id = expr.add(Language::Usize(weights_shape[2]));
            let usize_kw_id = expr.add(Language::Usize(weights_shape[3]));
            let weights_shape_id = expr.add(Language::Shape(Box::new([
                usize_1_id,
                usize_c_id,
                usize_kh_id,
                usize_kw_id,
            ])));

            let data_id = expr.add(Language::AccessWindows([
                data_id,
                weights_shape_id,
                stride_shape_id,
            ]));
            // Result is [batch 1 new_h new_w] [1 in_channel kw kh]

            // Squeeze the 4th dimension so it matches kernel shapes
            let squeeze_axis_id = expr.add(Language::Usize(4));
            let data_id = expr.add(Language::AccessSqueeze([data_id, squeeze_axis_id]));
            // Squeeze extraneous 1st dimension
            let squeeze_axis_id = expr.add(Language::Usize(1));
            let data_id = expr.add(Language::AccessSqueeze([data_id, squeeze_axis_id]));
            let data_id = access(expr, data_id, 3);
            // Result is [batch new_h new_w] [in_channel kw kh]

            let access_axis_id = expr.add(Language::Usize(1));
            let weights_id = expr.add(Language::Access([weights_id, access_axis_id]));

            let data_id = expr.add(Language::AccessCartesianProduct([weights_id, data_id]));

            let compute_type_id = expr.add(Language::ComputeType(ComputeType::DotProduct));
            let data_id = expr.add(Language::Compute([compute_type_id, data_id]));

            let data_id = access_transpose(expr, data_id, &[1, 0, 2, 3]);

            data_id
        }
        // If groups = num input channels (ie in depthwise separable mobilenet convs)
        // TODO(@gussmith23) Layout assumption
        n if n == in_channels => {
            // Kernel size is the same for each group. Each
            // kernel's shape is (1,1,kH,kW) where the first 1
            // lines up with batch and the second lines up with
            // input channels. The fact that the kernel's
            // channel size is 1 is what makes this grouped with
            // groups=in_channels.
            // TODO(@gussmith23) Layout assumption.
            let mut list = Vec::default();
            list.push(expr.add(Language::Usize(1)));
            list.push(expr.add(Language::Usize(1)));
            for v in weights_shape[2..].iter() {
                list.push(expr.add(Language::Usize(*v as usize)));
            }
            let weights_shape_id = expr.add(Language::Shape(Box::from(list.as_slice())));

            let mut to_be_concatted = Vec::default();

            for channel_idx in 0..in_channels {
                // Get this group's input channel
                // TODO(@gussmith23) layout assumption
                let data_id = access_slice(
                    expr,
                    data_id,
                    1,
                    channel_idx.try_into().unwrap(),
                    (channel_idx + 1).try_into().unwrap(),
                );
                let data_id = expr.add(Language::AccessWindows([
                    data_id,
                    weights_shape_id,
                    stride_shape_id,
                ]));
                let data_id = access(expr, data_id, 4);
                // Result should be
                // [1 1 new_H new_W] [1 1 kernel_H kernel_W]

                // Get this group's kernel
                // TODO(@gussmith23) layout assumption
                let weights_id = access_slice(
                    expr,
                    weights_id,
                    0,
                    channel_idx.try_into().unwrap(),
                    (channel_idx + 1).try_into().unwrap(),
                );
                let weights_id = access(expr, weights_id, 0);

                let data_id = expr.add(Language::AccessCartesianProduct([weights_id, data_id]));
                // Results should be
                // [1 1 new_H new_W] [2 1 1 kernel_H kernel_W]

                let data_id = compute(expr, ComputeType::DotProduct, data_id);
                // Results should be
                // [1 1 new_H new_W]

                to_be_concatted.push(data_id);
            }

            let mut concatted_id = to_be_concatted[0];
            for to_be_concatted_id in to_be_concatted[1..].iter() {
                // TODO(@gussmith23) Layout assumption
                concatted_id = access_concatenate(expr, concatted_id, *to_be_concatted_id, 1);
            }

            concatted_id
        }
        _ => panic!("Groups not implemented for groups={}", groups),
    };

    // Transpose from NCHW to original layout
    match data_layout {
        "NCHW" => data_id,
        "NHWC" => access_transpose(expr, data_id, &[0, 2, 3, 1]),
        _ => unreachable!(),
    }
}

/// Create access shape literal
///
/// ```
/// use glenside::language::from_relay::access_shape;
/// use egg::RecExpr;
/// use glenside::language::Language;
///
/// let mut expr = RecExpr::default();
/// let id = access_shape(&mut expr, &[1,2,3], &[4,5,6]);
/// assert_eq!(expr.pretty(80), "(access-shape (shape 1 2 3) (shape 4 5 6))");
/// ```
pub fn access_shape(expr: &mut RecExpr<Language>, shape: &[usize], item_shape: &[usize]) -> Id {
    let mut shape_ids = Vec::default();
    for s in shape {
        shape_ids.push(expr.add(Language::Usize(*s)));
    }
    let mut item_shape_ids = Vec::default();
    for i in item_shape {
        item_shape_ids.push(expr.add(Language::Usize(*i)));
    }
    let shape_id = expr.add(Language::Shape(shape_ids.into_boxed_slice()));
    let item_shape_id = expr.add(Language::Shape(item_shape_ids.into_boxed_slice()));
    expr.add(Language::AccessShape([shape_id, item_shape_id]))
}

pub fn access_shape_with_shape(expr: &mut RecExpr<Language>, shape: &[usize], item_shape: &[usize]) -> (Id, Id) {
    let mut shape_ids = Vec::default();
    for s in shape {
        shape_ids.push(expr.add(Language::Usize(*s)));
    }
    let mut item_shape_ids = Vec::default();
    for i in item_shape {
        item_shape_ids.push(expr.add(Language::Usize(*i)));
    }
    let shape_id = expr.add(Language::Shape(shape_ids.into_boxed_slice()));
    let item_shape_id = expr.add(Language::Shape(item_shape_ids.into_boxed_slice()));
    (expr.add(Language::AccessShape([shape_id, item_shape_id])), shape_id)
}

/// Concatenate accesses
///
/// ```
/// use std::str::FromStr;
/// use glenside::language::from_relay::access_concatenate;
/// use egg::RecExpr;
/// use glenside::language::Language;
///
/// let mut expr = RecExpr::default();
/// let a_id = expr.add(Language::Symbol("a".to_string()));
/// let a_id = expr.add(Language::AccessTensor(a_id));
/// let b_id = expr.add(Language::Symbol("b".to_string()));
/// let b_id = expr.add(Language::AccessTensor(b_id));
/// let id = access_concatenate(&mut expr, a_id, b_id, 2);
/// assert_eq!(expr.pretty(80), "(access-concatenate (access-tensor a) (access-tensor b) 2)");
/// ```
pub fn access_concatenate(expr: &mut RecExpr<Language>, a_id: Id, b_id: Id, axis: usize) -> Id {
    let axis_id = expr.add(Language::Usize(axis));
    expr.add(Language::AccessConcatenate([a_id, b_id, axis_id]))
}

/// Slice an access
///
/// ```
/// use std::str::FromStr;
/// use glenside::language::from_relay::access_slice;
/// use egg::RecExpr;
///
/// let mut expr = RecExpr::from_str("(access-tensor a)").unwrap();
/// let id = access_slice(&mut expr, 1.into(), 1, 4, 8);
/// assert_eq!(expr.pretty(80), "(access-slice (access-tensor a) 1 4 8)");
/// ```
pub fn access_slice(
    expr: &mut RecExpr<Language>,
    id: Id,
    axis: usize,
    low: usize,
    high: usize,
) -> Id {
    let axis_id = expr.add(Language::Usize(axis));
    let low_id = expr.add(Language::Usize(low));
    let high_id = expr.add(Language::Usize(high));
    expr.add(Language::AccessSlice([id, axis_id, low_id, high_id]))
}

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
/// let id = access_pad(&mut expr, 1.into(), PadType::ZeroPadding, 2, 3, 4 );
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
/// let id = access(&mut expr, 1.into(), 2);
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
/// let id = access_insert_axis(&mut expr, 1.into(), 2);
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
/// let id = compute(&mut expr, ComputeType::ReLU, 3.into());
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
    assert!(
        tensor_type.dtype.clone() == "float32".parse().unwrap()
            || tensor_type.dtype.clone() == "int32".parse().unwrap(),
        "only supporting float32x1 and int32x1 at the moment"
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
///
/// If `simplify_batch_norm_for_inference_hack` is enabled, any batch norm found
/// in the Relay program will be converted to a RelayOperatorCall node in
/// Glenside. Otherwise, the function will panic.
///
/// TODO(@gussmith23) Get rid of this hack eventually
///
/// `use_opaque_operators_for` is a list of [`RelayOperator`]s which should be
/// replaced with opaque versions of their operators, instead of expanding them
/// into Glenside code. This is a hack; this allows us to mostly ignore
/// operators that we don't yet do rewrites over (e.g. batch norms, pooling,
/// softmax) and allows us to more easily generate somewhat good code for them.
pub fn from_relay(
    module: &IRModule,
    simplify_batch_norm_for_inference_hack: bool,
    use_opaque_operators_for: &Vec<RelayOperator>,
) -> (RecExpr<Language>, Vec<(String, Vec<usize>)>, Vec<(Id, Id)>) {
    let main = module
        .lookup(module.get_global_var("main").unwrap())
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
    let mut visited = HashSet::new();
    create_worklist(func.body.clone(), &mut worklist, &mut visited);
    let mut map = HashMap::new();
    let mut relay_op_equivs = Vec::new();
    for expr in worklist {
        let (glenside_id, opaque_call) = compile_expression(
            expr.clone(),
            &mut glenside_expr,
            |expr| {
                *map.get(&expr).unwrap_or_else(|| {
                    panic!("Not found:\n{}", tvm::ir::expr::as_text(expr.clone()))
                })
            },
            simplify_batch_norm_for_inference_hack,
            use_opaque_operators_for,
        );
        map.insert(
            expr.clone(),
            glenside_id,
        );
        if let Some(call_id) = opaque_call {
            relay_op_equivs.push((glenside_id, call_id));
        }
    }

    (glenside_expr, names_and_shapes, relay_op_equivs)
}

// fn to_opaque_relay_call(expr: Expr) -> Option<RecExpr<Language>> {
//     if let Ok(call) = expr.clone().downcast::<tvm::ir::relay::Call>() {
//         if let Ok(primitive_op) = call
//             .op
//             .clone()
//             .upcast::<tvm::ir::expr::BaseExpr>()
//             .downcast::<tvm::ir::op::Op>()
//         {
//             match primitive_op.name.as_str().unwrap() {
//                 "nn.dense" => {
//                     RelayOperatorCall()
//                 }
//             }
//         }
//     } else {
//         None
//     }
// }

/// Generates an ordered list of Relay expressions to compile.
///
/// Compiling large Relay expressions with naive recursion overflows the stack,
/// so we first recursively generate a worklist which we can then iterate over.
/// The main goal of the worklist is to make sure an expression comes *after*
/// its children in the worklist; otherwise, we can't compile the expression!
///
/// To prevent redundant recursion which made large expressions prohibitively slow,
/// we keep track of the already visited Expr with "visited", and return if we have already
/// traversed an expression.
fn create_worklist(relay_expr: Expr, worklist: &mut Vec<Expr>, visited: &mut HashSet<Expr>) {
    if visited.contains(&relay_expr) {
        return;
    } else {
        visited.insert(relay_expr.clone());
    }

    if let Ok(_var) = relay_expr.clone().downcast::<tvm::ir::relay::Var>() {
    } else if let Ok(_constant) = relay_expr.clone().downcast::<tvm::ir::relay::Constant>() {
    } else if let Ok(call) = relay_expr.clone().downcast::<tvm::ir::relay::Call>() {
        for i in 0..call.args.len() {
            // Recursively add children (and their dependencies) to the worklist
            create_worklist(
                call.args.get(i.try_into().unwrap()).unwrap(),
                worklist,
                visited,
            );
        }
    } else if let Ok(tuple_get_item) = relay_expr
        .clone()
        .downcast::<tvm::ir::relay::TupleGetItem>()
    {
        create_worklist(tuple_get_item.tuple.clone(), worklist, visited);
    } else if let Ok(tuple) = relay_expr.clone().downcast::<tvm::ir::relay::Tuple>() {
        for i in 0..tuple.fields.len() {
            create_worklist(
                tuple.fields.get(i.try_into().unwrap()).unwrap(),
                worklist,
                visited,
            );
        }
    } else {
        // NOTE: if you're hitting this TODO, it might be that you have not
        // actually implemented the TVM Rust bindings for the Relay construct
        // you care about! I.e. if I am expecting to find a batch norm, but
        // haven't implemented the bindings, then the downcast to Call won't
        // work!
        todo!("Not implemented: {:?}", tvm::ir::as_text(relay_expr))
    }
    // add current node to worklist
    worklist.push(relay_expr.clone());
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
/// If `simplify_batch_norm_for_inference_hack` is enabled, any batch norm found
/// in the Relay program will be converted to a BatchNormInference node in
/// Glenside. Otherwise, the function will panic.
///
/// `use_opaque_operators_for` is a list of [`RelayOperator`]s which should be
/// replaced with opaque versions of their operators, instead of expanding them
/// into Glenside code. This is a hack; this allows us to mostly ignore
/// operators that we don't yet do rewrites over (e.g. batch norms, pooling,
/// softmax) and allows us to more easily generate somewhat good code for them.
fn compile_expression(
    relay_expr: Expr,
    glenside_expr: &mut RecExpr<Language>,
    // TODO(@gussmith23) Do we need to pass the recexpr into this closure?
    get_compiled_expression: impl Fn(Expr) -> Id,
    simplify_batch_norm_for_inference_hack: bool,
    use_opaque_operators_for: &Vec<RelayOperator>,
) -> (Id, Option<Id>) {
    if let Ok(var) = relay_expr.clone().downcast::<tvm::ir::relay::Var>() {
        let symbol = Language::Symbol(var.name_hint().to_string());
        let symbol_id = glenside_expr.add(symbol.clone());
        (glenside_expr.add(Language::AccessTensor(symbol_id)), None)
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
        (access_literal_id, None)
    } else if let Ok(tuple_get_item) = relay_expr
        .clone()
        .downcast::<tvm::ir::relay::TupleGetItem>()
    {
        if let Ok(call) = tuple_get_item
            .tuple
            .clone()
            .downcast::<tvm::ir::relay::Call>()
        {
            // For batch norms, we use a hack where if
            // simplify_batch_norm_for_inference_hack is set,
            // we compile the Relay batch norm to a single output.

            // All other expressions are compiled using Language::TupleGetItem construct

            if simplify_batch_norm_for_inference_hack
                && call
                    .op
                    .clone()
                    .upcast::<tvm::ir::expr::BaseExpr>()
                    .downcast::<tvm::ir::op::Op>()
                    .unwrap()
                    .name
                    .as_str()
                    .unwrap()
                    == "nn.batch_norm"
                && tuple_get_item.index == 0
            {
                // special case: compile Relay batch norm to a single output
                return (get_compiled_expression(tuple_get_item.tuple.clone()), None);
            }
        }

        // common case: Relay TupleGetItem gets converted to Glenside TupleGetItem
        // handles if tuple is not a CallNode
        let data_id = get_compiled_expression(tuple_get_item.tuple.clone());
        let index_id = glenside_expr.add(Language::Usize(tuple_get_item.index as usize));
        (glenside_expr.add(Language::TupleGetItem([data_id, index_id])), None)
    } else if let Ok(tuple) = relay_expr.clone().downcast::<tvm::ir::relay::Tuple>() {
        let mut fields = Vec::new();

        for i in 0..tuple.fields.len() {
            fields.push(get_compiled_expression(
                tuple.fields.get(i as isize).unwrap(),
            ))
        }

        (glenside_expr.add(Language::ConstructTuple(Box::from(fields.as_slice()))), None)
    } else if let Ok(call) = relay_expr.clone().downcast::<tvm::ir::relay::Call>() {
        if let Ok(primitive_op) = call
            .op
            .clone()
            .upcast::<tvm::ir::expr::BaseExpr>()
            .downcast::<tvm::ir::op::Op>()
        {
            match primitive_op.name.as_str().unwrap() {
                "nn.batch_norm" => {
                    assert!(simplify_batch_norm_for_inference_hack);
                    assert!(
                        use_opaque_operators_for
                            .contains(&crate::language::RelayOperator::RelayBatchNormInference),
                        "non-opaque implementation of batch norm not implemented!"
                    );

                    assert_eq!(call.args.len(), 5);
                    let data_id = get_compiled_expression(call.args.get(0).unwrap());
                    let gamma_id = get_compiled_expression(call.args.get(1).unwrap());
                    let beta_id = get_compiled_expression(call.args.get(2).unwrap());
                    let moving_mean_id = get_compiled_expression(call.args.get(3).unwrap());
                    let moving_var_id = get_compiled_expression(call.args.get(4).unwrap());
                    let attrs = call
                        .attrs
                        .clone()
                        .downcast::<tvm::ir::relay::attrs::nn::BatchNormAttrs>()
                        .unwrap();

                    // Should be ___C or _C___ (e.g. NHWC, NCHW)
                    assert!(attrs.axis == 3 || attrs.axis == 1);

                    let axis_id =
                        glenside_expr.add(Language::Usize(attrs.axis.try_into().unwrap()));
                    let epsilon_id = glenside_expr
                        .add(Language::NotNanFloat64(NotNan::new(attrs.epsilon).unwrap()));

                    let batch_norm_op_id = glenside_expr.add(Language::RelayOperator(
                        crate::language::RelayOperator::RelayBatchNormInference,
                    ));

                    (glenside_expr.add(Language::RelayOperatorCall(
                        vec![
                            batch_norm_op_id,
                            data_id,
                            gamma_id,
                            beta_id,
                            moving_mean_id,
                            moving_var_id,
                            axis_id,
                            epsilon_id,
                        ]
                        .into_boxed_slice(),
                    )), None)
                }
                "nn.softmax" => {
                    assert_eq!(call.args.len(), 1);
                    let data_id = get_compiled_expression(call.args.get(0).unwrap());
                    let attrs = call
                        .attrs
                        .clone()
                        .downcast::<tvm::ir::relay::attrs::nn::SoftmaxAttrs>()
                        .unwrap();

                    let softmax_id = glenside_expr.add(Language::RelayOperator(
                        crate::language::RelayOperator::RelaySoftmax,
                    ));
                    let ndims = call
                        .args
                        .get(0)
                        .unwrap()
                        .checked_type
                        .clone()
                        .downcast::<TensorType>()
                        .unwrap()
                        .shape
                        .len();

                    let axis: i64 = if attrs.axis < 0 {
                        ndims + i64::from(attrs.axis)
                    } else {
                        attrs.axis.into()
                    };
                    assert!(axis >= 0 && i64::from(axis) < ndims);
                    let axis_id = glenside_expr.add(Language::Usize(axis.try_into().unwrap()));
                    let opaque_call_id = glenside_expr.add(Language::RelayOperatorCall(
                        vec![softmax_id, data_id, axis_id].into_boxed_slice(),
                    ));
                    if use_opaque_operators_for
                        .contains(&crate::language::RelayOperator::RelaySoftmax)
                    {
                        return (opaque_call_id, None);
                    }

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
                            (compute(glenside_expr, ComputeType::Softmax, data_id), Some(opaque_call_id))
                        }
                        other @ _ => todo!("Softmax with axis value {} not yet supported", other),
                    }
                }
                "nn.relu" => {
                    assert_eq!(call.args.len(), 1);
                    let data_id = get_compiled_expression(call.args.get(0).unwrap());
                    let relu_id = glenside_expr.add(Language::RelayOperator(
                        crate::language::RelayOperator::RelayReLU,
                    ));
                    let opaque_call_id = glenside_expr.add(Language::RelayOperatorCall(
                        vec![relu_id, data_id].into_boxed_slice(),
                    ));
                    if use_opaque_operators_for.contains(&crate::language::RelayOperator::RelayReLU)
                    {
                        return (opaque_call_id, None)
                    } else {
                        (compute(glenside_expr, ComputeType::ReLU, data_id), Some(opaque_call_id))
                    }
                }
                "nn.leaky_relu" => {
                    assert_eq!(call.args.len(), 1);
                    let attrs = call
                        .attrs
                        .clone()
                        .downcast::<tvm::ir::relay::attrs::nn::LeakyReluAttrs>()
                        .unwrap();
                    let data_id = get_compiled_expression(call.args.get(0).unwrap());
                    let alpha_id = glenside_expr
                        .add(Language::NotNanFloat64(NotNan::new(attrs.alpha).unwrap()));
                    let leaky_relu_id = glenside_expr.add(Language::RelayOperator(
                        crate::language::RelayOperator::RelayLeakyReLU,
                    ));
                    let opaque_call_id = glenside_expr.add(Language::RelayOperatorCall(
                        vec![leaky_relu_id, data_id, alpha_id].into_boxed_slice(),
                    ));
                    if use_opaque_operators_for
                        .contains(&crate::language::RelayOperator::RelayLeakyReLU)
                    {
                        return (opaque_call_id, None);
                    } else {
                        todo!();
                    }
                }
                "sqrt" | "negative" => {
                    assert_eq!(call.args.len(), 1);
                    let data_id = get_compiled_expression(call.args.get(0).unwrap());
                    (compute(
                        glenside_expr,
                        match primitive_op.name.as_str().unwrap() {
                            "nn.relu" => ComputeType::ReLU,
                            "sqrt" => ComputeType::Sqrt,
                            "negative" => ComputeType::Negative,
                            _ => unreachable!(),
                        },
                        data_id,
                    ), None)
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

                    let data_id = get_compiled_expression(call.args.get(0).unwrap());

                    let layout_id = match attrs.layout.as_str().unwrap() {
                        "NCHW" => glenside_expr.add(Language::RelayActivationLayout(
                            crate::language::RelayActivationLayout::NCHW,
                        )),
                        "NHWC" => glenside_expr.add(Language::RelayActivationLayout(
                            crate::language::RelayActivationLayout::NHWC,
                        )),
                        l @ _ => panic!("Unsupported layout: {}", l),
                    };
                    let pool_size_id = shape(
                        glenside_expr,
                        vec![
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
                    let strides_id = shape(
                        glenside_expr,
                        vec![
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
                    let padding_id = shape(
                        glenside_expr,
                        vec![
                            attrs
                                .padding
                                .get(0)
                                .unwrap()
                                .downcast::<IntImm>()
                                .unwrap()
                                .value as usize,
                            attrs
                                .padding
                                .get(1)
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
                            attrs
                                .padding
                                .get(3)
                                .unwrap()
                                .downcast::<IntImm>()
                                .unwrap()
                                .value as usize,
                        ],
                    );

                    let max_pool_id = glenside_expr.add(Language::RelayOperator(
                        crate::language::RelayOperator::RelayMaxPool2D,
                    ));

                    let opaque_operator_call = glenside_expr.add(Language::RelayOperatorCall(
                        vec![
                            max_pool_id,
                            data_id,
                            pool_size_id,
                            strides_id,
                            padding_id,
                            layout_id,
                        ]
                        .into_boxed_slice(),
                    ));
                    if use_opaque_operators_for
                        .contains(&crate::language::RelayOperator::RelayMaxPool2D)
                    {
                        return (opaque_operator_call, None);
                    }

                    match attrs.layout.as_str().unwrap() {
                        "NCHW" => {
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

                            (data_id, Some(opaque_operator_call))
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

                    let data_id = get_compiled_expression(call.args.get(0).unwrap());

                    let global_avg_pool2d_operator_id =
                        glenside_expr.add(Language::RelayOperator(
                            crate::language::RelayOperator::RelayGlobalAvgPool2D,
                        ));
                    let layout_id = match attrs.layout.as_str().unwrap() {
                        "NCHW" => glenside_expr.add(Language::RelayActivationLayout(
                            crate::language::RelayActivationLayout::NCHW,
                        )),
                        "NHWC" => glenside_expr.add(Language::RelayActivationLayout(
                            crate::language::RelayActivationLayout::NHWC,
                        )),
                        l @ _ => panic!("Unsupported layout: {}", l),
                    };
                    let opaque_operator_call = glenside_expr.add(Language::RelayOperatorCall(
                        vec![global_avg_pool2d_operator_id, data_id, layout_id]
                            .into_boxed_slice(),
                    ));
                    if use_opaque_operators_for
                        .contains(&crate::language::RelayOperator::RelayGlobalAvgPool2D)
                    {
                        return (opaque_operator_call, None);
                    }

                    match attrs.layout.as_str().unwrap() {
                        "NCHW" => {
                            let data_id = access(glenside_expr, data_id, 2);
                            let data_id = compute(glenside_expr, ComputeType::ReduceMean, data_id);
                            let data_id = access_insert_axis(glenside_expr, data_id, 2);
                            let data_id = access_insert_axis(glenside_expr, data_id, 3);
                            let data_id = access(glenside_expr, data_id, 2);
                            (data_id, Some(opaque_operator_call))
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

                    (data_id, None)
                }
                "nn.dense" => {
                    // let attrs = call
                    //     .attrs
                    //     .clone()
                    //     .downcast::<tvm::ir::relay::attrs::nn::DenseAttrs>()
                    //     .unwrap();
                    assert_eq!(call.args.len(), 2);
                    // assert_eq!(
                    //     attrs.out_dtype,
                    //     // This datatype seems to indicate "null"?
                    //     DataType::new(3, 0, 0),
                    //     "Changing out_dtype not yet supported"
                    // );
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

                    let dense_op_id = glenside_expr.add(Language::RelayOperator(crate::language::RelayOperator::RelayDense));
                    let opaque_operator_call = glenside_expr.add(Language::RelayOperatorCall(vec![dense_op_id, data_id, weights_id].into_boxed_slice()));

                    let data_id = access(glenside_expr, data_id, 1);
                    let weights_id = access(glenside_expr, weights_id, 1);

                    let data_id =
                        glenside_expr.add(Language::AccessCartesianProduct([data_id, weights_id]));
                    (compute(glenside_expr, ComputeType::DotProduct, data_id), Some(opaque_operator_call))
                }
                "add" | "multiply" | "divide" | "maximum" | "minimum" => {
                    assert_eq!(call.args.len(), 2);
                    let mut a_id = get_compiled_expression(call.args.get(0).unwrap());
                    let mut a_shape =
                        shape_from_type(call.args.get(0).unwrap().checked_type.clone());
                    let mut b_id = get_compiled_expression(call.args.get(1).unwrap());
                    let mut b_shape =
                        shape_from_type(call.args.get(1).unwrap().checked_type.clone());

                    if primitive_op.name.as_str().unwrap() == "add"
                        && use_opaque_operators_for
                            .contains(&crate::language::RelayOperator::RelayAdd)
                    {
                        let add_operator_id = glenside_expr.add(Language::RelayOperator(
                            crate::language::RelayOperator::RelayAdd,
                        ));
                        return (glenside_expr.add(Language::RelayOperatorCall(
                            vec![add_operator_id, a_id, b_id].into_boxed_slice(),
                        )), None);
                    }
                    if primitive_op.name.as_str().unwrap() == "maximum"
                        && use_opaque_operators_for
                            .contains(&crate::language::RelayOperator::RelayMaximum)
                    {
                        let add_operator_id = glenside_expr.add(Language::RelayOperator(
                            crate::language::RelayOperator::RelayMaximum,
                        ));
                        return (glenside_expr.add(Language::RelayOperatorCall(
                            vec![add_operator_id, a_id, b_id].into_boxed_slice(),
                        )), None);
                    }
                    if primitive_op.name.as_str().unwrap() == "minimum"
                        && use_opaque_operators_for
                            .contains(&crate::language::RelayOperator::RelayMinimum)
                    {
                        let add_operator_id = glenside_expr.add(Language::RelayOperator(
                            crate::language::RelayOperator::RelayMinimum,
                        ));
                        return (glenside_expr.add(Language::RelayOperatorCall(
                            vec![add_operator_id, a_id, b_id].into_boxed_slice(),
                        )), None);
                    }

                    while a_shape.len() < b_shape.len() {
                        a_id = access_insert_axis(glenside_expr, a_id, 0);
                        a_shape.insert(0, 1);
                    }

                    while b_shape.len() < a_shape.len() {
                        b_id = access_insert_axis(glenside_expr, b_id, 0);
                        b_shape.insert(0, 1);
                    }

                    assert_eq!(a_shape.len(), b_shape.len());

                    assert!(a_shape.iter().zip(b_shape.iter()).map(|(a, b)| a <= b).all(|v| v) ||
                            a_shape.iter().zip(b_shape.iter()).map(|(a, b)| a >= b).all(|v| v),
                            "Can only handle simple broadcasts; all dims of a must be <= all dims of b (or vice-versa)");
                    if a_shape
                        .iter()
                        .zip(b_shape.iter())
                        .map(|(a, b)| a < b)
                        .any(|v| v)
                    {
                        let access_shape_id = access_shape(glenside_expr, &b_shape, &[]);
                        a_id =
                            glenside_expr.add(Language::AccessBroadcast([a_id, access_shape_id]));
                    } else if a_shape
                        .iter()
                        .zip(b_shape.iter())
                        .map(|(a, b)| a > b)
                        .any(|v| v)
                    {
                        let access_shape_id = access_shape(glenside_expr, &a_shape, &[]);
                        b_id =
                            glenside_expr.add(Language::AccessBroadcast([b_id, access_shape_id]));
                    }

                    let pair_id = access_pair(glenside_expr, a_id, b_id, 0);

                    match primitive_op.name.as_str().unwrap() {
                        "add" => {
                            let add_operator_id = glenside_expr.add(Language::RelayOperator(
                                crate::language::RelayOperator::RelayAdd,
                            ));
                            let operator_call = glenside_expr.add(Language::RelayOperatorCall(
                                vec![add_operator_id, a_id, b_id].into_boxed_slice(),
                            ));
                            (compute(glenside_expr, ComputeType::ElementwiseAdd, pair_id), Some(operator_call))
                        },
                        // TODO(mike): add operator support for these following
                        "multiply" => {
                            let mult_operator_id = glenside_expr.add(Language::RelayOperator(
                                crate::language::RelayOperator::RelayMultiply,
                            ));
                            let operator_call = glenside_expr.add(Language::RelayOperatorCall(
                                vec![mult_operator_id, a_id, b_id].into_boxed_slice(),
                            ));
                            (compute(glenside_expr, ComputeType::ElementwiseMul, pair_id), Some(operator_call))
                        }
                        "divide" => (compute(glenside_expr, ComputeType::ElementwiseDiv, pair_id), None),
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
                    let batch_flatten_operator_id = glenside_expr.add(Language::RelayOperator(
                        crate::language::RelayOperator::RelayBatchFlatten,
                    ));
                    let opaque_operator_call = glenside_expr.add(Language::RelayOperatorCall(
                        vec![batch_flatten_operator_id, data_id].into_boxed_slice(),
                    ));

                    if use_opaque_operators_for
                        .contains(&crate::language::RelayOperator::RelayBatchFlatten)
                    {
                        return (opaque_operator_call, None);
                    }

                    let data_id = access(glenside_expr, data_id, 1);
                    (glenside_expr.add(Language::AccessFlatten(data_id)), Some(opaque_operator_call))
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

                    // Get the axis valaue. If axis is negative, access from the
                    // back of the shape.
                    let axis = if attrs.axis >= 0 {
                        attrs.axis as i64
                    } else {
                        (call
                            .args
                            .get(0)
                            .unwrap()
                            .checked_type
                            .clone()
                            .downcast::<TensorType>()
                            .unwrap()
                            .shape
                            .len() as i64)
                            + attrs.axis as i64
                    };
                    assert!(axis >= 0);

                    let bias_add_operator_id = glenside_expr.add(Language::RelayOperator(
                        crate::language::RelayOperator::RelayBiasAdd,
                    ));
                    let axis_id = glenside_expr.add(Language::Usize(axis.try_into().unwrap()));
                    let opaque_operator_call = glenside_expr.add(Language::RelayOperatorCall(
                        vec![bias_add_operator_id, data_id, bias_id, axis_id]
                            .into_boxed_slice(),
                    ));

                    if use_opaque_operators_for
                        .contains(&crate::language::RelayOperator::RelayBiasAdd)
                    {
                        return (opaque_operator_call, None);
                    }

                    // Insert axes before
                    for _ in 0..axis {
                        let zero_id = glenside_expr.add(Language::Usize(0));
                        bias_id = glenside_expr.add(Language::AccessInsertAxis([bias_id, zero_id]));
                    }

                    // Insert axes after
                    for axis in (axis + 1) as i64
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

                    let data_shape =
                        shape_from_type(call.args.get(0).unwrap().checked_type.clone());
                    let access_shape_id = access_shape(glenside_expr, &data_shape, &[]);
                    let bias_id =
                        glenside_expr.add(Language::AccessBroadcast([bias_id, access_shape_id]));

                    let data_id = access_pair(glenside_expr, data_id, bias_id, 0);
                    let data_id = compute(glenside_expr, ComputeType::ElementwiseAdd, data_id);

                    (data_id, Some(opaque_operator_call))
                }
                "nn.conv1d" => {
                    assert_eq!(call.args.len(), 2);
                    let attrs = call
                        .attrs
                        .clone()
                        .downcast::<tvm::ir::relay::attrs::nn::Conv1DAttrs>()
                        .unwrap();
                    let data_id = get_compiled_expression(call.args.get(0).unwrap());
                    let data_shape = shape_from_type(call.args.get(0).unwrap().checked_type.clone());
                    let weights_id = get_compiled_expression(call.args.get(1).unwrap());
                    let weights_shape = shape_from_type(call.args.get(1).unwrap().checked_type.clone());
                    assert_eq!(attrs.padding.len(), 2);
                    assert_eq!(attrs.dilation.len(), 1);


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
                    let op_id = glenside_expr.add(Language::RelayOperator(crate::language::RelayOperator::RelayConv1D));
                    let conv1d_opcall = glenside_expr.add(Language::RelayOperatorCall(
                        vec![op_id, data_id, weights_id].into_boxed_slice()
                    ));
                    //Might need some more asserts for dilation, output layout (see Conv2d)
                    // assert_eq!(attrs.out_layout, "");
                    // println!("Checked layout");
                    // println!("{:?}", attrs.out_dtype);
                    // assert_eq!(
                    //     attrs.out_dtype,
                    //     // TODO(@gussmith23) How to actually constrain this?
                    //     tvm::DataType::new(3, 0, 0)
                    // );
                    // println!("Attr checked");
                    (conv1d(
                        glenside_expr,
                        data_id,
                        &data_shape,
                        weights_id,
                        &weights_shape,
                        &[
                            attrs
                                .strides
                                .get(0)
                                .unwrap()
                                .downcast::<IntImm>()
                                .unwrap()
                                .value as usize,
                        ],
                        &[
                            attrs
                                .padding
                                .get(0)
                                .unwrap()
                                .downcast::<IntImm>()
                                .unwrap()
                                .value as usize,
                            attrs
                                .padding
                                .get(1)
                                .unwrap()
                                .downcast::<IntImm>()
                                .unwrap()
                                .value as usize,
                        ],
                        &[
                            attrs
                                .dilation
                                .get(0)
                                .unwrap()
                                .downcast::<IntImm>()
                                .unwrap()
                                .value as usize
                        ],
                        attrs.groups.try_into().unwrap(),
                        "NCW",
                        "OIW",
                        "",
                    ), Some(conv1d_opcall))
                    
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
                    assert_eq!(attrs.out_layout, "");
                    assert_eq!(
                        attrs.out_dtype,
                        // TODO(@gussmith23) How to actually constrain this?
                        tvm::DataType::new(3, 0, 0)
                    );

                    (conv2d(
                        glenside_expr,
                        data_id,
                        &data_shape,
                        weights_id,
                        &weights_shape,
                        &[
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
                        &[
                            attrs
                                .padding
                                .get(0)
                                .unwrap()
                                .downcast::<IntImm>()
                                .unwrap()
                                .value as usize,
                            attrs
                                .padding
                                .get(1)
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
                            attrs
                                .padding
                                .get(3)
                                .unwrap()
                                .downcast::<IntImm>()
                                .unwrap()
                                .value as usize,
                        ],
                        &[
                            attrs
                                .dilation
                                .get(0)
                                .unwrap()
                                .downcast::<IntImm>()
                                .unwrap()
                                .value as usize,
                            attrs
                                .dilation
                                .get(1)
                                .unwrap()
                                .downcast::<IntImm>()
                                .unwrap()
                                .value as usize,
                        ],
                        attrs.groups.try_into().unwrap(),
                        attrs.data_layout.as_str().unwrap(),
                        attrs.kernel_layout.as_str().unwrap(),
                        attrs.out_layout.as_str().unwrap(),
                    ), None)
                }
                "nn.upsampling" => {
                    assert_eq!(call.args.len(), 1);
                    let attrs = call
                        .attrs
                        .clone()
                        .downcast::<tvm::ir::relay::attrs::nn::UpSamplingAttrs>()
                        .unwrap();
                    assert_eq!(attrs.layout.as_str().unwrap(), "NCHW");
                    let data_id = get_compiled_expression(call.args.get(0).unwrap());
                    let scale_h_id = glenside_expr
                        .add(Language::NotNanFloat64(NotNan::new(attrs.scale_h).unwrap()));
                    let scale_w_id = glenside_expr
                        .add(Language::NotNanFloat64(NotNan::new(attrs.scale_w).unwrap()));
                    let layout_id = match attrs.layout.as_str().unwrap() {
                        "NCHW" => glenside_expr.add(Language::RelayActivationLayout(
                            crate::language::RelayActivationLayout::NCHW,
                        )),
                        "NHWC" => glenside_expr.add(Language::RelayActivationLayout(
                            crate::language::RelayActivationLayout::NHWC,
                        )),
                        l @ _ => panic!("Unsupported layout: {}", l),
                    };
                    if use_opaque_operators_for
                        .contains(&crate::language::RelayOperator::RelayUpSampling)
                    {
                        let upsampling_id = glenside_expr.add(Language::RelayOperator(
                            crate::language::RelayOperator::RelayUpSampling,
                        ));
                        (glenside_expr.add(Language::RelayOperatorCall(
                            vec![upsampling_id, data_id, scale_h_id, scale_w_id, layout_id]
                                .into_boxed_slice(),
                        )), None)
                    } else {
                        todo!()
                    }
                }
                // "nn.conv1d" => {
                //     let op_id = glenside_expr.add(Language::RelayOperator(crate::language::RelayOperator::RelayConv1D));
                //     let data_id = get_compiled_expression(call.args.get(0).unwrap());
                //     let weight_id = get_compiled_expression(call.args.get(1).unwrap());
                //     let conv1d_opcall = glenside_expr.add(Language::RelayOperatorCall(
                //         vec![op_id, data_id, weight_id].into_boxed_slice()
                //     ));
                //     (conv1d_opcall, None)
                // }
                "erf" => {
                    let op_id = glenside_expr.add(Language::RelayOperator(crate::language::RelayOperator::RelayErf));
                    let data_id = get_compiled_expression(call.args.get(0).unwrap());
                    let opaque_operator_call = glenside_expr.add(Language::RelayOperatorCall(
                        vec![op_id, data_id].into_boxed_slice()
                    ));
                    (opaque_operator_call, None)
                }
                "mean" => {
                    let op_id = glenside_expr.add(Language::RelayOperator(crate::language::RelayOperator::RelayMean));
                    let data_id = get_compiled_expression(call.args.get(0).unwrap());
                    let attrs = call
                        .attrs
                        .clone()
                        .downcast::<tvm::ir::relay::attrs::reduce::ReduceAttrs>()
                        .unwrap();
                    // TODO(mike): support reducing on multiple axis?
                    assert_eq!(attrs.axis.len(), 1);
                    let axis_id;
                    if let Ok(axis) = attrs.axis.get(0) {
                        axis_id = glenside_expr.add(Language::Usize(axis.clone()
                                                                        .downcast::<tvm::ir::tir::IntImm>()
                                                                        .unwrap()
                                                                        .value as usize));
                    } else {
                        axis_id = glenside_expr.add(Language::Usize(0 as usize));
                    }
                    let opaque_operator_call = glenside_expr.add(Language::RelayOperatorCall(
                        vec![op_id, data_id, axis_id].into_boxed_slice()
                    ));
                    (opaque_operator_call, None)
                }
                "concatenate" => {
                    assert_eq!(call.args.len(), 1);
                    let attrs = call
                        .attrs
                        .clone()
                        .downcast::<tvm::ir::relay::attrs::transform::ConcatenateAttrs>()
                        .unwrap();

                    assert!(attrs.axis >= 0);

                    let tuple = call
                        .args
                        .get(0)
                        .unwrap()
                        .clone()
                        .downcast::<tvm::ir::relay::Tuple>()
                        .ok()
                        .unwrap();
                    let mut concatted_id = get_compiled_expression(tuple.fields.get(0).unwrap());
                    for i in 1..tuple.fields.len() {
                        // TODO(@gussmith23) Layout assumption
                        let to_be_concatted_id = get_compiled_expression(
                            tuple.fields.get(i.try_into().ok().unwrap()).unwrap(),
                        );
                        concatted_id = access_concatenate(
                            glenside_expr,
                            concatted_id,
                            to_be_concatted_id,
                            attrs.axis as usize,
                        );
                    }

                    (concatted_id, None)
                }
                "reshape" => {
                    assert_eq!(call.args.len(), 1);
                    let data_id = get_compiled_expression(call.args.get(0).unwrap());

                    // use relay type information to calculate new shape instead of using attrs
                    let new_shape =
                        shape_from_type(call.clone().upcast::<Expr>().checked_type.clone());
                    let (new_shape_id, shape_id) = access_shape_with_shape(glenside_expr, &new_shape, &[]);

                    let reshape_op = glenside_expr.add(Language::RelayOperator(crate::language::RelayOperator::RelayReshape));
                    let opaque_operator_call = glenside_expr.add(Language::RelayOperatorCall(vec![reshape_op, data_id, shape_id].into_boxed_slice()));

                    (glenside_expr.add(Language::AccessReshape([data_id, new_shape_id])), Some(opaque_operator_call))
                }
                "split" => {
                    assert_eq!(call.args.len(), 1);
                    let attrs = call
                        .attrs
                        .clone()
                        .downcast::<tvm::ir::relay::attrs::transform::SplitAttrs>()
                        .unwrap();
                    let data_id = get_compiled_expression(call.args.get(0).unwrap());

                    let axis = attrs.axis;
                    assert!(axis >= 0);
                    let axis_id = glenside_expr.add(Language::Usize(axis.try_into().unwrap()));

                    let indices_or_sections = &attrs
                        .indices_or_sections
                        .clone()
                        .downcast::<tvm::runtime::array::Array<tvm::runtime::object::ObjectRef>>()
                        .unwrap();
                    // assume for yolov3
                    assert_eq!(indices_or_sections.len(), 2);

                    let shape = shape_from_type(call.args.get(0).unwrap().checked_type.clone());

                    let mut ids = Vec::default();
                    let zero_id = glenside_expr.add(Language::Usize(0));
                    let zero_index_id = glenside_expr.add(Language::Usize(
                        indices_or_sections
                            .get(0)
                            .unwrap()
                            .downcast::<IntImm>()
                            .unwrap()
                            .value as usize,
                    ));
                    ids.push(glenside_expr.add(Language::AccessSlice([
                        data_id,
                        axis_id,
                        zero_id,
                        zero_index_id,
                    ])));

                    for i in 0..indices_or_sections.len() - 1 {
                        let left = glenside_expr.add(Language::Usize(
                            indices_or_sections
                                .get(i.try_into().ok().unwrap())
                                .unwrap()
                                .downcast::<IntImm>()
                                .unwrap()
                                .value as usize,
                        ));
                        let right = glenside_expr.add(Language::Usize(
                            indices_or_sections
                                .get((i + 1).try_into().ok().unwrap())
                                .unwrap()
                                .downcast::<IntImm>()
                                .unwrap()
                                .value as usize,
                        ));
                        ids.push(
                            glenside_expr
                                .add(Language::AccessSlice([data_id, axis_id, left, right])),
                        );
                    }

                    let last_index_id = glenside_expr.add(Language::Usize(
                        indices_or_sections
                            .get((indices_or_sections.len() - 1).try_into().ok().unwrap())
                            .unwrap()
                            .downcast::<IntImm>()
                            .unwrap()
                            .value as usize,
                    ));
                    let last_id = glenside_expr.add(Language::Usize(shape[axis as usize]));
                    ids.push(glenside_expr.add(Language::AccessSlice([
                        data_id,
                        axis_id,
                        last_index_id,
                        last_id,
                    ])));

                    (glenside_expr.add(Language::ConstructTuple(Box::from(ids.as_slice()))), None)
                }
                "sigmoid" => {
                    assert_eq!(call.args.len(), 1);
                    let data_id = get_compiled_expression(call.args.get(0).unwrap());

                    if use_opaque_operators_for
                        .contains(&crate::language::RelayOperator::RelaySigmoid)
                    {
                        let sigmoid_id = glenside_expr.add(Language::RelayOperator(
                            crate::language::RelayOperator::RelaySigmoid,
                        ));
                        (glenside_expr.add(Language::RelayOperatorCall(
                            vec![sigmoid_id, data_id].into_boxed_slice(),
                        )), None)
                    } else {
                        todo!()
                    }
                }
                "transpose" => {
                    assert_eq!(call.args.len(), 1);
                    let data_id = get_compiled_expression(call.args.get(0).unwrap());

                    let attrs = call
                        .attrs
                        .clone()
                        .downcast::<tvm::ir::relay::attrs::transform::TransposeAttrs>()
                        .unwrap();
                    let transpose_list = attrs
                        .axes
                        .clone()
                        .into_iter()
                        .map(|x| x.downcast::<IntImm>().unwrap().value as usize)
                        .collect::<Vec<usize>>();
                    (access_transpose(glenside_expr, data_id, &transpose_list), None)
                }
                "nn.avg_pool2d" => {
                    assert_eq!(call.args.len(), 1);
                    let data_id = get_compiled_expression(call.args.get(0).unwrap());

                    let attrs = call
                        .attrs
                        .clone()
                        .downcast::<tvm::ir::relay::attrs::nn::AvgPool2DAttrs>()
                        .unwrap();

                    // assume for efficientnet
                    assert_eq!(attrs.ceil_mode, false);
                    assert_eq!(attrs.count_include_pad, false);

                    assert_eq!(attrs.pool_size.len(), 2);
                    assert_eq!(attrs.padding.len(), 4);
                    assert_eq!(attrs.strides.len(), 2);

                    let pool_size_id = shape(
                        glenside_expr,
                        vec![
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
                    let padding_id = shape(
                        glenside_expr,
                        vec![
                            attrs
                                .padding
                                .get(0)
                                .unwrap()
                                .downcast::<IntImm>()
                                .unwrap()
                                .value as usize,
                            attrs
                                .padding
                                .get(1)
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
                            attrs
                                .padding
                                .get(3)
                                .unwrap()
                                .downcast::<IntImm>()
                                .unwrap()
                                .value as usize,
                        ],
                    );
                    let strides_id = shape(
                        glenside_expr,
                        vec![
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
                    let layout_id = match attrs.layout.as_str().unwrap() {
                        "NCHW" => glenside_expr.add(Language::RelayActivationLayout(
                            crate::language::RelayActivationLayout::NCHW,
                        )),
                        "NHWC" => glenside_expr.add(Language::RelayActivationLayout(
                            crate::language::RelayActivationLayout::NHWC,
                        )),
                        l @ _ => panic!("Unsupported layout: {}", l),
                    };
                    if use_opaque_operators_for
                        .contains(&crate::language::RelayOperator::RelayAvgPool2D)
                    {
                        let avg_pool2d_id = glenside_expr.add(Language::RelayOperator(
                            crate::language::RelayOperator::RelayAvgPool2D,
                        ));
                        (glenside_expr.add(Language::RelayOperatorCall(
                            vec![
                                avg_pool2d_id,
                                data_id,
                                pool_size_id,
                                strides_id,
                                padding_id,
                                layout_id,
                            ]
                            .into_boxed_slice(),
                        )), None)
                    } else {
                        todo!()
                    }
                }
                "squeeze" => {
                    assert_eq!(call.args.len(), 1);
                    let mut data_id = get_compiled_expression(call.args.get(0).unwrap());

                    let attrs = call
                        .attrs
                        .clone()
                        .downcast::<tvm::ir::relay::attrs::transform::SqueezeAttrs>()
                        .unwrap();

                    // assume for efficientnet
                    assert_eq!(attrs.axis.len(), 2);
                    for i in (0..attrs.axis.len()).rev() {
                        let usize_id = glenside_expr.add(Language::Usize(
                            attrs
                                .axis
                                .get(i.try_into().ok().unwrap())
                                .unwrap()
                                .downcast::<IntImm>()
                                .unwrap()
                                .value as usize,
                        ));
                        data_id = glenside_expr.add(Language::AccessSqueeze([data_id, usize_id]));
                    }

                    (data_id, None)
                }
                op => {
                    println!("{} operator not implemented", op);
                    todo!()
                },
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

                let module = tvm::ir::module::IRModule::parse("", $relay_str).unwrap();

                let (expr, shapes_vec, _) = super::from_relay(&module, false, &vec![]);

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
                    cmd.arg("--npy_out_filepath");
                    cmd.arg(&output_filepath);
                    cmd.arg("--npy_arg_filepath");
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
  (access (access-broadcast (access-insert-axis (access-tensor x) 0) (access-shape (shape 1 1024 7 7) (shape))) 0)
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
        conv1d,
        1e-60,
        r#"
    #[version = "0.0.5"]
    def @main(%data: Tensor[(1, 3, 32), float32], %weights: Tensor[(8, 3, 3), float32]) -> Tensor[(1, 8, 19), float32] {
        nn.conv1d(%data, %weights, strides=[2], padding=[3, 4]) /* ty=Tensor[(1, 8, 19), float32] */
    }
"#,


        r#"
(access-transpose
 (compute dot-product
  (access-cartesian-product
   (access (access-tensor weights) 1)
   (access-squeeze
    (access-windows
     (access-pad
      (access-tensor data)
      zero-padding
      2 3 4
     )
     (shape 3 3)
     (shape 1 2)
    )
    1
   )
  )
 )
 (list 1 0 2)
)
"#
    );

    // The first part of a separable convolution, as seen in Mobilenet.
    test!(
        conv2d_depthwise_separable_stage1,
        1e-6,
        r#"
#[version = "0.0.5"]
def @main(%data: Tensor[(1, 3, 32, 32), float32], %weight: Tensor[(3, 1, 3, 3), float32]) -> Tensor[(1, 3, 38, 20), float32] {
  nn.conv2d(%data, %weight, strides=[1, 2], padding=[3, 4, 5, 6], groups=3)
}
"#,
        // TODO(@gussmith23) I'm being lazy here
        r#"
(access-concatenate ?a ?b ?c)
"#
    );

    // TODO(@gussmith23) Relay/TVM doesn't seem to like nhwc w/o hwoi
    // So we can't run a test like this til we support hwoi!
    //     test!(
    //         conv2d_depthwise_separable_stage1_nhwc,
    //         1e-6,
    //         r#"
    // #[version = "0.0.5"]
    // def @main(%data: Tensor[(1, 32, 32, 3), float32], %weight: Tensor[(3, 1, 3, 3), float32]) -> Tensor[(1, 38, 20, 3), float32] {
    //   nn.conv2d(%data, %weight, strides=[1, 2], padding=[3, 4, 5, 6], groups=3, data_layout="NHWC")
    // }
    // "#,
    //         // TODO(@gussmith23) I'm being lazy here
    //         r#"
    // (access-transpose
    //  (access-concatenate ?a ?b ?c)
    //  (list 0 2 3 1)
    // )
    // "#
    //     );

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
        conv2d_nhwc_hwio,
        1e-5,
        r#"
#[version = "0.0.5"]
def @main(%data: Tensor[(1, 32, 32, 3), float32], %weights: Tensor[(3, 3, 3, 8), float32]) -> Tensor[(1, 17, 12, 8), float32] {
  nn.conv2d(%data, %weights, strides=[2, 3], padding=[1, 2, 3, 4], data_layout="NHWC", kernel_layout="HWIO")
}
"#,
        r#"
(access-transpose
 (access-transpose
  (compute dot-product
   (access-cartesian-product
    (access
     (access-transpose (access-tensor weights) (list 3 2 0 1))
     1
    )
    (access
     (access-squeeze
      (access-squeeze
       (access-windows
        (access
         (access-pad
          (access-pad
           (access-transpose (access-tensor data) (list 0 3 1 2))
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
 (list 0 2 3 1)
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
  (access (access-broadcast (access-insert-axis (access-tensor x) 0) (access-shape (shape 1 1024 7 7) (shape))) 0)
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
  (access (access-broadcast (access-insert-axis (access-tensor x) 0) (access-shape (shape 1 1024 7 7) (shape))) 0)
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

    test!(
        dense,
        1e-5,
        r#"
#[version = "0.0.5"]
def @main(%data: Tensor[(16, 32), float32], %weights: Tensor[(64, 32), float32]) -> Tensor[(16, 64), float32] {
  nn.dense(%data, %weights, units=64) /* ty=Tensor[(16, 64), float32] */
}
"#,
        r#"
(compute dot-product
 (access-cartesian-product
  (access (access-tensor data) 1)
  (access (access-tensor weights) 1)
 )
)
"#
    );

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
    (access-shape (shape 3 3) (shape))
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
    (access-shape (shape 3 3) (shape))
   )
   0
  )
 )
)
"#
    );

    test!(
        bias_add_axis_neg_1,
        1e-60,
        r#"
#[version = "0.0.5"]
def @main(%x: Tensor[(3, 3), float32], %y: Tensor[(3), float32]) -> Tensor[(3, 3), float32] {
  nn.bias_add(%x, %y, axis=-1)
}
"#,
        r#"
(compute elementwise-add
 (access-pair
  (access (access-tensor x) 0)
  (access
   (access-broadcast
    (access-insert-axis (access-tensor y) 0)
    (access-shape (shape 3 3) (shape))
   )
   0
  )
 )
)
"#
    );

    test!(
        bias_add_axis_neg_2,
        1e-60,
        r#"
#[version = "0.0.5"]
def @main(%x: Tensor[(3, 3), float32], %y: Tensor[(3), float32]) -> Tensor[(3, 3), float32] {
  nn.bias_add(%x, %y, axis=-2)
}
"#,
        r#"
(compute elementwise-add
 (access-pair
  (access (access-tensor x) 0)
  (access
   (access-broadcast
    (access-insert-axis (access-tensor y) 1)
    (access-shape (shape 3 3) (shape))
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

    test!(
        concatenate,
        1e-7,
        r#"
#[version = "0.0.5"]
def @main(%x: Tensor[(1, 5, 1, 1), float32], %y: Tensor[(1, 3, 1, 1), float32]) {
    %0 = (%x, %y);
    concatenate(%0, axis=1)
}
"#,
        r#"
(access-concatenate (access-tensor x) (access-tensor y) 1)
"#
    );

    // TODO: Uncomment out this test when reshape is implemented for the interpreter
    // Note: the #[ignore] doesn't work because it has to go after #[test]
    // and moving #[test] out of the macro causes dead code warnings!
    //
    //
    //     #[ignore = "reshape not implemented for interpreter"]
    //     test!(
    //         reshape,
    //         1e-7,
    //         r#"
    // #[version = "0.0.5"]
    // def @main(%x: Tensor[(1, 255, 52, 52), float32]) {
    //     reshape(%x, newshape=[-1, 3, 85, 52, 52])
    // }
    // "#,
    //         r#"
    // (access-reshape (access-tensor x) (access-shape (shape 1 3 85 52 52) (shape )))
    // "#
    //     );

    // TODO: add tuple output support to test macro
    // and need someway to handle tuples in interpreter
    //
    //     test!(
    //         split,
    //         1e-7,
    //         r#"
    // #[version = "0.0.5"]
    // def @main(%x: Tensor[(1, 6, 1), float32]) {
    //     split(%x, indices_or_sections=[2, 4], axis=1)
    // }
    // "#,
    //         r#"
    // (construct-tuple
    //     (access-slice (access-tensor x) 1 0 2)
    //     (access-slice (access-tensor x) 1 2 4)
    //     (access-slice (access-tensor x) 1 4 6)
    // )
    // "#
    //     );

    test!(
        transpose,
        1e-7,
        r#"
#[version = "0.0.5"]
def @main(%x: Tensor[(3, 5), float32]) {
    transpose(%x, axes=[1, 0])
}
"#,
        r#"
(access-transpose (access-tensor x) (list 1 0))
        "#
    );

    test!(
        squeeze,
        1e-7,
        r#"
#[version = "0.0.5"]
def @main(%x: Tensor[(1, 100, 1, 1), float32]) {
    squeeze(%x, axis=[2, 3])
}
"#,
        r#"
(access-squeeze (access-squeeze (access-tensor x) 3) 2)
        "#
    );

    // TODO: uncomment these tests once
    // from_relay can parse these relay operators without using the opaque relay op hack
    // and when the interpreter supports the operations
    //     test!(
    //         leakyrelu,
    //         1e-7,
    //         r#"
    // #[version = "0.0.5"]
    // def @main(%x: Tensor[(1, 100, 1, 1), float32]) {
    //     nn.leaky_relu(%x, alpha=0.1f)
    // }
    // "#,
    //         r#"
    // (relay-operator-call relay-leaky-relu (access-tensor x) 0.1)
    //         "#
    //     );

    //     test!(
    //         maximum,
    //         1e-7,
    //         r#"
    // #[version = "0.0.5"]
    // def @main(%x: Tensor[(1, 100, 5, 1), float32], %y: Tensor[(1, 100), float32]) {
    //     maximum(%x, %y)
    // }
    // "#,
    //         r#"
    // (relay-operator-call relay-maximum (access-tensor x) (access-tensor y))
    //         "#
    //     );

    //     test!(
    //         minimum,
    //         1e-7,
    //         r#"
    // #[version = "0.0.5"]
    // def @main(%x: Tensor[(1, 100, 5, 1), float32], %y: Tensor[(1, 100), float32]) {
    //     minimum(%x, %y)
    // }
    // "#,
    //         r#"
    // (relay-operator-call relay-minimum (access-tensor x) (access-tensor y))
    //         "#
    //     );

    //     test!(
    //         upsampling,
    //         1e-7,
    //         r#"
    // #[version = "0.0.5"]
    // def @main(%data: Tensor[(1, 256, 13, 13), float32]) {
    //     nn.upsampling(%data, scale_h=2f, scale_w=2f)
    // }
    // "#,
    //         r#"
    // (relay-operator-call relay-upsampling (access-tensor data) 2 2 relay-activation-layout-nchw)
    //         "#
    //     );

    //     test!(
    //         sigmoid,
    //         1e-7,
    //         r#"
    // #[version = "0.0.5"]
    // def @main(%data: Tensor[(10, 10), float32]) {
    //     sigmoid(%data)
    // }
    // "#,
    //         r#"
    // (relay-operator-call relay-sigmoid (access-tensor data))
    //         "#
    //     );

    //     test!(
    //         avg_pool2d,
    //         1e-7,
    //         r#"
    // #[version = "0.0.5"]
    // def @main(%data: Tensor[(1, 1280, 7, 7), float32]) {
    //     nn.avg_pool2d(%data, pool_size=[7, 7], padding=[0, 0, 0, 0])
    // }
    // "#,
    //         r#"
    // (relay-operator-call relay-avg-pool2d (access-tensor data)
    // (shape 7 7) (shape 1 1) (shape 0 0 0 0)
    // relay-activation-layout-nchw)
    //         "#
    //     );
}
