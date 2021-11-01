use crate::language::PadType;

use super::super::codegen::generate_worklist_for_codegen;
use super::Language;
use super::MyAnalysis;
use egg::EGraph;
use egg::Id;
use itertools::Itertools;
use ndarray::Dimension;
use std::collections::HashMap;
use std::convert::TryFrom;
use tvm::ir::attrs::Attrs;
use tvm::ir::relay::*;
use tvm::ir::span::Span;
use tvm::ir::tir::IntImm;
use tvm::ir::ty::TensorType;
use tvm::ir::ty::Type;
use tvm::ir::PrimExpr;
use tvm::runtime::array::Array;
use tvm::runtime::IsObjectRef;
use tvm::runtime::ObjectRef;
use tvm::Device;
use tvm::NDArray;

pub fn to_relay(egraph: &EGraph<Language, MyAnalysis>, id: Id, dev: Device) -> Expr {
    let worklist = generate_worklist_for_codegen(egraph, id);
    let mut hashmap = HashMap::new();
    for id in worklist {
        to_relay_impl(egraph, id, dev, &mut hashmap);
    }
    return hashmap[&id].clone();
}

fn to_relay_impl(
    egraph: &EGraph<Language, MyAnalysis>,
    id: Id,
    dev: Device,
    hashmap: &mut HashMap<Id, Expr>,
) {
    assert!(!hashmap.contains_key(&id), "Id is already in hashmap!");
    match {
        assert_eq!(
            egraph[id].len(),
            1,
            "egraph should have a single enode per eclass"
        );
        &egraph[id].nodes[0]
    } {
        Language::Usize(v) => {
            hashmap.insert(
                id,
                Constant::new(
                    NDArray::from_rust_ndarray(
                        &ndarray12::arr0(u32::try_from(*v).unwrap()).into_dyn(),
                        dev,
                        // TODO(@gussmith23) hardcoded code
                        DataType::new(1, 32, 1),
                    )
                    .unwrap(),
                    Span::null(),
                )
                .upcast(),
            );
        }
        Language::MoveAxis(_) => todo!(),
        Language::CartesianProduct(_) => todo!(),
        Language::MapDotProduct(_) => todo!(),
        Language::Slice(_) => todo!(),
        Language::Concatenate(_) => todo!(),
        Language::ElementwiseAdd(_) => todo!(),
        Language::BsgSystolicArray(_) => todo!(),
        Language::SystolicArray(_) => todo!(),
        Language::SystolicArrayWithBlocking(_) => todo!(),
        Language::SystolicArrayConv2dNchwOihwWithBlocking(_) => todo!(),
        Language::SystolicArrayConv2dNhwcHwioWithBlocking(_) => todo!(),
        Language::SystolicArrayConv2dIm2colNchwOihwWithBlocking(_) => todo!(),
        Language::SystolicArrayConv2dIm2colNhwcHwioWithBlocking(_) => todo!(),
        Language::AccessWindows([access_id, filters_shape_id, stride_shape_id]) => {
            // Get necessary type information.
            let (access_dimensions, compute_dimensions) = match &egraph[*access_id].data {
                crate::language::MyAnalysisData::AccessPattern(a) => {
                    (a.shape.slice(), a.item_shape.slice())
                }
                _ => panic!(),
            };
            let filters_shape = match &egraph[*filters_shape_id].data {
                crate::language::MyAnalysisData::Shape(u) => u.shape.slice(),
                _ => panic!(),
            };
            let stride_shape = match &egraph[*stride_shape_id].data {
                crate::language::MyAnalysisData::Shape(u) => u.shape.slice(),
                _ => panic!(),
            };

            let make_strided_slice = tvm::Function::get("relay.op._make.strided_slice").unwrap();

            // We use this function to give us the number of windows which will
            // be formed in each dimension. We can then use the per-dimension
            // window width (aka filter shape) and per-dimension strides to
            // calculate the start and end of each window.
            let num_windows_per_dim = crate::language::access_windows_resulting_shape(
                &ndarray::IxDyn(compute_dimensions),
                &ndarray::IxDyn(filters_shape),
                &ndarray::IxDyn(stride_shape),
            );

            let mut exprs = ndarray::Array::from_elem(num_windows_per_dim.clone(), Expr::null());

            // The Expr to slice.
            let input_expr = hashmap[access_id].clone();

            let window_indices_iter = num_windows_per_dim
                .iter()
                .map(|i| 0..*i)
                .multi_cartesian_product();

            for window_indices in window_indices_iter {
                let window_begin_indices: Vec<_> = window_indices
                    .iter()
                    .enumerate()
                    .map(|(i, val)| val * stride_shape[i])
                    .collect();
                let window_end_indices: Vec<_> = window_begin_indices
                    .iter()
                    .enumerate()
                    .map(|(i, val)| val + filters_shape[i])
                    .collect();

                // Slice out each window. The compute dimensions are the
                // dimensions actually getting sliced into windows. For each
                // access dimension, on the other hand, we take the entire
                // dimension each time (hence the repeat(0) as the starting
                // index and the access dimension length as the ending value).
                let sliced = make_strided_slice
                    .invoke(vec![
                        input_expr.clone().into(),
                        tvm::runtime::array::Array::from_vec(
                            std::iter::repeat(&0)
                                .take(access_dimensions.len())
                                .chain(window_begin_indices.iter())
                                .map(|i| IntImm::from(i32::try_from(*i).unwrap()))
                                .collect(),
                        )
                        .unwrap()
                        .into(),
                        tvm::runtime::array::Array::from_vec(
                            access_dimensions
                                .iter()
                                .chain(window_end_indices.iter())
                                .map(|i| IntImm::from(i32::try_from(*i).unwrap()))
                                .collect(),
                        )
                        .unwrap()
                        .into(),
                        tvm::runtime::array::Array::from_vec(
                            std::iter::repeat(IntImm::from(1))
                                .take(access_dimensions.len() + compute_dimensions.len())
                                .collect(),
                        )
                        .unwrap()
                        .into(),
                        "end".into(),
                    ])
                    .unwrap();

                exprs[ndarray::IxDyn(&window_indices)] = Expr::try_from(sliced).unwrap();
            }

            let make_stack = tvm::Function::get("relay.op._make.stack").unwrap();
            let stack_axis = i32::try_from(access_dimensions.len()).unwrap();
            for _ in 0..compute_dimensions.len() {
                exprs = exprs.map_axis(ndarray::Axis(exprs.ndim() - 1), |vals| {
                    let tuple = Tuple::new(
                        tvm::runtime::array::Array::from_vec(vals.as_slice().unwrap().to_vec())
                            .unwrap(),
                        Span::null(),
                    );

                    Expr::try_from(
                        make_stack
                            .invoke(vec![tuple.into(), stack_axis.into()])
                            .unwrap(),
                    )
                    .unwrap()
                });
            }

            hashmap.insert(
                id,
                exprs
                    .into_dimensionality::<ndarray::Ix0>()
                    .unwrap()
                    .into_scalar(),
            );
        }
        Language::ShapeOf(_) => todo!(),
        Language::SliceShape(_) => todo!(),
        Language::ShapeInsertAxis(_) => todo!(),
        Language::ShapeRemoveAxis(_) => todo!(),
        Language::Access([child_id, _]) => {
            hashmap.insert(id, hashmap[child_id].clone());
        }
        Language::AccessTranspose([access_id, reordering_id]) => {
            let reordering = match &egraph[*reordering_id].data {
                crate::language::MyAnalysisData::List(s) => s.clone(),
                _ => panic!(),
            };

            let make_transpose = tvm::Function::get("relay.op._make.transpose").unwrap();
            let out = make_transpose
                .invoke(vec![
                    hashmap[access_id].clone().into(),
                    tvm::runtime::array::Array::from_vec(
                        reordering
                            .iter()
                            .map(|i| IntImm::from(i32::try_from(*i).unwrap()))
                            .collect(),
                    )
                    .unwrap()
                    .into(),
                ])
                .unwrap();

            hashmap.insert(id, Expr::try_from(out).unwrap());
        }
        Language::AccessCartesianProduct(_) => todo!(),
        Language::Compute(_) => todo!(),
        Language::AccessReshape([access_id, _]) | Language::AccessFlatten(access_id) => {
            let shape = match &egraph[id].data {
                crate::language::MyAnalysisData::AccessPattern(a) => a.as_vec(),
                _ => panic!(),
            };

            let make_reshape = tvm::Function::get("relay.op._make.reshape").unwrap();
            let out = make_reshape
                .invoke(vec![
                    hashmap[access_id].clone().into(),
                    tvm::runtime::array::Array::from_vec(
                        shape
                            .iter()
                            .map(|i| IntImm::from(i32::try_from(*i).unwrap()))
                            .collect(),
                    )
                    .unwrap()
                    .into(),
                ])
                .unwrap();

            hashmap.insert(id, Expr::try_from(out).unwrap());
        }
        Language::Shape(_) => (),
        Language::List(_) => (),
        Language::ConstructTuple(_) => todo!(),
        Language::TupleGetItem(_) => todo!(),
        Language::AccessShape(_) => (),
        Language::AccessSlice(_) => todo!(),
        Language::AccessConcatenate(_) => todo!(),
        Language::AccessPair(_) => todo!(),
        Language::AccessShiftRight(_) => todo!(),
        Language::AccessTensor(child_id) => {
            hashmap.insert(id, hashmap[child_id].clone());
        }
        Language::AccessPad(
            [access_id, pad_type_id, axis_id, before_pad_amt_id, after_pad_amt_id],
        ) => {
            let shape = match &egraph[*access_id].data {
                crate::language::MyAnalysisData::AccessPattern(a) => a.as_vec(),
                _ => panic!(),
            };
            match &egraph[*pad_type_id].data {
                crate::language::MyAnalysisData::PadType(PadType::ZeroPadding) => (),
                _ => todo!("expected zero padding"),
            };
            let axis = match &egraph[*axis_id].data {
                crate::language::MyAnalysisData::Usize(u) => *u,
                _ => panic!(),
            };
            let pad_before_amt = match &egraph[*before_pad_amt_id].data {
                crate::language::MyAnalysisData::Usize(u) => *u,
                _ => panic!(),
            };
            let pad_after_amt = match &egraph[*after_pad_amt_id].data {
                crate::language::MyAnalysisData::Usize(u) => *u,
                _ => panic!(),
            };

            let mut padding: Vec<_> = std::iter::repeat(vec![IntImm::from(0), IntImm::from(0)])
                .take(shape.len())
                .collect();
            padding[axis][0] = IntImm::from(i32::try_from(pad_before_amt).unwrap());
            padding[axis][1] = IntImm::from(i32::try_from(pad_after_amt).unwrap());
            let padding = tvm::runtime::array::Array::from_vec(
                padding
                    .drain(..)
                    .map(|v| tvm::runtime::array::Array::from_vec(v).unwrap())
                    .collect(),
            )
            .unwrap();

            // TODO(@gussmith23) datatype assumption
            let pad_value = Constant::new(
                NDArray::from_rust_ndarray(
                    &ndarray12::arr0(0).into_dyn(),
                    dev,
                    DataType::float32(),
                )
                .unwrap(),
                Span::null(),
            );

            let make_pad = tvm::Function::get("relay.op.nn._make.pad").unwrap();
            let ret = make_pad
                .invoke(vec![
                    hashmap[access_id].clone().into(),
                    padding.into(),
                    pad_value.into(),
                    tvm::runtime::String::from("constant").into(),
                ])
                .unwrap();

            hashmap.insert(id, Expr::try_from(ret).unwrap());
        }
        Language::AccessSqueeze([access_id, dim_id]) => {
            let dim = match &egraph[*dim_id].data {
                crate::language::MyAnalysisData::Usize(u) => *u,
                _ => panic!(),
            };

            let make_squeeze = tvm::Function::get("relay.op._make.squeeze").unwrap();
            let ret = make_squeeze
                .invoke(vec![
                    hashmap[access_id].clone().into(),
                    tvm::runtime::array::Array::from_vec(vec![IntImm::from(
                        i32::try_from(dim).unwrap(),
                    )])
                    .unwrap()
                    .into(),
                ])
                .unwrap();

            hashmap.insert(id, Expr::try_from(ret).unwrap());
        }
        Language::AccessInsertAxis(_) => todo!(),
        Language::AccessBroadcast(_) => todo!(),
        Language::AccessLiteral(_) => todo!(),
        Language::Literal(_) => todo!(),
        Language::RelayOperatorCall(_) => todo!(),
        Language::NotNanFloat64(_) => todo!(),
        Language::RelayOperator(_) => todo!(),
        Language::RelayActivationLayout(_) => todo!(),
        Language::RelayKernelLayout(_) => todo!(),
        Language::PadType(_) => (),
        Language::ComputeType(_) => todo!(),
        Language::Symbol(name) => {
            let shape = match &egraph[id].data {
                crate::language::MyAnalysisData::Shape(s) => s.shape.slice(),
                _ => panic!(),
            };

            // TODO(@gussmith23) datatype assumption
            let type_annotation = TensorType::static_sh(
                shape.iter().map(|i| i32::try_from(*i).unwrap()).collect(),
                DataType::float32(),
                Span::null(),
            )
            .upcast();

            hashmap.insert(
                id,
                Var::new(name.clone(), type_annotation, Span::null()).upcast(),
            );
        }
        Language::AcceleratorCall(ids) => {
            let shape = match &egraph[id].data {
                crate::language::MyAnalysisData::AccessPattern(a) => a.as_vec(),
                _ => panic!(),
            };
            let make_accelerator_call =
                tvm::Function::get("relay.op._make.accelerator_call").unwrap();

            let shape_tvm_array = tvm::runtime::array::Array::from_vec(
                shape
                    .iter()
                    .map(|i| IntImm::from(i32::try_from(*i).unwrap()))
                    .collect(),
            )
            .unwrap();

            let shape_prim_expr_tvm_array: Array<PrimExpr> = tvm::runtime::array::Array::from_vec(
                shape
                    .iter()
                    .map(|i| IntImm::from(i32::try_from(*i).unwrap()).upcast())
                    .collect(),
            )
            .unwrap();

            let body = Expr::try_from(
                make_accelerator_call
                    .invoke(vec!["todo".into(), shape_tvm_array.into()])
                    .unwrap(),
            )
            .unwrap();

            let inner_args = (0..ids.len() - 1)
                .map(|i| Var::new(format!("inner_arg_{}", i), Type::null(), Span::null()))
                .collect();

            let inner_func: Expr = Function::new(
                tvm::runtime::array::Array::from_vec(inner_args).unwrap(),
                body,
                TensorType::new(
                    shape_prim_expr_tvm_array.clone(),
                    DataType::float32(),
                    Span::null(),
                )
                .upcast(),
                tvm::runtime::array::Array::from_vec(vec![]).unwrap(),
            )
            .upcast();

            let composite_name = "composite_name";
            let base_func_with_attr = tvm::Function::get("ir.BaseFuncWithAttr").unwrap();
            let inner_func = Expr::try_from(
                base_func_with_attr
                    .invoke(vec![
                        inner_func.into(),
                        "Composite".into(),
                        tvm::runtime::String::from(composite_name)
                            .upcast::<ObjectRef>()
                            .into(),
                    ])
                    .unwrap(),
            )
            .unwrap();

            let outer_args: Vec<Var> = (0..ids.len() - 1)
                .map(|i| Var::new(format!("outer_arg_{}", i), Type::null(), Span::null()))
                .collect();

            let body: Expr = Call::new(
                inner_func.clone(),
                Array::from_vec(outer_args.clone()).unwrap().upcast(),
                Attrs::null(),
                tvm::runtime::array::Array::from_vec(vec![]).unwrap(),
                Span::null(),
            )
            .upcast();

            let outer_func: Expr = Function::new(
                tvm::runtime::array::Array::from_vec(outer_args.clone()).unwrap(),
                body,
                TensorType::new(shape_prim_expr_tvm_array, DataType::float32(), Span::null())
                    .upcast(),
                tvm::runtime::array::Array::from_vec(vec![]).unwrap(),
            )
            .upcast();

            let region_counter = 0;
            let compiler_name = "compiler_name";
            let outer_func = Expr::try_from(
                base_func_with_attr
                    .invoke(vec![
                        outer_func.into(),
                        "Primitive".into(),
                        IntImm::new(DataType::int(32, 1), 1).into(),
                    ])
                    .unwrap(),
            )
            .unwrap();
            let outer_func = Expr::try_from(
                base_func_with_attr
                    .invoke(vec![
                        Expr::try_from(outer_func).unwrap().into(),
                        "global_symbol".into(),
                        tvm::runtime::String::from(format!(
                            "{}_{}",
                            composite_name, region_counter
                        ))
                        .upcast::<ObjectRef>()
                        .into(),
                    ])
                    .unwrap(),
            )
            .unwrap();
            let outer_func = Expr::try_from(
                base_func_with_attr
                    .invoke(vec![
                        Expr::try_from(outer_func).unwrap().into(),
                        "Compiler".into(),
                        tvm::runtime::String::from(compiler_name)
                            .upcast::<ObjectRef>()
                            .into(),
                    ])
                    .unwrap(),
            )
            .unwrap();

            let body = Call::new(
                outer_func,
                Array::from_vec(
                    ids[1..ids.len() - 1]
                        .iter()
                        .map(|id| hashmap.get(id).unwrap().clone())
                        .collect(),
                )
                .unwrap(),
                Attrs::null(),
                tvm::runtime::array::Array::from_vec(vec![]).unwrap(),
                Span::null(),
            )
            .upcast();

            hashmap.insert(id, body);
        }
        Language::AcceleratorFunc(_) => (),
        Language::FlexASRMaxPool(_) => todo!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egg::RecExpr;
    use ndarray12::arr0;
    use ndarray12::s;
    use ndarray12::ArrayD;
    use ndarray12::Dimension;
    use std::str::FromStr;
    use tvm::{
        compiler::graph_rt::{compile_module, CompilerConfig},
        ir::IRModule,
        runtime::graph_rt::GraphRt,
    };

    #[test]
    fn usize() {
        let hashmap = HashMap::new();
        let glenside_expr = RecExpr::<Language>::from_str("23").unwrap();
        let mut egraph = EGraph::new(MyAnalysis {
            name_to_shape: hashmap,
        });

        let id = egraph.add_expr(&glenside_expr);

        let out = to_relay(&egraph, id, Device::cpu(0));

        let module =
            compile_module(CompilerConfig::default(), IRModule::from_expr(out).unwrap()).unwrap();

        let mut rt = GraphRt::from_module(module, Device::cpu(0)).unwrap();
        rt.run().unwrap();
        assert_eq!(
            ArrayD::<u32>::try_from(&rt.get_output(0).unwrap()).unwrap(),
            arr0(23).into_dyn()
        );
    }

    #[test]
    fn symbol() {
        let mut name_to_shape = HashMap::new();
        name_to_shape.insert("a".to_string(), vec![1, 2, 3]);

        let glenside_expr = RecExpr::<Language>::from_str("a").unwrap();
        let mut egraph = EGraph::new(MyAnalysis { name_to_shape });

        let id = egraph.add_expr(&glenside_expr);

        let out = to_relay(&egraph, id, Device::cpu(0));

        let module =
            compile_module(CompilerConfig::default(), IRModule::from_expr(out).unwrap()).unwrap();

        let mut rt = GraphRt::from_module(module, Device::cpu(0)).unwrap();
        let input =
            ArrayD::from_shape_fn(vec![1, 2, 3], |d| d.slice().iter().sum::<usize>() as f32);
        rt.set_input(
            "a",
            NDArray::from_rust_ndarray(&input, Device::cpu(0), DataType::float32()).unwrap(),
        )
        .unwrap();

        rt.run().unwrap();

        assert_eq!(
            ArrayD::<f32>::try_from(&rt.get_output(0).unwrap()).unwrap(),
            input
        );
    }

    #[test]
    fn conv1d() {
        let mut name_to_shape = HashMap::new();
        name_to_shape.insert("data".to_string(), vec![1, 3, 32]);
        name_to_shape.insert("weights".to_string(), vec![8, 3, 3]);
        let glenside_expr = RecExpr::<Language>::from_str(
            r#"(access-transpose
            (access-reshape
              (accelerator-call
                vta-dense
                (access-flatten (access (access-tensor weights) 1))
                (access-flatten
                  (access-squeeze
                    (access-windows
                      (access (access-pad (access-tensor data) zero-padding 2 3 4) 1)
                      (shape 3 3)
                      (shape 1 2))
                    1))
                (shape 8 19))
              (access-shape (shape 8 1 19) (shape)))
            (list 1 0 2))"#,
        )
        .unwrap();
        let mut egraph = EGraph::new(MyAnalysis { name_to_shape });

        let id = egraph.add_expr(&glenside_expr);

        let out = to_relay(&egraph, id, Device::cpu(0));

        let irmodule = IRModule::from_expr(out).unwrap();

        // TODO(@gussmith23) Use something other than a string check...
        assert_eq!(
            tvm::ir::expr::as_text(irmodule),
            r#"#[version = "0.0.5"]
def @main(%weights: Tensor[(8, 3, 3), float32], %data: Tensor[(1, 3, 32), float32]) {
  %1 = nn.pad(%data, 0f, pad_width=[[0, 0], [0, 0], [3, 4]]);
  %2 = strided_slice(%1, begin=[0, 0, 0], end=[1, 3, 3], strides=[1, 1, 1]);
  %3 = strided_slice(%1, begin=[0, 0, 2], end=[1, 3, 5], strides=[1, 1, 1]);
  %4 = strided_slice(%1, begin=[0, 0, 4], end=[1, 3, 7], strides=[1, 1, 1]);
  %5 = strided_slice(%1, begin=[0, 0, 6], end=[1, 3, 9], strides=[1, 1, 1]);
  %6 = strided_slice(%1, begin=[0, 0, 8], end=[1, 3, 11], strides=[1, 1, 1]);
  %7 = strided_slice(%1, begin=[0, 0, 10], end=[1, 3, 13], strides=[1, 1, 1]);
  %8 = strided_slice(%1, begin=[0, 0, 12], end=[1, 3, 15], strides=[1, 1, 1]);
  %9 = strided_slice(%1, begin=[0, 0, 14], end=[1, 3, 17], strides=[1, 1, 1]);
  %10 = strided_slice(%1, begin=[0, 0, 16], end=[1, 3, 19], strides=[1, 1, 1]);
  %11 = strided_slice(%1, begin=[0, 0, 18], end=[1, 3, 21], strides=[1, 1, 1]);
  %12 = strided_slice(%1, begin=[0, 0, 20], end=[1, 3, 23], strides=[1, 1, 1]);
  %13 = strided_slice(%1, begin=[0, 0, 22], end=[1, 3, 25], strides=[1, 1, 1]);
  %14 = strided_slice(%1, begin=[0, 0, 24], end=[1, 3, 27], strides=[1, 1, 1]);
  %15 = strided_slice(%1, begin=[0, 0, 26], end=[1, 3, 29], strides=[1, 1, 1]);
  %16 = strided_slice(%1, begin=[0, 0, 28], end=[1, 3, 31], strides=[1, 1, 1]);
  %17 = strided_slice(%1, begin=[0, 0, 30], end=[1, 3, 33], strides=[1, 1, 1]);
  %18 = strided_slice(%1, begin=[0, 0, 32], end=[1, 3, 35], strides=[1, 1, 1]);
  %19 = strided_slice(%1, begin=[0, 0, 34], end=[1, 3, 37], strides=[1, 1, 1]);
  %20 = strided_slice(%1, begin=[0, 0, 36], end=[1, 3, 39], strides=[1, 1, 1]);
  %21 = (%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20);
  %22 = stack(%21, axis=1);
  %23 = (%22,);
  %24 = stack(%23, axis=1);
  %25 = squeeze(%24, axis=[1]);
  %26 = reshape(%weights, newshape=[8, 9]);
  %27 = reshape(%25, newshape=[19, 9]);
  %28 = fn (%outer_arg_0, %outer_arg_1, %outer_arg_2, Primitive=1, global_symbol="composite_name_0", Compiler="compiler_name") -> Tensor[(8, 19), float32] {
    %0 = fn (%inner_arg_0, %inner_arg_1, %inner_arg_2, Composite="composite_name") -> Tensor[(8, 19), float32] {
      accelerator_call(meta[relay.attrs.AcceleratorCallAttrs][0])
    };
    %0(%outer_arg_0, %outer_arg_1, %outer_arg_2)
  };
  %29 = %28(%26, %27);
  %30 = reshape(%29, newshape=[8, 1, 19]);
  transpose(%30, axes=[1, 0, 2])
}

/* For debugging purposes the metadata section has been omitted.
 * If you would like to see the full metadata section you can set the 
 * option to `True` when invoking `astext`. 
 */"#
        );
    }

    #[test]
    fn access_tensor() {
        let mut name_to_shape = HashMap::new();
        name_to_shape.insert("a".to_string(), vec![1, 2, 3]);

        let glenside_expr = RecExpr::<Language>::from_str("(access-tensor a)").unwrap();
        let mut egraph = EGraph::new(MyAnalysis { name_to_shape });

        let id = egraph.add_expr(&glenside_expr);

        let out = to_relay(&egraph, id, Device::cpu(0));

        let module =
            compile_module(CompilerConfig::default(), IRModule::from_expr(out).unwrap()).unwrap();

        let mut rt = GraphRt::from_module(module, Device::cpu(0)).unwrap();
        let input =
            ArrayD::from_shape_fn(vec![1, 2, 3], |d| d.slice().iter().sum::<usize>() as f32);
        rt.set_input(
            "a",
            NDArray::from_rust_ndarray(&input, Device::cpu(0), DataType::float32()).unwrap(),
        )
        .unwrap();

        rt.run().unwrap();

        assert_eq!(
            ArrayD::<f32>::try_from(&rt.get_output(0).unwrap()).unwrap(),
            input
        );
    }

    #[test]
    fn access() {
        let mut name_to_shape = HashMap::new();
        name_to_shape.insert("a".to_string(), vec![1, 2, 3]);

        let glenside_expr = RecExpr::<Language>::from_str("(access (access-tensor a) 2)").unwrap();
        let mut egraph = EGraph::new(MyAnalysis { name_to_shape });

        let id = egraph.add_expr(&glenside_expr);

        let out = to_relay(&egraph, id, Device::cpu(0));

        let module =
            compile_module(CompilerConfig::default(), IRModule::from_expr(out).unwrap()).unwrap();

        let mut rt = GraphRt::from_module(module, Device::cpu(0)).unwrap();
        let input =
            ArrayD::from_shape_fn(vec![1, 2, 3], |d| d.slice().iter().sum::<usize>() as f32);
        rt.set_input(
            "a",
            NDArray::from_rust_ndarray(&input, Device::cpu(0), DataType::float32()).unwrap(),
        )
        .unwrap();

        rt.run().unwrap();

        assert_eq!(
            ArrayD::<f32>::try_from(&rt.get_output(0).unwrap()).unwrap(),
            input
        );
    }

    #[test]
    fn flatten() {
        let mut name_to_shape = HashMap::new();
        name_to_shape.insert("a".to_string(), vec![1, 2, 3]);

        let glenside_expr =
            RecExpr::<Language>::from_str("(access-flatten (access (access-tensor a) 1))").unwrap();

        let mut egraph = EGraph::new(MyAnalysis { name_to_shape });

        let id = egraph.add_expr(&glenside_expr);

        let out = to_relay(&egraph, id, Device::cpu(0));

        let module =
            compile_module(CompilerConfig::default(), IRModule::from_expr(out).unwrap()).unwrap();

        let mut rt = GraphRt::from_module(module, Device::cpu(0)).unwrap();
        let input =
            ArrayD::from_shape_fn(vec![1, 2, 3], |d| d.slice().iter().sum::<usize>() as f32);
        rt.set_input(
            "a",
            NDArray::from_rust_ndarray(&input, Device::cpu(0), DataType::float32()).unwrap(),
        )
        .unwrap();

        rt.run().unwrap();

        let expected = input.into_shape(vec![1, 6]).unwrap();

        assert_eq!(
            ArrayD::<f32>::try_from(&rt.get_output(0).unwrap()).unwrap(),
            expected
        );
    }

    #[test]
    fn pad() {
        let mut name_to_shape = HashMap::new();
        name_to_shape.insert("a".to_string(), vec![2, 3]);

        let glenside_expr = RecExpr::<Language>::from_str(
            "(access-pad (access (access-tensor a) 1) zero-padding 1 1 2)",
        )
        .unwrap();

        let mut egraph = EGraph::new(MyAnalysis { name_to_shape });

        let id = egraph.add_expr(&glenside_expr);

        let out = to_relay(&egraph, id, Device::cpu(0));

        let module =
            compile_module(CompilerConfig::default(), IRModule::from_expr(out).unwrap()).unwrap();

        let mut rt = GraphRt::from_module(module, Device::cpu(0)).unwrap();
        let input = ndarray12::arr2(&[[1., 2., 3.], [4., 5., 6.]]).into_dyn();
        rt.set_input(
            "a",
            NDArray::from_rust_ndarray(&input, Device::cpu(0), DataType::float32()).unwrap(),
        )
        .unwrap();

        rt.run().unwrap();

        let expected =
            ndarray12::arr2(&[[0.0, 1., 2., 3., 0., 0.], [0., 4., 5., 6., 0., 0.]]).into_dyn();

        assert_eq!(
            ArrayD::<f32>::try_from(&rt.get_output(0).unwrap()).unwrap(),
            expected
        );
    }

    #[test]
    fn windows_0() {
        let mut name_to_shape = HashMap::new();
        name_to_shape.insert("t".to_string(), vec![3, 3, 3]);

        let glenside_expr = RecExpr::<Language>::from_str(
            "(access-windows
            (access (access-tensor t) 0)
            (shape 3 2 2)
            (shape 1 1 1))",
        )
        .unwrap();

        let mut egraph = EGraph::new(MyAnalysis { name_to_shape });

        let id = egraph.add_expr(&glenside_expr);

        let out = to_relay(&egraph, id, Device::cpu(0));

        let module =
            compile_module(CompilerConfig::default(), IRModule::from_expr(out).unwrap()).unwrap();

        let mut rt = GraphRt::from_module(module, Device::cpu(0)).unwrap();
        let input = ndarray12::array![
            [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
            [[10., 11., 12.], [13., 14., 15.], [16., 17., 18.]],
            [[19., 20., 21.], [22., 23., 24.], [25., 26., 27.]],
        ]
        .into_dyn();
        rt.set_input(
            "t",
            NDArray::from_rust_ndarray(&input, Device::cpu(0), DataType::float32()).unwrap(),
        )
        .unwrap();

        rt.run().unwrap();

        let out = ArrayD::<f32>::try_from(&rt.get_output(0).unwrap()).unwrap();
        assert_eq!(out.shape(), &[1, 2, 2, 3, 2, 2]);
        assert_eq!(
            out.slice(s![0, 0, 0, .., .., ..]),
            ndarray12::array![
                [[1., 2.], [4., 5.]],
                [[10., 11.], [13., 14.]],
                [[19., 20.], [22., 23.]],
            ]
        );
        assert_eq!(
            out.slice(s![0, 1, 0, .., .., ..]),
            ndarray12::array![
                [[4., 5.], [7., 8.]],
                [[13., 14.], [16., 17.]],
                [[22., 23.], [25., 26.]],
            ]
        );
    }

    #[test]
    fn windows_1() {
        let mut name_to_shape = HashMap::new();
        name_to_shape.insert("t".to_string(), vec![3, 3, 3]);

        let glenside_expr = RecExpr::<Language>::from_str(
            "(access-windows
            (access (access-tensor t) 1)
            (shape 2 2)
            (shape 1 1))",
        )
        .unwrap();

        let mut egraph = EGraph::new(MyAnalysis { name_to_shape });

        let id = egraph.add_expr(&glenside_expr);

        let out = to_relay(&egraph, id, Device::cpu(0));

        let module =
            compile_module(CompilerConfig::default(), IRModule::from_expr(out).unwrap()).unwrap();

        let mut rt = GraphRt::from_module(module, Device::cpu(0)).unwrap();
        let input = ndarray12::array![
            [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
            [[10., 11., 12.], [13., 14., 15.], [16., 17., 18.]],
            [[19., 20., 21.], [22., 23., 24.], [25., 26., 27.]],
        ]
        .into_dyn();
        rt.set_input(
            "t",
            NDArray::from_rust_ndarray(&input, Device::cpu(0), DataType::float32()).unwrap(),
        )
        .unwrap();

        rt.run().unwrap();

        let out = ArrayD::<f32>::try_from(&rt.get_output(0).unwrap()).unwrap();

        assert_eq!(out.shape(), &[3, 2, 2, 2, 2]);
        assert_eq!(
            out.slice(s![0, 0, 0, .., ..]),
            ndarray12::array![[1., 2.], [4., 5.]]
        );
        assert_eq!(
            out.slice(s![0, 1, 0, .., ..]),
            ndarray12::array![[4., 5.], [7., 8.]]
        );
        assert_eq!(
            out.slice(s![2, 0, 1, .., ..]),
            ndarray12::array![[20., 21.], [23., 24.]]
        );
    }

    #[test]
    fn windows_1_with_striding() {
        let mut name_to_shape = HashMap::new();
        name_to_shape.insert("t".to_string(), vec![3, 3, 4]);

        let glenside_expr = RecExpr::<Language>::from_str(
            "(access-windows
            (access (access-tensor t) 1)
            (shape 2 2)
            (shape 1 2))",
        )
        .unwrap();

        let mut egraph = EGraph::new(MyAnalysis { name_to_shape });

        let id = egraph.add_expr(&glenside_expr);

        let out = to_relay(&egraph, id, Device::cpu(0));

        let module =
            compile_module(CompilerConfig::default(), IRModule::from_expr(out).unwrap()).unwrap();

        let mut rt = GraphRt::from_module(module, Device::cpu(0)).unwrap();
        let input = ndarray12::array![
            [[1., 2., 3., 28.], [4., 5., 6., 29.], [7., 8., 9., 30.]],
            [
                [10., 11., 12., 31.],
                [13., 14., 15., 32.],
                [16., 17., 18., 33.]
            ],
            [
                [19., 20., 21., 34.],
                [22., 23., 24., 35.],
                [25., 26., 27., 36.]
            ],
        ]
        .into_dyn();
        rt.set_input(
            "t",
            NDArray::from_rust_ndarray(&input, Device::cpu(0), DataType::float32()).unwrap(),
        )
        .unwrap();

        rt.run().unwrap();

        let out = ArrayD::<f32>::try_from(&rt.get_output(0).unwrap()).unwrap();

        assert_eq!(out.shape(), &[3, 2, 2, 2, 2]);
        assert_eq!(
            out.slice(s![0, 0, 0, .., ..]),
            ndarray12::array![[1., 2.], [4., 5.]]
        );
        assert_eq!(
            out.slice(s![0, 1, 0, .., ..]),
            ndarray12::array![[4., 5.], [7., 8.]]
        );
        assert_eq!(
            out.slice(s![2, 0, 1, .., ..]),
            ndarray12::array![[21., 34.], [24., 35.]]
        );
    }

    #[test]
    fn windows_2() {
        let mut name_to_shape = HashMap::new();
        name_to_shape.insert("t".to_string(), vec![3, 3, 3]);

        let glenside_expr = RecExpr::<Language>::from_str(
            "(access-windows
            (access (access-tensor t) 2)
            (shape 2)
            (shape 1))",
        )
        .unwrap();

        let mut egraph = EGraph::new(MyAnalysis { name_to_shape });

        let id = egraph.add_expr(&glenside_expr);

        let out = to_relay(&egraph, id, Device::cpu(0));

        let module =
            compile_module(CompilerConfig::default(), IRModule::from_expr(out).unwrap()).unwrap();

        let mut rt = GraphRt::from_module(module, Device::cpu(0)).unwrap();
        let input = ndarray12::array![
            [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
            [[10., 11., 12.], [13., 14., 15.], [16., 17., 18.]],
            [[19., 20., 21.], [22., 23., 24.], [25., 26., 27.]],
        ]
        .into_dyn();
        rt.set_input(
            "t",
            NDArray::from_rust_ndarray(&input, Device::cpu(0), DataType::float32()).unwrap(),
        )
        .unwrap();

        rt.run().unwrap();

        let out = ArrayD::<f32>::try_from(&rt.get_output(0).unwrap()).unwrap();

        assert_eq!(out.shape(), &[3, 3, 2, 2]);
        assert_eq!(out.slice(s![0, 0, 0, ..]), ndarray12::array![1., 2.]);
        assert_eq!(out.slice(s![0, 1, 0, ..]), ndarray12::array![4., 5.]);
    }

    #[test]
    fn squeeze() {
        let mut name_to_shape = HashMap::new();
        name_to_shape.insert("a".to_string(), vec![1, 2, 3]);

        let glenside_expr =
            RecExpr::<Language>::from_str("(access-squeeze (access-tensor a) 0))").unwrap();

        let mut egraph = EGraph::new(MyAnalysis { name_to_shape });

        let id = egraph.add_expr(&glenside_expr);

        let out = to_relay(&egraph, id, Device::cpu(0));

        let module =
            compile_module(CompilerConfig::default(), IRModule::from_expr(out).unwrap()).unwrap();

        let mut rt = GraphRt::from_module(module, Device::cpu(0)).unwrap();
        let input =
            ArrayD::from_shape_fn(vec![1, 2, 3], |d| d.slice().iter().sum::<usize>() as f32);
        rt.set_input(
            "a",
            NDArray::from_rust_ndarray(&input, Device::cpu(0), DataType::float32()).unwrap(),
        )
        .unwrap();

        rt.run().unwrap();

        let expected = input.into_shape(vec![2, 3]).unwrap();

        assert_eq!(
            ArrayD::<f32>::try_from(&rt.get_output(0).unwrap()).unwrap(),
            expected
        );
    }

    #[test]
    fn reshape() {
        let mut name_to_shape = HashMap::new();
        name_to_shape.insert("a".to_string(), vec![1, 2, 3]);

        let glenside_expr = RecExpr::<Language>::from_str(
            "(access-reshape (access (access-tensor a) 1) (access-shape (shape 3) (shape 2)))",
        )
        .unwrap();

        let mut egraph = EGraph::new(MyAnalysis { name_to_shape });

        let id = egraph.add_expr(&glenside_expr);

        let out = to_relay(&egraph, id, Device::cpu(0));

        let module =
            compile_module(CompilerConfig::default(), IRModule::from_expr(out).unwrap()).unwrap();

        let mut rt = GraphRt::from_module(module, Device::cpu(0)).unwrap();
        let input =
            ArrayD::from_shape_fn(vec![1, 2, 3], |d| d.slice().iter().sum::<usize>() as f32);
        rt.set_input(
            "a",
            NDArray::from_rust_ndarray(&input, Device::cpu(0), DataType::float32()).unwrap(),
        )
        .unwrap();

        rt.run().unwrap();

        let expected = input.into_shape(vec![3, 2]).unwrap();

        assert_eq!(
            ArrayD::<f32>::try_from(&rt.get_output(0).unwrap()).unwrap(),
            expected
        );
    }

    #[test]
    fn transpose() {
        let mut name_to_shape = HashMap::new();
        name_to_shape.insert("a".to_string(), vec![1, 2, 3]);

        let glenside_expr = RecExpr::<Language>::from_str(
            "(access-transpose (access (access-tensor a) 1) (list 2 0 1))",
        )
        .unwrap();

        let mut egraph = EGraph::new(MyAnalysis { name_to_shape });

        let id = egraph.add_expr(&glenside_expr);

        let out = to_relay(&egraph, id, Device::cpu(0));

        let module =
            compile_module(CompilerConfig::default(), IRModule::from_expr(out).unwrap()).unwrap();

        let mut rt = GraphRt::from_module(module, Device::cpu(0)).unwrap();
        let input =
            ArrayD::from_shape_fn(vec![1, 2, 3], |d| d.slice().iter().sum::<usize>() as f32);
        rt.set_input(
            "a",
            NDArray::from_rust_ndarray(&input, Device::cpu(0), DataType::float32()).unwrap(),
        )
        .unwrap();

        rt.run().unwrap();

        let expected = input.permuted_axes(vec![2, 0, 1]);

        assert_eq!(
            ArrayD::<f32>::try_from(&rt.get_output(0).unwrap()).unwrap(),
            expected
        );
    }
}
