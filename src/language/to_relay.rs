use super::super::codegen::generate_worklist_for_codegen;
use super::Language;
use super::MyAnalysis;
use egg::EGraph;
use egg::Id;
use ndarray::Dimension;
use std::collections::HashMap;
use std::convert::TryFrom;
use tvm::ir::relay::*;
use tvm::ir::span::Span;
use tvm::ir::ty::TensorType;
use tvm::runtime::IsObjectRef;
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
        Language::AccessWindows(_) => todo!(),
        Language::ShapeOf(_) => todo!(),
        Language::SliceShape(_) => todo!(),
        Language::ShapeInsertAxis(_) => todo!(),
        Language::ShapeRemoveAxis(_) => todo!(),
        Language::Access(_) => todo!(),
        Language::AccessTranspose(_) => todo!(),
        Language::AccessCartesianProduct(_) => todo!(),
        Language::Compute(_) => todo!(),
        Language::AccessReshape(_) => todo!(),
        Language::AccessFlatten(_) => todo!(),
        Language::Shape(_) => todo!(),
        Language::List(_) => todo!(),
        Language::ConstructTuple(_) => todo!(),
        Language::TupleGetItem(_) => todo!(),
        Language::AccessShape(_) => todo!(),
        Language::AccessSlice(_) => todo!(),
        Language::AccessConcatenate(_) => todo!(),
        Language::AccessPair(_) => todo!(),
        Language::AccessShiftRight(_) => todo!(),
        Language::AccessTensor(child_id) => {
            hashmap.insert(id, hashmap[child_id].clone());
        }
        Language::AccessPad(_) => todo!(),
        Language::AccessSqueeze(_) => todo!(),
        Language::AccessInsertAxis(_) => todo!(),
        Language::AccessBroadcast(_) => todo!(),
        Language::AccessLiteral(_) => todo!(),
        Language::Literal(_) => todo!(),
        Language::RelayOperatorCall(_) => todo!(),
        Language::NotNanFloat64(_) => todo!(),
        Language::RelayOperator(_) => todo!(),
        Language::RelayActivationLayout(_) => todo!(),
        Language::RelayKernelLayout(_) => todo!(),
        Language::PadType(_) => todo!(),
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
        Language::AcceleratorCall(_) => todo!(),
        Language::AcceleratorFunc(_) => (),
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::*;
    use egg::RecExpr;
    use ndarray12::arr0;
    use ndarray12::ArrayD;
    use ndarray12::Dimension;
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
    #[should_panic = "not yet implemented"]
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

        let module =
            compile_module(CompilerConfig::default(), IRModule::from_expr(out).unwrap()).unwrap();

        let mut rt = GraphRt::from_module(module, Device::cpu(0)).unwrap();
        let data_input =
            ArrayD::from_shape_fn(vec![1, 3, 32], |d| d.slice().iter().sum::<usize>() as f32);
        let weight_input =
            ArrayD::from_shape_fn(vec![8, 3, 3], |d| d.slice().iter().sum::<usize>() as f32);
        rt.set_input(
            "data",
            NDArray::from_rust_ndarray(&data_input, Device::cpu(0), DataType::float32()).unwrap(),
        )
        .unwrap();
        rt.set_input(
            "weight",
            NDArray::from_rust_ndarray(&weight_input, Device::cpu(0), DataType::float32()).unwrap(),
        )
        .unwrap();

        rt.run().unwrap();

        todo!("Check output");
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
}
