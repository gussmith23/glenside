pub mod ilp;

use crate::language::{ComputeType, Language, MyAnalysis};
use egg::{CostFunction, EGraph, Id, Language as LanguageTrait, Pattern, Searcher};
use std::collections::HashSet;

pub fn find_all_systolic_array_configurations(
    egraph: &EGraph<Language, MyAnalysis>,
) -> HashSet<(usize, usize)> {
    let mut systolic_arrays = HashSet::new();
    for matches in "(systolic-array ?rows ?cols ?x ?y)"
        .parse::<Pattern<Language>>()
        .unwrap()
        .search(egraph)
        .iter()
    {
        for subst in matches.substs.iter() {
            systolic_arrays.insert((
                MyAnalysis::get_usize(subst["?rows".parse().unwrap()], egraph),
                MyAnalysis::get_usize(subst["?cols".parse().unwrap()], egraph),
            ));
        }
    }

    systolic_arrays
}

/// A cost function to extract a design using a single size of systolic array.
///
/// `INFINITY_VALUE` represents constructs with infinite cost, i.e., constructs
/// that shouldn't be extracted. To check whether the extracted program does not
/// contain any un-extractable constructs (i.e. compute statements), you can
/// check that the cost is less than `INFINITY_VALUE`. You might think we should
/// instead use [`usize::MAX`] and a saturating add, but this actually has the
/// potential to cause infinite loops in [`egg::Extractor::find_best()`]!
pub struct MonolithicCostFunction<'a> {
    pub systolic_array_configuration: (usize, usize),
    pub egraph: &'a EGraph<Language, MyAnalysis>,
    /// Whether to prioritize systolic-array or systolic-array-with-blocking
    // TODO(@gussmith23) This needs to be tested
    pub prefer_systolic_arrays_with_blocking: bool,
}
impl<'a> MonolithicCostFunction<'a> {
    pub const INFINITY_VALUE: <MonolithicCostFunction<'a> as egg::CostFunction<Language>>::Cost =
        1000000000;
}
impl egg::CostFunction<Language> for MonolithicCostFunction<'_> {
    type Cost = usize;

    fn cost<C>(&mut self, enode: &Language, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        let base_cost = match enode {
            &Language::SystolicArray([rows_id, cols_id, _tensor_0_id, _tensor_1_id])
            | &Language::SystolicArrayWithBlocking([rows_id, cols_id, _tensor_0_id, _tensor_1_id])
                if (
                    MyAnalysis::get_usize(rows_id, self.egraph),
                    MyAnalysis::get_usize(cols_id, self.egraph),
                ) != self.systolic_array_configuration =>
            {
                Self::INFINITY_VALUE
            }

            Language::Symbol(_)
            | Language::ConstantTensor(_)
            | Language::AccessLiteral(_)
            | Language::Literal(_)
            | Language::NotNanFloat64(_)
            | Language::SystolicArray(_)
            | Language::SystolicArrayWithBlocking(_)
            | Language::Num(_)
            | Language::ConstructTuple(_)
            | Language::TupleGetItem(_)
            | Language::AccessSlice(_)
            | Language::AccessConcatenate(_)
            | Language::AccessPad(_)
            | Language::AccessWindows(_)
            | Language::PadType(_)
            | Language::Access(_)
            | Language::AccessTensor(_)
            | Language::ShapeOf(_)
            | Language::ShapeRemoveAxis(_)
            | Language::ShapeInsertAxis(_)
            | Language::Shape(_)
            | Language::AccessSqueeze(_)
            | Language::AccessCartesianProduct(_)
            | Language::AccessFlatten(_)
            | Language::AccessReshape(_)
            | Language::AccessShiftRight(_)
            | Language::AccessInsertAxis(_)
            | Language::AccessBroadcast(_)
            | Language::AccessShape(_)
            | Language::List(_)
            | Language::SliceShape(_)
            | Language::AccessPair(_)
            // We don't penalize Compute, though we don't want to extract
            // compute statements. Instead, we penalize most ComputeTypes, and
            // let some types pass through until we've implemented some other
            // way to handle them.
            // TODO(@gussmith23) We shouldn't have to extract ANY computes!
            | Language::Compute(_)
            | Language::GetAccessShape(_)
            | Language::AccessTranspose(_) => 1,
            | Language::AcceleratorCall(_) => 0,
            | Language::AcceleratorFunc(_) => 0,

            // Penalize specific compute types. In the future, these constructs
            // shouldn't be extractable at all.
            // TODO(@gussmith23) We shouldn't have to extract ANY computes!
            Language::ComputeType(t) => match t {
                crate::language::ComputeType::DotProduct => Self::INFINITY_VALUE,
                crate::language::ComputeType::ReduceSum => 1,
                crate::language::ComputeType::ReLU => 1,
                crate::language::ComputeType::Sqrt => 1,
                crate::language::ComputeType::Negative => 1,
                crate::language::ComputeType::ElementwiseAdd => 1,
                crate::language::ComputeType::ElementwiseMul => 1,
                crate::language::ComputeType::ElementwiseDiv => 1,
                crate::language::ComputeType::ReduceMax => 1,
                crate::language::ComputeType::Softmax => 1,
                crate::language::ComputeType::ReduceMean => 1,
            }

            Language::SystolicArrayConv2dIm2colNchwOihwWithBlocking(_) => todo!(),
            Language::SystolicArrayConv2dIm2colNhwcHwioWithBlocking(_) => todo!(),
            Language::SystolicArrayConv2dNchwOihwWithBlocking(_) => todo!(),
            Language::SystolicArrayConv2dNhwcHwioWithBlocking(_) => todo!(),
            Language::DataType(_) => todo!(),

            // Don't extract relay nodes.
            Language::RelayOperatorCall(_) 
            | Language::RelayOperator(_)
            | Language::RelayActivationLayout(_)
            | Language::RelayKernelLayout(_) => Self::INFINITY_VALUE,
        };

        enode.fold(base_cost, |sum, id| sum + costs(id))
    }
}

/// This cost function applies the bare minimum amount of logic to produce a
/// valid hardware/software program. Most importantly, it blocks Compute nodes
/// from being extracted, as these nodes should be replaced by hardware atoms
/// (or, perhaps in the future, kernel calls). It also filters out old Glenside
/// constructs.
pub struct SimpleCostFunction {
    /// Whether to prioritize systolic-array or systolic-array-with-blocking
    // TODO(@gussmith23) This needs to be tested
    pub prefer_systolic_arrays_with_blocking: bool,
}
impl Default for SimpleCostFunction {
    fn default() -> Self {
        SimpleCostFunction {
            prefer_systolic_arrays_with_blocking: false,
        }
    }
}
impl CostFunction<Language> for SimpleCostFunction {
    type Cost = usize;

    fn cost<C>(&mut self, enode: &Language, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        use crate::language::Language::*;
        let base_cost = match enode {
            Language::RelayOperator(_) => todo!(),
            Language::GetAccessShape(_) => todo!(),
            Language::RelayOperatorCall(_) => todo!(),
            Language::RelayActivationLayout(_) => todo!(),
            Language::RelayKernelLayout(_) => todo!(),
            Language::DataType(_) => todo!(),
            Language::SystolicArrayConv2dNchwOihwWithBlocking(_) => todo!(),
            Language::SystolicArrayConv2dNhwcHwioWithBlocking(_) => todo!(),
            Language::SystolicArrayConv2dIm2colNchwOihwWithBlocking(_) => todo!(),
            Language::SystolicArrayConv2dIm2colNhwcHwioWithBlocking(_) => todo!(),
            Language::ConstructTuple(_) => todo!(),
            Language::TupleGetItem(_) => todo!(),

            // Cannot extract compute: compute must be lowered to an atom.
            Compute(_) => std::usize::MAX,
            AcceleratorFunc(_) => 1,
            AcceleratorCall(_) => 1,
            ConstantTensor(_) => 1,
            // Extracting hardware atoms is encouraged
            SystolicArray(_) => {
                if !self.prefer_systolic_arrays_with_blocking {
                    1
                } else {
                    usize::MAX
                }
            }
            SystolicArrayWithBlocking(_) => {
                if self.prefer_systolic_arrays_with_blocking {
                    1
                } else {
                    usize::MAX
                }
            }
            // Extracting various access patterns is essential.
            AccessWindows(_)
            | Access(_)
            | AccessLiteral(_)
            | AccessTranspose(_)
            | AccessCartesianProduct(_)
            | AccessReshape(_)
            | AccessFlatten(_)
            | AccessSlice(_)
            | AccessConcatenate(_)
            | AccessPair(_)
            | AccessShiftRight(_)
            | AccessTensor(_)
            | AccessSqueeze(_)
            | AccessPad(_)
            | AccessInsertAxis(_)
            | AccessBroadcast(_) => 1,
            // Other glenside constructs that are necessary.
            Shape(_) | ShapeOf(_) | SliceShape(_) | ShapeInsertAxis(_) | ShapeRemoveAxis(_)
            | List(_) | AccessShape(_) | Num(_) | PadType(_) | ComputeType(_) | Symbol(_)
            | Literal(_) | NotNanFloat64(_) => 1,
        };

        enode.fold(base_cost, |sum, id| sum.saturating_add(costs(id)))
    }
}

pub struct AcceleratorCostFunction(pub f64);

impl CostFunction<Language> for AcceleratorCostFunction {
    type Cost = f64;
    fn cost<C>(&mut self, enode: &Language, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        if let Language::AcceleratorCall(_) = &enode {
            return 0.0;
        }
        let base_cost: f64 = match enode {
            // We only consider accelerator calls and relay operators for now when
            // extracting a model
            Language::Access(_)
            | Language::List(_)
            | Language::Shape(_)
            | Language::GetAccessShape(_)
            | Language::Num(_)
            | Language::AccessLiteral(_)
            | Language::Literal(_)
            | Language::AcceleratorCall(_)
            | Language::AccessShape(_)
            | Language::AcceleratorFunc(_)
            | Language::Symbol(_)
            | Language::RelayOperator(_)
            | Language::PadType(_)
            | Language::ConstructTuple(_)
            | Language::ConstantTensor(_)
            | Language::TupleGetItem(_)
            | Language::DataType(_)
            | Language::AccessTensor(_) => 0.0,
            Language::RelayOperatorCall(_) => self.0 / 2.0,
            Language::AccessTranspose(_)
            | Language::RelayKernelLayout(_)
            | Language::RelayActivationLayout(_)
            | Language::NotNanFloat64(_)
            | Language::AccessPad(_)
            | Language::AccessFlatten(_)
            | Language::AccessWindows(_)
            | Language::AccessInsertAxis(_)
            | Language::AccessSqueeze(_) => 1.0,

            Language::Compute(_) => 1.0,
            Language::AccessReshape(_) => self.0,
            Language::ComputeType(compute_type) => match compute_type {
                ComputeType::DotProduct
                | ComputeType::Softmax
                | ComputeType::ReLU
                | ComputeType::ReduceSum
                | ComputeType::ReduceMean => self.0,
                _ => 1.0,
            },
            Language::AccessCartesianProduct(_)
            | Language::SystolicArray(_)
            | Language::AccessBroadcast(_)
            | Language::SystolicArrayConv2dIm2colNchwOihwWithBlocking(_)
            | Language::SystolicArrayConv2dIm2colNhwcHwioWithBlocking(_)
            | Language::SystolicArrayConv2dNchwOihwWithBlocking(_)
            | Language::SystolicArrayConv2dNhwcHwioWithBlocking(_)
            | Language::SystolicArrayWithBlocking(_)
            | Language::ShapeOf(_)
            | Language::SliceShape(_)
            | Language::ShapeInsertAxis(_)
            | Language::ShapeRemoveAxis(_)
            | Language::AccessSlice(_)
            | Language::AccessConcatenate(_)
            | Language::AccessShiftRight(_)
            | Language::AccessPair(_) => self.0 * 100.0,
        };
        enode.fold(base_cost, |sum, id| sum + costs(id))
    }
}

#[cfg(test)]
mod tests {
    use super::super::language::MyAnalysis;
    use super::*;
    use egg::{EGraph, Extractor};
    use std::collections::HashMap;

    #[test]
    fn find_systolic_array_configs_0() {
        let program = "
         (systolic-array 128 16
          (access
           (systolic-array 64 128
            (access
             (systolic-array 32 64
              (access (access-tensor v-32) 0)
              (access (access-tensor t-32-64) 0)
             )
             0
            )
            (access (access-tensor t-64-128) 0)
           )
           0
          )
          (access (access-tensor t-128-16) 0)
         )
         "
        .parse()
        .unwrap();

        let mut egraph = EGraph::new(MyAnalysis::default());
        egraph.add_expr(&program);
        egraph.rebuild();

        let configs = find_all_systolic_array_configurations(&egraph);

        assert_eq!(configs.len(), 3);
        assert!(configs.contains(&(128, 16)));
        assert!(configs.contains(&(64, 128)));
        assert!(configs.contains(&(32, 64)));
    }

    #[test]
    fn find_systolic_array_configs_1() {
        let program = "
         (systolic-array 32 32
          (access
           (systolic-array 32 32
            (access (access-tensor v-32) 0)
            (access (access-tensor t-32-32) 0)
           )
           0
          )
          (access
           (systolic-array 32 32
            (access (access-tensor t-32-32) 1)
            (access (access-tensor t-32-32) 0)
           )
           0
          )
         )
         "
        .parse()
        .unwrap();

        let mut egraph = EGraph::new(MyAnalysis::default());
        egraph.add_expr(&program);
        egraph.rebuild();

        let configs = find_all_systolic_array_configurations(&egraph);

        assert_eq!(configs.len(), 1);
        assert!(configs.contains(&(32, 32)));
    }

    #[test]
    fn extract_0() {
        let program = "
         (systolic-array 128 16
          (access
           (systolic-array 64 128
            (access
             (systolic-array 32 64
              (access (access-tensor v-32) 0)
              (access (access-tensor t-32-64) 0)
             )
             0
            )
            (access (access-tensor t-64-128) 0)
           )
           0
          )
          (access (access-tensor t-128-16) 0)
         )
         "
        .parse()
        .unwrap();

        let mut egraph = EGraph::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        egraph.rebuild();

        let ex = Extractor::new(
            &egraph,
            MonolithicCostFunction {
                systolic_array_configuration: (16, 128),
                egraph: &egraph,
                prefer_systolic_arrays_with_blocking: false,
            },
        );

        let (cost, _) = ex.find_best(id);
        assert!(cost >= MonolithicCostFunction::INFINITY_VALUE);
    }

    #[test]
    fn extract_1() {
        let program = "
         (systolic-array 32 32
          (access
           (systolic-array 32 32
            (access (access-tensor v-32) 0)
            (access (access-tensor t-32-32) 0)
           )
           0
          )
          (access
           (systolic-array 32 32
            (access (access-tensor t-32-32) 1)
            (access (access-tensor t-32-32) 0)
           )
           0
          )
         )
         "
        .parse()
        .unwrap();

        let mut egraph = EGraph::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        egraph.rebuild();

        let ex = Extractor::new(
            &egraph,
            MonolithicCostFunction {
                egraph: &egraph,
                systolic_array_configuration: (32, 32),
                prefer_systolic_arrays_with_blocking: false,
            },
        );

        let (cost, expr) = ex.find_best(id);
        assert!(cost < MonolithicCostFunction::INFINITY_VALUE);
        // TODO(@gussmith23) Do this check in a more intelligent way.
        //
        // For some reason, comparing RecExprs doesn't work anymore, after a
        // recent egg update. Even if the RecExprs are the same structurally.
        assert_eq!(
            expr.pretty(80),
            r#"(systolic-array
  32
  32
  (access
    (systolic-array
      32
      32
      (access (access-tensor v-32) 0)
      (access (access-tensor t-32-32) 0))
    0)
  (access
    (systolic-array
      32
      32
      (access (access-tensor t-32-32) 1)
      (access (access-tensor t-32-32) 0))
    0))"#
        );
    }

    #[test]
    fn simple_cost_function_0() {
        let mut map = HashMap::default();
        map.insert("input".to_string(), vec![32]);
        map.insert("weight0".to_string(), vec![32, 64]);
        map.insert("weight1".to_string(), vec![64, 128]);
        map.insert("weight2".to_string(), vec![128, 16]);
        let program = "
         (systolic-array 128 16
          (access
           (systolic-array 64 128
            (access
             (systolic-array 32 64
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
         "
        .parse()
        .unwrap();

        let mut egraph = EGraph::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        egraph.rebuild();

        let ex = Extractor::new(&egraph, SimpleCostFunction::default());

        let (cost, best) = ex.find_best(id);
        assert!(cost < std::usize::MAX);

        // TODO(@gussmith23) Do this check in a more intelligent way.
        //
        // For some reason, comparing RecExprs doesn't work anymore, after a
        // recent egg update. Even if the RecExprs are the same structurally.
        assert_eq!(
            best.pretty(80),
            "(systolic-array
  128
  16
  (access
    (systolic-array
      64
      128
      (access
        (systolic-array
          32
          64
          (access (access-tensor input) 0)
          (access (access-tensor weight0) 0))
        0)
      (access (access-tensor weight1) 0))
    0)
  (access (access-tensor weight2) 0))"
        );
    }
}
