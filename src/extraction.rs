use crate::language::{Language, MyAnalysis};
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
                if (
                    MyAnalysis::get_usize(rows_id, self.egraph),
                    MyAnalysis::get_usize(cols_id, self.egraph),
                ) != self.systolic_array_configuration =>
            {
                Self::INFINITY_VALUE
            }

            Language::Symbol(_)
            | Language::AccessLiteral(_)
            | Language::Literal(_)
            | Language::NotNanFloat64(_)
            | Language::SystolicArray(_)
            | Language::Usize(_)
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
            | Language::GetAccessShape(_)
            | Language::AccessShape(_)
            | Language::List(_)
            | Language::SliceShape(_)
            | Language::AccessPair(_)
            | Language::AccessTranspose(_) => 1,

            // Computes cannot be extracted.
            Language::ComputeType(_) | Language::Compute(_) => Self::INFINITY_VALUE,

            // Old constructs.
            Language::MoveAxis(_)
            | Language::CartesianProduct(_)
            | Language::ElementwiseAdd(_)
            | Language::BsgSystolicArray(_)
            | Language::MapDotProduct(_)
            | Language::Slice(_)
            | Language::Concatenate(_) => panic!(),
        };

        enode.fold(base_cost, |sum, id| sum + costs(id))
    }
}

/// This cost function applies the bare minimum amount of logic to produce a
/// valid hardware/software program. Most importantly, it blocks Compute nodes
/// from being extracted, as these nodes should be replaced by hardware atoms
/// (or, perhaps in the future, kernel calls). It also filters out old Glenside
/// constructs.
pub struct SimpleCostFunction;
impl CostFunction<Language> for SimpleCostFunction {
    type Cost = usize;

    fn cost<C>(&mut self, enode: &Language, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        use crate::language::Language::*;
        let base_cost = match enode {
            // Cannot extract compute: compute must be lowered to an atom.
            Compute(_) => std::usize::MAX,
            // Extracting hardware atoms is encouraged.
            SystolicArray(_) => 1,
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
            | List(_) | GetAccessShape(_) | AccessShape(_) | Usize(_) | PadType(_)
            | ComputeType(_) | Symbol(_) | Literal(_) | NotNanFloat64(_) => 1,
            // Old constructs that are no longer used
            MoveAxis(_) | CartesianProduct(_) | MapDotProduct(_) | Slice(_) | Concatenate(_)
            | ElementwiseAdd(_) | BsgSystolicArray(_) => std::usize::MAX,
        };

        enode.fold(base_cost, |sum, id| sum.saturating_add(costs(id)))
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

        let mut ex = Extractor::new(
            &egraph,
            MonolithicCostFunction {
                systolic_array_configuration: (16, 128),
                egraph: &egraph,
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

        let mut ex = Extractor::new(
            &egraph,
            MonolithicCostFunction {
                egraph: &egraph,
                systolic_array_configuration: (32, 32),
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

        let mut egraph = EGraph::new(MyAnalysis { name_to_shape: map });
        let id = egraph.add_expr(&program);
        egraph.rebuild();

        let mut ex = Extractor::new(&egraph, SimpleCostFunction);

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
