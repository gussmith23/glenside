use crate::language::{Language, MyAnalysis};
use egg::{CostFunction, EGraph, Id, Language as LanguageTrait, Pattern, Searcher};
use std::collections::HashSet;

pub fn find_all_systolic_array_configurations(
    egraph: &EGraph<Language, MyAnalysis>,
) -> HashSet<(usize, usize)> {
    let mut systolic_arrays = HashSet::new();
    for matches in "(bsg-systolic-array ?rows ?cols ?x ?y)"
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

pub struct MonolithicCostFunction<'a> {
    pub systolic_array_configuration: (usize, usize),
    pub egraph: &'a EGraph<Language, MyAnalysis>,
    pub infinite_cost_value: <MonolithicCostFunction<'a> as CostFunction<Language>>::Cost,
}
impl CostFunction<Language> for MonolithicCostFunction<'_> {
    type Cost = usize;

    fn cost<C>(&mut self, enode: &Language, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        // TODO(@gussmith23) Implement a real cost model
        let base_cost = match enode {
            &Language::BsgSystolicArray([rows_id, cols_id, _tensor_0_id, _tensor_1_id])
                if (
                    MyAnalysis::get_usize(rows_id, self.egraph),
                    MyAnalysis::get_usize(cols_id, self.egraph),
                ) != self.systolic_array_configuration =>
            {
                self.infinite_cost_value
            }

            Language::Symbol(_)
            | Language::BsgSystolicArray(_)
            | Language::Usize(_)
            | Language::Slice(_)
            | Language::Concatenate(_)
            | Language::ElementwiseAdd(_) => 1,
            _ => self.infinite_cost_value,
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
            | AccessMoveAxis(_)
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
            | GetAccessShape(_) | AccessShape(_) | Usize(_) | PadType(_) | ComputeType(_)
            | Symbol(_) => 1,
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
    fn find_systolic_array_configs() {
        let program = "
         (bsg-systolic-array 16 128
          (bsg-systolic-array 128 64
           (bsg-systolic-array 64 32
            v-32
            (move-axis t-32-64 1 0)
           )
           (move-axis t-64-128 1 0)
          )
          (move-axis t-128-16 1 0)
         )
         "
        .parse()
        .unwrap();

        let mut egraph = EGraph::new(MyAnalysis::default());
        egraph.add_expr(&program);
        egraph.rebuild();

        let configs = find_all_systolic_array_configurations(&egraph);

        assert_eq!(configs.len(), 3);
        assert!(configs.contains(&(16, 128)));
        assert!(configs.contains(&(128, 64)));
        assert!(configs.contains(&(64, 32)));

        let program = "
         (bsg-systolic-array 32 64
          (bsg-systolic-array 32 64
           v-32
           t-32-64
          )
          (bsg-systolic-array 32 64
           t-32-32
           t-32-64
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
        assert!(configs.contains(&(32, 64)));
    }

    #[test]
    fn extract() {
        const INFINITE: usize = 1000000;

        let program = "
         (bsg-systolic-array 16 128
          (bsg-systolic-array 128 64
           (bsg-systolic-array 64 32
            v-32
            t-32-64
           )
           t-64-128
          )
          t-128-16
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
                infinite_cost_value: INFINITE,
            },
        );

        let (cost, _) = ex.find_best(id);
        assert!(cost > INFINITE);

        let program = "
         (bsg-systolic-array 32 64
          (bsg-systolic-array 32 64
           v-32
           t-32-64
          )
          (bsg-systolic-array 32 64
           t-32-32
           t-32-64
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
                systolic_array_configuration: (32, 64),
                infinite_cost_value: INFINITE,
            },
        );

        let (cost, expr) = ex.find_best(id);
        assert!(cost < INFINITE);
        assert_eq!(
            expr,
            "
         (bsg-systolic-array 32 64
          (bsg-systolic-array 32 64
           v-32
           t-32-64
          )
          (bsg-systolic-array 32 64
           t-32-32
           t-32-64
          )
         )
         "
            .parse()
            .unwrap()
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
        assert!(cost < usize::MAX);

        assert_eq!(
            best,
            "
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
            .unwrap()
        );
    }
}
