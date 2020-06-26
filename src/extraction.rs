use crate::language::{Language, MyAnalysis};
use egg::{CostFunction, EGraph, Id, Language as LanguageTrait, Pattern, Searcher};
use std::collections::HashSet;

pub fn find_all_systolic_array_configurations(
    egraph: &EGraph<Language, MyAnalysis>,
) -> HashSet<(usize, usize)> {
    let mut systolic_arrays = HashSet::new();
    println!(
        "{:?}",
        "(bsg-systolic-array ?rows ?cols ?x ?y)"
            .parse::<Pattern<Language>>()
            .unwrap()
            .search(egraph)
    );
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

struct MonolithicCostFunction<'a> {
    systolic_array_configuration: (usize, usize),
    egraph: &'a EGraph<Language, MyAnalysis>,
    infinite_cost_value: <MonolithicCostFunction<'a> as CostFunction<Language>>::Cost,
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

#[cfg(test)]
mod tests {
    use super::super::language::MyAnalysis;
    use super::*;
    use egg::{EGraph, Extractor};

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

        let mut egraph = EGraph::new(MyAnalysis);
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

        let mut egraph = EGraph::new(MyAnalysis);
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

        let mut egraph = EGraph::new(MyAnalysis);
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

        let mut egraph = EGraph::new(MyAnalysis);
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
}
