use crate::language::{Language, MyAnalysis};
use egg::{
    Analysis, CostFunction, EClass, EGraph, Extractor, Id, Language as LanguageTrait, Pattern,
    RecExpr, Searcher,
};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;

// pub struct Extractor<'a, CF: CostFunction<L, N>, L: LanguageTrait, N: Analysis<L>> {
//     cost_function: CF,
//     costs: HashMap<Id, CF::Cost>,
//     egraph: &'a EGraph<L, N>,
// }

// pub trait CostFunction<L: LanguageTrait, A: Analysis<L>> {
//     type Cost: PartialOrd + Debug + Clone;

//     fn cost<C>(&mut self, enode: &L, egraph: &EGraph<L, A>, costs: C) -> Self::Cost
//     where
//         C: FnMut(Id) -> Self::Cost;

//     fn cost_rec(&mut self, expr: &RecExpr<L>, egraph: &EGraph<L, A>) -> Self::Cost {
//         let mut costs: HashMap<Id, Self::Cost> = HashMap::default();
//         for (i, node) in expr.as_ref().iter().enumerate() {
//             let cost = self.cost(node, egraph, |i| costs[&i].clone());
//             costs.insert(i as Id, cost);
//         }
//         let last_id = expr.as_ref().len() as Id - 1;
//         costs[&last_id].clone()
//     }
// }

// fn cmp<T: PartialOrd>(a: &Option<T>, b: &Option<T>) -> Ordering {
//     // None is high
//     match (a, b) {
//         (None, None) => Ordering::Equal,
//         (None, Some(_)) => Ordering::Greater,
//         (Some(_), None) => Ordering::Less,
//         (Some(a), Some(b)) => a.partial_cmp(&b).unwrap(),
//     }
// }

// impl<'a, CF, L, N> Extractor<'a, CF, L, N>
// where
//     CF: CostFunction<L, N>,
//     L: LanguageTrait,
//     N: Analysis<L>,
// {
//     pub fn new(egraph: &'a EGraph<L, N>, cost_function: CF) -> Self {
//         let costs = HashMap::default();
//         let mut extractor = Extractor {
//             costs,
//             egraph,
//             cost_function,
//         };
//         extractor.find_costs();

//         extractor
//     }

//     /// Find the cheapest (lowest cost) represented `RecExpr` in the
//     /// given eclass.
//     pub fn find_best(&mut self, eclass: Id) -> (CF::Cost, RecExpr<L>) {
//         let mut expr = RecExpr::default();
//         self.find_best_rec(&mut expr, eclass);
//         let cost = self.cost_function.cost_rec(&expr, self.egraph);
//         (cost, expr)
//     }

//     fn find_best_rec(&mut self, expr: &mut RecExpr<L>, eclass: Id) -> Id {
//         let eclass = self.egraph.find(eclass);

//         let best_node = self.egraph[eclass]
//             .iter()
//             .min_by(|a, b| {
//                 let a = self.node_total_cost(a);
//                 let b = self.node_total_cost(b);
//                 cmp(&a, &b)
//             })
//             .expect("eclass shouldn't be empty");

//         let node = best_node
//             .clone()
//             .map_children(|child| self.find_best_rec(expr, child));
//         expr.add(node)
//     }

//     fn node_total_cost(&mut self, node: &L) -> Option<CF::Cost> {
//         if node.children().iter().all(|id| self.costs.contains_key(id)) {
//             let costs = &self.costs;
//             Some(self.cost_function.cost(&node, self.egraph, |i| costs[&i].clone()))
//         } else {
//             None
//         }
//     }

//     fn find_costs(&mut self) {
//         let mut did_something = true;
//         while did_something {
//             did_something = false;

//             for class in self.egraph.classes() {
//                 let pass = self.make_pass(class);
//                 match (self.costs.get(&class.id), pass) {
//                     (None, Some(cost)) => {
//                         self.costs.insert(class.id, cost);
//                         did_something = true;
//                     }
//                     (Some(old), Some(new)) if new < *old => {
//                         self.costs.insert(class.id, new);
//                         did_something = true;
//                     }
//                     _ => (),
//                 }
//             }
//         }
//     }

//     fn make_pass(&mut self, eclass: &EClass<L, N::Data>) -> Option<CF::Cost> {
//         eclass
//             .iter()
//             .map(|n| self.node_total_cost(n))
//             .min_by(cmp)
//             .unwrap_or_else(|| panic!("Can't extract, eclass is empty: {:#?}", eclass))
//     }
// }

// how to implement?
// we want to get out all "monolithic" designs, i.e., all designs that can
// be implemented with one size of systolic array.
// So first, find all the different sizes of systolic array.
// Honestly, I think the place to start is actually to extract all possible
// hardware designs. Just have something that does all extractions.
// fn extract_all<A: Analysis<Language>>(egraph: &EGraph<Language, A>) {}

// fn extract_all_eclass_helper<A: Analysis<Language>>(egraph: &EGraph<Language, A>, eclass: Id) {
//     egraph[eclass]
//         .nodes
//         .iter()
//         .map(|enode: &Language| extract_all_enode_helper(egraph, enode));
// }

// fn extract_all_enode_helper<A: Analysis<Language>>(egraph: &EGraph<Language, A>, enode: &Language) {

// }

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
                MyAnalysis::get_usize(subst[&"?rows".parse().unwrap()], egraph),
                MyAnalysis::get_usize(subst[&"?cols".parse().unwrap()], egraph),
            ));
        }
    }

    systolic_arrays
}

// TODO(@gussmith23) Determine if I need this code.
// IT was going to extract all monolithic designs, but it just seems like
// overkill for my purposes right now.
// fn extract_monolithic_designs(
//     systolic_array_configuration: (usize, usize),
//     egraph: &EGraph<Language, MyAnalysis>,
// ) {
// }

// fn extract_monolithic_designs_helper(
//     systolic_array_configuration: (usize, usize),
//     egraph: &EGraph<Language, MyAnalysis>,
//     eclass: Id,
// ) -> Vec<RecExpr<Language>> {
//     egraph[eclass]
//         .nodes
//         .iter()
//         .filter(|enode: &&Language| {
//             // Filter out specific Language nodes.
//             // - Constructs that aren't valid in extraction
//             // - Systolic arrays that are not the right configuration
//             match enode {
//                 &&Language::BsgSystolicArray([rows_id, cols_id, _, _])
//                     if (
//                         MyAnalysis::get_usize(rows_id, egraph),
//                         MyAnalysis::get_usize(cols_id, egraph),
//                     ) == systolic_array_configuration =>
//                 {
//                     true
//                 }
//                 Language::Symbol(_)
//                 | Language::Slice(_)
//                 | Language::Concatenate(_)
//                 | Language::ElementwiseAdd(_) => true,
//                 _ => false,
//             }
//         })
//         .map(|enode: &&Language| match enode {
//             &&Language::Symbol(s) =>  {
//                 let mut expr = RecExpr::default();
//                 let id = expr.add(Language::Symbol(s.clone()));
//                 vec![(id, expr)]
//             }
//             &&Language::BsgSystolicArray([rows_id, cols_id, tensor_0_id, tensor_1_id])
//                 if (
//                     MyAnalysis::get_usize(rows_id, egraph),
//                     MyAnalysis::get_usize(cols_id, egraph),
//                 ) == systolic_array_configuration =>
//             {
//                 use itertools::Itertools;
//                 std::iter::once(extract_monolithic_designs_helper(
//                     systolic_array_configuration,
//                     egraph,
//                     rows_id,
//                 ))
//                 .chain(std::iter::once(extract_monolithic_designs_helper(
//                     systolic_array_configuration,
//                     egraph,
//                     cols_id,
//                 )))
//                 .chain(std::iter::once(extract_monolithic_designs_helper(
//                     systolic_array_configuration,
//                     egraph,
//                     tensor_0_id,
//                 )))
//                 .chain(std::iter::once(extract_monolithic_designs_helper(
//                     systolic_array_configuration,
//                     egraph,
//                     tensor_1_id,
//                 )))
//                     .multi_cartesian_product()
//                     .map(|())
//             }
//             &&Language::Slice([tensor_id, axis_id, low_id, high_id]) => {
//                 use itertools::Itertools;
//                 std::iter::once(extract_monolithic_designs_helper(
//                     systolic_array_configuration,
//                     egraph,
//                     tensor_id,
//                 ))
//                 .chain(std::iter::once(extract_monolithic_designs_helper(
//                     systolic_array_configuration,
//                     egraph,
//                     axis_id,
//                 )))
//                 .chain(std::iter::once(extract_monolithic_designs_helper(
//                     systolic_array_configuration,
//                     egraph,
//                     low_id,
//                 )))
//                 .chain(std::iter::once(extract_monolithic_designs_helper(
//                     systolic_array_configuration,
//                     egraph,
//                     high_id,
//                 )))
//                 .multi_cartesian_product();
//             }
//             &&Language::Concatenate([tensor_0_id, tensor_1_id, axis_id]) => {
//                 use itertools::Itertools;
//                 std::iter::once(extract_monolithic_designs_helper(
//                     systolic_array_configuration,
//                     egraph,
//                     tensor_0_id,
//                 ))
//                 .chain(std::iter::once(extract_monolithic_designs_helper(
//                     systolic_array_configuration,
//                     egraph,
//                     tensor_1_id,
//                 )))
//                 .chain(std::iter::once(extract_monolithic_designs_helper(
//                     systolic_array_configuration,
//                     egraph,
//                     axis_id,
//                 )))
//                 .multi_cartesian_product();
//             }
//             Language::ElementwiseAdd([]) => {}
//             _ => panic!(),
//         })
// }

struct MonolithicCostFunction<'a> {
    systolic_array_configuration: (usize, usize),
    egraph: &'a EGraph<Language, MyAnalysis>,
}
impl CostFunction<Language> for MonolithicCostFunction<'_> {
    type Cost = usize;

    fn cost<C>(&mut self, enode: &Language, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        const INFINITE: usize = 100000;
        match enode {
            &Language::BsgSystolicArray([rows_id, cols_id, _tensor_0_id, _tensor_1_id]) => {
                println!("{:?}", enode);
                assert!(rows_id != 4);
                1
                // INFINITE
            }
            _ => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::language::MyAnalysis;
    use super::*;
    use egg::EGraph;

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
            },
        );

        let (cost, _) = ex.find_best(id);
        assert!(cost > 100000);

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

        let mut ex = Extractor::new(
            &egraph,
            MonolithicCostFunction {
                egraph: &egraph,
                systolic_array_configuration: (32, 64),
            },
        );

        let (cost, expr) = ex.find_best(id);
        assert!(cost < 100000);
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
    fn mwe() {
        let program = "
          (bsg-systolic-array 128 64
           (bsg-systolic-array 64 32
            v-32
            t-32-64
           )
           t-64-128
          )
         "
        .parse()
        .unwrap();

        let mut egraph = EGraph::new(MyAnalysis);
        let id = egraph.add_expr(&program);
        egraph.rebuild();
        println!("{:?}", egraph.dump());

        let mut ex = Extractor::new(
            &egraph,
            MonolithicCostFunction {
                systolic_array_configuration: (16, 128),
                egraph: &egraph,
            },
        );

        let (cost, _) = ex.find_best(id);
        panic!()
    }
}
