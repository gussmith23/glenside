//! Integer Linear Programming-based egraph extractor for Glenside
//!
//! Neural networks expressed in Glenside are so large that the standard
//! [`egg::Extractor`] cannot be used; the [`egg::CostFunction`]s end up
//! overflowing [`usize::Max`]! This is a problem of common subexpressions
//! appearing many, many times. For example, when `access-slice` operators slice
//! the same tensor multiple times, they replicate the sliced tensor's
//! subexpression each time, blowing up the size of the overall expression
//! significantly.
//!
//! ILP extraction (and other complex extraction methods) have generally been
//! the answer for these types of extraction problems in egg.
//!
//! Our implementation will draw heavily from Remy Wang's [SPORES
//! paper](https://arxiv.org/abs/2002.07951) and [`warp`
//! repository](https://github.com/wormhole-optimization/warp/blob/d7db4a89ec47803bc2e7729946ca3810b6fb1d03/src/extract.rs).

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::Hash;
use std::hash::Hasher;

use bimap::BiMap;
use egg::RecExpr;
use egg::{Id, Language as LangugeTrait};
use lp_modeler::dsl::LpBinary;
use lp_modeler::dsl::LpObjective;
use lp_modeler::dsl::LpOperations;
use lp_modeler::dsl::LpProblem;

use crate::language::Language;
use crate::language::MyAnalysis;

type EGraph = egg::EGraph<Language, MyAnalysis>;

/// Thin wrapper over [`lp_modeler::LpProblem`].
pub struct EGraphLpProblem<'a> {
    pub egraph: &'a EGraph,
    pub problem: LpProblem,
    /// [`BiMap`] from eclass ids to variable names.
    pub bq_names: BiMap<Id, String>,
    /// [`BiMap`] from enodes to variable names.
    pub bn_names: BiMap<&'a Language, String>,
}

/// From an egraph, create an LP model with a few useful base constraints
///
/// Gives a variable to each eclass and each enode.
///
/// Code taken from Remy Wang's [`warp`
/// repository](https://github.com/wormhole-optimization/warp/blob/d7db4a89ec47803bc2e7729946ca3810b6fb1d03/src/extract.rs).
pub fn create_generic_egraph_lp_model<'a>(
    egraph: &'a EGraph,
    roots: &[Id],
    name: &'static str,
    objective: LpObjective,
) -> EGraphLpProblem<'a> {
    let mut problem = LpProblem::new(name, objective);

    // Variables representing each class
    let mut bq_names = BiMap::default();
    // Variables representing each enode
    let mut bn_names = BiMap::default();

    for eclass in egraph.classes() {
        bq_names
            .insert_no_overwrite(eclass.id, format!("bq_{}", eclass.id))
            .unwrap();

        for enode in eclass.nodes.iter() {
            let mut s = DefaultHasher::new();
            enode.hash(&mut s);
            let bn_name = "bn_".to_owned() + &s.finish().to_string();
            bn_names.insert_no_overwrite(enode, bn_name).unwrap();
        }
    }

    // All roots must be chosen.
    for id in roots {
        let root_var = LpBinary::new(bq_names.get_by_left(id).unwrap().as_str());
        problem += (0 + &root_var).equal(1);
    }

    for eclass in egraph.classes() {
        let bq = LpBinary::new(bq_names.get_by_left(&eclass.id).unwrap());

        if eclass.nodes.is_empty() {
            // Can't extract if this eclass has no variants to be extracted.
            problem += (0 + bq).equal(0);
        } else {
            // bq => OR bn
            // That is, if an eclass is selected, at least one of its variants
            // is selected.
            let mut expr = 1 - bq;
            for bn in eclass
                .nodes
                .iter()
                .map(|node| LpBinary::new(bn_names.get_by_left(&node).unwrap()))
            {
                expr = expr + bn;
            }
            problem += expr.ge(1);
        }

        for node in eclass.nodes.iter() {
            let bn = LpBinary::new(bn_names.get_by_left(&node).unwrap());
            for eclass in node.children().iter() {
                let bq = LpBinary::new(bq_names.get_by_left(eclass).unwrap());
                problem += (1 - &bn + bq).ge(1);
            }
        }
    }

    EGraphLpProblem {
        egraph,
        problem,
        bq_names,
        bn_names,
    }
}

pub fn into_recexpr(
    egraph_lp_problem: &EGraphLpProblem,
    results: &HashMap<String, f32>,
    roots: &[Id],
) -> RecExpr<Language> {
    /// Adds an eclass to the worklist, making sure the eclass's children go on
    /// the worklist first.
    fn make_worklist(
        egraph_lp_problem: &EGraphLpProblem,
        results: &HashMap<String, f32>,
        id: Id,
        worklist: &mut Vec<Id>,
    ) {
        fn add_to_worklist(id: Id, worklist: &mut Vec<Id>) {
            if !worklist.contains(&id) {
                worklist.push(id);
            }
        }

        // This id should be selected.
        assert_eq!(
            *results
                .get(egraph_lp_problem.bq_names.get_by_left(&id).unwrap())
                .unwrap(),
            1.0
        );

        // Find a variant of this eclass that's selected.
        let selected_variant = egraph_lp_problem.egraph[id]
            .nodes
            .iter()
            .find(|node| {
                *results
                    .get(egraph_lp_problem.bn_names.get_by_left(node).unwrap())
                    .unwrap()
                    == 1.0
            })
            .unwrap();

        // Build the worklist for the children
        for child in selected_variant.children() {
            make_worklist(egraph_lp_problem, results, *child, worklist);
        }

        // Add ourselves to worklist.
        add_to_worklist(id, worklist);
    }

    let mut worklist = Vec::default();

    for root in roots {
        make_worklist(egraph_lp_problem, results, *root, &mut worklist);
    }

    // Maps old ids to new ids
    let mut new_ids: HashMap<Id, Id> = HashMap::default();
    let mut expr = RecExpr::default();
    for id in worklist {
        // This id should be selected.
        assert_eq!(
            *results
                .get(egraph_lp_problem.bq_names.get_by_left(&id).unwrap())
                .unwrap(),
            1.0
        );

        // Find a variant of this eclass that's selected.
        // TODO(@gussmith23) We're repeating work here!
        // TODO(@gussmith23) Potential bug; do they find the same node?
        let selected_variant = egraph_lp_problem.egraph[id]
            .nodes
            .iter()
            .find(|node| {
                *results
                    .get(egraph_lp_problem.bn_names.get_by_left(node).unwrap())
                    .unwrap()
                    == 1.0
            })
            .unwrap();

        let converted_node = selected_variant
            .clone()
            .map_children(|id| *new_ids.get(&id).unwrap());
        let new_id = expr.add(converted_node);
        assert!(!new_ids.contains_key(&id));
        new_ids.insert(id, new_id);
    }

    expr
}

#[cfg(test)]
mod tests {
    use super::*;
    use lp_modeler::solvers::GurobiSolver;
    use lp_modeler::solvers::SolverTrait;
    use std::str::FromStr;

    #[test]
    fn extract_trivial() {
        let shape = vec![1, 20, 300, 3];
        let expr = RecExpr::from_str(format!("(access (access-tensor t) 0)",).as_str()).unwrap();
        let mut map = HashMap::default();
        map.insert("t".to_string(), shape.clone());
        let mut egraph = EGraph::new(MyAnalysis { name_to_shape: map });
        let id = egraph.add_expr(&expr);

        let model =
            create_generic_egraph_lp_model(&egraph, &[id], "trivial", LpObjective::Minimize);
        let solver = GurobiSolver::new();
        let result = solver.run(&model.problem);

        let var_values = match result.unwrap() {
            (lp_modeler::solvers::Status::Optimal, var_values) => var_values,
            _ => panic!(),
        };

        let out_expr = into_recexpr(&model, &var_values, &[id]);

        assert_eq!(expr, out_expr);
    }
}
