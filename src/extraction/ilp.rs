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
use std::hash::{Hash, Hasher};

use egg::{Id, Language as LangugeTrait, RecExpr};
use rplex::{var, Constraint, ConstraintType, Env, Problem, VariableValue, WeightedVariable};

use crate::language::{Language, MyAnalysis};

type EGraph = egg::EGraph<Language, MyAnalysis>;

/// Thin wrapper over [`lp_modeler::LpProblem`].
pub struct EGraphLpProblem<'a> {
    pub egraph: &'a EGraph,
    pub problem: Problem<'a>,
    pub bq_vars: HashMap<Id, usize>,
    pub bn_vars: HashMap<&'a Language, usize>,
}

/// From an egraph, create an LP model with a few useful base constraints
///
/// Gives a variable to each eclass and each enode.
///
/// Code taken from Remy Wang's [`warp`
/// repository](https://github.com/wormhole-optimization/warp/blob/d7db4a89ec47803bc2e7729946ca3810b6fb1d03/src/extract.rs).
pub fn create_generic_egraph_lp_model<'a>(
    env: &'a Env,
    egraph: &'a EGraph,
    roots: &[Id],
    name: &'static str,
) -> EGraphLpProblem<'a> {
    let mut problem = Problem::new(&env, name).unwrap();

    // Variables representing each class
    let mut bq_vars = HashMap::default();
    // Variables representing each enode
    let mut bn_vars = HashMap::default();

    for eclass in egraph.classes() {
        let bq_name = format!("bq_{}", eclass.id);
        let bq_var = var!(bq_name -> 1.0 as Binary);
        let column_index = problem.add_variable(bq_var).unwrap();
        assert!(!bq_vars.contains_key(&eclass.id));
        bq_vars.insert(eclass.id, column_index);

        for enode in eclass.nodes.iter() {
            let mut s = DefaultHasher::new();
            enode.hash(&mut s);
            let bn_name = "bn_".to_owned() + &s.finish().to_string();
            let bn_var = var!(bn_name -> 1.0 as Binary);
            let column_index = problem.add_variable(bn_var).unwrap();
            assert!(!bn_vars.contains_key(&enode));
            bn_vars.insert(enode, column_index);
        }
    }

    // All roots must be chosen.
    for id in roots {
        let column_index = bq_vars.get(id).unwrap();
        let mut con = Constraint::new(ConstraintType::Eq, 1.0, format!("root constraint {}", id));
        con.add_wvar(WeightedVariable::new_idx(*column_index, 1.0));
        problem.add_constraint(con).unwrap();
    }

    for eclass in egraph.classes() {
        let bq_column_index = bq_vars.get(&eclass.id).unwrap();

        if eclass.nodes.is_empty() {
            // Can't extract if this eclass has no variants to be extracted.
            let mut con = Constraint::new(
                ConstraintType::Eq,
                0.0,
                format!("can't extract {}", eclass.id),
            );
            con.add_wvar(WeightedVariable::new_idx(*bq_column_index, 1.0));
            problem.add_constraint(con).unwrap();
        } else {
            // bq => OR bn
            // That is, if an eclass is selected, at least one of its variants
            // is selected.
            // implemented as:
            // -bq + bn ... >= 0
            let mut con = Constraint::new(
                ConstraintType::GreaterThanEq,
                0.0,
                format!("must select enode for {}", eclass.id),
            );
            con.add_wvar(WeightedVariable::new_idx(*bq_column_index, -1.0));
            for bn in eclass.nodes.iter().map(|node| bn_vars.get(&node).unwrap()) {
                con.add_wvar(WeightedVariable::new_idx(*bn, 1.0));
            }
            problem.add_constraint(con).unwrap();
        }

        // If an enode is selected, then its child eclasses must be selected.
        // Implemented as
        // -bn + bq >= 0 for each bq
        for node in eclass.nodes.iter() {
            let bn = bn_vars.get(&node).unwrap();
            for eclass in node.children().iter() {
                let bq = bq_vars.get(eclass).unwrap();
                let mut con = Constraint::new(
                    ConstraintType::GreaterThanEq,
                    0.0,
                    format!("must select eclass {} if parent enode selected", eclass),
                );
                con.add_wvar(WeightedVariable::new_idx(*bn, -1.0));
                con.add_wvar(WeightedVariable::new_idx(*bq, 1.0));
                problem.add_constraint(con).unwrap();
            }
        }
    }

    EGraphLpProblem {
        egraph,
        problem,
        bq_vars,
        bn_vars,
    }
}

pub fn into_recexpr(
    egraph_lp_problem: &EGraphLpProblem,
    results: &Vec<VariableValue>,
    roots: &[Id],
) -> RecExpr<Language> {
    /// Adds an eclass to the worklist, making sure the eclass's children go on
    /// the worklist first.
    fn make_worklist(
        egraph_lp_problem: &EGraphLpProblem,
        results: &Vec<VariableValue>,
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
            match results[*egraph_lp_problem.bq_vars.get(&id).unwrap()] {
                VariableValue::Binary(b) => b,
                _ => panic!(),
            },
            true
        );

        // Find a variant of this eclass that's selected.
        let selected_variant = egraph_lp_problem.egraph[id]
            .nodes
            .iter()
            .find(
                |node| match results[*egraph_lp_problem.bn_vars.get(node).unwrap()] {
                    VariableValue::Binary(b) => b == true,
                    _ => panic!(),
                },
            )
            .unwrap();

        // Build the worklist for the children
        for child in selected_variant.children() {
            make_worklist(egraph_lp_problem, results, *child, worklist);
        }

        // Add ourselves to worklist.
        add_to_worklist(id, worklist);
    }

    let mut worklist = Vec::default();

    println!("Beginning to build worklist");

    for root in roots {
        make_worklist(egraph_lp_problem, results, *root, &mut worklist);
    }

    println!("done building worklist");

    // Maps old ids to new ids
    let mut new_ids: HashMap<Id, Id> = HashMap::default();
    let mut expr = RecExpr::default();
    for id in worklist {
        // This id should be selected.
        assert_eq!(
            match results[*egraph_lp_problem.bq_vars.get(&id).unwrap()] {
                VariableValue::Binary(b) => b,
                _ => panic!(),
            },
            true
        );

        // Find a variant of this eclass that's selected.
        // TODO(@gussmith23) We're repeating work here!
        // TODO(@gussmith23) Potential bug; do they find the same node?
        let selected_variant = egraph_lp_problem.egraph[id]
            .nodes
            .iter()
            .find(
                |node| match results[*egraph_lp_problem.bn_vars.get(node).unwrap()] {
                    VariableValue::Binary(b) => b == true,
                    _ => panic!(),
                },
            )
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
    use std::str::FromStr;

    #[test]
    fn extract_trivial() {
        let shape = vec![1, 20, 300, 3];
        let expr = RecExpr::from_str(format!("(access (access-tensor t) 0)",).as_str()).unwrap();
        let mut map = HashMap::default();
        map.insert("t".to_string(), shape.clone());
        let mut egraph = EGraph::new(MyAnalysis { name_to_shape: map });
        let id = egraph.add_expr(&expr);

        let env = Env::new().unwrap();
        let mut model = create_generic_egraph_lp_model(&env, &egraph, &[id], "trivial");
        let result = model.problem.solve().unwrap();

        let out_expr = into_recexpr(&model, &result.variables, &[id]);

        assert_eq!(expr, out_expr);
    }
}
