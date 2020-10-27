#![cfg(feature = "cplex")]
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
use rplex::Variable;
use rplex::VariableType;
use rplex::{var, Constraint, ConstraintType, Env, Problem, VariableValue, WeightedVariable};

use crate::language::{Language, MyAnalysis};

type EGraph = egg::EGraph<Language, MyAnalysis>;

/// Thin wrapper over [`lp_modeler::LpProblem`].
pub struct EGraphLpProblem<'a> {
    pub egraph: &'a EGraph,
    pub problem: Problem<'a>,
    pub bq_vars: HashMap<Id, usize>,
    pub bn_vars: HashMap<&'a Language, usize>,
    pub topo_sort_vars: HashMap<Id, usize>,
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
    /// Filtering function which determines whether this node should be
    /// extractable or not.
    fn filter_enode(enode: &Language, _egraph: &EGraph) -> bool {
        match enode {
            Language::CartesianProduct(_) => panic!(),
            Language::MapDotProduct(_) => panic!(),
            Language::Slice(_) => panic!(),
            Language::Concatenate(_) => panic!(),
            Language::ElementwiseAdd(_) => panic!(),
            Language::BsgSystolicArray(_) => panic!(),
            Language::SystolicArray(_) => true,
            Language::SystolicArrayWithBlocking(_) => true,
            Language::AccessWindows(_) => false,
            Language::ShapeOf(_) => panic!(),
            Language::SliceShape(_) => panic!(),
            Language::ShapeInsertAxis(_) => panic!(),
            Language::ShapeRemoveAxis(_) => panic!(),
            Language::Access(_) => true,
            Language::AccessTranspose(_) => true,
            Language::AccessCartesianProduct(_) => false,
            Language::Compute(_) => false,
            Language::AccessReshape(_) => true,
            Language::AccessFlatten(_) => true,
            Language::Shape(_) => true,
            Language::List(_) => true,
            Language::AccessShape(_) => false,
            Language::AccessSlice(_) => true,
            Language::AccessConcatenate(_) => false,
            Language::AccessPair(_) => false,
            Language::AccessShiftRight(_) => false,
            Language::AccessTensor(_) => true,
            Language::AccessPad(_) => true,
            Language::AccessSqueeze(_) => true,
            Language::AccessInsertAxis(_) => true,
            Language::AccessBroadcast(_) => true,
            Language::AccessLiteral(_) => true,
            Language::Literal(_) => true,
            Language::RelayOperatorCall(_) => true,
            Language::Usize(_) => true,
            Language::NotNanFloat64(_) => true,
            Language::RelayOperator(_) => true,
            Language::PadType(_) => true,
            Language::Symbol(_) => true,
            Language::ComputeType(_) => false,
            Language::MoveAxis(_) => panic!(),
        }
    }

    let mut problem = Problem::new(&env, name).unwrap();

    // Variables representing each class
    let mut bq_vars = HashMap::default();
    // Variables representing each enode
    let mut bn_vars = HashMap::default();
    // Variables for topographically sorting selected eclasses, to ensure there are no loops.
    let mut topo_sort_vars = HashMap::new();

    // Calculating this in the loop was slowing things down a LOT! Is the
    // conversion to f64 slow?
    let number_of_classes_f64 = egraph.number_of_classes() as f64;
    // Create all of the variables
    for eclass in egraph.classes() {
        {
            let bq_name = format!("bq_{}", eclass.id);
            let bq_var = var!(bq_name -> 1.0 as Binary);
            let column_index = problem.add_variable(bq_var).unwrap();
            assert!(!bq_vars.contains_key(&eclass.id));
            bq_vars.insert(eclass.id, column_index);
        }

        {
            let topo_sort_var_name = format!("topo_sort_{}", eclass.id);
            // TODO(@gussmith23) the `as f64` thing here is potentially a bug
            let topo_sort_var = Variable::new(
                VariableType::Integer,
                1.0,
                0.0,
                number_of_classes_f64,
                topo_sort_var_name,
            );
            let column_index = problem.add_variable(topo_sort_var).unwrap();
            assert!(!topo_sort_vars.contains_key(&eclass.id));
            topo_sort_vars.insert(eclass.id, column_index);
        }

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

        // We only allow the extraction of certain nodes.
        let nodes = eclass
            .nodes
            .iter()
            .filter(|node| filter_enode(node, egraph))
            .collect::<Vec<_>>();

        if nodes.is_empty() {
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
            for bn in nodes.iter().map(|node| bn_vars.get(node).unwrap()) {
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

        // If an enode is selected, then its children eclass's topological sort
        // variables must be strictly less than this eclass's topological sort
        // variable.
        // The constraint is:
        // For each eclass i, for each enode n in egraph[i], for each child
        // eclass j of enode n:
        // topo_var[i] >= topo_var[j] + 1 if n is selected
        // === topo_var[i] + some_large_number*(1-bn_vars[n]) >= topo_var[j] + 1
        // === topo_var[i] + some_large_number*(1-bn_vars[n]) - topo_var[j] >= 1
        // === topo_var[i] + some_large_number - some_large_number*bn_vars[n] - topo_var[j] >= 1
        // === topo_var[i] - some_large_number*bn_vars[n] - topo_var[j] >= 1 - some_large_number
        // some_large_number, in this case, can just be num_classes
        let this_eclass_topo_sort_var = topo_sort_vars.get(&eclass.id).unwrap();
        for node in eclass.nodes.iter() {
            let bn = bn_vars.get(&node).unwrap();
            for child_eclass in node.children().iter() {
                let child_eclass_topo_sort_var = topo_sort_vars.get(child_eclass).unwrap();
                // TODO(@gussmith23) potential bug
                let large_number = number_of_classes_f64;
                let mut con = Constraint::new(
                    ConstraintType::GreaterThanEq,
                    1.0 - large_number,
                    format!("topo sort {}", child_eclass),
                );
                con.add_wvar(WeightedVariable::new_idx(*this_eclass_topo_sort_var, 1.0));
                con.add_wvar(WeightedVariable::new_idx(*bn, -large_number));
                con.add_wvar(WeightedVariable::new_idx(*child_eclass_topo_sort_var, -1.0));
                problem.add_constraint(con).unwrap();
            }
        }
    }

    EGraphLpProblem {
        egraph,
        problem,
        bq_vars,
        bn_vars,
        topo_sort_vars,
    }
}

pub fn into_recexpr(
    egraph_lp_problem: &EGraphLpProblem,
    results: &Vec<VariableValue>,
) -> RecExpr<Language> {
    // Use the values assigned to the topological sorting variables to generate
    // the topological sort.
    let mut eclasses_in_topological_order = egraph_lp_problem
        .topo_sort_vars
        .iter()
        // Filter out any eclasses that weren't selected.
        .filter(|&(&eclass_id, _column_index): &(&Id, &usize)| {
            // Get the bq variable for this eclass (i.e. get its column index)
            // and use that to index into the solution.
            let bq_column_index = egraph_lp_problem.bq_vars[&eclass_id];
            match results[bq_column_index] {
                // We filter out this eclass if this bq variable indicates that
                // this eclass wasn't selected.
                VariableValue::Binary(b) => b,
                _ => panic!(),
            }
        })
        .collect::<Vec<_>>();
    // Finally, sort by variable value.
    eclasses_in_topological_order.sort_unstable_by_key(
        |&(_eclass_id, &column_index): &(&Id, &usize)| match results[column_index] {
            VariableValue::Integer(i) => i,
            _ => panic!(),
        },
    );

    let mut old_id_to_new_id_map = HashMap::new();
    let mut expr = RecExpr::default();

    for (&id, _column_index) in eclasses_in_topological_order.iter() {
        // This id should be selected.
        debug_assert_eq!(
            match results[*egraph_lp_problem.bq_vars.get(&id).unwrap()] {
                VariableValue::Binary(b) => b,
                _ => panic!(),
            },
            true
        );

        // Find a variant of this eclass that's selected.
        let variants = egraph_lp_problem.egraph[id]
            .nodes
            .iter()
            .filter(
                |node| match results[*egraph_lp_problem.bn_vars.get(node).unwrap()] {
                    VariableValue::Binary(b) => b,
                    _ => panic!(),
                },
            )
            .collect::<Vec<_>>();
        debug_assert!(variants.len() > 0);

        // TODO(@gussmith23) This isn't always true and/or it doesn't matter
        // It may or may not be true. A minimal solution to the ILP problem will
        // likely ensure that one node is extracted per node. However, it also
        // doesn't matter for us; if a node is extracted, then the ILP problem
        // will guarantee that its dependent eclasses are extracted. So this
        // check truly just exists because of my own curiosity...if it fails, it
        // shouldn't actually break anything, other than my hypothesis.
        debug_assert_eq!(variants.len(), 1);

        let selected_variant = variants[0];

        // The selected enode, but we convert its children to use the IDs in the
        // new expression.
        let converted_node = selected_variant.clone().map_children(|id| {
            *old_id_to_new_id_map
                .get(&id)
                .unwrap_or_else(|| panic!("id {} in enode {:?} not found!", id, selected_variant))
        });

        let new_id = expr.add(converted_node);

        assert!(
            old_id_to_new_id_map.insert(id, new_id).is_none(),
            "This id was already in the map!"
        );
    }

    expr
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    // TODO(@gussmith23) This test doesn't have to depend on running CPLEX.
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

        let out_expr = into_recexpr(&model, &result.variables);

        // This is an odd way to check expression equality, but normal equality
        // will fail if the underlying ids are different!
        assert_eq!(expr.pretty(80), out_expr.pretty(80));
    }
}
