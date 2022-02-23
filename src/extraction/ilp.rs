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

use egg::{Id, Language as LanguageTrait};
use rplex::Variable;
use rplex::VariableType;
use rplex::{var, Constraint, ConstraintType, Env, Problem, VariableValue, WeightedVariable};

use crate::language::{Language, MyAnalysis};

type EGraph = egg::EGraph<Language, MyAnalysis>;

pub fn filter_by_enode_type(enode: &Language, _eclass_id: Id, _egraph: &EGraph) -> bool {
    if match enode {
        Language::FlexASRMaxPool(_) => todo!(),
        Language::ConstructTuple(_)
        | Language::TupleGetItem(_) => todo!(),

                // Things we should never see.
                    Language::ShapeOf(_)
                    | Language::SliceShape(_)
                    | Language::ShapeInsertAxis(_)
                    | Language::ShapeRemoveAxis(_) => panic!(),

                // Things that should always pass through.
                Language::SystolicArray(_)
                    | Language::SystolicArrayWithBlocking(_)
            | Language::SystolicArrayConv2dNchwOihwWithBlocking(_)
            | Language::SystolicArrayConv2dNhwcHwioWithBlocking(_)
            | Language::SystolicArrayConv2dIm2colNchwOihwWithBlocking(_)
            | Language::SystolicArrayConv2dIm2colNhwcHwioWithBlocking(_)
            | Language::AcceleratorCall(_)
            | Language::AcceleratorFunc(_)
                    | Language::Literal(_)
                    | Language::RelayOperatorCall(_)
            | Language::RelayActivationLayout(_)
            | Language::RelayKernelLayout(_)
                    | Language::Usize(_)
                    | Language::Int32(_)
                    | Language::NotNanFloat64(_)
                    | Language::RelayOperator(_)
                    | Language::Symbol(_) => true,

                // Things I'm not sure about.
                Language::Shape(_) | Language::List(_) | Language::AccessTensor(_) => true,

                // Things that we allow to pass through for now, but shouldn't
                // in the future (once we implement things like memory
                // constructs).
                Language::AccessWindows(_)
                    | Language::Access(_)
                    | Language::AccessTranspose(_)
                    | Language::AccessReshape(_)
                    | Language::AccessFlatten(_)
                    | Language::AccessShape(_)
                // Concatenate needed for grouped convs
                    | Language::AccessConcatenate(_)
                // Slice needed for slice-pad rewrite, grouped convs
                    | Language::AccessSlice(_)
                    | Language::AccessPad(_)
                    | Language::PadType(_)
                    | Language::AccessSqueeze(_)
                    | Language::AccessInsertAxis(_)
                    | Language::AccessBroadcast(_)
                    | Language::ConstantTensor(_)
                    | Language::AccessLiteral(_) => true,

                // Things that should never pass through.
                Language::Compute(_)
                    | Language::ComputeType(_)
                    | Language::AccessCartesianProduct(_)
                    | Language::AccessPair(_)
                    | Language::AccessShiftRight(_) => false,

            }
        == false
    {
        return false;
    }

    true
}

/// Filtering function that drops enodes if this eclass contains
/// obviously-extractable constructs.
pub fn filter_obviously_less_preferable_nodes(
    enode: &Language,
    eclass_id: Id,
    egraph: &EGraph,
) -> bool {
    fn is_obviously_extractable(enode: &Language) -> bool {
        match enode {
            Language::FlexASRMaxPool(_) => todo!(),
            Language::ConstructTuple(_) | Language::TupleGetItem(_) => todo!(),

            // Things we should never see.
            Language::ShapeOf(_)
            | Language::SliceShape(_)
            | Language::ShapeInsertAxis(_)
            | Language::ShapeRemoveAxis(_) => panic!(),

            Language::SystolicArray(_)
            | Language::SystolicArrayConv2dNchwOihwWithBlocking(_)
            | Language::SystolicArrayConv2dNhwcHwioWithBlocking(_)
            | Language::SystolicArrayConv2dIm2colNchwOihwWithBlocking(_)
            | Language::SystolicArrayConv2dIm2colNhwcHwioWithBlocking(_)
            | Language::RelayOperatorCall(_)
            | Language::RelayActivationLayout(_)
            | Language::RelayKernelLayout(_)
            | Language::AcceleratorCall(_)
            | Language::AcceleratorFunc(_)
            | Language::SystolicArrayWithBlocking(_) => true,

            Language::Shape(_)
            | Language::List(_)
            | Language::AccessTensor(_)
            | Language::AccessWindows(_)
            | Language::Literal(_)
            | Language::Usize(_)
            | Language::Int32(_)
            | Language::NotNanFloat64(_)
            | Language::RelayOperator(_)
            | Language::Symbol(_)
            | Language::Access(_)
            | Language::AccessTranspose(_)
            | Language::AccessReshape(_)
            | Language::AccessFlatten(_)
            | Language::AccessShape(_)
            | Language::AccessConcatenate(_)
            | Language::AccessSlice(_)
            | Language::AccessPad(_)
            | Language::PadType(_)
            | Language::AccessSqueeze(_)
            | Language::AccessInsertAxis(_)
            | Language::AccessBroadcast(_)
            | Language::AccessLiteral(_)
            | Language::Compute(_)
            | Language::ComputeType(_)
            | Language::AccessCartesianProduct(_)
            | Language::AccessPair(_)
            | Language::ConstantTensor(_)
            | Language::AccessShiftRight(_) => false,
        }
    }

    // If this enode's set of siblings contains something that's "obviously
    // extractable" e.g. a systolic array, then remove this node if it's not one
    // of the obviously extractable things.
    if egraph[eclass_id].nodes.iter().any(is_obviously_extractable) {
        return is_obviously_extractable(enode);
    }

    true
}

/// Filtering function which filters out nodes which form simple loops. Returns
/// false if and only if this node is an access-flatten which is in an eclass by
/// itself, and if the eclass it points to has an access-reshape node that
/// points right back to this node's eclass.
pub fn filter_useless_access_flattens(enode: &Language, eclass_id: Id, egraph: &EGraph) -> bool {
    // Return early if this eclass contains nodes other than just this one.
    let this_node_is_alone = match egraph[eclass_id].nodes.as_slice() {
        [n] if n == enode => true,
        _ => false,
    };
    if !this_node_is_alone {
        return true;
    }

    match enode {
        Language::AccessFlatten(access_flatten_id) => {
            let some_node_points_back =
                egraph[*access_flatten_id]
                    .nodes
                    .iter()
                    .any(|node| match node {
                        Language::AccessReshape([reshape_access_id, _shape_id])
                            if *reshape_access_id == eclass_id =>
                        {
                            true
                        }
                        _ => false,
                    });

            // Filter this out (return false) if a reshape node in
            // `access_flatten_id` points back to this eclass.
            if some_node_points_back {
                false
            } else {
                true
            }
        }
        _ => true,
    }
}

/// Filtering function which filters out useless pad/slice loops
pub fn filter_useless_pad_slice_loops(enode: &Language, eclass_id: Id, egraph: &EGraph) -> bool {
    // Return early if this eclass contains nodes other than just this one.
    let this_node_is_alone = match egraph[eclass_id].nodes.as_slice() {
        [n] if n == enode => true,
        _ => false,
    };
    if !this_node_is_alone {
        return true;
    }

    match enode {
        Language::AccessPad([pad_arg_id, _, _, _, _]) => {
            let some_node_points_back = egraph[*pad_arg_id].nodes.iter().any(|node| match node {
                Language::AccessSlice([slice_arg_id, _, _, _]) if *slice_arg_id == eclass_id => {
                    true
                }
                _ => false,
            });

            // Filter this out (return false) if a slice node in eclass
            // `pad_arg_id` points back to this eclass.
            if some_node_points_back {
                false
            } else {
                true
            }
        }
        _ => true,
    }
}

/// Thin wrapper over [`lp_modeler::LpProblem`].
pub struct EGraphLpProblem<'a> {
    pub egraph: &'a EGraph,
    /// A [`Problem`] which contains all of the variables listed below, plus
    /// constraints over these variables. The user should set the optimization
    /// objective as they see fit, using the variables provided below.
    pub problem: Problem<'a>,
    /// Eclass variables. There is a variable for each eclass in the egraph.
    pub bq_vars: HashMap<Id, usize>,
    /// Enode variables. There is not necessarily a variable for each enode, as
    /// some will have been filtered out as unextractable. See
    /// [`create_generic_egraph_lp_model`]'s `filter_eclass_variants` argument.
    pub bn_vars: HashMap<&'a Language, usize>,
    /// Variables used to ensure that the extracted eclasses can be
    /// topologically sorted.
    pub topo_sort_vars: HashMap<Id, usize>,
}

/// From an egraph, create an LP model with a few useful base constraints
///
/// We create three types of variables: eclass variables, enode variables, and
/// topological sorting variables. See the definition of [`EGraphLpProblem`] for
/// more detail on these variables.
///
/// The `filter_enode` argument is a function which, given a reference to the
/// enode, plus the id of its containing [`EClass`] and a reference to the
/// [`egg::EGraph`] itself, determines whether to create a variable in the ILP
/// problem for this enode. This can be used to filter "useless" enodes.
///
/// Code taken from Remy Wang's [`warp`
/// repository](https://github.com/wormhole-optimization/warp/blob/d7db4a89ec47803bc2e7729946ca3810b6fb1d03/src/extract.rs).
pub fn create_generic_egraph_lp_model<'a>(
    env: &'a Env,
    egraph: &'a EGraph,
    filter_enode: impl Fn(&Language, Id, &EGraph) -> bool,
    roots: &[Id],
    name: &'static str,
) -> EGraphLpProblem<'a> {
    let mut problem = Problem::new(&env, name).unwrap();

    // Variables representing each class
    let mut bq_vars = HashMap::default();
    // Variables representing each enode
    let mut bn_vars = HashMap::default();
    // Variables for topographically sorting selected eclasses, to ensure there are no loops.
    let mut topo_sort_vars = HashMap::new();

    // Calculating this in the loop was slowing things down a LOT! Is the
    // conversion to f64 slow?
    // TODO(@gussmith23) potential bug
    let number_of_classes_f64 = egraph.number_of_classes() as f64;
    // Create all of the variables
    for eclass in egraph.classes() {
        let canonical_id = egraph.find(eclass.id);
        if bq_vars.contains_key(&canonical_id) {
            continue;
        }
        {
            let bq_name = format!("bq_{}", canonical_id);
            let bq_var = var!(bq_name -> 1.0 as Binary);
            let column_index = problem.add_variable(bq_var).unwrap();
            assert!(!bq_vars.contains_key(&canonical_id));
            bq_vars.insert(canonical_id, column_index);
        }

        {
            let topo_sort_var_name = format!("topo_sort_{}", canonical_id);
            // TODO(@gussmith23) the `as f64` thing here is potentially a bug
            let topo_sort_var = Variable::new(
                VariableType::Integer,
                1.0,
                0.0,
                number_of_classes_f64,
                topo_sort_var_name,
            );
            let column_index = problem.add_variable(topo_sort_var).unwrap();
            assert!(!topo_sort_vars.contains_key(&canonical_id));
            topo_sort_vars.insert(canonical_id, column_index);
        }

        // Filter out enodes that the user doesn't want variables for.
        let mut var_count = 0;
        for enode in egraph[canonical_id]
            .nodes
            .iter()
            .filter(|node| filter_enode(node, eclass.id, egraph))
        {
            let mut s = DefaultHasher::new();
            enode.hash(&mut s);
            let bn_name = "bn_".to_owned() + &s.finish().to_string();
            let bn_var = var!(bn_name -> 1.0 as Binary);
            let column_index = problem.add_variable(bn_var).unwrap();
            assert!(!bn_vars.contains_key(&enode));
            bn_vars.insert(enode, column_index);
            var_count += 1;
        }
        assert!(var_count > 0, "No variable selected for eclass {}: {:?}", eclass.id, eclass);
    }

    // All roots must be chosen.
    for id in roots.iter().map(|id| egraph.find(*id)) {
        let column_index = bq_vars.get(&id).unwrap();
        let mut con = Constraint::new(ConstraintType::Eq, 1.0, format!("root constraint {}", id));
        con.add_wvar(WeightedVariable::new_idx(*column_index, 1.0));
        problem.add_constraint(con).unwrap();
    }

    for (id, bq_idx) in bq_vars.iter() {
        // We only allow the extraction of certain nodes. This gets a list of
        // all of ILP variable indices for enode variables and their
        // corresponding enodes, for enodes that passed through the
        // `enode_filter` above.
        let bn_idxs_and_nodes = egraph[*id]
            .nodes
            .iter()
            .filter_map(|node| bn_vars.get(node).and_then(|idx| Some((*idx, node))))
            .collect::<Vec<_>>();

        if bn_idxs_and_nodes.is_empty() {
            // Can't extract if this eclass has no variants to be extracted.
            let mut con = Constraint::new(
                ConstraintType::Eq,
                0.0,
                format!("can't extract eclass {}", id),
            );
            con.add_wvar(WeightedVariable::new_idx(*bq_idx, 1.0));
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
                format!("must select enode for eclass {}", id),
            );
            con.add_wvar(WeightedVariable::new_idx(*bq_idx, -1.0));
            for (bn_idx, _) in &bn_idxs_and_nodes {
                con.add_wvar(WeightedVariable::new_idx(*bn_idx, 1.0));
            }
            problem.add_constraint(con).unwrap();
        }

        // If an eclass is selected, then at least one of its parents must be selected
        // if all its parents are filtered out, then this eclass must not be selected
        let mut available_parents = vec![];
        for p in egraph[*id].parents.iter() {
            if let Some(parent_idx) = bn_vars.get(&p.0) {
                available_parents.push(parent_idx);
            }
        }

        if available_parents.len() == 0 {
            let mut con = Constraint::new(ConstraintType::Eq,
                0.0,
                format!("Disable eclass {} because it doesn't have an available parent", id));
            con.add_wvar(WeightedVariable::new_idx(*bq_idx, 1.0));
            problem.add_constraint(con).unwrap();
            continue;
        } else {
            // bq => OR p_idx for p_idx in bq of eclass parents
            let mut con = Constraint::new(
                ConstraintType::GreaterThanEq,
                0.0,
                format!("Need to choose parents for {}", id),
            );
            con.add_wvar(WeightedVariable::new_idx(*bq_idx, -1.0));
            for p_idx in available_parents.into_iter() {
                con.add_wvar(WeightedVariable::new_idx(*p_idx, 1.0));
            }
            problem.add_constraint(con).unwrap();
        }

        // If an enode is selected, then its child eclasses must be selected.
        // Implemented as
        // -bn + bq >= 0 for each bq
        for (bn_idx, node) in &bn_idxs_and_nodes {
            for eclass_id in node.children().iter().map(|id| egraph.find(*id)) {
                let bq_idx = bq_vars.get(&eclass_id).unwrap();
                let mut con = Constraint::new(
                    ConstraintType::GreaterThanEq,
                    0.0,
                    format!("must select eclass {} if parent enode selected", eclass_id),
                );
                con.add_wvar(WeightedVariable::new_idx(*bn_idx, -1.0));
                con.add_wvar(WeightedVariable::new_idx(*bq_idx, 1.0));
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
        let this_eclass_topo_sort_var = topo_sort_vars.get(id).unwrap();
        for (bn_idx, node) in &bn_idxs_and_nodes {
            for child_eclass_id in node.children().iter().map(|id| egraph.find(*id)) {
                let child_eclass_topo_sort_var = topo_sort_vars.get(&child_eclass_id).unwrap();
                let large_number = number_of_classes_f64;
                let mut con = Constraint::new(
                    ConstraintType::GreaterThanEq,
                    1.0 - large_number,
                    format!("topo sort eclass {}", child_eclass_id),
                );
                con.add_wvar(WeightedVariable::new_idx(*this_eclass_topo_sort_var, 1.0));
                con.add_wvar(WeightedVariable::new_idx(*bn_idx, -large_number));
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

/// Returns an [`EGraph`] with just one enode per eclass, according to the
/// results of the ILP problem. I could use a [`RecExpr`] here, but all of the
/// post-processing we do (i.e. codegen) prefers to have an [`EGraph`] with our
/// analysis attached. Also returns a map, mapping the [`Id`]s in the old egraph
/// to their [`Id`]s in the new expression.
///
/// Actually adds everything to `egraph`, which can be passed in as a new, empty
/// egraph initialized with the correct analysis.
///
/// TODO(@gussmith23) Terrible naming. Not related to extraction.
pub fn extract_single_expression(
    egraph_lp_problem: &EGraphLpProblem,
    results: &Vec<VariableValue>,
    egraph: EGraph,
) -> (EGraph, HashMap<Id, Id>) {
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
    let mut expr = egraph;

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
            // Filter out enodes that weren't selected.
            .filter(|node| {
                egraph_lp_problem
                    .bn_vars
                    .get(node)
                    // If we get a valid index (i.e. there's a variable for this
                    // enode, because not all enodes have variables!) then use
                    // it to look up the result value.
                    .and_then(|&bn_idx: &usize| match results[bn_idx] {
                        VariableValue::Binary(b) => Some(b),
                        _ => panic!(),
                    })
                    .or(Some(false))
                    .unwrap()
            })
            .collect::<Vec<_>>();
        debug_assert!(variants.len() > 0);

        // TODO(@gussmith23) This isn't always true and/or it doesn't matter
        // It may or may not be true. A minimal solution to the ILP problem will
        // likely ensure that one node is extracted per node. However, it also
        // doesn't matter for us; if a node is extracted, then the ILP problem
        // will guarantee that its dependent eclasses are extracted. So this
        // check truly just exists because of my own curiosity...if it fails, it
        // shouldn't actually break anything, other than my hypothesis.
        debug_assert!(variants.len() == 1, "{:?}", variants);

        let selected_variant = variants[0];

        // The selected enode, but we convert its children to use the IDs in the
        // new expression.
        let converted_node = selected_variant.clone().map_children(|id| {
            *old_id_to_new_id_map
                .get(&egraph_lp_problem.egraph.find(id.clone()))
                .unwrap_or_else(|| panic!("id {} in enode {:?} not found!", id, selected_variant))
        });

        let new_id = expr.add(converted_node);

        assert!(
            old_id_to_new_id_map.insert(id, new_id).is_none(),
            "This id was already in the map!"
        );
    }

    (expr, old_id_to_new_id_map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use egg::RecExpr;
    use std::str::FromStr;

    // TODO(@gussmith23) This test doesn't have to depend on running CPLEX.
    #[test]
    fn extract_trivial() {
        let shape = vec![1, 20, 300, 3];
        let expr = RecExpr::from_str(format!("(access (access-tensor t) 0)",).as_str()).unwrap();
        let mut map = HashMap::default();
        map.insert("t".to_string(), shape.clone());
        let mut egraph = EGraph::new(MyAnalysis {
            name_to_shape: map.clone(),
        });
        let id = egraph.add_expr(&expr);

        let env = Env::new().unwrap();
        let mut model =
            create_generic_egraph_lp_model(&env, &egraph, |_, _, _| true, &[id], "trivial");
        let result = model.problem.solve().unwrap();

        let (out_expr, _old_id_to_new_id_map) = extract_single_expression(
            &model,
            &result.variables,
            EGraph::new(MyAnalysis { name_to_shape: map }),
        );

        for eclass in out_expr.classes() {
            assert_eq!(out_expr[eclass.id].nodes.len(), 1);
        }

        // This is an odd way to check expression equality, but normal equality
        // will fail if the underlying ids are different!
        //assert_eq!(expr.pretty(80), out_expr.pretty(80));
        // TODO(@gussmith23) Test incomplete
        // Can't check for equality easily this way anymore. Need a better way
        // to check.
    }
}
