#![cfg(feature = "tvm")]

use std::{collections::HashMap, path::PathBuf, str::FromStr};

use egg::{
    rewrite, CostFunction, EGraph, ENodeOrVar, Extractor, Id, Language as LanguageTrait, Pattern,
    RecExpr, Runner, Searcher, Var,
};
use glenside::language::{Language, MyAnalysis, MyAnalysisData};

/// Importing LSTM to Glenside.
///
/// LSTM is a good example of where multi-patterns in egg would be useful. LSTMs
/// have multiple outputs which (at least in the Relay definition that I'm
/// using) which don't necessarily all appear in a tuple together at the end.
/// This means we can't match on all the outputs at the same time, as there's no
/// single expression which represents the whole LSTM.
#[test]
fn lstm_relay_to_glenside() {
    test_logger::ensure_env_logger_initialized();

    let filename = PathBuf::from(format!(
        "{}/models/lstm-for-pldi.relay",
        env!("CARGO_MANIFEST_DIR")
    ));
    let relay = std::fs::read_to_string(&filename).unwrap();

    let module = tvm::ir::module::IRModule::parse("", relay).unwrap();

    let (expr, shapes_vec, dtypes_vec) = glenside::language::from_relay::from_relay(
        &module,
        false,
        &vec![
            glenside::language::RelayOperator::RelaySigmoid,
            glenside::language::RelayOperator::RelayTanh,
            glenside::language::RelayOperator::RelayLogSoftmax,
            glenside::language::RelayOperator::RelayAdd,
        ],
    );

    let mut egraph = EGraph::new(MyAnalysis {
        name_to_shape: shapes_vec.iter().cloned().collect(),
        name_to_dtype: dtypes_vec.iter().cloned().collect(),
    });

    let id = egraph.add_expr(&expr);
    egraph.rebuild();

    // Check that the types match the expected Relay types.
    match &egraph[id].data {
        MyAnalysisData::Tuple(v) => match v.as_slice() {
            [MyAnalysisData::AccessPattern(a), MyAnalysisData::Tuple(t)] => {
                assert_eq!(a.as_vec(), vec![350, 33278]);
                match t.as_slice() {
                    [MyAnalysisData::Tuple(t0), MyAnalysisData::Tuple(t1)] => {
                        assert_eq!(t0.len(), 0);
                        assert_eq!(t1.len(), 0);
                    }
                    _ => panic!(),
                }
            }
            _ => panic!(),
        },
        _ => panic!(),
    }

    // Build the pattern for LSTM.
    let pattern = {
        let filename = PathBuf::from(format!(
            "{}/models/lstm-for-pldi-pattern.relay",
            env!("CARGO_MANIFEST_DIR")
        ));
        let relay = std::fs::read_to_string(&filename).unwrap();
        let module = tvm::ir::module::IRModule::parse("", relay).unwrap();

        // The pattern in the Glenside language.
        let (orig_pattern, _, _) = glenside::language::from_relay::from_relay(
            &module,
            false,
            // Has to stay the same as the list above...
            &vec![
                glenside::language::RelayOperator::RelaySigmoid,
                glenside::language::RelayOperator::RelayTanh,
                glenside::language::RelayOperator::RelayLogSoftmax,
                glenside::language::RelayOperator::RelayAdd,
            ],
        );

        let pattern_ast = RecExpr::from(
            orig_pattern
                .as_ref()
                .iter()
                .map(|enode| {
                    // We have a single Var in this pattern: it's the "%x"
                    // argument to the pattern. In the pattern compiled to
                    // Glenside, it looks like (access-tensor x).
                    if let crate::Language::AccessTensor(id) = enode {
                        if let crate::Language::Symbol(v) = &orig_pattern[*id] {
                            if v == "x" {
                                return ENodeOrVar::Var(Var::from_str("?x".into()).unwrap());
                            }
                        }
                    }
                    // Construct the ENode-type node in the pattern AST by first
                    // recursively converting the children of this node.
                    ENodeOrVar::ENode(enode.clone())
                })
                .collect::<Vec<_>>(),
        );

        // Here, we don't use any Vars. This means we won't bind anything with
        // this pattern, BUT the pattern should be much faster according to Max.
        // let pattern_ast = RecExpr::from(
        //     orig_pattern
        //         .as_ref()
        //         .iter()
        //         .map(|enode| ENodeOrVar::ENode(enode.clone()))
        //         .collect::<Vec<_>>(),
        // );

        Pattern::from(pattern_ast)
    };

    assert_eq!(pattern.search(&egraph).len(), 1);

    let rewrite = rewrite!("flex-lstm"; 
        { pattern } => "(accelerator-call flex-lstm ?x hidden0 hidden1 rnn_weight_ih_l0 rnn_weight_hh_l0 rnn_bias_ih_l0 rnn_bias_hh_l0)");

    let runner = Runner::default().with_egraph(egraph).run(vec![&rewrite]);

    let matches = "
     (accelerator-call flex-lstm ?x hidden0 hidden1 rnn_weight_ih_l0 rnn_weight_hh_l0 rnn_bias_ih_l0 rnn_bias_hh_l0)"
        .parse::<Pattern<_>>()
        .unwrap()
        .search(&runner.egraph);
    assert_eq!(matches.len(), 1);

    struct Cost {
        memo: HashMap<Id, usize>,
    }
    impl CostFunction<Language> for Cost {
        type Cost = usize;

        fn cost<C>(&mut self, enode: &Language, mut costs: C) -> Self::Cost
        where
            C: FnMut(egg::Id) -> Self::Cost,
        {
            enode.fold(1, |sum, id| {
                usize::saturating_add(sum, *self.memo.entry(id).or_insert(costs(id)))
            })
        }
    }

    let (cost, _expr) = Extractor::new(
        &runner.egraph,
        Cost {
            memo: HashMap::default(),
        },
    )
    .find_best(id);

    assert!(cost < 500);
}
