#![cfg(feature = "tvm")]

use std::path::PathBuf;

use egg::EGraph;
use glenside::language::{MyAnalysis, MyAnalysisData};

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

    let (expr, shapes_vec, dtypes_vec, _) = glenside::language::from_relay::from_relay(
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
}
