use std::{
    collections::HashMap,
    fs::{read_dir, read_to_string},
    path::Path,
    str::FromStr,
};

use approx::AbsDiffEq;
use egg::RecExpr;
use glenside::language::{interpreter::interpret, Language};
use ndarray::ArrayD;
use ndarray_npy::read_npy;

/// This test runs all of the examples in the /glenside-programs directory.
#[test]
fn run_glenside_examples() {
    for entry in read_dir(Path::new(env!("CARGO_MANIFEST_DIR")).join("glenside-programs"))
        // Assert that we found something at the path...
        .unwrap()
        // ...and filter to just the directories.
        .filter(|f| f.as_ref().unwrap().path().is_dir())
    {
        // Go through the files and separate them out
        let mut source_file = None;
        let mut input_files = Vec::default();
        let mut output_file = None;
        for entry in read_dir(entry.as_ref().unwrap().path())
            .unwrap()
            .map(|e| e.unwrap())
        {
            if entry.path().file_name().unwrap() == "output.npy" {
                output_file = Some(entry.path().to_owned());
            } else if entry.path().extension().map_or(false, |s| s == "npy") {
                input_files.push(entry.path().to_owned());
            } else if entry.path().extension().map_or(false, |s| s == "glenside") {
                source_file = Some(entry.path().to_owned());
            }
        }

        let mut env = HashMap::default();
        for path in input_files.iter() {
            let key = path.file_stem().unwrap().to_str().unwrap();
            env.insert(key, read_npy::<_, ArrayD<f64>>(&path).unwrap());
        }

        let expr =
            RecExpr::<Language>::from_str(read_to_string(source_file.unwrap()).unwrap().as_str())
                .unwrap();
        match interpret(&expr, expr.as_ref().len() - 1, &env) {
            glenside::language::interpreter::Value::Access(a) => {
                assert!(a.tensor.abs_diff_eq(
                    &read_npy::<_, ArrayD<f64>>(output_file.unwrap())
                        .unwrap()
                        .into_dyn(),
                    1e-7
                ));
            }
            _ => panic!(),
        }
    }
}
