#![feature(test)]

extern crate test;
use approx::AbsDiffEq;
use ndarray_npy::{read_npy, write_npy};
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use test::Bencher;
/// Creates a Relay-to-Glenside benchmark test over a model
/// (currently the models we test are convolutional and only image-based)
/// The test does the following:
///  1. Reads the $relay_str_path file and parses as relay module
///  2. Converts the relay module to glenside
///  3. Generates random input (uniform [0, 255]) and parameters (uniform [-1, 1])
///  4. Runs relay module through TVM and benchmarks running glenside expr through interpreter
///  5. Compare output of using TVM vs interpreter
/// $test_name: the name of the created benchmark
/// $relay_str_path: the path of the file containing the Relay code
macro_rules! benchmark_model {
  (
    $(#[$meta:meta])*
    $test_name: ident, $relay_str_path:expr
  ) => {
        #[bench]
        $(#[$meta])*
        fn $test_name(b: &mut Bencher) {
            let filename = PathBuf::from($relay_str_path);
            let relay_str = std::fs::read_to_string(&filename).unwrap();

            // Random number generator for generating random tensors.
            const SEED: u64 = 23;
            let mut tensor_rng = SmallRng::seed_from_u64(SEED);

            let module = tvm::ir::module::IRModule::parse("", relay_str.clone());

            let (expr, shapes_vec) = glenside::language::from_relay::from_relay(&module);

            // Run interpreters and compare output.
            let script_filepath = format!(
                "{}/src/language/from_relay/run_relay.py",
                env!("CARGO_MANIFEST_DIR")
            );
            // https://www.reddit.com/r/rust/comments/38jhva/piping_string_to_child_process_stdin/crvlqcd/?utm_source=reddit&utm_medium=web2x&context=3
            // Output filename
            // TODO(@gussmith23) Do we want this RNG to use SEED?
            // I initially attempted to do this, but was running into issues
            // (I think the same filename kept being generated b/c I wasn't
            // using the RNG carefully...but maybe there's also something
            // wrong w/ how I'm reading files!)
            let output_filepath = std::env::temp_dir().with_file_name(format!(
                "output-{}.npy",
                rand::thread_rng()
                    .sample_iter(&rand::distributions::Alphanumeric)
                    .take(30)
                    .collect::<String>()
            ));

            let mut cmd = Command::new("python3");
            cmd.arg(script_filepath);
            cmd.arg(&output_filepath);
            cmd.stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped());
            let mut env = HashMap::default();
            for (index, (name, shape)) in shapes_vec.iter().enumerate() {
                // TODO(@gussmith23) output type assumption
                let value = if index == 0 {
                    // generate somewhat realistic image data
                    ndarray::ArrayD::<f32>::random_using(
                        shape.clone(),
                        Uniform::new(0f32, 255f32),
                        &mut tensor_rng,
                    )
                } else {
                    // generate parameters
                    ndarray::ArrayD::<f32>::random_using(
                        shape.clone(),
                        Uniform::new(-1f32, 1f32),
                        &mut tensor_rng,
                    )
                };
                env.insert(name.as_str(), value.clone());
                let filepath = std::env::temp_dir().with_file_name(format!(
                    "arg-{}.npy",
                    rand::thread_rng()
                        .sample_iter(&rand::distributions::Alphanumeric)
                        .take(30)
                        .collect::<String>()
                ));
                write_npy(&filepath, &value).unwrap();
                cmd.arg(filepath);
            }

            let mut proc = cmd.spawn().ok().expect("Failed to spawn process");
            proc.stdin
                .as_mut()
                .unwrap()
                .write_all(relay_str.as_bytes())
                .unwrap();
            let output = proc.wait_with_output().unwrap();
            // Check that it ran.
            assert!(
                output.status.success(),
                "Running Relay code failed with code {:?}.\nstdout:\n{}\nstderr:\n{}",
                output.status.code(),
                std::str::from_utf8(output.stdout.as_slice())
                    .expect("Could not convert stderr to UTF8"),
                std::str::from_utf8(output.stderr.as_slice())
                    .expect("Could not convert stderr to UTF8")
            );

            // TODO(@gussmith23) output type assumption
            let relay_output: ndarray::ArrayD<f32> = read_npy(output_filepath).unwrap();

            b.iter(|| {
                // use black box to prevent compiler optimizations
                let expr = test::black_box(&expr);
                let env = test::black_box(&env);

                let interpreter_output = match glenside::language::interpreter::interpret(
                    &expr,
                    expr.as_ref().len() - 1,
                    &env,
                ) {
                    glenside::language::interpreter::Value::Access(a) => a.tensor,
                    _ => panic!(),
                };

                // check to make sure no NaNs appear since NAN != NAN
                assert!(relay_output.iter().all(|v| !v.is_nan()));
                assert!(interpreter_output.iter().all(|v| !v.is_nan()));

                assert!(
                    relay_output.abs_diff_eq(&interpreter_output, 1e-5),
                    "{:?}\nvs.\n{:?}",
                    relay_output,
                    interpreter_output
                );
            });
        }
    };
}

// TODO: enable this test once memoization goes in
// note: mobilenet-shallow was generated:
// conv_block_1, separable_conv_block_1, separable_conv_block_2, and the layers at the end
// i.e all the conv_blocks from separable_conv_block_3 onward were removed
//
// tvm.relay.testing.mobilenet.mobile_net(num_classes = 10, data_shape=(1,3,32,32))
// followed by SimplifyInference
benchmark_model!(
    #[ignore = "this test causes stack overflow"]
    mobilenet_shallow,
    format!(
        "{}/models/mobilenet-simplified-for-inference-shallow.relay",
        env!("CARGO_MANIFEST_DIR")
    )
);

// TODO: enable this test once memoization goes in
// note: resnet-shallow was generated:
// tvm.relay.testing.resnet.get_workload(num_layers=2, num_classes=10, image_shape=(3,28,28))
// followed by SimplifyInference
benchmark_model!(
    #[ignore = "this test causes stack overflow"]
    resnet_shallow,
    format!(
        "{}/models/resnet-simplified-for-inference-shallow.relay",
        env!("CARGO_MANIFEST_DIR")
    )
);

// note: resnet was generated:
// tvm.relay.testing.resnet.get_workload()
// followed by SimplifyInference
benchmark_model!(
    #[ignore = "this test takes too long because the interpreter is slow"]
    resnet,
    format!(
        "{}/models/resnet-simplified-for-inference.relay",
        env!("CARGO_MANIFEST_DIR")
    )
);

// note: mobilenet was generated:
// tvm.relay.testing.mobilenet.get_workload()
// followed by SimplifyInference
benchmark_model!(
    #[ignore = "this test takes too long because the interpreter is slow"]
    mobilenet,
    format!(
        "{}/models/mobilenet-simplified-for-inference.relay",
        env!("CARGO_MANIFEST_DIR")
    )
);
