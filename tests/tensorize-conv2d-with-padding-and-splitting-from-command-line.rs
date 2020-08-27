use log::debug;
use std::process::Command;

#[test]
fn conv2d_im2col_tensorize_to_smaller_array_with_padding_and_slicing_from_command_line() {
    test_logger::ensure_env_logger_initialized();

    // The location of the Glenside program we'd like to compile.
    let program_filepath = format!("{}/data/conv2d/conv2d.glenside", env!("CARGO_MANIFEST_DIR"));

    // This file holds shape information about the Glenside program.
    // TODO(@gussmith23) This can be merged with the program itself.
    let shapes_filepath = format!(
        "{}/data/conv2d/conv2d-shapes.json",
        env!("CARGO_MANIFEST_DIR")
    );

    // The final generated executable
    let out_filepath = format!(
        "{}/conv2d-{}",
        std::env::temp_dir().to_string_lossy(),
        std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );

    // Where we'll put the compiled C code.
    let mut out_code_filepath = std::env::temp_dir();
    out_code_filepath.push(format!("conv2d.c",));

    // Where we'll put the JSON hardware design.
    let mut out_design_filepath = std::env::temp_dir();
    out_design_filepath.push(format!(
        "conv2d-hardware-design-{}.json",
        std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));

    // Run Glenside!
    let output = Command::new("target/debug/glenside")
        // Subcommand of the glenside binary.
        .arg("demo")
        // Argument 1: the name of the function to compile.
        // So our output will include a C function called mlp(...)
        .arg("conv2d")
        .arg(program_filepath)
        .arg(shapes_filepath)
        .arg(&out_code_filepath)
        .arg(&out_design_filepath)
        .arg("--iter-limit")
        .arg("40")
        .arg("--time-limit")
        .arg("40")
        .arg("--node-limit")
        .arg("500000")
        .arg("--find-monolithic-designs")
        .arg("(64,64)")
        .output()
        .expect("Failed to run glenside");

    // Check that it ran.
    assert!(
        output.status.success(),
        "Glenside binary failed with code {:?}. stderr:\n{}",
        output.status.code(),
        std::str::from_utf8(output.stderr.as_slice()).expect("Could not convert stderr to UTF8")
    );

    debug!(
        "Output C file filepath: {}",
        out_code_filepath.to_string_lossy()
    );
    debug!(
        "Output hardware json file filepath: {}",
        out_design_filepath.to_string_lossy()
    );

    // Now, we'll test that the C code it generated actually works.

    // The location of the test harness which tests the generated code.
    let main_c_filepath = format!(
        "{}/data/conv2d/conv2d-test-harness.c",
        env!("CARGO_MANIFEST_DIR")
    );

    // The location of the software implementation of a systolic array.
    let systolic_array_impl_filepath = format!(
        "{}/data/codegen-mlp/{}",
        env!("CARGO_MANIFEST_DIR"),
        "rtml_systolic_array_weight_stationary.c"
    );

    let output = Command::new("gcc")
        .arg("-g")
        .arg("-Werror")
        .arg(main_c_filepath)
        .arg(systolic_array_impl_filepath)
        .arg("-o")
        .arg(&out_filepath)
        // Include the temp dir, so that we can find the c code outputted by
        // glenside.
        .arg("-I")
        .arg(std::env::temp_dir())
        .output()
        .expect("Failed to compile main file with gcc");
    assert!(
        output.status.success(),
        "Compilation failed. stderr:\n{}",
        std::str::from_utf8(output.stderr.as_slice()).expect("Could not convert stderr to UTF8")
    );

    debug!("Output binary filepath: {}", out_filepath);

    let output = Command::new(out_filepath)
        .current_dir(std::env::temp_dir())
        .output()
        .expect("Failed to run result");
    assert!(
        output.status.success(),
        "Main binary failed with code {:?}. stderr:\n{}",
        output.status.code(),
        std::str::from_utf8(output.stderr.as_slice()).expect("Could not convert stderr to UTF8")
    );
}
