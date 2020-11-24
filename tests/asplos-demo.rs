#![cfg(ilp)]

use std::process::Command;

#[test]
fn asplos_demo_mlp() {
    // The location of the Glenside program we'd like to compile.
    let program_filepath = format!(
        "{}/data/asplos-demo/mlp.glenside",
        env!("CARGO_MANIFEST_DIR")
    );

    // This file holds shape information about the Glenside program.
    // TODO(@gussmith23) This can be merged with the program itself.
    let shapes_filepath = format!(
        "{}/data/asplos-demo/mlp-shapes.json",
        env!("CARGO_MANIFEST_DIR")
    );

    // The final generated executable
    let out_filepath = format!(
        "{}/asplos-demo-mlp-{}",
        std::env::temp_dir().to_string_lossy(),
        std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );

    // Where we'll put the compiled C code.
    let mut out_code_filepath = std::env::temp_dir();
    out_code_filepath.push("mlp.c");

    // Where we'll put the JSON hardware design.
    let mut out_design_filepath = std::env::temp_dir();
    out_design_filepath.push("hardware-design.json");

    // Run Glenside!
    let output = Command::new("target/debug/glenside")
        // Subcommand of the glenside binary.
        .arg("demo")
        // Argument 1: the name of the function to compile.
        // So our output will include a C function called mlp(...)
        .arg("mlp")
        .arg(program_filepath)
        .arg(shapes_filepath)
        .arg(&out_code_filepath)
        .arg(out_design_filepath)
        .output()
        .expect("Failed to run glenside");

    // Check that it ran.
    assert!(
        output.status.success(),
        "Glenside binary failed with code {:?}. stderr:\n{}",
        output.status.code(),
        std::str::from_utf8(output.stderr.as_slice()).expect("Could not convert stderr to UTF8")
    );

    // Now, we'll test that the C code it generated actually works.

    // The location of the test harness which tests the generated code.
    let main_c_filepath = format!(
        "{}/data/asplos-demo/mlp-test-harness.c",
        env!("CARGO_MANIFEST_DIR")
    );

    // The location of the software implementation of a systolic array.
    let systolic_array_impl_filepath = format!(
        "{}/data/codegen-mlp/{}",
        env!("CARGO_MANIFEST_DIR"),
        "rtml_systolic_array_weight_stationary.c"
    );

    let output = Command::new("gcc")
        .current_dir(std::env::temp_dir())
        .arg("-Werror")
        // The test harness #includes the generated mlp.c file, so we include
        // the tmp dir on the include path so that gcc will find mlp.c.
        .arg(format!("-I{}", std::env::temp_dir().to_string_lossy()))
        .arg(&main_c_filepath)
        .arg(&systolic_array_impl_filepath)
        .arg("-o")
        .arg(&out_filepath)
        .output()
        .expect("Failed to compile main file with gcc");

    // Check that it compiled.
    assert!(
        output.status.success(),
        "Compilation failed. stderr:\n{}",
        std::str::from_utf8(output.stderr.as_slice()).expect("Could not convert stderr to UTF8")
    );

    // Run the test harness.
    let output = Command::new(out_filepath)
        .output()
        .expect("Failed to run result");
    assert!(
        output.status.success(),
        "Main binary failed with code {:?}. stderr:\n{}",
        output.status.code(),
        std::str::from_utf8(output.stderr.as_slice()).expect("Could not convert stderr to UTF8")
    );

    // To see the resulting C file and JSON hardware design, go to $TMPDIR/mlp.c
    // and $TMPDIR/hardware-design.json.
}
