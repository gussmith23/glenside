#[test]
fn asplos_demo_generate_mlp_allocate_for_manycore() {
    /* Runs the following:

    target/debug/glenside asplos-demo  \
      --allocate-for-manycore          \
      mlp                              \
      data/asplos-demo/mlp.glenside    \
      data/asplos-demo/mlp-shapes.json \
      out.c                            \
      out.json

    See below comments for explanations of each argument. You can copy and paste
    the above command into a terminal in the root glenside directory (after
    running `cargo build`) to achieve the same results.

    */
    std::process::Command::new("target/debug/glenside")
        // Runs "asplos-demo" subcommand of Glenside
        .arg("asplos-demo")
        // Tells Glenside to allocate buffers for the Manycore
        .arg("--allocate-for-manycore")
        // The name of the resulting C function generated from the Glenside code
        .arg("mlp")
        // Input code files (Glenside code + tensor shape information)
        .arg("data/asplos-demo/mlp.glenside")
        .arg("data/asplos-demo/mlp-shapes.json")
        // Output files (C code + hardware design)
        .arg("out.c")
        .arg("out.json")
        .output()
        .unwrap();
}
