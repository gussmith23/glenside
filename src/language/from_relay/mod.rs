#[cfg(test)]
mod tests {
    use crate::language::{Language, MyAnalysis};
    use egg::{EGraph, Pattern, RecExpr, Searcher};
    use std::io::Write;
    use std::process::Command;
    use std::str::FromStr;

    macro_rules! from_relay_test {
        ($test_name:ident, $relay_str:expr, $glenside_str:expr) => {
            #[test]
            fn $test_name() {
                let script_filepath = format!(
                    "{}/src/language/from_relay/from_relay.py",
                    env!("CARGO_MANIFEST_DIR")
                );
                // https://www.reddit.com/r/rust/comments/38jhva/piping_string_to_child_process_stdin/crvlqcd/?utm_source=reddit&utm_medium=web2x&context=3
                let mut proc = Command::new("python3")
                    .arg(script_filepath)
                    .stdin(std::process::Stdio::piped())
                    .stdout(std::process::Stdio::piped())
                    .spawn()
                    .ok()
                    .expect("Failed to spawn process");
                proc.stdin
                    .as_mut()
                    .unwrap()
                    .write_all($relay_str.as_bytes())
                    .unwrap();
                let output = proc.wait_with_output().unwrap();
                let glenside_str = String::from_utf8(output.stdout).unwrap();
                let expr = RecExpr::from_str(glenside_str.as_str())
                    .expect("Could not parse glenside expression");
                let mut egraph = EGraph::new(MyAnalysis::default());
                let id = egraph.add_expr(&expr);

                let pattern = $glenside_str.parse::<Pattern<Language>>().unwrap();
                assert!(pattern.search_eclass(&egraph, id).is_some());
            }
        };
    }
    from_relay_test!(
        basic_test,
        r#"
#[version = "0.0.5"]
def @main(%x: Tensor[(3), float32], %y: Tensor[(3), float32]) -> Tensor[(3), float32] {
  add(%x, %y) /* ty=Tensor[(3), float32] */
}

"#,
        r#"
(some-glenside-expr)
"#
    );
}
