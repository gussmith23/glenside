use clap::{App, Arg, SubCommand};
use egg::{EGraph, RecExpr, Runner};
use glenside::language::rewrites::{PadLocation, PadSliceStrategy, SliceConcatenateStrategy};
use glenside::language::{Language, MyAnalysis, PadType};
use serde_json::Value;
use std::collections::HashMap;
use std::io::Write;
use std::str::FromStr;

fn main() {
    let matches = App::new("glenside")
        .subcommand(
            SubCommand::with_name("demo")
                .arg(Arg::with_name("NAME").required(true).index(1))
                .arg(Arg::with_name("PROGRAM").required(true).index(2))
                .arg(Arg::with_name("SHAPES").required(true).index(3))
                .arg(Arg::with_name("OUT_CODE_FILEPATH").required(true).index(4))
                .arg(
                    Arg::with_name("OUT_DESIGN_FILEPATH")
                        .required(true)
                        .index(5),
                )
                .arg(
                    Arg::with_name("allocate-for-manycore")
                        .help("Declares all buffers using the attributes required by the Manycore")
                        .long("allocate-for-manycore"),
                )
                .arg(
                    Arg::with_name("blocking")
                        .help(
                            "Determines how Glenside blocks up matrix \
                             multiplies to fit on systolic arrays. A value of \
                             'glenside' will have glenside insert systolic \
                             arrays for every valid matrix multiplication size \
                             it finds. A value of '(<rows>,<cols>)' \
                             (no whitespace) will have Glenside insert \
                             systolic arrays of size rows,cols which use BSG's \
                             automatic blocking, where possible. This flag can \
                             be passed multiple values. A \
                             value of just 'glenside' indicates that all \
                             blocking should be done explicitly within \
                             Glenside's search process.",
                        )
                        .long("blocking")
                        .min_values(1)
                        .default_value("glenside"),
                )
                .arg(
                    Arg::with_name("prefer-bsg-blocking")
                        .help(
                            "When extracting a design, favors systolic arrays \
                             that use BSG's automatic blocking.",
                        )
                        .long("prefer-bsg-blocking"),
                )
                .arg(
                    Arg::with_name("node-limit")
                        .long("node-limit")
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("time-limit")
                        .help("Time limit in seconds")
                        .long("time-limit")
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("iter-limit")
                        .long("iter-limit")
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("find-monolithic-designs")
                        .long("find-monolithic-designs")
                        .help(
                            "Takes an argument (rows,cols); Glenside will \
                               find monolithic designs of this size. Do not \
                               include any whitespace before, between, or \
                               after the parentheses.",
                        )
                        .takes_value(true),
                ),
        )
        .get_matches();

    if let Some(matches) = matches.subcommand_matches("demo") {
        // Read in shapes as JSON dict; convert to HashMap<String, Vec<usize>>
        let shapes_json: Value = serde_json::from_str(
            std::fs::read_to_string(matches.value_of("SHAPES").unwrap())
                .unwrap()
                .as_str(),
        )
        .unwrap();
        let mut shapes_map = HashMap::new();
        for (name, value) in shapes_json.as_object().unwrap().iter() {
            shapes_map.insert(
                name.clone(),
                value
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| value.as_u64().unwrap() as usize)
                    .collect::<Vec<_>>(),
            );
        }

        // Read in program into egraph
        let glenside_expr = RecExpr::<Language>::from_str(
            std::fs::read_to_string(matches.value_of("PROGRAM").unwrap())
                .unwrap()
                .as_str(),
        )
        .unwrap();

        let mut egraph = EGraph::new(MyAnalysis {
            name_to_shape: shapes_map.clone(),
        });
        let id = egraph.add_expr(&glenside_expr);

        let mut runner = Runner::default().with_egraph(egraph);

        if let Some(m) = matches.value_of("node-limit") {
            runner = runner.with_node_limit(m.parse().expect("node-limit should be an integer"));
        }
        if let Some(m) = matches.value_of("iter-limit") {
            runner = runner.with_iter_limit(m.parse().expect("iter-limit should be an integer"));
        }
        if let Some(m) = matches.value_of("time-limit") {
            runner = runner.with_time_limit(std::time::Duration::from_secs(
                m.parse().expect("time-limit should be an integer"),
            ));
        }

        // TODO(@gussmith23) Add flags to control different rewrites
        let mut rws = vec![
            glenside::language::rewrites::flatten_unflatten_any_access(),
            glenside::language::rewrites::bubble_reshape_through_cartesian_product(),
            glenside::language::rewrites::bubble_reshape_through_compute_dot_product(),
            glenside::language::rewrites::bubble_access_concatenate_through_access_cartesian_product_not_item_axis_left(),
            glenside::language::rewrites::bubble_access_concatenate_through_access_cartesian_product_not_item_axis_right(),
            glenside::language::rewrites::bubble_access_concatenate_through_access_cartesian_product_same_item_axis(),
            glenside::language::rewrites::bubble_access_concatenate_through_compute_dot_product_item_axis(),
            glenside::language::rewrites::bubble_access_concatenate_through_compute_dot_product_not_item_axis(),
            glenside::language::rewrites::bubble_access_slice_through_access_pad_inequal_axes(),
            glenside::language::rewrites::pad_slice_accesses(
                0,
                PadSliceStrategy::PadToClosestMultipleOf {
                    multiple_of: 64,
                    pad_location: PadLocation::End,
                    pad_type: PadType::ZeroPadding,
                },
            ),
            glenside::language::rewrites::pad_slice_accesses(
                1,
                PadSliceStrategy::PadToClosestMultipleOf {
                    multiple_of: 64,
                    pad_location: PadLocation::End,
                    pad_type: PadType::ZeroPadding,
                },
            ),
            glenside::language::rewrites::slice_concatenate_accesses(
                0,
                SliceConcatenateStrategy::DivideInto { segment_size: 64 },
            ),
            glenside::language::rewrites::slice_concatenate_accesses(
                1,
                SliceConcatenateStrategy::DivideInto { segment_size: 64 },
            ),
            glenside::language::rewrites::bubble_access_slice_through_access_cartesian_product_not_item_axis_left(),
            glenside::language::rewrites::bubble_access_slice_through_access_cartesian_product_not_item_axis_right(),
            glenside::language::rewrites::bubble_access_slice_through_access_cartesian_product_same_item_axis(),
            glenside::language::rewrites::bubble_access_slice_through_compute_dot_product_not_item_axis(),
            glenside::language::rewrites::bubble_access_slice_through_compute_dot_product_item_axis_not_tuple_axis(),
            glenside::language::rewrites::collapse_nested_accesses(),
            glenside::language::rewrites::collapse_nested_transposes(),
            glenside::language::rewrites::remove_trivial_transpose(),
        ];

        // Ensure that each rewrite is just added once.
        let mut added = std::collections::HashSet::new();
        for value in matches.values_of("blocking").unwrap() {
            if added.contains(value) {
                continue;
            }
            if value == "glenside" {
                rws.push(glenside::language::rewrites::systolic_array());
            } else {
                let parsed = value
                    .chars()
                    .skip(1)
                    .take(value.len() - 2)
                    .collect::<String>()
                    .split(",")
                    .map(|s| s.parse::<usize>().unwrap())
                    .collect::<Vec<_>>();
                assert_eq!(parsed.len(), 2);
                rws.push(glenside::language::rewrites::systolic_array_with_blocking(
                    parsed[0], parsed[1],
                ));
            }
            added.insert(value);
        }

        runner = runner.run(&rws);

        // TODO(@gussmith23) Explain difference between extraction and hw gen
        // Why do we "extract a monolithic design" and then "create a monolithic
        // design"? Why are these two separate steps? Well, extracting a
        // monolithic design doesn't actually mean creating a design; it means
        // extracting an expression which /could/ be monolithic. What comes out
        // at extraction time doesn't actually have hardware assigned yet; that
        // happens in design creation, a few lines later.
        let (_, extracted_expr) = if let Some(val) = matches.value_of("find-monolithic-designs") {
            let parsed = val
                .chars()
                .skip(1)
                .take(val.len() - 2)
                .collect::<String>()
                .split(",")
                .map(|s| s.parse::<usize>().unwrap())
                .collect::<Vec<_>>();
            assert_eq!(parsed.len(), 2);
            egg::Extractor::new(
                &runner.egraph,
                glenside::extraction::MonolithicCostFunction {
                    egraph: &runner.egraph,
                    systolic_array_configuration: (parsed[0], parsed[1]),
                    prefer_systolic_arrays_with_blocking: matches.is_present("prefer-bsg-blocking"),
                },
            )
            .find_best(id)
        } else {
            egg::Extractor::new(
                &runner.egraph,
                glenside::extraction::SimpleCostFunction {
                    prefer_systolic_arrays_with_blocking: matches.is_present("prefer-bsg-blocking"),
                },
            )
            .find_best(id)
        };

        let mut egraph = EGraph::new(MyAnalysis {
            name_to_shape: shapes_map,
        });
        let id = egraph.add_expr(&extracted_expr);
        let (hw_id_map, hw_atoms) = if let Some(val) = matches.value_of("find-monolithic-designs") {
            let parsed = val
                .chars()
                .skip(1)
                .take(val.len() - 2)
                .collect::<String>()
                .split(",")
                .map(|s| s.parse::<usize>().unwrap())
                .collect::<Vec<_>>();
            assert_eq!(parsed.len(), 2);
            glenside::codegen::create_hardware_design_monolithic(&egraph, (parsed[0], parsed[1]))
        } else {
            glenside::codegen::create_hardware_design_no_sharing(&egraph)
        };

        // Get expression arguments/inputs and sort alphabetically.
        let mut found_vars = glenside::codegen::find_vars(&egraph, id);
        found_vars.sort();

        let code = glenside::codegen::codegen(
            &egraph,
            id,
            &hw_id_map,
            matches.value_of("NAME").unwrap(),
            if !matches.is_present("allocate-for-manycore") {
                ""
            } else {
                r#"__attribute__ ((section (".uninitialized"))) __attribute__ ((aligned (256)))"#
            },
            &found_vars.iter().map(AsRef::as_ref).collect(),
        );

        let json = glenside::hw_design_language::design_to_json(
            &glenside::hw_design_language::HardwareDesign { atoms: hw_atoms },
        );

        std::fs::File::create(matches.value_of("OUT_CODE_FILEPATH").unwrap())
            .unwrap()
            .write_all(code.as_bytes())
            .unwrap();
        std::fs::File::create(matches.value_of("OUT_DESIGN_FILEPATH").unwrap())
            .unwrap()
            .write_all(serde_json::to_string_pretty(&json).unwrap().as_bytes())
            .unwrap();
    } else {
        todo!()
    }
}
