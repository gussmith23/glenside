use egg::{*, rewrite as rw};

define_language! {
    enum HardwareSoftwareLanguage {
        Num(i32),
        Add = "+",
        Mul = "*",
        For = "for",
        Symbol(String),
    }

}

fn main() {
    let hwswrules: &[Rewrite<HardwareSoftwareLanguage, ()>] = &[
        //rw!("break-for"; "(for ?index_var ?lower_bound ?upper_bound"),
        rw!("commute-add"; "(+ ?a ?b)" => "(+ ?b ?a)"),
        rw!("commute-mul"; "(* ?a ?b)" => "(* ?b ?a)"),

        rw!("add-0"; "(+ ?a 0)" => "?a"),
        rw!("mul-0"; "(* ?a 0)" => "0"),
        rw!("mul-1"; "(* ?a 1)" => "?a"),
    ];


    let start = "(for i 0 4 (for j 0 4 (+ (index a i) (index b j))))".parse().unwrap();

    let (egraph, report) = SimpleRunner::default().run_expr(start, &hwswrules);
    println!(
        "Stopped after {} iterations, reason: {:?}",
        report.iterations.len(),
        report.stop_reason
    );
    egraph.dot().to_svg("target/foo.svg").unwrap();
}
