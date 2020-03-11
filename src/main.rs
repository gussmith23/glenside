use egg::define_language;

fn main() {
    //dot_product()
    mlp()
}

fn mlp() {
    egg::define_language! {
        enum MlpLanguage {
            Num(i64),
            // TODO(gus) do we need this to be an intrinsic? Is this cheating?
            // Without this I'm not sure how to represent a matmul w/o using a lambda.
            Dotprod = "dotprod",
            Relu = "relu",
            Rows = "rows",
            Cols = "cols",
            Zipwith = "zipwith",
            // List constructor
            List = "list",
            // Tensor constructor:
            // Takes an identifier and a list which represents its shape
            // e.g. (tensor a (list 1 4))
            Tensor = "tensor",
            Symbol(String),
        }
    }

    let program = "
     (relu
      (zipwith dotprod
       (rows
        (relu
         (zipwith dotprod
          (rows
           (relu (zipwith dotprod (rows (tensor in (list 1 784)))
                                  (cols (tensor w1 (list 784 512))))))
          (cols (tensor w2 (list 512 512)))
         )
        )
       )
       (cols (tensor w3 (list 512 1)))
      )
     )
     "
    .parse()
    .unwrap();

    // TODO(gus) metadata? using area?
    // TODO(gus) here's a problem: if we represent area as metadata, how do
    // we capture the different areas of different designs that might share
    // an e-class?

    let rules: &[egg::Rewrite<MlpLanguage, ()>] = &[];

    let runner = egg::Runner::new().with_expr(&program).run(&rules);

    println!(
        "Stopped after {} iterations, reason: {:?}",
        runner.iterations.len(),
        runner.stop_reason
    );
    runner.egraph.dot().to_svg("mlp.svg").unwrap();
}

fn _dot_product() {
    egg::define_language! {
        enum DotProductLanguage {
            Num(i32),
            Mul = "*",
            For = "for",
            Sum = "sum",
            Symbol(String),
        }
    }

    // Metadata is simply the i32 value of the class, if known.
    type Meta = Option<i32>;
    impl egg::Metadata<DotProductLanguage> for Meta {
        type Error = ();

        fn merge(&self, other: &Self) -> Self {
            assert_eq!(
                self.expect("During metadata merge, expected value in self"),
                other.expect("During metadata merge, expected value in other")
            );
            *self
        }

        fn make(
            _egraph: &egg::EGraph<DotProductLanguage, Self>,
            enode: &egg::ENode<DotProductLanguage>,
        ) -> Self {
            // We only know the value in the case of a Num.
            match enode.op {
                DotProductLanguage::Num(i) => Some(i),
                _ => None,
            }
        }
    }

    #[derive(Debug)]
    struct Splitter {
        bound_var: egg::Var,
        bound: egg::Var,
        body: egg::Var,
    }

    impl egg::Applier<DotProductLanguage, Meta> for Splitter {
        fn apply_one(
            &self,
            egraph: &mut egg::EGraph<DotProductLanguage, Meta>,
            _eclass: egg::Id,
            subst: &egg::Subst,
        ) -> Vec<egg::Id> {
            // Figure out what the actual value of bound is.
            // Expect it to be known, for now.
            let bound_id: egg::Id = subst[&self.bound];
            let bound_val = &egraph[bound_id]
                .metadata
                .expect("Bound's exact value should be known, for now");

            println!("{}", bound_val);

            let _bound_var_id: egg::Id = subst[&self.bound_var];

            // Decide how to split the loop up: by all of its factors
            // https://gist.github.com/qolop/71ef78c394db822756d58cac9993db77
            let factors = (2..*bound_val)
                .into_iter()
                .filter(|&x| bound_val % x == 0)
                .collect::<Vec<i32>>();
            for factor in factors {
                let outer_bound = factor;
                let _outer_for_loop_bound_id = egraph.add(egg::ENode::new(
                    DotProductLanguage::Num(outer_bound),
                    vec![],
                ));

                let inner_bound = bound_val / factor;
                let _inner_for_loop_bound_id = egraph.add(egg::ENode::new(
                    DotProductLanguage::Num(inner_bound),
                    vec![],
                ));

                // I want to rewrite the original body so that, anywhere where
                // the original loop var is used, the new loop var will be used.

                // TODO stopped here.
                // rewrite! {
                //     "replace-loop-var";
                //     // Searcher should just search for the initial loop var
                // }

                // What is the new loop index variable?
                // let inner_for_loop_id =
                //     egraph.add(ENode::new(DotProductLanguage::For), vec![])
            }

            vec![]
            /*
            // Get strings for sexpr
            let index: QuestionMarkName = "?index".parse().unwrap();
            let bound = "?bound".parse().unwrap();
            let body = "?body".parse().unwrap();
            // Look up node IDs
            let index = subst[&self.bound_var];
            let bound = map[&bound][0];
            let bodyid = map[&body][0];
            // Get the int from the node for bound
            let boundval = egraph[bound].nodes[0].op.clone();
            let boundval = match boundval {
                Lang::Num(val) => val,
                _ => panic!(),
            };
            // Get the index variable string
            let indexstr = egraph[index].nodes[0].op.clone();
            let indexstr = match indexstr {
                Lang::String(name) => name,
                _ => panic!(),
            };

            // I don't know if we need to get anything other than an id for the  body.

            // Get list of factors for this bound and the new index (index') to use
            let factors = get_factors(boundval);
            // let newindex = egraph.add(Expr::unit(Lang::String(indexstr + "'"))).id;

            let mut res = Vec::new();
            for fact in factors {
                if fact != 1 && fact != boundval {
                    let secondbound = boundval / fact;
                    let firstboundid = egraph.add(Expr::unit(Lang::Num(fact))).id;
                    let secondboundid = egraph.add(Expr::unit(Lang::Num(secondbound))).id;
                    let childnode =
                    // TODO does it need the ID of a child that may have already existed?
                    egraph.add(Expr::new(Lang::For, smallvec![index, secondboundid, bodyid]));
                    let newnode = egraph.add(Expr::new(
                        Lang::For,
                        smallvec![index, firstboundid, childnode.id],
                    ));
                    res.push(newnode);
                }
            }
            res
            */
        }
    }

    let dot_product_rules: &[egg::Rewrite<DotProductLanguage, Meta>] = &[
        egg::rewrite!("loop-split"; "(for ?bound_var ?bound ?body)" => {
            Splitter {
                bound_var : "?bound_var".parse().unwrap(),
                bound : "?bound".parse().unwrap(),
                body : "?body".parse().unwrap(),
            }
        }),
        egg::rewrite!("hw-dot-prod-8"; "(for ?i 8 (* (index ?a ?i) (index ?b ?i)))" => "(hw-dot-prod-8 ?a ?b)"),
        egg::rewrite!("hw-dot-prod-8"; "(for ?i 8 (* (index ?a ?i) (index ?b ?i)))" => "(hw-dot-prod-8 ?a ?b)"),
    ];

    let program = "(sum (for i 10 (* (index a i) (index b i))))"
        .parse()
        .unwrap();

    let runner = egg::Runner::new()
        .with_expr(&program)
        .run(&dot_product_rules);

    println!(
        "Stopped after {} iterations, reason: {:?}",
        runner.iterations.len(),
        runner.stop_reason
    );
    runner.egraph.dot().to_svg("dot-product.svg").unwrap();
}

// TODO implement with "residual" computations
// TODO implement also with zero padding
// TODO use fold instead of for
