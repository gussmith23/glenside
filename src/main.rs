use egg::define_language;
use either::*;
use ndarray::s;
use num_traits::identities::Zero;
use std::collections::HashMap;

fn main() {
    //dot_product()
    mlp()
}

fn mlp() {
    egg::define_language! {
        enum MlpLanguage {
            // TODO(gus) do we need this to be an intrinsic? Is this cheating?
            // Without this I'm not sure how to represent a matmul w/o using a lambda.
            // TODO(gus) a more updated todo here would be: do we want to
            // consider (sooner rather than later) whether we can break down
            // dot product into its component parts (that is, whether it would
            // be valuable to, and then, whether it would be possible)
            Dotprod = "dotprod",
            Rows = "rows",
            Cols = "cols",
            CartesianProduct = "cartesian-product",
            // Map over a shaped list
            ShapedMap = "shaped-map",
            BsgSystolicArray = "bsg_systolic_array_weight_stationary",
            // Slice into list/tensor/whatever we're calling them
            Slice = "slice",
            ShapedAdd = "shaped-add",
            // TODO(gus) this will probably need to be signed at some point?
            Usize(usize),
            Symbol(String),
        }
    }

    type Environment<'a> = HashMap<&'a str, Value>;

    // The type system of our program. We support tensors (which support values,
    // as 0-dim tensors) and lists. We could imagine adding other datatypes in
    // the future (e.g. trees).
    // TODO(gus) how to represent user-defined ADTs?
    // StreamValue are the actual value types that can appear in streams.
    #[derive(Clone, PartialEq, Debug)]
    enum ListValue {
        Scalar(DataType),
        // TODO(gus) Doing this may open up a huge can of worms. I'm not quite
        // sure how to design the language though, and this is the easiest way
        // to get what I want: I want to be able to have Streams of Values.
        //Value(Value),
        //CartesianProductElement(ndarray::ArrayD<DataType>, ndarray::ArrayD<DataType>),
        Value(Value),
    }
    #[derive(Clone, PartialEq, Debug)]
    enum Value {
        List(std::vec::Vec<ListValue>),
        // Box needed for indirection, otherwise we have recursive structure.
        Tuple2(Box<ListValue>, Box<ListValue>),
        ShapedList(ndarray::ArrayD<ListValue>),
        Function(fn(ListValue) -> ListValue),
        // TODO(gus) is this a problem? If we have this, we might as well remove
        // ListValue entirely, right?
        //Scalar(DataType),
        // Ok, maybe that later, but we actually need integers:
        Usize(usize),
    }

    impl std::ops::Add for ListValue {
        type Output = ListValue;

        fn add(self: ListValue, other: ListValue) -> ListValue {
            match self {
                ListValue::Scalar(s) => match other {
                    ListValue::Scalar(s2) => ListValue::Scalar(s + s2),
                    _ => panic!(),
                },
                ListValue::Value(v) => match other {
                    ListValue::Value(v2) => ListValue::Value(v + v2),
                    _ => panic!(),
                },
            }
        }
    }

    // impl num_traits::identities::Zero for ListValue {
    //     fn zero() {
    //         // TODO(gus) This seems bad
    //         ListValue::Scalar(DataType::zero())
    // }

    impl std::ops::Add for Value {
        type Output = Value;

        fn add(self: Value, other: Value) -> Value {
            match self {
                Value::List(v) => match other {
                    Value::List(v2) => Value::List({
                        assert_eq!(v.len(), v2.len());
                        v.iter()
                            .zip(v2.iter())
                            .map(|(a, b)| a.clone() + b.clone())
                            .collect()
                    }),
                    _ => panic!(),
                },
                Value::ShapedList(t) => match other {
                    Value::ShapedList(t2) => Value::ShapedList(t + t2),
                    _ => panic!(),
                },
                Value::Tuple2(a, b) => match other {
                    Value::Tuple2(a2, b2) => Value::Tuple2(Box::new(*a + *a2), Box::new(*b + *b2)),
                    _ => panic!(),
                },
                Value::Usize(u) => match other {
                    Value::Usize(u2) => Value::Usize(u + u2),
                    _ => panic!(),
                },
                Value::Function(_) => panic!(),
            }
        }
    }
    fn interpret_eclass<M: egg::Metadata<MlpLanguage>>(
        egraph: &egg::EGraph<MlpLanguage, M>,
        eclass: &egg::EClass<MlpLanguage, M>,
        env: &Environment,
    ) -> Value {
        let results: std::vec::Vec<Value> = eclass
            .nodes
            .iter()
            .map(|enode: &egg::ENode<MlpLanguage>| interpret_enode(egraph, enode, env))
            .collect();
        assert!(results.len() > 0);
        let result: Value = results
            .iter()
            // TODO(gus) not good for performance.
            .fold(
                results[0].clone(),
                |v1: Value, v2: &Value| if v1 == *v2 { v1 } else { panic!() },
            );
        result
    }

    // I'm just doing this for now; it may not actually be what we want in the
    // end.
    //type Result = Meta;
    // Woah, this is giving a crazy error that is pointing to the
    // define_language macro usage. Not Donna deal with that right now.
    // TODO I'm wondering if the metadata can just act as an interpreter? It
    // kind of serves that purpose already.

    fn interpret_enode<M: egg::Metadata<MlpLanguage>>(
        egraph: &egg::EGraph<MlpLanguage, M>,
        enode: &egg::ENode<MlpLanguage>,
        env: &Environment,
    ) -> Value {
        use MlpLanguage::*;
        match &enode.op {
            Symbol(name) => env[&name[..]].clone(),
            Dotprod => {
                // Evaluating Dotprod produces different results based on
                // whether it gets arguments. Actually, this is true of all
                // functions. If it doesn't get any arguments, then it should
                // evaluate to a callable function.
                match enode.children.len() {
                    0 => {
                        // Expects a StreamValue::Pair as input.
                        fn dotprod(pair: ListValue) -> ListValue {
                            // Unpack the tensors to dot-product.
                            let (left, right): (
                                ndarray::ArrayD<ListValue>,
                                ndarray::ArrayD<ListValue>,
                            ) = match pair {
                                ListValue::Value(v) => match v {
                                    Value::Tuple2(left, right) => match (*left, *right) {
                                        (ListValue::Value(left), ListValue::Value(right)) => {
                                            match (left, right) {
                                                (
                                                    Value::ShapedList(left),
                                                    Value::ShapedList(right),
                                                ) => (left, right),
                                                _ => panic!(),
                                            }
                                        }
                                        _ => panic!(),
                                    },
                                    _ => panic!(),
                                },
                                _ => panic!(),
                            };

                            assert_eq!(left.ndim(), 1);
                            assert_eq!(right.ndim(), 1);

                            use std::iter::FromIterator;
                            fn unpack_and_flatten(
                                t: ndarray::ArrayD<ListValue>,
                            ) -> ndarray::Array1<DataType> {
                                ndarray::Array::from_iter(
                                    t.iter()
                                        .map(|el| match el {
                                            ListValue::Scalar(s) => s,
                                            _ => panic!(),
                                        })
                                        .cloned(),
                                )
                            };
                            let left = unpack_and_flatten(left);
                            let right = unpack_and_flatten(right);

                            ListValue::Scalar(left.dot(&right))
                        }
                        Value::Function(dotprod)
                    }
                    _ => panic!(),
                }
            }
            Rows => {
                // There should be one arg: a single tensor.
                assert_eq!(enode.children.len(), 1);
                // Expect that the result of interpreting it is a tensor.
                let arg_as_tensor: ndarray::ArrayD<ListValue> =
                    match interpret_eclass(egraph, &egraph[enode.children[0]], env) {
                        Value::ShapedList(t) => t,
                        _ => panic!(),
                    };

                Value::List(
                    arg_as_tensor
                        .axis_iter(ndarray::Axis(0))
                        .map(|view| ListValue::Value(Value::ShapedList(view.to_owned())))
                        .collect::<std::vec::Vec<ListValue>>(),
                )
            }
            Cols => {
                // There should be one arg: a single tensor.
                assert_eq!(enode.children.len(), 1);
                // Expect that the result of interpreting it is a tensor.
                let arg_as_tensor: ndarray::ArrayD<ListValue> =
                    match interpret_eclass(egraph, &egraph[enode.children[0]], env) {
                        Value::ShapedList(t) => t,
                        _ => panic!(),
                    };

                Value::List(
                    arg_as_tensor
                        .axis_iter(ndarray::Axis(1))
                        .map(|view| ListValue::Value(Value::ShapedList(view.to_owned())))
                        .collect::<std::vec::Vec<ListValue>>(),
                )
            }
            CartesianProduct => {
                // Semantics of cartesian product:
                // Rightmost thing varies the fastest.

                // There should be two args, both of which should be lists.
                assert_eq!(enode.children.len(), 2);
                let left: std::vec::Vec<ListValue> =
                    match interpret_eclass(egraph, &egraph[enode.children[0]], env) {
                        Value::List(l) => l,
                        _ => panic!(),
                    };
                let right: std::vec::Vec<ListValue> =
                    match interpret_eclass(egraph, &egraph[enode.children[1]], env) {
                        Value::List(l) => l,
                        _ => panic!(),
                    };

                // TODO(gus) figuring out exactly what should be the output of
                // CartesianProduct. It's a headache. I see three options:
                // 1. A 1-d List of pairs (Lists) of Tensors.
                // 2. A 2-d List of pairs (Lists) of Tensors.
                // 3. A Tensor whose elements are pairs (Lists) of Tensors.
                // I have a lot of notes on my thinking. Going to go with some
                // variant of 3.
                // Scratch that. That would require adding another tensor type
                // to our list of types. Don't want to do that for some reason.
                // Going with option 1 now.
                // For what it's worth, I'm back on doing 3.

                let new_shape = vec![left.len(), right.len()];

                use itertools::iproduct;
                let product_vector: std::vec::Vec<ListValue> = iproduct!(left, right)
                    .map(|tuple| {
                        ListValue::Value(Value::Tuple2(
                            Box::<_>::new(tuple.0),
                            Box::<_>::new(tuple.1),
                        ))
                    })
                    .collect();

                let reshaped_into_tensor: ndarray::ArrayD<ListValue> =
                    ndarray::ArrayD::<ListValue>::from_shape_vec(new_shape, product_vector)
                        .unwrap();

                Value::ShapedList(reshaped_into_tensor)
            }
            ShapedMap => {
                assert_eq!(enode.children.len(), 2);

                // The first child of map is a function, passed as an argument.
                // For example, a dot-product node with no children. In the
                // future we may support other things that will evaluate to
                // functions.
                // The node should evaluate to a function value in our
                // interpreter.
                let function: fn(ListValue) -> ListValue =
                    match interpret_eclass(egraph, &egraph[enode.children[0]], env) {
                        Value::Function(f) => f,
                        _ => panic!(),
                    };

                let input_shaped_list: ndarray::ArrayD<ListValue> =
                    match interpret_eclass(egraph, &egraph[enode.children[1]], env) {
                        Value::ShapedList(t) => t,
                        _ => panic!(),
                    };

                Value::ShapedList(
                    ndarray::ArrayD::<ListValue>::from_shape_vec(
                        input_shaped_list.shape(),
                        input_shaped_list.iter().cloned().map(function).collect(),
                    )
                    .unwrap(),
                )
            }
            Slice => {
                // TODO(gus) this is true for our minimal working example, not
                // expected to be true in the future, definitely not.
                assert_eq!(enode.children.len(), 5);

                let tensor: ndarray::ArrayD<_> =
                    match interpret_eclass(egraph, &egraph[enode.children[0]], env) {
                        Value::ShapedList(t) => t,
                        _ => panic!(),
                    };

                assert_eq!(tensor.shape().len(), 2);

                let ((row_slice_start, row_slice_end), (col_slice_start, col_slice_end)): (
                    (usize, usize),
                    (usize, usize),
                ) = match enode.children[1..5]
                    .iter()
                    .map(|eclass_id: &u32| {
                        match interpret_eclass(egraph, &egraph[*eclass_id], env) {
                            Value::Usize(u) => u,
                            _ => panic!(),
                        }
                    })
                    .collect::<std::vec::Vec<usize>>()
                    .as_slice()
                {
                    [row_slice_start, row_slice_end, col_slice_start, col_slice_end] => (
                        (*row_slice_start, *row_slice_end),
                        (*col_slice_start, *col_slice_end),
                    ),
                    _ => panic!(),
                };

                // A  consistent problem I was having with this library
                // was with converting to dynamic-dimension tensors.
                // Found out I can use into_dyn().
                Value::ShapedList(
                    tensor
                        .slice_move(s![
                            row_slice_start..row_slice_end,
                            col_slice_start..col_slice_end
                        ])
                        .into_dyn(),
                )
            }
            Usize(u) => Value::Usize(*u),
            ShapedAdd => {
                let tensors: std::vec::Vec<ndarray::ArrayD<_>> = enode
                    .children
                    .iter()
                    .map(
                        |eclass| match interpret_eclass(egraph, &egraph[*eclass], env) {
                            Value::ShapedList(t) => t,
                            _ => panic!(),
                        },
                    )
                    .collect();

                assert!(tensors.len() > 0);

                Value::ShapedList(tensors.iter().fold(
                    ndarray::Array::from_elem(
                        tensors[0].shape(),
                        ListValue::Scalar(DataType::zero()),
                    ),
                    |acc, t| acc + t,
                ))
            }
            BsgSystolicArray => panic!(),
        }
    }

    // TODO(gus) is this possible? what is this called?
    // FWIW, this isn't possible as-is, as rust complains of a cycle.
    //type Shape = Vec<Either<i64, Shape>>;
    // Only one level deep, will add more as needed. Need to figure out how to
    // achieve this in general.
    type Shape = Vec<Either<i64, Vec<Either<i64, Vec<i64>>>>>;

    /// Given a function in MLPLanguage (i.e. dotprod) and an input shape,
    /// what's the output shape?
    fn infer_shape(
        node: &MlpLanguage,
        inner_shape: &Vec<Either<i64, Vec<i64>>>,
    ) -> Either<i64, Vec<Either<i64, Vec<i64>>>> {
        use MlpLanguage::*;
        match node {
            Dotprod => {
                // TODO(gus) a BIG assumption that's making our lives easier
                // right now and probably hurting us in the long run is that
                // dotprod doesn't take two arguments, but it takes one argument
                // which is a pair of vectors in a "tuple".
                //println!("dotprod input shape: {:?}", inner_shape);
                // Check for tuple of size 2
                assert_eq!(inner_shape.len(), 2);
                assert_eq!(*inner_shape[0].as_ref().left().unwrap(), 2);
                let right: &Vec<_> = inner_shape[1].as_ref().right().unwrap();
                assert_eq!(right.len(), 1);
                Left(1)
            }
            _ => {
                //println!("Unrecognized node type: {:?}", node);
                panic!()
            }
        }
    }

    type DataType = f64;

    #[derive(Debug, Clone)]
    struct Meta {
        // TODO(gus) implement tensors with many dimensions by using a vector
        // here. I just didn't want to mess around with figuring out Vecs in
        // Rust, and refcounting, etc etc.
        shape: Option<Shape>,
        scalar_value: Option<i64>,
        // If this eclass represents an op in the language, what op?
        // I'm adding this because I am using ops in the language as
        // first-class citizens (e.g. I can pass Dotprod to ShapedMap).
        // When it comes time to inspect a ShapedMap that takes an op, for example, I
        // need to be able to quickly figure out what the op represented by the
        // eclass is.
        // I specifically need this in the following context: I have an
        // infer_shape function that takes an op and an input shape and tells me
        // what the output shape will be. I need to be able to get the op.
        // TODO(gus) I think this is a hack honestly---what I should do is have
        // a metadata slot for "infer shape" functions. When metadata merges, we
        // can combine the list of these functions. When we need to infer a
        // shape, we can run all of the functions and make sure their outputs
        // all match.
        //op: Option<MlpLanguage>,
        // Actually, for right now, I'm going to take the even easier route and
        // assume that any eclass who is expected to be a usage of an op as a
        // first-class type will only have a single enode, which should be an op.
        // Tensor value. Should replace scalar_value.
        value: Option<ndarray::ArrayD<DataType>>,
    }
    impl PartialEq for Meta {
        fn eq(&self, other: &Self) -> bool {
            return self.shape == other.shape && self.scalar_value == other.scalar_value;
        }
    }
    impl egg::Metadata<MlpLanguage> for Meta {
        type Error = ();

        fn merge(&self, other: &Self) -> Self {
            assert_eq!(self, other);
            self.clone()
        }

        fn make(egraph: &egg::EGraph<MlpLanguage, Self>, enode: &egg::ENode<MlpLanguage>) -> Self {
            // We only know the value in the case of a Num.
            use MlpLanguage::*;
            match &enode.op {
                CartesianProduct => {
                    assert_eq!(enode.children.len(), 2);
                    let initial_shape_left: &Shape =
                        egraph[enode.children[0]].metadata.shape.as_ref().unwrap();
                    assert_eq!(initial_shape_left.len(), 2);
                    let initial_shape_right: &Shape =
                        egraph[enode.children[1]].metadata.shape.as_ref().unwrap();
                    assert_eq!(initial_shape_left.len(), 2);
                    // println!(
                    //     "CartesianProduct: {:?}, {:?}",
                    //     initial_shape_left, initial_shape_right
                    // );
                    // TODO(gus) i think right now we need the innermost dims to
                    // match. While this makes sense for a matmul implemented
                    // with a cartesian product, it doesn't make sense for
                    // cartesian product in general.
                    assert_eq!(
                        initial_shape_left[1].as_ref().right().unwrap(),
                        initial_shape_right[1].as_ref().right().unwrap(),
                    );
                    let innermost_shape: Vec<i64> = initial_shape_left[1]
                        .as_ref()
                        .right()
                        .unwrap()
                        .iter()
                        .map(|i| i.clone().left().unwrap())
                        .collect();
                    let new_inner_shape: Vec<Either<i64, Vec<i64>>> =
                        vec![Left(2), Right(innermost_shape)];
                    let new_shape: Shape = vec![
                        Left(*initial_shape_left[0].as_ref().left().unwrap()),
                        Left(*initial_shape_right[0].as_ref().left().unwrap()),
                        Right(new_inner_shape),
                    ];
                    Meta {
                        shape: Some(new_shape),
                        scalar_value: None,
                        value: None,
                    }
                }
                Rows => {
                    assert_eq!(enode.children.len(), 1);
                    let initial_shape: &Shape =
                        egraph[enode.children[0]].metadata.shape.as_ref().unwrap();
                    //println!("{:?}", initial_shape);
                    assert_eq!(initial_shape.len(), 2);
                    // TODO(gus) should probably check that the shape is of a
                    // shape we expect
                    // Transform initial shape: (row (tensor...)) becomes a list
                    // which has shape (list num_rows (list other_dim...))
                    // TODO(gus) we're also doing "rows" and "columns" right
                    // now, which is very specific to 2 dimensions.
                    // We can generalize this later
                    let all_but_first: Vec<Either<i64, Vec<i64>>> = initial_shape[1..]
                        .as_ref()
                        .into_iter()
                        .map(|i| Left(i.clone().left().unwrap()))
                        .collect();
                    let new_shape: Shape = vec![
                        Left(initial_shape[0].clone().left().unwrap()),
                        Right(all_but_first),
                    ];
                    Meta {
                        shape: Some(new_shape),
                        scalar_value: None,
                        value: None,
                    }
                }
                Cols => {
                    assert_eq!(enode.children.len(), 1);
                    let initial_shape: &Shape =
                        egraph[enode.children[0]].metadata.shape.as_ref().unwrap();
                    //println!("Cols: {:?}", initial_shape);
                    assert_eq!(initial_shape.len(), 2);
                    let all_but_second: Vec<Either<i64, Vec<i64>>> = initial_shape[0..1]
                        .as_ref()
                        .into_iter()
                        .map(|i| Left(i.clone().left().unwrap()))
                        .collect();
                    let new_shape: Shape = vec![
                        Left(initial_shape[1].clone().left().unwrap()),
                        Right(all_but_second),
                    ];
                    Meta {
                        shape: Some(new_shape),
                        scalar_value: None,
                        value: None,
                    }
                }
                ShapedMap => {
                    assert_eq!(enode.children.len(), 2);
                    let shape: &Shape = egraph[enode.children[1]].metadata.shape.as_ref().unwrap();

                    // Get the MlpLanguage op that is being mapped onto the
                    // input.
                    let class: &egg::EClass<MlpLanguage, Meta> = &egraph[enode.children[0]];
                    // Assume that the class only has one child, which is an
                    // MlpLanguage op with no children. The fact that it has no
                    // children means it's not a call.
                    assert_eq!(class.len(), 1);
                    let node: &egg::ENode<MlpLanguage> = &class.nodes[0];
                    assert_eq!(node.children.len(), 0);
                    let op: &MlpLanguage = &node.op;

                    let mut new_shape: Shape = shape.clone();
                    // TODO(gus)
                    // we assume the last thing in the top level shape describes
                    // the list's elements' shape. (All things in the list have
                    // the same shape; that's a limitation of our shape type
                    // system.)
                    // TODO(gus) if infer_shape returns Left(1), what do we do?
                    // do we remove the dimension entirely?
                    // that's what i'm going to do now, because it is solving a
                    // headache for me---but it may cause more down the road.
                    // I think the type system is rotten. Worth a try but lists
                    // and tensors should probably just be different.
                    let inferred_shape =
                        infer_shape(op, shape.last().unwrap().as_ref().right().unwrap());
                    if inferred_shape == Left(1) {
                        new_shape.remove(new_shape.len() - 1);
                    } else {
                        new_shape[shape.len() - 1] = inferred_shape;
                    }
                    Meta {
                        shape: Some(new_shape),
                        scalar_value: None,
                        value: None,
                    }
                }
                Dotprod => {
                    assert_eq!(enode.children.len(), 0);
                    //println!("Dotprod");
                    Meta {
                        shape: None,
                        scalar_value: None,
                        value: None,
                    }
                }
                BsgSystolicArray => {
                    assert_eq!(enode.children.len(), 2);
                    let left_shape = &egraph[enode.children[0]].metadata.shape.as_ref().unwrap();
                    let right_shape = &egraph[enode.children[1]].metadata.shape.as_ref().unwrap();
                    // TODO(gus) very rough approx of what's actually right.
                    assert_eq!(left_shape[1], right_shape[0]);
                    assert_eq!(left_shape[2..], right_shape[2..]);
                    assert!(left_shape[2..]
                        .iter()
                        .all(|i| *i.as_ref().left().unwrap() == 1));
                    let new_shape: Shape =
                        [&left_shape[0..1], &right_shape[1..2], &left_shape[2..]].concat();
                    Meta {
                        shape: Some(new_shape),
                        scalar_value: None,
                        value: None,
                    }
                }
                Slice => panic!(),
                Usize(_) => panic!(),
                ShapedAdd => panic!(),
                Symbol(name) => {
                    //println!("Symbol");
                    Meta {
                        shape: Some(match &name[..] {
                            "in" => vec![Left(1), Left(784)],
                            "w1" => vec![Left(784), Left(512)],
                            "w2" => vec![Left(512), Left(512)],
                            "w3" => vec![Left(512), Left(10)],
                            _ => panic!("No shape defined for {}", name),
                        }),
                        scalar_value: None,
                        value: None,
                    }
                }
            }
        }
    }

    let slice_test_program_1 = "(slice a 0 1 0 1)".parse().unwrap();
    let slice_test_program_2 = "(slice a 1 2 0 2)".parse().unwrap();
    let input = pack_interpreter_input(
        ndarray::ArrayD::<DataType>::from_shape_vec(vec![2, 2], vec![1., 2., 3., 4.]).unwrap(),
    );
    let mut env = Environment::new();
    env.insert("a", input);
    let (egraph, id) = egg::EGraph::<MlpLanguage, ()>::from_expr(&slice_test_program_1);
    let out = unpack_interpreter_output(interpret_eclass(&egraph, &egraph[id], &env));
    assert_eq!(
        out,
        ndarray::ArrayD::<DataType>::from_shape_vec(vec![1, 1], vec![1.]).unwrap()
    );
    let (egraph, id) = egg::EGraph::<MlpLanguage, ()>::from_expr(&slice_test_program_2);
    let out = unpack_interpreter_output(interpret_eclass(&egraph, &egraph[id], &env));
    assert_eq!(
        out,
        ndarray::ArrayD::<DataType>::from_shape_vec(vec![1, 2], vec![3., 4.]).unwrap()
    );

    let shaped_add_test_program_1 = "(shaped-add a b)".parse().unwrap();
    let a = pack_interpreter_input(
        ndarray::ArrayD::<DataType>::from_shape_vec(vec![2, 2], vec![1., 2., 3., 4.]).unwrap(),
    );
    let b = pack_interpreter_input(
        ndarray::ArrayD::<DataType>::from_shape_vec(vec![2, 2], vec![2., 3., -4., -5.]).unwrap(),
    );
    let mut env = Environment::new();
    env.insert("a", a);
    env.insert("b", b);
    let (egraph, id) = egg::EGraph::<MlpLanguage, ()>::from_expr(&shaped_add_test_program_1);
    let out = unpack_interpreter_output(interpret_eclass(&egraph, &egraph[id], &env));
    assert_eq!(
        out,
        ndarray::ArrayD::<DataType>::from_shape_vec(vec![2, 2], vec![3., 5., -1., -1.]).unwrap()
    );

    let program = "
     (shaped-map dotprod
      (cartesian-product
       (rows
        (shaped-map dotprod
         (cartesian-product
          (rows
           (shaped-map dotprod (cartesian-product (rows in)
                                           (cols w1))))
          (cols w2)
         )
        )
       )
       (cols w3)
      )
     )
     "
    .parse()
    .unwrap();

    fn load_npy(path: &str) -> ndarray::ArrayD<DataType> {
        ndarray_npy::read_npy::<_, ndarray::ArrayD<DataType>>(path).unwrap()
    }
    fn pack_interpreter_input(array: ndarray::ArrayD<DataType>) -> Value {
        Value::ShapedList(
            ndarray::ArrayD::<ListValue>::from_shape_vec(
                array.shape(),
                array
                    .iter()
                    .cloned()
                    .map(|scalar| ListValue::Scalar(scalar))
                    .collect(),
            )
            .unwrap(),
        )
    }
    let in_val = pack_interpreter_input(load_npy("in.npy"));
    let w1_val = pack_interpreter_input(load_npy("w1.npy"));
    let w2_val = pack_interpreter_input(load_npy("w2.npy"));
    let w3_val = pack_interpreter_input(load_npy("w3.npy"));
    let out_true = load_npy("out.npy");
    let mut env = Environment::new();
    env.insert("in", in_val);
    env.insert("w1", w1_val);
    env.insert("w2", w2_val);
    env.insert("w3", w3_val);
    let (egraph, id) = egg::EGraph::<MlpLanguage, Meta>::from_expr(&program);
    egraph.dot().to_svg("mlp-before-rewrites.svg").unwrap();
    let out = interpret_eclass(&egraph, &egraph[id], &env);

    fn unpack_interpreter_output(output: Value) -> ndarray::ArrayD<DataType> {
        match output {
            Value::ShapedList(t) => ndarray::ArrayD::<DataType>::from_shape_vec(
                t.shape(),
                t.iter()
                    .cloned()
                    .map(|list_val| match list_val {
                        ListValue::Scalar(s) => s,
                        _ => panic!(),
                    })
                    .collect(),
            )
            .unwrap(),
            _ => panic!(),
        }
    }
    let out = unpack_interpreter_output(out);

    use approx::AbsDiffEq;
    assert!(out_true.abs_diff_eq(&out, 1e-8));

    // TODO(gus) metadata? using area?
    // TODO(gus) here's a problem: if we represent area as metadata, how do
    // we capture the different areas of different designs that might share
    // an e-class?

    let rewrite = egg::rewrite!("tensorize-dot-product";
    "(shaped-map dotprod
      (cartesian-product
       (rows ?t1)
       (cols ?t2)
      )
     )"=>"(bsg_systolic_array_weight_stationary ?t1 ?t2)");

    let rules: &[egg::Rewrite<MlpLanguage, Meta>] = &[rewrite];

    let runner = egg::Runner::new().with_expr(&program).run(&rules);

    println!(
        "Stopped after {} iterations, reason: {:?}",
        runner.iterations.len(),
        runner.stop_reason
    );
    runner
        .egraph
        .dot()
        .to_svg("mlp-after-rewrites.svg")
        .unwrap();
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

            //println!("{}", bound_val);

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
