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

impl std::default::Default for ListValue {
    fn default() -> Self {
        ListValue::Scalar(DataType::default())
    }
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

#[derive(Clone, Debug)]
enum MemoizedInterpreterResult {
    InterpretedValue(Value),
    StillInterpreting,
    // When an eclass cannot be interpreted, because all of its enodes
    // recursively depend on themselves.
    CanNotInterpret,
}

type MemoizationMap = std::collections::HashMap<egg::Id, MemoizedInterpreterResult>;

fn interpret_eclass<M: egg::Metadata<MlpLanguage>>(
    egraph: &egg::EGraph<MlpLanguage, M>,
    eclass: &egg::EClass<MlpLanguage, M>,
    env: &Environment,
    // TODO(gus) shoot, i think i'm mixing up where to put the "mut". Should
    // probably not be on both.
    mut memo_map: &mut MemoizationMap,
) -> MemoizedInterpreterResult {
    info!("Interpreting eclass {}", eclass.id);
    debug!("{:?}", eclass);

    if memo_map.contains_key(&eclass.id) {
        // TODO(gus) is this right?
        return memo_map[&eclass.id].clone();
    }

    // Insert a flag saying that we are now interpreting this eclass. As we
    // recurse, if we hit this eclass again, we'll know we have a recursive
    // cycle, which we handle later in this function.
    match memo_map.insert(eclass.id, MemoizedInterpreterResult::StillInterpreting) {
        Some(_) => panic!(),
        None => (),
    }

    assert!(eclass.nodes.len() > 0);
    let mut results: std::vec::Vec<MemoizedInterpreterResult> = eclass
        .nodes
        .iter()
        .map(|enode: &egg::ENode<MlpLanguage>| interpret_enode(egraph, enode, env, &mut memo_map))
        .collect();
    let pre_filtered_length = results.len();

    // At this point, we'll have a MemoizedInterpreterResult for every enode in
    // the eclass. These all represent potential values for the entire eclass.
    // That is, the whole point of the egraph is that eclasses contain
    // expressions with equivalent values. So, if our interpreter is built
    // correctly, each MemoizedInterpreterResult will be one of two things:
    // 1. an InterpretedValue holding the eclass's correct value, or
    // 2. a StillInterpreting, if the enode recursively depends on the eclass.

    // Now, we check two things:
    // 1. There's at least one InterpretedValue---i.e. at least one of the
    //    enodes resolved to a concrete value and doesn't recursively depend on
    //    its own eclass. Without this, we have no way to even guess the value
    //    of this eclass.
    // 2. All of the InterpretedValues have the same value. This should be true
    //    if the interpreter is built correctly and if the rewrites are correct.

    let filtered_results: std::vec::Vec<Value> = results
        .drain(..)
        .filter(|result| match result {
            MemoizedInterpreterResult::InterpretedValue(_) => true,
            _ => false,
        })
        .map(|result| match result {
            MemoizedInterpreterResult::InterpretedValue(v) => v,
            _ => unreachable!(),
        })
        .collect();
    debug!("{:?}", pre_filtered_length);
    debug!("{:?}", filtered_results.len());
    debug!("{:?}", eclass);
    //assert!(filtered_results.len() > 0, "This eclass's enodes all depend recursively on this eclass! Cannot interpret to a concrete value!");
    if filtered_results.len() == 0 {
        debug!(
            "eclass {} evaluates to CanNotInterpret. Hoping for the best!",
            eclass.id
        );
        match memo_map.insert(eclass.id, MemoizedInterpreterResult::CanNotInterpret) {
            Some(MemoizedInterpreterResult::StillInterpreting) => (),
            _ => panic!(),
        }
        return MemoizedInterpreterResult::CanNotInterpret;
    }
    assert!(
        filtered_results.iter().all(|v| *v == filtered_results[0]),
        "This class's enodes don't all evaluate to the same value!"
    );

    if filtered_results.len() < pre_filtered_length {
        // Some StillInterpreting values were cut out. Might want to log
        // something here later on.
    }

    // Now, if there were any StillInterpreting results, we will "guess" their
    // value to be the concrete value that the rest of the enodes interpreted
    // to. We do this by officially setting the eclass's value in the map.
    assert!(match &memo_map.get(&eclass.id).unwrap() {
        &MemoizedInterpreterResult::StillInterpreting => true,
        _ => panic!(),
    });
    // TODO(gus) cloning, inefficient and lazy
    match memo_map.insert(
        eclass.id,
        MemoizedInterpreterResult::InterpretedValue(filtered_results[0].clone()),
    ) {
        Some(MemoizedInterpreterResult::StillInterpreting) => (),
        _ => panic!(),
    }

    // After we guess, check that things all come out to the same value.
    // We may want to disable this for running-time reasons.
    let check_after_guess = true;
    if check_after_guess {
        // We're essentially just doing the same thing that we did above!
        let results: std::vec::Vec<MemoizedInterpreterResult> = eclass
            .nodes
            .iter()
            .map(|enode: &egg::ENode<MlpLanguage>| {
                interpret_enode(egraph, enode, env, &mut memo_map)
            })
            .collect();

        assert!(results.iter().all(|result| match result {
            MemoizedInterpreterResult::InterpretedValue(v) =>
                v == match memo_map.get(&eclass.id).unwrap() {
                    // TODO(gus) probably inefficient
                    MemoizedInterpreterResult::InterpretedValue(v) => v,
                    _ => panic!(),
                },
            MemoizedInterpreterResult::CanNotInterpret => {
                // TODO(gus) remove this once I figure out if this is expected.
                debug!("CanNotInterpret found while checking");
                true
            }
            MemoizedInterpreterResult::StillInterpreting =>
                panic!("After guessing, we still get a StillInterpreting result!"),
        }));
    }

    // TODO(gus) probably inefficient
    memo_map.get(&eclass.id).unwrap().clone()
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
    memo_map: &mut MemoizationMap,
) -> MemoizedInterpreterResult {
    debug!("interpreting enode: {:?}", enode);
    use MlpLanguage::*;
    match &enode.op {
        Symbol(name) => {
            debug!("interpreting symbol");
            MemoizedInterpreterResult::InterpretedValue(env[&name[..]].clone())
        }
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
                                            (Value::ShapedList(left), Value::ShapedList(right)) => {
                                                (left, right)
                                            }
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
                    MemoizedInterpreterResult::InterpretedValue(Value::Function(dotprod))
                }
                _ => panic!(),
            }
        }
        Rows => {
            // There should be one arg: a single tensor.
            assert_eq!(enode.children.len(), 1);

            // Expect that the result of interpreting it is a tensor.
            // TODO(gus) clean up this syntax.
            let arg_as_tensor = interpret_eclass(egraph, &egraph[enode.children[0]], env, memo_map);
            let arg_as_tensor = match arg_as_tensor {
                // Return early if we've found a recursive cycle
                MemoizedInterpreterResult::StillInterpreting => {
                    return MemoizedInterpreterResult::StillInterpreting
                }
                MemoizedInterpreterResult::CanNotInterpret => {
                    return MemoizedInterpreterResult::CanNotInterpret
                }
                MemoizedInterpreterResult::InterpretedValue(v) => v,
            };
            let arg_as_tensor: ndarray::ArrayD<ListValue> = match arg_as_tensor {
                Value::ShapedList(t) => t,
                _ => panic!(),
            };

            MemoizedInterpreterResult::InterpretedValue(Value::List(
                arg_as_tensor
                    .axis_iter(ndarray::Axis(0))
                    .map(|view| ListValue::Value(Value::ShapedList(view.to_owned())))
                    .collect::<std::vec::Vec<ListValue>>(),
            ))
        }
        Cols => {
            // There should be one arg: a single tensor.
            assert_eq!(enode.children.len(), 1);
            // Expect that the result of interpreting it is a tensor.
            let arg_as_tensor: ndarray::ArrayD<ListValue> =
                match interpret_eclass(egraph, &egraph[enode.children[0]], env, memo_map) {
                    MemoizedInterpreterResult::StillInterpreting => {
                        return MemoizedInterpreterResult::StillInterpreting
                    }
                    MemoizedInterpreterResult::CanNotInterpret => {
                        return MemoizedInterpreterResult::CanNotInterpret
                    }
                    MemoizedInterpreterResult::InterpretedValue(v) => match v {
                        Value::ShapedList(t) => t,
                        _ => panic!(),
                    },
                };

            MemoizedInterpreterResult::InterpretedValue(Value::List(
                arg_as_tensor
                    .axis_iter(ndarray::Axis(1))
                    .map(|view| ListValue::Value(Value::ShapedList(view.to_owned())))
                    .collect::<std::vec::Vec<ListValue>>(),
            ))
        }
        CartesianProduct => {
            // Semantics of cartesian product:
            // Rightmost thing varies the fastest.

            // There should be two args, both of which should be lists.
            assert_eq!(enode.children.len(), 2);
            let left: std::vec::Vec<ListValue> =
                match interpret_eclass(egraph, &egraph[enode.children[0]], env, memo_map) {
                    MemoizedInterpreterResult::StillInterpreting => {
                        return MemoizedInterpreterResult::StillInterpreting
                    }
                    MemoizedInterpreterResult::CanNotInterpret => {
                        return MemoizedInterpreterResult::CanNotInterpret
                    }
                    MemoizedInterpreterResult::InterpretedValue(v) => match v {
                        Value::List(l) => l,
                        _ => panic!(),
                    },
                };
            let right: std::vec::Vec<ListValue> =
                match interpret_eclass(egraph, &egraph[enode.children[1]], env, memo_map) {
                    MemoizedInterpreterResult::StillInterpreting => {
                        return MemoizedInterpreterResult::StillInterpreting
                    }
                    MemoizedInterpreterResult::CanNotInterpret => {
                        return MemoizedInterpreterResult::CanNotInterpret
                    }
                    MemoizedInterpreterResult::InterpretedValue(v) => match v {
                        Value::List(l) => l,
                        _ => panic!(),
                    },
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
                ndarray::ArrayD::<ListValue>::from_shape_vec(new_shape, product_vector).unwrap();

            MemoizedInterpreterResult::InterpretedValue(Value::ShapedList(reshaped_into_tensor))
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
                match interpret_eclass(egraph, &egraph[enode.children[0]], env, memo_map) {
                    MemoizedInterpreterResult::StillInterpreting => {
                        return MemoizedInterpreterResult::StillInterpreting
                    }
                    MemoizedInterpreterResult::CanNotInterpret => {
                        return MemoizedInterpreterResult::CanNotInterpret
                    }
                    MemoizedInterpreterResult::InterpretedValue(v) => match v {
                        Value::Function(f) => f,
                        _ => panic!(),
                    },
                };

            let input_shaped_list: ndarray::ArrayD<ListValue> =
                match interpret_eclass(egraph, &egraph[enode.children[1]], env, memo_map) {
                    MemoizedInterpreterResult::StillInterpreting => {
                        return MemoizedInterpreterResult::StillInterpreting
                    }
                    MemoizedInterpreterResult::CanNotInterpret => {
                        return MemoizedInterpreterResult::CanNotInterpret
                    }
                    MemoizedInterpreterResult::InterpretedValue(v) => match v {
                        Value::ShapedList(t) => t,
                        _ => panic!(),
                    },
                };

            MemoizedInterpreterResult::InterpretedValue(Value::ShapedList(
                ndarray::ArrayD::<ListValue>::from_shape_vec(
                    input_shaped_list.shape(),
                    input_shaped_list.iter().cloned().map(function).collect(),
                )
                .unwrap(),
            ))
        }
        Slice => {
            // TODO(gus) this is true for our minimal working example, not
            // expected to be true in the future, definitely not.
            assert_eq!(enode.children.len(), 5);

            let tensor: ndarray::ArrayD<_> =
                match interpret_eclass(egraph, &egraph[enode.children[0]], env, memo_map) {
                    MemoizedInterpreterResult::StillInterpreting => {
                        return MemoizedInterpreterResult::StillInterpreting
                    }
                    MemoizedInterpreterResult::CanNotInterpret => {
                        return MemoizedInterpreterResult::CanNotInterpret
                    }
                    MemoizedInterpreterResult::InterpretedValue(v) => match v {
                        Value::ShapedList(t) => t,
                        _ => panic!(),
                    },
                };

            assert_eq!(tensor.shape().len(), 2);

            let ((row_slice_start, row_slice_end), (col_slice_start, col_slice_end)): (
                (usize, usize),
                (usize, usize),
            ) = match enode.children[1..5]
                .iter()
                .map(|eclass_id: &u32| {
                    match interpret_eclass(egraph, &egraph[*eclass_id], env, memo_map) {
                        // TODO(gus) this panic is me just being lazy. if
                        // StillInterpreting is found, we should return
                        MemoizedInterpreterResult::InterpretedValue(v) => match v {
                            Value::Usize(u) => u,
                            _ => panic!(),
                        },
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
            MemoizedInterpreterResult::InterpretedValue(Value::ShapedList(
                tensor
                    .slice_move(s![
                        row_slice_start..row_slice_end,
                        col_slice_start..col_slice_end
                    ])
                    .into_dyn(),
            ))
        }
        Usize(u) => MemoizedInterpreterResult::InterpretedValue(Value::Usize(*u)),
        ShapedAdd => {
            let tensors: std::vec::Vec<ndarray::ArrayD<_>> = enode
                .children
                .iter()
                .map(
                    |eclass| match interpret_eclass(egraph, &egraph[*eclass], env, memo_map) {
                        MemoizedInterpreterResult::InterpretedValue(v) => match v {
                            Value::ShapedList(t) => t,
                            _ => panic!(),
                        },
                        // TODO(gus) this panic is me just being lazy.
                        _ => panic!(),
                    },
                )
                .collect();

            assert!(tensors.len() > 0);

            MemoizedInterpreterResult::InterpretedValue(Value::ShapedList(tensors.iter().fold(
                ndarray::Array::from_elem(tensors[0].shape(), ListValue::Scalar(DataType::zero())),
                |acc, t| acc + t,
            )))
        }
        BsgSystolicArray => panic!(),
        Concat => {
            // TODO(gus) this is true for our minimal working example, not
            // expected to be true in the future, definitely not.
            assert!(enode.children.len() >= 3);

            println!("Interpreting: {:?}", enode);
            // TODO(gus) it seems like there's a loop.
            // concat a has concat b as a child. concat b has concat a as a child.
            let mut tensors: std::vec::Vec<MemoizedInterpreterResult> = enode.children
                [0..enode.children.len() - 1]
                .iter()
                .map(|child| interpret_eclass(egraph, &egraph[*child], env, memo_map))
                .collect();
            // TODO(gus) the order of this and the next if block matters!
            if tensors.iter().any(|t| match t {
                MemoizedInterpreterResult::CanNotInterpret => true,
                _ => false,
            }) {
                return MemoizedInterpreterResult::CanNotInterpret;
            }
            if tensors.iter().any(|t| match t {
                MemoizedInterpreterResult::StillInterpreting => true,
                _ => false,
            }) {
                return MemoizedInterpreterResult::StillInterpreting;
            }
            let tensors: std::vec::Vec<ndarray::ArrayD<_>> = tensors
                .drain(..)
                .map(|result| match result {
                    MemoizedInterpreterResult::InterpretedValue(v) => match v {
                        Value::ShapedList(t) => t,
                        _ => panic!(),
                    },
                    _ => panic!(),
                })
                .collect();

            let concat_axis: usize = match interpret_eclass(
                egraph,
                &egraph[enode.children[enode.children.len() - 1]],
                env,
                memo_map,
            ) {
                MemoizedInterpreterResult::StillInterpreting => {
                    return MemoizedInterpreterResult::StillInterpreting
                }
                MemoizedInterpreterResult::CanNotInterpret => {
                    return MemoizedInterpreterResult::CanNotInterpret
                }
                MemoizedInterpreterResult::InterpretedValue(v) => match v {
                    Value::Usize(u) => u,
                    _ => panic!(),
                },
            };

            // Have to stack manually, because ndarray::stack doesn't actually
            // support tensors holding values that don't implement Copy! :(

            let shapes: std::vec::Vec<&[usize]> = tensors.iter().map(|t| t.shape()).collect();
            // All have equal number of dimensions
            assert!(shapes.iter().all(|shape| shape.len() == shapes[0].len()));
            // All dimensions match, except for along the concat axis.
            assert!((0..shapes[0].len())
                .all(|i| i == concat_axis
                    || ((0..shapes.len()).all(|j| shapes[j][i] == shapes[0][i]))));

            let mut new_shape: std::vec::Vec<usize> = shapes[0].to_vec();
            new_shape[concat_axis] += (1..shapes.len())
                .map(|i| shapes[i][concat_axis])
                .sum::<usize>();
            let new_shape = new_shape;

            let mut new_tensor: ndarray::ArrayD<_> = ndarray::Array::default(new_shape.to_vec());
            let mut current_start_index: usize = 0;
            for i in 0..tensors.len() {
                let original_concat_axis_size: usize = shapes[i][concat_axis];
                use std::convert::TryFrom;
                let slices: std::vec::Vec<ndarray::SliceOrIndex> = (0..new_shape.len())
                    .map(|axis| {
                        if axis == concat_axis {
                            ndarray::SliceOrIndex::Slice {
                                start: isize::try_from(current_start_index).unwrap(),
                                end: Some(
                                    isize::try_from(
                                        current_start_index + original_concat_axis_size,
                                    )
                                    .unwrap(),
                                ),
                                step: 1,
                            }
                        } else {
                            ndarray::SliceOrIndex::Slice {
                                start: 0,
                                end: None,
                                step: 1,
                            }
                        }
                    })
                    .collect();
                let slice_info: ndarray::SliceInfo<
                    std::vec::Vec<ndarray::SliceOrIndex>,
                    ndarray::IxDyn,
                > = ndarray::SliceInfo::new(slices).unwrap();

                new_tensor
                    .slice_mut(slice_info.as_ref())
                    .assign(&tensors[i]);

                current_start_index += original_concat_axis_size;
            }

            assert_eq!(current_start_index, new_shape[concat_axis]);

            MemoizedInterpreterResult::InterpretedValue(Value::ShapedList(new_tensor))
        }
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
